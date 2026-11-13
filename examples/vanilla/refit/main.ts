/**
 * vanilla/refit: a deforming mesh, in raw WebGPU. Its vertices move every
 * frame on the GPU, its tree follows them with a GPU refit, and the walk
 * traces it every frame. Nothing about the pose comes back to the CPU.
 *
 * - **The pose:** a compute pass of this page's own (twist, wave or stretch)
 *   writes the posed vertices into a storage buffer every frame, from the rest
 *   pose.
 * - **The refit:** `GpuMeshRefit` reads that buffer, rewrites the mesh's
 *   triangle records and refits its boxes bottom up, in the same compute pass,
 *   inside the scene's own buffers (`assembleTraceScene` + `uploadTraceScene`).
 *   Where the mesh sits there comes from `meshRefitPlacement`, which also
 *   names the scene's node format (quantized by default: the refit keeps an
 *   `f32` copy and packs it into the scene).
 * - **Drift:** a refit keeps the tree's shape, so its quality drifts with the
 *   pose. The refit's stats (`diagnostics: true`) give the refit tree's
 *   relative SAH cost; the page shows it over the built tree's, and over a
 *   tree built from the same pose now (an H-PLOC build every second, never
 *   placed). "rebuild now" builds H-PLOC from the posed buffer on the GPU and
 *   swaps the new tree in; "auto rebuild" does it when the drift stays past
 *   a limit, as `SceneTrees` does for its deforming meshes.
 * - **Swaps:** a rebuild assembles and uploads a new scene, makes a new refit
 *   for it, swaps both in between two frames, and only then destroys the old
 *   ones (a frame already submitted keeps them alive on the GPU).
 *
 * Needs `subgroups` (the refit and H-PLOC). The plan's second refit case,
 * thousands of moving instances with a top-level refit every frame, is
 * `vanilla/instances` (`?count=10000`).
 */

import {
    assembleTraceScene,
    createCpuTracer,
    createTinyBvhBuilder,
    GpuHplocBuilder,
    GpuMeshRefit,
    GpuMeshRefitKernels,
    meshRefitPlacement,
    relativeSahCost,
    uploadTraceScene,
    type GpuLayoutMeshBvh,
    type TraceSceneBuffers,
    type TraceSceneData,
    type TraceSceneInstance,
} from '@pmndrs/bvh';
import { traceSourceOf, type TraceSource } from '../../shared/gpu-rays.ts';
import { gpuMsText, GpuTimer } from '../../shared/gpu-timer.ts';
import { bytes } from '../../shared/overlay.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';
import { layoutInfo } from '../../shared/stats.ts';
import { createVanillaApp } from '../../shared/vanilla/app.ts';
import { checkViewAgainstCpu } from '../../shared/vanilla/check.ts';
import { ground, torusKnot } from '../../shared/vanilla/geometry.ts';
import { mat4, vec3 } from '../../shared/vanilla/math.ts';
import { RayCastView, VIEW_MODES, type ViewMode } from '../../shared/vanilla/raycast-view.ts';

const app = await createVanillaApp('Refit', 'vanilla/refit');
const { device, camera, overlay, canvas } = app;
const params = new URLSearchParams(location.search);

if (!GpuMeshRefit.isSupported(device) || !GpuHplocBuilder.isSupported(device)) {
    const message = 'This page needs the "subgroups" feature: GpuMeshRefit and GPU H-PLOC use it. Without it, SceneTrees.setPoseCpu refits on the CPU (not shown here).';
    overlay.error(message);
    throw new Error(message);
}

// ---- The mesh: a torus knot (`?segments=` along the tube), its rest pose and indices on the GPU.

const segments = Number(params.get('segments') ?? 600);
const mesh = torusKnot(1, 0.32, segments, 48);
const vertexCount = mesh.positions.length / 3;
const triangleCount = mesh.indices.length / 3;
const floor = ground(16, 16);
const KNOT_AT: [number, number, number] = [0, 2.3, 0];

const restBox = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
let radius = 0;
for (let i = 0; i < mesh.positions.length; i += 3) {
    for (let k = 0; k < 3; k++) {
        restBox[k] = Math.min(restBox[k], mesh.positions[i + k]);
        restBox[k + 3] = Math.max(restBox[k + 3], mesh.positions[i + k]);
    }
    radius = Math.max(radius, Math.hypot(mesh.positions[i], mesh.positions[i + 2]));
}
const centerY = (restBox[1] + restBox[4]) / 2;
const halfHeight = (restBox[4] - restBox[1]) / 2;
// The box every pose of the deform shader stays in (stretch: xz / sqrt(0.5), y * 1.8; wave: + 0.15 * halfHeight): the top level's box for the mesh.
const reach = radius * Math.SQRT2 + 0.15 * halfHeight + 1e-3;
const deformBox = new Float32Array([-reach, centerY - halfHeight * 1.8 - 1e-3, -reach, reach, centerY + halfHeight * 1.8 + 1e-3, reach]);

const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const rest = device.createBuffer({ label: 'refit rest pose', size: mesh.positions.byteLength, usage });
const posed = device.createBuffer({ label: 'refit posed vertices', size: mesh.positions.byteLength, usage });
const indexBuffer = device.createBuffer({ label: 'refit indices', size: mesh.indices.byteLength, usage });
device.queue.writeBuffer(rest, 0, mesh.positions as Float32Array<ArrayBuffer>);
device.queue.writeBuffer(posed, 0, mesh.positions as Float32Array<ArrayBuffer>);
device.queue.writeBuffer(indexBuffer, 0, mesh.indices as Uint32Array<ArrayBuffer>);
const vertices = { positionBuffer: posed, vertexCount, indexBuffer, indexFormat: 'uint32' as const };

// ---- The deform pass: rest pose in, posed vertices out.

const DEFORMS = ['twist', 'wave', 'stretch'] as const;
type Deform = (typeof DEFORMS)[number];
const DEFORM_WGSL = /* wgsl */ `
struct Deform { time: f32, amount: f32, vertexCount: u32, mode: u32, centerY: f32, halfHeight: f32, pad0: f32, pad1: f32 }
@group(0) @binding(0) var<uniform> deform: Deform;
@group(0) @binding(1) var<storage, read> rest: array<f32>;
@group(0) @binding(2) var<storage, read_write> posed: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let v = id.x + id.y * groups.x * 64u;
    if (v >= deform.vertexCount) { return; }
    var p = vec3f(rest[v * 3u], rest[v * 3u + 1u], rest[v * 3u + 2u]);
    let h = (p.y - deform.centerY) / deform.halfHeight;
    let a = deform.amount;
    let s = sin(deform.time);
    switch deform.mode {
        case 0u: {
            let angle = a * s * h * 3.14159;
            p = vec3f(cos(angle) * p.x - sin(angle) * p.z, p.y, sin(angle) * p.x + cos(angle) * p.z);
        }
        case 1u: {
            let r = length(p.xz);
            let direction = select(vec2f(0.0), p.xz / r, r > 1e-6);
            let w = 0.15 * a * deform.halfHeight * sin(h * 9.0 + deform.time * 3.0);
            p = vec3f(p.x + direction.x * w, p.y, p.z + direction.y * w);
        }
        default: {
            let k = 1.0 + a * (0.65 * s + 0.15);
            p = vec3f(p.x / sqrt(k), deform.centerY + (p.y - deform.centerY) * k, p.z / sqrt(k));
        }
    }
    posed[v * 3u] = p.x;
    posed[v * 3u + 1u] = p.y;
    posed[v * 3u + 2u] = p.z;
}`;
const deformUniform = device.createBuffer({ label: 'refit deform', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const deformPipeline = await device.createComputePipelineAsync({ label: 'refit deform', layout: 'auto', compute: { module: device.createShaderModule({ label: 'refit deform', code: DEFORM_WGSL }), entryPoint: 'main' } });
const deformGroup = device.createBindGroup({
    layout: deformPipeline.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: { buffer: deformUniform } },
        { binding: 1, resource: { buffer: rest } },
        { binding: 2, resource: { buffer: posed } },
    ],
});
const deformGroups = Math.ceil(vertexCount / 64);
const deformX = Math.min(deformGroups, device.limits.maxComputeWorkgroupsPerDimension);

// ---- Trees: H-PLOC from the posed buffer (the first from the rest pose), the floor's once with tinybvh.

const hploc = new GpuHplocBuilder(device, { label: 'refit' });
await hploc.prepare();
const tinybvh = await createTinyBvhBuilder();
const floorLayout = tinybvh.buildLayout(floor).layout;
const kernels = new GpuMeshRefitKernels(device, 'refit');
await kernels.prepare();
const buildTimer = GpuTimer.create(device, 1);

interface Built {
    readonly layout: GpuLayoutMeshBvh;
    readonly cpuMs: number;
    readonly gpuMs: number | null;
    /** The tree's relative SAH cost (summed node half-areas over the root's), as the refit's stats measure it. */
    readonly cost: number;
}

/**
 * One H-PLOC build from the posed vertices as they are when the build's copy
 * runs (after every frame submitted before it). Only rebuilds are `timed`
 * (they run one at a time; the comparison builds may overlap them).
 */
async function buildFromPose(timed: boolean): Promise<Built> {
    const start = performance.now();
    const batch = hploc.createBatch([{ positionBuffer: posed, vertexCount, indexBuffer, indexFormat: 'uint32', triangleCount }]);
    const timer = timed && buildTimer && !buildTimer.pending ? buildTimer : null;
    let trees = null;
    try {
        const encoder = device.createCommandEncoder({ label: 'refit build' });
        batch.encode(encoder, timer?.pass());
        timer?.resolveInto(encoder);
        device.queue.submit([encoder.finish()]);
        trees = await batch.finish();
        const cpuMs = performance.now() - start;
        const gpuMs = timer ? (await timer.result())[0] : null;
        if (trees.meshes[0].problem) throw new Error(`H-PLOC refused the mesh: ${trees.meshes[0].problem}`);
        const [layout] = await hploc.readLayouts(trees);
        // relativeSahCost reads 64-byte f32 nodes: the GPU builders write those unless asked for quantized ones (dequantizeNodes would give them).
        if ((layout!.nodeFormat ?? 'f32') !== 'f32') throw new Error(`refit: expected an f32 tree from H-PLOC, got ${layout!.nodeFormat} nodes`);
        return { layout: layout!, cpuMs, gpuMs, cost: relativeSahCost(new Float32Array(layout!.nodes), new Uint32Array(layout!.nodes), 0, layout!.nodeCount) };
    } catch (error) {
        timer?.reset();
        throw error;
    } finally {
        trees?.destroy();
        batch.destroy();
    }
}

// ---- The scene: the knot (its top-level box the deform box) on the floor.

const instances: TraceSceneInstance[] = [
    { meshIndex: 0, instanceIndex: 0, objectToWorld: mat4.compose(KNOT_AT, [0, 0, 0, 1], [1, 1, 1]), material: 1 },
    { meshIndex: 1, instanceIndex: 1, objectToWorld: mat4.identity(), material: 0 },
];

interface Current {
    readonly data: TraceSceneData;
    readonly buffers: TraceSceneBuffers;
    readonly source: TraceSource;
    readonly refit: GpuMeshRefit;
    readonly built: Built;
    readonly generation: number;
}
let current: Current | null = null;
let generation = 0;

function assemble(layout: GpuLayoutMeshBvh): TraceSceneData {
    // The scene's default node format (quantized). The top level boxes the knot by its deform box, so no pose leaves it.
    return assembleTraceScene([{ ...layout, bounds: deformBox }, floorLayout], instances);
}

/** When the drift first went past the limit (auto rebuild), or null. */
let driftSince: number | null = null;
/** The current tree's refit cost over its built cost, from its latest stats; NaN until the tree's first sample. */
let drift = NaN;
const stats = { rebuilds: 0, automatic: 0, swaps: 0, destroyed: 0 };
let rebuilds: Promise<void> = Promise.resolve();
let rebuildsAsked = 0;
/** Rebuilds run one after another; asking during one queues the next. */
function rebuild(reason: 'button' | 'auto' | 'start'): Promise<void> {
    rebuildsAsked++;
    const next = rebuilds.then(() => rebuildNow(reason)).finally(() => rebuildsAsked--);
    rebuilds = next.catch(() => undefined);
    return next;
}

/** Build new, swap, then destroy the old: the order the swap needs. */
async function rebuildNow(reason: 'button' | 'auto' | 'start'): Promise<void> {
    const built = await buildFromPose(true);
    const data = assemble(built.layout);
    const buffers = uploadTraceScene(device, data, 'refit');
    const refit = new GpuMeshRefit(device, { label: 'refit knot', kernels, diagnostics: true });
    const placement = meshRefitPlacement(data, 0); // offsets and the scene's node format
    const next: Current = { data, buffers, source: traceSourceOf(device, buffers), refit, built, generation: ++generation };
    refit.setTree(built.layout, { ...placement, generation: next.generation });
    // Swapped between frames (nothing here awaits): the next frame refits and traces the new scene.
    const old = current;
    current = next;
    if (reason !== 'start') stats.rebuilds++;
    if (reason === 'auto') stats.automatic++;
    if (old) {
        stats.swaps++;
        // Frames already submitted keep the old buffers alive on the GPU; nothing recorded from here on names them.
        old.refit.dispose();
        old.buffers.destroy();
        stats.destroyed++;
    }
    // The old tree's drift says nothing about the new one: wait for the new tree's stats before another auto rebuild.
    driftSince = null;
    drift = NaN;
    overlay.build({ builtBy: 'gpu-hploc', fallbackReason: null, cpuMs: built.cpuMs, gpuMs: built.gpuMs });
    overlay.tree(layoutInfo(built.layout));
    overlay.set('scene on the GPU', `${bytes(buffers.byteLength)} (${next.source.triangleLeaves}, ${next.source.nodeFormat} nodes), refit ${bytes(refit.memoryBytes)}`);
}

await rebuild('start');
const view = RayCastView.create(device, app.format);
await view.prepare(current!.source);
camera.frame([-8, 0, -8, 8, 5, 8]);
camera.target = [0, 1.6, 0];
camera.distance *= 0.36;

// ---- Controls.

const settings = { deform: 'twist' as Deform, amount: 0.8, speed: 1, animate: true, auto: true, limit: 1.25, compare: true, mode: 'normal' as ViewMode, shadows: true };
overlay.select('deform', DEFORMS, settings.deform, (value) => (settings.deform = value));
overlay.range('amount', 0, 1, 0.05, settings.amount, (value) => (settings.amount = value));
overlay.range('speed', 0, 3, 0.1, settings.speed, (value) => (settings.speed = value));
overlay.checkbox('animate', settings.animate, (value) => (settings.animate = value));
overlay.checkbox('auto rebuild', settings.auto, (value) => (settings.auto = value));
overlay.range('… past drift', 1.05, 2, 0.05, settings.limit, (value) => (settings.limit = value));
overlay.checkbox('compare with a rebuild', settings.compare, (value) => (settings.compare = value));
overlay.button('rebuild now', () => void rebuild('button'));
overlay.select('show', VIEW_MODES, settings.mode, (value) => (settings.mode = value));
overlay.checkbox('shadows (any hit)', settings.shadows, (value) => (settings.shadows = value));
overlay.note(
    `${triangleCount.toLocaleString('en-US')} triangles posed by a compute pass every frame, then GpuMeshRefit in the same pass, in the scene's own buffers. Drift: the refit tree's relative SAH cost over the built tree's. Auto rebuild: past the limit for half a second, H-PLOC from the posed buffer.`,
);

// ---- The frame: pose, refit (one compute pass, timed), trace.

const frameTimer = GpuTimer.create(device, 2);
let measuring = false;
let time = 0;
let frame = 0;
let compareCost = NaN;
let comparing = false;
let lastCompare = 0;
let smoke: Promise<void> | null = null;
const sun = vec3.normalize([0.45, 0.85, 0.3]);

function poseUniform(): ArrayBuffer {
    const buffer = new ArrayBuffer(32);
    const floats = new Float32Array(buffer);
    const words = new Uint32Array(buffer);
    floats[0] = time;
    floats[1] = settings.amount;
    words[2] = vertexCount;
    words[3] = DEFORMS.indexOf(settings.deform);
    floats[4] = centerY;
    floats[5] = halfHeight;
    return buffer;
}

app.loop({
    async update(now, delta) {
        frame++;
        if (settings.animate) time += delta * settings.speed;
        const scene = current!;
        scene.refit.pollStats();
        const latest = scene.refit.latestStats;
        // Only the stats of this scene's tree (its generation): a sample from before a swap says nothing about the new tree.
        if (latest && latest.generation === scene.generation) drift = latest.relativeCost / scene.built.cost;
        if (settings.auto && drift > settings.limit && rebuildsAsked === 0) {
            driftSince ??= now;
            if (now - driftSince > 500) void rebuild('auto');
        } else if (!(drift > settings.limit)) driftSince = null;
        if (settings.compare && !comparing && now - lastCompare > 1000) {
            // A tree from this pose, never placed: what a rebuild would give now.
            comparing = true;
            lastCompare = now;
            void buildFromPose(false)
                .then((fresh) => (compareCost = fresh.cost))
                .finally(() => (comparing = false));
        }
        const [{ measurement, pick }, times] = await Promise.all([view.collect(), measuring && frameTimer ? frameTimer.result() : Promise.resolve(null)]);
        if (times) {
            measuring = false;
            overlay.set('pose + refit (GPU)', `${gpuMsText(times[0])} pose, ${gpuMsText(times[1])} refit (${triangleCount.toLocaleString('en-US')} triangles)`);
        }
        if (measurement?.gpuMs) overlay.rays('primary (GPU timestamps)', measurement.primaryRays / (measurement.gpuMs / 1000));
        if (pick !== undefined) overlay.set('pixel', pick ? `instance ${pick.instance}, triangle ${pick.triangle}, t ${pick.t.toFixed(3)}` : '-');
        if (frame % 10 === 0) {
            const refitCost = Number.isFinite(drift) ? drift * scene.built.cost : NaN;
            overlay.set('drift (refit / built)', Number.isFinite(drift) ? `${drift.toFixed(3)}× (relative SAH ${refitCost.toFixed(1)} against ${scene.built.cost.toFixed(1)} built)` : 'waiting for the refit stats');
            overlay.set('refit / a rebuild now', Number.isFinite(compareCost) && Number.isFinite(refitCost) ? `${(refitCost / compareCost).toFixed(3)}× (a fresh H-PLOC tree of this pose: ${compareCost.toFixed(1)})` : settings.compare ? '…' : 'off');
            overlay.set('rebuilds', `${stats.rebuilds} (${stats.automatic} automatic), last ${scene.built.cpuMs.toFixed(1)} ms`);
        }
        if (isSmoke && frame === 30) smoke ??= smokeRun();
    },
    render(encoder, target) {
        const scene = current!;
        device.queue.writeBuffer(deformUniform, 0, poseUniform());
        const measure = frame % 15 === 0 && !measuring && frameTimer !== null && !frameTimer.pending;
        // The pose and the refit in two passes (one timestamp pair each); the refit reads what the pose wrote.
        const posePass = encoder.beginComputePass({ label: 'refit pose', timestampWrites: measure ? frameTimer!.pass() : undefined });
        posePass.setPipeline(deformPipeline);
        posePass.setBindGroup(0, deformGroup);
        posePass.dispatchWorkgroups(deformX, Math.ceil(deformGroups / deformX));
        posePass.end();
        const refitPass = encoder.beginComputePass({ label: 'refit', timestampWrites: measure ? frameTimer!.pass() : undefined });
        scene.refit.encode(refitPass, { nodes: scene.buffers.nodes, triangles: scene.buffers.triangles }, vertices);
        refitPass.end();
        if (measure) {
            frameTimer!.resolveInto(encoder);
            measuring = true;
        }
        scene.refit.copyStats(encoder);
        const width = canvas.width;
        const height = canvas.height;
        const pick = pointer ? ([Math.floor(pointer[0] * width), Math.floor(pointer[1] * height)] as const) : null;
        view.trace(encoder, scene.source, camera.basis(width / height), { mode: settings.mode, shadows: settings.shadows, sun, highlight: null, sceneRadius: 8 }, width, height, { measure: frame % 15 === 0, pick });
        view.draw(encoder, target);
    },
});

let pointer: [number, number] | null = null;
canvas.addEventListener('pointermove', (event) => (pointer = [event.clientX / canvas.clientWidth, event.clientY / canvas.clientHeight]));
canvas.addEventListener('pointerleave', () => (pointer = null));

/** The posed vertices, read back (for the check only: the page itself never reads the pose). */
async function readPose(): Promise<Float32Array> {
    const read = device.createBuffer({ label: 'refit pose read', size: posed.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'refit pose read' });
    encoder.copyBufferToBuffer(posed, 0, read, 0, posed.size);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const positions = new Float32Array(read.getMappedRange().slice(0));
    read.destroy();
    return positions;
}

/**
 * The view against the CPU walk over a tree built from scratch (tinybvh) of
 * the pose the GPU holds: a stale refit, or one that missed a triangle,
 * disagrees. Frozen first, so the pose, the refit and the frame are one.
 */
async function checkPose(label: string): Promise<Record<string, unknown>> {
    settings.animate = false;
    await nextFrames(4);
    const positions = await readPose();
    let moved = 0;
    for (let i = 0; i < positions.length; i++) moved = Math.max(moved, Math.abs(positions[i] - mesh.positions[i]));
    const posedLayout = tinybvh.buildLayout({ positions, indices: mesh.indices }).layout;
    const cpu = createCpuTracer(assembleTraceScene([{ ...posedLayout, bounds: deformBox }, floorLayout], instances));
    await nextFrames(2);
    const agreement = await checkViewAgainstCpu(view, (aspect) => camera.basis(aspect), cpu);
    const pixels = await view.readPixels();
    let onKnot = 0;
    for (let y = 0; y < pixels.height; y += 8) for (let x = 0; x < pixels.width; x += 8) if (pixels.hit(x, y)?.instance === 0) onKnot++;
    if (onKnot < 50) throw new Error(`${label}: only ${onKnot} sampled pixels hit the deforming mesh`);
    settings.animate = true;
    return { label, moved, onKnot, agreement, drift };
}

/** Waits for a refit sample of the current tree. */
async function nextDrift(): Promise<number> {
    const scene = current!;
    for (let i = 0; i < 240; i++) {
        await nextFrames(1);
        const stats = scene.refit.latestStats;
        if (current === scene && stats && stats.generation === scene.generation) return stats.relativeCost / scene.built.cost;
    }
    throw new Error('no refit stats for the current tree in 240 frames');
}

/**
 * Smoke: (1) twisted far from the built pose, the view agrees with the CPU
 * walk on a fresh tree of the same pose; (2) mid-run, while the mesh moves:
 * two rebuilds asked at once (the second queued) and the deform switched;
 * then, frozen, a rebuild from the frozen pose must refit to its own cost
 * (drift 1.0: the stats are the new tree's) and the view agree again.
 */
async function smokeRun(): Promise<void> {
    settings.auto = false;
    settings.deform = 'twist';
    settings.amount = 1;
    time = Math.PI / 2; // full twist
    const twisted = await checkPose('twisted, refit from the rest pose\'s tree');
    if (!(twisted.moved as number > 0.5)) throw new Error(`the pose barely moved (${twisted.moved})`);
    const twistedDrift = await nextDrift();

    const swapsBefore = stats.swaps;
    settings.animate = true;
    const first = rebuild('button');
    settings.deform = 'stretch';
    const second = rebuild('button');
    await nextFrames(5);
    settings.auto = true;
    await Promise.all([first, second]);
    await nextFrames(10);
    if (stats.swaps < swapsBefore + 2) throw new Error(`two rebuilds asked, ${stats.swaps - swapsBefore} swapped in`);

    settings.auto = false;
    settings.animate = false;
    await nextFrames(3);
    await rebuild('button'); // from the frozen pose
    const rebuiltDrift = await nextDrift();
    if (Math.abs(rebuiltDrift - 1) > 0.02) throw new Error(`a tree rebuilt from the frozen pose refits to ${rebuiltDrift}× its built cost (expected 1)`);
    const stretched = await checkPose('stretched, after the swaps');
    settings.deform = 'twist';
    settings.auto = true;
    await nextFrames(20);
    await ready(device, { ...overlay.snapshot(), checks: [twisted, stretched], twistedDrift, rebuiltDrift, stats, compareCost });
}

function nextFrames(count: number): Promise<void> {
    const target = frame + count;
    return new Promise((resolve) => {
        const poll = () => (frame >= target ? resolve() : requestAnimationFrame(poll));
        poll();
    });
}

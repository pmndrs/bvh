/**
 * compare/skinned: keeping a skinned mesh's tree on its pose, every frame.
 *
 * - **three-mesh-bvh** (its `skinnedMesh` example): a `SkinnedMeshBVH` per
 *   character, `refit()` on the CPU each frame (it skins every vertex with
 *   `SkinnedMesh.getVertexPosition`). Its trees live on the CPU; to trace the
 *   pose on the GPU it re-packs them with `BVHComputeData.update()`, which also
 *   skins every vertex on the CPU (timed as its own row, a few frames).
 * - **ours**: `SceneTreesObject3DSync` with the renderer poses each
 *   `SkinnedMesh` on the GPU (`BvhPosePass`, three's skinning in a compute
 *   pass) and `SceneTrees.setPose` refits its tree from that buffer
 *   (`GpuMeshRefit`), in the frame's encoder. CPU ms is the sync and the
 *   recording; GPU ms the pose passes (three's compute timestamps) plus what
 *   `SceneTrees.update` records (its `timestampWrites`), or, without
 *   `timestamp-query`, the wall time from the start to the queue being done.
 *
 * Both run on the same pose in each measured frame (same bones, same time), in
 * alternating order. The live view runs either or both; the pointer picks with
 * three-mesh-bvh's refit tree (CPU) and ours (one GPU ray), one marker each.
 *
 * What else: a plain WebGPU compute pass, written here and never seen by three,
 * traces a top-down grid over the characters through the refit trees every
 * frame (`traceModuleWgsl` in the page's own kernel, bound to SceneTrees'
 * buffers with no copy), shown as a height image. Its rays are the smoke check:
 * on a frozen pose they agree with three-mesh-bvh's refit tree.
 */

import { Bone, Color, DoubleSide, Float32BufferAttribute, Group, Mesh, MeshBasicMaterial, MeshStandardMaterial, Raycaster, Skeleton, SkinnedMesh, SphereGeometry, Uint16BufferAttribute, Vector2, Vector3, type BufferGeometry, type Intersection } from 'three';
import { BVHHelper, getBVHExtremes, SkinnedMeshBVH } from 'three-mesh-bvh';
import { BVHComputeData } from 'three-mesh-bvh/webgpu';
import { GpuRays, RAY_FLOATS, SceneTrees, traceBuffersOf, traceModuleEntries, traceModuleWgsl, type TraceSceneBuffers } from '@pmndrs/bvh';
import { SceneTreesObject3DSync } from '@pmndrs/bvh/three';
import { median } from '../../../compare/core/measure.ts';
import { emptyAgreement, agreementText } from '../../../compare/core/agree.ts';
import type { Measurement } from '../../../compare/core/suites.ts';
import { cameraRay, createApp } from '../../shared/app.ts';
import { quick, ResultsPanel } from '../../shared/compare.ts';
import { GpuTimer } from '../../shared/gpu-timer.ts';
import { SceneTreesLog } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

const app = await createApp('compare/skinned', 'compare/skinned', { trackTimestamp: true });
const { device, renderer, scene, camera, controls, overlay } = app;
camera.position.set(0, 6, 15);
controls.target.set(0, 1.8, 0);
const panel = new ResultsPanel(
    'compare/skinned',
    "Skinned characters kept on their pose every frame: three-mesh-bvh's SkinnedMeshBVH.refit() on the CPU (its skinnedMesh example) against our GPU pose pass and refit (SceneTreesObject3DSync with a renderer, SceneTrees.setPose). Both on the same pose in each measured frame; medians.",
);
const log = (line: string) => panel.status(line);

const params = new URLSearchParams(location.search);
const COUNT = Number(params.get('characters') ?? 9);
const RINGS = Number(params.get('rings') ?? (quick ? 160 : 400));
const BONES = 6;

/** A tube along y (rings of a sphere), skinned to `bones` bones (as three/skinned). */
function tube(rings: number, bones: number): BufferGeometry {
    const geometry = new SphereGeometry(0.4, 32, rings);
    geometry.scale(1, 5, 1);
    geometry.translate(0, 2, 0);
    const position = geometry.attributes.position;
    const skinIndex = new Uint16Array(position.count * 4);
    const skinWeight = new Float32Array(position.count * 4);
    for (let i = 0; i < position.count; i++) {
        const t = Math.min(bones - 1.001, Math.max(0, (position.getY(i) / 4) * (bones - 1)));
        const bone = Math.floor(t);
        const f = t - bone;
        skinIndex.set([bone, Math.min(bones - 1, bone + 1), 0, 0], i * 4);
        skinWeight.set([1 - f, f, 0, 0], i * 4);
    }
    geometry.setAttribute('skinIndex', new Uint16BufferAttribute(skinIndex, 4));
    geometry.setAttribute('skinWeight', new Float32BufferAttribute(skinWeight, 4));
    return geometry;
}

function character(geometry: BufferGeometry, color: number): SkinnedMesh {
    const chain: Bone[] = [];
    for (let b = 0; b < BONES; b++) {
        const bone = new Bone();
        bone.position.y = b === 0 ? 0 : 4 / (BONES - 1);
        if (b > 0) chain[b - 1].add(bone);
        chain.push(bone);
    }
    // Double-sided, so three-mesh-bvh's raycast tests both faces as our walk does.
    const mesh = new SkinnedMesh(geometry, new MeshStandardMaterial({ color, roughness: 0.55, side: DoubleSide }));
    mesh.add(chain[0]);
    mesh.updateMatrixWorld(true);
    mesh.bind(new Skeleton(chain));
    mesh.frustumCulled = false;
    return mesh;
}

const geometry = tube(RINGS, BONES);
const trianglesEach = geometry.index!.count / 3;
const world = new Group();
scene.add(world);
const characters: SkinnedMesh[] = [];
for (let i = 0; i < COUNT; i++) {
    const mesh = character(geometry, [0x8fb3ff, 0xb0e0a8, 0xf0b3d0][i % 3]);
    mesh.position.set(((i % 3) - 1) * 3.2, 0, (Math.floor(i / 3) - 1) * -3.2);
    characters.push(mesh);
    world.add(mesh);
}
// The ground is drawn only: the trees hold the characters.
const ground = new Mesh(new SphereGeometry(40, 32, 16, 0, Math.PI * 2, 0, 0.12), new MeshStandardMaterial({ color: 0x2a2f38 }));
ground.position.y = -39.7;
scene.add(ground);
const markerOurs = new Mesh(new SphereGeometry(0.09, 12, 8), new MeshBasicMaterial({ color: 0xffc878 }));
const markerTheirs = new Mesh(new SphereGeometry(0.06, 12, 8), new MeshBasicMaterial({ color: 0xff6fa8 }));
markerOurs.visible = markerTheirs.visible = false;
scene.add(markerOurs, markerTheirs);

let time = 0;
function animate(at: number): void {
    time = at;
    characters.forEach((mesh, m) => {
        mesh.skeleton.bones.forEach((bone, b) => {
            if (b === 0) return;
            const phase = time * 1.8 + m * 0.7 + b * 0.45;
            bone.rotation.set(Math.sin(phase) * 0.35, 0, Math.cos(phase * 0.8) * 0.3);
        });
    });
    scene.updateMatrixWorld(true);
}
animate(0);

// ---- three-mesh-bvh: one SkinnedMeshBVH per character (SAH, its default strategy is CENTER; as its example: default options).
const buildStart = performance.now();
const theirs = characters.map((mesh) => {
    const bvh = new SkinnedMeshBVH(mesh);
    (mesh as unknown as { boundsTree: SkinnedMeshBVH }).boundsTree = bvh;
    return bvh;
});
const theirBuildMs = performance.now() - buildStart;
const sahScore = (): number => {
    let score = 0;
    for (const bvh of theirs) for (const extremes of getBVHExtremes(bvh as never)) score += extremes.surfaceAreaScore;
    return score;
};
const initialScore = sahScore();
function refitTheirs(): number {
    const start = performance.now();
    for (const bvh of theirs) bvh.refit();
    return performance.now() - start;
}
// Their way to trace the pose on the GPU: re-pack the refit trees (and skin every vertex on the CPU) for BVHComputeData.
const theirGpuData = new BVHComputeData(characters, { attributes: { position: 'vec4f' }, autogenerateBvh: false });

// ---- ours: the sync poses on the GPU and SceneTrees refits.
const treeLog = new SceneTreesLog();
const trees = await SceneTrees.create(device, { label: 'compare skinned', onTree: treeLog.onTree });
const sync = new SceneTreesObject3DSync(trees, world, { renderer });
const rays = await GpuRays.create(device, trees);
const timer = GpuTimer.create(device, 1);
const threeTimestamps = renderer.hasFeature('timestamp-query');

interface OursFrame {
    readonly cpuMs: number;
    readonly gpuMs: number;
}
const gpuMethod = timer && threeTimestamps ? 'timestamps (pose passes: three\'s compute timestamps; refit: SceneTrees.update timestampWrites)' : 'wall clock from the start to the queue being done (an upper bound; no timestamp-query)';

/** One update of ours: sync (pose passes), SceneTrees.update in an encoder, submitted. With `measure`, waits for the GPU. */
async function updateOurs(measure: boolean): Promise<OursFrame> {
    if (measure) await device.queue.onSubmittedWorkDone();
    const start = performance.now();
    sync.update();
    const encoder = device.createCommandEncoder({ label: 'compare skinned update' });
    trees.update(encoder, measure && timer ? { timestampWrites: timer.pass() } : {});
    if (measure) timer?.resolveInto(encoder);
    device.queue.submit([encoder.finish()]);
    const cpuMs = performance.now() - start;
    if (!measure) return { cpuMs, gpuMs: NaN };
    await device.queue.onSubmittedWorkDone();
    const wallMs = performance.now() - start;
    if (timer && threeTimestamps) {
        const [refitMs] = await timer.result();
        const poseMs = (await renderer.resolveTimestampsAsync('compute')) ?? 0;
        return { cpuMs, gpuMs: refitMs + poseMs };
    }
    if (timer) await timer.result();
    return { cpuMs, gpuMs: wallMs };
}

updateOurs(false);
await trees.whenBuilt();
await sync.ready;
updateOurs(false);
overlay.build(treeLog.build(trees));
overlay.tree(treeLog.tree(trees, sync.meshHandles));
overlay.set('characters', `${COUNT} × ${trianglesEach.toLocaleString('en-US')} triangles`);
log(`${COUNT} characters × ${trianglesEach.toLocaleString('en-US')} triangles; three-mesh-bvh built its SkinnedMeshBVHs in ${theirBuildMs.toFixed(0)} ms (CPU, skinning every vertex)`);
if (threeTimestamps) await renderer.resolveTimestampsAsync('compute'); // drop what the warm-up recorded

// ---- Measured frames: the same pose for both, the order alternating.
const frames = quick ? 12 : 120;
const uploadFrames = quick ? 2 : 10;
const theirMs: number[] = [];
const uploadMs: number[] = [];
const ourCpu: number[] = [];
const ourGpu: number[] = [];
for (let i = 0; i < frames; i++) {
    animate((i + 1) / 60);
    const runTheirs = () => theirMs.push(refitTheirs());
    const runOurs = async () => {
        const frame = await updateOurs(true);
        ourCpu.push(frame.cpuMs);
        ourGpu.push(frame.gpuMs);
    };
    if (i % 2 === 0) {
        runTheirs();
        await runOurs();
    } else {
        await runOurs();
        runTheirs();
    }
    if (i < uploadFrames) {
        const start = performance.now();
        theirGpuData.update();
        uploadMs.push(performance.now() - start);
    }
}
const degradation = sahScore() / initialScore - 1;
const scene_ = `${COUNT}x${trianglesEach}-skinned`;
const row = (library: string, metric: string, value: number | null, method: string, note?: string): Measurement => ({ page: 'skinned', scene: scene_, library, metric, value, unit: 'ms', method, ranked: true, note });
const measurements: Measurement[] = [
    row('ours', 'cpu', median(ourCpu), 'main-thread ms per frame: SceneTreesObject3DSync.update (pose pass dispatches, pose boxes, setPose) and SceneTrees.update recorded and submitted; median'),
    row('ours', 'gpu', median(ourGpu), `GPU ms per frame: ${gpuMethod}; median`),
    row('tmbvh', 'cpu', median(theirMs), `main-thread ms per frame: SkinnedMeshBVH.refit() of every character (CPU skinning of every vertex); median of ${frames}`),
    row('tmbvh', 'gpu', null, 'its trees stay on the CPU', 'CPU only'),
    row('tmbvh-upload', 'cpu', median(uploadMs), `main-thread ms: refit trees re-packed for the GPU with BVHComputeData.update() (skins every vertex again; the upload itself happens at the next compute and is not counted); median of ${uploadFrames}`),
    row('tmbvh-upload', 'gpu', null, 'the upload is not timed', 'not timed'),
];
panel.table(`Per frame, ${COUNT} characters × ${trianglesEach.toLocaleString('en-US')} triangles`, measurements, ['cpu', 'gpu'], {
    ours: '@pmndrs/bvh: GPU pose pass + GPU refit',
    tmbvh: 'three-mesh-bvh: SkinnedMeshBVH.refit()',
    'tmbvh-upload': "three-mesh-bvh: BVHComputeData.update() after the refit (to trace it on the GPU)",
});
panel.paragraph(
    `Tree quality after the refits: three-mesh-bvh's summed SAH score ${(degradation * 100).toFixed(1)}% above its fresh build (getBVHExtremes, as its example shows). Ours: ${trees.deformingCounts.refits.toLocaleString('en-US')} refits and ${trees.deformingCounts.driftRebuilds} drift rebuilds so far (SceneTrees rebuilds a refit tree on the GPU when it drifts too far; there is no CPU copy to score).`,
);
panel.paragraph(`GPU timing: ${gpuMethod}.`);

// ---- What else: a plain WebGPU compute pass over the refit trees, every frame (three never sees it).
const GRID = 96;
const HALF = 5.5;
const TOP = 9;
function heightKernel(source: TraceSceneBuffers): string {
    return /* wgsl */ `
${traceModuleWgsl({ traceSlot: { group: 0, firstBinding: 0 }, triangleLeaves: source.triangleLeaves, positionLayout: source.positionLayout, nodeFormat: source.nodeFormat, rayMasks: false })}
@group(1) @binding(0) var<storage, read_write> heights: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= ${GRID * GRID}u) { return; }
    let cell = vec2f(f32(i % ${GRID}u), f32(i / ${GRID}u)) + 0.5;
    let xz = (cell / ${GRID}.0) * ${2 * HALF} - ${HALF};
    let hit = bvhTraceClosest(BvhRay(vec3f(xz.x, ${TOP}.0, xz.y), 0.0, vec3f(0.0, -1.0, 0.0), 1e30));
    heights[i] = hit.t;
}`;
}
const heightBuffer = device.createBuffer({ label: 'compare skinned heights', size: GRID * GRID * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const heightRead = device.createBuffer({ label: 'compare skinned heights read', size: GRID * GRID * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
const traceSource = traceBuffersOf(trees);
const sceneLayout = device.createBindGroupLayout({ label: 'compare skinned scene', entries: traceModuleEntries({ triangleLeaves: traceSource.triangleLeaves }).map((entry, binding) => ({ ...entry, binding })) });
const ioLayout = device.createBindGroupLayout({ label: 'compare skinned io', entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }] });
const heightPipeline = await device.createComputePipelineAsync({
    label: 'compare skinned heights',
    layout: device.createPipelineLayout({ bindGroupLayouts: [sceneLayout, ioLayout] }),
    compute: { module: device.createShaderModule({ label: 'compare skinned heights', code: heightKernel(traceSource) }), entryPoint: 'main' },
});
const ioGroup = device.createBindGroup({ layout: ioLayout, entries: [{ binding: 0, resource: { buffer: heightBuffer } }] });
let sceneGroup: GPUBindGroup | null = null;
let sceneKey: GPUBuffer[] = [];
function traceHeights(encoder: GPUCommandEncoder): void {
    const buffers = traceBuffersOf(trees);
    const list = [buffers.nodes, buffers.triangles, buffers.topLevel, ...(buffers.positions ? [buffers.positions] : [])];
    // Made again when SceneTrees replaced a buffer (its storage grew).
    if (!sceneGroup || list.some((buffer, i) => buffer !== sceneKey[i])) {
        sceneGroup = device.createBindGroup({ layout: sceneLayout, entries: list.map((buffer, binding) => ({ binding, resource: { buffer } })) });
        sceneKey = list;
    }
    const pass = encoder.beginComputePass({ label: 'compare skinned heights' });
    pass.setPipeline(heightPipeline);
    pass.setBindGroup(0, sceneGroup);
    pass.setBindGroup(1, ioGroup);
    pass.dispatchWorkgroups(Math.ceil((GRID * GRID) / 64));
    pass.end();
}
let heightsBusy = false;
async function readHeights(): Promise<Float32Array> {
    heightsBusy = true;
    try {
        const encoder = device.createCommandEncoder({ label: 'compare skinned heights read' });
        encoder.copyBufferToBuffer(heightBuffer, 0, heightRead, 0, GRID * GRID * 4);
        device.queue.submit([encoder.finish()]);
        await heightRead.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(heightRead.getMappedRange().slice(0));
        heightRead.unmap();
        return out;
    } finally {
        heightsBusy = false;
    }
}
const inset = document.createElement('canvas');
inset.width = inset.height = GRID;
Object.assign(inset.style, { width: '192px', height: '192px', imageRendering: 'pixelated', display: 'block', marginTop: '6px', border: '1px solid rgba(255,255,255,0.15)' });
const inset2d = inset.getContext('2d')!;
const image = inset2d.createImageData(GRID, GRID);
const tint = new Color();
function drawHeights(heights: Float32Array): number {
    let hits = 0;
    for (let i = 0; i < GRID * GRID; i++) {
        const t = heights[i];
        if (t >= 0) {
            hits++;
            tint.setHSL(0.6 - Math.min(1, Math.max(0, (TOP - t) / 5)) * 0.55, 0.7, 0.55);
        } else tint.setRGB(0.07, 0.08, 0.1);
        image.data.set([tint.r * 255, tint.g * 255, tint.b * 255, 255], i * 4);
    }
    inset2d.putImageData(image, 0, 0);
    return hits;
}
const whatElse = panel.whatElse(
    `The refit trees are plain WebGPU buffers: this page's own compute pass (WGSL with traceModuleWgsl, bound to SceneTrees' buffers, no copy, never seen by three) traces a ${GRID}×${GRID} top-down grid through them every frame, in the same encoder as the refit. Height of the first hit below, live. three-mesh-bvh's refit trees are JS objects on the CPU; its GPU functions read them only after BVHComputeData re-packs them.`,
);
whatElse.append(inset);
panel.csv(device);

// ---- Live view.
const MODES = ['both', 'ours only', 'three-mesh-bvh only'] as const;
let mode: (typeof MODES)[number] = 'both';
let speed = 1;
overlay.select('refit each frame', MODES, mode, (value) => (mode = value));
overlay.range('speed', 0, 3, 0.1, speed, (value) => (speed = value));
const helpers = characters.map((mesh, i) => {
    const helper = new BVHHelper(mesh, theirs[i], 8);
    helper.visible = false;
    scene.add(helper);
    return helper;
});
let showHelpers = false;
overlay.checkbox("three-mesh-bvh's refit trees (BVHHelper)", showHelpers, (value) => {
    showHelpers = value;
    for (const helper of helpers) helper.visible = value;
});
overlay.note("Hover a character: the orange marker is our GPU ray on the GPU-refit tree, the pink one three-mesh-bvh's CPU raycast on its refit tree. With one library off, its marker follows a stale tree.");

const pointer = new Vector2(2, 2);
addEventListener('pointermove', (event) => pointer.set((event.clientX / innerWidth) * 2 - 1, -(event.clientY / innerHeight) * 2 + 1));
if (isSmoke) pointer.set(0, 0.1);
const raycaster = new Raycaster();
(raycaster as unknown as { firstHitOnly: boolean }).firstHitOnly = true;
function theirPick(origin: Vector3, direction: Vector3): Intersection | null {
    raycaster.set(origin, direction);
    let best: Intersection | null = null;
    for (let i = 0; i < characters.length; i++) {
        const found: Intersection[] = [];
        theirs[i].raycastObject3D(characters[i], raycaster, found);
        for (const hit of found) if (!best || hit.distance < best.distance) best = hit;
    }
    return best;
}

/** Smoke check: on a frozen pose, the what-else grid's rays (ours, GPU refit) against three-mesh-bvh's refit trees (CPU). */
async function checkAgainstTheirs(heights: Float32Array): Promise<{ checked: number; hits: number }> {
    const agreement = emptyAgreement();
    let hits = 0;
    const origin = new Vector3();
    const down = new Vector3(0, -1, 0);
    const hitAt = (x: number, y: number) => x >= 0 && y >= 0 && x < GRID && y < GRID && heights[y * GRID + x] >= 0;
    for (let i = 0; i < GRID * GRID; i += 7) {
        const x = i % GRID;
        const y = Math.floor(i / GRID);
        // Silhouette cells (a neighbour differs) are skipped: f32 GPU skinning and three's f64 CPU skinning may round a grazing ray either way.
        const here = hitAt(x, y);
        if (hitAt(x - 1, y) !== here || hitAt(x + 1, y) !== here || hitAt(x, y - 1) !== here || hitAt(x, y + 1) !== here) continue;
        origin.set(((x + 0.5) / GRID) * 2 * HALF - HALF, TOP, ((y + 0.5) / GRID) * 2 * HALF - HALF);
        const cpu = theirPick(origin, down);
        const t = heights[i];
        agreement.checked++;
        const same = here === !!cpu && (!cpu || Math.abs(cpu.distance - t) <= 1e-3 * Math.max(1, cpu.distance));
        if (same) agreement.same++;
        else {
            agreement.disagree++;
            if (agreement.examples.length < 5) agreement.examples.push(`cell ${x},${y}: ours ${here ? t.toFixed(4) : 'miss'}, three-mesh-bvh ${cpu ? cpu.distance.toFixed(4) : 'miss'}`);
        }
        if (here) hits++;
    }
    log(`frozen pose: our GPU refit tree against three-mesh-bvh's refit tree on ${agreement.checked} top-down rays: ${agreementText(agreement)}`);
    if (agreement.disagree) throw new Error(`our GPU refit trees disagree with three-mesh-bvh's refit trees: ${agreement.examples.join('; ')}`);
    if (hits < 50) throw new Error(`only ${hits} of the checked rays hit a character`);
    return { checked: agreement.checked, hits };
}

let frame = 0;
let liveTheirs = 0;
let liveOurs = 0;
let pickedTheirsMs = 0;
const ray = new Float32Array(RAY_FLOATS);
app.loop(async (_now, delta) => {
    frame++;
    animate(time + delta * speed);
    if (mode !== 'ours only') {
        liveTheirs = liveTheirs * 0.9 + refitTheirs() * 0.1;
        if (showHelpers) for (const helper of helpers) helper.update();
    }
    if (mode !== 'three-mesh-bvh only') {
        const start = performance.now();
        sync.update();
        const encoder = device.createCommandEncoder({ label: 'compare skinned frame' });
        trees.update(encoder);
        traceHeights(encoder);
        device.queue.submit([encoder.finish()]);
        liveOurs = liveOurs * 0.9 + (performance.now() - start) * 0.1;
    }
    if (frame % 10 === 0) {
        overlay.set('three-mesh-bvh refit (CPU)', mode === 'ours only' ? 'off' : `${liveTheirs.toFixed(2)} ms per frame`);
        overlay.set('ours pose + refit (CPU)', mode === 'three-mesh-bvh only' ? 'off' : `${liveOurs.toFixed(2)} ms per frame`);
        overlay.set('three-mesh-bvh pick (CPU)', `${pickedTheirsMs.toFixed(2)} ms`);
        overlay.set('our refits', trees.deformingCounts.refits);
        overlay.set('our drift rebuilds', trees.deformingCounts.driftRebuilds);
    }
    // Three's timestamp queries (trackTimestamp, for the measured frames) pile up unless resolved: drop the live frames'.
    if (threeTimestamps && frame % 10 === 0) {
        void renderer.resolveTimestampsAsync('render');
        void renderer.resolveTimestampsAsync('compute');
    }
    if (frame % 6 === 0 && !heightsBusy && mode !== 'three-mesh-bvh only') {
        overlay.set('top-down grid hits', drawHeights(await readHeights()));
    }
    if (!rays.pending && pointer.x <= 1) {
        cameraRay(camera, pointer, ray);
        const origin = new Vector3(ray[0], ray[1], ray[2]);
        const direction = new Vector3(ray[4], ray[5], ray[6]);
        const start = performance.now();
        const cpu = theirPick(origin, direction);
        pickedTheirsMs = performance.now() - start;
        markerTheirs.visible = !!cpu;
        if (cpu) markerTheirs.position.copy(cpu.point);
        const hit = (await rays.traceClosest(ray)).hit(0);
        markerOurs.visible = hit !== null;
        if (hit) markerOurs.position.copy(origin).addScaledVector(direction, hit.t);
    }
    if (isSmoke && frame === 40) {
        // Freeze the pose, bring both trees to it, trace the grid and compare.
        speed = 0;
        mode = 'both';
        animate(time);
        refitTheirs();
        await sync.ready;
        sync.update();
        const encoder = device.createCommandEncoder({ label: 'compare skinned check' });
        trees.update(encoder);
        traceHeights(encoder);
        device.queue.submit([encoder.finish()]);
        const heights = await readHeights();
        drawHeights(heights);
        const checked = await checkAgainstTheirs(heights);
        await ready(device, { ...overlay.snapshot(), csv: panel.csvText, status: panel.statusText, theirRefitMs: median(theirMs), theirUploadMs: median(uploadMs), ourCpuMs: median(ourCpu), ourGpuMs: median(ourGpu), gpuMethod, agreedWithTheirs: checked });
    }
});

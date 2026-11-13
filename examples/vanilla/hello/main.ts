/**
 * vanilla/hello: the smallest use of `@pmndrs/bvh`, with raw WebGPU and no
 * engine.
 *
 * 1. A device asked for `subgroups` when the adapter has it (shared/vanilla/app.ts).
 * 2. `createMeshLayoutBuilder(device)` picks the builder: GPU H-PLOC with
 *    subgroups, tinybvh (WebAssembly) without; the overlay says which and why.
 * 3. The tree goes into a one-instance scene (`assembleTraceScene`,
 *    `uploadTraceScene`): three storage buffers the WGSL walk reads.
 * 4. Rays on the GPU with the package's batch kernel (`TraceBatchKernel`):
 *    the pointer's ray (read back: the hit triangle is highlighted), and a
 *    batch of thousands whose hit points a render pass draws straight from
 *    the hit buffer, nothing read back.
 * 5. The tree's boxes at a depth (`bvhHelperPositions`) in a line pass.
 *
 * The mesh is a torus knot, a `?model=` glTF or a dropped one (its meshes
 * merged into one world-space mesh: this page is about one tree).
 */

import {
    bvhHelperIndices,
    bvhHelperPositions,
    CpuMeshBvh,
    createCpuTracer,
    createMeshLayoutBuilder,
    createRandom,
    HIT_RECORD_BYTES,
    HIT_RECORD_WORD,
    IDENTITY_TRANSFORM,
    RAY_BYTES,
    RAY_FLOATS,
    TraceBatchKernel,
    uploadTraceScene,
    type GpuLayoutMeshBvh,
    type TraceSceneBuffers,
    type TraceSceneData,
} from '@pmndrs/bvh';
import { requestedModel } from '../../shared/assets.ts';
import { assembleLayouts, GpuRays, traceSourceOf } from '../../shared/gpu-rays.ts';
import { GpuTimer } from '../../shared/gpu-timer.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';
import { layoutInfo } from '../../shared/stats.ts';
import { createVanillaApp } from '../../shared/vanilla/app.ts';
import { checkAgainstCpu } from '../../shared/vanilla/check.ts';
import { mergeToWorld, singleMeshScene, torusKnot, type IndexedMesh, type VanillaScene } from '../../shared/vanilla/geometry.ts';
import { enableSceneDrop, loadGltfScene } from '../../shared/vanilla/gltf.ts';

const app = await createVanillaApp('Hello', 'vanilla/hello');
const { device, camera, overlay, canvas } = app;

const settings = { depth: 6, boxes: true, rays: 16_384, points: true };

// ---- Drawing: the mesh (vertex pulling from the same positions and indices the builder read), boxes and hit points.

const MESH_WGSL = /* wgsl */ `
struct Frame { viewProjection: mat4x4f, picked: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
struct Varyings { @builtin(position) clip: vec4f, @location(0) world: vec3f, @location(1) @interpolate(flat) triangle: u32 }
@vertex fn vertexMain(@builtin(vertex_index) vertex: u32) -> Varyings {
    let i = indices[vertex];
    let p = vec3f(positions[i * 3u], positions[i * 3u + 1u], positions[i * 3u + 2u]);
    return Varyings(frame.viewProjection * vec4f(p, 1.0), p, vertex / 3u);
}
@fragment fn fragmentMain(in: Varyings) -> @location(0) vec4f {
    let n = normalize(cross(dpdx(in.world), dpdy(in.world)));
    let light = 0.35 + 0.65 * abs(dot(n, normalize(vec3f(0.4, 0.8, 0.45))));
    let base = select(vec3f(0.56, 0.70, 1.0), vec3f(1.0, 0.55, 0.2), in.triangle == frame.picked);
    return vec4f(base * light, 1.0);
}`;

const LINES_WGSL = /* wgsl */ `
@group(0) @binding(0) var<uniform> viewProjection: mat4x4f;
@vertex fn vertexMain(@location(0) p: vec3f) -> @builtin(position) vec4f { return viewProjection * vec4f(p, 1.0); }
@fragment fn fragmentMain() -> @location(0) vec4f { return vec4f(1.0, 0.78, 0.47, 0.35); }`;

// Hit points: one small quad per ray, placed from the ray and its hit record on the GPU.
const POINTS_WGSL = /* wgsl */ `
struct Frame { viewProjection: mat4x4f, pixel: vec2f, pad: vec2f }
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> rays: array<vec4f>;
@group(0) @binding(2) var<storage, read> hits: array<f32>;
@vertex fn vertexMain(@builtin(vertex_index) vertex: u32, @builtin(instance_index) ray: u32) -> @builtin(position) vec4f {
    let t = hits[ray * ${HIT_RECORD_BYTES / 4}u + ${HIT_RECORD_WORD.t}u];
    if (t < 0.0) { return vec4f(2.0, 2.0, 2.0, 1.0); }
    let p = rays[ray * 2u].xyz + rays[ray * 2u + 1u].xyz * t;
    var clip = frame.viewProjection * vec4f(p, 1.0);
    let corner = vec2f(f32(vertex & 1u), f32((vertex >> 1u) & 1u)) * 2.0 - 1.0;
    clip = vec4f(clip.xy + corner * frame.pixel * 1.5 * clip.w, clip.z - 1e-4 * clip.w, clip.w);
    return clip;
}
@fragment fn fragmentMain() -> @location(0) vec4f { return vec4f(0.44, 0.81, 0.59, 1.0); }`;

const depthFormat: GPUTextureFormat = 'depth24plus';
const meshModule = device.createShaderModule({ label: 'hello mesh', code: MESH_WGSL });
const meshPipeline = device.createRenderPipeline({
    label: 'hello mesh',
    layout: 'auto',
    vertex: { module: meshModule, entryPoint: 'vertexMain' },
    fragment: { module: meshModule, entryPoint: 'fragmentMain', targets: [{ format: app.format }] },
    depthStencil: { format: depthFormat, depthWriteEnabled: true, depthCompare: 'less' },
});
const linesModule = device.createShaderModule({ label: 'hello boxes', code: LINES_WGSL });
const blend: GPUBlendState = { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } };
const linesPipeline = device.createRenderPipeline({
    label: 'hello boxes',
    layout: 'auto',
    vertex: { module: linesModule, entryPoint: 'vertexMain', buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }] },
    fragment: { module: linesModule, entryPoint: 'fragmentMain', targets: [{ format: app.format, blend }] },
    primitive: { topology: 'line-list' },
    // Drawn over the mesh (no depth test): the tree reads as an x-ray.
    depthStencil: { format: depthFormat, depthWriteEnabled: false, depthCompare: 'always' },
});
const pointsModule = device.createShaderModule({ label: 'hello points', code: POINTS_WGSL });
const pointsPipeline = device.createRenderPipeline({
    label: 'hello points',
    layout: 'auto',
    vertex: { module: pointsModule, entryPoint: 'vertexMain' },
    fragment: { module: pointsModule, entryPoint: 'fragmentMain', targets: [{ format: app.format }] },
    primitive: { topology: 'triangle-strip' },
    depthStencil: { format: depthFormat, depthWriteEnabled: false, depthCompare: 'less-equal' },
});
const meshUniform = device.createBuffer({ label: 'hello mesh frame', size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const linesUniform = device.createBuffer({ label: 'hello boxes frame', size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const pointsUniform = device.createBuffer({ label: 'hello points frame', size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
let depthTexture: GPUTexture | null = null;

// ---- The tree, the scene, the rays.

interface Built {
    readonly mesh: IndexedMesh;
    readonly layout: GpuLayoutMeshBvh;
    readonly data: TraceSceneData;
    readonly buffers: TraceSceneBuffers;
    readonly tree: CpuMeshBvh;
    readonly meshGroup: GPUBindGroup;
    readonly positions: GPUBuffer;
    readonly indices: GPUBuffer;
    readonly radius: number;
    readonly center: [number, number, number];
}
let built: Built | null = null;
let picked = 0xffffffff;
let picks: GpuRays | null = null;
// The batch kernel, compiled for the scene's formats.
let batchKernel: TraceBatchKernel | null = null;
let kernelFormats = '';
const timer = GpuTimer.create(device, 1);

// Scene swaps (a drop while one loads) run one after another.
let swaps: Promise<void> = Promise.resolve();
function useScene(scene: VanillaScene): Promise<void> {
    const next = swaps.then(() => swapScene(scene));
    swaps = next.catch(() => undefined);
    return next;
}

async function swapScene(scene: VanillaScene): Promise<void> {
    const mesh = mergeToWorld(scene);
    // The builder: GPU H-PLOC with subgroups, else tinybvh; the chooser says why.
    const prepareStart = performance.now();
    const builder = await createMeshLayoutBuilder(device); // prepares the GPU builder's pipelines
    const prepareMs = performance.now() - prepareStart;
    const start = performance.now();
    const [result] = await builder.buildLayouts([mesh]);
    const cpuMs = performance.now() - start;
    // createMeshLayoutBuilder takes no timestamp writes: CPU wall time only (from the call to the layout read back).
    overlay.build({ builtBy: result.builtBy, fallbackReason: builder.fallbackReason ?? result.fallbackReason, cpuMs });
    overlay.set('builder setup', `${prepareMs.toFixed(0)} ms (pipelines; once per builder)`);
    overlay.tree(layoutInfo(result.layout));
    overlay.set('mesh', `${scene.name}: ${(mesh.indices.length / 3).toLocaleString('en-US')} triangles${scene.instances.length > 1 ? `, ${scene.instances.length} instances merged` : ''}`);
    if (scene.loadMs !== undefined) overlay.set('load', `${scene.loadMs.toFixed(0)} ms`);

    // One instance at the origin: the scene the walk reads, in the formats the tree came in (the builder's defaults).
    const data = assembleLayouts([result.layout], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY_TRANSFORM }], [mesh]);
    const buffers = uploadTraceScene(device, data, 'hello');
    const source = traceSourceOf(device, buffers);
    // Everything new is made before anything old goes: the frames go on with the old scene meanwhile.
    const formats = `${source.triangleLeaves}/${source.positionLayout}/${source.nodeFormat}`;
    const kernel =
        batchKernel && formats === kernelFormats
            ? batchKernel
            : await TraceBatchKernel.create(device, { label: 'hello batch', triangleLeaves: source.triangleLeaves, positionLayout: source.positionLayout, nodeFormat: source.nodeFormat });
    const newPicks = await GpuRays.create(source);

    const positions = device.createBuffer({ label: 'hello positions', size: mesh.positions.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const indices = device.createBuffer({ label: 'hello indices', size: mesh.indices.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(positions, 0, mesh.positions as Float32Array<ArrayBuffer>);
    device.queue.writeBuffer(indices, 0, mesh.indices as Uint32Array<ArrayBuffer>);
    const meshGroup = device.createBindGroup({
        layout: meshPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: meshUniform } },
            { binding: 1, resource: { buffer: positions } },
            { binding: 2, resource: { buffer: indices } },
        ],
    });
    const box = result.layout.bounds;
    const center: [number, number, number] = [(box[0] + box[3]) / 2, (box[1] + box[4]) / 2, (box[2] + box[5]) / 2];
    const radius = Math.hypot(box[3] - box[0], box[4] - box[1], box[5] - box[2]) / 2;
    const tree = new CpuMeshBvh(result.layout, { source: mesh });

    // The swap, in one go (no await): frames after it see only the new scene.
    const old = { built, picks, kernel: batchKernel !== kernel ? batchKernel : null };
    built = { mesh, layout: result.layout, data, buffers, tree, meshGroup, positions, indices, radius, center };
    picks = newPicks;
    batchKernel = kernel;
    kernelFormats = formats;
    picked = 0xffffffff;
    camera.frame(box);
    makeBoxes();
    makeRays();
    // The old scene goes once the picks already asked of it have finished (their trace reads its buffers).
    old.kernel?.destroy();
    await old.picks?.retire();
    if (old.built) {
        old.built.buffers.destroy();
        old.built.positions.destroy();
        old.built.indices.destroy();
    }
}

// The tree's boxes at `settings.depth`, from a CPU view of the same layout.
let boxes: { vertices: GPUBuffer; indices: GPUBuffer; format: GPUIndexFormat; count: number } | null = null;
const linesGroup = device.createBindGroup({ layout: linesPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: linesUniform } }] });
function makeBoxes(): void {
    if (!built) return;
    boxes?.vertices.destroy();
    boxes?.indices.destroy();
    const corners = bvhHelperPositions(built.tree, settings.depth, false);
    const edges = bvhHelperIndices(corners, true);
    const vertices = device.createBuffer({ label: 'hello box corners', size: Math.max(16, corners.byteLength), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(vertices, 0, corners as Float32Array<ArrayBuffer>);
    // Index buffers are written in multiples of 4 bytes.
    const indexBytes = Math.ceil(edges.byteLength / 4) * 4;
    const indices = device.createBuffer({ label: 'hello box edges', size: Math.max(4, indexBytes), usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    const padded = new Uint8Array(indexBytes);
    padded.set(new Uint8Array(edges.buffer, edges.byteOffset, edges.byteLength));
    device.queue.writeBuffer(indices, 0, padded);
    boxes = { vertices, indices, format: edges instanceof Uint16Array ? 'uint16' : 'uint32', count: edges.length };
    overlay.set('boxes drawn', `${(corners.length / 24).toLocaleString('en-US')} at depth ${settings.depth}`);
}

// The batch: rays from a sphere around the mesh towards points inside its box.
let batch: { rays: GPUBuffer; hits: GPUBuffer; count: number; packed: Float32Array; group: GPUBindGroup } | null = null;
let seed = 1;
function makeRays(): void {
    if (!built) return;
    batch?.rays.destroy();
    batch?.hits.destroy();
    const count = settings.rays;
    const random = createRandom(seed++);
    const packed = new Float32Array(count * RAY_FLOATS);
    const { center, radius } = built;
    for (let i = 0; i < count; i++) {
        const z = random() * 2 - 1;
        const phi = random() * Math.PI * 2;
        const s = Math.sqrt(1 - z * z);
        const origin = [center[0] + Math.cos(phi) * s * radius * 2, center[1] + z * radius * 2, center[2] + Math.sin(phi) * s * radius * 2];
        const aim = [center[0] + (random() - 0.5) * radius, center[1] + (random() - 0.5) * radius, center[2] + (random() - 0.5) * radius];
        const d = [aim[0] - origin[0], aim[1] - origin[1], aim[2] - origin[2]];
        const length = Math.hypot(d[0], d[1], d[2]);
        packed.set([origin[0], origin[1], origin[2], 0, d[0] / length, d[1] / length, d[2] / length, 1e30], i * RAY_FLOATS);
    }
    const rays = device.createBuffer({ label: 'hello rays', size: count * RAY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(rays, 0, packed);
    const hits = device.createBuffer({ label: 'hello hits', size: count * HIT_RECORD_BYTES, usage: GPUBufferUsage.STORAGE });
    const group = device.createBindGroup({
        layout: pointsPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: pointsUniform } },
            { binding: 1, resource: { buffer: rays } },
            { binding: 2, resource: { buffer: hits } },
        ],
    });
    batch = { rays, hits, count, packed, group };
    overlay.set('batch', `${count.toLocaleString('en-US')} rays from around the mesh, every frame`);
}

// ---- Controls and the pointer.

overlay.range('box depth', 1, 20, 1, settings.depth, (value) => {
    settings.depth = value;
    makeBoxes();
});
overlay.checkbox('boxes', settings.boxes, (value) => (settings.boxes = value));
overlay.select('batch rays', ['1024', '16384', '65536', '262144'] as const, '16384', (value) => {
    settings.rays = Number(value);
    makeRays();
});
overlay.checkbox('hit points', settings.points, (value) => (settings.points = value));
overlay.button('new rays', makeRays);
overlay.note('Hover the mesh: one GPU ray picks a triangle. Green points: the batch\'s hits, drawn from the hit buffer with no readback. Drag to orbit, wheel to zoom, drop a glTF.');

let pointer: [number, number] | null = null;
canvas.addEventListener('pointermove', (event) => (pointer = [(event.clientX / canvas.clientWidth) * 2 - 1, 1 - (event.clientY / canvas.clientHeight) * 2]));
canvas.addEventListener('pointerleave', () => (pointer = null));

const model = requestedModel();
await useScene(model ? await loadGltfScene(model) : singleMeshScene('torus knot', torusKnot()));
enableSceneDrop(useScene, (error) => overlay.error(String(error)));

// ---- The frame.

let frame = 0;
let measuring = false;
let checked: Record<string, unknown> | null = null;
let swapped = 0;
app.loop({
    async update() {
        frame++;
        if (!built || !picks) return;
        // The previous frame's batch timing (its timestamps were resolved in that frame's encoder).
        if (measuring && timer) {
            const [ms] = await timer.result();
            measuring = false;
            if (batch && ms > 0) overlay.rays('batch (GPU timestamps)', batch.count / (ms / 1000));
        }
        if (pointer && !picks.pending) {
            const result = await picks.traceClosest(camera.packRay(pointer[0], pointer[1], app.aspect));
            const hit = result.hit(0);
            picked = hit ? hit.triangle : 0xffffffff;
            overlay.set('picked', hit ? `triangle ${hit.triangle.toLocaleString('en-US')} at t = ${hit.t.toFixed(3)}${hit.frontFace ? '' : ' (back face)'}` : '-');
        }
        if (isSmoke && frame === 30 && !checked) {
            checked = await smokeCheck();
            // A model swap with a pick in flight: the old GpuRays and scene are retired after it, not destroyed under it.
            const inFlight = picks.traceClosest(camera.packRay(0, 0, app.aspect));
            await useScene(singleMeshScene('torus knot (3, 5)', torusKnot(1, 0.25, 500, 32, 3, 5)));
            await inFlight;
            swapped = frame;
        }
        if (isSmoke && swapped && frame === swapped + 10) await ready(device, { ...overlay.snapshot(), first: checked, afterSwap: await smokeCheck() });
    },
    render(encoder, target) {
        if (!built || !batch || !batchKernel) return;
        if (!depthTexture || depthTexture.width !== canvas.width || depthTexture.height !== canvas.height) {
            depthTexture?.destroy();
            depthTexture = device.createTexture({ size: [canvas.width, canvas.height], format: depthFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT });
        }
        const viewProjection = camera.viewProjection(app.aspect);
        const meshFrame = new Uint32Array(20);
        new Float32Array(meshFrame.buffer).set(viewProjection);
        meshFrame[16] = picked;
        device.queue.writeBuffer(meshUniform, 0, meshFrame);
        device.queue.writeBuffer(linesUniform, 0, viewProjection as Float32Array<ArrayBuffer>);
        const pointsFrame = new Float32Array(20);
        pointsFrame.set(viewProjection);
        pointsFrame.set([1 / canvas.width, 1 / canvas.height], 16);
        device.queue.writeBuffer(pointsUniform, 0, pointsFrame);

        // The batch, traced every frame; timed every 20th.
        const measure = !measuring && frame % 20 === 0 ? timer : null;
        batchKernel.encode(encoder, built.buffers, { mode: 'closest', rays: batch.rays, results: batch.hits, count: batch.count, timestampWrites: measure?.pass() });
        if (measure) {
            measure.resolveInto(encoder);
            measuring = true;
        }

        const pass = encoder.beginRenderPass({
            colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: [0.08, 0.09, 0.11, 1] }],
            depthStencilAttachment: { view: depthTexture.createView(), depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1 },
        });
        pass.setPipeline(meshPipeline);
        pass.setBindGroup(0, built.meshGroup);
        pass.draw(built.mesh.indices.length);
        if (settings.points) {
            pass.setPipeline(pointsPipeline);
            pass.setBindGroup(0, batch.group);
            pass.draw(4, batch.count);
        }
        if (settings.boxes && boxes && boxes.count > 0) {
            pass.setPipeline(linesPipeline);
            pass.setBindGroup(0, linesGroup);
            pass.setVertexBuffer(0, boxes.vertices);
            pass.setIndexBuffer(boxes.indices, boxes.format);
            pass.drawIndexed(boxes.count);
        }
        pass.end();
    },
});

/** Smoke: 1,024 of the batch's rays traced again and read back, against the CPU port of the walk; the centre ray hits. */
async function smokeCheck(): Promise<Record<string, unknown>> {
    const rays = batch!.packed.subarray(0, 1024 * RAY_FLOATS);
    const result = await picks!.traceClosest(rays);
    const agreement = checkAgainstCpu(rays, (index) => result.hit(index), createCpuTracer(built!.data));
    const centre = await picks!.traceClosest(camera.packRay(0, 0, app.aspect));
    if (!centre.hit(0)) throw new Error('the ray through the middle of the view missed the mesh');
    if (!boxes || boxes.count === 0) throw new Error('no boxes to draw');
    overlay.set('check', `${agreement.agree} of ${agreement.rays} GPU hits agree with the CPU walk`);
    return { agreement };
}

// For the smoke run's screenshot: the pointer in the middle until the mouse moves.
if (isSmoke) pointer = [0, 0];

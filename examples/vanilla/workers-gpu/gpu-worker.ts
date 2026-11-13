/**
 * vanilla/workers-gpu's worker: a `GPUDevice` of its own, for two of the
 * page's three ways to run a GPU build.
 *
 * - **build** (way 2): GPU H-PLOC or LBVH on this device, the trees read back
 *   (`readLayouts`) and transferred to the page, which uploads them to its
 *   own device. Buffers can't be shared between devices, so the trees cross
 *   through the CPU.
 * - **render** (way 3): everything here. `SceneTrees` on this device (GPU
 *   trees adopted with no copy, tinybvh's from a pool this worker starts),
 *   the shared ray-cast view, and the frames presented into an
 *   `OffscreenCanvas` the page transferred. The page only sends the camera.
 *
 * Errors (this device's included) go to the page, which fails its smoke run.
 */

import { createTinyBvhBuilder, GpuHplocBuilder, GpuLbvhBuilder, SceneTrees, type GpuLayoutMeshBvh } from '@pmndrs/bvh';
import type { CameraBasis } from '../../shared/vanilla/camera.ts';
import { viewCheckRays } from '../../shared/vanilla/check.ts';
import { RayCastView } from '../../shared/vanilla/raycast-view.ts';
import type { FromWorker, GpuBuilderKind, PixelRecord, ToWorker } from './protocol.ts';

// Typed with the DOM's lib (the examples' one program): a worker's postMessage takes the same options object.
const post = (message: FromWorker, transfer: Transferable[] = []) => self.postMessage(message, { transfer });
const report = (error: unknown) => post({ type: 'error', message: error instanceof Error ? `${error.message}\n${error.stack ?? ''}` : String(error) });
self.addEventListener('error', (event) => report(event.error ?? event.message));
self.addEventListener('unhandledrejection', (event) => report(event.reason));

let devicePromise: Promise<GPUDevice> | null = null;
let deviceMs = 0;
function device(): Promise<GPUDevice> {
    devicePromise ??= (async () => {
        const start = performance.now();
        const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) throw new Error('the worker has no WebGPU adapter');
        const wanted: GPUFeatureName[] = ['subgroups', 'timestamp-query'];
        const created = await adapter.requestDevice({
            label: 'workers-gpu worker',
            requiredFeatures: wanted.filter((feature) => adapter.features.has(feature)),
            requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
        });
        created.addEventListener('uncapturederror', (event) => report(`worker device: ${(event as GPUUncapturedErrorEvent).error.message}`));
        void created.lost.then((lost) => lost.reason !== 'destroyed' && report(`worker device lost: ${lost.message}`));
        deviceMs = performance.now() - start;
        return created;
    })();
    return devicePromise;
}

// ---- Way 2: build and send back.

const builders = new Map<GpuBuilderKind, GpuHplocBuilder | GpuLbvhBuilder>();

async function build(message: Extract<ToWorker, { type: 'build' }>): Promise<void> {
    const gpu = await device();
    const firstDevice = deviceMs;
    deviceMs = 0;
    let prepareMs = 0;
    let layouts: GpuLayoutMeshBvh[];
    let builtBy: GpuBuilderKind | 'tinybvh' = message.builder;
    let fallbackReason: string | null = null;
    let buildMs = 0;
    let readMs = 0;
    if (GpuHplocBuilder.isSupported(gpu)) {
        let builder = builders.get(message.builder);
        if (!builder) {
            const start = performance.now();
            builder = message.builder === 'gpu-lbvh' ? new GpuLbvhBuilder(gpu, { label: 'worker lbvh' }) : new GpuHplocBuilder(gpu, { label: 'worker hploc' });
            await builder.prepare();
            builders.set(message.builder, builder);
            prepareMs = performance.now() - start;
        }
        const start = performance.now();
        const trees = await builder.buildMany(message.meshes);
        buildMs = performance.now() - start;
        try {
            const refused = trees.meshes.findIndex((mesh) => mesh.problem);
            if (refused >= 0) throw new Error(`${message.builder} refused mesh ${refused}: ${trees.meshes[refused].problem}`);
            const read = performance.now();
            layouts = (await builder.readLayouts(trees)) as GpuLayoutMeshBvh[];
            readMs = performance.now() - read;
        } finally {
            trees.destroy();
        }
    } else {
        // The declared fallback: tinybvh, here in the worker.
        builtBy = 'tinybvh';
        fallbackReason = 'the worker\'s device has no "subgroups" feature';
        const tinybvh = await createTinyBvhBuilder();
        const start = performance.now();
        layouts = message.meshes.map((mesh) => tinybvh.buildLayout(mesh).layout);
        buildMs = performance.now() - start;
    }
    const transfer = new Set<ArrayBuffer>();
    let bytes = 0;
    for (const layout of layouts) {
        for (const buffer of [layout.nodes, layout.triangles, layout.parents.buffer, layout.bounds.buffer]) {
            if (buffer instanceof ArrayBuffer && !transfer.has(buffer)) {
                transfer.add(buffer);
                bytes += buffer.byteLength;
            }
        }
    }
    post({ type: 'built', id: message.id, layouts, builtBy, fallbackReason, deviceMs: firstDevice, prepareMs, buildMs, readMs, bytes }, [...transfer]);
}

// ---- Way 3: render here.

interface Renderer {
    readonly id: number;
    readonly device: GPUDevice;
    readonly trees: SceneTrees;
    readonly view: RayCastView;
    readonly context: GPUCanvasContext;
    readonly canvas: OffscreenCanvas;
    readonly radius: number;
    stopped: boolean;
    sentFirst: boolean;
    sentBest: boolean;
    frames: number;
    mrays: number | null;
}

let renderer: Renderer | null = null;
let camera: { basis: CameraBasis; width: number; height: number } | null = null;
/** The basis of the last frame traced (the check's rays). */
let traced: { basis: CameraBasis; width: number; height: number } | null = null;

async function render(message: Extract<ToWorker, { type: 'render' }>): Promise<void> {
    const gpu = await device();
    deviceMs = 0;
    const context = message.canvas.getContext('webgpu') as GPUCanvasContext | null;
    if (!context) throw new Error('the OffscreenCanvas gave no webgpu context');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device: gpu, format, alphaMode: 'opaque' });
    // GPU trees adopted as the scene's buffers (no copy: `f32` nodes), tinybvh's from a pool this worker starts.
    const trees = await SceneTrees.create(gpu, { label: 'worker scene', gpuTreeOutputs: 'adopt', nodeFormat: 'f32' });
    const handles = message.meshes.map((mesh) => trees.addMesh(mesh));
    message.instances.forEach((instance, index) => trees.addInstance({ mesh: handles[instance.mesh], objectToWorld: instance.objectToWorld, instanceIndex: index, material: instance.material }));
    const view = RayCastView.create(gpu, format);
    await view.prepare(trees);
    const next: Renderer = { id: message.id, device: gpu, trees, view, context, canvas: message.canvas, radius: message.radius, stopped: false, sentFirst: false, sentBest: false, frames: 0, mrays: null };
    // New first, then the swap, then the old one stopped and freed.
    const old = renderer;
    renderer = next;
    renderers.set(next.id, next);
    if (old) stop(old.id, false);
    requestAnimationFrame(() => void frame(next).catch(report));
}

const sunLength = Math.hypot(0.45, 0.85, 0.3);
const sun: [number, number, number] = [0.45 / sunLength, 0.85 / sunLength, 0.3 / sunLength];

async function frame(state: Renderer): Promise<void> {
    if (state.stopped) return;
    let measurement;
    try {
        ({ measurement } = await state.view.collect());
    } catch (error) {
        if (state.stopped) return; // stopped while reading back: its buffers are gone
        throw error;
    }
    if (state.stopped) return;
    if (measurement?.gpuMs) state.mrays = measurement.primaryRays / (measurement.gpuMs / 1000) / 1e6;
    const gpu = state.device;
    if (camera && (state.canvas.width !== camera.width || state.canvas.height !== camera.height)) {
        state.canvas.width = camera.width;
        state.canvas.height = camera.height;
    }
    const encoder = gpu.createCommandEncoder({ label: 'worker frame' });
    state.trees.update(encoder);
    if (camera) {
        state.view.trace(encoder, state.trees, camera.basis, { mode: 'material', shadows: true, sun, highlight: null, sceneRadius: state.radius }, state.canvas.width, state.canvas.height, { measure: state.frames % 15 === 0 });
        state.view.draw(encoder, state.context.getCurrentTexture().createView());
        traced = { ...camera, width: state.canvas.width, height: state.canvas.height };
    }
    gpu.queue.submit([encoder.finish()]);
    state.frames++;
    const counts = state.trees.counts;
    if (!state.sentFirst && counts.meshes > 0 && counts.onGpuTrees + counts.onTinybvhTrees === counts.meshes) {
        state.sentFirst = true;
        post({ type: 'trees', id: state.id, stage: 'first', counts });
    }
    if (!state.sentBest && counts.meshes > 0 && counts.onTinybvhTrees === counts.meshes) {
        state.sentBest = true;
        post({ type: 'trees', id: state.id, stage: 'best', counts });
    }
    if (state.frames % 15 === 0) post({ type: 'stats', id: state.id, counts, resident: state.trees.memory.resident, frames: state.frames, mrays: state.mrays });
    requestAnimationFrame(() => void frame(state).catch(report));
}

async function check(id: number): Promise<void> {
    const state = renderer;
    if (!state || state.id !== id || !traced) throw new Error('check: nothing rendered yet');
    const pixels = await state.view.readPixels();
    const { at } = viewCheckRays(traced.basis, pixels.width, pixels.height);
    const hits: (PixelRecord | null)[] = at.map(([x, y]) => {
        const hit = pixels.hit(x, y);
        return hit && { t: hit.t, instance: hit.instance, triangle: hit.triangle };
    });
    post({ type: 'pixels', id, basis: traced.basis, width: pixels.width, height: pixels.height, hits });
}

const renderers = new Map<number, Renderer>();

/** Stops a renderer: no frame is recorded with it after this; frames already submitted keep its buffers alive on the GPU. */
function stop(id: number, answer = true): void {
    const state = renderers.get(id);
    if (state) {
        renderers.delete(id);
        state.stopped = true;
        if (renderer === state) renderer = null;
        state.trees.dispose();
        state.view.destroy();
    }
    if (answer) post({ type: 'stopped', id });
}

self.addEventListener('message', (event: MessageEvent<ToWorker>) => {
    const message = event.data;
    const run = async () => {
        if (message.type === 'build') await build(message);
        else if (message.type === 'render') await render(message);
        else if (message.type === 'camera') camera = { basis: message.basis, width: message.width, height: message.height };
        else if (message.type === 'check') await check(message.id);
        else if (message.type === 'stop') stop(message.id);
    };
    run().catch(report);
});

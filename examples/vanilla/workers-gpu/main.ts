/**
 * vanilla/workers-gpu: three ways to run a GPU build, in raw WebGPU, on the
 * same meshes as `vanilla/workers-cpu`, against tinybvh's worker pool alone:
 * the time to a first usable scene against the time to the best trees.
 *
 * 1. **On the render device** (`SceneTrees`, `gpuTreeOutputs: 'adopt'`):
 *    one H-PLOC batch for every mesh, whose output buffers become the
 *    scene's buffers with no copy (`f32` nodes: an adopted batch isn't
 *    quantized). tinybvh's trees from the worker pool then swap in.
 * 2. **On a worker's own device, transferred:** the worker builds with
 *    H-PLOC or LBVH, reads the trees back (`readLayouts`) and transfers them;
 *    the page assembles and uploads them to its device. Buffers can't be
 *    shared between devices, so the trees cross through the CPU, and the
 *    assembly runs on the main thread.
 * 3. **Everything in a worker** with an `OffscreenCanvas`: the worker's
 *    device builds (as in 1) and renders; the page only sends the camera.
 *    The main thread is free, but the GPU is the same GPU.
 * 4. **tinybvh only** (the `vanilla/workers-cpu` pool): the first trees are
 *    the best ones, and they take the longest.
 *
 * Each run's row: first usable (every mesh traceable), best (every mesh on
 * tinybvh's tree, where the way has them), the worst gap between the main
 * thread's frames meanwhile, and the main thread's own share. A new run is
 * built before it replaces the one on screen, which is then freed; a way 2
 * build that comes back after another run started is dropped.
 */

import { assembleTraceScene, createCpuTracer, createTinyBvhBuilder, SceneTrees, uploadTraceScene, type GpuLayoutMeshBvh, type TraceSceneBuffers, type TraceSceneInstance } from '@pmndrs/bvh';
import { createBvhWorkerPool, type BvhWorkerPool } from '@pmndrs/bvh/worker';
import { assembleLayouts, traceSourceOf, type TraceSource } from '../../shared/gpu-rays.ts';
import { bytes, type TreeInfo } from '../../shared/overlay.ts';
import { fail, isSmoke, ready } from '../../shared/smoke.ts';
import { layoutInfo, SceneTreesLog, sumInfo } from '../../shared/stats.ts';
import { createVanillaApp } from '../../shared/vanilla/app.ts';
import { OrbitCamera } from '../../shared/vanilla/camera.ts';
import { checkAgainstCpu, checkViewAgainstCpu, viewCheckRays } from '../../shared/vanilla/check.ts';
import { galleryScene, type IndexedMesh } from '../../shared/vanilla/geometry.ts';
import { vec3 } from '../../shared/vanilla/math.ts';
import { RayCastView } from '../../shared/vanilla/raycast-view.ts';
import type { FromWorker, GpuBuilderKind, ToWorker } from './protocol.ts';

const app = await createVanillaApp('Workers (GPU)', 'vanilla/workers-gpu');
const { device, camera, overlay, canvas } = app;
const params = new URLSearchParams(location.search);

const WAYS = ['render device', 'worker device', 'worker canvas', 'tinybvh only'] as const;
type Way = (typeof WAYS)[number];

const scene = galleryScene(Number(params.get('count') ?? 32), Number(params.get('detail') ?? 1));
const radius = Math.hypot(scene.bounds[3] - scene.bounds[0], scene.bounds[4] - scene.bounds[1], scene.bounds[5] - scene.bounds[2]) / 2;
const instances: TraceSceneInstance[] = scene.instances.map((instance, index) => ({ meshIndex: instance.mesh, instanceIndex: index, objectToWorld: instance.objectToWorld, material: instance.material }));
camera.frame(scene.bounds);
camera.distance *= 0.7;

// ---- The worker (ways 2 and 3).

const worker = new Worker(new URL('./gpu-worker.ts', import.meta.url), { type: 'module', name: 'workers-gpu' });
const send = (message: ToWorker, transfer: Transferable[] = []) => worker.postMessage(message, transfer);
const answers = new Map<string, (message: FromWorker) => void>();
worker.addEventListener('message', (event: MessageEvent<FromWorker>) => {
    const message = event.data;
    if (message.type === 'error') {
        fail(`worker: ${message.message}`);
        return;
    }
    if (message.type === 'stats') {
        if (current?.kind === 'worker canvas' && current.id === message.id) current.stats = message;
        return;
    }
    const key = message.type === 'trees' ? `${message.type}:${message.id}:${message.stage}` : `${message.type}:${message.id}`;
    answers.get(key)?.(message);
    answers.delete(key);
});
worker.addEventListener('error', (event) => fail(`worker: ${event.message}`));
/** The worker's answer of `type` (and `stage`) to request `id`. */
function answer<T extends FromWorker['type']>(type: T, id: number, stage?: 'first' | 'best'): Promise<Extract<FromWorker, { type: T }>> {
    return new Promise((resolve) => answers.set(stage ? `${type}:${id}:${stage}` : `${type}:${id}`, resolve as (message: FromWorker) => void));
}

// ---- What is on screen: a SceneTrees (ways 1 and 4), an uploaded scene (way 2) or the worker's canvas (way 3).

type Current =
    | { readonly kind: 'trees'; readonly id: number; readonly trees: SceneTrees; readonly pool: BvhWorkerPool; readonly log: SceneTreesLog }
    | { readonly kind: 'uploaded'; readonly id: number; readonly buffers: TraceSceneBuffers; readonly source: TraceSource; readonly tree: TreeInfo }
    | { readonly kind: 'worker canvas'; readonly id: number; readonly element: HTMLCanvasElement; readonly camera: OrbitCamera; stats: Extract<FromWorker, { type: 'stats' }> | null };
let current: Current | null = null;

/** One ray-cast view per node format: a scene swap never meets a kernel compiled for the other format. */
const views = new Map<string, RayCastView>();
async function viewFor(source: TraceSource): Promise<RayCastView> {
    const key = `${source.triangleLeaves}/${source.positionLayout}/${source.nodeFormat}`;
    let view = views.get(key);
    if (!view) {
        view = RayCastView.create(device, app.format);
        await view.prepare(source);
        views.set(key, view);
    }
    return view;
}
const sourceOf = (on: Current): TraceSource | null => (on.kind === 'trees' ? on.trees : on.kind === 'uploaded' ? on.source : null);

/** Swaps `next` in, then frees what it replaced (frames already submitted keep the old buffers alive on the GPU). */
function swapIn(next: Current): void {
    const old = current;
    current = next;
    canvas.style.visibility = next.kind === 'worker canvas' ? 'hidden' : 'visible';
    if (!old) return;
    if (old.kind === 'trees') {
        old.trees.dispose();
        old.pool.dispose();
    } else if (old.kind === 'uploaded') old.buffers.destroy();
    else {
        send({ type: 'stop', id: old.id });
        camera.target = [...old.camera.target];
        Object.assign(camera, { distance: old.camera.distance, yaw: old.camera.yaw, pitch: old.camera.pitch });
        camera.version++;
        old.element.remove();
    }
}

// ---- Runs.

interface RunResult {
    readonly way: Way;
    readonly detail: string;
    readonly firstMs: number | null;
    readonly bestMs: number | null;
    readonly worstFrameMs: number;
    /** Main-thread milliseconds the way itself took (way 2's assembly and upload), where it has any. */
    readonly mainMs: number | null;
}
const results = new Map<Way, RunResult>();
let runId = 0;
let staleDropped = 0;
let worstGap = 0;
let lastTick = performance.now();
const tick = (now: number) => {
    worstGap = Math.max(worstGap, now - lastTick);
    lastTick = now;
    requestAnimationFrame(tick);
};
requestAnimationFrame(tick);

function record(result: RunResult): void {
    results.set(result.way, result);
    for (const row of results.values()) {
        const ms = (value: number | null) => (value === null ? '-' : `${value.toFixed(0)} ms`);
        overlay.set(row.way, `first ${ms(row.firstMs)}, best ${ms(row.bestMs)}, worst frame ${row.worstFrameMs.toFixed(0)} ms${row.mainMs === null ? '' : `, main thread ${row.mainMs.toFixed(0)} ms`} (${row.detail})`);
    }
}

const settings = { way: 'render device' as Way, builder: 'gpu-hploc' as GpuBuilderKind };

/** Starts a run of `way`; resolves when its first usable scene is on screen (ways 1, 3, 4 keep going to their best trees). */
async function run(way: Way): Promise<RunResult | null> {
    const id = ++runId;
    settings.way = way;
    waySelect.value = way;
    const started = performance.now();
    worstGap = 0;
    if (way === 'worker device') return runWorkerDevice(id, started);
    if (way === 'worker canvas') return runWorkerCanvas(id, started);
    return runSceneTrees(id, way, started);
}

/** Ways 1 and 4: a SceneTrees on the render device, with or without GPU first trees. */
async function runSceneTrees(id: number, way: Way, started: number): Promise<RunResult | null> {
    const gpuFirst = way === 'render device';
    const pool = createBvhWorkerPool();
    const log = new SceneTreesLog();
    const trees = await SceneTrees.create(device, { label: `workers-gpu ${way}`, gpuFirstTrees: gpuFirst, workers: pool, onTree: log.onTree, ...(gpuFirst ? { gpuTreeOutputs: 'adopt', nodeFormat: 'f32' } : {}) });
    const handles = scene.meshes.map((mesh) => trees.addMesh(mesh));
    scene.instances.forEach((instance, index) => trees.addInstance({ mesh: handles[instance.mesh], objectToWorld: instance.objectToWorld, instanceIndex: index, material: instance.material }));
    await viewFor(trees);
    if (id !== runId) {
        trees.dispose();
        pool.dispose();
        return null;
    }
    swapIn({ kind: 'trees', id, trees, pool, log });
    // Polled each frame (first: every mesh has a placed tree); a run replaced meanwhile stops polling its (disposed) scene.
    const first = new Promise<number>((resolve) => watch(() => id !== runId || (trees.counts.meshes > 0 && trees.counts.onGpuTrees + trees.counts.onTinybvhTrees === trees.counts.meshes), resolve));
    const best = new Promise<number>((resolve) => watch(() => id !== runId || (trees.counts.meshes > 0 && trees.counts.onTinybvhTrees === trees.counts.meshes), resolve));
    const firstAt = await first;
    const bestAt = await best;
    if (id !== runId) return null;
    const result: RunResult = {
        way,
        detail: gpuFirst ? `H-PLOC on the render device, adopted; tinybvh ${pool.size} workers` : `tinybvh, ${pool.size} workers`,
        firstMs: firstAt - started,
        bestMs: bestAt - started,
        worstFrameMs: worstGap,
        mainMs: null,
    };
    record(result);
    return result;
}

/** Way 2: the worker's device builds, the trees come back as arrays, this thread assembles and uploads them. */
async function runWorkerDevice(id: number, started: number): Promise<RunResult | null> {
    const builder = settings.builder;
    const built = answer('built', id);
    send({ type: 'build', id, builder, meshes: scene.meshes });
    const message = await built;
    if (id !== runId) {
        // Another run started meanwhile: these trees were never uploaded, nothing to free on the GPU.
        staleDropped++;
        return null;
    }
    const mainStart = performance.now();
    const data = assembleLayouts(message.layouts as GpuLayoutMeshBvh[], instances, scene.meshes);
    const buffers = uploadTraceScene(device, data, 'workers-gpu transferred');
    const mainMs = performance.now() - mainStart;
    const source = traceSourceOf(device, buffers);
    await viewFor(source);
    if (id !== runId) {
        buffers.destroy(); // never on screen
        staleDropped++;
        return null;
    }
    swapIn({ kind: 'uploaded', id, buffers, source, tree: sumInfo((message.layouts as GpuLayoutMeshBvh[]).map(layoutInfo)) });
    await nextFrames(1);
    const firstMs = performance.now() - started;
    overlay.build({ builtBy: `${message.builtBy} (worker device) ×${message.layouts.length}`, fallbackReason: message.fallbackReason, cpuMs: message.buildMs });
    overlay.set(
        'worker build',
        `${message.deviceMs ? `device ${message.deviceMs.toFixed(0)} ms, ` : ''}${message.prepareMs ? `pipelines ${message.prepareMs.toFixed(0)} ms, ` : ''}build ${message.buildMs.toFixed(0)} ms, read back ${message.readMs.toFixed(0)} ms, ${bytes(message.bytes)} transferred`,
    );
    const result: RunResult = { way: 'worker device', detail: `${message.builtBy} on the worker's device, ${bytes(message.bytes)} through the CPU`, firstMs, bestMs: null, worstFrameMs: worstGap, mainMs };
    record(result);
    lastBuilt = message.builtBy;
    return result;
}
let lastBuilt: string | null = null;

/** Way 3: the worker renders into a canvas of its own. */
async function runWorkerCanvas(id: number, started: number): Promise<RunResult | null> {
    const element = document.createElement('canvas');
    element.style.cssText = 'position:fixed;inset:0;width:100vw;height:100vh;display:block;touch-action:none';
    document.body.insertBefore(element, canvas.nextSibling);
    const ratio = Math.min(devicePixelRatio, 2);
    element.width = Math.max(1, Math.round(element.clientWidth * ratio));
    element.height = Math.max(1, Math.round(element.clientHeight * ratio));
    const own = new OrbitCamera(element);
    Object.assign(own, { target: [...camera.target], distance: camera.distance, yaw: camera.yaw, pitch: camera.pitch, near: camera.near, far: camera.far });
    const offscreen = element.transferControlToOffscreen();
    const first = answer('trees', id, 'first');
    const best = answer('trees', id, 'best');
    send({ type: 'render', id, canvas: offscreen, meshes: scene.meshes, instances: scene.instances, radius }, [offscreen]);
    // Nothing awaited since the run started, so it is still the latest: the worker builds the new scene before it stops the old one's frames.
    swapIn({ kind: 'worker canvas', id, element, camera: own, stats: null });
    await first;
    const firstMs = performance.now() - started;
    await best;
    const bestMs = performance.now() - started;
    if (id !== runId) return null;
    const result: RunResult = { way: 'worker canvas', detail: 'SceneTrees and the view in the worker; tinybvh from its own pool', firstMs, bestMs, worstFrameMs: worstGap, mainMs: 0 };
    record(result);
    return result;
}

/** Resolves with the time of the first frame on which `condition` holds. */
function watch(condition: () => boolean, resolve: (at: number) => void): void {
    const poll = () => (condition() ? resolve(performance.now()) : requestAnimationFrame(poll));
    poll();
}

// ---- Controls.

const waySelect = overlay.select('build', WAYS, settings.way, (value) => void run(value));
overlay.select('worker builder', ['gpu-hploc', 'gpu-lbvh'] as const, settings.builder, (value) => (settings.builder = value));
overlay.button('run again', () => void run(settings.way));
overlay.button('run all four', () => void runAll());
overlay.note(
    'first: every mesh traceable on this page; best: every mesh on tinybvh\'s tree. Two devices share no buffers (way 2 copies through the CPU), and a worker\'s device is the same GPU. In way 3 the main thread only sends the camera.',
);

async function runAll(): Promise<void> {
    for (const way of WAYS) {
        await run(way);
        await nextFrames(10);
    }
}

// ---- The frame: ways 1, 2 and 4 trace here; way 3 traces in the worker, and this thread sends it the camera.

const sun = vec3.normalize([0.45, 0.85, 0.3]);
let frame = 0;
let smoke: Promise<void> | null = null;
let activeView: RayCastView | null = null;
app.loop({
    async update() {
        frame++;
        const view = activeView;
        if (view) {
            const { measurement } = await view.collect();
            if (measurement?.gpuMs) overlay.rays('primary (GPU timestamps)', measurement.primaryRays / (measurement.gpuMs / 1000));
        }
        if (frame % 15 === 0) showStats();
        if (isSmoke && frame === 5) smoke ??= smokeRun();
    },
    render(encoder, target) {
        const on = current;
        if (!on) return;
        if (on.kind === 'worker canvas') {
            const ratio = Math.min(devicePixelRatio, 2);
            const width = Math.max(1, Math.round(on.element.clientWidth * ratio));
            const height = Math.max(1, Math.round(on.element.clientHeight * ratio));
            send({ type: 'camera', basis: on.camera.basis(width / height), width, height });
            activeView = null;
            return;
        }
        if (on.kind === 'trees') on.trees.update(encoder);
        const source = sourceOf(on)!;
        const view = views.get(`${source.triangleLeaves}/${source.positionLayout}/${source.nodeFormat}`)!;
        view.trace(encoder, source, camera.basis(canvas.width / canvas.height), { mode: 'material', shadows: true, sun, highlight: null, sceneRadius: radius }, canvas.width, canvas.height, { measure: frame % 15 === 0 });
        view.draw(encoder, target);
        activeView = view;
    },
});

function showStats(): void {
    const on = current;
    if (!on) return;
    if (on.kind === 'trees') {
        overlay.build(on.log.build(on.trees));
        overlay.tree(on.log.tree(on.trees));
        overlay.set('on screen', `${on.trees.counts.onGpuTrees} GPU trees, ${on.trees.counts.onTinybvhTrees} tinybvh trees, ${bytes(on.trees.memory.resident)} (${on.trees.nodeFormat} nodes)`);
    } else if (on.kind === 'uploaded') {
        overlay.tree(on.tree);
        overlay.set('on screen', `the worker's trees, ${bytes(on.buffers.byteLength)} (${on.source.nodeFormat} nodes)`);
    } else if (on.stats) {
        const counts = on.stats.counts;
        overlay.build({ builtBy: `in the worker: gpu ×${counts.gpuTreesBuilt} → tinybvh ×${counts.onTinybvhTrees}`, fallbackReason: null });
        overlay.tree({ triangles: scene.triangles, bytes: on.stats.resident });
        overlay.set('on screen', `the worker's canvas: ${counts.onGpuTrees} GPU trees, ${counts.onTinybvhTrees} tinybvh trees, ${bytes(on.stats.resident)}, ${on.stats.frames} frames${on.stats.mrays ? `, ${on.stats.mrays.toFixed(0)} Mrays/s` : ''}`);
    }
}

// ---- Smoke.

const tinybvh = await createTinyBvhBuilder();
const cpuLayouts = new Map<IndexedMesh, GpuLayoutMeshBvh>();
function cpuTracer() {
    const layouts = scene.meshes.map((mesh) => {
        if (!cpuLayouts.has(mesh)) cpuLayouts.set(mesh, tinybvh.buildLayout(mesh).layout);
        return cpuLayouts.get(mesh)!;
    });
    return createCpuTracer(assembleTraceScene(layouts, instances));
}

/** The view on screen against the CPU walk; way 3 reads its pixels in the worker. */
async function checkView(label: string): Promise<Record<string, unknown>> {
    await nextFrames(3);
    const on = current!;
    let agreement;
    if (on.kind === 'worker canvas') {
        const pixels = answer('pixels', on.id);
        send({ type: 'check', id: on.id });
        const message = await pixels;
        const { packed } = viewCheckRays(message.basis, message.width, message.height);
        agreement = checkAgainstCpu(packed, (index) => message.hits[index], cpuTracer(), { perturb: 1e-6 });
    } else {
        agreement = await checkViewAgainstCpu(activeView!, (aspect) => camera.basis(aspect), cpuTracer());
    }
    if (agreement.hits < 200) throw new Error(`${label}: only ${agreement.hits} of ${agreement.rays} check rays hit`);
    return { label, agreement };
}

/**
 * Smoke: (1) on the render device, the first trees came from the GPU (with
 * subgroups) and the view agrees with the CPU walk, then every mesh ends on
 * tinybvh's tree; (2) mid-run change: a worker-device build (LBVH) is
 * started and tinybvh-only is started before it returns, so its trees must
 * be dropped, never shown; tinybvh-only runs to its best trees and agrees;
 * (3) the worker device with H-PLOC: built by H-PLOC there (with
 * subgroups), uploaded here, agrees; (4) the worker canvas: first and best
 * trees reported by the worker, every mesh on tinybvh's tree there, and its
 * own pixels agree with the CPU walk here.
 */
async function smokeRun(): Promise<void> {
    const subgroups = device.features.has('subgroups');
    const checks: Record<string, unknown>[] = [];
    const rows: (RunResult | null)[] = [];

    rows.push(await run('render device'));
    const on = current;
    if (on?.kind !== 'trees') throw new Error('render device: no SceneTrees on screen');
    if (subgroups && on.trees.counts.gpuTreesBuilt === 0) throw new Error('render device: the GPU built no first trees');
    for (const handle of on.log.meshes()) if (on.trees.meshStatus(handle).builtBy !== 'tinybvh') throw new Error(`render device: mesh ${handle} did not end on tinybvh's tree`);
    checks.push(await checkView('render device'));

    settings.builder = 'gpu-lbvh';
    const dropped = run('worker device');
    await nextFrames(1);
    rows.push(await run('tinybvh only'));
    if ((await dropped) !== null) throw new Error('a worker build that came back after another run started was shown');
    if (staleDropped === 0) throw new Error('the interrupted worker build was not dropped');
    checks.push(await checkView('tinybvh only'));

    settings.builder = 'gpu-hploc';
    rows.push(await run('worker device'));
    if (subgroups && lastBuilt !== 'gpu-hploc') throw new Error(`worker device: built by ${lastBuilt}, not H-PLOC`);
    checks.push(await checkView('worker device (H-PLOC)'));

    rows.push(await run('worker canvas'));
    const canvasRun = current;
    if (canvasRun?.kind !== 'worker canvas') throw new Error('worker canvas: not on screen');
    while (!canvasRun.stats || canvasRun.stats.counts.onTinybvhTrees !== canvasRun.stats.counts.meshes) await nextFrames(1);
    checks.push(await checkView('worker canvas'));
    if (rows.some((row) => row === null)) throw new Error('a run of the smoke sequence was superseded');
    showStats();
    await nextFrames(10);
    await ready(device, { ...overlay.snapshot(), rows, checks, staleDropped });
}

function nextFrames(count: number): Promise<void> {
    const target = frame + count;
    return new Promise((resolve) => {
        const poll = () => (frame >= target ? resolve() : requestAnimationFrame(poll));
        poll();
    });
}

void run(settings.way);

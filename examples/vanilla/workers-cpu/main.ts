/**
 * vanilla/workers-cpu: tinybvh builds for many meshes, on the main thread,
 * in one worker, and in a pool of workers (`@pmndrs/bvh/worker`,
 * `createBvhWorkerPool`), in raw WebGPU. Every mesh spins every frame, so a
 * stalled main thread shows as a stutter and in the frame-time graph.
 *
 * - **The same pool API three ways:** `createBvhWorkerPool({ worker: 'never' })`
 *   builds on the calling thread, one mesh per turn (the main thread);
 *   `{ size: 1 }` is one worker; the default is one less than the cores, at
 *   most 4. Each worker is single-threaded (tinybvh's WASM has no threads),
 *   so nothing needs `SharedArrayBuffer` or cross-origin isolation.
 *   Parallelism is across meshes.
 * - **Each tree is placed as it arrives:** the pool is handed to `SceneTrees`
 *   (`workers`, GPU first trees off), which writes each tree into the scene's
 *   buffers as its build answers; the timeline (bottom) has a row per mesh:
 *   the pool's span from asked to answered and a tick when the tree was placed.
 * - **What the main thread paid:** per run, the time to every tree, the
 *   longest gap between frames while building, and the long tasks
 *   (`PerformanceObserver`, over 50 ms) in that time.
 *
 * Same gallery as `vanilla/fast-first` (`?count=`, `?detail=`, `?model=`, a drop).
 */

import { assembleTraceScene, createCpuTracer, createTinyBvhBuilder, SceneTrees, type GpuLayoutMeshBvh, type TraceSceneInstance } from '@pmndrs/bvh';
import { createBvhWorkerPool, defaultWorkerPoolSize, type BvhWorkerPool } from '@pmndrs/bvh/worker';
import { requestedModel } from '../../shared/assets.ts';
import { bytes } from '../../shared/overlay.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';
import { SceneTreesLog } from '../../shared/stats.ts';
import { createVanillaApp } from '../../shared/vanilla/app.ts';
import { checkViewAgainstCpu } from '../../shared/vanilla/check.ts';
import { galleryScene, type IndexedMesh, type VanillaScene } from '../../shared/vanilla/geometry.ts';
import { enableSceneDrop, loadGltfScene } from '../../shared/vanilla/gltf.ts';
import { mat4, vec3 } from '../../shared/vanilla/math.ts';
import { RayCastView, VIEW_MODES, type ViewMode } from '../../shared/vanilla/raycast-view.ts';
import { Timeline } from '../../shared/vanilla/timeline.ts';
import { watchPool } from '../../shared/vanilla/watched-pool.ts';

const app = await createVanillaApp('Workers (CPU)', 'vanilla/workers-cpu');
const { device, camera, overlay, canvas } = app;
const params = new URLSearchParams(location.search);

const MODES = ['main thread', 'one worker', 'worker pool'] as const;
type Mode = (typeof MODES)[number];
const poolFor = (mode: Mode): BvhWorkerPool =>
    mode === 'main thread' ? createBvhWorkerPool({ worker: 'never' }) : createBvhWorkerPool(mode === 'one worker' ? { size: 1 } : {});

const timeline = new Timeline(
    'per mesh',
    {
        built: { label: 'pool: queued + building', color: 'rgba(160, 170, 185, 0.55)', shape: 'span' },
        placed: { label: 'tree placed', color: '#ffc878', shape: 'mark' },
    },
    1,
);

// ---- Long tasks and frame gaps, per run.

let longTasks: { start: number; duration: number }[] = [];
try {
    new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) longTasks.push({ start: entry.startTime, duration: entry.duration });
    }).observe({ type: 'longtask', buffered: false });
} catch {
    // No long-task timing here (not Chrome): the frame gaps still say it.
}
let lastTick = performance.now();
let worstGap = 0;
const tick = (now: number) => {
    worstGap = Math.max(worstGap, now - lastTick);
    lastTick = now;
    requestAnimationFrame(tick);
};
requestAnimationFrame(tick);

// ---- One run: a SceneTrees fed by one mode's pool.

interface Spinner {
    readonly handle: number;
    readonly instanceIndex: number;
    readonly mesh: number;
    readonly at: [number, number, number];
    readonly phase: number;
    readonly speed: number;
    readonly matrix: Float32Array<ArrayBuffer>;
}
interface Run {
    readonly mode: Mode;
    readonly trees: SceneTrees;
    readonly pool: BvhWorkerPool;
    readonly log: SceneTreesLog;
    readonly lanes: Map<number, number>;
    readonly spinners: Spinner[];
    /** The scene this run built (a check compares against it, not the page's latest). */
    readonly target: VanillaScene;
    readonly started: number;
    live: boolean;
    /** Set when every mesh has its tree: the run's numbers. */
    result: RunResult | null;
}
interface RunResult {
    readonly mode: Mode;
    readonly workers: string;
    readonly allTreesMs: number;
    readonly worstFrameMs: number;
    readonly longTasks: number;
    readonly longTaskMs: number;
}

let current: Run | null = null;
let scene: VanillaScene;
const results = new Map<Mode, RunResult>();
const settings = { mode: 'worker pool' as Mode, spin: true, view: 'material' as ViewMode };
let time = 0;

async function createRun(target: VanillaScene, mode: Mode): Promise<Run> {
    let run: Run | null = null;
    const positions = new Map<Float32Array, number>(target.meshes.map((mesh, index) => [mesh.positions, index]));
    const pool = watchPool(poolFor(mode), (geometry, asked) => {
        const lane = positions.get(geometry.positions);
        if (run?.live && lane !== undefined) timeline.span(lane, 'built', asked);
    });
    const log = new SceneTreesLog();
    let trees: SceneTrees;
    try {
        trees = await SceneTrees.create(device, {
            label: 'workers-cpu',
            gpuFirstTrees: false,
            workers: pool,
            onTree: (event) => {
                log.onTree(event);
                const lane = event.mesh === null ? undefined : run?.lanes.get(event.mesh);
                if (run?.live && lane !== undefined) timeline.mark(lane, 'placed');
            },
        });
    } catch (error) {
        pool.dispose(); // the scene that would own it never came
        throw error;
    }
    run = { mode, trees, pool, log, lanes: new Map(), spinners: [], target, started: performance.now(), live: true, result: null };
    const handles = target.meshes.map((geometry, lane) => {
        const handle = trees.addMesh(geometry);
        run!.lanes.set(handle, lane);
        return handle;
    });
    const floor = target.name.startsWith('gallery') ? target.meshes.length - 1 : -1;
    target.instances.forEach((instance, index) => {
        const m = instance.objectToWorld;
        const spinner: Spinner = {
            handle: trees.addInstance({ mesh: handles[instance.mesh], objectToWorld: m, instanceIndex: index, material: instance.material }),
            instanceIndex: index,
            mesh: instance.mesh,
            at: [m[12], m[13], m[14]],
            phase: Math.atan2(m[8], m[0]),
            speed: instance.mesh === floor || !target.name.startsWith('gallery') ? 0 : 0.6 + (index % 5) * 0.25,
            matrix: new Float32Array(m),
        };
        run!.spinners.push(spinner);
    });
    return run;
}

// Runs start one after another (the select, the buttons, the smoke run).
let starts: Promise<void> = Promise.resolve();
function start(mode: Mode, target = scene): Promise<void> {
    const next = starts.then(() => startNow(mode, target));
    starts = next.catch(() => undefined);
    return next;
}

/** Build new, swap, then dispose the old (its builds still running, maybe): nothing recorded after the swap names its buffers. */
async function startNow(mode: Mode, target: VanillaScene): Promise<void> {
    settings.mode = modeSelect.value = mode;
    const next = await createRun(target, mode);
    // Only now, with the new scene made: a start that failed leaves the page on the scene it shows.
    scene = target;
    const old = current;
    current = next;
    if (old) {
        old.live = false;
        old.trees.dispose();
        old.pool.dispose();
    }
    timeline.reset(target.meshes.length, `per mesh: ${mode} (${next.pool.usesWorkers ? `${next.pool.size} worker${next.pool.size > 1 ? 's' : ''}` : 'on this thread'})`, next.started);
    longTasks = [];
    worstGap = 0;
    overlay.set('scene', `${target.name}: ${target.meshes.length.toLocaleString('en-US')} meshes, ${target.triangles.toLocaleString('en-US')} triangles`);
}

function finishRun(run: Run): void {
    const ms = performance.now() - run.started;
    const during = longTasks.filter((task) => task.start >= run.started);
    run.result = {
        mode: run.mode,
        workers: run.pool.usesWorkers ? String(run.pool.size) : 'none',
        allTreesMs: ms,
        worstFrameMs: worstGap,
        longTasks: during.length,
        longTaskMs: during.reduce((sum, task) => sum + task.duration, 0),
    };
    results.set(run.mode, run.result);
    for (const result of results.values()) {
        overlay.set(
            result.mode,
            `${result.allTreesMs.toFixed(0)} ms to every tree, worst frame ${result.worstFrameMs.toFixed(0)} ms, ${result.longTasks} long tasks (${result.longTaskMs.toFixed(0)} ms)`,
        );
    }
}

// ---- Controls.

const modeSelect = overlay.select('build on', MODES, settings.mode, (value) => void start(value));
overlay.button('build again', () => void start(settings.mode));
overlay.button('run all three', () => void runAll());
overlay.checkbox('spin', settings.spin, (value) => (settings.spin = value));
overlay.select('show', VIEW_MODES, settings.view, (value) => (settings.view = value));
overlay.note(
    `tinybvh, one mesh per build. "main thread" stalls a frame for each mesh; one worker keeps frames smooth but builds one mesh at a time; the pool (${defaultWorkerPoolSize()} here) builds several at once. No SharedArrayBuffer, no cross-origin isolation.`,
);

async function runAll(): Promise<void> {
    for (const mode of MODES) {
        await start(mode);
        await current!.trees.whenRefined();
        await nextFrames(10);
    }
}

// ---- Start.

const model = requestedModel();
const first = model ? await loadGltfScene(model) : galleryScene(Number(params.get('count') ?? 32), Number(params.get('detail') ?? 1));
camera.frame(first.bounds);
camera.distance *= first.name.startsWith('gallery') ? 0.7 : 1;
await start(settings.mode, first);
const view = RayCastView.create(device, app.format);
await view.prepare(current!.trees);
enableSceneDrop(
    async (dropped) => {
        camera.frame(dropped.bounds);
        await start(settings.mode, dropped);
    },
    (error) => overlay.error(String(error)),
);

const sun = vec3.normalize([0.45, 0.85, 0.3]);
let frame = 0;
let smoke: Promise<void> | null = null;
app.loop({
    async update(_now, delta) {
        frame++;
        if (settings.spin) time += delta;
        const { measurement } = await view.collect();
        if (measurement?.gpuMs) overlay.rays('primary (GPU timestamps)', measurement.primaryRays / (measurement.gpuMs / 1000));
        const run = current!;
        if (!run.result && run.trees.counts.onTinybvhTrees === run.trees.counts.meshes && run.trees.counts.meshes > 0) finishRun(run);
        timeline.draw();
        if (frame % 15 === 0) showStats();
        if (isSmoke && frame === 5) smoke ??= smokeRun();
    },
    render(encoder, target) {
        const run = current!;
        for (const spinner of run.spinners) {
            if (spinner.speed === 0) continue;
            mat4.compose(spinner.at, mat4.axisAngle([0, 1, 0], spinner.phase + time * spinner.speed), [1, 1, 1], spinner.matrix);
            run.trees.setTransform(spinner.handle, spinner.matrix);
        }
        run.trees.update(encoder);
        const radius = Math.hypot(scene.bounds[3] - scene.bounds[0], scene.bounds[4] - scene.bounds[1], scene.bounds[5] - scene.bounds[2]) / 2;
        view.trace(encoder, run.trees, camera.basis(canvas.width / canvas.height), { mode: settings.view, shadows: true, sun, highlight: null, sceneRadius: radius }, canvas.width, canvas.height, { measure: frame % 15 === 0 });
        view.draw(encoder, target);
    },
});

function showStats(): void {
    const run = current!;
    // GPU first trees are off on purpose here (this page is about tinybvh): not a fallback.
    overlay.build({ ...run.log.build(run.trees), fallbackReason: null });
    overlay.tree(run.log.tree(run.trees));
    const counts = run.trees.counts;
    overlay.set('trees placed', `${counts.onTinybvhTrees} of ${counts.meshes}${run.result ? '' : ` (${((performance.now() - run.started) / 1000).toFixed(1)} s)`}`);
    overlay.set('pool now', `${run.pool.stats.running} running, ${run.pool.stats.queued} queued, ${run.pool.stats.workers} workers`);
    overlay.set('memory', `${bytes(run.trees.memory.resident)} resident`);
}

// ---- Smoke.

const tinybvh = await createTinyBvhBuilder();
const cpuLayouts = new Map<IndexedMesh, GpuLayoutMeshBvh>();

/** Spin stopped: the view against the CPU walk over the same meshes at the same transforms (trees from tinybvh on this thread). */
async function checkView(): Promise<Record<string, unknown>> {
    const spin = settings.spin;
    settings.spin = false;
    await nextFrames(3);
    const run = current!;
    const layouts = run.target.meshes.map((mesh) => {
        if (!cpuLayouts.has(mesh)) cpuLayouts.set(mesh, tinybvh.buildLayout(mesh).layout);
        return cpuLayouts.get(mesh)!;
    });
    const instances: TraceSceneInstance[] = run.spinners.map((spinner) => ({ meshIndex: spinner.mesh, instanceIndex: spinner.instanceIndex, objectToWorld: spinner.matrix }));
    const cpu = createCpuTracer(assembleTraceScene(layouts, instances));
    await nextFrames(2);
    const agreement = await checkViewAgainstCpu(view, (aspect) => camera.basis(aspect), cpu);
    if (agreement.hits < 200) throw new Error(`only ${agreement.hits} of ${agreement.rays} check rays hit`);
    settings.spin = spin;
    return { mode: run.mode, agreement };
}

/** Every mesh of the run on its tinybvh tree, with a "placed" tick and a pool span on its timeline row. */
function checkComplete(run: Run): void {
    const counts = run.trees.counts;
    if (counts.onTinybvhTrees !== counts.meshes) throw new Error(`${run.mode}: ${counts.onTinybvhTrees} of ${counts.meshes} meshes on tinybvh trees`);
    for (const handle of run.lanes.keys()) {
        if (run.trees.meshStatus(handle).builtBy !== 'tinybvh') throw new Error(`${run.mode}: mesh ${handle} ends on ${run.trees.meshStatus(handle).builtBy}`);
    }
    const placed = timeline.count('placed');
    const built = timeline.count('built');
    const missing = [...run.lanes.values()].filter((lane) => placed[lane] !== 1 || built[lane] !== 1);
    if (missing.length) throw new Error(`${run.mode}: timeline rows ${missing.slice(0, 5).join(', ')} lack their one build span and one placed tick`);
    if (!run.result) throw new Error(`${run.mode}: no result recorded`);
}

/**
 * Smoke: (1) mid-run changes: the first run (the pool) switched to "main
 * thread" while its builds are in flight, and that run switched to "one
 * worker" once a few of its trees are in (each old scene and pool disposed
 * with builds running); (2) main thread, one worker and the pool each run to
 * the end: every mesh on its tinybvh tree, one span and one tick on each
 * timeline row, the worker modes really in workers; (3) the view agrees with
 * the CPU walk.
 */
async function smokeRun(): Promise<void> {
    const interrupted: string[] = [];
    if (!current!.result) {
        interrupted.push(`${current!.mode} at ${current!.trees.counts.onTinybvhTrees} trees`);
        await start('main thread');
    } else await start('main thread');
    while (current!.trees.counts.onTinybvhTrees < 3) await nextFrames(1);
    if (current!.result) throw new Error('the main-thread run finished before it could be interrupted');
    interrupted.push(`main thread at ${current!.trees.counts.onTinybvhTrees} trees`);
    const before = current;
    await start('one worker');
    if (current === before) throw new Error('the switch did not swap the scene');
    const runs: RunResult[] = [];
    for (const mode of MODES) {
        await start(mode);
        const run = current!;
        await run.trees.whenRefined();
        await nextFrames(5);
        checkComplete(run);
        if (mode !== 'main thread' && !run.pool.usesWorkers) throw new Error(`${mode}: the pool built inline`);
        runs.push(run.result!);
    }
    const check = await checkView();
    showStats();
    await nextFrames(10);
    await ready(device, { ...overlay.snapshot(), interrupted, runs, check });
}

function nextFrames(count: number): Promise<void> {
    const target = frame + count;
    return new Promise((resolve) => {
        const poll = () => (frame >= target ? resolve() : requestAnimationFrame(poll));
        poll();
    });
}

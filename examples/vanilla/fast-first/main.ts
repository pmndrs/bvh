/**
 * vanilla/fast-first: ddgi's load pattern as a generic recipe, in raw WebGPU.
 * Rays see every mesh within milliseconds on a GPU tree, and tinybvh's better
 * trees replace them mesh by mesh as the worker pool finishes them, while the
 * walk keeps tracing every frame.
 *
 * - **`SceneTrees`** does it all: GPU H-PLOC first trees in one batch at the
 *   first update, every mesh also built by tinybvh in the `/worker` pool
 *   (biggest first), each tinybvh tree swapped into the mesh's range as it
 *   arrives. `update(encoder)` runs in every frame's encoder before the trace.
 * - **The timeline** (bottom): one row per mesh. A grey bar from the moment
 *   the pool was asked to the moment it answered (queued, then building), a
 *   blue tick when the mesh's GPU tree was placed, an orange one when
 *   tinybvh's was. The pool is the page's own (`createBvhWorkerPool`, passed
 *   to `SceneTrees` as `workers`), wrapped to see those calls.
 * - **The view**, by material, paints each mesh by the tree it is on now:
 *   blue for the GPU's, orange for tinybvh's (the timeline's colours).
 * - **Controls:** hold tinybvh's trees back (`holdRefinements`) and release
 *   them all or half at once, turn GPU first trees off for the next load (then
 *   everything waits for tinybvh: the declared fallback without subgroups),
 *   add or remove a mesh mid-load, load again.
 * - **Load again** makes a new `SceneTrees` (and pool), swaps it in between
 *   frames, and only then disposes the old one, builds in flight and all.
 *
 * A gallery of 32 meshes by default (`?count=`, `?detail=`), a `?model=` glTF
 * (Sponza's 405 meshes show it best) or a dropped one.
 */

import { assembleTraceScene, createCpuTracer, createTinyBvhBuilder, SceneTrees, type GpuLayoutMeshBvh, type SceneTreeEvent, type TraceSceneInstance } from '@pmndrs/bvh';
import { createBvhWorkerPool, type BvhWorkerPool } from '@pmndrs/bvh/worker';
import { requestedModel } from '../../shared/assets.ts';
import { bytes } from '../../shared/overlay.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';
import { SceneTreesLog } from '../../shared/stats.ts';
import { createVanillaApp } from '../../shared/vanilla/app.ts';
import { watchPool } from '../../shared/vanilla/watched-pool.ts';
import { checkViewAgainstCpu } from '../../shared/vanilla/check.ts';
import { galleryScene, torusKnot, type IndexedMesh, type VanillaScene } from '../../shared/vanilla/geometry.ts';
import { enableSceneDrop, loadGltfScene } from '../../shared/vanilla/gltf.ts';
import { mat4, vec3 } from '../../shared/vanilla/math.ts';
import { RayCastView, VIEW_MODES, type ViewMode } from '../../shared/vanilla/raycast-view.ts';
import { hashColor, Timeline } from '../../shared/vanilla/timeline.ts';

const app = await createVanillaApp('Fast first', 'vanilla/fast-first');
const { device, camera, overlay, canvas } = app;
const params = new URLSearchParams(location.search);

/** Material numbers whose ray-cast colours are blue and orange (`hashColor`): the tree a mesh is on. */
const ON_GPU_TREE = 24;
const ON_TINYBVH_TREE = 134;
const timeline = new Timeline(
    'per mesh, from the load',
    {
        worker: { label: 'pool: queued + building', color: 'rgba(160, 170, 185, 0.55)', shape: 'span' },
        gpu: { label: 'GPU tree placed', color: hashColor(ON_GPU_TREE), shape: 'mark' },
        tinybvh: { label: 'tinybvh tree placed', color: hashColor(ON_TINYBVH_TREE), shape: 'mark' },
    },
    1,
);

// ---- One load: a SceneTrees over the scene, with a pool the page can see into.

interface InstanceEntry {
    readonly handle: number;
    readonly instanceIndex: number;
    readonly objectToWorld: Float32Array;
    /** Painted by the tree its mesh is on (a gallery's floor stays grey). */
    readonly paint: boolean;
}
interface MeshEntry {
    readonly lane: number;
    readonly handle: number;
    readonly geometry: IndexedMesh;
    readonly instances: InstanceEntry[];
}
interface Run {
    readonly trees: SceneTrees;
    readonly pool: BvhWorkerPool;
    readonly log: SceneTreesLog;
    readonly meshes: Map<number, MeshEntry>;
    readonly gpuFirst: boolean;
    /** A gallery's floor (never removed by the button), or -1. */
    floorLane: number;
    /** Tree events, applied before the next update (painting inside `onTree` would change the scene during its own update). */
    readonly pending: SceneTreeEvent[];
    /** Positions array → timeline lane, for the pool's calls. */
    readonly lanes: Map<Float32Array, number>;
    nextLane: number;
    nextInstance: number;
    live: boolean;
    framesTraced: number;
    /** Frames traced while some mesh had no tree yet or was still on its GPU tree. */
    framesTracedLoading: number;
}

let current: Run | null = null;
let scene: VanillaScene;
// Smoke runs start with tinybvh held back, so the first check sees GPU first trees whatever the machine's load.
const settings = { gpuFirst: true, hold: isSmoke, mode: 'material' as ViewMode, shadows: true };

async function createRun(target: VanillaScene): Promise<Run> {
    let run: Run | null = null;
    // The page's pool, wrapped: each build's call and answer go on the timeline, by the geometry's positions array.
    const pool = watchPool(createBvhWorkerPool(), (geometry, asked) => {
        const lane = run?.lanes.get(geometry.positions);
        if (run?.live && lane !== undefined) timeline.span(lane, 'worker', asked);
    });
    const log = new SceneTreesLog();
    const pending: SceneTreeEvent[] = [];
    let trees: SceneTrees;
    try {
        trees = await SceneTrees.create(device, {
            label: 'fast-first',
            gpuFirstTrees: settings.gpuFirst,
            holdRefinements: settings.hold,
            workers: pool,
            onTree: (event) => {
                log.onTree(event);
                pending.push(event);
            },
        });
    } catch (error) {
        pool.dispose(); // the scene that would own it never came
        throw error;
    }
    run = { trees, pool, log, meshes: new Map(), gpuFirst: settings.gpuFirst, pending, lanes: new Map(), floorLane: -1, nextLane: 0, nextInstance: 0, live: true, framesTraced: 0, framesTracedLoading: 0 };
    const handles = target.meshes.map((geometry) => addMesh(run!, geometry).handle);
    // A gallery's last mesh is its floor: grey whatever tree it is on.
    run.floorLane = target.name.startsWith('gallery') ? target.meshes.length - 1 : -1;
    for (const instance of target.instances) addInstance(run, run.meshes.get(handles[instance.mesh])!, instance.objectToWorld, instance.mesh !== run.floorLane);
    return run;
}

function addMesh(run: Run, geometry: IndexedMesh): MeshEntry {
    const lane = run.nextLane++;
    run.lanes.set(geometry.positions, lane);
    const entry: MeshEntry = { lane, handle: run.trees.addMesh(geometry), geometry, instances: [] };
    run.meshes.set(entry.handle, entry);
    if (run === current) timeline.grow(run.nextLane);
    return entry;
}

function addInstance(run: Run, mesh: MeshEntry, objectToWorld: Float32Array, paint: boolean): void {
    const instanceIndex = run.nextInstance++;
    mesh.instances.push({ handle: run.trees.addInstance({ mesh: mesh.handle, objectToWorld, instanceIndex }), instanceIndex, objectToWorld, paint });
}

function removeMesh(run: Run, mesh: MeshEntry): void {
    for (const instance of mesh.instances) run.trees.removeInstance(instance.handle);
    run.trees.removeMesh(mesh.handle);
    run.meshes.delete(mesh.handle);
}

// Loads run one after another (a drop, the button, the smoke run).
let loads: Promise<void> = Promise.resolve();
function load(target: VanillaScene): Promise<void> {
    const next = loads.then(() => loadNow(target));
    loads = next.catch(() => undefined);
    return next;
}

/** Build new, swap, then dispose the old: the next frame updates and traces the new scene, and nothing recorded after names the old one's buffers. */
async function loadNow(target: VanillaScene): Promise<void> {
    const next = await createRun(target);
    // Only now, with the new scene made: a load that failed leaves the page on the scene it shows.
    scene = target;
    const old = current;
    current = next;
    timeline.reset(next.nextLane, `per mesh, from the load (${target.meshes.length} meshes, ${settings.gpuFirst ? 'GPU first trees' : 'tinybvh only'})`);
    if (old) {
        old.live = false;
        // The old scene's tinybvh builds may still be running in its pool: dispose drops them.
        old.trees.dispose();
        old.pool.dispose();
    }
    overlay.set('scene', `${target.name}: ${target.meshes.length.toLocaleString('en-US')} meshes, ${target.instances.length.toLocaleString('en-US')} instances, ${target.triangles.toLocaleString('en-US')} triangles`);
    overlay.set('pool', `${next.pool.size} ${next.pool.usesWorkers ? 'workers' : '(inline: no Worker here)'}`);
}

/** Tree events since the last frame: on the timeline, and each mesh's instances painted by the tree it is on now. */
function applyEvents(run: Run): void {
    for (const event of run.pending.splice(0)) {
        if (event.mesh === null) continue;
        const mesh = run.meshes.get(event.mesh);
        if (!mesh) continue; // removed since
        timeline.mark(mesh.lane, event.builtBy);
        const material = event.builtBy === 'gpu' ? ON_GPU_TREE : ON_TINYBVH_TREE;
        for (const instance of mesh.instances) if (instance.paint) run.trees.setInstance(instance.handle, { material });
    }
}

// ---- Controls.

overlay.checkbox('GPU first trees (next load)', settings.gpuFirst, (value) => (settings.gpuFirst = value));
overlay.checkbox('hold tinybvh trees', settings.hold, (value) => {
    settings.hold = value;
    current?.trees.setHoldRefinements(value);
    if (!value) current?.trees.releaseRefinements();
});
overlay.button('release half', () => current?.trees.releaseRefinements((mesh) => (current!.meshes.get(mesh)?.lane ?? 0) % 2 === 0));
overlay.button('add a mesh', () => current && addKnot(current));
overlay.button('remove a mesh', () => {
    const run = current;
    const last = run && [...run.meshes.values()].filter((mesh) => mesh.lane !== run.floorLane).pop();
    if (run && last) removeMesh(run, last);
});
overlay.button('load again', () => void load(scene));
overlay.select('show', VIEW_MODES, settings.mode, (value) => (settings.mode = value));
overlay.checkbox('shadows (any hit)', settings.shadows, (value) => (settings.shadows = value));
overlay.note('By material: blue meshes are on their GPU H-PLOC tree, orange on tinybvh\'s. The timeline below has one row per mesh. Drop a glTF, or ?model=.');

function addKnot(run: Run): MeshEntry {
    const geometry = torusKnot(0.75, 0.2, 900, 40, 3, 7);
    const mesh = addMesh(run, geometry);
    const at = run.nextLane * 0.9;
    addInstance(run, mesh, mat4.compose([Math.cos(at) * 2, 3.6, Math.sin(at) * 2], [0, 0, 0, 1], [1, 1, 1]), true);
    return mesh;
}

// ---- Start.

const model = requestedModel();
const first = model ? await loadGltfScene(model) : galleryScene(Number(params.get('count') ?? 32), Number(params.get('detail') ?? 1));
camera.frame(first.bounds);
camera.distance *= first.name.startsWith('gallery') ? 0.7 : 1;
await load(first);
const view = RayCastView.create(device, app.format);
await view.prepare(current!.trees);
enableSceneDrop(
    async (dropped) => {
        camera.frame(dropped.bounds);
        await load(dropped);
    },
    (error) => overlay.error(String(error)),
);

const sun = vec3.normalize([0.45, 0.85, 0.3]);
let frame = 0;
let smoke: Promise<void> | null = null;
app.loop({
    async update() {
        frame++;
        const { measurement } = await view.collect();
        if (measurement?.gpuMs) overlay.rays('primary (GPU timestamps)', measurement.primaryRays / (measurement.gpuMs / 1000));
        timeline.draw();
        if (frame % 15 === 0) showStats();
        if (isSmoke && frame === 5) smoke ??= smokeRun();
    },
    render(encoder, target) {
        const run = current!;
        applyEvents(run);
        // Builds started, trees placed (GPU first, tinybvh swaps), the top level: recorded before the trace reads them.
        run.trees.update(encoder);
        const counts = run.trees.counts;
        run.framesTraced++;
        if (counts.waitingForFirstTree > 0 || counts.onGpuTrees > 0) run.framesTracedLoading++;
        const radius = Math.hypot(scene.bounds[3] - scene.bounds[0], scene.bounds[4] - scene.bounds[1], scene.bounds[5] - scene.bounds[2]) / 2;
        view.trace(encoder, run.trees, camera.basis(canvas.width / canvas.height), { mode: settings.mode, shadows: settings.shadows, sun, highlight: null, sceneRadius: radius }, canvas.width, canvas.height, { measure: frame % 15 === 0 });
        view.draw(encoder, target);
    },
});

function showStats(): void {
    const run = current!;
    overlay.build(run.log.build(run.trees));
    overlay.tree(run.log.tree(run.trees));
    const counts = run.trees.counts;
    const timing = run.trees.timeline;
    overlay.set('first trees (every mesh)', timing.firstTreesMs === null ? (run.gpuFirst ? '…' : 'off') : `${timing.firstTreesMs.toFixed(0)} ms`);
    overlay.set('tinybvh trees (every mesh)', timing.tinybvhTreesMs === null ? `${counts.onTinybvhTrees} of ${counts.meshes}` : `${timing.tinybvhTreesMs.toFixed(0)} ms`);
    overlay.set('swaps (GPU tree → tinybvh)', counts.swaps);
    overlay.set('frames traced while loading', run.framesTracedLoading);
    overlay.set('memory', `${bytes(run.trees.memory.resident)} resident, ${bytes(run.trees.memory.peak)} peak`);
}

// ---- Smoke.

const tinybvh = await createTinyBvhBuilder();
const cpuLayouts = new Map<IndexedMesh, GpuLayoutMeshBvh>();

/** The CPU walk over the run's live meshes and instances, trees from tinybvh on this thread: the hits any tree must give. */
async function checkView(label: string): Promise<Record<string, unknown>> {
    const run = current!;
    const meshes = [...run.meshes.values()];
    const layouts = meshes.map((mesh) => {
        if (!cpuLayouts.has(mesh.geometry)) cpuLayouts.set(mesh.geometry, tinybvh.buildLayout(mesh.geometry).layout);
        return cpuLayouts.get(mesh.geometry)!;
    });
    const instances: TraceSceneInstance[] = meshes.flatMap((mesh, meshIndex) => mesh.instances.map((instance) => ({ meshIndex, instanceIndex: instance.instanceIndex, objectToWorld: instance.objectToWorld })));
    const cpu = createCpuTracer(assembleTraceScene(layouts, instances));
    await nextFrames(3);
    const counts = run.trees.counts;
    const agreement = await checkViewAgainstCpu(view, (aspect) => camera.basis(aspect), cpu);
    if (agreement.hits < 200) throw new Error(`${label}: only ${agreement.hits} of ${agreement.rays} check rays hit`);
    return { label, onGpuTrees: counts.onGpuTrees, onTinybvhTrees: counts.onTinybvhTrees, agreement };
}

/**
 * Smoke: (1) tinybvh held back, every mesh's first tree placed, the GPU built
 * first trees (with subgroups), the view agrees with the CPU walk on the GPU
 * trees; (2) mid-load: half released, a held mesh removed, a mesh added, the
 * view agrees on the mixed scene; then "load again" while the old scene's
 * builds are still held and running, swapped in and the old one disposed;
 * (3) the new load finishes: every mesh ends on its tinybvh tree, each
 * mesh's timeline row has its tinybvh tick, frames were traced throughout,
 * and the view agrees again. Every frame runs in a validation error scope
 * (shared/vanilla/app.ts): a frame that used a destroyed buffer fails it.
 */
async function smokeRun(): Promise<void> {
    const subgroups = device.features.has('subgroups');
    let run = current!;
    await run.trees.whenBuilt();
    await nextFrames(3);
    if (subgroups && run.trees.counts.gpuTreesBuilt === 0) throw new Error('GPU first trees are on, but the GPU built none');
    const held = await checkView('first trees, tinybvh held back');

    run.trees.releaseRefinements((mesh) => (run.meshes.get(mesh)?.lane ?? 0) % 2 === 0);
    const odd = [...run.meshes.values()].find((mesh) => mesh.lane % 2 === 1 && mesh.lane !== run.floorLane);
    if (odd) removeMesh(run, odd);
    const added = addKnot(run);
    await run.trees.whenBuilt();
    await nextFrames(10);
    const mixed = await checkView('half released, a mesh removed and one added');
    if (!run.log.events.some((event) => event.mesh === added.handle)) throw new Error('the added mesh got no tree');

    // Load again with the old scene's other half still held (its builds running or done and waiting): it must go away cleanly.
    settings.hold = false;
    const reload = load(scene);
    await nextFrames(2);
    await reload;
    const old = run;
    run = current!;
    if (run === old) throw new Error('load again did not swap the scene');
    await run.trees.whenBuilt();
    await run.trees.whenRefined();
    await nextFrames(10);
    const counts = run.trees.counts;
    if (counts.onTinybvhTrees !== counts.meshes) throw new Error(`${counts.onTinybvhTrees} of ${counts.meshes} meshes on tinybvh trees after whenRefined`);
    for (const mesh of run.meshes.values()) {
        const status = run.trees.meshStatus(mesh.handle);
        if (status.builtBy !== 'tinybvh') throw new Error(`mesh ${mesh.handle} ends on ${status.builtBy}`);
    }
    const ticks = timeline.count('tinybvh');
    const missing = [...run.meshes.values()].filter((mesh) => ticks[mesh.lane] === 0).length;
    if (missing) throw new Error(`${missing} meshes have no tinybvh tick on the timeline`);
    if (run.framesTracedLoading === 0) throw new Error('no frame was traced while the trees were arriving');
    const final = await checkView('every mesh on its tinybvh tree');
    showStats();
    await nextFrames(10);
    await ready(device, { ...overlay.snapshot(), checks: [held, mixed, final], counts, timeline: run.trees.timeline, gpuTicks: timeline.count('gpu').reduce((a, b) => a + b, 0), framesTracedLoading: run.framesTracedLoading });
}

function nextFrames(count: number): Promise<void> {
    const target = frame + count;
    return new Promise((resolve) => {
        const poll = () => (frame >= target ? resolve() : requestAnimationFrame(poll));
        poll();
    });
}

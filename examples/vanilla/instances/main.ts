/**
 * vanilla/instances: `SceneTrees` with thousands of moving instances and
 * runtime adds and removals, in raw WebGPU.
 *
 * - **Fast first, then tinybvh:** on a device with subgroups every mesh's
 *   first tree comes from GPU H-PLOC in one batch, and tinybvh's trees from
 *   the `/worker` pool replace them mesh by mesh as they arrive (the overlay's
 *   builder line and the tree events). Without subgroups every mesh waits for
 *   tinybvh: the declared fallback, named in the overlay.
 * - **Moving:** every instance moves every frame (`setTransform`), and
 *   `SceneTrees.update(encoder)` refits the top level on the GPU in the
 *   frame's own encoder, before the ray-cast pass reads it. The mesh trees
 *   never change.
 * - **Runtime changes:** add and remove instances, add a new mesh (its GPU
 *   tree first, its tinybvh tree later) or remove one.
 * - **Picking:** the ray-cast view keeps each pixel's hit; the instance under
 *   the pointer is read back and tinted (no extra ray).
 *
 * The view is the shared ray-cast kernel (shared/vanilla/raycast-view.ts),
 * shown by instance by default.
 */

import { createRandom, SceneTrees, type MeshGeometry } from '@pmndrs/bvh';
import { GpuRays } from '../../shared/gpu-rays.ts';
import { GpuBracket } from '../../shared/gpu-timer.ts';
import { bytes } from '../../shared/overlay.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';
import { SceneTreesLog } from '../../shared/stats.ts';
import { createVanillaApp } from '../../shared/vanilla/app.ts';
import { box, sphere, torus, torusKnot } from '../../shared/vanilla/geometry.ts';
import { mat4, vec3, type Vec3 } from '../../shared/vanilla/math.ts';
import { RayCastView, VIEW_MODES, type ViewMode } from '../../shared/vanilla/raycast-view.ts';

const app = await createVanillaApp('Instances', 'vanilla/instances');
const { device, camera, overlay, canvas } = app;

const params = new URLSearchParams(location.search);
const COUNT = Number(params.get('count') ?? 4000);
const FIELD = 16;
const random = createRandom(7);
const randomDirection = (): Vec3 => {
    const z = random() * 2 - 1;
    const phi = random() * Math.PI * 2;
    const s = Math.sqrt(1 - z * z);
    return [Math.cos(phi) * s, z, Math.sin(phi) * s];
};

// ---- The scene: meshes, and instances that orbit the centre.

const log = new SceneTreesLog();
const trees = await SceneTrees.create(device, { label: 'instances', onTree: log.onTree });

interface MeshEntry {
    readonly handle: number;
    readonly name: string;
    /** The largest distance of a vertex from the mesh's origin. */
    readonly radius: number;
    readonly instances: Set<number>;
}
interface Instance {
    readonly handle: number;
    readonly mesh: MeshEntry;
    readonly axis: Vec3;
    readonly start: Vec3;
    readonly speed: number;
    readonly spin: Vec3;
    readonly scale: number;
    readonly matrix: Float32Array<ArrayBuffer>;
}
const meshes: MeshEntry[] = [];
const instances = new Map<number, Instance>();
const added: MeshEntry[] = [];

function addMesh(name: string, geometry: MeshGeometry): MeshEntry {
    let radius = 0;
    for (let i = 0; i < geometry.positions.length; i += 3) radius = Math.max(radius, Math.hypot(geometry.positions[i], geometry.positions[i + 1], geometry.positions[i + 2]));
    const entry = { handle: trees.addMesh(geometry), name, radius, instances: new Set<number>() };
    meshes.push(entry);
    return entry;
}

function addInstances(mesh: MeshEntry, count: number): void {
    for (let i = 0; i < count; i++) {
        const motion = {
            mesh,
            axis: randomDirection(),
            start: vec3.scale(randomDirection(), 4 + random() * FIELD),
            speed: (random() - 0.5) * 0.6,
            spin: randomDirection(),
            scale: 0.35 + random() * 0.4,
            matrix: new Float32Array(16),
        };
        place(motion, time);
        // The hit's `instance` is the handle (SceneTrees' default instanceIndex); the material number is the mesh's slot here.
        const handle = trees.addInstance({ mesh: mesh.handle, objectToWorld: motion.matrix, material: meshes.indexOf(mesh) });
        instances.set(handle, { handle, ...motion });
        mesh.instances.add(handle);
    }
}

function removeInstances(count: number): void {
    const handles = [...instances.keys()].slice(-count);
    for (const handle of handles) {
        trees.removeInstance(handle);
        instances.get(handle)!.mesh.instances.delete(handle);
        instances.delete(handle);
    }
}

/** Rotates `v` by the unit quaternion `q`. */
function rotate(q: readonly number[], v: Vec3): Vec3 {
    const u: Vec3 = [q[0], q[1], q[2]];
    const t = vec3.scale(vec3.cross(u, v), 2);
    return vec3.add(vec3.add(v, vec3.scale(t, q[3])), vec3.cross(u, t));
}

function place(motion: Omit<Instance, 'handle'>, at: number): void {
    const position = rotate(mat4.axisAngle(motion.axis, at * motion.speed), motion.start);
    mat4.compose(position, mat4.axisAngle(motion.spin, at * 1.3), [motion.scale, motion.scale, motion.scale], motion.matrix);
}

let time = 0;
let moving = true;
const knot = addMesh('torus knot', torusKnot(0.5, 0.15, 160, 16));
const ball = addMesh('sphere', sphere(0.5, 24, 12));
const cube = addMesh('box', box(0.8, 0.8, 0.8));
const ring = addMesh('torus', torus(0.45, 0.16, 12, 24));
for (const [index, mesh] of [knot, ball, cube, ring].entries()) addInstances(mesh, Math.floor(COUNT / 4) + (index === 0 ? COUNT % 4 : 0));

function submitUpdate(): void {
    const encoder = device.createCommandEncoder({ label: 'instances update' });
    trees.update(encoder);
    device.queue.submit([encoder.finish()]);
}
submitUpdate();
await trees.whenBuilt();
submitUpdate();

const view = RayCastView.create(device, app.format);
// A SceneTrees is a trace source as it is: its buffers and formats are read every frame.
await view.prepare(trees);
const rays = await GpuRays.create(trees);
camera.frame([-FIELD - 4, -FIELD - 4, -FIELD - 4, FIELD + 4, FIELD + 4, FIELD + 4]);
camera.distance *= 0.5;

// ---- Controls.

const settings = { mode: 'instance' as ViewMode, shadows: false };
overlay.select('show', VIEW_MODES, settings.mode, (value) => (settings.mode = value));
overlay.checkbox('shadows (any hit)', settings.shadows, (value) => (settings.shadows = value));
overlay.checkbox('move', moving, (value) => (moving = value));
overlay.button('+500 instances', () => addInstances(meshes[Math.floor(random() * meshes.length)], 500));
overlay.button('−500 instances', () => removeInstances(500));
overlay.button('add a mesh', addRandomMesh);
overlay.button('remove a mesh', removeAddedMesh);
overlay.note('Every instance moves every frame; SceneTrees refits the top level in the frame\'s encoder. A new mesh gets a GPU H-PLOC tree first, then tinybvh\'s from the worker pool. Hover to pick.');

const KNOTS: readonly [number, number][] = [
    [2, 5],
    [3, 4],
    [3, 5],
    [2, 7],
];
function addRandomMesh(): MeshEntry {
    const [p, q] = KNOTS[added.length % KNOTS.length];
    const mesh = addMesh(`knot (${p}, ${q}) #${added.length + 1}`, torusKnot(0.45, 0.1, 400, 24, p, q));
    added.push(mesh);
    addInstances(mesh, 300);
    return mesh;
}
function removeAddedMesh(): void {
    const mesh = added.pop();
    if (!mesh) return;
    for (const handle of mesh.instances) {
        trees.removeInstance(handle);
        instances.delete(handle);
    }
    trees.removeMesh(mesh.handle);
    meshes.splice(meshes.indexOf(mesh), 1);
}

let pointer: [number, number] | null = null;
canvas.addEventListener('pointermove', (event) => (pointer = [event.clientX / canvas.clientWidth, event.clientY / canvas.clientHeight]));
canvas.addEventListener('pointerleave', () => (pointer = null));
let highlight: number | null = null;

// ---- The frame.

const sun = vec3.normalize([0.45, 0.85, 0.3]);
let frame = 0;
let updateMs = 0;
// SceneTrees.update takes no timestamp writes: two tiny timed passes around it give its GPU time (the top-level refit, plus whatever else the update recorded).
const bracket = GpuBracket.create(device);
let updateGpuMs: number | null = null;
let lastEvents = 0;
let smoke: Promise<void> | null = null;
app.loop({
    async update(_now, delta) {
        frame++;
        const [{ measurement, pick }, gpuMs] = await Promise.all([view.collect(), bracket ? bracket.result() : Promise.resolve(null)]);
        if (gpuMs !== null) updateGpuMs = gpuMs;
        if (measurement?.gpuMs) overlay.rays('primary (GPU timestamps)', measurement.primaryRays / (measurement.gpuMs / 1000));
        if (pick !== undefined) {
            const instance = pick ? instances.get(pick.instance) : undefined;
            highlight = instance ? instance.handle : null;
            overlay.set('picked', instance ? `instance ${instance.handle} (${instance.mesh.name})` : '-');
        }
        if (moving) time += delta;
        if (frame % 15 === 0) showStats();
        if (isSmoke && frame === 30) smoke ??= smokeRun();
    },
    render(encoder, target) {
        const start = performance.now();
        if (moving) {
            for (const instance of instances.values()) {
                place(instance, time);
                trees.setTransform(instance.handle, instance.matrix);
            }
        }
        // Builds started, finished trees placed, the top level refit: recorded before the trace reads it.
        const timed = bracket !== null && !bracket.busy && frame % 15 === 0;
        if (timed) bracket.begin(encoder);
        trees.update(encoder);
        if (timed) bracket.end(encoder);
        updateMs = updateMs * 0.9 + (performance.now() - start) * 0.1;
        const width = canvas.width;
        const height = canvas.height;
        const pick = pointer ? ([Math.floor(pointer[0] * width), Math.floor(pointer[1] * height)] as const) : null;
        view.trace(encoder, trees, camera.basis(width / height), { mode: settings.mode, shadows: settings.shadows, sun, highlight, sceneRadius: FIELD }, width, height, { measure: frame % 15 === 0, pick });
        view.draw(encoder, target);
    },
});

function showStats(): void {
    overlay.build(log.build(trees));
    overlay.tree(log.tree(trees));
    const counts = trees.counts;
    overlay.set('meshes, instances', `${counts.meshes}, ${counts.instances.toLocaleString('en-US')}`);
    overlay.set('top level', trees.topLevelPath === 'gpu' ? 'GPU (refit each frame)' : 'CPU (rebuilt when something moved)');
    overlay.set('setTransform + update (CPU)', `${updateMs.toFixed(2)} ms per frame`);
    overlay.set('update (GPU, bracketed)', updateGpuMs === null ? (bracket ? '…' : 'no timestamp-query') : `${updateGpuMs.toFixed(3)} ms (the top-level refit of ${trees.counts.instances.toLocaleString('en-US')} instances)`);
    overlay.set('swaps (GPU tree → tinybvh)', counts.swaps);
    overlay.set('memory', `${bytes(trees.memory.resident)} resident, ${bytes(trees.memory.peak)} peak`);
    if (log.events.length !== lastEvents) {
        lastEvents = log.events.length;
        const event = log.events[log.events.length - 1];
        overlay.set('last tree', `mesh ${event.mesh}: ${event.builtBy}${event.replaced ? ` replaced ${event.replaced}` : ''}, ${event.triangleCount.toLocaleString('en-US')} triangles, ${event.ms.toFixed(0)} ms`);
    }
}

/**
 * Smoke: the GPU built first trees (with subgroups); then 500 instances
 * removed, a mesh added with 300, every tinybvh tree swapped in; then, frozen, 64 rays at
 * instances' current centres must hit a live instance close to where it is.
 */
async function smokeRun(): Promise<void> {
    // Whichever tree arrives first is placed: on a busy machine tinybvh's can beat the GPU's for small meshes
    // (the overlay then says so). What must hold is that the GPU built first trees at all.
    const gpuFirst = log.events.filter((event) => event.builtBy === 'gpu').length;
    if (trees.gpuFirstTrees.enabled && trees.counts.gpuTreesBuilt === 0) throw new Error('GPU first trees are on, but the GPU built none');
    removeInstances(500);
    const mesh = addRandomMesh();
    await nextFrames(5);
    await trees.whenBuilt();
    await trees.whenRefined();
    await nextFrames(5);
    const counts = trees.counts;
    if (counts.onTinybvhTrees !== counts.meshes) throw new Error(`${counts.onTinybvhTrees} of ${counts.meshes} meshes on tinybvh trees after whenRefined`);
    if (!log.events.some((event) => event.mesh === mesh.handle)) throw new Error('the added mesh has no tree event');

    moving = false;
    await nextFrames(3);
    const origin = camera.position;
    const live = [...instances.values()];
    const count = 64;
    const packed = new Float32Array(count * 8);
    for (let i = 0; i < count; i++) {
        const target = live[Math.floor(random() * live.length)];
        const centre: Vec3 = [target.matrix[12], target.matrix[13], target.matrix[14]];
        packed.set([...origin, 0, ...vec3.normalize(vec3.sub(centre, origin)), 1e30], i * 8);
    }
    const result = await rays.traceClosest(packed);
    let checked = 0;
    for (let i = 0; i < count; i++) {
        const hit = result.hit(i);
        if (!hit) continue;
        const instance = instances.get(hit.instance);
        if (!instance) throw new Error(`ray ${i} hit instance ${hit.instance}, which was removed`);
        const point = vec3.add(origin, vec3.scale([packed[i * 8 + 4], packed[i * 8 + 5], packed[i * 8 + 6]], hit.t));
        const distance = vec3.length(vec3.sub(point, [instance.matrix[12], instance.matrix[13], instance.matrix[14]]));
        if (distance > instance.mesh.radius * instance.scale * 1.001 + 1e-4) throw new Error(`ray ${i}: the hit is ${distance} from instance ${hit.instance}'s centre: a stale top level?`);
        checked++;
    }
    if (checked < count / 2) throw new Error(`only ${checked} of ${count} rays at instances hit`);
    moving = true;
    showStats();
    await nextFrames(10);
    await ready(device, { ...overlay.snapshot(), gpuFirstTrees: gpuFirst, events: log.events.length, picksChecked: checked, counts: trees.counts });
}

function nextFrames(count: number): Promise<void> {
    const target = frame + count;
    return new Promise((resolve) => {
        const poll = () => (frame >= target ? resolve() : requestAnimationFrame(poll));
        poll();
    });
}

if (isSmoke) pointer = [0.5, 0.5];

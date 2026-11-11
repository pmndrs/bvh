/**
 * three/instances: an `InstancedMesh` and a `BatchedMesh` with thousands of
 * moving instances.
 *
 * Every frame the instances move, `SceneTreesObject3DSync.update()` pushes
 * the new matrices (only objects whose `instanceMatrix` version changed), and
 * `SceneTrees.update(encoder)` refits the top level on the GPU. The mesh
 * trees never change: each geometry has one, shared by all its instances.
 *
 * The pointer is picked on the GPU (one ray through `TraceBatchKernel`): the
 * hit's `instance` goes back to the three object and instance id through
 * `sync.sourceOf`, and that instance is highlighted with `setColorAt`.
 */

import { BatchedMesh, BoxGeometry, Color, ConeGeometry, IcosahedronGeometry, InstancedMesh, Matrix4, MeshStandardMaterial, Quaternion, TorusGeometry, Vector2, Vector3, type Mesh } from 'three';
import { SceneTrees } from '@pmndrs/bvh';
import { SceneTreesObject3DSync } from '@pmndrs/bvh/three';
import { createApp } from '../../shared/app.ts';
import { cameraRay, GpuRays } from '../../shared/gpu-rays.ts';
import { SceneTreesLog } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

const app = await createApp('Instances', 'three/instances');
const { device, scene, camera, overlay } = app;
camera.position.set(0, 18, 38);

const params = new URLSearchParams(location.search);
const INSTANCED = Number(params.get('instanced') ?? 6000);
const BATCHED = Number(params.get('batched') ?? 3000);
const FIELD = 16;

// Each instance orbits its own axis around the centre.
interface Motion {
    readonly axis: Vector3;
    readonly start: Vector3;
    readonly speed: number;
    readonly spin: Vector3;
    readonly scale: number;
}
function motions(count: number): Motion[] {
    return Array.from({ length: count }, () => ({
        axis: new Vector3().randomDirection(),
        start: new Vector3().randomDirection().multiplyScalar(4 + Math.random() * FIELD),
        speed: (Math.random() - 0.5) * 0.6,
        spin: new Vector3().randomDirection(),
        scale: 0.35 + Math.random() * 0.4,
    }));
}
const base = new Color(0x8fb3ff);
const batchBase = new Color(0xb0e0a8);
const highlight = new Color(0xffc878);

const instanced = new InstancedMesh(new IcosahedronGeometry(0.6, 1), new MeshStandardMaterial({ roughness: 0.6 }), INSTANCED);
for (let i = 0; i < INSTANCED; i++) instanced.setColorAt(i, base);
const instancedMotion = motions(INSTANCED);

const geometries = [new BoxGeometry(0.8, 0.8, 0.8), new TorusGeometry(0.45, 0.16, 10, 20), new ConeGeometry(0.5, 1, 12)];
const vertexCount = geometries.reduce((sum, g) => sum + g.attributes.position.count, 0);
const indexCount = geometries.reduce((sum, g) => sum + (g.index?.count ?? 0), 0);
const batched = new BatchedMesh(BATCHED, vertexCount, indexCount, new MeshStandardMaterial({ roughness: 0.5 }));
const geometryIds = geometries.map((geometry) => batched.addGeometry(geometry));
for (let i = 0; i < BATCHED; i++) {
    batched.addInstance(geometryIds[i % geometryIds.length]);
    batched.setColorAt(i, batchBase);
}
const batchedMotion = motions(BATCHED);
scene.add(instanced, batched);

const matrix = new Matrix4();
const position = new Vector3();
const rotation = new Quaternion();
const spin = new Quaternion();
const scale = new Vector3();
function place(motion: Motion, time: number): Matrix4 {
    rotation.setFromAxisAngle(motion.axis, time * motion.speed);
    position.copy(motion.start).applyQuaternion(rotation);
    spin.setFromAxisAngle(motion.spin, time * 1.3);
    scale.setScalar(motion.scale);
    return matrix.compose(position, spin, scale);
}
let moving = true;
let time = 0;
function move(delta: number): void {
    if (moving) time += delta;
    for (let i = 0; i < INSTANCED; i++) instanced.setMatrixAt(i, place(instancedMotion[i], time));
    instanced.instanceMatrix.needsUpdate = true;
    for (let i = 0; i < BATCHED; i++) batched.setMatrixAt(i, place(batchedMotion[i], time));
}
move(0);
scene.updateMatrixWorld(true);

// The trees: one per geometry, the top level over every instance.
const log = new SceneTreesLog();
const trees = await SceneTrees.create(device, { label: 'instances', onTree: log.onTree });
const syncStart = performance.now();
const sync = new SceneTreesObject3DSync(trees, scene);
const syncMs = performance.now() - syncStart;
const rays = await GpuRays.create(trees);
function updateTrees(): number {
    const start = performance.now();
    scene.updateMatrixWorld();
    sync.update();
    const encoder = device.createCommandEncoder({ label: 'instances update' });
    trees.update(encoder);
    device.queue.submit([encoder.finish()]);
    return performance.now() - start;
}
updateTrees();
await trees.whenBuilt();
updateTrees();
overlay.build(log.build(trees));
overlay.tree(log.tree(trees));
overlay.set('instances', `${INSTANCED.toLocaleString('en-US')} instanced + ${BATCHED.toLocaleString('en-US')} batched`);
overlay.set('first sync', `${syncMs.toFixed(1)} ms`);
overlay.set('top level', trees.topLevelPath === 'gpu' ? 'GPU (refit each frame)' : 'CPU (rebuilt each frame)');
overlay.checkbox('move', moving, (value) => (moving = value));
overlay.note('Hover an instance: one GPU ray picks it, sync.sourceOf names it, setColorAt highlights it.');

// Picking.
const pointer = new Vector2(2, 2);
addEventListener('pointermove', (event) => pointer.set((event.clientX / innerWidth) * 2 - 1, -(event.clientY / innerHeight) * 2 + 1));
let picked: { object: Mesh; instanceId: number } | null = null;
function setHighlight(next: { object: Mesh; instanceId: number } | null): void {
    if (picked && (picked.object !== next?.object || picked.instanceId !== next?.instanceId)) {
        const object = picked.object as InstancedMesh | BatchedMesh;
        object.setColorAt(picked.instanceId, object === instanced ? base : batchBase);
        if (object === instanced) instanced.instanceColor!.needsUpdate = true;
    }
    picked = next;
    if (next) {
        const object = next.object as InstancedMesh | BatchedMesh;
        object.setColorAt(next.instanceId, highlight);
        if (object === instanced) instanced.instanceColor!.needsUpdate = true;
    }
}

/** Smoke check: rays at random instances' current centres must hit an instance whose current box holds the hit. */
async function checkPicks(): Promise<number> {
    const count = 64;
    const packed = new Float32Array(count * 8);
    const box = new Vector3();
    for (let i = 0; i < count; i++) {
        const id = Math.floor(Math.random() * INSTANCED);
        instanced.getMatrixAt(id, matrix);
        box.setFromMatrixPosition(matrix).project(camera);
        cameraRay(camera, new Vector2(box.x, box.y), packed, i);
    }
    const result = await rays.traceClosest(packed);
    let checked = 0;
    for (let i = 0; i < count; i++) {
        const hit = result.hit(i);
        if (!hit) continue;
        const source = sync.sourceOf(hit.instance);
        if (!source) throw new Error(`hit instance ${hit.instance} has no three source`);
        const point = new Vector3(packed[i * 8], packed[i * 8 + 1], packed[i * 8 + 2]).addScaledVector(new Vector3(packed[i * 8 + 4], packed[i * 8 + 5], packed[i * 8 + 6]), hit.t);
        const object = source.object as InstancedMesh | BatchedMesh;
        object.getMatrixAt(source.instanceId, matrix);
        const centre = new Vector3().setFromMatrixPosition(matrix);
        // Every shape fits in a sphere of radius 0.8 before its scale (at most 0.75).
        if (point.distanceTo(centre) > 0.8 * 0.75 + 1e-3) throw new Error(`ray ${i}: hit ${point.toArray()} is ${point.distanceTo(centre)} from instance ${source.instanceId}'s centre: a stale top level?`);
        checked++;
    }
    if (checked < count / 2) throw new Error(`only ${checked} of ${count} rays at instances hit`);
    return checked;
}

let frame = 0;
let updateMs = 0;
app.loop(async (_now, delta) => {
    frame++;
    move(delta);
    const ms = updateTrees();
    updateMs = updateMs * 0.9 + ms * 0.1;
    if (frame % 10 === 0) overlay.set('sync + refit (CPU)', `${updateMs.toFixed(2)} ms per frame`);
    if (frame % 30 === 0) overlay.build(log.build(trees));
    if (!rays.pending && pointer.x <= 1) {
        const result = await rays.traceClosest(cameraRay(camera, pointer));
        const hit = result.hit(0);
        const source = hit ? sync.sourceOf(hit.instance) : null;
        setHighlight(source ? { object: source.object, instanceId: source.instanceId } : null);
        overlay.set('picked', source ? `${source.object === instanced ? 'InstancedMesh' : 'BatchedMesh'} #${source.instanceId}` : '-');
        if (result.gpuMs !== null) overlay.set('pick GPU', `${result.gpuMs.toFixed(3)} ms`);
    }
    if (isSmoke && frame === 60) {
        const checked = await checkPicks();
        await ready(device, { ...overlay.snapshot(), picksChecked: checked });
    }
});

// For the smoke run's screenshot and the hint: a fixed pointer until the mouse moves.
if (isSmoke) pointer.set(0, 0);

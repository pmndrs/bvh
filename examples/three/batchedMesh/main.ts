/**
 * three/batchedMesh: a port of three-mesh-bvh's `batchedMesh` example.
 * Source: three-mesh-bvh's example/batchedMesh.js (v0.9.18, MIT, Garrett Johnson); see THIRD_PARTY.md.
 *
 * What it demonstrates: `computeBatchedBoundsTree` and `acceleratedRaycast` on
 * a `BatchedMesh`. Three geometries (two torus knots and a sphere, about
 * 100k triangles) are three instances of one `BatchedMesh`; each geometry gets
 * its own tree in `batchedMesh.boundsTrees`, and each of the spinning
 * raycasters casts a `firstHitOnly` ray through the instances every frame.
 * Turn "use bounds tree" off to see three's own `BatchedMesh.raycast`.
 *
 * The strategy and indirect options are three-mesh-bvh's and are accepted: here
 * every strategy builds the same tinybvh tree and the geometry is never
 * reordered (docs/three/bounds-tree.mdx, "Build options").
 *
 * The smoke run casts a seeded set of rays through this batch and through a
 * second batch with no trees, three's own raycast, and requires the same
 * nearest hit (distance, instance and triangle).
 */

import {
    BatchedMesh,
    CylinderGeometry,
    Fog,
    Mesh,
    MeshBasicMaterial,
    MeshPhongMaterial,
    Object3D,
    Raycaster,
    SphereGeometry,
    TorusKnotGeometry,
    Vector3,
    type Intersection,
    type Material,
} from 'three';
import { AVERAGE, CENTER, SAH, acceleratedRaycast, computeBatchedBoundsTree, disposeBatchedBoundsTree, prepareBoundsTrees, type BvhBoundsTree } from '@pmndrs/bvh/three';
import { createApp } from '../../shared/app.ts';
import { layoutInfo, sumInfo } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

const app = await createApp('BatchedMesh', 'three/batchedMesh');
const { device, scene, camera, overlay } = app;
await prepareBoundsTrees();
scene.background = null;

// three-mesh-bvh's setup for a BatchedMesh; three's own raycast is kept for the comparison.
const threeBatchedRaycast = BatchedMesh.prototype.raycast;
Mesh.prototype.raycast = acceleratedRaycast; // the stand-in mesh three's BatchedMesh.raycast casts against
BatchedMesh.prototype.raycast = acceleratedRaycast;
// three-mesh-bvh's types are in the project too, with their own signatures for these two.
(BatchedMesh.prototype as unknown as { computeBoundsTree: typeof computeBatchedBoundsTree }).computeBoundsTree = computeBatchedBoundsTree;
(BatchedMesh.prototype as unknown as { disposeBoundsTree: typeof disposeBatchedBoundsTree }).disposeBoundsTree = disposeBatchedBoundsTree;

const bgColor = 0xcfd8dc;
const lineColor = 0xd81b60;
const POINT_DIST = 23;
scene.fog = new Fog(bgColor, 40, 100);
camera.fov = 60;
camera.position.set(0, 0, 50);
camera.far = 100;
camera.updateProjectionMatrix();
app.controls.update();

const strategies = { CENTER, SAH, AVERAGE } as const;
const params = {
    raycasters: { count: isSmoke ? 20 : 150, near: 0, far: POINT_DIST },
    mesh: { splitStrategy: 'CENTER' as keyof typeof strategies, useBoundsTree: true, indirect: false },
};

const material = new MeshPhongMaterial({ color: 0x263238 });
const containerObj = new Object3D();
containerObj.scale.multiplyScalar(10);
scene.add(containerObj);

function createBatchedMesh(parent: Object3D | null, meshMaterial: Material): BatchedMesh {
    const radius = 0.5;
    const knot = new TorusKnotGeometry(radius, 0.2, 200, 100, 2, 3);
    const knot2 = new TorusKnotGeometry(radius, 0.2, 200, 100, 3, 4);
    const sphere = new SphereGeometry(radius, 100, 100);
    const geometries = [knot, knot2, sphere];
    const vertices = geometries.reduce((sum, g) => sum + g.attributes.position.count, 0);
    const indices = geometries.reduce((sum, g) => sum + g.index!.count, 0);
    const batched = new BatchedMesh(3, vertices, indices, meshMaterial);
    const ids = geometries.map((g) => batched.addGeometry(g));
    for (const id of ids) batched.addInstance(id);
    parent?.add(batched);
    return batched;
}

const batchedMesh = createBatchedMesh(containerObj, material);
batchedMesh.rotation.set(1.9, 4.2, 0);
const dolly = new Object3D();
const SPEEDS = [0.0003, 0.0009, 0.0005];

/** Instance i at x = -1.5, 0, 1.5, spinning at its own speed (the original's updateBatchedMeshInstances). */
function poseInstances(mesh: BatchedMesh, time: number): void {
    for (let i = 0; i < 3; i++) {
        dolly.position.set((i - 1) * 1.5, 0, 0);
        dolly.rotation.set(SPEEDS[i] * time, SPEEDS[i] * time, 0);
        dolly.updateMatrix();
        mesh.setMatrixAt(i, dolly.matrix);
    }
}
poseInstances(batchedMesh, 0);

// The raycasters: a point orbiting the batch at POINT_DIST, casting at the centre.
const raycaster = new Raycaster();
const sphereGeometry = new SphereGeometry(0.25, 20, 20);
const cylinderGeometry = new CylinderGeometry(0.01, 0.01);
interface Caster {
    update(): number;
    remove(): void;
}
const casters: Caster[] = [];
let castMs = 0;
let castCount = 0;

function addCaster(): void {
    const obj = new Object3D();
    const casterMaterial = new MeshBasicMaterial({ color: lineColor });
    const origin = new Mesh(sphereGeometry, casterMaterial);
    const hitMesh = new Mesh(sphereGeometry, casterMaterial);
    hitMesh.scale.multiplyScalar(0.25);
    origin.scale.multiplyScalar(0.5);
    const cylinder = new Mesh(cylinderGeometry, new MeshBasicMaterial({ color: lineColor, transparent: true, opacity: 0.5 }));
    obj.add(cylinder, origin, hitMesh);
    scene.add(obj);
    origin.position.set(POINT_DIST, 0, 0);
    const x = Math.random() * 10;
    const y = Math.random() * 10;
    const z = Math.random() * 10;
    const origVec = new Vector3();
    const dirVec = new Vector3();
    const xDir = Math.random() - 0.5;
    const yDir = Math.random() - 0.5;
    const zDir = Math.random() - 0.5;
    casters.push({
        update: () => {
            const time = performance.now();
            obj.rotation.set(xDir * 0.0001 * time + x, yDir * 0.0001 * time + y, zDir * 0.0001 * time + z);
            origin.updateMatrixWorld();
            origVec.setFromMatrixPosition(origin.matrixWorld);
            dirVec.copy(origVec).multiplyScalar(-1).normalize();
            raycaster.set(origVec, dirVec);
            raycaster.firstHitOnly = true;
            const start = performance.now();
            const res = raycaster.intersectObject(containerObj, true);
            castMs += performance.now() - start;
            castCount++;
            const length = res.length ? res[0].distance : POINT_DIST;
            hitMesh.position.set(POINT_DIST - length, 0, 0);
            const lineLength = res.length ? length - raycaster.near : length - raycaster.near - (POINT_DIST - raycaster.far);
            cylinder.position.set(POINT_DIST - raycaster.near - lineLength / 2, 0, 0);
            cylinder.scale.set(1, lineLength, 1);
            cylinder.rotation.z = Math.PI / 2;
            return res.length;
        },
        remove: () => scene.remove(obj),
    });
}

let treeInfo = '';
let strategyBuilt: string | null = null;
let indirectBuilt = false;

function updateFromOptions(): void {
    raycaster.near = params.raycasters.near;
    raycaster.far = params.raycasters.far;
    while (casters.length > params.raycasters.count) casters.pop()!.remove();
    while (casters.length < params.raycasters.count) addCaster();

    // Rebuild when the option that built the trees changed, or drop them.
    if ((!params.mesh.useBoundsTree && batchedMesh.boundsTrees) || (batchedMesh.boundsTrees && (strategyBuilt !== params.mesh.splitStrategy || indirectBuilt !== params.mesh.indirect))) {
        batchedMesh.disposeBoundsTree();
        (batchedMesh as unknown as { boundsTrees: null }).boundsTrees = null;
        treeInfo = '';
    }
    if (params.mesh.useBoundsTree && !batchedMesh.boundsTrees) {
        const start = performance.now();
        batchedMesh.computeBoundsTree(-1, { targetLeafSize: 5, strategy: strategies[params.mesh.splitStrategy], indirect: params.mesh.indirect });
        const ms = performance.now() - start;
        strategyBuilt = params.mesh.splitStrategy;
        indirectBuilt = params.mesh.indirect;
        const trees = (batchedMesh.boundsTrees as unknown as (BvhBoundsTree | null)[]).filter((t): t is BvhBoundsTree => t !== null);
        overlay.tree(sumInfo(trees.map((tree) => layoutInfo(tree.layout))));
        treeInfo = `${trees.length} trees, ${ms.toFixed(1)} ms`;
    }
    overlay.set('bounds trees', treeInfo || 'none (three\'s BatchedMesh.raycast)');
}

overlay.range('raycasters', 1, 1000, 1, params.raycasters.count, (v) => ((params.raycasters.count = v), updateFromOptions()));
overlay.range('near', 0, POINT_DIST, 0.1, params.raycasters.near, (v) => ((params.raycasters.near = v), updateFromOptions()));
overlay.range('far', 0, POINT_DIST, 0.1, params.raycasters.far, (v) => ((params.raycasters.far = v), updateFromOptions()));
overlay.checkbox('use bounds tree', params.mesh.useBoundsTree, (v) => ((params.mesh.useBoundsTree = v), updateFromOptions()));
overlay.checkbox('indirect', params.mesh.indirect, (v) => ((params.mesh.indirect = v), updateFromOptions()));
overlay.select('split strategy', Object.keys(strategies) as (keyof typeof strategies)[], params.mesh.splitStrategy, (v) => ((params.mesh.splitStrategy = v), updateFromOptions()));
overlay.note('Demonstrates computeBatchedBoundsTree + acceleratedRaycast on a BatchedMesh: a tree per geometry, rays through three instances. Switch "use bounds tree" off for three\'s own BatchedMesh.raycast. Strategy and indirect are accepted for parity; every strategy builds the same tree.');
updateFromOptions();

let frames = 0;
let smokeDone = false;
let rateStart = performance.now();
app.loop(async () => {
    const time = performance.now();
    containerObj.rotation.set(0.0001 * time, 0.0001 * time, 0);
    containerObj.updateMatrixWorld();
    poseInstances(batchedMesh, time);
    for (const caster of casters) caster.update();
    frames++;
    if (time - rateStart > 1000 && castMs > 0) {
        overlay.rays(params.mesh.useBoundsTree ? 'acceleratedRaycast (CPU)' : 'three BatchedMesh.raycast (CPU)', (castCount / castMs) * 1000);
        castMs = 0;
        castCount = 0;
        rateStart = time;
    }
    if (isSmoke && frames === 3 && !smokeDone) {
        smokeDone = true;
        await smoke();
    }
});

/** Seeded rays through the batch with its trees, and through a second batch with none (three's raycast). */
async function smoke(): Promise<void> {
    if (!batchedMesh.boundsTrees) throw new Error('no bounds trees to check');
    const plain = createBatchedMesh(null, material);
    plain.rotation.copy(batchedMesh.rotation);
    plain.updateMatrixWorld(true);
    poseInstances(batchedMesh, 1234);
    poseInstances(plain, 1234);
    containerObj.rotation.set(0.12, 0.34, 0);
    containerObj.updateMatrixWorld(true);
    plain.matrixWorld.copy(batchedMesh.matrixWorld);
    plain.matrixWorldAutoUpdate = false;
    let seed = 99;
    const random = (): number => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
    const rays = 200;
    let hits = 0;
    let ours = 0;
    let theirs = 0;
    const origin = new Vector3();
    const direction = new Vector3();
    raycaster.near = 0;
    raycaster.far = Infinity;
    raycaster.firstHitOnly = true;
    for (let i = 0; i < rays; i++) {
        origin.set(random() - 0.5, random() - 0.5, random() - 0.5).normalize().multiplyScalar(POINT_DIST);
        // Aim at the centre, give or take.
        direction.copy(origin).multiplyScalar(-1).add(new Vector3((random() - 0.5) * 10, (random() - 0.5) * 10, (random() - 0.5) * 10)).normalize();
        raycaster.set(origin, direction);
        const a: Intersection[] = [];
        const b: Intersection[] = [];
        const t0 = performance.now();
        acceleratedRaycast.call(batchedMesh as never, raycaster, a);
        const t1 = performance.now();
        threeBatchedRaycast.call(plain, raycaster, b);
        const t2 = performance.now();
        ours += t1 - t0;
        theirs += t2 - t1;
        a.sort((x, y) => x.distance - y.distance);
        b.sort((x, y) => x.distance - y.distance);
        if (a.length === 0 !== (b.length === 0)) throw new Error(`ray ${i}: accelerated ${a.length} hits, three ${b.length}`);
        if (a[0] && b[0]) {
            hits++;
            const tolerance = 1e-4 * Math.max(1, b[0].distance);
            if (Math.abs(a[0].distance - b[0].distance) > tolerance) throw new Error(`ray ${i}: distance ${a[0].distance} vs three's ${b[0].distance}`);
            if (a[0].batchId !== b[0].batchId) throw new Error(`ray ${i}: instance ${a[0].batchId} vs three's ${b[0].batchId}`);
            if (a[0].faceIndex !== b[0].faceIndex) throw new Error(`ray ${i}: triangle ${a[0].faceIndex} vs three's ${b[0].faceIndex}`);
        }
    }
    if (hits < rays / 4) throw new Error(`only ${hits} of ${rays} rays hit the batch`);
    overlay.set('smoke', `${hits}/${rays} rays hit; same distance, instance and triangle as three's; ${((rays / ours) * 1000).toFixed(0)} vs ${((rays / theirs) * 1000).toFixed(0)} rays/s`);
    await ready(device, { ...overlay.snapshot(), rays, hits, acceleratedRaysPerSecond: (rays / ours) * 1000, threeRaysPerSecond: (rays / theirs) * 1000 });
}

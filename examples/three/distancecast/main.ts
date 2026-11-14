/**
 * three/distancecast: three-mesh-bvh's `distancecast` example on our tree.
 *
 * A noise terrain (250 × 250 vertices) with a tree, and a cylinder with its
 * own tree that you move with the gizmo. Every move asks
 * `terrain.geometry.boundsTree.closestPointToGeometry(cylinder, matrix, ...)`
 * (tree against tree): the two closest points, joined by a line, and the
 * cylinder turns red within `distance` of the terrain. A marching-cubes volume
 * shows where the field is within `distance`: each cell is one
 * `closestPointToPoint(point, target, distance, distance)` (with both
 * thresholds, as the original: any triangle within the distance ends the
 * search), filled a few milliseconds per frame, around the terrain's centre or
 * around the cylinder. The calls and arguments are three-mesh-bvh's; the
 * queries run over our CPU tree.
 *
 * Ported from three-mesh-bvh's example/distancecast.js (v0.9.18, MIT, Garrett
 * Johnson); see THIRD_PARTY.md. Changed: our shell and overlay, three's
 * `SimplexNoise` (seeded) for `simplex-noise`, no shadows, the smoke check.
 */

import { BoxGeometry, CylinderGeometry, DoubleSide, Group, Matrix4, Mesh, MeshBasicMaterial, MeshStandardMaterial, Object3D, PlaneGeometry, SphereGeometry, Vector3 } from 'three';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { SimplexNoise } from 'three/addons/math/SimplexNoise.js';
import { MarchingCubes } from 'three/addons/objects/MarchingCubes.js';
import { MeshBVH } from 'three-mesh-bvh';
import { computeBoundsTree, MeshBVHHelper, prepareBoundsTrees, type BvhBoundsTree } from '@pmndrs/bvh/three';
import { createApp } from '../../shared/app.ts';
import { layoutInfo } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

await prepareBoundsTrees();
const app = await createApp('Distancecast', 'three/distancecast');
const { renderer, scene, camera, controls, overlay } = app;
camera.position.set(0, 2, 5);
controls.target.set(0, 0, 0);

const volume = { display: 'distance to terrain' as 'hide' | 'distance to terrain' | 'distance to mesh', hideWhileGenerating: false, alwaysShowDistance: true, surfaceOnly: false, distance: 1, resolution: isSmoke ? 30 : 100, radius: 4 };

// The terrain: seeded noise, so every load (and the smoke check) is the same.
let seed = 11;
const random = { random: () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646 };
const noise = new SimplexNoise(random);
const SIZE = 50;
const DIM = 250;
const planeGeometry = new PlaneGeometry(SIZE, SIZE, DIM - 1, DIM - 1);
const position = planeGeometry.attributes.position;
for (let i = 0; i < DIM * DIM; i++) position.setZ(i, noise.noise(position.getX(i) / 15, position.getY(i) / 15) * 3);
planeGeometry.computeVertexNormals();
const buildStart = performance.now();
const terrainTree = computeBoundsTree.call(planeGeometry);
overlay.build({ builtBy: 'tinybvh (computeBoundsTree)', cpuMs: performance.now() - buildStart });
overlay.tree(layoutInfo(terrainTree.layout));
const terrain = new Mesh(planeGeometry, new MeshStandardMaterial({ color: 0xffffff, metalness: 0.1, roughness: 0.9, side: DoubleSide }));
terrain.rotation.x = -Math.PI / 2;
terrain.position.y = -3;
scene.add(terrain);

// The target: a cylinder with its own tree, in a container the gizmo moves.
const shapeMaterial = new MeshStandardMaterial({ roughness: 0.75, metalness: 0.1 });
const targetContainer = new Group();
scene.add(targetContainer);
const targetMesh = new Mesh(new CylinderGeometry(0.5, 0.25, 1, 20, 1), shapeMaterial);
const targetTree: BvhBoundsTree = computeBoundsTree.call(targetMesh.geometry);
targetMesh.geometry.computeBoundingSphere();
targetContainer.add(targetMesh);

const gizmo = new TransformControls(camera, renderer.domElement);
gizmo.attach(targetContainer);
let needsUpdate = false;
let regenerate = true;
gizmo.addEventListener('dragging-changed', (event) => {
    controls.enabled = !event.value;
    if (!event.value) needsUpdate = true;
});
gizmo.addEventListener('objectChange', () => (needsUpdate = true));
scene.add(gizmo.getHelper());
addEventListener('keydown', (event) => {
    if (event.key === 'w') gizmo.setMode('translate');
    if (event.key === 'e') gizmo.setMode('rotate');
});

// The volume: marching cubes, isolation 0.5 over cells set to 0 (within the distance) or 1.
const cubeMaterial = new MeshStandardMaterial({ flatShading: true, color: 0xe91e63, metalness: 0, roughness: 0.35, transparent: true, depthWrite: false, opacity: 0.15, premultipliedAlpha: true, side: DoubleSide });
// Its buffers hold maxPolyCount triangles and are uploaded whole on each update(): a smoke run (software WebGPU) keeps
// them small and updates them once, at the end.
const marchingCubes = new MarchingCubes(100, cubeMaterial, false, false, isSmoke ? 60_000 : 1_000_000);
marchingCubes.isolation = 0;
const marchingCubesContainer = new Group();
marchingCubesContainer.scale.multiplyScalar(5);
marchingCubesContainer.add(marchingCubes);
scene.add(marchingCubesContainer);

// The closest points and the line between them.
const sphere1 = new Mesh(new SphereGeometry(0.025, 20, 20), new MeshBasicMaterial({ color: 0xe91e63 }));
const sphere2 = sphere1.clone();
const lineCube = new Mesh(new BoxGeometry(), sphere1.material);
lineCube.position.z = 0.5;
const line = new Object3D();
line.add(lineCube);
scene.add(sphere1, sphere2, line);
scene.updateMatrixWorld(true);

const helper = new MeshBVHHelper(terrain, 10);
helper.visible = false;
scene.add(helper);

overlay.checkbox('display BVH', false, (value) => (helper.visible = value));
overlay.range('BVH depth', 1, 40, 1, 10, (value) => {
    helper.depth = value;
    helper.update();
});
overlay.select('volume', ['hide', 'distance to terrain', 'distance to mesh'] as const, volume.display, (value) => {
    volume.display = value;
    regenerate = true;
});
overlay.checkbox('hide while generating', volume.hideWhileGenerating, (value) => (volume.hideWhileGenerating = value));
overlay.checkbox('always show distance', volume.alwaysShowDistance, (value) => (volume.alwaysShowDistance = value));
overlay.checkbox('surface only', volume.surfaceOnly, (value) => {
    volume.surfaceOnly = value;
    regenerate = true;
});
overlay.range('distance', 0, 2, 0.01, volume.distance, (value) => {
    volume.distance = value;
    regenerate = true;
});
overlay.range('radius', 1, 20, 0.1, volume.radius, (value) => {
    volume.radius = value;
    regenerate = true;
});
overlay.range('resolution', 5, 200, 1, volume.resolution, (value) => {
    volume.resolution = value;
    regenerate = true;
});
overlay.select('gizmo', ['translate', 'rotate'] as const, 'translate', (value) => gizmo.setMode(value));
overlay.note('Move the cylinder with the gizmo (w: move, e: rotate): closestPointToGeometry joins it to the terrain. The volume is closestPointToPoint per cell, a few ms per frame.');

const targetToBvh = new Matrix4();
const distanceResult1: { point?: Vector3; distance?: number } = {};
const distanceResult2: { point?: Vector3; distance?: number } = {};
let lastQueryMs = 0;
function updateDistanceCheck(fastCheck = false): boolean {
    targetContainer.updateMatrixWorld();
    targetToBvh.copy(terrain.matrixWorld).invert().multiply(targetContainer.matrixWorld);
    const maxDistance = volume.distance;
    const minDistance = fastCheck ? volume.distance : 0;
    const start = performance.now();
    const found = !!terrainTree.closestPointToGeometry(targetMesh.geometry, targetToBvh, distanceResult1, distanceResult2, minDistance, maxDistance);
    lastQueryMs = performance.now() - start;
    const hit = found && distanceResult1.distance! < volume.distance;
    if (hit) {
        // The points are in each geometry's own frame.
        sphere1.position.copy(distanceResult1.point!).applyMatrix4(terrain.matrixWorld);
        sphere2.position.copy(distanceResult2.point!).applyMatrix4(targetToBvh).applyMatrix4(terrain.matrixWorld);
        line.position.copy(sphere1.position);
        line.lookAt(sphere2.position);
        line.scale.set(0.01, 0.01, sphere1.position.distanceTo(sphere2.position));
    }
    shapeMaterial.color.set(hit ? 0xe91e63 : 0x666666);
    shapeMaterial.emissive.set(0xe91e63).multiplyScalar(hit ? 0.25 : 0);
    line.visible = sphere1.visible = sphere2.visible = hit && !fastCheck;
    return hit;
}

let cells = 0;
let volumeMs = 0;
function* updateMarchingCubes(): Generator<number> {
    if (volume.display === 'hide') return;
    updateDistanceCheck();
    const toTarget = volume.display === 'distance to mesh';
    const { surfaceOnly, resolution, distance, radius } = volume;
    let fieldSize: number;
    let worldToBvh: Matrix4;
    let tree: BvhBoundsTree;
    if (toTarget) {
        targetContainer.add(marchingCubesContainer);
        worldToBvh = new Matrix4();
        fieldSize = (targetMesh.geometry.boundingSphere!.radius + distance) * 2;
        tree = targetTree;
    } else {
        scene.add(marchingCubesContainer);
        fieldSize = (radius + (4 * radius) / resolution) * 2;
        worldToBvh = terrain.matrixWorld.clone().invert();
        tree = terrainTree;
    }
    marchingCubesContainer.scale.setScalar(fieldSize / 2);
    marchingCubesContainer.updateMatrixWorld();
    marchingCubes.init(resolution);
    marchingCubes.updateMatrixWorld();
    // The cubes span -1 to 1 in their own frame.
    const dim = marchingCubes.matrixWorld.getMaxScaleOnAxis();
    const min = -dim;
    const size = marchingCubes.size;
    const cellWidth = (2 * dim) / size;
    marchingCubes.isolation = 0.5;
    marchingCubes.position.setScalar(1 / size);
    marchingCubes.reset();
    const point = new Vector3();
    const result: { distance?: number } = {};
    const start = performance.now();
    cells = 0;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            for (let z = 0; z < size; z++) {
                point.set(min + cellWidth / 2 + x * cellWidth, min + cellWidth / 2 + y * cellWidth, min + cellWidth / 2 + z * cellWidth);
                if (toTarget || surfaceOnly || point.length() < radius) {
                    point.applyMatrix4(worldToBvh);
                    const found = !!tree.closestPointToPoint(point, result, distance, distance);
                    marchingCubes.setCell(x, y, z, found && result.distance! < distance ? 0 : 1);
                    cells++;
                }
                yield (y * size * size + x * size + z + 1) / (size * size * size);
            }
        }
    }
    volumeMs = performance.now() - start;
    marchingCubes.blur(1);
}

let task: Generator<number> | null = null;
let percentage = 0;

/** Smoke check: closest points against three-mesh-bvh's on the same geometry (points and the cylinder at several places). */
function checkAgainstThreeMeshBvh(): { points: number; geometries: number; near: number } {
    const theirTerrain = new MeshBVH(planeGeometry.clone());
    const theirTarget = targetMesh.geometry.clone();
    (theirTarget as unknown as { boundsTree: MeshBVH }).boundsTree = new MeshBVH(theirTarget);
    let s = 3;
    const r = () => ((s = (s * 16807) % 2147483647) - 1) / 2147483646;
    const point = new Vector3();
    const ours: { point?: Vector3; distance?: number } = {};
    const theirs: { point: Vector3; distance: number } = { point: new Vector3(), distance: 0 };
    const close = (a: number, b: number) => Math.abs(a - b) <= 1e-5 * Math.max(1, a, b);
    let near = 0;
    for (let i = 0; i < 400; i++) {
        point.set((r() - 0.5) * 40, (r() - 0.5) * 40, (r() - 0.5) * 8);
        const a = terrainTree.closestPointToPoint(point, ours);
        const b = theirTerrain.closestPointToPoint(point, theirs as never);
        if (!a || !b || !close(ours.distance!, theirs.distance)) throw new Error(`point ${i}: ours ${ours.distance}, three-mesh-bvh ${theirs.distance}`);
        // With thresholds (the volume's call), as the example reads it: found and nearer than the distance.
        // (three-mesh-bvh may also return a point beyond maxThreshold, from a leaf whose box was nearer; we return null there.)
        const within = (found: unknown, target: { distance?: number }) => !!found && target.distance! < 1;
        const nearOurs: { distance?: number } = {};
        const nearTheirs: { distance?: number } = {};
        const foundOurs = within(terrainTree.closestPointToPoint(point, nearOurs, 1, 1), nearOurs);
        const foundTheirs = within(theirTerrain.closestPointToPoint(point, nearTheirs as never, 1, 1), nearTheirs);
        if (Math.abs(ours.distance! - 1) > 1e-4 && foundOurs !== foundTheirs) throw new Error(`point ${i} at ${ours.distance}: within 1, ours ${foundOurs}, three-mesh-bvh ${foundTheirs}`);
    }
    const matrix = new Matrix4();
    const t1: { point: Vector3; distance: number } = { point: new Vector3(), distance: 0 };
    const t2: { point: Vector3; distance: number } = { point: new Vector3(), distance: 0 };
    // The cylinder near the terrain, asked as the example asks (thresholds 0 and the distance, 1): an unbounded
    // tree-against-tree search from far away walks most of both trees, which is not what the page does.
    let placements = 0;
    for (let i = 0; i < 40; i++) {
        matrix.makeRotationFromEuler(targetContainer.rotation.set(r() * 6, r() * 6, r() * 6));
        const x = (r() - 0.5) * 30;
        const z = (r() - 0.5) * 30;
        // Just above the terrain there (its height in the terrain's frame is the noise; the frame is turned -90° about x).
        matrix.setPosition(x, -z, noise.noise(x / 15, -z / 15) * 3 + 0.2 + r() * 1.2);
        const a = terrainTree.closestPointToGeometry(targetMesh.geometry, matrix, distanceResult1, distanceResult2, 0, 1);
        const b = theirTerrain.closestPointToGeometry(theirTarget, matrix, t1 as never, t2 as never, 0, 1);
        const oursWithin = !!a && distanceResult1.distance! < 1;
        const theirsWithin = !!b && t1.distance < 1;
        if (oursWithin !== theirsWithin || (oursWithin && !close(distanceResult1.distance!, t1.distance))) throw new Error(`placement ${i}: ours ${oursWithin ? distanceResult1.distance : 'none within 1'}, three-mesh-bvh ${theirsWithin ? t1.distance : 'none within 1'}`);
        if (oursWithin) near++;
        placements++;
    }
    targetContainer.rotation.set(0, 0, 0);
    return { points: 400, geometries: placements, near };
}

let frame = 0;
let checked = false;
app.loop(() => {
    frame++;
    if (needsUpdate) {
        updateDistanceCheck(!volume.alwaysShowDistance);
        needsUpdate = false;
    }
    if (regenerate) {
        task = updateMarchingCubes();
        regenerate = false;
    }
    if (task) {
        const start = performance.now();
        while (performance.now() - start < 15) {
            const next = task.next();
            if (next.done) {
                task = null;
                break;
            }
            percentage = next.value;
        }
        if (!isSmoke || !task) marchingCubes.update();
        marchingCubes.visible = !volume.hideWhileGenerating && volume.display !== 'hide';
    }
    if (!task) marchingCubes.visible = volume.display !== 'hide';
    // A smoke run computes the volume (the queries are what it checks) but doesn't draw it: under SwiftShader, drawing
    // the marching-cubes mesh stalled the GPU process until the runner gave up on the page.
    if (isSmoke) marchingCubes.visible = false;
    if (frame % 10 === 0) {
        overlay.set('closestPointToGeometry', `${lastQueryMs.toFixed(3)} ms`);
        overlay.set('distance', distanceResult1.distance !== undefined ? distanceResult1.distance.toFixed(3) : '-');
        overlay.set('volume', task ? `${(percentage * 100).toFixed(0)}%` : `${cells.toLocaleString('en-US')} cells, ${volumeMs.toFixed(0)} ms of queries`);
    }
    if (isSmoke && !checked && !task && frame > 10 && !regenerate) {
        const agreement = checkAgainstThreeMeshBvh();
        if (cells === 0) throw new Error('the volume queried no cell');
        // The cylinder over the terrain, within the distance: the line between the closest points is shown.
        targetContainer.position.set(0, -2.2, 0);
        if (!updateDistanceCheck()) throw new Error('the cylinder 1.5 above the terrain\'s mean height found no point within the distance');
        void ready(app.device, { ...overlay.snapshot(), agreedWithThreeMeshBvh: agreement, cells });
        checked = true;
    }
});

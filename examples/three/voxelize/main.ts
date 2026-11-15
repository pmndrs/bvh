/**
 * three/voxelize: a port of three-mesh-bvh's `voxelize` example.
 * Source: three-mesh-bvh's example/voxelize.js (v0.9.18, MIT, Garrett Johnson); see THIRD_PARTY.md.
 *
 * What it demonstrates: two CPU queries on one tree, one per voxel of a grid.
 *
 * - `tree.intersectsBox(voxelBox, boxToMesh)` says whether the voxel touches
 *   the surface (the box is in world space, `boxToMesh` takes it into the
 *   model's, so the model can be moved, rotated and scaled);
 * - for a voxel that doesn't, "solid" casts a ray with `tree.raycastFirst(ray, DoubleSide)`:
 *   if the nearest surface it meets faces away from the ray, the voxel is
 *   inside the mesh (gold), else outside.
 *
 * The grid is filled a few milliseconds per frame, as the original does with
 * a generator. Models: a torus knot, a torus, an icosphere and any dropped
 * glTF (its first mesh; the tree comes from `GenerateMeshBvhWorker`, off the
 * main thread). The smoke run voxelizes a small knot and checks every voxel
 * against three's own `Triangle.intersectsBox` and `Mesh.raycast`.
 */

import {
    BoxGeometry,
    Box3,
    BoxHelper,
    Color,
    DoubleSide,
    IcosahedronGeometry,
    InstancedMesh,
    Matrix4,
    Mesh,
    MeshBasicMaterial,
    MeshStandardMaterial,
    Quaternion,
    Ray,
    Raycaster,
    TorusGeometry,
    TorusKnotGeometry,
    Triangle,
    Vector3,
    type BufferGeometry,
} from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { GenerateMeshBvhWorker, computeBoundsTree, prepareBoundsTrees, type BvhBoundsTree } from '@pmndrs/bvh/three';
import { createApp } from '../../shared/app.ts';
import { enableDrop, loadGltf, meshesOf, requestedModel, type LoadedModel } from '../../shared/loader.ts';
import { layoutInfo } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

const app = await createApp('Voxelize', 'three/voxelize');
const { device, scene, camera, controls, overlay } = app;
await prepareBoundsTrees();
camera.position.set(1, 0.5, 1);
controls.update();

interface Model {
    readonly mesh: Mesh;
    readonly tree: BvhBoundsTree;
}
const models = new Map<string, Model>();
const wireframe = new MeshBasicMaterial({ transparent: true, wireframe: true, depthWrite: false, opacity: 0.02 });
const params = { model: 'Torus Knot', scale: 0.75, resolution: isSmoke ? 12 : 30, solid: true, displayMesh: true, displayBounds: false, insideOnly: false };

function addModel(name: string, geometry: BufferGeometry, tree: BvhBoundsTree, setup: (mesh: Mesh) => void): void {
    const mesh = new Mesh(geometry, wireframe);
    setup(mesh);
    mesh.visible = false;
    scene.add(mesh);
    models.set(name, { mesh, tree });
}

const knot = new TorusKnotGeometry(0.3, 0.1, isSmoke ? 100 : 400, isSmoke ? 16 : 60);
addModel('Torus Knot', knot, computeBoundsTree.call(knot), (mesh) => mesh.rotation.set(0.3, 0.5, 0.1));
const torus = new TorusGeometry(0.3, 0.12, 60, 120);
addModel('Torus', torus, computeBoundsTree.call(torus), () => undefined);
const ico = new IcosahedronGeometry(0.4, 12);
addModel('Icosphere', ico, computeBoundsTree.call(ico), () => undefined);

const boxHelper = new BoxHelper(new Mesh(new BoxGeometry()), 0xffffff);
boxHelper.material.opacity = 0.35;
boxHelper.material.transparent = true;
scene.add(boxHelper);

// A dropped glTF: its first mesh, centred, one unit across; the tree is built in a worker.
const generator = new GenerateMeshBvhWorker();
async function addGltf(loaded: LoadedModel): Promise<void> {
    const source = meshesOf(loaded.root)[0];
    if (!source) throw new Error('the glTF has no mesh');
    loaded.root.updateMatrixWorld(true);
    const geometry = source.geometry.clone().applyMatrix4(source.matrixWorld);
    geometry.center();
    geometry.computeBoundingSphere();
    geometry.scale(0.5 / geometry.boundingSphere!.radius, 0.5 / geometry.boundingSphere!.radius, 0.5 / geometry.boundingSphere!.radius);
    const start = performance.now();
    const tree = await generator.generate(geometry);
    overlay.set('glTF tree (worker)', `${(performance.now() - start).toFixed(0)} ms`);
    const name = `glTF ${loaded.name}`;
    addModel(name, geometry, tree, () => undefined);
    params.model = name;
    needsUpdate = true;
    modelSelect.append(new Option(name, name, true, true));
}

let voxels: InstancedMesh | null = null;
let needsUpdate = true;
let task: Generator<void, void, void> | null = null;
/** 0 empty, 1 touches the surface, 2 inside, per voxel in the generator's order (x outer after y, z inner). */
let classes = new Uint8Array(0);
let lastRun = { ms: 0, count: 0 };

const outsideColor = new Color(0xffffff);
const insideColor = new Color(0xffc107).convertSRGBToLinear();
const position = new Vector3();
const quaternion = new Quaternion();
const worldMatrix = new Matrix4();
const box = new Box3();
const ray = new Ray();

function* voxelize(): Generator<void, void, void> {
    const resolution = params.resolution;
    const total = resolution ** 3;
    const dimensions = params.scale;
    const step = dimensions / resolution;
    if (voxels && voxels.instanceMatrix.count !== total) {
        (voxels.material as MeshStandardMaterial).dispose();
        voxels.dispose();
        voxels.parent?.remove(voxels);
        voxels = null;
    }
    const { mesh, tree } = models.get(params.model)!;
    if (!voxels) {
        // setColorAt first, so the colour buffer exists.
        voxels = new InstancedMesh(new RoundedBoxGeometry(1, 1, 1, 4, 0.1), new MeshStandardMaterial(), total);
        voxels.setColorAt(0, outsideColor);
        scene.add(voxels);
    }
    classes = new Uint8Array(total);
    const scale = new Vector3().setScalar(step);
    const minStart = -dimensions / 2 + step * 0.5;
    mesh.updateMatrixWorld(true);
    const invMat = new Matrix4().copy(mesh.matrixWorld).invert();
    ray.direction.set(0, 0, 1);
    let count = 0;
    let id = 0;
    const start = performance.now();
    for (let y = 0; y < resolution; y++) {
        for (let x = 0; x < resolution; x++) {
            for (let z = 0; z < resolution; z++, id++) {
                position.set(minStart + x * step, minStart + y * step, minStart + z * step);
                box.min.setScalar(-0.5 * step).add(position);
                box.max.setScalar(0.5 * step).add(position);
                if (tree.intersectsBox(box, invMat)) {
                    classes[id] = 1;
                    if (!params.insideOnly) {
                        worldMatrix.compose(position, quaternion, scale);
                        voxels.setMatrixAt(count, worldMatrix);
                        voxels.setColorAt(count, outsideColor);
                        count++;
                    }
                } else if (params.solid && insideMesh(tree, invMat)) {
                    classes[id] = 2;
                    worldMatrix.compose(position, quaternion, scale);
                    voxels.setMatrixAt(count, worldMatrix);
                    voxels.setColorAt(count, insideColor);
                    count++;
                }
                voxels.count = count;
                voxels.instanceMatrix.needsUpdate = true;
                voxels.instanceColor!.needsUpdate = true;
                yield;
            }
        }
    }
    voxels.count = count;
    lastRun = { ms: performance.now() - start, count };
}

const localRay = new Ray();
/** `position` is inside the mesh when the nearest surface along +z faces away from the ray (the original's rule). */
function insideMesh(tree: BvhBoundsTree, invMat: Matrix4): boolean {
    // Into the model's frame; its direction is renormalised for a scaled model.
    localRay.origin.copy(position).applyMatrix4(invMat);
    localRay.direction.copy(ray.direction).transformDirection(invMat);
    const hit = tree.raycastFirst(localRay, DoubleSide);
    return hit !== null && hit.face !== null && hit.face !== undefined && hit.face.normal.dot(localRay.direction) > 0;
}

/** The same classes from three alone: Triangle.intersectsBox over every triangle, Mesh.raycast for the ray. */
function bruteClasses(): Uint8Array {
    const resolution = params.resolution;
    const step = params.scale / resolution;
    const { mesh } = models.get(params.model)!;
    mesh.updateMatrixWorld(true);
    const geometry = mesh.geometry;
    const index = geometry.index!;
    const positions = geometry.attributes.position;
    const world: Triangle[] = [];
    const fromIndex = (i: number): Vector3 => new Vector3().fromBufferAttribute(positions, index.getX(i)).applyMatrix4(mesh.matrixWorld);
    for (let i = 0; i < index.count; i += 3) world.push(new Triangle(fromIndex(i), fromIndex(i + 1), fromIndex(i + 2)));
    const solidMesh = new Mesh(geometry, new MeshBasicMaterial({ side: DoubleSide }));
    solidMesh.matrixWorld.copy(mesh.matrixWorld);
    solidMesh.matrixWorldAutoUpdate = false;
    const raycaster = new Raycaster();
    const out = new Uint8Array(resolution ** 3);
    const minStart = -params.scale / 2 + step * 0.5;
    const cell = new Box3();
    const origin = new Vector3();
    const invMat = new Matrix4().copy(mesh.matrixWorld).invert();
    let id = 0;
    for (let y = 0; y < resolution; y++) {
        for (let x = 0; x < resolution; x++) {
            for (let z = 0; z < resolution; z++, id++) {
                origin.set(minStart + x * step, minStart + y * step, minStart + z * step);
                cell.min.setScalar(-0.5 * step).add(origin);
                cell.max.setScalar(0.5 * step).add(origin);
                if (world.some((triangle) => triangle.intersectsBox(cell))) {
                    out[id] = 1;
                    continue;
                }
                if (!params.solid) continue;
                raycaster.set(origin, new Vector3(0, 0, 1));
                const hits = raycaster.intersectObject(solidMesh);
                const first = hits[0];
                if (first?.face) {
                    const localDirection = new Vector3(0, 0, 1).transformDirection(invMat);
                    if (first.face.normal.dot(localDirection) > 0) out[id] = 2;
                }
            }
        }
    }
    return out;
}

const modelSelect = overlay.select('model', ['Torus Knot', 'Torus', 'Icosphere'], params.model, (v) => {
    params.model = v;
    needsUpdate = true;
});
overlay.range('resolution', 5, 75, 1, params.resolution, (v) => ((params.resolution = v), (needsUpdate = true)));
overlay.range('scale', 0.1, 4, 0.1, params.scale, (v) => ((params.scale = v), (needsUpdate = true)));
overlay.checkbox('solid', params.solid, (v) => ((params.solid = v), (needsUpdate = true)));
overlay.checkbox('inside only', params.insideOnly, (v) => ((params.insideOnly = v), (needsUpdate = true)));
overlay.checkbox('display mesh', params.displayMesh, (v) => (params.displayMesh = v));
overlay.checkbox('display bounds', params.displayBounds, (v) => (params.displayBounds = v));
overlay.button('rebuild', () => (needsUpdate = true));
overlay.note('Demonstrates tree.intersectsBox (does a voxel touch the surface?) and tree.raycastFirst with DoubleSide (does the nearest surface face away, so the voxel is inside?). White voxels touch the surface, gold ones are inside. Drop a glTF to voxelize its first mesh.');

const model = requestedModel();
if (model) await addGltf(await loadGltf(model));
enableDrop(
    (loaded) => addGltf(loaded),
    (error) => overlay.error(String(error)),
);
overlay.tree(layoutInfo(models.get(params.model)!.tree.layout));

let frames = 0;
let smokeDone = false;
app.loop(async () => {
    scene.updateMatrixWorld(true);
    if (needsUpdate) {
        task = voxelize();
        needsUpdate = false;
        overlay.tree(layoutInfo(models.get(params.model)!.tree.layout));
    }
    const budget = isSmoke ? 1e9 : 16;
    const start = performance.now();
    while (task && performance.now() - start < budget) {
        if (task.next().done) {
            task = null;
            overlay.set('voxels', `${lastRun.count.toLocaleString('en-US')} of ${(params.resolution ** 3).toLocaleString('en-US')}, ${lastRun.ms.toFixed(0)} ms`);
        }
    }
    boxHelper.object.scale.setScalar(params.scale);
    boxHelper.object.updateMatrixWorld(true);
    boxHelper.update();
    for (const [name, { mesh }] of models) mesh.visible = name === params.model && params.displayMesh;
    boxHelper.visible = params.displayBounds;
    frames++;
    if (isSmoke && !task && !smokeDone && frames > 2) {
        smokeDone = true;
        const t0 = performance.now();
        const brute = bruteClasses();
        const bruteMs = performance.now() - t0;
        let touch = 0;
        let inside = 0;
        let mismatches = 0;
        for (let i = 0; i < brute.length; i++) {
            if (brute[i] === 1) touch++;
            if (brute[i] === 2) inside++;
            if (brute[i] !== classes[i]) mismatches++;
        }
        if (touch === 0 || inside === 0) throw new Error(`the knot should have surface (${touch}) and inside (${inside}) voxels`);
        // Voxels whose box only grazes a triangle may differ in the last bit between the two tests.
        if (mismatches > brute.length * 0.002) throw new Error(`${mismatches} of ${brute.length} voxels differ from the brute-force classes`);
        overlay.set('smoke', `${brute.length} voxels: ${touch} touch, ${inside} inside, ${mismatches} differ; brute force ${bruteMs.toFixed(0)} ms`);
        await ready(device, { ...overlay.snapshot(), voxels: brute.length, touch, inside, mismatches, voxelizeMs: lastRun.ms, bruteMs });
    }
});

/**
 * babylon/picking: `scene.pick` over a big mesh against the CPU raycast of
 * `@pmndrs/bvh/babylon`, with picks per second for each.
 *
 * Picking needs no GPU, so this page runs on Babylon's WebGL engine too
 * (`?engine=webgl`); with WebGPU available it takes the WebGPUEngine.
 *
 * - `BabylonBvhPicker.create(scene)` builds a tinybvh tree for the mesh and
 *   returns picks as Babylon `PickingInfo`s.
 * - Babylon's side is `scene.pickWithRay` on the same rays. A mesh this big is
 *   what Babylon's docs say to give a submesh octree (`mesh.subdivide`,
 *   `createOrUpdateSubmeshesOctree`), so it has one (`?octree=0` for brute
 *   force).
 * - The pointer picks with the package's picker, and the overlay shows the
 *   `PickingInfo`; "benchmark" times both on the same random camera rays.
 *
 * `?triangles=` sets the mesh's size (default 400,000).
 */

import '@babylonjs/core/Culling/Octrees/octreeSceneComponent.js';
import { StandardMaterial } from '@babylonjs/core/Materials/standardMaterial.js';
import { Color3 } from '@babylonjs/core/Maths/math.color.js';
import { HemisphericLight } from '@babylonjs/core/Lights/hemisphericLight.js';
import { DirectionalLight } from '@babylonjs/core/Lights/directionalLight.js';
import { Matrix, Vector3 } from '@babylonjs/core/Maths/math.vector.js';
import { CreateSphere } from '@babylonjs/core/Meshes/Builders/sphereBuilder.js';
import { Mesh } from '@babylonjs/core/Meshes/mesh.js';
import { VertexData } from '@babylonjs/core/Meshes/mesh.vertexData.js';
import type { PickingInfo } from '@babylonjs/core/Collisions/pickingInfo.js';
import type { Ray } from '@babylonjs/core/Culling/ray.js';
import { BabylonBvhPicker } from '@pmndrs/bvh/babylon';
import { createRandom } from '@pmndrs/bvh';
import { createBabylonApp, runBabylon } from '../../shared/babylon/app.ts';
import { fail, isSmoke, ready } from '../../shared/smoke.ts';
import { layoutInfo } from '../../shared/stats.ts';
import { torusKnot } from '../../shared/vanilla/geometry.ts';

const app = await createBabylonApp('Babylon picking', 'babylon/picking');
const { scene, camera, overlay, engine } = app;
const query = new URLSearchParams(location.search);
const triangles = Math.max(2_000, Number(query.get('triangles') ?? (isSmoke ? 60_000 : 400_000)));
const useOctree = query.get('octree') !== '0';

// ---- One big mesh: a torus knot with Babylon's own normals, a material (Babylon picks a mesh without one by its bounding box).

const radial = 40;
const knot = torusKnot(2, 0.45, Math.max(8, Math.round(triangles / (2 * radial))), radial);
const mesh = new Mesh('knot', scene);
const data = new VertexData();
data.positions = knot.positions;
data.indices = knot.indices;
const normals = new Float32Array(knot.positions.length);
VertexData.ComputeNormals(knot.positions, knot.indices, normals);
data.normals = normals;
data.applyToMesh(mesh, true);
const material = new StandardMaterial('knot', scene);
material.diffuseColor = new Color3(0.45, 0.62, 0.95);
material.backFaceCulling = false;
mesh.material = material;
new HemisphericLight('sky', new Vector3(0.3, 1, 0.2), scene).intensity = 0.7;
new DirectionalLight('sun', new Vector3(-0.4, -1, -0.5), scene).intensity = 0.6;
camera.radius = 9;
const triangleCount = knot.indices.length / 3;

// Babylon's recommendation for picking on a big mesh: about a thousand triangles per submesh, an octree over them.
if (useOctree) {
    mesh.subdivide(Math.max(1, Math.ceil(triangleCount / 1000)));
    mesh.createOrUpdateSubmeshesOctree();
}

const marker = CreateSphere('marker', { diameter: 0.12 }, scene);
const markerMaterial = new StandardMaterial('marker', scene);
markerMaterial.emissiveColor = new Color3(1, 0.7, 0.3);
markerMaterial.disableLighting = true;
marker.material = markerMaterial;
marker.isPickable = false;
marker.isVisible = false;

// ---- The picker: its tree, timed.

const buildStart = performance.now();
const picker = await BabylonBvhPicker.create(scene);
const buildMs = performance.now() - buildStart;
overlay.build({ builtBy: 'tinybvh (CPU)', cpuMs: buildMs });
overlay.tree(layoutInfo(picker.cpuScene!.meshes[0]));
overlay.set('mesh', `${triangleCount.toLocaleString('en-US')} triangles`);
overlay.set('Babylon', useOctree ? 'submesh octree' : 'every triangle');

// ---- Random camera rays, the same for both sides.

const width = () => engine.getRenderWidth();
const height = () => engine.getRenderHeight();
function cameraRays(count: number, seed: number): Ray[] {
    const random = createRandom(seed);
    const rays: Ray[] = [];
    camera.getViewMatrix(true);
    camera.getProjectionMatrix(true);
    for (let i = 0; i < count; i++) rays.push(scene.createPickingRay(random() * width(), random() * height(), Matrix.Identity(), camera));
    return rays;
}

interface Timing {
    readonly ours: number;
    readonly babylon: number;
    readonly hits: number;
    readonly oursCount: number;
    readonly babylonCount: number;
}

/** Picks per second of each side: the picker over `oursCount` rays, Babylon over as many as fit in `budgetMs`. */
function benchmark(oursCount: number, budgetMs: number): Timing {
    const rays = cameraRays(oursCount, 7);
    let hits = 0;
    let start = performance.now();
    for (const ray of rays) if (picker.pickWithRay(ray).hit) hits++;
    const ours = (rays.length / (performance.now() - start)) * 1000;
    start = performance.now();
    let babylonCount = 0;
    for (const ray of rays) {
        scene.pickWithRay(ray);
        babylonCount++;
        if (performance.now() - start > budgetMs) break;
    }
    const babylon = (babylonCount / (performance.now() - start)) * 1000;
    return { ours, babylon, hits, oursCount: rays.length, babylonCount };
}

function show(timing: Timing): void {
    overlay.rays('picker (picks/s)', timing.ours);
    overlay.rays('scene.pick (picks/s)', timing.babylon);
    overlay.set('speed-up', `${(timing.ours / timing.babylon).toFixed(1)}x`);
    overlay.set('benchmark', `${timing.hits} of ${timing.oursCount} rays hit; Babylon timed on ${timing.babylonCount}`);
}

// ---- The pointer.

function describe(info: PickingInfo): string {
    return info.hit ? `${info.pickedMesh?.name} face ${info.faceId}, ${info.distance.toFixed(3)} away` : 'no hit';
}
let pointer: { x: number; y: number } | null = null;
app.canvas.addEventListener('pointermove', (event) => {
    const box = app.canvas.getBoundingClientRect();
    pointer = { x: ((event.clientX - box.left) / box.width) * width(), y: ((event.clientY - box.top) / box.height) * height() };
});
app.canvas.addEventListener('pointerleave', () => (pointer = null));
let alsoBabylon = false;
overlay.checkbox('also scene.pick on the pointer', alsoBabylon, (value) => (alsoBabylon = value));
overlay.button('benchmark', () => show(benchmark(500, 1500)));
overlay.note('The pointer picks with the package\'s picker (a PickingInfo, like scene.pick\'s). The benchmark picks the same random camera rays with both.');

runBabylon(app, () => {
    if (!pointer) {
        marker.isVisible = false;
        return;
    }
    const start = performance.now();
    const info = picker.pick(pointer.x, pointer.y);
    const ms = performance.now() - start;
    overlay.set('picker', `${describe(info)} (${ms.toFixed(3)} ms)`);
    marker.isVisible = info.hit;
    if (info.pickedPoint) marker.position.copyFrom(info.pickedPoint);
    if (alsoBabylon) {
        const babylonStart = performance.now();
        const babylon = scene.pick(pointer.x, pointer.y);
        overlay.set('scene.pick', `${describe(babylon)} (${(performance.now() - babylonStart).toFixed(3)} ms)`);
    }
});

// ---- Smoke: both sides on the same rays must agree, and the PickingInfo must be Babylon's.

if (isSmoke) {
    void (async () => {
        await new Promise((resolve) => setTimeout(resolve, 300));
        const rays = cameraRays(96, 3);
        let hits = 0;
        let disagreements = 0;
        for (const ray of rays) {
            const babylon = scene.pickWithRay(ray)!;
            const ours = picker.pickWithRay(ray);
            if (ours.hit !== babylon.hit) {
                disagreements++;
                continue;
            }
            if (!babylon.hit) continue;
            hits++;
            // The same triangle, or (a ray through an edge) the same point.
            if (Math.abs(ours.distance - babylon.distance) > 1e-3 * Math.max(1, babylon.distance) || ours.pickedMesh !== babylon.pickedMesh) disagreements++;
            else if (ours.faceId !== babylon.faceId && Math.min(babylon.bu, babylon.bv, 1 - babylon.bu - babylon.bv) > 1e-3) disagreements++;
        }
        if (hits < 20) throw new Error(`only ${hits} of ${rays.length} rays hit the mesh`);
        if (disagreements > 1) throw new Error(`${disagreements} of ${rays.length} rays disagree with scene.pick`);
        // The pointer's path: pick(x, y) through the active camera.
        const centre = picker.pick(width() / 2, height() / 2);
        const babylonCentre = scene.pick(width() / 2, height() / 2);
        if (centre.hit !== babylonCentre.hit || (centre.hit && Math.abs(centre.distance - babylonCentre.distance) > 1e-3)) throw new Error('pick(x, y) at the centre disagrees with scene.pick');
        const timing = benchmark(300, 600);
        show(timing);
        await ready(app.device, { ...overlay.snapshot(), hits, rays: rays.length, disagreements, ours: timing.ours, babylon: timing.babylon, engine: app.kind, triangles: triangleCount });
    })().catch(fail);
}

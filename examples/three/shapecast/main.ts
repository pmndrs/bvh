/**
 * three/shapecast: three-mesh-bvh's `shapecast` example on our tree.
 *
 * A spinning torus knot with a tree (`computeBoundsTree`, tinybvh) and a shape
 * you move with the transform gizmo or the sliders: a sphere
 * (`boundsTree.intersectsSphere`), a box placed by a matrix
 * (`boundsTree.intersectsBox(box, matrix)`) or another mesh with its own tree
 * (`boundsTree.intersectsGeometry(geometry, matrix)`, tree against tree). The
 * shape turns red where it touches the knot. The calls and arguments are
 * three-mesh-bvh's; the queries run over our CPU tree (`CpuMeshBvh`
 * underneath). `MeshBVHHelper` draws the tree's boxes.
 *
 * Ported from three-mesh-bvh's example/shapecast.js (v0.9.18, MIT, Garrett
 * Johnson); see THIRD_PARTY.md. Changed: our shell and overlay instead of
 * lil-gui and stats, and the smoke check.
 */

import { Box3, BoxGeometry, DoubleSide, Euler, Matrix4, Mesh, MeshStandardMaterial, Quaternion, Sphere, SphereGeometry, TorusKnotGeometry, Vector3 } from 'three';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { MeshBVH } from 'three-mesh-bvh';
import { computeBoundsTree, MeshBVHHelper, prepareBoundsTrees } from '@pmndrs/bvh/three';
import { createApp } from '../../shared/app.ts';
import { layoutInfo } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

await prepareBoundsTrees();
const app = await createApp('Shapecast', 'three/shapecast');
const { renderer, scene, camera, controls, overlay } = app;
camera.position.set(3, 3, 3);
controls.target.set(0, 0, 0);

const params = { speed: 1, shape: 'sphere' as 'sphere' | 'box' | 'geometry', position: new Vector3(), rotation: new Euler() };

// The target: a 80,000-triangle knot with a tree.
const knotGeometry = new TorusKnotGeometry(1, 0.4, 400, 100);
const buildStart = performance.now();
const knotTree = computeBoundsTree.call(knotGeometry);
const buildMs = performance.now() - buildStart;
const targetMesh = new Mesh(knotGeometry, new MeshStandardMaterial({ color: 0xffffff, side: DoubleSide }));
scene.add(targetMesh);
overlay.build({ builtBy: 'tinybvh (computeBoundsTree)', cpuMs: buildMs });
overlay.tree(layoutInfo(knotTree.layout));

// The shapes.
const shapeMaterial = new MeshStandardMaterial({ metalness: 0.1, transparent: true, opacity: 0.75, premultipliedAlpha: true });
const shapes = {
    sphere: new Mesh(new SphereGeometry(1, 50, 50), shapeMaterial),
    box: new Mesh(new BoxGeometry(1, 1, 1), shapeMaterial),
    geometry: new Mesh(new TorusKnotGeometry(0.5, 0.2, 200, 50), shapeMaterial),
};
computeBoundsTree.call(shapes.geometry.geometry);
for (const shape of Object.values(shapes)) scene.add(shape);

const helper = new MeshBVHHelper(targetMesh, 10);
helper.visible = false;
scene.add(helper);

const gizmo = new TransformControls(camera, renderer.domElement);
scene.add(gizmo.getHelper());
gizmo.addEventListener('dragging-changed', (event) => (controls.enabled = !event.value));
gizmo.addEventListener('change', () => {
    const shape = shapes[params.shape];
    params.position.copy(shape.position);
    params.rotation.copy(shape.rotation);
    for (const [axis, input] of positionInputs) input.value = String(shape.position[axis]);
});
addEventListener('keydown', (event) => {
    if (event.key === 'w') gizmo.setMode('translate');
    if (event.key === 'e') gizmo.setMode('rotate');
});

overlay.range('speed', 0, 10, 0.1, params.speed, (value) => (params.speed = value));
overlay.select('shape', ['sphere', 'box', 'geometry'] as const, params.shape, (value) => (params.shape = value));
overlay.select('gizmo', ['translate', 'rotate'] as const, 'translate', (value) => gizmo.setMode(value));
const positionInputs = (['x', 'y', 'z'] as const).map((axis) => [axis, overlay.range(`position ${axis}`, -5, 5, 0.001, 0, (value) => (params.position[axis] = value))] as const);
overlay.checkbox('display BVH', false, (value) => (helper.visible = value));
overlay.range('BVH depth', 1, 40, 1, 10, (value) => {
    helper.depth = value;
    helper.update();
});
overlay.note('Drag the gizmo (w: move, e: rotate) or the sliders; the shape turns red where it touches the knot. The query time is one intersects* call on the CPU.');

/** One query of the shape against the knot, as three-mesh-bvh's example asks it. */
const sphere = new Sphere();
const box = new Box3(new Vector3(-0.5, -0.5, -0.5), new Vector3(0.5, 0.5, 0.5));
const shapeToKnot = new Matrix4();
function query(kind: typeof params.shape, matrix: Matrix4): boolean {
    if (kind === 'sphere') {
        sphere.set(new Vector3(), 1).applyMatrix4(matrix);
        return knotTree.intersectsSphere(sphere);
    }
    if (kind === 'box') return knotTree.intersectsBox(box, matrix);
    return knotTree.intersectsGeometry(shapes.geometry.geometry, matrix);
}

/** Smoke check: 300 random placements of each shape, our answers against three-mesh-bvh's on the same geometry. */
function checkAgainstThreeMeshBvh(): { checked: number; hits: number } {
    const theirKnot = new MeshBVH(knotGeometry.clone());
    const theirShape = shapes.geometry.geometry.clone();
    (theirShape as unknown as { boundsTree: MeshBVH }).boundsTree = new MeshBVH(theirShape);
    let seed = 7;
    const random = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
    const matrix = new Matrix4();
    const position = new Vector3();
    const scale = new Vector3();
    const quaternion = new Quaternion();
    let checked = 0;
    let hits = 0;
    for (const kind of ['sphere', 'box', 'geometry'] as const) {
        for (let i = 0; i < 300; i++) {
            position.set(random() * 4 - 2, random() * 4 - 2, random() * 4 - 2);
            const rotation = new Euler(random() * 6.28, random() * 6.28, random() * 6.28);
            scale.setScalar(0.3 + random() * 0.8);
            matrix.compose(position, quaternion.setFromEuler(rotation), scale);
            const ours = query(kind, matrix);
            let theirs: boolean;
            if (kind === 'sphere') theirs = theirKnot.intersectsSphere(new Sphere(new Vector3(), 1).applyMatrix4(matrix));
            else if (kind === 'box') theirs = theirKnot.intersectsBox(box, matrix);
            else theirs = theirKnot.intersectsGeometry(theirShape, matrix);
            if (ours !== theirs) throw new Error(`${kind} ${i}: ours ${ours}, three-mesh-bvh ${theirs}`);
            checked++;
            if (ours) hits++;
        }
    }
    if (hits < 100 || hits > checked - 100) throw new Error(`the check's placements are lopsided: ${hits} of ${checked} touch`);
    return { checked, hits };
}

let frame = 0;
let queryMs = 0;
app.loop((_now, delta) => {
    frame++;
    targetMesh.rotation.y += params.speed * delta;
    targetMesh.updateMatrixWorld();
    for (const shape of Object.values(shapes)) shape.visible = false;
    const shape = shapes[params.shape];
    shape.visible = true;
    shape.position.copy(params.position);
    shape.rotation.copy(params.rotation);
    shape.updateMatrixWorld();
    if (gizmo.object !== shape) gizmo.attach(shape);

    shapeToKnot.copy(targetMesh.matrixWorld).invert().multiply(shape.matrixWorld);
    const start = performance.now();
    const hit = query(params.shape, shapeToKnot);
    queryMs = queryMs * 0.9 + (performance.now() - start) * 0.1;
    shapeMaterial.color.set(hit ? 0xe91e63 : 0x666666);
    shapeMaterial.emissive.set(0xe91e63).multiplyScalar(hit ? 0.25 : 0);
    if (frame % 10 === 0) {
        overlay.set(`${params.shape} query`, `${queryMs.toFixed(3)} ms`);
        overlay.set('touching', hit ? 'yes' : 'no');
    }
    if (isSmoke && frame === 30) {
        const checked = checkAgainstThreeMeshBvh();
        void ready(app.device, { ...overlay.snapshot(), agreedWithThreeMeshBvh: checked });
    }
});

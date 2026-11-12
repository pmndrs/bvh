/**
 * compare/scene: a scene-level tree over thousands of objects sharing one
 * mesh: ours (`CpuSceneBvh`) against bvh.js (a dynamic box tree) and
 * three-mesh-bvh's `ObjectBVH`. Build, frames of motion with each tree's
 * update timed, frustum and box culling, rays, and the answers checked against
 * ours before and after the motion. Live: the camera's frustum culled by the
 * chosen library every frame as the objects move.
 * What else: the same instances in `SceneTrees`, whose top level is refitted
 * on the GPU every frame and traced there.
 */

import { Frustum, InstancedMesh, Matrix4, MeshStandardMaterial } from 'three';
import { BufferAttribute, BufferGeometry } from 'three';
import { SceneTrees } from '@pmndrs/bvh';
import { SceneTreesObject3DSync } from '@pmndrs/bvh/three';
import { SCENE_ADAPTERS } from '../../../compare/adapters/scene.ts';
import { makeScene, moveObjects } from '../../../compare/core/scene.ts';
import { sceneSuite } from '../../../compare/core/scene-suite.ts';
import { createComparePage, expectAgreement, quick } from '../../shared/compare.ts';
import { ready } from '../../shared/smoke.ts';

const page = await createComparePage('compare/scene', 'compare/scene', 'A tree over many objects (each a 1k-triangle knot, the same mesh), built from their world boxes; then frames in which every object moves.');
const { app, panel } = page;
page.mesh.visible = false;
const log = (line: string) => panel.status(line);

const count = quick ? 2_000 : 10_000;
const scene = makeScene(count);
panel.status(`scene: ${scene.name}`);
const { measurements, built } = await sceneSuite(SCENE_ADAPTERS, scene, { log, queries: quick ? 128 : 1024, frames: quick ? 5 : 20, plan: quick ? { runs: 3, budgetMs: 250 } : {} });
const names = { ours: '@pmndrs/bvh (CpuSceneBvh)', 'bvh.js': 'bvh.js (HybridBuilder)', tmbvh: 'three-mesh-bvh ObjectBVH' };
panel.table(`Scene-level trees, ${scene.name}`, measurements, ['build', 'memory', 'update', 'frustum', 'box', 'raycast'], names);
panel.table('Answers after the frames of motion (disagreements with ours)', measurements, ['frustum-after-motion', 'box-after-motion', 'raycast-after-motion'], names);
panel.paragraph('Update: ours rebuilds its CPU top level (setInstances); bvh.js moves each leaf (reinserting those that left their parent\'s box); ObjectBVH refits. bvh.js has no triangle rays: it is a box tree over objects.');

// Draw the objects.
const geometry = new BufferGeometry();
geometry.setAttribute('position', new BufferAttribute(scene.mesh.positions, 3));
geometry.setIndex(new BufferAttribute(scene.mesh.indices, 1));
geometry.computeVertexNormals();
const instanced = new InstancedMesh(geometry, new MeshStandardMaterial({ color: 0x8fb3ff, roughness: 0.6 }), count);
const matrix = new Matrix4();
const writeMatrices = () => {
    scene.objects.forEach((object, i) => instanced.setMatrixAt(i, matrix.fromArray(object.matrix)));
    instanced.instanceMatrix.needsUpdate = true;
};
writeMatrices();
app.scene.add(instanced);
app.camera.position.set(0, 0, 0.01);
app.camera.far = 400;
app.camera.updateProjectionMatrix();
app.controls.target.set(0, 0, -1);

// What else: the same instances in SceneTrees, the top level refitted on the GPU every frame.
const trees = await SceneTrees.create(app.device, { label: 'compare scene' });
const sync = new SceneTreesObject3DSync(trees, app.scene);
const gpuUpdate = () => {
    const start = performance.now();
    sync.update();
    const encoder = app.device.createCommandEncoder();
    trees.update(encoder);
    app.device.queue.submit([encoder.finish()]);
    return performance.now() - start;
};
gpuUpdate();
await trees.whenBuilt();
const note = panel.whatElse(`The same ${count.toLocaleString('en-US')} instances in SceneTrees: the top level is ${trees.topLevelPath === 'gpu' ? 'refitted on the GPU' : 'rebuilt on the CPU (no subgroups)'} every frame as they move, and the WGSL walk traces it there, with nothing read back.`);
const gpuLine = document.createElement('div');
note.append(gpuLine);
panel.csv(app.device);

const libraries = new Map(built.map((entry) => [names[entry.adapter.id as keyof typeof names], entry]));
let chosen = [...libraries.keys()][0];
app.overlay.select('cull with', [...libraries.keys()], chosen, (value) => (chosen = value));
let moving = true;
app.overlay.checkbox('move objects', moving, (value) => (moving = value));

const checked = expectAgreement(measurements, ['tmbvh'], log);
const frustum = new Frustum();
const viewProjection = new Matrix4();
const planes = new Float32Array(24);
let frame = 0;
app.loop(async () => {
    frame++;
    const entry = libraries.get(chosen)!;
    if (moving) {
        moveObjects(scene, 100 + frame);
        writeMatrices();
        const t0 = performance.now();
        for (const { scene: tree } of built) if (tree === entry.scene) tree.update?.(scene.objects);
        const updateMs = performance.now() - t0;
        // Keep the others in step too (not timed), so switching libraries shows the current scene.
        for (const { scene: tree } of built) if (tree !== entry.scene) tree.update?.(scene.objects);
        app.overlay.set('CPU update', `${updateMs.toFixed(2)} ms (${entry.scene.updateKind})`);
        gpuLine.textContent = `SceneTrees sync + GPU top-level update: ${gpuUpdate().toFixed(2)} ms of CPU this frame.`;
    }
    app.camera.updateMatrixWorld();
    viewProjection.multiplyMatrices(app.camera.projectionMatrix, app.camera.matrixWorldInverse);
    frustum.setFromProjectionMatrix(viewProjection);
    frustum.planes.forEach((plane, i) => planes.set([plane.normal.x, plane.normal.y, plane.normal.z, plane.constant], i * 4));
    const t0 = performance.now();
    const visible = entry.scene.intersectsFrustum?.(planes, new Float32Array(viewProjection.elements)) ?? [];
    app.overlay.set('in view', `${visible.length.toLocaleString('en-US')} of ${count.toLocaleString('en-US')}, ${(performance.now() - t0).toFixed(3)} ms`);
    if (frame === 10) await ready(app.device, { ...app.overlay.snapshot(), csv: panel.csvText, status: panel.statusText, checkedAgainstOurs: checked });
});

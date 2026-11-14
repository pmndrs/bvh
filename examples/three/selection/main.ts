/**
 * three/selection: three-mesh-bvh's `selection` example on our tree.
 *
 * Draw a lasso or a box over a spinning torus knot (60,000 triangles) with
 * the left button; the triangles under it are highlighted. The tree's
 * `shapecast` narrows the search (select.ts): each box is projected to the
 * screen, its convex hull tested against the lasso, contained boxes taken
 * whole. Modes: a triangle is selected by its centroid, by its centroid when
 * that is visible (one `raycastFirst` toward the camera through the same
 * tree), or by any overlap. "use bounds tree" off visits every triangle, for
 * the timing against the tree. Right button orbits.
 *
 * Ported from three-mesh-bvh's example/selection.js (v0.9.18, MIT, Garrett
 * Johnson); see THIRD_PARTY.md. Changed: our shell and overlay, the knot
 * stays indexed (the original's `toNonIndexed()` mesh gets an index from
 * `MeshBVH`), no shadows, the smoke check.
 */

import { BufferAttribute, Float32BufferAttribute, GridHelper, Group, Line, LineBasicMaterial, MathUtils, Mesh, MeshBasicMaterial, MeshStandardMaterial, MOUSE, TorusKnotGeometry, TOUCH } from 'three';
import { computeBoundsTree, MeshBVHHelper, prepareBoundsTrees } from '@pmndrs/bvh/three';
import { createApp } from '../../shared/app.ts';
import { layoutInfo } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';
import { BoxSelection, computeSelectedTriangles, LassoSelection, type SelectionParams, type SelectionTool } from './select.ts';

await prepareBoundsTrees();
const app = await createApp('Selection', 'three/selection');
const { renderer, scene, camera, controls, overlay } = app;
camera.position.set(2, 4, 6);
controls.target.set(0, 0, 0);
controls.minDistance = 3;
controls.touches.ONE = TOUCH.PAN;
controls.mouseButtons.LEFT = MOUSE.PAN;
controls.touches.TWO = TOUCH.ROTATE;
controls.mouseButtons.RIGHT = MOUSE.ROTATE;
controls.enablePan = false;
scene.add(camera);

const params: { -readonly [K in keyof SelectionParams]: SelectionParams[K] } & { toolMode: 'lasso' | 'box'; liveUpdate: boolean; wireframe: boolean; rotate: boolean } = {
    toolMode: 'lasso',
    selectionMode: 'intersection',
    liveUpdate: false,
    selectWholeModel: false,
    wireframe: false,
    useBoundsTree: true,
    rotate: true,
};
let tool: SelectionTool = new LassoSelection();

// The selection's outline, drawn in front of the camera.
const selectionShape = new Line(undefined, new LineBasicMaterial({ color: 0xff9800, depthTest: false }));
selectionShape.renderOrder = 1;
selectionShape.position.z = -0.2;
selectionShape.frustumCulled = false;
camera.add(selectionShape);

const group = new Group();
scene.add(group);
const geometry = new TorusKnotGeometry(1.5, 0.5, 500, 60);
const buildStart = performance.now();
const tree = computeBoundsTree.call(geometry);
overlay.build({ builtBy: 'tinybvh (computeBoundsTree)', cpuMs: performance.now() - buildStart });
overlay.tree(layoutInfo(tree.layout));
const mesh = new Mesh(geometry, new MeshStandardMaterial({ polygonOffset: true, polygonOffsetFactor: 1 }));
group.add(mesh);
const helper = new MeshBVHHelper(mesh, 10);
helper.visible = false;
group.add(helper);

// The highlight: the same vertices, an index holding the selected triangles only.
const highlightGeometry = geometry.clone();
highlightGeometry.setIndex(new BufferAttribute(new Uint32Array(geometry.index!.count), 1));
highlightGeometry.drawRange.count = 0;
const highlightMesh = new Mesh(highlightGeometry, new MeshBasicMaterial({ opacity: 0.05, transparent: true, depthWrite: false, color: 0xff9800 }));
highlightMesh.renderOrder = 1;
const highlightWireframe = new Mesh(highlightGeometry, new MeshBasicMaterial({ opacity: 0.25, transparent: true, wireframe: true, depthWrite: false, color: 0xff9800 }));
highlightWireframe.renderOrder = 2;
group.add(highlightMesh, highlightWireframe);

const grid = new GridHelper(10, 10, 0xffffff, 0xffffff);
(grid.material as LineBasicMaterial).opacity = 0.2;
(grid.material as LineBasicMaterial).transparent = true;
grid.position.y = -2.75;
scene.add(grid);

let selectionShapeNeedsUpdate = false;
let selectionNeedsUpdate = false;
let selectedTriangles = 0;
let selectMs = 0;

overlay.select('tool', ['lasso', 'box'] as const, params.toolMode, (value) => {
    params.toolMode = value;
    tool = value === 'box' ? new BoxSelection() : new LassoSelection();
});
overlay.select('selection mode', ['centroid', 'centroid-visible', 'intersection'] as const, params.selectionMode, (value) => (params.selectionMode = value));
overlay.checkbox('select whole model', params.selectWholeModel, (value) => (params.selectWholeModel = value));
overlay.checkbox('live update', params.liveUpdate, (value) => (params.liveUpdate = value));
overlay.checkbox('use bounds tree', params.useBoundsTree, (value) => (params.useBoundsTree = value));
overlay.checkbox('wireframe', params.wireframe, (value) => ((mesh.material as MeshStandardMaterial).wireframe = value));
overlay.checkbox('rotate', params.rotate, (value) => (params.rotate = value));
overlay.checkbox('display BVH', false, (value) => (helper.visible = value));
overlay.range('BVH depth', 1, 30, 1, 10, (value) => {
    helper.depth = value;
    helper.update();
});
overlay.note('Drag with the left button to select (lasso or box), right button to orbit. The time is one computeSelectedTriangles: the shapecast and the tests.');

renderer.domElement.addEventListener('pointerdown', (event) => tool.handlePointerDown(event));
renderer.domElement.addEventListener('pointerup', () => {
    tool.handlePointerUp();
    selectionShape.visible = false;
    if (tool.points.length) selectionNeedsUpdate = true;
});
renderer.domElement.addEventListener('pointermove', (event) => {
    if ((event.buttons & 1) === 0) return;
    if (tool.handlePointerMove(event).changed) {
        selectionShapeNeedsUpdate = true;
        selectionShape.visible = true;
        if (params.liveUpdate) selectionNeedsUpdate = true;
    }
});

function updateSelection(): number[] {
    const start = performance.now();
    const indices = computeSelectedTriangles(mesh, tree, camera, tool, params);
    selectMs = performance.now() - start;
    const source = geometry.index!;
    const target = highlightGeometry.index!;
    if (indices.length && params.selectWholeModel) {
        for (let i = 0; i < source.count; i++) target.setX(i, source.getX(i));
        highlightGeometry.drawRange.count = Infinity;
    } else {
        for (let i = 0; i < indices.length; i++) target.setX(i, source.getX(indices[i]));
        highlightGeometry.drawRange.count = indices.length;
    }
    target.needsUpdate = true;
    selectedTriangles = indices.length / 3;
    return indices;
}

/** Smoke: a lasso (a circle on the screen) in every mode, through the tree and without it: the same triangles. */
function selectionCheck(): Record<string, number> {
    const lasso = new LassoSelection();
    for (let i = 0; i < 48; i++) {
        const angle = (i / 48) * Math.PI * 2;
        lasso.lassoPoints.push(Math.cos(angle) * 0.35 + 0.05, Math.sin(angle) * 0.3, 0);
    }
    tool = lasso;
    group.updateMatrixWorld(true);
    camera.updateMatrixWorld();
    const out: Record<string, number> = {};
    for (const mode of ['centroid', 'centroid-visible', 'intersection'] as const) {
        const withTree = computeSelectedTriangles(mesh, tree, camera, lasso, { selectionMode: mode, selectWholeModel: false, useBoundsTree: true });
        const start = performance.now();
        const without = computeSelectedTriangles(mesh, tree, camera, lasso, { selectionMode: mode, selectWholeModel: false, useBoundsTree: false });
        const bruteMs = performance.now() - start;
        const a = [...withTree].sort((x, y) => x - y);
        const b = [...without].sort((x, y) => x - y);
        if (a.length !== b.length || a.some((value, i) => value !== b[i])) throw new Error(`${mode}: ${a.length / 3} triangles through the tree, ${b.length / 3} without it`);
        if (a.length < 300) throw new Error(`${mode}: only ${a.length / 3} triangles selected`);
        out[`${mode} triangles`] = a.length / 3;
        out[`${mode} brute ms`] = bruteMs;
    }
    if (out['centroid-visible triangles'] >= out['centroid triangles']) throw new Error('centroid-visible should drop the hidden triangles centroid keeps');
    updateSelection();
    return out;
}

let frame = 0;
app.loop(() => {
    frame++;
    if (selectionShapeNeedsUpdate) {
        const points = tool.points;
        selectionShape.geometry.setAttribute('position', new Float32BufferAttribute([...points, ...points.slice(0, 3)], 3));
        selectionShapeNeedsUpdate = false;
    }
    if (selectionNeedsUpdate) {
        selectionNeedsUpdate = false;
        if (tool.points.length > 0) updateSelection();
    }
    const yScale = Math.tan((MathUtils.DEG2RAD * camera.fov) / 2) * selectionShape.position.z;
    selectionShape.scale.set(-yScale * camera.aspect, -yScale, 1);
    highlightWireframe.visible = true;
    if (params.rotate) {
        group.rotation.y += 0.01;
        if (params.liveUpdate && tool.dragging) selectionNeedsUpdate = true;
    }
    if (frame % 10 === 0) {
        overlay.set('selected', `${selectedTriangles.toLocaleString('en-US')} triangles`);
        overlay.set('selection', `${selectMs.toFixed(3)} ms`);
    }
    if (isSmoke && frame === 10) {
        params.rotate = false;
        void ready(app.device, { ...overlay.snapshot(), ...selectionCheck() });
    }
});

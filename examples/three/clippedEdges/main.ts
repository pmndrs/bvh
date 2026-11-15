/**
 * three/clippedEdges: a port of three-mesh-bvh's `clippedEdges` example.
 * Source: three-mesh-bvh's example/clippedEdges.js (v0.9.18, MIT, Garrett Johnson); see THIRD_PARTY.md.
 *
 * What it demonstrates: `tree.shapecast` with a plane as the shape. A
 * clipping plane sweeps through the model; every frame `intersectsBounds`
 * keeps the boxes the plane crosses, `intersectsTriangle` turns each crossed
 * triangle into one line segment, and those segments are the outline of the
 * cut. The cap is three's stencil trick (front faces count up, back faces
 * count down, the plane draws where the count is not zero).
 *
 * "use the BVH" off returns `CONTAINED` for the root box, so the same
 * callbacks visit every triangle: the time is the whole difference, and the
 * smoke run checks that both ways produce the same segments.
 *
 * The model is a torus knot (size select); `?model=<glTF>` or a dropped glTF
 * is merged into one geometry and clipped instead.
 */

import {
    BufferAttribute,
    BufferGeometry,
    DecrementWrapStencilOp,
    DoubleSide,
    BackSide,
    FrontSide,
    Group,
    IncrementWrapStencilOp,
    Line3,
    LineBasicMaterial,
    LineSegments,
    Matrix4,
    Mesh,
    MeshBasicMaterial,
    MeshStandardMaterial,
    NotEqualStencilFunc,
    PlaneGeometry,
    Plane,
    TorusKnotGeometry,
    Vector3,
    ZeroStencilOp,
    type Object3D,
} from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { CONTAINED, MeshBVHHelper, prepareBoundsTrees, computeBoundsTree, disposeBoundsTree, type BvhBoundsTree, type ExtendedTriangle } from '@pmndrs/bvh/three';
import { createApp } from '../../shared/app.ts';
import { enableDrop, loadGltf, meshesOf, requestedModel, type LoadedModel } from '../../shared/loader.ts';
import { layoutInfo } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

const app = await createApp('Clipped edges', 'three/clippedEdges', { stencil: true });
const { device, scene, camera, controls, overlay } = app;
await prepareBoundsTrees();
camera.position.set(0.9, 0.7, 0.9);
controls.update();

const SIZES = { '20k': [200, 50], '100k': [500, 100], '500k': [1000, 250] } as const;
type Size = keyof typeof SIZES;
const MAX_SEGMENTS = 300_000;

const params = { useBVH: true, animate: !isSmoke, animation: 'SPIN' as 'SPIN' | 'OSCILLATE', invert: false, displayModel: true, displayBVH: false, displayDepth: 10 };
let size: Size = isSmoke ? '20k' : '100k';

const clippingPlane = new Plane();
const clippingPlanes = [clippingPlane];
const localPlane = new Plane();
const inverseMatrix = new Matrix4();

// The cap: drawn where the stencil count is not zero, then zeroed.
const planeMesh = new Mesh(
    new PlaneGeometry(),
    new MeshBasicMaterial({ color: 0x80deea, side: DoubleSide, stencilWrite: true, stencilFunc: NotEqualStencilFunc, stencilFail: ZeroStencilOp, stencilZFail: ZeroStencilOp, stencilZPass: ZeroStencilOp }),
);
planeMesh.scale.setScalar(1.5);
planeMesh.renderOrder = 2;
scene.add(planeMesh);

// The outline: room for MAX_SEGMENTS segments, only the cut's are drawn.
const outlineGeometry = new BufferGeometry();
const outlinePositions = new Float32Array(MAX_SEGMENTS * 6);
outlineGeometry.setAttribute('position', new BufferAttribute(outlinePositions, 3));
const outline = new LineSegments(outlineGeometry, new LineBasicMaterial({ color: 0x4dd0e1 }));
outline.frustumCulled = false;
outline.renderOrder = 3;

let group: Group | null = null;
let colliderMesh: Mesh | null = null;
let tree: BvhBoundsTree | null = null;
let helper: MeshBVHHelper | null = null;
let frontModel: Mesh | null = null;
let backModel: Mesh | null = null;
let lastCut = { segments: 0, ms: 0 };

/** One mesh for the whole glTF, in the root's space (the example clips one geometry). */
function mergedGeometry(root: Object3D): BufferGeometry {
    root.updateMatrixWorld(true);
    const parts = meshesOf(root).map((mesh) => {
        const part = mesh.geometry.clone().applyMatrix4(mesh.matrixWorld);
        for (const name of Object.keys(part.attributes)) if (name !== 'position' && name !== 'normal') part.deleteAttribute(name);
        if (!part.attributes.normal) part.computeVertexNormals();
        return part;
    });
    return mergeGeometries(parts, false);
}

function useGeometry(geometry: BufferGeometry, label: string): void {
    if (group) {
        scene.remove(group);
        helper?.dispose();
        if (colliderMesh) disposeBoundsTree.call(colliderMesh.geometry);
    }
    // Centred and scaled to one unit across, so the plane (1.5 wide) always spans it.
    geometry.computeBoundingBox();
    geometry.center();
    geometry.computeBoundingSphere();
    geometry.scale(0.5 / geometry.boundingSphere!.radius, 0.5 / geometry.boundingSphere!.radius, 0.5 / geometry.boundingSphere!.radius);

    const start = performance.now();
    tree = computeBoundsTree.call(geometry, { targetLeafSize: 3 });
    const buildMs = performance.now() - start;
    overlay.tree(layoutInfo(tree.layout));
    overlay.set('model', label);
    overlay.set('tree build (tinybvh)', `${buildMs.toFixed(1)} ms`);

    // Front faces count up, back faces count down, both clipped by the plane.
    const front = new MeshStandardMaterial({ color: 0xffffff, roughness: 0.6, side: FrontSide, clippingPlanes, stencilWrite: true, stencilFail: IncrementWrapStencilOp, stencilZFail: IncrementWrapStencilOp, stencilZPass: IncrementWrapStencilOp });
    const back = new MeshBasicMaterial({ colorWrite: false, depthWrite: false, side: BackSide, clippingPlanes, stencilWrite: true, stencilFail: DecrementWrapStencilOp, stencilZFail: DecrementWrapStencilOp, stencilZPass: DecrementWrapStencilOp });
    frontModel = new Mesh(geometry, front);
    backModel = new Mesh(geometry, back);
    colliderMesh = new Mesh(geometry, new MeshBasicMaterial({ visible: false }));
    helper = new MeshBVHHelper(colliderMesh, params.displayDepth);
    group = new Group();
    group.add(frontModel, backModel, colliderMesh, helper, outline);
    // A model transform, so the plane has to be taken into the model's space like in the original.
    group.rotation.set(0.2, 0.4, 0);
    group.updateMatrixWorld(true);
    scene.add(group);
    time = 0;
    cutNow();
}

let time = 0;

/** The plane's intersection with every triangle the tree can't rule out, as segments in `outlinePositions`. */
function cut(useBVH: boolean): { segments: number; ms: number; sum: number } {
    const start = performance.now();
    let index = 0; // points written
    const line = new Line3();
    const point = new Vector3();
    const p1 = new Vector3();
    const p2 = new Vector3();
    const p3 = new Vector3();
    const write = (v: Vector3): void => {
        if (index < MAX_SEGMENTS * 2) outlinePositions.set([v.x, v.y, v.z], index * 3);
        index++;
    };
    const read = (i: number, v: Vector3): Vector3 => v.fromArray(outlinePositions, i * 3);
    tree!.shapecast({
        intersectsBounds: (box) => (useBVH ? localPlane.intersectsBox(box) : CONTAINED),
        intersectsTriangle: (tri: ExtendedTriangle) => {
            // Each triangle edge that crosses the plane adds a point; a triangle must give exactly two.
            let count = 0;
            for (const [a, b] of [[tri.a, tri.b], [tri.b, tri.c], [tri.c, tri.a]] as const) {
                line.start.copy(a);
                line.end.copy(b);
                if (localPlane.intersectLine(line, point)) {
                    write(point);
                    count++;
                }
            }
            // A plane through a vertex hits two edges at the same point: drop the repeat.
            if (count === 3) {
                read(index - 3, p1);
                read(index - 2, p2);
                read(index - 1, p3);
                if (p3.equals(p1) || p3.equals(p2)) {
                    count--;
                    index--;
                } else if (p1.equals(p2)) {
                    outlinePositions.set([p3.x, p3.y, p3.z], (index - 2) * 3);
                    count--;
                    index--;
                }
            }
            // One or three crossings (a vertex touching the plane) give no segment.
            if (count !== 2) index -= count;
            return false;
        },
    });
    const ms = performance.now() - start;
    let sum = 0;
    for (let i = 0; i < index * 3; i++) sum += Math.abs(outlinePositions[i]);
    return { segments: index / 2, ms, sum };
}

function cutNow(): void {
    if (!tree || !colliderMesh) return;
    colliderMesh.updateMatrixWorld(true);
    // The plane in the model's space.
    inverseMatrix.copy(colliderMesh.matrixWorld).invert();
    localPlane.copy(clippingPlane).applyMatrix4(inverseMatrix);
    const result = cut(params.useBVH);
    lastCut = result;
    outlineGeometry.setDrawRange(0, result.segments * 2);
    outlineGeometry.attributes.position.needsUpdate = true;
    // Offset the lines a hair so they don't z-fight with the cap.
    outline.position.copy(clippingPlane.normal).multiplyScalar(-0.00001);
    overlay.set('outline', `${result.segments.toLocaleString('en-US')} segments, ${result.ms.toFixed(3)} ms ${params.useBVH ? '(BVH)' : '(every triangle)'}`);
}

function setPlane(): void {
    if (params.animation === 'SPIN') {
        planeMesh.rotation.set(0.25 * time, 0.25 * time, 0.25 * time);
        planeMesh.position.set(0, 0, 0);
    } else {
        planeMesh.position.set(Math.sin(0.25 * time) * 0.325, 0, 0);
        planeMesh.rotation.set(0, Math.PI / 2, 0);
    }
    planeMesh.updateMatrixWorld();
    clippingPlane.normal.set(0, 0, params.invert ? 1 : -1);
    clippingPlane.constant = 0;
    clippingPlane.applyMatrix4(planeMesh.matrixWorld);
}

overlay.checkbox('use the BVH', params.useBVH, (v) => ((params.useBVH = v), cutNow()));
overlay.checkbox('animate', params.animate, (v) => (params.animate = v));
overlay.select('animation', ['SPIN', 'OSCILLATE'] as const, params.animation, (v) => ((params.animation = v), (time = 0)));
overlay.checkbox('invert', params.invert, (v) => ((params.invert = v), setPlane(), cutNow()));
overlay.checkbox('display model', params.displayModel, (v) => (params.displayModel = v));
overlay.checkbox('display BVH', params.displayBVH, (v) => (params.displayBVH = v));
overlay.range('BVH depth', 1, 20, 1, params.displayDepth, (v) => {
    params.displayDepth = v;
    if (helper) {
        helper.depth = v;
        helper.update();
    }
});
overlay.select('model size', Object.keys(SIZES) as Size[], size, (v) => {
    size = v;
    useGeometry(knotGeometry(), `torus knot, ${size}`);
});
overlay.note('Demonstrates tree.shapecast with a plane: intersectsBounds keeps the boxes the plane crosses, intersectsTriangle emits one segment per crossed triangle. Switch "use the BVH" off to visit every triangle. Drop a glTF to clip it.');

function knotGeometry(): BufferGeometry {
    const [tubular, radial] = SIZES[size];
    return new TorusKnotGeometry(0.5, 0.15, tubular, radial);
}

const model = requestedModel();
if (model) {
    const loaded = await loadGltf(model);
    useGeometry(mergedGeometry(loaded.root), loaded.name);
} else {
    useGeometry(knotGeometry(), `torus knot, ${size}`);
}
enableDrop(
    (loaded: LoadedModel) => useGeometry(mergedGeometry(loaded.root), loaded.name),
    (error) => overlay.error(String(error)),
);

let frames = 0;
let smokeDone = false;
app.loop(async (_now, delta) => {
    if (helper && frontModel && backModel) {
        helper.visible = params.displayBVH;
        frontModel.visible = backModel.visible = params.displayModel;
    }
    if (params.animate) {
        time += delta;
        setPlane();
        cutNow();
    } else if (frames === 0) {
        setPlane();
        cutNow();
    }
    frames++;
    if (isSmoke && frames > 5 && !smokeDone) {
        smokeDone = true;
        // The same cut with and without the tree: same segments.
        time = 1.7;
        setPlane();
        colliderMesh!.updateMatrixWorld(true);
        inverseMatrix.copy(colliderMesh!.matrixWorld).invert();
        localPlane.copy(clippingPlane).applyMatrix4(inverseMatrix);
        const withTree = cut(true);
        const without = cut(false);
        if (withTree.segments === 0) throw new Error('the plane cut no triangles');
        if (withTree.segments !== without.segments) throw new Error(`BVH cut ${withTree.segments} segments, every-triangle cut ${without.segments}`);
        if (Math.abs(withTree.sum - without.sum) > 1e-6 * without.sum) throw new Error(`segment coordinates differ: ${withTree.sum} vs ${without.sum}`);
        overlay.set('smoke', `BVH ${withTree.ms.toFixed(2)} ms, all ${without.ms.toFixed(2)} ms, ${withTree.segments} segments`);
        await ready(device, { ...overlay.snapshot(), segments: withTree.segments, bvhMs: withTree.ms, bruteMs: without.ms, lastCut });
    }
});

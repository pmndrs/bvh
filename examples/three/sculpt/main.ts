/**
 * three/sculpt: three-mesh-bvh's `sculpt` example on our tree.
 *
 * A welded icosphere (`mergeVertices`) with a tree. Under the pointer, the
 * brush is placed by a raycast (`acceleratedRaycast`, `firstHitOnly`); while a
 * button is down each stroke step asks the tree for the vertices inside the
 * brush sphere (`shapecast`: boxes answered `CONTAINED`, `INTERSECTED` or
 * `NOT_INTERSECTED`, a contained box's triangles taken whole), moves them
 * (clay, normal or flatten brush; right button or "invert" pushes in), mends
 * their normals and refits the tree so the next raycast sees the new surface.
 * Symmetry mirrors each step across x.
 *
 * One difference: three-mesh-bvh refits only the nodes the strokes visited
 * (`refit(nodeIndices)`); our `refit()` refits the whole tree (one flat pass,
 * `CpuMeshBvh.refit`), so the refit ms shown is the full tree's. The page
 * draws with a generated matcap (the original loads matcap images).
 *
 * Ported from three-mesh-bvh's example/sculpt.js (v0.9.18, MIT, Garrett
 * Johnson); see THIRD_PARTY.md. Changed: our shell and overlay, the matcap,
 * the whole-tree refit, the smoke check.
 */

import { BufferGeometry, CanvasTexture, DynamicDrawUsage, IcosahedronGeometry, LineBasicMaterial, LineSegments, Matrix4, Mesh, MeshMatcapMaterial, Plane, Raycaster, Sphere, SRGBColorSpace, Vector2, Vector3, type BufferAttribute, type Intersection } from 'three';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { acceleratedRaycast, computeBoundsTree, CONTAINED, INTERSECTED, MeshBVHHelper, NOT_INTERSECTED, prepareBoundsTrees, validateBounds, type BvhBoundsTree } from '@pmndrs/bvh/three';
import { createApp } from '../../shared/app.ts';
import { layoutInfo } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

await prepareBoundsTrees();
// three's own raycast, kept for the smoke check, then the tree's.
const threeRaycast = Mesh.prototype.raycast;
Mesh.prototype.raycast = acceleratedRaycast;
const app = await createApp('Sculpt', 'three/sculpt');
const { renderer, scene, camera, controls, overlay } = app;
camera.position.set(0, 0, 3);
controls.target.set(0, 0, 0);
controls.minDistance = 1.5;
renderer.domElement.style.touchAction = 'none';

const params = { matcap: 'clay' as 'clay' | 'red wax' | 'shiny green' | 'normal', size: 0.1, brush: 'clay' as 'normal' | 'clay' | 'flatten', intensity: 50, maxSteps: 10, invert: false, symmetrical: true, flatShading: false, detail: isSmoke ? 40 : 100 };

/** A matcap drawn on a canvas: a lit sphere in the given colours (the original loads four images). */
function matcap(dark: string, mid: string, light: string): CanvasTexture {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 256;
    const g = canvas.getContext('2d')!;
    const gradient = g.createRadialGradient(100, 90, 8, 128, 128, 128);
    gradient.addColorStop(0, light);
    gradient.addColorStop(0.45, mid);
    gradient.addColorStop(1, dark);
    g.fillStyle = gradient;
    g.fillRect(0, 0, 256, 256);
    const texture = new CanvasTexture(canvas);
    texture.colorSpace = SRGBColorSpace;
    return texture;
}
const matcaps = { clay: matcap('#4b2e2a', '#b67f6b', '#f3dbc6'), 'red wax': matcap('#210504', '#763c39', '#d8a090'), 'shiny green': matcap('#2a3f10', '#88ac2e', '#e3f2c3'), normal: matcap('#1c78c0', '#7877ee', '#75d9c7') };
const material = new MeshMatcapMaterial({ matcap: matcaps.clay, flatShading: params.flatShading });

let targetMesh: Mesh<BufferGeometry, MeshMatcapMaterial>;
let tree: BvhBoundsTree;
let helper: MeshBVHHelper | null = null;
let showHelper = false;
let refitMs = 0;

function reset(): void {
    if (targetMesh) {
        targetMesh.geometry.dispose();
        scene.remove(targetMesh);
    }
    // Welded, because three's icosahedron repeats vertices along its seams.
    let geometry: BufferGeometry = new IcosahedronGeometry(1, params.detail);
    geometry.deleteAttribute('uv');
    geometry = mergeVertices(geometry);
    (geometry.attributes.position as BufferAttribute).setUsage(DynamicDrawUsage);
    (geometry.attributes.normal as BufferAttribute).setUsage(DynamicDrawUsage);
    const start = performance.now();
    tree = computeBoundsTree.call(geometry, { setBoundingBox: false });
    overlay.build({ builtBy: 'tinybvh (computeBoundsTree)', cpuMs: performance.now() - start });
    overlay.tree(layoutInfo(tree.layout));
    targetMesh = new Mesh(geometry, material);
    targetMesh.frustumCulled = false;
    scene.add(targetMesh);
    if (helper) helper.removeFromParent();
    helper = new MeshBVHHelper(targetMesh, 10);
    helper.visible = showHelper;
    scene.add(helper);
}
reset();

// The brush cursor: a circle and its normal.
function brushCursor(): LineSegments {
    const points = [new Vector3(), new Vector3(0, 0, 1)];
    for (let i = 0; i < 50; i++) {
        points.push(new Vector3(Math.sin((2 * Math.PI * i) / 50), Math.cos((2 * Math.PI * i) / 50), 0));
        points.push(new Vector3(Math.sin((2 * Math.PI * (i + 1)) / 50), Math.cos((2 * Math.PI * (i + 1)) / 50), 0));
    }
    const cursor = new LineSegments(new BufferGeometry().setFromPoints(points), new LineBasicMaterial({ color: 0xfb8c00 }));
    return cursor;
}
const brush = brushCursor();
const symmetryBrush = brushCursor();
scene.add(brush, symmetryBrush);

interface Accumulated {
    readonly triangles: Set<number>;
    readonly indices: Set<number>;
}

const normalZ = new Vector3(0, 0, 1);
/** One brush step at `point` (world space): with `brushOnly`, only orients the cursor. */
function performStroke(point: Vector3, brushObject: LineSegments, brushOnly: boolean, accumulated: Accumulated | null, rightClick: boolean): void {
    const inverseMatrix = new Matrix4().copy(targetMesh.matrixWorld).invert();
    const sphere = new Sphere(point.clone().applyMatrix4(inverseMatrix), params.size);
    const indices = new Set<number>();
    const tempVec = new Vector3();
    const normal = new Vector3();
    const indexAttr = targetMesh.geometry.index!;
    const posAttr = targetMesh.geometry.attributes.position as BufferAttribute;
    const normalAttr = targetMesh.geometry.attributes.normal as BufferAttribute;
    tree.shapecast({
        intersectsBounds: (box) => {
            if (!sphere.intersectsBox(box)) return NOT_INTERSECTED;
            const { min, max } = box;
            for (let x = 0; x <= 1; x++) {
                for (let y = 0; y <= 1; y++) {
                    for (let z = 0; z <= 1; z++) {
                        tempVec.set(x === 0 ? min.x : max.x, y === 0 ? min.y : max.y, z === 0 ? min.z : max.z);
                        if (!sphere.containsPoint(tempVec)) return INTERSECTED;
                    }
                }
            }
            return CONTAINED;
        },
        intersectsTriangle: (triangle, index, contained) => {
            accumulated?.triangles.add(index);
            const va = indexAttr.getX(3 * index);
            const vb = indexAttr.getX(3 * index + 1);
            const vc = indexAttr.getX(3 * index + 2);
            const take = (vertex: number) => {
                indices.add(vertex);
                accumulated?.indices.add(vertex);
            };
            if (contained) {
                take(va);
                take(vb);
                take(vc);
            } else {
                if (sphere.containsPoint(triangle.a)) take(va);
                if (sphere.containsPoint(triangle.b)) take(vb);
                if (sphere.containsPoint(triangle.c)) take(vc);
            }
            return false;
        },
    });

    // The mean normal (and point, to flatten towards) of the vertices inside.
    const localPoint = point.clone().applyMatrix4(inverseMatrix);
    const planePoint = new Vector3();
    let totalPoints = 0;
    for (const index of indices) {
        normal.add(tempVec.fromBufferAttribute(normalAttr, index));
        if (!brushOnly) {
            totalPoints++;
            planePoint.add(tempVec.fromBufferAttribute(posAttr, index));
        }
    }
    normal.normalize();
    brushObject.quaternion.setFromUnitVectors(normalZ, normal);
    if (totalPoints) planePoint.multiplyScalar(1 / totalPoints);
    if (brushOnly) return;

    const targetHeight = params.intensity * 0.0001;
    const plane = new Plane().setFromNormalAndCoplanarPoint(normal, planePoint);
    for (const index of indices) {
        tempVec.fromBufferAttribute(posAttr, index);
        const dist = tempVec.distanceTo(localPoint);
        const negated = params.invert !== rightClick ? -1 : 1;
        let intensity = 1 - dist / params.size;
        if (params.brush === 'clay') {
            intensity = intensity ** 3;
            const planeDist = plane.distanceToPoint(tempVec);
            const clamped = negated * Math.min(intensity * 4, 1);
            tempVec.addScaledVector(normal, clamped * targetHeight - negated * planeDist * clamped * 0.3);
        } else if (params.brush === 'normal') {
            intensity = intensity ** 2;
            tempVec.addScaledVector(normal, negated * intensity * targetHeight);
        } else {
            intensity = intensity ** 2;
            const planeDist = plane.distanceToPoint(tempVec);
            tempVec.addScaledVector(normal, -planeDist * intensity * params.intensity * 0.01 * 0.5);
        }
        posAttr.setXYZ(index, tempVec.x, tempVec.y, tempVec.z);
        normalAttr.setXYZ(index, 0, 0, 0);
    }
    if (indices.size) posAttr.needsUpdate = true;
}

/** The touched vertices' normals, from their triangles. */
function updateNormals(triangles: Set<number>, indices: Set<number>): void {
    const indexAttr = targetMesh.geometry.index!;
    const posAttr = targetMesh.geometry.attributes.position as BufferAttribute;
    const normalAttr = targetMesh.geometry.attributes.normal as BufferAttribute;
    const a = new Vector3();
    const b = new Vector3();
    const c = new Vector3();
    const faceNormal = new Vector3();
    const tempVec = new Vector3();
    for (const triangle of triangles) {
        const v = [indexAttr.getX(3 * triangle), indexAttr.getX(3 * triangle + 1), indexAttr.getX(3 * triangle + 2)];
        a.fromBufferAttribute(posAttr, v[0]);
        b.fromBufferAttribute(posAttr, v[1]);
        c.fromBufferAttribute(posAttr, v[2]);
        faceNormal.subVectors(c, b).cross(tempVec.subVectors(a, b)).normalize();
        for (const vertex of v) {
            if (!indices.has(vertex)) continue;
            tempVec.fromBufferAttribute(normalAttr, vertex).add(faceNormal);
            normalAttr.setXYZ(vertex, tempVec.x, tempVec.y, tempVec.z);
        }
    }
    for (const index of indices) {
        tempVec.fromBufferAttribute(normalAttr, index).normalize();
        normalAttr.setXYZ(index, tempVec.x, tempVec.y, tempVec.z);
    }
    normalAttr.needsUpdate = true;
}

/** The moved vertices' tree: our refit is whole-tree (three-mesh-bvh refits only the nodes the strokes visited). */
function refit(): void {
    const start = performance.now();
    tree.refit();
    refitMs = performance.now() - start;
    if (helper?.visible) helper.update();
}

// ---- Pointer state, as the original.
const mouse = new Vector2();
const lastMouse = new Vector2();
let mouseState = false;
let lastMouseState = false;
let rightClick = false;
let brushActive = false;
let controlsActive = false;
const lastCastPose = new Vector3(Infinity, Infinity, Infinity);
const raycaster = new Raycaster();
(raycaster as unknown as { firstHitOnly: boolean }).firstHitOnly = true;
const toNdc = (event: PointerEvent) => mouse.set((event.clientX / innerWidth) * 2 - 1, -(event.clientY / innerHeight) * 2 + 1);
addEventListener('pointermove', (event) => {
    toNdc(event);
    brushActive = true;
});
addEventListener(
    'pointerdown',
    (event) => {
        if (event.target !== renderer.domElement) return;
        toNdc(event);
        mouseState = Boolean(event.buttons & 3);
        rightClick = Boolean(event.buttons & 2);
        brushActive = true;
        raycaster.setFromCamera(mouse, camera);
        controls.enabled = raycaster.intersectObject(targetMesh).length === 0;
    },
    true,
);
addEventListener('pointerup', (event) => {
    mouseState = Boolean(event.buttons & 3);
    if (event.pointerType === 'touch') brushActive = false;
});
addEventListener('contextmenu', (event) => event.preventDefault());
addEventListener('wheel', (event) => {
    const delta = event.deltaMode === 0 ? event.deltaY : event.deltaY * 40;
    params.size = Math.max(Math.min(params.size + delta * 0.0001, 0.25), 0.025);
    sizeInput.value = String(params.size);
});
controls.addEventListener('start', () => (controlsActive = true));
controls.addEventListener('end', () => (controlsActive = false));

overlay.select('matcap', ['clay', 'red wax', 'shiny green', 'normal'] as const, params.matcap, (value) => {
    params.matcap = value;
    material.matcap = matcaps[value];
    material.needsUpdate = true;
});
overlay.select('brush', ['normal', 'clay', 'flatten'] as const, params.brush, (value) => (params.brush = value));
const sizeInput = overlay.range('size', 0.025, 0.25, 0.005, params.size, (value) => (params.size = value));
overlay.range('intensity', 1, 100, 1, params.intensity, (value) => (params.intensity = value));
overlay.range('max steps', 1, 25, 1, params.maxSteps, (value) => (params.maxSteps = value));
overlay.checkbox('symmetrical', params.symmetrical, (value) => (params.symmetrical = value));
overlay.checkbox('invert', params.invert, (value) => (params.invert = value));
overlay.checkbox('flat shading', params.flatShading, (value) => {
    material.flatShading = value;
    material.needsUpdate = true;
});
overlay.checkbox('display BVH', showHelper, (value) => {
    showHelper = value;
    if (helper) {
        helper.visible = value;
        helper.update();
    }
});
overlay.range('BVH depth', 1, 20, 1, 10, (value) => {
    if (!helper) return;
    helper.depth = value;
    helper.update();
});
overlay.button('reset', reset);
overlay.button('rebuild BVH', () => {
    tree = computeBoundsTree.call(targetMesh.geometry, { setBoundingBox: false });
    helper?.update();
});
overlay.note('Drag on the sphere to sculpt (right button or "invert" pushes in), drag off it to orbit, wheel for the brush size.');

/** The stroke the original runs in its render loop, between the last cast point and the hit under the pointer. */
function strokeTowards(hit: Intersection): number {
    const mdx = (mouse.x - lastMouse.x) * innerWidth * devicePixelRatio;
    const mdy = (mouse.y - lastMouse.y) * innerHeight * devicePixelRatio;
    let mdist = Math.sqrt(mdx * mdx + mdy * mdy);
    let castDist = hit.point.distanceTo(lastCastPose);
    const step = params.size * 0.15;
    const percent = Math.max(step / castDist, 1 / params.maxSteps);
    const mstep = mdist * percent;
    let stepCount = 0;
    const accumulated: Accumulated = { triangles: new Set(), indices: new Set() };
    while (castDist > step && mdist > (params.size * 200) / hit.distance) {
        lastMouse.lerp(mouse, percent);
        lastCastPose.lerp(hit.point, percent);
        castDist -= step;
        mdist -= mstep;
        performStroke(lastCastPose, brush, false, accumulated, rightClick);
        if (params.symmetrical) {
            lastCastPose.x *= -1;
            performStroke(lastCastPose, symmetryBrush, false, accumulated, rightClick);
            lastCastPose.x *= -1;
        }
        if (++stepCount > params.maxSteps) break;
    }
    if (stepCount > 0) {
        updateNormals(accumulated.triangles, accumulated.indices);
        refit();
    }
    return stepCount;
}

/** Smoke: strokes across the sphere move vertices outward, the refit tree is valid, and raycasts agree with three's own. */
function sculptCheck(): Record<string, number> {
    const before = (targetMesh.geometry.attributes.position.array as Float32Array).slice();
    params.brush = 'normal';
    const accumulated: Accumulated = { triangles: new Set(), indices: new Set() };
    for (let i = 0; i < 40; i++) {
        const angle = (i / 40) * Math.PI * 0.8 - 0.4;
        performStroke(new Vector3(Math.sin(angle), 0.2, Math.cos(angle)).normalize(), brush, false, accumulated, false);
    }
    updateNormals(accumulated.triangles, accumulated.indices);
    refit();
    const after = targetMesh.geometry.attributes.position.array as Float32Array;
    let moved = 0;
    for (let i = 0; i < after.length; i++) if (after[i] !== before[i]) moved++;
    if (moved < 30) throw new Error(`the strokes moved only ${moved} coordinates`);
    if (!validateBounds(tree)) throw new Error('the refit tree does not bound its triangles');
    // Rays at the sculpted band: our refit tree's first hit against three's Mesh.raycast (no tree).
    const plain = new Mesh(targetMesh.geometry.clone(), material);
    plain.raycast = threeRaycast;
    let agreed = 0;
    for (let i = 0; i < 64; i++) {
        const angle = (i / 64) * Math.PI * 0.8 - 0.4;
        const target = new Vector3(Math.sin(angle), 0.2 + ((i % 5) - 2) * 0.03, Math.cos(angle));
        raycaster.set(target.clone().multiplyScalar(3), target.clone().negate().normalize());
        const ours = raycaster.intersectObject(targetMesh)[0];
        const theirs = raycaster.intersectObject(plain)[0];
        if (!ours || !theirs || Math.abs(ours.distance - theirs.distance) > 1e-5) throw new Error(`ray ${i}: ours ${ours?.distance}, three ${theirs?.distance}`);
        agreed++;
    }
    return { moved, agreedWithThree: agreed, refitMs };
}

let frame = 0;
app.loop(() => {
    frame++;
    if (controlsActive || !brushActive) {
        brush.visible = symmetryBrush.visible = false;
        lastCastPose.setScalar(Infinity);
    } else {
        raycaster.setFromCamera(mouse, camera);
        const hit = raycaster.intersectObject(targetMesh, true)[0];
        if (hit) {
            brush.visible = true;
            brush.scale.set(params.size, params.size, 0.1);
            brush.position.copy(hit.point);
            symmetryBrush.visible = params.symmetrical;
            symmetryBrush.scale.set(params.size, params.size, 0.1);
            symmetryBrush.position.copy(hit.point);
            symmetryBrush.position.x *= -1;
            controls.enabled = false;
            if (lastCastPose.x === Infinity) lastCastPose.copy(hit.point);
            if (!(mouseState || lastMouseState)) {
                performStroke(hit.point, brush, true, null, rightClick);
                if (params.symmetrical) performStroke(hit.point.clone().setX(-hit.point.x), symmetryBrush, true, null, rightClick);
                lastMouse.copy(mouse);
                lastCastPose.copy(hit.point);
            } else if (strokeTowards(hit) === 0) {
                performStroke(hit.point, brush, true, null, rightClick);
                if (params.symmetrical) performStroke(hit.point.clone().setX(-hit.point.x), symmetryBrush, true, null, rightClick);
            }
        } else {
            controls.enabled = true;
            brush.visible = symmetryBrush.visible = false;
            lastMouse.copy(mouse);
            lastCastPose.setScalar(Infinity);
        }
    }
    lastMouseState = mouseState;
    if (frame % 10 === 0) {
        overlay.set('triangles', (targetMesh.geometry.index!.count / 3).toLocaleString('en-US'));
        overlay.set('refit (whole tree)', `${refitMs.toFixed(2)} ms`);
    }
    if (isSmoke && frame === 10) void ready(app.device, { ...overlay.snapshot(), ...sculptCheck() });
});

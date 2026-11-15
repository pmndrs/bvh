/**
 * three/edgeIntersect: a port of three-mesh-bvh's `edgeIntersect` example.
 * Source: three-mesh-bvh's example/edgeIntersect.js (v0.9.18, MIT, Garrett Johnson); see THIRD_PARTY.md.
 *
 * What it demonstrates: `tree.bvhcast(otherTree, matrixToLocal, { intersectsTriangles })`,
 * a walk of two trees against each other. Two torus knots spin around each
 * other; for every pair of triangles whose boxes overlap, `ExtendedTriangle.intersectsTriangle`
 * gives the segment where they cross, and the segments are drawn as the red
 * line where the surfaces meet. Both meshes share one geometry and one tree.
 *
 * The smoke run (small knots) checks the segments against every pair of
 * triangles.
 */

import { BufferAttribute, BufferGeometry, CustomBlending, DoubleSide, Group, Line3, LineBasicMaterial, LineSegments, Matrix4, Mesh, MeshBasicMaterial, MeshPhongMaterial, TorusKnotGeometry } from 'three';
import { ExtendedTriangle, SAH, computeBoundsTree, prepareBoundsTrees, type BvhBoundsTree } from '@pmndrs/bvh/three';
import { createApp } from '../../shared/app.ts';
import { layoutInfo } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

const app = await createApp('Edge intersect', 'three/edgeIntersect');
const { device, scene, camera, overlay } = app;
await prepareBoundsTrees();
camera.position.set(2, 2, 2);

const SIZES = { '1.6k': [40, 20], '8k': [100, 40], '40k': [250, 80] } as const;
type Size = keyof typeof SIZES;
const params = { speed: isSmoke ? 0 : 1, distance: 1.1, displayMeshes: false };
let size: Size = isSmoke ? '1.6k' : '8k';

const material = new MeshPhongMaterial({ color: 0xffffff, side: DoubleSide, shininess: 0.01, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1, opacity: 0.5, blending: CustomBlending });
const depthMaterial = new MeshBasicMaterial({ polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1, colorWrite: false });

const group1 = new Group();
const group2 = new Group();
scene.add(group1, group2);
let geometry: BufferGeometry;
let tree: BvhBoundsTree;
let mesh1: Mesh;
let mesh2: Mesh;

const lineGeometry = new BufferGeometry();
lineGeometry.setAttribute('position', new BufferAttribute(new Float32Array(6), 3));
const line = new LineSegments(lineGeometry, new LineBasicMaterial({ color: 0xe88baa }));
line.frustumCulled = false;
const lineGroup = new Group();
lineGroup.add(line);
lineGroup.visible = false;
scene.add(lineGroup);

function makeKnots(): void {
    for (const group of [group1, group2]) group.clear();
    geometry?.dispose();
    const [tubular, radial] = SIZES[size];
    geometry = new TorusKnotGeometry(1, 0.4, tubular, radial);
    const start = performance.now();
    tree = computeBoundsTree.call(geometry, { targetLeafSize: 1, strategy: SAH });
    overlay.set('tree build (tinybvh)', `${(performance.now() - start).toFixed(1)} ms`);
    overlay.tree(layoutInfo(tree.layout));
    overlay.set('triangles per knot', tree.layout.triangleCount);
    mesh1 = new Mesh(geometry, material);
    mesh1.renderOrder = 2;
    mesh2 = new Mesh(geometry, material);
    mesh2.renderOrder = 2;
    const depth1 = new Mesh(geometry, depthMaterial);
    const depth2 = new Mesh(geometry, depthMaterial);
    depth1.renderOrder = depth2.renderOrder = 1;
    group1.add(depth1, mesh1);
    group2.add(depth2, mesh2);
}

const edge = new Line3();
const matrix2to1 = new Matrix4();
let segments: number[] = [];

/** Segments (in mesh1's space) where mesh2's triangles cross mesh1's, found by walking both trees. */
function intersectKnots(): { count: number; ms: number } {
    scene.updateMatrixWorld(true);
    matrix2to1.copy(mesh1.matrixWorld).invert().multiply(mesh2.matrixWorld);
    segments = [];
    const start = performance.now();
    tree.bvhcast(tree, matrix2to1, {
        intersectsTriangles(triangle1: ExtendedTriangle, triangle2: ExtendedTriangle) {
            if (triangle1.intersectsTriangle(triangle2, edge)) {
                const { start: s, end: e } = edge;
                segments.push(s.x, s.y, s.z, e.x, e.y, e.z);
            }
        },
    });
    return { count: segments.length / 6, ms: performance.now() - start };
}

function showSegments(): void {
    if (segments.length === 0) {
        lineGroup.visible = false;
        return;
    }
    const attribute = lineGeometry.attributes.position as BufferAttribute;
    if (attribute.array.length < segments.length) {
        lineGeometry.setAttribute('position', new BufferAttribute(new Float32Array(segments), 3));
    } else {
        (attribute.array as Float32Array).set(segments);
        attribute.needsUpdate = true;
    }
    lineGeometry.setDrawRange(0, segments.length / 3);
    lineGroup.position.copy(group1.position);
    lineGroup.rotation.copy(group1.rotation);
    lineGroup.scale.copy(group1.scale);
    lineGroup.visible = true;
}

function pose(delta: number): void {
    group1.position.y = -params.distance;
    group1.rotation.x -= delta * 3 * 0.0001 * params.speed * 0.5;
    group1.rotation.y -= delta * 1 * 0.0001 * params.speed * 0.5;
    group1.rotation.z -= delta * 2 * 0.0001 * params.speed * 0.5;
    group2.position.y = params.distance;
    group2.rotation.x += delta * 0.0001 * params.speed;
    group2.rotation.y += delta * 2 * 0.0001 * params.speed;
    group2.rotation.z += delta * 3 * 0.0001 * params.speed;
}

/** The same segments from every pair of triangles, with the two knots' world matrices. */
function bruteForce(): number {
    const position = geometry.attributes.position;
    const index = geometry.index!;
    const count = index.count / 3;
    const tris1: ExtendedTriangle[] = [];
    const tris2: ExtendedTriangle[] = [];
    const toTree1 = new Matrix4().copy(matrix2to1);
    for (let i = 0; i < count; i++) {
        const a = new ExtendedTriangle();
        a.a.fromBufferAttribute(position, index.getX(i * 3));
        a.b.fromBufferAttribute(position, index.getX(i * 3 + 1));
        a.c.fromBufferAttribute(position, index.getX(i * 3 + 2));
        const b = a.clone() as ExtendedTriangle;
        const t2 = new ExtendedTriangle(b.a.clone().applyMatrix4(toTree1), b.b.clone().applyMatrix4(toTree1), b.c.clone().applyMatrix4(toTree1));
        tris1.push(a);
        tris2.push(t2);
    }
    let found = 0;
    const out = new Line3();
    for (const t1 of tris1) for (const t2 of tris2) if (t1.intersectsTriangle(t2, out)) found++;
    return found;
}

overlay.range('speed', 0, 10, 0.001, params.speed, (v) => (params.speed = v));
overlay.range('distance', 0, 1.5, 0.001, params.distance, (v) => (params.distance = v));
overlay.checkbox('display meshes', params.displayMeshes, (v) => (params.displayMeshes = v));
overlay.select('knot triangles', Object.keys(SIZES) as Size[], size, (v) => {
    size = v;
    makeKnots();
});
overlay.note('Demonstrates tree.bvhcast: two trees walked against each other, triangle pairs from overlapping boxes only. The red line is every pair that crosses, as ExtendedTriangle.intersectsTriangle gives it.');
makeKnots();
group1.rotation.set(0.5, 0.2, 0.1);
group2.rotation.set(0.1, 0.9, 0.3);

let last = performance.now();
let frames = 0;
app.loop(async () => {
    const now = performance.now();
    pose(now - last);
    last = now;
    const result = intersectKnots();
    showSegments();
    overlay.set('crossing segments', `${result.count.toLocaleString('en-US')}, bvhcast ${result.ms.toFixed(2)} ms`);
    group1.visible = group2.visible = params.displayMeshes;
    frames++;
    if (isSmoke && frames === 3) {
        // The distance puts the knots in contact; every crossing pair must have been found.
        params.distance = 0.55;
        pose(0);
        const checked = intersectKnots();
        if (checked.count === 0) throw new Error('the knots do not touch in the smoke pose');
        const t0 = performance.now();
        const brute = bruteForce();
        const bruteMs = performance.now() - t0;
        if (brute !== checked.count) throw new Error(`bvhcast found ${checked.count} crossing pairs, every pair of triangles ${brute}`);
        overlay.set('smoke', `${brute} crossings both ways; bvhcast ${checked.ms.toFixed(2)} ms, every pair ${bruteMs.toFixed(0)} ms`);
        await ready(device, { ...overlay.snapshot(), crossings: brute, bvhcastMs: checked.ms, bruteMs });
    }
});

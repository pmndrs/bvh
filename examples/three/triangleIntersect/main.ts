/**
 * three/triangleIntersect: a port of three-mesh-bvh's `triangleIntersect` example.
 *
 * What it demonstrates: `ExtendedTriangle.intersectsTriangle(other, target)`,
 * the triangle-triangle test the tree walks (`bvhcast`) and the
 * `edgeIntersect` page use, alone. Two triangles, six corner sliders; when
 * the triangles cross, `target` (a `Line3`) is the segment where they meet and
 * the red cylinder and spheres show it. `needsUpdate` after moving a corner,
 * as with three-mesh-bvh.
 *
 * The smoke run tests thousands of random triangle pairs against a second
 * method (each edge through the other triangle with three's `Ray.intersectTriangle`)
 * and checks every returned segment lies in both triangles.
 */

import { CylinderGeometry, DoubleSide, BufferAttribute, BufferGeometry, Line3, Mesh, MeshPhongMaterial, Ray, SphereGeometry, Vector3, MathUtils, Matrix4, type Triangle } from 'three';
import { ExtendedTriangle } from '@pmndrs/bvh/three';
import { createApp } from '../../shared/app.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

const app = await createApp('Triangle intersect', 'three/triangleIntersect');
const { device, scene, camera, controls, overlay } = app;

const params = { sphereSize: 1, animate: !isSmoke };
const t1 = new ExtendedTriangle();
const t2 = new ExtendedTriangle();
t1.a.set(-1, 0, 0);
t1.b.set(2, 0, -2);
t1.c.set(2, 0, 2);
t2.a.set(1, 0, 0);
t2.b.set(-2, -2, 0);
t2.c.set(-2, 2, 0);
t1.needsUpdate = true;
t2.needsUpdate = true;
const rest = [t1.a, t1.b, t1.c, t2.a, t2.b, t2.c].map((v) => v.clone());
const corners = [t1.a, t1.b, t1.c, t2.a, t2.b, t2.c];
const line = new Line3();

const interMaterial = new MeshPhongMaterial({ color: 0xff0000, side: DoubleSide });
const t1Material = new MeshPhongMaterial({ color: 0x0000ff, side: DoubleSide });
const t2Material = new MeshPhongMaterial({ color: 0x00ff00, side: DoubleSide });
const sphereGeometry = new SphereGeometry(1);
const interSpheres = [0, 1].map(() => new Mesh(sphereGeometry, interMaterial));
const triSpheres = [t1Material, t1Material, t1Material, t2Material, t2Material, t2Material].map((material) => new Mesh(sphereGeometry, material));
scene.add(...interSpheres, ...triSpheres);
const intersectionMesh = new Mesh(new CylinderGeometry(), interMaterial);
scene.add(intersectionMesh);
const triangleGeometry = new BufferGeometry();
triangleGeometry.setAttribute('position', new BufferAttribute(new Float32Array(9), 3));
const t1Mesh = new Mesh(triangleGeometry.clone(), t1Material);
const t2Mesh = new Mesh(triangleGeometry.clone(), t2Material);
scene.add(t1Mesh, t2Mesh);

controls.target.set(0, 0, 0);
for (const v of corners) controls.target.add(v);
controls.target.multiplyScalar(1 / 6);
camera.position.set(10, 10, 10).add(controls.target);
controls.update();

function updateIntersectionMesh(mesh: Mesh, segment: Line3): void {
    mesh.geometry.dispose();
    const direction = new Vector3().subVectors(segment.start, segment.end);
    const geometry = new CylinderGeometry(1, 1, direction.length(), 6, 4, true);
    geometry.applyMatrix4(new Matrix4().makeTranslation(0, direction.length() / 2, 0));
    geometry.applyMatrix4(new Matrix4().makeRotationX(MathUtils.degToRad(90)));
    geometry.computeVertexNormals();
    mesh.geometry = geometry;
    mesh.position.copy(segment.start);
    mesh.lookAt(segment.end);
}

function updateTriangleMesh(mesh: Mesh, t: ExtendedTriangle): void {
    const position = mesh.geometry.getAttribute('position') as BufferAttribute;
    position.setXYZ(0, t.a.x, t.a.y, t.a.z);
    position.setXYZ(1, t.b.x, t.b.y, t.b.z);
    position.setXYZ(2, t.c.x, t.c.y, t.c.z);
    position.needsUpdate = true;
    mesh.geometry.computeVertexNormals();
}

const sliders: HTMLInputElement[] = [];
const names = ['a1', 'b1', 'c1', 'a2', 'b2', 'c2'];
overlay.range('sphere size', 0, 5, 0.001, params.sphereSize, (v) => (params.sphereSize = v));
overlay.checkbox('animate', params.animate, (v) => (params.animate = v));
corners.forEach((corner, i) => {
    for (const axis of ['x', 'y', 'z'] as const) {
        const input = overlay.range(`${names[i]}.${axis}`, -10, 10, 0.001, corner[axis], (v) => {
            corner[axis] = v;
            rest[i][axis] = v;
        });
        sliders.push(input);
    }
});
overlay.note('Demonstrates ExtendedTriangle.intersectsTriangle(other, target): the segment where two triangles cross, the test behind tree.bvhcast. Move the six corners; "animate" swings the second triangle.');

function update(time: number): void {
    if (params.animate) {
        // The second triangle swings about the first's plane.
        const angle = Math.sin(time * 0.0005) * 0.9;
        for (let i = 3; i < 6; i++) corners[i].copy(rest[i]).applyAxisAngle(new Vector3(1, 0, 0), angle);
        sliders.forEach((input, k) => {
            const value = corners[Math.floor(k / 3)][(['x', 'y', 'z'] as const)[k % 3]];
            input.value = String(value);
        });
    }
    intersectionMesh.visible = interSpheres[0].visible = interSpheres[1].visible = false;
    t1.needsUpdate = t2.needsUpdate = true;
    if (t1.intersectsTriangle(t2, line)) {
        updateIntersectionMesh(intersectionMesh, line);
        intersectionMesh.visible = interSpheres[0].visible = interSpheres[1].visible = true;
        overlay.set('intersection', `(${line.start.toArray().map((x) => x.toFixed(2))}) to (${line.end.toArray().map((x) => x.toFixed(2))})`);
    } else {
        line.start.set(Infinity, Infinity, Infinity);
        line.end.set(Infinity, Infinity, Infinity);
        overlay.set('intersection', 'none');
    }
    corners.forEach((corner, i) => triSpheres[i].position.copy(corner));
    interSpheres[0].position.copy(line.start);
    interSpheres[1].position.copy(line.end);
    updateTriangleMesh(t1Mesh, t1);
    updateTriangleMesh(t2Mesh, t2);
    for (const sphere of [...triSpheres, ...interSpheres]) sphere.scale.setScalar(0.005 * params.sphereSize * sphere.position.distanceTo(camera.position));
    for (const sphere of interSpheres) sphere.scale.multiplyScalar(1.5);
    intersectionMesh.scale.setScalar(0.5 * Math.min(interSpheres[0].scale.x, interSpheres[1].scale.x));
    intersectionMesh.scale.z = 1;
}

/** Whether the two triangles cross, by sending each edge of one through the other. */
function crossByEdges(a: Triangle, b: Triangle): boolean {
    const ray = new Ray();
    const out = new Vector3();
    const edges = (t: Triangle): [Vector3, Vector3][] => [
        [t.a, t.b],
        [t.b, t.c],
        [t.c, t.a],
    ];
    for (const [from, other] of [[a, b], [b, a]] as const) {
        for (const [p, q] of edges(from)) {
            ray.origin.copy(p);
            ray.direction.subVectors(q, p);
            const length = ray.direction.length();
            ray.direction.divideScalar(length);
            const hit = ray.intersectTriangle(other.a, other.b, other.c, false, out);
            if (hit && ray.origin.distanceTo(out) <= length) return true;
        }
    }
    return false;
}

function randomTriangle(t: ExtendedTriangle, random: () => number): void {
    for (const v of [t.a, t.b, t.c]) v.set(random() * 2 - 1, random() * 2 - 1, random() * 2 - 1);
    t.needsUpdate = true;
}

let frames = 0;
app.loop(async (now) => {
    update(now);
    frames++;
    if (isSmoke && frames === 3) {
        // The page's own pair first: these cross.
        for (let i = 0; i < 6; i++) corners[i].copy(rest[i]);
        t1.needsUpdate = t2.needsUpdate = true;
        if (!t1.intersectsTriangle(t2, line)) throw new Error('the default pair should cross');
        // Random pairs against the edge method.
        let seed = 12345;
        const random = (): number => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
        const a = new ExtendedTriangle();
        const b = new ExtendedTriangle();
        const segment = new Line3();
        const closest = new Vector3();
        const pairs = 5000;
        let crossing = 0;
        let worst = 0;
        for (let i = 0; i < pairs; i++) {
            randomTriangle(a, random);
            randomTriangle(b, random);
            const ours = a.intersectsTriangle(b, segment);
            const theirs = crossByEdges(a, b);
            if (ours !== theirs) throw new Error(`pair ${i}: intersectsTriangle ${ours}, edge test ${theirs}`);
            if (ours) {
                crossing++;
                // Both ends of the segment lie in both triangles.
                for (const end of [segment.start, segment.end]) {
                    worst = Math.max(worst, a.closestPointToPoint(end, closest).distanceTo(end), b.closestPointToPoint(end, closest).distanceTo(end));
                }
            }
        }
        if (crossing < pairs / 20) throw new Error(`only ${crossing} of ${pairs} random pairs cross; the check is too weak`);
        if (worst > 1e-4) throw new Error(`a segment end is ${worst} from a triangle`);
        overlay.set('smoke', `${pairs} random pairs, ${crossing} crossing, same answers; segment ends within ${worst.toExponential(1)} of both triangles`);
        await ready(device, { ...overlay.snapshot(), pairs, crossing, worst });
    }
});

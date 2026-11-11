/**
 * Shapecast and bvhcast against three-mesh-bvh 0.9.18 on a real mesh, in the
 * two callers' own patterns, reporting agreement and time per query:
 *
 * - **homefig's character controller** (`editor/.../first-person/bvh-ecctrl.tsx`,
 *   `collisionCheck` and `floatingCheck`): a capsule's segment and box in the
 *   mesh's space, `shapecast` culling boxes with `box.intersectsBox` and
 *   testing each triangle with `closestPointToSegment`, accumulating contacts.
 *   Agreement: the same contact triangles, the same accumulated normal, point
 *   and depth (to 1e-9; the sums run in tree order, which differs), the same
 *   float hit.
 * - **three-bvh-csg's `collectIntersectingTriangles`**
 *   (`src/core/operations/operationsUtils.js`): `bvhcast` between two brushes
 *   (trees with `maxLeafSize: 3`, indirect), skipping degenerate triangles,
 *   keeping pairs where `intersectsTriangle(other, edge, true)`, by original
 *   index through `resolveTriangleIndex`. Agreement: the same pairs and the
 *   same edges, exactly. (csg's coplanar special case is left out: it is its
 *   own code, run on the same pairs.)
 *
 *   node --experimental-transform-types bench/query-parity.ts [--obj path] [--queries 2000] [--rounds 3]
 *
 * The default mesh is the furnished room (`breakfast_room.obj`, from
 * BVH_ASSETS_DIR, default ~/Developer/webgpu-ddgi-radiance-cache/assets); a
 * torus knot stands in when it is missing. A timing and a cross-check, not a
 * gate: the gates are src/three/shapecast.test.ts and bvhcast.test.ts.
 */

import { existsSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { Box3, BufferGeometry, IcosahedronGeometry, Line3, Matrix3, Matrix4, type Mesh, Quaternion, TorusKnotGeometry, Triangle, Vector3 } from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { MeshBVH } from 'three-mesh-bvh';
import { BvhBoundsTree, prepareBoundsTrees } from '../src/three/raycast.ts';
import type { ExtendedTriangle } from '../src/three/shapes.ts';
import { createRandom } from '../src/trace/oracle-fixtures.ts';

const args = process.argv.slice(2);
const option = (name: string, fallback: string) => {
    const at = args.indexOf(`--${name}`);
    return at >= 0 ? args[at + 1] : fallback;
};
const assets = process.env.BVH_ASSETS_DIR ?? join(homedir(), 'Developer/webgpu-ddgi-radiance-cache/assets');
const objPath = option('obj', join(assets, 'downloads/furnished-room/breakfast_room.obj'));
const queries = Number(option('queries', '2000'));
const rounds = Number(option('rounds', '3'));

/** A tree as both libraries' `geometry.boundsTree` look to these patterns. */
interface Tree {
    shapecast(callbacks: { intersectsBounds: (box: Box3) => unknown; intersectsTriangle: (triangle: ExtendedTriangle, index: number) => unknown }): boolean;
    bvhcast(other: Tree, matrix: Matrix4, callbacks: { intersectsTriangles: (a: ExtendedTriangle, b: ExtendedTriangle, ia: number, ib: number) => unknown }): boolean;
    resolveTriangleIndex(i: number): number;
}

function loadMesh(): { geometry: BufferGeometry; name: string } {
    if (existsSync(objPath)) {
        const group = new OBJLoader().parse(readFileSync(objPath, 'utf8'));
        const parts: BufferGeometry[] = [];
        group.traverse((object) => {
            const mesh = object as Mesh;
            if (!mesh.isMesh) return;
            const part = new BufferGeometry();
            part.setAttribute('position', mesh.geometry.getAttribute('position'));
            parts.push(part.index ? part.toNonIndexed() : part);
        });
        return { geometry: mergeGeometries(parts), name: objPath.split('/').pop()! };
    }
    console.log(`(${objPath} not found: using a torus knot)`);
    return { geometry: new TorusKnotGeometry(1, 0.3, 600, 64), name: 'torus knot (stand-in)' };
}

const { geometry: room, name } = loadMesh();
await prepareBoundsTrees();
const roomTriangles = room.getAttribute('position').count / 3;
room.computeBoundingBox();
const size = room.boundingBox!.getSize(new Vector3());
const diagonal = size.length();

function time(label: string, run: () => void): number {
    run(); // warm-up
    const start = performance.now();
    for (let round = 0; round < rounds; round++) run();
    return (performance.now() - start) / rounds;
}

console.log(`${name}: ${roomTriangles.toLocaleString()} triangles, ${size.x.toFixed(2)} x ${size.y.toFixed(2)} x ${size.z.toFixed(2)}; ${queries} queries per pattern, ${rounds} rounds; times are means per query\n`);

// ---- homefig: capsule collision and float sensor ----

// homefig builds with computeBoundsTree({ maxLeafSize: 10 }).
let start = performance.now();
const theirsRoom = new MeshBVH(room.clone(), { maxLeafSize: 10, indirect: true }) as unknown as Tree;
const theirsBuild = performance.now() - start;
start = performance.now();
const oursRoom = new BvhBoundsTree(room, { maxLeafSize: 10 }) as unknown as Tree;
const oursBuild = performance.now() - start;

const random = createRandom(7);
const positions = room.getAttribute('position');
const corner = (triangle: number, i: number) => new Vector3().fromBufferAttribute(positions, triangle * 3 + i);
const capsuleRadius = diagonal * 0.01;
const capsuleLength = diagonal * 0.03;
const floatSensorRadius = capsuleRadius * 0.8;
const floatHeight = capsuleRadius * 0.5;
const floatPullBackHeight = capsuleRadius * 0.5;
const up = new Vector3(0, 1, 0);
const maxSlope = 1;

/** Capsules resting near random surfaces (so most touch something), with a mesh transform as homefig's originMatrix. */
const meshMatrix = new Matrix4().compose(new Vector3(0.3, -0.2, 0.1), new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), 0.4), new Vector3(1, 1, 1));
const capsules = Array.from({ length: queries }, () => {
    const triangle = Math.floor(random() * roomTriangles);
    const tri = new Triangle(corner(triangle, 0), corner(triangle, 1), corner(triangle, 2));
    const center = tri.getMidpoint(new Vector3());
    const normal = tri.getNormal(new Vector3());
    const offset = (random() - 0.3) * capsuleRadius * 2;
    const startPoint = center.addScaledVector(normal, offset).applyMatrix4(meshMatrix);
    return new Line3(startPoint, startPoint.clone().addScaledVector(up, capsuleLength));
});

interface Contact {
    triangles: number[];
    normal: Vector3;
    point: Vector3;
    depth: number;
    floatDistance: number;
    floatPoint: Vector3;
    /** Every hit point at the nearest float distance: homefig keeps the first one its walk reaches, so a tie goes by visit order. */
    floatTies: string[];
}

/**
 * homefig's collisionCheck then floatingCheck for one capsule against one
 * mesh, its arithmetic. `cullExact` also skips float-sensor triangles whose
 * own box misses the sensor's box. homefig's vertical test takes |dot|, so it
 * accepts hits above the sensor that lie outside the box it culls with; a
 * tree reaches those only when they share a leaf with a triangle inside, so
 * that part of the answer depends on the tree's leaves.
 */
function controller(tree: Tree, segment: Line3, collect: boolean, cullExact = false): Contact {
    const originMatrix = meshMatrix;
    const scale = new Vector3();
    originMatrix.decompose(new Vector3(), new Quaternion(), scale);
    const invert = originMatrix.clone().invert();
    const localSegment = segment.clone().applyMatrix4(invert);
    const scaledRadius = new Vector3(capsuleRadius / scale.x, capsuleRadius / scale.y, capsuleRadius / scale.z);
    const localBox = new Box3().makeEmpty().expandByPoint(localSegment.start).expandByPoint(localSegment.end);
    localBox.min.addScaledVector(scaledRadius, -1);
    localBox.max.add(scaledRadius);
    const triPoint = new Vector3();
    const capsulePoint = new Vector3();
    const delta = new Vector3();
    const accumulatedNormal = new Vector3();
    const accumulatedPoint = new Vector3();
    const contactNormal = new Vector3();
    let totalDepth = 0;
    const triangles: number[] = [];
    tree.shapecast({
        intersectsBounds: (box) => box.intersectsBox(localBox),
        intersectsTriangle: (tri, index) => {
            tri.closestPointToSegment(localSegment, triPoint, capsulePoint);
            delta.copy(triPoint).sub(capsulePoint).divide(scaledRadius);
            if (delta.lengthSq() < 1) {
                triPoint.applyMatrix4(originMatrix);
                capsulePoint.applyMatrix4(originMatrix);
                contactNormal.copy(capsulePoint).sub(triPoint).normalize();
                const depth = capsuleRadius - capsulePoint.distanceTo(triPoint);
                accumulatedNormal.addScaledVector(contactNormal, depth);
                accumulatedPoint.add(triPoint);
                totalDepth += depth;
                if (collect) triangles.push(index);
            }
        },
    });
    // floatingCheck: a sensor segment below the capsule (updateSegmentBBox), nearest walkable triangle by vertical distance.
    const sensorWorld = new Line3(segment.start.clone(), segment.start.clone().addScaledVector(up, -(floatHeight + capsuleRadius)));
    const expendPoint = sensorWorld.end.clone().addScaledVector(up, -floatPullBackHeight).applyMatrix4(invert);
    const sensor = sensorWorld.clone().applyMatrix4(invert);
    const normalInverse = new Matrix3().getNormalMatrix(invert);
    const normalMatrix = new Matrix3().getNormalMatrix(originMatrix);
    const scaledFloat = new Vector3(floatSensorRadius / scale.x, floatSensorRadius / scale.y, floatSensorRadius / scale.z);
    const sensorBox = new Box3().makeEmpty().expandByPoint(sensor.start).expandByPoint(expendPoint);
    sensorBox.min.addScaledVector(scaledFloat, -1);
    sensorBox.max.add(scaledFloat);
    const localUp = up.clone().applyMatrix3(normalInverse).normalize();
    const hitPoint = new Vector3();
    const segPoint = new Vector3();
    const deltaHit = new Vector3();
    const triNormal = new Vector3();
    let floatDistance = Infinity;
    const floatPoint = new Vector3(Infinity, Infinity, Infinity);
    let floatTies: string[] = [];
    const triangleBox = new Box3();
    tree.shapecast({
        intersectsBounds: (box) => box.intersectsBox(sensorBox),
        intersectsTriangle: (tri, index) => {
            if (cullExact && !triangleBox.setFromPoints(tri.points).intersectsBox(sensorBox)) return;
            tri.closestPointToSegment(sensor, hitPoint, segPoint);
            deltaHit.subVectors(hitPoint, sensor.start).divide(scaledFloat);
            const totalLengthSq = deltaHit.lengthSq();
            const dot = deltaHit.dot(localUp);
            const verticalLength = Math.abs(dot) / ((capsuleRadius + floatHeight + floatPullBackHeight) / floatSensorRadius);
            const horizontalLength = Math.sqrt(Math.max(0, totalLengthSq - dot * dot));
            if (horizontalLength < 1 && verticalLength < 1) {
                tri.getNormal(triNormal);
                triNormal.applyMatrix3(normalMatrix).normalize();
                if (verticalLength < floatDistance && triNormal.angleTo(up) < maxSlope) {
                    floatDistance = verticalLength;
                    floatPoint.copy(hitPoint).applyMatrix4(originMatrix);
                    if (collect) floatTies = [];
                }
                if (collect && verticalLength === floatDistance && triNormal.angleTo(up) < maxSlope) floatTies.push(`${index} ${hitPoint.clone().applyMatrix4(originMatrix).toArray()}`);
            }
        },
    });
    return { triangles: triangles.sort((a, b) => a - b), normal: accumulatedNormal, point: accumulatedPoint, depth: totalDepth, floatDistance, floatPoint, floatTies: floatTies.sort() };
}

let agree = 0;
let agreeAsWritten = 0;
let tieBroken = 0;
let contactTriangles = 0;
let withContact = 0;
let firstMismatch = '';
const near = (a: Vector3, b: Vector3) => a.distanceTo(b) <= 1e-9 * Math.max(1, a.length());
const sameFloat = (a: Contact, b: Contact) => a.floatDistance === b.floatDistance && a.floatTies.join() === b.floatTies.join();
for (let i = 0; i < queries; i++) {
    const ours = controller(oursRoom, capsules[i], true, true);
    const theirs = controller(theirsRoom, capsules[i], true, true);
    const same =
        ours.triangles.join() === theirs.triangles.join() &&
        near(ours.normal, theirs.normal) &&
        near(ours.point, theirs.point) &&
        Math.abs(ours.depth - theirs.depth) <= 1e-9 &&
        sameFloat(ours, theirs);
    if (same) {
        agree++;
        if (!ours.floatPoint.equals(theirs.floatPoint)) tieBroken++;
    } else if (!firstMismatch) firstMismatch = `capsule #${i}: ours ${ours.triangles.length} contacts, theirs ${theirs.triangles.length}`;
    if (same && sameFloat(controller(oursRoom, capsules[i], true), controller(theirsRoom, capsules[i], true))) agreeAsWritten++;
    contactTriangles += theirs.triangles.length;
    if (theirs.triangles.length > 0) withContact++;
}
const oursCapsule = time('ours', () => capsules.forEach((segment) => controller(oursRoom, segment, false)));
const theirsCapsule = time('theirs', () => capsules.forEach((segment) => controller(theirsRoom, segment, false)));

console.log("homefig's controller (collisionCheck + floatingCheck: two shapecasts per capsule)");
console.log(`  build: ours ${oursBuild.toFixed(0)} ms, three-mesh-bvh ${theirsBuild.toFixed(0)} ms (maxLeafSize 10)`);
console.log(`  agreement: ${agree} / ${queries} capsules (${withContact} touching, ${contactTriangles.toLocaleString()} contact triangles): the same contact triangles, sums equal to 1e-9, the same float distance and tied float hits${firstMismatch ? `; first mismatch ${firstMismatch}` : ''}`);
console.log(`    float point: in ${tieBroken} the tied hits are the same but homefig keeps the first its walk reaches, and the walks differ`);
console.log(`    homefig's float test as written (|dot| accepts hits above its cull box): ${agreeAsWritten} / ${queries} agree; the rest are triangles outside the box that a tree reaches only through a shared leaf`);
console.log(`  per capsule: ours ${((oursCapsule / queries) * 1000).toFixed(2)} µs, three-mesh-bvh ${((theirsCapsule / queries) * 1000).toFixed(2)} µs (ours / theirs ${(oursCapsule / theirsCapsule).toFixed(2)})\n`);

// ---- three-bvh-csg: brush against brush ----

/** csg's isTriDegenerate. */
function isTriDegenerate(tri: Triangle, eps = 1e-10): boolean {
    return tri.a.distanceToSquared(tri.b) < eps || tri.a.distanceToSquared(tri.c) < eps || tri.b.distanceToSquared(tri.c) < eps;
}

/** csg's collectIntersectingTriangles (without its coplanar special case): `a` in its own space, `b` placed by `matrix`. */
function collect(a: Tree, b: Tree, matrix: Matrix4, record: boolean): string[] {
    const pairs: string[] = [];
    const edge = new Line3();
    a.bvhcast(b, matrix, {
        intersectsTriangles(triangleA, triangleB, ia, ib) {
            if (!isTriDegenerate(triangleA) && !isTriDegenerate(triangleB)) {
                if (triangleA.intersectsTriangle(triangleB, edge, true)) {
                    const va = a.resolveTriangleIndex(ia);
                    const vb = b.resolveTriangleIndex(ib);
                    if (record) pairs.push(`${va} ${vb} ${edge.start.toArray()} ${edge.end.toArray()}`);
                }
            }
            return false;
        },
    });
    return pairs.sort();
}

// The brushes: the room as A (csg: maxLeafSize 3, indirect), and a sphere brush B moved through it.
const brush = new IcosahedronGeometry(diagonal * 0.05, 12);
const theirsA = new MeshBVH(room.clone(), { maxLeafSize: 3, indirect: true }) as unknown as Tree;
const oursA = new BvhBoundsTree(room, { maxLeafSize: 3 }) as unknown as Tree;
const theirsB = new MeshBVH(brush.clone(), { maxLeafSize: 3, indirect: true }) as unknown as Tree;
const oursB = new BvhBoundsTree(brush, { maxLeafSize: 3 }) as unknown as Tree;
const box = room.boundingBox!;
const csgQueries = Math.max(1, Math.round(queries / 20));
const placements = Array.from({ length: csgQueries }, () => {
    const position = new Vector3(box.min.x + random() * size.x, box.min.y + random() * size.y, box.min.z + random() * size.z);
    const rotation = new Quaternion().setFromAxisAngle(new Vector3(random(), random(), random()).normalize(), random() * Math.PI);
    // csg: a.matrixWorld^-1 * b.matrixWorld, with a at the origin.
    return new Matrix4().compose(position, rotation, new Vector3(1, 1, 1));
});
let csgAgree = 0;
let csgPairs = 0;
let csgFirstMismatch = '';
for (let i = 0; i < csgQueries; i++) {
    const ours = collect(oursA, oursB, placements[i], true);
    const theirs = collect(theirsA, theirsB, placements[i], true);
    if (ours.join('\n') === theirs.join('\n')) csgAgree++;
    else if (!csgFirstMismatch) csgFirstMismatch = `placement #${i}: ours ${ours.length} pairs, theirs ${theirs.length}`;
    csgPairs += theirs.length;
}
const oursCsg = time('ours', () => placements.forEach((m) => collect(oursA, oursB, m, false)));
const theirsCsg = time('theirs', () => placements.forEach((m) => collect(theirsA, theirsB, m, false)));
console.log(`three-bvh-csg's collectIntersectingTriangles (bvhcast, room x sphere brush of ${(brush.getAttribute('position').count / 3).toLocaleString()} triangles)`);
console.log(`  agreement: ${csgAgree} / ${csgQueries} placements (${csgPairs} intersecting pairs, same pairs and edges)${csgFirstMismatch ? `; first mismatch ${csgFirstMismatch}` : ''}`);
console.log(`  per placement: ours ${(oursCsg / csgQueries).toFixed(3)} ms, three-mesh-bvh ${(theirsCsg / csgQueries).toFixed(3)} ms (ours / theirs ${(oursCsg / theirsCsg).toFixed(2)})`);

/**
 * `closestPointToPoint` and `closestPointToGeometry` against three-mesh-bvh
 * 0.9.18 on a real mesh, reporting agreement and time per query:
 *
 * - **Points**: random points in the mesh's box (most of them in the open
 *   room) and points just off random surfaces. Agreement: the same distance
 *   (to 1e-9) and the same triangle, or another equally near.
 * - **Geometry**: a small icosphere (180 triangles, with its own tree) at
 *   random rigid placements in the room. Agreement: the same distance.
 *
 * Both trees are built as `computeBoundsTree()` would (three-mesh-bvh's
 * defaults, `indirect` so its `faceIndex` is the original triangle; ours).
 *
 *   node --experimental-transform-types bench/closest-point.ts [--obj path] [--queries 2000] [--rounds 3]
 *
 * The default mesh is the furnished room (`breakfast_room.obj`, from
 * BVH_ASSETS_DIR, default ~/Developer/webgpu-ddgi-radiance-cache/assets); a
 * torus knot stands in when it is missing. A timing and a cross-check, not a
 * gate: the gate is src/three/closest-point.test.ts.
 */

import { existsSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { BufferGeometry, IcosahedronGeometry, Matrix4, type Mesh, Quaternion, TorusKnotGeometry, Triangle, Vector3 } from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { MeshBVH, type HitPointInfo } from 'three-mesh-bvh';
import { BvhBoundsTree, prepareBoundsTrees, setBoundsTree } from '../src/three/raycast.ts';
import { createRandom, randomUnitVector } from '../src/trace/oracle-fixtures.ts';

const args = process.argv.slice(2);
const option = (name: string, fallback: string) => {
    const at = args.indexOf(`--${name}`);
    return at >= 0 ? args[at + 1] : fallback;
};
const assets = process.env.BVH_ASSETS_DIR ?? join(homedir(), 'Developer/webgpu-ddgi-radiance-cache/assets');
const objPath = option('obj', join(assets, 'downloads/furnished-room/breakfast_room.obj'));
const queries = Number(option('queries', '2000'));
const rounds = Number(option('rounds', '3'));

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
const triangles = room.getAttribute('position').count / 3;
room.computeBoundingBox();
const box = room.boundingBox!;
const size = box.getSize(new Vector3());
const diagonal = size.length();

function time(run: () => void): number {
    run(); // warm-up
    const start = performance.now();
    for (let round = 0; round < rounds; round++) run();
    return (performance.now() - start) / rounds;
}

interface Tree {
    closestPointToPoint(point: Vector3, target?: object, min?: number, max?: number): HitPointInfo | null;
    closestPointToGeometry(geometry: BufferGeometry, matrix: Matrix4, target1?: object, target2?: object | null): HitPointInfo | null;
}

console.log(`${name}: ${triangles.toLocaleString()} triangles, ${size.x.toFixed(2)} x ${size.y.toFixed(2)} x ${size.z.toFixed(2)}; ${queries} queries per pattern, ${rounds} rounds; times are means per query\n`);

let start = performance.now();
const theirs = new MeshBVH(room.clone(), { indirect: true }) as unknown as Tree;
const theirsBuild = performance.now() - start;
start = performance.now();
const ours = new BvhBoundsTree(room) as unknown as Tree;
const oursBuild = performance.now() - start;
console.log(`build: three-mesh-bvh ${theirsBuild.toFixed(0)} ms, ours ${oursBuild.toFixed(0)} ms\n`);

const random = createRandom(11);
const positions = room.getAttribute('position');
const corner = (triangle: number, i: number) => new Vector3().fromBufferAttribute(positions, triangle * 3 + i);
const triangleOf = (face: number) => new Triangle(corner(face, 0), corner(face, 1), corner(face, 2));

function pointSets(): { label: string; points: Vector3[] }[] {
    const inBox: Vector3[] = [];
    const nearSurface: Vector3[] = [];
    for (let i = 0; i < queries; i++) {
        inBox.push(new Vector3(box.min.x + random() * size.x, box.min.y + random() * size.y, box.min.z + random() * size.z));
        const face = Math.floor(random() * triangles);
        const a = random();
        const b = random() * (1 - a);
        const onSurface = corner(face, 0).multiplyScalar(1 - a - b).addScaledVector(corner(face, 1), a).addScaledVector(corner(face, 2), b);
        nearSurface.push(onSurface.addScaledVector(new Vector3(...randomUnitVector(random)), diagonal * 0.005 * random()));
    }
    return [
        { label: 'points in the box', points: inBox },
        { label: 'points near surfaces', points: nearSurface },
    ];
}

for (const { label, points } of pointSets()) {
    let same = 0;
    let tied = 0;
    let disagree = 0;
    for (const point of points) {
        const expected = theirs.closestPointToPoint(point)!;
        const result = ours.closestPointToPoint(point)!;
        if (Math.abs(expected.distance - result.distance) > 1e-9) disagree++;
        else if (expected.faceIndex === result.faceIndex) same++;
        else if (Math.abs(triangleOf(expected.faceIndex).closestPointToPoint(point, new Vector3()).distanceTo(point) - result.distance) < 1e-9) tied++;
        else disagree++;
    }
    const target = { point: new Vector3(), distance: 0, faceIndex: 0 };
    const theirsTime = time(() => {
        for (const point of points) theirs.closestPointToPoint(point, target);
    });
    const oursTime = time(() => {
        for (const point of points) ours.closestPointToPoint(point, target);
    });
    console.log(
        `closestPointToPoint, ${label}: agree ${same} same triangle + ${tied} tied / ${points.length}, ${disagree} disagree; ` +
            `three-mesh-bvh ${((theirsTime / points.length) * 1000).toFixed(2)} us, ours ${((oursTime / points.length) * 1000).toFixed(2)} us (${(oursTime / theirsTime).toFixed(2)}x)`,
    );
}

// ---- closestPointToGeometry: an icosphere with its own tree, rigidly placed in the room ----

const sphere = new IcosahedronGeometry(diagonal * 0.02, 2);
const theirsSphere = sphere.clone();
(theirsSphere as unknown as { boundsTree: MeshBVH }).boundsTree = new MeshBVH(theirsSphere);
setBoundsTree(sphere, new BvhBoundsTree(sphere));
const placements: Matrix4[] = [];
const geometryQueries = Math.max(1, Math.floor(queries / 10));
for (let i = 0; i < geometryQueries; i++) {
    const position = new Vector3(box.min.x + random() * size.x, box.min.y + random() * size.y, box.min.z + random() * size.z);
    const rotation = new Quaternion().setFromAxisAngle(new Vector3(...randomUnitVector(random)), random() * Math.PI * 2);
    placements.push(new Matrix4().compose(position, rotation, new Vector3(1, 1, 1)));
}
let agree = 0;
for (const matrix of placements) {
    const expected = theirs.closestPointToGeometry(theirsSphere, matrix, {}, {})!;
    const result = ours.closestPointToGeometry(sphere, matrix, {}, {})!;
    if (Math.abs(expected.distance - result.distance) <= 1e-9) agree++;
}
const theirsGeometryTime = time(() => {
    for (const matrix of placements) theirs.closestPointToGeometry(theirsSphere, matrix, {}, {});
});
const oursGeometryTime = time(() => {
    for (const matrix of placements) ours.closestPointToGeometry(sphere, matrix, {}, {});
});
console.log(
    `closestPointToGeometry, a ${sphere.index ? sphere.index.count / 3 : sphere.getAttribute('position').count / 3}-triangle icosphere: same distance ${agree} / ${placements.length}; ` +
        `three-mesh-bvh ${((theirsGeometryTime / placements.length) * 1000).toFixed(1)} us, ours ${((oursGeometryTime / placements.length) * 1000).toFixed(1)} us (${(oursGeometryTime / theirsGeometryTime).toFixed(2)}x)`,
);

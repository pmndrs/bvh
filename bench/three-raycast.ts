/**
 * CPU raycasts per second in Node: three's default `Mesh.raycast`, three-mesh-bvh's
 * `acceleratedRaycast` (0.9.18; its default CENTER tree and an SAH tree), and
 * `@pmndrs/bvh/three`'s `acceleratedRaycast` (tinybvh tree, the CPU walk).
 * The same rays and the same mesh for each, with and without `firstHitOnly`.
 * Every library gets its own copy of the geometry (three-mesh-bvh reorders the
 * index, and both use `geometry.boundsTree`).
 *
 *   node --experimental-transform-types bench/three-raycast.ts [--triangles 100000] [--seconds 1]
 *
 * A timing, not a gate: the correctness check is src/three/raycast.test.ts.
 */

import { type BufferGeometry, Mesh, MeshBasicMaterial, Raycaster, TorusKnotGeometry, Vector3, DoubleSide } from 'three';
import { acceleratedRaycast as tmbAcceleratedRaycast, MeshBVH, SAH, CENTER } from 'three-mesh-bvh';
import { acceleratedRaycast, computeBoundsTree, prepareBoundsTrees } from '../src/three/raycast.ts';
import { createRandom } from '../src/trace/oracle-fixtures.ts';

const args = process.argv.slice(2);
const option = (name: string, fallback: number) => {
    const at = args.indexOf(`--${name}`);
    return at >= 0 ? Number(args[at + 1]) : fallback;
};
const targetTriangles = option('triangles', 100_000);
const seconds = option('seconds', 1);

// A torus knot with about `targetTriangles` triangles (2 per tubular x radial segment).
const radial = 32;
const tubular = Math.max(8, Math.round(targetTriangles / (2 * radial)));
const source = new TorusKnotGeometry(1, 0.3, tubular, radial);
const triangles = source.index!.count / 3;

const threeRaycast = Mesh.prototype.raycast;
const material = new MeshBasicMaterial({ side: DoubleSide });

function meshOf(geometry: BufferGeometry, raycast: Mesh['raycast']): Mesh {
    const mesh = new Mesh(geometry, material);
    mesh.raycast = raycast;
    mesh.updateMatrixWorld();
    return mesh;
}

const random = createRandom(9);
const rays = Array.from({ length: 4096 }, () => {
    const origin = new Vector3((random() - 0.5) * 8, (random() - 0.5) * 8, (random() - 0.5) * 8);
    const target = new Vector3((random() - 0.5) * 2, (random() - 0.5) * 2, (random() - 0.5) * 2);
    return { origin, direction: target.sub(origin).normalize() };
});

function measure(mesh: Mesh, firstHitOnly: boolean, maxSeconds = seconds): { perSecond: number; hits: number } {
    const raycaster = new Raycaster() as Raycaster & { firstHitOnly?: boolean };
    raycaster.firstHitOnly = firstHitOnly;
    const out: unknown[] = [];
    let hits = 0;
    const run = (count: number) => {
        for (let i = 0; i < count; i++) {
            const ray = rays[i % rays.length];
            raycaster.ray.origin.copy(ray.origin);
            raycaster.ray.direction.copy(ray.direction);
            out.length = 0;
            mesh.raycast(raycaster, out as never);
            hits += out.length > 0 ? 1 : 0;
        }
    };
    run(Math.min(200, rays.length)); // warm-up
    hits = 0;
    let done = 0;
    const start = performance.now();
    while (performance.now() - start < maxSeconds * 1000) {
        run(64);
        done += 64;
    }
    const elapsed = (performance.now() - start) / 1000;
    return { perSecond: done / elapsed, hits: hits / done };
}

await prepareBoundsTrees();
const ours = source.clone();
let start = performance.now();
computeBoundsTree.call(ours);
const oursBuildMs = performance.now() - start;

const tmbCenter = source.clone();
start = performance.now();
(tmbCenter as BufferGeometry & { boundsTree?: unknown }).boundsTree = new MeshBVH(tmbCenter, { strategy: CENTER });
const tmbCenterBuildMs = performance.now() - start;
const tmbSah = source.clone();
start = performance.now();
(tmbSah as BufferGeometry & { boundsTree?: unknown }).boundsTree = new MeshBVH(tmbSah, { strategy: SAH });
const tmbSahBuildMs = performance.now() - start;

const rows: { name: string; buildMs: number | null; mesh: Mesh }[] = [
    { name: "three default Mesh.raycast", buildMs: null, mesh: meshOf(source.clone(), threeRaycast) },
    { name: 'three-mesh-bvh (CENTER, default)', buildMs: tmbCenterBuildMs, mesh: meshOf(tmbCenter, tmbAcceleratedRaycast) },
    { name: 'three-mesh-bvh (SAH)', buildMs: tmbSahBuildMs, mesh: meshOf(tmbSah, tmbAcceleratedRaycast) },
    { name: '@pmndrs/bvh/three', buildMs: oursBuildMs, mesh: meshOf(ours, acceleratedRaycast) },
];

console.log(`torus knot, ${triangles} triangles, ${rays.length} rays from outside, ${seconds} s per row, Node ${process.version}`);
console.log('| Raycast | Build ms | firstHitOnly rays/s | all hits rays/s | hit rate |');
console.log('| --- | --- | --- | --- | --- |');
for (const row of rows) {
    // three's default is linear in triangles: give it less time.
    const first = measure(row.mesh, true);
    const all = measure(row.mesh, false);
    console.log(`| ${row.name} | ${row.buildMs === null ? '-' : row.buildMs.toFixed(0)} | ${first.perSecond.toFixed(0)} | ${all.perSecond.toFixed(0)} | ${(first.hits * 100).toFixed(1)}% |`);
}

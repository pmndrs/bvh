/**
 * `CpuMeshBvh` with `corners: 'records'` (the default, 48-byte triangle records)
 * against `corners: 'positions'` (16-byte indexed triangles, corners read from
 * the source's positions): bytes, and time per query for closest-hit and
 * all-hit raycasts, `closestPointToPoint`, `intersectsBox` and `refit`, on one
 * real mesh. Both trees are the same tinybvh tree, so answers are identical
 * (checked here on every query; the gate is src/query/position-corners.test.ts).
 * Rounds alternate the order of the two, and the ratio is taken per round.
 *
 *   node --experimental-transform-types bench/cpu-corners.ts [--obj path] [--queries 20000] [--rounds 5]
 *
 * The default mesh is the furnished room (`breakfast_room.obj`, from
 * BVH_ASSETS_DIR, default ~/Developer/webgpu-ddgi-radiance-cache/assets); a
 * torus knot stands in when it is missing. A timing, not a gate.
 */

import { existsSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { BufferGeometry, type Mesh, TorusKnotGeometry } from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import type { MeshInput } from '../src/mesh-input.ts';
import { CpuMeshBvh } from '../src/query/cpu-mesh-bvh.ts';
import { createTinyBvhBuilder } from '../src/tinybvh.ts';
import { createRandom, randomUnitVector } from '../src/trace/oracle-fixtures.ts';

const args = process.argv.slice(2);
const option = (name: string, fallback: string) => {
    const at = args.indexOf(`--${name}`);
    return at >= 0 ? args[at + 1] : fallback;
};
const assets = process.env.BVH_ASSETS_DIR ?? join(homedir(), 'Developer/webgpu-ddgi-radiance-cache/assets');
const objPath = option('obj', join(assets, 'downloads/furnished-room/breakfast_room.obj'));
const queries = Number(option('queries', '20000'));
const rounds = Number(option('rounds', '5'));

function loadMesh(): { input: MeshInput; name: string } {
    let geometry: BufferGeometry;
    let name: string;
    if (existsSync(objPath)) {
        const group = new OBJLoader().parse(readFileSync(objPath, 'utf8'));
        const parts: BufferGeometry[] = [];
        group.traverse((object) => {
            const mesh = object as Mesh;
            if (!mesh.isMesh) return;
            const part = new BufferGeometry();
            part.setAttribute('position', mesh.geometry.getAttribute('position'));
            parts.push(part);
        });
        geometry = mergeGeometries(parts);
        name = 'furnished room';
    } else {
        geometry = new TorusKnotGeometry(1, 0.3, 400, 64);
        name = 'torus knot (furnished room not found)';
    }
    const positions = new Float32Array(geometry.getAttribute('position').array);
    const index = geometry.getIndex();
    return { input: { positions, indices: index ? new Uint32Array(index.array) : undefined }, name };
}

const { input, name } = loadMesh();
const builder = await createTinyBvhBuilder();
const layout = CpuMeshBvh.build(input, builder).layout;
const clone = () => ({ ...layout, nodes: layout.nodes.slice(0), triangles: layout.triangles.slice(0), parents: layout.parents.slice(), bounds: layout.bounds.slice() });
const records = new CpuMeshBvh(clone(), { source: input });
const positions = new CpuMeshBvh(clone(), { source: input, corners: 'positions' });
const box = records.getBoundingBox();
const extent = Math.max(box[3] - box[0], box[4] - box[1], box[5] - box[2]);

const random = createRandom(1);
const point = () => [0, 1, 2].map((axis) => box[axis] + random() * (box[axis + 3] - box[axis]));
const rays = Array.from({ length: queries }, () => ({ origin: point(), direction: randomUnitVector(random) }));
const points = Array.from({ length: queries }, point);
const boxes = points.map((p) => {
    const h = extent * 0.01;
    return { min: { x: p[0] - h, y: p[1] - h, z: p[2] - h }, max: { x: p[0] + h, y: p[1] + h, z: p[2] + h } };
});

type Query = (tree: CpuMeshBvh) => unknown;
const cases: [string, number, Query][] = [
    ['raycastFirst', queries, (tree) => rays.map((ray) => tree.raycastFirst(ray, 2)?.t ?? -1)],
    ['raycast (all hits)', queries / 4, (tree) => rays.slice(0, queries / 4).map((ray) => tree.raycast(ray, 2).length)],
    ['closestPointToPoint', queries, (tree) => points.map((p) => tree.closestPointToPoint(p)?.distance ?? -1)],
    ['intersectsBox', queries, (tree) => boxes.map((b) => tree.intersectsBox(b))],
    ['refit', 10, (tree) => {
        for (let i = 0; i < 10; i++) tree.refit();
        return Array.from(tree.layout.bounds);
    }],
];

console.log(`${name}: ${records.triangleCount} triangles, ${input.positions.length / 3} vertices; ${rounds} rounds, ${queries} queries`);
console.log(`bytes: records ${records.estimateMemoryInBytes()}, positions ${positions.estimateMemoryInBytes()} (triangles ${records.layout.triangles.byteLength} against ${positions.layout.triangles.byteLength})`);
for (const [label, count, query] of cases) {
    const ratios: number[] = [];
    const times: [number[], number[]] = [[], []];
    for (let round = 0; round < rounds; round++) {
        const order = round % 2 === 0 ? [0, 1] : [1, 0];
        const answers: string[] = [];
        for (const which of order) {
            const tree = which === 0 ? records : positions;
            const start = performance.now();
            const answer = query(tree);
            times[which].push(performance.now() - start);
            answers[which] = JSON.stringify(answer);
        }
        if (answers[0] !== answers[1]) throw new Error(`${label}: the two trees disagree`);
        ratios.push(times[1][round] / times[0][round]);
    }
    const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
    const perQuery = (values: number[]) => ((median(values) / count) * 1000).toFixed(2);
    console.log(`${label}: records ${perQuery(times[0])} us, positions ${perQuery(times[1])} us per query; positions/records time ${median(ratios).toFixed(3)} (rounds ${ratios.map((r) => r.toFixed(2)).join(', ')})`);
}

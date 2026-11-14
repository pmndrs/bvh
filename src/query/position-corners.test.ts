/**
 * `CpuMeshBvh` with `corners: 'positions'` (docs/internals/design/indexed-leaves.md,
 * "CpuMeshBvh"): 16-byte indexed triangles, corners read through their
 * vertex indices from the source's own positions. The records path
 * (`corners: 'records'`, the default) is the twin: every query must give its
 * answer exactly (the same booleans, hits, `t`, points and distances bit for
 * bit, the same triangle on ties), before and after refits.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import { GPU_LAYOUT_INDEXED_TRIANGLE_BYTES, GPU_LAYOUT_TRIANGLE_BYTES } from '../gpu-layout.ts';
import { indexedFromRecords } from '../indexed-leaves.ts';
import { buildLbvhOnCpu } from '../lbvh-cpu.ts';
import type { MeshInput } from '../mesh-input.ts';
import { createTinyBvhBuilder, type TinyBvhBuilder } from '../tinybvh.ts';
import { createRandom, randomTriangleSoup, randomUnitVector, tessellatedQuad, type Random } from '../trace/oracle-fixtures.ts';
import { BVH_BACK_SIDE, BVH_DOUBLE_SIDE, BVH_FRONT_SIDE, CpuMeshBvh, type BvhClosestPoint } from './cpu-mesh-bvh.ts';
import { getBVHExtremes, validateBounds } from './debug.ts';
import { type BvhBox3Like } from './shapes.ts';

let builder: TinyBvhBuilder;
beforeAll(async () => {
    builder = await createTinyBvhBuilder();
});

/** A soup with degenerate triangles (repeated, collinear, point), as fast-paths.test.ts makes them. */
function soupWithDegenerates(random: Random, count: number): MeshInput {
    const { positions } = randomTriangleSoup(random, count, 4, 0.8);
    for (let triangle = 0; triangle < count; triangle += 7) {
        const base = triangle * 9;
        const kind = (triangle / 7) % 3;
        if (kind === 0) positions.copyWithin(base + 3, base, base + 3);
        if (kind === 1) for (let axis = 0; axis < 3; axis++) positions[base + 6 + axis] = 2 * positions[base + 3 + axis] - positions[base + axis];
        if (kind === 2) for (let corner = 1; corner < 3; corner++) positions.copyWithin(base + corner * 3, base, base + 3);
    }
    return { positions };
}

/** A wavy indexed grid whose positions are a view into a bigger buffer (byteOffset not 0), with 16-bit indices. */
function viewedGrid(): MeshInput {
    const grid = tessellatedQuad(12, 2);
    const padded = new Float32Array(grid.positions.length + 7);
    padded.set(grid.positions, 5);
    const positions = padded.subarray(5, 5 + grid.positions.length);
    for (let i = 2; i < positions.length; i += 3) positions[i] = Math.fround(Math.sin(positions[i - 2] * 3) * 0.2);
    return { positions, indices: Uint16Array.from(grid.indices!) };
}

interface Pair {
    readonly name: string;
    readonly mesh: MeshInput;
    readonly records: CpuMeshBvh;
    readonly positions: CpuMeshBvh;
}

/** Each mesh as a records tree and a positions tree, built three ways: tinybvh records, tinybvh indexed, the LBVH twin (children before parents). */
function pairs(): Pair[] {
    const random = createRandom(7);
    const meshes: [string, MeshInput][] = [
        ['indexed grid', tessellatedQuad(16, 2)],
        ['soup with degenerates', soupWithDegenerates(random, 600)],
        ['viewed 16-bit grid', viewedGrid()],
        ['one triangle', { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]) }],
    ];
    return meshes.flatMap(([name, mesh]) => [
        { name: `${name}, tinybvh`, mesh, records: CpuMeshBvh.build(mesh, builder), positions: new CpuMeshBvh(CpuMeshBvh.build(mesh, builder).layout, { source: mesh, corners: 'positions' }) },
        { name: `${name}, tinybvh indexed`, mesh, records: CpuMeshBvh.build(mesh, builder), positions: CpuMeshBvh.build(mesh, builder, { corners: 'positions' }) },
        { name: `${name}, LBVH`, mesh, records: new CpuMeshBvh(buildLbvhOnCpu(mesh), { source: mesh }), positions: new CpuMeshBvh(buildLbvhOnCpu(mesh), { source: mesh, corners: 'positions' }) },
    ]);
}

function meshBox(mesh: MeshInput): number[] {
    const out = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    for (let i = 0; i < mesh.positions.length; i++) {
        out[i % 3] = Math.min(out[i % 3], mesh.positions[i]);
        out[3 + (i % 3)] = Math.max(out[3 + (i % 3)], mesh.positions[i]);
    }
    return out;
}

function randomPoint(random: Random, box: readonly number[], margin: number): [number, number, number] {
    return [0, 1, 2].map((axis) => box[axis] - margin + random() * (box[axis + 3] - box[axis] + 2 * margin)) as [number, number, number];
}

function closest(tree: CpuMeshBvh, point: readonly number[], min = 0, max = Infinity): (BvhClosestPoint & { point: { x: number; y: number; z: number } }) | null {
    return tree.closestPointToPoint(point, {}, min, max) as never;
}

/** Everything the queries answer for one tree, as plain data (a records and a positions tree must give equal data). */
function answers(tree: CpuMeshBvh, mesh: MeshInput, other: CpuMeshBvh, seed: number): unknown[] {
    const random = createRandom(seed);
    const box = meshBox(mesh);
    const extent = Math.max(box[3] - box[0], box[4] - box[1], box[5] - box[2], 1);
    const out: unknown[] = [];
    for (let i = 0; i < 120; i++) {
        const origin = randomPoint(random, box, extent);
        const target = randomPoint(random, box, 0);
        const direction = [target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]];
        const side = [BVH_FRONT_SIDE, BVH_BACK_SIDE, BVH_DOUBLE_SIDE][i % 3];
        out.push(tree.raycast({ origin, direction }, side), tree.raycastFirst({ origin, direction }, side, 0.1, 1e6), tree.rayCrossings(origin, randomUnitVector(random)));
        const accepted = tree.intersectRay(origin as [number, number, number], direction as [number, number, number], 0, Infinity, false, (triangle) => triangle % 3 !== 0, BVH_DOUBLE_SIDE);
        out.push(accepted);
        const point = randomPoint(random, box, extent * 0.2);
        const hit = closest(tree, point);
        out.push(hit && [hit.point.x, hit.point.y, hit.point.z, hit.distance, hit.faceIndex], closest(tree, point, 0.05, 0.3));
        const h = 0.02 + random() * 0.4;
        const query: BvhBox3Like = { min: { x: point[0] - h, y: point[1] - h, z: point[2] - h }, max: { x: point[0] + h, y: point[1] + h, z: point[2] + h } };
        const turn = Math.cos(i), sine = Math.sin(i);
        out.push(tree.intersectsBox(query), tree.intersectsBox(query, [turn, sine, 0, 0, -sine, turn, 0, 0, 0, 0, 1, 0, 0.1, 0, 0, 1]));
        out.push(tree.intersectsSphere({ center: { x: point[0], y: point[1], z: point[2] }, radius: h }));
        if (i % 10 === 0) out.push(tree.isPointInside(point), tree.isPointInside(point, 'winding'));
    }
    // Shapecast: every triangle handed over, in walk order, with its corners and index.
    const visited: number[] = [];
    tree.shapecast({
        intersectsBounds: (bounds) => bounds.min.x < (box[0] + box[3]) / 2 + 0.3,
        intersectsTriangle: (triangle, index) => {
            visited.push(index, triangle.a.x, triangle.a.y, triangle.a.z, triangle.b.x, triangle.b.y, triangle.b.z, triangle.c.x, triangle.c.y, triangle.c.z);
            return false;
        },
    });
    out.push(visited);
    const ranges: number[] = [];
    tree.traverse({ intersectsBounds: () => true, intersectsRange: (offset, count) => void ranges.push(offset, count) });
    out.push(ranges);
    // Every tree position's corners and triangle, both ways round.
    const corners: number[] = [];
    for (let position = 0; position < tree.triangleCount; position++) {
        corners.push(...tree.getTriangle(position, new Array<number>(9)), tree.resolveTriangleIndex(position));
        corners.push(tree.treePositionOf(tree.resolveTriangleIndex(position)));
    }
    out.push(corners);
    const info = tree.getTriangleHitPointInfo(randomPoint(random, box, 0), tree.resolveTriangleIndex(0));
    out.push(info);
    // Against another tree (bvhcast, geometry) and against a mesh.
    const pairsFound: number[] = [];
    tree.bvhcast(other, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.05, 0.02, 0.01, 1], { intersectsTriangles: (_a, _b, i1, i2) => void pairsFound.push(i1, i2) });
    out.push(pairsFound, tree.intersectsGeometry(other), tree.intersectsGeometry(mesh, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.3, 0, 0, 1]));
    const near1 = {}, near2 = {};
    out.push(tree.closestPointToGeometry(other, [0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, extent * 0.6, 1], near1, near2), near2);
    out.push(tree.getBoundingBox(), getBVHExtremes(tree), validateBounds(tree));
    const range = new Float64Array(6);
    tree.writePrimitiveRangeBounds(0, Math.min(5, tree.triangleCount), range);
    out.push(Array.from(range));
    return out;
}

/** Answers with every number turned into its bits, so -0 and NaN compare exactly too. */
function exact(value: unknown): unknown {
    return JSON.parse(JSON.stringify(value, (_key, v: unknown) => (typeof v === 'number' ? `${Object.is(v, -0) ? '-0' : String(v)}` : v instanceof Float32Array || v instanceof Float64Array ? Array.from(v, (x) => String(x)) : v)));
}

describe("CpuMeshBvh, corners: 'positions'", () => {
    it('keeps 16-byte indexed triangles in tree order and answers every query as the records tree does, bit for bit', () => {
        for (const { name, mesh, records, positions } of pairs()) {
            expect(records.layout.triangleLeaves, name).toBeUndefined();
            expect(positions.layout.triangleLeaves, name).toBe('indexed');
            expect(positions.layout.triangles.byteLength, name).toBe(positions.triangleCount * GPU_LAYOUT_INDEXED_TRIANGLE_BYTES);
            expect(records.estimateMemoryInBytes() - positions.estimateMemoryInBytes(), name).toBe(records.triangleCount * (GPU_LAYOUT_TRIANGLE_BYTES - GPU_LAYOUT_INDEXED_TRIANGLE_BYTES));
            // The same nodes after the constructor's checks and depth-first reorder (records moved 48 bytes, indexed 16).
            expect(new Uint8Array(positions.layout.nodes), name).toEqual(new Uint8Array(records.layout.nodes));
            expect(exact(answers(positions, mesh, positions, 3)), name).toEqual(exact(answers(records, mesh, records, 3)));
        }
    });

    it('refits from moved positions to the records tree\'s boxes bit for bit, in place or from a new array, and then answers as it does', () => {
        for (const { name, mesh, records, positions } of pairs().filter((pair) => !pair.name.startsWith('one triangle'))) {
            // A copy the test may edit in place (the positions tree reads it by reference).
            let source: MeshInput = { positions: mesh.positions.slice(), indices: mesh.indices };
            const live = new CpuMeshBvh(positions.layout, { source, corners: 'positions' });
            for (let round = 0; round < 3; round++) {
                const moved = mesh.positions.map((v, i) => Math.fround(v + Math.sin(i * 0.37 + round) * 0.3));
                records.refit({ positions: moved, indices: mesh.indices });
                if (round === 1) {
                    // In place: the tree's source edited, then refit() with no argument.
                    source.positions.set(moved);
                    live.refit();
                } else {
                    source = { positions: moved, indices: mesh.indices };
                    live.refit(source);
                }
                expect(new Uint8Array(live.layout.nodes), `${name}, round ${round}`).toEqual(new Uint8Array(records.layout.nodes));
                expect(Array.from(live.layout.bounds), `${name}, round ${round}`).toEqual(Array.from(records.layout.bounds));
                const current = { positions: source.positions, indices: mesh.indices };
                expect(exact(answers(live, current, live, 11 + round)), `${name}, round ${round}`).toEqual(exact(answers(records, current, records, 11 + round)));
            }
        }
    });

    it('serializes an indexed file that loads back with the same geometry, and converts a records tree to the same triangles', () => {
        const grid = tessellatedQuad(10, 1);
        const records = CpuMeshBvh.build(grid, builder);
        const positions = CpuMeshBvh.build(grid, builder, { corners: 'positions' });
        const loaded = CpuMeshBvh.deserialize(positions.serialize(), { source: grid, corners: 'positions' });
        expect(loaded.layout.triangleLeaves).toBe('indexed');
        expect(new Uint8Array(loaded.layout.triangles)).toEqual(new Uint8Array(positions.layout.triangles));
        // Loaded the default way, the same file is expanded to the records tree's bytes.
        const expanded = CpuMeshBvh.deserialize(positions.serialize(), { source: grid });
        expect(new Uint8Array(expanded.layout.triangles)).toEqual(new Uint8Array(records.layout.triangles));
        // A records tree taken with corners: 'positions' holds what indexedFromRecords makes of it.
        const converted = new CpuMeshBvh(CpuMeshBvh.build(grid, builder).layout, { source: grid, corners: 'positions' });
        expect(new Uint8Array(converted.layout.triangles)).toEqual(new Uint8Array(indexedFromRecords(records.layout, grid).triangles));
    });

    it('needs its source, refuses an index past the positions, and a refit that throws leaves the tree as it was', () => {
        const grid = tessellatedQuad(4, 1);
        const layout = CpuMeshBvh.build(grid, builder, { corners: 'positions' }).layout;
        expect(() => new CpuMeshBvh(CpuMeshBvh.build(grid, builder).layout, { corners: 'positions' })).toThrow(/needs the geometry/);
        expect(() => new CpuMeshBvh(layout, { source: grid, corners: 'triangles' as never })).toThrow(/corners must be/);
        const copy = () => ({ ...layout, nodes: layout.nodes.slice(0), triangles: layout.triangles.slice(0), parents: layout.parents.slice(), bounds: layout.bounds.slice() });
        expect(() => new CpuMeshBvh(copy(), { source: { positions: grid.positions.subarray(0, 6) }, corners: 'positions' })).toThrow(/past the 2 positions/);
        const tree = new CpuMeshBvh(copy(), { source: grid, corners: 'positions' });
        const before = new Uint8Array(tree.serialize());
        const used = grid.positions.slice();
        used[4] = Infinity;
        expect(() => tree.refit({ positions: used, indices: grid.indices })).toThrow(/vertex 1 has a non-finite position/);
        expect(() => tree.refit({ positions: grid.positions.subarray(0, 9), indices: grid.indices })).toThrow(/past the 3 positions/);
        expect(new Uint8Array(tree.serialize())).toEqual(before);
        // An unused non-finite vertex is allowed, as on the records path.
        const extra = new Float32Array(grid.positions.length + 3);
        extra.set(grid.positions);
        extra.fill(NaN, grid.positions.length);
        expect(() => tree.refit({ positions: extra, indices: grid.indices })).not.toThrow();
    });
});

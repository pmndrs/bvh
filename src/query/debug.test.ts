/**
 * three-mesh-bvh's debug functions on our trees: `getBVHExtremes` on a tree
 * whose figures are known by hand and on larger ones (its invariants),
 * `validateBounds` on sound and broken trees, `estimateMemoryInBytes`, and
 * the node walk they use (`traverseNodes`).
 */

import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { CpuMeshBvh } from './cpu-mesh-bvh.ts';
import { estimateMemoryInBytes, getBVHExtremes, validateBounds } from './debug.ts';
import { createTinyBvhBuilder, type TinyBvhBuilder } from '../tinybvh.ts';
import { createRandom, randomTriangleSoup } from '../trace/oracle-fixtures.ts';

let builder: TinyBvhBuilder;
beforeAll(async () => {
    builder = await createTinyBvhBuilder();
});
afterEach(() => {
    vi.restoreAllMocks();
});

/** Two unit right triangles in the z = 0 plane, 10 apart along x. */
function twoTriangles(): CpuMeshBvh {
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 10, 0, 0, 11, 0, 0, 10, 1, 0]);
    return CpuMeshBvh.build({ positions }, builder, { maxLeafPrimitives: 1 });
}

function soup(count: number, seed: number): CpuMeshBvh {
    return CpuMeshBvh.build(randomTriangleSoup(createRandom(seed), count, 1, 0.1), builder);
}

describe('traverseNodes', () => {
    it('walks the root at depth 0 with the tree bounds, then children left before right, and stops descending on a truthy return', () => {
        const tree = twoTriangles();
        const visits: [number, boolean, number[], number, number][] = [];
        tree.traverseNodes((depth, isLeaf, box, offsetOrSplit, count) => {
            visits.push([depth, isLeaf, [...box], offsetOrSplit, count]);
        });
        expect(visits.length).toBe(3);
        expect(visits[0]).toEqual([0, false, [0, 0, 0, 11, 1, 0], 0, 0]);
        expect(visits[1][0]).toBe(1);
        expect(visits[1][1]).toBe(true);
        expect(visits[2][1]).toBe(true);
        expect([visits[1][3], visits[2][3]].sort()).toEqual([0, 1]);
        let seen = 0;
        tree.traverseNodes(() => {
            seen++;
            return true;
        });
        expect(seen).toBe(1);
        const big = soup(2000, 3);
        let deepest = 0;
        big.traverseNodes((depth) => {
            deepest = Math.max(deepest, depth);
            return depth >= 2;
        });
        expect(deepest).toBe(2);
    });

    it('reports a one-leaf tree as its root leaf', () => {
        const tree = CpuMeshBvh.build({ positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]) }, builder);
        const visits: unknown[] = [];
        tree.traverseNodes((depth, isLeaf, box, offset, count) => {
            visits.push([depth, isLeaf, [...box], offset, count]);
        });
        expect(visits).toEqual([[0, true, [0, 0, 0, 1, 1, 0], 0, 1]]);
        expect(getBVHExtremes(tree)).toEqual([{ nodeCount: 1, leafNodeCount: 1, depth: { min: 0, max: 0 }, primitives: { min: 1, max: 1 }, splits: [0, 0, 0], surfaceAreaScore: 2 * 1.25 }]);
    });
});

describe('getBVHExtremes', () => {
    it('gives the figures of a known tree', () => {
        // Root box 11 x 1 x 0 (area 22, inner: x1), two leaves 1 x 1 x 0 (area 2, one triangle: x1.25 each), split along x.
        expect(getBVHExtremes(twoTriangles())).toEqual([{ nodeCount: 3, leafNodeCount: 2, depth: { min: 1, max: 1 }, primitives: { min: 1, max: 1 }, splits: [1, 0, 0], surfaceAreaScore: 22 + 2 * 2 * 1.25 }]);
    });

    it('agrees with the tree on larger meshes', () => {
        for (const [count, seed] of [
            [500, 1],
            [5000, 2],
        ]) {
            const tree = soup(count, seed);
            const [extremes] = getBVHExtremes(tree);
            expect(extremes.nodeCount).toBe(2 * extremes.leafNodeCount - 1);
            expect(extremes.splits[0] + extremes.splits[1] + extremes.splits[2]).toBe(extremes.leafNodeCount - 1);
            expect(extremes.depth.max).toBe(tree.depth);
            expect(extremes.primitives.min).toBeGreaterThanOrEqual(1);
            expect(extremes.primitives.max).toBeLessThanOrEqual(8);
            let triangles = 0;
            tree.traverseNodes((_depth, isLeaf, _box, _offset, leafCount) => {
                if (isLeaf) triangles += leafCount;
            });
            expect(triangles).toBe(count);
            expect(extremes.surfaceAreaScore).toBeGreaterThan(0);
        }
    });
});

describe('validateBounds', () => {
    it('passes a built tree and fails one whose boxes no longer hold their triangles', () => {
        const assert = vi.spyOn(console, 'assert').mockImplementation(() => {});
        const input = randomTriangleSoup(createRandom(9), 1000, 1, 0.1);
        const tree = CpuMeshBvh.build(input, builder);
        expect(validateBounds(tree)).toBe(true);
        expect(assert.mock.calls.every(([condition]) => condition)).toBe(true);

        // Positions edited in place: the source no longer fits until a refit.
        const kept = input.positions.slice();
        for (let i = 0; i < 9; i++) input.positions[i] += 5;
        expect(validateBounds(tree)).toBe(false);
        expect(assert.mock.calls.some(([condition, message]) => !condition && message === 'Leaf bounds does not fully contain primitives.')).toBe(true);
        tree.refit();
        expect(validateBounds(tree)).toBe(true);
        input.positions.set(kept);
        tree.refit();

        // A tree without a source, with a child box shrunk: its subtree no longer fits.
        const copy = new CpuMeshBvh({ ...tree.layout, nodes: tree.layout.nodes.slice(0), triangles: tree.layout.triangles.slice(0), parents: tree.layout.parents.slice(), bounds: tree.layout.bounds.slice() });
        expect(validateBounds(copy)).toBe(true);
        const floats = new Float32Array(copy.layout.nodes);
        floats[4] = floats[0]; // node 0's left child: max x = min x
        assert.mockClear();
        expect(validateBounds(copy)).toBe(false);
    });
});

describe('estimateMemoryInBytes', () => {
    it('is the tree method, at least the layout', () => {
        const tree = soup(1000, 4);
        expect(estimateMemoryInBytes(tree)).toBe(tree.estimateMemoryInBytes());
        expect(estimateMemoryInBytes(tree)).toBeGreaterThanOrEqual(tree.layout.nodes.byteLength + tree.layout.triangles.byteLength);
    });
});

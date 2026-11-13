import { describe, expect, it } from 'vitest';
import { wgslNameProblems } from './wgsl-names.ts';
import { GPU_LAYOUT_EMPTY_CHILD, GPU_LAYOUT_NO_PARENT, GPU_LAYOUT_NODE_WORD, GPU_LAYOUT_NODE_WORDS } from './gpu-layout.ts';
import { bvhRefitWgsl } from './gpu-refit.ts';
import { BVH_FLOAT_MAX, halfArea, leafSlotsOf, parentSlotsOf, PRIMITIVE_BOX_FLOATS, refitOnCpu, refitOrderOf, refitRecordsOnCpu, relativeSahCost } from './refit-cpu.ts';
import { createTinyBvhBuilder } from './tinybvh.ts';
import { createRandom, randomTriangleSoup } from './trace/oracle-fixtures.ts';

/**
 * A three-primitive tree in the GPU layout:
 *
 *     node 0: left = node 1 (inner), right = leaf of primitive 2
 *     node 1: left = leaf of primitive 0, right = leaf of primitive 1
 */
function threePrimitiveTree(): { floats: Float32Array; words: Uint32Array; parents: Uint32Array } {
    const words = new Uint32Array(2 * GPU_LAYOUT_NODE_WORDS);
    const node = (index: number, left: [number, number], right: [number, number]): void => {
        const base = index * GPU_LAYOUT_NODE_WORDS;
        words[base + GPU_LAYOUT_NODE_WORD.leftIndex] = left[0];
        words[base + GPU_LAYOUT_NODE_WORD.leftCount] = left[1];
        words[base + GPU_LAYOUT_NODE_WORD.rightIndex] = right[0];
        words[base + GPU_LAYOUT_NODE_WORD.rightCount] = right[1];
    };
    node(0, [1, 0], [2, 1]);
    node(1, [0, 1], [1, 1]);
    return { floats: new Float32Array(words.buffer), words, parents: new Uint32Array([GPU_LAYOUT_NO_PARENT, 0]) };
}

function boxes(...list: number[][]): Float32Array {
    const out = new Float32Array(list.length * PRIMITIVE_BOX_FLOATS);
    list.forEach((box, index) => {
        out.set(box.slice(0, 3), index * PRIMITIVE_BOX_FLOATS);
        out.set(box.slice(3, 6), index * PRIMITIVE_BOX_FLOATS + 4);
    });
    return out;
}

/** The six box floats of child slot `slot` (node * 2 + side). */
function slotBox(floats: Float32Array, slot: number): number[] {
    const base = (slot >> 1) * GPU_LAYOUT_NODE_WORDS + (slot & 1) * 8;
    return [floats[base], floats[base + 1], floats[base + 2], floats[base + 4], floats[base + 5], floats[base + 6]];
}

describe('refit topology', () => {
    it('lists leaf slots in primitive order and parent slots by side', () => {
        const tree = threePrimitiveTree();
        expect([...leafSlotsOf(tree.words, 0, 2)]).toEqual([2, 3, 1]);
        expect([...parentSlotsOf(tree.words, 0, 2, tree.parents)]).toEqual([GPU_LAYOUT_NO_PARENT, 0]);
    });

    it('skips the missing right child of a one-primitive tree', () => {
        const words = new Uint32Array(GPU_LAYOUT_NODE_WORDS);
        words[GPU_LAYOUT_NODE_WORD.leftCount] = 1;
        words[GPU_LAYOUT_NODE_WORD.rightIndex] = GPU_LAYOUT_EMPTY_CHILD;
        expect([...leafSlotsOf(words, 0, 1)]).toEqual([0]);
    });
});

describe('refitOnCpu', () => {
    it('gives every slot the union below it, and sums the inner nodes’ half-areas', () => {
        const tree = threePrimitiveTree();
        const result = refitOnCpu(tree.floats, tree.words, 0, 2, tree.parents, boxes([0, 0, 0, 1, 1, 1], [2, 0, 0, 3, 1, 1], [0, 5, 0, 1, 6, 1]));
        expect(slotBox(tree.floats, 2)).toEqual([0, 0, 0, 1, 1, 1]);
        expect(slotBox(tree.floats, 3)).toEqual([2, 0, 0, 3, 1, 1]);
        expect(slotBox(tree.floats, 0)).toEqual([0, 0, 0, 3, 1, 1]);
        expect(slotBox(tree.floats, 1)).toEqual([0, 5, 0, 1, 6, 1]);
        expect([...result.rootBox]).toEqual([0, 0, 0, 3, 6, 1]);
        const expectedSum = halfArea([0, 0, 0, 3, 1, 1]) + halfArea([0, 0, 0, 3, 6, 1]);
        expect(result.summedHalfArea).toBeCloseTo(expectedSum, 10);
        expect(relativeSahCost(tree.floats, tree.words, 0, 2)).toBeCloseTo(expectedSum / halfArea([0, 0, 0, 3, 6, 1]), 10);
    });

    it('stores an empty primitive as a point at the largest float, and leaves it out of unions', () => {
        const tree = threePrimitiveTree();
        const empty = [BVH_FLOAT_MAX, BVH_FLOAT_MAX, BVH_FLOAT_MAX, -BVH_FLOAT_MAX, -BVH_FLOAT_MAX, -BVH_FLOAT_MAX];
        refitOnCpu(tree.floats, tree.words, 0, 2, tree.parents, boxes([0, 0, 0, 1, 1, 1], empty, [0, 5, 0, 1, 6, 1]));
        expect(slotBox(tree.floats, 3)).toEqual(new Array(6).fill(Math.fround(BVH_FLOAT_MAX)));
        expect(slotBox(tree.floats, 0)).toEqual([0, 0, 0, 1, 1, 1]);
    });
});

describe('refitOnCpu on quantized nodes', () => {
    it('throws instead of overwriting 32-byte nodes read with the 64-byte stride', async () => {
        const tinyBvh = await createTinyBvhBuilder();
        const geometry = randomTriangleSoup(createRandom(12), 300, 4, 0.5);
        const quantized = tinyBvh.buildLayout(geometry, { nodeFormat: 'quantized' }).layout;
        const before = new Uint8Array(quantized.nodes.slice(0));
        const primitiveBoxes = new Float32Array(quantized.triangleCount * PRIMITIVE_BOX_FLOATS);
        const words = new Uint32Array(quantized.nodes);
        expect(() => refitOnCpu(new Float32Array(quantized.nodes), words, 0, quantized.nodeCount, quantized.parents, primitiveBoxes)).toThrow(/not a 64-byte f32 node/);
        // It checks every node before it writes one: the quantized words are untouched.
        expect(new Uint8Array(quantized.nodes)).toEqual(before);
    });
});

describe('refitRecordsOnCpu', () => {
    it("is refitOnCpu over boxes made from the records' corners, bit for bit, with or without a kept order", async () => {
        const builder = await createTinyBvhBuilder();
        const geometry = randomTriangleSoup(createRandom(21), 3000, 5, 1);
        const { layout } = builder.buildLayout(geometry);
        // Move every corner in the records, then refit both ways from the same records.
        const records = new Float32Array(layout.triangles.slice(0));
        for (let k = 0; k < records.length; k++) if (k % 4 !== 3) records[k] = Math.fround(records[k] * 1.3 + Math.sin(k));
        const primitiveBoxes = new Float32Array(layout.triangleCount * PRIMITIVE_BOX_FLOATS);
        for (let t = 0; t < layout.triangleCount; t++) {
            for (let axis = 0; axis < 3; axis++) {
                const values = [0, 4, 8].map((corner) => records[t * 12 + corner + axis]);
                primitiveBoxes[t * PRIMITIVE_BOX_FLOATS + axis] = Math.min(...values);
                primitiveBoxes[t * PRIMITIVE_BOX_FLOATS + 4 + axis] = Math.max(...values);
            }
        }
        const viaBoxes = layout.nodes.slice(0);
        const viaRecords = layout.nodes.slice(0);
        const a = refitOnCpu(new Float32Array(viaBoxes), new Uint32Array(viaBoxes), 0, layout.nodeCount, layout.parents, primitiveBoxes);
        const order = refitOrderOf(layout.parents, layout.nodeCount);
        const b = refitRecordsOnCpu(new Float32Array(viaRecords), new Uint32Array(viaRecords), 0, layout.nodeCount, layout.parents, records, order);
        expect(new Uint8Array(viaRecords)).toEqual(new Uint8Array(viaBoxes));
        expect([...b.rootBox]).toEqual([...a.rootBox]);
        expect(b.summedHalfArea).toBe(a.summedHalfArea);
        // Children before parents, every node once.
        const at = new Int32Array(layout.nodeCount);
        order.forEach((node, position) => (at[node] = position));
        expect(new Set(order).size).toBe(layout.nodeCount);
        for (let node = 1; node < layout.nodeCount; node++) expect(at[node]).toBeLessThan(at[layout.parents[node]]);
        expect(() => refitOnCpu(new Float32Array(viaBoxes), new Uint32Array(viaBoxes), 0, layout.nodeCount, layout.parents, primitiveBoxes, order.subarray(1))).toThrow(RangeError);
    });
});

/** The refit before the CPU speed-up (per-node sort, box arrays), kept as the reference the fast ones must match byte for byte. */
function oldRefitOnCpu(words: Uint32Array, floats: Float32Array, nodeCount: number, parents: Uint32Array, primitiveBoxes: Float32Array): { rootBox: number[]; summedHalfArea: number } {
    type OldBox = number[];
    const empty = (): OldBox => [BVH_FLOAT_MAX, BVH_FLOAT_MAX, BVH_FLOAT_MAX, -BVH_FLOAT_MAX, -BVH_FLOAT_MAX, -BVH_FLOAT_MAX];
    const union = (t: OldBox, o: OldBox): void => {
        for (let a = 0; a < 3; a++) {
            t[a] = Math.min(t[a], o[a]);
            t[a + 3] = Math.max(t[a + 3], o[a + 3]);
        }
    };
    const depth = new Uint32Array(nodeCount);
    for (let node = 0; node < nodeCount; node++) {
        let level = 0;
        for (let up = parents[node]; up !== GPU_LAYOUT_NO_PARENT; up = parents[up]) level++;
        depth[node] = level;
    }
    const order = Array.from({ length: nodeCount }, (_, node) => node).sort((a, b) => depth[b] - depth[a]);
    const nodeBoxes: OldBox[] = [];
    let summedHalfArea = 0;
    for (const node of order) {
        const base = node * GPU_LAYOUT_NODE_WORDS;
        const box = empty();
        for (let side = 0; side < 2; side++) {
            const index = words[base + (side === 0 ? GPU_LAYOUT_NODE_WORD.leftIndex : GPU_LAYOUT_NODE_WORD.rightIndex)];
            if (index === GPU_LAYOUT_EMPTY_CHILD) continue;
            const count = words[base + (side === 0 ? GPU_LAYOUT_NODE_WORD.leftCount : GPU_LAYOUT_NODE_WORD.rightCount)];
            let child: OldBox;
            if (count > 0) {
                child = empty();
                for (let p = index; p < index + count; p++) {
                    const at = p * PRIMITIVE_BOX_FLOATS;
                    union(child, [primitiveBoxes[at], primitiveBoxes[at + 1], primitiveBoxes[at + 2], primitiveBoxes[at + 4], primitiveBoxes[at + 5], primitiveBoxes[at + 6]]);
                }
            } else child = nodeBoxes[index];
            const slot = base + side * 8;
            const stored = child[0] > child[3] ? new Array<number>(6).fill(BVH_FLOAT_MAX) : child;
            floats[slot] = stored[0];
            floats[slot + 1] = stored[1];
            floats[slot + 2] = stored[2];
            floats[slot + 4] = stored[3];
            floats[slot + 5] = stored[4];
            floats[slot + 6] = stored[5];
            union(box, child);
        }
        nodeBoxes[node] = box;
        summedHalfArea += halfArea(box);
    }
    return { rootBox: nodeBoxes[0], summedHalfArea };
}

describe('refit against the previous implementation', () => {
    it('refitOnCpu and refitRecordsOnCpu match it byte for byte on random trees', async () => {
        const builder = await createTinyBvhBuilder();
        for (const [seed, count] of [[1, 1], [2, 2], [3, 7], [4, 100], [5, 1500]] as const) {
            const geometry = randomTriangleSoup(createRandom(seed), count, 5, 1);
            const { layout } = builder.buildLayout(geometry);
            const records = new Float32Array(layout.triangles.slice(0));
            for (let k = 0; k < records.length; k++) if (k % 4 !== 3) records[k] = Math.fround(records[k] * 0.7 + Math.cos(k + seed));
            const primitiveBoxes = new Float32Array(layout.triangleCount * PRIMITIVE_BOX_FLOATS);
            for (let t = 0; t < layout.triangleCount; t++) {
                for (let axis = 0; axis < 3; axis++) {
                    const values = [0, 4, 8].map((corner) => records[t * 12 + corner + axis]);
                    primitiveBoxes[t * PRIMITIVE_BOX_FLOATS + axis] = Math.min(...values);
                    primitiveBoxes[t * PRIMITIVE_BOX_FLOATS + 4 + axis] = Math.max(...values);
                }
            }
            // Every seventh primitive box empty, to cover the empty-slot form (boxes only; records have no empty form).
            const withEmpties = primitiveBoxes.slice();
            for (let t = 0; t < layout.triangleCount; t += 7) {
                withEmpties.fill(BVH_FLOAT_MAX, t * PRIMITIVE_BOX_FLOATS, t * PRIMITIVE_BOX_FLOATS + 3);
                withEmpties.fill(-BVH_FLOAT_MAX, t * PRIMITIVE_BOX_FLOATS + 4, t * PRIMITIVE_BOX_FLOATS + 7);
            }
            for (const boxesIn of [primitiveBoxes, withEmpties]) {
                const reference = layout.nodes.slice(0);
                const expected = oldRefitOnCpu(new Uint32Array(reference), new Float32Array(reference), layout.nodeCount, layout.parents, boxesIn);
                const viaBoxes = layout.nodes.slice(0);
                const a = refitOnCpu(new Float32Array(viaBoxes), new Uint32Array(viaBoxes), 0, layout.nodeCount, layout.parents, boxesIn);
                expect(new Uint8Array(viaBoxes)).toEqual(new Uint8Array(reference));
                expect([...a.rootBox]).toEqual(expected.rootBox);
                expect(a.summedHalfArea).toBe(expected.summedHalfArea);
            }
            const reference = layout.nodes.slice(0);
            const expected = oldRefitOnCpu(new Uint32Array(reference), new Float32Array(reference), layout.nodeCount, layout.parents, primitiveBoxes);
            const viaRecords = layout.nodes.slice(0);
            const b = refitRecordsOnCpu(new Float32Array(viaRecords), new Uint32Array(viaRecords), 0, layout.nodeCount, layout.parents, records);
            expect(new Uint8Array(viaRecords)).toEqual(new Uint8Array(reference));
            expect([...b.rootBox]).toEqual(expected.rootBox);
            expect(b.summedHalfArea).toBe(expected.summedHalfArea);
        }
    });
});

describe('refit WGSL', () => {
    const wgsl = bvhRefitWgsl();

    it('needs subgroups and declares both kernels', () => {
        expect(wgsl.startsWith('enable subgroups;')).toBe(true);
        expect(wgsl).toContain('fn bvhRefitClear(');
        expect(wgsl).toContain('fn bvhRefitClimb(');
        expect(wgsl).toContain('subgroupShuffleDown(');
    });

    it('uses no WGSL reserved word as a name', () => {
        expect(wgslNameProblems(wgsl)).toEqual([]);
    });
});

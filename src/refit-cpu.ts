/**
 * CPU twins of the GPU refit (gpu-refit.ts), for trees in the tracer's GPU
 * layout (gpu-layout.ts):
 *
 * - {@link leafSlotsOf}: which child slots are leaves, in primitive order. The
 *   GPU refit starts one thread per leaf slot.
 * - {@link relativeSahCost}: the tree-quality signal a refit is judged by.
 * - {@link refitOnCpu}: the same bottom-up refit the GPU does, written plainly.
 *   Box unions are exact (min and max of f32 values), so a correct GPU refit
 *   matches it bit for bit; the GPU test pages compare the two.
 *
 * A child slot is `node * 2 + side` (side 0 left, 1 right).
 */

import {
    GPU_LAYOUT_CHILD_STRIDE_WORDS,
    GPU_LAYOUT_EMPTY_CHILD,
    GPU_LAYOUT_NO_PARENT,
    GPU_LAYOUT_NODE_WORD,
    GPU_LAYOUT_NODE_WORDS,
} from './gpu-layout.ts';

/** Largest finite f32. An empty box is stored as a point here, which no ray can reach (see gpu-refit.ts). */
export const BVH_FLOAT_MAX = 3.4028234663852886e38;

/** Floats per primitive box in the refit's input: `(minX, minY, minZ, unused, maxX, maxY, maxZ, unused)`. */
export const PRIMITIVE_BOX_FLOATS = 8;

/** The child slot's `index` word. */
function indexWord(side: number): number {
    return side === 0 ? GPU_LAYOUT_NODE_WORD.leftIndex : GPU_LAYOUT_NODE_WORD.rightIndex;
}

/** The child slot's `count` word. */
function countWord(side: number): number {
    return side === 0 ? GPU_LAYOUT_NODE_WORD.leftCount : GPU_LAYOUT_NODE_WORD.rightCount;
}

/**
 * Every child slot that is a leaf, as `node * 2 + side`, ordered by the
 * leaf's first primitive. Builders store primitives in tree order, so
 * neighbouring entries are often siblings, which the GPU refit's subgroup
 * step pairs up.
 */
export function leafSlotsOf(words: Uint32Array, nodeWordOffset: number, nodeCount: number): Uint32Array {
    const leaves: { first: number; slot: number }[] = [];
    for (let node = 0; node < nodeCount; node++) {
        const base = nodeWordOffset + node * GPU_LAYOUT_NODE_WORDS;
        for (let side = 0; side < 2; side++) {
            const index = words[base + indexWord(side)];
            if (index === GPU_LAYOUT_EMPTY_CHILD) continue;
            if (words[base + countWord(side)] > 0) leaves.push({ first: index, slot: node * 2 + side });
        }
    }
    leaves.sort((a, b) => a.first - b.first);
    return Uint32Array.from(leaves, (leaf) => leaf.slot);
}

/**
 * Converts a parent array (parent node per node, as builders write it) into
 * parent slots, which say which side of the parent each node hangs on.
 */
export function parentSlotsOf(words: Uint32Array, nodeWordOffset: number, nodeCount: number, parents: Uint32Array): Uint32Array {
    const slots = new Uint32Array(nodeCount);
    for (let node = 0; node < nodeCount; node++) {
        const parent = parents[node];
        if (parent === GPU_LAYOUT_NO_PARENT) {
            slots[node] = GPU_LAYOUT_NO_PARENT;
            continue;
        }
        const base = nodeWordOffset + parent * GPU_LAYOUT_NODE_WORDS;
        const isLeft = words[base + GPU_LAYOUT_NODE_WORD.leftIndex] === node && words[base + GPU_LAYOUT_NODE_WORD.leftCount] === 0;
        slots[node] = parent * 2 + (isLeft ? 0 : 1);
    }
    return slots;
}

/**
 * Nodes grouped by depth, deepest level first, for a refit that goes level by
 * level (the GPU refit's repair pass): every node comes after all its
 * descendants. `levelStarts[level]` to `levelStarts[level + 1]` index into
 * `levelNodes`; there are `levelStarts.length - 1` levels.
 */
export function levelOrderOf(parentSlots: Uint32Array, nodeCount: number): { levelStarts: Uint32Array; levelNodes: Uint32Array } {
    const depth = new Int32Array(nodeCount).fill(-1);
    const depthOf = (node: number): number => {
        // Walk up to a node of known depth, then fill the path back down (no recursion: trees can be deep).
        const path: number[] = [];
        let at = node;
        while (at !== GPU_LAYOUT_NO_PARENT && depth[at] < 0) {
            path.push(at);
            const parentSlot = parentSlots[at];
            at = parentSlot === GPU_LAYOUT_NO_PARENT ? GPU_LAYOUT_NO_PARENT : parentSlot >> 1;
        }
        let level = at === GPU_LAYOUT_NO_PARENT ? -1 : depth[at];
        for (let index = path.length - 1; index >= 0; index--) depth[path[index]] = ++level;
        return depth[node];
    };
    let deepest = -1;
    for (let node = 0; node < nodeCount; node++) deepest = Math.max(deepest, depthOf(node));
    const levelCount = deepest + 1;
    // Level 0 of the output is the deepest depth.
    const counts = new Uint32Array(levelCount);
    for (let node = 0; node < nodeCount; node++) counts[deepest - depth[node]]++;
    const levelStarts = new Uint32Array(levelCount + 1);
    for (let level = 0; level < levelCount; level++) levelStarts[level + 1] = levelStarts[level] + counts[level];
    const fill = levelStarts.slice(0, levelCount);
    const levelNodes = new Uint32Array(nodeCount);
    for (let node = 0; node < nodeCount; node++) levelNodes[fill[deepest - depth[node]]++] = node;
    return { levelStarts, levelNodes };
}

/** `[minX, minY, minZ, maxX, maxY, maxZ]`, with an empty box as `+max, -max`. */
type Box = [number, number, number, number, number, number];

function emptyBox(): Box {
    return [BVH_FLOAT_MAX, BVH_FLOAT_MAX, BVH_FLOAT_MAX, -BVH_FLOAT_MAX, -BVH_FLOAT_MAX, -BVH_FLOAT_MAX];
}

function unionInto(target: Box, other: Box): void {
    for (let axis = 0; axis < 3; axis++) {
        target[axis] = Math.min(target[axis], other[axis]);
        target[axis + 3] = Math.max(target[axis + 3], other[axis + 3]);
    }
}

/** Half the surface area: proportional to the chance that a random ray hits the box. 0 for an empty box. */
export function halfArea(box: ArrayLike<number>): number {
    if (box[0] > box[3]) return 0;
    const x = box[3] - box[0];
    const y = box[4] - box[1];
    const z = box[5] - box[2];
    return x * y + y * z + z * x;
}

function readSlot(floats: Float32Array, nodeWordOffset: number, slot: number): Box {
    const base = nodeWordOffset + (slot >> 1) * GPU_LAYOUT_NODE_WORDS + (slot & 1) * GPU_LAYOUT_CHILD_STRIDE_WORDS;
    const box: Box = [floats[base], floats[base + 1], floats[base + 2], floats[base + 4], floats[base + 5], floats[base + 6]];
    // The stored form of an empty box (a point at the largest float) reads back as empty.
    return box[0] === Math.fround(BVH_FLOAT_MAX) ? emptyBox() : box;
}

/**
 * The tree-quality signal: the summed half-area of every inner node (the
 * root included) over the root's. It is the expected number of inner nodes a
 * random ray through the root's box visits, the traversal half of the SAH
 * cost. A refit keeps the tree's shape while things move, so this grows as
 * the shape stops suiting the new positions; a rebuild resets it.
 *
 * Reads the boxes stored in the nodes. 1 for a one-node tree, 0 for an empty one.
 */
export function relativeSahCost(floats: Float32Array, words: Uint32Array, nodeWordOffset: number, nodeCount: number): number {
    if (nodeCount === 0) return 0;
    let summed = 0;
    let rootArea = 0;
    for (let node = 0; node < nodeCount; node++) {
        const box = readSlot(floats, nodeWordOffset, node * 2);
        if (words[nodeWordOffset + node * GPU_LAYOUT_NODE_WORDS + GPU_LAYOUT_NODE_WORD.rightIndex] !== GPU_LAYOUT_EMPTY_CHILD) {
            unionInto(box, readSlot(floats, nodeWordOffset, node * 2 + 1));
        }
        const area = halfArea(box);
        summed += area;
        if (node === 0) rootArea = area;
    }
    return rootArea > 0 ? summed / rootArea : 0;
}

export interface CpuRefitResult {
    /** `[minX, minY, minZ, maxX, maxY, maxZ]` of the root. */
    readonly rootBox: Float64Array;
    /** Summed half-area of every inner node, the root included (the numerator of {@link relativeSahCost}). */
    readonly summedHalfArea: number;
}

/**
 * Refits the tree in `floats`/`words` (one buffer, two views) in place from
 * `primitiveBoxes` ({@link PRIMITIVE_BOX_FLOATS} per primitive), keeping its
 * shape: each leaf slot gets the union of its primitives' boxes, each inner
 * slot the union of its node's two slots. Empty primitive boxes (`min > max`)
 * stay empty, and an empty slot is stored as a point at the largest float.
 *
 * `order` is {@link refitOrderOf}`(parents, nodeCount)`, for a caller that
 * refits the same tree again and again (it depends on the shape only);
 * without it, it is worked out here.
 */
export function refitOnCpu(
    floats: Float32Array,
    words: Uint32Array,
    nodeWordOffset: number,
    nodeCount: number,
    parents: Uint32Array,
    primitiveBoxes: Float32Array,
    order: Uint32Array = refitOrderOf(parents, nodeCount),
): CpuRefitResult {
    return refitInOrder(floats, words, nodeWordOffset, nodeCount, order, primitiveBoxes, false, 'refitOnCpu');
}

/**
 * {@link refitOnCpu} for a tree whose primitives are its own 48-byte triangle
 * records (`triangleFloats`, in tree order): each leaf's box is the min and
 * max of its records' corners, read in place, with no primitive-box array in
 * between. The result is refitOnCpu's with boxes made from the same corners
 * (unions of f32 values are exact). `CpuMeshBvh.refit` uses it.
 */
export function refitRecordsOnCpu(
    floats: Float32Array,
    words: Uint32Array,
    nodeWordOffset: number,
    nodeCount: number,
    parents: Uint32Array,
    triangleFloats: Float32Array,
    order: Uint32Array = refitOrderOf(parents, nodeCount),
): CpuRefitResult {
    return refitInOrder(floats, words, nodeWordOffset, nodeCount, order, triangleFloats, true, 'refitRecordsOnCpu');
}

/** Words of a triangle record, and where its corners start (gpu-layout.ts). */
const RECORD_WORDS = 12;
const CORNER_0 = 0;
const CORNER_1 = 4;
const CORNER_2 = 8;

/** The refit of both entry points: `primitives` are boxes ({@link PRIMITIVE_BOX_FLOATS} each) or triangle records. */
function refitInOrder(
    floats: Float32Array,
    words: Uint32Array,
    nodeWordOffset: number,
    nodeCount: number,
    order: Uint32Array,
    primitives: Float32Array,
    records: boolean,
    caller: string,
): CpuRefitResult {
    if (nodeCount === 0) return { rootBox: new Float64Array(emptyBox()), summedHalfArea: 0 };
    if (order.length !== nodeCount) throw new RangeError(`${caller}: the order has ${order.length} nodes, the tree ${nodeCount}`);
    const primitiveBoxes = primitives;
    // Each node's box, six doubles: `[minX, minY, minZ, maxX, maxY, maxZ]`. One array, no box objects.
    const nodeBoxes = new Float64Array(nodeCount * 6);
    const max = BVH_FLOAT_MAX;
    const storedMax = Math.fround(BVH_FLOAT_MAX);
    let summedHalfArea = 0;
    for (let at = 0; at < nodeCount; at++) {
        const node = order[at];
        const base = nodeWordOffset + node * GPU_LAYOUT_NODE_WORDS;
        let n0 = max, n1 = max, n2 = max, x0 = -max, x1 = -max, x2 = -max;
        for (let side = 0; side < 2; side++) {
            const index = words[base + (side === 0 ? GPU_LAYOUT_NODE_WORD.leftIndex : GPU_LAYOUT_NODE_WORD.rightIndex)];
            if (index === GPU_LAYOUT_EMPTY_CHILD) continue;
            const count = words[base + (side === 0 ? GPU_LAYOUT_NODE_WORD.leftCount : GPU_LAYOUT_NODE_WORD.rightCount)];
            let c0 = max, c1 = max, c2 = max, d0 = -max, d1 = -max, d2 = -max;
            if (count > 0 && records) {
                for (let primitive = index; primitive < index + count; primitive++) {
                    const r = primitive * RECORD_WORDS;
                    const a = primitives[r + CORNER_0], b = primitives[r + CORNER_1], c = primitives[r + CORNER_2];
                    c0 = Math.min(c0, a, b, c);
                    d0 = Math.max(d0, a, b, c);
                    const e = primitives[r + CORNER_0 + 1], f = primitives[r + CORNER_1 + 1], g = primitives[r + CORNER_2 + 1];
                    c1 = Math.min(c1, e, f, g);
                    d1 = Math.max(d1, e, f, g);
                    const h = primitives[r + CORNER_0 + 2], i = primitives[r + CORNER_1 + 2], j = primitives[r + CORNER_2 + 2];
                    c2 = Math.min(c2, h, i, j);
                    d2 = Math.max(d2, h, i, j);
                }
            } else if (count > 0) {
                for (let primitive = index; primitive < index + count; primitive++) {
                    const p = primitive * PRIMITIVE_BOX_FLOATS;
                    c0 = Math.min(c0, primitiveBoxes[p]);
                    c1 = Math.min(c1, primitiveBoxes[p + 1]);
                    c2 = Math.min(c2, primitiveBoxes[p + 2]);
                    d0 = Math.max(d0, primitiveBoxes[p + 4]);
                    d1 = Math.max(d1, primitiveBoxes[p + 5]);
                    d2 = Math.max(d2, primitiveBoxes[p + 6]);
                }
            } else {
                const b = index * 6;
                c0 = nodeBoxes[b];
                c1 = nodeBoxes[b + 1];
                c2 = nodeBoxes[b + 2];
                d0 = nodeBoxes[b + 3];
                d1 = nodeBoxes[b + 4];
                d2 = nodeBoxes[b + 5];
            }
            // An empty box is stored as a point at the largest float (no ray reaches it).
            const slot = base + side * GPU_LAYOUT_CHILD_STRIDE_WORDS;
            const empty = c0 > d0;
            floats[slot] = empty ? storedMax : c0;
            floats[slot + 1] = empty ? storedMax : c1;
            floats[slot + 2] = empty ? storedMax : c2;
            floats[slot + 4] = empty ? storedMax : d0;
            floats[slot + 5] = empty ? storedMax : d1;
            floats[slot + 6] = empty ? storedMax : d2;
            n0 = Math.min(n0, c0);
            n1 = Math.min(n1, c1);
            n2 = Math.min(n2, c2);
            x0 = Math.max(x0, d0);
            x1 = Math.max(x1, d1);
            x2 = Math.max(x2, d2);
        }
        const b = node * 6;
        nodeBoxes[b] = n0;
        nodeBoxes[b + 1] = n1;
        nodeBoxes[b + 2] = n2;
        nodeBoxes[b + 3] = x0;
        nodeBoxes[b + 4] = x1;
        nodeBoxes[b + 5] = x2;
        // halfArea: 0 for an empty box.
        if (!(n0 > x0)) {
            const x = x0 - n0;
            const y = x1 - n1;
            const z = x2 - n2;
            summedHalfArea += x * y + y * z + z * x;
        }
    }
    return { rootBox: nodeBoxes.slice(0, 6), summedHalfArea };
}

/**
 * The order {@link refitOnCpu} visits nodes in: every node, deepest first and
 * by index within a depth (a stable counting sort on depth), so children come
 * before parents. Depths are found in one pass: each walk up stops at a node
 * whose depth is known, and fills the path back down.
 */
export function refitOrderOf(parents: Uint32Array, nodeCount: number): Uint32Array {
    const depth = new Int32Array(nodeCount).fill(-1);
    const path = new Uint32Array(nodeCount);
    let deepest = 0;
    for (let node = 0; node < nodeCount; node++) {
        if (depth[node] >= 0) continue;
        let length = 0;
        let at = node;
        while (at !== GPU_LAYOUT_NO_PARENT && depth[at] < 0) {
            path[length++] = at;
            at = parents[at];
        }
        let level = at === GPU_LAYOUT_NO_PARENT ? -1 : depth[at];
        while (length > 0) depth[path[--length]] = ++level;
        if (level > deepest) deepest = level;
    }
    const starts = new Uint32Array(deepest + 2);
    for (let node = 0; node < nodeCount; node++) starts[deepest - depth[node] + 1]++;
    for (let level = 0; level <= deepest; level++) starts[level + 1] += starts[level];
    const order = new Uint32Array(nodeCount);
    for (let node = 0; node < nodeCount; node++) order[starts[deepest - depth[node]]++] = node;
    return order;
}

/**
 * Wide nodes (docs/internals/design/wide-nodes.md): a mesh's binary tree
 * collapsed into a 4-wide (`'bvh4'`) or a compressed 8-wide (`'cwbvh8'`)
 * tree, for the wide walks in `trace/wgsl-wide.ts`. Every builder still
 * writes the binary `f32` tree; a scene converts it when it is assembled or
 * placed with a wide `nodeFormat`.
 *
 * Ported from tinybvh 1.9.1 (MIT, commit c9be3103; `tiny_bvh_base.h`):
 *
 * - `collapse` is `MBVH<M>::ConvertFrom`: Ylitie, Karras and Laine 2017,
 *   section 4.2, the bottom-up dynamic program over "cost of this subtree in
 *   i slots", then the breadth-first decode. Before it, the CWBVH path runs
 *   `BVH::CombineLeafs(3)` and `BVH::SplitLeafs(3)` (`combineLeaves`,
 *   `splitLeaves`), as `BVH8_CWBVH::Build` does.
 * - `packBvh4` is `BVH4_GPU::ConvertFrom` (64-byte node: the parent's box,
 *   8-bit child planes, four child words).
 * - `packCwbvh` is `BVH8_CWBVH::ConvertFrom` (80-byte node: the parent's
 *   corner, per-axis power-of-two exponents, the internal-node mask, child
 *   and triangle base indices, eight meta bytes, 8-bit child planes), with
 *   its greedy octant slot assignment.
 *
 * What differs from tinybvh, and why (the design note lists each):
 *
 * - **Leaves point into the scene's triangle buffer.** tinybvh stores the
 *   triangles by value next to the nodes (BVH4_GPU) or in its own array
 *   (CWBVH). Here a leaf names a range of the mesh's triangles, so records,
 *   indexed leaves, the filter flag, merged identity and materials are what
 *   they are in the binary scene. CWBVH needs a node's leaf triangles one
 *   after another in slot order, so the converted tree carries its
 *   triangles reordered (`triangleOrder`); BVH4 keeps the binary order.
 * - **BVH4 planes use CWBVH's power-of-two scale** (tinybvh's BVH4_GPU
 *   scales by extent / 255). Then every decoded plane `p + q * 2^e` is one
 *   rounding of an exact value, so the encoder can check, as the GPU will
 *   compute it, that each plane lies on or outside the exact child box.
 * - **Outward rounding is checked**, not assumed: a plane whose decoded `f32`
 *   value lands inside the exact box moves one step out, and an exponent too
 *   small for the parent's extent goes one up. tinybvh rounds with
 *   floor/ceil on `f32` differences, which can land a plane one rounding
 *   inside. A box that contains the exact one costs visits, never hits; that
 *   is what lets the wide walks give the binary walk's hits.
 * - **Costs in `f64`**: the dynamic program runs in doubles (tinybvh: `float`),
 *   so a tie may be broken differently; it changes the shape, never a hit.
 * - **The binary input is this package's 64-byte node** (two child boxes per
 *   node, leaves as child references), read into tinybvh's one-box-per-node
 *   form first.
 */

import {
    GPU_LAYOUT_EMPTY_CHILD,
    GPU_LAYOUT_NODE_WORD,
    GPU_LAYOUT_NODE_WORDS,
    GPU_LAYOUT_CHILD_STRIDE_WORDS,
    type GpuLayoutMeshBvh,
} from './gpu-layout.ts';

/** The wide formats. The top level stays a binary `f32` tree. */
export type WideNodeFormat = 'bvh4' | 'cwbvh8';

/** One BVH4 node: the parent's box corner and plane scale, 8-bit child planes, four child words; 64 bytes. */
export const BVH4_NODE_BYTES = 64;
export const BVH4_NODE_WORDS = BVH4_NODE_BYTES / 4;
/** One CWBVH node: 80 bytes, five `vec4u`. */
export const CWBVH_NODE_BYTES = 80;
export const CWBVH_NODE_WORDS = CWBVH_NODE_BYTES / 4;

/**
 * Word positions inside a BVH4 node (tinybvh's `BVH4_GPU::BVHNode`):
 *
 *     word 0-2  the node box's min corner, f32        word 3   child 0-3 x-min planes, one byte each
 *     word 4-6  the plane scale 2^e per axis, f32     word 7   child 0-3 x-max planes
 *     word 8    child 0-3 y-min planes                word 9   y-max
 *     word 10   z-min                                 word 11  z-max
 *     word 12-15 child 0-3 (see BVH4_CHILD)
 *
 * A plane decodes as `min + f32(q) * scale` (exact product, one rounding).
 */
export const BVH4_NODE_WORD = {
    min: 0,
    xMin: 3,
    scale: 4,
    xMax: 7,
    yMin: 8,
    yMax: 9,
    zMin: 10,
    zMax: 11,
    child: 12,
} as const;

/**
 * A BVH4 child word: 0 for no child; bit 31 set for a leaf of
 * `((word >> 28) & 7) + 1` triangles from mesh triangle `word & 0x0fffffff`;
 * else an inner node at mesh node `word` (never 0: that is the root).
 */
export const BVH4_CHILD = {
    none: 0,
    leafBit: 0x80000000,
    countShift: 28,
    countMask: 7,
    indexMask: 0x0fffffff,
} as const;

/**
 * Word positions inside a CWBVH node (Ylitie et al. 2017, figure 2; tinybvh's
 * `BVH8_CWBVH` layout):
 *
 *     word 0-2  p, the node box's min corner, f32
 *     word 3    exponent bytes ex + 127, ey + 127, ez + 127, then imask (bit per slot: inner child)
 *     word 4    child base index (mesh node of the first inner child; the rest follow, in slot order)
 *     word 5    triangle base index (mesh triangle of the first leaf child's first triangle)
 *     word 6-7  meta, one byte per slot: 0 empty; inner (1 << 5) | (24 + slot);
 *               leaf (unary triangle count 0b1, 0b11, 0b111) << 5 | triangle offset from the base
 *     word 8-9   x-min planes, slots 0-7     word 10-11 y-min     word 12-13 z-min
 *     word 14-15 x-max                       word 16-17 y-max     word 18-19 z-max
 *
 * A plane decodes as `p + f32(q) * 2^e`.
 */
export const CWBVH_NODE_WORD = {
    p: 0,
    exponentsAndImask: 3,
    childBase: 4,
    triangleBase: 5,
    meta: 6,
    xMin: 8,
    yMin: 10,
    zMin: 12,
    xMax: 14,
    yMax: 16,
    zMax: 18,
} as const;

/** Most triangles a CWBVH leaf slot holds (a 3-bit unary count). */
export const CWBVH_MAX_LEAF_TRIANGLES = 3;

/** A mesh tree converted to a wide format: what a scene places for it. */
export interface WideMeshTree {
    readonly nodeFormat: WideNodeFormat;
    /** `nodeCount` wide nodes (64 or 80 bytes); node 0 is the root. */
    readonly nodes: ArrayBuffer;
    readonly nodeCount: number;
    /**
     * Where the converted tree's triangle `k` was in the binary tree's
     * triangles; null when the order is unchanged (BVH4).
     */
    readonly triangleOrder: Uint32Array | null;
    /** Stack entries the mesh walk needs for this tree (`maxStackNeed`). */
    readonly stackNeed: number;
    /** Levels of wide nodes on the longest path. */
    readonly depth: number;
    /** Inner child slots used per node (for the design note's fullness numbers). */
    readonly childrenPerNode: number;
    /** Triangles per leaf slot. */
    readonly trianglesPerLeaf: number;
    /** What {@link refitWide} needs to repack the same nodes from a refit binary tree. */
    readonly topology: WideTopology;
}

/**
 * The collapse's decisions, kept so a refit can repack the same wide nodes:
 * each wide node's binary node (`src`), and each packed node's wide node and
 * slot assignment. Opaque outside this file.
 */
export interface WideTopology {
    readonly nodes: readonly WideNode[];
    /** Packed node -> its wide node. */
    readonly addressNode: Int32Array;
    /** CWBVH: packed node -> slot -> wide child (-1: empty), 8 per node. BVH4: child k is slot k. */
    readonly slots: Int32Array | null;
}

/** Bytes per node of a wide format. */
export function wideNodeBytes(format: WideNodeFormat): number {
    return format === 'cwbvh8' ? CWBVH_NODE_BYTES : BVH4_NODE_BYTES;
}

export function isWideNodeFormat(format: unknown): format is WideNodeFormat {
    return format === 'bvh4' || format === 'cwbvh8';
}

// ---- tinybvh's binary form -----------------------------------------------------

/**
 * The binary tree with one box per node, as tinybvh's `BVH` holds it: node
 * `n` is a leaf when `count[n] > 0` (triangles `first[n]`..), else an inner
 * node with children `left[n]` and `right[n]`. Nodes are numbered parents
 * before children (pre-order), which the sweeps below rely on.
 */
interface BinaryTree {
    size: number;
    box: Float64Array; // 6 per node
    left: Int32Array;
    right: Int32Array;
    first: Uint32Array;
    count: Uint32Array;
}

function allocateBinary(capacity: number): BinaryTree {
    return {
        size: 0,
        box: new Float64Array(capacity * 6),
        left: new Int32Array(capacity).fill(-1),
        right: new Int32Array(capacity).fill(-1),
        first: new Uint32Array(capacity),
        count: new Uint32Array(capacity),
    };
}

/** Reads a 64-byte-node tree into the one-box-per-node form, in pre-order. */
function readBinary(layout: GpuLayoutMeshBvh, extraCapacity: number): BinaryTree {
    if (layout.nodeFormat !== undefined && layout.nodeFormat !== 'f32') {
        throw new RangeError(`wide nodes: the tree must have f32 nodes, it has ${layout.nodeFormat}`);
    }
    const floats = new Float32Array(layout.nodes, 0, layout.nodeCount * GPU_LAYOUT_NODE_WORDS);
    const words = new Uint32Array(layout.nodes, 0, layout.nodeCount * GPU_LAYOUT_NODE_WORDS);
    // Every node and every leaf child becomes a node: at most 2 * nodeCount + 1.
    const tree = allocateBinary(2 * layout.nodeCount + 1 + extraCapacity);
    const W = GPU_LAYOUT_NODE_WORD;
    const childBox = (node: number, side: number, out: number): void => {
        const at = node * GPU_LAYOUT_NODE_WORDS + side * GPU_LAYOUT_CHILD_STRIDE_WORDS;
        for (let axis = 0; axis < 3; axis++) {
            tree.box[out * 6 + axis] = floats[at + W.leftMin + axis];
            tree.box[out * 6 + 3 + axis] = floats[at + W.leftMax + axis];
        }
    };
    const rootRight = words[W.rightIndex];
    // A single-triangle tree: its root holds one leaf. tinybvh's root is that leaf.
    if (rootRight === GPU_LAYOUT_EMPTY_CHILD) {
        childBox(0, 0, 0);
        tree.first[0] = words[W.leftIndex];
        tree.count[0] = words[W.leftCount];
        tree.size = 1;
        return tree;
    }
    // The root's box: the union of its children (tinybvh keeps a box per node).
    for (let axis = 0; axis < 3; axis++) {
        tree.box[axis] = Math.min(floats[W.leftMin + axis], floats[W.rightMin + axis]);
        tree.box[3 + axis] = Math.max(floats[W.leftMax + axis], floats[W.rightMax + axis]);
    }
    tree.size = 1;
    // Pre-order: a stack of (our node, its binary id).
    const stack: number[] = [0, 0];
    while (stack.length > 0) {
        const id = stack.pop()!;
        const node = stack.pop()!;
        const base = node * GPU_LAYOUT_NODE_WORDS;
        const ids = [0, 0];
        for (let side = 0; side < 2; side++) {
            const child = tree.size++;
            ids[side] = child;
            childBox(node, side, child);
            const index = words[base + (side === 0 ? W.leftIndex : W.rightIndex)];
            const count = words[base + (side === 0 ? W.leftCount : W.rightCount)];
            if (count > 0) {
                tree.first[child] = index;
                tree.count[child] = count;
            }
        }
        tree.left[id] = ids[0];
        tree.right[id] = ids[1];
        // Push right then left so the left subtree gets the lower numbers.
        for (let side = 1; side >= 0; side--) {
            const count = words[base + (side === 0 ? W.leftCount : W.rightCount)];
            if (count === 0) stack.push(words[base + (side === 0 ? W.leftIndex : W.rightIndex)], ids[side]);
        }
    }
    return tree;
}

function isLeaf(tree: BinaryTree, n: number): boolean {
    return tree.count[n] > 0;
}

/** `BVH::CombineLeafs(primCount)`: an inner node whose subtree holds at most `primCount` triangles becomes one leaf. */
function combineLeaves(tree: BinaryTree, primCount: number): void {
    // Post-order: children (higher numbers) before parents.
    const total = new Uint32Array(tree.size);
    const lo = new Uint32Array(tree.size);
    const hi = new Uint32Array(tree.size);
    for (let n = tree.size - 1; n >= 0; n--) {
        if (isLeaf(tree, n)) {
            total[n] = tree.count[n];
            lo[n] = tree.first[n];
            hi[n] = tree.first[n] + tree.count[n];
            continue;
        }
        if (tree.left[n] < 0) continue; // a node a combine above made unreachable
        const l = tree.left[n];
        const r = tree.right[n];
        total[n] = total[l] + total[r];
        lo[n] = Math.min(lo[l], lo[r]);
        hi[n] = Math.max(hi[l], hi[r]);
        // tinybvh assumes a subtree's triangles are contiguous (its builders make them so); checked here.
        if (total[n] <= primCount && hi[n] - lo[n] === total[n]) {
            tree.first[n] = lo[n];
            tree.count[n] = total[n];
        }
    }
}

/**
 * `BVH::SplitLeafs(maxPrims)`: a leaf of more than `maxPrims` triangles
 * becomes an inner node whose left child takes the first `maxPrims` and
 * whose right child the rest (split again until it fits). Both children keep
 * the leaf's box, as tinybvh does. New nodes are appended, after their parent.
 */
function splitLeaves(tree: BinaryTree, maxPrims: number): BinaryTree {
    let extra = 0;
    for (let n = 0; n < tree.size; n++) if (isLeaf(tree, n) && tree.count[n] > maxPrims) extra += 2 * Math.ceil((tree.count[n] - maxPrims) / maxPrims);
    if (extra === 0) return tree;
    const grown = allocateBinary(tree.size + extra);
    grown.box.set(tree.box.subarray(0, tree.size * 6));
    grown.left.set(tree.left.subarray(0, tree.size));
    grown.right.set(tree.right.subarray(0, tree.size));
    grown.first.set(tree.first.subarray(0, tree.size));
    grown.count.set(tree.count.subarray(0, tree.size));
    grown.size = tree.size;
    for (let n = 0; n < grown.size; n++) {
        if (!isLeaf(grown, n) || grown.count[n] <= maxPrims) continue;
        const left = grown.size;
        const right = grown.size + 1;
        grown.size += 2;
        grown.box.copyWithin(left * 6, n * 6, n * 6 + 6);
        grown.box.copyWithin(right * 6, n * 6, n * 6 + 6);
        grown.first[left] = grown.first[n];
        grown.count[left] = maxPrims;
        grown.first[right] = grown.first[n] + maxPrims;
        grown.count[right] = grown.count[n] - maxPrims;
        grown.count[n] = 0;
        grown.left[n] = left;
        grown.right[n] = right;
    }
    return grown;
}

/** Half the surface area, as tinybvh's `SA`. */
function surfaceArea(box: Float64Array, n: number): number {
    const x = box[n * 6 + 3] - box[n * 6];
    const y = box[n * 6 + 4] - box[n * 6 + 1];
    const z = box[n * 6 + 5] - box[n * 6 + 2];
    return x * y + y * z + z * x;
}

// ---- The collapse: MBVH<M>::ConvertFrom ---------------------------------------------

/** One node of the collapsed M-wide tree (tinybvh's `MBVHNode`). */
interface WideNode {
    /** The binary node it stands for (its box comes from there). */
    readonly src: number;
    /** Box: min x, y, z, max x, y, z. */
    box: Float64Array;
    /** Leaves: the triangle range. */
    readonly firstTri: number;
    readonly triCount: number;
    /** Inner nodes: the children's positions in the node list. */
    readonly children: number[];
}

const C_TRAV = 1;
const C_INT = 1;
/** tinybvh: `c_leaf = C_INT * 0.8f`. */
const C_LEAF = C_INT * 0.8;
const USED = 1;
const CONTIG = 2;
const ASLEAF = 4;

/**
 * Collapses the binary tree into an M-wide one (`MBVH<M>::ConvertFrom`),
 * merging subtrees of at most `leafPrimLimit` contiguous triangles into one
 * leaf (0: never). Returns the nodes breadth-first: a node's children are
 * consecutive entries.
 */
function collapse(tree: BinaryTree, M: number, leafPrimLimit: number): WideNode[] {
    const boxOf = (n: number): Float64Array => tree.box.slice(n * 6, n * 6 + 6);
    // The root is a leaf: add a level, as tinybvh does (CWBVH needs an inner root).
    if (isLeaf(tree, 0)) {
        return [
            { src: 0, box: boxOf(0), firstTri: 0, triCount: 0, children: [1] },
            { src: 0, box: boxOf(0), firstTri: tree.first[0], triCount: tree.count[0], children: [] },
        ];
    }
    const N = tree.size;
    const cost = new Float64Array(N * M);
    const dist = new Uint8Array(N * M);
    const subFirst = new Uint32Array(N);
    const subCount = new Uint32Array(N);
    const flag = new Uint8Array(N);
    // Pass 1: mark the nodes in the tree (children follow their parents).
    flag[0] = USED;
    for (let n = 0; n < N; n++) {
        if (!(flag[n] & USED) || isLeaf(tree, n)) continue;
        flag[tree.left[n]] |= USED;
        flag[tree.right[n]] |= USED;
    }
    // Pass 2: the dynamic program, bottom-up.
    for (let n = N - 1; n >= 0; n--) {
        if (!(flag[n] & USED)) continue;
        const base = n * M;
        const sa = surfaceArea(tree.box, n);
        if (isLeaf(tree, n)) {
            // A leaf takes a single slot, however many it is offered.
            subFirst[n] = tree.first[n];
            subCount[n] = tree.count[n];
            flag[n] |= CONTIG | ASLEAF;
            const c = sa * (C_LEAF + C_INT * tree.count[n]);
            for (let i = 0; i < M; i++) cost[base + i] = c;
            continue;
        }
        const li = tree.left[n];
        const ri = tree.right[n];
        const L = li * M;
        const R = ri * M;
        subCount[n] = subCount[li] + subCount[ri];
        subFirst[n] = Math.min(subFirst[li], subFirst[ri]);
        // A subtree can only become one leaf if it owns a contiguous index range.
        if (flag[li] & flag[ri] & CONTIG && subFirst[n] + subCount[n] === Math.max(subFirst[li] + subCount[li], subFirst[ri] + subCount[ri])) flag[n] |= CONTIG;
        // Cost of spreading this subtree over i slots: the best split over the children.
        for (let i = 2; i <= M; i++) {
            let best = Infinity;
            let bestK = 1;
            for (let k = 1; k < i; k++) {
                const c = cost[L + k - 1] + cost[R + (i - k) - 1];
                if (c < best) {
                    best = c;
                    bestK = k;
                }
            }
            cost[base + i - 1] = best;
            dist[base + i - 1] = bestK;
        }
        // Cost in a single slot: a new wide node, or the whole subtree as one leaf.
        const cNode = sa * C_TRAV + cost[base + M - 1];
        let cLeaf = Infinity;
        if (flag[n] & CONTIG && subCount[n] <= leafPrimLimit) cLeaf = sa * (C_LEAF + C_INT * subCount[n]);
        if (cLeaf < cNode) {
            cost[base] = cLeaf;
            flag[n] |= ASLEAF;
        } else cost[base] = cNode;
        // A subtree may always use fewer slots than offered; dist 0 means a single slot.
        for (let i = 2; i <= M; i++) {
            if (cost[base] < cost[base + i - 1]) {
                cost[base + i - 1] = cost[base];
                dist[base + i - 1] = 0;
            }
        }
    }
    // Pass 3: decode breadth-first, so a node's children are consecutive.
    const decodeWalk = (src: number): number[] => {
        const task: number[] = [src];
        const taskSlots: number[] = [M];
        const term: number[] = [];
        while (task.length > 0) {
            const c = task.pop()!;
            const slots = taskSlots.pop()!;
            if (slots > 1 && !isLeaf(tree, c) && dist[c * M + slots - 1] !== 0) {
                const k = dist[c * M + slots - 1];
                // Right first, so that left pops first.
                task.push(tree.right[c]);
                taskSlots.push(slots - k);
                task.push(tree.left[c]);
                taskSlots.push(k);
                continue;
            }
            term.push(c);
        }
        return term;
    };
    const out: WideNode[] = [];
    const srcNode: number[] = [0];
    out.push({ src: 0, box: boxOf(0), firstTri: 0, triCount: 0, children: [] });
    for (let i = 0; i < out.length; i++) {
        const src = srcNode[i];
        if (src < 0) continue; // a leaf
        const term = decodeWalk(src);
        for (const c of term) {
            out[i].children.push(out.length);
            if (flag[c] & ASLEAF) {
                out.push({ src: c, box: boxOf(c), firstTri: subFirst[c], triCount: subCount[c], children: [] });
                srcNode.push(-1);
            } else {
                out.push({ src: c, box: boxOf(c), firstTri: 0, triCount: 0, children: [] });
                srcNode.push(c);
            }
        }
    }
    return out;
}

// ---- Quantization ------------------------------------------------------------------

/**
 * The plane exponent for an axis of a node: the smallest `e` (in [-126, 126])
 * with `lo + 255 * 2^e` decoding, in `f32`, at or above `hi`. tinybvh:
 * `ceilf(log2f(extent / 255))`, which a rounded `f32` extent can put one too low.
 */
function planeExponent(lo: number, hi: number): number {
    const extent = hi - lo;
    if (!(extent > 0)) return -126;
    let e = Math.max(-126, Math.min(126, Math.ceil(Math.log2(extent / 255))));
    while (e > -126 && Math.fround(lo + 255 * 2 ** (e - 1)) >= hi) e--;
    while (e < 126 && Math.fround(lo + 255 * 2 ** e) < hi) e++;
    return e;
}

/** A min plane rounded down: the largest q with `f32(p + q * 2^e) <= value`. */
function quantizeMin(value: number, p: number, scale: number): number {
    let q = Math.max(0, Math.min(255, Math.floor((value - p) / scale)));
    while (q > 0 && Math.fround(p + q * scale) > value) q--;
    return q;
}

/** A max plane rounded up: the smallest q with `f32(p + q * 2^e) >= value`. */
function quantizeMax(value: number, p: number, scale: number): number {
    let q = Math.max(0, Math.min(255, Math.ceil((value - p) / scale)));
    while (q < 255 && Math.fround(p + q * scale) < value) q++;
    if (Math.fround(p + q * scale) < value) throw new RangeError(`wide nodes: a child box does not fit its parent (${value} > ${Math.fround(p + q * scale)})`);
    return q;
}

/** A box no ray can reach: inverted, or not finite. */
function isEmptyBox(box: Float64Array): boolean {
    for (let axis = 0; axis < 3; axis++) {
        const lo = box[axis];
        const hi = box[axis + 3];
        if (!(lo <= hi) || !Number.isFinite(lo) || !Number.isFinite(hi)) return true;
    }
    return false;
}

/** The node's own box for quantization: its `f32` box (the union of its children's, as the builders write). */
function nodeCorner(nodes: readonly WideNode[], node: WideNode): { lo: number[]; hi: number[] } {
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (const c of node.children) {
        const box = nodes[c].box;
        if (isEmptyBox(box)) continue;
        for (let axis = 0; axis < 3; axis++) {
            lo[axis] = Math.min(lo[axis], box[axis]);
            hi[axis] = Math.max(hi[axis], box[axis + 3]);
        }
    }
    for (let axis = 0; axis < 3; axis++) {
        if (!(lo[axis] <= hi[axis])) {
            lo[axis] = 0;
            hi[axis] = 0;
        }
        // The corner is stored as f32; the box values are f32 already.
        lo[axis] = Math.fround(lo[axis]);
        hi[axis] = Math.fround(hi[axis]);
    }
    return { lo, hi };
}

/** Quantized planes of one child: [loX, loY, loZ, hiX, hiY, hiZ]; an empty box gets lo 255, hi 0 (never entered). */
function childPlanes(box: Float64Array, lo: number[], scale: number[]): number[] {
    if (isEmptyBox(box)) return [255, 255, 255, 0, 0, 0];
    const planes = [0, 0, 0, 0, 0, 0];
    for (let axis = 0; axis < 3; axis++) {
        planes[axis] = quantizeMin(box[axis], lo[axis], scale[axis]);
        planes[axis + 3] = quantizeMax(box[axis + 3], lo[axis], scale[axis]);
    }
    return planes;
}

// ---- BVH4: BVH4_GPU::ConvertFrom --------------------------------------------------

/** Writes a BVH4 node's corner, scale and planes (not its child words) at `address`. */
function writeBvh4Boxes(nodes: readonly WideNode[], node: WideNode, words: Uint32Array, floats: Float32Array, address: number): void {
    const W = BVH4_NODE_WORD;
    const base = address * BVH4_NODE_WORDS;
    const { lo, hi } = nodeCorner(nodes, node);
    const scale = [0, 1, 2].map((axis) => 2 ** planeExponent(lo[axis], hi[axis]));
    for (let axis = 0; axis < 3; axis++) {
        floats[base + W.min + axis] = lo[axis];
        floats[base + W.scale + axis] = scale[axis];
    }
    const planeWords = [0, 0, 0, 0, 0, 0];
    node.children.forEach((c, slot) => {
        const planes = childPlanes(nodes[c].box, lo, scale);
        for (let k = 0; k < 6; k++) planeWords[k] |= planes[k] << (8 * slot);
    });
    words[base + W.xMin] = planeWords[0] >>> 0;
    words[base + W.yMin] = planeWords[1] >>> 0;
    words[base + W.zMin] = planeWords[2] >>> 0;
    words[base + W.xMax] = planeWords[3] >>> 0;
    words[base + W.yMax] = planeWords[4] >>> 0;
    words[base + W.zMax] = planeWords[5] >>> 0;
}

function packBvh4(nodes: WideNode[]): { buffer: ArrayBuffer; count: number; stackNeed: number; depth: number; addressNode: Int32Array } {
    let inner = 0;
    for (const node of nodes) if (node.children.length > 0) inner++;
    const buffer = new ArrayBuffer(Math.max(1, inner) * BVH4_NODE_BYTES);
    const words = new Uint32Array(buffer);
    const floats = new Float32Array(buffer);
    const addressNode = new Int32Array(Math.max(1, inner));
    const W = BVH4_NODE_WORD;
    // Depth-first with a stack, as tinybvh: a node gets its address when it is converted.
    let next = 0;
    let stackNeed = 0;
    let depth = 0;
    const stack: { node: number; parentWord: number; need: number; level: number }[] = [{ node: 0, parentWord: -1, need: 0, level: 1 }];
    while (stack.length > 0) {
        const { node: index, parentWord, need, level } = stack.pop()!;
        const node = nodes[index];
        const address = next++;
        addressNode[address] = index;
        if (parentWord >= 0) words[parentWord] = address;
        const base = address * BVH4_NODE_WORDS;
        if (node.children.length > 4) throw new Error('wide nodes: a BVH4 node with more than 4 children');
        writeBvh4Boxes(nodes, node, words, floats, address);
        let innerChildren = 0;
        node.children.forEach((c, slot) => {
            const child = nodes[c];
            if (child.children.length === 0) {
                if (child.triCount < 1 || child.triCount > 8) throw new RangeError(`wide nodes: a BVH4 leaf of ${child.triCount} triangles`);
                if (child.firstTri > BVH4_CHILD.indexMask) throw new RangeError('wide nodes: too many triangles for a BVH4 leaf word');
                words[base + W.child + slot] = (BVH4_CHILD.leafBit | ((child.triCount - 1) << BVH4_CHILD.countShift) | child.firstTri) >>> 0;
            } else innerChildren++;
        });
        // The walk pushes every inner child it enters and pops the nearest; one entry stays per pushed sibling.
        const here = need + innerChildren;
        stackNeed = Math.max(stackNeed, here);
        depth = Math.max(depth, level);
        node.children.forEach((c, slot) => {
            if (nodes[c].children.length > 0) stack.push({ node: c, parentWord: base + W.child + slot, need: here - 1, level: level + 1 });
        });
    }
    return { buffer, count: next, stackNeed, depth, addressNode };
}

// ---- CWBVH: BVH8_CWBVH::ConvertFrom ------------------------------------------------

/** tinybvh's greedy slot assignment: child i goes to slot s, by its centroid's direction from the node's. */
function octantSlots(nodes: WideNode[], node: WideNode): number[] {
    const box = node.box;
    const centroid = [0, 1, 2].map((axis) => (box[axis] + box[axis + 3]) * 0.5);
    const cost: number[][] = Array.from({ length: 8 }, () => new Array<number>(8).fill(Infinity));
    const assignment = new Array<number>(8).fill(-1);
    const slotEmpty = new Array<boolean>(8).fill(true);
    const childCount = node.children.length;
    for (let i = 0; i < childCount; i++) {
        const c = nodes[node.children[i]].box;
        const d = [0, 1, 2].map((axis) => (c[axis] + c[axis + 3]) * 0.5 - centroid[axis]);
        for (let s = 0; s < 8; s++) cost[s][i] = (s & 4 ? -d[0] : d[0]) + (s & 2 ? -d[1] : d[1]) + (s & 1 ? -d[2] : d[2]);
    }
    for (;;) {
        let minCost = Infinity;
        let minSlot = -1;
        let minChild = -1;
        for (let s = 0; s < 8; s++) {
            if (!slotEmpty[s]) continue;
            for (let i = 0; i < 8; i++) {
                if (assignment[i] === -1 && cost[s][i] < minCost) {
                    minCost = cost[s][i];
                    minSlot = s;
                    minChild = i;
                }
            }
        }
        if (minSlot === -1) break;
        slotEmpty[minSlot] = false;
        assignment[minChild] = minSlot;
    }
    // Children left over (none with finite costs) take the first free slots.
    for (let i = 0; i < childCount; i++) {
        if (assignment[i] !== -1) continue;
        for (let s = 0; s < 8; s++) {
            if (slotEmpty[s]) {
                slotEmpty[s] = false;
                assignment[i] = s;
                break;
            }
        }
    }
    // slot -> child position in the node list, -1 for an empty slot.
    const slotChild = new Array<number>(8).fill(-1);
    for (let i = 0; i < childCount; i++) slotChild[assignment[i]] = node.children[i];
    return slotChild;
}

/** Writes a CWBVH node's corner, exponents and planes at `address` (keeping its imask byte and the rest). */
function writeCwbvhBoxes(nodes: readonly WideNode[], node: WideNode, slotChild: ArrayLike<number>, words: Uint32Array, floats: Float32Array, bytes: Uint8Array, address: number): void {
    const W = CWBVH_NODE_WORD;
    const base = address * CWBVH_NODE_WORDS;
    const byteBase = address * CWBVH_NODE_BYTES;
    const { lo, hi } = nodeCorner(nodes, node);
    const exponents = [0, 1, 2].map((axis) => planeExponent(lo[axis], hi[axis]));
    const scale = exponents.map((e) => 2 ** e);
    const planeWords = [W.xMin, W.yMin, W.zMin, W.xMax, W.yMax, W.zMax];
    for (let slot = 0; slot < 8; slot++) {
        const c = slotChild[slot];
        if (c < 0) continue;
        // Plane bytes: lo x at word 8, lo y 10, lo z 12, hi x 14, hi y 16, hi z 18; slot = byte within the pair.
        const planes = childPlanes(nodes[c].box, lo, scale);
        for (let k = 0; k < 6; k++) bytes[byteBase + planeWords[k] * 4 + slot] = planes[k];
    }
    for (let axis = 0; axis < 3; axis++) floats[base + W.p + axis] = lo[axis];
    const imask = words[base + W.exponentsAndImask] >>> 24;
    words[base + W.exponentsAndImask] = ((exponents[0] + 127) | ((exponents[1] + 127) << 8) | ((exponents[2] + 127) << 16) | (imask << 24)) >>> 0;
}

function packCwbvh(nodes: WideNode[], triangleCount: number): {
    buffer: ArrayBuffer;
    count: number;
    triangleOrder: Uint32Array;
    depth: number;
    slotsUsed: number;
    leafSlots: number;
    addressNode: Int32Array;
    slots: Int32Array;
} {
    let inner = 0;
    for (const node of nodes) if (node.children.length > 0) inner++;
    const buffer = new ArrayBuffer(Math.max(1, inner) * CWBVH_NODE_BYTES);
    const words = new Uint32Array(buffer);
    const floats = new Float32Array(buffer);
    const bytes = new Uint8Array(buffer);
    const triangleOrder = new Uint32Array(triangleCount);
    const addressNode = new Int32Array(Math.max(1, inner));
    const slots = new Int32Array(Math.max(1, inner) * 8).fill(-1);
    const W = CWBVH_NODE_WORD;
    let nodePtr = 1; // the root is node 0
    let triPtr = 0;
    let depth = 0;
    let slotsUsed = 0;
    let leafSlots = 0;
    const stack: { node: number; address: number; level: number }[] = [{ node: 0, address: 0, level: 1 }];
    while (stack.length > 0) {
        const { node: index, address, level } = stack.pop()!;
        depth = Math.max(depth, level);
        const node = nodes[index];
        const base = address * CWBVH_NODE_WORDS;
        const byteBase = address * CWBVH_NODE_BYTES;
        const slotChild = octantSlots(nodes, node);
        addressNode[address] = index;
        slots.set(slotChild, address * 8);
        let imask = 0;
        let internalChildCount = 0;
        let leafTriangles = 0;
        let childBase = 0;
        let triangleBase = 0;
        for (let slot = 0; slot < 8; slot++) {
            const c = slotChild[slot];
            if (c < 0) continue;
            slotsUsed++;
            const child = nodes[c];
            if (child.children.length > 0) {
                const childAddress = nodePtr++;
                if (internalChildCount++ === 0) childBase = childAddress;
                imask |= 1 << slot;
                bytes[byteBase + W.meta * 4 + slot] = (1 << 5) | (24 + slot);
                stack.push({ node: c, address: childAddress, level: level + 1 });
                continue;
            }
            const count = child.triCount;
            if (count < 1 || count > CWBVH_MAX_LEAF_TRIANGLES) throw new RangeError(`wide nodes: a CWBVH leaf of ${count} triangles`);
            if (leafTriangles === 0) triangleBase = triPtr;
            const unary = count === 1 ? 0b001 : count === 2 ? 0b011 : 0b111;
            bytes[byteBase + W.meta * 4 + slot] = (unary << 5) | leafTriangles;
            leafTriangles += count;
            leafSlots++;
            for (let j = 0; j < count; j++) triangleOrder[triPtr++] = child.firstTri + j;
        }
        words[base + W.exponentsAndImask] = (imask << 24) >>> 0;
        writeCwbvhBoxes(nodes, node, slotChild, words, floats, bytes, address);
        words[base + W.childBase] = childBase;
        words[base + W.triangleBase] = triangleBase;
    }
    if (triPtr !== triangleCount) throw new Error(`wide nodes: the CWBVH holds ${triPtr} of ${triangleCount} triangles`);
    return { buffer, count: nodePtr, triangleOrder, depth, slotsUsed, leafSlots, addressNode, slots };
}

/** Mesh walk stack entries for a CWBVH tree of this depth: one node group per level (Ylitie: the stack holds groups). */
export function cwbvhStackNeed(depth: number): number {
    return depth;
}

/**
 * Converts a binary `f32` tree to a wide one. The binary tree is left as it
 * is. BVH4 keeps the triangle order; CWBVH returns the order its leaves need
 * (`triangleOrder`), which {@link reorderTriangles} applies.
 */
export function convertToWide(layout: GpuLayoutMeshBvh, format: WideNodeFormat): WideMeshTree {
    if (format === 'bvh4') {
        const binary = readBinary(layout, 0);
        const nodes = collapse(binary, 4, 0);
        const packed = packBvh4(nodes);
        let slots = 0;
        let leaves = 0;
        let leafTriangles = 0;
        for (const node of nodes) {
            slots += node.children.length;
            for (const c of node.children) if (nodes[c].children.length === 0) {
                leaves++;
                leafTriangles += nodes[c].triCount;
            }
        }
        return {
            nodeFormat: 'bvh4',
            nodes: packed.buffer,
            nodeCount: packed.count,
            triangleOrder: null,
            stackNeed: packed.stackNeed,
            depth: packed.depth,
            childrenPerNode: slots / packed.count,
            trianglesPerLeaf: leaves > 0 ? leafTriangles / leaves : 0,
            topology: { nodes, addressNode: packed.addressNode, slots: null },
        };
    }
    if (format !== 'cwbvh8') throw new RangeError(`wide nodes: unknown format ${String(format)}`);
    const binary = cwbvhBinary(layout);
    const nodes = collapse(binary, 8, CWBVH_MAX_LEAF_TRIANGLES);
    const packed = packCwbvh(nodes, layout.triangleCount);
    return {
        nodeFormat: 'cwbvh8',
        nodes: packed.buffer,
        nodeCount: packed.count,
        triangleOrder: packed.triangleOrder,
        stackNeed: cwbvhStackNeed(packed.depth),
        depth: packed.depth,
        childrenPerNode: packed.slotsUsed / packed.count,
        trianglesPerLeaf: packed.leafSlots > 0 ? layout.triangleCount / packed.leafSlots : 0,
        topology: { nodes, addressNode: packed.addressNode, slots: packed.slots },
    };
}

/** The binary tree CWBVH collapses: `CombineLeafs(3)`, then `SplitLeafs(3)`, as `BVH8_CWBVH::Build`. */
function cwbvhBinary(layout: GpuLayoutMeshBvh): BinaryTree {
    const binary = readBinary(layout, 0);
    combineLeaves(binary, CWBVH_MAX_LEAF_TRIANGLES);
    return splitLeaves(binary, CWBVH_MAX_LEAF_TRIANGLES);
}

/**
 * A wide tree's nodes repacked from a refit of its binary tree (`refitted`:
 * the same topology as the tree it was converted from, new boxes). The wide
 * nodes, their addresses, slots and triangle order stay; each node's corner,
 * exponents and planes are computed again from the refit boxes. So a
 * deforming mesh keeps its node and triangle ranges; the collapse's choices
 * stay those of the pose it was built in (as a binary refit keeps the
 * build's splits).
 */
export function refitWide(tree: WideMeshTree, refitted: GpuLayoutMeshBvh): ArrayBuffer {
    const binary = tree.nodeFormat === 'cwbvh8' ? cwbvhBinary(refitted) : readBinary(refitted, 0);
    const { nodes, addressNode, slots } = tree.topology;
    const fresh: WideNode[] = nodes.map((node) => ({ ...node, box: binary.box.slice(node.src * 6, node.src * 6 + 6) }));
    const buffer = tree.nodes.slice(0);
    const words = new Uint32Array(buffer);
    const floats = new Float32Array(buffer);
    const bytes = new Uint8Array(buffer);
    for (let address = 0; address < tree.nodeCount; address++) {
        const node = fresh[addressNode[address]];
        if (tree.nodeFormat === 'cwbvh8') writeCwbvhBoxes(fresh, node, slots!.subarray(address * 8, address * 8 + 8), words, floats, bytes, address);
        else writeBvh4Boxes(fresh, node, words, floats, address);
    }
    return buffer;
}

/** Triangles (records or indexed, `unitBytes` each) in a converted tree's order. The input is left as it is. */
export function reorderTriangles(triangles: ArrayBuffer, unitBytes: number, order: Uint32Array | null): ArrayBuffer {
    if (order === null) return triangles;
    const out = new ArrayBuffer(order.length * unitBytes);
    const unitWords = unitBytes / 4;
    const from = new Uint32Array(triangles, 0, order.length * unitWords);
    const to = new Uint32Array(out);
    for (let k = 0; k < order.length; k++) to.set(from.subarray(order[k] * unitWords, order[k] * unitWords + unitWords), k * unitWords);
    return out;
}

// ---- Decoding, for the checks and the CPU walk --------------------------------------

/** A decoded child of a wide node: its box as the GPU decodes it, and what it is. */
export interface WideChild {
    readonly slot: number;
    /** min x, y, z, max x, y, z, each an f32 value. */
    readonly box: readonly number[];
    /** Inner child: its mesh node; leaf: -1. */
    readonly node: number;
    /** Leaf: its first mesh triangle and count. */
    readonly firstTriangle: number;
    readonly triangleCount: number;
}

function bytesOf(word: number): number[] {
    return [word & 0xff, (word >>> 8) & 0xff, (word >>> 16) & 0xff, word >>> 24];
}

/** A wide node's children, decoded as the WGSL decodes them (each plane one `f32` rounding of `p + q * 2^e`). */
export function decodeWideNode(format: WideNodeFormat, nodes: ArrayBuffer, node: number): WideChild[] {
    const words = new Uint32Array(nodes);
    const floats = new Float32Array(nodes);
    const out: WideChild[] = [];
    if (format === 'bvh4') {
        const W = BVH4_NODE_WORD;
        const base = node * BVH4_NODE_WORDS;
        const p = [floats[base], floats[base + 1], floats[base + 2]];
        const s = [floats[base + W.scale], floats[base + W.scale + 1], floats[base + W.scale + 2]];
        const planes = [W.xMin, W.yMin, W.zMin, W.xMax, W.yMax, W.zMax].map((w) => bytesOf(words[base + w]));
        for (let slot = 0; slot < 4; slot++) {
            const info = words[base + W.child + slot];
            if (info === BVH4_CHILD.none) continue;
            const box = [0, 1, 2, 3, 4, 5].map((k) => Math.fround(p[k % 3] + planes[k][slot] * s[k % 3]));
            const leaf = (info & BVH4_CHILD.leafBit) !== 0;
            out.push({
                slot,
                box,
                node: leaf ? -1 : info,
                firstTriangle: leaf ? info & BVH4_CHILD.indexMask : 0,
                triangleCount: leaf ? ((info >>> BVH4_CHILD.countShift) & BVH4_CHILD.countMask) + 1 : 0,
            });
        }
        return out;
    }
    const W = CWBVH_NODE_WORD;
    const base = node * CWBVH_NODE_WORDS;
    const p = [floats[base], floats[base + 1], floats[base + 2]];
    const e4 = words[base + W.exponentsAndImask];
    const s = [0, 1, 2].map((axis) => 2 ** (((e4 >>> (8 * axis)) & 0xff) - 127));
    const imask = e4 >>> 24;
    const childBase = words[base + W.childBase];
    const triangleBase = words[base + W.triangleBase];
    const meta = [...bytesOf(words[base + W.meta]), ...bytesOf(words[base + W.meta + 1])];
    const planeAt = (w: number, slot: number): number => bytesOf(words[base + w + (slot >> 2)])[slot & 3];
    const planeWords = [W.xMin, W.yMin, W.zMin, W.xMax, W.yMax, W.zMax];
    for (let slot = 0; slot < 8; slot++) {
        const m = meta[slot];
        if (m === 0) continue;
        const box = [0, 1, 2, 3, 4, 5].map((k) => Math.fround(p[k % 3] + planeAt(planeWords[k], slot) * s[k % 3]));
        if (imask & (1 << slot)) {
            // The relative index: inner children below this slot.
            let below = 0;
            for (let t = 0; t < slot; t++) if (imask & (1 << t)) below++;
            out.push({ slot, box, node: childBase + below, firstTriangle: 0, triangleCount: 0 });
        } else {
            const unary = m >>> 5;
            out.push({ slot, box, node: -1, firstTriangle: triangleBase + (m & 0x1f), triangleCount: unary === 1 ? 1 : unary === 3 ? 2 : 3 });
        }
    }
    return out;
}

/*!
 * The GPU collapse below ports tinybvh's MBVH<M>::ConvertFrom, BVH4_GPU::ConvertFrom and
 * BVH8_CWBVH::ConvertFrom (tiny_bvh_base.h) to WGSL;
 * its licence (licenses/tinybvh.txt):
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2024-2026, Jacco Bikker / Breda University of Applied Sciences.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * WGSL of the GPU wide collapse (gpu-wide-collapse.ts; docs/internals/design/wide-nodes.md,
 * section 2): binary `f32` trees, as H-PLOC writes them, into BVH4 or CWBVH nodes on the GPU,
 * so a wide scene gets first trees without the workers. The same algorithm as the CPU port of
 * tinybvh's collapse (wide-nodes.ts), in three entry points:
 *
 * - `bvhwCost`: the dynamic program (Ylitie et al. 2017, section 4.2), bottom-up. Dispatched
 *   once per tree level (the deepest tree's `maxDepth` times): a node is computed in the first
 *   dispatch after all its inner children are, so nothing waits on another thread.
 * - `bvhwLevel`: one wide level. Each frontier entry is a wide node: it runs the decode walk
 *   over the cost table, packs the node (CWBVH: tinybvh's greedy octant slots; planes rounded
 *   outward and checked in `f32` as the GPU decodes them), takes a consecutive block for its
 *   inner children from its mesh's node counter, copies its leaf triangles into slot order
 *   (CWBVH) from its mesh's triangle counter, and appends its inner children to the next
 *   frontier.
 * - `bvhwArgs`: one thread; turns the next frontier's length into the next level's indirect
 *   dispatch and clears the length after it.
 *
 * Differences from the CPU port, each one a choice the CPU makes in the same place:
 * - CWBVH's leaves of at most 3 triangles come from the builder (`maxLeafPrimitives: 3`), so
 *   there is no `SplitLeafs`. `CombineLeafs(3)` is folded into the cost pass: a contiguous
 *   subtree of at most 3 triangles is costed and decoded as one leaf, as it would be after it.
 * - Costs in `f32`. Node and triangle addresses come from atomic counters, so a node's place
 *   in the buffer depends on scheduling; the tree it describes does not.
 */

import { GPU_LAYOUT_EMPTY_CHILD, GPU_LAYOUT_NODE_WORDS } from './gpu-layout.ts';
import { BVH4_CHILD, BVH4_NODE_WORDS, CWBVH_NODE_WORDS, type WideNodeFormat } from './wide-nodes.ts';

/** Workgroup size of every entry point. */
export const WIDE_COLLAPSE_WORKGROUP = 64;
/** u32 words of scratch per binary node: dist (4 bits per slot count), subFirst, subCount, flags. */
export const WIDE_COLLAPSE_INFO_WORDS = 4;
/** Words of a mesh's entry in the mesh table. */
export const WIDE_COLLAPSE_MESH_WORDS = 8;
/** Per-mesh counters in the control buffer: nodes used, triangles used, deepest stack need, a spare. */
export const WIDE_COLLAPSE_COUNTER_WORDS = 4;
/** Control buffer: the two frontier lengths, then a problem counter, then the per-mesh counters. */
export const WIDE_COLLAPSE_CONTROL_HEADER = 4;
/** Problem flags (control word 2): a leaf too large for the format, a frontier or output overflow. */
export const WIDE_COLLAPSE_PROBLEM = { leafTooLarge: 1, frontierFull: 2, outputFull: 4 } as const;

const hex = (value: number): string => `0x${(value >>> 0).toString(16)}u`;

/** Mesh table entry (`WIDE_COLLAPSE_MESH_WORDS` u32). */
const MESH_STRUCT = /* wgsl */ `
struct BvhwMesh {
    srcNode: u32,        // the mesh's root in the source node buffer (binary nodes)
    nodeCount: u32,      // its binary nodes
    srcTriangle: u32,    // its first triangle in the source triangle buffer (units)
    triangleCount: u32,
    dstNode: u32,        // its first wide node in the output (wide nodes); room for nodeCount
    dstTriangle: u32,    // CWBVH: its first triangle in the output's triangle region (units)
    scratchBase: u32,    // its first node in the scratch arrays (the meshes' node counts summed before it)
    pad1: u32,
}`;

export function wideCollapseWgsl(format: WideNodeFormat, unitWords: number): string {
    const M = format === 'cwbvh8' ? 8 : 4;
    const leafLimit = format === 'cwbvh8' ? 3 : 0;
    const cwbvh = format === 'cwbvh8';
    const wideWords = cwbvh ? CWBVH_NODE_WORDS : BVH4_NODE_WORDS;
    const fStride = M + 6; // costs, then the node's box
    return /* wgsl */ `
// ---- GPU wide collapse (${format}; gpu-wide-collapse-wgsl.ts) ----
${MESH_STRUCT}

struct BvhwParams {
    meshCount: u32,
    totalNodes: u32,        // binary nodes in the job, all meshes
    frontierCapacity: u32,
    parity: u32,            // bvhwLevel: frontier half read (parity) and written (1 - parity)
    outputNodeWords: u32,   // words of the output's node region (the triangle region follows)
    level: u32,
    pad0: u32,
    pad1: u32,
}

const M: u32 = ${M}u;
const LEAF_LIMIT: u32 = ${leafLimit}u;
const NODE_WORDS: u32 = ${GPU_LAYOUT_NODE_WORDS}u;
const WIDE_WORDS: u32 = ${wideWords}u;
const UNIT_WORDS: u32 = ${unitWords}u;
const F_STRIDE: u32 = ${fStride}u;
const EMPTY_CHILD: u32 = ${hex(GPU_LAYOUT_EMPTY_CHILD)};
const LEAF_SLOT: u32 = 0x80000000u;   // a vnode with this bit is a leaf slot: (node << 1) | side
const USED_DONE: u32 = 8u;
const CONTIG: u32 = 2u;
const ASLEAF: u32 = 4u;
const C_LEAF: f32 = 0.8;
const BIG: f32 = 3.0e38;
const CONTROL_HEADER: u32 = ${WIDE_COLLAPSE_CONTROL_HEADER}u;
const COUNTER_WORDS: u32 = ${WIDE_COLLAPSE_COUNTER_WORDS}u;
const PROBLEM_LEAF: u32 = ${WIDE_COLLAPSE_PROBLEM.leafTooLarge}u;
const PROBLEM_FRONTIER: u32 = ${WIDE_COLLAPSE_PROBLEM.frontierFull}u;
const PROBLEM_OUTPUT: u32 = ${WIDE_COLLAPSE_PROBLEM.outputFull}u;

@group(0) @binding(0) var<storage, read> bvhwNodes: array<u32>;
@group(0) @binding(1) var<storage, read> bvhwTriangles: array<u32>;
@group(0) @binding(2) var<storage, read> bvhwMeshes: array<BvhwMesh>;
@group(0) @binding(3) var<storage, read_write> bvhwF: array<f32>;
@group(0) @binding(4) var<storage, read_write> bvhwU: array<u32>;
@group(0) @binding(5) var<storage, read_write> bvhwOut: array<u32>;
@group(0) @binding(6) var<storage, read_write> bvhwControl: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> bvhwFrontier: array<vec4u>;
@group(0) @binding(8) var<uniform> bvhwParams: BvhwParams;
@group(0) @binding(9) var<storage, read_write> bvhwIndirect: array<u32>;

// The mesh holding scratch node 'node' (scratch bases ascend with the mesh index).
fn bvhwMeshOf(node: u32) -> u32 {
    var low = 0u;
    var high = bvhwParams.meshCount;
    while (high - low > 1u) {
        let middle = (low + high) / 2u;
        if (bvhwMeshes[middle].scratchBase <= node) {
            low = middle;
        } else {
            high = middle;
        }
    }
    return low;
}

fn bvhwWord(node: u32, word: u32) -> u32 {
    return bvhwNodes[node * NODE_WORDS + word];
}

// Child 'side' of global node 'node': its box (min, max).
fn bvhwChildMin(node: u32, side: u32) -> vec3f {
    let at = node * NODE_WORDS + side * 8u;
    return vec3f(bitcast<f32>(bvhwNodes[at]), bitcast<f32>(bvhwNodes[at + 1u]), bitcast<f32>(bvhwNodes[at + 2u]));
}
fn bvhwChildMax(node: u32, side: u32) -> vec3f {
    let at = node * NODE_WORDS + side * 8u + 4u;
    return vec3f(bitcast<f32>(bvhwNodes[at]), bitcast<f32>(bvhwNodes[at + 1u]), bitcast<f32>(bvhwNodes[at + 2u]));
}
fn bvhwChildIndex(node: u32, side: u32) -> u32 {
    return bvhwWord(node, select(3u, 7u, side == 1u));
}
fn bvhwChildCount(node: u32, side: u32) -> u32 {
    return bvhwWord(node, select(11u, 15u, side == 1u));
}

fn bvhwHalfArea(lo: vec3f, hi: vec3f) -> f32 {
    let e = max(hi - lo, vec3f(0.0));
    return e.x * e.y + e.y * e.z + e.z * e.x;
}

// ---- The dynamic program -----------------------------------------------------------

// The cost of child 'side' of source node 'node' spread over i + 1 slots (i in 0..M-1); a leaf takes
// one slot whatever it is offered. 'scratch' is the mesh's scratch base.
fn bvhwChildCost(node: u32, scratch: u32, side: u32, i: u32) -> f32 {
    let count = bvhwChildCount(node, side);
    if (count > 0u) {
        return bvhwHalfArea(bvhwChildMin(node, side), bvhwChildMax(node, side)) * (C_LEAF + f32(count));
    }
    return bvhwF[(scratch + bvhwChildIndex(node, side)) * F_STRIDE + i];
}

@compute @workgroup_size(${WIDE_COLLAPSE_WORKGROUP})
fn bvhwCost(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
    // g: the node in the scratch arrays; node: the same node in the source buffer.
    let g = id.y * groups.x * ${WIDE_COLLAPSE_WORKGROUP}u + id.x;
    if (g >= bvhwParams.totalNodes) {
        return;
    }
    let mesh = bvhwMeshes[bvhwMeshOf(g)];
    let scratch = mesh.scratchBase;
    let node = mesh.srcNode + (g - scratch);
    let info = g * ${WIDE_COLLAPSE_INFO_WORDS}u;
    if ((bvhwU[info + 3u] & USED_DONE) != 0u) {
        return;
    }
    // A single-triangle tree's root: nothing to cost (the decode puts its leaf under a root of its own).
    if (bvhwChildIndex(node, 1u) == EMPTY_CHILD) {
        bvhwU[info + 3u] = USED_DONE;
        return;
    }
    var subFirst = array<u32, 2>(0u, 0u);
    var subCount = array<u32, 2>(0u, 0u);
    var flags = array<u32, 2>(0u, 0u);
    for (var side = 0u; side < 2u; side += 1u) {
        let count = bvhwChildCount(node, side);
        let index = bvhwChildIndex(node, side);
        if (count > 0u) {
            subFirst[side] = index;
            subCount[side] = count;
            flags[side] = CONTIG | ASLEAF;
            continue;
        }
        let childInfo = (scratch + index) * ${WIDE_COLLAPSE_INFO_WORDS}u;
        if ((bvhwU[childInfo + 3u] & USED_DONE) == 0u) {
            return;   // a child not done yet: try again in the next dispatch
        }
        subFirst[side] = bvhwU[childInfo + 1u];
        subCount[side] = bvhwU[childInfo + 2u];
        flags[side] = bvhwU[childInfo + 3u];
    }
    let lo = min(bvhwChildMin(node, 0u), bvhwChildMin(node, 1u));
    let hi = max(bvhwChildMax(node, 0u), bvhwChildMax(node, 1u));
    let sa = bvhwHalfArea(lo, hi);
    let count = subCount[0] + subCount[1];
    let first = min(subFirst[0], subFirst[1]);
    var nodeFlags = 0u;
    if ((flags[0] & flags[1] & CONTIG) != 0u && first + count == max(subFirst[0] + subCount[0], subFirst[1] + subCount[1])) {
        nodeFlags = CONTIG;
    }
    let f = g * F_STRIDE;
    bvhwF[f + M] = lo.x;
    bvhwF[f + M + 1u] = lo.y;
    bvhwF[f + M + 2u] = lo.z;
    bvhwF[f + M + 3u] = hi.x;
    bvhwF[f + M + 4u] = hi.y;
    bvhwF[f + M + 5u] = hi.z;
    bvhwU[info + 1u] = first;
    bvhwU[info + 2u] = count;
    // tinybvh's CombineLeafs(3), which BVH8_CWBVH::Build runs before the collapse: a subtree of at most
    // LEAF_LIMIT contiguous triangles is one leaf, whatever the costs say.
    if (LEAF_LIMIT > 0u && (nodeFlags & CONTIG) != 0u && count <= LEAF_LIMIT) {
        let leafCost = sa * (C_LEAF + f32(count));
        for (var i = 0u; i < M; i += 1u) {
            bvhwF[f + i] = leafCost;
        }
        bvhwU[info] = 0u;
        bvhwU[info + 3u] = CONTIG | ASLEAF | USED_DONE;
        return;
    }
    var cost: array<f32, ${M}>;
    var dist = 0u;
    for (var i = 2u; i <= M; i += 1u) {
        var best = BIG;
        var bestK = 1u;
        for (var k = 1u; k < i; k += 1u) {
            let c = bvhwChildCost(node, scratch, 0u, k - 1u) + bvhwChildCost(node, scratch, 1u, i - k - 1u);
            if (c < best) {
                best = c;
                bestK = k;
            }
        }
        cost[i - 1u] = best;
        dist |= bestK << (4u * (i - 1u));
    }
    let cNode = sa + cost[M - 1u];
    var cLeaf = BIG;
    if ((nodeFlags & CONTIG) != 0u && count <= LEAF_LIMIT) {
        cLeaf = sa * (C_LEAF + f32(count));
    }
    if (cLeaf < cNode) {
        cost[0] = cLeaf;
        nodeFlags |= ASLEAF;
    } else {
        cost[0] = cNode;
    }
    for (var i = 2u; i <= M; i += 1u) {
        if (cost[0] < cost[i - 1u]) {
            cost[i - 1u] = cost[0];
            dist &= ~(15u << (4u * (i - 1u)));   // 0: a single slot
        }
    }
    for (var i = 0u; i < M; i += 1u) {
        bvhwF[f + i] = cost[i];
    }
    bvhwU[info] = dist;
    bvhwU[info + 3u] = nodeFlags | USED_DONE;
}

// ---- One wide level: decode and pack -------------------------------------------------

// A decoded child: its box, and either an inner binary node (local index) or a leaf range.
struct BvhwTerm {
    lo: vec3f,
    hi: vec3f,
    node: u32,       // inner: the binary node it stands for (mesh-local); leaf: 0xffffffff
    first: u32,      // leaf: mesh-local first triangle
    count: u32,      // leaf: triangles
}

// The vnode (inner binary node, or leaf slot) for child 'side' of local node 'n'.
fn bvhwChildVnode(base: u32, n: u32, side: u32) -> u32 {
    if (bvhwChildCount(base + n, side) > 0u) {
        return LEAF_SLOT | (n << 1u) | side;
    }
    return bvhwChildIndex(base + n, side);
}

fn bvhwTermOf(base: u32, scratch: u32, vnode: u32) -> BvhwTerm {
    if ((vnode & LEAF_SLOT) != 0u) {
        let n = (vnode & ~LEAF_SLOT) >> 1u;
        let side = vnode & 1u;
        return BvhwTerm(bvhwChildMin(base + n, side), bvhwChildMax(base + n, side), 0xffffffffu, bvhwChildIndex(base + n, side), bvhwChildCount(base + n, side));
    }
    let f = (scratch + vnode) * F_STRIDE;
    let lo = vec3f(bvhwF[f + M], bvhwF[f + M + 1u], bvhwF[f + M + 2u]);
    let hi = vec3f(bvhwF[f + M + 3u], bvhwF[f + M + 4u], bvhwF[f + M + 5u]);
    let info = (scratch + vnode) * ${WIDE_COLLAPSE_INFO_WORDS}u;
    if ((bvhwU[info + 3u] & ASLEAF) != 0u) {
        return BvhwTerm(lo, hi, 0xffffffffu, bvhwU[info + 1u], bvhwU[info + 2u]);
    }
    return BvhwTerm(lo, hi, vnode, 0u, 0u);
}

// 2^e, exactly (e in [-126, 127]): the bits of the f32 itself, not exp2, which may be approximate.
fn bvhwPow2(e: i32) -> f32 {
    return bitcast<f32>(u32(e + 127) << 23u);
}

// The plane exponent: the smallest e in [-126, 126] with lo + 255 * 2^e at or above hi, in f32.
fn bvhwExponent(lo: f32, hi: f32) -> i32 {
    if (!(hi > lo)) {
        return -126;
    }
    var e = clamp(i32(ceil(log2((hi - lo) / 255.0))), -126, 126);
    for (var guard = 0; guard < 4 && e > -126 && lo + 255.0 * bvhwPow2(e - 1) >= hi; guard += 1) {
        e -= 1;
    }
    for (var guard = 0; guard < 8 && e < 126 && lo + 255.0 * bvhwPow2(e) < hi; guard += 1) {
        e += 1;
    }
    return e;
}

// Outward-rounded planes, checked in f32 as the walk decodes them (p + f32(q) * scale).
fn bvhwQuantizeMin(value: f32, p: f32, scale: f32) -> u32 {
    var q = u32(clamp(floor((value - p) / scale), 0.0, 255.0));
    for (var guard = 0u; guard < 8u && q > 0u && p + f32(q) * scale > value; guard += 1u) {
        q -= 1u;
    }
    return q;
}
fn bvhwQuantizeMax(value: f32, p: f32, scale: f32) -> u32 {
    var q = u32(clamp(ceil((value - p) / scale), 0.0, 255.0));
    for (var guard = 0u; guard < 8u && q < 255u && p + f32(q) * scale < value; guard += 1u) {
        q += 1u;
    }
    return q;
}

fn bvhwEmpty(lo: vec3f, hi: vec3f) -> bool {
    return !all(lo <= hi) || any(abs(lo) > vec3f(BIG)) || any(abs(hi) > vec3f(BIG));
}

fn bvhwOutput(word: u32, value: u32) {
    bvhwOut[word] = value;
}

@compute @workgroup_size(${WIDE_COLLAPSE_WORKGROUP})
fn bvhwLevel(@builtin(global_invocation_id) id: vec3u) {
    let parity = bvhwParams.parity;
    let length = atomicLoad(&bvhwControl[parity]);
    if (id.x >= length) {
        return;
    }
    let entry = bvhwFrontier[parity * bvhwParams.frontierCapacity + id.x];
    let meshIndex = entry.x;
    let src = entry.y;           // the binary node this wide node stands for (mesh-local), or a leaf slot for a one-leaf root
    let address = entry.z;       // its wide node, mesh-local
    let need = entry.w;          // BVH4: stack entries above it; CWBVH: its depth - 1
    let mesh = bvhwMeshes[meshIndex];
    let base = mesh.srcNode;
    let scratch = mesh.scratchBase;
    let counters = CONTROL_HEADER + meshIndex * COUNTER_WORDS;

    // The decode walk (DecodeWalk): the binary nodes that become this node's children, in order.
    var terms: array<u32, ${M}>;
    var termCount = 0u;
    if ((src & LEAF_SLOT) == 0u && bvhwChildIndex(base + src, 1u) == EMPTY_CHILD) {
        terms[0] = LEAF_SLOT | (src << 1u);   // a single-triangle tree: its leaf under the root (tinybvh adds the level)
        termCount = 1u;
    } else {
        var task: array<u32, ${M + 2}>;
        var taskSlots: array<u32, ${M + 2}>;
        var tasks = 1u;
        task[0] = src;
        taskSlots[0] = M;
        loop {
            if (tasks == 0u) {
                break;
            }
            tasks -= 1u;
            let c = task[tasks];
            let slots = taskSlots[tasks];
            if ((c & LEAF_SLOT) == 0u && slots > 1u) {
                let k = (bvhwU[(scratch + c) * ${WIDE_COLLAPSE_INFO_WORDS}u] >> (4u * (slots - 1u))) & 15u;
                if (k != 0u) {
                    // Right first, so left pops first.
                    task[tasks] = bvhwChildVnode(base, c, 1u);
                    taskSlots[tasks] = slots - k;
                    tasks += 1u;
                    task[tasks] = bvhwChildVnode(base, c, 0u);
                    taskSlots[tasks] = k;
                    tasks += 1u;
                    continue;
                }
            }
            if (termCount < M) {
                terms[termCount] = c;
                termCount += 1u;
            }
        }
    }

    // The node's corner: the union of its children's boxes, and the plane scale per axis.
    var boxLo: array<vec3f, ${M}>;
    var boxHi: array<vec3f, ${M}>;
    var nodes: array<u32, ${M}>;
    var firsts: array<u32, ${M}>;
    var counts: array<u32, ${M}>;
    var lo = vec3f(BIG);
    var hi = vec3f(-BIG);
    for (var j = 0u; j < termCount; j += 1u) {
        let term = bvhwTermOf(base, scratch, terms[j]);
        boxLo[j] = term.lo;
        boxHi[j] = term.hi;
        nodes[j] = term.node;
        firsts[j] = term.first;
        counts[j] = term.count;
        if (!bvhwEmpty(term.lo, term.hi)) {
            lo = min(lo, term.lo);
            hi = max(hi, term.hi);
        }
    }
    let emptyAxis = lo > hi;
    lo = select(lo, vec3f(0.0), emptyAxis);
    hi = select(hi, vec3f(0.0), emptyAxis);
    let exponent = vec3i(bvhwExponent(lo.x, hi.x), bvhwExponent(lo.y, hi.y), bvhwExponent(lo.z, hi.z));
    let scale = vec3f(bvhwPow2(exponent.x), bvhwPow2(exponent.y), bvhwPow2(exponent.z));

    // Slots: BVH4 in decode order; CWBVH by tinybvh's greedy octant assignment.
    var slotOf: array<u32, ${M}>;
    for (var j = 0u; j < M; j += 1u) {
        slotOf[j] = j;
    }
    ${cwbvh ? `{
        let centroid = (lo + hi) * 0.5;
        var assigned = 0u;     // bit per child
        var taken = 0u;        // bit per slot
        loop {
            var minCost = BIG;
            var minSlot = 8u;
            var minChild = 8u;
            for (var s = 0u; s < 8u; s += 1u) {
                if ((taken & (1u << s)) != 0u) {
                    continue;
                }
                for (var j = 0u; j < termCount; j += 1u) {
                    if ((assigned & (1u << j)) != 0u) {
                        continue;
                    }
                    let d = (boxLo[j] + boxHi[j]) * 0.5 - centroid;
                    let cost = select(d.x, -d.x, (s & 4u) != 0u) + select(d.y, -d.y, (s & 2u) != 0u) + select(d.z, -d.z, (s & 1u) != 0u);
                    if (cost < minCost) {
                        minCost = cost;
                        minSlot = s;
                        minChild = j;
                    }
                }
            }
            if (minSlot == 8u) {
                break;
            }
            taken |= 1u << minSlot;
            assigned |= 1u << minChild;
            slotOf[minChild] = minSlot;
        }
        // Children left over (no finite cost) take the first free slots.
        for (var j = 0u; j < termCount; j += 1u) {
            if ((assigned & (1u << j)) != 0u) {
                continue;
            }
            for (var s = 0u; s < 8u; s += 1u) {
                if ((taken & (1u << s)) == 0u) {
                    taken |= 1u << s;
                    assigned |= 1u << j;
                    slotOf[j] = s;
                    break;
                }
            }
        }
    }` : ''}

    // Inner children: one consecutive block of wide nodes (in slot order for CWBVH).
    var innerCount = 0u;
    var leafTriangles = 0u;
    for (var j = 0u; j < termCount; j += 1u) {
        if (nodes[j] != 0xffffffffu) {
            innerCount += 1u;
        } else {
            leafTriangles += counts[j];
            if (counts[j] > ${cwbvh ? 3 : 8}u) {
                atomicOr(&bvhwControl[2], PROBLEM_LEAF);
            }
        }
    }
    var childBase = 0u;
    if (innerCount > 0u) {
        childBase = atomicAdd(&bvhwControl[counters], innerCount);
        if (childBase + innerCount > mesh.nodeCount + 1u) {
            atomicOr(&bvhwControl[2], PROBLEM_OUTPUT);
            return;
        }
    }
    ${cwbvh ? `var triangleBase = 0u;
    if (leafTriangles > 0u) {
        triangleBase = atomicAdd(&bvhwControl[counters + 1u], leafTriangles);
    }` : ''}
    ${cwbvh ? 'atomicMax(&bvhwControl[counters + 2u], need + 1u);' : `let needHere = need + innerCount;
    atomicMax(&bvhwControl[counters + 2u], needHere);`}

    let out = (mesh.dstNode + address) * WIDE_WORDS;
    if (out + WIDE_WORDS > bvhwParams.outputNodeWords) {
        atomicOr(&bvhwControl[2], PROBLEM_OUTPUT);
        return;
    }
    // Planes, a byte per child, and the children.
    var planes: array<u32, 12>;   // BVH4: x/y/z min, x/y/z max words; CWBVH: two words each
    for (var w = 0u; w < 12u; w += 1u) {
        planes[w] = 0u;
    }
    var inner = 0u;
    var leafOffset = 0u;
    var imask = 0u;
    var meta0 = 0u;
    var meta1 = 0u;
    // Visit children in slot order, so CWBVH's inner children and leaf triangles follow their slots.
    for (var s = 0u; s < M; s += 1u) {
        var j = M;
        for (var t = 0u; t < termCount; t += 1u) {
            if (slotOf[t] == s) {
                j = t;
            }
        }
        if (j == M) {
            continue;
        }
        var q = array<u32, 6>(255u, 255u, 255u, 0u, 0u, 0u);
        if (!bvhwEmpty(boxLo[j], boxHi[j])) {
            q[0] = bvhwQuantizeMin(boxLo[j].x, lo.x, scale.x);
            q[1] = bvhwQuantizeMin(boxLo[j].y, lo.y, scale.y);
            q[2] = bvhwQuantizeMin(boxLo[j].z, lo.z, scale.z);
            q[3] = bvhwQuantizeMax(boxHi[j].x, lo.x, scale.x);
            q[4] = bvhwQuantizeMax(boxHi[j].y, lo.y, scale.y);
            q[5] = bvhwQuantizeMax(boxHi[j].z, lo.z, scale.z);
        }
        ${cwbvh ? `for (var k = 0u; k < 6u; k += 1u) {
            planes[k * 2u + (s >> 2u)] |= q[k] << (8u * (s & 3u));
        }
        var metaByte = 0u;
        if (nodes[j] != 0xffffffffu) {
            let childAddress = childBase + inner;
            inner += 1u;
            imask |= 1u << s;
            metaByte = (1u << 5u) | (24u + s);
            let slot = atomicAdd(&bvhwControl[1u - parity], 1u);
            if (slot >= bvhwParams.frontierCapacity) {
                atomicOr(&bvhwControl[2], PROBLEM_FRONTIER);
            } else {
                bvhwFrontier[(1u - parity) * bvhwParams.frontierCapacity + slot] = vec4u(meshIndex, nodes[j], childAddress, need + 1u);
            }
        } else {
            let count = counts[j];
            let unary = select(select(7u, 3u, count == 2u), 1u, count == 1u);
            metaByte = (unary << 5u) | leafOffset;
            // The leaf's triangles, copied into slot order.
            let triangleOut = bvhwParams.outputNodeWords + (mesh.dstTriangle + triangleBase + leafOffset) * UNIT_WORDS;
            for (var k = 0u; k < count * UNIT_WORDS; k += 1u) {
                bvhwOut[triangleOut + k] = bvhwTriangles[(mesh.srcTriangle + firsts[j]) * UNIT_WORDS + k];
            }
            leafOffset += count;
        }
        if (s < 4u) {
            meta0 |= metaByte << (8u * s);
        } else {
            meta1 |= metaByte << (8u * (s - 4u));
        }` : `for (var k = 0u; k < 6u; k += 1u) {
            planes[k] |= q[k] << (8u * s);
        }
        var info = 0u;
        if (nodes[j] != 0xffffffffu) {
            let childAddress = childBase + inner;
            inner += 1u;
            info = childAddress;
            let slot = atomicAdd(&bvhwControl[1u - parity], 1u);
            if (slot >= bvhwParams.frontierCapacity) {
                atomicOr(&bvhwControl[2], PROBLEM_FRONTIER);
            } else {
                bvhwFrontier[(1u - parity) * bvhwParams.frontierCapacity + slot] = vec4u(meshIndex, nodes[j], childAddress, needHere - 1u);
            }
        } else {
            info = ${hex(BVH4_CHILD.leafBit)} | ((counts[j] - 1u) << ${BVH4_CHILD.countShift}u) | firsts[j];
        }
        bvhwOutput(out + 12u + s, info);`}
    }
    ${cwbvh ? `bvhwOutput(out, bitcast<u32>(lo.x));
    bvhwOutput(out + 1u, bitcast<u32>(lo.y));
    bvhwOutput(out + 2u, bitcast<u32>(lo.z));
    bvhwOutput(out + 3u, u32(exponent.x + 127) | (u32(exponent.y + 127) << 8u) | (u32(exponent.z + 127) << 16u) | (imask << 24u));
    bvhwOutput(out + 4u, childBase);
    bvhwOutput(out + 5u, triangleBase);
    bvhwOutput(out + 6u, meta0);
    bvhwOutput(out + 7u, meta1);
    // Plane words 8-19: lo x, lo y, lo z, hi x, hi y, hi z, two words each.
    for (var w = 0u; w < 12u; w += 1u) {
        bvhwOutput(out + 8u + w, planes[w]);
    }` : `bvhwOutput(out, bitcast<u32>(lo.x));
    bvhwOutput(out + 1u, bitcast<u32>(lo.y));
    bvhwOutput(out + 2u, bitcast<u32>(lo.z));
    bvhwOutput(out + 3u, planes[0]);
    bvhwOutput(out + 4u, bitcast<u32>(scale.x));
    bvhwOutput(out + 5u, bitcast<u32>(scale.y));
    bvhwOutput(out + 6u, bitcast<u32>(scale.z));
    bvhwOutput(out + 7u, planes[3]);
    bvhwOutput(out + 8u, planes[1]);
    bvhwOutput(out + 9u, planes[4]);
    bvhwOutput(out + 10u, planes[2]);
    bvhwOutput(out + 11u, planes[5]);
    // Child words: empty slots stay 0 (the output is cleared before the collapse).`}
}

// One thread: the frontier just written becomes the next level's indirect dispatch; the one just read is cleared.
@compute @workgroup_size(1)
fn bvhwArgs() {
    let parity = bvhwParams.parity;
    let next = min(atomicLoad(&bvhwControl[1u - parity]), bvhwParams.frontierCapacity);
    bvhwIndirect[0] = (next + ${WIDE_COLLAPSE_WORKGROUP - 1}u) / ${WIDE_COLLAPSE_WORKGROUP}u;
    bvhwIndirect[1] = 1u;
    bvhwIndirect[2] = 1u;
    atomicStore(&bvhwControl[parity], 0u);
}
`;
}

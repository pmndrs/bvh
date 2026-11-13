/*!
 * The BVH4 and CWBVH walks below are ported from tinybvh's BVH4_GPU::Intersect and
 * BVH8_CWBVH::Intersect (tiny_bvh_base.h; Ylitie, Karras and Laine 2017);
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
 * The wide walks (docs/internals/design/wide-nodes.md): `bvhWalkClosest` and
 * `bvhWalkAny` for a scene whose mesh trees are BVH4 or CWBVH nodes
 * (`nodeFormat: 'bvh4' | 'cwbvh8'`). The top level stays the binary `f32`
 * tree and is walked as in wgsl.ts; at an instance leaf the ray moves into
 * object space and a mesh walk (`bvhMeshClosest`, `bvhMeshAny`) runs to the
 * end of the instance's tree with its own stack, then the top-level walk
 * goes on. Same names, same filter hook, same masks, the same watertight
 * triangle test and the same hit as the binary walk.
 *
 * - **BVH4** is tinybvh's `BVH4_GPU::Intersect`: decode the four child boxes,
 *   test them, sort them by entry distance with a five-swap network, test
 *   the leaf children's triangles at once, push the inner children far to
 *   near, pop. Stack entries are child words.
 * - **CWBVH** is Ylitie, Karras and Laine 2017 as tinybvh's
 *   `BVH8_CWBVH::Intersect` writes it: a stack of node groups (child base
 *   index, a hit bit per inner child in bits 24-31 and the parent's inner
 *   mask in bits 0-7); children are taken highest bit first, and the bit of
 *   an inner child is `24 + (slot ^ octant)`, so the slot order the encoder
 *   chose by octant gives a near-to-far order with no sort. Leaf children set
 *   triangle bits 0-23 (the triangle group), tested before the next child.
 *   `__byte_perm` and `prmt` are `unpack4xU8`, `extractBits` and a multiply
 *   by 0xff.
 *
 * Plane decode: every plane is `p + f32(q) * 2^e` (an exact product, one
 * rounding), and the slab test is the binary walk's `bvhBoxEntry`
 * arithmetic, so a decoded box that contains the exact box is entered by
 * every ray that enters the exact box (the slab test is monotone in its
 * planes). Ylitie folds the decode into the slab test
 * (`q * (2^e / d) + (p - o) / d`), one FMA per plane, which is not exact;
 * this port keeps the exact planes (see the design note's lever list).
 */

import {
    BVH4_CHILD,
    BVH4_NODE_WORDS,
    CWBVH_NODE_WORDS,
    type WideNodeFormat,
} from '../wide-nodes.ts';

/** Where the CWBVH walk keeps its stack of node groups (`TraceModuleOptions.wideStack`). */
export type WideStack = 'private' | 'workgroup';

/** Mesh-walk stack entries, per format. `assembleTraceScene` checks every tree against these. */
export const WIDE_MESH_STACK_SIZE: Record<WideNodeFormat, number> = { bvh4: 64, cwbvh8: 32 };

/** Workgroup short stack: entries per lane held in workgroup memory; deeper entries spill to a private array. */
export const WIDE_SHARED_STACK_ENTRIES = 8;

/** Lanes the workgroup short stack has room for: the batch kernel's workgroup size. */
export const WIDE_SHARED_STACK_LANES = 64;

const hex = (value: number): string => `0x${(value >>> 0).toString(16)}u`;

/** Options of {@link wideWalkWgsl}. */
export interface WideWalkOptions {
    readonly format: WideNodeFormat;
    readonly rayMasks: boolean;
    /** Any-hit child order: `'far'` (the default) or `'near'`; `'larger'` has no wide form. */
    readonly anyHitOrder: 'far' | 'near';
    /** CWBVH only. */
    readonly stack: WideStack;
    /** BVH4 only: `'nested'` (a mesh loop per instance) or `'single'` (one loop, lever L7). */
    readonly walk: WideWalk;
    /** How child planes are decoded: `'exact'` or `'folded'` (lever L1); see {@link wideEntryWordsWgsl}. */
    readonly decode: WideDecode;
}

/** How the wide walks decode a child's planes (`TraceModuleOptions.wideDecode`). */
export type WideDecode = 'exact' | 'folded';

/** How a BVH4 scene is walked (`TraceModuleOptions.wideWalk`): a mesh loop per instance, or one loop for both levels. */
export type WideWalk = 'nested' | 'single';

/** The single-loop walk's stack: the top level's (the binary walk's) and the deepest BVH4 mesh's together. */
export const WIDE_SINGLE_STACK_SIZE = 136;

/**
 * The per-triangle part of a leaf, shared by both formats: test, filter, and
 * keep (closest) or return (any). `slot` is the triangle's scene slot.
 */
function triangleTestWgsl(kind: 'closest' | 'any', rayMasks: boolean): string {
    const closest = kind === 'closest';
    const candidateMask = rayMasks ? 'rayMask' : 'BVH_RAY_MASK_ALL';
    const query = closest ? 'BVH_QUERY_CLOSEST' : 'BVH_QUERY_ANY';
    return `let triangle = bvhLoadTriangle(slot, meshVertexBase);
            let triangleHit = bvhIntersectWatertight(triangle, setup, tMin, ${closest ? '(*best).t' : 'tMax'});
            if (!triangleHit.hit) {
                continue;
            }
            let hitInstance = bvhHitInstance(instance, triangle, slot);
            let frontFace = triangleHit.frontFacing != ((triangle.flags & BVH_TRIANGLE_FLAG_FLIP_FRONT_FACE) != 0u);
            if ((triangle.flags & BVH_TRIANGLE_FLAG_FILTER) != 0u) {
                let candidate = BvhCandidate(hitInstance, triangle.originalIndex, triangleHit.t, triangleHit.barycentrics, frontFace, ${query}, ${candidateMask});
                if (bvhTrianglePasses(candidate)) {
                    continue;
                }
            }
            ${closest ? '*best = BvhMeshHit(true, triangleHit.t, slot, triangleHit.barycentrics, frontFace, hitInstance, triangle.originalIndex);' : 'return true;'}`;
}

/**
 * Tests the triangles of one leaf or triangle group: bit `k` of `bits` is
 * the triangle at scene slot `firstSlot + k`, lowest bit first (the binary
 * walk's order within a leaf).
 */
function triangleGroupWgsl(kind: 'closest' | 'any', rayMasks: boolean): string {
    const closest = kind === 'closest';
    const maskParameter = rayMasks ? ', rayMask: u32' : '';
    return /* wgsl */ `
fn ${closest ? 'bvhTrianglesClosest' : 'bvhTrianglesAny'}(instance: BvhInstance, setup: BvhRaySetup, tMin: f32, ${closest ? 'best: ptr<function, BvhMeshHit>' : 'tMax: f32'}, meshVertexBase: u32, firstSlot: u32, group: u32${maskParameter}) ${closest ? '' : '-> bool '}{
    var bits = group;
    loop {
        if (bits == 0u) {
            break;
        }
        let k = firstTrailingBit(bits);
        bits &= bits - 1u;
        let slot = firstSlot + k;
        {
            ${triangleTestWgsl(kind, rayMasks)}
        }
    }
    ${closest ? '' : 'return false;'}
}
`;
}

/** The decoded boxes of four children, tested at once: entry distance per child, or BVH_BOX_MISS. */
const WIDE_BOX_WGSL = /* wgsl */ `
// Four children's planes on one axis: min and max as 8-bit steps of 2^e from the node corner p.
// Each decodes as p + f32(q) * 2^e: the product is exact, so fused or not it is one rounding.
fn bvhWidePlanes(p: f32, scale: f32, q: u32) -> vec4f {
    return p + vec4f(unpack4xU8(q)) * scale;
}

// The slab test of bvhBoxEntry for four boxes at once, with the near and far plane picked by the
// direction's sign (the same values min/max pick for a box with min <= max; an inverted box,
// an empty child's, is never entered).
fn bvhWideEntry(
    xMin: vec4f, xMax: vec4f, yMin: vec4f, yMax: vec4f, zMin: vec4f, zMax: vec4f,
    setup: BvhRaySetup, tMin: f32, tMax: f32,
) -> vec4f {
    let inverse = setup.inverseDirection;
    let negative = inverse < vec3f(0.0);
    let x0 = (select(xMin, xMax, negative.x) - setup.origin.x) * inverse.x;
    let x1 = (select(xMax, xMin, negative.x) - setup.origin.x) * inverse.x;
    let y0 = (select(yMin, yMax, negative.y) - setup.origin.y) * inverse.y;
    let y1 = (select(yMax, yMin, negative.y) - setup.origin.y) * inverse.y;
    let z0 = (select(zMin, zMax, negative.z) - setup.origin.z) * inverse.z;
    let z1 = (select(zMax, zMin, negative.z) - setup.origin.z) * inverse.z;
    let enter = max(max(x0, max(y0, z0)), vec4f(tMin));
    let exit = min(min(x1, min(y1, z1)) * BVH_BOX_EXIT_PADDING, vec4f(tMax));
    return select(vec4f(BVH_BOX_MISS), enter, enter <= exit);
}
`;

/**
 * Four children's entry distances from their plane words (min x, max x, min y, max y, min z,
 * max z: a byte per child), with the node corner `p` and plane scale `2^e`, by `wideDecode`:
 *
 * - `'exact'` (the default): each plane decoded as `p + f32(q) * 2^e` (one rounding of an exact
 *   value), then the binary walk's slab arithmetic (`bvhWideEntry`). A decoded box that contains
 *   the exact box is entered by every ray that enters the exact box, so hits equal the binary walk's.
 * - `'folded'` (lever L1, Ylitie's form): per node and axis `a = 2^e / d` and `b = (p - o) / d`,
 *   then each plane's distance is one FMA, `q * a + b`. That is not the exact plane's distance,
 *   and its error is absolute (it grows with `|p - o|`), so each axis's near distance moves down
 *   and its far distance up by `2^-20 * (|b| + (|p| + 255 * 2^e) * |1/d|)`: more than the
 *   folded arithmetic's error, the exact path's, and the half ulp by which the plane rounds,
 *   together. The box only grows, so the hit argument holds; the trace-oracle checks it.
 */
function wideEntryWordsWgsl(decode: WideDecode): string {
    if (decode === 'exact') {
        return /* wgsl */ `
fn bvhWideEntryWords(p: vec3f, scale: vec3f, xMin: u32, xMax: u32, yMin: u32, yMax: u32, zMin: u32, zMax: u32, setup: BvhRaySetup, tMin: f32, tMax: f32) -> vec4f {
    return bvhWideEntry(
        bvhWidePlanes(p.x, scale.x, xMin), bvhWidePlanes(p.x, scale.x, xMax),
        bvhWidePlanes(p.y, scale.y, yMin), bvhWidePlanes(p.y, scale.y, yMax),
        bvhWidePlanes(p.z, scale.z, zMin), bvhWidePlanes(p.z, scale.z, zMax),
        setup, tMin, tMax,
    );
}
`;
    }
    return /* wgsl */ `
// The folded decode's padding factor, 2^-20 (see wideEntryWordsWgsl in wgsl-wide.ts).
const BVH_WIDE_FOLD_PAD: f32 = 9.5367431640625e-7;

fn bvhWideEntryWords(p: vec3f, scale: vec3f, xMin: u32, xMax: u32, yMin: u32, yMax: u32, zMin: u32, zMax: u32, setup: BvhRaySetup, tMin: f32, tMax: f32) -> vec4f {
    let inverse = setup.inverseDirection;
    let a = scale * inverse;
    let b = (p - setup.origin) * inverse;
    let pad = (abs(b) + (abs(p) + 255.0 * scale) * abs(inverse)) * BVH_WIDE_FOLD_PAD;
    let negative = inverse < vec3f(0.0);
    // The near plane is the min plane for a positive direction, the max plane for a negative one.
    let x0 = fma(vec4f(unpack4xU8(select(xMin, xMax, negative.x))), vec4f(a.x), vec4f(b.x)) - pad.x;
    let x1 = fma(vec4f(unpack4xU8(select(xMax, xMin, negative.x))), vec4f(a.x), vec4f(b.x)) + pad.x;
    let y0 = fma(vec4f(unpack4xU8(select(yMin, yMax, negative.y))), vec4f(a.y), vec4f(b.y)) - pad.y;
    let y1 = fma(vec4f(unpack4xU8(select(yMax, yMin, negative.y))), vec4f(a.y), vec4f(b.y)) + pad.y;
    let z0 = fma(vec4f(unpack4xU8(select(zMin, zMax, negative.z))), vec4f(a.z), vec4f(b.z)) - pad.z;
    let z1 = fma(vec4f(unpack4xU8(select(zMax, zMin, negative.z))), vec4f(a.z), vec4f(b.z)) + pad.z;
    let enter = max(max(x0, max(y0, z0)), vec4f(tMin));
    let exit = min(min(x1, min(y1, z1)) * BVH_BOX_EXIT_PADDING, vec4f(tMax));
    return select(vec4f(BVH_BOX_MISS), enter, enter <= exit);
}
`;
}

/**
 * One BVH4 node (tinybvh's `BVH4_GPU::Intersect` step) as text: decode and test the four
 * children of the node at mesh node `node`, sort them, test the leaf children's triangles
 * (`leafCall`) and push the inner ones (`push`), far to near (so the nearest is popped first)
 * for closest-hit, the other way for any-hit 'far'. `tFar` is the box test's far limit.
 */
function bvh4NodeStepWgsl(closest: boolean, anyHitOrder: 'far' | 'near', tFar: string, leafCall: (info: string) => string, push: (info: string) => string): string {
    const order = closest || anyHitOrder === 'near' ? [0, 1, 2, 3] : [3, 2, 1, 0];
    const children = order
        .map(
            (k) => `
        if (dist${k} < BVH_BOX_MISS) {
            if ((info${k} & BVH4_CHILD_LEAF) != 0u) {
                ${leafCall(`info${k}`)}
            } else {
                ${push(`info${k}`)}
            }
        }`,
        )
        .join('');
    return `let base = (instance.rootNodeOffset + node) * BVH4_VEC4_PER_NODE;
        let d0 = bvhTraceNodes[base];
        let d1 = bvhTraceNodes[base + 1u];
        let d2 = bvhTraceNodes[base + 2u];
        let d3 = bvhTraceNodes[base + 3u];
        let p = bitcast<vec3f>(d0.xyz);
        let scale = bitcast<vec3f>(d1.xyz);
        let entry = bvhWideEntryWords(p, scale, d0.w, d1.w, d2.x, d2.y, d2.z, d2.w, setup, tMin, ${tFar});
        // A missing child (word 0) is never entered.
        let dist = select(entry, vec4f(BVH_BOX_MISS), d3 == vec4u(0u));
        var dist0 = dist.x;
        var dist1 = dist.y;
        var dist2 = dist.z;
        var dist3 = dist.w;
        var info0 = d3.x;
        var info1 = d3.y;
        var info2 = d3.z;
        var info3 = d3.w;
        // tinybvh's sorting network: afterwards dist0 >= dist1 >= dist2 >= dist3.
        if (dist0 < dist2) { let t = dist0; dist0 = dist2; dist2 = t; let i = info0; info0 = info2; info2 = i; }
        if (dist1 < dist3) { let t = dist1; dist1 = dist3; dist3 = t; let i = info1; info1 = info3; info3 = i; }
        if (dist0 < dist1) { let t = dist0; dist0 = dist1; dist1 = t; let i = info0; info0 = info1; info1 = i; }
        if (dist2 < dist3) { let t = dist2; dist2 = dist3; dist3 = t; let i = info2; info2 = info3; info3 = i; }
        if (dist1 < dist2) { let t = dist1; dist1 = dist2; dist2 = t; let i = info1; info1 = info2; info2 = i; }
        ${children}`;
}

/** A BVH4 leaf child's call into the triangle test: closest keeps `best`, any returns at the first blocker. */
function bvh4LeafCall(closest: boolean, maskArgument: string, tMax: string): (info: string) => string {
    const first = (info: string) => `instance.triangleOffset + (${info} & BVH4_CHILD_INDEX_MASK)`;
    const bits = (info: string) => `(2u << ((${info} >> BVH4_CHILD_COUNT_SHIFT) & 7u)) - 1u`;
    return closest
        ? (info) => `bvhTrianglesClosest(instance, setup, tMin, &best, meshVertexBase, ${first(info)}, ${bits(info)}${maskArgument});`
        : (info) => `if (bvhTrianglesAny(instance, setup, tMin, ${tMax}, meshVertexBase, ${first(info)}, ${bits(info)}${maskArgument})) {
                    return true;
                }`;
}

/** BVH4: tinybvh's BVH4_GPU walk, one instance's tree to the end (`wideWalk: 'nested'`). */
function bvh4MeshWalkWgsl(kind: 'closest' | 'any', rayMasks: boolean, anyHitOrder: 'far' | 'near'): string {
    const closest = kind === 'closest';
    // The mesh stack is a module-scope private array (meshStackWgsl): a function-scope one would be
    // zero-filled at every instance entry.
    const stackName = closest ? 'bvhMeshStackClosest' : 'bvhMeshStackAny';
    const maskParameter = rayMasks ? ', rayMask: u32' : '';
    const maskArgument = rayMasks ? ', rayMask' : '';
    const step = bvh4NodeStepWgsl(closest, anyHitOrder, closest ? 'best.t' : 'tMax', bvh4LeafCall(closest, maskArgument, 'tMax'), (info) => `${stackName}[stackSize] = ${info};
                stackSize += 1u;`);
    return /* wgsl */ `
fn ${closest ? 'bvhMeshClosest' : 'bvhMeshAny'}(instance: BvhInstance, setup: BvhRaySetup, tMin: f32, tMax: f32, meshVertexBase: u32${maskParameter}) -> ${closest ? 'BvhMeshHit' : 'bool'} {
    ${closest ? 'var best = BvhMeshHit(false, tMax, 0u, vec2f(0.0), true, 0u, 0u);' : ''}
    var stackSize = 0u;
    var node = 0u;
    loop {
        ${step}
        if (stackSize == 0u) {
            break;
        }
        stackSize -= 1u;
        node = ${stackName}[stackSize];
    }
    return ${closest ? 'best' : 'false'};
}
`;
}

/**
 * BVH4 in one loop (`wideWalk: 'single'`, lever L7): the binary walk's shape, one loop and one
 * stack for both levels with the instance marker, so a lane that finishes an instance's tree
 * goes on with the top level in the same iteration as lanes still in theirs. Top-level entries
 * are the binary walk's references; mesh entries are BVH4 inner child words (node indices, never
 * the marker). The node step is the nested walk's.
 */
function bvh4SingleWalkWgsl(kind: 'closest' | 'any', rayMasks: boolean, anyHitOrder: 'far' | 'near'): string {
    const closest = kind === 'closest';
    const maskParameter = rayMasks ? ', rayMask: u32' : '';
    const maskArgument = rayMasks ? ', rayMask' : '';
    const name = closest ? 'bvhWalkClosest' : 'bvhWalkAny';
    const tMin = 'ray.tMin';
    const leaf = bvh4LeafCall(closest, maskArgument, 'farthest');
    // Closest: a leaf that changes the best hit names this instance's record.
    const leafCall = closest
        ? (info: string) => `{
                    let before = best;
                    ${leaf(info)}
                    if (best.found && (!before.found || best.t != before.t || best.slot != before.slot || best.instance != before.instance)) {
                        bestRecord = instanceRecord;
                    }
                }`
        : leaf;
    const step = bvh4NodeStepWgsl(closest, anyHitOrder, closest ? 'best.t' : 'farthest', leafCall, (info) => `stack[stackSize] = ${info};
                stackSize += 1u;`);
    return /* wgsl */ `
fn ${name}(ray: BvhRay${maskParameter}) -> ${closest ? 'BvhHit' : 'bool'} {
    let header = bvhTraceTopLevel[0];
    let instanceCount = header.x;
    let topLevelNodeCount = header.y;
    ${closest ? `var best = BvhMeshHit(false, ray.tMax, 0u, vec2f(0.0), true, 0u, 0u);
    var bestRecord = 0u;
    var instanceRecord = 0u;` : 'let farthest = ray.tMax;'}
    if (topLevelNodeCount == 0u) {
        return ${closest ? 'BvhHit(-1.0, 0u, 0u, vec2f(0.0), true, BVH_NO_MATERIAL, vec3f(0.0))' : 'false'};
    }
    let firstTopLevelNode = BVH_TOP_HEADER_VEC4S + instanceCount * BVH_TOP_VEC4_PER_RECORD;
    let worldSetup = bvhMakeRaySetup(ray.origin, ray.direction);
    var setup = worldSetup;
    let tMin = ${tMin};
    var insideInstance = false;
    var instance: BvhInstance;
    var meshVertexBase = 0u;

    var stack: array<u32, BVH_SINGLE_STACK_SIZE>;
    var stackSize = 0u;
    var current = bvhMakeReference(0u, 0u);   // the top level's root; inside an instance, a mesh node

    loop {
        if (insideInstance) {
            let node = current;
            ${step}
        } else {
            let count = current >> BVH_REF_COUNT_SHIFT;
            let index = current & BVH_REF_INDEX_MASK;
            if (count == 0u) {
                let topNode = bvhLoadTopLevelNode(firstTopLevelNode, index);
                let tTop = ${closest ? 'best.t' : 'farthest'};
                let leftEntry = bvhBoxEntry(topNode.leftMin, topNode.leftMax, setup, tMin, tTop);
                var rightEntry = BVH_BOX_MISS;
                if (topNode.rightIndex != BVH_EMPTY_CHILD) {
                    rightEntry = bvhBoxEntry(topNode.rightMin, topNode.rightMax, setup, tMin, tTop);
                }
                let leftReference = bvhMakeReference(topNode.leftIndex, topNode.leftCount);
                let rightReference = bvhMakeReference(topNode.rightIndex, topNode.rightCount);
                let hitLeft = leftEntry < BVH_BOX_MISS;
                let hitRight = rightEntry < BVH_BOX_MISS;
                if (hitLeft && hitRight) {
                    if (${closest || anyHitOrder === 'near' ? 'leftEntry <= rightEntry' : 'leftEntry > rightEntry'}) {
                        stack[stackSize] = rightReference;
                        current = leftReference;
                    } else {
                        stack[stackSize] = leftReference;
                        current = rightReference;
                    }
                    stackSize += 1u;
                    continue;
                }
                if (hitLeft) {
                    current = leftReference;
                    continue;
                }
                if (hitRight) {
                    current = rightReference;
                    continue;
                }
            } else {
                // A top-level leaf is one instance: into its object space, and on at its tree's root.
                instance = bvhLoadInstance(index);
                ${closest ? 'instanceRecord = index;' : ''}
                ${rayMasks ? 'if ((bvhInstanceMask(instance) & rayMask) != 0u) {' : '{'}
                    setup = bvhMakeRaySetup(bvhTransformPoint(instance, ray.origin), bvhTransformDirection(instance, ray.direction));
                    meshVertexBase = bvhMeshVertexBase(instance.triangleOffset);
                    stack[stackSize] = BVH_REF_INSTANCE_MARKER;
                    stackSize += 1u;
                    insideInstance = true;
                    current = 0u;
                    continue;
                }
            }
        }
        // Take the next entry off the stack; the marker ends an instance's tree.
        loop {
            if (stackSize == 0u) {
                ${closest ? `if (!best.found) {
                    return BvhHit(-1.0, 0u, 0u, vec2f(0.0), true, BVH_NO_MATERIAL, vec3f(0.0));
                }
                return bvhFinishClosest(BvhHit(best.t, best.instance, best.originalIndex, best.barycentrics, best.frontFace, BVH_NO_MATERIAL, vec3f(0.0)), bestRecord, best.slot);` : 'return false;'}
            }
            stackSize -= 1u;
            let entry = stack[stackSize];
            if (entry == BVH_REF_INSTANCE_MARKER) {
                insideInstance = false;
                setup = worldSetup;
                continue;
            }
            current = entry;
            break;
        }
    }
}
`;
}

/** CWBVH: Ylitie et al. as tinybvh's BVH8_CWBVH walk. */
function cwbvhMeshWalkWgsl(kind: 'closest' | 'any', rayMasks: boolean, anyHitOrder: 'far' | 'near', stack: WideStack): string {
    const closest = kind === 'closest';
    const maskParameter = rayMasks ? ', rayMask: u32' : '';
    const maskArgument = rayMasks ? ', rayMask' : '';
    // octinv = 7 - octant visits the nearest slots first (Ylitie). Any-hit 'far' XORs with the
    // octant itself instead, which reverses the order: the farthest slots first.
    const nearFirst = closest || anyHitOrder === 'near';
    const octantInverse = nearFirst ? '7u - octant' : 'octant';
    const shared = stack === 'workgroup';
    // Module-scope stacks (meshStackWgsl), one per query: a function-scope array would be zero-filled at every instance entry.
    const suffix = closest ? 'Closest' : 'Any';
    const push = shared ? `bvhWidePush${suffix}(stackSize, ngroup);` : `bvhMeshStack${suffix}[stackSize] = ngroup;`;
    const pop = shared ? `ngroup = bvhWidePop${suffix}(stackSize);` : `ngroup = bvhMeshStack${suffix}[stackSize];`;
    const triangles = closest
        ? `bvhTrianglesClosest(instance, setup, tMin, &best, meshVertexBase, instance.triangleOffset + triangleBase, triangleBits${maskArgument});`
        : `if (bvhTrianglesAny(instance, setup, tMin, tMax, meshVertexBase, instance.triangleOffset + triangleBase, triangleBits${maskArgument})) {
                return true;
            }`;
    return /* wgsl */ `
fn ${closest ? 'bvhMeshClosest' : 'bvhMeshAny'}(instance: BvhInstance, setup: BvhRaySetup, tMin: f32, tMax: f32, meshVertexBase: u32${maskParameter}) -> ${closest ? 'BvhMeshHit' : 'bool'} {
    ${closest ? 'var best = BvhMeshHit(false, tMax, 0u, vec2f(0.0), true, 0u, 0u);' : ''}
    var stackSize = 0u;
    let negative = setup.inverseDirection < vec3f(0.0);
    let octant = select(0u, 4u, negative.x) | select(0u, 2u, negative.y) | select(0u, 1u, negative.z);
    let octantInverse = ${octantInverse};
    let octantInverse4 = octantInverse * 0x01010101u;
    // The root as a group of one (tinybvh): child base 0, hit bit 31 and no inner mask, so it is child 0.
    var ngroup = vec2u(0u, 0x80000000u);
    loop {
        // Take the next inner child of the group: the highest hit bit.
        let hits = ngroup.y;
        let childBit = firstLeadingBit(hits);
        let childBase = ngroup.x;
        ngroup.y &= ~(1u << childBit);
        if (ngroup.y > 0x00ffffffu) {
            ${push}
            stackSize += 1u;
        }
        let slot = (childBit - 24u) ^ octantInverse;
        let relative = countOneBits(hits & ~(0xffffffffu << slot));
        let at = (instance.rootNodeOffset + childBase + relative) * CWBVH_VEC4_PER_NODE;
        let n0 = bvhTraceNodes[at];
        let n1 = bvhTraceNodes[at + 1u];
        let n2 = bvhTraceNodes[at + 2u];
        let n3 = bvhTraceNodes[at + 3u];
        let n4 = bvhTraceNodes[at + 4u];
        let p = bitcast<vec3f>(n0.xyz);
        let e4 = n0.w;
        // 2^e from the biased exponent bytes: the bytes are the f32 exponent field.
        let scale = bitcast<vec3f>(vec3u(extractBits(e4, 0u, 8u), extractBits(e4, 8u, 8u), extractBits(e4, 16u, 8u)) << vec3u(23u));
        let tFar = ${closest ? 'best.t' : 'tMax'};
        // Slots 0-3, then 4-7. Plane words: lo x n2.x|y, lo y n2.z|w, lo z n3.x|y, hi x n3.z|w, hi y n4.x|y, hi z n4.z|w.
        let entryA = bvhWideEntryWords(p, scale, n2.x, n3.z, n2.z, n4.x, n3.x, n4.z, setup, tMin, tFar);
        let entryB = bvhWideEntryWords(p, scale, n2.y, n3.w, n2.w, n4.y, n3.y, n4.w, setup, tMin, tFar);
        let hitmask = bvhCwbvhHitBits(n1.z, octantInverse4, entryA) | bvhCwbvhHitBits(n1.w, octantInverse4, entryB);
        ngroup = vec2u(n1.x, (hitmask & 0xff000000u) | (e4 >> 24u));
        let triangleBase = n1.y;
        let triangleBits = hitmask & 0x00ffffffu;
        if (triangleBits != 0u) {
            ${triangles}
        }
        if (ngroup.y > 0x00ffffffu) {
            continue;
        }
        if (stackSize == 0u) {
            break;
        }
        stackSize -= 1u;
        ${pop}
    }
    return ${closest ? 'best' : 'false'};
}
`;
}

/** CWBVH's per-four-slots hit bits: Ylitie's meta decode, with the byte permutes as integer arithmetic. */
const CWBVH_HIT_BITS_WGSL = /* wgsl */ `
// meta4: four meta bytes. An inner child's meta is (1 << 5) | (24 + slot): bits 3 and 4 both set,
// which no leaf offset (0-23) has. Its hit bit is 24 + (slot ^ octant); a leaf's bits are its unary
// triangle count shifted to its triangle offset.
fn bvhCwbvhHitBits(meta4: u32, octantInverse4: u32, entry: vec4f) -> u32 {
    let isInner4 = (meta4 & (meta4 << 1u)) & 0x10101010u;
    // prmt's sign extension: 0xff in each byte of an inner child.
    let innerMask4 = ((isInner4 >> 4u) & 0x01010101u) * 0xffu;
    let bitIndex4 = (meta4 ^ (octantInverse4 & innerMask4)) & 0x1f1f1f1fu;
    let childBits4 = (meta4 >> 5u) & 0x07070707u;
    let bits = unpack4xU8(childBits4) << unpack4xU8(bitIndex4);
    let hit = select(vec4u(0u), bits, entry < vec4f(BVH_BOX_MISS));
    return hit.x | hit.y | hit.z | hit.w;
}
`;

/**
 * The mesh walks' stacks, at module scope: `var<private>` is zero-initialised once per invocation,
 * where a function-scope array would be cleared at every instance entry. One per query, so a
 * triangle filter in a closest-hit walk may itself call `bvhTraceAny`.
 */
function meshStackWgsl(format: WideNodeFormat, stack: WideStack): string {
    if (format === 'bvh4') {
        return `var<private> bvhMeshStackClosest: array<u32, BVH_WIDE_STACK_SIZE>;
var<private> bvhMeshStackAny: array<u32, BVH_WIDE_STACK_SIZE>;`;
    }
    if (stack === 'private') {
        return `var<private> bvhMeshStackClosest: array<vec2u, BVH_WIDE_STACK_SIZE>;
var<private> bvhMeshStackAny: array<vec2u, BVH_WIDE_STACK_SIZE>;`;
    }
    const lanes = WIDE_SHARED_STACK_LANES;
    const entries = WIDE_SHARED_STACK_ENTRIES;
    const region = (suffix: string) => `
var<workgroup> bvhWideShared${suffix}: array<vec2u, ${entries * lanes}>;
var<private> bvhWideSpill${suffix}: array<vec2u, BVH_WIDE_SPILL_SIZE>;

fn bvhWidePush${suffix}(at: u32, entry: vec2u) {
    if (at < BVH_WIDE_SHARED_ENTRIES) {
        bvhWideShared${suffix}[at * BVH_WIDE_SHARED_LANES + bvhWideStackLane] = entry;
    } else {
        bvhWideSpill${suffix}[at - BVH_WIDE_SHARED_ENTRIES] = entry;
    }
}

fn bvhWidePop${suffix}(at: u32) -> vec2u {
    if (at < BVH_WIDE_SHARED_ENTRIES) {
        return bvhWideShared${suffix}[at * BVH_WIDE_SHARED_LANES + bvhWideStackLane];
    }
    return bvhWideSpill${suffix}[at - BVH_WIDE_SHARED_ENTRIES];
}
`;
    return `// The workgroup short stack: the first entries per lane in workgroup memory, the rest in a private spill array.
const BVH_WIDE_SHARED_ENTRIES: u32 = ${entries}u;
const BVH_WIDE_SHARED_LANES: u32 = ${lanes}u;
const BVH_WIDE_SPILL_SIZE: u32 = BVH_WIDE_STACK_SIZE - BVH_WIDE_SHARED_ENTRIES;
var<private> bvhWideStackLane: u32;

// A kernel that traces with the workgroup stack calls this first, with its local_invocation_index
// (below ${lanes}): the trace functions cannot see the builtin themselves. Entry i of lane l sits at
// i * lanes + l, so a warp at the same depth touches consecutive words.
fn bvhSetStackLane(localIndex: u32) {
    bvhWideStackLane = localIndex;
}
${region('Closest')}${region('Any')}`;
}

/** The top-level walk of a wide scene: the binary walk's top level, with the mesh walk at each instance. */
function topLevelWalkWgsl(kind: 'closest' | 'any', rayMasks: boolean, anyHitOrder: 'far' | 'near'): string {
    const closest = kind === 'closest';
    const maskParameter = rayMasks ? ', rayMask: u32' : '';
    const maskArgument = rayMasks ? ', rayMask' : '';
    const name = closest ? 'bvhWalkClosest' : 'bvhWalkAny';
    const noHit = closest ? 'result' : 'false';
    const enterInstance = closest
        ? `let meshHit = bvhMeshClosest(instance, setup, ray.tMin, farthest, meshVertexBase${maskArgument});
                if (meshHit.found) {
                    result = BvhHit(meshHit.t, meshHit.instance, meshHit.originalIndex, meshHit.barycentrics, meshHit.frontFace, BVH_NO_MATERIAL, vec3f(0.0));
                    hitRecord = index;
                    hitSlot = meshHit.slot;
                    farthest = meshHit.t;
                }`
        : `if (bvhMeshAny(instance, setup, ray.tMin, farthest, meshVertexBase${maskArgument})) {
                    return true;
                }`;
    return /* wgsl */ `
fn ${name}(ray: BvhRay${maskParameter}) -> ${closest ? 'BvhHit' : 'bool'} {
    ${closest ? `var result = BvhHit(-1.0, 0u, 0u, vec2f(0.0), true, BVH_NO_MATERIAL, vec3f(0.0));
    var hitRecord = 0u;
    var hitSlot = 0u;` : ''}
    let header = bvhTraceTopLevel[0];
    let instanceCount = header.x;
    let topLevelNodeCount = header.y;
    if (topLevelNodeCount == 0u) {
        return ${noHit};
    }
    let firstTopLevelNode = BVH_TOP_HEADER_VEC4S + instanceCount * BVH_TOP_VEC4_PER_RECORD;
    let worldSetup = bvhMakeRaySetup(ray.origin, ray.direction);
    ${closest ? 'var farthest = ray.tMax;' : 'let farthest = ray.tMax;'}

    var stack: array<u32, BVH_STACK_SIZE>;
    var stackSize = 0u;
    var current = bvhMakeReference(0u, 0u);

    loop {
        let count = current >> BVH_REF_COUNT_SHIFT;
        let index = current & BVH_REF_INDEX_MASK;
        if (count == 0u) {
            let node = bvhLoadTopLevelNode(firstTopLevelNode, index);
            let leftEntry = bvhBoxEntry(node.leftMin, node.leftMax, worldSetup, ray.tMin, farthest);
            var rightEntry = BVH_BOX_MISS;
            if (node.rightIndex != BVH_EMPTY_CHILD) {
                rightEntry = bvhBoxEntry(node.rightMin, node.rightMax, worldSetup, ray.tMin, farthest);
            }
            let leftReference = bvhMakeReference(node.leftIndex, node.leftCount);
            let rightReference = bvhMakeReference(node.rightIndex, node.rightCount);
            let hitLeft = leftEntry < BVH_BOX_MISS;
            let hitRight = rightEntry < BVH_BOX_MISS;
            if (hitLeft && hitRight) {
                if (${closest || anyHitOrder === 'near' ? 'leftEntry <= rightEntry' : 'leftEntry > rightEntry'}) {
                    stack[stackSize] = rightReference;
                    current = leftReference;
                } else {
                    stack[stackSize] = leftReference;
                    current = rightReference;
                }
                stackSize += 1u;
                continue;
            }
            if (hitLeft) {
                current = leftReference;
                continue;
            }
            if (hitRight) {
                current = rightReference;
                continue;
            }
        } else {
            // A top-level leaf is one instance: walk its mesh tree to the end in object space.
            let instance = bvhLoadInstance(index);
            ${rayMasks ? 'if ((bvhInstanceMask(instance) & rayMask) != 0u) {' : '{'}
                let setup = bvhMakeRaySetup(bvhTransformPoint(instance, ray.origin), bvhTransformDirection(instance, ray.direction));
                let meshVertexBase = bvhMeshVertexBase(instance.triangleOffset);
                ${enterInstance}
            }
        }
        if (stackSize == 0u) {
            return ${closest ? 'bvhFinishClosest(result, hitRecord, hitSlot)' : 'false'};
        }
        stackSize -= 1u;
        current = stack[stackSize];
    }
}
`;
}

/**
 * Everything a wide trace module adds after the shared parts of wgsl.ts:
 * the node constants, the stack, the mesh walks and the top-level walks
 * (`bvhWalkClosest`, `bvhWalkAny`, with the binary walk's signatures).
 * Any-hit's top level goes far first unless `anyHitOrder` is `'near'`.
 */
export function wideWalkWgsl(options: WideWalkOptions): string {
    const { format, rayMasks, anyHitOrder, stack, walk, decode } = options;
    const cwbvh = format === 'cwbvh8';
    const single = walk === 'single';
    if (single && cwbvh) throw new RangeError("wideWalk 'single' is for nodeFormat 'bvh4'");
    const walks = single
        ? `${bvh4SingleWalkWgsl('closest', rayMasks, anyHitOrder)}${bvh4SingleWalkWgsl('any', rayMasks, anyHitOrder)}`
        : cwbvh
        ? `${cwbvhMeshWalkWgsl('closest', rayMasks, anyHitOrder, stack)}${cwbvhMeshWalkWgsl('any', rayMasks, anyHitOrder, stack)}`
        : `${bvh4MeshWalkWgsl('closest', rayMasks, anyHitOrder)}${bvh4MeshWalkWgsl('any', rayMasks, anyHitOrder)}`;
    return /* wgsl */ `
// ---- Wide mesh nodes (${format}, docs/internals/design/wide-nodes.md) ----------------------

const BVH_WIDE_STACK_SIZE: u32 = ${WIDE_MESH_STACK_SIZE[format]}u;${single ? `
const BVH_SINGLE_STACK_SIZE: u32 = ${WIDE_SINGLE_STACK_SIZE}u;` : ''}
${cwbvh ? `const CWBVH_VEC4_PER_NODE: u32 = ${CWBVH_NODE_WORDS / 4}u;` : `const BVH4_VEC4_PER_NODE: u32 = ${BVH4_NODE_WORDS / 4}u;
const BVH4_CHILD_LEAF: u32 = ${hex(BVH4_CHILD.leafBit)};
const BVH4_CHILD_COUNT_SHIFT: u32 = ${BVH4_CHILD.countShift}u;
const BVH4_CHILD_INDEX_MASK: u32 = ${hex(BVH4_CHILD.indexMask)};`}

// A mesh walk's closest hit (found: false when nothing nearer than the tMax it was given).
struct BvhMeshHit {
    found: bool,
    t: f32,
    slot: u32,             // the triangle's scene slot
    barycentrics: vec2f,
    frontFace: bool,
    instance: u32,         // the hit's instance (a merged mesh's: its source's)
    originalIndex: u32,
}
${WIDE_BOX_WGSL}
${wideEntryWordsWgsl(decode)}
${cwbvh ? CWBVH_HIT_BITS_WGSL : ''}
${single ? '' : meshStackWgsl(format, stack)}
${triangleGroupWgsl('closest', rayMasks)}
${triangleGroupWgsl('any', rayMasks)}
${walks}
${single ? '' : `${topLevelWalkWgsl('closest', rayMasks, anyHitOrder)}
${topLevelWalkWgsl('any', rayMasks, anyHitOrder)}`}`;
}

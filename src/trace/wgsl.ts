/**
 * WGSL for the `rays` query backend: `bvhTraceClosest` and `bvhTraceAny` over
 * the two-level BVH, plus a batch kernel that traces a buffer of rays.
 *
 * The source is a TypeScript string (not a `.wgsl` file) so the layout
 * constants from `layout.ts` are spliced in and the shader can never disagree
 * with the builders about a word offset.
 *
 * The walk, in one paragraph. A single loop and one small stack serve both
 * levels. The top-level tree has one leaf per instance. Reaching an instance
 * leaf pushes a marker, moves the ray into that instance's object space, and
 * continues at the root of the instance's mesh tree. The mesh tree's leaves
 * are triangles. When the marker is popped, the mesh tree is finished and the
 * world-space ray is restored. At every node both child boxes are tested; one
 * is visited and the other pushed (closest-hit: the nearer first; any-hit: the
 * farther first, see `walkWgsl`). Boxes that start beyond the closest hit
 * found so far are skipped.
 *
 * Distances stay in world units because the ray's direction is *not*
 * renormalized after the transform into object space: a point at parameter
 * `t` on the object-space ray is the same point as at `t` on the world ray,
 * whatever scale, shear or mirror the instance has.
 *
 * The public WGSL API (stable from 0.1):
 *
 *     struct BvhRay  { origin: vec3f, tMin: f32, direction: vec3f, tMax: f32 }   // 32 bytes
 *     struct BvhHit  { t: f32, instance: u32, triangle: u32, barycentrics: vec2f, frontFace: bool,
 *                      material: u32, normal: vec3f }
 *     fn bvhTraceClosest(ray: BvhRay) -> BvhHit
 *     fn bvhTraceAny(ray: BvhRay) -> bool
 *     fn bvhTraceClosestMasked(ray: BvhRay, mask: u32) -> BvhHit   // unless rayMasks: false
 *     fn bvhTraceAnyMasked(ray: BvhRay, mask: u32) -> bool         // unless rayMasks: false
 *
 *     // The triangle filter, supplied by the caller (or the default, every triangle blocks):
 *     struct BvhCandidate { instance: u32, triangle: u32, t: f32, barycentrics: vec2f,
 *                           frontFace: bool, query: u32, rayMask: u32 }
 *     fn bvhTrianglePasses(candidate: BvhCandidate) -> bool         // true: the ray passes through
 *
 * Every other name the module declares starts with `bvh`, `Bvh` or `BVH_`;
 * callers should keep clear of those prefixes.
 */

import {
    EMPTY_CHILD_INDEX,
    REF_COUNT_SHIFT,
    REF_INDEX_MASK,
    REF_INSTANCE_MARKER,
    TOP_LEVEL_HEADER_BYTES,
    INSTANCE_RECORD_BYTES,
    BVH_NODE_BYTES,
    TRIANGLE_BYTES,
    TRACE_STACK_SIZE,
    TRIANGLE_FLAG_FILTER,
    TRIANGLE_FLAG_FLIP_FRONT_FACE,
    INSTANCE_FLAG,
    INSTANCE_MASK_SHIFT,
    INSTANCE_MASK_ALL,
    MATERIAL_SHIFT,
    NO_MATERIAL,
    RAY_MASK_ALL,
    BVH_QUERY_CLOSEST,
    BVH_QUERY_ANY,
} from './layout.ts';
import { DEFAULT_TRIANGLE_FILTER_WGSL } from './triangle-filter.ts';

/** Where the trace module's bindings land inside a kernel (structurally the same as ddgi's `BindingSlot`). */
export interface BindingSlot {
    readonly group: number;
    /** Binding number of the module's first entry; the rest follow in order. */
    readonly firstBinding: number;
}

/** A bind group layout entry without its binding number, which the caller's kernel builder assigns. */
export type ModuleBindingEntry = Omit<GPUBindGroupLayoutEntry, 'binding'>;

/** `GPUShaderStage.COMPUTE`. The global does not exist in Node, where the tests import this file. */
const SHADER_STAGE_COMPUTE = 0x4;

/** Bindings the trace module declares, in order. All are read-only storage buffers. */
export const TRACE_MODULE_ENTRIES: readonly ModuleBindingEntry[] = [
    // BVH nodes of every mesh.
    { visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
    // Triangles of every mesh, in tree order.
    { visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
    // Top level: header, instance records, top-level nodes.
    { visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
];

const hex = (value: number): string => `0x${value.toString(16)}u`;

/** The shared structs and constants, written once so the batch kernel reuses them. */
const COMMON_WGSL = /* wgsl */ `
// ---- Contract types (see docs/api/platform.md) ------------------------------

struct BvhRay {
    origin: vec3f,
    tMin: f32,
    direction: vec3f,   // normalized
    tMax: f32,
}

struct BvhHit {
    t: f32,              // distance along the ray; negative means no hit
    instance: u32,       // platform instance index
    triangle: u32,       // triangle index within the instance's mesh
    barycentrics: vec2f, // weights of vertices 1 and 2
    frontFace: bool,     // true on the side the mesh was authored as front (also when mirrored)
    material: u32,       // the instance's material index (a merged hit: its source's), BVH_NO_MATERIAL for none
    normal: vec3f,       // world-space geometric normal of the authored front side, unit length; 0 on a miss
}

// BvhHit.material of a hit on an instance given no material, and of a miss.
const BVH_NO_MATERIAL: u32 = ${hex(NO_MATERIAL)};

// What the triangle filter (bvhTrianglePasses, trace/triangle-filter.ts) is
// told about a candidate hit on a flagged triangle. A struct, so fields can be
// added later without breaking filters. Function-scope only: bool is not
// host-shareable, and this never lives in a buffer.
struct BvhCandidate {
    instance: u32,       // platform instance index (BvhHit.instance)
    triangle: u32,       // original triangle index within the instance's mesh
    t: f32,              // distance along the ray, in world units
    barycentrics: vec2f, // weights of vertices 1 and 2
    frontFace: bool,     // the authored front side, also when mirrored
    query: u32,          // BVH_QUERY_CLOSEST or BVH_QUERY_ANY
    rayMask: u32,        // the trace's mask; all ones without one or with masks off
}

// BvhCandidate.query, the query that asks the filter.
const BVH_QUERY_CLOSEST: u32 = ${BVH_QUERY_CLOSEST}u;
const BVH_QUERY_ANY: u32 = ${BVH_QUERY_ANY}u;

// The ray mask of bvhTraceClosest/bvhTraceAny: every instance is visible.
const BVH_RAY_MASK_ALL: u32 = ${hex(RAY_MASK_ALL)};

// ---- Memory layout (generated from layout.ts) -------------------------------

// One BVH node holds both child boxes. A child with count > 0 is a leaf of
// 'count' primitives starting at 'index'; count == 0 means an inner node at 'index'.
struct BvhNode {
    leftMin: vec3f,  leftIndex: u32,
    leftMax: vec3f,  rightIndex: u32,
    rightMin: vec3f, leftCount: u32,
    rightMax: vec3f, rightCount: u32,
}

// A triangle is its three corners, stored as they are so that a corner shared
// by two triangles has the same bits in both (the watertight test needs that).
struct BvhTriangle {
    vertex0: vec3f, originalIndex: u32,
    vertex1: vec3f, flags: u32,
    vertex2: vec3f, padding: u32,   // 0, or the source instance in a merged mesh
}

// The world-to-object transform is a 3x4 matrix stored as three rows.
struct BvhInstance {
    worldToObjectRow0: vec4f,
    worldToObjectRow1: vec4f,
    worldToObjectRow2: vec4f,
    rootNodeOffset: u32,   // where this mesh's nodes start, in nodes
    triangleOffset: u32,   // where this mesh's triangles start, in triangles
    instanceIndex: u32,    // platform instance index
    flags: u32,            // INSTANCE_FLAG bits 0-3, the mask in 8-15, material + 1 in 16-31 (0: none)
}

// Mesh trees and the top level use the same 64-byte node.
const BVH_VEC4_PER_NODE: u32 = ${BVH_NODE_BYTES / 16}u;

// The top-level buffer is [header][instance records][top-level nodes], and is
// read as vec4u so no integer ever passes through a float register.
const BVH_TOP_VEC4_PER_RECORD: u32 = ${INSTANCE_RECORD_BYTES / 16}u;
const BVH_TOP_HEADER_VEC4S: u32 = ${TOP_LEVEL_HEADER_BYTES / 16}u;

// ---- Traversal constants -----------------------------------------------------

// Stack entries are child references: the index in the low bits, the leaf
// primitive count in the top bits (0 means an inner node).
const BVH_REF_COUNT_SHIFT: u32 = ${REF_COUNT_SHIFT}u;
const BVH_REF_INDEX_MASK: u32 = ${hex(REF_INDEX_MASK)};
// Pushed on entering an instance. Cannot collide with a real reference: leaves hold at most 8 primitives.
const BVH_REF_INSTANCE_MARKER: u32 = ${hex(REF_INSTANCE_MARKER)};
// The right child of a node in a tree with a single primitive.
const BVH_EMPTY_CHILD: u32 = ${hex(EMPTY_CHILD_INDEX)};

// Both levels together must fit; the builder checks depth against this.
const BVH_STACK_SIZE: u32 = ${TRACE_STACK_SIZE}u;

// Returned by the box test for "no hit". Far larger than any scene distance.
const BVH_BOX_MISS: f32 = 3.0e38;

// Direction components smaller than this are treated as this, so 1/direction
// stays finite and 0 * (1/direction) can never produce a NaN.
const BVH_MIN_DIRECTION: f32 = 1.0e-30;

// Rounding in (box - origin) * (1 / direction) can put a ray that touches a
// box a few ulps outside its slab interval. Growing the exit distance by this
// much keeps such rays (Ize, "Robust BVH Ray Traversal"). About 3 f32 steps.
const BVH_BOX_EXIT_PADDING: f32 = 1.0000004;
`;

/**
 * The walk, written once and generated as two functions:
 *
 * - `bvhWalkClosest` keeps the nearest hit and, after each one, skips boxes
 *   that start beyond it;
 * - `bvhWalkAny` returns at the first triangle in [tMin, tMax], so it needs no
 *   hit record at all, and it visits the *farther* child first.
 *
 * Why any-hit goes far first (T1.3, measured on shadow rays toward the sun):
 * a shadow ray starts on a surface, so the nearer child is usually the
 * geometry it starts on, which can't block it. What blocks it (a ceiling, a
 * roof, a wall across the room) lies ahead, in the farther child. Any blocker
 * will do, so there is nothing to gain from finding the nearest one first.
 * On the bench scenes far-first made any-hit much faster than near-first, with
 * every answer unchanged (numbers: PROGRESS, T1.3, and bench/results/trace.csv).
 */
function walkWgsl(kind: 'closest' | 'any', rayMasks: boolean): string {
    const closest = kind === 'closest';
    // With masks compiled out the walk has no mask parameter, no mask test, and
    // hands the filter all ones; it is otherwise the same code.
    const maskParameter = rayMasks ? ', rayMask: u32' : '';
    const candidateMask = rayMasks ? 'rayMask' : 'BVH_RAY_MASK_ALL';
    const query = closest ? 'BVH_QUERY_CLOSEST' : 'BVH_QUERY_ANY';
    const name = closest ? 'bvhWalkClosest' : 'bvhWalkAny';
    const returnType = closest ? 'BvhHit' : 'bool';
    const noHit = closest ? 'result' : 'false';
    // Closest-hit keeps where its hit came from (the instance record and the triangle's
    // slot), and works out the normal and material once, after the walk (bvhFinishClosest).
    const resultDeclaration = closest
        ? `var result = BvhHit(-1.0, 0u, 0u, vec2f(0.0), true, BVH_NO_MATERIAL, vec3f(0.0));
    var hitRecord = 0u;     // instance record of the closest hit
    var hitSlot = 0u;       // its triangle, in the scene's triangle buffer`
        : '';
    const noHitAfterWalk = closest ? 'bvhFinishClosest(result, hitRecord, hitSlot)' : 'false';
    const recordDeclaration = closest ? 'var instanceRecord = 0u;   // the record of the instance being walked' : '';
    const recordAssignment = closest ? 'instanceRecord = index;' : '';
    // Closest-hit narrows the ray after every hit; any-hit never needs to.
    const farDeclaration = closest ? 'var farthest = ray.tMax;   // hits beyond this are not interesting any more' : 'let farthest = ray.tMax;';
    const onTriangleHit = closest
        ? `// The test ran in object space, where the winding is the one the mesh was
                    // authored with, so frontFacing is the authored side even for a mirrored
                    // instance (DECISIONS 2026-11-04); a merged mesh's triangle from a mirrored
                    // source carries the flip bit instead (see hitInstance above).
                    result = BvhHit(triangleHit.t, hitInstance, triangle.originalIndex, triangleHit.barycentrics, frontFace, BVH_NO_MATERIAL, vec3f(0.0));
                    hitRecord = instanceRecord;
                    hitSlot = slot;
                    farthest = triangleHit.t;`
        : 'return true;';
    return /* wgsl */ `
fn ${name}(ray: BvhRay${maskParameter}) -> ${returnType} {
    ${resultDeclaration}
    let header = bvhTraceTopLevel[0];
    let instanceCount = header.x;
    let topLevelNodeCount = header.y;
    if (topLevelNodeCount == 0u) {
        return ${noHit};
    }
    let firstTopLevelNode = BVH_TOP_HEADER_VEC4S + instanceCount * BVH_TOP_VEC4_PER_RECORD;

    let worldSetup = bvhMakeRaySetup(ray.origin, ray.direction);
    var setup = worldSetup;

    var insideInstance = false;
    var instance: BvhInstance;
    ${recordDeclaration}
    ${farDeclaration}

    var stack: array<u32, BVH_STACK_SIZE>;
    var stackSize = 0u;
    var current = bvhMakeReference(0u, 0u);   // the root of the top level

    loop {
        let count = current >> BVH_REF_COUNT_SHIFT;
        let index = current & BVH_REF_INDEX_MASK;

        if (count == 0u) {
            // An inner node: test both children, then go to one and push the other.
            var node: BvhNode;
            if (insideInstance) {
                node = bvhLoadMeshNode(instance.rootNodeOffset + index);
            } else {
                node = bvhLoadTopLevelNode(firstTopLevelNode, index);
            }

            // Only a tree with a single primitive has an empty child, and it is the right one.
            let leftEntry = bvhBoxEntry(node.leftMin, node.leftMax, setup, ray.tMin, farthest);
            var rightEntry = BVH_BOX_MISS;
            if (node.rightIndex != BVH_EMPTY_CHILD) {
                rightEntry = bvhBoxEntry(node.rightMin, node.rightMax, setup, ray.tMin, farthest);
            }
            let leftReference = bvhMakeReference(node.leftIndex, node.leftCount);
            let rightReference = bvhMakeReference(node.rightIndex, node.rightCount);
            let hitLeft = leftEntry < BVH_BOX_MISS;
            let hitRight = rightEntry < BVH_BOX_MISS;

            if (hitLeft && hitRight) {
                // Closest-hit visits the nearer child first, any-hit the farther one (see above).
                if (${closest ? 'leftEntry <= rightEntry' : 'leftEntry > rightEntry'}) {
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
            // Neither child is hit: fall through to pop.
        } else if (!insideInstance) {
            // A top-level leaf is one instance. Move the ray into its object space
            // and start at the root of its mesh tree.
            instance = bvhLoadInstance(index);
            ${recordAssignment}
            ${rayMasks ? `// A masked trace skips an instance whose mask shares no bit with the
            // ray's, before entering its tree (it falls through to pop).
            if ((bvhInstanceMask(instance) & rayMask) != 0u) {` : '{'}
                setup = bvhMakeRaySetup(
                    bvhTransformPoint(instance, ray.origin),
                    bvhTransformDirection(instance, ray.direction),
                );
                stack[stackSize] = BVH_REF_INSTANCE_MARKER;
                stackSize += 1u;
                insideInstance = true;
                current = bvhMakeReference(0u, 0u);
                continue;
            }
        } else {
            // A mesh leaf: 'count' triangles starting at 'index'.
            for (var i = 0u; i < count; i += 1u) {
                let slot = instance.triangleOffset + index + i;
                let triangle = bvhLoadTriangle(slot);
                let triangleHit = bvhIntersectWatertight(triangle, setup, ray.tMin, farthest);
                if (!triangleHit.hit) {
                    continue;
                }
                // A merged mesh (mergeStaticInstances, static-merge.ts) holds several source
                // instances in world space: its triangles name their source instance in the
                // padding word, and the flip bit marks those of a mirrored source, whose
                // world-space winding is the reverse of the authored one. Elsewhere padding
                // is 0 and the bit clear, so this is the instance's index and the side as tested.
                let hitInstance = select(instance.instanceIndex, triangle.padding, (instance.flags & BVH_INSTANCE_FLAG_MERGED) != 0u);
                let frontFace = triangleHit.frontFacing != ((triangle.flags & BVH_TRIANGLE_FLAG_FLIP_FRONT_FACE) != 0u);
                // A flagged triangle asks the filter first (triangle-filter.ts). A ray
                // that passes records nothing and leaves 'farthest' alone, so what
                // lies behind the triangle is still found.
                if ((triangle.flags & BVH_TRIANGLE_FLAG_FILTER) != 0u) {
                    let candidate = BvhCandidate(
                        hitInstance, triangle.originalIndex, triangleHit.t, triangleHit.barycentrics,
                        frontFace, ${query}, ${candidateMask},
                    );
                    if (bvhTrianglePasses(candidate)) {
                        continue;
                    }
                }
                ${onTriangleHit}
            }
        }

        // Nothing left to descend into here: take the next entry off the stack.
        loop {
            if (stackSize == 0u) {
                return ${noHitAfterWalk};
            }
            stackSize -= 1u;
            let entry = stack[stackSize];
            if (entry == BVH_REF_INSTANCE_MARKER) {
                // The instance's tree is finished; go back to the world-space ray.
                insideInstance = false;
                setup = worldSetup;
                continue;
            }
            current = entry;
            break;
        }
    }
    // No return here: the loop above only exits through 'return'.
}
`;
}

/** The old API took the slot `{ group, firstBinding }` as its first argument; in plain JS that would be silently misread. */
function rejectOldPositionalCall(name: string, options: object | null | undefined): void {
    if (options !== null && typeof options === 'object' && ('group' in options || 'firstBinding' in options)) {
        throw new TypeError(
            `${name}: got 'group' or 'firstBinding' in the options. The slot is no longer positional; pass { traceSlot: { group, firstBinding } }.`,
        );
    }
}

/** Options of {@link traceModuleWgsl}. */
export interface TraceModuleOptions {
    /** Where the module's three bindings land. Default: group 0, bindings 0-2. */
    readonly traceSlot?: BindingSlot;
    /**
     * The triangle filter, `fn bvhTrianglePasses(candidate: BvhCandidate) -> bool`,
     * included in the module. Default: {@link DEFAULT_TRIANGLE_FILTER_WGSL}
     * (every triangle blocks). `null` leaves it out, for a kernel that
     * declares `bvhTrianglePasses` itself.
     */
    readonly triangleFilter?: string | null;
    /**
     * Instance masks against ray masks. Default true: the module also has
     * `bvhTraceClosestMasked(ray, mask)` and `bvhTraceAnyMasked(ray, mask)`,
     * and the walk skips an instance whose `(instanceMask & rayMask) == 0`.
     * False compiles masks out: no `*Masked` functions, no mask test in the
     * walk, instance masks are ignored, and the filter sees `rayMask` as all
     * ones. It is the same walk either way, with the mask code left out.
     */
    readonly rayMasks?: boolean;
}

const DEFAULT_TRACE_SLOT: BindingSlot = { group: 0, firstBinding: 0 };

/**
 * The trace module: contract types, layout, the triangle filter (unless
 * `triangleFilter: null`), and `bvhTraceClosest` / `bvhTraceAny` (plus their
 * `*Masked` forms unless `rayMasks: false`). Bindings land at `traceSlot`:
 * nodes, triangles, top level, in that order. The walk calls
 * `bvhTrianglePasses` for flagged triangles (triangle-filter.ts).
 */
export function traceModuleWgsl(options: TraceModuleOptions = {}): string {
    rejectOldPositionalCall('traceModuleWgsl', options);
    const slot = options.traceSlot ?? DEFAULT_TRACE_SLOT;
    const rayMasks = options.rayMasks ?? true;
    const triangleFilter = options.triangleFilter === undefined ? DEFAULT_TRIANGLE_FILTER_WGSL : options.triangleFilter;
    const group = slot.group;
    const nodes = slot.firstBinding;
    const triangles = slot.firstBinding + 1;
    const topLevel = slot.firstBinding + 2;
    return /* wgsl */ `${COMMON_WGSL}
// Nodes and triangles are read as plain vec4s, so every load is one 16-byte
// read whatever the backend makes of a struct with vec3 members.
@group(${group}) @binding(${nodes}) var<storage, read> bvhTraceNodes: array<vec4f>;
@group(${group}) @binding(${triangles}) var<storage, read> bvhTraceTriangles: array<vec4f>;
@group(${group}) @binding(${topLevel}) var<storage, read> bvhTraceTopLevel: array<vec4u>;

fn bvhMakeReference(index: u32, count: u32) -> u32 {
    return index | (count << BVH_REF_COUNT_SHIFT);
}

// ---- Loading mesh nodes and triangles -----------------------------------------

const BVH_VEC4_PER_TRIANGLE: u32 = ${TRIANGLE_BYTES / 16}u;

// Triangle flags bit 0: consult the triangle filter (bvhTrianglePasses;
// trace/triangle-filter.ts).
const BVH_TRIANGLE_FLAG_FILTER: u32 = ${TRIANGLE_FLAG_FILTER}u;
// Triangle flags bit 1: the hit's frontFace is flipped (a merged mesh's triangle
// from a mirrored source instance).
const BVH_TRIANGLE_FLAG_FLIP_FRONT_FACE: u32 = ${TRIANGLE_FLAG_FLIP_FRONT_FACE}u;
// Instance flags bit 3: a merged mesh; the hit's instance is the triangle's padding word.
const BVH_INSTANCE_FLAG_MERGED: u32 = ${INSTANCE_FLAG.merged}u;

fn bvhLoadMeshNode(nodeIndex: u32) -> BvhNode {
    let base = nodeIndex * BVH_VEC4_PER_NODE;
    let a = bvhTraceNodes[base];
    let b = bvhTraceNodes[base + 1u];
    let c = bvhTraceNodes[base + 2u];
    let d = bvhTraceNodes[base + 3u];
    return BvhNode(
        a.xyz, bitcast<u32>(a.w),
        b.xyz, bitcast<u32>(b.w),
        c.xyz, bitcast<u32>(c.w),
        d.xyz, bitcast<u32>(d.w),
    );
}

fn bvhLoadTriangle(triangleIndex: u32) -> BvhTriangle {
    let base = triangleIndex * BVH_VEC4_PER_TRIANGLE;
    let a = bvhTraceTriangles[base];
    let b = bvhTraceTriangles[base + 1u];
    let c = bvhTraceTriangles[base + 2u];
    return BvhTriangle(a.xyz, bitcast<u32>(a.w), b.xyz, bitcast<u32>(b.w), c.xyz, bitcast<u32>(c.w));
}

// ---- Loading from the top-level buffer ---------------------------------------

fn bvhLoadInstance(recordIndex: u32) -> BvhInstance {
    let base = BVH_TOP_HEADER_VEC4S + recordIndex * BVH_TOP_VEC4_PER_RECORD;
    let row0 = bvhTraceTopLevel[base];
    let row1 = bvhTraceTopLevel[base + 1u];
    let row2 = bvhTraceTopLevel[base + 2u];
    let ids = bvhTraceTopLevel[base + 3u];
    return BvhInstance(
        bitcast<vec4f>(row0), bitcast<vec4f>(row1), bitcast<vec4f>(row2),
        ids.x, ids.y, ids.z, ids.w,
    );
}

// 'firstNode' is the vec4 index where the top-level nodes begin.
fn bvhLoadTopLevelNode(firstNode: u32, nodeIndex: u32) -> BvhNode {
    let base = firstNode + nodeIndex * BVH_VEC4_PER_NODE;
    let a = bvhTraceTopLevel[base];
    let b = bvhTraceTopLevel[base + 1u];
    let c = bvhTraceTopLevel[base + 2u];
    let d = bvhTraceTopLevel[base + 3u];
    return BvhNode(
        bitcast<vec3f>(a.xyz), a.w,
        bitcast<vec3f>(b.xyz), b.w,
        bitcast<vec3f>(c.xyz), c.w,
        bitcast<vec3f>(d.xyz), d.w,
    );
}

// ---- Rays ----------------------------------------------------------------------------

// A ray in the space of the level being walked (world space at the top level,
// an instance's object space inside it), with everything the tests reuse
// precomputed once instead of per box or per triangle.
struct BvhRaySetup {
    // The direction is not stored: the tests use only these precomputed forms of
    // it. It is not normalized in object space, so t stays in world units.
    origin: vec3f,
    inverseDirection: vec3f,   // for the box test
    // Watertight triangle test: the permute-and-shear that maps a point (relative
    // to the ray origin) into the space where the ray runs along +z'. A point p
    // becomes (dot(p, shearRowX), dot(p, shearRowY), dot(p, shearRowZ)).
    shearRowX: vec3f,
    shearRowY: vec3f,
    shearRowZ: vec3f,
}

fn bvhSafeInverse(direction: vec3f) -> vec3f {
    let magnitude = max(abs(direction), vec3f(BVH_MIN_DIRECTION));
    let directionSign = select(vec3f(1.0), vec3f(-1.0), direction < vec3f(0.0));
    return 1.0 / (magnitude * directionSign);
}

fn bvhMakeRaySetup(origin: vec3f, direction: vec3f) -> BvhRaySetup {
    // The axis the ray runs along most becomes z'; x' is the axis after it
    // (x -> y -> z -> x) and y' the one after that. Swapping x' and y' when the
    // ray points the negative way keeps triangle winding, so the sign of the
    // determinant still tells front from back.
    //
    // The axes are unit vectors picked with select(), never a vector indexed by
    // a per-ray value: a runtime index into a vector can push it into slow
    // per-thread scratch memory (T0.10 part C measured that on NVIDIA).
    let magnitude = abs(direction);
    let zIsX = magnitude.x > magnitude.y && magnitude.x > magnitude.z;
    let zIsY = !zIsX && magnitude.y > magnitude.z;
    let axisZ = select(select(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0), zIsY), vec3f(1.0, 0.0, 0.0), zIsX);
    var axisX = axisZ.zxy;   // the next axis: x for z, y for x, z for y
    var axisY = axisX.zxy;
    // Each of these dot products has one product with 1 and two with 0, so it is
    // the direction's component exactly, as the indexed version was.
    let directionZ = dot(direction, axisZ);
    if (directionZ < 0.0) {
        let swapped = axisX;
        axisX = axisY;
        axisY = swapped;
    }
    let shearX = dot(direction, axisX) / directionZ;
    let shearY = dot(direction, axisY) / directionZ;
    let shearZ = 1.0 / directionZ;
    // The rows hold only 0, 1 and the shear factors. A row times a corner is then
    // a[x'] - shearX * a[z'] (and so on) with exactly the old rounding: products
    // with 0 and 1 are exact. A corner shared by two triangles is transformed the
    // same way in both, which is what keeps the test watertight.
    return BvhRaySetup(
        origin,
        bvhSafeInverse(direction),
        axisX - shearX * axisZ,
        axisY - shearY * axisZ,
        shearZ * axisZ,
    );
}

fn bvhTransformPoint(instance: BvhInstance, point: vec3f) -> vec3f {
    return vec3f(
        dot(instance.worldToObjectRow0.xyz, point) + instance.worldToObjectRow0.w,
        dot(instance.worldToObjectRow1.xyz, point) + instance.worldToObjectRow1.w,
        dot(instance.worldToObjectRow2.xyz, point) + instance.worldToObjectRow2.w,
    );
}

// Deliberately not normalized afterwards: see the note at the top of this file.
fn bvhTransformDirection(instance: BvhInstance, direction: vec3f) -> vec3f {
    return vec3f(
        dot(instance.worldToObjectRow0.xyz, direction),
        dot(instance.worldToObjectRow1.xyz, direction),
        dot(instance.worldToObjectRow2.xyz, direction),
    );
}

// The material code (material + 1, 0: none) in bits 16-31 of an instance's or
// a merged triangle's flags word, as BvhHit.material.
const BVH_MATERIAL_SHIFT: u32 = ${MATERIAL_SHIFT}u;

fn bvhMaterialFromFlags(flags: u32) -> u32 {
    let code = flags >> BVH_MATERIAL_SHIFT;
    return select(code - 1u, BVH_NO_MATERIAL, code == 0u);
}

// The closest hit's normal and material, worked out once after the walk from
// the instance record and the triangle slot it kept (not per triangle test).
//
// The normal is the authored front side's: cross(v1 - v0, v2 - v0) of the
// stored corners. A normal instance stores them in object space, so the
// normal goes to world space by the inverse transpose of object-to-world,
// which is the transpose of the stored world-to-object rows; for a mirrored
// instance that keeps the authored side (the winding flips, the normal does
// not), as frontFace does. A merged mesh stores world-space corners, so the
// cross is already in world space, and the flip bit (a mirrored source)
// turns it back to the authored side. Either way frontFace is
// dot(normal, direction) < 0. The material is the instance's, or, in a
// merged mesh, the triangle's (its source's).
fn bvhFinishClosest(found: BvhHit, record: u32, slot: u32) -> BvhHit {
    var hit = found;
    if (hit.t < 0.0) {
        return hit;
    }
    let instance = bvhLoadInstance(record);
    let triangle = bvhLoadTriangle(slot);
    let merged = (instance.flags & BVH_INSTANCE_FLAG_MERGED) != 0u;
    var normal = cross(triangle.vertex1 - triangle.vertex0, triangle.vertex2 - triangle.vertex0);
    if (merged) {
        normal = select(normal, -normal, (triangle.flags & BVH_TRIANGLE_FLAG_FLIP_FRONT_FACE) != 0u);
    } else {
        normal = instance.worldToObjectRow0.xyz * normal.x + instance.worldToObjectRow1.xyz * normal.y + instance.worldToObjectRow2.xyz * normal.z;
    }
    let normalLength = length(normal);
    hit.normal = select(vec3f(0.0), normal / normalLength, normalLength > 0.0);
    hit.material = bvhMaterialFromFlags(select(instance.flags, triangle.flags, merged));
    return hit;
}

// ---- Intersection tests ----------------------------------------------------------

// Slab test. Returns the distance at which the ray enters the box, or
// BVH_BOX_MISS when it does not touch the box within [tMin, tMax].
fn bvhBoxEntry(boxMin: vec3f, boxMax: vec3f, setup: BvhRaySetup, tMin: f32, tMax: f32) -> f32 {
    let toMin = (boxMin - setup.origin) * setup.inverseDirection;
    let toMax = (boxMax - setup.origin) * setup.inverseDirection;
    let nearSides = min(toMin, toMax);
    let farSides = max(toMin, toMax);
    let tEnter = max(max(nearSides.x, max(nearSides.y, nearSides.z)), tMin);
    let tExit = min(min(farSides.x, min(farSides.y, farSides.z)) * BVH_BOX_EXIT_PADDING, tMax);
    return select(BVH_BOX_MISS, tEnter, tEnter <= tExit);
}

struct BvhTriangleHit {
    hit: bool,
    t: f32,
    barycentrics: vec2f,   // weights of vertices 1 and 2
    // True when the ray runs against the triangle's winding normal, in the
    // space the test ran in (object space).
    frontFacing: bool,
}

fn bvhNoTriangleHit() -> BvhTriangleHit {
    return BvhTriangleHit(false, 0.0, vec2f(0.0), true);
}

// One edge function of the watertight test: for the edge from 'start' to 'end'
// (corners already sheared into the 2-D space where the ray is the origin), the
// scaled barycentric weight of the corner opposite it, end.x * start.y - end.y * start.x.
//
// Two triangles that share an edge walk it in opposite directions, and their
// values must be exact negatives of each other, or a ray can slip between
// them. Written plainly, that holds only if both products are rounded on their
// own. A shader compiler may fuse one product into the subtraction (an FMA),
// and which product it fuses changes the rounding, so the two triangles can
// round the same edge to the same sign. Dawn compiles for Metal in relaxed math
// mode, which allows that fusing: about 0.1% of the oracle's shared-edge rays
// slipped through on an Apple M5 Pro while NVIDIA and Intel let none through.
//
// So the two corners are put in a fixed order first (by x, then y). Both
// triangles then run the same arithmetic on the same operands and get the same
// bits, whatever the compiler fuses, and the one that walks the edge the other
// way negates the result, which is exact. A shared corner has the same sheared
// bits in every triangle (bvhMakeRaySetup), so both triangles agree on the order.
fn bvhEdgeFunction(start: vec2f, end: vec2f) -> f32 {
    let reversed = start.x > end.x || (start.x == end.x && start.y > end.y);
    let first = select(start, end, reversed);
    let second = select(end, start, reversed);
    // An edge that shears to a single point runs along the ray; exactly, its
    // value is 0. A fused x * y - y * x would give the rounding error instead,
    // with the same sign in both triangles (neither negates, as neither is reversed).
    let ordered = select(second.x * first.y - second.y * first.x, 0.0, all(first == second));
    return select(ordered, -ordered, reversed);
}

// Woop, Benthin and Wald, "Watertight Ray/Triangle Intersection" (2013).
// Triangles store three corners. The three edge functions are computed from
// the same corner bits for both triangles that share an edge, with opposite
// signs (bvhEdgeFunction), so a ray cannot pass between them. A zero counts as
// inside for both, so a ray exactly on an edge hits both neighbours rather than neither.
fn bvhIntersectWatertight(triangle: BvhTriangle, setup: BvhRaySetup, tMin: f32, tMax: f32) -> BvhTriangleHit {
    // Corners relative to the ray origin.
    let a = triangle.vertex0 - setup.origin;
    let b = triangle.vertex1 - setup.origin;
    let c = triangle.vertex2 - setup.origin;

    // Shear and scale into the space where the ray runs along +z' through the origin
    // (see bvhMakeRaySetup for why these are dot products with rows).
    let ax = dot(a, setup.shearRowX);
    let ay = dot(a, setup.shearRowY);
    let bx = dot(b, setup.shearRowX);
    let by = dot(b, setup.shearRowY);
    let cx = dot(c, setup.shearRowX);
    let cy = dot(c, setup.shearRowY);

    // Edge functions: the (scaled) barycentric weights of corners 0, 1 and 2,
    // from the edges opposite them (b to c, c to a, a to b).
    let projectedA = vec2f(ax, ay);
    let projectedB = vec2f(bx, by);
    let projectedC = vec2f(cx, cy);
    let weight0 = bvhEdgeFunction(projectedB, projectedC);
    let weight1 = bvhEdgeFunction(projectedC, projectedA);
    let weight2 = bvhEdgeFunction(projectedA, projectedB);
    if ((weight0 < 0.0 || weight1 < 0.0 || weight2 < 0.0) && (weight0 > 0.0 || weight1 > 0.0 || weight2 > 0.0)) {
        return bvhNoTriangleHit();   // the weights disagree in sign: outside the triangle
    }

    let determinant = weight0 + weight1 + weight2;
    if (determinant == 0.0) {
        return bvhNoTriangleHit();   // edge-on
    }
    // z' of each corner, scaled by 1 / direction along z'.
    let scaledT = weight0 * dot(a, setup.shearRowZ) + weight1 * dot(b, setup.shearRowZ) + weight2 * dot(c, setup.shearRowZ);
    let inverseDeterminant = 1.0 / determinant;
    let t = scaledT * inverseDeterminant;
    if (t < tMin || t > tMax) {
        return bvhNoTriangleHit();
    }
    // With the axis swap in bvhMakeRaySetup the determinant is positive for a front hit.
    return BvhTriangleHit(true, t, vec2f(weight1, weight2) * inverseDeterminant, determinant > 0.0);
}

${rayMasks ? INSTANCE_MASK_WGSL : ''}
${walkWgsl('closest', rayMasks)}
${walkWgsl('any', rayMasks)}
${rayMasks ? MASKED_TRACE_WGSL : UNMASKED_TRACE_WGSL}
${triangleFilter ?? '// bvhTrianglePasses is supplied by the kernel.'}
`;
}

// Only in a module with ray masks.
const INSTANCE_MASK_WGSL = /* wgsl */ `
// The instance mask: 8 bits at bits 8-15 of the instance flags (layout.ts).
const BVH_INSTANCE_MASK_SHIFT: u32 = ${INSTANCE_MASK_SHIFT}u;
const BVH_INSTANCE_MASK_ALL: u32 = ${hex(INSTANCE_MASK_ALL)};

fn bvhInstanceMask(instance: BvhInstance) -> u32 {
    return (instance.flags >> BVH_INSTANCE_MASK_SHIFT) & BVH_INSTANCE_MASK_ALL;
}
`;

const MASKED_TRACE_WGSL = /* wgsl */ `
// Nearest hit; flagged triangles the filter lets through are skipped. Walks with all-ones, so an instance whose mask is 0 is never hit (as in DXR).
fn bvhTraceClosest(ray: BvhRay) -> BvhHit {
    return bvhWalkClosest(ray, BVH_RAY_MASK_ALL);
}

// True if anything blocks the ray (the filter has the same say as above). Stops at the first blocker found.
fn bvhTraceAny(ray: BvhRay) -> bool {
    return bvhWalkAny(ray, BVH_RAY_MASK_ALL);
}

// As bvhTraceClosest, but instances whose mask shares no bit with 'mask' are skipped.
fn bvhTraceClosestMasked(ray: BvhRay, mask: u32) -> BvhHit {
    return bvhWalkClosest(ray, mask);
}

// As bvhTraceAny, but instances whose mask shares no bit with 'mask' are skipped.
fn bvhTraceAnyMasked(ray: BvhRay, mask: u32) -> bool {
    return bvhWalkAny(ray, mask);
}
`;

const UNMASKED_TRACE_WGSL = /* wgsl */ `
// Nearest hit; flagged triangles the filter lets through are skipped. (Ray masks are compiled out.)
fn bvhTraceClosest(ray: BvhRay) -> BvhHit {
    return bvhWalkClosest(ray);
}

// True if anything blocks the ray (the filter has the same say as above). Stops at the first blocker found.
fn bvhTraceAny(ray: BvhRay) -> bool {
    return bvhWalkAny(ray);
}
`;

/**
 * What a batch writes per ray in closest-hit mode, with `hitNormal` (the
 * default): `BvhHitRecord`, 40 bytes. `BvhHit` itself cannot live in a buffer
 * because `bool` is not host-shareable in WGSL, so `frontFace` is stored as 0
 * or 1, and the normal as three f32 (a `vec3f` would pad the record to 48).
 *
 *     word 0 t: f32            word 4-5 barycentrics: vec2f
 *     word 1 instance: u32     word 6   material: u32 (NO_MATERIAL for none)
 *     word 2 triangle: u32     word 7-9 normal x, y, z: f32
 *     word 3 frontFace: u32
 */
export const HIT_RECORD_BYTES = 40;

/**
 * Word positions inside a `BvhHitRecord` (the 40-byte record of
 * {@link HIT_RECORD_BYTES}), for code that reads batch results back.
 * `frontFace` sits before `barycentrics` so the `vec2f` starts on an 8-byte
 * boundary and the record has no padding (`wgsl.test.ts` checks this against
 * the struct in `batchEntryWgsl`).
 */
export const HIT_RECORD_WORD = {
    t: 0,
    instance: 1,
    triangle: 2,
    frontFace: 3,
    barycentrics: 4,
    material: 6,
    normal: 7,
} as const;

/**
 * The compact record of a batch kernel built with `hitNormal: false`:
 * `BvhHitRecordCompact`, 24 bytes, words 0-5 as in {@link HIT_RECORD_WORD}
 * and nothing else. It has no padding to hold the material in (a seventh word
 * would round the record up to 32 bytes, as the `vec2f` aligns it to 8), so
 * it carries no material either: `instance` (the source instance, also for a
 * merged hit) names it in the caller's own per-instance table.
 */
export const HIT_RECORD_COMPACT_BYTES = 24;

/** Word positions inside the compact record ({@link HIT_RECORD_COMPACT_BYTES}). */
export const HIT_RECORD_COMPACT_WORD = {
    t: 0,
    instance: 1,
    triangle: 2,
    frontFace: 3,
    barycentrics: 4,
} as const;

/** Bytes per closest-hit result of a batch kernel built with this `hitNormal` (default true). */
export function hitRecordBytes(hitNormal = true): number {
    return hitNormal ? HIT_RECORD_BYTES : HIT_RECORD_COMPACT_BYTES;
}

/** Workgroup size of the batch kernels. */
export const BATCH_WORKGROUP_SIZE = 64;

/**
 * The batch kernel's own part: its bindings in `group` (rays in, results
 * out, and with `rayMasks` the batch's ray mask as a uniform at binding 2) and
 * the entry point `bvhBatch`, which traces `arrayLength(bvhBatchRays)` rays and
 * writes one result per ray. The trace module comes separately (the
 * platform's kernel builder puts it in group 0). It must be built with the
 * same `rayMasks`.
 *
 * The ray count is the size of the bound `rays` range, so it needs no uniform.
 * The mask does: dispatches with different masks recorded into one command
 * buffer need a uniform buffer each.
 *
 * Closest-hit results are `BvhHitRecord` ({@link HIT_RECORD_BYTES}, 40 bytes,
 * with material and normal) or, with `hitNormal: false`,
 * `BvhHitRecordCompact` ({@link HIT_RECORD_COMPACT_BYTES}, 24 bytes).
 */
export function batchEntryWgsl(mode: 'closest' | 'any', group: number, rayMasks = true, hitNormal = true): string {
    const recordType = hitNormal ? 'BvhHitRecord' : 'BvhHitRecordCompact';
    const recordStruct = hitNormal
        ? `struct BvhHitRecord {
    t: f32,
    instance: u32,
    triangle: u32,
    frontFace: u32,   // 0 or 1
    barycentrics: vec2f,
    material: u32,    // BVH_NO_MATERIAL for none
    normalX: f32,     // three scalars, not a vec3f, so the record has no padding
    normalY: f32,
    normalZ: f32,
}`
        : `struct BvhHitRecordCompact {
    t: f32,
    instance: u32,
    triangle: u32,
    frontFace: u32,   // 0 or 1
    barycentrics: vec2f,
}`;
    const recordValue = hitNormal
        ? 'BvhHitRecord(hit.t, hit.instance, hit.triangle, select(0u, 1u, hit.frontFace), hit.barycentrics, hit.material, hit.normal.x, hit.normal.y, hit.normal.z)'
        : 'BvhHitRecordCompact(hit.t, hit.instance, hit.triangle, select(0u, 1u, hit.frontFace), hit.barycentrics)';
    const resultDeclaration =
        mode === 'closest'
            ? `${recordStruct}

@group(${group}) @binding(1) var<storage, read_write> bvhBatchResults: array<${recordType}>;`
            : `// 1 when the ray is blocked, 0 when it is clear.
@group(${group}) @binding(1) var<storage, read_write> bvhBatchResults: array<u32>;`;

    const body =
        mode === 'closest'
            ? `let hit = ${rayMasks ? 'bvhTraceClosestMasked(bvhBatchRays[rayIndex], bvhBatchParams.rayMask)' : 'bvhTraceClosest(bvhBatchRays[rayIndex])'};
    bvhBatchResults[rayIndex] = ${recordValue};`
            : `bvhBatchResults[rayIndex] = select(0u, 1u, ${rayMasks ? 'bvhTraceAnyMasked(bvhBatchRays[rayIndex], bvhBatchParams.rayMask)' : 'bvhTraceAny(bvhBatchRays[rayIndex])'});`;

    // One ray mask for the whole batch (no per-ray mask buffer).
    const params = rayMasks
        ? `
struct BvhBatchParams {
    rayMask: u32,
}

@group(${group}) @binding(2) var<uniform> bvhBatchParams: BvhBatchParams;`
        : '';

    return /* wgsl */ `
@group(${group}) @binding(0) var<storage, read> bvhBatchRays: array<BvhRay>;
${resultDeclaration}
${params}

@compute @workgroup_size(${BATCH_WORKGROUP_SIZE})
fn bvhBatch(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroupCount: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    // Dispatches are 2D so a very large batch does not exceed 65535 workgroups in x.
    let rayIndex = (workgroup.y * workgroupCount.x + workgroup.x) * ${BATCH_WORKGROUP_SIZE}u + localIndex;
    if (rayIndex >= arrayLength(&bvhBatchRays)) {
        return;
    }
    ${body}
}
`;
}

/** Options of {@link batchKernelWgsl}. */
export interface BatchKernelOptions {
    /** Where the trace module's bindings land. Default: group 0, bindings 0-2. The batch's own bindings are group 1. */
    readonly traceSlot?: BindingSlot;
    /** The triangle filter (`bvhTrianglePasses`). Default: {@link DEFAULT_TRIANGLE_FILTER_WGSL}, every triangle blocks. */
    readonly triangleFilter?: string;
    /**
     * Default true: the batch traces with one ray mask for the whole batch,
     * read from a uniform at group 1, binding 2 ({@link BATCH_PARAMS_BYTES},
     * word {@link BATCH_PARAMS_WORD}.rayMask). False compiles masks out and
     * the batch has no binding 2.
     */
    readonly rayMasks?: boolean;
    /**
     * Default true: closest-hit results are the 40-byte `BvhHitRecord` with
     * the hit's material and normal ({@link HIT_RECORD_BYTES},
     * {@link HIT_RECORD_WORD}). False: the 24-byte `BvhHitRecordCompact`
     * without either ({@link HIT_RECORD_COMPACT_BYTES},
     * {@link HIT_RECORD_COMPACT_WORD}). Any-hit results are one u32 either way.
     */
    readonly hitNormal?: boolean;
}

/** Bytes of the batch kernel's uniform (`BvhBatchParams`, group 1 binding 2, only with ray masks). Padded to 16. */
export const BATCH_PARAMS_BYTES = 16;

/** Word positions inside the batch kernel's uniform. */
export const BATCH_PARAMS_WORD = { rayMask: 0 } as const;

/**
 * A whole batch kernel: the trace module at `traceSlot` with the triangle
 * filter, and the batch's own group 1 (rays, results, and with ray masks the
 * batch's mask). The entry point is `bvhBatch`.
 */
export function batchKernelWgsl(mode: 'closest' | 'any', options: BatchKernelOptions = {}, ...extra: unknown[]): string {
    if (extra.length > 0) {
        throw new TypeError(
            'batchKernelWgsl(mode, options) takes two arguments; the old (mode, slot, filter) form is gone. Pass { traceSlot: { group, firstBinding }, triangleFilter } instead.',
        );
    }
    rejectOldPositionalCall('batchKernelWgsl', options);
    const rayMasks = options.rayMasks ?? true;
    return `${traceModuleWgsl({ traceSlot: options.traceSlot, triangleFilter: options.triangleFilter ?? DEFAULT_TRIANGLE_FILTER_WGSL, rayMasks })}
${batchEntryWgsl(mode, 1, rayMasks, options.hitNormal ?? true)}`;
}

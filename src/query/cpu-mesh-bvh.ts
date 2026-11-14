/**
 * One mesh's tree as a CPU query object (P8, three-mesh-bvh's `MeshBVH`):
 * raycasts, a generic walk over the nodes for shapecasts, refit, triangle
 * lookups and saving, over a tree in the GPU layout (64-byte nodes, 48-byte
 * triangle records in tree order, `gpu-layout.ts`). tinybvh, the GPU
 * builders, `deserializeLayout` and `readLayouts` all write that layout, so
 * any of their trees can be queried here without a conversion.
 *
 * **One ray walk.** {@link CpuMeshBvh.raycast} and
 * {@link CpuMeshBvh.raycastFirst} run `createCpuTracer`, the step-for-step
 * port of the WGSL walk, with its watertight triangle test, over a top level
 * of one identity instance. Side culling is the walk's own (CPU only: the
 * WGSL walk has no side option, so a double-sided raycast is the GPU walk
 * exactly), and a closest hit needs no triangle filter. "Every hit" and
 * per-triangle acceptance go through the walk's triangle filter
 * (`filterEveryTriangle`), on a second tracer that only those raycasts use.
 * There is no second ray walk and no second triangle test: a ray against this
 * tree finds what the GPU walk finds.
 *
 * **One box walk.** {@link CpuMeshBvh.traverse} is the generic walk the
 * shapecast and closest-point queries build on, with three-mesh-bvh's
 * `shapecast` contract (0.9.18): the caller decides which boxes to enter and
 * what to do with each leaf's triangle range. It is not
 * used for rays, because the ray walk must stay the WGSL walk's twin (nearer
 * child first, `t` narrowing, the 64-entry stack).
 *
 * **Triangle indices.** The tree never reorders the geometry. Its triangle
 * records are copies, in tree order, each carrying its original index. A
 * raycast hit's `triangle` is that original index. A *tree position* (what
 * {@link CpuMeshBvh.traverse} hands to `intersectsRange`) becomes the original
 * index through {@link CpuMeshBvh.resolveTriangleIndex}. That is
 * three-mesh-bvh's `indirect` mode, which this tree is always in.
 *
 * **Checked layouts.** The constructor checks the layout once (every child
 * in range, every node reached exactly once from the root, leaf ranges and
 * original indices within the triangles, every tree position in exactly one
 * leaf, parents matching), so a corrupt or crafted file throws there instead
 * of making a walk loop. When some subtree's triangles aren't one run of tree
 * positions (tinybvh's trees: its optimizer moves nodes, not triangles), the
 * constructor puts the triangle records in depth-first order, in place: the
 * same tree, with each subtree one run, as {@link CpuMeshBvh.traverse} needs.
 *
 * **Ownership.** `new CpuMeshBvh(layout)` takes the layout's buffers as its
 * own storage: the reorder above and {@link CpuMeshBvh.refit} write them in
 * place (nodes and triangles together, so the tree stays valid). Pass a copy
 * if the caller keeps using the same buffers elsewhere, for example a GPU
 * upload it will redo from them later.
 *
 * **Indexed trees** (`triangleLeaves: 'indexed'`, docs/internals/design/indexed-leaves.md)
 * are taken with their geometry (`options.source`). `options.corners` says
 * where the queries read corners:
 *
 * - `'records'` (the default): the tree is expanded to records once, in the
 *   constructor (`recordsFromIndexed`). A CPU tree costs 48 bytes per triangle
 *   whichever format it was built in, `layout` is records, and
 *   {@link CpuMeshBvh.serialize} writes records. The corners are a copy, so
 *   editing the source changes nothing until {@link CpuMeshBvh.refit}.
 * - `'positions'`: the tree keeps its 16-byte indexed triangles (a records
 *   tree is converted, `indexedFromRecords`), and every corner is read
 *   through its vertex indices from `source.positions`, by reference, with no
 *   copy: 16 bytes per triangle beyond the geometry. Every query gives the
 *   records path's answer bit for bit (the same `f32` corners, the same
 *   arithmetic). The corners are the source's live values: after editing
 *   them, call {@link CpuMeshBvh.refit} before querying, since only the refit
 *   makes the boxes fit them again. `layout` is indexed and
 *   {@link CpuMeshBvh.serialize} writes an indexed file (load it with the same
 *   geometry as `source`).
 *
 * **Quantized trees** (`nodeFormat: 'quantized'`, docs/internals/design/compressed-nodes.md)
 * are decoded to 64-byte nodes once, in the constructor (`dequantizeNodes`).
 * Each decoded plane is exact (the frame rule), so {@link CpuMeshBvh.traverse}
 * hands `intersectsBounds` plain `f32` boxes that are a little looser than the
 * exact ones and always contain their subtree: `BVH_CONTAINED` stays right.
 * `layout` is then `f32`, {@link CpuMeshBvh.refit} makes its boxes exact again,
 * and {@link CpuMeshBvh.serialize} writes `f32` nodes.
 */

import {
    GPU_LAYOUT_CHILD_STRIDE_WORDS,
    GPU_LAYOUT_EMPTY_CHILD,
    GPU_LAYOUT_INDEXED_ID_MASK,
    GPU_LAYOUT_INDEXED_TRIANGLE_BYTES,
    GPU_LAYOUT_INDEXED_TRIANGLE_WORD,
    GPU_LAYOUT_MAX_LEAF_PRIMITIVES,
    GPU_LAYOUT_NO_PARENT,
    GPU_LAYOUT_NODE_BYTES,
    GPU_LAYOUT_NODE_WORD,
    GPU_LAYOUT_NODE_WORDS,
    GPU_LAYOUT_TRIANGLE_BYTES,
    GPU_LAYOUT_TRIANGLE_WORD,
    type GpuLayoutMeshBvh,
} from '../gpu-layout.ts';
import type { MeshInput } from '../mesh-input.ts';
import { dequantizeNodes } from '../quantized-nodes.ts';
import { BVH_FLOAT_MAX, refitOrderOf } from '../refit-cpu.ts';
import { deserializeLayout, serializeLayout } from '../serialize.ts';
import { indexedFromRecords, recordsFromIndexed } from '../indexed-leaves.ts';
import { createTinyBvhBuilder, type TinyBvhBuilder } from '../tinybvh.ts';
import { createCpuTracer, CPU_TRACE_BACK_SIDE, CPU_TRACE_DOUBLE_SIDE, CPU_TRACE_FRONT_SIDE, type CpuTraceOptions, type CpuTracer } from '../trace/cpu-traversal.ts';
import { buildMeshBvhWithTinyBvh } from '../trace/mesh-bvh-topology.ts';
import { buildTopLevel, IDENTITY_TRANSFORM, type TopLevelBvh } from '../trace/top-level.ts';
import type { TriangleCandidate } from '../trace/triangle-filter.ts';
import { INSIDE_TEST_DIRECTIONS, INSIDE_TEST_MAJORITY, type InsideRule } from '../trace/query-wgsl.ts';
import {
    alignedPackedBoxIntersectsBox,
    BvhBox3,
    boxArraysIntersect,
    ExtendedTriangle,
    invertMatrix,
    matrixElements,
    orientedBoxIntersectsBox,
    packedBoxIntersectsTriangle,
    packOrientedBox,
    PACKED_ORIENTED_BOX_FLOATS,
    transformBoxArray,
    BvhMatrix4,
    OrientedBox,
    sphereIntersectsBox,
    triangleDistanceToTriangle,
    triangleIntersectsTriangle,
    transformPoint,
    BvhVector3,
    type BvhBox3Like,
    type BvhVector3Like,
    type BvhMatrix4Like,
    type BvhQueryTriangle,
    type BvhSphereLike,
    type ExtendedTriangleState,
} from './shapes.ts';

const TRIANGLE_WORDS = GPU_LAYOUT_TRIANGLE_BYTES / 4;
/** Words of an indexed triangle: vertex indices at 0-2, id and flags at 3. */
const INDEXED_WORDS = GPU_LAYOUT_INDEXED_TRIANGLE_BYTES / 4;
const INDEXED_ID_WORD = GPU_LAYOUT_INDEXED_TRIANGLE_WORD.idFlags;
/** {@link CpuMeshBvh.getTriangle}'s and readTriangleInto's corner offsets (they run no user code). */
const CORNER_SCRATCH = new Int32Array(3);
/** Scratch for closestPointToPoint and intersectsBox, which run no user code (so they can't be re-entered mid-call). */
/** The query point (0-2), the point found on a triangle (3-5) and its sliver flag (6): {@link closestOnTriangle}'s numbers. */
const CLOSEST_NEAR = new Float64Array(7);
const CLOSEST_BEST = new BvhVector3();
const BOX_QUERY_OBB = new OrientedBox();
const BOX_QUERY_PACKED = new Float64Array(PACKED_ORIENTED_BOX_FLOATS);
const BOX_QUERY_NODE_BOX = new BvhBox3();
/** Words from a left child's index and count to the right child's (3 to 7, 11 to 15). */
const SIDE_REFERENCE_STRIDE = GPU_LAYOUT_NODE_WORD.rightIndex - GPU_LAYOUT_NODE_WORD.leftIndex;
if (GPU_LAYOUT_NODE_WORD.rightCount - GPU_LAYOUT_NODE_WORD.leftCount !== SIDE_REFERENCE_STRIDE) throw new Error('CpuMeshBvh: the node layout changed');
const VERTEX0 = GPU_LAYOUT_TRIANGLE_WORD.vertex0;
const VERTEX1 = GPU_LAYOUT_TRIANGLE_WORD.vertex1;
const VERTEX2 = GPU_LAYOUT_TRIANGLE_WORD.vertex2;

/** Which faces a raycast hits. The values are three's `FrontSide`, `BackSide` and `DoubleSide`. */
export const BVH_FRONT_SIDE = 0;
export const BVH_BACK_SIDE = 1;
export const BVH_DOUBLE_SIDE = 2;

/** What {@link BvhTraverseCallbacks.intersectsBounds} returns. The values are three-mesh-bvh's. */
export const BVH_NOT_INTERSECTED = 0;
export const BVH_INTERSECTED = 1;
/** The box is wholly inside the shape: its whole subtree's range is reported in one `intersectsRange` call with `contained`, and no more boxes below it are asked about. */
export const BVH_CONTAINED = 2;

/** A point or direction: `[x, y, z]`, or anything with `x`, `y`, `z` (a three `Vector3`). */
export type BvhVec3Like = ArrayLike<number> | { readonly x: number; readonly y: number; readonly z: number };

/** Where {@link CpuMeshBvh.getBoundingBox} writes: six numbers, or a box with `min` and `max` points (a three `Box3`). */
export type BvhBoxTarget = Float32Array | number[] | { readonly min: { x: number; y: number; z: number }; readonly max: { x: number; y: number; z: number } };

/** A ray: `[x, y, z]` arrays, or a three `Ray`. The direction need not be unit length. */
export interface BvhRayLike {
    readonly origin: BvhVec3Like;
    readonly direction: BvhVec3Like;
}

/**
 * Options of {@link CpuMeshBvh.raycast}. A three `Material` passes as one
 * (it has `side`), as three-mesh-bvh's `materialOrSide` does.
 */
export interface BvhRaycastOptions {
    /** {@link BVH_FRONT_SIDE} (default), {@link BVH_BACK_SIDE} or {@link BVH_DOUBLE_SIDE}. */
    readonly side?: number;
    /** Nearest distance along the ray (in the ray's units of length, whatever the direction's length). Default 0. */
    readonly near?: number;
    /** Farthest distance along the ray. Default `Infinity`. */
    readonly far?: number;
}

/** One hit, as {@link CpuMeshBvh.raycast} returns it. */
export interface BvhRaycastHit {
    /** Distance from the ray's origin to the point. */
    readonly distance: number;
    /** The point is `origin + t * direction` (with the direction as given). */
    readonly t: number;
    readonly point: [number, number, number];
    /** The triangle's original index (in the input's index order). */
    readonly triangle: number;
    /** Barycentric weights of corners 1 and 2; corner 0 has `1 - u - v`. */
    readonly u: number;
    readonly v: number;
    /** True when the ray hit the authored front side (counter-clockwise corners, seen from the ray). */
    readonly frontFace: boolean;
}

/** A hit as the walk reports it, before it becomes a {@link BvhRaycastHit}: `t` is in units of the direction given. */
export interface BvhRayIntersection {
    readonly t: number;
    readonly triangle: number;
    readonly u: number;
    readonly v: number;
    readonly frontFace: boolean;
}

/** A ray's crossings with a mesh ({@link CpuMeshBvh.rayCrossings}), as the GPU's `BvhRayCrossings`. */
export interface BvhRayCrossings {
    /** Every triangle the ray crosses at `t >= 0`. */
    readonly count: number;
    /** +1 per triangle left through its front side, -1 per one entered through it. */
    readonly winding: number;
}

/** The inside test's votes ({@link CpuMeshBvh.insideVotes}), each 0 to 3: 2 or more is inside. */
export interface BvhInsideVotes {
    readonly parity: number;
    readonly winding: number;
}

/**
 * The inside test's votes for `point` from a crossings function: one call per
 * ray of `INSIDE_TEST_DIRECTIONS`. Several meshes vote as one solid when the
 * function adds their crossings up.
 */
export function insideVotesOf(point: BvhVec3Like, crossings: (origin: readonly [number, number, number], direction: readonly [number, number, number]) => BvhRayCrossings): BvhInsideVotes {
    const origin = readVec3(point);
    let parity = 0;
    let winding = 0;
    for (const direction of INSIDE_TEST_DIRECTIONS) {
        const ray = crossings(origin, direction);
        parity += ray.count & 1;
        winding += ray.winding !== 0 ? 1 : 0;
    }
    return { parity, winding };
}

/** Decides whether a hit counts, by original triangle and side. Used by {@link CpuMeshBvh.intersectRay}. */
export type BvhTriangleAccept = (triangle: number, frontFace: boolean) => boolean;

/**
 * The callbacks of {@link CpuMeshBvh.traverse}: three-mesh-bvh's `shapecast`
 * callbacks, with its argument order. `box` is
 * `[minX, minY, minZ, maxX, maxY, maxZ]` in a scratch array that is only
 * valid during the call; copy it to keep it.
 *
 * `nodeIndex` names the box: the child slot it is in (`parentNode * 2 + side`,
 * 0 for left, 1 for right), unique per box in this tree and below
 * `2 * nodeCount`. A tree whose root is one leaf reports that leaf as 0.
 *
 * Ranges are **tree positions** (three-mesh-bvh's indirect mode): read them
 * with {@link CpuMeshBvh.getTriangle} or map them to the original triangle with
 * {@link CpuMeshBvh.resolveTriangleIndex}.
 */
export interface BvhTraverseCallbacks {
    /**
     * Whether the walk enters a box. A falsy answer (`false`, `undefined`,
     * {@link BVH_NOT_INTERSECTED}) skips it; {@link BVH_CONTAINED} reports the
     * box's whole subtree as one contained range without asking about the
     * boxes below it; anything else truthy ({@link BVH_INTERSECTED}, `true`)
     * enters it. `isLeaf`: the box holds a triangle range rather than two more
     * boxes. `score` is `boundsTraverseOrder`'s for this box (undefined
     * without one). The root's box is never asked about, as in three-mesh-bvh:
     * the first boxes asked are the root's children, at depth 1.
     */
    intersectsBounds(box: Float32Array, isLeaf: boolean, score: number | undefined, depth: number, nodeIndex: number): unknown;
    /**
     * A range of tree positions, `offset` to `offset + count - 1`: an entered
     * leaf's (`contained` false), or a contained box's whole subtree (one call,
     * `contained` true). A truthy return stops the walk, and
     * {@link CpuMeshBvh.traverse} returns true.
     */
    intersectsRange(offset: number, count: number, contained: boolean, depth: number, nodeIndex: number, box: Float32Array): unknown;
    /** Scores a box, once per child; of two siblings the lower score is visited first (left first on a tie). Without it, left before right. */
    boundsTraverseOrder?(box: Float32Array): number;
}

/**
 * The callbacks of {@link CpuMeshBvh.shapecast}: three-mesh-bvh's `shapecast`
 * callbacks (0.9.18), with its argument order and meaning. `Box` is what the
 * boxes come as (a {@link BvhBox3} here, a three `Box3` in
 * `@pmndrs/bvh/three`), `Triangle` what the triangles come as
 * (an {@link ExtendedTriangle} here, the three-flavoured one there). Both are
 * scratch objects, valid only during the call.
 */
export interface BvhShapecastCallbacks<Triangle = BvhQueryTriangle, Box = BvhBox3Like> {
    /** As {@link BvhTraverseCallbacks.intersectsBounds}, with the box as `Box`. */
    intersectsBounds(box: Box, isLeaf: boolean, score: number | undefined, depth: number, nodeIndex: number): unknown;
    /**
     * Each triangle of an entered leaf or a contained box, in tree order:
     * `triangleIndex` is the original index, `contained` whether a box above
     * it was reported contained. A truthy return stops the cast.
     */
    intersectsTriangle?(triangle: Triangle, triangleIndex: number, contained: boolean, depth: number): unknown;
    /**
     * A range of tree positions, before its triangles are handed to
     * `intersectsTriangle`; a truthy return stops the cast (and skips them).
     * Without `intersectsTriangle` it decides alone. With neither, a cast
     * stops at the first contained box (three-mesh-bvh's default).
     */
    intersectsRange?(offset: number, count: number, contained: boolean, depth: number, nodeIndex: number, box: Box): unknown;
    /** As {@link BvhTraverseCallbacks.boundsTraverseOrder}. */
    boundsTraverseOrder?(box: Box): number;
}

/**
 * The callbacks of {@link CpuMeshBvh.bvhcast}: three-mesh-bvh's (0.9.18).
 * Offsets and indices are **tree positions** in each tree (three-mesh-bvh's
 * indirect mode): map them with each tree's `resolveTriangleIndex`.
 */
export interface BvhBvhcastCallbacks<Triangle = BvhQueryTriangle> {
    /** Two leaves whose boxes overlap (the other's moved into this tree's space). A truthy return stops the cast. */
    intersectsRanges?(offset1: number, count1: number, offset2: number, count2: number, depth1: number, nodeIndex1: number, depth2: number, nodeIndex2: number): unknown;
    /**
     * Every pair of triangles of two such leaves: `triangle1` from this tree,
     * `triangle2` from the other moved by `matrixToLocal`; `i1`, `i2` tree
     * positions. A truthy return stops the cast.
     */
    intersectsTriangles?(triangle1: Triangle, triangle2: Triangle, i1: number, i2: number, depth1: number, nodeIndex1: number, depth2: number, nodeIndex2: number): unknown;
}

/**
 * What {@link CpuMeshBvh.closestPointToPoint} and
 * {@link CpuMeshBvh.closestPointToGeometry} write: three-mesh-bvh's
 * `HitPointInfo`. `Point` is a {@link BvhVector3} here, a three `Vector3` in
 * `@pmndrs/bvh/three`.
 */
export interface BvhClosestPoint<Point extends BvhVector3Like = BvhVector3Like> {
    /** The closest point, in the tree's space (for `target2` of `closestPointToGeometry`, the other geometry's). */
    point: Point;
    distance: number;
    /** The triangle's original index (in the input's index order). */
    faceIndex: number;
}

/** A target to fill: an existing `point` is written in place (any `x`, `y`, `z` object), a missing one is made. */
export type BvhClosestPointTarget<Point extends BvhVector3Like = BvhVector3Like> = Partial<BvhClosestPoint<Point>>;

/** Per-vertex data for {@link CpuMeshBvh.getTriangleHitPointInfo}: `itemSize` numbers per vertex. */
export interface BvhVertexAttribute {
    readonly array: ArrayLike<number>;
    readonly itemSize: number;
}

export interface BvhTriangleHitPointOptions {
    /** The index buffer the tree was built from; default the tree's source (if any). */
    readonly indices?: Uint16Array | Uint32Array | null;
    /** Attributes to interpolate at the point (uv, normal, ...), by name. */
    readonly attributes?: Readonly<Record<string, BvhVertexAttribute>>;
}

/** What {@link CpuMeshBvh.getTriangleHitPointInfo} returns. */
export interface BvhTriangleHitPointInfo {
    /** The original triangle index. */
    readonly triangle: number;
    /** The three vertex indices, or null when the tree doesn't know its index buffer (see the method). */
    readonly vertices: [number, number, number] | null;
    /** The corners as the tree stores them (f32), in authored order. */
    readonly corners: [[number, number, number], [number, number, number], [number, number, number]];
    /** Weights of corners 0, 1, 2 at the point (projected on the triangle's plane); null for a degenerate triangle. */
    readonly barycoord: [number, number, number] | null;
    /** Unit normal of the authored front side; zero for a degenerate triangle. */
    readonly faceNormal: [number, number, number];
    /** Each requested attribute, interpolated with `barycoord` (`itemSize` numbers). Empty when `barycoord` is null. */
    readonly attributes: Record<string, number[]>;
}

/** Options of {@link CpuMeshBvh.build} and {@link CpuMeshBvh.create}. */
export interface CpuMeshBvhBuildOptions {
    /** Leaves with more triangles than this are cut in half until they fit. 1 to 8; default 8. */
    readonly maxLeafPrimitives?: number;
    /**
     * Called with 0 before the build and 1 after it. tinybvh builds in one
     * call into WebAssembly and reports nothing in between.
     */
    readonly onProgress?: (progress: number) => void;
    /** As {@link CpuMeshBvhOptions.corners}. `'positions'` builds indexed triangles directly (tinybvh writes them). Default `'records'`. */
    readonly corners?: CpuMeshBvhCorners;
}

export interface CpuMeshBvhOptions {
    /**
     * The geometry the tree was built from. {@link CpuMeshBvh.refit} rereads
     * its positions, and {@link CpuMeshBvh.getTriangleHitPointInfo} its index.
     * Kept by reference, not copied.
     */
    readonly source?: MeshInput | null;
    /**
     * Where the queries read triangle corners (see the file comment):
     * `'records'` (the default: 48 bytes per triangle, a copy of the corners)
     * or `'positions'` (16 bytes per triangle: the tree's indexed triangles,
     * corners read from `source.positions` by reference; needs `source`).
     * Answers are the same bit for bit; `'positions'` follows the source's
     * edits, so refit after editing it.
     */
    readonly corners?: CpuMeshBvhCorners;
}

/** Where a {@link CpuMeshBvh} reads its triangles' corners: a records copy, or the source's positions. */
export type CpuMeshBvhCorners = 'records' | 'positions';

/** `[x, y, z]` from an array or a three `Vector3`. */
export function readVec3(value: BvhVec3Like): [number, number, number] {
    if ('x' in value) return [value.x, value.y, value.z];
    const array = value as ArrayLike<number>;
    return [array[0], array[1], array[2]];
}

/** Clamps a requested leaf size to the layout's 1-8. */
export function clampLeafPrimitives(size: number): number {
    if (!Number.isFinite(size)) throw new RangeError(`a leaf size must be a finite number, got ${size}`);
    return Math.min(GPU_LAYOUT_MAX_LEAF_PRIMITIVES, Math.max(1, Math.floor(size)));
}

/**
 * Weights of `a`, `b`, `c` at `p` projected on their plane (three's
 * `Triangle.getBarycoord`), in doubles. Null for a degenerate triangle.
 */
export function barycoordOnTriangle(p: ArrayLike<number>, a: ArrayLike<number>, b: ArrayLike<number>, c: ArrayLike<number>): [number, number, number] | null {
    const v0 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const v1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const v2 = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    const dot = (x: number[], y: number[]) => x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    const dot00 = dot(v0, v0);
    const dot01 = dot(v0, v1);
    const dot02 = dot(v0, v2);
    const dot11 = dot(v1, v1);
    const dot12 = dot(v1, v2);
    const denominator = dot00 * dot11 - dot01 * dot01;
    if (denominator === 0) return null;
    const inverse = 1 / denominator;
    const u = (dot11 * dot02 - dot01 * dot12) * inverse;
    const v = (dot00 * dot12 - dot01 * dot02) * inverse;
    return [1 - u - v, v, u];
}

/** Unit normal of `a, b, c` (counter-clockwise front), or zero when degenerate. */
export function triangleNormal(a: ArrayLike<number>, b: ArrayLike<number>, c: ArrayLike<number>): [number, number, number] {
    const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const n: [number, number, number] = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const length = Math.hypot(n[0], n[1], n[2]);
    return length > 0 ? [n[0] / length, n[1] / length, n[2] / length] : [0, 0, 0];
}

/** Interpolates `itemSize` numbers per vertex at vertices `a, b, c` with `weights`. */
export function interpolateVertexAttribute(attribute: BvhVertexAttribute, vertices: readonly [number, number, number], weights: readonly [number, number, number]): number[] {
    const { array, itemSize } = attribute;
    const out = new Array<number>(itemSize).fill(0);
    for (let corner = 0; corner < 3; corner++) {
        const base = vertices[corner] * itemSize;
        for (let i = 0; i < itemSize; i++) out[i] += array[base + i] * weights[corner];
    }
    return out;
}

/**
 * One mesh's tree, for CPU queries. See the file comment.
 *
 * `Hit` is what {@link raycast} returns: {@link BvhRaycastHit} here; an
 * engine adapter (`@pmndrs/bvh/three`'s `BvhBoundsTree`) overrides
 * {@link createHit} to return its engine's type.
 */
export class CpuMeshBvh<Hit = BvhRaycastHit> {
    /** The tree in the GPU layout. Its buffers are this object's own storage: {@link refit} writes them in place. */
    readonly layout: GpuLayoutMeshBvh;
    /** The geometry the tree was built from, if known (kept by reference). */
    protected source: MeshInput | null;
    private readonly nodeFloats: Float32Array;
    private readonly nodeWords: Uint32Array;
    private readonly triangleFloats: Float32Array;
    private readonly triangleWords: Uint32Array;
    /** `corners: 'positions'`: the triangles are indexed and {@link cornerFloats} is the source's positions. */
    private readonly positionCorners: boolean;
    /** Where corners are read: the records' floats, or (`corners: 'positions'`) `source.positions`. */
    private cornerFloats: Float32Array;
    private topLevel: TopLevelBvh;
    /** The walk with no triangle filter: closest-hit raycasts that need no per-triangle decision (side culling is native). */
    private tracer: CpuTracer;
    /** The walk asking {@link passes} about every triangle: all-hit raycasts and ones with an acceptance test. */
    private filterTracer: CpuTracer;
    /** Deepest box (as {@link traverse} reports depth), found when the layout was checked. */
    private readonly treeDepth: number;
    /** Tree position of each original triangle, made on first use. */
    private positions: Uint32Array | null = null;
    /** The current raycast's acceptance test and hit list (null: closest only). */
    private accept: BvhTriangleAccept | null = null;
    private collect: BvhRayIntersection[] | null = null;
    /** {@link refit}'s node order (the shape's, fixed), kept between refits: {@link REVERSE_NODE_ORDER} when every child follows its parent. */
    private refitOrder: Uint32Array | null = null;
    /** The flat queries' walk stacks (slots, and closest point's scores), made on first use. */
    private slotStack: Uint32Array | null = null;
    private scoreStack: Float64Array | null = null;

    constructor(input: GpuLayoutMeshBvh, options: CpuMeshBvhOptions = {}) {
        const corners = options.corners ?? 'records';
        if (corners !== 'records' && corners !== 'positions') throw new RangeError(`CpuMeshBvh: corners must be 'records' or 'positions', got ${String(corners)}`);
        const positionCorners = corners === 'positions';
        // 'records' (the default): indexed trees are queried as records, and the queries, the depth-first reorder and
        // the refit read corners from the records. 'positions': indexed triangles, corners read from the source's
        // positions (docs/internals/design/indexed-leaves.md, "CpuMeshBvh").
        if (input.triangleLeaves === 'indexed' && !options.source) throw new RangeError('CpuMeshBvh: an indexed tree needs its geometry (options.source) for its corners');
        if (positionCorners && !options.source) throw new RangeError("CpuMeshBvh: corners: 'positions' needs the geometry (options.source) to read corners from");
        // Quantized nodes are decoded once (docs/internals/design/compressed-nodes.md, "CpuMeshBvh"): CPU trees
        // stay 64 bytes per node, and every box the queries see contains the exact one.
        const decoded = dequantizeNodes(input);
        const indexed = decoded.triangleLeaves === 'indexed';
        let layout: GpuLayoutMeshBvh;
        if (positionCorners) layout = indexed ? decoded : indexedFromRecords(decoded, options.source!);
        else layout = indexed ? recordsFromIndexed(decoded, options.source!.positions) : decoded;
        if (layout.nodes.byteLength !== layout.nodeCount * GPU_LAYOUT_NODE_BYTES) throw new RangeError(`CpuMeshBvh: ${layout.nodeCount} nodes need ${layout.nodeCount * GPU_LAYOUT_NODE_BYTES} bytes, got ${layout.nodes.byteLength}`);
        const triangleBytes = positionCorners ? GPU_LAYOUT_INDEXED_TRIANGLE_BYTES : GPU_LAYOUT_TRIANGLE_BYTES;
        if (layout.triangles.byteLength !== layout.triangleCount * triangleBytes) {
            throw new RangeError(`CpuMeshBvh: ${layout.triangleCount} triangles need ${layout.triangleCount * triangleBytes} bytes, got ${layout.triangles.byteLength}`);
        }
        if (layout.nodeCount < 1 || layout.triangleCount < 1) throw new RangeError('CpuMeshBvh: the tree is empty');
        if (layout.parents.length !== layout.nodeCount) throw new RangeError(`CpuMeshBvh: ${layout.nodeCount} nodes need ${layout.nodeCount} parents, got ${layout.parents.length}`);
        if (layout.bounds.length < 6) throw new RangeError('CpuMeshBvh: the layout has no bounds');
        this.layout = layout;
        this.source = options.source ?? null;
        this.nodeFloats = new Float32Array(layout.nodes);
        this.nodeWords = new Uint32Array(layout.nodes);
        this.triangleFloats = new Float32Array(layout.triangles);
        this.triangleWords = new Uint32Array(layout.triangles);
        this.positionCorners = positionCorners;
        this.cornerFloats = positionCorners ? options.source!.positions : this.triangleFloats;
        if (positionCorners) checkVertexIndices(this.triangleWords, layout.triangleCount, Math.floor(options.source!.positions.length / 3));
        this.treeDepth = checkTreeShape(this.nodeWords, this.triangleWords, layout.nodeCount, layout.triangleCount, layout.parents, positionCorners);
        this.topLevel = this.makeTopLevel();
        this.tracer = this.makeTracer(false);
        this.filterTracer = this.makeTracer(true);
    }

    /** Builds a tree with tinybvh (`createTinyBvhBuilder()`), on this thread. Keeps `input` as its source. */
    static build(input: MeshInput, builder: TinyBvhBuilder, options: CpuMeshBvhBuildOptions = {}): CpuMeshBvh {
        return new CpuMeshBvh(CpuMeshBvh.buildLayout(input, builder, options), { source: input, corners: options.corners });
    }

    /** As {@link build}, loading tinybvh first. For many builds, keep one builder and use {@link build}. */
    static async create(input: MeshInput, options: CpuMeshBvhBuildOptions & { readonly builder?: TinyBvhBuilder } = {}): Promise<CpuMeshBvh> {
        return CpuMeshBvh.build(input, options.builder ?? (await createTinyBvhBuilder()), options);
    }

    /** Reads bytes from {@link serialize} (or `serializeLayout`). */
    static deserialize<T>(this: new (layout: GpuLayoutMeshBvh, options?: CpuMeshBvhOptions) => T, bytes: ArrayBuffer | ArrayBufferView, options: CpuMeshBvhOptions = {}): T {
        return new this(deserializeLayout(bytes), options);
    }

    /** The tinybvh build behind {@link build}, for subclasses: progress 0, the layout, progress 1. */
    protected static buildLayout(input: MeshInput, builder: TinyBvhBuilder, options: CpuMeshBvhBuildOptions = {}): GpuLayoutMeshBvh {
        options.onProgress?.(0);
        const layout = buildMeshBvhWithTinyBvh(input, builder, { maxLeafPrimitives: options.maxLeafPrimitives, ...(options.corners === 'positions' ? { triangleLeaves: 'indexed' as const } : {}) });
        options.onProgress?.(1);
        return layout;
    }

    get triangleCount(): number {
        return this.layout.triangleCount;
    }

    get nodeCount(): number {
        return this.layout.nodeCount;
    }

    /** The deepest box's depth, as {@link traverse} reports it (three-mesh-bvh's depth: the root's children are at 1). */
    get depth(): number {
        return this.treeDepth;
    }

    /** The tree as bytes (`serializeLayout`'s format). The source geometry isn't included. */
    serialize(): ArrayBuffer {
        return serializeLayout(this.layout);
    }

    /**
     * Box around every triangle, current after {@link refit}: into `target`
     * as `[minX, minY, minZ, maxX, maxY, maxZ]`, or into its `min` and `max`
     * (a three `Box3`). Default a new `Float32Array(6)`.
     */
    getBoundingBox<T extends BvhBoxTarget = Float32Array>(target?: T): T {
        const bounds = this.layout.bounds;
        const out = target ?? (new Float32Array(6) as T);
        if ('min' in out) {
            out.min.x = bounds[0];
            out.min.y = bounds[1];
            out.min.z = bounds[2];
            out.max.x = bounds[3];
            out.max.y = bounds[4];
            out.max.z = bounds[5];
        } else {
            for (let i = 0; i < 6; i++) out[i] = bounds[i];
        }
        return out;
    }

    /**
     * Bytes this object holds: the layout (nodes; triangles, 48-byte records or with `corners: 'positions'` 16 bytes;
     * parents), the one-instance top level and lookups made so far. Not the source geometry.
     */
    estimateMemoryInBytes(): number {
        return this.layout.nodes.byteLength + this.layout.triangles.byteLength + this.layout.parents.byteLength + this.layout.bounds.byteLength + this.topLevel.buffer.byteLength + (this.positions?.byteLength ?? 0);
    }

    /** The original triangle index at a tree position (as {@link traverse} reports them). */
    resolveTriangleIndex(treePosition: number): number {
        if (this.positionCorners) return this.triangleWords[treePosition * INDEXED_WORDS + INDEXED_ID_WORD] & GPU_LAYOUT_INDEXED_ID_MASK;
        return this.triangleWords[treePosition * TRIANGLE_WORDS + GPU_LAYOUT_TRIANGLE_WORD.originalIndex];
    }

    /**
     * Word offsets of the corners at a tree position in {@link cornerFloats},
     * into `out[0..2]`: in the record, or (`corners: 'positions'`) the
     * vertices in the source's positions.
     */
    private cornerWords(treePosition: number, out: Int32Array): void {
        if (this.positionCorners) {
            const words = this.triangleWords;
            const base = treePosition * INDEXED_WORDS;
            out[0] = words[base] * 3;
            out[1] = words[base + 1] * 3;
            out[2] = words[base + 2] * 3;
        } else {
            const base = treePosition * TRIANGLE_WORDS;
            out[0] = base + VERTEX0;
            out[1] = base + VERTEX1;
            out[2] = base + VERTEX2;
        }
    }

    /** The tree position of an original triangle. */
    treePositionOf(triangle: number): number {
        if (!this.positions) {
            const positions = new Uint32Array(this.triangleCount);
            for (let position = 0; position < this.triangleCount; position++) positions[this.resolveTriangleIndex(position)] = position;
            this.positions = positions;
        }
        if (!(triangle >= 0 && triangle < this.triangleCount)) throw new RangeError(`triangle ${triangle} is not in this tree (${this.triangleCount} triangles)`);
        return this.positions[triangle];
    }

    /** The corners of the triangle at a tree position, as the tree stores them: 9 numbers into `target`. */
    getTriangle(treePosition: number, target: Float32Array | number[] = new Float32Array(9)): Float32Array | number[] {
        const words = CORNER_SCRATCH;
        this.cornerWords(treePosition, words);
        const floats = this.cornerFloats;
        for (let corner = 0; corner < 3; corner++) {
            for (let axis = 0; axis < 3; axis++) target[corner * 3 + axis] = floats[words[corner] + axis];
        }
        return target;
    }

    /**
     * The walk under {@link raycast}: every hit (or the closest only) with `t`
     * in `[tMin, tMax]`, in units of `direction` as given. `side` culls
     * natively in the walk ({@link BVH_FRONT_SIDE} front faces only,
     * {@link BVH_BACK_SIDE} back faces only; default {@link BVH_DOUBLE_SIDE},
     * the WGSL walk's behaviour; CPU only). `accept` decides which of the rest
     * count (by original triangle and side), the others are passed through.
     * All hits come in walk order, not sorted. A closest-hit query without
     * `accept` runs the walk with no triangle filter at all.
     */
    intersectRay(
        origin: readonly [number, number, number],
        direction: readonly [number, number, number],
        tMin: number,
        tMax: number,
        firstHitOnly: boolean,
        accept: BvhTriangleAccept | null = null,
        side: number = BVH_DOUBLE_SIDE,
    ): BvhRayIntersection[] {
        const traceOptions = SIDE_TRACE_OPTIONS[side] ?? { side };
        if (firstHitOnly && !accept) {
            const hit = this.tracer.closest({ origin, direction, tMin, tMax }, traceOptions);
            return hit ? [{ t: hit.t, triangle: hit.triangleIndex, u: hit.u, v: hit.v, frontFace: hit.frontFace }] : [];
        }
        const previousAccept = this.accept;
        const previousCollect = this.collect;
        const collect: BvhRayIntersection[] | null = firstHitOnly ? null : [];
        this.accept = accept;
        this.collect = collect;
        try {
            const hit = this.filterTracer.closest({ origin, direction, tMin, tMax }, traceOptions);
            if (collect) return collect;
            return hit ? [{ t: hit.t, triangle: hit.triangleIndex, u: hit.u, v: hit.v, frontFace: hit.frontFace }] : [];
        } finally {
            this.accept = previousAccept;
            this.collect = previousCollect;
        }
    }

    /**
     * Every hit on the faces `side` selects, sorted by distance. The second
     * argument is the side (three's `FrontSide`/`BackSide`/`DoubleSide`
     * values), a {@link BvhRaycastOptions}, or anything with `side` (a three
     * material); `near` and `far` may follow, as in three-mesh-bvh.
     */
    raycast(ray: BvhRayLike, sideOrOptions?: number | BvhRaycastOptions, near?: number, far?: number): Hit[] {
        return this.cast(ray, sideOrOptions, near, far, false);
    }

    /** The closest hit, or null. Arguments as {@link raycast}. */
    raycastFirst(ray: BvhRayLike, sideOrOptions?: number | BvhRaycastOptions, near?: number, far?: number): Hit | null {
        return this.cast(ray, sideOrOptions, near, far, true)[0] ?? null;
    }

    /**
     * Every triangle the ray from `origin` along `direction` crosses at
     * `t >= 0`, both faces: how many, and their signed sum (+1 for each
     * triangle the ray leaves through its front side, -1 for each it enters
     * through it). The CPU twin of the GPU's `bvhRayCrossings`: the same
     * watertight walk, in doubles. A ray exactly on a shared edge counts both
     * triangles, as the walk does.
     */
    rayCrossings(origin: BvhVec3Like, direction: BvhVec3Like): BvhRayCrossings {
        const hits = this.intersectRay(readVec3(origin), readVec3(direction), 0, Infinity, false);
        let winding = 0;
        for (const hit of hits) winding += hit.frontFace ? -1 : 1;
        return { count: hits.length, winding };
    }

    /**
     * The inside test's votes for `point`: of the three rays
     * ({@link INSIDE_TEST_DIRECTIONS}), how many see it inside by parity (an
     * odd crossing count) and how many by winding (a non-zero signed sum).
     * The CPU twin of the GPU's `bvhInsideVotes`. Three full ray walks.
     */
    insideVotes(point: BvhVec3Like): BvhInsideVotes {
        return insideVotesOf(point, (origin, direction) => this.rayCrossings(origin, direction));
    }

    /**
     * Whether `point` is inside the mesh: at least two of three rays vote it
     * inside ({@link insideVotes}). `rule` is `'parity'` (the default; odd
     * crossings, any winding) or `'winding'` (non-zero signed crossings;
     * overlapping closed parts stay inside, triangles must be wound
     * consistently). Exact for a closed mesh away from its surface; on an open
     * mesh, rays through a hole disagree and the vote decides (docs/tracing/sdf.mdx).
     * The CPU twin of the GPU's `bvhIsInside`.
     */
    isPointInside(point: BvhVec3Like, rule: InsideRule = 'parity'): boolean {
        const votes = this.insideVotes(point);
        return (rule === 'winding' ? votes.winding : votes.parity) >= INSIDE_TEST_MAJORITY;
    }

    private cast(ray: BvhRayLike, sideOrOptions: number | BvhRaycastOptions | undefined, nearArgument: number | undefined, farArgument: number | undefined, firstHitOnly: boolean): Hit[] {
        const options: BvhRaycastOptions = typeof sideOrOptions === 'number' ? { side: sideOrOptions } : (sideOrOptions ?? {});
        const side = options.side ?? BVH_FRONT_SIDE;
        const near = nearArgument ?? options.near ?? 0;
        const far = farArgument ?? options.far ?? Infinity;
        const origin = readVec3(ray.origin);
        const direction = readVec3(ray.direction);
        const length = Math.hypot(direction[0], direction[1], direction[2]);
        if (!(length > 0)) return [];
        if (side !== BVH_FRONT_SIDE && side !== BVH_BACK_SIDE && side !== BVH_DOUBLE_SIDE) return [];
        const found = this.intersectRay(origin, direction, Math.max(0, near) / length, far / length, firstHitOnly, null, side);
        if (!firstHitOnly) found.sort((a, b) => a.t - b.t);
        const ray3 = { origin, direction };
        return found.map((hit) => this.createHit(hit, ray3));
    }

    /** Turns one walk hit into the type {@link raycast} returns. Engine adapters override it. */
    protected createHit(hit: BvhRayIntersection, ray: { readonly origin: readonly number[]; readonly direction: readonly number[] }): Hit {
        const { origin, direction } = ray;
        const point: [number, number, number] = [origin[0] + direction[0] * hit.t, origin[1] + direction[1] * hit.t, origin[2] + direction[2] * hit.t];
        const distance = hit.t * Math.hypot(direction[0], direction[1], direction[2]);
        const result: BvhRaycastHit = { distance, t: hit.t, point, triangle: hit.triangle, u: hit.u, v: hit.v, frontFace: hit.frontFace };
        return result as Hit;
    }

    /** The walk's triangle filter: true lets the ray through (not accepted, or collecting every hit). */
    private passes(candidate: TriangleCandidate): boolean {
        if (this.accept && !this.accept(candidate.triangle, candidate.frontFace)) return true;
        if (this.collect) {
            this.collect.push({ t: candidate.t, triangle: candidate.triangle, u: candidate.barycentrics[0], v: candidate.barycentrics[1], frontFace: candidate.frontFace });
            return true;
        }
        return false;
    }

    /**
     * Walks the tree, asking `callbacks` which boxes to enter and handing over
     * triangle ranges, with three-mesh-bvh's `shapecast` contract
     * ({@link BvhTraverseCallbacks}): depth first, the root's box never asked
     * about, each child scored once by `boundsTraverseOrder` and the lower
     * score visited first, a falsy `intersectsBounds` skipping the box, a
     * contained box reported as one range, and a truthy `intersectsRange`
     * stopping the walk. Returns true when it was stopped.
     */
    traverse(callbacks: BvhTraverseCallbacks): boolean {
        const { intersectsBounds, intersectsRange, boundsTraverseOrder: order } = callbacks;
        const words = this.nodeWords;
        const floats = this.nodeFloats;
        const box = new Float32Array(6);
        // A root with only a left child is that child (a one-leaf tree: three's root leaf, reported unasked).
        let root = 0;
        while (words[root * GPU_LAYOUT_NODE_WORDS + GPU_LAYOUT_NODE_WORD.rightIndex] === GPU_LAYOUT_EMPTY_CHILD) {
            const base = root * GPU_LAYOUT_NODE_WORDS;
            const count = words[base + GPU_LAYOUT_NODE_WORD.leftCount];
            if (count > 0) {
                readSlotBox(floats, root * 2, box);
                return Boolean(intersectsRange.call(callbacks, words[base + GPU_LAYOUT_NODE_WORD.leftIndex], count, false, 0, root * 2, box));
            }
            root = words[base + GPU_LAYOUT_NODE_WORD.leftIndex];
        }
        // Stack entries: child slot, depth and score (three-mesh-bvh's recursion, unrolled).
        const slots: number[] = [];
        const depths: number[] = [];
        const scores: (number | undefined)[] = [];
        const limit = 2 * (this.treeDepth + 2);
        const pushChildren = (node: number, depth: number): void => {
            const hasRight = words[node * GPU_LAYOUT_NODE_WORDS + GPU_LAYOUT_NODE_WORD.rightIndex] !== GPU_LAYOUT_EMPTY_CHILD;
            let first = node * 2;
            let second = node * 2 + 1;
            let firstScore: number | undefined;
            let secondScore: number | undefined;
            if (order) {
                readSlotBox(floats, first, box);
                firstScore = order.call(callbacks, box);
                if (hasRight) {
                    readSlotBox(floats, second, box);
                    secondScore = order.call(callbacks, box);
                    if (secondScore < firstScore) {
                        [first, second] = [second, first];
                        [firstScore, secondScore] = [secondScore, firstScore];
                    }
                }
            }
            if (hasRight) {
                slots.push(second);
                depths.push(depth);
                scores.push(secondScore);
            }
            slots.push(first);
            depths.push(depth);
            scores.push(firstScore);
            if (slots.length > limit) throw new RangeError(`CpuMeshBvh.traverse: the stack passed ${limit} entries; the layout is not a tree`);
        };
        pushChildren(root, 1);
        while (slots.length > 0) {
            const slot = slots.pop()!;
            const depth = depths.pop()!;
            const score = scores.pop();
            const base = (slot >> 1) * GPU_LAYOUT_NODE_WORDS;
            const left = (slot & 1) === 0;
            const index = words[base + (left ? GPU_LAYOUT_NODE_WORD.leftIndex : GPU_LAYOUT_NODE_WORD.rightIndex)];
            const count = words[base + (left ? GPU_LAYOUT_NODE_WORD.leftCount : GPU_LAYOUT_NODE_WORD.rightCount)];
            const isLeaf = count > 0;
            readSlotBox(floats, slot, box);
            const answer = intersectsBounds.call(callbacks, box, isLeaf, score, depth, slot);
            if (answer === BVH_CONTAINED) {
                if (this.reportContained(callbacks, slot, depth, box)) return true;
            } else if (answer) {
                if (isLeaf) {
                    readSlotBox(floats, slot, box);
                    if (intersectsRange.call(callbacks, index, count, false, depth, slot, box)) return true;
                } else {
                    pushChildren(index, depth + 1);
                }
            }
        }
        return false;
    }

    /**
     * three-mesh-bvh's node walk (`BVH.traverse(callback)`, 0.9.18), for the
     * debug utilities and the helper: every node in depth-first order, left
     * before right, starting with the root at depth 0 (its box the tree's
     * bounds; a one-leaf tree's root is that leaf). `box` is a new
     * `Float32Array(6)`, `[minX, minY, minZ, maxX, maxY, maxZ]`, per node. A
     * leaf gets its first tree position and triangle count; an inner node
     * gets its split axis (0, 1, 2), which our layout doesn't store: it is
     * taken as the axis along which the children's centres lie furthest
     * apart. A truthy return skips the node's children.
     */
    traverseNodes(callback: (depth: number, isLeaf: boolean, box: Float32Array, offsetOrSplit: number, count: number) => unknown): void {
        const words = this.nodeWords;
        const floats = this.nodeFloats;
        const left = new Float32Array(6);
        const right = new Float32Array(6);
        const splitAxis = (node: number): number => {
            readSlotBox(floats, node * 2, left);
            readSlotBox(floats, node * 2 + 1, right);
            let axis = 0;
            let widest = -1;
            for (let i = 0; i < 3; i++) {
                const gap = Math.abs(left[i] + left[i + 3] - right[i] - right[i + 3]);
                if (gap > widest) {
                    widest = gap;
                    axis = i;
                }
            }
            return axis;
        };
        let root = 0;
        while (words[root * GPU_LAYOUT_NODE_WORDS + GPU_LAYOUT_NODE_WORD.rightIndex] === GPU_LAYOUT_EMPTY_CHILD) {
            const base = root * GPU_LAYOUT_NODE_WORDS;
            const count = words[base + GPU_LAYOUT_NODE_WORD.leftCount];
            if (count > 0) {
                callback(0, true, Float32Array.from(this.layout.bounds.subarray(0, 6)), words[base + GPU_LAYOUT_NODE_WORD.leftIndex], count);
                return;
            }
            root = words[base + GPU_LAYOUT_NODE_WORD.leftIndex];
        }
        if (callback(0, false, Float32Array.from(this.layout.bounds.subarray(0, 6)), splitAxis(root), 0)) return;
        // Child slots and their depths, right pushed first so left comes out first.
        const slots = [root * 2 + 1, root * 2];
        const depths = [1, 1];
        while (slots.length > 0) {
            const slot = slots.pop()!;
            const depth = depths.pop()!;
            const { index, count } = readSlot(words, slot);
            const box = new Float32Array(6);
            readSlotBox(floats, slot, box);
            if (count > 0) {
                callback(depth, true, box, index, count);
                continue;
            }
            if (callback(depth, false, box, splitAxis(index), 0)) continue;
            if (words[index * GPU_LAYOUT_NODE_WORDS + GPU_LAYOUT_NODE_WORD.rightIndex] !== GPU_LAYOUT_EMPTY_CHILD) {
                slots.push(index * 2 + 1);
                depths.push(depth + 1);
            }
            slots.push(index * 2);
            depths.push(depth + 1);
        }
    }

    /**
     * three-mesh-bvh's `writePrimitiveRangeBounds`: the box around the
     * triangles at tree positions `offset` to `offset + count - 1`, into
     * `target` at `targetOffset` as `[minX, minY, minZ, maxX, maxY, maxZ]`.
     * Read from the source geometry when the tree has one (so edits made
     * since the build or the last {@link refit} show, as three-mesh-bvh reads
     * the geometry), else from the corners the tree stores.
     */
    writePrimitiveRangeBounds(offset: number, count: number, target: Float32Array | Float64Array | number[], targetOffset = 0): void {
        for (let axis = 0; axis < 3; axis++) {
            target[targetOffset + axis] = Infinity;
            target[targetOffset + 3 + axis] = -Infinity;
        }
        const source = this.source;
        const corners = new Array<number>(9);
        for (let position = offset, end = offset + count; position < end; position++) {
            if (source) {
                const triangle = this.resolveTriangleIndex(position);
                for (let corner = 0; corner < 3; corner++) {
                    const vertex = source.indices ? source.indices[triangle * 3 + corner] : triangle * 3 + corner;
                    for (let axis = 0; axis < 3; axis++) corners[corner * 3 + axis] = Math.fround(source.positions[vertex * 3 + axis]);
                }
            } else {
                this.getTriangle(position, corners);
            }
            for (let corner = 0; corner < 3; corner++) {
                for (let axis = 0; axis < 3; axis++) {
                    const value = corners[corner * 3 + axis];
                    if (value < target[targetOffset + axis]) target[targetOffset + axis] = value;
                    if (value > target[targetOffset + 3 + axis]) target[targetOffset + 3 + axis] = value;
                }
            }
        }
    }

    /** A contained box: its subtree's range, one run of tree positions (the constructor sees to it), in one call. True when told to stop. */
    private reportContained(callbacks: BvhTraverseCallbacks, slot: number, depth: number, box: Float32Array): boolean {
        const words = this.nodeWords;
        let first = slot;
        for (let at = readSlot(words, first); at.count === 0; at = readSlot(words, first)) first = at.index * 2;
        let last = slot;
        for (let at = readSlot(words, last); at.count === 0; at = readSlot(words, last)) {
            last = words[at.index * GPU_LAYOUT_NODE_WORDS + GPU_LAYOUT_NODE_WORD.rightIndex] === GPU_LAYOUT_EMPTY_CHILD ? at.index * 2 : at.index * 2 + 1;
        }
        const offset = readSlot(words, first).index;
        const end = readSlot(words, last).index + readSlot(words, last).count;
        return Boolean(callbacks.intersectsRange(offset, end - offset, true, depth, slot, box));
    }

    /**
     * three-mesh-bvh's `shapecast` (0.9.18): {@link traverse}, with each
     * leaf's triangles handed to `intersectsTriangle` as an
     * {@link ExtendedTriangle} (a scratch object, reloaded per triangle, with
     * `needsUpdate` set) and the original triangle index. Boxes come as
     * {@link BvhBox3}. Returns true when a callback stopped the cast.
     */
    shapecast(callbacks: BvhShapecastCallbacks): boolean {
        const triangle = trianglePool.pop() ?? new ExtendedTriangle();
        try {
            return this.shapecastWith(callbacks, triangle, new BvhBox3());
        } finally {
            trianglePool.push(triangle);
        }
    }

    /**
     * The one shapecast: {@link shapecast} with the caller's scratch triangle
     * and box (an engine adapter passes its own types; they only need `x`,
     * `y`, `z` corners and `min` / `max` points).
     */
    protected shapecastWith<T extends ExtendedTriangleState, B extends BvhBox3Like>(callbacks: BvhShapecastCallbacks<T, B>, triangle: T, box: B): boolean {
        const { intersectsBounds, intersectsTriangle, intersectsRange, boundsTraverseOrder } = callbacks;
        const fill = (array: Float32Array): B => {
            box.min.x = array[0];
            box.min.y = array[1];
            box.min.z = array[2];
            box.max.x = array[3];
            box.max.y = array[4];
            box.max.z = array[5];
            return box;
        };
        let range: BvhTraverseCallbacks['intersectsRange'];
        if (intersectsTriangle) {
            range = (offset, count, contained, depth, nodeIndex, array) => {
                if (intersectsRange && intersectsRange.call(callbacks, offset, count, contained, depth, nodeIndex, fill(array))) return true;
                for (let position = offset, end = offset + count; position < end; position++) {
                    this.readTriangleInto(position, triangle);
                    if (intersectsTriangle.call(callbacks, triangle, this.resolveTriangleIndex(position), contained, depth)) return true;
                }
                return false;
            };
        } else if (intersectsRange) {
            range = (offset, count, contained, depth, nodeIndex, array) => intersectsRange.call(callbacks, offset, count, contained, depth, nodeIndex, fill(array));
        } else {
            range = (_offset, _count, contained) => contained;
        }
        return this.traverse({
            intersectsBounds: (array, isLeaf, score, depth, nodeIndex) => intersectsBounds.call(callbacks, fill(array), isLeaf, score, depth, nodeIndex),
            intersectsRange: range,
            boundsTraverseOrder: boundsTraverseOrder ? (array) => boundsTraverseOrder.call(callbacks, fill(array)) : undefined,
        });
    }

    /** Loads the corners at a tree position into a triangle (as the tree stores them) and sets its `needsUpdate`. */
    readTriangleInto(treePosition: number, triangle: ExtendedTriangleState): void {
        const floats = this.cornerFloats;
        const words = CORNER_SCRATCH;
        this.cornerWords(treePosition, words);
        const points = triangle.points;
        for (let corner = 0; corner < 3; corner++) {
            const word = words[corner];
            const point = points[corner];
            point.x = floats[word];
            point.y = floats[word + 1];
            point.z = floats[word + 2];
        }
        triangle.needsUpdate = true;
    }

    /**
     * Whether any triangle touches the box `box` placed by `boxToMesh` (into
     * this tree's space; default identity), three-mesh-bvh's `intersectsBox`:
     * an {@link OrientedBox} shapecast. A three `Box3` and `Matrix4` pass.
     */
    intersectsBox(box: BvhBox3Like, boxToMesh?: BvhMatrix4Like | null): boolean {
        // The shapecast above, flattened (it runs no user code): the same boxes tested in the same
        // walk, with the oriented box's numbers packed once and each triangle tested from the
        // corners (records, or the source's positions; packedBoxIntersectsTriangle: the same axes and arithmetic). An
        // axis-aligned box with finite corners tests node boxes on the aligned axes only
        // (alignedPackedBoxIntersectsBox says why that is the same answer).
        const obb = BOX_QUERY_OBB.set(box.min, box.max, boxToMesh);
        const packed = packOrientedBox(obb, BOX_QUERY_PACKED);
        const { min, max } = box;
        const aligned = isIdentityMatrix(boxToMesh) && min.x - min.x + (min.y - min.y) + (min.z - min.z) + (max.x - max.x) + (max.y - max.y) + (max.z - max.z) === 0;
        const words = this.nodeWords;
        const floats = this.nodeFloats;
        let root = 0;
        while (words[root * GPU_LAYOUT_NODE_WORDS + GPU_LAYOUT_NODE_WORD.rightIndex] === GPU_LAYOUT_EMPTY_CHILD) {
            const base = root * GPU_LAYOUT_NODE_WORDS;
            const count = words[base + GPU_LAYOUT_NODE_WORD.leftCount];
            // A one-leaf tree: its triangles are tested without a box test, as traverse reports the root leaf.
            if (count > 0) return this.packedBoxMeetsRange(packed, words[base + GPU_LAYOUT_NODE_WORD.leftIndex], count);
            root = words[base + GPU_LAYOUT_NODE_WORD.leftIndex];
        }
        const stack = this.walkSlots();
        let top = 0;
        stack[top++] = root * 2 + 1;
        stack[top++] = root * 2;
        const nodeBox = aligned ? null : BOX_QUERY_NODE_BOX;
        while (top > 0) {
            const slot = stack[--top];
            const node = slot >> 1;
            const side = slot & 1;
            const boxWord = node * GPU_LAYOUT_NODE_WORDS + side * GPU_LAYOUT_CHILD_STRIDE_WORDS;
            const minX = floats[boxWord], minY = floats[boxWord + 1], minZ = floats[boxWord + 2];
            const maxX = floats[boxWord + 4], maxY = floats[boxWord + 5], maxZ = floats[boxWord + 6];
            if (nodeBox) {
                nodeBox.min.x = minX;
                nodeBox.min.y = minY;
                nodeBox.min.z = minZ;
                nodeBox.max.x = maxX;
                nodeBox.max.y = maxY;
                nodeBox.max.z = maxZ;
                if (!orientedBoxIntersectsBox(obb, nodeBox)) continue;
            } else if (!alignedPackedBoxIntersectsBox(packed, minX, minY, minZ, maxX, maxY, maxZ)) {
                continue;
            }
            const referenceWord = node * GPU_LAYOUT_NODE_WORDS + side * SIDE_REFERENCE_STRIDE;
            const index = words[referenceWord + GPU_LAYOUT_NODE_WORD.leftIndex];
            const count = words[referenceWord + GPU_LAYOUT_NODE_WORD.leftCount];
            if (count > 0) {
                if (this.packedBoxMeetsRange(packed, index, count)) return true;
                continue;
            }
            // A typed array drops a write past its end without a word: never on a checked tree, but say so if it happens.
            if (top + 2 > stack.length) throw walkStackError(stack.length);
            if (words[index * GPU_LAYOUT_NODE_WORDS + GPU_LAYOUT_NODE_WORD.rightIndex] !== GPU_LAYOUT_EMPTY_CHILD) stack[top++] = index * 2 + 1;
            stack[top++] = index * 2;
        }
        return false;
    }

    /** Whether a packed box meets any triangle at tree positions `offset` to `offset + count - 1`. */
    private packedBoxMeetsRange(packed: Float64Array, offset: number, count: number): boolean {
        const t = this.cornerFloats;
        if (this.positionCorners) {
            const words = this.triangleWords;
            for (let position = offset, end = offset + count; position < end; position++) {
                const b = position * INDEXED_WORDS;
                if (packedBoxIntersectsTriangle(packed, t, words[b] * 3, words[b + 1] * 3, words[b + 2] * 3)) return true;
            }
            return false;
        }
        for (let position = offset, end = offset + count; position < end; position++) {
            const b = position * TRIANGLE_WORDS;
            if (packedBoxIntersectsTriangle(packed, t, b + VERTEX0, b + VERTEX1, b + VERTEX2)) return true;
        }
        return false;
    }

    /** This tree's walk stack for the flat queries (they run no user code, so one per tree is enough): `traverse`'s bound. */
    private walkSlots(): Uint32Array {
        return (this.slotStack ??= new Uint32Array(2 * (this.treeDepth + 2) + 2));
    }

    /** Whether any triangle touches the sphere (three-mesh-bvh's `intersectsSphere`). A three `Sphere` passes. */
    intersectsSphere(sphere: BvhSphereLike): boolean {
        return this.coreShapecast({
            intersectsBounds: (bounds) => sphereIntersectsBox(sphere, bounds),
            intersectsTriangle: (triangle) => triangle.intersectsSphere(sphere),
        });
    }

    /**
     * Whether any triangle touches a triangle of `other` placed by
     * `geometryToBvh` (into this tree's space; default identity),
     * three-mesh-bvh's `intersectsGeometry`. `other` is a tree or a mesh
     * (`{ positions, indices }`). Against a tree it is a {@link bvhcast}
     * testing each pair as three-mesh-bvh does (the other's triangle, moved,
     * against this one's), so the answer is the same; three-mesh-bvh 0.9.18
     * shapecasts the other tree from each leaf instead, which measured 1.5x
     * slower here. Against a mesh, every triangle of it is tested against
     * the leaves its bounding box reaches, as three-mesh-bvh does.
     */
    intersectsGeometry(other: CpuMeshBvh<unknown> | MeshInput, geometryToBvh?: BvhMatrix4Like | null): boolean {
        const toBvh = new BvhMatrix4().copy(geometryToBvh ?? IDENTITY_MATRIX);
        if (other instanceof CpuMeshBvh) {
            return CpuMeshBvh.prototype.bvhcast.call(this, other, toBvh, { intersectsTriangles: (triangle1, triangle2) => triangleIntersectsTriangle(triangle2, triangle1) });
        }
        const inverse = new BvhMatrix4();
        invertMatrix(toBvh, inverse);
        const otherBounds = meshBounds(other);
        const obb = new OrientedBox().set({ x: otherBounds[0], y: otherBounds[1], z: otherBounds[2] }, { x: otherBounds[3], y: otherBounds[4], z: otherBounds[5] }, toBvh);
        const thisTriangle = trianglePool.pop() ?? new ExtendedTriangle();
        const otherTriangle = trianglePool.pop() ?? new ExtendedTriangle();
        try {
            // A mesh: this tree's triangles moved into its space, against each of its triangles.
            const { positions, indices } = other;
            const otherCount = indices ? Math.floor(indices.length / 3) : Math.floor(positions.length / 9);
            return this.traverse({
                intersectsBounds: (array) => obb.intersectsBox(boxOfArray(array)),
                intersectsRange: (offset, count) => {
                    for (let position = offset, end = offset + count; position < end; position++) {
                        this.readTriangleInto(position, thisTriangle);
                        for (const point of thisTriangle.points) transformPoint(point, inverse.elements);
                        thisTriangle.needsUpdate = true;
                        for (let triangle = 0; triangle < otherCount; triangle++) {
                            for (let corner = 0; corner < 3; corner++) {
                                const vertex = indices ? indices[triangle * 3 + corner] : triangle * 3 + corner;
                                const point = otherTriangle.points[corner];
                                point.x = positions[vertex * 3];
                                point.y = positions[vertex * 3 + 1];
                                point.z = positions[vertex * 3 + 2];
                            }
                            otherTriangle.needsUpdate = true;
                            if (triangleIntersectsTriangle(thisTriangle, otherTriangle)) return true;
                        }
                    }
                    return false;
                },
            });
        } finally {
            trianglePool.push(thisTriangle, otherTriangle);
        }
    }

    /**
     * three-mesh-bvh's `bvhcast` (0.9.18): walks this tree and `other`
     * together, `other` placed by `matrixToLocal` (its space to this one's),
     * and hands over every pair of leaves whose boxes overlap
     * (`intersectsRanges`) and every pair of their triangles
     * (`intersectsTriangles`, the other's moved into this space). The walk is
     * three-mesh-bvh's, step for step: descend the other tree while both its
     * children overlap, switch trees when one doesn't or a leaf is reached.
     * Offsets and indices are tree positions (indirect mode). Returns true
     * when a callback stopped the cast. Casts may nest.
     */
    bvhcast(other: CpuMeshBvh<unknown>, matrixToLocal: BvhMatrix4Like | null | undefined, callbacks: BvhBvhcastCallbacks): boolean {
        const triangle1 = trianglePool.pop() ?? new ExtendedTriangle();
        const triangle2 = trianglePool.pop() ?? new ExtendedTriangle();
        try {
            return this.bvhcastWith(other, matrixToLocal, callbacks, triangle1, triangle2);
        } finally {
            trianglePool.push(triangle1, triangle2);
        }
    }

    /** The one bvhcast: {@link bvhcast} with the caller's scratch triangles (an engine adapter passes its own type). */
    protected bvhcastWith<T extends ExtendedTriangleState>(other: CpuMeshBvh<unknown>, matrixToLocal: BvhMatrix4Like | null | undefined, callbacks: BvhBvhcastCallbacks<T>, triangle1: T, triangle2: T): boolean {
        if (!(other instanceof CpuMeshBvh)) throw new TypeError('CpuMeshBvh.bvhcast: the other tree must be a CpuMeshBvh');
        const { intersectsRanges, intersectsTriangles } = callbacks;
        if (!intersectsRanges && !intersectsTriangles) throw new TypeError('CpuMeshBvh.bvhcast: pass intersectsRanges or intersectsTriangles');
        const toLocal = matrixElements(matrixToLocal);
        let report: RangesReport;
        if (intersectsTriangles) {
            const pairs: RangesReport = (offset1, count1, offset2, count2, depth1, nodeIndex1, depth2, nodeIndex2) => {
                for (let i2 = offset2, end2 = offset2 + count2; i2 < end2; i2++) {
                    other.readTriangleInto(i2, triangle2);
                    for (const point of triangle2.points) transformPoint(point, toLocal);
                    for (let i1 = offset1, end1 = offset1 + count1; i1 < end1; i1++) {
                        this.readTriangleInto(i1, triangle1);
                        if (intersectsTriangles.call(callbacks, triangle1, triangle2, i1, i2, depth1, nodeIndex1, depth2, nodeIndex2)) return true;
                    }
                }
                return false;
            };
            report = intersectsRanges ? (o1, c1, o2, c2, d1, n1, d2, n2) => Boolean(intersectsRanges.call(callbacks, o1, c1, o2, c2, d1, n1, d2, n2)) || pairs(o1, c1, o2, c2, d1, n1, d2, n2) : pairs;
        } else {
            report = (o1, c1, o2, c2, d1, n1, d2, n2) => Boolean(intersectsRanges!.call(callbacks, o1, c1, o2, c2, d1, n1, d2, n2));
        }
        return dualTraverse(this.walkView(), other.walkView(), toLocal, report);
    }

    /** This tree as {@link dualTraverse} reads it. */
    private walkView(): TreeView {
        return { words: this.nodeWords, floats: this.nodeFloats, bounds: this.layout.bounds };
    }

    /** {@link shapecast} with the core's types, whatever an adapter made of `shapecast`: the derived queries use it. */
    private coreShapecast(callbacks: BvhShapecastCallbacks): boolean {
        return CpuMeshBvh.prototype.shapecast.call(this, callbacks);
    }

    /**
     * three-mesh-bvh's `closestPointToPoint` (0.9.18), step for step: a
     * {@link shapecast} visiting the nearer box first (`boundsTraverseOrder`:
     * squared distance from the point to the box), entering only boxes nearer
     * than the best so far and than `maxThreshold`, and testing each triangle
     * with three's `Triangle.closestPointToPoint`. Stops at the first
     * triangle nearer than `minThreshold` (set both thresholds equal to ask
     * "is anything within this distance?"). Returns `target` with `point`,
     * `distance` and `faceIndex` (the original triangle), or null, leaving
     * `target` alone, when no box is nearer than `maxThreshold`.
     *
     * As in three-mesh-bvh, `maxThreshold` culls boxes, not triangles: a
     * triangle in a box nearer than it may still be returned past it.
     *
     * One departure: on a sliver (the squared sine of its angle at `a` at
     * most 1e-7) the nearer of three's point and the nearest point of its
     * three edges is kept, as the GPU's `bvhClosestPointToTriangle` does,
     * because three's regions cancel there (see `closestOnTriangle`).
     */
    closestPointToPoint(point: BvhVec3Like, target: BvhClosestPointTarget = {}, minThreshold = 0, maxThreshold = Infinity): BvhClosestPoint | null {
        let px: number, py: number, pz: number;
        if ('x' in point) {
            px = point.x;
            py = point.y;
            pz = point.z;
        } else {
            px = point[0];
            py = point[1];
            pz = point[2];
        }
        // The shapecast described above, flattened (it runs no user code): the same walk, the same
        // child order and scores, the same triangles in the same order, three's closest-point
        // arithmetic on the same f32 corners (closestOnTriangle), so the same answer bit for bit.
        // A child is pushed only if its score passes the test it would meet when popped: the best
        // distance only shrinks, so a box that fails now fails then.
        const minThresholdSq = minThreshold * minThreshold;
        const maxThresholdSq = maxThreshold * maxThreshold;
        const words = this.nodeWords;
        const floats = this.nodeFloats;
        const t = this.cornerFloats;
        const triangleWords = this.triangleWords;
        const indexed = this.positionCorners;
        const near = CLOSEST_NEAR;
        near[0] = px;
        near[1] = py;
        near[2] = pz;
        let best = Infinity;
        let bestTriangle = -1;
        let bestX = 0, bestY = 0, bestZ = 0;
        let root = 0;
        let rootLeaf = -1;
        let rootCount = 0;
        while (words[root * GPU_LAYOUT_NODE_WORDS + GPU_LAYOUT_NODE_WORD.rightIndex] === GPU_LAYOUT_EMPTY_CHILD) {
            const base = root * GPU_LAYOUT_NODE_WORDS;
            const count = words[base + GPU_LAYOUT_NODE_WORD.leftCount];
            if (count > 0) {
                // A one-leaf tree: its triangles are tested without a box test, as traverse reports the root leaf.
                rootLeaf = words[base + GPU_LAYOUT_NODE_WORD.leftIndex];
                rootCount = count;
                break;
            }
            root = words[base + GPU_LAYOUT_NODE_WORD.leftIndex];
        }
        const slots = this.walkSlots();
        const scores = this.walkScores();
        let top = 0;
        // `next` is a node whose children go on the stack, `leaf`/`leafCount` a leaf to test.
        let next = rootLeaf < 0 ? root : -1;
        let leaf = rootLeaf;
        let leafCount = rootCount;
        for (;;) {
            if (leaf >= 0) {
                let stop = false;
                for (let position = leaf, end = leaf + leafCount; position < end; position++) {
                    let a: number, b: number, c: number, original: number;
                    if (indexed) {
                        const w = position * INDEXED_WORDS;
                        a = triangleWords[w] * 3;
                        b = triangleWords[w + 1] * 3;
                        c = triangleWords[w + 2] * 3;
                        original = triangleWords[w + INDEXED_ID_WORD] & GPU_LAYOUT_INDEXED_ID_MASK;
                    } else {
                        const w = position * TRIANGLE_WORDS;
                        a = w + VERTEX0;
                        b = w + VERTEX1;
                        c = w + VERTEX2;
                        original = triangleWords[w + GPU_LAYOUT_TRIANGLE_WORD.originalIndex];
                    }
                    closestOnTriangle(t, a, b, c, near);
                    const dx = px - near[3];
                    const dy = py - near[4];
                    const dz = pz - near[5];
                    const distanceSq = dx * dx + dy * dy + dz * dz;
                    if (distanceSq < best) {
                        bestX = near[3];
                        bestY = near[4];
                        bestZ = near[5];
                        best = distanceSq;
                        bestTriangle = original;
                    }
                    if (distanceSq < minThresholdSq) {
                        stop = true;
                        break;
                    }
                }
                if (stop) break;
                leaf = -1;
            } else if (next >= 0) {
                // traverse's pushChildren with boundsTraverseOrder: both scored, the lower visited first (ties: left).
                const nodeWord = next * GPU_LAYOUT_NODE_WORDS;
                if (top + 2 > slots.length) throw walkStackError(slots.length);
                let first = next * 2;
                let firstScore = boxDistanceSq(floats, nodeWord, px, py, pz);
                if (words[nodeWord + GPU_LAYOUT_NODE_WORD.rightIndex] !== GPU_LAYOUT_EMPTY_CHILD) {
                    let second = first + 1;
                    let secondScore = boxDistanceSq(floats, nodeWord + GPU_LAYOUT_CHILD_STRIDE_WORDS, px, py, pz);
                    if (secondScore < firstScore) {
                        const slot = first;
                        first = second;
                        second = slot;
                        const score = firstScore;
                        firstScore = secondScore;
                        secondScore = score;
                    }
                    if (secondScore < best && secondScore < maxThresholdSq) {
                        slots[top] = second;
                        scores[top++] = secondScore;
                    }
                }
                if (firstScore < best && firstScore < maxThresholdSq) {
                    slots[top] = first;
                    scores[top++] = firstScore;
                }
                next = -1;
            } else {
                if (top === 0) break;
                const score = scores[--top];
                if (!(score < best && score < maxThresholdSq)) continue;
                const slot = slots[top];
                const referenceWord = (slot >> 1) * GPU_LAYOUT_NODE_WORDS + (slot & 1) * SIDE_REFERENCE_STRIDE;
                const index = words[referenceWord + GPU_LAYOUT_NODE_WORD.leftIndex];
                const count = words[referenceWord + GPU_LAYOUT_NODE_WORD.leftCount];
                if (count > 0) {
                    leaf = index;
                    leafCount = count;
                } else {
                    next = index;
                }
            }
        }
        if (best === Infinity) return null;
        return this.writeClosestPoint(target, CLOSEST_BEST.set(bestX, bestY, bestZ), Math.sqrt(best), bestTriangle);
    }

    /** Scores beside {@link walkSlots}, for the closest-point walk. */
    private walkScores(): Float64Array {
        return (this.scoreStack ??= new Float64Array(2 * (this.treeDepth + 2) + 2));
    }

    /**
     * three-mesh-bvh's `closestPointToGeometry` (0.9.18), step for step: the
     * closest points between this tree's triangles and `other`'s, `other`
     * placed by `geometryToBvh` (its space to this tree's; default identity).
     * A {@link shapecast} of this tree ordered by the distance from `other`'s
     * box (as an {@link OrientedBox}) to each box; for each leaf, a shapecast
     * of `other`'s tree ordered by the distance from that leaf's box (moved
     * into `other`'s space) to its boxes, or, when `other` is a mesh
     * (`{ positions, indices }`), every triangle of it. Pairs are measured with
     * `ExtendedTriangle.distanceToTriangle`, and the cast stops at a pair
     * nearer than `minThreshold`; boxes no nearer than the best so far or than
     * `maxThreshold` are skipped.
     *
     * `target1` gets the point on this tree, the distance and this tree's
     * triangle; `target2` (pass null to skip it) the point on `other`, in
     * `other`'s space, the distance measured there, and `other`'s triangle.
     * Both triangle indices are original indices (three-mesh-bvh 0.9.18's
     * indirect mode returns tree positions here instead). Returns `target1`, or
     * null with the targets untouched when nothing is nearer than
     * `maxThreshold`. The box scores are taken in each tree's own space, as in
     * three-mesh-bvh, so a `geometryToBvh` with scale skews the culling (not
     * the distances).
     */
    closestPointToGeometry(
        other: CpuMeshBvh<unknown> | MeshInput,
        geometryToBvh?: BvhMatrix4Like | null,
        target1: BvhClosestPointTarget = {},
        target2: BvhClosestPointTarget | null = {},
        minThreshold = 0,
        maxThreshold = Infinity,
    ): BvhClosestPoint | null {
        const toBvh = new BvhMatrix4().copy(geometryToBvh ?? IDENTITY_MATRIX);
        const toOther = new BvhMatrix4();
        invertMatrix(toBvh, toOther);
        const otherTree = other instanceof CpuMeshBvh ? other : null;
        const bounds = otherTree ? otherTree.layout.bounds : meshBounds(other as MeshInput);
        // three-mesh-bvh's obb (the other geometry's box, in this space) and obb2 (this tree's current leaf, in the other's).
        const obb = new OrientedBox().set({ x: bounds[0], y: bounds[1], z: bounds[2] }, { x: bounds[3], y: bounds[4], z: bounds[5] }, toBvh);
        const obb2 = new OrientedBox().set({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 0 }, toOther);
        const toBvhElements = toBvh.elements;
        const triangle = trianglePool.pop() ?? new ExtendedTriangle();
        const triangle2 = trianglePool.pop() ?? new ExtendedTriangle();
        const temp1 = new BvhVector3();
        const temp2 = target2 ? new BvhVector3() : null;
        const best1 = new BvhVector3();
        const best2 = new BvhVector3();
        let closestDistance = Infinity;
        let closestTriangle = -1;
        let closestOtherTriangle = -1;

        /** `other`'s triangle (in `triangle2`, moved into this space) against this leaf's range. True stops the cast. */
        const measure = (offset: number, count: number, otherTriangleIndex: number): boolean => {
            for (let i = offset, end = offset + count; i < end; i++) {
                this.readTriangleInto(i, triangle);
                const distance = triangleDistanceToTriangle(triangle, triangle2, temp1, temp2);
                if (distance < closestDistance) {
                    best1.copy(temp1);
                    if (temp2) best2.copy(temp2);
                    closestDistance = distance;
                    closestTriangle = i;
                    closestOtherTriangle = otherTriangleIndex;
                }
                if (distance < minThreshold) return true;
            }
            return false;
        };
        const moveTriangle2 = (): void => {
            for (const point of triangle2.points) transformPoint(point, toBvhElements);
            triangle2.needsUpdate = true;
        };

        try {
            this.coreShapecast({
                boundsTraverseOrder: (box) => obb.distanceToBox(box),
                intersectsBounds: (_box, _isLeaf, score) => score! < closestDistance && score! < maxThreshold,
                intersectsRange: (offset, count, _contained, _depth, _nodeIndex, box) => {
                    if (otherTree) {
                        // The leaf just entered, in the other tree's space (three-mesh-bvh saves it in intersectsBounds; same box).
                        obb2.min.copy(box.min);
                        obb2.max.copy(box.max);
                        obb2.needsUpdate = true;
                        return otherTree.coreShapecast({
                            boundsTraverseOrder: (otherBox) => obb2.distanceToBox(otherBox),
                            intersectsBounds: (_otherBox, _otherIsLeaf, score) => score! < closestDistance && score! < maxThreshold,
                            intersectsRange: (otherOffset, otherCount) => {
                                for (let i2 = otherOffset, end2 = otherOffset + otherCount; i2 < end2; i2++) {
                                    otherTree.readTriangleInto(i2, triangle2);
                                    moveTriangle2();
                                    if (measure(offset, count, otherTree.resolveTriangleIndex(i2))) return true;
                                }
                                return false;
                            },
                        });
                    }
                    const { positions, indices } = other as MeshInput;
                    const otherCount = indices ? Math.floor(indices.length / 3) : Math.floor(positions.length / 9);
                    for (let i2 = 0; i2 < otherCount; i2++) {
                        for (let corner = 0; corner < 3; corner++) {
                            const vertex = indices ? indices[i2 * 3 + corner] : i2 * 3 + corner;
                            triangle2.points[corner].set(positions[vertex * 3], positions[vertex * 3 + 1], positions[vertex * 3 + 2]);
                        }
                        moveTriangle2();
                        if (measure(offset, count, i2)) return true;
                    }
                    return false;
                },
            });
        } finally {
            trianglePool.push(triangle, triangle2);
        }
        if (closestDistance === Infinity) return null;
        const result = this.writeClosestPoint(target1, best1, closestDistance, this.resolveTriangleIndex(closestTriangle));
        if (target2) {
            // Both points into the other's space; the distance is measured there.
            transformPoint(best2, toOther.elements);
            transformPoint(best1, toOther.elements);
            const distance = Math.hypot(best1.x - best2.x, best1.y - best2.y, best1.z - best2.z);
            this.writeClosestPoint(target2, best2, distance, closestOtherTriangle);
        }
        return result;
    }

    /** Fills a closest-point target: its `point` in place, or a new one from {@link createPoint}. */
    private writeClosestPoint(target: BvhClosestPointTarget, point: BvhVector3Like, distance: number, faceIndex: number): BvhClosestPoint {
        if (target.point) {
            target.point.x = point.x;
            target.point.y = point.y;
            target.point.z = point.z;
        } else {
            target.point = this.createPoint(point.x, point.y, point.z);
        }
        target.distance = distance;
        target.faceIndex = faceIndex;
        return target as BvhClosestPoint;
    }

    /** The point a closest-point query makes when its target has none. Engine adapters override it (a three `Vector3`). */
    protected createPoint(x: number, y: number, z: number): BvhVector3Like {
        return new BvhVector3(x, y, z);
    }

    /**
     * Refits the boxes to moved vertices, keeping the tree's shape (as the GPU
     * refit does, `refitOnCpu`): the triangle records get the new corners,
     * then every box is recomputed bottom up. `input` replaces the source
     * (default: reread the source's positions, which may have been edited in
     * place). The triangle count and the index buffer's meaning must not have
     * changed. A tree refitted after large moves stays correct but gets slower
     * to walk; rebuild it then.
     *
     * With `corners: 'positions'` there are no records to write: the leaf
     * boxes are recomputed from the positions through the tree's own vertex
     * indices (`input`'s index buffer only has to have the same triangle
     * count), and `input`, when given, becomes the source the queries read
     * corners from. The boxes are the records path's bit for bit.
     */
    refit(input?: MeshInput): void {
        const source = input ?? this.source;
        if (!source) throw new Error('CpuMeshBvh.refit: the tree has no source geometry; pass one (refit({ positions, indices }))');
        const { positions, indices } = source;
        const sourceTriangles = indices ? Math.floor(indices.length / 3) : Math.floor(positions.length / 9);
        if (sourceTriangles !== this.triangleCount) throw new RangeError(`CpuMeshBvh.refit: the source has ${sourceTriangles} triangles, the tree ${this.triangleCount}`);
        const vertexCount = Math.floor(positions.length / 3);
        const triangleCount = this.triangleCount;
        if (this.positionCorners) {
            // Validation only reads (a throw leaves the tree as it was): the tree's vertex indices, then the positions.
            checkVertexIndices(this.triangleWords, triangleCount, vertexCount, 'CpuMeshBvh.refit');
            if (!refitSourceLooksValid(positions, null, triangleCount, vertexCount)) this.checkRefitSource(positions, null, vertexCount);
            this.source = source;
            this.cornerFloats = positions;
            this.refitOrder ??= childrenFollowParents(this.nodeWords, this.nodeCount) ? REVERSE_NODE_ORDER : refitOrderOf(this.layout.parents, this.nodeCount);
            refitIndexedTreeFromSource(this.nodeFloats, this.nodeWords, this.nodeCount, this.refitOrder === REVERSE_NODE_ORDER ? null : this.refitOrder, this.triangleWords, positions, this.layout.bounds);
            this.topLevel = this.makeTopLevel();
            this.tracer = this.makeTracer(false);
            this.filterTracer = this.makeTracer(true);
            return;
        }
        // Validation only reads, so a refit that throws leaves the tree as it was. Two straight
        // scans (every index in range, every position finite) pass for nearly every mesh; only
        // when one fails does the per-triangle check run, which throws for a used vertex only (an
        // unused non-finite vertex is allowed, as before). Then one pass writes the corners into
        // the records and refits the boxes (refitTreeFromSource).
        if (!refitSourceLooksValid(positions, indices, triangleCount, vertexCount)) this.checkRefitSource(positions, indices, vertexCount);
        this.source = source;
        this.refitOrder ??= childrenFollowParents(this.nodeWords, this.nodeCount) ? REVERSE_NODE_ORDER : refitOrderOf(this.layout.parents, this.nodeCount);
        refitTreeFromSource(this.nodeFloats, this.nodeWords, this.nodeCount, this.refitOrder === REVERSE_NODE_ORDER ? null : this.refitOrder, this.triangleFloats, this.triangleWords, positions, indices ?? null, this.layout.bounds);
        this.topLevel = this.makeTopLevel();
        this.tracer = this.makeTracer(false);
        this.filterTracer = this.makeTracer(true);
    }

    /** The per-triangle check behind {@link refit}: throws for a triangle past the positions or a used vertex that isn't finite. */
    private checkRefitSource(positions: Float32Array, indices: Uint16Array | Uint32Array | null | undefined, vertexCount: number): void {
        const words = this.triangleWords;
        const triangleCount = this.triangleCount;
        for (let position = 0; position < triangleCount; position++) {
            const triangle = this.resolveTriangleIndex(position);
            // corners: 'positions': the tree's own vertex indices (`indices` is null then).
            const at = position * INDEXED_WORDS;
            const a = this.positionCorners ? words[at] : indices ? indices[triangle * 3] : triangle * 3;
            const b = this.positionCorners ? words[at + 1] : indices ? indices[triangle * 3 + 1] : triangle * 3 + 1;
            const c = this.positionCorners ? words[at + 2] : indices ? indices[triangle * 3 + 2] : triangle * 3 + 2;
            if (a >= vertexCount || b >= vertexCount || c >= vertexCount) {
                const vertex = a >= vertexCount ? a : b >= vertexCount ? b : c;
                throw new RangeError(`CpuMeshBvh.refit: triangle ${triangle} uses vertex ${vertex}, past the ${vertexCount} positions`);
            }
            const a3 = a * 3;
            const b3 = b * 3;
            const c3 = c * 3;
            const ax = positions[a3], ay = positions[a3 + 1], az = positions[a3 + 2];
            const bx = positions[b3], by = positions[b3 + 1], bz = positions[b3 + 2];
            const cx = positions[c3], cy = positions[c3 + 1], cz = positions[c3 + 2];
            // v - v is NaN for NaN and both infinities, 0 otherwise: one test for the nine values.
            if (ax - ax + (ay - ay) + (az - az) + (bx - bx) + (by - by) + (bz - bz) + (cx - cx) + (cy - cy) + (cz - cz) !== 0) {
                const vertex = !(Number.isFinite(ax) && Number.isFinite(ay) && Number.isFinite(az)) ? a : !(Number.isFinite(bx) && Number.isFinite(by) && Number.isFinite(bz)) ? b : c;
                throw new RangeError(`CpuMeshBvh.refit: vertex ${vertex} has a non-finite position`);
            }
        }
    }

    /**
     * Where `point` lies on an original triangle: barycentrics, the corners,
     * the face normal, and any vertex attributes given, interpolated. The
     * vertex indices come from `options.indices`, else the source's index
     * buffer, else (a source without one) `3 * triangle + corner`. A tree
     * without a source (deserialized, or made from a layout) has `vertices`
     * null unless `options.indices` is given, and then can't interpolate
     * attributes.
     */
    getTriangleHitPointInfo(point: BvhVec3Like, triangle: number, options: BvhTriangleHitPointOptions = {}): BvhTriangleHitPointInfo {
        const corners9 = this.getTriangle(this.treePositionOf(triangle), new Array<number>(9));
        const corners: BvhTriangleHitPointInfo['corners'] = [
            [corners9[0], corners9[1], corners9[2]],
            [corners9[3], corners9[4], corners9[5]],
            [corners9[6], corners9[7], corners9[8]],
        ];
        const indices = options.indices !== undefined ? options.indices : (this.source?.indices ?? null);
        let vertices: [number, number, number] | null = null;
        if (indices) vertices = [indices[triangle * 3], indices[triangle * 3 + 1], indices[triangle * 3 + 2]];
        else if (this.source || options.indices === null) vertices = [triangle * 3, triangle * 3 + 1, triangle * 3 + 2];
        const barycoord = barycoordOnTriangle(readVec3(point), corners[0], corners[1], corners[2]);
        const attributes: Record<string, number[]> = {};
        const requested = options.attributes ?? {};
        if (barycoord) {
            for (const name of Object.keys(requested)) {
                if (!vertices) throw new Error(`CpuMeshBvh.getTriangleHitPointInfo: "${name}" needs the vertex indices; this tree has no source, so pass options.indices (null for an unindexed mesh)`);
                attributes[name] = interpolateVertexAttribute(requested[name], vertices, barycoord);
            }
        }
        return { triangle, vertices, corners, barycoord, faceNormal: triangleNormal(corners[0], corners[1], corners[2]), attributes };
    }

    private makeTopLevel(): TopLevelBvh {
        return buildTopLevel([{ instanceIndex: 0, nodeOffset: 0, triangleOffset: 0, meshBounds: this.layout.bounds, objectToWorld: IDENTITY_TRANSFORM }]);
    }

    private makeTracer(filtered: boolean): CpuTracer {
        // corners: 'positions': the walk's indexed loader over the tree's own triangles (no mesh header: vertex base 0)
        // and the source's positions, packed: the same f32 corners as the records, so the same hits bit for bit.
        const scene = this.positionCorners
            ? { nodes: this.layout.nodes, triangles: this.layout.triangles, topLevel: this.topLevel, triangleLeaves: 'indexed' as const, positions: this.cornerFloats, positionLayout: 'packed' as const }
            : { nodes: this.layout.nodes, triangles: this.layout.triangles, topLevel: this.topLevel };
        const indexed = { indexedWithoutHeaders: this.positionCorners };
        return filtered ? createCpuTracer(scene, { ...indexed, triangleFilter: (candidate) => this.passes(candidate), filterEveryTriangle: true }) : createCpuTracer(scene, indexed);
    }
}

/** The walk's per-trace options for each side, made once (`side` indexes it: three's values). */
const SIDE_TRACE_OPTIONS: readonly CpuTraceOptions[] = [{ side: CPU_TRACE_FRONT_SIDE }, { side: CPU_TRACE_BACK_SIDE }, { side: CPU_TRACE_DOUBLE_SIDE }];

/** Child slot `slot`'s reference: node or first tree position, and triangle count (0 for a node). */
function readSlot(words: Uint32Array, slot: number): { index: number; count: number } {
    const base = (slot >> 1) * GPU_LAYOUT_NODE_WORDS;
    const left = (slot & 1) === 0;
    return {
        index: words[base + (left ? GPU_LAYOUT_NODE_WORD.leftIndex : GPU_LAYOUT_NODE_WORD.rightIndex)],
        count: words[base + (left ? GPU_LAYOUT_NODE_WORD.leftCount : GPU_LAYOUT_NODE_WORD.rightCount)],
    };
}

/**
 * Checks once that the nodes form one tree from node 0 (see the file comment)
 * and returns its depth (as `traverse` reports it). Throws a `RangeError`
 * naming the first problem. When the leaves, left subtree first, don't come in
 * tree-position order, rewrites the triangle records and leaf indices in that
 * order, so each subtree's triangles are one run.
 */
function checkTreeShape(words: Uint32Array, triangleWords: Uint32Array, nodeCount: number, triangleCount: number, parents: Uint32Array, indexed: boolean): number {
    const fail = (problem: string): never => {
        throw new RangeError(`CpuMeshBvh: the layout is not a valid tree: ${problem}`);
    };
    // Records (48 bytes) or, with corners: 'positions', indexed triangles (16): the reorder moves whole triangles.
    const stride = indexed ? INDEXED_WORDS : TRIANGLE_WORDS;
    for (let position = 0; position < triangleCount; position++) {
        const original = indexed
            ? triangleWords[position * INDEXED_WORDS + INDEXED_ID_WORD] & GPU_LAYOUT_INDEXED_ID_MASK
            : triangleWords[position * TRIANGLE_WORDS + GPU_LAYOUT_TRIANGLE_WORD.originalIndex];
        if (original >= triangleCount) fail(`the triangle at tree position ${position} has original index ${original}, past the ${triangleCount} triangles`);
    }
    if (parents[0] !== GPU_LAYOUT_NO_PARENT) fail(`the root has parent ${parents[0]}`);
    if (words[GPU_LAYOUT_NODE_WORD.leftIndex] === GPU_LAYOUT_EMPTY_CHILD) fail('the root has no left child');
    const visited = new Uint8Array(nodeCount);
    const covered = new Uint8Array(triangleCount);
    visited[0] = 1;
    let reached = 1;
    let depth = 0;
    let next = 0;
    let inOrder = true;
    /** Leaf slots, left subtree first. */
    const leaves: number[] = [];
    // Child slots, left subtree before right, and their depths.
    const slots: number[] = [1, 0];
    const depths: number[] = [1, 1];
    while (slots.length > 0) {
        const slot = slots.pop()!;
        const slotDepth = depths.pop()!;
        const node = slot >> 1;
        const { index, count } = readSlot(words, slot);
        if (index === GPU_LAYOUT_EMPTY_CHILD) {
            if ((slot & 1) === 0) fail(`node ${node} has no left child`);
            continue;
        }
        if (slotDepth > depth) depth = slotDepth;
        if (count > 0) {
            if (count > GPU_LAYOUT_MAX_LEAF_PRIMITIVES) fail(`node ${node}'s leaf holds ${count} triangles, more than ${GPU_LAYOUT_MAX_LEAF_PRIMITIVES}`);
            if (index + count > triangleCount) fail(`node ${node}'s leaf covers tree positions ${index} to ${index + count - 1}, past the ${triangleCount} triangles`);
            for (let position = index; position < index + count; position++) {
                if (covered[position]) fail(`tree position ${position} is in two leaves`);
                covered[position] = 1;
            }
            if (index !== next) inOrder = false;
            next = index + count;
            leaves.push(slot);
            continue;
        }
        if (index >= nodeCount) fail(`node ${node} has child ${index}, past the ${nodeCount} nodes`);
        if (visited[index]) fail(`node ${index} is reached twice (a cycle or a shared child)`);
        if (parents[index] !== node) fail(`node ${index}'s parent is ${parents[index]}, but node ${node} holds it`);
        visited[index] = 1;
        reached++;
        slots.push(index * 2 + 1, index * 2);
        depths.push(slotDepth + 1, slotDepth + 1);
    }
    if (reached !== nodeCount) fail(`${nodeCount - reached} of the ${nodeCount} nodes are not reached from the root`);
    for (let position = 0; position < triangleCount; position++) if (!covered[position]) fail(`tree position ${position} is in no leaf`);
    if (!inOrder) {
        const copy = triangleWords.slice();
        let at = 0;
        for (const slot of leaves) {
            const indexWord = (slot >> 1) * GPU_LAYOUT_NODE_WORDS + ((slot & 1) === 0 ? GPU_LAYOUT_NODE_WORD.leftIndex : GPU_LAYOUT_NODE_WORD.rightIndex);
            const { index, count } = readSlot(words, slot);
            triangleWords.set(copy.subarray(index * stride, (index + count) * stride), at * stride);
            words[indexWord] = at;
            at += count;
        }
    }
    // A one-leaf tree: traverse reports that leaf as the root, at depth 0.
    if (words[GPU_LAYOUT_NODE_WORD.rightIndex] === GPU_LAYOUT_EMPTY_CHILD && words[GPU_LAYOUT_NODE_WORD.leftCount] > 0) depth = 0;
    return depth;
}

/** Copies child slot `slot`'s box (node `slot >> 1`, side `slot & 1`) into `box`. */
function readSlotBox(floats: Float32Array, slot: number, box: Float32Array): void {
    const base = (slot >> 1) * GPU_LAYOUT_NODE_WORDS + ((slot & 1) === 0 ? GPU_LAYOUT_NODE_WORD.leftMin : GPU_LAYOUT_NODE_WORD.rightMin);
    box[0] = floats[base];
    box[1] = floats[base + 1];
    box[2] = floats[base + 2];
    box[3] = floats[base + 4];
    box[4] = floats[base + 5];
    box[5] = floats[base + 6];
}

/** Scratch triangles for shapecasts (a cast may run inside another's callback, so each takes its own). */
const trianglePool: ExtendedTriangle[] = [];

const IDENTITY_MATRIX: readonly number[] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** {@link CpuMeshBvh.refit}'s order when every child's index is above its parent's: the nodes from last to first, no array. */
const REVERSE_NODE_ORDER = new Uint32Array(0);

/** Whether every inner child's node index is above its parent's (tinybvh's and the depth-first layouts), so last-to-first visits children first. */
function childrenFollowParents(words: Uint32Array, nodeCount: number): boolean {
    for (let node = 0; node < nodeCount; node++) {
        const base = node * GPU_LAYOUT_NODE_WORDS;
        const left = words[base + GPU_LAYOUT_NODE_WORD.leftIndex];
        if (words[base + GPU_LAYOUT_NODE_WORD.leftCount] === 0 && left !== GPU_LAYOUT_EMPTY_CHILD && left <= node) return false;
        const right = words[base + GPU_LAYOUT_NODE_WORD.rightIndex];
        if (words[base + GPU_LAYOUT_NODE_WORD.rightCount] === 0 && right !== GPU_LAYOUT_EMPTY_CHILD && right <= node) return false;
    }
    return true;
}

/**
 * {@link CpuMeshBvh.refit}'s quick validation: every index of the
 * `triangleCount` triangles below `vertexCount`, and every position finite,
 * in two straight scans. False sends the refit to the per-triangle check.
 */
function refitSourceLooksValid(positions: Float32Array, indices: Uint16Array | Uint32Array | null | undefined, triangleCount: number, vertexCount: number): boolean {
    if (indices) {
        for (let i = 0, n = triangleCount * 3; i < n; i++) if (indices[i] >= vertexCount) return false;
    }
    // v - v is NaN for NaN and both infinities, 0 otherwise.
    for (let i = 0, n = vertexCount * 3; i < n; i++) {
        const v = positions[i];
        if (v - v !== 0) return false;
    }
    return true;
}

/**
 * {@link CpuMeshBvh.refit}'s one pass, after validation: in `order` (null:
 * the nodes from last to first), each leaf slot's records get their corners
 * from `positions` and the slot their box, and each inner slot gets the box
 * around its child's two slots, as stored a moment before. The values are
 * `refitRecordsOnCpu`'s bit for bit (min and max of the same `f32` corners;
 * with finite corners no box is empty, so a stored slot is the box itself),
 * without its node-box array or a second read of the records. `bounds` gets
 * the root's box.
 */
function refitTreeFromSource(
    floats: Float32Array,
    words: Uint32Array,
    nodeCount: number,
    order: Uint32Array | null,
    records: Float32Array,
    recordWords: Uint32Array,
    positions: Float32Array,
    indices: Uint16Array | Uint32Array | null,
    bounds: Float32Array,
): void {
    for (let k = 0; k < nodeCount; k++) {
        const nodeWord = (order ? order[k] : nodeCount - 1 - k) * GPU_LAYOUT_NODE_WORDS;
        const left = words[nodeWord + GPU_LAYOUT_NODE_WORD.leftIndex];
        const leftCount = words[nodeWord + GPU_LAYOUT_NODE_WORD.leftCount];
        if (leftCount > 0) refitLeafSlot(floats, nodeWord, records, recordWords, left, leftCount, positions, indices);
        else unionOfChild(floats, words, left * GPU_LAYOUT_NODE_WORDS, floats, nodeWord);
        const right = words[nodeWord + GPU_LAYOUT_NODE_WORD.rightIndex];
        if (right === GPU_LAYOUT_EMPTY_CHILD) continue;
        const rightCount = words[nodeWord + GPU_LAYOUT_NODE_WORD.rightCount];
        if (rightCount > 0) refitLeafSlot(floats, nodeWord + GPU_LAYOUT_CHILD_STRIDE_WORDS, records, recordWords, right, rightCount, positions, indices);
        else unionOfChild(floats, words, right * GPU_LAYOUT_NODE_WORDS, floats, nodeWord + GPU_LAYOUT_CHILD_STRIDE_WORDS);
    }
    // The root's box: the union of node 0's slots.
    unionOfChild(floats, words, 0, ROOT_BOX, 0);
    bounds[0] = ROOT_BOX[0];
    bounds[1] = ROOT_BOX[1];
    bounds[2] = ROOT_BOX[2];
    bounds[3] = ROOT_BOX[4];
    bounds[4] = ROOT_BOX[5];
    bounds[5] = ROOT_BOX[6];
}

/** {@link refitTreeFromSource}'s root box, as a slot (min at 0-2, max at 4-6). */
const ROOT_BOX = new Float32Array(8);

/**
 * `corners: 'positions'`: throws unless every vertex index of the
 * `triangleCount` indexed triangles is below `vertexCount` (the corners are
 * read through them, so one past the positions would read nothing).
 */
function checkVertexIndices(triangleWords: Uint32Array, triangleCount: number, vertexCount: number, where = 'CpuMeshBvh'): void {
    for (let w = 0, end = triangleCount * INDEXED_WORDS; w < end; w += INDEXED_WORDS) {
        for (let corner = 0; corner < 3; corner++) {
            const vertex = triangleWords[w + corner];
            if (vertex >= vertexCount) {
                const triangle = triangleWords[w + INDEXED_ID_WORD] & GPU_LAYOUT_INDEXED_ID_MASK;
                throw new RangeError(`${where}: triangle ${triangle} uses vertex ${vertex}, past the ${vertexCount} positions`);
            }
        }
    }
}

/**
 * {@link refitTreeFromSource} for `corners: 'positions'`: the same pass and
 * the same boxes, with each leaf's corners read through the indexed
 * triangles' own vertex indices and nothing written but the boxes.
 */
function refitIndexedTreeFromSource(floats: Float32Array, words: Uint32Array, nodeCount: number, order: Uint32Array | null, triangleWords: Uint32Array, positions: Float32Array, bounds: Float32Array): void {
    for (let k = 0; k < nodeCount; k++) {
        const nodeWord = (order ? order[k] : nodeCount - 1 - k) * GPU_LAYOUT_NODE_WORDS;
        const left = words[nodeWord + GPU_LAYOUT_NODE_WORD.leftIndex];
        const leftCount = words[nodeWord + GPU_LAYOUT_NODE_WORD.leftCount];
        if (leftCount > 0) refitIndexedLeafSlot(floats, nodeWord, triangleWords, left, leftCount, positions);
        else unionOfChild(floats, words, left * GPU_LAYOUT_NODE_WORDS, floats, nodeWord);
        const right = words[nodeWord + GPU_LAYOUT_NODE_WORD.rightIndex];
        if (right === GPU_LAYOUT_EMPTY_CHILD) continue;
        const rightCount = words[nodeWord + GPU_LAYOUT_NODE_WORD.rightCount];
        if (rightCount > 0) refitIndexedLeafSlot(floats, nodeWord + GPU_LAYOUT_CHILD_STRIDE_WORDS, triangleWords, right, rightCount, positions);
        else unionOfChild(floats, words, right * GPU_LAYOUT_NODE_WORDS, floats, nodeWord + GPU_LAYOUT_CHILD_STRIDE_WORDS);
    }
    unionOfChild(floats, words, 0, ROOT_BOX, 0);
    bounds[0] = ROOT_BOX[0];
    bounds[1] = ROOT_BOX[1];
    bounds[2] = ROOT_BOX[2];
    bounds[3] = ROOT_BOX[4];
    bounds[4] = ROOT_BOX[5];
    bounds[5] = ROOT_BOX[6];
}

/** One leaf slot of {@link refitIndexedTreeFromSource}: the box around its triangles' corners (as {@link refitLeafSlot}'s). */
function refitIndexedLeafSlot(floats: Float32Array, slot: number, triangleWords: Uint32Array, first: number, count: number, positions: Float32Array): void {
    const max = BVH_FLOAT_MAX;
    let c0 = max, c1 = max, c2 = max, d0 = -max, d1 = -max, d2 = -max;
    for (let r = first * INDEXED_WORDS, end = (first + count) * INDEXED_WORDS; r < end; r += INDEXED_WORDS) {
        const a3 = triangleWords[r] * 3;
        const b3 = triangleWords[r + 1] * 3;
        const c3 = triangleWords[r + 2] * 3;
        const ax = positions[a3], ay = positions[a3 + 1], az = positions[a3 + 2];
        const bx = positions[b3], by = positions[b3 + 1], bz = positions[b3 + 2];
        const cx = positions[c3], cy = positions[c3 + 1], cz = positions[c3 + 2];
        c0 = Math.min(c0, ax, bx, cx);
        d0 = Math.max(d0, ax, bx, cx);
        c1 = Math.min(c1, ay, by, cy);
        d1 = Math.max(d1, ay, by, cy);
        c2 = Math.min(c2, az, bz, cz);
        d2 = Math.max(d2, az, bz, cz);
    }
    floats[slot] = c0;
    floats[slot + 1] = c1;
    floats[slot + 2] = c2;
    floats[slot + 4] = d0;
    floats[slot + 5] = d1;
    floats[slot + 6] = d2;
}

/**
 * One leaf slot of {@link refitTreeFromSource}: its records' corners from
 * `positions`, and the box around them (`refitInOrder`'s leaf box: min and
 * max from the largest float, over the `f32` corners).
 */
function refitLeafSlot(floats: Float32Array, slot: number, records: Float32Array, recordWords: Uint32Array, first: number, count: number, positions: Float32Array, indices: Uint16Array | Uint32Array | null): void {
    const max = BVH_FLOAT_MAX;
    let c0 = max, c1 = max, c2 = max, d0 = -max, d1 = -max, d2 = -max;
    for (let r = first * TRIANGLE_WORDS, end = (first + count) * TRIANGLE_WORDS; r < end; r += TRIANGLE_WORDS) {
        const t3 = recordWords[r + GPU_LAYOUT_TRIANGLE_WORD.originalIndex] * 3;
        const a3 = (indices ? indices[t3] : t3) * 3;
        const b3 = (indices ? indices[t3 + 1] : t3 + 1) * 3;
        const c3 = (indices ? indices[t3 + 2] : t3 + 2) * 3;
        const ax = Math.fround(positions[a3]), ay = Math.fround(positions[a3 + 1]), az = Math.fround(positions[a3 + 2]);
        const bx = Math.fround(positions[b3]), by = Math.fround(positions[b3 + 1]), bz = Math.fround(positions[b3 + 2]);
        const cx = Math.fround(positions[c3]), cy = Math.fround(positions[c3 + 1]), cz = Math.fround(positions[c3 + 2]);
        records[r + VERTEX0] = ax;
        records[r + VERTEX0 + 1] = ay;
        records[r + VERTEX0 + 2] = az;
        records[r + VERTEX1] = bx;
        records[r + VERTEX1 + 1] = by;
        records[r + VERTEX1 + 2] = bz;
        records[r + VERTEX2] = cx;
        records[r + VERTEX2 + 1] = cy;
        records[r + VERTEX2 + 2] = cz;
        c0 = Math.min(c0, ax, bx, cx);
        d0 = Math.max(d0, ax, bx, cx);
        c1 = Math.min(c1, ay, by, cy);
        d1 = Math.max(d1, ay, by, cy);
        c2 = Math.min(c2, az, bz, cz);
        d2 = Math.max(d2, az, bz, cz);
    }
    floats[slot] = c0;
    floats[slot + 1] = c1;
    floats[slot + 2] = c2;
    floats[slot + 4] = d0;
    floats[slot + 5] = d1;
    floats[slot + 6] = d2;
}

/**
 * The box around node `child`'s slots (`child` its first word), into `target`
 * at `slot` (min at 0-2, max at 4-6): `refitInOrder`'s inner box, min and max
 * from the largest float, read back from the slots stored a moment before.
 */
function unionOfChild(floats: Float32Array, words: Uint32Array, child: number, target: Float32Array, slot: number): void {
    const max = BVH_FLOAT_MAX;
    let c0 = Math.min(max, floats[child]), c1 = Math.min(max, floats[child + 1]), c2 = Math.min(max, floats[child + 2]);
    let d0 = Math.max(-max, floats[child + 4]), d1 = Math.max(-max, floats[child + 5]), d2 = Math.max(-max, floats[child + 6]);
    if (words[child + GPU_LAYOUT_NODE_WORD.rightIndex] !== GPU_LAYOUT_EMPTY_CHILD) {
        const right = child + GPU_LAYOUT_CHILD_STRIDE_WORDS;
        c0 = Math.min(c0, floats[right]);
        c1 = Math.min(c1, floats[right + 1]);
        c2 = Math.min(c2, floats[right + 2]);
        d0 = Math.max(d0, floats[right + 4]);
        d1 = Math.max(d1, floats[right + 5]);
        d2 = Math.max(d2, floats[right + 6]);
    }
    target[slot] = c0;
    target[slot + 1] = c1;
    target[slot + 2] = c2;
    target[slot + 4] = d0;
    target[slot + 5] = d1;
    target[slot + 6] = d2;
}

/** The flat walks' guard: a stack (sized from the checked tree's depth) about to overflow means the layout isn't the tree it was. */
function walkStackError(length: number): RangeError {
    return new RangeError(`CpuMeshBvh: the walk stack passed ${length} entries; the layout is not a tree (was it changed after the check?)`);
}

/** Whether a matrix is absent or exactly the identity (an axis-aligned query box). */
function isIdentityMatrix(m: BvhMatrix4Like | null | undefined): boolean {
    if (!m) return true;
    const e = matrixElements(m);
    for (let i = 0; i < 16; i++) if (e[i] !== IDENTITY_MATRIX[i]) return false;
    return true;
}

/** Squared distance from a point to a child box (`minWord` its min, `minWord + 4` its max): three's `clamp` then `distanceToSquared`. */
function boxDistanceSq(floats: Float32Array, minWord: number, px: number, py: number, pz: number): number {
    const dx = Math.max(floats[minWord], Math.min(floats[minWord + 4], px)) - px;
    const dy = Math.max(floats[minWord + 1], Math.min(floats[minWord + 5], py)) - py;
    const dz = Math.max(floats[minWord + 2], Math.min(floats[minWord + 6], pz)) - pz;
    return dx * dx + dy * dy + dz * dz;
}

/**
 * Below this squared sine of a triangle's angle at `a` (about 3e-4, 1:3000) it is a
 * sliver: the GPU's `BVH_SLIVER_SINE_SQ` (src/trace/wgsl.ts, `bvhClosestPointToTriangle`).
 */
const SLIVER_SINE_SQ = 1e-7;

/**
 * The point of the triangle whose corners start at words `a`, `b`, `c` of
 * `floats` (a record's, or the source's positions) closest to `io[0..2]`,
 * into `io[3..5]`.
 * Numbers go through `io`, not arguments: V8 doesn't inline a function this
 * size, and every number argument of a call that isn't inlined is boxed.
 *
 * Three's `Triangle.closestPointToPoint` ({@link closestByRegions}), bit for
 * bit, except on slivers (squared sine below {@link SLIVER_SINE_SQ}: a
 * segment, a point, or nearly one), where its regions' differences of
 * products cancel and can pick a wrong region even in doubles: measured on
 * f32 slivers, up to 1.7 units too far on a unit triangle (the GPU query
 * oracle found a point lying on one reported 2.2e-3 away). There the nearer
 * of three's point and the nearest point of the three edges is kept. Three's
 * point is on the triangle to rounding (never measured nearer than the true
 * distance), so the nearer one is never worse; on the same test the error
 * falls to 2e-6 at most. Ties keep three's point. The GPU takes the edges
 * alone on slivers; the two agree to within the sliver's height.
 */
function closestOnTriangle(floats: Float32Array, a: number, b: number, c: number, out: Float64Array): void {
    closestByRegions(floats, a, b, c, out);
    if (out[6] !== 0) closestOnEdges(floats, a, b, c, out);
}

/**
 * Replaces `io[3..5]` with the nearest point of the triangle's three edges
 * (`a-b`, `a-c`, `b-c`, the first strictly nearest) when that is strictly
 * nearer to `io[0..2]`, or when `io[3..5]` isn't a number.
 */
function closestOnEdges(records: Float32Array, a: number, b: number, c: number, out: Float64Array): void {
    const px = out[0], py = out[1], pz = out[2];
    let dx = out[3] - px, dy = out[4] - py, dz = out[5] - pz;
    let best = dx * dx + dy * dy + dz * dz;
    if (best !== best) best = Infinity;
    for (let side = 0; side < 3; side++) {
        const s = side === 2 ? b : a;
        const e = side === 0 ? b : c;
        const sx = records[s], sy = records[s + 1], sz = records[s + 2];
        const ex = records[e] - sx, ey = records[e + 1] - sy, ez = records[e + 2] - sz;
        const lengthSq = ex * ex + ey * ey + ez * ez;
        let t = lengthSq > 0 ? ((px - sx) * ex + (py - sy) * ey + (pz - sz) * ez) / lengthSq : 0;
        t = t > 0 ? (t < 1 ? t : 1) : 0;
        const qx = sx + ex * t, qy = sy + ey * t, qz = sz + ez * t;
        dx = qx - px;
        dy = qy - py;
        dz = qz - pz;
        const distanceSq = dx * dx + dy * dy + dz * dz;
        if (distanceSq < best) {
            best = distanceSq;
            out[3] = qx;
            out[4] = qy;
            out[5] = qz;
        }
    }
}

/**
 * Three's `Triangle.closestPointToPoint` (`triangleClosestPointToPoint`) on the
 * corners at `a`, `b`, `c`, plain numbers, the same operations in the same order: the
 * same bits. `io[6]` is set to 1 when the triangle is a sliver (squared sine at
 * `a` at most {@link SLIVER_SINE_SQ}), else 0.
 */
function closestByRegions(records: Float32Array, a: number, b: number, c: number, out: Float64Array): void {
    const ax = records[a], ay = records[a + 1], az = records[a + 2];
    const bx = records[b], by = records[b + 1], bz = records[b + 2];
    const cx = records[c], cy = records[c + 1], cz = records[c + 2];
    const px = out[0], py = out[1], pz = out[2];
    const abx = bx - ax, aby = by - ay, abz = bz - az;
    const acx = cx - ax, acy = cy - ay, acz = cz - az;
    const nx = aby * acz - abz * acy, ny = abz * acx - abx * acz, nz = abx * acy - aby * acx;
    out[6] = nx * nx + ny * ny + nz * nz <= SLIVER_SINE_SQ * (abx * abx + aby * aby + abz * abz) * (acx * acx + acy * acy + acz * acz) ? 1 : 0;
    const apx = px - ax, apy = py - ay, apz = pz - az;
    const d1 = abx * apx + aby * apy + abz * apz;
    const d2 = acx * apx + acy * apy + acz * apz;
    if (d1 <= 0 && d2 <= 0) {
        out[3] = ax;
        out[4] = ay;
        out[5] = az;
        return;
    }
    const bpx = px - bx, bpy = py - by, bpz = pz - bz;
    const d3 = abx * bpx + aby * bpy + abz * bpz;
    const d4 = acx * bpx + acy * bpy + acz * bpz;
    if (d3 >= 0 && d4 <= d3) {
        out[3] = bx;
        out[4] = by;
        out[5] = bz;
        return;
    }
    const vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0 && d1 - d3 > 0) {
        const v = d1 / (d1 - d3);
        out[3] = ax + abx * v;
        out[4] = ay + aby * v;
        out[5] = az + abz * v;
        return;
    }
    const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
    const d5 = abx * cpx + aby * cpy + abz * cpz;
    const d6 = acx * cpx + acy * cpy + acz * cpz;
    if (d6 >= 0 && d5 <= d6) {
        out[3] = cx;
        out[4] = cy;
        out[5] = cz;
        return;
    }
    const vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        const w = d2 / (d2 - d6);
        out[3] = ax + acx * w;
        out[4] = ay + acy * w;
        out[5] = az + acz * w;
        return;
    }
    const va = d3 * d6 - d5 * d4;
    if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
        const w = (d4 - d3) / (d4 - d3 + (d5 - d6));
        out[3] = bx + (cx - bx) * w;
        out[4] = by + (cy - by) * w;
        out[5] = bz + (cz - bz) * w;
        return;
    }
    const denom = 1 / (va + vb + vc);
    const v = vb * denom;
    const w = vc * denom;
    out[3] = ax + abx * v + acx * w;
    out[4] = ay + aby * v + acy * w;
    out[5] = az + abz * v + acz * w;
}

const arrayBox = new BvhBox3();

/** A box array `[min..., max...]` as the scratch {@link BvhBox3} (for a test that calls no user code). */
function boxOfArray(array: ArrayLike<number>): BvhBox3 {
    arrayBox.min.set(array[0], array[1], array[2]);
    arrayBox.max.set(array[3], array[4], array[5]);
    return arrayBox;
}

/** The box around every position of a mesh (three's `computeBoundingBox`: all vertices, indexed or not). */
function meshBounds(input: MeshInput): Float64Array {
    const out = new Float64Array([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
    const positions = input.positions;
    for (let i = 0; i + 2 < positions.length; i += 3) {
        for (let axis = 0; axis < 3; axis++) {
            const value = positions[i + axis];
            if (value < out[axis]) out[axis] = value;
            if (value > out[3 + axis]) out[3 + axis] = value;
        }
    }
    return out;
}

/** A tree's arrays, for {@link dualTraverse}. */
interface TreeView {
    readonly words: Uint32Array;
    readonly floats: Float32Array;
    readonly bounds: ArrayLike<number>;
}

/**
 * A node of a {@link TreeView}: a child slot (`>= 0`: its box and reference
 * are in the parent), or `-(node + 1)` for the root node (its box the tree's
 * bounds).
 */
type NodeHandle = number;

type RangesReport = (offset1: number, count1: number, offset2: number, count2: number, depth1: number, nodeIndex1: number, depth2: number, nodeIndex2: number) => boolean;

/** The root's handle: the first node with two children, or the one leaf of a one-leaf tree (as `traverse` finds it). */
function rootHandle(tree: TreeView): NodeHandle {
    const words = tree.words;
    let root = 0;
    while (words[root * GPU_LAYOUT_NODE_WORDS + GPU_LAYOUT_NODE_WORD.rightIndex] === GPU_LAYOUT_EMPTY_CHILD) {
        const base = root * GPU_LAYOUT_NODE_WORDS;
        if (words[base + GPU_LAYOUT_NODE_WORD.leftCount] > 0) return root * 2;
        root = words[base + GPU_LAYOUT_NODE_WORD.leftIndex];
    }
    return -(root + 1);
}

/** Triangles in a handle's leaf; 0 for an inner node. */
function handleCount(tree: TreeView, handle: NodeHandle): number {
    if (handle < 0) return 0;
    return tree.words[(handle >> 1) * GPU_LAYOUT_NODE_WORDS + ((handle & 1) === 0 ? GPU_LAYOUT_NODE_WORD.leftCount : GPU_LAYOUT_NODE_WORD.rightCount)];
}

/** A leaf's first tree position, or an inner node's node index. */
function handleIndex(tree: TreeView, handle: NodeHandle): number {
    if (handle < 0) return -handle - 1;
    return tree.words[(handle >> 1) * GPU_LAYOUT_NODE_WORDS + ((handle & 1) === 0 ? GPU_LAYOUT_NODE_WORD.leftIndex : GPU_LAYOUT_NODE_WORD.rightIndex)];
}

/** An inner node's right child, or -1 when it has none (then never overlapping). */
function rightChild(tree: TreeView, node: number): NodeHandle {
    return tree.words[node * GPU_LAYOUT_NODE_WORDS + GPU_LAYOUT_NODE_WORD.rightIndex] === GPU_LAYOUT_EMPTY_CHILD ? -1 : node * 2 + 1;
}

function readHandleBox(tree: TreeView, handle: NodeHandle, out: Float64Array): Float64Array {
    if (handle < 0) {
        for (let i = 0; i < 6; i++) out[i] = tree.bounds[i];
        return out;
    }
    const base = (handle >> 1) * GPU_LAYOUT_NODE_WORDS + ((handle & 1) === 0 ? GPU_LAYOUT_NODE_WORD.leftMin : GPU_LAYOUT_NODE_WORD.rightMin);
    const floats = tree.floats;
    out[0] = floats[base];
    out[1] = floats[base + 1];
    out[2] = floats[base + 2];
    out[3] = floats[base + 4];
    out[4] = floats[base + 5];
    out[5] = floats[base + 6];
    return out;
}

/**
 * three-mesh-bvh 0.9.18's `bvhcast` walk (`core/cast/bvhcast.js`) on our
 * nodes; `matrix2to1` takes tree B into tree A's space. The order, the box
 * tests (three's `Box3.applyMatrix4` and `intersectsBox`) and the depth
 * bookkeeping are three-mesh-bvh's, including the depth it passes on a switch.
 */
function dualTraverse(treeA: TreeView, treeB: TreeView, matrix2to1: ArrayLike<number>, report: RangesReport): boolean {
    const inverse = new BvhMatrix4();
    invertMatrix(matrix2to1, inverse);
    const matrix1to2 = inverse.elements;
    // three-mesh-bvh's _leftBox1/_rightBox1/_leftBox2/_rightBox2 (read and tested before any recursion), and a pool for switched boxes.
    const left1 = new Float64Array(6);
    const right1 = new Float64Array(6);
    const left2 = new Float64Array(6);
    const right2 = new Float64Array(6);
    const pool: Float64Array[] = [];
    const take = (): Float64Array => pool.pop() ?? new Float64Array(6);
    const overlaps = (box: Float64Array, tree: TreeView, handle: NodeHandle, scratch: Float64Array): boolean => handle >= 0 && boxArraysIntersect(box, readHandleBox(tree, handle, scratch));

    /** Switches trees: `box` (node `from` of tree2, in tree1's space) against node1's children, descending tree1. */
    const descendOther = (
        from: NodeHandle,
        node1: NodeHandle,
        tree1: TreeView,
        tree2: TreeView,
        m2to1: ArrayLike<number>,
        m1to2: ArrayLike<number>,
        depth1: number,
        depth2: number,
        box: Float64Array,
        reversed: boolean,
    ): boolean => {
        const inner1 = handleIndex(tree1, node1);
        const cl1 = inner1 * 2;
        const cr1 = rightChild(tree1, inner1);
        const intersectCl1 = overlaps(box, tree1, cl1, left1);
        const intersectCr1 = overlaps(box, tree1, cr1, right1);
        return (
            (intersectCl1 && walk(from, cl1, tree2, tree1, m1to2, m2to1, depth2, depth1 + 1, box, !reversed)) ||
            (intersectCr1 && walk(from, cr1, tree2, tree1, m1to2, m2to1, depth2, depth1 + 1, box, !reversed))
        );
    };

    const walk = (
        node1: NodeHandle,
        node2: NodeHandle,
        tree1: TreeView,
        tree2: TreeView,
        m2to1: ArrayLike<number>,
        m1to2: ArrayLike<number>,
        depth1: number,
        depth2: number,
        currBox: Float64Array,
        reversed: boolean,
    ): boolean => {
        const count1 = handleCount(tree1, node1);
        const count2 = handleCount(tree2, node2);
        if (count1 > 0 && count2 > 0) {
            const offset1 = handleIndex(tree1, node1);
            const offset2 = handleIndex(tree2, node2);
            return reversed ? report(offset2, count2, offset1, count1, depth2, node2, depth1, node1) : report(offset1, count1, offset2, count2, depth1, node1, depth2, node2);
        }
        if (count2 > 0) {
            // tree2 reached a leaf: carry on down tree1 with that leaf's box.
            const newBox = take();
            transformBoxArray(readHandleBox(tree2, node2, newBox), m2to1, newBox);
            const result = descendOther(node2, node1, tree1, tree2, m2to1, m1to2, depth1, depth2, newBox, reversed);
            pool.push(newBox);
            return result;
        }
        const inner2 = handleIndex(tree2, node2);
        const cl2 = inner2 * 2;
        const cr2 = rightChild(tree2, inner2);
        const leftIntersects = overlaps(currBox, tree2, cl2, left2);
        const rightIntersects = overlaps(currBox, tree2, cr2, right2);
        if (leftIntersects && rightIntersects) {
            return walk(node1, cl2, tree1, tree2, m2to1, m1to2, depth1, depth2 + 1, currBox, reversed) || walk(node1, cr2, tree1, tree2, m2to1, m1to2, depth1, depth2 + 1, currBox, reversed);
        }
        if (!leftIntersects && !rightIntersects) return false;
        const child2 = leftIntersects ? cl2 : cr2;
        if (count1 > 0) return walk(node1, child2, tree1, tree2, m2to1, m1to2, depth1, depth2 + 1, currBox, reversed);
        // Only one child overlaps: switch, and carry on down tree1 with that child's box.
        const newBox = take();
        transformBoxArray(leftIntersects ? left2 : right2, m2to1, newBox);
        const result = descendOther(child2, node1, tree1, tree2, m2to1, m1to2, depth1, depth2, newBox, reversed);
        pool.push(newBox);
        return result;
    };

    const rootA = rootHandle(treeA);
    const rootB = rootHandle(treeB);
    const localBox = new Float64Array(6);
    transformBoxArray(readHandleBox(treeA, rootA, localBox), matrix1to2, localBox);
    return walk(rootA, rootB, treeA, treeB, matrix2to1, matrix1to2, 0, 0, localBox, false);
}

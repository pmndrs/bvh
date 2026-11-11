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
 */

import { GPU_LAYOUT_EMPTY_CHILD, GPU_LAYOUT_MAX_LEAF_PRIMITIVES, GPU_LAYOUT_NO_PARENT, GPU_LAYOUT_NODE_BYTES, GPU_LAYOUT_NODE_WORD, GPU_LAYOUT_NODE_WORDS, GPU_LAYOUT_TRIANGLE_BYTES, GPU_LAYOUT_TRIANGLE_WORD, type GpuLayoutMeshBvh } from '../gpu-layout.ts';
import type { MeshInput } from '../mesh-input.ts';
import { PRIMITIVE_BOX_FLOATS, refitOnCpu } from '../refit-cpu.ts';
import { deserializeLayout, serializeLayout } from '../serialize.ts';
import { createTinyBvhBuilder, type TinyBvhBuilder } from '../tinybvh.ts';
import { createCpuTracer, CPU_TRACE_BACK_SIDE, CPU_TRACE_DOUBLE_SIDE, CPU_TRACE_FRONT_SIDE, type CpuTraceOptions, type CpuTracer } from '../trace/cpu-traversal.ts';
import { buildMeshBvhWithTinyBvh } from '../trace/mesh-bvh-topology.ts';
import { buildTopLevel, IDENTITY_TRANSFORM, type TopLevelBvh } from '../trace/top-level.ts';
import type { TriangleCandidate } from '../trace/triangle-filter.ts';
import {
    BvhBox3,
    boxArraysIntersect,
    ExtendedTriangle,
    invertMatrix,
    matrixElements,
    transformBoxArray,
    BvhMatrix4,
    OrientedBox,
    sphereIntersectsBox,
    triangleClosestPointToPoint,
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
const CORNER_WORDS = [GPU_LAYOUT_TRIANGLE_WORD.vertex0, GPU_LAYOUT_TRIANGLE_WORD.vertex1, GPU_LAYOUT_TRIANGLE_WORD.vertex2] as const;

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
}

export interface CpuMeshBvhOptions {
    /**
     * The geometry the tree was built from. {@link CpuMeshBvh.refit} rereads
     * its positions, and {@link CpuMeshBvh.getTriangleHitPointInfo} its index.
     * Kept by reference, not copied.
     */
    readonly source?: MeshInput | null;
}

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

    constructor(layout: GpuLayoutMeshBvh, options: CpuMeshBvhOptions = {}) {
        if (layout.nodes.byteLength !== layout.nodeCount * GPU_LAYOUT_NODE_BYTES) throw new RangeError(`CpuMeshBvh: ${layout.nodeCount} nodes need ${layout.nodeCount * GPU_LAYOUT_NODE_BYTES} bytes, got ${layout.nodes.byteLength}`);
        if (layout.triangles.byteLength !== layout.triangleCount * GPU_LAYOUT_TRIANGLE_BYTES) {
            throw new RangeError(`CpuMeshBvh: ${layout.triangleCount} triangles need ${layout.triangleCount * GPU_LAYOUT_TRIANGLE_BYTES} bytes, got ${layout.triangles.byteLength}`);
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
        this.treeDepth = checkTreeShape(this.nodeWords, this.triangleWords, layout.nodeCount, layout.triangleCount, layout.parents);
        this.topLevel = this.makeTopLevel();
        this.tracer = this.makeTracer(false);
        this.filterTracer = this.makeTracer(true);
    }

    /** Builds a tree with tinybvh (`createTinyBvhBuilder()`), on this thread. Keeps `input` as its source. */
    static build(input: MeshInput, builder: TinyBvhBuilder, options: CpuMeshBvhBuildOptions = {}): CpuMeshBvh {
        return new CpuMeshBvh(CpuMeshBvh.buildLayout(input, builder, options), { source: input });
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
        const layout = buildMeshBvhWithTinyBvh(input, builder, { maxLeafPrimitives: options.maxLeafPrimitives });
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

    /** Bytes this object holds: the layout (nodes, triangle records, parents), the one-instance top level and lookups made so far. Not the source geometry. */
    estimateMemoryInBytes(): number {
        return this.layout.nodes.byteLength + this.layout.triangles.byteLength + this.layout.parents.byteLength + this.layout.bounds.byteLength + this.topLevel.buffer.byteLength + (this.positions?.byteLength ?? 0);
    }

    /** The original triangle index at a tree position (as {@link traverse} reports them). */
    resolveTriangleIndex(treePosition: number): number {
        return this.triangleWords[treePosition * TRIANGLE_WORDS + GPU_LAYOUT_TRIANGLE_WORD.originalIndex];
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
        const base = treePosition * TRIANGLE_WORDS;
        for (let corner = 0; corner < 3; corner++) {
            for (let axis = 0; axis < 3; axis++) target[corner * 3 + axis] = this.triangleFloats[base + CORNER_WORDS[corner] + axis];
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
        const floats = this.triangleFloats;
        const base = treePosition * TRIANGLE_WORDS;
        const points = triangle.points;
        for (let corner = 0; corner < 3; corner++) {
            const word = base + CORNER_WORDS[corner];
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
        const obb = new OrientedBox().set(box.min, box.max, boxToMesh);
        return this.coreShapecast({
            intersectsBounds: (bounds) => obb.intersectsBox(bounds),
            intersectsTriangle: (triangle) => obb.intersectsTriangle(triangle),
        });
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
     */
    closestPointToPoint(point: BvhVec3Like, target: BvhClosestPointTarget = {}, minThreshold = 0, maxThreshold = Infinity): BvhClosestPoint | null {
        const [px, py, pz] = readVec3(point);
        const query = new BvhVector3(px, py, pz);
        const temp = new BvhVector3();
        const best = new BvhVector3();
        const minThresholdSq = minThreshold * minThreshold;
        const maxThresholdSq = maxThreshold * maxThreshold;
        let closestDistanceSq = Infinity;
        let closestTriangle = -1;
        this.coreShapecast({
            boundsTraverseOrder: (box) => {
                // three: temp.copy(point).clamp(box.min, box.max); temp.distanceToSquared(point)
                const dx = Math.max(box.min.x, Math.min(box.max.x, px)) - px;
                const dy = Math.max(box.min.y, Math.min(box.max.y, py)) - py;
                const dz = Math.max(box.min.z, Math.min(box.max.z, pz)) - pz;
                return dx * dx + dy * dy + dz * dz;
            },
            intersectsBounds: (_box, _isLeaf, score) => score! < closestDistanceSq && score! < maxThresholdSq,
            intersectsTriangle: (triangle, triangleIndex) => {
                triangleClosestPointToPoint(triangle.a, triangle.b, triangle.c, query, temp);
                const dx = px - temp.x;
                const dy = py - temp.y;
                const dz = pz - temp.z;
                const distanceSq = dx * dx + dy * dy + dz * dz;
                if (distanceSq < closestDistanceSq) {
                    best.copy(temp);
                    closestDistanceSq = distanceSq;
                    closestTriangle = triangleIndex;
                }
                return distanceSq < minThresholdSq;
            },
        });
        if (closestDistanceSq === Infinity) return null;
        return this.writeClosestPoint(target, best, Math.sqrt(closestDistanceSq), closestTriangle);
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
     */
    refit(input?: MeshInput): void {
        const source = input ?? this.source;
        if (!source) throw new Error('CpuMeshBvh.refit: the tree has no source geometry; pass one (refit({ positions, indices }))');
        const { positions, indices } = source;
        const sourceTriangles = indices ? Math.floor(indices.length / 3) : Math.floor(positions.length / 9);
        if (sourceTriangles !== this.triangleCount) throw new RangeError(`CpuMeshBvh.refit: the source has ${sourceTriangles} triangles, the tree ${this.triangleCount}`);
        const vertexCount = Math.floor(positions.length / 3);
        const words = this.triangleWords;
        // Check everything first, so a refit that throws leaves the tree as it was.
        for (let position = 0; position < this.triangleCount; position++) {
            const triangle = words[position * TRIANGLE_WORDS + GPU_LAYOUT_TRIANGLE_WORD.originalIndex];
            for (let corner = 0; corner < 3; corner++) {
                const vertex = indices ? indices[triangle * 3 + corner] : triangle * 3 + corner;
                if (vertex >= vertexCount) throw new RangeError(`CpuMeshBvh.refit: triangle ${triangle} uses vertex ${vertex}, past the ${vertexCount} positions`);
                for (let axis = 0; axis < 3; axis++) {
                    if (!Number.isFinite(positions[vertex * 3 + axis])) throw new RangeError(`CpuMeshBvh.refit: vertex ${vertex} has a non-finite position`);
                }
            }
        }
        const boxes = new Float32Array(this.triangleCount * PRIMITIVE_BOX_FLOATS);
        const floats = this.triangleFloats;
        for (let position = 0; position < this.triangleCount; position++) {
            const base = position * TRIANGLE_WORDS;
            const triangle = words[base + GPU_LAYOUT_TRIANGLE_WORD.originalIndex];
            const at = position * PRIMITIVE_BOX_FLOATS;
            boxes[at] = boxes[at + 1] = boxes[at + 2] = Infinity;
            boxes[at + 4] = boxes[at + 5] = boxes[at + 6] = -Infinity;
            for (let corner = 0; corner < 3; corner++) {
                const vertex = indices ? indices[triangle * 3 + corner] : triangle * 3 + corner;
                const word = base + CORNER_WORDS[corner];
                for (let axis = 0; axis < 3; axis++) {
                    floats[word + axis] = positions[vertex * 3 + axis];
                    const stored = floats[word + axis];
                    if (stored < boxes[at + axis]) boxes[at + axis] = stored;
                    if (stored > boxes[at + 4 + axis]) boxes[at + 4 + axis] = stored;
                }
            }
        }
        this.source = source;
        const { rootBox } = refitOnCpu(this.nodeFloats, this.nodeWords, 0, this.nodeCount, this.layout.parents, boxes);
        for (let i = 0; i < 6; i++) this.layout.bounds[i] = rootBox[i];
        this.topLevel = this.makeTopLevel();
        this.tracer = this.makeTracer(false);
        this.filterTracer = this.makeTracer(true);
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
        const scene = { nodes: this.layout.nodes, triangles: this.layout.triangles, topLevel: this.topLevel };
        return filtered ? createCpuTracer(scene, { triangleFilter: (candidate) => this.passes(candidate), filterEveryTriangle: true }) : createCpuTracer(scene);
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
function checkTreeShape(words: Uint32Array, triangleWords: Uint32Array, nodeCount: number, triangleCount: number, parents: Uint32Array): number {
    const fail = (problem: string): never => {
        throw new RangeError(`CpuMeshBvh: the layout is not a valid tree: ${problem}`);
    };
    for (let position = 0; position < triangleCount; position++) {
        const original = triangleWords[position * TRIANGLE_WORDS + GPU_LAYOUT_TRIANGLE_WORD.originalIndex];
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
            triangleWords.set(copy.subarray(index * TRIANGLE_WORDS, (index + count) * TRIANGLE_WORDS), at * TRIANGLE_WORDS);
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

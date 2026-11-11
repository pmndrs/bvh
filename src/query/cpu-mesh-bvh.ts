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
 * of one identity instance. Side culling and "every hit" go through the
 * walk's triangle filter (`filterEveryTriangle`). There is no second ray walk
 * and no second triangle test: a ray against this tree finds what the GPU
 * walk finds.
 *
 * **One box walk.** {@link CpuMeshBvh.traverse} is the generic walk the
 * shapecast and closest-point queries build on: the caller decides which
 * boxes to enter and what to do with each leaf's triangle range. It is not
 * used for rays, because the ray walk must stay the WGSL walk's twin (nearer
 * child first, `t` narrowing, the 64-entry stack).
 *
 * **Triangle indices.** The tree never reorders the geometry. Its triangle
 * records are copies, in tree order, each carrying its original index. A
 * raycast hit's `triangle` is that original index. A *tree position* (what
 * {@link CpuMeshBvh.traverse} hands to `intersectsRange`) becomes the original
 * index through {@link CpuMeshBvh.resolveTriangleIndex}. That is
 * three-mesh-bvh's `indirect` mode, which this tree is always in.
 */

import { GPU_LAYOUT_EMPTY_CHILD, GPU_LAYOUT_MAX_LEAF_PRIMITIVES, GPU_LAYOUT_NODE_BYTES, GPU_LAYOUT_NODE_WORD, GPU_LAYOUT_NODE_WORDS, GPU_LAYOUT_TRIANGLE_BYTES, GPU_LAYOUT_TRIANGLE_WORD, type GpuLayoutMeshBvh } from '../gpu-layout.ts';
import type { MeshInput } from '../mesh-input.ts';
import { PRIMITIVE_BOX_FLOATS, refitOnCpu } from '../refit-cpu.ts';
import { deserializeLayout, serializeLayout } from '../serialize.ts';
import { createTinyBvhBuilder, type TinyBvhBuilder } from '../tinybvh.ts';
import { createCpuTracer, type CpuTracer } from '../trace/cpu-traversal.ts';
import { buildMeshBvhWithTinyBvh } from '../trace/mesh-bvh-topology.ts';
import { buildTopLevel, IDENTITY_TRANSFORM, type TopLevelBvh } from '../trace/top-level.ts';
import type { TriangleCandidate } from '../trace/triangle-filter.ts';

const TRIANGLE_WORDS = GPU_LAYOUT_TRIANGLE_BYTES / 4;
const CORNER_WORDS = [GPU_LAYOUT_TRIANGLE_WORD.vertex0, GPU_LAYOUT_TRIANGLE_WORD.vertex1, GPU_LAYOUT_TRIANGLE_WORD.vertex2] as const;

/** Which faces a raycast hits. The values are three's `FrontSide`, `BackSide` and `DoubleSide`. */
export const BVH_FRONT_SIDE = 0;
export const BVH_BACK_SIDE = 1;
export const BVH_DOUBLE_SIDE = 2;

/** What {@link BvhTraverseCallbacks.intersectsBounds} returns. The values are three-mesh-bvh's. */
export const BVH_NOT_INTERSECTED = 0;
export const BVH_INTERSECTED = 1;
/** The box is wholly inside the shape: every range below it is reported with `contained` and no more boxes are asked about. */
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
 * The callbacks of {@link CpuMeshBvh.traverse}. `box` is
 * `[minX, minY, minZ, maxX, maxY, maxZ]` in a scratch array that is only
 * valid during the call; copy it to keep it.
 */
export interface BvhTraverseCallbacks {
    /**
     * Whether the walk enters a box: {@link BVH_NOT_INTERSECTED} (or false)
     * skips it, {@link BVH_INTERSECTED} (or true) enters it,
     * {@link BVH_CONTAINED} enters it and reports every range below it as
     * contained without asking about more boxes. `isLeaf`: the box holds a
     * triangle range rather than two more boxes. `score` is
     * `boundsTraverseOrder`'s, when there is one. The root's box comes first, at depth 0.
     */
    intersectsBounds(box: Float32Array, isLeaf: boolean, score: number | undefined, depth: number): number | boolean;
    /**
     * A leaf's triangles: tree positions `start` to `start + count - 1`
     * (read them with {@link CpuMeshBvh.getTriangle} or map them with
     * {@link CpuMeshBvh.resolveTriangleIndex}). Return true to stop the walk.
     */
    intersectsRange(start: number, count: number, contained: boolean, depth: number, box: Float32Array): boolean;
    /** Orders two sibling boxes: the lower score is visited first. Without it, left before right. */
    boundsTraverseOrder?(box: Float32Array): number;
}

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

function readVec3(value: BvhVec3Like): [number, number, number] {
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
    private tracer: CpuTracer;
    /** Tree position of each original triangle, made on first use. */
    private positions: Uint32Array | null = null;
    /** The current raycast's acceptance test and hit list (null: closest only). */
    private accept: BvhTriangleAccept | null = null;
    private collect: BvhRayIntersection[] | null = null;
    private readonly scratchBox = new Float32Array(6);

    constructor(layout: GpuLayoutMeshBvh, options: CpuMeshBvhOptions = {}) {
        if (layout.nodes.byteLength !== layout.nodeCount * GPU_LAYOUT_NODE_BYTES) throw new RangeError(`CpuMeshBvh: ${layout.nodeCount} nodes need ${layout.nodeCount * GPU_LAYOUT_NODE_BYTES} bytes, got ${layout.nodes.byteLength}`);
        if (layout.triangles.byteLength !== layout.triangleCount * GPU_LAYOUT_TRIANGLE_BYTES) {
            throw new RangeError(`CpuMeshBvh: ${layout.triangleCount} triangles need ${layout.triangleCount * GPU_LAYOUT_TRIANGLE_BYTES} bytes, got ${layout.triangles.byteLength}`);
        }
        if (layout.nodeCount < 1 || layout.triangleCount < 1) throw new RangeError('CpuMeshBvh: the tree is empty');
        this.layout = layout;
        this.source = options.source ?? null;
        this.nodeFloats = new Float32Array(layout.nodes);
        this.nodeWords = new Uint32Array(layout.nodes);
        this.triangleFloats = new Float32Array(layout.triangles);
        this.triangleWords = new Uint32Array(layout.triangles);
        this.topLevel = this.makeTopLevel();
        this.tracer = this.makeTracer();
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
     * in `[tMin, tMax]`, in units of `direction` as given. `accept` decides
     * which hits count (by original triangle and side); the others are passed
     * through. All hits come in walk order, not sorted.
     */
    intersectRay(
        origin: readonly [number, number, number],
        direction: readonly [number, number, number],
        tMin: number,
        tMax: number,
        firstHitOnly: boolean,
        accept: BvhTriangleAccept | null = null,
    ): BvhRayIntersection[] {
        const previousAccept = this.accept;
        const previousCollect = this.collect;
        const collect: BvhRayIntersection[] | null = firstHitOnly ? null : [];
        this.accept = accept;
        this.collect = collect;
        try {
            const hit = this.tracer.closest({ origin, direction, tMin, tMax });
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
        const accept: BvhTriangleAccept | null =
            side === BVH_FRONT_SIDE ? (_, frontFace) => frontFace : side === BVH_BACK_SIDE ? (_, frontFace) => !frontFace : side === BVH_DOUBLE_SIDE ? null : () => false;
        const found = this.intersectRay(origin, direction, Math.max(0, near) / length, far / length, firstHitOnly, accept);
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
     * each entered leaf's triangle range (see {@link BvhTraverseCallbacks}).
     * Depth first; siblings in `boundsTraverseOrder` order. Returns true when
     * `intersectsRange` stopped it. No depth limit (the stack is a JS array).
     */
    traverse(callbacks: BvhTraverseCallbacks): boolean {
        const box = this.scratchBox;
        const words = this.nodeWords;
        const floats = this.nodeFloats;
        const order = callbacks.boundsTraverseOrder;
        box.set(this.layout.bounds);
        const rootScore = order ? order(box) : undefined;
        const root = Number(callbacks.intersectsBounds(box, false, rootScore, 0));
        if (root === BVH_NOT_INTERSECTED) return false;
        // Stack entries: child slot (node * 2 + side), depth, contained (0/1).
        const stack: number[] = [];
        const pushChildren = (node: number, depth: number, contained: number): void => {
            const base = node * GPU_LAYOUT_NODE_WORDS;
            const hasRight = words[base + GPU_LAYOUT_NODE_WORD.rightIndex] !== GPU_LAYOUT_EMPTY_CHILD;
            let first = node * 2;
            let second = node * 2 + 1;
            if (hasRight && order && !contained) {
                readSlotBox(floats, first, box);
                const left = order(box);
                readSlotBox(floats, second, box);
                if (order(box) < left) [first, second] = [second, first];
            }
            if (hasRight) stack.push(second, depth, contained);
            stack.push(first, depth, contained);
        };
        pushChildren(0, 1, root === BVH_CONTAINED ? 1 : 0);
        while (stack.length > 0) {
            const contained = stack.pop()!;
            const depth = stack.pop()!;
            const slot = stack.pop()!;
            const base = (slot >> 1) * GPU_LAYOUT_NODE_WORDS;
            const left = (slot & 1) === 0;
            const index = words[base + (left ? GPU_LAYOUT_NODE_WORD.leftIndex : GPU_LAYOUT_NODE_WORD.rightIndex)];
            const count = words[base + (left ? GPU_LAYOUT_NODE_WORD.leftCount : GPU_LAYOUT_NODE_WORD.rightCount)];
            const isLeaf = count > 0;
            readSlotBox(floats, slot, box);
            let inside = contained;
            if (!contained) {
                const score = order ? order(box) : undefined;
                const answer = Number(callbacks.intersectsBounds(box, isLeaf, score, depth));
                if (answer === BVH_NOT_INTERSECTED) continue;
                if (answer === BVH_CONTAINED) inside = 1;
            }
            if (isLeaf) {
                if (callbacks.intersectsRange(index, count, inside === 1, depth, box)) return true;
            } else {
                pushChildren(index, depth + 1, inside);
            }
        }
        return false;
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
        if (input) this.source = input;
        const source = this.source;
        if (!source) throw new Error('CpuMeshBvh.refit: the tree has no source geometry; pass one (refit({ positions, indices }))');
        const { positions, indices } = source;
        const sourceTriangles = indices ? Math.floor(indices.length / 3) : Math.floor(positions.length / 9);
        if (sourceTriangles !== this.triangleCount) throw new RangeError(`CpuMeshBvh.refit: the source has ${sourceTriangles} triangles, the tree ${this.triangleCount}`);
        const vertexCount = Math.floor(positions.length / 3);
        const boxes = new Float32Array(this.triangleCount * PRIMITIVE_BOX_FLOATS);
        const floats = this.triangleFloats;
        for (let position = 0; position < this.triangleCount; position++) {
            const base = position * TRIANGLE_WORDS;
            const triangle = this.triangleWords[base + GPU_LAYOUT_TRIANGLE_WORD.originalIndex];
            const at = position * PRIMITIVE_BOX_FLOATS;
            boxes[at] = boxes[at + 1] = boxes[at + 2] = Infinity;
            boxes[at + 4] = boxes[at + 5] = boxes[at + 6] = -Infinity;
            for (let corner = 0; corner < 3; corner++) {
                const vertex = indices ? indices[triangle * 3 + corner] : triangle * 3 + corner;
                if (vertex >= vertexCount) throw new RangeError(`CpuMeshBvh.refit: triangle ${triangle} uses vertex ${vertex}, past the ${vertexCount} positions`);
                const word = base + CORNER_WORDS[corner];
                for (let axis = 0; axis < 3; axis++) {
                    const value = positions[vertex * 3 + axis];
                    if (!Number.isFinite(value)) throw new RangeError(`CpuMeshBvh.refit: vertex ${vertex} has a non-finite position`);
                    floats[word + axis] = value;
                    const stored = floats[word + axis];
                    if (stored < boxes[at + axis]) boxes[at + axis] = stored;
                    if (stored > boxes[at + 4 + axis]) boxes[at + 4 + axis] = stored;
                }
            }
        }
        const { rootBox } = refitOnCpu(this.nodeFloats, this.nodeWords, 0, this.nodeCount, this.layout.parents, boxes);
        for (let i = 0; i < 6; i++) this.layout.bounds[i] = rootBox[i];
        this.topLevel = this.makeTopLevel();
        this.tracer = this.makeTracer();
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

    private makeTracer(): CpuTracer {
        return createCpuTracer(
            { nodes: this.layout.nodes, triangles: this.layout.triangles, topLevel: this.topLevel },
            { triangleFilter: (candidate) => this.passes(candidate), filterEveryTriangle: true },
        );
    }
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

/**
 * CPU raycasting for three meshes over this package's trees (Q1): drop-ins
 * for three-mesh-bvh's `computeBoundsTree`, `disposeBoundsTree`,
 * `acceleratedRaycast`, `MeshBVH` (as {@link BvhBoundsTree}) and
 * `getTriangleHitPointInfo`. The tree is the core `CpuMeshBvh`
 * (`src/query/cpu-mesh-bvh.ts`), so `/three` and the core share one walk:
 * `createCpuTracer`, the step-for-step port of the WGSL walk.
 *
 * ```ts
 * import { acceleratedRaycast, computeBoundsTree, disposeBoundsTree, prepareBoundsTrees } from '@pmndrs/bvh/three';
 * await prepareBoundsTrees();                     // loads tinybvh (WebAssembly) once
 * BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
 * BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
 * Mesh.prototype.raycast = acceleratedRaycast;
 * geometry.computeBoundsTree({ strategy: SAH, maxLeafSize: 10 }); // three-mesh-bvh's options
 * raycaster.firstHitOnly = true;                  // optional: only the closest hit
 * ```
 *
 * **What matches three's `Mesh.raycast`.** The same intersections: the
 * triangles in the draw range (with an array of materials, in the groups
 * whose material exists), the material's `side` (front faces only for
 * `FrontSide`, back faces only for `BackSide`), `raycaster.near`/`far` on the
 * world distance, and the same fields: `distance`, `point`, `object`, `face`
 * (`a`, `b`, `c`, local `normal`, `materialIndex`), `faceIndex`, `barycoord`,
 * `uv`, `uv1` and `normal` (interpolated, facing the ray) when the geometry
 * has them. `InstancedMesh` and `BatchedMesh` raycast through `Mesh.raycast`,
 * so they use it too (`instanceId` / `batchId` are added by three). Distances
 * agree to float rounding: the walk's watertight test runs on the
 * f32 corners in doubles; three's Möller-Trumbore runs in doubles too. A
 * `BatchedMesh` uses the tree of its geometry (`batchedMesh.geometry.computeBoundsTree()`):
 * three raycasts each instance through a stand-in mesh that shares the
 * geometry's attributes, and the tree is found by its position attribute.
 *
 * **Without `firstHitOnly`** every intersection is returned, as three does
 * (sorted by `Raycaster.intersectObject`): the walk's triangle filter collects
 * each hit and lets the ray through, so the walk visits every box the ray
 * crosses. That costs more than a closest hit; set `firstHitOnly` when one is
 * enough.
 *
 * **Falls back to three's raycast** for a mesh whose geometry has no tree
 * from {@link computeBoundsTree} (or one from another library), a
 * `SkinnedMesh`, or a mesh with morph targets (the tree holds the rest pose).
 *
 * **Ranges off a multiple of 3, overlapping groups.** three steps through a
 * range three indices at a time from its start, so a draw range or group that
 * starts at an index that isn't a multiple of 3 tests triangles made of
 * neighbouring triangles' corners, and one whose length isn't a multiple of 3
 * ends in a triangle that reads past it (`threeTriangleRanges`). Both are
 * matched: such a range is walked in a second tree over the triangles that
 * start at index 1 (or 2) mod 3, built on its first raycast with the tree's
 * build options (it needs tinybvh, as `computeBoundsTree` does; without it the
 * raycast falls back to three's) and refitted with the tree. Groups that
 * overlap are walked once each, as three tests them once per group.
 *
 * **Staleness.** The tree is built from the positions at {@link computeBoundsTree}
 * time: when the position attribute's version, or the index (its object or
 * version), changed since, the raycast falls back to three's own with a
 * one-time warning until `boundsTree.refit()` (same index) or
 * `computeBoundsTree` runs again.
 */

import {
    BackSide,
    Box3,
    DoubleSide,
    FrontSide,
    Mesh,
    Triangle,
    Vector2,
    Vector3,
    type BufferAttribute,
    type BufferGeometry,
    type Face,
    type Intersection,
    type InterleavedBufferAttribute,
    type Material,
    type Matrix4,
    type Raycaster,
} from 'three';
import type { GpuLayoutMeshBvh } from '../gpu-layout.ts';
import type { MeshInput } from '../mesh-input.ts';
import {
    barycoordOnTriangle,
    BVH_BACK_SIDE,
    BVH_DOUBLE_SIDE,
    BVH_FRONT_SIDE,
    clampLeafPrimitives,
    CpuMeshBvh,
    readVec3,
    type BvhRayIntersection,
    type BvhRayLike,
    type BvhRaycastOptions,
    type BvhShapecastCallbacks,
    type BvhBvhcastCallbacks,
    type BvhClosestPoint,
    type BvhClosestPointTarget,
    type BvhTriangleAccept,
    type BvhVec3Like,
    type CpuMeshBvhBuildOptions,
} from '../query/cpu-mesh-bvh.ts';
import type { BvhMatrix4Like } from '../query/shapes.ts';
import { ExtendedTriangle } from './shapes.ts';
import { deserializeLayout } from '../serialize.ts';
import { createTinyBvhBuilder, type TinyBvhBuilder } from '../tinybvh.ts';
import { invertAffineInto } from '../trace/top-level.ts';
import { positionsOf, threeTriangleRanges } from './scene.ts';

let sharedBuilder: TinyBvhBuilder | null = null;
let sharedBuilderLoading: Promise<TinyBvhBuilder> | null = null;

/** Loads tinybvh (WebAssembly) for {@link computeBoundsTree}; call once before the first one. */
export async function prepareBoundsTrees(): Promise<TinyBvhBuilder> {
    sharedBuilderLoading ??= createTinyBvhBuilder().then((builder) => (sharedBuilder = builder));
    return sharedBuilderLoading;
}

/** The tinybvh builder {@link prepareBoundsTrees} loaded, or null before it resolved. */
export function preparedBoundsTreeBuilder(): TinyBvhBuilder | null {
    return sharedBuilder;
}

/** three-mesh-bvh's split strategies, taken by {@link BvhBoundsTreeOptions.strategy}. Every one builds the same tree (see there). */
export const CENTER = 0;
export const AVERAGE = 1;
export const SAH = 2;
export type BvhSplitStrategy = typeof CENTER | typeof AVERAGE | typeof SAH;

/**
 * Options of {@link computeBoundsTree} and `new BvhBoundsTree(geometry, options)`:
 * three-mesh-bvh's `MeshBVH` options, mapped to ours where they map
 * (docs/three.md, "Build options").
 */
export interface BvhBoundsTreeOptions {
    /** The tinybvh builder; default the one {@link prepareBoundsTrees} loaded. */
    readonly builder?: TinyBvhBuilder;
    /**
     * Accepted for three-mesh-bvh code; every value builds the same tree:
     * tinybvh's default builder (binned SAH over a grid of cells, "SIMD
     * Bonsai"), whose trees have a lower SAH cost than a centre or average
     * split. There is no CENTER or AVERAGE builder here. Values other than 0, 1 and 2 throw.
     */
    readonly strategy?: number;
    /**
     * Largest leaf, in triangles; larger leaves are cut in half. Our layout
     * holds 1 to 8 triangles per leaf: a larger value (three-mesh-bvh's
     * default is 10) gives 8, and one below 1 gives 1. tinybvh's SAH makes
     * smaller leaves where they pay. Default 8.
     */
    readonly maxLeafSize?: number;
    /** three-mesh-bvh 0.9's name for {@link maxLeafSize}. Wins over the other two names. */
    readonly targetLeafSize?: number;
    /** three-mesh-bvh's deprecated name for {@link maxLeafSize}. */
    readonly maxLeafTris?: number;
    /**
     * Accepted, not honoured: tinybvh decides the depth (`boundsTree.depth`),
     * and a value below the built tree's depth logs a warning once. The ray
     * walk's stack holds 64 entries, and a raycast on a deeper tree throws a
     * `RangeError` (a tinybvh tree that deep takes a degenerate mesh). The box
     * walk (`traverse`) has no such limit.
     */
    readonly maxDepth?: number;
    /**
     * Sets `geometry.boundingBox` from the tree when the geometry has none.
     * Default true, as three-mesh-bvh. `boundsTree.refit()` keeps a box set
     * this way current; a box three computed (or you set) is left alone.
     */
    readonly setBoundingBox?: boolean;
    /**
     * Accepted and ignored. The tree never reorders the geometry's index (its
     * triangle records are copies carrying their original index), which is
     * what three-mesh-bvh's `indirect: true` does. `resolveTriangleIndex` maps a tree position.
     */
    readonly indirect?: boolean;
    /** Called with 0 before the build and 1 after it. tinybvh builds in one call and reports nothing between. */
    readonly onProgress?: (progress: number) => void;
    /** Accepted and ignored. */
    readonly verbose?: boolean;
    /** Accepted and ignored: no SharedArrayBuffer is needed (docs/PARITY.md, 4.8). */
    readonly useSharedArrayBuffer?: boolean;
    /** Not supported: the tree holds every triangle, and raycasts follow the draw range instead. Throws when set. */
    readonly range?: unknown;
    /**
     * A tree already built for this geometry, in our layout (from `generate`,
     * a worker pool or `deserializeLayout`): the build is skipped. It must
     * hold the geometry's triangles, numbered in index order.
     */
    readonly layout?: GpuLayoutMeshBvh;
    /**
     * The geometry as `layout` was built from it ({@link snapshotGeometry},
     * taken before the build started; `GenerateMeshBvhWorker` does this). The
     * tree counts as built from those versions, so edits made while the build
     * ran make it stale (raycasts fall back to three's until `refit`). Default:
     * the geometry as it is now.
     */
    readonly snapshot?: BvhGeometrySnapshot;
}

/** A geometry's triangles and the versions of its position attribute and index at one moment ({@link snapshotGeometry}). */
export interface BvhGeometrySnapshot {
    readonly input: MeshInput;
    readonly position: BufferAttribute | InterleavedBufferAttribute;
    readonly positionVersion: number;
    readonly index: BufferAttribute | null;
    readonly indexVersion: number;
}

/** The geometry's triangles ({@link meshInputFromGeometry}) and its position and index versions, now. */
export function snapshotGeometry(geometry: BufferGeometry): BvhGeometrySnapshot {
    const input = meshInputFromGeometry(geometry);
    const position = geometry.getAttribute('position') as BufferAttribute | InterleavedBufferAttribute;
    const index = geometry.index;
    return { input, position, positionVersion: positionVersionOf(position), index, indexVersion: index ? index.version : 0 };
}

/** Warned once that a `maxDepth` below a built tree's depth isn't honoured. */
let maxDepthWarned = false;

/** The core build options for three-mesh-bvh-style options (see {@link BvhBoundsTreeOptions}). Throws on a strategy or a range it can't take. */
export function buildOptionsFromThreeMeshBvh(options: BvhBoundsTreeOptions): CpuMeshBvhBuildOptions {
    if (options.strategy !== undefined && options.strategy !== CENTER && options.strategy !== AVERAGE && options.strategy !== SAH) {
        throw new RangeError(`@pmndrs/bvh/three: unknown strategy ${String(options.strategy)} (CENTER 0, AVERAGE 1, SAH 2)`);
    }
    if (options.range !== undefined && options.range !== null) {
        throw new Error('@pmndrs/bvh/three: the `range` option is not supported; build over the whole geometry (raycasts follow its draw range)');
    }
    const leaf = options.targetLeafSize ?? options.maxLeafSize ?? options.maxLeafTris;
    return {
        ...(leaf !== undefined ? { maxLeafPrimitives: clampLeafPrimitives(leaf) } : {}),
        ...(options.onProgress ? { onProgress: options.onProgress } : {}),
    };
}

/**
 * One walk of a raycast: ranges that start on the same index mod 3 (`shift`)
 * and don't overlap, so each triangle of the walked tree belongs to at most one.
 */
interface RaycastPass {
    /** The ranges' start index mod 3: 0 walks the tree itself, 1 and 2 its shifted trees. */
    readonly shift: number;
    /**
     * Per triangle of the walked tree: its range's position in `materials` + 1,
     * or 0 when outside every range. Null with one range: then `[first, end)`
     * holds the triangles of `materials[0]`.
     */
    readonly ranges: Uint16Array | null;
    readonly first: number;
    readonly end: number;
    /** The ranges as `[first, end)` triangles of the walked tree, ascending (they don't overlap). */
    readonly spans: readonly (readonly [number, number])[];
    /** The materials of the ranges; their `side` is read at each raycast. */
    readonly materials: (Material | null)[];
    readonly groupIndices: number[];
}

/** A hit of {@link BvhBoundsTree.raycastLocal}, in the geometry's space. */
export interface BvhLocalHit {
    t: number;
    /** three's `faceIndex`. */
    triangle: number;
    /** The index (vertex, without an index) of the triangle's first corner. */
    firstIndex: number;
    u: number;
    v: number;
    materialIndex: number;
}

/** What a raycast against one mesh needs to know about each candidate triangle. */
interface RaycastState {
    /** Usually one; more for ranges off a multiple of 3 or groups that overlap. */
    readonly passes: RaycastPass[];
    /** What it was made for: the draw range, groups (start, count, materialIndex) and, for an array, its materials. */
    readonly drawStart: number;
    readonly drawCount: number;
    readonly groups: number[];
    readonly arrayMaterials: (Material | undefined)[] | null;
}

/** States per material kept for different draw ranges (a BatchedMesh raycasts each geometry's range); past this the cache starts again. */
const MAX_STATES_PER_MATERIAL = 256;

/** A position attribute's version (an interleaved one's buffer's). */
export function positionVersionOf(attribute: BufferAttribute | InterleavedBufferAttribute): number {
    const interleaved = attribute as InterleavedBufferAttribute;
    return interleaved.isInterleavedBufferAttribute === true ? interleaved.data.version : (attribute as BufferAttribute).version;
}

/**
 * A geometry's triangles as the core takes them: the positions (the
 * attribute's own array when it is plain f32 xyz, else a compact copy) and
 * the index, cut to whole triangles.
 */
export function meshInputFromGeometry(geometry: BufferGeometry): MeshInput {
    const position = geometry.getAttribute('position') as BufferAttribute | InterleavedBufferAttribute | undefined;
    if (!position) throw new Error('@pmndrs/bvh/three: a bounds tree needs a position attribute');
    const index = geometry.index;
    const triangleCount = Math.floor((index ? index.count : position.count) / 3);
    let indices: Uint16Array | Uint32Array | null = null;
    if (index) {
        const array = index.array;
        indices =
            array instanceof Uint16Array || array instanceof Uint32Array
                ? array.subarray(0, triangleCount * 3)
                : Uint32Array.from({ length: triangleCount * 3 }, (_, i) => index.getX(i));
    }
    const positions = index ? positionsOf(position) : positionsOf(position, 0, triangleCount * 3);
    return { positions, indices };
}

/** Triangles to cut shifted trees from: a geometry's, or a {@link MeshInput}'s. */
interface TriangleSource {
    /** `x, y, z` per vertex when the positions are plain f32 (used in place); else null and read from `attribute`. */
    readonly positions: Float32Array | null;
    readonly attribute: BufferAttribute | InterleavedBufferAttribute | null;
    readonly vertexCount: number;
    readonly index: ArrayLike<number> | null;
    /** Index count, or the vertex count without an index. */
    readonly length: number;
}

function sourceOfGeometry(geometry: BufferGeometry): TriangleSource {
    const position = geometry.getAttribute('position') as BufferAttribute | InterleavedBufferAttribute;
    const plain = position as BufferAttribute;
    const isPlain = (position as InterleavedBufferAttribute).isInterleavedBufferAttribute !== true && plain.array instanceof Float32Array && plain.itemSize === 3 && !plain.normalized;
    const index = geometry.index;
    return {
        positions: isPlain ? (plain.array as Float32Array).subarray(0, position.count * 3) : null,
        attribute: isPlain ? null : position,
        vertexCount: position.count,
        index: index ? index.array : null,
        length: index ? index.count : position.count,
    };
}

function sourceOfInput(input: MeshInput): TriangleSource {
    const vertexCount = Math.floor(input.positions.length / 3);
    const index = input.indices ?? null;
    return { positions: input.positions, attribute: null, vertexCount, index, length: index ? index.length : vertexCount };
}

/** How many triangles start at index `shift` mod 3 (1 or 2) and end within `source`. */
function shiftedTriangleCount(length: number, shift: number): number {
    return Math.max(0, Math.floor((length - shift) / 3));
}

/**
 * The triangles of a source that start at index `shift` mod 3 (1 or 2):
 * shifted triangle `t` is indices `shift + 3t` to `shift + 3t + 2` (its three
 * `faceIndex` is `t`). `triangles` lists the ones to take (ascending; the
 * tree's triangle `i` is `triangles[i]`), or null for every one. Positions are
 * used in place when they are plain f32 and the triangles' vertices are one
 * run; otherwise only the vertices the triangles use are copied. Null when a
 * triangle isn't in the source (a cut `MeshInput`).
 */
function shiftedMeshInput(source: TriangleSource, shift: number, triangles: Uint32Array | null): MeshInput | null {
    const available = shiftedTriangleCount(source.length, shift);
    const count = triangles ? triangles.length : available;
    if (count === 0) return null;
    if (triangles && triangles[count - 1] >= available) return null;
    const at = (i: number) => (triangles ? triangles[i] : i);
    const contiguous = at(count - 1) - at(0) === count - 1;
    const read = (vertex: number, out: Float32Array, to: number) => {
        if (source.positions) {
            out[to] = source.positions[vertex * 3];
            out[to + 1] = source.positions[vertex * 3 + 1];
            out[to + 2] = source.positions[vertex * 3 + 2];
        } else {
            out[to] = source.attribute!.getX(vertex);
            out[to + 1] = source.attribute!.getY(vertex);
            out[to + 2] = source.attribute!.getZ(vertex);
        }
    };
    const index = source.index;
    if (!index) {
        const first = shift + at(0) * 3;
        if (contiguous && source.positions) return { positions: source.positions.subarray(first * 3, (first + count * 3) * 3), indices: null };
        const positions = new Float32Array(count * 9);
        for (let i = 0; i < count; i++) for (let corner = 0; corner < 3; corner++) read(shift + at(i) * 3 + corner, positions, (i * 3 + corner) * 3);
        return { positions, indices: null };
    }
    if (source.positions) {
        // The index's own values over the positions in place: no position copy.
        const first = shift + at(0) * 3;
        if (contiguous && (index instanceof Uint16Array || index instanceof Uint32Array)) return { positions: source.positions, indices: index.subarray(first, first + count * 3) };
        const indices = new Uint32Array(count * 3);
        for (let i = 0; i < count; i++) for (let corner = 0; corner < 3; corner++) indices[i * 3 + corner] = index[shift + at(i) * 3 + corner];
        return { positions: source.positions, indices };
    }
    // Positions that need converting: only the vertices these triangles use, renumbered in order of first use.
    const remap = new Map<number, number>();
    const indices = new Uint32Array(count * 3);
    for (let i = 0; i < count; i++) {
        for (let corner = 0; corner < 3; corner++) {
            const vertex = index[shift + at(i) * 3 + corner];
            let local = remap.get(vertex);
            if (local === undefined) remap.set(vertex, (local = remap.size));
            indices[i * 3 + corner] = local;
        }
    }
    const positions = new Float32Array(remap.size * 3);
    for (const [vertex, local] of remap) if (vertex < source.vertexCount) read(vertex, positions, local * 3);
    return { positions, indices };
}

/** A tree over shifted triangles (see {@link shiftedMeshInput}). */
interface ShiftedTree {
    readonly shift: number;
    readonly tree: CpuMeshBvh;
    /** The tree's triangle `i` is shifted triangle `triangles[i]`; null when it holds every shifted triangle, in order. */
    readonly triangles: Uint32Array | null;
}

/** Subset shifted trees kept per tree (draw ranges and groups); past this, they are made again. */
const MAX_SHIFTED_SUBSETS = 16;

/** three-mesh-bvh's `HitPointInfo`: what `closestPointToPoint` and `closestPointToGeometry` return. */
export type BvhClosestPointInfo = BvhClosestPoint<Vector3>;

/** A three-mesh-bvh `raycast` result: three's `Intersection` in the geometry's space, without `object`. */
export type BvhBoundsTreeHit = Omit<Intersection, 'object'>;

/**
 * A geometry's tree (three-mesh-bvh's `MeshBVH`): the core {@link CpuMeshBvh}
 * over all its triangles, in index order so a hit's triangle is three's
 * `faceIndex`, with three's types on top. `raycast` and `raycastFirst` take a
 * three `Ray` and a side, a material or an array of materials (by group), and
 * return three intersections in the geometry's space, as three-mesh-bvh's do.
 * {@link acceleratedRaycast} runs the same walk for `Mesh.raycast`.
 */
export class BvhBoundsTree extends CpuMeshBvh<BvhBoundsTreeHit> {
    readonly geometry: BufferGeometry;
    /** The position attribute and index the tree was built from (or last refitted to), and their versions then. */
    readonly position: BufferAttribute | InterleavedBufferAttribute;
    positionVersion: number;
    readonly index: BufferAttribute | null;
    readonly indexVersion: number;
    /** Warned once that the geometry changed since the build. */
    staleWarned = false;
    /** The `geometry.boundingBox` this tree set (`setBoundingBox`), which `refit` keeps current. */
    private ownBoundingBox: Box3 | null = null;
    private state: RaycastPass | null = null;
    /**
     * Trees over triangles starting at index 1 and 2 mod 3, made on first use
     * (`shiftedTree`): keyed `shift` for one over every such triangle, or
     * `shift:spans` for one over just a pass's ranges' triangles.
     */
    private readonly shifted = new Map<string, ShiftedTree>();
    /** While a pass walks a subset shifted tree: its triangle → shifted triangle. */
    private mapping: Uint32Array | null = null;
    private readonly buildOptions: CpuMeshBvhBuildOptions;
    private readonly builder: TinyBvhBuilder | null;
    private readonly rangeCache = new WeakMap<object, RaycastState[]>();
    /** The walk's acceptance test for the current state: inside a range, and on a side its material shows. */
    private readonly acceptState: BvhTriangleAccept = (local, frontFace) => {
        const state = this.state!;
        const triangle = this.mapping ? this.mapping[local] : local;
        const range = state.ranges ? (state.ranges[triangle] ?? 0) : triangle >= state.first && triangle < state.end ? 1 : 0;
        if (range === 0) return false;
        const side = state.materials[range - 1]?.side ?? FrontSide;
        if (side === FrontSide) return frontFace;
        if (side === BackSide) return !frontFace;
        return side === DoubleSide;
    };

    constructor(geometry: BufferGeometry, options: BvhBoundsTreeOptions = {}) {
        const buildOptions = buildOptionsFromThreeMeshBvh(options);
        const snapshot = options.snapshot ?? snapshotGeometry(geometry);
        const input = snapshot.input;
        const triangleCount = input.indices ? input.indices.length / 3 : input.positions.length / 9;
        let layout = options.layout;
        if (layout) {
            if (layout.triangleCount !== triangleCount) throw new RangeError(`@pmndrs/bvh/three: the tree has ${layout.triangleCount} triangles, the geometry ${triangleCount}`);
        } else {
            const builder = options.builder ?? sharedBuilder;
            if (!builder) throw new Error('@pmndrs/bvh/three: call `await prepareBoundsTrees()` once before computeBoundsTree (it loads tinybvh)');
            layout = CpuMeshBvh.buildLayout(input, builder, buildOptions);
        }
        super(layout, { source: input });
        this.buildOptions = buildOptions;
        this.builder = options.builder ?? null;
        this.geometry = geometry;
        this.position = snapshot.position;
        this.positionVersion = snapshot.positionVersion;
        this.index = snapshot.index;
        this.indexVersion = snapshot.indexVersion;
        if ((options.setBoundingBox ?? true) && !geometry.boundingBox) geometry.boundingBox = this.ownBoundingBox = this.getBoundingBox(new Box3());
        if (options.maxDepth !== undefined && options.maxDepth < this.depth && !maxDepthWarned) {
            maxDepthWarned = true;
            console.warn(`@pmndrs/bvh/three: maxDepth ${options.maxDepth} is not honoured; tinybvh built a tree of depth ${this.depth} (warned once)`);
        }
    }

    /**
     * Refits the tree to the geometry's current positions, keeping its shape
     * (three-mesh-bvh's `refit()`). Raycasts use the tree again afterwards.
     * The index must be the same object and version: after an index change,
     * call `computeBoundsTree` again. With `input`, refits to those arrays
     * instead (as the core does) and leaves the geometry's versions alone.
     */
    override refit(input?: MeshInput): void {
        if (input) {
            super.refit(input);
            this.refitShifted(sourceOfInput(input));
            this.updateOwnBoundingBox();
            return;
        }
        const position = this.geometry.getAttribute('position') as BufferAttribute | InterleavedBufferAttribute | undefined;
        if (position !== this.position || this.geometry.index !== this.index || (this.index ? this.index.version : 0) !== this.indexVersion) {
            throw new Error('BvhBoundsTree.refit: the position attribute or the index was replaced or changed; call computeBoundsTree again');
        }
        super.refit(meshInputFromGeometry(this.geometry));
        this.refitShifted(sourceOfGeometry(this.geometry));
        this.positionVersion = positionVersionOf(position);
        this.staleWarned = false;
        this.updateOwnBoundingBox();
    }

    /**
     * three-mesh-bvh's `shapecast` with three's types: boxes as a `Box3`,
     * triangles as `ExtendedTriangle` (a three `Triangle`), the triangle's
     * original index. The core's cast underneath (`CpuMeshBvh.shapecast`).
     */
    override shapecast(callbacks: BvhShapecastCallbacks<ExtendedTriangle, Box3>): boolean {
        const triangle = threeTrianglePool.pop() ?? new ExtendedTriangle();
        try {
            return this.shapecastWith(callbacks, triangle, new Box3());
        } finally {
            threeTrianglePool.push(triangle);
        }
    }

    /**
     * three-mesh-bvh's `bvhcast(otherBvh, matrixToLocal, { intersectsRanges, intersectsTriangles })`
     * with three's types: the triangles as `ExtendedTriangle`, the other's
     * moved into this space; offsets and indices are tree positions
     * (`resolveTriangleIndex` maps them, as with three-mesh-bvh's `indirect`).
     */
    override bvhcast(other: CpuMeshBvh<unknown>, matrixToLocal: Matrix4 | BvhMatrix4Like | null | undefined, callbacks: BvhBvhcastCallbacks<ExtendedTriangle>): boolean {
        const triangle1 = threeTrianglePool.pop() ?? new ExtendedTriangle();
        const triangle2 = threeTrianglePool.pop() ?? new ExtendedTriangle();
        try {
            return this.bvhcastWith(other, matrixToLocal, callbacks, triangle1, triangle2);
        } finally {
            threeTrianglePool.push(triangle1, triangle2);
        }
    }

    /**
     * three-mesh-bvh's `intersectsGeometry(geometry, geometryToBvh)`: against
     * the geometry's own tree when it has one of ours (`geometry.boundsTree`),
     * else against its triangles. A tree or `{ positions, indices }` pass too.
     */
    override intersectsGeometry(other: BufferGeometry | CpuMeshBvh<unknown> | MeshInput, geometryToBvh?: Matrix4 | BvhMatrix4Like | null): boolean {
        if (other instanceof CpuMeshBvh || !('isBufferGeometry' in other)) return super.intersectsGeometry(other, geometryToBvh);
        const tree = (other as unknown as WithBoundsTree).boundsTree;
        return super.intersectsGeometry(tree instanceof CpuMeshBvh ? tree : meshInputFromGeometry(other), geometryToBvh);
    }

    /**
     * three-mesh-bvh's `closestPointToPoint(point, target, minThreshold, maxThreshold)`:
     * the core's ({@link CpuMeshBvh.closestPointToPoint}), with the point a
     * three `Vector3` (a new one when `target` has none). `faceIndex` is three's.
     */
    override closestPointToPoint(point: Vector3 | BvhVec3Like, target: BvhClosestPointTarget<Vector3> = {}, minThreshold = 0, maxThreshold = Infinity): BvhClosestPointInfo | null {
        return super.closestPointToPoint(point, target, minThreshold, maxThreshold) as BvhClosestPointInfo | null;
    }

    /**
     * three-mesh-bvh's `closestPointToGeometry(geometry, geometryToBvh, target1, target2, minThreshold, maxThreshold)`:
     * the core's ({@link CpuMeshBvh.closestPointToGeometry}) against the
     * geometry's own tree when it has one of ours (`geometry.boundsTree`), else
     * against its triangles. A tree or `{ positions, indices }` pass too.
     * Points are three `Vector3`s; `target2` gets the other geometry's point,
     * in its space.
     */
    override closestPointToGeometry(
        other: BufferGeometry | CpuMeshBvh<unknown> | MeshInput,
        geometryToBvh?: Matrix4 | BvhMatrix4Like | null,
        target1: BvhClosestPointTarget<Vector3> = {},
        target2: BvhClosestPointTarget<Vector3> | null = {},
        minThreshold = 0,
        maxThreshold = Infinity,
    ): BvhClosestPointInfo | null {
        let input: CpuMeshBvh<unknown> | MeshInput;
        if (other instanceof CpuMeshBvh || !('isBufferGeometry' in other)) input = other;
        else {
            const tree = (other as unknown as WithBoundsTree).boundsTree;
            input = tree instanceof CpuMeshBvh ? tree : meshInputFromGeometry(other);
        }
        return super.closestPointToGeometry(input, geometryToBvh, target1, target2, minThreshold, maxThreshold) as BvhClosestPointInfo | null;
    }

    protected override createPoint(x: number, y: number, z: number): Vector3 {
        return new Vector3(x, y, z);
    }

    /** A `geometry.boundingBox` this tree set follows the refitted tree. */
    private updateOwnBoundingBox(): void {
        if (this.ownBoundingBox && this.geometry.boundingBox === this.ownBoundingBox) this.getBoundingBox(this.ownBoundingBox);
    }

    override raycast(ray: BvhRayLike, materialOrSide?: number | BvhRaycastOptions | Material | Material[], near?: number, far?: number): BvhBoundsTreeHit[] {
        if (Array.isArray(materialOrSide)) return this.raycastGroups(ray, materialOrSide, near ?? 0, far ?? Infinity, false);
        return super.raycast(ray, materialOrSide, near, far);
    }

    override raycastFirst(ray: BvhRayLike, materialOrSide?: number | BvhRaycastOptions | Material | Material[], near?: number, far?: number): BvhBoundsTreeHit | null {
        if (Array.isArray(materialOrSide)) return this.raycastGroups(ray, materialOrSide, near ?? 0, far ?? Infinity, true)[0] ?? null;
        return super.raycastFirst(ray, materialOrSide, near, far);
    }

    /** An array of materials: each group's triangles, with its material's side. */
    private raycastGroups(ray: BvhRayLike, materials: Material[], near: number, far: number, firstHitOnly: boolean): BvhBoundsTreeHit[] {
        const origin = new Vector3(...readVec3(ray.origin));
        const direction = new Vector3(...readVec3(ray.direction));
        const length = direction.length();
        if (!(length > 0)) return [];
        const hits = this.raycastLocal(
            [origin.x, origin.y, origin.z],
            [direction.x, direction.y, direction.z],
            Math.max(0, near) / length,
            far / length,
            materials,
            { start: 0, count: Infinity },
            firstHitOnly,
        );
        if (!hits) throw new Error('@pmndrs/bvh/three: a group starts off a multiple of 3; call `await prepareBoundsTrees()` so its triangles can get a tree');
        hits.sort((a, b) => a.t - b.t);
        return hits.map((hit) => ({
            distance: hit.t * length,
            point: origin.clone().addScaledVector(direction, hit.t),
            ...intersectionFields(this.geometry, hit.triangle, hit.firstIndex, hit.u, hit.v, direction, hit.materialIndex),
        }));
    }

    protected override createHit(hit: BvhRayIntersection, ray: { readonly origin: readonly number[]; readonly direction: readonly number[] }): BvhBoundsTreeHit {
        const direction = new Vector3(ray.direction[0], ray.direction[1], ray.direction[2]);
        const point = new Vector3(ray.origin[0], ray.origin[1], ray.origin[2]).addScaledVector(direction, hit.t);
        return { distance: hit.t * direction.length(), point, ...intersectionFields(this.geometry, hit.triangle, hit.triangle * 3, hit.u, hit.v, direction, 0) };
    }

    /**
     * Intersects a local-space ray (origin, direction: three's ray through the
     * inverse of the world matrix, not renormalized, so `t` is the world
     * distance) with the triangles three would test for `material`.
     * `firstHitOnly`: the closest only. Returns local hits, or null when a
     * range off a multiple of 3 needs a shifted tree and tinybvh isn't loaded.
     */
    raycastLocal(
        origin: readonly [number, number, number],
        direction: readonly [number, number, number],
        near: number,
        far: number,
        material: Material | Material[],
        drawRange: { readonly start: number; readonly count: number },
        firstHitOnly: boolean,
    ): BvhLocalHit[] | null {
        const state = this.stateFor(material, drawRange);
        if (state.passes.length === 1 && state.passes[0].shift === 0) return this.walkPass(this, state.passes[0], origin, direction, near, far, firstHitOnly);
        const out: BvhLocalHit[] = [];
        for (const pass of state.passes) {
            const shifted = pass.shift === 0 ? null : this.shiftedTree(pass);
            if (shifted === undefined) return null;
            if (pass.shift !== 0 && shifted === null) continue;
            for (const hit of this.walkPass(shifted ? shifted.tree : this, pass, origin, direction, near, far, firstHitOnly, shifted?.triangles ?? null)) {
                if (firstHitOnly && out.length > 0) {
                    if (hit.t >= out[0].t) continue;
                    out.length = 0;
                }
                out.push(hit);
            }
        }
        return out;
    }

    /** One pass of {@link raycastLocal}: its ranges' triangles in `tree`. */
    private walkPass(
        tree: CpuMeshBvh<unknown>,
        pass: RaycastPass,
        origin: readonly [number, number, number],
        direction: readonly [number, number, number],
        near: number,
        far: number,
        firstHitOnly: boolean,
        mapping: Uint32Array | null = null,
    ): BvhLocalHit[] {
        const shift = pass.shift;
        const shifted = (triangle: number) => (mapping ? mapping[triangle] : triangle);
        // A subset tree holds only its pass's triangles: with one range, every one is in it.
        if (!pass.ranges && (mapping !== null || (pass.first === 0 && pass.end >= tree.triangleCount))) {
            // One material over every triangle: its side culls natively in the walk, with no per-triangle callback.
            const side = pass.materials[0]?.side ?? FrontSide;
            if (side !== FrontSide && side !== BackSide && side !== DoubleSide) return [];
            const walkSide = side === FrontSide ? BVH_FRONT_SIDE : side === BackSide ? BVH_BACK_SIDE : BVH_DOUBLE_SIDE;
            const materialIndex = pass.groupIndices[0];
            return tree.intersectRay(origin, direction, near, far, firstHitOnly, null, walkSide).map((hit) => {
                const triangle = shifted(hit.triangle);
                return { t: hit.t, triangle, firstIndex: shift + triangle * 3, u: hit.u, v: hit.v, materialIndex };
            });
        }
        const previous = this.state;
        const previousMapping = this.mapping;
        this.state = pass;
        this.mapping = mapping;
        try {
            return tree.intersectRay(origin, direction, near, far, firstHitOnly, this.acceptState).map((hit) => {
                const triangle = shifted(hit.triangle);
                return {
                    t: hit.t,
                    triangle,
                    firstIndex: shift + triangle * 3,
                    u: hit.u,
                    v: hit.v,
                    materialIndex: pass.groupIndices[pass.ranges ? pass.ranges[triangle] - 1 : 0],
                };
            });
        } finally {
            this.state = previous;
            this.mapping = previousMapping;
        }
    }

    /**
     * The tree a shifted pass (ranges starting at index 1 or 2 mod 3) walks,
     * built on first use with this tree's build options: over just the pass's
     * ranges' triangles, or over every triangle of its shift when the ranges
     * hold half of them or more (then any pass of that shift uses it). Null
     * when the ranges hold no triangle, undefined when tinybvh isn't loaded.
     */
    private shiftedTree(pass: RaycastPass): ShiftedTree | null | undefined {
        const shift = pass.shift;
        const whole = this.shifted.get(`${shift}`);
        if (whole) return whole;
        const available = shiftedTriangleCount(this.geometry.index ? this.geometry.index.count : this.geometry.getAttribute('position').count, shift);
        const spans = pass.spans.map(([a, b]) => [Math.min(a, available), Math.min(b, available)] as const).filter(([a, b]) => b > a);
        const count = spans.reduce((sum, [a, b]) => sum + b - a, 0);
        if (count === 0) return null;
        const everyTriangle = count * 2 >= available;
        const key = everyTriangle ? `${shift}` : `${shift}:${spans.map(([a, b]) => `${a}-${b}`).join(',')}`;
        const made = this.shifted.get(key);
        if (made) return made;
        const builder = this.builder ?? sharedBuilder;
        if (!builder) return undefined;
        let triangles: Uint32Array | null = null;
        if (!everyTriangle) {
            triangles = new Uint32Array(count);
            let at = 0;
            for (const [a, b] of spans) for (let t = a; t < b; t++) triangles[at++] = t;
        }
        const input = shiftedMeshInput(sourceOfGeometry(this.geometry), shift, triangles);
        if (!input) return null;
        if (!everyTriangle && this.shifted.size >= MAX_SHIFTED_SUBSETS) {
            for (const [made, tree] of this.shifted) if (tree.triangles) this.shifted.delete(made);
        }
        const tree: ShiftedTree = { shift, tree: CpuMeshBvh.build(input, builder, this.buildOptions), triangles };
        this.shifted.set(key, tree);
        return tree;
    }

    /** Refits the shifted trees from `source` (the tree's own refit input); one whose triangles `source` cuts off is dropped, made again on its next use. */
    private refitShifted(source: TriangleSource): void {
        for (const [key, shifted] of [...this.shifted]) {
            const input = shiftedMeshInput(source, shifted.shift, shifted.triangles);
            if (input && (input.indices ? input.indices.length : input.positions.length / 3) === shifted.tree.triangleCount * 3) shifted.tree.refit(input);
            else this.shifted.delete(key);
        }
    }

    /** True when the geometry's positions and index are the ones the tree was built from (or refitted to). */
    current(): boolean {
        const position = this.geometry.getAttribute('position') as BufferAttribute | InterleavedBufferAttribute | undefined;
        const index = this.geometry.index;
        return position === this.position && positionVersionOf(position) === this.positionVersion && index === this.index && (index ? index.version : 0) === this.indexVersion;
    }

    /**
     * Which triangles count for this material and draw range: cached per
     * material and draw range, checked against the groups and (for an array)
     * the materials it holds. Each material's `side` is read live.
     */
    private stateFor(material: Material | Material[], drawRange: { readonly start: number; readonly count: number }): RaycastState {
        let states = this.rangeCache.get(material);
        if (states) {
            for (const state of states) if (this.matches(state, material, drawRange)) return state;
        } else {
            this.rangeCache.set(material, (states = []));
        }
        if (states.length >= MAX_STATES_PER_MATERIAL) states.length = 0;
        const state = this.makeState(material, drawRange);
        states.push(state);
        return state;
    }

    private matches(state: RaycastState, material: Material | Material[], drawRange: { readonly start: number; readonly count: number }): boolean {
        if (state.drawStart !== drawRange.start || state.drawCount !== drawRange.count) return false;
        if (!state.arrayMaterials) return true; // one material: groups don't matter
        const array = material as Material[];
        if (array.length !== state.arrayMaterials.length) return false;
        for (let i = 0; i < array.length; i++) if (array[i] !== state.arrayMaterials[i]) return false;
        const groups = this.geometry.groups;
        if (groups.length * 3 !== state.groups.length) return false;
        for (let i = 0; i < groups.length; i++) {
            const group = groups[i];
            if (group.start !== state.groups[i * 3] || group.count !== state.groups[i * 3 + 1] || (group.materialIndex ?? 0) !== state.groups[i * 3 + 2]) return false;
        }
        return true;
    }

    private makeState(material: Material | Material[], drawRange: { readonly start: number; readonly count: number }): RaycastState {
        // Each range goes into the first pass with its shift whose ranges it doesn't overlap.
        const passes: { shift: number; spans: [number, number][]; materials: (Material | null)[]; groupIndices: number[] }[] = [];
        for (const range of threeTriangleRanges(this.geometry, material, drawRange)) {
            const shift = range.start % 3;
            const first = (range.start - shift) / 3;
            const end = first + range.count / 3;
            let pass = passes.find((candidate) => candidate.shift === shift && candidate.spans.every(([a, b]) => end <= a || first >= b));
            if (!pass) passes.push((pass = { shift, spans: [], materials: [], groupIndices: [] }));
            pass.spans.push([first, end]);
            pass.materials.push(range.threeMaterial);
            pass.groupIndices.push(range.group >= 0 ? range.materialIndex : 0);
        }
        const isArray = Array.isArray(material);
        return {
            passes: passes.map(({ shift, spans, materials, groupIndices }): RaycastPass => {
                let ranges: Uint16Array | null = null;
                if (spans.length > 1) {
                    const fill = new Uint16Array(Math.max(...spans.map(([, end]) => end)));
                    spans.forEach(([first, end], i) => fill.fill(i + 1, first, end));
                    ranges = fill;
                }
                const sorted = spans.map(([a, b]) => [a, b] as const).sort((a, b) => a[0] - b[0]);
                return { shift, ranges, first: spans.length === 1 ? spans[0][0] : 0, end: spans.length === 1 ? spans[0][1] : 0, spans: sorted, materials, groupIndices };
            }),
            drawStart: drawRange.start,
            drawCount: drawRange.count,
            groups: isArray ? this.geometry.groups.flatMap((group) => [group.start, group.count, group.materialIndex ?? 0]) : [],
            arrayMaterials: isArray ? [...material] : null,
        };
    }
}

const vA = new Vector3();
const vB = new Vector3();
const vC = new Vector3();

/** The vertex indices of the triangle whose first corner is at index `first` (three's `face.a`, `b`, `c`). */
function cornersOf(geometry: BufferGeometry, first: number): [number, number, number] {
    const index = geometry.index;
    return index ? [index.getX(first), index.getX(first + 1), index.getX(first + 2)] : [first, first + 1, first + 2];
}

/**
 * The fields three's `Mesh.raycast` gives a hit on `triangle` at barycentrics
 * `u`, `v` (corners 1 and 2): `face` (corners, face normal, material index),
 * `faceIndex`, `barycoord`, and `uv`, `uv1` and `normal` (interpolated,
 * turned against `localDirection`) when the geometry has them.
 */
export function intersectionFields(
    geometry: BufferGeometry,
    triangle: number,
    firstIndex: number,
    u: number,
    v: number,
    localDirection: Vector3,
    materialIndex: number,
): Pick<Intersection, 'face' | 'faceIndex' | 'barycoord' | 'uv' | 'uv1' | 'normal'> {
    const { position, uv, uv1, normal } = geometry.attributes;
    const [a, b, c] = cornersOf(geometry, firstIndex);
    const barycoord = new Vector3(1 - u - v, u, v);
    vA.fromBufferAttribute(position, a);
    vB.fromBufferAttribute(position, b);
    vC.fromBufferAttribute(position, c);
    const face: Face = { a, b, c, normal: Triangle.getNormal(vA, vB, vC, new Vector3()), materialIndex };
    const fields: Pick<Intersection, 'face' | 'faceIndex' | 'barycoord' | 'uv' | 'uv1' | 'normal'> = { face, faceIndex: triangle, barycoord };
    if (uv) fields.uv = Triangle.getInterpolatedAttribute(uv as BufferAttribute, a, b, c, barycoord, new Vector2()) as Vector2;
    if (uv1) fields.uv1 = Triangle.getInterpolatedAttribute(uv1 as BufferAttribute, a, b, c, barycoord, new Vector2()) as Vector2;
    if (normal) {
        const n = Triangle.getInterpolatedAttribute(normal as BufferAttribute, a, b, c, barycoord, new Vector3()) as Vector3;
        if (n.dot(localDirection) > 0) n.multiplyScalar(-1);
        fields.normal = n;
    }
    return fields;
}

/** What {@link getTriangleHitPointInfo} returns (three-mesh-bvh's `HitPointInfo`). */
export interface BvhHitPointInfo {
    face: Face;
    uv: Vector2 | null;
    barycoord: Vector3;
}

/** A target for {@link getTriangleHitPointInfo}: any of its fields may be missing, and are then made. */
export interface BvhHitPointInfoTarget {
    face?: Partial<Face>;
    uv?: Vector2 | null;
    barycoord?: Vector3;
}

/**
 * Where `point` (in the geometry's space) lies on `triangleIndex`
 * (three-mesh-bvh's `getTriangleHitPointInfo`): its face (vertex indices, face
 * normal, material index from the groups), its uv when the geometry has one,
 * and its barycentric weights. The weights are the core's
 * (`barycoordOnTriangle`, the point projected on the plane); a degenerate
 * triangle gets zero weights.
 */
export function getTriangleHitPointInfo(point: Vector3, geometry: BufferGeometry, triangleIndex: number, target?: BvhHitPointInfoTarget): BvhHitPointInfo {
    const [a, b, c] = cornersOf(geometry, triangleIndex * 3);
    const position = geometry.attributes.position;
    vA.fromBufferAttribute(position, a);
    vB.fromBufferAttribute(position, b);
    vC.fromBufferAttribute(position, c);
    const weights = barycoordOnTriangle([point.x, point.y, point.z], [vA.x, vA.y, vA.z], [vB.x, vB.y, vB.z], [vC.x, vC.y, vC.z]) ?? [0, 0, 0];
    let materialIndex = 0;
    for (const group of geometry.groups) {
        if (triangleIndex * 3 >= group.start && triangleIndex * 3 < group.start + group.count) {
            materialIndex = group.materialIndex ?? 0;
            break;
        }
    }
    // As three-mesh-bvh: a target's missing fields are made, and its uv is kept when the geometry has none.
    const info = (target ?? { uv: null }) as BvhHitPointInfo;
    info.face ??= { a: 0, b: 0, c: 0, normal: new Vector3(), materialIndex: 0 };
    info.face.normal ??= new Vector3();
    info.barycoord ??= new Vector3();
    info.face.a = a;
    info.face.b = b;
    info.face.c = c;
    info.face.materialIndex = materialIndex;
    Triangle.getNormal(vA, vB, vC, info.face.normal);
    info.barycoord.set(weights[0], weights[1], weights[2]);
    const uv = geometry.attributes.uv;
    if (uv) info.uv = Triangle.getInterpolatedAttribute(uv as BufferAttribute, a, b, c, info.barycoord, info.uv ?? new Vector2()) as Vector2;
    else if (info.uv === undefined) info.uv = null;
    return info;
}

/**
 * Reads bytes from `boundsTree.serialize()` back into a tree for `geometry`,
 * the geometry it was built from (three-mesh-bvh's `MeshBVH.deserialize`; the
 * bytes are our layout, not three-mesh-bvh's format). Doesn't set
 * `geometry.boundsTree`; `setBoundsTree` does.
 */
export function deserializeBoundsTree(bytes: ArrayBuffer | ArrayBufferView, geometry: BufferGeometry, options: BvhBoundsTreeOptions = {}): BvhBoundsTree {
    return new BvhBoundsTree(geometry, { ...options, layout: deserializeLayout(bytes) });
}

/** Scratch triangles for {@link BvhBoundsTree.shapecast} (casts may nest). */
const threeTrianglePool: ExtendedTriangle[] = [];

/** `geometry.boundsTree`, whoever made it (three-mesh-bvh types it as its own). */
type WithBoundsTree = { boundsTree?: unknown };

/**
 * Trees by position attribute: a BatchedMesh raycasts each instance through
 * a stand-in mesh whose geometry shares the batched geometry's attributes.
 */
const treesByPosition = new WeakMap<object, BvhBoundsTree>();

/** Makes `tree` the geometry's `boundsTree` and lets BatchedMesh stand-ins find it. */
export function setBoundsTree(geometry: BufferGeometry, tree: BvhBoundsTree): void {
    (geometry as unknown as WithBoundsTree).boundsTree = tree;
    treesByPosition.set(tree.position, tree);
}

/**
 * `BufferGeometry.prototype.computeBoundsTree`: builds `this.boundsTree`, a
 * {@link BvhBoundsTree} (the core `CpuMeshBvh` with three's types). Takes
 * three-mesh-bvh's options ({@link BvhBoundsTreeOptions}).
 */
export function computeBoundsTree(this: BufferGeometry, options: BvhBoundsTreeOptions = {}): BvhBoundsTree {
    const tree = new BvhBoundsTree(this, options);
    setBoundsTree(this, tree);
    return tree;
}

/** `BufferGeometry.prototype.disposeBoundsTree`: drops `this.boundsTree`. */
export function disposeBoundsTree(this: BufferGeometry): void {
    const tree = (this as unknown as WithBoundsTree).boundsTree;
    if (tree instanceof BvhBoundsTree && treesByPosition.get(tree.position) === tree) treesByPosition.delete(tree.position);
    (this as unknown as WithBoundsTree).boundsTree = undefined;
}

/**
 * The tree for a geometry: its own, or (a BatchedMesh's stand-in, one
 * material) the tree of the geometry whose position and index it shares.
 * Groups are the tree's geometry's, so a shared tree never serves an array of materials.
 */
function treeOf(geometry: BufferGeometry, material: Material | Material[]): BvhBoundsTree | null {
    const own = (geometry as unknown as WithBoundsTree).boundsTree;
    if (own instanceof BvhBoundsTree) return own.geometry === geometry ? own : null;
    if (own !== undefined || Array.isArray(material)) return null; // another library's tree, or groups that may differ
    const position = geometry.attributes.position;
    const shared = position ? treesByPosition.get(position) : undefined;
    return shared && shared.index === geometry.index ? shared : null;
}

/** A tree whose geometry changed since its build: three's raycast is used until `refit` or `computeBoundsTree` runs again. */
function stale(tree: BvhBoundsTree): boolean {
    if (tree.current()) return false;
    if (!tree.staleWarned) {
        tree.staleWarned = true;
        console.warn(
            "@pmndrs/bvh/three: a geometry's positions or index changed since computeBoundsTree; raycasts against it use three's own raycast until boundsTree.refit() or computeBoundsTree runs again",
        );
    }
    return true;
}

// three's own raycast, captured before anyone replaces it, for the fallback.
const threeRaycast = Mesh.prototype.raycast;
/** World-to-object rows (3 x 4, row-major), as the top level stores them. */
const rows = new Float64Array(12);
const worldPoint = new Vector3();
const localPoint = new Vector3();

/** `Mesh.prototype.raycast` over the geometry's {@link BvhBoundsTree}, or three's own without one. */
export function acceleratedRaycast(this: Mesh, raycaster: Raycaster, intersects: Intersection[]): void {
    const tree = this.material === undefined ? null : treeOf(this.geometry, this.material);
    const morphs = this.morphTargetInfluences !== undefined && Object.keys(this.geometry.morphAttributes).length > 0;
    if (!tree || (this as unknown as { isSkinnedMesh?: boolean }).isSkinnedMesh || morphs || stale(tree)) {
        threeRaycast.call(this, raycaster, intersects);
        return;
    }
    const material = this.material;
    if (material === undefined) return;
    if (invertAffineInto(this.matrixWorld.elements, rows) === 0) return;
    const { origin, direction } = raycaster.ray;
    const m = rows;
    const localOrigin: [number, number, number] = [
        m[0] * origin.x + m[1] * origin.y + m[2] * origin.z + m[3],
        m[4] * origin.x + m[5] * origin.y + m[6] * origin.z + m[7],
        m[8] * origin.x + m[9] * origin.y + m[10] * origin.z + m[11],
    ];
    const localDirection: [number, number, number] = [
        m[0] * direction.x + m[1] * direction.y + m[2] * direction.z,
        m[4] * direction.x + m[5] * direction.y + m[6] * direction.z,
        m[8] * direction.x + m[9] * direction.y + m[10] * direction.z,
    ];
    // o_local + t * d_local is the world point o + t * d (d_local is not renormalized), so t is the
    // world distance over |d| (1 for three's normalized rays).
    const toT = 1 / direction.length();
    const firstHitOnly = raycaster.firstHitOnly === true;
    const hits = tree.raycastLocal(localOrigin, localDirection, raycaster.near * toT, raycaster.far * toT, material, this.geometry.drawRange, firstHitOnly);
    if (!hits) {
        threeRaycast.call(this, raycaster, intersects); // a range off a multiple of 3, and tinybvh isn't loaded for its tree
        return;
    }
    if (hits.length === 0) return;
    const localDir = new Vector3(...localDirection);
    for (const hit of hits) {
        localPoint.set(localOrigin[0], localOrigin[1], localOrigin[2]).addScaledVector(localDir, hit.t);
        worldPoint.copy(localPoint).applyMatrix4(this.matrixWorld);
        const distance = raycaster.ray.origin.distanceTo(worldPoint);
        if (distance < raycaster.near || distance > raycaster.far) continue;
        intersects.push({ distance, point: worldPoint.clone(), object: this, ...intersectionFields(this.geometry, hit.triangle, hit.firstIndex, hit.u, hit.v, localDir, hit.materialIndex) });
    }
}

/**
 * three's types with what the drop-ins add, in three-mesh-bvh's shape, so
 * `geometry.computeBoundsTree()`, `geometry.boundsTree` and
 * `raycaster.firstHitOnly` typecheck once `@pmndrs/bvh/three` is imported.
 * three-mesh-bvh declares the same members with its own types: a program
 * that imports both (while migrating, or comparing the two as our tests do)
 * gets two declarations of `boundsTree`, `computeBoundsTree` and
 * `disposeBoundsTree` with different types, which TypeScript refuses
 * (TS2717). The `@ts-ignore`s keep such a program of our sources compiling
 * (the tests compare against three-mesh-bvh); the first declaration read
 * gives the type. The emitted `.d.ts` drops them: a consumer importing both
 * needs `skipLibCheck`.
 */
/* eslint-disable @typescript-eslint/ban-ts-comment -- @ts-expect-error would fail in a program without three-mesh-bvh */
declare module 'three' {
    interface BufferGeometry {
        /** The tree {@link computeBoundsTree} (or {@link setBoundsTree}) gave this geometry. */
        // @ts-ignore TS2717 next to three-mesh-bvh's augmentation (see above)
        boundsTree?: BvhBoundsTree;
        /** Set it to {@link computeBoundsTree}: `BufferGeometry.prototype.computeBoundsTree = computeBoundsTree`. */
        // @ts-ignore TS2717 next to three-mesh-bvh's augmentation
        computeBoundsTree: typeof computeBoundsTree;
        /** Set it to {@link disposeBoundsTree}. */
        // @ts-ignore TS2717 next to three-mesh-bvh's augmentation
        disposeBoundsTree: typeof disposeBoundsTree;
    }

    interface Raycaster {
        /** {@link acceleratedRaycast} (and `raycastScene`): only the closest hit. Default false, every hit. */
        firstHitOnly?: boolean;
    }
}
/* eslint-enable @typescript-eslint/ban-ts-comment */

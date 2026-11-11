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
 * **Not matched:** a draw range or group that starts at an index that isn't a
 * multiple of 3 (three then tests triangles made of neighbouring triangles'
 * corners; the tree only holds the authored ones), and a trailing partial
 * triangle. The tree is built from the positions at {@link computeBoundsTree}
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
    type Raycaster,
} from 'three';
import type { GpuLayoutMeshBvh } from '../gpu-layout.ts';
import type { MeshInput } from '../mesh-input.ts';
import { barycoordOnTriangle, clampLeafPrimitives, CpuMeshBvh, type BvhRayIntersection, type BvhRayLike, type BvhRaycastOptions, type BvhTriangleAccept, type CpuMeshBvhBuildOptions } from '../query/cpu-mesh-bvh.ts';
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
     * Accepted, not honoured: tinybvh decides the depth. The ray walk's stack
     * holds 64 entries, and a raycast on a deeper tree throws a `RangeError`
     * (a tinybvh tree that deep takes a degenerate mesh). `boundsTree.layout.maxDepth`
     * has the depth. The box walk (`traverse`) has no limit.
     */
    readonly maxDepth?: number;
    /**
     * true: sets `geometry.boundingBox` from the tree when the geometry has
     * none. Default **false**, unlike three-mesh-bvh (true): three's own
     * raycast culls by that box, and a box set here goes stale when the
     * positions are edited (three only computes it when asked).
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
}

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

/** What a raycast against one mesh needs to know about each candidate triangle. */
interface RaycastState {
    /**
     * Per triangle: its group's position in `materials` + 1, or 0 when outside
     * every range. Null with one range: then `[first, end)` holds the triangles of `materials[0]`.
     */
    readonly ranges: Uint16Array | null;
    readonly first: number;
    readonly end: number;
    /** The materials of the ranges; their `side` is read at each raycast. */
    readonly materials: (Material | null)[];
    readonly groupIndices: number[];
    /** What it was made for: the draw range, groups (start, count, materialIndex) and, for an array, its materials. */
    readonly drawStart: number;
    readonly drawCount: number;
    readonly groups: number[];
    readonly arrayMaterials: (Material | undefined)[] | null;
}

/** States per material kept for different draw ranges (a BatchedMesh raycasts each geometry's range); past this the cache starts again. */
const MAX_STATES_PER_MATERIAL = 256;

function positionVersionOf(attribute: BufferAttribute | InterleavedBufferAttribute): number {
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
    private state: RaycastState | null = null;
    private readonly rangeCache = new WeakMap<object, RaycastState[]>();
    /** The walk's acceptance test for the current state: inside a range, and on a side its material shows. */
    private readonly acceptState: BvhTriangleAccept = (triangle, frontFace) => {
        const state = this.state!;
        const range = state.ranges ? state.ranges[triangle] : triangle >= state.first && triangle < state.end ? 1 : 0;
        if (range === 0) return false;
        const side = state.materials[range - 1]?.side ?? FrontSide;
        if (side === FrontSide) return frontFace;
        if (side === BackSide) return !frontFace;
        return side === DoubleSide;
    };

    constructor(geometry: BufferGeometry, options: BvhBoundsTreeOptions = {}) {
        const buildOptions = buildOptionsFromThreeMeshBvh(options);
        const input = meshInputFromGeometry(geometry);
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
        const position = geometry.getAttribute('position') as BufferAttribute | InterleavedBufferAttribute;
        this.geometry = geometry;
        this.position = position;
        this.positionVersion = positionVersionOf(position);
        this.index = geometry.index;
        this.indexVersion = geometry.index ? geometry.index.version : 0;
        if (options.setBoundingBox === true && !geometry.boundingBox) geometry.boundingBox = this.getBoundingBox(new Box3());
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
            return;
        }
        const position = this.geometry.getAttribute('position') as BufferAttribute | InterleavedBufferAttribute | undefined;
        if (position !== this.position || this.geometry.index !== this.index || (this.index ? this.index.version : 0) !== this.indexVersion) {
            throw new Error('BvhBoundsTree.refit: the position attribute or the index was replaced or changed; call computeBoundsTree again');
        }
        super.refit(meshInputFromGeometry(this.geometry));
        this.positionVersion = positionVersionOf(position);
        this.staleWarned = false;
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
        const origin = new Vector3().copy(ray.origin as Vector3);
        const direction = new Vector3().copy(ray.direction as Vector3);
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
        hits.sort((a, b) => a.t - b.t);
        return hits.map((hit) => ({
            distance: hit.t * length,
            point: origin.clone().addScaledVector(direction, hit.t),
            ...intersectionFields(this.geometry, hit.triangle, hit.u, hit.v, direction, hit.materialIndex),
        }));
    }

    protected override createHit(hit: BvhRayIntersection, ray: { readonly origin: readonly number[]; readonly direction: readonly number[] }): BvhBoundsTreeHit {
        const direction = new Vector3(ray.direction[0], ray.direction[1], ray.direction[2]);
        const point = new Vector3(ray.origin[0], ray.origin[1], ray.origin[2]).addScaledVector(direction, hit.t);
        return { distance: hit.t * direction.length(), point, ...intersectionFields(this.geometry, hit.triangle, hit.u, hit.v, direction, 0) };
    }

    /**
     * Intersects a local-space ray (origin, direction: three's ray through the
     * inverse of the world matrix, not renormalized, so `t` is the world
     * distance) with the triangles three would test for `material`.
     * `firstHitOnly`: the closest only. Returns local hits.
     */
    raycastLocal(
        origin: readonly [number, number, number],
        direction: readonly [number, number, number],
        near: number,
        far: number,
        material: Material | Material[],
        drawRange: { readonly start: number; readonly count: number },
        firstHitOnly: boolean,
    ): { t: number; triangle: number; u: number; v: number; materialIndex: number }[] {
        const state = this.stateFor(material, drawRange);
        const previous = this.state;
        this.state = state;
        try {
            return this.intersectRay(origin, direction, near, far, firstHitOnly, this.acceptState).map((hit) => ({
                t: hit.t,
                triangle: hit.triangle,
                u: hit.u,
                v: hit.v,
                materialIndex: state.groupIndices[state.ranges ? state.ranges[hit.triangle] - 1 : 0],
            }));
        } finally {
            this.state = previous;
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
        const materials: (Material | null)[] = [];
        const groupIndices: number[] = [];
        const spans: [number, number][] = [];
        for (const range of threeTriangleRanges(this.geometry, material, drawRange)) {
            if (range.start % 3 !== 0) continue; // not authored triangles (see the file comment)
            materials.push(range.threeMaterial);
            groupIndices.push(range.group >= 0 ? range.materialIndex : 0);
            spans.push([range.start / 3, Math.min(this.triangleCount, (range.start + range.count) / 3)]);
        }
        let ranges: Uint16Array | null = null;
        if (spans.length > 1) {
            const fill = new Uint16Array(this.triangleCount);
            spans.forEach(([first, end], i) => fill.fill(i + 1, first, end));
            ranges = fill;
        }
        const isArray = Array.isArray(material);
        return {
            ranges,
            first: spans.length === 1 ? spans[0][0] : 0,
            end: spans.length === 1 ? spans[0][1] : 0,
            materials,
            groupIndices,
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

/** The vertex indices of a triangle (three's `face.a`, `b`, `c`). */
function cornersOf(geometry: BufferGeometry, triangle: number): [number, number, number] {
    const index = geometry.index;
    return index ? [index.getX(triangle * 3), index.getX(triangle * 3 + 1), index.getX(triangle * 3 + 2)] : [triangle * 3, triangle * 3 + 1, triangle * 3 + 2];
}

/**
 * The fields three's `Mesh.raycast` gives a hit on `triangle` at barycentrics
 * `u`, `v` (corners 1 and 2): `face` (corners, face normal, material index),
 * `faceIndex`, `barycoord`, and `uv`, `uv1` and `normal` (interpolated,
 * turned against `localDirection`) when the geometry has them.
 */
function intersectionFields(
    geometry: BufferGeometry,
    triangle: number,
    u: number,
    v: number,
    localDirection: Vector3,
    materialIndex: number,
): Pick<Intersection, 'face' | 'faceIndex' | 'barycoord' | 'uv' | 'uv1' | 'normal'> {
    const { position, uv, uv1, normal } = geometry.attributes;
    const [a, b, c] = cornersOf(geometry, triangle);
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

/**
 * Where `point` (in the geometry's space) lies on `triangleIndex`
 * (three-mesh-bvh's `getTriangleHitPointInfo`): its face (vertex indices, face
 * normal, material index from the groups), its uv when the geometry has one,
 * and its barycentric weights. The weights are the core's
 * (`barycoordOnTriangle`, the point projected on the plane); a degenerate
 * triangle gets zero weights.
 */
export function getTriangleHitPointInfo(point: Vector3, geometry: BufferGeometry, triangleIndex: number, target?: BvhHitPointInfo): BvhHitPointInfo {
    const [a, b, c] = cornersOf(geometry, triangleIndex);
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
    const info = target ?? { face: { a: 0, b: 0, c: 0, normal: new Vector3(), materialIndex: 0 }, uv: null, barycoord: new Vector3() };
    info.face.a = a;
    info.face.b = b;
    info.face.c = c;
    info.face.materialIndex = materialIndex;
    Triangle.getNormal(vA, vB, vC, info.face.normal);
    info.barycoord.set(weights[0], weights[1], weights[2]);
    const uv = geometry.attributes.uv;
    if (uv) info.uv = Triangle.getInterpolatedAttribute(uv as BufferAttribute, a, b, c, info.barycoord, info.uv ?? new Vector2()) as Vector2;
    else info.uv = null;
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
    const firstHitOnly = (raycaster as Raycaster & { firstHitOnly?: boolean }).firstHitOnly === true;
    const hits = tree.raycastLocal(localOrigin, localDirection, raycaster.near * toT, raycaster.far * toT, material, this.geometry.drawRange, firstHitOnly);
    if (hits.length === 0) return;
    const localDir = new Vector3(...localDirection);
    for (const hit of hits) {
        localPoint.set(localOrigin[0], localOrigin[1], localOrigin[2]).addScaledVector(localDir, hit.t);
        worldPoint.copy(localPoint).applyMatrix4(this.matrixWorld);
        const distance = raycaster.ray.origin.distanceTo(worldPoint);
        if (distance < raycaster.near || distance > raycaster.far) continue;
        intersects.push({ distance, point: worldPoint.clone(), object: this, ...intersectionFields(this.geometry, hit.triangle, hit.u, hit.v, localDir, hit.materialIndex) });
    }
}

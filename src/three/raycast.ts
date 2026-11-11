/**
 * CPU raycasting for three meshes over this package's trees (Q1): drop-ins
 * for three-mesh-bvh's `computeBoundsTree`, `disposeBoundsTree` and
 * `acceleratedRaycast`, with the CPU walk (`createCpuTracer`, the step-for-step
 * port of the WGSL walk) over one tree per geometry.
 *
 * ```ts
 * import { acceleratedRaycast, computeBoundsTree, disposeBoundsTree, prepareBoundsTrees } from '@pmndrs/bvh/three';
 * await prepareBoundsTrees();                     // loads tinybvh (WebAssembly) once
 * BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
 * BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
 * Mesh.prototype.raycast = acceleratedRaycast;
 * geometry.computeBoundsTree();
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
 * one-time warning until `computeBoundsTree` runs again.
 */

import {
    BackSide,
    DoubleSide,
    FrontSide,
    Mesh,
    Triangle,
    Vector2,
    Vector3,
    type BufferAttribute,
    type BufferGeometry,
    type Intersection,
    type InterleavedBufferAttribute,
    type Material,
    type Raycaster,
} from 'three';
import { createTinyBvhBuilder, type TinyBvhBuilder } from '../tinybvh.ts';
import { createCpuTracer, type CpuTracer } from '../trace/cpu-traversal.ts';
import { TRIANGLE_FLAG_FILTER } from '../trace/layout.ts';
import { buildMeshBvhWithTinyBvh } from '../trace/mesh-bvh-topology.ts';
import type { MeshBvh } from '../trace/mesh-bvh.ts';
import { assembleTraceScene } from '../trace/scene-assembly.ts';
import { IDENTITY_TRANSFORM, invertAffineInto } from '../trace/top-level.ts';
import type { TriangleCandidate } from '../trace/triangle-filter.ts';
import { positionsOf, threeTriangleRanges } from './scene.ts';

let sharedBuilder: TinyBvhBuilder | null = null;
let sharedBuilderLoading: Promise<TinyBvhBuilder> | null = null;

/** Loads tinybvh (WebAssembly) for {@link computeBoundsTree}; call once before the first one. */
export async function prepareBoundsTrees(): Promise<TinyBvhBuilder> {
    sharedBuilderLoading ??= createTinyBvhBuilder().then((builder) => (sharedBuilder = builder));
    return sharedBuilderLoading;
}

export interface BvhBoundsTreeOptions {
    /** The tinybvh builder; default the one {@link prepareBoundsTrees} loaded. */
    readonly builder?: TinyBvhBuilder;
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
    collect: TriangleCandidate[] | null;
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
 * A geometry's tree for CPU raycasts: one mesh of all its triangles (in index
 * order, so a hit's triangle is three's `faceIndex`) under an identity
 * instance, every triangle flagged so the walk asks which ones count.
 */
export class BvhBoundsTree {
    readonly geometry: BufferGeometry;
    readonly triangleCount: number;
    readonly bvh: MeshBvh;
    /** The position attribute and index the tree was built from, and their versions then. */
    readonly position: BufferAttribute | InterleavedBufferAttribute;
    readonly positionVersion: number;
    readonly index: BufferAttribute | null;
    readonly indexVersion: number;
    /** Warned once that the geometry changed since the build. */
    staleWarned = false;
    private readonly tracer: CpuTracer;
    private state: RaycastState | null = null;
    private readonly rangeCache = new WeakMap<object, RaycastState[]>();

    constructor(geometry: BufferGeometry, options: BvhBoundsTreeOptions = {}) {
        const builder = options.builder ?? sharedBuilder;
        if (!builder) throw new Error('@pmndrs/bvh/three: call `await prepareBoundsTrees()` once before computeBoundsTree (it loads tinybvh)');
        const position = geometry.getAttribute('position') as BufferAttribute | InterleavedBufferAttribute | undefined;
        if (!position) throw new Error('@pmndrs/bvh/three: computeBoundsTree needs a position attribute');
        const index = geometry.index;
        const vertexCount = index ? index.count : position.count;
        this.triangleCount = Math.floor(vertexCount / 3);
        let indices: Uint16Array | Uint32Array | null = null;
        if (index) {
            const array = index.array;
            indices =
                array instanceof Uint16Array || array instanceof Uint32Array
                    ? array.subarray(0, this.triangleCount * 3)
                    : Uint32Array.from({ length: this.triangleCount * 3 }, (_, i) => index.getX(i));
        }
        const positions = index ? positionsOf(position) : positionsOf(position, 0, this.triangleCount * 3);
        this.geometry = geometry;
        this.position = position;
        this.positionVersion = positionVersionOf(position);
        this.index = index;
        this.indexVersion = index ? index.version : 0;
        this.bvh = buildMeshBvhWithTinyBvh({ positions, indices, triangleFlags: new Uint32Array(this.triangleCount).fill(TRIANGLE_FLAG_FILTER) }, builder);
        const scene = assembleTraceScene([this.bvh], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY_TRANSFORM }]);
        this.tracer = createCpuTracer(scene, { triangleFilter: (candidate) => this.passes(candidate) });
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
        state.collect = firstHitOnly ? null : [];
        this.state = state;
        try {
            const hit = this.tracer.closest({ origin, direction, tMin: near, tMax: far });
            const found = firstHitOnly ? (hit ? [{ t: hit.t, triangle: hit.triangleIndex, barycentrics: [hit.u, hit.v] }] : []) : state.collect!;
            return found.map((candidate) => ({
                t: candidate.t,
                triangle: candidate.triangle,
                u: candidate.barycentrics[0],
                v: candidate.barycentrics[1],
                materialIndex: state.groupIndices[state.ranges ? state.ranges[candidate.triangle] - 1 : 0],
            }));
        } finally {
            state.collect = null;
            this.state = null;
        }
    }

    /** The walk's triangle filter: true lets the ray through (outside the ranges, the culled side, or collecting every hit). */
    private passes(candidate: TriangleCandidate): boolean {
        const state = this.state!;
        const triangle = candidate.triangle;
        const range = state.ranges ? state.ranges[triangle] : triangle >= state.first && triangle < state.end ? 1 : 0;
        if (range === 0) return true;
        const side = state.materials[range - 1]?.side ?? FrontSide;
        if (side === FrontSide && !candidate.frontFace) return true;
        if (side === BackSide && candidate.frontFace) return true;
        if (side !== DoubleSide && side !== FrontSide && side !== BackSide) return true;
        if (state.collect) {
            state.collect.push({ ...candidate, barycentrics: [candidate.barycentrics[0], candidate.barycentrics[1]] });
            return true;
        }
        return false;
    }

    /** True when the geometry's positions and index are the ones the tree was built from. */
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
            collect: null,
            drawStart: drawRange.start,
            drawCount: drawRange.count,
            groups: isArray ? this.geometry.groups.flatMap((group) => [group.start, group.count, group.materialIndex ?? 0]) : [],
            arrayMaterials: isArray ? [...material] : null,
        };
    }
}

/** `geometry.boundsTree`, whoever made it (three-mesh-bvh types it as its own). */
type WithBoundsTree = { boundsTree?: unknown };

/**
 * Trees by position attribute: a BatchedMesh raycasts each instance through
 * a stand-in mesh whose geometry shares the batched geometry's attributes.
 */
const treesByPosition = new WeakMap<object, BvhBoundsTree>();

/** `BufferGeometry.prototype.computeBoundsTree`: builds `this.boundsTree` (a {@link BvhBoundsTree}). */
export function computeBoundsTree(this: BufferGeometry, options: BvhBoundsTreeOptions = {}): BvhBoundsTree {
    const tree = new BvhBoundsTree(this, options);
    (this as unknown as WithBoundsTree).boundsTree = tree;
    treesByPosition.set(tree.position, tree);
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

/** A tree whose geometry changed since its build: three's raycast is used until `computeBoundsTree` runs again. */
function stale(tree: BvhBoundsTree): boolean {
    if (tree.current()) return false;
    if (!tree.staleWarned) {
        tree.staleWarned = true;
        console.warn(
            "@pmndrs/bvh/three: a geometry's positions or index changed since computeBoundsTree; raycasts against it use three's own raycast until computeBoundsTree runs again",
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
const vA = new Vector3();
const vB = new Vector3();
const vC = new Vector3();

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
    const position = this.geometry.attributes.position;
    const index = this.geometry.index;
    const uv = this.geometry.attributes.uv;
    const uv1 = this.geometry.attributes.uv1;
    const normal = this.geometry.attributes.normal;
    const localDir = new Vector3(...localDirection);
    for (const hit of hits) {
        const a = index ? index.getX(hit.triangle * 3) : hit.triangle * 3;
        const b = index ? index.getX(hit.triangle * 3 + 1) : hit.triangle * 3 + 1;
        const c = index ? index.getX(hit.triangle * 3 + 2) : hit.triangle * 3 + 2;
        localPoint.set(localOrigin[0], localOrigin[1], localOrigin[2]).addScaledVector(localDir, hit.t);
        worldPoint.copy(localPoint).applyMatrix4(this.matrixWorld);
        const distance = raycaster.ray.origin.distanceTo(worldPoint);
        if (distance < raycaster.near || distance > raycaster.far) continue;
        const barycoord = new Vector3(1 - hit.u - hit.v, hit.u, hit.v);
        vA.fromBufferAttribute(position, a);
        vB.fromBufferAttribute(position, b);
        vC.fromBufferAttribute(position, c);
        const face = { a, b, c, normal: Triangle.getNormal(vA, vB, vC, new Vector3()), materialIndex: hit.materialIndex };
        const intersection: Intersection = { distance, point: worldPoint.clone(), object: this, face, faceIndex: hit.triangle, barycoord };
        if (uv) intersection.uv = Triangle.getInterpolatedAttribute(uv as BufferAttribute, a, b, c, barycoord, new Vector2()) as Vector2;
        if (uv1) intersection.uv1 = Triangle.getInterpolatedAttribute(uv1 as BufferAttribute, a, b, c, barycoord, new Vector2()) as Vector2;
        if (normal) {
            const n = Triangle.getInterpolatedAttribute(normal as BufferAttribute, a, b, c, barycoord, new Vector3()) as Vector3;
            if (n.dot(localDir) > 0) n.multiplyScalar(-1);
            intersection.normal = n;
        }
        intersects.push(intersection);
    }
}

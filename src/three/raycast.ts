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
 * f32 corners in doubles; three's Möller-Trumbore runs in doubles too.
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
 * time: after editing positions, compute it again.
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
    /** Per triangle: its group's position in `materials` + 1, or 0 when outside every range. */
    ranges: Uint16Array;
    materials: (Material | null)[];
    groupIndices: number[];
    collect: TriangleCandidate[] | null;
    /** The draw range and groups this was made for. */
    readonly key: string;
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
    private readonly tracer: CpuTracer;
    private state: RaycastState | null = null;
    private readonly rangeCache = new WeakMap<object, RaycastState>();

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
                materialIndex: state.groupIndices[state.ranges[candidate.triangle] - 1],
            }));
        } finally {
            state.collect = null;
            this.state = null;
        }
    }

    /** The walk's triangle filter: true lets the ray through (outside the ranges, the culled side, or collecting every hit). */
    private passes(candidate: TriangleCandidate): boolean {
        const state = this.state!;
        const range = state.ranges[candidate.triangle];
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

    /** Which triangles count for this material and draw range (cached per material, refreshed when the draw range or groups change). */
    private stateFor(material: Material | Material[], drawRange: { readonly start: number; readonly count: number }): RaycastState {
        const key = `${drawRange.start}:${drawRange.count}:${this.geometry.groups.map((g) => `${g.start},${g.count},${g.materialIndex}`).join(';')}`;
        const cached = this.rangeCache.get(material);
        if (cached && cached.key === key) return cached;
        const ranges = new Uint16Array(this.triangleCount);
        const materials: (Material | null)[] = [];
        const groupIndices: number[] = [];
        for (const range of threeTriangleRanges(this.geometry, material, drawRange)) {
            if (range.start % 3 !== 0) continue; // not authored triangles (see the file comment)
            materials.push(range.threeMaterial);
            groupIndices.push(range.group >= 0 ? range.materialIndex : 0);
            ranges.fill(materials.length, range.start / 3, Math.min(this.triangleCount, (range.start + range.count) / 3));
        }
        const state: RaycastState = { ranges, materials, groupIndices, collect: null, key };
        this.rangeCache.set(material, state);
        return state;
    }
}

/** `geometry.boundsTree`, whoever made it (three-mesh-bvh types it as its own). */
type WithBoundsTree = { boundsTree?: unknown };

/** `BufferGeometry.prototype.computeBoundsTree`: builds `this.boundsTree` (a {@link BvhBoundsTree}). */
export function computeBoundsTree(this: BufferGeometry, options: BvhBoundsTreeOptions = {}): BvhBoundsTree {
    const tree = new BvhBoundsTree(this, options);
    (this as unknown as WithBoundsTree).boundsTree = tree;
    return tree;
}

/** `BufferGeometry.prototype.disposeBoundsTree`: drops `this.boundsTree`. */
export function disposeBoundsTree(this: BufferGeometry): void {
    (this as unknown as WithBoundsTree).boundsTree = undefined;
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
    const geometry = this.geometry as unknown as WithBoundsTree & { morphAttributes: BufferGeometry["morphAttributes"] };
    const tree = geometry.boundsTree;
    const morphs = this.morphTargetInfluences !== undefined && Object.keys(geometry.morphAttributes).length > 0;
    if (!(tree instanceof BvhBoundsTree) || tree.geometry !== this.geometry || (this as unknown as { isSkinnedMesh?: boolean }).isSkinnedMesh || morphs) {
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

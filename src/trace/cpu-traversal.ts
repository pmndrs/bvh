/**
 * A CPU port of the WGSL walk in `wgsl.ts`, step for step.
 *
 * It exists so the builder and the traversal logic (child order,
 * two-level descent, object-space rays, stack use) can be tested in plain Node
 * against the brute-force reference, with no GPU. It also reports the deepest
 * stack the walk used, which the GPU cannot tell us without a debug counter.
 * It runs in doubles, so it matches the GPU's decisions but not its rounding.
 *
 * The triangle filter, as in the WGSL walk: a triangle whose flag bit 0 is set
 * asks `options.triangleFilter`, and a ray it lets through goes on without
 * narrowing. Without a filter, flagged triangles block (triangle-filter.ts).
 *
 * Ray masks, as in the WGSL walk: each trace takes a `mask` (default all
 * ones), and an instance whose mask shares no bit with it is skipped at the
 * top level. `rayMasks: false` mirrors a trace module built without masks:
 * instance masks are ignored and the filter sees all ones.
 */

import {
    INSTANCE_RECORD_WORDS,
    INSTANCE_WORD,
    BVH_NODE_WORDS,
    EMPTY_CHILD_INDEX,
    NODE_WORD,
    TOP_HEADER_WORD,
    TOP_LEVEL_HEADER_WORDS,
    TRACE_STACK_SIZE,
    TRIANGLE_FLAG_FILTER,
    TRIANGLE_FLAG_FLIP_FRONT_FACE,
    TRIANGLE_WORD,
    TRIANGLE_WORD_SOURCE_INSTANCE,
    INSTANCE_FLAG,
    BVH_QUERY_ANY,
    BVH_QUERY_CLOSEST,
    RAY_MASK_ALL,
    instanceMaskFromFlags,
    materialFromFlags,
    TRIANGLE_WORDS,
    REF_INSTANCE_MARKER,
    makeReference,
    referenceCount,
    referenceIndex,
} from './layout.ts';
import type { TraceRay } from './rays.ts';
import type { TraceSceneData } from './scene-assembly.ts';
import type { TrianglePasses } from './triangle-filter.ts';

export interface CpuHit {
    readonly t: number;
    readonly instanceIndex: number;
    readonly triangleIndex: number;
    readonly u: number;
    readonly v: number;
    readonly frontFace: boolean;
    /** The instance's material index (a merged hit: its source's), or `NO_MATERIAL`. As `BvhHit.material`. */
    readonly material: number;
    /** World-space geometric normal of the authored front side, unit length. As `BvhHit.normal`. */
    readonly normal: readonly [number, number, number];
}

/** Counters the walk fills in when asked. */
export interface CpuTraceStats {
    maxStackSize: number;
    nodesVisited: number;
    trianglesTested: number;
}

export function newTraceStats(): CpuTraceStats {
    return { maxStackSize: 0, nodesVisited: 0, trianglesTested: 0 };
}

/** Per-trace options. */
export interface CpuTraceOptions {
    /** The ray mask (the WGSL `*Masked` functions' `mask`). Default all ones. Ignored with `rayMasks: false`. */
    readonly mask?: number;
    /** Counters to fill in. */
    readonly stats?: CpuTraceStats;
}

export interface CpuTracer {
    closest(ray: TraceRay, options?: CpuTraceOptions): CpuHit | null;
    any(ray: TraceRay, options?: CpuTraceOptions): boolean;
}

const BOX_MISS = Infinity;
const MIN_DIRECTION = 1e-30;
const BOX_EXIT_PADDING = 1.0000004;

function safeInverse(component: number): number {
    const magnitude = Math.max(Math.abs(component), MIN_DIRECTION);
    return 1 / (component < 0 ? -magnitude : magnitude);
}

/** A ray in the space of the level being walked, with what the triangle tests need precomputed. */
interface RaySetup {
    readonly origin: readonly number[];
    readonly direction: readonly number[];
    readonly inverseDirection: readonly number[];
    /** Watertight test: the axis the ray runs along most (z'), and the two others (x', y'), winding-preserving. */
    readonly axes: readonly [number, number, number];
    /** Watertight test: shear that turns the ray into +z'. */
    readonly shear: readonly [number, number, number];
}

function makeRaySetup(origin: readonly number[], direction: readonly number[]): RaySetup {
    const magnitude = direction.map(Math.abs);
    let kz = 2;
    if (magnitude[0] > magnitude[1] && magnitude[0] > magnitude[2]) kz = 0;
    else if (magnitude[1] > magnitude[2]) kz = 1;
    let kx = (kz + 1) % 3;
    let ky = (kx + 1) % 3;
    // Swapping x' and y' for a ray pointing along -z' keeps triangle winding.
    if (direction[kz] < 0) [kx, ky] = [ky, kx];
    return {
        origin,
        direction,
        inverseDirection: direction.map(safeInverse),
        axes: [kx, ky, kz],
        shear: [direction[kx] / direction[kz], direction[ky] / direction[kz], 1 / direction[kz]],
    };
}

interface TriangleHit {
    t: number;
    u: number;
    v: number;
    frontFace: boolean;
}

export interface CpuTracerOptions {
    /** The CPU twin of the kernel's `bvhTrianglePasses`, for flagged triangles; without it they block. */
    readonly triangleFilter?: TrianglePasses;
    /** As the trace module's `rayMasks` option. Default true. */
    readonly rayMasks?: boolean;
    /**
     * Ask `triangleFilter` about every triangle, flagged or not, as if each had
     * `TRIANGLE_FLAG_FILTER` set. The CPU mesh tree (`CpuMeshBvh`) uses it for
     * side culling and all-hit queries without writing the flag into a layout
     * it may share. Default false. Not in the WGSL walk.
     */
    readonly filterEveryTriangle?: boolean;
}

/** The parts of a {@link TraceSceneData} the walk reads. */
export type CpuTraceScene = Pick<TraceSceneData, 'nodes' | 'triangles' | 'topLevel'>;

export function createCpuTracer(scene: CpuTraceScene, options: CpuTracerOptions = {}): CpuTracer {
    const triangleFilter = options.triangleFilter;
    const filterEveryTriangle = options.filterEveryTriangle ?? false;
    const rayMasks = options.rayMasks ?? true;
    const nodeFloats = new Float32Array(scene.nodes);
    const nodeWords = new Uint32Array(scene.nodes);
    const triangleFloats = new Float32Array(scene.triangles);
    const triangleWords = new Uint32Array(scene.triangles);
    const topFloats = new Float32Array(scene.topLevel.buffer);
    const topWords = new Uint32Array(scene.topLevel.buffer);

    const instanceCount = topWords[TOP_HEADER_WORD.instanceCount];
    const topNodeCount = topWords[TOP_HEADER_WORD.nodeCount];
    const firstTopLevelNodeWord = TOP_LEVEL_HEADER_WORDS + instanceCount * INSTANCE_RECORD_WORDS;

    /** Entry distance into the box stored at `floats[minWord..]`/`floats[maxWord..]`, or BOX_MISS. */
    function boxEntry(
        floats: Float32Array,
        minWord: number,
        maxWord: number,
        setup: RaySetup,
        tMin: number,
        tMax: number,
    ): number {
        let enter = tMin;
        let exit = Infinity;
        for (let axis = 0; axis < 3; axis++) {
            const toMin = (floats[minWord + axis] - setup.origin[axis]) * setup.inverseDirection[axis];
            const toMax = (floats[maxWord + axis] - setup.origin[axis]) * setup.inverseDirection[axis];
            enter = Math.max(enter, Math.min(toMin, toMax));
            exit = Math.min(exit, Math.max(toMin, toMax));
        }
        exit = Math.min(exit * BOX_EXIT_PADDING, tMax);
        return enter <= exit ? enter : BOX_MISS;
    }

    function traverse(ray: TraceRay, stopAtFirstHit: boolean, traceOptions: CpuTraceOptions): CpuHit | null {
        if (topNodeCount === 0) return null;
        const stats = traceOptions.stats;
        const rayMask = rayMasks ? (traceOptions.mask ?? RAY_MASK_ALL) >>> 0 : RAY_MASK_ALL;

        const worldSetup = makeRaySetup(ray.origin, ray.direction);
        let setup = worldSetup;

        let insideInstance = false;
        let rootNodeOffset = 0;
        let triangleOffset = 0;
        let instanceIndex = 0;
        let instanceMerged = false;
        let instanceRecord = 0;

        let closestT = ray.tMax;
        let result: CpuHit | null = null;
        // As the WGSL: where the closest hit came from, for its normal and material after the walk.
        let hitRecord = 0;
        let hitBase = 0;
        const finish = (): CpuHit | null => (result === null || stopAtFirstHit ? result : finishClosest(result, hitRecord, hitBase));

        const stack: number[] = [];
        let current = makeReference(0, 0);

        for (;;) {
            const count = referenceCount(current);
            const index = referenceIndex(current);
            let descended = false;

            if (count === 0) {
                if (stats) stats.nodesVisited++;
                const floats = insideInstance ? nodeFloats : topFloats;
                const words = insideInstance ? nodeWords : topWords;
                const nodeWord = insideInstance
                    ? (rootNodeOffset + index) * BVH_NODE_WORDS
                    : firstTopLevelNodeWord + index * BVH_NODE_WORDS;

                const leftEntry = boxEntry(floats, nodeWord + NODE_WORD.leftMin, nodeWord + NODE_WORD.leftMax, setup, ray.tMin, closestT);
                const rightIndex = words[nodeWord + NODE_WORD.rightIndex];
                const rightEntry =
                    rightIndex === EMPTY_CHILD_INDEX
                        ? BOX_MISS
                        : boxEntry(floats, nodeWord + NODE_WORD.rightMin, nodeWord + NODE_WORD.rightMax, setup, ray.tMin, closestT);
                const leftReference = makeReference(words[nodeWord + NODE_WORD.leftIndex], words[nodeWord + NODE_WORD.leftCount]);
                const rightReference = makeReference(rightIndex, words[nodeWord + NODE_WORD.rightCount]);
                const hitLeft = leftEntry < BOX_MISS;
                const hitRight = rightEntry < BOX_MISS;

                if (hitLeft && hitRight) {
                    // As in the WGSL: closest-hit takes the nearer child first, any-hit the farther one.
                    const leftFirst = stopAtFirstHit ? leftEntry > rightEntry : leftEntry <= rightEntry;
                    if (leftFirst) {
                        stack.push(rightReference);
                        current = leftReference;
                    } else {
                        stack.push(leftReference);
                        current = rightReference;
                    }
                    descended = true;
                } else if (hitLeft) {
                    current = leftReference;
                    descended = true;
                } else if (hitRight) {
                    current = rightReference;
                    descended = true;
                }
            } else if (!insideInstance && rayMasks && (instanceMaskFromFlags(topWords[TOP_LEVEL_HEADER_WORDS + index * INSTANCE_RECORD_WORDS + INSTANCE_WORD.flags]) & rayMask) === 0) {
                // As the WGSL: a masked-out instance is skipped before entering its tree (fall through to pop).
            } else if (!insideInstance) {
                const record = TOP_LEVEL_HEADER_WORDS + index * INSTANCE_RECORD_WORDS;
                instanceRecord = record;
                rootNodeOffset = topWords[record + INSTANCE_WORD.rootNodeOffset];
                triangleOffset = topWords[record + INSTANCE_WORD.triangleOffset];
                instanceIndex = topWords[record + INSTANCE_WORD.instanceIndex];
                instanceMerged = (topWords[record + INSTANCE_WORD.flags] & INSTANCE_FLAG.merged) !== 0;
                const row = (rowStart: number): number[] => [
                    topFloats[record + rowStart],
                    topFloats[record + rowStart + 1],
                    topFloats[record + rowStart + 2],
                    topFloats[record + rowStart + 3],
                ];
                const rows = [row(INSTANCE_WORD.worldToObjectRow0), row(INSTANCE_WORD.worldToObjectRow1), row(INSTANCE_WORD.worldToObjectRow2)];
                const origin = rows.map((r) => r[0] * ray.origin[0] + r[1] * ray.origin[1] + r[2] * ray.origin[2] + r[3]);
                // Not renormalized: t stays in world units.
                const direction = rows.map((r) => r[0] * ray.direction[0] + r[1] * ray.direction[1] + r[2] * ray.direction[2]);
                setup = makeRaySetup(origin, direction);
                stack.push(REF_INSTANCE_MARKER);
                insideInstance = true;
                current = makeReference(0, 0);
                descended = true;
            } else {
                for (let i = 0; i < count; i++) {
                    if (stats) stats.trianglesTested++;
                    const base = (triangleOffset + index + i) * TRIANGLE_WORDS;
                    const hit = intersectWatertight(triangleFloats, base, setup, ray.tMin, closestT);
                    if (hit === null) continue;
                    const triangleIndex = triangleWords[base + TRIANGLE_WORD.originalIndex];
                    // As the WGSL: a merged mesh's triangle names its source instance, and the
                    // flip bit marks one from a mirrored source (world-space winding reversed).
                    const hitInstance = instanceMerged ? triangleWords[base + TRIANGLE_WORD_SOURCE_INSTANCE] : instanceIndex;
                    const frontFace = hit.frontFace !== ((triangleWords[base + TRIANGLE_WORD.flags] & TRIANGLE_FLAG_FLIP_FRONT_FACE) !== 0);
                    // As the WGSL leaf test: a flagged triangle's filter may let the ray through,
                    // and then nothing is recorded and closestT stays.
                    const consult = filterEveryTriangle || (triangleWords[base + TRIANGLE_WORD.flags] & TRIANGLE_FLAG_FILTER) !== 0;
                    if (
                        consult &&
                        triangleFilter &&
                        triangleFilter({
                            instance: hitInstance,
                            triangle: triangleIndex,
                            t: hit.t,
                            barycentrics: [hit.u, hit.v],
                            frontFace,
                            query: stopAtFirstHit ? BVH_QUERY_ANY : BVH_QUERY_CLOSEST,
                            rayMask,
                        })
                    ) {
                        continue;
                    }
                    result = {
                        t: hit.t,
                        instanceIndex: hitInstance,
                        triangleIndex,
                        u: hit.u,
                        v: hit.v,
                        // Object-space winding is the authored one, so this is the authored side even when
                        // mirrored; a merged mesh's mirrored-source triangle carries the flip bit instead.
                        frontFace,
                        material: 0,
                        normal: [0, 0, 0],
                    };
                    hitRecord = instanceRecord;
                    hitBase = base;
                    closestT = hit.t;
                    if (stopAtFirstHit) return result;
                }
            }

            if (descended) {
                if (stats) stats.maxStackSize = Math.max(stats.maxStackSize, stack.length);
                if (stack.length > TRACE_STACK_SIZE) {
                    throw new RangeError(`the trace stack overflowed: ${stack.length} entries, room for ${TRACE_STACK_SIZE}`);
                }
                continue;
            }

            // Pop.
            for (;;) {
                if (stack.length === 0) return finish();
                const entry = stack.pop() as number;
                if (entry === REF_INSTANCE_MARKER) {
                    insideInstance = false;
                    setup = worldSetup;
                    continue;
                }
                current = entry;
                break;
            }
        }
    }

    /** As the WGSL `bvhFinishClosest`: the normal and material, once per closest hit. */
    function finishClosest(hit: CpuHit, record: number, base: number): CpuHit {
        const flags = topWords[record + INSTANCE_WORD.flags];
        const merged = (flags & INSTANCE_FLAG.merged) !== 0;
        const triangleFlags = triangleWords[base + TRIANGLE_WORD.flags];
        const corner = (word: number, axis: number): number => triangleFloats[base + word + axis];
        const e1 = [0, 1, 2].map((axis) => corner(TRIANGLE_WORD.vertex1, axis) - corner(TRIANGLE_WORD.vertex0, axis));
        const e2 = [0, 1, 2].map((axis) => corner(TRIANGLE_WORD.vertex2, axis) - corner(TRIANGLE_WORD.vertex0, axis));
        const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        let world: number[];
        if (merged) {
            // World-space corners already; the flip bit turns a mirrored source's normal back to its authored side.
            const sign = (triangleFlags & TRIANGLE_FLAG_FLIP_FRONT_FACE) !== 0 ? -1 : 1;
            world = n.map((value) => value * sign);
        } else {
            // The transpose of the world-to-object rows: the inverse transpose of object-to-world.
            const row = (start: number, axis: number): number => topFloats[record + start + axis];
            world = [0, 1, 2].map(
                (axis) => row(INSTANCE_WORD.worldToObjectRow0, axis) * n[0] + row(INSTANCE_WORD.worldToObjectRow1, axis) * n[1] + row(INSTANCE_WORD.worldToObjectRow2, axis) * n[2],
            );
        }
        const length = Math.hypot(world[0], world[1], world[2]);
        const normal: [number, number, number] = length > 0 ? [world[0] / length, world[1] / length, world[2] / length] : [0, 0, 0];
        return { ...hit, material: materialFromFlags(merged ? triangleFlags : flags), normal };
    }

    return {
        closest: (ray, traceOptions = {}) => traverse(ray, false, traceOptions),
        any: (ray, traceOptions = {}) => traverse(ray, true, traceOptions) !== null,
    };
}

/** Woop, Benthin and Wald 2013. Triangles store `v0`, `v1`, `v2`. Edge-function zeros count for both sides. */
function intersectWatertight(
    floats: Float32Array,
    base: number,
    setup: RaySetup,
    tMin: number,
    tMax: number,
): TriangleHit | null {
    const [kx, ky, kz] = setup.axes;
    const [shearX, shearY, shearZ] = setup.shear;
    const relative = (word: number): number[] => [
        floats[word] - setup.origin[0],
        floats[word + 1] - setup.origin[1],
        floats[word + 2] - setup.origin[2],
    ];
    const a = relative(base + TRIANGLE_WORD.vertex0);
    const b = relative(base + TRIANGLE_WORD.vertex1);
    const c = relative(base + TRIANGLE_WORD.vertex2);

    const ax = a[kx] - shearX * a[kz];
    const ay = a[ky] - shearY * a[kz];
    const bx = b[kx] - shearX * b[kz];
    const by = b[ky] - shearY * b[kz];
    const cx = c[kx] - shearX * c[kz];
    const cy = c[ky] - shearY * c[kz];

    const u = cx * by - cy * bx;
    const v = ax * cy - ay * cx;
    const w = bx * ay - by * ax;
    if ((u < 0 || v < 0 || w < 0) && (u > 0 || v > 0 || w > 0)) return null;

    const determinant = u + v + w;
    if (determinant === 0) return null;
    const scaledT = u * (shearZ * a[kz]) + v * (shearZ * b[kz]) + w * (shearZ * c[kz]);
    const inverseDeterminant = 1 / determinant;
    const t = scaledT * inverseDeterminant;
    if (t < tMin || t > tMax) return null;
    return { t, u: v * inverseDeterminant, v: w * inverseDeterminant, frontFace: determinant > 0 };
}

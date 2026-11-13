/**
 * A CPU port of the WGSL walk in `wgsl.ts`, step for step.
 *
 * It exists so the builder and the traversal logic (child order,
 * two-level descent, object-space rays, stack use) can be tested in plain Node
 * against the brute-force reference, with no GPU. It also reports the deepest
 * stack the walk used, which the GPU cannot tell us without a debug counter.
 * It runs in doubles, so it matches the GPU's decisions but not its rounding.
 * An f32 copy would not match it either: GPUs may round division and `sqrt`
 * within a few ulp and fuse `dot()`, and the ray setup divides (DIVERGENCES T4,
 * the `wgsl-f32-arithmetic` and `trace-cpu-agreement` GPU pages).
 *
 * The triangle filter, as in the WGSL walk: a triangle whose flag bit 0 is set
 * asks `options.triangleFilter`, and a ray it lets through goes on without
 * narrowing. Without a filter, flagged triangles block (triangle-filter.ts).
 *
 * Ray masks, as in the WGSL walk: each trace takes a `mask` (default all
 * ones), and an instance whose mask shares no bit with it is skipped at the
 * top level. `rayMasks: false` mirrors a trace module built without masks:
 * instance masks are ignored and the filter sees all ones.
 *
 * Quantized nodes (`nodeFormat: 'quantized'`), as in the WGSL walk: the frame
 * is read once per instance entered, and each mesh node is decoded as it is
 * visited (`origin + q * step`, exact, so the boxes are the GPU's bit for bit).
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
    INDEXED_FLAG_SHIFT,
    INDEXED_ID_MASK,
    INDEXED_TRIANGLE_WORD,
    INDEXED_TRIANGLE_WORDS,
    MATERIAL_SHIFT,
    MERGED_SIDE_MATERIAL_SHIFT,
    MERGED_SIDE_SOURCE_BITS,
    MESH_HEADER_WORD,
    positionStrideFloats,
    INSTANCE_FLAG,
    BVH_QUERY_ANY,
    BVH_QUERY_CLOSEST,
    RAY_MASK_ALL,
    instanceMaskFromFlags,
    hitMaterialCode,
    indexedMaterialCode,
    materialFromCode,
    TRIANGLE_WORDS,
    REF_INSTANCE_MARKER,
    makeReference,
    referenceCount,
    referenceIndex,
} from './layout.ts';
import type { TraceRay } from './rays.ts';
import { decodeQuantizedNode, readNodeFrame, type NodeFrame } from '../quantized-nodes.ts';
import { GPU_LAYOUT_QUANTIZED_NODE_WORDS as QUANTIZED_NODE_WORDS } from '../gpu-layout.ts';
import type { TraceSceneData } from './scene-assembly.ts';
import type { TrianglePasses } from './triangle-filter.ts';
import type { AnyHitOrder } from './wgsl.ts';

export interface CpuHit {
    readonly t: number;
    readonly instanceIndex: number;
    readonly triangleIndex: number;
    readonly u: number;
    readonly v: number;
    readonly frontFace: boolean;
    /** The triangle's material (`triangleMaterials`; a merged hit: its source's) when it has one, else the instance's, or `NO_MATERIAL`. As `BvhHit.material`. */
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

/** {@link CpuTraceOptions.side}: which faces count as hits. The values are three's `FrontSide`, `BackSide` and `DoubleSide`. */
export const CPU_TRACE_FRONT_SIDE = 0;
export const CPU_TRACE_BACK_SIDE = 1;
export const CPU_TRACE_DOUBLE_SIDE = 2;

/** Per-trace options. */
export interface CpuTraceOptions {
    /**
     * Side culling, CPU only (the WGSL walk has no side option):
     * {@link CPU_TRACE_FRONT_SIDE} takes front-face hits only,
     * {@link CPU_TRACE_BACK_SIDE} back-face hits only (the authored side, after
     * the flip bit). A culled triangle is passed through like a filtered one:
     * nothing is recorded, the filter isn't asked, and the closest distance
     * stays. Default {@link CPU_TRACE_DOUBLE_SIDE}, the WGSL walk's behaviour.
     */
    readonly side?: number;
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
    /** As the trace module's `anyHitOrder` option: any-hit's child order. Default `'far'`. Answers are the same either way. */
    readonly anyHitOrder?: AnyHitOrder;
    /**
     * Ask `triangleFilter` about every triangle, flagged or not, as if each had
     * `TRIANGLE_FLAG_FILTER` set. The CPU mesh tree (`CpuMeshBvh`) uses it for
     * all-hit queries and per-triangle acceptance (three's groups) without
     * writing the flag into a layout it may share. Default false. Not in the WGSL walk.
     */
    readonly filterEveryTriangle?: boolean;
}

/**
 * The parts of a {@link TraceSceneData} the walk reads. `triangleLeaves`
 * (default `'records'`), `positions` and `positionLayout` (default `'vec4'`)
 * only matter for an indexed scene.
 */
export type CpuTraceScene = Pick<TraceSceneData, 'nodes' | 'triangles' | 'topLevel'> & Partial<Pick<TraceSceneData, 'triangleLeaves' | 'positions' | 'positionLayout' | 'nodeFormat'>>;

/** Where a triangle's corners and ids are, for either format: word offsets into `floats`, and its id and flags. */
interface TriangleSource {
    /** Holds the corners: the records themselves, or the positions buffer. */
    floats: Float32Array;
    a: number;
    b: number;
    c: number;
    original: number;
    /** Record flags bits 0-1 (filter, flip), and for records bits 16-31. */
    flags: number;
}

export function createCpuTracer(scene: CpuTraceScene, options: CpuTracerOptions = {}): CpuTracer {
    const triangleFilter = options.triangleFilter;
    const filterEveryTriangle = options.filterEveryTriangle ?? false;
    const rayMasks = options.rayMasks ?? true;
    const anyHitOrder = options.anyHitOrder ?? 'far';
    // Half the surface area of a box, as the WGSL's bvhBoxHalfArea (f32 products, so the order is the GPU's).
    const halfArea = (floats: Float32Array, minWord: number, maxWord: number): number => {
        const x = Math.fround(floats[maxWord] - floats[minWord]);
        const y = Math.fround(floats[maxWord + 1] - floats[minWord + 1]);
        const z = Math.fround(floats[maxWord + 2] - floats[minWord + 2]);
        return Math.fround(Math.fround(Math.fround(x * y) + Math.fround(y * z)) + Math.fround(z * x));
    };
    const nodeFloats = new Float32Array(scene.nodes);
    const nodeWords = new Uint32Array(scene.nodes);
    const quantized = scene.nodeFormat === 'quantized';
    // Quantized: the node being visited, decoded into a 64-byte node, and the instance's frame.
    const decodedNode = new ArrayBuffer(BVH_NODE_WORDS * 4);
    const decodedFloats = new Float32Array(decodedNode);
    const decodedWords = new Uint32Array(decodedNode);
    let frame: NodeFrame | null = null;
    const triangleFloats = new Float32Array(scene.triangles);
    const triangleWords = new Uint32Array(scene.triangles);
    const indexed = scene.triangleLeaves === 'indexed';
    const positionFloats = indexed ? new Float32Array(scene.positions ?? new ArrayBuffer(0)) : triangleFloats;
    const stride = positionStrideFloats(scene.positionLayout);
    /** Indexed: the vertex base in the header before a mesh's triangles. */
    const vertexBaseOf = (triangleOffset: number): number => (indexed ? triangleWords[(triangleOffset - 1) * INDEXED_TRIANGLE_WORDS + MESH_HEADER_WORD.vertexBase] : 0);
    /** The triangle at `slot` (in the scene's triangle buffer), read as the WGSL's bvhLoadTriangle reads it. */
    const loadTriangle = (slot: number, vertexBase: number, out: TriangleSource): TriangleSource => {
        if (indexed) {
            const base = slot * INDEXED_TRIANGLE_WORDS;
            out.a = (vertexBase + triangleWords[base]) * stride;
            out.b = (vertexBase + triangleWords[base + 1]) * stride;
            out.c = (vertexBase + triangleWords[base + 2]) * stride;
            const idFlags = triangleWords[base + INDEXED_TRIANGLE_WORD.idFlags];
            out.original = idFlags & INDEXED_ID_MASK;
            out.flags = idFlags >>> INDEXED_FLAG_SHIFT;
        } else {
            const base = slot * TRIANGLE_WORDS;
            out.a = base + TRIANGLE_WORD.vertex0;
            out.b = base + TRIANGLE_WORD.vertex1;
            out.c = base + TRIANGLE_WORD.vertex2;
            out.original = triangleWords[base + TRIANGLE_WORD.originalIndex];
            out.flags = triangleWords[base + TRIANGLE_WORD.flags];
        }
        return out;
    };
    /**
     * A merged mesh's triangle: its source instance and material code. Records
     * hold both in the triangle (word 11, flags bits 16-31); an indexed scene
     * reads the group's side table and source table (`writeMergedIndexedIdentity`).
     */
    const mergedSource = (triangleOffset: number, slot: number, flags: number): { instance: number; materialCode: number } => {
        if (!indexed) return { instance: triangleWords[slot * TRIANGLE_WORDS + TRIANGLE_WORD_SOURCE_INSTANCE], materialCode: flags >>> MATERIAL_SHIFT };
        const header = (triangleOffset - 1) * INDEXED_TRIANGLE_WORDS;
        const local = slot - triangleOffset;
        const side = triangleWords[(triangleOffset - triangleWords[header + MESH_HEADER_WORD.sideTableDistance]) * INDEXED_TRIANGLE_WORDS + local];
        const sourceSlot = side & MERGED_SIDE_SOURCE_BITS;
        const instance = triangleWords[(triangleOffset - triangleWords[header + MESH_HEADER_WORD.sourceTableDistance]) * INDEXED_TRIANGLE_WORDS + sourceSlot];
        return { instance, materialCode: side >>> MERGED_SIDE_MATERIAL_SHIFT };
    };
    const scratch: TriangleSource = { floats: positionFloats, a: 0, b: 0, c: 0, original: 0, flags: 0 };
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
        const side = traceOptions.side ?? CPU_TRACE_DOUBLE_SIDE;
        const cull = side !== CPU_TRACE_DOUBLE_SIDE;
        const keepFront = side === CPU_TRACE_FRONT_SIDE;

        const worldSetup = makeRaySetup(ray.origin, ray.direction);
        let setup = worldSetup;

        let insideInstance = false;
        let rootNodeOffset = 0;
        let triangleOffset = 0;
        let vertexBase = 0;
        let instanceIndex = 0;
        let instanceMerged = false;
        let instanceRecord = 0;

        let closestT = ray.tMax;
        let result: CpuHit | null = null;
        // As the WGSL: where the closest hit came from, for its normal and material after the walk.
        let hitRecord = 0;
        let hitSlot = 0;
        const finish = (): CpuHit | null => (result === null || stopAtFirstHit ? result : finishClosest(result, hitRecord, hitSlot));

        const stack: number[] = [];
        let current = makeReference(0, 0);

        for (;;) {
            const count = referenceCount(current);
            const index = referenceIndex(current);
            let descended = false;

            if (count === 0) {
                if (stats) stats.nodesVisited++;
                const decode = insideInstance && quantized;
                if (decode) decodeQuantizedNode(nodeWords, (rootNodeOffset + index) * QUANTIZED_NODE_WORDS, frame!, decodedFloats, decodedWords, 0);
                const floats = decode ? decodedFloats : insideInstance ? nodeFloats : topFloats;
                const words = decode ? decodedWords : insideInstance ? nodeWords : topWords;
                const nodeWord = decode ? 0 : insideInstance
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
                    // As in the WGSL: closest-hit takes the nearer child first; any-hit the farther one
                    // (anyHitOrder 'far'), the nearer ('near') or the larger ('larger').
                    const leftFirst = !stopAtFirstHit || anyHitOrder === 'near'
                        ? leftEntry <= rightEntry
                        : anyHitOrder === 'larger'
                          ? halfArea(floats, nodeWord + NODE_WORD.leftMin, nodeWord + NODE_WORD.leftMax) >= halfArea(floats, nodeWord + NODE_WORD.rightMin, nodeWord + NODE_WORD.rightMax)
                          : leftEntry > rightEntry;
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
                if (quantized) frame = readNodeFrame(nodeWords, (rootNodeOffset - 1) * QUANTIZED_NODE_WORDS);
                triangleOffset = topWords[record + INSTANCE_WORD.triangleOffset];
                vertexBase = vertexBaseOf(triangleOffset);
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
                    const slot = triangleOffset + index + i;
                    const triangle = loadTriangle(slot, vertexBase, scratch);
                    const hit = intersectWatertight(positionFloats, triangle.a, triangle.b, triangle.c, setup, ray.tMin, closestT);
                    if (hit === null) continue;
                    const triangleIndex = triangle.original;
                    const triangleFlags = triangle.flags;
                    // As the WGSL: a merged mesh's triangle names its source instance, and the
                    // flip bit marks one from a mirrored source (world-space winding reversed).
                    const hitInstance = instanceMerged ? mergedSource(triangleOffset, slot, triangleFlags).instance : instanceIndex;
                    const frontFace = hit.frontFace !== ((triangleFlags & TRIANGLE_FLAG_FLIP_FRONT_FACE) !== 0);
                    // CPU only: side culling passes the ray through the other side's faces.
                    if (cull && frontFace !== keepFront) continue;
                    // As the WGSL leaf test: a flagged triangle's filter may let the ray through,
                    // and then nothing is recorded and closestT stays.
                    const consult = filterEveryTriangle || (triangleFlags & TRIANGLE_FLAG_FILTER) !== 0;
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
                    hitSlot = slot;
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
    function finishClosest(hit: CpuHit, record: number, slot: number): CpuHit {
        const flags = topWords[record + INSTANCE_WORD.flags];
        const merged = (flags & INSTANCE_FLAG.merged) !== 0;
        const offset = topWords[record + INSTANCE_WORD.triangleOffset];
        const triangle = loadTriangle(slot, vertexBaseOf(offset), { floats: positionFloats, a: 0, b: 0, c: 0, original: 0, flags: 0 });
        const triangleFlags = triangle.flags;
        const e1 = [0, 1, 2].map((axis) => positionFloats[triangle.b + axis] - positionFloats[triangle.a + axis]);
        const e2 = [0, 1, 2].map((axis) => positionFloats[triangle.c + axis] - positionFloats[triangle.a + axis]);
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
        // As the WGSL: the triangle's material when it has one, else the instance's.
        const triangleCode = merged
            ? mergedSource(offset, slot, triangleFlags).materialCode
            : indexed
              ? indexedMaterialCode(triangleWords, offset, triangle.original)
              : triangleFlags >>> MATERIAL_SHIFT;
        const material = materialFromCode(hitMaterialCode(triangleCode, flags));
        return { ...hit, material, normal };
    }

    return {
        closest: (ray, traceOptions = {}) => traverse(ray, false, traceOptions),
        any: (ray, traceOptions = {}) => traverse(ray, true, traceOptions) !== null,
    };
}

/**
 * Woop, Benthin and Wald 2013. The corners are at words `aWord`, `bWord`,
 * `cWord` of `floats` (a record's, or the positions buffer's). Edge-function
 * zeros count for both sides.
 */
function intersectWatertight(
    floats: Float32Array,
    aWord: number,
    bWord: number,
    cWord: number,
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
    const a = relative(aWord);
    const b = relative(bWord);
    const c = relative(cWord);

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

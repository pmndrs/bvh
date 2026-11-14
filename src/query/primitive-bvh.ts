/*!
 * Points and line segments as BVH primitives: the raycast pruning (each box
 * grown by the threshold, nearer boxes first and boxes past the closest hit
 * skipped for `firstHitOnly`) and the shapecast's `intersectsPoint` /
 * `intersectsLine` are ported from three-mesh-bvh 0.9.18 (src/core/PointsBVH.js,
 * src/core/LineBVH.js, gkjohnson/three-mesh-bvh). See THIRD_PARTY.md.
 *
 * MIT License
 *
 * Copyright (c) 2018 Garrett Johnson
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
/**
 * Points and line segments as primitives (PARITY 3.2 and 3.3,
 * three-mesh-bvh's `PointsBVH`, `LineSegmentsBVH`, `LineLoopBVH` and
 * `LineBVH`), engine-free: {@link CpuPointsBvh} and {@link CpuSegmentsBvh},
 * beside `CpuMeshBvh`.
 *
 * **Primitives are entries.** The input is three's: positions, an optional
 * index and a range of entries (`start`, `count`, as a draw range). A point is
 * one entry; a segment starts at an entry and takes the next one, as three's
 * `Line.raycast` steps: `'segments'` (`LineSegments`) at every second entry,
 * `'strip'` (`Line`) at every entry but the last, `'loop'` (`LineLoop`) at
 * every entry, the last closing back to the first. A primitive's `index` is
 * that entry (the position in the index, or the vertex without one), which is
 * what three's `Line.raycast` reports; three's `Points.raycast` reports the
 * entry's vertex, which `@pmndrs/bvh/three` looks up.
 *
 * **The tree** uses the GPU layout's 64-byte nodes (`gpu-layout.ts`: two child
 * boxes, leaves of 1 to 8 primitives at a run of tree positions) and its own
 * primitive records in tree order: a point is 16 bytes (`x, y, z` as `f32`,
 * then its index as `u32`), a segment 32 (start, index, end, 0). Coordinates
 * are copies of the input's `f32` positions, so a three geometry's points
 * are read exactly as three reads them.
 *
 * **The builder** is a centre split, three-mesh-bvh's default `CENTER`
 * strategy, written here in TypeScript: the longest axis of the primitives'
 * centres is split at its middle, with a median split when that leaves a side
 * empty or the tree passes depth {@link PRIMITIVE_MEDIAN_DEPTH}. tinybvh is not
 * used: it builds over triangles, and points given to it as degenerate
 * triangles come out as one node per point (a box of area 0 always pays to
 * split under SAH) with 48-byte records, about four times the memory of these
 * trees, and no SAH helps there either. Building needs no WebAssembly, so it
 * is synchronous.
 *
 * **Answers are three's.** {@link CpuPointsBvh.raycast} and
 * {@link CpuSegmentsBvh.raycast} take three's thresholds (a point within the
 * threshold of the ray, strictly; a segment within it or on it) and compute
 * every distance with three's `Ray` operations in three's order
 * (`ray-primitives.ts`), so a hit is three's, bit for bit, given the same ray.
 */

import { GPU_LAYOUT_EMPTY_CHILD, GPU_LAYOUT_MAX_LEAF_PRIMITIVES, GPU_LAYOUT_NODE_BYTES, GPU_LAYOUT_NODE_WORD, GPU_LAYOUT_NODE_WORDS } from '../gpu-layout.ts';
import { BVH_CONTAINED, clampLeafPrimitives, readVec3, type BvhBoxTarget, type BvhRayLike, type BvhTraverseCallbacks, type BvhVec3Like } from './cpu-mesh-bvh.ts';
import { rayDistanceSqToPoint, rayDistanceSqToSegment, segmentClosestPointToPoint } from './ray-primitives.ts';
import { BvhBox3, BvhSegment, BvhVector3, type BvhBox3Like, type BvhSegmentLike, type BvhVector3Like } from './shapes.ts';

/** What a primitive tree holds. */
export type BvhPrimitiveKind = 'points' | 'segments';

/**
 * How entries make segments, as three's line objects read them: `'segments'`
 * (`LineSegments`, pairs), `'strip'` (`Line`, each entry to the next) or
 * `'loop'` (`LineLoop`, a strip closed from the last entry back to the first).
 */
export type BvhSegmentMode = 'segments' | 'strip' | 'loop';

/** Points (or the vertices of segments) as three draws them: positions, an optional index and a range of entries. */
export interface BvhPointsInput {
    /** `x, y, z` per vertex. */
    readonly positions: Float32Array;
    /** Vertex per entry; omit and entry `i` is vertex `i`. */
    readonly indices?: Uint16Array | Uint32Array | null;
    /** First entry, as a draw range's `start`. Default 0. */
    readonly start?: number;
    /** Entries from `start`, as a draw range's `count` (clamped to the entries there are). Default all. */
    readonly count?: number;
}

/** Segments: {@link BvhPointsInput} and how its entries pair up. */
export interface BvhSegmentsInput extends BvhPointsInput {
    /** Default `'segments'`. */
    readonly mode?: BvhSegmentMode;
}

export interface BvhPrimitiveBuildOptions {
    /** Most primitives in a leaf: 1 to 8 (the layout's limit), default 8. */
    readonly maxLeafPrimitives?: number;
    /** Called with 0 before the build and 1 after it. */
    readonly onProgress?: (progress: number) => void;
}

export interface BvhPrimitiveTreeOptions {
    /** The input the tree was built from, kept by reference: `refit()` rereads it. */
    readonly source?: BvhPointsInput | BvhSegmentsInput | null;
}

/** A primitive tree's data (the nodes in the GPU layout, the records in tree order). */
export interface BvhPrimitiveLayout {
    readonly kind: BvhPrimitiveKind;
    /** For segments: how the entries pair up (`'segments'` for points, where it means nothing). */
    readonly mode: BvhSegmentMode;
    /** The entries the tree covers, `[start, end)`. */
    readonly start: number;
    readonly end: number;
    /** `nodeCount` 64-byte nodes (`GPU_LAYOUT_NODE_WORD`); node 0 is the root. Empty for a tree of no primitives. */
    readonly nodes: ArrayBuffer;
    readonly nodeCount: number;
    /** `primitiveCount` records in tree order: {@link POINT_RECORD_BYTES} or {@link SEGMENT_RECORD_BYTES} each. */
    readonly primitives: ArrayBuffer;
    readonly primitiveCount: number;
    /** Box around every primitive, `[minX, minY, minZ, maxX, maxY, maxZ]` (inverted when empty). */
    readonly bounds: Float32Array;
}

/** A point record: `x, y, z` (`f32`), then the entry (`u32`). */
export const POINT_RECORD_BYTES = 16;
/** A segment record: start `x, y, z`, the entry, end `x, y, z`, then 0. */
export const SEGMENT_RECORD_BYTES = 32;
/** Word of the entry in either record. */
export const PRIMITIVE_RECORD_INDEX_WORD = 3;
/** Past this depth the builder splits at the median, so a tree stays shallow on any input. */
export const PRIMITIVE_MEDIAN_DEPTH = 40;

/** One hit of a primitive raycast, in the tree's space. */
export interface BvhPrimitiveRaycastHit {
    /** The primitive's entry (see the file comment). */
    readonly index: number;
    /** From the ray's origin to {@link pointOnRay}: three's `distance`. */
    readonly distance: number;
    /** Between the ray and the primitive: three's `distanceToRay` (points). */
    readonly distanceToRay: number;
    /** The ray's point nearest the primitive (three's `point` for `Points`). */
    readonly pointOnRay: [number, number, number];
    /** The primitive's point nearest the ray: the point itself, or the point on the segment (three's `point` for lines). */
    readonly pointOnPrimitive: [number, number, number];
}

/** Options of {@link CpuPointsBvh.raycast} and {@link CpuSegmentsBvh.raycast}. */
export interface BvhPrimitiveRaycastOptions {
    /**
     * How far from the ray a primitive may be, in the tree's units: three's
     * `Raycaster.params.Points.threshold` and `params.Line.threshold` (default
     * 1, as three's). A point counts when strictly closer, a segment when no
     * farther, as in three.
     */
    readonly threshold?: number;
    /** Hits nearer than this (by {@link BvhPrimitiveRaycastHit.distance}) are left out. Default 0. */
    readonly near?: number;
    /** Hits farther than this are left out. Default `Infinity`. */
    readonly far?: number;
    /**
     * Decides whether a hit counts, after the threshold, `near` and `far`
     * (for example a draw range, or a distance in another space). The
     * closest hit of {@link CpuPointsBvh.raycastFirst} is the closest that passes.
     */
    readonly accept?: (hit: BvhPrimitiveRaycastHit) => boolean;
}

/** What {@link CpuPointsBvh.closestPointToPoint} writes. */
export interface BvhPrimitiveClosestPoint<Point extends BvhVector3Like = BvhVector3Like> {
    /** The nearest point of the primitives (the point, or on the segment). */
    point: Point;
    distance: number;
    /** The primitive's entry. */
    index: number;
}

/** Callbacks of a primitive shapecast: three-mesh-bvh's, with the primitive's own callback. */
export interface BvhPrimitiveShapecastCallbacks<Primitive, Box = BvhBox3Like> {
    /** As `CpuMeshBvh.traverse`'s `intersectsBounds`, with the box as `Box`. */
    intersectsBounds(box: Box, isLeaf: boolean, score: number | undefined, depth: number, nodeIndex: number): unknown;
    /** A range of tree positions before its primitives; a truthy return stops the cast. */
    intersectsRange?(offset: number, count: number, contained: boolean, depth: number, nodeIndex: number, box: Box): unknown;
    /** As `CpuMeshBvh.traverse`'s `boundsTraverseOrder`. */
    boundsTraverseOrder?(box: Box): number;
    /** Internal: the per-primitive callback, set by the subclasses. */
    readonly primitive?: (primitive: Primitive, index: number, contained: boolean, depth: number) => unknown;
}

/** {@link CpuPointsBvh.shapecast}'s callbacks: three-mesh-bvh's `PointsBVH.shapecast`. */
export interface BvhPointsShapecastCallbacks<Point = BvhVector3Like, Box = BvhBox3Like> extends Omit<BvhPrimitiveShapecastCallbacks<Point, Box>, 'primitive'> {
    /** Each point of an entered leaf or contained box, in tree order, as a scratch point, with its entry. A truthy return stops the cast. */
    intersectsPoint?(point: Point, index: number, contained: boolean, depth: number): unknown;
}

/** {@link CpuSegmentsBvh.shapecast}'s callbacks: three-mesh-bvh's `LineSegmentsBVH.shapecast`. */
export interface BvhSegmentsShapecastCallbacks<Segment = BvhSegmentLike, Box = BvhBox3Like> extends Omit<BvhPrimitiveShapecastCallbacks<Segment, Box>, 'primitive'> {
    /** Each segment of an entered leaf or contained box, in tree order, as a scratch segment, with its entry. A truthy return stops the cast. */
    intersectsLine?(line: Segment, index: number, contained: boolean, depth: number): unknown;
}

const KIND_CODE: Record<BvhPrimitiveKind, number> = { points: 0, segments: 1 };
const MODE_CODE: Record<BvhSegmentMode, number> = { segments: 0, strip: 1, loop: 2 };
const KINDS: readonly BvhPrimitiveKind[] = ['points', 'segments'];
const MODES: readonly BvhSegmentMode[] = ['segments', 'strip', 'loop'];
/** "BVHP" in a file's first word. */
const FILE_MAGIC = 0x50485642;
const FILE_VERSION = 1;
const FILE_HEADER_BYTES = 64;
const FILE_HEADER_WORD = { magic: 0, version: 1, kind: 2, mode: 3, start: 4, end: 5, nodeCount: 6, primitiveCount: 7, bounds: 8 } as const;

const LEFT_INDEX = GPU_LAYOUT_NODE_WORD.leftIndex;
const RIGHT_INDEX = GPU_LAYOUT_NODE_WORD.rightIndex;
const LEFT_COUNT = GPU_LAYOUT_NODE_WORD.leftCount;
const RIGHT_COUNT = GPU_LAYOUT_NODE_WORD.rightCount;

/** The entries of an input: `[start, end)` clamped as three clamps a draw range, and the vertex of each. */
interface Entries {
    readonly start: number;
    readonly end: number;
    readonly vertexCount: number;
    vertex(entry: number): number;
}

function entriesOf(input: BvhPointsInput, where: string): Entries {
    const { positions } = input;
    const indices = input.indices ?? null;
    const vertexCount = Math.floor(positions.length / 3);
    const length = indices ? indices.length : vertexCount;
    const requestedStart = input.start ?? 0;
    const requestedCount = input.count ?? Infinity;
    if (!(requestedStart >= 0) || !Number.isFinite(requestedStart) || Math.floor(requestedStart) !== requestedStart) throw new RangeError(`${where}: start must be a whole number from 0 (got ${requestedStart})`);
    if (!(requestedCount >= 0)) throw new RangeError(`${where}: count must be 0 or more (got ${requestedCount})`);
    // three: start = max(0, drawRange.start), end = min(length, drawRange.start + drawRange.count).
    const start = requestedStart;
    const end = Math.max(start, Math.min(length, Math.floor(requestedStart + requestedCount)));
    const vertex = indices ? (entry: number) => indices[entry] : (entry: number) => entry;
    for (let entry = start; entry < end; entry++) {
        if (vertex(entry) >= vertexCount) throw new RangeError(`${where}: entry ${entry} names vertex ${vertex(entry)}, past the ${vertexCount} vertices`);
    }
    return { start, end, vertexCount, vertex };
}

/** How many segments `mode` makes of `n` entries. */
function segmentCount(mode: BvhSegmentMode, n: number): number {
    if (n <= 0) return 0;
    if (mode === 'segments') return Math.floor(n / 2);
    if (mode === 'strip') return n - 1;
    return n;
}

/** The entry segment `k` starts at. */
function segmentEntry(mode: BvhSegmentMode, start: number, k: number): number {
    return mode === 'segments' ? start + 2 * k : start + k;
}

/** The entry a segment starting at `entry` ends at: the next, or for a loop's last the first. */
function segmentEndEntry(mode: BvhSegmentMode, start: number, end: number, entry: number): number {
    return mode === 'loop' && entry === end - 1 ? start : entry + 1;
}

/**
 * Writes each primitive's record from the input's positions: in input order,
 * or (`inTreeOrder`, a refit) in place, each record rereading the entry it names.
 */
function writeRecords(kind: BvhPrimitiveKind, mode: BvhSegmentMode, entries: Entries, positions: Float32Array, floats: Float32Array, words: Uint32Array, inTreeOrder: boolean): void {
    const n = kind === 'points' ? entries.end - entries.start : segmentCount(mode, entries.end - entries.start);
    const stride = kind === 'points' ? POINT_RECORD_BYTES / 4 : SEGMENT_RECORD_BYTES / 4;
    for (let k = 0; k < n; k++) {
        const base = k * stride;
        const entry = inTreeOrder ? words[base + PRIMITIVE_RECORD_INDEX_WORD] : kind === 'points' ? entries.start + k : segmentEntry(mode, entries.start, k);
        const a = entries.vertex(entry) * 3;
        floats[base] = positions[a];
        floats[base + 1] = positions[a + 1];
        floats[base + 2] = positions[a + 2];
        words[base + PRIMITIVE_RECORD_INDEX_WORD] = entry;
        if (kind === 'segments') {
            const b = entries.vertex(segmentEndEntry(mode, entries.start, entries.end, entry)) * 3;
            floats[base + 4] = positions[b];
            floats[base + 5] = positions[b + 1];
            floats[base + 6] = positions[b + 2];
            words[base + 7] = 0;
        }
    }
}

/**
 * Builds a primitive tree: the centre split (see the file comment) over the
 * records, which end up in tree order. Synchronous; no WebAssembly.
 */
export function buildPrimitiveLayout(kind: BvhPrimitiveKind, input: BvhPointsInput | BvhSegmentsInput, options: BvhPrimitiveBuildOptions = {}): BvhPrimitiveLayout {
    const where = kind === 'points' ? 'CpuPointsBvh' : 'CpuSegmentsBvh';
    const mode: BvhSegmentMode = kind === 'segments' ? ((input as BvhSegmentsInput).mode ?? 'segments') : 'segments';
    if (!(mode in MODE_CODE)) throw new RangeError(`${where}: unknown mode ${String(mode)}`);
    const maxLeaf = options.maxLeafPrimitives === undefined ? GPU_LAYOUT_MAX_LEAF_PRIMITIVES : clampLeafPrimitives(options.maxLeafPrimitives);
    options.onProgress?.(0);
    const entries = entriesOf(input, where);
    const positions = input.positions;
    for (let i = 0; i < entries.vertexCount * 3; i++) {
        if (!Number.isFinite(positions[i])) throw new RangeError(`${where}: vertex ${Math.floor(i / 3)} has a non-finite position`);
    }
    const n = kind === 'points' ? entries.end - entries.start : segmentCount(mode, entries.end - entries.start);
    const stride = kind === 'points' ? POINT_RECORD_BYTES / 4 : SEGMENT_RECORD_BYTES / 4;
    // Records in input order first, then permuted into tree order once the order is known.
    const inputRecords = new ArrayBuffer(n * stride * 4);
    writeRecords(kind, mode, entries, positions, new Float32Array(inputRecords), new Uint32Array(inputRecords), false);
    const inputFloats = new Float32Array(inputRecords);
    // Per primitive: its box (exact in f32: the corners are f32) and its centre.
    const boxes = new Float32Array(n * 6);
    const centres = new Float64Array(n * 3);
    for (let k = 0; k < n; k++) {
        const base = k * stride;
        for (let axis = 0; axis < 3; axis++) {
            const a = inputFloats[base + axis];
            const b = kind === 'segments' ? inputFloats[base + 4 + axis] : a;
            boxes[k * 6 + axis] = Math.min(a, b);
            boxes[k * 6 + 3 + axis] = Math.max(a, b);
            centres[k * 3 + axis] = (a + b) * 0.5;
        }
    }
    const order = new Uint32Array(n);
    for (let k = 0; k < n; k++) order[k] = k;
    const nodeCapacity = Math.max(1, n - 1);
    const nodes = new ArrayBuffer(n === 0 ? 0 : nodeCapacity * GPU_LAYOUT_NODE_BYTES);
    const nodeFloats = new Float32Array(nodes);
    const nodeWords = new Uint32Array(nodes);
    let nodeCount = 0;
    const bounds = Float32Array.of(Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity);
    const rangeBox = new Float32Array(6);
    const writeChild = (node: number, side: number, first: number, count: number): void => {
        boxOfRange(boxes, order, first, count, rangeBox);
        const base = node * GPU_LAYOUT_NODE_WORDS + side * 8;
        for (let axis = 0; axis < 3; axis++) {
            nodeFloats[base + axis] = rangeBox[axis];
            nodeFloats[base + 4 + axis] = rangeBox[3 + axis];
        }
    };
    if (n > 0) {
        boxOfRange(boxes, order, 0, n, bounds);
        nodeCount = 1;
        if (n === 1) {
            writeChild(0, 0, 0, 1);
            nodeWords[LEFT_INDEX] = 0;
            nodeWords[LEFT_COUNT] = 1;
            nodeWords[RIGHT_INDEX] = GPU_LAYOUT_EMPTY_CHILD;
            nodeWords[RIGHT_COUNT] = 0;
            for (let axis = 0; axis < 3; axis++) {
                nodeFloats[8 + axis] = 0;
                nodeFloats[12 + axis] = 0;
            }
        } else {
            // Work: node, first tree position, count, depth.
            const work: number[] = [0, 0, n, 0];
            while (work.length > 0) {
                const depth = work.pop()!;
                const count = work.pop()!;
                const first = work.pop()!;
                const node = work.pop()!;
                const leftCount = splitRange(centres, order, first, count, depth >= PRIMITIVE_MEDIAN_DEPTH);
                const halves = [
                    [first, leftCount],
                    [first + leftCount, count - leftCount],
                ] as const;
                for (let side = 0; side < 2; side++) {
                    const [childFirst, childCount] = halves[side];
                    writeChild(node, side, childFirst, childCount);
                    const indexWord = node * GPU_LAYOUT_NODE_WORDS + (side === 0 ? LEFT_INDEX : RIGHT_INDEX);
                    const countWord = node * GPU_LAYOUT_NODE_WORDS + (side === 0 ? LEFT_COUNT : RIGHT_COUNT);
                    if (childCount <= maxLeaf) {
                        nodeWords[indexWord] = childFirst;
                        nodeWords[countWord] = childCount;
                    } else {
                        const child = nodeCount++;
                        nodeWords[indexWord] = child;
                        nodeWords[countWord] = 0;
                        work.push(child, childFirst, childCount, depth + 1);
                    }
                }
            }
        }
    }
    // The records in tree order.
    const primitives = new ArrayBuffer(n * stride * 4);
    const outWords = new Uint32Array(primitives);
    const inWords = new Uint32Array(inputRecords);
    for (let position = 0; position < n; position++) outWords.set(inWords.subarray(order[position] * stride, (order[position] + 1) * stride), position * stride);
    options.onProgress?.(1);
    return {
        kind,
        mode,
        start: entries.start,
        end: entries.end,
        nodes: n === 0 ? nodes : nodes.slice(0, nodeCount * GPU_LAYOUT_NODE_BYTES),
        nodeCount,
        primitives,
        primitiveCount: n,
        bounds,
    };
}

/** The box around the primitives at `order[first..first+count)`, into `out`. */
function boxOfRange(boxes: Float32Array, order: Uint32Array, first: number, count: number, out: Float32Array): void {
    let x0 = Infinity;
    let y0 = Infinity;
    let z0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    let z1 = -Infinity;
    for (let i = first, end = first + count; i < end; i++) {
        const base = order[i] * 6;
        if (boxes[base] < x0) x0 = boxes[base];
        if (boxes[base + 1] < y0) y0 = boxes[base + 1];
        if (boxes[base + 2] < z0) z0 = boxes[base + 2];
        if (boxes[base + 3] > x1) x1 = boxes[base + 3];
        if (boxes[base + 4] > y1) y1 = boxes[base + 4];
        if (boxes[base + 5] > z1) z1 = boxes[base + 5];
    }
    out[0] = x0;
    out[1] = y0;
    out[2] = z0;
    out[3] = x1;
    out[4] = y1;
    out[5] = z1;
}

/**
 * Splits `order[first..first+count)` in place and returns the left count (1
 * to `count - 1`): at the middle of the centres' longest axis (three-mesh-bvh's
 * `CENTER`), or at the median along it when `median` is set or the middle
 * leaves a side empty.
 */
function splitRange(centres: Float64Array, order: Uint32Array, first: number, count: number, median: boolean): number {
    const end = first + count;
    let axis = 0;
    let widest = -1;
    let low = 0;
    let high = 0;
    for (let a = 0; a < 3; a++) {
        let min = Infinity;
        let max = -Infinity;
        for (let i = first; i < end; i++) {
            const value = centres[order[i] * 3 + a];
            if (value < min) min = value;
            if (value > max) max = value;
        }
        if (max - min > widest) {
            widest = max - min;
            axis = a;
            low = min;
            high = max;
        }
    }
    const half = count >> 1;
    if (widest <= 0) return half; // every centre the same: any halves
    if (!median) {
        const pivot = (low + high) * 0.5;
        let left = first;
        let right = end - 1;
        while (left <= right) {
            if (centres[order[left] * 3 + axis] < pivot) {
                left++;
            } else {
                const swap = order[left];
                order[left] = order[right];
                order[right] = swap;
                right--;
            }
        }
        const leftCount = left - first;
        if (leftCount > 0 && leftCount < count) return leftCount;
    }
    selectNth(centres, order, axis, first, end - 1, first + half);
    return half;
}

/** Quickselect: puts the `nth` smallest centre along `axis` at `nth`, smaller ones before it. */
function selectNth(centres: Float64Array, order: Uint32Array, axis: number, lo: number, hi: number, nth: number): void {
    while (hi > lo) {
        const pivot = centres[order[(lo + hi) >> 1] * 3 + axis];
        let i = lo;
        let j = hi;
        while (i <= j) {
            while (centres[order[i] * 3 + axis] < pivot) i++;
            while (centres[order[j] * 3 + axis] > pivot) j--;
            if (i <= j) {
                const swap = order[i];
                order[i] = order[j];
                order[j] = swap;
                i++;
                j--;
            }
        }
        if (nth <= j) hi = j;
        else if (nth >= i) lo = i;
        else return;
    }
}

/** A child slot's box (slot = node * 2 + side). */
function readSlotBox(floats: Float32Array, slot: number, box: Float32Array): void {
    const base = (slot >> 1) * GPU_LAYOUT_NODE_WORDS + (slot & 1) * 8;
    box[0] = floats[base];
    box[1] = floats[base + 1];
    box[2] = floats[base + 2];
    box[3] = floats[base + 4];
    box[4] = floats[base + 5];
    box[5] = floats[base + 6];
}

function slotIndex(words: Uint32Array, slot: number): number {
    return words[(slot >> 1) * GPU_LAYOUT_NODE_WORDS + ((slot & 1) === 0 ? LEFT_INDEX : RIGHT_INDEX)];
}

function slotCount(words: Uint32Array, slot: number): number {
    return words[(slot >> 1) * GPU_LAYOUT_NODE_WORDS + ((slot & 1) === 0 ? LEFT_COUNT : RIGHT_COUNT)];
}

/**
 * Checks that the nodes are one tree from node 0 whose leaves, left subtree
 * first, cover the tree positions in order, each once (so every subtree is one
 * run, as `traverse` reports a contained box). Returns its depth (the root's
 * children at 1). Throws a `RangeError` naming the first problem.
 */
function checkTree(words: Uint32Array, nodeCount: number, primitiveCount: number, where: string): number {
    if (nodeCount === 0) {
        if (primitiveCount !== 0) throw new RangeError(`${where}: ${primitiveCount} primitives and no nodes`);
        return 0;
    }
    const reached = new Uint8Array(nodeCount);
    let next = 0;
    let deepest = 0;
    // Slots, right pushed first so leaves come out left first.
    const slots: number[] = [];
    const depths: number[] = [];
    const visit = (node: number, depth: number): void => {
        if (node >= nodeCount) throw new RangeError(`${where}: a child points past the ${nodeCount} nodes`);
        if (reached[node]) throw new RangeError(`${where}: node ${node} is reached twice`);
        reached[node] = 1;
        if (words[node * GPU_LAYOUT_NODE_WORDS + RIGHT_INDEX] !== GPU_LAYOUT_EMPTY_CHILD) {
            slots.push(node * 2 + 1);
            depths.push(depth);
        } else if (node !== 0 || slotCount(words, 0) === 0) {
            throw new RangeError(`${where}: node ${node} has no right child`);
        }
        slots.push(node * 2);
        depths.push(depth);
    };
    visit(0, 1);
    while (slots.length > 0) {
        const slot = slots.pop()!;
        const depth = depths.pop()!;
        if (depth > deepest) deepest = depth;
        const index = slotIndex(words, slot);
        const count = slotCount(words, slot);
        if (count === 0) {
            visit(index, depth + 1);
            continue;
        }
        if (count > GPU_LAYOUT_MAX_LEAF_PRIMITIVES) throw new RangeError(`${where}: a leaf holds ${count} primitives, past ${GPU_LAYOUT_MAX_LEAF_PRIMITIVES}`);
        if (index !== next) throw new RangeError(`${where}: the leaves don't cover the tree positions in order (expected ${next}, got ${index})`);
        next += count;
    }
    if (next !== primitiveCount) throw new RangeError(`${where}: the leaves cover ${next} of ${primitiveCount} primitives`);
    for (let node = 0; node < nodeCount; node++) if (!reached[node]) throw new RangeError(`${where}: node ${node} is not reached from the root`);
    return slotCount(words, 0) > 0 && words[RIGHT_INDEX] === GPU_LAYOUT_EMPTY_CHILD ? 0 : deepest;
}

/** Normalizes `d` as three's `Vector3.normalize` does, unless it is unit length already (three's rays are; their bits are kept). */
function unitDirection(direction: BvhVec3Like): [number, number, number] {
    const d = readVec3(direction);
    const lengthSq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    if (Math.abs(lengthSq - 1) > 1e-12) {
        const inverse = 1 / (Math.sqrt(lengthSq) || 1);
        d[0] *= inverse;
        d[1] *= inverse;
        d[2] *= inverse;
    }
    return d;
}

/**
 * Where the ray enters `box` grown by `grow` on every side: the distance
 * along the unit ray (0 when the origin is inside), or -1 when it misses.
 */
function enterGrownBox(box: Float32Array, grow: number, o: readonly number[], inverse: readonly number[]): number {
    let near = 0;
    let far = Infinity;
    for (let axis = 0; axis < 3; axis++) {
        const min = box[axis] - grow;
        const max = box[3 + axis] + grow;
        const inv = inverse[axis];
        if (inv === Infinity || inv === -Infinity) {
            // Parallel to these planes: inside them or never.
            if (o[axis] < min || o[axis] > max) return -1;
            continue;
        }
        let t0 = (min - o[axis]) * inv;
        let t1 = (max - o[axis]) * inv;
        if (t0 > t1) {
            const swap = t0;
            t0 = t1;
            t1 = swap;
        }
        if (t0 > near) near = t0;
        if (t1 < far) far = t1;
        if (near > far) return -1;
    }
    return near;
}

/** Squared distance from a point to a box (0 inside). */
function boxDistanceSq(box: Float32Array, x: number, y: number, z: number): number {
    const dx = x < box[0] ? box[0] - x : x > box[3] ? x - box[3] : 0;
    const dy = y < box[1] ? box[1] - y : y > box[4] ? y - box[4] : 0;
    const dz = z < box[2] ? box[2] - z : z > box[5] ? z - box[5] : 0;
    return dx * dx + dy * dy + dz * dz;
}

/** Scratch for one primitive test: point on the ray (0-2), on the primitive (3-5). */
const TEST = new Float64Array(6);

/**
 * The shared part of {@link CpuPointsBvh} and {@link CpuSegmentsBvh}: the
 * nodes, the box walk with `CpuMeshBvh.traverse`'s contract, the ray walk
 * with a threshold, closest point, refit and saving.
 */
export abstract class CpuPrimitiveBvh {
    readonly layout: BvhPrimitiveLayout;
    /** The input the tree was built from, if known (by reference). */
    protected source: BvhPointsInput | BvhSegmentsInput | null;
    protected readonly nodeFloats: Float32Array;
    protected readonly nodeWords: Uint32Array;
    protected readonly recordFloats: Float32Array;
    protected readonly recordWords: Uint32Array;
    /** Words per record. */
    protected readonly stride: number;
    private readonly treeDepth: number;

    constructor(layout: BvhPrimitiveLayout, options: BvhPrimitiveTreeOptions = {}) {
        const where = layout.kind === 'points' ? 'CpuPointsBvh' : 'CpuSegmentsBvh';
        const recordBytes = layout.kind === 'points' ? POINT_RECORD_BYTES : SEGMENT_RECORD_BYTES;
        if (layout.nodes.byteLength !== layout.nodeCount * GPU_LAYOUT_NODE_BYTES) throw new RangeError(`${where}: ${layout.nodeCount} nodes need ${layout.nodeCount * GPU_LAYOUT_NODE_BYTES} bytes, got ${layout.nodes.byteLength}`);
        if (layout.primitives.byteLength !== layout.primitiveCount * recordBytes) throw new RangeError(`${where}: ${layout.primitiveCount} primitives need ${layout.primitiveCount * recordBytes} bytes, got ${layout.primitives.byteLength}`);
        if (layout.bounds.length < 6) throw new RangeError(`${where}: the layout has no bounds`);
        this.layout = layout;
        this.source = options.source ?? null;
        this.nodeFloats = new Float32Array(layout.nodes);
        this.nodeWords = new Uint32Array(layout.nodes);
        this.recordFloats = new Float32Array(layout.primitives);
        this.recordWords = new Uint32Array(layout.primitives);
        this.stride = recordBytes / 4;
        this.treeDepth = checkTree(this.nodeWords, layout.nodeCount, layout.primitiveCount, where);
        for (let position = 0; position < layout.primitiveCount; position++) {
            const entry = this.recordWords[position * this.stride + PRIMITIVE_RECORD_INDEX_WORD];
            if (entry < layout.start || entry >= layout.end) throw new RangeError(`${where}: tree position ${position} names entry ${entry}, outside ${layout.start}-${layout.end}`);
        }
    }

    get primitiveCount(): number {
        return this.layout.primitiveCount;
    }

    get nodeCount(): number {
        return this.layout.nodeCount;
    }

    /** The deepest box's depth, as {@link traverse} reports it (the root's children at 1). */
    get depth(): number {
        return this.treeDepth;
    }

    /** The entry of the primitive at a tree position (as {@link traverse} reports them). */
    resolvePrimitiveIndex(treePosition: number): number {
        return this.recordWords[treePosition * this.stride + PRIMITIVE_RECORD_INDEX_WORD];
    }

    /** Box around every primitive, current after {@link refit}, as `CpuMeshBvh.getBoundingBox`. */
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

    /** Bytes this object holds: nodes, records and bounds. Not the source. */
    estimateMemoryInBytes(): number {
        return this.layout.nodes.byteLength + this.layout.primitives.byteLength + this.layout.bounds.byteLength;
    }

    /**
     * The tree as bytes: a 64-byte header (`"BVHP"`, format 1, kind, mode,
     * entry range, counts, bounds), the nodes, the records. {@link CpuPointsBvh.deserialize}
     * and {@link CpuSegmentsBvh.deserialize} read it back, checked.
     */
    serialize(): ArrayBuffer {
        const { layout } = this;
        const bytes = new ArrayBuffer(FILE_HEADER_BYTES + layout.nodes.byteLength + layout.primitives.byteLength);
        const words = new Uint32Array(bytes, 0, FILE_HEADER_BYTES / 4);
        words[FILE_HEADER_WORD.magic] = FILE_MAGIC;
        words[FILE_HEADER_WORD.version] = FILE_VERSION;
        words[FILE_HEADER_WORD.kind] = KIND_CODE[layout.kind];
        words[FILE_HEADER_WORD.mode] = MODE_CODE[layout.mode];
        words[FILE_HEADER_WORD.start] = layout.start;
        words[FILE_HEADER_WORD.end] = layout.end;
        words[FILE_HEADER_WORD.nodeCount] = layout.nodeCount;
        words[FILE_HEADER_WORD.primitiveCount] = layout.primitiveCount;
        new Float32Array(bytes, FILE_HEADER_WORD.bounds * 4, 6).set(layout.bounds.subarray(0, 6));
        new Uint8Array(bytes, FILE_HEADER_BYTES, layout.nodes.byteLength).set(new Uint8Array(layout.nodes));
        new Uint8Array(bytes, FILE_HEADER_BYTES + layout.nodes.byteLength).set(new Uint8Array(layout.primitives));
        return bytes;
    }

    /**
     * Walks the tree with `CpuMeshBvh.traverse`'s contract (three-mesh-bvh's
     * `shapecast`): depth first, the root's box never asked about, each child
     * scored once by `boundsTraverseOrder` and the lower score first, a falsy
     * `intersectsBounds` skipping the box, a contained box reported as one
     * range, a truthy `intersectsRange` stopping the walk. Ranges are tree
     * positions: map them with {@link resolvePrimitiveIndex}. Returns true when stopped.
     */
    traverse(callbacks: BvhTraverseCallbacks): boolean {
        const { intersectsBounds, intersectsRange, boundsTraverseOrder: order } = callbacks;
        if (this.layout.nodeCount === 0) return false;
        const words = this.nodeWords;
        const floats = this.nodeFloats;
        const box = new Float32Array(6);
        if (words[RIGHT_INDEX] === GPU_LAYOUT_EMPTY_CHILD) {
            readSlotBox(floats, 0, box);
            return Boolean(intersectsRange.call(callbacks, words[LEFT_INDEX], words[LEFT_COUNT], false, 0, 0, box));
        }
        const slots: number[] = [];
        const depths: number[] = [];
        const scores: (number | undefined)[] = [];
        const pushChildren = (node: number, depth: number): void => {
            let first = node * 2;
            let second = node * 2 + 1;
            let firstScore: number | undefined;
            let secondScore: number | undefined;
            if (order) {
                readSlotBox(floats, first, box);
                firstScore = order.call(callbacks, box);
                readSlotBox(floats, second, box);
                secondScore = order.call(callbacks, box);
                if (secondScore < firstScore) {
                    [first, second] = [second, first];
                    [firstScore, secondScore] = [secondScore, firstScore];
                }
            }
            slots.push(second);
            depths.push(depth);
            scores.push(secondScore);
            slots.push(first);
            depths.push(depth);
            scores.push(firstScore);
        };
        pushChildren(0, 1);
        while (slots.length > 0) {
            const slot = slots.pop()!;
            const depth = depths.pop()!;
            const score = scores.pop();
            const index = slotIndex(words, slot);
            const count = slotCount(words, slot);
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

    /** A contained box: its subtree's run of tree positions, in one call. */
    private reportContained(callbacks: BvhTraverseCallbacks, slot: number, depth: number, box: Float32Array): boolean {
        const words = this.nodeWords;
        let first = slot;
        while (slotCount(words, first) === 0) first = slotIndex(words, first) * 2;
        let last = slot;
        while (slotCount(words, last) === 0) last = slotIndex(words, last) * 2 + 1;
        const offset = slotIndex(words, first);
        const end = slotIndex(words, last) + slotCount(words, last);
        return Boolean(callbacks.intersectsRange(offset, end - offset, true, depth, slot, box));
    }

    /**
     * three-mesh-bvh's node walk, as `CpuMeshBvh.traverseNodes` (for the
     * helper and the debug functions): every node depth first, the root at
     * depth 0 with the tree's bounds; a leaf gets its first tree position and
     * count, an inner node the axis its children's centres lie furthest apart on.
     */
    traverseNodes(callback: (depth: number, isLeaf: boolean, box: Float32Array, offsetOrSplit: number, count: number) => unknown): void {
        if (this.layout.nodeCount === 0) return;
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
        const rootBox = Float32Array.from(this.layout.bounds.subarray(0, 6));
        if (words[RIGHT_INDEX] === GPU_LAYOUT_EMPTY_CHILD) {
            callback(0, true, rootBox, words[LEFT_INDEX], words[LEFT_COUNT]);
            return;
        }
        if (callback(0, false, rootBox, splitAxis(0), 0)) return;
        const slots = [1, 0];
        const depths = [1, 1];
        while (slots.length > 0) {
            const slot = slots.pop()!;
            const depth = depths.pop()!;
            const index = slotIndex(words, slot);
            const count = slotCount(words, slot);
            const box = new Float32Array(6);
            readSlotBox(floats, slot, box);
            if (count > 0) {
                callback(depth, true, box, index, count);
                continue;
            }
            if (callback(depth, false, box, splitAxis(index), 0)) continue;
            slots.push(index * 2 + 1, index * 2);
            depths.push(depth + 1, depth + 1);
        }
    }

    /**
     * The box around the primitives at tree positions `offset` to
     * `offset + count - 1`: read from the source when the tree has one (so
     * edits since the last {@link refit} show, as `validateBounds` wants),
     * else from the records.
     */
    writePrimitiveRangeBounds(offset: number, count: number, target: Float32Array | Float64Array | number[], targetOffset = 0): void {
        for (let axis = 0; axis < 3; axis++) {
            target[targetOffset + axis] = Infinity;
            target[targetOffset + 3 + axis] = -Infinity;
        }
        const corners = this.layout.kind === 'points' ? 1 : 2;
        const point = [0, 0, 0];
        for (let position = offset, end = offset + count; position < end; position++) {
            for (let corner = 0; corner < corners; corner++) {
                this.readCorner(position, corner, point, this.source);
                for (let axis = 0; axis < 3; axis++) {
                    const value = point[axis];
                    if (value < target[targetOffset + axis]) target[targetOffset + axis] = value;
                    if (value > target[targetOffset + 3 + axis]) target[targetOffset + 3 + axis] = value;
                }
            }
        }
    }

    /** A primitive's corner (0, or 1 for a segment's end): from `source` when given, else the record. */
    private readCorner(position: number, corner: number, out: number[], source: BvhPointsInput | null): void {
        if (source) {
            const { start, end, mode } = this.layout;
            const entry = this.resolvePrimitiveIndex(position);
            const at = corner === 0 ? entry : segmentEndEntry(mode, start, end, entry);
            const vertex = source.indices ? source.indices[at] : at;
            for (let axis = 0; axis < 3; axis++) out[axis] = Math.fround(source.positions[vertex * 3 + axis]);
        } else {
            const base = position * this.stride + corner * 4;
            for (let axis = 0; axis < 3; axis++) out[axis] = this.recordFloats[base + axis];
        }
    }

    /**
     * Rereads the primitives' positions (from `input`, else the source the
     * tree was built from) and refits every box, keeping the tree's shape:
     * three-mesh-bvh's `refit()`. The index and range must be the build's.
     */
    refit(input?: BvhPointsInput | BvhSegmentsInput): void {
        const where = this.layout.kind === 'points' ? 'CpuPointsBvh.refit' : 'CpuSegmentsBvh.refit';
        const source = input ?? this.source;
        if (!source) throw new Error(`${where}: the tree has no source; pass the input`);
        const entries = entriesOf({ ...source, start: this.layout.start, count: this.layout.end - this.layout.start }, where);
        if (entries.end !== this.layout.end) throw new RangeError(`${where}: the input has entries up to ${entries.end}, the tree up to ${this.layout.end}`);
        writeRecords(this.layout.kind, this.layout.mode, entries, source.positions, this.recordFloats, this.recordWords, true);
        if (input) this.source = input;
        this.refitBoxes();
    }

    /** The boxes from the records, nodes last to first (children follow their parents in these trees, so a single pass does it). */
    private refitBoxes(): void {
        const { nodeCount } = this.layout;
        const words = this.nodeWords;
        const floats = this.nodeFloats;
        const box = new Float32Array(6);
        const child = new Float32Array(6);
        // Children follow parents in built trees; a deserialized tree is checked for it here.
        for (let node = nodeCount - 1; node >= 0; node--) {
            for (let side = 0; side < 2; side++) {
                const slot = node * 2 + side;
                const index = slotIndex(words, slot);
                const count = slotCount(words, slot);
                if (side === 1 && index === GPU_LAYOUT_EMPTY_CHILD) continue;
                if (count > 0) {
                    this.recordRangeBox(index, count, box);
                } else {
                    if (index <= node) throw new RangeError('refit: a child comes before its parent; this tree cannot be refitted in one pass');
                    readSlotBox(floats, index * 2, box);
                    if (words[index * GPU_LAYOUT_NODE_WORDS + RIGHT_INDEX] !== GPU_LAYOUT_EMPTY_CHILD) {
                        readSlotBox(floats, index * 2 + 1, child);
                        for (let axis = 0; axis < 3; axis++) {
                            box[axis] = Math.min(box[axis], child[axis]);
                            box[3 + axis] = Math.max(box[3 + axis], child[3 + axis]);
                        }
                    }
                }
                const base = node * GPU_LAYOUT_NODE_WORDS + side * 8;
                for (let axis = 0; axis < 3; axis++) {
                    floats[base + axis] = box[axis];
                    floats[base + 4 + axis] = box[3 + axis];
                }
            }
        }
        const bounds = this.layout.bounds;
        this.recordRangeBox(0, this.layout.primitiveCount, box);
        for (let i = 0; i < 6; i++) bounds[i] = box[i];
    }

    private recordRangeBox(offset: number, count: number, out: Float32Array): void {
        const corners = this.layout.kind === 'points' ? 1 : 2;
        const floats = this.recordFloats;
        for (let axis = 0; axis < 3; axis++) {
            out[axis] = Infinity;
            out[3 + axis] = -Infinity;
        }
        for (let position = offset, end = offset + count; position < end; position++) {
            for (let corner = 0; corner < corners; corner++) {
                const base = position * this.stride + corner * 4;
                for (let axis = 0; axis < 3; axis++) {
                    const value = floats[base + axis];
                    if (value < out[axis]) out[axis] = value;
                    if (value > out[3 + axis]) out[3 + axis] = value;
                }
            }
        }
    }

    /** The test of one primitive against a ray: the squared distance, points into {@link TEST}. */
    protected abstract testRay(position: number, o: readonly number[], d: readonly number[]): number;

    /** Whether a squared distance is within the threshold: strictly for points, three's `<`; for segments three's `!(distSq > thresholdSq)`. */
    protected abstract withinThreshold(distanceSq: number, thresholdSq: number): boolean;

    /** The squared distance from `p` to the primitive at a tree position; its nearest point into `out[0..2]`. */
    protected abstract closestTo(position: number, px: number, py: number, pz: number, out: Float64Array): number;

    /**
     * Every primitive within `threshold` of the ray, sorted by distance (then
     * entry): three's `Points.raycast` / `Line.raycast` in the tree's space.
     * The direction is normalized first unless it is unit length (three's rays are).
     */
    raycast(ray: BvhRayLike, options: BvhPrimitiveRaycastOptions = {}): BvhPrimitiveRaycastHit[] {
        return this.cast(ray, options, false);
    }

    /** The closest primitive within `threshold` of the ray (ties: the lower entry), or null: three-mesh-bvh's `firstHitOnly`. */
    raycastFirst(ray: BvhRayLike, options: BvhPrimitiveRaycastOptions = {}): BvhPrimitiveRaycastHit | null {
        return this.cast(ray, options, true)[0] ?? null;
    }

    private cast(ray: BvhRayLike, options: BvhPrimitiveRaycastOptions, firstHitOnly: boolean): BvhPrimitiveRaycastHit[] {
        const hits: BvhPrimitiveRaycastHit[] = [];
        if (this.layout.nodeCount === 0) return hits;
        const threshold = options.threshold ?? 1;
        const near = options.near ?? 0;
        const far = options.far ?? Infinity;
        const accept = options.accept;
        const thresholdSq = threshold * threshold;
        const o = readVec3(ray.origin);
        const d = unitDirection(ray.direction);
        const inverse = [1 / d[0], 1 / d[1], 1 / d[2]];
        // Boxes grow by the threshold (three-mesh-bvh's expandByScalar) and a little more, so that
        // rounding in the box test never drops a primitive three's test keeps.
        const bounds = this.layout.bounds;
        const scale = Math.max(Math.abs(bounds[0]), Math.abs(bounds[1]), Math.abs(bounds[2]), Math.abs(bounds[3]), Math.abs(bounds[4]), Math.abs(bounds[5]), Math.abs(o[0]), Math.abs(o[1]), Math.abs(o[2]));
        const grow = Math.max(threshold, 0) + (Math.max(threshold, 0) + scale) * 1e-10;
        let best = null as BvhPrimitiveRaycastHit | null;
        const box = new Float32Array(6);
        const words = this.nodeWords;
        const floats = this.nodeFloats;
        const testLeaf = (offset: number, count: number): void => {
            for (let position = offset, end = offset + count; position < end; position++) {
                const distanceSq = this.testRay(position, o, d);
                if (!this.withinThreshold(distanceSq, thresholdSq)) continue;
                // raycaster.ray.origin.distanceTo(intersectPoint), in the tree's space
                const ex = o[0] - TEST[0];
                const ey = o[1] - TEST[1];
                const ez = o[2] - TEST[2];
                const distance = Math.sqrt(ex * ex + ey * ey + ez * ez);
                if (distance < near || distance > far) continue;
                const index = this.resolvePrimitiveIndex(position);
                if (firstHitOnly && best && (distance > best.distance || (distance === best.distance && index > best.index))) continue;
                const hit: BvhPrimitiveRaycastHit = {
                    index,
                    distance,
                    distanceToRay: Math.sqrt(distanceSq),
                    pointOnRay: [TEST[0], TEST[1], TEST[2]],
                    pointOnPrimitive: [TEST[3], TEST[4], TEST[5]],
                };
                if (accept && !accept(hit)) continue;
                if (firstHitOnly) best = hit;
                else hits.push(hit);
            }
        };
        if (words[RIGHT_INDEX] === GPU_LAYOUT_EMPTY_CHILD) {
            testLeaf(words[LEFT_INDEX], words[LEFT_COUNT]);
        } else {
            // Stack of child slots with their entry distances; nearer popped first.
            const slots: number[] = [];
            const entries: number[] = [];
            const pushChildren = (node: number): void => {
                readSlotBox(floats, node * 2, box);
                const left = enterGrownBox(box, grow, o, inverse);
                readSlotBox(floats, node * 2 + 1, box);
                const right = enterGrownBox(box, grow, o, inverse);
                const pushSlot = (slot: number, entry: number) => {
                    if (entry < 0) return;
                    slots.push(slot);
                    entries.push(entry);
                };
                if (right > left) {
                    pushSlot(node * 2 + 1, right);
                    pushSlot(node * 2, left);
                } else {
                    pushSlot(node * 2, left);
                    pushSlot(node * 2 + 1, right);
                }
            };
            pushChildren(0);
            while (slots.length > 0) {
                const slot = slots.pop()!;
                const entry = entries.pop()!;
                // firstHitOnly: a box entered past the closest hit holds nothing nearer.
                if (firstHitOnly && best && entry > best.distance) continue;
                const index = slotIndex(words, slot);
                const count = slotCount(words, slot);
                if (count > 0) testLeaf(index, count);
                else pushChildren(index);
            }
        }
        if (firstHitOnly) return best ? [best] : [];
        hits.sort((a, b) => a.distance - b.distance || a.index - b.index);
        return hits;
    }

    /**
     * The primitive nearest `point` and the nearest point on it (ties: the
     * lower entry), as `CpuMeshBvh.closestPointToPoint`: null when none is
     * within `maxThreshold`; the walk stops at the first one within
     * `minThreshold`. `target.point` is written in place when present.
     */
    closestPointToPoint(point: BvhVec3Like, target: Partial<BvhPrimitiveClosestPoint> = {}, minThreshold = 0, maxThreshold = Infinity): BvhPrimitiveClosestPoint | null {
        if (this.layout.nodeCount === 0) return null;
        const [px, py, pz] = readVec3(point);
        let bestSq = maxThreshold * maxThreshold;
        let bestIndex = -1;
        let found = false;
        const minSq = minThreshold * minThreshold;
        const near = new Float64Array(3);
        const best = new Float64Array(3);
        const words = this.nodeWords;
        const floats = this.nodeFloats;
        const box = new Float32Array(6);
        const testLeaf = (offset: number, count: number): boolean => {
            for (let position = offset, end = offset + count; position < end; position++) {
                const distanceSq = this.closestTo(position, px, py, pz, near);
                if (distanceSq > bestSq) continue;
                const index = this.resolvePrimitiveIndex(position);
                if (found && distanceSq === bestSq && index > bestIndex) continue;
                bestSq = distanceSq;
                bestIndex = index;
                found = true;
                best.set(near);
                if (distanceSq <= minSq) return true;
            }
            return false;
        };
        if (words[RIGHT_INDEX] === GPU_LAYOUT_EMPTY_CHILD) {
            testLeaf(words[LEFT_INDEX], words[LEFT_COUNT]);
        } else {
            const slots: number[] = [];
            const scores: number[] = [];
            const pushChildren = (node: number): void => {
                readSlotBox(floats, node * 2, box);
                const left = boxDistanceSq(box, px, py, pz);
                readSlotBox(floats, node * 2 + 1, box);
                const right = boxDistanceSq(box, px, py, pz);
                if (right > left) {
                    slots.push(node * 2 + 1, node * 2);
                    scores.push(right, left);
                } else {
                    slots.push(node * 2, node * 2 + 1);
                    scores.push(left, right);
                }
            };
            pushChildren(0);
            while (slots.length > 0) {
                const slot = slots.pop()!;
                const score = scores.pop()!;
                if (score > bestSq) continue;
                const index = slotIndex(words, slot);
                const count = slotCount(words, slot);
                if (count > 0) {
                    if (testLeaf(index, count)) break;
                } else {
                    pushChildren(index);
                }
            }
        }
        if (!found) return null;
        const out = target.point ?? new BvhVector3();
        out.x = best[0];
        out.y = best[1];
        out.z = best[2];
        target.point = out;
        target.distance = Math.sqrt(bestSq);
        target.index = bestIndex;
        return target as BvhPrimitiveClosestPoint;
    }

    /** The one shapecast: {@link traverse}, each primitive of a reached range loaded into `scratch` for `callbacks.primitive`. */
    protected shapecastWith<P, B extends BvhBox3Like>(callbacks: BvhPrimitiveShapecastCallbacks<P, B>, scratch: P, box: B, load: (position: number, into: P) => void): boolean {
        const { intersectsBounds, intersectsRange, boundsTraverseOrder, primitive } = callbacks;
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
        if (primitive) {
            range = (offset, count, contained, depth, nodeIndex, array) => {
                if (intersectsRange && intersectsRange.call(callbacks, offset, count, contained, depth, nodeIndex, fill(array))) return true;
                for (let position = offset, end = offset + count; position < end; position++) {
                    load(position, scratch);
                    if (primitive(scratch, this.resolvePrimitiveIndex(position), contained, depth)) return true;
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
}

/** Reads a file from {@link CpuPrimitiveBvh.serialize}, checking its header and sizes. */
function layoutFromBytes(bytes: ArrayBuffer | ArrayBufferView, kind: BvhPrimitiveKind): BvhPrimitiveLayout {
    const where = kind === 'points' ? 'CpuPointsBvh.deserialize' : 'CpuSegmentsBvh.deserialize';
    const view = bytes instanceof ArrayBuffer ? new Uint8Array(bytes) : new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (view.byteLength < FILE_HEADER_BYTES) throw new RangeError(`${where}: ${view.byteLength} bytes is shorter than the header`);
    const own = view.slice().buffer;
    const words = new Uint32Array(own, 0, FILE_HEADER_BYTES / 4);
    if (words[FILE_HEADER_WORD.magic] !== FILE_MAGIC) throw new RangeError(`${where}: not a primitive tree (no "BVHP" magic)`);
    if (words[FILE_HEADER_WORD.version] !== FILE_VERSION) throw new RangeError(`${where}: format ${words[FILE_HEADER_WORD.version]}, this build reads ${FILE_VERSION}`);
    const fileKind = KINDS[words[FILE_HEADER_WORD.kind]];
    if (fileKind !== kind) throw new RangeError(`${where}: the file holds ${fileKind ?? 'an unknown kind'}, not ${kind}`);
    const mode = MODES[words[FILE_HEADER_WORD.mode]];
    if (!mode) throw new RangeError(`${where}: unknown mode ${words[FILE_HEADER_WORD.mode]}`);
    const nodeCount = words[FILE_HEADER_WORD.nodeCount];
    const primitiveCount = words[FILE_HEADER_WORD.primitiveCount];
    const recordBytes = kind === 'points' ? POINT_RECORD_BYTES : SEGMENT_RECORD_BYTES;
    const nodeBytes = nodeCount * GPU_LAYOUT_NODE_BYTES;
    const expected = FILE_HEADER_BYTES + nodeBytes + primitiveCount * recordBytes;
    if (view.byteLength !== expected) throw new RangeError(`${where}: ${nodeCount} nodes and ${primitiveCount} primitives need ${expected} bytes, got ${view.byteLength}`);
    return {
        kind,
        mode,
        start: words[FILE_HEADER_WORD.start],
        end: words[FILE_HEADER_WORD.end],
        nodes: own.slice(FILE_HEADER_BYTES, FILE_HEADER_BYTES + nodeBytes),
        nodeCount,
        primitives: own.slice(FILE_HEADER_BYTES + nodeBytes),
        primitiveCount,
        bounds: new Float32Array(own.slice(FILE_HEADER_WORD.bounds * 4, FILE_HEADER_WORD.bounds * 4 + 24)),
    };
}

/**
 * A tree over points (three-mesh-bvh's `PointsBVH`), engine-free: raycasts
 * with three's `Points.raycast` threshold and answers, shapecasts, closest
 * point, refit and saving. Build with {@link CpuPointsBvh.build} (synchronous).
 */
export class CpuPointsBvh extends CpuPrimitiveBvh {
    constructor(layout: BvhPrimitiveLayout, options: BvhPrimitiveTreeOptions = {}) {
        if (layout.kind !== 'points') throw new RangeError(`CpuPointsBvh: the layout holds ${layout.kind}`);
        super(layout, options);
    }

    /** Builds a tree over the input's entries (see the file comment); keeps `input` as its source. */
    static build(input: BvhPointsInput, options: BvhPrimitiveBuildOptions = {}): CpuPointsBvh {
        return new CpuPointsBvh(buildPrimitiveLayout('points', input, options), { source: input });
    }

    /** Reads bytes from {@link serialize}, checked. */
    static deserialize(bytes: ArrayBuffer | ArrayBufferView, options: BvhPrimitiveTreeOptions = {}): CpuPointsBvh {
        return new CpuPointsBvh(layoutFromBytes(bytes, 'points'), options);
    }

    /** The point at a tree position, as the tree stores it. */
    getPoint<T extends BvhVector3Like = BvhVector3>(treePosition: number, target?: T): T {
        const out = target ?? (new BvhVector3() as unknown as T);
        const base = treePosition * this.stride;
        out.x = this.recordFloats[base];
        out.y = this.recordFloats[base + 1];
        out.z = this.recordFloats[base + 2];
        return out;
    }

    /**
     * three-mesh-bvh's `PointsBVH.shapecast`: `intersectsBounds`,
     * `intersectsRange`, `boundsTraverseOrder` as `CpuMeshBvh.shapecast`'s, and
     * `intersectsPoint(point, index, contained, depth)` per point with its entry.
     */
    shapecast(callbacks: BvhPointsShapecastCallbacks<BvhVector3, BvhBox3>): boolean {
        return this.shapecastPoints(callbacks, new BvhVector3(), new BvhBox3());
    }

    /** {@link shapecast} with the caller's scratch point and box (an engine adapter passes its own types). */
    protected shapecastPoints<P extends BvhVector3Like, B extends BvhBox3Like>(callbacks: BvhPointsShapecastCallbacks<P, B>, point: P, box: B): boolean {
        const intersectsPoint = callbacks.intersectsPoint;
        return this.shapecastWith<P, B>(
            { ...callbacks, primitive: intersectsPoint ? (p, index, contained, depth) => intersectsPoint.call(callbacks, p, index, contained, depth) : undefined },
            point,
            box,
            (position, into) => this.getPoint(position, into),
        );
    }

    protected testRay(position: number, o: readonly number[], d: readonly number[]): number {
        const base = position * this.stride;
        const px = this.recordFloats[base];
        const py = this.recordFloats[base + 1];
        const pz = this.recordFloats[base + 2];
        TEST[3] = px;
        TEST[4] = py;
        TEST[5] = pz;
        return rayDistanceSqToPoint(o[0], o[1], o[2], d[0], d[1], d[2], px, py, pz, TEST, 0);
    }

    protected withinThreshold(distanceSq: number, thresholdSq: number): boolean {
        return distanceSq < thresholdSq;
    }

    protected closestTo(position: number, px: number, py: number, pz: number, out: Float64Array): number {
        const base = position * this.stride;
        out[0] = this.recordFloats[base];
        out[1] = this.recordFloats[base + 1];
        out[2] = this.recordFloats[base + 2];
        const dx = out[0] - px;
        const dy = out[1] - py;
        const dz = out[2] - pz;
        return dx * dx + dy * dy + dz * dz;
    }
}

/**
 * A tree over line segments (three-mesh-bvh's `LineSegmentsBVH`,
 * `LineLoopBVH` and `LineBVH`, by {@link BvhSegmentsInput.mode}),
 * engine-free: raycasts with three's `Line.raycast` threshold and answers,
 * shapecasts, closest point, refit and saving.
 */
export class CpuSegmentsBvh extends CpuPrimitiveBvh {
    constructor(layout: BvhPrimitiveLayout, options: BvhPrimitiveTreeOptions = {}) {
        if (layout.kind !== 'segments') throw new RangeError(`CpuSegmentsBvh: the layout holds ${layout.kind}`);
        super(layout, options);
    }

    /** Builds a tree over the input's segments (see the file comment); keeps `input` as its source. */
    static build(input: BvhSegmentsInput, options: BvhPrimitiveBuildOptions = {}): CpuSegmentsBvh {
        return new CpuSegmentsBvh(buildPrimitiveLayout('segments', input, options), { source: input });
    }

    /** Reads bytes from {@link serialize}, checked. */
    static deserialize(bytes: ArrayBuffer | ArrayBufferView, options: BvhPrimitiveTreeOptions = {}): CpuSegmentsBvh {
        return new CpuSegmentsBvh(layoutFromBytes(bytes, 'segments'), options);
    }

    /** How the tree's entries pair up. */
    get mode(): BvhSegmentMode {
        return this.layout.mode;
    }

    /** The segment at a tree position, as the tree stores it. */
    getSegment<T extends BvhSegmentLike = BvhSegment>(treePosition: number, target?: T): T {
        const out = target ?? (new BvhSegment() as unknown as T);
        const base = treePosition * this.stride;
        const f = this.recordFloats;
        out.start.x = f[base];
        out.start.y = f[base + 1];
        out.start.z = f[base + 2];
        out.end.x = f[base + 4];
        out.end.y = f[base + 5];
        out.end.z = f[base + 6];
        return out;
    }

    /**
     * three-mesh-bvh's `LineSegmentsBVH.shapecast`: `intersectsBounds`,
     * `intersectsRange`, `boundsTraverseOrder` as `CpuMeshBvh.shapecast`'s, and
     * `intersectsLine(line, index, contained, depth)` per segment with its entry.
     */
    shapecast(callbacks: BvhSegmentsShapecastCallbacks<BvhSegment, BvhBox3>): boolean {
        return this.shapecastSegments(callbacks, new BvhSegment(), new BvhBox3());
    }

    /** {@link shapecast} with the caller's scratch segment and box. */
    protected shapecastSegments<S extends BvhSegmentLike, B extends BvhBox3Like>(callbacks: BvhSegmentsShapecastCallbacks<S, B>, line: S, box: B): boolean {
        const intersectsLine = callbacks.intersectsLine;
        return this.shapecastWith<S, B>(
            { ...callbacks, primitive: intersectsLine ? (s, index, contained, depth) => intersectsLine.call(callbacks, s, index, contained, depth) : undefined },
            line,
            box,
            (position, into) => this.getSegment(position, into),
        );
    }

    protected testRay(position: number, o: readonly number[], d: readonly number[]): number {
        const base = position * this.stride;
        const f = this.recordFloats;
        return rayDistanceSqToSegment(o[0], o[1], o[2], d[0], d[1], d[2], f[base], f[base + 1], f[base + 2], f[base + 4], f[base + 5], f[base + 6], TEST, 0);
    }

    protected withinThreshold(distanceSq: number, thresholdSq: number): boolean {
        return !(distanceSq > thresholdSq);
    }

    protected closestTo(position: number, px: number, py: number, pz: number, out: Float64Array): number {
        const base = position * this.stride;
        const f = this.recordFloats;
        return segmentClosestPointToPoint(f[base], f[base + 1], f[base + 2], f[base + 4], f[base + 5], f[base + 6], px, py, pz, out, 0);
    }
}

/**
 * One segment against a ray with three's `Line.raycast` test, outside any
 * tree (`@pmndrs/bvh/three` tests a `LineLoop`'s closing segment for a draw
 * range this way): the hit, or null when the segment is farther than
 * `threshold`. The direction must be unit length.
 */
export function raySegmentHit(
    origin: readonly [number, number, number],
    direction: readonly [number, number, number],
    start: BvhVec3Like,
    end: BvhVec3Like,
    threshold: number,
    index: number,
): BvhPrimitiveRaycastHit | null {
    const [ax, ay, az] = readVec3(start);
    const [bx, by, bz] = readVec3(end);
    const out = new Float64Array(6);
    const distanceSq = rayDistanceSqToSegment(origin[0], origin[1], origin[2], direction[0], direction[1], direction[2], ax, ay, az, bx, by, bz, out, 0);
    if (distanceSq > threshold * threshold) return null;
    const ex = origin[0] - out[0];
    const ey = origin[1] - out[1];
    const ez = origin[2] - out[2];
    return { index, distance: Math.sqrt(ex * ex + ey * ey + ez * ez), distanceToRay: Math.sqrt(distanceSq), pointOnRay: [out[0], out[1], out[2]], pointOnPrimitive: [out[3], out[4], out[5]] };
}

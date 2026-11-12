/**
 * Indexed leaves (docs/internals/design/indexed-leaves.md): the 16-byte triangle format
 * `vec4u(i0, i1, i2, id + flags)` beside the 48-byte records, and the
 * helpers every builder and the scene share to write it.
 *
 * - {@link packIndexedTrianglesInOrder}: what a builder writes, from the
 *   geometry and its tree order (the indexed twin of `packTrianglesInOrder`).
 * - {@link indexedFromRecords} and {@link recordsFromIndexed}: one tree in the
 *   other format, the same nodes. tinybvh writes records inside its WASM, so
 *   its indexed trees are converted here (a WASM change would save the
 *   transient 48-byte copy).
 * - {@link weldExactBits} and {@link weldPositions}: the weld at load. An
 *   exact-bit weld merges vertices whose three coordinates have the same
 *   bits; nothing the walk computes changes (the watertight test reads bits,
 *   not indices), it only saves memory. A tolerance weld moves vertices and
 *   is opt-in.
 */

import {
    GPU_LAYOUT_INDEXED_FLAG_SHIFT,
    GPU_LAYOUT_INDEXED_ID_MASK,
    GPU_LAYOUT_INDEXED_TRIANGLE_BYTES,
    GPU_LAYOUT_TRIANGLE_BYTES,
    GPU_LAYOUT_TRIANGLE_WORD,
    type GpuLayoutMeshBvh,
    type GpuLayoutTriangleLeaves,
} from './gpu-layout.ts';
import { checkTriangleFlags, type MeshInput } from './mesh-input.ts';

const RECORD_WORDS = GPU_LAYOUT_TRIANGLE_BYTES / 4;
const INDEXED_WORDS = GPU_LAYOUT_INDEXED_TRIANGLE_BYTES / 4;
/** Record flags bits that move into an indexed word 3: 0 (filter) and 1 (flip). */
const MOVED_FLAGS = 3;

/** Checks a `triangleLeaves` option: undefined is `'records'`. */
export function checkTriangleLeaves(leaves: GpuLayoutTriangleLeaves | undefined, where: string): GpuLayoutTriangleLeaves {
    if (leaves === undefined || leaves === 'records') return 'records';
    if (leaves === 'indexed') return 'indexed';
    throw new RangeError(`${where}: triangleLeaves must be 'records' or 'indexed', got ${String(leaves)}`);
}

/** Word 3 of an indexed triangle: the original index and record flags bits 0-1. */
export function indexedIdFlags(original: number, recordFlags: number): number {
    return ((original & GPU_LAYOUT_INDEXED_ID_MASK) | ((recordFlags & MOVED_FLAGS) << GPU_LAYOUT_INDEXED_FLAG_SHIFT)) >>> 0;
}

/** Record flags bits 0-1 (filter, flip) from an indexed word 3. */
export function recordFlagsOfIndexed(idFlags: number): number {
    return (idFlags >>> GPU_LAYOUT_INDEXED_FLAG_SHIFT) & MOVED_FLAGS;
}

/** Indexed triangles in tree order (`order[position]` is the triangle there), 16 bytes each, mesh-relative vertex indices. */
export function packIndexedTrianglesInOrder(geometry: MeshInput, order: Uint32Array): ArrayBuffer {
    const { positions, indices, triangleFlags } = geometry;
    checkTriangleFlags(triangleFlags, order.length, 'mesh');
    const vertexCount = positions.length / 3;
    const buffer = new ArrayBuffer(order.length * GPU_LAYOUT_INDEXED_TRIANGLE_BYTES);
    const words = new Uint32Array(buffer);
    for (let position = 0; position < order.length; position++) {
        const triangle = order[position];
        const base = position * INDEXED_WORDS;
        for (let corner = 0; corner < 3; corner++) {
            const vertex = indices ? indices[triangle * 3 + corner] : triangle * 3 + corner;
            if (vertex >= vertexCount) throw new RangeError(`triangle ${triangle} uses vertex ${vertex}, but there are ${vertexCount}`);
            words[base + corner] = vertex;
        }
        words[base + 3] = indexedIdFlags(triangle, triangleFlags ? triangleFlags[triangle] : 0);
    }
    return buffer;
}

/**
 * The same tree with indexed triangles: each record's original index names
 * the geometry's triangle, whose vertex indices are written in corner order
 * (the records' corners are that triangle's, in that order), and flags bits
 * 0-1 move to bits 28-29. A merged record's material and source instance are
 * not carried (the scene's side table holds them): convert fresh trees.
 * An indexed layout comes back as it is.
 */
export function indexedFromRecords(layout: GpuLayoutMeshBvh, geometry: MeshInput): GpuLayoutMeshBvh {
    if (layout.triangleLeaves === 'indexed') return layout;
    const records = new Uint32Array(layout.triangles, 0, layout.triangleCount * RECORD_WORDS);
    return { ...layout, triangles: indexedTrianglesFromRecordWords(records, layout.triangleCount, geometry), triangleLeaves: 'indexed' };
}

/** Indexed triangles from 48-byte records' words (`records` may view another memory, tinybvh's WASM). */
export function indexedTrianglesFromRecordWords(records: Uint32Array, recordCount: number, geometry: MeshInput): ArrayBuffer {
    const { positions, indices } = geometry;
    const vertexCount = positions.length / 3;
    const triangleCount = indices ? indices.length / 3 : vertexCount / 3;
    const triangles = new ArrayBuffer(recordCount * GPU_LAYOUT_INDEXED_TRIANGLE_BYTES);
    const words = new Uint32Array(triangles);
    for (let slot = 0; slot < recordCount; slot++) {
        const triangle = records[slot * RECORD_WORDS + GPU_LAYOUT_TRIANGLE_WORD.originalIndex];
        if (triangle >= triangleCount) throw new RangeError(`indexedFromRecords: slot ${slot} holds triangle ${triangle}, the mesh has ${triangleCount}`);
        for (let corner = 0; corner < 3; corner++) words[slot * INDEXED_WORDS + corner] = indices ? indices[triangle * 3 + corner] : triangle * 3 + corner;
        words[slot * INDEXED_WORDS + 3] = indexedIdFlags(triangle, records[slot * RECORD_WORDS + GPU_LAYOUT_TRIANGLE_WORD.flags]);
    }
    return triangles;
}

/**
 * The same tree with 48-byte records: each indexed triangle's corners copied
 * from `positions` (`x, y, z` per vertex, the positions its indices name),
 * its original index and flags bits 0-1. A records layout comes back as it is.
 */
export function recordsFromIndexed(layout: GpuLayoutMeshBvh, positions: Float32Array): GpuLayoutMeshBvh {
    if (layout.triangleLeaves !== 'indexed') return layout;
    const indexed = new Uint32Array(layout.triangles, 0, layout.triangleCount * INDEXED_WORDS);
    const triangles = new ArrayBuffer(layout.triangleCount * GPU_LAYOUT_TRIANGLE_BYTES);
    const floats = new Float32Array(triangles);
    const words = new Uint32Array(triangles);
    const vertexCount = positions.length / 3;
    const corners = [GPU_LAYOUT_TRIANGLE_WORD.vertex0, GPU_LAYOUT_TRIANGLE_WORD.vertex1, GPU_LAYOUT_TRIANGLE_WORD.vertex2];
    for (let slot = 0; slot < layout.triangleCount; slot++) {
        const base = slot * RECORD_WORDS;
        for (let corner = 0; corner < 3; corner++) {
            const vertex = indexed[slot * INDEXED_WORDS + corner];
            if (vertex >= vertexCount) throw new RangeError(`recordsFromIndexed: slot ${slot} uses vertex ${vertex}, there are ${vertexCount}`);
            for (let axis = 0; axis < 3; axis++) floats[base + corners[corner] + axis] = positions[vertex * 3 + axis];
        }
        const idFlags = indexed[slot * INDEXED_WORDS + 3];
        words[base + GPU_LAYOUT_TRIANGLE_WORD.originalIndex] = idFlags & GPU_LAYOUT_INDEXED_ID_MASK;
        words[base + GPU_LAYOUT_TRIANGLE_WORD.flags] = recordFlagsOfIndexed(idFlags);
    }
    const { triangleLeaves: _leaves, ...rest } = layout;
    return { ...rest, triangles };
}

/** The result of a weld: unique positions and where each input vertex went. */
export interface WeldResult {
    /** `x, y, z` per kept vertex, in order of first use. */
    readonly positions: Float32Array;
    /** Per input vertex: its kept vertex. */
    readonly remap: Uint32Array;
    readonly vertexCount: number;
}

/**
 * Merges vertices whose three coordinates have identical bits (so `0` and
 * `-0` stay apart, and a NaN only merges with the same NaN bits). Exact: no
 * coordinate changes, so the walk computes the same bits as before.
 * Vertices keep the order of their first appearance.
 */
export function weldExactBits(positions: Float32Array): WeldResult {
    const count = positions.length / 3;
    const bits = new Uint32Array(positions.buffer, positions.byteOffset, count * 3);
    let size = 16;
    while (size < count * 2) size *= 2;
    const mask = size - 1;
    const table = new Int32Array(size).fill(-1);
    const remap = new Uint32Array(count);
    const kept = new Uint32Array(count);
    let keptCount = 0;
    for (let vertex = 0; vertex < count; vertex++) {
        const x = bits[vertex * 3];
        const y = bits[vertex * 3 + 1];
        const z = bits[vertex * 3 + 2];
        let hash = Math.imul(x, 0x9e3779b1) ^ Math.imul(y ^ (y >>> 15), 0x85ebca77) ^ Math.imul(z ^ (z >>> 13), 0xc2b2ae3d);
        hash ^= hash >>> 16;
        let at = hash & mask;
        for (;;) {
            const entry = table[at];
            if (entry < 0) {
                table[at] = keptCount;
                kept[keptCount] = vertex;
                remap[vertex] = keptCount++;
                break;
            }
            const other = kept[entry] * 3;
            if (bits[other] === x && bits[other + 1] === y && bits[other + 2] === z) {
                remap[vertex] = entry;
                break;
            }
            at = (at + 1) & mask;
        }
    }
    const out = new Float32Array(keptCount * 3);
    const outBits = new Uint32Array(out.buffer);
    for (let k = 0; k < keptCount; k++) {
        outBits[k * 3] = bits[kept[k] * 3];
        outBits[k * 3 + 1] = bits[kept[k] * 3 + 1];
        outBits[k * 3 + 2] = bits[kept[k] * 3 + 2];
    }
    return { positions: out, remap, vertexCount: keptCount };
}

/**
 * Merges vertices closer than `tolerance` on every axis into the first such
 * vertex seen (a grid of cells `tolerance` wide; each vertex looks in its
 * cell and the 26 around it). It *moves* geometry away from what the app
 * draws by up to `tolerance` per axis, so it is opt-in: it closes seams whose
 * copies differ by an ulp or so, which no exact weld or format can.
 */
export function weldWithinTolerance(positions: Float32Array, tolerance: number): WeldResult {
    if (!(tolerance > 0) || !Number.isFinite(tolerance)) throw new RangeError(`weld tolerance must be a positive finite number, got ${tolerance}`);
    const count = positions.length / 3;
    const cells = new Map<string, number[]>();
    const remap = new Uint32Array(count);
    const kept: number[] = [];
    const cellOf = (value: number): number => Math.floor(value / tolerance);
    for (let vertex = 0; vertex < count; vertex++) {
        const x = positions[vertex * 3];
        const y = positions[vertex * 3 + 1];
        const z = positions[vertex * 3 + 2];
        const cx = cellOf(x);
        const cy = cellOf(y);
        const cz = cellOf(z);
        let found = -1;
        for (let dx = -1; dx <= 1 && found < 0; dx++) {
            for (let dy = -1; dy <= 1 && found < 0; dy++) {
                for (let dz = -1; dz <= 1 && found < 0; dz++) {
                    const list = cells.get(`${cx + dx},${cy + dy},${cz + dz}`);
                    if (!list) continue;
                    for (const candidate of list) {
                        const at = kept[candidate] * 3;
                        if (Math.abs(positions[at] - x) <= tolerance && Math.abs(positions[at + 1] - y) <= tolerance && Math.abs(positions[at + 2] - z) <= tolerance) {
                            found = candidate;
                            break;
                        }
                    }
                }
            }
        }
        if (found < 0) {
            found = kept.length;
            kept.push(vertex);
            const key = `${cx},${cy},${cz}`;
            const list = cells.get(key);
            if (list) list.push(found);
            else cells.set(key, [found]);
        }
        remap[vertex] = found;
    }
    const out = new Float32Array(kept.length * 3);
    kept.forEach((vertex, k) => out.set(positions.subarray(vertex * 3, vertex * 3 + 3), k * 3));
    return { positions: out, remap, vertexCount: kept.length };
}

/** Options of {@link weldPositions}. */
export interface WeldOptions {
    /** 0 (the default): the exact-bit weld. Above 0: {@link weldWithinTolerance}, which moves vertices. */
    readonly tolerance?: number;
}

/**
 * The geometry with its vertices welded (`Uint32Array` indices into the kept
 * positions). Triangles keep their order, corner order, flags and materials, so original
 * triangle indices and barycentrics mean what they meant.
 */
export function weldPositions(geometry: MeshInput, options: WeldOptions = {}): MeshInput {
    const tolerance = options.tolerance ?? 0;
    const weld = tolerance > 0 ? weldWithinTolerance(geometry.positions, tolerance) : weldExactBits(geometry.positions);
    const source = geometry.indices;
    const entries = source ? source.length : geometry.positions.length / 3;
    const indices = new Uint32Array(entries);
    for (let entry = 0; entry < entries; entry++) indices[entry] = weld.remap[source ? source[entry] : entry];
    return { positions: weld.positions, indices, triangleFlags: geometry.triangleFlags, ...(geometry.triangleMaterials ? { triangleMaterials: geometry.triangleMaterials } : {}) };
}

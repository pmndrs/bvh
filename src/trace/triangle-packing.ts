/**
 * Turning mesh geometry into the tracer's triangle records. Shared by the two
 * ways to make a mesh BVH: our own builder (`mesh-bvh.ts`) and a tree built
 * elsewhere (`mesh-bvh-topology.ts`). Internal: not exported from the module index.
 */

import { MAX_REFERENCE_INDEX, TRIANGLE_BYTES, TRIANGLE_WORD, TRIANGLE_WORDS } from '../gpu-layout.ts';
import type { MeshGeometry } from './mesh-bvh.ts';
import { packedTriangleFlags } from '../mesh-input.ts';

export function countTriangles(geometry: MeshGeometry): number {
    const entries = geometry.indices ? geometry.indices.length : geometry.positions.length / 3;
    if (!Number.isInteger(entries) || entries % 3 !== 0) {
        throw new RangeError('triangle geometry must have a multiple of three vertices or indices');
    }
    return entries / 3;
}

/** Checked access to a mesh's triangles. */
export interface CheckedGeometry {
    readonly triangleCount: number;
    /** The flags word of each original triangle as the records hold it: the input's flags with its material in bits 16-31, or null for all 0. */
    readonly flags: Uint32Array | null;
    /** The vertex used by `corner` (0, 1 or 2) of `triangle`; throws if it does not exist. */
    vertexOf(triangle: number, corner: number): number;
}

/** Validates the geometry the way every mesh BVH needs it, and returns checked vertex access. */
export function checkGeometry(geometry: MeshGeometry): CheckedGeometry {
    const { positions, indices } = geometry;
    if (positions.length % 3 !== 0) throw new RangeError('positions must hold three floats per vertex');
    const vertexCount = positions.length / 3;
    const triangleCount = countTriangles(geometry);
    if (triangleCount === 0) throw new RangeError('a mesh BVH needs at least one triangle');
    if (triangleCount > MAX_REFERENCE_INDEX) throw new RangeError(`too many triangles for one BVH: ${triangleCount}`);
    const flags = packedTriangleFlags(geometry, triangleCount, 'mesh');
    return {
        triangleCount,
        flags,
        vertexOf(triangle, corner) {
            const vertex = indices ? indices[triangle * 3 + corner] : triangle * 3 + corner;
            if (vertex >= vertexCount) throw new RangeError(`triangle ${triangle} uses vertex ${vertex}, but there are ${vertexCount}`);
            return vertex;
        },
    };
}

/**
 * Grows `box` (at `offset`, `[minX, minY, minZ, maxX, maxY, maxZ]`) around
 * `triangle`'s three corners. Exact: the positions are already f32.
 */
export function growBoxByTriangle(
    box: Float32Array,
    offset: number,
    positions: Float32Array,
    checked: CheckedGeometry,
    triangle: number,
): void {
    for (let corner = 0; corner < 3; corner++) {
        const vertex = checked.vertexOf(triangle, corner) * 3;
        for (let axis = 0; axis < 3; axis++) {
            const value = positions[vertex + axis];
            if (value < box[offset + axis]) box[offset + axis] = value;
            if (value > box[offset + 3 + axis]) box[offset + 3 + axis] = value;
        }
    }
}

/**
 * Writes the triangles in tree order (`order[i]` is the original triangle at
 * position `i`), 48 bytes each. Corners are stored as they are, so a vertex
 * shared by two triangles has identical bits in both (the watertight test needs that).
 */
export function packTriangles(geometry: MeshGeometry, checked: CheckedGeometry, order: Uint32Array): ArrayBuffer {
    const { positions } = geometry;
    const { flags } = checked;
    const triangleCount = checked.triangleCount;
    const trianglesBuffer = new ArrayBuffer(triangleCount * TRIANGLE_BYTES);
    const floats = new Float32Array(trianglesBuffer);
    const words = new Uint32Array(trianglesBuffer);
    for (let slot = 0; slot < triangleCount; slot++) {
        const original = order[slot];
        const base = slot * TRIANGLE_WORDS;
        const a = checked.vertexOf(original, 0) * 3;
        const b = checked.vertexOf(original, 1) * 3;
        const c = checked.vertexOf(original, 2) * 3;
        for (let axis = 0; axis < 3; axis++) {
            floats[base + TRIANGLE_WORD.vertex0 + axis] = positions[a + axis];
            floats[base + TRIANGLE_WORD.vertex1 + axis] = positions[b + axis];
            floats[base + TRIANGLE_WORD.vertex2 + axis] = positions[c + axis];
        }
        words[base + TRIANGLE_WORD.originalIndex] = original;
        words[base + TRIANGLE_WORD.flags] = flags ? flags[original] : 0;
        words[base + TRIANGLE_WORD.padding] = 0;
    }
    return trianglesBuffer;
}

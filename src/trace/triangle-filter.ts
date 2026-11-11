/**
 * The walk's per-triangle filter hook, in the spirit of a DXR any-hit shader.
 *
 * A triangle whose `flags` word has {@link TRIANGLE_FLAG_FILTER} set is not
 * taken as a hit straight away: the walk first asks the filter. When the
 * filter says the ray passes, nothing is recorded and the closest distance
 * found so far is left alone, so whatever lies behind the triangle is still
 * found. Unflagged triangles never reach the filter, so a scene pays for the
 * hook only on the triangles that need it.
 *
 * - On the GPU the kernel supplies `bvhTriangleFilter` as its own piece of WGSL
 *   (see {@link NO_TRIANGLE_FILTER_WGSL} for the signature); the walk in
 *   `wgsl.ts` calls it.
 * - On the CPU, `createCpuTracer(scene, { triangleFilter })` takes a
 *   {@link TriangleFilter} with the same arguments.
 * - The brute-force reference (`buildReferenceScene(..., { triangleFilter })`)
 *   asks the same filter about *every* triangle, since it has no flags. A
 *   walk that agrees with it therefore also shows the flags cover every
 *   triangle the filter can let through.
 *
 * Without a filter, flagged triangles block like any other.
 *
 * Flags are set per original triangle: at build time through
 * `MeshGeometry.triangleFlags`, or afterwards over tree-ordered records with
 * {@link writeTriangleFilterFlags} (no rebuild; the GPU equivalent writes the
 * same bit of word `TRIANGLE_WORD.flags`).
 */

import { TRIANGLE_FLAG_FILTER, TRIANGLE_WORD, TRIANGLE_WORDS } from './layout.ts';

/** What the filter is told about a candidate hit. */
export interface TriangleFilterHit {
    /** Distance along the ray, in world units. */
    readonly t: number;
    /** Barycentric weights of corners 1 and 2. */
    readonly u: number;
    readonly v: number;
    /** The authored front side (object-space winding), also for a mirrored instance. */
    readonly frontFace: boolean;
}

/**
 * Decides one candidate hit on a flagged triangle: true when the ray should
 * pass through it. `triangleIndex` is the original (authored) triangle index.
 */
export type TriangleFilter = (instanceIndex: number, triangleIndex: number, hit: TriangleFilterHit) => boolean;

/**
 * `bvhTriangleFilter` for a kernel without a filter: every triangle blocks,
 * flagged or not. A kernel with a filter defines the same function itself:
 * `fn bvhTriangleFilter(instance: u32, triangle: u32, t: f32, barycentrics: vec2f, frontFace: bool) -> bool`,
 * returning true when the ray passes.
 */
export const NO_TRIANGLE_FILTER_WGSL = /* wgsl */ `
// ---- No triangle filter: every triangle blocks (trace/triangle-filter.ts) ----
fn bvhTriangleFilter(instance: u32, triangle: u32, t: f32, barycentrics: vec2f, frontFace: bool) -> bool {
    return false;
}
`;

/** A run of tree-ordered triangle records, in triangles. */
export interface TriangleRecordRange {
    readonly triangleOffset: number;
    readonly triangleCount: number;
}

/**
 * Sets or clears {@link TRIANGLE_FLAG_FILTER} on tree-ordered triangle records
 * in place, from each record's original triangle index. Other flag bits are
 * kept. `triangles` is a mesh tree's buffer (`MeshBvh.triangles`) or the
 * assembled scene's shared one (`TraceSceneData.triangles`, with one mesh's
 * `range`); without `range` every record is rewritten. Returns how many
 * records are flagged.
 */
export function writeTriangleFilterFlags(
    triangles: ArrayBuffer | Uint32Array,
    filtered: (originalIndex: number) => boolean,
    range?: TriangleRecordRange,
): number {
    const words = triangles instanceof Uint32Array ? triangles : new Uint32Array(triangles);
    const first = range?.triangleOffset ?? 0;
    const end = range ? first + range.triangleCount : words.length / TRIANGLE_WORDS;
    let flagged = 0;
    for (let triangle = first; triangle < end; triangle++) {
        const word = triangle * TRIANGLE_WORDS + TRIANGLE_WORD.flags;
        const on = filtered(words[triangle * TRIANGLE_WORDS + TRIANGLE_WORD.originalIndex]);
        words[word] = ((words[word] & ~TRIANGLE_FLAG_FILTER) | (on ? TRIANGLE_FLAG_FILTER : 0)) >>> 0;
        if (on) flagged++;
    }
    return flagged;
}

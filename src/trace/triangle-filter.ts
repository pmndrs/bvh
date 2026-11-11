/**
 * The walk's per-triangle filter hook, in the spirit of a DXR any-hit shader.
 *
 * A triangle whose `flags` word has {@link TRIANGLE_FLAG_FILTER} set is not
 * taken as a hit straight away: the walk first asks the filter whether the
 * ray passes through it. When it does, nothing is recorded and the closest
 * distance found so far is left alone, so whatever lies behind the triangle
 * is still found. Unflagged triangles never reach the filter, so a scene pays
 * for the hook only on the triangles that need it.
 *
 * - On the GPU the filter is `fn bvhTrianglePasses(candidate: BvhCandidate) -> bool`
 *   (see {@link DEFAULT_TRIANGLE_FILTER_WGSL}). `traceModuleWgsl` and
 *   `batchKernelWgsl` take it as their `triangleFilter` option; the walk in
 *   `wgsl.ts` calls it.
 * - On the CPU, `createCpuTracer(scene, { triangleFilter })` takes a
 *   {@link TrianglePasses} with the same fields.
 * - The brute-force reference (`buildReferenceScene(..., { triangleFilter })`)
 *   asks the same filter about *every* triangle it really hits, since it has
 *   no flags. A walk that agrees with it therefore also shows the flags cover
 *   every triangle the filter can let through.
 *
 * Without a filter, flagged triangles block like any other.
 *
 * The candidate is a struct so fields can be added later without breaking
 * filters. `query` says which trace asks (closest or any hit), so one filter
 * can, say, let shadow rays through a surface that closest-hit rays still
 * see. `rayMask` is the trace's mask, or all ones when the trace has none or
 * masks are compiled out.
 *
 * Flags are set per original triangle: at build time through
 * `MeshGeometry.triangleFlags` (bit 0 only; the other bits are reserved and
 * throw), or afterwards over tree-ordered records with
 * {@link writeTriangleFilterFlags} (no rebuild; the GPU equivalent writes the
 * same bit of word `TRIANGLE_WORD.flags`). A merged static mesh is flagged on
 * its group's range, by source instance.
 */

import {
    INDEXED_FLAG_FILTER,
    INDEXED_ID_MASK,
    INDEXED_TRIANGLE_WORDS,
    MERGED_SIDE_SOURCE_BITS,
    MESH_HEADER_WORD,
    TRIANGLE_FLAG_FILTER,
    TRIANGLE_WORD,
    TRIANGLE_WORDS,
    type BvhQuery,
    type TriangleLeaves,
} from './layout.ts';
import { isIndexedSceneTriangles } from './indexed-scene-buffers.ts';

/** A candidate hit on a flagged triangle, as the filter sees it. The same fields as the WGSL `BvhCandidate`. */
export interface TriangleCandidate {
    /** The platform instance index (`BvhHit.instance`). */
    readonly instance: number;
    /** The original (authored) triangle index within the instance's mesh. */
    readonly triangle: number;
    /** Distance along the ray, in world units. */
    readonly t: number;
    /** Barycentric weights of corners 1 and 2. */
    readonly barycentrics: readonly [number, number];
    /** The authored front side (object-space winding), also for a mirrored instance. */
    readonly frontFace: boolean;
    /** `BVH_QUERY_CLOSEST` (0) or `BVH_QUERY_ANY` (1). */
    readonly query: BvhQuery;
    /** The trace's ray mask; all ones (`RAY_MASK_ALL`) when it has none or masks are off. */
    readonly rayMask: number;
}

/** Decides one candidate hit on a flagged triangle: true when the ray passes through it. */
export type TrianglePasses = (candidate: TriangleCandidate) => boolean;

/**
 * @deprecated The old name of {@link TrianglePasses}. Its arguments changed:
 * it now takes one {@link TriangleCandidate} instead of `(instanceIndex, triangleIndex, hit)`.
 */
export type TriangleFilter = TrianglePasses;

/** The WGSL signature a triangle filter must declare. */
export const TRIANGLE_FILTER_SIGNATURE_WGSL = 'fn bvhTrianglePasses(candidate: BvhCandidate) -> bool';

/**
 * The default triangle filter: every triangle blocks, flagged or not. A kernel
 * with a filter of its own declares the same function
 * (`fn bvhTrianglePasses(candidate: BvhCandidate) -> bool`), returning true
 * when the ray passes.
 */
export const DEFAULT_TRIANGLE_FILTER_WGSL = /* wgsl */ `
// ---- Default triangle filter: every triangle blocks (trace/triangle-filter.ts) ----
${TRIANGLE_FILTER_SIGNATURE_WGSL} {
    return false;
}
`;

/** @deprecated The old name of {@link DEFAULT_TRIANGLE_FILTER_WGSL}; it now declares `bvhTrianglePasses`. */
export const NO_TRIANGLE_FILTER_WGSL = DEFAULT_TRIANGLE_FILTER_WGSL;

/** A run of tree-ordered triangle records, in triangles. `TraceSceneData.meshes` and `TraceSceneData.mergedGroups` entries are one. */
export interface TriangleRecordRange {
    readonly triangleOffset: number;
    readonly triangleCount: number;
    /** The range is a merged static group's (`TraceSceneData.mergedGroups`): its records name their source instance. */
    readonly merged?: boolean;
    /** The range is an input mesh merged into `TraceSceneData.mergedGroups[mergedInto]`; it holds no records. */
    readonly mergedInto?: number;
}

/**
 * Sets or clears {@link TRIANGLE_FLAG_FILTER} on tree-ordered triangle records
 * in place, from each record's original triangle index. Other flag bits are
 * kept. Returns how many records are flagged.
 *
 * On a merged static group's range (an entry of `TraceSceneData.mergedGroups`,
 * `merged: true`) the original index is the triangle's index in its *source*
 * mesh, and `filtered` also gets the record's source instance index
 * (`TraceSceneInstance.instanceIndex`; `mergedGroups[g].sourceMeshes` maps it
 * to the mesh). Elsewhere the second argument is `undefined`. A mesh that was
 * merged has an empty `TraceSceneData.meshes` entry with `mergedInto`; passing
 * it throws, since it would flag nothing: flag its group's range instead.
 *
 * `triangles` is a mesh tree (`MeshBvh`) or its buffer, or the assembled
 * scene (`TraceSceneData`) or its shared buffer, with one mesh's entry of
 * `TraceSceneData.meshes` as `range`; without `range` every record is
 * rewritten. A range outside the buffer throws a `RangeError`.
 *
 * Pass the tree or the scene itself and the format comes from its
 * `triangleLeaves`. A bare buffer is taken as records unless `triangleLeaves`
 * says `'indexed'`; an indexed scene's buffer passed bare without it throws,
 * since its 16-byte units read as 48-byte records would overwrite vertex
 * indices. Without `range`, an indexed buffer must be a lone tree's: an
 * indexed scene's holds headers and merged side tables between the meshes, so
 * it throws.
 *
 * `assembleTraceScene` copies each mesh's records into the scene's buffer, so
 * flags written to a `MeshBvh` after assembly do not reach the scene: write
 * them before assembling, or write the scene's buffer with the mesh's range
 * (and upload it again).
 */
/** A tree or a scene whose triangles `writeTriangleFilterFlags` writes, in its own format. */
export interface TriangleFilterTarget {
    readonly triangles: ArrayBuffer;
    readonly triangleLeaves?: TriangleLeaves;
    /** Present on a scene (`TraceSceneData`). */
    readonly meshes?: readonly unknown[];
}

export function writeTriangleFilterFlags(
    target: ArrayBuffer | Uint32Array | TriangleFilterTarget,
    filtered: (originalIndex: number, sourceInstance: number | undefined) => boolean,
    range?: TriangleRecordRange,
    leaves?: TriangleLeaves,
): number {
    const owner = target instanceof ArrayBuffer || target instanceof Uint32Array ? null : target;
    const triangles = owner ? owner.triangles : (target as ArrayBuffer | Uint32Array);
    const buffer = triangles instanceof Uint32Array ? triangles.buffer : triangles;
    const indexedScene = isIndexedSceneTriangles(buffer) || (owner?.triangleLeaves === 'indexed' && owner.meshes !== undefined);
    const ownLeaves = owner ? (owner.triangleLeaves ?? 'records') : indexedScene ? 'indexed' : undefined;
    if (leaves !== undefined && ownLeaves !== undefined && leaves !== ownLeaves) {
        throw new RangeError(`writeTriangleFilterFlags: the triangles are ${ownLeaves}, not ${leaves}`);
    }
    if (!owner && indexedScene && leaves === undefined) {
        throw new RangeError("writeTriangleFilterFlags: this is an indexed scene's buffer; pass the scene itself, or triangleLeaves: 'indexed'");
    }
    const triangleLeaves = leaves ?? ownLeaves ?? 'records';
    if (triangleLeaves === 'indexed' && !range && indexedScene) {
        throw new RangeError(
            "writeTriangleFilterFlags: an indexed scene's buffer needs a range (an entry of TraceSceneData.meshes or mergedGroups): " +
                'between its meshes it holds headers and side tables, not triangles',
        );
    }
    if (triangleLeaves === 'indexed') return writeIndexedFilterFlags(triangles, filtered, range);
    if (range?.mergedInto !== undefined && range.triangleCount === 0) {
        throw new RangeError(
            `writeTriangleFilterFlags: this mesh was merged into static group ${range.mergedInto} and has no records of its own; ` +
                `pass TraceSceneData.mergedGroups[${range.mergedInto}] as the range and decide by the callback's second argument, the source instance`,
        );
    }
    const words = triangles instanceof Uint32Array ? triangles : new Uint32Array(triangles);
    const merged = range?.merged === true;
    const recordCount = Math.floor(words.length / TRIANGLE_WORDS);
    const first = range?.triangleOffset ?? 0;
    const count = range ? range.triangleCount : recordCount;
    if (!Number.isInteger(first) || !Number.isInteger(count) || first < 0 || count < 0 || first + count > recordCount) {
        throw new RangeError(`writeTriangleFilterFlags: records ${first}..${first + count} are outside the buffer's ${recordCount} records`);
    }
    let flagged = 0;
    for (let triangle = first; triangle < first + count; triangle++) {
        const word = triangle * TRIANGLE_WORDS + TRIANGLE_WORD.flags;
        const base = triangle * TRIANGLE_WORDS;
        const on = filtered(words[base + TRIANGLE_WORD.originalIndex], merged ? words[base + TRIANGLE_WORD.padding] : undefined);
        words[word] = ((words[word] & ~TRIANGLE_FLAG_FILTER) | (on ? TRIANGLE_FLAG_FILTER : 0)) >>> 0;
        if (on) flagged++;
    }
    return flagged;
}

/**
 * {@link writeTriangleFilterFlags} on an indexed scene's (or tree's)
 * triangles: the filter flag is bit 28 of word 3, and on a merged group's
 * range the source instance comes from the group's side and source tables.
 */
function writeIndexedFilterFlags(
    triangles: ArrayBuffer | Uint32Array,
    filtered: (originalIndex: number, sourceInstance: number | undefined) => boolean,
    range?: TriangleRecordRange,
): number {
    if (range?.mergedInto !== undefined && range.triangleCount === 0) {
        throw new RangeError(
            `writeTriangleFilterFlags: this mesh was merged into static group ${range.mergedInto} and has no triangles of its own; ` +
                `pass TraceSceneData.mergedGroups[${range.mergedInto}] as the range and decide by the callback's second argument, the source instance`,
        );
    }
    const words = triangles instanceof Uint32Array ? triangles : new Uint32Array(triangles);
    const merged = range?.merged === true;
    const unitCount = Math.floor(words.length / INDEXED_TRIANGLE_WORDS);
    const first = range?.triangleOffset ?? 0;
    const count = range ? range.triangleCount : unitCount;
    if (!Number.isInteger(first) || !Number.isInteger(count) || first < 0 || count < 0 || first + count > unitCount || (merged && first < 1)) {
        throw new RangeError(`writeTriangleFilterFlags: triangles ${first}..${first + count} are outside the buffer's ${unitCount}`);
    }
    const header = (first - 1) * INDEXED_TRIANGLE_WORDS;
    const sideStart = merged ? (first - words[header + MESH_HEADER_WORD.sideTableDistance]) * INDEXED_TRIANGLE_WORDS : 0;
    const sourceStart = merged ? (first - words[header + MESH_HEADER_WORD.sourceTableDistance]) * INDEXED_TRIANGLE_WORDS : 0;
    let flagged = 0;
    for (let triangle = first; triangle < first + count; triangle++) {
        const word = triangle * INDEXED_TRIANGLE_WORDS + 3;
        const source = merged ? words[sourceStart + (words[sideStart + triangle - first] & MERGED_SIDE_SOURCE_BITS)] : undefined;
        const on = filtered(words[word] & INDEXED_ID_MASK, source);
        words[word] = ((words[word] & ~INDEXED_FLAG_FILTER) | (on ? INDEXED_FLAG_FILTER : 0)) >>> 0;
        if (on) flagged++;
    }
    return flagged;
}

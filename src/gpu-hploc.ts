/**
 * GPU H-PLOC builder (stage B2 of docs/internals/design/bvh-framework.md, the target
 * builder): mesh trees built on the GPU in the tracer's layout, behind the
 * same contract as the LBVH (B1, gpu-lbvh.ts). It reuses B1's pieces: the
 * batches (mesh and tile tables, one dispatch for every mesh), triangle boxes,
 * Morton codes, the segmented radix sort, the scans, and the readback. H-PLOC
 * (Benthin et al., HPG 2024) climbs the same radix tree as the LBVH and
 * merges clusters with PLOC on the way up (gpu-hploc-wgsl.ts), which gives
 * trees close to a full SAH build's.
 *
 * **Steps**, all in one compute pass:
 *
 * 1. Triangle boxes, the centroid box and the mesh box (B1's kernel).
 * 2. Morton codes: B1's 30-bit cubic codes (`'cubic-30'`, the default, one
 *    sort word), or tinybvh's 63-bit codes (`'tinybvh-63'`, two sort words),
 *    which make the trees tinybvh's `BuildHPLOC` makes.
 * 3. The stable segmented radix sort, once per code word.
 * 4. The split metric between neighbours, then the clear of the handed-over words.
 * 5. The climb with the merges.
 * 6. Which made nodes stay (the SAH leaf collapse was decided as each node was
 *    made), a scan to number them and one to place their leaves' triangles,
 *    then the nodes, the triangles' tree positions, the triangles and each
 *    mesh's stats.
 *
 * **Quality** (CPU twin, hploc-cpu.ts, on Sponza): SAH cost 1.03 times
 * tinybvh's default build with the default codes and leaf collapse, against
 * 1.22 for the LBVH. With tinybvh's codes and one-triangle leaves the twin's
 * trees are exactly tinybvh's `BuildHPLOC`'s.
 *
 * **Limits.** A batch's largest buffer holds 64 bytes per triangle (the output
 * nodes); {@link planGpuHplocBatches} splits a scene to fit the device. A mesh
 * may have at most {@link HPLOC_MAX_MESH_TRIANGLES} triangles. A whole batch holds
 * about 252 bytes per triangle plus its inputs ({@link hplocBatchBytes}; 32 fewer
 * with `triangleLeaves: 'indexed'`, whose triangles take 16 bytes, not 48); the
 * planner's `maxBatchBytes` caps that, and {@link GpuHplocScratchPool} shares
 * one scratch across batches built one after another (bvh-builders.md, "Build
 * peak memory").
 *
 * **Checks.** As the LBVH: a mesh's tree is refused (`problem` set) on a bad
 * index, a handed-over word still unwritten after the bounded wait, a merge
 * that found no pair (NaN boxes), or a tree too deep; callers then use the
 * CPU builder (mesh-builder.ts).
 *
 * Needs the `subgroups` feature; without it the declared fallback is tinybvh
 * (B0). The merge's shape depends on the device's smallest subgroup size
 * (`device.adapterInfo.subgroupMinSize`), never on a fixed warp width.
 */

import { GPU_LAYOUT_NODE_BYTES, GPU_LAYOUT_TRIANGLE_BYTES, gpuLayoutTriangleBytes, type GpuLayoutMeshBvh, type GpuLayoutNodeFormat, type GpuLayoutTriangleLeaves } from './gpu-layout.ts';
import { quantizeTreeBuffers } from './gpu-quantize.ts';
import {
    allocatedBytes,
    batchTotals,
    budgetOf,
    bufferLimit,
    buildTables,
    checkGeometry,
    checkOptions,
    copyBatchGeometry,
    geometryTotals,
    MAP_MODE_READ,
    meshResult,
    placeMeshes,
    planMeshBatches,
    readTreeLayouts,
    summarizeBatchBuffers,
    tableWords,
    UNIFORM_STRIDE,
    uploadBatchGeometry,
    USAGE,
    type BatchBytes,
    type BatchPlanOptions,
    type BatchTotals,
    type MeshPlacement,
} from './gpu-lbvh.ts';
import type { GpuMeshGeometry, GpuTreeBuffers } from './mesh-input.ts';
import { LBVH_FIRST_ROUND_TRIANGLES, LBVH_PARAMS_BYTES, LBVH_STATS_BYTES, LBVH_STATS_WORDS, LBVH_WORKGROUP_SIZE } from './gpu-lbvh-wgsl.ts';
import {
    HPLOC_CLIMB_WORDS,
    HPLOC_MAX_MESH_TRIANGLES,
    HPLOC_MORTON,
    HPLOC_PARAMS_BYTES,
    HPLOC_RECORD_WORDS,
    HPLOC_STAT,
    HPLOC_TIE_BREAK,
    hplocSlotsFor,
    hplocWgsl,
} from './gpu-hploc-wgsl.ts';
import type { HplocMortonCodes, HplocOptions } from './hploc-cpu.ts';
import { createRadixSort, createRadixSortPipelines, RADIX_BINS, radixPassCount, type RadixSort, type RadixSortPipelines } from './gpu-radix-sort.ts';
import { createScanBindGroups, createScanPipelines, encodeScan, SCAN_PARAMS_BYTES, scanParams, scanTileSumWords, type ScanBindGroups } from './gpu-scan.ts';
import { dispatchWorkgroups2d } from './gpu-workgroup-scan.ts';

export type { HplocMortonCodes, HplocOptions, HplocTieBreak } from './hploc-cpu.ts';
export { HPLOC_MAX_MESH_TRIANGLES } from './gpu-hploc-wgsl.ts';

/** The smallest subgroup WebGPU allows, assumed when the device doesn't say. */
const MIN_SUBGROUP_SIZE = 4;

/** The biggest per-triangle share of any one batch buffer: the output nodes, 64 bytes. */
const LARGEST_BYTES_PER_TRIANGLE = Math.max(GPU_LAYOUT_NODE_BYTES, GPU_LAYOUT_TRIANGLE_BYTES, (HPLOC_CLIMB_WORDS + HPLOC_RECORD_WORDS + 1) * 4, 7 * 4);

/** Bits of each code word the sorts sort. */
const SORT_BITS = { 'cubic-30': [30], 'tinybvh-63': [32, 31] } as const satisfies Record<HplocMortonCodes, readonly number[]>;

/** Options of {@link planGpuHplocBatches}. */
export interface HplocBatchPlanOptions extends BatchPlanOptions {
    /** The Morton codes the batches will use (`'tinybvh-63'` needs 8 more bytes per triangle). Default `'cubic-30'`. */
    readonly mortonCodes?: HplocMortonCodes;
}

/**
 * Splits meshes into batches whose buffers this device can bind, in order. A
 * mesh too big for any batch, or for H-PLOC, comes back in `tooBig`.
 *
 * `maxBatchBytes` (optional; default none, today's path) also caps each
 * batch's GPU bytes while it builds, its inputs, scratch and outputs
 * ({@link hplocBatchBytes}), so a scene builds in several smaller batches, one
 * after another, with a lower peak (docs/internals/design/bvh-builders.md, "Build peak
 * memory"). A mesh bigger than the cap on its own still builds, alone in its
 * batch.
 */
export function planGpuHplocBatches(device: GPUDevice, geometries: readonly GpuMeshGeometry[], options: HplocBatchPlanOptions = {}): { batches: number[][]; tooBig: number[] } {
    const mortonCodes = options.mortonCodes ?? 'cubic-30';
    return planMeshBatches(device, geometries, LARGEST_BYTES_PER_TRIANGLE, HPLOC_MAX_MESH_TRIANGLES, budgetOf(options, (totals) => hplocBatchBytes(totals, mortonCodes, options.triangleLeaves).total));
}

/** Where the climb's scratch and the links buffer put each part, in words (see the batch's constructor). */
function hplocLayout(totals: BatchTotals) {
    const { triangles, inner } = totals;
    // hplocScratch: [climb words per radix-tree node][a record per sorted position (made-node numbers are
    // representatives, gpu-hploc-wgsl.ts "Numbering")][a stored cluster per sorted position].
    const recordsOffset = inner * HPLOC_CLIMB_WORDS;
    const listsOffset = recordsOffset + triangles * HPLOC_RECORD_WORDS;
    // hplocLinks, all per sorted position: [the node scan, one more at the end][the leaf-run scan, one more]
    // [2 children per made node][parents: leaves, made nodes][the sorted position at each tree position]. The scans are bound at
    // offsets, which must be 256-byte aligned.
    const scanWords = triangles + 1;
    const leafRunsOffset = Math.ceil(scanWords / 64) * 64;
    const childrenOffset = leafRunsOffset + scanWords;
    const parentsOffset = childrenOffset + triangles * 2;
    const treeOrderOffset = parentsOffset + triangles * 2;
    // The scratch is cleared in whole vec4s.
    const scratchQuads = Math.ceil((listsOffset + triangles) / 4);
    return { recordsOffset, listsOffset, scanWords, leafRunsOffset, childrenOffset, parentsOffset, treeOrderOffset, scratchQuads };
}

/** Where each uniform record sits in the params buffer (256 bytes apart): LbvhParams, each sort's passes and scan, the two scans, HplocParams. */
function hplocParamRecords(mortonCodes: HplocMortonCodes) {
    // A sort's uniforms: one record per pass, then its scan's.
    const sortRecords = SORT_BITS[mortonCodes].map((bits) => radixPassCount(bits) + 1);
    const rankScanRecord = 1 + sortRecords.reduce((total, records) => total + records, 0);
    const leafRunScanRecord = rankScanRecord + 1;
    const hplocParamsRecord = leafRunScanRecord + 1;
    return { sortRecords, rankScanRecord, leafRunScanRecord, hplocParamsRecord };
}

/**
 * The H-PLOC batch's buffers, as {@link GpuHplocBatch} asks for them (before rounding up to 4 bytes, at least 16).
 * `parents: false` (a batch built without its parents output, {@link GpuHplocBatchOptions.parents}) leaves a 4-byte placeholder.
 */
export function hplocBatchBuffers(totals: BatchTotals, mortonCodes: HplocMortonCodes = 'cubic-30', triangleLeaves: GpuLayoutTriangleLeaves = 'records', parents = true) {
    const { meshes, triangles, inner } = totals;
    const layout = hplocLayout(totals);
    const sortOffsetWords = totals.sortTiles * RADIX_BINS;
    return {
        positions: totals.positionBytes,
        indices: Math.max(4, totals.indexBytes),
        flags: Math.max(1, totals.flagWords) * 4,
        // Every mesh's stats, then the word lbvhStart clears for the LBVH's climb queue (unused here).
        stats: meshes * LBVH_STATS_BYTES + 16,
        outNodes: inner * GPU_LAYOUT_NODE_BYTES,
        outTriangles: triangles * gpuLayoutTriangleBytes(triangleLeaves),
        outParents: parents ? inner * 4 : 4,
        tables: tableWords(totals) * 4,
        primitiveBoxes: triangles * 32,
        keysA: triangles * 4,
        valuesA: triangles * 4,
        keysB: triangles * 4,
        valuesB: triangles * 4,
        sortOffsets: sortOffsetWords * 4,
        tileSums: Math.max(scanTileSumWords(sortOffsetWords), scanTileSumWords(triangles + 1)) * 4,
        codes: mortonCodes === 'tinybvh-63' ? triangles * 8 : 16,
        deltas: triangles * 8,
        scratch: layout.scratchQuads * 16,
        links: (layout.treeOrderOffset + triangles) * 4,
        params: (hplocParamRecords(mortonCodes).hplocParamsRecord + 1) * UNIFORM_STRIDE,
    };
}

export type HplocBatchBuffer = keyof ReturnType<typeof hplocBatchBuffers>;
/** The buffers a scratch pool holds: every one but the outputs. */
export type HplocPoolBuffer = Exclude<HplocBatchBuffer, 'outNodes' | 'outTriangles' | 'outParents'>;

/** GPU bytes an H-PLOC batch of these totals holds while it builds, by buffer: its inputs, scratch and uniforms (`scratch`) and its trees (`outputs`). */
export function hplocBatchBytes(totals: BatchTotals, mortonCodes: HplocMortonCodes = 'cubic-30', triangleLeaves: GpuLayoutTriangleLeaves = 'records', parents = true): BatchBytes<HplocBatchBuffer> {
    return summarizeBatchBuffers(hplocBatchBuffers(totals, mortonCodes, triangleLeaves, parents));
}

/** Options of one H-PLOC batch ({@link GpuHplocBuilder.createBatch}): the build's, and whether it writes parents. */
export interface GpuHplocBatchOptions extends HplocOptions {
    /**
     * Write each node's parent into the trees' `parents` buffer (4 bytes per
     * node). Default true. False (P9 lever 7, "parents only when needed")
     * leaves out that output: `parents` is then a 16-byte placeholder, the
     * trees say so (`GpuTreeBuffers.withoutParents`), and `readLayouts` finds
     * each read-back tree's parents from its nodes on the CPU. The nodes and
     * triangles are the same either way.
     */
    readonly parents?: boolean;
}

const POOL_USAGE: Record<HplocPoolBuffer, number> = {
    positions: USAGE.STORAGE | USAGE.COPY_DST,
    indices: USAGE.STORAGE | USAGE.COPY_DST,
    flags: USAGE.STORAGE | USAGE.COPY_DST,
    stats: USAGE.STORAGE | USAGE.COPY_SRC,
    tables: USAGE.STORAGE | USAGE.COPY_DST,
    primitiveBoxes: USAGE.STORAGE | USAGE.COPY_DST,
    keysA: USAGE.STORAGE | USAGE.COPY_DST,
    valuesA: USAGE.STORAGE | USAGE.COPY_DST,
    keysB: USAGE.STORAGE | USAGE.COPY_DST,
    valuesB: USAGE.STORAGE | USAGE.COPY_DST,
    sortOffsets: USAGE.STORAGE | USAGE.COPY_DST,
    tileSums: USAGE.STORAGE | USAGE.COPY_DST,
    codes: USAGE.STORAGE | USAGE.COPY_DST,
    deltas: USAGE.STORAGE | USAGE.COPY_DST,
    scratch: USAGE.STORAGE | USAGE.COPY_DST,
    links: USAGE.STORAGE | USAGE.COPY_DST,
    params: USAGE.UNIFORM | USAGE.COPY_DST,
};
const POOL_BUFFERS = Object.keys(POOL_USAGE) as HplocPoolBuffer[];

/**
 * The bytes of each buffer of a scratch pool shared by these batches, built
 * one after another: each buffer at the largest size any batch needs (in
 * practice the largest batch's, since nearly every buffer grows with its
 * triangles). Outputs are not pooled: each batch hands its own to the caller.
 */
export function hplocScratchPoolBytes(batches: readonly (readonly GpuMeshGeometry[])[], mortonCodes: HplocMortonCodes = 'cubic-30'): Record<HplocPoolBuffer, number> {
    const sizes = Object.fromEntries(POOL_BUFFERS.map((name) => [name, 0])) as Record<HplocPoolBuffer, number>;
    for (const geometries of batches) {
        const { buffers } = hplocBatchBytes(geometryTotals(geometries), mortonCodes);
        for (const name of POOL_BUFFERS) sizes[name] = Math.max(sizes[name], buffers[name]);
    }
    return sizes;
}

/**
 * The inputs, scratch and uniforms of H-PLOC batches built one after another
 * ({@link GpuHplocBuilder.createScratchPool}), allocated once at the largest
 * batch's size, so a capped load (`maxBatchBytes`) doesn't allocate and free
 * them per batch. One batch at a time may use it: record, submit and `finish`
 * a batch before creating the next. Free it with `destroy` after the last.
 */
export class GpuHplocScratchPool {
    readonly mortonCodes: HplocMortonCodes;
    /** Each buffer's bytes. */
    readonly sizes: Readonly<Record<HplocPoolBuffer, number>>;
    private readonly buffers: Record<HplocPoolBuffer, GPUBuffer>;
    private destroyed = false;

    constructor(device: GPUDevice, sizes: Record<HplocPoolBuffer, number>, mortonCodes: HplocMortonCodes, label: string) {
        this.mortonCodes = mortonCodes;
        this.sizes = sizes;
        this.buffers = Object.fromEntries(
            POOL_BUFFERS.map((name) => [name, device.createBuffer({ label: `${label} pool ${name}`, size: sizes[name], usage: POOL_USAGE[name] })]),
        ) as Record<HplocPoolBuffer, GPUBuffer>;
    }

    /** GPU bytes the pool holds (0 once destroyed). */
    get bytes(): number {
        return this.destroyed ? 0 : POOL_BUFFERS.reduce((sum, name) => sum + this.sizes[name], 0);
    }

    get isDestroyed(): boolean {
        return this.destroyed;
    }

    /** The pool's buffer `name`, checked to hold `bytes`. */
    take(name: HplocPoolBuffer, bytes: number): GPUBuffer {
        if (this.destroyed) throw new Error('@pmndrs/bvh: the scratch pool was destroyed');
        if (bytes > this.sizes[name]) throw new RangeError(`the scratch pool's ${name} holds ${this.sizes[name]} bytes, and this batch needs ${bytes}; size the pool from every batch`);
        return this.buffers[name];
    }

    destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
        for (const name of POOL_BUFFERS) this.buffers[name].destroy();
    }
}

/** The steps of a batch's build, in order, as `GpuHplocBatch.encode` can time them. */
export const HPLOC_STAGES = ['boxes', 'morton', 'sort', 'deltas', 'clear', 'climb', 'mark', 'rank', 'emit', 'triangles'] as const;
export type HplocStage = (typeof HPLOC_STAGES)[number];

const KERNELS = [
    'lbvhStart',
    'lbvhBoxes',
    'lbvhMorton',
    'hplocMortonTinybvh',
    'hplocGatherHigh',
    'hplocDeltasKernel',
    'hplocClear',
    'hplocClimb',
    'hplocMark',
    'hplocEmit',
    'hplocTrianglesKernel',
    'hplocFinish',
] as const;
type HplocKernel = (typeof KERNELS)[number];

/** The bindings (gpu-lbvh-wgsl.ts and gpu-hploc-wgsl.ts) each kernel uses. Pipelines have 'auto' layouts, which hold exactly these. */
const BINDING = {
    positions: 0,
    indices: 1,
    flags: 2,
    primitiveBoxes: 3,
    keys: 4,
    values: 5,
    rank: 9,
    outNodes: 10,
    outTriangles: 11,
    outParents: 12,
    stats: 13,
    params: 14,
    tables: 15,
    statsView: 17,
    deltas: 22,
    codes: 23,
    scratch: 24,
    scratchView: 25,
    scratchFill: 26,
    links: 27,
    hplocParams: 28,
    valuesIn: 29,
} as const;
type BindingName = keyof typeof BINDING;

const KERNEL_BINDINGS: Record<HplocKernel, readonly BindingName[]> = {
    lbvhStart: ['rank', 'stats', 'params'],
    lbvhBoxes: ['positions', 'indices', 'primitiveBoxes', 'stats', 'params', 'tables'],
    lbvhMorton: ['primitiveBoxes', 'keys', 'values', 'params', 'tables', 'statsView'],
    hplocMortonTinybvh: ['primitiveBoxes', 'keys', 'values', 'codes', 'params', 'tables', 'statsView'],
    hplocGatherHigh: ['keys', 'values', 'codes', 'valuesIn', 'params', 'tables'],
    hplocDeltasKernel: ['keys', 'values', 'codes', 'deltas', 'params', 'tables', 'hplocParams'],
    hplocClear: ['scratchFill', 'hplocParams'],
    hplocClimb: ['primitiveBoxes', 'values', 'deltas', 'scratch', 'links', 'stats', 'params', 'tables', 'hplocParams'],
    hplocMark: ['scratchView', 'links', 'statsView', 'params', 'tables', 'hplocParams'],
    hplocEmit: ['primitiveBoxes', 'values', 'scratchView', 'links', 'outNodes', 'outParents', 'statsView', 'params', 'tables', 'hplocParams'],
    hplocTrianglesKernel: ['positions', 'indices', 'flags', 'values', 'links', 'outTriangles', 'params', 'tables', 'hplocParams'],
    hplocFinish: ['primitiveBoxes', 'values', 'scratchView', 'links', 'stats', 'params', 'tables', 'hplocParams'],
};

/** Kernels that read the sorted keys and values, which end in buffers a or b depending on the codes. */
const READS_SORTED = new Set<HplocKernel>(['hplocDeltasKernel', 'hplocClimb', 'hplocEmit', 'hplocTrianglesKernel', 'hplocFinish']);

/** The device's smallest subgroup, as it reports it (Chrome: `device.adapterInfo.subgroupMinSize`). */
function minSubgroupSize(device: GPUDevice): number {
    const info = (device as GPUDevice & { adapterInfo?: { subgroupMinSize?: number } }).adapterInfo;
    const size = info?.subgroupMinSize;
    return typeof size === 'number' && size >= MIN_SUBGROUP_SIZE ? size : MIN_SUBGROUP_SIZE;
}

function checkHplocOptions(options: HplocOptions): Required<HplocOptions> {
    const mortonCodes = options.mortonCodes ?? 'cubic-30';
    const tieBreak = options.tieBreak ?? 'pairs';
    if (!(mortonCodes in HPLOC_MORTON)) throw new RangeError(`unknown Morton codes: ${String(mortonCodes)}`);
    if (!(tieBreak in HPLOC_TIE_BREAK)) throw new RangeError(`unknown tie break: ${String(tieBreak)}`);
    return { ...checkOptions(options), mortonCodes, tieBreak };
}

/** Compiles the builder's kernels once per device and builds any number of batches with them. */
export class GpuHplocBuilder {
    /** The builder needs subgroups; without them, callers use the declared fallback (tinybvh, B0). */
    static isSupported(device: GPUDevice): boolean {
        return device.features.has('subgroups');
    }

    /** Cluster slots per lane in the merge on this device (1 when subgroups are at least 32 wide). */
    readonly slots: number;
    private readonly device: GPUDevice;
    private readonly label: string;
    private pipelines: Record<HplocKernel, GPUComputePipeline> | null = null;
    private sortPipelines: RadixSortPipelines | null = null;

    constructor(device: GPUDevice, options: { label?: string } = {}) {
        if (!GpuHplocBuilder.isSupported(device)) throw new Error('@pmndrs/bvh: the GPU H-PLOC builder needs a device with the "subgroups" feature');
        this.device = device;
        this.label = options.label ?? 'gpu hploc';
        this.slots = hplocSlotsFor(minSubgroupSize(device));
    }

    /**
     * Compiles the kernels (asynchronously, so the main thread doesn't stall).
     * Every compile is asked for before the first `await`, so a caller that
     * starts other compiles in the same task (ddgi-core's `prepare`) has these
     * queued alongside, not behind them.
     */
    async prepare(): Promise<void> {
        if (this.pipelines) return;
        const device = this.device;
        const module = device.createShaderModule({ label: this.label, code: hplocWgsl(this.slots) });
        const scan = createScanPipelines(device, this.label);
        const [sort, ...kernels] = await Promise.all([
            createRadixSortPipelines(device, scan, this.label),
            ...KERNELS.map((entryPoint) =>
                device.createComputePipelineAsync({ label: `${this.label} ${entryPoint}`, layout: 'auto', compute: { module, entryPoint } }),
            ),
        ]);
        this.sortPipelines = sort;
        this.pipelines = Object.fromEntries(KERNELS.map((name, index) => [name, kernels[index]])) as Record<HplocKernel, GPUComputePipeline>;
    }

    get ready(): boolean {
        return this.pipelines !== null;
    }

    /**
     * Builds every mesh's tree in one batch and one submit. A refused mesh has `problem` set; the others are fine.
     * `timestampWrites` (optional) times the batch's one compute pass: resolve its query set after this resolves.
     */
    async buildMany(geometries: readonly GpuMeshGeometry[], options: GpuHplocBatchOptions = {}, timestampWrites?: GPUComputePassTimestampWrites): Promise<GpuTreeBuffers> {
        const batch = this.createBatch(geometries, options);
        try {
            const encoder = this.device.createCommandEncoder({ label: this.label });
            batch.encode(encoder, timestampWrites);
            this.device.queue.submit([encoder.finish()]);
            return await batch.finish();
        } finally {
            batch.destroy();
        }
    }

    /** Builds one mesh's tree. Throws if the build was refused. */
    async build(geometry: GpuMeshGeometry, options: HplocOptions = {}): Promise<GpuTreeBuffers> {
        const trees = await this.buildMany([geometry], options);
        const problem = trees.meshes[0].problem;
        if (problem) {
            trees.destroy();
            throw new Error(`@pmndrs/bvh: GPU H-PLOC build refused: ${problem}`);
        }
        return trees;
    }

    /**
     * Uploads the inputs and sets up every buffer and bind group of a batch, so
     * it can be recorded with `encode` (any number of times: each recording
     * rebuilds the same trees) and read with `finish`. Throws if the batch is
     * too big for the device (see {@link planGpuHplocBatches}).
     */
    createBatch(geometries: readonly GpuMeshGeometry[], options: GpuHplocBatchOptions = {}, pool?: GpuHplocScratchPool): GpuHplocBatch {
        if (!this.pipelines || !this.sortPipelines) throw new Error('@pmndrs/bvh: call prepare() before building');
        if (geometries.length === 0) throw new RangeError('a batch needs at least one mesh');
        const settings = checkHplocOptions(options);
        geometries.forEach(checkGeometry);
        const plan = planGpuHplocBatches(this.device, geometries);
        if (plan.batches.length !== 1 || plan.tooBig.length > 0) {
            throw new RangeError(
                `these meshes don't fit one batch on this device (buffers of at most ${bufferLimit(this.device)} bytes, ` +
                    `meshes of at most ${HPLOC_MAX_MESH_TRIANGLES} triangles); split them with planGpuHplocBatches`,
            );
        }
        if (pool && pool.mortonCodes !== settings.mortonCodes) throw new RangeError(`the scratch pool was made for '${pool.mortonCodes}' codes, not '${settings.mortonCodes}'`);
        return new GpuHplocBatch(this.device, this.label, this.pipelines, this.sortPipelines, geometries, settings, pool ?? null, options.parents ?? true);
    }

    /**
     * A scratch pool for these batches (each a list of meshes, as
     * {@link planGpuHplocBatches} splits them), to build them one after another
     * with `createBatch(batch, options, pool)`: their inputs, scratch and
     * uniforms are allocated once, at the largest batch's size
     * ({@link hplocScratchPoolBytes}). Free it with `destroy` after the last
     * batch's `finish`.
     */
    createScratchPool(batches: readonly (readonly GpuMeshGeometry[])[], options: Pick<HplocOptions, 'mortonCodes'> = {}): GpuHplocScratchPool {
        const mortonCodes = options.mortonCodes ?? 'cubic-30';
        return new GpuHplocScratchPool(this.device, hplocScratchPoolBytes(batches, mortonCodes), mortonCodes, this.label);
    }

    /** Copies every usable tree of a batch back into CPU memory (one readback). */
    readLayouts(trees: GpuTreeBuffers): Promise<(GpuLayoutMeshBvh | null)[]> {
        return readTreeLayouts(this.device, trees, this.label);
    }
}

type BatchBuffer = 'positions' | 'indices' | 'flags' | 'stats' | 'outNodes' | 'outTriangles' | 'outParents';

/** One batch of mesh builds: inputs, scratch and outputs, recorded into one compute pass. */
export class GpuHplocBatch {
    readonly meshCount: number;
    /** The triangle format the batch writes. */
    readonly triangleLeaves: GpuLayoutTriangleLeaves;
    /** The node format `finish` hands out (the batch builds `f32` nodes either way). */
    readonly nodeFormat: GpuLayoutNodeFormat;
    /** The batch writes each node's parent ({@link GpuHplocBatchOptions.parents}). */
    readonly parents: boolean;
    private readonly device: GPUDevice;
    private readonly label: string;
    private readonly mortonCodes: HplocMortonCodes;
    private readonly pipelines: Record<HplocKernel, GPUComputePipeline>;
    private readonly sortPipelines: RadixSortPipelines;
    private readonly meshes: MeshPlacement[];
    private readonly tileCount: number;
    /** Workgroups of the clear: one vec4 of the climb's scratch per thread. */
    private readonly clearWorkgroups: number;
    /** Buffers freed with the batch; the outputs are handed to `finish`'s caller instead. */
    private readonly owned: GPUBuffer[] = [];
    private readonly buffers: Record<BatchBuffer, GPUBuffer>;
    private readonly groups: Record<HplocKernel, GPUBindGroup>;
    /** One sort per code word, in order: the low word's first. */
    private readonly sorts: RadixSort[];
    private readonly rankScan: ScanBindGroups;
    private readonly leafRunScan: ScanBindGroups;
    private finished = false;
    private destroyed = false;

    constructor(
        device: GPUDevice,
        label: string,
        pipelines: Record<HplocKernel, GPUComputePipeline>,
        sortPipelines: RadixSortPipelines,
        geometries: readonly GpuMeshGeometry[],
        options: Required<HplocOptions>,
        pool: GpuHplocScratchPool | null = null,
        parents = true,
    ) {
        this.device = device;
        this.parents = parents;
        this.label = label;
        this.mortonCodes = options.mortonCodes;
        this.pipelines = pipelines;
        this.sortPipelines = sortPipelines;
        this.meshCount = geometries.length;
        this.meshes = placeMeshes(geometries);
        const last = this.meshes[this.meshes.length - 1];
        const triangleTotal = last.triangleStart + last.triangleCount;
        const innerTotal = last.innerStart + last.innerCount;
        const { tables, tilesOffset, tileCount, sortTilesOffset, sortTable } = buildTables(this.meshes);
        this.tileCount = tileCount;
        const tinybvhCodes = options.mortonCodes === 'tinybvh-63';

        // Sizes from hplocBatchBuffers, which the planner's byte cap and the scratch pool also use.
        const totals = batchTotals(this.meshes);
        const sizes = hplocBatchBuffers(totals, options.mortonCodes, options.triangleLeaves, parents);
        this.triangleLeaves = options.triangleLeaves;
        this.nodeFormat = options.nodeFormat;
        if (tables.byteLength !== sizes.tables || sortTable.offsetWords * 4 !== sizes.sortOffsets) throw new Error('hplocBatchBuffers is out of step with buildTables');
        const output = USAGE.STORAGE | USAGE.COPY_SRC | USAGE.COPY_DST;
        const make = (name: string, bytes: number, usage: number, owned = true): GPUBuffer => {
            const buffer = device.createBuffer({ label: `${label} ${name}`, size: allocatedBytes(bytes), usage });
            if (owned) this.owned.push(buffer);
            return buffer;
        };
        // Inputs, scratch and uniforms: the pool's buffers when there is one (a pooled batch owns none), else the batch's own.
        const take = (name: HplocPoolBuffer): GPUBuffer => (pool ? pool.take(name, allocatedBytes(sizes[name])) : make(name, sizes[name], POOL_USAGE[name]));
        this.buffers = {
            positions: take('positions'),
            indices: take('indices'),
            flags: take('flags'),
            stats: take('stats'),
            outNodes: make('nodes', sizes.outNodes, output, false),
            outTriangles: make('triangles', sizes.outTriangles, output, false),
            outParents: make('parents', sizes.outParents, output, false),
        };
        const { recordsOffset, listsOffset, scanWords, leafRunsOffset, childrenOffset, parentsOffset, treeOrderOffset, scratchQuads } = hplocLayout(totals);
        this.clearWorkgroups = Math.ceil(scratchQuads / LBVH_WORKGROUP_SIZE);
        const scratch = {
            tables: take('tables'),
            primitiveBoxes: take('primitiveBoxes'),
            keysA: take('keysA'),
            valuesA: take('valuesA'),
            keysB: take('keysB'),
            valuesB: take('valuesB'),
            sortOffsets: take('sortOffsets'),
            tileSums: take('tileSums'),
            codes: take('codes'),
            deltas: take('deltas'),
            scratch: take('scratch'),
            links: take('links'),
        };
        device.queue.writeBuffer(scratch.tables, 0, tables);
        uploadBatchGeometry(device.queue, this.meshes, this.buffers);

        // Uniform records, 256 bytes apart: LbvhParams, each sort's passes and scan, the rank scan, HplocParams.
        const sortBits = SORT_BITS[options.mortonCodes];
        const { sortRecords, rankScanRecord, leafRunScanRecord, hplocParamsRecord } = hplocParamRecords(options.mortonCodes);
        const params = take('params');
        const lbvhParams = new ArrayBuffer(LBVH_PARAMS_BYTES);
        // LbvhParams, field by field (gpu-lbvh-wgsl.ts).
        const words = new Uint32Array(lbvhParams);
        const floats = new Float32Array(lbvhParams);
        words[0] = this.meshCount; // meshCount
        words[1] = tileCount; // tileCount
        words[2] = tilesOffset; // tilesOffset
        words[3] = innerTotal; // innerTotal
        words[4] = options.maxLeafPrimitives; // maxLeaf
        floats[5] = options.traversalCost; // traversalCost
        floats[6] = options.intersectionCost; // intersectionCost
        words[7] = LBVH_FIRST_ROUND_TRIANGLES; // climbLimit (the LBVH's; unused here)
        words[8] = options.triangleLeaves === 'indexed' ? 1 : 0; // indexedLeaves
        // HplocParams, field by field (gpu-hploc-wgsl.ts).
        const hplocParams = new Uint32Array([
            triangleTotal, // triangleTotal
            innerTotal, // innerTotal
            HPLOC_MORTON[options.mortonCodes], // mortonCodes
            recordsOffset, // recordsOffset
            listsOffset, // listsOffset
            childrenOffset, // childrenOffset
            parentsOffset, // parentsOffset
            treeOrderOffset, // treeOrderOffset
            HPLOC_TIE_BREAK[options.tieBreak], // tieBreak
            leafRunsOffset, // leafRunsOffset
            scratchQuads, // scratchQuads
            parents ? 1 : 0, // writeParents
        ]);
        if (hplocParams.byteLength !== HPLOC_PARAMS_BYTES) throw new Error('HplocParams changed size');

        // The sorted keys and values end in a after an even number of passes, in b after an odd one.
        const sortBuffers = {
            keysA: scratch.keysA,
            valuesA: scratch.valuesA,
            keysB: scratch.keysB,
            valuesB: scratch.valuesB,
            offsets: scratch.sortOffsets,
            tileSums: scratch.tileSums,
            tables: scratch.tables,
            tilesOffset: sortTilesOffset,
            params,
            paramsStride: UNIFORM_STRIDE,
        };
        let record = 1;
        this.sorts = sortBits.map((bits, index) => {
            const sort = createRadixSort(device, sortPipelines, { ...sortBuffers, paramsOffset: record * UNIFORM_STRIDE }, sortTable, bits, `${label} sort ${index}`);
            record += sortRecords[index];
            return sort;
        });
        // 'tinybvh-63': the high-word sort starts from buffers a (hplocGatherHigh writes them from the first sort's b).
        if (tinybvhCodes && this.sorts[0].resultInA) throw new Error('the low-word sort must end in buffers b');
        const sortedInA = this.sorts[this.sorts.length - 1].resultInA;
        const sortedKeys = sortedInA ? scratch.keysA : scratch.keysB;
        const sortedValues = sortedInA ? scratch.valuesA : scratch.valuesB;

        const resources: Record<BindingName, GPUBindingResource> = {
            positions: { buffer: this.buffers.positions },
            indices: { buffer: this.buffers.indices },
            flags: { buffer: this.buffers.flags },
            primitiveBoxes: { buffer: scratch.primitiveBoxes },
            keys: { buffer: scratch.keysA },
            values: { buffer: scratch.valuesA },
            rank: { buffer: scratch.links },
            outNodes: { buffer: this.buffers.outNodes },
            outTriangles: { buffer: this.buffers.outTriangles },
            outParents: { buffer: this.buffers.outParents },
            stats: { buffer: this.buffers.stats },
            params: { buffer: params, offset: 0, size: LBVH_PARAMS_BYTES },
            tables: { buffer: scratch.tables },
            statsView: { buffer: this.buffers.stats },
            deltas: { buffer: scratch.deltas },
            codes: { buffer: scratch.codes },
            scratch: { buffer: scratch.scratch },
            scratchView: { buffer: scratch.scratch },
            scratchFill: { buffer: scratch.scratch },
            links: { buffer: scratch.links },
            hplocParams: { buffer: params, offset: hplocParamsRecord * UNIFORM_STRIDE, size: HPLOC_PARAMS_BYTES },
            valuesIn: { buffer: scratch.valuesB },
        };
        const resourceFor = (kernel: HplocKernel, name: BindingName): GPUBindingResource => {
            if (READS_SORTED.has(kernel) && name === 'keys') return { buffer: sortedKeys };
            if (READS_SORTED.has(kernel) && name === 'values') return { buffer: sortedValues };
            return resources[name];
        };
        this.groups = Object.fromEntries(
            KERNELS.map((kernel) => [
                kernel,
                device.createBindGroup({
                    label: `${label} ${kernel}`,
                    layout: pipelines[kernel].getBindGroupLayout(0),
                    entries: KERNEL_BINDINGS[kernel].map((name) => ({ binding: BINDING[name], resource: resourceFor(kernel, name) })),
                }),
            ]),
        ) as Record<HplocKernel, GPUBindGroup>;

        this.rankScan = createScanBindGroups(
            device,
            sortPipelines.scan,
            {
                data: { buffer: scratch.links, offset: 0, size: scanWords * 4 },
                tileSums: { buffer: scratch.tileSums },
                params: { buffer: params, offset: rankScanRecord * UNIFORM_STRIDE, size: SCAN_PARAMS_BYTES },
                count: scanWords,
            },
            `${label} rank scan`,
        );
        this.leafRunScan = createScanBindGroups(
            device,
            sortPipelines.scan,
            {
                data: { buffer: scratch.links, offset: leafRunsOffset * 4, size: scanWords * 4 },
                tileSums: { buffer: scratch.tileSums },
                params: { buffer: params, offset: leafRunScanRecord * UNIFORM_STRIDE, size: SCAN_PARAMS_BYTES },
                count: scanWords,
            },
            `${label} leaf-run scan`,
        );
        // Every uniform is written once, here: they never change, so a batch can be recorded again and again.
        device.queue.writeBuffer(params, 0, lbvhParams);
        for (const sort of this.sorts) sort.writeParams(device.queue);
        device.queue.writeBuffer(params, rankScanRecord * UNIFORM_STRIDE, scanParams(scanWords));
        device.queue.writeBuffer(params, leafRunScanRecord * UNIFORM_STRIDE, scanParams(scanWords));
        device.queue.writeBuffer(params, hplocParamsRecord * UNIFORM_STRIDE, hplocParams);
    }

    /**
     * Records every mesh's build into `encoder`: copies of any GPU-buffer
     * inputs, then one compute pass. `timestampWrites` (optional) times the pass.
     *
     * For a breakdown, `stageTimestamps` instead puts each step in a pass of
     * its own and times it with the writes it returns. The extra passes cost a
     * little time themselves, so use it to compare steps, not to time the build.
     */
    encode(encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites, stageTimestamps?: (stage: HplocStage) => GPUComputePassTimestampWrites): void {
        if (this.destroyed) throw new Error('@pmndrs/bvh: the batch was destroyed');
        copyBatchGeometry(encoder, this.meshes, this.buffers);
        const maxPerDimension = this.device.limits.maxComputeWorkgroupsPerDimension;
        const shared: GPUComputePassEncoder | null = stageTimestamps ? null : encoder.beginComputePass({ label: this.label, timestampWrites });
        const stage = (name: HplocStage, recordStage: (pass: GPUComputePassEncoder) => void): void => {
            if (shared) {
                recordStage(shared);
                return;
            }
            const pass = encoder.beginComputePass({ label: `${this.label} ${name}`, timestampWrites: stageTimestamps?.(name) });
            recordStage(pass);
            pass.end();
        };
        const run = (pass: GPUComputePassEncoder, kernel: HplocKernel, workgroups: number): void => {
            pass.setPipeline(this.pipelines[kernel]);
            pass.setBindGroup(0, this.groups[kernel]);
            dispatchWorkgroups2d(pass, workgroups, maxPerDimension);
        };
        const perMesh = Math.ceil(this.meshCount / LBVH_WORKGROUP_SIZE);
        const tinybvhCodes = this.mortonCodes === 'tinybvh-63';
        stage('boxes', (pass) => {
            run(pass, 'lbvhStart', perMesh);
            run(pass, 'lbvhBoxes', this.tileCount);
        });
        stage('morton', (pass) => run(pass, tinybvhCodes ? 'hplocMortonTinybvh' : 'lbvhMorton', this.tileCount));
        stage('sort', (pass) => {
            this.sorts[0].encode(pass);
            if (tinybvhCodes) {
                run(pass, 'hplocGatherHigh', this.tileCount);
                this.sorts[1].encode(pass);
            }
        });
        stage('deltas', (pass) => run(pass, 'hplocDeltasKernel', this.tileCount));
        stage('clear', (pass) => run(pass, 'hplocClear', this.clearWorkgroups));
        stage('climb', (pass) => run(pass, 'hplocClimb', this.tileCount));
        stage('mark', (pass) => run(pass, 'hplocMark', this.tileCount));
        stage('rank', (pass) => {
            encodeScan(pass, this.sortPipelines.scan, this.rankScan, maxPerDimension);
            encodeScan(pass, this.sortPipelines.scan, this.leafRunScan, maxPerDimension);
        });
        // The tree positions come from the emit pass, so the triangles go after it.
        stage('emit', (pass) => run(pass, 'hplocEmit', this.tileCount));
        stage('triangles', (pass) => {
            run(pass, 'hplocTrianglesKernel', this.tileCount);
            run(pass, 'hplocFinish', perMesh);
        });
        shared?.end();
    }

    /**
     * Reads every mesh's stats back (one copy and one map), checks them, and
     * returns the trees. Call after submitting an `encode`. Frees the batch's
     * scratch; the trees stay until the caller destroys them.
     */
    async finish(): Promise<GpuTreeBuffers> {
        if (this.finished || this.destroyed) throw new Error('@pmndrs/bvh: the batch was already finished');
        const device = this.device;
        const bytes = this.meshCount * LBVH_STATS_BYTES;
        const readback = device.createBuffer({ label: `${this.label} stats readback`, size: bytes, usage: USAGE.MAP_READ | USAGE.COPY_DST });
        try {
            const encoder = device.createCommandEncoder({ label: `${this.label} stats` });
            encoder.copyBufferToBuffer(this.buffers.stats, 0, readback, 0, bytes);
            device.queue.submit([encoder.finish()]);
            await readback.mapAsync(MAP_MODE_READ);
            const words = new Uint32Array(readback.getMappedRange().slice(0));
            // From here the outputs are the caller's; if the read above failed, destroy() frees them with the rest.
            this.finished = true;
            const { outNodes, outTriangles, outParents } = this.buffers;
            const trees: GpuTreeBuffers = {
                nodes: outNodes,
                triangles: outTriangles,
                triangleLeaves: this.triangleLeaves,
                parents: outParents,
                ...(this.parents ? {} : { withoutParents: true as const }),
                meshes: this.meshes.map((mesh, index) => {
                    const stats = words.subarray(index * LBVH_STATS_WORDS, (index + 1) * LBVH_STATS_WORDS);
                    return meshResult(mesh, index, stats, hplocProblems(stats));
                }),
                destroy() {
                    outNodes.destroy();
                    outTriangles.destroy();
                    outParents.destroy();
                },
            };
            // Built f32 (the climb and the leaf collapse need f32 boxes), then packed.
            return this.nodeFormat === 'quantized' ? quantizeTreeBuffers(device, trees, this.label) : trees;
        } finally {
            readback.destroy();
            this.destroy();
        }
    }

    /** GPU bytes the batch holds until `finish` or `destroy`: its inputs, scratch and uniforms (not the outputs; 0 on a scratch pool, which counts them). */
    get scratchBytes(): number {
        return this.destroyed ? 0 : this.owned.reduce((sum, buffer) => sum + buffer.size, 0);
    }

    /** GPU bytes of the output trees (nodes, triangles, parents), which `finish` hands to the caller. */
    get outputBytes(): number {
        const { outNodes, outTriangles, outParents } = this.buffers;
        return outNodes.size + outTriangles.size + outParents.size;
    }

    /** Frees the scratch, the inputs and the uniforms. Trees handed out by `finish` are not touched. */
    destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
        for (const buffer of this.owned) buffer.destroy();
        // The outputs are only handed out by finish(); before that they are the batch's.
        if (!this.finished) {
            this.buffers.outNodes.destroy();
            this.buffers.outTriangles.destroy();
            this.buffers.outParents.destroy();
        }
    }
}

/** What H-PLOC's own stats words say is wrong with a mesh's tree. */
function hplocProblems(stats: Uint32Array): string[] {
    const problems: string[] = [];
    if (stats[HPLOC_STAT.failedMerges] > 0) problems.push(`${stats[HPLOC_STAT.failedMerges]} merges found no mutual pair (NaN boxes?)`);
    if (stats[HPLOC_STAT.narrowSubgroups] > 0) problems.push(`${stats[HPLOC_STAT.narrowSubgroups]} subgroups were narrower than the device reported`);
    if (stats[HPLOC_STAT.tooDeep] > 0) problems.push('the tree is deeper than H-PLOC allows');
    return problems;
}

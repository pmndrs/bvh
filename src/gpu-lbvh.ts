/**
 * GPU LBVH builder (stage B1 of docs/internals/design/bvh-framework.md): a tree per
 * mesh, built on the GPU in the tracer's layout (gpu-layout.ts), so the
 * tracer and everything that traces (`bvhTraceClosest`, `bvhTraceAny`, DDGI)
 * read its trees exactly as they read tinybvh's (B0). It is the stepping
 * stone to H-PLOC (B2), which reuses the Morton codes, the sort and the
 * bottom-up pass.
 *
 * **Steps**, all in one compute pass (WGSL in gpu-lbvh-wgsl.ts):
 *
 * 1. Triangle boxes, and the box around their centroids (subgroup, then
 *    workgroup, then one global atomic per workgroup).
 * 2. 30-bit Morton codes over each mesh's own centroid box, 10 bits per axis,
 *    in cubic cells (see `lbvhMorton` for why not stretched per axis).
 * 3. A stable radix sort of (code, triangle) pairs (gpu-radix-sort.ts), so equal
 *    codes stay in triangle order; the climb's comparison also breaks ties by
 *    sorted position, the "index" tie-break HIPRT uses.
 * 4. Apetrei's bottom-up build: one thread per sorted triangle climbs, the
 *    second thread to reach a node finishes it. It computes each node's box,
 *    SAH cost and height on the way, and marks a node whose triangles are
 *    cheaper as one leaf (SAH leaf collapse, at most `maxLeafPrimitives`).
 *    The climb runs in two rounds (`LBVH_FIRST_ROUND_TRIANGLES` says why); the
 *    second is an indirect dispatch sized by the first.
 * 5. The inner nodes that stay are numbered by a scan (gpu-scan.ts); the
 *    triangles are written in sorted order, which is tree order; then the
 *    nodes are written in the tracer's layout with their parents.
 *
 * **Batches.** A scene has many meshes (Sponza 132, most of them small). Each
 * dispatch has a fixed cost (on D3D12 a barrier between dispatches), so a build
 * per mesh pays it per mesh: one per mesh came to 97 ms for Sponza, of which
 * the largest mesh took 2.3. So a batch builds every mesh with the same
 * dispatches: the meshes' data sit one after another in the batch's buffers,
 * a mesh table says where, a tile table gives each workgroup its mesh, and the
 * sort and the scans run over all meshes at once (each mesh a segment). The
 * output is one node buffer, one triangle buffer and one parent buffer for
 * the whole batch, each mesh at an offset, which is how the tracer's scene
 * buffers hold trees anyway.
 *
 * **Limits.** A batch's largest buffers hold 76 bytes per triangle (the climb's
 * scratch); {@link planGpuLbvhBatches} splits a scene into batches that fit
 * the device's binding size.
 *
 * **Checks.** A mesh's tree is refused (`problem` set) if a corner names a
 * vertex past the end, or if a handed-over word was still unwritten after the
 * climb's bounded wait (see gpu-lbvh-wgsl.ts). Callers then use the CPU
 * builder (mesh-builder.ts does that).
 *
 * Needs the `subgroups` feature ({@link GpuLbvhBuilder.isSupported}). Without
 * it, the declared fallback is tinybvh (B0) in a worker (DECISIONS 2026-11-04:
 * subgroups first, one fallback).
 */

import { quantizeTreeBuffers } from './gpu-quantize.ts';
import {
    GPU_LAYOUT_MAX_TRIANGLES,
    GPU_LAYOUT_NODE_BYTES,
    GPU_LAYOUT_TRIANGLE_BYTES,
    gpuLayoutTriangleBytes,
    type GpuLayoutNodeFormat,
    type GpuLayoutTriangleLeaves,
    type GpuLayoutMeshBvh,
} from './gpu-layout.ts';
import {
    align4,
    batchTotals,
    budgetOf,
    bufferLimit,
    buildTables,
    checkGeometry,
    checkOptions,
    checkTriangleLeading,
    copyBatchGeometry,
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
} from './gpu-batch.ts';
import {
    LBVH_FIRST_ROUND_TRIANGLES,
    LBVH_NODE_INFO_WORDS,
    LBVH_PARAMS_BYTES,
    LBVH_SCRATCH_NODE_WORDS,
    LBVH_STATE_WORDS,
    LBVH_STATS_BYTES,
    LBVH_STATS_WORDS,
    LBVH_WORKGROUP_SIZE,
    lbvhWgsl,
} from './gpu-lbvh-wgsl.ts';
import { createRadixSort, createRadixSortPipelines, RADIX_BINS, radixPassCount, type RadixSort, type RadixSortPipelines } from './gpu-radix-sort.ts';
import { createScanBindGroups, createScanPipelines, encodeScan, SCAN_PARAMS_BYTES, scanParams, scanTileSumWords, type ScanBindGroups } from './gpu-scan.ts';
import { dispatchWorkgroups2d } from './gpu-workgroup-scan.ts';
import type { GpuBuildOptions, GpuMeshGeometry, GpuTreeBuffers } from './mesh-input.ts';

/** Bits of the Morton codes: 10 per axis. */
export const LBVH_MORTON_BITS = 30;

/** The biggest per-triangle (or per-inner-node) share of any one batch buffer: the climb's scratch, 76 bytes. */
const LARGEST_BYTES_PER_TRIANGLE = Math.max((LBVH_SCRATCH_NODE_WORDS + LBVH_STATE_WORDS) * 4, GPU_LAYOUT_NODE_BYTES, GPU_LAYOUT_TRIANGLE_BYTES);

/** The steps of a batch's build, in order, as `GpuLbvhBatch.encode` can time them. */
export const LBVH_STAGES = ['boxes', 'morton', 'sort', 'clear', 'climb', 'mark', 'rank', 'triangles', 'emit'] as const;
export type LbvhStage = (typeof LBVH_STAGES)[number];

const KERNELS = [
    'lbvhStart',
    'lbvhBoxes',
    'lbvhMorton',
    'lbvhClear',
    'lbvhClimb',
    'lbvhPlanSecondRound',
    'lbvhClimbOn',
    'lbvhMark',
    'lbvhEmit',
    'lbvhFinish',
    'lbvhTriangles',
    'lbvhLeafBoxes',
] as const;
type LbvhKernel = (typeof KERNELS)[number];

/** The bindings (gpu-lbvh-wgsl.ts) each kernel uses. Pipelines have 'auto' layouts, which hold exactly these. */
const BINDING = {
    positions: 0,
    indices: 1,
    flags: 2,
    primitiveBoxes: 3,
    keys: 4,
    values: 5,
    scratch: 6,
    nodeInfo: 8,
    rank: 9,
    outNodes: 10,
    outTriangles: 11,
    outParents: 12,
    stats: 13,
    params: 14,
    tables: 15,
    scratchView: 16,
    statsView: 17,
    scratchFill: 18,
    queue: 20,
    continueArgs: 21,
} as const;
type BindingName = keyof typeof BINDING;

const KERNEL_BINDINGS: Record<LbvhKernel, readonly BindingName[]> = {
    lbvhStart: ['rank', 'stats', 'params'],
    lbvhBoxes: ['positions', 'indices', 'primitiveBoxes', 'stats', 'params', 'tables'],
    lbvhMorton: ['primitiveBoxes', 'keys', 'values', 'params', 'tables', 'statsView'],
    lbvhClear: ['params', 'tables', 'scratchFill'],
    lbvhClimb: ['primitiveBoxes', 'keys', 'values', 'scratch', 'nodeInfo', 'stats', 'params', 'tables', 'queue'],
    lbvhPlanSecondRound: ['stats', 'params', 'queue', 'continueArgs'],
    lbvhClimbOn: ['primitiveBoxes', 'keys', 'values', 'scratch', 'nodeInfo', 'stats', 'params', 'tables', 'queue'],
    lbvhMark: ['nodeInfo', 'rank', 'params', 'tables'],
    lbvhEmit: ['nodeInfo', 'rank', 'outNodes', 'outTriangles', 'outParents', 'params', 'tables', 'scratchView', 'statsView'],
    lbvhFinish: ['primitiveBoxes', 'rank', 'stats', 'params', 'tables'],
    lbvhTriangles: ['positions', 'indices', 'flags', 'values', 'outTriangles', 'params', 'tables'],
    lbvhLeafBoxes: ['primitiveBoxes', 'keys', 'values', 'params', 'tables', 'scratchFill'],
};

/**
 * Splits meshes into batches whose buffers this device can bind, in order.
 * A mesh too big for any batch comes back in `tooBig`; callers build it on the CPU.
 *
 * `maxBatchBytes` (optional, default none) also caps each batch's own GPU
 * memory: its inputs, scratch and outputs ({@link lbvhBatchBytes}). A mesh
 * bigger than the cap on its own still builds, alone in its batch.
 */
export function planGpuLbvhBatches(device: GPUDevice, geometries: readonly GpuMeshGeometry[], options: BatchPlanOptions = {}): { batches: number[][]; tooBig: number[] } {
    return planMeshBatches(device, geometries, LARGEST_BYTES_PER_TRIANGLE, GPU_LAYOUT_MAX_TRIANGLES, budgetOf(options, (totals) => lbvhBatchBytes(totals, options.triangleLeaves).total));
}

/** Compiles the builder's kernels once per device and builds any number of batches with them. */
export class GpuLbvhBuilder {
    /** The builder needs subgroups; without them, callers use the declared fallback (tinybvh, B0). */
    static isSupported(device: GPUDevice): boolean {
        return device.features.has('subgroups');
    }

    private readonly device: GPUDevice;
    private readonly label: string;
    private pipelines: Record<LbvhKernel, GPUComputePipeline> | null = null;
    private sortPipelines: RadixSortPipelines | null = null;

    constructor(device: GPUDevice, options: { label?: string } = {}) {
        if (!GpuLbvhBuilder.isSupported(device)) throw new Error('@pmndrs/bvh: the GPU LBVH builder needs a device with the "subgroups" feature');
        this.device = device;
        this.label = options.label ?? 'gpu lbvh';
    }

    /** Compiles the kernels (asynchronously, so the main thread doesn't stall). */
    async prepare(): Promise<void> {
        if (this.pipelines) return;
        const device = this.device;
        const module = device.createShaderModule({ label: this.label, code: lbvhWgsl() });
        const scan = await createScanPipelines(device, this.label);
        const [sort, ...kernels] = await Promise.all([
            createRadixSortPipelines(device, scan, this.label),
            ...KERNELS.map((entryPoint) =>
                device.createComputePipelineAsync({ label: `${this.label} ${entryPoint}`, layout: 'auto', compute: { module, entryPoint } }),
            ),
        ]);
        this.sortPipelines = sort;
        this.pipelines = Object.fromEntries(KERNELS.map((name, index) => [name, kernels[index]])) as Record<LbvhKernel, GPUComputePipeline>;
    }

    get ready(): boolean {
        return this.pipelines !== null;
    }

    /**
     * Builds every mesh's tree in one batch and one submit. A refused mesh has `problem` set; the others are fine.
     * `timestampWrites` (optional) times the batch's one compute pass: resolve its query set after this resolves.
     */
    async buildMany(geometries: readonly GpuMeshGeometry[], options: GpuBuildOptions = {}, timestampWrites?: GPUComputePassTimestampWrites): Promise<GpuTreeBuffers> {
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
    async build(geometry: GpuMeshGeometry, options: GpuBuildOptions = {}): Promise<GpuTreeBuffers> {
        const trees = await this.buildMany([geometry], options);
        const problem = trees.meshes[0].problem;
        if (problem) {
            trees.destroy();
            throw new Error(`@pmndrs/bvh: GPU LBVH build refused: ${problem}`);
        }
        return trees;
    }

    /**
     * Uploads the inputs and sets up every buffer and bind group of a batch, so
     * it can be recorded with `encode` (any number of times: each recording
     * rebuilds the same trees) and read with `finish`. Throws if the batch is
     * too big for the device (see {@link planGpuLbvhBatches}).
     */
    createBatch(geometries: readonly GpuMeshGeometry[], options: GpuBuildOptions = {}): GpuLbvhBatch {
        if (!this.pipelines || !this.sortPipelines) throw new Error('@pmndrs/bvh: call prepare() before building');
        if (geometries.length === 0) throw new RangeError('a batch needs at least one mesh');
        const settings = checkOptions(options);
        checkTriangleLeading(settings.triangleLeading, geometries.length);
        geometries.forEach(checkGeometry);
        const plan = planGpuLbvhBatches(this.device, geometries);
        if (plan.batches.length !== 1 || plan.tooBig.length > 0) {
            throw new RangeError(`these meshes don't fit one batch on this device (buffers of at most ${bufferLimit(this.device)} bytes); split them with planGpuLbvhBatches`);
        }
        return new GpuLbvhBatch(this.device, this.label, this.pipelines, this.sortPipelines, geometries, settings);
    }

    /** Copies every usable tree of a batch back into CPU memory (one readback), each exactly as tinybvh's `buildLayout` returns one. */
    readLayouts(trees: GpuTreeBuffers): Promise<(GpuLayoutMeshBvh | null)[]> {
        return readTreeLayouts(this.device, trees, this.label);
    }
}

/** The LBVH batch's buffers, as {@link GpuLbvhBatch} asks for them (before {@link allocatedBytes}). */
export function lbvhBatchBuffers(totals: BatchTotals, triangleLeaves: GpuLayoutTriangleLeaves = 'records') {
    const { meshes, triangles, inner } = totals;
    const sortOffsetWords = totals.sortTiles * RADIX_BINS;
    const rankScanRecord = 1 + radixPassCount(LBVH_MORTON_BITS) + 1;
    return {
        positions: totals.positionBytes,
        indices: Math.max(4, totals.indexBytes),
        flags: Math.max(1, totals.flagWords) * 4,
        // Every mesh's stats, then the climb's queue count.
        stats: meshes * LBVH_STATS_BYTES + 16,
        outNodes: inner * GPU_LAYOUT_NODE_BYTES,
        outTriangles: totals.outTriangles * gpuLayoutTriangleBytes(triangleLeaves),
        outParents: inner * 4,
        continueArgs: 16,
        tables: tableWords(totals) * 4,
        primitiveBoxes: triangles * 32,
        keysA: triangles * 4,
        valuesA: triangles * 4,
        // After the sort, also the climb's queue: a count, then at most one arrival per inner node.
        keysB: (triangles + 1) * 4,
        valuesB: triangles * 4,
        sortOffsets: sortOffsetWords * 4,
        tileSums: Math.max(scanTileSumWords(sortOffsetWords), scanTileSumWords(inner + 1)) * 4,
        // Slots, then exchange words and heights (gpu-lbvh-wgsl.ts, lbvhScratch).
        scratch: inner * (LBVH_SCRATCH_NODE_WORDS + LBVH_STATE_WORDS) * 4,
        nodeInfo: inner * LBVH_NODE_INFO_WORDS * 4,
        rank: (inner + 1) * 4,
        params: (rankScanRecord + 1) * UNIFORM_STRIDE,
    };
}

/** GPU bytes an LBVH batch of these totals holds while it builds, by buffer. */
export function lbvhBatchBytes(totals: BatchTotals, triangleLeaves: GpuLayoutTriangleLeaves = 'records'): BatchBytes<keyof ReturnType<typeof lbvhBatchBuffers>> {
    return summarizeBatchBuffers(lbvhBatchBuffers(totals, triangleLeaves));
}

type BatchBuffer = 'positions' | 'indices' | 'flags' | 'stats' | 'outNodes' | 'outTriangles' | 'outParents';

/** One batch of mesh builds: inputs, scratch and outputs, recorded into one compute pass. */
export class GpuLbvhBatch {
    readonly meshCount: number;
    /** The triangle format the batch writes. */
    readonly triangleLeaves: GpuLayoutTriangleLeaves;
    /** The node format `finish` hands out (the batch builds `f32` nodes either way). */
    readonly nodeFormat: GpuLayoutNodeFormat;
    private readonly device: GPUDevice;
    private readonly label: string;
    private readonly pipelines: Record<LbvhKernel, GPUComputePipeline>;
    private readonly sortPipelines: RadixSortPipelines;
    private readonly meshes: MeshPlacement[];
    private readonly tileCount: number;
    /** Buffers freed with the batch; the outputs are handed to `finish`'s caller instead. */
    private readonly owned: GPUBuffer[] = [];
    private readonly buffers: Record<BatchBuffer, GPUBuffer>;
    /** The climb's second round is dispatched with the size the first round leaves here. */
    private readonly continueArgs: GPUBuffer;
    private readonly groups: Record<LbvhKernel, GPUBindGroup>;
    private readonly sort: RadixSort;
    private readonly rankScan: ScanBindGroups;
    private finished = false;
    private destroyed = false;

    constructor(
        device: GPUDevice,
        label: string,
        pipelines: Record<LbvhKernel, GPUComputePipeline>,
        sortPipelines: RadixSortPipelines,
        geometries: readonly GpuMeshGeometry[],
        options: Required<GpuBuildOptions>,
    ) {
        this.device = device;
        this.triangleLeaves = options.triangleLeaves;
        this.nodeFormat = options.nodeFormat;
        this.label = label;
        this.pipelines = pipelines;
        this.sortPipelines = sortPipelines;
        this.meshCount = geometries.length;
        this.meshes = placeMeshes(geometries, options.triangleLeading);
        const last = this.meshes[this.meshes.length - 1];
        const innerTotal = last.innerStart + last.innerCount;
        const { tables, tilesOffset, tileCount, sortTilesOffset, sortTable } = buildTables(this.meshes);
        this.tileCount = tileCount;

        const storage = USAGE.STORAGE | USAGE.COPY_DST;
        const output = USAGE.STORAGE | USAGE.COPY_SRC | USAGE.COPY_DST;
        const make = (name: string, bytes: number, usage: number, owned = true): GPUBuffer => {
            const buffer = device.createBuffer({ label: `${label} ${name}`, size: Math.max(16, align4(bytes)), usage });
            if (owned) this.owned.push(buffer);
            return buffer;
        };
        // Sizes from lbvhBatchBuffers, which the planner's byte cap also uses.
        const sizes = lbvhBatchBuffers(batchTotals(this.meshes), options.triangleLeaves);
        this.buffers = {
            positions: make('positions', sizes.positions, storage),
            indices: make('indices', sizes.indices, storage),
            flags: make('flags', sizes.flags, storage),
            stats: make('stats', sizes.stats, USAGE.STORAGE | USAGE.COPY_SRC),
            outNodes: make('nodes', sizes.outNodes, output, false),
            outTriangles: make('triangles', sizes.outTriangles, output, false),
            outParents: make('parents', sizes.outParents, output, false),
        };
        this.continueArgs = make('climb second round', sizes.continueArgs, USAGE.STORAGE | USAGE.INDIRECT);
        const scratch = {
            tables: make('tables', sizes.tables, storage),
            primitiveBoxes: make('primitive boxes', sizes.primitiveBoxes, storage),
            keysA: make('keys a', sizes.keysA, storage),
            valuesA: make('values a', sizes.valuesA, storage),
            keysB: make('keys b', sizes.keysB, storage),
            valuesB: make('values b', sizes.valuesB, storage),
            sortOffsets: make('sort offsets', sizes.sortOffsets, storage),
            tileSums: make('scan tile sums', sizes.tileSums, storage),
            scratch: make('inner-node scratch', sizes.scratch, storage),
            nodeInfo: make('inner-node info', sizes.nodeInfo, storage),
            rank: make('inner-node rank', sizes.rank, storage),
        };
        if (tables.byteLength !== sizes.tables || sortTable.offsetWords * 4 !== sizes.sortOffsets) throw new Error('lbvhBatchBuffers is out of step with buildTables');
        device.queue.writeBuffer(scratch.tables, 0, tables);
        uploadBatchGeometry(device.queue, this.meshes, this.buffers);

        const passes = radixPassCount(LBVH_MORTON_BITS);
        const rankScanRecord = 1 + passes + 1;
        const params = make('params', sizes.params, USAGE.UNIFORM | USAGE.COPY_DST);
        const paramsData = new ArrayBuffer(LBVH_PARAMS_BYTES);
        // LbvhParams, field by field (gpu-lbvh-wgsl.ts).
        const words = new Uint32Array(paramsData);
        const floats = new Float32Array(paramsData);
        words[0] = this.meshCount; // meshCount
        words[1] = tileCount; // tileCount
        words[2] = tilesOffset; // tilesOffset
        words[3] = innerTotal; // innerTotal
        words[4] = options.maxLeafPrimitives; // maxLeaf
        floats[5] = options.traversalCost; // traversalCost
        floats[6] = options.intersectionCost; // intersectionCost
        words[7] = LBVH_FIRST_ROUND_TRIANGLES; // climbLimit
        words[8] = options.triangleLeaves === 'indexed' ? 1 : 0; // indexedLeaves

        const resources: Record<BindingName, GPUBindingResource> = {
            positions: { buffer: this.buffers.positions },
            indices: { buffer: this.buffers.indices },
            flags: { buffer: this.buffers.flags },
            primitiveBoxes: { buffer: scratch.primitiveBoxes },
            keys: { buffer: scratch.keysA },
            values: { buffer: scratch.valuesA },
            scratch: { buffer: scratch.scratch },
            nodeInfo: { buffer: scratch.nodeInfo },
            rank: { buffer: scratch.rank },
            outNodes: { buffer: this.buffers.outNodes },
            outTriangles: { buffer: this.buffers.outTriangles },
            outParents: { buffer: this.buffers.outParents },
            stats: { buffer: this.buffers.stats },
            params: { buffer: params, offset: 0, size: LBVH_PARAMS_BYTES },
            tables: { buffer: scratch.tables },
            scratchView: { buffer: scratch.scratch },
            statsView: { buffer: this.buffers.stats },
            scratchFill: { buffer: scratch.scratch },
            queue: { buffer: scratch.keysB },
            continueArgs: { buffer: this.continueArgs },
        };
        this.groups = Object.fromEntries(
            KERNELS.map((kernel) => [
                kernel,
                device.createBindGroup({
                    label: `${label} ${kernel}`,
                    layout: pipelines[kernel].getBindGroupLayout(0),
                    entries: KERNEL_BINDINGS[kernel].map((name) => ({ binding: BINDING[name], resource: resources[name] })),
                }),
            ]),
        ) as Record<LbvhKernel, GPUBindGroup>;

        this.sort = createRadixSort(
            device,
            sortPipelines,
            {
                keysA: scratch.keysA,
                valuesA: scratch.valuesA,
                keysB: scratch.keysB,
                valuesB: scratch.valuesB,
                offsets: scratch.sortOffsets,
                tileSums: scratch.tileSums,
                tables: scratch.tables,
                tilesOffset: sortTilesOffset,
                params,
                paramsOffset: UNIFORM_STRIDE,
                paramsStride: UNIFORM_STRIDE,
            },
            sortTable,
            LBVH_MORTON_BITS,
            `${label} sort`,
        );
        // The climb reads the sorted codes and triangles from buffers a.
        if (!this.sort.resultInA) throw new Error(`a ${LBVH_MORTON_BITS}-bit sort must end in buffers a`);
        this.rankScan = createScanBindGroups(
            device,
            sortPipelines.scan,
            {
                data: { buffer: scratch.rank },
                tileSums: { buffer: scratch.tileSums },
                params: { buffer: params, offset: rankScanRecord * UNIFORM_STRIDE, size: SCAN_PARAMS_BYTES },
                count: innerTotal + 1,
            },
            `${label} rank scan`,
        );
        // Every uniform is written once, here: they never change, so a batch can be recorded again and again.
        device.queue.writeBuffer(params, 0, paramsData);
        this.sort.writeParams(device.queue);
        device.queue.writeBuffer(params, rankScanRecord * UNIFORM_STRIDE, scanParams(innerTotal + 1));
    }

    /**
     * Records every mesh's build into `encoder`: copies of any GPU-buffer
     * inputs, then one compute pass. `timestampWrites` (optional) times the pass.
     *
     * For a breakdown, `stageTimestamps` instead puts each step in a pass of
     * its own and times it with the writes it returns. The extra passes cost a
     * little time themselves, so use it to compare steps, not to time the build.
     */
    encode(encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites, stageTimestamps?: (stage: LbvhStage) => GPUComputePassTimestampWrites): void {
        if (this.destroyed) throw new Error('@pmndrs/bvh: the batch was destroyed');
        copyBatchGeometry(encoder, this.meshes, this.buffers);
        const maxPerDimension = this.device.limits.maxComputeWorkgroupsPerDimension;
        const shared: GPUComputePassEncoder | null = stageTimestamps ? null : encoder.beginComputePass({ label: this.label, timestampWrites });
        const stage = (name: LbvhStage, record: (pass: GPUComputePassEncoder) => void): void => {
            if (shared) {
                record(shared);
                return;
            }
            const pass = encoder.beginComputePass({ label: `${this.label} ${name}`, timestampWrites: stageTimestamps?.(name) });
            record(pass);
            pass.end();
        };
        const run = (pass: GPUComputePassEncoder, kernel: LbvhKernel, workgroups: number): void => {
            pass.setPipeline(this.pipelines[kernel]);
            pass.setBindGroup(0, this.groups[kernel]);
            dispatchWorkgroups2d(pass, workgroups, maxPerDimension);
        };
        const perMesh = Math.ceil(this.meshCount / LBVH_WORKGROUP_SIZE);
        stage('boxes', (pass) => {
            run(pass, 'lbvhStart', perMesh);
            run(pass, 'lbvhBoxes', this.tileCount);
        });
        stage('morton', (pass) => run(pass, 'lbvhMorton', this.tileCount));
        stage('sort', (pass) => this.sort.encode(pass));
        stage('clear', (pass) => run(pass, 'lbvhClear', this.tileCount));
        stage('climb', (pass) => {
            run(pass, 'lbvhClimb', this.tileCount);
            run(pass, 'lbvhPlanSecondRound', 1);
            pass.setPipeline(this.pipelines.lbvhClimbOn);
            pass.setBindGroup(0, this.groups.lbvhClimbOn);
            pass.dispatchWorkgroupsIndirect(this.continueArgs, 0);
        });
        stage('mark', (pass) => run(pass, 'lbvhMark', this.tileCount));
        stage('rank', (pass) => encodeScan(pass, this.sortPipelines.scan, this.rankScan, maxPerDimension));
        // The triangles go first: the emit pass reads its leaves' boxes from them (indexed: from the slots lbvhLeafBoxes fills).
        stage('triangles', (pass) => {
            run(pass, 'lbvhTriangles', this.tileCount);
            if (this.triangleLeaves === 'indexed') run(pass, 'lbvhLeafBoxes', this.tileCount);
        });
        stage('emit', (pass) => {
            run(pass, 'lbvhEmit', this.tileCount);
            run(pass, 'lbvhFinish', perMesh);
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
                meshes: this.meshes.map((mesh, index) => meshResult(mesh, index, words.subarray(index * LBVH_STATS_WORDS, (index + 1) * LBVH_STATS_WORDS))),
                destroy() {
                    outNodes.destroy();
                    outTriangles.destroy();
                    outParents.destroy();
                },
            };
            // Built f32 (the climb hands boxes over in the nodes' f32 words), then packed.
            return this.nodeFormat === 'quantized' ? quantizeTreeBuffers(device, trees, this.label) : trees;
        } finally {
            readback.destroy();
            this.destroy();
        }
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

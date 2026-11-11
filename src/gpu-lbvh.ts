/**
 * GPU LBVH builder (stage B1 of docs/design/bvh-framework.md): a tree per
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

import {
    GPU_LAYOUT_MAX_LEAF_PRIMITIVES,
    GPU_LAYOUT_MAX_TRIANGLES,
    GPU_LAYOUT_NODE_BYTES,
    GPU_LAYOUT_TRIANGLE_BYTES,
    type GpuLayoutMeshBvh,
} from './gpu-layout.ts';
import {
    LBVH_FIRST_ROUND_TRIANGLES,
    LBVH_INDEX_FORMAT,
    LBVH_MESH,
    LBVH_MESH_WORDS,
    LBVH_NODE_INFO_WORDS,
    LBVH_PARAMS_BYTES,
    LBVH_SCRATCH_NODE_WORDS,
    LBVH_STAT,
    LBVH_STATE_WORDS,
    LBVH_STATS_BYTES,
    LBVH_STATS_WORDS,
    LBVH_TILE_WORDS,
    LBVH_WORKGROUP_SIZE,
    lbvhWgsl,
} from './gpu-lbvh-wgsl.ts';
import { createRadixSort, createRadixSortPipelines, RADIX_BINS, radixPassCount, SORT_TILE_WORDS, sortTilesOf, sortTileTable, type RadixSort, type RadixSortPipelines } from './gpu-radix-sort.ts';
import { createScanBindGroups, createScanPipelines, encodeScan, SCAN_PARAMS_BYTES, scanParams, scanTileSumWords, type ScanBindGroups } from './gpu-scan.ts';
import { dispatchWorkgroups2d } from './gpu-workgroup-scan.ts';
import { checkTriangleFlags, isGpuMeshBuffers, meshTriangleCount, type GpuBuildOptions, type GpuMeshGeometry, type GpuMeshTreeRange, type GpuTreeBuffers } from './mesh-input.ts';

/** Bits of the Morton codes: 10 per axis. */
export const LBVH_MORTON_BITS = 30;

/** Buffer usages and the map mode; the globals don't exist in Node, where tests import this file. */
export const USAGE = { MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80, INDIRECT: 0x100 } as const;
export const MAP_MODE_READ = 0x1;

/** Uniform blocks sit 256 bytes apart: the default `minUniformBufferOffsetAlignment`. */
export const UNIFORM_STRIDE = 256;

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
};

export function checkOptions(options: GpuBuildOptions): Required<GpuBuildOptions> {
    const maxLeaf = options.maxLeafPrimitives ?? GPU_LAYOUT_MAX_LEAF_PRIMITIVES;
    if (!Number.isInteger(maxLeaf) || maxLeaf < 1 || maxLeaf > GPU_LAYOUT_MAX_LEAF_PRIMITIVES) {
        throw new RangeError(`maxLeafPrimitives must be an integer from 1 to ${GPU_LAYOUT_MAX_LEAF_PRIMITIVES}, got ${maxLeaf}`);
    }
    const traversalCost = options.traversalCost ?? 1;
    const intersectionCost = options.intersectionCost ?? 1;
    if (!(traversalCost > 0) || !(intersectionCost > 0)) throw new RangeError('the SAH costs must be positive');
    return { maxLeafPrimitives: maxLeaf, traversalCost, intersectionCost };
}

export function checkGeometry(geometry: GpuMeshGeometry, index: number): void {
    const where = `mesh ${index}`;
    const triangles = meshTriangleCount(geometry);
    if (isGpuMeshBuffers(geometry)) {
        if (!Number.isInteger(triangles) || triangles < 1) throw new RangeError(`${where}: needs at least one triangle`);
        if (!Number.isInteger(geometry.vertexCount) || geometry.vertexCount < 1) throw new RangeError(`${where}: needs at least one vertex`);
        const positionStart = (geometry.positionOffsetFloats ?? 0) * 4;
        if (positionStart + positionBytes(geometry) > geometry.positionBuffer.size) throw new RangeError(`${where}: the position buffer is too small for ${geometry.vertexCount} vertices`);
        if (geometry.indexBuffer) {
            const bytesPerIndex = geometry.indexFormat === 'uint16' ? 2 : 4;
            if (((geometry.indexOffset ?? 0) + triangles * 3) * bytesPerIndex > geometry.indexBuffer.size) throw new RangeError(`${where}: the index buffer is too small for ${triangles} triangles`);
        }
        if (geometry.triangleFlagsBuffer && triangles * 4 > geometry.triangleFlagsBuffer.size) throw new RangeError(`${where}: the flags buffer is too small for ${triangles} triangles`);
    } else {
        const { positions, triangleFlags } = geometry;
        if (positions.length % 3 !== 0 || positions.length === 0 || !Number.isInteger(triangles) || triangles < 1) {
            throw new RangeError(`${where}: needs at least one triangle, with three floats per vertex and three indices per triangle`);
        }
        if (triangleFlags && triangleFlags.length < triangles) throw new RangeError(`${where}: triangleFlags must have one entry per triangle`);
        checkTriangleFlags(triangleFlags, triangles, where);
    }
    if (triangles > GPU_LAYOUT_MAX_TRIANGLES) throw new RangeError(`${where}: too many triangles for one tree: ${triangles}`);
}

/** Bytes one buffer of this device may hold and bind. */
export function bufferLimit(device: GPUDevice): number {
    return Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
}

/** Bytes of vertex positions a geometry needs in a batch's position buffer. */
export function positionBytes(geometry: GpuMeshGeometry): number {
    if (!isGpuMeshBuffers(geometry)) return geometry.positions.byteLength;
    return ((geometry.vertexCount - 1) * (geometry.positionStrideFloats ?? 3) + 3) * 4;
}

/**
 * Splits meshes into batches whose buffers this device can bind, in order.
 * A mesh too big for any batch comes back in `tooBig`; callers build it on the CPU.
 *
 * `maxBatchBytes` (optional, default none) also caps each batch's own GPU
 * memory: its inputs, scratch and outputs ({@link lbvhBatchBytes}). A mesh
 * bigger than the cap on its own still builds, alone in its batch.
 */
export function planGpuLbvhBatches(device: GPUDevice, geometries: readonly GpuMeshGeometry[], options: BatchPlanOptions = {}): { batches: number[][]; tooBig: number[] } {
    return planMeshBatches(device, geometries, LARGEST_BYTES_PER_TRIANGLE, GPU_LAYOUT_MAX_TRIANGLES, budgetOf(options, (totals) => lbvhBatchBytes(totals).total));
}

/** Options of the batch planners. */
export interface BatchPlanOptions {
    /**
     * Each batch's GPU bytes (inputs, scratch and outputs) stay at or under
     * this, except a single mesh bigger than it, which gets a batch of its own.
     * Default: no cap (batches as big as the device's buffers allow).
     */
    readonly maxBatchBytes?: number;
}

/** A cap on a batch's bytes, as a function of its totals. */
export interface BatchBudget {
    readonly maxBatchBytes: number;
    readonly batchBytes: (totals: BatchTotals) => number;
}

/** The planner's budget for `options`, or undefined when there is no cap. Throws on a cap that isn't a positive number. */
export function budgetOf(options: BatchPlanOptions, batchBytes: (totals: BatchTotals) => number): BatchBudget | undefined {
    const cap = options.maxBatchBytes;
    if (cap === undefined) return undefined;
    if (!(cap > 0)) throw new RangeError(`maxBatchBytes must be a positive number of bytes, got ${cap}`);
    return { maxBatchBytes: cap, batchBytes };
}

/**
 * The planner of {@link planGpuLbvhBatches} for any GPU builder: its largest
 * buffer holds `bytesPerTriangle` per triangle, and a mesh of more than
 * `maxMeshTriangles` triangles is too big for it whatever the device. With a
 * `budget`, a batch also closes before a mesh that would take its bytes past
 * the cap (a mesh over the cap on its own still gets a batch, alone).
 */
export function planMeshBatches(
    device: GPUDevice,
    geometries: readonly GpuMeshGeometry[],
    bytesPerTriangle: number,
    maxMeshTriangles: number,
    budget?: BatchBudget,
): { batches: number[][]; tooBig: number[] } {
    const limit = bufferLimit(device);
    const batches: number[][] = [];
    const tooBig: number[] = [];
    let current: number[] = [];
    let triangles = 0;
    let positions = 0;
    let totals = NO_MESHES;
    geometries.forEach((geometry, index) => {
        const meshTriangles = Math.max(1, meshTriangleCount(geometry));
        const meshPositions = positionBytes(geometry);
        if (meshTriangles > maxMeshTriangles || meshTriangles * bytesPerTriangle > limit || meshPositions > limit) {
            tooBig.push(index);
            return;
        }
        const meshTotals = budget ? geometryTotals([geometry]) : NO_MESHES;
        const overLimit = (triangles + meshTriangles) * bytesPerTriangle > limit || positions + meshPositions > limit;
        const overBudget = budget !== undefined && current.length > 0 && budget.batchBytes(addBatchTotals(totals, meshTotals)) > budget.maxBatchBytes;
        if (overLimit || overBudget) {
            batches.push(current);
            current = [];
            triangles = 0;
            positions = 0;
            totals = NO_MESHES;
        }
        current.push(index);
        triangles += meshTriangles;
        positions += meshPositions;
        totals = addBatchTotals(totals, meshTotals);
    });
    if (current.length > 0) batches.push(current);
    return { batches, tooBig };
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

    /** Builds every mesh's tree in one batch and one submit. A refused mesh has `problem` set; the others are fine. */
    async buildMany(geometries: readonly GpuMeshGeometry[], options: GpuBuildOptions = {}): Promise<GpuTreeBuffers> {
        const batch = this.createBatch(geometries, options);
        try {
            const encoder = this.device.createCommandEncoder({ label: this.label });
            batch.encode(encoder);
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

/** Copies every usable tree in shared GPU buffers back into CPU memory, with one readback. Any GPU builder's trees. */
export async function readTreeLayouts(device: GPUDevice, trees: GpuTreeBuffers, label: string): Promise<(GpuLayoutMeshBvh | null)[]> {
    let nodeEnd = 0;
    let triangleEnd = 0;
    for (const mesh of trees.meshes) {
        nodeEnd = Math.max(nodeEnd, mesh.nodeOffset + mesh.nodeCount);
        triangleEnd = Math.max(triangleEnd, mesh.triangleOffset + mesh.triangleCount);
    }
    const nodeBytes = nodeEnd * GPU_LAYOUT_NODE_BYTES;
    const triangleBytes = triangleEnd * GPU_LAYOUT_TRIANGLE_BYTES;
    const parentBytes = nodeEnd * 4;
    const readback = device.createBuffer({ label: `${label} readback`, size: nodeBytes + triangleBytes + parentBytes, usage: USAGE.MAP_READ | USAGE.COPY_DST });
    try {
        const encoder = device.createCommandEncoder({ label: `${label} readback` });
        encoder.copyBufferToBuffer(trees.nodes, 0, readback, 0, nodeBytes);
        encoder.copyBufferToBuffer(trees.triangles, 0, readback, nodeBytes, triangleBytes);
        encoder.copyBufferToBuffer(trees.parents, 0, readback, nodeBytes + triangleBytes, parentBytes);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(MAP_MODE_READ);
        const bytes = readback.getMappedRange();
        return trees.meshes.map((mesh) => {
            if (mesh.problem) return null;
            const nodeStart = mesh.nodeOffset * GPU_LAYOUT_NODE_BYTES;
            const triangleStart = nodeBytes + mesh.triangleOffset * GPU_LAYOUT_TRIANGLE_BYTES;
            const parentStart = nodeBytes + triangleBytes + mesh.nodeOffset * 4;
            return {
                nodes: bytes.slice(nodeStart, nodeStart + mesh.nodeCount * GPU_LAYOUT_NODE_BYTES),
                triangles: bytes.slice(triangleStart, triangleStart + mesh.triangleCount * GPU_LAYOUT_TRIANGLE_BYTES),
                parents: new Uint32Array(bytes.slice(parentStart, parentStart + mesh.nodeCount * 4)),
                nodeCount: mesh.nodeCount,
                triangleCount: mesh.triangleCount,
                maxDepth: mesh.maxDepth,
                leafCount: mesh.leafCount,
                bounds: mesh.bounds.slice(),
            };
        });
    } finally {
        readback.destroy();
    }
}

/** Where one mesh sits in a batch's buffers. */
export interface MeshPlacement {
    readonly geometry: GpuMeshGeometry;
    readonly triangleStart: number;
    readonly triangleCount: number;
    readonly innerStart: number;
    readonly innerCount: number;
    readonly vertexCount: number;
    /** Bytes into the batch's positions where the copied or uploaded vertices start. */
    readonly positionByteStart: number;
    readonly positionStride: number;
    readonly indexFormat: number;
    /** Bytes into the batch's indices where the copied or uploaded indices start, and the mesh's first index after that. */
    readonly indexByteStart: number;
    readonly indexSkip: number;
    readonly flagsWordStart: number;
    readonly hasFlags: boolean;
}

/** Rounds up to a multiple of 4 bytes, where buffer copies and bindings need their offsets and sizes. */
export const align4 = (bytes: number): number => Math.ceil(bytes / 4) * 4;

/** Index bytes a mesh's block in a batch's indices holds. */
export function indexBytes(mesh: Pick<MeshPlacement, 'indexFormat' | 'indexSkip' | 'triangleCount'>): number {
    const bytesPerIndex = mesh.indexFormat === LBVH_INDEX_FORMAT.uint16 ? 2 : mesh.indexFormat === LBVH_INDEX_FORMAT.uint32 ? 4 : 0;
    return (mesh.indexSkip + mesh.triangleCount * 3) * bytesPerIndex;
}

/** Lays the meshes out one after another in every batch buffer. */
export function placeMeshes(geometries: readonly GpuMeshGeometry[]): MeshPlacement[] {
    let triangleStart = 0;
    let innerStart = 0;
    let positionByteStart = 0;
    let indexByteStart = 0;
    let flagsWordStart = 0;
    return geometries.map((geometry) => {
        const triangleCount = meshTriangleCount(geometry);
        let vertexCount: number;
        let positionStride = 3;
        let indexFormat: number = LBVH_INDEX_FORMAT.none;
        let indexSkip = 0;
        let hasFlags: boolean;
        if (isGpuMeshBuffers(geometry)) {
            vertexCount = geometry.vertexCount;
            positionStride = geometry.positionStrideFloats ?? 3;
            if (geometry.indexBuffer) {
                indexFormat = geometry.indexFormat === 'uint16' ? LBVH_INDEX_FORMAT.uint16 : LBVH_INDEX_FORMAT.uint32;
                // Copies start on a 4-byte boundary, so an odd 16-bit offset copies one index too many and skips it.
                if (indexFormat === LBVH_INDEX_FORMAT.uint16) indexSkip = (geometry.indexOffset ?? 0) % 2;
            }
            hasFlags = Boolean(geometry.triangleFlagsBuffer);
        } else {
            vertexCount = geometry.positions.length / 3;
            if (geometry.indices) indexFormat = geometry.indices instanceof Uint16Array ? LBVH_INDEX_FORMAT.uint16 : LBVH_INDEX_FORMAT.uint32;
            hasFlags = Boolean(geometry.triangleFlags);
        }
        const placement: MeshPlacement = {
            geometry,
            triangleStart,
            triangleCount,
            innerStart,
            innerCount: Math.max(1, triangleCount - 1),
            vertexCount,
            positionByteStart,
            positionStride,
            indexFormat,
            indexByteStart,
            indexSkip,
            flagsWordStart,
            hasFlags,
        };
        triangleStart += triangleCount;
        innerStart += placement.innerCount;
        positionByteStart += align4(positionBytes(geometry));
        indexByteStart += align4(indexBytes(placement));
        flagsWordStart += hasFlags ? triangleCount : 0;
        return placement;
    });
}

/** The mesh table, the workgroup tiles and the sort's tiles, in one array, and where each part starts. */
export function buildTables(meshes: readonly MeshPlacement[]): {
    tables: Uint32Array<ArrayBuffer>;
    tilesOffset: number;
    tileCount: number;
    sortTilesOffset: number;
    sortTable: { tileCount: number; offsetWords: number };
} {
    const tiles: number[] = [];
    for (const [index, mesh] of meshes.entries()) {
        for (let first = 0; first < mesh.triangleCount; first += LBVH_WORKGROUP_SIZE) tiles.push(index, first);
    }
    const sortTable = sortTileTable(meshes.map((mesh) => mesh.triangleCount));
    const tilesOffset = meshes.length * LBVH_MESH_WORDS;
    const sortTilesOffset = tilesOffset + tiles.length;
    const tables = new Uint32Array(sortTilesOffset + sortTable.tiles.length);
    for (const [index, mesh] of meshes.entries()) {
        const at = index * LBVH_MESH_WORDS;
        tables[at + LBVH_MESH.triangleStart] = mesh.triangleStart;
        tables[at + LBVH_MESH.triangleCount] = mesh.triangleCount;
        tables[at + LBVH_MESH.innerStart] = mesh.innerStart;
        tables[at + LBVH_MESH.innerCount] = mesh.innerCount;
        tables[at + LBVH_MESH.vertexCount] = mesh.vertexCount;
        tables[at + LBVH_MESH.positionOffset] = mesh.positionByteStart / 4;
        tables[at + LBVH_MESH.positionStride] = mesh.positionStride;
        tables[at + LBVH_MESH.indexFormat] = mesh.indexFormat;
        const indexUnits = mesh.indexFormat === LBVH_INDEX_FORMAT.uint16 ? mesh.indexByteStart / 2 : mesh.indexByteStart / 4;
        tables[at + LBVH_MESH.indexOffset] = indexUnits + mesh.indexSkip;
        tables[at + LBVH_MESH.flagsOffset] = mesh.flagsWordStart;
        tables[at + LBVH_MESH.hasFlags] = mesh.hasFlags ? 1 : 0;
    }
    tables.set(tiles, tilesOffset);
    tables.set(sortTable.tiles, sortTilesOffset);
    return { tables, tilesOffset, tileCount: tiles.length / LBVH_TILE_WORDS, sortTilesOffset, sortTable };
}

/**
 * What a batch's buffer sizes depend on. Every field is a sum over the batch's
 * meshes, so a planner can add meshes one at a time ({@link addBatchTotals}).
 */
export interface BatchTotals {
    readonly meshes: number;
    readonly triangles: number;
    /** Radix-tree inner nodes: `max(1, triangles - 1)` per mesh. */
    readonly inner: number;
    /** Vertex bytes in the batch's positions (each mesh's rounded up to 4). */
    readonly positionBytes: number;
    /** Index bytes in the batch's indices (each mesh's rounded up to 4). */
    readonly indexBytes: number;
    /** Words of triangle flags (meshes with flags only). */
    readonly flagWords: number;
    /** Workgroup tiles: `ceil(triangles / LBVH_WORKGROUP_SIZE)` per mesh. */
    readonly tiles: number;
    /** Sort tiles: `ceil(triangles / SORT_TILE_KEYS)` per mesh. */
    readonly sortTiles: number;
}

const NO_MESHES: BatchTotals = { meshes: 0, triangles: 0, inner: 0, positionBytes: 0, indexBytes: 0, flagWords: 0, tiles: 0, sortTiles: 0 };

export function addBatchTotals(a: BatchTotals, b: BatchTotals): BatchTotals {
    return {
        meshes: a.meshes + b.meshes,
        triangles: a.triangles + b.triangles,
        inner: a.inner + b.inner,
        positionBytes: a.positionBytes + b.positionBytes,
        indexBytes: a.indexBytes + b.indexBytes,
        flagWords: a.flagWords + b.flagWords,
        tiles: a.tiles + b.tiles,
        sortTiles: a.sortTiles + b.sortTiles,
    };
}

/** The totals of meshes placed in one batch. */
export function batchTotals(meshes: readonly MeshPlacement[]): BatchTotals {
    return meshes.reduce(
        (totals, mesh) =>
            addBatchTotals(totals, {
                meshes: 1,
                triangles: mesh.triangleCount,
                inner: mesh.innerCount,
                positionBytes: align4(positionBytes(mesh.geometry)),
                indexBytes: align4(indexBytes(mesh)),
                flagWords: mesh.hasFlags ? mesh.triangleCount : 0,
                tiles: Math.ceil(mesh.triangleCount / LBVH_WORKGROUP_SIZE),
                sortTiles: sortTilesOf(mesh.triangleCount),
            }),
        NO_MESHES,
    );
}

/** The totals of these geometries in one batch. */
export function geometryTotals(geometries: readonly GpuMeshGeometry[]): BatchTotals {
    return batchTotals(placeMeshes(geometries));
}

/** Bytes a buffer asked for `bytes` gets: rounded up to 4, at least 16. */
export const allocatedBytes = (bytes: number): number => Math.max(16, align4(bytes));

/** Words of the tables buffer: the mesh table, the workgroup tiles, the sort tiles (at least one). */
export function tableWords(totals: BatchTotals): number {
    return totals.meshes * LBVH_MESH_WORDS + totals.tiles * LBVH_TILE_WORDS + Math.max(1, totals.sortTiles) * SORT_TILE_WORDS;
}

/** The bytes of a batch's buffers by name, split into what the batch frees (`scratch`, with its inputs) and what it hands out (`outputs`). */
export interface BatchBytes<Name extends string = string> {
    /** Each buffer's bytes as allocated. */
    readonly buffers: Readonly<Record<Name, number>>;
    /** Inputs, scratch and uniforms: freed with the batch (or held in a scratch pool). */
    readonly scratch: number;
    /** Output nodes, triangles and parents: handed to the caller. */
    readonly outputs: number;
    readonly total: number;
}

/** The buffers a batch hands to its caller; every other buffer is scratch. */
export const OUTPUT_BUFFERS: readonly string[] = ['outNodes', 'outTriangles', 'outParents'];

/** Sums requested sizes into {@link BatchBytes}. */
export function summarizeBatchBuffers<Name extends string>(requested: Readonly<Record<Name, number>>): BatchBytes<Name> {
    const buffers = Object.fromEntries(Object.entries<number>(requested).map(([name, bytes]) => [name, allocatedBytes(bytes)])) as Record<Name, number>;
    let scratch = 0;
    let outputs = 0;
    for (const [name, bytes] of Object.entries<number>(buffers)) {
        if (OUTPUT_BUFFERS.includes(name)) outputs += bytes;
        else scratch += bytes;
    }
    return { buffers, scratch, outputs, total: scratch + outputs };
}

/** The LBVH batch's buffers, as {@link GpuLbvhBatch} asks for them (before {@link allocatedBytes}). */
export function lbvhBatchBuffers(totals: BatchTotals) {
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
        outTriangles: triangles * GPU_LAYOUT_TRIANGLE_BYTES,
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
export function lbvhBatchBytes(totals: BatchTotals): BatchBytes<keyof ReturnType<typeof lbvhBatchBuffers>> {
    return summarizeBatchBuffers(lbvhBatchBuffers(totals));
}

type BatchBuffer = 'positions' | 'indices' | 'flags' | 'stats' | 'outNodes' | 'outTriangles' | 'outParents';

/** One batch of mesh builds: inputs, scratch and outputs, recorded into one compute pass. */
export class GpuLbvhBatch {
    readonly meshCount: number;
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
        this.label = label;
        this.pipelines = pipelines;
        this.sortPipelines = sortPipelines;
        this.meshCount = geometries.length;
        this.meshes = placeMeshes(geometries);
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
        const sizes = lbvhBatchBuffers(batchTotals(this.meshes));
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
        // The triangles go first: the emit pass reads its leaves' boxes from them.
        stage('triangles', (pass) => run(pass, 'lbvhTriangles', this.tileCount));
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
            return {
                nodes: outNodes,
                triangles: outTriangles,
                parents: outParents,
                meshes: this.meshes.map((mesh, index) => meshResult(mesh, index, words.subarray(index * LBVH_STATS_WORDS, (index + 1) * LBVH_STATS_WORDS))),
                destroy() {
                    outNodes.destroy();
                    outTriangles.destroy();
                    outParents.destroy();
                },
            };
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

/** The buffers a batch's geometry goes into: every mesh's vertices, indices and flags, one after another. */
export interface BatchGeometryBuffers {
    readonly positions: GPUBuffer;
    readonly indices: GPUBuffer;
    readonly flags: GPUBuffer;
}

/** Writes the CPU geometry of a batch into its buffers, once, when the batch is made. */
export function uploadBatchGeometry(queue: GPUQueue, meshes: readonly MeshPlacement[], buffers: BatchGeometryBuffers): void {
    for (const mesh of meshes) {
        const geometry = mesh.geometry;
        if (isGpuMeshBuffers(geometry)) continue;
        queue.writeBuffer(buffers.positions, mesh.positionByteStart, geometry.positions.buffer as ArrayBuffer, geometry.positions.byteOffset, geometry.positions.byteLength);
        const { indices, triangleFlags } = geometry;
        if (indices) {
            // writeBuffer copies whole words; a 16-bit index list of odd length gets one padding index.
            let data: Uint16Array | Uint32Array = indices;
            if (data instanceof Uint16Array && data.length % 2 === 1) {
                const padded = new Uint16Array(data.length + 1);
                padded.set(data);
                data = padded;
            }
            queue.writeBuffer(buffers.indices, mesh.indexByteStart, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
        }
        if (triangleFlags) {
            const flags = triangleFlags.subarray(0, mesh.triangleCount);
            queue.writeBuffer(buffers.flags, mesh.flagsWordStart * 4, flags.buffer as ArrayBuffer, flags.byteOffset, flags.byteLength);
        }
    }
}

/** Records copies of a batch's GPU-buffer geometry into its buffers: the vertices, indices and flags may change between builds. */
export function copyBatchGeometry(encoder: GPUCommandEncoder, meshes: readonly MeshPlacement[], buffers: BatchGeometryBuffers): void {
    for (const mesh of meshes) {
        const geometry = mesh.geometry;
        if (!isGpuMeshBuffers(geometry)) continue;
        const positionStart = (geometry.positionOffsetFloats ?? 0) * 4;
        // Rounded up to whole words, within the buffer (its size was checked when the batch was made).
        const positionSize = Math.min(align4(positionBytes(geometry)), geometry.positionBuffer.size - positionStart);
        encoder.copyBufferToBuffer(geometry.positionBuffer, positionStart, buffers.positions, mesh.positionByteStart, positionSize);
        if (geometry.indexBuffer) {
            const bytesPerIndex = mesh.indexFormat === LBVH_INDEX_FORMAT.uint16 ? 2 : 4;
            const indexStart = ((geometry.indexOffset ?? 0) - mesh.indexSkip) * bytesPerIndex;
            const indexSize = Math.min(align4(indexBytes(mesh)), geometry.indexBuffer.size - indexStart);
            encoder.copyBufferToBuffer(geometry.indexBuffer, indexStart, buffers.indices, mesh.indexByteStart, indexSize);
        }
        if (geometry.triangleFlagsBuffer) {
            encoder.copyBufferToBuffer(geometry.triangleFlagsBuffer, 0, buffers.flags, mesh.flagsWordStart * 4, mesh.triangleCount * 4);
        }
    }
}

/** A mesh's stats words as its tree, checked; `problems` holds what the builder's own checks found. */
export function meshResult(mesh: MeshPlacement, index: number, stats: Uint32Array, problems: string[] = []): GpuMeshTreeRange {
    const floats = new Float32Array(stats.buffer, stats.byteOffset, stats.length);
    const nodeCount = stats[LBVH_STAT.nodeCount];
    if (stats[LBVH_STAT.badIndices] > 0) problems.push(`${stats[LBVH_STAT.badIndices]} triangle corners name a vertex past the last (${mesh.vertexCount})`);
    if (stats[LBVH_STAT.incomplete] > 0) problems.push(`${stats[LBVH_STAT.incomplete]} handed-over words were still unwritten after the wait`);
    if (stats[LBVH_STAT.rootsFinished] !== 1) problems.push(`${stats[LBVH_STAT.rootsFinished]} threads finished the root, not 1`);
    if (nodeCount < 1 || nodeCount > mesh.innerCount) problems.push(`the tree has ${nodeCount} nodes`);
    return {
        nodeOffset: stats[LBVH_STAT.nodeOffset],
        nodeCount,
        triangleOffset: mesh.triangleStart,
        triangleCount: mesh.triangleCount,
        leafCount: stats[LBVH_STAT.leafCount],
        maxDepth: stats[LBVH_STAT.maxDepth],
        bounds: floats.slice(LBVH_STAT.bounds, LBVH_STAT.bounds + 6),
        problem: problems.length > 0 ? `mesh ${index}: ${problems.join('; ')}` : null,
    };
}

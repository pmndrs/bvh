/**
 * The batch plumbing the GPU builders share (the LBVH, gpu-lbvh.ts, and
 * H-PLOC, gpu-hploc.ts): checking options and geometry, planning batches that
 * fit the device and a byte cap, placing a batch's meshes one after another
 * in its buffers, the mesh, workgroup and sort tables, the totals a batch's
 * buffer sizes are computed from, uploading or copying the geometry, reading
 * each mesh's stats as its tree, and reading trees back. The batch layout
 * itself (the mesh table's words, the stats words) is defined with the LBVH's
 * WGSL in gpu-lbvh-wgsl.ts, which H-PLOC's kernels reuse.
 *
 * Moved out of gpu-lbvh.ts unchanged, so neither builder imports the other's
 * file for it.
 */

import { checkNodeFormat, dequantizeNodes } from './quantized-nodes.ts';
import { parentsOfNodes } from './refit-cpu.ts';
import {
    GPU_LAYOUT_MAX_LEAF_PRIMITIVES,
    GPU_LAYOUT_MAX_TRIANGLES,
    gpuLayoutNodeBufferBytes,
    gpuLayoutNodeBytes,
    gpuLayoutTriangleBytes,
    type GpuLayoutMeshBvh,
    type GpuLayoutTriangleLeaves,
} from './gpu-layout.ts';
import { checkTriangleLeaves } from './indexed-leaves.ts';
import { LBVH_INDEX_FORMAT, LBVH_MESH, LBVH_MESH_WORDS, LBVH_STAT, LBVH_TILE_WORDS, LBVH_WORKGROUP_SIZE } from './gpu-lbvh-wgsl.ts';
import { SORT_TILE_WORDS, sortTilesOf, sortTileTable } from './gpu-radix-sort.ts';
import { isGpuMeshBuffers, packedTriangleFlags, meshTriangleCount, type GpuBuildOptions, type GpuMeshGeometry, type GpuMeshTreeRange, type GpuTreeBuffers } from './mesh-input.ts';

/** Buffer usages and the map mode; the globals don't exist in Node, where tests import this file. */
export const USAGE = { MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80, INDIRECT: 0x100 } as const;
export const MAP_MODE_READ = 0x1;

/** Uniform blocks sit 256 bytes apart: the default `minUniformBufferOffsetAlignment`. */
export const UNIFORM_STRIDE = 256;

export function checkOptions(options: GpuBuildOptions): Required<GpuBuildOptions> {
    const maxLeaf = options.maxLeafPrimitives ?? GPU_LAYOUT_MAX_LEAF_PRIMITIVES;
    if (!Number.isInteger(maxLeaf) || maxLeaf < 1 || maxLeaf > GPU_LAYOUT_MAX_LEAF_PRIMITIVES) {
        throw new RangeError(`maxLeafPrimitives must be an integer from 1 to ${GPU_LAYOUT_MAX_LEAF_PRIMITIVES}, got ${maxLeaf}`);
    }
    const traversalCost = options.traversalCost ?? 1;
    const intersectionCost = options.intersectionCost ?? 1;
    if (!(traversalCost > 0) || !(intersectionCost > 0)) throw new RangeError('the SAH costs must be positive');
    const triangleLeading = options.triangleLeading ?? [];
    if (!triangleLeading.every((units) => Number.isInteger(units) && units >= 0)) throw new RangeError('triangleLeading must hold non-negative integers');
    return {
        maxLeafPrimitives: maxLeaf,
        traversalCost,
        intersectionCost,
        triangleLeaves: checkTriangleLeaves(options.triangleLeaves, 'GPU build'),
        nodeFormat: checkNodeFormat(options.nodeFormat, 'GPU build'),
        triangleLeading,
    };
}

/** Checks a batch's `triangleLeading` against its meshes: none, or one entry per mesh. */
export function checkTriangleLeading(triangleLeading: readonly number[], meshCount: number): void {
    if (triangleLeading.length !== 0 && triangleLeading.length !== meshCount) {
        throw new RangeError(`triangleLeading has ${triangleLeading.length} entries for ${meshCount} meshes`);
    }
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
        const { positions } = geometry;
        if (positions.length % 3 !== 0 || positions.length === 0 || !Number.isInteger(triangles) || triangles < 1) {
            throw new RangeError(`${where}: needs at least one triangle, with three floats per vertex and three indices per triangle`);
        }
        // Checks the flags (bit 0 only) and the materials.
        packedTriangleFlags(geometry, triangles, where);
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

/** Options of the batch planners. */
export interface BatchPlanOptions {
    /**
     * Each batch's GPU bytes (inputs, scratch and outputs) stay at or under
     * this, except a single mesh bigger than it, which gets a batch of its own.
     * Default: no cap (batches as big as the device's buffers allow).
     */
    readonly maxBatchBytes?: number;
    /** The trees' triangle format, whose output size the cap counts. Default `'records'`. */
    readonly triangleLeaves?: GpuLayoutTriangleLeaves;
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

/** How far a readback of a batch's usable trees reaches into each buffer (nodes, parents, triangles, in elements); problem meshes are skipped. */
export function treeReadbackExtents(meshes: readonly GpuMeshTreeRange[]): { nodeEnd: number; parentEnd: number; triangleEnd: number } {
    let nodeEnd = 0;
    let parentEnd = 0;
    let triangleEnd = 0;
    for (const mesh of meshes) {
        if (mesh.problem) continue;
        nodeEnd = Math.max(nodeEnd, mesh.nodeOffset + mesh.nodeCount);
        parentEnd = Math.max(parentEnd, (mesh.parentOffset ?? mesh.nodeOffset) + mesh.nodeCount);
        triangleEnd = Math.max(triangleEnd, mesh.triangleOffset + mesh.triangleCount);
    }
    return { nodeEnd, parentEnd, triangleEnd };
}

/** Copies every usable tree in shared GPU buffers back into CPU memory, with one readback. Any GPU builder's trees. */
export async function readTreeLayouts(device: GPUDevice, trees: GpuTreeBuffers, label: string): Promise<(GpuLayoutMeshBvh | null)[]> {
    const { nodeEnd, parentEnd, triangleEnd } = treeReadbackExtents(trees.meshes);
    const quantized = trees.nodeFormat === 'quantized';
    const nodeSize = gpuLayoutNodeBytes(trees.nodeFormat);
    const nodeBytes = nodeEnd * nodeSize;
    const triangleSize = gpuLayoutTriangleBytes(trees.triangleLeaves);
    const triangleBytes = triangleEnd * triangleSize;
    // Without a parents output (P9 lever 7) each tree's parents come from its nodes, on the CPU.
    const parentBytes = trees.withoutParents ? 0 : parentEnd * 4;
    const readback = device.createBuffer({ label: `${label} readback`, size: Math.max(4, nodeBytes + triangleBytes + parentBytes), usage: USAGE.MAP_READ | USAGE.COPY_DST });
    try {
        const encoder = device.createCommandEncoder({ label: `${label} readback` });
        encoder.copyBufferToBuffer(trees.nodes, 0, readback, 0, nodeBytes);
        encoder.copyBufferToBuffer(trees.triangles, 0, readback, nodeBytes, triangleBytes);
        if (parentBytes > 0) encoder.copyBufferToBuffer(trees.parents, 0, readback, nodeBytes + triangleBytes, parentBytes);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(MAP_MODE_READ);
        const bytes = readback.getMappedRange();
        return trees.meshes.map((mesh) => {
            if (mesh.problem) return null;
            // Quantized: the frame slot before the root, then the nodes.
            const nodeStart = (mesh.nodeOffset - (quantized ? 1 : 0)) * nodeSize;
            const triangleStart = nodeBytes + mesh.triangleOffset * triangleSize;
            const parentStart = nodeBytes + triangleBytes + (mesh.parentOffset ?? mesh.nodeOffset) * 4;
            const nodes = bytes.slice(nodeStart, nodeStart + gpuLayoutNodeBufferBytes(mesh.nodeCount, trees.nodeFormat));
            return {
                nodes,
                ...(quantized ? { nodeFormat: 'quantized' as const } : {}),
                triangles: bytes.slice(triangleStart, triangleStart + mesh.triangleCount * triangleSize),
                ...(trees.triangleLeaves === 'indexed' ? { triangleLeaves: 'indexed' as const } : {}),
                parents: trees.withoutParents
                    ? parentsOfNodes(new Uint32Array(dequantizeNodes({ nodes, nodeFormat: trees.nodeFormat, nodeCount: mesh.nodeCount }).nodes), 0, mesh.nodeCount)
                    : new Uint32Array(bytes.slice(parentStart, parentStart + mesh.nodeCount * 4)),
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
    /** The mesh's first triangle in the output triangles: `triangleStart` plus every gap up to and including its own (`triangleLeading`). */
    readonly outTriangleStart: number;
    /** Output triangles left free before the mesh's (`GpuBuildOptions.triangleLeading`). */
    readonly triangleLeading: number;
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

/** Lays the meshes out one after another in every batch buffer; the output triangles leave `triangleLeading[mesh]` free before each mesh's. */
export function placeMeshes(geometries: readonly GpuMeshGeometry[], triangleLeading: readonly number[] = []): MeshPlacement[] {
    let triangleStart = 0;
    let outTriangleStart = 0;
    let innerStart = 0;
    let positionByteStart = 0;
    let indexByteStart = 0;
    let flagsWordStart = 0;
    return geometries.map((geometry, meshIndex) => {
        const triangleCount = meshTriangleCount(geometry);
        const leading = triangleLeading[meshIndex] ?? 0;
        outTriangleStart += leading;
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
            hasFlags = Boolean(geometry.triangleFlags || geometry.triangleMaterials);
        }
        const placement: MeshPlacement = {
            geometry,
            triangleStart,
            triangleCount,
            outTriangleStart,
            triangleLeading: leading,
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
        outTriangleStart += triangleCount;
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
        tables[at + LBVH_MESH.outTriangleStart] = mesh.outTriangleStart;
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
    /** Output triangles: {@link triangles} plus the gaps left before meshes (`triangleLeading`). */
    readonly outTriangles: number;
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

const NO_MESHES: BatchTotals = { meshes: 0, triangles: 0, outTriangles: 0, inner: 0, positionBytes: 0, indexBytes: 0, flagWords: 0, tiles: 0, sortTiles: 0 };

export function addBatchTotals(a: BatchTotals, b: BatchTotals): BatchTotals {
    return {
        meshes: a.meshes + b.meshes,
        triangles: a.triangles + b.triangles,
        outTriangles: a.outTriangles + b.outTriangles,
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
                outTriangles: mesh.triangleLeading + mesh.triangleCount,
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
        const { indices } = geometry;
        // The input's flags with each triangle's material in bits 16-31, as the records get them.
        const triangleFlags = packedTriangleFlags(geometry, mesh.triangleCount, 'mesh');
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
        triangleOffset: mesh.outTriangleStart,
        triangleCount: mesh.triangleCount,
        leafCount: stats[LBVH_STAT.leafCount],
        maxDepth: stats[LBVH_STAT.maxDepth],
        bounds: floats.slice(LBVH_STAT.bounds, LBVH_STAT.bounds + 6),
        problem: problems.length > 0 ? `mesh ${index}: ${problems.join('; ')}` : null,
    };
}

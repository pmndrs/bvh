/**
 * The GPU twin of `writeMergedTriangleIdentity` (static-merge.ts), for a merged
 * static group's tree built on the GPU: the tree's records are already in the
 * scene's triangle buffer, in tree order, each with the builder's original
 * index (an index into the group's geometry). One thread per record turns
 * that into the merged identity, exactly as the CPU function does:
 *
 * - word 3 (original index) becomes the triangle's index in its source mesh;
 * - word 11 (padding) becomes the source instance index;
 * - flags (word 7) get bit 1 (`TRIANGLE_FLAG_FLIP_FRONT_FACE`) for a mirrored
 *   source and bits 16-31 the triangle's material + 1 (`sourceMaterials`: the
 *   source triangle's, else the source instance's), OR-ed in, so bit 0 (the
 *   filter flag) stays.
 *
 * The group's sources are contiguous in its geometry (`StaticMergeSource.firstTriangle`),
 * so the table is one entry per source, not per triangle: the kernel finds a
 * record's source by binary search. Run it once on fresh records, right after
 * they are copied in: it reads the builder's index.
 */

import {
    INDEXED_FLAG_SHIFT,
    INDEXED_ID_MASK,
    INDEXED_TRIANGLE_WORDS,
    MATERIAL_SHIFT,
    MESH_HEADER_WORD,
    TRIANGLE_FLAG_FLIP_FRONT_FACE,
    TRIANGLE_WORD,
    TRIANGLE_WORDS,
} from '../gpu-layout.ts';
import type { StaticMergeGroup } from '../trace/static-merge.ts';
import { dispatchWorkgroups2d, WORKGROUP_INDEX_WGSL } from '../gpu-workgroup-scan.ts';

const WORKGROUP_SIZE = 64;
/** Words per source in the table: `(firstTriangle, instanceIndex, flagsToSet, 0)`. */
export const MERGED_SOURCE_WORDS = 4;
const USAGE = { COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80 } as const;

export function mergedIdentityWgsl(): string {
    return /* wgsl */ `
struct BvhMergedParams {
    triangleWordOffset: u32,
    triangleCount: u32,
    sourceCount: u32,
    sideWordOffset: u32,   // indexed: where the group's side table starts, in words
}

@group(0) @binding(0) var<storage, read_write> bvhMergedTriangles: array<u32>;
@group(0) @binding(1) var<storage, read> bvhMergedSources: array<vec4u>;
@group(0) @binding(2) var<uniform> bvhMergedParams: BvhMergedParams;
// Each merged triangle's material code (material + 1, 0: none), 16 bits, two to a word, by geometry triangle.
@group(0) @binding(3) var<storage, read> bvhMergedMaterials: array<u32>;

fn bvhMergedMaterialCode(merged: u32) -> u32 {
    return (bvhMergedMaterials[merged >> 1u] >> ((merged & 1u) * 16u)) & 0xffffu;
}

const BVH_MERGED_WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE}u;
const BVH_MERGED_TRIANGLE_WORDS: u32 = ${TRIANGLE_WORDS}u;
${WORKGROUP_INDEX_WGSL.replace('fn flatWorkgroupIndex', 'fn bvhMergedWorkgroupIndex')}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn bvhMergedIdentity(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let slot = bvhMergedWorkgroupIndex(workgroup, groups) * BVH_MERGED_WORKGROUP_SIZE + localIndex;
    if (slot >= bvhMergedParams.triangleCount) {
        return;
    }
    let base = bvhMergedParams.triangleWordOffset + slot * BVH_MERGED_TRIANGLE_WORDS;
    let merged = bvhMergedTriangles[base + ${TRIANGLE_WORD.originalIndex}u];
    // The last source whose first triangle is at or before this one.
    var low = 0u;
    var high = bvhMergedParams.sourceCount;
    while (high - low > 1u) {
        let middle = (low + high) / 2u;
        if (bvhMergedSources[middle].x <= merged) {
            low = middle;
        } else {
            high = middle;
        }
    }
    let source = bvhMergedSources[low];
    bvhMergedTriangles[base + ${TRIANGLE_WORD.originalIndex}u] = merged - source.x;
    bvhMergedTriangles[base + ${TRIANGLE_WORD.padding}u] = source.y;
    bvhMergedTriangles[base + ${TRIANGLE_WORD.flags}u] = bvhMergedTriangles[base + ${TRIANGLE_WORD.flags}u] | source.z | (bvhMergedMaterialCode(merged) << ${MATERIAL_SHIFT}u);
}

// The indexed twin (writeMergedIndexedIdentity): word 3 of a 16-byte triangle becomes the
// source triangle with the flip flag added (bit 29), and the group's side table, at
// sideWordOffset, gets the slot's source number and material (source | material << 16).
// The source table and the header are written by the CPU: they don't depend on tree order.
@compute @workgroup_size(${WORKGROUP_SIZE})
fn bvhMergedIdentityIndexed(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let slot = bvhMergedWorkgroupIndex(workgroup, groups) * BVH_MERGED_WORKGROUP_SIZE + localIndex;
    if (slot >= bvhMergedParams.triangleCount) {
        return;
    }
    let word = bvhMergedParams.triangleWordOffset + slot * ${INDEXED_TRIANGLE_WORDS}u + 3u;
    let idFlags = bvhMergedTriangles[word];
    let merged = idFlags & ${INDEXED_ID_MASK}u;
    var low = 0u;
    var high = bvhMergedParams.sourceCount;
    while (high - low > 1u) {
        let middle = (low + high) / 2u;
        if (bvhMergedSources[middle].x <= merged) {
            low = middle;
        } else {
            high = middle;
        }
    }
    let source = bvhMergedSources[low];
    let flags = (idFlags >> ${INDEXED_FLAG_SHIFT}u) | (source.z & ${TRIANGLE_FLAG_FLIP_FRONT_FACE}u);
    bvhMergedTriangles[word] = (merged - source.x) | (flags << ${INDEXED_FLAG_SHIFT}u);
    bvhMergedTriangles[bvhMergedParams.sideWordOffset + slot] = low | (bvhMergedMaterialCode(merged) << ${MATERIAL_SHIFT}u);
}
`;
}

/** The kernel's per-triangle material codes for a group (`sourceMaterials`), two to a word. */
export function mergedMaterialTable(group: StaticMergeGroup): Uint32Array<ArrayBuffer> {
    const codes = group.sourceMaterials;
    const table = new Uint32Array(Math.max(1, Math.ceil(codes.length / 2)));
    for (let triangle = 0; triangle < codes.length; triangle++) table[triangle >> 1] |= codes[triangle] << ((triangle & 1) * 16);
    return table;
}

/** The kernel's per-source table for a group, sources in geometry order: (first triangle, instance index, flip flag, 0). */
export function mergedSourceTable(group: StaticMergeGroup): Uint32Array<ArrayBuffer> {
    const table = new Uint32Array(Math.max(1, group.sources.length) * MERGED_SOURCE_WORDS);
    group.sources.forEach((source, index) => {
        table[index * MERGED_SOURCE_WORDS] = source.firstTriangle;
        table[index * MERGED_SOURCE_WORDS + 1] = source.instanceIndex;
        table[index * MERGED_SOURCE_WORDS + 2] = source.mirrored ? TRIANGLE_FLAG_FLIP_FRONT_FACE : 0;
    });
    return table;
}

/** Records the identity pass over merged groups' records in the scene's triangle buffer. */
export class GpuMergedIdentity {
    private readonly device: GPUDevice;
    private readonly label: string;
    private pipeline: GPUComputePipeline | null = null;
    private indexedPipeline: GPUComputePipeline | null = null;
    private compiling: Promise<void> | null = null;
    /** Buffers of passes recorded since the last {@link releaseRetired}, freed once their frame was submitted. */
    private retired: GPUBuffer[] = [];

    constructor(device: GPUDevice, label = 'bvh') {
        this.device = device;
        this.label = `${label} merged identity`;
    }

    prepare(): Promise<void> {
        this.compiling ??= (async () => {
            const module = this.device.createShaderModule({ label: this.label, code: mergedIdentityWgsl() });
            [this.pipeline, this.indexedPipeline] = await Promise.all(
                ['bvhMergedIdentity', 'bvhMergedIdentityIndexed'].map((entryPoint) => this.device.createComputePipelineAsync({ label: this.label, layout: 'auto', compute: { module, entryPoint } })),
            );
        })();
        return this.compiling;
    }

    get ready(): boolean {
        return this.pipeline !== null;
    }

    /** Bytes of tables and uniforms waiting for {@link releaseRetired}. */
    get pendingBytes(): number {
        return this.retired.reduce((sum, buffer) => sum + buffer.size, 0);
    }

    /**
     * Records the pass for one group whose `triangleCount` fresh records start
     * at record `triangleOffset` of `triangles`, after whatever wrote them in
     * `encoder`.
     */
    encode(encoder: GPUCommandEncoder, triangles: GPUBuffer, group: StaticMergeGroup, triangleOffset: number, triangleCount: number, indexed?: { readonly vertexBase: number }): void {
        const pipeline = indexed ? this.indexedPipeline : this.pipeline;
        if (!pipeline) throw new Error('@pmndrs/bvh: call GpuMergedIdentity.prepare() before encode()');
        if (triangleCount !== group.sourceInstances.length) throw new RangeError(`the merged tree has ${triangleCount} triangles, its group ${group.sourceInstances.length}`);
        const table = mergedSourceTable(group);
        const sources = this.device.createBuffer({ label: `${this.label} sources`, size: table.byteLength, usage: USAGE.STORAGE | USAGE.COPY_DST });
        this.device.queue.writeBuffer(sources, 0, table);
        const materialTable = mergedMaterialTable(group);
        const materials = this.device.createBuffer({ label: `${this.label} materials`, size: materialTable.byteLength, usage: USAGE.STORAGE | USAGE.COPY_DST });
        this.device.queue.writeBuffer(materials, 0, materialTable);
        const params = this.device.createBuffer({ label: `${this.label} params`, size: 16, usage: USAGE.UNIFORM | USAGE.COPY_DST });
        if (indexed) {
            this.writeIndexedTables(triangles, group, triangleOffset, triangleCount, indexed.vertexBase);
            const sideStart = (triangleOffset - 1 - Math.ceil(triangleCount / 4)) * INDEXED_TRIANGLE_WORDS;
            this.device.queue.writeBuffer(params, 0, new Uint32Array([triangleOffset * INDEXED_TRIANGLE_WORDS, triangleCount, group.sources.length, sideStart]));
        } else {
            this.device.queue.writeBuffer(params, 0, new Uint32Array([triangleOffset * TRIANGLE_WORDS, triangleCount, group.sources.length, 0]));
        }
        this.retired.push(sources, params, materials);
        const pass = encoder.beginComputePass({ label: this.label });
        pass.setPipeline(pipeline);
        pass.setBindGroup(
            0,
            this.device.createBindGroup({
                label: this.label,
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: triangles } },
                    { binding: 1, resource: { buffer: sources } },
                    { binding: 2, resource: { buffer: params } },
                    { binding: 3, resource: { buffer: materials } },
                ],
            }),
        );
        dispatchWorkgroups2d(pass, Math.ceil(triangleCount / WORKGROUP_SIZE), this.device.limits.maxComputeWorkgroupsPerDimension);
        pass.end();
    }

    /**
     * An indexed group's region before its triangles: the source table
     * (instance index per source) and the header (vertex base, side and
     * source table distances), as `writeMergedIndexedIdentity` lays them out.
     * The side table is the kernel's.
     */
    private writeIndexedTables(triangles: GPUBuffer, group: StaticMergeGroup, triangleOffset: number, triangleCount: number, vertexBase: number): void {
        const sideUnits = Math.ceil(triangleCount / 4);
        const sourceUnits = Math.ceil(group.sources.length / 4);
        const sourceStart = triangleOffset - 1 - sideUnits - sourceUnits;
        if (sourceStart < 0) throw new RangeError(`the merged group at ${triangleOffset} has no room for its side tables`);
        const sources = new Uint32Array(sourceUnits * INDEXED_TRIANGLE_WORDS);
        group.sources.forEach((source, slot) => (sources[slot] = source.instanceIndex));
        this.device.queue.writeBuffer(triangles, sourceStart * 16, sources);
        const header = new Uint32Array(INDEXED_TRIANGLE_WORDS);
        header[MESH_HEADER_WORD.vertexBase] = vertexBase;
        header[MESH_HEADER_WORD.sideTableDistance] = 1 + sideUnits;
        header[MESH_HEADER_WORD.sourceTableDistance] = 1 + sideUnits + sourceUnits;
        this.device.queue.writeBuffer(triangles, (triangleOffset - 1) * 16, header);
    }

    /** Frees the tables of passes recorded before the last submit. */
    releaseRetired(): void {
        for (const buffer of this.retired) buffer.destroy();
        this.retired = [];
    }

    dispose(): void {
        this.releaseRetired();
    }
}

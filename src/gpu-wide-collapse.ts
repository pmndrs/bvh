/**
 * The GPU wide collapse (docs/internals/design/wide-nodes.md, section 2): binary `f32` trees
 * in GPU buffers (as H-PLOC writes them) into BVH4 or CWBVH nodes, on the GPU. The kernels are
 * gpu-wide-collapse-wgsl.ts; this file plans a job's buffers, records its passes and reads its
 * counts back.
 *
 * A job takes any number of meshes from one node buffer and one triangle buffer. Its output is
 * one buffer: each mesh's wide nodes from `dstNode` (room for the mesh's binary node count plus
 * one, the most a collapse can need), then, for CWBVH, each mesh's triangles in its leaves'
 * order from `dstTriangle`. BVH4 keeps the binary order, so its triangles are not copied. The
 * counts read back say how many nodes each mesh used, which is what a scene copies out.
 */

import { GPU_LAYOUT_INDEXED_TRIANGLE_BYTES, GPU_LAYOUT_TRIANGLE_BYTES } from './gpu-layout.ts';
import {
    WIDE_COLLAPSE_CONTROL_HEADER,
    WIDE_COLLAPSE_COUNTER_WORDS,
    WIDE_COLLAPSE_INFO_WORDS,
    WIDE_COLLAPSE_MESH_WORDS,
    WIDE_COLLAPSE_PROBLEM,
    WIDE_COLLAPSE_WORKGROUP,
    wideCollapseWgsl,
} from './gpu-wide-collapse-wgsl.ts';
import { wideNodeBytes, type WideNodeFormat } from './wide-nodes.ts';

const STORAGE = 0x80;
const COPY_DST = 0x8;
const COPY_SRC = 0x4;
const UNIFORM = 0x40;
const INDIRECT = 0x100;
const MAP_READ = 0x1;
const PARAMS_BYTES = 32;
const MAX_GROUPS_X = 65535;

/** One mesh of a job: where its binary tree sits in the job's buffers. */
export interface WideCollapseInput {
    /** Its root in the node buffer, in 64-byte nodes. */
    readonly srcNode: number;
    readonly nodeCount: number;
    /** Its first triangle in the triangle buffer, in triangles. */
    readonly srcTriangle: number;
    readonly triangleCount: number;
    /** Levels of binary nodes on its longest path (the builder's `maxDepth`). */
    readonly maxDepth: number;
}

/** One mesh of a finished job: where its wide tree sits in the output, and how big it is. */
export interface WideCollapseOutput {
    /** Its root in the output's node region, in wide nodes. */
    readonly dstNode: number;
    /** Wide nodes used. */
    readonly nodeCount: number;
    /** CWBVH: its first triangle in the output's triangle region, in triangles (BVH4: -1, the source order stands). */
    readonly dstTriangle: number;
    readonly triangleCount: number;
    /** Stack entries the mesh walk needs (BVH4: siblings pushed; CWBVH: depth). */
    readonly stackNeed: number;
}

/** A recorded job: its output buffer, and its counts once the GPU is done. */
export interface WideCollapseRun {
    readonly output: GPUBuffer;
    /** Bytes of the output's node region; the triangle region (CWBVH) starts here. */
    readonly nodeRegionBytes: number;
    readonly format: WideNodeFormat;
    /** Reads the counts back (after the encoder was submitted). Throws on a problem flag. */
    counts(): Promise<WideCollapseOutput[]>;
    /** Frees the job's buffers, the output included. */
    destroy(): void;
}

export interface GpuWideCollapseOptions {
    readonly format: WideNodeFormat;
    /** The triangles' format: CWBVH copies 48-byte records or 16-byte indexed triangles. Default `'records'`. */
    readonly triangleLeaves?: 'records' | 'indexed';
    readonly label?: string;
}

export class GpuWideCollapse {
    private constructor(
        private readonly device: GPUDevice,
        readonly format: WideNodeFormat,
        private readonly unitBytes: number,
        private readonly label: string,
        private readonly cost: GPUComputePipeline,
        private readonly level: GPUComputePipeline,
        private readonly args: GPUComputePipeline,
    ) {}

    static async create(device: GPUDevice, options: GpuWideCollapseOptions): Promise<GpuWideCollapse> {
        const unitBytes = options.triangleLeaves === 'indexed' ? GPU_LAYOUT_INDEXED_TRIANGLE_BYTES : GPU_LAYOUT_TRIANGLE_BYTES;
        const label = options.label ?? 'wide collapse';
        const module = device.createShaderModule({ label, code: wideCollapseWgsl(options.format, unitBytes / 4) });
        const make = (entryPoint: string) => device.createComputePipelineAsync({ label: `${label} ${entryPoint}`, layout: 'auto', compute: { module, entryPoint } });
        const [cost, level, args] = await Promise.all([make('bvhwCost'), make('bvhwLevel'), make('bvhwArgs')]);
        return new GpuWideCollapse(device, options.format, unitBytes, label, cost, level, args);
    }

    /** Records a job into `encoder` (its own compute pass). Submit the encoder, then await `counts()`. */
    encode(encoder: GPUCommandEncoder, nodes: GPUBuffer, triangles: GPUBuffer, meshes: readonly WideCollapseInput[]): WideCollapseRun {
        if (meshes.length === 0) throw new RangeError('GpuWideCollapse: a job needs a mesh');
        const device = this.device;
        const cwbvh = this.format === 'cwbvh8';
        const M = cwbvh ? 8 : 4;
        const wideBytes = wideNodeBytes(this.format);
        const buffers: GPUBuffer[] = [];
        const make = (name: string, size: number, usage: number): GPUBuffer => {
            const buffer = device.createBuffer({ label: `${this.label} ${name}`, size: Math.max(16, Math.ceil(size / 16) * 16), usage });
            buffers.push(buffer);
            return buffer;
        };

        // The plan: scratch per binary node, room per mesh for nodeCount + 1 wide nodes, CWBVH's triangles after.
        let totalNodes = 0;
        let dstNodes = 0;
        let dstTriangles = 0;
        let maxDepth = 1;
        const table = new Uint32Array(meshes.length * WIDE_COLLAPSE_MESH_WORDS);
        const outputs: { dstNode: number; dstTriangle: number; triangleCount: number }[] = [];
        meshes.forEach((mesh, index) => {
            const at = index * WIDE_COLLAPSE_MESH_WORDS;
            table.set([mesh.srcNode, mesh.nodeCount, mesh.srcTriangle, mesh.triangleCount, dstNodes, dstTriangles, totalNodes, 0], at);
            outputs.push({ dstNode: dstNodes, dstTriangle: cwbvh ? dstTriangles : -1, triangleCount: mesh.triangleCount });
            totalNodes += mesh.nodeCount;
            dstNodes += mesh.nodeCount + 1;
            if (cwbvh) dstTriangles += mesh.triangleCount;
            maxDepth = Math.max(maxDepth, mesh.maxDepth);
        });
        const nodeRegionBytes = dstNodes * wideBytes;
        const frontierCapacity = dstNodes;
        const meshTable = make('meshes', table.byteLength, STORAGE | COPY_DST);
        device.queue.writeBuffer(meshTable, 0, table);
        const scratchF = make('costs', totalNodes * (M + 6) * 4, STORAGE);
        const scratchU = make('info', totalNodes * WIDE_COLLAPSE_INFO_WORDS * 4, STORAGE | COPY_DST);
        const output = device.createBuffer({ label: `${this.label} output`, size: Math.max(16, nodeRegionBytes + dstTriangles * this.unitBytes), usage: STORAGE | COPY_SRC | COPY_DST });
        const control = make('control', (WIDE_COLLAPSE_CONTROL_HEADER + meshes.length * WIDE_COLLAPSE_COUNTER_WORDS) * 4, STORAGE | COPY_SRC | COPY_DST);
        const frontier = make('frontier', 2 * frontierCapacity * 16, STORAGE | COPY_DST);
        const indirect = make('indirect', 16, STORAGE | INDIRECT | COPY_DST);
        const readback = make('readback', (WIDE_COLLAPSE_CONTROL_HEADER + meshes.length * WIDE_COLLAPSE_COUNTER_WORDS) * 4, MAP_READ | COPY_DST);

        // Each mesh's root starts the first frontier; each mesh's node counter starts past its root.
        const controlWords = new Uint32Array(WIDE_COLLAPSE_CONTROL_HEADER + meshes.length * WIDE_COLLAPSE_COUNTER_WORDS);
        controlWords[0] = meshes.length;
        meshes.forEach((_, index) => (controlWords[WIDE_COLLAPSE_CONTROL_HEADER + index * WIDE_COLLAPSE_COUNTER_WORDS] = 1));
        device.queue.writeBuffer(control, 0, controlWords);
        const roots = new Uint32Array(meshes.length * 4);
        meshes.forEach((_, index) => roots.set([index, 0, 0, 0], index * 4));
        device.queue.writeBuffer(frontier, 0, roots);
        device.queue.writeBuffer(indirect, 0, new Uint32Array([Math.ceil(meshes.length / WIDE_COLLAPSE_WORKGROUP), 1, 1, 0]));

        // One small uniform per dispatch kind and level: the parity of the frontier halves.
        const levels = maxDepth + 1;
        const params = (parity: number, level: number): GPUBuffer => {
            const buffer = make(`params ${level}`, PARAMS_BYTES, UNIFORM | COPY_DST);
            device.queue.writeBuffer(buffer, 0, new Uint32Array([meshes.length, totalNodes, frontierCapacity, parity, nodeRegionBytes / 4, level, 0, 0]));
            return buffer;
        };
        const costParams = params(0, 0);
        const entries = (pipeline: GPUComputePipeline, bindings: Record<number, GPUBuffer>) =>
            device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: Object.entries(bindings).map(([binding, buffer]) => ({ binding: Number(binding), resource: { buffer } })) });
        const costGroup = entries(this.cost, { 0: nodes, 2: meshTable, 3: scratchF, 4: scratchU, 8: costParams });

        encoder.clearBuffer(scratchU);
        encoder.clearBuffer(output);
        const pass = encoder.beginComputePass({ label: `${this.label} (${this.format})` });
        // The dynamic program: one dispatch per level of the deepest tree.
        const groups = Math.ceil(totalNodes / WIDE_COLLAPSE_WORKGROUP);
        const x = Math.min(groups, MAX_GROUPS_X);
        const y = Math.ceil(groups / x);
        pass.setPipeline(this.cost);
        pass.setBindGroup(0, costGroup);
        for (let level = 0; level < maxDepth; level++) pass.dispatchWorkgroups(x, y);
        // The decode, one wide level at a time (indirect: the frontier's length decides).
        for (let level = 0; level < levels; level++) {
            const parity = level % 2;
            const levelParams = params(parity, level);
            pass.setPipeline(this.level);
            // BVH4 copies no triangles, so its level kernel has no binding 1.
            const levelBindings: Record<number, GPUBuffer> = { 0: nodes, 2: meshTable, 3: scratchF, 4: scratchU, 5: output, 6: control, 7: frontier, 8: levelParams };
            if (cwbvh) levelBindings[1] = triangles;
            pass.setBindGroup(0, entries(this.level, levelBindings));
            pass.dispatchWorkgroupsIndirect(indirect, 0);
            pass.setPipeline(this.args);
            pass.setBindGroup(0, entries(this.args, { 6: control, 8: levelParams, 9: indirect }));
            pass.dispatchWorkgroups(1);
        }
        pass.end();
        encoder.copyBufferToBuffer(control, 0, readback, 0, readback.size);

        const format = this.format;
        let destroyed = false;
        return {
            output,
            nodeRegionBytes,
            format,
            async counts(): Promise<WideCollapseOutput[]> {
                await readback.mapAsync(MAP_READ);
                const words = new Uint32Array(readback.getMappedRange().slice(0));
                readback.unmap();
                const problems = words[2];
                if (problems !== 0) {
                    const names = Object.entries(WIDE_COLLAPSE_PROBLEM).filter(([, bit]) => (problems & bit) !== 0).map(([name]) => name);
                    throw new Error(`GpuWideCollapse (${format}): ${names.join(', ')}`);
                }
                return outputs.map((mesh, index) => {
                    const at = WIDE_COLLAPSE_CONTROL_HEADER + index * WIDE_COLLAPSE_COUNTER_WORDS;
                    const used = words[at + 1];
                    if (cwbvh && used !== mesh.triangleCount) throw new Error(`GpuWideCollapse: mesh ${index}'s leaves hold ${used} of ${mesh.triangleCount} triangles`);
                    return { dstNode: mesh.dstNode, nodeCount: words[at], dstTriangle: mesh.dstTriangle, triangleCount: mesh.triangleCount, stackNeed: words[at + 2] };
                });
            },
            destroy(): void {
                if (destroyed) return;
                destroyed = true;
                output.destroy();
                for (const buffer of buffers) buffer.destroy();
            },
        };
    }
}

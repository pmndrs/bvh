/**
 * The GPU side of the walk for apps that don't write their own kernel: uploads an
 * assembled scene's three buffers (`uploadTraceScene`) and runs batches of rays
 * from buffers through the WGSL walk (`TraceBatchKernel`, over `batchKernelWgsl`).
 * Ported from ddgi-core's `platform/trace/trace-gpu.ts`. Its platform-module
 * wrapper (`createTraceModule`) stays in ddgi: an app that wants the walk inline
 * in its own kernel includes `traceModuleWgsl` and binds the three buffers itself.
 *
 * The caller owns the device (nothing here creates one), and errors are the
 * caller's to listen for with `addEventListener('uncapturederror')`.
 */

import type { TraceSceneData } from './scene-assembly.ts';
import { BATCH_PARAMS_BYTES, BATCH_PARAMS_WORD, BATCH_WORKGROUP_SIZE, batchKernelWgsl, type AnyHitOrder, type WideDecode, type WideStack, type WideWalk } from './wgsl.ts';
import { RAY_MASK_ALL, type NodeFormat, type PositionLayout, type TriangleLeaves } from './layout.ts';
import { RAY_BYTES } from './rays.ts';

/** The three storage buffers the tracer reads, and an indexed scene's fourth. */
export interface TraceSceneBuffers {
    readonly nodes: GPUBuffer;
    readonly triangles: GPUBuffer;
    readonly topLevel: GPUBuffer;
    /** An indexed scene's positions (`TraceSceneData.positions`), bound fourth. Absent or null for records. */
    readonly positions?: GPUBuffer | null;
    /** The scene's triangle format; absent means records. A kernel refuses a scene of the other format. */
    readonly triangleLeaves?: TriangleLeaves;
    /** An indexed scene's position layout; absent means `'vec4'`. */
    readonly positionLayout?: PositionLayout;
    /** The scene's node format; absent means `'f32'`. A kernel refuses a scene of the other format. */
    readonly nodeFormat?: NodeFormat;
    /** Bytes held on the GPU. */
    readonly byteLength: number;
    destroy(): void;
}

function checkStorageBinding(device: GPUDevice, name: string, byteLength: number): void {
    const bindingLimit = device.limits.maxStorageBufferBindingSize;
    if (byteLength > bindingLimit) throw new RangeError(`trace buffer '${name}' needs maxStorageBufferBindingSize >= ${byteLength}, the device allows ${bindingLimit}`);
    const bufferLimit = device.limits.maxBufferSize;
    if (byteLength > bufferLimit) throw new RangeError(`trace buffer '${name}' needs maxBufferSize >= ${byteLength}, the device allows ${bufferLimit}`);
    if (byteLength === 0) throw new RangeError(`trace buffer '${name}' is empty`);
}

/**
 * Copies an assembled scene to the GPU. Throws a RangeError naming the
 * limit to request if a buffer is larger than one storage binding may be
 * (128 MB on a default device, about 2.7 M triangles).
 */
export function uploadTraceScene(device: GPUDevice, scene: TraceSceneData, label = 'trace'): TraceSceneBuffers {
    const parts: [string, ArrayBuffer][] = [
        ['nodes', scene.nodes],
        ['triangles', scene.triangles],
        ['top level', scene.topLevel.buffer],
    ];
    const indexed = scene.triangleLeaves === 'indexed';
    if (indexed) parts.push(['positions', scene.positions]);
    for (const [name, data] of parts) checkStorageBinding(device, name, data.byteLength);

    const buffers = parts.map(([name, data]) => {
        const buffer = device.createBuffer({
            label: `${label} ${name}`,
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buffer, 0, data);
        return buffer;
    });
    const [nodes, triangles, topLevel, positions] = buffers;
    return {
        nodes,
        triangles,
        topLevel,
        positions: indexed ? positions : null,
        triangleLeaves: scene.triangleLeaves ?? 'records',
        positionLayout: scene.positionLayout ?? 'vec4',
        nodeFormat: scene.nodeFormat ?? 'f32',
        byteLength: scene.byteLength,
        destroy: () => buffers.forEach((buffer) => buffer.destroy()),
    };
}

export interface TraceBatch {
    readonly mode: 'closest' | 'any';
    /** At least `count` rays of 32 bytes (`BvhRay`). */
    readonly rays: GPUBuffer;
    /** Closest: `count` records of `HIT_RECORD_BYTES` (`HIT_RECORD_COMPACT_BYTES` with `hitNormal: false`). Any: `count` u32, 1 = blocked. */
    readonly results: GPUBuffer;
    readonly count: number;
    /** The batch's ray mask (one for the whole batch). Default all ones. Ignored by a kernel made with `rayMasks: false`. */
    readonly mask?: number;
    /** Optional GPU timing for the dispatch's pass, for example from `GpuTimer.computePass()`. */
    readonly timestampWrites?: GPUComputePassTimestampWrites;
}

interface SceneBindGroups {
    readonly closest: GPUBindGroup;
    readonly any: GPUBindGroup;
}

export interface TraceBatchKernelOptions {
    readonly label?: string;
    /** The triangle filter (`bvhTrianglePasses`); default: every triangle blocks. */
    readonly triangleFilter?: string;
    /** Default true: each batch has a ray mask (`TraceBatch.mask`). False compiles masks out. */
    readonly rayMasks?: boolean;
    /** Default true: 40-byte closest-hit records with material and normal. False: the 24-byte compact record. */
    readonly hitNormal?: boolean;
    /** The scenes' triangle format, compiled in. Default `'records'`. */
    readonly triangleLeaves?: TriangleLeaves;
    /** Indexed only: `'vec4'` (the default) or `'packed'` positions, compiled in. */
    readonly positionLayout?: PositionLayout;
    /** The scenes' node format, compiled in. Default `'f32'`. */
    readonly nodeFormat?: NodeFormat;
    /** The any-hit walk's child order, compiled in (`AnyHitOrder`). Default `'far'` (best for rays from surfaces); `'larger'` for rays from outside the geometry or across open space. */
    readonly anyHitOrder?: AnyHitOrder;
    /** `nodeFormat: 'cwbvh8'`: the mesh walk's stack, `'private'` (the default) or `'workgroup'` (`TraceModuleOptions.wideStack`). */
    readonly wideStack?: WideStack;
    /** `nodeFormat: 'bvh4'`: the walk's shape, `'nested'` (the default) or `'single'` (`TraceModuleOptions.wideWalk`). */
    readonly wideWalk?: WideWalk;
    /** Wide formats: `'exact'` (the default) or `'folded'` plane decode (`TraceModuleOptions.wideDecode`). */
    readonly wideDecode?: WideDecode;
}

/**
 * Runs batches of rays through the trace module. Group 0 is the scene; group 1
 * is the batch's rays and results (and, with ray masks, its mask). One kernel
 * serves any number of scenes.
 */
export class TraceBatchKernel {
    private readonly sceneGroups = new WeakMap<TraceSceneBuffers, SceneBindGroups>();
    /** One small uniform per mask used: dispatches recorded into one command buffer may use different masks. */
    private readonly maskBuffers = new Map<number, GPUBuffer>();

    private constructor(
        private readonly device: GPUDevice,
        private readonly closestPipeline: GPUComputePipeline,
        private readonly anyPipeline: GPUComputePipeline,
        readonly rayMasks: boolean,
        readonly triangleLeaves: TriangleLeaves,
        readonly positionLayout: PositionLayout,
        readonly nodeFormat: NodeFormat,
    ) {}

    static async create(device: GPUDevice, options: TraceBatchKernelOptions = {}): Promise<TraceBatchKernel> {
        const { label = 'trace', triangleFilter, rayMasks = true, hitNormal = true, triangleLeaves = 'records', positionLayout = 'vec4', nodeFormat = 'f32', anyHitOrder, wideStack, wideWalk, wideDecode } = options;
        const pipelineFor = (mode: 'closest' | 'any') =>
            device.createComputePipelineAsync({
                label: `${label} batch ${mode}`,
                layout: 'auto',
                compute: {
                    module: device.createShaderModule({ label: `${label} batch ${mode}`, code: batchKernelWgsl(mode, { triangleFilter, rayMasks, hitNormal, triangleLeaves, positionLayout, nodeFormat, anyHitOrder, wideStack, wideWalk, wideDecode }) }),
                    entryPoint: 'bvhBatch',
                },
            });
        const [closestPipeline, anyPipeline] = await Promise.all([pipelineFor('closest'), pipelineFor('any')]);
        return new TraceBatchKernel(device, closestPipeline, anyPipeline, rayMasks, triangleLeaves, positionLayout, nodeFormat);
    }

    private maskBuffer(mask: number): GPUBuffer {
        let buffer = this.maskBuffers.get(mask);
        if (!buffer) {
            buffer = this.device.createBuffer({ label: `trace batch mask ${mask.toString(16)}`, size: BATCH_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            const words = new Uint32Array(BATCH_PARAMS_BYTES / 4);
            words[BATCH_PARAMS_WORD.rayMask] = mask;
            this.device.queue.writeBuffer(buffer, 0, words);
            this.maskBuffers.set(mask, buffer);
        }
        return buffer;
    }

    /** Frees the mask uniforms. The pipelines need no freeing. */
    destroy(): void {
        for (const buffer of this.maskBuffers.values()) buffer.destroy();
        this.maskBuffers.clear();
    }

    private groupsFor(scene: TraceSceneBuffers): SceneBindGroups {
        let groups = this.sceneGroups.get(scene);
        if (!groups) {
            const leaves = scene.triangleLeaves ?? 'records';
            if (leaves !== this.triangleLeaves) throw new RangeError(`TraceBatchKernel: the kernel reads ${this.triangleLeaves} triangles, the scene holds ${leaves}`);
            if (leaves === 'indexed' && (scene.positionLayout ?? 'vec4') !== this.positionLayout) {
                throw new RangeError(`TraceBatchKernel: the kernel reads ${this.positionLayout} positions, the scene holds ${scene.positionLayout ?? 'vec4'}`);
            }
            const nodes = scene.nodeFormat ?? 'f32';
            if (nodes !== this.nodeFormat) throw new RangeError(`TraceBatchKernel: the kernel reads ${this.nodeFormat} nodes, the scene holds ${nodes}`);
            if (leaves === 'indexed' && !scene.positions) throw new RangeError('TraceBatchKernel: an indexed scene needs its positions buffer');
            const make = (pipeline: GPUComputePipeline) =>
                this.device.createBindGroup({
                    label: 'trace scene',
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: scene.nodes } },
                        { binding: 1, resource: { buffer: scene.triangles } },
                        { binding: 2, resource: { buffer: scene.topLevel } },
                        ...(leaves === 'indexed' ? [{ binding: 3, resource: { buffer: scene.positions! } }] : []),
                    ],
                });
            groups = { closest: make(this.closestPipeline), any: make(this.anyPipeline) };
            this.sceneGroups.set(scene, groups);
        }
        return groups;
    }

    /** Records one dispatch into `encoder`. */
    encode(encoder: GPUCommandEncoder, scene: TraceSceneBuffers, batch: TraceBatch): void {
        if (batch.count <= 0) return;
        const pipeline = batch.mode === 'closest' ? this.closestPipeline : this.anyPipeline;
        // The ray count is the size of the bound range; the mask is the only uniform.
        const batchGroup = this.device.createBindGroup({
            label: `trace batch ${batch.mode}`,
            layout: pipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: batch.rays, offset: 0, size: batch.count * RAY_BYTES } },
                { binding: 1, resource: { buffer: batch.results } },
                ...(this.rayMasks ? [{ binding: 2, resource: { buffer: this.maskBuffer((batch.mask ?? RAY_MASK_ALL) >>> 0) } }] : []),
            ],
        });
        const workgroups = Math.ceil(batch.count / BATCH_WORKGROUP_SIZE);
        const x = Math.min(workgroups, this.device.limits.maxComputeWorkgroupsPerDimension);
        const y = Math.ceil(workgroups / x);
        const groups = this.groupsFor(scene);
        const pass = encoder.beginComputePass({
            label: `trace batch ${batch.mode}`,
            timestampWrites: batch.timestampWrites,
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, batch.mode === 'closest' ? groups.closest : groups.any);
        pass.setBindGroup(1, batchGroup);
        pass.dispatchWorkgroups(x, y);
        pass.end();
    }
}

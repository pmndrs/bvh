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
import { BATCH_PARAMS_BYTES, BATCH_PARAMS_WORD, BATCH_WORKGROUP_SIZE, batchKernelWgsl } from './wgsl.ts';
import { RAY_MASK_ALL } from './layout.ts';
import { RAY_BYTES } from './rays.ts';

/** The three storage buffers the tracer reads. */
export interface TraceSceneBuffers {
    readonly nodes: GPUBuffer;
    readonly triangles: GPUBuffer;
    readonly topLevel: GPUBuffer;
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
    const [nodes, triangles, topLevel] = buffers;
    return {
        nodes,
        triangles,
        topLevel,
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
    ) {}

    static async create(device: GPUDevice, options: TraceBatchKernelOptions = {}): Promise<TraceBatchKernel> {
        const { label = 'trace', triangleFilter, rayMasks = true, hitNormal = true } = options;
        const pipelineFor = (mode: 'closest' | 'any') =>
            device.createComputePipelineAsync({
                label: `${label} batch ${mode}`,
                layout: 'auto',
                compute: {
                    module: device.createShaderModule({ label: `${label} batch ${mode}`, code: batchKernelWgsl(mode, { triangleFilter, rayMasks, hitNormal }) }),
                    entryPoint: 'bvhBatch',
                },
            });
        const [closestPipeline, anyPipeline] = await Promise.all([pipelineFor('closest'), pipelineFor('any')]);
        return new TraceBatchKernel(device, closestPipeline, anyPipeline, rayMasks);
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
            const make = (pipeline: GPUComputePipeline) =>
                this.device.createBindGroup({
                    label: 'trace scene',
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: scene.nodes } },
                        { binding: 1, resource: { buffer: scene.triangles } },
                        { binding: 2, resource: { buffer: scene.topLevel } },
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

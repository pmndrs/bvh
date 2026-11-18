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
import { BATCH_PARAMS_BYTES, BATCH_PARAMS_WORD, BATCH_WORKGROUP_SIZE, batchKernelWgsl, closestPointKernelWgsl, hitRecordBytes, WALK_COUNTS_BYTES, type AnyHitOrder, type QuantizedDecode, type WideDecode, type WideStack, type WideWalk } from './wgsl.ts';
import { CLOSEST_POINT_RECORD_BYTES, POINT_QUERY_BYTES } from './query-wgsl.ts';
import { RAY_MASK_ALL, DEFAULT_NODE_FORMAT, type NodeFormat, type PositionLayout, type TriangleLeaves } from '../gpu-layout.ts';
import { RAY_BYTES } from './rays.ts';
/**
 * The three storage buffers the tracer reads, and an indexed scene's fourth.
 *
 * **Data says what it is.** The formats are required: options default to
 * quantized nodes, but buffers name their own format (`uploadTraceScene` and
 * `SceneTrees.traceBuffers` always do), and a kernel refuses buffers that
 * don't. A hand-built object must name `nodeFormat`, `triangleLeaves` and,
 * for indexed leaves, `positionLayout`.
 */
export interface TraceSceneBuffers {
    readonly nodes: GPUBuffer;
    readonly triangles: GPUBuffer;
    readonly topLevel: GPUBuffer;
    /** An indexed scene's positions (`TraceSceneData.positions`), bound fourth. Absent or null for records. */
    readonly positions?: GPUBuffer | null;
    /** The scene's triangle format. A kernel refuses a scene of the other format. */
    readonly triangleLeaves: TriangleLeaves;
    /** The position layout (read for indexed scenes only). */
    readonly positionLayout: PositionLayout;
    /** The scene's node format. A kernel refuses a scene of the other format. */
    readonly nodeFormat: NodeFormat;
    /** Bytes held on the GPU. */
    readonly byteLength: number;
    destroy(): void;
}

/**
 * The formats of buffers handed to a walk, checked: plain-JavaScript callers
 * can leave a field out that the types require, and a guessed format would
 * read the bytes wrongly. Throws a RangeError naming the missing field.
 */
export function traceSourceFormats(
    source: { readonly nodeFormat?: unknown; readonly triangleLeaves?: unknown; readonly positionLayout?: unknown },
    where: string,
): { nodeFormat: NodeFormat; triangleLeaves: TriangleLeaves; positionLayout: PositionLayout } {
    const { nodeFormat, triangleLeaves } = source;
    if (nodeFormat !== 'f32' && nodeFormat !== 'quantized' && nodeFormat !== 'bvh4' && nodeFormat !== 'cwbvh8') {
        throw new RangeError(`${where}: the buffers must name their nodeFormat ('f32', 'quantized', or the experimental 'bvh4' or 'cwbvh8'), got ${String(nodeFormat)}. uploadTraceScene and SceneTrees name it; data is never given a default`);
    }
    if (triangleLeaves !== 'records' && triangleLeaves !== 'indexed') {
        throw new RangeError(`${where}: the buffers must name their triangleLeaves ('records' or 'indexed'), got ${String(triangleLeaves)}`);
    }
    const positionLayout = source.positionLayout ?? (triangleLeaves === 'records' ? 'vec4' : undefined);
    if (positionLayout !== 'vec4' && positionLayout !== 'packed') {
        throw new RangeError(`${where}: indexed buffers must name their positionLayout ('vec4' or 'packed'), got ${String(source.positionLayout)}`);
    }
    return { nodeFormat, triangleLeaves, positionLayout };
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
        nodeFormat: scene.nodeFormat ?? 'f32', // assembly always sets it; absent in scene data means f32
        byteLength: scene.byteLength,
        destroy: () => buffers.forEach((buffer) => buffer.destroy()),
    };
}

/** The formats a kernel was compiled for. */
interface KernelFormats {
    readonly triangleLeaves: TriangleLeaves;
    readonly positionLayout: PositionLayout;
    readonly nodeFormat: NodeFormat;
}

/** The scene's bind group entries (group 0), after checking that its formats are the kernel's. */
function sceneEntries(scene: TraceSceneBuffers, kernel: KernelFormats, where: string): GPUBindGroupEntry[] {
    const formats = traceSourceFormats(scene, where);
    const leaves = formats.triangleLeaves;
    if (leaves !== kernel.triangleLeaves) throw new RangeError(`${where}: the kernel reads ${kernel.triangleLeaves} triangles, the scene holds ${leaves}`);
    if (leaves === 'indexed' && formats.positionLayout !== kernel.positionLayout) {
        throw new RangeError(`${where}: the kernel reads ${kernel.positionLayout} positions, the scene holds ${formats.positionLayout}`);
    }
    const nodes = formats.nodeFormat;
    if (nodes !== kernel.nodeFormat) throw new RangeError(`${where}: the kernel reads ${kernel.nodeFormat} nodes, the scene holds ${nodes}`);
    if (leaves === 'indexed' && !scene.positions) throw new RangeError(`${where}: an indexed scene needs its positions buffer`);
    return [
        { binding: 0, resource: { buffer: scene.nodes } },
        { binding: 1, resource: { buffer: scene.triangles } },
        { binding: 2, resource: { buffer: scene.topLevel } },
        ...(leaves === 'indexed' ? [{ binding: 3, resource: { buffer: scene.positions! } }] : []),
    ];
}

/** One small uniform per mask used (`BATCH_PARAMS_BYTES`): dispatches recorded into one command buffer may use different masks. */
class MaskUniforms {
    private readonly buffers = new Map<number, GPUBuffer>();

    constructor(
        private readonly device: GPUDevice,
        private readonly label: string,
    ) {}

    get(mask: number): GPUBuffer {
        let buffer = this.buffers.get(mask);
        if (!buffer) {
            buffer = this.device.createBuffer({ label: `${this.label} mask ${mask.toString(16)}`, size: BATCH_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            const words = new Uint32Array(BATCH_PARAMS_BYTES / 4);
            words[BATCH_PARAMS_WORD.rayMask] = mask;
            this.device.queue.writeBuffer(buffer, 0, words);
            this.buffers.set(mask, buffer);
        }
        return buffer;
    }

    destroy(): void {
        for (const buffer of this.buffers.values()) buffer.destroy();
        this.buffers.clear();
    }
}

/** Throws a RangeError when a batch's buffer holds fewer than `count` items of `itemBytes` (WebGPU would read or write past it silently, or fail validation far from the cause). */
function checkBatchBuffer(where: string, name: string, buffer: GPUBuffer, count: number, itemBytes: number): void {
    const needed = count * itemBytes;
    if (buffer.size < needed) throw new RangeError(`${where}: ${name} holds ${buffer.size} bytes, ${count} items need ${needed} (${itemBytes} each)`);
}

/** A 2D dispatch of `count` items in workgroups of `BATCH_WORKGROUP_SIZE`, so a large batch stays under the per-dimension limit. */
function dispatchSize(device: GPUDevice, count: number): [number, number] {
    const workgroups = Math.ceil(count / BATCH_WORKGROUP_SIZE);
    const x = Math.min(workgroups, device.limits.maxComputeWorkgroupsPerDimension);
    return [x, Math.ceil(workgroups / x)];
}

export interface TraceBatch {
    readonly mode: 'closest' | 'any';
    /** At least `count` rays of 32 bytes (`BvhRay`). */
    readonly rays: GPUBuffer;
    /** Closest: `count` records of `HIT_RECORD_BYTES` (`HIT_RECORD_COMPACT_BYTES` with `hitNormal: false`). Any: `count` u32, 1 = blocked. A kernel made with `counters: true`: `count` records of `WALK_COUNTS_BYTES` in either mode. */
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
    /** The scenes' node format, compiled in. Default `'quantized'` (`DEFAULT_NODE_FORMAT`), as scenes are by default; a kernel refuses a scene of the other format. */
    readonly nodeFormat?: NodeFormat;
    /** The any-hit walk's child order, compiled in (`AnyHitOrder`). Default `'far'` (best for rays from surfaces); `'larger'` for rays from outside the geometry or across open space. */
    readonly anyHitOrder?: AnyHitOrder;
    /** How the walk decodes quantized nodes, compiled in (`QuantizedDecode`). Default `'exact'`; `'folded'` is the measured lever (quantized nodes only). */
    readonly quantizedDecode?: QuantizedDecode;
    /**
     * Compile the counted walk (`TraceModuleOptions.counters`): every ray's
     * result is a `BvhWalkCounts` (`WALK_COUNTS_BYTES`, words in
     * `WALK_COUNTS_WORD`), the hit and the walk's work, in either mode. For
     * heat maps and tools; trace with a kernel without it for speed. Default false.
     */
    readonly counters?: boolean;
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
    private readonly masks: MaskUniforms;

    private constructor(
        private readonly device: GPUDevice,
        private readonly closestPipeline: GPUComputePipeline,
        private readonly anyPipeline: GPUComputePipeline,
        readonly rayMasks: boolean,
        readonly triangleLeaves: TriangleLeaves,
        readonly positionLayout: PositionLayout,
        readonly nodeFormat: NodeFormat,
        /** Closest-hit records are 40 bytes (true) or the compact 24. */
        readonly hitNormal: boolean,
        /** Results are `BvhWalkCounts` records (`counters: true`). */
        readonly counters: boolean = false,
    ) {
        this.masks = new MaskUniforms(device, 'trace batch');
    }

    static async create(device: GPUDevice, options: TraceBatchKernelOptions = {}): Promise<TraceBatchKernel> {
        const { label = 'trace', triangleFilter, rayMasks = true, hitNormal = true, triangleLeaves = 'records', positionLayout = 'vec4', nodeFormat = DEFAULT_NODE_FORMAT, anyHitOrder, quantizedDecode, counters = false, wideStack, wideWalk, wideDecode } = options;
        const pipelineFor = (mode: 'closest' | 'any') =>
            device.createComputePipelineAsync({
                label: `${label} batch ${mode}`,
                layout: 'auto',
                compute: {
                    module: device.createShaderModule({ label: `${label} batch ${mode}`, code: batchKernelWgsl(mode, { triangleFilter, rayMasks, hitNormal, triangleLeaves, positionLayout, nodeFormat, anyHitOrder, quantizedDecode, counters, wideStack, wideWalk, wideDecode }) }),
                    entryPoint: 'bvhBatch',
                },
            });
        const [closestPipeline, anyPipeline] = await Promise.all([pipelineFor('closest'), pipelineFor('any')]);
        return new TraceBatchKernel(device, closestPipeline, anyPipeline, rayMasks, triangleLeaves, positionLayout, nodeFormat, hitNormal, counters);
    }

    /** Frees the mask uniforms. The pipelines need no freeing. */
    destroy(): void {
        this.masks.destroy();
    }

    private groupsFor(scene: TraceSceneBuffers): SceneBindGroups {
        let groups = this.sceneGroups.get(scene);
        if (!groups) {
            const entries = sceneEntries(scene, this, 'TraceBatchKernel');
            const make = (pipeline: GPUComputePipeline) => this.device.createBindGroup({ label: 'trace scene', layout: pipeline.getBindGroupLayout(0), entries });
            groups = { closest: make(this.closestPipeline), any: make(this.anyPipeline) };
            this.sceneGroups.set(scene, groups);
        }
        return groups;
    }

    /** Records one dispatch into `encoder`. */
    encode(encoder: GPUCommandEncoder, scene: TraceSceneBuffers, batch: TraceBatch): void {
        if (batch.count <= 0) return;
        checkBatchBuffer('TraceBatchKernel', 'the rays buffer', batch.rays, batch.count, RAY_BYTES);
        checkBatchBuffer('TraceBatchKernel', 'the results buffer', batch.results, batch.count, this.counters ? WALK_COUNTS_BYTES : batch.mode === 'closest' ? hitRecordBytes(this.hitNormal) : 4);
        const pipeline = batch.mode === 'closest' ? this.closestPipeline : this.anyPipeline;
        // The ray count is the size of the bound range; the mask is the only uniform.
        const batchGroup = this.device.createBindGroup({
            label: `trace batch ${batch.mode}`,
            layout: pipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: batch.rays, offset: 0, size: batch.count * RAY_BYTES } },
                { binding: 1, resource: { buffer: batch.results } },
                ...(this.rayMasks ? [{ binding: 2, resource: { buffer: this.masks.get((batch.mask ?? RAY_MASK_ALL) >>> 0) } }] : []),
            ],
        });
        const [x, y] = dispatchSize(this.device, batch.count);
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

/** One batch of closest-point queries (`ClosestPointBatchKernel.encode`). */
export interface ClosestPointBatch {
    /** At least `count` queries of `POINT_QUERY_BYTES` (16): `vec4f(point.xyz, maxDistance)`, world space. */
    readonly queries: GPUBuffer;
    /** `count` records of `CLOSEST_POINT_RECORD_BYTES` (48), words in `CLOSEST_POINT_RECORD_WORD`. */
    readonly results: GPUBuffer;
    readonly count: number;
    /** The batch's instance mask (one for the whole batch). Default all ones. Ignored by a kernel made with `rayMasks: false`. */
    readonly mask?: number;
    /** Optional GPU timing for the dispatch's pass. */
    readonly timestampWrites?: GPUComputePassTimestampWrites;
}

/** Options of {@link ClosestPointBatchKernel.create}: the scenes' formats, compiled in, as for {@link TraceBatchKernel}. */
export interface ClosestPointBatchKernelOptions {
    readonly label?: string;
    /** Default true: each batch has an instance mask (`ClosestPointBatch.mask`). False compiles masks out. */
    readonly rayMasks?: boolean;
    /** The scenes' triangle format. Default `'records'`. */
    readonly triangleLeaves?: TriangleLeaves;
    /** Indexed only: `'vec4'` (the default) or `'packed'` positions. */
    readonly positionLayout?: PositionLayout;
    /** The scenes' node format. Default `'quantized'` (`DEFAULT_NODE_FORMAT`); a kernel refuses a scene of the other format. */
    readonly nodeFormat?: NodeFormat;
}

/**
 * Runs batches of closest-point queries over a scene (`bvhClosestPoint`, the
 * trace module with `queries: true`): points in, the nearest triangle point
 * of the whole scene out, per query. Group 0 is the scene, as for
 * {@link TraceBatchKernel}; group 1 the batch's queries and results (and its mask).
 */
export class ClosestPointBatchKernel {
    private readonly sceneGroups = new WeakMap<TraceSceneBuffers, GPUBindGroup>();
    private readonly masks: MaskUniforms;

    private constructor(
        private readonly device: GPUDevice,
        private readonly pipeline: GPUComputePipeline,
        readonly rayMasks: boolean,
        readonly triangleLeaves: TriangleLeaves,
        readonly positionLayout: PositionLayout,
        readonly nodeFormat: NodeFormat,
    ) {
        this.masks = new MaskUniforms(device, 'closest point batch');
    }

    static async create(device: GPUDevice, options: ClosestPointBatchKernelOptions = {}): Promise<ClosestPointBatchKernel> {
        const { label = 'closest point', rayMasks = true, triangleLeaves = 'records', positionLayout = 'vec4', nodeFormat = DEFAULT_NODE_FORMAT } = options;
        const pipeline = await device.createComputePipelineAsync({
            label: `${label} batch`,
            layout: 'auto',
            compute: {
                module: device.createShaderModule({ label: `${label} batch`, code: closestPointKernelWgsl({ rayMasks, triangleLeaves, positionLayout, nodeFormat }) }),
                entryPoint: 'bvhClosestPointBatch',
            },
        });
        return new ClosestPointBatchKernel(device, pipeline, rayMasks, triangleLeaves, positionLayout, nodeFormat);
    }

    /** Frees the mask uniforms. The pipeline needs no freeing. */
    destroy(): void {
        this.masks.destroy();
    }

    private groupFor(scene: TraceSceneBuffers): GPUBindGroup {
        let group = this.sceneGroups.get(scene);
        if (!group) {
            group = this.device.createBindGroup({ label: 'closest point scene', layout: this.pipeline.getBindGroupLayout(0), entries: sceneEntries(scene, this, 'ClosestPointBatchKernel') });
            this.sceneGroups.set(scene, group);
        }
        return group;
    }

    /** Records one dispatch into `encoder`. */
    encode(encoder: GPUCommandEncoder, scene: TraceSceneBuffers, batch: ClosestPointBatch): void {
        if (batch.count <= 0) return;
        checkBatchBuffer('ClosestPointBatchKernel', 'the queries buffer', batch.queries, batch.count, POINT_QUERY_BYTES);
        checkBatchBuffer('ClosestPointBatchKernel', 'the results buffer', batch.results, batch.count, CLOSEST_POINT_RECORD_BYTES);
        const batchGroup = this.device.createBindGroup({
            label: 'closest point batch',
            layout: this.pipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: batch.queries, offset: 0, size: batch.count * POINT_QUERY_BYTES } },
                { binding: 1, resource: { buffer: batch.results } },
                ...(this.rayMasks ? [{ binding: 2, resource: { buffer: this.masks.get((batch.mask ?? RAY_MASK_ALL) >>> 0) } }] : []),
            ],
        });
        const [x, y] = dispatchSize(this.device, batch.count);
        const sceneGroup = this.groupFor(scene);
        const pass = encoder.beginComputePass({ label: 'closest point batch', timestampWrites: batch.timestampWrites });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, sceneGroup);
        pass.setBindGroup(1, batchGroup);
        pass.dispatchWorkgroups(x, y);
        pass.end();
    }
}

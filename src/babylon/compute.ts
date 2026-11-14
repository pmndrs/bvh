/**
 * Our buffers in a Babylon `ComputeShader`.
 *
 * `ComputeShader.setStorageBuffer(name, buffer)` takes a Babylon `StorageBuffer`
 * or a `DataBuffer`, and on WebGPU a `DataBuffer` is a thin wrapper around a
 * `GPUBuffer` (`WebGPUDataBuffer`). So an external buffer (`SceneTrees.buffers`,
 * `uploadTraceScene`'s) is bound with no copy: {@link gpuBufferToDataBuffer}
 * wraps it, {@link bindTraceBuffers} sets the walk's three or four. That
 * constructor is exported by Babylon but marked internal; it and
 * `engine._device` are the two internals this subpath uses.
 *
 * Babylon owns nothing here: it never destroys a buffer made this way, so the
 * owner (`SceneTrees`, the upload) destroys it as before. When a `SceneTrees`
 * grows a buffer (`version` changes), call {@link bindTraceBuffers} again.
 */

import { ComputeShader, type IComputeShaderOptions } from '@babylonjs/core/Compute/computeShader.js';
import type { ComputeBindingMapping } from '@babylonjs/core/Engines/Extensions/engine.computeShader.js';
import type { AbstractEngine } from '@babylonjs/core/Engines/abstractEngine.js';
import type { DataBuffer } from '@babylonjs/core/Buffers/dataBuffer.js';
import { WebGPUDataBuffer } from '@babylonjs/core/Meshes/WebGPU/webgpuDataBuffer.js';
import { traceModuleWgsl, type TraceModuleOptions } from '../trace/wgsl.ts';

/** The names `traceModuleWgsl` gives its bindings, in binding order (the fourth is an indexed scene's positions). */
export const TRACE_BINDING_NAMES = ['bvhTraceNodes', 'bvhTraceTriangles', 'bvhTraceTopLevel', 'bvhTracePositions'] as const;

/** The buffers the walk reads: `SceneTrees.buffers`, `uploadTraceScene`'s result, or any object with these. */
export interface BabylonTraceBuffers {
    readonly nodes: GPUBuffer;
    readonly triangles: GPUBuffer;
    readonly topLevel: GPUBuffer;
    /** An indexed scene's positions, bound fourth. */
    readonly positions?: GPUBuffer | null;
}

const wrapped = new WeakMap<GPUBuffer, WebGPUDataBuffer>();

/**
 * A Babylon `DataBuffer` over `buffer`, for `ComputeShader.setStorageBuffer`
 * (and `setUniformBuffer`). The same wrapper comes back for the same buffer.
 * The buffer must belong to the engine's device and carry the usage the
 * binding needs (`STORAGE`, `UNIFORM`).
 */
export function gpuBufferToDataBuffer(buffer: GPUBuffer): DataBuffer {
    let data = wrapped.get(buffer);
    if (!data) wrapped.set(buffer, (data = new WebGPUDataBuffer(buffer, buffer.size)));
    return data;
}

/**
 * `bindingsMapping` entries for the walk's buffers, keyed by the names the
 * module declares: spread them into the shader's own mapping.
 *
 * ```ts
 * bindingsMapping: { ...traceBindingsMapping(), params: { group: 1, binding: 0 } }
 * ```
 */
export function traceBindingsMapping(options: Pick<TraceModuleOptions, 'traceSlot' | 'triangleLeaves'> = {}): ComputeBindingMapping {
    const slot = options.traceSlot ?? { group: 0, firstBinding: 0 };
    const count = options.triangleLeaves === 'indexed' ? 4 : 3;
    const mapping: ComputeBindingMapping = {};
    for (let i = 0; i < count; i++) mapping[TRACE_BINDING_NAMES[i]] = { group: slot.group, binding: slot.firstBinding + i };
    return mapping;
}

/**
 * Binds the walk's buffers to `shader` (`setStorageBuffer` on each name of
 * {@link TRACE_BINDING_NAMES}). Call it again after `SceneTrees.version`
 * changes. An indexed scene's `positions` is bound when present.
 */
export function bindTraceBuffers(shader: ComputeShader, buffers: BabylonTraceBuffers): void {
    shader.setStorageBuffer(TRACE_BINDING_NAMES[0], gpuBufferToDataBuffer(buffers.nodes));
    shader.setStorageBuffer(TRACE_BINDING_NAMES[1], gpuBufferToDataBuffer(buffers.triangles));
    shader.setStorageBuffer(TRACE_BINDING_NAMES[2], gpuBufferToDataBuffer(buffers.topLevel));
    if (buffers.positions) shader.setStorageBuffer(TRACE_BINDING_NAMES[3], gpuBufferToDataBuffer(buffers.positions));
}

export interface TraceComputeShaderOptions {
    /** The walk's compile options (`nodeFormat`, `triangleLeaves`, `rayMasks`, ...); they must match the buffers. */
    readonly trace?: TraceModuleOptions;
    /** The kernel's own resources (`name: { group, binding }`), beside the walk's. */
    readonly bindings?: ComputeBindingMapping;
    /** Babylon's other compute shader options (`entryPoint` defaults to `main`). */
    readonly shader?: Partial<Omit<IComputeShaderOptions, 'bindingsMapping'>>;
}

/**
 * A `ComputeShader` whose WGSL is `traceModuleWgsl(options.trace)` followed by
 * `kernel` (your declarations and an `@compute` entry point that calls
 * `bvhTraceClosest` and friends), with the walk's names mapped to
 * `traceSlot` (group 0, bindings 0-2 unless set). Bind the buffers with
 * {@link bindTraceBuffers}, your own resources with Babylon's setters, then
 * `dispatch`.
 */
export function createTraceComputeShader(name: string, engine: AbstractEngine, kernel: string, options: TraceComputeShaderOptions = {}): ComputeShader {
    const trace = options.trace ?? {};
    return new ComputeShader(
        name,
        engine,
        { computeSource: `${traceModuleWgsl(trace)}\n${kernel}` },
        { ...options.shader, bindingsMapping: { ...traceBindingsMapping(trace), ...options.bindings } },
    );
}


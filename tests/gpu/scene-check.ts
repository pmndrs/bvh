/**
 * Shared by the scene-tree pages (bvh-top-level-gpu, scene-trees-*): read a
 * scene's three buffers back from the GPU, trace rays through them on the GPU
 * (`TraceBatchKernel`) and on the CPU (the CPU walk over the read-back bytes),
 * and compare against a reference: the CPU walk over a scene assembled on the
 * CPU from the same meshes and instances.
 */

import { createCpuTracer, type CpuHit } from '../../src/trace/cpu-traversal.ts';
import { HIT_RECORD_BYTES, HIT_RECORD_WORD } from '../../src/trace/wgsl.ts';

/** The kernel's default 40-byte record (with material and normal). */
const HIT_RECORD_WORDS = HIT_RECORD_BYTES / 4;
import { packRays, RAY_BYTES, type TraceRay } from '../../src/trace/rays.ts';
import type { TraceSceneData } from '../../src/trace/scene-assembly.ts';
import type { TopLevelBvh } from '../../src/trace/top-level.ts';
import { TOP_HEADER_WORD, type NodeFormat, type PositionLayout, type TriangleLeaves } from '../../src/gpu-layout.ts';
import type { TraceBatchKernel, TraceSceneBuffers } from '../../src/trace/trace-gpu.ts';

export async function readBuffer(device: GPUDevice, source: GPUBuffer, byteLength = source.size): Promise<ArrayBuffer> {
    const size = Math.ceil(byteLength / 4) * 4;
    const readback = device.createBuffer({ label: 'scene check readback', size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'scene check readback' });
    encoder.copyBufferToBuffer(source, 0, readback, 0, size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();
    return copy;
}

/**
 * The buffers as a `TraceSceneData` the CPU walk can read (only `nodes`,
 * `triangles`, `topLevel.buffer` and, for an indexed scene, `positions` and
 * the format are used).
 */
export async function readScene(
    device: GPUDevice,
    buffers: { nodes: GPUBuffer; triangles: GPUBuffer; topLevel: GPUBuffer; positions?: GPUBuffer | null },
    /** The buffers' formats; pass them from the scene (`SceneTrees.nodeFormat`): absent node format means f32 here. */
    format: { triangleLeaves?: TriangleLeaves; positionLayout?: PositionLayout; nodeFormat?: NodeFormat } = {},
): Promise<TraceSceneData> {
    const [nodes, triangles, topLevelBytes, positions] = await Promise.all([
        readBuffer(device, buffers.nodes),
        readBuffer(device, buffers.triangles),
        readBuffer(device, buffers.topLevel),
        format.triangleLeaves === 'indexed' && buffers.positions ? readBuffer(device, buffers.positions) : Promise.resolve(new ArrayBuffer(0)),
    ]);
    const header = new Uint32Array(topLevelBytes, 0, 4);
    const topLevel: TopLevelBvh = {
        buffer: topLevelBytes,
        instanceCount: header[TOP_HEADER_WORD.instanceCount],
        nodeCount: header[TOP_HEADER_WORD.nodeCount],
        maxDepth: header[TOP_HEADER_WORD.maxDepth],
        skippedSingular: 0,
        recordOrder: new Uint32Array(0),
    };
    return {
        nodes,
        nodeFormat: format.nodeFormat ?? 'f32',
        triangles,
        triangleLeaves: format.triangleLeaves ?? 'records',
        positions,
        positionLayout: format.positionLayout ?? 'vec4',
        vertexCount: 0,
        topLevel,
        meshes: [],
        maxMeshDepth: 0,
        requiredStackSize: 0,
        byteLength: nodes.byteLength + triangles.byteLength + topLevelBytes.byteLength + positions.byteLength,
        mergedGroups: [],
    };
}

export interface GpuHits {
    closest(index: number): CpuHit | null;
    any(index: number): boolean;
}

/** Traces `rays` through the bound buffers on the GPU, both queries, with `mask`. */
export async function traceOnGpu(device: GPUDevice, kernel: TraceBatchKernel, scene: TraceSceneBuffers, rays: readonly TraceRay[], mask?: number): Promise<GpuHits> {
    const count = rays.length;
    const rayBuffer = device.createBuffer({ label: 'scene check rays', size: count * RAY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(rayBuffer, 0, packRays(rays) as Float32Array<ArrayBuffer>);
    const hits = device.createBuffer({ label: 'scene check hits', size: count * HIT_RECORD_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const blocked = device.createBuffer({ label: 'scene check blocked', size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const encoder = device.createCommandEncoder({ label: 'scene check trace' });
    kernel.encode(encoder, scene, { mode: 'closest', rays: rayBuffer, results: hits, count, mask });
    kernel.encode(encoder, scene, { mode: 'any', rays: rayBuffer, results: blocked, count, mask });
    device.queue.submit([encoder.finish()]);
    const [hitBytes, blockedBytes] = await Promise.all([readBuffer(device, hits), readBuffer(device, blocked)]);
    for (const buffer of [rayBuffer, hits, blocked]) buffer.destroy();
    const floats = new Float32Array(hitBytes);
    const words = new Uint32Array(hitBytes);
    const anyWords = new Uint32Array(blockedBytes);
    return {
        closest(index) {
            const base = index * HIT_RECORD_WORDS;
            const t = floats[base];
            if (!(t >= 0)) return null;
            const w = HIT_RECORD_WORD;
            return {
                t,
                instanceIndex: words[base + w.instance],
                triangleIndex: words[base + w.triangle],
                frontFace: words[base + w.frontFace] !== 0,
                u: floats[base + w.barycentrics],
                v: floats[base + w.barycentrics + 1],
                material: words[base + w.material],
                normal: [floats[base + w.normal], floats[base + w.normal + 1], floats[base + w.normal + 2]],
            };
        },
        any: (index) => anyWords[index] !== 0,
    };
}

/** Same hit: the same instance, triangle, side and material, `t` within rounding. An exact tie (same `t`) on another triangle also counts. */
export function sameHit(a: CpuHit | null, b: CpuHit | null): boolean {
    if (a === null || b === null) return a === b;
    if (a.material !== b.material && a.instanceIndex === b.instanceIndex) return false;
    // T_TOLERANCE, as the oracle: the GPU walk runs in f32, the CPU walk in doubles.
    const close = Math.abs(a.t - b.t) <= 1e-4 * Math.max(1, b.t);
    if (!close) return false;
    if (a.instanceIndex === b.instanceIndex && a.triangleIndex === b.triangleIndex) {
        const normalClose = a.normal.every((value, axis) => Math.abs(value - b.normal[axis]) <= 1e-3);
        return a.frontFace === b.frontFace && normalClose;
    }
    // Another triangle at the same distance: an exact tie, or a hit at a shared edge (the oracle's
    // "ambiguous": f32 walks may take either neighbour there, and both are right).
    const margin = (hit: CpuHit) => Math.min(hit.u, hit.v, 1 - hit.u - hit.v);
    return Math.abs(a.t - b.t) <= 1e-6 * Math.max(1, b.t) || Math.min(margin(a), margin(b)) < 1e-4;
}

export interface Disagreement {
    readonly ray: number;
    readonly what: string;
}

/**
 * Compares a GPU answer and the CPU walk over the GPU's own buffers with the
 * CPU walk over the reference scene. Returns the disagreements (at most `limit` listed) and counts.
 */
export function compareWithReference(
    rays: readonly TraceRay[],
    gpu: GpuHits,
    gpuScene: TraceSceneData,
    reference: TraceSceneData,
    options: { mask?: number; limit?: number } = {},
): { readonly disagreements: Disagreement[]; readonly count: number; readonly hits: number } {
    const onGpuBuffers = createCpuTracer(gpuScene);
    const onReference = createCpuTracer(reference);
    const disagreements: Disagreement[] = [];
    let count = 0;
    let hits = 0;
    const note = (ray: number, what: string) => {
        count++;
        if (disagreements.length < (options.limit ?? 8)) disagreements.push({ ray, what });
    };
    rays.forEach((ray, index) => {
        const expected = onReference.closest(ray, { mask: options.mask });
        if (expected) hits++;
        const fromGpu = gpu.closest(index);
        const fromBuffers = onGpuBuffers.closest(ray, { mask: options.mask });
        if (!sameHit(fromGpu, expected)) note(index, `GPU closest ${JSON.stringify(fromGpu)} vs reference ${JSON.stringify(expected)}`);
        else if (!sameHit(fromBuffers, expected)) note(index, `CPU walk over the GPU buffers ${JSON.stringify(fromBuffers)} vs reference ${JSON.stringify(expected)}`);
        const blocked = onReference.any(ray, { mask: options.mask });
        if (gpu.any(index) !== blocked) note(index, `GPU any ${gpu.any(index)} vs reference ${blocked}`);
    });
    return { disagreements, count, hits };
}

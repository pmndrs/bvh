/**
 * The GPU half of quantized nodes (docs/internals/design/compressed-nodes.md): one
 * compute pass that packs 64-byte `f32` nodes into a frame slot and 32-byte
 * quantized nodes, mesh by mesh. It is the GPU twin of `quantizeNodeWords`
 * (quantized-nodes.ts), bit for bit: the frame comes from each mesh's root by
 * the same rule, and every rounding test is on exact `x / step` values (each
 * thread finds its mesh's frame itself, so the pass needs no second dispatch).
 *
 * Used where `f32` nodes are on the GPU and quantized ones are wanted:
 *
 * - the GPU builders' `nodeFormat: 'quantized'` output (H-PLOC and LBVH build
 *   `f32` nodes, as their climbs need, then pack them);
 * - `SceneTrees` placing a GPU batch's `f32` trees into quantized storage;
 * - `GpuMeshRefit`'s re-quantize after a refit of its `f32` copy (the refit's
 *   climb hands boxes over through the node's own `f32` words, so a deforming
 *   mesh keeps a 64-byte copy, and this pass writes the scene's nodes).
 *
 * One thread per node, any number of meshes per dispatch: thread `g` finds its
 * mesh in a table by binary search. Each mesh entry is `vec4u(source node
 * offset, target root slot, node count, first thread)`; the frame goes to slot
 * `target root slot - 1`.
 */

import { GPU_LAYOUT_EMPTY_CHILD, GPU_LAYOUT_NODE_WORDS, GPU_LAYOUT_QUANTIZED_NODE_BYTES, GPU_LAYOUT_QUANTIZED_NODE_WORDS } from './gpu-layout.ts';
import type { GpuMeshTreeRange, GpuTreeBuffers } from './mesh-input.ts';
import { FRAME_EXACT_STEPS, FRAME_LIMIT, FRAME_MAX_SPAN, FRAME_MIN_EXPONENT, QUANTIZE_MARGIN_STEPS, QUANTIZED_EMPTY_AXIS } from './quantized-nodes.ts';
import { dispatchWorkgroups2d } from './gpu-workgroup-scan.ts';

const WORKGROUP_SIZE = 64;
const hex = (value: number): string => `0x${value.toString(16)}u`;
/** WGSL f32 literal of a power of two (exact in decimal with enough digits). */
const f32Literal = (value: number): string => `${value.toExponential(12)}`;

/**
 * The frame rule and the outward rounding in WGSL, shared by the pass below
 * and by any kernel that quantizes one node (`bvhq*`).
 */
export const QUANTIZE_WGSL = /* wgsl */ `
const BVHQ_EMPTY_CHILD: u32 = ${hex(GPU_LAYOUT_EMPTY_CHILD)};
const BVHQ_EMPTY_AXIS: u32 = ${hex(QUANTIZED_EMPTY_AXIS)};
const BVHQ_LIMIT: f32 = ${f32Literal(FRAME_LIMIT)};
const BVHQ_EXACT_STEPS: f32 = ${f32Literal(FRAME_EXACT_STEPS)};
const BVHQ_MAX_SPAN: f32 = ${FRAME_MAX_SPAN}.0;
const BVHQ_MIN_EXPONENT: i32 = ${FRAME_MIN_EXPONENT};
const BVHQ_MARGIN: f32 = ${QUANTIZE_MARGIN_STEPS};

// An axis with nothing a ray can reach: min above max, or out of the frame's range.
fn bvhqEmptyAxis(lo: f32, hi: f32) -> bool {
    return !(lo <= hi) || !(abs(lo) < BVHQ_LIMIT) || !(abs(hi) < BVHQ_LIMIT);
}

// floor(log2(x)) - 1 for a positive x (a lower bound is all the search needs).
fn bvhqExponentBelow(x: f32) -> i32 {
    return i32((bitcast<u32>(x) >> 23u) & 0xffu) - 128;
}

// One axis's frame: (exponent of the step, origin / step), as quantized-nodes.ts axisFrame.
struct BvhqAxis {
    exponent: i32,
    base: f32,
}

fn bvhqAxisFrame(lo: f32, hi: f32) -> BvhqAxis {
    if (bvhqEmptyAxis(lo, hi)) {
        return BvhqAxis(0, 0.0);
    }
    var exponent = BVHQ_MIN_EXPONENT;
    let largest = max(abs(lo), abs(hi));
    if (largest > 0.0) {
        exponent = max(exponent, bvhqExponentBelow(largest) - 24);
    }
    if (hi > lo) {
        exponent = max(exponent, bvhqExponentBelow(hi - lo) - 17);
    }
    // Monotone in the exponent: the first that works is the CPU's answer. Bounded for safety.
    for (var i = 0; i < 300; i += 1) {
        let step = ldexp(1.0, exponent);
        let exact = abs(lo) < BVHQ_EXACT_STEPS * step && abs(hi) < BVHQ_EXACT_STEPS * step;
        if (exact) {
            let low = floor(ldexp(lo, -exponent));
            if (ceil(ldexp(hi, -exponent)) - low <= BVHQ_MAX_SPAN) {
                return BvhqAxis(exponent, low - 1.0);
            }
        }
        exponent += 1;
    }
    return BvhqAxis(0, 0.0);
}

// One axis of a child box as an axis word, rounded outward with the margin.
fn bvhqAxisWord(lo: f32, hi: f32, frame: BvhqAxis) -> u32 {
    if (bvhqEmptyAxis(lo, hi)) {
        return BVHQ_EMPTY_AXIS;
    }
    let low = ldexp(lo, -frame.exponent);
    let high = ldexp(hi, -frame.exponent);
    var qMin = floor(low) - frame.base;
    if (low - floor(low) < BVHQ_MARGIN) {
        qMin -= 1.0;
    }
    var qMax = ceil(high) - frame.base;
    if (ceil(high) - high < BVHQ_MARGIN) {
        qMax += 1.0;
    }
    // Inside the frame by its rule. The clamp is only a guard: a child box outside the root's (which no builder or refit produces) is clamped here, where the CPU quantizer throws.
    return u32(clamp(qMin, 0.0, 65535.0)) | (u32(clamp(qMax, 0.0, 65535.0)) << 16u);
}
`;

const QUANTIZE_PASS_WGSL = /* wgsl */ `
${QUANTIZE_WGSL}

const BVHQ_SOURCE_NODE_WORDS: u32 = ${GPU_LAYOUT_NODE_WORDS}u;
const BVHQ_TARGET_NODE_WORDS: u32 = ${GPU_LAYOUT_QUANTIZED_NODE_WORDS}u;

struct BvhqParams {
    meshCount: u32,
    threadCount: u32,
    _a: u32,
    _b: u32,
}

@group(0) @binding(0) var<storage, read> bvhqSource: array<u32>;
@group(0) @binding(1) var<storage, read_write> bvhqTarget: array<u32>;
// Per mesh: source node offset, target root slot, node count, first thread.
@group(0) @binding(2) var<storage, read> bvhqMeshes: array<vec4u>;
@group(0) @binding(3) var<uniform> bvhqParams: BvhqParams;

fn bvhqChildBox(node: u32, side: u32) -> array<f32, 6> {
    let at = node * BVHQ_SOURCE_NODE_WORDS + side * 8u;
    return array<f32, 6>(
        bitcast<f32>(bvhqSource[at]), bitcast<f32>(bvhqSource[at + 1u]), bitcast<f32>(bvhqSource[at + 2u]),
        bitcast<f32>(bvhqSource[at + 4u]), bitcast<f32>(bvhqSource[at + 5u]), bitcast<f32>(bvhqSource[at + 6u]),
    );
}

// The index word of a node's child (left: word 3, right: word 7) and its count (11, 15).
fn bvhqChildIndex(node: u32, side: u32) -> u32 {
    return bvhqSource[node * BVHQ_SOURCE_NODE_WORDS + 3u + side * 4u];
}

fn bvhqChildCount(node: u32, side: u32) -> u32 {
    return bvhqSource[node * BVHQ_SOURCE_NODE_WORDS + 11u + side * 4u];
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn bvhqQuantize(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroupCount: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let thread = (workgroup.y * workgroupCount.x + workgroup.x) * ${WORKGROUP_SIZE}u + localIndex;
    if (thread >= bvhqParams.threadCount) {
        return;
    }
    // The last mesh whose first thread is at or before this one.
    var low = 0u;
    var high = bvhqParams.meshCount;
    while (high - low > 1u) {
        let middle = (low + high) / 2u;
        if (bvhqMeshes[middle].w <= thread) {
            low = middle;
        } else {
            high = middle;
        }
    }
    let mesh = bvhqMeshes[low];
    let local = thread - mesh.w;
    if (local >= mesh.z) {
        return;
    }
    let root = mesh.x;

    // The frame, from the union of the root's non-empty children (per axis).
    var lo = vec3f(3.0e38);
    var hi = vec3f(-3.0e38);
    for (var side = 0u; side < 2u; side += 1u) {
        if (bvhqChildIndex(root, side) == BVHQ_EMPTY_CHILD) {
            continue;
        }
        var box = bvhqChildBox(root, side);
        for (var axis = 0u; axis < 3u; axis += 1u) {
            if (!bvhqEmptyAxis(box[axis], box[axis + 3u])) {
                lo[axis] = min(lo[axis], box[axis]);
                hi[axis] = max(hi[axis], box[axis + 3u]);
            }
        }
    }
    var frame = array<BvhqAxis, 3>(bvhqAxisFrame(lo.x, hi.x), bvhqAxisFrame(lo.y, hi.y), bvhqAxisFrame(lo.z, hi.z));

    let frameAt = (mesh.y - 1u) * BVHQ_TARGET_NODE_WORDS;
    if (local == 0u) {
        for (var axis = 0u; axis < 3u; axis += 1u) {
            let step = ldexp(1.0, frame[axis].exponent);
            bvhqTarget[frameAt + axis] = bitcast<u32>(frame[axis].base * step);
            bvhqTarget[frameAt + 4u + axis] = bitcast<u32>(step);
        }
        bvhqTarget[frameAt + 3u] = 0u;
        bvhqTarget[frameAt + 7u] = 0u;
    }

    let node = root + local;
    let out = (mesh.y + local) * BVHQ_TARGET_NODE_WORDS;
    for (var side = 0u; side < 2u; side += 1u) {
        let at = out + side * 4u;
        let index = bvhqChildIndex(node, side);
        if (index == BVHQ_EMPTY_CHILD) {
            bvhqTarget[at] = BVHQ_EMPTY_AXIS;
            bvhqTarget[at + 1u] = BVHQ_EMPTY_AXIS;
            bvhqTarget[at + 2u] = BVHQ_EMPTY_AXIS;
            bvhqTarget[at + 3u] = BVHQ_EMPTY_CHILD;
            continue;
        }
        var box = bvhqChildBox(node, side);
        for (var axis = 0u; axis < 3u; axis += 1u) {
            bvhqTarget[at + axis] = bvhqAxisWord(box[axis], box[axis + 3u], frame[axis]);
        }
        bvhqTarget[at + 3u] = index | (bvhqChildCount(node, side) << 28u);
    }
}
`;

/** One mesh of a quantize job: where its `f32` nodes are and where its quantized ones go. */
export interface QuantizeMesh {
    /** The mesh's root in the source buffer, in 64-byte nodes. */
    readonly sourceNodeOffset: number;
    /** The mesh's root in the target buffer, in 32-byte slots; its frame goes to the slot before. */
    readonly targetRootSlot: number;
    readonly nodeCount: number;
}

/** A prepared pass over fixed buffers and meshes: dispatch it as often as needed, destroy it when done. */
export interface QuantizeJob {
    readonly threadCount: number;
    /** Records the pass into an open compute pass. */
    dispatch(pass: GPUComputePassEncoder): void;
    /** Records the pass in a compute pass of its own. */
    encode(encoder: GPUCommandEncoder): void;
    /** Frees the job's table and uniform (after the work that uses them is submitted). */
    destroy(): void;
}

/** The pipeline of the quantize pass, made once per device. */
export class GpuNodeQuantizer {
    private readonly device: GPUDevice;
    private readonly pipeline: GPUComputePipeline;

    // Plain fields, not parameter properties: Node's type stripping (the tools) can't run those.
    private constructor(device: GPUDevice, pipeline: GPUComputePipeline) {
        this.device = device;
        this.pipeline = pipeline;
    }

    static async create(device: GPUDevice): Promise<GpuNodeQuantizer> {
        const pipeline = await device.createComputePipelineAsync({
            label: 'bvh quantize nodes',
            layout: 'auto',
            compute: { module: device.createShaderModule({ label: 'bvh quantize nodes', code: QUANTIZE_PASS_WGSL }), entryPoint: 'bvhqQuantize' },
        });
        return new GpuNodeQuantizer(device, pipeline);
    }

    /** Synchronous creation, for callers that cannot wait (the pipeline compiles in the background). */
    static createSync(device: GPUDevice): GpuNodeQuantizer {
        const pipeline = device.createComputePipeline({
            label: 'bvh quantize nodes',
            layout: 'auto',
            compute: { module: device.createShaderModule({ label: 'bvh quantize nodes', code: QUANTIZE_PASS_WGSL }), entryPoint: 'bvhqQuantize' },
        });
        return new GpuNodeQuantizer(device, pipeline);
    }

    /**
     * A job packing `meshes` from `source` (64-byte nodes) into `target`
     * (32-byte slots). `source` needs STORAGE usage, `target` STORAGE too.
     */
    prepare(source: GPUBuffer, target: GPUBuffer, meshes: readonly QuantizeMesh[], label = 'bvh quantize'): QuantizeJob {
        const used = meshes.filter((mesh) => mesh.nodeCount > 0);
        const table = new Uint32Array(Math.max(1, used.length) * 4);
        let threadCount = 0;
        used.forEach((mesh, index) => {
            if (mesh.targetRootSlot < 1) throw new RangeError(`${label}: mesh ${index}'s root slot ${mesh.targetRootSlot} leaves no slot for its frame`);
            table.set([mesh.sourceNodeOffset, mesh.targetRootSlot, mesh.nodeCount, threadCount], index * 4);
            threadCount += mesh.nodeCount;
        });
        const device = this.device;
        const meshBuffer = device.createBuffer({ label: `${label} meshes`, size: table.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(meshBuffer, 0, table);
        const params = device.createBuffer({ label: `${label} params`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(params, 0, Uint32Array.of(used.length, threadCount, 0, 0));
        const bindGroup = device.createBindGroup({
            label,
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: source } },
                { binding: 1, resource: { buffer: target } },
                { binding: 2, resource: { buffer: meshBuffer } },
                { binding: 3, resource: { buffer: params } },
            ],
        });
        const pipeline = this.pipeline;
        const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;
        const dispatch = (pass: GPUComputePassEncoder): void => {
            if (threadCount === 0) return;
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            dispatchWorkgroups2d(pass, Math.ceil(threadCount / WORKGROUP_SIZE), maxPerDimension);
        };
        return {
            threadCount,
            dispatch,
            encode: (encoder) => {
                if (threadCount === 0) return;
                const pass = encoder.beginComputePass({ label });
                dispatch(pass);
                pass.end();
            },
            destroy: () => {
                meshBuffer.destroy();
                params.destroy();
            },
        };
    }
}

const quantizers = new WeakMap<GPUDevice, GpuNodeQuantizer>();

/** The device's quantize pipeline, made on first use (synchronously; it compiles in the background). */
export function nodeQuantizerFor(device: GPUDevice): GpuNodeQuantizer {
    let quantizer = quantizers.get(device);
    if (!quantizer) {
        quantizer = GpuNodeQuantizer.createSync(device);
        quantizers.set(device, quantizer);
    }
    return quantizer;
}

/**
 * The quantized node ranges for a batch's `f32` mesh ranges: each usable mesh
 * gets its frame slot and its nodes, packed in order; a mesh with a problem (or
 * no nodes) gets `nodeOffset: 0, nodeCount: 0`, so nothing reads past the
 * quantized buffer's end. Parents stay where they were (`parentOffset`).
 */
export function quantizedMeshRanges(source: readonly GpuMeshTreeRange[]): { meshes: GpuMeshTreeRange[]; jobs: QuantizeMesh[]; slots: number } {
    let slots = 0;
    const meshes: GpuMeshTreeRange[] = [];
    const jobs: QuantizeMesh[] = [];
    for (const mesh of source) {
        const parentOffset = mesh.parentOffset ?? mesh.nodeOffset;
        if (mesh.problem || mesh.nodeCount < 1) {
            meshes.push({ ...mesh, nodeOffset: 0, nodeCount: 0, parentOffset });
            continue;
        }
        const root = slots + 1;
        slots += mesh.nodeCount + 1;
        jobs.push({ sourceNodeOffset: mesh.nodeOffset, targetRootSlot: root, nodeCount: mesh.nodeCount });
        meshes.push({ ...mesh, nodeOffset: root, parentOffset });
    }
    return { meshes, jobs, slots };
}

/**
 * A GPU batch's `f32` trees with quantized nodes: a new node buffer holding,
 * per usable mesh, its frame slot and its nodes (packed by the quantize pass,
 * submitted here), and the `f32` node buffer destroyed. Ranges keep their
 * parents where they were (`parentOffset`); `nodeOffset` becomes the root's
 * slot. Meshes with a problem get no nodes (`nodeOffset` and `nodeCount` 0). Any GPU
 * builder's trees; quantized trees come back as they are.
 */
export function quantizeTreeBuffers(device: GPUDevice, trees: GpuTreeBuffers, label: string): GpuTreeBuffers {
    if (trees.nodeFormat === 'quantized') return trees;
    const { meshes, jobs, slots } = quantizedMeshRanges(trees.meshes);
    const nodes = device.createBuffer({
        label: `${label} quantized nodes`,
        size: Math.max(1, slots) * GPU_LAYOUT_QUANTIZED_NODE_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const job = nodeQuantizerFor(device).prepare(trees.nodes, nodes, jobs, `${label} quantize`);
    const encoder = device.createCommandEncoder({ label: `${label} quantize` });
    job.encode(encoder);
    device.queue.submit([encoder.finish()]);
    // Both are only freed once the submitted pass is done with them.
    job.destroy();
    trees.nodes.destroy();
    const { triangles, parents } = trees;
    return {
        ...trees,
        nodes,
        nodeFormat: 'quantized',
        meshes,
        destroy() {
            nodes.destroy();
            triangles.destroy();
            parents.destroy();
        },
    };
}

/**
 * Refit of one mesh's tree (a BLAS) from moved vertices, on the GPU (stage
 * B1): for deforming meshes, whose triangles move but keep their
 * connectivity. The tree keeps its shape; its triangles and boxes follow the
 * vertex buffer the adapter supplies (a skinned or morphed mesh's output, say).
 *
 * Two steps in the caller's compute pass:
 *
 * 1. `meshRefitTriangles`, one thread per triangle slot (tree order): reads
 *    the slot's original triangle number, fetches its three corners from the
 *    vertex buffer, writes them into the tree's triangle record (so the
 *    tracer's watertight test sees the moved triangle) and writes the slot's
 *    box for the refit. Only the corners' xyz words are written: the
 *    original index (word 3), the flags (word 7: alpha test and the like)
 *    and the padding stay as the builder, or whoever wrote them since, left
 *    them.
 * 2. The B1a refit (gpu-refit.ts) over the mesh's nodes: a bottom-up climb
 *    with a per-node counter, its subgroup pairing, and its repair pass.
 *
 * Any builder's tree works (the GPU LBVH's or tinybvh's): only the layout
 * matters. Never refit a merged static group's range (`mergeStaticInstances`):
 * its records are world space and name source-mesh triangles, so a refit from
 * a source's vertices writes wrong triangles. This class takes raw offsets and
 * can't tell; get a scene mesh's offsets from `meshRefitPlacement`, which
 * refuses merged meshes. The refit takes the tree's shape from a CPU copy of the layout
 * (`setTree`), which a GPU-built tree gets once with `GpuLbvhBuilder.readLayouts`;
 * the shape never changes while the mesh deforms, so that is a one-off at
 * build time, not a per-frame readback.
 *
 * **Quantized nodes** (docs/internals/design/compressed-nodes.md): the climb hands
 * boxes over through the node's own `f32` words, which 16-bit planes can't
 * take part in. So a mesh whose target holds quantized nodes keeps a 64-byte
 * `f32` copy of its nodes here (the refit runs on it unchanged), and a third
 * step in the same pass, the quantize pass (gpu-quantize.ts), packs the copy
 * into the target's frame slot and nodes. The frame follows the pose. Cost:
 * 64 bytes per node of the mesh, for deforming meshes only.
 *
 * **Wide nodes** (docs/internals/design/wide-nodes.md, section 2): the same
 * `f32` copy, of the binary tree the wide one was collapsed from, refit
 * unchanged; then the wide pack (gpu-wide-refit.ts) writes every wide node's
 * corner, exponents and planes from the boxes its refit map
 * ({@link WideRefitMap}) points at. CWBVH keeps its triangles in its leaves'
 * order, so the triangle step writes record `k`'s box where the binary tree
 * has that triangle (the map's `triangleOrder`). Cost: 64 bytes per binary
 * node, 16 or 32 per wide node, and 4 per triangle (CWBVH).
 *
 * A refit tree slowly gets worse as triangles move away from where it was
 * built. The refit's quality signal (`latestStats.relativeCost`, with
 * `diagnostics: true`) is the same as the top level's, so a caller can rebuild
 * (with a GPU builder) when it drifts. ddgi-core rebuilds only on drift held
 * for about a second (DECISIONS 2026-11-09).
 */

import { GPU_LAYOUT_TRIANGLE_BYTES, GPU_LAYOUT_TRIANGLE_WORD, type GpuLayoutMeshBvh } from './gpu-layout.ts';
import { LBVH_INDEX_FORMAT } from './gpu-lbvh-wgsl.ts';
import { GpuBvhRefit, GpuBvhRefitKernels, type BvhRefitStats, type GpuBvhRefitOptions } from './gpu-refit.ts';
import { leafSlotsOf, parentSlotsOf, PRIMITIVE_BOX_FLOATS } from './refit-cpu.ts';
import { dispatchWorkgroups2d, WORKGROUP_INDEX_WGSL } from './gpu-workgroup-scan.ts';
import { nodeQuantizerFor, type QuantizeJob } from './gpu-quantize.ts';
import { dequantizeNodes } from './quantized-nodes.ts';
import { GpuWideRefitKernels, WIDE_REFIT_PARAMS_BYTES, WIDE_REFIT_WORKGROUP } from './gpu-wide-refit.ts';
import { isWideNodeFormat, wideNodeBytes, wideRefitSlots, type WideNodeFormat, type WideRefitMap } from './wide-nodes.ts';

const WORKGROUP_SIZE = 64;
const PARAM_WORDS = 8;
const USAGE = { COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80 } as const;

/** WGSL of the triangle step. Plain WGSL: the refit after it needs subgroups, this does not. */
export function meshRefitTrianglesWgsl(): string {
    const triangle = GPU_LAYOUT_TRIANGLE_WORD;
    return /* wgsl */ `
struct MeshRefitParams {
    triangleCount: u32,
    triangleWordOffset: u32,   // where the mesh's first triangle record starts, in words
    vertexCount: u32,
    positionStride: u32,       // floats from one vertex to the next
    positionOffset: u32,       // float index of vertex 0's x
    indexFormat: u32,          // 0: a plain triangle list, 16 or 32: index bits
    indexOffset: u32,          // in indices, not bytes
    boxOrdered: u32,           // 1: slot k's box goes to refitBoxOrder[k] (a CWBVH tree's leaf order), 0: to k
}

@group(0) @binding(0) var<storage, read_write> refitTriangles: array<u32>;
@group(0) @binding(1) var<storage, read> refitPositions: array<f32>;
@group(0) @binding(2) var<storage, read> refitIndices: array<u32>;
// Two per triangle slot, in tree order: (min.xyz, 0), (max.xyz, 0). The refit reads them.
@group(0) @binding(3) var<storage, read_write> refitBoxes: array<vec4f>;
@group(0) @binding(4) var<uniform> refitParams: MeshRefitParams;
// The binary tree's slot of each record (wide trees whose records are in another order); a dummy when boxOrdered is 0.
@group(0) @binding(8) var<storage, read> refitBoxOrder: array<u32>;

const REFIT_WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE}u;
const REFIT_TRIANGLE_WORDS: u32 = ${GPU_LAYOUT_TRIANGLE_BYTES / 4}u;
const REFIT_CORNER_STRIDE: u32 = ${triangle.vertex1 - triangle.vertex0}u;
const REFIT_ORIGINAL_INDEX: u32 = ${triangle.originalIndex}u;
const REFIT_INDEX_UINT16: u32 = ${LBVH_INDEX_FORMAT.uint16}u;
const REFIT_INDEX_UINT32: u32 = ${LBVH_INDEX_FORMAT.uint32}u;
${WORKGROUP_INDEX_WGSL}

// Where record 'slot''s box goes: the binary tree's slot of that triangle.
fn refitBoxSlot(slot: u32) -> u32 {
    if (refitParams.boxOrdered != 0u) {
        return refitBoxOrder[slot];
    }
    return slot;
}

fn refitVertexOf(original: u32, corner: u32) -> u32 {
    let entry = refitParams.indexOffset + original * 3u + corner;
    var vertex = original * 3u + corner;
    if (refitParams.indexFormat == REFIT_INDEX_UINT32) {
        vertex = refitIndices[entry];
    } else if (refitParams.indexFormat == REFIT_INDEX_UINT16) {
        vertex = (refitIndices[entry >> 1u] >> ((entry & 1u) * 16u)) & 0xffffu;
    }
    // The indices were checked when the tree was built; this only keeps a bad buffer's reads in range.
    return min(vertex, max(refitParams.vertexCount, 1u) - 1u);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn meshRefitTriangles(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let slot = flatWorkgroupIndex(workgroup, groups) * REFIT_WORKGROUP_SIZE + localIndex;
    if (slot >= refitParams.triangleCount) {
        return;
    }
    let base = refitParams.triangleWordOffset + slot * REFIT_TRIANGLE_WORDS;
    let original = refitTriangles[base + REFIT_ORIGINAL_INDEX];
    var lower = vec3f(0x1.fffffep+127f);
    var upper = vec3f(-0x1.fffffep+127f);
    for (var corner = 0u; corner < 3u; corner += 1u) {
        let at = refitParams.positionOffset + refitVertexOf(original, corner) * refitParams.positionStride;
        let position = vec3f(refitPositions[at], refitPositions[at + 1u], refitPositions[at + 2u]);
        let word = base + corner * REFIT_CORNER_STRIDE;
        refitTriangles[word] = bitcast<u32>(position.x);
        refitTriangles[word + 1u] = bitcast<u32>(position.y);
        refitTriangles[word + 2u] = bitcast<u32>(position.z);
        lower = min(lower, position);
        upper = max(upper, position);
    }
    let boxSlot = refitBoxSlot(slot);
    refitBoxes[boxSlot * 2u] = vec4f(lower, 0.0);
    refitBoxes[boxSlot * 2u + 1u] = vec4f(upper, 0.0);
}
${INDEXED_REFIT_WGSL}`;
}

/**
 * The indexed refit (docs/internals/design/indexed-leaves.md, "Deforming meshes"): the
 * triangles keep their vertex indices, so nothing is written to them. Step 1
 * copies the posed vertices into the mesh's range of the scene's positions
 * buffer (a copy, not the app's buffer: rays must keep the pose the boxes were
 * built for when a refit is skipped); step 2 writes each slot's box from the
 * corners its indices name there. The mesh's vertices must not be welded:
 * vertex v of the posed buffer is scene vertex `vertexBase + v`.
 */
const INDEXED_REFIT_WGSL = /* wgsl */ `
struct MeshRefitIndexedParams {
    vertexBase: u32,           // the mesh's first vertex in the scene's positions
    sceneStride: u32,          // floats per scene vertex: 4 (vec4) or 3 (packed)
    unused0: u32,
    unused1: u32,
}

@group(0) @binding(5) var<storage, read_write> refitScenePositions: array<f32>;
@group(0) @binding(6) var<uniform> refitIndexed: MeshRefitIndexedParams;
@group(0) @binding(7) var<storage, read> refitIndexedTriangles: array<u32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn meshRefitCopyVertices(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let vertex = flatWorkgroupIndex(workgroup, groups) * REFIT_WORKGROUP_SIZE + localIndex;
    if (vertex >= refitParams.vertexCount) {
        return;
    }
    let source = refitParams.positionOffset + vertex * refitParams.positionStride;
    let to = (refitIndexed.vertexBase + vertex) * refitIndexed.sceneStride;
    refitScenePositions[to] = refitPositions[source];
    refitScenePositions[to + 1u] = refitPositions[source + 1u];
    refitScenePositions[to + 2u] = refitPositions[source + 2u];
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn meshRefitIndexedBoxes(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let slot = flatWorkgroupIndex(workgroup, groups) * REFIT_WORKGROUP_SIZE + localIndex;
    if (slot >= refitParams.triangleCount) {
        return;
    }
    let base = refitParams.triangleWordOffset + slot * 4u;
    var lower = vec3f(0x1.fffffep+127f);
    var upper = vec3f(-0x1.fffffep+127f);
    for (var corner = 0u; corner < 3u; corner += 1u) {
        let vertex = min(refitIndexedTriangles[base + corner], max(refitParams.vertexCount, 1u) - 1u);
        let at = (refitIndexed.vertexBase + vertex) * refitIndexed.sceneStride;
        let position = vec3f(refitScenePositions[at], refitScenePositions[at + 1u], refitScenePositions[at + 2u]);
        lower = min(lower, position);
        upper = max(upper, position);
    }
    let boxSlot = refitBoxSlot(slot);
    refitBoxes[boxSlot * 2u] = vec4f(lower, 0.0);
    refitBoxes[boxSlot * 2u + 1u] = vec4f(upper, 0.0);
}
`;

/** The moved vertices, on the GPU. Buffers need `STORAGE` usage. */
export interface MeshRefitVertices {
    readonly positionBuffer: GPUBuffer;
    readonly vertexCount: number;
    /** Floats from one vertex to the next. Default 3. */
    readonly positionStrideFloats?: number;
    /** Float index of vertex 0's x. Default 0. */
    readonly positionOffsetFloats?: number;
    /** The mesh's indices, as when the tree was built; omit for a plain triangle list. */
    readonly indexBuffer?: GPUBuffer | null;
    readonly indexFormat?: 'uint16' | 'uint32';
    readonly indexOffset?: number;
}

/** Where the mesh's tree lives on the GPU: its own buffers, or its place in a scene's shared ones. */
export interface MeshRefitTarget {
    /** Nodes, the mesh's root at `nodeWordOffset`. */
    readonly nodes: GPUBuffer;
    /** Triangle records, the mesh's first at `triangleWordOffset`. */
    readonly triangles: GPUBuffer;
    /** An indexed tree's positions buffer (the scene's), which the refit writes the posed vertices into. */
    readonly positions?: GPUBuffer | null;
}

/**
 * The mesh refit's pipelines (the triangle step and the B1a refit), compiled
 * once and shared by any number of {@link GpuMeshRefit}s on one device. A
 * caller with several deforming meshes needs one `GpuMeshRefit` per mesh
 * (each has its own parameters and topology, written with
 * `queue.writeBuffer`, so two meshes can't share one in a frame), and
 * sharing the kernels means a mesh added later compiles nothing.
 */
export class GpuMeshRefitKernels {
    readonly refit: GpuBvhRefitKernels;
    /** The wide pack, for trees placed with this wide format; null without one. */
    readonly wide: GpuWideRefitKernels | null;
    private readonly device: GPUDevice;
    private readonly label: string;
    private trianglePipeline: GPUComputePipeline | null = null;
    private copyPipeline: GPUComputePipeline | null = null;
    private indexedBoxPipeline: GPUComputePipeline | null = null;
    private compiling: Promise<void> | null = null;

    /** `wideFormat`: also compile the wide pack, for meshes whose trees sit in a scene of that format. */
    constructor(device: GPUDevice, label = 'mesh refit', wideFormat: WideNodeFormat | null = null) {
        this.device = device;
        this.label = label;
        this.refit = new GpuBvhRefitKernels(device, label);
        this.wide = wideFormat ? new GpuWideRefitKernels(device, wideFormat, `${label} wide`) : null;
    }

    prepare(): Promise<void> {
        this.compiling ??= (async () => {
            const module = this.device.createShaderModule({ label: this.label, code: meshRefitTrianglesWgsl() });
            const entry = (entryPoint: string) => this.device.createComputePipelineAsync({ label: `${this.label} ${entryPoint}`, layout: 'auto', compute: { module, entryPoint } });
            const [pipeline, copy, indexedBoxes] = await Promise.all([entry('meshRefitTriangles'), entry('meshRefitCopyVertices'), entry('meshRefitIndexedBoxes'), this.refit.prepare(), this.wide?.prepare()]);
            this.trianglePipeline = pipeline;
            this.copyPipeline = copy;
            this.indexedBoxPipeline = indexedBoxes;
        })();
        return this.compiling;
    }

    get ready(): boolean {
        return this.trianglePipeline !== null && this.refit.ready && (!this.wide || this.wide.pipeline !== null);
    }

    /** The triangle step's pipeline, or null before `prepare` has finished. */
    get triangles(): GPUComputePipeline | null {
        return this.trianglePipeline;
    }

    /** The indexed refit's two pipelines (copy the posed vertices, then the boxes), or null before `prepare` has finished. */
    get indexed(): { copy: GPUComputePipeline; boxes: GPUComputePipeline } | null {
        return this.copyPipeline && this.indexedBoxPipeline ? { copy: this.copyPipeline, boxes: this.indexedBoxPipeline } : null;
    }
}

/** Options of one mesh refit: {@link GpuBvhRefitOptions}, with the shared kernels of the mesh refit. */
export interface GpuMeshRefitOptions extends Omit<GpuBvhRefitOptions, 'kernels'> {
    /** Kernels shared with other mesh refits; without them this refit compiles its own in `prepare`. */
    readonly kernels?: GpuMeshRefitKernels;
    /** Without shared kernels: the wide format the trees will be placed in (its pack is compiled too). */
    readonly wideFormat?: WideNodeFormat;
}

/** Refits one mesh's tree from moved vertices, every frame if need be. Needs `subgroups` (for the refit). */
export class GpuMeshRefit {
    static isSupported(device: GPUDevice): boolean {
        return GpuBvhRefit.isSupported(device);
    }

    private readonly device: GPUDevice;
    private readonly label: string;
    private readonly refit: GpuBvhRefit;
    private readonly kernels: GpuMeshRefitKernels;
    private readonly params: GPUBuffer;
    private boxes: GPUBuffer;
    private triangleCount = 0;
    private triangleWordOffset = 0;
    private group: { group: GPUBindGroup; key: readonly unknown[] } | null = null;
    /** Indexed trees: the mesh's vertex base and the scene's floats per vertex; null for records. */
    private indexed: { vertexBase: number; vertexCount: number | null; sceneStride: number } | null = null;
    private readonly indexedParams: GPUBuffer;
    private indexedGroups: { copy: GPUBindGroup; boxes: GPUBindGroup; key: readonly unknown[] } | null = null;
    /** A quantized target: the f32 copy the refit runs on, the target's root slot, and the quantize job for the current target buffer. */
    private quantized: { shadow: GPUBuffer; rootSlot: number; nodeCount: number; job: QuantizeJob | null; jobTarget: GPUBuffer | null } | null = null;
    /** A wide target: the f32 copy of the binary tree, the refit map's slots, the pack's params, and its bind group for the current target buffer. */
    private wide: { format: WideNodeFormat; shadow: GPUBuffer; slots: GPUBuffer; params: GPUBuffer; nodeCount: number; group: GPUBindGroup | null; groupTarget: GPUBuffer | null } | null = null;
    /** The binary slot of each record (a CWBVH tree's), or a 16-byte dummy. */
    private order: GPUBuffer;
    private ordered = false;

    constructor(device: GPUDevice, options: GpuMeshRefitOptions = {}) {
        this.device = device;
        this.label = options.label ?? 'mesh refit';
        this.kernels = options.kernels ?? new GpuMeshRefitKernels(device, this.label, options.wideFormat ?? null);
        this.refit = new GpuBvhRefit(device, { label: this.label, diagnostics: options.diagnostics, kernels: this.kernels.refit });
        this.params = device.createBuffer({ label: `${this.label} params`, size: PARAM_WORDS * 4, usage: USAGE.UNIFORM | USAGE.COPY_DST });
        this.boxes = device.createBuffer({ label: `${this.label} boxes`, size: 64, usage: USAGE.STORAGE });
        this.indexedParams = device.createBuffer({ label: `${this.label} indexed params`, size: 16, usage: USAGE.UNIFORM | USAGE.COPY_DST });
        this.order = device.createBuffer({ label: `${this.label} box order`, size: 16, usage: USAGE.STORAGE | USAGE.COPY_DST });
    }

    /** Compiles the kernels, or waits for the shared ones. */
    prepare(): Promise<void> {
        return this.kernels.prepare();
    }

    get ready(): boolean {
        return this.kernels.ready;
    }

    /** The refit's stats with `diagnostics: true` (see gpu-refit.ts). */
    get latestStats(): BvhRefitStats | null {
        return this.refit.latestStats;
    }

    /**
     * Takes the tree's shape from a CPU copy of its layout, and where the mesh
     * sits in the GPU buffers (in words; 0 for a tree in its own buffers).
     */
    setTree(
        layout: GpuLayoutMeshBvh,
        placement: {
            nodeWordOffset?: number;
            triangleWordOffset?: number;
            generation?: number;
            /**
             * Indexed trees, required: the mesh's first vertex in the target's
             * positions buffer (0 for a tree in its own buffers;
             * `meshRefitPlacement` for a scene mesh).
             */
            vertexBase?: number;
            /** Indexed trees: the mesh's vertices there; `encode` then refuses a posed buffer of another size. */
            vertexCount?: number;
            /** Indexed trees: the positions buffer's layout. Default `'vec4'`. */
            positionLayout?: 'vec4' | 'packed';
            /**
             * The target's node format. Default: the layout's. With
             * `'quantized'`, `nodeWordOffset` is the root's (its frame slot is
             * the 8 words before; default 8, a tree in its own buffer), and
             * the refit runs on an `f32` copy then packs it into the target.
             * With `'bvh4'` or `'cwbvh8'`, `layout` is the binary tree the
             * wide one was collapsed from, `nodeWordOffset` the wide root's,
             * and `wide` is required; the refit runs on an `f32` copy, then
             * the wide pack writes the target's nodes.
             */
            nodeFormat?: 'f32' | 'quantized' | WideNodeFormat;
            /** Wide targets: the tree's refit map (`wideRefitMap`, or the GPU collapse's). */
            wide?: WideRefitMap;
        } = {},
    ): void {
        const wideFormat = isWideNodeFormat(placement.nodeFormat) ? placement.nodeFormat : null;
        if (wideFormat) {
            if (!placement.wide || placement.wide.nodeFormat !== wideFormat) throw new RangeError(`GpuMeshRefit.setTree: a ${wideFormat} target needs its refit map (placement.wide, from wideRefitMap or the GPU collapse)`);
            if (this.kernels.wide?.format !== wideFormat) throw new RangeError(`GpuMeshRefit.setTree: the kernels were made for ${this.kernels.wide?.format ?? 'no'} wide format, the target is ${wideFormat} (GpuMeshRefitKernels' wideFormat)`);
            if (placement.wide.triangleOrder && placement.wide.triangleOrder.length !== layout.triangleCount) throw new RangeError('GpuMeshRefit.setTree: the refit map\'s triangle order is not the layout\'s triangle count');
        }
        if (layout.triangleLeaves === 'indexed' && placement.vertexBase === undefined) {
            throw new RangeError('GpuMeshRefit.setTree: an indexed tree needs placement.vertexBase (0 for a tree in its own buffers; meshRefitPlacement gives a scene mesh\'s)');
        }
        this.indexed =
            layout.triangleLeaves === 'indexed'
                ? { vertexBase: placement.vertexBase!, vertexCount: placement.vertexCount ?? null, sceneStride: placement.positionLayout === 'packed' ? 3 : 4 }
                : null;
        if (this.indexed) this.device.queue.writeBuffer(this.indexedParams, 0, new Uint32Array([this.indexed.vertexBase, this.indexed.sceneStride, 0, 0]));
        const quantized = !wideFormat && (placement.nodeFormat ?? layout.nodeFormat ?? 'f32') === 'quantized';
        // The shape (child indices and counts) from f32 nodes; a quantized layout is decoded for it.
        const f32Nodes = dequantizeNodes(layout).nodes;
        const words = new Uint32Array(f32Nodes, 0, layout.nodeCount * 16);
        let nodeWordOffset = placement.nodeWordOffset ?? 0;
        this.quantized?.job?.destroy();
        if (quantized) {
            const rootWord = placement.nodeWordOffset ?? 8;
            if (rootWord < 8 || rootWord % 8 !== 0) throw new RangeError(`GpuMeshRefit.setTree: a quantized target's nodeWordOffset (${rootWord}) must be a 32-byte slot after the frame slot`);
            const bytes = layout.nodeCount * 64;
            let shadow = this.quantized?.shadow ?? null;
            if (!shadow || shadow.size < bytes) {
                shadow?.destroy();
                shadow = this.device.createBuffer({ label: `${this.label} f32 nodes`, size: Math.max(64, bytes), usage: USAGE.STORAGE | USAGE.COPY_DST });
            }
            // The refit rewrites every box; only the indices and counts must be right.
            this.device.queue.writeBuffer(shadow, 0, f32Nodes, 0, bytes);
            this.quantized = { shadow, rootSlot: rootWord / 8, nodeCount: layout.nodeCount, job: null, jobTarget: null };
            nodeWordOffset = 0;
        } else if (this.quantized) {
            this.quantized.shadow.destroy();
            this.quantized = null;
        }
        this.setWide(layout, wideFormat ? placement.wide! : null, f32Nodes, placement.nodeWordOffset ?? 0);
        if (wideFormat) nodeWordOffset = 0;
        this.refit.setShape({
            nodeWordOffset,
            nodeCount: layout.nodeCount,
            parentSlots: parentSlotsOf(words, 0, layout.nodeCount, layout.parents),
            leafSlots: leafSlotsOf(words, 0, layout.nodeCount),
            generation: placement.generation ?? 0,
        });
        this.triangleCount = layout.triangleCount;
        const boxBytes = Math.max(64, layout.triangleCount * PRIMITIVE_BOX_FLOATS * 4);
        if (this.boxes.size < boxBytes) {
            this.boxes.destroy();
            this.boxes = this.device.createBuffer({ label: `${this.label} boxes`, size: boxBytes, usage: USAGE.STORAGE });
        }
        this.triangleWordOffset = placement.triangleWordOffset ?? 0;
    }

    /** Records the triangle step and the refit into `pass`. The vertices must be written by an earlier command. */
    encode(pass: GPUComputePassEncoder, target: MeshRefitTarget, vertices: MeshRefitVertices): void {
        const pipeline = this.kernels.triangles;
        if (!pipeline) throw new Error('@pmndrs/bvh: call prepare() before encode()');
        if (this.triangleCount === 0) return;
        const indexFormat = vertices.indexBuffer ? LBVH_INDEX_FORMAT[vertices.indexFormat ?? 'uint32'] : LBVH_INDEX_FORMAT.none;
        this.device.queue.writeBuffer(
            this.params,
            0,
            // MeshRefitParams, field by field.
            new Uint32Array([
                this.triangleCount, // triangleCount
                this.triangleWordOffset, // triangleWordOffset
                vertices.vertexCount, // vertexCount
                vertices.positionStrideFloats ?? 3, // positionStride
                vertices.positionOffsetFloats ?? 0, // positionOffset
                indexFormat, // indexFormat
                vertices.indexOffset ?? 0, // indexOffset
                this.ordered ? 1 : 0, // boxOrdered
            ]),
        );
        const maxGroups = this.device.limits.maxComputeWorkgroupsPerDimension;
        if (this.indexed) {
            // Indexed: copy the posed vertices into the scene's positions, then the boxes from there; the triangles stay.
            const pipelines = this.kernels.indexed!;
            if (!target.positions) throw new Error('GpuMeshRefit: an indexed tree needs the target positions buffer');
            if (this.indexed.vertexCount !== null && vertices.vertexCount !== this.indexed.vertexCount) {
                throw new RangeError(`GpuMeshRefit: the posed buffer has ${vertices.vertexCount} vertices, the mesh has ${this.indexed.vertexCount} in the scene's positions`);
            }
            const groups = this.indexedBindGroups(pipelines, target.triangles, target.positions, vertices.positionBuffer);
            pass.setPipeline(pipelines.copy);
            pass.setBindGroup(0, groups.copy);
            dispatchWorkgroups2d(pass, Math.ceil(vertices.vertexCount / WORKGROUP_SIZE), maxGroups);
            pass.setPipeline(pipelines.boxes);
            pass.setBindGroup(0, groups.boxes);
            dispatchWorkgroups2d(pass, Math.ceil(this.triangleCount / WORKGROUP_SIZE), maxGroups);
        } else {
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, this.bindGroup(pipeline, target.triangles, vertices.positionBuffer, vertices.indexBuffer ?? vertices.positionBuffer));
            dispatchWorkgroups2d(pass, Math.ceil(this.triangleCount / WORKGROUP_SIZE), maxGroups);
        }
        if (this.quantized) {
            // Refit the f32 copy, then pack it into the target's frame slot and nodes (ordered after it in the pass).
            this.refit.encode(pass, this.quantized.shadow, this.boxes);
            this.quantizeJob(target.nodes).dispatch(pass);
        } else if (this.wide) {
            // Refit the binary f32 copy, then pack every wide node from it (ordered after it in the pass).
            this.refit.encode(pass, this.wide.shadow, this.boxes);
            this.encodeWidePack(pass, target.nodes);
        } else {
            this.refit.encode(pass, target.nodes, this.boxes);
        }
    }

    /**
     * Wide targets: packs the wide nodes from binary `f32` nodes refit on the
     * CPU (`refitLayoutToPose`; a CPU pose of a tree whose wide nodes exist
     * only on the GPU, as the GPU collapse's), without the triangle step or
     * the GPU refit. The nodes go into the `f32` copy with a queue write, so
     * record this pass in an encoder submitted after this call.
     */
    encodePackFrom(pass: GPUComputePassEncoder, target: Pick<MeshRefitTarget, 'nodes'>, binaryNodes: ArrayBuffer): void {
        if (!this.wide) throw new Error('GpuMeshRefit.encodePackFrom: the tree was not placed with a wide nodeFormat');
        this.device.queue.writeBuffer(this.wide.shadow, 0, binaryNodes, 0, Math.min(binaryNodes.byteLength, this.wide.shadow.size));
        this.encodeWidePack(pass, target.nodes);
    }

    private setWide(layout: GpuLayoutMeshBvh, map: WideRefitMap | null, f32Nodes: ArrayBuffer, wideWordOffset: number): void {
        const wide = this.wide;
        if (!map) {
            if (wide) {
                wide.shadow.destroy();
                wide.slots.destroy();
                wide.params.destroy();
                this.wide = null;
            }
            this.setOrder(null);
            return;
        }
        const M = wideRefitSlots(map.nodeFormat);
        if (wideWordOffset % (wideNodeBytes(map.nodeFormat) / 4) !== 0) throw new RangeError(`GpuMeshRefit.setTree: a ${map.nodeFormat} target's nodeWordOffset (${wideWordOffset}) must be a node's first word`);
        const shadowBytes = Math.max(64, layout.nodeCount * 64);
        const slotBytes = Math.max(16, map.nodeCount * M * 4);
        const grown = (buffer: GPUBuffer | undefined, size: number, name: string, usage: number): GPUBuffer => {
            if (buffer && buffer.size >= size) return buffer;
            buffer?.destroy();
            return this.device.createBuffer({ label: `${this.label} ${name}`, size, usage });
        };
        const shadow = grown(wide?.shadow, shadowBytes, 'binary f32 nodes', USAGE.STORAGE | USAGE.COPY_DST);
        const slots = grown(wide?.slots, slotBytes, 'refit map', USAGE.STORAGE | USAGE.COPY_DST);
        const params = wide?.params ?? this.device.createBuffer({ label: `${this.label} wide params`, size: WIDE_REFIT_PARAMS_BYTES, usage: USAGE.UNIFORM | USAGE.COPY_DST });
        // The refit rewrites every box; only the indices and counts must be right.
        this.device.queue.writeBuffer(shadow, 0, f32Nodes, 0, layout.nodeCount * 64);
        this.device.queue.writeBuffer(slots, 0, map.slots.buffer, map.slots.byteOffset, map.nodeCount * M * 4);
        this.device.queue.writeBuffer(params, 0, new Uint32Array([map.nodeCount, wideWordOffset, 0, 0]));
        this.wide = { format: map.nodeFormat, shadow, slots, params, nodeCount: map.nodeCount, group: null, groupTarget: null };
        this.setOrder(map.triangleOrder);
    }

    private setOrder(order: Uint32Array | null): void {
        this.ordered = order !== null;
        if (!order) return;
        const bytes = Math.max(16, order.byteLength);
        if (this.order.size < bytes) {
            this.order.destroy();
            this.order = this.device.createBuffer({ label: `${this.label} box order`, size: bytes, usage: USAGE.STORAGE | USAGE.COPY_DST });
            this.group = null;
            this.indexedGroups = null;
        }
        this.device.queue.writeBuffer(this.order, 0, order.buffer, order.byteOffset, order.byteLength);
    }

    private encodeWidePack(pass: GPUComputePassEncoder, nodes: GPUBuffer): void {
        const wide = this.wide!;
        const pipeline = this.kernels.wide?.pipeline;
        if (!pipeline) throw new Error('@pmndrs/bvh: call prepare() before encode()');
        if (!wide.group || wide.groupTarget !== nodes) {
            wide.group = this.device.createBindGroup({
                label: `${this.label} wide pack`,
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: wide.shadow } },
                    { binding: 1, resource: { buffer: wide.slots } },
                    { binding: 2, resource: { buffer: nodes } },
                    { binding: 3, resource: { buffer: wide.params } },
                ],
            });
            wide.groupTarget = nodes;
        }
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, wide.group);
        dispatchWorkgroups2d(pass, Math.ceil(wide.nodeCount / WIDE_REFIT_WORKGROUP), this.device.limits.maxComputeWorkgroupsPerDimension);
    }

    private quantizeJob(nodes: GPUBuffer): QuantizeJob {
        const quantized = this.quantized!;
        if (!quantized.job || quantized.jobTarget !== nodes) {
            quantized.job?.destroy();
            quantized.job = nodeQuantizerFor(this.device).prepare(quantized.shadow, nodes, [{ sourceNodeOffset: 0, targetRootSlot: quantized.rootSlot, nodeCount: quantized.nodeCount }], `${this.label} quantize`);
            quantized.jobTarget = nodes;
        }
        return quantized.job;
    }

    /** GPU bytes of this refit's own buffers (the tree belongs to the caller). */
    get memoryBytes(): number {
        const wide = this.wide ? this.wide.shadow.size + this.wide.slots.size + this.wide.params.size : 0;
        return this.refit.memoryBytes + this.params.size + this.boxes.size + this.order.size + (this.quantized?.shadow.size ?? 0) + wide;
    }

    /** Stats readback (diagnostics only), as for the top-level refit: poll at a frame's start, copy after the submit's encode. */
    pollStats(): void {
        this.refit.pollStats();
    }

    copyStats(encoder: GPUCommandEncoder): void {
        this.refit.copyStats(encoder);
    }

    dispose(): void {
        this.refit.dispose();
        this.params.destroy();
        this.indexedParams.destroy();
        this.boxes.destroy();
        this.quantized?.job?.destroy();
        this.quantized?.shadow.destroy();
        this.quantized = null;
        this.order.destroy();
        if (this.wide) {
            this.wide.shadow.destroy();
            this.wide.slots.destroy();
            this.wide.params.destroy();
            this.wide = null;
        }
    }

    private indexedBindGroups(
        pipelines: { copy: GPUComputePipeline; boxes: GPUComputePipeline },
        triangles: GPUBuffer,
        scenePositions: GPUBuffer,
        posed: GPUBuffer,
    ): { copy: GPUBindGroup; boxes: GPUBindGroup } {
        const key = [triangles, scenePositions, posed, this.boxes, this.order];
        if (this.indexedGroups && this.indexedGroups.key.every((entry, index) => entry === key[index])) return this.indexedGroups;
        const copy = this.device.createBindGroup({
            label: `${this.label} copy vertices`,
            layout: pipelines.copy.getBindGroupLayout(0),
            entries: [
                { binding: 1, resource: { buffer: posed } },
                { binding: 4, resource: { buffer: this.params } },
                { binding: 5, resource: { buffer: scenePositions } },
                { binding: 6, resource: { buffer: this.indexedParams } },
            ],
        });
        const boxes = this.device.createBindGroup({
            label: `${this.label} indexed boxes`,
            layout: pipelines.boxes.getBindGroupLayout(0),
            entries: [
                { binding: 3, resource: { buffer: this.boxes } },
                { binding: 4, resource: { buffer: this.params } },
                { binding: 5, resource: { buffer: scenePositions } },
                { binding: 6, resource: { buffer: this.indexedParams } },
                { binding: 7, resource: { buffer: triangles } },
                { binding: 8, resource: { buffer: this.order } },
            ],
        });
        this.indexedGroups = { copy, boxes, key };
        return this.indexedGroups;
    }

    private bindGroup(pipeline: GPUComputePipeline, triangles: GPUBuffer, positions: GPUBuffer, indices: GPUBuffer): GPUBindGroup {
        const key = [triangles, positions, indices, this.boxes, this.order];
        if (this.group && this.group.key.every((entry, index) => entry === key[index])) return this.group.group;
        const group = this.device.createBindGroup({
            label: this.label,
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: triangles } },
                { binding: 1, resource: { buffer: positions } },
                { binding: 2, resource: { buffer: indices } },
                { binding: 3, resource: { buffer: this.boxes } },
                { binding: 4, resource: { buffer: this.params } },
                { binding: 8, resource: { buffer: this.order } },
            ],
        });
        this.group = { group, key };
        return group;
    }
}

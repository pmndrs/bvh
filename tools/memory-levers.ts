/// <reference types="node" />
/**
 * The BX.20 levers that keep the GPU layout (docs/internals/design/memory.md,
 * "BX.20: the levers that keep the layout"), measured on a scene's meshes in
 * Node for tools/memory-breakdown.ts. Each lever against its default:
 *
 * - lever 4, idle workers: the tinybvh WASM memory a pool worker keeps, after
 *   building every mesh of the scene in turn (`idleTimeoutMs` frees it);
 * - lever 4, deforming meshes' retained layouts: each mesh's tinybvh tree
 *   (`f32`, as deforming meshes get it) as `SceneTrees` keeps it for CPU
 *   poses, whole or as `compactPoseLayout` gives it (`deformingLayouts`);
 * - lever 7, parents: the CPU parents arrays tinybvh's trees carry, and the
 *   H-PLOC batches' parents output (`hplocBatchBytes` with and without);
 * - refit box pooling: every mesh's refit boxes against one buffer sized for
 *   the largest (`refitBoxes: 'shared'`);
 * - with `sceneTrees`, a real `SceneTrees` on the fake GPU (tests/support), its
 *   "GPU" trees the H-PLOC CPU twin's laid out as a batch's outputs: the node
 *   buffer before and after `trim()` (copy with the default cap, and adopt
 *   uncapped), the quantize passes and their tables per tree and batched, and
 *   the peak with `parents: 'refit'`.
 */

import { createFakeGpu, installWebGpuGlobals, type FakeBuffer } from '../tests/support/fake-gpu.ts';
import { SCENE_KERNELS } from '../tests/support/scene-stand-ins.ts';
import { hplocBatchBytes, hplocScratchPoolBytes, planGpuHplocBatches, type GpuHplocBatch, type GpuHplocBatchOptions, type GpuHplocScratchPool } from '../src/gpu-hploc.ts';
import { geometryTotals, placeMeshes } from '../src/gpu-batch.ts';
import type { GpuLayoutMeshBvh } from '../src/gpu-layout.ts';
import { buildHplocOnCpu } from '../src/hploc-cpu.ts';
import type { GpuMeshGeometry, GpuMeshTreeRange, GpuTreeBuffers, MeshInput } from '../src/mesh-input.ts';
import { compactPoseLayout, PRIMITIVE_BOX_FLOATS } from '../src/refit-cpu.ts';
import { GpuFirstTrees, type HplocBatchBuilder } from '../src/scene/gpu-first-trees.ts';
import { DEFAULT_GPU_MAX_BATCH_BYTES, SceneTrees, type SceneTreesMemory, type SceneTreesOptions } from '../src/scene/scene-trees.ts';
import { createTinyBvhBuilder } from '../src/tinybvh.ts';
import { buildMeshBvhWithTinyBvh } from '../src/trace/mesh-bvh-topology.ts';
import type { Mat4 } from '../src/trace/top-level.ts';
import { defaultWorkerPoolSize } from '../src/worker/pool.ts';

const MIB = 1024 * 1024;
const mib = (bytes: number): string => (bytes / MIB).toFixed(1);
const sum = (values: Iterable<number>): number => {
    let total = 0;
    for (const value of values) total += value;
    return total;
};

type Geometry = MeshInput & { positions: Float32Array; indices: Uint32Array };

interface SceneInstance {
    readonly meshIndex: number;
    readonly objectToWorld: Mat4;
}

/**
 * The H-PLOC batch as `GpuFirstTrees` drives it, on the fake GPU: its outputs
 * are allocated as `hplocBatchBytes` says (one node per triangle less one per
 * mesh, parents only when asked), and hold the CPU twin's trees (the trees the
 * GPU builder writes) at the batch's offsets. Bytes in `memory` are the real
 * builder's: inputs and scratch from `hplocBatchBytes`, the pool from
 * `hplocScratchPoolBytes`.
 */
class TwinHploc implements HplocBatchBuilder {
    private readonly device: GPUDevice;
    private readonly twins: Map<GpuMeshGeometry, GpuLayoutMeshBvh>;

    constructor(device: GPUDevice, twins: Map<GpuMeshGeometry, GpuLayoutMeshBvh>) {
        this.device = device;
        this.twins = twins;
    }

    prepare(): Promise<void> {
        return Promise.resolve();
    }

    createScratchPool(batches: readonly (readonly GpuMeshGeometry[])[]): GpuHplocScratchPool {
        const sizes = hplocScratchPoolBytes(batches);
        const total = sum(Object.values<number>(sizes));
        let destroyed = false;
        const pool = {
            mortonCodes: 'cubic-30',
            sizes,
            get bytes() {
                return destroyed ? 0 : total;
            },
            get isDestroyed() {
                return destroyed;
            },
            destroy: () => {
                destroyed = true;
            },
        };
        return pool as unknown as GpuHplocScratchPool;
    }

    createBatch(geometries: readonly GpuMeshGeometry[], options: GpuHplocBatchOptions = {}, pool?: GpuHplocScratchPool): GpuHplocBatch {
        const parents = options.parents !== false;
        const bytes = hplocBatchBytes(geometryTotals(geometries), 'cubic-30', 'records', parents);
        const twin = (geometry: GpuMeshGeometry): GpuLayoutMeshBvh => {
            let tree = this.twins.get(geometry);
            if (!tree) {
                tree = buildHplocOnCpu(geometry as Geometry);
                this.twins.set(geometry, tree);
            }
            return tree;
        };
        const device = this.device;
        let destroyed = false;
        const batch = {
            meshCount: geometries.length,
            parents,
            outputBytes: bytes.outputs,
            get scratchBytes() {
                return pool || destroyed ? 0 : bytes.scratch;
            },
            encode: () => {},
            finish: async (): Promise<GpuTreeBuffers> => {
                const usage = 0x4 | 0x8 | 0x80;
                const nodes = device.createBuffer({ label: 'twin hploc nodes', size: bytes.buffers.outNodes, usage });
                const triangles = device.createBuffer({ label: 'twin hploc triangles', size: bytes.buffers.outTriangles, usage });
                const parentsBuffer = device.createBuffer({ label: 'twin hploc parents', size: bytes.buffers.outParents, usage });
                const meshes: GpuMeshTreeRange[] = placeMeshes(geometries).map((placed, index) => {
                    const tree = twin(geometries[index]);
                    device.queue.writeBuffer(nodes, placed.innerStart * 64, tree.nodes, 0, tree.nodeCount * 64);
                    device.queue.writeBuffer(triangles, placed.triangleStart * 48, tree.triangles, 0, tree.triangleCount * 48);
                    if (parents) device.queue.writeBuffer(parentsBuffer, placed.innerStart * 4, tree.parents.buffer as ArrayBuffer, tree.parents.byteOffset, tree.nodeCount * 4);
                    return { nodeOffset: placed.innerStart, nodeCount: tree.nodeCount, triangleOffset: placed.triangleStart, triangleCount: tree.triangleCount, leafCount: tree.leafCount, maxDepth: tree.maxDepth, bounds: tree.bounds, problem: null };
                });
                destroyed = true;
                return { nodes, triangles, parents: parentsBuffer, ...(parents ? {} : { withoutParents: true as const }), meshes, destroy: () => [nodes, triangles, parentsBuffer].forEach((buffer) => buffer.destroy()) };
            },
            destroy: () => {
                destroyed = true;
            },
        };
        return batch as unknown as GpuHplocBatch;
    }

    readLayouts(): Promise<never[]> {
        return Promise.resolve([]);
    }
}

interface SceneTreesRun {
    readonly settled: SceneTreesMemory;
    readonly trimmed: SceneTreesMemory;
    readonly gpuBatches: number;
    readonly quantizeDispatches: number;
    readonly quantizeTableBuffers: number;
    readonly quantizeBufferBytes: number;
}

/** Loads the scene into a `SceneTrees` on the fake GPU (every tinybvh tree swapped in), then trims it. */
async function runSceneTrees(geometries: readonly Geometry[], instances: readonly SceneInstance[], twins: Map<GpuMeshGeometry, GpuLayoutMeshBvh>, options: Partial<SceneTreesOptions>): Promise<SceneTreesRun> {
    let quantizeDispatches = 0;
    const kernels = {
        ...SCENE_KERNELS,
        bvhqQuantize: (...args: Parameters<(typeof SCENE_KERNELS)['bvhqQuantize']>) => {
            quantizeDispatches++;
            SCENE_KERNELS.bvhqQuantize(...args);
        },
    };
    const gpu = createFakeGpu({ maxStorageBufferBindingSize: 2 ** 31, maxBufferSize: 2 ** 31 }, { kernels });
    const scene = await SceneTrees.create(gpu.device, { mergeStaticInstances: false, ...options, firstTreeBuilder: new GpuFirstTrees(gpu.device, { hploc: new TwinHploc(gpu.device, twins) }) });
    const frame = () => {
        const encoder = gpu.createEncoder();
        scene.update(encoder);
        gpu.submit(encoder);
    };
    const meshes = geometries.map((geometry) => scene.addMesh(geometry));
    instances.forEach((instance, index) => scene.addInstance({ mesh: meshes[instance.meshIndex], objectToWorld: instance.objectToWorld, instanceIndex: index }));
    frame();
    await scene.whenBuilt();
    frame();
    await scene.whenRefined();
    frame();
    frame();
    const settled = scene.memory;
    const gpuBatches = scene.counts.gpuBatches;
    scene.trim();
    frame();
    frame();
    const trimmed = scene.memory;
    const tables = gpu.buffers.filter((buffer: FakeBuffer) => buffer.label.endsWith('quantize meshes') || buffer.label.endsWith('quantize params'));
    scene.dispose();
    return { settled, trimmed, gpuBatches, quantizeDispatches, quantizeTableBuffers: tables.length, quantizeBufferBytes: sum(tables.map((buffer) => buffer.size)) };
}

export async function measureLevers(geometries: readonly Geometry[], instances: readonly SceneInstance[], out: Record<string, unknown>, options: { readonly sceneTrees: boolean }): Promise<void> {
    installWebGpuGlobals();
    const levers: Record<string, unknown> = {};
    out.levers = levers;
    console.log('  BX.20 levers (docs/internals/design/memory.md):');

    // Lever 4, idle workers: a worker's tinybvh memory after building every mesh in turn (it never shrinks; recycleAboveBytes ends it past 256 MiB).
    {
        const builder = await createTinyBvhBuilder();
        let heap = 0;
        for (const geometry of geometries) {
            builder.buildLayout(geometry);
            heap = Math.max(heap, builder.lastBuildHeapBytes);
        }
        const workers = defaultWorkerPoolSize();
        levers.idleWorkers = { heapPerWorker: heap, poolSizeHere: workers, poolOfFour: 4 * heap };
        console.log(`    idleTimeoutMs: a worker keeps ${mib(heap)} MiB of WASM memory after every mesh of the scene; a pool of 4 holds ${mib(4 * heap)} MiB while idle (this machine's default pool: ${workers})`);
    }

    // Lever 4, deforming meshes: the layout SceneTrees keeps for CPU poses, if every mesh deformed (tinybvh's f32 trees).
    // Lever 7, CPU: the parents arrays tinybvh's trees carry (static meshes' could go).
    const tinyBvh = await createTinyBvhBuilder();
    let full = 0;
    let compact = 0;
    let parentsCpu = 0;
    let largest = { triangles: 0, full: 0, compact: 0 };
    let boxesAll = 0;
    let boxesLargest = 0;
    for (const geometry of geometries) {
        const tree = buildMeshBvhWithTinyBvh(geometry, tinyBvh);
        const wholeBytes = tree.nodes.byteLength + tree.triangles.byteLength + tree.parents.byteLength + tree.bounds.byteLength;
        const pose = compactPoseLayout(tree);
        const compactBytes = pose.nodes.byteLength + pose.parents.byteLength + pose.bounds.byteLength + pose.slotTriangles.byteLength + (pose.slotFlags?.byteLength ?? 0) + (pose.slotPadding?.byteLength ?? 0);
        full += wholeBytes;
        compact += compactBytes;
        parentsCpu += tree.parents.byteLength;
        if (tree.triangleCount > largest.triangles) largest = { triangles: tree.triangleCount, full: wholeBytes, compact: compactBytes };
        // GpuMeshRefit's box buffer: 8 floats per triangle, at least 64 bytes.
        const boxes = Math.max(64, tree.triangleCount * PRIMITIVE_BOX_FLOATS * 4);
        boxesAll += boxes;
        boxesLargest = Math.max(boxesLargest, boxes);
    }
    levers.deformingLayouts = { full, compact, saved: full - compact, largestMesh: largest };
    console.log(`    deformingLayouts: if every mesh deformed, 'full' keeps ${mib(full)} MiB on the CPU, 'compact' ${mib(compact)} MiB (saves ${mib(full - compact)}); the largest mesh ${mib(largest.full)} against ${mib(largest.compact)}`);
    levers.refitBoxes = { perMesh: boxesAll, shared: boxesLargest, saved: boxesAll - boxesLargest };
    console.log(`    refitBoxes: if every mesh deformed, 'per-mesh' box buffers ${mib(boxesAll)} MiB of GPU, 'shared' ${mib(boxesLargest)} (saves ${mib(boxesAll - boxesLargest)})`);

    // Lever 7, GPU: the H-PLOC parents output, one uncapped batch and the default cap's largest batch (the capped peak holds one batch).
    {
        const one = { with: hplocBatchBytes(geometryTotals(geometries)), without: hplocBatchBytes(geometryTotals(geometries), 'cubic-30', 'records', false) };
        const device = { limits: { maxStorageBufferBindingSize: 2 ** 31, maxBufferSize: 2 ** 31 } } as unknown as GPUDevice;
        const plan = planGpuHplocBatches(device, geometries, { maxBatchBytes: DEFAULT_GPU_MAX_BATCH_BYTES });
        const batches = plan.batches.map((indices) => geometryTotals(indices.map((index) => geometries[index])));
        const largestBatch = batches.reduce((best, totals) => (totals.triangles > best.triangles ? totals : best), batches[0]);
        const capped = { with: hplocBatchBytes(largestBatch), without: hplocBatchBytes(largestBatch, 'cubic-30', 'records', false) };
        levers.parents = {
            cpuParentsArrays: parentsCpu,
            uncappedBatch: { with: one.with.total, without: one.without.total, saved: one.with.total - one.without.total },
            defaultCapLargestBatch: { batches: plan.batches.length, with: capped.with.total, without: capped.without.total, saved: capped.with.total - capped.without.total },
        };
        console.log(
            `    parents: 'refit': tinybvh's parents arrays ${mib(parentsCpu)} MiB (CPU, until the upload); H-PLOC parents output ${mib(one.with.total - one.without.total)} MiB of the uncapped batch's ${mib(one.with.total)}, ` +
                `${mib(capped.with.total - capped.without.total)} MiB of the default cap's largest batch (${plan.batches.length} batches)`,
        );
    }

    if (!options.sceneTrees) return;
    // SceneTrees on the fake GPU, its GPU trees the H-PLOC twin's (built once, shared by every run): node trim, quantize jobs, parents in the peak.
    const twins = new Map<GpuMeshGeometry, GpuLayoutMeshBvh>();
    const runs: Record<string, SceneTreesRun> = {};
    for (const [name, runOptions] of [
        ['copy, quantized, default cap', {}],
        ['copy, quantized, default cap, quantizeJobs batched', { quantizeJobs: 'batched' as const }],
        ['copy, quantized, default cap, parents refit', { parents: 'refit' as const }],
        ['copy, quantized, uncapped', { gpuMaxBatchBytes: Infinity }],
        ['copy, quantized, uncapped, parents refit', { gpuMaxBatchBytes: Infinity, parents: 'refit' as const }],
        ['adopt, f32, uncapped', { gpuTreeOutputs: 'adopt' as const, nodeFormat: 'f32' as const, gpuMaxBatchBytes: Infinity }],
        ['adopt, f32, uncapped, parents refit', { gpuTreeOutputs: 'adopt' as const, nodeFormat: 'f32' as const, gpuMaxBatchBytes: Infinity, parents: 'refit' as const }],
    ] as const) {
        const start = performance.now();
        const run = await runSceneTrees(geometries, instances, twins, runOptions);
        runs[name] = run;
        console.log(
            `    SceneTrees ${name}: settled nodes ${mib(run.settled.nodes)} MiB (${mib(run.settled.nodesInUse)} in use), trimmed ${mib(run.trimmed.nodes)} MiB; settled total ${mib(run.settled.total)}, trimmed ${mib(run.trimmed.resident)}; ` +
                `peak ${mib(run.settled.peak)} MiB; ${run.gpuBatches} GPU batches, ${run.quantizeDispatches} quantize dispatches, ${run.quantizeTableBuffers} quantize buffers (${run.quantizeBufferBytes} bytes); ${Math.round((performance.now() - start) / 1000)} s`,
        );
    }
    levers.sceneTrees = Object.fromEntries(
        Object.entries(runs).map(([name, run]) => [
            name,
            {
                nodesSettled: run.settled.nodes,
                nodesInUse: run.settled.nodesInUse,
                nodesTrimmed: run.trimmed.nodes,
                residentSettled: run.settled.resident,
                residentTrimmed: run.trimmed.resident,
                peak: run.settled.peak,
                gpuBatches: run.gpuBatches,
                quantizeDispatches: run.quantizeDispatches,
                quantizeTableBuffers: run.quantizeTableBuffers,
                quantizeBufferBytes: run.quantizeBufferBytes,
            },
        ]),
    );
}

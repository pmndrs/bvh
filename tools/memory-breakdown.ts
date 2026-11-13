/// <reference types="node" />
/**
 * Memory breakdown of our trees against three-mesh-bvh's on real scenes, in Node
 * with no GPU (BX.19; the write-up is docs/internals/design/memory.md).
 *
 *   BVH_ASSETS_DIR=... BVH_MANIFESTS_DIR=... node tools/memory-breakdown.ts sponza
 *   node tools/memory-breakdown.ts bistro-exterior --json out.json
 *   node tools/memory-breakdown.ts sponza --no-twins --no-tmbvh   (skip the slow parts)
 *
 * What it reports, per scene (all sizes are exact byte counts of the buffers or
 * typed arrays; "MiB" is 2^20 bytes, as SceneTrees.memory and the HANDOFF use):
 *
 * - the geometry the app already holds (positions, 32-bit indices as loaded);
 * - ours, from `buildTraceScene` over tinybvh trees (the tracer's default
 *   build: the very scene SceneTrees settles on): nodes, triangle records,
 *   (and the same with indexed leaves, with quantized nodes, and with both),
 *   parents (a CPU array that is never uploaded), the top level; the same
 *   scene with `mergeStaticInstances`; tinybvh trees at leaves of 1, 2, 4, 8;
 * - the CPU twins' trees (H-PLOC, LBVH): used node counts, against the one node
 *   per triangle the GPU builders allocate for their outputs;
 * - the GPU build batches (H-PLOC, LBVH): inputs, scratch and outputs per batch,
 *   from the builders' allocation code (`gpu-hploc.ts`, `gpu-lbvh.ts`), copied
 *   formula for formula over the real `placeMeshes`/`buildTables`;
 * - the CPU copy `GpuMeshRefit.setTree` needs, and the refit's GPU buffers;
 * - the worker pool: input copies and the WASM heap of one worker;
 * - three-mesh-bvh 0.9.18: node buffers (`_roots`), index, `estimateMemoryInBytes`,
 *   for CENTER leaf 10 (its defaults), SAH leaf 4 and 10, and indirect.
 */

import { writeFileSync } from 'node:fs';
import { BufferAttribute, BufferGeometry } from 'three';
import { CENTER, estimateMemoryInBytes, MeshBVH, SAH } from 'three-mesh-bvh';
import { GPU_LAYOUT_NODE_BYTES, GPU_LAYOUT_TRIANGLE_BYTES } from '../src/gpu-layout.ts';
import { LBVH_SCRATCH_NODE_WORDS, LBVH_STATE_WORDS, LBVH_NODE_INFO_WORDS, LBVH_STATS_BYTES } from '../src/gpu-lbvh-wgsl.ts';
import { HPLOC_CLIMB_WORDS, HPLOC_RECORD_WORDS } from '../src/gpu-hploc-wgsl.ts';
import { align4, buildTables, indexBytes, placeMeshes, planMeshBatches, positionBytes, UNIFORM_STRIDE } from '../src/gpu-lbvh.ts';
import { radixPassCount } from '../src/gpu-radix-sort.ts';
import { scanTileSumWords } from '../src/gpu-scan.ts';
import { PRIMITIVE_BOX_FLOATS } from '../src/refit-cpu.ts';
import { createTinyBvhBuilder } from '../src/tinybvh.ts';
import type { MeshInput } from '../src/mesh-input.ts';
import { buildHplocOnCpu } from '../src/hploc-cpu.ts';
import { buildLbvhOnCpu } from '../src/lbvh-cpu.ts';
import { buildMeshBvhWithTinyBvh } from '../src/trace/mesh-bvh-topology.ts';
import { buildTraceScene } from '../src/trace/scene-assembly.ts';
import type { MeshBvh } from '../src/trace/mesh-bvh.ts';
import { readSceneInstances } from './scene-meshes.ts';

const MIB = 1024 * 1024;
const mib = (bytes: number): string => (bytes / MIB).toFixed(1);
const sum = (values: Iterable<number>): number => {
    let total = 0;
    for (const value of values) total += value;
    return total;
};

type Geometry = MeshInput & { positions: Float32Array; indices: Uint32Array };

/** A GPU batch's buffers, by role. All in bytes, each as the builder's `make` rounds it (4-byte aligned, at least 16). */
interface BatchBytes {
    readonly triangles: number;
    readonly inputs: number;
    readonly scratch: number;
    readonly outputs: number;
    readonly parts: Record<string, number>;
}

const make = (bytes: number): number => Math.max(16, align4(bytes));

/** `GpuHplocBatch`'s constructor (src/gpu-hploc.ts "createBuffer"), formula for formula. Default `cubic-30` codes. */
function hplocBatch(geometries: readonly Geometry[], codes: 'cubic-30' | 'tinybvh-63' = 'cubic-30', triangleBytes = GPU_LAYOUT_TRIANGLE_BYTES): BatchBytes {
    const meshes = placeMeshes(geometries);
    const last = meshes[meshes.length - 1];
    const T = last.triangleStart + last.triangleCount;
    const I = last.innerStart + last.innerCount;
    const { tables, sortTable } = buildTables(meshes);
    const lastPositionEnd = last.positionByteStart + align4(positionBytes(last.geometry));
    const lastIndexEnd = meshes.reduce((end, mesh) => Math.max(end, mesh.indexByteStart + align4(indexBytes(mesh))), 4);
    const lastFlagEnd = meshes.reduce((end, mesh) => Math.max(end, mesh.flagsWordStart + (mesh.hasFlags ? mesh.triangleCount : 0)), 1);
    const scanWords = T + 1;
    const leafRunsOffset = Math.ceil(scanWords / 64) * 64;
    const childrenOffset = leafRunsOffset + scanWords;
    const parentsOffset = childrenOffset + T * 2;
    const treeOrderOffset = parentsOffset + T * 2;
    const listsOffset = I * HPLOC_CLIMB_WORDS + T * HPLOC_RECORD_WORDS;
    const scratchQuads = Math.ceil((listsOffset + T) / 4);
    const sortBits = codes === 'cubic-30' ? [30] : [32, 31];
    const records = 1 + sortBits.reduce((total, bits) => total + radixPassCount(bits) + 1, 0) + 2 + 1;
    const parts: Record<string, number> = {
        'positions (input)': make(lastPositionEnd),
        'indices (input)': make(lastIndexEnd),
        'triangle flags (input)': make(lastFlagEnd * 4),
        stats: make(meshes.length * LBVH_STATS_BYTES + 16),
        'output nodes': make(I * GPU_LAYOUT_NODE_BYTES),
        'output triangles': make(T * triangleBytes),
        'output parents': make(I * 4),
        tables: make(tables.byteLength),
        'primitive boxes': make(T * 32),
        'keys a': make(T * 4),
        'values a': make(T * 4),
        'keys b': make(T * 4),
        'values b': make(T * 4),
        'sort offsets': make(sortTable.offsetWords * 4),
        'scan tile sums': make(Math.max(scanTileSumWords(sortTable.offsetWords), scanTileSumWords(T + 1)) * 4),
        codes: make(codes === 'tinybvh-63' ? T * 8 : 16),
        deltas: make(T * 8),
        'climb scratch': make(scratchQuads * 16),
        links: make((treeOrderOffset + T) * 4),
        'uniform params': make(records * UNIFORM_STRIDE),
    };
    return classify(T, parts);
}

/** `GpuLbvhBatch`'s constructor (src/gpu-lbvh.ts), formula for formula. */
function lbvhBatch(geometries: readonly Geometry[]): BatchBytes {
    const meshes = placeMeshes(geometries);
    const last = meshes[meshes.length - 1];
    const T = last.triangleStart + last.triangleCount;
    const I = last.innerStart + last.innerCount;
    const { tables, sortTable } = buildTables(meshes);
    const lastPositionEnd = last.positionByteStart + align4(positionBytes(last.geometry));
    const lastIndexEnd = meshes.reduce((end, mesh) => Math.max(end, mesh.indexByteStart + align4(indexBytes(mesh))), 4);
    const lastFlagEnd = meshes.reduce((end, mesh) => Math.max(end, mesh.flagsWordStart + (mesh.hasFlags ? mesh.triangleCount : 0)), 1);
    const rankScanRecord = 1 + radixPassCount(30) + 1;
    const parts: Record<string, number> = {
        'positions (input)': make(lastPositionEnd),
        'indices (input)': make(lastIndexEnd),
        'triangle flags (input)': make(lastFlagEnd * 4),
        stats: make(meshes.length * LBVH_STATS_BYTES + 16),
        'output nodes': make(I * GPU_LAYOUT_NODE_BYTES),
        'output triangles': make(T * GPU_LAYOUT_TRIANGLE_BYTES),
        'output parents': make(I * 4),
        'climb second round': 16,
        tables: make(tables.byteLength),
        'primitive boxes': make(T * 32),
        'keys a': make(T * 4),
        'values a': make(T * 4),
        'keys b': make((T + 1) * 4),
        'values b': make(T * 4),
        'sort offsets': make(sortTable.offsetWords * 4),
        'scan tile sums': make(Math.max(scanTileSumWords(sortTable.offsetWords), scanTileSumWords(I + 1)) * 4),
        'inner-node scratch': make(I * (LBVH_SCRATCH_NODE_WORDS + LBVH_STATE_WORDS) * 4),
        'inner-node info': make(I * LBVH_NODE_INFO_WORDS * 4),
        'inner-node rank': make((I + 1) * 4),
        'uniform params': make((rankScanRecord + 1) * UNIFORM_STRIDE),
    };
    return classify(T, parts);
}

function classify(triangles: number, parts: Record<string, number>): BatchBytes {
    let inputs = 0;
    let outputs = 0;
    let scratch = 0;
    for (const [name, bytes] of Object.entries(parts)) {
        if (name.endsWith('(input)') || name === 'stats') inputs += bytes;
        else if (name.startsWith('output ')) outputs += bytes;
        else scratch += bytes;
    }
    return { triangles, inputs, scratch, outputs, parts };
}

/** The planner's batches (src/gpu-lbvh.ts `planMeshBatches`) for a buffer limit, with the builders' bytes-per-triangle constants (the largest buffer's share). */
function plan(geometries: readonly Geometry[], limitBytes: number, bytesPerTriangle: number, maxMeshTriangles: number): { batches: number[][]; tooBig: number[] } {
    const device = { limits: { maxStorageBufferBindingSize: limitBytes, maxBufferSize: limitBytes } } as unknown as GPUDevice;
    return planMeshBatches(device, geometries, bytesPerTriangle, maxMeshTriangles);
}

/** H-PLOC: the largest buffer share is 64 bytes per triangle (gpu-hploc.ts LARGEST_BYTES_PER_TRIANGLE), meshes up to 2^26-ish triangles. */
const HPLOC_BYTES_PER_TRIANGLE = 64;
/** LBVH: 76 bytes per triangle, the climb's scratch (gpu-lbvh.ts LARGEST_BYTES_PER_TRIANGLE). */
const LBVH_BYTES_PER_TRIANGLE = (LBVH_SCRATCH_NODE_WORDS + LBVH_STATE_WORDS) * 4;
const NO_MESH_LIMIT = 2 ** 28;

interface BatchPlanResult {
    readonly limitMiB: number;
    readonly batches: number;
    readonly tooBig: number;
    /** Every batch created before the first submit (what GpuFirstTrees does): the sum of every batch's inputs, scratch and outputs. */
    readonly allAtOnce: number;
    /** Batches built one after another with their scratch freed: all outputs, plus the biggest batch's inputs and scratch. */
    readonly oneAtATime: number;
    readonly outputs: number;
    readonly largestScratchAndInputs: number;
    readonly allScratchAndInputs: number;
}

function planBatches(geometries: readonly Geometry[], limitBytes: number, builder: 'hploc' | 'lbvh'): BatchPlanResult {
    const { batches, tooBig } = plan(geometries, limitBytes, builder === 'hploc' ? HPLOC_BYTES_PER_TRIANGLE : LBVH_BYTES_PER_TRIANGLE, NO_MESH_LIMIT);
    const sizes = batches.map((members) => (builder === 'hploc' ? hplocBatch : lbvhBatch)(members.map((index) => geometries[index])));
    const outputs = sum(sizes.map((size) => size.outputs));
    const scratchAndInputs = sizes.map((size) => size.scratch + size.inputs);
    return {
        limitMiB: limitBytes / MIB,
        batches: batches.length,
        tooBig: tooBig.length,
        allAtOnce: outputs + sum(scratchAndInputs),
        oneAtATime: outputs + Math.max(0, ...scratchAndInputs),
        outputs,
        largestScratchAndInputs: Math.max(0, ...scratchAndInputs),
        allScratchAndInputs: sum(scratchAndInputs),
    };
}

interface TreeStats {
    nodes: number;
    triangles: number;
    leaves: number;
    parents: number;
    nodeBytes: number;
    triangleBytes: number;
    parentBytes: number;
    /** Sum over meshes of what `GpuMeshRefit.setShape` allocates, if every mesh were deforming (see refitBytes). */
    maxDepth: number;
}

function emptyStats(): TreeStats {
    return { nodes: 0, triangles: 0, leaves: 0, parents: 0, nodeBytes: 0, triangleBytes: 0, parentBytes: 0, maxDepth: 0 };
}

function addTree(stats: TreeStats, tree: MeshBvh): void {
    stats.nodes += tree.nodeCount;
    stats.triangles += tree.triangleCount;
    stats.leaves += tree.leafCount;
    stats.nodeBytes += tree.nodeCount * GPU_LAYOUT_NODE_BYTES;
    stats.triangleBytes += tree.triangleCount * GPU_LAYOUT_TRIANGLE_BYTES;
    stats.parentBytes += tree.parents.byteLength;
    stats.maxDepth = Math.max(stats.maxDepth, tree.maxDepth);
}

/**
 * What a deforming mesh adds, from GpuMeshRefit.setTree and GpuBvhRefit.setShape (gpu-mesh-refit.ts, gpu-refit.ts):
 * the CPU copy the caller holds (a `GpuLayoutMeshBvh`: nodes, triangles, parents), setTree's CPU arrays (parent slots,
 * leaf slots; transient), and the GPU buffers: primitive boxes (8 floats per triangle), the topology buffer (parent slots,
 * leaf slots, level starts, level nodes, created at twice its word count) and the scratch (3 words per node, twice).
 */
function refitBytes(tree: { nodeCount: number; triangleCount: number; leafCount: number; maxDepth: number }): { cpuCopy: number; cpuCopyNeeded: number; cpuTransient: number; gpu: number } {
    const topologyWords = tree.nodeCount + tree.leafCount * 1 + (tree.maxDepth + 1) + tree.nodeCount;
    return {
        cpuCopy: tree.nodeCount * 64 + tree.triangleCount * 48 + tree.nodeCount * 4,
        // setTree reads only the nodes and parents.
        cpuCopyNeeded: tree.nodeCount * 64 + tree.nodeCount * 4,
        cpuTransient: tree.nodeCount * 4 + tree.leafCount * 4 + tree.nodeCount * 4 + topologyWords * 4,
        gpu: Math.max(64, tree.triangleCount * PRIMITIVE_BOX_FLOATS * 4) + Math.max(64, topologyWords * 8) + Math.max(64, tree.nodeCount * 3 * 8),
    };
}

/** three-mesh-bvh over every mesh with the options; its own geometry per mesh (it reorders the index in place). */
function threeMeshBvh(geometries: readonly Geometry[], options: { strategy: number; targetLeafSize: number; indirect?: boolean }): Record<string, number> {
    const result = { nodeBytes: 0, nodeCount: 0, indirectBytes: 0, indexBytes: 0, addedIndexBytes: 0, positionBytes: 0, estimate: 0, buildMs: 0 };
    const start = performance.now();
    for (const geometry of geometries) {
        const bufferGeometry = new BufferGeometry();
        bufferGeometry.setAttribute('position', new BufferAttribute(geometry.positions, 3));
        // The app's index array is reordered in place, so each build gets its own copy.
        bufferGeometry.setIndex(new BufferAttribute(geometry.indices.slice(), 1));
        const bvh = new MeshBVH(bufferGeometry, { strategy: options.strategy, targetLeafSize: options.targetLeafSize, indirect: options.indirect ?? false, verbose: false } as never);
        const internals = bvh as unknown as { _roots: ArrayBuffer[]; _indirectBuffer: ArrayBufferView | null };
        result.nodeBytes += sum(internals._roots.map((root) => root.byteLength));
        result.nodeCount += sum(internals._roots.map((root) => root.byteLength / 32));
        result.indirectBytes += internals._indirectBuffer?.byteLength ?? 0;
        result.indexBytes += bufferGeometry.index!.array.byteLength;
        result.positionBytes += geometry.positions.byteLength;
        result.estimate += estimateMemoryInBytes(bvh);
    }
    result.buildMs = performance.now() - start;
    return result;
}

async function main(): Promise<void> {
    const args = process.argv.slice(2);
    const sceneId = args.find((arg) => !arg.startsWith('--') && args[args.indexOf(arg) - 1] !== '--json') ?? 'sponza';
    const jsonPath = args.includes('--json') ? args[args.indexOf('--json') + 1] : null;
    const skipTwins = args.includes('--no-twins');
    const skipTmbvh = args.includes('--no-tmbvh');
    const out: Record<string, unknown> = { scene: sceneId };

    const { meshes: sceneMeshes, instances } = readSceneInstances(sceneId);
    const geometries = sceneMeshes as Geometry[];
    const triangleTotal = sum(geometries.map((geometry) => geometry.indices.length / 3));
    const vertexTotal = sum(geometries.map((geometry) => geometry.positions.length / 3));
    const geometry = {
        meshes: geometries.length,
        instances: instances.length,
        triangles: triangleTotal,
        vertices: vertexTotal,
        positionBytes: sum(geometries.map((g) => g.positions.byteLength)),
        indexBytes: sum(geometries.map((g) => g.indices.byteLength)),
        largestMeshTriangles: Math.max(...geometries.map((g) => g.indices.length / 3)),
    };
    out.geometry = geometry;
    console.log(`${sceneId}: ${geometry.meshes} meshes, ${geometry.instances} instances, ${triangleTotal} triangles, ${vertexTotal} vertices`);
    console.log(`  geometry the app holds: positions ${mib(geometry.positionBytes)} MiB, 32-bit indices ${mib(geometry.indexBytes)} MiB`);

    // Ours: tinybvh trees through buildTraceScene, the path SceneTrees' CPU side and the benches take.
    const tinyBvh = await createTinyBvhBuilder();
    const sceneInstances = instances.map((instance, index) => ({ meshIndex: instance.meshIndex, instanceIndex: index, objectToWorld: instance.objectToWorld, static: true }));
    const stats = emptyStats();
    const perMesh: { nodeCount: number; triangleCount: number; leafCount: number; maxDepth: number }[] = [];
    const record = (tree: MeshBvh): MeshBvh => {
        addTree(stats, tree);
        perMesh.push({ nodeCount: tree.nodeCount, triangleCount: tree.triangleCount, leafCount: tree.leafCount, maxDepth: tree.maxDepth });
        return tree;
    };
    // Records, f32 nodes, instanced: the layout every section below is a variant of (named: the defaults became merged and quantized on 2026-11-13).
    let scene = await buildTraceScene(geometries, sceneInstances, { build: (g) => record(buildMeshBvhWithTinyBvh(g, tinyBvh)), mergeStaticInstances: false, nodeFormat: 'f32' });
    const ours = {
        nodes: scene.nodes.byteLength,
        triangles: scene.triangles.byteLength,
        topLevel: scene.topLevel.buffer.byteLength,
        topLevelNodes: scene.topLevel.nodeCount,
        parentsCpu: stats.parentBytes,
        nodeCount: stats.nodes,
        leafCount: stats.leaves,
        averageLeafTriangles: stats.triangles / stats.leaves,
        maxDepth: stats.maxDepth,
        maxMeshDepth: scene.maxMeshDepth,
    };
    out.ours = ours;
    console.log(`  ours, records and f32 nodes (tinybvh, leaves of up to 8): nodes ${mib(ours.nodes)} MiB (${ours.nodeCount} nodes, ${ours.averageLeafTriangles.toFixed(2)} triangles per leaf), triangle records ${mib(ours.triangles)}, parents (CPU only) ${mib(ours.parentsCpu)}, top level ${mib(ours.topLevel)}`);
    scene = undefined as never;

    // Indexed leaves (docs/internals/design/indexed-leaves.md): the same tinybvh trees with 16-byte triangles and the scene's own
    // positions, welded by exact bits (the default) or not, as vec4f (the default) or packed f32; instanced and merged.
    const indexedOut: Record<string, { nodes: number; triangles: number; positions: number; topLevel: number; vertices: number; total: number }> = {};
    for (const [name, options] of [
        ['instanced, vec4, welded', { positionLayout: 'vec4' as const, weld: true, merge: false }],
        ['instanced, vec4, unwelded', { positionLayout: 'vec4' as const, weld: false, merge: false }],
        ['instanced, packed, welded', { positionLayout: 'packed' as const, weld: true, merge: false }],
        ['merged, vec4, welded', { positionLayout: 'vec4' as const, weld: true, merge: true }],
    ] as const) {
        const built = await buildTraceScene(geometries, sceneInstances, {
            build: (g) => buildMeshBvhWithTinyBvh(g, tinyBvh, { triangleLeaves: 'indexed' }),
            triangleLeaves: 'indexed',
            positionLayout: options.positionLayout,
            weld: options.weld,
            mergeStaticInstances: options.merge,
            nodeFormat: 'f32',
        });
        const entry = {
            nodes: built.nodes.byteLength,
            triangles: built.triangles.byteLength,
            positions: built.positions.byteLength,
            topLevel: built.topLevel.buffer.byteLength,
            vertices: built.vertexCount,
            total: built.byteLength,
        };
        indexedOut[name] = entry;
        console.log(
            `  indexed (${name}): nodes ${mib(entry.nodes)} MiB, triangles ${mib(entry.triangles)}, positions ${mib(entry.positions)} (${entry.vertices} vertices), top level ${mib(entry.topLevel)}; total ${mib(entry.total)} MiB`,
        );
    }
    out.indexed = indexedOut;

    // Quantized nodes (docs/internals/design/compressed-nodes.md): the same tinybvh trees with 32-byte nodes and a 32-byte frame
    // slot per placed tree, with records and with indexed leaves (vec4, welded); instanced and merged.
    const quantizedOut: Record<string, { nodes: number; triangles: number; positions: number; topLevel: number; total: number }> = {};
    for (const [name, options] of [
        ['instanced, records', { indexed: false, merge: false }],
        ['instanced, indexed', { indexed: true, merge: false }],
        ['merged, records', { indexed: false, merge: true }],
        ['merged, indexed', { indexed: true, merge: true }],
    ] as const) {
        const built = await buildTraceScene(geometries, sceneInstances, {
            build: (g) => buildMeshBvhWithTinyBvh(g, tinyBvh, options.indexed ? { triangleLeaves: 'indexed' } : {}),
            nodeFormat: 'quantized',
            ...(options.indexed ? { triangleLeaves: 'indexed' as const } : {}),
            mergeStaticInstances: options.merge,
        });
        const entry = { nodes: built.nodes.byteLength, triangles: built.triangles.byteLength, positions: built.positions.byteLength, topLevel: built.topLevel.buffer.byteLength, total: built.byteLength };
        quantizedOut[name] = entry;
        console.log(`  quantized nodes (${name}): nodes ${mib(entry.nodes)} MiB, triangles ${mib(entry.triangles)}, positions ${mib(entry.positions)}, top level ${mib(entry.topLevel)}; total ${mib(entry.total)} MiB`);
    }
    out.quantized = quantizedOut;

    // Leaf size trend: tinybvh at the layout's allowed leaf sizes.
    const leafTrend: Record<number, { nodes: number; nodeBytes: number; leaves: number; triangles: number }> = {};
    for (const maxLeaf of [1, 2, 4, 8]) {
        let nodes = 0;
        let leaves = 0;
        for (const mesh of geometries) {
            const layout = tinyBvh.buildLayout(mesh, { maxLeafPrimitives: maxLeaf }).layout;
            nodes += layout.nodeCount;
            leaves += layout.leafCount;
        }
        leafTrend[maxLeaf] = { nodes, nodeBytes: nodes * GPU_LAYOUT_NODE_BYTES, leaves, triangles: triangleTotal };
        console.log(`  tinybvh leaves of up to ${maxLeaf}: ${nodes} nodes = ${mib(nodes * GPU_LAYOUT_NODE_BYTES)} MiB, ${(triangleTotal / leaves).toFixed(2)} triangles per leaf`);
    }
    out.leafTrend = leafTrend;

    // Merged static: every instance static, meshes used once merge into world-space groups.
    const mergedStats = emptyStats();
    const merged = await buildTraceScene(geometries, sceneInstances, {
        build: (g) => {
            const tree = buildMeshBvhWithTinyBvh(g, tinyBvh);
            addTree(mergedStats, tree);
            return tree;
        },
        mergeStaticInstances: true,
        nodeFormat: 'f32',
    });
    const mergedGroups = merged.mergedGroups.length;
    const mergedGroupTriangles = sum(merged.mergedGroups.map((group) => group.triangleCount));
    out.merged = {
        nodes: merged.nodes.byteLength,
        triangles: merged.triangles.byteLength,
        topLevel: merged.topLevel.buffer.byteLength,
        groups: mergedGroups,
        mergedTriangles: mergedGroupTriangles,
        instancedMeshesStillNeeded: merged.meshes.filter((mesh, index) => mesh.triangleCount > 0 && index < geometries.length).length,
        parentsCpu: mergedStats.parentBytes,
        // mergeStaticGeometry's CPU plan: world-space positions and 32-bit indices, plus 4 + 4 + 1 + 2 bytes of source ids per triangle.
        planCpuBytes: undefined as number | undefined,
    };
    // The plan's geometry is rebuilt here only to count it (assembly drops it after the build).
    {
        const { mergeStaticGeometry } = await import('../src/trace/static-merge.ts');
        const plan = mergeStaticGeometry(geometries, sceneInstances);
        const planBytes = sum(
            plan.groups.map(
                (group) => group.geometry.positions.byteLength + (group.geometry.indices?.byteLength ?? 0) + group.sourceInstances.byteLength + group.sourceTriangles.byteLength + group.flipFrontFace.byteLength + group.sourceMaterials.byteLength,
            ),
        );
        (out.merged as { planCpuBytes: number }).planCpuBytes = planBytes;
    }
    const mergedOut = out.merged as { nodes: number; triangles: number; topLevel: number; groups: number; mergedTriangles: number; instancedMeshesStillNeeded: number; planCpuBytes: number };
    console.log(
        `  merged static: nodes ${mib(mergedOut.nodes)} MiB, triangle records ${mib(mergedOut.triangles)}, top level ${mib(mergedOut.topLevel)}, ${mergedGroups} groups of ${mergedGroupTriangles} triangles, ` +
            `${mergedOut.instancedMeshesStillNeeded} meshes keep their own trees; merge plan on the CPU ${mib(mergedOut.planCpuBytes)} MiB`,
    );

    // The refit's CPU copy and GPU buffers, if every mesh were deforming.
    const refit = perMesh.map(refitBytes);
    const largest = perMesh.reduce((best, mesh, index) => (mesh.triangleCount > perMesh[best].triangleCount ? index : best), 0);
    out.refit = {
        cpuCopyAllMeshes: sum(refit.map((r) => r.cpuCopy)),
        cpuCopyNeededAllMeshes: sum(refit.map((r) => r.cpuCopyNeeded)),
        cpuTransientAllMeshes: sum(refit.map((r) => r.cpuTransient)),
        gpuAllMeshes: sum(refit.map((r) => r.gpu)),
        largestMesh: { triangles: perMesh[largest].triangleCount, ...refit[largest] },
    };
    const refitOut = out.refit as { cpuCopyAllMeshes: number; cpuCopyNeededAllMeshes: number; gpuAllMeshes: number };
    console.log(`  refit if every mesh deformed: CPU copy ${mib(refitOut.cpuCopyAllMeshes)} MiB (setTree needs ${mib(refitOut.cpuCopyNeededAllMeshes)}), refit GPU buffers ${mib(refitOut.gpuAllMeshes)} MiB`);

    // GPU build batches: all meshes in one batch (a large-limit device), and at the planner's limits.
    const oneBatchHploc = hplocBatch(geometries);
    const oneBatchHplocIndexed = hplocBatch(geometries, 'cubic-30', 16);
    out.gpuBatchHplocIndexed = oneBatchHplocIndexed;
    console.log(`  H-PLOC one batch, indexed leaves: outputs ${mib(oneBatchHplocIndexed.outputs)} MiB, total ${mib(oneBatchHplocIndexed.inputs + oneBatchHplocIndexed.scratch + oneBatchHplocIndexed.outputs)} MiB, ${((oneBatchHplocIndexed.inputs + oneBatchHplocIndexed.scratch + oneBatchHplocIndexed.outputs) / oneBatchHplocIndexed.triangles).toFixed(1)} bytes per triangle (records ${((oneBatchHploc.inputs + oneBatchHploc.scratch + oneBatchHploc.outputs) / oneBatchHploc.triangles).toFixed(1)})`);
    const oneBatchLbvh = lbvhBatch(geometries);
    out.gpuBatchHploc = oneBatchHploc;
    out.gpuBatchLbvh = oneBatchLbvh;
    console.log(`  H-PLOC one batch (${oneBatchHploc.triangles} triangles): inputs ${mib(oneBatchHploc.inputs)} + scratch ${mib(oneBatchHploc.scratch)} + outputs ${mib(oneBatchHploc.outputs)} = ${mib(oneBatchHploc.inputs + oneBatchHploc.scratch + oneBatchHploc.outputs)} MiB`);
    console.log(`  LBVH    one batch: inputs ${mib(oneBatchLbvh.inputs)} + scratch ${mib(oneBatchLbvh.scratch)} + outputs ${mib(oneBatchLbvh.outputs)} = ${mib(oneBatchLbvh.inputs + oneBatchLbvh.scratch + oneBatchLbvh.outputs)} MiB`);
    const plans: Record<string, BatchPlanResult[]> = { hploc: [], lbvh: [] };
    for (const builder of ['hploc', 'lbvh'] as const) {
        for (const limitMiB of [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]) {
            const result = planBatches(geometries, limitMiB * MIB, builder);
            plans[builder].push(result);
            console.log(
                `  ${builder} planned at ${limitMiB} MiB buffers: ${result.batches} batches (${result.tooBig} too big), all at once ${mib(result.allAtOnce)} MiB, ` +
                    `one at a time ${mib(result.oneAtATime)} MiB (outputs ${mib(result.outputs)} + biggest batch ${mib(result.largestScratchAndInputs)})`,
            );
        }
    }
    out.batchPlans = plans;

    // The CPU twins: the exact trees the GPU builders write (their node counts are the used ones).
    if (!skipTwins) {
        const twins: Record<string, { nodes: number; leaves: number; triangles: number; ms: number }> = {};
        for (const [name, build] of [
            ['H-PLOC twin', (g: Geometry) => buildHplocOnCpu(g)],
            ['LBVH twin', (g: Geometry) => buildLbvhOnCpu(g)],
        ] as const) {
            const start = performance.now();
            const twin = { nodes: 0, leaves: 0, triangles: 0, ms: 0 };
            for (const mesh of geometries) {
                const layout = build(mesh);
                twin.nodes += layout.nodeCount;
                twin.leaves += layout.leafCount;
                twin.triangles += layout.triangleCount;
            }
            twin.ms = performance.now() - start;
            twins[name] = twin;
            console.log(`  ${name}: ${twin.nodes} nodes used = ${mib(twin.nodes * GPU_LAYOUT_NODE_BYTES)} MiB (the GPU allocates ${triangleTotal - geometries.length} = ${mib((triangleTotal - geometries.length) * GPU_LAYOUT_NODE_BYTES)} MiB), ${(twin.triangles / twin.leaves).toFixed(2)} triangles per leaf, ${Math.round(twin.ms / 1000)} s`);
        }
        out.twins = twins;
    }

    // Worker copies: the largest mesh's input (copied into a worker unless transferInput), and the WASM heap one worker keeps.
    {
        const biggest = geometries.reduce((best, g) => (g.indices.length > best.indices.length ? g : best), geometries[0]);
        // tinybvh's WASM memory only grows: the size it has after the largest mesh is what a worker keeps for its life.
        const memories: WebAssembly.Memory[] = [];
        const instantiate = WebAssembly.instantiate;
        (WebAssembly as { instantiate: unknown }).instantiate = async (...input: unknown[]) => {
            const result = await (instantiate as (...a: unknown[]) => Promise<WebAssembly.Instance>)(...input);
            const memory = (result.exports as { memory?: WebAssembly.Memory }).memory;
            if (memory) memories.push(memory);
            return result;
        };
        const fresh = await createTinyBvhBuilder();
        (WebAssembly as { instantiate: unknown }).instantiate = instantiate;
        const initialHeap = memories[0]?.buffer.byteLength ?? 0;
        fresh.buildLayout(biggest);
        const after = memories[0]?.buffer.byteLength ?? 0;
        const largestOutput = tinyBvh.buildLayout(biggest).layout;
        out.worker = {
            largestMeshInputBytes: biggest.positions.byteLength + biggest.indices.byteLength,
            largestMeshOutputBytes: largestOutput.nodes.byteLength + largestOutput.triangles.byteLength + largestOutput.parents.byteLength,
            wasmHeapAfterLargestMesh: after,
            wasmHeapBeforeBuild: initialHeap,
            largestMeshTriangles: biggest.indices.length / 3,
        };
        const w = out.worker as { largestMeshInputBytes: number; largestMeshOutputBytes: number; wasmHeapAfterLargestMesh: number };
        console.log(`  one worker: input copy of the largest mesh ${mib(w.largestMeshInputBytes)} MiB, its output (transferred, not copied) ${mib(w.largestMeshOutputBytes)} MiB, WASM heap a worker keeps after it ${mib(w.wasmHeapAfterLargestMesh)} MiB`);
    }

    // The CPU twins' leaves and tinybvh's tree shape are done; the comparison library.
    if (!skipTmbvh) {
        const tmbvh: Record<string, Record<string, number>> = {};
        for (const [name, options] of [
            ['CENTER, leaf 10 (defaults)', { strategy: CENTER, targetLeafSize: 10 }],
            ['SAH, leaf 4', { strategy: SAH, targetLeafSize: 4 }],
            ['SAH, leaf 10', { strategy: SAH, targetLeafSize: 10 }],
            ['CENTER, leaf 10, indirect', { strategy: CENTER, targetLeafSize: 10, indirect: true }],
        ] as const) {
            const result = threeMeshBvh(geometries, options);
            tmbvh[name] = result;
            const handCount = result.nodeBytes + result.indirectBytes + result.indexBytes + result.positionBytes;
            console.log(
                `  three-mesh-bvh ${name}: nodes ${mib(result.nodeBytes)} MiB (${result.nodeCount} nodes of 32 bytes), indirect ${mib(result.indirectBytes)}, index ${mib(result.indexBytes)}, positions ${mib(result.positionBytes)}; ` +
                    `by hand ${mib(handCount)} MiB, estimateMemoryInBytes ${mib(result.estimate)} MiB, build ${Math.round(result.buildMs / 1000)} s`,
            );
        }
        out.tmbvh = tmbvh;
    }

    if (jsonPath) writeFileSync(jsonPath, `${JSON.stringify(out, null, 1)}\n`);
}

await main();

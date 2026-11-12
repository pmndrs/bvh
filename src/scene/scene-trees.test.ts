/**
 * SceneTrees in Node: which tree each mesh gets and when, runtime adds and
 * removals, merged static groups, the adopt lever, and that the CPU walk over
 * the scene's buffers (read from the fake GPU) gives the hits of the scene
 * assembled on the CPU from the same meshes and instances, in every state.
 *
 * Node has no GPU, so a stand-in first-tree builder makes the "GPU" trees:
 * tinybvh trees with one triangle per leaf (a different shape from the pool's,
 * as H-PLOC's is), in fake buffers laid out like an H-PLOC batch's (nodes
 * packed at the front of a buffer with room for one node per triangle). The
 * merged-identity pass runs as a CPU stand-in of its WGSL. The device has no
 * subgroups, so the top level is the CPU one; the GPU top level, the real
 * H-PLOC and the identity kernel are checked on GPUs (tests/gpu/pages/scene-trees-*).
 */

import { describe, expect, it } from 'vitest';
import { createFakeGpu, type FakeBinding, type FakeBuffer, type FakeGpu } from '../../tests/support/fake-gpu.ts';
import { createCpuTracer, type CpuHit } from '../trace/cpu-traversal.ts';
import { INDEXED_FLAG_SHIFT, INDEXED_ID_MASK, INSTANCE_FLAG, INSTANCE_RECORD_WORDS, INSTANCE_WORD, TOP_HEADER_WORD, TOP_LEVEL_HEADER_WORDS, TRIANGLE_WORD, TRIANGLE_WORDS } from '../trace/layout.ts';
import { MERGED_INSTANCE_INDEX } from '../trace/static-merge.ts';
import { buildMeshBvhAsync } from '../trace/mesh-bvh-build.ts';
import type { MeshGeometry } from '../trace/mesh-bvh.ts';
import { createRandom, randomAffine, randomRays, randomTriangleSoup, tessellatedQuad } from '../trace/oracle-fixtures.ts';
import type { TraceRay } from '../trace/rays.ts';
import { assembleTraceScene, type TraceSceneData, type TraceSceneInstance } from '../trace/scene-assembly.ts';
import type { TopLevelBvh } from '../trace/top-level.ts';
import { GpuFirstTrees, GpuTreeOutputs, type FirstTreeBuilder, type GpuTreeBatchResult, type GpuTreeBuildOptions, type GpuTreeBuilds, type GpuTreeRequest } from './gpu-first-trees.ts';
import { MERGED_SOURCE_WORDS } from './merged-identity-gpu.ts';
import { DEFAULT_GPU_MAX_BATCH_BYTES, SceneTrees, type SceneInstanceDescriptor, type SceneTreeEvent, type SceneTreesOptions } from './scene-trees.ts';
import type { GpuMeshTreeRange } from '../mesh-input.ts';
import { gatherStandIn } from '../../tests/support/top-level-gather.ts';
import { quantizeNodeWords } from '../quantized-nodes.ts';
import { FakeHploc, settle } from '../../tests/support/fake-hploc.ts';
import { addToTally, compareClosest, newTally } from '../trace/oracle-compare.ts';
import { buildReferenceScene, traceClosestReference } from '../trace/reference-tracer.ts';

/** The merged-identity WGSL, line by line on the CPU. */
function mergedIdentityStandIn(bindings: ReadonlyMap<number, FakeBinding>): void {
    const triangles = new Uint32Array(bindings.get(0)!.buffer.bytes.buffer);
    const sources = new Uint32Array(bindings.get(1)!.buffer.bytes.buffer);
    const [wordOffset, count, sourceCount] = new Uint32Array(bindings.get(2)!.buffer.bytes.buffer);
    for (let slot = 0; slot < count; slot++) {
        const base = wordOffset + slot * TRIANGLE_WORDS;
        const merged = triangles[base + TRIANGLE_WORD.originalIndex];
        let low = 0;
        let high = sourceCount;
        while (high - low > 1) {
            const middle = (low + high) >> 1;
            if (sources[middle * MERGED_SOURCE_WORDS] <= merged) low = middle;
            else high = middle;
        }
        const at = low * MERGED_SOURCE_WORDS;
        triangles[base + TRIANGLE_WORD.originalIndex] = merged - sources[at];
        triangles[base + TRIANGLE_WORD.padding] = sources[at + 1];
        triangles[base + TRIANGLE_WORD.flags] = (triangles[base + TRIANGLE_WORD.flags] | sources[at + 2] | (materialOf(bindings, merged) << 16)) >>> 0;
    }
}

/** A merged triangle's material code from the identity pass's per-triangle table (binding 3, two codes to a word). */
function materialOf(bindings: ReadonlyMap<number, FakeBinding>, merged: number): number {
    const table = new Uint32Array(bindings.get(3)!.buffer.bytes.buffer);
    return (table[merged >> 1] >>> ((merged & 1) * 16)) & 0xffff;
}

/** The quantize pass (gpu-quantize.ts, `bvhqQuantize`) on the CPU: the CPU encoder it matches bit for bit (bvh-blas-refit checks that on GPUs). */
function quantizeStandIn(bindings: ReadonlyMap<number, FakeBinding>): void {
    const source = bindings.get(0)!.buffer.bytes;
    const target = new Uint32Array(bindings.get(1)!.buffer.bytes.buffer);
    const table = new Uint32Array(bindings.get(2)!.buffer.bytes.buffer);
    const [meshCount] = new Uint32Array(bindings.get(3)!.buffer.bytes.buffer);
    for (let mesh = 0; mesh < meshCount; mesh++) {
        const [sourceNodeOffset, rootSlot, nodeCount] = table.subarray(mesh * 4, mesh * 4 + 3);
        const words = new Uint32Array(source.buffer.slice(source.byteOffset + sourceNodeOffset * 64, source.byteOffset + (sourceNodeOffset + nodeCount) * 64));
        quantizeNodeWords(new Float32Array(words.buffer), words, 0, nodeCount, target, (rootSlot - 1) * 8);
    }
}

/** The indexed merged-identity WGSL (`bvhMergedIdentityIndexed`), line by line on the CPU. */
function mergedIdentityIndexedStandIn(bindings: ReadonlyMap<number, FakeBinding>): void {
    const triangles = new Uint32Array(bindings.get(0)!.buffer.bytes.buffer);
    const sources = new Uint32Array(bindings.get(1)!.buffer.bytes.buffer);
    const [wordOffset, count, sourceCount, sideWordOffset] = new Uint32Array(bindings.get(2)!.buffer.bytes.buffer);
    for (let slot = 0; slot < count; slot++) {
        const word = wordOffset + slot * 4 + 3;
        const merged = triangles[word] & INDEXED_ID_MASK;
        let low = 0;
        let high = sourceCount;
        while (high - low > 1) {
            const middle = (low + high) >> 1;
            if (sources[middle * MERGED_SOURCE_WORDS] <= merged) low = middle;
            else high = middle;
        }
        const at = low * MERGED_SOURCE_WORDS;
        const flags = (triangles[word] >>> INDEXED_FLAG_SHIFT) | (sources[at + 2] & 2);
        triangles[word] = ((merged - sources[at]) | (flags << INDEXED_FLAG_SHIFT)) >>> 0;
        triangles[sideWordOffset + slot] = (low | (materialOf(bindings, merged) << 16)) >>> 0;
    }
}

/** A first-tree builder that answers when the test says so; each `build` is one batch. */
class StandInGpuBuilder implements FirstTreeBuilder {
    readonly requests: GpuTreeRequest[][] = [];
    private readonly pending: { members: { slot: number; generation: number; geometry: MeshGeometry }[]; readLayouts: boolean; resolve: (result: GpuTreeBatchResult) => void }[] = [];
    /** Each build's `readLayouts`. */
    readonly readLayouts: boolean[] = [];
    /** Answers `readLayouts` with no layouts, as a failed readback does. */
    failLayouts = false;
    readonly refuse = new Set<number>();
    /** Labels the batch's nodes quantized, as a misconfigured custom builder would. */
    quantizedOutputs = false;

    constructor(
        private readonly device: GPUDevice,
        private readonly triangleLeaves: 'records' | 'indexed' = 'records',
    ) {}

    async prepare(): Promise<void> {
        await buildMeshBvhAsync({ positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]) });
    }

    build(requests: readonly GpuTreeRequest[], options: GpuTreeBuildOptions = {}): GpuTreeBuilds {
        this.requests.push([...requests]);
        this.readLayouts.push(options.readLayouts ?? false);
        const readLayouts = options.readLayouts ?? false;
        const members = requests.map(({ slot, generation, geometry }) => {
            const cpu = geometry as MeshGeometry;
            return {
                slot,
                generation,
                geometry: { positions: cpu.positions.slice(), indices: cpu.indices ? Uint32Array.from(cpu.indices) : null, triangleFlags: cpu.triangleFlags?.slice() ?? null, triangleMaterials: cpu.triangleMaterials ?? null },
            };
        });
        const batch = new Promise<GpuTreeBatchResult>((resolve) => this.pending.push({ members, readLayouts, resolve }));
        return { notBuilt: [], batches: [batch], buildingBytes: 1000 };
    }

    get waiting(): number {
        return this.pending.length;
    }

    /** Answers the oldest batch waiting, or the one at `index`. */
    async finishBatch(index = 0): Promise<void> {
        const batch = this.pending.splice(index, 1)[0];
        if (!batch) throw new Error('no batch is waiting');
        const trees = await Promise.all(batch.members.map((member) => buildMeshBvhAsync(member.geometry, { maxLeafPrimitives: 1, triangleLeaves: this.triangleLeaves })));
        const bytes = this.triangleLeaves === 'indexed' ? 16 : 48;
        // As H-PLOC's outputs: room for one node per triangle, the used nodes packed at the front.
        const triangleTotal = trees.reduce((sum, tree) => sum + tree.triangleCount, 0);
        const usage = 0x4 | 0x8 | 0x80;
        const nodes = this.device.createBuffer({ label: 'stand-in nodes', size: triangleTotal * 64, usage });
        const triangles = this.device.createBuffer({ label: 'stand-in triangles', size: triangleTotal * bytes, usage });
        const parents = this.device.createBuffer({ label: 'stand-in parents', size: triangleTotal * 4, usage });
        const meshes: GpuMeshTreeRange[] = [];
        let nodeOffset = 0;
        let triangleOffset = 0;
        trees.forEach((tree, index) => {
            this.device.queue.writeBuffer(nodes, nodeOffset * 64, tree.nodes, 0, tree.nodeCount * 64);
            this.device.queue.writeBuffer(triangles, triangleOffset * bytes, tree.triangles, 0, tree.triangleCount * bytes);
            const refused = this.refuse.has(batch.members[index].slot);
            meshes.push({ nodeOffset, nodeCount: tree.nodeCount, triangleOffset, triangleCount: tree.triangleCount, leafCount: tree.leafCount, maxDepth: tree.maxDepth, bounds: tree.bounds, problem: refused ? 'refused by the test' : null });
            nodeOffset += tree.nodeCount;
            triangleOffset += tree.triangleCount;
        });
        const usable = meshes.filter((mesh) => mesh.problem === null).length;
        const outputs = new GpuTreeOutputs({ nodes, triangles, parents, meshes, triangleLeaves: this.triangleLeaves, ...(this.quantizedOutputs ? { nodeFormat: 'quantized' as const } : {}), destroy: () => [nodes, triangles, parents].forEach((buffer) => buffer.destroy()) }, usable);
        batch.resolve({
            built: meshes.flatMap((tree, index) =>
                tree.problem ? [] : [{ slot: batch.members[index].slot, generation: batch.members[index].generation, outputs, tree, layout: batch.readLayouts && !this.failLayouts ? trees[index] : null }],
            ),
            refused: meshes.flatMap((tree, index) => (tree.problem ? [{ slot: batch.members[index].slot, generation: batch.members[index].generation, reason: tree.problem }] : [])),
            buildMs: 1,
            failure: null,
        });
        await Promise.resolve();
        await Promise.resolve();
    }
}

interface Instance extends SceneInstanceDescriptor {
    readonly meshIndex: number;
}

async function setUp(options: Partial<SceneTreesOptions> & { stand?: boolean } = {}) {
    const gpu: FakeGpu = createFakeGpu({}, { kernels: { bvhMergedIdentity: mergedIdentityStandIn, bvhMergedIdentityIndexed: mergedIdentityIndexedStandIn, bvhqQuantize: quantizeStandIn } });
    const stand = options.stand === false ? null : new StandInGpuBuilder(gpu.device, options.triangleLeaves);
    const events: SceneTreeEvent[] = [];
    const scene = await SceneTrees.create(gpu.device, { ...options, firstTreeBuilder: stand ?? undefined, onTree: (event) => events.push(event) });
    const frame = () => {
        const encoder = gpu.createEncoder();
        scene.update(encoder);
        gpu.submit(encoder);
    };
    return { gpu, scene, stand, frame, events };
}

/** The scene's buffers as the fake GPU holds them, for the CPU walk. */
function readScene(scene: SceneTrees): TraceSceneData {
    const { nodes, triangles, topLevel, positions } = scene.buffers;
    const bytes = (buffer: GPUBuffer) => (buffer as unknown as FakeBuffer).bytes.slice().buffer;
    const top = bytes(topLevel);
    const header = new Uint32Array(top, 0, 4);
    return {
        nodes: bytes(nodes),
        nodeFormat: scene.nodeFormat,
        triangles: bytes(triangles),
        triangleLeaves: scene.triangleLeaves,
        positions: positions ? bytes(positions) : new ArrayBuffer(0),
        positionLayout: scene.positionLayout,
        vertexCount: 0,
        topLevel: { buffer: top, instanceCount: header[TOP_HEADER_WORD.instanceCount], nodeCount: header[TOP_HEADER_WORD.nodeCount] } as TopLevelBvh,
        meshes: [],
        maxMeshDepth: 0,
        requiredStackSize: 0,
        byteLength: 0,
        mergedGroups: [],
    };
}

/** The scene assembled on the CPU from tinybvh trees: the reference (instanced, never merged). */
async function reference(geometries: readonly MeshGeometry[], instances: readonly Instance[]): Promise<TraceSceneData> {
    const used = new Set(instances.map((instance) => instance.meshIndex));
    const meshes = await Promise.all(geometries.map((geometry, index) => (used.has(index) ? buildMeshBvhAsync(geometry) : null)));
    const traced: TraceSceneInstance[] = instances.map((instance) => ({
        meshIndex: instance.meshIndex,
        instanceIndex: instance.instanceIndex!,
        objectToWorld: instance.objectToWorld,
        visible: instance.visible,
        contributesToGi: instance.contributesToGi,
        mask: instance.mask,
        material: instance.material,
    }));
    return assembleTraceScene(meshes, traced);
}

const key = (hit: CpuHit | null) => (hit ? `${hit.instanceIndex}/${hit.triangleIndex}/${hit.frontFace}/${hit.material}` : '-');

/** Rays where the scene and the reference disagree on the closest hit (same instance, triangle, side and material; t to rounding) or on any-hit. */
function disagreements(actual: TraceSceneData, expected: TraceSceneData, rays: readonly TraceRay[]): string[] {
    const a = createCpuTracer(actual);
    const b = createCpuTracer(expected);
    const problems: string[] = [];
    rays.forEach((ray, index) => {
        const x = a.closest(ray);
        const y = b.closest(ray);
        const same = key(x) === key(y) && (x === null || Math.abs(x.t - y!.t) <= 1e-4 * Math.max(1, y!.t));
        const tie = x !== null && y !== null && Math.abs(x.t - y.t) <= 1e-6 * Math.max(1, y.t);
        if (!same && !tie) problems.push(`ray ${index}: ${key(x)} vs ${key(y)}`);
        if (a.any(ray) !== b.any(ray)) problems.push(`ray ${index}: any differs`);
    });
    return problems;
}

function sceneOf(count: number) {
    const random = createRandom(11);
    const geometries = Array.from({ length: count }, (_, index) => tessellatedQuad(3 + index, 1));
    const instances: Instance[] = geometries.map((_, index) => ({
        mesh: -1,
        meshIndex: index,
        instanceIndex: 100 + index,
        objectToWorld: randomAffine(random, { extent: 4, mirror: index % 2 === 1 }),
        material: index % 3 === 0 ? undefined : index,
        static: true,
    }));
    const rays = randomRays(random, 600, [-6, -6, -6, 6, 6, 6], 1);
    return { geometries, instances, rays };
}

function addAll(scene: SceneTrees, geometries: readonly MeshGeometry[], instances: readonly Instance[]): { meshes: number[]; handles: number[] } {
    const meshes = geometries.map((geometry) => scene.addMesh(geometry));
    const handles = instances.map((instance) => scene.addInstance({ ...instance, mesh: meshes[instance.meshIndex] }));
    return { meshes, handles };
}

describe('SceneTrees', () => {
    it('without a GPU builder: tinybvh trees, the CPU top level, and runtime adds and removals that trace like the CPU-assembled scene', async () => {
        const { scene, frame, events } = await setUp({ stand: false });
        expect(scene.gpuFirstTrees.enabled).toBe(false);
        expect(scene.topLevelPath).toBe('cpu');
        const { geometries, instances, rays } = sceneOf(4);
        const { meshes, handles } = addAll(scene, geometries, instances);
        frame();
        expect(scene.building).toBe(true);
        await scene.whenRefined();
        frame();
        expect(scene.building).toBe(false);
        expect(scene.counts).toMatchObject({ meshes: 4, onTinybvhTrees: 4, onGpuTrees: 0, swaps: 0 });
        expect(scene.meshStatus(meshes[0]).builtBy).toBe('tinybvh');
        expect(scene.meshStatus(meshes[0]).fallbackReason).toMatch(/subgroups/);
        expect(events.every((event) => event.builtBy === 'tinybvh' && event.replaced === null)).toBe(true);
        expect(disagreements(readScene(scene), await reference(geometries, instances), rays)).toEqual([]);

        // Runtime: a new mesh and instance, a move, a hide, a removal.
        const extra = tessellatedQuad(5, 0.8);
        const extraMesh = scene.addMesh(extra);
        const live: Instance[] = [...instances];
        const extraInstance: Instance = { mesh: extraMesh, meshIndex: 4, instanceIndex: 500, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1] };
        scene.addInstance(extraInstance);
        live.push(extraInstance);
        const moved = [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1];
        scene.setTransform(handles[1], moved);
        live[1] = { ...live[1], objectToWorld: moved };
        scene.setInstance(handles[2], { visible: false });
        live[2] = { ...live[2], visible: false };
        scene.removeInstance(handles[3]);
        live.splice(3, 1);
        scene.removeMesh(meshes[3]);
        expect(() => scene.removeMesh(meshes[0])).toThrow(/still used/);
        frame();
        await scene.whenRefined();
        frame();
        expect(disagreements(readScene(scene), await reference([...geometries, extra], live), rays)).toEqual([]);
        const memory = scene.memory;
        // Meshes of 3, 4 and 5 cells a side and the extra one of 5 (mesh 3 is gone): two triangles a cell.
        expect(memory.trianglesInUse).toBe((9 + 16 + 25 + 25) * 2 * 48);
        expect(memory.total).toBe(memory.resident + memory.transient);
        expect(memory.peak).toBeGreaterThanOrEqual(memory.total);
        scene.dispose();
    });

    it('GPU trees first, tinybvh swapped in mesh by mesh, with the same hits mid-swap, merged groups included', async () => {
        const { scene, stand, frame, events } = await setUp({ holdRefinements: true, mergeStaticInstances: true });
        const { geometries, instances, rays } = sceneOf(5);
        // One instance of mesh 4 more, so mesh 4 is used twice and stays instanced.
        const twin: Instance = { mesh: -1, meshIndex: 4, instanceIndex: 200, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1] };
        const all = [...instances, twin];
        const { meshes } = addAll(scene, geometries, all);
        frame();
        expect(scene.counts.mergedGroups).toBe(1);
        expect(scene.meshStatus(meshes[0]).mergedInto).toBe(0);
        expect(scene.meshStatus(meshes[4]).mergedInto).toBeNull();
        // One GPU batch: the merged group's geometry and mesh 4.
        expect(stand!.requests).toHaveLength(1);
        expect(stand!.requests[0]).toHaveLength(2);
        const expected = await reference(geometries, all);
        await stand!.finishBatch();
        frame();
        expect(scene.counts).toMatchObject({ onGpuTrees: 2, onTinybvhTrees: 0 });
        expect(disagreements(readScene(scene), expected, rays)).toEqual([]);

        await scene.whenRefined();
        // Mid-swap: the merged group on tinybvh, mesh 4 still on its GPU tree.
        scene.releaseRefinements((mesh) => mesh === -1);
        frame();
        expect(scene.groupStatus(0).builtBy).toBe('tinybvh');
        expect(scene.meshStatus(meshes[4]).builtBy).toBe('gpu');
        expect(disagreements(readScene(scene), expected, rays)).toEqual([]);
        scene.releaseRefinements();
        frame();
        expect(scene.counts).toMatchObject({ onGpuTrees: 0, onTinybvhTrees: 2, swaps: 2 });
        expect(disagreements(readScene(scene), expected, rays)).toEqual([]);
        expect(events.filter((event) => event.replaced === 'gpu').map((event) => event.group ?? `mesh ${event.mesh}`)).toEqual([0, `mesh ${meshes[4]}`]);
        expect(scene.timeline.firstTreesMs).not.toBeNull();
        expect(scene.timeline.tinybvhTreesMs).not.toBeNull();

        // Merged instances never move, hide or go.
        expect(() => scene.setTransform(0, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1])).toThrow(/merged static instance/);
        expect(() => scene.setInstance(0, { visible: false })).toThrow(/merged/);
        expect(() => scene.removeInstance(0)).toThrow(/merged/);
        // A GPU tree's outputs are freed one update after its copy.
        frame();
        expect(scene.memory.gpuBuildOutputs).toBe(0);

        // Runtime: a mesh added later gets a GPU tree placed after the others (its batch offsets are not the scene's).
        const late = tessellatedQuad(4, 0.5);
        const lateInstance: Instance = { mesh: scene.addMesh(late), meshIndex: 5, instanceIndex: 300, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -3, 1] };
        scene.addInstance(lateInstance);
        frame();
        await stand!.finishBatch();
        frame();
        expect(scene.meshStatus(lateInstance.mesh).builtBy).toBe('gpu');
        expect(scene.meshStatus(lateInstance.mesh).placement!.nodeOffset).toBeGreaterThan(0);
        expect(disagreements(readScene(scene), await reference([...geometries, late], [...all, lateInstance]), rays)).toEqual([]);
        scene.dispose();
    });

    it('refuses a custom builder\'s quantized trees instead of quantizing them again', async () => {
        const warnings: string[] = [];
        const { scene, stand, frame } = await setUp({ nodeFormat: 'quantized', holdRefinements: true, warn: (message) => warnings.push(message) });
        const mesh = scene.addMesh(tessellatedQuad(2, 1));
        scene.addInstance({ mesh, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] });
        stand!.quantizedOutputs = true;
        frame();
        await stand!.finishBatch();
        await scene.whenRefined();
        frame();
        expect(scene.meshStatus(mesh).builtBy).toBe('tinybvh');
        expect(scene.meshStatus(mesh).fallbackReason).toMatch(/quantized nodes/);
        expect(warnings.some((message) => /quantized nodes/.test(message))).toBe(true);
        scene.dispose();
    });

    it('leaves a refused or too-deep GPU tree to tinybvh, refuses a too-deep tinybvh tree too, and says why', async () => {
        // Depths (levels): quad(8) is 6 from tinybvh and 7 from the stand-in (one triangle a leaf); quad(16) is 8 and 9.
        const warnings: string[] = [];
        const { scene, stand, frame } = await setUp({ holdRefinements: true, gpuMaxDepth: 6, warn: (message) => warnings.push(message) });
        const big = scene.addMesh(tessellatedQuad(8, 1));
        const small = scene.addMesh(tessellatedQuad(1, 1));
        const refused = scene.addMesh(tessellatedQuad(2, 1));
        const tooDeep = scene.addMesh(tessellatedQuad(16, 1));
        for (const mesh of [big, small, refused, tooDeep]) scene.addInstance({ mesh, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] });
        stand!.refuse.add(refused);
        frame();
        await stand!.finishBatch();
        await scene.whenRefined();
        frame();
        expect(scene.meshStatus(small).builtBy).toBe('gpu');
        expect(scene.meshStatus(big).builtBy).toBe('tinybvh');
        expect(scene.meshStatus(big).fallbackReason).toMatch(/gpuMaxDepth/);
        expect(scene.meshStatus(refused).fallbackReason).toBe('refused by the test');
        // Neither tree of the deepest mesh is placed, so no frame can need more stack than gpuMaxDepth allows.
        expect(scene.meshStatus(tooDeep).builtBy).toBeNull();
        expect(scene.meshStatus(tooDeep).fallbackReason).toMatch(/tinybvh's tree is 8 levels deep, more than gpuMaxDepth \(6\)/);
        expect(warnings.some((message) => /tinybvh's tree is 8 levels deep/.test(message))).toBe(true);
        expect(scene.building).toBe(false);
        frame();
        scene.dispose();
    });

    it("adopt: the load batch's outputs become the buffers (no copy), with a lower peak and the same hits", async () => {
        const peaks: Record<string, number> = {};
        for (const mode of ['copy', 'adopt'] as const) {
            // Uncapped (Infinity): the single-batch path, whose copy mode holds the outputs until the next update.
            const { scene, stand, frame, gpu } = await setUp({ holdRefinements: true, gpuTreeOutputs: mode, gpuMaxBatchBytes: Infinity });
            const { geometries, instances, rays } = sceneOf(6);
            addAll(scene, geometries, instances);
            frame();
            await stand!.finishBatch();
            frame();
            const expected = await reference(geometries, instances);
            expect(disagreements(readScene(scene), expected, rays)).toEqual([]);
            const standIn = gpu.buffers.find((buffer) => buffer.label === 'stand-in nodes')!;
            expect(scene.buffers.nodes === (standIn as unknown as GPUBuffer)).toBe(mode === 'adopt');
            await scene.whenRefined();
            scene.releaseRefinements((mesh) => mesh % 2 === 0);
            frame();
            expect(disagreements(readScene(scene), expected, rays)).toEqual([]);
            scene.releaseRefinements();
            frame();
            frame();
            expect(disagreements(readScene(scene), expected, rays)).toEqual([]);
            expect(scene.counts.swaps).toBe(6);
            peaks[mode] = scene.memory.peak;
            scene.dispose();
            if (mode === 'adopt') expect(standIn.destroyed).toBe(true);
        }
        expect(peaks.adopt).toBeLessThan(peaks.copy);
    });

    it('deforming meshes: refit from each pose (with subgroups), never merged, the deform box in the top level; poses skipped without subgroups', async () => {
        const gpu = createFakeGpu({}, { features: ['subgroups'], kernels: { bvhMergedIdentity: mergedIdentityStandIn } });
        const stand = new StandInGpuBuilder(gpu.device);
        const scene = await SceneTrees.create(gpu.device, { firstTreeBuilder: stand, topLevel: 'cpu', mergeStaticInstances: true });
        const passes: string[] = [];
        const frame = () => {
            const encoder = gpu.createEncoder();
            scene.update(encoder);
            passes.push(...(encoder as unknown as { commands: { dispatch?: { entryPoint: string } }[] }).commands.flatMap((command) => (command.dispatch ? [command.dispatch.entryPoint] : [])));
            gpu.submit(encoder);
        };
        const box = [-5, -5, -5, 5, 5, 5];
        const cloth = scene.addMesh(tessellatedQuad(4, 1), { deforming: { bounds: box } });
        const rock = scene.addMesh(tessellatedQuad(2, 1));
        const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        scene.addInstance({ mesh: cloth, objectToWorld: identity, static: true });
        scene.addInstance({ mesh: rock, objectToWorld: identity, static: true });
        expect(() => scene.setPose(rock, { positionBuffer: gpu.device.createBuffer({ size: 64, usage: 0x80 }), vertexCount: 4 })).toThrow(/not added as deforming/);
        frame();
        // Only the rock merged; the cloth's GPU build is a batch of its own (it reads its layout back).
        expect(scene.counts.mergedGroups).toBe(1);
        expect(scene.meshStatus(cloth).mergedInto).toBeNull();
        expect(stand.requests.map((batch) => batch.length).sort()).toEqual([1, 1]);
        while (stand.waiting > 0) await stand.finishBatch();
        await scene.whenRefined();
        frame();
        expect(scene.meshStatus(cloth).builtBy).toBe('tinybvh');
        // The cloth's top-level box is its deform box.
        const top = readScene(scene).topLevel;
        expect(top.instanceCount).toBe(2);
        const vertices = gpu.device.createBuffer({ size: 25 * 12, usage: 0x80 | 0x4 });
        scene.setPose(cloth, { positionBuffer: vertices, vertexCount: 25, indexBuffer: gpu.device.createBuffer({ size: 32 * 12, usage: 0x80 }), indexFormat: 'uint32' });
        passes.length = 0;
        frame();
        expect(passes).toContain('meshRefitTriangles');
        expect(scene.deformingCounts).toMatchObject({ refitting: 1, refits: 1 });
        passes.length = 0;
        frame();
        expect(passes).not.toContain('meshRefitTriangles');
        expect(scene.memory.meshRefits).toBeGreaterThan(0);
        expect(() => scene.setDeformBounds(rock, box)).toThrow(/not added as deforming|no mesh/);
        expect(() => scene.setDeformBounds(cloth, [1, 0, 0, 0, 0, 0])).toThrow(/not a box/);
        scene.dispose();

        // setDeformBounds moves a deforming mesh's top-level box: a box away from the quad culls it, the old one finds it again.
        // The CPU top level here (the fake GPU runs no kernels); the GPU top level is checked on the three-skinned page.
        for (const topLevel of ['cpu'] as const) {
            const builder = new StandInGpuBuilder(gpu.device);
            const bounded = await SceneTrees.create(gpu.device, { firstTreeBuilder: builder, topLevel });
            const sheet = bounded.addMesh(tessellatedQuad(4, 1), { deforming: { bounds: box } });
            bounded.addInstance({ mesh: sheet, objectToWorld: identity });
            const tick = () => {
                const encoder = gpu.createEncoder();
                bounded.update(encoder);
                gpu.submit(encoder);
            };
            tick();
            while (builder.waiting > 0) await builder.finishBatch();
            await bounded.whenRefined();
            tick();
            tick();
            const ray = { origin: [0.1, 0.2, 3] as const, direction: [0, 0, -1] as const, tMin: 0, tMax: 100 };
            expect(createCpuTracer(readScene(bounded)).closest(ray)?.t, topLevel).toBeCloseTo(3, 5);
            bounded.setDeformBounds(sheet, [10, 10, 10, 11, 11, 11]);
            tick();
            expect(createCpuTracer(readScene(bounded)).closest(ray)).toBeNull();
            bounded.setDeformBounds(sheet, box);
            tick();
            expect(createCpuTracer(readScene(bounded)).closest(ray)?.t).toBeCloseTo(3, 5);
            bounded.dispose();
        }

        // Without subgroups: no refit, the pose is counted as skipped.
        const plain = await setUp({ stand: false });
        const mesh = plain.scene.addMesh(tessellatedQuad(2, 1), { deforming: true });
        plain.scene.addInstance({ mesh, objectToWorld: identity });
        plain.scene.setPose(mesh, { positionBuffer: plain.gpu.device.createBuffer({ size: 64, usage: 0x80 }), vertexCount: 9 });
        expect(plain.scene.deformingCounts.posesSkipped).toBe(1);
        plain.scene.dispose();
    });
    it('throws when a merged-away mesh gets a new instance', async () => {
        const { scene, frame } = await setUp({ stand: false, mergeStaticInstances: true });
        const { meshes } = addAll(scene, ...(({ geometries, instances }) => [geometries, instances] as const)(sceneOf(3)));
        frame();
        expect(scene.meshStatus(meshes[0]).mergedInto).toBe(0);
        expect(() => scene.addInstance({ mesh: meshes[0], objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] })).toThrow(new RegExp(`mesh ${meshes[0]} was merged away.*add the mesh again`));
        expect(scene.counts.instances).toBe(3);
        scene.dispose();
    });

    it('keeps no tree arrays on the heap: placed trees hold numbers only', async () => {
        const { scene, stand, frame } = await setUp({ holdRefinements: true });
        const { geometries, instances } = sceneOf(3);
        addAll(scene, geometries, instances);
        frame();
        await stand!.finishBatch();
        frame();
        await scene.whenRefined();
        scene.releaseRefinements();
        frame();
        const placed = (scene as unknown as { placed: Map<number, object> }).placed;
        expect(placed.size).toBe(3);
        const buffers: string[] = [];
        const walk = (value: unknown, path: string, seen: Set<object>): void => {
            if (typeof value !== 'object' || value === null || seen.has(value)) return;
            seen.add(value);
            if (value instanceof ArrayBuffer || ArrayBuffer.isView(value)) {
                buffers.push(path);
                return;
            }
            for (const [key, child] of Object.entries(value)) walk(child, `${path}.${key}`, seen);
        };
        for (const [slot, tree] of placed) walk(tree, `placed[${slot}]`, new Set());
        expect(buffers).toEqual([]);
        scene.dispose();
    });

    it('reuses removed instance slots on the CPU top level', async () => {
        const { scene, frame } = await setUp({ stand: false });
        const mesh = scene.addMesh(tessellatedQuad(2, 1));
        const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        const first = scene.addInstance({ mesh, objectToWorld: identity });
        const second = scene.addInstance({ mesh, objectToWorld: identity });
        frame();
        scene.removeInstance(first);
        frame();
        expect(scene.addInstance({ mesh, objectToWorld: identity })).toBe(first);
        expect(scene.addInstance({ mesh, objectToWorld: identity })).toBe(second + 1);
        scene.dispose();
    });

    it('adopt: reports the tree an adopted one replaced', async () => {
        // A drift rebuild's batch adopted over the tree it replaces (started by hand: the fake GPU never reads drift back).
        const { scene, stand, frame, events } = await setUp({ holdRefinements: true, gpuTreeOutputs: 'adopt' });
        const mesh = scene.addMesh(tessellatedQuad(3, 1));
        scene.addInstance({ mesh, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] });
        frame();
        await stand!.finishBatch();
        frame();
        expect(events.map((event) => event.replaced)).toEqual([null]);
        const internal = scene as unknown as { startGpuBuilds(builder: unknown, requests: GpuTreeRequest[], kind: { readLayouts: boolean; rebuild: boolean }): boolean };
        internal.startGpuBuilds(stand, [{ slot: mesh, generation: 1, geometry: tessellatedQuad(3, 1) }], { readLayouts: false, rebuild: true });
        await stand!.finishBatch();
        frame();
        expect(events.map((event) => [event.builtBy, event.replaced])).toEqual([['gpu', null], ['gpu', 'gpu']]);
        scene.dispose();
    });

    it("adopt: a deforming mesh's batch that finishes first waits for the main load batch, which is adopted", async () => {
        const gpu = createFakeGpu({}, { features: ['subgroups'], kernels: { bvhMergedIdentity: mergedIdentityStandIn } });
        const builder = new StandInGpuBuilder(gpu.device);
        // Uncapped: with a cap the main batch is adopted between updates and the cloth placed in the same update (the capped tests cover that).
        const deforming = await SceneTrees.create(gpu.device, { firstTreeBuilder: builder, topLevel: 'cpu', gpuTreeOutputs: 'adopt', holdRefinements: true, gpuMaxBatchBytes: Infinity });
        const tick = () => {
            const encoder = gpu.createEncoder();
            deforming.update(encoder);
            gpu.submit(encoder);
        };
        const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        const cloth = deforming.addMesh(tessellatedQuad(2, 1), { deforming: true });
        const rock = deforming.addMesh(tessellatedQuad(6, 1));
        deforming.addInstance({ mesh: cloth, objectToWorld: identity });
        deforming.addInstance({ mesh: rock, objectToWorld: identity });
        tick();
        expect(builder.requests.map((batch) => batch.map((request) => request.slot))).toEqual([[rock], [cloth]]);
        await builder.finishBatch(1);
        tick();
        expect(deforming.meshStatus(cloth).builtBy).toBeNull();
        expect(deforming.building).toBe(true);
        await builder.finishBatch();
        tick();
        const [, mainNodes] = gpu.buffers.filter((buffer) => buffer.label === 'stand-in nodes');
        expect(deforming.buffers.nodes === (mainNodes as unknown as GPUBuffer)).toBe(true);
        expect(deforming.meshStatus(rock).builtBy).toBe('gpu');
        tick();
        expect(deforming.meshStatus(cloth).builtBy).toBe('gpu');
        deforming.dispose();
    });

    it('throws before recording anything when a storage would pass a device limit, so a frame that is not submitted loses nothing', async () => {
        // Nodes may grow to 128 KB, triangles can't pass 200 KB: the big mesh needs 240 KB of triangles.
        const gpu = createFakeGpu({ maxStorageBufferBindingSize: 200 * 1024 });
        const scene = await SceneTrees.create(gpu.device, { gpuFirstTrees: false });
        const frame = () => {
            const encoder = gpu.createEncoder();
            scene.update(encoder);
            gpu.submit(encoder);
        };
        const random = createRandom(5);
        const small = tessellatedQuad(4, 1);
        const first: Instance = { mesh: scene.addMesh(small), meshIndex: 0, instanceIndex: 0, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] };
        scene.addInstance(first);
        frame();
        await scene.whenRefined();
        frame();
        const expected = await reference([small], [first]);
        const rays = randomRays(random, 300, [-2, -2, -2, 2, 2, 2], 1);
        expect(disagreements(readScene(scene), expected, rays)).toEqual([]);

        const big = tessellatedQuad(50, 1);
        scene.addInstance({ mesh: scene.addMesh(big), instanceIndex: 1, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 5, 1] });
        frame();
        await scene.whenRefined();
        const version = scene.version;
        for (let attempt = 0; attempt < 2; attempt++) {
            const encoder = gpu.createEncoder();
            // The caller drops the frame: no submit.
            expect(() => scene.update(encoder)).toThrow(/trace triangles buffer needs maxStorageBufferBindingSize/);
            expect((encoder as unknown as { commands: unknown[] }).commands).toEqual([]);
        }
        expect(scene.version).toBe(version);
        expect(scene.memory.retired).toBe(0);
        expect(disagreements(readScene(scene), expected, rays)).toEqual([]);
        scene.dispose();
    });

    it('indexed: a positions growth past the device limit throws with nothing changed and the mesh still waiting, every update', async () => {
        // 96 KB of vec4 positions for the soup, 64 KB allowed: the positions are placed first, when its builds start.
        const gpu = createFakeGpu({ maxStorageBufferBindingSize: 64 * 1024 });
        const scene = await SceneTrees.create(gpu.device, { gpuFirstTrees: false, triangleLeaves: 'indexed' });
        const frame = () => {
            const encoder = gpu.createEncoder();
            scene.update(encoder);
            gpu.submit(encoder);
        };
        const random = createRandom(6);
        const small = tessellatedQuad(4, 1);
        const first: Instance = { mesh: scene.addMesh(small), meshIndex: 0, instanceIndex: 0, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] };
        scene.addInstance(first);
        frame();
        await scene.whenRefined();
        frame();
        const expected = await reference([small], [first]);
        const rays = randomRays(random, 300, [-2, -2, -2, 2, 2, 2], 1);
        expect(disagreements(readScene(scene), expected, rays)).toEqual([]);

        const big = scene.addMesh(randomTriangleSoup(random, 2000, 2, 0.2));
        const bigInstance = scene.addInstance({ mesh: big, instanceIndex: 1, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 5, 1] });
        const version = scene.version;
        const positions = scene.memory.positions;
        for (let attempt = 0; attempt < 2; attempt++) {
            const encoder = gpu.createEncoder();
            // Each update tries the mesh again: it was not dropped from the new meshes.
            expect(() => scene.update(encoder)).toThrow(/trace positions buffer needs maxStorageBufferBindingSize/);
            expect((encoder as unknown as { commands: unknown[] }).commands).toEqual([]);
            expect(scene.meshStatus(big).builtBy).toBe(null);
        }
        expect(scene.version).toBe(version);
        expect(scene.memory.positions).toBe(positions);
        expect(scene.memory.retired).toBe(0);
        expect(disagreements(readScene(scene), expected, rays)).toEqual([]);

        // Without the mesh the scene carries on.
        scene.removeInstance(bigInstance);
        scene.removeMesh(big);
        frame();
        await scene.whenRefined();
        frame();
        expect(disagreements(readScene(scene), expected, rays)).toEqual([]);
        scene.dispose();
    });

    it('with the GPU top level (fake GPU, gather stand-in): mesh table, records, merged records and slot reuse', async () => {
        const gpu = createFakeGpu({}, { features: ['subgroups'], kernels: { bvhTopLevelGather: gatherStandIn } });
        const scene = await SceneTrees.create(gpu.device, { gpuFirstTrees: false, mergeStaticInstances: true });
        expect(scene.topLevelPath).toBe('gpu');
        const frame = () => {
            const encoder = gpu.createEncoder();
            scene.update(encoder);
            gpu.submit(encoder);
        };
        const settle = async () => {
            frame();
            await scene.whenRefined();
            frame();
            await scene.whenBuilt();
            frame();
        };
        const { geometries, instances } = sceneOf(4);
        // Records by instance index: [root node offset, triangle offset]. The refit doesn't run here, so the top level's boxes aren't checked (the GPU pages trace it).
        const records = () => {
            const top = readScene(scene).topLevel;
            const words = new Uint32Array(top.buffer);
            return new Map(
                Array.from({ length: top.instanceCount }, (_, record) => {
                    const base = TOP_LEVEL_HEADER_WORDS + record * INSTANCE_RECORD_WORDS;
                    return [words[base + INSTANCE_WORD.instanceIndex], { offsets: [words[base + INSTANCE_WORD.rootNodeOffset], words[base + INSTANCE_WORD.triangleOffset]], flags: words[base + INSTANCE_WORD.flags] }] as const;
                }),
            );
        };
        const offsetsOf = (mesh: number) => {
            const placement = scene.meshStatus(mesh).placement!;
            return [placement.nodeOffset, placement.triangleOffset];
        };
        // Meshes 0-1 static (merged into one group), 2-3 dynamic.
        const live: Instance[] = instances.map((instance, index) => ({ ...instance, static: index < 2 }));
        const { meshes, handles } = addAll(scene, geometries, live);
        // A mesh with no instances: placed, but no frame counts its depth.
        const unused = scene.addMesh(tessellatedQuad(2, 1));
        await settle();
        expect(scene.counts.mergedGroups).toBe(1);
        expect(scene.topLevelInstanceCount).toBe(3);
        const first = records();
        expect([...first.keys()].sort()).toEqual([102, 103, MERGED_INSTANCE_INDEX]);
        expect(first.get(102)!.offsets).toEqual(offsetsOf(meshes[2]));
        expect(first.get(103)!.offsets).toEqual(offsetsOf(meshes[3]));
        const group = scene.groupStatus(0).placement!;
        expect(first.get(MERGED_INSTANCE_INDEX)!.offsets).toEqual([group.nodeOffset, group.triangleOffset]);
        expect(first.get(MERGED_INSTANCE_INDEX)!.flags & INSTANCE_FLAG.merged).toBe(INSTANCE_FLAG.merged);
        // The mesh table holds each placed tree's offsets.
        const table = new Uint32Array((((scene as unknown as { gpuTopLevel: { meshTable: GPUBuffer } }).gpuTopLevel.meshTable) as unknown as FakeBuffer).bytes.buffer);
        for (const mesh of [meshes[2], meshes[3]]) {
            const placement = scene.meshStatus(mesh).placement!;
            expect([table[mesh * 8 + 3], table[mesh * 8 + 7]]).toEqual([placement.nodeOffset, placement.triangleOffset]);
        }
        // An unused mesh's depth never reaches the stack check.
        const placed = (scene as unknown as { placed: Map<number, { maxDepth: number }> }).placed;
        placed.set(unused, { ...placed.get(unused)!, maxDepth: 63 });
        frame();

        // A removed instance's slot is reused by the next one.
        scene.removeInstance(handles[3]);
        await settle();
        const again: Instance = { ...live[3], instanceIndex: 400 };
        expect(scene.addInstance({ ...again, mesh: meshes[3] })).toBe(handles[3]);
        await settle();
        expect(scene.topLevelInstanceCount).toBe(3);
        const after = records();
        expect([...after.keys()].sort()).toEqual([102, 400, MERGED_INSTANCE_INDEX]);
        expect(after.get(400)!.offsets).toEqual(offsetsOf(meshes[3]));
        scene.dispose();
    });
});

/** Smaller than any one-mesh batch: every mesh is a GPU batch of its own. */
const SMALL_CAP = 4 * 1024;

/** A capped scene on the real GpuFirstTrees queue, with a stand-in H-PLOC builder. */
async function setUpCapped(options: Partial<SceneTreesOptions> = {}) {
    const gpu: FakeGpu = createFakeGpu({}, { kernels: { bvhMergedIdentity: mergedIdentityStandIn, bvhMergedIdentityIndexed: mergedIdentityIndexedStandIn, bvhqQuantize: quantizeStandIn } });
    const hploc = new FakeHploc(gpu.device);
    const scene = await SceneTrees.create(gpu.device, { gpuMaxBatchBytes: SMALL_CAP, ...options, firstTreeBuilder: new GpuFirstTrees(gpu.device, { hploc, triangleLeaves: options.triangleLeaves }) });
    const frame = () => {
        const encoder = gpu.createEncoder();
        scene.update(encoder);
        gpu.submit(encoder);
    };
    return { gpu, hploc, scene, frame };
}

/** Rejects instead of hanging when `promise` takes longer than a test should. */
function withinTime<T>(promise: Promise<T>, what: string, ms = 5000): Promise<T> {
    return Promise.race([promise, new Promise<T>((_, reject) => setTimeout(() => reject(new Error(`${what} did not resolve (deadlock?)`)), ms))]);
}

describe('SceneTrees with indexed leaves', () => {
    for (const positionLayout of ['vec4', 'packed'] as const) {
        it(`GPU trees first, tinybvh swapped in, merged groups with their side tables, runtime adds and removals: the same hits as records (${positionLayout})`, async () => {
            const { scene, stand, frame } = await setUp({ holdRefinements: true, mergeStaticInstances: true, triangleLeaves: 'indexed', positionLayout });
            const { geometries, instances, rays } = sceneOf(5);
            const twin: Instance = { mesh: -1, meshIndex: 4, instanceIndex: 200, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1] };
            const all = [...instances, twin];
            const { meshes } = addAll(scene, geometries, all);
            frame();
            expect(scene.counts.mergedGroups).toBe(1);
            const expected = await reference(geometries, all);
            await stand!.finishBatch();
            frame();
            expect(scene.counts).toMatchObject({ onGpuTrees: 2, onTinybvhTrees: 0 });
            expect(disagreements(readScene(scene), expected, rays)).toEqual([]);
            await scene.whenRefined();
            scene.releaseRefinements((mesh) => mesh === -1);
            frame();
            expect(scene.groupStatus(0).builtBy).toBe('tinybvh');
            expect(disagreements(readScene(scene), expected, rays)).toEqual([]);
            scene.releaseRefinements();
            frame();
            expect(scene.counts).toMatchObject({ onGpuTrees: 0, onTinybvhTrees: 2, swaps: 2 });
            expect(disagreements(readScene(scene), expected, rays)).toEqual([]);

            // A mesh added later, then removed; its positions range is freed and reused.
            const late = tessellatedQuad(4, 0.5);
            const lateInstance: Instance = { mesh: scene.addMesh(late), meshIndex: 5, instanceIndex: 300, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -3, 1] };
            const lateHandle = scene.addInstance(lateInstance);
            frame();
            await stand!.finishBatch();
            frame();
            expect(disagreements(readScene(scene), await reference([...geometries, late], [...all, lateInstance]), rays)).toEqual([]);
            const memory = scene.memory;
            expect(memory.positions).toBeGreaterThan(0);
            expect(memory.total).toBe(memory.resident + memory.transient);
            scene.removeInstance(lateHandle);
            scene.removeMesh(lateInstance.mesh);
            frame();
            expect(disagreements(readScene(scene), expected, rays)).toEqual([]);
            expect(scene.meshStatus(meshes[4]).builtBy).toBe('tinybvh');
            scene.dispose();
        });
    }

    it('is smaller than records, welds by exact bits, and refuses the adopt lever', async () => {
        const sizes = [];
        for (const triangleLeaves of ['records', 'indexed'] as const) {
            const { scene, frame } = await setUp({ stand: false, triangleLeaves });
            const { geometries, instances, rays } = sceneOf(4);
            addAll(scene, geometries, instances);
            frame();
            await scene.whenRefined();
            frame();
            expect(disagreements(readScene(scene), await reference(geometries, instances), rays)).toEqual([]);
            sizes.push(scene.memory.trianglesInUse);
            scene.dispose();
        }
        // Meshes of 3-6 cells a side, two triangles a cell: 16 bytes each and a 16-byte header per mesh, against 48.
        const triangles = (9 + 16 + 25 + 36) * 2;
        expect(sizes).toEqual([triangles * 48, triangles * 16 + 4 * 16]);
        await expect(setUp({ stand: false, triangleLeaves: 'indexed', gpuTreeOutputs: 'adopt' })).rejects.toThrow(/gpuTreeOutputs: 'copy'/);
    });
});

describe('SceneTrees with quantized nodes', () => {
    for (const triangleLeaves of ['records', 'indexed'] as const) {
        it(`GPU trees packed on the way in, tinybvh's quantized in the workers, merged groups, runtime adds and removals: the same hits as f32 (${triangleLeaves})`, async () => {
            const { scene, stand, frame } = await setUp({ holdRefinements: true, mergeStaticInstances: true, triangleLeaves, nodeFormat: 'quantized' });
            expect(scene.nodeFormat).toBe('quantized');
            expect(scene.traceBuffers.nodeFormat).toBe('quantized');
            const { geometries, instances, rays } = sceneOf(5);
            const twin: Instance = { mesh: -1, meshIndex: 4, instanceIndex: 200, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1] };
            const all = [...instances, twin];
            addAll(scene, geometries, all);
            frame();
            const expected = await reference(geometries, all);
            await stand!.finishBatch();
            frame();
            expect(scene.counts).toMatchObject({ onGpuTrees: 2, onTinybvhTrees: 0 });
            expect(disagreements(readScene(scene), expected, rays)).toEqual([]);
            scene.releaseRefinements();
            await scene.whenRefined();
            frame();
            expect(scene.counts).toMatchObject({ onGpuTrees: 0, onTinybvhTrees: 2, swaps: 2 });
            expect(disagreements(readScene(scene), expected, rays)).toEqual([]);

            const late = tessellatedQuad(4, 0.5);
            const lateInstance: Instance = { mesh: scene.addMesh(late), meshIndex: 5, instanceIndex: 300, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -3, 1] };
            const lateHandle = scene.addInstance(lateInstance);
            frame();
            await stand!.finishBatch();
            frame();
            expect(disagreements(readScene(scene), await reference([...geometries, late], [...all, lateInstance]), rays)).toEqual([]);
            scene.removeInstance(lateHandle);
            scene.removeMesh(lateInstance.mesh);
            frame();
            expect(disagreements(readScene(scene), expected, rays)).toEqual([]);
            scene.dispose();
        });
    }

    it('takes half the node bytes plus a frame slot per tree, and refuses the adopt lever', async () => {
        const sizes = [];
        let trees = 0;
        for (const nodeFormat of ['f32', 'quantized'] as const) {
            const { scene, frame } = await setUp({ stand: false, nodeFormat });
            const { geometries, instances, rays } = sceneOf(4);
            addAll(scene, geometries, instances);
            frame();
            await scene.whenRefined();
            frame();
            expect(disagreements(readScene(scene), await reference(geometries, instances), rays)).toEqual([]);
            sizes.push(scene.memory.nodesInUse);
            trees = scene.counts.onTinybvhTrees;
            scene.dispose();
        }
        expect(sizes[1]).toBe(sizes[0] / 2 + trees * 32);
        await expect(setUp({ stand: false, nodeFormat: 'quantized', gpuTreeOutputs: 'adopt' })).rejects.toThrow(/gpuTreeOutputs: 'copy'/);
    });
});

describe('SceneTrees with gpuMaxBatchBytes (the real queue, a stand-in H-PLOC)', () => {
    for (const variant of [
        { name: 'copy, holdRefinements', options: { holdRefinements: true }, allGpu: true },
        { name: 'adopt, holdRefinements', options: { holdRefinements: true, gpuTreeOutputs: 'adopt' as const }, allGpu: true },
        { name: 'adopt, merged, holdRefinements', options: { holdRefinements: true, gpuTreeOutputs: 'adopt' as const, mergeStaticInstances: true }, allGpu: true },
        { name: 'adopt', options: { gpuTreeOutputs: 'adopt' as const }, allGpu: false },
        { name: 'copy', options: {}, allGpu: false },
        { name: 'copy, quantized, holdRefinements', options: { holdRefinements: true, nodeFormat: 'quantized' as const }, allGpu: true },
        { name: 'copy, quantized', options: { nodeFormat: 'quantized' as const }, allGpu: false },
    ]) {
        it(`${variant.name}: frame(); await whenBuilt(); frame() places every first tree, batch after batch without updates in between`, async () => {
            const { scene, frame, hploc } = await setUpCapped(variant.options);
            const { geometries, instances, rays } = sceneOf(5);
            // Some instances not static, so the merged variant keeps instanced meshes (more than one batch) too.
            const mixed = instances.map((instance, index) => ({ ...instance, static: index % 2 === 0 }));
            addAll(scene, geometries, mixed);
            frame();
            await withinTime(scene.whenBuilt(), 'whenBuilt');
            frame();
            expect(hploc.violations).toEqual([]);
            expect(hploc.maxActive).toBe(1);
            expect(scene.counts.gpuBatches).toBeGreaterThan(1);
            expect(scene.counts.waitingForFirstTree).toBe(0);
            if (variant.allGpu) expect(scene.counts).toMatchObject({ onTinybvhTrees: 0, gpuTreesBuilt: scene.counts.onGpuTrees });
            expect(disagreements(readScene(scene), await reference(geometries, mixed), rays)).toEqual([]);
            // The swap, then nothing left behind: no outputs, no reserve, no pool.
            await withinTime(scene.whenRefined(), 'whenRefined');
            scene.releaseRefinements();
            frame();
            frame();
            expect(scene.counts.onGpuTrees).toBe(0);
            expect(disagreements(readScene(scene), await reference(geometries, mixed), rays)).toEqual([]);
            expect(scene.memory).toMatchObject({ gpuBuildOutputs: 0, gpuBuilding: 0, reserved: 0 });
            expect(hploc.poolsAlive).toBe(0);
            scene.dispose();
        });
    }

    it('reserves room for the batches to come, gives back the room of a removed mesh and of one tinybvh got to first, and loses no tree', async () => {
        const { scene, frame, hploc } = await setUpCapped({ holdRefinements: true });
        hploc.manual = true;
        const { instances, rays } = sceneOf(5);
        // Big enough that batch 0 grows the storages past their first size (the growth that reserves).
        const geometries = instances.map((_, index) => tessellatedQuad(40 + 4 * index, 1));
        const { meshes, handles } = addAll(scene, geometries, instances);
        frame();
        expect(hploc.batches.length).toBe(1);
        await hploc.finishNext();
        // Batch 0 is stored (no update since), and the next batch started: its outputs were let go first.
        expect(hploc.events.filter((event) => /^(start|let go)/.test(event))).toEqual(['start 0', 'let go 0', 'start 1']);
        const triangles = (index: number) => scene.meshStatus(meshes[index]).placement;
        expect(triangles(0)).toBeNull();
        const pendingTriangles = geometries.slice(1).reduce((sum, geometry) => sum + geometry.positions.length / 9, 0);
        // Triangles of meshes 1-4 exactly, plus their nodes at batch 0's nodes per triangle.
        expect(scene.memory.reserved).toBeGreaterThanOrEqual(pendingTriangles * 48);
        const reservedBefore = scene.memory.reserved;
        const buffersBefore = { nodes: scene.buffers.nodes, triangles: scene.buffers.triangles };

        // Mesh 4 goes before its batch starts: its room is given back now.
        scene.removeInstance(handles[4]);
        scene.removeMesh(meshes[4]);
        const shareOf4 = reservedBefore - scene.memory.reserved;
        expect(shareOf4).toBeGreaterThanOrEqual((geometries[4].positions.length / 9) * 48);

        // Mesh 1's batch builds; meanwhile tinybvh's tree of mesh 3 is placed first (its hold released): its room goes too.
        // (whenRefined would wait for the GPU batches too, which wait for the test here.)
        for (let wait = 0; wait < 500 && scene.counts.tinybvhBuilding > 0; wait++) await settle(1);
        scene.releaseRefinements((mesh) => mesh === meshes[3]);
        frame();
        expect(scene.meshStatus(meshes[3]).builtBy).toBe('tinybvh');
        const afterTinybvh = scene.memory.reserved;
        expect(afterTinybvh).toBeLessThan(reservedBefore - shareOf4);

        while (hploc.waitingBatches > 0) await hploc.finishNext();
        await settle();
        frame();
        // Every remaining batch (and tinybvh's tree, in mesh 3's room) went into room set aside then: no growth since batch 0.
        expect(scene.buffers.nodes).toBe(buffersBefore.nodes);
        expect(scene.buffers.triangles).toBe(buffersBefore.triangles);
        expect(scene.memory.reserved).toBe(0);
        expect(scene.meshStatus(meshes[0]).builtBy).toBe('gpu');
        expect(scene.meshStatus(meshes[1]).builtBy).toBe('gpu');
        expect(scene.meshStatus(meshes[2]).builtBy).toBe('gpu');
        // Batches in order, each after the one before let go; mesh 3's GPU tree was dropped, not lost (tinybvh's is placed).
        const order = hploc.events.filter((event) => /^(start|let go)/.test(event));
        expect(order).toEqual(order.map((_, index) => (index % 2 === 0 ? `start ${index / 2}` : `let go ${(index - 1) / 2}`)));
        expect(hploc.violations).toEqual([]);
        const kept = instances.slice(0, 4);
        expect(disagreements(readScene(scene), await reference(geometries.slice(0, 4), kept), rays)).toEqual([]);
        scene.dispose();
    });

    for (const mode of ['copy', 'adopt'] as const) {
        it(`${mode}: when a batch cannot be stored between updates, whenBuilt and the next update agree (every mesh gets a tree), and the stall costs one update`, async () => {
            const { scene, frame, hploc } = await setUpCapped({ holdRefinements: true, gpuTreeOutputs: mode });
            const { instances, rays } = sceneOf(5);
            const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
            // A small mesh first, so the buffers hold something (adopt: they are its batch's outputs).
            const seed = tessellatedQuad(2, 1);
            scene.addInstance({ mesh: scene.addMesh(seed), instanceIndex: 99, objectToWorld: identity });
            frame();
            await withinTime(scene.whenBuilt(), 'whenBuilt (seed)');
            frame();
            // Then meshes too big for the room left, each a batch of its own.
            const geometries = instances.slice(0, 4).map((_, index) => tessellatedQuad(40 + 4 * index, 1));
            const { meshes } = addAll(scene, geometries, instances.slice(0, 4));
            frame();
            // As if that update had recorded storage writes its frame has still to run (a refit does, every frame): growing between updates isn't safe.
            (scene as unknown as { encoderWritesStorage: boolean }).encoderWritesStorage = true;
            await withinTime(scene.whenBuilt(), 'whenBuilt');
            expect((scene as unknown as { outputsWaitingForUpdate: Set<unknown> }).outputsWaitingForUpdate.size).toBe(1);
            expect(scene.counts.waitingForFirstTree).toBe(0);
            // One update: what whenBuilt counted is what it places. Mesh 0 on its (stalled) GPU tree, the meshes behind on tinybvh's.
            frame();
            expect(meshes.map((mesh) => scene.meshStatus(mesh).builtBy)).toEqual(['gpu', 'tinybvh', 'tinybvh', 'tinybvh']);
            // That same update let the stalled batch's outputs go (its copy in a submit of its own): the next batch started without another.
            await settle();
            expect(hploc.events).toContain('let go 1');
            expect(hploc.events).toContain('start 2');
            for (let update = 0; update < 20 && scene.counts.gpuBatches < 5; update++) {
                frame();
                await settle(2);
            }
            frame();
            expect(scene.counts.gpuBatches).toBe(5);
            expect(hploc.violations).toEqual([]);
            const all = [{ mesh: -1, meshIndex: 4, instanceIndex: 99, objectToWorld: identity }, ...instances.slice(0, 4)];
            expect(disagreements(readScene(scene), await reference([...geometries, seed], all), rays)).toEqual([]);
            scene.dispose();
        });
    }

    it('grows between updates into empty buffers even when the last update recorded storage writes (nothing to lose), so the load never stalls', async () => {
        const { scene, frame } = await setUpCapped({ holdRefinements: true });
        const { instances } = sceneOf(3);
        const geometries = instances.map((_, index) => tessellatedQuad(40 + 4 * index, 1));
        addAll(scene, geometries, instances);
        frame();
        (scene as unknown as { encoderWritesStorage: boolean }).encoderWritesStorage = true;
        await withinTime(scene.whenBuilt(), 'whenBuilt');
        expect((scene as unknown as { outputsWaitingForUpdate: Set<unknown> }).outputsWaitingForUpdate.size).toBe(0);
        frame();
        expect(scene.counts).toMatchObject({ onGpuTrees: 3, onTinybvhTrees: 0 });
        scene.dispose();
    });

    it('caps at 256 MiB by default; Infinity turns the cap off, and a cap must be positive', async () => {
        expect(DEFAULT_GPU_MAX_BATCH_BYTES).toBe(256 * 1024 * 1024);
        const asked: (number | undefined)[] = [];
        for (const gpuMaxBatchBytes of [undefined, Infinity, 1 << 20]) {
            const gpu = createFakeGpu();
            const stand = new StandInGpuBuilder(gpu.device);
            const recording: FirstTreeBuilder = {
                prepare: () => stand.prepare(),
                build: (requests, options) => (asked.push(options?.maxBatchBytes), stand.build(requests)),
            };
            const scene = await SceneTrees.create(gpu.device, { firstTreeBuilder: recording, gpuMaxBatchBytes });
            scene.addMesh(tessellatedQuad(2, 1));
            scene.update(gpu.createEncoder());
            scene.dispose();
        }
        expect(asked).toEqual([DEFAULT_GPU_MAX_BATCH_BYTES, undefined, 1 << 20]);
        await expect(SceneTrees.create(createFakeGpu().device, { gpuMaxBatchBytes: 0 })).rejects.toThrow(/positive number of bytes/);
    });

    it('frees the slots of meshes whose geometry the queue reads later as their tinybvh builds start (no entry left behind)', async () => {
        const { scene, frame } = await setUpCapped({ holdRefinements: true });
        const { geometries, instances } = sceneOf(3);
        addAll(scene, geometries, instances);
        frame();
        expect((scene as unknown as { gpuReadsGeometryLater: Set<number> }).gpuReadsGeometryLater.size).toBe(0);
        await withinTime(scene.whenBuilt(), 'whenBuilt');
        scene.dispose();
    });
});

/** Closest hits against the brute-force reference, material and normal included when both name the same triangle. */
function referenceMismatches(scene: SceneTrees, geometries: readonly MeshGeometry[], instances: readonly Instance[], rays: readonly TraceRay[]): { mismatches: string[]; checked: number } {
    const tracer = createCpuTracer(readScene(scene));
    const expected = buildReferenceScene(geometries, instances.map((instance) => ({ ...instance, instanceIndex: instance.instanceIndex! })));
    const tally = newTally();
    let checked = 0;
    rays.forEach((ray, index) => {
        const hit = tracer.closest(ray);
        const want = traceClosestReference(expected, ray);
        addToTally(tally, compareClosest(hit && { hit: true, ...hit }, want, ray), `ray ${index}`);
        if (hit && want && hit.triangleIndex === want.triangleIndex && hit.instanceIndex === want.instanceIndex) checked++;
    });
    return { mismatches: tally.examples.slice(0, tally.mismatches), checked };
}

describe('SceneTrees with per-triangle materials', () => {
    for (const triangleLeaves of ['records', 'indexed'] as const) {
        it(`${triangleLeaves}: GPU trees, tinybvh's and merged groups report each triangle's material, else the instance's`, async () => {
            const { scene, stand, frame } = await setUp({ holdRefinements: true, mergeStaticInstances: true, triangleLeaves });
            const { geometries: plain, instances, rays } = sceneOf(5);
            // Materials on meshes 0, 1 (merged) and 4 (instanced twice); every fourth triangle has none and takes its instance's.
            const geometries = plain.map((geometry, index) =>
                index === 2 || index === 3 ? geometry : { ...geometry, triangleMaterials: Array.from({ length: geometry.indices!.length / 3 }, (_, k) => (k % 4 === 3 ? 0xffff : 20 + ((k + index) % 5))) },
            );
            const twin: Instance = { mesh: -1, meshIndex: 4, instanceIndex: 200, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1], material: 9 };
            const all = [...instances, twin];
            addAll(scene, geometries, all);
            frame();
            expect(scene.counts.mergedGroups).toBe(1);
            await stand!.finishBatch();
            frame();
            expect(scene.counts).toMatchObject({ onGpuTrees: 2, onTinybvhTrees: 0 });
            const many = [...rays, ...randomRays(createRandom(3), 3000, [-6, -6, -6, 6, 6, 6], 1)];
            const onGpu = referenceMismatches(scene, geometries, all, many);
            expect(onGpu.mismatches).toEqual([]);
            expect(onGpu.checked).toBeGreaterThan(100);
            await scene.whenRefined();
            scene.releaseRefinements();
            frame();
            expect(scene.counts).toMatchObject({ onGpuTrees: 0, onTinybvhTrees: 2 });
            const onTinybvh = referenceMismatches(scene, geometries, all, many);
            expect(onTinybvh.mismatches).toEqual([]);
            expect(onTinybvh.checked).toBeGreaterThan(100);
            expect(() => scene.addMesh({ ...plain[0], triangleMaterials: [70000] })).toThrow(/triangleMaterials/);
            scene.dispose();
        });
    }
});

/** Each triangle's material: every fourth none (it takes its instance's), the others from `seed`. */
function withMaterials(geometry: MeshGeometry, seed: number): MeshGeometry {
    const count = (geometry.indices ? geometry.indices.length : geometry.positions.length / 3) / 3;
    return { ...geometry, triangleMaterials: Array.from({ length: count }, (_, k) => (k % 4 === 3 ? 0xffff : seed + (k % 7))) };
}

describe('SceneTrees with per-triangle materials: the indexed material table through growth, slot reuse and capped batches', () => {
    const at = (x: number, y = 0, z = 0) => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];
    // Down onto the quads (z = 0, along x), with some random ones.
    const random = createRandom(21);
    const rays: TraceRay[] = [
        ...Array.from({ length: 2000 }, () => ({ origin: [-8 + 16 * random(), -1 + 2 * random(), 5] as const, direction: [0, 0, -1] as const, tMin: 0, tMax: 100 })),
        ...randomRays(random, 500, [-8, -8, -8, 8, 8, 8], 1),
    ];

    for (const builder of ['tinybvh', 'gpu'] as const) {
        it(`${builder}: a mesh added after placement forces a growth; the placed table and the new one both hold`, async () => {
            const { scene, stand, frame } = await setUp({ stand: builder === 'gpu', holdRefinements: builder === 'gpu', triangleLeaves: 'indexed' });
            const first = withMaterials(tessellatedQuad(3, 1), 10);
            const a = scene.addMesh(first);
            const instances: Instance[] = [{ mesh: a, meshIndex: 0, instanceIndex: 1, objectToWorld: at(-3), material: 5 }];
            scene.addInstance(instances[0]);
            frame();
            if (stand) await stand.finishBatch();
            else await scene.whenRefined();
            frame();
            const before = scene.buffers.triangles;
            // Far bigger than what the storages hold: they grow, copying mesh a's table and header.
            const second = withMaterials(tessellatedQuad(60, 1), 30);
            const b = scene.addMesh(second);
            const added: Instance = { mesh: b, meshIndex: 1, instanceIndex: 2, objectToWorld: at(3, 0, 0.5), material: 6 };
            scene.addInstance(added);
            instances.push(added);
            frame();
            if (stand) await stand.finishBatch();
            else await scene.whenRefined();
            frame();
            expect(scene.buffers.triangles).not.toBe(before);
            expect(scene.meshStatus(b).builtBy).toBe(builder);
            const result = referenceMismatches(scene, [first, second], instances, rays);
            expect(result.mismatches).toEqual([]);
            expect(result.checked).toBeGreaterThan(200);
            scene.dispose();
        });
    }

    it('a removed mesh slot reused by meshes with and without a table: each reports its own materials', async () => {
        const { scene, frame } = await setUp({ stand: false, triangleLeaves: 'indexed' });
        const plain = tessellatedQuad(4, 1);
        const tabled = withMaterials(tessellatedQuad(4, 1), 40);
        const keep = withMaterials(tessellatedQuad(3, 1), 10);
        const settleScene = async () => {
            frame();
            await scene.whenRefined();
            frame();
        };
        const k = scene.addMesh(keep);
        const gone = scene.addMesh(tabled);
        const kept: Instance = { mesh: k, meshIndex: 0, instanceIndex: 1, objectToWorld: at(-3) };
        scene.addInstance(kept);
        const goneInstance = scene.addInstance({ mesh: gone, instanceIndex: 2, objectToWorld: at(3), material: 7 });
        await settleScene();
        // The tabled mesh goes; a mesh without a table takes its slot (and its room).
        scene.removeInstance(goneInstance);
        scene.removeMesh(gone);
        await settleScene();
        const reused = scene.addMesh(plain);
        expect(reused).toBe(gone);
        const plainInstance: Instance = { mesh: reused, meshIndex: 1, instanceIndex: 3, objectToWorld: at(3), material: 8 };
        const plainHandle = scene.addInstance(plainInstance);
        await settleScene();
        let result = referenceMismatches(scene, [keep, plain], [kept, plainInstance], rays);
        expect(result.mismatches).toEqual([]);
        expect(result.checked).toBeGreaterThan(200);
        // And back: a tabled mesh in the slot of one without.
        scene.removeInstance(plainHandle);
        scene.removeMesh(reused);
        await settleScene();
        const again = scene.addMesh(withMaterials(plain, 50));
        expect(again).toBe(gone);
        const againInstance: Instance = { mesh: again, meshIndex: 1, instanceIndex: 4, objectToWorld: at(3) };
        scene.addInstance(againInstance);
        await settleScene();
        result = referenceMismatches(scene, [keep, withMaterials(plain, 50)], [kept, againInstance], rays);
        expect(result.mismatches).toEqual([]);
        expect(result.checked).toBeGreaterThan(200);
        scene.dispose();
    });

    for (const mode of ['copy', 'adopt'] as const) {
        it(`gpuMaxBatchBytes (${mode}): capped batches stored between updates keep each mesh's table`, async () => {
            const { scene, frame, hploc } = await setUpCapped({ holdRefinements: true, gpuTreeOutputs: mode, triangleLeaves: mode === 'adopt' ? 'records' : 'indexed' });
            const { instances } = sceneOf(5);
            const geometries = instances.map((_, index) => (index === 2 ? tessellatedQuad(30 + 4 * index, 1) : withMaterials(tessellatedQuad(30 + 4 * index, 1), 10 * index)));
            const live = instances.map((instance, index) => ({ ...instance, static: false, objectToWorld: at(-6 + 3 * index, 0, 0.1 * index) }));
            // Count the between-update stores (copy: the copies; adopt: the adoption or copies).
            const target = scene as unknown as { copyBetweenUpdates: (...args: unknown[]) => unknown };
            const copy = target.copyBetweenUpdates.bind(scene);
            let stored = 0;
            target.copyBetweenUpdates = (...args) => {
                const result = copy(...args);
                if (result) stored++;
                return result;
            };
            addAll(scene, geometries, live);
            frame();
            await withinTime(scene.whenBuilt(), 'whenBuilt');
            frame();
            expect(hploc.violations).toEqual([]);
            expect(scene.counts.gpuBatches).toBeGreaterThan(1);
            if (mode === 'copy') expect(stored).toBeGreaterThan(0);
            expect(scene.counts).toMatchObject({ onTinybvhTrees: 0, onGpuTrees: 5 });
            const onGpu = referenceMismatches(scene, geometries, live, rays);
            expect(onGpu.mismatches).toEqual([]);
            expect(onGpu.checked).toBeGreaterThan(200);
            await withinTime(scene.whenRefined(), 'whenRefined');
            scene.releaseRefinements();
            frame();
            const onTinybvh = referenceMismatches(scene, geometries, live, rays);
            expect(onTinybvh.mismatches).toEqual([]);
            scene.dispose();
        });
    }

    it('setPoseCpu on a GPU-built tree keeps the table (indexed, with subgroups, the stand-in builder)', async () => {
        const gpu = createFakeGpu({}, { features: ['subgroups'], kernels: { bvhMergedIdentity: mergedIdentityStandIn, bvhMergedIdentityIndexed: mergedIdentityIndexedStandIn } });
        const stand = new StandInGpuBuilder(gpu.device, 'indexed');
        const scene = await SceneTrees.create(gpu.device, { firstTreeBuilder: stand, holdRefinements: true, triangleLeaves: 'indexed', topLevel: 'cpu' });
        const frame = () => {
            const encoder = gpu.createEncoder();
            scene.update(encoder);
            gpu.submit(encoder);
        };
        const geometry = withMaterials(tessellatedQuad(5, 1), 60);
        const cloth = scene.addMesh(geometry, { deforming: true });
        const instance: Instance = { mesh: cloth, meshIndex: 0, instanceIndex: 1, objectToWorld: at(0), material: 3 };
        scene.addInstance(instance);
        frame();
        await stand.finishBatch();
        frame();
        expect(scene.meshStatus(cloth).builtBy).toBe('gpu');
        const pose = geometry.positions.map((value, index) => (index % 3 === 2 ? value + 0.4 * Math.sin(index) : value));
        scene.setPoseCpu(cloth, pose);
        frame();
        expect(scene.deformingCounts.cpuPoses).toBe(1);
        const result = referenceMismatches(scene, [{ ...geometry, positions: pose }], [instance], rays);
        expect(result.mismatches).toEqual([]);
        expect(result.checked).toBeGreaterThan(100);
        scene.dispose();
    });
});

describe('SceneTrees.setPoseCpu', () => {
    const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    /** The quad moved by (5, 0, 1): a pose outside the rest pose's box. */
    const posed = (geometry: MeshGeometry): Float32Array => geometry.positions.map((value, index) => value + (index % 3 === 0 ? 5 : index % 3 === 2 ? 1 + 0.1 * Math.sin(index) : 0));
    const down = (x: number, y: number) => ({ origin: [x, y, 3] as const, direction: [0, 0, -1] as const, tMin: 0, tMax: 100 });

    for (const triangleLeaves of ['records', 'indexed'] as const) {
        for (const nodeFormat of ['f32', 'quantized'] as const) {
            it(`without subgroups (${triangleLeaves}, ${nodeFormat}): the CPU walk over the uploaded buffers hits the posed geometry, and a skipped GPU pose warns once`, async () => {
                const warnings: string[] = [];
                const { gpu, scene, frame } = await setUp({ stand: false, triangleLeaves, nodeFormat, warn: (message) => warnings.push(message) });
                const geometry = { ...tessellatedQuad(6, 1), triangleMaterials: Array.from({ length: 72 }, (_, k) => k % 3) };
                const still = tessellatedQuad(3, 1);
                const cloth = scene.addMesh(geometry, { deforming: true });
                const rock = scene.addMesh(still);
                const instances: Instance[] = [
                    { mesh: cloth, meshIndex: 0, instanceIndex: 1, objectToWorld: identity, material: 4 },
                    { mesh: rock, meshIndex: 1, instanceIndex: 2, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -4, 0, 0, 1] },
                ];
                for (const instance of instances) scene.addInstance(instance);
                // A pose given before the tree exists is written when it arrives.
                const first = posed(geometry);
                scene.setPoseCpu(cloth, first);
                frame();
                await scene.whenRefined();
                frame();
                expect(scene.deformingCounts.cpuPoses).toBe(1);
                const rays = randomRays(createRandom(5), 3000, [-6, -3, -3, 8, 3, 3], 1);
                const check = (pose: Float32Array) => {
                    const result = referenceMismatches(scene, [{ ...geometry, positions: pose }, still], instances, rays);
                    expect(result.mismatches).toEqual([]);
                    expect(result.checked).toBeGreaterThan(50);
                };
                check(first);
                const tracer = () => createCpuTracer(readScene(scene));
                // The top-level box moved with the pose: the rest pose's place is empty, the posed one is hit.
                expect(tracer().closest(down(0.1, 0.2))).toBeNull();
                expect(tracer().closest(down(5.1, 0.2))?.t).toBeCloseTo(2, 0);

                // A GPU pose is skipped without subgroups, with one warning that names setPoseCpu.
                const vertices = gpu.device.createBuffer({ size: 49 * 12, usage: 0x80 | 0x4 });
                scene.setPose(cloth, { positionBuffer: vertices, vertexCount: 49 });
                scene.setPose(cloth, { positionBuffer: vertices, vertexCount: 49 });
                expect(scene.deformingCounts.posesSkipped).toBe(2);
                expect(warnings.filter((message) => message.includes('setPoseCpu'))).toHaveLength(1);

                // Back to rest, then another pose: each is written at the next update.
                scene.setPoseCpu(cloth, geometry.positions);
                frame();
                check(geometry.positions);
                expect(tracer().closest(down(0.1, 0.2))?.t).toBeCloseTo(3, 5);
                const second = geometry.positions.map((value, index) => (index % 3 === 2 ? value - 0.5 + 0.3 * Math.cos(index) : value));
                scene.setPoseCpu(cloth, second);
                frame();
                check(second);
                expect(scene.deformingCounts.cpuPoses).toBe(3);
                // Nothing due: nothing written.
                frame();
                expect(scene.deformingCounts.cpuPoses).toBe(3);

                expect(() => scene.setPoseCpu(rock, still.positions)).toThrow(/not added as deforming/);
                expect(() => scene.setPoseCpu(cloth, new Float32Array(9))).toThrow(/49 vertices/);
                scene.dispose();
            });
        }
    }

    for (const triangleLeaves of ['records', 'indexed'] as const) {
        it(`without subgroups, on a GPU-built first tree (${triangleLeaves}): the deforming mesh's batch reads its layout back, and the pose is written`, async () => {
            const { scene, stand, frame } = await setUp({ holdRefinements: true, triangleLeaves });
            const geometry = { ...tessellatedQuad(6, 1), triangleMaterials: Array.from({ length: 72 }, (_, k) => k % 3) };
            const still = tessellatedQuad(3, 1);
            const cloth = scene.addMesh(geometry, { deforming: true });
            const rock = scene.addMesh(still);
            const instances: Instance[] = [
                { mesh: cloth, meshIndex: 0, instanceIndex: 1, objectToWorld: identity, material: 4 },
                { mesh: rock, meshIndex: 1, instanceIndex: 2, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -4, 0, 0, 1] },
            ];
            for (const instance of instances) scene.addInstance(instance);
            const pose = posed(geometry);
            scene.setPoseCpu(cloth, pose);
            frame();
            // The deforming mesh gets a batch of its own, with readLayouts, though there is no GPU refit.
            expect(stand!.requests.map((batch) => batch.map((request) => request.slot))).toEqual([[rock], [cloth]]);
            expect(stand!.readLayouts).toEqual([false, true]);
            await stand!.finishBatch();
            await stand!.finishBatch();
            frame();
            expect(scene.counts).toMatchObject({ onGpuTrees: 2, onTinybvhTrees: 0 });
            expect(scene.meshStatus(cloth).builtBy).toBe('gpu');
            expect(scene.deformingCounts.cpuPoses).toBe(1);
            const result = referenceMismatches(scene, [{ ...geometry, positions: pose }, still], instances, randomRays(createRandom(5), 3000, [-6, -3, -3, 8, 3, 3], 1));
            expect(result.mismatches).toEqual([]);
            expect(result.checked).toBeGreaterThan(50);
            expect(createCpuTracer(readScene(scene)).closest(down(0.1, 0.2))).toBeNull();
            scene.dispose();
        });
    }

    it('a GPU tree placed without its layout (a failed readback) warns once, and the pose waits for the next tree that has one', async () => {
        const warnings: string[] = [];
        const { scene, stand, frame } = await setUp({ holdRefinements: true, warn: (message) => warnings.push(message) });
        stand!.failLayouts = true;
        const geometry = tessellatedQuad(4, 1);
        const cloth = scene.addMesh(geometry, { deforming: true });
        scene.addInstance({ mesh: cloth, objectToWorld: identity });
        scene.setPoseCpu(cloth, posed(geometry));
        frame();
        await stand!.finishBatch();
        frame();
        frame();
        expect(scene.meshStatus(cloth).builtBy).toBe('gpu');
        expect(scene.deformingCounts.cpuPoses).toBe(0);
        expect(warnings.filter((message) => message.includes('no layout'))).toHaveLength(1);
        await scene.whenRefined();
        scene.releaseRefinements();
        frame();
        expect(scene.meshStatus(cloth).builtBy).toBe('tinybvh');
        expect(scene.deformingCounts.cpuPoses).toBe(1);
        expect(createCpuTracer(readScene(scene)).closest(down(5.1, 0.2))?.t).toBeCloseTo(2, 0);
        scene.dispose();
    });

    it("with subgroups: a CPU pose replaces the GPU one (no refit pass), and a GPU pose afterwards replaces it", async () => {
        const gpu = createFakeGpu({}, { features: ['subgroups'], kernels: { bvhMergedIdentity: mergedIdentityStandIn } });
        const scene = await SceneTrees.create(gpu.device, { gpuFirstTrees: false, topLevel: 'cpu' });
        const passes: string[] = [];
        const frame = () => {
            const encoder = gpu.createEncoder();
            scene.update(encoder);
            passes.push(...(encoder as unknown as { commands: { dispatch?: { entryPoint: string } }[] }).commands.flatMap((command) => (command.dispatch ? [command.dispatch.entryPoint] : [])));
            gpu.submit(encoder);
        };
        const geometry = tessellatedQuad(4, 1);
        const cloth = scene.addMesh(geometry, { deforming: true });
        scene.addInstance({ mesh: cloth, objectToWorld: identity });
        frame();
        await scene.whenRefined();
        frame();
        const vertices = gpu.device.createBuffer({ size: 25 * 12, usage: 0x80 | 0x4 });
        scene.setPose(cloth, { positionBuffer: vertices, vertexCount: 25 });
        scene.setPoseCpu(cloth, posed(geometry));
        passes.length = 0;
        frame();
        expect(passes).not.toContain('meshRefitTriangles');
        expect(scene.deformingCounts.cpuPoses).toBe(1);
        expect(createCpuTracer(readScene(scene)).closest(down(5.1, 0.2))?.t).toBeCloseTo(2, 0);
        scene.setPose(cloth, { positionBuffer: vertices, vertexCount: 25 });
        passes.length = 0;
        frame();
        expect(passes).toContain('meshRefitTriangles');
        expect(scene.deformingCounts.cpuPoses).toBe(1);
        // The top level has the deform box back (the fake GPU runs no refit, so the tree still holds the CPU pose, outside it).
        expect(createCpuTracer(readScene(scene)).closest(down(5.1, 0.2))).toBeNull();
        scene.dispose();
    });
});

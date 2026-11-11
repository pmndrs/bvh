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
import { TOP_HEADER_WORD, TRIANGLE_WORD, TRIANGLE_WORDS } from '../trace/layout.ts';
import { buildMeshBvhAsync } from '../trace/mesh-bvh-build.ts';
import type { MeshGeometry } from '../trace/mesh-bvh.ts';
import { createRandom, randomAffine, randomRays, tessellatedQuad } from '../trace/oracle-fixtures.ts';
import type { TraceRay } from '../trace/rays.ts';
import { assembleTraceScene, type TraceSceneData, type TraceSceneInstance } from '../trace/scene-assembly.ts';
import type { TopLevelBvh } from '../trace/top-level.ts';
import { GpuTreeOutputs, type FirstTreeBuilder, type GpuTreeBatchResult, type GpuTreeBuilds, type GpuTreeRequest } from './gpu-first-trees.ts';
import { MERGED_SOURCE_WORDS } from './merged-identity-gpu.ts';
import { SceneTrees, type SceneInstanceDescriptor, type SceneTreeEvent, type SceneTreesOptions } from './scene-trees.ts';
import type { GpuMeshTreeRange } from '../mesh-input.ts';

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
        triangles[base + TRIANGLE_WORD.flags] = (triangles[base + TRIANGLE_WORD.flags] | sources[at + 2]) >>> 0;
    }
}

/** A first-tree builder that answers when the test says so; each `build` is one batch. */
class StandInGpuBuilder implements FirstTreeBuilder {
    readonly requests: GpuTreeRequest[][] = [];
    private readonly pending: { members: { slot: number; generation: number; geometry: MeshGeometry }[]; resolve: (result: GpuTreeBatchResult) => void }[] = [];
    readonly refuse = new Set<number>();

    constructor(private readonly device: GPUDevice) {}

    async prepare(): Promise<void> {
        await buildMeshBvhAsync({ positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]) });
    }

    build(requests: readonly GpuTreeRequest[]): GpuTreeBuilds {
        this.requests.push([...requests]);
        const members = requests.map(({ slot, generation, geometry }) => {
            const cpu = geometry as MeshGeometry;
            return { slot, generation, geometry: { positions: cpu.positions.slice(), indices: cpu.indices ? Uint32Array.from(cpu.indices) : null, triangleFlags: cpu.triangleFlags?.slice() ?? null } };
        });
        const batch = new Promise<GpuTreeBatchResult>((resolve) => this.pending.push({ members, resolve }));
        return { notBuilt: [], batches: [batch], buildingBytes: 1000 };
    }

    get waiting(): number {
        return this.pending.length;
    }

    async finishBatch(): Promise<void> {
        const batch = this.pending.shift();
        if (!batch) throw new Error('no batch is waiting');
        const trees = await Promise.all(batch.members.map((member) => buildMeshBvhAsync(member.geometry, { maxLeafPrimitives: 1 })));
        // As H-PLOC's outputs: room for one node per triangle, the used nodes packed at the front.
        const triangleTotal = trees.reduce((sum, tree) => sum + tree.triangleCount, 0);
        const usage = 0x4 | 0x8 | 0x80;
        const nodes = this.device.createBuffer({ label: 'stand-in nodes', size: triangleTotal * 64, usage });
        const triangles = this.device.createBuffer({ label: 'stand-in triangles', size: triangleTotal * 48, usage });
        const parents = this.device.createBuffer({ label: 'stand-in parents', size: triangleTotal * 4, usage });
        const meshes: GpuMeshTreeRange[] = [];
        let nodeOffset = 0;
        let triangleOffset = 0;
        trees.forEach((tree, index) => {
            this.device.queue.writeBuffer(nodes, nodeOffset * 64, tree.nodes, 0, tree.nodeCount * 64);
            this.device.queue.writeBuffer(triangles, triangleOffset * 48, tree.triangles, 0, tree.triangleCount * 48);
            const refused = this.refuse.has(batch.members[index].slot);
            meshes.push({ nodeOffset, nodeCount: tree.nodeCount, triangleOffset, triangleCount: tree.triangleCount, leafCount: tree.leafCount, maxDepth: tree.maxDepth, bounds: tree.bounds, problem: refused ? 'refused by the test' : null });
            nodeOffset += tree.nodeCount;
            triangleOffset += tree.triangleCount;
        });
        const usable = meshes.filter((mesh) => mesh.problem === null).length;
        const outputs = new GpuTreeOutputs({ nodes, triangles, parents, meshes, destroy: () => [nodes, triangles, parents].forEach((buffer) => buffer.destroy()) }, usable);
        batch.resolve({
            built: meshes.flatMap((tree, index) => (tree.problem ? [] : [{ slot: batch.members[index].slot, generation: batch.members[index].generation, outputs, tree, layout: null }])),
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
    const gpu: FakeGpu = createFakeGpu({}, { kernels: { bvhMergedIdentity: mergedIdentityStandIn } });
    const stand = options.stand === false ? null : new StandInGpuBuilder(gpu.device);
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
    const { nodes, triangles, topLevel } = scene.buffers;
    const bytes = (buffer: GPUBuffer) => (buffer as unknown as FakeBuffer).bytes.slice().buffer;
    const top = bytes(topLevel);
    const header = new Uint32Array(top, 0, 4);
    return {
        nodes: bytes(nodes),
        triangles: bytes(triangles),
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

    it('leaves a refused or too-deep GPU tree to tinybvh and says why', async () => {
        const { scene, stand, frame } = await setUp({ holdRefinements: true, gpuMaxDepth: 2 });
        const big = scene.addMesh(tessellatedQuad(8, 1));
        const small = scene.addMesh(tessellatedQuad(1, 1));
        const refused = scene.addMesh(tessellatedQuad(2, 1));
        for (const mesh of [big, small, refused]) scene.addInstance({ mesh, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] });
        stand!.refuse.add(refused);
        frame();
        await stand!.finishBatch();
        await scene.whenRefined();
        frame();
        expect(scene.meshStatus(small).builtBy).toBe('gpu');
        expect(scene.meshStatus(big).builtBy).toBe('tinybvh');
        expect(scene.meshStatus(big).fallbackReason).toMatch(/gpuMaxDepth/);
        expect(scene.meshStatus(refused).fallbackReason).toBe('refused by the test');
        scene.dispose();
    });

    it("adopt: the load batch's outputs become the buffers (no copy), with a lower peak and the same hits", async () => {
        const peaks: Record<string, number> = {};
        for (const mode of ['copy', 'adopt'] as const) {
            const { scene, stand, frame, gpu } = await setUp({ holdRefinements: true, gpuTreeOutputs: mode });
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
        scene.dispose();

        // Without subgroups: no refit, the pose is counted as skipped.
        const plain = await setUp({ stand: false });
        const mesh = plain.scene.addMesh(tessellatedQuad(2, 1), { deforming: true });
        plain.scene.addInstance({ mesh, objectToWorld: identity });
        plain.scene.setPose(mesh, { positionBuffer: plain.gpu.device.createBuffer({ size: 64, usage: 0x80 }), vertexCount: 9 });
        expect(plain.scene.deformingCounts.posesSkipped).toBe(1);
        plain.scene.dispose();
    });
});

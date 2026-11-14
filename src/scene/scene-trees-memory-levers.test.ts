/**
 * The P9 memory levers of BX.20 that keep the GPU layout (docs/internals/design/memory.md,
 * "Levers ranked by the bytes each could save"), each against the default it sits beside:
 * the same trees and the same hits with the lever on, and the bytes it saves.
 *
 * Node, the fake GPU and the stand-in H-PLOC (tests/support/fake-hploc.ts), as the
 * scene-trees tests; the GPU pages (scene-trees-*) check the real kernels.
 */

import { describe, expect, it } from 'vitest';
import { createFakeGpu, type FakeBuffer, type FakeGpu } from '../../tests/support/fake-gpu.ts';
import { FakeHploc } from '../../tests/support/fake-hploc.ts';
import { SCENE_KERNELS } from '../../tests/support/scene-stand-ins.ts';
import { createCpuTracer, type CpuHit } from '../trace/cpu-traversal.ts';
import { TOP_HEADER_WORD } from '../trace/layout.ts';
import type { MeshGeometry } from '../trace/mesh-bvh.ts';
import { createRandom, randomAffine, randomRays, tessellatedQuad } from '../trace/oracle-fixtures.ts';
import type { TraceRay } from '../trace/rays.ts';
import type { TraceSceneData } from '../trace/scene-assembly.ts';
import type { TopLevelBvh } from '../trace/top-level.ts';
import { GpuFirstTrees } from './gpu-first-trees.ts';
import { SceneTrees, type SceneInstanceDescriptor, type SceneTreesOptions } from './scene-trees.ts';

const SMALL_CAP = 4 * 1024;

interface Instance extends SceneInstanceDescriptor {
    readonly meshIndex: number;
}

async function setUp(options: Partial<SceneTreesOptions> = {}) {
    const dispatches: string[] = [];
    const kernels = Object.fromEntries(
        Object.entries(SCENE_KERNELS).map(([entry, kernel]) => [
            entry,
            (...args: Parameters<typeof kernel>) => {
                dispatches.push(entry);
                kernel(...args);
            },
        ]),
    );
    const gpu: FakeGpu = createFakeGpu({}, { kernels });
    const hploc = new FakeHploc(gpu.device);
    const scene = await SceneTrees.create(gpu.device, { ...options, firstTreeBuilder: new GpuFirstTrees(gpu.device, { hploc, triangleLeaves: options.triangleLeaves }) });
    const frame = () => {
        const encoder = gpu.createEncoder();
        scene.update(encoder);
        gpu.submit(encoder);
    };
    return { gpu, hploc, scene, frame, dispatches };
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

const key = (hit: CpuHit | null) => (hit ? `${hit.instanceIndex}/${hit.triangleIndex}/${hit.frontFace}/${hit.material}/${hit.t}` : '-');

/** Every ray's closest hit (instance, triangle, side, material and t, exactly) and any-hit, for comparing two scenes. */
function hits(scene: SceneTrees, rays: readonly TraceRay[]): string[] {
    const tracer = createCpuTracer(readScene(scene));
    return rays.map((ray) => `${key(tracer.closest(ray))} ${tracer.any(ray)}`);
}

function sceneOf(count: number) {
    const random = createRandom(23);
    const geometries: MeshGeometry[] = Array.from({ length: count }, (_, index) => tessellatedQuad(6 + 2 * index, 1));
    const instances: Instance[] = geometries.map((_, index) => ({
        mesh: -1,
        meshIndex: index,
        instanceIndex: 100 + index,
        objectToWorld: randomAffine(random, { extent: 4, mirror: index % 2 === 1 }),
        material: index % 3 === 0 ? undefined : index,
        // Half static: with merging on, a merged group and instanced meshes.
        static: index % 2 === 0,
    }));
    const rays = randomRays(random, 500, [-6, -6, -6, 6, 6, 6], 1);
    return { geometries, instances, rays };
}

function addAll(scene: SceneTrees, geometries: readonly MeshGeometry[], instances: readonly Instance[]): number[] {
    const meshes = geometries.map((geometry) => scene.addMesh(geometry));
    for (const instance of instances) scene.addInstance({ ...instance, mesh: meshes[instance.meshIndex] });
    return meshes;
}

const bytesOf = (buffer: GPUBuffer) => (buffer as unknown as FakeBuffer).bytes;

/** Loads `sceneOf(count)` holding refinements: the scene on GPU trees, then settled on tinybvh's; each state's buffers and hits. */
async function load(options: Partial<SceneTreesOptions>, count = 6) {
    const { gpu, scene, frame, dispatches } = await setUp({ holdRefinements: true, ...options });
    const { geometries, instances, rays } = sceneOf(count);
    addAll(scene, geometries, instances);
    frame();
    await scene.whenBuilt();
    frame();
    const gpuTrees = { nodes: bytesOf(scene.buffers.nodes).slice(), triangles: bytesOf(scene.buffers.triangles).slice(), hits: hits(scene, rays), quantizeDispatches: dispatches.filter((entry) => entry === 'bvhqQuantize').length };
    await scene.whenRefined();
    scene.releaseRefinements();
    frame();
    frame();
    expect(scene.counts.onGpuTrees).toBe(0);
    const settled = { nodes: bytesOf(scene.buffers.nodes).slice(), triangles: bytesOf(scene.buffers.triangles).slice(), hits: hits(scene, rays), memory: scene.memory };
    return { gpu, scene, frame, gpuTrees, settled, rays, quantizeTables: gpu.buffers.filter((buffer) => buffer.label.endsWith('quantize meshes')).length };
}

describe("quantizeJobs: 'batched' (one quantize pass per update, not per tree)", () => {
    for (const variant of [
        { name: 'uncapped, merged', options: { gpuMaxBatchBytes: Infinity } },
        { name: 'capped (a batch per mesh, copied between updates), merged', options: { gpuMaxBatchBytes: SMALL_CAP } },
        { name: 'uncapped, indexed, unmerged', options: { gpuMaxBatchBytes: Infinity, triangleLeaves: 'indexed' as const, mergeStaticInstances: false } },
    ]) {
        it(`${variant.name}: the same buffers bit for bit and the same hits, with fewer quantize dispatches and tables`, async () => {
            const perTree = await load({ ...variant.options, quantizeJobs: 'per-tree' });
            const batched = await load({ ...variant.options, quantizeJobs: 'batched' });
            for (const state of ['gpuTrees', 'settled'] as const) {
                expect(batched[state].nodes).toEqual(perTree[state].nodes);
                expect(batched[state].triangles).toEqual(perTree[state].triangles);
                expect(batched[state].hits).toEqual(perTree[state].hits);
            }
            const trees = perTree.scene.counts.gpuTreesBuilt;
            expect(perTree.gpuTrees.quantizeDispatches).toBe(trees);
            expect(perTree.quantizeTables).toBe(trees);
            // One per batch's node buffer (one batch uncapped; capped, each batch is copied in a submit of its own).
            expect(batched.gpuTrees.quantizeDispatches).toBe(perTree.scene.counts.gpuBatches);
            expect(batched.quantizeTables).toBe(perTree.scene.counts.gpuBatches);
            if (variant.options.gpuMaxBatchBytes === Infinity) expect(batched.gpuTrees.quantizeDispatches).toBe(1);
            perTree.scene.dispose();
            batched.scene.dispose();
        });
    }

    it('refuses an unknown value', async () => {
        await expect(setUp({ quantizeJobs: 'all' as never })).rejects.toThrow(/quantizeJobs/);
    });
});

/** A quad's positions lifted into a hill: a CPU pose that changes every triangle. */
function hill(geometry: MeshGeometry): Float32Array {
    const positions = geometry.positions.slice();
    for (let at = 0; at < positions.length; at += 3) positions[at + 2] += 0.5 + 0.25 * Math.sin(positions[at] * 1.7) * Math.cos(positions[at + 1] * 1.3);
    return positions;
}

describe('SceneTrees.trim and trimWhenRefined (the node buffer packed after the swaps)', () => {
    for (const variant of [
        { name: "adopt (f32, the batch's one node per triangle)", options: { gpuTreeOutputs: 'adopt' as const, nodeFormat: 'f32' as const, gpuMaxBatchBytes: Infinity, mergeStaticInstances: false } },
        { name: 'adopt, merged', options: { gpuTreeOutputs: 'adopt' as const, nodeFormat: 'f32' as const, gpuMaxBatchBytes: Infinity } },
        { name: 'copy, the defaults (quantized, merged, capped)', options: {} },
        { name: 'copy, indexed, uncapped', options: { triangleLeaves: 'indexed' as const, gpuMaxBatchBytes: Infinity } },
    ]) {
        it(`${variant.name}: trim() packs the nodes to their size, with the same hits; trimWhenRefined does it unasked`, async () => {
            const plain = await load(variant.options);
            const before = plain.scene.memory;
            expect(before.nodes).toBeGreaterThan(before.nodesInUse);
            const version = plain.scene.version;
            plain.scene.trim();
            plain.frame();
            const after = plain.scene.memory;
            expect(after.nodes).toBe(before.nodesInUse);
            expect(after.nodesInUse).toBe(before.nodesInUse);
            expect(plain.scene.trimCount).toBe(1);
            expect(plain.scene.version).toBeGreaterThan(version);
            expect(hits(plain.scene, plain.rays)).toEqual(plain.settled.hits);
            // The old buffer goes at the next update; a trim with nothing to give back does nothing.
            plain.scene.trim();
            plain.frame();
            expect(plain.scene.trimCount).toBe(1);
            expect(plain.scene.memory.retired).toBe(0);

            const auto = await load({ ...variant.options, trimWhenRefined: true });
            expect(auto.scene.trimCount).toBe(1);
            expect(auto.settled.memory.nodes).toBe(before.nodesInUse);
            expect(auto.settled.hits).toEqual(plain.settled.hits);

            // A mesh added after the trim grows the buffer again and traces like the rest.
            const extra = tessellatedQuad(5, 1);
            const mesh = plain.scene.addMesh(extra);
            plain.scene.addInstance({ mesh, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1], instanceIndex: 999 });
            plain.frame();
            await plain.scene.whenRefined();
            plain.frame();
            plain.frame();
            const tracer = createCpuTracer(readScene(plain.scene));
            expect(tracer.closest({ origin: [0.3, 0.4, 5], direction: [0, 0, -1], tMin: 0, tMax: 100 })?.instanceIndex).toBe(999);
            plain.scene.dispose();
            auto.scene.dispose();
        });
    }

    for (const nodeFormat of ['f32', 'quantized'] as const) {
        it(`${nodeFormat}: a deforming mesh's refit and CPU pose follow its tree when a trim moves it`, async () => {
            const gpu = createFakeGpu({}, { features: ['subgroups'], kernels: SCENE_KERNELS });
            const scenes = [];
            for (const trimWhenRefined of [false, true]) {
                const hploc = new FakeHploc(gpu.device);
                const scene = await SceneTrees.create(gpu.device, { nodeFormat, topLevel: 'cpu', trimWhenRefined, firstTreeBuilder: new GpuFirstTrees(gpu.device, { hploc }) });
                const frame = () => {
                    const encoder = gpu.createEncoder();
                    scene.update(encoder);
                    gpu.submit(encoder);
                };
                const { geometries, instances } = sceneOf(4);
                const meshes = geometries.map((geometry) => scene.addMesh(geometry));
                const handles = instances.map((instance) => scene.addInstance({ ...instance, mesh: meshes[instance.meshIndex], static: false }));
                const sheet = tessellatedQuad(4, 1);
                const cloth = scene.addMesh(sheet, { deforming: { bounds: [-5, -5, -5, 5, 5, 5] } });
                scene.addInstance({ mesh: cloth, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], instanceIndex: 50 });
                frame();
                await scene.whenRefined();
                frame();
                frame();
                // A mesh before the cloth goes: its range is a hole the trim closes, moving the cloth's tree.
                scene.removeInstance(handles[0]);
                scene.removeMesh(meshes[0]);
                frame();
                frame();
                // The CPU pose after the trim, written where the tree is now.
                scene.setPoseCpu(cloth, hill(sheet));
                frame();
                frame();
                const placement = scene.meshStatus(cloth).placement!;
                const internal = scene as unknown as { deforming: { meshes: Map<number, { placement: { nodeWordOffset: number }; refit: { quantized: { rootSlot: number } | null; refit: { params: FakeBuffer } } }> } };
                const refit = internal.deforming.meshes.get(cloth)!;
                const wordsPerNode = nodeFormat === 'quantized' ? 8 : 16;
                expect(refit.placement.nodeWordOffset).toBe(placement.nodeOffset * wordsPerNode);
                if (nodeFormat === 'quantized') expect(refit.refit.quantized!.rootSlot).toBe(placement.nodeOffset);
                else expect(refit.refit.refit.params.words[0]).toBe(placement.nodeOffset * wordsPerNode);
                // Once after the load, once after the removal.
                expect(scene.trimCount).toBe(trimWhenRefined ? 2 : 0);
                scenes.push({ scene, placement });
            }
            // Random rays, and a grid straight down onto the posed cloth (z 0.25 to 0.75 over [-1, 1]^2).
            const grid: TraceRay[] = Array.from({ length: 400 }, (_, index) => ({ origin: [-0.95 + 0.1 * (index % 20), -0.95 + 0.1 * Math.floor(index / 20), 4], direction: [0, 0, -1], tMin: 0, tMax: 100 }));
            const rays = [...randomRays(createRandom(4), 800, [-6, -6, -6, 6, 6, 6], 1), ...grid];
            // The trim moved the cloth's tree; the refit and the CPU pose went with it.
            expect(scenes[1].placement.nodeOffset).not.toBe(scenes[0].placement.nodeOffset);
            const trimmedHits = hits(scenes[1].scene, rays);
            expect(trimmedHits).toEqual(hits(scenes[0].scene, rays));
            expect(trimmedHits.filter((hit) => hit.startsWith('50/')).length).toBeGreaterThan(200);
            expect(scenes[1].scene.memory.nodes).toBe(scenes[1].scene.memory.nodesInUse);
            for (const { scene } of scenes) scene.dispose();
        });
    }
});

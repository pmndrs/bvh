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
import { compactPoseLayout, parentsOfNodes, refitLayoutToPose } from '../refit-cpu.ts';
import { buildHplocOnCpu } from '../hploc-cpu.ts';
import { buildLbvhOnCpu } from '../lbvh-cpu.ts';
import { dequantizeNodes, quantizeNodes } from '../quantized-nodes.ts';
import { createBvhWorkerPool } from '../worker/pool.ts';
import { createCpuTracer, type CpuHit } from '../trace/cpu-traversal.ts';
import { buildMeshBvhAsync } from '../trace/mesh-bvh-build.ts';
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

describe('the 2026-11-14 levers with the BX.20 ones (where they share code)', () => {
    for (const variant of [
        { name: 'uncapped, merged', options: { gpuMaxBatchBytes: Infinity } },
        { name: 'capped, merged', options: { gpuMaxBatchBytes: SMALL_CAP } },
        { name: 'uncapped, indexed, unmerged', options: { gpuMaxBatchBytes: Infinity, triangleLeaves: 'indexed' as const, mergeStaticInstances: false } },
    ]) {
        it(`${variant.name}: gpuQuantizeOnEmit with quantizeJobs 'batched' and parents 'refit' gives the quantize pass's buffers bit for bit, with no quantize dispatch`, async () => {
            const reference = await load({ ...variant.options, quantizeJobs: 'per-tree' });
            const levers = await load({ ...variant.options, gpuQuantizeOnEmit: true, quantizeJobs: 'batched', parents: 'refit' });
            for (const state of ['gpuTrees', 'settled'] as const) {
                expect(levers[state].nodes).toEqual(reference[state].nodes);
                expect(levers[state].triangles).toEqual(reference[state].triangles);
                expect(levers[state].hits).toEqual(reference[state].hits);
            }
            // The emit wrote every GPU tree's quantized nodes: nothing was left for a pass, batched or not.
            expect(levers.gpuTrees.quantizeDispatches).toBe(0);
            expect(reference.gpuTrees.quantizeDispatches).toBeGreaterThan(0);
            reference.scene.dispose();
            levers.scene.dispose();
        });
    }

    it("adopt (quantized, indexed) with parents 'refit', quantizeJobs 'batched' and trimWhenRefined: the copy scene's hits, a trimmed buffer, the adopted one replaced", async () => {
        const copy = await load({ gpuMaxBatchBytes: Infinity, triangleLeaves: 'indexed' });
        const adopted = await load({ gpuMaxBatchBytes: Infinity, triangleLeaves: 'indexed', gpuTreeOutputs: 'adopt', parents: 'refit', quantizeJobs: 'batched', trimWhenRefined: true });
        expect(adopted.gpuTrees.hits).toEqual(copy.gpuTrees.hits);
        expect(adopted.gpuTrees.quantizeDispatches).toBe(0);
        expect(adopted.settled.hits).toEqual(copy.settled.hits);
        expect(adopted.scene.trimCount).toBe(1);
        expect(adopted.settled.memory.nodes).toBe(adopted.settled.memory.nodesInUse);
        expect(adopted.gpu.buffers.find((buffer) => buffer.label === 'fake hploc nodes')!.destroyed).toBe(true);
        copy.scene.dispose();
        adopted.scene.dispose();
    });

    it("positionsBuffer with trimWhenRefined and quantizeJobs 'batched': the app's buffer stays bound and untouched through the trim", async () => {
        const { geometries, instances, rays } = sceneOf(4);
        const reference = await load({ gpuMaxBatchBytes: Infinity, triangleLeaves: 'indexed', mergeStaticInstances: false }, 4);
        const gpu = createFakeGpu({}, { kernels: SCENE_KERNELS });
        const bases: number[] = [];
        let count = 0;
        for (const geometry of geometries) {
            bases.push(count);
            count += geometry.positions.length / 3;
        }
        const data = new Float32Array(count * 4);
        geometries.forEach((geometry, index) => {
            for (let vertex = 0; vertex < geometry.positions.length / 3; vertex++) data.set(geometry.positions.subarray(vertex * 3, vertex * 3 + 3), (bases[index] + vertex) * 4);
        });
        const app = gpu.device.createBuffer({ label: 'app positions', size: data.byteLength, usage: 0x80 | 0x8 });
        gpu.device.queue.writeBuffer(app, 0, data);
        const hploc = new FakeHploc(gpu.device);
        const appScene = await SceneTrees.create(gpu.device, { holdRefinements: true, gpuMaxBatchBytes: Infinity, triangleLeaves: 'indexed', trimWhenRefined: true, quantizeJobs: 'batched', positionsBuffer: app, firstTreeBuilder: new GpuFirstTrees(gpu.device, { hploc, triangleLeaves: 'indexed' }) });
        const meshes = geometries.map((geometry, index) => appScene.addMesh(geometry, { vertexBase: bases[index] }));
        for (const instance of instances) appScene.addInstance({ ...instance, mesh: meshes[instance.meshIndex] });
        const appFrame = () => {
            const encoder = gpu.createEncoder();
            appScene.update(encoder);
            gpu.submit(encoder);
        };
        appFrame();
        await appScene.whenBuilt();
        appFrame();
        expect(hits(appScene, rays)).toEqual(reference.gpuTrees.hits);
        await appScene.whenRefined();
        appScene.releaseRefinements();
        appFrame();
        appFrame();
        expect(appScene.trimCount).toBe(1);
        expect(appScene.buffers.positions).toBe(app);
        expect(Array.from(new Float32Array(bytesOf(app).slice().buffer))).toEqual(Array.from(data));
        expect(hits(appScene, rays)).toEqual(reference.settled.hits);
        appScene.dispose();
        reference.scene.dispose();
    });
});

describe('SceneTrees.trim and trimWhenRefined (the node buffer packed after the swaps)', () => {
    for (const variant of [
        { name: "adopt (f32, the batch's one node per triangle)", options: { gpuTreeOutputs: 'adopt' as const, nodeFormat: 'f32' as const, gpuMaxBatchBytes: Infinity, mergeStaticInstances: false } },
        { name: 'adopt, merged', options: { gpuTreeOutputs: 'adopt' as const, nodeFormat: 'f32' as const, gpuMaxBatchBytes: Infinity } },
        { name: 'copy, the defaults (quantized, merged, capped)', options: {} },
        { name: 'copy, indexed, uncapped', options: { triangleLeaves: 'indexed' as const, gpuMaxBatchBytes: Infinity } },
        // With the 2026-11-14 levers: adoption in quantized and indexed scenes (frame slots and headers in the adopted buffers), and the emit.
        { name: 'adopt, quantized (the emit), unmerged', options: { gpuTreeOutputs: 'adopt' as const, gpuMaxBatchBytes: Infinity, mergeStaticInstances: false } },
        { name: 'adopt, quantized, indexed, merged', options: { gpuTreeOutputs: 'adopt' as const, triangleLeaves: 'indexed' as const, gpuMaxBatchBytes: Infinity } },
        { name: 'copy, quantized on emit, capped, no parents', options: { gpuQuantizeOnEmit: true, parents: 'refit' as const } },
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

describe("compactPoseLayout and deformingLayouts: 'compact' (a deforming mesh keeps nodes, parents and 4 bytes per triangle)", () => {
    /** A quad with app flags (bit 0) on some triangles, materials on others, and the padding word set by hand on a few slots. */
    const flagged = (cells: number): MeshGeometry => {
        const quad = tessellatedQuad(cells, 1);
        const triangles = quad.indices!.length / 3;
        return { ...quad, triangleFlags: Uint32Array.from({ length: triangles }, (_, index) => index % 3 === 0 ? 1 : 0), triangleMaterials: Uint16Array.from({ length: triangles }, (_, index) => (index % 4 === 3 ? 0xffff : index % 5)) };
    };

    for (const triangleLeaves of ['records', 'indexed'] as const) {
        for (const nodeFormat of ['f32', 'quantized'] as const) {
            it(`${triangleLeaves}, ${nodeFormat}: refitLayoutToPose gives the same posed tree from the pose layout as from the whole layout`, async () => {
                for (const geometry of [flagged(7), tessellatedQuad(5, 1)]) {
                    const tree = await buildMeshBvhAsync(geometry, { triangleLeaves, nodeFormat });
                    if (triangleLeaves === 'records') {
                        // Padding words are kept too (a merged record's source instance lives there; any non-zero word must survive).
                        const words = new Uint32Array(tree.triangles);
                        for (let slot = 0; slot < tree.triangleCount; slot += 7) words[slot * 12 + 11] = 1000 + slot;
                    }
                    const compact = compactPoseLayout(tree);
                    expect(compact.slotTriangles.byteLength).toBe(tree.triangleCount * 4);
                    expect('triangles' in compact).toBe(false);
                    expect(compact.nodes).toBe(tree.nodes);
                    expect(compact.parents).toBe(tree.parents);
                    if (triangleLeaves === 'indexed' || geometry.triangleFlags === undefined) expect(compact.slotFlags).toBeNull();
                    const moved = hill(geometry);
                    const whole = refitLayoutToPose(tree, moved, geometry.indices);
                    const posed = refitLayoutToPose(compact, moved, geometry.indices);
                    expect(new Uint8Array(posed.nodes)).toEqual(new Uint8Array(whole.nodes));
                    expect(posed.triangles === null ? null : new Uint8Array(posed.triangles)).toEqual(whole.triangles === null ? null : new Uint8Array(whole.triangles));
                    expect(posed.bounds).toEqual(whole.bounds);
                }
            });
        }
    }

    for (const triangleLeaves of ['records', 'indexed'] as const) {
        it(`${triangleLeaves}: setPoseCpu writes the same buffers with 'compact' as with 'full', which keeps 48 bytes per triangle more`, async () => {
            const results: Record<string, { nodes: Uint8Array; triangles: Uint8Array; positions: Uint8Array | null; kept: number; hits: string[] }> = {};
            const geometry = flagged(9);
            const rays: TraceRay[] = Array.from({ length: 400 }, (_, index) => ({ origin: [-0.95 + 0.1 * (index % 20), -0.95 + 0.1 * Math.floor(index / 20), 4], direction: [0, 0, -1], tMin: 0, tMax: 100 }));
            for (const deformingLayouts of ['full', 'compact'] as const) {
                const { scene, frame } = await setUp({ triangleLeaves, deformingLayouts, mergeStaticInstances: false });
                const cloth = scene.addMesh(geometry, { deforming: true });
                scene.addInstance({ mesh: cloth, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], instanceIndex: 5 });
                // A pose on the GPU tree (its layout read back), then on tinybvh's.
                frame();
                await scene.whenBuilt();
                scene.setPoseCpu(cloth, hill(geometry));
                frame();
                await scene.whenRefined();
                scene.releaseRefinements();
                frame();
                scene.setPoseCpu(cloth, hill({ ...geometry, positions: hill(geometry) }));
                frame();
                expect(scene.deformingCounts.cpuPoses).toBe(2);
                const internal = scene as unknown as { meshes: Map<number, { cpuPose: { layout: { triangles?: ArrayBuffer; slotTriangles?: Uint32Array; slotFlags?: Uint32Array | null; slotPadding?: Uint32Array | null } } }> };
                const layout = internal.meshes.get(cloth)!.cpuPose.layout;
                const kept = (layout.triangles?.byteLength ?? 0) + (layout.slotTriangles?.byteLength ?? 0) + (layout.slotFlags?.byteLength ?? 0) + (layout.slotPadding?.byteLength ?? 0);
                results[deformingLayouts] = {
                    nodes: bytesOf(scene.buffers.nodes).slice(),
                    triangles: bytesOf(scene.buffers.triangles).slice(),
                    positions: scene.buffers.positions ? bytesOf(scene.buffers.positions).slice() : null,
                    kept,
                    hits: hits(scene, rays),
                };
                scene.dispose();
            }
            expect(results.compact.nodes).toEqual(results.full.nodes);
            expect(results.compact.triangles).toEqual(results.full.triangles);
            expect(results.compact.positions).toEqual(results.full.positions);
            expect(results.compact.hits).toEqual(results.full.hits);
            expect(results.full.hits.filter((hit) => hit.startsWith('5/')).length).toBeGreaterThan(300);
            const triangles = geometry.indices!.length / 3;
            // Records: 48 bytes per triangle against 4 (original) + 4 (flags, some non-zero here); indexed: 16 against 4.
            expect(results.full.kept).toBe(triangles * (triangleLeaves === 'records' ? 48 : 16));
            expect(results.compact.kept).toBe(triangles * (triangleLeaves === 'records' ? 8 : 4));
        });
    }
});

describe("parents: 'refit' (parents only where a refit reads them, P9 lever 7)", () => {
    it('parentsOfNodes finds the parents every builder writes, from f32 and decoded quantized nodes', async () => {
        const random = createRandom(31);
        for (const geometry of [tessellatedQuad(9, 1), { positions: Float32Array.from({ length: 900 }, () => random() * 4 - 2) }, tessellatedQuad(1, 1)]) {
            for (const tree of [await buildMeshBvhAsync(geometry), await buildMeshBvhAsync(geometry, { maxLeafPrimitives: 1 }), buildHplocOnCpu(geometry), buildLbvhOnCpu(geometry)]) {
                expect(parentsOfNodes(new Uint32Array(tree.nodes), 0, tree.nodeCount)).toEqual(tree.parents.subarray(0, tree.nodeCount));
                const quantized = quantizeNodes(tree);
                expect(parentsOfNodes(new Uint32Array(dequantizeNodes(quantized).nodes), 0, tree.nodeCount)).toEqual(tree.parents.subarray(0, tree.nodeCount));
            }
        }
    });

    it('a tinybvh build without parents gives the same tree and an empty parents array', async () => {
        const geometry = tessellatedQuad(8, 1);
        for (const options of [{}, { nodeFormat: 'quantized' as const }, { triangleLeaves: 'indexed' as const }]) {
            const full = await buildMeshBvhAsync(geometry, options);
            const bare = await buildMeshBvhAsync(geometry, { ...options, parents: false });
            expect(new Uint8Array(bare.nodes)).toEqual(new Uint8Array(full.nodes));
            expect(new Uint8Array(bare.triangles)).toEqual(new Uint8Array(full.triangles));
            expect(bare.parents.length).toBe(0);
            expect(full.parents.length).toBe(full.nodeCount);
        }
    });

    for (const variant of [
        { name: 'the defaults (quantized, merged, capped)', options: {} },
        { name: 'adopt', options: { gpuTreeOutputs: 'adopt' as const, nodeFormat: 'f32' as const, gpuMaxBatchBytes: Infinity, mergeStaticInstances: false } },
    ]) {
        it(`${variant.name}: the same buffers, no batch asked for parents, static meshes' tinybvh trees without them, a deforming mesh's with them`, async () => {
            const loads: Record<string, { gpuTrees: { nodes: Uint8Array; triangles: Uint8Array; hits: string[] }; settled: { nodes: Uint8Array; triangles: Uint8Array; hits: string[] } }> = {};
            const asked: Record<string, boolean[]> = {};
            for (const parents of ['always', 'refit'] as const) {
                const original = FakeHploc.prototype.createBatch;
                asked[parents] = [];
                FakeHploc.prototype.createBatch = function (this: FakeHploc, ...args: Parameters<FakeHploc['createBatch']>) {
                    asked[parents].push((args[1] as { parents?: boolean } | undefined)?.parents !== false);
                    return original.apply(this, args);
                };
                try {
                    const loaded = await load({ ...variant.options, parents });
                    loads[parents] = loaded;
                    loaded.scene.dispose();
                } finally {
                    FakeHploc.prototype.createBatch = original;
                }
            }
            for (const state of ['gpuTrees', 'settled'] as const) {
                expect(loads.refit[state].nodes).toEqual(loads.always[state].nodes);
                expect(loads.refit[state].triangles).toEqual(loads.always[state].triangles);
                expect(loads.refit[state].hits).toEqual(loads.always[state].hits);
            }
            expect(asked.always.length).toBeGreaterThan(0);
            expect(asked.always.every(Boolean)).toBe(true);
            expect(asked.refit.some(Boolean)).toBe(false);
        });
    }

    it("the tinybvh builds: parents only for deforming meshes, whose CPU poses still work", async () => {
        const requests: { parents: boolean }[] = [];
        const workers = createBvhWorkerPool({ onBuild: () => {} });
        const build = workers.build.bind(workers);
        workers.build = (geometry, options, request) => {
            requests.push({ parents: options?.parents !== false });
            return build(geometry, options, request);
        };
        const { scene, frame } = await setUp({ parents: 'refit', workers, mergeStaticInstances: false });
        const sheet = tessellatedQuad(6, 1);
        const cloth = scene.addMesh(sheet, { deforming: true });
        const rock = scene.addMesh(tessellatedQuad(3, 1));
        scene.addInstance({ mesh: cloth, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], instanceIndex: 1 });
        scene.addInstance({ mesh: rock, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1], instanceIndex: 2 });
        frame();
        await scene.whenRefined();
        scene.releaseRefinements();
        frame();
        frame();
        expect(scene.counts.onTinybvhTrees).toBe(2);
        // The cloth (deforming) keeps its parents, the rock's build leaves them out.
        expect(requests.map((request) => request.parents).sort()).toEqual([false, true]);
        scene.setPoseCpu(cloth, hill(sheet));
        frame();
        expect(scene.deformingCounts.cpuPoses).toBe(1);
        const hit = createCpuTracer(readScene(scene)).closest({ origin: [0.1, 0.1, 5], direction: [0, 0, -1], tMin: 0, tMax: 100 });
        expect(hit?.instanceIndex).toBe(1);
        expect(hit!.t).toBeLessThan(4.6);
        scene.dispose();
        workers.dispose();
    });

    it('refuses an unknown value', async () => {
        await expect(setUp({ parents: 'never' as never })).rejects.toThrow(/parents/);
    });
});

describe("refitBoxes: 'shared' (deforming meshes' refits take turns with one box buffer)", () => {
    it('binds one buffer sized for the largest mesh in every refit of the pass, and saves the others', async () => {
        const memory: Record<string, number> = {};
        for (const refitBoxes of ['per-mesh', 'shared'] as const) {
            const gpu = createFakeGpu({}, { features: ['subgroups'], kernels: SCENE_KERNELS });
            const scene = await SceneTrees.create(gpu.device, { gpuFirstTrees: false, topLevel: 'cpu', refitBoxes });
            const boxBuffers: FakeBuffer[] = [];
            const frame = () => {
                const encoder = gpu.createEncoder();
                scene.update(encoder);
                const commands = (encoder as unknown as { commands: { dispatch?: { entryPoint: string; bindings: ReadonlyMap<number, { buffer: FakeBuffer }> } }[] }).commands;
                for (const command of commands) if (command.dispatch?.entryPoint === 'meshRefitTriangles') boxBuffers.push(command.dispatch.bindings.get(3)!.buffer);
                gpu.submit(encoder);
            };
            const sizes = [3, 8, 5];
            const meshes = sizes.map((cells) => scene.addMesh(tessellatedQuad(cells, 1), { deforming: true }));
            for (const mesh of meshes) scene.addInstance({ mesh, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] });
            frame();
            await scene.whenRefined();
            frame();
            sizes.forEach((cells, index) => scene.setPose(meshes[index], { positionBuffer: gpu.device.createBuffer({ size: (cells + 1) ** 2 * 12, usage: 0x80 }), vertexCount: (cells + 1) ** 2 }));
            frame();
            expect(boxBuffers).toHaveLength(3);
            if (refitBoxes === 'shared') {
                expect(new Set(boxBuffers).size).toBe(1);
                // The largest mesh's boxes: 8 x 8 cells, 2 triangles each, 32 bytes per triangle.
                expect(boxBuffers[0].size).toBe(8 * 8 * 2 * 32);
            } else {
                expect(new Set(boxBuffers).size).toBe(3);
            }
            memory[refitBoxes] = scene.memory.meshRefits;
            scene.dispose();
        }
        expect(memory['per-mesh'] - memory.shared).toBe((3 * 3 * 2 + 5 * 5 * 2) * 32);
    });
});

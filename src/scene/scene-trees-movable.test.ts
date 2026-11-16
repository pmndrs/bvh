/**
 * A movable scene file loaded into SceneTrees (`placeMovableScene`): every
 * mesh tree and merged group's tree is placed before the first update, so no
 * build starts (neither the GPU's nor tinybvh's); the scene traces what the
 * file describes (`assembleMovableScene`), and after an instance moves, what a
 * fresh build with it moved gives. Groups that don't match the stored trees
 * are built instead, with a warning. The fake GPU and the stand-in H-PLOC are
 * scene-trees-api.test.ts's.
 */

import { describe, expect, it } from 'vitest';
import { createFakeGpu, type FakeBinding, type FakeBuffer } from '../../tests/support/fake-gpu.ts';
import { FakeHploc } from '../../tests/support/fake-hploc.ts';
import { TOP_HEADER_WORD } from '../gpu-layout.ts';
import { quantizeNodeWords } from '../quantized-nodes.ts';
import { assembleMovableScene, buildMovableScene, deserializeMovableScene, placeMovableScene, serializeMovableScene } from '../serialize-movable-scene.ts';
import { createTinyBvhBuilder } from '../tinybvh.ts';
import { createCpuTracer } from '../trace/cpu-traversal.ts';
import type { MeshGeometry } from '../trace/mesh-bvh.ts';
import { buildMeshBvhWithTinyBvh } from '../trace/mesh-bvh-topology.ts';
import { createRandom, randomAffine, randomRays, randomTriangleSoup, tessellatedQuad } from '../trace/oracle-fixtures.ts';
import { buildTraceScene, type TraceSceneData, type TraceSceneInstance } from '../trace/scene-assembly.ts';
import type { TopLevelBvh } from '../trace/top-level.ts';
import { GpuFirstTrees } from './gpu-first-trees.ts';
import { SceneTrees, type SceneBuildEvent, type SceneTreesOptions } from './scene-trees.ts';

const tinyBvh = await createTinyBvhBuilder();

/** The quantize pass (gpu-quantize.ts, `bvhqQuantize`) on the CPU. */
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
        topLevel: { buffer: top, instanceCount: header[TOP_HEADER_WORD.instanceCount], nodeCount: header[TOP_HEADER_WORD.nodeCount] } as TopLevelBvh,
    } as unknown as TraceSceneData;
}

/** A level (meshes 0-2, static, used once: they merge) and props (meshes 3 and 4, instanced, movable). */
function sceneInputs() {
    const random = createRandom(17);
    const geometries: MeshGeometry[] = [tessellatedQuad(6, 2), randomTriangleSoup(random, 70, 1, 0.4), tessellatedQuad(3, 1), randomTriangleSoup(random, 50, 0.8, 0.4), tessellatedQuad(4, 0.7)];
    const instances: TraceSceneInstance[] = [
        { meshIndex: 0, instanceIndex: 0, objectToWorld: randomAffine(random, { extent: 2 }), material: 1, static: true },
        { meshIndex: 1, instanceIndex: 1, objectToWorld: randomAffine(random, { extent: 2, mirror: true }), static: true },
        { meshIndex: 2, instanceIndex: 2, objectToWorld: randomAffine(random, { extent: 2 }), material: 2, static: true },
        { meshIndex: 3, instanceIndex: 10, objectToWorld: randomAffine(random, { extent: 2 }), material: 3 },
        { meshIndex: 3, instanceIndex: 11, objectToWorld: randomAffine(random, { extent: 2, mirror: true }), mask: 0xf0 },
        { meshIndex: 4, instanceIndex: 12, objectToWorld: randomAffine(random, { extent: 2 }) },
    ];
    return { geometries, instances, rays: randomRays(random, 2000, [-4, -4, -4, 4, 4, 4], 1), random };
}

const inputs = sceneInputs();

type Variant = { name: string; triangleLeaves: 'records' | 'indexed'; nodeFormat: 'f32' | 'quantized'; scene: Partial<SceneTreesOptions> };

const build = (variant: Pick<Variant, 'triangleLeaves'>) => (geometry: MeshGeometry) => buildMeshBvhWithTinyBvh(geometry, tinyBvh, { triangleLeaves: variant.triangleLeaves, maxLeafPrimitives: 4 });

async function setUp(variant: Variant, options: Partial<SceneTreesOptions> = {}) {
    const gpu = createFakeGpu({}, { kernels: { bvhqQuantize: quantizeStandIn } });
    const hploc = new FakeHploc(gpu.device);
    const builds: SceneBuildEvent[] = [];
    const warnings: string[] = [];
    const scene = await SceneTrees.create(gpu.device, {
        workers: { worker: 'never' },
        gpuMaxBatchBytes: Infinity,
        triangleLeaves: variant.triangleLeaves,
        nodeFormat: variant.nodeFormat,
        ...variant.scene,
        ...options,
        onBuild: (event) => builds.push(event),
        warn: (message) => warnings.push(message),
        firstTreeBuilder: new GpuFirstTrees(gpu.device, { hploc }),
    });
    const frame = () => {
        const encoder = gpu.createEncoder();
        scene.update(encoder);
        gpu.submit(encoder);
    };
    return { scene, hploc, builds, warnings, frame };
}

/** Closest hits (instance, triangle, distance, material) and any hits of the scene against `expected`'s, masks included. */
function expectSameHits(scene: SceneTrees, expected: TraceSceneData): void {
    const got = createCpuTracer(readScene(scene));
    const want = createCpuTracer(expected);
    let hits = 0;
    inputs.rays.forEach((ray, index) => {
        for (const mask of [0xff, 0x0f]) {
            const a = got.closest(ray, { mask });
            const b = want.closest(ray, { mask });
            expect(a && [a.instanceIndex, a.triangleIndex, a.t, a.material, a.frontFace], `ray ${index}`).toEqual(b && [b.instanceIndex, b.triangleIndex, b.t, b.material, b.frontFace]);
            expect(got.any(ray, { mask }), `ray ${index}`).toBe(want.any(ray, { mask }));
            if (b) hits++;
        }
    });
    expect(hits).toBeGreaterThan(200);
}

describe('SceneTrees with a movable scene file (placeMovableScene)', () => {
    const variants: Variant[] = [
        { name: 'records, quantized', triangleLeaves: 'records', nodeFormat: 'quantized', scene: {} },
        { name: 'records, f32', triangleLeaves: 'records', nodeFormat: 'f32', scene: {} },
        { name: 'indexed, quantized (weld: false)', triangleLeaves: 'indexed', nodeFormat: 'quantized', scene: { weld: false } },
    ];
    for (const variant of variants) {
        it(`${variant.name}: nothing builds, the walk agrees with the file's scene, and a moved instance with a fresh build`, async () => {
            const data = await buildMovableScene(inputs.geometries, inputs.instances, { build: build(variant), triangleLeaves: variant.triangleLeaves, nodeFormat: variant.nodeFormat });
            expect(data.mergedGroups.length).toBe(1);
            // The file without geometry: the app's geometry, checked against each mesh's hash.
            const loaded = deserializeMovableScene(serializeMovableScene(data), { geometries: inputs.geometries });

            const { scene, hploc, builds, warnings, frame } = await setUp(variant);
            const placed = placeMovableScene(scene, loaded);
            expect(placed).toMatchObject({ meshes: [0, 1, 2, 3, 4], mergedTreesPlaced: true });
            // Asked before the first update (as a page does): it resolves at that update, which builds nothing.
            const built = scene.whenBuilt();
            frame();
            await built;
            expect(hploc.batches).toEqual([]);
            expect(builds).toEqual([]);
            expect(warnings).toEqual([]);
            // Meshes 3 and 4, and the merged group: every tree is the file's.
            expect(scene.counts).toMatchObject({ mergedGroups: 1, onAppTrees: 3, onGpuTrees: 0, onTinybvhTrees: 0, waitingForFirstTree: 0 });
            expect(scene.groupStatus(0)).toMatchObject({ builtBy: 'app', sourceInstances: [0, 1, 2] });
            expect([0, 1, 2].map((instance) => scene.instanceStatus(placed.instances[instance]).mergedInto)).toEqual([0, 0, 0]);
            expectSameHits(scene, assembleMovableScene(loaded));

            // The props move; the merged level can't.
            const moved = loaded.instances.map((instance) => (instance.static ? instance : { ...instance, objectToWorld: randomAffine(inputs.random, { extent: 2 }) }));
            moved.forEach((instance, index) => !instance.static && scene.setTransform(placed.instances[index], instance.objectToWorld));
            expect(() => scene.setTransform(placed.instances[0], moved[3].objectToWorld)).toThrow(/merged/);
            frame();
            const freshScene = await buildTraceScene(inputs.geometries, moved, { build: build(variant), triangleLeaves: variant.triangleLeaves, nodeFormat: variant.nodeFormat });
            expectSameHits(scene, freshScene);
            expect(builds).toEqual([]);
            scene.dispose();
        });
    }

    it('groups that are not the bake\'s are built instead, with a warning; a scene past its first update merges nothing more', async () => {
        const variant = variants[0];
        const loaded = deserializeMovableScene(serializeMovableScene(await buildMovableScene(inputs.geometries, inputs.instances, { build: build(variant) }), { geometry: true }));

        // A static instance moved since the bake (the same triangles, other world-space geometry), or another static
        // instance before the first update: the group is not the bake's.
        const otherPlace = randomAffine(inputs.random, { extent: 2 });
        const cases = [
            { scene: { ...loaded, instances: loaded.instances.map((instance, index) => (index === 1 ? { ...instance, objectToWorld: otherPlace } : instance)) }, extra: false, reason: /world-space geometry is not the one its tree was baked from/ },
            { scene: loaded, extra: true, reason: /group 0 has 168 triangles, its tree 160/ },
        ];
        for (const { scene: file, extra, reason } of cases) {
            const { scene, warnings, frame } = await setUp(variant, { gpuFirstTrees: false });
            placeMovableScene(scene, file);
            if (extra) scene.addInstance({ mesh: scene.addMesh(tessellatedQuad(2, 0.5)), objectToWorld: otherPlace, instanceIndex: 20, static: true });
            frame();
            await scene.whenBuilt();
            frame();
            expect(warnings).toEqual([expect.stringMatching(/the trees don't match the static groups the first update merged .*; building the groups instead/)]);
            expect(warnings[0]).toMatch(reason);
            expect(scene.groupStatus(0).builtBy).toBe('tinybvh');
            if (!extra) expectSameHits(scene, await buildTraceScene(inputs.geometries, file.instances, { build: build(variant) }));
            expect(scene.placeMergedTrees(loaded.mergedGroups)).toBe(false);
            scene.dispose();
        }

        // A scene that doesn't merge: the static instances stay instances, and the meshes only they used are built.
        const unmerged = await setUp(variant, { gpuFirstTrees: false, mergeStaticInstances: false });
        expect(placeMovableScene(unmerged.scene, loaded).mergedTreesPlaced).toBe(false);
        unmerged.frame();
        await unmerged.scene.whenBuilt();
        unmerged.frame();
        expect(unmerged.scene.counts).toMatchObject({ mergedGroups: 0, onAppTrees: 2, onTinybvhTrees: 3 });
        expectSameHits(unmerged.scene, await buildTraceScene(inputs.geometries, loaded.instances, { build: build(variant), mergeStaticInstances: false }));
        unmerged.scene.dispose();
    });

    it('refuses a scene of other formats, and meshes without geometry', async () => {
        const data = await buildMovableScene(inputs.geometries, inputs.instances, { build: build({ triangleLeaves: 'indexed' }), triangleLeaves: 'indexed' });
        const loaded = deserializeMovableScene(serializeMovableScene(data));
        const records = await setUp(variants[0]);
        expect(() => placeMovableScene(records.scene, loaded)).toThrow(/triangleLeaves: 'indexed'/);
        const indexed = await setUp(variants[2]);
        expect(() => placeMovableScene(indexed.scene, loaded)).toThrow(/mesh 0 has no geometry/);
        const welded = await setUp({ ...variants[2], scene: {} });
        expect(() => placeMovableScene(welded.scene, { ...data, mergedGroups: [] })).toThrow(/weld: false/);
        expect(() => welded.scene.placeMergedTrees(data.mergedGroups)).toThrow(/weld: false/);
        const f32 = await setUp(variants[1]);
        const quantized = await buildMovableScene(inputs.geometries, inputs.instances, { build: build(variants[1]) });
        expect(() => placeMovableScene(f32.scene, quantized)).toThrow(/nodeFormat: 'quantized'/);
        for (const { scene } of [records, indexed, welded, f32]) scene.dispose();
    });
});

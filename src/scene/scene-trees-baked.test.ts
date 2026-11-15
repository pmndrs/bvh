/**
 * Baked trees in SceneTrees: a file per mesh (`serializeLayout` with the
 * mesh's geometry), read with `deserializeLayout(bytes, { geometry })` and
 * given to `placeTree` before the first update. No build starts (neither the
 * GPU's nor tinybvh's), and the scene traces what `assembleTraceScene` makes
 * of the same trees, in every format: f32 and quantized trees, records, and
 * an indexed bake next to the mesh's own positions (`weld: false`). The fake
 * GPU and the stand-in H-PLOC are scene-trees-api.test.ts's.
 */

import { describe, expect, it } from 'vitest';
import { createFakeGpu, type FakeBinding, type FakeBuffer } from '../../tests/support/fake-gpu.ts';
import { FakeHploc } from '../../tests/support/fake-hploc.ts';
import { TOP_HEADER_WORD, type GpuLayoutMeshBvh } from '../gpu-layout.ts';
import { quantizeNodeWords } from '../quantized-nodes.ts';
import { deserializeLayout, serializeLayout } from '../serialize.ts';
import { createCpuTracer } from '../trace/cpu-traversal.ts';
import type { MeshGeometry } from '../trace/mesh-bvh.ts';
import { buildMeshBvhAsync } from '../trace/mesh-bvh-build.ts';
import { createRandom, randomTriangleSoup, tessellatedQuad } from '../trace/oracle-fixtures.ts';
import { assembleTraceScene, type TraceSceneData } from '../trace/scene-assembly.ts';
import type { TopLevelBvh } from '../trace/top-level.ts';
import { GpuFirstTrees } from './gpu-first-trees.ts';
import { SceneTrees, type SceneBuildEvent, type SceneTreesOptions } from './scene-trees.ts';

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

/** The quads lie in z = 0, so the rays go down z. */
const translate = (x: number, y: number) => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, 0, 1];

describe('SceneTrees with baked trees (placeTree)', () => {
    const random = createRandom(7);
    const geometries: MeshGeometry[] = [tessellatedQuad(4, 1), randomTriangleSoup(random, 80, 1, 0.4), tessellatedQuad(7, 0.8)];
    const placements = [translate(0, 0), translate(3, 0), translate(0, 3), translate(3, 3)];
    const meshOf = [0, 1, 2, 1];

    const variants: { name: string; scene: Partial<SceneTreesOptions>; tree: { triangleLeaves?: 'indexed'; nodeFormat?: 'quantized' } }[] = [
        { name: 'f32 records trees, quantized scene', scene: {}, tree: {} },
        { name: 'f32 records trees, f32 scene', scene: { nodeFormat: 'f32' }, tree: {} },
        { name: 'quantized trees', scene: {}, tree: { nodeFormat: 'quantized' } },
        { name: 'indexed trees next to the meshes\' positions', scene: { triangleLeaves: 'indexed', weld: false }, tree: { triangleLeaves: 'indexed' } },
        { name: 'quantized indexed trees, packed positions', scene: { triangleLeaves: 'indexed', weld: false, positionLayout: 'packed' }, tree: { triangleLeaves: 'indexed', nodeFormat: 'quantized' } },
    ];
    for (const variant of variants) {
        it(`${variant.name}: placed before the first update, nothing builds, and the walk agrees with an assembled scene`, async () => {
            // The bake: one file per mesh, written with its geometry.
            const files = await Promise.all(geometries.map(async (geometry) => serializeLayout(await buildMeshBvhAsync(geometry, { maxLeafPrimitives: 4, ...variant.tree }), { geometry })));

            const gpu = createFakeGpu({}, { kernels: { bvhqQuantize: quantizeStandIn } });
            const hploc = new FakeHploc(gpu.device);
            const builds: SceneBuildEvent[] = [];
            const scene = await SceneTrees.create(gpu.device, {
                workers: { worker: 'never' },
                gpuMaxBatchBytes: Infinity,
                ...variant.scene,
                onBuild: (event) => builds.push(event),
                firstTreeBuilder: new GpuFirstTrees(gpu.device, { hploc }),
            });
            // The load: deserializeLayout checks each file against the mesh's geometry (its hash), placeTree takes it.
            const trees: GpuLayoutMeshBvh[] = [];
            geometries.forEach((geometry, index) => {
                const mesh = scene.addMesh(geometry);
                const tree = deserializeLayout(files[index], { geometry });
                trees.push(tree);
                scene.placeTree(mesh, tree);
            });
            placements.forEach((objectToWorld, index) => scene.addInstance({ mesh: meshOf[index], objectToWorld, instanceIndex: 20 + index }));
            const encoder = gpu.createEncoder();
            scene.update(encoder);
            gpu.submit(encoder);
            await scene.whenBuilt();
            expect(hploc.batches).toEqual([]);
            expect(builds).toEqual([]);
            expect(scene.counts).toMatchObject({ onAppTrees: 3, onGpuTrees: 0, onTinybvhTrees: 0, waitingForFirstTree: 0 });

            const indexed = variant.tree.triangleLeaves === 'indexed';
            const assembled = assembleTraceScene(
                trees,
                placements.map((objectToWorld, index) => ({ meshIndex: meshOf[index], instanceIndex: 20 + index, objectToWorld })),
                {},
                { nodeFormat: scene.nodeFormat, ...(indexed ? { triangleLeaves: 'indexed', geometries, weld: false, positionLayout: scene.positionLayout } : {}) },
            );
            const got = createCpuTracer(readScene(scene));
            const expected = createCpuTracer(assembled);
            let hits = 0;
            for (let i = 0; i < 400; i++) {
                const ray = { origin: [((i * 37) % 97) / 18 - 1.2, ((i * 53) % 89) / 17 - 1.2, 3] as const, direction: [0.05, 0.03, -1] as const, tMin: 0, tMax: 100 };
                const a = got.closest(ray);
                const b = expected.closest(ray);
                expect(a && [a.instanceIndex, a.triangleIndex, a.t], `ray ${i}`).toEqual(b && [b.instanceIndex, b.triangleIndex, b.t]);
                if (b) hits++;
            }
            expect(hits).toBeGreaterThan(150);
            scene.dispose();
        });
    }

    it('refuses a baked file next to other geometry before it reaches the scene', async () => {
        const [quad, soup] = geometries;
        const file = serializeLayout(await buildMeshBvhAsync(quad, { triangleLeaves: 'indexed' }), { geometry: quad });
        const moved = { positions: quad.positions.map((value, index) => (index === 1 ? value + 1 : value)), indices: quad.indices };
        expect(() => deserializeLayout(file, { geometry: moved })).toThrow(/baked from other geometry/);
        expect(() => deserializeLayout(file, { geometry: soup })).toThrow(/other geometry/);
    });
});

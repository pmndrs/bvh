/**
 * Scene files (serialize-scene.ts): every format a scene can have round-trips
 * byte for byte (writing a loaded scene gives the same bytes), the loaded
 * scene traces the same hits as the assembled one, and files from another
 * version, of another kind or cut short are refused with a message that says
 * what to do.
 */

import { describe, expect, it } from 'vitest';
import { GPU_LAYOUT_VERSION } from './gpu-layout.ts';
import { TRIANGLE_MATERIAL_NONE } from './mesh-input.ts';
import { deserializeTraceScene, serializeTraceScene } from './serialize-scene.ts';
import { isSerializedCurrent, readSerializedHeader, SERIALIZED_SCENE_FORMAT_VERSION, serializeLayout } from './serialize.ts';
import { createTinyBvhBuilder } from './tinybvh.ts';
import { clusterTraceScene } from './trace/clustered-scene.ts';
import { createCpuTracer } from './trace/cpu-traversal.ts';
import type { MeshGeometry } from './trace/mesh-bvh.ts';
import { buildMeshBvhWithTinyBvh } from './trace/mesh-bvh-topology.ts';
import { createRandom, randomAffine, randomRays, randomTriangleSoup, tessellatedQuad } from './trace/oracle-fixtures.ts';
import { buildTraceScene, type TraceSceneData, type TraceSceneInstance } from './trace/scene-assembly.ts';

const tinyBvh = await createTinyBvhBuilder();

/** Materials 0-6 and "none" in turns. */
function withMaterials(geometry: MeshGeometry, seed: number): MeshGeometry {
    const count = (geometry.indices ? geometry.indices.length : geometry.positions.length / 3) / 3;
    return { ...geometry, triangleMaterials: Uint16Array.from({ length: count }, (_, k) => ((k + seed) % 4 === 3 ? TRIANGLE_MATERIAL_NONE : (k * 7 + seed) % 7)) };
}

/** Meshes with and without triangle materials, instances with materials and masks; static ones merge (two share a mesh and don't). */
function sceneInputs() {
    const random = createRandom(41);
    const geometries: MeshGeometry[] = [
        withMaterials(tessellatedQuad(5, 1), 1),
        randomTriangleSoup(random, 60, 1, 0.5),
        withMaterials(randomTriangleSoup(random, 50, 1, 0.5), 2),
        { ...tessellatedQuad(3, 1), triangleFlags: Uint32Array.from({ length: 18 }, (_, k) => k % 2) },
        withMaterials(tessellatedQuad(2, 1), 4),
    ];
    const instances: TraceSceneInstance[] = geometries.map((_, meshIndex) => ({
        meshIndex,
        instanceIndex: 10 + meshIndex,
        objectToWorld: randomAffine(random, { extent: 3, mirror: meshIndex % 2 === 1 }),
        material: meshIndex % 2 === 0 ? 40 + meshIndex : undefined,
        mask: meshIndex === 1 ? 0x0f : undefined,
        static: meshIndex !== 4,
    }));
    instances.push({ meshIndex: 4, instanceIndex: 30, objectToWorld: randomAffine(random, { extent: 3 }), material: 50 });
    const rays = randomRays(random, 3000, [-5, -5, -5, 5, 5, 5], 1);
    return { geometries, instances, rays };
}

const inputs = sceneInputs();

function build(options: { triangleLeaves: 'records' | 'indexed'; nodeFormat: 'f32' | 'quantized'; merge: boolean; positionLayout?: 'vec4' | 'packed' }): Promise<TraceSceneData> {
    return buildTraceScene(inputs.geometries, inputs.instances, {
        build: (geometry) => buildMeshBvhWithTinyBvh(geometry, tinyBvh, { triangleLeaves: options.triangleLeaves, maxLeafPrimitives: 4 }),
        triangleLeaves: options.triangleLeaves,
        nodeFormat: options.nodeFormat,
        positionLayout: options.positionLayout,
        mergeStaticInstances: options.merge,
    });
}

/** The loaded scene traces what the assembled one does: closest hits (instance, triangle, distance, material, face) and any hits, masks and the filter included. */
function expectSameHits(loaded: TraceSceneData, scene: TraceSceneData): void {
    const options = { triangleFilter: ({ triangle, frontFace }: { triangle: number; frontFace: boolean }) => triangle % 3 !== 1 || !frontFace };
    const a = createCpuTracer(loaded, options);
    const b = createCpuTracer(scene, options);
    let hits = 0;
    const materials = new Set<number>();
    inputs.rays.forEach((ray, index) => {
        for (const mask of [0xff, 0xf0]) {
            const got = a.closest(ray, { mask });
            const expected = b.closest(ray, { mask });
            expect(got, `ray ${index}`).toEqual(expected);
            expect(a.any(ray, { mask })).toBe(b.any(ray, { mask }));
            if (expected) {
                hits++;
                materials.add(expected.material);
            }
        }
    });
    expect(hits).toBeGreaterThan(300);
    expect(materials.size).toBeGreaterThan(4); // triangle materials, instance materials and none
}

describe('serializeTraceScene / deserializeTraceScene', () => {
    for (const triangleLeaves of ['records', 'indexed'] as const) {
        for (const nodeFormat of ['f32', 'quantized'] as const) {
            for (const merge of [false, true]) {
                it(`round-trips a ${triangleLeaves}, ${nodeFormat}, ${merge ? 'merged' : 'instanced'} scene byte for byte, and its hits`, async () => {
                    const scene = await build({ triangleLeaves, nodeFormat, merge });
                    if (merge) expect(scene.mergedGroups.length).toBeGreaterThan(0);
                    const bytes = serializeTraceScene(scene);
                    const loaded = deserializeTraceScene(bytes);

                    expect(new Uint8Array(loaded.nodes)).toEqual(new Uint8Array(scene.nodes));
                    expect(new Uint8Array(loaded.triangles)).toEqual(new Uint8Array(scene.triangles));
                    expect(new Uint8Array(loaded.topLevel.buffer)).toEqual(new Uint8Array(scene.topLevel.buffer));
                    expect(new Uint8Array(loaded.positions)).toEqual(new Uint8Array(scene.positions));
                    expect(loaded.topLevel).toEqual(scene.topLevel);
                    expect(loaded.meshes).toEqual(scene.meshes);
                    expect(loaded.mergedGroups).toEqual(scene.mergedGroups);
                    const { nodes: _n, triangles: _t, positions: _p, topLevel: _l, meshes: _m, mergedGroups: _g, ...rest } = loaded;
                    const { nodes: _n2, triangles: _t2, positions: _p2, topLevel: _l2, meshes: _m2, mergedGroups: _g2, ...expected } = scene;
                    expect(rest).toEqual(expected); // formats, counts, depths, byteLength
                    // Writing the loaded scene gives the same bytes.
                    expect(new Uint8Array(serializeTraceScene(loaded))).toEqual(new Uint8Array(bytes));
                    expectSameHits(loaded, scene);

                    const header = readSerializedHeader(bytes);
                    expect(header).toMatchObject({
                        kind: 'scene',
                        formatVersion: SERIALIZED_SCENE_FORMAT_VERSION,
                        layoutVersion: GPU_LAYOUT_VERSION,
                        current: true,
                        reason: null,
                        triangleLeaves,
                        nodeFormat,
                        positionLayout: 'vec4',
                        meshCount: inputs.geometries.length,
                        mergedGroupCount: scene.mergedGroups.length,
                        instanceCount: scene.topLevel.instanceCount,
                        vertexCount: scene.vertexCount,
                        requiredStackSize: scene.requiredStackSize,
                        byteLength: bytes.byteLength,
                    });
                });
            }
        }
    }

    it('keeps packed positions and a clustered scene\'s placements', async () => {
        const packed = await build({ triangleLeaves: 'indexed', nodeFormat: 'quantized', merge: true, positionLayout: 'packed' });
        const loaded = deserializeTraceScene(serializeTraceScene(packed));
        expect(loaded.positionLayout).toBe('packed');
        expect(new Uint8Array(loaded.positions)).toEqual(new Uint8Array(packed.positions));
        expectSameHits(loaded, packed);

        const clustered = clusterTraceScene(await build({ triangleLeaves: 'records', nodeFormat: 'quantized', merge: false }), { primitiveLimit: 8 });
        const back = deserializeTraceScene(serializeTraceScene(clustered));
        expect(back.meshes).toEqual(clustered.meshes);
        expectSameHits(back, clustered);
    });

    it('refuses a scene from another version, a tree file, a cut-short file and one too deep for the walk', async () => {
        const scene = await build({ triangleLeaves: 'records', nodeFormat: 'quantized', merge: true });
        const bytes = serializeTraceScene(scene);
        expect(isSerializedCurrent(bytes)).toBe(true);

        const otherLayout = bytes.slice(0);
        new DataView(otherLayout).setUint32(8, GPU_LAYOUT_VERSION + 1, true);
        expect(readSerializedHeader(otherLayout)).toMatchObject({ kind: 'scene', current: false, layoutVersion: GPU_LAYOUT_VERSION + 1 });
        expect(isSerializedCurrent(otherLayout)).toBe(false);
        expect(() => deserializeTraceScene(otherLayout)).toThrow(`GPU layout version ${GPU_LAYOUT_VERSION + 1}, this package uses ${GPU_LAYOUT_VERSION}; rebuild the scene`);

        const otherFormat = bytes.slice(0);
        new DataView(otherFormat).setUint32(4, SERIALIZED_SCENE_FORMAT_VERSION + 1, true);
        expect(() => deserializeTraceScene(otherFormat)).toThrow(`byte format ${SERIALIZED_SCENE_FORMAT_VERSION + 1}, this package reads ${SERIALIZED_SCENE_FORMAT_VERSION}; bake the scene again`);

        const unknownPositions = bytes.slice(0);
        new DataView(unknownPositions).setUint32(12, new DataView(bytes).getUint32(12, true) | (7 << 16), true);
        expect(() => deserializeTraceScene(unknownPositions)).toThrow(/position layout 7/);

        const tooDeep = bytes.slice(0);
        new DataView(tooDeep).setUint32(32, 65, true);
        expect(() => deserializeTraceScene(tooDeep)).toThrow(/needs 65 trace stack entries/);

        expect(() => deserializeTraceScene(bytes.slice(0, bytes.byteLength - 4))).toThrow(/cut short/);
        expect(() => deserializeTraceScene(new ArrayBuffer(16))).toThrow(/header/);
        expect(() => deserializeTraceScene(new ArrayBuffer(64))).toThrow(/bad magic/);
        const tree = serializeLayout(buildMeshBvhWithTinyBvh(inputs.geometries[0], tinyBvh));
        expect(() => deserializeTraceScene(tree)).toThrow(/deserializeLayout/);
        expect(isSerializedCurrent(new ArrayBuffer(64))).toBe(false);
    });

    it('reads from a view into a larger buffer, and returns copies', async () => {
        const scene = await build({ triangleLeaves: 'indexed', nodeFormat: 'f32', merge: false });
        const bytes = new Uint8Array(serializeTraceScene(scene));
        const padded = new Uint8Array(bytes.byteLength + 64);
        padded.set(bytes, 32);
        const loaded = deserializeTraceScene(padded.subarray(32, 32 + bytes.byteLength));
        padded.fill(0);
        expectSameHits(loaded, scene);
    });
});

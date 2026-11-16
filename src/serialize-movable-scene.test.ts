/**
 * Movable scene files (serialize-movable-scene.ts): every format round-trips
 * byte for byte, with and without the meshes' geometry; the scene they
 * describe traces what `buildTraceScene` builds from the same meshes and
 * instances (merged static groups included); geometry the scene wasn't baked
 * from is refused; other versions and other kinds of file are refused with a
 * message that names the right reader. Loading into `SceneTrees` is
 * scene/scene-trees-movable.test.ts.
 */

import { describe, expect, it } from 'vitest';
import { GPU_LAYOUT_VERSION } from './gpu-layout.ts';
import { TRIANGLE_MATERIAL_NONE } from './mesh-input.ts';
import { assembleMovableScene, buildMovableScene, deserializeMovableScene, serializeMovableScene, type MovableSceneData } from './serialize-movable-scene.ts';
import { deserializeTraceScene, serializeTraceScene } from './serialize-scene.ts';
import { deserializeLayout, isSerializedCurrent, readSerializedHeader, SERIALIZED_MOVABLE_SCENE_FORMAT_VERSION, serializeLayout } from './serialize.ts';
import { createTinyBvhBuilder } from './tinybvh.ts';
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

/**
 * A level with props: meshes 0-3 are used once by static instances (they
 * merge), mesh 4 has two movable instances, mesh 5 one movable and one
 * static (a mesh used twice never merges), mesh 6 none yet.
 */
function sceneInputs() {
    const random = createRandom(43);
    const geometries: MeshGeometry[] = [
        withMaterials(tessellatedQuad(5, 1), 1),
        randomTriangleSoup(random, 60, 1, 0.5),
        withMaterials(randomTriangleSoup(random, 50, 1, 0.5), 2),
        { ...tessellatedQuad(3, 1), triangleFlags: Uint32Array.from({ length: 18 }, (_, k) => k % 2) },
        withMaterials(tessellatedQuad(2, 1), 4),
        randomTriangleSoup(random, 40, 1, 0.5),
        tessellatedQuad(2, 1),
    ];
    const instances: TraceSceneInstance[] = [0, 1, 2, 3].map((meshIndex) => ({
        meshIndex,
        instanceIndex: 10 + meshIndex,
        objectToWorld: randomAffine(random, { extent: 3, mirror: meshIndex % 2 === 1 }),
        material: meshIndex % 2 === 0 ? 40 + meshIndex : undefined,
        mask: meshIndex === 1 ? 0x0f : undefined,
        static: true,
    }));
    instances.push(
        { meshIndex: 4, instanceIndex: 30, objectToWorld: randomAffine(random, { extent: 3 }), material: 50 },
        { meshIndex: 4, instanceIndex: 31, objectToWorld: randomAffine(random, { extent: 3, mirror: true }), mask: 0xf0 },
        { meshIndex: 5, instanceIndex: 32, objectToWorld: randomAffine(random, { extent: 3 }) },
        { meshIndex: 5, instanceIndex: 33, objectToWorld: randomAffine(random, { extent: 3 }), static: true, visible: true },
    );
    const rays = randomRays(random, 3000, [-5, -5, -5, 5, 5, 5], 1);
    return { geometries, instances, rays };
}

const inputs = sceneInputs();

type Formats = { triangleLeaves: 'records' | 'indexed'; nodeFormat: 'f32' | 'quantized'; merge: boolean };

const build = (geometry: MeshGeometry, triangleLeaves: 'records' | 'indexed') => buildMeshBvhWithTinyBvh(geometry, tinyBvh, { triangleLeaves, maxLeafPrimitives: 4 });

function bake(formats: Formats, instances = inputs.instances): Promise<MovableSceneData> {
    return buildMovableScene(inputs.geometries, instances, {
        build: (geometry) => build(geometry, formats.triangleLeaves),
        triangleLeaves: formats.triangleLeaves,
        nodeFormat: formats.nodeFormat,
        mergeStaticInstances: formats.merge,
    });
}

function fresh(formats: Formats, instances = inputs.instances): Promise<TraceSceneData> {
    return buildTraceScene(inputs.geometries, instances, {
        build: (geometry) => build(geometry, formats.triangleLeaves),
        triangleLeaves: formats.triangleLeaves,
        nodeFormat: formats.nodeFormat,
        mergeStaticInstances: formats.merge,
    });
}

/** Closest hits (instance, triangle, distance, material, face) and any hits, masks and the filter included. */
export function expectSameHits(got: TraceSceneData, expected: TraceSceneData, rays = inputs.rays): void {
    const options = { triangleFilter: ({ triangle, frontFace }: { triangle: number; frontFace: boolean }) => triangle % 3 !== 1 || !frontFace };
    const a = createCpuTracer(got, options);
    const b = createCpuTracer(expected, options);
    let hits = 0;
    const materials = new Set<number>();
    rays.forEach((ray, index) => {
        for (const mask of [0xff, 0xf0]) {
            const hit = a.closest(ray, { mask });
            const want = b.closest(ray, { mask });
            expect(hit, `ray ${index}`).toEqual(want);
            expect(a.any(ray, { mask })).toBe(b.any(ray, { mask }));
            if (want) {
                hits++;
                materials.add(want.material);
            }
        }
    });
    expect(hits).toBeGreaterThan(300);
    expect(materials.size).toBeGreaterThan(4);
}

describe('buildMovableScene / serializeMovableScene / deserializeMovableScene', () => {
    for (const triangleLeaves of ['records', 'indexed'] as const) {
        for (const nodeFormat of ['f32', 'quantized'] as const) {
            for (const merge of [true, false]) {
                it(`${triangleLeaves}, ${nodeFormat}, ${merge ? 'merged' : 'unmerged'}: round-trips byte for byte, with and without geometry, and traces as buildTraceScene's scene`, async () => {
                    const formats = { triangleLeaves, nodeFormat, merge };
                    const data = await bake(formats);
                    // Meshes 0-3 merge (in groups of equal mask) and get no tree of their own.
                    if (merge) {
                        expect(data.mergedGroups.length).toBeGreaterThan(0);
                        expect(data.meshes.slice(0, 4).map((mesh) => mesh.tree)).toEqual([null, null, null, null]);
                    } else {
                        expect(data.mergedGroups).toEqual([]);
                    }
                    expect(data.meshes.slice(4).every((mesh) => mesh.tree !== null)).toBe(true); // mesh 6 has no instance yet, and a tree all the same
                    expect(data.meshes.every((mesh) => !mesh.tree || (mesh.tree.nodeFormat ?? 'f32') === nodeFormat)).toBe(true);

                    for (const geometry of [false, true]) {
                        const bytes = serializeMovableScene(data, { geometry });
                        const header = readSerializedHeader(bytes);
                        expect(header).toMatchObject({
                            kind: 'movable-scene',
                            formatVersion: SERIALIZED_MOVABLE_SCENE_FORMAT_VERSION,
                            layoutVersion: GPU_LAYOUT_VERSION,
                            current: true,
                            reason: null,
                            triangleLeaves,
                            nodeFormat,
                            meshCount: inputs.geometries.length,
                            treeCount: data.meshes.filter((mesh) => mesh.tree).length,
                            mergedGroupCount: data.mergedGroups.length,
                            instanceCount: inputs.instances.length,
                            geometry,
                            byteLength: bytes.byteLength,
                        });
                        const loaded = deserializeMovableScene(bytes, geometry ? {} : { geometries: inputs.geometries });
                        // The same bytes back out.
                        expect(new Uint8Array(serializeMovableScene(loaded, { geometry }))).toEqual(new Uint8Array(bytes));
                        expect(loaded.instances.map((instance) => [instance.instanceIndex, instance.static ?? false])).toEqual(inputs.instances.map((instance) => [instance.instanceIndex, instance.static ?? false]));
                        if (geometry) {
                            // The file's own geometry, bit for bit.
                            loaded.meshes.forEach((mesh, index) => {
                                const source = inputs.geometries[index];
                                expect(Array.from(mesh.geometry!.positions)).toEqual(Array.from(source.positions));
                                expect(mesh.geometry!.indices ? Array.from(mesh.geometry!.indices) : null).toEqual(source.indices ? Array.from(source.indices) : null);
                                expect(mesh.geometry!.triangleMaterials ? Array.from(mesh.geometry!.triangleMaterials) : null).toEqual(source.triangleMaterials ? Array.from(source.triangleMaterials) : null);
                                expect(mesh.geometry!.triangleFlags ? Array.from(mesh.geometry!.triangleFlags) : null).toEqual(source.triangleFlags ? Array.from(source.triangleFlags) : null);
                            });
                        }
                        expectSameHits(assembleMovableScene(loaded), await fresh(formats));
                    }
                });
            }
        }
    }

    it('a moved instance traces as a fresh build with it moved', async () => {
        const formats = { triangleLeaves: 'records', nodeFormat: 'quantized', merge: true } as const;
        const loaded = deserializeMovableScene(serializeMovableScene(await bake(formats), { geometry: true }));
        const random = createRandom(5);
        const moved = loaded.instances.map((instance) => (instance.static ? instance : { ...instance, objectToWorld: randomAffine(random, { extent: 3 }) }));
        expectSameHits(assembleMovableScene({ ...loaded, instances: moved }), await fresh(formats, moved));
    });

    it('keeps deforming meshes f32 and out of the merge', async () => {
        const data = await buildMovableScene(inputs.geometries, inputs.instances, {
            build: (geometry) => build(geometry, 'records'),
            deforming: [true, undefined, { bounds: [-9, -9, -9, 9, 9, 9] }],
        });
        expect(data.meshes[0]).toMatchObject({ deforming: true, tree: { triangleCount: 50 } });
        expect(data.meshes[0].tree!.nodeFormat).toBeUndefined();
        expect(data.meshes[2].deforming).toEqual({ bounds: [-9, -9, -9, 9, 9, 9] });
        expect(data.meshes[1].tree).toBeNull(); // merged
        const loaded = deserializeMovableScene(serializeMovableScene(data), { geometries: inputs.geometries });
        expect(loaded.meshes.map((mesh) => mesh.deforming)).toEqual(data.meshes.map((mesh) => mesh.deforming));
        const sameInstances = inputs.instances.map((instance) => (instance.meshIndex === 0 || instance.meshIndex === 2 ? { ...instance, static: false } : instance));
        expectSameHits(assembleMovableScene(loaded), await fresh({ triangleLeaves: 'records', nodeFormat: 'quantized', merge: true }, sameInstances));
    });

    it('refuses geometry the scene was not baked from: a tree\'s mesh, and a merged-away mesh', async () => {
        const data = await bake({ triangleLeaves: 'indexed', nodeFormat: 'quantized', merge: true });
        const bytes = serializeMovableScene(data);
        const nudged = (geometry: MeshGeometry): MeshGeometry => ({ ...geometry, positions: geometry.positions.map((value, index) => (index === 4 ? value + 0.5 : value)) });
        // Mesh 4 has a tree; mesh 1 was merged away and has only its hash.
        for (const mesh of [4, 1]) {
            const geometries = inputs.geometries.map((geometry, index) => (index === mesh ? nudged(geometry) : geometry));
            expect(() => deserializeMovableScene(bytes, { geometries }), `mesh ${mesh}`).toThrow(new RegExp(`mesh ${mesh} was baked from other geometry`));
        }
        expect(() => deserializeMovableScene(bytes, { geometries: inputs.geometries.slice(1) })).toThrow(/7 meshes, and 6 geometries/);
        // Without geometry the file loads, and can't be assembled (indexed, merged) or placed.
        const bare = deserializeMovableScene(bytes);
        expect(bare.meshes.every((mesh) => mesh.geometry === undefined)).toBe(true);
        expect(() => assembleMovableScene(bare)).toThrow(/mesh 0 has no geometry/);
    });

    it('refuses other versions, other kinds of file, and cut-short files; the other readers refuse it', async () => {
        const data = await bake({ triangleLeaves: 'records', nodeFormat: 'quantized', merge: true });
        const bytes = serializeMovableScene(data, { geometry: true });
        expect(isSerializedCurrent(bytes)).toBe(true);

        const otherLayout = bytes.slice(0);
        new DataView(otherLayout).setUint32(8, GPU_LAYOUT_VERSION + 1, true);
        expect(readSerializedHeader(otherLayout)).toMatchObject({ kind: 'movable-scene', current: false });
        expect(isSerializedCurrent(otherLayout)).toBe(false);
        expect(() => deserializeMovableScene(otherLayout)).toThrow(`GPU layout version ${GPU_LAYOUT_VERSION + 1}, this package uses ${GPU_LAYOUT_VERSION}; rebuild the scene`);

        const otherFormat = bytes.slice(0);
        new DataView(otherFormat).setUint32(4, SERIALIZED_MOVABLE_SCENE_FORMAT_VERSION + 1, true);
        expect(() => deserializeMovableScene(otherFormat)).toThrow(`byte format ${SERIALIZED_MOVABLE_SCENE_FORMAT_VERSION + 1}, this package reads ${SERIALIZED_MOVABLE_SCENE_FORMAT_VERSION}; bake the scene again`);

        const unknownBits = bytes.slice(0);
        new DataView(unknownBits).setUint32(12, new DataView(bytes).getUint32(12, true) | (1 << 16), true);
        expect(() => deserializeMovableScene(unknownBits)).toThrow(/bits this package doesn't know/);

        expect(() => deserializeMovableScene(bytes.slice(0, bytes.byteLength - 4))).toThrow(/cut short/);
        expect(() => deserializeMovableScene(new ArrayBuffer(16))).toThrow(/header/);
        expect(() => deserializeMovableScene(new ArrayBuffer(64))).toThrow(/bad magic/);

        const tree = serializeLayout(build(inputs.geometries[0], 'records'));
        expect(() => deserializeMovableScene(tree)).toThrow(/deserializeLayout/);
        const snapshot = serializeTraceScene(await fresh({ triangleLeaves: 'records', nodeFormat: 'quantized', merge: true }));
        expect(() => deserializeMovableScene(snapshot)).toThrow(/deserializeTraceScene/);
        expect(() => deserializeTraceScene(bytes)).toThrow(/deserializeMovableScene/);
        expect(() => deserializeLayout(bytes)).toThrow(/deserializeMovableScene/);
    });

    it('reads from a view into a larger buffer, and returns copies', async () => {
        const formats = { triangleLeaves: 'indexed', nodeFormat: 'f32', merge: true } as const;
        const bytes = new Uint8Array(serializeMovableScene(await bake(formats), { geometry: true }));
        const padded = new Uint8Array(bytes.byteLength + 64);
        padded.set(bytes, 32);
        const loaded = deserializeMovableScene(padded.subarray(32, 32 + bytes.byteLength));
        padded.fill(0);
        expectSameHits(assembleMovableScene(loaded), await fresh(formats));
    });
});

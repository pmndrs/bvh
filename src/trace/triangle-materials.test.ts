/**
 * Per-triangle materials (`MeshInput.triangleMaterials`): every builder writes
 * them into flags bits 16-31 (records), indexed scenes keep them in a table
 * before the mesh's header, merged groups in their side tables, and a hit's
 * material is the triangle's when it has one, else its instance's. The CPU
 * walk is checked against the reference over every format and builder.
 */

import { describe, expect, it } from 'vitest';
import { createTinyBvhBuilder } from '../index.ts';
import { buildHplocOnCpu } from '../hploc-cpu.ts';
import { checkGpuLayout } from '../layout-check.ts';
import { buildLbvhOnCpu } from '../lbvh-cpu.ts';
import { packedTriangleFlags, TRIANGLE_MATERIAL_NONE, triangleMaterialCodes } from '../mesh-input.ts';
import { refitLayoutToPose } from '../refit-cpu.ts';
import { createCpuTracer } from './cpu-traversal.ts';
import { MATERIAL_SHIFT, NO_MATERIAL, TRIANGLE_WORD, TRIANGLE_WORDS } from './layout.ts';
import type { MeshBvh, MeshGeometry } from './mesh-bvh.ts';
import { buildMeshBvhWithTinyBvh } from './mesh-bvh-topology.ts';
import { addToTally, compareAny, compareClosest, newTally } from './oracle-compare.ts';
import { createRandom, randomAffine, randomRays, randomTriangleSoup, tessellatedQuad } from './oracle-fixtures.ts';
import { buildReferenceScene, traceAnyReference, traceClosestReference } from './reference-tracer.ts';
import { assembleTraceScene, buildTraceScene, type TraceSceneInstance } from './scene-assembly.ts';

const tinyBvh = await createTinyBvhBuilder();

const triangles = (geometry: MeshGeometry): number => (geometry.indices ? geometry.indices.length : geometry.positions.length / 3) / 3;

/** Materials 0-6 and "none" in turns, so some triangles take their instance's. */
function withMaterials(geometry: MeshGeometry, seed: number): MeshGeometry {
    const count = triangles(geometry);
    const materials = Uint16Array.from({ length: count }, (_, k) => ((k + seed) % 4 === 3 ? TRIANGLE_MATERIAL_NONE : (k * 7 + seed) % 7));
    return { ...geometry, triangleMaterials: materials };
}

function scene() {
    const random = createRandom(29);
    const geometries: MeshGeometry[] = [
        withMaterials(tessellatedQuad(5, 1), 1),
        randomTriangleSoup(random, 40, 1, 0.5),
        withMaterials(randomTriangleSoup(random, 50, 1, 0.5), 2),
        // Filter flags beside materials: the flags keep bit 0 only, the materials go to bits 16-31.
        { ...withMaterials(tessellatedQuad(3, 1), 3), triangleFlags: Uint32Array.from({ length: 18 }, (_, k) => k % 2) },
        // Used twice: never merged.
        withMaterials(tessellatedQuad(2, 1), 4),
    ];
    const instances: TraceSceneInstance[] = geometries.map((_, meshIndex) => ({
        meshIndex,
        instanceIndex: 10 + meshIndex,
        objectToWorld: randomAffine(random, { extent: 3, mirror: meshIndex % 2 === 1 }),
        material: meshIndex % 2 === 0 ? 40 + meshIndex : undefined,
        static: true,
    }));
    instances.push({ meshIndex: 4, instanceIndex: 30, objectToWorld: randomAffine(random, { extent: 3 }), material: 50, static: true });
    const rays = randomRays(random, 1500, [-5, -5, -5, 5, 5, 5], 1);
    return { geometries, instances, rays };
}

type Build = (geometry: MeshGeometry) => MeshBvh;
const builders: Record<string, (leaves: 'records' | 'indexed') => Build> = {
    tinybvh: (triangleLeaves) => (geometry) => buildMeshBvhWithTinyBvh(geometry, tinyBvh, { triangleLeaves }),
    'lbvh-cpu': (triangleLeaves) => (geometry) => buildLbvhOnCpu(geometry, { triangleLeaves }),
    'hploc-cpu': (triangleLeaves) => (geometry) => buildHplocOnCpu(geometry, { triangleLeaves }),
};

describe('per-triangle materials', () => {
    const { geometries, instances, rays } = scene();
    const reference = buildReferenceScene(geometries, instances);

    for (const [name, builder] of Object.entries(builders)) {
        for (const triangleLeaves of ['records', 'indexed'] as const) {
            for (const nodeFormat of ['f32', 'quantized'] as const) {
                for (const mergeStaticInstances of [false, true]) {
                    it(`${name}, ${triangleLeaves}, ${nodeFormat}, ${mergeStaticInstances ? 'merged' : 'instanced'}: the CPU walk reports the reference's materials`, async () => {
                        const built = await buildTraceScene(geometries, instances, { build: builder(triangleLeaves), triangleLeaves, nodeFormat, mergeStaticInstances });
                        if (mergeStaticInstances) expect(built.mergedGroups.length).toBeGreaterThan(0);
                        const tracer = createCpuTracer(built);
                        const closest = newTally();
                        const any = newTally();
                        let ownMaterial = 0;
                        let instanceMaterial = 0;
                        let none = 0;
                        rays.forEach((ray, index) => {
                            const hit = tracer.closest(ray);
                            const expected = traceClosestReference(reference, ray);
                            addToTally(closest, compareClosest(hit && { hit: true, ...hit }, expected, ray), `ray ${index}`);
                            addToTally(any, compareAny(tracer.any(ray), traceAnyReference(reference, ray)), `ray ${index}`);
                            if (!hit || !expected || expected.ambiguous) return;
                            const source = instances.find((instance) => instance.instanceIndex === hit.instanceIndex)!;
                            const own = geometries[source.meshIndex].triangleMaterials?.[hit.triangleIndex];
                            if (own !== undefined && own !== TRIANGLE_MATERIAL_NONE) {
                                expect(hit.material).toBe(own);
                                ownMaterial++;
                            } else if (source.material !== undefined) {
                                expect(hit.material).toBe(source.material);
                                instanceMaterial++;
                            } else {
                                expect(hit.material).toBe(NO_MATERIAL);
                                none++;
                            }
                        });
                        expect(closest.mismatches, closest.examples.join('\n')).toBe(0);
                        expect(any.mismatches, any.examples.join('\n')).toBe(0);
                        // Every branch of the rule is exercised.
                        expect(ownMaterial).toBeGreaterThan(50);
                        expect(instanceMaterial).toBeGreaterThan(10);
                        expect(none).toBeGreaterThan(10);
                    });
                }
            }
        }
    }

    it('every builder writes material + 1 into flags bits 16-31 of its records, beside the filter bit, and passes the layout check', () => {
        const geometry = geometries[3];
        const codes = triangleMaterialCodes(geometry.triangleMaterials, triangles(geometry), 'test')!;
        for (const [name, builder] of Object.entries(builders)) {
            for (const triangleLeaves of ['records', 'indexed'] as const) {
                const tree = builder(triangleLeaves)(geometry);
                expect(checkGpuLayout(tree, geometry), `${name} ${triangleLeaves}`).toEqual([]);
                if (triangleLeaves === 'indexed') continue;
                const words = new Uint32Array(tree.triangles);
                for (let slot = 0; slot < tree.triangleCount; slot++) {
                    const original = words[slot * TRIANGLE_WORDS + TRIANGLE_WORD.originalIndex];
                    expect(words[slot * TRIANGLE_WORDS + TRIANGLE_WORD.flags], `${name} slot ${slot}`).toBe(((codes[original] << MATERIAL_SHIFT) | (original % 2)) >>> 0);
                }
            }
        }
        // A tree whose flags lack the materials fails the check.
        const bare = buildMeshBvhWithTinyBvh({ ...geometry, triangleMaterials: null }, tinyBvh);
        expect(checkGpuLayout(bare, geometry).join('\n')).toMatch(/flags/);
    });

    it('a CPU refit keeps the flags and material bits', () => {
        const geometry = geometries[0];
        const tree = buildMeshBvhWithTinyBvh(geometry, tinyBvh);
        const moved = geometry.positions.map((value, index) => (index % 3 === 2 ? value + 0.5 : value));
        const posed = refitLayoutToPose(tree, moved, geometry.indices);
        const before = new Uint32Array(tree.triangles);
        const after = new Uint32Array(posed.triangles!);
        for (let slot = 0; slot < tree.triangleCount; slot++) {
            expect(after[slot * TRIANGLE_WORDS + TRIANGLE_WORD.flags]).toBe(before[slot * TRIANGLE_WORDS + TRIANGLE_WORD.flags]);
        }
        expect(posed.bounds[2]).toBeCloseTo(tree.bounds[2] + 0.5, 6);
    });

    it("assembleTraceScene with a prebuilt tree: records and indexed take the same materials (the geometry's, else the tree's bits), and a mismatch throws", () => {
        const geometry = geometries[0];
        const plain = { positions: geometry.positions, indices: geometry.indices };
        const tree = buildMeshBvhWithTinyBvh(geometry, tinyBvh);
        const instances: TraceSceneInstance[] = [{ meshIndex: 0, instanceIndex: 0, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], material: 9 }];
        const down = Array.from({ length: 400 }, (_, k) => ({ origin: [-0.95 + 1.9 * ((k % 20) / 19), -0.95 + 1.9 * (Math.floor(k / 20) / 19), 2] as const, direction: [0, 0, -1] as const, tMin: 0, tMax: 10 }));
        const materials = (scene: ReturnType<typeof assembleTraceScene>) => down.map((ray) => createCpuTracer(scene).closest(ray)?.material ?? -1);
        const records = materials(assembleTraceScene([tree], instances));
        // The tree carries the materials, the geometry doesn't: indexed reads the tree's bits, as records do.
        const indexed = materials(assembleTraceScene([tree], instances, {}, { triangleLeaves: 'indexed', geometries: [plain] }));
        expect(indexed).toEqual(records);
        expect(new Set(records)).toEqual(new Set([1, 9]));
        // Both, agreeing: the same.
        expect(materials(assembleTraceScene([tree], instances, {}, { triangleLeaves: 'indexed', geometries: [geometry] }))).toEqual(records);
        // A tree built without materials: the geometry's are used.
        const bare = buildMeshBvhWithTinyBvh(plain, tinyBvh);
        expect(materials(assembleTraceScene([bare], instances, {}, { triangleLeaves: 'indexed', geometries: [geometry] }))).toEqual(records);
        // Both, disagreeing: throws, naming the mesh and triangle.
        const other = withMaterials(plain, 2);
        expect(() => assembleTraceScene([tree], instances, {}, { triangleLeaves: 'indexed', geometries: [other] })).toThrow(/mesh 0: geometries\[0\]\.triangleMaterials gives triangle \d+/);
    });

    it('takes 0-65534 and 0xffff or 0xffffffff for none, refuses anything else naming the triangle, and app flags still may set bit 0 only', () => {
        const geometry = tessellatedQuad(1, 1);
        expect(Array.from(packedTriangleFlags({ ...geometry, triangleMaterials: [0, 0xfffe] }, 2, 'm')!)).toEqual([1 << 16, 0xffff0000]);
        expect(Array.from(packedTriangleFlags({ ...geometry, triangleMaterials: [0xffff, 0xffffffff] }, 2, 'm')!)).toEqual([0, 0]);
        expect(() => packedTriangleFlags({ ...geometry, triangleMaterials: [1, 0x10000] }, 2, 'mesh 3')).toThrow(/mesh 3: triangleMaterials\[1\] is 65536/);
        expect(() => packedTriangleFlags({ ...geometry, triangleMaterials: [1.5, 0] }, 2, 'm')).toThrow(/triangleMaterials\[0\]/);
        expect(() => packedTriangleFlags({ ...geometry, triangleMaterials: [1] }, 2, 'm')).toThrow(/one entry per triangle/);
        expect(() => buildMeshBvhWithTinyBvh({ ...geometry, triangleFlags: Uint32Array.of(0, 1 << 16) }, tinyBvh)).toThrow(/triangleFlags\[1\]/);
        expect(() => buildMeshBvhWithTinyBvh({ ...geometry, triangleMaterials: [-1, 0] }, tinyBvh)).toThrow(/triangleMaterials\[0\]/);
    });
});

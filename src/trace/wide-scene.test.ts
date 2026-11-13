// Wide scenes (docs/internals/design/wide-nodes.md) against f32 scenes, through the CPU walk:
// instanced and merged, records and indexed, masks and a filter. The triangle test is the same and
// every decoded box holds its exact box, so every hit has the f32 hit's t, and its instance and
// triangle unless another triangle lies at exactly that t (the walk order picks among ties).
// Any-hit blocks the same rays.
import { describe, expect, it } from 'vitest';
import { createTinyBvhBuilder } from '../index.ts';
import { createCpuTracer, type CpuHit } from './cpu-traversal.ts';
import { TRIANGLE_FLAG_FILTER } from './layout.ts';
import type { MeshBvh, MeshGeometry } from './mesh-bvh.ts';
import { buildMeshBvhWithTinyBvh } from './mesh-bvh-topology.ts';
import { createRandom, randomAffine, randomRays, randomTriangleSoup, tessellatedQuad } from './oracle-fixtures.ts';
import { assembleTraceScene, buildTraceScene, meshRefitPlacement, type TraceSceneData, type TraceSceneInstance } from './scene-assembly.ts';
import type { TraceRay } from './rays.ts';
import type { TrianglePasses } from './triangle-filter.ts';
import { traceModuleWgsl } from './wgsl.ts';

const tinyBvh = await createTinyBvhBuilder();
const f32 = (geometry: MeshGeometry): MeshBvh => buildMeshBvhWithTinyBvh(geometry, tinyBvh);

function mixedScene(seed: number): { meshes: MeshGeometry[]; instances: TraceSceneInstance[] } {
    const random = createRandom(seed);
    const meshes: MeshGeometry[] = [];
    const instances: TraceSceneInstance[] = [];
    for (let i = 0; i < 10; i++) {
        const mesh = i < 3 ? tessellatedQuad(8 + i, 2) : randomTriangleSoup(random, 80 + i * 40, 2, 0.5);
        const count = mesh.indices ? mesh.indices.length / 3 : mesh.positions.length / 9;
        meshes.push(i % 3 === 1 ? { ...mesh, triangleFlags: Uint32Array.from({ length: count }, (_, t) => (t % 3 === 0 ? TRIANGLE_FLAG_FILTER : 0)) } : mesh);
        instances.push({ meshIndex: i, instanceIndex: 300 + i, objectToWorld: randomAffine(random, { extent: 8, mirror: i % 4 === 1 }), static: i % 5 !== 4, mask: i % 3 === 2 ? 0x0f : undefined, material: i });
    }
    meshes.push(randomTriangleSoup(random, 1, 2, 0.6));
    for (let k = 0; k < 3; k++) instances.push({ meshIndex: 10, instanceIndex: 10 + k, objectToWorld: randomAffine(random, { extent: 8, mirror: k === 1 }) });
    return { meshes, instances };
}

const filter: TrianglePasses = ({ instance, triangle, frontFace, query }) => (instance * 7 + triangle) % 3 !== 1 && (query === 0 || !frontFace);

function rays(seed: number, count: number): TraceRay[] {
    const random = createRandom(seed);
    return [...randomRays(random, count / 2, [-10, -10, -10, 10, 10, 10], 2, { randomInterval: true }), ...randomRays(random, count / 2, [-8, -8, -8, 8, 8, 8], 0)];
}

function trace(scene: TraceSceneData, traced: readonly TraceRay[], anyHitOrder: 'far' | 'near' = 'far'): { closest: (CpuHit | null)[]; any: boolean[] } {
    const tracer = createCpuTracer(scene, { triangleFilter: filter, anyHitOrder });
    const masks = [undefined, 0x0f, 0xf0];
    return {
        closest: traced.map((ray, index) => tracer.closest(ray, { mask: masks[index % 3] })),
        any: traced.map((ray, index) => tracer.any(ray, { mask: masks[index % 3] })),
    };
}

describe('a wide scene', () => {
    const traced = rays(31, 3000);
    for (const nodeFormat of ['bvh4', 'cwbvh8'] as const) {
        for (const mergeStaticInstances of [false, true]) {
            for (const triangleLeaves of ['records', 'indexed'] as const) {
                it(`${nodeFormat} gives the f32 scene's hits (${mergeStaticInstances ? 'merged' : 'instanced'}, ${triangleLeaves})`, async () => {
                    const { meshes, instances } = mixedScene(17);
                    const options = { build: f32, mergeStaticInstances, triangleLeaves };
                    const exact = await buildTraceScene(meshes, instances, options);
                    const wide = await buildTraceScene(meshes, instances, { ...options, nodeFormat });
                    expect(wide.nodeFormat).toBe(nodeFormat);
                    for (const order of ['far', 'near'] as const) {
                        const expected = trace(exact, traced, order);
                        const actual = trace(wide, traced, order);
                        expect(expected.closest.filter((hit) => hit !== null).length).toBeGreaterThan(traced.length / 10);
                        let ties = 0;
                        expected.closest.forEach((hit, index) => {
                            const got = actual.closest[index];
                            if (hit === null) return expect(got).toBeNull();
                            expect(got?.t).toBe(hit.t);
                            if (got!.instanceIndex !== hit.instanceIndex || got!.triangleIndex !== hit.triangleIndex) ties++;
                            else expect(got).toEqual(hit);
                        });
                        expect(ties).toBeLessThan(traced.length / 200);
                        expect(actual.any).toEqual(expected.any);
                    }
                });
            }
        }
    }

    it('collapses smaller than f32 and is never refit', async () => {
        const { meshes, instances } = mixedScene(5);
        const trees = meshes.map(f32);
        const exact = assembleTraceScene(trees, instances);
        const bvh4 = assembleTraceScene(trees, instances, {}, { nodeFormat: 'bvh4' });
        const cwbvh = assembleTraceScene(trees, instances, {}, { nodeFormat: 'cwbvh8' });
        expect(bvh4.nodes.byteLength).toBeLessThan(exact.nodes.byteLength);
        expect(cwbvh.nodes.byteLength).toBeLessThan(bvh4.nodes.byteLength);
        expect(cwbvh.triangles.byteLength).toBe(exact.triangles.byteLength);
        expect(() => meshRefitPlacement(cwbvh, 0)).toThrow(/not refit/);
        expect(() => assembleTraceScene(trees, instances, {}, { nodeFormat: 'cwbvh16' as never })).toThrow(RangeError);
        expect(() => traceModuleWgsl({ nodeFormat: 'bvh4', anyHitOrder: 'larger' })).toThrow(/larger/);
        expect(() => traceModuleWgsl({ nodeFormat: 'bvh4', wideStack: 'workgroup' })).toThrow(/cwbvh8/);
        expect(traceModuleWgsl({ nodeFormat: 'cwbvh8', wideStack: 'workgroup' })).toContain('var<workgroup> bvhWideSharedStack');
        expect(traceModuleWgsl({ nodeFormat: 'cwbvh8' })).toContain('var<storage, read> bvhTraceNodes: array<vec4u>;');
        // The f32 walk is untouched by the wide formats.
        expect(traceModuleWgsl()).not.toContain('bvhMeshClosest');
    });
});

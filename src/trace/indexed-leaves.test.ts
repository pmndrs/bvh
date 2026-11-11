// Indexed leaves (docs/design/indexed-leaves.md) against records: the same tree in either format,
// and the same hits, bit for bit, from the same scene in either format.
import { describe, expect, it } from 'vitest';
import {
    buildHplocOnCpu,
    buildLbvhOnCpu,
    checkGpuLayout,
    CpuMeshBvh,
    createTinyBvhBuilder,
    deserializeLayout,
    GPU_LAYOUT_INDEXED_TRIANGLE_BYTES,
    indexedFromRecords,
    recordsFromIndexed,
    serializeLayout,
    weldExactBits,
    weldPositions,
    weldWithinTolerance,
} from '../index.ts';
import { createCpuTracer, type CpuHit } from './cpu-traversal.ts';
import { INDEXED_FLAG_FILTER, INDEXED_TRIANGLE_WORDS, MAX_MATERIAL, MESH_HEADER_WORD, TRIANGLE_FLAG_FILTER } from './layout.ts';
import type { MeshBvh, MeshGeometry } from './mesh-bvh.ts';
import { buildMeshBvhFromTopology, buildMeshBvhWithTinyBvh } from './mesh-bvh-topology.ts';
import { createRandom, randomAffine, randomRays, randomTriangleSoup, tessellatedQuad } from './oracle-fixtures.ts';
import { assembleTraceScene, buildTraceScene, meshRefitPlacement, type TraceSceneData, type TraceSceneInstance } from './scene-assembly.ts';
import { writeTriangleFilterFlags, type TrianglePasses } from './triangle-filter.ts';
import type { TraceRay } from './rays.ts';
import { TRACE_MODULE_ENTRIES, traceModuleEntries, traceModuleWgsl } from './wgsl.ts';

describe('the indexed walk module', () => {
    it('declares a fourth binding, the positions, read as vec4f or packed f32, around the same walk', () => {
        const recordsModule = traceModuleWgsl();
        const indexedModule = traceModuleWgsl({ triangleLeaves: 'indexed', traceSlot: { group: 2, firstBinding: 4 } });
        const packed = traceModuleWgsl({ triangleLeaves: 'indexed', positionLayout: 'packed' });
        expect(recordsModule).not.toContain('bvhTracePositions');
        expect(indexedModule).toContain('@group(2) @binding(7) var<storage, read> bvhTracePositions: array<vec4f>;');
        expect(indexedModule).toContain('var<storage, read> bvhTraceTriangles: array<vec4u>;');
        expect(packed).toContain('bvhTracePositions: array<f32>;');
        expect(traceModuleEntries({ triangleLeaves: 'indexed' })).toHaveLength(4);
        expect(traceModuleEntries()).toBe(TRACE_MODULE_ENTRIES);
        const walk = (module: string) => module.slice(module.indexOf('fn bvhWalkClosest('), module.indexOf('fn bvhWalkAny('));
        expect(walk(indexedModule)).toBe(walk(recordsModule));
        expect(() => traceModuleWgsl({ triangleLeaves: 'packed' as never })).toThrow(RangeError);
    });
});

const tinyBvh = await createTinyBvhBuilder();
const records = (geometry: MeshGeometry): MeshBvh => buildMeshBvhWithTinyBvh(geometry, tinyBvh);
const indexed = (geometry: MeshGeometry): MeshBvh => buildMeshBvhWithTinyBvh(geometry, tinyBvh, { triangleLeaves: 'indexed' });

const withFlags = (geometry: MeshGeometry, every: number): MeshGeometry => {
    const count = geometry.indices ? geometry.indices.length / 3 : geometry.positions.length / 9;
    return { ...geometry, triangleFlags: Uint32Array.from({ length: count }, (_, t) => (t % every === 0 ? TRIANGLE_FLAG_FILTER : 0)) };
};

/** An unindexed soup with every corner duplicated as its own vertex, and some exact duplicates the weld merges. */
function duplicatedQuad(cells: number): MeshGeometry {
    const quad = tessellatedQuad(cells, 1.5);
    const positions = new Float32Array(quad.indices!.length * 3);
    quad.indices!.forEach((vertex, entry) => positions.set(quad.positions.subarray(vertex * 3, vertex * 3 + 3), entry * 3));
    return { positions };
}

/** Mirrored, non-uniform, masked, filtered, with materials, unindexed and duplicated corners, one mesh used twice. */
function mixedScene(seed: number): { meshes: MeshGeometry[]; instances: TraceSceneInstance[] } {
    const random = createRandom(seed);
    const meshes: MeshGeometry[] = [];
    const instances: TraceSceneInstance[] = [];
    const masks = [undefined, 0x0f, 0xf0];
    for (let i = 0; i < 12; i++) {
        const mesh = i % 3 === 0 ? tessellatedQuad(4 + (i % 4), 1.5) : i % 3 === 1 ? duplicatedQuad(3 + (i % 3)) : randomTriangleSoup(random, 60 + i * 5, 2, 0.6);
        meshes.push(i % 4 === 1 ? withFlags(mesh, 3) : mesh);
        instances.push({
            meshIndex: i,
            instanceIndex: 700 - i * 11,
            objectToWorld: randomAffine(random, { extent: 8, mirror: i % 3 === 1 }),
            static: i % 6 !== 5,
            mask: masks[i % 3],
            material: i % 4 === 3 ? undefined : i === 2 ? MAX_MATERIAL : i * 101,
        });
    }
    meshes.push(withFlags(randomTriangleSoup(random, 80, 2, 0.6), 2));
    for (let k = 0; k < 2; k++) {
        instances.push({ meshIndex: 12, instanceIndex: 10 + k, objectToWorld: randomAffine(random, { extent: 8, mirror: k === 1 }), static: true, material: k === 0 ? 7 : undefined });
    }
    return { meshes, instances };
}

/** Lets rays through some flagged triangles, by instance, triangle, side and query. */
const filter: TrianglePasses = ({ instance, triangle, frontFace, query }) => (instance * 7 + triangle) % 3 !== 1 && (query === 0 || !frontFace);

function rays(seed: number, count: number): TraceRay[] {
    const random = createRandom(seed);
    return [...randomRays(random, count / 2, [-10, -10, -10, 10, 10, 10], 2, { randomInterval: true }), ...randomRays(random, count / 2, [-8, -8, -8, 8, 8, 8], 0)];
}

/** Every closest and any hit, with masks and the filter: the full CpuHit, so t, barycentrics, material and normal must be bit-identical. */
function traceAll(scene: TraceSceneData, traced: readonly TraceRay[]): unknown[] {
    const tracer = createCpuTracer(scene, { triangleFilter: filter });
    const out: unknown[] = [];
    const masks = [undefined, 0x0f, 0xf0, 0x01];
    traced.forEach((ray, index) => {
        const mask = masks[index % masks.length];
        out.push(tracer.closest(ray, { mask }), tracer.any(ray, { mask }));
    });
    return out;
}

describe('indexed trees', () => {
    const quad = withFlags(tessellatedQuad(9, 2), 4);
    const soup = { ...randomTriangleSoup(createRandom(3), 500, 5, 1) };

    it('tinybvh writes the same tree with 16-byte triangles, and converts back to its records', () => {
        for (const geometry of [quad, soup, duplicatedQuad(4)]) {
            const asRecords = records(geometry);
            const asIndexed = indexed(geometry);
            expect(asIndexed.triangleLeaves).toBe('indexed');
            expect(asIndexed.triangles.byteLength).toBe(asIndexed.triangleCount * GPU_LAYOUT_INDEXED_TRIANGLE_BYTES);
            expect(new Uint8Array(asIndexed.nodes)).toEqual(new Uint8Array(asRecords.nodes));
            expect(new Uint8Array(asIndexed.triangles)).toEqual(new Uint8Array(indexedFromRecords(asRecords, geometry).triangles));
            expect(new Uint8Array(recordsFromIndexed(asIndexed, geometry.positions).triangles)).toEqual(new Uint8Array(asRecords.triangles));
            expect(checkGpuLayout(asIndexed, geometry)).toEqual([]);
        }
    });

    it('the CPU builders write indexed trees that pass the layout check', () => {
        for (const layout of [
            buildLbvhOnCpu(quad, { triangleLeaves: 'indexed' }),
            buildHplocOnCpu(quad, { triangleLeaves: 'indexed' }),
            buildLbvhOnCpu({ positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]) }, { triangleLeaves: 'indexed' }),
        ]) {
            expect(layout.triangleLeaves).toBe('indexed');
            expect(checkGpuLayout(layout, layout.triangleCount === 1 ? { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]) } : quad)).toEqual([]);
        }
        const fromTopology = buildMeshBvhFromTopology(quad, tinyBvh.build(quad).topology, { triangleLeaves: 'indexed' });
        expect(checkGpuLayout(fromTopology, quad)).toEqual([]);
    });

    it('the layout check reports a wrong vertex index and a wrong flag', () => {
        const tree = indexed(quad);
        const words = new Uint32Array(tree.triangles.slice(0));
        words[5] = (words[5] + 1) % (quad.positions.length / 3);
        expect(checkGpuLayout({ ...tree, triangles: words.buffer }, quad).join()).toMatch(/corner 1 names vertex/);
        const flagged = new Uint32Array(tree.triangles.slice(0));
        flagged[3] ^= INDEXED_FLAG_FILTER;
        expect(checkGpuLayout({ ...tree, triangles: flagged.buffer }, quad).join()).toMatch(/flags/);
    });

    it('serializes with the format in header word 7, and refuses a format it does not know', () => {
        const tree = indexed(quad);
        const bytes = serializeLayout(tree);
        expect(new Uint32Array(bytes, 0, 16)[7]).toBe(1);
        const back = deserializeLayout(bytes);
        expect(back.triangleLeaves).toBe('indexed');
        expect(new Uint8Array(back.triangles)).toEqual(new Uint8Array(tree.triangles));
        expect(new Uint32Array(serializeLayout(records(quad)), 0, 16)[7]).toBe(0);
        expect(deserializeLayout(serializeLayout(records(quad))).triangleLeaves).toBeUndefined();
        const bad = bytes.slice(0);
        new Uint32Array(bad, 0, 16)[7] = 2;
        expect(() => deserializeLayout(bad)).toThrow(/triangle format 2/);
    });

    it('CpuMeshBvh takes an indexed tree with its source and finds what the records tree finds', () => {
        const fromIndexed = new CpuMeshBvh(indexed(soup), { source: soup });
        const fromRecords = new CpuMeshBvh(records(soup), { source: soup });
        expect(() => new CpuMeshBvh(indexed(soup))).toThrow(/options.source/);
        for (const ray of rays(5, 300)) {
            const r = { origin: ray.origin, direction: ray.direction };
            expect(fromIndexed.raycastFirst(r)).toEqual(fromRecords.raycastFirst(r));
        }
    });
});

describe('the weld', () => {
    it('merges vertices with identical bits only, keeping first-use order', () => {
        const positions = new Float32Array([0, 0, 0, 1, 2, 3, 0, 0, 0, -0, 0, 0, 1, 2, 3, 1, 2, 3.0000002]);
        const weld = weldExactBits(positions);
        expect(weld.vertexCount).toBe(4);
        expect(Array.from(weld.remap)).toEqual([0, 1, 0, 2, 1, 3]);
        expect(Array.from(weld.positions.subarray(6, 9))).toEqual([-0, 0, 0]);
        expect(Object.is(weld.positions[6], -0)).toBe(true);
    });

    it('keeps every triangle corner bit for bit', () => {
        const geometry = duplicatedQuad(5);
        const welded = weldPositions(geometry);
        expect(welded.positions.length).toBe(tessellatedQuad(5, 1.5).positions.length);
        const bits = (positions: Float32Array, vertex: number) => Array.from(new Uint32Array(positions.buffer, positions.byteOffset + vertex * 12, 3));
        for (let entry = 0; entry < welded.indices!.length; entry++) expect(bits(welded.positions, welded.indices![entry])).toEqual(bits(geometry.positions, entry));
    });

    it('a tolerance weld merges near copies and moves them onto the first', () => {
        const positions = new Float32Array([1, 1, 1, Math.fround(1 + 2 ** -23), 1, 1, 1.5, 1, 1]);
        expect(weldExactBits(positions).vertexCount).toBe(3);
        const weld = weldWithinTolerance(positions, 1e-5);
        expect(weld.vertexCount).toBe(2);
        expect(Array.from(weld.remap)).toEqual([0, 0, 1]);
        expect(() => weldWithinTolerance(positions, 0)).toThrow(RangeError);
    });
});

describe('an indexed scene against the records scene', () => {
    const { meshes, instances } = mixedScene(11);
    const traced = rays(12, 1500);
    const recordsScene = assembleTraceScene(meshes.map(records), instances);
    const expected = traceAll(recordsScene, traced);

    it('gives identical hits from records, from indexed trees and from converted records, in both position layouts, welded or not', () => {
        const scenes = [
            assembleTraceScene(meshes.map(indexed), instances, {}, { triangleLeaves: 'indexed', geometries: meshes }),
            assembleTraceScene(meshes.map(records), instances, {}, { triangleLeaves: 'indexed', geometries: meshes, positionLayout: 'packed' }),
            assembleTraceScene(meshes.map(indexed), instances, {}, { triangleLeaves: 'indexed', geometries: meshes, weld: false }),
        ];
        for (const scene of scenes) {
            expect(scene.triangleLeaves).toBe('indexed');
            expect(traceAll(scene, traced)).toEqual(expected);
        }
        // Some hits, some misses, some filtered: the comparison means something.
        const hits = expected.filter((hit, index) => index % 2 === 0 && hit !== null).length;
        expect(hits).toBeGreaterThan(100);
        expect(hits).toBeLessThan(traced.length);
    });

    it('is smaller, welds duplicated corners and puts each mesh header before its triangles', () => {
        const scene = assembleTraceScene(meshes.map(indexed), instances, {}, { triangleLeaves: 'indexed', geometries: meshes });
        const unwelded = assembleTraceScene(meshes.map(indexed), instances, {}, { triangleLeaves: 'indexed', geometries: meshes, weld: false });
        const triangles = meshes.reduce((sum, mesh) => sum + (mesh.indices ? mesh.indices.length : mesh.positions.length / 3) / 3, 0);
        expect(scene.triangles.byteLength).toBe((triangles + meshes.length) * 16);
        expect(scene.vertexCount).toBeLessThan(unwelded.vertexCount);
        expect(scene.positions.byteLength).toBe(scene.vertexCount * 16);
        expect(scene.byteLength).toBeLessThan(recordsScene.byteLength);
        const words = new Uint32Array(scene.triangles);
        expect(words[(scene.meshes[1].triangleOffset - 1) * INDEXED_TRIANGLE_WORDS + MESH_HEADER_WORD.vertexBase]).toBeGreaterThan(0);
        expect(meshRefitPlacement(scene, 1).triangleWordOffset).toBe(scene.meshes[1].triangleOffset * 4);
    });

    it('a merged scene gives the same hits in both formats (side table: source instance and material)', async () => {
        const recordsMerged = await buildTraceScene(meshes, instances, { build: records, mergeStaticInstances: true });
        const indexedMerged = await buildTraceScene(meshes, instances, { build: indexed, mergeStaticInstances: true, triangleLeaves: 'indexed' });
        const packedMerged = await buildTraceScene(meshes, instances, { build: records, mergeStaticInstances: true, triangleLeaves: 'indexed', positionLayout: 'packed' });
        expect(indexedMerged.mergedGroups.length).toBe(3);
        const mergedExpected = traceAll(recordsMerged, traced);
        expect(traceAll(indexedMerged, traced)).toEqual(mergedExpected);
        expect(traceAll(packedMerged, traced)).toEqual(mergedExpected);
    });

    it('writeTriangleFilterFlags flags the same triangles on an indexed merged range, by source instance', async () => {
        const recordsMerged = await buildTraceScene(meshes, instances, { build: records, mergeStaticInstances: true });
        const indexedMerged = await buildTraceScene(meshes, instances, { build: indexed, mergeStaticInstances: true, triangleLeaves: 'indexed' });
        const decide = (original: number, source: number | undefined) => (original + (source ?? 0)) % 5 === 0;
        recordsMerged.mergedGroups.forEach((group, g) => {
            const a = writeTriangleFilterFlags(recordsMerged.triangles, decide, group);
            const b = writeTriangleFilterFlags(indexedMerged.triangles, decide, indexedMerged.mergedGroups[g], 'indexed');
            expect(b).toBe(a);
            expect(a).toBeGreaterThan(0);
        });
        const a = writeTriangleFilterFlags(recordsMerged.triangles, decide, recordsMerged.meshes[5]);
        expect(writeTriangleFilterFlags(indexedMerged.triangles, decide, indexedMerged.meshes[5], 'indexed')).toBe(a);
        expect(traceAll(indexedMerged, traced)).toEqual(traceAll(recordsMerged, traced));
    });

    it('needs the geometries, and refuses indexed trees in a records scene', () => {
        expect(() => assembleTraceScene(meshes.map(indexed), instances, {}, { triangleLeaves: 'indexed' })).toThrow(/geometries\[0\]/);
        expect(() => assembleTraceScene(meshes.map(indexed), instances)).toThrow(/is indexed/);
        expect(() => assembleTraceScene(meshes.map(records), instances, {}, { geometries: meshes })).toThrow(/triangleLeaves/);
    });
});

describe('seams', () => {
    // Two triangles sharing an edge, with the shared corners as separate vertices: bit-equal, or one ulp apart.
    const seam = (ulp: boolean): MeshGeometry => ({
        // Triangle 0: (0,0,0) (1,0,0) (1,1,0); triangle 1: (0,0,0) (1,1,0) (0,1,0), its copy of (1,1,0) one ulp off in x when asked.
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, ulp ? 1 + 2 ** -23 : 1, 1, 0, 0, 1, 0]),
    });
    const instance: TraceSceneInstance = { meshIndex: 0, instanceIndex: 0, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] };
    const diagonal = (count: number): TraceRay[] =>
        Array.from({ length: count }, (_, k) => {
            const s = (k + 0.5) / count;
            return { origin: [s, s, 1], direction: [0, 0, -1], tMin: 0, tMax: 10 };
        });

    it('a split seam with identical bits lets nothing through, in both formats', async () => {
        const mesh = seam(false);
        for (const options of [{}, { triangleLeaves: 'indexed' as const }, { triangleLeaves: 'indexed' as const, weld: false }]) {
            const scene = await buildTraceScene([mesh], [instance], { build: records, ...options });
            const tracer = createCpuTracer(scene);
            expect(diagonal(1000).filter((ray) => tracer.closest(ray) === null).length).toBe(0);
        }
    });

    it('a one-ulp seam behaves the same in both formats, and the tolerance weld closes it', async () => {
        const mesh = seam(true);
        const leaks = async (options: object): Promise<CpuHit[]> => {
            const scene = await buildTraceScene([mesh], [instance], { build: records, ...options });
            const tracer = createCpuTracer(scene);
            return diagonal(2000).map((ray) => tracer.closest(ray) as CpuHit);
        };
        const asRecords = await leaks({});
        expect(await leaks({ triangleLeaves: 'indexed' })).toEqual(asRecords);
        expect((await leaks({ weldTolerance: 1e-6 })).every((hit) => hit !== null)).toBe(true);
        expect((await leaks({ triangleLeaves: 'indexed', weldTolerance: 1e-6 })).every((hit) => hit !== null)).toBe(true);
    });
});

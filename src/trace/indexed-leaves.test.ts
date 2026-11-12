// Indexed leaves (docs/internals/design/indexed-leaves.md) against records: the same tree in either format,
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
import { PRIMITIVE_BOX_FLOATS, refitOnCpu } from '../refit-cpu.ts';
import { TRACE_MODULE_ENTRIES, traceModuleEntries, traceModuleWgsl } from './wgsl.ts';
import { hplocBatchBytes } from '../gpu-hploc.ts';
import { geometryTotals, lbvhBatchBytes } from '../gpu-lbvh.ts';

describe('GPU batch sizes', () => {
    it('count 16 bytes per triangle of output with indexed leaves, so more triangles fit a capped batch', () => {
        const totals = geometryTotals([tessellatedQuad(40, 1), tessellatedQuad(10, 1)]);
        const triangles = totals.triangles;
        for (const bytes of [(leaves: 'records' | 'indexed') => hplocBatchBytes(totals, 'cubic-30', leaves), (leaves: 'records' | 'indexed') => lbvhBatchBytes(totals, leaves)]) {
            expect(bytes('records').buffers.outTriangles).toBe(triangles * 48);
            expect(bytes('indexed').buffers.outTriangles).toBe(triangles * 16);
            expect(bytes('records').total - bytes('indexed').total).toBe(triangles * 32);
        }
    });
});

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
        expect(meshRefitPlacement(unwelded, 1).triangleWordOffset).toBe(unwelded.meshes[1].triangleOffset * 4);
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

    it('writeTriangleFilterFlags takes the format from the scene or tree, and refuses an indexed scene buffer bare or without a range', () => {
        const scene = assembleTraceScene(meshes.map(indexed), instances, {}, { triangleLeaves: 'indexed', geometries: meshes });
        const before = new Uint32Array(scene.triangles.slice(0));
        const decide = (original: number) => original % 3 === 0;
        // Bare, without the format: 16-byte units read as 48-byte records would overwrite vertex indices.
        expect(() => writeTriangleFilterFlags(scene.triangles, decide, scene.meshes[1])).toThrow(/indexed scene's buffer.*triangleLeaves: 'indexed'/);
        expect(() => writeTriangleFilterFlags(new Uint32Array(scene.triangles), decide, scene.meshes[1])).toThrow(/indexed scene's buffer/);
        expect(() => writeTriangleFilterFlags(scene.triangles, decide, scene.meshes[1], 'records')).toThrow(/indexed, not records/);
        expect(() => writeTriangleFilterFlags(scene, decide, scene.meshes[1], 'records')).toThrow(/indexed, not records/);
        // Without a range a scene buffer would toggle bit 28 of headers and side tables.
        expect(() => writeTriangleFilterFlags(scene, decide)).toThrow(/needs a range/);
        expect(() => writeTriangleFilterFlags(scene.triangles, decide, undefined, 'indexed')).toThrow(/needs a range/);
        expect(new Uint32Array(scene.triangles)).toEqual(before);

        const flagged = writeTriangleFilterFlags(scene, decide, scene.meshes[1]);
        expect(flagged).toBeGreaterThan(0);
        const after = new Uint32Array(scene.triangles);
        for (let word = 0; word < after.length; word++) {
            const unit = Math.floor(word / INDEXED_TRIANGLE_WORDS);
            const inMesh = unit >= scene.meshes[1].triangleOffset && unit < scene.meshes[1].triangleOffset + scene.meshes[1].triangleCount;
            if (inMesh && word % INDEXED_TRIANGLE_WORDS === 3) expect(after[word] & ~INDEXED_FLAG_FILTER).toBe(before[word] & ~INDEXED_FLAG_FILTER);
            else expect(after[word]).toBe(before[word]);
        }
        // A lone tree has no headers: no range is fine, and the tree gives its format.
        const tree = indexed(meshes[0]);
        expect(writeTriangleFilterFlags(tree, decide)).toBe(writeTriangleFilterFlags(tree.triangles, decide, undefined, 'indexed'));
    });

    it('needs the geometries, and refuses indexed trees in a records scene', () => {
        expect(() => assembleTraceScene(meshes.map(indexed), instances, {}, { triangleLeaves: 'indexed' })).toThrow(/geometries\[0\]/);
        expect(() => assembleTraceScene(meshes.map(indexed), instances)).toThrow(/is indexed/);
        expect(() => assembleTraceScene(meshes.map(records), instances, {}, { geometries: meshes })).toThrow(/triangleLeaves/);
    });
});

/**
 * What `GpuMeshRefit` does to an indexed scene mesh, on the CPU: copy the pose
 * to the scene's positions at the placement's vertex base (meshRefitCopyVertices),
 * box each slot from there (meshRefitIndexedBoxes), refit the mesh's nodes.
 */
function refitIndexedSceneMesh(scene: TraceSceneData, meshIndex: number, tree: MeshBvh, pose: Float32Array): void {
    const placement = meshRefitPlacement(scene, meshIndex);
    const base = placement.vertexBase ?? 0;
    const stride = scene.positionLayout === 'packed' ? 3 : 4;
    const positions = new Float32Array(scene.positions);
    for (let vertex = 0; vertex < pose.length / 3; vertex++) positions.set(pose.subarray(vertex * 3, vertex * 3 + 3), (base + vertex) * stride);
    const words = new Uint32Array(scene.triangles);
    const boxes = new Float32Array(tree.triangleCount * PRIMITIVE_BOX_FLOATS);
    for (let slot = 0; slot < tree.triangleCount; slot++) {
        const lower = [Infinity, Infinity, Infinity];
        const upper = [-Infinity, -Infinity, -Infinity];
        for (let corner = 0; corner < 3; corner++) {
            const at = (base + words[placement.triangleWordOffset + slot * INDEXED_TRIANGLE_WORDS + corner]) * stride;
            for (let axis = 0; axis < 3; axis++) {
                lower[axis] = Math.min(lower[axis], positions[at + axis]);
                upper[axis] = Math.max(upper[axis], positions[at + axis]);
            }
        }
        boxes.set(lower, slot * PRIMITIVE_BOX_FLOATS);
        boxes.set(upper, slot * PRIMITIVE_BOX_FLOATS + 4);
    }
    refitOnCpu(new Float32Array(scene.nodes), new Uint32Array(scene.nodes), placement.nodeWordOffset, tree.nodeCount, tree.parents, boxes);
}

describe('refitting a mesh of an indexed static scene', () => {
    const { meshes, instances } = mixedScene(21);
    const traced = rays(22, 1500);
    const closest = (scene: TraceSceneData) => {
        const tracer = createCpuTracer(scene, { triangleFilter: filter });
        return traced.map((ray) => tracer.closest(ray));
    };

    it('refuses a welded mesh, exact or tolerance, and says to assemble with weld: false', async () => {
        const welded = assembleTraceScene(meshes.map(indexed), instances, {}, { triangleLeaves: 'indexed', geometries: meshes });
        expect(() => meshRefitPlacement(welded, 4)).toThrow(/welded.*weld: false/);
        const tolerance = await buildTraceScene(meshes, instances, { build: indexed, triangleLeaves: 'indexed', weld: false, weldTolerance: 1e-6 });
        expect(() => meshRefitPlacement(tolerance, 4)).toThrow(/welded.*weld: false/);
        const unwelded = assembleTraceScene(meshes.map(indexed), instances, {}, { triangleLeaves: 'indexed', geometries: meshes, weld: false, positionLayout: 'packed' });
        expect(meshRefitPlacement(unwelded, 4)).toEqual({
            nodeWordOffset: unwelded.meshes[4].nodeOffset * 16,
            triangleWordOffset: unwelded.meshes[4].triangleOffset * INDEXED_TRIANGLE_WORDS,
            vertexBase: unwelded.meshes[4].vertexBase,
            vertexCount: meshes[4].positions.length / 3,
            positionLayout: 'packed',
        });
        expect(unwelded.meshes[4].vertexBase).toBeGreaterThan(0);
    });

    it('writes the pose at the mesh\'s vertex base: every mesh hits what a scene built in that pose hits', async () => {
        for (const positionLayout of ['vec4', 'packed'] as const) {
            // A non-first mesh, shrunk about its centre so the pose stays inside the box the top level has for it.
            const meshIndex = 4;
            const source = meshes[meshIndex].positions;
            const centre = [0, 1, 2].map((axis) => source.filter((_, k) => k % 3 === axis).reduce((a, b) => a + b, 0) / (source.length / 3));
            const pose = source.map((value, k) => centre[k % 3] + (value - centre[k % 3]) * 0.6);
            const posed = meshes.map((mesh, index) => (index === meshIndex ? { ...mesh, positions: pose } : mesh));

            const trees = meshes.map(indexed);
            const scene = await buildTraceScene(meshes, instances, { build: indexed, triangleLeaves: 'indexed', weld: false, positionLayout });
            refitIndexedSceneMesh(scene, meshIndex, trees[meshIndex], pose);
            const reference = assembleTraceScene(posed.map(records), instances);
            const expected = closest(reference);
            expect(closest(scene)).toEqual(expected);
            // The pose changed what rays hit, and other meshes are hit too.
            expect(expected).not.toEqual(closest(assembleTraceScene(meshes.map(records), instances)));
            expect(new Set(expected.flatMap((hit) => (hit ? [hit.instanceIndex] : []))).size).toBeGreaterThan(5);
        }
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

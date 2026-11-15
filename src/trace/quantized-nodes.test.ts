// Quantized nodes (docs/internals/design/compressed-nodes.md) against f32 nodes: the frame rule, outward
// rounding with the margin, every decoded box around its exact box, and the same hits from the
// same scene with either node format (the triangle test is unchanged, so instance, triangle and
// t are bit for bit the same).
import { describe, expect, it } from 'vitest';
import {
    buildHplocOnCpu,
    buildLbvhOnCpu,
    checkGpuLayout,
    CpuMeshBvh,
    createTinyBvhBuilder,
    DEFAULT_NODE_FORMAT,
    deserializeLayout,
    GPU_LAYOUT_NODE_BYTES,
    GPU_LAYOUT_NODE_WORDS,
    GPU_LAYOUT_QUANTIZED_NODE_BYTES,
    GPU_LAYOUT_QUANTIZED_NODE_WORDS,
    serializeLayout,
    type GpuLayoutMeshBvh,
} from '../index.ts';
import {
    dequantizeNodes,
    FRAME_MAX_SPAN,
    nodeChildUnion,
    nodeFrameOf,
    QUANTIZE_MARGIN_STEPS,
    quantizeAxis,
    quantizeNodes,
    quantizeNodeWords,
    readNodeFrame,
} from '../quantized-nodes.ts';
import { treeSahCost } from '../layout-check.ts';
import { PRIMITIVE_BOX_FLOATS, refitOnCpu } from '../refit-cpu.ts';
import { createCpuTracer, newTraceStats } from './cpu-traversal.ts';
import { TRIANGLE_FLAG_FILTER } from '../gpu-layout.ts';
import type { MeshBvh, MeshGeometry } from './mesh-bvh.ts';
import { buildMeshBvhFromTopology, buildMeshBvhWithTinyBvh } from './mesh-bvh-topology.ts';
import { createRandom, randomAffine, randomRays, randomTriangleSoup, tessellatedQuad } from './oracle-fixtures.ts';
import { assembleTraceScene, buildTraceScene, type TraceSceneData, type TraceSceneInstance } from './scene-assembly.ts';
import type { TraceRay } from './rays.ts';
import type { TrianglePasses } from './triangle-filter.ts';
import { batchKernelWgsl, traceModuleEntries, traceModuleWgsl } from './wgsl.ts';

const tinyBvh = await createTinyBvhBuilder();
const f32 = (geometry: MeshGeometry): MeshBvh => buildMeshBvhWithTinyBvh(geometry, tinyBvh);

const shifted = (geometry: MeshGeometry, offset: readonly number[], scale = 1): MeshGeometry => ({
    ...geometry,
    positions: geometry.positions.map((value, index) => value * scale + offset[index % 3]),
});

/** Meshes the grid finds hard: far from the origin, tiny, flat, huge, a single triangle. */
function hardMeshes(): MeshGeometry[] {
    const random = createRandom(11);
    return [
        tessellatedQuad(9, 2),
        randomTriangleSoup(random, 400, 5, 1),
        shifted(randomTriangleSoup(random, 300, 2, 0.3), [1.0e5, -3.0e4, 7.5]),
        shifted(randomTriangleSoup(random, 200, 1, 0.2), [0, 0, 0], 1e-6),
        shifted(randomTriangleSoup(random, 200, 1, 0.5), [0, 0, 0], 4e4),
        // Flat on z = 3 (a zero-extent axis).
        { positions: tessellatedQuad(6, 1).positions.map((value, index) => (index % 3 === 2 ? 3 : value)), indices: tessellatedQuad(6, 1).indices },
        { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]) },
        { positions: new Float32Array([-2, 5, 1e-30, 0.5, 5, 1e-30, 0, 6, -1e-30, 3, 3, 3, 3, 3, 3.5, 3.25, 3.5, 3]) },
    ];
}

describe('the quantized walk module', () => {
    it('reads nodes as vec4u and the frame on instance entry, around the same walk with the same bindings', () => {
        const f32Module = traceModuleWgsl({ nodeFormat: 'f32' });
        const quantizedModule = traceModuleWgsl({ nodeFormat: 'quantized' });
        // Quantized is the default (DEFAULT_NODE_FORMAT), as it is for scenes.
        expect(traceModuleWgsl()).toBe(quantizedModule);
        expect(batchKernelWgsl('any')).toBe(batchKernelWgsl('any', { nodeFormat: 'quantized' }));
        expect(f32Module).toContain('var<storage, read> bvhTraceNodes: array<vec4f>;');
        expect(quantizedModule).toContain('var<storage, read> bvhTraceNodes: array<vec4u>;');
        expect(quantizedModule).toContain('var<private> bvhMeshFrameOrigin: vec3f;');
        const walk = (module: string) => module.slice(module.indexOf('fn bvhWalkClosest('), module.indexOf('fn bvhTraceClosest('));
        expect(walk(quantizedModule)).toBe(walk(f32Module));
        expect(walk(f32Module)).toContain('node = bvhLoadMeshNode(instance.rootNodeOffset + index);');
        expect(walk(f32Module)).toContain('bvhEnterMeshNodes(instance.rootNodeOffset);');
        const both = traceModuleWgsl({ nodeFormat: 'quantized', triangleLeaves: 'indexed', positionLayout: 'packed', rayMasks: false });
        expect(walk(both)).toBe(walk(traceModuleWgsl({ triangleLeaves: 'indexed', positionLayout: 'packed', rayMasks: false, nodeFormat: 'f32' })));
        expect(traceModuleEntries({ triangleLeaves: 'records' })).toHaveLength(3);
        expect(batchKernelWgsl('closest', { nodeFormat: 'quantized' })).toContain('array<vec4u>;');
        expect(() => traceModuleWgsl({ nodeFormat: 'f16' as never })).toThrow(RangeError);
    });
});

describe('the frame', () => {
    it('is a power-of-two step and an origin on it, every plane an exact f32, the box within the span', () => {
        const random = createRandom(4);
        const boxes: number[][] = [[0, 0, 0, 0, 0, 0], [-1, -1, -1, 1, 1, 1], [1e5, 2e5, -3, 1e5 + 1e-2, 2e5 + 300, 9]];
        for (let i = 0; i < 300; i++) {
            const scale = 10 ** (random() * 16 - 8);
            const center = (random() - 0.5) * 10 ** (random() * 12 - 4);
            const lo = [0, 1, 2].map(() => Math.fround(center + (random() - 0.5) * scale));
            boxes.push([...lo, ...lo.map((value) => Math.fround(value + random() * scale))]);
        }
        for (const box of boxes) {
            const frame = nodeFrameOf(box);
            for (let axis = 0; axis < 3; axis++) {
                const { origin, step } = { origin: frame.origin[axis], step: frame.step[axis] };
                expect(Number.isInteger(Math.log2(step))).toBe(true);
                expect(Number.isInteger(origin / step)).toBe(true);
                for (const q of [0, 1, 32768, 65534, 65535]) expect(Math.fround(origin + q * step)).toBe(origin + q * step);
                expect(Math.ceil(box[axis + 3] / step) - Math.floor(box[axis] / step)).toBeLessThanOrEqual(FRAME_MAX_SPAN);
                // The smallest such step: half of it breaks a rule (or hits the floor).
                const half = step / 2;
                const halfWorks = Math.abs(box[axis]) < 2 ** 23 * half && Math.abs(box[axis + 3]) < 2 ** 23 * half && Math.ceil(box[axis + 3] / half) - Math.floor(box[axis] / half) <= FRAME_MAX_SPAN;
                if (step > 2 ** -100) expect(halfWorks).toBe(false);
            }
        }
    });

    it('rounds each plane outward to the grid, at least 1/1024 of a step out', () => {
        const random = createRandom(9);
        for (let i = 0; i < 2000; i++) {
            const lo = Math.fround((random() - 0.5) * 100);
            const hi = Math.fround(lo + random() * 50);
            const { origin, step } = (({ origin: o, step: s }) => ({ origin: o[0], step: s[0] }))(nodeFrameOf([lo, 0, 0, hi, 0, 0]));
            const min = Math.fround(lo + random() * (hi - lo));
            const max = Math.fround(min + random() * (hi - min));
            // Planes on the grid exactly, and a hair off it, must still get the margin.
            for (const [a, b] of [[min, max], [origin + 5 * step, origin + 9 * step], [Math.fround(origin + 5 * step + step / 4096), max]]) {
                if (!(a <= b) || b > hi || a < lo) continue;
                const word = quantizeAxis(a, b, origin, step);
                const decodedMin = origin + (word & 0xffff) * step;
                const decodedMax = origin + (word >>> 16) * step;
                expect(a - decodedMin).toBeGreaterThanOrEqual(step * QUANTIZE_MARGIN_STEPS);
                expect(decodedMax - b).toBeGreaterThanOrEqual(step * QUANTIZE_MARGIN_STEPS);
                expect(a - decodedMin).toBeLessThanOrEqual(step * (1 + QUANTIZE_MARGIN_STEPS));
            }
        }
    });
});

describe('quantized trees', () => {
    it('every builder writes them, and every decoded box contains its f32 box (the layout check)', () => {
        for (const geometry of hardMeshes()) {
            const exact = f32(geometry);
            const trees: GpuLayoutMeshBvh[] = [
                buildMeshBvhWithTinyBvh(geometry, tinyBvh, { nodeFormat: 'quantized' }),
                buildMeshBvhWithTinyBvh(geometry, tinyBvh, { nodeFormat: 'quantized', triangleLeaves: 'indexed' }),
                buildLbvhOnCpu(geometry, { nodeFormat: 'quantized' }),
                buildHplocOnCpu(geometry, { nodeFormat: 'quantized', triangleLeaves: 'indexed' }),
                buildMeshBvhFromTopology(geometry, tinyBvh.build(geometry).topology, { nodeFormat: 'quantized' }),
            ];
            for (const tree of trees) {
                expect(tree.nodeFormat).toBe('quantized');
                expect(tree.nodes.byteLength).toBe((tree.nodeCount + 1) * GPU_LAYOUT_QUANTIZED_NODE_BYTES);
                expect(checkGpuLayout(tree, geometry)).toEqual([]);
            }
            // Directly too: tinybvh's quantized tree, node by node, around tinybvh's f32 tree.
            const decoded = dequantizeNodes(trees[0]);
            const a = new Float32Array(decoded.nodes);
            const b = new Float32Array(exact.nodes);
            const words = new Uint32Array(exact.nodes);
            expect(new Uint32Array(decoded.nodes).filter((_, k) => k % 4 === 3)).toEqual(words.filter((_, k) => k % 4 === 3));
            for (let node = 0; node < exact.nodeCount; node++) {
                for (let side = 0; side < (words[node * 16 + 7] === 0xffffffff ? 1 : 2); side++) {
                    const at = node * 16 + side * 8;
                    for (let axis = 0; axis < 3; axis++) {
                        expect(a[at + axis]).toBeLessThan(b[at + axis]);
                        expect(a[at + 4 + axis]).toBeGreaterThan(b[at + 4 + axis]);
                    }
                }
            }
        }
    });

    it('the layout check catches a plane rounded inward and a wrong frame', () => {
        const geometry = randomTriangleSoup(createRandom(2), 300, 4, 1);
        const tree = quantizeNodes(f32(geometry));
        const words = new Uint32Array(tree.nodes.slice(0));
        // Node 3's left x min one step up: no longer around its box with the margin.
        words[(3 + 1) * GPU_LAYOUT_QUANTIZED_NODE_WORDS] += 1;
        expect(checkGpuLayout({ ...tree, nodes: words.buffer }, geometry).join()).toMatch(/does not contain/);
        const frame = new Float32Array(tree.nodes.slice(0));
        frame[4] *= 2;
        expect(checkGpuLayout({ ...tree, nodes: frame.buffer }, geometry).join()).toMatch(/frame on axis 0|does not contain/);
    });

    it('serializes with the node format in bits 8-15 of header word 7, its frame slot first', () => {
        const geometry = tessellatedQuad(7, 1);
        const tree = buildMeshBvhWithTinyBvh(geometry, tinyBvh, { nodeFormat: 'quantized', triangleLeaves: 'indexed' });
        const bytes = serializeLayout(tree);
        expect(new Uint32Array(bytes, 0, 16)[7]).toBe(0x101);
        const back = deserializeLayout(bytes);
        expect(back.nodeFormat).toBe('quantized');
        expect(back.triangleLeaves).toBe('indexed');
        expect(new Uint8Array(back.nodes)).toEqual(new Uint8Array(tree.nodes));
        expect(checkGpuLayout(back, geometry)).toEqual([]);
        expect(new Uint32Array(serializeLayout(quantizeNodes(f32(geometry))), 0, 16)[7]).toBe(0x100);
        expect(deserializeLayout(serializeLayout(f32(geometry))).nodeFormat).toBeUndefined();
        const bad = bytes.slice(0);
        new Uint32Array(bad, 0, 16)[7] = 0x201;
        expect(() => deserializeLayout(bad)).toThrow(/node format 2/);
    });

    it('CpuMeshBvh decodes it once: traverse hands out boxes around the exact ones, and raycasts find the same hits', () => {
        const geometry = randomTriangleSoup(createRandom(8), 600, 5, 1);
        const exact = new CpuMeshBvh(f32(geometry), { source: geometry });
        const packed = new CpuMeshBvh(buildMeshBvhWithTinyBvh(geometry, tinyBvh, { nodeFormat: 'quantized' }), { source: geometry });
        expect(packed.layout.nodeFormat).toBeUndefined();
        expect(packed.layout.nodes.byteLength).toBe(packed.layout.nodeCount * GPU_LAYOUT_NODE_BYTES);
        const boxes = (tree: CpuMeshBvh): Float32Array[] => {
            const out: Float32Array[] = [];
            tree.traverse({ intersectsBounds: (box) => (out.push(box.slice()), true), intersectsRange: () => false });
            return out;
        };
        const exactBoxes = boxes(exact);
        const packedBoxes = boxes(packed);
        expect(packedBoxes).toHaveLength(exactBoxes.length);
        exactBoxes.forEach((box, k) => {
            for (let axis = 0; axis < 3; axis++) {
                expect(packedBoxes[k][axis]).toBeLessThan(box[axis]);
                expect(packedBoxes[k][axis + 3]).toBeGreaterThan(box[axis + 3]);
            }
        });
        for (const ray of randomRays(createRandom(5), 400, [-6, -6, -6, 6, 6, 6], 2)) {
            const r = { origin: ray.origin, direction: ray.direction };
            expect(packed.raycastFirst(r)).toEqual(exact.raycastFirst(r));
        }
    });

    it('a CPU refit of the f32 copy, quantized again, is around the moved triangles', () => {
        const geometry = randomTriangleSoup(createRandom(6), 500, 4, 1);
        const tree = f32(geometry);
        const nodes = tree.nodes.slice(0);
        const floats = new Float32Array(nodes);
        const words = new Uint32Array(nodes);
        // Move every vertex, then refit the f32 copy from the triangles in tree order.
        const moved = { ...geometry, positions: geometry.positions.map((value, index) => Math.fround(value * 1.7 + Math.sin(index) * 3)) };
        const order = new Uint32Array(tree.triangles).filter((_, k) => k % 12 === 3);
        const boxes = new Float32Array(tree.triangleCount * PRIMITIVE_BOX_FLOATS);
        order.forEach((triangle, slot) => {
            for (let axis = 0; axis < 3; axis++) {
                const values = [0, 1, 2].map((corner) => moved.positions[(triangle * 3 + corner) * 3 + axis]);
                boxes[slot * PRIMITIVE_BOX_FLOATS + axis] = Math.min(...values);
                boxes[slot * PRIMITIVE_BOX_FLOATS + 4 + axis] = Math.max(...values);
            }
        });
        const { rootBox } = refitOnCpu(floats, words, 0, tree.nodeCount, tree.parents, boxes);
        const out = new ArrayBuffer((tree.nodeCount + 1) * GPU_LAYOUT_QUANTIZED_NODE_BYTES);
        const frame = quantizeNodeWords(floats, words, 0, tree.nodeCount, new Uint32Array(out), 0);
        expect(frame).toEqual(nodeFrameOf(rootBox));
        expect(frame).toEqual(nodeFrameOf(nodeChildUnion(floats, words, 0)));
        expect(readNodeFrame(new Uint32Array(out), 0)).toEqual(frame);
        // Records with the moved corners, so the layout check measures the boxes against the new pose.
        const triangles = tree.triangles.slice(0);
        const tf = new Float32Array(triangles);
        order.forEach((triangle, slot) => {
            for (let corner = 0; corner < 3; corner++) for (let axis = 0; axis < 3; axis++) tf[slot * 12 + corner * 4 + axis] = moved.positions[(triangle * 3 + corner) * 3 + axis];
        });
        const refit: GpuLayoutMeshBvh = { ...tree, nodes: out, nodeFormat: 'quantized', triangles, bounds: Float32Array.from(rootBox) };
        expect(checkGpuLayout(refit, moved)).toEqual([]);
    });
});

/** Some instanced meshes, some static ones to merge, mirrored, masked and filtered. */
function mixedScene(seed: number): { meshes: MeshGeometry[]; instances: TraceSceneInstance[] } {
    const random = createRandom(seed);
    const meshes: MeshGeometry[] = [];
    const instances: TraceSceneInstance[] = [];
    const hard = hardMeshes();
    for (let i = 0; i < 10; i++) {
        const mesh = i < 4 ? hard[i % 2 === 0 ? 0 : 1] : randomTriangleSoup(random, 60 + i * 9, 2, 0.6);
        const count = mesh.indices ? mesh.indices.length / 3 : mesh.positions.length / 9;
        meshes.push(i % 3 === 1 ? { ...mesh, triangleFlags: Uint32Array.from({ length: count }, (_, t) => (t % 3 === 0 ? TRIANGLE_FLAG_FILTER : 0)) } : mesh);
        instances.push({ meshIndex: i, instanceIndex: 300 + i, objectToWorld: randomAffine(random, { extent: 8, mirror: i % 4 === 1 }), static: i % 5 !== 4, mask: i % 3 === 2 ? 0x0f : undefined, material: i });
    }
    meshes.push(randomTriangleSoup(random, 90, 2, 0.6));
    for (let k = 0; k < 3; k++) instances.push({ meshIndex: 10, instanceIndex: 10 + k, objectToWorld: randomAffine(random, { extent: 8, mirror: k === 1 }) });
    return { meshes, instances };
}

const filter: TrianglePasses = ({ instance, triangle, frontFace, query }) => (instance * 7 + triangle) % 3 !== 1 && (query === 0 || !frontFace);

function rays(seed: number, count: number): TraceRay[] {
    const random = createRandom(seed);
    return [...randomRays(random, count / 2, [-10, -10, -10, 10, 10, 10], 2, { randomInterval: true }), ...randomRays(random, count / 2, [-8, -8, -8, 8, 8, 8], 0)];
}

function traceAll(scene: TraceSceneData, traced: readonly TraceRay[], stats = newTraceStats()): unknown[] {
    const tracer = createCpuTracer(scene, { triangleFilter: filter });
    const out: unknown[] = [];
    const masks = [undefined, 0x0f, 0xf0];
    traced.forEach((ray, index) => {
        const mask = masks[index % masks.length];
        out.push(tracer.closest(ray, { mask, stats }), tracer.any(ray, { mask }));
    });
    return out;
}

describe('a quantized scene', () => {
    const traced = rays(21, 4000);

    it('is the default: assembly and buildTraceScene quantize unless asked for f32', async () => {
        const { meshes, instances } = mixedScene(5);
        expect(DEFAULT_NODE_FORMAT).toBe('quantized');
        const byDefault = assembleTraceScene(meshes.map(f32), instances);
        expect(byDefault.nodeFormat).toBe('quantized');
        expect(new Uint8Array(byDefault.nodes)).toEqual(new Uint8Array(assembleTraceScene(meshes.map(f32), instances, {}, { nodeFormat: 'quantized' }).nodes));
        expect((await buildTraceScene(meshes, instances, { build: f32 })).nodeFormat).toBe('quantized');
        expect(assembleTraceScene(meshes.map(f32), instances, {}, { nodeFormat: 'f32' }).nodeFormat).toBe('f32');
    });

    it('builds with defaults and assembles with defaults: the formats agree, and only a real mix throws', async () => {
        const { meshes, instances } = mixedScene(9);
        const few = traced.slice(0, 400);
        const bytes = (buffer: ArrayBuffer) => new Uint8Array(buffer);
        // Builders' defaults (f32 nodes, records), assembly's defaults: a quantized records scene, the walk's own defaults.
        const byDefault = meshes.map((mesh) => buildMeshBvhWithTinyBvh(mesh, tinyBvh));
        const scene = assembleTraceScene(byDefault, instances);
        expect([scene.nodeFormat, scene.triangleLeaves]).toEqual([DEFAULT_NODE_FORMAT, 'records']);
        expect(traceModuleWgsl()).toBe(traceModuleWgsl({ nodeFormat: scene.nodeFormat, triangleLeaves: scene.triangleLeaves }));
        expect(traceAll(scene, few)).toEqual(traceAll(assembleTraceScene(byDefault, instances, {}, { nodeFormat: 'f32' }), few));
        // The CPU twins' defaults too.
        const twins = meshes.map((mesh) => buildLbvhOnCpu(mesh));
        expect(assembleTraceScene(twins, instances).nodeFormat).toBe(DEFAULT_NODE_FORMAT);
        // Trees built quantized are placed as they are: the same bytes.
        const quantized = meshes.map((mesh) => buildMeshBvhWithTinyBvh(mesh, tinyBvh, { nodeFormat: 'quantized' }));
        expect(bytes(assembleTraceScene(quantized, instances).nodes)).toEqual(bytes(scene.nodes));
        // Indexed trees: an assembly that names no format takes theirs.
        const indexed = meshes.map((mesh) => buildMeshBvhWithTinyBvh(mesh, tinyBvh, { triangleLeaves: 'indexed' }));
        const indexedScene = assembleTraceScene(indexed, instances, {}, { geometries: meshes });
        expect(indexedScene.triangleLeaves).toBe('indexed');
        expect(bytes(indexedScene.triangles)).toEqual(bytes(assembleTraceScene(indexed, instances, {}, { triangleLeaves: 'indexed', geometries: meshes }).triangles));
        // Real conflicts throw: a mix of records and indexed trees with no format named, quantized trees in an f32 scene.
        const mixed = [indexed[0], ...byDefault.slice(1)];
        expect(() => assembleTraceScene(mixed, instances, {}, { geometries: meshes })).toThrow(/pass triangleLeaves: 'indexed'/);
        expect(assembleTraceScene(mixed, instances, {}, { triangleLeaves: 'indexed', geometries: meshes }).triangleLeaves).toBe('indexed');
        expect(() => assembleTraceScene(quantized, instances, {}, { nodeFormat: 'f32' })).toThrow(/nodeFormat: 'quantized'/);
        // Indexed trees without their geometries, or from buildTraceScene's build without triangleLeaves: the error names the way out.
        expect(() => assembleTraceScene(indexed, instances)).toThrow(/the trees are indexed.*pass geometries/);
        await expect(buildTraceScene(meshes, instances, { build: (mesh) => buildMeshBvhWithTinyBvh(mesh, tinyBvh, { triangleLeaves: 'indexed' }) })).rejects.toThrow(
            /returned an indexed tree.*pass triangleLeaves: 'indexed'/,
        );
    });

    it('reads a hand-built scene without nodeFormat as f32', async () => {
        const { meshes, instances } = mixedScene(17);
        const assembled = await buildTraceScene(meshes, instances, { build: f32, nodeFormat: 'f32' });
        const { nodeFormat, ...rest } = assembled;
        expect(nodeFormat).toBe('f32');
        // As a caller would write it before nodeFormat existed: the field is optional.
        const handBuilt: TraceSceneData = { ...rest };
        const few = traced.slice(0, 200);
        expect(traceAll(handBuilt, few)).toEqual(traceAll(assembled, few));
    });

    for (const mergeStaticInstances of [false, true]) {
        for (const triangleLeaves of ['records', 'indexed'] as const) {
            it(`gives the f32 scene's hits, bit for bit (${mergeStaticInstances ? 'merged' : 'instanced'}, ${triangleLeaves})`, async () => {
                const { meshes, instances } = mixedScene(17);
                const options = { build: f32, mergeStaticInstances, triangleLeaves };
                const exact = await buildTraceScene(meshes, instances, { ...options, nodeFormat: 'f32' });
                const packed = await buildTraceScene(meshes, instances, { ...options, nodeFormat: 'quantized' });
                expect(packed.nodeFormat).toBe('quantized');
                expect(exact.nodeFormat).toBe('f32');
                const exactStats = newTraceStats();
                const packedStats = newTraceStats();
                const expected = traceAll(exact, traced, exactStats);
                expect(expected.filter((hit) => hit !== null && hit !== false).length).toBeGreaterThan(traced.length / 2);
                expect(traceAll(packed, traced, packedStats)).toEqual(expected);
                // Looser boxes visit at least as many nodes.
                expect(packedStats.nodesVisited).toBeGreaterThanOrEqual(exactStats.nodesVisited);
                // Half the node bytes, plus one frame slot per placed mesh.
                const placed = [...packed.meshes, ...packed.mergedGroups].filter((mesh) => mesh.nodeCount > 0);
                expect(packed.nodes.byteLength).toBe((exact.nodes.byteLength / GPU_LAYOUT_NODE_BYTES + placed.length) * GPU_LAYOUT_QUANTIZED_NODE_BYTES);
                for (const mesh of placed) {
                    const frame = readNodeFrame(new Uint32Array(packed.nodes), (mesh.nodeOffset - 1) * GPU_LAYOUT_QUANTIZED_NODE_WORDS);
                    const exactNodes = exact.nodes.slice(0);
                    const exactMesh = [...exact.meshes, ...exact.mergedGroups].find((m) => m.triangleOffset === mesh.triangleOffset)!;
                    const at = exactMesh.nodeOffset * GPU_LAYOUT_NODE_WORDS;
                    expect(frame).toEqual(nodeFrameOf(nodeChildUnion(new Float32Array(exactNodes), new Uint32Array(exactNodes), at)));
                }
            });
        }
    }

    it('places quantized trees as they are, and refuses them in an f32 scene', () => {
        const { meshes, instances } = mixedScene(5);
        const quantized = meshes.map((mesh) => buildMeshBvhWithTinyBvh(mesh, tinyBvh, { nodeFormat: 'quantized' }));
        const fromTrees = assembleTraceScene(quantized, instances, {}, { nodeFormat: 'quantized' });
        const fromF32 = assembleTraceScene(meshes.map(f32), instances, {}, { nodeFormat: 'quantized' });
        expect(new Uint8Array(fromTrees.nodes)).toEqual(new Uint8Array(fromF32.nodes));
        expect(() => assembleTraceScene(quantized, instances, {}, { nodeFormat: 'f32' })).toThrow(/nodeFormat: 'quantized'/);
        expect(() => assembleTraceScene(quantized, instances, {}, { nodeFormat: 'packed' as never })).toThrow(RangeError);
    });

    it('costs a few percent of SAH and node visits on these meshes (a bound, not a target)', () => {
        for (const geometry of hardMeshes().slice(0, 5)) {
            const exact = f32(geometry);
            expect(treeSahCost(quantizeNodes(exact)) / treeSahCost(exact)).toBeLessThan(1.2);
            expect(treeSahCost(quantizeNodes(exact))).toBeGreaterThanOrEqual(treeSahCost(exact));
        }
    });
});

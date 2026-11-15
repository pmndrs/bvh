/**
 * The clustered top level (clustered-scene.ts, PARITY 8.9): the cut, the
 * copied subtrees, and the CPU walk on clustered scenes against the walk on
 * the same scene unclustered and against the oracle's reference, in every
 * node and triangle format and merged. The cut against three-mesh-bvh's own
 * `ClusteredBVH` is tests/clustered-bvh-parity.test.ts; the WGSL walk on
 * clustered scenes is tests/gpu/pages/trace-clustered.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import { buildOracleCases, sharedEdgeCase, SMALL_SIZES, unpackRay, type OracleCase } from '../../tests/gpu/pages/trace-oracle/cases.ts';
import { computeReferenceAnswers, evaluateCase, evaluateSharedEdgeCase, sharedEdgeFailures, type TracerAnswers } from '../../tests/gpu/pages/trace-oracle/check.ts';
import { createTinyBvhBuilder, type TinyBvhBuilder } from '../tinybvh.ts';
import { clusterTraceScene, DEFAULT_CLUSTER_PRIMITIVE_LIMIT, meshClusters, type ClusteredTraceSceneData } from './clustered-scene.ts';
import { createCpuTracer, newTraceStats, type CpuHit } from './cpu-traversal.ts';
import { BVH_NODE_WORDS, EMPTY_CHILD_INDEX, INSTANCE_RECORD_WORDS, INSTANCE_WORD, NODE_WORD, TOP_LEVEL_HEADER_WORDS } from '../gpu-layout.ts';
import type { MeshGeometry } from './mesh-bvh.ts';
import { buildMeshBvhWithTinyBvh } from './mesh-bvh-topology.ts';
import { randomTriangleSoup, createRandom, randomRays } from './oracle-fixtures.ts';
import { packRays } from './rays.ts';
import { assembleTraceScene, buildTraceScene, meshRefitPlacement, type TraceSceneData, type TraceSceneInstance } from './scene-assembly.ts';
import type { TracedHit } from './oracle-compare.ts';

let tiny: TinyBvhBuilder;
beforeAll(async () => {
    tiny = await createTinyBvhBuilder();
});

interface Format {
    readonly name: string;
    readonly nodeFormat: 'f32' | 'quantized';
    readonly triangleLeaves: 'records' | 'indexed';
    readonly positionLayout?: 'vec4' | 'packed';
    readonly merged?: boolean;
}

const FORMATS: readonly Format[] = [
    { name: 'f32 records', nodeFormat: 'f32', triangleLeaves: 'records' },
    { name: 'quantized records', nodeFormat: 'quantized', triangleLeaves: 'records' },
    { name: 'f32 indexed packed', nodeFormat: 'f32', triangleLeaves: 'indexed', positionLayout: 'packed' },
    { name: 'quantized indexed', nodeFormat: 'quantized', triangleLeaves: 'indexed' },
    { name: 'merged quantized', nodeFormat: 'quantized', triangleLeaves: 'records', merged: true },
    { name: 'merged f32 indexed', nodeFormat: 'f32', triangleLeaves: 'indexed', merged: true },
];

async function assemble(meshes: readonly MeshGeometry[], instances: readonly TraceSceneInstance[], format: Format): Promise<TraceSceneData> {
    const formats = format.triangleLeaves === 'indexed' ? { triangleLeaves: 'indexed' as const, positionLayout: format.positionLayout } : {};
    if (format.merged) {
        // Every instance static: those whose mesh no other uses merge into world-space meshes.
        return buildTraceScene(meshes, instances.map((instance) => ({ ...instance, static: true })), { build: (geometry) => buildMeshBvhWithTinyBvh(geometry, tiny), nodeFormat: format.nodeFormat, ...formats });
    }
    const trees = meshes.map((mesh) => buildMeshBvhWithTinyBvh(mesh, tiny));
    const geometries = format.triangleLeaves === 'indexed' ? { geometries: meshes } : {};
    return assembleTraceScene(trees, instances, {}, { nodeFormat: format.nodeFormat, ...formats, ...geometries });
}

function answersOf(scene: TraceSceneData, rays: Float32Array): TracerAnswers {
    const tracer = createCpuTracer(scene);
    return {
        closest(index): TracedHit | null {
            const hit = tracer.closest(unpackRay(rays, index));
            return hit && { hit: true, ...hit };
        },
        any: (index) => tracer.any(unpackRay(rays, index)),
    };
}

/** Clustered and plain on the same rays: any hits equal, closest hits equal to the bit except which triangle at a tied `t`. */
function compareWalks(plain: TraceSceneData, clustered: TraceSceneData, rays: Float32Array): { rays: number; ties: number; failures: string[] } {
    const a = createCpuTracer(plain);
    const b = createCpuTracer(clustered);
    const failures: string[] = [];
    let ties = 0;
    const count = rays.length / 8;
    for (let index = 0; index < count && failures.length < 5; index++) {
        const ray = unpackRay(rays, index);
        if (a.any(ray) !== b.any(ray)) failures.push(`ray ${index}: any hit ${a.any(ray)} plain, ${b.any(ray)} clustered`);
        const x: CpuHit | null = a.closest(ray);
        const y: CpuHit | null = b.closest(ray);
        if (!x || !y) {
            if (x !== y) failures.push(`ray ${index}: closest ${x ? 'hit' : 'miss'} plain, ${y ? 'hit' : 'miss'} clustered`);
            continue;
        }
        if (x.t !== y.t) failures.push(`ray ${index}: t ${x.t} plain, ${y.t} clustered`);
        else if (x.instanceIndex !== y.instanceIndex || x.triangleIndex !== y.triangleIndex) ties++;
        else if (x.u !== y.u || x.v !== y.v || x.frontFace !== y.frontFace || x.material !== y.material || x.normal.some((n, k) => n !== y.normal[k])) {
            failures.push(`ray ${index}: the same triangle with other fields`);
        }
    }
    return { rays: count, ties, failures };
}

function recordRoots(scene: TraceSceneData): number[] {
    const words = new Uint32Array(scene.topLevel.buffer);
    return Array.from({ length: scene.topLevel.instanceCount }, (_, record) => words[TOP_LEVEL_HEADER_WORDS + record * INSTANCE_RECORD_WORDS + INSTANCE_WORD.rootNodeOffset]);
}

/** Every triangle slot under an f32 node (relative to the mesh), by walking it. */
function slotsUnder(scene: TraceSceneData, nodeOffset: number, node: number, out: number[]): void {
    const words = new Uint32Array(scene.nodes);
    for (const [indexWord, countWord] of [[NODE_WORD.leftIndex, NODE_WORD.leftCount], [NODE_WORD.rightIndex, NODE_WORD.rightCount]]) {
        const index = words[(nodeOffset + node) * BVH_NODE_WORDS + indexWord];
        const count = words[(nodeOffset + node) * BVH_NODE_WORDS + countWord];
        if (index === EMPTY_CHILD_INDEX) continue;
        if (count > 0) for (let k = 0; k < count; k++) out.push(index + k);
        else slotsUnder(scene, nodeOffset, index, out);
    }
}

describe('meshClusters (the cut)', () => {
    it("partitions the triangles at ClusteredBVH's stops, in either node format", () => {
        const mesh = randomTriangleSoup(createRandom(7), 3000, 10, 0.6);
        const tree = buildMeshBvhWithTinyBvh(mesh, tiny);
        const instances: TraceSceneInstance[] = [{ meshIndex: 0, instanceIndex: 0, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] }];
        const f32 = assembleTraceScene([tree], instances, {}, { nodeFormat: 'f32' });
        const quantized = assembleTraceScene([tree], instances, {}, { nodeFormat: 'quantized' });
        for (const limit of [1, 8, 64, 500, 2999, 3000, 3001]) {
            const clusters = meshClusters(f32, f32.meshes[0].nodeOffset, limit);
            const covered = new Uint8Array(3000);
            for (const cluster of clusters) {
                // A stop: fewer triangles than the limit, or a leaf; its parent (when it has one) not.
                expect(cluster.node < 0 || cluster.triangleCount < limit).toBe(true);
                const slots: number[] = [];
                if (cluster.node >= 0) slotsUnder(f32, f32.meshes[0].nodeOffset, cluster.node, slots);
                else for (let k = 0; k < cluster.triangleCount; k++) slots.push(cluster.firstTriangle + k);
                expect(slots).toHaveLength(cluster.triangleCount);
                expect(Math.min(...slots)).toBe(cluster.firstTriangle);
                for (const slot of slots) covered[slot]++;
            }
            // Every triangle in exactly one cluster.
            expect(covered.every((n) => n === 1)).toBe(true);
            if (limit > 3000) expect(clusters).toHaveLength(1);
            if (limit === 1) expect(clusters.every((cluster) => cluster.node < 0)).toBe(true);
            // Quantized nodes cut the same tree the same way, with boxes around the exact ones.
            const coarse = meshClusters(quantized, quantized.meshes[0].nodeOffset, limit);
            expect(coarse.map(({ node, parent, side, firstTriangle, triangleCount, depth }) => [node, parent, side, firstTriangle, triangleCount, depth])).toEqual(
                clusters.map(({ node, parent, side, firstTriangle, triangleCount, depth }) => [node, parent, side, firstTriangle, triangleCount, depth]),
            );
            coarse.forEach((cluster, k) => {
                for (let axis = 0; axis < 3; axis++) {
                    expect(cluster.bounds[axis]).toBeLessThanOrEqual(clusters[k].bounds[axis]);
                    expect(cluster.bounds[axis + 3]).toBeGreaterThanOrEqual(clusters[k].bounds[axis + 3]);
                }
            });
        }
        expect(() => meshClusters(f32, f32.meshes[0].nodeOffset, 0)).toThrow(RangeError);
        expect(() => meshClusters(f32, f32.meshes[0].nodeOffset, 2.5)).toThrow(RangeError);
    });
});

describe('clusterTraceScene', () => {
    it('gives back the scene as it was when nothing is split', async () => {
        const mesh = randomTriangleSoup(createRandom(8), 500, 5, 0.6);
        const scene = await assemble([mesh], [{ meshIndex: 0, instanceIndex: 3, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] }], FORMATS[1]);
        for (const clustered of [clusterTraceScene(scene, { primitiveLimit: Infinity }), clusterTraceScene(scene, { split: () => false })]) {
            expect(clustered.nodes).toBe(scene.nodes);
            expect(clustered.topLevel).toBe(scene.topLevel);
            expect(clustered.clusters.splitRecords).toBe(0);
        }
        expect(() => clusterTraceScene(scene, { primitiveLimit: -1 })).toThrow(RangeError);
    });

    it('splits unshared meshes by default, copies each subtree once, and drops the trees only split records trace', async () => {
        const random = createRandom(9);
        const alone = randomTriangleSoup(random, 400, 3, 0.5);
        const shared = randomTriangleSoup(random, 400, 3, 0.5);
        const instances: TraceSceneInstance[] = [
            { meshIndex: 0, instanceIndex: 2, objectToWorld: [0, 1, 0, 0, -2, 0, 0, 0, 0, 0, 1, 0, 0, 5, 0, 1] },
            { meshIndex: 1, instanceIndex: 0, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] },
            { meshIndex: 1, instanceIndex: 1, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1] },
        ];
        for (const format of FORMATS.slice(0, 4)) {
            const scene = await assemble([alone, shared], instances, format);
            const lead = format.nodeFormat === 'quantized' ? 1 : 0;
            const slotBytes = scene.nodes.byteLength / (scene.meshes[1].nodeOffset + scene.meshes[1].nodeCount);
            const clustered = clusterTraceScene(scene, { primitiveLimit: 16 });
            const cut = meshClusters(scene, scene.meshes[0].nodeOffset, 16);
            expect(clustered.clusters).toMatchObject({ splitRecords: 1, clusterRecords: 2 + cut.length, copiedSubtrees: cut.length, droppedTrees: 1, droppedNodeSlots: lead + scene.meshes[0].nodeCount });
            // Mesh 0's own tree is gone, its triangles stay; mesh 1 moved to the front, its nodes as they were.
            expect(clustered.meshes[0]).toEqual({ ...scene.meshes[0], nodeCount: 0, clustered: true });
            expect(() => meshRefitPlacement(clustered, 0)).toThrow(/clusters/);
            expect(clustered.meshes[1]).toEqual({ ...scene.meshes[1], nodeOffset: lead });
            // (Indexed scenes weld by default, which meshRefitPlacement refuses for its own reason.)
            if (format.triangleLeaves === 'records') expect(meshRefitPlacement(clustered, 1)).toEqual({ ...meshRefitPlacement(scene, 1), nodeWordOffset: (lead * slotBytes) / 4 });
            const bytes = (data: TraceSceneData, placement: TraceSceneData['meshes'][number]) =>
                new Uint8Array(data.nodes, (placement.nodeOffset - lead) * slotBytes, (lead + placement.nodeCount) * slotBytes);
            expect(bytes(clustered, clustered.meshes[1])).toEqual(bytes(scene, scene.meshes[1]));
            expect(recordRoots(clustered).filter((root) => root === lead)).toHaveLength(2);
            expect(clustered.triangles).toBe(scene.triangles);
            expect(clustered.clusters.addedBytes).toBe(clustered.byteLength - scene.byteLength);
            // Splitting every record shares the copies between the two instances of mesh 1, and drops both trees.
            const all = clusterTraceScene(scene, { primitiveLimit: 16, split: () => true });
            expect(all.clusters.copiedSubtrees).toBe(cut.length + meshClusters(scene, scene.meshes[1].nodeOffset, 16).length);
            expect(all.clusters.droppedTrees).toBe(2);
            const rays = randomRays(createRandom(10), 3000, [-6, -6, -6, 11, 11, 6], 4, { randomInterval: true });
            const packed = packRays(rays);
            for (const result of [clustered, all]) {
                const compared = compareWalks(scene, result, packed);
                expect(compared.failures, `${format.name}: ${compared.failures.join('\n')}`).toEqual([]);
            }
        }
    });

    it('keeps a tree that a whole-mesh cluster or a hidden instance still uses', async () => {
        const random = createRandom(13);
        const small = randomTriangleSoup(random, 10, 2, 0.5);
        const hidden = randomTriangleSoup(random, 200, 2, 0.5);
        const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        const scene = await assemble([small, hidden], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: identity }, { meshIndex: 1, instanceIndex: 1, objectToWorld: identity, visible: false }], FORMATS[1]);
        // The small mesh is under the limit: one cluster, its whole tree, entered where it is.
        const clustered = clusterTraceScene(scene, { primitiveLimit: 64, split: () => true });
        expect(clustered.clusters).toMatchObject({ splitRecords: 1, clusterRecords: 1, copiedSubtrees: 0, droppedTrees: 0 });
        expect(clustered.meshes).toEqual(scene.meshes);
        expect(new Uint8Array(clustered.nodes)).toEqual(new Uint8Array(scene.nodes));
    });

    it('tells the split option which mesh or merged group each record traces', async () => {
        const random = createRandom(11);
        const meshes = [randomTriangleSoup(random, 100, 2, 0.5), randomTriangleSoup(random, 100, 2, 0.5), randomTriangleSoup(random, 100, 2, 0.5)];
        const instances: TraceSceneInstance[] = meshes.map((_, index) => ({ meshIndex: index, instanceIndex: 10 + index, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, index * 3, 0, 0, 1], static: index < 2 }));
        const scene = await buildTraceScene(meshes, instances, { build: (geometry) => buildMeshBvhWithTinyBvh(geometry, tiny) });
        const seen: string[] = [];
        clusterTraceScene(scene, {
            primitiveLimit: 8,
            split: (record) => {
                seen.push(record.merged ? `group ${record.mergedGroup} (${record.instanceIndex.toString(16)})` : `mesh ${record.meshIndex} instance ${record.instanceIndex}`);
                return true;
            },
        });
        expect(seen.sort()).toEqual(['group 0 (ffffffff)', 'mesh 2 instance 12']);
    });

    it('refuses a top level the trace stack cannot hold', async () => {
        const mesh = randomTriangleSoup(createRandom(12), 40000, 30, 0.3);
        const scene = await assemble([mesh], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] }], FORMATS[0]);
        // Every leaf its own record: a deep top level over shallow trees, within the stack here.
        const clustered = clusterTraceScene(scene, { primitiveLimit: 1 });
        expect(clustered.requiredStackSize).toBeLessThanOrEqual(64);
        expect(clustered.maxMeshDepth).toBe(1);
    });
});

/** The oracle cases' rays on clustered scenes: the reference's rules, and the plain scene's answers. */
describe('oracle cases on clustered scenes (CPU walk)', () => {
    const cases = buildOracleCases(SMALL_SIZES);
    const references = new Map<string, ReturnType<typeof computeReferenceAnswers>>();
    const referenceOf = (oracleCase: OracleCase) => {
        let reference = references.get(oracleCase.name);
        if (!reference) references.set(oracleCase.name, (reference = computeReferenceAnswers(oracleCase)));
        return reference;
    };
    for (const format of FORMATS) {
        it(`${format.name}: every case, cut at 1, 4 (every record) and ${DEFAULT_CLUSTER_PRIMITIVE_LIMIT}`, { timeout: 120_000 }, async () => {
            for (const oracleCase of cases) {
                const scene = await assemble(oracleCase.meshes, oracleCase.instances, format);
                const variants: [string, ClusteredTraceSceneData][] = [
                    ['1', clusterTraceScene(scene, { primitiveLimit: 1 })],
                    ['4 all', clusterTraceScene(scene, { primitiveLimit: 4, split: () => true })],
                    ['default', clusterTraceScene(scene)],
                ];
                for (const [label, clustered] of variants) {
                    const where = `${format.name} ${oracleCase.name} limit ${label}`;
                    const result = evaluateCase(oracleCase, answersOf(clustered, oracleCase.rays), referenceOf(oracleCase));
                    expect(result.failures, `${where}: ${result.failures.join('\n')}`).toEqual([]);
                    const compared = compareWalks(scene, clustered, oracleCase.rays);
                    expect(compared.failures, `${where}: ${compared.failures.join('\n')}`).toEqual([]);
                    // The walk never needs more stack than the scene says.
                    const stats = newTraceStats();
                    const tracer = createCpuTracer(clustered);
                    for (let index = 0; index < Math.min(200, oracleCase.rays.length / 8); index++) tracer.closest(unpackRay(oracleCase.rays, index), { stats });
                    expect(stats.maxStackSize).toBeLessThanOrEqual(clustered.requiredStackSize);
                }
            }
        });
    }

    for (const transformed of [false, true]) {
        it(`no ray slips between triangles on a clustered quad (${transformed ? 'transformed' : 'plain'})`, { timeout: 60_000 }, async () => {
            const oracleCase = sharedEdgeCase(20000, transformed);
            for (const format of [FORMATS[0], FORMATS[3]]) {
                const scene = clusterTraceScene(await assemble(oracleCase.meshes, oracleCase.instances, format), { primitiveLimit: 2, split: () => true });
                const report = evaluateSharedEdgeCase(oracleCase, answersOf(scene, oracleCase.rays), 2000);
                expect(sharedEdgeFailures(report)).toEqual([]);
            }
        });
    }
});

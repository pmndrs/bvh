/// <reference types="node" />
/**
 * Quantized nodes (docs/internals/design/compressed-nodes.md, section 3 step 1) on real
 * scenes, in Node with no GPU: how much the outward-rounded boxes cost.
 *
 *   BVH_ASSETS_DIR=... BVH_MANIFESTS_DIR=... node --experimental-transform-types tools/quantized-nodes.ts sponza bistro-exterior
 *   node --experimental-transform-types tools/quantized-nodes.ts sponza --rays 20000 --json out.json
 *
 * Per scene, for the instanced tree (tinybvh per mesh, one frame each) and the
 * merged tree (`mergeStaticInstances`, every instance static, one frame per
 * merged group):
 *
 * - SAH: the summed SAH cost of every placed tree (`treeSahCost`, relative to
 *   the root box, times the root box's half-area), on the decoded boxes over
 *   the exact ones; and per merged group alone (the case most at risk);
 * - the CPU walk's node visits and triangle tests per ray, quantized over
 *   `f32`, on the air-grid rays (a probe grid over the scene, the bench's
 *   `air-grid` set, closest and any hit) and on secondary rays from their
 *   closest hits (a cosine-free hemisphere sample off each hit, a stand-in for
 *   the bench's `surface` set, which needs the GPU to place);
 * - every hit is checked: quantized must give the `f32` scene's instance,
 *   triangle and `t` bit for bit (the triangle test is unchanged), and the
 *   tool exits non-zero if not. A tie (another triangle at exactly the same
 *   `t`, a ray through a shared edge) is counted apart: the walk keeps the last
 *   one it tests, and looser boxes can change the order.
 */

import { writeFileSync } from 'node:fs';
import { createTinyBvhBuilder, type GpuLayoutMeshBvh } from '../src/index.ts';
import { treeSahCost } from '../src/layout-check.ts';
import { quantizeNodes } from '../src/quantized-nodes.ts';
import { createCpuTracer, newTraceStats, type CpuHit, type CpuTraceStats } from '../src/trace/cpu-traversal.ts';
import { buildMeshBvhWithTinyBvh } from '../src/trace/mesh-bvh-topology.ts';
import { createRandom, randomUnitVector } from '../src/trace/oracle-fixtures.ts';
import { assembleTraceScene, type TraceSceneData, type TraceSceneInstance } from '../src/trace/scene-assembly.ts';
import { mergeStaticGeometry } from '../src/trace/static-merge.ts';
import type { MeshBvh } from '../src/trace/mesh-bvh.ts';
import type { TraceRay } from '../src/trace/rays.ts';
import { makeProbeGrid, makeProbeRays, makeSphereDirections, worldBounds } from '../tests/gpu/pages/trace-bench/rays.ts';
import { readSceneInstances } from './scene-meshes.ts';

const args = process.argv.slice(2);
const flag = (name: string): string | undefined => {
    const at = args.indexOf(name);
    return at >= 0 ? args[at + 1] : undefined;
};
const rayTarget = Number(flag('--rays') ?? 40000);
const jsonPath = flag('--json');
const sceneIds = args.filter((arg, index) => !arg.startsWith('--') && !args[index - 1]?.startsWith('--'));
if (sceneIds.length === 0) sceneIds.push('sponza');

const halfArea = (box: ArrayLike<number>): number => {
    const x = box[3] - box[0];
    const y = box[4] - box[1];
    const z = box[5] - box[2];
    return x * y + y * z + z * x;
};
/** A tree's SAH cost in absolute terms (relative cost times its box's half-area), so trees can be summed. */
const absoluteSah = (tree: GpuLayoutMeshBvh): number => treeSahCost(tree) * halfArea(tree.bounds);

interface WalkTotals {
    rays: number;
    nodes: number;
    triangles: number;
}

function walk(scene: TraceSceneData, rays: readonly TraceRay[], kind: 'closest' | 'any'): { totals: WalkTotals; hits: (CpuHit | boolean | null)[] } {
    const tracer = createCpuTracer(scene);
    const stats: CpuTraceStats = newTraceStats();
    const hits = rays.map((ray) => (kind === 'closest' ? tracer.closest(ray, { stats }) : tracer.any(ray, { stats })));
    return { totals: { rays: rays.length, nodes: stats.nodesVisited, triangles: stats.trianglesTested }, hits };
}

/** Secondary rays off each closest hit: origin nudged along the normal, direction in the hemisphere the normal faces. */
function secondaryRays(rays: readonly TraceRay[], hits: readonly (CpuHit | boolean | null)[], seed: number): TraceRay[] {
    const random = createRandom(seed);
    const out: TraceRay[] = [];
    rays.forEach((ray, index) => {
        const hit = hits[index];
        if (!hit || hit === true) return;
        const facing = hit.frontFace ? 1 : -1;
        const n = hit.normal.map((value) => value * facing);
        const point = ray.origin.map((value, axis) => value + ray.direction[axis] * hit.t);
        const scale = Math.max(1e-4, ...point.map(Math.abs)) * 1e-4;
        const sampled = randomUnitVector(random);
        const sign = sampled[0] * n[0] + sampled[1] * n[1] + sampled[2] * n[2] < 0 ? -1 : 1;
        const direction = sampled.map((value) => value * sign);
        out.push({ origin: point.map((value, axis) => Math.fround(value + n[axis] * scale)) as [number, number, number], direction: direction.map(Math.fround) as [number, number, number], tMin: 0, tMax: 1e6 });
    });
    return out;
}

/**
 * 'same', 'tie' (the same t, another triangle: two triangles hit at exactly one distance,
 * where the walk's order picks, and the looser boxes can change the order) or 'different'.
 */
function compareHits(a: CpuHit | boolean | null, b: CpuHit | boolean | null): 'same' | 'tie' | 'different' {
    if (a === null || b === null || typeof a === 'boolean' || typeof b === 'boolean') return a === b ? 'same' : 'different';
    if (a.t !== b.t) return 'different';
    return a.instanceIndex === b.instanceIndex && a.triangleIndex === b.triangleIndex ? 'same' : 'tie';
}

const results: Record<string, unknown> = {};
let mismatches = 0;
const ratio = (a: number, b: number): string => (b > 0 ? (a / b).toFixed(4) : 'n/a');

const tinyBvh = await createTinyBvhBuilder();
for (const sceneId of sceneIds) {
    const { meshes, instances: placed } = readSceneInstances(sceneId);
    const instances: TraceSceneInstance[] = placed.map((instance, index) => ({ meshIndex: instance.meshIndex, instanceIndex: index, objectToWorld: instance.objectToWorld, static: true }));
    const trees: MeshBvh[] = meshes.map((mesh) => buildMeshBvhWithTinyBvh(mesh, tinyBvh));
    const plan = mergeStaticGeometry(meshes, instances);
    const groupTrees = plan.groups.map((group) => buildMeshBvhWithTinyBvh(group.geometry, tinyBvh));
    const sourceTrees = trees.map((tree, index) => (plan.meshNeeded[index] ? tree : null));
    const scenes = {
        instanced: { f32: assembleTraceScene(trees, instances), quantized: assembleTraceScene(trees, instances, {}, { nodeFormat: 'quantized' }), trees },
        merged: {
            f32: assembleTraceScene(sourceTrees, instances, {}, { merged: { plan, meshes: groupTrees } }),
            quantized: assembleTraceScene(sourceTrees, instances, {}, { merged: { plan, meshes: groupTrees }, nodeFormat: 'quantized' }),
            trees: [...sourceTrees.filter((tree): tree is MeshBvh => tree !== null), ...groupTrees],
        },
    };
    const nodeCount = (list: readonly MeshBvh[]) => list.reduce((total, tree) => total + tree.nodeCount, 0);
    console.log(`${sceneId}: ${meshes.length} meshes, ${instances.length} instances; merged: ${plan.groups.length} groups, ${plan.instanced.length} instances stay instanced`);

    const box = worldBounds(meshes, placed);
    const grid = makeProbeGrid(box, Math.max(1, Math.round(rayTarget / 64)));
    const packedRays = makeProbeRays(grid, makeSphereDirections(64, 1234));
    const airGrid: TraceRay[] = [];
    for (let at = 0; at < packedRays.length; at += 8) {
        airGrid.push({ origin: [packedRays[at], packedRays[at + 1], packedRays[at + 2]], tMin: packedRays[at + 3], direction: [packedRays[at + 4], packedRays[at + 5], packedRays[at + 6]], tMax: packedRays[at + 7] });
    }

    const sceneResult: Record<string, unknown> = {};
    for (const [treeName, pair] of Object.entries(scenes)) {
        const sahF32 = pair.trees.reduce((total, tree) => total + absoluteSah(tree), 0);
        const sahQuantized = pair.trees.reduce((total, tree) => total + absoluteSah(quantizeNodes(tree)), 0);
        console.log(`  ${treeName}: ${nodeCount(pair.trees)} nodes, f32 ${(pair.f32.nodes.byteLength / 2 ** 20).toFixed(1)} MiB, quantized ${(pair.quantized.nodes.byteLength / 2 ** 20).toFixed(1)} MiB; SAH quantized / f32 ${ratio(sahQuantized, sahF32)}`);
        const groups = treeName === 'merged' ? groupTrees.map((tree) => ({ triangles: tree.triangleCount, sah: treeSahCost(quantizeNodes(tree)) / treeSahCost(tree) })) : [];
        for (const group of groups) console.log(`    merged group of ${group.triangles} triangles: SAH quantized / f32 ${group.sah.toFixed(4)}`);

        const walks: Record<string, unknown> = {};
        const closestF32 = walk(pair.f32, airGrid, 'closest');
        const surface = secondaryRays(airGrid, closestF32.hits, 99);
        for (const [setName, rays] of [['air-grid', airGrid], ['secondary', surface]] as const) {
            for (const kind of ['closest', 'any'] as const) {
                const a = setName === 'air-grid' && kind === 'closest' ? closestF32 : walk(pair.f32, rays, kind);
                const b = walk(pair.quantized, rays, kind);
                let bad = 0;
                let ties = 0;
                a.hits.forEach((hit, index) => {
                    const verdict = compareHits(hit, b.hits[index]);
                    if (verdict === 'different') bad++;
                    if (verdict === 'tie') ties++;
                });
                mismatches += bad;
                const perRay = (totals: WalkTotals) => ({ nodes: totals.nodes / totals.rays, triangles: totals.triangles / totals.rays });
                const pa = perRay(a.totals);
                const pb = perRay(b.totals);
                console.log(
                    `    ${setName} ${kind}: ${rays.length} rays; nodes/ray ${pa.nodes.toFixed(2)} -> ${pb.nodes.toFixed(2)} (${ratio(pb.nodes, pa.nodes)}), triangles/ray ${pa.triangles.toFixed(2)} -> ${pb.triangles.toFixed(2)} (${ratio(pb.triangles, pa.triangles)}); hits differing ${bad}, ties ${ties}`,
                );
                walks[`${setName}/${kind}`] = { rays: rays.length, f32: pa, quantized: pb, mismatches: bad, ties };
            }
        }
        sceneResult[treeName] = { nodes: nodeCount(pair.trees), nodeBytes: { f32: pair.f32.nodes.byteLength, quantized: pair.quantized.nodes.byteLength }, sah: { f32: sahF32, quantized: sahQuantized }, groups, walks };
    }
    results[sceneId] = sceneResult;
}
if (jsonPath) writeFileSync(jsonPath, JSON.stringify(results, null, 2));
if (mismatches > 0) {
    console.error(`${mismatches} hits differ between the f32 and quantized scenes`);
    process.exit(1);
}

/// <reference types="node" />
/**
 * Wide nodes (docs/internals/design/wide-nodes.md) on real scenes, in Node with no GPU:
 * what the collapse costs and what it saves.
 *
 *   BVH_ASSETS_DIR=... BVH_MANIFESTS_DIR=... node --experimental-transform-types tools/wide-nodes.ts sponza bistro-interior
 *   node --experimental-transform-types tools/wide-nodes.ts sponza --rays 20000 --json out.json
 *
 * Per scene, for the instanced tree (tinybvh per mesh) and the merged tree
 * (`mergeStaticInstances`, every instance static):
 *
 * - the collapse: milliseconds to convert every placed tree to `bvh4` and to
 *   `cwbvh8` on this thread (`convertToWide`), binary and wide node counts,
 *   children per wide node, triangles per leaf slot and the deepest mesh stack;
 * - bytes: the scene's nodes, triangles and top level in each format;
 * - the CPU walk's node visits and triangle tests per ray in each format, on
 *   the air-grid rays and secondary rays off their hits (as tools/quantized-nodes.ts).
 *   A wide node visit tests 4 or 8 boxes, a binary one 2, so visits are not
 *   work; box tests per ray are reported too (2 per binary visit, the child
 *   slots per wide visit);
 * - every hit is checked against `f32`: the same `t` bit for bit, the same
 *   instance and triangle unless another triangle lies at that `t` (a tie,
 *   counted apart); the tool exits non-zero on any other difference.
 */

import { writeFileSync } from 'node:fs';
import { createTinyBvhBuilder } from '../src/index.ts';
import { convertToWide, type WideNodeFormat } from '../src/wide-nodes.ts';
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
const FORMATS: readonly ('f32' | WideNodeFormat)[] = ['f32', 'bvh4', 'cwbvh8'];
const mib = (bytes: number): string => (bytes / 2 ** 20).toFixed(1);

const tinyBvh = await createTinyBvhBuilder();
for (const sceneId of sceneIds) {
    const { meshes, instances: placed } = readSceneInstances(sceneId);
    const instances: TraceSceneInstance[] = placed.map((instance, index) => ({ meshIndex: instance.meshIndex, instanceIndex: index, objectToWorld: instance.objectToWorld, static: true }));
    const trees: MeshBvh[] = meshes.map((mesh) => buildMeshBvhWithTinyBvh(mesh, tinyBvh));
    const plan = mergeStaticGeometry(meshes, instances);
    const groupTrees = plan.groups.map((group) => buildMeshBvhWithTinyBvh(group.geometry, tinyBvh));
    const sourceTrees = trees.map((tree, index) => (plan.meshNeeded[index] ? tree : null));
    const used = new Set(instances.map((instance) => instance.meshIndex));
    const kinds = {
        instanced: { trees: trees.filter((_, index) => used.has(index)), assemble: (nodeFormat: 'f32' | WideNodeFormat) => assembleTraceScene(trees, instances, {}, nodeFormat === 'f32' ? {} : { nodeFormat }) },
        merged: {
            trees: [...sourceTrees.filter((tree): tree is MeshBvh => tree !== null), ...groupTrees],
            assemble: (nodeFormat: 'f32' | WideNodeFormat) => assembleTraceScene(sourceTrees, instances, {}, { merged: { plan, meshes: groupTrees }, ...(nodeFormat === 'f32' ? {} : { nodeFormat }) }),
        },
    };
    console.log(`${sceneId}: ${meshes.length} meshes, ${instances.length} instances; merged: ${plan.groups.length} groups, ${plan.instanced.length} instances stay instanced`);

    const box = worldBounds(meshes, placed);
    const grid = makeProbeGrid(box, Math.max(1, Math.round(rayTarget / 64)));
    const packedRays = makeProbeRays(grid, makeSphereDirections(64, 1234));
    const airGrid: TraceRay[] = [];
    for (let at = 0; at < packedRays.length; at += 8) {
        airGrid.push({ origin: [packedRays[at], packedRays[at + 1], packedRays[at + 2]], tMin: packedRays[at + 3], direction: [packedRays[at + 4], packedRays[at + 5], packedRays[at + 6]], tMax: packedRays[at + 7] });
    }

    const sceneResult: Record<string, unknown> = {};
    for (const [kindName, kind] of Object.entries(kinds)) {
        const collapse: Record<string, unknown> = {};
        for (const format of ['bvh4', 'cwbvh8'] as const) {
            const started = performance.now();
            let binaryNodes = 0;
            let wideNodes = 0;
            let slots = 0;
            let leafSlots = 0;
            let triangles = 0;
            let stack = 0;
            for (const tree of kind.trees) {
                const wide = convertToWide(tree, format);
                binaryNodes += tree.nodeCount;
                wideNodes += wide.nodeCount;
                slots += wide.childrenPerNode * wide.nodeCount;
                leafSlots += wide.trianglesPerLeaf > 0 ? tree.triangleCount / wide.trianglesPerLeaf : 0;
                triangles += tree.triangleCount;
                stack = Math.max(stack, wide.stackNeed);
            }
            const ms = performance.now() - started;
            collapse[format] = { ms, trees: kind.trees.length, binaryNodes, wideNodes, childrenPerNode: slots / wideNodes, trianglesPerLeaf: triangles / leafSlots, stackNeed: stack };
            console.log(`  ${kindName} collapse to ${format}: ${ms.toFixed(0)} ms for ${kind.trees.length} trees; ${binaryNodes} binary nodes -> ${wideNodes} wide; ${(slots / wideNodes).toFixed(2)} children per node, ${(triangles / leafSlots).toFixed(2)} triangles per leaf slot, stack need ${stack}`);
        }
        const scenes = Object.fromEntries(FORMATS.map((format) => [format, kind.assemble(format)])) as Record<'f32' | WideNodeFormat, TraceSceneData>;
        const bytes = Object.fromEntries(FORMATS.map((format) => [format, { nodes: scenes[format].nodes.byteLength, triangles: scenes[format].triangles.byteLength, topLevel: scenes[format].topLevel.buffer.byteLength, total: scenes[format].byteLength }]));
        console.log(`  ${kindName} bytes: ${FORMATS.map((format) => `${format} nodes ${mib(scenes[format].nodes.byteLength)} MiB (scene ${mib(scenes[format].byteLength)})`).join(', ')}`);

        const walks: Record<string, unknown> = {};
        const closestF32 = walk(scenes.f32, airGrid, 'closest');
        const surface = secondaryRays(airGrid, closestF32.hits, 99);
        for (const [setName, rays] of [['air-grid', airGrid], ['secondary', surface]] as const) {
            for (const kindOfHit of ['closest', 'any'] as const) {
                const base = setName === 'air-grid' && kindOfHit === 'closest' ? closestF32 : walk(scenes.f32, rays, kindOfHit);
                const perFormat: Record<string, unknown> = { f32: { nodes: base.totals.nodes / base.totals.rays, boxes: (2 * base.totals.nodes) / base.totals.rays, triangles: base.totals.triangles / base.totals.rays } };
                const line: string[] = [`f32 ${(base.totals.nodes / base.totals.rays).toFixed(1)} visits / ${((2 * base.totals.nodes) / base.totals.rays).toFixed(1)} boxes / ${(base.totals.triangles / base.totals.rays).toFixed(1)} tris`];
                for (const format of ['bvh4', 'cwbvh8'] as const) {
                    const other = walk(scenes[format], rays, kindOfHit);
                    let bad = 0;
                    let ties = 0;
                    base.hits.forEach((hit, index) => {
                        const verdict = compareHits(hit, other.hits[index]);
                        if (verdict === 'different') bad++;
                        if (verdict === 'tie') ties++;
                    });
                    mismatches += bad;
                    const width = format === 'bvh4' ? 4 : 8;
                    const visits = other.totals.nodes / other.totals.rays;
                    perFormat[format] = { nodes: visits, boxes: visits * width, triangles: other.totals.triangles / other.totals.rays, mismatches: bad, ties };
                    line.push(`${format} ${visits.toFixed(1)} visits (${ratio(visits, base.totals.nodes / base.totals.rays)}) / ${(other.totals.triangles / other.totals.rays).toFixed(1)} tris (${ratio(other.totals.triangles, base.totals.triangles)}); differing ${bad}, ties ${ties}`);
                }
                console.log(`    ${setName} ${kindOfHit} (${rays.length} rays): ${line.join('; ')}`);
                walks[`${setName}/${kindOfHit}`] = { rays: rays.length, ...perFormat };
            }
        }
        sceneResult[kindName] = { collapse, bytes, walks };
    }
    results[sceneId] = sceneResult;
}
if (jsonPath) writeFileSync(jsonPath, JSON.stringify(results, null, 2));
if (mismatches > 0) {
    console.error(`${mismatches} hits differ between the f32 and wide scenes`);
    process.exit(1);
}

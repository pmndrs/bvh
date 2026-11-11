/// <reference types="node" />
/**
 * `mergeStaticInstances` on a real scene, without a GPU: how many instances
 * merge, the tinybvh build time of the merged trees against the per-mesh
 * builds (their sum and their largest), and the bytes of nodes and triangle
 * records, merged against instanced. Every instance is marked static.
 *
 *   node --experimental-transform-types tools/merge-static.ts            Sponza
 *   node --experimental-transform-types tools/merge-static.ts bistro-interior
 *
 * Reads the scenes like tree-quality.ts (BVH_ASSETS_DIR, BVH_MANIFESTS_DIR; tests/gpu/README.md).
 */

import { createTinyBvhBuilder } from '../src/index.ts';
import { assembleTraceScene, buildMeshBvhWithTinyBvh, mergeStaticGeometry, type MeshBvh, type TraceSceneInstance } from '../src/trace/index.ts';
import { readSceneInstances } from './scene-meshes.ts';

const MB = 1024 * 1024;

async function main(): Promise<void> {
    const sceneIds = process.argv.slice(2).length > 0 ? process.argv.slice(2) : ['sponza'];
    const tinyBvh = await createTinyBvhBuilder();
    for (const sceneId of sceneIds) {
        const { meshes, instances: placed } = readSceneInstances(sceneId);
        const instances: TraceSceneInstance[] = placed.map((instance, index) => ({ meshIndex: instance.meshIndex, instanceIndex: index, objectToWorld: instance.objectToWorld, static: true }));
        const triangles = meshes.reduce((total, mesh) => total + mesh.indices.length / 3, 0);
        console.log(`${sceneId}: ${meshes.length} meshes, ${instances.length} instances, ${triangles} unique triangles`);

        // Instanced: every mesh's own tree, one after another, each timed.
        const perMesh: number[] = [];
        const trees: MeshBvh[] = meshes.map((mesh) => {
            const start = performance.now();
            const tree = buildMeshBvhWithTinyBvh(mesh, tinyBvh);
            perMesh.push(performance.now() - start);
            return tree;
        });
        const instanced = assembleTraceScene(trees, instances);
        const sum = perMesh.reduce((a, b) => a + b, 0);
        const max = Math.max(...perMesh);

        const planStart = performance.now();
        const plan = mergeStaticGeometry(meshes, instances);
        const planMs = performance.now() - planStart;
        const mergedTriangles = plan.groups.reduce((total, group) => total + group.sourceInstances.length, 0);
        const groupMs: number[] = [];
        const groupTrees = plan.groups.map((group) => {
            const start = performance.now();
            const tree = buildMeshBvhWithTinyBvh(group.geometry, tinyBvh);
            groupMs.push(performance.now() - start);
            return tree;
        });
        const sourceTrees = trees.map((tree, index) => (plan.meshNeeded[index] ? tree : null));
        const unmergedMs = perMesh.reduce((total, ms, index) => total + (plan.meshNeeded[index] ? ms : 0), 0);
        const assembleStart = performance.now();
        const merged = assembleTraceScene(sourceTrees, instances, {}, { merged: { plan, meshes: groupTrees } });
        const assembleMs = performance.now() - assembleStart;

        const sizes = (scene: typeof merged) => ({
            nodes: scene.nodes.byteLength,
            triangles: scene.triangles.byteLength,
            topLevel: scene.topLevel.buffer.byteLength,
        });
        const a = sizes(instanced);
        const b = sizes(merged);
        console.log(`  merged: ${plan.groups.length} groups, ${plan.groups.reduce((n, g) => n + g.sources.length, 0)} instances, ${mergedTriangles} triangles; ${plan.instanced.length} instances stay instanced`);
        console.log(`  tinybvh per-mesh builds: sum ${sum.toFixed(0)} ms, largest ${max.toFixed(0)} ms (${perMesh.length} meshes)`);
        console.log(`  tinybvh merged builds: ${groupMs.map((ms) => ms.toFixed(0)).join(' + ')} ms (+ ${unmergedMs.toFixed(0)} ms for the meshes still instanced); merge geometry ${planMs.toFixed(0)} ms, assemble ${assembleMs.toFixed(0)} ms`);
        console.log(`  bytes instanced: nodes ${(a.nodes / MB).toFixed(1)} MB, triangles ${(a.triangles / MB).toFixed(1)} MB, top level ${(a.topLevel / 1024).toFixed(1)} KB; nodes ${instanced.meshes.reduce((n, m) => n + m.nodeCount, 0)}, depth top ${instanced.topLevel.maxDepth} mesh ${instanced.maxMeshDepth}`);
        console.log(`  bytes merged:    nodes ${(b.nodes / MB).toFixed(1)} MB, triangles ${(b.triangles / MB).toFixed(1)} MB, top level ${(b.topLevel / 1024).toFixed(1)} KB; nodes ${merged.mergedGroups.reduce((n, m) => n + m.nodeCount, 0) + merged.meshes.reduce((n, m) => n + m.nodeCount, 0)}, depth top ${merged.topLevel.maxDepth} mesh ${merged.maxMeshDepth}`);
        const geometryBytes = plan.groups.reduce((n, g) => n + g.geometry.positions.byteLength + g.geometry.indices!.byteLength + g.sourceInstances.byteLength * 2 + g.flipFrontFace.byteLength, 0);
        console.log(`  transient CPU input of the merged builds (world positions, indices, ids): ${(geometryBytes / MB).toFixed(1)} MB`);
    }
}

await main();

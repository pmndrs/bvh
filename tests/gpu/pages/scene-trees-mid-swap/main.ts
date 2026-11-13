/**
 * The mid-swap oracle for `SceneTrees` (the generic successor of ddgi's
 * `platform-gpu-trees` and `platform-mid-swap` trace-oracle variants, which
 * tested ddgi's own platform code).
 *
 * Every oracle case (tests/gpu/pages/trace-oracle/cases.ts: thin walls,
 * mirrors and non-uniform scales, instancing, a closed thin box no ray may
 * escape, random scenes, a deep mesh, and rays at the seams of a tessellated
 * quad, plain and through a transformed instance) is loaded into a
 * `SceneTrees` with `holdRefinements`, and traced in three states:
 *
 * - `gpu`: every mesh on its GPU H-PLOC tree (the first trees);
 * - `mid-swap`: the merged groups and the meshes with odd handles swapped to
 *   tinybvh's tree, the rest still on the GPU's;
 * - `tinybvh`: every tree swapped.
 *
 * In each state the GPU walk (`TraceBatchKernel` over `SceneTrees.traceBuffers`,
 * with the GPU top level) and the CPU walk over the read-back buffers must
 * give the hits of the CPU walk over the case's scene assembled on the CPU
 * (`assembleTraceScene`, tinybvh trees, instanced): same instance, triangle,
 * side, material and normal, `t` to rounding. A case whose rays must all hit
 * fails on any GPU miss (an escape, or a ray that slipped through a seam).
 *
 * Variants: `gpuTreeOutputs` `copy` and `adopt` with the default cap
 * (`gpuMaxBatchBytes` left out: 256 MiB, one batch for these cases) and with
 * `Infinity` (no cap, the single-batch path), each with
 * `mergeStaticInstances` off and on (on: each instance gets its own copy of
 * its mesh and two of every three are static, as the trace-oracle's merged
 * variants do, so the merged groups' GPU trees get their identity from the
 * GPU pass and their tinybvh trees from the CPU one). The `-capped` variants
 * set `gpuMaxBatchBytes` to 4 KiB, so every mesh is a GPU batch of its
 * own, built one after another on one scratch pool (a case with several
 * meshes must build in more than one batch). Every variant loads with
 * `frame(); await whenBuilt(); frame()`: with a cap the batches go on between
 * updates, so that is enough for every first tree to be a GPU tree.
 *
 * The `indexed-*` variants run the same with `triangleLeaves: 'indexed'`
 * (`vec4` positions, `packed` in the capped ones), copy only: GPU trees
 * written indexed by H-PLOC, tinybvh's converted in the worker, merged groups'
 * side tables from the GPU and the CPU identity pass.
 *
 * The `quantized-*` variants run with `nodeFormat: 'quantized'`
 * (docs/internals/design/compressed-nodes.md), copy only: GPU trees packed by the
 * quantize pass on their way into the storage, tinybvh's quantized in the
 * workers, merged groups one frame each, capped batches stored between
 * updates; one also with indexed leaves (packed positions).
 *
 * URL option: `?sharedRays=100000` shrinks the seam cases.
 */

import { SceneTrees } from '../../../../src/scene/scene-trees.ts';
import { GpuFirstTrees } from '../../../../src/scene/gpu-first-trees.ts';
import { planGpuHplocBatches } from '../../../../src/gpu-hploc.ts';
import { createBvhWorkerPool } from '../../../../src/worker/pool.ts';
import { assembleTraceScene, type TraceSceneInstance } from '../../../../src/trace/scene-assembly.ts';
import { buildMeshBvhWithTinyBvh } from '../../../../src/trace/mesh-bvh-topology.ts';
import { createTinyBvhBuilder } from '../../../../src/tinybvh.ts';
import type { MeshGeometry } from '../../../../src/trace/mesh-bvh.ts';
import { unpackRay, type TraceRay } from '../../../../src/trace/rays.ts';
import { TraceBatchKernel } from '../../../../src/trace/trace-gpu.ts';
import { assert, runGpuTest } from '../../client.ts';
import { compareWithReference, readScene, traceOnGpu } from '../../scene-check.ts';
import { buildOracleCases, FULL_SIZES, sharedEdgeCase, type OracleCase } from '../trace-oracle/cases.ts';

interface Variant {
    readonly name: string;
    readonly gpuTreeOutputs: 'copy' | 'adopt';
    readonly merged: boolean;
    readonly maxBatchBytes?: number;
    /** Indexed leaves (docs/internals/design/indexed-leaves.md), with this position layout; absent: records. */
    readonly indexed?: 'vec4' | 'packed';
    /** Quantized nodes (docs/internals/design/compressed-nodes.md); absent: f32. */
    readonly quantized?: boolean;
}

/** Smaller than any one-mesh batch (its uniforms alone are 2.8 kB), so every mesh is a batch of its own. */
const SMALL_CAP = 4 * 1024;

const VARIANTS: readonly Variant[] = [
    // No maxBatchBytes: the default cap (256 MiB, one batch for these cases); `-uncapped`: Infinity, the single-batch path.
    { name: 'copy', gpuTreeOutputs: 'copy', merged: false },
    { name: 'copy-uncapped', gpuTreeOutputs: 'copy', merged: false, maxBatchBytes: Infinity },
    { name: 'adopt-uncapped', gpuTreeOutputs: 'adopt', merged: false, maxBatchBytes: Infinity },
    { name: 'copy-merged', gpuTreeOutputs: 'copy', merged: true },
    { name: 'adopt', gpuTreeOutputs: 'adopt', merged: false },
    { name: 'adopt-merged', gpuTreeOutputs: 'adopt', merged: true },
    { name: 'copy-capped', gpuTreeOutputs: 'copy', merged: false, maxBatchBytes: SMALL_CAP },
    { name: 'copy-merged-capped', gpuTreeOutputs: 'copy', merged: true, maxBatchBytes: SMALL_CAP },
    { name: 'adopt-capped', gpuTreeOutputs: 'adopt', merged: false, maxBatchBytes: SMALL_CAP },
    { name: 'adopt-merged-capped', gpuTreeOutputs: 'adopt', merged: true, maxBatchBytes: SMALL_CAP },
    // Indexed leaves: copy only (an adopted batch has no mesh headers).
    { name: 'indexed-copy', gpuTreeOutputs: 'copy', merged: false, indexed: 'vec4' },
    { name: 'indexed-copy-merged', gpuTreeOutputs: 'copy', merged: true, indexed: 'vec4' },
    { name: 'indexed-copy-capped', gpuTreeOutputs: 'copy', merged: false, maxBatchBytes: SMALL_CAP, indexed: 'packed' },
    { name: 'indexed-copy-merged-capped', gpuTreeOutputs: 'copy', merged: true, maxBatchBytes: SMALL_CAP, indexed: 'packed' },
    // Quantized nodes: copy only (an adopted batch's nodes are f32).
    { name: 'quantized-copy', gpuTreeOutputs: 'copy', merged: false, quantized: true },
    { name: 'quantized-copy-merged', gpuTreeOutputs: 'copy', merged: true, quantized: true },
    { name: 'quantized-copy-capped', gpuTreeOutputs: 'copy', merged: false, maxBatchBytes: SMALL_CAP, quantized: true },
    { name: 'quantized-indexed-copy-merged-capped', gpuTreeOutputs: 'copy', merged: true, maxBatchBytes: SMALL_CAP, indexed: 'packed', quantized: true },
];

/** Each instance its own mesh copy, two of every three static (the trace-oracle's merged variants). */
function forMerge(meshes: readonly MeshGeometry[], instances: readonly TraceSceneInstance[]) {
    return {
        meshes: instances.map((instance) => meshes[instance.meshIndex]),
        instances: instances.map((instance, position) => ({ ...instance, meshIndex: position, static: position % 3 !== 2 })),
    };
}

await runGpuTest(
    async ({ device, log }) => {
        assert(GpuFirstTrees.isSupported(device), 'GPU first trees need subgroups');
        const sharedRays = Number(new URLSearchParams(location.search).get('sharedRays') ?? 1_000_000);
        const cases: OracleCase[] = [...buildOracleCases(FULL_SIZES), sharedEdgeCase(sharedRays, false), sharedEdgeCase(sharedRays, true)];
        const tinyBvh = await createTinyBvhBuilder();
        const kernels = {
            // The plain variants are f32 (named: the default node format is quantized).
            records: await TraceBatchKernel.create(device, { nodeFormat: 'f32' }),
            vec4: await TraceBatchKernel.create(device, { triangleLeaves: 'indexed', positionLayout: 'vec4', nodeFormat: 'f32' }),
            packed: await TraceBatchKernel.create(device, { triangleLeaves: 'indexed', positionLayout: 'packed', nodeFormat: 'f32' }),
            'quantized-records': await TraceBatchKernel.create(device, { nodeFormat: 'quantized' }),
            'quantized-vec4': await TraceBatchKernel.create(device, { triangleLeaves: 'indexed', positionLayout: 'vec4', nodeFormat: 'quantized' }),
            'quantized-packed': await TraceBatchKernel.create(device, { triangleLeaves: 'indexed', positionLayout: 'packed', nodeFormat: 'quantized' }),
        };
        const firstTrees = new GpuFirstTrees(device, { label: 'mid-swap' });
        const indexedFirstTrees = new GpuFirstTrees(device, { label: 'mid-swap indexed', triangleLeaves: 'indexed' });
        await Promise.all([firstTrees.prepare(), indexedFirstTrees.prepare()]);
        const pool = createBvhWorkerPool({ size: 4 });
        assert(pool.usesWorkers, 'the worker pool fell back to building inline in a browser');
        const failures: string[] = [];
        // Capped variants in which some case built in more than one batch (each must have one).
        const cappedCasesWithBatches = new Set<string>();
        const results: Record<string, unknown> = {};
        let totalRays = 0;

        for (const oracleCase of cases) {
            const rays: TraceRay[] = Array.from({ length: oracleCase.rays.length / 8 }, (_, index) => unpackRay(oracleCase.rays, index));
            const reference = assembleTraceScene(
                oracleCase.meshes.map((mesh) => buildMeshBvhWithTinyBvh(mesh, tinyBvh)),
                oracleCase.instances,
                {},
                { nodeFormat: 'f32' },
            );
            for (const variant of VARIANTS) {
                const { meshes, instances } = variant.merged ? forMerge(oracleCase.meshes, oracleCase.instances) : { meshes: oracleCase.meshes, instances: oracleCase.instances };
                const scene = await SceneTrees.create(device, {
                    label: `${oracleCase.name} ${variant.name}`,
                    holdRefinements: true,
                    gpuTreeOutputs: variant.gpuTreeOutputs,
                    gpuMaxBatchBytes: variant.maxBatchBytes,
                    mergeStaticInstances: variant.merged,
                    workers: pool,
                    firstTreeBuilder: variant.indexed ? indexedFirstTrees : firstTrees,
                    ...(variant.indexed ? { triangleLeaves: 'indexed' as const, positionLayout: variant.indexed } : {}),
                    nodeFormat: variant.quantized ? 'quantized' : 'f32',
                });
                const kernel = kernels[`${variant.quantized ? 'quantized-' : ''}${variant.indexed ?? 'records'}` as keyof typeof kernels];
                const handles = meshes.map((mesh) => scene.addMesh(mesh));
                for (const instance of instances) scene.addInstance({ ...instance, mesh: handles[instance.meshIndex] });
                const frame = () => {
                    const encoder = device.createCommandEncoder();
                    scene.update(encoder);
                    device.queue.submit([encoder.finish()]);
                };
                const check = async (state: string) => {
                    await device.queue.onSubmittedWorkDone();
                    const gpu = await traceOnGpu(device, kernel, scene.traceBuffers, rays);
                    const gpuScene = await readScene(device, scene.buffers, { triangleLeaves: scene.triangleLeaves, positionLayout: scene.positionLayout, nodeFormat: scene.nodeFormat });
                    const comparison = compareWithReference(rays, gpu, gpuScene, reference);
                    let misses = 0;
                    if (oracleCase.everyRayMustHit) for (let index = 0; index < rays.length; index++) if (gpu.closest(index) === null || !gpu.any(index)) misses++;
                    const name = `${oracleCase.name} ${variant.name} ${state}`;
                    const counts = scene.counts;
                    results[name] = { disagreements: comparison.count, misses, gpuTrees: counts.onGpuTrees, tinybvhTrees: counts.onTinybvhTrees };
                    totalRays += rays.length;
                    if (comparison.count > 0) failures.push(`${name}: ${comparison.count} disagreements, first ${JSON.stringify(comparison.disagreements.slice(0, 2))}`);
                    if (misses > 0) failures.push(`${name}: ${misses} rays that must hit missed`);
                    return counts;
                };

                // The same with a cap: the batches go on between updates (each stored in its own submit), so no update is needed until whenBuilt.
                frame();
                await scene.whenBuilt();
                frame();
                if (variant.maxBatchBytes === SMALL_CAP) {
                    const { gpuBatches } = scene.counts;
                    results[`${oracleCase.name} ${variant.name} batches`] = gpuBatches;
                    if (gpuBatches > 1) cappedCasesWithBatches.add(variant.name);
                    // Unmerged, the scene builds exactly the case's meshes: as many batches as the planner makes of them.
                    const planned = planGpuHplocBatches(device, meshes, { maxBatchBytes: SMALL_CAP }).batches.length;
                    if (!variant.merged && gpuBatches !== planned) failures.push(`${oracleCase.name} ${variant.name}: ${gpuBatches} GPU batches, planned ${planned}`);
                }
                const first = await check('gpu');
                if (first.onTinybvhTrees > 0 || first.onGpuTrees === 0) failures.push(`${oracleCase.name} ${variant.name}: the first trees were ${first.onGpuTrees} GPU and ${first.onTinybvhTrees} tinybvh`);
                if (variant.merged && first.mergedGroups === 0) failures.push(`${oracleCase.name} ${variant.name}: nothing merged`);
                await scene.whenRefined();
                // Merged groups (negative keys) and meshes with odd handles swap now; the rest keep their GPU trees.
                scene.releaseRefinements((mesh) => mesh < 0 || mesh % 2 === 1);
                frame();
                const mid = await check('mid-swap');
                const total = mid.onGpuTrees + mid.onTinybvhTrees;
                if (total > 1 && (mid.onGpuTrees === 0 || mid.onTinybvhTrees === 0)) failures.push(`${oracleCase.name} ${variant.name}: mid-swap had ${mid.onGpuTrees} GPU and ${mid.onTinybvhTrees} tinybvh trees`);
                scene.releaseRefinements();
                frame();
                const last = await check('tinybvh');
                if (last.onGpuTrees > 0) failures.push(`${oracleCase.name} ${variant.name}: ${last.onGpuTrees} GPU trees left after the swap`);
                log(`${oracleCase.name} ${variant.name}: gpu ${first.onGpuTrees}, mid ${mid.onGpuTrees}/${mid.onTinybvhTrees}, swaps ${last.swaps}, groups ${last.mergedGroups}`);
                scene.dispose();
            }
        }
        pool.dispose();
        for (const kernel of Object.values(kernels)) kernel.destroy();
        for (const variant of VARIANTS) {
            if (variant.maxBatchBytes === SMALL_CAP && !cappedCasesWithBatches.has(variant.name)) failures.push(`${variant.name}: no case built in more than one GPU batch`);
        }
        if (failures.length > 0) throw new Error(failures.slice(0, 10).join(' | '));
        return { cases: cases.length, variants: VARIANTS.length, totalRays, ...results };
    },
    { requiredFeatures: ['subgroups'] },
);

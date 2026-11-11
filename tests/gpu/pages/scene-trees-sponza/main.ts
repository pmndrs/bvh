/**
 * SceneTrees on Intel Sponza (manual): memory and timing of the load, for P9.
 *
 * Variants (`?variants=...`; the default runs all six below):
 *
 * - `copy`: `gpuTreeOutputs: 'copy'` (the default, ddgi's path);
 * - `adopt`: the load batch's outputs become the scene's buffers (no copy;
 *   each GPU tree's range freed at its swap);
 * - `copy-merged`: `copy` with `mergeStaticInstances` (every instance static);
 * - `copy-cap512`, `copy-cap256`, `copy-cap128` (any `-cap<MB>` suffix):
 *   `gpuMaxBatchBytes` of that many MiB, the build-peak lever (several
 *   batches one after another on one scratch pool).
 *
 * Each loads the scene with `holdRefinements`, so the GPU trees are in use on
 * their own first, then releases tinybvh's trees. Recorded, from the first update:
 *
 * - `firstTreesMs`: every mesh on a GPU tree (placed by an update);
 * - `tinybvhBuiltMs`: every tinybvh tree built (in the worker pool);
 * - memory with the GPU trees in use (`gpuTreesTotal`), the peak over the
 *   load (`peak`), and settled after every swap and one more update (`settled`);
 * - `gpuBatches` and `gpuBuildMs`: the GPU batches and the sum of their times
 *   from submit to stats (the build plus the readback's latency).
 *
 * Correctness gate, not timing: in the GPU-tree state and settled, the GPU
 * walk must give the hits of the CPU walk over the scene assembled on the
 * CPU from tinybvh trees (20,000 rays from inside the scene's box).
 */

import { SceneTrees, type SceneTreesMemory } from '../../../../src/scene/scene-trees.ts';
import { hplocBatchBytes, hplocScratchPoolBytes, planGpuHplocBatches } from '../../../../src/gpu-hploc.ts';
import { geometryTotals } from '../../../../src/gpu-lbvh.ts';
import { createBvhWorkerPool, defaultWorkerPoolSize } from '../../../../src/worker/pool.ts';
import { assembleTraceScene, type TraceSceneData } from '../../../../src/trace/scene-assembly.ts';
import { buildMeshBvhWithTinyBvh } from '../../../../src/trace/mesh-bvh-topology.ts';
import { createTinyBvhBuilder } from '../../../../src/tinybvh.ts';
import { createCpuTracer } from '../../../../src/trace/cpu-traversal.ts';
import { createRandom, randomRays } from '../../../../src/trace/oracle-fixtures.ts';
import { TraceBatchKernel } from '../../../../src/trace/trace-gpu.ts';
import { drainDevice, reportGpuTest, watchDevice } from '../../client.ts';
import { sameHit, traceOnGpu } from '../../scene-check.ts';
import { loadBenchScene } from '../trace-bench/scene-source.ts';
import { requestBenchDevice } from '../trace-bench/run.ts';

const MB = 1024 * 1024;
const mb = (bytes: number) => Math.round((bytes / MB) * 10) / 10;
const RAYS = 20000;

function summary(memory: SceneTreesMemory) {
    return { total: mb(memory.total), nodes: mb(memory.nodes), nodesInUse: mb(memory.nodesInUse), triangles: mb(memory.triangles), topLevel: mb(memory.topLevel), gpuBuildOutputs: mb(memory.gpuBuildOutputs), retired: mb(memory.retired), peak: mb(memory.peak) };
}

const startTime = performance.now();
const logLines: string[] = [];
const log = (line: string) => {
    logLines.push(line);
    console.log(line);
};
let deviceErrors: string[] = [];
let adapter: GPUAdapter | null = null;
try {
    const requested = await requestBenchDevice();
    adapter = requested.adapter;
    const device = requested.device;
    deviceErrors = watchDevice(device);
    const variants = (new URLSearchParams(location.search).get('variants') ?? 'copy,adopt,copy-merged,copy-cap512,copy-cap256,copy-cap128').split(',');
    const scene = await loadBenchScene('sponza', 'scene');
    log(`sponza: ${scene.meshes.length} meshes, ${scene.instances.length} instances, ${scene.uniqueTriangles} triangles (load ${Math.round(scene.loadMs)} ms)`);
    const kernel = await TraceBatchKernel.create(device);
    const box = scene.geometryBounds;
    const rays = randomRays(createRandom(9), RAYS, [box.min[0], box.min[1], box.min[2], box.max[0], box.max[1], box.max[2]], 0);
    // The reference: tinybvh trees assembled on the CPU (built once, here on the main thread).
    const tinyBvh = await createTinyBvhBuilder();
    let reference: TraceSceneData | null = assembleTraceScene(
        scene.meshes.map((mesh) => buildMeshBvhWithTinyBvh(mesh, tinyBvh)),
        scene.instances.map((instance, index) => ({ meshIndex: instance.meshIndex, instanceIndex: index, objectToWorld: instance.objectToWorld })),
    );
    const referenceTracer = createCpuTracer(reference);
    const expected = rays.map((ray) => referenceTracer.closest(ray));
    const expectedAny = rays.map((ray) => referenceTracer.any(ray));
    reference = null;

    const results: Record<string, unknown> = { meshes: scene.meshes.length, instances: scene.instances.length, triangles: scene.uniqueTriangles, workers: defaultWorkerPoolSize() };
    // The prediction from the code (hplocBatchBytes): every buffer of one uncapped batch, and per cap the batches and the pool.
    const whole = hplocBatchBytes(geometryTotals(scene.meshes));
    const triangles = geometryTotals(scene.meshes).triangles;
    results.predicted = {
        triangles,
        batchMB: mb(whole.total),
        scratchMB: mb(whole.scratch),
        outputsMB: mb(whole.outputs),
        bytesPerTriangle: Object.fromEntries(Object.entries<number>(whole.buffers).map(([name, bytes]) => [name, Math.round((bytes / triangles) * 100) / 100])),
        capped: Object.fromEntries(
            [512, 256, 128].map((capMB) => {
                const plan = planGpuHplocBatches(device, scene.meshes, { maxBatchBytes: capMB * MB });
                const groups = plan.batches.map((batch) => batch.map((index) => scene.meshes[index]));
                const pool = Object.values<number>(hplocScratchPoolBytes(groups)).reduce((sum, bytes) => sum + bytes, 0);
                const largestOutputs = Math.max(...groups.map((group) => hplocBatchBytes(geometryTotals(group)).outputs));
                const largestBatch = Math.max(...groups.map((group) => hplocBatchBytes(geometryTotals(group)).total));
                return [capMB, { batches: plan.batches.length, largestBatchMB: mb(largestBatch), poolMB: mb(pool), poolPlusOutputsMB: mb(pool + largestOutputs) }];
            }),
        ),
    };
    log(`predicted one batch: ${mb(whole.total)} MB (${mb(whole.scratch)} scratch and inputs, ${mb(whole.outputs)} outputs) for ${triangles} triangles`);
    const failures: string[] = [];
    // A first load (`warmup`, `copy`, not reported) compiles every pipeline, so no variant's times include it.
    for (const variant of ['warmup', ...variants]) {
        const pool = createBvhWorkerPool();
        const capMB = /-cap(\d+)/.exec(variant)?.[1];
        const trees = await SceneTrees.create(device, {
            gpuMaxBatchBytes: capMB ? Number(capMB) * MB : undefined,
            label: `sponza ${variant}`,
            holdRefinements: true,
            gpuTreeOutputs: variant.startsWith('adopt') ? 'adopt' : 'copy',
            mergeStaticInstances: variant.includes('merged'),
            workers: pool,
        });
        const meshes = scene.meshes.map((mesh) => trees.addMesh(mesh));
        scene.instances.forEach((instance, index) => trees.addInstance({ mesh: meshes[instance.meshIndex], instanceIndex: index, objectToWorld: instance.objectToWorld, static: true }));
        const frame = () => {
            const encoder = device.createCommandEncoder();
            trees.update(encoder);
            device.queue.submit([encoder.finish()]);
        };
        const check = async (state: string) => {
            await device.queue.onSubmittedWorkDone();
            const gpu = await traceOnGpu(device, kernel, trees.traceBuffers, rays);
            let wrong = 0;
            for (let index = 0; index < rays.length; index++) {
                if (!sameHit(gpu.closest(index), expected[index]) || gpu.any(index) !== expectedAny[index]) wrong++;
            }
            if (wrong > 0) failures.push(`${variant} ${state}: ${wrong} of ${rays.length} rays disagree with the CPU reference`);
            return wrong;
        };
        // One update per frame, as an app would, until the first trees are all in.
        const start = performance.now();
        let refinedAt: number | null = null;
        void trees.whenRefined().then(() => (refinedAt = performance.now() - start));
        frame();
        while (trees.timeline.firstTreesMs === null || trees.building) {
            await new Promise((resolve) => requestAnimationFrame(resolve));
            frame();
        }
        await device.queue.onSubmittedWorkDone();
        const gpuTreesMemory = trees.memory;
        const counts = trees.counts;
        const gpuWrong = await check('GPU trees');
        // Release the swap, one update per frame until every tinybvh tree is placed.
        await trees.whenRefined();
        trees.releaseRefinements();
        frame();
        while (trees.counts.onGpuTrees > 0) {
            await new Promise((resolve) => requestAnimationFrame(resolve));
            frame();
        }
        frame();
        await device.queue.onSubmittedWorkDone();
        frame();
        const settled = trees.memory;
        const settledWrong = await check('settled');
        if (variant !== 'warmup') results[variant] = {
            capMB: capMB ? Number(capMB) : null,
            firstTreesMs: Math.round(trees.timeline.firstTreesMs ?? -1),
            gpuBatches: counts.gpuBatches,
            gpuBuildMs: Math.round(counts.gpuBuildMs),
            tinybvhBuiltMs: Math.round(refinedAt ?? -1),
            gpuTreesOnFirst: counts.onGpuTrees,
            tinybvhOnFirst: counts.onTinybvhTrees,
            mergedGroups: counts.mergedGroups,
            gpuTreesState: summary(gpuTreesMemory),
            settled: summary(settled),
            peakMB: mb(settled.peak),
            disagreements: { gpuTrees: gpuWrong, settled: settledWrong },
        };
        log(`${variant}: first trees ${Math.round(trees.timeline.firstTreesMs ?? -1)} ms, ${counts.gpuBatches} GPU batches ${Math.round(counts.gpuBuildMs)} ms, tinybvh built ${Math.round(refinedAt ?? -1)} ms, GPU-tree state ${mb(gpuTreesMemory.total)} MB, peak ${mb(settled.peak)} MB, settled ${mb(settled.total)} MB`);
        trees.dispose();
        pool.dispose();
        await device.queue.onSubmittedWorkDone();
    }
    kernel.destroy();
    await drainDevice(device);
    reportGpuTest({ status: failures.length ? 'fail' : 'pass', message: failures.join(' | ') || undefined, data: results, adapter, deviceErrors, startTime, logLines });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? `${error.message}` : String(error), adapter, deviceErrors, startTime, logLines });
}

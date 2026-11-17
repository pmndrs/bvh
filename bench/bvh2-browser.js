import { SceneTrees } from '../src/scene/scene-trees.ts';
import { buildMeshBvhFromTopology } from '../src/trace/mesh-bvh-topology.ts';
import { singleLeafTopology } from '../src/topology.ts';
import { TraceBatchKernel } from '../src/trace/trace-gpu.ts';
import { traceOnGpu, readScene } from '../tests/gpu/scene-check.ts';

const WARMUP_CALLS = 300;
const BLOCK_COUNT = 5;
const CALLS_PER_BLOCK = 300;

function statistics(samples) {
    const sorted = [...samples].sort((left, right) => left - right);
    const percentile = (fraction) => sorted[Math.floor((sorted.length - 1) * fraction)];
    return {
        calls: samples.length,
        p50: percentile(0.5),
        p95: percentile(0.95),
        mean: samples.reduce((sum, sample) => sum + sample, 0) / samples.length,
    };
}

/**
 * Real WebGPU CPU host-call benchmark. Serve the repository root with Vite:
 *
 * const { runBvh2BrowserBenchmark } = await import('/bench/bvh2-browser.js');
 * await runBvh2BrowserBenchmark({ smoke: true }); // correctness smoke
 * await runBvh2BrowserBenchmark(); // five blocks of 300 calls per workload
 *
 * Set response headers Cross-Origin-Opener-Policy: same-origin and
 * Cross-Origin-Embedder-Policy: require-corp for approximately 5 us clocks.
 * Without isolation, idle calls may be below the browser's clock precision.
 * Compare immutable revisions under the same otherwise idle conditions.
 *
 * Only scene.update(encoder) is timed. Encoder creation, setPoseCpu,
 * submission, queue completion, and statistics are outside the bracket.
 * Every frame drains the queue. Results do not measure GPU execution time.
 * `device` may be supplied to use an existing subgroup-capable GPUDevice.
 */
export async function runBvh2BrowserBenchmark({ device, smoke = false } = {}) {
    const ownsDevice = !device;
    if (!device) {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter?.features.has('subgroups')) throw new Error('This benchmark requires a WebGPU adapter with subgroups');
        device = await adapter.requestDevice({ requiredFeatures: ['subgroups'] });
    }
    const validationErrors = [];
    const onUncapturedError = (event) => validationErrors.push(event.error.message);
    device.addEventListener('uncapturederror', onUncapturedError);
    try {
        const results = [];
        for (const meshCount of [256, 4096]) {
            for (const sparsePose of [false, true]) {
                const result = await benchmarkWorkload(device, meshCount, sparsePose, smoke);
                if (validationErrors.length > 0) throw new Error(validationErrors.join(' | '));
                results.push(result);
                console.log('SceneTrees host update', result);
            }
        }
        return { crossOriginIsolated, unit: 'milliseconds per update call', warmupCalls: WARMUP_CALLS, results };
    } finally {
        device.removeEventListener('uncapturederror', onUncapturedError);
        if (ownsDevice) device.destroy();
    }
}

async function benchmarkWorkload(device, meshCount, sparsePose, smoke) {
    const scene = await SceneTrees.create(device, {
        mergeStaticInstances: false,
        gpuFirstTrees: false,
        workers: { size: 1 },
    });
    try {
        let targetMesh;
        for (let index = 0; index < meshCount; index++) {
            const geometry = { positions: Float32Array.of(0, 0, 0, 1, 0, 0, 0, 1, 0) };
            const isTarget = sparsePose && index === 0;
            const mesh = scene.addMesh(geometry, isTarget ? { deforming: { bounds: [-1, -1, -1, 2, 2, 2] } } : {});
            if (index === 0) targetMesh = mesh;
            // A CPU pose needs an f32 source layout; SceneTrees places its quantized copy.
            const layout = buildMeshBvhFromTopology(geometry, singleLeafTopology(Uint32Array.of(0), 0, 1), {
                nodeFormat: isTarget ? 'f32' : 'quantized',
            });
            scene.placeTree(mesh, layout);
            scene.addInstance({
                mesh,
                instanceIndex: index,
                objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, (index % 64) * 3, Math.floor(index / 64) * 3, 0, 1],
            });
        }
        const poses = [
            Float32Array.of(0, 0, 0, 1, 0, 0, 0, 1, 0),
            Float32Array.of(0, 0, 0.5, 1, 0, 0.5, 0, 1, 0.5),
        ];
        let poseIndex = 0;
        const applyNextPose = () => {
            if (sparsePose) scene.setPoseCpu(targetMesh, poses[++poseIndex % 2]);
        };
        const frame = () => {
            const encoder = device.createCommandEncoder();
            scene.update(encoder);
            device.queue.submit([encoder.finish()]);
        };
        frame();
        await scene.whenBuilt();
        frame();
        await scene.whenRefined();
        frame();
        await device.queue.onSubmittedWorkDone();
        for (let call = 0; call < WARMUP_CALLS; call++) {
            applyNextPose();
            frame();
            await device.queue.onSubmittedWorkDone();
        }
        const settled = scene.counts;
        if (scene.topLevelPath !== 'gpu' || settled.onAppTrees !== meshCount || settled.waitingForFirstTree || settled.tinybvhBuilding || settled.gpuTreesBuilt || settled.swaps) {
            throw new Error(`Workload is not settled on the GPU top level: ${JSON.stringify(settled)}`);
        }
        const samples = [];
        const blocks = [];
        for (let block = 0; block < (smoke ? 1 : BLOCK_COUNT); block++) {
            const blockSamples = [];
            for (let call = 0; call < (smoke ? 3 : CALLS_PER_BLOCK); call++) {
                applyNextPose();
                const encoder = device.createCommandEncoder();
                const start = performance.now();
                scene.update(encoder);
                const elapsed = performance.now() - start;
                blockSamples.push(elapsed);
                device.queue.submit([encoder.finish()]);
                await device.queue.onSubmittedWorkDone();
            }
            samples.push(...blockSamples);
            blocks.push(statistics(blockSamples));
        }
        if (sparsePose) {
            scene.setPoseCpu(targetMesh, poses[1]);
            frame();
            await device.queue.onSubmittedWorkDone();
        }
        const correctness = await checkWorkload(device, scene, meshCount, sparsePose);
        return { meshCount, sparsePose, pooled: statistics(samples), blocks, settled, correctness };
    } finally {
        scene.dispose();
    }
}

async function checkWorkload(device, scene, meshCount, sparsePose) {
    const kernel = await TraceBatchKernel.create(device);
    let targetHit;
    let neighborHit;
    try {
        const hits = await traceOnGpu(device, kernel, scene.traceBuffers, [
            { origin: [0.2, 0.2, 2], direction: [0, 0, -1], tMin: 0, tMax: 10 },
            { origin: [3.2, 0.2, 2], direction: [0, 0, -1], tMin: 0, tMax: 10 },
        ]);
        targetHit = hits.closest(0);
        neighborHit = hits.closest(1);
        const expectedDistance = sparsePose ? 1.5 : 2;
        if (!targetHit || targetHit.instanceIndex !== 0 || Math.abs(targetHit.t - expectedDistance) > 1e-5) {
            throw new Error(`Wrong target hit: ${JSON.stringify(targetHit)}`);
        }
        if (!neighborHit || neighborHit.instanceIndex !== 1 || Math.abs(neighborHit.t - 2) > 1e-5) {
            throw new Error(`Wrong static neighbor hit: ${JSON.stringify(neighborHit)}`);
        }
    } finally {
        kernel.destroy();
    }
    const readback = await readScene(device, scene.buffers, { nodeFormat: scene.nodeFormat });
    if (readback.topLevel.instanceCount !== meshCount) throw new Error('Wrong top-level instance count');
    return { targetHit, neighborHit, instanceCount: readback.topLevel.instanceCount, counts: scene.counts, deforming: scene.deformingCounts, topLevelPath: scene.topLevelPath };
}

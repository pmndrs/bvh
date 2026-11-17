/** Reproducible host-side SceneTrees.update benchmark; no WebGPU adapter required.
 * Run with: node --experimental-transform-types --no-warnings bench/bvh2-bookkeeping.ts
 */
import { performance } from 'node:perf_hooks';
import { createFakeGpu, type FakeBuffer } from '../tests/support/fake-gpu.ts';
import { buildMeshBvhAsync } from '../src/trace/mesh-bvh-build.ts';
import { SceneTrees } from '../src/scene/scene-trees.ts';

const GEOMETRY = { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]) };
// One vertex moves inside the fixed triangle bounds; the other pose translates the triangle in z.
const SAME_AABB_POSES = [0.125, 0.25].map((x) => new Float32Array([0, 0, 0, 1, 0, 0, x, 1, 0]));
const TRANSLATED_POSES = [0.125, -0.125].map((z) => new Float32Array([0, 0, z, 1, 0, z, 0, 1, z]));
const IDENTITY = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
const REPEATS = 11;
const WARMUP = 30;

function percentile(values: number[], p: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.ceil(p * sorted.length) - 1)];
}

async function fixture(count: number, poses: boolean) {
    const gpu = createFakeGpu();
    const scene = await SceneTrees.create(gpu.device, { gpuFirstTrees: false, topLevel: 'cpu', mergeStaticInstances: false, refine: false });
    const tree = await buildMeshBvhAsync(GEOMETRY);
    let poseMesh = -1;
    for (let i = 0; i < count; i++) {
        const mesh = scene.addMesh(GEOMETRY, i === 0 && poses ? { deforming: true } : {});
        scene.placeTree(mesh, tree);
        if (i === 0) poseMesh = mesh;
        const transform = IDENTITY.slice();
        transform[12] = i * 2;
        scene.addInstance({ mesh, objectToWorld: transform });
    }
    const update = () => {
        const encoder = gpu.createEncoder();
        scene.update(encoder);
        gpu.submit(encoder);
    };
    update(); // place all app-supplied trees and build the initial CPU top level
    return { scene, update, poseMesh };
}

async function measure(count: number, scenario: 'settled' | 'pose-same-aabb' | 'pose-translated') {
    const hasPose = scenario !== 'settled';
    const f = await fixture(count, hasPose);
    let pose = 0;
    const topBefore = (f.scene.buffers.topLevel as unknown as FakeBuffer).bytes.slice();
    const step = () => {
        if (hasPose) {
            pose ^= 1;
            const positions = scenario === 'pose-same-aabb' ? SAME_AABB_POSES[pose] : TRANSLATED_POSES[pose];
            f.scene.setPoseCpu(f.poseMesh, positions);
            f.update();
            return;
        }
        f.update();
    };
    for (let i = 0; i < WARMUP; i++) step();
    const updatesPerBlock = hasPose ? 5 : 100;
    const poseCountBefore = f.scene.deformingCounts.cpuPoses;
    const blocks: number[] = [];
    for (let repeat = 0; repeat < REPEATS; repeat++) {
        const start = performance.now();
        for (let i = 0; i < updatesPerBlock; i++) step();
        blocks.push((performance.now() - start) * 1000 / updatesPerBlock);
    }
    const poseCountAfter = f.scene.deformingCounts.cpuPoses;
    if (hasPose && poseCountAfter - poseCountBefore !== REPEATS * updatesPerBlock) throw new Error('CPU pose count did not match updates');
    const topAfter = (f.scene.buffers.topLevel as unknown as FakeBuffer).bytes;
    const topLevelChanged = topBefore.some((value, index) => value !== topAfter[index]);
    if (scenario === 'pose-same-aabb' && topLevelChanged) throw new Error('same-AABB pose unexpectedly changed the CPU top-level bytes');
    if (scenario === 'pose-translated' && !topLevelChanged) throw new Error('translated pose did not change the CPU top-level bytes');
    f.scene.dispose();
    return { p50_us: +percentile(blocks, 0.5).toFixed(2), p95_us: +percentile(blocks, 0.95).toFixed(2), topLevelChanged, poseWrites: hasPose ? poseCountAfter - poseCountBefore : 0, blocks_us: blocks.map((x) => +x.toFixed(2)) };
}

for (const count of [256, 4096]) {
    for (const scenario of ['settled', 'pose-same-aabb', 'pose-translated'] as const) {
        console.log(JSON.stringify({ meshes: count, instances: count, scenario, warmup: WARMUP, updatesPerBlock: scenario === 'settled' ? 100 : 5, repeats: REPEATS, ...(await measure(count, scenario)) }));
    }
}

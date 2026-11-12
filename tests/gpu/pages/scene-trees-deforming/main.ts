/**
 * Deforming meshes in `SceneTrees`: GPU refits from poses, on the GPU tree and
 * then on tinybvh's, and a drift rebuild.
 *
 * A 40 x 40 cloth (a deforming mesh, 16-bit indices, its vertices in a GPU
 * buffer interleaved with other data) and a static rock, both instanced
 * (the cloth twice, once mirrored), with `mergeStaticInstances` on, so the
 * rock merges and the cloth (deforming) must not. Each checked pose, the
 * vertices are written, `setPose` is called, and after the update the GPU walk
 * and the CPU walk over the read-back buffers must give the hits of the scene
 * assembled on the CPU from tinybvh trees of the deformed cloth:
 *
 * - poses 1-3 on the cloth's GPU H-PLOC tree (its layout read back for the
 *   refit), with tinybvh's held;
 * - poses 4-6 after the swap, on tinybvh's tree;
 * - then a bend held still for 150 updates: the refit's cost passes 1.1x the
 *   built tree's and holds, so the cloth is rebuilt on the GPU from its pose
 *   (a drift rebuild), and the rebuilt tree is checked too;
 * - then poses on the CPU (`setPoseCpu`), on the rebuilt GPU tree: refit on the
 *   CPU and copied in, the GPU pose they replace no longer refit.
 *
 * The cloth has per-triangle materials (every fourth triangle without one,
 * taking its instance's), so every check compares materials too, through the
 * refits and the drift rebuild (whose flags come from a buffer of their own).
 */

import { SceneTrees } from '../../../../src/scene/scene-trees.ts';
import { DRIFT_HOLD_FRAMES } from '../../../../src/scene/deforming-trees.ts';
import { assembleTraceScene, type TraceSceneInstance } from '../../../../src/trace/scene-assembly.ts';
import { buildMeshBvhWithTinyBvh } from '../../../../src/trace/mesh-bvh-topology.ts';
import { createTinyBvhBuilder } from '../../../../src/tinybvh.ts';
import { createRandom, randomRays } from '../../../../src/trace/oracle-fixtures.ts';
import { TraceBatchKernel } from '../../../../src/trace/trace-gpu.ts';
import type { MeshGeometry } from '../../../../src/trace/mesh-bvh.ts';
import { assert, runGpuTest } from '../../client.ts';
import { compareWithReference, readScene, traceOnGpu } from '../../scene-check.ts';

const CELLS = 40;
const STRIDE = 5;
const OFFSET = 1;

function cloth(): { positions: Float32Array; indices: Uint16Array; triangleMaterials: Uint16Array } {
    const positions = new Float32Array((CELLS + 1) * (CELLS + 1) * 3);
    for (let row = 0; row <= CELLS; row++) {
        for (let column = 0; column <= CELLS; column++) positions.set([column / CELLS * 4 - 2, 0, row / CELLS * 4 - 2], (row * (CELLS + 1) + column) * 3);
    }
    const indices = new Uint16Array(CELLS * CELLS * 6);
    for (let row = 0; row < CELLS; row++) {
        for (let column = 0; column < CELLS; column++) {
            const corner = row * (CELLS + 1) + column;
            indices.set([corner, corner + 1, corner + CELLS + 2, corner, corner + CELLS + 2, corner + CELLS + 1], (row * CELLS + column) * 6);
        }
    }
    const triangleMaterials = Uint16Array.from({ length: CELLS * CELLS * 2 }, (_, triangle) => (triangle % 4 === 3 ? 0xffff : triangle % 7));
    return { positions, indices, triangleMaterials };
}

function pose(rest: Float32Array, wave: number, bend: number): Float32Array {
    const moved = new Float32Array(rest.length);
    for (let at = 0; at < rest.length; at += 3) {
        const x = rest[at];
        const z = rest[at + 2];
        const y = Math.sin(x * 2 + wave) * Math.cos(z * 1.5 - wave) * 0.4 + bend * x * x;
        moved.set([x, y, z + bend * Math.sin(x)], at);
    }
    return moved;
}

await runGpuTest(
    async ({ device, log }) => {
        assert(SceneTrees && device.features.has('subgroups'), 'deforming meshes need subgroups');
        const tinyBvh = await createTinyBvhBuilder();
        const failures: string[] = [];
        const results: Record<string, unknown> = {};
        // Records, then indexed leaves (the refit copies each pose into the scene's positions).
        // Then quantized nodes (docs/internals/design/compressed-nodes.md): the refit keeps an f32 copy and packs it into the scene.
        for (const [triangleLeaves, nodeFormat] of [['records', 'f32'], ['indexed', 'f32'], ['records', 'quantized'], ['indexed', 'quantized']] as const) {
            const label = nodeFormat === 'quantized' ? `quantized ${triangleLeaves}` : triangleLeaves;
            const kernel = await TraceBatchKernel.create(device, { triangleLeaves, nodeFormat });
            const rest = cloth();
            const rock: MeshGeometry = { positions: new Float32Array([-1, -1, -1, 1, -1, -1, 0, 1, -1, 0, 0, 1]), indices: new Uint32Array([0, 1, 2, 0, 1, 3, 1, 2, 3, 0, 2, 3]) };
            const scene = await SceneTrees.create(device, { label: `deforming ${label}`, holdRefinements: true, mergeStaticInstances: true, workers: { size: 2 }, triangleLeaves, nodeFormat });
            const clothMesh = scene.addMesh(rest, { deforming: { bounds: [-3, -3, -3, 3, 3, 3] } });
            const rockMesh = scene.addMesh(rock);
            const instances: (TraceSceneInstance & { readonly mesh: number })[] = [
                { mesh: clothMesh, meshIndex: 0, instanceIndex: 10, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], static: true, material: 20 },
                { mesh: clothMesh, meshIndex: 0, instanceIndex: 11, objectToWorld: [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 1, 0, 1] },
                { mesh: rockMesh, meshIndex: 1, instanceIndex: 12, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 3, 0, 1], static: true, material: 3 },
            ];
            for (const instance of instances) scene.addInstance(instance);

            // The cloth's vertices on the GPU, interleaved; indices as 16-bit pairs in a u32 buffer.
            const vertexCount = rest.positions.length / 3;
            const vertexBuffer = device.createBuffer({ label: 'cloth vertices', size: vertexCount * STRIDE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
            const indexBuffer = device.createBuffer({ label: 'cloth indices', size: Math.ceil(rest.indices.byteLength / 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
            const paddedIndices = new Uint16Array(indexBuffer.size / 2);
            paddedIndices.set(rest.indices);
            device.queue.writeBuffer(indexBuffer, 0, paddedIndices);
            const setPose = (positions: Float32Array) => {
                const interleaved = new Float32Array(vertexCount * STRIDE);
                for (let vertex = 0; vertex < vertexCount; vertex++) interleaved.set(positions.subarray(vertex * 3, vertex * 3 + 3), vertex * STRIDE + OFFSET);
                device.queue.writeBuffer(vertexBuffer, 0, interleaved);
                scene.setPose(clothMesh, { positionBuffer: vertexBuffer, vertexCount, positionStrideFloats: STRIDE, positionOffsetFloats: OFFSET, indexBuffer, indexFormat: 'uint16' });
            };
            const frame = () => {
                const encoder = device.createCommandEncoder();
                scene.update(encoder);
                device.queue.submit([encoder.finish()]);
            };
            const random = createRandom(4);
            const rays = randomRays(random, 30000, [-3, -3, -3, 8, 5, 3], 1);
            const check = async (state: string, positions: Float32Array) => {
                const name = `${label} ${state}`;
                await device.queue.onSubmittedWorkDone();
                const reference = assembleTraceScene(
                    [buildMeshBvhWithTinyBvh({ positions, indices: rest.indices, triangleMaterials: rest.triangleMaterials }, tinyBvh), buildMeshBvhWithTinyBvh(rock, tinyBvh)],
                    instances.map(({ mesh: _mesh, ...instance }) => instance),
                );
                const gpu = await traceOnGpu(device, kernel, scene.traceBuffers, rays);
                const comparison = compareWithReference(rays, gpu, await readScene(device, scene.buffers, { triangleLeaves: scene.triangleLeaves, positionLayout: scene.positionLayout, nodeFormat: scene.nodeFormat }), reference);
                results[name] = { disagreements: comparison.count, hits: comparison.hits, builtBy: scene.meshStatus(clothMesh).builtBy };
                log(`${name}: ${comparison.hits} hits, ${comparison.count} disagreements, cloth on ${scene.meshStatus(clothMesh).builtBy}`);
                if (comparison.count > 0) failures.push(`${name}: ${comparison.count} disagreements, first ${JSON.stringify(comparison.disagreements.slice(0, 2))}`);
            };

            frame();
            await scene.whenBuilt();
            frame();
            assert(scene.counts.mergedGroups === 1 && scene.meshStatus(clothMesh).mergedInto === null, 'the rock should merge and the cloth not');
            assert(scene.meshStatus(clothMesh).builtBy === 'gpu', `the cloth's first tree came from ${scene.meshStatus(clothMesh).builtBy}`);
            for (let step = 1; step <= 3; step++) {
                const positions = pose(rest.positions, step * 0.7, 0);
                setPose(positions);
                frame();
                await check(`gpu tree pose ${step}`, positions);
            }
            await scene.whenRefined();
            scene.releaseRefinements();
            let positions = pose(rest.positions, 2.1, 0);
            setPose(positions);
            frame();
            assert(scene.meshStatus(clothMesh).builtBy === 'tinybvh', 'the cloth did not swap to tinybvh');
            await check('tinybvh tree after swap', positions);
            for (let step = 4; step <= 6; step++) {
                positions = pose(rest.positions, step * 0.7, 0);
                setPose(positions);
                frame();
                await check(`tinybvh tree pose ${step}`, positions);
            }

            // A bend held still: the refit drifts and holds, so the cloth is rebuilt from its pose.
            positions = pose(rest.positions, 0, 0.6);
            let updates = 0;
            const rebuildsBefore = scene.deformingCounts.driftRebuilds;
            while (scene.deformingCounts.driftRebuilds === rebuildsBefore && updates < DRIFT_HOLD_FRAMES * 4) {
                // A pose every update (the same one), as an animation holding still would give.
                setPose(positions);
                frame();
                await device.queue.onSubmittedWorkDone();
                await new Promise((resolve) => setTimeout(resolve, 1));
                updates++;
            }
            const drift = scene.driftRatio(clothMesh);
            assert(scene.deformingCounts.driftRebuilds > rebuildsBefore, `no drift rebuild after ${updates} updates (drift ratio ${drift})`);
            // Let the rebuild arrive and be placed.
            for (let wait = 0; wait < 20 && scene.meshStatus(clothMesh).builtBy !== 'gpu'; wait++) {
                setPose(positions);
                frame();
                await device.queue.onSubmittedWorkDone();
                await new Promise((resolve) => setTimeout(resolve, 5));
            }
            assert(scene.meshStatus(clothMesh).builtBy === 'gpu', 'the drift rebuild was not placed');
            await check('after the drift rebuild', positions);

            // Poses on the CPU, on the rebuilt GPU tree (its layout was read back): no GPU refit, the top-level box follows.
            const refitsBefore = scene.deformingCounts.refits;
            for (let step = 7; step <= 8; step++) {
                positions = pose(rest.positions, step * 0.7, 0.2);
                scene.setPoseCpu(clothMesh, positions);
                frame();
                await check(`cpu pose ${step}`, positions);
            }
            assert(scene.deformingCounts.cpuPoses === 2, `${scene.deformingCounts.cpuPoses} CPU poses written, expected 2`);
            assert(scene.deformingCounts.refits === refitsBefore, 'a GPU refit ran after a CPU pose replaced the GPU one');

            const counts = scene.deformingCounts;
            results[`${label} refits`] = counts.refits;
            results[`${label} driftRebuilds`] = counts.driftRebuilds;
            results[`${label} updatesToRebuild`] = updates;
            results[`${label} driftRatioAtRebuild`] = drift;
            results[`${label} cpuPoses`] = counts.cpuPoses;
            scene.dispose();
            kernel.destroy();
            vertexBuffer.destroy();
            indexBuffer.destroy();
        }
        if (failures.length > 0) throw new Error(failures.join(' | '));
        return results;
    },
    { requiredFeatures: ['subgroups'] },
);

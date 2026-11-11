/**
 * GpuTopLevel on the GPU: the gather and the refit against the CPU top level.
 *
 * Four tinybvh meshes, 400 instances (random rotations, non-uniform scales,
 * mirrors, masks, materials, some hidden or not contributing). Every check
 * frame:
 *
 * - the gathered records, read back, must equal `buildTopLevel`'s bit for bit
 *   for every traced instance (world-to-object rows, offsets, index, flags
 *   bits 0-15), with bits 16-31 holding material + 1;
 * - the GPU walk (`TraceBatchKernel`) over the GPU top level, and the CPU walk
 *   over the read-back buffers, must give the hits of the CPU walk over the
 *   scene assembled on the CPU, for a full ray mask and a partial one.
 *
 * Then instances move, hide and go away: checked after the refit alone (the
 * removed ones are empty records until the next shape) and after the new shape.
 */

import { GpuTopLevel, type GpuTopLevelInstance } from '../../../../src/trace/top-level-gpu.ts';
import { assembleTraceScene, type TraceSceneData, type TraceSceneInstance } from '../../../../src/trace/scene-assembly.ts';
import { buildMeshBvhWithTinyBvh } from '../../../../src/trace/mesh-bvh-topology.ts';
import { createTinyBvhBuilder } from '../../../../src/tinybvh.ts';
import { createRandom, randomAffine, randomRays, randomTriangleSoup, tessellatedQuad } from '../../../../src/trace/oracle-fixtures.ts';
import { INSTANCE_RECORD_WORDS, INSTANCE_WORD, TOP_LEVEL_HEADER_WORDS } from '../../../../src/trace/layout.ts';
import { TraceBatchKernel, type TraceSceneBuffers } from '../../../../src/trace/trace-gpu.ts';
import { assert, runGpuTest } from '../../client.ts';
import { compareWithReference, readBuffer, readScene, traceOnGpu } from '../../scene-check.ts';

const INSTANCES = 400;
const RAYS = 20000;

await runGpuTest(
    async ({ device, log }) => {
        assert(GpuTopLevel.isSupported(device), 'the GPU top level needs subgroups');
        const random = createRandom(2026);
        const tinyBvh = await createTinyBvhBuilder();
        const geometries = [tessellatedQuad(12, 1), randomTriangleSoup(random, 200, 1, 0.3), randomTriangleSoup(random, 50, 0.5, 0.4), tessellatedQuad(3, 0.7)];
        const meshes = geometries.map((geometry) => buildMeshBvhWithTinyBvh(geometry, tinyBvh));
        const placed = assembleTraceScene(meshes, []);
        const storage = (name: string, data: ArrayBuffer) => {
            const buffer = device.createBuffer({ label: name, size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
            device.queue.writeBuffer(buffer, 0, data);
            return buffer;
        };
        const nodes = storage('nodes', placed.nodes);
        const triangles = storage('triangles', placed.triangles);

        const top = new GpuTopLevel(device, { label: 'page', diagnostics: true });
        const kernel = await TraceBatchKernel.create(device);
        await top.prepare();
        meshes.forEach((mesh, index) => top.setMeshTree(index, { nodeOffset: placed.meshes[index].nodeOffset, triangleOffset: placed.meshes[index].triangleOffset, bounds: mesh.bounds }));

        const instances = new Map<number, GpuTopLevelInstance>();
        for (let slot = 0; slot < INSTANCES; slot++) {
            const instance: GpuTopLevelInstance = {
                meshSlot: slot % meshes.length,
                instanceIndex: 1000 + slot,
                objectToWorld: randomAffine(random, { extent: 20, mirror: slot % 5 === 0 }),
                visible: slot % 11 !== 3,
                contributesToGi: slot % 13 !== 4,
                mask: slot % 3 === 0 ? 0x01 : slot % 3 === 1 ? 0x0e : undefined,
                material: slot % 4 === 0 ? undefined : slot * 7,
            };
            instances.set(slot, instance);
            top.setInstance(slot, instance);
        }

        const frame = () => {
            const encoder = device.createCommandEncoder({ label: 'page frame' });
            top.update(encoder, { deepestMesh: placed.maxMeshDepth || Math.max(...meshes.map((mesh) => mesh.maxDepth)) });
            device.queue.submit([encoder.finish()]);
        };
        const reference = (): TraceSceneData =>
            assembleTraceScene(
                meshes,
                [...instances.values()].map(
                    (instance): TraceSceneInstance => ({
                        meshIndex: instance.meshSlot,
                        instanceIndex: instance.instanceIndex,
                        objectToWorld: instance.objectToWorld,
                        visible: instance.visible,
                        contributesToGi: instance.contributesToGi,
                        mask: instance.mask,
                    }),
                ),
            );
        const rays = randomRays(random, RAYS, [-25, -25, -25, 25, 25, 25], 2);

        const failures: string[] = [];
        const results: Record<string, unknown> = {};
        const check = async (stage: string, checkRecords: boolean) => {
            await device.queue.onSubmittedWorkDone();
            const expected = reference();
            const gpuScene = await readScene(device, { nodes, triangles, topLevel: top.buffer });
            if (checkRecords) {
                // Records by instance index: the CPU top level's (only traced instances) against the gathered ones.
                const recordsOf = (buffer: ArrayBuffer, count: number) => {
                    const words = new Uint32Array(buffer);
                    const byIndex = new Map<number, Uint32Array>();
                    for (let record = 0; record < count; record++) {
                        const base = TOP_LEVEL_HEADER_WORDS + record * INSTANCE_RECORD_WORDS;
                        const entry = words.subarray(base, base + INSTANCE_RECORD_WORDS);
                        if (entry[INSTANCE_WORD.flags] !== 0) byIndex.set(entry[INSTANCE_WORD.instanceIndex], entry);
                    }
                    return byIndex;
                };
                const cpuRecords = recordsOf(expected.topLevel.buffer, expected.topLevel.instanceCount);
                const gpuRecords = recordsOf(gpuScene.topLevel.buffer, gpuScene.topLevel.instanceCount);
                if (cpuRecords.size !== gpuRecords.size) failures.push(`${stage}: ${gpuRecords.size} traced records on the GPU, ${cpuRecords.size} on the CPU`);
                const materialOf = new Map([...instances.values()].map((instance) => [instance.instanceIndex, instance.material]));
                let recordProblems = 0;
                for (const [instanceIndex, cpu] of cpuRecords) {
                    const gpu = gpuRecords.get(instanceIndex);
                    const material = materialOf.get(instanceIndex);
                    const ok =
                        gpu !== undefined &&
                        gpu.subarray(0, 15).every((word, k) => word === cpu[k]) &&
                        (gpu[INSTANCE_WORD.flags] & 0xffff) === cpu[INSTANCE_WORD.flags] &&
                        gpu[INSTANCE_WORD.flags] >>> 16 === (material === undefined ? 0 : material + 1);
                    if (!ok && recordProblems++ < 4) failures.push(`${stage}: instance ${instanceIndex}'s record differs: GPU ${gpu ? Array.from(gpu).join(',') : 'none'} CPU ${Array.from(cpu).join(',')}`);
                }
            }
            const buffers: TraceSceneBuffers = { nodes, triangles, topLevel: top.buffer, byteLength: 0, destroy() {} };
            for (const mask of [undefined, 0x0e]) {
                const gpu = await traceOnGpu(device, kernel, buffers, rays, mask);
                const comparison = compareWithReference(rays, gpu, gpuScene, expected, { mask });
                const name = `${stage} mask ${mask === undefined ? 'all' : mask.toString(16)}`;
                results[name] = { disagreements: comparison.count, hits: comparison.hits };
                log(`${name}: ${comparison.hits} hits, ${comparison.count} disagreements`);
                if (comparison.count > 0) failures.push(`${name}: ${comparison.count} disagreements, first ${JSON.stringify(comparison.disagreements.slice(0, 3))}`);
                if (comparison.hits < RAYS / 20) failures.push(`${name}: only ${comparison.hits} hits; the scene is too sparse to test anything`);
            }
        };

        frame();
        await top.whenIdle();
        frame();
        assert(top.instanceCount === INSTANCES, `the shape holds ${top.instanceCount} records, expected ${INSTANCES}`);
        await check('built', true);

        // Moves and hides: a refit on the same shape.
        for (let slot = 0; slot < INSTANCES; slot += 4) {
            const moved = { ...instances.get(slot)!, objectToWorld: randomAffine(random, { extent: 20, mirror: slot % 8 === 0 }) };
            instances.set(slot, moved);
            top.setTransform(slot, moved.objectToWorld);
        }
        for (let slot = 1; slot < INSTANCES; slot += 9) {
            const hidden = { ...instances.get(slot)!, visible: false };
            instances.set(slot, hidden);
            top.setInstance(slot, hidden);
        }
        frame();
        const shapesBefore = top.counters.shapesBuilt;
        await check('moved', true);
        assert(top.counters.shapesBuilt === shapesBefore, 'a move built a new shape');

        // Removals: empty records at once (refit only), then a new shape.
        for (let slot = 2; slot < INSTANCES; slot += 7) {
            instances.delete(slot);
            top.setInstance(slot, null);
        }
        frame();
        await check('removed, old shape', false);
        await top.whenIdle();
        frame();
        assert(top.instanceCount === instances.size, `the new shape holds ${top.instanceCount} records, expected ${instances.size}`);
        await check('removed, new shape', true);

        const topLevelBytes = (await readBuffer(device, top.buffer)).byteLength;
        top.dispose();
        kernel.destroy();
        nodes.destroy();
        triangles.destroy();
        if (failures.length > 0) throw new Error(failures.slice(0, 8).join(' | '));
        return { ...results, shapesBuilt: top.counters.shapesBuilt, refits: top.counters.refits, refitGaveUp: top.counters.refitGaveUp, topLevelBytes };
    },
    { requiredFeatures: ['subgroups'] },
);

/**
 * SceneTrees at run time: meshes and instances added, moved, hidden,
 * re-masked, re-materialed and removed while the scene runs, on the GPU top
 * level and GPU first trees, with merged static instances from the load.
 *
 * The load: the trace-oracle's random instanced case, its instances static
 * (so the ones whose mesh is used once merge) plus a few movable ones. Then
 * eight rounds, each a mix of: new meshes with instances (built on the GPU at
 * run time, then swapped to tinybvh), moves, hides, mask and material
 * changes, instance removals, mesh removals (one while its builds still run,
 * which cancels its tinybvh build). tinybvh's trees are held each round, so
 * the new meshes are first traced on their run-time GPU trees: then the GPU
 * walk must agree with the CPU-assembled scene; and again after the swap. Merged instances must refuse to move, hide or go.
 */

import { SceneTrees, type SceneInstanceDescriptor } from '../../../../src/scene/scene-trees.ts';
import { assembleTraceScene, type TraceSceneInstance } from '../../../../src/trace/scene-assembly.ts';
import { buildMeshBvhWithTinyBvh } from '../../../../src/trace/mesh-bvh-topology.ts';
import { createTinyBvhBuilder } from '../../../../src/tinybvh.ts';
import { createRandom, randomAffine, randomRays, randomTriangleSoup, tessellatedQuad } from '../../../../src/trace/oracle-fixtures.ts';
import { TraceBatchKernel } from '../../../../src/trace/trace-gpu.ts';
import type { MeshGeometry } from '../../../../src/trace/mesh-bvh.ts';
import { assert, runGpuTest } from '../../client.ts';
import { compareWithReference, readScene, traceOnGpu } from '../../scene-check.ts';

interface LiveInstance extends SceneInstanceDescriptor {
    readonly handle: number;
    readonly geometry: MeshGeometry;
    readonly instanceIndex: number;
}

await runGpuTest(
    async ({ device, log }) => {
        assert(device.features.has('subgroups'), 'this page tests the GPU paths');
        const random = createRandom(31);
        const tinyBvh = await createTinyBvhBuilder();
        const kernel = await TraceBatchKernel.create(device);
        const scene = await SceneTrees.create(device, { label: 'runtime', mergeStaticInstances: true, holdRefinements: true, workers: { size: 3 } });
        assert(scene.topLevelPath === 'gpu' && scene.gpuFirstTrees.enabled, 'expected the GPU top level and GPU first trees');
        const meshGeometry = new Map<number, MeshGeometry>();
        const live = new Map<number, LiveInstance>();
        let nextIndex = 0;
        const addMesh = (geometry: MeshGeometry) => {
            const mesh = scene.addMesh(geometry);
            meshGeometry.set(mesh, geometry);
            return mesh;
        };
        const addInstance = (mesh: number, extra: Partial<SceneInstanceDescriptor> = {}) => {
            const descriptor = { mesh, objectToWorld: randomAffine(random, { extent: 12, mirror: random() < 0.3 }), instanceIndex: nextIndex++, ...extra };
            const handle = scene.addInstance(descriptor);
            live.set(handle, { ...descriptor, handle, geometry: meshGeometry.get(mesh)!, instanceIndex: descriptor.instanceIndex });
            return handle;
        };
        const newGeometry = () => (random() < 0.5 ? randomTriangleSoup(random, 50 + Math.floor(random() * 300), 1.5, 0.4) : tessellatedQuad(2 + Math.floor(random() * 10), 1 + random()));

        // The load: 30 meshes, used once (static, so they merge) or twice (instanced), and 5 movable instances.
        const merged = new Set<number>();
        for (let index = 0; index < 30; index++) {
            const mesh = addMesh(newGeometry());
            const first = addInstance(mesh, { static: true, material: index % 4 === 0 ? undefined : index });
            if (index % 5 === 0) addInstance(mesh, { static: true });
            else merged.add(first);
        }
        for (let index = 0; index < 5; index++) addInstance(addMesh(newGeometry()), { mask: 0x3 });

        const frame = () => {
            const encoder = device.createCommandEncoder();
            scene.update(encoder);
            device.queue.submit([encoder.finish()]);
        };
        const rays = randomRays(random, 20000, [-14, -14, -14, 14, 14, 14], 0);
        const failures: string[] = [];
        const results: Record<string, unknown> = {};
        /** The CPU scene of the instances whose mesh has a tree now (or every instance, `settled`). */
        const reference = (onlyPlaced: boolean) => {
            const meshes: MeshGeometry[] = [];
            const meshIndex = new Map<MeshGeometry, number>();
            const instances: TraceSceneInstance[] = [];
            for (const instance of live.values()) {
                if (onlyPlaced && !merged.has(instance.handle) && scene.meshStatus(instance.mesh).builtBy === null) continue;
                if (!meshIndex.has(instance.geometry)) {
                    meshIndex.set(instance.geometry, meshes.length);
                    meshes.push(instance.geometry);
                }
                instances.push({ meshIndex: meshIndex.get(instance.geometry)!, instanceIndex: instance.instanceIndex, objectToWorld: instance.objectToWorld, visible: instance.visible, mask: instance.mask, material: instance.material });
            }
            return assembleTraceScene(meshes.map((mesh) => buildMeshBvhWithTinyBvh(mesh, tinyBvh)), instances);
        };
        const check = async (name: string, onlyPlaced: boolean) => {
            await device.queue.onSubmittedWorkDone();
            const gpu = await traceOnGpu(device, kernel, scene.traceBuffers, rays);
            const comparison = compareWithReference(rays, gpu, await readScene(device, scene.buffers), reference(onlyPlaced));
            results[name] = { disagreements: comparison.count, hits: comparison.hits, gpuTrees: scene.counts.onGpuTrees, tinybvhTrees: scene.counts.onTinybvhTrees, instances: live.size };
            log(`${name}: ${comparison.hits} hits, ${comparison.count} disagreements (${scene.counts.onGpuTrees} GPU / ${scene.counts.onTinybvhTrees} tinybvh trees)`);
            if (comparison.count > 0) failures.push(`${name}: ${comparison.count} disagreements, first ${JSON.stringify(comparison.disagreements.slice(0, 2))}`);
        };
        const settle = async () => {
            await scene.whenRefined();
            scene.setHoldRefinements(false);
            frame();
            scene.setHoldRefinements(true);
            frame();
            await scene.whenBuilt();
            frame();
        };

        frame();
        await settle();
        assert(scene.counts.mergedGroups >= 1, 'nothing merged at load');
        await check('load', false);
        const mergedHandle = [...merged][0];
        for (const attempt of [() => scene.setTransform(mergedHandle, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1]), () => scene.setInstance(mergedHandle, { visible: false }), () => scene.removeInstance(mergedHandle)]) {
            let threw = false;
            try {
                attempt();
            } catch {
                threw = true;
            }
            if (!threw) failures.push('a merged instance accepted a change');
        }

        for (let round = 1; round <= 8; round++) {
            const movable = [...live.values()].filter((instance) => !merged.has(instance.handle));
            // New meshes and instances.
            for (let added = 0; added < 3; added++) {
                const mesh = addMesh(newGeometry());
                addInstance(mesh, { material: round });
                if (added === 0) addInstance(mesh, { mask: 0x1 });
            }
            // Moves, hides, masks, materials.
            for (const instance of movable.slice(0, 6)) {
                const objectToWorld = randomAffine(random, { extent: 12 });
                scene.setTransform(instance.handle, objectToWorld);
                live.set(instance.handle, { ...instance, objectToWorld });
            }
            const toggled = movable[round % movable.length];
            if (toggled) {
                const change = { visible: round % 2 === 0, mask: round % 3 === 0 ? 0x2 : 0xff, material: round % 2 === 0 ? round * 10 : undefined };
                scene.setInstance(toggled.handle, { ...change, material: change.material ?? null });
                live.set(toggled.handle, { ...live.get(toggled.handle)!, ...change });
            }
            // Removals: an instance, and a mesh with all its instances.
            const removedInstance = movable[(round * 3) % movable.length];
            if (removedInstance && live.has(removedInstance.handle)) {
                scene.removeInstance(removedInstance.handle);
                live.delete(removedInstance.handle);
            }
            const doomed = [...meshGeometry.keys()].find((mesh) => [...live.values()].every((instance) => instance.mesh !== mesh || !merged.has(instance.handle)) && round % 2 === 1 && [...live.values()].some((instance) => instance.mesh === mesh));
            if (doomed !== undefined) {
                for (const instance of [...live.values()]) {
                    if (instance.mesh !== doomed) continue;
                    scene.removeInstance(instance.handle);
                    live.delete(instance.handle);
                }
                scene.removeMesh(doomed);
                meshGeometry.delete(doomed);
            }
            // A mesh removed while its builds run: its tinybvh build is cancelled.
            if (round === 4) {
                const brief = addMesh(randomTriangleSoup(random, 2000, 1, 0.3));
                frame();
                scene.removeMesh(brief);
                meshGeometry.delete(brief);
            }
            frame();
            await scene.whenBuilt();
            frame();
            await check(`round ${round} first trees`, true);
            if (scene.counts.onGpuTrees === 0) failures.push(`round ${round}: no mesh was on a GPU tree after the first update`);
            await settle();
            await check(`round ${round} settled`, false);
        }

        const memory = scene.memory;
        const counts = scene.counts;
        scene.dispose();
        kernel.destroy();
        if (failures.length > 0) throw new Error(failures.slice(0, 8).join(' | '));
        return { ...results, swaps: counts.swaps, gpuTreesBuilt: counts.gpuTreesBuilt, mergedGroups: counts.mergedGroups, peakMB: memory.peak / 2 ** 20 };
    },
    { requiredFeatures: ['subgroups'] },
);

/**
 * updateSceneTreesAsync and BvhTraceNodes.updateAsync (PARITY 8.4,
 * three-mesh-bvh's `BVHComputeData.updateAsync`): on a real SceneTrees over a
 * fake GPU, a scene read by the sync has every mesh's tree placed when the
 * promise resolves; the steps run in the documented order; an upload source
 * just rebinds.
 */

import { BoxGeometry, Group, Mesh, MeshBasicMaterial, SphereGeometry, TorusKnotGeometry } from 'three';
import { describe, expect, it } from 'vitest';
import { createFakeGpu, installWebGpuGlobals } from '../../tests/support/fake-gpu.ts';
import { SceneTrees } from '../scene/scene-trees.ts';
import { SceneTreesObject3DSync } from './scene-trees-sync.ts';
import { updateSceneTreesAsync, type SceneTreesAsyncTarget } from './update-async.ts';
import { BvhTraceNodes } from './trace-nodes.ts';

installWebGpuGlobals();

describe('updateSceneTreesAsync (BVHComputeData.updateAsync)', () => {
    it('reads the scene, builds and places every tree before it resolves (real SceneTrees, fake GPU)', async () => {
        const gpu = createFakeGpu();
        const trees = await SceneTrees.create(gpu.device, { gpuFirstTrees: false, topLevel: 'cpu' });
        const root = new Group();
        root.add(new Mesh(new TorusKnotGeometry(1, 0.3, 32, 6), new MeshBasicMaterial()), new Mesh(new SphereGeometry(0.5, 8, 6), new MeshBasicMaterial()));
        root.children[1].position.x = 3;
        root.updateMatrixWorld(true);
        const sync = new SceneTreesObject3DSync(trees, root);
        expect(trees.counts.onTinybvhTrees).toBe(0);
        await updateSceneTreesAsync(trees, { sync, until: 'refined' });
        expect(trees.counts).toMatchObject({ meshes: 2, instances: 2, onTinybvhTrees: 2, waitingForFirstTree: 0, tinybvhBuilding: 0 });

        // A new object later: the same call adds it and waits for its tree.
        root.add(new Mesh(new BoxGeometry(1, 1, 1, 2, 2, 2), new MeshBasicMaterial()));
        root.updateMatrixWorld(true);
        await updateSceneTreesAsync(trees, { sync });
        expect(trees.counts).toMatchObject({ meshes: 3, instances: 3, waitingForFirstTree: 0 });
        trees.dispose();
    });

    it('runs sync, ready, update + submit, the wait, update + submit, then rebinds, in that order', async () => {
        const log: string[] = [];
        const device = {
            createCommandEncoder: () => ({ finish: () => 'commands' }),
            queue: { submit: (commands: unknown[]) => log.push(`submit ${commands.length}`) },
        } as unknown as GPUDevice;
        const trees: SceneTreesAsyncTarget = {
            device,
            update: () => void log.push('trees.update'),
            whenBuilt: async () => void log.push('whenBuilt'),
            whenRefined: async () => void log.push('whenRefined'),
        };
        const sync = {
            sync: () => log.push('sync.sync'),
            update: () => log.push('sync.update'),
            get ready() {
                log.push('sync.ready');
                return Promise.resolve();
            },
        };
        await updateSceneTreesAsync(trees, { sync, nodes: { update: () => log.push('nodes.update') } });
        expect(log).toEqual(['sync.sync', 'sync.ready', 'sync.update', 'trees.update', 'submit 1', 'whenBuilt', 'sync.update', 'trees.update', 'submit 1', 'nodes.update']);
        log.length = 0;
        await updateSceneTreesAsync(trees, { until: 'refined' });
        expect(log).toEqual(['trees.update', 'submit 1', 'whenRefined', 'trees.update', 'submit 1']);
    });

    it('BvhTraceNodes.updateAsync drives a SceneTrees source, and only rebinds an upload', async () => {
        const gpu = createFakeGpu();
        const trees = await SceneTrees.create(gpu.device, { gpuFirstTrees: false, topLevel: 'cpu' });
        const root = new Group();
        root.add(new Mesh(new SphereGeometry(0.5, 8, 6), new MeshBasicMaterial()));
        root.updateMatrixWorld(true);
        const sync = new SceneTreesObject3DSync(trees, root);
        const backend = { isWebGPUBackend: true, device: gpu.device, get: () => ({}), destroyAttribute() {} };
        const renderer = { backend, _attributes: { has: () => false } };
        const nodes = new BvhTraceNodes(renderer, trees);
        let rebinds = 0;
        const update = nodes.update.bind(nodes);
        nodes.update = () => {
            rebinds++;
            return update();
        };
        await nodes.updateAsync({ sync });
        expect(trees.counts).toMatchObject({ meshes: 1, waitingForFirstTree: 0 });
        expect(rebinds).toBe(1);
        expect(nodes.version).toBe(trees.version);

        const buffer = () => ({ label: 'b', destroy() {} }) as unknown as GPUBuffer;
        const uploaded = new BvhTraceNodes(renderer, { nodeFormat: 'f32', triangleLeaves: 'records', positionLayout: 'vec4', buffers: { nodes: buffer(), triangles: buffer(), topLevel: buffer() } });
        await uploaded.updateAsync();
        await expect(uploaded.updateAsync({ sync })).rejects.toThrow(/SceneTrees source/);
        trees.dispose();
    });
});

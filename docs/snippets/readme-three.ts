// Code samples for README.md (see readme.ts).

// #region readme-three
import { SceneTrees } from '@pmndrs/bvh';
import { BvhTraceNodes, deviceFromRendererAsync, SceneTreesObject3DSync } from '@pmndrs/bvh/three';

// #endregion
import type { BvhTslNode } from '@pmndrs/bvh/three';
import type { Scene, WebGPURenderer } from 'three/webgpu';

export async function readmeThree(renderer: WebGPURenderer, scene: Scene, origin: BvhTslNode, direction: BvhTslNode): Promise<BvhTslNode> {
    // #region readme-three
    const device = await deviceFromRendererAsync(renderer); // the renderer's own device
    const trees = await SceneTrees.create(device);
    const sync = new SceneTreesObject3DSync(trees, scene); // every mesh and instance under `scene`
    const bvh = new BvhTraceNodes(renderer, trees);
    const hit = bvh.traceClosest(origin, direction); // TSL: a struct node, hit.get('t') < 0 on a miss

    // Each frame:
    sync.update();
    const encoder = device.createCommandEncoder();
    trees.update(encoder);
    device.queue.submit([encoder.finish()]);
    bvh.update();
    // #endregion
    return hit;
}

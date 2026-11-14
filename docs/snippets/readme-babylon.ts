// Code samples for README.md (see readme.ts).

// #region readme-babylon
import { SceneTrees } from '@pmndrs/bvh';
import { BabylonBvhPicker, bvhSceneFromBabylon, deviceFromEngine } from '@pmndrs/bvh/babylon';

// #endregion
import type { PickingInfo } from '@babylonjs/core/Collisions/pickingInfo.js';
import type { WebGPUEngine } from '@babylonjs/core/Engines/webgpuEngine.js';
import type { Scene } from '@babylonjs/core/scene.js';

export async function readmeBabylon(engine: WebGPUEngine, scene: Scene): Promise<PickingInfo> {
    // #region readme-babylon
    // CPU picking, on any Babylon engine: a PickingInfo, as scene.pick returns it.
    const picker = await BabylonBvhPicker.create(scene);
    const info = picker.pick(scene.pointerX, scene.pointerY);

    // The walk on the WebGPUEngine's own device (engine._device, checked).
    const device = deviceFromEngine(engine);
    const trees = await SceneTrees.create(device);
    const read = bvhSceneFromBabylon(scene);
    const meshes = read.meshes.map((geometry) => trees.addMesh(geometry));
    for (const instance of read.instances) trees.addInstance({ mesh: meshes[instance.meshIndex], objectToWorld: instance.objectToWorld, instanceIndex: instance.instanceIndex });
    // Each frame: trees.update(encoder) and submit, then bindTraceBuffers(shader, trees.buffers) on a ComputeShader
    // from createTraceComputeShader (docs: engines/babylon).
    // #endregion
    return info;
}

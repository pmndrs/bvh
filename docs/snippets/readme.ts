// Code samples for README.md (see getting-started.ts for how they are checked; the README marks
// its blocks with `<!-- snippet: name -->`).

// #region readme-vanilla
import { IDENTITY_TRANSFORM, SceneTrees, traceModuleWgsl } from '@pmndrs/bvh';

// #endregion

export async function readmeVanilla(device: GPUDevice, positions: Float32Array, indices: Uint32Array): Promise<string> {
    // #region readme-vanilla
    const trees = await SceneTrees.create(device); // GPU H-PLOC trees with subgroups, tinybvh otherwise
    const mesh = trees.addMesh({ positions, indices });
    trees.addInstance({ mesh, objectToWorld: IDENTITY_TRANSFORM });

    // Each frame: builds, swaps and refits, recorded before anything traces.
    const encoder = device.createCommandEncoder();
    trees.update(encoder);
    device.queue.submit([encoder.finish()]);

    // Your kernel calls bvhTraceClosest / bvhTraceAny; bind trees.buffers.nodes, .triangles, .topLevel at group 0.
    const code = `${traceModuleWgsl()}\n/* ... let hit = bvhTraceClosest(BvhRay(origin, 0.0, direction, 1e30)); ... */`;
    // #endregion
    return code;
}

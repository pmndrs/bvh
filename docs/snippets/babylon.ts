// Code samples for docs/engines/babylon.mdx and the README's Babylon quick start (see getting-started.ts for how they are checked).

// #region babylon-engine
import { WebGPUEngine } from '@babylonjs/core/Engines/webgpuEngine.js';
import { deviceFromEngine } from '@pmndrs/bvh/babylon';

// #endregion

export async function babylonEngine(canvas: HTMLCanvasElement): Promise<GPUDevice> {
    // #region babylon-engine
    // Babylon asks for WebGPU's default limits and no optional features: ask for the adapter's limits
    // (a scene past 128 MiB of triangles needs them) and for its features (the GPU builders need `subgroups`).
    const engine = await WebGPUEngine.CreateAsync(canvas, { enableAllFeatures: true, setMaximumLimits: true });
    const device = deviceFromEngine(engine); // engine._device, checked; throws on the WebGL engine
    // #endregion
    return device;
}

// #region babylon-picking
import { BabylonBvhPicker } from '@pmndrs/bvh/babylon';
import type { PickingInfo } from '@babylonjs/core/Collisions/pickingInfo.js';
import type { Scene } from '@babylonjs/core/scene.js';

// #endregion

export async function babylonPicking(scene: Scene): Promise<PickingInfo> {
    // #region babylon-picking
    const picker = await BabylonBvhPicker.create(scene); // loads tinybvh, builds a tree per geometry
    const info = picker.pick(scene.pointerX, scene.pointerY); // a PickingInfo, as scene.pick returns
    if (info.hit) console.log(info.pickedMesh?.name, info.faceId, info.distance);

    // The picker sees the scene as of the last update(): call it after meshes move, show, hide or change.
    picker.update();
    // #endregion
    return info;
}

// #region babylon-trees
import { SceneTrees } from '@pmndrs/bvh';
import { bvhSceneFromBabylon } from '@pmndrs/bvh/babylon';

// #endregion

export async function babylonTrees(device: GPUDevice, scene: Scene): Promise<SceneTrees> {
    // #region babylon-trees
    const read = bvhSceneFromBabylon(scene); // one mesh per geometry, one instance per Mesh and InstancedMesh
    const trees = await SceneTrees.create(device);
    const meshes = read.meshes.map((geometry) => trees.addMesh(geometry));
    for (const instance of read.instances) {
        trees.addInstance({
            mesh: meshes[instance.meshIndex],
            objectToWorld: instance.objectToWorld, // Babylon's world matrix memory is column-major as it is
            instanceIndex: instance.instanceIndex, // so a hit's `instance` indexes read.sources
            mask: instance.mask,
            visible: instance.visible,
        });
    }
    // A hit's instance names the Babylon mesh: read.sources[hit.instance].object
    // #endregion
    return trees;
}

// #region babylon-compute
import { Constants } from '@babylonjs/core/Engines/constants.js';
import { StorageBuffer } from '@babylonjs/core/Buffers/storageBuffer.js';
import type { WebGPUEngine as Engine } from '@babylonjs/core/Engines/webgpuEngine.js';
import { bindTraceBuffers, createTraceComputeShader } from '@pmndrs/bvh/babylon';

// #endregion

export function babylonCompute(engine: Engine, trees: SceneTrees, rayCount: number): { dispatch(): void; rays: StorageBuffer; results: StorageBuffer } {
    // #region babylon-compute
    // The walk (traceModuleWgsl) comes first; the kernel's own bindings follow its three (0-2).
    const kernel = `
        @group(0) @binding(3) var<storage, read> rays: array<BvhRay>;
        @group(0) @binding(4) var<storage, read_write> results: array<vec2f>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            if (id.x >= arrayLength(&rays)) { return; }
            let hit = bvhTraceClosest(rays[id.x]);
            results[id.x] = vec2f(hit.t, f32(hit.instance));
        }`;
    const shader = createTraceComputeShader('bvh trace', engine, kernel, {
        trace: { nodeFormat: trees.nodeFormat, triangleLeaves: trees.triangleLeaves, positionLayout: trees.positionLayout },
        bindings: { rays: { group: 0, binding: 3 }, results: { group: 0, binding: 4 } },
    });
    const rays = new StorageBuffer(engine, rayCount * 32, Constants.BUFFER_CREATIONFLAG_READWRITE); // BvhRay is 32 bytes: origin, tMin, direction, tMax
    const results = new StorageBuffer(engine, rayCount * 8, Constants.BUFFER_CREATIONFLAG_READWRITE);
    shader.setStorageBuffer('rays', rays);
    shader.setStorageBuffer('results', results);

    // Each frame, after trees.update(encoder) was submitted: bind SceneTrees' own GPUBuffers (no copy) and dispatch.
    // Bind again whenever trees.version changes (a buffer grew).
    const dispatch = () => {
        bindTraceBuffers(shader, trees.buffers);
        shader.dispatch(Math.ceil(rayCount / 64));
    };
    // #endregion
    return { dispatch, rays, results };
}

// Code samples for docs/getting-started/*.mdx. `pnpm typecheck` checks this file
// (tsconfig.examples.json), and tests/docs-snippets.test.ts checks that each
// `#region` here is the code block of the same name in the MDX, character for character.
// Edit the region here, then run `pnpm docs:snippets` to copy it into the pages.

// #region device
export async function requestBvhDevice(): Promise<GPUDevice> {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available');
    const device = await adapter.requestDevice({
        // The GPU builders and refits need subgroups. Without them, tinybvh builds every tree on the CPU.
        requiredFeatures: adapter.features.has('subgroups') ? ['subgroups'] : [],
        // A big scene needs more than the default 128 MiB per storage binding.
        requiredLimits: {
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxBufferSize: adapter.limits.maxBufferSize,
        },
    });
    return device;
}
// #endregion

// #region quick-start-vanilla
import { HIT_RECORD_BYTES, IDENTITY_TRANSFORM, packRays, RAY_BYTES, SceneTrees, TraceBatchKernel } from '@pmndrs/bvh';

// #endregion

export async function quickStartVanilla(device: GPUDevice, positions: Float32Array, indices: Uint32Array): Promise<GPUBuffer> {
    // #region quick-start-vanilla
    const trees = await SceneTrees.create(device);
    const mesh = trees.addMesh({ positions, indices });
    trees.addInstance({ mesh, objectToWorld: IDENTITY_TRANSFORM });

    const update = () => {
        const encoder = device.createCommandEncoder();
        trees.update(encoder); // starts builds, places finished trees, refits the top level
        device.queue.submit([encoder.finish()]);
    };
    update();
    await trees.whenBuilt(); // every mesh has a tree
    update();

    // One ray, traced with the package's batch kernel.
    const kernel = await TraceBatchKernel.create(device);
    const rays = device.createBuffer({ size: RAY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(rays, 0, packRays([{ origin: [0, 0, 5], direction: [0, 0, -1], tMin: 0, tMax: 1e30 }]));
    const results = device.createBuffer({ size: HIT_RECORD_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const encoder = device.createCommandEncoder();
    kernel.encode(encoder, trees.traceBuffers, { mode: 'closest', rays, results, count: 1 });
    device.queue.submit([encoder.finish()]);
    // #endregion
    return results;
}

// #region read-hit
import { HIT_RECORD_WORD } from '@pmndrs/bvh';

// #endregion

export async function readHit(device: GPUDevice, results: GPUBuffer): Promise<void> {
    // #region read-hit
    const readback = device.createBuffer({ size: HIT_RECORD_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(results, 0, readback, 0, HIT_RECORD_BYTES);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const floats = new Float32Array(readback.getMappedRange());
    const words = new Uint32Array(floats.buffer);
    const t = floats[HIT_RECORD_WORD.t]; // < 0: a miss
    if (t >= 0) console.log('hit triangle', words[HIT_RECORD_WORD.triangle], 'of instance', words[HIT_RECORD_WORD.instance], 'at', t);
    readback.unmap();
    // #endregion
}

// #region quick-start-three
import { Fn, If, vec3 } from 'three/tsl';
import { WebGPURenderer, type Scene } from 'three/webgpu';
import { BvhTraceNodes, deviceFromRendererAsync, SceneTreesObject3DSync, type BvhTslNode } from '@pmndrs/bvh/three';

// #endregion

export async function quickStartThree(scene: Scene): Promise<() => void> {
    // #region quick-start-three
    const renderer = new WebGPURenderer();
    const device = await deviceFromRendererAsync(renderer); // the renderer's own GPUDevice
    const trees = await SceneTrees.create(device);
    const sync = new SceneTreesObject3DSync(trees, scene); // adds every mesh and instance under `scene`
    const bvh = new BvhTraceNodes(renderer, trees); // the walk in three's node system

    // A TSL compute pass that traces (a material's nodes can call the same helpers):
    const kernel = Fn(() => {
        const hit = bvh.traceClosest(vec3(0, 1, 5), vec3(0, 0, -1)).toVar() as BvhTslNode;
        If(hit.get('t').greaterThanEqual(0), () => {
            /* hit.get('normal'), hit.get('instance'), hit.get('material'), ... */
        });
    })().compute(1);

    // Each frame, before three uses the nodes:
    const frame = () => {
        scene.updateMatrixWorld();
        sync.update(); // moved objects only
        const encoder = device.createCommandEncoder();
        trees.update(encoder);
        device.queue.submit([encoder.finish()]);
        bvh.update(); // rebinds if SceneTrees replaced a buffer
        renderer.compute(kernel);
    };
    // #endregion
    return frame;
}

// #region quick-start-three-cpu
import { BufferGeometry, Mesh, Raycaster, type Camera, type Vector2 } from 'three';
import { acceleratedRaycast, computeBoundsTree, disposeBoundsTree, prepareBoundsTrees } from '@pmndrs/bvh/three';

await prepareBoundsTrees(); // loads tinybvh (WebAssembly) once
BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
Mesh.prototype.raycast = acceleratedRaycast;
// #endregion

export function quickStartThreeCpu(mesh: Mesh, camera: Camera, pointer: Vector2): number {
    // #region quick-start-three-cpu
    mesh.geometry.computeBoundsTree();
    const raycaster = new Raycaster();
    raycaster.setFromCamera(pointer, camera); // pointer: -1 to 1 on both axes
    raycaster.firstHitOnly = true; // optional: the closest hit only
    const hits = raycaster.intersectObject(mesh); // the same intersections three's raycast returns
    // #endregion
    return hits.length;
}

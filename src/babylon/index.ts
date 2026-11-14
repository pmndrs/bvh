/**
 * Babylon.js in, the core's inputs out: meshes to scene inputs, picking with `scene.pick`'s result, and the engine's device and a `ComputeShader` for the WGSL walk. `@babylonjs/core` is an optional peer dependency.
 *
 * @module @pmndrs/bvh/babylon
 */

// @pmndrs/bvh/babylon: Babylon.js in, the engine-free core's inputs out. The only
// part of the package that imports @babylonjs/*; see docs/engines/babylon.mdx.

// The WebGPU engine's device (engine._device, checked).
export { deviceFromEngine } from './device.ts';

// Meshes and instances to buildTraceScene / SceneTrees / CpuSceneBvh inputs.
export {
    BABYLON_MASK_PICKABLE,
    BABYLON_MASK_VISIBLE,
    babylonMeshesOf,
    bvhSceneFromBabylon,
    meshGeometryFromBabylon,
    type BvhBabylonMeshSource,
    type BvhBabylonScene,
    type BvhBabylonSource,
    type BvhSceneFromBabylonOptions,
} from './scene.ts';

// scene.pick through the CPU raycast.
export { BabylonBvhPicker, type BabylonBvhPickerOptions, type BvhPickOptions } from './picking.ts';

// Our buffers in a ComputeShader, with no copy.
export { bindTraceBuffers, createTraceComputeShader, gpuBufferToDataBuffer, TRACE_BINDING_NAMES, traceBindingsMapping, wgslForBabylon, type BabylonTraceBuffers, type TraceComputeShaderOptions } from './compute.ts';

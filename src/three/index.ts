// @pmndrs/bvh/three: three.js in, the engine-free core's inputs out. The only
// part of the package that imports three (a peer dependency); see docs/three.md.

// Scenes: three objects to buildTraceScene / SceneTrees inputs, and back.
export {
    bvhSceneFromObject3D,
    deviceFromRenderer,
    deviceFromRendererAsync,
    instanceMaskFromLayers,
    rayMaskFromLayers,
    threeFaceIndex,
    threeTriangleRanges,
    type BvhSceneFromObject3DOptions,
    type BvhThreeMaterialInfo,
    type BvhThreeMeshSource,
    type BvhThreeScene,
    type BvhThreeSource,
} from './scene.ts';
export { SceneTreesObject3DSync, type Object3DSyncCounts, type SceneTreesTarget } from './scene-trees-sync.ts';

// The walk in three's node system (three-mesh-bvh's BVHComputeData pattern), for TSL and wgslFn.
export { BvhTraceNodes, bvhTraceHitStruct, type BvhTraceBufferSource, type BvhTraceNodesOptions, type BvhTslNode } from './trace-nodes.ts';

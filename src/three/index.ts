/**
 * three.js in, the core's inputs out: scenes from Object3D, the sync, BvhTraceNodes for TSL and WGSL, and three-mesh-bvh-style CPU raycasts and queries. three is an optional peer dependency.
 *
 * @module @pmndrs/bvh/three
 */

// @pmndrs/bvh/three: three.js in, the engine-free core's inputs out. The only
// part of the package that imports three (a peer dependency); see docs/internals/three.md.

// Scenes: three objects to buildTraceScene / SceneTrees inputs, and back.
export {
    bvhSceneFromObject3D,
    deviceFromRenderer,
    deviceFromRendererAsync,
    instanceMaskFromLayers,
    rayMaskFromLayers,
    threeFaceIndex,
    threeGroupOf,
    threeMaterialOf,
    threeTriangleRanges,
    type BvhSceneFromObject3DOptions,
    type BvhThreeGroupRange,
    type BvhThreeMaterialInfo,
    type BvhThreeMeshSource,
    type BvhThreeScene,
    type BvhThreeSource,
} from './scene.ts';
export { SceneTreesObject3DSync, type Object3DPoseCounts, type Object3DSyncCounts, type SceneTreesObject3DSyncOptions, type SceneTreesTarget } from './scene-trees-sync.ts';
// Skinned and morphed meshes on the GPU: three's skinning and morph math in a TSL compute pass, feeding SceneTrees.setPose.
export { BvhPosePass, poseVertices, threeDeforms, type BvhPosePassLike, type BvhPosePassOptions } from './deform.ts';

// CPU raycasts (Q1) and the CPU tree: three-mesh-bvh-style drop-ins over the core's CpuMeshBvh.
export {
    acceleratedRaycast,
    AVERAGE,
    BvhBoundsTree,
    // three-mesh-bvh's name, so switching is an import change: `import { MeshBVH } from '@pmndrs/bvh/three'`.
    BvhBoundsTree as MeshBVH,
    buildOptionsFromThreeMeshBvh,
    CENTER,
    computeBoundsTree,
    deserializeBoundsTree,
    disposeBoundsTree,
    getTriangleHitPointInfo,
    meshInputFromGeometry,
    prepareBoundsTrees,
    SAH,
    setBoundsTree,
    snapshotGeometry,
    type BvhBoundsTreeHit,
    type BvhBoundsTreeOptions,
    type BvhClosestPointInfo,
    // three-mesh-bvh's name for it.
    type BvhClosestPointInfo as HitPointInfo,
    type BvhGeometrySnapshot,
    type BvhHitPointInfo,
    type BvhHitPointInfoTarget,
    type BvhSplitStrategy,
} from './raycast.ts';
// A ray against the whole scene on the CPU (three-mesh-bvh's ObjectBVH.raycast), and frustum culling over the top level.
export { BvhSceneRaycaster, raycastScene, type BvhSceneRaycasterOptions, type BvhSceneShapecastCallbacks } from './scene-raycast.ts';
// Shapecast and bvhcast (Q2): three-mesh-bvh's shape classes with three's types, over the core's math.
export { ExtendedTriangle, OrientedBox, SeparatingAxisBounds } from './shapes.ts';
// three-mesh-bvh's names for what intersectsBounds returns.
export { BVH_CONTAINED as CONTAINED, BVH_INTERSECTED as INTERSECTED, BVH_NOT_INTERSECTED as NOT_INTERSECTED } from '../query/cpu-mesh-bvh.ts';
export { GenerateMeshBvhWorker, type GenerateMeshBvhWorkerOptions } from './generate.ts';
// three-mesh-bvh's helper: a tree's boxes as line segments.
export { BVHHelper, MeshBVHHelper } from './helper.ts';
// three-mesh-bvh's debug functions (the core's, which take any CpuMeshBvh, so a BvhBoundsTree too).
export { estimateMemoryInBytes, getBVHExtremes, validateBounds, type BvhExtremes, type BvhExtremes as ExtremeInfo } from '../query/debug.ts';
export { getJSONStructure } from './debug.ts';
// three-mesh-bvh's serialized format (MeshBVH.serialize / deserialize take it too), next to ours.
export { meshBvhFromThreeMeshBvh, serializeThreeMeshBvh, type BvhJsonNode, type ThreeMeshBvhSerialized, type ThreeMeshBvhSerialized as SerializedBVH } from '../query/three-mesh-bvh-format.ts';

// The walk in three's node system (three-mesh-bvh's BVHComputeData pattern), for TSL and wgslFn.
export { BvhTraceNodes, bvhClosestPointHitStruct, bvhTraceHitStruct, type BvhTraceBuffers, type BvhTraceBufferSource, type BvhTraceNodesOptions, type BvhTslNode } from './trace-nodes.ts';

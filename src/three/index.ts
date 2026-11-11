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
// Shapecast and bvhcast (Q2): three-mesh-bvh's shape classes with three's types, over the core's math.
export { ExtendedTriangle, OrientedBox, SeparatingAxisBounds } from './shapes.ts';
// three-mesh-bvh's names for what intersectsBounds returns.
export { BVH_CONTAINED as CONTAINED, BVH_INTERSECTED as INTERSECTED, BVH_NOT_INTERSECTED as NOT_INTERSECTED } from '../query/cpu-mesh-bvh.ts';
export { GenerateMeshBvhWorker, type GenerateMeshBvhWorkerOptions } from './generate.ts';

// The walk in three's node system (three-mesh-bvh's BVHComputeData pattern), for TSL and wgslFn.
export { BvhTraceNodes, bvhTraceHitStruct, type BvhTraceBufferSource, type BvhTraceNodesOptions, type BvhTslNode } from './trace-nodes.ts';

/**
 * The engine-free core: builders, refits, the GPU layout, the WGSL walk and its CPU port, SceneTrees, CPU queries and serialization.
 *
 * @module @pmndrs/bvh
 */

// @pmndrs/bvh: engine-free BVH builders and refits for WebGPU, writing one
// shared GPU layout (docs/internals/design/bvh-framework.md). This file is the public
// API: everything a user needs, and nothing that is only there for a builder's
// own plumbing. Each name is listed on purpose; the package has no other entry
// point, so a new export here is an API decision.

// The builder contract: what goes in, what comes out.
export {
    isGpuMeshBuffers,
    meshTriangleCount,
    TRIANGLE_MATERIAL_NONE,
    type GpuBuildOptions,
    type GpuMeshBuffers,
    type GpuMeshGeometry,
    type GpuMeshTreeRange,
    type GpuTreeBuffers,
    type MeshInput,
} from './mesh-input.ts';

// The GPU layout every builder writes and every walk reads. It is part of the
// API: changing it bumps GPU_LAYOUT_VERSION.
export {
    DEFAULT_NODE_FORMAT,
    GPU_LAYOUT_CHILD_STRIDE_WORDS,
    GPU_LAYOUT_EMPTY_CHILD,
    GPU_LAYOUT_MAX_LEAF_PRIMITIVES,
    GPU_LAYOUT_MAX_TRIANGLES,
    GPU_LAYOUT_NO_PARENT,
    GPU_LAYOUT_NODE_BYTES,
    GPU_LAYOUT_NODE_WORD,
    GPU_LAYOUT_NODE_WORDS,
    GPU_LAYOUT_TRIANGLE_BYTES,
    GPU_LAYOUT_TRIANGLE_WORD,
    GPU_LAYOUT_VERSION,
    GPU_LAYOUT_INDEXED_FLAG_SHIFT,
    GPU_LAYOUT_INDEXED_ID_MASK,
    GPU_LAYOUT_INDEXED_TRIANGLE_BYTES,
    GPU_LAYOUT_INDEXED_TRIANGLE_WORD,
    GPU_LAYOUT_MESH_HEADER_WORD,
    GPU_LAYOUT_NODE_FRAME_WORD,
    GPU_LAYOUT_QUANTIZED_NODE_BYTES,
    GPU_LAYOUT_QUANTIZED_NODE_WORD,
    GPU_LAYOUT_QUANTIZED_NODE_WORDS,
    gpuLayoutNodeBufferBytes,
    gpuLayoutNodeBytes,
    gpuLayoutNodeSlots,
    gpuLayoutTriangleBytes,
    type GpuLayoutMeshBvh,
    type GpuLayoutNodeFormat,
    type GpuLayoutTriangleLeaves,
} from './gpu-layout.ts';
// Quantized nodes (docs/internals/design/compressed-nodes.md): packing a tree's nodes, unpacking them, the frame rule.
export { dequantizeNodes, nodeFrameOf, quantizeNodes, quantizeNodeWords, readNodeFrame, type NodeFrame } from './quantized-nodes.ts';
export { GpuNodeQuantizer, quantizeTreeBuffers, type QuantizeJob, type QuantizeMesh } from './gpu-quantize.ts';
// Indexed leaves (docs/internals/design/indexed-leaves.md): converting a tree between the formats, and the weld.
export {
    indexedFromRecords,
    recordsFromIndexed,
    weldExactBits,
    weldPositions,
    weldWithinTolerance,
    type WeldOptions,
    type WeldResult,
} from './indexed-leaves.ts';
export { deserializeLayout, SERIALIZED_FORMAT_VERSION, serializedLayoutBytes, serializeLayout } from './serialize.ts';

// A tree's shape alone, before it is written in the GPU layout.
export { singleLeafTopology, TOPOLOGY_EMPTY_CHILD, TOPOLOGY_WORDS_PER_NODE, type MeshTreeTopology } from './topology.ts';

// Builders. createMeshLayoutBuilder picks one for you.
export { createMeshLayoutBuilder, type MeshLayoutBuilder, type MeshLayoutBuilderKind, type MeshLayoutResult } from './mesh-builder.ts';
export {
    createTinyBvhBuilder,
    TINYBVH_RELEASE_ABOVE_BYTES,
    topologyFromTinyBvhNodes,
    type TinyBvhBuild,
    type TinyBvhBuilder,
    type TinyBvhBuilderOptions,
    type TinyBvhBuildOptions,
    type TinyBvhLayoutBuild,
    type TinyBvhLayoutOptions,
} from './tinybvh.ts';
export {
    GpuHplocBatch,
    GpuHplocBuilder,
    GpuHplocScratchPool,
    HPLOC_STAGES,
    hplocBatchBytes,
    hplocScratchPoolBytes,
    planGpuHplocBatches,
    type HplocBatchPlanOptions,
    type HplocStage,
} from './gpu-hploc.ts';
export { HPLOC_STAT } from './gpu-hploc-wgsl.ts';
export {
    geometryTotals,
    GpuLbvhBatch,
    GpuLbvhBuilder,
    LBVH_STAGES,
    lbvhBatchBytes,
    planGpuLbvhBatches,
    type BatchBytes,
    type BatchPlanOptions,
    type BatchTotals,
    type LbvhStage,
} from './gpu-lbvh.ts';
export { LBVH_STAT } from './gpu-lbvh-wgsl.ts';

// CPU twins of the GPU builders: the same trees, built in plain TypeScript, as references.
export { buildHplocOnCpu, type HplocMortonCodes, type HplocOptions, type HplocTieBreak } from './hploc-cpu.ts';
export { buildLbvhOnCpu } from './lbvh-cpu.ts';

// Refits: keep a tree's shape, recompute its boxes.
export {
    BVH_REFIT_STAT,
    BVH_REFIT_STATS_BYTES,
    BVH_REFIT_WORKGROUP_SIZE,
    GpuBvhRefit,
    GpuBvhRefitKernels,
    type BvhRefitShape,
    type BvhRefitStats,
    type GpuBvhRefitOptions,
} from './gpu-refit.ts';
export { GpuMeshRefit, GpuMeshRefitKernels, type GpuMeshRefitOptions, type MeshRefitTarget, type MeshRefitVertices } from './gpu-mesh-refit.ts';
export { BVH_INSTANCE_BOX_PADDING, BVH_INSTANCE_BOX_WGSL } from './gpu-instance-box.ts';
export {
    BVH_FLOAT_MAX,
    halfArea,
    leafSlotsOf,
    levelOrderOf,
    parentSlotsOf,
    PRIMITIVE_BOX_FLOATS,
    refitLayoutToPose,
    refitOnCpu,
    refitOrderOf,
    refitRecordsOnCpu,
    relativeSahCost,
    type CpuRefitResult,
    type PosedLayout,
} from './refit-cpu.ts';

// CPU queries on one mesh's tree (P8): raycasts, the box walk shapecasts build on, refit, saving.
export {
    barycoordOnTriangle,
    BVH_BACK_SIDE,
    BVH_CONTAINED,
    BVH_DOUBLE_SIDE,
    BVH_FRONT_SIDE,
    BVH_INTERSECTED,
    BVH_NOT_INTERSECTED,
    CpuMeshBvh,
    type BvhBoxTarget,
    type BvhBvhcastCallbacks,
    type BvhClosestPoint,
    type BvhClosestPointTarget,
    type BvhRayIntersection,
    type BvhRayLike,
    type BvhRaycastHit,
    type BvhRaycastOptions,
    type BvhShapecastCallbacks,
    type BvhTraverseCallbacks,
    type BvhTriangleAccept,
    type BvhTriangleHitPointInfo,
    type BvhTriangleHitPointOptions,
    type BvhVec3Like,
    type BvhVertexAttribute,
    type CpuMeshBvhBuildOptions,
    type CpuMeshBvhOptions,
} from './query/cpu-mesh-bvh.ts';

// three-mesh-bvh's debug functions on a CpuMeshBvh.
// Scene-level CPU queries (three-mesh-bvh's ObjectBVH): a ray against every instance, frustum and box culling over the top level.
export { CpuSceneBvh, type CpuFrustumPlanes, type CpuSceneHit, type CpuSceneRaycastOptions } from './query/cpu-scene.ts';
export { BVH_PRIMITIVE_INTERSECT_COST, BVH_TRAVERSAL_COST, bvhHelperIndices, bvhHelperPositions, estimateMemoryInBytes, getBVHExtremes, validateBounds, type BvhExtremes } from './query/debug.ts';

// Query shapes (three-mesh-bvh's ExtendedTriangle and OrientedBox, engine-free) and their math.
export {
    BvhBox3,
    BvhMatrix4,
    BvhPlane,
    BvhSegment,
    BvhVector3,
    closestPointsSegmentToSegment,
    ExtendedTriangle,
    OrientedBox,
    orientedBoxClosestPointToPoint,
    orientedBoxDistanceToBox,
    orientedBoxDistanceToPoint,
    orientedBoxIntersectsBox,
    orientedBoxIntersectsTriangle,
    SeparatingAxisBounds,
    sphereIntersectsBox,
    sphereIntersectsTriangle,
    triangleClosestPointToPoint,
    triangleClosestPointToSegment,
    triangleContainsPoint,
    triangleDistanceToPoint,
    triangleDistanceToTriangle,
    triangleGetNormal,
    triangleIntersectsTriangle,
    updateExtendedTriangle,
    updateOrientedBox,
    type BvhBox3Like,
    type BvhMatrix4Like,
    type BvhMatrix4Target,
    type BvhPlaneLike,
    type BvhQueryTriangle,
    type BvhSegmentLike,
    type BvhSphereLike,
    type BvhTriangleLike,
    type BvhVector3Like,
    type ExtendedTriangleState,
    type OrientedBoxState,
} from './query/shapes.ts';

// Checks on any tree.
export { checkGpuLayout, treeSahCost, type LayoutCheckMergedIdentity, type LayoutCheckOptions } from './layout-check.ts';

// The tracer's layout, the WGSL walk (closest/any) and its CPU port, the oracle,
// tinybvh trees in the tracer's layout, the top-level builder and the worker wrappers (BX.2).
export * from './trace/index.ts';

// A scene's trees on one device: GPU H-PLOC trees first, tinybvh's from the worker pool swapped in
// mesh by mesh, the GPU top level (or the CPU one without subgroups), runtime adds and removals,
// merged static groups, and memory stats.
export {
    DEFAULT_GPU_MAX_BATCH_BYTES,
    GPU_TREE_MAX_DEPTH,
    SceneTrees,
    type GpuTreeOutputsMode,
    type SceneInstanceChange,
    type SceneInstanceDescriptor,
    type SceneInstanceStatus,
    type SceneMeshOptions,
    type SceneMeshStatus,
    type SceneTreeCounts,
    type SceneTreeEvent,
    type SceneTreesMemory,
    type SceneTreesOptions,
    type SceneTreesTimeline,
    type TreeBuilder,
} from './scene/scene-trees.ts';
export { DRIFT_HOLD_FRAMES, DRIFT_THRESHOLD, STATS_EVERY_FRAMES, type DeformingTreeCounts } from './scene/deforming-trees.ts';
export { GpuFirstTrees, GpuTreeOutputs, type FirstTreeBuilder, type GpuBuiltTree, type GpuTreeBatchResult, type GpuTreeBuilds, type GpuTreeRequest } from './scene/gpu-first-trees.ts';

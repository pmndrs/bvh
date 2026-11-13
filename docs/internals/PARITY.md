# Parity with three-mesh-bvh 0.9.18 (BX.16)

The audit P8 asks for (`docs/internals/design/bvh-package-split.md`, P8). It compares our package, as of `main` on 2026-11-11 (rows 1.1-1.7, 2.1, 2.2, 4.1-4.4, 4.7, 6.1, 6.2, 6.4, 6.5, 7.1-7.4 and 9.1 updated after build-out items 1 to 7 and part of 12; rows 8.7 and 8.8 and the accepted notes on 2026-11-13), with three-mesh-bvh 0.9.18 as installed in `node_modules/three-mesh-bvh`. The sources are its `src/index.js`, `src/index.d.ts`, `src/webgpu/index.js` and `index.d.ts`, `src/workers/index.js` and `index.d.ts`, and its README example list (the package ships no `example/` folder).

**Status**
- **have**: we have an equivalent. Our code is cited.
- **partial**: some of it exists. The gap is stated.
- **missing**: nothing yet.
- **won't**: not a target, with the reason.

**Who needs it** lists the call sites in the surveyed projects (read only, 2026-11-11). "None" means no call site was found in the folders named for the survey. Paths are relative to `/Users/dex/Developer/`. Short names:

| Short name | Folder surveyed |
| --- | --- |
| csg | `three-bvh-csg/src` |
| homefig | `homefig/editor/packages` |
| drei | `drei/src` |
| pathtracer | `three-gpu-pathtracer/src` |
| Denoiser | `Denoiser/examples/three-temporal/src` |
| upscaler | `upscaler/.claude/worktrees/bold-blackburn-d78522/examples/s5-radiance-cascades` (see note 1) |
| ddgi-research | `ddgi-research/experiments/gpu-ray-provider` |
| reflections | `reflections/packages` |

Notes on the survey:
1. `upscaler/examples/` has no `s5-radiance-cascades` folder. The only copy is in the worktree named above, so that is the path cited.
2. csg's `closestPointToPoint` calls (`three-bvh-csg/src/objects/HalfEdgeHelper.js:259-260`, `src/core/CDTTriangleSplitter.js:61`) are on three's `Triangle` and `Line3`, not on a BVH. They are not a BVH query.
3. csg has no `MeshBVHHelper` call in `src`. drei uses it only in a story (`drei/src/external/Performance/Bvh/Bvh.stories.tsx:5,29`).
4. pathtracer's `StaticGeometryGenerator` is its own class (`three-gpu-pathtracer/src/core/utils/StaticGeometryGenerator.js:92`), not three-mesh-bvh's. Whether it was derived from it is unverified.

## 1. CPU queries

| # | three-mesh-bvh | Status | Ours / what is missing | Who needs it |
| --- | --- | --- | --- | --- |
| 1.1 | `MeshBVH.raycast` (all hits) | have | `CpuMeshBvh.raycast(ray, { side, near, far })` (`src/query/cpu-mesh-bvh.ts`); `geometry.boundsTree.raycast(ray, materialOrSide, near, far)` with three's types, and `acceleratedRaycast`, the same fields as three's `Mesh.raycast` (`src/three/raycast.ts`); `createCpuTracer().closest/any` for bare rays (`src/trace/cpu-traversal.ts`). All run the one CPU walk | homefig (`viewer/src/components/viewer/scene-bvh.tsx:84`, `editor/.../first-person-controls.tsx:332`), drei (`external/Performance/Bvh/Bvh.tsx:55,115`) |
| 1.2 | `MeshBVH.raycastFirst(ray, materialOrSide, near, far)` | have | `CpuMeshBvh.raycastFirst` takes a ray (`[x, y, z]` arrays or a three `Ray`) and a side, `{ side, near, far }` or a material; `BvhBoundsTree.raycastFirst(ray, DoubleSide)` returns a three intersection with `face.normal`, as csg reads it (`src/three/bounds-tree.test.ts`) | csg (`core/operations/operationsUtils.js:50`, `raycastFirst(_ray, DoubleSide)`) |
| 1.3 | `shapecast` (`intersectsBounds`, `intersectsTriangle`, `boundsTraverseOrder`, ...) | have | `CpuMeshBvh.shapecast({ intersectsBounds, intersectsTriangle, intersectsRange, boundsTraverseOrder })` (`src/query/cpu-mesh-bvh.ts`), on the one box walk (`traverse`): the triangle as a scratch `ExtendedTriangle`, its original index, `contained`, `depth`; `BvhBoundsTree.shapecast` hands over `Box3` and the three-flavoured `ExtendedTriangle`. Against three-mesh-bvh on oriented boxes, spheres and homefig's capsule pattern: the same triangles and closest points (`src/three/shapecast.test.ts`) | homefig (`editor/.../first-person/bvh-ecctrl.tsx:440,547`) |
| 1.4 | `closestPointToPoint` | have | `CpuMeshBvh.closestPointToPoint(point, target, minThreshold, maxThreshold)` (`src/query/cpu-mesh-bvh.ts`): three-mesh-bvh's shapecast step for step (a flat walk since 2026-11-13, the same answers bit for bit), `point` / `distance` / `faceIndex` (original triangle); `BvhBoundsTree`'s returns a `Vector3`. Against three-mesh-bvh: the same distance on 800 random points (`src/three/closest-point.test.ts`) and on 4,000 in a 270k-triangle room, the same triangle unless tied; 0.32x its time in the open, 0.36x near surfaces (`bench/closest-point.ts`, 2026-11-13). One departure: on slivers (the squared sine of the angle at `a` at most 1e-7) the nearer of three's point and the edges' nearest point is kept, as the GPU's `bvhClosestPointToTriangle` does, because three's regions cancel there (a point on a sliver was reported 2.2e-3 away). `maxThreshold` culls boxes, not triangles, as three-mesh-bvh's does (accepted 2026-11-13) | None (see note 2) |
| 1.5 | `closestPointToGeometry` | have | `CpuMeshBvh.closestPointToGeometry(treeOrMesh, geometryToBvh, target1, target2, minThreshold, maxThreshold)`; `BvhBoundsTree`'s takes a `BufferGeometry` (its `boundsTree` when it is ours). three-mesh-bvh's two nested shapecasts; both `faceIndex`es are original triangles in every mode (three-mesh-bvh 0.9.18's indirect mode gives tree positions, which looks like a bug there; ours accepted 2026-11-13). The same distances on 85 random pairs and 200 room placements (1.08x its time) | None |
| 1.6 | `intersectsGeometry`, `intersectsBox`, `intersectsSphere` | have | `CpuMeshBvh.intersectsBox(box, boxToMesh)`, `intersectsSphere(sphere)`, `intersectsGeometry(treeOrMesh, geometryToBvh)`; `BvhBoundsTree.intersectsGeometry(geometry, matrix)` uses `geometry.boundsTree` when it is ours, else the bare triangles. Same booleans as three-mesh-bvh (`src/three/shapecast.test.ts`). `intersectsGeometry` against a tree runs a `bvhcast` where three-mesh-bvh 0.9.18 shapecasts the other tree from each leaf: the same answers, 1.5x faster here (accepted 2026-11-13) | None directly. homefig gets the box test through `shapecast` (`intersectsBounds: box.intersectsBox`) |
| 1.7 | `bvhcast` (tree against tree) | have | `CpuMeshBvh.bvhcast(other, matrixToLocal, { intersectsRanges, intersectsTriangles })`: three-mesh-bvh's walk step for step on our nodes, tree positions as in indirect mode; `BvhBoundsTree.bvhcast` hands over three-flavoured `ExtendedTriangle`s. three-bvh-csg's pattern finds the same pairs and segments on random mesh pairs (`src/three/bvhcast.test.ts`) and on a 270k-triangle room (`bench/query-parity.ts`: 100 / 100, 0.94x three-mesh-bvh's time). `intersectsGeometry` uses it against a tree. The depth bookkeeping is three-mesh-bvh's, copied as it is (accepted 2026-11-13) | csg (`core/operations/operationsUtils.js:70`, the boolean-op core) |

Not counted: `BVH.getBoundingBox` (now `CpuMeshBvh.getBoundingBox`, which writes a three `Box3` too) and `shiftPrimitiveOffsets` (none).

## 2. Shapes for queries

| # | three-mesh-bvh | Status | Ours / what is missing | Who needs it |
| --- | --- | --- | --- | --- |
| 2.1 | `ExtendedTriangle` (`intersectsTriangle`, `closestPointToSegment`, `distanceToPoint`, `distanceToTriangle`, `intersectsSphere`) | have | Engine-free `ExtendedTriangle` and the math as functions over plain points (`src/query/shapes.ts`, from `@pmndrs/bvh`); `/three`'s `ExtendedTriangle` extends three's `Triangle` and calls the same functions (`src/three/shapes.ts`). Bit-for-bit equal to three-mesh-bvh's on 4,000 random pairs with coplanar and degenerate cases (`src/three/shapes.test.ts`). `ExtendedTriangle.set` and `copy` set `needsUpdate` (accepted 2026-11-13) | csg (`core/CDTTriangleSplitter.js:2,140,153`, `core/LegacyTriangleSplitter.js:2,16`), homefig (`closestPointToSegment` on the triangle that `shapecast` hands over, `bvh-ecctrl.tsx:443,550`) |
| 2.2 | `OrientedBox` | have | Engine-free `OrientedBox` (`src/query/shapes.ts`) and `/three`'s (`Vector3` / `Matrix4` fields): `intersectsBox`, `intersectsTriangle`, `closestPointToPoint`, `distanceToPoint`, `distanceToBox`. Bit-for-bit equal to three-mesh-bvh's on 3,000 random cases. three-mesh-bvh 0.9.18's `distanceToBox` with targets throws on overlapping boxes under three 0.187 (`Box3.closestPointToPoint` is gone); ours doesn't. `/three`'s `OrientedBox` is its own class, not a `Box3` subclass (accepted 2026-11-13) | None |

## 3. Primitives

| # | three-mesh-bvh | Status | Ours / what is missing | Who needs it |
| --- | --- | --- | --- | --- |
| 3.1 | Triangles (`MeshBVH`) | have | Every builder and walk (`src/index.ts`, `src/trace/`) | all |
| 3.2 | `PointsBVH` | missing | The layout has leaves of triangles only (`src/gpu-layout.ts`) | None (drei's `Points.tsx` has its own raycast, no BVH) |
| 3.3 | `LineSegmentsBVH`, `LineLoopBVH`, `LineBVH` | missing | as above | None |
| 3.4 | `SkinnedMeshBVH` (live skinned pose) | partial | GPU: `SceneTreesObject3DSync` with a `renderer` poses `SkinnedMesh` and morph-target meshes each update (`BvhPosePass`, a TSL compute pass with three's math, into `SceneTrees.setPose`; the top-level box follows the skeleton). Checked on the `three-skinned` GPU page against three's CPU pose. CPU picking falls back to three's raycast (`src/three/raycast.ts`), which is exact. Accepted limits (2026-11-13): the deform box is one per mesh, set from the CPU (`BvhPosePass.bounds`), with no GPU readback of the refit root; per-instance morphs (`InstancedMesh.morphTexture`) aren't read | None |

## 4. Build options

| # | three-mesh-bvh | Status | Ours / what is missing | Who needs it |
| --- | --- | --- | --- | --- |
| 4.1 | `strategy`: CENTER, AVERAGE, SAH | have | `computeBoundsTree({ strategy })` and `new BvhBoundsTree(geometry, { strategy })` take 0, 1 or 2 (`CENTER`, `AVERAGE`, `SAH` are exported from `/three`) and build the same tree for each: tinybvh's default builder (binned SAH over a grid of cells). There is no CENTER or AVERAGE builder; a centre split would only be a worse tree (accepted 2026-11-13). Other values throw. Mapping in `docs/internals/three.md`, "Build options". Other builders: GPU H-PLOC and LBVH (`src/gpu-hploc.ts`, `src/gpu-lbvh.ts`) | drei (`Bvh.tsx:5`, imports `SAH` / `SplitStrategy` for its options), pathtracer (`core/PathTracingSceneGenerator.js:200`, `strategy: SAH`), upscaler (`RealtimeRadianceCascades.js:627`, `strategy: SAH`) |
| 4.2 | `maxLeafTris` / `maxLeafSize` / `targetLeafSize` (default 10) | have | All three names map to `maxLeafPrimitives`, clamped to the layout's 1 to 8 (10 gives 8; `targetLeafSize` wins). tinybvh's SAH may make smaller leaves; larger ones are cut in half | homefig (`viewer/src/systems/wall/wall-system.tsx:37`, `roof-system.tsx:32`: `maxLeafSize: 10`), csg (`core/Brush.js:84`: 3), pathtracer (`PathTracingSceneGenerator.js:201`: 1), upscaler (`:627`: `targetLeafSize: 4`) |
| 4.3 | `maxDepth` | partial | Accepted, not honoured (Dennis accepted this on 2026-11-13): tinybvh decides the depth (`layout.maxDepth` reports it). The CPU ray walk's stack holds 64 entries (`TRACE_STACK_SIZE`) and throws a `RangeError` past it; `traverse` has no limit. `GPU_TREE_MAX_DEPTH` (`src/scene/scene-trees.ts`) bounds GPU-built trees | drei passes 40 (`Bvh.tsx:48`) |
| 4.4 | `indirect` (leave the geometry's index alone) | have | Always, in effect: the triangle records are copies in tree order, each carrying its original index, and the geometry is never touched. `indirect` is accepted and ignored. `resolveTriangleIndex(treePosition)` gives the original index (what three-mesh-bvh's indirect mode returns); raycast hits already carry it. No `indirectBuffer` | csg (`core/Brush.js:84`, `operationsUtils.js:84-85`), pathtracer (`PathTracingSceneGenerator.js:203`) |
| 4.5 | Geometry groups and draw ranges | have | `threeTriangleRanges`, one mesh per group (`src/three/scene.ts:165`); `docs/internals/three.md` "Scenes". Ranges off a multiple of 3, ending mid-triangle or overlapping are read as three steps through them, in the scene and in the CPU raycast (shifted trees); tested against three's raycast. One difference from three, accepted 2026-11-13: three r187 reports the triangle that reads past the end of the index at a `NaN` distance (and leaves its sorted list out of order); we report nothing there. A known three behaviour, recorded here, not filed upstream | None found (homefig `item-renderer.tsx:133` mentions a crash on array length; unverified) |
| 4.6 | `range` (build from a triangle range) | have | `SceneMeshOptions` and `GpuMeshTreeRange` (`src/scene/scene-trees.ts:230`, `src/mesh-input.ts:86`). Unverified that it matches `range` exactly | None |
| 4.7 | `setBoundingBox`, `verbose`, `onProgress` | partial | `onProgress` is called with 0 and 1 only: tinybvh builds in one WebAssembly call and reports nothing between. `setBoundingBox` defaults to true, as in three-mesh-bvh: it sets `geometry.boundingBox` from the tree when the geometry has none, and `boundsTree.refit()` keeps that box current (`src/three/raycast.ts`). `verbose` is accepted and ignored | pathtracer (`PathTracingSceneGenerator.js:113,202`, `onProgress`), drei (`setBoundingBox: true`, `verbose: false`) |
| 4.8 | `useSharedArrayBuffer` | won't | Our trees are copied to the GPU; no SharedArrayBuffer is needed, and the worker pool needs no cross-origin isolation (`src/worker/pool.ts`) | csg (`core/Brush.js:84`; a build flag only) |

## 5. Scene level

| # | three-mesh-bvh | Status | Ours / what is missing | Who needs it |
| --- | --- | --- | --- | --- |
| 5.1 | `ObjectBVH` as the GPU scene input (`BVHComputeData`) | have | `SceneTrees` with `SceneTreesObject3DSync` and the GPU top level (`src/scene/scene-trees.ts`, `src/three/scene-trees-sync.ts`) | Denoiser (`reflections-rooms.ts:300`, `shadows-split.ts:311-315`), upscaler (`RealtimeRadianceCascades.js:629`), ddgi-research (`src/provider.js:10`), reflections (`three/src/trace/MeshBVHTraceProvider.ts:492`, which already has a sibling `PmndrsBvhTraceProvider.ts`) |
| 5.2 | `ObjectBVH.raycast(raycaster)` (CPU scene raycast) | have | `raycastScene(picker, raycaster)` / `BvhSceneRaycaster.raycast` in `/three` (`src/three/scene-raycast.ts`) returns three's `intersectObject(root, true)` from one CPU walk over the top level (`CpuSceneBvh`, `src/query/cpu-scene.ts`); tested against three's raycast. Layers 8-31 and skinned poses differ (`docs/internals/three.md`). Accepted 2026-11-13: it is built from the three objects, not from `SceneTrees` (whose trees are GPU-only); layers 8-31 aren't seen; hidden objects are hit by default, as three's `Raycaster` hits them | None (homefig and drei raycast per mesh) |
| 5.3 | `ObjectBVH.shapecast` | missing | | None found |
| 5.4 | `InstancedMesh` | have | `bvhSceneFromObject3D` (instances below `count`, `src/three/scene.ts`); raycast through `Mesh.raycast` (`docs/internals/three.md`) | None found (homefig `bvh-ecctrl.tsx:248` excludes `InstancedMesh` from colliders; unverified why) |
| 5.5 | `BatchedMesh` and `computeBatchedBoundsTree(index)` | partial | Scene read and raycast work (`docs/internals/three.md`: call `batchedMesh.geometry.computeBoundsTree()`). No per-geometry-index `computeBatchedBoundsTree` | None |
| 5.6 | Frustum culling over the object BVH | have | `BvhSceneRaycaster.intersectsFrustum(frustum, layers)` and `CpuSceneBvh.intersectsFrustum(planes)` / `intersectsBox(box)` walk the top level; tested against brute force over every world box | None |

## 6. Deforming meshes, three integration

| # | three-mesh-bvh | Status | Ours / what is missing | Who needs it |
| --- | --- | --- | --- | --- |
| 6.1 | `refit()` | have | `CpuMeshBvh.refit(input?)` rewrites the triangle records from the source positions and runs `refitOnCpu` (`src/refit-cpu.ts`); `geometry.boundsTree.refit()` rereads the position attribute and the raycast uses the tree again. GPU refit: `src/gpu-mesh-refit.ts`, `SceneTrees.setPose`, `src/scene/deforming-trees.ts` | pathtracer (`core/PathTracingSceneGenerator.js:219`), ddgi-research (own packer, `src/scene.js:49`, `src/provider.js:83`), reflections (`MeshBVHTraceProvider.ts:507`) |
| 6.2 | `computeBoundsTree`, `disposeBoundsTree`, `acceleratedRaycast`, `firstHitOnly` | have | `src/three/raycast.ts` (`prepareBoundsTrees()` is awaited once first). `geometry.boundsTree` is a `BvhBoundsTree`, the core `CpuMeshBvh` with three's types. Options: rows 4.1-4.7. `docs/internals/three.md` has the bench | homefig (`scene-bvh.tsx:78-116`, `build-collider-world.ts:12,288-297`, `first-person-controls.tsx:47,323-332`), drei (`Bvh.tsx:5,34,55-62,111-124`) |
| 6.3 | `StaticGeometryGenerator` (skinned and morphed meshes to a static geometry) | partial | GPU: `BvhPosePass` (`src/three/deform.ts`) poses into a GPU buffer that `SceneTrees.setPose` refits from, nothing read back. No CPU static geometry (three's raycast is the CPU path) | pathtracer (own copy, note 4: `core/PathTracingSceneGenerator.js:3,94`) |
| 6.4 | `MeshBVHHelper` / `BVHHelper` | have | `BVHHelper` and `MeshBVHHelper` in `/three` (`src/three/helper.ts`): three-mesh-bvh's `Group` with `depth`, `displayParents`, `displayEdges`, `color`, `opacity`, `instanceId`, `edgeMaterial` / `meshMaterial`, `update()`, `dispose()`, the constructor forms `(mesh or bvh, depth)` and `(mesh, bvh, depth)`, following the mesh's transform. The boxes and indices come from the core (`bvhHelperPositions`, `bvhHelperIndices`). Segment counts tested in Node against the tree's node counts at each depth (`src/three/helper.test.ts`); not drawn in a browser yet. No `BatchedMesh` per-geometry tree (`boundsTrees`), as row 5.5. `MeshBVHHelper` is exported without three-mesh-bvh's deprecation warning (accepted 2026-11-13) | drei (`Bvh.stories.tsx:5,29`, story only); csg: none in `src` |
| 6.5 | `getTriangleHitPointInfo` | have | `getTriangleHitPointInfo(point, geometry, triangleIndex, target)` in `/three` (tested against three-mesh-bvh's), and `CpuMeshBvh.getTriangleHitPointInfo(point, triangle, { attributes })` in the core: barycentrics, corners, face normal and any attributes interpolated | None |

## 7. Utilities

| # | three-mesh-bvh | Status | Ours / what is missing | Who needs it |
| --- | --- | --- | --- | --- |
| 7.1 | `MeshBVH.serialize` / `deserialize` | partial | `CpuMeshBvh.serialize()` / `CpuMeshBvh.deserialize(bytes)`, `deserializeBoundsTree(bytes, geometry)` in `/three`, over `serializeLayout` (`src/serialize.ts`). Our layout's format, not three-mesh-bvh's, and no `setIndex` | None |
| 7.2 | `estimateMemoryInBytes` | have | `CpuMeshBvh.estimateMemoryInBytes()` per tree (layout, top level, lookups; not the source geometry), and three-mesh-bvh's function `estimateMemoryInBytes(tree)` over it (`src/query/debug.ts`, from `@pmndrs/bvh` and `/three`); `SceneTrees.memory` per scene (`src/scene/scene-trees.ts`) | None |
| 7.3 | `getBVHExtremes` | have | `getBVHExtremes(tree)` (`src/query/debug.ts`, from `@pmndrs/bvh` and `/three`): three-mesh-bvh's output (`nodeCount`, `leafNodeCount`, `depth`, `primitives`, `splits`, `surfaceAreaScore` with its costs), one entry for our one root, over `CpuMeshBvh.traverseNodes` (three-mesh-bvh's node walk). The split axis isn't stored in our layout: `splits` counts the axis the children's centres lie furthest apart on (accepted 2026-11-13). `treeSahCost` (`src/layout-check.ts`) is the normalised SAH cost | None |
| 7.4 | `validateBounds` | have | `validateBounds(tree)` (`src/query/debug.ts`, from `@pmndrs/bvh` and `/three`): three-mesh-bvh's checks and `console.assert` messages, reading the source geometry (`writePrimitiveRangeBounds`), so edits without `refit()` fail as in three-mesh-bvh (accepted 2026-11-13). `checkGpuLayout` checks every child box bit for bit, plus the tree's shape (`src/layout-check.ts`) | None |
| 7.5 | `getJSONStructure` | missing | | None |

## 8. GPU

| # | three-mesh-bvh | Status | Ours / what is missing | Who needs it |
| --- | --- | --- | --- | --- |
| 8.1 | `fns.raycastFirstHit` (WGSL / TSL) | have | `bvhTraceClosest` (WGSL), `BvhTraceNodes.traceClosest` (TSL) (`src/trace/wgsl.ts`, `src/three/trace-nodes.ts`; mapping in `docs/internals/three.md`) | Denoiser (`split/area.ts:42`, `reflections-rooms.ts:313`, `reflections-hybrid.ts:431,465`, `shadows.ts:323`), upscaler (`RealtimeRadianceCascades.js:2067`), ddgi-research (`src/provider.js:54`), reflections (`MeshBVHTraceProvider.ts`) |
| 8.2 | `fns.raycastAnyHit` | have | `bvhTraceAny` / `traceAny` | as 8.1 |
| 8.3 | `BVHComputeData`, `update`, `updateTransforms` | have | `SceneTrees` + `SceneTreesObject3DSync` + `BvhTraceNodes` (`docs/internals/three.md`) | as 5.1 |
| 8.4 | `BVHComputeData.updateAsync` | partial | `SceneTrees.whenBuilt()` / `whenRefined()` (`src/scene/scene-trees.ts:830,836`) | None |
| 8.5 | `fns.closestPointToPoint` (GPU) | have | `bvhClosestPoint(point, maxDistance) -> BvhClosestHit` (distance, instance, triangle, material, barycentrics, point, normal) and `bvhClosestPointMasked` in the trace module with `queries: true` (`src/trace/query-wgsl.ts`), over the whole scene (top level and mesh trees, instanced or merged, every node and triangle format), in world space through scaled, rotated and mirrored instances; `ClosestPointBatchKernel` (points in, 48-byte records out; `src/trace/trace-gpu.ts`); TSL `BvhTraceNodes.closestPoint` with `queries: true`. Differences: scene-wide rather than one mesh; `maxDistance` culls triangles too (three-mesh-bvh's `maxThreshold` culls boxes only); no `minThreshold` (the walk stops at distance 0). Checked against `CpuMeshBvh.closestPointToPoint` by `tests/gpu/pages/query-oracle` (2026-11-13) | None found |
| 8.6 | `getShapecastFn` (GPU shapecast) | partial | `traceModuleWgsl({ shapecast })`: one shapecast walk compiled around the caller's WGSL hooks `bvhShapecastBounds` (a box's score; negative skips it, lower first), `bvhShapecastKeep` (re-asked on pop, for distance culling) and `bvhShapecastTriangle` (world-space corners, instance, triangle; true stops), as `bvhTrianglePasses` is compiled into the ray walk; `bvhShapecast()` / `bvhShapecastMasked(mask)`. Helpers built on it: `bvhIntersectsSphere`, `bvhIntersectsBox`, `bvhIntersectsOrientedBox` (`queries: true`), checked against `intersectsSphere` / `intersectsBox` by `query-oracle`. Partial: the hooks are WGSL, not TSL callbacks as `getShapecastFn` takes (`BvhTraceNodes` passes `shapecast` through to its module); no `intersectsRange` or `contained` | None found |
| 8.7 | `fns.sampleTrianglePoint` | won't | Out of scope: the package ends at the hit (HANDOFF "Rules", `docs/internals/README.md`). Interpolating attributes at a point on a triangle belongs to the renderer or the tracing layer above the package; a hit gives `triangle` and `barycentrics` for it (2026-11-13) | None found |
| 8.8a | WGSL helpers `intersectRayTriangle`, `closestPointToTriangle` | have | `bvhIntersectRayTriangle(ray, a, b, c) -> BvhTriangleHit` (the walk's own watertight test, `bvhMakeRaySetup` + `bvhIntersectWatertight`, not a copy) and `bvhClosestPointToTriangle(point, a, b, c) -> BvhTrianglePoint` (three's `Triangle.closestPointToPoint`, with barycentrics), in every trace module and binding-free in `triangleGeometryWgsl()` (`src/trace/wgsl.ts`); checked by `query-oracle` (2026-11-13) | None found |
| 8.8b | WGSL helper `ndcToCameraRay` | won't | Out of scope: making camera rays is the renderer's job, above the package, which ends at the hit (2026-11-13) | None found |
| 8.9 | `ClusteredBVH` | missing | | None found |
| 8.10 | SDF generation (GPU distance queries, README example) | missing | Unblocked (2026-11-13): 8.5 and 8.6 exist. A signed field also needs an inside test (ray parity or winding number), not built. The `three/sdf` example (HANDOFF) can start on `bvhClosestPoint` | None found |
| 8.11 | `wgslTagFn`, `rayStruct`, `rayIntersectionResultStruct` | have | three's `wgslFn(code, [bvh.module])`, `BvhRay`, `BvhHit` (mapping table in `docs/internals/three.md`) | Denoiser (`shadows.ts:36`), upscaler (`RealtimeRadianceCascades.js:14`), ddgi-research (`tests/gpu.js:5`) |
| 8.12 | GLSL and WebGL2: `MeshBVHUniformStruct`, `BVHShaderGLSL`, `shaderStructs`, `shaderIntersectFunction`, `VertexAttributeTexture` classes | won't | WebGL2 isn't a target (HANDOFF "Rules") | drei (`legacy/Materials/MeshRefractionMaterial/MeshRefractionMaterial.tsx:8,20,181,255`), pathtracer (`materials/pathtracing/PhysicalPathTracingMaterial.js:4,87`, `materials/surface/AmbientOcclusionMaterial.js:3,49`, `uniforms/FloatAttributeTextureArray.js:2`) |

## 9. Workers

| # | three-mesh-bvh | Status | Ours / what is missing | Who needs it |
| --- | --- | --- | --- | --- |
| 9.1 | `GenerateMeshBVHWorker.generate(geometry, options)` | have | `new GenerateMeshBvhWorker().generate(geometry, options)` in `/three` resolves to a `BvhBoundsTree` (`src/three/generate.ts`); `generateCpuMeshBvh(pool, input, options)` in `/worker` to a `CpuMeshBvh` (`src/worker/generate.ts`). Both build on `createBvhWorkerPool` (inline in Node). Unlike three-mesh-bvh's, the geometry's arrays are copied, not moved, and several builds may run at once | pathtracer (`core/PathTracingSceneGenerator.js:107-117,209`: `setBVHWorker`, `generateAsync`) |
| 9.2 | `ParallelMeshBVHWorker` (SharedArrayBuffer, parallel single-tree build) | won't | Needs cross-origin isolation. A large single mesh goes to the GPU H-PLOC builder instead (`src/gpu-hploc.ts`). Reconsider if a CPU-only user needs it | None |

## Counts

52 rows: **33 have, 8 partial, 6 missing, 5 won't** (4.8, 8.7, 8.8b, 8.12, 9.2). On 2026-11-13 the GPU queries landed: 8.5 and 8.8a have, 8.6 partial (WGSL hooks, not TSL callbacks); 8.10 is unblocked. Before them: 31 have, 7 partial, 9 missing. Row 8.8 was split on 2026-11-13 (8.8a geometry; 8.8b `ndcToCameraRay`, won't), and 8.7 became won't under the scope rule (the package ends at the hit). At the audit, before items 1 to 5: 12 have, 13 partial, 23 missing (51 rows). Every "won't" has its reason in its row; P8's done-when also wants them in the "Coming from three-mesh-bvh" docs page.

## Proposed build-out order

Weighted by how many surveyed projects need a feature, and by effort (S: days or less, M: about a week, L: more). Sizes are rough estimates.

P8's order was: Q2 closest point and shapecast plus shapes; the drop-in three integration; object BVH queries and frustum culling; GPU closest point and shapecast; indirect mode and groups; points and lines. The evidence changes it as follows:

- The drop-in three integration is already there (6.2). What is left is its options, which homefig, drei, pathtracer and upscaler all pass. That is small and has the most callers, so it goes first.
- Object BVH queries, frustum culling, and every GPU feature past the walk have no call site in the survey. They drop to the end.
- Groups are done (4.5); only the indirect API matters, for two projects.
- Points and lines have no call site.

| Order | Item | Rows | Size | Unblocks |
| --- | --- | --- | --- | --- |
| 1 (done) | Pass `strategy`, `maxLeafSize` / `targetLeafSize` and `maxDepth` through `computeBoundsTree` and the builders, mapping SAH to the SAH-quality builder and tolerating the other names. Add `onProgress`. | 4.1, 4.2, 4.3, 4.7 | S | homefig, drei, pathtracer, upscaler (all already pass these options) |
| 2 (done) | Q2 shapecast on the CPU tree, with `ExtendedTriangle` (math-only core, three-typed wrapper in `/three`) and `OrientedBox`, plus `intersectsBox` / `intersectsSphere` / `intersectsGeometry` on top of shapecast | 1.3, 1.6, 2.1, 2.2 | M | homefig (character controller), csg (the triangle class) |
| 3 (done) | Public CPU tree object: `raycastFirst(ray, side)`, `refit()` on `BvhBoundsTree` (wiring in `refitOnCpu`), the indirect API (`resolveTriangleIndex`), `getTriangleHitPointInfo` | 1.2, 4.4, 6.1, 6.5 | M | csg (`raycastFirst`, indirect), pathtracer (`refit`, indirect) |
| 4 (done) | `bvhcast` (tree against tree) | 1.7 | M | csg (booleans) |
| 5 (done) | `generate(geometry, options)` drop-in over the worker pool | 9.1 | S | pathtracer (`setBVHWorker`) |
| 6 (done) | Q2 `closestPointToPoint` and `closestPointToGeometry` on the CPU | 1.4, 1.5 | M | no surveyed call site; the P8 `distancecast` compare and SDF need it. Lower than its P8 slot |
| 7 (done) | `MeshBVHHelper` in `/three` | 6.4 | S | drei (story only) |
| 8 | `StaticGeometryGenerator` | 6.3 | M | pathtracer (has its own) |
| 9 | CPU object BVH queries: scene `raycast`, `shapecast`, frustum culling | 5.2, 5.3, 5.6 | M | none found; P8 lists it |
| 10 (part done) | GPU closest point and shapecast in WGSL, with `intersectRayTriangle` and `closestPointToTriangle` (done 2026-11-13), then SDF (needs an inside test); `ClusteredBVH` after (`sampleTrianglePoint` and `ndcToCameraRay` won't: out of scope) | 8.5, 8.6, 8.8a, 8.9, 8.10 | L | none found (needs a user first, e.g. the SDF compare) |
| 11 | Points, lines, `SkinnedMeshBVH` | 3.2-3.4 | L | none found |
| 12 (part done) | Per-tree utilities: `estimateMemoryInBytes` and `getBVHExtremes` (done); three-mesh-bvh-format `serialize`, `getJSONStructure` (left) | 7.1-7.3, 7.5 | S | none found |

A project's full switch waits on every item in its column:
- csg needs nothing more (2, 3 and 4 are done). Its `Brush` builds `new MeshBVH(geometry, { maxLeafSize: 3, indirect: true, useSharedArrayBuffer })`, which `/three`'s `MeshBVH` accepts.
- homefig needs nothing more (1 and 2 are done).
- drei needs nothing more (1 and 7 are done; its story imports `MeshBVHHelper`, which `/three` exports). Its refraction material is GLSL (won't, 8.12).
- pathtracer's CPU side has what it needs (1, 3 and 5 are done). Its shaders are GLSL (won't).
- Denoiser, upscaler, ddgi-research and reflections need nothing beyond what is on `main` (rows 8.1 to 8.3 and 5.1), apart from the switch itself (`docs/internals/three.md`, "Switching the known consumers").

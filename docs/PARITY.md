# Parity with three-mesh-bvh 0.9.18 (BX.16)

The audit P8 asks for (`docs/design/bvh-package-split.md`, P8). It compares our package, as of `main` on 2026-11-11, with three-mesh-bvh 0.9.18 as installed in `node_modules/three-mesh-bvh`. The sources are its `src/index.js`, `src/index.d.ts`, `src/webgpu/index.js` and `index.d.ts`, `src/workers/index.js` and `index.d.ts`, and its README example list (the package ships no `example/` folder).

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
| 1.1 | `MeshBVH.raycast` (all hits) | have | `acceleratedRaycast` and `BvhBoundsTree`, same fields as three's `Mesh.raycast` (`src/three/raycast.ts`); `createCpuTracer().closest/any` for bare rays (`src/trace/cpu-traversal.ts`) | homefig (`viewer/src/components/viewer/scene-bvh.tsx:84`, `editor/.../first-person-controls.tsx:332`), drei (`external/Performance/Bvh/Bvh.tsx:55,115`) |
| 1.2 | `MeshBVH.raycastFirst(ray, materialOrSide, near, far)` | partial | The first-hit path is there (`firstHitOnly`, row 6.2). No method on a tree object that takes a `Ray` and a `Side`: `CpuTracer.closest` takes our `TraceRay` (`src/trace/rays.ts`), and `side` is read per material inside `src/three/raycast.ts` | csg (`core/operations/operationsUtils.js:50`, `raycastFirst(_ray, DoubleSide)`) |
| 1.3 | `shapecast` (`intersectsBounds`, `intersectsTriangle`, `boundsTraverseOrder`, ...) | missing | Q2 | homefig (`editor/.../first-person/bvh-ecctrl.tsx:440,547`) |
| 1.4 | `closestPointToPoint` | missing | Q2. Our reference tracer (`src/trace/reference-tracer.ts`) only casts rays | None (see note 2) |
| 1.5 | `closestPointToGeometry` | missing | Q2 | None |
| 1.6 | `intersectsGeometry`, `intersectsBox`, `intersectsSphere` | missing | Q2 | None directly. homefig gets the box test through `shapecast` (`intersectsBounds: box.intersectsBox`) |
| 1.7 | `bvhcast` (tree against tree) | missing | | csg (`core/operations/operationsUtils.js:70`, the boolean-op core) |

Not counted: `BVH.getBoundingBox` and `shiftPrimitiveOffsets`. The root box is in every tree's layout (`src/gpu-layout.ts`), but no public method was found: unverified.

## 2. Shapes for queries

| # | three-mesh-bvh | Status | Ours / what is missing | Who needs it |
| --- | --- | --- | --- | --- |
| 2.1 | `ExtendedTriangle` (`intersectsTriangle`, `closestPointToSegment`, `distanceToPoint`, `distanceToTriangle`, `intersectsSphere`) | missing | A math-only version would keep the core engine-free; the three-typed one would live in `/three` | csg (`core/CDTTriangleSplitter.js:2,140,153`, `core/LegacyTriangleSplitter.js:2,16`), homefig (`closestPointToSegment` on the triangle that `shapecast` hands over, `bvh-ecctrl.tsx:443,550`) |
| 2.2 | `OrientedBox` | missing | | None |

## 3. Primitives

| # | three-mesh-bvh | Status | Ours / what is missing | Who needs it |
| --- | --- | --- | --- | --- |
| 3.1 | Triangles (`MeshBVH`) | have | Every builder and walk (`src/index.ts`, `src/trace/`) | all |
| 3.2 | `PointsBVH` | missing | The layout has leaves of triangles only (`src/gpu-layout.ts`) | None (drei's `Points.tsx` has its own raycast, no BVH) |
| 3.3 | `LineSegmentsBVH`, `LineLoopBVH`, `LineBVH` | missing | as above | None |
| 3.4 | `SkinnedMeshBVH` (live skinned pose) | missing | Skinned meshes use the rest pose; `/three` falls back to three's raycast (`src/three/raycast.ts:352`). The GPU side takes poses through `SceneTrees.setPose` (row 6.3) | None |

## 4. Build options

| # | three-mesh-bvh | Status | Ours / what is missing | Who needs it |
| --- | --- | --- | --- | --- |
| 4.1 | `strategy`: CENTER, AVERAGE, SAH | partial | Builders in place of strategies: tinybvh (`src/tinybvh.ts`), GPU H-PLOC and LBVH (`src/gpu-hploc.ts`, `src/gpu-lbvh.ts`), a CPU SAH builder (`src/trace/sah-builder.ts`). No `strategy` option on `computeBoundsTree`; its options take only `builder` (`BvhBoundsTreeOptions`, `src/three/raycast.ts:85`). `treeSahCost` measures quality (`src/layout-check.ts:256`) | drei (`Bvh.tsx:5`, imports `SAH` / `SplitStrategy` for its options), pathtracer (`core/PathTracingSceneGenerator.js:200`, `strategy: SAH`), upscaler (`RealtimeRadianceCascades.js:627`, `strategy: SAH`) |
| 4.2 | `maxLeafTris` / `maxLeafSize` / `targetLeafSize` (default 10) | partial | `maxLeafPrimitives`, 1 to 8, default 8 (`src/tinybvh.ts:112`, `src/mesh-input.ts:76`); not passed through `computeBoundsTree` | homefig (`viewer/src/systems/wall/wall-system.tsx:37`, `roof-system.tsx:32`: `maxLeafSize: 10`), csg (`core/Brush.js:84`: 3), pathtracer (`PathTracingSceneGenerator.js:201`: 1), upscaler (`:627`: `targetLeafSize: 4`) |
| 4.3 | `maxDepth` | partial | A hard limit exists, `GPU_TREE_MAX_DEPTH` (`src/scene/scene-trees.ts:86`); no user option | None |
| 4.4 | `indirect` (leave the geometry's index alone) | partial | Our triangle records copy the vertices and carry the original index (`checkGpuLayout` verifies it, `src/layout-check.ts`), so the geometry is never reordered. No `indirectBuffer` and no `resolveTriangleIndex` | csg (`core/Brush.js:84`, `operationsUtils.js:84-85`), pathtracer (`PathTracingSceneGenerator.js:203`) |
| 4.5 | Geometry groups and draw ranges | have | `threeTriangleRanges`, one mesh per group (`src/three/scene.ts:165`); `docs/three.md` "Scenes". Gap: a range that does not start on a multiple of 3 (HANDOFF "Still to do in /three") | None found (homefig `item-renderer.tsx:133` mentions a crash on array length; unverified) |
| 4.6 | `range` (build from a triangle range) | have | `SceneMeshOptions` and `GpuMeshTreeRange` (`src/scene/scene-trees.ts:230`, `src/mesh-input.ts:86`). Unverified that it matches `range` exactly | None |
| 4.7 | `setBoundingBox`, `verbose`, `onProgress` | missing | A build reports no progress; the worker pool has stats (`src/worker/pool.ts`) | pathtracer (`PathTracingSceneGenerator.js:113,202`, `onProgress`) |
| 4.8 | `useSharedArrayBuffer` | won't | Our trees are copied to the GPU; no SharedArrayBuffer is needed, and the worker pool needs no cross-origin isolation (`src/worker/pool.ts`) | csg (`core/Brush.js:84`; a build flag only) |

## 5. Scene level

| # | three-mesh-bvh | Status | Ours / what is missing | Who needs it |
| --- | --- | --- | --- | --- |
| 5.1 | `ObjectBVH` as the GPU scene input (`BVHComputeData`) | have | `SceneTrees` with `SceneTreesObject3DSync` and the GPU top level (`src/scene/scene-trees.ts`, `src/three/scene-trees-sync.ts`) | Denoiser (`reflections-rooms.ts:300`, `shadows-split.ts:311-315`), upscaler (`RealtimeRadianceCascades.js:629`), ddgi-research (`src/provider.js:10`), reflections (`three/src/trace/MeshBVHTraceProvider.ts:492`, which already has a sibling `PmndrsBvhTraceProvider.ts`) |
| 5.2 | `ObjectBVH.raycast(raycaster)` (CPU scene raycast) | missing | The CPU walk covers a `TraceSceneData` with instances (`createCpuTracer`, `src/trace/cpu-traversal.ts:138`), but there is no `Raycaster` entry point over a scene | None (homefig and drei raycast per mesh) |
| 5.3 | `ObjectBVH.shapecast` | missing | | None found |
| 5.4 | `InstancedMesh` | have | `bvhSceneFromObject3D` (instances below `count`, `src/three/scene.ts`); raycast through `Mesh.raycast` (`docs/three.md`) | None found (homefig `bvh-ecctrl.tsx:248` excludes `InstancedMesh` from colliders; unverified why) |
| 5.5 | `BatchedMesh` and `computeBatchedBoundsTree(index)` | partial | Scene read and raycast work (`docs/three.md`: call `batchedMesh.geometry.computeBoundsTree()`). No per-geometry-index `computeBatchedBoundsTree` | None |
| 5.6 | Frustum culling over the object BVH | missing | P8 lists it; no example named it in the README list | None |

## 6. Deforming meshes, three integration

| # | three-mesh-bvh | Status | Ours / what is missing | Who needs it |
| --- | --- | --- | --- | --- |
| 6.1 | `refit()` | partial | GPU refit of a mesh tree (`src/gpu-mesh-refit.ts`, `SceneTrees.setPose`, `src/scene/deforming-trees.ts`); `refitOnCpu` works on layout arrays (`src/refit-cpu.ts:197`). Not reachable from `BvhBoundsTree`: the `/three` CPU tree is built once, and an edited position attribute makes it fall back to three (`src/three/raycast.ts`) | pathtracer (`core/PathTracingSceneGenerator.js:219`), ddgi-research (own packer, `src/scene.js:49`, `src/provider.js:83`), reflections (`MeshBVHTraceProvider.ts:507`) |
| 6.2 | `computeBoundsTree`, `disposeBoundsTree`, `acceleratedRaycast`, `firstHitOnly` | have | `src/three/raycast.ts` (`prepareBoundsTrees()` is awaited once first; options take only `builder`, see 4.1 and 4.2). `docs/three.md` has the bench | homefig (`scene-bvh.tsx:78-116`, `build-collider-world.ts:12,288-297`, `first-person-controls.tsx:47,323-332`), drei (`Bvh.tsx:5,34,55-62,111-124`) |
| 6.3 | `StaticGeometryGenerator` (skinned and morphed meshes to a static geometry) | missing | `SceneTrees.setPose` takes vertices you pose yourself | pathtracer (own copy, note 4: `core/PathTracingSceneGenerator.js:3,94`) |
| 6.4 | `MeshBVHHelper` / `BVHHelper` | missing | HANDOFF lists "a helper" as to do in `/three` | drei (`Bvh.stories.tsx:5,29`, story only); csg: none in `src` |
| 6.5 | `getTriangleHitPointInfo` | partial | Our hits carry `barycoord`, `uv`, `uv1`, `normal` and `face` (`docs/three.md` "CPU raycast"), and `threeFaceIndex` exists (`src/three/scene.ts`). No function that computes them for an arbitrary point and triangle | None |

## 7. Utilities

| # | three-mesh-bvh | Status | Ours / what is missing | Who needs it |
| --- | --- | --- | --- | --- |
| 7.1 | `MeshBVH.serialize` / `deserialize` | partial | `serializeLayout` / `deserializeLayout` for our GPU layout (`src/serialize.ts`, `src/index.ts`); not three-mesh-bvh's format, and no `setIndex` | None |
| 7.2 | `estimateMemoryInBytes` | partial | `SceneTrees.memory` (`src/scene/scene-trees.ts:199,792`), per scene; no per-tree call | None |
| 7.3 | `getBVHExtremes` | partial | `treeSahCost` (`src/layout-check.ts:256`) and `checkGpuLayout`'s counts (depth, leaf count); no depth or leaf-size histogram (unverified) | None |
| 7.4 | `validateBounds` | have | `checkGpuLayout` checks every child box bit for bit, plus the tree's shape (`src/layout-check.ts`) | None |
| 7.5 | `getJSONStructure` | missing | | None |

## 8. GPU

| # | three-mesh-bvh | Status | Ours / what is missing | Who needs it |
| --- | --- | --- | --- | --- |
| 8.1 | `fns.raycastFirstHit` (WGSL / TSL) | have | `bvhTraceClosest` (WGSL), `BvhTraceNodes.traceClosest` (TSL) (`src/trace/wgsl.ts`, `src/three/trace-nodes.ts`; mapping in `docs/three.md`) | Denoiser (`split/area.ts:42`, `reflections-rooms.ts:313`, `reflections-hybrid.ts:431,465`, `shadows.ts:323`), upscaler (`RealtimeRadianceCascades.js:2067`), ddgi-research (`src/provider.js:54`), reflections (`MeshBVHTraceProvider.ts`) |
| 8.2 | `fns.raycastAnyHit` | have | `bvhTraceAny` / `traceAny` | as 8.1 |
| 8.3 | `BVHComputeData`, `update`, `updateTransforms` | have | `SceneTrees` + `SceneTreesObject3DSync` + `BvhTraceNodes` (`docs/three.md`) | as 5.1 |
| 8.4 | `BVHComputeData.updateAsync` | partial | `SceneTrees.whenBuilt()` / `whenRefined()` (`src/scene/scene-trees.ts:830,836`) | None |
| 8.5 | `fns.closestPointToPoint` (GPU) | missing | | None found |
| 8.6 | `getShapecastFn` (GPU shapecast) | missing | | None found |
| 8.7 | `fns.sampleTrianglePoint` | missing | | None found |
| 8.8 | WGSL helpers `intersectRayTriangle`, `closestPointToTriangle`, `ndcToCameraRay` | missing | The watertight ray-triangle test sits inside our walk (`src/trace/wgsl.ts`), not exported as a function | None found |
| 8.9 | `ClusteredBVH` | missing | | None found |
| 8.10 | SDF generation (GPU distance queries, README example) | missing | Needs 8.5 and 8.6 | None found |
| 8.11 | `wgslTagFn`, `rayStruct`, `rayIntersectionResultStruct` | have | three's `wgslFn(code, [bvh.module])`, `BvhRay`, `BvhHit` (mapping table in `docs/three.md`) | Denoiser (`shadows.ts:36`), upscaler (`RealtimeRadianceCascades.js:14`), ddgi-research (`tests/gpu.js:5`) |
| 8.12 | GLSL and WebGL2: `MeshBVHUniformStruct`, `BVHShaderGLSL`, `shaderStructs`, `shaderIntersectFunction`, `VertexAttributeTexture` classes | won't | WebGL2 isn't a target (HANDOFF "Rules") | drei (`legacy/Materials/MeshRefractionMaterial/MeshRefractionMaterial.tsx:8,20,181,255`), pathtracer (`materials/pathtracing/PhysicalPathTracingMaterial.js:4,87`, `materials/surface/AmbientOcclusionMaterial.js:3,49`, `uniforms/FloatAttributeTextureArray.js:2`) |

## 9. Workers

| # | three-mesh-bvh | Status | Ours / what is missing | Who needs it |
| --- | --- | --- | --- | --- |
| 9.1 | `GenerateMeshBVHWorker.generate(geometry, options)` | partial | `createBvhWorkerPool` and `createMeshBvhBuilder` (`src/worker/pool.ts`, `src/worker/index.ts`): a pool of single-threaded workers with a Node fallback. No `generate(geometry)` that returns a `MeshBVH`-like tree | pathtracer (`core/PathTracingSceneGenerator.js:107-117,209`: `setBVHWorker`, `generateAsync`) |
| 9.2 | `ParallelMeshBVHWorker` (SharedArrayBuffer, parallel single-tree build) | won't | Needs cross-origin isolation. A large single mesh goes to the GPU H-PLOC builder instead (`src/gpu-hploc.ts`). Reconsider if a CPU-only user needs it | None |

## Counts

51 rows: **12 have, 13 partial, 23 missing, 3 won't** (4.8, 8.12, 9.2). Every "won't" has its reason in its row; P8's done-when also wants them in the "Coming from three-mesh-bvh" docs page.

## Proposed build-out order

Weighted by how many surveyed projects need a feature, and by effort (S: days or less, M: about a week, L: more). Sizes are rough estimates.

P8's order was: Q2 closest point and shapecast plus shapes; the drop-in three integration; object BVH queries and frustum culling; GPU closest point and shapecast; indirect mode and groups; points and lines. The evidence changes it as follows:

- The drop-in three integration is already there (6.2). What is left is its options, which homefig, drei, pathtracer and upscaler all pass. That is small and has the most callers, so it goes first.
- Object BVH queries, frustum culling, and every GPU feature past the walk have no call site in the survey. They drop to the end.
- Groups are done (4.5); only the indirect API matters, for two projects.
- Points and lines have no call site.

| Order | Item | Rows | Size | Unblocks |
| --- | --- | --- | --- | --- |
| 1 | Pass `strategy`, `maxLeafSize` / `targetLeafSize` and `maxDepth` through `computeBoundsTree` and the builders, mapping SAH to the SAH-quality builder and tolerating the other names. Add `onProgress`. | 4.1, 4.2, 4.3, 4.7 | S | homefig, drei, pathtracer, upscaler (all already pass these options) |
| 2 | Q2 shapecast on the CPU tree, with `ExtendedTriangle` (math-only core, three-typed wrapper in `/three`) and `OrientedBox`, plus `intersectsBox` / `intersectsSphere` / `intersectsGeometry` on top of shapecast | 1.3, 1.6, 2.1, 2.2 | M | homefig (character controller), csg (the triangle class) |
| 3 | Public CPU tree object: `raycastFirst(ray, side)`, `refit()` on `BvhBoundsTree` (wiring in `refitOnCpu`), the indirect API (`resolveTriangleIndex`), `getTriangleHitPointInfo` | 1.2, 4.4, 6.1, 6.5 | M | csg (`raycastFirst`, indirect), pathtracer (`refit`, indirect) |
| 4 | `bvhcast` (tree against tree) | 1.7 | M | csg (booleans) |
| 5 | `generate(geometry, options)` drop-in over the worker pool | 9.1 | S | pathtracer (`setBVHWorker`) |
| 6 | Q2 `closestPointToPoint` and `closestPointToGeometry` on the CPU | 1.4, 1.5 | M | no surveyed call site; the P8 `distancecast` compare and SDF need it. Lower than its P8 slot |
| 7 | `MeshBVHHelper` in `/three` | 6.4 | S | drei (story only) |
| 8 | `StaticGeometryGenerator` | 6.3 | M | pathtracer (has its own) |
| 9 | CPU object BVH queries: scene `raycast`, `shapecast`, frustum culling | 5.2, 5.3, 5.6 | M | none found; P8 lists it |
| 10 | GPU closest point and shapecast in WGSL, with `sampleTrianglePoint`, then SDF; `ClusteredBVH` after | 8.5-8.10 | L | none found (needs a user first, e.g. the SDF compare) |
| 11 | Points, lines, `SkinnedMeshBVH` | 3.2-3.4 | L | none found |
| 12 | Per-tree utilities: `estimateMemoryInBytes`, `getBVHExtremes`, three-mesh-bvh-format `serialize`, `getJSONStructure` | 7.1-7.3, 7.5 | S | none found |

A project's full switch waits on every item in its column:
- csg needs 2, 3 and 4.
- homefig needs 1 and 2.
- drei needs 1. Its refraction material is GLSL (won't, 8.12).
- pathtracer needs 1, 3 and 5 for its CPU side. Its shaders are GLSL (won't).
- Denoiser, upscaler, ddgi-research and reflections need nothing beyond what is on `main` (rows 8.1 to 8.3 and 5.1), apart from the switch itself (`docs/three.md`, "Switching the known consumers").

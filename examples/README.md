# Examples

A Vite multi-page site (BX.5, BX.6, BX.9 in docs/internals/design/bvh-package-split.md). Each example is one page with its source beside it and the shared overlay.

```
pnpm examples:dev                       # http://localhost:5310
pnpm examples:build                     # examples/dist (static)
pnpm examples:smoke                     # every example, headless Chrome, through the GPU runner
pnpm examples:smoke --filter three-raycast
```

The library is read from source through its public names only: `@pmndrs/bvh`, `@pmndrs/bvh/three` and `@pmndrs/bvh/worker` are aliased to `src/index.ts`, `src/three/index.ts` and `src/worker/index.ts` (`vite.config.ts`, `tsconfig.examples.json`). ESLint refuses a path into `src/` here. `examples/` is outside the library build (`tsconfig.build.json` only reads `src/`) and is type-checked by `pnpm typecheck`.

## The shell

- `index.html` lists every example from `shared/examples.ts` (also the build inputs and the smoke list).
- `shared/overlay.ts` is the overlay every page shows: `builtBy` and `fallbackReason`, build ms (CPU wall time; GPU timestamps where the device has `timestamp-query`, else what was measured, said so), SAH cost, depth, nodes, triangles and bytes, Mrays/s where rays are traced, a main-thread frame-time graph, a back link and "Source ↗".
- `shared/app.ts`: `WebGPURenderer` (three requests every adapter feature, `subgroups` and `timestamp-query` included), `deviceFromRendererAsync`, camera, orbit controls, the frame loop.
- `shared/loader.ts`: defaults from three's built-in geometries; `?model=<path>` loads a glTF from the assets base; drag and drop of a `.glb` (or a `.gltf` dropped together with its `.bin` and textures).
- `shared/gpu-rays.ts`: picking and batch rays over a `SceneTrees` scene with `TraceBatchKernel`; `shared/gpu-timer.ts`: timestamp queries; `shared/stats.ts`: overlay numbers from a layout (`treeSahCost`) or a `SceneTrees` (`onTree` events, `counts`, `meshStatus`, `memory`).
- `shared/smoke.ts`: smoke mode (below).
- `shared/builders.ts`: one mesh's tree from a chosen builder, timed (CPU wall time, GPU timestamps), with the chooser's fallback; the inspector and the vanilla pages share it.
- `shared/vanilla/`: the three-free shell of the vanilla pages (below). `overlay.ts`, `smoke.ts`, `stats.ts`, `gpu-rays.ts`, `gpu-timer.ts`, `builders.ts` and `assets.ts` are three-free too; `app.ts`, `loader.ts` and `compare.ts` are the three shell. ESLint refuses three, `@pmndrs/bvh/three` and the three shell in `vanilla/` and `shared/vanilla/`.

### Assets

Nothing large is committed. The dev server serves `BVH_ASSETS_DIR` (default `../webgpu-ddgi/assets`, the GPU runner's variable) at `/assets/`, so:

```
BVH_ASSETS_DIR=../webgpu-ddgi/assets pnpm examples:dev
# http://localhost:5310/three/load-gltf/?model=downloads/sponza-intel/main_sponza/NewSponza_Main_glTF_003.gltf
```

`?assets=<url>` points the loader at another base (a built site has no `/assets/`).

### Smoke runs

`pnpm examples:smoke` is `node tests/gpu/run.ts --examples`: the runner serves this site with its own Vite config (plus the runner's preflight page), opens each example with `?token=`, and judges it like a test page. An example in smoke mode runs as it would for a person and, once it has shown what it is about, calls `ready()` (`shared/smoke.ts`), which leaves the report on `window.__gpuTestReport` with the overlay's numbers as `data`. Uncaught errors, unhandled rejections, WebGPU `uncapturederror`s, console errors and timeouts fail it. Each example also checks one thing it shows (for example, `acceleratedRaycast` agreeing with three's raycast, the GPU batch hitting).

## Vanilla WebGPU examples (P4 item 1)

Raw WebGPU and `@pmndrs/bvh` only: no three, no engine. They prove the "not a three library" claim.

- `shared/vanilla/app.ts`: a `GPUDevice` asked for `subgroups` and `timestamp-query` when the adapter has them, plus the adapter's storage limits; a WebGPU canvas; an orbit camera (`camera.ts`: a view-projection matrix for raster passes, the ray basis for compute passes, packed camera rays); the shared overlay; a frame loop of `update` (may await a readback) then `render` (records into the frame's encoder; the loop submits).
- `shared/vanilla/geometry.ts`: procedural meshes (torus knot, sphere, box, torus, ground) as plain `positions` and `indices`, scenes of meshes and column-major instances with a material number, `mergeToWorld`. `gltf.ts`: a minimal glTF reader (positions, indices, node transforms; `.glb` or `.gltf` with its `.bin`; no Draco, meshopt or quantized positions) for `?model=` and drag and drop.
- `shared/vanilla/raycast-view.ts`: the ray-cast kernel the pages share: `traceModuleWgsl` in the page's own compute shader, one closest-hit ray per pixel, a colour from the hit alone (normal, material, instance, triangle, distance), optional any-hit hard shadows, each pixel's hit record kept for picking, and a blit pass. A demo of the walk, not a renderer: no sampling, BSDFs or accumulation (the scope rule in HANDOFF.md).

| Page | Shows | Smoke check |
| --- | --- | --- |
| `vanilla/hello` | The smallest use: `createMeshLayoutBuilder(device)` picks the builder and says why, `assembleTraceScene` + `uploadTraceScene`, the tree's boxes at a depth in a line pass (`bvhHelperPositions`), the pointer's ray picks a triangle (`TraceBatchKernel`, read back), and a batch of 1k-262k rays every frame whose hit points a render pass draws straight from the hit buffer | 1,024 batch rays agree with the CPU port of the walk (`createCpuTracer`) on the same scene; the centre ray hits; then a model swap with a pick in flight, and the check again on the new mesh |
| `vanilla/raytrace` | The WGSL walk in the page's own kernel over a scene of meshes and instances; switch tinybvh, H-PLOC and LBVH live (build ms, SAH, Mrays/s from timestamps, scene bytes); shadows through `bvhTraceAny`; the hovered pixel's hit record | with each builder, 1,024 pixels of the view agree with the CPU walk (a ray through a shared edge may name either triangle; a difference a nudge of the camera ray by 1e-6 reproduces counts as a silhouette graze; anything else fails) |
| `vanilla/instances` | `SceneTrees` with 4,000 moving instances of four meshes (`?count=`): `setTransform` for each every frame and `update(encoder)` in the frame's own encoder before the ray-cast pass; GPU H-PLOC first trees, then tinybvh's from the `/worker` pool swapped in (the builder line, swaps and the last tree event); buttons add and remove instances and meshes at run time; the hovered instance picked from the view's hit records | the GPU built first trees (with subgroups; on a busy machine tinybvh's may still be placed first); after 500 instances removed and a mesh added, every mesh on a tinybvh tree; 64 rays at live instances hit an instance near its current centre, never a removed one |
| `vanilla/builders` | tinybvh, GPU H-PLOC and GPU LBVH on one mesh (a 100k-triangle knot, `?model=` or a drop, merged), one row each: build CPU ms and GPU timestamps (after a warm-up), SAH and SAH against tinybvh, depth, nodes, bytes, Mrays/s on the same seeded rays through `TraceBatchKernel` (median of timed dispatches), and the share of hits at tinybvh's distance; largest-leaf option; a row's tree in the ray-cast view on click. Without subgroups the GPU rows are built by tinybvh and say why | three rows; at least half the rays hit; every tree's hits at tinybvh's distance for at least 99.9% of 65k rays; tinybvh's hits agree with the CPU walk on 1,024; with subgroups the GPU rows were built by their builders; then two rebuilds asked at once (largest leaf 2) run one after the other while frames trace, and the checks again |
| `vanilla/refit` | A deforming mesh (a 57,600-triangle torus knot, `?segments=`): this page's compute pass twists, waves or stretches the rest pose into a storage buffer every frame, and `GpuMeshRefit` (shared `GpuMeshRefitKernels`, `diagnostics: true`) refits its tree from that buffer inside the uploaded scene's buffers, at the offsets and node format `meshRefitPlacement` gives (quantized by default: the refit keeps an `f32` copy). The top level boxes the mesh by the box every pose stays in. Shown: pose and refit GPU ms (timestamps), drift (the refit tree's relative SAH cost over the built tree's, from the refit's stats) and the refit against a fresh H-PLOC tree of the same pose (built every second, never placed); "rebuild now" and "auto rebuild" past a drift limit build H-PLOC from the posed buffer, assemble and upload a new scene, make a new refit, swap both in between frames and only then destroy the old ones. The plan's second case (10,000 moving instances, top-level refit) is `vanilla/instances` with `?count=10000` | twisted far from the built pose, 1,024 pixels agree with the CPU walk over a tinybvh tree built from the pose read back (a stale or partial refit disagrees), at least 50 sampled pixels on the deforming mesh; then, moving, two rebuilds asked at once (the second queued) while the deform switches, both swapped in; then, frozen, a rebuild from the frozen pose refits to 1.00× its own cost (the stats are the new tree's) and the view agrees with the CPU walk again |

Measured on the Apple M5 (Chrome, headless, `pnpm examples:smoke`, a loaded machine), default scenes: `hello`'s 16k-ray batch at 60-95 Mrays/s (timestamps); `raytrace`'s 1280×577 view of the showcase scene in 0.4-1.3 ms per pass; `builders` on the 100k-triangle knot: tinybvh 40-44 ms, H-PLOC 5-6 ms CPU (0.84 ms GPU), LBVH 4-10 ms (0.46-3.2 ms GPU), SAH 56.2, 57.0 and 62.0, every tree's hits identical to tinybvh's; `instances` 3,800 instances moved and refit in 2-4 ms of CPU per frame.

## three.js examples

| Page | Shows | Smoke check |
| --- | --- | --- |
| `three/raycast` | `acceleratedRaycast` against three's `Mesh.raycast` (rays/s of each, live), pointer picking with either, and a GPU batch (4k to 1M rays) in one dispatch through `SceneTrees` + `TraceBatchKernel`, its hits drawn as points | 64 rays agree with three's raycast; the batch hits |
| `three/pathtracer` | A TSL compute path tracer over `BvhTraceNodes`: `traceClosest` per bounce in a `Loop` with a uniform bound, `traceAny` sun shadows, `hit.material` into a fixed-capacity palette, progressive accumulation. Compiled with `compileComputeAsync`; the shader text is the same for any scene | rays counted (one atomic per pixel) |
| `three/skinned` | Nine `SkinnedMesh` characters: `BvhPosePass` skins on the GPU, `SceneTrees.setPose` refits; refits, drift rebuilds and moved boxes counted; GPU picking on the moving surface | 48 GPU picks agree with three's CPU-skinned raycast on a frozen pose (bounding spheres recomputed), none allowed to differ |
| `three/instances` | 6,000 `InstancedMesh` + 3,000 `BatchedMesh` instances moving every frame (`?instanced=`, `?batched=`), top-level refit each frame, GPU picking, `setColorAt` highlight | 64 rays at instances: each hit lies inside the hit instance's current bounds |
| `three/load-gltf` | Fast first: GPU H-PLOC trees, then tinybvh from a `/worker` pool swapped in; a per-mesh timeline; rays traced on the first trees and after the swap. Materials are compiled with `compileAsync` while the trees build | every mesh has a tree; GPU first trees were used |
| `three/inspector` | tinybvh, GPU H-PLOC and GPU LBVH on one mesh: build CPU ms and GPU timestamps, SAH, depth, nodes, bytes; leaf size, `triangleLeaves`, `nodeFormat`; `MeshBVHHelper` at a depth; a CPU heat map of nodes per ray; a table of every build | six builds (3 builders × records/f32 and indexed/quantized), heat map |
| `three/sdf` | Not built: the library has no SDF generation (a gap) | |

Measured on the Apple M5 (Chrome, headless, `pnpm examples:smoke`), default scenes: the GPU batch traces 65k rays in about 1.9 ms (35 Mrays/s, a 100k-triangle knot from outside, timestamps); `acceleratedRaycast` 0.40 M rays/s against three's 306 (100k triangles); the path tracer about 120 Mrays/s (wall time of one dispatch); Sponza (405 meshes, 3.7 M triangles) through `load-gltf` has every first tree in about 150-230 ms and every tinybvh tree in about 600 ms after the first update.

## Comparison pages (`compare/`, BX.10)

Each runs the suites of `compare/core/` (the same as `pnpm compare:node`) on the page's mesh (`?scene=furnished-room`, `?scene=sponza`, `?model=`, a torus knot by default; meshes merged into one world-space mesh), shows a table with every answer's agreement with ours, works live with the library chosen in the overlay, and has a "what else you can do" box. Smoke runs use a 20k-triangle knot and short timings; `BVH_PAGE_QUERY='scene=sponza&full'` gives the runner a full run, whose CSV rows come back in the report (`data.csv`). Results and method: `compare/RESULTS.md`.

| Page | Compares | What else (live) | Smoke check |
| --- | --- | --- | --- |
| `compare/raycast` | CPU first and all hits per second, every library; a GPU batch against three-mesh-bvh/webgpu | the batch is a raw WebGPU pass three never sees | three-mesh-bvh (CPU and GPU) and three agree with ours on every ray |
| `compare/build` | build, refit, memory; main-thread stalls on the main thread and in workers (ours and three-mesh-bvh's) | a GPU H-PLOC build of the same mesh | refits agree |
| `compare/tree-quality` | SAH, depth, boxes, leaves, bytes; `MeshBVHHelper` of either | three builders, one layout | both trees have an SAH cost |
| `compare/queries` | box, sphere, closest point (three-mesh-bvh, Rapier) | the GPU builder's tree answers the same CPU queries | every answer agrees |
| `compare/scene` | scene trees over thousands of moving objects: ours, bvh.js, `ObjectBVH`; live frustum culling | the GPU top-level refit every frame | `ObjectBVH` agrees before and after the motion |
| `compare/gpu` | GPU closest and any hit against three-mesh-bvh/webgpu, paired rounds | a GPU-built tree traced by the same kernel | every ray agrees |

## Public-API gaps the examples ran into

- `SceneTrees` keeps no CPU copy of its trees, so there is no SAH cost per tree for the overlay; and its GPU build time (`counts.gpuBuildMs`) is submit-to-stats wall time, not timestamps. `SceneTrees.update(encoder)` takes no timestamp writes, so the top-level refit's GPU time can't be shown (the instances page shows CPU ms of sync + update).
- `SceneTreesObject3DSync` doesn't list its mesh handles; the pages collect them from `onTree` events.
- The GPU walk's step counters (the counted walk the benches use) aren't public, so the inspector's heat map is a CPU walk over the same tree.
- `GpuRays` here (picking and batches over a `SceneTrees`: buffers, readback, hit decoding) is about 100 lines every app will write; a small helper in the package would do.
- `createMeshLayoutBuilder` takes no timestamp writes and does its pipeline setup in the call that makes it, so `vanilla/hello` shows build CPU time only (setup apart); `vanilla/builders` drives `GpuHplocBuilder`/`GpuLbvhBuilder` batches itself for GPU times (`shared/builders.ts`).
- `createMeshLayoutBuilder({ prefer: 'tinybvh' })` reports `fallbackReason: 'the CPU builder was asked for'`: an overlay has to know that's not a fallback.
- An uploaded scene (`uploadTraceScene`) and a `SceneTrees` aren't one type: `traceSourceOf` (`shared/gpu-rays.ts`) wraps the first so one kernel and `GpuRays` take either.
- The plan's `vanilla/raytrace` had ambient occlusion; it would need random directions (sampling), which the scope rule leaves to renderers above the package, so the page has hard any-hit shadows only.

## Library issues the examples ran into

- **Pose-pass compile race (fixed):** the sync started each pose variant's `compileComputeAsync` without waiting for it, and an `update()` before three's async pipeline existed dispatched `setPipeline(undefined)` (a TypeError, seen on the M5 with nine skinned meshes on one geometry). `update()` now skips posing an object while its variant compiles (`poseCounts.waiting`; its tree keeps its last pose), and `await sync.ready` has every object posed from the first update. `three/skinned` doesn't wait.
- **three's `SkinnedMesh` raycast culls with a stale sphere:** `SkinnedMesh.raycast` tests `mesh.boundingSphere`, which three computes once (the renderer's frustum culling does, at the first frame) and never again. Rays that hit the current pose outside that sphere miss in three's raycast. The `three/skinned` check calls `computeBoundingSphere()` for the frozen pose before comparing.
- **Limits:** a scene past 128 MiB of triangles needs `requiredLimits` on the `WebGPURenderer` (three asks for WebGPU's defaults). `shared/app.ts` asks for the adapter's `maxStorageBufferBindingSize` and `maxBufferSize`; without it, a 1.8 M-triangle scene fails with a clear `RangeError` naming the limit.

# Examples

A Vite multi-page site (BX.5, BX.9 in docs/design/bvh-package-split.md). Each example is one page with its source beside it and the shared overlay.

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

### Assets

Nothing large is committed. The dev server serves `BVH_ASSETS_DIR` (default `../webgpu-ddgi/assets`, the GPU runner's variable) at `/assets/`, so:

```
BVH_ASSETS_DIR=../webgpu-ddgi/assets pnpm examples:dev
# http://localhost:5310/three/load-gltf/?model=downloads/sponza-intel/main_sponza/NewSponza_Main_glTF_003.gltf
```

`?assets=<url>` points the loader at another base (a built site has no `/assets/`).

### Smoke runs

`pnpm examples:smoke` is `node tests/gpu/run.ts --examples`: the runner serves this site with its own Vite config (plus the runner's preflight page), opens each example with `?token=`, and judges it like a test page. An example in smoke mode runs as it would for a person and, once it has shown what it is about, calls `ready()` (`shared/smoke.ts`), which leaves the report on `window.__gpuTestReport` with the overlay's numbers as `data`. Uncaught errors, unhandled rejections, WebGPU `uncapturederror`s, console errors and timeouts fail it. Each example also checks one thing it shows (for example, `acceleratedRaycast` agreeing with three's raycast, the GPU batch hitting).

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

## Library issues the examples ran into

- **Pose-pass compile race (fixed):** the sync started each pose variant's `compileComputeAsync` without waiting for it, and an `update()` before three's async pipeline existed dispatched `setPipeline(undefined)` (a TypeError, seen on the M5 with nine skinned meshes on one geometry). `update()` now skips posing an object while its variant compiles (`poseCounts.waiting`; its tree keeps its last pose), and `await sync.ready` has every object posed from the first update. `three/skinned` doesn't wait.
- **three's `SkinnedMesh` raycast culls with a stale sphere:** `SkinnedMesh.raycast` tests `mesh.boundingSphere`, which three computes once (the renderer's frustum culling does, at the first frame) and never again. Rays that hit the current pose outside that sphere miss in three's raycast. The `three/skinned` check calls `computeBoundingSphere()` for the frozen pose before comparing.
- **Limits:** a scene past 128 MiB of triangles needs `requiredLimits` on the `WebGPURenderer` (three asks for WebGPU's defaults). `shared/app.ts` asks for the adapter's `maxStorageBufferBindingSize` and `maxBufferSize`; without it, a 1.8 M-triangle scene fails with a clear `RangeError` naming the limit.

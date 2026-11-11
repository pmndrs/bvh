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

| Page | Shows |
| --- | --- |
| `three/raycast` | `acceleratedRaycast` against three's `Mesh.raycast` (rays/s live), and a GPU batch of up to a million rays in one dispatch through `SceneTrees` + `TraceBatchKernel` |

## Public-API gaps the examples ran into

- No `declare module 'three'` augmentation in `/three`: `raycaster.firstHitOnly`, `geometry.boundsTree` and `geometry.computeBoundsTree` need casts (three-mesh-bvh ships one).
- `SceneTrees` keeps no CPU copy of its trees, so there's no SAH cost per tree for the overlay; and its GPU build time (`counts.gpuBuildMs`) is submit-to-stats wall time, not timestamps.
- `SceneTreesObject3DSync` doesn't list its mesh handles; the examples collect them from `onTree` events.

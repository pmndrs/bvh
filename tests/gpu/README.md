# GPU pages and runner

```
pnpm test:gpu                         # every page (also runs the runner's self-tests)
pnpm test:gpu:page trace-oracle       # one page, or several comma-separated: bvh-lbvh,bvh-hploc
node tests/gpu/run.ts --filter trace-bench-smoke --verbose --json out.json
```

`--filter` matches by substring of the page folder name (`bvh-lbvh` also runs `bvh-lbvh-fallback` and `-profile`). The runner starts a vite server on `tests/gpu/`, launches your installed Chrome headless with WebGPU, runs each `pages/<name>/` and fails on a failed report, any browser error or WebGPU validation warning, or a timeout. It was ported from webgpu-ddgi's `tools/gpu/`; most of what is below that is general (page contract, `test.json`, `--gpu`) is in `tests/gpu/RUNNER.md`, ddgi's README for the runner.

Pages that run quickly and need no assets: `bvh-lbvh`, `bvh-hploc`, `bvh-blas-refit`, `bvh-lbvh-fallback`, `bvh-top-level-gpu`, `scene-trees-mid-swap`, `scene-trees-runtime`, `scene-trees-deforming`, `three-trace` (three's `WebGPURenderer`, `@pmndrs/bvh/three`), `three-skinned` (skinned and morphed meshes posed by `BvhPosePass`), `three-shapecast` (`BvhTraceNodes.getShapecastFn`: TSL and `wgslFn` hooks compiled into the scene shapecast, against `bvhIntersectsSphere` and a CPU count; loads with `updateSceneTreesAsync`; `profile=ci` runs 1,024 spheres), `trace-inline`, `trace-oracle`, `query-oracle` (the GPU spatial queries against `CpuMeshBvh`; `profile=ci` as trace-oracle, about 43 s on SwiftShader on an M5, `?queries=N` for the points per case). `trace-clustered` traces clustered scenes (`clusterTraceScene`) against the CPU reference and the plain scene ray for ray, closest-point distances included (`profile=ci` about 41 s on SwiftShader on Windows); its `parity.test.ts` checks the cut against three-mesh-bvh's `ClusteredBVH` in Node. `scene-trees-sponza` (manual) loads Sponza and reports `SceneTrees` memory and load times. `trace-cpu-agreement` (manual, needs the assets unless `?scene=none`) compares the CPU walk with the GPU's ray for ray (Sponza primary and air rays, the oracle's seam and random rays): any hits must agree everywhere, closest hits never disagree on hit or miss, primary rays name the same triangle; bits are reported. `compile-times` (manual, about a minute) times the package's pipeline compiles on fresh devices, one item at a time (the walk's batch kernels, the queries, the SDF kernel, the builders, the refits, the top level, `BvhTraceNodes` in three): pass 1 cold for the browser (the runner's fresh profile), pass 2 warm (Chrome's shader cache); `?items=hploc,sdf` for some, `BVH_CHROME_FLAGS=--disable-gpu-shader-disk-cache` for no browser cache. `wgsl-f32-arithmetic` (manual, seconds) reports how the GPU rounds the f32 operations the walk uses (division, reciprocal, `dot`, fused or not, `sqrt`), against CPU models. The `trace-bench-*` pages load scenes (see below); the Sponza and Bistro ones take minutes.

## Scene assets

The scene benches (`trace-bench-*`, `tools/*.ts`) read models and scene manifests from webgpu-ddgi, not from this repository. Two environment variables point at them (paths relative to this repository's root, or absolute):

| Variable | Holds | Default |
| --- | --- | --- |
| `BVH_ASSETS_DIR` | `downloads/<asset>/...`, the converted models (`BreakfastRoom.glb`, `main_sponza_1k/NewSponza_Main_1k.gltf`, ...) | `../webgpu-ddgi/assets` |
| `BVH_MANIFESTS_DIR` | scene manifest JSON files (`sponza.json`, `furnished-room.json`, ...) | `../webgpu-ddgi/scenes/manifests` |

The models must be the converted ones that webgpu-ddgi's `pnpm assets:fetch` produces; a checkout that only has the raw downloads will fail with "model file not found". Nothing from these folders is copied into this repository. A page that needs a scene that isn't there fails with a message naming the variable.

## Surface-ray pages (instanced tree against merged tree)

`trace-bench-surface-sponza` and `trace-bench-surface-bistro-interior` (manual pages, run by exact name) trace rays that start on surfaces (`surface`, RC1.5's screen-probe set) and in the air (`air-grid`), closest and any hit, through the per-mesh instanced tree and the merged tree (`buildTraceScene(..., { mergeStaticInstances: true })`, every instance `static: true`). The two trees are timed in turn, 5 rounds with the order reversed every other round, then the surface sets again at the end; the merged-over-instanced ratio is taken within each round and is the result (the M5 drifts by up to about 30%). Counters (steps, instances entered) come from a counted copy of the walk built from the production walk's text (`counted-walk-wgsl.ts`). After the merged-against-instanced rounds the page times the 40-byte hit record against the 24-byte one (`hitNormal: true` against `false`, closest-hit, both trees and both sets, the same rounds and order reversal); rows `surface-rays/paired/hit_record_40_over_24/...`. Last it times indexed leaves against records (docs/internals/design/indexed-leaves.md): the same tinybvh trees assembled as records, indexed with `vec4` positions and indexed with packed positions, both trees, both sets, closest and any, three formats back to back with the order reversed every other round; rows `surface-rays/paired/indexed_over_records/...` and `indexed_packed_over_records/...`. Every indexed answer must be bit-identical to the records one. The page fails only if the trees disagree on hits, if indexed leaves change an answer, or the compact kernel's hits differ from the full kernel's; timing never fails it. `?profile=quick` runs 3 short rounds. Then it times quantized nodes against `f32` (docs/internals/design/compressed-nodes.md): the same trees and records, both trees, both sets, closest and any; rows `surface-rays/paired/quantized_over_f32/...`, the counted walk per format (`surface-rays/node_format/...`). Every quantized answer must have the `f32` answer's `t` bit for bit and the same record unless another triangle lies at that `t` (a tie, counted apart). `?skip=hit-record,leaves` leaves out the hit-record and indexed sections. The runner passes page options from `BVH_PAGE_QUERY` (for example `BVH_PAGE_QUERY='skip=hit-record,leaves'`).

```
BVH_ASSETS_DIR=<assets> BVH_MANIFESTS_DIR=<manifests> pnpm test:gpu:page trace-bench-surface-sponza --json out.json
node tests/gpu/pages/trace-bench/surface-to-csv.ts out.json [more.json ...] [--append]   # rows for bench/results/trace.csv
```

## Not here

- `platform-bench.ts` and the `platform` tracer of `trace-bench` compared against ddgi's own platform tracer; the `platform*` variants of `trace-oracle` and the createGI/technique version of `trace-inline` tested ddgi's platform. They live on in webgpu-ddgi.
- The batch kernel and scene upload the pages use (`TraceBatchKernel`, `uploadTraceScene`) are in `src/trace/trace-gpu.ts`.

## SwiftShader on a machine with a GPU

CI's flags (`.github/workflows/ci.yml`: `--no-sandbox --use-webgpu-adapter=swiftshader --use-angle=swiftshader --enable-unsafe-swiftshader`) give no WebGPU adapter in Chrome 157 on Windows: `--use-angle=swiftshader` is the one that breaks it. To check a page on the CPU while the real GPUs are busy (another project benchmarking, say), use:

```
BVH_CHROME_FLAGS="--no-sandbox --use-webgpu-adapter=swiftshader --enable-unsafe-swiftshader --disable-gpu-compositing" BVH_ALLOW_SOFTWARE_ADAPTER=1 BVH_PAGE_QUERY=profile=ci pnpm test:gpu:page <page>
```
The runner reports the adapter as `google/swiftshader`. Chrome 157's SwiftShader has `subgroups`, so the GPU builders run, but on the CPU: big GPU batches are slow (`three-load-gltf`'s 1.7M triangles take about 18 s), and tinybvh's workers can finish first.

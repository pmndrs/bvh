# GPU pages and runner

```
pnpm test:gpu                         # every page (also runs the runner's self-tests)
pnpm test:gpu:page trace-oracle       # one page, or several comma-separated: bvh-lbvh,bvh-hploc
node tests/gpu/run.ts --filter trace-bench-smoke --verbose --json out.json
```

`--filter` matches by substring of the page folder name (`bvh-lbvh` also runs `bvh-lbvh-fallback` and `-profile`). The runner starts a vite server on `tests/gpu/`, launches your installed Chrome headless with WebGPU, runs each `pages/<name>/` and fails on a failed report, any browser error or WebGPU validation warning, or a timeout. It was ported from webgpu-ddgi's `tools/gpu/`; most of what is below that is general (page contract, `test.json`, `--gpu`) is in `tests/gpu/RUNNER.md`, ddgi's README for the runner.

Pages that run quickly and need no assets: `bvh-lbvh`, `bvh-hploc`, `bvh-blas-refit`, `bvh-lbvh-fallback`, `bvh-top-level-gpu`, `scene-trees-mid-swap`, `scene-trees-runtime`, `scene-trees-deforming`, `three-trace` (three's `WebGPURenderer`, `@pmndrs/bvh/three`), `three-skinned` (skinned and morphed meshes posed by `BvhPosePass`), `trace-inline`, `trace-oracle`. `scene-trees-sponza` (manual) loads Sponza and reports `SceneTrees` memory and load times. The `trace-bench-*` pages load scenes (see below); the Sponza and Bistro ones take minutes.

## Scene assets

The scene benches (`trace-bench-*`, `tools/*.ts`) read models and scene manifests from webgpu-ddgi, not from this repository. Two environment variables point at them (paths relative to this repository's root, or absolute):

| Variable | Holds | Default |
| --- | --- | --- |
| `BVH_ASSETS_DIR` | `downloads/<asset>/...`, the converted models (`BreakfastRoom.glb`, `main_sponza_1k/NewSponza_Main_1k.gltf`, ...) | `../webgpu-ddgi/assets` |
| `BVH_MANIFESTS_DIR` | scene manifest JSON files (`sponza.json`, `furnished-room.json`, ...) | `../webgpu-ddgi/scenes/manifests` |

The models must be the converted ones that webgpu-ddgi's `pnpm assets:fetch` produces; a checkout that only has the raw downloads will fail with "model file not found". Nothing from these folders is copied into this repository. A page that needs a scene that isn't there fails with a message naming the variable.

## Surface-ray pages (instanced tree against merged tree)

`trace-bench-surface-sponza` and `trace-bench-surface-bistro-interior` (manual pages, run by exact name) trace rays that start on surfaces (`surface`, RC1.5's screen-probe set) and in the air (`air-grid`), closest and any hit, through the per-mesh instanced tree and the merged tree (`buildTraceScene(..., { mergeStaticInstances: true })`, every instance `static: true`). The two trees are timed in turn, 5 rounds with the order reversed every other round, then the surface sets again at the end; the merged-over-instanced ratio is taken within each round and is the result (the M5 drifts by up to about 30%). Counters (steps, instances entered) come from a counted copy of the walk built from the production walk's text (`counted-walk-wgsl.ts`). After the merged-against-instanced rounds the page times the 40-byte hit record against the 24-byte one (`hitNormal: true` against `false`, closest-hit, both trees and both sets, the same rounds and order reversal); rows `surface-rays/paired/hit_record_40_over_24/...`. Last it times indexed leaves against records (docs/design/indexed-leaves.md): the same tinybvh trees assembled as records, indexed with `vec4` positions and indexed with packed positions, both trees, both sets, closest and any, three formats back to back with the order reversed every other round; rows `surface-rays/paired/indexed_over_records/...` and `indexed_packed_over_records/...`. Every indexed answer must be bit-identical to the records one. The page fails only if the trees disagree on hits, if indexed leaves change an answer, or the compact kernel's hits differ from the full kernel's; timing never fails it. `?profile=quick` runs 3 short rounds. Then it times quantized nodes against `f32` (docs/design/compressed-nodes.md): the same trees and records, both trees, both sets, closest and any; rows `surface-rays/paired/quantized_over_f32/...`, the counted walk per format (`surface-rays/node_format/...`). Every quantized answer must have the `f32` answer's `t` bit for bit and the same record unless another triangle lies at that `t` (a tie, counted apart). `?skip=hit-record,leaves` leaves out the hit-record and indexed sections. The runner passes page options from `BVH_PAGE_QUERY` (for example `BVH_PAGE_QUERY='skip=hit-record,leaves'`).

```
BVH_ASSETS_DIR=<assets> BVH_MANIFESTS_DIR=<manifests> pnpm test:gpu:page trace-bench-surface-sponza --json out.json
node tests/gpu/pages/trace-bench/surface-to-csv.ts out.json [more.json ...] [--append]   # rows for bench/results/trace.csv
```

## Not here

- `platform-bench.ts` and the `platform` tracer of `trace-bench` compared against ddgi's own platform tracer; the `platform*` variants of `trace-oracle` and the createGI/technique version of `trace-inline` tested ddgi's platform. They live on in webgpu-ddgi.
- The batch kernel and scene upload the pages use (`TraceBatchKernel`, `uploadTraceScene`) are in `src/trace/trace-gpu.ts`.

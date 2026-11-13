# Handoff: pmndrs/bvh after the split (2026-11-10)

This repo was split out of `DennisSmolek/webgpu-ddgi`, where the code lived in `packages/bvh`. The full plan is [docs/internals/design/bvh-package-split.md](docs/internals/design/bvh-package-split.md): the decisions (D1–D8), the target layout, phases P0–P9, examples, docs and tasks BX.0–BX.20. Read it first.

## Rules the work runs under (from Dennis)

- **The core and three matter most.** Other tools of Dennis's need this BVH, and they shouldn't have to pull it out of the GI repo.
- **Engine-free core**, with three and Babylon behind subpaths. Babylon is "it works" only (D5). WebGL2 isn't a target.
- **Feature-forward WebGPU:**
  - `subgroups` first, with one declared fallback (tinybvh on the CPU);
  - no parallel GPU paths.
- **Faithful port first.** Ports of tinybvh's H-PLOC and HIPRT's LBVH stay as published. Any speed change is a measured lever, with a note on why and how to undo it.
- **Licensed code only.** Nothing is ported from repos without a licence (D8).
- **Hardware:**
  - RTX 3060 and M5 Pro are the targets;
  - the Intel UHD is for correctness only;
  - benches run on a loaded GPU;
  - no build may take minutes.
- **Correctness gates, not timing gates**, at this prototype stage.
- **Models:** Opus for code and review, Sonnet or Haiku for small tasks, docs and tests.
- **Git:** commit locally; pushing `main` is fine; no npm publish, tags or releases.
- **Levers:** a flexible setting with a measured win may become the default (the 256 MiB cap).
- **The package ends at the hit** (2026-11-13). The library covers building, refitting, scene trees, the walk and spatial queries. A hit carries the triangle, instance, barycentrics, face normal and an opaque material number.
  - Anything that decides what a hit *looks like* goes in Dennis's separate path tracer repo, or in the shared tracing layer above us: sampling, random numbers, BSDFs, lights, accumulation, textures, and shading-normal or UV interpolation.
  - Physics-style queries (shapecast, closest point, boxes) stay here.
  - Check every consumer request against this rule. Per-triangle materials passed because the material is only a number.
  - `examples/three/pathtracer` is a demo of the walk. It and its pending compare page may move to the new repo.

## Current state (2026-11-13). Read this first.

Everything below is on `main` and pushed to origin. No release, tag or npm publish.

**Dennis's decisions**
- **Source of truth:** this repo, for BVH and walk code. webgpu-ddgi's copy is frozen until P3 (split point `d288adf`).
- **Commits:** local; pushing `main` is fine; no releases. Opus writes the code and does the reviews; Sonnet or Haiku take the mechanical tasks.
- **Walk API:** WGSL names `bvh*`; filter hook `bvhTrianglePasses(BvhCandidate)` (the candidate has `query` and `rayMask`; ddgi's materials stay in ddgi); 8-bit instance masks, compiled out with `rayMasks: false`.
- **Package scope:** every project that needs a BVH, not just ddgi (see Consumers).
- **Licence:** MIT, Dennis Smolek (`LICENSE`).
- **Build-peak cap:** `SceneTrees` `gpuMaxBatchBytes` defaults to 256 MiB; `Infinity` turns it off.
- **Defaults after the RTX 3060 run (2026-11-13; numbers in `docs/internals/WINDOWS-RUNBOOK.md`, section 7):**
  - `mergeStaticInstances` is **on** in `SceneTrees` and `buildTraceScene`. It only merges instances marked `static: true` (added before the first update, in `SceneTrees`); a scene that marks nothing is unchanged. RTX 3060: 1.16-1.67x; closest hits name the instanced tree's instance and triangle on 99.949-99.996% of rays (near-ties at a shared distance can pick another), any hits agree on every ray. Behaviour change: `static: true` now means merged, so moving, hiding or removing such an instance throws in `SceneTrees` and is skipped with a warning in the `/three` sync, and static instances that would merge must have distinct `instanceIndex` values (`mergeStaticGeometry` throws).
  - `nodeFormat: 'quantized'` is the **default for every scene and walk option** (`DEFAULT_NODE_FORMAT` in `src/gpu-layout.ts`): `SceneTrees`, `assembleTraceScene`, `buildTraceScene`, `traceModuleWgsl`, `batchKernelWgsl`, `TraceBatchKernel.create` (the places where the library makes both sides). **Options default to quantized; data must say what it is:** `TraceSceneBuffers` and `BvhTraceNodes` sources must name their formats (types and a run-time check, `traceSourceFormats`; `BvhTraceNodes` reads an upload's from its buffers), `GpuMeshRefit.setTree` with a `nodeWordOffset` needs `nodeFormat` (`meshRefitPlacement` gives it), and `refitOnCpu` refuses quantized nodes. Mesh builders still write `f32` trees (a scene quantizes them), `CpuMeshBvh` and `CpuSceneBvh` stay `f32`, and CPU data that names no format (`MeshBvh`, layouts, `TraceSceneData`, files) is `f32`. `gpuTreeOutputs: 'adopt'` needs `nodeFormat: 'f32'`. Deforming meshes pay the 64-byte `f32` copy by default now (96 bytes per node in all). RTX 3060: closest 1.02-1.22x, any 0.98-1.03x.
  - `triangleLeaves` stays `'records'`: indexed waits for an RTX 3060 run of quantized and indexed together (Open). `assembleTraceScene` with no `triangleLeaves` takes its trees' format (indexed trees give an indexed scene; a records/indexed mix throws), so defaults on both sides agree; with no `nodeFormat` it quantizes `f32` trees and places quantized ones, and only an explicit `'f32'` with quantized trees throws.
  - The 40-byte hit record stays the default. It costs 6-12% on the RTX 3060 (0.88-0.94x the 24-byte rate; the M5 measured 0.96-1.15x); shadow and occlusion users should pass `hitNormal: false`.
  - `anyHitOrder` stays `'far'`: `'larger'` is 1.11-1.48x on instanced trees and 1.03-1.07x on merged air rays, but 0.80-0.87x on merged surface rays, and merged trees are the default now.
- **Accepted as built (2026-11-13):** the former "Needs Dennis" list. The design departures are rows A1-A4 of `docs/internals/DIVERGENCES.md` (section 7) (indexed vertex base in a per-mesh header; the 65,536-source cap; the quantized frame in the tree's first node slot; `parentOffset`), and the design docs state them. The three-mesh-bvh differences are notes in `docs/internals/PARITY.md` (`/three` ranges and three r187's `NaN` hit, skinning limits, CPU scene raycast, `strategy` and `maxDepth`, shapecast, `bvhcast`, `OrientedBox`, `ExtendedTriangle`, closest point, utilities, the helper). three's `NaN`-distance behaviour is recorded, not filed upstream.
- **Scope rule applied to parity (2026-11-13):** PARITY 8.7 `sampleTrianglePoint` and 8.8b `ndcToCameraRay` are won't (renderer work, above the hit); 8.8a `intersectRayTriangle` and `closestPointToTriangle` stay missing (geometry, in scope).
- **Wasm reference:** the Linux build; a Windows build differs in 2 `__FILE__` separator bytes (`docs/internals/design/bvh-builders.md`).

**On `main`** (lint, typecheck, build and the licence check pass; every correctness page passes on the Apple M5)
- **GPU core, `GPU_LAYOUT_VERSION` 2 (unreleased):** ported refit kernels and the Metal watertight fix `bvhEdgeFunction` (rays through shared edges 4,513 per 2M to 0); `BvhHit.normal` and `BvhHit.material`; GPU page runner `pnpm test:gpu`, `pnpm test:gpu:page <name>` (`tests/gpu/README.md`).
- **`mergeStaticInstances`** (on by default since 2026-11-13, for instances marked static): merged world-space tree, exact hit identity (triangle word 11 = source instance, flag bit 1 flips the face, instance bit 3 marks merged; app flags may use bit 0 only). M5 under the radiance-cache bench lock: Sponza surface rays 1.85x closest-hit, 1.36x any-hit; Bistro interior 1.22x and 1.11x. RTX 3060 Laptop (2026-11-13): Sponza 1.44-1.67x closest, 1.24-1.48x any; Bistro interior 1.19-1.20x and 1.16-1.35x. Pages `trace-bench-surface-*`; rows in `bench/results/trace.csv`.
- **`SceneTrees`** (`src/scene/`): GPU H-PLOC trees first, tinybvh trees from `@pmndrs/bvh/worker` swapped in per mesh; growable storage; `GpuTopLevel`; runtime add/remove, merged static groups, deforming meshes. Sponza: first trees 100-118 ms, all tinybvh trees about 370 ms.
- **Build-peak cap:** capped H-PLOC batches built one after another on one scratch pool, stored between updates. Sponza on the M5: peak 1,483 to 711 MiB, settled 470 to 459, first trees 171 to 166 ms. RTX 3060 and Intel not run. Breakdown: `docs/internals/design/memory.md`; the other levers are BX.20.
- **Indexed leaves** (off by default, waiting for a combined run; `docs/internals/design/indexed-leaves.md`): `triangleLeaves: 'records' | 'indexed'`, `positionLayout: 'vec4' | 'packed'`; 16-byte triangles, per-mesh 16-byte header, library-owned positions (exact-bit weld on; `weld: false`; opt-in `weldTolerance`), merged groups via a side table (capped at 65,536 sources), format in header word 7. Sponza 448.1 to 318.0 MiB beyond geometry (306.7 packed), Bistro exterior 210.1 to 155.5. M5 speed (indicative): `vec4` 0.83-0.99x records, packed 0.75-0.92x; RTX 3060: `vec4` 0.92-0.97x, packed 0.82-0.95x.
- **Quantized nodes** (the default for scenes and walks since 2026-11-13; `docs/internals/design/compressed-nodes.md`, option A): `nodeFormat: 'f32' | 'quantized'`; 32-byte nodes, per-mesh grid, frame slot before the root, outward rounding, CPU and GPU quantizers bit-identical, deforming meshes refit an `f32` copy and re-quantize, bits 8-15 of header word 7. Sponza 448.1 to 355.5 MiB (225.5 with indexed), Bistro exterior 210.1 to 169.9 (115.3); SAH +0.01-0.66%; M5 speed 0.93-0.97x `f32` (indicative); RTX 3060 closest 1.02-1.22x, any 0.98-1.03x.
- **`CpuMeshBvh`** (`src/query/`, root export): the one CPU walk (`raycast`, `raycastFirst`, `traverse`, `refit`, checked `serialize`), plus `CpuSceneBvh`.
  - **Shapecast and `bvhcast`:** `shapecast` with three-mesh-bvh's contract, `intersectsBox`/`Sphere`/`Geometry`, `bvhcast` (dual walk step for step), engine-free `ExtendedTriangle`/`OrientedBox` (bit-for-bit with three-mesh-bvh). `bench/query-parity.ts` (homefig capsule controller, three-bvh-csg): full agreement, 0.93x and 0.94x its time.
  - **Closest point:** `closestPointToPoint`, `closestPointToGeometry`; `bench/closest-point.ts`: 0.92x for points in the open, 1.1-1.2x near surfaces, 1.08x geometry against geometry.
  - **Debug helpers:** `getBVHExtremes`, `validateBounds`, `estimateMemoryInBytes`; `/three` has `BVHHelper`/`MeshBVHHelper` (segment counts tested in Node, not yet looked at in a browser).
  - **Scene queries:** `CpuSceneBvh` (scene raycast, `intersectsFrustum`/`intersectsBox`).
- **`@pmndrs/bvh/three`** (three is a peer, `>=0.187.0 <0.188.0`; `docs/internals/three.md`): `bvhSceneFromObject3D`, `SceneTreesObject3DSync`, `BvhTraceNodes` (binds through `threeInternals`, checked at run time), `computeBoundsTree` with three-mesh-bvh's options, `acceleratedRaycast`, `geometry.boundsTree` as `BvhBoundsTree` (`MeshBVH` alias; `GenerateMeshBvhWorker.generate` over the worker pool).
- **The `/three` gaps (2026-11-12):**
  - **Draw ranges and groups:** off a multiple of 3, ending mid-triangle or overlapping, they match three's `Mesh.raycast` step for step, in the sync and in `acceleratedRaycast` (shifted trees).
  - **GPU skinning:** with a `renderer`, the sync poses `SkinnedMesh` and morph meshes with `BvhPosePass` into `SceneTrees.setPose`, and moves top-level boxes with `SceneTrees.setDeformBounds`. Page `three-skinned` (M5): 0 disagreements over 16,384 rays at 4 frames.
  - **Scene raycast:** `BvhSceneRaycaster` + `raycastScene` return three's `intersectObject(root, true)`; `intersectsFrustum(frustum, layers)`.
- **Parity** (`docs/internals/PARITY.md`, 52 rows): 31 have, 7 partial, 9 missing, 5 won't.
- **Packaging and CI:** `tsc` builds `dist/`; `scripts/verify-artifacts.mjs` checks the licence text, worker URLs and `/three`; ESLint keeps the core engine-free (only `src/three` imports three). `.github/workflows/ci.yml` has a `wasm` job (rebuilds on Linux and macOS, `--check`, uploads the `.wasm`) and a SwiftShader GPU job (`--allow-software`; gates `bvh-lbvh-fallback`, `trace-inline`, `trace-oracle`; 15 minute limit). Neither has ever run on GitHub, because nothing has been pushed.
- **Docs:** `docs/internals/DIVERGENCES.md` (49 rows: K1-K9 the walk against tinybvh's kernel, section 6; A1-A4 the as-built departures from our design docs, section 7; 8 not ported; 8 open items), `docs/internals/MIGRATING-FROM-DDGI.md` (P3), `docs/internals/MIGRATING-FROM-REFLECTIONS-BRANCH.md`, `docs/internals/three.md`, `docs/internals/design/trace.md`, `docs/internals/WINDOWS-RUNBOOK.md` (the next pass on the RTX 3060 and Intel).

**Landed since 2026-11-12 midday** (all on `main`)
- **Comparison suite** (`compare/`, `compare/RESULTS.md`, `docs/performance/comparison.mdx`; `pnpm compare:node`): one adapter interface over @pmndrs/bvh, three-mesh-bvh, three, bvh-tree, Babylon, bvh.js and Rapier; compare pages `raycast`, `build`, `tree-quality`, `queries`, `scene`, `gpu` in the examples site. Apple M5, indicative.
  - **Wins:** Sponza CPU closest-hit 580k rays/s (three-mesh-bvh 0.57-0.75x, bvh-tree 0.12x); tree SAH cost 90 against 122/141/205; GPU 0.96-1.12x three-mesh-bvh/webgpu closest and 1.25-1.28x any-hit on Sponza and the knot; scene trees ahead of `ObjectBVH` at 10k objects.
  - **Losses:** see Open.
- **Docs site:** 34 MDX pages in `docs/`, type-checked snippets (`pnpm docs:snippets`, `docs:check`), TypeDoc at `/api`, new README; internals moved to `docs/internals/`. The docs workflow builds on push and PR and deploys only on manual dispatch.
- **three.js examples** (`examples/`, Vite site; `pnpm examples:dev`, `examples:build`, `examples:smoke`): raycast, pathtracer, skinned, instances, load-gltf, inspector.
- **Per-triangle materials and `setPoseCpu`:** `MeshInput.triangleMaterials` (records flags bits 16-31, indexed per-mesh table, merged side table); `SceneTrees.setPoseCpu` is a CPU refit for deforming meshes, with or without subgroups.
- **`/three` fold is the default:** multi-material meshes fold into one tree in the scene builder, the sync and the scene raycaster; `multiMaterial: 'groups'` splits them. `/three` also augments three's types in three-mesh-bvh's shape (`boundsTree`, `computeBoundsTree`, `firstHitOnly`).
- **Perf follow-ups:** `anyHitOrder: 'far'` (default) `| 'larger' | 'near'`, compiled into the one walk (`'larger'` is 1.73x three-mesh-bvh on outside any-hit rays, was 0.32x; costs 0.78x on merged surface rays); CPU refit 2.8-3.6x faster; tinybvh WASM heap released after big builds (`releaseAboveBytes`, pool `recycleAboveBytes`; 1,707 to 391 MiB retained).
- **CI:** `profile=ci` for `trace-oracle` and `trace-inline` on the SwiftShader job (all variants and cases, about a tenth of the rays; pass rules unchanged; per-phase timings in the report).
- **reflections** now runs on bvh `main` and reads `dist/`: run `pnpm build` after any source change it needs.

**Consumers** (survey of ~/Developer, 2026-11-11)
- **ddgi and its rc-\* copies:** on the frozen copy; they switch at P3.
- **reflections:** now on bvh `main`, reading `dist/` (see the build note above). After the next `pnpm build` it gets quantized scene nodes by default; its `assembleTraceScene` and `traceModuleWgsl` defaults agree, so only code that reads `nodes` with the 64-byte stride needs `nodeFormat: 'f32'` (`docs/internals/MIGRATING-FROM-REFLECTIONS-BRANCH.md`, section 6).
- **Denoiser, upscaler, ddgi-research:** use three-mesh-bvh `/webgpu`; can switch through `/three`.
- **three-bvh-csg, homefig, drei, three-gpu-pathtracer:** the CPU features they use exist (shapecast, `bvhcast`, `ExtendedTriangle`, closest point); none has switched.

**Open**
- **RTX 3060 surface-ray benches done** (2026-11-13, `bd8ba54`, runbook section 7), and the defaults flipped on them (see Dennis's decisions): merged on, quantized nodes on, records, the 40-byte record and `'far'` kept.
- **`triangleLeaves` default: waits for the next Windows pass.** Indexed leaves stay off until the RTX 3060 runs quantized nodes and indexed leaves together (the surface-ray pages time each variant alone against `f32` records today; a combined section, or `quantized` + `indexed` against `quantized` records, is needed). Dennis decides from that run.
- **Windows still to run:** the section 1 correctness pages (not reported in the push; the defaults changed since, so `trace-oracle`, `trace-inline` and the `scene-trees-*` pages should run on this commit), section 3 `scene-trees-sponza` (memory and the cap on D3D12, now with quantized nodes the default), compile times, the Intel UHD, and quantized and indexed together (above).
- **Publishing the docs** (Dennis): Pages settings and DNS.
- **Old worktree** `~/Developer/bvh-worktrees/reflections-integration`: reflections no longer uses it; Dennis removes it (a session couldn't delete it).
- **Compare pages left:** `skinned`, `pathtracer`, `sdf`.
- **Where others still win:** Rapier's box queries (2.3-2.7x) and closest point (1.7-2.4x); three-mesh-bvh's refit (1.6-2x) and memory (our CPU trees hold 9-10x more; GPU scenes about 7x on Sponza with quantized nodes, 355.5 against 51.2 MiB); also its CPU box queries (1.2-1.4x), furnished-room GPU closest-hit (1.1-1.2x) and bvh.js's scene build (3x).
- **Not started:** `three/sdf`, Babylon (P6), the vanilla examples, ddgi P3 (Dennis tells that session when the core is done).
- **Quantize jobs:** `SceneTrees` runs one per copied GPU tree (`copyGpuTree`); batching an update's jobs into one dispatch would cut per-tree buffers and passes.
- **`scene-trees-mid-swap`:** one unexplained failure after 405 s under a parallel `pnpm build` (message not captured; page limit 900 s); passed 3 times since. If it fails again, keep the full log.
- **Indexed leaves, not built:** WASM writer for tinybvh, indexed output from the GPU LBVH, `CpuMeshBvh` reading through positions, adopt lever with indexed scenes, binding the app's positions.
- **Quantized nodes, not built:** H-PLOC emitting them directly, packing inside the tinybvh WASM, the folded decode.
- **`bvh:wasm --check`** needs the pinned emsdk, not on every machine.
- **`/three` to do:** Windows compile cost of the pose pass, `ObjectBVH.shapecast` (PARITY 5.3), a browser look at `MeshBVHHelper`.
- **API tidy:** `export *` in `src/trace/index.ts`, batch helpers in `gpu-lbvh.ts`, merging the two layout files.
- **Later CPU queries (P8):** CPU `StaticGeometryGenerator` (PARITY 8), three-mesh-bvh-format `serialize` and `getJSONStructure` (12).


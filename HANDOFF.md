# Handoff: pmndrs/bvh after the split (2026-11-10)

This repo was split out of `DennisSmolek/webgpu-ddgi`, where the code lived in `packages/bvh`. The full plan is [docs/design/bvh-package-split.md](docs/design/bvh-package-split.md): the decisions (D1–D8), the target layout, phases P0–P9, examples, docs and tasks BX.0–BX.20. Read it first.

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

## Current state (end of 2026-11-12). Read this first.

Everything below is on `main` and local only (nothing has been pushed). The earlier "Update" paragraphs are folded in here; the file had no history sections below this one when it was rewritten.

**Dennis's decisions**
- **Source of truth:** this repo, for BVH and walk code. webgpu-ddgi's copy is frozen until P3 (split point `d288adf`).
- **Commits:** local only, never pushed. Opus writes the code and does the reviews; Sonnet or Haiku take the mechanical tasks.
- **Walk API:** WGSL names `bvh*`; filter hook `bvhTrianglePasses(BvhCandidate)` (the candidate has `query` and `rayMask`; ddgi's materials stay in ddgi); 8-bit instance masks, compiled out with `rayMasks: false`.
- **Package scope:** every project that needs a BVH, not just ddgi (see Consumers).
- **Licence:** MIT, Dennis Smolek (`LICENSE`).
- **Build-peak cap:** `SceneTrees` `gpuMaxBatchBytes` defaults to 256 MiB; `Infinity` turns it off.
- **Off by default until the RTX 3060 is measured:** `mergeStaticInstances`, `triangleLeaves: 'indexed'`, `nodeFormat: 'quantized'`. The 40-byte hit record is on by default; `hitNormal: false` gives 24 bytes.
- **Wasm reference:** the Linux build; a Windows build differs in 2 `__FILE__` separator bytes (`docs/design/bvh-builders.md`).

**On `main`** (lint, typecheck, build and the licence check pass; every correctness page passes on the Apple M5)
- **GPU core, `GPU_LAYOUT_VERSION` 2 (unreleased):** ported refit kernels and the Metal watertight fix `bvhEdgeFunction` (rays through shared edges 4,513 per 2M to 0); `BvhHit.normal` and `BvhHit.material`; GPU page runner `pnpm test:gpu`, `pnpm test:gpu:page <name>` (`tests/gpu/README.md`).
- **`mergeStaticInstances`:** merged world-space tree, exact hit identity (triangle word 11 = source instance, flag bit 1 flips the face, instance bit 3 marks merged; app flags may use bit 0 only). M5 under the radiance-cache bench lock: Sponza surface rays 1.85x closest-hit, 1.36x any-hit; Bistro interior 1.22x and 1.11x. Pages `trace-bench-surface-*`; rows in `bench/results/trace.csv`.
- **`SceneTrees`** (`src/scene/`): GPU H-PLOC trees first, tinybvh trees from `@pmndrs/bvh/worker` swapped in per mesh; growable storage; `GpuTopLevel`; runtime add/remove, merged static groups, deforming meshes. Sponza: first trees 100-118 ms, all tinybvh trees about 370 ms.
- **Build-peak cap:** capped H-PLOC batches built one after another on one scratch pool, stored between updates. Sponza on the M5: peak 1,483 to 711 MiB, settled 470 to 459, first trees 171 to 166 ms. RTX 3060 and Intel not run. Breakdown: `docs/design/memory.md`; the other levers are BX.20.
- **Indexed leaves** (off by default; `docs/design/indexed-leaves.md`): `triangleLeaves: 'records' | 'indexed'`, `positionLayout: 'vec4' | 'packed'`; 16-byte triangles, per-mesh 16-byte header, library-owned positions (exact-bit weld on; `weld: false`; opt-in `weldTolerance`), merged groups via a side table (capped at 65,536 sources), format in header word 7. Sponza 448.1 to 318.0 MiB beyond geometry (306.7 packed), Bistro exterior 210.1 to 155.5. M5 speed (indicative): `vec4` 0.83-0.99x records, packed 0.75-0.92x.
- **Quantized nodes** (off by default; `docs/design/compressed-nodes.md`, option A): `nodeFormat: 'f32' | 'quantized'`; 32-byte nodes, per-mesh grid, frame slot before the root, outward rounding, CPU and GPU quantizers bit-identical, deforming meshes refit an `f32` copy and re-quantize, bits 8-15 of header word 7. Sponza 448.1 to 355.5 MiB (225.5 with indexed), Bistro exterior 210.1 to 169.9 (115.3); SAH +0.01-0.66%; M5 speed 0.93-0.97x `f32` (indicative).
- **`CpuMeshBvh`** (`src/query/`, root export): the one CPU walk (`raycast`, `raycastFirst`, `traverse`, `refit`, checked `serialize`), plus `CpuSceneBvh`.
  - **Shapecast and `bvhcast`:** `shapecast` with three-mesh-bvh's contract, `intersectsBox`/`Sphere`/`Geometry`, `bvhcast` (dual walk step for step), engine-free `ExtendedTriangle`/`OrientedBox` (bit-for-bit with three-mesh-bvh). `bench/query-parity.ts` (homefig capsule controller, three-bvh-csg): full agreement, 0.93x and 0.94x its time.
  - **Closest point:** `closestPointToPoint`, `closestPointToGeometry`; `bench/closest-point.ts`: 0.92x for points in the open, 1.1-1.2x near surfaces, 1.08x geometry against geometry.
  - **Debug helpers:** `getBVHExtremes`, `validateBounds`, `estimateMemoryInBytes`; `/three` has `BVHHelper`/`MeshBVHHelper` (segment counts tested in Node, not yet looked at in a browser).
  - **Scene queries:** `CpuSceneBvh` (scene raycast, `intersectsFrustum`/`intersectsBox`).
- **`@pmndrs/bvh/three`** (three is a peer, `>=0.187.0 <0.188.0`; `docs/three.md`): `bvhSceneFromObject3D`, `SceneTreesObject3DSync`, `BvhTraceNodes` (binds through `threeInternals`, checked at run time), `computeBoundsTree` with three-mesh-bvh's options, `acceleratedRaycast`, `geometry.boundsTree` as `BvhBoundsTree` (`MeshBVH` alias; `GenerateMeshBvhWorker.generate` over the worker pool).
- **The `/three` gaps (2026-11-12):**
  - **Draw ranges and groups:** off a multiple of 3, ending mid-triangle or overlapping, they match three's `Mesh.raycast` step for step, in the sync and in `acceleratedRaycast` (shifted trees).
  - **GPU skinning:** with a `renderer`, the sync poses `SkinnedMesh` and morph meshes with `BvhPosePass` into `SceneTrees.setPose`, and moves top-level boxes with `SceneTrees.setDeformBounds`. Page `three-skinned` (M5): 0 disagreements over 16,384 rays at 4 frames.
  - **Scene raycast:** `BvhSceneRaycaster` + `raycastScene` return three's `intersectObject(root, true)`; `intersectsFrustum(frustum, layers)`.
- **Parity** (`docs/PARITY.md`, 51 rows): 31 have, 7 partial, 10 missing, 3 won't.
- **Packaging and CI:** `tsc` builds `dist/`; `scripts/verify-artifacts.mjs` checks the licence text, worker URLs and `/three`; ESLint keeps the core engine-free (only `src/three` imports three). `.github/workflows/ci.yml` has a `wasm` job (rebuilds on Linux and macOS, `--check`, uploads the `.wasm`) and a SwiftShader GPU job (`--allow-software`; gates `bvh-lbvh-fallback`, `trace-inline`, `trace-oracle`; 15 minute limit). Neither has ever run on GitHub, because nothing has been pushed.
- **Docs:** `docs/DIVERGENCES.md` (32 rows plus 8 "needs Dennis" items), `docs/MIGRATING-FROM-DDGI.md` (P3), `docs/MIGRATING-FROM-REFLECTIONS-BRANCH.md`, `docs/three.md`, `docs/design/trace.md`, `docs/WINDOWS-RUNBOOK.md` (the next pass on the RTX 3060 and Intel).

**Consumers** (survey of ~/Developer, 2026-11-11)
- **ddgi and its rc-\* copies:** on the frozen copy; they switch at P3.
- **reflections:** linked to the old-API branch `reflections-integration`; can move to `main` (see its migration doc).
- **Denoiser, upscaler, ddgi-research:** use three-mesh-bvh `/webgpu`; can switch through `/three`.
- **three-bvh-csg, homefig, drei, three-gpu-pathtracer:** the CPU features they use exist (shapecast, `bvhcast`, `ExtendedTriangle`, closest point); none has switched.

**Open**
- **RTX 3060 and Intel have run none of the 2026-11-12 work.** Steps and decision rules: `docs/WINDOWS-RUNBOOK.md`. It decides three defaults (`mergeStaticInstances`, indexed leaves, quantized nodes), the 40-byte hit record's cost (M5 only: ratios 0.96-1.15, medians 1.02-1.11) and the cap's effect (`docs/design/memory.md`).
- **`scene-trees-mid-swap`:** one unexplained failure after 405 s under a parallel `pnpm build` (message not captured; the page limit is 900 s). It has passed 3 times since (about 170 s, 0 disagreements over 87.7M rays). If it fails again, keep the full log.
- **Indexed leaves, not built:** WASM writer for tinybvh (it converts after), indexed output from the GPU LBVH, `CpuMeshBvh` reading corners through positions (it expands to records), the adopt lever with indexed scenes, binding the app's positions (option A).
- **Quantized nodes, not built:** H-PLOC emitting quantized nodes directly, packing inside the tinybvh WASM (workers pack in JS: Sponza's tinybvh trees arrive in 439 ms against 351), the folded decode. Performance: `SceneTrees` runs one quantize job per copied GPU tree (`copyGpuTree`); batching an update's jobs into one dispatch would cut per-tree buffers and passes.
- **`bvh:wasm --check`** needs the pinned emsdk, which isn't downloaded on every machine.
- **Still to do in `/three`:** Windows compile cost of the pose pass (one shader per variant, not measured on Windows), `ObjectBVH.shapecast` (PARITY 5.3), a browser look at `MeshBVHHelper`.
- **API tidy:** `export *` in `src/trace/index.ts`, the batch helpers in `gpu-lbvh.ts`, folding the two layout files into one.
- **Next phases:** Babylon (P6), P4 examples and docs site, the comparison suite (P5), the rest of CPU queries (P8): a CPU `StaticGeometryGenerator` (PARITY 8), `ObjectBVH.shapecast` (9), three-mesh-bvh-format `serialize` and `getJSONStructure` (12).

**Needs Dennis**
- **Indexed leaves:** the vertex base lives in a per-mesh header, not added into the records as the design said (records stay relocatable); merged groups capped at 65,536 sources.
- **Quantized nodes:** the frame travels in the tree's first node slot, not derived from `bounds` as the design said, so no frame rule is in the file format; GPU tree ranges gained `parentOffset`.
- **`/three` ranges:** three r187 reports the triangle that reads past the end of the index at a `NaN` distance (and leaves its sorted list out of order); we report nothing there. Looks like a three bug worth filing.
- **`/three` skinning:** the deform box is per mesh and set from the CPU (no GPU readback of the refit root); per-instance morphs (`morphTexture`) aren't read; CPU picking on skinned meshes stays on three's raycast.
- **CPU scene raycast:** built from the three objects, not from `SceneTrees` (GPU-only trees); layers 8-31 aren't seen; hidden objects are hit by default, as three's `Raycaster`.
- **CPU tree:** `strategy` CENTER/AVERAGE build the SAH tree; `maxDepth` accepted, not honoured.
- **Shapecast, `bvhcast`:** `/three`'s `OrientedBox` is its own class (not a `Box3` subclass); `intersectsGeometry` against a tree runs a `bvhcast` (three-mesh-bvh 0.9.18 shapecasts the other tree from each leaf; same answers, 1.5x faster here); `ExtendedTriangle.set`/`copy` set `needsUpdate`; `bvhcast` copies three-mesh-bvh's depth bookkeeping.
- **Closest point:** `maxThreshold` culls boxes, not triangles, as three-mesh-bvh does; `closestPointToGeometry`'s `faceIndex`es are original triangles in every case (three-mesh-bvh 0.9.18's indirect mode returns tree positions, which looks like a bug there).
- **Utilities:** `getBVHExtremes` returns one entry (one root); `splits` counts an inferred axis (or drop the field, or store the axis); `validateBounds` reads the source geometry, so it reports edits not yet refitted.
- **Helper:** `MeshBVHHelper` is exported without three-mesh-bvh's deprecation warning; no `BatchedMesh` `boundsTrees` lookup (PARITY 5.5).

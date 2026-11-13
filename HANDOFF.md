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
- **Off by default, now measured on the RTX 3060 (2026-11-13), defaults await Dennis:** `mergeStaticInstances`, `triangleLeaves: 'indexed'`, `nodeFormat: 'quantized'`. The 40-byte hit record is on by default; `hitNormal: false` gives 24 bytes. `anyHitOrder` defaults to `'far'`. The numbers are in `docs/internals/WINDOWS-RUNBOOK.md`, section 7.
- **Wasm reference:** the Linux build; a Windows build differs in 2 `__FILE__` separator bytes (`docs/internals/design/bvh-builders.md`).

**On `main`** (lint, typecheck, build and the licence check pass; every correctness page passes on the Apple M5)
- **GPU core, `GPU_LAYOUT_VERSION` 2 (unreleased):** ported refit kernels and the Metal watertight fix `bvhEdgeFunction` (rays through shared edges 4,513 per 2M to 0); `BvhHit.normal` and `BvhHit.material`; GPU page runner `pnpm test:gpu`, `pnpm test:gpu:page <name>` (`tests/gpu/README.md`).
- **`mergeStaticInstances`:** merged world-space tree, exact hit identity (triangle word 11 = source instance, flag bit 1 flips the face, instance bit 3 marks merged; app flags may use bit 0 only). M5 under the radiance-cache bench lock: Sponza surface rays 1.85x closest-hit, 1.36x any-hit; Bistro interior 1.22x and 1.11x. Pages `trace-bench-surface-*`; rows in `bench/results/trace.csv`.
- **`SceneTrees`** (`src/scene/`): GPU H-PLOC trees first, tinybvh trees from `@pmndrs/bvh/worker` swapped in per mesh; growable storage; `GpuTopLevel`; runtime add/remove, merged static groups, deforming meshes. Sponza: first trees 100-118 ms, all tinybvh trees about 370 ms.
- **Build-peak cap:** capped H-PLOC batches built one after another on one scratch pool, stored between updates. Sponza on the M5: peak 1,483 to 711 MiB, settled 470 to 459, first trees 171 to 166 ms. RTX 3060 and Intel not run. Breakdown: `docs/internals/design/memory.md`; the other levers are BX.20.
- **Indexed leaves** (off by default; `docs/internals/design/indexed-leaves.md`): `triangleLeaves: 'records' | 'indexed'`, `positionLayout: 'vec4' | 'packed'`; 16-byte triangles, per-mesh 16-byte header, library-owned positions (exact-bit weld on; `weld: false`; opt-in `weldTolerance`), merged groups via a side table (capped at 65,536 sources), format in header word 7. Sponza 448.1 to 318.0 MiB beyond geometry (306.7 packed), Bistro exterior 210.1 to 155.5. M5 speed (indicative): `vec4` 0.83-0.99x records, packed 0.75-0.92x.
- **Quantized nodes** (off by default; `docs/internals/design/compressed-nodes.md`, option A): `nodeFormat: 'f32' | 'quantized'`; 32-byte nodes, per-mesh grid, frame slot before the root, outward rounding, CPU and GPU quantizers bit-identical, deforming meshes refit an `f32` copy and re-quantize, bits 8-15 of header word 7. Sponza 448.1 to 355.5 MiB (225.5 with indexed), Bistro exterior 210.1 to 169.9 (115.3); SAH +0.01-0.66%; M5 speed 0.93-0.97x `f32` (indicative).
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
- **Parity** (`docs/internals/PARITY.md`, 51 rows): 31 have, 7 partial, 10 missing, 3 won't.
- **Packaging and CI:** `tsc` builds `dist/`; `scripts/verify-artifacts.mjs` checks the licence text, worker URLs and `/three`; ESLint keeps the core engine-free (only `src/three` imports three). `.github/workflows/ci.yml` has a `wasm` job (rebuilds on Linux and macOS, `--check`, uploads the `.wasm`) and a SwiftShader GPU job (`--allow-software`; gates `bvh-lbvh-fallback`, `trace-inline`, `trace-oracle`; 15 minute limit). Neither has ever run on GitHub, because nothing has been pushed.
- **Docs:** `docs/internals/DIVERGENCES.md` (32 rows plus 8 "needs Dennis" items), `docs/internals/MIGRATING-FROM-DDGI.md` (P3), `docs/internals/MIGRATING-FROM-REFLECTIONS-BRANCH.md`, `docs/internals/three.md`, `docs/internals/design/trace.md`, `docs/internals/WINDOWS-RUNBOOK.md` (the next pass on the RTX 3060 and Intel).

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
- **reflections:** now on bvh `main`, reading `dist/` (see the build note above).
- **Denoiser, upscaler, ddgi-research:** use three-mesh-bvh `/webgpu`; can switch through `/three`.
- **three-bvh-csg, homefig, drei, three-gpu-pathtracer:** the CPU features they use exist (shapecast, `bvhcast`, `ExtendedTriangle`, closest point); none has switched.

**Open**
- **RTX 3060 surface-ray benches done** (2026-11-13, `bd8ba54`, runbook section 7):
  - merged 1.17-1.67x;
  - quantized 0.98-1.22x, faster for closest hits on NVIDIA (the M5 was 0.93-0.97);
  - indexed `vec4` 0.92-0.97x, packed 0.82-0.95x;
  - 40-byte record 0.88-0.94x (the M5 was 0.96-1.15);
  - `'larger'` 1.11-1.48x on instanced trees, 0.80-0.87x on merged surface rays.

  The defaults wait for Dennis's call. After it, one pass flips the code and updates the user docs that say "until the RTX 3060 is measured".
- **Windows still to run:** the section 1 correctness pages (not reported in the push), section 3 `scene-trees-sponza` (memory and the cap on D3D12), compile times, the Intel UHD, and quantized and indexed together.
- **Publishing the docs** (Dennis): Pages settings and DNS.
- **Old worktree** `~/Developer/bvh-worktrees/reflections-integration`: reflections no longer uses it; Dennis removes it (a session couldn't delete it).
- **Compare pages left:** `skinned`, `pathtracer`, `sdf`.
- **Where others still win:** Rapier's box queries (2.3-2.7x) and closest point (1.7-2.4x); three-mesh-bvh's refit (1.6-2x) and memory (we hold 9-10x more in the default layout); also its CPU box queries (1.2-1.4x), furnished-room GPU closest-hit (1.1-1.2x) and bvh.js's scene build (3x).
- **Not started:** `three/sdf`, Babylon (P6), the vanilla examples, ddgi P3 (Dennis tells that session when the core is done).
- **Quantize jobs:** `SceneTrees` runs one per copied GPU tree (`copyGpuTree`); batching an update's jobs into one dispatch would cut per-tree buffers and passes.
- **`scene-trees-mid-swap`:** one unexplained failure after 405 s under a parallel `pnpm build` (message not captured; page limit 900 s); passed 3 times since. If it fails again, keep the full log.
- **Indexed leaves, not built:** WASM writer for tinybvh, indexed output from the GPU LBVH, `CpuMeshBvh` reading through positions, adopt lever with indexed scenes, binding the app's positions.
- **Quantized nodes, not built:** H-PLOC emitting them directly, packing inside the tinybvh WASM, the folded decode.
- **`bvh:wasm --check`** needs the pinned emsdk, not on every machine.
- **`/three` to do:** Windows compile cost of the pose pass, `ObjectBVH.shapecast` (PARITY 5.3), a browser look at `MeshBVHHelper`.
- **API tidy:** `export *` in `src/trace/index.ts`, batch helpers in `gpu-lbvh.ts`, merging the two layout files.
- **Later CPU queries (P8):** CPU `StaticGeometryGenerator` (PARITY 8), three-mesh-bvh-format `serialize` and `getJSONStructure` (12).

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

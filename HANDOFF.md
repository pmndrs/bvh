# Handoff: pmndrs/bvh after the split (2026-11-10)

This repo was split out of `DennisSmolek/webgpu-ddgi`, where the code lived in `packages/bvh`. The full plan is [docs/design/bvh-package-split.md](docs/design/bvh-package-split.md): the decisions (D1–D8), the target layout, phases P0–P9, examples, docs and tasks BX.0–BX.20. Read it first.

## Current state (end of 2026-11-11). Read this first; the sections further down are history.

**Dennis's decisions:**
- **Source of truth:** this repo, for BVH and walk code. webgpu-ddgi's copy is frozen until P3, and its split point is `d288adf`.
- **Commits:** local only, never pushed. Opus writes the code and does the reviews; Sonnet or Haiku take the mechanical tasks.
- **Walk API:**
  - WGSL names use `bvh*`.
  - Filter hook: `bvhTrianglePasses(BvhCandidate)`, where the candidate includes `query` and `rayMask`. ddgi's materials stay in ddgi.
  - 8-bit instance masks, compiled out with `rayMasks: false`.
- **`mergeStaticInstances`** comes before the release. It stays off by default until the RTX 3060 has been measured.
- **Package scope:** it should serve every project that needs a BVH, not just ddgi. See "Consumers" below.

**On `main`** (306 tests; lint, typecheck, build and the licence check pass; every correctness page passes on the Apple M5):
- **Ported from ddgi `main` after the split:** shared refit kernels, and the Metal watertight fix `bvhEdgeFunction`. On the M5, rays slipping through shared edges went from 4,513 per 2M to 0.
- **GPU page runner:** `pnpm test:gpu`, `pnpm test:gpu:page <name>`; see `tests/gpu/README.md`. Scene pages need `BVH_ASSETS_DIR=~/Developer/webgpu-ddgi-radiance-cache/assets` and `BVH_MANIFESTS_DIR=~/Developer/webgpu-ddgi-radiance-cache/scenes/manifests`.
- **Walk API for 0.1**, at `GPU_LAYOUT_VERSION` 2 (unreleased):
  - names, filter and masks as above;
  - `BvhHit.normal` and `BvhHit.material` (material + 1 in instance flags bits 16-31);
  - a 40-byte hit record by default, or 24 bytes with `hitNormal: false`.
- **`mergeStaticInstances`:** a merged world-space tree with exact hit identity. Triangle word 11 holds the source instance, flag bit 1 flips the face, and instance bit 3 marks a merged instance. App triangle flags may use bit 0 only.
  - Measured on the package under the radiance-cache lane's bench lock: Sponza surface rays 1.85x closest-hit, 1.36x any-hit; Bistro interior 1.22x and 1.11x.
  - Pages: `trace-bench-surface-*`. Rows are in `bench/results/trace.csv`.
- **`SceneTrees`** (`src/scene/`):
  - GPU H-PLOC trees first, then tinybvh trees from `@pmndrs/bvh/worker` swapped in per mesh;
  - growable storage;
  - `GpuTopLevel`;
  - meshes and instances added or removed at runtime, merged static groups, deforming meshes.
  - On Sponza: first trees in 100-118 ms, every tinybvh tree in about 370 ms. Peak memory is 1,483 MB, during the H-PLOC build itself.
- **`@pmndrs/bvh/three`** (three is a peer, `>=0.187.0 <0.188.0`); see `docs/three.md`:
  - `bvhSceneFromObject3D` and `SceneTreesObject3DSync`;
  - `BvhTraceNodes`, for TSL and `wgslFn`. It binds through three internals, collected in `threeInternals`, and checks that binding at run time;
  - `computeBoundsTree` and `acceleratedRaycast`, which run at three-mesh-bvh's speed.
- **Packaging:**
  - `tsc` builds `dist/`;
  - `scripts/verify-artifacts.mjs` checks that the licence text, the worker URLs and `/three` survive the build;
  - ESLint enforces the engine-free core: only `src/three` may import three;
  - CI is in `.github/workflows/ci.yml`. Its `wasm` and SwiftShader jobs have never run, because nothing has been pushed.
- **Docs:**
  - `docs/DIVERGENCES.md`: 32 rows, plus 8 items marked "needs Dennis";
  - `docs/MIGRATING-FROM-DDGI.md` (P3);
  - `docs/MIGRATING-FROM-REFLECTIONS-BRANCH.md`;
  - `docs/three.md`;
  - `docs/design/trace.md`.

**Consumers** (from a survey of ~/Developer on 2026-11-11):
- **ddgi and its rc-\* copies:** still on the frozen copy. They switch at P3.
- **reflections:** linked to the old-API branch `reflections-integration`. It can move to `main`; see its migration doc.
- **Denoiser, upscaler, ddgi-research:** use three-mesh-bvh `/webgpu`. They can switch through `/three`, as `docs/three.md` describes.
- **three-bvh-csg, homefig, drei, three-gpu-pathtracer:** need CPU queries (P8).

**Open:**
- **RTX 3060 and Intel:** none of today's work has run on them. The radiance-cache lane runs its RTX pass for the merged tree.
- **The 40-byte hit record's cost:** not measured.
- **The `LICENSE` file:** missing, and the holder (Dennis Smolek or Poimandres) is undecided.
- **`bvh:wasm --check`:** needs the pinned emsdk, which isn't downloaded.
- **Peak memory:** smaller H-PLOC batches are the next lever to measure.
- **Still to do in `/three`:** draw ranges and groups that don't start on a multiple of 3, skinned and morphed meshes, a CPU scene raycast, a helper, and Windows compile cost.
- **API tidy:** `export *` in `src/trace/index.ts`, the batch helpers in `gpu-lbvh.ts`, and folding the two layout files into one.
- **Next phases:** Babylon (P6), the P4 examples and docs site, the comparison suite (P5), and CPU queries (P8).

## Where things stood on 2026-11-10

- **P1 is done (BX.1 and BX.2), but not tested after BX.2.**
  - BX.1 (`e181f68`): renamed the shared types into `src/mesh-input.ts`, curated `src/index.ts`, added `GPU_LAYOUT_VERSION` and `serializeLayout`/`deserializeLayout`. It was tested in webgpu-ddgi on an RTX 3060: the unit tests and the GPU pages passed, and both oracles reported 0 mismatches.
  - BX.2 (`638e108`): moved the tracer code out of ddgi-core into `src/trace/`:
    - the layout's WGSL structs, the WGSL walk (`giTraceClosest`/`giTraceAny`) and its CPU port;
    - the oracle;
    - tinybvh trees in the tracer's layout, the CPU SAH builder, the top-level builder and the worker wrappers.

    Only typecheck and lint ran in webgpu-ddgi. Dennis asked to leave the tests to you. The commit only moved files and rewrote imports, with no logic or WGSL changes.
- **P2 has started.** The history was extracted with `git filter-repo` (71 commits). A bootstrap commit adds a standalone `package.json`, `tsconfig.json`, `vitest.config.ts` and `.gitignore`, plus the `native/build-wasm.ts` root fix. Nothing has been installed, built or run in this repo yet.
- **webgpu-ddgi:** branch `bvh-bx1` there holds BX.1 and BX.2. It isn't merged into ddgi's `m3` yet. ddgi keeps consuming the in-repo `packages/bvh` until P3.

### How the history was cut

```
--path packages/bvh/  --path tools/bvh/  --path-glob 'tools/gpu/pages/bvh-*'  --path-glob 'tools/gpu/pages/trace-*'
--path docs/design/{bvh-builders,bvh-framework,bvh-package-split,trace}.md  --path bench/results/trace.csv  --path THIRD_PARTY.md
--path packages/ddgi-core/src/platform/trace/<the 24 moved files>
--path-rename packages/ddgi-core/src/platform/trace/:src/trace/   --path-rename packages/bvh/:
--path-rename tools/gpu/pages/:tests/gpu/pages/   --path-rename tools/bvh/:tools/
```

The commit messages are ddgi's and refer to ddgi task ids (T0.10, B0, …), so read them against webgpu-ddgi's TASKS.md.

## Known broken or unfinished (in order)

1. **Run what was skipped:** `pnpm install`, `pnpm typecheck`, `pnpm test`.
   - Expect the moved tests (`src/trace/*.test.ts`) to pass as they did in ddgi.
   - The worker wrappers (`mesh-bvh-async.ts`, `top-level-async.ts`) fall back to running inline under Node.
2. **`tests/gpu/pages/*` and `tools/*` still import webgpu-ddgi paths** (`packages/ddgi-core/...`, `packages/bvh/src/...`) and the ddgi GPU page runner (`tools/gpu/run.ts`). That runner isn't in this repo.
   - Port the runner from webgpu-ddgi, or write a small Vite multi-page runner. Repoint the imports to `src/`.
   - The `trace-bench` pages compare against ddgi's platform tracer (`platform-bench.ts`). Drop that part, or keep it as a ddgi-side bench.
   - `tsconfig.json` leaves `tests/` and `tools/` out until this is done.
3. **`native/build-wasm.ts`** now treats this repo's root as both the package root and the repository root.
   - The generated header line in `src/tinybvh-wasm.ts` still says `packages/bvh/native/build-wasm.ts`. If you change that string in the generator, regenerate the file, or `pnpm bvh:wasm --check` will report a difference.
   - The committed `.wasm` is the Linux build; CI reproduces it on Ubuntu and macOS. A Windows build differs in two `__FILE__` path-separator bytes only (see `docs/design/bvh-builders.md`).
4. **`THIRD_PARTY.md` is ddgi's whole file.** Keep only these rows:
   - tinybvh;
   - tinybvh's `BuildHPLOC`;
   - the Emscripten runtime;
   - HIPRT;
   - three-mesh-bvh (comparison only);
   - the scene assets the examples and benches use (Sponza, Bistro, Breakfast Room).

   Drop gputex, RTXGI and the like. Fix the `packages/bvh/` paths. The licence texts are in `licenses/`.
5. **Docs paths:** `README.md` and `docs/design/*.md` still say `packages/bvh/...` and refer to ddgi-core. `trace.md` is ddgi's design doc for the tracer, so trim it to the parts about the walk and the layout.
6. **API leftovers:**
   - `src/trace/index.ts` uses `export *`. The rest of `src/index.ts` is a curated list, so curate this one too.
   - `/worker` isn't its own subpath yet (D3).
   - The GPU batch helpers that `gpu-hploc.ts` imports from `gpu-lbvh.ts` (`planMeshBatches`, `placeMeshes`, `USAGE`, …) should move to a neutral file.
   - `src/trace/layout.ts` re-exports `src/gpu-layout.ts` under the tracer's older names. The two layout files can be folded into one, matching `layout/` in the plan's target layout.
7. **Then the rest of P2:**
   - a library build (ESM plus `.d.ts`; keep the inlined `.wasm` and its `/*! */` licence comment);
   - ESLint with the no-engine-imports rule for `src/` (copy it from webgpu-ddgi's `eslint.config.mjs`);
   - CI;
   - `0.1.0-alpha` on npm under the `next` tag.

   After that comes P3: ddgi switches to the published package.

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
- **Models:** Opus for code and review, Sonnet for small tasks and tests.

## First numbers for P9 (memory)

On Sponza, from ddgi's tree-swap page:

| State | Total |
| --- | --- |
| tinybvh trees only | 470 MB (nodes 194 MB, triangle records 276 MB) |
| While the first GPU-built trees are in use | 1.16 GB (666 MB of H-PLOC build output held until the swap) |
| After the swap | 493 MB |

The triangle records and holding the build output until the swap are the first levers to look at.

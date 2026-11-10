# Handoff: pmndrs/bvh after the split (2026-11-10)

This repo was split out of `DennisSmolek/webgpu-ddgi`, where the code lived in `packages/bvh`. The full plan is [docs/design/bvh-package-split.md](docs/design/bvh-package-split.md): the decisions (D1–D8), the target layout, phases P0–P9, examples, docs and tasks BX.0–BX.20. Read it first.

## Where things stand

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
   - The `.wasm` has only been compared on Windows. The plan has the check run on Ubuntu and macOS in CI.
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

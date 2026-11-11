# Extracting `@pmndrs/bvh` into `pmndrs/bvh`

Status: **agreed by Dennis 2026-11-09**; BX.1 in progress on branch `bvh-bx1`. The target repo **[pmndrs/bvh](https://github.com/pmndrs/bvh) exists** (created 2026-11-09, private, empty until P2), with decisions D1–D8 answered (below) and two follow-up phases added (P8 feature parity with three-mesh-bvh, P9 memory audit). It replaces the split outline of 2026-11-06 (in this file's history), which said what had to be true before a split. This plan adds what the new repo should look like, the docs, the examples and the comparison against three-mesh-bvh, and puts it all in order. A copy lives in the ddgi research repo (`plans/pmndrs-bvh-extraction.md`).

## Goal

`pmndrs/bvh` is a WebGPU-first BVH library that works with any engine. It builds trees on the CPU (tinybvh in WebAssembly) or on the GPU (H-PLOC, LBVH), refits them on the GPU, and ships the WGSL ray walk that reads them. three.js is where most users are, so it is the first-class integration. The library is not a three.js library, though:

- the core runs on plain WebGPU and in Node;
- the vanilla examples come first;
- a Babylon example shows that the "any engine" claim holds.

**Priority: the core and three.** Dennis has other tools that need a BVH at this level, and they shouldn't have to pull it out of the GI repo. So P1–P3 (prove the boundary, extract, ddgi switches) come first and shouldn't wait on the examples. Babylon is kept basic.

The examples should make three cases obvious:

1. **CPU vs GPU builds.** Build each way, side by side, with the cost of each.
2. **Workers.** Builds off the main thread, CPU and GPU, with a frame-time graph showing the main thread stays smooth.
3. **Against three-mesh-bvh.** Its classic examples rebuilt on our trees, with the same rays and scenes.

> Paths in the sections below describe webgpu-ddgi as it was when this plan was written, unless they are in the target layout. This repo's current layout is in the root README and HANDOFF.md.

## Where things stand (2026-11-09, `webgpu-ddgi@m2` 5de856b)

**In `packages/bvh` today**
- 35 files with 27 commits of history (since 2026-11-04).
- Imports nothing from ddgi or any engine, and lint enforces it.
- The `.wasm` is inlined, so it loads the same way in a worker, on a page, in Node and in any bundler.
- Contents:
  - tinybvh WASM (CPU);
  - GPU LBVH and GPU H-PLOC, each with a CPU twin;
  - GPU top-level refit and BLAS refit;
  - radix sort and scan;
  - the layout checker and the batch planner;
  - the `createMeshLayoutBuilder` fallback chooser.
- Unit tests, GPU pages (`tools/gpu/pages/bvh-*`), licence texts, and a reproducible `.wasm` build (`pnpm bvh:wasm --check`).
- A user README (`packages/bvh/README.md`) and two design docs (`docs/design/bvh-builders.md`, `bvh-framework.md`).

**Still in ddgi-core, though not DDGI-specific** (`packages/ddgi-core/src/platform/trace/`)
- The layout's WGSL structs (`layout.ts`).
- The WGSL ray walk (`bvhTraceClosest`, `bvhTraceAny`) and its watertight triangle test (`wgsl.ts`).
- A CPU port of that walk, step for step (`cpu-traversal.ts`).
- The oracle (`oracle-compare.ts`, `oracle-fixtures.ts`, `reference-tracer.ts`).
- Turning a tree's shape into the layout (`mesh-bvh-topology.ts`).
- The top level's CPU shape builder and worker (`top-level-shape.ts`, `top-level-worker.ts`).
- The tinybvh worker wrapper (`mesh-bvh-async.ts`, `mesh-bvh-worker.ts`).
- The trace bench pages and `bench/results/trace.csv`.

**Gaps a standalone library would have today**
- **No build output.** `package.json` exports `.ts` sources and is `private`.
- **No CPU queries for users.** There is no raycast, closest point or shapecast on the CPU. Every classic three-mesh-bvh example depends on these.
- **No serialization** (saving a built tree and loading it again).
- **No helpers for sharing a GPU device with an engine.**
- **No CI.**
- **API rough edges** (from the outline):
  - H-PLOC returned types named after the LBVH (`GpuLbvhTrees`) (fixed in BX.1);
  - `TinyBvhInput` doubled as the general mesh input (fixed in BX.1);
  - the `./*` export exposes internals;
  - the refits need a CPU copy of the tree (`setTree`).

**Outside facts checked today**
- `pmndrs/bvh` doesn't exist on GitHub, and `@pmndrs/bvh` isn't on npm. Both names are free.
- three-mesh-bvh (0.9.x) has its own WebGPU path: `src/webgpu/` holds WGSL ray functions, TSL nodes and a `ClusteredBVH`. It also has two worker builders, `GenerateMeshBVHWorker` and `ParallelMeshBVHWorker`. The parallel one uses SharedArrayBuffer, which needs cross-origin isolation, so its examples ship `coi-serviceworker.js`. This means the comparison can be WebGPU against WebGPU and worker against worker, not just our GPU against their CPU.

## Decisions (Dennis, 2026-11-09)

| # | Question | Decided | Notes |
| --- | --- | --- | --- |
| D1 | Does the ray walk move into the package? | **Yes, both walks**: the WGSL walk and its CPU port, with the oracle. ddgi-core imports its walk from the package. | Dennis: "Seems like we have 2." Read as both walks moving; confirm if that's wrong. |
| D2 | CPU queries (raycast, closest point, shapecast) over the same layout? | **Yes, in two steps.** Q1: closest-hit and any-hit raycast, grown from `cpu-traversal.ts`, plus `three` `Mesh.raycast` acceleration. Q2: closest point and a generic shapecast. | **WebGL2 is not a target.** It isn't ruled out, and the CPU path happens to work there, but no work goes into it. |
| D3 | Repo shape | **One package with subpaths**: `@pmndrs/bvh`, `/three`, `/babylon`, `/worker`. three and Babylon are optional peer dependencies. | Matches ddgi's "one package, many import paths". |
| D4 | Keep the git history? | **Yes**, with `git filter-repo`. | |
| D5 | How far does Babylon go? | **Super basic: "it works" for now.** A Sonnet subagent does the spike and the examples. If it gets stuck, it stops and the question comes back to Dennis. | The core and three matter most. This doesn't touch ddgi's three-only adapter rule. |
| D6 | Intel Sponza on the public examples site? | **Yes, use it.** No more delay on asset licensing; revisit only if someone raises it. | The upscaler project already pushed a Sponza branch. |
| D7 | Docs tooling | **pmndrs/docs** (MDX) plus TypeDoc. | |
| D8 | Licences for the unlicensed repos | **Nothing ported from them; keep it that way.** | |

## Target repo layout

```
pmndrs/bvh
├─ src/                      @pmndrs/bvh (core: no engine imports, lint-enforced)
│  ├─ build/                 tinybvh WASM, GPU H-PLOC, GPU LBVH, sort/scan, batch planner, chooser
│  ├─ refit/                 GPU top-level refit, GPU BLAS refit, CPU refit twin
│  ├─ layout/                GPU_LAYOUT_VERSION, word positions, WGSL structs, checker, serialize
│  ├─ trace/                 WGSL walk (closest/any), CPU walk, oracle
│  ├─ query/                 CPU raycast (Q1), closest point / shapecast (Q2)
│  ├─ worker/                @pmndrs/bvh/worker: worker pool, transferable build requests
│  ├─ three/                 @pmndrs/bvh/three: BufferGeometry in, raycast accel, helpers, TSL walk node
│  └─ babylon/               @pmndrs/bvh/babylon: Mesh in, picking, device access (after the spike)
├─ native/                   tinybvh-wasm.cpp, build-wasm.ts (emsdk pinned)
├─ licenses/                 tinybvh, HIPRT, emscripten runtime
├─ examples/                 Vite multi-page app, deployed to GitHub Pages
│  ├─ vanilla/   three/   compare/   babylon/   node/
│  └─ shared/                stats overlay, frame-time graph, scene loader, builder picker
├─ bench/                    trace-bench, tree-quality, build-profile, results CSVs
├─ tests/gpu/                the GPU page runner and the bvh-* pages (from tools/gpu)
├─ docs/                     MDX for pmndrs/docs
└─ .github/workflows/        CI (below)
```

## Phases

The phases run in the BVH lane, next to ddgi's M3 and M4, and don't block them. P1 happens inside webgpu-ddgi, so the new boundary is proven before anything leaves.

### P0: Decisions (this doc)

D1–D8 are answered. Then the split outline in webgpu-ddgi points to this plan.

### P1: Prove the boundary inside webgpu-ddgi

1. **API trim.**
   - Rename the shared types and move them to a neutral `mesh-input.ts`: `TinyBvhInput` → `MeshInput`, `GpuLbvhBufferGeometry` → `GpuMeshBuffers`, `GpuLbvhGeometry` → `GpuMeshGeometry`, `GpuLbvhOptions` → `GpuBuildOptions`, `GpuLbvhMeshTree` → `GpuMeshTreeRange`, `GpuLbvhTrees` → `GpuTreeBuffers`. (`GpuMeshTrees` was the first idea, but ddgi-core already has a class by that name.)
   - Replace the `./*` export with named subpaths.
   - Export `GPU_LAYOUT_VERSION`.
   - Let the refits take their shape from a GPU-built tree without a CPU read-back where possible. Otherwise, document the one read-back.
2. **Move into `packages/bvh`:**
   - the WGSL structs and the walk (if D1 is yes);
   - `cpu-traversal.ts`;
   - the oracle and its fixtures;
   - `mesh-bvh-topology.ts`;
   - a generic top-level shape builder over boxes;
   - the tinybvh worker wrapper, as `/worker`.

   ddgi-core then imports all of these from the package.
3. **Serialization:** `serializeLayout` and `deserializeLayout`, with a header that carries the layout version.
4. **Done when:**
   - ddgi's unit tests and `trace-oracle` pass unchanged on the RTX 3060 and the Intel UHD;
   - the GPU pages pass;
   - the lint rule still holds (no engine or ddgi imports in `packages/bvh`).

### P2: Extract

1. Run `git filter-repo` on a fresh clone. Keep `packages/bvh`, the `bvh-*` design docs, the `tools/gpu/pages/bvh-*` and `trace-*` pages, `bench/results/trace.csv` (the BVH rows) and the THIRD_PARTY entries. Rename them to the layout above.
2. Add a library build (tsup or `tsc` to ESM plus `.d.ts`). The `.wasm` stays inlined, with its `/*! */` licence comment. Port ddgi's check that fails the build if a built file loses its licence text.
3. Set up the tooling: ESLint (with the no-engine-imports rule for core), Vitest and the GPU page runner.
4. **CI (GitHub Actions)**
   - On every PR: lint, typecheck, unit tests, and the CPU twins against their GPU kernels' fixtures.
   - `bvh:wasm --check` on Ubuntu and macOS. Only Windows builds have been compared so far.
   - Headless Chrome with SwiftShader runs the GPU pages for correctness on Linux. SwiftShader probably has no `subgroups`, so this exercises the fallback path. That is a feature: it keeps the fallback honest.
   - **The real GPUs run before each release, by checklist** (RTX 3060, Intel UHD, M5 Pro). Later, a self-hosted runner on the RTX machine could take over.
5. Publish `0.1.0-alpha` under the `next` tag.

### P3: ddgi switches to the package

- `@pmndrs/ddgi` depends on `@pmndrs/bvh ~0.1.x` as a normal dependency, not a peer.
- Local development links a checkout with pnpm overrides.
- At startup, ddgi checks `GPU_LAYOUT_VERSION` against its own WGSL.
- `packages/bvh` is removed from webgpu-ddgi. Its README there points to the new repo.
- **Done when** ddgi's `test:gpu` set and the oracle pass on the published package.

### P4: Examples and docs (the bulk of the work; see below)

Order:

1. the vanilla examples;
2. the three examples;
3. the comparison suite;
4. Node;
5. Babylon (P6).

Every example uses the shared overlay, which shows:

- the builder picked and why (`builtBy`, `fallbackReason`);
- build ms (CPU wall time, GPU timestamp queries);
- SAH cost, depth, node and triangle counts, bytes;
- trace Mrays/s where rays are traced;
- a main-thread frame-time graph.

### P5: The comparison against three-mesh-bvh

The rules and the list are below. The numbers go in the docs' benchmark page, with how each was measured.

### P6: Babylon (basic, Sonnet subagent)

A Sonnet subagent does all of this. The bar is "it works", not polish.

1. **Spike:**
   - get the `GPUDevice` from `WebGPUEngine`;
   - bind our buffers in a Babylon `ComputeShader`;
   - pick through our CPU raycast.
2. **If it works:** the two examples and a thin `@pmndrs/bvh/babylon`.
3. **If it gets stuck:** it stops and writes down what blocked it. Dennis decides then. The CPU picking example ships either way, since picking needs no device.

### P7: Release 0.1

- Docs site and examples live.
- The pre-release checklist run on all three GPUs.
- The M5 Pro numbers measured. That device is a target that has never been measured, and it may not report `subgroupMinSize`.
- The licence audit (D8) done.
- Announce.

## Follow-up phases (after 0.1)

### P8: Feature parity with three-mesh-bvh

**1. Parity audit.** Go through three-mesh-bvh's public API (its `index.d.ts`, the `/webgpu` and `/worker` entry points) and its example list. Each feature goes into a table as one of:

- **have**;
- **partial**;
- **missing**;
- **won't**, with the reason.

**Won't** covers things like GLSL and WebGL GPU traversal (WebGL2 isn't a target). Areas to cover:

| Area | three-mesh-bvh has | Ours today |
| --- | --- | --- |
| CPU queries | `raycast`, `raycastFirst`, `shapecast`, `closestPointToPoint`, `closestPointToGeometry`, `intersectsGeometry`, `intersectsBox`, `intersectsSphere`, `bvhcast` (tree against tree) | Q1 raycast (P4); the rest is missing |
| Shapes for queries | `ExtendedTriangle`, `OrientedBox` | Missing |
| Primitives | Triangles, points, lines and line segments | Triangles only |
| Build options | CENTER, AVERAGE and SAH strategies, `maxLeafTris` (default 10), `maxDepth`, `indirect` (leaves the index buffer alone), geometry groups and draw ranges | Builders instead of strategies; leaves of 1–8 triangles; no indirect mode; no groups |
| Scene level | Object BVH (scene raycast, frustum culling), `BatchedMesh`, `InstancedMesh` | GPU top level with refit; no CPU scene queries, no frustum culling |
| Deforming meshes | CPU `refit()`, `StaticGeometryGenerator` for skinned and morphed meshes | GPU BLAS refit, CPU refit twin |
| three integration | `computeBoundsTree`, `disposeBoundsTree`, `acceleratedRaycast`, `firstHitOnly`, `MeshBVHHelper`, `getTriangleHitPointInfo` | `/three` (P4) |
| Utilities | `serialize`/`deserialize`, `estimateMemoryInBytes`, `getBVHExtremes`, `validateBounds` | Serialization (P1), `checkGpuLayout`, `treeSahCost` |
| GPU | WGSL and TSL ray functions, shapecast and distance functions on the GPU, `ClusteredBVH`, SDF | WGSL walk (closest, any); no GPU closest point or shapecast |
| Workers | `GenerateMeshBVHWorker`, `ParallelMeshBVHWorker` | `/worker` pool (P4) |

**2. Build-out.** Work through the missing rows in this order:

1. Q2 (closest point and shapecast) and the query shapes;
2. the drop-in three integration (`computeBoundsTree`-style);
3. the object BVH queries and frustum culling over our top level;
4. GPU closest point and shapecast in WGSL;
5. indirect mode and groups;
6. points and lines.

Each feature lands with its compare page. The remaining three-mesh-bvh examples get ported as their features arrive: `shapecast`, `distancecast`, `sculpt`, `selection`, `physics`, `characterMovement`, `clippedEdges`, `collectTriangles`, `edgeIntersect`, `lineIntersection`, `pointCloudIntersection`, `triangleIntersect`, `voxelize` and `batchedMesh`.

**Done when** every row is have or won't, and every won't has its reason in the docs' "Coming from three-mesh-bvh" page.

### P9: Memory audit and fixes

The only number so far is from T0.10 part C (PROGRESS): Sponza, ours 302 MB against three-mesh-bvh's 153 MB. It was taken with our old CPU builder at leaf size 4, and nobody wrote down what it counted. **The breakdown can start any time after P2.** It should land before 0.1, so the comparison page can explain the gap and not just show it. The fixes come after.

**First numbers, from ddgi's `ddgi-tree-swap` page** (2026-11-09, RTX 3060, Intel Sponza with its 449 meshes, BX.1 run):

- tinybvh trees only: **470 MB** (nodes 194 MB, triangle records 276 MB, top level 0.1 MB);
- while the first GPU trees are in use: **1.16 GB**, because 666 MB of H-PLOC build outputs stay alive until the swap to tinybvh's trees;
- after the swap: 493 MB (nodes buffer 217 MB, 194 MB of it in use).

So the triangle records are the biggest single item, and the build outputs held until the swap more than double the peak.

**1. Breakdown (done 2026-11-11 as BX.19: [memory.md](memory.md), `tools/memory-breakdown.ts`).** Measure every buffer on Sponza and Bistro exterior (GPU and CPU, resident and transient):

- nodes;
- triangle records;
- parents;
- the top level;
- H-PLOC and LBVH build scratch;
- worker copies;
- the CPU copy that `setTree` needs;
- the time a mesh holds both its GPU tree and tinybvh's during the swap.

Measure three-mesh-bvh the same way. The accounting has to be fair. three-mesh-bvh reuses the geometry's own positions and reorders its index buffer, so most of its cost is nodes. Our 48-byte triangle records copy the vertices: about 276 MB for Sponza with curtains (DECISIONS). That is probably most of the gap. The fair number is **"bytes beyond the geometry the app already holds"**, reported next to the totals.

**2. Levers.** Each lever is a switch beside the current layout, measured on bytes and on closest-hit and any-hit Mrays/s (Sponza and Bistro interior, RTX 3060 loaded, the oracle passing). Each becomes a default only with a written reason. Any lever that changes the layout bumps `GPU_LAYOUT_VERSION`.

- **Indexed leaves.** Triangles read the app's vertex and index buffers instead of a copy. This saves the most, and costs one indirection per triangle test.
- **Smaller triangle records.** For example, one vertex plus two edges, or positions quantized to the leaf's box. The oracle has to prove the walk is still watertight.
- **Compressed nodes.** Child boxes quantized against the parent (64 → 32 bytes or less). B3's CWBVH nodes were about 4× smaller. B3 was dropped for speed, not memory, so memory reopens it as an option (the code is on `bvh-b3`).
- **Larger leaves.** Allowing up to 10 or 16 triangles means fewer nodes. three-mesh-bvh defaults to 10.
- **Parents only when needed.** Keep the parents buffer only when a refit uses it.
- **Scratch and copies.** Pool build scratch across batches and free it after the build. Free a GPU tree as soon as its tinybvh tree replaces it. Drop CPU copies once they've been used.

**3. Done when:**
- the breakdown is written up;
- every lever is measured;
- Dennis has picked which levers become defaults and which stay options;
- the comparison page shows both totals and the bytes beyond the geometry.

## Examples

Each example is one page with its source next to it, a "view source" link, and the shared overlay. Unless one says otherwise, each takes a glTF file dropped onto the page.

### Vanilla WebGPU (no engine)

These come first, because they prove the "not a three library" claim.

| Example | Shows |
| --- | --- |
| `vanilla/hello` | The smallest use: request a device (with `subgroups` if the adapter has it), `createMeshLayoutBuilder`, build, and draw the tree's boxes with a tiny WebGPU line pass. It prints which builder ran and why. |
| `vanilla/raytrace` | A compute-shader ray caster using the package's WGSL walk: primary rays, AO and hard shadows. You can switch the builder (tinybvh, LBVH, H-PLOC) live and see the change in build ms, SAH and Mrays/s. |
| `vanilla/builders` | A side-by-side table for the same mesh: tinybvh, H-PLOC and LBVH, each with build ms, SAH cost against tinybvh, trace speed and memory. This is the builder guide in a page. |
| `vanilla/workers-cpu` | tinybvh on the main thread vs one worker vs a worker pool that builds different meshes in parallel. Each pool worker is single-threaded, so this needs no cross-origin isolation. A spinning cube and the frame-time graph show the jank (or its absence). |
| `vanilla/workers-gpu` | Three ways to run a GPU build. (1) On the render device, where the trees stay on the GPU with no copy. (2) In a worker with its own `GPUDevice`, then read back and transferred to the main thread. (3) Everything in a worker with an `OffscreenCanvas`. The page is honest about the trade-offs: buffers can't be shared between devices, and the GPU is still shared. |
| `vanilla/fast-first` | ddgi's load pattern as a generic recipe. GPU H-PLOC trees are ready in milliseconds, and tinybvh's trees from the worker pool replace them mesh by mesh as they arrive. A timeline shows each mesh's swap. This is the one that shows CPU, GPU and workers together. |
| `vanilla/refit` | (a) A deforming mesh: a compute pass moves the vertices, and its buffer feeds the BLAS refit. `relativeCost` drifts, and a rebuild triggers past a limit. (b) 10,000 moving instances with a top-level refit every frame, with its GPU time shown. |
| `node/bake` | A Node CLI: build with tinybvh (the WASM needs no GPU), `serializeLayout` to a file, then load it in the browser with no build at all. It's the answer to "precompute my BVH". |

### three.js (`WebGPURenderer`)

| Example | Shows |
| --- | --- |
| `three/raycast` | `@pmndrs/bvh/three` speeds up `Mesh.raycast` (Q1), compared with three's default raycast. A GPU batch raycast does thousands of rays in one dispatch. |
| `three/pathtracer` | A small TSL/WGSL compute path tracer that reads our trees through a TSL node wrapping the walk. |
| `three/skinned` | A TSL compute pass skins the vertices into a storage buffer (three skins in the vertex shader, so the example adds this pass), and that buffer feeds `GpuMeshRefit`. Nothing returns to the CPU. Picking still hits the moving mesh. |
| `three/instances` | `InstancedMesh` and `BatchedMesh` with thousands of moving instances: a top-level refit every frame, picking and per-instance highlight. |
| `three/load-gltf` | The `fast-first` recipe on a real glTF in three, with the per-mesh timeline. |
| `three/sdf` | Generating an SDF on the GPU from our trees. |
| `three/inspector` | A `BVHHelper` that draws the boxes by depth, with stats per builder and leaf size, and a heat map of nodes visited per ray. |

### Comparison suite against three-mesh-bvh (`compare/`)

**What it shows:** the same three.js app running on three-mesh-bvh's pipeline and on ours. The aim is to be faster, and more flexible and useful for serious applications.

The flexibility case matters as much as the speed. three-mesh-bvh's GPU side lives inside three's shader space: its traversal is TSL nodes and WGSL/GLSL snippets meant to be compiled into three materials and three compute nodes. Ours is plain WebGPU buffers and a WGSL walk, so the same trees can be used:

- from a raw compute pass that sits beside three's renderer;
- from another engine;
- from a worker;
- by another library on the same device (ddgi already does this).

Our trees are also built and refit on the GPU. Each compare page has a "what else you can do" note that shows one thing only our pipeline allows. Example: on `compare/skinned`, the refit tree is also read by a plain compute pass that three never sees.

**Rules**
- The same three version and the same geometry in both.
- The same rays, from a fixed seed.
- Warm runs.
- Medians of several runs, on the RTX 3060 with the GPU loaded (the hardware rules in `ddgi-hardware-target`).
- Each page shows both libraries live, with a toggle or split screen, and records a CSV row.
- three-mesh-bvh is a pinned dev dependency of the examples only.

| three-mesh-bvh example | Our page | What's compared |
| --- | --- | --- |
| `raycast` | `compare/raycast` | CPU raycasts per second (Q1) and GPU batch raycasts per second against theirs |
| `asyncGenerate` (`GenerateMeshBVHWorker`, `ParallelMeshBVHWorker`) | `compare/async-build` | Build time and main-thread stalls: their worker and parallel worker, our tinybvh worker pool, and GPU H-PLOC |
| `webgpu_gpuPathTracingSimple` (and the WebGL `gpuPathTracingSimple`) | `compare/pathtracer` | The same path tracer on their WGSL/TSL BVH functions and on our walk: Mrays/s, closest-hit and any-hit |
| `skinnedMesh` | `compare/skinned` | Their CPU `refit()` against our GPU BLAS refit, with the CPU and GPU ms per frame |
| `inspector` | `compare/tree-quality` | SAH cost, depth, node count and bytes. Theirs uses CENTER, AVERAGE or SAH; ours uses tinybvh, H-PLOC or LBVH, at the same leaf sizes. |
| `objectbvh_sceneRaycast`, `objectbvh_frustumCulling` | `compare/scene-bvh` | A scene-level tree over many objects: their object BVH against our GPU top level with refit |
| `webgpu_sdfGeneration` | `compare/sdf` | SDF build time on WebGPU |
| `shapecast`, `distancecast`, `sculpt`, `selection`, `physics`, `characterMovement`, `clippedEdges` | **Not covered until Q2** | A docs table lists these as gaps. `characterMovement` and `distancecast` are the first to add once closest point and shapecast land. |

The docs page puts the numbers next to a **feature table that says what three-mesh-bvh has and we don't**:

- WebGL2 GPU traversal;
- points and lines as primitives;
- indirect mode;
- mature shapecasts;
- years of use.

It also lists what we have and they don't:

- GPU builds;
- GPU refits;
- the oracle;
- engine independence.

The tracer's earlier numbers (closest-hit 1.20× and any-hit 0.93–1.22× three-mesh-bvh on Sponza, memory 302 MB against 153 MB) were measured inside ddgi. They are re-measured on these pages. The memory gap will be named, not hidden.

### Babylon (P6, after the spike)

| Example | Shows |
| --- | --- |
| `babylon/picking` | `scene.pick` over a big mesh against our CPU raycast, with picks per second. It needs no WebGPU, so it runs with Babylon's WebGL engine too. |
| `babylon/gpu-trace` | AO or shadows from our WGSL walk in a Babylon `ComputeShader`, on the engine's own device |

## Docs

**README** (the repo front page)
- The one-paragraph pitch.
- Install.
- Three quick starts of about ten lines each: vanilla, three and Babylon.
- The "which builder" table (from today's package README).
- Requirements: WebGPU, with `subgroups` for the GPU paths and the CPU fallback otherwise.
- Headline numbers, with a link to how they were measured.
- Licences.

**Docs site** (pmndrs/docs MDX in `docs/`)

1. **Getting started:** install, the device with `subgroups`, and the first build.
2. **Concepts:** the GPU layout (the 64-byte two-child node, 48-byte triangles, parents), the builder contract, the oracle rule, and `GPU_LAYOUT_VERSION`.
3. **Choosing a builder:** tinybvh, H-PLOC, LBVH and the chooser, what each needs, and the fallback chain.
4. **Workers:** CPU pools, GPU builds on a worker's own device, `OffscreenCanvas`, and what to transfer and what not.
5. **Sharing a device with an engine:**
   - three: `renderer.backend.device`;
   - Babylon: `engine._device`, and why that is fragile;
   - vanilla.
6. **Refitting:** the top level against BLAS, quality drift, and when to rebuild.
7. **Tracing in WGSL:** how to bind the walk, the closest and any entry points, the stack size and the watertight test.
8. **CPU queries:** raycast, then closest point and shapecast once Q2 lands.
9. **Serialization and baking in Node.**
10. **Coming from three-mesh-bvh:** API mapping, what's missing, and when to stay with three-mesh-bvh.
11. **Benchmarks:** the method (loaded GPU, medians, devices) and the tables.
12. **Adding a builder from a paper:** the contract, the oracle, the bench rows, and the licence rule (MIT or Apache-2.0, or built from the paper alone). Candidates already researched: DOBB-BVH and tetrahedral cages (`research/` in the ddgi research repo).
13. **API reference** (TypeDoc).

**Moved over:** `bvh-builders.md` (internals) becomes `docs/internals/` for contributors. Each stage's PROGRESS entries go into a `HISTORY.md`.

## Risks

- **Reaching into Babylon's internals.** `engine._device` can change without notice. The spike decides; the fallback is CPU-only picking.
- **No `subgroups` on Safari or Firefox.** The GPU builders then fall back to tinybvh. The examples show `fallbackReason` up front, so a slower result isn't mistaken for the library being slow.
- **Q1/Q2 scope creep.** Shapecast generality is three-mesh-bvh's strength. Q2 ships closest point and a generic shapecast, and nothing more until users ask.
- **ddgi churn during M3/M4.** P1 changes ddgi-core's imports. Land it at a quiet point, as B1 was (DECISIONS), with ddgi's full `test:gpu` set as the gate.
- **The 302 MB against 153 MB memory gap** against three-mesh-bvh. It will be visible in the comparison. P9 audits it and fixes what it can.

## Suggested task breakdown

| Task | Phase | Depends on | Model tier |
| --- | --- | --- | --- |
| BX.0 Decisions D1–D8 (**done 2026-11-09**) | P0 | none | Dennis |
| BX.1 API trim + `GPU_LAYOUT_VERSION` + serialization | P1 | BX.0 | Opus |
| BX.2 Move the walk, CPU walk, oracle, topology and worker wrapper into `packages/bvh` | P1 | BX.1 | Opus |
| BX.3 Filter-repo extraction, library build, lint/test/wasm-check CI | P2 | BX.2 | Opus |
| BX.4 ddgi switches to the published alpha | P3 | BX.3 | Opus |
| BX.5 Examples shell (Vite multi-page, shared overlay, frame-time graph, scene loader) | P4 | BX.3 | Sonnet |
| BX.6 Vanilla examples (hello, raytrace, builders, refit) | P4 | BX.5 | Opus |
| BX.7 Worker examples and `/worker` pool (workers-cpu, workers-gpu, fast-first) | P4 | BX.5 | Opus |
| BX.8 Q1 CPU raycast + `/three` raycast acceleration | P4 | BX.2 | Opus |
| BX.9 three examples | P4 | BX.6, BX.8 | Opus |
| BX.10 Comparison suite and benchmark page | P5 | BX.9 | Opus (review on Opus) |
| BX.11 Node bake example | P4 | BX.1 | Sonnet |
| BX.12 Docs site (pmndrs/docs) + TypeDoc + README | P4 | BX.6 | Opus |
| BX.13 Babylon spike, then basic examples (stop and report if stuck) | P6 | BX.8 | Sonnet |
| BX.15 Release checklist on three GPUs, licence audit, 0.1 | P7 | BX.10, BX.12, BX.19 | Opus |
| BX.16 Parity audit table (three-mesh-bvh API and examples against ours) | P8 | BX.8 | Opus |
| BX.14 Q2 closest point + shapecast + query shapes, with `distancecast`/`shapecast`/`characterMovement` compares | P8 | BX.16 | Opus |
| BX.17 Parity build-out, one task per row group (three drop-in, object BVH queries, GPU closest point/shapecast, indirect and groups, points and lines) | P8 | BX.14 | Opus |
| BX.18 Remaining compare ports as each feature lands | P8 | BX.17 | Opus; small ones Sonnet |
| BX.19 Memory breakdown, ours and three-mesh-bvh, fair accounting | P9 | BX.3 | Opus |
| BX.20 Memory levers, measured one by one, defaults decided by Dennis | P9 | BX.19 | Opus |

Once BX.5 exists, BX.6, BX.7, BX.8 and BX.11 can run in parallel. BX.19 is measurement only and can run alongside them.

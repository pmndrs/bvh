# BVH builders: how they work

How each builder in this package works inside. To learn what the package is and how to use it, read [README.md](../../README.md) first. The plan and the stages are in [bvh-framework.md](bvh-framework.md), and the measured numbers are in PROGRESS (B0, B0b, B1a, B1, B2, B3).

Paths below are relative to the repository root.

The sections run in build order: tinybvh (B0) first, then the GPU refit (B1a), then the GPU LBVH (B1), then H-PLOC (B2), which reuses most of the LBVH.

## tinybvh in WebAssembly (stages B0 and B0b)

`src/tinybvh.ts` runs [tinybvh](https://github.com/jbikker/tinybvh) 1.9.1 (MIT, Jacco Bikker), compiled to WebAssembly. The `.wasm` is committed, base64-encoded, in `src/tinybvh-wasm.ts`, so it loads the same way in a worker, on a page, in Node and in any bundler. It is single-threaded; the tracer runs it in a worker (`src/trace/mesh-bvh-async.ts`).

It uses tinybvh's default builder, which with SIMD is Bonsai (Ganestam et al., 2015). From one tree it gives either output:

- `build(geometry)` returns the tree's shape (`MeshTreeTopology`, `src/topology.ts`).
- `buildLayout(geometry)` (stage B0b) returns the tracer's GPU layout, written inside the WASM by `native/tinybvh-wasm.cpp`: 64-byte nodes, 48-byte triangles in tree order, parents, and the mesh's box.
  - The bytes are exactly what `src/trace/`'s `buildMeshBvhFromTopology` makes from `build`'s shape. A test checks it (`src/trace/mesh-bvh-tinybvh-layout.test.ts`).
  - As there, the boxes are recomputed from the triangles, not taken from tinybvh.
  - It is several times faster than doing the conversion in JavaScript. The JavaScript walks every node and triangle through typed arrays and is bound by memory traffic; the WASM packs each leaf's triangles and computes its box in one pass, with SIMD compares instead of branches.
- `buildLayout(geometry, { builder: 'hploc' })` runs tinybvh's scalar `BuildHPLOC` instead (leaves of one triangle). It is slow, and is there only as the oracle for the GPU H-PLOC (B2).

`tools/layout-identity.ts` rebuilds every mesh of Sponza and both Bistro scenes both ways and reports how many differ (it should be none):

```sh
node tools/layout-identity.ts            # all three scenes
node tools/layout-identity.ts sponza     # one scene
```

The layout itself is defined in `src/trace/layout.ts`; `src/gpu-layout.ts` holds its sizes and word positions, which layout.ts imports. The C++ repeats it as structs with `static_assert`s on every offset; change both together.

The build needs WebAssembly SIMD (SIMD128), which every current Chrome, Edge, Firefox and Safari has (Chrome 91, Firefox 89 and Safari 16.4 onwards), so there is no scalar fallback build.

### Rebuilding the .wasm

```sh
pnpm bvh:wasm           # install the pinned Emscripten if needed, then build
pnpm bvh:wasm --check   # build, and fail unless it matches the committed .wasm
```

`native/build-wasm.ts` does everything:

1. **Toolchain.** It uses emsdk **6.0.13** (emsdk commit `35659964bc2762fffec0d257997fb61475f765b7`) in `tools/emsdk/`, which is gitignored.
   - If that folder is missing it clones emsdk there and runs `emsdk install 6.0.13` and `emsdk activate 6.0.13` inside it, which downloads about 700 MB.
   - It never uses or touches any other Emscripten on the machine: it drops every `EM*` variable from the environment and sets its own config and cache, both inside `tools/emsdk/`.
2. **Sources.** It downloads tinybvh's headers at commit `c9be3103cc9693b456b0a0353927c11cac7d710f` and checks each one's SHA-256.
3. **Build.** It compiles `native/tinybvh-wasm.cpp` (our small C interface) with the flags listed, with a reason for each, in `BUILD_FLAGS`: `-O3`, no threads, and `-msimd128 -mavx`.
   - Emscripten turns tinybvh's AVX code into WASM SIMD.
   - AVX turns on tinybvh's fast builders. Without it tinybvh falls back to its scalar reference builder, which is about 3 times slower on Sponza.
   - The build runs in a temporary folder with relative paths only, so no folder name ends up in the output.

Needs: Node 24, git and network access. On Linux and macOS it also needs Python 3, because emsdk brings its own Python only on Windows.

**Reproducibility.** On Windows (2026-11-04), repeated builds from different temporary folders gave the same bytes as the committed `.wasm` (`--check` passes). A Linux or macOS build has not been compared yet.

### How the licences travel with the .wasm

`licenses/tinybvh.txt` is tinybvh's licence; the build checks it against tinybvh's own `LICENSE` file. `licenses/emscripten-runtime.txt` covers the parts of Emscripten's runtime that the `.wasm` contains: Emscripten's own code and musl (MIT), dlmalloc (public domain), and LLVM's libc++ and compiler-rt (Apache 2.0 with LLVM Exceptions).

The generated `src/tinybvh-wasm.ts` starts with both texts in a `/*! */` comment, which bundlers keep. So every built file that contains the `.wasm` also contains its licences. ddgi's `packages/ddgi/scripts/verify-artifacts.mjs` (in webgpu-ddgi) fails its package build if one doesn't; this repo's own check is still to be ported.

## GPU refit (stage B1a)

`src/gpu-refit.ts` (`GpuBvhRefit`) refits a tree in the GPU layout on the GPU, keeping its shape.

- One thread per leaf climbs to the root. At each node the second thread to arrive (a per-node counter, no float atomics) takes the union of the two child boxes.
- Boxes travel as 32-bit atomic words, so no plain store has to be visible across workgroups. A bounded re-read covers WGSL's relaxed atomic order.
- If a read still gives up, a one-workgroup repair pass refits the whole tree level by level in the same compute pass, so a wrong box is never kept. The stats (read back only with `diagnostics: true`) count it.
- Sibling leaves held by neighbouring subgroup lanes finish their node in registers. The check compares node and side, never lane numbers, so it works on any subgroup size.
- The climb also sums every inner node's half-area: the tree-quality signal (`relativeSahCost`) a caller uses to decide when to rebuild.
- It needs the `subgroups` feature (`GpuBvhRefit.isSupported`).

`src/gpu-instance-box.ts` is the WGSL for an instance's padded world box (Arvo's method). `src/refit-cpu.ts` holds the CPU twins: leaf and parent slots, the cost signal, and a plain refit the tests compare the GPU with (box unions are exact, so a correct GPU refit matches it bit for bit).

The tracer's top level (ddgi-core's `trace/top-level-gpu.ts`, still in webgpu-ddgi) uses all three: a gather kernel writes each instance's world box, the refit runs every frame something moved, and a worker builds a new shape on the CPU now and then (instances added or removed, or the refit tree's cost past a limit).

## GPU LBVH (stage B1)

`src/gpu-lbvh.ts` (`GpuLbvhBuilder`) builds mesh trees on the GPU, in exactly the layout tinybvh's `buildLayout` writes (`src/gpu-layout.ts`), so the tracer reads them unchanged. The kernels are in `src/gpu-lbvh-wgsl.ts`, ported from HIPRT's LBVH (MIT, `licenses/hiprt.txt`):

1. Triangle boxes and each mesh's centroid box (subgroup, then workgroup, then one atomic per workgroup).
2. 30-bit Morton codes in cubic cells over each mesh's own centroid box.
   - Stretching every axis to 1024 cells, as HIPRT's plain code does, made flat and long meshes cost 2.4–2.6 times tinybvh's SAH. Cubic cells cost 1.02–1.22 times.
3. A stable, segmented radix sort (`src/gpu-radix-sort.ts`: 5-bit digits, reduce-then-scan, no look-back), on the scan in `src/gpu-scan.ts`.
   - No decoupled look-back, because WebGPU doesn't promise forward progress between workgroups.
4. Apetrei's bottom-up climb with HIPRT's index tie-break. The SAH leaf collapse is decided on the way up. The climb runs in two rounds; the second is an indirect dispatch.
5. A scan numbers the nodes that stay; triangles, nodes and parents are written in the tracer's layout.

**Batches.** It builds every mesh of a batch with the same dispatches (a mesh table and a tile table say which workgroup works on which mesh), so a scene of many small meshes doesn't pay per mesh.

- The trees come back in three shared GPU buffers with an offset per mesh (`GpuTreeBuffers`).
- `readLayouts` copies them into CPU memory.
- `planGpuLbvhBatches` splits a scene into batches that fit the device's buffer limits.

**Handing words between threads.** Every word one climbing thread hands to another goes through atomics with a bounded wait, as in the B1a refit. A tree whose wait ran out is refused (`problem`), never used.

**Subgroup size.** Workgroup scans number threads in subgroup order (`src/gpu-workgroup-scan.ts`), so nothing depends on the subgroup size (Intel's UHD runs 8 or 16 wide) or the lane layout. The builder needs `subgroups`.

**Fallback.** `src/mesh-builder.ts` (`createMeshLayoutBuilder`) picks a GPU builder when the device has subgroups and tinybvh (B0) otherwise. It also sends single meshes to tinybvh when they are too big for the device or the GPU refused them. The result says which builder made each tree.

**Deforming meshes.** `src/gpu-mesh-refit.ts` (`GpuMeshRefit`) refits any builder's mesh tree from a vertex buffer the adapter supplies.

- One kernel rewrites the tree-ordered triangles from the moved vertices and writes their boxes; then the B1a refit climbs.
- The shape comes from a CPU copy of the layout once (`setTree`), not every frame.

**Checking a tree.** `src/layout-check.ts` checks any layout against its mesh: triangles bit for bit, one tree from node 0, every box exactly the box below it, parents, depth and counts. `treeSahCost` gives its SAH cost. `src/lbvh-cpu.ts` is the same LBVH written plainly on the CPU, for the unit tests and as a cross-check (not a builder to ship).

GPU pages: `bvh-lbvh` (scan, sort, trees), `bvh-blas-refit`, `bvh-lbvh-fallback`, `bvh-lbvh-profile` (manual, step-by-step timing), the `watertight-lbvh` variant of `trace-oracle`, and `trace-bench-lbvh-sponza` / `-quick` (build time and trace speed against tinybvh's trees).

## GPU H-PLOC (stage B2)

`src/gpu-hploc.ts` (`GpuHplocBuilder`) is the target GPU builder: H-PLOC (Benthin et al., HPG 2024), behind the same contract as the LBVH and with the same output (`GpuTreeBuffers`, the tracer's layout).

- It reuses the LBVH's batches, triangle boxes, Morton codes, sort, scans and readback, and climbs the same radix tree.
- On the way up it merges clusters with PLOC, so its trees are close to a full SAH build's.
- The kernels are in `src/gpu-hploc-wgsl.ts`, ported from tinybvh's `BuildHPLOC` (MIT, `licenses/tinybvh.txt`).

How the climb works:

1. A node of at most 16 triangles holds them as they are. A bigger node (and the root) holds its children's clusters, at most 32, and merges mutual nearest neighbours (search radius 8, by merged surface area) until 16 are left (the root: one).
2. The whole subgroup merges one node at a time, with each cluster in a lane and slot: `ceil(32 / smallest subgroup size)` slots per lane, so one on 32-wide GPUs and four on the Intel UHD. It shuffles by cluster number, so the tree is the same on every subgroup size; the RTX and the Intel build identical trees.
3. The SAH leaf collapse (B1's rule) is decided as each node is made. A scan numbers the nodes that stay, and a second gives each kept node's leaves their run of triangle positions (each leaf's triangles together).

As in the LBVH, everything one climbing thread hands to another is an atomic, written once and read with a bounded wait. A node's stored clusters reuse positions on the way up, so each entry carries its writer's radix-tree level.

The smallest subgroup size comes from `device.adapterInfo.subgroupMinSize`. If the browser doesn't report it, the builder assumes 4 (eight slots per lane): correct, but slower.

**Options** (`HplocOptions`): the LBVH's, plus:

- `mortonCodes`: `'cubic-30'` (B1's codes, the default) or `'tinybvh-63'` (tinybvh's 63-bit codes, two sort words).
- `tieBreak`: `'pairs'` (the default) or `'tinybvh'`. On exactly equal merged areas tinybvh's rule chains the tree: 3000 identical triangles give depth 128, twice the tracer's stack. `'pairs'` pairs ties up and keeps the tree balanced.

**The oracle.** `src/hploc-cpu.ts` is H-PLOC on the CPU, step for step, with the same options.

- With tinybvh's codes and tie rule it builds exactly the tree of tinybvh's `BuildHPLOC` (`buildLayout(geometry, { builder: 'hploc' })`). The unit tests check that.
- The GPU's trees match the twin's (same SAH cost and node count).
- With tinybvh's options the GPU's trees match tinybvh's: SAH cost within 1% on the `bvh-hploc` page, and on whole scenes in the trace bench.

`tools/tree-quality.ts` compares the CPU builders and twins on real scenes without a GPU (`node tools/tree-quality.ts sponza`).

GPU pages: `bvh-hploc`, `bvh-hploc-profile` (manual), the `watertight-hploc` variant of `trace-oracle`, and `trace-bench-hploc-sponza` / `-quick`.

`createMeshLayoutBuilder` picks H-PLOC by default when the device has subgroups (`prefer: 'gpu-lbvh'` gives the LBVH), with the same fallback to tinybvh.

## Build peak memory (P9)

A GPU build batch holds its inputs, its scratch and its outputs at once, so on a scene load the batch is the biggest memory number: Sponza (132 meshes, 5,744,384 triangles) peaks at **1,482.7 MiB** in `SceneTrees`, against 470 MiB settled. `hplocBatchBytes` and `lbvhBatchBytes` (`src/gpu-hploc.ts`, `src/gpu-lbvh.ts`) give every buffer's bytes for any batch. Each batch's constructor allocates from the same functions, and checks itself against `buildTables`, so the prediction can't drift from the code. `docs/design/memory.md` has the same itemization, from its own tool.

### Bytes per triangle

T is triangles and I is radix-tree inner nodes, `max(1, T - 1)` per mesh (so I is about T). Words are 4 bytes.

| Buffer | H-PLOC | LBVH | What it is |
| --- | ---: | ---: | --- |
| positions | 12 per vertex (Sponza 6.5/tri) | same | input, copied into the batch |
| indices | 12 (`uint32`), 6 (`uint16`) | same | input |
| flags | 4 if the mesh has them | same | input |
| primitive boxes | 32 | 32 | scratch: one box per triangle |
| Morton keys and values, a and b (the sort's ping-pong) | 16 | 16 (+4 bytes) | scratch |
| sort offsets, tile tables, scan tile sums, stats, uniforms | 0.1 | 0.1 | scratch: per 2048-key tile, per mesh, or fixed |
| tinybvh's 63-bit codes (`mortonCodes: 'tinybvh-63'` only) | 8 | – | scratch |
| deltas (the split metric) | 8 | – | scratch |
| climb scratch | 52 (3 words per I, a 9-word record and a cluster per T) | 76 (19 words per I) | scratch: handed-over words, records, clusters |
| links (H-PLOC) / node info + rank (LBVH) | 28 (two scans, 2 children, 2 parents, tree order) | 16 | scratch |
| **scratch** | **136** | **140** | |
| output nodes | 64 per I | 64 | output |
| output triangle records | 48 | 48 | output |
| output parents | 4 per I | 4 | output |
| **outputs** | **116** | **116** | |
| **total** | **252 + inputs** | **256 + inputs** | Sponza: 270.6 and 274.6 B/tri with its 18.5 of inputs |

**Predicted against measured.** One H-PLOC batch of all of Sponza: inputs 101.3 + scratch 745.6 + outputs 635.5 = **1,482.4 MiB** predicted (`hplocBatchBytes`, printed by the `scene-trees-sponza` page). `SceneTrees.memory.peak` measured **1,482.7 MiB** in every mode (`copy`, `adopt`, `copy-merged`). The 0.3 MiB difference is the scene's empty buffers and top level. The peak comes while the batch builds, before the scene's own buffers hold anything.

### The lever: `maxBatchBytes`

Off by default (today's path: one batch per update, as big as the device allows).

- **`SceneTrees.create(device, { gpuMaxBatchBytes })`**, or **`GpuFirstTrees.build(requests, { maxBatchBytes })`**: the update's meshes are split into batches of at most that many bytes (inputs, scratch and outputs, exactly as `hplocBatchBytes` counts them), in mesh order.
  - The batches are built one after another, each with its own submit and stats readback, on one **`GpuHplocScratchPool`**.
  - The pool holds the inputs, scratch and uniforms. It is allocated once, each buffer at the largest size any batch needs (`hplocScratchPoolBytes`; in practice the largest batch's), and freed after the last batch.
  - Each batch's outputs are its own. A batch starts only once the previous batch's outputs are let go (`GpuTreeOutputs.letGo`): copied into the scene's storage by an update and released by the next, or adopted.
  - So the peak is about **the storage plus one batch**. Without that wait, every batch's outputs would pile up.
- **The storage grows once.** When a capped load's first batch is placed, the node and triangle storages also reserve room for the meshes whose GPU build is still running. Their triangles are known exactly, and their nodes are estimated at the placed GPU trees' nodes per triangle. Without the reservation, each batch's arrival doubled the buffers: the first cut of the lever measured 1,175–1,295 MiB peaks with 311 MiB retired at once, and 623 MiB settled.
- **Standalone:**
  - `planGpuHplocBatches(device, meshes, { maxBatchBytes })` and `planGpuLbvhBatches(...)` plan the batches;
  - `GpuHplocBuilder.createScratchPool(batches)` and `createBatch(meshes, options, pool)` build them on one pool;
  - `createMeshLayoutBuilder(device, { maxBatchBytes })` does all of it (it reads each batch's trees back and frees them before the next). The LBVH gets the planner's cap but no pool: its batches allocate and free their own scratch, one after another.
- **A mesh bigger than the cap** builds alone, in its own batch, and the pool then follows that batch. The cap is a target, not a guarantee.
- **Costs:**
  - Every batch after the first waits for two `SceneTrees.update`s, so the caller must keep updating; awaiting `whenBuilt` without updating never resolves.
  - While a later batch waits, its meshes' CPU arrays are copied into the tinybvh worker rather than moved (`transferGeometry`, merged groups), because the GPU build still has to upload them.
  - Each batch has its own submit and readback.
- **Undo:** leave `gpuMaxBatchBytes` / `maxBatchBytes` unset. The uncapped path is the code as it was.

This is batching, which is ours. The H-PLOC and LBVH kernels are unchanged, and so is every tree they build: the oracle pages compare the trees with tinybvh's and the CPU walk. So `docs/DIVERGENCES.md` has no row for it.

### Measured trade-off (Sponza, Apple M5, `scene-trees-sponza`)

Copy mode, one update per animation frame (60 Hz), after a warm-up load that compiles the pipelines. The columns are:

- **peak:** `SceneTrees.memory.peak` over the load and the swap;
- **in use:** every mesh on its GPU tree, before the swap to tinybvh's trees;
- **settled:** after every swap;
- **first trees:** from the first update until every mesh is on a GPU tree;
- **GPU build:** the sum of each batch's time from its submit to its stats on the CPU.

Correctness on every row: the GPU walk gave the CPU reference's hits on 20,000 rays, with the GPU trees in use and settled, with 0 disagreements.

| Cap | Batches | Peak MiB | In use MiB | Settled MiB | First trees ms | GPU build ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none (today) | 1 | 1,482.7 | 1,105.6 | 470.0 | 166 | 108 |
| 512 MiB | 3 | 963.0 | 675.9 | 460.7 | 214 | 111 |
| 256 MiB | 7 | 712.1 | 463.7 | 458.6 | 331 | 93 |
| 128 MiB | 13 | 581.4 | 485.4 | 453.3 | 443 | 121 |

- Each capped peak is the storage (about 461 MiB with its reservation) plus one batch, just as predicted: storage + pool 287 + outputs 215 = 963 at 512 MiB, and storage + 145 + 108 = 712 at 256 MiB.
- Settled memory is a little lower than today's, because the reservation estimates the nodes rather than adding 1/8 headroom.
- The GPU time doesn't change. The first trees come about 2 frames per batch later.
- `adopt` (1,482.7 peak, 613.6 settled) and `copy-merged` (one world-space group of 5.7M triangles, so one batch whatever the cap) were not run capped on Sponza. The `scene-trees-mid-swap` oracle covers `copy`, `copy-merged` and `adopt` with a 4 KiB cap (a batch per mesh), for correctness.

**Default:** no cap, for Dennis to decide. No cap is better on every count but one: 256 MiB halves the peak (−770 MiB) and settles 11 MiB lower, but the first trees come 165 ms later at 60 Hz. A 256 MiB default is the proposal.

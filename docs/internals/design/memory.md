# Memory: where our bytes go, against three-mesh-bvh (BX.19)

This is the breakdown P9 of [bvh-package-split.md](bvh-package-split.md) asked for: every buffer of our trees on Sponza and Bistro exterior, the GPU build's transient buffers, and three-mesh-bvh's, with the accounting made fair. It is measurement only: nothing in `src/` changed. The levers (BX.20) come after Dennis has read it.

Everything here comes from `tools/memory-breakdown.ts` (Node, no GPU) except two things taken from the `scene-trees-sponza` page on the Apple M5, which the tool's numbers are checked against, and the hit-record timing at the end.

```
BVH_ASSETS_DIR=... BVH_MANIFESTS_DIR=... node --experimental-transform-types tools/memory-breakdown.ts sponza [--json out.json] [--no-twins] [--no-tmbvh]
```

**Units.** MiB are 2^20 bytes, as `SceneTrees.memory` and the HANDOFF use them (the HANDOFF's "276 MB" of triangle records is 263.0 MiB). Every figure is an exact byte count of a buffer or typed array unless the text says it is a formula or an estimate.

## The scenes

| | Sponza | Bistro exterior |
| --- | --- | --- |
| Manifest | `sponza` (Intel New Sponza with curtains) | `bistro-exterior` |
| Meshes (each used by one instance) | 132 | 1,296 |
| Triangles | 5,744,384 | 2,829,226 |
| Vertices | 3,108,999 | 2,920,632 |
| Largest mesh, triangles | 216,266 | 82,015 |
| Positions the app holds (`float32 x 3`) | 35.6 MiB | 33.4 MiB |
| Indices the app holds (32-bit, as loaded) | 65.7 MiB | 32.4 MiB |
| **Geometry the app already holds** | **101.3 MiB** | **65.8 MiB** |

Only positions and indices count as geometry: normals, UVs and tangents are not read by either library. The loader always hands over 32-bit indices, so the index bytes are what a 32-bit index buffer costs; an app with 16-bit indices on small meshes would hold less, and so would three-mesh-bvh's reordered index.

## Headline: ours against three-mesh-bvh

Two numbers per library, side by side:

- **Beyond geometry**: bytes the BVH adds to what the app already holds. For three-mesh-bvh that is the node buffers (it reuses the positions and reorders the geometry's own index in place; a non-indexed geometry would also get a new index, and `indirect` mode adds an index-sized buffer). For us it is every byte, because our triangle records copy the vertices and nothing of the app's geometry is read.
- **Total in memory**: that plus the geometry the app holds (counted once; the app keeps it for rendering either way).

Ours is the tinybvh tree set that `SceneTrees` settles on (`buildTraceScene` over `buildMeshBvhWithTinyBvh`, leaves of up to 8), GPU-resident bytes in use.

| Sponza, MiB | Beyond geometry | Total in memory |
| --- | ---: | ---: |
| **Ours**, `f32` nodes (nodes 185.1 + triangle records 263.0 + top level <0.1) | **448.1** | **549.4** |
| three-mesh-bvh, defaults (CENTER, leaf 10) | 51.2 | 152.6 |
| three-mesh-bvh, SAH, leaf 4 | 109.3 | 210.7 |
| three-mesh-bvh, SAH, leaf 10 | 50.5 | 151.8 |
| three-mesh-bvh, CENTER, leaf 10, `indirect` | 66.9 | 168.2 |
| Ours against defaults | 8.8x | 3.6x |

| Bistro exterior, MiB | Beyond geometry | Total in memory |
| --- | ---: | ---: |
| **Ours**, `f32` nodes (nodes 80.4 + triangle records 129.5 + top level 0.2) | **210.1** | **275.9** |
| three-mesh-bvh, defaults (CENTER, leaf 10) | 26.3 | 92.1 |
| three-mesh-bvh, SAH, leaf 4 | 53.4 | 119.2 |
| three-mesh-bvh, SAH, leaf 10 | 24.9 | 90.7 |
| three-mesh-bvh, CENTER, leaf 10, `indirect` | 32.0 | 97.8 |
| Ours against defaults | 8.0x | 3.0x |

These tables were made with `f32` nodes, the default until 2026-11-13. With quantized nodes, the default since (lever 3, "Quantized nodes" below), ours is 355.5 MiB beyond geometry and 456.8 in total on Sponza (6.9x and 3.0x three-mesh-bvh's defaults), and 169.9 and 235.7 on Bistro exterior (6.5x and 2.6x).

Reading it:

- The old "302 MB against 153 MB" is a total against a total. 153 is right for three-mesh-bvh on Sponza (152.6 here), but about two thirds of it is the geometry the app holds anyway. The fair comparison is the **beyond-geometry** column: eight to nine times, not two.
- Per triangle: we add 81.8 bytes (33.8 of nodes, 48 of record); three-mesh-bvh adds 9.3 (CENTER) to 20.0 (SAH leaf 4).
- A "settled" `SceneTrees` holds 470.0 MiB of GPU buffers for Sponza (measured, `scene-trees-sponza`, copy mode): the 448.1 above plus 21.8 MiB of growth room in the node buffer (206.9 MiB allocated, 185.1 in use). In `adopt` mode it settles at 613.6 MiB, because the batch's node buffer keeps one node per triangle (350.6 MiB for 185.1 in use).
- Where the two differ: three-mesh-bvh's nodes are 32 bytes with one box each and leaves of about 7 triangles (CENTER, target 10); ours are 64 bytes with two child boxes each and leaves of 1.9 (tinybvh's SAH on these scenes). It has 1.68M nodes on Sponza, we have 3.03M. Neither choice is free: small leaves are why the walk is fast.

## How each number was measured

### Ours (`buildTraceScene` over tinybvh)

| | Sponza | Bistro ext. | How |
| --- | ---: | ---: | --- |
| Nodes (64 B) | 185.1 MiB (3,033,036) | 80.4 MiB (1,317,646) | `TraceSceneData.nodes.byteLength`; the settled GPU buffer is the same bytes plus growth room |
| Triangle records (48 B) | 263.0 MiB | 129.5 MiB | `TraceSceneData.triangles.byteLength` (one per triangle) |
| Parents (4 B per node) | 11.6 MiB | 5.0 MiB | the trees' `parents` arrays: CPU only, never uploaded to the scene's storage; they exist in the worker's result and are dropped after the upload |
| Top level, instanced | 16.9 KiB | 162 KiB | `TopLevelBvh.buffer.byteLength` (CPU build); `GpuTopLevel` adds about 170 bytes per instance for its tables (`SceneTrees.memory.topLevel` read 0.1 MiB on Sponza) |
| Top level, merged | 192 B | 192 B | the same, for one merged group |
| Average triangles per leaf | 1.89 | 2.15 | triangles over `leafCount` |
| Maximum depth | 28 | 27 | |

**Merged static** (`mergeStaticInstances`, every instance static): all 132 (Sponza) and all 1,296 (Bistro) meshes are used by one instance each, so each scene becomes **one** world-space group, and no mesh keeps its own tree. Nodes 185.2 MiB and 80.5 MiB, triangle records unchanged (263.0, 129.5), top level 192 bytes. The resident cost is the same as instanced (+0.1 MiB of nodes; `scene-trees-sponza` read 470.4 against 470.0). What it adds is **CPU, transient**: `mergeStaticGeometry`'s plan holds world-space positions and indices plus 11 bytes per triangle of source ids, 161.6 MiB (Sponza) and 95.5 MiB (Bistro) while the group builds. One group of 5.7M triangles is also one H-PLOC batch of 5.7M triangles (below).

**Leaf size** (tinybvh, `maxLeafPrimitives` 1, 2, 4, 8; nodes in MiB):

| | 1 | 2 | 4 | 8 (default) |
| --- | ---: | ---: | ---: | ---: |
| Sponza | 350.6 | 195.9 | 185.6 | 185.1 |
| Bistro exterior | 172.6 | 100.1 | 83.6 | 80.4 |

Leaves above 2 triangles hardly matter to tinybvh's SAH on these scenes, which is why raising the cap from 4 to 8 saved 0.5 and 3.2 MiB.

**H-PLOC and LBVH trees** (CPU twins, which write the same trees as the GPU builders; the H-PLOC twin's 183.9 MiB of Sponza nodes is what `SceneTrees` reports as `nodesInUse` on the GPU):

| | Sponza nodes used | Bistro nodes used | Allocated by the GPU builders (one node per triangle less one per mesh) |
| --- | ---: | ---: | ---: |
| H-PLOC twin | 3,013,617 = 183.9 MiB | 1,349,477 = 82.4 MiB | 350.6 MiB / 172.6 MiB |
| LBVH twin | 3,263,771 = 199.2 MiB | 1,426,854 = 87.1 MiB | 350.6 MiB / 172.6 MiB |
| tinybvh (for reference) | 3,033,036 = 185.1 MiB | 1,317,646 = 80.4 MiB | |

Triangle records (48 B per triangle) and parents (4 B per node) are the same sizes for all three builders. The GPU builders allocate their output nodes for the worst case (one node per triangle, 64 bytes), then leaf collapse leaves about half unused: that is the 165.5 MiB of slack on Sponza behind `adopt`'s 613.6 MiB (350.6 allocated, 185.1 in use after the swap).

### The top level

`GpuTopLevel` is small at these instance counts (132 and 1,296): its buffers are the top-level buffer (header, 64-byte records, 64-byte nodes), a record-slot list, instance boxes (32 B), a 28-word instance table and an 8-word mesh table per mesh, plus the refit's topology. About 0.1 MiB (Sponza measured) to 0.3 MiB.

### The CPU copy `GpuMeshRefit.setTree` needs

Only for **deforming** meshes (a static scene has none). Computed per mesh from `GpuMeshRefit.setTree` and `GpuBvhRefit.setShape` (`tools/memory-breakdown.ts`, `refitBytes`), summed here as if every mesh deformed:

| | Sponza | Bistro ext. | What |
| --- | ---: | ---: | --- |
| CPU copy of the layout the caller holds | 459.7 MiB | 215.0 MiB | `readLayouts` result: nodes + triangle records + parents |
| of which `setTree` reads | 196.7 MiB | 85.4 MiB | nodes + parents only (it never reads the 48-byte records) |
| `setTree`'s own CPU arrays (transient) | 69.4 MiB | 30.2 MiB | parent slots, leaf slots, level tables, the topology upload |
| Refit's GPU buffers | 314.2 MiB | 146.8 MiB | per-triangle boxes (32 B), topology (twice its words), scratch (24 B per node, twice) |

The largest Sponza mesh (216,266 triangles) alone: 17.8 MiB CPU copy (7.9 needed), 12.2 MiB GPU.

### Worker copies

From `@pmndrs/bvh/worker` (`src/worker/pool.ts`, `src/trace/mesh-bvh-async.ts`):

- **Inputs are copied** into the worker (structured clone) unless `transferInput` is set: one mesh at a time per worker, so at most the largest mesh's positions and indices times the pool size (Sponza: 3.7 MiB x 4 workers; Bistro: 1.8 MiB x 4).
- **Results are transferred**, not copied: nodes, triangles, parents and bounds move as `ArrayBuffer`s. On the main thread they exist from the worker's reply until the upload and the garbage collector let them go (the largest Sponza mesh: 17.8 MiB).
- **Each worker keeps a WASM heap.** tinybvh's memory starts at 64 MiB (measured by capturing the instance's `memory` and building the largest mesh: it stayed 64 MiB, because the input, tinybvh's tree and the layout fit). The default pool is up to 4 workers, so 256 MiB of process memory, held for as long as the pool lives. This is CPU memory, not GPU.
- **One huge mesh grows it, and it never shrinks.** Sponza merged into one 5,744,384-triangle mesh (the memory size after each step of the wrapper's calls, then `compare/node/memory.ts` for the heap growth), the memory after each step: start 64 MiB; our input copies (positions 36 MiB, indices 66 MiB) 101 MiB; `tbvh_build` (tinybvh's 16-byte vertex copy, 47 MiB, and its own build arrays) 700 MiB; `tbvh_write_layout` 1,349 MiB. The layout's growth (649 MiB) is mostly the output: nodes and parents allocated for the worst case of `triangles - 1` nodes (373 MiB, of which the 3,033,780 nodes used take 197 MiB) and the 263 MiB of triangle records, which are then copied out. So about half is tinybvh's (its arrays), half the output, and 102 MiB our copies. Wrapper-side levers, all measured:
  - our positions copy is freed right after `tbvh_build` (tinybvh keeps its own; the indices it reads in place): 1,349 to 1,316 MiB;
  - `createTinyBvhBuilder`'s `releaseAboveBytes` (default 256 MiB): after a build that leaves the memory larger, the builder drops its instance and starts a fresh one on the next build. In a fresh Node process the build's heap growth (V8 heap, array buffers and WASM memory after gc, the tree alive) went from 1,707 MiB (1,252 MiB of it WASM) to 391 MiB: the tree's 460 MiB less the 64 MiB first instance, which was released too;
  - the worker pool's `recycleAboveBytes` (default 256 MiB): a worker whose build grew past it is ended after answering (`stats.recycled`), so the memory goes back at once instead of at the worker's next garbage collection.
  Not wrapper-side: allocating the layout's nodes to the count used, or freeing tinybvh's build arrays before the layout is written, would cut the peak itself, but both change `native/` (a rebuild with the pinned emsdk). Per-mesh builds (how `SceneTrees` builds Sponza) stay at 64 MiB and never release.

### GPU transient: H-PLOC and LBVH build batches

Computed by `hplocBatch` and `lbvhBatch` in the tool, which repeat the `createBuffer` formulas of `GpuHplocBatch` (`src/gpu-hploc.ts`) and `GpuLbvhBatch` (`src/gpu-lbvh.ts`) over the real `placeMeshes`, `buildTables`, `sortTileTable` and `scanTileSumWords`, and each buffer is rounded as the builders' `make` does (4-byte aligned, at least 16). The batch plan is the real `planMeshBatches` with the builders' bytes-per-triangle constants (H-PLOC 64, LBVH 76) and a fake device with the given buffer limit.

One batch holding all of Sponza (5,744,384 triangles), per triangle in the right column:

| Buffer | H-PLOC MiB | B/tri | LBVH MiB | B/tri |
| --- | ---: | ---: | ---: | ---: |
| Inputs: positions, indices, flags, stats | 101.3 | 18.5 | 101.3 | 18.5 |
| **Outputs**: nodes | 350.6 | 64 | 350.6 | 64 |
| **Outputs**: triangle records | 263.0 | 48 | 263.0 | 48 |
| **Outputs**: parents | 21.9 | 4 | 21.9 | 4 |
| Scratch: primitive boxes | 175.3 | 32 | 175.3 | 32 |
| Scratch: sort keys and values, a and b | 87.5 | 16 | 87.5 | 16 |
| Scratch: deltas | 43.8 | 8 | | |
| Scratch: climb scratch (H-PLOC) | 284.9 | 52 | | |
| Scratch: links (H-PLOC) | 153.4 | 28 | | |
| Scratch: inner-node scratch, info, rank (LBVH) | | | 503.9 | 92 |
| Scratch: tables, sort offsets, params | 0.5 | 0.1 | 0.5 | 0.1 |
| **Inputs + scratch** | **847.0** | 154.6 | **869.0** | 158.6 |
| **Outputs** | **635.5** | 116 | **635.5** | 116 |
| **Batch total** | **1,482.4** | 270.6 | **1,504.3** | 274.6 |

Bistro exterior (2,829,226 triangles): H-PLOC 66.0 + 367.4 + 312.9 = 746.3 MiB; LBVH 757.0 MiB.

**How it squares with the measured Sponza numbers** (`scene-trees-sponza`, `gpuTreeOutputs: 'copy'`, Apple M5, today's commit; the same numbers as the HANDOFF):

- **Peak 1,482.7 MiB** is this batch, 1,482.4 MiB, to 0.3 MiB. The peak is sampled while the batch builds, before the scene's own buffers have anything in them: `SceneTrees.memory` counts `gpuBuilding`, which is `batch.scratchBytes + batch.outputBytes` summed over the batches `GpuFirstTrees` recorded. Half of the peak (50.3%) is scratch, 6.8% inputs, 42.9% outputs.
- **1,105.6 MiB "in use"** is 470.0 MiB resident (nodes 206.9 + triangles 263.0 + top level 0.1) plus 635.5 MiB of outputs held until the next update: the scratch and inputs (847.0 MiB) are gone by then, because `finish()` destroys them.
- **470.0 MiB settled** after the swap to tinybvh's trees.

**Every batch is created before the first submit.** `GpuFirstTrees.build` makes all the batches (each allocates its buffers in its constructor) and encodes them into one command encoder, so splitting a scene into more batches does **not** lower the peak by itself: the sum is the same (the "all at once" column below, 1,482.5 MiB at every limit). The planner's batches only help if they run one after another, each batch's scratch freed before the next is made. Then the peak is all the outputs plus the biggest batch's inputs and scratch:

| Buffer limit (triangles per H-PLOC batch) | Sponza batches | Sponza one at a time, MiB | Saved vs 1,482.4 | Bistro ext. batches | Bistro one at a time, MiB | Saved vs 746.3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 MiB (262k) | 27 | 674.4 | 808.0 | 12 | 354.5 | 391.8 |
| 32 MiB (524k) | 12 | 712.3 | 770.1 | 6 | 394.9 | 351.4 |
| 64 MiB (1.05M) | 6 | 786.9 | 695.5 | 3 | 474.8 | 271.5 |
| 128 MiB (2.1M; WebGPU's default `maxStorageBufferBindingSize`) | 3 | 944.8 | 537.6 | 2 | 634.5 | 111.8 |
| 256 MiB | 2 | 1,227.9 | 254.5 | 1 | 746.3 | 0 |
| 512 MiB and up (this Mac's one batch) | 1 | 1,482.4 | 0 | 1 | 746.3 | 0 |

This Mac's adapter limits let one batch hold all of Sponza (the page's one-batch peak shows it). The floor is the outputs themselves (635.5 MiB), which `adopt` turns into the scene's storage (no copy), at the cost of the node slack above.

**The lever, measured** (`gpuMaxBatchBytes`, **256 MiB by default** in `SceneTrees`, `Infinity` for none; `maxBatchBytes` on the standalone builders, no cap unless given; [bvh-builders.md, "Build peak memory"](bvh-builders.md#build-peak-memory-p9)). It goes one step past the "one at a time" column above, because it also lets go of each batch's outputs before the next batch starts:

- the batches build one after another on one scratch pool;
- each batch's outputs are copied into the storage (in a submit of their own, with no update needed) and freed before the next batch;
- the storage reserves room for the rest at the first growth.

So the floor is the storage plus one batch, not all the outputs. On Sponza (Apple M5, copy mode), the peak goes from 1,482.7 MiB to 963.0, 711.0 and 581.4 MiB at caps of 512, 256 and 128 MiB, with settled memory 461, 459 and 453 MiB, and the first trees as soon as uncapped (147, 166 and 192 ms against 171). The default is 256 MiB: −771 MiB of peak on Sponza, for first trees that the first cut delayed by about 165 ms and that now come no later.

## three-mesh-bvh 0.9.18

Built per mesh on the same geometry (a fresh `BufferGeometry` per mesh with the scene's positions and its own copy of the index, because it reorders the index in place), `verbose: false`. The leaf option in 0.9.18 is `targetLeafSize` (default 10, a target with a depth limit of 40, not a hard cap); there is no `maxLeafTris`.

| MiB | Sponza CENTER leaf 10 | SAH leaf 4 | SAH leaf 10 | CENTER `indirect` | Bistro CENTER | SAH 4 | SAH 10 | `indirect` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nodes, `_roots` (32 B each) | 51.2 | 109.3 | 50.5 | 51.2 | 26.3 | 53.4 | 24.9 | 26.3 |
| Node count | 1,679,026 | 3,583,138 | 1,654,408 | 1,679,026 | 862,496 | 1,749,666 | 816,940 | 862,496 |
| Added index (`_indirectBuffer`) | 0 | 0 | 0 | 15.7 | 0 | 0 | 0 | 5.7 |
| Index (the geometry's own, reordered) | 65.7 | 65.7 | 65.7 | 65.7 | 32.4 | 32.4 | 32.4 | 32.4 |
| Positions (the geometry's own) | 35.6 | 35.6 | 35.6 | 35.6 | 33.4 | 33.4 | 33.4 | 33.4 |
| **Beyond geometry** | 51.2 | 109.3 | 50.5 | 66.9 | 26.3 | 53.4 | 24.9 | 32.0 |
| **Total in memory** (by hand) | 152.6 | 210.7 | 151.8 | 168.2 | 92.1 | 119.2 | 90.7 | 97.8 |
| `estimateMemoryInBytes` | 152.7 | 210.8 | 152.0 | 168.4 | 93.6 | 120.7 | 92.2 | 99.3 |

- **By hand** is the sum of every `_roots` buffer's `byteLength` (node size `BYTES_PER_NODE` = 32: six floats of bounds and two words), plus `_indirectBuffer`, plus the index and position arrays. `estimateMemoryInBytes` is within 0.1% on Sponza and 1.6% on Bistro: it walks the object graph and adds the size of every key name and number, which is the difference, about 1.5 MiB over 1,296 `MeshBVH` objects. It counts the geometry too, so it is a total, not a beyond-geometry figure.
- **Index.** The geometry here is indexed, so nothing is added: `MeshBVH` sorts the index in place (65.7 and 32.4 MiB, the same bytes as the app's). A non-indexed geometry gets a new 32-bit index of one entry per vertex (12 bytes per triangle) that is then the app's to keep. `indirect` leaves the geometry's index alone and adds a separate triangle-order array (4 bytes per triangle, 16-bit under 65,536 triangles).
- Build times on this Mac, for scale only: CENTER 1-2 s, SAH 3-9 s per scene.

## Where our bytes go

Sponza, GPU-resident, settled in `copy` mode: 470.0 MiB, of which

| | MiB | Share |
| --- | ---: | ---: |
| Triangle records, 48 B x 5.74M | 263.0 | 56.0% |
| Nodes in use, 64 B x 3.03M | 185.1 | 39.4% |
| Node buffer growth room | 21.8 | 4.6% |
| Top level | 0.1 | |

Bistro exterior (computed, no growth room): 129.5 records (61.6%), 80.4 nodes (38.3%), 0.2 top level. **The triangle records are the biggest single item**, as the first numbers said, and the nodes come second. The transient buffers during the load are larger than all of it: 1,482 MiB at the peak.

### Levers ranked by the bytes each could save

Computed from the tables above, Sponza first and Bistro exterior second (MiB). "Resident" is what stays after the load, "peak" is the load's high-water mark. These are bounds from the layout arithmetic; none has been built, and speed (the bench's closest-hit and any-hit) and watertightness are unmeasured, so they are what the levers could save, not what they will.

| # | Lever | Kind | Saves (Sponza / Bistro) | Computed as |
| ---: | --- | --- | ---: | --- |
| 1 | **Scratch pooling, or batches built one after another with their scratch freed** | peak | **537.6 to 808.0 / 111.8 to 391.8** | peak 1,482.4 down to 944.8 (128 MiB batches, the default device) or 674.4 (16 MiB). The scratch and inputs are 57% of the peak. Smaller batches alone save nothing (all batches are created before the submit). Resident bytes unchanged. |
| 2 | **Indexed leaves** | resident | **241.1 to 263.0 / 118.7 to 129.5** | drop the 48-byte records; keep a 4-byte original-triangle id each (21.9 / 10.8 MiB) or, like three-mesh-bvh, reorder the app's index in place (nothing kept). Net of the geometry the walk would then read (101.3 / 65.8 MiB), if the app does not already hold it on the GPU: 139.8 to 161.7 / 52.9 to 63.7. |
| 3 | **Compressed nodes** (built as quantized nodes, below) | resident | **92.6 to 138.8 / 40.2 to 60.3** | 64 to 32 bytes halves the 185.1 / 80.4; 64 to 16 (CWBVH-like) saves three quarters. |
| 4 | Freeing CPU copies | CPU | 256 / 256 (WASM heaps); 263.0 / 129.5 (deforming meshes) | terminating idle pool workers frees 4 x 64 MiB of WASM heap; a deforming mesh's retained layout needs only nodes and parents, so its 48-byte records can go (the whole 263.0 MiB if every mesh deforms). Static scenes have no CPU copy to free. |
| 5 | Larger leaves | resident | 0.5 to 141.3 / 3.2 to 58.8 | measured: allowing 8 instead of 4 saved 0.5 / 3.2 MiB, because tinybvh's SAH picks 1.9 / 2.1 triangles per leaf. The upper bound is leaves forced to 8 full triangles (nodes = triangles / 8 x 64 B = 43.8 / 21.6 MiB), which is a builder change, a speed question, and past the layout's cap of 8. |
| 6 | Smaller triangle records | resident | 43.8 to 109.6 / 21.6 to 54.0 | 48 to 40 bytes (corners 36 + a hit-time table for id and flags, which still needs the walk to read flags for the filter) saves 8 per triangle; 28 bytes (one corner and two 16-bit edges relative to the leaf, plus id) saves 20, and costs bit-exact shared vertices that the watertight test relies on. Overlaps lever 2. |
| 7 | Parents only when needed | peak / CPU | 21.9 / 10.8 (peak), 11.6 / 5.0 (CPU) | nothing resident: static trees never upload parents. The H-PLOC `outParents` buffer is in the peak (21.9 MiB); the CPU `parents` arrays travel with every tinybvh result. |

Also, not on the P9 list but measured:

- **Node buffer growth room**: 21.8 MiB in `copy` mode, 165.5 MiB in `adopt` mode (350.6 allocated for 185.1 in use). Trimming the node buffer after the swaps saves that much: a lever for `adopt`, which is otherwise the lowest-peak path to a settled scene.
- **Refit buffers** (if every mesh deforms): 314.2 / 146.8 MiB of GPU, of which 175.3 / 86.3 MiB are per-triangle boxes (32 B each) that one buffer sized for the largest mesh (6.6 / 2.5 MiB) could serve in turn: up to 168.7 / 83.8 MiB with pooling. Static scenes pay none of it.
- **Merged static** adds a CPU plan of 161.6 / 95.5 MiB while it builds, and makes one batch of a whole scene; the peak is then the batch of lever 1 whatever the instance count.

Levers 4 and 7 and the three notes above (the node buffer's room, refit buffers, and the quantize jobs) are built as options since BX.20, all off by default: "BX.20: the levers that keep the layout" below has what each saved.

**Reading the ranking.** The load's peak, not the settled scene, is the biggest number and the cheapest to cut (lever 1 changes scheduling, not the layout, so `GPU_LAYOUT_VERSION` stays). Of the resident levers, only indexed leaves and compressed nodes are large. Together, keeping a 4-byte id per triangle and 32-byte nodes, they would take Sponza from 448.1 to about 115 MiB beyond geometry (216 if the geometry has to be added to the GPU for them), against three-mesh-bvh's 51. What is left is the 3.03M nodes: their count follows from leaves of 1.9 triangles, which is lever 5, and the speed measurement has to decide that one.

## Indexed leaves (lever 2, built; off by default, the memory option)

Lever 2 as option B of [indexed-leaves.md](indexed-leaves.md): 16-byte tree-ordered triangles `vec4u(i0, i1, i2, id + flags)`, a 16-byte header per mesh, and our own copy of the positions, welded by exact bits. `tools/memory-breakdown.ts` (Node, the same tinybvh trees, `buildTraceScene`), MiB:

| | Sponza | Bistro exterior |
| --- | ---: | ---: |
| Records (the default), `f32` nodes: nodes + 48 B triangles + top level | 185.1 + 263.0 + 0.0 = **448.1** | 80.4 + 129.5 + 0.2 = **210.1** |
| Indexed, `vec4f` positions, welded (the indexed default) | 185.1 + 87.7 + 45.2 = **318.0** (−130.1) | 80.4 + 43.2 + 31.7 = **155.5** (−54.6) |
| Indexed, `vec4f`, unwelded | 320.2 | 168.3 |
| Indexed, packed `f32` positions, welded | **306.7** (−141.4) | **147.5** (−62.6) |
| Merged static, indexed, `vec4f`, welded (records merged: 448.2 / 210.0) | 185.2 + 109.6 + 45.2 = 340.0 | 80.5 + 54.0 + 31.7 = 166.2 |
| Vertices after the exact-bit weld (before) | 2,963,523 (3,108,999) | 2,077,661 (2,920,632) |

- The triangles are exactly 16 bytes each plus 16 per mesh (87.7 against 263.0 on Sponza). The positions are the new cost: 45.2 MiB of `vec4f` on Sponza after the weld. The weld saves 5% of Sponza's vertices and 29% of Bistro exterior's.
- A merged group adds its 4-byte side table per triangle (21.9 MiB on Sponza, 10.8 on Bistro), as section 5 of the design said.
- Against three-mesh-bvh, which keeps no positions of its own: the positions copy is what the app already holds (35.6 / 33.4 MiB as `f32 x 3`), so an app that could share its positions buffer would save it again (option A, not built).
- **The build peak**: an H-PLOC batch writes 16-byte triangles too, so a batch goes from 270.6 to 238.6 bytes per triangle on Sponza (276.6 to 244.6 on Bistro exterior); one uncapped Sponza batch from 1,482.4 to 1,307.1 MiB. With `gpuMaxBatchBytes`, about 13% more triangles fit under the cap.
- **`SceneTrees` on Sponza** (`scene-trees-sponza`, M5, the default 256 MiB cap, settled after the swaps): records 458.6 MiB, peak 711; indexed `vec4` 327.1 (nodes 194.1 with growth room, triangles 87.7, positions 45.2), peak 580; indexed packed 315.8, peak 568.7. Merged, uncapped: records 470.4, peak 1,482.6; indexed 362.2, peak 1,350.9. First trees 130 ms indexed against 118 (records), tinybvh 300 ms against 285. 0 disagreements with the CPU reference in every state.
- The CPU trees (`CpuMeshBvh`) stay on records by default: an indexed tree is expanded with its source on construction. `corners: 'positions'` (2026-11-14) keeps it indexed, reading corners from the source's positions; `/three`'s `BvhBoundsTree` uses it for indexed trees since 2026-11-15 (indexed-leaves.md, "What is in").
- Speed (M5, indicative): `vec4` indexed traces at 0.83-0.99x records' rate (median of rounds), packed at 0.75-0.92x. RTX 3060 Laptop (2026-11-13, median of three runs): `vec4` 0.92-0.97x, packed 0.82-0.95x. Details in indexed-leaves.md, "Measured". Records stay the default (Dennis, 2026-11-13): indexed leaves wait for an RTX 3060 run together with quantized nodes, the new default.

## Quantized nodes (lever 3, built; the default since 2026-11-13)

Lever 3 as option A of [compressed-nodes.md](compressed-nodes.md): 32-byte two-child nodes with 16-bit planes on a per-mesh grid, and one 32-byte frame slot per placed tree. `tools/memory-breakdown.ts` (Node, the same tinybvh trees, `buildTraceScene`), MiB beyond the app's geometry:

| | Sponza | Bistro exterior |
| --- | ---: | ---: |
| `f32` nodes, records (the default before 2026-11-13) | 185.1 + 263.0 + 0.0 = **448.1** | 80.4 + 129.5 + 0.2 = **210.1** |
| **Quantized nodes, records (the default now)** | 92.6 + 263.0 + 0.0 = **355.5** (−92.6) | 40.3 + 129.5 + 0.2 = **169.9** (−40.2) |
| Indexed leaves (`vec4`, welded), `f32` nodes | 318.0 | 155.5 |
| **Quantized nodes and indexed leaves** | 92.6 + 87.7 + 45.2 = **225.5** (−222.6) | 40.3 + 43.2 + 31.7 = **115.3** (−94.8) |
| Merged static, quantized, records / indexed | 355.5 / 247.4 | 169.8 / 125.9 |

- The nodes halve exactly, plus 32 bytes per tree (132 on Sponza, 1,296 on Bistro exterior: 4 KiB and 41 KiB). The design's 225.4 / 115.3 estimate holds.
- **The build peak**: GPU batches still write 64-byte nodes (the climbs need `f32` words), so the batch sizes and `gpuMaxBatchBytes` planning don't change; `SceneTrees` packs them on the way into storage, so the settled storage, and the capped peak's floor (storage plus one batch), drop by the node saving. A GPU builder asked for quantized output (`nodeFormat: 'quantized'`) holds both node buffers for the moment of its `finish`.
- **Deforming meshes** keep a 64-byte copy of their nodes on the GPU in `GpuMeshRefit` (the refit's climb runs on it): 64 bytes per node, deforming meshes only. With quantized nodes the default, every deforming mesh pays it unless the scene asks for `f32`: 96 bytes per node in all (32 in the scene, 64 in the copy) against 64 in an `f32` scene. Sponza's largest mesh has about 115k nodes, so about 7 MiB of copy if it deformed.
- `SceneTrees` on Sponza (`scene-trees-sponza`, M5, default 256 MiB cap, settled): 360.8 MiB quantized against 458.6 `f32`, peak 613.2 against 711; more in compressed-nodes.md, "Measured".
- Speed: M5 0.93-0.97x `f32` (indicative); RTX 3060 Laptop (2026-11-13, median of three runs) closest hits 1.02-1.22x, any hits 0.98-1.03x. The RTX 3060 run made it the default (compressed-nodes.md, question 2).
- The CPU trees (`CpuMeshBvh`) stay 64 bytes per node: a quantized tree is decoded on construction.

## BX.20: the levers that keep the layout (built; the defaults since 2026-11-15)

Six switches, each beside the default it would replace. None changes a byte the walk reads, so `GPU_LAYOUT_VERSION` stays 2. Each is checked to give the same trees and hits as its default: in Node (`src/scene/scene-trees-memory-levers.test.ts`, the worker pool's tests, `growable-storage.test.ts`; the fake GPU and the H-PLOC stand-in) and on SwiftShader (below). The bytes are from `tools/memory-breakdown.ts` (`tools/memory-levers.ts`), Node, on Sponza and Bistro exterior; the `SceneTrees` rows load the real `SceneTrees` on the fake GPU, its GPU trees the H-PLOC CPU twin's laid out as a batch's outputs, and its settled and peak numbers repeat the M5's `scene-trees-sponza` ones (360.8 settled and 613.3 peak with the defaults, 613.2 measured; adopt 613.6 settled, the same as measured). MiB:

| Lever | Option (default) | Kind | Sponza | Bistro exterior | How measured |
| --- | --- | --- | ---: | ---: | --- |
| 4a. Idle workers end | `BvhWorkerPoolOptions.idleTimeoutMs` (`Infinity`) | CPU, while idle | **256.0** (4 x 64.0) | 256.0 | each worker's tinybvh memory after building every mesh of the scene in turn (64.0 MiB: no per-mesh build grows it), times the default pool of 4 |
| 4b. Deforming meshes keep nodes, parents and slot words | `SceneTrees` `deformingLayouts: 'compact'` (`'full'`) | CPU, resident, deforming meshes only | **241.0** if every mesh deformed (459.7 to 218.6); the largest mesh 17.8 to 8.8 | 118.7 (215.0 to 96.3) | each mesh's `f32` tinybvh tree whole against `compactPoseLayout` of it |
| 7. Parents only when needed | `SceneTrees` `parents: 'refit'` (`'always'`) | GPU peak; CPU, transient | peak **3.7** with the default cap (613.3 to 609.6), **21.9** uncapped (1,482.6 to 1,460.6); CPU 11.6 | peak 3.7 (425.4 to 421.8), 10.8 uncapped; CPU 5.0 | `SceneTrees.memory.peak` on the fake GPU; the CPU number is tinybvh's parents arrays, which live from the worker's answer to the upload |
| Node buffer trim | `SceneTrees.trim()`, `trimWhenRefined` (off) | GPU, resident | adopt **165.5** (613.6 to 448.1 settled); copy 5.3 with the default cap, 10.9 uncapped | adopt 92.2 (302.3 to 210.1); copy 1.4 capped, 6.1 uncapped | `SceneTrees.memory.nodes` settled and after `trim()` |
| Refit boxes shared | `SceneTrees` `refitBoxes: 'shared'` (`'per-mesh'`) | GPU, resident, deforming meshes only | **168.7** if every mesh deformed (175.3 to 6.6) | 83.8 (86.3 to 2.5) | 32 bytes per triangle per mesh against the largest mesh's |
| Quantize jobs batched | `SceneTrees` `quantizeJobs: 'batched'` (`'per-tree'`) | GPU, transient | 264 buffers to 14 (4,224 bytes to 2,224), 132 passes to 7 | 2,592 buffers to 6 (41,472 bytes to 20,784), 1,296 passes to 3 | buffers and dispatches the fake GPU saw over a load |

**How each works, and what it costs.**

- **4a, `idleTimeoutMs`.** A worker with nothing to build for that long is ended (`stats.idleEnded`); the next build starts a fresh one (a module load and a WASM start, a few milliseconds, as after `recycleAboveBytes`). Inline (no `Worker`), the calling thread's tinybvh instance is dropped the same way and goes at the garbage collector's next pass. The timers are `unref`'d in Node.
- **4b, `deformingLayouts: 'compact'`.** A deforming mesh keeps its tree for `setPoseCpu`. A CPU pose rewrites every corner, so of the 48-byte records only the original triangle (word 3), the flags (7) and the padding (11) carry anything; `compactPoseLayout` keeps 4 bytes per triangle of the first, and the other two only when any is non-zero (Sponza's and Bistro's meshes have flags on none). `refitLayoutToPose` rebuilds the records from it, bit for bit (tested against the whole layout, records and indexed, `f32` and quantized, with flags, materials and padding). The nodes and parents stay: the CPU refit needs both. Cost: the pose builds the 48-byte records it used to slice, the same allocation. Static scenes keep nothing either way.
- **7, `parents: 'refit'`.** Only a refit reads parents. With it, no H-PLOC batch writes its parents output: `hplocParams.writeParents` (the struct's old `unused` word, so its size is the same) is 0 and `outParents` is a 16-byte placeholder (`GpuHplocBatchOptions.parents: false`, `GpuTreeBuffers.withoutParents`). A deforming mesh's read-back layout gets its parents from its nodes on the CPU (`parentsOfNodes`; equal to the GPU's on every `bvh-hploc` case). tinybvh builds of meshes that don't deform leave the parents array out (`MeshBvhBuildOptions.parents: false`). The batch planner still counts 4 bytes per node of parents, so capped batches don't change. The saving is small with the cap, because the capped peak holds one batch's parents.
- **Node buffer trim.** `trim()` asks for it; `trimWhenRefined` asks at every quiet update (no GPU build, nothing arrived or reserved, no tinybvh build left) whose buffer holds more than its trees. The placed trees are packed to the front of a buffer of their exact size in that update's encoder (`GrowableStorage.compact`), the old buffer retired to the next update, placements, the GPU top level's mesh table and deforming meshes' refits (`GpuMeshRefit.moveTree`) moved, `version` bumped. Costs: a copy of every tree on the GPU and, for that update, the old and new buffers both (the new one is the size in use, so 448 MiB more for a moment on adopted Sponza, still under the load's peak); the next tree after a trim grows the buffer by doubling; placements an app read before (`meshStatus`) go stale. With `adopt` it takes away the lever's settled cost: adopt's settled scene becomes copy's.
- **`refitBoxes: 'shared'`.** Deforming meshes' refits are recorded one after another in one pass each update, and their primitive boxes are written and read only inside each refit's own dispatches, so they can take turns with one buffer (`GpuRefitBoxPool`), sized for the largest deforming mesh. A growth retires the old buffer until the next update.
- **`quantizeJobs: 'batched'`.** Every GPU tree an update places (or a capped batch stores between updates) is packed by one pass, one job per batch's node buffer, instead of a pass, a mesh table and a uniform per tree. The bytes are small; on real drivers each buffer also rounds up to the allocator's granularity, and the dispatch count drops from one per tree to one per batch. Timing it is the `quantized-copy-default-qbatch` row of `scene-trees-sponza`.

**Checked on SwiftShader** (Windows, the CPU heavily loaded; `profile=ci`; correctness, not speed): `scene-trees-deforming` with two new variants (records with `f32` and with quantized nodes) that turn on every `SceneTrees` lever above, with a second deforming mesh so two refits share one box buffer, a removal whose trim moves the cloth's tree (the GPU refit, the drift rebuild and the CPU poses after it match the reference), compact layouts for the CPU poses and parents found on the CPU: 0 disagreements. `scene-trees-mid-swap`, three new `-levers` variants (adopt merged, quantized copy merged, quantized copy capped: trim, batched quantize, no parents) over every oracle case, with `?variants=levers,copy,adopt-merged&sharedRays=100000`: 0 disagreements, the node buffer the size of its trees after each swap. (The full page, 1M seam rays and every variant, ran past its 900 s limit on SwiftShader under this load before reporting; it has no lever-specific path the filtered run lacks.) `bvh-hploc` builds every case again with `parents: false`: the same nodes and triangles bit for bit, the parents from the nodes equal to the GPU's. `scene-trees-runtime` passes unchanged. `scene-trees-sponza` gained `adopt-trim`, `copy-default-trim`, `copy-default-noparents`, `copy-noparents`, `quantized-copy-default` and `quantized-copy-default-qbatch` for the next real-GPU run (`tools/windows-clean-run.ps1`); not run here (no real GPU).

**On the RTX 3060** (`scene-trees-sponza`, 2026-11-14, `WINDOWS-RUNBOOK.md` section 9; every variant 0 disagreements): `parents: 'refit'` peak 711.0 to 707.3 MB capped, 1,482.7 to 1,460.7 uncapped; `trimWhenRefined` settled 458.6 to 448.1 (copy, default cap), adopt with trim 613.6 to 448.1; `quantizeJobs: 'batched'` the same bytes (613.2 peak), first trees 257 against 309 ms in one run (single runs vary that much: two identical capped rows gave 471 and 256 ms). `refitBoxes`, `deformingLayouts` and `idleTimeoutMs` act on deforming meshes and idle pools, which Sponza doesn't exercise.

**Dennis's decision (2026-11-15): all six as recommended below, and the quantized emit (`gpuQuantizeOnEmit`) on for quantized scenes.** The table's "Option (default)" column gives the old default, which is now the way to turn each off. The recommendations as they were made:

- **`refitBoxes: 'shared'`: make it the default.** No cost found: the same boxes, one buffer instead of one per deforming mesh. Undo: `'per-mesh'`.
- **`parents: 'refit'`: make it the default.** Nothing reads the parents it drops; a deforming mesh's are found from its nodes in one pass over them (not timed). Saves 3.7 MiB of the capped peak and 11.6 MiB of transient CPU on Sponza, 21.9 MiB of peak uncapped. Undo: `'always'`.
- **`deformingLayouts: 'compact'`: make it the default.** Half the CPU a deforming mesh keeps (Sponza, every mesh deforming: 459.7 to 218.6 MiB; 44 of its 48 record bytes per triangle go), same poses bit for bit.
- **`quantizeJobs: 'batched'`: make it the default after the RTX 3060 run** times `quantized-copy-default-qbatch` against `quantized-copy-default` (expected: no slower; one pass instead of 132 on Sponza, 1,296 on Bistro exterior).
- **`trimWhenRefined`: on by default for `adopt`, off for `copy`.** With `adopt` it saves 165.5 MiB settled on Sponza, which is the lever's whole settled cost; with the default cap and `copy` it saves 5.3 MiB for a copy of every node and a moment of two node buffers. `trim()` stays for apps that know they are done loading.
- **`idleTimeoutMs`: a finite default for pools `SceneTrees` makes (say 30 s), `Infinity` for pools the app makes.** 256 MiB of CPU back once the load is over, for a few milliseconds of worker start on the next runtime add.

## The 40-byte hit record

`batchKernelWgsl(mode, { hitNormal })` writes a 40-byte record by default (material and geometric normal) and a 24-byte one with `hitNormal: false`. The surface-ray bench now times both: closest-hit, on the instanced and the merged tree, on the `surface` and `air-grid` sets, 5 rounds, the two records back to back with the order reversed every other round (`hitRecordRoundOrder`), the ratio taken within each round. The compact kernel's hit, instance, triangle, front face and barycentrics were identical to the full kernel's on every ray of every set (the page fails otherwise). Apple M5, headless Chrome 157, once per scene, commit `f6eaaca`; **indicative only**, as the M5 drifts and a run is five rounds.

Ratio = rate with the 40-byte record over rate with the 24-byte record (above 1: the 40-byte record was *faster*). Median of the rounds, and the range:

| | Tree | `surface` rays | `air-grid` rays |
| --- | --- | --- | --- |
| Sponza | instanced | 1.108 (1.077-1.146) | 1.041 (1.025-1.060) |
| Sponza | merged | 1.052 (0.958-1.083) | 1.020 (0.993-1.043) |
| Bistro interior | instanced | 1.059 (1.010-1.088) | 1.046 (1.005-1.061) |
| Bistro interior | merged | 1.030 (0.991-1.032) | 1.023 (1.011-1.055) |

The 40-byte record costs nothing measurable here: 37 of the 40 rounds (4 cases x 5 rounds x 2 scenes) came out above 1 (the three below, 0.958, 0.991 and 0.993, were merged-tree rounds), and no median fell below 1. That the bigger record is up to 11% faster is not something to rely on or to explain from this data (the two kernels are different pipelines; the walk is the same). The reading is that the walk dominates and the extra two loads after it are in the noise on this GPU. The rates behind it (Mrays/s, median of rounds, 40 / 24 byte): Sponza instanced surface 40.7 / 36.8, merged surface 73.7 / 69.5; Bistro interior instanced surface 22.1 / 20.7, merged surface 26.5 / 25.7. The same runs re-measured merged over instanced (40-byte records, today's tree): Sponza surface 1.80x closest and 1.41x any-hit; Bistro interior 1.19x and 1.10x (rows in `bench/results/trace.csv`, scenes `sponza` and `bistro-interior`, metrics `surface-rays/paired/ratio/...` and `surface-rays/paired/hit_record_40_over_24/...`).

**On the RTX 3060 Laptop it costs 6-12%** (2026-11-13, commit `63af716`, Windows, headless Chrome 157; three runs of 5 paired rounds; each figure is the median over the runs of each run's median of rounds; WINDOWS-RUNBOOK.md, section 7):

| | Tree | `surface` rays | `air-grid` rays |
| --- | --- | --- | --- |
| Sponza | instanced | 0.916 | 0.895 |
| Sponza | merged | 0.911 | 0.881 |
| Bistro interior | instanced | 0.934 | 0.938 |
| Bistro interior | merged | 0.933 | 0.931 |

So 0.88-0.94x, against 0.96-1.15x on the M5. On NVIDIA the bigger record shows; on the M5 it didn't. Why is not measured (the two kernels are different pipelines; the walk is the same). The compact kernel's hits matched the full one's on every ray. **Dennis kept the 40-byte record as the default** (2026-11-13): a hit should carry its material and normal unless the caller says otherwise. Shadow and occlusion passes that read neither should pass `hitNormal: false`, or trace any-hit rays (one word per ray). Intel has not run it. Rows: device `RTX 3060 Laptop`, metrics `median/surface-rays/paired/hit_record_40_over_24/...`.

## Reproducing

- `tools/memory-breakdown.ts` prints the figures above for a scene (about 1 minute for Sponza with the twins and three-mesh-bvh); `--json` keeps them.
- `pnpm test:gpu:page scene-trees-sponza` reports `SceneTrees`' measured memory for the check against the batch totals.
- `pnpm test:gpu:page trace-bench-surface-sponza` and `trace-bench-surface-bistro-interior`, then `node tests/gpu/pages/trace-bench/surface-to-csv.ts <runner.json> --append`, for the hit-record rows.

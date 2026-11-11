# Compressed nodes (P9 lever 3)

## Status (2026-11-12): built (option A), off by default

Dennis's answers to section 6: A first, CWBVH stays dropped; the default stays `f32` until the RTX 3060 is measured; merged groups get one frame per group, measured first (the loss is small, below, so no per-instance `f32` fallback); the node format is bits 8-15 of header word 7, `GPU_LAYOUT_VERSION` stays 2; deforming meshes keep a 64-byte `f32` copy and a re-quantize pass; the folded decode is a later, measured option, not built.

**What is in.** One walk, with the node format compiled in as `rayMasks` and `triangleLeaves` are:

- `nodeFormat: 'f32' | 'quantized'` (the option is named `quantized`, not `packed`) on `traceModuleWgsl`, `batchKernelWgsl`, `TraceBatchKernel`, `uploadTraceScene` (from the scene), `assembleTraceScene`/`buildTraceScene`, the builders (tinybvh, the topology builder, the LBVH and H-PLOC CPU twins, the GPU LBVH and H-PLOC, `createMeshLayoutBuilder`), `SceneTrees`; `BvhTraceNodes` reads it from its source. Only `bvhLoadMeshNode` and an instance-entry call `bvhEnterMeshNodes(rootNodeOffset)` differ (an empty function for `f32`); the walk's text is identical (unit-tested), and the counted walk's anchor still applies.
- **The node** is two `vec4u`, one per child: `(x, y, z, reference)`, each axis word the min plane in bits 0-15 and the max in 16-31 (`GPU_LAYOUT_QUANTIZED_NODE_WORD`). Section 1's sketch put the three left words, then the three right, then both references; per child is symmetric and is what the decode reads.
- **The frame** lives in the tree, not in a rule over `bounds`: a quantized tree's `nodes` starts with its 32-byte frame slot (origin `vec3f`, step `vec3f`), then the nodes (node `i` is slot `i + 1`), so a scene copies the whole thing and the slot lands at `rootNodeOffset - 1`. Serialized files carry it the same way, so no frame rule is part of the file format. The frame is derived from the union of the root's two children (`nodeFrameOf`, quantized-nodes.ts): per axis the smallest power-of-two step with every coordinate under 2^23 steps from 0 and the box at most 65,532 steps wide; origin = (floor(min / step) - 1) * step. Both conditions are monotone in the step, so the CPU and the GPU (gpu-quantize.ts) find the same frame.
- **Rounding** tests only exact values `x / step`: `qMin = floor(min / step) - origin / step`, one lower if the fraction is below 1/1024; the max likewise up. So the CPU and GPU encoders agree bit for bit (`bvh-blas-refit` checks it), and every decoded plane lies at least 1/1024 of a step outside the exact one. Empty boxes (min above max, the refit's point at the largest float, non-finite) encode as both planes at 65,535.
- `checkGpuLayout` checks a quantized tree's decoded boxes against the exact boxes of its triangles (containment with the margin) and its frame against `nodeFrameOf` of the exact root box. `treeSahCost` measures the decoded boxes. `CpuMeshBvh` decodes once in its constructor (CPU trees stay 64 bytes; `traverse` hands callbacks boxes around the exact ones).
- **GPU builders** still build `f32` (the climbs hand boxes over in `f32` words) and pack in `finish()` with the quantize pass (`quantizeTreeBuffers`): a compact node buffer, each mesh's frame slot then its nodes; ranges gain `parentOffset`. `readTreeLayouts` reads either.
- **Refit**: `GpuMeshRefit.setTree(layout, { nodeFormat: 'quantized' })` (`meshRefitPlacement` passes it) keeps the 64-byte copy, refits it unchanged, and dispatches the quantize pass in the same compute pass into the target's frame slot and nodes. On the CPU: `refitOnCpu` on an `f32` copy, then `quantizeNodeWords`.
- **`SceneTrees({ nodeFormat: 'quantized' })`**: 32-byte storage with a frame slot before each root. tinybvh trees arrive quantized from the workers (deforming meshes' arrive `f32`, for their refit's copy, and are packed when placed); GPU batch trees are packed by the quantize pass instead of the node copy; the capped-batch reserve counts the frame slots; `gpuTreeOutputs: 'adopt'` is refused. Works with indexed leaves.

 is 64 bytes: two child boxes as `vec3f` plus an index and a count each (`src/gpu-layout.ts:37-62`, WGSL `BvhNode` at `src/trace/wgsl.ts:162-167`). On Sponza the nodes are 185.1 of the 448.1 MiB we add beyond geometry, and 80.4 of 210.1 on Bistro exterior ([memory.md](memory.md), lever 3). This document picks a smaller node, says how the refit, the walk and the rest of the package meet it, and says what to measure before it can become the default.

The risk to beat is B3. Compressed 8-wide nodes (CWBVH) were 3.9-4.6x smaller and traced at 0.96-1.02x of our BVH2 on Sponza and 0.82x on Bistro interior (RTX 3060; ddgi `PROGRESS.md`, "2026-11-05 · B3 CWBVH spike"; [DIVERGENCES.md](../DIVERGENCES.md), "Deliberately not ported"). Dennis dropped B3 for speed.

## 1. Format

Counts from memory.md: 3,033,036 nodes on Sponza and 1,317,646 on Bistro exterior (tinybvh, leaves of up to 8).

| Option | Bytes per node | Sponza MiB (saves) | Bistro ext. MiB (saves) | Decode per inner node |
| --- | ---: | ---: | ---: | --- |
| Today: `f32` boxes | 64 | 185.1 | 80.4 | none; 4 `vec4f` loads |
| **A.** Two children, 16-bit planes on a per-mesh grid | **32** | **92.6 (−92.6)** | **40.2 (−40.2)** | 2 `vec4u` loads, 12 extracts, 12 conversions, 12 FMAs |
| A'. Two children, 8-bit planes against a per-node frame | 36 (48 aligned) | 104.1-138.8 | 45.2-60.3 | as A, plus the frame |
| B. CWBVH, 8-wide, 8-bit | 80 per wide node | 40.5 measured (−144.6) | about 17.5-20.6, estimated | a different walk |
| B'. 4-wide, 8-bit | 64 per wide node | about 62, estimated | about 27, estimated | a different walk |
| C. `f16` boxes, absolute | 32 | 92.6 | 40.2 | 2 `vec4u` loads, 6 `unpack2x16float` |

Each mesh in A also takes one 32-byte frame slot (132 x 32 B on Sponza, 41 KiB on Bistro). B's Sponza figure is B3's measurement (42,488,000 bytes against our 194.1 MB, which is 185.1 MiB). B and B' on Bistro are 80.4 MiB divided by B3's 3.9-4.6x and by three, respectively.

**A, recommended.** It is still the Aila-Laine node, at half the size:

```
struct BvhPackedNode {     // 32 bytes: two vec4u
    leftX: u32, leftY: u32, leftZ: u32,     // per axis: min in bits 0-15, max in bits 16-31
    rightX: u32, rightY: u32, rightZ: u32,
    leftRef: u32, rightRef: u32,            // the stack's own reference: index bits 0-27, count bits 28-31
}
```

- **References.** The reference already used on the stack (`makeReference`, `src/trace/layout.ts:331-345`) fits a child in one word: indices fit in 28 bits (`GPU_LAYOUT_MAX_TRIANGLES`, `gpu-layout.ts:157`), and the count fits in 4. The empty right child becomes `0xffffffff`. That is also the instance marker, but the walk tests for an empty child before it pushes, as it does today (`wgsl.ts:307-311`).
- **The grid.** Each mesh gets a frame: a power-of-two `step` per axis and an `origin` that is a multiple of it. A plane decodes as `origin + f32(q) * step`. The CPU chooses `step` as the larger of two values: the smallest power of two that covers the mesh's extent in 65,535 steps, and the f32 ulp of the mesh's largest coordinate. Then `f32(q)`, the product and the sum are all exact, so the decoded plane is a real `f32`, and a fused or unfused decode gives the same bits. The slab test that follows is today's `bvhBoxEntry` (`wgsl.ts:771-779`), with the same exit padding. For a 20 m mesh the step is 2^-11 m, about 0.5 mm.
- **Where the frame lives.** In the node buffer, in a slot just before the mesh's root (`rootNodeOffset - 1`). The walk reads it once per instance entry, as indexed leaves read their mesh header (`bvhMeshVertexBase`, `wgsl.ts:479`). The frame can be derived from the mesh bounds by a fixed rule, so a serialized tree needs no new field (section 4).
- **Rounding outward.** The encoder rounds a min down to a grid point and a max up to one. It then decodes the plane in `f32` and requires it to lie at least 1/1024 of a step outside the exact box (today's boxes are exact, recomputed from the triangles: DIVERGENCES W2). Otherwise it moves the plane one more step. B3 learned this rule: with tight planes (rounded to the next step and no further), 1.7% of edge-aimed rays slipped through on the Intel UHD. A 1/1024-step margin fixed it on both GPUs (ddgi `docs/NOTES.md`, "Quantized child boxes need a margin"). Exact decoding should make the margin unnecessary, but the Metal backend compiles with relaxed math (DIVERGENCES T1), and the margin costs nothing measurable.
- **Watertightness lives in the triangle test, not in the boxes.** The walk only needs each box to contain its subtree, so that no leaf a ray hits is skipped. Whether the ray hits a triangle, and whether it slips between two, is decided by the watertight test on the unchanged `f32` corners (`wgsl.ts:812-880`). A looser box costs node visits, never hits. A box that rounds inward loses hits, so the layout check must verify every decoded plane against the exact box.

**A' (per-node frame, 8-bit).** Its loss is relative to each node's own size, which is better at the deepest levels. But the frame (12 + 3 bytes) and the two references make the node 36 bytes, which does not fit a 32-byte, two-`vec4` load, and padding it to 48 bytes saves only a quarter. Not recommended.

**B (CWBVH, wide).** Re-read: B3 was not slow because of box decoding. Decoding by bits instead of by conversion moved it only 3-8%. Its speed stayed flat from 1264 to 1530 MHz while BVH2's rose, which points to dependent memory loads. Bistro interior (1,186 small meshes) did worse, probably because every instance entry decodes a whole wide root. Does memory change the verdict? Partly. The bar becomes "not slower than A", not "1.3x faster". B saves 52 MiB more than A on Sponza. In exchange it brings a second walk, a wide refit (the B1a climb is binary: `src/gpu-refit.ts:363-436`), a GPU collapse pass after H-PLOC, and the 0.82x on many-mesh scenes. **Keep B dropped.** Revisit it only if A ships and memory is still the problem. The code is on ddgi branch `bvh-b3` (bb86f7c). B' has every cost of B and saves less.

**C (`f16`).** It is the simplest option: `unpack2x16float` is core WGSL (the optional `shader-f16` feature isn't needed), the decode is exact, and no frame is needed. But its step grows with distance from the origin, not with node size. At 16-32 units from the origin it is 1/64 of a unit (15.6 mm in metres), and values above 65,504 overflow to infinity. Small triangles away from the origin get boxes many times their size. Merged groups (world space, `trace.md`, "Precision") are the worst case. Not recommended, though it is a cheap point to include in the stage 1 SAH measurement.

## 2. Refit

The B1a climb hands boxes between threads through the node's own words, as 32-bit atomics with an "unwritten" mark (`bvhSlotWord`, `bvhStoreSlot`, `bvhLoadSlot`, `gpu-refit.ts:201-263`; clear at `:336-360`). Packed 16-bit planes can't take part in that handover.

**Decision: the climb doesn't change. Deforming meshes keep an `f32` shadow tree, and a new pass quantizes it.**

- `GpuMeshRefit` holds a 64-byte copy of the mesh's nodes in its own buffers. The clear, climb and repair passes run on it unchanged (DIVERGENCES S3 stays as it is).
- A fourth dispatch, `bvhRefitQuantize`, runs one thread per node after the repair pass, in the same compute pass, so it is ordered after it. It reads the new root box from the stats words `bvhFinishRoot` writes (`gpu-refit.ts:308-318`), derives the frame, writes the frame slot, and re-quantizes every node into the scene's buffer. The frame follows the pose: a mesh that grows beyond its bind pose stays correct.
- Cost: 64 B per node of GPU memory, for deforming meshes only (the largest Sponza mesh has about 115k nodes, so about 7 MiB), and one cheap pass, about the size of the clear pass.
- **CPU:** `refitOnCpu` (`refit-cpu.ts:197`) refits the 64-byte CPU copy as today; the same encoder as the builders' packs it before the upload.
- **The loss doesn't accumulate.** Each refit computes exact boxes from the current pose and quantizes them once. Quantized planes never feed the next refit. Even if they did, a union of grid-aligned planes is grid-aligned, so re-quantizing it is exact on A's per-mesh grid. That is not true of A' or B, whose per-node frames would round again at every level. `relativeSahCost` (`refit-cpu.ts:167`) keeps measuring the exact boxes. The refit's rebuild signal is unchanged, and slightly optimistic.
- **The top level** is refit every frame on these same atomics (`GpuTopLevel`). See section 4.

## 3. Traversal quality

Each decoded plane lies 0 to just over 1 step outside the exact one. That matters only for nodes a few hundred steps across or less, which are the leaves' parents. For example, a leaf 1/500 of a 20 m mesh is about 80 steps wide, so it can grow about 2.5% per axis. My estimate is 1-5% more box hits and triangle tests on object-space meshes, and more on large merged groups. Against that, each node fetch is 32 bytes instead of 64, and B3 found the walk bound by memory loads. **Expected net: 0.95-1.05x closest-hit, to be measured.** B3 can't calibrate this, because it recorded no visit counts (only `children_per_node` 5.94, `triangles_per_leaf` 2.19 and `requantized_slots`; ddgi `bench/results/trace.csv`, `ours-cwbvh/*`). The literature measured 8-bit, per-node quantization inside an 8-wide walk, not this format.

Measure it in three steps, the first without a GPU:

1. **CPU, offline:** the SAH cost of the decoded tree over the exact one (`relativeSahCost` on decoded boxes), and the CPU walk's `nodesVisited` and `trianglesTested` (`CpuTraceStats`, `cpu-traversal.ts:71-75`) on the bench's rays. Sponza and Bistro exterior, instanced and merged, A and C. If the merged Bistro tree's SAH cost rises by more than about 5%, stop and see open question 3.
2. **Counted walk:** `steps`, `meshNodes` and `triangleTests` per ray (`tests/gpu/pages/trace-bench/counted-walk-wgsl.ts`). The packed walk must keep the anchor line `node = bvhLoadMeshNode(instance.rootNodeOffset + index);` (`:89`) so the counted copy still applies.
3. **The paired surface-ray bench:** `trace-bench-surface-sponza` and `-bistro-interior`. Packed over `f32` in alternating order, with the ratio taken within each round (as `hitRecordRoundOrder` does). Closest and any hit; instanced and merged; `surface` and `air-grid`; records and indexed. The hits must be identical to the `f32` walk's, bit for bit, or the page fails. Run on the M5, then the RTX 3060, with Intel for correctness.

## 4. Interactions

- **Indexed leaves.** The two are independent: `nodeFormat` only changes `bvhLoadMeshNode` (`wgsl.ts:596-608`) and adds a frame read at instance entry. `triangleLeaves` only changes `bvhLoadTriangle`. Combined, beyond geometry: Sponza goes from 448.1 to **225.4 MiB** (indexed `vec4`, 318.0, less 92.6), or 214.1 with packed positions. Bistro exterior goes from 210.1 to **115.3** (107.3 packed). The memory.md estimate of "about 115 MiB" for Sponza assumed 4-byte ids. With today's 16-byte indexed triangles, the figure is 225 (three-mesh-bvh defaults: 51.2).
- **Merged static groups.** A group is one mesh with one frame covering the whole group, so its step is the scene's extent / 65,535 (Sponza about 0.6 mm if the scene is about 40 m). That is the case most at risk (step 1 measures it). The merged precision note (`trace.md`, "Precision") already says to keep merged geometry near the origin.
- **The top level stays `f32`.** It is 17 KiB to 0.3 MiB, `GpuTopLevel` refits it every frame through the atomic handover, and the walk reads it with a separate loader (`bvhLoadTopLevelNode`, `wgsl.ts:627-640`). Compressing it saves nothing worth a quantize pass every frame.
- **`CpuMeshBvh`.** CPU trees stay 64 bytes, as they stay on records for indexed leaves (`cpu-mesh-bvh.ts:396-398`). A packed tree is decoded on construction. Each plane is decoded exactly (`origin + q * step`, exact by construction, so `f32` and `f64` agree), so `traverse` (`:608-640`) hands `intersectsBounds` a box that is looser than today's but always contains its subtree. `BVH_CONTAINED` stays correct, because a shape that contains the looser box contains the triangles. Decoding before the walk means the callbacks see plain `Float32Array` boxes, as they do today.
- **Serialization.** Header word 7 (`serialize.ts:8-18`, `:119-121`) becomes a bit field: bits 0-7 hold the triangle format and bits 8-15 hold the node format (0 for `f32`, 1 for packed). A reader that predates this sees 256 or 257 and refuses with its existing message instead of misreading the nodes. Word 14 would also work: old readers ignore it, but their byte-length check (`:125`) would then refuse the file with a misleading error. The frame comes from the bounds in words 8-13 (`nodeFrameOf(bounds)`), and that rule becomes part of the format. It stays a variant of `GPU_LAYOUT_VERSION` 2, like indexed leaves.
- **Builders.** In stage 1, every builder writes 64-byte nodes and packing comes afterwards. tinybvh's trees are packed on the way out of the WASM (as `indexedTrianglesFromRecordWords` does), before the worker transfers them. GPU trees are packed by a compute pass that replaces the node `copyBufferToBuffer` into scene storage (`scene-trees.ts:1690`). As with indexed leaves (`:663`), `adopt` is refused. `checkGpuLayout` and `layout-check.ts` must check that every decoded plane contains the exact child box.
- **Build peak.** Batch outputs stay 64 B per node (`outNodes`, `gpu-hploc.ts:168`), so the bytes-per-triangle constants and `gpuMaxBatchBytes` planning don't change. Settled storage drops by the node saving, and so does the capped peak, whose floor is "the storage plus one batch" (memory.md, "The lever, measured"). H-PLOC writing packed nodes directly is a later step. Its leaf collapse knows each mesh's box only after the climb.

## 5. Rollout

`f32` nodes stay a setting beside packed nodes (Dennis's rule): `nodeFormat: 'f32' | 'packed'` on `traceModuleWgsl`, `batchKernelWgsl`, `assembleTraceScene`/`buildTraceScene`, `SceneTrees`, and `BvhTraceNodes` through its source. The default stays `'f32'`.

1. **CPU:** the frame rule, the encoder and decoder, the CPU walk, `layout-check`, and the offline SAH and `nodesVisited` measurement (section 3, step 1). Stop here if merged Bistro loses more than about 5%.
2. **WGSL:** `bvhLoadMeshNode` for packed nodes, the frame read, `uploadTraceScene`, oracle variants `packed-*` (edge rays included: B3's leak class), the counted walk, and the paired bench on the M5 and the RTX 3060.
3. **Builders and `SceneTrees`:** tinybvh output packing, the GPU pack-on-copy pass, the frame slots, growable storage at 32 bytes per node.
4. **Refit:** the shadow tree and `bvhRefitQuantize` in `GpuMeshRefit`, the CPU path, `scene-trees-deforming` and `bvh-blas-refit` with packed nodes.
5. **`CpuMeshBvh`, `/three`'s `BvhBoundsTree`, serialization.**
6. **The default** is decided only after the RTX 3060 numbers from stage 2.

**Hard to undo after the release:** the bit field in header word 7 and the frame rule in files; the frame slot before each mesh root (anything that reads `rootNodeOffset` or `buffers.nodes` with a 64-byte stride); and custom WGSL that reads `bvhTraceNodes` directly, if packed becomes the default.

## 6. Open questions for Dennis

1. A (32 bytes, saves 92.6 MiB on Sponza) first, with CWBVH kept dropped? Or does memory now justify re-running B3's walk against A?
2. What closest-hit cost on the RTX 3060 is about 93 MiB on Sponza (40 MiB on Bistro) worth before packed becomes the default? Is the answer different with indexed leaves on?
3. If merged groups lose too much at one frame per group, should they stay on `f32` nodes? That needs a per-instance bit (bits 4-7 are reserved, `layout.ts:202`) and a branch in the walk, which indexed leaves avoided. Or should we accept the loss?
4. Header word 7 as a bit field, or a separate word?
5. Should the deforming refit's 64-byte shadow tree be acceptable, or should the climb learn to hand boxes over in scratch?
6. Should the folded decode (`fma(q, step * invDir, (origin - rayOrigin) * invDir)`, fewer operations but no exact plane) be tried after stage 2, or ruled out to keep the exact-plane argument?

## Measured (2026-11-12, Apple M5, Chrome 157 headless)

**Correctness** (the gates):

- `trace-oracle`: 8 quantized variants (tinybvh, masks compiled out, the GPU LBVH and H-PLOC packed by the GPU quantize pass and `checkGpuLayout`-checked, with indexed leaves, merged tinybvh, merged H-PLOC, merged H-PLOC indexed packed) and 21 quantized hook passes: 0 mismatches, 0 of 2M shared-edge rays slipped (B3's leak class), split seams hold. On the one-ulp seams quantized lets **fewer** rays through than `f32` (10,927 against 11,287 of 200k on the plain seam, the same 9,913 on the transformed one): a looser box takes rays an exact box's slab test rounds away. The page checks "never more". `three-mesh-bvh`-built trees already differ from tinybvh's there (11,034), so the count depends on boxes, not only on the triangle test.
- `trace-inline` (quantized, quantized indexed), `scene-trees-mid-swap` (4 quantized variants: copy, merged, capped, merged capped indexed packed; 132 states, 0 disagreements), `scene-trees-deforming` and `three-trace` (quantized, records and indexed), `scene-trees-sponza`: all pass.
- `bvh-blas-refit`: GPU refits of quantized trees (grid, soup, one triangle; 5 poses each) are, word for word, the CPU refit packed by `quantizeNodeWords`, and pass the layout check against each pose; a quantized indexed scene refit through `meshRefitPlacement` hits what a scene built in the pose hits.
- Unit tests: the frame rule, outward rounding with the margin, every builder's quantized tree passing the layout check (meshes far from the origin, tiny, huge, flat, one triangle), serialization, `CpuMeshBvh` boxes around the exact ones, the CPU refit plus re-quantize, `SceneTrees` (fake GPU) through GPU-first, swaps, merged groups, runtime changes and capped batches, and quantized scenes giving the `f32` scene's instance, triangle and `t` bit for bit on 4,000 rays (instanced and merged, records and indexed).
- The surface-ray bench: every quantized answer has the `f32` answer's `t` bit for bit; 3 to 8 rays per set (of about 230k) name another triangle at exactly that `t` (ties at shared edges, counted apart; the walk keeps the last hit at equal `t`, and looser boxes can change the order). Any-hit is identical on every ray.

**Quality** (`tools/quantized-nodes.ts`, CPU walk, no GPU; air-grid rays about 43k per scene, and secondary rays from their hits):

| | SAH quantized / `f32` | Node visits per ray | Triangle tests per ray |
| --- | ---: | ---: | ---: |
| Sponza instanced | 1.0012 | +0.0 to +0.1% | +0.1 to +0.9% |
| Sponza merged (one 5.7M-triangle group, one frame) | 1.0066 | +0.1 to +0.5% | +0.6 to +2.6% |
| Bistro exterior instanced | 1.0005 | +0.0 to +0.1% | -0.1 to +0.5% |
| Bistro exterior merged (2.8M, one frame) | 1.0036 | +0.1 to +0.3% | +0.0 to +0.8% |
| Bistro interior instanced / merged | 1.0001 / 1.0011 | +0.0 to +0.2% | -0.2 to +0.4% |

The merged groups lose more than per-mesh trees, as section 3 expected, but far below the ~5% that would call for a per-instance `f32` fallback (question 3): one frame per group stays. The counted GPU walk agrees: steps per ray +0.0 to +0.3% instanced, +0.1 to +1.2% merged (Sponza merged surface closest: 70.27 to 71.09 steps, triangle tests 7.17 to 7.65); Bistro interior +0.0 to +0.3%.

**Speed** (indicative only: one run of 5 paired rounds per scene, `f32` and quantized back to back with the order reversed every other round; ratio = quantized rate over `f32` rate, above 1 faster; median of rounds and range):

| | instanced surface | instanced air grid | merged surface | merged air grid |
| --- | --- | --- | --- | --- |
| Sponza closest | 0.944 (0.87-0.99) | 0.965 (0.96-0.99) | 0.930 (0.89-1.02) | 0.953 (0.95-0.96) |
| Sponza any | 0.941 (0.87-0.98) | 0.955 (0.94-0.99) | 0.927 (0.92-0.93) | 0.957 (0.92-0.98) |
| Bistro interior closest | 0.969 (0.92-0.97) | 0.969 (0.97-0.98) | 0.966 (0.93-0.97) | 0.963 (0.95-0.97) |
| Bistro interior any | 0.964 (0.96-1.00) | 0.958 (0.96-0.97) | 0.962 (0.96-0.97) | 0.957 (0.95-0.96) |

On the M5, quantized nodes cost 3-7%: with visits nearly unchanged, that is the decode (12 extracts, conversions and FMAs per node) not paid back by the halved node fetch on this GPU's unified memory. Section 3's 0.95-1.05x estimate holds at its low end. The RTX 3060, where B3 found the walk bound by memory loads, decides the default (question 2). Rows: `bench/results/trace.csv`, metrics `surface-rays/paired/quantized_over_f32/...` and `surface-rays/node_format/...`, commit `13e9066`.

**Bytes**: [memory.md](memory.md), "Quantized nodes". Beyond geometry, Sponza 448.1 to 355.5 MiB quantized and **225.5** with indexed leaves; Bistro exterior 210.1 to 169.9 and **115.3**. `SceneTrees` on Sponza (`scene-trees-sponza`, default 256 MiB cap, settled): `f32` 458.6 MiB, peak 711; quantized 360.8, peak 613.2; indexed 327.1, peak 580; quantized and indexed **230.0, peak 483**. tinybvh trees take longer to arrive with quantized nodes (all placed at 439 ms against 351; 512 with indexed leaves against 431): the workers pack 3M nodes. First GPU trees 139 ms against 130. Merged, uncapped: settled 366.7 MiB, the peak (1,482.6) is the one big batch, unchanged.

## Left to do

- The RTX 3060 and Intel: the oracle pages and the paired bench (`BVH_PAGE_QUERY='skip=hit-record,leaves' pnpm test:gpu:page trace-bench-surface-sponza`). The default stays `f32` until then.
- H-PLOC writing quantized nodes in its emit (it packs after the build; the batch holds both node buffers for a moment).
- The folded decode (`fma(q, step * invDir, ...)`), as a measured option.
- Packing tinybvh trees inside the WASM instead of in JavaScript after it (the worker cost above).


# Indexed leaves (P9 lever 2)

## Status (2026-11-13): built, off by default

Dennis's answers to section 7: option B (16-byte record, our own positions); `vec4f` positions by default with packed `f32` as a compile-time variant; merged groups indexed with a side table; a variant flag inside `GPU_LAYOUT_VERSION` 2; an exact-bit weld at load, on by default, with an opt-in tolerance weld. After the RTX 3060 run (2026-11-13, "Measured" below) records stay the default: indexed leaves wait for an RTX 3060 run of quantized nodes and indexed leaves together, now that quantized nodes are the default ([compressed-nodes.md](compressed-nodes.md)). He accepted both as-built departures from section 1 below (the vertex base in a per-mesh header, and the 65,536-source cap); [DIVERGENCES.md](../DIVERGENCES.md) rows A1 and A2.

**What is in.** One walk, with the triangle source compiled in as `rayMasks` is:

- `assembleTraceScene` with no `triangleLeaves` takes its trees' format (2026-11-13): indexed trees give an indexed scene (it needs `geometries`), records trees a records one, and a mix throws, naming `triangleLeaves: 'indexed'` (which converts the records trees).
- `triangleLeaves: 'records' | 'indexed'` and `positionLayout: 'vec4' | 'packed'` on `traceModuleWgsl`, `batchKernelWgsl`, `TraceBatchKernel`, `assembleTraceScene`/`buildTraceScene`, `SceneTrees`; `BvhTraceNodes` reads them from its source. Only `bvhLoadTriangle` (and the merged-hit lookups) differ; the walk's text is identical (a unit test compares it).
- The record: `vec4u(i0, i1, i2, id | filter << 28 | flip << 29)` (`GPU_LAYOUT_INDEXED_TRIANGLE_WORD`). Vertex indices stay **mesh-relative in the scene too**: each mesh's region starts with a 16-byte header (`GPU_LAYOUT_MESH_HEADER_WORD`: vertex base, and for a merged group its side and source table distances) at `triangleOffset - 1`, which the walk reads once per instance entered. This differs from section 1 ("the scene adds the vertex base"): records written by any builder are placed and moved (growable storage, swaps, GPU copies) without rewriting them, and the cost is 16 bytes per mesh and one load per instance entry. This is the format (accepted 2026-11-13).
- Merged groups: word 3 holds the source triangle, filter and flip; a 4-byte side table per triangle (source slot, material code) and a source table (slot to `instanceIndex`) sit before the header. The plan caps a group at 65,536 sources (`MAX_MERGED_SOURCES`, the side table's 16-bit source slot; accepted 2026-11-13): more static instances make more groups.
- Per-triangle materials (`MeshInput.triangleMaterials`): word 3 has no room (bits 30-31 stay reserved), so an unmerged mesh that has them gets a material table before its header: each original triangle's code (material + 1, 0 for none), 16 bits, by original index (the walk has it from word 3), eight to a unit; header word 3 (`materialTableDistance`) is its distance, 0 for none. By original index, so the scene writes it from the geometry whatever builder made the tree, GPU trees included. A merged triangle's code is in its side table (its source triangle's, else its source instance's).
- Builders: tinybvh (since 2026-11-14 written inside the WASM, `tbvh_write_layout_as`, so the 48-byte records are never allocated; an older `.wasm` converts them on the way out, `indexedTrianglesFromRecordWords`), the topology builder, the LBVH and H-PLOC CPU twins, the GPU H-PLOC (`lbvhWriteTriangle`, `LbvhParams` grew to 48 bytes), and since 2026-11-14 the GPU LBVH: its emit took leaf boxes from the records and has no binding left for positions or primitive boxes (8 per stage), so in indexed batches `lbvhLeafBoxes` writes each leaf's box into the leaf's own child slot of its parent in the climb's scratch (a slot the climb never uses), which the emit already binds ([bvh-builders.md](bvh-builders.md), "GPU LBVH"); `createMeshLayoutBuilder` no longer converts its trees.
- `checkGpuLayout` checks the indices and word 3, then the rest on the same tree with its corners copied out. Serialization writes the format in header word 7 (0 records, 1 indexed; old files have 0); a reader refuses another value. `CpuMeshBvh` takes an indexed tree with its `source` and by default expands it to records in the constructor (CPU trees 48 bytes per triangle); since 2026-11-14 `corners: 'positions'` keeps the 16-byte indexed triangles and reads every corner through the source's own `Float32Array` positions, with no copy (section 5's plan): every query and refit bit for bit the records path's, 32 bytes less per triangle (175.3 MiB for Sponza's 5.74M triangles), query times level within a loaded CPU's noise, refits 0.76-0.95x (`bench/cpu-corners.ts`, indicative). A refit then writes no triangles: it recomputes the boxes from the positions. Since 2026-11-15 `/three`'s `BvhBoundsTree` passes `corners: 'positions'` for an indexed tree (a `layout` option or file in that format, `deserializeBoundsTree` / `MeshBVH.deserialize`), so it stays indexed with no records expansion, and its shifted trees (draw ranges and groups off a multiple of 3) are built indexed too; a records tree (what `computeBoundsTree` builds) stays records. Nothing in `/three` reads the records directly (shapecast triangles, `getTriangleHitPointInfo`, the helper, `StaticGeometryGenerator`, three-mesh-bvh's format and refit go through `CpuMeshBvh`), so nothing converts lazily. `src/three/indexed-bounds-tree.test.ts` checks the indexed twin of a records tree gives its answers exactly. One difference: an indexed tree reads the geometry's live positions (as three-mesh-bvh does), so a query between an edit and `refit()` sees the new corners in the old boxes, where a records tree sees its old copy; `acceleratedRaycast` falls back to three's raycast in that window either way.
- Weld: `weldExactBits` (on in `assembleTraceScene`/`SceneTrees`, `weld: false` turns it off; never on deforming meshes, whose poses are copied vertex for vertex), `weldWithinTolerance`/`weldPositions` (`buildTraceScene({ weldTolerance })`, opt-in, before the build, either format).
- Deforming meshes: `GpuMeshRefit` copies the pose into the mesh's range of the scene's positions (`meshRefitCopyVertices`) and refits the boxes from there; the triangles are not written.
- `SceneTrees`: positions storage (growable, a range per mesh, placed when its builds start), headers and side tables, `buffers.positions`, `memory.positions`. Since 2026-11-14 indexed scenes adopt too (`gpuTreeOutputs: 'adopt'`): their GPU builds leave each mesh's leading units free before its triangles (`GpuTreeRequest.triangleLeading`, the builders' per-mesh `outTriangleStart`), the scene writes the headers (and merged groups' side tables, through the identity pass) into them at the adoption, and a batch whose builder ignored the gaps is copied instead. With quantized nodes, adoption takes batches whose nodes H-PLOC wrote quantized (`gpuQuantizeOnEmit`, [compressed-nodes.md](compressed-nodes.md)).
- **The app's own positions** (option A's binding, section 3; 2026-11-14): `SceneTrees({ positionsBuffer })` binds the app's buffer as `buffers.positions` and makes no copy (45.2 MiB less on Sponza, `vec4`). It must hold every mesh's vertices as `f32` in the walk's stride, each mesh from the `vertexBase` given to `addMesh`, with the geometry's exact bits, and `STORAGE` usage. The weld rules decide the rest: the trees must name the app's own vertices, so the scene doesn't weld (an explicit `weld: true` throws); merged groups' world-space vertices are not in the app's buffer, so static instances don't merge (an explicit `mergeStaticInstances: true` throws); and a deforming mesh's refit writes its pose into the scene's copy ("It must be a copy", section 5), so deforming meshes throw. three.js apps can't use it (section 3).

**Measured** on the Apple M5 (2026-11-12) and the RTX 3060 Laptop (2026-11-13): see "Measured" at the end.

---

Design (2026-11-11), kept as written. Today each tree-ordered triangle is a 48-byte record: three copied corners, the original index, flags and padding (`src/gpu-layout.ts:72-89`, written by `packTriangles`, `src/trace/triangle-packing.ts:74-95`). On Sponza the records are 263.0 of the 448.1 MiB we add beyond the app's geometry ([memory.md](memory.md)).

The risk to beat: "reading corners through an index buffer cost 10-30% of trace speed" ([trace.md](trace.md):40, DIVERGENCES W4). That read went record, then the app's index, then positions: three dependent loads. The format below needs two.

## 1. Record format

Counts from memory.md: Sponza has 5,744,384 triangles and 3,108,999 vertices; Bistro exterior has 2,829,226 and 2,920,632.

| Option | Per triangle | Corners from | Sponza MiB | Bistro MiB |
| --- | --- | --- | ---: | ---: |
| Today | 48 B | the record | 263.0 | 129.5 |
| **A.** `u32` id + flags | 4 B | the app's index and positions, bound | 21.9 | 10.8 |
| **B.** `vec4u(i0, i1, i2, id + flags)`, tree order | 16 B | our `f32` copy, 12 B/vertex | 87.7 + 35.6 = **123.2** | 43.2 + 33.4 = **76.6** |
| B, `vec4f` positions | 16 B | our copy, 16 B/vertex | 87.7 + 47.4 = 135.1 | 43.2 + 44.6 = 87.7 |
| C. Reorder the app's index in place | 0 B | the app's | 0 | 0 |

- **A saves the most** (241.1 / 118.7 MiB), but only if the walk can bind the app's buffers (section 3). If we have to copy the index, A costs 4 + 12 B per triangle, the same as B, plus a third dependent load.
- **B saves 139.7 / 52.9 MiB.** Beyond geometry, Sponza goes from 448.1 to 308.4 MiB and Bistro from 210.1 to 157.2. One `vec4u` load gives the vertex indices and the id.
- **C** rewrites the app's geometry and leaves no room for flags or the original index. Rejected.

**Recommendation: B.** Word 3 holds:

- bits 0-27: the original index (it fits: `GPU_LAYOUT_MAX_TRIANGLES` is `0x0fffffff`, `gpu-layout.ts:95`);
- bit 28: the filter flag;
- bit 29: the flip flag;
- bits 30-31: reserved.

Material bits move to a side table, used only by merged meshes (section 5). Builders write vertex indices relative to the mesh. The scene adds the mesh's vertex base when it places the tree, because the instance record has no free word for it. (As built, the scene does not add it: the vertex base sits in a 16-byte header before each mesh's triangles, read once per instance entered. See Status.)

## 2. Watertightness

The test needs identical coordinate *bits*, not shared indices. `bvhEdgeFunction` orders an edge's corners by value (`src/trace/wgsl.ts:680-687`), so two triangles that see the same bits run the same arithmetic, whichever vertex each one names.

- **Today, on a UV seam:** the two triangles name different vertices, and `packTriangles` copies whatever bits each one holds. Exporters normally duplicate the bits exactly, so the seam is watertight. If the copies differ by one ulp, rays leak there today, and records don't help.
- **Indexed:** the walk reads the same bits, so it behaves identically on seams and on non-indexed geometry.
- **The real risks:**
  - **Decoding in the walk.** A `float16`, normalized or scaled position would be decoded once per use. Each inlined decode may be fused differently, and the CPU twin and the builders' boxes could round differently from the GPU; a box that doesn't contain its triangle loses hits. **The walk reads `f32` only; decoding happens at load.**
  - **A skipped deforming refit** (section 5).
- **The oracle doesn't cover seams.** `tessellatedQuad` (`src/trace/oracle-fixtures.ts:58-89`) shares vertices by index only. Add two fixtures:
  - **a split seam** (every corner duplicated): 0 leaks in both modes;
  - **a one-ulp seam:** it leaks in both modes, which documents the limit and tests the weld.
- **A welded index at load.** `SceneTrees` can weld its copy by exact bits. That is safe, because no arithmetic changes, and it saves memory: Bistro has 1.03 vertices per triangle, Sponza 0.54. A tolerance weld would close one-ulp seams but move geometry away from what the app draws, so it would be opt-in only.

## 3. Where positions come from

- **The walk binds one positions buffer for the scene**, so its stride is fixed at compile time. Apps keep each mesh in its own buffer and format, so binding app buffers (A) works only for an app that already keeps one shared `f32` storage buffer. That is a later variant.
- **Recommended: `SceneTrees` owns a packed `f32` copy**, 12 B per vertex against 48 B per triangle.
  - It already receives CPU `MeshGeometry` (`src/scene/scene-trees.ts:687`).
  - The H-PLOC and LBVH batches already upload exactly this buffer as an input (`lbvhCorner`, `src/gpu-lbvh-wgsl.ts:391-394`). In `adopt` mode that input becomes the scene's positions, with no second copy.
  - `buffers` gains `positions` (`scene-trees.ts:650-651`), growable like the others.
- **Formats:**
  - `float16`, normalized and interleaved positions are decoded on the CPU at load.
  - `GpuMeshBuffers` input (`src/mesh-input.ts:54-70`: `f32`, a stride and offset in floats, `uint16` or `uint32` indices) is copied into the scene buffer by a compute pass.
- **three: we can't bind its attribute buffers.**
  - Plain attributes have no `STORAGE` usage (three 0.187, `WebGPUBackend.js:2928`), and indices get it only as a `StorageBufferAttribute` (`:2909-2913`).
  - three pads or widens some attributes (`WebGPUAttributeUtils.js:118-198`).
  - Each attribute is its own buffer.
  - It would add a fourth internal to `threeInternals` (`src/three/trace-nodes.ts:109-125`).
  - `src/three/scene.ts` already compacts positions to `f32` (`positionsOf`, `:141`, cached at `:201-215`), so three apps get the `SceneTrees` copy without extra work.

## 4. The walk

- **One walk:** `traceModuleWgsl({ triangleLeaves: 'records' | 'indexed' })`, compile-time like `rayMasks`.
  - Only `bvhLoadTriangle` changes (`wgsl.ts:476-482`). It still returns `BvhTriangle` (`wgsl.ts:149-153`), so the triangle test, the filter, the merged logic and `bvhFinishClosest` stay the same.
  - Indexed mode adds a fourth binding to `TRACE_MODULE_ENTRIES` (`wgsl.ts:80-87`), within the default 8 storage buffers per stage.
  - `BvhTraceNodes` strips exactly three bindings today (`trace-nodes.ts:177-185`), so it has to learn the fourth.
  - The CPU walk follows (`src/trace/cpu-traversal.ts:295-307`, `:366-372`, `:398-414`).
- **Loads per triangle test:**
  - **today:** 3 contiguous `vec4f`;
  - **indexed:** 1 `vec4u`, then 3 dependent position reads (9 scalar loads packed, 3 as `vec4f`).
- **Expected cost:**
  - **Against it:** a deeper chain and scattered reads.
  - **For it:** a third of the bytes per triangle, and leaf neighbours share cached vertices.
  - **Estimate:** anywhere from a small gain to the old 10-30% loss.
- **Measuring:** the `trace-bench-surface-sponza` and `-bistro-interior` pages, records against indexed in paired rounds with the order alternating (as `hitRecordRoundOrder` does), the ratio taken within each round.
  - Cases: closest and any hit; instanced and merged trees; `surface` and `air-grid` rays; both position strides.
  - Hardware: the M5 and the RTX 3060, plus Intel for correctness.
  - Hits must be identical in both modes.

## 5. Interactions

- **Merged static instances.**
  - **Today:** records hold world-space corners, and assembly overwrites `originalIndex`, `padding` and flags (`src/trace/static-merge.ts:291-310`).
  - **Indexed:** `mergeStaticGeometry` already makes world positions and indices per group (`:198-219`; each vertex is transformed once, `:261-264`). They become the group's range in the positions buffer, and word 3 carries the source triangle, filter and flip.
  - **Source and material:** a 4 B hit-time side table per merged triangle (source slot and material), plus a small table from slot to `instanceIndex`.
  - **Cost:** 20 B per triangle plus 12 B per world vertex.
  - **Not recommended:** keeping merged groups on records, which would need a per-instance mode bit (bits 4-7 are reserved, `src/gpu-layout.ts`, `INSTANCE_FLAG`) and a runtime branch in the leaf loop.
- **Deforming meshes and refit.**
  - The triangle step (`src/gpu-mesh-refit.ts:9-16`) stops writing 36 B of corners per triangle. It copies the posed vertices into the mesh's position range (12 B per vertex) and writes the boxes as before.
  - **The mesh must not be welded.** The copy writes the app's vertex v to scene vertex `vertexBase + v`. `meshRefitPlacement` returns the mesh's `vertexBase`, `vertexCount` and the positions layout for `setTree`, and throws for a mesh the scene welded (the exact-bit weld, on by default, or `weldTolerance`): assemble meshes you will refit with `weld: false`. `setTree` requires `vertexBase` for an indexed tree (0 for a tree in its own buffers), and `encode` refuses a pose whose vertex count differs from the placement's.
  - **It must be a copy, not the app's posed buffer:** when `setPose` skips a refit (`scene-trees.ts:704-719`), rays must keep the pose the boxes were built for.
  - `refitOnCpu` and `CpuMeshBvh.refit` (`src/query/cpu-mesh-bvh.ts:610-628`) write positions instead of corners.
- **`CpuMeshBvh`.**
  - The depth-first reorder (`cpu-mesh-bvh.ts:27-41`) moves 16-byte records instead of 48-byte ones.
  - Positions are the source's own `Float32Array`, with no CPU copy, so three's `BvhBoundsTree` costs 16 B per triangle beyond geometry.
- **The hit normal.** `bvhFinishClosest` reloads by slot (`wgsl.ts:621-623`): one indexed load per ray, with the same math.
- **The filter, masks and material bits.**
  - The filter and flip bits come in the same `vec4u`.
  - Masks are per instance and don't change.
  - A merged triangle's material comes from the side table, read once per hit.
  - The app's `triangleFlags` still sets only bit 0, which the packer moves to bit 28.
- **H-PLOC and LBVH.**
  - `lbvhWriteTriangle` (`gpu-lbvh-wgsl.ts:1079-1098`, also used by `hplocTrianglesKernel`, `gpu-hploc-wgsl.ts:1143-1162`) writes 4 words instead of 12.
  - The LBVH emit reads leaf boxes back from the records (`gpu-lbvh-wgsl.ts:956-968`), so it reads positions instead.
  - `outTriangles` (`gpu-hploc.ts:168`) shrinks from 263.0 to 87.7 MiB on Sponza: a batch goes from 270.6 to 238.6 B per triangle, so 13% more triangles fit under the 256 MiB cap.
  - tinybvh writes the layout inside the WASM (W4), so it needs a WASM change.
  - `layout-check.ts` must also check the vertex indices.
- **Serialization and `GPU_LAYOUT_VERSION`.**
  - **Recommendation:** a variant inside version 2, which is unreleased. Header word 7, unused today (`src/serialize.ts:8-18`), becomes the triangle format: 0 for records, 1 for indexed.
  - Readers refuse a format they don't know.
  - An indexed file has no positions, so loading it takes the same geometry, and the loader checks the vertex count.

## 6. Rollout

**Records stay as a setting beside indexed leaves**, as Dennis's rule asks: `triangleLeaves` on `traceModuleWgsl`, the builders and `SceneTrees`. The default stays records until the bench says otherwise.

1. **CPU:** the format constants, the packer, the CPU walk, `layout-check`, the seam fixtures. 0 leaks in both modes.
2. **WGSL:** the loader, the fourth binding, `BvhTraceNodes`, `uploadTraceScene`. Then the paired bench on the M5 and the RTX 3060.
3. **Builders:** tinybvh's WASM, the H-PLOC and LBVH writers, the LBVH emit box.
4. **`SceneTrees`:** the positions storage, the vertex base at placement, the deforming copy, merged groups with their side table.
5. **`CpuMeshBvh`, three's bounds tree, serialization.**
6. **The default:** decided from stage 2's numbers. The exact-bit weld comes after, once its saving is measured.

**Hard to undo after the release:**

- the public fourth binding and `SceneTrees.buffers.positions`;
- the format word in files;
- mesh-relative vertex indices and the per-mesh header in scene storage (as built; this section planned absolute indices);
- custom walks reading `buffers.triangles` as 48-byte records, if indexed becomes the default.

## 7. Questions for Dennis, answered

1. B first (our own `f32` copy, about 140 MiB saved on Sponza); A (binding the app's buffers) is kept for later.
2. `vec4f` positions by default; packed `f32` is a compile-time variant (`positionLayout: 'packed'`).
3. Merged groups go indexed, with the 4-byte side table.
4. A variant flag inside version 2 (header word 7).
5. The exact-bit weld is on by default; the tolerance weld is opt-in (`weldTolerance`).
6. Not yet (2026-11-13). On the RTX 3060, `vec4` cost 3-8% (0.92-0.97x) and packed 5-18% (0.82-0.95x), each timed alone. Records stay the default until a run of indexed leaves with quantized nodes, the new default, on the RTX 3060.
   **Records stay (2026-11-15), provisional until the M5 times the pair.** The RTX run of the pair (`WINDOWS-RUNBOOK.md`, sections 9 and 10): on quantized nodes `vec4` costs 5-17% of closest-hit rate (0.83-0.95x; packed 0.77-0.87x), more than on `f32` nodes, for 130 MB on Sponza (366.5 to 236.4 MB settled, uncapped). Indexed `vec4` is the memory option; packed only when its further 11 MB matter. The RTX 3060 and the M5 decide; the Intel UHD answers correctness only.

## Measured (2026-11-12, Apple M5, Chrome 157 headless)

**Correctness** (the gates):

- `trace-oracle`: 16 variants, 8 of them indexed (tinybvh `vec4` and packed, masks compiled out, tolerance weld, the GPU LBVH (converted) and H-PLOC (written on the GPU, `checkGpuLayout`-checked), merged tinybvh and merged H-PLOC packed), plus the hook cases indexed and merged-indexed: 0 mismatches, 0 rays slipped through the 2M shared-edge rays.
- Seam fixtures (every variant, 200k rays each, plain and transformed): **split seams** let 0 rays through in every format. **One-ulp seams** leak in every format, records and indexed alike: 11,287 and 9,913 of 200k for tinybvh, ray for ray the same in `watertight-tinybvh`, `indexed-tinybvh` and `indexed-tinybvh-packed` (the page checks the slipped rays match); the double-precision reference misses 1,717 and 2,223 of its 20k sample there too. **The tolerance weld closes them**: `indexed-tinybvh-weld` (1e-6, before the build) lets 0 through. This is the documented limit: indexed leaves are as watertight as records, no more; a seam whose copies differ needs the opt-in weld.
- `trace-inline` (records, indexed, indexed packed), `scene-trees-mid-swap` (4 indexed variants: copy, merged, capped and merged-capped with packed positions; 132 indexed states, 0 disagreements, 0 misses), `three-trace` (records then indexed through `BvhTraceNodes`), `scene-trees-deforming` (records then indexed), `bvh-blas-refit` (indexed refit, `vec4` and packed): all pass.
- The surface-ray bench checks every indexed answer against records' bit for bit (whole 40-byte hit record, any-hit word): identical on every ray of every case, both scenes.

**Speed** (indicative only: one run of 5 paired rounds per scene, the three formats back to back with the order reversed every other round; ratio = indexed rate over records rate, above 1 faster). Median of rounds (range):

| Sponza | `vec4` closest | `vec4` any | packed closest | packed any |
| --- | --- | --- | --- | --- |
| instanced, surface | 0.947 (0.80-1.04) | 0.964 (0.59-1.07) | 0.912 (0.52-1.10) | 0.918 (0.63-1.14) |
| instanced, air grid | 0.950 (0.92-1.28) | 0.974 (0.88-1.11) | 0.874 (0.82-1.50) | 0.921 (0.81-1.02) |
| merged, surface | 0.834 (0.77-0.92) | 0.978 (0.93-1.00) | 0.751 (0.69-0.92) | 0.924 (0.85-0.98) |
| merged, air grid | 0.925 (0.89-0.94) | 0.964 (0.91-0.98) | 0.881 (0.87-0.91) | 0.885 (0.87-0.96) |

| Bistro interior | `vec4` closest | `vec4` any | packed closest | packed any |
| --- | --- | --- | --- | --- |
| instanced, surface | 0.977 (0.88-1.04) | 0.974 (0.93-1.07) | 0.881 (0.72-0.96) | 0.912 (0.80-1.04) |
| instanced, air grid | 0.994 (0.98-1.04) | 0.971 (0.93-1.02) | 0.902 (0.87-0.93) | 0.868 (0.80-0.89) |
| merged, surface | 0.939 (0.89-1.03) | 0.979 (0.95-1.04) | 0.808 (0.69-0.96) | 0.879 (0.81-1.01) |
| merged, air grid | 0.993 (0.97-1.01) | 0.943 (0.92-0.97) | 0.889 (0.82-0.91) | 0.827 (0.80-0.85) |

Reading it, on the M5 only: `vec4` positions cost 1-6% on instanced trees and up to 17% on Sponza's merged tree with surface rays (the fastest case, so the extra dependent load shows most); packed `f32` costs another 5-10% everywhere, so `vec4` is the right default and packed is a memory option (11 MiB more saved on Sponza). Nowhere near the old 10-30% everywhere, but not free either. The Sponza instanced rounds swung widely (0.5-1.5), so its medians are the least sure. Rows: `bench/results/trace.csv`, metrics `surface-rays/paired/indexed_over_records/...` and `indexed_packed_over_records/...`, commit `8654783`.

**Speed on the RTX 3060 Laptop** (2026-11-13, commit `63af716`, Windows, headless Chrome 157; three runs of 5 paired rounds per scene; each figure is the median over the runs of each run's median of rounds; `f32` nodes throughout; [WINDOWS-RUNBOOK.md](../WINDOWS-RUNBOOK.md), section 7):

| Sponza | `vec4` closest | `vec4` any | packed closest | packed any |
| --- | --- | --- | --- | --- |
| instanced, surface | 0.949 | 0.967 | 0.911 | 0.940 |
| instanced, air grid | 0.956 | 0.974 | 0.932 | 0.945 |
| merged, surface | 0.938 | 0.949 | 0.895 | 0.915 |
| merged, air grid | 0.925 | 0.955 | 0.891 | 0.914 |

| Bistro interior | `vec4` closest | `vec4` any | packed closest | packed any |
| --- | --- | --- | --- | --- |
| instanced, surface | 0.966 | 0.933 | 0.881 | 0.841 |
| instanced, air grid | 0.967 | 0.928 | 0.883 | 0.843 |
| merged, surface | 0.947 | 0.920 | 0.863 | 0.821 |
| merged, air grid | 0.954 | 0.925 | 0.856 | 0.827 |

Every indexed answer matched records' on every ray. Reading it: `vec4` costs 3-8% everywhere on NVIDIA, steadier than the M5 (no 17% case), and packed costs 5-11% on Sponza and 12-18% on Bistro interior. Dennis kept records as the default (question 6): the memory case for indexed leaves is strongest with quantized nodes, and that pair has not been timed together. Rows: `bench/results/trace.csv`, scenes `sponza` and `bistro-interior`, device `RTX 3060 Laptop`, metrics `median/surface-rays/paired/indexed_over_records/...` and `indexed_packed_over_records/...`.

**Bytes**: [memory.md](memory.md), "Indexed leaves": `SceneTrees` on Sponza settles at 327.1 MiB indexed against 458.6 records, peak 580 against 711 MiB; Sponza 448.1 to 318.0 MiB beyond geometry (306.7 packed), Bistro exterior 210.1 to 155.5 (147.5); an H-PLOC batch 270.6 to 238.6 bytes per triangle.

## Left to do

- The M5: indexed leaves with quantized nodes together (surface-ray page section 8), which confirms or overturns the records default. The RTX 3060 and Intel oracle pages passed in the 2026-11-14 clean run.
- Timings on a GPU: the indexed GPU LBVH build, indexed adoption on Sponza (`scene-trees-sponza` variant `quantized-indexed-adopt`).
- Built 2026-11-14 (all off by default, defaults unchanged): tinybvh's indexed triangles inside the WASM (Sponza per mesh 0.89x the build time, the WASM peak 1,315.8 to 1,140.4 MiB as one mesh; `tools/tinybvh-formats.ts`), the GPU LBVH's indexed output, `CpuMeshBvh`'s `corners: 'positions'`, adoption in indexed scenes, and the app's positions (`positionsBuffer`). SwiftShader: `bvh-lbvh` (13 of 13 indexed trees the records trees converted), `bvh-hploc`, `scene-trees-runtime` and `scene-trees-mid-swap` (variants `indexed-adopt`, `indexed-adopt-merged-capped`, `quantized-indexed-adopt-merged`, `indexed-app-positions`) pass.
- Built 2026-11-15: `/three`'s `BvhBoundsTree` keeps indexed trees indexed (`corners: 'positions'`, shifted trees too; records trees unchanged), so a three CPU tree loaded from an indexed file costs 16 bytes per triangle beyond geometry, not 48 (see the status above). `computeBoundsTree` still builds records: an option to build indexed there is not added.

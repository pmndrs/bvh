# Indexed leaves (P9 lever 2)

Design only; nothing in `src/` changes. Today each tree-ordered triangle is a 48-byte record: three copied corners, the original index, flags and padding (`src/gpu-layout.ts:72-89`, written by `packTriangles`, `src/trace/triangle-packing.ts:74-95`). On Sponza the records are 263.0 of the 448.1 MiB we add beyond the app's geometry ([memory.md](memory.md)).

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

Material bits move to a side table, used only by merged meshes (section 5). Builders write vertex indices relative to the mesh. The scene adds the mesh's vertex base when it places the tree, because the instance record has no free word for it.

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
  - **Not recommended:** keeping merged groups on records, which would need a per-instance mode bit (bits 4-7 are reserved, `src/trace/layout.ts:129`) and a runtime branch in the leaf loop.
- **Deforming meshes and refit.**
  - The triangle step (`src/gpu-mesh-refit.ts:9-16`) stops writing 36 B of corners per triangle. It copies the posed vertices into the mesh's position range (12 B per vertex) and writes the boxes as before.
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
- absolute vertex indices in scene storage;
- custom walks reading `buffers.triangles` as 48-byte records, if indexed becomes the default.

## 7. Open questions for Dennis

1. Is B (our own `f32` copy, about 140 MiB saved on Sponza) the first step, with A (about 241 MiB, binding the app's buffers) kept for later, or dropped?
2. Packed `f32` positions (12 B, 9 loads) or `vec4f` (16 B, 3 loads)? Should the bench choose, or do we ship one?
3. Should merged groups go indexed with the 4 B side table, rather than stay on records with a mode bit?
4. Should the triangle format be a variant flag inside version 2, or a bump to 3?
5. Should the weld by exact bits be on by default? Do we want an opt-in tolerance weld?
6. What closest-hit cost on the RTX 3060 is 140 MiB worth before indexed becomes the default?

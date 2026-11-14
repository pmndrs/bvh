# Wide nodes: BVH4 and CWBVH (the traversal experiment)

## Status (2026-11-14): built, off by default; correctness gates pass on the Apple M5; the M5 quiet pass is in section 8: the gate is met on merged trees only, per scene (CWBVH on Bistro interior, BVH4 folded on Sponza); the RTX 3060 has not run

Two experimental scene node formats sit next to `'f32' | 'quantized'`: `nodeFormat: 'bvh4'` (4-wide, tinybvh's `BVH4_GPU` node) and `nodeFormat: 'cwbvh8'` (Ylitie, Karras and Laine 2017, the compressed 8-wide BVH, tinybvh's `BVH8_CWBVH` node). Both are off by default. The hit contract does not change: `BvhHit`, the 40- and 24-byte records, `GPU_LAYOUT_VERSION` 2, the `bvh*` names, the `bvhTrianglePasses` filter hook, ray masks and the watertight `bvhEdgeFunction` triangle test.

**Why now.** The pathtracer's research picked the wide BVH as its first traversal experiment (pathtracer `DECISIONS` D2, D21; `research/notes/05-gpu-architecture.md` §4.2-4.3). Today's binary walk does 104 Mrays/s closest-hit on incoherent rays in Sponza on the RTX 3060 Laptop. Native CWBVH is 6.3x that on a Pascal Titan X, and mutracer, the fastest web tracer in our bench, already uses tinybvh's CWBVH. Quantized binary nodes measured 0.93-0.97x of `f32` on the M5 (`compressed-nodes.md`), so compression alone does not help; width should. The B3 spike (ddgi branch `bvh-b3`, RTX 3060, 0.96-1.02x on Sponza, 0.82x on Bistro interior) was a GPU collapse plus a straight CWBVH walk inside ddgi; this is the faithful port, in the package, behind a setting.

**The decision gate** (pathtracer note 05): at least **1.3x closest-hit on incoherent rays on both the M5 and the RTX 3060 Laptop**, or the binary `f32` walk stays the default. The literature leads us to expect about 1.2x for BVH4 in software Vulkan, and 1.9-2.1x for CWBVH natively in CUDA, less the decompression, which costs about 2x the ALU in WGSL.

**What is in**

- `src/wide-nodes.ts`: the collapse and both packers, ported from tinybvh 1.9.1 (`c9be3103`, MIT; THIRD_PARTY.md), a refit that keeps the collapse (`refitWide`), the refit map for the GPU refit (`wideRefitMap`, `repackWide`), and a decoder for the checks and the CPU walk.
- `src/gpu-wide-refit.ts`: the GPU refit's pack, which `GpuMeshRefit` runs after the binary refit for a wide target (section 2).
- `src/trace/wgsl-wide.ts`: the wide walks, compiled by `traceModuleWgsl({ nodeFormat: 'bvh4' | 'cwbvh8', wideStack })`. `TraceBatchKernel` and `batchKernelWgsl` take the same options.
- `assembleTraceScene` / `buildTraceScene({ nodeFormat })`: every builder still writes the binary `f32` tree; the scene collapses each tree as it places it. CWBVH reorders each mesh's triangles in the scene.
- `SceneTrees({ nodeFormat })`: tinybvh's trees from the workers, collapsed in the workers too (`MeshBvhBuildOptions.wideNodes`: the worker sends the wide tree with the binary one, its buffers transferred). H-PLOC's first trees are collapsed on the GPU (`src/gpu-wide-collapse.ts`, section 2) and placed with their wide sizes, then swapped for tinybvh's as in an `f32` scene. Merged groups, runtime adds and removals, the GPU top level and indexed leaves work. Deforming meshes take GPU first trees too (collapsed with their refit maps) and are refit on the GPU (`setPose`, drift rebuilds) or the CPU (`setPoseCpu`), section 2. `gpuTreeOutputs: 'adopt'` is refused.
- `createCpuTracer` walks wide scenes (decoding each node as the WGSL does), so every scene check in Node and in the GPU pages reads wide buffers.
- `/three`'s `BvhTraceNodes` binds a wide source (nodes as `uvec4`; private stack only).
- Pages and tools: the `trace-oracle` wide variants and hook passes; `trace-wide-compile` (every wide kernel compiles, sizes and compile times); `scene-trees-mid-swap` `wide-*` variants; `bvh-wide-collapse` (the GPU collapse); `bvh-wide-refit` (the GPU refit) and the wide variants of `scene-trees-deforming`; section 8 of `trace-bench-surface-*`; `tools/wide-nodes.ts` (collapse time, bytes and CPU-walk work per ray, no GPU).

## 1. Layouts

Both are per-mesh trees in the scene's node buffer, read as `array<vec4u>`. `rootNodeOffset` counts in the format's node size (64 or 80 bytes). Child indices are mesh-local, and triangle indices are relative to the mesh's `triangleOffset`, as in the binary layout, so a tree can be placed anywhere. The top level stays the binary `f32` tree.

**BVH4, 64 bytes** (`BVH4_NODE_WORD`), tinybvh's `BVH4_GPU::BVHNode`:

| Words | Content |
| --- | --- |
| 0-2, 3 | the node box's min corner `p` (f32); x-min planes of children 0-3, one byte each |
| 4-6, 7 | the plane scale per axis, `2^e` (f32); x-max planes |
| 8, 9, 10, 11 | y-min, y-max, z-min, z-max planes, one byte per child |
| 12-15 | child words: 0 none; bit 31 set: a leaf of `((w >> 28) & 7) + 1` triangles from mesh triangle `w & 0x0fffffff`; else an inner node at mesh node `w` |

A plane decodes as `p + f32(q) * 2^e`.

**CWBVH, 80 bytes** (`CWBVH_NODE_WORD`), Ylitie et al. figure 2 as tinybvh lays it out:

| Words | Content |
| --- | --- |
| 0-2 | `p`, the node box's min corner (f32) |
| 3 | biased exponents `ex + 127`, `ey + 127`, `ez + 127`, then `imask` (bit per slot: an inner child) |
| 4 | child base index: the mesh node of the first inner child (the others follow in slot order) |
| 5 | triangle base index: the mesh triangle of the first leaf slot's first triangle |
| 6-7 | meta, a byte per slot: 0 empty; inner `(1 << 5) \| (24 + slot)`; leaf `unary(count) << 5 \| offset` (count 1-3 as `0b1`, `0b11`, `0b111`; offset 0-23 from the triangle base) |
| 8-19 | 8-bit planes, 8 slots each: lo x, lo y, lo z, hi x, hi y, hi z |

The planes decode as in BVH4. A leaf slot's triangles must follow each other, and a node's leaf slots must follow each other in slot order, so the converted tree carries its own triangle order (`WideMeshTree.triangleOrder`). The scene places the mesh's triangle records (or indexed triangles) in that order. Each record still names its original triangle, so hits, merged identity (`writeMergedTriangleIdentity` reads ids by slot) and materials are unchanged.

## 2. The collapse

`convertToWide(tree, format)`, ported from `MBVH<M>::ConvertFrom`:

1. The 64-byte binary nodes are read into tinybvh's form, one box per node, in pre-order. A single-triangle tree's root is that leaf. tinybvh then adds a level, because CWBVH needs an inner root.
2. CWBVH only: `CombineLeafs(3)` (a subtree of at most 3 triangles becomes one leaf), then `SplitLeafs(3)` (a larger leaf is cut into chains of at most 3). The split children keep the parent leaf's box, as tinybvh's do. That is `BVH8_CWBVH::Build`.
3. The dynamic program (Ylitie §4.2): bottom-up, the cost of each subtree spread over 1 to M slots, with `c_trav = c_int = 1` and `c_leaf = 0.8`. With CWBVH's `leafPrimLimit = 3`, a contiguous subtree of up to 3 triangles may become one leaf.
4. The decode, breadth-first, so a node's children are consecutive.
5. The pack: BVH4 depth-first (`BVH4_GPU::ConvertFrom`). CWBVH depth-first with tinybvh's greedy slot assignment: each child goes to the slot `s` that minimises `±dx ± dy ± dz` of its centroid's offset (the signs are the bits of `s`), so octant order is near-to-far.

**Quantization.** Per node and axis: `e` is the smallest exponent with `p + 255 * 2^e` decoding, in `f32`, at or above the node's max. A min plane is the largest `q` with `f32(p + q * 2^e) <= min`, and a max plane the smallest `q` with `f32(p + q * 2^e) >= max`. Both are checked against the value the GPU will compute (`Math.fround`), not assumed. The decode is monotone in `q`, so each plane is searched both ways from the quotient's estimate: the estimate can be one step short when the sum rounds down. The CPU and the GPU (`bvhwQuantizeMin`, `bvhwQuantizeMax`) both search, so they write the same bytes for the same boxes. Before 2026-11-14 each searched one way only, and the GPU's `f32` estimate sometimes landed one step tighter; both were valid. So every decoded box contains its exact box (`src/wide-nodes.test.ts` walks every node of every test tree). An empty child (non-finite or inverted box) gets lo 255 and hi 0, which the sign-swizzled slab test never enters.

**Shape, as tinybvh makes it** (`tools/wide-nodes.ts`, tinybvh trees, leaves of up to 8):

| | Sponza (132 meshes, 3,033,036 binary nodes) | Bistro interior (1,186 meshes, 482,907) |
| --- | --- | --- |
| BVH4 nodes, children per node, triangles per leaf | 1,219,869; 3.49; 1.89 | 196,552; 3.46; 2.15 |
| CWBVH nodes, children per node, triangles per leaf | 531,226; **5.94; 2.19** | 99,229; 5.84; 2.16 |
| Deepest mesh stack, BVH4 / CWBVH (instanced; merged) | 43 / 11 (53 / 14) | 30 / 9 (50 / 13) |

CWBVH's Sponza figures equal those B3 recorded from tinybvh's own CWBVH build (5.94 children per node, 2.19 triangles per leaf, ddgi `bench/results/trace.csv` `ours-cwbvh/*`), so the port collapses as tinybvh does.

**Time** (one M5 core, warm, every Sponza tree): BVH4 1.35 s, CWBVH 1.83 s, about 0.5 µs per binary node. The first version, with an object per node, took 11.8 s and 19.6 s. `SceneTrees` asks the tinybvh workers for it (`wideNodes`), so it runs beside the build, off the main thread; a tree that arrives without it (a custom pool) is collapsed on arrival. Bistro interior: 0.42 s and 0.44 s.

**The GPU collapse** (`src/gpu-wide-collapse.ts`, `src/gpu-wide-collapse-wgsl.ts`) is the same algorithm for trees in GPU buffers, as an H-PLOC batch leaves them. `SceneTrees` runs it on each batch's outputs before the batch's trees arrive, so a wide scene gets GPU first trees without the workers (`buildTraceScene` with GPU-built layouts still collapses on the CPU). A job holds any number of meshes of one node buffer and one triangle buffer:

1. `bvhwCost`: the dynamic program, bottom-up, dispatched once per level of the deepest tree (`maxDepth` dispatches). A node is computed in the first dispatch after all its inner children are, so no thread waits on another and the passes' order is all the synchronisation it needs (a climb with atomics would be one dispatch; this is simpler, and the batch's depth is small). It stores each node's costs over 1 to M slots, its split table (4 bits per slot count), its triangle range, its CONTIG and ASLEAF flags and its box.
2. `bvhwLevel`, then `bvhwArgs`, once per wide level (`maxDepth + 1` pairs; an empty level dispatches nothing, by `dispatchWorkgroupsIndirect`): each wide node on the frontier runs the decode walk over the split table, packs itself (CWBVH: tinybvh's greedy octant slots; planes rounded outward and checked in `f32` as the walks decode them, with `2^e` built from its bits, not `exp2`), takes one consecutive block of node addresses for its inner children from its mesh's counter, copies its leaf triangles into slot order (CWBVH) from its mesh's triangle counter, and appends its inner children to the next frontier. `bvhwArgs` turns the next frontier's length into the next indirect dispatch.
3. The per-mesh counts (nodes used, stack need) come back with a readback; the scene places each tree at its wide size and copies it out of the job's output.

Two choices differ from the CPU port. CWBVH's leaves of at most 3 come from the builder (`SceneTrees` asks H-PLOC for `maxLeafPrimitives: 3`), so there is no `SplitLeafs`; `CombineLeafs(3)` is folded into the cost pass (a contiguous subtree of at most 3 triangles becomes one leaf, whatever its costs). And costs are `f32`, and node addresses come from atomic counters, so a node's place in the buffer depends on scheduling; the tree does not. On 21 test trees (H-PLOC's and the LBVH's CPU twins and tinybvh, the oracle's hard meshes) the GPU collapse makes 9,088 BVH4 nodes against the CPU's 9,089 and 2,354 CWBVH nodes against 2,355, every tree passes `checkWideTree`, and a scene of each gives the `f32` scene's hits on 63,000 rays per format (`bvh-wide-collapse`). The first page run, before `CombineLeafs` was folded in, made twice the CWBVH nodes: tinybvh's forced combine matters.

Its cost: per binary node, M + 6 floats and 4 words of scratch; per mesh, room for its binary node count plus one wide node in the job's output (the most a collapse can need), freed with the batch. Not timed yet.

The decode walk keeps each task as a reference, `(binary node << 1) | side`, rather than as the node it names: the reference gives both the child and the word where its box is, which the refit map needs. A first version kept the box words in a third per-task array instead. BVH4's decode then lost children (a two-triangle tree kept only its right leaf) on Apple's compiler (Chrome, Metal), with or without a map, while the same array left unused passed. It was not reduced further; the references avoid it.

**The GPU refit** (`src/gpu-wide-refit.ts`, in `GpuMeshRefit`). Every wide child's box is a box of the binary tree it was collapsed from. A combined subtree's box is its root's; a split leaf's pieces keep their leaf's, as tinybvh's `SplitLeafs` does. So a deforming mesh keeps an `f32` copy of its binary tree, which the binary refit (B1a) refits unchanged, and then every wide node is packed again from the boxes its slots point at. The corner is its children's union, so nodes don't depend on each other: one dispatch, one thread per wide node. Child words, CWBVH's imask, meta and bases, and the triangle order stay as the collapse left them, as a binary refit keeps its builder's splits.

The pointers are a **refit map** (`WideRefitMap`). Per wide node and slot, it holds the word of the slot's box in the binary nodes (`node * 16 + side * 8`), a marker for the binary root (a tree of at most 3 triangles combined into one leaf: its box is its two children's union), or none. For CWBVH it also holds each record's triangle in the binary order: the refit's triangle step writes record `k`'s corners in place and its box where the binary tree has that triangle. Where a map comes from:

- `wideRefitMap(tree, layout)` makes one from the CPU collapse's topology (tinybvh's trees, collapsed in the workers).
- The GPU collapse writes one for each mesh that asks (`WideCollapseInput.refitMap`; `SceneTrees` asks for deforming meshes) after its output's triangle region, and it comes back with the counts. Box words are mesh-local, as the read-back layout's.

`repackWide` is the pack on the CPU. For a CPU-collapsed tree it equals `refitWide` byte for byte (`src/wide-nodes.test.ts`, split leaves and combined roots included), and the GPU pack equals both word for word (`bvh-wide-refit`). Cost per deforming mesh: 64 bytes per binary node (the copy), 16 or 32 per wide node (the slots) and, for CWBVH, 4 per triangle (the order). The quantized refit's copy is the same 64 bytes per binary node. Not timed.

In `SceneTrees`, deforming meshes now take GPU first trees in wide scenes too, with `setPose` and drift rebuilds. A CPU pose (`setPoseCpu`) repacks a tinybvh tree on the CPU (`refitWide`). A GPU-collapsed tree's wide nodes exist only on the GPU, so it goes through the same GPU pack instead (`GpuMeshRefit.encodePackFrom`: the CPU-refit binary nodes are written into the copy, then packed).

## 3. Traversal

`wgsl-wide.ts`. The top level is the binary walk's, unchanged. At an instance leaf, the ray moves into object space and a mesh walk (`bvhMeshClosest`, `bvhMeshAny`) runs to the end of the instance's tree with its own stack, then the top-level walk goes on. The binary walk instead has one loop and one stack for both levels, with an instance marker. The nested form keeps each mesh walk's stack and state its own, which CWBVH's two-word entries need. The closest hit is finished once, after the walk (`bvhFinishClosest`), as in the binary walk.

**BVH4** (`BVH4_GPU::Intersect`): load 4 `vec4u`, decode the four child boxes (`unpack4xU8`, one FMA per plane, four children at once in `vec4f`), test them, and sort the entry distances with tinybvh's five-swap network. Leaf children are tested at once. Inner children are pushed far to near, then the nearest is popped.

**CWBVH** (`BVH8_CWBVH::Intersect`): the stack holds node groups, `vec2u(child base, hit bits 24-31 | imask bits 0-7)`. Take the group's highest hit bit (`firstLeadingBit`), push the rest of the group if anything is left, and find the child at `base + countOneBits(imask & below(slot))`. Load 5 `vec4u`, decode eight child boxes in two halves of four, and build the hit mask from the meta bytes. An inner child sets bit `24 + (slot ^ octantInverse)`, and a leaf sets its unary triangle bits at its offset. The node bits become the next group, and the triangle bits (the triangle group) are tested before the next child. No sort is needed: the octant slot order and `octantInverse = 7 - octant(ray)` give near-to-far. The byte permutes (`prmt`, `__byte_perm`) are integer arithmetic: the sign extension of `is_inner` is `((x >> 4) & 0x01010101) * 0xff`, and byte extraction is `unpack4xU8` or `extractBits`. No subgroup operations.

**Exact plane decode.** Both walks decode each plane as `p + f32(q) * 2^e` (the product is exact, so fused or not it is one rounding) and run the binary walk's `bvhBoxEntry` arithmetic on it: `(plane - origin) * inverseDirection`, near and far plane picked by the direction's sign, the exit padded by `BVH_BOX_EXIT_PADDING`. The slab test is monotone in its planes. A decoded box contains the exact box, so any ray the binary walk sends into a leaf, the wide walk sends into the leaf that holds the same triangles. The triangle test is the same code on the same corners. **So the wide walks find the binary walk's hits**; a closest hit can name another triangle only when two triangles lie at exactly the same `t` (the walk order picks among ties). Ylitie folds the decode into the slab test (`q * (2^e / d) + (p - o) / d`, one FMA per plane, the order tinybvh's CPU code also uses). That is not exact: with large `|p - o|` the error is absolute, not relative to the box's distance, which is B3's leak class on Intel. The folded decode is built as a setting, `wideDecode: 'folded'` (lever L1, section 9).

**Any-hit order.** `'far'` (the default) pushes BVH4's children near to far, so the farthest is popped first. In CWBVH it uses `octantInverse = octant(ray)`, which reverses the slot order. `'near'` is the closest-hit order. `'larger'` has no wide form (it needs each child's area), and `traceModuleWgsl` refuses it.

**Kernel size** (`trace-wide-compile`, every combination of format, stack, mode, masks, leaves and record): the largest wide kernel is 42.3 KB of WGSL (`cwbvh8` workgroup, closest, indexed), against 35.9-38.4 KB for the binary ones. All are under the 50 KB budget for Windows DXC.

## 4. The CWBVH stack: private or workgroup

`wideStack: 'private' | 'workgroup'` (CWBVH only; `TraceModuleOptions`, `TraceBatchKernel`).

- `'private'` (the default): `var stack: array<vec2u, 32>` in the function. It is a dynamically indexed private array, which drivers place in scratch (NVIDIA: local memory, cached in L1).
- `'workgroup'`: the first 8 entries per lane in workgroup memory (`8 x 64 lanes x 8 bytes = 4 KiB`), indexed `entry * 64 + lane` so a warp at the same depth touches consecutive words; deeper entries spill to a 24-entry private array. A kernel must call `bvhSetStackLane(local_invocation_index)` before tracing, because the trace functions cannot see the builtin (the batch kernel does). The workgroup must have at most 64 lanes.

Ylitie et al. keep the stack in a mix of shared and local memory. CWBVH pushes at most one group per level, so 8 entries cover every tree with 8 or fewer levels: Sponza's deepest instanced mesh needs 11 and its merged tree 14 (`tools/wide-nodes.ts`). The spill is there so that no tree is refused. BVH4 pushes up to three siblings per level (Sponza: 43 entries instanced, 53 merged), so it keeps a private `array<u32, 64>` and has no workgroup option. `assembleTraceScene` and `SceneTrees` refuse a tree that needs more than the walk's stack (64 for BVH4, 32 for CWBVH).

## 5. Departures from tinybvh and Ylitie, and why

| # | Source | Here | Why |
| --- | --- | --- | --- |
| W1 | tinybvh stores triangles by value: inline after each BVH4 node, or in CWBVH's own array | leaves name ranges of the scene's triangle buffer; CWBVH reorders that buffer per mesh | One hit contract and one triangle test: records, indexed leaves, the filter flag, merged identity and materials stay what they are. The cost: a dependent load per triangle group, as in the binary walk. |
| W2 | BVH4_GPU scales planes by `extent / 255` | CWBVH's power-of-two scale `2^e` | It is exact: each decoded plane is one rounding, so the encoder can prove it contains the exact plane. |
| W3 | tinybvh rounds with `floor`/`ceil` on `f32` differences | each plane is the tightest the decoded `f32` plane allows (searched both ways from the estimate), and the exponent goes one up when `p + 255 * 2^e` would round below the max | tinybvh's rounding can land a plane one rounding inside. A box that contains the exact one is what gives the binary walk's hits; searching both ways makes the CPU and the GPU agree. |
| W4 | Ylitie and tinybvh fold the decode into the slab test | exact decode, then the binary walk's slab test, by default; the folded form behind `wideDecode: 'folded'`, padded per axis by a bound on its error | Hits identical to the binary walk's (section 3). The exact decode costs more ALU per plane; the padded fold is lever L1, and keeps the hits as long as the pad covers the error (the oracle checks it). |
| W5 | tinybvh's cost arithmetic is `float` | `f64` | No difference measured in the shape (Sponza's fullness numbers equal tinybvh's). A tie in the DP could fall the other way; that changes the shape, never a hit. |
| W6 | tinybvh's input is its own `BVH` | this package's 64-byte binary node, read into the one-box-per-node form; `CombineLeafs` checks that a subtree's triangles are contiguous (tinybvh assumes it) | Every builder here (tinybvh, LBVH, H-PLOC, three-mesh-bvh conversions) writes this layout. |
| W7 | one walk per level (tinybvh's GPU kernels trace a TLAS of BLASes with the same walk) | the binary top level, then the wide walk per instance | The top level is refit every frame by `GpuTopLevel` on binary nodes; it is 17 KiB to 0.3 MiB, not where the time goes. |
| W8 | tinybvh's BVH4 processes leaves far to near after the sort | the same (leaves are tested at once, in sorted order) | Faithful. Each hit leaf is tested anyway; the order only matters for ties. |
| W9 | CWBVH triangles tested highest bit first | lowest bit first (`firstTrailingBit`), shared with BVH4 | The binary walk's order within a leaf; only ties can differ. |
| W10 | no refit in tinybvh's CWBVH | `refitWide` on the CPU, and the GPU refit (section 2): the binary tree refit, the same wide nodes (addresses, slots, triangle order) re-quantized | Deforming meshes keep their ranges. The collapse's choices stay the build pose's, as a binary refit keeps its splits. |

No NVIDIA RTX SDK code. The paper was read; tinybvh's MIT code is what was ported. mutracer was not looked at.

## 6. Interactions

- **Merged static groups:** collapsed like any tree. The group's identity is written into the scene's copy after the reorder; it reads ids by slot.
- **Indexed leaves:** independent. CWBVH reorders the 16-byte indexed triangles; the merged side table is written by slot afterwards.
- **Quantized:** exclusive. Wide nodes are already 8-bit; a wide scene collapses `f32` trees and refuses quantized ones.
- **Refit:** `GpuMeshRefit` refits wide trees (section 2, "The GPU refit"). `meshRefitPlacement` gives a wide scene's mesh its wide root's word and format; `setTree` also needs the tree's refit map. For an assembled scene that is `wideRefitMap(convertToWide(layout, format), layout)`, the tree the assembly placed (neither is exported while the formats are experimental; `SceneTrees` does it itself). In `SceneTrees`, deforming meshes refit on the GPU (`setPose`, drift rebuilds) and on the CPU (`setPoseCpu`).
- **Serialization, `CpuMeshBvh`:** trees are never wide, only scenes, so neither changes.
- **Hard to undo:** nothing ships by default. The `nodeFormat` values and `wideStack` are experimental options.

## 7. Correctness (Apple M5, Chrome 157 headless)

- **`trace-oracle`, full profile** (665.9 s, passes). 17 wide variants: BVH4 and CWBVH on tinybvh, GPU LBVH and H-PLOC trees, and three-mesh-bvh trees (CWBVH); masks compiled out; any-hit near; indexed (vec4, packed); merged groups (tinybvh, H-PLOC, indexed); per-triangle materials; the CWBVH workgroup stack. Results: 0 mismatches against the CPU reference; **0 of 2M shared-edge rays slipped** in every variant; one-ulp seams leak exactly as `watertight-tinybvh` does (11,287 and 9,913 of 200k; merged, transformed: 25,923, the same as every binary merged variant). The 5 wide hook passes (filters, the query, instance masks against ray masks; BVH4 and CWBVH; masks on and off; merged; indexed): 33 cases, 0 mismatches.
- **`trace-wide-compile`:** 64 kernels compile; the largest is 42.3 KB.
- **`scene-trees-mid-swap`** `wide-*` variants (BVH4 and CWBVH, merged, indexed packed with the workgroup stack; H-PLOC's first trees collapsed on the GPU, then half swapped to tinybvh's, then all tinybvh's, collapsed in the workers; the GPU top level): 132 states, 0 disagreements, every must-hit ray hits.
- **`bvh-wide-collapse`:** the GPU collapse alone (section 2): 21 trees per format pass `checkWideTree` and give `f32`'s hits on 63,000 rays.
- **`bvh-wide-refit`:** the GPU refit (section 2) on 12 trees per format. Six meshes each come as tinybvh's tree collapsed on the CPU and as H-PLOC's CPU twin collapsed on the GPU with its map. Each is placed between padding, posed and refit on the GPU. The wide nodes and records equal the CPU's word for word (`refitWide`, `repackWide`; 0 words differ), the padding is untouched, `checkWideTree` passes on the posed records, and the hits equal the posed `f32` tree's on 36,000 rays per format, with 0 ties. The CPU pose's pack (`encodePackFrom`) gives the same nodes.
- **`scene-trees-deforming`** with BVH4 and CWBVH, records and indexed: a cloth's GPU tree (collapsed on the GPU with its map) refit from 3 poses, then tinybvh's tree after the swap from 4 more, a held bend until a drift rebuild (H-PLOC, collapsed on the GPU), then 2 CPU poses on the rebuilt tree through the GPU pack. Every check gives 0 disagreements with tinybvh `f32` trees of the same pose, the same as the `f32` and quantized variants.
- **`trace-bench-surface-sponza` and `-bistro-interior`** (section 8, the quiet pass): every wide answer equals `f32`'s, of about 33.3M answers per run, ties apart.
- **Unit tests:** `src/wide-nodes.test.ts` (every triangle reached exactly once; every decoded box holds every corner below it; CWBVH's leaf rules; the refit repacks to the same bytes in the same pose and holds a new pose; the refit map's repack equals `refitWide` on every test mesh, split leaves and combined roots included). `src/trace/wide-scene.test.ts` (instanced and merged, records and indexed, masks and a filter, any-hit far and near: every closest hit has the `f32` hit's `t`, ties counted, any-hit identical). `src/scene/scene-trees.test.ts` (`SceneTrees with wide nodes`; `setPoseCpu` for `bvh4` and `cwbvh8`).
- **`tools/wide-nodes.ts`** (CPU walk, about 20k air-grid and 10-16k secondary rays per scene): 0 hits differ from `f32`; 6-15 ties on Bistro interior merged (none elsewhere).

## 8. Measured

**Memory** (`tools/wide-nodes.ts`; MiB; the scene is nodes, triangles and top level, the "beyond geometry" of `memory.md`):

| | Nodes f32 | BVH4 | CWBVH | Scene f32 | BVH4 | CWBVH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sponza | 185.1 | 74.5 | **40.5** | 448.1 | 337.4 | **303.5** |
| Bistro interior | 29.5 | 12.0 | 7.6 | 77.4 | 59.9 | 55.5 |

CWBVH's 40.5 MiB on Sponza is B3's measured 42,488,000 bytes. Quantized binary nodes are 92.6 MiB on Sponza; CWBVH saves 52 MiB more, as `compressed-nodes.md` predicted. With indexed leaves the triangles shrink too; the two are independent.

**Work per ray on the CPU walk** (`tools/wide-nodes.ts`; node visits, wide over `f32`; a wide visit tests 4 or 8 boxes, a binary one 2; triangle tests, wide over `f32`):

| | Visits BVH4 | Visits CWBVH | Triangle tests BVH4 | CWBVH |
| --- | ---: | ---: | ---: | ---: |
| Sponza instanced, air grid closest / secondary closest | 0.68 / 0.68 | 0.58 / 0.56 | 1.04 / 1.08 | 1.14 / 1.22 |
| Sponza merged, air grid closest / secondary closest | 0.55 / 0.55 | 0.39 / 0.38 | 1.16 / 1.24 | 1.22 / 1.36 |
| Bistro interior instanced, air / secondary closest | 0.78 / 0.73 | 0.73 / 0.67 | 1.02 / 1.03 | 1.04 / 1.04 |
| Bistro interior merged, air / secondary closest | 0.49 / 0.47 | 0.36 / 0.34 | 1.00 / 1.02 | 0.94 / 1.00 |
| any hit (both scenes) | 0.48-0.80 | 0.33-0.75 | 0.97-1.04 | 0.98-1.06 |

The wide trees take a third to two thirds of the node visits, as width should. CWBVH tests up to 36% more triangles on Sponza, where `CombineLeafs(3)` and the leaf limit of 3 give up boxes that the binary tree's smaller leaves had. Many-mesh Bistro interior gains least on its instanced tree: each instance entry costs a whole wide root (B3's 0.82x).

**GPU speed: the quiet pass** (2026-11-14, run by the pathtracer's quiet-pass session with every app closed; HANDOFF "Quiet pass"). Apple M5 Pro, 18 cores, Chrome 157 headless; `trace-bench-surface-*` section 8 at commit `11d19f6`, full profile: 5 paired rounds, formats back to back, the order reversed every other round, each timing 0.5 s of back-to-back dispatches after warm-up; under the shared GPU lock, 1-minute load 1.2-2.2 before and after every run, 1 Chrome GPU process (the page's own); every run counted as quiet (`machine/quiet` 1). Three runs per scene. Wide over `f32`: **min..max of the three runs' medians.** The rows are in `bench/results/trace.csv` (commit `11d19f6`). `bvh4-single` is lever L7 (the single-loop walk), `-folded` lever L1 (the folded plane decode), section 9.

| Closest hit | Sponza primary | Sponza surface | Sponza air grid | Bistro int. primary | Bistro int. surface | Bistro int. air grid |
| --- | --- | --- | --- | --- | --- | --- |
| BVH4, merged | 1.26..1.29 | **1.30..1.34** | 1.27..1.28 | 1.35..1.35 | 1.03..1.04 | 1.25..1.25 |
| BVH4 folded, merged | 1.31..1.37 | **1.36..1.43** | **1.34..1.34** | 1.40..1.40 | 1.05..1.05 | 1.27..1.28 |
| BVH4 single, merged | 0.97..1.00 | 1.05..1.10 | 1.02..1.02 | 1.15..1.15 | 0.96..0.96 | 1.12..1.12 |
| CWBVH, merged | 0.93..0.99 | 0.98..1.02 | 1.01..1.01 | 1.42..1.45 | **1.40..1.41** | **1.39..1.40** |
| CWBVH folded, merged | 1.03..1.09 | 1.06..1.09 | 1.10..1.11 | 1.53..1.57 | **1.46..1.46** | **1.45..1.46** |
| CWBVH workgroup, merged | 0.90..0.96 | 0.96..0.99 | 0.97..0.99 | 1.40..1.41 | **1.38..1.38** | **1.36..1.36** |
| BVH4, instanced | 1.40..1.41 | 0.69..0.70 | 0.86..0.87 | 1.35..1.35 | 0.67..0.67 | 0.82..0.83 |
| BVH4 folded, instanced | 1.49..1.50 | 0.75..0.75 | 0.92..0.92 | 1.36..1.40 | 0.68..0.68 | 0.85..0.85 |
| BVH4 single, instanced | 0.93..0.93 | 0.88..0.89 | 0.91..0.91 | 1.03..1.04 | 0.96..0.96 | 0.97..0.97 |
| CWBVH, instanced | 1.16..1.20 | 0.56..0.57 | 0.74..0.74 | 1.19..1.25 | 0.66..0.66 | 0.79..0.80 |
| CWBVH folded, instanced | 1.28..1.32 | 0.63..0.64 | 0.83..0.83 | 1.28..1.33 | 0.70..0.70 | 0.84..0.84 |
| CWBVH workgroup, instanced | 1.14..1.16 | 0.55..0.56 | 0.73..0.73 | 1.16..1.20 | 0.65..0.65 | 0.78..0.78 |

| Any hit | Sponza primary | surface | air grid | Bistro int. primary | surface | air grid |
| --- | --- | --- | --- | --- | --- | --- |
| BVH4, merged | 0.92..0.94 | 1.19..1.21 | 1.09..1.11 | 1.17..1.18 | 0.89..0.89 | 1.00..1.00 |
| BVH4 folded, merged | 0.97..1.00 | 1.24..1.29 | 1.14..1.17 | 1.23..1.23 | 0.91..0.91 | 1.02..1.02 |
| BVH4 single, merged | 0.76..0.78 | 1.03..1.06 | 0.96..0.99 | 1.06..1.06 | 0.85..0.85 | 0.95..0.95 |
| CWBVH, merged | 0.57..0.59 | 1.07..1.10 | 1.22..1.24 | 1.48..1.48 | 1.52..1.52 | 1.54..1.54 |
| CWBVH folded, merged | 0.66..0.66 | 1.19..1.21 | 1.34..1.37 | 1.62..1.62 | 1.60..1.60 | 1.62..1.62 |
| CWBVH workgroup, merged | 0.55..0.56 | 1.03..1.04 | 1.17..1.19 | 1.40..1.41 | 1.47..1.47 | 1.49..1.49 |
| BVH4, instanced | 0.88..0.88 | 0.63..0.63 | 0.62..0.62 | 0.97..0.98 | 0.65..0.65 | 0.72..0.72 |
| BVH4 folded, instanced | 0.91..0.92 | 0.67..0.67 | 0.66..0.66 | 1.01..1.02 | 0.67..0.67 | 0.74..0.74 |
| BVH4 single, instanced | 0.84..0.84 | 0.90..0.90 | 0.92..0.92 | 0.98..0.99 | 0.93..0.93 | 0.94..0.94 |
| CWBVH, instanced | 0.84..0.85 | 0.64..0.64 | 0.68..0.68 | 1.03..1.04 | 0.74..0.74 | 0.79..0.79 |
| CWBVH folded, instanced | 0.92..0.93 | 0.72..0.72 | 0.76..0.76 | 1.11..1.11 | 0.78..0.78 | 0.84..0.84 |
| CWBVH workgroup, instanced | 0.78..0.79 | 0.63..0.63 | 0.66..0.66 | 0.98..0.98 | 0.72..0.72 | 0.78..0.78 |

**The lever A/Bs**, within a round (the same trees, the same rays):

| | Sponza closest | Sponza any | Bistro int. closest | Bistro int. any |
| --- | --- | --- | --- | --- |
| L1: BVH4 folded over BVH4 (every tree and set) | 1.04..1.08 | 1.04..1.07 | 1.01..1.04 | 1.02..1.05 |
| L1: CWBVH folded over CWBVH (every tree and set) | 1.08..1.13 | 1.09..1.13 | 1.03..1.09 | 1.05..1.10 |
| L7: BVH4 single over nested, instanced, surface / air grid | 1.26..1.27 / 1.06 | 1.42 / 1.48 | 1.45 / 1.18 | 1.43..1.44 / 1.30 |
| L7: BVH4 single over nested, instanced, primary | 0.66 | 0.95 | 0.77 | 1.01 |
| L7: BVH4 single over nested, merged (every set) | 0.77..0.81 | 0.83..0.89 | 0.85..0.93 | 0.90..0.96 |

`f32` rates for scale (Mrays/s, closest, median of rounds per run): Sponza instanced surface 67.7-68.5, merged surface 112.9-122.2; Bistro interior instanced surface 35.6, merged surface 42.0.

Hits: 0 answers differ from `f32` in any format, walk, decode and run, of about 33.3M answers per run; ties (another triangle at exactly the same `t`) 231 over Sponza's three runs and 1,803 over Bistro interior's.

**Clock and order.** The M5's clock ramps over the first second or two of work (the pathtracer's trace-timing page saw a pass flip between 9 and 4.5 ms). Here every timing is 0.5 s of back-to-back dispatches inside a page that keeps the GPU busy from start to end, and a case's formats run back to back with the order reversed every other round. The best-effort runs at `92ec9e7` showed no ramp in the raw rates: across the 5 rounds the median spread of a loaded rate was 0.3%; round 1 over the median of rounds 2-5 was 1.000 (0.90-1.11); the single-dispatch rate over the loaded rate 1.000 (0.95-1.04). In the quiet pass the three runs' medians agree to 0-7%, the widest on primary-ray closest hits.

What the quiet pass says:

- **On one big tree (merged), width pays, and which width depends on the scene.** Bistro interior (many small meshes merged into one tree): CWBVH 1.39-1.41x closest-hit on incoherent rays, 1.45-1.46x folded, and 1.52-1.62x any-hit; BVH4 1.03-1.28x. Sponza (fewer, larger meshes): BVH4 1.27-1.34x, 1.34-1.43x folded; CWBVH 0.98-1.02x, 1.06-1.11x folded. The CPU walk counted 0.34-0.39x the node visits for CWBVH on both merged trees, so Sponza's CWBVH loses its gain elsewhere: it tests 22-36% more triangles there (merged leaves of up to 3, and split leaves keep the parent's box: L4), where Bistro interior's merged tree tests 0.94-1.00x.
- **L1, the folded decode, is faster on every row:** 1.01-1.08x for BVH4 and 1.03-1.13x for CWBVH, more on Sponza than on Bistro interior. It changes no answer.
- **L7, the single loop, helps instanced trees but does not rescue them.** On incoherent instanced rays it is 1.06-1.48x the nested walk, but the best instanced result is still 0.88-0.97x `f32` closest-hit. It loses on coherent primary rays (0.66-0.77x closest) and on every merged row (0.77-0.96x). So the nested walk's shape is part of the instanced loss, not all of it: each instance entry still costs a whole wide root, as on the CPU walk.
- **On the instanced trees (132 and 1,186 meshes) every format is slower than `f32` on incoherent rays** (0.55-0.97x, the single loop included) though they visit 0.56-0.78x the nodes; coherent primary rays gain with the nested walk (1.14-1.50x closest).
- **The CWBVH workgroup stack is 1-8% slower** than the private stack on the M5.
- **Against the gate (1.3x closest-hit on incoherent rays, M5 and RTX 3060):** on the M5 it is met only on merged trees and only per scene. On Bistro interior every CWBVH variant meets it (1.36-1.46x) and no BVH4 does (1.03-1.28x). On Sponza BVH4 folded meets it (1.34-1.43x), exact BVH4 meets it on surface rays only (1.30-1.34x; air grid 1.27-1.28x), and no CWBVH does (at most 1.11x). No format meets it on both scenes, and none on instanced trees. The RTX 3060 has not run.

The best-effort runs (`92ec9e7`, Chrome open; their rows say `machine/quiet` 1 because the load check cannot see Chrome) agree with the quiet pass within 0-6%. Between them every plane became the tightest the `f32` decode allows (section 2), a change of at most one step per plane.

**Windows (RTX 3060 Laptop): what to run**

```
BVH_ASSETS_DIR=<ddgi assets> BVH_MANIFESTS_DIR=<ddgi manifests> BVH_PAGE_QUERY='skip=hit-record,leaves,nodes,any-order' pnpm test:gpu:page trace-bench-surface-sponza --json wide-sponza-1.json
```
Run it three times per scene (`trace-bench-surface-sponza`, `trace-bench-surface-bistro-interior`), one page at a time with nothing else on the GPU, then `node tests/gpu/pages/trace-bench/surface-to-csv.ts wide-*.json --append`. Also run `pnpm test:gpu:page trace-oracle` (the wide variants are the gate on NVIDIA: 0 slips) and `pnpm test:gpu:page trace-wide-compile` (DXC compile times per kernel, against the binary kernels in the same run). The runbook (`WINDOWS-RUNBOOK.md`) has the commands with paths.

## 9. Levers not built (measured options, after the gate)

- **L1, the folded decode: built** as `wideDecode: 'folded'` for both formats (`TraceModuleOptions`, `TraceBatchKernel`; default `'exact'`). Per node and axis `a = 2^e / d`, `b = (p - o) / d`; each plane's distance is `fma(q, a, b)` (Ylitie's form), and each axis's near distance moves down and far distance up by `2^-20 * (|b| + (|p| + 255 * 2^e) * |1/d|)`. That covers the folded arithmetic's absolute error, the exact path's, and the half ulp by which `p + q * 2^e` rounds (the encoder checks the rounded plane, not the real one), with a margin of about 4x. So the folded box contains the exact path's computed box, and the hits stay the binary walk's. An axis whose `1/d` is huge (a direction component near 0) gets a huge pad: that axis stops culling for that ray, which costs visits, never hits. Checked by the `trace-oracle` variants `wide-*-folded-*` (tinybvh, LBVH single loop, merged indexed H-PLOC with the workgroup stack; CWBVH's hook passes run folded): on the M5, full profile, 0 of 2M shared-edge rays slip (B3's leak class), the one-ulp seams leak 10,927 of 200k against records' 11,287 (the padded boxes take rays the exact slab test rounds away, as quantized nodes do: never more), and the 21 folded CWBVH hook cases have 0 mismatches. Section 8's `bvh4-folded` and `cwbvh8-folded` layouts (the same scenes through folded kernels) time it: on the M5 quiet pass it is faster on every row, 1.01-1.08x for BVH4 and 1.03-1.13x for CWBVH.
- **L2, triangle-group postponing** (Ylitie §3.3): push the triangle group when few lanes of the warp want to test triangles. It needs subgroup ballots in uniform control flow.
- **L3, compressed triangles** (`CWBVH_COMPRESSED_TRIS`, Baldwin-Weber): not compatible with the watertight test, so ruled out by the hit contract.
- **L4, exact boxes for split leaves:** tinybvh's `SplitLeafs` copies the parent leaf's box; boxes from the triangles would cut CWBVH's extra triangle tests on Sponza.
- **L5, the collapse on the GPU and in the workers: both built** (section 2), and the GPU refit. Neither GPU pass is timed: a one-dispatch climb would replace the `maxDepth` cost dispatches if they show.
- **L6, BVH4's workgroup stack**, if the private stack turns out to be what CWBVH's workgroup stack fixes.
- **L7, a single-loop wide walk: built for BVH4** as `wideWalk: 'single'` (`TraceModuleOptions`, `TraceBatchKernel`; default `'nested'`). One loop and one 136-entry stack for both levels with the instance marker, as the binary walk has, so a lane that finishes an instance's tree steps the top level in the same iteration as lanes still inside theirs. Top-level entries are the binary walk's references, mesh entries BVH4's inner child words (node indices, never the marker); the node step is the nested walk's text (`bvh4NodeStepWgsl`). The closest hit keeps the record of the instance that last changed it. CWBVH would need the top level's references packed into its two-word entries; not built until BVH4's A/B says the shape matters. Checked like the nested walk: `trace-oracle` variants `wide-bvh4-single-*` (records, masks off with any-hit near, merged indexed H-PLOC, materials; the BVH4 hook passes run single), `trace-wide-compile`, and section 8's `bvh4-single` layout (BVH4's scene through the single-loop kernel). On the M5 quiet pass it is 1.06-1.48x the nested walk on incoherent instanced rays, yet still 0.88-0.97x `f32` closest-hit there, and slower on primary rays (0.66-0.77x closest) and on every merged row (0.77-0.96x).

## 10. Follow-ups

1. The RTX 3060 runs (section 8, runbook 2d), then the gate. The M5 quiet pass is done (section 8).
2. L7 for CWBVH only if instanced trees matter enough: BVH4's single loop gains on incoherent instanced rays but stays below `f32`, so the instanced loss needs more than the walk's shape.
3. Time the GPU collapse, the time to first trees, and the GPU refit's pack per deforming mesh (all built, none timed).
4. If the gate passes: triangle postponing (L2) as a measured lever, `serialize` for wide scenes, and `/three`'s `wideStack`.

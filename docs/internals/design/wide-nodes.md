# Wide nodes: BVH4 and CWBVH (the traversal experiment)

## Status (2026-11-13): built, off by default; correctness gates pass on the Apple M5; speed so far only provisional (a loaded machine), and the gate is not met there

Two experimental scene node formats sit next to `'f32' | 'quantized'`: `nodeFormat: 'bvh4'` (4-wide, tinybvh's `BVH4_GPU` node) and `nodeFormat: 'cwbvh8'` (Ylitie, Karras and Laine 2017, the compressed 8-wide BVH, tinybvh's `BVH8_CWBVH` node). Both are off by default. The hit contract does not change: `BvhHit`, the 40- and 24-byte records, `GPU_LAYOUT_VERSION` 2, the `bvh*` names, the `bvhTrianglePasses` filter hook, ray masks and the watertight `bvhEdgeFunction` triangle test.

**Why now.** The pathtracer's research picked the wide BVH as its first traversal experiment (pathtracer `DECISIONS` D2, D21; `research/notes/05-gpu-architecture.md` §4.2-4.3). Today's binary walk does 104 Mrays/s closest-hit on incoherent rays in Sponza on the RTX 3060 Laptop. Native CWBVH is 6.3x that on a Pascal Titan X, and mutracer, the fastest web tracer in our bench, already uses tinybvh's CWBVH. Quantized binary nodes measured 0.93-0.97x of `f32` on the M5 (`compressed-nodes.md`), so compression alone does not help; width should. The B3 spike (ddgi branch `bvh-b3`, RTX 3060, 0.96-1.02x on Sponza, 0.82x on Bistro interior) was a GPU collapse plus a straight CWBVH walk inside ddgi; this is the faithful port, in the package, behind a setting.

**The decision gate** (pathtracer note 05): at least **1.3x closest-hit on incoherent rays on both the M5 and the RTX 3060 Laptop**, or the binary `f32` walk stays the default. The literature leads us to expect about 1.2x for BVH4 in software Vulkan, and 1.9-2.1x for CWBVH natively in CUDA, less the decompression, which costs about 2x the ALU in WGSL.

**What is in**

- `src/wide-nodes.ts`: the collapse and both packers, ported from tinybvh 1.9.1 (`c9be3103`, MIT; THIRD_PARTY.md), a refit that keeps the collapse (`refitWide`), and a decoder for the checks and the CPU walk.
- `src/trace/wgsl-wide.ts`: the wide walks, compiled by `traceModuleWgsl({ nodeFormat: 'bvh4' | 'cwbvh8', wideStack })`. `TraceBatchKernel` and `batchKernelWgsl` take the same options.
- `assembleTraceScene` / `buildTraceScene({ nodeFormat })`: every builder still writes the binary `f32` tree; the scene collapses each tree as it places it. CWBVH reorders each mesh's triangles in the scene.
- `SceneTrees({ nodeFormat })`: tinybvh's trees from the workers, collapsed on the main thread when they arrive. There is no GPU collapse yet, so GPU first trees are off. Merged groups, runtime adds and removals, the GPU top level and indexed leaves work. Deforming meshes take CPU poses (`setPoseCpu`: the binary tree refit, then the same wide nodes repacked in place). `setPose` (the GPU refit) is skipped with a warning. `gpuTreeOutputs: 'adopt'` is refused.
- `createCpuTracer` walks wide scenes (decoding each node as the WGSL does), so every scene check in Node and in the GPU pages reads wide buffers.
- `/three`'s `BvhTraceNodes` binds a wide source (nodes as `uvec4`; private stack only).
- Pages and tools: 17 `trace-oracle` variants and 5 hook passes; `trace-wide-compile` (every wide kernel compiles, sizes and compile times); `scene-trees-mid-swap` `wide-*` variants; section 8 of `trace-bench-surface-*`; `tools/wide-nodes.ts` (collapse time, bytes and CPU-walk work per ray, no GPU).

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

**Quantization.** Per node and axis: `e` is the smallest exponent with `p + 255 * 2^e` decoding, in `f32`, at or above the node's max. A min plane is the largest `q` with `f32(p + q * 2^e) <= min`, and a max plane the smallest `q` with `f32(p + q * 2^e) >= max`. Both are checked against the value the GPU will compute (`Math.fround`), not assumed. So every decoded box contains its exact box (`src/wide-nodes.test.ts` walks every node of every test tree). An empty child (non-finite or inverted box) gets lo 255 and hi 0, which the sign-swizzled slab test never enters.

**Shape, as tinybvh makes it** (`tools/wide-nodes.ts`, tinybvh trees, leaves of up to 8):

| | Sponza (132 meshes, 3,033,036 binary nodes) | Bistro interior (1,186 meshes, 482,907) |
| --- | --- | --- |
| BVH4 nodes, children per node, triangles per leaf | 1,219,869; 3.49; 1.89 | 196,552; 3.46; 2.15 |
| CWBVH nodes, children per node, triangles per leaf | 531,226; **5.94; 2.19** | 99,229; 5.84; 2.16 |
| Deepest mesh stack, BVH4 / CWBVH (instanced; merged) | 43 / 11 (53 / 14) | 30 / 9 (50 / 13) |

CWBVH's Sponza figures equal those B3 recorded from tinybvh's own CWBVH build (5.94 children per node, 2.19 triangles per leaf, ddgi `bench/results/trace.csv` `ours-cwbvh/*`), so the port collapses as tinybvh does.

**Time** (one M5 core, warm, every Sponza tree): BVH4 1.35 s, CWBVH 1.83 s, about 0.5 µs per binary node. The first version, with an object per node, took 11.8 s and 19.6 s. `SceneTrees` runs the collapse on the main thread as each tinybvh tree arrives (the largest Sponza mesh is a few percent of that). Moving it into the workers is the first follow-up. Bistro interior: 0.42 s and 0.44 s.

**The GPU collapse is not built.** H-PLOC and LBVH trees are collapsed on the CPU like tinybvh's (`buildTraceScene` with any builder works, and the oracle runs them). `SceneTrees` skips GPU first trees for wide formats, so a wide scene waits for tinybvh (Sponza's trees take about 370 ms in the workers, plus the collapse). The follow-up is the dynamic program as a bottom-up pass after H-PLOC, the decode as a top-down pass per level, and the pack per node.

## 3. Traversal

`wgsl-wide.ts`. The top level is the binary walk's, unchanged. At an instance leaf, the ray moves into object space and a mesh walk (`bvhMeshClosest`, `bvhMeshAny`) runs to the end of the instance's tree with its own stack, then the top-level walk goes on. The binary walk instead has one loop and one stack for both levels, with an instance marker. The nested form keeps each mesh walk's stack and state its own, which CWBVH's two-word entries need. The closest hit is finished once, after the walk (`bvhFinishClosest`), as in the binary walk.

**BVH4** (`BVH4_GPU::Intersect`): load 4 `vec4u`, decode the four child boxes (`unpack4xU8`, one FMA per plane, four children at once in `vec4f`), test them, and sort the entry distances with tinybvh's five-swap network. Leaf children are tested at once. Inner children are pushed far to near, then the nearest is popped.

**CWBVH** (`BVH8_CWBVH::Intersect`): the stack holds node groups, `vec2u(child base, hit bits 24-31 | imask bits 0-7)`. Take the group's highest hit bit (`firstLeadingBit`), push the rest of the group if anything is left, and find the child at `base + countOneBits(imask & below(slot))`. Load 5 `vec4u`, decode eight child boxes in two halves of four, and build the hit mask from the meta bytes. An inner child sets bit `24 + (slot ^ octantInverse)`, and a leaf sets its unary triangle bits at its offset. The node bits become the next group, and the triangle bits (the triangle group) are tested before the next child. No sort is needed: the octant slot order and `octantInverse = 7 - octant(ray)` give near-to-far. The byte permutes (`prmt`, `__byte_perm`) are integer arithmetic: the sign extension of `is_inner` is `((x >> 4) & 0x01010101) * 0xff`, and byte extraction is `unpack4xU8` or `extractBits`. No subgroup operations.

**Exact plane decode.** Both walks decode each plane as `p + f32(q) * 2^e` (the product is exact, so fused or not it is one rounding) and run the binary walk's `bvhBoxEntry` arithmetic on it: `(plane - origin) * inverseDirection`, near and far plane picked by the direction's sign, the exit padded by `BVH_BOX_EXIT_PADDING`. The slab test is monotone in its planes. A decoded box contains the exact box, so any ray the binary walk sends into a leaf, the wide walk sends into the leaf that holds the same triangles. The triangle test is the same code on the same corners. **So the wide walks find the binary walk's hits**; a closest hit can name another triangle only when two triangles lie at exactly the same `t` (the walk order picks among ties). Ylitie folds the decode into the slab test (`q * (2^e / d) + (p - o) / d`, one FMA per plane, the order tinybvh's CPU code also uses). That is not exact: with large `|p - o|` the error is absolute, not relative to the box's distance, which is B3's leak class on Intel. The folded decode is a measured lever, not built (section 9).

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
| W3 | tinybvh rounds with `floor`/`ceil` on `f32` differences | the outward rounding is checked against the decoded `f32` plane, and the exponent goes one up when `p + 255 * 2^e` would round below the max | tinybvh's rounding can land a plane one rounding inside. A box that contains the exact one is what gives the binary walk's hits. |
| W4 | Ylitie and tinybvh fold the decode into the slab test | exact decode, then the binary walk's slab test | Hits identical to the binary walk's (section 3). Costs more ALU per plane; the folded form is lever L1. |
| W5 | tinybvh's cost arithmetic is `float` | `f64` | No difference measured in the shape (Sponza's fullness numbers equal tinybvh's). A tie in the DP could fall the other way; that changes the shape, never a hit. |
| W6 | tinybvh's input is its own `BVH` | this package's 64-byte binary node, read into the one-box-per-node form; `CombineLeafs` checks that a subtree's triangles are contiguous (tinybvh assumes it) | Every builder here (tinybvh, LBVH, H-PLOC, three-mesh-bvh conversions) writes this layout. |
| W7 | one walk per level (tinybvh's GPU kernels trace a TLAS of BLASes with the same walk) | the binary top level, then the wide walk per instance | The top level is refit every frame by `GpuTopLevel` on binary nodes; it is 17 KiB to 0.3 MiB, not where the time goes. |
| W8 | tinybvh's BVH4 processes leaves far to near after the sort | the same (leaves are tested at once, in sorted order) | Faithful. Each hit leaf is tested anyway; the order only matters for ties. |
| W9 | CWBVH triangles tested highest bit first | lowest bit first (`firstTrailingBit`), shared with BVH4 | The binary walk's order within a leaf; only ties can differ. |
| W10 | no refit in tinybvh's CWBVH | `refitWide`: the binary tree refit, the same wide nodes (addresses, slots, triangle order) re-quantized | Deforming meshes keep their ranges. The collapse's choices stay the build pose's, as a binary refit keeps its splits. |

No NVIDIA RTX SDK code. The paper was read; tinybvh's MIT code is what was ported. mutracer was not looked at.

## 6. Interactions

- **Merged static groups:** collapsed like any tree. The group's identity is written into the scene's copy after the reorder; it reads ids by slot.
- **Indexed leaves:** independent. CWBVH reorders the 16-byte indexed triangles; the merged side table is written by slot afterwards.
- **Quantized:** exclusive. Wide nodes are already 8-bit; a wide scene collapses `f32` trees and refuses quantized ones.
- **Refit:** `meshRefitPlacement` refuses wide scenes, so there is no `GpuMeshRefit` on wide nodes. Deforming meshes in `SceneTrees` take CPU poses through `refitWide`. A GPU wide refit would be the binary refit (B1a climb) followed by a repack pass, one thread per wide node, over the `WideTopology` tables; it is a follow-up.
- **Serialization, `CpuMeshBvh`:** trees are never wide, only scenes, so neither changes.
- **Hard to undo:** nothing ships by default. The `nodeFormat` values and `wideStack` are experimental options.

## 7. Correctness (Apple M5, Chrome 157 headless)

- **`trace-oracle`, full profile** (665.9 s, passes). 17 wide variants: BVH4 and CWBVH on tinybvh, GPU LBVH and H-PLOC trees, and three-mesh-bvh trees (CWBVH); masks compiled out; any-hit near; indexed (vec4, packed); merged groups (tinybvh, H-PLOC, indexed); per-triangle materials; the CWBVH workgroup stack. Results: 0 mismatches against the CPU reference; **0 of 2M shared-edge rays slipped** in every variant; one-ulp seams leak exactly as `watertight-tinybvh` does (11,287 and 9,913 of 200k; merged, transformed: 25,923, the same as every binary merged variant). The 5 wide hook passes (filters, the query, instance masks against ray masks; BVH4 and CWBVH; masks on and off; merged; indexed): 33 cases, 0 mismatches.
- **`trace-wide-compile`:** 64 kernels compile; the largest is 42.3 KB.
- **`scene-trees-mid-swap`** `wide-*` variants (BVH4 and CWBVH, merged, indexed packed with the workgroup stack; tinybvh trees collapsed on arrival, the GPU top level): 44 states, 0 disagreements, every must-hit ray hits (573 s for the whole page).
- **`trace-bench-surface-sponza`** (section 8): every wide answer equals `f32`'s on 7.8M rays per run, ties apart.
- **Unit tests:** `src/wide-nodes.test.ts` (every triangle reached exactly once; every decoded box holds every corner below it; CWBVH's leaf rules; the refit repacks to the same bytes in the same pose and holds a new pose). `src/trace/wide-scene.test.ts` (instanced and merged, records and indexed, masks and a filter, any-hit far and near: every closest hit has the `f32` hit's `t`, ties counted, any-hit identical). `src/scene/scene-trees.test.ts` (`SceneTrees with wide nodes`; `setPoseCpu` for `bvh4` and `cwbvh8`).
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

**GPU speed, provisional** (Apple M5 Pro, Chrome 157, `trace-bench-surface-sponza` section 8 at commit `00d3054`, `profile=quick`: 3 paired rounds; held the GPU lock, but **the machine was loaded**: 1-minute load 31.7 before and 239.8 after on 18 cores, 27-35 Chrome GPU processes of other jobs; rows in `bench/results/trace.csv` with `machine/quiet` 0). Wide over `f32`, median of rounds (min-max). **Not a gate result.**

| Sponza, closest hit | primary (coherent) | surface (incoherent) | air grid (incoherent) |
| --- | --- | --- | --- |
| BVH4, instanced | **1.40** (1.36-1.42) | 0.70 (0.69-0.71) | 0.86 (0.86-0.86) |
| CWBVH, instanced | 1.19 (1.11-1.23) | 0.57 (0.55-0.59) | 0.75 (0.74-0.76) |
| CWBVH workgroup stack, instanced | 1.14 (1.02-1.21) | 0.57 (0.55-0.59) | 0.73 (0.71-0.75) |
| BVH4, merged | 1.28 (1.27-1.29) | **1.32** (1.31-1.33) | 1.29 (1.28-1.29) |
| CWBVH, merged | 0.97 (0.95-1.00) | 1.00 (0.99-1.00) | 1.02 (1.02-1.04) |
| CWBVH workgroup stack, merged | 0.91 (0.89-0.96) | 0.98 (0.96-0.98) | 1.00 (0.99-1.02) |

Any hit: BVH4 0.66-0.94 instanced and 0.94-1.22 merged; CWBVH 0.68-0.84 instanced and 0.58-1.27 merged (`wide_*_over_f32/*/any` rows). Rates, `f32`: instanced 46-125 Mrays/s on incoherent closest hits, merged 85-171 (the binary walk's merged tree is already 1.8x its instanced one, HANDOFF).

Hits: 0 of 7.8M answers differ from `f32` in any format; 0-8 ties per set of about 230k.

What the provisional run says, to be confirmed on a quiet machine:

- **On one big tree (merged), width pays, but only BVH4's:** 1.28-1.32x closest-hit, which is about the 1.2x the literature gives BVH4 in software Vulkan. CWBVH is even (0.97-1.02x) despite 0.38x the node visits on the CPU walk: on the M5 its 8-child decode (two halves of four, exact planes) and the meta and hit-mask arithmetic cost what the visits save. The folded decode (L1) is the obvious next measurement.
- **On the instanced tree (132 meshes), both are slower on incoherent rays** (BVH4 0.70-0.86x, CWBVH 0.57-0.75x) though they visit 0.56-0.68x the nodes. The suspect is the walk's shape, not the nodes: the wide walk nests a mesh loop inside the top-level loop (section 3), so in a warp whose rays enter different instances the lanes wait for each other at the inner loop's end, where the binary walk's single loop keeps them all busy. Coherent primary rays, which enter the same instances, gain (1.40x). Lever L7 (a single loop for BVH4) tests it.
- **The CWBVH workgroup stack is no faster** than the private stack on the M5 (0.98-1.0x of it); it is kept for the RTX 3060, where private arrays live in local memory.
- **Against the gate:** BVH4 merged is at 1.28-1.32x, just under and over 1.3x on a loaded machine; the instanced tree fails it. Nothing here passes the gate as measured; the quiet rerun and the RTX 3060 decide.

The first provisional run, before the mesh stacks moved to module scope (each mesh walk call zero-filled a 256-byte stack), was 0.68 / 0.57 instanced surface and 1.27 / 0.95 merged surface: the fix moved those by 2-4%.

**Still to measure on a quiet M5 (under the GPU lock, load below half the cores), three full-profile runs per scene:** `trace-bench-surface-sponza` and `trace-bench-surface-bistro-interior` with `BVH_PAGE_QUERY='skip=hit-record,leaves,nodes,any-order'`. Bistro interior has not been timed at all.

**Windows (RTX 3060 Laptop): what to run**

```
BVH_ASSETS_DIR=<ddgi assets> BVH_MANIFESTS_DIR=<ddgi manifests> BVH_PAGE_QUERY='skip=hit-record,leaves,nodes,any-order' pnpm test:gpu:page trace-bench-surface-sponza --json wide-sponza-1.json
```
Run it three times per scene (`trace-bench-surface-sponza`, `trace-bench-surface-bistro-interior`), one page at a time with nothing else on the GPU, then `node tests/gpu/pages/trace-bench/surface-to-csv.ts wide-*.json --append`. Also run `pnpm test:gpu:page trace-oracle` (the wide variants are the gate on NVIDIA: 0 slips) and `pnpm test:gpu:page trace-wide-compile` (DXC compile times per kernel, against the binary kernels in the same run). The runbook (`WINDOWS-RUNBOOK.md`) has the commands with paths.

## 9. Levers not built (measured options, after the gate)

- **L1, the folded decode** (Ylitie's): one FMA per plane, with a padding by the absolute error bound of `(p - o) / d` to keep boxes conservative. It halves the box ALU; whether it keeps the binary walk's hits has to be measured on the shared-edge rays (B3's Intel leak).
- **L2, triangle-group postponing** (Ylitie §3.3): push the triangle group when few lanes of the warp want to test triangles. It needs subgroup ballots in uniform control flow.
- **L3, compressed triangles** (`CWBVH_COMPRESSED_TRIS`, Baldwin-Weber): not compatible with the watertight test, so ruled out by the hit contract.
- **L4, exact boxes for split leaves:** tinybvh's `SplitLeafs` copies the parent leaf's box; boxes from the triangles would cut CWBVH's extra triangle tests on Sponza.
- **L5, the collapse in the workers, and on the GPU** (section 2).
- **L6, BVH4's workgroup stack**, if the private stack turns out to be what CWBVH's workgroup stack fixes.
- **L7, a single-loop wide walk:** one loop and one stack for both levels with an instance marker, as the binary walk has, so lanes in different instances keep stepping. Straightforward for BVH4 (its stack entries are words, as the binary walk's are); CWBVH needs the top level's references packed into its two-word entries. The provisional instanced-tree numbers (section 8) point at this first.

## 10. Follow-ups

1. The quiet-machine M5 runs and the RTX 3060 runs (section 8, runbook 2d), then the gate.
2. L7 (single-loop BVH4 walk) and L1 (folded decode) as A/B settings, measured the same way.
3. GPU collapse after H-PLOC (so wide scenes get first trees without the worker), and a GPU wide refit (binary refit plus a repack pass).
4. Collapse in the workers.
5. If the gate passes: the folded decode (L1) and triangle postponing (L2) as measured levers, `serialize` for wide scenes, and `/three`'s `wideStack`.

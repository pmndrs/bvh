# Decisions before 0.1.0 (2026-11-15)

Seven open choices from HANDOFF.md ("For Dennis to decide", and the two provisional walk defaults). Each block says what
the choice is, what to pick, what it costs, and how hard it is to undo in 0.2. "Before 0.1.0" means it touches a file
format or a public type that people will build on; defaults and additive options can change in 0.2 with a changelog line.

| Item | Recommendation | Decide before 0.1.0 |
| --- | --- | --- |
| 1. SDF sign default | Keep `'parity'` | No |
| 2. `builtBy: 'app'`; `material` sentinel | Keep both | 2a no; 2b yes (confirm: public type) |
| 3. `deserialize` either format; morph re-bake | Keep both | No |
| 4. Scene file snapshot; JSON part; hash coverage | Keep all three | 4a no; 4b and 4c yes (confirm: file format) |
| 5. Points and lines CPU-only, own builder and file | Keep; bump the `BVHP` version by hand if node words change | No |
| 6. Adopt and quantized emit; `positionsBuffer` turns weld and merge off | Keep both | No |
| 7. `triangleLeaves: 'records'`, `quantizedDecode: 'exact'` | Ship them; both are safe to flip in 0.2 | No |

**Decided by Dennis, 2026-11-16:** every recommendation below is taken, with three notes.

- **1:** keep `'parity'`, but find out why three-mesh-bvh's SDF was 1.7-2.8x faster on SwiftShader and measure it on the
  RTX. Done: 1.3-1.5x on the RTX, all of it the sign; now 1.2-1.3x, plus an opt-in `'first-hit'` sign as fast as
  three-mesh-bvh's (`docs/tracing/sdf.mdx`, "Speed").
- **3a:** keep both formats, and document that `MeshBVH.deserialize` taking our bytes may be removed later (done: the
  JSDoc, `docs/three/bounds-tree.mdx` and the serialization page say so).
- **4a:** keep the snapshot, and add movable scene files as an option, with the scenarios written down. Done:
  `'PBVM'` files and "Which file?" in `docs/serialization/serialization-and-baking.mdx`.
- **5:** points and lines stay CPU-only for 0.1.0; Dennis sees them as a weakness to come back to.
- **7:** ship with `'records'` and `'exact'`. The README sample now passes the scene's formats to `traceModuleWgsl`, so
  code copied from it keeps matching if a default flips.

Numbers are from the RTX 3060 Laptop unless marked; M5 numbers are marked "indicative" where the design notes call them
that. The runbook is `docs/internals/WINDOWS-RUNBOOK.md`.

---

## 1. SDF default sign: `'parity'`

**The question.** `SdfGenerator`, `generateSdfCpu`, `generateSdfTexture` and `CpuMeshBvh.isPointInside` decide inside
by a three-ray vote. The vote can count crossings (`'parity'`, the default) or add up signed crossings (`'winding'`).
This only matters for SDF users whose meshes overlap, nest or have inconsistent winding.

**Recommendation.** **Keep `'parity'`, because it is right on any closed mesh whatever its winding, and `'winding'` is
one option away per call.**

| Option | Pros | Cons |
| --- | --- | --- |
| `'parity'` (now) | Right on closed meshes with either winding, and on inconsistently wound ones | Two closed meshes that overlap: the overlap reads as outside |
| `'winding'` | Right where instances overlap (winding 2) | Wrong where inconsistent winding cancels; three's imported meshes often have flipped faces |

**Performance / quality.** No runtime difference: the kernel computes both votes from the same three walks
(`bvhInsideVotes` returns `.x` parity and `.y` winding) and selects one by a uniform (`src/trace/sdf.ts`, `params.sign`).
The geometry table is in `docs/tracing/sdf.mdx`, "Why ray parity with a vote". Our signed voxel costs four walks against
three-mesh-bvh's two; on SwiftShader three-mesh-bvh's SDF was 1.7-2.8x faster (HANDOFF.md, "SDF (8.10)"). On the RTX:
not measured (runbook section 9, "Not covered").

**What others do.**
- three-mesh-bvh: its `webgpu_sdfGeneration` example signs by one first-hit ray along +Z and the side it hits, so it
  depends on winding and on that one ray (`examples/compare/sdf/main.ts`, header; the example is not in the npm package).
- three.js: no SDF generator.
- Babylon: no SDF generator in `@babylonjs/core` 9.35.

**Cost to change later.** A default value. Flipping it changes the field for overlapping or badly wound meshes only, in
five places (`SdfOptions.sign`, `generateSdfCpu`, `generateSdfTexture`, `isPointInside`, and the WGSL `bvhIsInside`,
which is documented as the parity vote). Easy in 0.2 with a changelog line.

**Decide before 0.1.0?** No.

---

## 2a. `builtBy: 'app'` in `TreeBuilder`

**The question.** `placeTree` lets the app place its own tree (a baked file, a worker's GPU device). Events and status
report it as `builtBy: 'app'`, a third member of `TreeBuilder = 'gpu' | 'tinybvh' | 'app'`
(`src/scene/scene-trees.ts`). App code that switches on `builtBy` or keys a record by it must handle the third value.

**Recommendation.** **Keep it, because it is already in `0.1.0-alpha.0` and a placed tree is neither of the other two.**

| Option | Pros | Cons |
| --- | --- | --- |
| `'app'` (now) | Says what happened; `counts.onAppTrees` matches it | Exhaustive switches over `TreeBuilder` need a third case |
| Report placed trees as `'tinybvh'` | Two values only | Wrong for a GPU-built bake, and hides that no build ran |

**Performance / quality.** No runtime cost.

**What others do.** three-mesh-bvh has no scene-level builder report; its `MeshBVH` is built or deserialized, and
nothing records which (`node_modules/three-mesh-bvh/src/core/MeshBVH.js`). three.js and Babylon have no BVH builder.

**Cost to change later.** Public type. Adding a member later breaks exhaustive switches; removing one breaks code that
names it. It shipped in the alpha (`5a61a86` is an ancestor of `9712461`), so keeping it is the cheap path. No code in
the repo indexes by `builtBy` with only two cases (checked `src/`, `examples/`, `compare/`, `bench/`).

One stale JSDoc: `SceneTreeEvent.replaced` says "`'gpu'` for a swap", but `placeTrees` reports whatever was there, so
`placeTree` over a tinybvh tree gives `replaced: 'tinybvh'`, and a drift rebuild over a placed tree gives `'app'`.

**Decide before 0.1.0?** No.

## 2b. `GpuRayHit.material` uses `NO_MATERIAL`

**The question.** A hit with no material reports `NO_MATERIAL` (`0xffffffff`) in `GpuRayHit.material`, the same value
as WGSL `BVH_NO_MATERIAL`, `CpuHit.material` and `BvhClosestHit.material` (`src/gpu-layout.ts`). The alternative is
`number | null` on the JS side only.

**Recommendation.** **Keep `NO_MATERIAL`, because one sentinel in WGSL, CPU hits and decoded GPU hits means code moves
between them unchanged.**

| Option | Pros | Cons |
| --- | --- | --- |
| `NO_MATERIAL` everywhere (now) | One value across WGSL, CPU walk, batch records; `materialCode` already accepts it as "none" on input | A JS user can index an array with `0xffffffff` and get `undefined` |
| `number \| null` in `GpuRayHit` only | Idiomatic JS; TypeScript forces the check | `CpuHit` and WGSL still use the sentinel, so two conventions |

**Performance / quality.** No runtime cost: the WGSL side is one select after the walk (`docs/internals/design/trace.md`,
"Hit normal and material"). The 40-byte record that carries it costs 6-12% on the RTX (runbook section 9: 0.88-0.94x the
24-byte record); that is a separate, settled default.

**What others do.**
- three.js: `face.materialIndex` is 0 when the mesh has no groups (`node_modules/three/src/objects/Mesh.js`, line 502);
  no "none" value.
- three-mesh-bvh `/webgpu`: no material in its hit struct (`rayIntersectionResultStruct`,
  `src/webgpu/tsl/structs.js`; `BVHComputeData.js` has "TODO: add material support").
- Babylon: `PickingInfo.subMeshId`, 0 by default; no sentinel.

**Cost to change later.** Public type: `number` to `number | null` breaks TypeScript users and any `=== NO_MATERIAL`
check. Cheap to confirm now, not cheap to change after.

**Decide before 0.1.0?** Yes, confirm it.

---

## 3a. `BvhBoundsTree.deserialize` takes either format

**The question.** `/three`'s `BvhBoundsTree.deserialize(data, geometry)` reads three-mesh-bvh's `SerializedBVH` object
and also our bytes (`src/three/raycast.ts`, `deserializeFor`). It tells them apart by type (an object against an
`ArrayBuffer` or view), not by sniffing contents.

**Recommendation.** **Keep it, because three-mesh-bvh code keeps working and the bytes branch can't collide with it.**

| Option | Pros | Cons |
| --- | --- | --- |
| Either format (now) | Drop-in for `MeshBVH.deserialize`; our files load through the same call | One method, two input kinds |
| three-mesh-bvh's format only; ours via `deserializeBoundsTree` | Narrower contract | Users must know which function matches which file |

**Performance / quality.** No runtime cost (one `instanceof`). Reading three-mesh-bvh's format rebuilds the tree's
shape with exact boxes (`docs/internals/three.md`, the `MeshBVH.serialize` row).

**What others do.** three-mesh-bvh's `deserialize` takes only its object, with `setIndex` and a version 0 fix-up
(`node_modules/three-mesh-bvh/src/core/MeshBVH.js`, lines 160-185). three.js and Babylon have no BVH file.

**Cost to change later.** Public API. Removing the bytes branch would break callers; there is no reason to.

**Decide before 0.1.0?** No.

## 3b. `StaticGeometryGenerator` re-bakes on morph-weight-only changes

**The question.** Our port's `didChange` also compares `morphTargetInfluences` (`src/three/static-geometry.ts`).
three-mesh-bvh 0.9.18's doesn't, so a mesh whose only change is a morph weight keeps its old baked positions. Affects
three users baking morphed meshes for CPU queries.

**Recommendation.** **Keep the re-bake, because three-mesh-bvh's behaviour returns stale geometry, and the output is
otherwise bit for bit equal to its.**

| Option | Pros | Cons |
| --- | --- | --- |
| Compare influences (now) | Correct after a morph-only change | Differs from three-mesh-bvh on that one case |
| Match three-mesh-bvh exactly | Same skip behaviour | Stale positions until something else changes |

**Performance / quality.** One short array compare per mesh per `generate`. Not measured.

**What others do.** three-mesh-bvh: `didChange` checks `matrixWorld`, the position version, bone matrices and the
primitive count only (`node_modules/three-mesh-bvh/src/utils/StaticGeometryGenerator.js`, lines 442-456). three.js
re-poses on every `getVertexPosition` call, so it never goes stale.

**Cost to change later.** Behaviour only; one line. Worth an upstream issue to three-mesh-bvh.

**Decide before 0.1.0?** No.

---

## 4a. Scene files are snapshots

**The question.** A `'PBVS'` scene file holds the assembled scene with its top level baked, so its instances can't
move, and merged meshes hold their instances' transforms (`src/serialize-scene.ts`, file comment). Apps with moving
instances bake one tree per mesh and place them in `SceneTrees` (`docs/serialization/serialization-and-baking.mdx`).

**Recommendation.** **Keep the snapshot, because the per-mesh path already covers moving scenes and a movable scene file
can be added later as its own kind.**

| Option | Pros | Cons |
| --- | --- | --- |
| Snapshot (now) | Loads straight to `uploadTraceScene`, no build or assembly; same bytes back out | Instances fixed |
| Store mesh trees and instances, rebuild the top level at load | Instances can move | A top-level build at load; merged groups would have to be dropped or re-merged |

**Performance / quality.** Load is a copy. Not measured.

**What others do.** three-mesh-bvh serializes one geometry's tree only, no scene (`MeshBVH.serialize`). three.js and
Babylon have no BVH file; Babylon's picking octree is built at run time (`@babylonjs/core/Culling/Octrees`).

**Cost to change later.** Additive: a movable scene file would be a new magic or format version, and today's files keep
loading.

**Decide before 0.1.0?** No.

## 4b. The scene file's description part is JSON

**The question.** Between the 16-word header and the buffers, a scene file stores the placements and top-level stats as
JSON padded to a word (`src/serialize-scene.ts`). The alternative is fixed binary records.

**Recommendation.** **Keep JSON, because the description is small next to the buffers and new optional fields
(`clustered`, `welded`) can be added without moving a word.**

| Option | Pros | Cons |
| --- | --- | --- |
| JSON (now) | Readable; optional fields cost nothing; the same scene always writes the same bytes (`placementJson` fixes field order) | A parse at load; a reader must ignore unknown fields to stay forward-compatible |
| Binary records | No parse; fixed size | Every new field is a format version |

**Performance / quality.** Not measured; the description is a few hundred bytes per mesh against megabytes of nodes.

**What others do.** three-mesh-bvh's serialized form is a JS object (`version`, `roots`, `index`, `indirectBuffer`), not
a file format. glTF uses the same split (JSON plus binary buffers).

**Cost to change later.** File format: switching to binary means `SERIALIZED_SCENE_FORMAT_VERSION` 2, and old files
are refused, not converted (by design). Confirm now.

**Decide before 0.1.0?** Yes, confirm it.

## 4c. The geometry hash covers positions and indices only

**The question.** A tree file written with `{ geometry }` stores a 64-bit hash of positions (their bits) and indices
(their values) in header words 14-15 (`meshGeometryHash`, `src/serialize.ts`). Loading next to other geometry throws.
Changed triangle flags or materials are not caught.

**Recommendation.** **Keep positions and indices, because they decide the tree's shape and an indexed file's
correctness, and the loader often has no flags or materials to hash (three's `BufferGeometry` has none).**

| Option | Pros | Cons |
| --- | --- | --- |
| Positions and indices (now) | Catches the mistake that traces wrong triangles; works from any geometry | A changed material number loads silently (wrong material, right hit) |
| Also hash flags and materials | Catches stale material numbers | The loader needs them; three geometry doesn't have them, so the check gets skipped where it matters most |

**Performance / quality.** One pass over positions and indices at write and at a checked load. Not measured.

**What others do.** three-mesh-bvh has no hash or check; `deserialize` trusts the geometry it is given. three.js and
Babylon: no BVH file.

**Cost to change later.** File format. The tree header's 16 words are all used, so changing what the hash covers means a
format version bump, and old bakes would be refused. Confirm now. (Scene files carry no hash: an indexed scene holds its
own positions.)

**Decide before 0.1.0?** Yes, confirm it.

---

## 5. Points and lines: CPU-only "have", own builder and file format

**The question.** PARITY rows 3.2 and 3.3 count `PointsBVH` and the line BVHs as "have" with CPU trees only.
`CpuPointsBvh` and `CpuSegmentsBvh` use a centre split in TypeScript instead of tinybvh, and their own `"BVHP"` file
(`src/query/primitive-bvh.ts`). Affects three users raycasting `Points` and lines.

**Recommendation.** **Keep all three, because three-mesh-bvh's are CPU-only too, SAH is the wrong cost for points, and
nobody surveyed calls them on the GPU.**

| Option | Pros | Cons |
| --- | --- | --- |
| CPU, centre split, `BVHP` (now) | About 27 bytes per point (43 per segment); no WASM to load; matches three's `Points.raycast` field for field | A second builder and a second file format to keep |
| Points as degenerate triangles through tinybvh | Every query and the triangle file for free | 49,915 nodes for 50,000 points (SAH splits zero-area boxes to one per leaf): about 112 bytes per point |
| Also a GPU walk | GPU picking of points | A parallel GPU path with no caller; scenes are triangle-only throughout |

**Performance / quality.** Build about 0.9 s for a million random points, the same as three-mesh-bvh's `PointsBVH`
(Node 24 on the Windows laptop under load). Memory figures above. Source: `docs/internals/design/points-and-lines.md`,
"Why a centre split and not tinybvh".

**What others do.** three-mesh-bvh: `PointsBVH` and `LineBVH` are CPU, with `CENTER` as its default split, and no
`serialize` (only `MeshBVH` has one; `src/core/PointsBVH.js`, `LineBVH.js`). Its `/webgpu` packs triangles only
(`BVHComputeData.js`, "TODO: Add support for other geometry types"). three.js: `Points.raycast` and `Line.raycast` are
linear scans. Babylon: no point or line BVH.

**Cost to change later.** The "have" label is docs. The `BVHP` format has its own version word (`FILE_VERSION` 1) and
two free header words (14-15), so a hash can be added later without a break. One trap: `BVHP` stores the GPU layout's
64-byte nodes but not `GPU_LAYOUT_VERSION`, and `readSerializedHeader` / `isSerializedCurrent` don't recognise it. If the
node words ever change, bump `FILE_VERSION` by hand.

**Decide before 0.1.0?** No.

---

## 6a. Adopt in quantized scenes needs the quantized emit

**The question.** With `gpuTreeOutputs: 'adopt'`, the scene keeps H-PLOC's batch buffers as storage, so in a quantized
scene the batch must already hold quantized nodes. HANDOFF describes this as "adopt turns on the quantized emit". The
code has moved on: since 2026-11-15 `gpuQuantizeOnEmit` defaults to true for every quantized scene, copy or adopt, and
`adopt` with an explicit `gpuQuantizeOnEmit: false` throws (`src/scene/scene-trees.ts`, constructor). So the open
question is only whether that explicit conflict should throw or be overridden.

**Recommendation.** **Keep it as built (emit on by default, the conflict throws), because the emit writes the same nodes
word for word and a silent override would hide a contradictory option set.**

| Option | Pros | Cons |
| --- | --- | --- |
| Throw on `adopt` + `gpuQuantizeOnEmit: false` (now) | Options mean what they say | One more error to read |
| Silently turn the emit on | Never throws | `false` is ignored without a word |

**Performance / quality.** Runbook section 10 (Sponza): the emit builds in 35.1 ms against 41.6 for the `f32` build
plus the quantize pass; uncapped peak 1,482.6 to 1,307.3 MB; at the default cap the peak is the same (613.2 and 613.5).
Quantized adopt settles at 438.3 MB against 366.5 for copy (uncapped), which is why `'copy'` stays the default. Nodes
identical: `bvh-hploc` checks word for word (passed on both GPUs, section 9).

**What others do.** three-mesh-bvh builds on the CPU and packs `f32` nodes (64 bytes, `GPU_BYTES_PER_NODE`,
`src/webgpu/utils/packBVHBufferUtils.js`); there is no GPU emit to compare. three.js and Babylon: no GPU builder.

**Cost to change later.** An option default and an error. Easy either way in 0.2; the nodes are the same, so no data
changes.

**Decide before 0.1.0?** No.

## 6b. `positionsBuffer` turns weld and merge off

**The question.** `SceneTrees({ positionsBuffer })` binds the app's own `f32` positions instead of a scene copy (indexed
scenes only). The trees must then name the app's vertices, so the scene doesn't weld, and merged groups' world-space
vertices aren't in the app's buffer, so static instances don't merge. Both default to off when the buffer is given, and
an explicit `true` throws. Deforming meshes throw too.

**Recommendation.** **Keep it, because both follow from binding the app's buffer, and the option stays opt-in.**

| Option | Pros | Cons |
| --- | --- | --- |
| Defaults flip off, explicit `true` throws (now) | Works with one option; contradictions are errors | Loses the merge speed-up silently |
| Require `weld: false, mergeStaticInstances: false` spelled out | The cost is visible at the call | Three options for one feature |

**Performance / quality.** Saves the scene's positions copy: 45.2 MB on Sponza with `vec4`
(`docs/internals/design/indexed-leaves.md`, "Status", bullet "The app's own positions"). Loses merging, which is
1.44-1.72x closest and 1.23-1.58x any on Sponza, 1.19-1.20x and 1.17-1.36x on Bistro interior (runbook section 9). That
trade is why it stays opt-in (HANDOFF "Defaults after the clean run").

**What others do.** three-mesh-bvh `/webgpu` copies each geometry's index and attributes into its own buffers, deduped
per geometry (`BVHComputeData.js`, lines 311-438); it never binds the app's. three.js can't offer its attribute buffers
for this: plain attributes have no `STORAGE` usage (indexed-leaves.md, section 3). Babylon: no GPU BVH.

**Cost to change later.** Option behaviour. Making it stricter later (require the explicit `false`s) would break callers;
loosening is free. Keep the lenient form.

**Decide before 0.1.0?** No.

---

## 7. The provisional walk defaults: `triangleLeaves: 'records'` and `quantizedDecode: 'exact'`

**The question.** Both were kept after the RTX clean run and wait on the M5 timing sections 8 and 9 of the surface-ray
page (HANDOFF "Open"). Dennis may ship 0.1.0 first. Is that safe, and can they flip in 0.2?

**Recommendation.** **Ship 0.1.0 with both, because both are the faster choice on the RTX today, neither writes a file
format, and the library already checks formats at run time where it makes both sides.**

### 7a. `triangleLeaves: 'records'`

| Option | Pros | Cons |
| --- | --- | --- |
| `'records'` (now) | Fastest closest hits on the RTX; no 65,536-source cap on merged groups | 130 MB more on Sponza than indexed |
| `'indexed'` `vec4` | Sponza settles at 236.4 MB against 366.5 (uncapped, quantized) | Closest hits 0.83-0.95x (Sponza), 0.89-0.93x (Bistro); merged groups capped at 65,536 sources (DIVERGENCES A2); welds positions by default |

**Performance / quality.** RTX, quantized nodes (runbook section 9, section 8 row): indexed `vec4` closest 0.83-0.95x,
any 0.95-1.03x on Sponza; packed 0.77-0.87x. Memory: runbook section 10 (366.5 to 236.4 MB settled, uncapped). M5 on
`f32` nodes (indicative, HANDOFF "Indexed leaves"): `vec4` 0.83-0.99x, packed 0.75-0.92x. M5 on quantized nodes: not
measured (the pending run). Both formats agree on every ray (section 8, `different` 0).

**Is shipping safe and reversible?** Yes, with one caveat.
- Files say their format in the header, so old bakes load after any flip (`src/serialize.ts`).
- `SceneTrees`, `assembleTraceScene`, `TraceBatchKernel` and `BvhTraceNodes` default together, and kernels refuse buffers
  of the other format (`traceSourceFormats`, `src/trace/trace-gpu.ts`).
- The caveat: hand-written WGSL that calls `traceModuleWgsl()` with no options and binds a scene it didn't make with
  defaults (a records scene file, an explicit option elsewhere) has no check, and would read the wrong layout after a
  flip. Several docs snippets call `traceModuleWgsl()` bare (`docs/tracing/wgsl-walk.mdx`, `docs/snippets/readme.ts`).
  Before 0.1.0, it would help to show passing the scene's formats (`trees.triangleLeaves`, `trees.nodeFormat`), as
  `docs/tracing/sdf.mdx` already does. The `nodeFormat` flip on 2026-11-13 carried the same risk.
- Flipping also brings the 65,536-source cap and default welding to apps that never chose indexed leaves. Changelog both.

### 7b. `quantizedDecode: 'exact'`

| Option | Pros | Cons |
| --- | --- | --- |
| `'exact'` (now) | Same walk text as before the option; no padding | 0-6% slower on merged trees (RTX) |
| `'folded'` | 1.00-1.06x on merged trees (the default kind) | 0.93-0.97x on instanced trees; refused with `f32` nodes and with `anyHitOrder: 'larger'` |

**Performance / quality.** RTX (runbook section 9): folded over exact, instanced 0.93-0.97x on Sponza and 0.94-0.97x on
Bistro; merged 1.00-1.06x and 1.00-1.05x; indexed 0.92-0.99x and 1.00-1.08x. Folded boxes are padded so they always hold
the exact box (`docs/internals/design/compressed-nodes.md`, question 6); `trace-oracle` 0 mismatches. M5: not measured.

**Is shipping safe and reversible?** Yes, fully. It is a walk-only compile option: no buffer, file or scene changes, and
hits are the same. The one thing to handle in a flip: `'folded'` throws with `f32` nodes and `anyHitOrder: 'larger'`, so
a folded default would have to fall back to exact there rather than throw.

### What others do (both)

three-mesh-bvh `/webgpu` binds an index and its own copy of the attributes (indexed, in effect) with 64-byte `f32` nodes
(`BVHComputeData.js`; `packBVHBufferUtils.js`, `GPU_BYTES_PER_NODE = 64`). Its Sponza footprint beyond geometry is about
51 MiB against our 355.5 with quantized records (`docs/internals/design/memory.md`, line 55: 6.9x). three.js and Babylon
have no GPU walk.

**Cost to change later.** Defaults, not formats. 7b is free to flip; 7a is safe for every path that goes through the
library's own kernels, and needs a changelog note for hand-written WGSL.

**Decide before 0.1.0?** No. Ship, and settle both after the M5 run.

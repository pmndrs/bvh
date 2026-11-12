# BVH walk and GPU layout

This is the part of ddgi's tracer design (the T0.10 spike) that belongs to the BVH: the GPU layout the builders write, the WGSL walk that reads it, the watertight triangle test, and the oracle and CPU walk that check them. How the builders produce the layout is in [bvh-builders.md](bvh-builders.md). The code is in `src/trace/` (`layout.ts`, `wgsl.ts`, `cpu-traversal.ts`, `oracle-compare.ts`, `oracle-fixtures.ts`) and `src/gpu-layout.ts`. Probe tracing and the GI scheduling that sit on top of it stay in webgpu-ddgi.

## In one paragraph

Each mesh gets its own bounding-volume hierarchy (a "BVH": a tree of boxes that lets a ray skip most triangles). A second, small tree over all object instances sits on top. On the GPU, a WGSL function walks the small tree, and for each object the ray might hit, it moves the ray into that object's local space and walks the object's tree. There are two versions of the walk: "closest hit" (what does this ray hit first?) and "any hit" (is anything in the way?). The second stops at the first blocker, which makes shadow rays cheap.

## Data layout

Everything is plain arrays in a few storage buffers, so it survives a worker hand-off (transferable `ArrayBuffer`s) and an upload without conversion. `GPU_LAYOUT_VERSION` identifies the layout.

### Per-mesh tree ("bottom level")

- **Node: 64 bytes, two child boxes per node** (the Aila-Laine layout used by most GPU BVH2 tracers). One node read gives both children's boxes, so the walk tests both and visits the nearer one first without a second memory fetch.

  ```
  struct BvhNode {            // 64 bytes
      leftMin:  vec3f, leftIndex:  u32,
      leftMax:  vec3f, rightIndex: u32,
      rightMin: vec3f, leftCount:  u32,   // count > 0: child is a leaf with `count` triangles starting at index
      rightMax: vec3f, rightCount: u32,
  }
  ```
  A child with count 0 is an inner node at `index`. A leaf child stores the first triangle and the number of triangles.
- **Triangles: 48 bytes each, stored in tree order** (leaves point at contiguous runs, so no index indirection): `v0` + original triangle index, `v1` + flags, `v2` + pad. The three corners are stored as they are, so a vertex shared by two triangles has identical bits in both; the watertight test needs that. The original index maps a hit back to the mesh's UVs and material group. Flags bit 0 (`TRIANGLE_FLAG_FILTER`) sends the triangle to the triangle filter (below). Flags bit 1 (`TRIANGLE_FLAG_FLIP_FRONT_FACE`) flips the hit's `frontFace`, and the padding word holds the source instance index; both are only used by merged static meshes (below) and are 0 as every builder writes them. Flags bits 2-15 are reserved, 0; bits 16-31 hold the triangle's material (material + 1, 0 for none): the mesh input's `triangleMaterials` (0-65534 per original triangle, `TRIANGLE_MATERIAL_NONE` 0xffff for none), which every builder writes, CPU and GPU (the GPU builders get it in the flags the CPU uploads), or a merged triangle's source's, which `assembleTraceScene` writes. Refits leave the word as it is. Input flags (`MeshInput.triangleFlags`) may set bit 0 only: every builder, `buildTraceScene` and `mergeStaticGeometry` throw, naming the mesh and triangle, for any of bits 1-31, so a material goes in through `triangleMaterials` only. A GPU flags buffer (`triangleFlagsBuffer`) is not checked.
- **Indexed leaves (`triangleLeaves: 'indexed'`, off by default; [indexed-leaves.md](indexed-leaves.md)).** The other triangle format: 16 bytes per triangle in tree order, `vec4u(i0, i1, i2, id)`, word 3 holding the original index in bits 0-27, the filter flag in bit 28 and the flip flag in bit 29. The corners come from a positions buffer the scene owns (`vec4f` per vertex by default, or packed `f32` with `positionLayout: 'packed'`), welded by exact bits at load (no coordinate changes, so a shared vertex still has the same bits in every triangle). Vertex indices are the mesh's own; a 16-byte header before each mesh's triangles (`triangleOffset - 1`) holds its vertex base, read once per instance entered. A merged group keeps its source instance and material in a side table before the header (4 bytes per triangle) instead of words 11 and 16-31; another mesh with `triangleMaterials` keeps a material table there (16 bits per original triangle, header word 3 its distance, 0 for none). Serialized trees name the format in header word 7. It is a variant of layout version 2.
- **Quantized nodes (`nodeFormat: 'quantized'`, off by default; [compressed-nodes.md](compressed-nodes.md)).** The other node format: 32 bytes per node, two `vec4u`, one per child: three axis words (the min plane in bits 0-15, the max plane in bits 16-31) and the child's stack reference (index in bits 0-27, leaf count in 28-31; a missing right child is `0xffffffff`). Planes are step counts on a per-mesh grid, the frame: a power-of-two step and an origin on it per axis, chosen from the root's box so that every plane `origin + q * step` is an exact `f32`. The frame sits in a 32-byte slot right before the mesh's root (`rootNodeOffset - 1`; a tree's `nodes` starts with it), read once per instance entered. Boxes round outward with a 1/1024-step margin, so a decoded box always contains the exact one and the triangle test (unchanged) decides every hit. The top level stays `f32`. Deforming meshes refit a 64-byte `f32` copy and a pass re-quantizes it. Serialized trees name the format in bits 8-15 of header word 7. It is a variant of layout version 2.
- **Leaves:** at most 4 triangles in the tracer's own builder, fewer where the SAH cost says so. The builders take a largest-leaf option.
- **Deforming meshes** keep the tree's shape. Refitting bottom-up needs each node's parent, so the builder also writes a parent index per node (4 bytes per node, kept in a separate array that only the refit pass uses). Without subgroups there is no GPU refit: `SceneTrees.setPose` is skipped (counted, and warned once), and `SceneTrees.setPoseCpu(mesh, positions)` refits on the CPU instead (`refitLayoutToPose`), with or without subgroups, and copies the nodes (re-quantized for quantized nodes) and corners (indexed: vertices) in through the update's encoder; the top-level box becomes the posed root box.

All meshes' nodes go in one node buffer, and all triangles in one triangle buffer, at per-mesh offsets.

### Instances and the top level

- **Instance record: 64 bytes.** It holds the world-to-object transform as a 3x4 matrix (48 bytes), the mesh's root node offset, its triangle offset, the instance index, and flags. Flags bits 0-3 are `contributesToGi`, `visible`, `mirrored` and `merged` (below); bits 8-15 hold the 8-bit instance mask (as in DXR; `0xff`, visible to every ray, unless the instance gives one); bits 16-31 hold the instance's material index (`material + 1`, 0 for none; `MATERIAL_SHIFT`; a merged record leaves them 0). Layout version 2 added the mask, and before its release the `merged` bit, the meaning of the triangle's padding word and flags bit 1, and the material bits; nothing moved. Rays are moved into object space, so moving an object never touches its tree. Mirrored and non-uniform scales work because the hit distance is measured on the object-space ray, whose direction we deliberately *don't* renormalize, so `t` stays in world units.
- **Top-level tree:** the same 64-byte node layout, over instance world boxes. Leaves point at instance records. It can be built on the CPU with binned SAH (`src/trace/top-level.ts`), or, on a device with subgroups, refitted on the GPU each frame something moves with a new shape built in a worker now and then (`GpuTopLevel` in `src/trace/top-level-gpu.ts`, over `GpuBvhRefit`; the declared fallback without subgroups is the CPU rebuild). The walk's WGSL is the same either way. `GpuTopLevel` keeps its own instance table on the GPU (object-to-world rows, world-to-object rows inverted on the CPU in doubles so the records are the CPU top level's bits, mesh slot, instance index, flags); its gather writes each record's flags as packed on the CPU from the instance input: the mask in bits 8-15 (`0xff` by default) and the material in bits 16-31 (`materialCode`, `material + 1`), as `buildTopLevel` writes them. A merged group's record keeps `instanceIndex` `0xffffffff` and bit 3, and `setTransform` (or `setInstance` with another transform) throws for it. `tests/gpu/pages/bvh-top-level-gpu` checks the gathered records bit for bit against `buildTopLevel` and the walk against the CPU-assembled scene.
- **Buffers:** instance records and top-level nodes share one buffer (records first, after a 64-byte header). That gives three storage buffers for tracing: nodes, triangles, and top level. This matters because a default device allows only 8 per shader stage.

### Size limits

A default device caps one storage binding at 128 MB. At 48 bytes per triangle that's about 2.7 M triangles in one binding (8 M with indexed leaves, 16 bytes each, with the positions in a binding of their own). Sponza (about 262k triangles) fits easily. Bistro exterior (about 2.8 M) doesn't. Nothing is split across bindings: the layout keeps its 48-byte triangle records, because reading corners through an index buffer instead cost 10-30% of trace speed. Indexed leaves (above), which read one `vec4u` and then the positions, measured 0.83-0.99x records' rate on the M5 with `vec4` positions (indexed-leaves.md, "Measured"). The host asks for the adapter's largest storage binding and buffer (`requiredLimits`), and throws a limit error naming the limit and the size needed on a device that can't hold the scene.

## The walk (WGSL)

- **One loop, one small stack per thread** (`array<u32, 64>` in private memory; three-mesh-bvh uses 60), shared by both levels. Entering an instance pushes a marker, switches the ray to object space and continues with that mesh's root. Popping the marker switches back to the world-space ray.
- **Child order:** both child boxes are tested (slab test with a precomputed inverse direction). When both are hit, one is pushed and the walk goes to the other. Boxes beyond the current closest hit are skipped.
  - **Closest hit** goes to the nearer child first.
  - **Any hit** goes to the *farther* child first. A shadow ray starts on a surface, so the nearer child is mostly the geometry it starts on, and what blocks it lies ahead. Any blocker will do. This made any-hit 1.1-2.8x faster on the bench scenes, with the same answers.
- **Closest hit** keeps the nearest `t`, instance, triangle and barycentrics, and whether the hit was on the front face. Back-face hits count as hits.
- **Front face means the side the mesh was authored as front,** also through a mirrored transform. The triangle test runs in object space, where the winding is the authored one, so no flip is needed. The exception is a merged static mesh, which is tested in world space: its triangles from mirrored sources carry flags bit 1, which flips the result.
- **Any hit** returns at the first triangle hit within `[tMin, tMax]`.
- **Triangle test: watertight** (Woop, Benthin and Wald 2013), with per-ray shear rows computed once per ray. It replaced Moller-Trumbore, which let rays slip through shared edges (below).
- **Stack overflow:** the builders report the maximum depth. The top-level and mesh depths together must fit in the stack (47 levels are left for a mesh tree under a top level; a deeper GPU-built tree is refused).
- **Hit records** are written with a `u32` front-face flag, because buffers can't hold `bool`. A single-triangle tree has an empty right child.

### The public WGSL API

`traceModuleWgsl({ traceSlot, triangleFilter, rayMasks, triangleLeaves, positionLayout, nodeFormat })` returns the module. Everything it declares starts with `bvh`, `Bvh` or `BVH_`. With `triangleLeaves: 'indexed'` it declares a fourth binding after the top level, `bvhTracePositions` (`TRACE_MODULE_ENTRIES_INDEXED`, `traceModuleEntries(options)`); only the triangle loading differs, the walk is the same text. With `nodeFormat: 'quantized'` the bindings are the same (`bvhTraceNodes` is read as `array<vec4u>`); only `bvhLoadMeshNode` and the frame read on instance entry (`bvhEnterMeshNodes`, into two private variables) differ, and the walk is the same text again (a unit test compares it).

```
struct BvhRay { origin: vec3f, tMin: f32, direction: vec3f, tMax: f32 }   // 32 bytes
struct BvhHit { t: f32, instance: u32, triangle: u32, barycentrics: vec2f, frontFace: bool,
                material: u32, normal: vec3f }
fn bvhTraceClosest(ray: BvhRay) -> BvhHit
fn bvhTraceAny(ray: BvhRay) -> bool
fn bvhTraceClosestMasked(ray: BvhRay, mask: u32) -> BvhHit   // only with rayMasks (the default)
fn bvhTraceAnyMasked(ray: BvhRay, mask: u32) -> bool         // only with rayMasks

struct BvhCandidate { instance: u32, triangle: u32, t: f32, barycentrics: vec2f,
                      frontFace: bool, query: u32, rayMask: u32 }
fn bvhTrianglePasses(candidate: BvhCandidate) -> bool         // supplied by the caller
```

- **Hit normal and material.** `BvhHit.normal` is the world-space geometric normal of the authored front side, unit length (0 on a miss), so `frontFace == dot(normal, ray.direction) < 0`. It is worked out once per closest hit, after the walk (`bvhFinishClosest`), from the instance record and the triangle slot the walk kept, not per triangle test: for a normal instance `cross(v1 - v0, v2 - v0)` in object space, moved to world space by the transpose of the world-to-object rows (the inverse transpose of object-to-world; a mirror keeps the authored side); for a merged triangle the same cross of its world-space corners, negated when flags bit 1 is set. Geometric only: the layout has no vertex normals, so an interpolated normal needs the caller's own attributes (`triangle` and `barycentrics`). `BvhHit.material` is the triangle's material when it has one (flags bits 16-31; indexed: the mesh's material table or a merged group's side table), else the instance's `material` (`TraceInstance.material`/`TraceSceneInstance.material`, 0-65534, else a `RangeError`, instance flags bits 16-31; a merged record has none), else `BVH_NO_MATERIAL` (0xffffffff, `NO_MATERIAL`), which a miss reports too: one select in `bvhFinishClosest` (`hitMaterialCode` on the CPU), nothing in the loop. A merged triangle's material is its source triangle's, else its source instance's. It adds no binding and two loads per ray, after the walk. Its speed cost, measured on the Apple M5 (surface-ray pages, paired rounds with the order reversed; indicative only): none. The 40-byte record's closest-hit rate over the 24-byte record's had medians of 1.02 to 1.11 and a range of 0.96 to 1.15 (details in [memory.md](memory.md)); the default batch record grew from 24 to 40 bytes, and the surface-ray rows measured before this change are at `c36b263`. `batchKernelWgsl(mode, { hitNormal: false })` keeps the 24-byte record, without material.
- **Triangle filter.** A triangle whose flags bit 0 is set asks `bvhTrianglePasses` before its hit is taken; true means the ray passes through, and then nothing is recorded and the closest distance stays, so what lies behind is still found. Unflagged triangles never ask, so a scene pays only on the triangles that need it. `query` is `BVH_QUERY_CLOSEST` (0) or `BVH_QUERY_ANY` (1), so one filter can, say, let shadow rays through a surface that closest-hit rays still see. `rayMask` is the trace's mask, all ones for the plain functions or with masks compiled out. The candidate is a struct so fields can be added without breaking filters. `triangleFilter` defaults to `DEFAULT_TRIANGLE_FILTER_WGSL` (every triangle blocks); `null` leaves it for the kernel to declare.
- **Ray masks.** A masked trace skips an instance whose `(instanceMask & rayMask) == 0` at the top level, before entering its tree. The plain functions mean all ones, so **mask 0 is never hit**, by the plain functions too, unless masks are compiled out (`rayMasks: false`, where every instance is visible). The default mask written by `buildTopLevel` is `0xff`. The test reads the flags word the instance load already fetches, so it costs one shift, AND and compare per instance leaf (not per triangle).
- **`rayMasks: false`** compiles the mask code out of the same walk (no second walk): no `*Masked` functions, no mask parameter or test, instance masks ignored, `rayMask` all ones in the candidate. Nothing changes size in either mode: `BvhRay` stays 32 bytes and instance records 64.
- **Batch kernel.** `batchKernelWgsl(mode, { traceSlot, triangleFilter, rayMasks, hitNormal })`: rays and results in group 1, and with ray masks one mask for the whole batch, a uniform at group 1 binding 2 (`BATCH_PARAMS_BYTES`). There is no per-ray mask buffer. Closest-hit records:

  | `hitNormal` | Record | Bytes | Words |
  | --- | --- | --- | --- |
  | `true` (default) | `BvhHitRecord` (`HIT_RECORD_BYTES`, `HIT_RECORD_WORD`) | 40 | 0 `t`, 1 `instance`, 2 `triangle`, 3 `frontFace` (0/1), 4-5 `barycentrics`, 6 `material`, 7-9 `normal` x, y, z (three f32, so no padding) |
  | `false` | `BvhHitRecordCompact` (`HIT_RECORD_COMPACT_BYTES`, `HIT_RECORD_COMPACT_WORD`) | 24 | 0-5 as above, nothing else |

  The 24-byte record has no padding, and a seventh word would round it up to 32 (the `vec2f` aligns it to 8), so the compact record carries no material: `instance`, which is the source instance on a merged hit too, names it in the caller's own table. `hitRecordBytes(hitNormal)` gives the size. Any-hit results are one u32 either way.
- **CPU walk and reference.** `createCpuTracer(scene, { triangleFilter, rayMasks })` with `closest(ray, { mask, stats })` and `any(ray, { mask, stats })`; `buildReferenceScene(meshes, instances, { triangleFilter, rayMasks })` with `traceClosestReference(scene, ray, { mask })` and `traceAnyReference(scene, ray, { mask })`. Both report `material` and `normal` as the WGSL does (the reference from the world-space winding, reversed for a mirrored instance), and `compareClosest` compares them when the tracer gives them (material equal, normal cosine >= `NORMAL_COSINE_TOLERANCE`, 0.9999). The filter is a `TrianglePasses`, taking a `TriangleCandidate` with the WGSL struct's fields. The reference asks it about every triangle it really hits (it has no flags), so test filters should give one answer per triangle: f32 and f64 disagree right at a decision boundary inside a triangle.

## Merged static instances (`mergeStaticInstances`, off by default)

Every instance box a ray enters costs a transform, a marker push and a descent into that mesh's tree. In scenes made of one instance per mesh (Sponza, Bistro), the boxes overlap heavily, and rays that start on surfaces pay for it: on Sponza a surface origin sits inside 5.8 instance boxes, and a ray enters 9.8 instances, 8.1 of them without a hit. Merging puts instances that never move into one world-space tree under an identity transform, so such rays walk one tree.

**What merges.** An instance with `static: true` whose mesh no other instance uses and has triangles, whose transform can be inverted, and that is visible and contributes to GI. Hidden and non-GI instances stay normal instances: the top level skips them either way, so the scene traces as with the option off. Merged instances are grouped by mask, and each group becomes one mesh with one identity-transform instance record. Merged sources must have distinct `instanceIndex` values (a hit names its source by it); `mergeStaticGeometry` throws otherwise. Everything else stays a normal instance, in the same top level. A material index is not a grouping key: each merged triangle carries its source's material in flags bits 16-31.

**API.**
- `buildTraceScene(geometries, instances, { build, mergeStaticInstances })` builds and assembles. `build` is any builder (`buildMeshBvhWithTinyBvh`, a worker builder's `build`, a GPU layout).
- Lower level: `mergeStaticGeometry(meshes, instances)` returns a plan. Each group has `geometry`, a normal mesh input with world-space positions and `Uint32Array` indices, plus per-triangle `sourceInstances`, `sourceTriangles` and `flipFrontFace` beside it. Build each group's geometry with any builder, then call `assembleTraceScene(meshes, instances, topLevelOptions, { merged: { plan, meshes: groupTrees } })`. Meshes that only merged instances use need no tree of their own (`plan.meshNeeded`, pass `null`).
- With the option off, or without `merged`, the scene is byte for byte what it was before the option existed (a unit test pins `main`'s hash).

**Exact hits with no memory growth.** The builders never see the ids. Assembly writes them into its copy of the built records: `originalIndex` becomes the triangle's index in its source mesh, the padding word (word 11) the source instance index, flags bit 1 is set for a mirrored source, because world space reverses that source's winding, and flags bits 16-31 get the source's material + 1. The source's filter flag (bit 0) is kept. The record of a group has `INSTANCE_FLAG.merged` (bit 3) and `instanceIndex` `0xffffffff`. The walk then does, per hit:

```wgsl
let hitInstance = select(instance.instanceIndex, triangle.padding, (instance.flags & BVH_INSTANCE_FLAG_MERGED) != 0u);
let frontFace = triangleHit.frontFacing != ((triangle.flags & BVH_TRIANGLE_FLAG_FLIP_FRONT_FACE) != 0u);
```

`BvhHit.instance`, `BvhCandidate.instance` (the filter sees the source instance) and `frontFace` come out as for the instanced scene; the CPU walk does the same. The reference needs no change: it traces the source instances in world space. Barycentrics are unchanged, because corner order is kept; `t` agrees with the instanced scene to rounding. After the walk, `bvhFinishClosest` takes a merged hit's material from the triangle's flags bits 16-31 and its normal from the world-space edge cross, negated on the flip bit, so `material` matches the instanced scene exactly and `normal` to rounding (the oracle's cosine tolerance). Assembly refuses a plan whose source's material changed, as it does a changed mask.

**Filter flags on a merged mesh.** Either set them before the build (`triangleFlags` on the source mesh, which `mergeStaticGeometry` copies per source triangle), or write them after assembly on the group's range: `writeTriangleFilterFlags(scene.triangles, (original, sourceInstance) => ..., scene.mergedGroups[g])`. On a merged range the callback's second argument is the record's source instance (`mergedGroups[g].sourceMeshes` maps it to the mesh), because `original` alone is ambiguous across sources; elsewhere it is `undefined`. A merged mesh's own `scene.meshes[i]` entry is empty and has `mergedInto: g`, and `writeTriangleFilterFlags` throws for it rather than silently flag nothing.

**Watertight.** Each source mesh's vertices are transformed once each, in doubles and rounded once to f32, and the indices are kept. A vertex two triangles share has the same world-space bits in both, so the watertight test still holds. The oracle's merged variants let none of 2M shared-edge rays through.

**Swapping trees.** A merged group is a normal mesh, so its tree from the GPU H-PLOC or LBVH (`createMeshLayoutBuilder`) or from tinybvh goes into `assembleTraceScene` the same way, and assembly writes the ids into its own copy each time. To serve the GPU tree first and tinybvh's later, assemble again with the new tree and the same plan; assembly checks that the plan still fits the instances (below). Which tree is in use is the app's choice. (`writeMergedTriangleIdentity` writes the ids into a tree's records directly, once, for an app that places records itself.)

**Static means static.** The source transforms are baked into the merged geometry, and **only re-assembly detects a moved source**: `assembleTraceScene` throws, naming the instance, if a merged instance's transform differs from the one in the plan, or if its mask, `visible` or `contributesToGi` changed. Nothing checks between assemblies. A caller that moves or hides a merged source and only rebuilds or refits the top level keeps tracing it where and as it was. The checks in `buildTopLevel` and in the top-level shape build and check (`TopLevelBuildInput.merged`) only catch a `merged` record whose own transform isn't the identity, which nothing in this package produces; they guard hand-written records, not moved sources. To move, hide or re-mask an instance, leave it out of the merge (`static: false`) or make a new plan.

**Never refit a merged range.** Its records hold world-space corners and source-mesh triangle indices, so `GpuMeshRefit` over it, fed a source's object-space vertices, would write wrong triangles. `TraceSceneData.meshes[i]` of a merged mesh is empty and carries `mergedInto`; `meshRefitPlacement(scene, i)` gives the word offsets for `GpuMeshRefit.setTree` (and, in an indexed scene, the vertex base, vertex count and positions layout) and refuses merged meshes, and welded meshes of an indexed scene (assemble meshes you will refit with `weld: false`). `GpuMeshRefit` itself takes raw offsets and can't tell.

**When it helps, and when it doesn't.**
- It helps scenes with many meshes used once whose boxes overlap, which is most authored scenes (architecture, Sponza, Bistro), most of all for rays that start on surfaces.
- It does little for heavily instanced scenes. Meshes used more than once never merge, because copying them would cost memory per copy.
- Rays from points in the air already enter few instances, so they gain less.

**Precision.** Merged triangles are stored in world space as f32. Geometry far from the origin loses precision that object space kept: at 10 km the f32 step is about 1 mm, and the triangle test and boxes work at that resolution. Merge only what sits near the origin, or re-centre the world first.

**Build time.** One big build replaces many small ones, and a single build can't spread over the worker pool. On Sponza (132 meshes, 5.74 M triangles, all static, Node on an M5 Pro, `tools/merge-static.ts`), tinybvh took 1.53 s for the merged tree. The per-mesh builds took 1.25 s summed, and the largest took 54 ms, so a pool of workers finishes the instanced build much sooner. Bistro interior took 264 ms merged, against 200 ms summed. Nothing comes near minutes, but on the main thread it is a 1.5 s stall: build in a worker. The GPU H-PLOC builds a merged mesh like any other.

**Memory.** On Sponza the merged scene has 185.2 MB of nodes (185.1 MB instanced) and 263.0 MB of triangle records (the same). The ids and materials use words that were already there. The top level shrinks from 16.5 KB to 0.2 KB. The merged build input (world positions, indices and per-triangle ids, about 150 MB on Sponza) lives only until the tree is built and assembled.

**Measured (study RC5.4, webgpu-ddgi-radiance-cache, Apple M5 Pro).** One world-space tree over all triangles, built by tinybvh at the default leaf size, traced surface rays faster than per-mesh instances:

| Scene | Closest hit | Any hit |
| --- | --- | --- |
| Sponza | 1.86x | 1.39x |
| Bistro interior | 1.24x | 1.13x |

Source: `docs/internals/design/radiance-cache-speed.md` (lever 20) and `tools/gpu/pages/trace-bench/surface-rays-bench.ts` (`TreeLayout = 'merged'`) in webgpu-ddgi-radiance-cache. That bench matched hits only approximately (0.999 agreement). Here hit identity is exact (instance, triangle, side), while `t` agrees to rounding, so on a near-tie the merged walk may pick the other triangle. **This package, re-measured (2026-11-11, `c36b263`, Apple M5 Pro, Chrome 157, the radiance-cache team's bench lock, 3 runs of 5 paired rounds each).** Pages `trace-bench-surface-sponza` and `trace-bench-surface-bistro-interior`; the rows are in `bench/results/trace.csv` (`surface-rays/paired/...`). The table gives the median over runs of each run's paired median, merged over instanced:

| Scene | Surface, closest | Surface, any | Air grid, closest | Air grid, any |
| --- | --- | --- | --- | --- |
| Sponza | 1.85x (1.83–1.93) | 1.36x (1.33–1.44) | 1.49x | 1.20x |
| Bistro interior | 1.22x (1.21–1.24) | 1.11x (1.10–1.11) | 1.19x | 1.31x |

These match RC5.4 within noise, so the package keeps the gain. On Sponza, steps per surface ray drop from 113.8 to 70.3, and instance entries from 9.8 to 1.0. The RTX 3060 hasn't been measured yet, so the option stays off by default until it is.

## Why the triangle test is watertight

- **Moller-Trumbore leaks at shared edges; the watertight test doesn't.** Two million rays were aimed exactly at shared triangle edges. With Moller-Trumbore, 11.6% slipped through on an RTX 3060 and 3.0% on an Intel GPU. Rays into the corners of a sealed thin box also leaked (354 of 6,000 on NVIDIA). The watertight test let none through on either GPU.
- Watertight needs bit-identical shared vertices, so the layout stores `v0, v1, v2` (still 48 bytes) instead of `v0` plus edges.
- It first cost 45% of the speed on NVIDIA. The cause was indexing a vector with a per-ray value (`a[kx]`), which can push it into slow per-thread scratch memory. Building per-ray rows once with `select()` fixed it (Sponza closest-hit 61 to 101 Mrays/s on the RTX 3060) with the same arithmetic, so it stays watertight. Reading nodes and triangles as `vec4f` arrays added a little more (3% on the RTX 3060, 23% on Intel). An unordered any-hit walk and a 32-entry stack bought nothing.

## The oracle and the CPU walk

- **CPU walk** (`src/trace/cpu-traversal.ts`): a step-for-step TypeScript port of the WGSL walk, over the same buffers. It runs in Node, so unit tests cover the walk and the layout without a GPU.
- **Reference tracer** (`src/trace/reference-tracer.ts`): a plain tracer over the triangles with no tree, for checking answers.
- **Oracle** (`src/trace/oracle-fixtures.ts`, `oracle-compare.ts`): fixed cases (thin walls, mirrored and non-uniform transforms, instancing, rays aimed exactly at shared triangle edges, which must not slip through, and the triangle filter and instance masks with ray masks on and off, `tests/gpu/pages/trace-oracle/hooks.ts`) that the GPU walk, the CPU walk and the reference tracer must agree on. The `trace-oracle` GPU page (`tests/gpu/pages/trace-oracle`) runs it on a device and reports mismatches; it must report none on every GPU we test. Every variant also runs two seam fixtures (`seamQuad`, `tests/gpu/pages/trace-oracle/cases.ts`): a split seam (every corner its own vertex, the same bits) must let nothing through in either triangle format; a one-ulp seam (the copies one ulp apart) leaks in both, and the indexed variants must leak on exactly the rays the records variant does, unless a tolerance weld (`weldTolerance`) closes it. The `indexed-*` variants run the same cases on indexed leaves. The `quantized-*` variants run them on quantized nodes (tinybvh, the GPU LBVH and H-PLOC packed by the GPU quantize pass, merged, with indexed leaves): 0 mismatches and 0 shared-edge slips; on the one-ulp seams they let fewer rays through than `f32` (10,927 against 11,287 of 200k), never more, as their boxes are looser.
- **Layout check** (`src/layout-check.ts`): checks a built tree against the layout (child indices, leaf ranges, boxes that contain their children).

## Benchmark method

- **Rays:** probe-style rays: a grid of probes fills the scene bounds, and each probe fires 256 rays in random directions from a fixed seed. Any-hit rays go from each closest hit toward a fixed direction. The rays are generated once and stored in a buffer, so every tracer gets identical input.
- **Timing:** GPU timestamps around the trace dispatch, median of several runs after warm-ups. Without timestamps, a fence-based wall time over 20 dispatches, labelled as such.
- **three-mesh-bvh:** `three-mesh-bvh/webgpu` (`BVHComputeData`) only runs inside three's node system, so its side is a three `WebGPURenderer` compute node that reads the same ray buffer, with its `raycastFirstHit` and `raycastAnyHit` timed by GPU timestamps. Its per-mesh trees are built with `{ strategy: SAH }` explicitly, because its default is CENTER. Both sides use normalized directions.
- **Reference results** (Sponza closest-hit, median of 3): 104.3 Mrays/s on the RTX 3060 (1.20x three-mesh-bvh) and 12.8 on Intel (1.97x). three-mesh-bvh trees converted to our layout traced at 96.7 Mrays/s and took 7.4 s to build, against 101.6 Mrays/s and 5.3 s for tinybvh's. The pages are `tests/gpu/pages/trace-bench*`; results go in `bench/results/trace.csv`.

# BVH walk and GPU layout

This is the part of ddgi's tracer design (the T0.10 spike) that belongs to the BVH: the GPU layout the builders write, the WGSL walk that reads it, the watertight triangle test, and the oracle and CPU walk that check them. How the builders produce the layout is in [bvh-builders.md](bvh-builders.md). The code is in `src/trace/` (`layout.ts`, `wgsl.ts`, `cpu-traversal.ts`, `oracle-compare.ts`, `oracle-fixtures.ts`) and `src/gpu-layout.ts`. Probe tracing and the GI scheduling that sit on top of it stay in webgpu-ddgi.

## In one paragraph

Each mesh gets its own bounding-volume hierarchy (a "BVH": a tree of boxes that lets a ray skip most triangles). A second, small tree over all object instances sits on top. On the GPU, a WGSL function walks the small tree, and for each object the ray might hit, it moves the ray into that object's local space and walks the object's tree. There are two versions of the walk: "closest hit" (what does this ray hit first?) and "any hit" (is anything in the way?). The second stops at the first blocker, which makes shadow rays cheap.

## Data layout

Everything is plain arrays in a few storage buffers, so it survives a worker hand-off (transferable `ArrayBuffer`s) and an upload without conversion. `GPU_LAYOUT_VERSION` identifies the layout.

### Per-mesh tree ("bottom level")

- **Node: 64 bytes, two child boxes per node** (the Aila-Laine layout used by most GPU BVH2 tracers). One node read gives both children's boxes, so the walk tests both and visits the nearer one first without a second memory fetch.

  ```
  struct GiBvhNode {          // 64 bytes
      leftMin:  vec3f, leftIndex:  u32,
      leftMax:  vec3f, rightIndex: u32,
      rightMin: vec3f, leftCount:  u32,   // count > 0: child is a leaf with `count` triangles starting at index
      rightMax: vec3f, rightCount: u32,
  }
  ```
  A child with count 0 is an inner node at `index`. A leaf child stores the first triangle and the number of triangles.
- **Triangles: 48 bytes each, stored in tree order** (leaves point at contiguous runs, so no index indirection): `v0` + original triangle index, `v1` + flags, `v2` + pad. The three corners are stored as they are, so a vertex shared by two triangles has identical bits in both; the watertight test needs that. The original index maps a hit back to the mesh's UVs and material group. Flags bit 0 marks "needs alpha test".
- **Leaves:** at most 4 triangles in the tracer's own builder, fewer where the SAH cost says so. The builders take a largest-leaf option.
- **Deforming meshes** keep the tree's shape. Refitting bottom-up needs each node's parent, so the builder also writes a parent index per node (4 bytes per node, kept in a separate array that only the refit pass uses).

All meshes' nodes go in one node buffer, and all triangles in one triangle buffer, at per-mesh offsets.

### Instances and the top level

- **Instance record: 64 bytes.** It holds the world-to-object transform as a 3x4 matrix (48 bytes), the mesh's root node offset, its triangle offset, the instance index, and flags. Rays are moved into object space, so moving an object never touches its tree. Mirrored and non-uniform scales work because the hit distance is measured on the object-space ray, whose direction we deliberately *don't* renormalize, so `t` stays in world units.
- **Top-level tree:** the same 64-byte node layout, over instance world boxes. Leaves point at instance records. It can be built on the CPU with binned SAH (`src/trace/top-level.ts`), or, on a device with subgroups, refitted on the GPU each frame something moves with a new shape built in a worker now and then (`GpuBvhRefit`; the declared fallback without subgroups is the CPU rebuild). The walk's WGSL is the same either way.
- **Buffers:** instance records and top-level nodes share one buffer (records first, after a 64-byte header). That gives three storage buffers for tracing: nodes, triangles, and top level. This matters because a default device allows only 8 per shader stage.

### Size limits

A default device caps one storage binding at 128 MB. At 48 bytes per triangle that's about 2.7 M triangles in one binding. Sponza (about 262k triangles) fits easily. Bistro exterior (about 2.8 M) doesn't. Nothing is split across bindings: the layout keeps its 48-byte triangle records, because reading corners through an index buffer instead cost 10-30% of trace speed. The host asks for the adapter's largest storage binding and buffer (`requiredLimits`), and throws a limit error naming the limit and the size needed on a device that can't hold the scene.

## The walk (WGSL)

- **One loop, one small stack per thread** (`array<u32, 64>` in private memory; three-mesh-bvh uses 60), shared by both levels. Entering an instance pushes a marker, switches the ray to object space and continues with that mesh's root. Popping the marker switches back to the world-space ray.
- **Child order:** both child boxes are tested (slab test with a precomputed inverse direction). When both are hit, one is pushed and the walk goes to the other. Boxes beyond the current closest hit are skipped.
  - **Closest hit** goes to the nearer child first.
  - **Any hit** goes to the *farther* child first. A shadow ray starts on a surface, so the nearer child is mostly the geometry it starts on, and what blocks it lies ahead. Any blocker will do. This made any-hit 1.1-2.8x faster on the bench scenes, with the same answers.
- **Closest hit** keeps the nearest `t`, instance, triangle and barycentrics, and whether the hit was on the front face. Back-face hits count as hits.
- **Front face means the side the mesh was authored as front,** also through a mirrored transform. The triangle test runs in object space, where the winding is the authored one, so no flip is needed.
- **Any hit** returns at the first triangle hit within `[tMin, tMax]`.
- **Triangle test: watertight** (Woop, Benthin and Wald 2013), with per-ray shear rows computed once per ray. It replaced Moller-Trumbore, which let rays slip through shared edges (below).
- **Stack overflow:** the builders report the maximum depth. The top-level and mesh depths together must fit in the stack (47 levels are left for a mesh tree under a top level; a deeper GPU-built tree is refused).
- **Hit records** are written with a `u32` front-face flag, because buffers can't hold `bool`. A single-triangle tree has an empty right child.

## Why the triangle test is watertight

- **Moller-Trumbore leaks at shared edges; the watertight test doesn't.** Two million rays were aimed exactly at shared triangle edges. With Moller-Trumbore, 11.6% slipped through on an RTX 3060 and 3.0% on an Intel GPU. Rays into the corners of a sealed thin box also leaked (354 of 6,000 on NVIDIA). The watertight test let none through on either GPU.
- Watertight needs bit-identical shared vertices, so the layout stores `v0, v1, v2` (still 48 bytes) instead of `v0` plus edges.
- It first cost 45% of the speed on NVIDIA. The cause was indexing a vector with a per-ray value (`a[kx]`), which can push it into slow per-thread scratch memory. Building per-ray rows once with `select()` fixed it (Sponza closest-hit 61 to 101 Mrays/s on the RTX 3060) with the same arithmetic, so it stays watertight. Reading nodes and triangles as `vec4f` arrays added a little more (3% on the RTX 3060, 23% on Intel). An unordered any-hit walk and a 32-entry stack bought nothing.

## The oracle and the CPU walk

- **CPU walk** (`src/trace/cpu-traversal.ts`): a step-for-step TypeScript port of the WGSL walk, over the same buffers. It runs in Node, so unit tests cover the walk and the layout without a GPU.
- **Reference tracer** (`src/trace/reference-tracer.ts`): a plain tracer over the triangles with no tree, for checking answers.
- **Oracle** (`src/trace/oracle-fixtures.ts`, `oracle-compare.ts`): fixed cases (thin walls, mirrored and non-uniform transforms, instancing, and rays aimed exactly at shared triangle edges, which must not slip through) that the GPU walk, the CPU walk and the reference tracer must agree on. The `trace-oracle` GPU page (`tests/gpu/pages/trace-oracle`) runs it on a device and reports mismatches; it must report none on every GPU we test.
- **Layout check** (`src/layout-check.ts`): checks a built tree against the layout (child indices, leaf ranges, boxes that contain their children).

## Benchmark method

- **Rays:** probe-style rays: a grid of probes fills the scene bounds, and each probe fires 256 rays in random directions from a fixed seed. Any-hit rays go from each closest hit toward a fixed direction. The rays are generated once and stored in a buffer, so every tracer gets identical input.
- **Timing:** GPU timestamps around the trace dispatch, median of several runs after warm-ups. Without timestamps, a fence-based wall time over 20 dispatches, labelled as such.
- **three-mesh-bvh:** `three-mesh-bvh/webgpu` (`BVHComputeData`) only runs inside three's node system, so its side is a three `WebGPURenderer` compute node that reads the same ray buffer, with its `raycastFirstHit` and `raycastAnyHit` timed by GPU timestamps. Its per-mesh trees are built with `{ strategy: SAH }` explicitly, because its default is CENTER. Both sides use normalized directions.
- **Reference results** (Sponza closest-hit, median of 3): 104.3 Mrays/s on the RTX 3060 (1.20x three-mesh-bvh) and 12.8 on Intel (1.97x). three-mesh-bvh trees converted to our layout traced at 96.7 Mrays/s and took 7.4 s to build, against 101.6 Mrays/s and 5.3 s for tinybvh's. The pages are `tests/gpu/pages/trace-bench*`; results go in `bench/results/trace.csv`.

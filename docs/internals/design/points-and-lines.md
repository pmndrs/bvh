# Points and lines as primitives (PARITY 3.2, 3.3)

Status: **CPU built** (2026-11-14), **GPU not built**. This note records the CPU design, why the GPU part waits, and the
GPU design for when someone needs it.

## What exists (CPU)

- `src/query/primitive-bvh.ts`: `CpuPointsBvh` and `CpuSegmentsBvh` (one base, `CpuPrimitiveBvh`), engine-free.
- `src/query/ray-primitives.ts`: three r187's `Ray.distanceSqToPoint`, `closestPointToPoint`, `distanceSqToSegment` and
  `Line3.closestPointToPoint`, ported operation for operation (bit-for-bit, tested against three's `Ray`).
- `src/three/points-lines.ts`: `PointsBVH`, `LineSegmentsBVH`, `LineLoopBVH`, `LineBVH` and the `acceleratedRaycast`
  path for `Points` and the line objects, three's `Points.raycast` / `Line.raycast` step for step.
- Docs: `docs/cpu-queries/points-and-lines.mdx`, `docs/three/points-and-lines.mdx`. Example: `examples/three/point-cloud`
  (three-mesh-bvh's `pointCloudIntersection`).

### Primitives are entries

The input is three's: positions, an optional index, an entry range (`start`, `count`). A point is one entry; a segment
starts at an entry (`'segments'` every second entry, `'strip'` every entry but the last, `'loop'` every entry, the last
closing to the first). Each record carries its entry, which is what three's `Line.raycast` reports as `index`; for
points `/three` maps the entry to its vertex as three does. The geometry is never reordered.

### Layout

- Nodes: the GPU layout's 64-byte two-child nodes (`GPU_LAYOUT_NODE_WORD`), leaves of 1 to 8 primitives, children after
  their parents, each subtree one run of tree positions (the constructor checks this; refit is one reverse pass).
- Records: a point is 16 bytes (`x, y, z` as `f32`, entry as `u32`), a segment 32 (start, entry, end, 0). Word 3 is the
  entry in both, as `originalIndex` is in a triangle record.
- File: a 64-byte header (`"BVHP"`, format 1, kind, mode, entry range, counts, bounds), the nodes, the records. Own
  format, not `serializeLayout`'s: that one is the triangle layout with `GPU_LAYOUT_VERSION`.

### Why a centre split and not tinybvh

The builder contract (`MeshInput` in, layout out) takes triangles. Points fed to tinybvh as degenerate triangles
`(p, p, p)` build, and segments as `(a, b, b)`, and every CPU query would then come for free from `CpuMeshBvh`. But
measured on 50,000 random points, tinybvh made 49,915 nodes: a box of area 0 always pays to split under SAH, so leaves
hold one point. With 48-byte triangle records that is about 112 bytes per point, against about 27 for the centre split
here (43 per segment). Its build time matches three-mesh-bvh's `PointsBVH` (about 0.9 s for a million random points,
Node 24 on the Windows laptop under load). SAH itself is the problem for points (leaf cost 0), so a JS SAH would need a made-up leaf area. The centre split
is three-mesh-bvh's default (`CENTER`), runs in TypeScript with no WebAssembly (so `computeBoundsTree({ type })` needs
no `prepareBoundsTrees()`), and falls back to a median cut when a side would be empty or past depth 40, so no input
makes a deep tree.

### Matching three

The `/three` raycast does what three's does, in order: bounding-sphere test (three-mesh-bvh skips it, and it can matter:
under a stretched scale three's sphere test drops hits its own per-point test would keep), the local ray and threshold
(over the object's mean `scale`, so a mirrored object's threshold can be negative; it counts squared), the draw range
read at each raycast, `near`/`far` on the world distance, pushes in entry order. Trees cover every entry and filter by
the range, so a draw range change needs no rebuild. Two cases need more than one walk: a `LineSegments` range from an
odd entry (a second tree over odd-aligned pairs, made on first use), and a `LineLoop` range's closing segment (tested on
its own with `raySegmentHit`). `firstHitOnly` ranks by world distance: the tree's closest is by local distance, and a
second walk over hits within 1e-9 of it ranks them in world space, so the pick is three's sorted first, ties to the
lower entry.

## GPU: not built

The task allowed a GPU walk only if it fits as an opt-in module without changing the triangle walk's text. It could, as
a standalone walk, but it would be a second GPU path with nothing asking for it:

- **No caller.** No surveyed project calls `PointsBVH` or the line BVHs (PARITY, "Who needs it"), and three-mesh-bvh
  has no GPU points or lines either (its `/webgpu` packs triangles only).
- **Scenes are triangle-only throughout.** `SceneTrees`, `assembleTraceScene`, the top level, merged static groups,
  quantization, indexed leaves and the shapecast template (`shapecastWalkWgsl`, whose mesh leaf calls
  `bvhLoadTriangle`) all assume a triangle leaf. A points walk inside scenes means a per-mesh primitive kind in the
  instance or mesh header and a branch in every leaf, which changes the triangle walk's text and its register pressure
  (the walk is benched to 1-2%).
- **A standalone walk is a parallel path.** One tree, its own bind group, its own oracle page, SwiftShader and
  real-GPU runs. The rules say no parallel GPU paths without a reason; there isn't one yet.

### Design for later

When a consumer needs GPU picking or queries on points or lines:

1. **Upload as is.** The CPU tree's buffers are GPU-ready: `layout.nodes` (64-byte `f32` nodes, the walk's `BvhNode`)
   and `layout.primitives` (16- or 32-byte records, `vec4f`/two `vec4f` with the entry bitcast in `.w`). Quantized
   nodes would come from `quantizeLayout`-style code over the same nodes; the frame logic doesn't care about leaves.
2. **A standalone module, `primitiveWalkWgsl({ kind, nodeFormat })`,** beside `traceModuleWgsl`, not inside it: its own
   bindings (`bvhPrimitiveNodes`, `bvhPrimitives`, one `mat4x4f` object-to-world and its inverse), entry points
   `bvhPointsClosest(ray, threshold) -> BvhPrimitiveHit` (entry, distance, distanceToRay, pointOnRay, pointOnPrimitive),
   `bvhSegmentsClosest`, `bvhPrimitivesClosestPoint(point, maxDistance)`, and a shapecast stamped from the same template
   idea with a `bvhShapecastPoint`/`bvhShapecastSegment` hook. Node handling copied from the ray walk's text (the same
   child test with each box grown by the threshold, nearer child first), the leaf test from `ray-primitives.ts`.
3. **Oracle:** a GPU page against `CpuPointsBvh.raycast` per ray, exact entry and `f32`-tolerant distances (the GPU
   rounds `sqrt` and division within an ulp, DIVERGENCES T4), on random, clustered and coincident points, every segment
   mode, thresholds including 0, and objects with mirrored transforms.
4. **Scenes, only if needed:** a mesh kind in bits of the instance flags (points 1, segments 2), separate leaf loops
   selected per instance, compiled in only when the scene has such meshes (`primitiveKinds` option), so a triangle-only
   scene's walk text stays byte-identical. Measure the triangle walk before and after on the Sponza surface bench.
5. **`/three`:** `BvhTraceNodes` gains nothing until step 4; before it, a TSL `wgslFn` over the standalone module.

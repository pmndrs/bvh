# Clustered top level (PARITY 8.9, three-mesh-bvh's `ClusteredBVH`)

## Finding (2026-11-14)

**What it is.** `ClusteredBVH` (three-mesh-bvh 0.9.18, `src/webgpu/ClusteredBVH.js`, MIT) is the top-level tree (TLAS) of `three-mesh-bvh/webgpu`. It is not exported from the package's entry points and has no example or docs page of its own: `BVHComputeData.update()` builds one every time, with `primitiveLimit: 64` (or `Infinity` when the scene has fewer than three meshes). Its primitives are not objects but *clusters*: for each mesh (each `getBVH(object, instance)` tree), it walks the mesh's tree from the root and stops at the first node with fewer than `primitiveLimit` triangles, or at a leaf (`_traverseClusters`; a leaf at or above the limit is kept, with a `console.warn`). Each stop becomes one TLAS leaf, packed as two words (object and instance in one, the mesh tree's root and node index in the other, 24 bits for the node), with the node's box moved to the TLAS frame by the object's world matrix (`writePrimitiveBounds`). `InstancedMesh`, `BatchedMesh` and `SkinnedMesh` (the `shouldCluster` option, named the other way round from what it does) are referenced whole: one leaf per instance at the tree's root. `BVHComputeData` then uploads only the subtrees the leaves enter (`appendBVHSubtree`), not the mesh nodes above the cut, and gives each (instance, root) one transform slot that its clusters share. `updateTransforms()` refits the TLAS (`refit()` re-runs `writePrimitiveBounds`) and rewrites the transforms.

**What it is for.** Separating meshes whose boxes overlap. With one leaf per instance, a ray in a scene of big overlapping meshes enters every instance box around it and descends into each; with clusters the TLAS sees the parts of each mesh and skips the ones the ray misses. In the literature this is partial re-braiding (Benthin, Woop, Wald and Afra, "Improved two-level BVHs using partial re-braiding", HPG 2017), without the per-ray opening heuristic: the cut is fixed at build time by triangle count.

**Who uses it.** Everyone on `three-mesh-bvh/webgpu` (Denoiser, upscaler, ddgi-research, reflections in the PARITY survey), implicitly: there is no way to turn it off besides `primitiveLimit: Infinity`, which `BVHComputeData` only chooses for scenes under three meshes. No surveyed project constructs `ClusteredBVH` itself.

**How it maps onto our layout.**
- *Mesh trees.* Ours are two-wide nodes with child references relative to the instance record's `rootNodeOffset`, and leaves relative to its `triangleOffset` (`docs/internals/design/trace.md`). A cluster at an inner node is a subtree whose child indices must be renumbered from its own root before the walk can enter it there; a cluster at a leaf is a child slot, which becomes a one-node tree. The triangles need no copy: the cluster's record keeps the mesh's `triangleOffset`, so records, indexed leaves (vertex base, material table, merged side tables, all read through `triangleOffset - n`) and merged groups work unchanged. Quantized subtrees take a copy of the mesh's frame slot, so their planes decode to the same values.
- *The top level.* A cluster is an instance record: the instance's own 64 bytes (world-to-object rows, instance index, flags with mask, material, mirrored and merged bits), with `rootNodeOffset` at the cluster's copy. The top level is built over the clusters' world boxes with the same SAH builder as `buildTopLevel`, one record per leaf. `recordOrder` maps each record to its instance's input position, as for merged groups.
- *`SceneTrees`.* Its top level is `GpuTopLevel` (gathered and refit on the GPU from a per-instance table, one record per instance). A clustered mode there would need a per-cluster table, cluster boxes kept per mesh tree, and copies that follow tree swaps and refits. Not done (below).
- *The WGSL walk.* Unchanged. It already enters a record's tree at node 0 relative to `rootNodeOffset` and reads the frame at `rootNodeOffset - 1`, so a cluster record is indistinguishable from an instance. No `traceModuleWgsl` declaration changes, with or without the feature.

**Decision: in scope, built.** It is acceleration-structure work (the shape of the top level), not shading, not WebGL, and needs no second walk. Built as `clusterTraceScene` on assembled scenes (`TraceSceneData`), CPU only, with the walk untouched.

## What was built

`src/trace/clustered-scene.ts` (root export):

- `clusterTraceScene(scene, { primitiveLimit = 64, split, topLevel }) -> ClusteredTraceSceneData`: a new scene with the same triangles and positions; the scene's trees still entered at their roots, compacted, then one copy per distinct cluster subtree (shared by every record of the mesh); a new top level; placements that follow the trees; and `clusters` (counts, `addedBytes`). `split(record)` picks the records to cut; the default cuts records whose mesh no other record uses (`sharedBy === 1`), the analogue of `ClusteredBVH` referencing instanced meshes whole. Nothing to cut returns the input scene object as it is.
- `meshClusters(scene, nodeOffset, primitiveLimit)`: one mesh's cut, `_traverseClusters` step for step on our nodes (left before right): each cluster's root (or parent and side for a leaf), lowest triangle slot, triangle count, object-space box (the union of the cluster root's child boxes as stored, decoded for quantized nodes) and depth.
- Mesh trees only split records trace are dropped (as `BVHComputeData` uploads only referenced subtrees). Their `scene.meshes` (or `mergedGroups`) entries keep `triangleOffset`/`triangleCount` (filter flags are still written there) with `nodeCount: 0` and the new `TraceSceneMeshPlacement.clustered: true`, which `meshRefitPlacement` refuses. A scene whose records enter a root no placement describes (written by other code) keeps its node buffer whole and appends the copies.

**World boxes.** A cluster's world box is its object-space box through the inverse of the record's f32 world-to-object rows (in doubles), with `transformBounds`'s padding. That is the transform the walk actually applies, so the box is the one its triangles occupy as the walk sees them; `buildTopLevel` uses the caller's object-to-world instead (it has it; this function reads only the scene).

**Departures from `ClusteredBVH`.**
- The default `split` keys on "mesh used by one record", not on the three object type; two `Mesh`es sharing a geometry are cut by three-mesh-bvh and kept whole here (pass `split: () => true`).
- A leaf at or above the limit is cut silently (no warning).
- No `updateTransforms`: moving instances means assembling and clustering again (a CPU top-level build, as `buildTopLevel` is).
- Our top level stays one leaf per record (`maxLeafPrimitives: 1`); three-mesh-bvh forces one object per leaf too (`_strictLeafSize: 1`).

## Checks

- `src/trace/clustered-scene.test.ts`: the cut partitions every triangle at `ClusteredBVH`'s stops, the same in `f32` and quantized nodes (quantized boxes around the exact ones); the CPU walk on clustered scenes against the oracle's reference (trace-oracle's `evaluateCase`) and against the plain scene ray for ray (any hits equal, closest `t` equal to the bit, same triangle unless tied), every oracle case, cuts at 1, 4 on every record and 64, in `f32`/quantized, records/indexed (packed), merged; shared-edge quads cut to leaves let nothing through; compaction (placements, dropped trees, `meshRefitPlacement`), the `split` callback's view, the stack check.
- `tests/gpu/pages/trace-clustered/parity.test.ts`: three-mesh-bvh's own `ClusteredBVH` and ours on the same tree (built by three-mesh-bvh, converted by `tmbvh-convert.ts`, which keeps its node structure and order) give the same clusters, run for run, for limits 1 to beyond the triangle count.
- `tests/gpu/pages/trace-clustered` (GPU, SwiftShader `profile=ci`, Windows, 2026-11-14: pass, 41 s): the batch kernel on the plain scene and three cuts, six formats (`f32`, quantized, `f32` indexed packed, quantized indexed, merged quantized, merged quantized indexed), every oracle case: the oracle's rules against the CPU reference; ray for ray against the plain scene's GPU answers; shared-edge quads cut to leaves (0 slips of 20,000, plain and transformed); `ClosestPointBatchKernel` distances equal to the bit on 1,000 points per case. Closest hits at faces meeting at an edge can name the other face a few ulps away (`thin-box` in `f32`, 4 rays of 2,400): the f32 walk culls a box whose rounded entry lands past the hit it holds, and plain and clustered trees meet the faces in other orders. The page counts these as near ties (within 2^-20 relative, different triangle); the doubles CPU walk has none.

## Measured (CPU counts, not time)

Sixteen overlapping meshes of 4,000 triangles, one instance each, quantized nodes, 20,000 random rays, the CPU walk's nodes visited per ray (both levels):

| Scene | Records | Bytes | Closest | Any |
| --- | ---: | ---: | ---: | ---: |
| One leaf per mesh | 16 | 5,074,144 | 271.9 | 139.6 |
| `primitiveLimit` 256 | 384 | 5,121,248 | 214.4 | 130.5 |
| 64 (default) | 1,515 | 5,266,016 | 185.8 | 120.1 |
| 16 | 6,263 | 5,873,824 | 135.4 | 97.7 |
| Merged static | 1 | 4,841,184 | 53.2 | 48.6 |

With `f32` nodes: 7,073,216 bytes plain, 7,169,152 clustered at 64. Before compaction (keeping the scene's own trees) the default cut cost 43% more bytes in quantized scenes; with it, 3.8%. No GPU timing: the real GPUs were busy with another project's benchmarks. A `trace-bench-surface-*` variant with the cluster cut is the next step for timing (Sponza and Bistro, where `mergeStaticInstances` already wins 1.2-1.9x).

## Not done

- **`SceneTrees` / `GpuTopLevel` / `/three` sync:** no clustered mode (see the mapping above). `BVHComputeData` users switching through `/three` get `SceneTrees`' top level, one record per instance, plus merged static instances.
- **GPU timing** on the RTX 3060 and M5 (Sponza, Bistro interior; clustered against instanced and merged).
- **A per-ray opening decision** (true re-braiding): out of scope of the port.

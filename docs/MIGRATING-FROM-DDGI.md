# Migrating webgpu-ddgi to `@pmndrs/bvh` (P3)

For the switch in phase P3, when ddgi-core stops using its own copy of the tracer (`packages/ddgi-core/src/platform/trace/`) and imports this package. This note lists what changed in the walk's API before 0.1 (branch `bx-walk-api`), and how ddgi's material hook maps onto the new one. ddgi's code is cited by file, in `packages/ddgi-core/src/platform/`.

## 1. Names

Every WGSL name the trace module declares now starts with `bvh`, `Bvh` or `BVH_`. ddgi's own names (`giMaterialPasses`, `giScene*`, `GI_GATHER_*` and so on) are ddgi's and don't change, but they must not start with `bvh`.

| Old (ddgi, and this repo before 0.1) | New |
| --- | --- |
| `giTraceClosest(ray: GiRay) -> GiHit` | `bvhTraceClosest(ray: BvhRay) -> BvhHit` |
| `giTraceAny(ray: GiRay) -> bool` | `bvhTraceAny(ray: BvhRay) -> bool` |
| (none) | `bvhTraceClosestMasked(ray: BvhRay, mask: u32) -> BvhHit` |
| (none) | `bvhTraceAnyMasked(ray: BvhRay, mask: u32) -> bool` |
| `GiRay`, `GiHit` | `BvhRay`, `BvhHit` (same fields, `BvhRay` still 32 bytes) |
| `GiBvhNode`, `GiBvhTriangle` | `BvhNode`, `BvhTriangle` (not `BvhBvh*`) |
| `GiInstance`, `GiRaySetup`, `GiTriangleHit` | `BvhInstance`, `BvhRaySetup`, `BvhTriangleHit` |
| `GiHitRecord` (batch results) | `BvhHitRecord`: 40 bytes by default, words 0-5 as before plus `material` (6) and the normal (7-9); `batchKernelWgsl(mode, { hitNormal: false })` keeps the old 24-byte layout as `BvhHitRecordCompact` (`HIT_RECORD_COMPACT_BYTES`, `HIT_RECORD_COMPACT_WORD`) |
| `GiHit` | `BvhHit` gained `material: u32` and `normal: vec3f` at the end (see trace.md, "Hit normal and material"); code that builds a `BvhHit` with five arguments must pass seven |
| `giBatch` (batch entry point), `giBatchRays`, `giBatchResults` | `bvhBatch`, `bvhBatchRays`, `bvhBatchResults` |
| `giTraceNodes`, `giTraceTriangles`, `giTraceTopLevel` | `bvhTraceNodes`, `bvhTraceTriangles`, `bvhTraceTopLevel` |
| `giEdgeFunction`, `giIntersectWatertight`, `giBoxEntry`, `giMakeRaySetup`, every other `gi*` helper | the same with `bvh` |
| `GI_*` constants (`GI_STACK_SIZE`, `GI_BOX_MISS`, `GI_TRIANGLE_FLAG_FILTER`, ...) | `BVH_*` |
| `giMaterialPasses(instance, triangle, barycentrics, frontFace)` (ddgi), `giTriangleFilter(instance, triangle, t, barycentrics, frontFace)` (this repo, 61f9688) | `bvhTrianglePasses(candidate: BvhCandidate) -> bool` |
| `NO_MATERIALS_WGSL` (ddgi), `NO_TRIANGLE_FILTER_WGSL` | `DEFAULT_TRIANGLE_FILTER_WGSL` (the old name stays as a deprecated alias) |
| `TRIANGLE_FLAG_CONSULT_MATERIAL` (ddgi), `TRIANGLE_FLAG_ALPHA_TEST` | `TRIANGLE_FLAG_FILTER` (bit 0, unchanged; `TRIANGLE_FLAG_ALPHA_TEST` stays as a deprecated alias) |
| `TriangleFilter` (this repo, 61f9688) | `TrianglePasses` (old name a deprecated alias, but its arguments changed) |
| `traceModuleWgsl(slot)` | `traceModuleWgsl({ traceSlot, triangleFilter, rayMasks })` |
| `batchKernelWgsl(mode, slot?, filter?)` | `batchKernelWgsl(mode, { traceSlot, triangleFilter, rayMasks })` |

TypeScript names (`traceModuleWgsl`, `createCpuTracer`, `INSTANCE_FLAG.contributesToGi`, ...) keep their names unless the table says otherwise.

## 2. The material hook becomes a `bvhTrianglePasses` wrapper

ddgi's walk called `giMaterialPasses(instance, triangle, barycentrics, frontFace)` for triangles with flag bit 0 set. The package's walk calls `bvhTrianglePasses(candidate)` for the same triangles, with the same meaning: true lets the ray through. The candidate carries more:

```wgsl
struct BvhCandidate {
    instance: u32,       // platform instance index: ddgi's scene-copy slot, as before
    triangle: u32,       // original triangle index in the mesh
    t: f32,              // new: distance along the ray
    barycentrics: vec2f,
    frontFace: bool,     // the authored side, also when mirrored (unchanged rule)
    query: u32,          // new: BVH_QUERY_CLOSEST (0) or BVH_QUERY_ANY (1)
    rayMask: u32,        // new: the trace's ray mask, all ones without one
}
```

The least change keeps `giMaterialPasses` as it is and adds a wrapper next to it in `material-test-wgsl.ts`:

```wgsl
fn bvhTrianglePasses(candidate: BvhCandidate) -> bool {
    return giMaterialPasses(candidate.instance, candidate.triangle, candidate.barycentrics, candidate.frontFace);
}
```

and builds the `rays` module with `traceModuleWgsl({ traceSlot, triangleFilter: null, rayMasks })`, since the material test module (which depends on the scene and texture modules) supplies the function. `NO_MATERIALS_WGSL` becomes `DEFAULT_TRIANGLE_FILTER_WGSL`, or `triangleFilter` left at its default, for `trace-gpu.ts`'s low-level batch kernel.

ddgi's four rules map onto the candidate like this:

| ddgi rule (`WalkMaterial`) | Reads | Notes |
| --- | --- | --- |
| See-through: `blocksLight: false` | the material only | Unchanged. |
| Single-sided: `doubleSided: false` and `!frontFace` | `candidate.frontFace` | Unchanged; the side rule is the same. |
| Alpha cutout: alpha below `alphaCutoff` | `candidate.barycentrics` (for the uv) | Unchanged. |
| Stochastic: hash of frame, instance, triangle, barycentrics `>= alpha` | `candidate.barycentrics` | Unchanged. The hash could mix in `candidate.t` too, but that changes which rays pass, so leave it for a measured change. |

What the new fields allow, without forcing anything now:

- `query` can make a material cast no shadow (pass only for `BVH_QUERY_ANY`) while closest-hit rays still see it, which `giMaterialPasses` couldn't express. It can also skip the stochastic test for one query.
- `rayMask` lets a filter see which kind of ray asks, if ddgi gives its techniques different ray masks.

`materialNeedsLookup` and the `gi/triangle-flags` kernels decide the flags; nothing about them changes beyond the bit's name.

## 3. The CPU walk and the reference

| ddgi | `@pmndrs/bvh` |
| --- | --- |
| `createCpuTracer(scene, { materialAt, uvAt })` | `createCpuTracer(scene, { triangleFilter, rayMasks })` |
| `tracer.closest(ray, stats)`, `tracer.any(ray, stats)` | `tracer.closest(ray, { mask, stats })`, `tracer.any(ray, { mask, stats })` |
| `buildReferenceScene(meshes, instances, materialAt, uvAt)` | `buildReferenceScene(meshes, instances, { triangleFilter, rayMasks })` |
| `traceClosestReference(scene, ray)`, `traceAnyReference(scene, ray)` | the same, with an optional `{ mask }` |

`triangleFilter` is a `TrianglePasses`: `(candidate: TriangleCandidate) => boolean`, with the WGSL struct's fields (`barycentrics` is `[u, v]`). ddgi's `materialPasses` closure in its `cpu-traversal.ts` becomes:

```ts
const triangleFilter: TrianglePasses = ({ instance, triangle, barycentrics: [u, v], frontFace }) => {
    const material = materialAt(instance, triangle);
    const uvsOf = (channel: 0 | 1) => uvAt?.(instance, triangle, channel) ?? null;
    return materialLetsRayPass(material, frontFace, () => materialAlphaAt(material, u, v, uvsOf), () => stochasticHashValue(instance, triangle, u, v));
};
```

The package's reference asks the filter about every triangle it really hits (inside the triangle and the ray interval, never about a near miss). It doesn't know where a filter's answer changes inside a triangle, so it can't call a hit near a cutout's alpha edge ambiguous the way ddgi's reference does (`nearAlphaEdge` in ddgi's `reference-tracer.ts`). Filters whose answer is the same for the whole triangle (see-through, single-sided, a cutout with an opacity below its cutoff) compare cleanly against it. For cutouts read from textures and for stochastic surfaces, keep ddgi's alpha-aware reference for its material tests (or wrap it), or the comparisons will flake at alpha edges.

## 4. Triangle flags: `writeTriangleFlagsOnCpu` and `writeTriangleFilterFlags`

ddgi's `writeTriangleFlagsOnCpu(triangleWords, range, mask, slotOf)` (`trace/triangle-flags.ts`) is ddgi's CPU twin of its flags pass: a triangle is flagged when its material slot's bit is in the mesh's mask. The package's `writeTriangleFilterFlags(triangles, filtered, range?)` takes the decision as a predicate on the original index:

```ts
writeTriangleFilterFlags(triangleWords, (originalIndex) => (mask & slotMaskBit(slotOf(originalIndex))) !== 0, range);
```

- `range` is `{ triangleOffset, triangleCount }`, the same shape as ddgi's `MeshTriangleRange` without `meshSlot`. Each `TraceSceneData.meshes` entry now has `nodeCount` and `triangleCount` too, so it can be passed as the range directly.
- It returns how many records it flagged, keeps the other flag bits, and throws a `RangeError` for a range outside the buffer (ddgi's version doesn't check).
- The predicate gets a second argument: on a merged static group's range (`TraceSceneData.mergedGroups[g]`), the record's source instance; `undefined` elsewhere (§6).
- `MeshInput.triangleFlags` may set bit 0 only. Bits 1-31 are reserved (bit 1 is the merged flip bit, 16-31 a merged triangle's material, which assembly writes), and every builder throws for them.
- `assembleTraceScene` copies each mesh's records into the scene's buffer. Flags written to a `MeshBvh` after assembly don't reach the scene: write the scene's `triangles` with the mesh's range, as ddgi's install hook already does on the GPU.

ddgi's GPU pass (`gi/triangle-flags`) writes the same bit of the same word, so it needs no change.

## 5. The layout: version 2 and instance masks

`GPU_LAYOUT_VERSION` goes from 1 to 2 (`TRACER_GPU_LAYOUT_VERSION` too). Serialized trees from version 1 are refused by `deserializeLayout` ("rebuild the tree"); trees are cheap to rebuild. Mesh nodes and triangle records didn't change, only the meaning of the instance record's flags word:

| Bits of `INSTANCE_WORD.flags` | Meaning |
| --- | --- |
| 0-2 | `contributesToGi`, `visible`, `mirrored` (unchanged) |
| 3-7 | reserved, 0 |
| 8-15 | the instance mask (`INSTANCE_MASK_SHIFT`, `INSTANCE_MASK_BITS`): 8 bits as in DXR |
| 16-31 | the instance's material + 1 (`MATERIAL_SHIFT`), 0 for none (hits report `NO_MATERIAL`) |

A masked trace skips an instance whose `(instanceMask & rayMask) == 0` at the top level, before entering its tree. `bvhTraceClosest` and `bvhTraceAny` use all ones, so **an instance whose mask is 0 is never hit** by them either (as in DXR) unless masks are compiled out with `rayMasks: false`, where every instance is visible. The cost with masks on is one shift, AND and compare per instance leaf, not per triangle. `buildTopLevel` and `assembleTraceScene` write `0xff` (visible to every ray) unless the instance gives `mask` (0-255; anything else is a `RangeError`).

**What ddgi must change:** every place ddgi writes instance records itself must write the mask bits, or, with masks on, every instance has mask 0 and no ray sees it:

- `trace/top-level-gpu.ts`, the GPU gather: `let flags = GI_GATHER_FLAG_TRACED | select(0u, GI_GATHER_FLAG_MIRRORED, mirrored);` needs `| (0xffu << 8u)` (or the instance's own mask from the scene copy);
- `trace/top-level.ts` goes away with the switch (the package's `buildTopLevel` writes the default);
- anything else that writes the record's word 15.

ddgi's own scene-copy `INSTANCE_FLAG` (`scene/layout.ts`: `live`, `contributesToGi`, `visible`, `mirrored`, `singular`) is a different word in a different buffer and is not affected.

**If ddgi doesn't need masks yet,** it can build its module with `rayMasks: false`: the walk then has no mask test and ignores the mask bits, so records without them still work, and nothing else changes. This is the same walk with the mask code compiled out, not a second walk. `bvhTraceClosestMasked`/`bvhTraceAnyMasked` don't exist in that module, and `candidate.rayMask` is all ones.

The batch kernel (`batchKernelWgsl`, and `TraceBatchKernel` in `src/trace/trace-gpu.ts`, exported from the package) takes one mask per batch, from a uniform at group 1 binding 2 (`BATCH_PARAMS_BYTES`), only when masks are on. No per-ray mask buffer.

## 6. Merging static instances (`mergeStaticInstances`, optional)

Off by default, and nothing changes when it stays off. It targets the surface-ray slowdown that RC5.4 found (lever 20 in ddgi-radiance-cache's `radiance-cache-speed.md`). To enable it in ddgi:

1. Mark never-moving instances `static: true` in the `TraceSceneInstance`s ddgi builds from its scene copy. Instances that ddgi can move, hide or toggle at runtime must stay `static: false`, because a merged instance can't move. Static instances that are already hidden or don't contribute to GI are left unmerged. Merged instances need distinct `instanceIndex` values.
2. Call `mergeStaticGeometry(meshes, instances)`. Build each `plan.groups[g].geometry` with the same builder as the other meshes: the GPU H-PLOC first, then tinybvh when the swap comes. Assemble with `assembleTraceScene(meshes, instances, {}, { merged: { plan, meshes: groupTrees } })`. At the tinybvh swap, assemble again with tinybvh's group trees. Meshes with `plan.meshNeeded[i] === 0` need no tree of their own.
3. **ddgi's GPU top level** (`trace/top-level-gpu.ts`, the gather) writes records itself. It must write one record per merged group, with the identity transform, `INSTANCE_FLAG.merged` (bit 3, `| 8u`) and the group's mask, and no records for the merged sources. Their slots must not be refitted. Leave bits 16-31 of the flags at 0 (a merged record has no material of its own; its triangles carry theirs). Records ddgi writes for normal instances may put a material + 1 there, or leave 0 for none. Nothing notices a merged source that moves or is hidden until ddgi assembles again with the plan (which then throws), so ddgi must not merge anything it can move, hide or re-mask. Merged groups' triangle ranges must never be refit (`GpuMeshRefit`): get refit offsets from `meshRefitPlacement`, which refuses merged meshes.
4. **ddgi's triangle-flags pass** (`gi/triangle-flags`) must keep bits 1-31 when it rewrites a merged mesh's flags word: set or clear bit 0 only, don't overwrite (bit 1 is the merged flip bit, bits 16-31 the material). Its per-mesh ranges for merged sources are gone: `TraceSceneData.meshes[i]` of a merged mesh is empty (`triangleCount` 0) and carries `mergedInto: g`, and `writeTriangleFilterFlags` throws when given it. Two ways remain:
   - Before the build: set `triangleFlags` (bit 0 only; anything else throws) on the source mesh. `mergeStaticGeometry` copies them per source triangle.
   - After assembly, on the CPU: `writeTriangleFilterFlags(scene.triangles, (originalIndex, sourceInstance) => ..., scene.mergedGroups[g])`. On a merged range `originalIndex` is the triangle's index in its source mesh and `sourceInstance` the source's `instanceIndex` (`mergedGroups[g].sourceInstances` / `sourceMeshes` map it to the mesh); on other ranges `sourceInstance` is `undefined`. Upload the range again.
   - On the GPU, a pass over the group's range (`mergedGroups[g].triangleOffset` / `triangleCount`) must read the source instance from the record's padding word (word 11) and the source triangle from word 3, because a merged range interleaves its sources in tree order. ddgi's per-mesh dispatch can't be reused as is.
5. Hits, filter candidates and `frontFace` name the source instance and its triangle as before, so the material lookups don't change. The per-instance material (`TraceSceneInstance.material`) is copied into each merged triangle's flags bits 16-31. Material is not a grouping key.

## 7. Checklist for the switch

1. Replace ddgi's `platform/trace/wgsl.ts` walk with `traceModuleWgsl({ traceSlot, triangleFilter: null, rayMasks })`; rename the callers' `giTrace*`, `GiRay`, `GiHit`.
2. Add the `bvhTrianglePasses` wrapper to `material-test-wgsl.ts`; replace `NO_MATERIALS_WGSL` with `DEFAULT_TRIANGLE_FILTER_WGSL`.
3. Write the instance mask bits in the GPU gather (or choose `rayMasks: false`).
4. Move the CPU tracer and reference calls to the options objects; wrap the material rules as a `TrianglePasses`.
5. Rebuild any serialized trees.
6. Run ddgi's oracle and material pages; both must report 0 mismatches.
7. Optionally, turn on `mergeStaticInstances` (section 6) and re-run the surface-ray benches.

## 8. Scene trees: `SceneTrees` replaces ddgi's tree half of `ray-queries.ts`

`gpu-mesh-trees.ts`, `growable-storage.ts`, `top-level-gpu.ts`, `deforming-trees.ts` and the tree half of `ray-queries.ts` are in the package (`src/scene/`, `src/trace/top-level-gpu.ts`). ddgi can keep its own until P3, or drive `SceneTrees` from its scene copy:

| ddgi | `@pmndrs/bvh` |
| --- | --- |
| `meshesUploaded` / `meshesRemoved` | `addMesh(geometry, { deforming })` / `removeMesh(mesh)` |
| scene-copy instances, `setTransform`, visibility | `addInstance`, `setTransform`, `setInstance({ visible, contributesToGi, mask, material })`, `removeInstance` |
| `meshesPosed` with `deformBounds` | `setPose(mesh, vertices)`, `deforming: { bounds }` |
| `update(encoder, upload)` | `update(encoder)` |
| `module.resources()`, `module.version` | `buffers`, `version` (or `traceBuffers` for `TraceBatchKernel`) |
| `meshTreeCounts`, `memory`, `holdRefinements`, `releaseRefinements` | `counts`, `memory`, `holdRefinements`, `releaseRefinements`, plus `onTree` events and `meshStatus` (`builtBy`, `fallbackReason`) |

The GPU top level's gather reads its own instance table, not the scene copy: hits name `instanceIndex` (ddgi passes its scene slot). ddgi's install hook for triangle flags (`requestTriangleFlags`) has no twin yet: write flags before the build (`MeshInput.triangleFlags`) or after placement on `meshStatus(mesh).placement`.


Three rules differ from ddgi's:

- **When `update` throws.** A storage limit (node or triangle buffer) or a failed tinybvh build is thrown before anything is recorded: the encoder is untouched and the scene unchanged, so the frame can be dropped. A top-level error (trace stack, a limit) comes after the frame was recorded: submit the encoder anyway.
- **Merged-away meshes.** With `mergeStaticInstances`, a mesh that only merged instances used has no tree and no geometry after the first update; `addInstance` of it throws. Add the mesh again for new instances.
- **`gpuMaxDepth` covers tinybvh's trees too.** A deeper tinybvh tree is refused (with `fallbackReason`), and the mesh keeps its GPU tree, if any.

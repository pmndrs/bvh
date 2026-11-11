# Moving `@pmndrs/reflections` from the `reflections-integration` branch to `main`

`@pmndrs/reflections` traces through this package from the `reflections-integration` branch (commit `9aaaa11`, layout version 1, `gi*` names). `main` has the hit normal and per-instance material from that commit, on the 0.1 API. This note covers exactly what reflections calls today, in `packages/core/src/trace/pmndrsBvh.ts` and `packages/three/src/trace/PmndrsBvhTraceProvider.ts`. The three file only builds engine-free inputs (`material` per instance, `matrixWorld`, positions, indices) and needs no change.

## 1. Names and signatures

| Branch (`9aaaa11`) | `main` |
| --- | --- |
| `giTraceClosest(ray: GiRay) -> GiHit` | `bvhTraceClosest(ray: BvhRay) -> BvhHit` |
| `giTraceAny(ray: GiRay) -> bool` | `bvhTraceAny(ray: BvhRay) -> bool` |
| `GiRay(origin, tMin, direction, tMax)` | `BvhRay(origin, tMin, direction, tMax)`, the same 32 bytes |
| `GiHit { t, instance, triangle, barycentrics, frontFace, material, normal }` | `BvhHit` with the same fields in the same order |
| `GI_NO_MATERIAL` (WGSL) | `BVH_NO_MATERIAL` (0xffffffff); TypeScript `NO_MATERIAL`, unchanged |
| `traceModuleWgsl(slot)` | `traceModuleWgsl({ traceSlot: slot })`. The old call **throws** a `TypeError` (an object with `group` or `firstBinding` is refused) instead of being misread. Other options: `triangleFilter` (default: every triangle blocks), `rayMasks` (default true) |
| `TRACE_MODULE_ENTRIES` | unchanged: three read-only storage buffers (nodes, triangles, top level), so the provider's own binding stays at `firstBinding + 3` |
| `HIT_RECORD_WORD.material` (6), `.normal` (7) | unchanged; the record is still 40 bytes by default (section 3) |
| `INSTANCE_MATERIAL_SHIFT` (8), `MAX_INSTANCE_MATERIAL` (2^24 - 2), `instanceFlagsWord`, `instanceMaterialOf` | `MATERIAL_SHIFT` (16), `MAX_MATERIAL` (65534), `materialCode`, `materialFromFlags` |
| `GPU_LAYOUT_VERSION` 1 | 2 (section 6) |

Every name the trace module declares now starts with `bvh`, `Bvh` or `BVH_`, so reflections' own WGSL names must not (`pbvh*`, `traceFirstHit`, `Hit`, `HitMaterial` are fine). `createMeshLayoutBuilder(device, { prefer, label })`, `buildLayouts`, `assembleTraceScene(layouts, instances)` and `buildTopLevel(instances)` keep the signatures `PmndrsBvhModule` declares.

The provider's WGSL becomes:

```wgsl
${bvh.traceModuleWgsl({ traceSlot: slot })}
fn traceFirstHit(origin: vec3f, dir: vec3f, tMax: f32) -> Hit {
  let h = bvhTraceClosest(BvhRay(origin, 0.0, dir, tMax));
  if (h.t < 0.0) { return noHit(); }
  return Hit(h.t, h.triangle, h.barycentrics, h.material, h.instance, h.normal, h.frontFace);
}
fn traceAny(origin: vec3f, dir: vec3f, tMax: f32) -> bool {
  return bvhTraceAny(BvhRay(origin, 0.0, dir, tMax));
}
```

and `PmndrsBvhModule.traceModuleWgsl` is typed `(options: { traceSlot?: BindingSlot; triangleFilter?: string | null; rayMasks?: boolean }) => string`.

## 2. Hit normal and material

Same meaning as on the branch:

- **`BvhHit.normal`**: the world-space geometric normal of the authored front side, unit length (0 on a miss), so `frontFace == dot(normal, direction) < 0`, also through a mirror. Geometric only: the layout has no vertex normals, so reflections' `traceHitShadingNormal` (its own normals at `hit.triangle` with `hit.barycentrics`) stays as it is.
- **`BvhHit.material`**: the instance's `material`, or `BVH_NO_MATERIAL`. Set it with `material` on the `assembleTraceScene` and `buildTopLevel` instances, as reflections already does. Passing `NO_MATERIAL` or leaving it out both mean none.

What changed:

- **Range.** A material is 0-65534 (it was up to 2^24 - 2). Anything else throws a `RangeError` naming the instance. Reflections' check (`instance.material < materials.length`) stays, and a scene with more than 65535 materials fails at assembly.
- **Storage.** It is `material + 1` in instance flags bits 16-31 (it was bits 8-31; bits 8-15 now hold the mask). Reflections never reads the word itself, so nothing changes there.
- **Cost.** The walk no longer works out the normal and material at every accepted triangle. It keeps the instance record and triangle slot and works them out once, after the walk.

## 3. Batch hit records

Reflections traces inline, through `traceModuleWgsl`, so this only matters for `HIT_RECORD_WORD`, which `loadPmndrsBvh` checks. `batchKernelWgsl(mode, { hitNormal })`:

| `hitNormal` | Struct | Bytes | Layout (32-bit words) |
| --- | --- | --- | --- |
| `true` (default) | `BvhHitRecord` | `HIT_RECORD_BYTES` = 40 | `HIT_RECORD_WORD`: 0 t, 1 instance, 2 triangle, 3 frontFace (0/1), 4-5 barycentrics, 6 material, 7-9 normal |
| `false` | `BvhHitRecordCompact` | `HIT_RECORD_COMPACT_BYTES` = 24 | `HIT_RECORD_COMPACT_WORD`: words 0-5 as above, no material and no normal |

The 24-byte record has no padding. A seventh word would round it up to 32 bytes, so the compact record leaves the material out. `instance` names the material in the caller's own table, and on a merged hit it is the source instance. `hitRecordBytes(hitNormal)` gives the size. The existing check `HIT_RECORD_WORD.normal !== undefined && HIT_RECORD_WORD.material !== undefined` still passes on `main`. Only its error message names the branch.

## 4. Instance masks, and why mask 0 is never hit

Version 2 adds an 8-bit instance mask in instance flags bits 8-15. A masked trace (`bvhTraceClosestMasked(ray, mask)`, `bvhTraceAnyMasked`) skips an instance whose `(instanceMask & rayMask) == 0`. The plain `bvhTraceClosest`/`bvhTraceAny` trace with all ones. So **an instance record whose mask bits are 0 is never hit, by the plain functions too.**

- `buildTopLevel` and `assembleTraceScene` write `0xff` (visible to every ray) unless the instance gives `mask` (0-255). Reflections builds every record through them, in `createPmndrsBvhTraceProvider` and in `setTransforms`, so its records are visible. Nothing to do.
- Code that writes instance records itself must set bits 8-15, or no ray sees those instances. A GPU top level is one example.
- `traceModuleWgsl({ traceSlot, rayMasks: false })` compiles the mask code out. The walk then has no `*Masked` functions and no mask test, and it ignores the bits. Reflections doesn't use masks, so this is an option. It is the same walk with the mask code removed, and the quick bench showed no measurable difference.

## 5. Merging static instances (optional, off by default)

`mergeStaticInstances` puts static instances whose mesh no other instance uses into one world-space tree. Rays that start on surfaces, such as reflection rays, then walk one tree instead of entering many overlapping instance boxes. Measured in webgpu-ddgi-radiance-cache on an M5 Pro, closest hit was 1.86x faster on Sponza and 1.24x on Bistro interior. Hits are unchanged:

- `instance` names the source instance;
- `triangle` and `barycentrics` are the source mesh's;
- `frontFace` and `normal` are the authored side's;
- `material` is the source's, carried in the merged triangle's flags bits 16-31. Material is not a grouping key.

What it takes in reflections:

1. Mark only instances that never move with `static: true` on the `assembleTraceScene` instances. `setTransforms` rebuilds the top level from new matrices, so any instance it may move must stay `static: false`.
2. Replace the single `assembleTraceScene` call with this sequence:
   - `const plan = mergeStaticGeometry(meshes, instances)`;
   - build `plan.groups[g].geometry` with the same `buildLayouts`;
   - meshes with `plan.meshNeeded[i] === 0` need no tree (pass `null`);
   - `assembleTraceScene(layouts, instances, {}, { merged: { plan, meshes: groupLayouts } })`.

   Alternatively, `buildTraceScene(geometries, instances, { build, mergeStaticInstances: true })` does all of it.
3. **`setTransforms` must change.** A merged mesh's entry in `scene.meshes` is empty, and the merged groups need their own records, so `buildTopLevel` over every instance is wrong. Assemble again with the same plan and trees, and upload only `topLevel.buffer`, because nodes and triangles come out the same. Assembly throws if a merged instance's transform, mask, visibility or material changed. Only re-assembly checks this.
4. Merged instances need distinct `instanceIndex` values. Reflections uses the position, so they are.

Leaving it off changes nothing: the scene is byte for byte the instanced one.

## 6. Layout version 2

`GPU_LAYOUT_VERSION` is 2. It isn't released yet and stays 2 for everything in this note. Compared with version 1:

- **Instance flags:** bits 0-2 are unchanged. Bit 3 is `merged`, bits 8-15 the mask, and bits 16-31 the material + 1.
- **Triangle records:** flags bit 1 flips `frontFace` on merged triangles, the padding word (word 11) holds a merged triangle's source instance, and flags bits 16-31 its material. A mesh input's `triangleFlags` may still set bit 0 only.
- **Sizes:** nothing moved. Nodes are 64 bytes, triangles 48 and instance records 64.

In reflections, set `PMNDRS_BVH_LAYOUT_VERSION = 2`, or `loadPmndrsBvh` refuses `main`. Serialized trees from version 1 are refused by `deserializeLayout`. Reflections builds its trees at startup, so it has none.

## Checklist

1. `PMNDRS_BVH_LAYOUT_VERSION = 2`.
2. `bvh.traceModuleWgsl({ traceSlot: slot })`, and update the `PmndrsBvhModule` type.
3. In `providerWgsl`, rename `giTraceClosest`, `giTraceAny` and `GiRay` to `bvhTraceClosest`, `bvhTraceAny` and `BvhRay`.
4. Keep materials at 65534 or below.
5. Optional: `rayMasks: false`. Also optional: `mergeStaticInstances` for static scenes, with the `setTransforms` change in section 5.
6. Run reflections' `pnpm check` against `main`.

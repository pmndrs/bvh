# Scene files: the snapshot and the movable scene

Decision 4a in `DECISIONS-0.1.md` kept the `'PBVS'` scene file a snapshot and left room for "a movable scene file
[...] as its own kind". Dennis asked for it as an option (2026-11-16): instanced objects move often. This note says what
each kind is for and why the movable one is built the way it is. The user-facing page is
`docs/serialization/serialization-and-baking.mdx` ("Which file?", "Movable scene files").

## The scenarios

| Scene | File | Why | What loading costs |
| --- | --- | --- | --- |
| A static level | Scene (`'PBVS'`) | Everything is baked, the top level too | `deserializeTraceScene` (copies) and `uploadTraceScene`: no build, no assembly |
| A level with a few moving props | Movable (`'PBVM'`), the level `static: true` | The level merges into world-space groups whose trees are in the file; the props stay instances | No mesh or group build; the first `SceneTrees.update` builds the top level over the props and a few group records |
| Many instanced objects moving every frame | Movable, nothing static | One shared tree per mesh; the top level changes every frame anyway | No mesh build; the top level as `SceneTrees` keeps it every frame (GPU refit and worker shapes, or the CPU top level) |
| Streaming, editors | A tree per mesh (`placeTree`), or a movable file per part with nothing static | Parts come and go; no part owns the top level | No mesh build per part. Merging only happens at the first update, so a part loaded later keeps its static instances as instances and builds the meshes only they used |
| Deforming meshes | A tree per mesh, or a movable file with `deforming` | The rest pose's tree is a starting point | The refit per pose, and drift rebuilds, as without a file |

The snapshot stays the best file when nothing moves: its load is a copy, and a movable file of the same scene still
needs a top-level build (cheap next to the meshes, but not nothing) and a `SceneTrees`.

## Why a new kind, not a new version of `'PBVS'`

The snapshot's buffers are the assembled scene's: every tree at its offset in one node buffer and one triangle buffer,
the top level, and placements (`TraceSceneMeshPlacement`) pointing into them. A `SceneTrees` places trees itself (its
own allocators, growth, quantize passes, indexed headers, merged identity) and builds its own top level, so it can use
none of that layout. What it can use is what it already takes from an app: one tree per mesh (`placeTree`) and
instances (`addInstance`). So the movable file stores those, and reuses the tree file whole:

- each tree is a complete `'PBVH'` file (`serializeLayout`, with its geometry's hash in words 14-15), read back with
  `deserializeLayout`, so the tree format, its checks and its hash are the tree file's, unchanged;
- the description between the header and the trees is JSON, padded to a word (decision 4b), with fixed field order
  so a scene writes the same bytes every time;
- the hash is `meshGeometryHash`, positions and indices only (decision 4c).

It has its own magic (`'PBVM'`) and format version (`SERIALIZED_MOVABLE_SCENE_FORMAT_VERSION`, 1), and the same words
0-2, so `readSerializedHeader` and `isSerializedCurrent` read all three kinds and `'PBVS'` files load as before.
`deserializeLayout` and `deserializeTraceScene` refuse a `'PBVM'` file by name, and `deserializeMovableScene` refuses
the other two.

## Merged static groups

Three options were open:

1. **Merge at load, build the groups then.** The file holds every mesh's tree and the static flags; `SceneTrees` merges
   at its first update and builds each group (GPU tree first, tinybvh's later). Simple, but the level, the biggest part
   of such a scene, is built at every load: Sponza's merged group takes tinybvh 1.5 s in a worker, and the GPU's
   H-PLOC tree is the less good one until it swaps.
2. **Store the groups as plain meshes.** Loses hit identity: a merged triangle reports its source instance and material
   (`writeMergedTriangleIdentity`), which a plain mesh under one instance can't.
3. **Merge at bake time and store the groups' trees (built).** `buildMovableScene` runs `mergeStaticGeometry` over the
   same meshes and instances as `SceneTrees.planMerges` will (the scene's `triangleLeaves`, deforming meshes' instances
   left out), builds each group's world-space geometry, and stores its tree with that geometry's hash. Meshes only
   merged instances use get no tree. At load, `placeMovableScene` hands the trees to
   `SceneTrees.placeMergedTrees`; the first update merges as always, then checks each group against its tree (the
   number of groups, triangle counts, the hash of the group's geometry) and places them as app trees, so no build
   starts. The ids are written into the records as any group's are (into a copy: the caller's tree is left as baked).

Option 3 is built. A mismatch (the app added other static instances before the first update, or a static instance's
transform changed) is not an error: the trees are dropped, the groups are built as in option 1, and `warn` says why. The
scene is right either way; only the load is slower. A movable file baked with `mergeStaticInstances: false` is option
1.

The merge is deterministic: the plan depends only on the meshes, the instances (in order) and their fields, and the
transforms round-trip through JSON exactly (JSON numbers are doubles, and `SceneTrees` keeps transforms as doubles).
The loader adds meshes and instances in the file's order, so a fresh `SceneTrees` makes the bake's groups.

## The geometry

`SceneTrees.addMesh` needs geometry even when the tree is placed (bounds, an indexed scene's positions, the merge's
world-space groups, a deforming mesh's poses). Apps that move instances usually have it (they draw the meshes), so by
default the file holds only each mesh's hash and `deserializeMovableScene(bytes, { geometries })` checks the app's
against it (`deserializeLayout`'s checks for meshes with a tree; the hash for merged-away meshes). `{ geometry: true }`
stores positions, indices, triangle flags and materials (as `u16`, `0xffff` for none) so a file loads with no model:
the bake CLI writes that, so the example page can load any movable file.

## API

- `buildMovableScene(meshes, instances, { build, triangleLeaves, nodeFormat, mergeStaticInstances, deforming })` builds
  the trees (an `f32` tree is quantized here for a quantized scene, except a deforming mesh's).
- `serializeMovableScene(scene, { geometry })`, `deserializeMovableScene(bytes, { geometries })`.
- `placeMovableScene(trees, scene)` adds everything to a `SceneTrees` and returns the mesh and instance handles.
- `assembleMovableScene(scene)`: the same scene as a `TraceSceneData` (what `buildTraceScene` gives), for an upload,
  the CPU walk, or checking a `SceneTrees`.
- `SceneTrees.placeMergedTrees(trees)`: the hook the loader uses; false when the scene merges nothing more
  (`mergeStaticInstances: false`, or past its first update).

## Checks

`src/serialize-movable-scene.test.ts`: every format (records and indexed, `f32` and quantized, merged or not, with and
without geometry) writes the same bytes back and traces as `buildTraceScene` does; a moved instance traces as a fresh
build with it moved; deforming meshes stay out of the merge; other geometry, versions and kinds are refused.
`src/scene/scene-trees-movable.test.ts` (fake GPU): loaded into `SceneTrees`, no build starts (GPU or tinybvh), every
tree is `'app'`, the walk agrees with `assembleMovableScene` and, after the props move, with a fresh build; a group
that isn't the bake's is built with a warning. The `node/bake` page's smoke run loads the default scene as a movable
file into `SceneTrees` on the GPU and checks the view against the CPU walk, still and moved.

# `@pmndrs/bvh/three`

> The user docs for this subpath are the pages in `docs/three/` (the docs site). This file keeps the full contributor notes: how each part matches three, what is tested, and the open questions.

The three.js side of the package. It's the only part that imports three, which is an optional peer dependency pinned to `>=0.187.0 <0.188.0`: `BvhTraceNodes` relies on three's WebGPU backend internals, so each three release is checked before the range widens. The core stays engine-free, and lint enforces that.

## Coming from three-mesh-bvh's WebGPU path

| three-mesh-bvh 0.9 (`three-mesh-bvh/webgpu`) | `@pmndrs/bvh/three` |
| --- | --- |
| `new BVHComputeData(root, { attributes })` | `bvhSceneFromObject3D(root, options)` gives the meshes and instances. `SceneTrees` builds the trees (GPU H-PLOC first, tinybvh swapped in) on `deviceFromRenderer(renderer)`. `new SceneTreesObject3DSync(trees, root, options)` keeps it in step. `new BvhTraceNodes(renderer, trees)` exposes the trees in three's node system. |
| `bvh.update()` (repack everything) | `sync.sync()` when objects are added or removed: only the meshes and instances that changed are added or removed. |
| `bvh.updateTransforms()` | `sync.update()` each frame (only moved, non-static instances), then `trees.update(encoder)`, then `bvh.update()`, which rebinds the storage nodes when `SceneTrees.version` changed. |
| `bvh.storage.nodes / transforms / index / attributes` | `bvh.storage.nodes / triangles / topLevel`, which hold our layout (docs/internals/design/trace.md). There's no per-vertex attribute buffer: use `hit.instance` (via `sync.sourceOf`), `hit.triangle` (the triangle in its mesh, via `threeFaceIndex`) and `hit.barycentrics` against your own attributes, or `hit.material`. |
| `bvh.fns.raycastFirstHit(ray, &hit) -> bool` | WGSL: `bvhTraceClosest(ray: BvhRay) -> BvhHit` (`hit.t < 0` on a miss). TSL: `bvh.traceClosest(origin, direction, tMin, tMax)`, a `BvhTraceHit` struct node. |
| `bvh.fns.raycastAnyHit(ray) -> bool` | WGSL: `bvhTraceAny(ray) -> bool`. TSL: `bvh.traceAny(origin, direction, tMin, tMax)`. |
| (none) | Ray masks: `bvhTraceClosestMasked(ray, mask)` / `bvhTraceAnyMasked`, or TSL `traceClosestMasked` / `traceAnyMasked`. Use `rayMaskFromLayers(camera.layers)`. |
| `bvh.fns.closestPointToPoint` | With `new BvhTraceNodes(renderer, trees, { queries: true })`: WGSL `bvhClosestPoint(point, maxDistance) -> BvhClosestHit` over the whole scene in world space; TSL `bvh.closestPoint(point, maxDistance)`, a `BvhClosestPointHit` struct node (`bvhClosestPointHitStruct`). Also `bvhIntersectsSphere`, `bvhIntersectsBox`, `bvhIntersectsOrientedBox` (docs/tracing/gpu-queries.mdx). |
| `getShapecastFn({ shapeStruct, intersectsBoundsFn, intersectRangeFn, ... })` | `bvh.getShapecastFn({ name, shape, intersectsBounds, intersectsTriangle, keep })` (needs `queries: true`): hooks as three function nodes (a `wgslFn`, or a TSL `Fn` with `setLayout`), compiled into a walk stamped per cast (`shapecastWalkWgsl`, nestable) by `BvhShapecastWalkNode`, a `BvhTraceModuleNode` whose `textFor` builds each hook's call when the shader generates (so three emits the hook as its own function, renamed if taken, and the adapter calls the name three gave it). The shape (any TSL type or a `StructTypeNode`) is a `var<private>` set by the entry and passed to every hook first; the triangle hook gets only the leading arguments it declares (`a, b, c, instance, triangle, material, normal`). Returns `(shape?, mask?) => bool`. Differences from three-mesh-bvh's: one world-space triangle per call rather than a leaf's index range, no result pointer (state through a `wgslFn`'s `var<private>`). Node test `src/three/shapecast-fn.test.ts` (three's WGSL builder on a stub renderer); GPU page `three-shapecast`. The module's own option `shapecast` (WGSL hooks `bvhShapecastBounds`, `bvhShapecastKeep`, `bvhShapecastTriangle` in `bvhShapecast()`) stays. |
| `await bvh.updateAsync()` | `await bvh.updateAsync({ sync, until })` on `BvhTraceNodes`, or `updateSceneTreesAsync(trees, { sync, nodes, until })`: `sync.sync()`, `await sync.ready`, an update in its own encoder (starts every build), `whenBuilt()` / `whenRefined()`, another update (places the trees), `nodes.update()`. three-mesh-bvh awaits its `getBVH` for every object and then repacks; here the builds already run asynchronously, so this only waits for them. |
| `intersectRayTriangle`, `closestPointToTriangle` (WGSL helpers) | `bvhIntersectRayTriangle(ray, a, b, c)` (the walk's watertight test) and `bvhClosestPointToTriangle(point, a, b, c)`, in `bvh.module`, or binding-free in `triangleGeometryWgsl()`. |
| `wgslTagFn` with `${bvh.fns.raycastFirstHit}` | three's own `wgslFn(code, [bvh.module])`, calling the walk by name. |
| `rayStruct` (`origin, direction, maxDist`; 0 means unbounded) | `BvhRay(origin, tMin, direction, tMax)`, with an explicit interval. |
| `rayIntersectionResultStruct` | `BvhHit { t, instance, triangle, barycentrics, frontFace, material, normal }` in WGSL; `bvhTraceHitStruct` (`BvhTraceHit`, the same members) for TSL. |
| `hit.dist`, `hit.didHit` | `hit.t`, and `hit.t >= 0.0` |
| `hit.normal` (geometric, world) and `hit.side` | `hit.normal` is the world geometric normal of the authored front side. `hit.frontFace` is true when the ray hit that side (`side = select(-1, 1, frontFace)`). |
| `hit.barycoord` (weights of a, b, c) | `hit.barycentrics` holds the weights of corners 1 and 2. Corner 0 gets `1 - u - v`. |
| `hit.indices` (the vertex indices) | `hit.triangle`: the triangle's index within its mesh (a group, draw range or batch range), as built. |
| `hit.objectIndex` (the transform slot) | `hit.instance`, the SceneTrees instance. `sync.sourceOf(hit.instance)` gives `{ object, instanceId, group }`. |
| `writeTransformData` override (material in the pad words) | `material(object, info)` in the options, which becomes `hit.material` (0-65534). |
| `MeshBVH` per geometry (`geometry.boundsTree`) | Trees are per geometry and triangle range, built by SceneTrees (GPU, then a tinybvh worker pool), or by `buildTraceScene` with any builder. For CPU queries, `geometry.boundsTree` is a `BvhBoundsTree` ("CPU raycast" below). |

## Scenes (`bvhSceneFromObject3D`)

- **Objects and instances:**
  - Reads `Mesh`, `InstancedMesh` (instances below its `count`), `BatchedMesh` (active instances; `getVisibleAt` too) and `SkinnedMesh` (rest pose; skinning and morph targets aren't applied by `bvhSceneFromObject3D`; the sync poses them on the GPU, below).
  - Reads the triangles three's `Mesh.raycast` tests: the draw range, and with an array of materials, the groups within it. `bvhSceneFromObject3D`, the sync and `BvhSceneRaycaster` all fold a multi-material mesh into one mesh (its groups one after another) and one instance per object or instance (`multiMaterial: 'triangles'`, the default), so instance N is the same source in all three. Each triangle carries its group's material (`triangleMaterials`): `material(object, info)` is asked once per group (`info.group`, `info.materialIndex`), undefined leaves the group without one, and objects whose answers differ get meshes of their own. An object whose instances give a group different answers is not folded (one folded mesh per instance would copy the mesh 10,000 times for 10,000 instances): it keeps one mesh per group, shared by its instances, with the answers on the instances. A folded source has `group` and `materialIndex` -1; `meshSources[i].groups` (`sync.meshSourceOf`, `picker.meshSourceOf`), `threeGroupOf(meshSource, triangle)` and `threeMaterialOf(source, meshSource, triangle)` map a hit back to its group and three material (`threeFaceIndex` works as before). `multiMaterial: 'groups'` keeps one mesh and instance per group everywhere. A posed (skinned or morphed) multi-material object folds into one posed mesh.
  - Ranges are read as three steps through them: three indices at a time from the range's start while the index is below its end. A range that starts off a multiple of 3 gives triangles made of neighbouring triangles' corners, and one whose length isn't a multiple of 3 ends in a triangle that reads past it; both are in the mesh, as in three. Overlapping groups are both read. A triangle that would read past the index (or the positions) is dropped: three r187 reports it at a `NaN` distance with `NaN` corners (and its sorted list comes out of order there), which no ray can hit (`threeTriangleRanges`). A known three behaviour, accepted by Dennis on 2026-11-13; not filed upstream.
  - Takes indexed, non-indexed and interleaved positions. Interleaved (or normalized, or non-float) positions are compacted once per geometry and shared by all its groups' meshes.
  - Makes one mesh per geometry and range, shared by every object and instance that uses it.
  - Call `updateMatrixWorld()` first: the world matrices are read as they are.
- **Visibility:** an object counts as visible when it and every ancestor are. Hidden objects stay in as hidden instances.
- **Layers:** layers 0-7 become instance mask bits 0-7, and layers 8-31 are dropped. An object on none of layers 0-7 gets mask 0, which no ray hits, not even the plain `bvhTraceClosest`/`bvhTraceAny` (all ones). A default object (layer 0) gets mask `0x01`. `layers: false` gives every instance `0xff`.
- **Static:** `static: true`, or `(object, instanceId) => boolean`, marks instances for `mergeStaticInstances` (on by default in `SceneTrees` and `buildTraceScene`, so marked instances merge at the first update). The default is `false`: nothing merges unless marked. The sync never pushes changes to static instances.
- **Merged static content can't change.** After `SceneTrees.update` merged an object's static instances, `sync.sync()` skips removing that object or changing its geometry, draw range, groups or materials, and warns once per object (`skipped` in the counts). The merged triangles stay, and `sync.sourceOf` still maps their hits. Mark objects that may go away `static: false`. A new object on a geometry whose mesh was merged away gets the mesh added again.
- **Materials:** the `material` callback is read at `sync.sync()` only. After changing what it returns, call `sync.sync()`.
- **`sync.update()` cost:** it skips an object whose `matrixWorld`, effective visibility, layers and `instanceMatrix.version` (a `BatchedMesh`'s matrices texture version) are unchanged, without allocating. Set `instanceMatrix.needsUpdate = true` after `setMatrixAt`, as three needs anyway. A `BatchedMesh`'s per-instance visibility is still compared each frame, since three gives it no version.
- **Failures:** `sync.sync()` walks and plans before it changes anything, so a throwing `material` callback leaves everything as it was. Each `SceneTrees` call updates the sync's records along with it, so a later `sync()` picks up where a failed one stopped.
- **Without SceneTrees:** `buildTraceScene(scene.meshes, scene.instances, { build })` builds it all at once (merging the instances marked static; `mergeStaticInstances: false` turns that off). `uploadTraceScene` then gives buffers that `BvhTraceNodes` takes as `{ buffers }`.

## Skinned and morphed meshes (`BvhPosePass`)

```ts
const sync = new SceneTreesObject3DSync(trees, scene, { renderer }); // the renderer runs the pose passes
// each frame, after the mixer and scene.updateMatrixWorld():
sync.update();               // transforms, then every visible posed object's pass, then SceneTrees.setPose
trees.update(encoder); submit; bvh.update();
```

- **Which meshes:** with a `renderer`, every `SkinnedMesh` and every mesh with position morph targets (`threeDeforms(mesh)`), unless `deforming: false` or a `(mesh) => boolean` says otherwise. Without a renderer they stay in their rest pose, as before. `BatchedMesh` is never posed, nor an `InstancedMesh` with per-instance morphs (`morphTexture`) and more than one instance: the pass would give every instance `morphTargetInfluences`, so the sync warns once and reads it in its rest pose.
- **Their own meshes:** a posed object gets its own deforming SceneTrees mesh per triangle range (not shared with other objects on the geometry, since each has its own pose). Its instances never merge.
- **The pass** (`BvhPosePass`, `src/three/deform.ts`) is a TSL compute pass with three r187's math in three's order: morphs as `morphReference` does them (`p *= base`, then `p += target * influence` for each non-zero influence), then skinning as `getSkinnedPosition` does it (`bindMatrixInverse * sum(bone * weight * (bindMatrix * p))`, after `skeleton.update()`). It writes one `vec4` per vertex into a buffer of ours, which `SceneTrees.setPose` refits the tree from (`GpuMeshRefit`): nothing is read back. On the M5 the posed vertices match three's CPU pose (`getVertexPosition`, `applyBoneTransform`) to 6.4e-7.
- **Compile cost:** one shader per variant (skinned, morphed, both). Counts are uniforms, the morph loop is a `Loop` over a uniform, the storage arrays are runtime-sized, so every posed mesh of a variant shares one pipeline and the WGSL is the same on every load. `pass.prepare()` compiles it with `compileComputeAsync`; the sync calls it for the first mesh of each variant when the mesh is added (and adds it without a dispatch), so the first `update()` doesn't compile. three makes that pipeline asynchronously and a dispatch before it exists fails, so until the compile settles `update()` doesn't pose that variant's objects: they keep their last pose (the rest pose before the first) and their trees as they were, counted in `sync.poseCounts.waiting`. `await sync.ready` (it never rejects) has every object posed from the first `update()`. Calling `BvhPosePass.update()` yourself: await its `prepare()` first.
- **The top-level box follows the pose:** SceneTrees keeps one box per deforming mesh in its top level. The sync sets it (`SceneTrees.setDeformBounds`) from `pass.bounds()`, a box that holds the pose computed from the skeleton on the CPU (per bone, the box of the vertices it moves, carried through the bone matrix: 16 corner transforms per bone). It only moves when the pose leaves the box or the box grows past twice the padded pose; `deformSlack` (default 0.25 of the largest extent, on every side) sets the room. A walking character's root motion is followed. Negative skin weights fall back to a box from every vertex on the CPU, with the kernel's own formula (its `bindMatrixInverse` translation counts once per unit of weight sum, as on the GPU; three's `getVertexPosition` counts it once). The box has room for the kernel's f32 rounding: `(8 + targets) · 2⁻²³` of the largest coordinate the kernel sees (16 roundings plus 2 per morph target, each at most half an ulp), on top of `1e-4` of the extent, and its ends are rounded outward to f32, so a character 5,000 units from the origin (f32 steps of 4.9e-4) stays inside it. A box that isn't finite (NaN bone matrices) throws from `sync.update()` rather than keeping the old one.
- **A replaced skeleton** (another object, or another bone count) gets a new pass at the next `sync.update()`; the old pass throws from `update()` and `bounds()` (`pass.current()` says whether it still matches).
- **Inputs are read once:** the rest positions, skin indices and weights and morph targets are packed at the pass's creation (through the attributes, so interleaved and normalized ones work). A new geometry object gets a new pass at `sync.sync()`; editing those attributes in place is not followed.
- **Not read:** `InstancedMesh.morphTexture` (per-instance morphs: refused, see above), morphed normals (the walk uses geometric normals).
- **Hidden objects aren't posed.** `sync.pose()` runs the passes alone; `sync.poseCounts` says how many ran and how many boxes moved; `sync.posePassOf(object)` gives the pass (its `buffer` holds the last pose).
- **CPU picking on posed meshes stays on three's raycast** (`acceleratedRaycast` falls back for a `SkinnedMesh` or morph targets): three poses each vertex on the CPU there, so it's exact and linear in the vertex count. three-mesh-bvh's `StaticGeometryGenerator` route (pose on the CPU, then build or refit a tree) costs the same CPU pose per frame plus a refit, and only pays off for many raycasts per frame on one pose, or for shapecasts and closest-point queries on the pose (PARITY 6.3): `StaticGeometryGenerator` in `/three` (`src/three/static-geometry.ts`) is three-mesh-bvh 0.9.18's class ported as it is (MIT, licence text in its header), bit for bit equal to its output and to three's `getVertexPosition`; its diff also compares `morphTargetInfluences` (three-mesh-bvh's doesn't, and keeps a mesh whose only change is a morph weight).
- **Tested:** `src/three/deform.test.ts` (the box holds three's CPU pose and an f32 emulation of the kernel over random frames: relative and absolute morphs, root motion, weights across 3-4 bones, a bind matrix that isn't the identity, characters thousands of units from the origin, negative weights that don't sum to 1; `dispose()` releases every attribute; the sync's bookkeeping, a replaced skeleton, a NaN box, `morphTexture`, one compile per variant), and the GPU page `three-skinned` (below), which also reads the posed buffer back and checks every vertex lies in the box `setDeformBounds` got.

## In three's node system (`BvhTraceNodes`)

```ts
const device = deviceFromRenderer(renderer);          // after await renderer.init()
const trees = await SceneTrees.create(device);
const sync = new SceneTreesObject3DSync(trees, scene, { static: (o) => !o.userData.moves });
const bvh = new BvhTraceNodes(renderer, trees);

// TSL
const hit = bvh.traceClosest(origin, direction, 0, 1e30).toVar();
If(hit.get('t').greaterThanEqual(0), () => { /* hit.get('normal'), hit.get('material'), ... */ });

// WGSL
const shadow = wgslFn(`fn shadow(p: vec3f, l: vec3f, d: f32) -> f32 {
  return select(1.0, 0.0, bvhTraceAny(BvhRay(p, 1e-3, l, d)));
}`, [bvh.module]);

// each frame
scene.updateMatrixWorld();
sync.update();
const encoder = device.createCommandEncoder();
trees.update(encoder);
device.queue.submit([encoder.finish()]);
bvh.update();
renderer.compute(kernel); // or render
```

- **Buffers:** the storage nodes are bound to SceneTrees' own `GPUBuffer`s, with no copy. `BvhTraceNodes` sets three's backend buffer for each node before three first uses it. That's three's internals, not its API. They're listed in one place, `threeInternals` in `src/three/trace-nodes.ts`.
- **Indexed leaves:** a `SceneTrees` made with `triangleLeaves: 'indexed'` (docs/internals/design/indexed-leaves.md) has a fourth buffer, its positions. `BvhTraceNodes` reads the format from its source (`triangleLeaves`, `positionLayout`), binds the fourth storage node (`bvh.storage.positions`) and compiles the walk for it; nothing else changes for the caller.
- **Checked:** once three has used a storage node, `bvh.update()` checks that three still binds our buffer. If a three version looks buffers up differently, `update()` throws an error naming that version as unsupported. Without the check, the walk would read a 16-byte placeholder.
- **Disposal:** `renderer.dispose()` and a stand-in attribute's `dispose()` don't destroy SceneTrees' buffers. `BvhTraceNodes` wraps the renderer's `backend.destroyAttribute` so a throwaway buffer takes our buffer's place before three destroys it.
- **When buffers are replaced:** `bvh.update()` swaps in a new attribute bound to the new buffer, and three makes new bind groups. Nothing recompiles. **Call `bvh.update()` after every `SceneTrees.update` that may grow a buffer, and before three uses the nodes again.** Otherwise three keeps binding the old buffer, which SceneTrees has replaced.
- **Shader size:** the walk goes into each shader once, as plain WGSL (one loop with a 64-entry stack), and each TSL helper is one small function. The names are fixed, so the shader text is the same on every load.

## CPU raycast

These are drop-ins for three-mesh-bvh's functions of the same names. They run the CPU walk (`createCpuTracer`) over one tinybvh tree per geometry, held by the core's `CpuMeshBvh` ("The tree object" below):

```ts
await prepareBoundsTrees();                         // loads tinybvh (WebAssembly) once
BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
Mesh.prototype.raycast = acceleratedRaycast;
geometry.computeBoundsTree();
raycaster.firstHitOnly = true;                      // optional
```

- **Types:** `/three` augments three's types as three-mesh-bvh does: `BufferGeometry.boundsTree` (a `BvhBoundsTree`), `computeBoundsTree`, `disposeBoundsTree` and `Raycaster.firstHitOnly`, so the lines above need no casts. A program that imports both libraries gets two declarations of the geometry members with different types (TS2717): `skipLibCheck` (on in most templates) hides it, and the first declaration read gives the type. Our own sources carry `@ts-ignore` there, as our tests import three-mesh-bvh; the emitted `.d.ts` drops it.
- **What three would return:** the same intersections as three's `Mesh.raycast`:
  - the draw range, and with an array of materials, the groups;
  - `side` (front, back or double);
  - `near` and `far`.
- **The same fields:** `distance`, `point`, `face` (`a`, `b`, `c`, `normal`, `materialIndex`), `faceIndex`, `barycoord`, `uv`, `uv1` and `normal`. `InstancedMesh` and `BatchedMesh` get these through `Mesh.raycast`.
- **BatchedMesh:** call `batchedMesh.geometry.computeBoundsTree()`. three raycasts each instance through a stand-in mesh that shares the batched geometry's attributes, and the tree is found by its position attribute. Each geometry's range is cached once.
- **BatchedMesh per geometry (PARITY 5.5):** `computeBatchedBoundsTree(index = -1, options)` / `disposeBatchedBoundsTree(index)`, three-mesh-bvh's, for `BatchedMesh.prototype.computeBoundsTree`: `batchedMesh.boundsTrees[geometryId]` is a `BvhBoundsTree` over a view geometry (the batched attributes shared, an index over the range's whole triangles: a subarray of the batch's, or `start, start + 1, ...` without one), registered by the batched position attribute and `start:count`. `acceleratedRaycast` finds it from the stand-in's draw range (before the whole-geometry tree), walks all of it and maps a hit's local first index `3k` to `start + 3k` (three's `faceIndex` is that over 3). A changed batch index version, or position version, is the stale path (three's raycast, one warning). On a `BatchedMesh` itself, `acceleratedRaycast` runs three's captured `BatchedMesh.prototype.raycast`, so three-mesh-bvh's setup (both prototypes patched) works. `_geometryCount` / `_geometryInfo` are read as three-mesh-bvh reads them (r187 internals). three-mesh-bvh builds these trees without `indirect`, so it reorders the batch's index in place; ours never touch it.
- **Materials:** the triangle ranges are cached per material and draw range, and checked against the groups and the materials an array holds. Swapping `mesh.material[i]` is picked up, and each material's `side` is read at every raycast.
- **All hits:** without `firstHitOnly`, every hit is returned, as three does. The triangle filter collects them.
- **Falls back to three's raycast** when the geometry has no tree, for a `SkinnedMesh`, or when there are morph targets.
- **Queries read the tree's triangles.** `shapecast`, `bvhcast` and `intersects*` hand over the corners stored in the tree, not the geometry's current positions. After editing positions, call `boundsTree.refit()` first (three-mesh-bvh reads the geometry, so it sees edits straight away). `intersectsGeometry` against a bare geometry recomputes that geometry's bounds rather than trusting `geometry.boundingBox`.
- **Edited geometry:** if the position attribute's version, or the index (object or version), changed since `computeBoundsTree`, the raycast falls back to three's own and warns once. It stays that way until `geometry.boundsTree.refit()` (same index) or `computeBoundsTree` runs again.
- **Ranges off a multiple of 3 and overlapping groups** are matched. A range that starts at index 1 or 2 mod 3 is walked in a second tree over the triangles that start there, built with the tree's options on the first raycast that needs it (it needs tinybvh, loaded by `prepareBoundsTrees`; without it the raycast falls back to three's) and refitted with the tree (`refit()` from the geometry, `refit(input)` from that input). It holds only the shifted ranges' triangles (a 150-triangle group gets a 150-triangle tree), or every triangle of its shift when the ranges hold half of them or more; plain f32 positions are used in place, and otherwise only the vertices those triangles use are copied. Up to 16 subset trees are kept per geometry. Overlapping groups are walked once each, so a triangle in two groups is reported twice, as three does. The one difference: three reports the triangle that reads past the end of the index at a `NaN` distance (and its sort is then out of order); we report nothing there.
- **Tested:** `src/three/raycast.test.ts` checks it against three on 8,000 random rays, and 5,000 more through ranges and groups off a multiple of 3.

### A whole scene at once (`BvhSceneRaycaster`, `raycastScene`)

three-mesh-bvh's `ObjectBVH`: one walk over the scene's top level and its mesh trees, in place of a raycast per object.

```ts
await prepareBoundsTrees();
const picker = new BvhSceneRaycaster(scene);          // a tree per geometry and range, and the top level
scene.updateMatrixWorld(); picker.update();           // when objects moved: the top level only
const hits = raycastScene(picker, raycaster);         // what raycaster.intersectObject(scene, true) returns
const seen = picker.intersectsFrustum(frustum, camera.layers); // { object, instanceId, group } per object or instance
```

- **What it returns:** three's `intersectObject(root, true)` with three's own `Mesh.raycast`: the same triangles (ranges and groups as in "Scenes"), `side`, `near`/`far`, `instanceId` and `batchId`, and the same fields, sorted. Hidden objects are hit too, as three's `Raycaster` hits them (`includeHidden: false` skips them); a `BatchedMesh`'s hidden instances are skipped, as three skips them. `raycaster.firstHitOnly` returns the one closest hit over the scene.
- **Differs from three:** layers 8-31 (the instance mask holds layers 0-7; `layers: false` ignores layers), skinned and morphed meshes (rest pose here; three poses them on the CPU), and three's `NaN`-distance hit past the end of an index (not reported). Points, lines and sprites aren't read.
- **`update()`** walks the scene again. With the same meshes it rebuilds only the top level (`CpuSceneBvh.setInstances`); new geometry or ranges get their trees, a geometry whose position attribute or index changed (object or version, so set `needsUpdate`) gets its tree rebuilt, and trees nobody uses are dropped. It returns whether meshes changed.
- **When to call `update()`:** raycasts and frustum queries see the scene as of the last `update()`. Call `scene.updateMatrixWorld()` and `update()` after objects move, show or hide, are added or removed, after instance matrices or counts change, and after geometry edits. A raycast that finds a geometry edited since warns once (and answers from the old tree); moved objects aren't detected.
- **Frustum culling:** `intersectsFrustum(frustum, layers?)` walks the top level with three's `Frustum.intersectsBox` test on each node's box: a subtree outside a plane is skipped, one inside every plane is taken whole. It returns the objects and instances whose world box (the walk's own, the mesh tree's box through the instance's matrix) meets the frustum.
- **Engine-free core:** `CpuSceneBvh` (`src/query/cpu-scene.ts`, from the root): `raycast(ray, { firstHitOnly, mask, accept })`, `intersectsFrustum(planes, { mask })`, `intersectsBox(box, { mask })`, `setInstances(instances)`, over any builder's trees.
- **Not over `SceneTrees`:** its trees live on the GPU (no CPU copy), so the CPU scene is built from the same three objects. CPU picking over a `SceneTrees` scene would need its trees read back or kept on the CPU too; that is a memory lever we don't have yet.
- **Tested:** `src/three/scene-raycast.test.ts` (10,000 rays against three's `intersectObject`, before and after moves, all hits and closest; 300 camera frustums against brute force over each object's world box) and `src/query/cpu-scene.test.ts` (brute force over every triangle of every instance, masks, `setInstances`; frustums and boxes against every world box).

### The tree object (`geometry.boundsTree`)

`computeBoundsTree` makes a `BvhBoundsTree`: the core's `CpuMeshBvh` (`src/query/cpu-mesh-bvh.ts`, exported from `@pmndrs/bvh`) with three's types on top. `acceleratedRaycast` runs through it, so `/three` and the core share one walk and one triangle test. `/three` also exports it as `MeshBVH`, so switching from three-mesh-bvh is one import change (`import { MeshBVH } from '@pmndrs/bvh/three'`); the rows below are where the two differ. In place of three-mesh-bvh's `MeshBVH`:

| three-mesh-bvh `MeshBVH` | `BvhBoundsTree` |
| --- | --- |
| `new MeshBVH(geometry, options)` | `new BvhBoundsTree(geometry, options)` (after `await prepareBoundsTrees()`), or `geometry.computeBoundsTree(options)` |
| `raycast(ray, materialOrSide, near, far)`, `raycastFirst(...)` | The same arguments (a side, a material, or an array of materials by group), three intersections in the geometry's space, without `object` |
| `refit()` | `refit()`: rereads the position attribute, keeps the tree's shape, and the raycast uses the tree again. The index must not have changed. A `geometry.boundingBox` the tree set (`setBoundingBox`) is updated too. Every input is checked before anything is written, so a refit that throws leaves the tree as it was |
| `resolveTriangleIndex(i)` | The same: a tree position (from `traverse`) to the triangle's index. The geometry is never reordered, as with `indirect: true` |
| `getBoundingBox(box3)` | The same |
| `estimateMemoryInBytes(bvh)`, `getBVHExtremes(bvh)`, `validateBounds(bvh)` | The same functions, from `/three` (and `@pmndrs/bvh`; they take any `CpuMeshBvh`). `estimateMemoryInBytes` is the tree's method (the layout, its top level and lookups; three-mesh-bvh walks its object graph). `getBVHExtremes` gives three-mesh-bvh's `ExtremeInfo` for our one root; the split axis isn't stored in our layout, so `splits` counts the axis the children's centres lie furthest apart on. `validateBounds` reads the geometry, so edits without `refit()` fail, with three-mesh-bvh's `console.assert` messages |
| `new MeshBVHHelper(mesh, depth)` / `BVHHelper` | The same classes from `/three` (`MeshBVHHelper` without the deprecation warning): a `Group` drawing the tree's boxes down to `depth` as line segments (`displayEdges = false`: see-through faces), parents too with `displayParents`; `color`, `opacity`, `edgeMaterial`, `meshMaterial`, `instanceId`; constructed from a mesh (its `geometry.boundsTree`), a tree, or both, with a depth (default 10). Call `update()` after changing the tree or those fields (drei's `useHelper` calls it each frame). It follows the mesh's world matrix, and an instance's for an `InstancedMesh`. The boxes come from the core (`bvhHelperPositions` / `bvhHelperIndices` in `@pmndrs/bvh`), so another engine can draw the same thing |
| `traverse(callback)` (the node walk: `depth`, `isLeaf`, `boundingData`, `offset` / `splitAxis`, `count`) | `traverseNodes(callback)`: the same arguments and order, root at depth 0, a truthy return skips the children. Our `traverse` is the shapecast walk below, not this |
| `MeshBVH.serialize(bvh)`, `MeshBVH.deserialize(data, geometry, { setIndex })` | The same statics on `BvhBoundsTree` (PARITY 7.1), in three-mesh-bvh's `SerializedBVH` both ways (`src/query/three-mesh-bvh-format.ts`): writing gives indirect mode (one root of 32-byte nodes from `traverseNodes`, `indirectBuffer` = each tree position's original triangle, 16-bit up to 2^16 triangles as three-mesh-bvh decides, `index` a copy unless `cloneBuffers: false`); reading takes indirect trees, reordered-index trees (`setIndex` copies `data.index` into the geometry as three-mesh-bvh does), several roots (joined pairwise under new inner nodes) and version 0 right children, keeps the shape through `buildMeshBvhFromTopology` (exact boxes, leaves past 8 cut), and refuses a tree that leaves triangles out. `deserialize` also takes our bytes. `boundsTree.serialize()` and `deserializeBoundsTree(bytes, geometry)` stay our layout (`serializeLayout`). `deserialize` is overloaded so the core's static (`CpuMeshBvh.deserialize(bytes, options)`) still type-checks for subclasses |
| `getJSONStructure(bvh)` | The same (PARITY 7.5), a port of three-mesh-bvh's over `traverseNodes`; `Box3` bounds from `/three`, `BvhBox3` from the core (or `makeBox`). On a tree written in three-mesh-bvh's format it equals three-mesh-bvh's own output, node for node (`src/three/serialize-format.test.ts`) |
| `ObjectBVH.shapecast({ intersectsBounds, intersectsObject, intersectsRange, boundsTraverseOrder })` | `BvhSceneRaycaster.shapecast` (PARITY 5.3): the core's `CpuSceneBvh.shapecast` / `traverse` over the top level (`CpuMeshBvh.traverse`'s contract, call for call against three-mesh-bvh's `shapecastTraverse` in `src/query/scene-shapecast.test.ts`; contained ranges are runs of top-level positions, which the SAH builder partitions in place), with `Box3` boxes and `intersectsObject(object, instanceId, contained, depth, source)` (`instanceId` 0 for a plain mesh, as three-mesh-bvh). Boxes are the padded world boxes of each object's mesh tree, not three-mesh-bvh's; sphere casts find the same objects as `ObjectBVH.shapecast` (`src/three/scene-shapecast.test.ts`) |
| `getTriangleHitPointInfo(point, geometry, triangleIndex, target)` | The same function, from `/three` |
| `shapecast({ intersectsBounds, intersectsTriangle, intersectsRange, boundsTraverseOrder })` | The same contract: boxes as a `Box3`, triangles as `ExtendedTriangle` (a scratch three `Triangle` with `needsUpdate` set) and the **original** triangle index, `contained` and `depth`; a truthy return stops the cast; `intersectsRange` (tree positions) decides first when both are given; with neither, the cast stops at the first `CONTAINED` box. Casts may nest. It is the core's `traverse` underneath (one box walk) |
| `intersectsBox(box, boxToMesh)`, `intersectsSphere(sphere)` | The same (an `OrientedBox` / sphere shapecast). `boxToMesh` may be left out (identity) |
| `intersectsGeometry(geometry, geometryToBvh)` | The same answer. With one of our trees on `geometry.boundsTree` it is a `bvhcast` testing each pair as three-mesh-bvh does (three-mesh-bvh shapecasts the other tree from each leaf instead; the `bvhcast` measured 1.5x faster here); otherwise every triangle of the geometry is tested against the leaves its box reaches. A `BvhBoundsTree` / `CpuMeshBvh` or `{ positions, indices }` may be passed in place of the geometry |
| `ExtendedTriangle`, `OrientedBox` (and `SeparatingAxisBounds`, `CONTAINED`, `INTERSECTED`, `NOT_INTERSECTED`) | The same classes from `/three`: `ExtendedTriangle` extends three's `Triangle` (`intersectsTriangle(other, target, suppressLog)`, `intersectsSphere`, `closestPointToSegment`, `distanceToPoint`, `distanceToTriangle`, `update()`, the cached `satAxes` / `satBounds` / `plane` / degeneracy); `OrientedBox` has `min`, `max`, `matrix` (`Vector3`, `Matrix4`) as three-mesh-bvh's, not a `Box3` subclass. No math lives in `/three`: both call the core's functions (`src/query/shapes.ts`), ported operation for operation, so they give three-mesh-bvh's results bit for bit (`src/three/shapes.test.ts`). `set` and `copy` set `needsUpdate` (three-mesh-bvh's `copy` does not) |
| `bvhcast(otherBvh, matrixToLocal, { intersectsRanges, intersectsTriangles })` | The same walk, step for step (three-mesh-bvh's `bvhcast.js` on our nodes: descend the other tree while both children overlap, switch trees when one doesn't or a leaf is reached, three's `Box3.applyMatrix4` / `intersectsBox` box tests, its depth bookkeeping). `intersectsTriangles(triangle1, triangle2, i1, i2, depth1, nodeIndex1, depth2, nodeIndex2)` gets `ExtendedTriangle`s, the other's moved into this space, and **tree positions** (as three-mesh-bvh's `indirect` mode, so three-bvh-csg's `resolveTriangleIndex(ia)` works unchanged); `intersectsRanges` runs first when both are given. Unlike three-mesh-bvh, casts may nest, and a cast with neither callback throws. `nodeIndex` names a leaf by its child slot |
| `closestPointToPoint(point, target, minThreshold, maxThreshold)` | The same: three-mesh-bvh's shapecast step for step (nearer box first, boxes culled by the best so far and `maxThreshold`, an early stop below `minThreshold`), filling `target` with `point` (a `Vector3`, written in place when there is one), `distance` and `faceIndex` (the original triangle), or null with `target` untouched. As in three-mesh-bvh, `maxThreshold` culls boxes, not triangles. Against three-mesh-bvh on 800 random points: the same distance every time, the same triangle or another equally near (`src/three/closest-point.test.ts`) |
| `closestPointToGeometry(geometry, geometryToBvh, target1, target2, minThreshold, maxThreshold)` | The same casts (this tree ordered by the distance from the geometry's oriented box; per leaf, the geometry's own tree when it has one of ours, else every triangle of it). `target1` is on this tree; `target2` (null skips it) on the geometry, in its space. Both `faceIndex`es are original triangles (three-mesh-bvh 0.9.18's indirect mode gives tree positions there). A tree or `{ positions, indices }` may be passed in place of the geometry. As in three-mesh-bvh, the box scores are taken in each tree's own space, so a scaling `geometryToBvh` skews the culling |
| (none) | `traverse({ intersectsBounds, intersectsRange, boundsTraverseOrder })` is the raw box walk under `shapecast` and the closest-point queries (a `Float32Array(6)` box, tree-position ranges) |
| `new GenerateMeshBVHWorker().generate(geometry, options)` | `new GenerateMeshBvhWorker().generate(geometry, options)`: a `BvhBoundsTree` built on `@pmndrs/bvh/worker`'s pool (inline in Node). The geometry's arrays are copied, not moved, so it stays usable, and several builds may run at once. Edits made while it builds leave the tree stale (raycasts fall back to three's until `refit()`). `generator.dispose()` stops the pool and rejects its unfinished builds |

Without three, the same queries take plain shapes: `shapecast` hands over the core's `ExtendedTriangle` and boxes with `min` / `max` points; `intersectsBox(box, matrix)` takes `{ min, max }` and sixteen column-major numbers (or anything with `elements`), `intersectsSphere` `{ center, radius }`; `ExtendedTriangle`, `OrientedBox` and the math (`triangleIntersectsTriangle`, `closestPointsSegmentToSegment`, ...) are exported from `@pmndrs/bvh`. Anything with `x`, `y`, `z` passes as a point, so three's types work there too.

Without three: `CpuMeshBvh.create({ positions, indices })` (or `CpuMeshBvh.build(input, tinyBvhBuilder)`, or `new CpuMeshBvh(layout)` over any tree in our layout), then `raycast(ray, { side, near, far })` with `BVH_FRONT_SIDE` / `BVH_BACK_SIDE` / `BVH_DOUBLE_SIDE` (three's values). Side culling is native to the CPU walk (`createCpuTracer`'s per-trace `side`, CPU only: the WGSL walk has no side option, and a double-sided CPU raycast is the GPU walk exactly). A layout from a file is checked when the tree is made (children in range, each node reached once, leaves and original indices within the triangles), so a corrupt one throws instead of hanging a walk. `generateCpuMeshBvh(pool, input, options)` from `@pmndrs/bvh/worker` builds one off the main thread.

### Build options

`computeBoundsTree`, `new BvhBoundsTree` and `GenerateMeshBvhWorker.generate` take three-mesh-bvh's options:

| Option | What happens |
| --- | --- |
| `strategy` (`CENTER` 0, `AVERAGE` 1, `SAH` 2) | Accepted. Every value builds the same tree: tinybvh's default builder (binned SAH over a grid of cells, "SIMD Bonsai"), whose trees have a lower SAH cost than a centre or average split. There is no CENTER or AVERAGE builder. Other values throw |
| `targetLeafSize`, `maxLeafSize`, `maxLeafTris` | The largest leaf, in that order of precedence, clamped to our layout's 1 to 8 triangles: three-mesh-bvh's default 10 gives 8. tinybvh's SAH may make smaller leaves; larger ones are cut in half |
| `maxDepth` | Accepted, not honoured: tinybvh decides the depth (`boundsTree.depth`). A value below the built tree's depth logs one warning. The ray walk's stack holds 64 entries and throws a `RangeError` past it, which takes a degenerate mesh |
| `setBoundingBox` | Default `true`, as three-mesh-bvh: sets `geometry.boundingBox` from the tree when it has none, and `boundsTree.refit()` keeps that box current. A box three computed or you set is left alone. `false` leaves `geometry.boundingBox` unset |
| `indirect` | Accepted and ignored: the index is never reordered |
| `onProgress` | Called with 0 before the build and 1 after it. tinybvh builds in one WebAssembly call, with nothing to report in between |
| `verbose`, `useSharedArrayBuffer` | Accepted and ignored |
| `range` | Throws: the tree holds every triangle, and raycasts follow the draw range |
| `builder`, `layout`, `snapshot` (ours) | A tinybvh builder to use; a tree already built for this geometry, to skip the build; the geometry's versions when that tree's build started (`snapshotGeometry`), so edits during the build count |

What the surveyed callers pass, all accepted: homefig `{ maxLeafSize: 10 }`; drei `{ strategy: SAH, verbose: false, setBoundingBox: true, maxDepth: 40, maxLeafTris: 10, indirect: false }`; three-gpu-pathtracer `{ strategy: SAH, maxLeafTris: 1, indirect: true, onProgress }`; upscaler `{ strategy: SAH, targetLeafSize: 4 }`; three-bvh-csg `{ maxLeafSize: 3, indirect: true, useSharedArrayBuffer }` (`src/three/bounds-tree.test.ts`).

**The two callers' patterns** (`node --experimental-transform-types bench/query-parity.ts`, Node 22 on the Apple M5, the furnished room's 269,764 triangles, 2,000 capsules and 100 brush placements, means of 3 rounds):

| Pattern | Agreement with three-mesh-bvh 0.9.18 | Ours | three-mesh-bvh |
| --- | --- | --- | --- |
| homefig's controller (`collisionCheck` + `floatingCheck`: two capsule shapecasts) | 2,000 / 2,000 capsules: the same 4.9M contact triangles, the same float distance and tied hits | 1.71 ms per capsule | 1.84 ms |
| three-bvh-csg's `collectIntersectingTriangles` (`bvhcast`, room against a 3,380-triangle sphere brush) | 100 / 100 placements: the same 21,160 pairs and segments | 1.51 ms per placement | 1.61 ms |

Two things in homefig's own code depend on the tree, with either library: the float sensor keeps the first of several tied hits its walk reaches (176 of 2,000 capsules pick a different one of the same tied points), and its vertical test takes `|dot|`, so it accepts hits above the sensor, outside the box it culls with, which a tree reaches only when they share a leaf with a triangle inside (18 of 2,000 differ for that). With the cull box applied to each triangle too, all 2,000 agree.

**Closest point** (`node --experimental-transform-types bench/closest-point.ts --rounds 10`, same machine and room, 2,000 points per row, 200 placements; rerun 2026-11-13 after the CPU query pass made `closestPointToPoint` a flat walk, under load):

| Query | Agreement with three-mesh-bvh 0.9.18 | Ours | three-mesh-bvh |
| --- | --- | --- | --- |
| `closestPointToPoint`, random points in the room's box | 2,000 / 2,000 the same distance; 1,759 the same triangle, 241 another equally near | 1.6 us | 4.9 us |
| `closestPointToPoint`, points just off random surfaces | 2,000 / 2,000; 1,755 the same triangle, 245 tied | 2.4 us | 6.6 us |
| `closestPointToGeometry`, a 180-triangle icosphere with its own tree, rigidly placed | 200 / 200 the same distance | 28.4 ms | 27.7 ms |

Raycasts per second, Node 22 on the Apple M5 while other jobs were running (`node --experimental-transform-types bench/three-raycast.ts --seconds 2`, three runs, median). The mesh is a torus knot, the rays come from outside, and 81% of them hit:

| 100k triangles | Build ms | `firstHitOnly` | All hits |
| --- | --- | --- | --- |
| three's `Mesh.raycast` | none | 294 | 288 |
| three-mesh-bvh 0.9.18 (CENTER, its default) | 31 | 664,652 | 238,685 |
| three-mesh-bvh 0.9.18 (SAH) | 95 | 647,330 | 286,503 |
| `@pmndrs/bvh/three` (tinybvh) | 28 | 692,630 | 315,420 |

At 10k triangles: three 3,390; three-mesh-bvh 607k (CENTER) and 656k (SAH); ours 629k (`firstHitOnly`). All hits: 285k, 281k and 363k.

## Switching the known consumers

- **reflections:** use `bvhSceneFromObject3D` in place of `bvhTraceSceneFromObjects`. It also covers BatchedMesh, groups and draw ranges. `SceneTreesObject3DSync.update()` replaces `refit()`, which rebuilt the CPU top level each time.
- **Denoiser (`three-temporal`):** use `BVHComputeData(room)` → `SceneTrees` + sync + `BvhTraceNodes`, and `${bvh.fns.raycastFirstHit}(ray, &hit)` → `bvhTraceClosest(BvhRay(o, 0.0, d, tMax))` in a `wgslFn` with `[bvh.module]`. The per-vertex `mat` attribute becomes `material(object)` (`hit.material`) and a material table.
- **upscaler (radiance cascades):** build `SceneTrees`/`buildTraceScene` from the scene, not one world-space `MeshBVH`. The hit's `triangle` and `barycentrics` replace `hit.indices.x / 3` and `barycoord.yz`.
- **ddgi-research (`gpu-ray-provider`):** drop the `IdentifiedBVH` subclass. `objectID`/`instanceID` come from `sync.sourceOf(hit.instance)` on the CPU, or as `material` on the GPU, and the mirror sign is already in `hit.normal`/`frontFace`.

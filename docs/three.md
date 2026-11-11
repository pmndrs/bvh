# `@pmndrs/bvh/three`

The three.js side of the package. It's the only part that imports three, which is an optional peer dependency pinned to `>=0.187.0 <0.188.0`: `BvhTraceNodes` relies on three's WebGPU backend internals, so each three release is checked before the range widens. The core stays engine-free, and lint enforces that.

## Coming from three-mesh-bvh's WebGPU path

| three-mesh-bvh 0.9 (`three-mesh-bvh/webgpu`) | `@pmndrs/bvh/three` |
| --- | --- |
| `new BVHComputeData(root, { attributes })` | `bvhSceneFromObject3D(root, options)` gives the meshes and instances. `SceneTrees` builds the trees (GPU H-PLOC first, tinybvh swapped in) on `deviceFromRenderer(renderer)`. `new SceneTreesObject3DSync(trees, root, options)` keeps it in step. `new BvhTraceNodes(renderer, trees)` exposes the trees in three's node system. |
| `bvh.update()` (repack everything) | `sync.sync()` when objects are added or removed: only the meshes and instances that changed are added or removed. |
| `bvh.updateTransforms()` | `sync.update()` each frame (only moved, non-static instances), then `trees.update(encoder)`, then `bvh.update()`, which rebinds the storage nodes when `SceneTrees.version` changed. |
| `bvh.storage.nodes / transforms / index / attributes` | `bvh.storage.nodes / triangles / topLevel`, which hold our layout (docs/design/trace.md). There's no per-vertex attribute buffer: use `hit.instance` (via `sync.sourceOf`), `hit.triangle` (the triangle in its mesh, via `threeFaceIndex`) and `hit.barycentrics` against your own attributes, or `hit.material`. |
| `bvh.fns.raycastFirstHit(ray, &hit) -> bool` | WGSL: `bvhTraceClosest(ray: BvhRay) -> BvhHit` (`hit.t < 0` on a miss). TSL: `bvh.traceClosest(origin, direction, tMin, tMax)`, a `BvhTraceHit` struct node. |
| `bvh.fns.raycastAnyHit(ray) -> bool` | WGSL: `bvhTraceAny(ray) -> bool`. TSL: `bvh.traceAny(origin, direction, tMin, tMax)`. |
| (none) | Ray masks: `bvhTraceClosestMasked(ray, mask)` / `bvhTraceAnyMasked`, or TSL `traceClosestMasked` / `traceAnyMasked`. Use `rayMaskFromLayers(camera.layers)`. |
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
  - Reads `Mesh`, `InstancedMesh` (instances below its `count`), `BatchedMesh` (active instances; `getVisibleAt` too) and `SkinnedMesh` (rest pose; skinning and morph targets aren't applied).
  - Reads the triangles three's `Mesh.raycast` tests: the draw range, and with an array of materials, the groups within it. Each group is its own mesh, so it can carry its own material.
  - Ranges are read as three steps through them: three indices at a time from the range's start while the index is below its end. A range that starts off a multiple of 3 gives triangles made of neighbouring triangles' corners, and one whose length isn't a multiple of 3 ends in a triangle that reads past it; both are in the mesh, as in three. Overlapping groups are both read. A triangle that would read past the index (or the positions) is dropped: three reports it at a `NaN` distance with `NaN` corners, which no ray can hit (`threeTriangleRanges`).
  - Takes indexed, non-indexed and interleaved positions. Interleaved (or normalized, or non-float) positions are compacted once per geometry and shared by all its groups' meshes.
  - Makes one mesh per geometry and range, shared by every object and instance that uses it.
  - Call `updateMatrixWorld()` first: the world matrices are read as they are.
- **Visibility:** an object counts as visible when it and every ancestor are. Hidden objects stay in as hidden instances.
- **Layers:** layers 0-7 become instance mask bits 0-7, and layers 8-31 are dropped. An object on none of layers 0-7 gets mask 0, which no ray hits, not even the plain `bvhTraceClosest`/`bvhTraceAny` (all ones). A default object (layer 0) gets mask `0x01`. `layers: false` gives every instance `0xff`.
- **Static:** `static: true`, or `(object, instanceId) => boolean`, marks instances for `mergeStaticInstances`. The sync never pushes changes to static instances.
- **Merged static content can't change.** After `SceneTrees.update` merged an object's static instances, `sync.sync()` skips removing that object or changing its geometry, draw range, groups or materials, and warns once per object (`skipped` in the counts). The merged triangles stay, and `sync.sourceOf` still maps their hits. Mark objects that may go away `static: false`. A new object on a geometry whose mesh was merged away gets the mesh added again.
- **Materials:** the `material` callback is read at `sync.sync()` only. After changing what it returns, call `sync.sync()`.
- **`sync.update()` cost:** it skips an object whose `matrixWorld`, effective visibility, layers and `instanceMatrix.version` (a `BatchedMesh`'s matrices texture version) are unchanged, without allocating. Set `instanceMatrix.needsUpdate = true` after `setMatrixAt`, as three needs anyway. A `BatchedMesh`'s per-instance visibility is still compared each frame, since three gives it no version.
- **Failures:** `sync.sync()` walks and plans before it changes anything, so a throwing `material` callback leaves everything as it was. Each `SceneTrees` call updates the sync's records along with it, so a later `sync()` picks up where a failed one stopped.
- **Without SceneTrees:** `buildTraceScene(scene.meshes, scene.instances, { build, mergeStaticInstances })` builds it all at once. `uploadTraceScene` then gives buffers that `BvhTraceNodes` takes as `{ buffers }`.

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
- **Indexed leaves:** a `SceneTrees` made with `triangleLeaves: 'indexed'` (docs/design/indexed-leaves.md) has a fourth buffer, its positions. `BvhTraceNodes` reads the format from its source (`triangleLeaves`, `positionLayout`), binds the fourth storage node (`bvh.storage.positions`) and compiles the walk for it; nothing else changes for the caller.
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

- **What three would return:** the same intersections as three's `Mesh.raycast`:
  - the draw range, and with an array of materials, the groups;
  - `side` (front, back or double);
  - `near` and `far`.
- **The same fields:** `distance`, `point`, `face` (`a`, `b`, `c`, `normal`, `materialIndex`), `faceIndex`, `barycoord`, `uv`, `uv1` and `normal`. `InstancedMesh` and `BatchedMesh` get these through `Mesh.raycast`.
- **BatchedMesh:** call `batchedMesh.geometry.computeBoundsTree()`. three raycasts each instance through a stand-in mesh that shares the batched geometry's attributes, and the tree is found by its position attribute. Each geometry's range is cached once.
- **Materials:** the triangle ranges are cached per material and draw range, and checked against the groups and the materials an array holds. Swapping `mesh.material[i]` is picked up, and each material's `side` is read at every raycast.
- **All hits:** without `firstHitOnly`, every hit is returned, as three does. The triangle filter collects them.
- **Falls back to three's raycast** when the geometry has no tree, for a `SkinnedMesh`, or when there are morph targets.
- **Queries read the tree's triangles.** `shapecast`, `bvhcast` and `intersects*` hand over the corners stored in the tree, not the geometry's current positions. After editing positions, call `boundsTree.refit()` first (three-mesh-bvh reads the geometry, so it sees edits straight away). `intersectsGeometry` against a bare geometry recomputes that geometry's bounds rather than trusting `geometry.boundingBox`.
- **Edited geometry:** if the position attribute's version, or the index (object or version), changed since `computeBoundsTree`, the raycast falls back to three's own and warns once. It stays that way until `geometry.boundsTree.refit()` (same index) or `computeBoundsTree` runs again.
- **Ranges off a multiple of 3 and overlapping groups** are matched. A range that starts at index 1 or 2 mod 3 is walked in a second tree over the triangles that start there, built with the tree's options on the first raycast that needs it (it needs tinybvh, loaded by `prepareBoundsTrees`; without it the raycast falls back to three's) and refitted with the tree. Overlapping groups are walked once each, so a triangle in two groups is reported twice, as three does. The one difference: three reports the triangle that reads past the end of the index at a `NaN` distance (and its sort is then out of order); we report nothing there.
- **Tested:** `src/three/raycast.test.ts` checks it against three on 8,000 random rays, and 5,000 more through ranges and groups off a multiple of 3.

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
| `MeshBVH.serialize(bvh)`, `MeshBVH.deserialize(data, geometry)` | `boundsTree.serialize()` and `deserializeBoundsTree(bytes, geometry)`. The bytes are our layout (`serializeLayout`), not three-mesh-bvh's format |
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

**Closest point** (`node --experimental-transform-types bench/closest-point.ts --rounds 10`, same machine and room, 2,000 points per row, 200 placements):

| Query | Agreement with three-mesh-bvh 0.9.18 | Ours | three-mesh-bvh |
| --- | --- | --- | --- |
| `closestPointToPoint`, random points in the room's box | 2,000 / 2,000 the same distance; 1,759 the same triangle, 241 another equally near | 5.1 us | 5.5 us |
| `closestPointToPoint`, points just off random surfaces | 2,000 / 2,000; 1,755 the same triangle, 245 tied | 8.1-8.6 us | 7.5 us |
| `closestPointToGeometry`, a 180-triangle icosphere with its own tree, rigidly placed | 200 / 200 the same distance | 30.8 ms | 28.7 ms |

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

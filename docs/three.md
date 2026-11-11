# `@pmndrs/bvh/three`

The three.js side of the package. It's the only part that imports three, which is an optional peer dependency (written against 0.187). The core stays engine-free, and lint enforces that.

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
| `MeshBVH` per geometry (`geometry.boundsTree`) | Trees are per geometry and triangle range, built by SceneTrees (GPU, then a tinybvh worker pool), or by `buildTraceScene` with any builder. |

## Scenes (`bvhSceneFromObject3D`)

- **Objects and instances:**
  - Reads `Mesh`, `InstancedMesh` (instances below its `count`), `BatchedMesh` (active instances; `getVisibleAt` too) and `SkinnedMesh` (rest pose; skinning and morph targets aren't applied).
  - Reads the triangles three's `Mesh.raycast` tests: the draw range, and with an array of materials, the groups within it. Each group is its own mesh, so it can carry its own material.
  - Takes indexed, non-indexed and interleaved positions.
  - Makes one mesh per geometry and range, shared by every object and instance that uses it.
  - Call `updateMatrixWorld()` first: the world matrices are read as they are.
- **Visibility:** an object counts as visible when it and every ancestor are. Hidden objects stay in as hidden instances.
- **Layers:** layers 0-7 become instance mask bits 0-7, and layers 8-31 are dropped. An object on none of layers 0-7 gets mask 0, which no ray hits, not even the plain `bvhTraceClosest`/`bvhTraceAny` (all ones). A default object (layer 0) gets mask `0x01`. `layers: false` gives every instance `0xff`.
- **Static:** `static: true`, or `(object, instanceId) => boolean`, marks instances for `mergeStaticInstances`. The sync never pushes changes to static instances.
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

- **Buffers:** the storage nodes are bound to SceneTrees' own `GPUBuffer`s, with no copy. `BvhTraceNodes` sets three's backend buffer for each node before three first uses it. That's three's internals, not its API.
- **When buffers are replaced:** `bvh.update()` swaps in a new attribute bound to the new buffer, and three makes new bind groups. Nothing recompiles.
- **Shader size:** the walk goes into each shader once, as plain WGSL (one loop with a 64-entry stack), and each TSL helper is one small function. The names are fixed, so the shader text is the same on every load.

## CPU raycast

These are drop-ins for three-mesh-bvh's functions of the same names. They run the CPU walk (`createCpuTracer`) over one tinybvh tree per geometry:

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
- **All hits:** without `firstHitOnly`, every hit is returned, as three does. The triangle filter collects them.
- **Falls back to three's raycast** when the geometry has no tree, for a `SkinnedMesh`, or when there are morph targets.
- **Not matched:** a range that starts off a multiple of 3. Moved positions need `computeBoundsTree` again.
- **Tested:** `src/three/raycast.test.ts` checks it against three on 8,000 random rays.

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

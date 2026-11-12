# Comparison suite: results (BX.10)

How @pmndrs/bvh compares with the other JavaScript BVH and picking libraries, measured by `compare/` (the plan's "Comparison suite", `docs/internals/design/bvh-package-split.md`). **Every number here is from one Apple M5 (Node 22 and headless Chrome), and is indicative**, not a promise for other machines.

```
BVH_ASSETS_DIR=... BVH_MANIFESTS_DIR=... pnpm compare:node            # CPU, headless: bench/results/compare.csv (+ compare.md)
pnpm compare:node --scenes torus-knot --quick --no-memory              # a minute
pnpm examples:dev   # then /compare/raycast/, /compare/build/, ... (?scene=furnished-room or sponza)
BVH_PAGE_QUERY='scene=sponza&full' pnpm examples:smoke --filter compare-gpu --json out.json   # a page's full run; out.json carries its CSV rows
```

## The libraries

| Library | Version | Licence | What is compared |
| --- | --- | --- | --- |
| @pmndrs/bvh (ours) | this repository | MIT | `CpuMeshBvh` over tinybvh (CPU queries); `TraceBatchKernel` (GPU); `CpuSceneBvh` and `SceneTrees` (scene level); the H-PLOC and LBVH builders' CPU twins (tree quality) |
| three-mesh-bvh | 0.9.18 | MIT | `MeshBVH` with CENTER (its default) and SAH; `ObjectBVH`; `three-mesh-bvh/webgpu` (`BVHComputeData`); `GenerateMeshBVHWorker` |
| three | 0.187.0 | MIT | `Mesh.raycast`: the no-BVH baseline |
| bvh-tree (benraziel) | 1.0.1 | MIT | A plain-JS triangle BVH (centre split, 10 per leaf), `intersectRay` |
| Babylon.js (`@babylonjs/core`) | 9.35.0 | Apache-2.0 | An engine's picking: `Mesh.subdivide` + `AbstractMesh.createOrUpdateSubmeshesOctree` + `mesh.intersects(ray)`, and the same without the octree (brute force), on a `NullEngine` |
| bvh.js (agargaro) | 0.0.13 | MIT | A scene-level dynamic box BVH: build, `move`, frustum and box culling over objects (not triangles) |
| Rapier (`@dimforge/rapier3d-compat`) | 0.21.0 | Apache-2.0 | A physics engine's queries on a trimesh collider: `castRayAndGetNormal`, `projectPointAndGetFeature`, `intersectionsWithShape` (WASM) |

All are dev dependencies used by `compare/` and the compare pages only; nothing of them ships in the package. Their licence files were checked in the installed packages; rows in `THIRD_PARTY.md`.

## Method

- **One adapter interface** (`compare/adapters/types.ts`): `build`, `raycastFirst`, `raycastAll`, `intersectsBox`, `intersectsSphere`, `closestPoint`, `refit`, `memory`, `quality`, and a `features` table. A capability a library doesn't have is absent and says so (the tables show `n/a` with the reason); nothing is emulated behind its back. three-mesh-bvh's `MeshBVH` reorders the index, so its hits are mapped back to the original triangles by a lookup made after the build (not timed).
- **The same geometry and the same queries.** Each scene is merged into one world-space mesh (positions and a 32-bit index) that every library is given. Rays, boxes, spheres and points come from fixed seeds (`compare/core/rays.ts`): `outside` rays start on a sphere around the mesh and aim at a random point of its box; `inside` rays start inside the box (shrunk 10%) in a random direction (the rooms' case). Every query is double-sided, from 0 to infinity.
- **Agreement.** Every answer of every library is checked against ours on the same query (`compare/core/agree.ts`): *same* (triangle and distance within 1e-4 relative), *tie* (another triangle at the same distance, a repeat of a hit on a shared edge, a culled object whose box touches the query's boundary), or *disagree*. **A library with any disagreement on a measurement gets no ratio against ours on it**: a faster wrong answer is not faster. Rapier reports no triangle id for a trimesh hit (its feature id is 0 for every ray hit, and a vertex, edge or face of the triangle for a projection), so it agrees by distance and its answers count as ties.
- **Timing.** Warm runs: a first pass over the queries fixes how many each library runs (all, or what fits in 1.5 s, so brute force can run on Sponza), and is repeated until 250 ms of warm-up. Then 5 timed runs, **interleaved between the libraries with the order reversed every other round** (the surface-ray bench's paired method, because the M5 drifts with heat), and the median. Builds: the median of 3 (the cold first build included). Wall clock (`performance.now()`), one thread.
- **GPU** (`compare/gpu`): ours (`TraceBatchKernel` in its leanest form: no ray masks, the 24-byte hit record) and three-mesh-bvh/webgpu (`MeshBVH` SAH, `BVHComputeData`, a TSL kernel calling `bvh_RaycastFirstHit` / `bvh_RaycastAnyHit`, as `tests/gpu/pages/trace-bench/tmbvh-bench.ts`) on the renderer's device, the same packed rays. GPU timestamps of the one pass (three's `trackTimestamp` for theirs). Paired rounds: both timed back to back in each round (median of 7 dispatches after warm-ups), order reversed every other round, ratio within each round, median of the rounds.
- **Memory** is what a library holds beyond the app's geometry (positions and index the app keeps to draw), as `docs/internals/design/memory.md` counts it. Two numbers: the library's own count where it has one (ours: `estimateMemoryInBytes`; three-mesh-bvh: its node buffers; bvh-tree: its arrays, without its node objects; Rapier: its WASM heap's growth), and the **heap growth of one build in a fresh Node process** (`compare/node/memory.ts`: `heapUsed + external` after gc, before and after, with the structure alive; three-mesh-bvh's index copy, which replaces the app's index in an app, taken off).
- **CSV**: `bench/results/compare.csv`, trace.csv's columns plus `method`: one row per value (`compare/<page>/<metric>/<library>/<unit>`), plus `/queries`, `/checked`, `/disagree` and `/ties` rows.

The machine is an Apple M5 Pro (`os.cpus()` names it so): **Apple M5, indicative** applies to every number below. Node v22.22.2; Chrome headless through `pnpm examples:smoke` for the GPU. Commit 4824f5b plus the uncommitted measuring changes that followed it. Rows: `bench/results/compare.csv`; the runner's full tables: `bench/results/compare.md`.

## Capabilities

`n/a` in the tables below means the library has no such query; the note says why. From the adapters' `features` (`pnpm compare:node --features`):

| Capability | @pmndrs/bvh (tinybvh) | three-mesh-bvh (CENTER, default) | three-mesh-bvh (SAH) | bvh-tree | Babylon.js picking (submesh octree) | Babylon.js picking (brute force) | Rapier (trimesh collider) | three Mesh.raycast (no BVH) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raycastFirst | yes | yes | yes | partial: no first-hit query: every hit is found and the nearest taken | yes: mesh.intersects through the submesh octree | yes: mesh.intersects, every triangle | yes: world.castRayAndGetNormal; no triangle id (its feature id is 0 for every trimesh hit), so it agrees by distance | partial: no first-hit mode: every hit is found and sorted, the first taken |
| raycastAll | yes | yes | yes | yes | no: one hit per mesh (multiPickWithRay); no all-hits pick on a mesh | no: one hit per mesh (multiPickWithRay); no all-hits pick on a mesh | no: intersectionsWithRay gives one hit per collider, not per triangle | yes |
| gpuTrace | yes: the WGSL walk over the same layout (TraceBatchKernel, BvhTraceNodes) | yes: three-mesh-bvh/webgpu: WGSL/TSL inside three's node system (BVHComputeData) | yes: three-mesh-bvh/webgpu: WGSL/TSL inside three's node system (BVHComputeData) | no | no: GPU picking renders ids to a texture (GPUPicker), no ray queries | no: GPU picking renders ids to a texture (GPUPicker), no ray queries | no | no |
| intersectsBox | yes | yes | yes | no | no | no | yes: world.intersectionsWithShape with a Cuboid | no |
| intersectsSphere | yes | yes | yes | no | no | no | yes: world.intersectionsWithShape with a Ball | no |
| closestPoint | yes | yes | yes | no | no | no | yes: world.projectPointAndGetFeature; its feature is a vertex, edge or face of the triangle, not the triangle, so it agrees by distance | no |
| refit | yes: CPU refit; GPU refit of BLAS and the top level in SceneTrees | yes: CPU refit | yes: CPU refit | no | partial: refreshBoundingInfo + createOrUpdateSubmeshesOctree rebuilds the octree | partial: refreshBoundingInfo | no: a trimesh is rebuilt (setShape), not refitted | no |
| workerBuild | yes: tinybvh worker pool (@pmndrs/bvh/worker), no cross-origin isolation | yes: GenerateMeshBVHWorker; ParallelMeshBVHWorker needs SharedArrayBuffer | yes: GenerateMeshBVHWorker; ParallelMeshBVHWorker needs SharedArrayBuffer | no | no | no | no | no |
| gpuBuild | yes: GPU H-PLOC and LBVH builders | no: builds on the CPU only | no: builds on the CPU only | no | no | no | no | no |
| treeStats | yes | yes | yes | partial: no API; read from its node objects here | no | no | no | no |
| serialize | yes | yes | yes | no | no | no | yes: World.takeSnapshot | no |

## compare/raycast: CPU rays per second

Thousands of rays per second, ratio to ours in brackets (above 1: faster than ours), given only where every answer agreed. Torus knot 100,000 triangles; furnished room 269,764; Sponza (with curtains, merged) 5,744,384. 4,096 seeded rays per set; brute force runs as many as fit in 1.5 s (at least 16).

**Closest hit**

| Library | Knot, outside | Knot, inside | Room, outside | Room, inside | Sponza, outside | Sponza, inside |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| @pmndrs/bvh | 413 | 392 | 509 | 407 | 580 | 479 |
| three-mesh-bvh, CENTER | 348 (0.84) | 332 (0.85) | 301 (0.59) | 279 (0.69) | 332 (0.57) | 239 (0.50) |
| three-mesh-bvh, SAH | 374 (0.91) | 365 (0.93) | 407 (0.80) | 374 (0.92) | 437 (0.75) | 301 (0.63) |
| bvh-tree | 142 (0.34) | 208 (0.53) | 206 (0.41) | 266 (0.65) | 69 (0.12) | 107 (0.22) |
| Rapier | 381 (0.92) | **501 (1.28)** | **520 (1.02)** | **471 (1.16)** | 550 (0.95) | 470 (0.98) |
| Babylon, submesh octree | 8.6 (0.02) | 10.3 (0.03) | 5.6, 13 of 4,096 disagree | 7.2, 2 disagree | 1.0, 1 of 1,700 disagree | 1.6, 3 of 2,461 disagree |
| Babylon, brute force | 0.53 | 0.51 | 0.12, 1 of 198 disagree | 0.13 | 0.009 | 0.009 |
| three `Mesh.raycast` | 0.28 | 0.30 | 0.069 | 0.073 | 0.004 | 0.005 |

**Every hit** (Babylon and Rapier have no all-hits query on one mesh)

| Library | Knot, outside | Knot, inside | Room, outside | Room, inside | Sponza, outside | Sponza, inside |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| @pmndrs/bvh | 181 | 223 | 264 | 343 | 158 | 175 |
| three-mesh-bvh, CENTER | 135 (0.75) | 184 (0.83) | 183 (0.69) | 274 (0.80) | 102 (0.65) | 142 (0.81) |
| three-mesh-bvh, SAH | 150 (0.83) | 183 (0.82) | 230 (0.87) | 325 (0.95) | 115 (0.73) | 156 (0.89) |
| bvh-tree | 118 (0.65) | 151 (0.68) | 196 (0.74) | 282 (0.82) | 69 (0.44) | 93 (0.53) |
| three `Mesh.raycast` | 0.26 | 0.24 | 0.067 | 0.071 | 0.005 | 0.004 |

Agreement: three-mesh-bvh, bvh-tree and three agree with ours on every ray of every set (ties only: another triangle at the same distance, or a shared-edge hit counted twice). Rapier agrees on every distance (it gives no triangle). Babylon disagrees on a few rays per scene, each a hit nearer than everyone else's (its ray-triangle test isn't watertight), so those rates get no ratio.

## compare/gpu: GPU rays, ours against three-mesh-bvh/webgpu

1,048,576 seeded rays per dispatch on the M5's GPU, GPU timestamps, 7 paired rounds (ratio within each round). Any hit: inside rays within a quarter of the diagonal, outside rays within two diagonals (occlusion). Ours runs twice: the default kernel, and with `anyHitOrder: 'larger'` (any-hit enters the larger child first; closest hit is the same kernel). Both are checked against each other on every ray (0 differ) and against three-mesh-bvh.

| Scene | Closest, outside | Closest, inside | Any, outside: default / `'larger'` | Any, inside: default / `'larger'` | GPU bytes, ours / theirs |
| --- | --- | --- | --- | --- | --- |
| Knot (100k) | 1.07x (163 / 155 Mrays/s), 2 of their rays disagree | 1.08x (137 / 127) | 1.26x / 1.17x (214, 198 / 171) | 1.25x / 1.20x | 8.2 / 3.8 MB |
| Furnished room (270k) | **0.85-0.91x** (229 / 253) | **0.86-0.88x**, 1 disagrees | **0.32x / 1.73x** (304, 1,547 / 982) | 1.05x / 1.13x | 20.6 / 10.6 MB |
| Sponza (5.74M) | **0.96x**, 7 disagree | 1.12x, 2 disagree | 1.28x / 1.48x (353, 406 / 276) | 1.28x / 1.39x | 470 / 225 MB |

The ratio is ours over theirs. Where three-mesh-bvh disagrees on a ray (another distance; once a hit at t = 0) the ratio is shown but not ranked. GPU bytes are what each walk reads: ours, nodes, 48-byte triangle records and the top level; theirs, nodes, index, its vec4 copy of the positions and transforms. Absolute rates move by up to 30% between runs with the M5's heat and load; the paired ratios moved by under 0.05.

**The room's any hit from outside (0.31x before).** Not a harness artefact: both walks get the same packed rays and `tMax`, both stop at the first triangle hit in the interval (three-mesh-bvh's `bvh_RaycastAnyHit` returns `STOP_TRAVERSAL` from its range callback; its order is fixed, right child first), every ray is blocked in both, and ours is the leanest kernel. The counted walks (`?counts` on the page, `compare/gpu/gpu-counts.ts`: a counted copy of each library's own loop, same rays) found the cause. Mean work was close (ours 19.1 box tests and 3.9 triangles per ray, theirs 11.9 and 3.9), but the slowest ray of each 32 consecutive rays (one SIMD group) needed **78.5 box tests in ours against 24.3 in theirs**, which is the 3.2x. Our default any-hit order goes to the farther child first (chosen for shadow rays that start on a surface, T1.3); a ray from outside the room then descends into the far side's furniture before it meets the shell it already crossed. Near-first helped a little (0.41x; slowest of 32: 66.5). Entering the child with the larger box first (`anyHitOrder: 'larger'`: the likelier blocker, by the SAH's own area argument) cut the slowest of 32 to 23.0 and gave 1.73x. It is a setting, not the default: on the surface-ray bench it is 1.15-1.42x far-first on instanced trees but 0.78x on merged surface rays (`docs/internals/design/trace.md`, "Any-hit child order"). Workgroup size (64 both), the triangle test and the top level were not the cause: closest-hit, which shares them, was within 0.85-1.12x.

## compare/build: build, refit, memory

Median of 3 builds on one thread, ms; ratio to ours.

| Library | Knot build | Room build | Sponza build | Knot refit | Room refit | Sponza refit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| @pmndrs/bvh (tinybvh) | 33 | 84 | 1,499 | 32 | 75 | 1,178 |
| three-mesh-bvh, CENTER | 25 (0.75) | 82 (0.97) | 1,538 (1.03) | 4.9 (0.15) | 13 (0.18) | 170 (0.14) |
| three-mesh-bvh, SAH | 110 (3.3) | 352 (4.2) | 7,011 (4.7) | 4.5 (0.14) | 13 (0.17) | 177 (0.15) |
| bvh-tree | 92 (2.8) | 270 (3.2) | 11,301 (7.5) | n/a | n/a | n/a |
| Babylon, submesh octree | 9.9 (0.30) | 17 (0.20) | 421 (0.28) | 2.2 (octree rebuilt) | 4.6 | 57 |
| Rapier | 22 (0.65) | 70 (0.83) | 1,329 (0.89) | n/a (rebuilt) | n/a | n/a |
| ours, H-PLOC (CPU twin) | 282 | 872 | 10,699 | | | |
| ours, LBVH (CPU twin) | 113 | 335 | 9,170 | | | |

The twins are TypeScript references for the GPU builders, which are much faster: on the furnished room, `compare/build`'s GPU H-PLOC tree was traceable 14.6 ms after the call (10.4 ms of GPU batches, submit to readback), against 84 ms for tinybvh on the CPU. After the refits, every library's closest hits on 512 rays agree with our refitted tree except Babylon's (1 ray on Sponza).

**Main-thread stalls** (furnished room, Chrome, median of 3: wall ms / longest gap between animation frames meanwhile): ours on the main thread 56 / 56, ours in a worker (`GenerateMeshBvhWorker`) 56 / **22**; three-mesh-bvh CENTER on the main thread 50 / 50, in its `GenerateMeshBVHWorker` 50 / **17**. Both workers keep frames flowing; theirs is a little smoother. On Sponza the build page blocks the main thread for longer than the runner's 30 s DevTools timeout (bvh-tree's 11 s builds), so Sponza's build numbers come from Node only.

## compare/tree-quality

SAH with three-mesh-bvh's costs (traversal 1, triangle 1.25), over the root box's area, lower is better; boxes counted as three-mesh-bvh counts nodes (one of our 64-byte nodes holds two boxes).

| Library | Knot SAH | Room SAH | Sponza SAH | Sponza depth | Sponza boxes | Sponza leaves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| @pmndrs/bvh (tinybvh) | 58.5 | 40.2 | 90.2 | 35 | 6,067,561 | 3,033,781 |
| ours, H-PLOC | 59.3 | **39.3** | **88.4** | 46 | 6,040,519 | 3,020,260 |
| ours, LBVH | 64.4 | 48.5 | 112.6 | 38 | 7,133,901 | 3,566,951 |
| three-mesh-bvh, CENTER | 83.0 | 62.2 | 140.6 | 35 | 1,687,521 | 843,761 |
| three-mesh-bvh, SAH | 76.0 | 52.3 | 121.6 | 30 | 1,656,493 | 828,247 |
| bvh-tree | 83.6 | 64.3 | 204.6 | 48 | 1,752,923 | 876,462 |

Our leaves hold about 2 triangles, theirs about 7 (target 10): lower SAH and faster walks, for 3.6 times the boxes. Babylon's octree and Rapier's tree (inside WASM) have no shape to read.

## compare/queries

Thousands of queries per second (2,048 seeded queries each), ratio to ours. Only three-mesh-bvh and Rapier have these queries; every answer agreed (Rapier's by distance).

| Library | Knot: box / sphere / closest point | Room | Sponza |
| --- | --- | --- | --- |
| @pmndrs/bvh | 275 / 561 / 34 | 204 / 950 / 132 | 318 / 923 / 165 |
| three-mesh-bvh, CENTER | 333 (1.21) / 524 (0.93) / 47 (1.41) | 239 (1.17) / 412 (0.43) / 102 (0.77) | 375 (1.18) / 797 (0.86) / 136 (0.82) |
| three-mesh-bvh, SAH | 351 (1.28) / 561 (1.00) / 50 (1.48) | 247 (1.21) / 506 (0.53) / 187 (1.41) | 459 (1.44) / 926 (1.00) / 175 (1.06) |
| Rapier | **632 (2.30)** / 98 (0.18) / **81 (2.43)** | **542 (2.65)** / 243 (0.26) / **322 (2.43)** | **853 (2.68)** / 263 (0.28) / **277 (1.67)** |

## compare/scene: scene-level trees over many objects

Objects sharing one 1k-triangle mesh, seeded placement in a 200-unit cube; 1,024 seeded frusta, boxes and rays; 20 frames in which every object moves (update: the median frame).

| 10,000 objects | Build | Update | Frustum /s | Box /s | Ray /s | Bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| @pmndrs/bvh `CpuSceneBvh` | 37 ms | 33 ms (top level rebuilt) | 8.4k | 177k | 200k | 1.3 MiB |
| bvh.js | **13 ms** | 51 ms (leaves moved) | 12.0k, 1 frustum disagrees | **215k (1.21)** | n/a | not counted |
| three-mesh-bvh `ObjectBVH` | 63 ms | **16 ms** (refit) | 3.3k (0.40) | 62k (0.35) | 69k (0.34) | 0.7 MiB |

| 100,000 objects | Build | Update | Frustum /s | Box /s | Ray /s | Bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| @pmndrs/bvh `CpuSceneBvh` | 391 ms | 396 ms | 960 | 22.7k | 85.5k | 12.3 MiB |
| bvh.js | **124 ms** | 1,134 ms | 854, 14 frusta disagree | 16.7k (0.73) | n/a | not counted |
| three-mesh-bvh `ObjectBVH` | 491 ms | **176 ms** | 262 (0.27) | 5.7k (0.25) | 33.3k (0.39) | 6.5 MiB |

After the motion, three-mesh-bvh agrees on every query (boundary ties only); bvh.js misses or adds an object on 2 (10k) and 8 (100k) frusta, past the boundary tolerance, so its frustum rate is not ranked. Ours is the CPU path here; on the page the same instances also sit in `SceneTrees`, whose top level is refitted on the GPU every frame.

## Where another library wins

- **Rapier**: box queries 2.3 to 2.7 times our rate, closest point 1.7 to 2.4 times, and closest-hit rays on the knot (inside, 1.28x) and the room (1.02x outside, 1.16x inside); within 5% of ours on Sponza. It gives no triangle id, and holds 2.5 times our tree's memory (below).
- **three-mesh-bvh**: CPU refit 6 to 7 times faster (32-byte one-box nodes over the app's positions; ours rewrites 48-byte records and two-box nodes); box queries 1.2 to 1.4 times ours; closest point up to 1.5 times on the knot and the room; CENTER builds as fast as tinybvh; `ObjectBVH` updates twice as fast as our CPU top-level rebuild. On the GPU its `/webgpu` walk beats ours on the furnished room's closest hits (0.85 to 0.91x) and on Sponza's closest from outside (0.96x, though 7 of its rays disagree). Its any hit from outside the room was 3.2 times our default kernel's (982 against 304 Mrays/s); with `anyHitOrder: 'larger'` ours is 1.73 times theirs there (above). Its worker build stalls frames a little less than ours.
- **bvh.js**: builds a scene tree about 3 times faster than ours and culls boxes 1.2 times faster at 10,000 objects; its frustum culling is faster too, but disagrees on a few objects.
- **Babylon**: builds fastest (it builds almost nothing); its picking is the floor it looks like.
- Ours leads closest and all hits against every library on Sponza, against all but Rapier on the room, and sphere queries everywhere (up to 2.3 times three-mesh-bvh's).

## Memory

What each library holds beyond the app's geometry. "Own": the library's exact buffers; "heap": one build's growth in a fresh Node process (V8 heap plus array buffers and WASM memory, after gc).

| | Knot own / heap | Room own / heap | Sponza own / heap | Sponza bytes per triangle (heap) |
| --- | --- | --- | --- | ---: |
| @pmndrs/bvh | 8.0 / 8.1 MiB | 20.1 / 20.2 MiB | **459.7 / 1,740 MiB** (1,285 of it tinybvh's WASM heap) | 84 (tree) + 235 (builder heap) |
| three-mesh-bvh, CENTER | 0.85 / 0.94 MiB | 2.4 / 2.5 MiB | 51.5 / 47.1 MiB | 9 |
| bvh-tree | 12.2 (arrays only) / 19.6 MiB | 32.9 / 54.9 MiB | 701 / 1,152 MiB | 210 |
| Babylon, submesh octree | not counted / 6.6 MiB | 19.0 MiB heap | 363 MiB heap | 66 |
| Rapier | 20.2 / 20.3 MiB (WASM) | 54.4 / 54.5 MiB | 1,145 / 1,141 MiB | 208 |
| three `Mesh.raycast` | 0 / 0.05 MiB | 0 / 0.05 MiB | 0 / 0.04 MiB | 0 |

The gap, named:

- **Ours against three-mesh-bvh: 9 to 10 times** the bytes beyond the geometry (Sponza 460 against 47 to 52 MiB; 84 against 9 bytes per triangle), as `docs/internals/design/memory.md` found per mesh (8 to 9 times). Our tree copies every triangle's corners into a 48-byte record (56% of it), so the GPU walk reads one buffer and never the app's index; our nodes are 64 bytes with two child boxes; our leaves hold about 2 triangles against their 7, so there are 3.6 times the boxes. three-mesh-bvh reuses the app's positions and reorders the app's own index in place. The same trade shows on the GPU: 470 against 225 MB, where their walk reads its own vec4 copy of the positions. Indexed leaves and quantized nodes (`docs/internals/design/indexed-leaves.md`, `compressed-nodes.md`) are the levers that close part of it.
- **New here: tinybvh's WASM heap.** Building Sponza as **one** 5.74M-triangle mesh grows tinybvh's WASM memory by 1,285 MiB, and a WASM memory never shrinks: the process keeps it as long as the builder lives (`createTinyBvhBuilder`; each worker of the pool has its own). Per-mesh builds (how `SceneTrees` builds Sponza's meshes) stay at its 64 MiB start (`memory.md`); the knot and the room didn't grow it at all. An app that builds one huge mesh on the main thread should drop the builder afterwards, or build on the GPU or in a worker that ends.
- bvh-tree and Rapier hold about 210 bytes per triangle each (bvh-tree: a JS array of 9 doubles and two 28-byte box arrays per triangle, plus node objects; Rapier: its copy of the trimesh and parry's tree in WASM). Babylon holds 66 (submeshes, the octree and the `Vector3` position cache its picking makes).

## Honest gaps

What three-mesh-bvh has and we don't (`docs/internals/PARITY.md`), and what the others have:

- **WebGL2** GPU traversal (GLSL, `MeshBVHUniformStruct`): won't (8.12). Our GPU side needs WebGPU.
- **Points and lines** (`PointsBVH`, `LineSegmentsBVH`): missing (3.2, 3.3); our leaves are triangles only.
- **GPU closest point, GPU shapecast, `sampleTrianglePoint`, `ClusteredBVH`, SDF generation**: missing (8.5 to 8.10). Only rays run on our GPU walk.
- `ObjectBVH.shapecast` (5.3) and `getJSONStructure` (7.5): missing. Partial: `SkinnedMeshBVH` (3.4, GPU posing only), `maxDepth` (4.3, accepted, not honoured), per-index `computeBatchedBoundsTree` (5.5), `StaticGeometryGenerator` (6.3, GPU only), three-mesh-bvh's serialization format (7.1), `onProgress` (0 and 1 only). `ParallelMeshBVHWorker`: won't (cross-origin isolation; big meshes go to the GPU builder).
- **Memory**: 9 to 10 times three-mesh-bvh's beyond the geometry, plus tinybvh's WASM heap after a huge single-mesh build (above).
- **CPU refit speed** (6 to 7 times slower than three-mesh-bvh's), and years of use behind their shapecasts.
- Rapier is a physics engine (collisions, character controllers, joints) and answers box and point queries faster; Babylon is a whole engine; bvh.js inserts, deletes and moves single objects and has LOD frustum culling. We have none of that.

What we have and they don't: GPU builds (H-PLOC, LBVH), GPU refits of mesh trees and the top level, one layout that every builder writes and every walk (CPU and WGSL) reads, a WGSL walk outside three's node system (a raw compute pass, another engine, a worker's device), and the oracle (the CPU port of the walk that checks the GPU's answers).

## What the suite doesn't cover yet

- The pages' CPU numbers in Chrome are not tabled here; Node's are (the same suites). The compare/build page can't finish Sponza within the runner's DevTools timeout.
- The plan's `compare/skinned`, `compare/pathtracer`, `compare/sdf` and the `shapecast`/`distancecast`/`characterMovement` ports are not here (BX.18).
- Measured on the M5 only; the plan's RTX 3060 runs belong to BX.15.
- Rapier gives no triangle ids, so its agreement is by distance only.

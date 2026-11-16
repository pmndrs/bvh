# Comparison suite: results (BX.10)

How @pmndrs/bvh compares with the other JavaScript BVH and picking libraries, measured by `compare/` (the plan's "Comparison suite", `docs/internals/design/bvh-package-split.md`). Two machines, each number labelled with its own: **the Windows laptop re-run of 2026-11-16** (next section: the RTX 3060 Laptop for the GPU, an Intel Core i9-11900H with Node 24 for the CPU), on the defaults of 2026-11-15, and **the Apple M5 tables of 2026-11-12 and 2026-11-13** (Node 22 and headless Chrome) below it, kept as they were. All indicative, not a promise for other machines.

```
BVH_ASSETS_DIR=... BVH_MANIFESTS_DIR=... pnpm compare:node            # CPU, headless: bench/results/compare.csv (+ compare.md)
pnpm compare:node --scenes torus-knot --quick --no-memory              # a minute
pnpm examples:dev   # then /compare/raycast/, /compare/build/, ... (?scene=furnished-room or sponza)
BVH_PAGE_QUERY='scene=sponza&full' pnpm examples:smoke --filter compare-gpu --json out.json   # a page's full run; out.json carries its CSV rows
BVH_PAGE_QUERY='scene=sponza&full&nodeFormat=f32' pnpm examples:smoke --filter compare-gpu --gpu nvidia   # ours on f32 nodes (the default before 2026-11-13)
```

## The Windows laptop re-run, 2026-11-16 (the 2026-11-15 defaults)

HANDOFF asked for the rows the new defaults could change to be measured again: GPU trace (quantized nodes are the default since 2026-11-13), memory and refit (`/three`'s `BvhBoundsTree` reads the app's positions for indexed geometry since 2026-11-15, `corners: 'positions'`; the suite now has that as a second "ours" row), and the scene build. The CPU side is one full `pnpm compare:node` (`bench/results/compare-windows.csv` and `.md`, 1,186 rows; the suite's method: median of 5 interleaved runs, builds median of 3) on an **Intel Core i9-11900H, Windows 11, Node 24.13.1**, with nothing else running. It also re-ran the rows the defaults can't change: they are reported here as measured, not folded into the M5 tables. The GPU side is `compare/gpu` on the **RTX 3060 Laptop**, headless Chrome 157 on D3D12, `full` (1,048,576 rays per set, 7 paired rounds), two runs per scene, started cool (48-52 °C) on an idle GPU; ours with the library's defaults (quantized nodes), and once more with `?nodeFormat=f32`.

**GPU rays, ours against three-mesh-bvh/webgpu** (ours' rate over theirs, paired, two runs; Mrays/s ours / theirs):

| Scene | Closest, outside | Closest, inside | Any, outside: default / `'larger'` | Any, inside: default / `'larger'` | GPU MiB, ours / theirs |
| --- | --- | --- | --- | --- | --- |
| Knot (100k) | 1.29-1.30x (140 / 108), 3 of their rays disagree | 1.31-1.33x (144 / 109) | 1.43x / 1.27-1.29x (184, 163 / 129) | 1.37-1.43x / 1.30-1.31x | 6.2 / 3.6 |
| Furnished room (270k) | **0.51-0.52x** (100 / 194) | **0.84-0.86x** (144 / 168) | **0.35-0.36x** / 1.21x (305, 1,044 / 862) | **0.87x** / **0.96x** (358, 390 / 413) | 16.0 / 10.1 |
| Sponza (5.74M) | 1.09-1.10x (160 / 146), 6 disagree | 1.39-1.41x (136 / 98), 3 disagree | 1.18-1.20x / 1.43-1.44x (294, 349 / 244) | 1.28-1.31x / 1.40-1.43x, 1 disagrees | 355.5 / 214.3 |

**The room's closest hit from outside is the quantized nodes.** With `?nodeFormat=f32` ours is 0.96x theirs there (182-186 Mrays/s, two runs), against 0.51-0.52x (100-101) quantized: quantized nodes cost 46% on that set, where the surface-ray benches found them 0.99-1.21x on Sponza and Bistro (WINDOWS-RUNBOOK section 9). The suite gives each scene to the walk as one merged mesh, so the room is one 270k-triangle mesh on one quantization grid; that is the likely cause, not checked. Elsewhere, `f32` (one run each, the room's two): the knot 1.19x closest from outside (quantized 1.29-1.30x), Sponza 1.12x (quantized 1.09-1.10x), the room from inside 0.90-0.91x (quantized 0.84-0.86x). Any hit from outside the room stays a loss with the default far-first order (0.25-0.26x on `f32`) and a win with `'larger'` (1.17-1.21x).

**Checked 2026-11-17: not the grid, the child order on ties** (WINDOWS-RUNBOOK section 14). The room's grid step is 0.12 mm and its decoded SAH only 0.17% above the exact one, the least of the four scenes measured. The root's two children share their outer planes up to the last bits; from outside, a ray enters both through one plane. On `f32` boxes the right child (the whole room) came out a few ulp nearer, quantized boxes round both planes to one grid plane, and the walk takes the left child on a tie (tinybvh's rule), into geometry behind the wall. Quantized: 28.8 box tests, 10.4 triangles and 124 box tests on the slowest of 32 rays per ray, `f32` 24.5, 6.9 and 90, three-mesh-bvh 19.9, 13.1 and 69 (`?counts`, which counted ours wrongly on quantized scenes until this check). Sending ties beyond `tMin` to the right child (an experiment, commit `7d6fda5`, not the default) made ours **2.33-2.35x** theirs there quantized (445-449 Mrays/s) and 2.45-2.47x on `f32`, Sponza from outside 1.15-1.19x, and left every other set level; Dennis decides.

**Memory, refit and build on the CPU** (i9-11900H; in brackets their time over ours, below 1 is faster than ours; memory beyond the app's geometry):

| Library | Knot: build / refit / memory | Room | Sponza |
| --- | --- | --- | --- |
| @pmndrs/bvh (records, the `CpuMeshBvh` default) | 45.6 ms / 8.32 ms / 8.0 MiB | 133 ms / 23.2 ms / 20.1 MiB | 3,161 ms / 595 ms / 459.7 MiB |
| @pmndrs/bvh, `corners: 'positions'` (what `/three` builds) | 47.2 / 6.13 (0.74) / 4.9 MiB | 122 / 18.4 (0.79) / 11.8 MiB | 3,036 / 441 (0.74) / 284.4 MiB |
| three-mesh-bvh, CENTER | 24.5 (0.54) / 3.84 (0.46) / 0.85 MiB | 79.4 (0.60) / 10.7 (0.46) / 2.4 MiB | 2,045 (0.65) / 239 (0.40) / 51.5 MiB |
| three-mesh-bvh, SAH | 118 (2.59) / 4.05 (0.49) / 0.85 MiB | 353 (2.65) / 10.8 (0.46) / 2.3 MiB | 8,599 (2.72) / 231 (0.39) / 50.6 MiB |
| Rapier | 48.0 (1.05) / rebuilt / 20.3 MiB heap | 156 (1.17) / rebuilt / 54.5 MiB | 3,695 (1.17) / rebuilt / 1,141 MiB |
| bvh-tree | 102 (2.24) / n/a / 12.2 MiB | 424 (3.18) / n/a / 32.9 MiB | 15,461 (4.89) / n/a / 701 MiB |

Every refit agreed with ours on 512 rays (Babylon: 1 ray on Sponza, its test isn't watertight). `corners: 'positions'` answered every query as the records tree does (bit for bit by design; the suite saw no disagreement), at 0.91-1.06x its query and ray rates.

**Scene-level trees** (10,000 / 100,000 objects): build ours 36.7 / 414 ms, bvh.js 9.6 / 150 ms (0.26x / 0.36x our time), `ObjectBVH` 39.4 / 479 ms; update ours 32.6 / 412 ms, `ObjectBVH` 14.1 / 173 ms (0.43x / 0.42x), bvh.js 34.2 / 992 ms. Box culling at 10,000: bvh.js 1.49x our rate (0.50x at 100,000); frustum: bvh.js disagrees on 1 and 14 frusta, unranked.

**The rows the defaults can't change, on this CPU** (for the record; the M5 tables below are unchanged): CPU closest-hit rays, three-mesh-bvh SAH 1.37-1.39x our rate on the knot, 1.14-1.18x on the room, 0.99-1.18x on Sponza (CENTER 1.30-1.34x, 0.77-0.86x, 0.80-0.93x); Rapier 1.17-1.18x, 1.10-1.25x, 1.13-1.14x. All hits: three-mesh-bvh SAH 1.36-1.47x, 1.23-1.26x, 1.12-1.19x. Box, sphere and closest-point queries are still ours: three-mesh-bvh 0.27-0.91x our rate, except sphere queries on the knot (CENTER 1.02x, SAH 1.09x); Rapier 0.21-0.78x, its closest point on the knot a tie (1.01x). On the M5 our CPU rays led every library on Sponza and all but Rapier on the room; on this x86 laptop under Node 24, three-mesh-bvh's and Rapier's walks are ahead. One run, not repeated, not explained.

**Where others still win, 2026-11-16** (this section's numbers):

- **three-mesh-bvh, GPU:** the furnished room's closest hits, 0.51-0.52x ours from outside (0.96x with `f32` nodes: the quantized default is the cost) and 0.84-0.86x from inside; its any hit from inside the room 0.87x (0.96x with `'larger'`), and from outside 0.35x with the default order (1.21x with `'larger'`).
- **three-mesh-bvh, CPU refit:** 2.2-2.5x faster (CENTER) than our default records tree on every scene (knot 3.84 against 8.32 ms, Sponza 239 against 595 ms) and 1.6-1.9x faster than `corners: 'positions'` (Sponza 441 ms). The M5's "level on the knot and the room" did not hold on this CPU.
- **Memory:** three-mesh-bvh holds 51.5 MiB beyond Sponza's geometry; ours 459.7 MiB with records (8.9x), 284.4 MiB with `corners: 'positions'` (5.5x). On the GPU, 355.5 MiB against 214.3 (1.7x).
- **Rapier:** CPU closest-hit rays 1.10-1.25x our rate on all three scenes (on the M5: the knot from inside and the room); closest point on the knot a tie.
- **three-mesh-bvh, CPU rays, on this machine:** SAH 1.14-1.47x our rate on the knot and the room, up to 1.19x on Sponza (above).
- **bvh.js:** scene builds 2.8-3.8x faster (0.26-0.36x our time) and box culling 1.49x at 10,000 objects. **`ObjectBVH`** updates 2.3x faster than our CPU top-level rebuild.


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

The machine is an Apple M5 Pro (`os.cpus()` names it so): **Apple M5, indicative** applies to every number below. Node v22.22.2; Chrome headless through `pnpm examples:smoke` for the GPU. The raycast, build, tree-quality, memory and scene tables are from commit 4824f5b plus the uncommitted measuring changes that followed it, at a low load. The queries table and the refit columns are from the CPU query pass (2026-11-13, commit ad229e5; `compare/queries` below), whose run is now the rows in `bench/results/compare.csv` and the runner's full tables in `bench/results/compare.md`; that run was taken while other projects loaded the machine (load average 8-10), so its absolute rates are lower and noisier than the older tables' and only its ratios are quoted.

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

These rows (Apple M5, 2026-11-12) were taken with `f32` nodes, the default before 2026-11-13. For quantized nodes, the default now, see the RTX 3060 re-run of 2026-11-16 above: the room's closest hit from outside fell to 0.51x there (0.96x on `f32`). The ratio is ours over theirs. Where three-mesh-bvh disagrees on a ray (another distance; once a hit at t = 0) the ratio is shown but not ranked. GPU bytes are what each walk reads: ours, nodes, 48-byte triangle records and the top level; theirs, nodes, index, its vec4 copy of the positions and transforms. Absolute rates move by up to 30% between runs with the M5's heat and load; the paired ratios moved by under 0.05.

**The room's any hit from outside (0.31x before).** Not a harness artefact: both walks get the same packed rays and `tMax`, both stop at the first triangle hit in the interval (three-mesh-bvh's `bvh_RaycastAnyHit` returns `STOP_TRAVERSAL` from its range callback; its order is fixed, right child first), every ray is blocked in both, and ours is the leanest kernel. The counted walks (`?counts` on the page, `compare/gpu/gpu-counts.ts`: a counted copy of each library's own loop, same rays) found the cause. Mean work was close (ours 19.1 box tests and 3.9 triangles per ray, theirs 11.9 and 3.9), but the slowest ray of each 32 consecutive rays (one SIMD group) needed **78.5 box tests in ours against 24.3 in theirs**, which is the 3.2x. Our default any-hit order goes to the farther child first (chosen for shadow rays that start on a surface, T1.3); a ray from outside the room then descends into the far side's furniture before it meets the shell it already crossed. Near-first helped a little (0.41x; slowest of 32: 66.5). Entering the child with the larger box first (`anyHitOrder: 'larger'`: the likelier blocker, by the SAH's own area argument) cut the slowest of 32 to 23.0 and gave 1.73x. It is a setting, not the default: on the surface-ray bench it is 1.15-1.42x far-first on instanced trees but 0.78x on merged surface rays (`docs/internals/design/trace.md`, "Any-hit child order"). Workgroup size (64 both), the triangle test and the top level were not the cause: closest-hit, which shares them, was within 0.85-1.12x.

## compare/build: build, refit, memory

Median of 3 builds on one thread, ms; ratio to ours.

| Library | Knot build | Room build | Sponza build | Knot refit | Room refit | Sponza refit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| @pmndrs/bvh (tinybvh) | 33 | 84 | 1,499 | **3.2** (32 at first, 4.5 after the first fix) | **11.5** (was 75, then 13) | **364** (was 1,178, then 359) |
| three-mesh-bvh, CENTER | 25 (0.75) | 82 (0.97) | 1,538 (1.03) | 3.1 (0.98) | 11.5 (0.99) | 248 (0.68) |
| three-mesh-bvh, SAH | 110 (3.3) | 352 (4.2) | 7,011 (4.7) | 3.1 (0.96) | 10.3 (0.89) | 241 (0.66) |
| bvh-tree | 92 (2.8) | 270 (3.2) | 11,301 (7.5) | n/a | n/a | n/a |
| Babylon, submesh octree | 9.9 (0.30) | 17 (0.20) | 421 (0.28) | 2.2 (octree rebuilt) | 4.6 | 57 |
| Rapier | 22 (0.65) | 70 (0.83) | 1,329 (0.89) | n/a (rebuilt) | n/a | n/a |
| ours, H-PLOC (CPU twin) | 282 | 872 | 10,699 | | | |
| ours, LBVH (CPU twin) | 113 | 335 | 9,170 | | | |

**Refit, the 6-7x.** The refit columns are new for ours and three-mesh-bvh CENTER: medians of 21 refits interleaved between the two (order reversed every other round, alternating two moved copies), at a low load; the SAH row keeps the earlier run's times, without ratios. Before, the same paired run gave ours 16.1 / 37.3 / 1,001 ms against CENTER's 2.6 / 7.6 / 169 (0.16-0.20). The cost was ours, not the layout: `refitOnCpu` found its node order by walking every node's parent chain and sorting with a comparator, kept a JavaScript array per node box, and built a temporary array per primitive union; `CpuMeshBvh.refit` validated every corner in one pass, allocated a per-triangle box array, then rewrote the corners and the boxes. Now the order is a stable counting sort on depth (worked out once per tree and kept), node boxes are one `Float64Array`, and the refit reads the leaves' corners straight from the triangle records (`refitRecordsOnCpu`, bit for bit `refitOnCpu` over boxes of the same corners); validation is still a complete first pass that only reads, so a refit that throws leaves the tree untouched. That left ours at 1.6-2x three-mesh-bvh's: it still writes 48-byte records (nine corner floats per triangle) and two-box 64-byte nodes, where three-mesh-bvh reads the app's positions in place.

**Refit, the CPU query pass (2026-11-13).** The refit columns are now the `bench/results/compare.csv` run (commit ad229e5, load average 8-10, the suite's median of 5 refits per library; the older columns were 21 paired refits at a low load). Profiled first: on Sponza the per-corner validation pass took 92 ms, writing the records 107 ms and `refitRecordsOnCpu` 381 ms (its f32 format pre-pass 50 ms of that), measured phase by phase under load. Now validation is two straight scans (every index below the vertex count, every position finite; the per-triangle check runs only when one fails, so an unused non-finite vertex is still allowed and a refit that throws still leaves the tree untouched), and one pass writes each leaf's records and its box, then each inner box from its child's two stored slots: nodes from last to first when every child's index is above its parent's (tinybvh's layout, checked once per tree), else in the parents' depth order. No node-box array, no second read of the records; the boxes are `refitRecordsOnCpu`'s bit for bit (tested on tinybvh and LBVH trees). Same process, interleaved against the code before: 2x (0.49-0.55 of the old time on the knot, the room and Sponza). Against three-mesh-bvh CENTER over two full before and two full after runs, as the table gives it (its time over ours, below 1: faster than ours): 0.24-0.56 before, 0.94-1.08 after on the knot and the room, 0.68-0.79 after on Sponza (the table's run: 0.98, 0.99, 0.68). Sponza is what is left: ours still writes 48-byte records and touches 3 million 64-byte nodes (leaves of about 2 triangles), where three-mesh-bvh writes 1.7 million 32-byte nodes (leaves of about 7) and reads the app's positions in place. `refitOnCpu`'s format pre-pass (every node read as an `f32` node before anything is written, so quantized words are refused whole) is one unrolled loop now: 4-7% of a `refitOnCpu` call, from 12-14%.

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

Thousands of queries per second (2,048 seeded queries each), ratio to ours (above 1: faster than ours). Only three-mesh-bvh and Rapier have these queries; every answer agreed (Rapier's by distance). The run of `bench/results/compare.csv` (commit ad229e5, load average 8-10); the ratios in brackets are that run's, with the second after-run's in the text below.

| Library | Knot: box / sphere / closest point | Room | Sponza |
| --- | --- | --- | --- |
| @pmndrs/bvh | 1,090 / 828 / 173 | 1,080 / 1,090 / 597 | 731 / 456 / 312 |
| three-mesh-bvh, CENTER | 444 (0.41) / 779 (0.94) / 88 (0.51) | 285 (0.26) / 520 (0.48) / 138 (0.23) | 243 (0.33) / 462 (1.01) / 87 (0.28) |
| three-mesh-bvh, SAH | 457 (0.42) / 873 (1.05) / 93 (0.54) | 386 (0.36) / 708 (0.65) / 251 (0.42) | 301 (0.41) / 566 (1.24) / 114 (0.37) |
| Rapier | 904 (0.83) / 149 (0.18) / 171 (0.99) | 721 (0.67) / 355 (0.33) / 417 (0.70) | 411 (0.56) / 170 (0.37) / 199 (0.64) |

**The CPU query pass (2026-11-13).** Before it, Rapier answered box queries 2.3-2.7x and closest point 1.7-2.4x our rate, and three-mesh-bvh box queries 1.2-1.4x (the table that stood here, commit 4824f5b). Profiled first (`node --cpu-prof`, the suite's queries): an `intersectsBox` spent about a quarter of its time making its `OrientedBox` (eight corner objects, axes, bounds, an inverse) and in the garbage collector, and most of the rest in the shapecast's closures, copying each node box into a `Box3`; the separating-axis test itself was 8.5%. `closestPointToPoint` spent 44% in `traverse` and its child scoring, 10% in the triangle test. Both are now flat walks over the typed arrays with no objects per node or triangle (`src/query/cpu-mesh-bvh.ts`): the same boxes tested in the same walk order, the oriented box packed once into numbers, the triangle tests three-mesh-bvh's arithmetic on plain numbers read from the records, and closest point's children pushed only when their score passes (the best only shrinks, so this changes nothing). An axis-aligned box tests node boxes on its aligned axes only: its other three axes are `(min - max)` along x, y and z, and a product with one number is monotone, so they can't part what the aligned ones didn't. One more lever was measured, not guessed: V8 doesn't inline the triangle tests, and boxes every number argument of a call it doesn't inline, so they take their numbers through scratch arrays. Every answer is the old code's bit for bit, ties included (`src/query/fast-paths.test.ts`: touching, inverted, flat, NaN and infinite boxes, rotated boxes, degenerate and NaN triangles, a tie fan, thresholds, one-leaf trees). Same process, interleaved rounds against the code before: box 3.1-3.3x, closest point 2.6-3.3x. Against the libraries over two full before and two full after runs (each library's ratio, before / after): Rapier box 2.07-2.94 / 0.56-0.88, closest point 1.97-2.73 / 0.64-1.04 (the knot is the tie: 0.99 and 1.04); three-mesh-bvh box 0.85-1.74 / 0.26-0.47, closest point 0.75-1.64 / 0.23-0.54. Sphere queries were not changed, and are a loss against three-mesh-bvh SAH: 1.05-1.13 times our rate on the knot and 0.95-1.24 on Sponza (CENTER 0.94-1.03 and 0.74-1.01); ours leads on the room.

Since that run, `closestPointToPoint` also handles slivers: on a triangle whose squared sine at `a` is at most 1e-7 (the GPU's `BVH_SLIVER_SINE_SQ`) it keeps the nearer of three's point and the nearest point of the three edges, because three's regions cancel there even in doubles (up to 1.7 units too far on a unit triangle in a sweep of f32 slivers; the GPU query oracle found a point lying on one reported 2.2e-3 away). It costs 2-5% of a closest-point query, not in the table above.

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

## Where another library wins (Apple M5, 2026-11-13; the newer list is in the 2026-11-16 re-run above)

- **Rapier**: closest-hit rays on the knot (inside, 1.28x) and the room (1.02x outside, 1.16x inside); within 5% of ours on Sponza. Closest point on the knot is a tie (0.99-1.04x ours); box queries (2.3 to 2.7 times our rate) and closest point (1.7 to 2.4 times) were its lead until the CPU query pass (2026-11-13), and are now 0.56-0.88x and 0.64-0.77x ours elsewhere. It gives no triangle id, and holds 2.5 times our tree's memory (below).
- **three-mesh-bvh**: CPU refit on Sponza 1.3 to 1.5 times faster (6 to 7 before the first refit fix, 1.6 to 2 before the CPU query pass; a tie on the knot and the room now, 0.93-1.06; 32-byte one-box nodes over the app's positions, where ours rewrites 48-byte records and two-box nodes); sphere queries a little faster than ours with SAH on the knot (1.05-1.13) and Sponza (0.95-1.24; CENTER 0.94-1.03 and 0.74-1.01); its box queries (1.2 to 1.4 times ours before) and closest point (up to 1.5 times) are 0.23-0.54x ours now; CENTER builds as fast as tinybvh; `ObjectBVH` updates twice as fast as our CPU top-level rebuild. On the GPU its `/webgpu` walk beats ours on the furnished room's closest hits (0.85 to 0.91x) and on Sponza's closest from outside (0.96x, though 7 of its rays disagree). Its any hit from outside the room was 3.2 times our default kernel's (982 against 304 Mrays/s); with `anyHitOrder: 'larger'` ours is 1.73 times theirs there (above). Its worker build stalls frames a little less than ours.
- **bvh.js**: builds a scene tree about 3 times faster than ours and culls boxes 1.2 times faster at 10,000 objects; its frustum culling is faster too, but disagrees on a few objects.
- **Babylon**: builds fastest (it builds almost nothing); its picking is the floor it looks like.
- Ours leads closest and all hits against every library on Sponza, against all but Rapier on the room; box queries and closest point against every library (Rapier's closest point on the knot a tie); sphere queries against Rapier everywhere and three-mesh-bvh on the room (up to 2 times its rate).

## Memory

What each library holds beyond the app's geometry. "Own": the library's exact buffers; "heap": one build's growth in a fresh Node process (V8 heap plus array buffers and WASM memory, after gc).

| | Knot own / heap | Room own / heap | Sponza own / heap | Sponza bytes per triangle (heap) |
| --- | --- | --- | --- | ---: |
| @pmndrs/bvh | 8.0 / 8.1 MiB | 20.1 / 20.2 MiB | **459.7 / 391 MiB** (was 1,740, 1,285 of it tinybvh's WASM heap, before the builder released it) | 84 (tree) |
| three-mesh-bvh, CENTER | 0.85 / 0.94 MiB | 2.4 / 2.5 MiB | 51.5 / 47.1 MiB | 9 |
| bvh-tree | 12.2 (arrays only) / 19.6 MiB | 32.9 / 54.9 MiB | 701 / 1,152 MiB | 210 |
| Babylon, submesh octree | not counted / 6.6 MiB | 19.0 MiB heap | 363 MiB heap | 66 |
| Rapier | 20.2 / 20.3 MiB (WASM) | 54.4 / 54.5 MiB | 1,145 / 1,141 MiB | 208 |
| three `Mesh.raycast` | 0 / 0.05 MiB | 0 / 0.05 MiB | 0 / 0.04 MiB | 0 |

The gap, named:

- **Ours against three-mesh-bvh: 9 to 10 times** the bytes beyond the geometry (Sponza 460 against 47 to 52 MiB; 84 against 9 bytes per triangle), as `docs/internals/design/memory.md` found per mesh (8 to 9 times). Our tree copies every triangle's corners into a 48-byte record (56% of it), so the GPU walk reads one buffer and never the app's index; our nodes are 64 bytes with two child boxes; our leaves hold about 2 triangles against their 7, so there are 3.6 times the boxes. three-mesh-bvh reuses the app's positions and reorders the app's own index in place. The same trade shows on the GPU: 470 against 225 MB with `f32` nodes, where their walk reads its own vec4 copy of the positions (quantized nodes, the default since 2026-11-13, measured 360.8 MiB settled against 458.6 with `f32` nodes under the default cap, `scene-trees-sponza` on the M5). Indexed leaves and quantized nodes (`docs/internals/design/indexed-leaves.md`, `compressed-nodes.md`) are the levers that close part of it.
- **tinybvh's WASM heap (fixed on the wrapper side).** Building Sponza as **one** 5.74M-triangle mesh grew tinybvh's WASM memory to 1,349 MiB, and a WASM memory never shrinks, so the builder kept it for its life. Where it goes, step by step: our input copies 102 MiB, tinybvh's build (its vertex copy and build arrays) 599 MiB, the layout it writes 649 MiB (nodes allocated for the worst case, 373 MiB for 197 used, and 263 MiB of records, all copied out). Now: our positions copy is freed after tinybvh's build (1,316 MiB peak); `createTinyBvhBuilder` drops an instance whose build left it past `releaseAboveBytes` (default 256 MiB) and starts a fresh one, so the build's heap growth in a fresh process is **391 MiB instead of 1,707** (the table's ours-Sponza heap); and the worker pool ends a worker whose build grew past `recycleAboveBytes` (default 256 MiB). The peak itself stays: lowering it needs `native/` changes (`docs/internals/design/memory.md`). Per-mesh builds (how `SceneTrees` builds Sponza's meshes) stay at its 64 MiB start; the knot and the room didn't grow it at all.
- bvh-tree and Rapier hold about 210 bytes per triangle each (bvh-tree: a JS array of 9 doubles and two 28-byte box arrays per triangle, plus node objects; Rapier: its copy of the trimesh and parry's tree in WASM). Babylon holds 66 (submeshes, the octree and the `Vector3` position cache its picking makes).

## Honest gaps

What three-mesh-bvh has and we don't (`docs/internals/PARITY.md`), and what the others have:

- **WebGL2** GPU traversal (GLSL, `MeshBVHUniformStruct`): won't (8.12). Our GPU side needs WebGPU.
- **Points and lines** (`PointsBVH`, `LineSegmentsBVH`): missing (3.2, 3.3); our leaves are triangles only.
- **GPU closest point, GPU shapecast, the WGSL triangle helpers, `ClusteredBVH`, SDF generation**: missing (8.5, 8.6, 8.8a, 8.9, 8.10). Only rays run on our GPU walk. `sampleTrianglePoint` and `ndcToCameraRay` won't be done (8.7, 8.8b): the package ends at the hit.
- `ObjectBVH.shapecast` (5.3) and `getJSONStructure` (7.5): missing. Partial: `SkinnedMeshBVH` (3.4, GPU posing only), `maxDepth` (4.3, accepted, not honoured), per-index `computeBatchedBoundsTree` (5.5), `StaticGeometryGenerator` (6.3, GPU only), three-mesh-bvh's serialization format (7.1), `onProgress` (0 and 1 only). `ParallelMeshBVHWorker`: won't (cross-origin isolation; big meshes go to the GPU builder).
- **Memory**: 9 to 10 times three-mesh-bvh's beyond the geometry for our CPU trees (what this suite measures); our GPU scenes, with quantized nodes by default, are about 7 times on Sponza (`docs/internals/design/memory.md`). A huge single-mesh build still peaks at about 230 bytes per triangle of WASM memory while it runs (above); the builder now releases it afterwards.
- **CPU refit speed** on a huge mesh (1.3 to 1.5 times slower than three-mesh-bvh's on Sponza, from 6 to 7 and then 1.6 to 2; level on the knot and the room), and years of use behind their shapecasts.
- Rapier is a physics engine (collisions, character controllers, joints) and answers box and point queries faster; Babylon is a whole engine; bvh.js inserts, deletes and moves single objects and has LOD frustum culling. We have none of that.

What we have and they don't: GPU builds (H-PLOC, LBVH), GPU refits of mesh trees and the top level, one layout that every builder writes and every walk (CPU and WGSL) reads, a WGSL walk outside three's node system (a raw compute pass, another engine, a worker's device), and the oracle (the CPU port of the walk that checks the GPU's answers).

## What the suite doesn't cover yet

- The pages' CPU numbers in Chrome are not tabled here; Node's are (the same suites). The compare/build page can't finish Sponza within the runner's DevTools timeout.
- The plan's `compare/skinned`, `compare/pathtracer`, `compare/sdf` and the `shapecast`/`distancecast`/`characterMovement` ports are not here (BX.18).
- The CPU suites ran once on the RTX laptop's i9-11900H (2026-11-16, above); the M5 has not run `corners: 'positions'` or the quantized GPU rows.
- Rapier gives no triangle ids, so its agreement is by distance only.

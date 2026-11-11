# Handoff: pmndrs/bvh after the split (2026-11-10)

This repo was split out of `DennisSmolek/webgpu-ddgi`, where the code lived in `packages/bvh`. The full plan is [docs/design/bvh-package-split.md](docs/design/bvh-package-split.md): the decisions (D1–D8), the target layout, phases P0–P9, examples, docs and tasks BX.0–BX.20. Read it first.

## Current state (end of 2026-11-11). Read this first; the sections further down are history.

**Dennis's decisions:**
- **Source of truth:** this repo, for BVH and walk code. webgpu-ddgi's copy is frozen until P3, and its split point is `d288adf`.
- **Commits:** local only, never pushed. Opus writes the code and does the reviews; Sonnet or Haiku take the mechanical tasks.
- **Walk API:**
  - WGSL names use `bvh*`.
  - Filter hook: `bvhTrianglePasses(BvhCandidate)`, where the candidate includes `query` and `rayMask`. ddgi's materials stay in ddgi.
  - 8-bit instance masks, compiled out with `rayMasks: false`.
- **`mergeStaticInstances`** comes before the release. It stays off by default until the RTX 3060 has been measured.
- **Package scope:** it should serve every project that needs a BVH, not just ddgi. See "Consumers" below.

**On `main`** (429 tests; lint, typecheck, build and the licence check pass; every correctness page passes on the Apple M5):
- **Ported from ddgi `main` after the split:** shared refit kernels, and the Metal watertight fix `bvhEdgeFunction`. On the M5, rays slipping through shared edges went from 4,513 per 2M to 0.
- **GPU page runner:** `pnpm test:gpu`, `pnpm test:gpu:page <name>`; see `tests/gpu/README.md`. Scene pages need `BVH_ASSETS_DIR=~/Developer/webgpu-ddgi-radiance-cache/assets` and `BVH_MANIFESTS_DIR=~/Developer/webgpu-ddgi-radiance-cache/scenes/manifests`.
- **Walk API for 0.1**, at `GPU_LAYOUT_VERSION` 2 (unreleased):
  - names, filter and masks as above;
  - `BvhHit.normal` and `BvhHit.material` (material + 1 in instance flags bits 16-31);
  - a 40-byte hit record by default, or 24 bytes with `hitNormal: false`.
- **`mergeStaticInstances`:** a merged world-space tree with exact hit identity. Triangle word 11 holds the source instance, flag bit 1 flips the face, and instance bit 3 marks a merged instance. App triangle flags may use bit 0 only.
  - Measured on the package under the radiance-cache lane's bench lock: Sponza surface rays 1.85x closest-hit, 1.36x any-hit; Bistro interior 1.22x and 1.11x.
  - Pages: `trace-bench-surface-*`. Rows are in `bench/results/trace.csv`.
- **Indexed leaves** (2026-11-12, off by default; `docs/design/indexed-leaves.md`): `triangleLeaves: 'records' | 'indexed'` and `positionLayout: 'vec4' | 'packed'` on the walk (compiled in, one walk), the builders, `assembleTraceScene`/`buildTraceScene`, `SceneTrees` and `BvhTraceNodes`. 16-byte `vec4u(i0, i1, i2, id + flags)` triangles, a 16-byte header per mesh (vertex base, read once per instance entered), our own positions (exact-bit weld on, `weld: false`; opt-in `weldTolerance`), merged groups with a side table, serialized format in header word 7. Sponza 448.1 → 318.0 MiB beyond geometry (306.7 packed), Bistro exterior 210.1 → 155.5. M5 speed: `vec4` 0.83-0.99x records, packed 0.75-0.92x (indicative). All correctness pages pass with indexed variants; one-ulp seams leak alike in both formats, the tolerance weld closes them.
- **`SceneTrees`** (`src/scene/`):
  - GPU H-PLOC trees first, then tinybvh trees from `@pmndrs/bvh/worker` swapped in per mesh;
  - growable storage;
  - `GpuTopLevel`;
  - meshes and instances added or removed at runtime, merged static groups, deforming meshes.
  - On Sponza: first trees in 100-118 ms, every tinybvh tree in about 370 ms. Peak memory is 1,483 MB, during the H-PLOC build itself.
- **`@pmndrs/bvh/three`** (three is a peer, `>=0.187.0 <0.188.0`); see `docs/three.md`:
  - `bvhSceneFromObject3D` and `SceneTreesObject3DSync`;
  - `BvhTraceNodes`, for TSL and `wgslFn`. It binds through three internals, collected in `threeInternals`, and checks that binding at run time;
  - `computeBoundsTree` and `acceleratedRaycast`, which run at three-mesh-bvh's speed. `computeBoundsTree` takes three-mesh-bvh's options (`strategy`, the three leaf-size names, `maxDepth`, `setBoundingBox`, `indirect`, `onProgress`; mapping in `docs/three.md`, "Build options");
  - `geometry.boundsTree` is a `BvhBoundsTree`: the core `CpuMeshBvh` with three's types (`raycast`/`raycastFirst` on a three `Ray`, `refit()`, `resolveTriangleIndex`, `serialize`, `getBoundingBox`, `estimateMemoryInBytes`), plus `getTriangleHitPointInfo` and `GenerateMeshBvhWorker.generate(geometry, options)` over the worker pool.
- **CPU queries** (`src/query/`, exported from the root; PARITY items 1, 2, 3 and 5): `CpuMeshBvh` over one tree in our layout. `raycast` and `raycastFirst` run the one CPU walk (`createCpuTracer`, now with `filterEveryTriangle` for side culling and all hits); `traverse` is the box walk; `shapecast` (three-mesh-bvh's contract, `ExtendedTriangle` per triangle, original indices), `intersectsBox`, `intersectsSphere` and `intersectsGeometry` run on it; `bvhcast` is three-mesh-bvh's dual walk step for step on our nodes (`intersectsGeometry` uses it against a tree); `refit` uses `refitOnCpu`. `src/query/shapes.ts` is three-mesh-bvh's `ExtendedTriangle` / `OrientedBox` math, engine-free, bit-for-bit equal to it; `/three` exports three-typed `ExtendedTriangle` (a `Triangle`) and `OrientedBox` over the same functions. `/worker` has `generateCpuMeshBvh`. `bench/query-parity.ts` runs homefig's capsule controller and three-bvh-csg's `collectIntersectingTriangles` against three-mesh-bvh on the furnished room: full agreement, 0.93x and 0.94x its time. `closestPointToPoint` and `closestPointToGeometry` (PARITY item 6) are three-mesh-bvh's shapecasts step for step (`BvhBoundsTree`'s return `Vector3`s, take a `BufferGeometry`); against three-mesh-bvh the same distances everywhere tested, the same triangle unless tied; `bench/closest-point.ts` on the room: 0.92x its time for points in the open, 1.1-1.2x near surfaces, 1.08x geometry against geometry. three-mesh-bvh's debug functions `getBVHExtremes`, `validateBounds` and `estimateMemoryInBytes` (`src/query/debug.ts`, from the root and `/three`) run on `CpuMeshBvh.traverseNodes`, three-mesh-bvh's node walk. `/three` has three-mesh-bvh's `BVHHelper` / `MeshBVHHelper` (PARITY item 7; segment counts tested in Node, not yet looked at in a browser). `docs/PARITY.md` now counts 28 have, 5 partial, 15 missing.
- **Packaging:**
  - `tsc` builds `dist/`;
  - `scripts/verify-artifacts.mjs` checks that the licence text, the worker URLs and `/three` survive the build;
  - ESLint enforces the engine-free core: only `src/three` may import three;
  - CI is in `.github/workflows/ci.yml`. Its `wasm` and SwiftShader jobs have never run, because nothing has been pushed.
- **Docs:**
  - `docs/DIVERGENCES.md`: 32 rows, plus 8 items marked "needs Dennis";
  - `docs/MIGRATING-FROM-DDGI.md` (P3);
  - `docs/MIGRATING-FROM-REFLECTIONS-BRANCH.md`;
  - `docs/three.md`;
  - `docs/design/trace.md`.

**Consumers** (from a survey of ~/Developer on 2026-11-11):
- **ddgi and its rc-\* copies:** still on the frozen copy. They switch at P3.
- **reflections:** linked to the old-API branch `reflections-integration`. It can move to `main`; see its migration doc.
- **Denoiser, upscaler, ddgi-research:** use three-mesh-bvh `/webgpu`. They can switch through `/three`, as `docs/three.md` describes.
- **three-bvh-csg, homefig, drei, three-gpu-pathtracer:** drei and pathtracer's CPU side have what they use. homefig's controller and csg's boolean core have what they use (shapecast, `bvhcast`, `ExtendedTriangle`); neither has switched.

**Open:**
- **One unexplained `scene-trees-mid-swap` failure (2026-11-12).** On the indexed-leaves branch it failed once after 405 s while `pnpm build` ran in parallel, and the message wasn't captured. The page limit is 900 s, so it wasn't the page timeout; a DevTools request timeout under load is possible. It has passed 3 times since (about 170 s each, 0 disagreements over 87.7M rays). If it fails again, keep the full log.
- **Indexed leaves:** the RTX 3060 and Intel passes (oracle pages, then `trace-bench-surface-*`, whose last section times indexed against records) decide whether indexed becomes the default (Dennis). Not built: a WASM writer for tinybvh (it converts after), indexed output from the GPU LBVH (no binding left in its emit), `CpuMeshBvh` reading corners through positions (it expands to records), the adopt lever with indexed scenes, binding the app's positions (option A). Needs Dennis: the vertex base lives in a per-mesh header, not added into the records as the design said (records stay relocatable); merged groups are now capped at 65,536 sources.
- **RTX 3060 and Intel:** none of today's work has run on them. The radiance-cache lane runs its RTX pass for the merged tree.
- **The 40-byte hit record's cost:** measured on the M5 only (2026-11-11): no measurable cost, ratios 0.96-1.15 against the 24-byte record, medians 1.02-1.11; see `docs/design/memory.md`. RTX 3060 and Intel not run.
- **The `LICENSE` file:** missing, and the holder (Dennis Smolek or Poimandres) is undecided.
- **`bvh:wasm --check`:** needs the pinned emsdk, which isn't downloaded.
- **Peak memory:** BX.19's breakdown is in `docs/design/memory.md`. The build-peak lever is in: `SceneTrees` caps each H-PLOC batch at **256 MiB by default** (`gpuMaxBatchBytes`, `Infinity` for none), building the batches one after another on one scratch pool (one queue per `GpuFirstTrees`), each stored between updates. Sponza on the M5: peak 1,483 → 711 MiB (−771), settled 470 → 459, first trees 171 → 166 ms (the first cut cost about +165 ms; batches no longer wait for updates). RTX 3060 and Intel not run. The rest of the levers are BX.20.
- **`/three` ranges (2026-11-12):** draw ranges and groups that start off a multiple of 3, end mid-triangle or overlap now match three's `Mesh.raycast` step for step, in `bvhSceneFromObject3D` (`threeTriangleRanges` counts three's loop) and in `acceleratedRaycast` (ranges at 1 or 2 mod 3 walk a shifted tree built on first use and refitted with the tree; overlapping groups walk once each). Needs Dennis: three r187 reports the triangle that reads past the end of the index at a `NaN` distance (its watertight test fails no comparison on `NaN` corners), which also leaves its sorted list out of order; we report nothing there. That looks like a three bug worth filing.
- **`/three` skinned and morphed meshes (2026-11-12):** with a `renderer`, `SceneTreesObject3DSync` adds every `SkinnedMesh` and morph-target mesh as a deforming mesh of its own and, each `update()`, runs its `BvhPosePass` (`src/three/deform.ts`: a TSL compute pass, three r187's morph and skinning math in three's order) into `SceneTrees.setPose`; nothing is read back. New core API `SceneTrees.setDeformBounds(mesh, box)`: the sync moves a posed mesh's top-level box from the skeleton (a conservative per-bone box on the CPU, `deformSlack` room), so root motion works. GPU page `three-skinned` (M5): posed vertices within 6.4e-7 of three's `getVertexPosition`, 0 disagreements over 16,384 rays at 4 frames, records and indexed, with 4 drift rebuilds and 10 box moves; it fails (164-1,038 disagreements) when the box moves are switched off. CPU picking on skinned meshes stays on three's raycast. Needs Dennis: the deform box is per mesh, set from the CPU (no GPU readback of the refit root); per-instance morphs (`morphTexture`) aren't read.
- **Still to do in `/three`:** a CPU scene raycast, and Windows compile cost (the pose pass has one shader per variant, not measured on Windows).
- **API tidy:** `export *` in `src/trace/index.ts`, the batch helpers in `gpu-lbvh.ts`, and folding the two layout files into one.
- **Next phases:** Babylon (P6), the P4 examples and docs site, the comparison suite (P5), and the rest of CPU queries (P8): `StaticGeometryGenerator` (PARITY item 8), object BVH queries (9), and item 12's three-mesh-bvh-format `serialize` and `getJSONStructure`.
- **Needs Dennis (CPU tree):** `setBoundingBox` defaults to false (three-mesh-bvh: true); `strategy` CENTER/AVERAGE build the SAH tree; `maxDepth` is accepted, not honoured; no `MeshBVH` alias is exported (`BvhBoundsTree` takes the same constructor arguments).
- **Needs Dennis (shapecast, bvhcast):** `/three`'s `OrientedBox` is its own class with `Vector3`/`Matrix4` fields, as three-mesh-bvh's is, not a `Box3` subclass (inherited `Box3` methods would ignore the matrix); `intersectsGeometry` against a tree runs a `bvhcast` (three-mesh-bvh 0.9.18 shapecasts the other tree from each leaf; same answers, 1.5x faster here); `ExtendedTriangle.set`/`copy` set `needsUpdate` (three-mesh-bvh's `copy` doesn't); `bvhcast` copies three-mesh-bvh's depth bookkeeping, which passes the switched-to node's parent depth, and names leaves by child slot.
- **Needs Dennis (closest point):** `maxThreshold` culls boxes, not triangles, as in three-mesh-bvh, so a point a little past it can come back (filtering triangles too would be one line, and would differ from three-mesh-bvh); `closestPointToGeometry`'s `faceIndex`es are original triangles in every case (three-mesh-bvh 0.9.18's indirect mode returns tree positions, which looks like a bug there).
- **Needs Dennis (utilities):** our trees have one root, so `getBVHExtremes` returns one entry; the layout doesn't store split axes, so `splits` counts the axis the children's centres lie furthest apart on (or we drop the field, or store the axis); `validateBounds` reads the source geometry (as three-mesh-bvh reads its geometry), so it reports edits not yet refitted, not a broken layout (`checkGpuLayout` does that).
- **Needs Dennis (helper):** `MeshBVHHelper` is exported without three-mesh-bvh's deprecation warning (drei's story uses that name); there is no `BatchedMesh` `boundsTrees` lookup, since we have no per-geometry batched trees (PARITY 5.5).

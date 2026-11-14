# History

A short, dated history of how `@pmndrs/bvh` got here. It is written from `HANDOFF.md`, the git log and the dated
entries in the design notes under `docs/internals/`. The measurements and the reasons are in those notes; this page
only says what happened and when.

The code began in a different repository: `DennisSmolek/webgpu-ddgi`, as `packages/bvh`, the ray tracer under a GPU
global-illumination library. The commits before 2026-11-10 below are from there. The package has never been published
to npm, tagged or released; its version is 0.0.0.

## 2026-10-30 to 2026-11-04: a tracer, then a package

- **2026-10-30:** webgpu-ddgi starts.
- **2026-11-04, the tracing spike (T0.10):** a two-level BVH with a WGSL closest-hit and any-hit walk and a GPU oracle.
  The watertight ray-triangle test lets no ray through a shared edge. Benchmarked on an RTX 3060 and an Intel UHD,
  then optimized: Sponza went from 61 to 104 Mrays/s on the RTX 3060, 1.20x three-mesh-bvh's.
- **2026-11-04, decisions:** the BVH becomes its own engine-free package, H-PLOC is the target builder, and tinybvh is
  compiled to WebAssembly as the CPU builder (B0). Sponza's CPU build falls from 28 s (the first TypeScript builder) to about 2 s.

## 2026-11-05: the GPU builders

- **B1a:** a GPU top-level refit every frame, with shape rebuilds in a worker when quality drifts.
- **B1:** a GPU LBVH (Morton codes, radix sort, Apetrei climb), batched over a whole scene's meshes, and a bottom-level
  refit. Sponza builds in 31.2 ms.
- **B2:** GPU H-PLOC, ported from tinybvh's `BuildHPLOC`. Sponza builds in 46.6 ms with trees 1.032x tinybvh's SAH cost.
- **B3:** CWBVH (compressed 8-wide nodes) tried and dropped by Dennis: no faster on Sponza, slower on Bistro.

## 2026-11-06 to 2026-11-10: use and split

- **2026-11-06:** the first user-facing README and the plan for splitting the package out.
- **2026-11-08:** ddgi loads with GPU H-PLOC trees first and swaps tinybvh's in per mesh (the pattern that became
  `SceneTrees`).
- **2026-11-09 and 2026-11-10:** the public API is trimmed, `GPU_LAYOUT_VERSION` and layout serialization are added,
  and the walk, oracle and builders move into `@pmndrs/bvh`. The standalone repository is bootstrapped, and
  `HANDOFF.md` is written for the next sessions.

## 2026-11-11: the repository stands up

The ddgi parts are ported in and repointed: the GPU page runner, the refit kernels and the Metal watertight fix
(`bvhEdgeFunction`). The walk's API is settled for 0.1 (`bvh*` WGSL names, `bvhTrianglePasses`, 8-bit instance masks).
The package builds to `dist/` with a licence check and an engine-free lint rule, and CI is written (it has not yet run
on GitHub). Built the same day: `mergeStaticInstances`, hit normals and materials, `SceneTrees` with the worker pool,
`@pmndrs/bvh/three` (scene sync, `BvhTraceNodes`, CPU raycast drop-ins), the three-mesh-bvh parity audit
(`PARITY.md`), and the memory breakdown.

## 2026-11-12: CPU queries, layout variants, docs

- **CPU queries:** `CpuMeshBvh` with raycast, shapecast, `bvhcast`, closest point and the debug helpers, matching
  three-mesh-bvh's contracts.
- **Layout variants:** indexed leaves and quantized nodes, designed first and built the same day.
- **Memory:** the build-peak cap on `SceneTrees` (256 MiB by default).
- **`/three` gaps:** draw ranges and groups, and GPU skinning and morph posing.
- **Docs and examples:** the docs site (34 pages, checked code samples, TypeDoc), the three.js examples, and the
  comparison suite against three-mesh-bvh, three, bvh-tree, Babylon.js, bvh.js and Rapier.

## 2026-11-13: defaults, vanilla examples, GPU queries

- **Defaults changed after an RTX 3060 run:** quantized nodes and merged static instances are on by default. Indexed
  leaves stayed off.
- **The scope rule:** the package ends at the hit. What a hit looks like belongs to renderers above it.
- **Vanilla WebGPU examples:** `hello`, `raytrace`, `instances`, `builders`, `refit`, `fast-first`, `workers-cpu` and
  `workers-gpu`, on a three-free shell.
- **GPU spatial queries:** closest point, sphere and box tests and a caller-hooked shapecast in WGSL.
- **CPU query pass:** box and closest-point queries rewritten as flat typed-array walks (about 3x faster).

## 2026-11-14: parity

The last big day. Built, each merged on its own: Babylon.js support (`@pmndrs/bvh/babylon`), signed distance fields,
scene shapecast and batched trees, `StaticGeometryGenerator`, three-mesh-bvh's serialization format, serialization and
baking in Node, `ClusteredBVH`, points and lines, `SkinnedMeshBVH`, and ports of three-mesh-bvh's examples. The audit
against three-mesh-bvh 0.9.18 ended at **45 have, 2 partial, 0 missing and 5 won't** (renderer work that sits above the
hit, and three's own types).

Pathtracer's requests (pmndrs/bvh#1) landed the same day: workers that never lose a build when they can't load, the
exported shared-edge ray fixtures, and a measured answer on a CPU walk that rounds like the GPU (it can't).

## What is next

- The RTX 3060 and Intel UHD still owe correctness and memory runs on the 2026-11 defaults
  ([the runbook](docs/internals/WINDOWS-RUNBOOK.md)); Dennis decides the indexed-leaves default from them.
- Publishing the docs site, which needs Pages settings and DNS ([Publishing the docs](docs/internals/README.md)).
- A first npm release, once the API is settled for 0.1.

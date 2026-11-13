# @pmndrs/bvh

BVH builders, refits and a ray walk for WebGPU, with no engine inside. You give it triangles; it gives you bounding
volume hierarchies in one fixed GPU layout, built on the CPU (tinybvh in WebAssembly) or on the GPU (H-PLOC or LBVH),
refit on the GPU when things move, and read by a WGSL walk you call from your own shaders. The same trees serve raw
compute passes, three.js, workers and other libraries on the same device. CPU raycasts, shapecasts and closest-point
queries run over the same layout, with three-mesh-bvh's API in `@pmndrs/bvh/three`.

**Docs: [bvh.docs.pmnd.rs](https://bvh.docs.pmnd.rs)** (the source is [`docs/`](docs/getting-started/introduction.mdx);
the API reference is at [bvh.docs.pmnd.rs/api](https://bvh.docs.pmnd.rs/api/)).

> **Status:** version 0.0.0, not on npm yet. The API can change before 0.1. Its first user is DDGI
> ([webgpu-ddgi](https://github.com/DennisSmolek/webgpu-ddgi)), but the package imports nothing from it or from any engine.

## Install

```bash
npm install @pmndrs/bvh
npm install three@0.187   # only for @pmndrs/bvh/three (an optional peer dependency)
```

| Import | What it holds |
| --- | --- |
| `@pmndrs/bvh` | The core: builders, refits, the layout, the WGSL walk and its CPU port, `SceneTrees`, CPU queries, serialization |
| `@pmndrs/bvh/three` | three.js scenes in, `BvhTraceNodes` for TSL and WGSL, three-mesh-bvh's CPU API |
| `@pmndrs/bvh/worker` | The tinybvh worker pool |

## Quick start: vanilla WebGPU

<!-- snippet: readme-vanilla -->
```ts
import { IDENTITY_TRANSFORM, SceneTrees, traceModuleWgsl } from '@pmndrs/bvh';

const trees = await SceneTrees.create(device); // GPU H-PLOC trees with subgroups, tinybvh otherwise
const mesh = trees.addMesh({ positions, indices });
trees.addInstance({ mesh, objectToWorld: IDENTITY_TRANSFORM });

// Each frame: builds, swaps and refits, recorded before anything traces.
const encoder = device.createCommandEncoder();
trees.update(encoder);
device.queue.submit([encoder.finish()]);

// Your kernel calls bvhTraceClosest / bvhTraceAny; bind trees.buffers.nodes, .triangles, .topLevel at group 0.
const code = `${traceModuleWgsl()}\n/* ... let hit = bvhTraceClosest(BvhRay(origin, 0.0, direction, 1e30)); ... */`;
```

A device with `subgroups`, a full kernel and the batch kernel: [First build](https://bvh.docs.pmnd.rs/getting-started/first-build).

## Quick start: three.js

<!-- snippet: readme-three -->
```ts
import { SceneTrees } from '@pmndrs/bvh';
import { BvhTraceNodes, deviceFromRendererAsync, SceneTreesObject3DSync } from '@pmndrs/bvh/three';

const device = await deviceFromRendererAsync(renderer); // the renderer's own device
const trees = await SceneTrees.create(device);
const sync = new SceneTreesObject3DSync(trees, scene); // every mesh and instance under `scene`
const bvh = new BvhTraceNodes(renderer, trees);
const hit = bvh.traceClosest(origin, direction); // TSL: a struct node, hit.get('t') < 0 on a miss

// Each frame:
sync.update();
const encoder = device.createCommandEncoder();
trees.update(encoder);
device.queue.submit([encoder.finish()]);
bvh.update();
```

CPU raycasts as drop-ins for three-mesh-bvh (`computeBoundsTree`, `acceleratedRaycast`, `MeshBVH`):
[CPU raycast in three](https://bvh.docs.pmnd.rs/three/cpu-raycast). Moving from three-mesh-bvh:
[the mapping, and what's missing](https://bvh.docs.pmnd.rs/three/coming-from-three-mesh-bvh).

## Quick start: Babylon.js

Planned. A `@pmndrs/bvh/babylon` subpath (picking first, then the walk on Babylon's device) comes after 0.1. The CPU
queries work with any engine's vertex arrays today.

## Which builder

| Builder | What it is | Use it when | Needs |
| --- | --- | --- | --- |
| **tinybvh** (`createTinyBvhBuilder`) | tinybvh 1.9.1 in WebAssembly, on the CPU. Its default builder makes the best trees here. | You want the best trees and can wait (Sponza: 1.6-1.9 s). Run it in a worker. It is also the fallback for everything else. | WebAssembly SIMD |
| **GPU H-PLOC** (`GpuHplocBuilder`) | H-PLOC (Benthin et al., 2024) in WGSL, ported from tinybvh. The default GPU builder. | You need trees fast: at load, or to rebuild a mesh that changed a lot. Its trees trace about as fast as tinybvh's. | `subgroups` |
| **GPU LBVH** (`GpuLbvhBuilder`) | A linear BVH in WGSL, ported from HIPRT. | The fastest build when tree quality matters less. Its trees trace about 20% slower. | `subgroups` |
| **The chooser** (`createMeshLayoutBuilder`) | H-PLOC with subgroups, else tinybvh; meshes too big or refused go to tinybvh. | A list of meshes, trees back on the CPU, each saying who built it and why. | nothing |
| **`SceneTrees`** | H-PLOC trees first, tinybvh's swapped in from workers, the GPU top level, runtime changes. | A whole scene on the GPU. | nothing |
| **GPU refits** (`GpuBvhRefit`, `GpuMeshRefit`) | Keep a tree's shape, recompute its boxes on the GPU. | Instances that move every frame; skinned and morphed meshes. | `subgroups` |

## Requirements

- **WebGPU** for anything on the GPU. The CPU builder and queries need only WebAssembly SIMD (every current browser,
  Node 22+).
- **The `subgroups` feature** for the GPU builders and refits. Without it there is one fallback: tinybvh builds every
  tree on the CPU, and each result says so (`fallbackReason`). Chrome has it on the GPUs tested; Safari and Firefox
  don't expose it yet.
- **three 0.187** for `@pmndrs/bvh/three` (`>=0.187.0 <0.188.0`; `BvhTraceNodes` reads three's backend internals).
- WebGL2 is not a target.

## Headline numbers

Sponza (5,744,384 triangles in 132 meshes), RTX 3060 Laptop, headless Chrome 156 on Windows:

| What | Number |
| --- | --- |
| GPU H-PLOC build, whole scene | **46.6 ms** (median of the runs valid under the clock rule) |
| CPU tinybvh build, whole scene | **1.6-1.9 s** (one browser worker) |
| H-PLOC tree quality | SAH cost **1.032x** tinybvh's; closest-hit **0.95x**, any-hit **1.02x** of tinybvh's trace speed |
| Closest-hit rate on tinybvh trees | 104.3 Mrays/s, **1.20x** three-mesh-bvh's |
| GPU top-level refit, 10,000 moving instances | about **0.09 ms** |

These rows predate the 2026-11 defaults (they used instanced trees and `f32` nodes). On the same laptop (headless
Chrome 157, 2026-11-13), Sponza's rays traced 1.24-1.67x faster on the merged tree, and closest hits 1.04-1.22x faster
with quantized nodes.

Memory, Sponza, bytes beyond the app's own geometry: 355.5 MiB in the default layout (quantized nodes) against
three-mesh-bvh's 51.2 MiB, 225.5 MiB with indexed leaves as well (off by default). The GPU build's peak is capped at
about the scene plus 256 MiB (613 MiB on Sponza, Apple M5, indicative).

How each number was measured, and the rest: [Builds and tracing](https://bvh.docs.pmnd.rs/performance/builds-and-tracing)
and [Memory](https://bvh.docs.pmnd.rs/performance/memory). The RTX 3060 has run the 2026-11 trace benches (merged
static instances, the layout variants, the hit record, the any-hit order); `SceneTrees` memory and the cap on it, and
the Intel UHD, are still to come.

## Licences

The package is MIT. It contains or ports code from:

- **tinybvh** 1.9.1 by Jacco Bikker (MIT): compiled into the `.wasm`, and its `BuildHPLOC` ported to WGSL and TypeScript. Text: `licenses/tinybvh.txt`.
- **HIPRT** by AMD (MIT): the LBVH kernels and Morton codes, ported to WGSL. Text: `licenses/hiprt.txt`.
- **Emscripten's runtime**, linked into the `.wasm`: Emscripten's code and musl (MIT), dlmalloc (public domain), LLVM libc++ and compiler-rt (Apache 2.0 with LLVM Exceptions). Text: `licenses/emscripten-runtime.txt`.

The `.wasm` carries its licence texts in a `/*! */` comment that bundlers keep, and the build fails if a built file
loses them. Sources and commits are in [THIRD_PARTY.md](THIRD_PARTY.md). Only MIT or Apache-2.0 code is ported.

## Working on the package

- `pnpm lint`, `pnpm typecheck`, `pnpm test`, `pnpm build`.
- GPU pages in headless Chrome: `pnpm test:gpu`, `pnpm test:gpu:page trace-oracle` ([tests/gpu/README.md](tests/gpu/README.md)).
- Examples: `pnpm examples:dev`, `pnpm examples:build`, `pnpm examples:smoke` ([examples/README.md](examples/README.md)).
- Docs: `pnpm docs:snippets` copies the code samples (`docs/snippets/*.ts`, type-checked and tested) into the pages,
  `pnpm docs:api` builds the TypeDoc reference, `pnpm docs:site` builds the website with pmndrs/docs.
- Rebuild the `.wasm` with `pnpm bvh:wasm`, or check it with `pnpm bvh:wasm --check`.
- Contributor notes (design, divergences from the sources, the three-mesh-bvh audit, runbooks):
  [docs/internals/](docs/internals/README.md).

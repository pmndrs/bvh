# Contributor notes

These are the internal design notes, kept beside the docs site but not part of it: the site is the `.mdx` pages in
`docs/` (pmndrs/docs), and `tools/docs-site.ts` leaves this folder out when it builds. User-facing docs go in the
`.mdx` pages; reasons, measurements in full, and history go here.

| File | What it holds |
| --- | --- |
| `design/bvh-package-split.md` | The plan: decisions D1-D8, phases P0-P9, examples, docs, tasks BX.0-BX.20 |
| `design/bvh-framework.md` | The builder framework and its stages |
| `design/bvh-builders.md` | How tinybvh, the LBVH, H-PLOC and the refits work; rebuilding the `.wasm`; the build-peak cap |
| `design/trace.md` | The GPU layout, the WGSL walk, the watertight test, the oracle, merged static instances, the bench method |
| `design/indexed-leaves.md`, `design/compressed-nodes.md` | The two layout variants, with their measurements |
| `design/memory.md` | Every buffer, ours and three-mesh-bvh's, and the levers |
| `DIVERGENCES.md` | Every way the code differs from its sources, and why |
| `PARITY.md` | The audit against three-mesh-bvh 0.9.18, row by row |
| `three.md` | `@pmndrs/bvh/three` in full detail (the user pages are `docs/three/*.mdx`) |
| `WINDOWS-RUNBOOK.md` | The next pass on the RTX 3060 and the Intel UHD |
| `MIGRATING-FROM-DDGI.md`, `MIGRATING-FROM-REFLECTIONS-BRANCH.md` | Switching the known consumers to this package |

## Scope: the package ends at the hit

`@pmndrs/bvh` answers *what does this ray hit* and *what is near this shape*. It covers:
- building and refitting;
- scene trees and the walk, with masks, the triangle filter hook and the any-hit order;
- spatial queries.

A hit is the triangle, instance, barycentrics, face normal and a material number that the library carries but never reads.

What a hit *looks like* belongs to renderers built on top: Dennis's path tracer, and a shared tracing layer for it, reflections, shadows and ddgi's GI. That covers sampling, random numbers, BSDFs, lights, accumulation, texture lookups, and interpolating shading normals and UVs.

Physics uses shapecast, closest point and box queries, which stay here. When a consumer asks for something, check it against this line first. `examples/three/pathtracer` is a demo of the walk, not a feature.

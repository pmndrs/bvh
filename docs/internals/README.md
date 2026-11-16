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
| `design/clustered-bvh.md` | three-mesh-bvh's `ClusteredBVH` (PARITY 8.9): what it is, how it maps onto our layout, `clusterTraceScene` and its checks |
| `design/scene-files.md` | The two scene files: the snapshot and the movable scene (for `SceneTrees`), the scenarios each is for, and how merged static groups are baked |
| `DIVERGENCES.md` | Every way the code differs from its sources, and why |
| `PARITY.md` | The audit against three-mesh-bvh 0.9.18, row by row |
| `three.md` | `@pmndrs/bvh/three` in full detail (the user pages are `docs/three/*.mdx`) |
| `babylon.md` | `@pmndrs/bvh/babylon`: the P6 spike's findings (device, buffer binding, what went wrong), the decisions, the API gaps (the user page is `docs/engines/babylon.mdx`) |
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

## Publishing the docs

The site is the `.mdx` pages in `docs/` built by pmndrs/docs, with the TypeDoc reference at `/api`. It was first
deployed on 2026-11-15. The workflow is `.github/workflows/docs.yml` ("Build documentation and deploy to GitHub Pages"):

- On every push to `main` and every pull request that touches `docs/`, `src/`, `typedoc.json`, `tools/docs-site.ts` or
  the workflow, it checks the code samples (`pnpm docs:check`), builds the API reference (`pnpm docs:api`) and the site
  (`pnpm docs:site _site`). It only builds, to catch MDX and link errors; nothing is deployed.
- It **deploys only when someone runs it by hand**: Actions, "Build documentation and deploy to GitHub Pages", Run
  workflow, on `main`. The deploy job uses the `github-pages` environment.

What the owner has to do once:

1. **Pages settings** (repository Settings, Pages): set the source to **GitHub Actions**. Without this the deploy job
   fails at `actions/deploy-pages`.
2. **Custom domain** (the same page): enter `bvh.docs.pmnd.rs`, and turn on "Enforce HTTPS" once the certificate is
   issued. The workflow builds the site for that address (`NEXT_PUBLIC_URL`), and the README and the docs link to it.
3. **DNS** (the `pmnd.rs` zone, which the pmndrs organisation controls): a `CNAME` record for `bvh.docs` pointing at
   `pmndrs.github.io`. GitHub checks it before it serves the domain.
4. **Run the workflow** on `main` (above). To deploy on every push to `main` instead, change the `if:` of the upload
   step and the deploy job from `github.event_name == 'workflow_dispatch'` to also allow `push`.

Pages is on (source GitHub Actions, 2026-11-15), so until the domain is set the site is at `https://pmndrs.github.io/bvh/`.
The build reads that address from the Pages settings and sets `BASE_PATH` (`/bvh`) and `NEXT_PUBLIC_URL` from it, so
the same workflow builds for the custom domain once it is set, with no edit. A local build for `/bvh`, served under
`/bvh/`, loads every page, asset and the API reference. The README links to `bvh.docs.pmnd.rs`, which go nowhere until
the domain works.

To look at the site locally: `pnpm docs:api`, `pnpm docs:site`, then `npx -y serve docs-site`. `docs:site` needs network
access the first time (it runs `npx @pmndrs/docs@4`).

## Releasing to npm

Pushing to `main` never publishes. A release is the "Release to npm" workflow, run by hand on `main`. How to run it,
the one-time npm and GitHub setup, and what to do when it fails: [RELEASING.md](RELEASING.md).

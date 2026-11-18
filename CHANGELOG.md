# Changelog

The release workflow uses the section for the version it publishes as that GitHub Release's notes. A version with no
section here gets GitHub's generated notes instead. The longer story, by date, is in [HISTORY.md](HISTORY.md).

## 0.1.0

The first release on `latest`. A BVH for WebGPU that doesn't depend on any engine. GPU and CPU builders write one
shared GPU layout, and a WGSL walk and its CPU port trace it.

- **Builders:** tinybvh 1.9.1 in WebAssembly on the CPU, GPU H-PLOC and GPU LBVH in WGSL, and a chooser that picks one
  and says why. Without the `subgroups` feature, tinybvh builds every tree.
- **Scenes:** `SceneTrees` keeps a whole scene on one device. GPU trees come first and tinybvh's are swapped in from
  a worker pool. The top level is built on the GPU, and meshes and instances can come and go. GPU refits handle moving
  instances and skinned or morphed meshes. Big meshes can be cut into clusters. An update with nothing changed costs a
  few microseconds on the CPU, and a changed mesh uploads only its own record.
- **Tracing and queries:** the WGSL walk handles closest and any hits, with masks and a triangle filter, and ray
  batches. When a ray reaches two boxes at exactly the same distance, the closest-hit walk takes the right one first
  unless the ray starts inside both (tinybvh always takes the left): rays from outside a closed model were up to 4.4x
  faster on the RTX 3060, the rest level. GPU closest point, sphere and box tests and your own shapecast. Signed
  distance fields filled on the GPU; the default sign is a ray vote that is right for any closed mesh, and
  `sign: 'first-hit'` is three-mesh-bvh's faster one-ray sign for closed, outward-wound meshes. On the RTX 3060 our SDF
  is 1.2-1.3x three-mesh-bvh's time on a 100k mesh with the vote and the same with first-hit.
- **CPU queries:** raycast, shapecast, `bvhcast`, closest point and frustum culling, with three-mesh-bvh's contracts,
  and trees over points and lines.
- **Serialization and baking:** build trees or whole scenes once, in Node or at build time, and load them later with
  no build. A scene file is either a snapshot (the top level baked too, loads as a copy) or movable (mesh trees, the
  merged static groups' trees and the instances; it loads into `SceneTrees` with no tree built, and its instances can
  move). `pnpm examples:bake --scene [--movable]` bakes either.
- **three.js** (`@pmndrs/bvh/three`, three 0.187): scenes from `Object3D`, GPU skinning, `BvhTraceNodes` for TSL, and
  three-mesh-bvh's CPU API as drop-ins. Against three-mesh-bvh 0.9.18: 45 features matched, 2 partly, 5 left out on
  purpose (`docs/internals/PARITY.md`).
- **Babylon.js** (`@pmndrs/bvh/babylon`, Babylon 9): picking that returns `scene.pick`'s result, scenes from meshes,
  and the walk in a `ComputeShader`.
- **Defaults:** quantized nodes, merged static instances, triangle records in the leaves, and a 256 MiB cap on the GPU
  build's peak memory. The memory levers that measured as wins are on.
- **Startup:** each walk pipeline takes about 0.15-0.25 s to compile the first time on Windows (the SDF kernel 0.7 s,
  the GPU builder 0.3-1.4 s), and about 10 ms once Chrome has cached it. Create kernels early and with the async calls
  (Builds and tracing, "Shader compile time").
- **Checked on:** an RTX 3060 and an Intel UHD on Windows, an Apple M5, and SwiftShader in CI. Every GPU result is
  checked against the CPU walk.

Docs: https://pmndrs.github.io/bvh/

## 0.1.0-alpha.0

A first publish, by hand, to claim the package name and check the release path. Same code as 0.1.0's start, on the
`alpha` tag.

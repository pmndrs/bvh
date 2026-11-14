# `@pmndrs/bvh/babylon`

> The user page is [docs/engines/babylon.mdx](../engines/babylon.mdx). This file keeps the spike's findings and the decisions behind the subpath. Scope per D5: "it works", not polish.

## The spike (P6, 2026-11-14)

**It worked.** Babylon 9.35.0, `WebGPUEngine`, headless Chrome 157 on SwiftShader (Windows):

- **Device:** `engine._device` (declared `@internal` in Babylon's `.d.ts`, so TypeScript sees it). `WebGPUEngine.CreateAsync(canvas, { enableAllFeatures: true, setMaximumLimits: true })` gives the adapter's features (`subgroups`, so the GPU first trees ran: "gpu (H-PLOC) x5") and limits; by default Babylon asks for neither.
- **Buffers:** the `ComputeShader` path, not a raw pass beside Babylon. `ComputeShader.setStorageBuffer(name, buffer)` takes a Babylon `StorageBuffer` **or a `DataBuffer`**, and on WebGPU a `DataBuffer` is `WebGPUDataBuffer(gpuBuffer, capacity)`, a wrapper that owns nothing. So `SceneTrees.buffers` bind with no copy (`gpuBufferToDataBuffer`, `bindTraceBuffers`), and Babylon never destroys them. Babylon's own `StorageBuffer` (params, hits) and a storage `RawTexture` (`Constants.TEXTURE_CREATIONFLAG_STORAGE`) sit in the same bind group.
- **Order:** our `trees.update(encoder)` is submitted on the device before Babylon's frame; the dispatch is recorded in `scene.onBeforeRenderObservable`. Babylon's compute pass and render passes share its frame submit.
- **Picking:** `CpuSceneBvh` over tinybvh trees, returning a real `PickingInfo`. Node tests compare it with `scene.pickWithRay` on a `NullEngine` (random rays, mirrored and non-uniform instances, `InstancedMesh`es, ray length, predicates, `pick(x, y)`).

### What it cost (Babylon 9.35)

1. **WGSL comments with a semicolon break the shader.** Babylon's WGSL preprocessor splits the code at `;` including inside `//` comments. The walk's comments hold some ("distance along the ray; negative means no hit"), so the struct came out cut and Tint said "expected ':' for struct member". `wgslForBabylon` removes comments; `createTraceComputeShader` applies it. A cleaner fix would be removing the semicolons from the walk's comments (src/trace owns that); not asked, so noted as an API gap below.
2. **`engine.createComputeContext is not a function`** until `Engines/WebGPU/Extensions/engine.computeShader.js` is imported (a side-effect module; the deep-import style leaves it out). `src/babylon/compute.ts` imports it.
3. **A foreground `Layer` hid the scene's meshes** on the WebGPU engine until `layer.applyPostProcess = true` (drawn in the camera's draw step instead of after the post-process step). Cause not found; the shadow image itself was right in both cases. Recorded in the example.
4. **`scene.createPickingRay` takes canvas pixels**, which hardware scaling divides (`engine.getHardwareScalingLevel()`); comparing rays at a scaled render size needs the multiply.
5. **`InstancedMesh` must be imported** for `createInstance` in a deep-import build ("needs to be imported before as it contains a side-effect"). Not our import to make: an app that has instances already imports it.
6. **Babylon's picking ray starts on the near plane** (`minZ`), so its `distance` is from there; the GPU kernel starts at the camera. The example compares points, not distances.
7. **`bu`/`bv` are the weights of vertices 0 and 1** (`1 - u - v` and `u` in the core's terms), not 1 and 2.
8. **A mesh with no material is picked by its bounding box** in Babylon. The picker always reads triangles.

### Running the examples on SwiftShader (Windows)

`BVH_CHROME_FLAGS="--no-sandbox --use-webgpu-adapter=swiftshader --use-angle=swiftshader --enable-unsafe-swiftshader"` (CI's) gave **no WebGPU adapter** on Chrome 157 / Windows 11 (`requestAdapter()` null). Dropping `--use-angle=swiftshader` worked and reported `architecture=swiftshader`. The Linux CI job is unchanged by this. WebGL pages can't be run by the runner on Windows software-only (its preflight needs a WebGPU adapter), so `babylon/picking?engine=webgl` was run once with the runner's own Chrome and DevTools modules and `--disable-gpu --enable-unsafe-swiftshader`; it passed (the same report as WebGPU, 96 rays, 0 disagreements).

## The API (decisions)

- **Deep imports** (`@babylonjs/core/...js`), never the barrel, so the package tree-shakes Babylon and the examples' chunk is only what they use.
- **A picker, not a sync.** three has `SceneTreesObject3DSync`, whose job is keeping `SceneTrees` in step with a mutable scene graph, GPU skinning included. Babylon has `bvhSceneFromBabylon` (once), `BabylonBvhPicker.update()` for the CPU side, and the example's loop for `trees.setTransform`. A sync is the first thing to add if Babylon grows beyond "it works".
- **Instance mask:** bit 0 every instance, bit 1 pickable. Babylon's default pick takes enabled, visible and pickable meshes; shadow rays and plain `bvhTrace*` (all ones) see unpickable ones too, which is what a shadow caster wants.
- **Peer range** `>=9.35.0 <10.0.0`, `devDependency` 9.35.0 (the compare suite's, so `compare/RESULTS.md` still describes the installed version). 9.36.0 was current on 2026-11-14; not adopted, since `compare/` records 9.35.0 in its tables.
- **ESLint:** `src/babylon` may import `@babylonjs/*` and nothing else engine-like; the core and `src/three` may not import it or the folder (`**/babylon/**`); `scripts/verify-artifacts.mjs` also checks the built `dist/` (only `dist/babylon` names `@babylonjs/*`, only `dist/three` names `three`).

## API gaps (for the owners of the other folders)

- **src/trace:** the walk's WGSL comments contain semicolons, which Babylon's preprocessor splits at (`wgslForBabylon` works around it). Comments without `;` would let `traceModuleWgsl()` go to a `ComputeShader` unmodified.
- **src/scene:** a `SceneTrees` needs `instanceIndex` passed explicitly for `hit.instance` to equal a position in `bvhSceneFromBabylon(...).sources`; it does (`addInstance({ instanceIndex })`), but there is no bulk "add a scene" call. three's sync is that call.

## Not done

- Thin instances, skinned and morphed meshes, submesh ranges and material groups, lines.
- A node-material or `ShaderMaterial` binding (three's `BvhTraceNodes`).
- Real-GPU runs (Windows, RTX 3060 and Intel UHD; Apple M5): the scheduled clean run. Everything above was run on SwiftShader only. The picks-per-second numbers on the examples are not meaningful there.

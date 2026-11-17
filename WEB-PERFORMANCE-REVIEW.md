# BVH performance and code review

November 17, 2026 · Reviewed `23af802` · [Read in Pages](https://chatgpt.com/space/page_1353ece616bc81919a4ea527a0f00dd9)

The first change I’d make is to the GPU top-level table upload. With room for 4,096 meshes, changing one mesh writes 128 KiB for a 32-byte record. The next opportunities are JavaScript frame bookkeeping and CPU ray queries. The H-PLOC shader experiment has a less certain payoff.

## BVH1  Upload the changed mesh records

`GpuTopLevel.setMeshTree` marks the whole mesh table dirty. Its next `update` writes the capacity-sized mirror. The instance table in the same class already tracks a dirty range. The GPU path also writes its four gather parameters on every update, including idle frames, and writes them again when it swaps in a shape. See `src/trace/top-level-gpu.ts` lines 514, 609, 614, 679, 815 and 864.

Track the lowest and highest modified mesh slots. Upload the full initialized mirror on first use and after buffer growth, and a contiguous slice for sparse changes. Write the gather parameters when the record count or either capacity changes. Count bytes and queue calls for a single edited mesh, scattered edits, dense placement, deletion and growth. Compare gathered records byte for byte with the CPU top level. The 128 KiB figure is a byte count derived from the layout, not a frame-time estimate.

## BVH2  Scale frame bookkeeping with scene changes

`SceneTrees.update` checks first-tree readiness even with no waiters and walks every mesh looking for CPU poses. The GPU top-level path also scans every placed tree to find the maximum reachable depth. A changed deform box scans every instance to find the users of that mesh. See `src/scene/scene-trees.ts` lines 1242, 2302, 2934, 3012 and 3112.

On Node 24.13.1 with the repository’s fake device and CPU top level, settled updates took 0.0036–0.0061 ms for 256 meshes and 0.0459–0.0464 ms for 4,096. Each mesh had one triangle, one instance and an app-supplied tree. Each result is the median of 11 blocks of 100 update-and-submit calls after 30 warm-ups, repeated twice per size. This isolates host bookkeeping. The absolute cost in this fixture is small.

Start with the waiter guard and a set of meshes with pending CPU poses. Retain a pending pose while its layout is unavailable. For scenes with animated crowds, measure the instance scan separately and consider a mesh-to-instance index. Cache maximum depth only with invalidation for placement, replacement, removal and reachability changes. Check the traversal stack limit when a deeper tree arrives.

## BVH3  Count CPU ray crossings directly

`CpuMeshBvh.rayCrossings` collects an intersection object for every crossing, then walks the array to compute two integers. The CPU SDF and inside tests repeat this operation across sample points and directions. A count-and-winding accumulator can use the existing watertight intersection and face rules while avoiding the result list. Keep scratch state local to a query or safe for nested callbacks. See `src/query/cpu-mesh-bvh.ts` lines 735 and 772 and `src/trace/sdf.ts` line 160.

Benchmark CPU SDF generation on a closed mesh and overlapping shells. Compare per-ray counts and signed sums on shared edges, mirrored geometry and inconsistent winding. Separately profile the instance transform at `src/trace/cpu-traversal.ts` line 402 before rewriting its temporary arrays. The Windows Node comparison on the 100k-triangle knot recorded 236.7k first-hit outside rays/s here versus 318.0k for three-mesh-bvh CENTER. That comparison establishes headroom without attributing the gap to allocation.

## BVH4  Test the cost of the H PLOC emit stage

Every retained node in a multi-triangle mesh computes the same quantization frame from that mesh’s root in `hplocEmit`. Time the emit stage, then try sharing the frame within a tile or deriving it in an earlier stage with a valid dependency. The early returns for padded lanes and discarded nodes matter if adding a workgroup barrier. In the same function, the parent store has the same guard twice, and the parent lookup precedes both guards. Move the lookup under one guard and inspect the emitted code. See `src/gpu-hploc-wgsl.ts` lines 1058, 1201 and 1227.

The existing RTX Sponza result already favors quantized emit, 35.1 ms versus 41.6 ms for the full build with the separate packing pass. Use that 35.1 ms path as the baseline. Compare both a large mesh and many small meshes, time the full build as well as emit, and require byte-identical nodes and parents across indexed and record leaves. The paper distinguishes binary construction time from total build time. The package also sorts, packs and exposes buffers to a web application. [H-PLOC paper](https://gpuopen.com/download/HPLOC.pdf)

## BVH5  Show the cost from first frame to steady state

Extend `examples/vanilla/fast-first`, which already has per-mesh build events and controls to hold refinements. Add a repeatable camera and a small readout for first traceable frame, refined frame, steady GPU trace time and peak memory. A reader can then see whether refinement pays back during their session. For a fixed ray workload, divide the extra refinement cost by the trace time saved per frame and label which costs overlap.

Keep one runnable route near the top of the README for each audience: raw WebGPU, three.js and CPU picking. The vanilla quick start currently ends with a commented-out shader call. Link directly to the working dispatch. Move historical decisions behind the guide. For example, `docs/internals/design/bvh-builders.md` line 139 still describes quantized emit profiling as pending, while `compressed-nodes.md` line 169 records the result. Similarly, the 4.4× traversal gain belongs to the furnished-room outside-ray workload.

## Verification

Reviewed `main` at `23af802`. All 873 Vitest tests passed. On Windows, the NVIDIA Ampere adapter passed `bvh-top-level-gpu` and `trace-inline`. The former checked 400 instances before and after moves, hides and removals. The frame-bookkeeping measurements above were fresh Node measurements. GPU build and comparison numbers come from `docs/internals/design/compressed-nodes.md`, `docs/internals/WINDOWS-RUNBOOK.md` §14 and `bench/results/compare-windows.md`. Proposed changes still need paired browser measurements.

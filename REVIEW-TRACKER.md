# BVH and downsampler review tracker

November 17, 2026. Both repositories have a local branch named `codex/web-performance-review`. DS1, BVH1 and DS2 have implementation results below. The other seven follow-ups remain open. Each review contains the code locations, baseline and proposed verification.

| Library | Reviewed revision | Review page | Local review |
| --- | --- | --- | --- |
| BVH | `main` at `23af802` | [BVH performance and code review](https://chatgpt.com/space/page_1353ece616bc81919a4ea527a0f00dd9) | [BVH Markdown](WEB-PERFORMANCE-REVIEW.md) |
| Downsampler | `windows-3` at `d8368d0` | [Downsampler performance and code review](https://chatgpt.com/space/page_31c5c354d25c81918908d3903f124634) | [Downsampler Markdown](../downsampler-web-review/WEB-PERFORMANCE-REVIEW.md) |

## Implementation queue

| ID | Status | Change | Result or evidence to close it |
| --- | --- | --- | --- |
| DS1 | Implemented `5c59385` | Preserve full and partial plans on failed replacement | Full retry went from one validation error and six stale mips to zero errors and six correct mips. Both caches retain the original plan on failure. |
| BVH1 | Implemented `94a7220` | Upload dirty mesh slots and changed gather parameters | One edit at 4,096-slot capacity went from 131,088 bytes in two writes to 32 bytes in one. Idle frames went from one write to zero. |
| DS2 | Implemented `db4c381` | Reuse bindings and texture views across moving dirty rectangles | In the original 64-stroke test, buffer creations fell from 64 to 2 and bind groups from 256 to 8. Moving-stroke CPU p95 fell from 70 to 20 µs on NVIDIA and from 75 to 20–25 µs on Intel. |
| BVH2 | Measured | Avoid full mesh scans on settled frames | Compare 256 and 4,096 settled meshes, then sparse CPU poses and many instances. Preserve waiter, pose and stack-limit behavior. |
| BVH3 | Open | Accumulate CPU SDF crossings during traversal | Compare signed counts and higher-level results. Profile CPU time and allocations on closed and overlapping meshes. |
| DS3 | Experiment | Warm up the requested compute or render pipeline asynchronously | Compare cold and warm frame gaps, pending-request deduplication and compile events across adapters. |
| BVH4 | Experiment | Share the quantization frame during H-PLOC emit | Compare full build and emit times on small and large meshes. Verify output words against the existing quantize pass. |
| DS4 | Awaiting measurement | Revisit capped halo defaults after the M5 result | Pair the same odd and even sizes, formats and policies with existing Windows results. |
| BVH5 | Open | Extend Fast first and shorten the README routes | Show time to trace, time to refine, steady trace cost and peak memory for one repeatable workload. |
| DS5 | Open | Extend the brush and Hi-Z examples and reconcile install guidance | Reproduce a brush replay without timing the reference readback. Show actual occlusion decisions at odd sizes. |

## Verification on Windows

### DS1 · Failed plan replacement

On Chrome 148.0.7778.97 with NVIDIA Ampere, a four-byte scratch buffer forced a replacement to fail. After editing a 17×15 patch on a 65×47 RGBA8 texture, the test retried using the original options and compared all six generated mip levels byte for byte against a fresh full recompute.

| Retry after rejection | Before · `2efe7d3` | After · `5c59385` |
| --- | --- | --- |
| Full chain | 1 queue validation error, 6/6 levels stale | 0 errors, 0/6 levels mismatched |
| Partial update | Rebuilt the evicted plan and produced correct pixels | Retained the original plan and produced correct pixels |

Build, TypeScript checks, 129 core tests and all 10 cases in `downsampler/options` passed. The browser regression also checks that both caches return the original prepared batches after a failed replacement.

### BVH1 · Mesh table uploads

The probe counted calls to `GPUQueue.writeBuffer` on the mesh table and its gather-parameter buffer on Chrome 148 / NVIDIA Ampere. The table had 4,096 slots. Byte counts describe uploads and exclude other scene buffers.

| One update | Before · `ac03b7a` (writes / bytes) | After · `94a7220` (writes / bytes) |
| --- | ---: | ---: |
| Initialize capacity | 2 / 131,088 | 2 / 131,088 |
| First shape swap | 2 / 32 | 1 / 16 |
| Idle | 1 / 16 | 0 / 0 |
| Edit slot 1,000 | 2 / 131,088 | 1 / 32 |
| Remove slot 1,000 | 2 / 131,088 | 1 / 32 |
| Edit all 4,096 slots | 2 / 131,088 | 1 / 131,072 |
| Grow to 8,192 slots | 2 / 262,160 | 2 / 262,160 |

Sparse writes started at byte 32,000. Multiple edits share one span from the first changed slot through the last, so widely separated edits can still upload most of the table. Build and package checks, lint, type checks and all 874 unit tests passed. In the two passing NVIDIA browser checks, `bvh-top-level-gpu` compared 400 instances against the CPU, and `scene-trees-runtime` ran eight update rounds.

### DS2 · Moving brush strokes

The optimized storage path in `encode` and `encodePass` shares views and bind groups and gives each update immutable uniform slots. Its recent rectangles have a separate cache from public `prepare()` batches. Buffer output, render and compatibility paths retain their existing bindings. The private layout can compile a separate pipeline on first interactive use, even after warming public `prepare()`.

**Original review workload.** After one warm-up, 64 consecutive one-pixel moves of a 32×32 brush on a 1920×1080 RGBA8 texture created **64 → 2 buffers** and **256 → 8 bind groups**, on both repetitions. Both versions created zero compute pipelines. A fixed brush created zero of each. This NVIDIA test requested the library’s recommended device limits and submitted each four-dispatch stroke separately.

**Paired benchmark.** Chrome 148 on Windows, same 1080p texture and brush size, browser-default device limits (five dispatches). Each round used 64 coordinates, with build and case order reversed every other round. CPU p50/p95 cover 320 individual recordings from encoder creation through `finish()` on an uninstrumented device. GPU p50/p95 summarize five separate pass timestamps, each divided by 64 strokes. Timer resolution was 5 µs on the CPU. Each cell is **before → after**, in **µs per stroke**, comparing `5c59385` to `db4c381`.

| Adapter | Brush | CPU p50 | CPU p95 | GPU p50 | GPU p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| NVIDIA Ampere | Moving | 50 → 10 | 70 → 20 | 20.86 → 19.30 | 20.94 → 19.31 |
| NVIDIA Ampere | Fixed | 5 → 5 | 15 → 15 | 19.01 → 18.74 | 19.25 → 18.80 |
| Intel Gen12 LP | Moving | 45 → 10 | 75 → 20 | 22.78 → 21.61 | 24.31 → 21.67 |
| Intel Gen12 LP | Fixed | 5 → 5 | 10 → 10 | 19.58 → 21.08 | 21.53 → 23.18 |

A separate instrumented pass on each adapter counted **319 → 1 buffers**, **1,595 → 5 bind groups**, **4,785 → 0 texture views**, and **0 → 0 pipelines** over 320 moving strokes. The first coordinate was still cached. Fixed strokes created zero of all four. Moving strokes requested **408,320 → 327,680 bytes** of new buffers during that interval. The implementation allocates in chunks and leaves old chunks to garbage collection while recorded commands can reference them. These are cumulative allocation requests within the measured window, excluding warm-up. They do not measure peak GPU memory.

A second independent Intel run reproduced the tradeoff. Moving CPU p95 was 75 → 25 µs and moving GPU p50 was 23.25 → 21.98 µs. Fixed-brush GPU p50 was 19.58 → 20.18 µs and p95 was 21.55 → 23.35 µs. Fixed CPU times and allocation counts matched the first run. Across both runs the fixed-brush GPU median increased by 0.60–1.50 µs.

**Correctness.** Build, TypeScript checks and all 129 core tests passed. On NVIDIA, the quick region matrix passed 505 cases in 63 groups, and the quick halo matrix passed 156 cases in seven groups. The final `region/options` and `downsampler/options` rerun passed 13 cases. Intel passed 36 targeted options, layers and tail cases. NVIDIA in WebGPU compatibility mode passed all 14 selected options and layer cases available to it.

The new storage regression recorded 514 updates across two layers using both `encode()` and `encodePass()`, allowed the queue to settle, then submitted the two command buffers in reverse order. Every mip matched a clean recompute before the test reused an earlier public prepared plan. Both core adapters passed. The compatibility checks exercised the existing fallback.

**Reproduce the paired measurement.** Build the downsampler at both commits, then run the following from the candidate worktree, substituting the path of the baseline build. Use `--power=low` for Intel.

```powershell
$baselineDist = 'C:\path\to\baseline\dist'
node results/probes/ds2-region.mjs "--baseline=$baselineDist" --candidate=dist --power=high
```

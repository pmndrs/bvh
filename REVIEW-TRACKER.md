# BVH and downsampler review tracker

November 17, 2026. The downsampler Windows review is ready on `codex/web-performance-review`, including `windows-3` at `0210cd1`. DS1, DS2, DS3 and DS5 are implemented. DS4 is closed with paired Windows data and two M5 runs. Capped halo stays opt-in. BVH1 and BVH2 are implemented. BVH3, BVH4 and BVH5 remain open.

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
| BVH2 | Implemented `35451af` | Avoid full mesh scans on settled frames | At 4,096 meshes on NVIDIA, settled CPU time inside `SceneTrees.update()` fell from 150 µs to roughly 5 µs (one timer tick). Posing one mesh with changed bounds fell from 245 to 45 µs. |
| BVH3 | Open | Accumulate CPU SDF crossings during traversal | Compare signed counts and higher-level results. Profile CPU time and allocations on closed and overlapping meshes. |
| DS3 | Implemented `5702f5b`, `a7f3060` | Warm actual encode pipelines asynchronously | Fresh RTX storage submit-to-completion fell from 128.0 to 3.2 ms after 70.5 ms of prior warmup. |
| BVH4 | Experiment | Share the quantization frame during H-PLOC emit | Compare full build and emit times on small and large meshes. Verify output words against the existing quantize pass. |
| DS4 | Closed `221aa42`, `95f13a8` | Measure capped halo across GPUs | The M5 missed the 5% rule and regressed at 1024² RGBA8. Halo stays opt-in. |
| BVH5 | Open | Extend Fast first and shorten the README routes | Show time to trace, time to refine, steady trace cost and peak memory for one repeatable workload. |
| DS5 | Implemented `f80f5a9` | Demonstrate moving-brush performance and conservative occlusion | Paired replay, timing fallback and odd-size Hi-Z decisions have automated example checks. Setup starts from a checkout. |

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

### Downsampler Mac handoff · DS3, DS4 and DS5

The completed review branch is `c9b90f7`. It includes the Apple M5 Pro size rule and both Windows verification runs, through `srgb-store` at `b6e97c5`. The final runtime is `ff785d9`. [Downsampler review](../downsampler-web-review/WEB-PERFORMANCE-REVIEW.md) · [Mac checkout and acceptance](../downsampler-web-review/docs/M5-HANDOFF-SRGB.md)

| Fresh Chrome 157 profile, storage | First submit-to-done, sync → warmed | Prior warmup |
| --- | ---: | ---: |
| RTX 3060 Laptop | 128.0 → 3.2 ms | 70.5 ms |
| Intel UHD | 128.1 → 4.1 ms | 73.7 ms |
| SwiftShader | 414.8 → 19.2 ms | 365.4 ms |

The A/B used one build. All 972 mip readbacks matched over 108 page runs. The first warmed encode created zero pipelines. See the [storage, buffer, sRGB render and reload measurements](../downsampler-web-review/results/windows/ds3/summary.md).

The M5 exact-policy halo/plain median was 0.98–0.99, with a reproducible 3–4% loss at 1024² RGBA8. Capped halo stays opt-in. The brush example replays 40 strokes and checks the resulting chain outside the timed region. Hi-Z checks five candidates against full-resolution depth, including thin occluders.

The combined runtime at `ff785d9` passed typecheck, 161 core tests, 21 targeted cases per hardware adapter, eight on SwiftShader and seven in RTX compatibility mode. The full RTX suite ran 35,840 cases across 3,239 groups. Intel quick ran 21,791 cases across 1,864 groups. Only the two historical FSR f16 fixtures failed (five sizes on RTX, two on Intel). All 21,074 RTX compatibility cases passed. All 24 example pages and presets passed on both hardware adapters. A clean bundle clone produced a 56-file installable tarball, and all five export paths passed fresh consumer checks. [Windows matrix and package results](../downsampler-web-review/WEB-PERFORMANCE-REVIEW.md#windows-verification)

### BVH2 · Settled frames and sparse CPU poses

Compared `f9e240f` (includes BVH1) with `35451af`. In a settled scene, `update()` skips the first-tree readiness scan when nobody is waiting, visits only queued CPU poses, and reuses the maximum reachable mesh depth for the GPU stack check. Placing or removing a tree, or adding its first or removing its last user, invalidates that depth. A changed mesh box updates the shared mesh record once. An unchanged box still allows the new triangle or vertex data to reach the GPU.

**Chrome 157 · Windows · NVIDIA Ampere · GPU top level.** Each registered mesh had its own app-supplied one-triangle tree and one instance. In the sparse case, one mesh alternated between two Z positions, changing its root box on every frame. Each row includes 1,500 `SceneTrees.update(encoder)` calls after 300 warm-ups. The browser drained the queue between frames. `setPoseCpu`, encoder creation, queue submission and GPU execution were outside the clock. Values below are **µs per call** across five blocks of 300, rounded to the browser’s 5 µs timer resolution.

| Meshes | Frame | CPU p50 before → after | CPU p95 before → after |
| ---: | --- | ---: | ---: |
| 256 | Settled | 15 → ~5 | 25 → 10 |
| 4,096 | Settled | 150 → ~5 | 190 → 10 |
| 256 | One changed mesh box | 80 → 50 | 260 → 110 |
| 4,096 | One changed mesh box | 245 → 45 | 410 → 85 |

The optimized settled median reached one timer tick at both sizes. A GPU ray hit instance 0 at distance 2 in the settled fixture and 1.5 after posing it. Readback confirmed all 256 or 4,096 instances remained present. The runner measured the baseline first, then the candidate.

**CPU top-level fallback · Node 24.13.1 · repository fake device.** The fixture used one app-supplied triangle and one instance per mesh. The sparse cases posed one mesh every update. Each revision ran twice, with the run order reversed the second time. Each cell gives the range of the two run medians in **µs per update**. After 30 warm-ups, each median used 11 block means (100 frames per settled block, five per pose block). This clock includes the public `setPoseCpu()` call when applicable, encoder creation, update and fake submission.

| Meshes | Frame | Baseline p50 | BVH2 p50 |
| ---: | --- | ---: | ---: |
| 256 | Settled | 4.80–5.59 | 2.13–2.50 |
| 4,096 | Settled | 49.10–51.42 | 2.32–2.53 |
| 256 | One pose, same root box | 421–444 | 10.8–11.4 |
| 4,096 | One pose, same root box | 6,157–6,179 | 11.6–12.2 |
| 256 | One pose, translated root box | 405–429 | 408–417 |
| 4,096 | One pose, translated root box | 6,203–6,222 | 6,145–6,721 |

Every pose run verified 55 writes during the timed blocks. Moving a vertex inside the existing root box kept the CPU top-level bytes identical while the mesh refit ran. Translating the box changed those bytes and still required rebuilding the CPU top level. At 4,096 meshes, the translated-box p95 of the block means increased from 6.69–6.84 ms to 7.18–7.77 ms. This fallback showed no consistent median improvement for a translated box.

**Correctness.** All 879 unit tests, lint, type checks, build and package verification passed. Five new regressions cover settled map scans, unchanged bounds on both top-level paths, depth changes and slot reuse, and sparse/dense CPU poses. The CPU top level traced two instances at their new positions after a changed box. The GPU path issued a mesh-table write and zero instance-table writes for that change. An interior-vertex change updated ray hits through both instances with no top-level upload. The existing waiter and deferred-layout tests also passed.

Three NVIDIA browser checks passed with zero ray disagreements in `bvh-top-level-gpu` (400 instances), `scene-trees-runtime` (eight mutation rounds), and `scene-trees-deforming` (GPU/CPU pose switches across six record/indexed configurations, including replacement trees and trims). The GPU refit still runs over its top-level records when a box moves. On devices supporting GPU deformation, polling and refit scheduling still visit the registered deforming meshes.

**Reproduce.** Use the same benchmark files in both source checkouts. Run the [CPU fixture](bench/bvh2-bookkeeping.ts) from each root with Node 24. For the [browser fixture](bench/bvh2-browser.js), serve the root with Vite and the isolation headers below. Open `/README.md` on the local server and run the JavaScript snippet in Chrome DevTools. The browser fixture returns pooled and per-block statistics in milliseconds.

```powershell
node --experimental-transform-types --no-warnings bench/bvh2-bookkeeping.ts
node --input-type=module -e "import {createServer} from 'vite'; const server = await createServer({server:{headers:{'Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp'}}}); await server.listen(); server.printUrls()"
```

```js
const { runBvh2BrowserBenchmark } = await import('/bench/bvh2-browser.js');
await runBvh2BrowserBenchmark();
```

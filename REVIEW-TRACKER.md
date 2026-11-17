# BVH and downsampler review tracker

November 17, 2026. Both repositories have a local branch named `codex/web-performance-review`. The suggested changes are open. Start with DS1 for correctness, then BVH1 and DS2 for performance. Each review contains the code locations, baseline and proposed verification.

| Library | Reviewed revision | Review page | Local review |
| --- | --- | --- | --- |
| BVH | `main` at `23af802` | [BVH performance and code review](https://chatgpt.com/space/page_1353ece616bc81919a4ea527a0f00dd9) | [BVH Markdown](WEB-PERFORMANCE-REVIEW.md) |
| Downsampler | `windows-3` at `d8368d0` | [Downsampler performance and code review](https://chatgpt.com/space/page_31c5c354d25c81918908d3903f124634) | [Downsampler Markdown](../downsampler-web-review/WEB-PERFORMANCE-REVIEW.md) |

## Implementation queue

| ID | Status | Change | Evidence to close it |
| --- | --- | --- | --- |
| DS1 | Reproduced | Publish a replacement plan before retiring the previous plan | Reject an undersized scratch buffer, retry the original options and compare pixels with a clean chain. No GPU validation errors. |
| BVH1 | Open | Upload dirty mesh slots and changed gather parameters | Record queue calls and bytes for idle, sparse and dense updates. Match GPU records through growth and removal. |
| DS2 | Measured | Reuse resources across moving dirty rectangles | Reproduce the 64-update brush case. Reduce buffer and bind-group creation, preserve every mip with multiple strokes per submission and compare CPU p95. |
| BVH2 | Measured | Avoid full mesh scans on settled frames | Compare 256 and 4,096 settled meshes, then sparse CPU poses and many instances. Preserve waiter, pose and stack-limit behavior. |
| BVH3 | Open | Accumulate CPU SDF crossings during traversal | Compare signed counts and higher-level results. Profile CPU time and allocations on closed and overlapping meshes. |
| DS3 | Experiment | Warm up the requested compute or render pipeline asynchronously | Compare cold and warm frame gaps, pending-request deduplication and compile events across adapters. |
| BVH4 | Experiment | Share the quantization frame during H-PLOC emit | Compare full build and emit times on small and large meshes. Verify output words against the existing quantize pass. |
| DS4 | Awaiting measurement | Revisit capped halo defaults after the M5 result | Pair the same odd and even sizes, formats and policies with existing Windows results. |
| BVH5 | Open | Extend Fast first and shorten the README routes | Show time to trace, time to refine, steady trace cost and peak memory for one repeatable workload. |
| DS5 | Open | Extend the brush and Hi-Z examples and reconcile install guidance | Reproduce a brush replay without timing the reference readback. Show actual occlusion decisions at odd sizes. |

## Verification record

- BVH passed 873 unit tests plus the NVIDIA top-level and inline-trace browser checks. Node bookkeeping timings use a stand-in device.
- Downsampler's pinned checkout built and passed 129 core tests. The active checkout passed 130 with its local harness edits. The cache failure and brush allocation counts were observed in Chrome 148 on NVIDIA Ampere.
- The separate `halo-batches` checkout supplied historical context. The library under review already includes opt-in halo on `windows-3`. New GPU checks covered NVIDIA on Windows. Apple and Intel comparisons in the reviews come from the repositories' reports.

For each completed item, record the implementation commit, adapter and browser where applicable, baseline, result and correctness check in its row.

# Windows runbook: RTX 3060 Laptop and Intel UHD

The next pass on the Windows laptop. Run the steps in order; each says what its result decides. Nothing from 2026-11-12 (indexed leaves, quantized nodes, the `/three` gaps, the build-peak cap, CPU queries) has run on these GPUs yet. Where a decision rule is written in the docs it is quoted; otherwise it says "Dennis decides".

## 0. Setup

```
pnpm install
pnpm test
```
`pnpm test` is `vitest run` (no GPU). Any failure here stops the pass.

Scene assets are not in this repository. Point the two variables at the converted models and manifests in the webgpu-ddgi checkouts (what `pnpm assets:fetch` there produced; raw downloads fail with "model file not found"):

```
$env:BVH_ASSETS_DIR    = 'C:\path\to\webgpu-ddgi-radiance-cache\assets'
$env:BVH_MANIFESTS_DIR = 'C:\path\to\webgpu-ddgi-radiance-cache\scenes\manifests'
```
Defaults if unset: `../webgpu-ddgi/assets` and `../webgpu-ddgi/scenes/manifests` (`tests/gpu/README.md`). Use whichever checkout has the converted assets (HANDOFF used `webgpu-ddgi-radiance-cache` on the Mac). Absolute Windows paths are fine.

## 1. Correctness pages, once per GPU

`--gpu nvidia` or `--gpu intel` demands that GPU and fails the run if Chrome gives another (`tests/gpu/args.ts`). `--gpu intel` needs Windows (`tests/gpu/RUNNER.md`). `pnpm test:gpu:page` is `node tests/gpu/run.ts --filter`, so flags go after the page names. `--filter` matches by substring and takes several names comma-separated.

```
pnpm test:gpu:page trace-oracle --gpu nvidia
pnpm test:gpu:page trace-inline --gpu nvidia
pnpm test:gpu:page bvh-lbvh,bvh-hploc,bvh-blas-refit,bvh-top-level-gpu --gpu nvidia
pnpm test:gpu:page scene-trees-runtime,scene-trees-deforming,scene-trees-mid-swap --gpu nvidia
pnpm test:gpu:page three-trace,three-skinned --gpu nvidia
```
Then the same with `--gpu intel`. Add `--json out-nvidia.json` to keep a run.

Notes:
- `bvh-lbvh` also runs `bvh-lbvh-fallback` and `bvh-lbvh-profile` (substring match), and `bvh-hploc` also runs `bvh-hploc-profile`; the profile pages are manual, so they run only when named like this. Drop them with `--timeout` or run exact names from `tests/gpu/pages/` if they are too slow.
- `scene-trees-mid-swap` takes about 16 minutes on the RTX 3060 (30 variants; its timeout is 30 minutes). `--verbose` prints each finished variant, also when it times out.
- Decides: whether any benchmark below is worth running. A failing oracle or inline page on a GPU means stop and report; do not benchmark.
- Watch `three-skinned` and the first run of each page for D3D12 compile time (HANDOFF: pose pass "not measured on Windows").

## 2. Paired benches

All are manual pages. Run each alone, under the bench practice in section 5. They need the asset variables. Add `--json <file>` to each.

### 2a. `mergeStaticInstances` (merged against instanced tree)
```
pnpm test:gpu:page trace-bench-surface-sponza --gpu nvidia --json sponza-nv.json
pnpm test:gpu:page trace-bench-surface-bistro-interior --gpu nvidia --json bistro-nv.json
```
Read `surface-rays/paired/...` (merged over instanced; closest and any; `surface` and `air-grid`). The M5 gave Sponza 1.85x closest-hit and 1.36x any-hit, Bistro interior 1.22x and 1.11x (HANDOFF).
- Rule: "The RTX 3060 hasn't been measured yet, so the option stays off by default until it is." (`docs/internals/design/trace.md`, merged static instances). No threshold is written for the flip: **Dennis decides.**

### 2b. `triangleLeaves: 'indexed'` and `nodeFormat: 'quantized'`
They are timed in the same surface-ray pages, after the merged rounds: first the 40-byte hit record, then indexed against records (`indexed_over_records/...` and `indexed_packed_over_records/...`), then quantized against `f32` (`quantized_over_f32/...`, plus the counted walk in `surface-rays/node_format/...`). One run of 2a contains all of them. To run only the quantized section (`skip=hit-record,leaves`), pass the page option through `BVH_PAGE_QUERY`:
```
$env:BVH_PAGE_QUERY = 'skip=hit-record,leaves'
pnpm test:gpu:page trace-bench-surface-sponza --gpu nvidia --json sponza-q.json
$env:BVH_PAGE_QUERY = ''
```
`profile=quick` (3 short rounds) is only for a smoke check. The page fails if an indexed answer differs from the records one, or a quantized `t` differs from `f32` bit for bit (a tie counts apart); timing never fails it.
- Indexed rule: "The RTX 3060 decides the default (section 7, question 6)" (`docs/internals/design/indexed-leaves.md`). M5 reading there: `vec4` costs 1-6% on instanced trees and up to 17% on Sponza's merged tree, packed 5-10% more, "so `vec4` is the right default and packed is a memory option". No percentage is written: **Dennis decides.**
- Quantized rule: "The RTX 3060, where B3 found the walk bound by memory loads, decides the default (question 2)" (`docs/internals/design/compressed-nodes.md`). Question 2 asks what closest-hit cost on the RTX 3060 about 93 MiB on Sponza (40 MiB on Bistro) is worth, and whether the answer differs with indexed leaves on. No percentage is written: **Dennis decides.** M5 was 3-7% slower. Read `quantized_over_f32` alone and with indexed on.

### 2b'. Quantized nodes and indexed leaves together (section 8 of the page)
The same pages time indexed leaves on quantized nodes against quantized records, the pairing the `triangleLeaves` default waits on (HANDOFF, Open). Rows `surface-rays/paired/quantized_indexed_over_quantized_records/...` and `quantized_indexed_packed_over_quantized_records/...` (above 1: indexed is faster), bytes in `surface-rays/combined_leaves/...`. The page fails if an answer differs from quantized records other than a tie. `skip=combined` leaves it out. `tools/windows-clean-run.ps1` runs it with the rest. **Dennis decides** the default from these ratios and the bytes.

### 2b''. The folded decode of quantized nodes (section 9 of the page)
The same pages time `quantizedDecode: 'folded'` against the exact decode on the quantized records scenes (and the quantized indexed ones when section 8 runs): rows `surface-rays/paired/folded_over_exact/...` and `folded_indexed_over_exact_indexed/...` (above 1: folded is faster). The page fails if an answer differs from the exact walk's other than a tie. `skip=folded` leaves it out. It has passed `trace-oracle` and `trace-inline` on SwiftShader only. **Dennis decides** whether it becomes the default.

### 2c. The 40-byte hit record
Same pages as 2a (leave `skip=hit-record` out). Rows `surface-rays/paired/hit_record_40_over_24/...`; a ratio above 1 means the 40-byte record was faster. M5: 0.96-1.15, medians 1.02-1.11 ("It has not been measured on the RTX 3060 or on Intel", `docs/internals/design/memory.md`, "The 40-byte hit record"). No threshold is written: **Dennis decides.**

Ratios are taken within each round, order reversed every other round; the median of rounds is the result. The M5 drifted by up to about 30%, so compare ratios, not Mrays/s.

## 3. `scene-trees-sponza` memory

```
pnpm test:gpu:page scene-trees-sponza --gpu nvidia
```
Manual page, 900 s limit. It reports `SceneTrees` memory (GPU-tree state, peak, settled) and times for copy, adopt and merged modes, uncapped (`Infinity`), the default 256 MiB cap and caps of 512, 256 and 128 MiB, and the quantized and indexed variants. Gate: the GPU walk matches the CPU reference; a capped row's peak is gated at storage plus one batch. M5 references: default cap peak 711 MiB, settled 458.6; quantized 360.8 (peak 613.2); indexed 327.1 (peak 580); both 230.0 (peak 483); uncapped peak 1,482.6 (`docs/internals/design/compressed-nodes.md`, `docs/internals/design/bvh-builders.md`).
- Rule written: "Default: 256 MiB (Dennis's decision)" (`docs/internals/design/bvh-builders.md`), weighed against first trees arriving later (M5: 166 against 171 ms, no cost now). Whether it holds on D3D12: **Dennis decides** (no threshold written). Also compare tinybvh arrival times (quantized 439 ms against 351 on the M5).
- Levers built 2026-11-14, not in the default variant list (off by default; correctness passed on SwiftShader only). Run them by name:
  ```
  $env:BVH_PAGE_QUERY = 'variants=quantized-copy,quantized-copy-emit,quantized-copy-default,quantized-copy-emit-default,quantized-adopt,quantized-indexed-copy,quantized-indexed-adopt'
  pnpm test:gpu:page scene-trees-sponza --gpu nvidia --json sponza-trees-levers.json
  $env:BVH_PAGE_QUERY = ''
  ```
  `-emit` is `gpuQuantizeOnEmit` (H-PLOC writes the quantized nodes; predicted one uncapped batch 1,482.4 to 1,307.1 MiB): compare `peak`, `firstTreesMs` and `gpuBuildMs` with the same name without `-emit`. `quantized-adopt` and `quantized-indexed-adopt` are the adopt lever in quantized and indexed scenes (adoption implies the emit): compare `peak` and `settled` with the `-copy` rows. Also `bvh-hploc-profile` (manual) times the quantize pass against the emit. **Dennis decides** whether any of them becomes a default.

## 4. Appending rows to `bench/results/trace.csv`

```
node tests/gpu/pages/trace-bench/surface-to-csv.ts sponza-nv.json bistro-nv.json --append
```
Columns: `date,commit,device,browser,scene,metric,value`. Without `--append` it only prints the rows (`tests/gpu/README.md`). Check the commit and device columns name the RTX 3060 Laptop or the Intel part before committing the file.

## 5. Windows caveats and bench practice

- **`.wasm`:** since 2026-11-14 a Windows build is byte for byte the Linux and macOS one: `-fmacro-prefix-map=.\=./` with `-fno-file-reproducible` gives tinybvh's `__FILE__` the Linux spelling (`./tiny_bvh_base.h`, not `.\tiny_bvh_base.h`), so `pnpm bvh:wasm --check` passes here and a Windows-built `.wasm` may be committed (`docs/internals/design/bvh-builders.md`, "Reproducibility"). `pnpm bvh:wasm` installs the pinned emsdk into `tools/emsdk` (gitignored, about 700 MB) on first use.
- **D3D12 compile times:** the walks and kernels compile through Dawn to D3D12. The `compile-times` page times them (section 13: 140-250 ms per walk pipeline, the SDF kernel 0.7 s, H-PLOC 0.3 s on the RTX and 1.4 s on the Intel UHD, cold); the pose pass's compile cost on Windows is still unmeasured. `docs/internals/DIVERGENCES.md` row L9 records one D3D12 trap: an atomic load is a read-modify-write there, so many threads reading one word serialise. The `webgpu-tsl-compile` skill covers slow first frames.
- **Bench lock, loaded GPU:** `docs/internals/design/bvh-package-split.md` (Rules): "Medians of several runs, on the RTX 3060 with the GPU loaded (the hardware rules in `ddgi-hardware-target`)". The M5 numbers were taken "under the radiance-cache lane's bench lock" (HANDOFF). The lock's own rules are not written in this repository (`ddgi-hardware-target` is not here), so: run one bench page at a time and nothing else on the GPU; no `pnpm build` or other work in parallel (the one suspect in the unexplained `scene-trees-mid-swap` failure); same power mode every run; three runs per page, then the median of the paired medians. The Intel UHD is for correctness only, with no speed target (`README.md`). `loaded-timing.ts` submits dispatches back to back so the GPU runs at its loaded clock.
- **Leftovers:** if the runner was killed with `taskkill`, delete the leftover `bvh-gpu-chrome-*` folder in the temp directory and end any headless `chrome.exe` (`tests/gpu/RUNNER.md`).

## 6. What to report back

Per GPU: each correctness page's result; the five ratios (merged, indexed `vec4`, indexed packed, quantized, hit record) with ranges; the Sponza memory rows; compile-time observations. Then Dennis decides the three defaults.

Which GPU's numbers count (Dennis, 2026-11-15): the RTX 3060 and the M5 decide speed and memory defaults. The Intel UHD is the fallback: it must pass every correctness page, and its timings decide nothing (HANDOFF, Hardware).

## 7. Results: RTX 3060 Laptop, 2026-11-13 (commit `63af716`)

Run on Windows in headless Chrome 157.0.8103.51. Sponza and Bistro interior, section 2 only, three runs each. The rows are in `bench/results/trace.csv`: `run1`-`run3` and `median/` prefixes, plus unprefixed median rows (`bd8ba54`). Each figure below is the median over the runs of each run's median-of-rounds ratio. Above 1 means the option is faster.

| Lever | Sponza | Bistro interior | M5 (earlier) | Memory |
| --- | --- | --- | --- | --- |
| `mergeStaticInstances` (merged over instanced) | closest 1.44-1.67, any 1.24-1.48 | closest 1.19-1.20, any 1.16-1.35 | 1.85 closest, 1.36 any (Sponza surface) | top level 16.5 KiB to 192 B |
| `nodeFormat: 'quantized'` over `f32` | closest 1.04-1.22, any 0.98-1.03 | closest 1.02-1.08, any 0.98-1.01 | 0.93-0.97 | nodes 185 to 93 MiB (Bistro 29.5 to 14.8) |
| `triangleLeaves: 'indexed'`, `vec4` | 0.92-0.97 | 0.92-0.97 | 0.83-0.99 | triangles 263 to 133 MiB, positions included (Bistro 47.7 to 24.8) |
| indexed, `positionLayout: 'packed'` | 0.89-0.95 | 0.82-0.88 | 0.75-0.92 | about 11 MiB less than `vec4` on Sponza |
| 40-byte hit record over 24 | 0.88-0.92 | 0.93-0.94 | 0.96-1.15 | 16 bytes per hit |
| `anyHitOrder: 'larger'` over `'far'` | instanced 1.11-1.45; merged air 1.07, merged surface 0.87 | instanced 1.18-1.48; merged air 1.03, merged surface 0.80 | 1.15-1.42 instanced, 0.78 merged surface | none |
| `anyHitOrder: 'near'` over `'far'` | 0.59-1.18 | 0.75-1.21 | n/a | none |

The bench checked correctness as it ran. Compact against full hit record, indexed against records, and every any-hit order against far agreed on every ray. Merged against instanced: closest hits named the instanced tree's instance and triangle on 99.949-99.996% of rays (the page requires 99.9%). The rest are near-ties at a shared distance, where `t` agrees to rounding and the merged walk can pick another instance or triangle. Any hits blocked exactly the same rays. Quantized node `t` values differed from `f32` on no ray.

Not in this push, and not seen in this session:
- section 1 correctness pages (`trace-oracle`, `trace-inline` and the rest);
- section 3 (`scene-trees-sponza` memory and the build-peak cap on D3D12);
- D3D12 compile times;
- the Intel UHD;
- the quantized and indexed variants together (each was timed alone against the default).

**Decided from this run (Dennis, 2026-11-13):** `mergeStaticInstances` on by default (it merges only instances marked `static`); `nodeFormat: 'quantized'` the default for every scene and walk; `triangleLeaves` stays `'records'` until a run of quantized nodes and indexed leaves together; the 40-byte hit record stays the default (shadow-only users pass `hitNormal: false`); `anyHitOrder` stays `'far'`. The bench pages keep their baselines by name (instanced, `f32` nodes, records), so their rows mean what they meant before the defaults changed.

Next pass: the section 1 correctness pages on the new defaults, section 3 (`scene-trees-sponza`, now quantized by default), D3D12 compile times, the Intel UHD, and quantized nodes with indexed leaves together. The surface-ray pages time each variant alone today; the combined comparison needs a section added to `tests/gpu/pages/trace-bench/surface-rays-bench.ts` first.

## 8. pmndrs/bvh#1 branch: RTX 3060 Laptop, 2026-11-14 (branch `x2-pathtracer-requests`, `376adbd`)

Headless Chrome on D3D12; correctness only, no timings. Assets: `BVH_ASSETS_DIR=C:\Users\denni\Documents\dev\webgpu-ddgi\assets`, `BVH_MANIFESTS_DIR=...\webgpu-ddgi\scenes\manifests` (the default relative paths don't resolve from this checkout).

- `trace-oracle` PASS (338 s), `trace-inline` PASS (60 s), on the new defaults (quantized nodes, merged static instances). Section 1's other pages were not run.
- `trace-cpu-agreement` PASS: the doubles CPU walk gives the GPU's instance and triangle on all 98,304 Sponza primary rays (merged quantized and instanced `f32` scenes); any hits agree on every ray of every set; `t` within 41 f32 steps on primary rays; seam rays name the other neighbour on about 29% (ties at an edge).
- `wgsl-f32-arithmetic` (reports only): `dot()` is `fma(z, fma(y, x * x))`; `1.0 / x` correctly rounded on 86.9% of inputs, else 1 ulp off; `a / b` about `a * (1 / b)`; `sqrt`, `inverseSqrt` 1 ulp off at times; written products and sums not fused.
- `scene-trees-runtime` and the 20 example smoke pages passed with the worker handshake (item 2).
- GPU idle between runs: P8, 255 MHz graphics, 49 °C.

## 9. Clean run: RTX 3060 Laptop and Intel UHD, 2026-11-14 (commit `6c32897`)

`tools/windows-clean-run.ps1 -Commit 6c32897`, headless Chrome 157 on D3D12, after the downsampler finished. Everything in `bench/runs/2026-11-14-6c32897/` (`summary.md`, a log and JSON per step); the bench rows are appended to `bench/results/trace.csv` (`run1`-`run3` and `median/` prefixes, as in section 7).

**Correctness (section 1 plus `query-oracle`, `sdf-oracle`, `trace-clustered`, `trace-cpu-agreement`, `bvh-shared`, `three-shapecast`), both GPUs:** everything passed but two pages, both since fixed in the tests, not the library:
- `scene-trees-mid-swap` timed out at 900 s on both. Rerun alone with progress logging (`d9f0f8c`): PASS in 965 s on the RTX, every case and variant through. Not a hang; the page grew to 30 variants (indexed, quantized, levers) since the M5's 170 s. Its timeout is now 1,800 s.
- `trace-cpu-agreement` on the Intel UHD: 1 of 98,304 Sponza primary rays named another triangle, in both scenes. Both hits are well inside their triangles of mesh 25, `t` 2 ulp apart, and the f32 scene orders the pair the other way: two overlapping coplanar faces. The page now counts another triangle within 64 ulp of `t` as a near tie (reported, not failed); passes on both GPUs (`17b0699`). The RTX has no near tie.

**Ratios on the RTX** (each run's median of rounds, range over the 3 runs; above 1 means the option is faster):

| Lever | Sponza | Bistro interior |
| --- | --- | --- |
| merged over instanced | closest 1.44-1.72, any 1.23-1.58 | closest 1.19-1.20, any 1.17-1.36 |
| `quantized` over `f32` | closest 0.99-1.21, any 0.97-1.04 | closest 1.01-1.09, any 0.95-1.02 |
| indexed `vec4` over records (f32 nodes) | 0.91-1.03 | 0.92-0.97 |
| indexed packed over records (f32 nodes) | 0.88-0.99 | 0.82-0.88 |
| **indexed `vec4` on quantized over quantized records** (section 8) | closest 0.83-0.95, any 0.95-1.03 | closest 0.89-0.93, any 0.92-0.94 |
| **indexed packed on quantized over quantized records** | closest 0.77-0.87, any 0.90-1.03 | closest 0.77-0.81, any 0.83-0.86 |
| folded decode over exact (section 9) | instanced 0.93-0.97, merged 1.00-1.06 | instanced 0.94-0.97, merged 1.00-1.05 |
| folded decode, indexed | instanced 0.92-0.99, merged 1.02-1.08 | instanced 0.95-0.99, merged 1.00-1.03 |
| 40-byte hit record over 24 | 0.88-0.92 | 0.92-0.94 |
| `anyHitOrder: 'larger'` over `'far'` | instanced 1.11-1.45; merged air 1.06, surface 0.87 | instanced 1.18-1.49; merged air 1.05-1.07, surface 0.80 |
| `anyHitOrder: 'near'` over `'far'` | 0.58-1.25 | 0.74-1.22 |

Section 8's walks agreed on every ray (`different` 0 in every entry). Indexed leaves on quantized nodes cost 5-17% on closest hits (packed 13-23%), more than on f32 nodes; any hits roughly break even on Sponza surface rays.

**Sponza memory (section 3), RTX:** 132 meshes, 5,744,384 triangles; every variant 0 disagreements.

| Variant | Cap | Batches | Peak MB | Settled MB (nodes / triangles / positions) |
| --- | --- | --- | --- | --- |
| copy, uncapped | none | 1 | 1,482.7 | 470.0 (206.9 / 263 / 0) |
| adopt, uncapped | none | 1 | 1,482.7 | 613.6 (350.6 / 263 / 0) |
| copy, default cap | 256 | 7 | 711.0 | 458.6 (195.6 / 263 / 0) |
| copy, cap 512 / 128 | 512 / 128 | 3 / 13 | 963.0 / 581.4 | 460.7 / 453.3 |
| indexed `vec4`, default cap | 256 | 6 | 580.0 | 327.1 (194.1 / 87.7 / 45.2) |
| indexed packed, default cap | 256 | 6 | 568.7 | 315.8 (194.1 / 87.7 / 33.9) |
| quantized, default cap | 256 | 7 | 613.2 | 360.8 (97.8 / 263 / 0) |
| default cap + `trimWhenRefined` | 256 | 7 | 711.0 | 448.1 (185.1 / 263 / 0) |
| adopt + trim | none | 1 | 1,482.7 | 448.1 |
| default cap, `parents: 'refit'` | 256 | 7 | 707.3 | 458.6 |

`quantizeJobs: 'batched'` changed nothing here (613.2 either way).

**Examples:** all 41 smoke pages passed on the RTX, `three-load-gltf` included (it fails only on SwiftShader). Screenshots now show the canvas; the ones looked at draw correctly. Two cosmetic notes: `three/pathtracer`'s HUD shows depth, nodes and triangles 0 for its GPU tree, and `babylon/gpu-trace`'s shadows look blocky. Both fixed 2026-11-15 in the examples: the HUD now counts merged static groups' trees (the pathtracer's meshes are all static, so its one tree is a group's), and `gpu-trace` traced at an eighth of the size in every smoke run, not only on a software adapter.

Not covered by this run: D3D12 compile times (no page records them), the Intel UHD benches (correctness only), SDF against three-mesh-bvh on the RTX beyond `compare/sdf`'s smoke.

## 10. The quantized emit and adopt levers: RTX 3060 Laptop, 2026-11-15 (commit `c48fe00`)

Section 3's lever variants by name and `bvh-hploc-profile`, one run each, RTX idle and the downsampler finished. Files in `bench/runs/2026-11-15-c48fe00-levers/`. Every variant 0 disagreements.

| Variant (`scene-trees-sponza`) | Peak MB | Settled MB (nodes / triangles / positions) | First trees ms | GPU build ms |
| --- | --- | --- | --- | --- |
| quantized copy, uncapped | 1,482.6 | 366.5 (103.5 / 263 / 0) | 181 | 51 |
| quantized copy, uncapped, emit | **1,307.3** | 366.5 | 175 | 50 |
| quantized copy, default cap | 613.2 | 360.8 (97.8 / 263 / 0) | 215 | 71 |
| quantized copy, default cap, emit | 613.5 | 360.1 | 232 | 112 |
| quantized adopt (emit implied) | 1,307.3 | 438.3 (175.3 / 263 / 0) | 216 | 78 |
| quantized indexed copy, uncapped | 1,350.9 | **236.4** (103.5 / 87.7 / 45.2) | 213 | 47 |
| quantized indexed adopt | 1,175.6 | 308.2 (175.3 / 87.7 / 45.2) | 209 | 44 |

`bvh-hploc-profile` (Sponza, 132 meshes, one batch): the `f32` build plus the quantize pass 41.6 ms (the pass 0.9), the quantized emit 35.1 ms; node bytes during the build 464.1 to 183.8 MB.

Reading it: the emit builds the same nodes (`bvh-hploc` checks word for word; it passed on both GPUs in section 9) about 16% faster and takes 175 MB off an uncapped peak; at the default cap the peak is the same. Quantized nodes with indexed leaves settle at 236.4 MB against 366.5 with records (uncapped, like for like): the 130 MB the `triangleLeaves` default weighs against section 8's 5-17%. Single-run times vary as much as the differences here (section 9's two identical capped copy rows: first trees 471 and 256 ms), so the first-trees column decides nothing.

## 11. SDF against three-mesh-bvh: RTX 3060 Laptop, 2026-11-16 (commit `6e05ea9`)

`compare/sdf` (`pnpm examples:smoke --filter compare-sdf --gpu nvidia`, `BVH_PAGE_QUERY='full&repeat=4'`, 5 paired rounds; `&knot=2000,250` for the 1M knot, `&breakdown` for each kernel part alone). Headless Chrome 157, D3D12. The GPU ran hot (80-88 °C, Windows' software thermal slowdown active), so the milliseconds drift between runs; the paired ratios hold. Files in `bench/runs/2026-11-16-sdf/` (local).

**Paired ratio, three-mesh-bvh's time over ours** (above 1: ours faster; median of each run's median of rounds, range over runs):

| Ours | 100k knot, 32³ / 64³ / 128³ (3 runs) | 1M knot, 32³ / 64³ / 128³ (2 runs) |
| --- | --- | --- |
| parity, before (all three rays) | 0.75 / 0.69 / 0.69 (0.66-0.77) | 1.03 / 0.76 (one breakdown run, 32³ / 64³) |
| parity, now (third ray on a split) | 0.83 / 0.76 / 0.77 (0.71-0.87) | 1.05 / 0.94 / 0.89 (0.82-1.06) |
| `sign: 'first-hit'` | 0.95 / 0.98 / 0.99 (0.90-1.07) | 1.13 / 1.17 / 1.04 (1.01-1.20) |
| unsigned | 1.02 / 1.07 / 1.09 (1.00-1.15) | 1.15 / 1.13 / 1.06 (1.03-1.22) |

**Breakdown, 100k knot, 128³** (the coolest run, ms per field, back to back): closest point alone 76.8 (theirs 73.4); our three all-hit rays alone 40.5, two-or-three 27.0, one 12.6; our first-hit ray 5.1, theirs 3.8. Whole fields: three rays 112.7, two-or-three 103.5, first-hit 81.3, theirs 79.2. The winding sum costs nothing (113.9 without it). On a merged scene (no per-box instance transform) the closest point takes 63.6 (17% less), the default field 87.4. Seeding the closest point with the first hit's distance saved nothing (143.4 against 144.4).

Reading it: the closest point costs the same in both; the gap is the sign. An all-hit ray costs about 2.5 first-hit rays, and the vote needs two or three. On the 1M knot the closest point dominates and the libraries are level. The 1.7-2.8x on SwiftShader does not carry over to the RTX (1.3-1.5x before this run, 1.2-1.3x now). Every breakdown kernel that should equal a library field did, bit for bit, at every size on both knots (the early-out vote against all three rays, parity and winding).

Correctness: `sdf-oracle` passed on the RTX (full profile, with the new first-hit runs) and on SwiftShader (`profile=ci`); `three-sdf` and `compare-sdf` smoke passed on the RTX; `pnpm test` 854 passed.

Not done: the M5; an exact-identity fast path in the shapecast walk (an instance whose transform is the identity could skip the per-box transform as merged trees do: the measured 17% on the closest point), which touches every query and wants the query oracles on both GPUs first.

## 12. The identity fast path in the shapecast walk: measured, not built. RTX 3060 Laptop and Intel UHD, 2026-11-16

The open item of section 11: an instance whose transform is exactly the identity could skip the per-box and per-corner matrix of the query walk. Two forms were built (commit `95614a4`, undone by the next commit) and timed with `compare/sdf ?breakdown&only=...` (100k knot, `full&repeat=4&dims=64,128`, 5 paired rounds per run, 3 runs; `docs/internals/design/trace.md`, "Spatial queries"). Files in `bench/runs/2026-11-16-identity/` (local).

**The GPU was not idle.** Another project's GPU suite (the downsampler's `test/run.js`, six headless Chromes on the RTX) ran from about 22:20 to past 23:30, through every run here; it was not stopped. The RTX averaged 85 °C over each run, with Windows' thermal slowdown flag on 84 of 93 two-second samples of the second run (mean SM clock 1,052 MHz). Only the paired ratios (both kernels in every round, the order reversed every other round) are reported; the milliseconds are not comparable with section 11's.

**Paired ratio, the old path's time over each form's** (above 1: the form is faster; each run's median of rounds):

| Form | 64³ (runs 1-3) | 128³ (runs 1-3) | Bit for bit with the old path |
| --- | --- | --- | --- |
| Bit for bit: the padded box without the matrix, zeros, subnormals and inf/NaN through it | 0.86 / 0.96 / 0.93 | 0.86 / 0.87 / 0.91 | yes (checked, below) |
| The same on boxes only, corners through the matrix | 0.87 / 0.94 / 0.94 | 0.87 / 0.90 / 0.90 | yes |
| The identity instance read as merged (exact boxes, corners as stored) | 1.09 / 1.18 / 1.20 | 1.16 / 1.18 / 1.20 | no (ties, touching, `-0`) |
| The same mesh merged (`static: true`, its own tree) | 0.97 / 1.25 / 1.08 | 1.07 / 1.04 / 1.04 | no (another tree) |
| Signed field (closest point and the early-out vote), bit-for-bit form | 1.05 / 0.91 / 0.96 | 0.94 / 0.96 / 0.99 | yes |

**One run on an idle RTX** (after the other suite stopped, 23:40; started at 49 °C, mean SM clock 1,453 MHz, the thermal flag on 33 of 45 samples), ms per field at 128³: the old path 70.4, the bit-for-bit form 76.9, read as merged 58.7, the mesh merged 61.9. Ratios as above, 64³ / 128³: bit for bit 0.93 / 0.95, boxes only 0.95 / 0.95, read as merged 1.17 / 1.22, merged 1.10 / 1.17. The same picture.

An earlier run while the other suite was busiest gave the bit-for-bit box form without its guard 1.02-1.03 and with corners as stored 1.04-1.11: the guard alone turns a small gain into a loss.

**Correctness of the bit-for-bit form, full profile:** `query-oracle`, `sdf-oracle`, `trace-oracle`, `trace-inline`, `three-shapecast` and `three-trace` passed on the RTX and the Intel UHD, and the CI job's seven pages on SwiftShader (`profile=ci`). `query-oracle` ran every query a second time with the form taken out and compared every answer word plus a hash of every box and corner the shapecast hook is handed: 234,000 queries per GPU (13 cases, 9 variants, 2,000 points), 0 differ on either GPU (23,400 on SwiftShader). Its new case, a floor at `y = 0` with `+0`, `-0` and exact zeros and some subnormal corners, showed why the guard is needed: without it the Intel UHD gave `-0` normals where the matrix gives `+0` (`normal.z` 0x80000000 against 0) and other box and corner hashes. On SwiftShader `query-oracle` with the second dispatch took 197 s (compiles 73 s) under the same load.

**Decision left to Dennis:** nothing changed in the walk. The bit-for-bit form is slower, so it is not a lever. The form that is faster (about 18%) changes which of two equally near triangles a closest point names, so it would need an opt-in option on every module that compiles queries (`traceModuleWgsl`, `closestPointKernelWgsl` / `ClosestPointBatchKernel`, `sdfKernelWgsl` / `SdfGenerator`, `BvhTraceNodes`). `static: true` (merged) already gets that path. Kept: the `zeros-and-subnormals` case in `query-oracle` (after the undo `query-oracle` passed again: RTX and Intel UHD full profile, SwiftShader `profile=ci`; `compare-sdf` smoke passed on the RTX), and `compare/sdf`'s `closest-identity-as-merged` breakdown kernel (a text patch, not the library) with `?only=`.

## 13. D3D12 compile times: RTX 3060 Laptop and Intel UHD, 2026-11-16

The new manual page `compile-times` (`tests/gpu/pages/compile-times`, reports only) times the package's pipeline compiles: each item on a fresh device, one at a time, from the first `createShaderModule` to the last pipeline promise (the library's own `prepare()` / `create()` where it has one, its pipelines asked for together). Chrome 157.0.8103.51 headless, D3D12 (WGSL through Tint to HLSL, then DXC and the driver). The machine was idle (the other project's suite had stopped); compiles run on the CPU (an i9-11900H) as much as the GPU, and nothing else ran. Files in `bench/runs/2026-11-16-compile/` (local).

```
BVH_PAGE_QUERY='items=hploc' pnpm test:gpu:page compile-times --gpu nvidia     # one item alone: pass 1 cold, pass 2 warm
BVH_PAGE_QUERY='items=sdf&nonce' pnpm test:gpu:page compile-times --gpu intel  # the walk's code new to the driver's cache too
```

**Each item alone in a fresh Chrome profile** (the runner makes one per run), ms. Cold is pass 1 (no browser shader cache); warm is pass 2 on a new device in the same browser (Chrome's shader cache answers). "Nudged" changes one constant of the walk's modules (`?nonce`) so the compiled code is new to the driver's own cache as well: a user's true first launch.

| Item | RTX cold / warm | RTX nudged | Intel cold / warm | Intel nudged |
| --- | --- | --- | --- | --- |
| ray walk, closest-hit batch kernel, quantized (default) | 180 / 7 | 195 | 140 / 2 | 133 |
| the same, `f32` nodes | 167 / 6 | 174 | 131 / 2 | 125 |
| any-hit batch kernel, quantized | 138 / 2 | 172 | 115 / 2 | 112 |
| closest hit, quantized, indexed leaves | 195 / 2 | | 147 / 3 | |
| closest hit in a module with `queries: true` (unused) | 200 / 3 | | 147 / 3 | |
| closest hit with `createComputePipeline` (sync), to `onSubmittedWorkDone` | 169 / 6 | | 137 / 2 | |
| `TraceBatchKernel.create()` (closest and any together) | 205 / 2 | | 142 / 3 | |
| `ClosestPointBatchKernel.create()`, quantized / `f32` | 245 / 3, 238 / 3 | 247 | 206 / 3, 199 / 3 | 202 |
| a caller's shapecast (`shapecast: hooks`) | 174 / 9 | 188 | 187 / 2 | 136 |
| `SdfGenerator.create()` | **768** / 3 | 749 | **688** / 4 | 594 |
| `GpuHplocBuilder.prepare()` (with its radix sort and scan) | 331 / 12 | | **1,393** / 7 | |
| `GpuLbvhBuilder.prepare()` | 341 / 11 | | 419 / 4 | |
| `GpuBvhRefitKernels.prepare()` | 106 / 1 | | 138 / 1 | |
| `GpuMeshRefitKernels.prepare()` (and the refit) | 112 / 1 | | 149 / 2 | |
| `GpuNodeQuantizer.create()` | 89 / 1 | | 86 / 1 | |
| `GpuTopLevel.prepare()` | 115 / 4 | | 140 / 4 | |
| `BvhTraceNodes` in a TSL compute kernel (`traceClosest`), first dispatch less the second | 181 / 4 | 147 | 194 / 5 | 170 |
| `BvhTraceNodes` in a node material (`traceAny`), `renderer.compileAsync` | 185 / 7 | 183 | 210 / 16 | 264 |

Reading it:
- **One walk pipeline costs 140-250 ms cold on either GPU; the SDF kernel 0.6-0.8 s** (it holds the closest point and three crossing walks); H-PLOC's `prepare()` 0.33 s on the RTX and **1.4 s on the Intel UHD**. `SceneTrees.create` compiles H-PLOC, the quantizer, the top level and the refits (in parallel); a page that also traces, queries and makes an SDF adds about 0.2 s per kernel and 0.7 s for the SDF.
- **Warm is 1-16 ms.** Chrome's shader cache answers across devices in one browser, and the default profile keeps it on disk for the next visit (Chrome's GPU shader disk cache; not timed across a browser restart here, since the runner makes a fresh profile each run). With `--disable-gpu-shader-disk-cache` pass 2 was still 1-23 ms: the GPU process keeps its cache in memory for the session.
- **The driver's own cache saves little:** nudged code (new to every cache) compiled within about 10% of the cold times on both GPUs, mostly within the run-to-run spread, so the cold column is close to a user's first launch.
- **A synchronous pipeline costs the same time but stalls the queue:** `createComputePipeline` returned at once, and the next `onSubmittedWorkDone` came 169 ms later (RTX); any frame submitted behind it waits that long. The library makes every pipeline with `createComputePipelineAsync`.
- Items share code: run together in one profile (`pass 1` of an all-items run), later items are partly warm (the mesh refit after the refit 31 ms, the top level 41, a second closest-hit kernel 3-11), so the table times each alone.

Not covered: the Apple M5 (the page runs there as is), a cold start across a browser restart with the disk cache kept, `three-skinned`'s pose pass.

## 14. The furnished room's closest hits from outside: the child order on ties, not the grid. RTX 3060 Laptop, 2026-11-17

The 2026-11-16 re-run (`compare/RESULTS.md`) found ours 0.51x three-mesh-bvh/webgpu on `compare/gpu`'s furnished room, closest hits from outside, with quantized nodes and 0.96x with `?nodeFormat=f32`. The guess was the 16-bit grid of one 270k-triangle mesh. It is not. The GPU was idle at the start (48 °C, P8, nothing in `nvidia-smi`'s compute list but the desktop Chrome and Camera Hub; no other headless suite); the runs below waited for 55 °C or less before each page and peaked at 82 °C (the surface-ray pages). Files in the session's scratch folder, not kept.

**The grid is fine.** `compare/node/closest-ties.ts` (Node, no GPU) prints each scene's frame and decoded SAH:

| Scene (one merged mesh) | Step (m) | Steps across | SAH quantized / `f32` |
| --- | --- | --- | --- |
| Furnished room, 269,764 triangles | 1.2e-4 / 6.1e-5 / 1.2e-4 | 33,375 / 54,879 / 41,595 | 1.0017 |
| Torus knot, 100k | 6.1e-5 / 6.1e-5 / 3.1e-5 | 53,768 / 56,165 / 52,428 | 1.0014 |
| Sponza, 5.74M | 9.8e-4 / 4.9e-4 / 4.9e-4 | 37,117 / 40,731 / 48,673 | 1.0066 |
| Bistro interior, 1.04M | 2.4e-4 / 1.2e-4 / 4.9e-4 | 63,999 / 59,673 / 39,757 | 1.0011 |

`checkGpuLayout` finds nothing on the room's quantized tree (every decoded box holds its triangles, outward by the margin), and its frame is `nodeFrameOf` of the exact root box. No flat or zero-size box misbehaves; the room is near the origin (bounds -2.04..2.04, -0.11..3.24, -2.54..2.54). The room's SAH grows the least of the four scenes, so no SAH threshold could have singled it out.

**The cause: exact ties in the child order.** Closest-hit enters the child with the nearer entry distance first; on an exact tie it takes the left one (tinybvh's rule, DIVERGENCES K2). The room tree's root has two children that share five of their six planes, up to the last bits of an `f32`: left spans z -2.54..0.07, right the whole room. A ray from outside enters both through the same outer plane. On `f32` boxes the right child's entry came out 1-4 ulp nearer (5.385883 against 5.385885, say), so the walk entered it first; quantized boxes round both planes to the same grid plane, the entries tie exactly, and the walk goes left, into geometry behind the wall it hasn't reached yet, before it comes back for the right child. Of 8,192 outside rays, 978 decided the root the other way, every one a quantized tie going left; 707 tested leaves the `f32` walk never touched. Ties at inner nodes over those rays: 7,016 on `f32`, 9,917 quantized. The `f32` walk loses to ties too: outside rays meet shared outer planes all the time.

Work per ray (compare's seeded rays; closest hit; nodes / triangles / the slowest ray of each 32, what a SIMD group waits for):

| Rays | `f32`, ties left (now) | quantized, ties left (now) | `f32`, face ties right | quantized, face ties right |
| --- | --- | --- | --- | --- |
| Room, outside (CPU model, 8,192) | 11.06 / 6.87 / 43.2 | 13.40 / 9.97 / 60.7 | 9.14 / 5.37 / 20.9 | 9.74 / 5.44 / 21.1 |
| Room, outside (GPU counted walk, 1M; box tests / triangles / slowest box tests) | 24.45 / 6.89 / 89.5 | 28.83 / 10.36 / 124.4 | 20.73 / 5.42 / 43.6 | 21.53 / 5.45 / 43.9 |
| Room, inside (GPU, 1M) | 27.27 / 6.49 / 101.2 | 27.30 / 6.52 / 101.4 | the same | the same |
| Sponza, outside (CPU) | 22.15 / 4.32 / 68.4 | 22.38 / 4.36 / 68.8 | 20.38 / 4.01 / 67.4 | 20.72 / 4.03 / 67.7 |
| Sponza, inside / secondary (CPU) | 30.42 / 3.63, 39.16 / 4.89 | 30.54 / 3.69, 39.40 / 5.04 | 30.45 / 3.68, 39.14 / 4.90 | 30.57 / 3.74, 39.37 / 5.05 |
| Knot, Bistro interior (CPU, every set) | | | within 1% of now | within 1% of now |

(three-mesh-bvh's counted walk on the room from outside: 19.92 box tests, 13.13 triangles, 68.7 on the slowest of 32. Before this pass `?counts` counted ours wrongly on quantized scenes, 4 box tests and no hits: its counted walk was compiled for `f32` nodes; fixed, `compare/gpu/gpu-counts.ts`.)

**The lever: ties beyond `tMin` go right.** `leftEntry < rightEntry || leftEntry == ray.tMin` in place of `leftEntry <= rightEntry` (closest hit only; any-hit's orders unchanged): a tie at `tMin` (the ray starts inside both boxes, the usual case for rays from surfaces and inside rooms) still goes left, so those walks are unchanged; a tie further on (both boxes entered through one shared plane) goes right. Built as commit `7d6fda5` in the WGSL walk and the CPU walk, undone by `e372387`. `compare/gpu`, `full` (1,048,576 rays, 7 paired rounds), two runs each, Mrays/s and ours over three-mesh-bvh:

| Scene, rays | Now (ties left) | Face ties right |
| --- | --- | --- |
| Room, outside, quantized | 101.3 / 101.4 (0.518, 0.518) | **444.8 / 449.3 (2.33, 2.35)** |
| Room, outside, `f32` | 183.0 / 183.1 (0.946, 0.951) | **470.8 / 464.2 (2.47, 2.45)** |
| Room, inside, quantized | 144.5 / 143.5 (0.868, 0.846) | 143.5 / 143.7 (0.850, 0.849) |
| Room, inside, `f32` | 151.5 / 151.0 (0.909, 0.915) | 149.4 / 149.9 (0.891, 0.905) |
| Sponza, outside, quantized | 159.9 / 160.6 (1.09, 1.10) | 170.0 / 170.0 (1.19, 1.15) |
| Sponza, inside, quantized | 137.2 / 138.3 (1.38, 1.37) | 137.8 / 137.4 (1.36, 1.37) |
| Knot, outside / inside, quantized | 141.5, 142.3 / 143.8, 144.2 | 141.2, 138.9 / 143.2, 143.3 |

The surface-ray benches, with a paired section that timed the rule against today's in the same page (5 rounds, closest hit, `f32` and quantized, instanced and merged, `surface` and `air-grid`; a scratch section, not committed): every ratio 0.996-1.003 on Sponza and 0.998-1.004 on Bistro interior, work per ray the same to two decimals, every `t` the same; another triangle at the same `t` on 0-19 of about 230,000 surface rays per set and up to 209 air-grid rays (Bistro). With the rule, quantized over `f32` on the room from outside is 0.96, the decode's usual cost (the room's inside rays: 0.95-0.96).

Other rules, measured and not recommended: every tie right (the same on outside rays, but ties at `tMin` going right cost inside and surface rays 1-3% more nodes and 2-4% more triangles on the CPU, and Sponza inside 0.96x on the GPU, one run); the larger child on a tie (as good on outside rays, worse inside, a half-area per tie; the knot 0.97-0.99x); from inside both boxes the nearer exit, else the larger (the fewest nodes on the CPU, 1.5-3% fewer on inside and surface rays, but 0.95x on Sponza's inside rays and 0.98x on the knot on the GPU: recomputing the exits costs more than it saves).

**Correctness of `7d6fda5`:** `trace-oracle`, `trace-inline`, `query-oracle`, `sdf-oracle` and `three-trace` on the RTX and the Intel UHD (full profile), and the CI job's seven pages on SwiftShader (`profile=ci`): all passed (RTX: `trace-oracle` 405 s, every `mismatches` 0, no shared-edge slip beyond the f32 counts the page allows; Intel UHD `trace-oracle` 420 s; SwiftShader 280 s for the seven). `pnpm test` 873 passed with it.

**Decision left to Dennis:** the default walk is unchanged. Options, with what each costs:
- **Face ties right** (`7d6fda5`; **taken by Dennis 2026-11-17**, the revert of `e372387` on `main`): one comparison more per inner node with two children hit; the room from outside 4.4x on quantized nodes and 2.5x on `f32`, Sponza from outside 1.06x, every other measured set level; a closest hit can name the other of two triangles at exactly the same `t` (as `mergeStaticInstances` and quantized nodes already can). A departure from tinybvh's tie rule (DIVERGENCES K2); the M5 has not run it.
- **`nodeFormat: 'f32'` for such scenes:** 1.8x on these rays (183 against 101 Mrays/s), half of what the tie rule gives, for 2x the node bytes. Not needed if the tie rule goes in.
- **A finer or per-subtree frame, or `f32` for meshes whose decoded SAH grows past a threshold:** would not help. The planes that tie differ by about 1e-6 m against a 1.2e-4 m step: no 16-bit grid keeps that, and the room's SAH growth (0.17%) is the smallest of the four scenes.

To reproduce: `node --experimental-transform-types --no-warnings --import ./compare/node/register.mjs compare/node/closest-ties.ts furnished-room torus-knot sponza bistro-interior --rules left,face-right,right,larger,hybrid` (CPU), and `BVH_PAGE_QUERY='scene=furnished-room&full&counts' pnpm examples:smoke --filter compare-gpu --gpu nvidia` with and without `&nodeFormat=f32` (GPU).

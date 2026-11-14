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
- `scene-trees-mid-swap` takes about 170 s on the M5. If it fails, keep the full log (HANDOFF, Open).
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

### 2c. The 40-byte hit record
Same pages as 2a (leave `skip=hit-record` out). Rows `surface-rays/paired/hit_record_40_over_24/...`; a ratio above 1 means the 40-byte record was faster. M5: 0.96-1.15, medians 1.02-1.11 ("It has not been measured on the RTX 3060 or on Intel", `docs/internals/design/memory.md`, "The 40-byte hit record"). No threshold is written: **Dennis decides.**

Ratios are taken within each round, order reversed every other round; the median of rounds is the result. The M5 drifted by up to about 30%, so compare ratios, not Mrays/s.

## 3. `scene-trees-sponza` memory

```
pnpm test:gpu:page scene-trees-sponza --gpu nvidia
```
Manual page, 900 s limit. It reports `SceneTrees` memory (GPU-tree state, peak, settled) and times for copy, adopt and merged modes, uncapped (`Infinity`), the default 256 MiB cap and caps of 512, 256 and 128 MiB, and the quantized and indexed variants. Gate: the GPU walk matches the CPU reference; a capped row's peak is gated at storage plus one batch. M5 references: default cap peak 711 MiB, settled 458.6; quantized 360.8 (peak 613.2); indexed 327.1 (peak 580); both 230.0 (peak 483); uncapped peak 1,482.6 (`docs/internals/design/compressed-nodes.md`, `docs/internals/design/bvh-builders.md`).
- Rule written: "Default: 256 MiB (Dennis's decision)" (`docs/internals/design/bvh-builders.md`), weighed against first trees arriving later (M5: 166 against 171 ms, no cost now). Whether it holds on D3D12: **Dennis decides** (no threshold written). Also compare tinybvh arrival times (quantized 439 ms against 351 on the M5).

## 4. Appending rows to `bench/results/trace.csv`

```
node tests/gpu/pages/trace-bench/surface-to-csv.ts sponza-nv.json bistro-nv.json --append
```
Columns: `date,commit,device,browser,scene,metric,value`. Without `--append` it only prints the rows (`tests/gpu/README.md`). Check the commit and device columns name the RTX 3060 Laptop or the Intel part before committing the file.

## 5. Windows caveats and bench practice

- **`.wasm`:** since 2026-11-14 a Windows build is byte for byte the Linux and macOS one: `-fmacro-prefix-map=.\=./` with `-fno-file-reproducible` gives tinybvh's `__FILE__` the Linux spelling (`./tiny_bvh_base.h`, not `.\tiny_bvh_base.h`), so `pnpm bvh:wasm --check` passes here and a Windows-built `.wasm` may be committed (`docs/internals/design/bvh-builders.md`, "Reproducibility"). `pnpm bvh:wasm` installs the pinned emsdk into `tools/emsdk` (gitignored, about 700 MB) on first use.
- **D3D12 compile times:** the walks and kernels compile through Dawn to D3D12. Note the time to the first result of each page on a cold run; the pose pass's compile cost on Windows is unmeasured (HANDOFF). `docs/internals/DIVERGENCES.md` row L9 records one D3D12 trap: an atomic load is a read-modify-write there, so many threads reading one word serialise. The `webgpu-tsl-compile` skill covers slow first frames.
- **Bench lock, loaded GPU:** `docs/internals/design/bvh-package-split.md` (Rules): "Medians of several runs, on the RTX 3060 with the GPU loaded (the hardware rules in `ddgi-hardware-target`)". The M5 numbers were taken "under the radiance-cache lane's bench lock" (HANDOFF). The lock's own rules are not written in this repository (`ddgi-hardware-target` is not here), so: run one bench page at a time and nothing else on the GPU; no `pnpm build` or other work in parallel (the one suspect in the unexplained `scene-trees-mid-swap` failure); same power mode every run; three runs per page, then the median of the paired medians. The Intel UHD is for correctness only, with no speed target (`README.md`). `loaded-timing.ts` submits dispatches back to back so the GPU runs at its loaded clock.
- **Leftovers:** if the runner was killed with `taskkill`, delete the leftover `bvh-gpu-chrome-*` folder in the temp directory and end any headless `chrome.exe` (`tests/gpu/RUNNER.md`).

## 6. What to report back

Per GPU: each correctness page's result; the five ratios (merged, indexed `vec4`, indexed packed, quantized, hit record) with ranges; the Sponza memory rows; compile-time observations. Then Dennis decides the three defaults.

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

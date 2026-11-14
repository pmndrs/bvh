/// <reference types="node" />
/**
 * Turns test-runner `--json` files from the surface-ray pages
 * (`trace-bench-surface-*`) into rows for `bench/results/trace.csv`, in the
 * same columns as `to-csv.ts` (date, commit, device, browser, scene, metric, value).
 *
 *   node tests/gpu/pages/trace-bench/surface-to-csv.ts <runner.json> [more.json ...] [--append] [--out <file>] [--commit <hash>] [--device <text>]
 *
 * Several files of one scene and GPU are repeated runs: the timed rows come
 * out as `run1/`, `run2/`, ... plus `median/` rows; the counted walk's rows and
 * the correctness rows, which are the same in every run, come out once.
 *
 * Metrics (scene is the page's scene; `<set>` is `surface` or `air-grid`,
 * `<mode>` is `closest` or `any`, `<tree>` is `instanced` or `merged`):
 *
 * - `surface-rays/paired/ratio/<set>/<mode>/{median,min,max}_of_rounds`: merged
 *   rate over instanced rate within a round (above 1: the merged tree is faster),
 *   and `.../round<k>` for each round, `.../repeat_at_end` for the repeat;
 * - `surface-rays/paired/<tree>/<set>/<mode>/mrays_per_s_{median,min,max}_of_rounds`:
 *   loaded Mrays/s (back-to-back passes timed by timestamps), and
 *   `single_mrays_per_s_median_of_rounds` for waited-for dispatches;
 * - `surface-rays/paired/hit_record_40_over_24/<tree>/<set>/closest/{median,min,max}_of_rounds`
 *   and `.../round<k>`: the closest-hit rate with the 40-byte hit record (`hitNormal: true`)
 *   over the rate with the 24-byte one, within a round (below 1: the 40-byte record is slower);
 *   `surface-rays/paired/hit_record/<tree>/<set>/closest/{record40,record24}_mrays_per_s_{median,min,max}_of_rounds`
 *   are the rates, and `.../compact_agreement` the share of rays where the two kernels agree;
 * - `surface-rays/paired/indexed_over_records/<tree>/<set>/<mode>/{median,min,max}_of_rounds`
 *   and `indexed_packed_over_records/...` (and `.../round<k>`): indexed leaves (`vec4`, `packed`
 *   positions) over records, the same trees, within a round (above 1: indexed is faster);
 *   `surface-rays/paired/triangle_leaves/<tree>/<set>/<mode>/<layout>_mrays_per_s_*` the rates,
 *   `.../<layout>/agreement_with_records` the share of identical answers (must be 1), and
 *   `surface-rays/triangle_leaves/<layout>/<tree>/bytes_{nodes,triangles,positions,topLevel}`;
 * - `surface-rays/paired/quantized_over_f32/<tree>/<set>/<mode>/{median,min,max}_of_rounds`
 *   (and `.../round<k>`): quantized nodes over f32 nodes, the same trees, within a round (above 1:
 *   quantized is faster); `surface-rays/paired/node_format/<tree>/<set>/<mode>/<format>_mrays_per_s_*`
 *   the rates; `surface-rays/node_format/<tree>/<set>/<mode>/{same,ties,different}` the answers
 *   against f32 (different must be 0), `.../<format>/{steps,mesh_nodes,triangle_tests}` the counted
 *   walk per ray, and `surface-rays/node_format/<format>/<tree>/bytes_nodes`;
 * - `surface-rays/paired/quantized_indexed_over_quantized_records/<tree>/<set>/<mode>/{median,min,max}_of_rounds`
 *   and `quantized_indexed_packed_over_quantized_records/...` (and `.../round<k>`): indexed leaves on
 *   quantized nodes over quantized records (section 8, above 1: indexed is faster);
 *   `surface-rays/paired/combined_leaves/<tree>/<set>/<mode>/quantized-<layout>_mrays_per_s_*` the rates,
 *   `surface-rays/combined_leaves/<tree>/<set>/<mode>/quantized-<layout>/{same,ties,different}` the answers
 *   against quantized records (different must be 0), and `surface-rays/combined_leaves/quantized-<layout>/<tree>/bytes_*`;
 * - `surface-rays/paired/any_{larger,near}_over_far/<tree>/<set>/any/{median,min,max}_of_rounds`
 *   (and `.../round<k>`): any-hit with `anyHitOrder: 'larger'` or `'near'` over the default `'far'`,
 *   within a round (above 1: that order is faster); `surface-rays/paired/any_hit_order/<tree>/<set>/any/<order>_mrays_per_s_*`
 *   the rates, `.../<order>/agreement_with_far` the share of identical answers, and
 *   `surface-rays/any_hit_order/<tree>/<set>/any/<order>/{steps,mesh_nodes,triangle_tests}` the counted walk per ray;
 * - `surface-rays/<tree>/<set>/<mode>/{steps,instances_entered,instances_entered_without_hit,...}`:
 *   the counted walk's mean work per ray;
 * - `surface-rays/<tree>/<set>/<mode>/agreement_with_instanced`: merged tree only;
 * - `surface-rays/<tree>/top_level_instances` and the tree's sizes.
 *
 * Without --append or --out the rows are printed.
 */

import { existsSync } from 'node:fs';
import { appendFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import type { RunSummary } from '../../contract.ts';
import { gitCommit } from '../../measured-commit.ts';
import { browserFromUserAgent, deviceNameFromAdapter, formatValue, repeatedRunRows, type MetricRow } from './csv-rows.ts';
import type { SurfaceRaysData, SurfaceSetResult } from './surface-rays-bench.ts';
import { median } from './statistics.ts';

export const CSV_HEADER = 'date,commit,device,browser,scene,metric,value';
const DEFAULT_CSV = resolve(import.meta.dirname, '..', '..', '..', '..', 'bench', 'results', 'trace.csv');

/** The counted walk's values per set, by metric name. */
const COUNT_METRICS: [string, (set: SurfaceSetResult) => number][] = [
    ['steps', (set) => set.meanSteps],
    ['steps_p90', (set) => set.p90Steps],
    ['instances_entered', (set) => set.meanInstanceEntries],
    ['instances_entered_without_hit', (set) => set.meanInstanceEntriesWithoutHit],
    ['top_level_nodes', (set) => set.meanTopLevelNodes],
    ['mesh_nodes', (set) => set.meanMeshNodes],
    ['triangle_tests', (set) => set.meanTriangleTests],
    ['hit_fraction', (set) => set.hitFraction],
    ['counted_agreement', (set) => set.countedAgreement],
    ['agreement_with_instanced', (set) => (set.hitAgreement ? set.hitAgreement.exact / set.hitAgreement.rays : (set.blockedAgreement ?? Number.NaN))],
];

const middle = (values: readonly number[]): number => median(values);

/** A run's rows: `timed` changes from run to run, `counted` doesn't (the walk's work depends only on the rays and trees). */
export function surfaceRaysRows(data: SurfaceRaysData): { timed: MetricRow[]; counted: MetricRow[] } {
    const scene = data.scene;
    const timed: MetricRow[] = [];
    const counted: MetricRow[] = [
        { scene, metric: 'surface-rays/probes_placed', value: data.probesPlaced },
        { scene, metric: 'surface-rays/grid_probes', value: data.gridProbes },
        { scene, metric: 'surface-rays/instances', value: data.instanceCount },
        { scene, metric: 'surface-rays/paired/rounds', value: data.paired.rounds },
    ];
    for (const [set, rays] of Object.entries(data.raysPerSet)) counted.push({ scene, metric: `surface-rays/rays/${set}`, value: rays });
    for (const [tree, summary] of Object.entries(data.trees)) {
        counted.push(
            { scene, metric: `surface-rays/${tree}/top_level_instances`, value: summary.topLevelInstances },
            { scene, metric: `surface-rays/${tree}/top_level_nodes`, value: summary.topLevelNodes },
            { scene, metric: `surface-rays/${tree}/bytes_nodes`, value: summary.bytesNodes },
            { scene, metric: `surface-rays/${tree}/bytes_triangles`, value: summary.bytesTriangles },
            { scene, metric: `surface-rays/${tree}/bytes_top_level`, value: summary.bytesTopLevel },
        );
        timed.push({ scene, metric: `surface-rays/${tree}/build_ms`, value: summary.buildMs });
    }
    for (const set of data.sets) {
        const prefix = `surface-rays/${set.tree}/${set.set}/${set.mode}`;
        for (const [name, read] of COUNT_METRICS) {
            const value = read(set);
            if (Number.isFinite(value)) counted.push({ scene, metric: `${prefix}/${name}`, value });
        }
    }
    for (const entry of data.paired.entries) {
        const prefix = `surface-rays/paired/${entry.tree}/${entry.set}/${entry.mode}`;
        const loaded = [...entry.mraysPerRound].sort((a, b) => a - b);
        timed.push(
            { scene, metric: `${prefix}/mrays_per_s_median_of_rounds`, value: middle(loaded) },
            { scene, metric: `${prefix}/mrays_per_s_min_of_rounds`, value: loaded[0] },
            { scene, metric: `${prefix}/mrays_per_s_max_of_rounds`, value: loaded[loaded.length - 1] },
            { scene, metric: `${prefix}/single_mrays_per_s_median_of_rounds`, value: middle(entry.singleMraysPerRound) },
        );
        if (Number.isFinite(entry.repeatMrays)) timed.push({ scene, metric: `${prefix}/mrays_per_s_repeat_at_end`, value: entry.repeatMrays });
    }
    for (const ratio of data.paired.ratios) {
        const prefix = `surface-rays/paired/ratio/${ratio.name.replace('merged_over_instanced/', '')}`;
        timed.push(
            { scene, metric: `${prefix}/median_of_rounds`, value: ratio.median },
            { scene, metric: `${prefix}/min_of_rounds`, value: ratio.min },
            { scene, metric: `${prefix}/max_of_rounds`, value: ratio.max },
        );
        ratio.perRound.forEach((value, round) => timed.push({ scene, metric: `${prefix}/round${round + 1}`, value }));
    }
    for (const repeat of data.paired.repeatRatios) {
        timed.push({ scene, metric: `surface-rays/paired/ratio/${repeat.name.replace('merged_over_instanced/', '')}/repeat_at_end`, value: repeat.ratio });
    }
    const hitRecord = data.hitRecord;
    if (hitRecord) {
        for (const entry of hitRecord.entries) {
            const prefix = `surface-rays/paired/hit_record/${entry.tree}/${entry.set}/closest/${entry.record}`;
            const loaded = [...entry.mraysPerRound].sort((a, b) => a - b);
            timed.push(
                { scene, metric: `${prefix}_mrays_per_s_median_of_rounds`, value: middle(loaded) },
                { scene, metric: `${prefix}_mrays_per_s_min_of_rounds`, value: loaded[0] },
                { scene, metric: `${prefix}_mrays_per_s_max_of_rounds`, value: loaded[loaded.length - 1] },
            );
        }
        for (const ratio of hitRecord.ratios) {
            const prefix = `surface-rays/paired/${ratio.name}`;
            timed.push(
                { scene, metric: `${prefix}/median_of_rounds`, value: ratio.median },
                { scene, metric: `${prefix}/min_of_rounds`, value: ratio.min },
                { scene, metric: `${prefix}/max_of_rounds`, value: ratio.max },
            );
            ratio.perRound.forEach((value, round) => timed.push({ scene, metric: `${prefix}/round${round + 1}`, value }));
        }
        for (const entry of hitRecord.agreement) counted.push({ scene, metric: `surface-rays/paired/hit_record/${entry.tree}/${entry.set}/closest/compact_agreement`, value: entry.same / entry.rays });
    }
    const leaves = data.triangleLeaves;
    if (leaves) {
        for (const entry of leaves.entries) {
            const prefix = `surface-rays/paired/triangle_leaves/${entry.tree}/${entry.set}/${entry.mode}/${entry.layout}`;
            const loaded = [...entry.mraysPerRound].sort((a, b) => a - b);
            timed.push(
                { scene, metric: `${prefix}_mrays_per_s_median_of_rounds`, value: middle(loaded) },
                { scene, metric: `${prefix}_mrays_per_s_min_of_rounds`, value: loaded[0] },
                { scene, metric: `${prefix}_mrays_per_s_max_of_rounds`, value: loaded[loaded.length - 1] },
            );
        }
        for (const ratio of leaves.ratios) {
            const prefix = `surface-rays/paired/${ratio.name}`;
            timed.push(
                { scene, metric: `${prefix}/median_of_rounds`, value: ratio.median },
                { scene, metric: `${prefix}/min_of_rounds`, value: ratio.min },
                { scene, metric: `${prefix}/max_of_rounds`, value: ratio.max },
            );
            ratio.perRound.forEach((value, round) => timed.push({ scene, metric: `${prefix}/round${round + 1}`, value }));
        }
        for (const entry of leaves.agreement) counted.push({ scene, metric: `surface-rays/paired/triangle_leaves/${entry.tree}/${entry.set}/${entry.mode}/${entry.layout}/agreement_with_records`, value: entry.same / entry.rays });
        for (const [name, bytes] of Object.entries(leaves.bytes)) {
            for (const [part, value] of Object.entries(bytes)) counted.push({ scene, metric: `surface-rays/triangle_leaves/${name}/bytes_${part}`, value });
        }
    }
    const nodes = data.nodeFormat;
    if (nodes) {
        for (const entry of nodes.entries) {
            const prefix = `surface-rays/paired/node_format/${entry.tree}/${entry.set}/${entry.mode}/${entry.layout}`;
            const loaded = [...entry.mraysPerRound].sort((a, b) => a - b);
            timed.push(
                { scene, metric: `${prefix}_mrays_per_s_median_of_rounds`, value: middle(loaded) },
                { scene, metric: `${prefix}_mrays_per_s_min_of_rounds`, value: loaded[0] },
                { scene, metric: `${prefix}_mrays_per_s_max_of_rounds`, value: loaded[loaded.length - 1] },
            );
        }
        for (const ratio of nodes.ratios) {
            const prefix = `surface-rays/paired/${ratio.name}`;
            timed.push(
                { scene, metric: `${prefix}/median_of_rounds`, value: ratio.median },
                { scene, metric: `${prefix}/min_of_rounds`, value: ratio.min },
                { scene, metric: `${prefix}/max_of_rounds`, value: ratio.max },
            );
            ratio.perRound.forEach((value, round) => timed.push({ scene, metric: `${prefix}/round${round + 1}`, value }));
        }
        for (const entry of nodes.agreement) {
            const prefix = `surface-rays/node_format/${entry.tree}/${entry.set}/${entry.mode}`;
            counted.push({ scene, metric: `${prefix}/same`, value: entry.same }, { scene, metric: `${prefix}/ties`, value: entry.ties }, { scene, metric: `${prefix}/different`, value: entry.different });
        }
        for (const entry of nodes.counts) {
            for (const format of ['f32', 'quantized'] as const) {
                const prefix = `surface-rays/node_format/${entry.tree}/${entry.set}/${entry.mode}/${format}`;
                const summary = entry[format];
                counted.push(
                    { scene, metric: `${prefix}/steps`, value: summary.meanSteps },
                    { scene, metric: `${prefix}/mesh_nodes`, value: summary.meanMeshNodes },
                    { scene, metric: `${prefix}/triangle_tests`, value: summary.meanTriangleTests },
                );
            }
        }
        for (const [name, value] of Object.entries(nodes.bytes)) {
            const [format, tree] = name.split('/');
            counted.push({ scene, metric: `surface-rays/node_format/${format}/${tree}/bytes_nodes`, value });
        }
    }
    const combined = data.combinedLeaves;
    if (combined) {
        for (const entry of combined.entries) {
            const prefix = `surface-rays/paired/combined_leaves/${entry.tree}/${entry.set}/${entry.mode}/quantized-${entry.layout}`;
            const loaded = [...entry.mraysPerRound].sort((a, b) => a - b);
            timed.push(
                { scene, metric: `${prefix}_mrays_per_s_median_of_rounds`, value: middle(loaded) },
                { scene, metric: `${prefix}_mrays_per_s_min_of_rounds`, value: loaded[0] },
                { scene, metric: `${prefix}_mrays_per_s_max_of_rounds`, value: loaded[loaded.length - 1] },
            );
        }
        for (const ratio of combined.ratios) {
            const prefix = `surface-rays/paired/${ratio.name}`;
            timed.push(
                { scene, metric: `${prefix}/median_of_rounds`, value: ratio.median },
                { scene, metric: `${prefix}/min_of_rounds`, value: ratio.min },
                { scene, metric: `${prefix}/max_of_rounds`, value: ratio.max },
            );
            ratio.perRound.forEach((value, round) => timed.push({ scene, metric: `${prefix}/round${round + 1}`, value }));
        }
        for (const entry of combined.agreement) {
            const prefix = `surface-rays/combined_leaves/${entry.tree}/${entry.set}/${entry.mode}/quantized-${entry.layout}`;
            counted.push({ scene, metric: `${prefix}/same`, value: entry.same }, { scene, metric: `${prefix}/ties`, value: entry.ties }, { scene, metric: `${prefix}/different`, value: entry.different });
        }
        for (const [name, bytes] of Object.entries(combined.bytes)) {
            for (const [part, value] of Object.entries(bytes)) counted.push({ scene, metric: `surface-rays/combined_leaves/${name}/bytes_${part}`, value });
        }
    }
    const orders = data.anyHitOrder;
    if (orders) {
        for (const entry of orders.entries) {
            const prefix = `surface-rays/paired/any_hit_order/${entry.tree}/${entry.set}/any/${entry.order}`;
            const loaded = [...entry.mraysPerRound].sort((a, b) => a - b);
            timed.push(
                { scene, metric: `${prefix}_mrays_per_s_median_of_rounds`, value: middle(loaded) },
                { scene, metric: `${prefix}_mrays_per_s_min_of_rounds`, value: loaded[0] },
                { scene, metric: `${prefix}_mrays_per_s_max_of_rounds`, value: loaded[loaded.length - 1] },
            );
        }
        for (const ratio of orders.ratios) {
            const prefix = `surface-rays/paired/${ratio.name}`;
            timed.push(
                { scene, metric: `${prefix}/median_of_rounds`, value: ratio.median },
                { scene, metric: `${prefix}/min_of_rounds`, value: ratio.min },
                { scene, metric: `${prefix}/max_of_rounds`, value: ratio.max },
            );
            ratio.perRound.forEach((value, round) => timed.push({ scene, metric: `${prefix}/round${round + 1}`, value }));
        }
        for (const entry of orders.agreement) counted.push({ scene, metric: `surface-rays/paired/any_hit_order/${entry.tree}/${entry.set}/any/${entry.order}/agreement_with_far`, value: entry.same / entry.rays });
        for (const entry of orders.counts) {
            const prefix = `surface-rays/any_hit_order/${entry.tree}/${entry.set}/any/${entry.order}`;
            counted.push(
                { scene, metric: `${prefix}/steps`, value: entry.counts.meanSteps },
                { scene, metric: `${prefix}/mesh_nodes`, value: entry.counts.meanMeshNodes },
                { scene, metric: `${prefix}/triangle_tests`, value: entry.counts.meanTriangleTests },
            );
        }
    }
    return { timed, counted };
}

async function main(): Promise<void> {
    const { values, positionals } = parseArgs({
        args: process.argv.slice(2),
        allowPositionals: true,
        options: { append: { type: 'boolean' }, out: { type: 'string' }, device: { type: 'string' }, commit: { type: 'string' } },
    });
    if (positionals.length === 0) {
        console.error('Usage: node tests/gpu/pages/trace-bench/surface-to-csv.ts <runner.json> [more.json ...] [--append] [--out <file>] [--device <text>] [--commit <hash>]');
        process.exit(2);
    }
    const groups = new Map<string, { prefix: string[]; runs: MetricRow[][]; counted: MetricRow[] }>();
    for (const file of positionals) {
        const summary = JSON.parse(await readFile(file, 'utf8')) as RunSummary;
        const device = values.device ?? deviceNameFromAdapter(summary.adapter);
        for (const page of summary.pages) {
            const data = page.data as unknown as SurfaceRaysData;
            if (data.benchmark !== 'surface-rays') continue;
            if (page.outcome !== 'pass') {
                console.error(`skipping ${page.name} in ${file}: ${page.outcome}`);
                continue;
            }
            const commit = values.commit ?? summary.commit ?? gitCommit();
            const prefix = [summary.startedAt.slice(0, 10), commit, device, browserFromUserAgent(data.userAgent, data.browserVersion)];
            const key = `${device}|${data.scene}`;
            const rows = surfaceRaysRows(data);
            const group = groups.get(key) ?? { prefix, runs: [], counted: rows.counted };
            group.runs.push(rows.timed);
            groups.set(key, group);
        }
    }
    const lines: string[] = [];
    for (const { prefix, runs, counted } of groups.values()) {
        const rows = [...counted, ...(runs.length === 1 ? runs[0] : repeatedRunRows(runs))];
        lines.push(...rows.map((row) => [...prefix, row.scene, row.metric, formatValue(row.value)].join(',')));
    }
    if (lines.length === 0) {
        console.error('no passing surface-ray pages in those files');
        process.exit(1);
    }
    if (!values.append && !values.out) {
        console.log([CSV_HEADER, ...lines].join('\n'));
        return;
    }
    const target = values.out ? resolve(values.out) : DEFAULT_CSV;
    await mkdir(dirname(target), { recursive: true });
    if (!existsSync(target)) await writeFile(target, `${CSV_HEADER}\n`);
    await appendFile(target, `${lines.join('\n')}\n`);
    console.error(`added ${lines.length} rows to ${target}`);
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(import.meta.filename)) {
    main().catch((error: unknown) => {
        console.error(error instanceof Error ? error.message : error);
        process.exit(1);
    });
}

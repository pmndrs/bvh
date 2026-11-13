/// <reference types="node" />
/**
 * Summarises the wide-node section (section 8) of `trace-bench-surface-*` runner `--json`
 * files (docs/internals/design/wide-nodes.md): per scene, format, tree, ray set and mode,
 * the wide-over-f32 ratio as min..max of the runs' medians (and the single-loop BVH4 walk
 * over the nested one, lever L7), the runs' machine load, and how many answers differed
 * from f32 (must be 0; ties apart).
 *
 *   node --experimental-transform-types tools/wide-summary.ts run1.json run2.json run3.json [more ...] [--markdown]
 *
 * Files of several scenes may be mixed; each scene is summarised on its own.
 */

import { readFileSync } from 'node:fs';
import type { RunSummary } from '../tests/gpu/contract.ts';
import type { SurfaceRaysData } from '../tests/gpu/pages/trace-bench/surface-rays-bench.ts';

const args = process.argv.slice(2);
const markdown = args.includes('--markdown');
const files = args.filter((arg) => !arg.startsWith('--'));
if (files.length === 0) {
    console.error('usage: node --experimental-transform-types tools/wide-summary.ts <runner.json> [more ...] [--markdown]');
    process.exit(2);
}

interface SceneRuns {
    ratios: Map<string, number[]>;
    loads: string[];
    quiet: boolean[];
    different: number;
    ties: number;
    runs: number;
}

const scenes = new Map<string, SceneRuns>();
const middle = (values: readonly number[]): number => [...values].sort((a, b) => a - b)[values.length >> 1];

for (const file of files) {
    const summary = JSON.parse(readFileSync(file, 'utf8')) as RunSummary;
    for (const page of summary.pages) {
        const data = page.data as unknown as SurfaceRaysData;
        if (data?.benchmark !== 'surface-rays' || !data.wideNodes) continue;
        if (page.outcome !== 'pass') {
            console.error(`skipping ${page.name} in ${file}: ${page.outcome}`);
            continue;
        }
        const scene = scenes.get(data.scene) ?? { ratios: new Map(), loads: [], quiet: [], different: 0, ties: 0, runs: 0 };
        scenes.set(data.scene, scene);
        scene.runs++;
        const wide = data.wideNodes;
        for (const ratio of wide.ratios) {
            const list = scene.ratios.get(ratio.name) ?? [];
            list.push(ratio.median);
            scene.ratios.set(ratio.name, list);
        }
        // L7: the single-loop BVH4 walk over the nested one, per round, then the median.
        const entry = (tree: string, set: string, mode: string, layout: string) => wide.entries.find((e) => e.tree === tree && e.set === set && e.mode === mode && e.layout === layout);
        for (const e of wide.entries) {
            if (e.layout !== 'bvh4-single') continue;
            const nested = entry(e.tree, e.set, e.mode, 'bvh4');
            if (!nested) continue;
            const name = `l7_bvh4-single_over_bvh4/${e.tree}/${e.set}/${e.mode}`;
            const list = scene.ratios.get(name) ?? [];
            list.push(middle(e.mraysPerRound.map((rate, round) => rate / nested.mraysPerRound[round])));
            scene.ratios.set(name, list);
        }
        for (const a of wide.agreement) {
            scene.different += a.different;
            scene.ties += a.ties;
        }
        const load = page.machineLoad;
        scene.loads.push(load ? `${load.before.loadAverage[0].toFixed(1)}/${load.after.loadAverage[0].toFixed(1)} (${load.before.chromeGpuProcesses} GPU procs)` : 'not recorded');
        scene.quiet.push(load?.quiet ?? false);
    }
}

for (const [name, scene] of scenes) {
    console.log(`\n${name}: ${scene.runs} runs; load 1m before/after: ${scene.loads.join(', ')}; quiet: ${scene.quiet.map((q) => (q ? 'yes' : 'NO')).join(', ')}`);
    console.log(`  answers differing from f32: ${scene.different} (must be 0); ties: ${scene.ties}`);
    const rows = [...scene.ratios.entries()].sort(([a], [b]) => a.localeCompare(b));
    if (markdown) {
        console.log('\n| ratio | min..max of run medians |\n| --- | --- |');
        for (const [ratio, values] of rows) console.log(`| ${ratio} | ${Math.min(...values).toFixed(2)}..${Math.max(...values).toFixed(2)} |`);
    } else {
        for (const [ratio, values] of rows) console.log(`  ${ratio}: ${Math.min(...values).toFixed(3)}..${Math.max(...values).toFixed(3)}`);
    }
}
if ([...scenes.values()].some((scene) => scene.different > 0)) process.exit(1);

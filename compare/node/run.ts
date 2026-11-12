/// <reference types="node" />
/**
 * `pnpm compare:node`: the CPU comparisons headless, written to
 * bench/results/compare.csv (trace.csv's columns plus `method`) and printed as
 * markdown tables.
 *
 *   BVH_ASSETS_DIR=... BVH_MANIFESTS_DIR=... pnpm compare:node [--scenes torus-knot,furnished-room,sponza] [--quick] [--no-memory] [--no-scene] [--append]
 *
 * Per scene (merged into one world-space mesh): every library built three
 * times (median), then CPU raycasts (first and all hits, outside and inside
 * rays), box, sphere and closest-point queries, refits, tree quality, and
 * memory (each library in a fresh process, `memory.ts`). Then the scene-level
 * trees over 10,000 objects. Every answer is checked against ours.
 *
 * Don't run it while something else loads the machine (builds, GPU pages):
 * the numbers are wall-clock times.
 */

import { execFileSync } from 'node:child_process';
import { appendFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { cpus } from 'node:os';
import { dirname, resolve } from 'node:path';
import { CpuMeshBvh } from '@pmndrs/bvh';
import { MESH_ADAPTERS } from '../adapters/index.ts';
import { oursBuilders, oursQuality } from '../adapters/ours.ts';
import { SCENE_ADAPTERS } from '../adapters/scene.ts';
import { median } from '../core/measure.ts';
import { csvRows, CSV_HEADER, formatValue, table, type RunContext } from '../core/report.ts';
import { makeScene } from '../core/scene.ts';
import { sceneSuite } from '../core/scene-suite.ts';
import { buildAll, queriesSuite, raycastSuite, refitSuite, type Measurement } from '../core/suites.ts';
import { loadScene, SCENES } from './scenes.ts';

const ROOT = resolve(import.meta.dirname, '../..');
const args = process.argv.slice(2);
const flag = (name: string) => args.includes(`--${name}`);
const option = (name: string, fallback: string) => {
    const at = args.indexOf(`--${name}`);
    return at >= 0 ? args[at + 1] : fallback;
};
const quick = flag('quick');
const scenes = option('scenes', SCENES.join(',')).split(',');
const outFile = resolve(ROOT, option('out', 'bench/results/compare.csv'));

const git = (...a: string[]) => execFileSync('git', a, { cwd: ROOT, encoding: 'utf8' }).trim();
const dirty = git('status', '--porcelain', '--', 'src', 'compare', 'pnpm-lock.yaml') !== '';
const context: RunContext = {
    date: new Date().toISOString().slice(0, 10),
    commit: `${git('rev-parse', '--short', 'HEAD')}${dirty ? '-dirty' : ''}`,
    device: cpus()[0]?.model ?? 'unknown',
    browser: `Node ${process.version}`,
};
const NAMES: Record<string, string> = Object.fromEntries([...MESH_ADAPTERS, ...SCENE_ADAPTERS.filter((a) => a.id !== 'ours')].map((a) => [a.id, a.name]));
NAMES['ours-hploc'] = '@pmndrs/bvh H-PLOC (CPU twin)';
NAMES['ours-lbvh'] = '@pmndrs/bvh LBVH (CPU twin)';

const log = (line: string) => console.error(`  ${line}`);
const plan = quick ? { runs: 3, budgetMs: 400 } : { runs: 5, budgetMs: 1500 };
const all: Measurement[] = [];
const output: string[] = [`# compare:node, ${context.date}, ${context.commit}, ${context.device}, ${context.browser} (indicative)`];
const print = (text: string) => {
    output.push(text);
    console.log(text);
};

function memoryInFreshProcess(adapterId: string, sceneId: string): { total: number; reported: number | null } | null {
    try {
        const text = execFileSync(process.execPath, ['--expose-gc', '--max-old-space-size=24000', '--experimental-transform-types', '--no-warnings', '--import', './compare/node/register.mjs', 'compare/node/memory.ts', adapterId, sceneId], { cwd: ROOT, encoding: 'utf8', env: process.env, stdio: ['ignore', 'pipe', 'pipe'] });
        const line = text.split('\n').find((l) => l.startsWith('{'));
        return line ? (JSON.parse(line) as { total: number; reported: number | null }) : null;
    } catch (error) {
        log(`memory ${adapterId} ${sceneId}: ${(error as Error).message.split('\n')[0]}`);
        return null;
    }
}

for (const sceneId of scenes) {
    console.error(`${sceneId}: loading`);
    const geometry = loadScene(sceneId);
    print(`\n## ${sceneId}: ${geometry.triangleCount.toLocaleString('en-US')} triangles, ${(geometry.positions.length / 3).toLocaleString('en-US')} vertices`);
    const { built, measurements } = await buildAll(MESH_ADAPTERS, geometry, { log, buildRuns: quick ? 1 : 3 });
    const results = [...measurements];
    results.push(...(await raycastSuite(built, geometry, { log, rays: quick ? 1024 : 4096, plan })));
    results.push(...(await queriesSuite(built, geometry, { log, queries: quick ? 512 : 2048, plan })));
    results.push(...(await refitSuite(built, geometry)));
    for (const { mesh } of built) mesh.dispose();

    // Our other builders' trees (the GPU builders' CPU twins), for tree quality and build time.
    for (const builder of oursBuilders) {
        const samples: number[] = [];
        let tree: CpuMeshBvh | null = null;
        for (let r = 0; r < (quick ? 1 : 3); r++) {
            const t0 = performance.now();
            const layout = builder.build(geometry);
            samples.push(performance.now() - t0);
            tree = new CpuMeshBvh(layout);
        }
        const quality = oursQuality(tree!);
        const method = 'the CPU twin of the GPU builder (the same tree the GPU writes), its SAH etc. as for ours';
        results.push({ page: 'build', scene: sceneId, library: builder.id, metric: 'build', value: median(samples), unit: 'ms', method: `median of ${samples.length} CPU builds of the twin (TypeScript; the GPU builder itself is timed on the GPU pages)`, ranked: true });
        for (const [metric, value, unit] of [['sah', quality.sah, 'relative'], ['depth', quality.depth, 'levels'], ['nodes', quality.nodes, 'boxes'], ['leaves', quality.leaves, 'leaves']] as const) results.push({ page: 'tree-quality', scene: sceneId, library: builder.id, metric, value, unit, method, ranked: true });
        results.push({ page: 'build', scene: sceneId, library: builder.id, metric: 'memory', value: quality.bytes, unit: 'bytes', method: 'CpuMeshBvh.estimateMemoryInBytes of the twin\'s tree', ranked: true });
    }

    if (!flag('no-memory')) {
        for (const adapter of MESH_ADAPTERS) {
            const memory = memoryInFreshProcess(adapter.id, sceneId);
            results.push({ page: 'build', scene: sceneId, library: adapter.id, metric: 'heap', value: memory?.total ?? null, unit: 'bytes', method: 'fresh Node process: growth of heapUsed + external (array buffers, WASM memory) across one build, after gc, the structure alive; geometry copies the adapter makes for the library taken off', ranked: memory !== null });
            log(`${adapter.id}: heap growth ${memory ? formatValue(memory.total, 'bytes') : 'n/a'}`);
        }
    }
    all.push(...results);
    const scene = results.filter((m) => m.scene === sceneId);
    print(`\n### Build, refit and memory\n\n${table(scene.filter((m) => m.page === 'build'), ['build', 'refit', 'memory', 'heap', 'refit-agreement'], NAMES)}`);
    print(`\n### Tree quality\n\n${table(scene.filter((m) => m.page === 'tree-quality'), ['sah', 'depth', 'nodes', 'leaves'], NAMES)}`);
    print(`\n### CPU raycasts (rays/s)\n\n${table(scene.filter((m) => m.page === 'raycast'), ['first/outside', 'first/inside', 'all/outside', 'all/inside'], NAMES)}`);
    print(`\n### Queries (queries/s)\n\n${table(scene.filter((m) => m.page === 'queries'), ['box', 'sphere', 'closest-point'], NAMES)}`);
    const disagreements = scene.filter((m) => m.agreement && m.agreement.disagree > 0);
    if (disagreements.length) print(`\nDisagreements (first examples):\n${disagreements.map((m) => `- ${NAMES[m.library] ?? m.library} ${m.metric}: ${m.agreement!.examples.slice(0, 3).join('; ')}`).join('\n')}`);
}

if (!flag('no-scene')) {
    for (const count of quick ? [10_000] : [10_000, 100_000]) {
        const scene = makeScene(count);
        console.error(`scene: ${scene.name}`);
        const { measurements, built } = await sceneSuite(SCENE_ADAPTERS, scene, { log, queries: quick ? 256 : 1024, frames: quick ? 5 : 20, plan });
        for (const entry of built) entry.scene.dispose();
        all.push(...measurements);
        const names = { ...NAMES, ours: '@pmndrs/bvh (CpuSceneBvh)', tmbvh: 'three-mesh-bvh ObjectBVH' };
        print(`\n## Scene-level trees: ${scene.name} (one 1k-triangle mesh each)\n\n${table(measurements, ['build', 'memory', 'update', 'frustum', 'box', 'raycast', 'frustum-after-motion'], names)}`);
    }
}

mkdirSync(dirname(outFile), { recursive: true });
const rows = csvRows(context, all);
if (flag('append') && existsSync(outFile)) appendFileSync(outFile, `${rows.join('\n')}\n`);
else writeFileSync(outFile, `${CSV_HEADER}\n${rows.join('\n')}\n`);
writeFileSync(outFile.replace(/\.csv$/, '.md'), `${output.join('\n')}\n`);
console.error(`${rows.length} rows -> ${outFile}`);

/**
 * compare/build: build time, refit time and memory beyond the geometry for
 * every library; main-thread stalls of a build on the main thread against a
 * build in a worker (ours: the tinybvh worker pool behind
 * `GenerateMeshBvhWorker`; three-mesh-bvh: its `GenerateMeshBVHWorker`).
 * What else: a GPU H-PLOC build of the same mesh on the renderer's device
 * (`SceneTrees`), which no other library here has.
 */

import { BufferAttribute, BufferGeometry, Group } from 'three';
import { CENTER, MeshBVH } from 'three-mesh-bvh';
import { GenerateMeshBVHWorker } from 'three-mesh-bvh/worker';
import { SceneTrees } from '@pmndrs/bvh';
import { GenerateMeshBvhWorker, prepareBoundsTrees, SceneTreesObject3DSync, computeBoundsTree } from '@pmndrs/bvh/three';
import { MESH_ADAPTERS } from '../../../compare/adapters/index.ts';
import { buildAll, refitSuite, type Measurement } from '../../../compare/core/suites.ts';
import { median } from '../../../compare/core/measure.ts';
import { createComparePage, expectAgreement, quick } from '../../shared/compare.ts';
import { SceneTreesLog } from '../../shared/stats.ts';
import { ready } from '../../shared/smoke.ts';

const page = await createComparePage('compare/build', 'compare/build', 'Build and refit times on this thread (median of 3), memory beyond the app\'s geometry, and the main thread\'s longest frame while a build runs on it or in a worker.');
const { app, geometry, panel, sceneName } = page;
const log = (line: string) => panel.status(line);

const adapters = MESH_ADAPTERS.filter((adapter) => adapter.id !== 'three' && !(quick && adapter.id === 'babylon-brute'));
const { built, measurements } = await buildAll(adapters, geometry, { log, buildRuns: quick ? 1 : 3 });
const refits = await refitSuite(built, geometry);
for (const { mesh } of built) mesh.dispose();
panel.table('Build, refit and memory beyond the geometry', [...measurements, ...refits], ['build', 'refit', 'memory', 'refit-agreement']);
panel.paragraph('Memory is each library\'s own count where it has one (hover a cell for how). JS-object libraries and Babylon are measured by heap growth in the Node runner (pnpm compare:node), which this page can\'t do.');

/** Runs `work` while counting the main thread's longest gap between animation frames. */
async function withStalls<T>(work: () => Promise<T>): Promise<{ value: T; wallMs: number; worstFrameMs: number }> {
    let last = performance.now();
    let worst = 0;
    let running = true;
    const tick = () => {
        const now = performance.now();
        worst = Math.max(worst, now - last);
        last = now;
        if (running) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    await new Promise((resolve) => requestAnimationFrame(resolve));
    const start = performance.now();
    const value = await work();
    const wallMs = performance.now() - start;
    await new Promise((resolve) => requestAnimationFrame(resolve));
    running = false;
    worst = Math.max(worst, performance.now() - last);
    return { value, wallMs, worstFrameMs: worst };
}

const threeGeometry = () => {
    const out = new BufferGeometry();
    out.setAttribute('position', new BufferAttribute(geometry.positions.slice(), 3));
    out.setIndex(new BufferAttribute(geometry.indices.slice(), 1));
    return out;
};

await prepareBoundsTrees();
const ourWorker = new GenerateMeshBvhWorker();
const theirWorker = new GenerateMeshBVHWorker();
const stallRows: Measurement[] = [];
const runs = quick ? 1 : 3;
const cases: { library: string; metric: string; how: string; run: (g: BufferGeometry) => Promise<unknown> }[] = [
    { library: 'ours', metric: 'main-thread', how: 'computeBoundsTree on the main thread (tinybvh)', run: async (g) => computeBoundsTree.call(g) },
    { library: 'ours', metric: 'worker', how: 'GenerateMeshBvhWorker.generate (the tinybvh worker pool)', run: (g) => ourWorker.generate(g) },
    { library: 'tmbvh-center', metric: 'main-thread', how: 'new MeshBVH(geometry) on the main thread (CENTER, its default)', run: async (g) => new MeshBVH(g, { strategy: CENTER }) },
    { library: 'tmbvh-center', metric: 'worker', how: 'GenerateMeshBVHWorker.generate (one worker, CENTER)', run: (g) => theirWorker.generate(g, { strategy: CENTER }) },
];
// A first worker build of each spins its worker up; not timed.
await ourWorker.generate(threeGeometry());
await theirWorker.generate(threeGeometry());
for (const c of cases) {
    const walls: number[] = [];
    const worst: number[] = [];
    for (let r = 0; r < runs; r++) {
        const g = threeGeometry();
        const result = await withStalls(() => c.run(g));
        walls.push(result.wallMs);
        worst.push(result.worstFrameMs);
    }
    const method = `${c.how}; median of ${runs}, wall clock from the call to the tree, and the longest gap between animation frames meanwhile`;
    stallRows.push({ page: 'build', scene: sceneName, library: c.library, metric: `${c.metric} wall`, value: median(walls), unit: 'ms', method, ranked: true });
    stallRows.push({ page: 'build', scene: sceneName, library: c.library, metric: `${c.metric} longest frame`, value: median(worst), unit: 'ms', method, ranked: true });
    log(`${c.library} ${c.metric}: ${median(walls).toFixed(1)} ms, longest frame ${median(worst).toFixed(1)} ms`);
}
ourWorker.dispose();
theirWorker.dispose();
panel.table('Main-thread stalls: on the main thread against in a worker', stallRows, ['main-thread wall', 'main-thread longest frame', 'worker wall', 'worker longest frame']);

// What else: the same mesh built on the GPU (H-PLOC), on the renderer's device.
const treesLog = new SceneTreesLog();
const trees = await SceneTrees.create(app.device, { label: 'compare build', onTree: treesLog.onTree, holdRefinements: true });
const root = new Group();
root.add(page.mesh);
app.scene.add(root);
const sync = new SceneTreesObject3DSync(trees, root);
const gpuStart = performance.now();
sync.update();
const encoder = app.device.createCommandEncoder();
trees.update(encoder);
app.device.queue.submit([encoder.finish()]);
await trees.whenBuilt();
const gpuWall = performance.now() - gpuStart;
// The next update places the built tree (counts and builtBy then say so).
sync.update();
const placeEncoder = app.device.createCommandEncoder();
trees.update(placeEncoder);
app.device.queue.submit([placeEncoder.finish()]);
const info = treesLog.build(trees);
app.overlay.build(info);
panel.whatElse(
    `The same ${geometry.triangleCount.toLocaleString('en-US')} triangles built on the GPU by our H-PLOC builder (SceneTrees, first trees): ${info.builtBy}, ${gpuWall.toFixed(1)} ms from the call to a traceable tree (GPU batches ${info.gpuMs?.toFixed(1) ?? '-'} ms, submit to readback). No other library here builds on the GPU; a GPU-built tree is refitted on the GPU too, and the tinybvh tree from the worker pool swaps in later.`,
);
panel.csv(app.device);

const checked = expectAgreement(refits, ['tmbvh-center', 'tmbvh-sah'], log);
let frames = 0;
app.loop(async () => {
    if (++frames === 10) await ready(app.device, { ...app.overlay.snapshot(), csv: panel.csvText, status: panel.statusText, checkedAgainstOurs: checked, gpuBuiltBy: info.builtBy });
});

/**
 * compare/tree-quality: SAH cost (three-mesh-bvh's costs, over the root box's
 * area), depth, boxes and leaves, and bytes, for every library that has a tree
 * to read: ours (tinybvh, and the CPU twins of the GPU H-PLOC and LBVH
 * builders), three-mesh-bvh (CENTER, AVERAGE, SAH) and bvh-tree. Babylon's
 * octree and Rapier's WASM tree have no shape to read here. The overlay draws
 * either library's boxes at a depth (live toggle).
 */

import { BufferAttribute, BufferGeometry } from 'three';
import { AVERAGE, MeshBVH, MeshBVHHelper as TmbvhHelper } from 'three-mesh-bvh';
import { CpuMeshBvh } from '@pmndrs/bvh';
import { BvhBoundsTree, MeshBVHHelper, prepareBoundsTrees } from '@pmndrs/bvh/three';
import { MESH_ADAPTERS } from '../../../compare/adapters/index.ts';
import { oursBuilders, oursQuality } from '../../../compare/adapters/ours.ts';
import { tmbvhQuality } from '../../../compare/adapters/three-mesh-bvh.ts';
import { median } from '../../../compare/core/measure.ts';
import { buildAll, type Measurement } from '../../../compare/core/suites.ts';
import { createComparePage, quick } from '../../shared/compare.ts';
import { ready } from '../../shared/smoke.ts';

const page = await createComparePage('compare/tree-quality', 'compare/tree-quality', 'The shape of each tree on the same mesh: SAH cost with three-mesh-bvh\'s costs (traversal 1, triangle 1.25) over the root box\'s area, lower is better; depth; boxes and leaves; bytes.');
const { app, geometry, panel, sceneName } = page;
const log = (line: string) => panel.status(line);

const adapters = MESH_ADAPTERS.filter((adapter) => adapter.features.treeStats.status !== 'no');
const { built, measurements } = await buildAll(adapters, geometry, { log, buildRuns: 1 });
const rows: Measurement[] = measurements.filter((m) => m.page === 'tree-quality' || m.metric === 'memory' || m.metric === 'build');

const push = (library: string, quality: ReturnType<typeof oursQuality>, buildMs: number, how: string) => {
    for (const [metric, value, unit] of [['sah', quality.sah, 'relative'], ['depth', quality.depth, 'levels'], ['nodes', quality.nodes, 'boxes'], ['leaves', quality.leaves, 'leaves'], ['memory', quality.bytes, 'bytes'], ['build', buildMs, 'ms']] as const) {
        rows.push({ page: 'tree-quality', scene: sceneName, library, metric, value, unit, method: how, ranked: true });
    }
};
for (const builder of oursBuilders) {
    const samples: number[] = [];
    let layout = builder.build(geometry);
    for (let r = 0; r < (quick ? 1 : 3); r++) {
        const t0 = performance.now();
        layout = builder.build(geometry);
        samples.push(performance.now() - t0);
    }
    push(builder.id, oursQuality(new CpuMeshBvh(layout)), median(samples), 'the CPU twin of our GPU builder: the tree the GPU writes, built in TypeScript (its GPU build time is on compare/build)');
}
{
    const three = new BufferGeometry();
    three.setAttribute('position', new BufferAttribute(geometry.positions, 3));
    three.setIndex(new BufferAttribute(geometry.indices.slice(), 1));
    const t0 = performance.now();
    const bvh = new MeshBVH(three, { strategy: AVERAGE });
    push('tmbvh-average', tmbvhQuality(bvh), performance.now() - t0, 'new MeshBVH(geometry, { strategy: AVERAGE })');
}
const names = { 'tmbvh-average': 'three-mesh-bvh (AVERAGE)', 'ours-hploc': '@pmndrs/bvh H-PLOC (CPU twin)', 'ours-lbvh': '@pmndrs/bvh LBVH (CPU twin)', ...Object.fromEntries(MESH_ADAPTERS.map((a) => [a.id, a.name])) };
panel.table('Tree quality', rows, ['sah', 'depth', 'nodes', 'leaves', 'memory', 'build'], names);
panel.paragraph('Boxes: three-mesh-bvh counts boxes, one per node; our 64-byte nodes hold two child boxes, so the box count is what is compared. Our leaves hold about 2 triangles (tinybvh\'s SAH), three-mesh-bvh\'s about 7 (target 10): smaller leaves cost bytes and walk faster. Babylon (an octree of submeshes) and Rapier (parry\'s tree inside WASM) have no tree to read.');
for (const { mesh } of built) mesh.dispose();

// Live: either library's boxes at a depth.
await prepareBoundsTrees();
const oursTree = new BvhBoundsTree(page.mesh.geometry);
const theirGeometry = page.mesh.geometry.clone();
const theirs = new MeshBVH(theirGeometry);
// The page's mesh sits at the origin, so the trees' own helpers (no mesh) line up with it.
const ourHelper = new MeshBVHHelper(oursTree, 8);
const theirHelper = new TmbvhHelper(theirs, 8);
app.scene.add(ourHelper, theirHelper);
let shown: 'ours' | 'three-mesh-bvh' = 'ours';
const show = () => {
    ourHelper.visible = shown === 'ours';
    theirHelper.visible = shown === 'three-mesh-bvh';
};
show();
app.overlay.select('boxes of', ['ours', 'three-mesh-bvh'] as const, shown, (value) => ((shown = value), show()));
app.overlay.range('depth', 1, 20, 1, 8, (depth) => {
    ourHelper.depth = depth;
    theirHelper.depth = depth;
    ourHelper.update();
    theirHelper.update();
});
panel.whatElse('Our trees come from three builders that write one layout: tinybvh on the CPU (or in workers), and H-PLOC and LBVH on the GPU, whose CPU twins are measured above. A GPU tree can stand in first and the tinybvh tree swap in later (SceneTrees), with every walk reading either.');
panel.csv(app.device);

const sah = (library: string) => rows.find((m) => m.library === library && m.metric === 'sah')?.value ?? null;
if (sah('ours') === null || sah('tmbvh-sah') === null) throw new Error('no SAH cost for ours or three-mesh-bvh');
let frames = 0;
app.loop(async () => {
    if (++frames === 10) await ready(app.device, { ...app.overlay.snapshot(), csv: panel.csvText, sahOurs: sah('ours'), sahTmbvh: sah('tmbvh-sah') });
});

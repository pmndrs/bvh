/**
 * compare/gpu: GPU traversal, ours (`TraceBatchKernel`, plain WebGPU) against
 * three-mesh-bvh's `/webgpu` (`BVHComputeData` in a TSL compute kernel), on
 * the renderer's device, closest and any hit, rays from inside and from
 * outside, in paired rounds (compare/gpu/gpu-suite.ts). The hits of either
 * library are drawn as points (live toggle).
 * What else: a tree built on the GPU (H-PLOC, `SceneTrees`) traced by the same
 * kernel, its answers checked against the tinybvh tree's. `?nodeFormat=f32` runs ours on
 * `f32` nodes (the default before 2026-11-13; quantized now).
 */

import { BufferAttribute, BufferGeometry, Group, Points, PointsMaterial } from 'three';
import { GpuRays, RAY_FLOATS, SceneTrees } from '@pmndrs/bvh';
import { SceneTreesObject3DSync } from '@pmndrs/bvh/three';
import { agreementText, checkFirst, emptyAgreement } from '../../../compare/core/agree.ts';
import { diagonal } from '../../../compare/core/geometry.ts';
import { makeRays, RAY_SETS } from '../../../compare/core/rays.ts';
import type { Measurement } from '../../../compare/core/suites.ts';
import { createOursGpu, createTmbvhGpu, packRaysFor, type GpuTracer } from '../../../compare/gpu/gpu-trace.ts';
import { gpuSuite } from '../../../compare/gpu/gpu-suite.ts';
import type { WorkPerRay } from '../../../compare/gpu/gpu-counts.ts';
import { createComparePage, expectAgreement, quick } from '../../shared/compare.ts';
import { ready } from '../../shared/smoke.ts';

const page = await createComparePage('compare/gpu', 'compare/gpu', 'GPU rays per second, ours against three-mesh-bvh/webgpu (SAH), on the same device: closest hit, and any hit (inside rays: within a quarter of the scene\'s diagonal; outside rays: within two diagonals, an occlusion test). Timed in paired rounds (the M5 drifts with heat): the ratio is taken within each round.', { trackTimestamp: true });
const { app, geometry, panel, sceneName } = page;
const log = (line: string) => panel.status(line);
const count = quick ? 16_384 : 1_048_576;
const scale = diagonal(geometry);
// Any hit: inside rays as short shadow-like rays (a quarter of the diagonal); outside rays as long occlusion tests.
const anyLength = (set: string) => (set === 'outside' ? scale * 2 : scale * 0.25);

// Ours twice: the library's default kernel, and with anyHitOrder 'larger' (the any-hit order for rays that
// start away from surfaces; closest hit is the same kernel). ?counts: the walks' work per ray as well.
const params = new URLSearchParams(location.search);
// ?nodeFormat=f32: ours on f32 nodes (the default before 2026-11-13; quantized now).
const nodeFormat = params.get('nodeFormat') === 'f32' ? ('f32' as const) : undefined;
const tracers: GpuTracer[] = [await createOursGpu(app.device, geometry, { nodeFormat }), await createOursGpu(app.device, geometry, { anyHitOrder: 'larger', nodeFormat }), await createTmbvhGpu(app.renderer, geometry)];
const workPerRay: ({ set: string; mode: string; library: string } & WorkPerRay)[] = [];
const all: Measurement[] = [];
const packedBySet = new Map<string, Float32Array>();
for (const set of RAY_SETS) {
    const rays = makeRays(geometry, set, count, 3);
    const closest = packRaysFor(rays, 1e30);
    packedBySet.set(set, closest);
    const results = await gpuSuite(tracers, sceneName, { closest, any: packRaysFor(rays, anyLength(set)), count, anyTMax: anyLength(set), scale, set }, quick ? { rounds: 2, dispatches: 3, warmups: 1 } : { rounds: 7 }, log);
    all.push(...results);
    if (params.has('counts')) {
        for (const mode of ['closest', 'any'] as const) {
            for (const tracer of tracers) {
                const work = await tracer.counts(mode);
                workPerRay.push({ set, mode, library: tracer.id, ...work });
                log(`${tracer.id} ${mode}/${set} per ray: ${work.boxTests.toFixed(1)} box tests, ${work.leaves.toFixed(1)} leaves, ${work.triangleTests.toFixed(1)} triangle tests, ${work.simdMaxBoxTests.toFixed(1)} box tests on the slowest of 32; ${(work.hitShare * 100).toFixed(1)}% hit`);
            }
        }
    }
}
if (workPerRay.length) {
    panel.paragraph(
        `Work per ray (counted copies of each walk, same rays): ${workPerRay.map((w) => `${w.library} ${w.mode}/${w.set}: ${w.boxTests.toFixed(1)} box tests, ${w.leaves.toFixed(1)} leaves, ${w.triangleTests.toFixed(1)} triangles`).join('; ')}.`,
    );
}
const names = {
    ours: '@pmndrs/bvh (TraceBatchKernel)',
    'ours-larger': "@pmndrs/bvh, anyHitOrder 'larger'",
    tmbvh: 'three-mesh-bvh/webgpu',
    'ours-over-tmbvh': 'ours / three-mesh-bvh, paired',
    'ours-larger-over-tmbvh': "ours ('larger') / three-mesh-bvh, paired",
};
panel.table(`GPU traversal, ${count.toLocaleString('en-US')} rays per dispatch`, all, ['closest/outside', 'any/outside', 'closest/inside', 'any/inside', 'gpu-bytes'], names);
panel.table('Paired ratio (median of rounds; above 1: ours faster)', all, ['closest/outside/ratio', 'any/outside/ratio', 'closest/inside/ratio', 'any/inside/ratio'], names);
panel.paragraph(`Timing: ${tracers.map((t) => `${t.name}: ${t.timing}`).join('; ')}. GPU bytes are what each walk reads: our 48-byte triangle records hold the corners; three-mesh-bvh reads its own vec4 copy of the positions and the index.`);

// Live: the closest hits of either library (the outside set) as points.
const points = new Points(new BufferGeometry(), new PointsMaterial({ color: 0x6fcf97, size: 2, sizeAttenuation: false }));
app.scene.add(points);
const shown = Math.min(count, 32_768);
const hitsOf = new Map<string, Float32Array>();
const packed = packedBySet.get('outside')!;
for (const tracer of tracers) {
    tracer.setRays('closest', packed.subarray(0, shown * RAY_FLOATS));
    await tracer.dispatch('closest');
    const t = await tracer.read('closest');
    const positions = new Float32Array(shown * 3);
    let n = 0;
    for (let i = 0; i < shown; i++) {
        if (!(t[i] >= 0)) continue;
        for (let k = 0; k < 3; k++) positions[n * 3 + k] = packed[i * RAY_FLOATS + k] + packed[i * RAY_FLOATS + 4 + k] * t[i];
        n++;
    }
    hitsOf.set(tracer.name, positions.subarray(0, n * 3));
}
const showHits = (name: string) => points.geometry.setAttribute('position', new BufferAttribute(hitsOf.get(name)!, 3));
showHits(tracers[0].name);
app.overlay.select('hits of', tracers.map((t) => t.name), tracers[0].name, showHits);

// What else: a GPU-built tree (H-PLOC first trees, held), traced by the same kernel.
const root = new Group();
page.mesh.removeFromParent();
root.add(page.mesh);
app.scene.add(root);
const trees = await SceneTrees.create(app.device, { label: 'compare gpu', holdRefinements: true });
const sync = new SceneTreesObject3DSync(trees, root);
sync.update();
let encoder = app.device.createCommandEncoder();
trees.update(encoder);
app.device.queue.submit([encoder.finish()]);
await trees.whenBuilt();
sync.update();
encoder = app.device.createCommandEncoder();
trees.update(encoder);
app.device.queue.submit([encoder.finish()]);
const gpuRays = await GpuRays.create(app.device, trees);
const sample = packed.subarray(0, Math.min(count, 65_536) * RAY_FLOATS);
const built = await gpuRays.traceClosest(sample);
tracers[0].setRays('closest', sample);
await tracers[0].dispatch('closest');
const reference = await tracers[0].read('closest');
const agreement = emptyAgreement();
for (let i = 0; i < built.count; i++) {
    const hit = built.hit(i);
    checkFirst(agreement, i, reference[i] >= 0 ? { triangle: 0, t: reference[i] } : null, hit ? { triangle: 0, t: hit.t } : null, scale);
}
panel.whatElse(
    `Trees built on the GPU are traced by the same kernel: the GPU H-PLOC tree of this mesh (SceneTrees' first tree${trees.counts.onGpuTrees ? '' : '; this device fell back to tinybvh'}), ${built.count.toLocaleString('en-US')} rays in ${(built.gpuMs ?? built.wallMs).toFixed(2)} ms, against the tinybvh tree: ${agreementText(agreement)}. three-mesh-bvh builds on the CPU only.`,
);
panel.csv(app.device);
for (const tracer of tracers) tracer.dispose();
gpuRays.destroy();

const checked = expectAgreement(all, ['tmbvh', 'ours-larger'], log);
// Our two kernels are the same walk in another child order: they must give the same answers on every ray.
for (const m of all) if (m.library === 'ours-larger' && m.agreement && m.agreement.disagree > 0) throw new Error(`anyHitOrder 'larger' disagrees with the default kernel on ${m.metric}: ${m.agreement.disagree} rays`);
if (agreement.disagree) throw new Error(`the GPU-built tree disagrees with the tinybvh tree on ${agreement.disagree} rays`);
let frames = 0;
app.loop(async () => {
    if (++frames === 10) await ready(app.device, { ...app.overlay.snapshot(), csv: panel.csvText, status: panel.statusText, workPerRay, checkedAgainstOurs: checked, gpuBuiltChecked: agreement.checked });
});

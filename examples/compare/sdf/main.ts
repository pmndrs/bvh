/**
 * compare/sdf: SDF generation time on the GPU, ours (`SdfGenerator`, plain
 * WebGPU) against three-mesh-bvh's `webgpu_sdfGeneration` kernel (a TSL compute
 * kernel over `BVHComputeData`), on the renderer's device, at several
 * resolutions, in paired rounds (compare/gpu/gpu-sdf.ts).
 *
 * The kernels do different work per voxel: both find the closest point; for
 * the sign theirs casts one first-hit ray along +Z and takes the side it hits,
 * ours (the default, parity) casts two or three all-hit rays and takes a vote
 * (robust to a ray through an edge or a hole). `ours-first-hit` is our opt-in
 * `sign: 'first-hit'`, their way on our walk; `ours-unsigned` is the closest
 * point alone, the floor of either. Distances must agree; signs are compared
 * and reported.
 * What else: the field of the scene's GPU-built tree (SceneTrees), filled into
 * a three texture by `generateSdfTexture` and shown as a slice.
 *
 * Options: `?dims=32,64` (the resolutions), `?repeat=N` (N dispatches back to
 * back per timing, their mean: the GPU at its loaded clock; default 1, one
 * waited-for dispatch), `?breakdown` (each part of either kernel on its own, and
 * other ways to sign, `OURS_BREAKDOWN` in compare/gpu/gpu-sdf.ts; the kernels
 * that must give a library field bit for bit are checked).
 */

import { BufferAttribute, BufferGeometry, DoubleSide, Group, Mesh, MeshStandardMaterial, PlaneGeometry, Vector3 } from 'three';
import { abs, fract, Fn, mix, positionWorld, smoothstep, vec3, vec4 } from 'three/tsl';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import { readSdf, SceneTrees } from '@pmndrs/bvh';
import { generateSdfTexture, SceneTreesObject3DSync } from '@pmndrs/bvh/three';
import { emptyAgreement, type Agreement } from '../../../compare/core/agree.ts';
import { median, yieldNow } from '../../../compare/core/measure.ts';
import type { Measurement } from '../../../compare/core/suites.ts';
import { createOursBreakdownSdf, createOursSdf, createTmbvhSdf, OURS_BREAKDOWN, sdfBox, weldedAndCentred, type GpuSdf, type OursBreakdown } from '../../../compare/gpu/gpu-sdf.ts';
import { frameObject } from '../../shared/loader.ts';
import { createComparePage, expectAgreement, quick } from '../../shared/compare.ts';
import { ready } from '../../shared/smoke.ts';

const page = await createComparePage(
    'compare/sdf',
    'compare/sdf',
    "SDF generation on the GPU, ours (SdfGenerator) against three-mesh-bvh/webgpu's webgpu_sdfGeneration kernel (SAH), on the same device, the same welded geometry and the same dim³ voxel centres over the mesh's box grown by 5% of its diagonal (centred on the origin, as their kernel needs). Timed in paired rounds: the ratio is taken within each round.",
    { trackTimestamp: true },
);
const { app, geometry, panel, sceneName } = page;
const log = (line: string) => panel.status(line);
const mesh = weldedAndCentred(geometry);
const box = sdfBox(mesh.positions, 0.05);
const scale = Math.hypot(box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2]);
const query = new URLSearchParams(location.search);
const resolutions = query.get('dims')?.split(',').map(Number) ?? (quick ? [32] : [32, 64, 128]);
const rounds = quick ? 2 : 5;
const perRound = quick ? 2 : 5;
const repeat = Math.max(1, Number(query.get('repeat')) || 1);
const breakdown = query.has('breakdown');

const generators: GpuSdf[] = [
    await createOursSdf(app.device, mesh, box, 'parity'),
    await createOursSdf(app.device, mesh, box, 'first-hit'),
    await createOursSdf(app.device, mesh, box, 'unsigned'),
    await createTmbvhSdf(app.renderer, mesh, box),
];
if (breakdown) {
    // ?only=closest,closest-general: just those breakdown kernels (by OURS_BREAKDOWN key), for a focused paired run.
    const only = query.get('only')?.split(',');
    const wanted = (variant: OursBreakdown) => !only || only.includes(variant);
    if (!only) generators.push(await createOursSdf(app.device, mesh, box, 'winding'));
    for (const variant of Object.keys(OURS_BREAKDOWN) as OursBreakdown[]) if (wanted(variant)) generators.push(await createOursBreakdownSdf(app.device, mesh, box, variant));
    if (!only) {
        generators.push(await createOursBreakdownSdf(app.device, mesh, box, 'votes-all', 'winding'), await createOursBreakdownSdf(app.device, mesh, box, 'votes-early', 'winding'));
        generators.push(await createTmbvhSdf(app.renderer, mesh, box, 'closest'), await createTmbvhSdf(app.renderer, mesh, box, 'ray'));
    }
}
/** Breakdown kernels that must give another field bit for bit: [kernel, the field it must equal]. */
const identical: readonly (readonly [string, string])[] = breakdown
    ? ([
          ['ours-votes-all', 'ours'],
          ['ours-votes-early', 'ours'],
          ['ours-no-winding', 'ours'],
          ['ours-votes-early-merged', 'ours'],
          ['ours-closest', 'ours-unsigned'],
          ['ours-closest-merged', 'ours-unsigned'],
          ['ours-first-hit-bounded', 'ours-first-hit'],
          ['ours-votes-all-winding', 'ours-winding'],
          ['ours-votes-early-winding', 'ours-winding'],
          // The identity fast path is the same field bit for bit (query-wgsl.ts, QUERY_IDENTITY_CHECK_WGSL).
          ['ours-closest-general', 'ours-unsigned'],
          ['ours-votes-early-general', 'ours'],
          ['ours-closest-exact', 'ours-unsigned'],
          ['ours-closest-boxes-only', 'ours-unsigned'],
      ] as const).filter(([kernel]) => generators.some((generator) => generator.id === kernel))
    : [];
/** Paired speed-ups logged in breakdown runs: [before, after], before's time over after's per round. */
const SPEEDUPS: readonly (readonly [string, string])[] = [
    ['ours-closest-general', 'ours-closest'],
    ['ours-votes-early-general', 'ours-votes-early'],
    ['ours-closest-general', 'ours-closest-merged'],
    ['ours-closest', 'ours-closest-merged'],
    ['ours-closest-general', 'ours-closest-exact'],
    ['ours-closest-general', 'ours-closest-boxes-only'],
];
/** Kernels whose field is not an SDF to compare with ours (the parts alone). */
const partsOnly = new Set(['ours-rays3', 'ours-rays-early', 'ours-ray1', 'ours-first-hit-only', 'tmbvh-ray']);
const bitFailures: string[] = [];
log(`${sceneName}: ${(mesh.positions.length / 3).toLocaleString('en-US')} welded vertices; box ${box.min.map((v) => v.toFixed(2)).join(', ')} to ${box.max.map((v) => v.toFixed(2)).join(', ')}`);

/** Distances (|value|) within float tolerance; signs where ours is farther than 1e-4 of the box from the surface. */
function compareFields(ours: Float32Array, theirs: Float32Array): { distance: Agreement; sign: Agreement } {
    const distance = emptyAgreement();
    const sign = emptyAgreement();
    for (let i = 0; i < ours.length; i++) {
        const a = Math.abs(ours[i]);
        const b = Math.abs(theirs[i]);
        distance.checked++;
        if (Math.abs(a - b) <= 1e-4 * Math.max(a, b) + 1e-6 * scale) distance.same++;
        else {
            distance.disagree++;
            if (distance.examples.length < 5) distance.examples.push(`voxel ${i}: ours ${ours[i].toPrecision(6)}, theirs ${theirs[i].toPrecision(6)}`);
        }
        if (a <= 1e-4 * scale) continue;
        sign.checked++;
        if (Math.sign(ours[i]) === Math.sign(theirs[i])) sign.same++;
        else {
            sign.disagree++;
            if (sign.examples.length < 5) sign.examples.push(`voxel ${i}: ours ${ours[i].toPrecision(4)}, theirs ${theirs[i].toPrecision(4)}`);
        }
    }
    return { distance, sign };
}

const all: Measurement[] = [];
for (const dim of resolutions) {
    // Warm up (compiles, allocates the texture), then rounds with the order reversed every other round.
    for (const generator of generators) await generator.generate(dim, repeat);
    const times = new Map<string, number[]>();
    for (let round = 0; round < rounds; round++) {
        const who = round % 2 === 1 ? [...generators].reverse() : generators;
        for (const generator of who) {
            const samples: number[] = [];
            for (let i = 0; i < perRound; i++) samples.push(await generator.generate(dim, repeat));
            times.set(generator.id, [...(times.get(generator.id) ?? []), median(samples)]);
        }
        await yieldNow();
    }
    const fields = new Map<string, Float32Array>();
    for (const generator of generators) fields.set(generator.id, await generator.read());
    const ours = fields.get('ours')!;
    for (const [kernel, library] of identical) {
        const a = new Uint32Array(fields.get(kernel)!.buffer);
        const b = new Uint32Array(fields.get(library)!.buffer);
        let differ = 0;
        for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) differ++;
        log(`${kernel} against ${library} ${dim}³: ${differ ? `${differ} voxels differ` : 'bit for bit'}`);
        if (differ) bitFailures.push(`${kernel} against ${library} at ${dim}³: ${differ} voxels differ`);
    }
    const voxels = dim ** 3;
    for (const generator of generators) {
        const { distance, sign } = generator.id === 'ours' || partsOnly.has(generator.id) ? { distance: undefined, sign: undefined } : compareFields(ours, fields.get(generator.id)!);
        const ms = median(times.get(generator.id)!);
        const method = `${dim}³ voxels; per round the median of ${perRound} generations after a warm-up (${generator.timing}); median of ${rounds} rounds, the order reversed every other round`;
        all.push({ page: 'sdf', scene: sceneName, library: generator.id, metric: `build/${dim}`, value: ms, unit: 'ms', method, agreement: distance, ranked: !distance || distance.disagree === 0 });
        all.push({ page: 'sdf', scene: sceneName, library: generator.id, metric: `voxels/s/${dim}`, value: (voxels / ms) * 1000, unit: 'voxels/s', method, ranked: !distance || distance.disagree === 0 });
        // The unsigned field has no sign to compare.
        const signed = sign && generator.id !== 'ours-unsigned';
        if (signed) all.push({ page: 'sdf', scene: sceneName, library: generator.id, metric: `sign/${dim}`, value: sign.checked ? sign.same / sign.checked : null, unit: 'share', method: 'signs equal to ours where ours is farther than 1e-4 of the box from the surface', agreement: sign, ranked: false });
        log(`${generator.id} ${dim}³: ${ms.toFixed(2)} ms${distance ? `; distances ${distance.disagree} of ${distance.checked} disagree` : ''}${signed ? `, signs ${sign.disagree} of ${sign.checked} differ` : ''}`);
    }
    const theirs = times.get('tmbvh')!;
    for (const id of generators.map((g) => g.id).filter((id) => id.startsWith('ours'))) {
        const a = times.get(id)!;
        const ratios = a.map((time, i) => theirs[i] / time);
        all.push({ page: 'sdf', scene: sceneName, library: `${id}-over-tmbvh`, metric: `build/${dim}/ratio`, value: median(ratios), unit: 'ratio', method: `three-mesh-bvh's time over ${id} within each round (above 1: ours faster), median of ${rounds} rounds (min ${Math.min(...ratios).toFixed(2)}, max ${Math.max(...ratios).toFixed(2)})`, ranked: true });
        if (breakdown) log(`ratio tmbvh/${id} ${dim}³: ${median(ratios).toFixed(3)} (min ${Math.min(...ratios).toFixed(2)}, max ${Math.max(...ratios).toFixed(2)})`);
    }
    // The identity fast path's before over after, within each round (above 1: the fast path is faster).
    for (const [before, after] of SPEEDUPS) {
        const a = times.get(before);
        const b = times.get(after);
        if (!a || !b) continue;
        const ratios = a.map((time, i) => time / b[i]);
        all.push({ page: 'sdf', scene: sceneName, library: `${before}-over-${after}`, metric: `build/${dim}/ratio`, value: median(ratios), unit: 'ratio', method: `${before}'s time over ${after}'s within each round, median of ${rounds} rounds (min ${Math.min(...ratios).toFixed(3)}, max ${Math.max(...ratios).toFixed(3)})`, ranked: false });
        log(`ratio ${before}/${after} ${dim}³: ${median(ratios).toFixed(3)} (min ${Math.min(...ratios).toFixed(3)}, max ${Math.max(...ratios).toFixed(3)})`);
    }
}

const names: Record<string, string> = {
    ours: '@pmndrs/bvh (parity vote)',
    'ours-first-hit': '@pmndrs/bvh (first hit, 1 ray)',
    'ours-unsigned': '@pmndrs/bvh (unsigned)',
    tmbvh: 'three-mesh-bvh/webgpu (1 ray, side)',
    'ours-over-tmbvh': 'theirs / ours, paired',
    'ours-unsigned-over-tmbvh': 'theirs / ours unsigned, paired',
    'ours-first-hit-over-tmbvh': 'theirs / ours first hit, paired',
};
for (const generator of generators) names[generator.id] ??= generator.name;
for (const generator of generators) names[`${generator.id}-over-tmbvh`] ??= `theirs / ${generator.id}, paired`;
panel.table('SDF build time', all, resolutions.map((dim) => `build/${dim}`), names);
panel.table('Voxels per second', all, resolutions.map((dim) => `voxels/s/${dim}`), names);
panel.table('Paired ratio (median of rounds; above 1: ours faster)', all, resolutions.map((dim) => `build/${dim}/ratio`), names);
panel.table('Signs equal to ours (away from the surface)', all, resolutions.map((dim) => `sign/${dim}`), names);
panel.paragraph(
    `Timing: ${generators.map((g) => `${g.name}: ${g.timing}`).join('; ')}. Per voxel, three-mesh-bvh walks the tree twice (closest point, one first-hit ray), ours three or four times by default (closest point, then two all-hit rays that cannot stop early, and a third when those two disagree); sign 'first-hit' walks it twice, as theirs; the unsigned row is the closest point alone. A sign that differs is a voxel where one ray and a vote of three rays disagree: an edge, a seam or an open part.`,
);

// What else: the scene's GPU-built tree (SceneTrees' H-PLOC first tree) fills a three texture directly, shown as a slice.
const threeGeometry = new BufferGeometry();
threeGeometry.setAttribute('position', new BufferAttribute(mesh.positions, 3));
threeGeometry.setIndex(new BufferAttribute(mesh.indices, 1));
threeGeometry.computeVertexNormals();
const centred = new Mesh(threeGeometry, new MeshStandardMaterial({ color: 0x8fb3ff, roughness: 0.6, side: DoubleSide, transparent: true, opacity: 0.35 }));
const root = new Group();
root.add(centred);
page.mesh.removeFromParent();
app.scene.add(root);
frameObject(root, app.camera, app.controls);
const trees = await SceneTrees.create(app.device, { label: 'compare sdf', holdRefinements: true });
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
const shownDim = resolutions[resolutions.length - 1];
const sdf = await generateSdfTexture(app.renderer, trees, { box: { min: new Vector3(...box.min), max: new Vector3(...box.max) }, resolution: shownDim });
const fromTrees = await readSdf(app.device, { texture: sdf.gpuTexture }, sdf.grid);
await generators[0].generate(shownDim);
const fromTiny = await generators[0].read();
const treeAgreement = compareFields(fromTiny, fromTrees);
const size = sdf.box.getSize(new Vector3());
const slice = new Mesh(
    new PlaneGeometry(size.x, size.z),
    Object.assign(new MeshBasicNodeMaterial({ side: DoubleSide }), {
        colorNode: Fn(() => {
            const d = sdf.sample(positionWorld);
            const base = mix(vec3(0.95, 0.55, 0.25), vec3(0.3, 0.5, 0.95), d.lessThan(0).select(1, 0));
            const contour = smoothstep(0.85, 1, abs(fract(d.mul(10 / scale)).sub(0.5)).mul(2)).mul(0.35);
            return vec4(base.mul(contour.oneMinus()), 1);
        })(),
    }),
);
slice.rotation.x = -Math.PI / 2;
app.scene.add(slice);
app.overlay.range('slice height', 0, 1, 0.01, 0.5, (value) => (slice.position.y = box.min[1] + (box.max[1] - box.min[1]) * value));
panel.whatElse(
    `The field of the scene's own trees, any format, straight into a three texture: SceneTrees' ${trees.counts.onGpuTrees ? 'GPU H-PLOC first tree' : 'tree (this device fell back to tinybvh)'} through generateSdfTexture, ${shownDim}³ in ${sdf.ms.toFixed(1)} ms (wall), sampled by the slice below in TSL. Against the tinybvh tree's field: distances ${treeAgreement.distance.disagree} of ${treeAgreement.distance.checked} disagree, signs ${treeAgreement.sign.disagree} of ${treeAgreement.sign.checked} differ. Also: the winding rule for overlapping parts, a distance band (maxDistance) that culls the walk, instance masks, and slabs per submit for big grids.`,
);
panel.csv(app.device);

// The smoke check: distances agree everywhere (three-mesh-bvh's and our unsigned field), and the SceneTrees field agrees with ours.
const distances = all.filter((m) => m.metric.startsWith('build/') && !m.metric.endsWith('/ratio'));
const checked = expectAgreement(distances, ['tmbvh', 'ours-unsigned'], log);
if (bitFailures.length) throw new Error(`breakdown kernels differ from the library's fields: ${bitFailures.join('; ')}`);
if (treeAgreement.distance.disagree) throw new Error(`the SceneTrees field disagrees with the tinybvh tree's on ${treeAgreement.distance.disagree} voxels`);
for (const generator of generators) generator.dispose();
let frames = 0;
app.loop(async () => {
    if (++frames === 10) await ready(app.device, { ...app.overlay.snapshot(), csv: panel.csvText, status: panel.statusText, checkedAgainstOurs: checked });
});

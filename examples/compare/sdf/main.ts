/**
 * compare/sdf: SDF generation time on the GPU, ours (`SdfGenerator`, plain
 * WebGPU) against three-mesh-bvh's `webgpu_sdfGeneration` kernel (a TSL compute
 * kernel over `BVHComputeData`), on the renderer's device, at several
 * resolutions, in paired rounds (compare/gpu/gpu-sdf.ts).
 *
 * The kernels do different work per voxel: both find the closest point; for
 * the sign theirs casts one first-hit ray along +Z and takes the side it hits,
 * ours casts three all-hit rays and takes a parity vote (robust to a ray
 * through an edge or a hole). `ours-unsigned` is the closest point alone, the
 * floor of either. Distances must agree; signs are compared and reported.
 * What else: the field of the scene's GPU-built tree (SceneTrees), filled into
 * a three texture by `generateSdfTexture` and shown as a slice.
 */

import { BufferAttribute, BufferGeometry, DoubleSide, Group, Mesh, MeshStandardMaterial, PlaneGeometry, Vector3 } from 'three';
import { abs, fract, Fn, mix, positionWorld, smoothstep, vec3, vec4 } from 'three/tsl';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import { readSdf, SceneTrees } from '@pmndrs/bvh';
import { generateSdfTexture, SceneTreesObject3DSync } from '@pmndrs/bvh/three';
import { emptyAgreement, type Agreement } from '../../../compare/core/agree.ts';
import { median, yieldNow } from '../../../compare/core/measure.ts';
import type { Measurement } from '../../../compare/core/suites.ts';
import { createOursSdf, createTmbvhSdf, sdfBox, weldedAndCentred, type GpuSdf } from '../../../compare/gpu/gpu-sdf.ts';
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
const resolutions = quick ? [32] : [32, 64, 128];
const rounds = quick ? 2 : 5;
const perRound = quick ? 2 : 5;

const generators: GpuSdf[] = [await createOursSdf(app.device, mesh, box, 'parity'), await createOursSdf(app.device, mesh, box, 'unsigned'), await createTmbvhSdf(app.renderer, mesh, box)];
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
    for (const generator of generators) await generator.generate(dim);
    const times = new Map<string, number[]>();
    for (let round = 0; round < rounds; round++) {
        const who = round % 2 === 1 ? [...generators].reverse() : generators;
        for (const generator of who) {
            const samples: number[] = [];
            for (let i = 0; i < perRound; i++) samples.push(await generator.generate(dim));
            times.set(generator.id, [...(times.get(generator.id) ?? []), median(samples)]);
        }
        await yieldNow();
    }
    const fields = new Map<string, Float32Array>();
    for (const generator of generators) fields.set(generator.id, await generator.read());
    const ours = fields.get('ours')!;
    const voxels = dim ** 3;
    for (const generator of generators) {
        const { distance, sign } = generator.id === 'ours' ? { distance: undefined, sign: undefined } : compareFields(ours, fields.get(generator.id)!);
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
    for (const id of ['ours', 'ours-unsigned'] as const) {
        const a = times.get(id)!;
        const ratios = a.map((time, i) => theirs[i] / time);
        all.push({ page: 'sdf', scene: sceneName, library: `${id}-over-tmbvh`, metric: `build/${dim}/ratio`, value: median(ratios), unit: 'ratio', method: `three-mesh-bvh's time over ${id === 'ours' ? 'ours' : 'ours unsigned'} within each round (above 1: ours faster), median of ${rounds} rounds (min ${Math.min(...ratios).toFixed(2)}, max ${Math.max(...ratios).toFixed(2)})`, ranked: true });
    }
}

const names = {
    ours: '@pmndrs/bvh (parity, 3 rays)',
    'ours-unsigned': '@pmndrs/bvh (unsigned)',
    tmbvh: 'three-mesh-bvh/webgpu (1 ray, side)',
    'ours-over-tmbvh': 'theirs / ours, paired',
    'ours-unsigned-over-tmbvh': 'theirs / ours unsigned, paired',
};
panel.table('SDF build time', all, resolutions.map((dim) => `build/${dim}`), names);
panel.table('Voxels per second', all, resolutions.map((dim) => `voxels/s/${dim}`), names);
panel.table('Paired ratio (median of rounds; above 1: ours faster)', all, resolutions.map((dim) => `build/${dim}/ratio`), names);
panel.table('Signs equal to ours (away from the surface)', all, resolutions.map((dim) => `sign/${dim}`), names);
panel.paragraph(
    `Timing: ${generators.map((g) => `${g.name}: ${g.timing}`).join('; ')}. Per voxel, three-mesh-bvh walks the tree twice (closest point, one first-hit ray), ours four times (closest point, three all-hit rays that cannot stop early); the unsigned row is the closest point alone. A sign that differs is a voxel where one ray along +Z and a vote of three rays disagree: an edge, a seam or an open part.`,
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
if (treeAgreement.distance.disagree) throw new Error(`the SceneTrees field disagrees with the tinybvh tree's on ${treeAgreement.distance.disagree} voxels`);
for (const generator of generators) generator.dispose();
let frames = 0;
app.loop(async () => {
    if (++frames === 10) await ready(app.device, { ...app.overlay.snapshot(), csv: panel.csvText, status: panel.statusText, checkedAgainstOurs: checked });
});

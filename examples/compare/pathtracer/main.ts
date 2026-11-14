/**
 * compare/pathtracer: the same small path tracer (three/pathtracer's kernel,
 * examples/three/pathtracer/kernel.ts: PCG sampling, cosine-weighted diffuse
 * bounces, one sun shadow ray per hit, accumulation) over two walks, on the
 * renderer's device:
 *
 * - **ours**: `SceneTrees` (the sync, `static: true`; tinybvh trees, merged
 *   static instances, quantized nodes: the defaults) through `BvhTraceNodes`:
 *   `traceClosest` / `traceAny`, `hit.material` into a palette;
 * - **three-mesh-bvh**: `BVHComputeData` over the same meshes (a `MeshBVH` with
 *   SAH per geometry, its clustered top level) through its WGSL functions
 *   `bvh_RaycastFirstHit` / `bvh_RaycastAnyHit` in a `wgslFn`, the hit's
 *   `objectIndex` into a palette of the same colours.
 *
 * Measured in paired rounds (ours and theirs back to back, the order reversed
 * every other round): the full tracer (closest and any hit), the tracer with
 * the shadow rays compiled out (closest hit alone: the same closest-hit rays,
 * since the shadow ray draws no random number), and from the two the any-hit
 * (shadow) rays alone. Rays are counted in the kernel (one atomic per pixel);
 * time is three's compute timestamps, else wall clock of one dispatch with the
 * queue drained before and after. Each pixel's first hit is checked against
 * ours.
 *
 * What else: the scene's first trees built on the GPU (H-PLOC, `SceneTrees`
 * with `holdRefinements`), traced by the same kernel: no CPU build stands
 * between a loaded scene and its first traced frame.
 */

import { BoxGeometry, Color, Group, Mesh, MeshStandardMaterial, PlaneGeometry, SphereGeometry, TorusKnotGeometry, Vector3, type Material, type Object3D } from 'three';
import { storage, texture, uv, wgslFn } from 'three/tsl';
import { MeshBasicNodeMaterial, QuadMesh, StorageBufferAttribute, StorageTexture, StructTypeNode, type ComputeNode } from 'three/webgpu';
import { MeshBVH, SAH } from 'three-mesh-bvh';
import { BVHComputeData } from 'three-mesh-bvh/webgpu';
import { SceneTrees } from '@pmndrs/bvh';
import { BvhTraceNodes, SceneTreesObject3DSync, type BvhTslNode } from '@pmndrs/bvh/three';
import { agreementText, checkFirst, emptyAgreement, type Agreement } from '../../../compare/core/agree.ts';
import { median, yieldNow } from '../../../compare/core/measure.ts';
import type { Measurement } from '../../../compare/core/suites.ts';
import { createApp } from '../../shared/app.ts';
import { quick, ResultsPanel } from '../../shared/compare.ts';
import { frameObject, loadGltf, meshesOf, requestedModel } from '../../shared/loader.ts';
import { SceneTreesLog } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';
import { pathTracerKernel, pathTracerTargets, pathTracerUniforms, type PathTracerTargets, type PathTracerWalk } from '../../three/pathtracer/kernel.ts';

const app = await createApp('compare/pathtracer', 'compare/pathtracer', { trackTimestamp: true });
const { device, renderer, camera, controls, overlay } = app;
const panel = new ResultsPanel(
    'compare/pathtracer',
    "The same small path tracer (three/pathtracer's kernel: jittered camera rays, diffuse bounces, a sun shadow ray per hit) on our walk (BvhTraceNodes over SceneTrees) and on three-mesh-bvh/webgpu's (BVHComputeData's bvh_RaycastFirstHit / bvh_RaycastAnyHit, SAH), same scene, same rays, paired rounds. Each pixel's first hit is checked against ours.",
);
const log = (line: string) => panel.status(line);

// ---- The scene: three/pathtracer's default, or ?model=.
function defaultScene(): Object3D {
    const root = new Group();
    const add = (mesh: Mesh, x: number, y: number, z: number) => {
        mesh.position.set(x, y, z);
        root.add(mesh);
        return mesh;
    };
    const lambert = (color: number) => new MeshStandardMaterial({ color });
    add(new Mesh(new PlaneGeometry(14, 14), lambert(0xd8d4cc)), 0, 0, 0).rotation.x = -Math.PI / 2;
    add(new Mesh(new PlaneGeometry(14, 6), lambert(0xc8695a)), 0, 3, -7);
    add(new Mesh(new TorusKnotGeometry(1, 0.32, quick ? 200 : 600, quick ? 24 : 64), lambert(0x8fb3ff)), 0, 1.9, 0);
    add(new Mesh(new SphereGeometry(0.9, 64, 32), lambert(0xf0e0a0)), -3, 0.9, 1.5);
    add(new Mesh(new BoxGeometry(1.4, 2.4, 1.4), lambert(0xb0e0a8)), 3, 1.2, 1).rotation.y = 0.5;
    return root;
}
const model = requestedModel();
const world = model ? (await loadGltf(model)).root : defaultScene();
app.scene.add(world);
world.updateMatrixWorld(true);
const sceneBox = frameObject(world, camera, controls);
if (!model) {
    camera.position.set(0, 3.5, 9);
    controls.target.set(0, 1.4, 0);
    controls.update();
}
const meshes = meshesOf(world);
const triangles = meshes.reduce((sum, mesh) => sum + (mesh.geometry.index ? mesh.geometry.index.count : mesh.geometry.getAttribute('position').count) / 3, 0);
overlay.set('scene', `${model ?? 'default'}: ${meshes.length} meshes, ${Math.round(triangles).toLocaleString('en-US')} triangles`);

// One colour per mesh (its first material's), the same palette order for both libraries.
const PALETTE = 256;
const colourOf = (mesh: Mesh): Color => ((Array.isArray(mesh.material) ? mesh.material[0] : mesh.material) as Material & { color?: Color }).color ?? new Color(0.8, 0.8, 0.8);
const paletteSlot = new Map<Mesh, number>();
for (const mesh of meshes) paletteSlot.set(mesh, Math.min(paletteSlot.size, PALETTE - 1));
function paletteNode(slotOf: (index: number) => Mesh | undefined): BvhTslNode {
    const data = new Float32Array(PALETTE * 4).fill(0.8);
    for (let i = 0; i < PALETTE; i++) {
        const mesh = slotOf(i);
        if (mesh) data.set([...colourOf(mesh).toArray(), 1], i * 4);
    }
    return storage(new StorageBufferAttribute(data, 4), 'vec4', PALETTE).toReadOnly();
}

// ---- Ours: SceneTrees and BvhTraceNodes.
const oursLog = new SceneTreesLog();
const buildStart = performance.now();
const trees = await SceneTrees.create(device, { label: 'compare pathtracer', onTree: oursLog.onTree });
const sync = new SceneTreesObject3DSync(trees, world, { static: true, material: (mesh) => paletteSlot.get(mesh) ?? 0 });
function updateOurs(target: SceneTrees, targetSync: SceneTreesObject3DSync): void {
    targetSync.update();
    const encoder = device.createCommandEncoder({ label: 'compare pathtracer update' });
    target.update(encoder);
    device.queue.submit([encoder.finish()]);
}
updateOurs(trees, sync);
await trees.whenRefined();
updateOurs(trees, sync);
const oursBuildMs = performance.now() - buildStart;
const bvh = new BvhTraceNodes(renderer, trees);
const meshByPalette = new Map([...paletteSlot].map(([mesh, slot]) => [slot, mesh]));
const oursPalette = paletteNode((i) => meshByPalette.get(i));
const oursWalk = (nodes: BvhTraceNodes): PathTracerWalk => ({
    closest(origin, direction, tMin, tMax) {
        const hit = nodes.traceClosest(origin, direction, tMin, tMax).toVar() as BvhTslNode;
        return { t: hit.get('t'), normal: hit.get('normal'), albedo: oursPalette.element(hit.get('material')).xyz };
    },
    any: (origin, direction, tMin, tMax) => nodes.traceAny(origin, direction, tMin, tMax) as BvhTslNode,
});
overlay.build(oursLog.build(trees));
overlay.tree(oursLog.tree(trees, sync.meshHandles));

// ---- three-mesh-bvh: MeshBVH (SAH) per geometry, BVHComputeData over the meshes.
const theirStart = performance.now();
const theirGeometries = new Set(meshes.map((mesh) => mesh.geometry));
for (const geometry of theirGeometries) (geometry as unknown as { boundsTree: MeshBVH }).boundsTree = new MeshBVH(geometry, { strategy: SAH });
const theirData = new BVHComputeData(meshes, { attributes: { position: 'vec4f' } });
theirData.update();
const theirBuildMs = performance.now() - theirStart;
// Its hit names a transform slot (`objectIndex`); BVHComputeData keeps the slot of each object in a private map.
const transformMap = (theirData as unknown as { _getTransformMap(bvh: unknown): Map<string, { object: Mesh; slot: number }>; bvh: unknown })._getTransformMap((theirData as unknown as { bvh: unknown }).bvh);
const meshBySlot = new Map([...transformMap.values()].map((info) => [info.slot, info.object]));
const theirPalette = paletteNode((i) => meshBySlot.get(i));
const theirHitStruct = new StructTypeNode({ t: 'float', normal: 'vec3', slot: 'uint' }, 'TmbvhPathHit');
const fns = theirData.fns as unknown as Record<'raycastFirstHit' | 'raycastAnyHit', never>;
const theirClosest = wgslFn(
    /* wgsl */ `
    fn tmbvhPathClosest( origin: vec3f, direction: vec3f, maxDist: f32 ) -> TmbvhPathHit {
        var ray: Ray;
        ray.origin = origin;
        ray.direction = direction;
        ray.maxDist = maxDist;
        var hit: IntersectionResult;
        bvh_RaycastFirstHit( ray, &hit );
        return TmbvhPathHit( select( -1.0, hit.dist, hit.didHit ), hit.normal, hit.objectIndex );
    }
`,
    [theirHitStruct as never, fns.raycastFirstHit],
) as unknown as (...args: unknown[]) => BvhTslNode;
const theirAny = wgslFn(
    /* wgsl */ `
    fn tmbvhPathAny( origin: vec3f, direction: vec3f, maxDist: f32 ) -> bool {
        var ray: Ray;
        ray.origin = origin;
        ray.direction = direction;
        ray.maxDist = maxDist;
        return bvh_RaycastAnyHit( ray );
    }
`,
    [fns.raycastAnyHit],
) as unknown as (...args: unknown[]) => BvhTslNode;
// Their walk has no tMin (the kernel offsets each bounce's origin along the normal, as for ours); maxDist bounds the ray.
const theirWalk: PathTracerWalk = {
    closest(origin, direction, _tMin, tMax) {
        const hit = theirClosest(origin, direction, tMax).toVar() as BvhTslNode;
        return { t: hit.get('t'), normal: hit.get('normal'), albedo: theirPalette.element(hit.get('slot')).xyz };
    },
    any: (origin, direction, _tMin, tMax) => theirAny(origin, direction, tMax),
};
log(`builds: ours (SceneTrees, GPU first trees then tinybvh) ${oursBuildMs.toFixed(0)} ms to the tinybvh trees; three-mesh-bvh (MeshBVH SAH per geometry + BVHComputeData.update) ${theirBuildMs.toFixed(0)} ms`);

// ---- What else: the scene's GPU-built trees (H-PLOC first trees, held), traced by the same kernel.
const gpuLog = new SceneTreesLog();
const gpuStart = performance.now();
const gpuTrees = await SceneTrees.create(device, { label: 'compare pathtracer gpu-built', onTree: gpuLog.onTree, holdRefinements: true });
const gpuSync = new SceneTreesObject3DSync(gpuTrees, world, { static: true, material: (mesh) => paletteSlot.get(mesh) ?? 0 });
updateOurs(gpuTrees, gpuSync);
await gpuTrees.whenBuilt();
updateOurs(gpuTrees, gpuSync);
const gpuBuiltMs = performance.now() - gpuStart;
const gpuBvh = new BvhTraceNodes(renderer, gpuTrees);

// ---- Kernels: per walk, the full tracer and the closest-only one, at one fixed size.
const W = quick ? 256 : Math.min(1280, Math.floor(innerWidth));
const H = Math.max(16, Math.round((W * innerHeight) / innerWidth));
const u = pathTracerUniforms();
u.width.value = W;
u.height.value = H;
interface Tracer {
    readonly id: 'ours' | 'tmbvh' | 'ours-gpu-built';
    readonly name: string;
    readonly targets: PathTracerTargets;
    readonly full: ComputeNode;
    readonly closest: ComputeNode;
}
function tracer(id: Tracer['id'], name: string, walk: PathTracerWalk): Tracer {
    const targets = pathTracerTargets(W, H, new StorageTexture(W, H), true);
    return { id, name, targets, full: pathTracerKernel(walk, u, targets, W, H), closest: pathTracerKernel(walk, u, targets, W, H, { shadows: false }) };
}
const tracers: Tracer[] = [
    tracer('ours', '@pmndrs/bvh (BvhTraceNodes)', oursWalk(bvh)),
    tracer('tmbvh', 'three-mesh-bvh/webgpu (BVHComputeData, SAH)', theirWalk),
    tracer('ours-gpu-built', '@pmndrs/bvh, GPU-built trees (H-PLOC)', oursWalk(gpuBvh)),
];
const compileStart = performance.now();
for (const t of tracers) {
    await renderer.compileComputeAsync(t.full);
    await renderer.compileComputeAsync(t.closest);
}
log(`compiled six kernels in ${(performance.now() - compileStart).toFixed(0)} ms`);

const timestamps = renderer.hasFeature('timestamp-query');
const timing = timestamps ? "three's compute timestamps" : 'wall clock of one dispatch, the queue drained before and after';
camera.updateMatrixWorld();
function setCamera(): void {
    u.cameraWorld.value.copy(camera.matrixWorld);
    u.projectionInverse.value.copy(camera.projectionMatrixInverse);
    u.cameraPosition.value.setFromMatrixPosition(camera.matrixWorld);
}
setCamera();
async function raysCounted(t: Tracer): Promise<number> {
    return new Uint32Array(await renderer.getArrayBufferAsync(t.targets.rayCounter))[0];
}
/** One timed dispatch: milliseconds and rays traced. */
async function dispatch(t: Tracer, kernel: ComputeNode): Promise<{ ms: number; rays: number }> {
    const before = await raysCounted(t);
    let ms: number;
    if (timestamps) {
        await renderer.resolveTimestampsAsync('compute');
        renderer.compute(kernel);
        ms = (await renderer.resolveTimestampsAsync('compute')) ?? NaN;
    } else {
        await device.queue.onSubmittedWorkDone();
        const start = performance.now();
        renderer.compute(kernel);
        await device.queue.onSubmittedWorkDone();
        ms = performance.now() - start;
    }
    return { ms, rays: ((await raysCounted(t)) - before) >>> 0 };
}

// Warm-up (uploads, first dispatch), then rounds.
u.frameIndex.value = 1;
for (const t of tracers) for (const kernel of [t.full, t.closest]) for (let i = 0; i < (quick ? 1 : 3); i++) await dispatch(t, kernel);
const rounds = quick ? 2 : 7;
const perRound = quick ? 2 : 5;
const rates = new Map<string, number[]>();
const push = (key: string, value: number) => rates.set(key, [...(rates.get(key) ?? []), value]);
for (let round = 0; round < rounds; round++) {
    const who = round % 2 === 1 ? [...tracers].reverse() : tracers;
    for (const t of who) {
        const full: { ms: number; rays: number }[] = [];
        const closest: { ms: number; rays: number }[] = [];
        for (let i = 0; i < perRound; i++) {
            // The same frame index for every walk: the same jitter, the same closest-hit rays.
            u.frameIndex.value = 100 + i;
            full.push(await dispatch(t, t.full));
            closest.push(await dispatch(t, t.closest));
        }
        const fullMs = median(full.map((d) => d.ms));
        const closestMs = median(closest.map((d) => d.ms));
        const fullRays = median(full.map((d) => d.rays));
        const closestRays = median(closest.map((d) => d.rays));
        push(`${t.id}/path`, (fullRays / fullMs) * 1000);
        push(`${t.id}/closest`, (closestRays / closestMs) * 1000);
        push(`${t.id}/any`, fullMs > closestMs ? ((fullRays - closestRays) / (fullMs - closestMs)) * 1000 : NaN);
        push(`${t.id}/rays`, fullRays);
        push(`${t.id}/shadow-rays`, fullRays - closestRays);
        await yieldNow();
    }
}

// First hits of one frame, every pixel, against ours.
u.frameIndex.value = 7;
const firstHits = new Map<string, Float32Array>();
for (const t of tracers) {
    renderer.compute(t.closest);
    firstHits.set(t.id, new Float32Array(await renderer.getArrayBufferAsync(t.targets.primary!)));
}
const scale = sceneBox.getSize(new Vector3()).length();
const reference = firstHits.get('ours')!;
const agreements = new Map<string, Agreement>();
for (const t of tracers) {
    const agreement = emptyAgreement();
    const hits = firstHits.get(t.id)!;
    for (let i = 0; i < W * H; i++) checkFirst(agreement, i, reference[i] >= 0 ? { triangle: 0, t: reference[i] } : null, hits[i] >= 0 ? { triangle: 0, t: hits[i] } : null, scale);
    agreements.set(t.id, agreement);
}
// Pixels through a silhouette may graze either way between two different triangle tests: up to 0.1% is reported, not failed.
const allowed = Math.max(2, Math.floor(W * H * 0.001));
const scene_ = model ?? `default-${Math.round(triangles)}`;
const measurements: Measurement[] = [];
for (const t of tracers) {
    const agreement = agreements.get(t.id)!;
    for (const metric of ['path', 'closest', 'any'] as const) {
        const values = rates.get(`${t.id}/${metric}`)!.filter(Number.isFinite);
        const rays = median(rates.get(`${t.id}/${metric === 'any' ? 'shadow-rays' : 'rays'}`)!);
        const what =
            metric === 'path'
                ? `every ray of the full tracer (closest and any hit, ${rays.toLocaleString('en-US')} per frame)`
                : metric === 'closest'
                  ? `closest-hit rays of the tracer with its shadow rays compiled out (${rays.toLocaleString('en-US')} per frame)`
                  : `any-hit (shadow) rays: their count over the time the full tracer takes beyond the closest-only one (${rays.toLocaleString('en-US')} per frame)`;
        measurements.push({
            page: 'pathtracer',
            scene: scene_,
            library: t.id,
            metric,
            value: values.length ? median(values) : null,
            unit: 'rays/s',
            method: `${W}×${H}, ${u.bounces.value} bounces: ${what}; per round the median of ${perRound} dispatches (${timing}), median of ${rounds} rounds, the walks back to back, the order reversed every other round`,
            agreement: metric === 'closest' ? agreement : undefined,
            ranked: agreement.disagree <= allowed,
            note: values.length ? undefined : 'the full tracer was not slower than the closest-only one',
        });
    }
}
for (const variant of ['ours', 'ours-gpu-built']) {
    for (const metric of ['path', 'closest', 'any']) {
        const a = rates.get(`${variant}/${metric}`)!;
        const b = rates.get(`tmbvh/${metric}`)!;
        const perRoundRatio = a.map((rate, i) => rate / b[i]).filter(Number.isFinite);
        measurements.push({
            page: 'pathtracer',
            scene: scene_,
            library: `${variant}-over-tmbvh`,
            metric: `${metric}/ratio`,
            value: perRoundRatio.length ? median(perRoundRatio) : null,
            unit: 'ratio',
            method: `${variant}'s rate over three-mesh-bvh's within each round, median of ${rounds} rounds`,
            ranked: true,
        });
    }
}
const names = {
    ours: '@pmndrs/bvh (BvhTraceNodes, tinybvh trees)',
    tmbvh: 'three-mesh-bvh/webgpu (BVHComputeData, SAH)',
    'ours-gpu-built': '@pmndrs/bvh, GPU-built trees (H-PLOC)',
    'ours-over-tmbvh': 'ours / three-mesh-bvh, paired',
    'ours-gpu-built-over-tmbvh': 'ours (GPU-built) / three-mesh-bvh, paired',
};
panel.table(`Path tracer, ${W}×${H}, rays per second`, measurements, ['path', 'closest', 'any'], names);
panel.table('Paired ratio (median of rounds; above 1: ours faster)', measurements, ['path/ratio', 'closest/ratio', 'any/ratio'], names);
for (const t of tracers) log(`${t.id}: first hits against ours: ${agreementText(agreements.get(t.id)!)}`);
panel.paragraph(`Timing: ${timing}. Rays are counted in the kernel. The any-hit rate is derived (full minus closest-only), so it is noisier than the other two. Both walks are at their defaults: ours merges the static meshes into one tree (mergeStaticInstances) and quantizes its nodes; three-mesh-bvh clusters them under its top level.`);
panel.whatElse(
    `Trees built on the GPU: a second SceneTrees over the same scene, its tinybvh trees held back, had every mesh traceable on GPU H-PLOC trees in ${gpuBuiltMs.toFixed(0)} ms (${gpuTrees.counts.onGpuTrees ? `${gpuTrees.counts.onGpuTrees} GPU trees` : 'this device fell back to tinybvh'}; a first GPU build includes compiling the builder's pipelines), and the same kernel traces them (the third row). three-mesh-bvh builds on the CPU: ${theirBuildMs.toFixed(0)} ms here before its first traced frame.`,
);
panel.csv(device);

// ---- Live view: one walk traces the view, accumulating.
const quad = new QuadMesh(new MeshBasicNodeMaterial());
let shown: Tracer = tracers[0];
const showTracer = (name: string) => {
    shown = tracers.find((t) => t.name === name)!;
    const material = quad.material as MeshBasicNodeMaterial;
    material.colorNode = texture(shown.targets.target, uv());
    material.needsUpdate = true;
    reset = true;
};
let reset = true;
showTracer(shown.name);
overlay.select('trace with', tracers.map((t) => t.name), shown.name, showTracer);
overlay.range('bounces', 0, 4, 1, u.bounces.value, (value) => {
    u.bounces.value = value;
    reset = true;
});
controls.addEventListener('change', () => (reset = true));
overlay.note(`Orbit to restart the accumulation. The view is ${W}×${H} (fixed at load). ?model= traces a glTF (one base colour per mesh, both walks).`);

const checked = [...agreements.entries()].filter(([id]) => id !== 'ours');
for (const [id, agreement] of checked) {
    const message = `${id}: ${agreement.disagree} of ${agreement.checked} first hits disagree with ours (${agreement.examples.slice(0, 3).join('; ')})`;
    if (agreement.disagree > allowed) {
        if (quick) throw new Error(message);
        log(message);
    }
}

let frame = 0;
let raysPerSecond = 0;
app.loop(async () => {
    frame++;
    if (reset) {
        (shown.targets.accum.array as Float32Array).fill(0);
        shown.targets.accum.needsUpdate = true;
        reset = false;
    }
    camera.updateMatrixWorld();
    setCamera();
    u.frameIndex.value = 1000 + frame;
    // Three's timestamp queries pile up unless resolved (with trackTimestamp on): drop the render pass's.
    if (timestamps && frame % 10 === 0) void renderer.resolveTimestampsAsync('render');
    if (frame % 20 === 0) {
        const { ms, rays } = await dispatch(shown, shown.full);
        raysPerSecond = (rays / ms) * 1000;
        overlay.rays(`${shown.id} (one frame)`, raysPerSecond);
    } else renderer.compute(shown.full);
    quad.render(renderer);
    if (isSmoke && frame >= 30) {
        await ready(device, {
            ...overlay.snapshot(),
            csv: panel.csvText,
            status: panel.statusText,
            size: `${W}x${H}`,
            rates: Object.fromEntries([...rates].map(([key, values]) => [key, median(values.filter(Number.isFinite))])),
            firstHits: Object.fromEntries([...agreements].map(([id, a]) => [id, { checked: a.checked, disagree: a.disagree }])),
            gpuBuiltMs,
            theirBuildMs,
        });
    }
    return false; // the quad is the frame
});

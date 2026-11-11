/**
 * three/raycast: `@pmndrs/bvh/three` speeds up `Mesh.raycast`.
 *
 * - CPU: the same random rays through three's own `Mesh.raycast` and through
 *   `acceleratedRaycast` (a tinybvh tree per geometry, `computeBoundsTree`),
 *   each given a few milliseconds per frame; the overlay shows both rates.
 *   The pointer is picked with whichever one is selected.
 * - GPU: the scene also goes into `SceneTrees` on the renderer's device
 *   (GPU H-PLOC trees first, tinybvh's from the worker pool swapped in), and
 *   a batch of rays is traced in one dispatch through `TraceBatchKernel`.
 *   A sample of the hits is drawn as points.
 */

import { BufferAttribute, BufferGeometry, Mesh, MeshStandardMaterial, Points, PointsMaterial, Raycaster, SphereGeometry, Vector2, Vector3, type Intersection, type Object3D, MeshBasicMaterial, Group, TorusKnotGeometry } from 'three';
import { RAY_FLOATS, SceneTrees } from '@pmndrs/bvh';
import { SceneTreesObject3DSync, acceleratedRaycast, computeBoundsTree, prepareBoundsTrees, type BvhBoundsTree } from '@pmndrs/bvh/three';
import { createApp } from '../../shared/app.ts';
import { GpuRays } from '../../shared/gpu-rays.ts';
import { enableDrop, frameObject, loadGltf, meshesOf, requestedModel, type LoadedModel } from '../../shared/loader.ts';
import { layoutInfo, SceneTreesLog, sumInfo } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

const app = await createApp('Raycast', 'three/raycast');
const { device, scene, camera, controls, overlay } = app;
await prepareBoundsTrees();

const threeRaycast = Mesh.prototype.raycast;
const SIZES = { '10k': [500, 10], '100k': [1000, 50], '1M': [5000, 100] } as const;
type Size = keyof typeof SIZES;

let mode: 'acceleratedRaycast' | 'three' = 'acceleratedRaycast';
let size: Size = '100k';
let batchSize = 65536;
let root: Object3D = new Group();
let meshes: Mesh[] = [];
let trees: SceneTrees | null = null;
let sync: SceneTreesObject3DSync | null = null;
let gpuRays: GpuRays | null = null;
let log = new SceneTreesLog();
const cpuRays: { origin: Vector3; direction: Vector3 }[] = [];
let lastBatch: { rate: number; hits: number } | null = null;

const marker = new Mesh(new SphereGeometry(1, 16, 12), new MeshBasicMaterial({ color: 0xffc878 }));
marker.visible = false;
const hitPoints = new Points(new BufferGeometry(), new PointsMaterial({ color: 0x6fcf97, size: 2, sizeAttenuation: false }));
scene.add(marker, hitPoints);

function knot(sizeName: Size): Object3D {
    const [tubular, radial] = SIZES[sizeName];
    const mesh = new Mesh(new TorusKnotGeometry(1, 0.3, tubular, radial), new MeshStandardMaterial({ color: 0x8fb3ff, roughness: 0.5 }));
    const group = new Group();
    group.add(mesh);
    return group;
}

async function useScene(next: Object3D, label: string): Promise<void> {
    scene.remove(root);
    sync?.dispose();
    trees?.dispose();
    gpuRays?.destroy();
    root = next;
    scene.add(root);
    const box = frameObject(root, camera, controls);
    const radius = box.getSize(new Vector3()).length() / 2;
    marker.scale.setScalar(radius * 0.012);
    meshes = meshesOf(root);

    // CPU: one tree per geometry, timed.
    const start = performance.now();
    const built: BvhBoundsTree[] = [];
    for (const geometry of new Set(meshes.map((mesh) => mesh.geometry))) built.push(computeBoundsTree.call(geometry));
    const cpuMs = performance.now() - start;
    const triangles = built.reduce((sum, tree) => sum + tree.layout.triangleCount, 0);
    overlay.tree(sumInfo(built.map((tree) => layoutInfo(tree.layout))));
    overlay.set('scene', label);
    overlay.set('triangles', triangles);
    overlay.set('CPU trees (tinybvh)', `${cpuMs.toFixed(1)} ms`);

    // The CPU rays: from a sphere around the scene toward random points in its box.
    const center = box.getCenter(new Vector3());
    cpuRays.length = 0;
    for (let i = 0; i < 4096; i++) {
        const origin = new Vector3().randomDirection().multiplyScalar(radius * 2).add(center);
        const target = new Vector3(Math.random(), Math.random(), Math.random()).multiply(box.getSize(new Vector3())).add(box.min);
        cpuRays.push({ origin, direction: target.sub(origin).normalize() });
    }

    // GPU: SceneTrees on the renderer's device, kept in step by the sync.
    log = new SceneTreesLog();
    trees = await SceneTrees.create(device, { label: 'raycast', onTree: log.onTree });
    sync = new SceneTreesObject3DSync(trees, root);
    gpuRays = await GpuRays.create(trees);
    updateTrees();
    await trees.whenBuilt();
    updateTrees();
    overlay.build(log.build(trees));
}

function updateTrees(): void {
    if (!trees || !sync) return;
    root.updateMatrixWorld(true);
    sync.update();
    const encoder = device.createCommandEncoder();
    trees.update(encoder);
    device.queue.submit([encoder.finish()]);
}

/** `batchSize` rays from a sphere around the scene toward its box, packed for the kernel. */
function batchRays(): Float32Array {
    const packed = new Float32Array(batchSize * RAY_FLOATS);
    for (let i = 0; i < batchSize; i++) {
        const ray = cpuRays[i % cpuRays.length];
        // Jitter the reused CPU rays a little so the batch covers the mesh.
        const d = ray.direction;
        packed.set([ray.origin.x, ray.origin.y, ray.origin.z, 0, d.x + (Math.random() - 0.5) * 0.02, d.y + (Math.random() - 0.5) * 0.02, d.z + (Math.random() - 0.5) * 0.02, 1e30], i * RAY_FLOATS);
        const base = i * RAY_FLOATS + 4;
        const length = Math.hypot(packed[base], packed[base + 1], packed[base + 2]);
        for (let k = 0; k < 3; k++) packed[base + k] /= length;
    }
    return packed;
}

async function traceBatch(): Promise<void> {
    if (!gpuRays || gpuRays.pending) return;
    const packed = batchRays();
    const result = await gpuRays.traceClosest(packed);
    const ms = result.gpuMs ?? result.wallMs;
    lastBatch = { rate: (result.count / ms) * 1000, hits: result.hits };
    overlay.rays(result.gpuMs !== null ? 'GPU batch (timestamps)' : 'GPU batch (wall)', lastBatch.rate);
    overlay.set('GPU batch', `${result.count.toLocaleString('en-US')} rays, ${result.hits.toLocaleString('en-US')} hits, ${ms.toFixed(2)} ms`);
    const shown = Math.min(result.count, 16384);
    const positions = new Float32Array(shown * 3);
    let n = 0;
    for (let i = 0; i < shown; i++) {
        const hit = result.hit(i);
        if (!hit) continue;
        const base = i * RAY_FLOATS;
        for (let k = 0; k < 3; k++) positions[n * 3 + k] = packed[base + k] + packed[base + 4 + k] * hit.t;
        n++;
    }
    hitPoints.geometry.setAttribute('position', new BufferAttribute(positions.subarray(0, n * 3), 3));
    hitPoints.geometry.setDrawRange(0, n);
}

// CPU rates: a few milliseconds per frame for each raycast, the counts averaged over a second.
const raycaster = new Raycaster();
raycaster.firstHitOnly = true; // three-mesh-bvh's flag, typed by /three's augmentation of three
const rates = { three: { rays: 0, ms: 0 }, accelerated: { rays: 0, ms: 0 } };
let nextRay = 0;
let rateStart = performance.now();
const hits: Intersection[] = [];

function castCpu(raycast: typeof threeRaycast, budgetMs: number, bucket: { rays: number; ms: number }): void {
    const start = performance.now();
    let elapsed = 0;
    do {
        const ray = cpuRays[nextRay++ % cpuRays.length];
        raycaster.ray.origin.copy(ray.origin);
        raycaster.ray.direction.copy(ray.direction);
        hits.length = 0;
        for (const mesh of meshes) raycast.call(mesh, raycaster, hits);
        bucket.rays++;
        elapsed = performance.now() - start;
    } while (elapsed < budgetMs);
    bucket.ms += elapsed;
}

const pointer = new Vector2(2, 2);
addEventListener('pointermove', (event) => pointer.set((event.clientX / innerWidth) * 2 - 1, -(event.clientY / innerHeight) * 2 + 1));

overlay.select('pick with', ['acceleratedRaycast', 'three'] as const, mode, (value) => (mode = value));
overlay.select('mesh', Object.keys(SIZES) as Size[], size, (value) => {
    size = value;
    void useScene(knot(size), `torus knot ${size} triangles`);
});
overlay.select('GPU batch', ['4096', '65536', '1048576'] as const, '65536', (value) => (batchSize = Number(value)));
overlay.note('Drop a glTF to raycast it instead. Rates: rays traced in a few ms per frame, averaged over a second.');

const model = requestedModel();
if (model) {
    const loaded = await loadGltf(model);
    await useScene(loaded.root, loaded.name);
} else {
    await useScene(knot(size), `torus knot ${size} triangles`);
}
enableDrop((loaded: LoadedModel) => useScene(loaded.root, loaded.name), (error) => overlay.error(String(error)));

let frame = 0;
app.loop(async () => {
    frame++;
    castCpu(threeRaycast, 3, rates.three);
    castCpu(acceleratedRaycast, 3, rates.accelerated);
    const now = performance.now();
    if (now - rateStart > 1000) {
        overlay.rays('three Mesh.raycast (CPU)', (rates.three.rays / rates.three.ms) * 1000);
        overlay.rays('acceleratedRaycast (CPU)', (rates.accelerated.rays / rates.accelerated.ms) * 1000);
        overlay.set('speed-up', `${((rates.accelerated.rays / rates.accelerated.ms) / (rates.three.rays / rates.three.ms)).toFixed(0)}×`);
        rates.three = { rays: 0, ms: 0 };
        rates.accelerated = { rays: 0, ms: 0 };
        rateStart = now;
    }
    // The pointer.
    raycaster.setFromCamera(pointer, camera);
    hits.length = 0;
    const pickStart = performance.now();
    for (const mesh of meshes) (mode === 'three' ? threeRaycast : acceleratedRaycast).call(mesh, raycaster, hits);
    const pickMs = performance.now() - pickStart;
    hits.sort((a, b) => a.distance - b.distance);
    marker.visible = hits.length > 0;
    if (hits[0]) marker.position.copy(hits[0].point);
    overlay.set('pick', hits[0] ? `face ${hits[0].faceIndex}, ${pickMs.toFixed(3)} ms` : `miss, ${pickMs.toFixed(3)} ms`);
    // The GPU batch every 30 frames (trees swap to tinybvh's as they arrive).
    if (frame % 30 === 1) {
        updateTrees();
        if (trees) overlay.build(log.build(trees));
        await traceBatch();
    }
    // Smoke: once both CPU rates and a GPU batch are on the overlay.
    const shown = overlay.snapshot().rays as Record<string, number> | undefined;
    if (isSmoke && lastBatch && shown?.['acceleratedRaycast (CPU)']) {
        if (lastBatch.hits === 0) throw new Error('the GPU batch hit nothing');
        // acceleratedRaycast must answer what three's own raycast answers.
        let checked = 0;
        for (const ray of cpuRays.slice(0, 64)) {
            raycaster.ray.origin.copy(ray.origin);
            raycaster.ray.direction.copy(ray.direction);
            const ours: Intersection[] = [];
            const theirs: Intersection[] = [];
            for (const mesh of meshes) {
                acceleratedRaycast.call(mesh, raycaster, ours);
                threeRaycast.call(mesh, raycaster, theirs);
            }
            const a = ours.sort((x, y) => x.distance - y.distance)[0];
            const b = theirs.sort((x, y) => x.distance - y.distance)[0];
            if (!a !== !b || (a && b && Math.abs(a.distance - b.distance) > 1e-4 * Math.max(1, b.distance))) throw new Error(`ray ${checked}: acceleratedRaycast ${a?.distance} vs three ${b?.distance}`);
            checked++;
        }
        await ready(device, { ...overlay.snapshot(), batchHits: lastBatch.hits, checkedAgainstThree: checked });
    }
});

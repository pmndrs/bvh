/**
 * compare/raycast: CPU first-hit and all-hit raycasts per second for every
 * library, the same seeded rays, each answer checked against ours; then a GPU
 * batch, ours against three-mesh-bvh's `/webgpu`. The pointer picks with the
 * library chosen in the overlay (live), and the overlay shows how long it took.
 */

import { Mesh, MeshBasicMaterial, Raycaster, SphereGeometry, Vector2 } from 'three';
import { MESH_ADAPTERS } from '../../../compare/adapters/index.ts';
import { diagonal } from '../../../compare/core/geometry.ts';
import { makeRays } from '../../../compare/core/rays.ts';
import { buildAll, raycastSuite, type Built } from '../../../compare/core/suites.ts';
import { createOursGpu, createTmbvhGpu, packRaysFor } from '../../../compare/gpu/gpu-trace.ts';
import { gpuSuite } from '../../../compare/gpu/gpu-suite.ts';
import { createComparePage, expectAgreement, quick } from '../../shared/compare.ts';
import { ready } from '../../shared/smoke.ts';

const page = await createComparePage('compare/raycast', 'compare/raycast', 'CPU raycasts per second (first hit and all hits, rays from outside and from inside the mesh) and a GPU batch.', { trackTimestamp: true });
const { app, geometry, panel } = page;
const log = (line: string) => panel.status(line);

const adapters = quick ? MESH_ADAPTERS.filter((adapter) => adapter.id !== 'babylon-brute') : MESH_ADAPTERS;
const { built } = await buildAll(adapters, geometry, { log, buildRuns: 1 });
const plan = quick ? { runs: 3, budgetMs: 250 } : { runs: 5, budgetMs: 1500 };
const cpu = await raycastSuite(built, geometry, { log, rays: quick ? 512 : 4096, plan });
panel.table('CPU raycasts, rays per second', cpu, ['first/outside', 'first/inside', 'all/outside', 'all/inside']);

// The GPU batch: the same rays, many more of them.
const count = quick ? 16_384 : 262_144;
const rays = makeRays(geometry, 'inside', count, 2);
const tracers = [await createOursGpu(app.device, geometry), await createTmbvhGpu(app.renderer, geometry)];
const gpu = await gpuSuite(tracers, page.sceneName, { closest: packRaysFor(rays, 1e30), any: packRaysFor(rays, diagonal(geometry) * 0.25), count, anyTMax: diagonal(geometry) * 0.25, scale: diagonal(geometry), set: 'inside' }, quick ? { rounds: 2, dispatches: 3, warmups: 1 } : {}, log);
panel.table(`GPU batch, ${count.toLocaleString('en-US')} rays (paired rounds)`, gpu, ['closest/inside', 'any/inside', 'closest/inside/ratio', 'any/inside/ratio', 'gpu-bytes'], { ours: '@pmndrs/bvh', tmbvh: 'three-mesh-bvh/webgpu', 'ours-over-tmbvh': 'ours / three-mesh-bvh' });
for (const tracer of tracers) tracer.dispose();

panel.whatElse(
    'The GPU batch above is a plain WebGPU compute pass (TraceBatchKernel) over buffers that three never sees: the same tree could be traced from another engine, a worker with its own device, or a raw pass beside three\'s renderer. three-mesh-bvh\'s GPU walk exists only as TSL/WGSL nodes compiled into three\'s compute.',
);
panel.csv(app.device);

// Live: the pointer picks with the library chosen.
const pickers = new Map<string, Built>(built.filter((entry) => entry.mesh.raycastFirst).map((entry) => [entry.adapter.name, entry]));
let picker = [...pickers.keys()][0];
app.overlay.select('pick with', [...pickers.keys()], picker, (value) => (picker = value));
const marker = new Mesh(new SphereGeometry(diagonal(geometry) * 0.006, 12, 8), new MeshBasicMaterial({ color: 0xffc878 }));
marker.visible = false;
app.scene.add(marker);
const pointer = new Vector2(2, 2);
addEventListener('pointermove', (event) => pointer.set((event.clientX / innerWidth) * 2 - 1, -(event.clientY / innerHeight) * 2 + 1));
const raycaster = new Raycaster();

// Smoke check: three-mesh-bvh (CPU and GPU) and three's own raycast answer what ours answers.
const checked = expectAgreement([...cpu, ...gpu], ['tmbvh-center', 'tmbvh-sah', 'three', 'tmbvh'], log);
let frames = 0;
app.loop(async () => {
    raycaster.setFromCamera(pointer, app.camera);
    const { origin, direction } = raycaster.ray;
    const entry = pickers.get(picker)!;
    const start = performance.now();
    const hit = entry.mesh.raycastFirst!({ origin: [origin.x, origin.y, origin.z], direction: [direction.x, direction.y, direction.z] });
    const ms = performance.now() - start;
    marker.visible = hit !== null;
    if (hit) marker.position.copy(origin).addScaledVector(direction, hit.t);
    app.overlay.set('pick', hit ? `triangle ${hit.triangle}, ${ms.toFixed(3)} ms` : `miss, ${ms.toFixed(3)} ms`);
    if (++frames === 10) await ready(app.device, { ...app.overlay.snapshot(), csv: panel.csvText, status: panel.statusText, checkedAgainstOurs: checked, measurements: panel.measurements().length });
});

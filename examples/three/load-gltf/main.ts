/**
 * three/load-gltf: the fast-first recipe on a real glTF.
 *
 * `SceneTrees` builds every mesh's first tree on the GPU (H-PLOC, one batch
 * for many meshes, ready in milliseconds), and at the same time hands each
 * mesh to a tinybvh worker pool (`@pmndrs/bvh/worker`). tinybvh's trees have
 * a lower SAH cost; each one replaces its mesh's GPU tree at the next
 * `SceneTrees.update` after it arrives. The timeline shows, per mesh, when its
 * GPU tree and its tinybvh tree were placed. A GPU batch of rays is traced
 * through whatever trees are in place, before and after the swap.
 *
 * Without `?model=`, a generated scene of 60 meshes from about 2k to 200k triangles.
 * Drop a glTF to load it instead.
 */

import { Color, Group, IcosahedronGeometry, Mesh, MeshStandardMaterial, SphereGeometry, TorusKnotGeometry, Vector3, type Object3D } from 'three';
import { GpuRays, RAY_FLOATS, SceneTrees, type SceneTreeEvent } from '@pmndrs/bvh';
import { SceneTreesObject3DSync } from '@pmndrs/bvh/three';
import { createBvhWorkerPool, defaultWorkerPoolSize, type BvhWorkerPool } from '@pmndrs/bvh/worker';
import { createApp } from '../../shared/app.ts';
import { enableDrop, frameObject, loadGltf, meshesOf, requestedModel, type LoadedModel } from '../../shared/loader.ts';
import { SceneTreesLog } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

const app = await createApp('Load glTF (fast first)', 'three/load-gltf');
const { device, scene, camera, controls, overlay } = app;

const params = new URLSearchParams(location.search);
let poolSize = Number(params.get('workers') ?? defaultWorkerPoolSize());
let gpuFirst = params.get('gpuFirst') !== '0';

/** 60 meshes of very different sizes, so the GPU batch and the pool both have work. */
function generatedScene(): Object3D {
    const root = new Group();
    for (let i = 0; i < 60; i++) {
        // 2k to 200k triangles, most of them small.
        const detail = Math.round(32 + Math.pow(Math.random(), 3) * 968);
        const geometry =
            i % 3 === 0
                ? new TorusKnotGeometry(0.6, 0.2, detail, Math.max(6, Math.round(detail / 10)))
                : i % 3 === 1
                  ? new SphereGeometry(0.7, Math.round(detail / 2.2), Math.round(detail / 4.4))
                  : new IcosahedronGeometry(0.7, Math.round(detail / 10));
        const mesh = new Mesh(geometry, new MeshStandardMaterial({ color: new Color().setHSL(Math.random(), 0.45, 0.6), roughness: 0.6 }));
        mesh.position.set((i % 10) * 2 - 9, Math.floor(i / 10) * 2 - 5, Math.sin(i) * 2);
        mesh.rotation.set(Math.random() * 6, Math.random() * 6, 0);
        root.add(mesh);
    }
    return root;
}

// The timeline panel.
const panel = document.createElement('div');
panel.style.cssText = 'position:fixed;right:10px;bottom:10px;width:min(560px,calc(100vw - 20px));background:rgba(10,12,16,.86);border:1px solid rgba(255,255,255,.12);border-radius:10px;padding:8px 10px;font:11px/1.4 ui-monospace,Menlo,monospace;color:#cfd8e3;z-index:10';
const caption = document.createElement('div');
const canvas = document.createElement('canvas');
canvas.style.cssText = 'display:block;width:100%;height:180px;margin-top:4px';
panel.append(caption, canvas);
document.body.append(panel);

function drawTimeline(events: readonly SceneTreeEvent[]): void {
    const width = (canvas.width = canvas.clientWidth * devicePixelRatio);
    const height = (canvas.height = canvas.clientHeight * devicePixelRatio);
    const context = canvas.getContext('2d')!;
    context.clearRect(0, 0, width, height);
    const meshes = new Map<number, { triangles: number; gpu?: number; tinybvh?: number; app?: number }>();
    for (const event of events) {
        if (event.mesh === null) continue;
        const entry = meshes.get(event.mesh) ?? { triangles: event.triangleCount };
        entry[event.builtBy] = event.ms;
        meshes.set(event.mesh, entry);
    }
    const rows = [...meshes.values()].sort((a, b) => b.triangles - a.triangles);
    const end = Math.max(1, ...rows.flatMap((row) => [row.gpu ?? 0, row.tinybvh ?? 0])) * 1.05;
    const left = 70 * devicePixelRatio;
    const rowHeight = Math.max(1, Math.min(10 * devicePixelRatio, height / Math.max(1, rows.length)));
    context.font = `${9 * devicePixelRatio}px ui-monospace, monospace`;
    rows.forEach((row, r) => {
        const y = r * rowHeight;
        const x = (ms: number) => left + ((width - left) * ms) / end;
        if (row.gpu !== undefined) {
            context.fillStyle = '#8fb3ff';
            context.fillRect(x(0), y + rowHeight * 0.3, Math.max(2, x(row.gpu) - x(0)), rowHeight * 0.2);
            context.fillRect(x(row.gpu) - 1, y, 3, rowHeight * 0.8);
        }
        if (row.tinybvh !== undefined) {
            context.fillStyle = '#ffc878';
            context.fillRect(x(row.tinybvh) - 1, y, 3, rowHeight * 0.8);
            if (row.gpu !== undefined) {
                context.fillStyle = 'rgba(143,179,255,0.25)';
                context.fillRect(x(row.gpu), y + rowHeight * 0.3, x(row.tinybvh) - x(row.gpu), rowHeight * 0.2);
            }
        }
        if (rowHeight >= 8 * devicePixelRatio || r % Math.ceil((9 * devicePixelRatio) / rowHeight) === 0) {
            context.fillStyle = '#8592a3';
            context.fillText(`${(row.triangles / 1000).toFixed(row.triangles < 10000 ? 1 : 0)}k`, 2, y + rowHeight * 0.8);
        }
    });
    caption.innerHTML = `per mesh, by triangles: <span style="color:#8fb3ff">▮ GPU H-PLOC tree placed</span> · <span style="color:#ffc878">▮ tinybvh tree swapped in</span> · 0 to ${end.toFixed(0)} ms after the first update`;
}

let current: { root: Object3D; trees: SceneTrees; sync: SceneTreesObject3DSync; pool: BvhWorkerPool; log: SceneTreesLog; rays: GpuRays } | null = null;
const traced: Record<string, number> = {};
let tracedFirst = false;
let tracedAfter = false;

async function load(root: Object3D, label: string, loadMs: number | null): Promise<void> {
    if (current) {
        scene.remove(current.root);
        current.sync.dispose();
        current.trees.dispose();
        current.pool.dispose();
        current.rays.destroy();
    }
    scene.add(root);
    frameObject(root, camera, controls);
    const meshes = meshesOf(root);
    const triangles = meshes.reduce((sum, mesh) => sum + (mesh.geometry.index ? mesh.geometry.index.count : mesh.geometry.attributes.position.count) / 3, 0);
    overlay.set('scene', label);
    overlay.set('meshes', `${meshes.length} (${Math.round(triangles).toLocaleString('en-US')} triangles)`);
    if (loadMs !== null) overlay.set('glTF load', `${loadMs.toFixed(0)} ms`);

    const pool = createBvhWorkerPool({ size: poolSize });
    overlay.set('worker pool', `${pool.size} worker${pool.size === 1 ? '' : 's'}${pool.usesWorkers ? '' : ' (inline: no Worker)'}`);
    const log = new SceneTreesLog();
    const trees = await SceneTrees.create(device, { label: 'load-gltf', workers: pool, gpuFirstTrees: gpuFirst, onTree: log.onTree });
    const sync = new SceneTreesObject3DSync(trees, root, { static: true });
    const rays = await GpuRays.create(device, trees);
    current = { root, trees, sync, pool, log, rays };
    // Start the builds, then compile the scene's materials before its first frame (compileAsync), placing trees
    // as they arrive meanwhile. Otherwise the first render compiles every pipeline synchronously, and no
    // SceneTrees.update runs (so no tree is placed) until it is done.
    update();
    const placing = setInterval(() => {
        update();
        if (!tracedFirst && trees.counts.onGpuTrees > 0) {
            tracedFirst = true;
            void trace('on GPU trees');
        }
    }, 8);
    const compileStart = performance.now();
    try {
        await app.renderer.compileAsync(scene, camera);
    } finally {
        clearInterval(placing);
    }
    overlay.set('material compile', `${(performance.now() - compileStart).toFixed(0)} ms`);
    for (const key of Object.keys(traced)) delete traced[key];
    tracedFirst = false;
    tracedAfter = false;
}

function update(): void {
    if (!current) return;
    current.root.updateMatrixWorld();
    current.sync.update();
    const encoder = device.createCommandEncoder({ label: 'load-gltf update' });
    current.trees.update(encoder);
    device.queue.submit([encoder.finish()]);
}

/** 65,536 rays from the camera's side of the scene through whatever trees are placed now. */
async function trace(phase: string): Promise<void> {
    if (!current || current.rays.pending) return;
    const count = 65536;
    const packed = new Float32Array(count * RAY_FLOATS);
    const target = controls.target;
    const spread = camera.position.distanceTo(target) * 0.4;
    for (let i = 0; i < count; i++) {
        const d = new Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).multiplyScalar(spread).add(target).sub(camera.position).normalize();
        packed.set([camera.position.x, camera.position.y, camera.position.z, 0, d.x, d.y, d.z, 1e30], i * RAY_FLOATS);
    }
    const result = await current.rays.traceClosest(packed);
    const ms = result.gpuMs ?? result.wallMs;
    traced[phase] = (count / ms) * 1000;
    overlay.rays(`${phase} (${result.gpuMs !== null ? 'timestamps' : 'wall'})`, traced[phase]);
    if (result.hits === 0) overlay.set(`${phase} hits`, 0);
}

overlay.range('pool workers', 1, 8, 1, poolSize, (value) => (poolSize = value));
overlay.checkbox('GPU first trees', gpuFirst, (value) => (gpuFirst = value));
overlay.button('reload', () => void reload());
overlay.note('Drop a glTF, or pass ?model=downloads/<asset>/<file>.gltf (served from BVH_ASSETS_DIR).');

let source: { path: string | null; loaded: LoadedModel | null } = { path: requestedModel(), loaded: null };
async function reload(): Promise<void> {
    if (source.loaded) await load(source.loaded.root, source.loaded.name, source.loaded.loadMs);
    else if (source.path) {
        source.loaded = await loadGltf(source.path);
        await load(source.loaded.root, source.loaded.name, source.loaded.loadMs);
    } else await load(generatedScene(), 'generated (60 meshes)', null);
}
await reload();
enableDrop(async (model) => {
    source = { path: null, loaded: model };
    await reload();
}, (error) => overlay.error(String(error)));

function refresh(): void {
    if (!current) return;
    const { trees, log, pool, sync } = current;
    overlay.build(log.build(trees));
    overlay.tree(log.tree(trees, sync.meshHandles));
    const counts = trees.counts;
    overlay.set('swaps', `${counts.swaps} (${counts.tinybvhBuilding} tinybvh building, ${pool.stats.queued} queued)`);
    const timeline = trees.timeline;
    overlay.set('first trees', timeline.firstTreesMs === null ? '-' : `${timeline.firstTreesMs.toFixed(0)} ms`);
    overlay.set('all tinybvh', timeline.tinybvhTreesMs === null ? '-' : `${timeline.tinybvhTreesMs.toFixed(0)} ms`);
    overlay.set('memory now / peak', `${(trees.memory.total / 1048576).toFixed(1)} / ${(trees.memory.peak / 1048576).toFixed(1)} MB`);
    drawTimeline(log.events);
}

let frame = 0;
app.loop(async () => {
    frame++;
    update();
    if (!current) return;
    const { trees, log } = current;
    if (frame % 5 === 0) refresh();
    if (!tracedFirst && trees.counts.onGpuTrees > 0) {
        tracedFirst = true;
        await trace('on GPU trees');
    }
    if (!tracedAfter && trees.timeline.tinybvhTreesMs !== null) {
        tracedAfter = true;
        await trace('on tinybvh trees');
        refresh();
        if (isSmoke) {
            const events = log.events;
            if (trees.gpuFirstTrees.enabled && !events.some((event) => event.builtBy === 'gpu')) throw new Error('no mesh got a GPU first tree');
            if (trees.counts.waitingForFirstTree > 0) throw new Error('meshes still without a tree');
            await ready(device, { ...overlay.snapshot(), events: events.length });
        }
    }
});

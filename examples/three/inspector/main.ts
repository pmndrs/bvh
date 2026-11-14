/**
 * three/inspector: one mesh's tree from each builder, side by side.
 *
 * - **Builders:** tinybvh (WebAssembly, on this thread), GPU H-PLOC and GPU
 *   LBVH (on the renderer's device, timed with timestamp queries where the
 *   device has them). Without `subgroups`, `createMeshLayoutBuilder` says why
 *   the GPU builders aren't used, and tinybvh builds instead.
 * - **Options:** the largest leaf (1 to 8 triangles), `triangleLeaves`
 *   (`'records'` or `'indexed'`) and `nodeFormat` (`'f32'` or `'quantized'`).
 *   The table keeps one row per build, so builders and options compare.
 * - **Boxes:** `MeshBVHHelper` draws the tree's boxes at a chosen depth.
 * - **Heat map:** the GPU walk's own work per ray from the camera: the
 *   tree in a one-instance scene on the device, traced by the counted walk
 *   (`GpuRays.countWalk`, the production walk with step counters, compiled
 *   in only for it): steps per ray (nodes and leaves processed, both levels).
 */

import { BufferAttribute, BufferGeometry, Mesh, MeshStandardMaterial, TorusKnotGeometry, Vector3, type Object3D } from 'three';
import { CpuMeshBvh, GpuRays, IDENTITY_TRANSFORM, RAY_FLOATS, uploadTraceScene, type GpuLayoutNodeFormat, type GpuLayoutTriangleLeaves, type MeshInput, type TraceSceneBuffers } from '@pmndrs/bvh';
import { MeshBVHHelper, meshInputFromGeometry } from '@pmndrs/bvh/three';
import { createApp } from '../../shared/app.ts';
import { assembleLayouts, MeshBuilds, type BuilderKind, type TimedBuild } from '../../shared/builders.ts';
import { enableDrop, frameObject, loadGltf, meshesOf, requestedModel } from '../../shared/loader.ts';
import { bytes } from '../../shared/overlay.ts';
import { layoutInfo } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

const app = await createApp('Inspector', 'three/inspector');
const { device, scene, camera, controls, overlay } = app;

type Builder = BuilderKind;
const settings = {
    builder: 'tinybvh' as Builder,
    leaf: 4,
    triangleLeaves: 'records' as GpuLayoutTriangleLeaves,
    nodeFormat: 'f32' as GpuLayoutNodeFormat,
    depth: 8,
    parents: false,
    heat: false,
};

// What is inspected: one mesh. A loaded model's meshes are merged into one geometry in world space.
let mesh: Mesh | null = null;
let input: MeshInput;
function useObject(root: Object3D, label: string): void {
    if (mesh) scene.remove(mesh);
    root.updateMatrixWorld(true);
    const parts = meshesOf(root);
    const positions: number[] = [];
    const indices: number[] = [];
    const v = new Vector3();
    for (const part of parts) {
        const partInput = meshInputFromGeometry(part.geometry);
        const base = positions.length / 3;
        const count = partInput.positions.length / 3;
        for (let i = 0; i < count; i++) positions.push(...v.fromArray(partInput.positions, i * 3).applyMatrix4(part.matrixWorld).toArray());
        if (partInput.indices) for (const index of partInput.indices) indices.push(base + index);
        else for (let i = 0; i < count; i++) indices.push(base + i);
    }
    input = { positions: new Float32Array(positions), indices: new Uint32Array(indices) };
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(input.positions, 3));
    geometry.setIndex(new BufferAttribute(input.indices!, 1));
    geometry.computeVertexNormals();
    mesh = new Mesh(geometry, new MeshStandardMaterial({ color: 0x8fb3ff, roughness: 0.6, transparent: true, opacity: 0.85 }));
    scene.add(mesh);
    overlay.set('mesh', `${label}: ${(indices.length / 3).toLocaleString('en-US')} triangles${parts.length > 1 ? `, ${parts.length} meshes merged` : ''}`);
    frameObject(mesh, camera, controls);
}

// The builders (shared/builders.ts: the chooser's fallback, warm-up, timestamps).
const builds = await MeshBuilds.create(device);
type Built = TimedBuild;

async function build(): Promise<Built> {
    return builds.build(settings.builder, input, { maxLeafPrimitives: settings.leaf, triangleLeaves: settings.triangleLeaves, nodeFormat: settings.nodeFormat });
}

// The comparison table.
const table = document.createElement('table');
table.style.cssText = 'position:fixed;right:10px;bottom:10px;z-index:10;border-collapse:collapse;background:rgba(10,12,16,.86);border:1px solid rgba(255,255,255,.12);font:11px/1.5 ui-monospace,Menlo,monospace;color:#cfd8e3;max-width:calc(100vw - 20px);display:block;overflow:auto;max-height:40vh';
table.innerHTML = '<thead><tr style="color:#8592a3"><th>builder</th><th>leaf</th><th>leaves</th><th>nodes</th><th>build CPU</th><th>build GPU</th><th>SAH</th><th>depth</th><th>node count</th><th>bytes</th><th>steps/ray</th></tr></thead><tbody></tbody>';
for (const cell of table.querySelectorAll('th')) cell.style.padding = '3px 8px';
document.body.append(table);

/** What is inspected now: the tree, its CPU query tree, its row, and the tree on the device for the heat map's counted walk. */
let current: { built: Built; tree: CpuMeshBvh; row: HTMLTableRowElement; buffers: TraceSceneBuffers; rays: GpuRays } | null = null;
let helper: MeshBVHHelper | null = null;

async function rebuild(): Promise<void> {
    if (!mesh) return;
    const built = await build();
    const tree = new CpuMeshBvh(built.layout, { source: input });
    // The tree in a one-instance scene on the device, in the formats it came in, for the counted walk.
    const buffers = uploadTraceScene(device, assembleLayouts([built.layout], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY_TRANSFORM }], [input as { positions: Float32Array; indices: Uint32Array }]), 'inspector');
    const rays = await GpuRays.create(device, buffers, { label: 'inspector heat map' });
    const info = layoutInfo(built.layout);
    overlay.build({ builtBy: built.builtBy, fallbackReason: built.fallbackReason, cpuMs: built.cpuMs, gpuMs: built.gpuMs });
    overlay.tree(info);
    const row = document.createElement('tr');
    const cells = [
        built.builtBy,
        String(settings.leaf),
        settings.triangleLeaves,
        settings.nodeFormat,
        `${built.cpuMs.toFixed(1)} ms`,
        built.gpuMs === undefined ? '-' : built.gpuMs === null ? 'n/a' : `${built.gpuMs.toFixed(2)} ms`,
        info.sah!.toFixed(2),
        String(info.depth),
        info.nodes!.toLocaleString('en-US'),
        bytes(info.bytes),
        '-',
    ];
    for (const text of cells) {
        const cell = document.createElement('td');
        cell.textContent = text;
        cell.style.padding = '2px 8px';
        row.append(cell);
    }
    table.tBodies[0].prepend(row);
    const old = current;
    current = { built, tree, row, buffers, rays };
    // The old scene goes once a heat map still tracing it is done.
    if (old) void old.rays.retire().then(() => old.buffers.destroy());
    if (helper) {
        scene.remove(helper);
        helper.dispose();
    }
    helper = new MeshBVHHelper(mesh, tree, settings.depth);
    helper.displayParents = settings.parents;
    helper.update();
    scene.add(helper);
    heatDirty = true;
}

// The heat map: a small canvas over the view, one ray per cell.
const heat = document.createElement('canvas');
heat.style.cssText = 'position:fixed;inset:0;width:100vw;height:100vh;pointer-events:none;image-rendering:pixelated;opacity:.8;z-index:5';
heat.hidden = true;
document.body.append(heat);
let heatDirty = true;
controls.addEventListener('change', () => (heatDirty = true));

/** The heat map from the GPU's counted walk: steps per ray, one ray per cell. Also how many cells' rays hit. */
async function heatMap(): Promise<{ mean: number; max: number; hits: number; rays: number; gpuMs: number | null }> {
    const on = current;
    if (!on) return { mean: 0, max: 0, hits: 0, rays: 0, gpuMs: null };
    const columns = 160;
    const rows = Math.max(1, Math.round((columns * innerHeight) / innerWidth));
    const packed = new Float32Array(columns * rows * RAY_FLOATS);
    const origin = camera.position;
    const direction = new Vector3();
    for (let y = 0; y < rows; y++) {
        for (let x = 0; x < columns; x++) {
            direction.set(((x + 0.5) / columns) * 2 - 1, 1 - ((y + 0.5) / rows) * 2, 0.5).unproject(camera).sub(origin).normalize();
            packed.set([origin.x, origin.y, origin.z, 0, direction.x, direction.y, direction.z, 1e30], (y * columns + x) * RAY_FLOATS);
        }
    }
    const result = await on.rays.countWalk(packed);
    heat.width = columns;
    heat.height = rows;
    const context = heat.getContext('2d')!;
    const image = context.createImageData(columns, rows);
    let hits = 0;
    for (let i = 0; i < result.count; i++) {
        const counts = result.counts(i);
        if (counts.hit) hits++;
        const f = Math.min(1, counts.steps / Math.max(1, result.maxSteps));
        // Dark blue (few) through orange to white (many).
        image.data.set([Math.round(255 * Math.min(1, f * 2)), Math.round(255 * Math.max(0, f * 1.6 - 0.5)), Math.round(255 * (f < 0.5 ? 0.35 - f * 0.5 : (f - 0.5) * 2)), Math.round(110 + 145 * f)], i * 4);
    }
    context.putImageData(image, 0, 0);
    return { mean: result.meanSteps, max: result.maxSteps, hits, rays: result.count, gpuMs: result.gpuMs };
}

/** The heat map's rays (the same grid) through the CPU tree: how many hit. */
function cpuHitCount(): number {
    const columns = heat.width;
    const rows = heat.height;
    const direction = new Vector3();
    let hits = 0;
    for (let y = 0; y < rows; y++) {
        for (let x = 0; x < columns; x++) {
            direction.set(((x + 0.5) / columns) * 2 - 1, 1 - ((y + 0.5) / rows) * 2, 0.5).unproject(camera).sub(camera.position).normalize();
            if (current!.tree.raycastFirst({ origin: camera.position, direction }, { side: 2 })) hits++;
        }
    }
    return hits;
}

// Controls.
overlay.select('builder', ['tinybvh', 'gpu-hploc', 'gpu-lbvh'] as const, settings.builder, (value) => {
    settings.builder = value;
    void rebuild();
});
overlay.range('largest leaf', 1, 8, 1, settings.leaf, (value) => {
    settings.leaf = value;
    void rebuild();
});
overlay.select('triangleLeaves', ['records', 'indexed'] as const, settings.triangleLeaves, (value) => {
    settings.triangleLeaves = value;
    void rebuild();
});
overlay.select('nodeFormat', ['f32', 'quantized'] as const, settings.nodeFormat, (value) => {
    settings.nodeFormat = value;
    void rebuild();
});
overlay.range('helper depth', 0, 30, 1, settings.depth, (value) => {
    settings.depth = value;
    if (helper) {
        helper.depth = value;
        helper.update();
    }
});
overlay.checkbox('helper parents', settings.parents, (value) => {
    settings.parents = value;
    if (helper) {
        helper.displayParents = value;
        helper.update();
    }
});
overlay.checkbox('heat map (GPU walk steps per ray)', settings.heat, (value) => {
    settings.heat = value;
    heat.hidden = !value;
    heatDirty = true;
});
overlay.note('Each build adds a row to the table. Drop a glTF to inspect it (its meshes merged).');

const model = requestedModel();
if (model) {
    const loaded = await loadGltf(model);
    useObject(loaded.root, loaded.name);
} else {
    useObject(new Mesh(new TorusKnotGeometry(1, 0.3, 600, 40)), 'torus knot');
}
enableDrop(async (loaded) => {
    useObject(loaded.root, loaded.name);
    await rebuild();
}, (error) => overlay.error(String(error)));
await rebuild();

let frame = 0;
let smokeRuns = 0;
const smokeRows: unknown[] = [];
app.loop(async () => {
    frame++;
    if (settings.heat && heatDirty && current && !current.rays.pending && frame % 10 === 0) {
        heatDirty = false;
        const on = current;
        const { mean, max, gpuMs } = await heatMap();
        overlay.set('heat map', `${mean.toFixed(1)} steps per ray (max ${max}), the GPU's counted walk${gpuMs !== null ? `, ${gpuMs.toFixed(2)} ms` : ''}`);
        on.row.lastElementChild!.textContent = mean.toFixed(1);
    }
    // Smoke: every builder, both node formats, indexed once, and the heat map.
    if (isSmoke && frame % 5 === 0 && smokeRuns < 6) {
        const runs: [Builder, GpuLayoutTriangleLeaves, GpuLayoutNodeFormat][] = [
            ['tinybvh', 'records', 'f32'],
            ['gpu-hploc', 'records', 'f32'],
            ['gpu-lbvh', 'records', 'f32'],
            ['tinybvh', 'indexed', 'quantized'],
            ['gpu-hploc', 'indexed', 'quantized'],
            ['gpu-lbvh', 'indexed', 'quantized'],
        ];
        [settings.builder, settings.triangleLeaves, settings.nodeFormat] = runs[smokeRuns++];
        settings.heat = true;
        heat.hidden = false;
        await rebuild();
        const { mean, hits, rays } = await heatMap();
        if (!(mean > 1)) throw new Error(`heat map: ${mean} steps per ray`);
        // The counted walk is the production walk: its rays hit what the CPU walk hits (a grazing ray may differ).
        const cpuHits = cpuHitCount();
        if (Math.abs(hits - cpuHits) > Math.max(2, rays * 0.01)) throw new Error(`heat map: the counted walk hit ${hits} of ${rays} rays, the CPU walk ${cpuHits}`);
        const info = layoutInfo(current!.built.layout);
        smokeRows.push({ builder: current!.built.builtBy, leaves: settings.triangleLeaves, nodes: settings.nodeFormat, cpuMs: current!.built.cpuMs, gpuMs: current!.built.gpuMs, sah: info.sah, stepsPerRay: mean, heatHits: hits });
        if (smokeRuns === 6) await ready(device, { ...overlay.snapshot(), builds: smokeRows });
    }
});

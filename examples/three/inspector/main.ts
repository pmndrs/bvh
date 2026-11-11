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
 * - **Heat map:** nodes an ideal closest-hit walk enters per ray from the
 *   camera: the ray's closest hit (`raycastFirst`), then every box entered
 *   before it (`traverse`), on the CPU over the same tree. The GPU walk's own
 *   step counters aren't public (examples/README.md).
 */

import { BufferAttribute, BufferGeometry, Mesh, MeshStandardMaterial, TorusKnotGeometry, Vector3, type Object3D } from 'three';
import {
    BVH_INTERSECTED,
    BVH_NOT_INTERSECTED,
    CpuMeshBvh,
    createMeshLayoutBuilder,
    createTinyBvhBuilder,
    GpuHplocBuilder,
    GpuLbvhBuilder,
    indexedFromRecords,
    type GpuLayoutMeshBvh,
    type GpuLayoutNodeFormat,
    type GpuLayoutTriangleLeaves,
    type GpuTreeBuffers,
    type MeshInput,
} from '@pmndrs/bvh';
import { MeshBVHHelper, meshInputFromGeometry } from '@pmndrs/bvh/three';
import { createApp } from '../../shared/app.ts';
import { GpuTimer } from '../../shared/gpu-timer.ts';
import { enableDrop, frameObject, loadGltf, meshesOf, requestedModel } from '../../shared/loader.ts';
import { bytes } from '../../shared/overlay.ts';
import { layoutInfo } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

const app = await createApp('Inspector', 'three/inspector');
const { device, scene, camera, controls, overlay } = app;

type Builder = 'tinybvh' | 'gpu-hploc' | 'gpu-lbvh';
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

// The builders.
const tinybvh = await createTinyBvhBuilder();
const timer = GpuTimer.create(device, 1);
const gpuBuilders = new Map<Builder, GpuHplocBuilder | GpuLbvhBuilder>();
async function gpuBuilder(kind: 'gpu-hploc' | 'gpu-lbvh'): Promise<GpuHplocBuilder | GpuLbvhBuilder> {
    let builder = gpuBuilders.get(kind);
    if (!builder) {
        builder = kind === 'gpu-hploc' ? new GpuHplocBuilder(device, { label: 'inspector' }) : new GpuLbvhBuilder(device, { label: 'inspector' });
        await builder.prepare();
        gpuBuilders.set(kind, builder);
    }
    return builder;
}

interface Built {
    readonly layout: GpuLayoutMeshBvh;
    readonly builtBy: Builder;
    readonly fallbackReason: string | null;
    readonly cpuMs: number;
    readonly gpuMs: number | null | undefined;
}

/** One GPU build, recorded and timed: the batch's passes in one timestamp pair. */
async function buildOnGpu(kind: 'gpu-hploc' | 'gpu-lbvh', options: { maxLeafPrimitives: number; triangleLeaves: GpuLayoutTriangleLeaves; nodeFormat: GpuLayoutNodeFormat }): Promise<Omit<Built, 'builtBy' | 'fallbackReason'>> {
    const builder = await gpuBuilder(kind);
    // The LBVH writes records only; indexed trees are converted after the readback.
    const lbvhIndexed = kind === 'gpu-lbvh' && options.triangleLeaves === 'indexed';
    const start = performance.now();
    const batch = builder.createBatch([input], lbvhIndexed ? { ...options, triangleLeaves: 'records' } : options);
    let trees: GpuTreeBuffers | null = null;
    try {
        const encoder = device.createCommandEncoder({ label: 'inspector build' });
        const timestampWrites = timer?.pass();
        batch.encode(encoder, timestampWrites);
        if (timer) timer.resolveInto(encoder);
        device.queue.submit([encoder.finish()]);
        trees = await batch.finish();
        const gpuMs = timer ? (await timer.result())[0] : null;
        const cpuMs = performance.now() - start;
        const problem = trees.meshes[0].problem;
        if (problem) throw new Error(`${kind} refused the mesh: ${problem}`);
        const [read] = await builder.readLayouts(trees);
        const layout = lbvhIndexed ? indexedFromRecords(read!, input) : read!;
        return { layout, cpuMs, gpuMs };
    } finally {
        trees?.destroy();
        batch.destroy();
    }
}

async function build(): Promise<Built> {
    const options = { maxLeafPrimitives: settings.leaf, triangleLeaves: settings.triangleLeaves, nodeFormat: settings.nodeFormat };
    if (settings.builder !== 'tinybvh') {
        // The chooser says whether this device can run the GPU builder at all, and why not.
        const chooser = await createMeshLayoutBuilder(device, { prefer: settings.builder });
        if (chooser.kind === settings.builder) {
            await buildOnGpu(settings.builder, options); // warm-up: the first build pays for pipeline and buffer setup
            return { ...(await buildOnGpu(settings.builder, options)), builtBy: settings.builder, fallbackReason: null };
        }
        const start = performance.now();
        const layout = tinybvh.buildLayout(input, options).layout;
        return { layout, builtBy: 'tinybvh', fallbackReason: chooser.fallbackReason, cpuMs: performance.now() - start, gpuMs: undefined };
    }
    tinybvh.buildLayout(input, options); // warm-up
    const start = performance.now();
    const layout = tinybvh.buildLayout(input, options).layout;
    return { layout, builtBy: 'tinybvh', fallbackReason: null, cpuMs: performance.now() - start, gpuMs: undefined };
}

// The comparison table.
const table = document.createElement('table');
table.style.cssText = 'position:fixed;right:10px;bottom:10px;z-index:10;border-collapse:collapse;background:rgba(10,12,16,.86);border:1px solid rgba(255,255,255,.12);font:11px/1.5 ui-monospace,Menlo,monospace;color:#cfd8e3;max-width:calc(100vw - 20px);display:block;overflow:auto;max-height:40vh';
table.innerHTML = '<thead><tr style="color:#8592a3"><th>builder</th><th>leaf</th><th>leaves</th><th>nodes</th><th>build CPU</th><th>build GPU</th><th>SAH</th><th>depth</th><th>node count</th><th>bytes</th><th>nodes/ray</th></tr></thead><tbody></tbody>';
for (const cell of table.querySelectorAll('th')) cell.style.padding = '3px 8px';
document.body.append(table);

let current: { built: Built; tree: CpuMeshBvh; row: HTMLTableRowElement } | null = null;
let helper: MeshBVHHelper | null = null;

async function rebuild(): Promise<void> {
    if (!mesh) return;
    const built = await build();
    const tree = new CpuMeshBvh(built.layout, { source: input });
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
    current = { built, tree, row };
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

function heatMap(): { mean: number; max: number } {
    if (!current) return { mean: 0, max: 0 };
    const tree = current.tree;
    const columns = 160;
    const rows = Math.max(1, Math.round((columns * innerHeight) / innerWidth));
    heat.width = columns;
    heat.height = rows;
    const counts = new Float32Array(columns * rows);
    const origin = camera.position.clone();
    const direction = new Vector3();
    const inverse = new Vector3();
    let entered = 0;
    let limit = Infinity;
    const callbacks = {
        intersectsBounds(box: Float32Array) {
            // Slab test: the box's entry distance along the ray, culled past the closest hit.
            let near = 0;
            let far = limit;
            for (let k = 0; k < 3; k++) {
                const o = origin.getComponent(k);
                const inv = inverse.getComponent(k);
                let t0 = (box[k] - o) * inv;
                let t1 = (box[k + 3] - o) * inv;
                if (t0 > t1) [t0, t1] = [t1, t0];
                near = Math.max(near, t0);
                far = Math.min(far, t1);
            }
            if (near > far) return BVH_NOT_INTERSECTED;
            entered++;
            return BVH_INTERSECTED;
        },
        intersectsRange: () => false,
    };
    let sum = 0;
    let max = 0;
    for (let y = 0; y < rows; y++) {
        for (let x = 0; x < columns; x++) {
            direction.set(((x + 0.5) / columns) * 2 - 1, 1 - ((y + 0.5) / rows) * 2, 0.5).unproject(camera).sub(origin).normalize();
            inverse.set(1 / direction.x, 1 / direction.y, 1 / direction.z);
            const hit = tree.raycastFirst({ origin, direction }, { side: 2 });
            limit = hit ? hit.distance : Infinity;
            entered = 1; // the root
            tree.traverse(callbacks);
            counts[y * columns + x] = entered;
            sum += entered;
            max = Math.max(max, entered);
        }
    }
    const context = heat.getContext('2d')!;
    const image = context.createImageData(columns, rows);
    for (let i = 0; i < counts.length; i++) {
        const f = Math.min(1, counts[i] / Math.max(1, max));
        // Dark blue (few) through orange to white (many).
        image.data.set([Math.round(255 * Math.min(1, f * 2)), Math.round(255 * Math.max(0, f * 1.6 - 0.5)), Math.round(255 * (f < 0.5 ? 0.35 - f * 0.5 : (f - 0.5) * 2)), Math.round(110 + 145 * f)], i * 4);
    }
    context.putImageData(image, 0, 0);
    return { mean: sum / counts.length, max };
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
overlay.checkbox('heat map (nodes per ray)', settings.heat, (value) => {
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
    if (settings.heat && heatDirty && current && frame % 10 === 0) {
        heatDirty = false;
        const start = performance.now();
        const { mean, max } = heatMap();
        overlay.set('heat map', `${mean.toFixed(1)} nodes per ray (max ${max}), ${(performance.now() - start).toFixed(0)} ms on the CPU`);
        current.row.lastElementChild!.textContent = mean.toFixed(1);
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
        const { mean } = heatMap();
        if (!(mean > 1)) throw new Error(`heat map: ${mean} nodes per ray`);
        const info = layoutInfo(current!.built.layout);
        smokeRows.push({ builder: current!.built.builtBy, leaves: settings.triangleLeaves, nodes: settings.nodeFormat, cpuMs: current!.built.cpuMs, gpuMs: current!.built.gpuMs, sah: info.sah, nodesPerRay: mean });
        if (smokeRuns === 6) await ready(device, { ...overlay.snapshot(), builds: smokeRows });
    }
});

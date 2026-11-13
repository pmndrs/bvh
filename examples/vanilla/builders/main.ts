/**
 * vanilla/builders: the three builders side by side on the same mesh, in raw
 * WebGPU. The builder guide in a page.
 *
 * For tinybvh (WebAssembly, on this thread), GPU H-PLOC and GPU LBVH (on this
 * page's device), each row shows:
 *
 * - build ms: CPU wall time, and GPU timestamps where the device has
 *   `timestamp-query` (shared/builders.ts; a warm-up build first);
 * - the tree: SAH cost (and against tinybvh's), depth, nodes, bytes;
 * - trace speed: the same seeded rays through `TraceBatchKernel`, closest
 *   hit, median of several timed dispatches;
 * - agreement: each tree's hits against tinybvh's (the same distance).
 *
 * Without subgroups the GPU rows are built by tinybvh, and the row says why.
 * Click a row to see its tree in the ray-cast view. A torus knot by default,
 * a `?model=` glTF or a dropped one (merged into one world-space mesh).
 */

import {
    createCpuTracer,
    createRandom,
    GpuLbvhBuilder,
    HIT_RECORD_BYTES,
    HIT_RECORD_WORD,
    IDENTITY_TRANSFORM,
    RAY_BYTES,
    RAY_FLOATS,
    TraceBatchKernel,
    uploadTraceScene,
    type TraceSceneBuffers,
    type TraceSceneData,
} from '@pmndrs/bvh';
import { requestedModel } from '../../shared/assets.ts';
import { BUILDER_KINDS, MeshBuilds, type BuilderKind, type TimedBuild } from '../../shared/builders.ts';
import { assembleLayouts, traceSourceOf, type TraceSource } from '../../shared/gpu-rays.ts';
import { GpuTimer } from '../../shared/gpu-timer.ts';
import { bytes, type TreeInfo } from '../../shared/overlay.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';
import { layoutInfo } from '../../shared/stats.ts';
import { createVanillaApp } from '../../shared/vanilla/app.ts';
import { checkAgainstCpu } from '../../shared/vanilla/check.ts';
import { mergeToWorld, singleMeshScene, torusKnot, type IndexedMesh, type VanillaScene } from '../../shared/vanilla/geometry.ts';
import { enableSceneDrop, loadGltfScene } from '../../shared/vanilla/gltf.ts';
import { vec3 } from '../../shared/vanilla/math.ts';
import { RayCastView, VIEW_MODES, type ViewMode } from '../../shared/vanilla/raycast-view.ts';

const app = await createVanillaApp('Builders', 'vanilla/builders');
const { device, camera, overlay, canvas } = app;

const params = new URLSearchParams(location.search);
const RAY_COUNT = isSmoke ? 65_536 : 262_144;
const DISPATCHES = isSmoke ? 3 : 7;
const settings = { leaf: 'default' as string, mode: 'normal' as ViewMode };

const builds = await MeshBuilds.create(device);
const timer = GpuTimer.create(device, 1);
const view = RayCastView.create(device, app.format);
const kernels = new Map<string, TraceBatchKernel>();

interface Row {
    readonly kind: BuilderKind;
    readonly built: TimedBuild;
    readonly info: TreeInfo;
    readonly data: TraceSceneData;
    readonly buffers: TraceSceneBuffers;
    readonly source: TraceSource;
    readonly traceMs: number | null;
    readonly timing: 'timestamps' | 'submit to done';
    /** t per ray (negative: a miss) and the hit triangle. */
    readonly t: Float32Array;
    readonly triangles: Uint32Array;
    agree: number | null;
    readonly element: HTMLTableRowElement;
}
interface Rays {
    readonly packed: Float32Array;
    readonly buffer: GPUBuffer;
    readonly results: GPUBuffer;
    readonly read: GPUBuffer;
}
/** One build pass: a mesh, its rays, a row per builder. Pages swap whole passes, never half of one. */
interface Pass {
    readonly mesh: IndexedMesh;
    readonly name: string;
    readonly radius: number;
    readonly rays: Rays;
    readonly rows: Row[];
}
let current: Pass | null = null;
let selected: Row | null = null;

// ---- The table.

const table = document.createElement('table');
table.style.cssText =
    'position:fixed;right:10px;bottom:10px;z-index:10;border-collapse:collapse;background:rgba(10,12,16,.88);border:1px solid rgba(255,255,255,.12);font:11px/1.5 ui-monospace,Menlo,monospace;color:#cfd8e3;max-width:calc(100vw - 20px);display:block;overflow:auto;max-height:45vh;cursor:pointer';
const HEADINGS = ['builder', 'built by', 'build CPU', 'build GPU', 'SAH', 'vs tinybvh', 'depth', 'nodes', 'bytes', 'Mrays/s', 'hits as tinybvh'];
table.innerHTML = `<thead><tr style="color:#8592a3">${HEADINGS.map((heading) => `<th style="padding:3px 8px;text-align:left">${heading}</th>`).join('')}</tr></thead><tbody></tbody>`;
document.body.append(table);
const caption = table.createCaption();
caption.style.cssText = 'caption-side:bottom;padding:4px 8px 6px;font:11px/1.4 system-ui,sans-serif;color:#8592a3;text-align:left';

function cells(row: Row, rows: readonly Row[]): string[] {
    const tiny = rows.find((other) => other.kind === 'tinybvh');
    const ms = (value: number | null | undefined, digits: number) => (value === undefined ? '-' : value === null ? 'n/a' : `${value.toFixed(digits)} ms`);
    return [
        row.kind,
        row.built.builtBy + (row.built.fallbackReason ? ' *' : ''),
        ms(row.built.cpuMs, 1),
        ms(row.built.gpuMs, 2),
        row.info.sah!.toFixed(2),
        tiny ? `${((row.info.sah! / tiny.info.sah!) * 100).toFixed(1)}%` : '-',
        String(row.info.depth),
        row.info.nodes!.toLocaleString('en-US'),
        bytes(row.info.bytes),
        row.traceMs ? (RAY_COUNT / (row.traceMs / 1000) / 1e6).toFixed(1) + (row.timing === 'timestamps' ? '' : ' (wall)') : '-',
        row.agree === null ? '-' : `${(row.agree * 100).toFixed(2)}%`,
    ];
}
function refreshTable(pass: Pass, status = ''): void {
    for (const row of pass.rows) {
        row.element.innerHTML = cells(row, pass.rows)
            .map((text) => `<td style="padding:2px 8px">${text}</td>`)
            .join('');
        row.element.style.background = row === selected ? 'rgba(255,200,120,0.14)' : '';
    }
    const reasons = pass.rows.filter((row) => row.built.fallbackReason).map((row) => `${row.kind}: ${row.built.fallbackReason}`);
    caption.textContent = `${pass.name}: ${(pass.mesh.indices.length / 3).toLocaleString('en-US')} triangles. ${RAY_COUNT.toLocaleString('en-US')} closest-hit rays from around the mesh, median of ${DISPATCHES} dispatches.${reasons.length ? ` * built by tinybvh instead (${reasons.join('; ')}).` : ''} Click a row to view its tree.${status ? ` ${status}` : ''}`;
}

function select(row: Row): void {
    selected = row;
    overlay.build({ builtBy: row.built.builtBy, fallbackReason: row.built.fallbackReason, cpuMs: row.built.cpuMs, gpuMs: row.built.gpuMs });
    overlay.tree(row.info);
    if (row.traceMs) overlay.rays(`closest, ${row.kind} tree${row.timing === 'timestamps' ? '' : ' (wall time)'}`, RAY_COUNT / (row.traceMs / 1000));
    if (current) refreshTable(current);
}

// ---- Work runs one job at a time (builds, row clicks): timings never overlap, and nothing is freed under a job.

let work: Promise<void> = Promise.resolve();
function queue(job: () => Promise<void>): Promise<void> {
    const next = work.then(job);
    work = next.catch((error: unknown) => overlay.error(String(error)));
    return next;
}

// ---- Rays: the same seeded set for every tree of a pass.

function makeRays(mesh: IndexedMesh): { rays: Rays; box: number[]; radius: number } {
    const random = createRandom(42);
    const packed = new Float32Array(RAY_COUNT * RAY_FLOATS);
    const b = bounds(mesh.positions);
    const center = [(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2];
    const radius = Math.hypot(b[3] - b[0], b[4] - b[1], b[5] - b[2]) / 2;
    for (let i = 0; i < RAY_COUNT; i++) {
        const z = random() * 2 - 1;
        const phi = random() * Math.PI * 2;
        const s = Math.sqrt(1 - z * z);
        const origin = [center[0] + Math.cos(phi) * s * radius * 2, center[1] + z * radius * 2, center[2] + Math.sin(phi) * s * radius * 2];
        const aim = [center[0] + (random() - 0.5) * radius, center[1] + (random() - 0.5) * radius, center[2] + (random() - 0.5) * radius];
        const d = vec3.normalize([aim[0] - origin[0], aim[1] - origin[1], aim[2] - origin[2]]);
        packed.set([origin[0], origin[1], origin[2], 0, d[0], d[1], d[2], 1e30], i * RAY_FLOATS);
    }
    const buffer = device.createBuffer({ label: 'builders rays', size: RAY_COUNT * RAY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, packed);
    const results = device.createBuffer({ label: 'builders hits', size: RAY_COUNT * HIT_RECORD_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ label: 'builders hits read', size: RAY_COUNT * HIT_RECORD_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    return { rays: { packed, buffer, results, read }, box: b, radius };
}

function bounds(positions: Float32Array): number[] {
    const b = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    for (let i = 0; i < positions.length; i += 3) {
        for (let k = 0; k < 3; k++) {
            b[k] = Math.min(b[k], positions[i + k]);
            b[k + 3] = Math.max(b[k + 3], positions[i + k]);
        }
    }
    return b;
}

async function kernelFor(source: TraceSource): Promise<TraceBatchKernel> {
    const key = `${source.triangleLeaves}/${source.positionLayout}/${source.nodeFormat}`;
    let kernel = kernels.get(key);
    if (!kernel) {
        kernel = await TraceBatchKernel.create(device, { label: 'builders', triangleLeaves: source.triangleLeaves, positionLayout: source.positionLayout, nodeFormat: source.nodeFormat });
        kernels.set(key, kernel);
    }
    return kernel;
}

/** The rays through one tree: the median of timed dispatches, then the hits read back. */
async function trace(source: TraceSource, rays: Rays): Promise<{ ms: number | null; timing: Row['timing']; t: Float32Array; triangles: Uint32Array }> {
    const kernel = await kernelFor(source);
    const { buffer, results, read } = rays;
    timer?.reset(); // a trace that failed midway (jobs run one at a time, so none is in flight)
    const times: number[] = [];
    for (let run = 0; run < DISPATCHES + 1; run++) {
        const encoder = device.createCommandEncoder({ label: 'builders trace' });
        kernel.encode(encoder, source.traceBuffers, { mode: 'closest', rays: buffer, results, count: RAY_COUNT, timestampWrites: timer?.pass() });
        timer?.resolveInto(encoder);
        const start = performance.now();
        device.queue.submit([encoder.finish()]);
        if (timer) times.push((await timer.result())[0]);
        else {
            await device.queue.onSubmittedWorkDone();
            times.push(performance.now() - start);
        }
    }
    times.shift(); // the first is a warm-up
    times.sort((a, b) => a - b);
    const encoder = device.createCommandEncoder({ label: 'builders hits' });
    encoder.copyBufferToBuffer(results, 0, read, 0, RAY_COUNT * HIT_RECORD_BYTES);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const mapped = read.getMappedRange();
    const floats = new Float32Array(mapped);
    const words = new Uint32Array(mapped);
    const stride = HIT_RECORD_BYTES / 4;
    const t = new Float32Array(RAY_COUNT);
    const triangles = new Uint32Array(RAY_COUNT);
    for (let i = 0; i < RAY_COUNT; i++) {
        t[i] = floats[i * stride + HIT_RECORD_WORD.t];
        triangles[i] = words[i * stride + HIT_RECORD_WORD.triangle];
    }
    read.unmap();
    return { ms: times[Math.floor(times.length / 2)] || null, timing: timer ? 'timestamps' : 'submit to done', t, triangles };
}

/** Builds one row: the tree, its one-instance scene on the GPU (in the formats the tree came in), its trace. */
async function buildRow(kind: BuilderKind, mesh: IndexedMesh, rays: Rays): Promise<Row> {
    const leaf = settings.leaf === 'default' ? {} : { maxLeafPrimitives: Number(settings.leaf) };
    const built = await builds.build(kind, mesh, leaf);
    const data = assembleLayouts([built.layout], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY_TRANSFORM }], [mesh]);
    const buffers = uploadTraceScene(device, data, kind);
    const source = traceSourceOf(device, buffers);
    const traced = await trace(source, rays);
    const element = document.createElement('tr');
    const row: Row = { kind, built, info: layoutInfo(built.layout), data, buffers, source, traceMs: traced.ms, timing: traced.timing, t: traced.t, triangles: traced.triangles, agree: null, element };
    element.addEventListener('click', () => {
        void queue(async () => {
            // Only a row of the pass on show (a click on a row being replaced does nothing).
            if (!current?.rows.includes(row)) return;
            await view.prepare(row.source);
            select(row);
        });
    });
    return row;
}

/** Every builder on `mesh`, one after another (so their timings don't overlap); then the pass replaces the one on show. */
function buildAll(mesh: IndexedMesh = current!.mesh, name: string = current!.name): Promise<void> {
    return queue(async () => {
        const { rays, box, radius } = makeRays(mesh);
        const pass: Pass = { mesh, name, radius, rays, rows: [] };
        const pending: Row[] = [];
        try {
            for (const kind of BUILDER_KINDS) {
                caption.textContent = `${name}: building with ${kind}…`;
                pending.push(await buildRow(kind, mesh, rays));
            }
        } catch (error) {
            for (const row of pending) row.buffers.destroy();
            for (const buffer of [rays.buffer, rays.results, rays.read]) buffer.destroy();
            throw error;
        }
        pass.rows.push(...pending);
        // Agreement: the same distance as tinybvh's tree (or both missed), ray by ray.
        const tiny = pass.rows.find((row) => row.kind === 'tinybvh')!;
        for (const row of pass.rows) {
            let same = 0;
            for (let i = 0; i < RAY_COUNT; i++) {
                const a = row.t[i];
                const b = tiny.t[i];
                if ((a < 0 && b < 0) || Math.abs(a - b) <= 1e-5 * Math.max(1, b)) same++;
            }
            row.agree = same / RAY_COUNT;
        }
        const show = pass.rows.find((row) => row.kind === (params.get('show') ?? 'gpu-hploc')) ?? pass.rows[0];
        await view.prepare(show.source);

        // The swap, in one go (no await): the next frame traces the new pass.
        const old = current;
        current = pass;
        table.tBodies[0].replaceChildren(...pass.rows.map((row) => row.element));
        if (!old || old.mesh !== mesh) camera.frame(box);
        select(show);
        // Nothing traces the old pass after this: its frames were submitted, and jobs run one at a time.
        if (old) {
            for (const row of old.rows) row.buffers.destroy();
            for (const buffer of [old.rays.buffer, old.rays.results, old.rays.read]) buffer.destroy();
        }
    });
}

function useScene(scene: VanillaScene): Promise<void> {
    const mesh = mergeToWorld(scene);
    overlay.set('mesh', `${scene.name}: ${(mesh.indices.length / 3).toLocaleString('en-US')} triangles${scene.instances.length > 1 ? `, ${scene.instances.length} instances merged` : ''}`);
    return buildAll(mesh, scene.name);
}

overlay.select('largest leaf', ['default', '1', '2', '4', '8'] as const, 'default', (value) => {
    settings.leaf = value;
    void buildAll();
});
overlay.select('show', VIEW_MODES, settings.mode, (value) => (settings.mode = value));
overlay.button('build all again', () => void buildAll());
overlay.note('Each builder makes a tree of the same mesh; each tree is traced with the same rays by the same kernel. GPU rows show the GPU time of the whole build batch. Drop a glTF to compare on it.');

const model = requestedModel();
await useScene(model ? await loadGltfScene(model) : singleMeshScene('torus knot', torusKnot(1, 0.3, 1000, 50)));
enableSceneDrop(useScene, (error) => overlay.error(String(error)));

/**
 * Smoke: three rows; every tree's hits at tinybvh's distance; at least half
 * the rays hit; tinybvh's hits agree with the CPU walk on 1,024 rays; with
 * subgroups, the GPU rows were built by their builders.
 */
function checkPass(pass: Pass): Record<string, unknown> {
    if (pass.rows.length !== BUILDER_KINDS.length) throw new Error(`${pass.rows.length} rows`);
    for (const row of pass.rows) {
        if (row.agree === null || row.agree < 0.999) throw new Error(`${row.kind}: ${row.agree} of the rays hit at tinybvh's distance`);
        let hits = 0;
        for (let i = 0; i < RAY_COUNT; i++) if (row.t[i] >= 0) hits++;
        if (hits < RAY_COUNT / 2) throw new Error(`${row.kind}: only ${hits} of ${RAY_COUNT} rays hit`);
        if (row.kind !== 'tinybvh' && GpuLbvhBuilder.isSupported(device) && row.built.builtBy !== row.kind) throw new Error(`${row.kind}: built by ${row.built.builtBy} on a device with subgroups`);
    }
    const tiny = pass.rows.find((row) => row.kind === 'tinybvh')!;
    const subset = pass.rays.packed.subarray(0, 1024 * RAY_FLOATS);
    const cpu = checkAgainstCpu(subset, (i) => (tiny.t[i] >= 0 ? { t: tiny.t[i], instance: 0, triangle: tiny.triangles[i] } : null), createCpuTracer(tiny.data));
    return {
        cpu,
        rows: pass.rows.map((row) => ({ kind: row.kind, builtBy: row.built.builtBy, fallbackReason: row.built.fallbackReason, cpuMs: row.built.cpuMs, gpuMs: row.built.gpuMs, ...row.info, traceMs: row.traceMs, agree: row.agree })),
    };
}

let frame = 0;
let smoke: Promise<void> | null = null;
app.loop({
    async update() {
        frame++;
        await view.collect();
        if (isSmoke && frame === 20) smoke ??= smokeRun();
    },
    render(encoder, target) {
        if (!selected || !current) return;
        const width = canvas.width;
        const height = canvas.height;
        view.trace(encoder, selected.source, camera.basis(width / height), { mode: settings.mode, shadows: false, sun: [0, 1, 0], highlight: null, sceneRadius: current.radius }, width, height);
        view.draw(encoder, target);
    },
});

async function smokeRun(): Promise<void> {
    const first = checkPass(current!);
    // Rebuilds while frames trace: a leaf change and "build all again" asked at once run one after the other,
    // and the pass on show is never freed under a frame.
    settings.leaf = '2';
    const a = buildAll();
    const b = buildAll();
    await nextFrames(2);
    await Promise.all([a, b]);
    await nextFrames(5);
    const second = checkPass(current!);
    await ready(device, { ...overlay.snapshot(), first, afterRebuild: second });
}

function nextFrames(count: number): Promise<void> {
    const target = frame + count;
    return new Promise((resolve) => {
        const poll = () => (frame >= target ? resolve() : requestAnimationFrame(poll));
        poll();
    });
}

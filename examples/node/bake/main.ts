/**
 * node/bake: a tree baked in Node, loaded here with no build at all.
 *
 * `pnpm examples:bake` built this tree with tinybvh in Node (the WebAssembly
 * needs no GPU), `serializeLayout` wrote it to a file, and this page does
 * `deserializeLayout(await (await fetch(url)).arrayBuffer())`: no builder, no
 * worker, no tinybvh in the page (a smoke run builds the default model fresh
 * once, to compare). The overlay's `builtBy` says the tree is the baked file.
 *
 * The file is `fixtures/torus-knot.bvh`, `?bake=<url>` names another, and a
 * dropped `.bvh` replaces it. The walk traces it in the shared ray-cast
 * kernel, one ray per pixel; the baked triangles are the only geometry the
 * page has (a baked `records` tree carries them), so nothing else is drawn.
 */

import { createCpuTracer, createTinyBvhBuilder, deserializeLayout, GPU_LAYOUT_VERSION, IDENTITY_TRANSFORM, SERIALIZED_FORMAT_VERSION, uploadTraceScene, type GpuLayoutMeshBvh, type TraceSceneBuffers, type TraceSceneData, type TraceSource } from '@pmndrs/bvh';
import { assembleLayouts } from '../../shared/builders.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';
import { bytes as formatBytes } from '../../shared/overlay.ts';
import { layoutInfo } from '../../shared/stats.ts';
import { createVanillaApp } from '../../shared/vanilla/app.ts';
import { checkViewAgainstCpu } from '../../shared/vanilla/check.ts';
import { vec3 } from '../../shared/vanilla/math.ts';
import { RayCastView, VIEW_MODES, type ViewMode } from '../../shared/vanilla/raycast-view.ts';
import fixtureUrl from './fixtures/torus-knot.bvh?url';
import { DEFAULT_LEAF, defaultMesh } from './model.ts';

const app = await createVanillaApp('Node bake', 'node/bake');
const { device, camera, overlay, canvas } = app;

const requested = new URLSearchParams(location.search).get('bake');
const settings = { mode: 'normal' as ViewMode, shadows: true };
const view = RayCastView.create(device, app.format);
let current: { layout: GpuLayoutMeshBvh; data: TraceSceneData; buffers: TraceSceneBuffers; source: TraceSource; radius: number } | null = null;

/** The header's two versions, read before loading: what a version check shows (deserializeLayout refuses a mismatch). */
function header(bytes: ArrayBuffer): { formatVersion: number; layoutVersion: number } {
    const words = new DataView(bytes);
    return { formatVersion: words.getUint32(4, true), layoutVersion: words.getUint32(8, true) };
}

/** Loads a baked tree from `bytes`: no build. Takes the old scene's place between frames. */
async function useBake(bytes: ArrayBuffer, name: string, fetchMs: number): Promise<void> {
    const { formatVersion, layoutVersion } = header(bytes);
    const start = performance.now();
    const layout = deserializeLayout(bytes); // throws on another layout version: rebuild (bake again)
    const loadMs = performance.now() - start;
    if (layout.triangleLeaves === 'indexed') throw new Error(`${name} is an indexed bake: it holds no positions, and this page has no geometry to load it next to. Bake without --indexed.`);

    // Nothing was built: the builder is the file.
    overlay.build({ builtBy: `baked file: ${name}`, fallbackReason: null, cpuMs: 0 });
    overlay.tree(layoutInfo(layout));
    overlay.set('file', `${name}, ${formatBytes(bytes.byteLength)}`);
    overlay.set('fetch + deserialize', `${fetchMs.toFixed(1)} ms + ${loadMs.toFixed(2)} ms (no build)`);
    overlay.set('header', `byte format ${formatVersion}, GPU layout version ${layoutVersion} (this package: ${SERIALIZED_FORMAT_VERSION} and ${GPU_LAYOUT_VERSION})`);
    overlay.set('formats', `${layout.triangleLeaves ?? 'records'} triangles, ${layout.nodeFormat ?? 'f32'} nodes`);

    const data = assembleLayouts([layout], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY_TRANSFORM }], []);
    const buffers = uploadTraceScene(device, data, 'bake');
    const source: TraceSource = buffers; // an uploaded scene is a TraceSource as it is
    await view.prepare(source);
    const old = current;
    const b = layout.bounds;
    current = { layout, data, buffers, source, radius: Math.hypot(b[3] - b[0], b[4] - b[1], b[5] - b[2]) / 2 };
    camera.frame(b);
    old?.buffers.destroy();
}

async function loadUrl(url: string): Promise<void> {
    const start = performance.now();
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    const bytes = await response.arrayBuffer();
    await useBake(bytes, new URL(url, location.href).pathname.split('/').pop() ?? url, performance.now() - start);
}

overlay.select('show', VIEW_MODES, settings.mode, (value) => (settings.mode = value));
overlay.checkbox('shadows (any hit)', settings.shadows, (value) => (settings.shadows = value));
overlay.note('Nothing is built here: the tree is a file baked by `pnpm examples:bake` in Node, read with deserializeLayout and traced by the WGSL walk. Drop another .bvh, or open ?bake=<url>. Drag to orbit.');

let pointer: [number, number] | null = null;
canvas.addEventListener('pointermove', (event) => (pointer = [event.clientX / canvas.clientWidth, event.clientY / canvas.clientHeight]));
canvas.addEventListener('pointerleave', () => (pointer = null));
let highlight: number | null = null;

addEventListener('dragover', (event) => event.preventDefault());
addEventListener('drop', (event) => {
    event.preventDefault();
    const file = [...(event.dataTransfer?.files ?? [])][0];
    if (!file) return;
    const start = performance.now();
    void file
        .arrayBuffer()
        .then((bytes) => useBake(bytes, file.name, performance.now() - start))
        .catch((error: unknown) => overlay.error(String(error)));
});

try {
    await loadUrl(requested ?? fixtureUrl);
} catch (error) {
    overlay.error(String(error));
    throw error;
}

const sun = vec3.normalize([0.45, 0.85, 0.3]);
let frame = 0;
let smoke: Promise<void> | null = null;
app.loop({
    async update() {
        frame++;
        const { measurement, pick } = await view.collect();
        if (measurement && measurement.gpuMs !== null && measurement.gpuMs > 0) {
            overlay.rays('primary (GPU timestamps)', measurement.primaryRays / (measurement.gpuMs / 1000));
            overlay.set('trace pass', `${measurement.gpuMs.toFixed(2)} ms, ${view.size.width}×${view.size.height}, ${measurement.hits.toLocaleString('en-US')} hits`);
        } else if (measurement) overlay.set('trace pass', 'no timestamp-query on this device');
        if (pick !== undefined) {
            highlight = pick ? pick.instance : null;
            overlay.set('pixel', pick ? `triangle ${pick.triangle}, t ${pick.t.toFixed(3)}` : '-');
        }
        if (isSmoke && frame === 20) smoke ??= smokeRun();
    },
    render(encoder, target) {
        if (!current) return;
        const width = canvas.width;
        const height = canvas.height;
        const pick = pointer ? ([Math.floor(pointer[0] * width), Math.floor(pointer[1] * height)] as const) : null;
        view.trace(encoder, current.source, camera.basis(width / height), { mode: settings.mode, shadows: settings.shadows, sun, highlight, sceneRadius: current.radius }, width, height, { measure: frame % 15 === 0, pick });
        view.draw(encoder, target);
    },
});

/**
 * Smoke: the view's 1,024 pixels against the CPU walk over the loaded tree
 * (at least 50 of them hits); then, when the page showed the default file, the
 * default model built fresh with tinybvh in the page: its layout bytes are the
 * baked file's bytes. Anything else fails the run.
 */
async function smokeRun(): Promise<void> {
    const loaded = current!;
    const agreement = await checkViewAgainstCpu(view, (aspect) => camera.basis(aspect), createCpuTracer(loaded.data));
    // checkViewAgainstCpu throws on any difference it can't explain.
    if (agreement.hits < 50) throw new Error(`the baked tree's view hits too little: ${JSON.stringify(agreement)}`);

    let fresh: Record<string, unknown> | null = null;
    if (!requested) {
        const tiny = await createTinyBvhBuilder();
        const built = tiny.buildLayout(defaultMesh(), { maxLeafPrimitives: DEFAULT_LEAF }).layout;
        const same = (a: ArrayBuffer, b: ArrayBuffer) => a.byteLength === b.byteLength && new Uint8Array(a).every((byte, index) => byte === new Uint8Array(b)[index]);
        fresh = {
            nodes: same(built.nodes, loaded.layout.nodes),
            triangles: same(built.triangles, loaded.layout.triangles),
            parents: built.parents.every((parent, index) => parent === loaded.layout.parents[index]),
            nodeCount: built.nodeCount === loaded.layout.nodeCount,
        };
        if (!Object.values(fresh).every(Boolean)) throw new Error(`the baked tree differs from a fresh build of the same model: ${JSON.stringify(fresh)}`);
        overlay.set('check', 'identical to a fresh tinybvh build; the view agrees with the CPU walk');
    } else overlay.set('check', 'the view agrees with the CPU walk');
    await ready(device, { ...overlay.snapshot(), agreement, fresh, builtByBake: overlay.snapshot().build });
}

// For the smoke run's screenshot: the pointer in the middle until the mouse moves.
if (isSmoke) pointer = [0.5, 0.5];

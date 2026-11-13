/**
 * vanilla/raytrace: a compute-shader ray caster over the package's WGSL walk,
 * in raw WebGPU.
 *
 * - **The walk in your own kernel:** `traceModuleWgsl` is included in the
 *   kernel's text and bound at group 0 (shared/vanilla/raycast-view.ts). One
 *   thread per pixel calls `bvhTraceClosest`, and, with shadows on, one
 *   `bvhTraceAny` towards a fixed light.
 * - **Shown from the hit alone:** its normal, material number, instance,
 *   triangle or distance. No sampling, BSDFs or accumulation: the package
 *   ends at the hit (HANDOFF.md, the scope rule).
 * - **Builders, live:** every mesh's tree from tinybvh, GPU H-PLOC or GPU
 *   LBVH (`createMeshLayoutBuilder({ prefer })`), assembled with its
 *   instances (`assembleTraceScene`, `uploadTraceScene`); build ms, SAH and
 *   Mrays/s change with it.
 * - **Picking with no extra ray:** each pixel's hit record stays in a buffer;
 *   the one under the pointer is read back and its instance tinted.
 *
 * A showcase scene by default, a `?model=` glTF or a dropped one.
 */

import { createCpuTracer, createMeshLayoutBuilder, RAY_FLOATS, uploadTraceScene, type MeshLayoutBuilder, type MeshLayoutBuilderKind, type TraceSceneBuffers, type TraceSceneData } from '@pmndrs/bvh';
import { requestedModel } from '../../shared/assets.ts';
import { assembleLayouts, traceSourceOf, type TraceSource } from '../../shared/gpu-rays.ts';
import { bytes } from '../../shared/overlay.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';
import { layoutInfo, sumInfo } from '../../shared/stats.ts';
import { createVanillaApp } from '../../shared/vanilla/app.ts';
import { checkAgainstCpu } from '../../shared/vanilla/check.ts';
import { showcaseScene, type VanillaScene } from '../../shared/vanilla/geometry.ts';
import { enableSceneDrop, loadGltfScene } from '../../shared/vanilla/gltf.ts';
import { vec3 } from '../../shared/vanilla/math.ts';
import { RayCastView, VIEW_MODES, type ViewMode } from '../../shared/vanilla/raycast-view.ts';

const app = await createVanillaApp('Ray-cast view', 'vanilla/raytrace');
const { device, camera, overlay, canvas } = app;

const settings = {
    builder: (await createMeshLayoutBuilder(device)).kind as MeshLayoutBuilderKind,
    mode: 'normal' as ViewMode,
    shadows: true,
    scale: 1,
};
const view = RayCastView.create(device, app.format);
let scene: VanillaScene;
let current: { data: TraceSceneData; buffers: TraceSceneBuffers; source: TraceSource; radius: number } | null = null;

// One chooser per builder, made once: making one prepares its GPU pipelines (not part of a build's time).
const choosers = new Map<MeshLayoutBuilderKind, Promise<MeshLayoutBuilder>>();
const chooserFor = (kind: MeshLayoutBuilderKind) => {
    if (!choosers.has(kind)) choosers.set(kind, createMeshLayoutBuilder(device, { prefer: kind }));
    return choosers.get(kind)!;
};

// Rebuilds (the builder select, a drop, the smoke run) run one after another.
let rebuilds: Promise<void> = Promise.resolve();
function rebuild(): Promise<void> {
    const next = rebuilds.then(rebuildNow);
    rebuilds = next.catch(() => undefined);
    return next;
}

/** Builds every mesh with `settings.builder`, assembles the instances, uploads, and compiles the view for the scene's formats. */
async function rebuildNow(): Promise<void> {
    const target = scene; // a drop while this builds starts its own rebuild after this one
    const builder = await chooserFor(settings.builder);
    const start = performance.now();
    const results = await builder.buildLayouts(target.meshes);
    const cpuMs = performance.now() - start;
    const by = new Map<string, number>();
    for (const result of results) by.set(result.builtBy, (by.get(result.builtBy) ?? 0) + 1);
    // tinybvh asked for by name is no fallback; a GPU builder the device can't run is.
    const reasons = new Set([settings.builder === 'tinybvh' ? null : builder.fallbackReason, ...results.map((result) => result.fallbackReason)].filter((reason): reason is string => !!reason));
    overlay.build({ builtBy: [...by].map(([kind, count]) => `${kind} ×${count}`).join(', '), fallbackReason: reasons.size ? [...reasons].join('; ') : null, cpuMs });
    overlay.tree(sumInfo(results.map((result) => layoutInfo(result.layout))));

    // The meshes' trees placed by the instances, in the formats the trees came in (the builders' defaults).
    const data = assembleLayouts(
        results.map((result) => result.layout),
        target.instances.map((instance, index) => ({ meshIndex: instance.mesh, instanceIndex: index, objectToWorld: instance.objectToWorld, material: instance.material })),
        target.meshes,
    );
    const buffers = uploadTraceScene(device, data, 'raytrace');
    const source = traceSourceOf(device, buffers);
    await view.prepare(source);
    // No await from here: the next frame traces the new scene with the kernel just made for it.
    const old = current;
    const b = target.bounds;
    current = { data, buffers, source, radius: Math.hypot(b[3] - b[0], b[4] - b[1], b[5] - b[2]) / 2 };
    old?.buffers.destroy();
    overlay.set('scene on the GPU', `${bytes(buffers.byteLength)} (${source.triangleLeaves}, ${source.nodeFormat} nodes)`);
}

async function useScene(next: VanillaScene): Promise<void> {
    scene = next;
    overlay.set('scene', `${next.name}: ${next.meshes.length.toLocaleString('en-US')} meshes, ${next.instances.length.toLocaleString('en-US')} instances, ${next.triangles.toLocaleString('en-US')} triangles`);
    if (next.loadMs !== undefined) overlay.set('load', `${next.loadMs.toFixed(0)} ms`);
    camera.frame(next.bounds);
    if (next.name === 'showcase') camera.distance *= 0.65;
    await rebuild();
}

overlay.select('builder', ['tinybvh', 'gpu-hploc', 'gpu-lbvh'] as const, settings.builder, (value) => {
    settings.builder = value;
    void rebuild();
});
overlay.select('show', VIEW_MODES, settings.mode, (value) => (settings.mode = value));
overlay.checkbox('shadows (any hit)', settings.shadows, (value) => (settings.shadows = value));
overlay.select('resolution', ['1', '0.5', '0.25'] as const, '1', (value) => (settings.scale = Number(value)));
overlay.note('One closest-hit ray per pixel through the WGSL walk in this page\'s own kernel; shadows add an any-hit ray per hit. Hover: the pixel\'s hit record, read back. Drag to orbit, drop a glTF.');

let pointer: [number, number] | null = null;
canvas.addEventListener('pointermove', (event) => (pointer = [event.clientX / canvas.clientWidth, event.clientY / canvas.clientHeight]));
canvas.addEventListener('pointerleave', () => (pointer = null));
let highlight: number | null = null;

const model = requestedModel();
await useScene(model ? await loadGltfScene(model) : showcaseScene());
enableSceneDrop(useScene, (error) => overlay.error(String(error)));

const sun = vec3.normalize([0.45, 0.85, 0.3]);
let frame = 0;
let smoke: Promise<void> | null = null;
app.loop({
    async update() {
        frame++;
        const { measurement, pick } = await view.collect();
        if (measurement && measurement.gpuMs !== null && measurement.gpuMs > 0) {
            overlay.rays('primary (GPU timestamps)', measurement.primaryRays / (measurement.gpuMs / 1000));
            overlay.rays('primary + shadow', (measurement.primaryRays + measurement.shadowRays) / (measurement.gpuMs / 1000));
            overlay.set('trace pass', `${measurement.gpuMs.toFixed(2)} ms, ${view.size.width}×${view.size.height}, ${measurement.hits.toLocaleString('en-US')} hits`);
        } else if (measurement) overlay.set('trace pass', 'no timestamp-query on this device');
        if (pick !== undefined) {
            highlight = pick ? pick.instance : null;
            overlay.set('pixel', pick ? `instance ${pick.instance}, triangle ${pick.triangle}, material ${pick.material}, t ${pick.t.toFixed(3)}` : '-');
        }
        if (isSmoke && frame === 20) smoke ??= smokeRun();
    },
    render(encoder, target) {
        if (!current) return;
        const width = Math.max(1, Math.round(canvas.width * settings.scale));
        const height = Math.max(1, Math.round(canvas.height * settings.scale));
        const pick = pointer ? ([Math.floor(pointer[0] * width), Math.floor(pointer[1] * height)] as const) : null;
        view.trace(encoder, current.source, camera.basis(width / height), { mode: settings.mode, shadows: settings.shadows, sun, highlight, sceneRadius: current.radius }, width, height, { measure: frame % 15 === 0, pick });
        view.draw(encoder, target);
    },
});

/**
 * Smoke: with each builder, 1,024 pixels of the last frame against the CPU
 * port of the walk over the same assembled scene (the same camera rays).
 */
async function smokeRun(): Promise<void> {
    const rows: unknown[] = [];
    // Every builder, the one the page started with last (so the screenshot matches the overlay's select).
    const first = settings.builder;
    const order = [...(['tinybvh', 'gpu-hploc', 'gpu-lbvh'] as const).filter((kind) => kind !== first), first];
    for (const builder of order) {
        settings.builder = builder;
        await rebuild();
        await nextFrames(3);
        const pixels = await view.readPixels();
        const basis = camera.basis(pixels.width / pixels.height);
        const columns = 32;
        const rowsOfRays = 32;
        const packed = new Float32Array(columns * rowsOfRays * RAY_FLOATS);
        const at: [number, number][] = [];
        for (let j = 0; j < rowsOfRays; j++) {
            for (let i = 0; i < columns; i++) {
                const x = Math.floor(((i + 0.5) / columns) * pixels.width);
                const y = Math.floor(((j + 0.5) / rowsOfRays) * pixels.height);
                const ndc = [((x + 0.5) / pixels.width) * 2 - 1, 1 - ((y + 0.5) / pixels.height) * 2];
                const direction = vec3.normalize(vec3.add(basis.forward, vec3.add(vec3.scale(basis.right, ndc[0]), vec3.scale(basis.up, ndc[1]))));
                packed.set([...basis.origin, 0, ...direction, 1e30], at.length * RAY_FLOATS);
                at.push([x, y]);
            }
        }
        const agreement = checkAgainstCpu(packed, (index) => pixels.hit(at[index][0], at[index][1]), createCpuTracer(current!.data), { perturb: 1e-6 }); // the kernel makes its own camera rays: ulp-level grazes allowed
        rows.push({ builder, built: overlay.snapshot().build, tree: overlay.snapshot().tree, agreement });
    }
    overlay.set('check', 'every builder\'s view agrees with the CPU walk');
    await nextFrames(20);
    await ready(device, { ...overlay.snapshot(), builders: rows });
}

function nextFrames(count: number): Promise<void> {
    const target = frame + count;
    return new Promise((resolve) => {
        const poll = () => (frame >= target ? resolve() : requestAnimationFrame(poll));
        poll();
    });
}

// For the smoke run's screenshot: the pointer in the middle until the mouse moves.
if (isSmoke) pointer = [0.5, 0.5];

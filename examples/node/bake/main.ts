/**
 * node/bake: a tree or a whole scene baked in Node, loaded here with no build
 * at all.
 *
 * `pnpm examples:bake` built the file with tinybvh in Node (the WebAssembly
 * needs no GPU) and wrote it with `serializeLayout` (one mesh's tree) or, with
 * `--scene`, `serializeTraceScene` (every tree, the top level, instances and
 * materials). This page reads the header first (`readSerializedHeader`: the
 * kind, the versions, whether this package loads it), then
 * `deserializeLayout` or `deserializeTraceScene`: no builder, no worker, no
 * tinybvh in the page (a smoke run bakes the defaults fresh once, to compare).
 * The overlay's `builtBy` says the trees are the baked file.
 *
 * - A tree with `records` triangles carries its triangles: it is traced alone.
 * - A tree with `indexed` triangles holds no positions: it is loaded next to
 *   the geometry it was baked from, the default knot or the glTF named by
 *   `?model=` (merged to world space as the CLI does). `deserializeLayout(bytes,
 *   { geometry })` refuses other geometry (the file carries its hash).
 * - A scene file uploads as it is (an indexed one carries its positions).
 * - A movable scene file (`--scene --movable`) goes into `SceneTrees`
 *   (`deserializeMovableScene`, `placeMovableScene`): every tree, the merged
 *   groups' too, is placed with no build, the top level is built here, and
 *   the props (instances of a mesh used more than once) move.
 *
 * The file is `fixtures/torus-knot.bvh`, `?bake=<url>` names another (try
 * `fixtures/knots.scene.bvh`), and a dropped `.bvh` replaces it. The walk
 * traces it in the shared ray-cast kernel, one ray per pixel.
 */

import {
    assembleMovableScene,
    createCpuTracer,
    createTinyBvhBuilder,
    deserializeLayout,
    deserializeMovableScene,
    deserializeTraceScene,
    GPU_LAYOUT_VERSION,
    IDENTITY_TRANSFORM,
    INSTANCE_RECORD_BYTES,
    NODE_WORD,
    placeMovableScene,
    readSerializedHeader,
    SceneTrees,
    SERIALIZED_FORMAT_VERSION,
    SERIALIZED_MOVABLE_SCENE_FORMAT_VERSION,
    SERIALIZED_SCENE_FORMAT_VERSION,
    serializeLayout,
    TOP_LEVEL_HEADER_BYTES,
    uploadTraceScene,
    type GpuLayoutMeshBvh,
    type MeshGeometry,
    type MovableSceneData,
    type PlacedMovableScene,
    type SerializedHeader,
    type TraceSceneBuffers,
    type TraceSceneData,
    type TraceSceneInstance,
    type TraceSource,
} from '@pmndrs/bvh';
import { assembleLayouts } from '../../shared/builders.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';
import { bytes as formatBytes } from '../../shared/overlay.ts';
import { layoutInfo } from '../../shared/stats.ts';
import { createVanillaApp } from '../../shared/vanilla/app.ts';
import { checkViewAgainstCpu, type Agreement } from '../../shared/vanilla/check.ts';
import { mergeToWorld } from '../../shared/vanilla/geometry.ts';
import { loadGltfScene } from '../../shared/vanilla/gltf.ts';
import { vec3 } from '../../shared/vanilla/math.ts';
import { RayCastView, VIEW_MODES, type ViewMode } from '../../shared/vanilla/raycast-view.ts';
import fixtureUrl from './fixtures/torus-knot.bvh?url';
import sceneFixtureUrl from './fixtures/knots.scene.bvh?url';
import { bakeMovableScene, bakeScene, DEFAULT_LEAF, defaultMesh, defaultScene } from './model.ts';

const app = await createVanillaApp('Node bake', 'node/bake');
const { device, camera, overlay, canvas } = app;

const params = new URLSearchParams(location.search);
const requested = params.get('bake');
const modelPath = params.get('model');
const settings = { mode: 'normal' as ViewMode, shadows: true, move: true };
const view = RayCastView.create(device, app.format);

/** A movable scene in SceneTrees: the scene's own instances, and where the props are now. */
interface Movable {
    readonly trees: SceneTrees;
    readonly placed: PlacedMovableScene;
    readonly scene: MovableSceneData;
    instances: TraceSceneInstance[];
}
let current: {
    kind: SerializedHeader['kind'];
    layout: GpuLayoutMeshBvh | null;
    /** The CPU walk's copy of what the view traces (a movable scene's, with its props where they are now). */
    data: () => TraceSceneData;
    buffers: TraceSceneBuffers | null;
    movable: Movable | null;
    source: TraceSource;
    radius: number;
} | null = null;

/** Moves a movable scene's props (its instances that aren't static) up and down, by `time` seconds into the bob. */
function moveProps(movable: Movable, time: number): void {
    movable.instances = movable.scene.instances.map((instance, index) => {
        if (instance.static) return instance;
        const objectToWorld = Array.from(instance.objectToWorld);
        objectToWorld[13] += 0.6 * Math.sin(time * 1.7 + index);
        movable.trees.setTransform(movable.placed.instances[index], objectToWorld);
        return { ...instance, objectToWorld };
    });
}

/** The geometry an indexed tree is loaded next to: the glTF of `?model=` merged to world space (as the CLI bakes it), else the default knot. */
let appGeometry: Promise<MeshGeometry> | null = null;
function geometryForIndexed(): Promise<MeshGeometry> {
    appGeometry ??= modelPath ? loadGltfScene(modelPath).then(mergeToWorld) : Promise.resolve(defaultMesh());
    return appGeometry;
}

/** The world box of a scene: its top level's root (the union of the root's two child boxes; an empty child is an inverted box). */
function sceneBounds(data: TraceSceneData): number[] {
    const floats = new Float32Array(data.topLevel.buffer, TOP_LEVEL_HEADER_BYTES + data.topLevel.instanceCount * INSTANCE_RECORD_BYTES, 16);
    const out = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    for (const [min, max] of [
        [NODE_WORD.leftMin, NODE_WORD.leftMax],
        [NODE_WORD.rightMin, NODE_WORD.rightMax],
    ]) {
        if ([0, 1, 2].some((axis) => !(floats[min + axis] <= floats[max + axis]))) continue;
        for (let axis = 0; axis < 3; axis++) {
            out[axis] = Math.min(out[axis], floats[min + axis]);
            out[axis + 3] = Math.max(out[axis + 3], floats[max + axis]);
        }
    }
    return out;
}

/** Loads a baked tree or scene from `bytes`: no build. Takes the old scene's place between frames. */
async function useBake(bytes: ArrayBuffer, name: string, fetchMs: number): Promise<void> {
    // The header first: what the file is and whether this package reads it (a stale bake says why).
    const header = readSerializedHeader(bytes);
    const own = header.kind === 'tree' ? SERIALIZED_FORMAT_VERSION : header.kind === 'scene' ? SERIALIZED_SCENE_FORMAT_VERSION : SERIALIZED_MOVABLE_SCENE_FORMAT_VERSION;
    overlay.set('header', `${header.kind}, byte format ${header.formatVersion}, GPU layout version ${header.layoutVersion} (this package: ${own} and ${GPU_LAYOUT_VERSION})`);
    if (!header.current) throw new Error(`${name} is a stale bake: ${header.reason}`);

    let layout: GpuLayoutMeshBvh | null = null;
    let data: TraceSceneData;
    let movable: Movable | null = null;
    let bounds: ArrayLike<number>;
    let loadMs: number;
    let geometryNote = '';
    if (header.kind === 'movable-scene') {
        if (!header.geometry) throw new Error(`${name} was written without its meshes' geometry, and this page has no model to load it next to: bake it with the CLI (--scene --movable), which stores it`);
        const start = performance.now();
        const scene = deserializeMovableScene(bytes); // mesh trees, merged groups' trees, instances, and the geometry SceneTrees takes
        // The scene's formats; an indexed movable scene's trees name the meshes' own vertices, so no weld.
        const trees = await SceneTrees.create(device, { label: 'bake', triangleLeaves: scene.triangleLeaves, nodeFormat: scene.nodeFormat, ...(scene.triangleLeaves === 'indexed' ? { weld: false } : {}) });
        const placed = placeMovableScene(trees, scene); // every tree placed: the first update builds nothing but the top level
        loadMs = performance.now() - start;
        movable = { trees, placed, scene, instances: [...scene.instances] };
        data = assembleMovableScene(scene);
        bounds = sceneBounds(data);
        const moving = scene.instances.filter((instance) => !instance.static).length;
        const groups = `${header.mergedGroupCount} merged static group${header.mergedGroupCount === 1 ? '' : 's'}`;
        overlay.set('scene', `${header.meshCount} meshes, ${header.instanceCount} instances (${moving} move, the rest in ${groups}); the top level is built here`);
        overlay.tree({ sah: null, depth: data.maxMeshDepth, nodes: null, triangles: null, bytes: data.byteLength });
    } else if (header.kind === 'scene') {
        const start = performance.now();
        data = deserializeTraceScene(bytes); // the whole scene: trees, top level, instances, materials
        loadMs = performance.now() - start;
        bounds = sceneBounds(data);
        overlay.set('scene', `${header.meshCount} meshes, ${header.instanceCount} instance records (${header.mergedGroupCount} merged static groups)`);
        overlay.tree({ sah: null, depth: data.maxMeshDepth, nodes: null, triangles: null, bytes: data.byteLength });
    } else {
        // An indexed tree reads the positions it was baked from: hand them over, and deserializeLayout checks them.
        const geometry = header.triangleLeaves === 'indexed' ? await geometryForIndexed() : null;
        const start = performance.now();
        layout = deserializeLayout(bytes, geometry ? { geometry } : {});
        loadMs = performance.now() - start;
        data = assembleLayouts([layout], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY_TRANSFORM }], geometry ? [geometry] : []);
        bounds = layout.bounds;
        if (geometry) geometryNote = header.geometryHash ? `, next to ${modelPath ?? 'the default knot'} (hash checked)` : `, next to ${modelPath ?? 'the default knot'} (no hash in the file)`;
        overlay.set('scene', 'one tree, one instance');
        overlay.tree(layoutInfo(layout));
    }

    // Nothing was built: the builder is the file.
    overlay.build({ builtBy: `baked ${header.kind}: ${name}`, fallbackReason: null, cpuMs: 0 });
    overlay.set('file', `${name}, ${formatBytes(bytes.byteLength)}`);
    overlay.set('fetch + deserialize', `${fetchMs.toFixed(1)} ms + ${loadMs.toFixed(2)} ms (${movable ? 'with SceneTrees.create and placeMovableScene; ' : ''}no build)`);
    overlay.set('formats', `${header.triangleLeaves} triangles${geometryNote}, ${header.nodeFormat} nodes`);

    // A movable scene traces from SceneTrees (updated each frame); the others upload as they are.
    const buffers = movable ? null : uploadTraceScene(device, data, 'bake');
    const source: TraceSource = movable ? movable.trees : buffers!; // an uploaded scene and a SceneTrees are both TraceSources
    await view.prepare(source);
    const old = current;
    const b = bounds;
    const cpu = movable ? () => assembleMovableScene({ ...movable.scene, instances: movable.instances }) : () => data;
    current = { kind: header.kind, layout, data: cpu, buffers, movable, source, radius: Math.hypot(b[3] - b[0], b[4] - b[1], b[5] - b[2]) / 2 };
    camera.frame(Array.from(b));
    old?.buffers?.destroy();
    old?.movable?.trees.dispose();
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
overlay.checkbox('move the props (a movable scene)', settings.move, (value) => (settings.move = value));
overlay.note(
    'Nothing is built here: the file was baked by `pnpm examples:bake` in Node (a tree, a scene with --scene, a movable scene with --scene --movable), checked with readSerializedHeader, read with deserializeLayout, deserializeTraceScene or deserializeMovableScene (into SceneTrees, whose props move) and traced by the WGSL walk. Drop another .bvh, or open ?bake=<url> (an indexed tree loads next to the default knot, or ?model=<glTF>). Drag to orbit.',
);

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
/** Resolves once `count` more frames were traced (the smoke run reads the view after switching files). */
const frameWaiters: { at: number; resolve: () => void }[] = [];
const afterFrames = (count: number) => new Promise<void>((resolve) => frameWaiters.push({ at: frame + count, resolve }));
app.loop({
    async update() {
        frame++;
        for (const waiter of frameWaiters.filter((entry) => entry.at <= frame)) {
            frameWaiters.splice(frameWaiters.indexOf(waiter), 1);
            waiter.resolve();
        }
        const { measurement, pick } = await view.collect();
        if (measurement && measurement.gpuMs !== null && measurement.gpuMs > 0) {
            overlay.rays('primary (GPU timestamps)', measurement.primaryRays / (measurement.gpuMs / 1000));
            overlay.set('trace pass', `${measurement.gpuMs.toFixed(2)} ms, ${view.size.width}×${view.size.height}, ${measurement.hits.toLocaleString('en-US')} hits`);
        } else if (measurement) overlay.set('trace pass', 'no timestamp-query on this device');
        if (pick !== undefined) {
            highlight = pick ? pick.instance : null;
            overlay.set('pixel', pick ? `instance ${pick.instance}, triangle ${pick.triangle}, t ${pick.t.toFixed(3)}` : '-');
        }
        if (isSmoke && frame === 20) smoke ??= smokeRun();
    },
    render(encoder, target) {
        if (!current) return;
        if (current.movable) {
            if (settings.move) moveProps(current.movable, performance.now() / 1000);
            current.movable.trees.update(encoder); // trees placed at the first update, the top level brought up to date every frame
        }
        const width = canvas.width;
        const height = canvas.height;
        const pick = pointer ? ([Math.floor(pointer[0] * width), Math.floor(pointer[1] * height)] as const) : null;
        view.trace(encoder, current.source, camera.basis(width / height), { mode: settings.mode, shadows: settings.shadows, sun, highlight, sceneRadius: current.radius }, width, height, { measure: frame % 15 === 0, pick });
        view.draw(encoder, target);
    },
});

/** The view's last frame against the CPU walk over the loaded scene (at least 50 hits); throws on any difference it can't explain. */
async function checkView(what: string): Promise<Agreement> {
    const agreement = await checkViewAgainstCpu(view, (aspect) => camera.basis(aspect), createCpuTracer(current!.data()));
    if (agreement.hits < 50) throw new Error(`${what}: the view hits too little: ${JSON.stringify(agreement)}`);
    return agreement;
}

const sameBytes = (a: ArrayBuffer | Uint8Array, b: ArrayBuffer | Uint8Array) => {
    const x = a instanceof Uint8Array ? a : new Uint8Array(a);
    const y = b instanceof Uint8Array ? b : new Uint8Array(b);
    return x.byteLength === y.byteLength && x.every((byte, index) => byte === y[index]);
};

/**
 * Smoke: the view against the CPU walk over each kind of file, in turn.
 *
 * 1. The page's file. When it is the default, the default model built fresh
 *    with tinybvh in the page gives its exact layout bytes.
 * 2. With no `?bake=`: an indexed tree of the default knot, baked in the
 *    page with its geometry and loaded next to it.
 * 3. With no `?bake=`: the committed scene file (`--scene`), whose bytes are
 *    those of the default scene baked fresh in the page.
 * 4. With no `?bake=`: the default scene as a movable scene (`--scene
 *    --movable`), baked in the page and loaded into SceneTrees: every tree is
 *    the file's (none built), and the view agrees with the CPU walk with the
 *    props where the file put them, then moved.
 *
 * Anything else fails the run.
 */
async function smokeRun(): Promise<void> {
    const tree = await checkView('the page\'s file');
    if (requested) {
        overlay.set('check', `the view agrees with the CPU walk (${current!.kind})`);
        await ready(device, { ...overlay.snapshot(), agreement: { [current!.kind]: tree } });
        return;
    }

    const loaded = current!.layout!;
    const tiny = await createTinyBvhBuilder();
    const built = tiny.buildLayout(defaultMesh(), { maxLeafPrimitives: DEFAULT_LEAF }).layout;
    const fresh = {
        nodes: sameBytes(built.nodes, loaded.nodes),
        triangles: sameBytes(built.triangles, loaded.triangles),
        parents: built.parents.every((parent, index) => parent === loaded.parents[index]),
        nodeCount: built.nodeCount === loaded.nodeCount,
    };
    if (!Object.values(fresh).every(Boolean)) throw new Error(`the baked tree differs from a fresh build of the same model: ${JSON.stringify(fresh)}`);

    // An indexed bake of the default knot, loaded next to the knot's positions.
    const knot = defaultMesh();
    const indexedTree = tiny.buildLayout(knot, { maxLeafPrimitives: DEFAULT_LEAF, triangleLeaves: 'indexed' }).layout;
    await useBake(serializeLayout(indexedTree, { geometry: knot }), 'torus-knot (indexed, baked in the page)', 0);
    await afterFrames(3);
    const indexed = await checkView('the indexed tree');
    if (current!.data().triangleLeaves !== 'indexed') throw new Error('the indexed bake was not traced as an indexed scene');

    // The committed scene file, and the same scene baked in the page.
    await loadUrl(sceneFixtureUrl);
    await afterFrames(3);
    const scene = await checkView('the scene file');
    const committed = await (await fetch(sceneFixtureUrl)).arrayBuffer();
    const sceneFresh = sameBytes((await bakeScene(defaultScene(), {}, tiny)).bytes, committed);
    if (!sceneFresh) throw new Error('the committed scene file differs from the default scene baked fresh in the page: run pnpm examples:bake --scene');

    // A movable scene: into SceneTrees, the props still, then moved.
    settings.move = false;
    const movableBake = await bakeMovableScene(defaultScene(), {}, tiny);
    await useBake(movableBake.bytes.buffer as ArrayBuffer, 'knots (movable, baked in the page)', 0);
    const { trees, scene: movableScene } = current!.movable!;
    await trees.whenBuilt();
    await afterFrames(3);
    const movableStill = await checkView('the movable scene');
    moveProps(current!.movable!, 1.3);
    await afterFrames(3);
    const movableMoved = await checkView('the movable scene, props moved');
    const counts = trees.counts;
    const ownTrees = movableScene.meshes.filter((mesh) => mesh.tree).length + movableScene.mergedGroups.length;
    if (counts.onAppTrees !== ownTrees || counts.onGpuTrees !== 0 || counts.onTinybvhTrees !== 0 || counts.mergedGroups !== movableScene.mergedGroups.length) {
        throw new Error(`the movable scene built trees of its own: ${JSON.stringify(counts)} (expected ${ownTrees} trees from the file)`);
    }

    overlay.set('check', 'tree, indexed tree, scene and movable scene (still and moved) agree with the CPU walk; the files are identical to fresh bakes; the movable scene built no tree');
    await ready(device, {
        ...overlay.snapshot(),
        agreement: { tree, indexed, scene, movableStill, movableMoved },
        fresh,
        sceneFresh,
        movableTrees: { fromFile: counts.onAppTrees, gpu: counts.onGpuTrees, tinybvh: counts.onTinybvhTrees, mergedGroups: counts.mergedGroups },
        builtByBake: overlay.snapshot().build,
    });
}

// For the smoke run's screenshot: the pointer in the middle until the mouse moves.
if (isSmoke) pointer = [0.5, 0.5];

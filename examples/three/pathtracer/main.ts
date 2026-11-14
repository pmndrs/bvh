/**
 * three/pathtracer: a small TSL compute path tracer over `SceneTrees`,
 * through `BvhTraceNodes` (the WGSL walk in three's node system).
 *
 * One compute node (kernel.ts, which compare/pathtracer runs over three-mesh-bvh's
 * walk too), one invocation per pixel, one sample per frame:
 * - a camera ray, then up to `bounces` diffuse bounces (a `Loop` with a
 *   uniform bound, so the walk is in the shader once);
 * - at each hit, `bvh.traceAny` toward the sun for a hard shadow;
 * - `hit.material` (from the sync's `material` callback) indexes a palette
 *   of the meshes' colours;
 * - samples accumulate in a storage buffer until the camera moves.
 *
 * The shader text depends on nothing per scene (the palette has a fixed
 * capacity), so it compiles once and is the same on every load. The pipeline
 * is compiled with `compileComputeAsync` before the first dispatch.
 * Mrays/s counts the rays each invocation actually traced (one atomic add per
 * pixel) over the wall time of one dispatch with the queue drained before and
 * after (an upper bound on its GPU time).
 */

import { Color, Group, Mesh, MeshStandardMaterial, PlaneGeometry, BoxGeometry, SphereGeometry, TorusKnotGeometry, type Material, type Object3D } from 'three';
import { storage, texture, uv } from 'three/tsl';
import { MeshBasicNodeMaterial, QuadMesh, StorageBufferAttribute, StorageTexture, type ComputeNode } from 'three/webgpu';
import { SceneTrees } from '@pmndrs/bvh';
import { BvhTraceNodes, SceneTreesObject3DSync, type BvhTslNode } from '@pmndrs/bvh/three';
import { createApp } from '../../shared/app.ts';
import { enableDrop, frameObject, loadGltf, requestedModel } from '../../shared/loader.ts';
import { SceneTreesLog } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';
import { pathTracerKernel, pathTracerTargets, pathTracerUniforms, type PathTracerTargets, type PathTracerWalk } from './kernel.ts';

const app = await createApp('Path tracer', 'three/pathtracer');
const { device, renderer, camera, controls, overlay } = app;

const PALETTE_CAPACITY = 256;
const palette = new Float32Array(PALETTE_CAPACITY * 4).fill(0.8);
const paletteIndex = new Map<Material, number>();
function materialOf(mesh: Mesh, info: { materialIndex: number }): number {
    const material = (Array.isArray(mesh.material) ? mesh.material[Math.max(0, info.materialIndex)] : mesh.material) as Material & { color?: Color };
    let index = paletteIndex.get(material);
    if (index === undefined) {
        index = Math.min(paletteIndex.size, PALETTE_CAPACITY - 1);
        paletteIndex.set(material, index);
        const color = material.color ?? new Color(0.8, 0.8, 0.8);
        palette.set([color.r, color.g, color.b, 1], index * 4);
    }
    return index;
}

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
    add(new Mesh(new TorusKnotGeometry(1, 0.32, 300, 32), lambert(0x8fb3ff)), 0, 1.9, 0);
    add(new Mesh(new SphereGeometry(0.9, 64, 32), lambert(0xf0e0a0)), -3, 0.9, 1.5);
    add(new Mesh(new BoxGeometry(1.4, 2.4, 1.4), lambert(0xb0e0a8)), 3, 1.2, 1).rotation.y = 0.5;
    return root;
}

let world: Object3D;
const model = requestedModel();
if (model) world = (await loadGltf(model)).root;
else world = defaultScene();
app.scene.add(world);
frameObject(world, camera, controls);
if (!model) {
    camera.position.set(0, 3.5, 9);
    controls.target.set(0, 1.4, 0);
    controls.update();
}
enableDrop(() => overlay.error('The path tracer takes ?model= on load; reload with it to trace a glTF.'), (error) => overlay.error(String(error)));

// The trees, kept in step with the scene, and the walk in three's node system.
const log = new SceneTreesLog();
const trees = await SceneTrees.create(device, { label: 'pathtracer', onTree: log.onTree });
const sync = new SceneTreesObject3DSync(trees, world, { material: materialOf, static: true });
const bvh = new BvhTraceNodes(renderer, trees);
function updateTrees(): void {
    world.updateMatrixWorld();
    sync.update();
    const encoder = device.createCommandEncoder({ label: 'pathtracer update' });
    trees.update(encoder);
    device.queue.submit([encoder.finish()]);
    bvh.update(); // rebinds the storage nodes if SceneTrees replaced a buffer
}
updateTrees();
await trees.whenBuilt();
updateTrees();

// The kernel (kernel.ts, shared with compare/pathtracer): the sampling over our walk.
const u = pathTracerUniforms();
const { cameraWorld, projectionInverse, cameraPosition, frameIndex, bounces, width, height } = u;
const paletteNode = storage(new StorageBufferAttribute(palette, 4), 'vec4', PALETTE_CAPACITY).toReadOnly();
const walk: PathTracerWalk = {
    closest(origin, direction, tMin, tMax) {
        const hit = bvh.traceClosest(origin, direction, tMin, tMax).toVar() as BvhTslNode;
        return { t: hit.get('t'), normal: hit.get('normal'), albedo: paletteNode.element(hit.get('material')).xyz };
    },
    any: (origin, direction, tMin, tMax) => bvh.traceAny(origin, direction, tMin, tMax) as BvhTslNode,
};

// Size-dependent resources, made again on resize.
let targets: PathTracerTargets;
let kernel: ComputeNode;
const quad = new QuadMesh(new MeshBasicNodeMaterial());

function build(): void {
    const scale = isSmoke ? 0.5 : Math.min(1, 1280 / innerWidth);
    const w = Math.max(16, Math.floor(innerWidth * scale));
    const h = Math.max(16, Math.floor(innerHeight * scale));
    width.value = w;
    height.value = h;
    // The ray counter carries over a resize (the Mrays/s readout takes differences).
    const counter = targets?.rayCounter;
    targets = pathTracerTargets(w, h, new StorageTexture(w, h));
    if (counter) targets = { ...targets, rayCounter: counter };
    kernel = pathTracerKernel(walk, u, targets, w, h);
    const material = quad.material as MeshBasicNodeMaterial;
    material.colorNode = texture(targets.target, uv());
    material.needsUpdate = true;
}

build();
const compileStart = performance.now();
await renderer.compileComputeAsync(kernel!);
overlay.set('compile', `${(performance.now() - compileStart).toFixed(0)} ms`);
let resetNeeded = true;
addEventListener('resize', () => {
    build();
    resetNeeded = true;
});
controls.addEventListener('change', () => (resetNeeded = true));
overlay.range('bounces', 0, 4, 1, bounces.value, (value) => {
    bounces.value = value;
    resetNeeded = true;
});
overlay.note('Orbit to restart the accumulation. ?model= traces a glTF (base colours only, no textures).');
overlay.build(log.build(trees));
overlay.tree(log.tree(trees, sync.meshHandles));
overlay.set('materials', paletteIndex.size);

let frame = 0;
let samples = 0;
let gpuMs = 0;
let measuredFrames = 0;
let lastRays = 0;
let raysPerSecond = 0;
let computeMs = 0;
app.loop(async () => {
    frame++;
    if (resetNeeded) {
        (targets.accum.array as Float32Array).fill(0);
        targets.accum.needsUpdate = true;
        samples = 0;
        resetNeeded = false;
    }
    camera.updateMatrixWorld();
    cameraWorld.value.copy(camera.matrixWorld);
    projectionInverse.value.copy(camera.projectionMatrixInverse);
    cameraPosition.value.setFromMatrixPosition(camera.matrixWorld);
    frameIndex.value = frame;
    // Every 20th frame is timed alone: the queue drained, the dispatch, the queue drained again.
    // (three's compute timestamps came out too low to trust here; this wall time is an upper bound.)
    const timed = frame % 20 === 0;
    if (timed) await device.queue.onSubmittedWorkDone();
    const start = performance.now();
    renderer.compute(kernel);
    if (timed) {
        await device.queue.onSubmittedWorkDone();
        computeMs = performance.now() - start;
    }
    samples++;
    quad.render(renderer);
    if (timed) {
        gpuMs += computeMs;
        measuredFrames++;
        const count = new Uint32Array(await renderer.getArrayBufferAsync(targets.rayCounter))[0];
        const rays = (count - lastRays) >>> 0;
        lastRays = count;
        raysPerSecond = (rays / 20 / computeMs) * 1000;
        overlay.rays('path tracer (wall, one frame)', raysPerSecond);
        overlay.set('rays per frame', Math.round(rays / 20));
        overlay.set('compute', `${computeMs.toFixed(2)} ms`);
        overlay.set('samples', samples);
        overlay.set('resolution', `${width.value} × ${height.value}`);
    }
    if (isSmoke && frame >= 60 && lastRays > 0) await ready(device, { ...overlay.snapshot(), raysTraced: lastRays, gpuMs, measuredFrames });
    return false; // the quad is the frame
});

/**
 * babylon/gpu-trace: hard shadows from the package's WGSL walk, on the device
 * of Babylon's own `WebGPUEngine`.
 *
 * 1. `deviceFromEngine(engine)` is `engine._device` (checked); the scene's
 *    meshes and instances go into a `SceneTrees` on that device
 *    (`bvhSceneFromBabylon`).
 * 2. A Babylon `ComputeShader` is the walk plus this page's kernel
 *    (`createTraceComputeShader`): per pixel, a camera ray from the active
 *    camera's matrices, `bvhTraceClosest` for the surface, `bvhTraceAny`
 *    toward the sun for the shadow. `SceneTrees`' own `GPUBuffer`s are bound
 *    with no copy (`bindTraceBuffers`).
 * 3. The kernel writes a shadow mask into a storage texture, which a Babylon
 *    `Layer` blends over Babylon's own render of the scene (its own lighting,
 *    no shadow generator): the walk gives the shadows, Babylon the shading.
 *
 * No AO and no sampling: the package ends at the hit (a demo of the walk).
 * The sun turns and a sphere bobs, so the top-level refit and the shadows
 * both move every frame.
 */

import { Constants } from '@babylonjs/core/Engines/constants.js';
import { StorageBuffer } from '@babylonjs/core/Buffers/storageBuffer.js';
import { HemisphericLight } from '@babylonjs/core/Lights/hemisphericLight.js';
import { DirectionalLight } from '@babylonjs/core/Lights/directionalLight.js';
import { Layer } from '@babylonjs/core/Layers/layer.js';
import '@babylonjs/core/Layers/layerSceneComponent.js';
import { StandardMaterial } from '@babylonjs/core/Materials/standardMaterial.js';
import { RawTexture } from '@babylonjs/core/Materials/Textures/rawTexture.js';
import { Color3 } from '@babylonjs/core/Maths/math.color.js';
import { Matrix, Vector3 } from '@babylonjs/core/Maths/math.vector.js';
import '@babylonjs/core/Meshes/instancedMesh.js'; // ball.createInstance needs it
import { CreateBox } from '@babylonjs/core/Meshes/Builders/boxBuilder.js';
import { CreateGround } from '@babylonjs/core/Meshes/Builders/groundBuilder.js';
import { CreateSphere } from '@babylonjs/core/Meshes/Builders/sphereBuilder.js';
import { CreateTorusKnot } from '@babylonjs/core/Meshes/Builders/torusKnotBuilder.js';
import type { Mesh } from '@babylonjs/core/Meshes/mesh.js';
import type { WebGPUEngine } from '@babylonjs/core/Engines/webgpuEngine.js';
import { SceneTrees } from '@pmndrs/bvh';
import { BabylonBvhPicker, bindTraceBuffers, bvhSceneFromBabylon, createTraceComputeShader, deviceFromEngine, type BvhBabylonScene } from '@pmndrs/bvh/babylon';
import { fail, isSmoke, ready } from '../../shared/smoke.ts';
import { createBabylonApp, runBabylon } from '../../shared/babylon/app.ts';
import { SceneTreesLog } from '../../shared/stats.ts';

const app = await createBabylonApp('GPU trace (Babylon)', 'babylon/gpu-trace', { requireWebGpu: true });
const { engine, scene, camera, overlay } = app;
const device = deviceFromEngine(engine);

// The walk costs a ray per pixel: `?scale=2` traces at half the canvas size. The smoke run (software WebGPU) traces a few thousand pixels.
const scale = Number(new URLSearchParams(location.search).get('scale') ?? (isSmoke ? 8 : 1));
if (scale > 1) {
    engine.setHardwareScalingLevel(scale);
    engine.resize();
}

// ---- A Babylon scene: a ground, a box, a knot, spheres (three of them instances of one), lit by Babylon's own lights.

function material(color: Color3): StandardMaterial {
    const m = new StandardMaterial('material', scene);
    m.diffuseColor = color;
    m.specularColor = new Color3(0.15, 0.15, 0.15);
    return m;
}

const ground = CreateGround('ground', { width: 24, height: 24, subdivisions: 4 }, scene);
ground.material = material(new Color3(0.62, 0.64, 0.68));
const box = CreateBox('box', { size: 1.6 }, scene);
box.position.set(-3, 0.8, -1);
box.rotation.y = 0.5;
box.material = material(new Color3(0.9, 0.55, 0.3));
const knot = CreateTorusKnot('knot', { radius: 1, tube: 0.3, radialSegments: 160, tubularSegments: 24 }, scene);
knot.position.set(1.5, 1.6, 0.5);
knot.material = material(new Color3(0.45, 0.65, 0.95));
const ball = CreateSphere('ball', { diameter: 1.4, segments: 32 }, scene);
ball.position.set(-0.5, 0.7, 3);
ball.material = material(new Color3(0.5, 0.85, 0.55));
const bobbing: Mesh = CreateSphere('bobbing', { diameter: 1, segments: 24 }, scene);
bobbing.material = ball.material;
[4, -5].forEach((x, i) => {
    const twin = ball.createInstance(`ball-${i}`);
    twin.position.set(x, 0.7, i === 0 ? 3.5 : -3);
    twin.scaling.setAll(i === 0 ? 1.2 : 0.8);
    twin.scaling.y *= i === 0 ? 1 : 1.4;
});
const bobbingHome = new Vector3(3, 2, -3);
bobbing.position.copyFrom(bobbingHome);

const sky = new HemisphericLight('sky', new Vector3(0.2, 1, 0.1), scene);
sky.intensity = 0.55;
const sun = new DirectionalLight('sun', new Vector3(-0.5, -1, -0.4), scene);
sun.intensity = 0.8;

camera.radius = 17;
camera.beta = 1.05;
camera.target = new Vector3(0, 0.8, 0);

// ---- The scene on the engine's device.

const read: BvhBabylonScene = bvhSceneFromBabylon(scene);
const log = new SceneTreesLog();
const trees = await SceneTrees.create(device, { label: 'babylon gpu-trace', onTree: log.onTree });
const meshHandles = read.meshes.map((geometry) => trees.addMesh(geometry));
const instanceHandles = read.instances.map((instance) =>
    trees.addInstance({ mesh: meshHandles[instance.meshIndex], objectToWorld: instance.objectToWorld, instanceIndex: instance.instanceIndex, mask: instance.mask, visible: instance.visible }),
);
const bobbingInstance = instanceHandles[read.sources.findIndex((source) => source.object === bobbing)];

// ---- The kernel: the walk, then one thread per pixel.

const KERNEL = /* wgsl */ `
struct Params {
    invViewProjection: mat4x4f,
    eye: vec4f,      // xyz: the camera
    light: vec4f,    // xyz: unit vector toward the sun, w: the shadow's alpha
    size: vec4f,     // x, y: the image; z: ray offset along the normal
}
@group(0) @binding(3) var<storage, read> params: Params;
@group(0) @binding(4) var shadowMask: texture_storage_2d<rgba8unorm, write>;
// Per pixel, two vec4f: (t, instance, triangle, shadowed) and the shadow ray's origin. The page reads them back in smoke mode only.
@group(0) @binding(5) var<storage, read_write> hits: array<vec4f>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let width = u32(params.size.x);
    let height = u32(params.size.y);
    if (id.x >= width || id.y >= height) { return; }
    let uv = (vec2f(id.xy) + vec2f(0.5)) / params.size.xy;
    let far = params.invViewProjection * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    let direction = normalize(far.xyz / far.w - params.eye.xyz);
    let hit = bvhTraceClosest(BvhRay(params.eye.xyz, 0.0, direction, 1e30));
    var alpha = 0.0;
    var shadowed = 0.0;
    var origin = vec3f(0.0);
    if (hit.t >= 0.0) {
        // The normal of the side the ray hit, whichever way the mesh was wound.
        let facing = select(-hit.normal, hit.normal, hit.frontFace);
        origin = params.eye.xyz + direction * hit.t + facing * params.size.z * max(1.0, hit.t);
        if (bvhTraceAny(BvhRay(origin, 0.0, params.light.xyz, 1e30))) {
            alpha = params.light.w;
            shadowed = 1.0;
        }
    }
    textureStore(shadowMask, id.xy, vec4f(0.02, 0.03, 0.08, alpha));
    let slot = (id.y * width + id.x) * 2u;
    hits[slot] = vec4f(hit.t, f32(hit.instance), f32(hit.triangle), shadowed);
    hits[slot + 1u] = vec4f(origin, 0.0);
}`;

const trace = { nodeFormat: trees.nodeFormat, triangleLeaves: trees.triangleLeaves, positionLayout: trees.positionLayout, rayMasks: false } as const;
const kernel = createTraceComputeShader('bvh shadows', engine, KERNEL, {
    trace,
    bindings: { params: { group: 0, binding: 3 }, shadowMask: { group: 0, binding: 4 }, hits: { group: 0, binding: 5 } },
});

// A compile error reaches the console only; make it the page's failure.
kernel.onError = (_effect, errors) => fail(new Error(`the compute shader did not compile: ${errors}`));

const PARAMS_BYTES =64 + 16 * 3;
const paramsBuffer = new StorageBuffer(engine as WebGPUEngine, PARAMS_BYTES, Constants.BUFFER_CREATIONFLAG_READWRITE, 'bvh params');
const params = new Float32Array(PARAMS_BYTES / 4);
kernel.setStorageBuffer('params', paramsBuffer);

// The mask image and the hits buffer follow the canvas size.
let width = 0;
let height = 0;
let mask: RawTexture | null = null;
let hitsBuffer: StorageBuffer | null = null;
// A foreground layer drawn with the camera's post-processing step: a Layer without applyPostProcess made the WebGPUEngine drop the scene's own meshes (Babylon 9.35).
const layer = new Layer('shadows', null, scene, false);
layer.applyPostProcess = true;

function resizeTargets(): void {
    const w = engine.getRenderWidth();
    const h = engine.getRenderHeight();
    if (w === width && h === height && mask) return;
    width = w;
    height = h;
    const next = new RawTexture(null, w, h, Constants.TEXTUREFORMAT_RGBA, scene, false, false, Constants.TEXTURE_NEAREST_SAMPLINGMODE, Constants.TEXTURETYPE_UNSIGNED_BYTE, Constants.TEXTURE_CREATIONFLAG_STORAGE);
    const nextHits = new StorageBuffer(engine as WebGPUEngine, w * h * 32, Constants.BUFFER_CREATIONFLAG_READWRITE, 'bvh hits');
    kernel.setStorageTexture('shadowMask', next);
    kernel.setStorageBuffer('hits', nextHits);
    layer.texture = next;
    mask?.dispose();
    hitsBuffer?.dispose();
    mask = next;
    hitsBuffer = nextHits;
}

// ---- Each frame: move things, update the trees, bind what changed, dispatch.

// The smoke run holds the scene still (the check compares one frame) unless `?animate=1`, which animates through ten frames first: the bobbing sphere's setTransform and the top-level refit.
const animateInSmoke = new URLSearchParams(location.search).get('animate') === '1';
const settings = { shadow: 0.55, animate: !isSmoke || animateInSmoke, angle: 0.6, bias: 2e-3 };
let boundVersion = -1;
let frames = 0;
let lightDirection = new Vector3();

function towardSun(angle: number): Vector3 {
    return new Vector3(Math.cos(angle) * 0.7, 0.85, Math.sin(angle) * 0.7).normalize();
}

function frame(): void {
    frames++;
    const time = performance.now() / 1000;
    if (settings.animate) settings.angle = 0.6 + time * 0.25;
    lightDirection = towardSun(settings.angle);
    sun.direction = lightDirection.scale(-1);
    // The bobbing sphere: Babylon's mesh and our instance move together.
    bobbing.position.y = bobbingHome.y + (settings.animate ? Math.sin(time * 1.3) * 1.2 : 0);
    trees.setTransform(bobbingInstance, bobbing.computeWorldMatrix(true).m);

    // SceneTrees' builds, swaps and refits are recorded in our own encoder and submitted before Babylon's frame uses the buffers.
    const encoder = device.createCommandEncoder({ label: 'bvh update' });
    trees.update(encoder);
    device.queue.submit([encoder.finish()]);
    if (trees.version !== boundVersion) {
        bindTraceBuffers(kernel, trees.buffers); // after a buffer grew, rebind
        boundVersion = trees.version;
    }
    resizeTargets();

    const view = camera.getViewMatrix(true);
    const projection = camera.getProjectionMatrix(true);
    // Babylon's matrices are row-vector, and their memory is the column-major layout of the same transform: they go in as they are.
    const inverse = view.multiply(projection).invert();
    params.set(inverse.m, 0);
    params.set([camera.globalPosition.x, camera.globalPosition.y, camera.globalPosition.z, 0], 16);
    params.set([lightDirection.x, lightDirection.y, lightDirection.z, settings.shadow], 20);
    params.set([width, height, settings.bias, 0], 24);
    paramsBuffer.update(params);
    kernel.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
}

overlay.range('shadow', 0, 1, 0.05, settings.shadow, (value) => (settings.shadow = value));
overlay.checkbox('animate', settings.animate, (value) => (settings.animate = value));
overlay.note('Shadows come from the package\'s WGSL walk (bvhTraceAny toward the sun), on the engine\'s own device; Babylon draws the scene and the shading.');
overlay.set('instances', `${read.instances.length} (${read.meshes.length} meshes)`);
overlay.set('device', `${device.adapterInfo.vendor || device.adapterInfo.architecture || 'adapter'} ${device.adapterInfo.description || ''}`.trim());

runBabylon(app, () => {
    frame();
    overlay.build(log.build(trees));
    overlay.tree(log.tree(trees));
    const counts = trees.counts;
    overlay.set('trees', `${counts.onGpuTrees} gpu, ${counts.onTinybvhTrees} tinybvh, ${counts.waitingForFirstTree} waiting`);
});

// ---- Smoke: the GPU's pixels against the CPU walk and Babylon's own scene.pick.

let stage = 'waiting for trees';
if (isSmoke) {
    setTimeout(() => fail(new Error(`smoke run stuck: ${stage} (${frames} frames)`)), 60_000);
    void (async () => {
        // Trees first: builds start in update(), and arrive over the next frames.
        for (let i = 0; i < 1200 && (trees.counts.waitingForFirstTree > 0 || frames < 6); i++) await new Promise((resolve) => setTimeout(resolve, 25));
        if (trees.counts.waitingForFirstTree > 0) throw new Error(`${trees.counts.waitingForFirstTree} meshes still have no tree`);
        // Two more frames with the trees placed and the version bound, then freeze the scene (no animation: the check compares one frame).
        if (animateInSmoke) {
            const animated = frames + 10;
            while (frames < animated) await new Promise((resolve) => setTimeout(resolve, 25));
            settings.animate = false;
        }
        const settled = frames + 3;
        while (frames < settled) await new Promise((resolve) => setTimeout(resolve, 25));
        const result = await check();
        await ready(device, { ...result, ...overlay.snapshot() });
    })().catch(fail);
}

async function check(): Promise<Record<string, unknown>> {
    const picker = await BabylonBvhPicker.create(scene);
    stage = 'reading the hits';
    const buffer = new Float32Array((await hitsBuffer!.read(0, undefined, undefined, true)).buffer.slice(0) as ArrayBuffer);
    stage = 'checking';
    const cpu = picker.cpuScene!;
    const step = Math.max(2, Math.floor(Math.min(width, height) / 36));
    let compared = 0;
    let hitDisagree = 0;
    let distanceDisagree = 0;
    let gpuHits = 0;
    let shadowed = 0;
    let shadowChecked = 0;
    let shadowDisagree = 0;
    let babylonCompared = 0;
    let babylonDisagree = 0;
    for (let y = 3; y < height; y += step) {
        for (let x = 3; x < width; x += step) {
            const slot = (y * width + x) * 8;
            const gpuT = buffer[slot];
            const gpuInstance = buffer[slot + 1];
            // createPickingRay takes canvas pixels (scene.pointerX), which hardware scaling shrinks into render pixels.
            const ray = scene.createPickingRay((x + 0.5) * engine.getHardwareScalingLevel(), (y + 0.5) * engine.getHardwareScalingLevel(), Matrix.Identity(), camera);
            const direction = ray.direction.normalizeToNew();
            const [hit] = cpu.raycast({ origin: [ray.origin.x, ray.origin.y, ray.origin.z], direction: [direction.x, direction.y, direction.z], tMin: 0, tMax: 1e30 }, { firstHitOnly: true });
            compared++;
            if ((gpuT >= 0) !== (hit !== undefined)) {
                hitDisagree++;
                continue;
            }
            if (!hit) continue;
            gpuHits++;
            // Babylon's picking ray starts on the near plane, the kernel's at the camera: the same instance, and the same point.
            const eye = camera.globalPosition;
            const point = [ray.origin.x + direction.x * hit.t, ray.origin.y + direction.y * hit.t, ray.origin.z + direction.z * hit.t];
            const gpuPoint = [eye.x + direction.x * gpuT, eye.y + direction.y * gpuT, eye.z + direction.z * gpuT];
            if (hit.instance !== gpuInstance || Math.hypot(point[0] - gpuPoint[0], point[1] - gpuPoint[1], point[2] - gpuPoint[2]) > 2e-3 * Math.max(1, gpuT)) distanceDisagree++;
            // The shadow ray from the origin the GPU used, against the CPU walk.
            const origin: [number, number, number] = [buffer[slot + 4], buffer[slot + 5], buffer[slot + 6]];
            const blocked = cpu.raycast({ origin, direction: [lightDirection.x, lightDirection.y, lightDirection.z], tMin: 0, tMax: 1e30 }, { firstHitOnly: true }).length > 0;
            shadowChecked++;
            if (blocked) shadowed++;
            if (blocked !== (buffer[slot + 3] === 1)) shadowDisagree++;
            // Babylon's own picking, on every fourth hit (it tests every triangle of the scene).
            if (gpuHits % 4 === 0) {
                const babylon = scene.pickWithRay(ray);
                babylonCompared++;
                if (!babylon?.hit || babylon.pickedMesh !== read.sources[gpuInstance]?.object) babylonDisagree++;
            }
        }
    }
    const failures: string[] = [];
    if (gpuHits < 50) failures.push(`only ${gpuHits} sampled pixels hit anything`);
    // Rays through a silhouette or a shared edge may round differently on the two sides.
    if (hitDisagree + distanceDisagree > Math.max(2, compared * 0.01)) failures.push(`${hitDisagree} hit/miss and ${distanceDisagree} instance/distance disagreements over ${compared} pixels`);
    if (shadowDisagree > Math.max(2, shadowChecked * 0.01)) failures.push(`${shadowDisagree} of ${shadowChecked} shadow rays disagree with the CPU walk`);
    if (shadowed < 5 || shadowed === shadowChecked) failures.push(`${shadowed} of ${shadowChecked} sampled hits are shadowed: the shadows are missing or everything is`);
    if (babylonDisagree > Math.max(2, babylonCompared * 0.05)) failures.push(`${babylonDisagree} of ${babylonCompared} hits name another mesh than scene.pick`);
    // The layer's texture holds the mask: some pixels carry the shadow's alpha.
    stage = 'reading the mask';
    const pixels = (await mask!.readPixels()) as Uint8Array | null;
    let maskedPixels = 0;
    if (pixels) for (let i = 3; i < pixels.length; i += 4) if (pixels[i] > 0) maskedPixels++;
    if (!pixels || maskedPixels === 0) failures.push('the shadow mask texture is empty');
    const report = { compared, gpuHits, shadowChecked, shadowed, hitDisagree, distanceDisagree, shadowDisagree, babylonCompared, babylonDisagree, maskedPixels, width, height, frames };
    if (failures.length) throw new Error(`${failures.join('; ')} (${JSON.stringify(report)})`);
    return report;
}

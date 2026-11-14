/**
 * The three.js frame every example starts from: a `WebGPURenderer` (three
 * requests every feature the adapter has, `subgroups` and `timestamp-query`
 * included), its device for `@pmndrs/bvh`, a camera with orbit controls, a
 * light rig, the overlay, and a frame loop that feeds the overlay's graph.
 */

import { AmbientLight, Color, DirectionalLight, PerspectiveCamera, Raycaster, Scene, type Camera, type Vector2 } from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { WebGPURenderer } from 'three/webgpu';
import { RAY_FLOATS } from '@pmndrs/bvh';
import { deviceFromRendererAsync } from '@pmndrs/bvh/three';
import { Overlay } from './overlay.ts';
import { fail, watchDevice } from './smoke.ts';

export interface ExampleApp {
    readonly renderer: WebGPURenderer;
    readonly device: GPUDevice;
    readonly scene: Scene;
    readonly camera: PerspectiveCamera;
    readonly controls: OrbitControls;
    readonly overlay: Overlay;
    /** Starts the frame loop: `frame` runs before each render; return false to skip three's render. */
    loop(frame: (now: number, delta: number) => boolean | void | Promise<boolean | void>): void;
}

export async function createApp(title: string, folder: string, options: { readonly trackTimestamp?: boolean; readonly stencil?: boolean } = {}): Promise<ExampleApp> {
    document.title = `${title} · @pmndrs/bvh examples`;
    const overlay = new Overlay(title, folder);
    if (!navigator.gpu) {
        const message = 'This browser has no WebGPU (navigator.gpu). The examples need Chrome, Edge or Safari 26 with WebGPU.';
        overlay.error(message);
        throw new Error(message);
    }
    // Big scenes need storage bindings past WebGPU's default 128 MiB: ask for what the adapter allows.
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    const requiredLimits = adapter
        ? { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize }
        : undefined;
    const renderer = new WebGPURenderer({ antialias: true, requiredLimits, powerPreference: 'high-performance', trackTimestamp: options.trackTimestamp ?? false, stencil: options.stencil ?? false });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(innerWidth, innerHeight);
    document.body.append(renderer.domElement);
    const device = await deviceFromRendererAsync(renderer);
    watchDevice(device);
    if (!device.features.has('subgroups')) overlay.error('No "subgroups" on this device: the GPU builders fall back to tinybvh (see builtBy).');

    const scene = new Scene();
    scene.background = new Color(0x15181d);
    scene.add(new AmbientLight(0xffffff, 0.6));
    const sun = new DirectionalLight(0xffffff, 2.2);
    sun.position.set(3, 5, 2);
    scene.add(sun);
    const camera = new PerspectiveCamera(50, innerWidth / innerHeight, 0.05, 500);
    camera.position.set(6, 4, 8);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    addEventListener('resize', () => {
        camera.aspect = innerWidth / innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(innerWidth, innerHeight);
    });

    let last = performance.now();
    let running = false;
    return {
        renderer,
        device,
        scene,
        camera,
        controls,
        overlay,
        loop(frame) {
            if (running) throw new Error('loop() already started');
            running = true;
            let busy = false;
            renderer.setAnimationLoop(async (now: number) => {
                overlay.frame(now);
                // A frame that awaits (a readback) holds the next one off instead of piling up.
                if (busy) return;
                busy = true;
                try {
                    const delta = Math.min(0.1, (now - last) / 1000);
                    last = now;
                    controls.update();
                    const draw = await frame(now, delta);
                    if (draw !== false) renderer.render(scene, camera);
                } catch (error) {
                    renderer.setAnimationLoop(null);
                    fail(error);
                } finally {
                    busy = false;
                }
            });
        },
    };
}

const raycaster = new Raycaster();

/** The camera ray through `ndc` (-1..1), packed for `GpuRays.traceClosest` (@pmndrs/bvh). */
export function cameraRay(camera: Camera, ndc: Vector2, out = new Float32Array(RAY_FLOATS), at = 0): Float32Array {
    raycaster.setFromCamera(ndc, camera);
    const { origin, direction } = raycaster.ray;
    out.set([origin.x, origin.y, origin.z, 0, direction.x, direction.y, direction.z, 1e30], at * RAY_FLOATS);
    return out;
}

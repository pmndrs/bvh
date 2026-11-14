/**
 * The frame the Babylon examples start from: an engine (WebGPU when the page
 * can have it, Babylon's WebGL engine otherwise, or as `?engine=webgl` /
 * `?engine=webgpu` asks), a scene with an orbit camera, the shared overlay,
 * and Babylon's own render loop feeding the overlay's graph.
 *
 * `engine=webgpu` fails when there is no adapter; `engine=webgl` is how the
 * picking page shows that CPU picking needs no WebGPU. Without `?engine=` the
 * page takes WebGPU if an adapter exists.
 *
 * Only the Babylon modules the examples use are imported (deep imports), so
 * the build tree-shakes Babylon.
 */

import { Engine } from '@babylonjs/core/Engines/engine.js';
import { WebGPUEngine } from '@babylonjs/core/Engines/webgpuEngine.js';
import type { AbstractEngine } from '@babylonjs/core/Engines/abstractEngine.js';
import { ArcRotateCamera } from '@babylonjs/core/Cameras/arcRotateCamera.js';
import { Color4 } from '@babylonjs/core/Maths/math.color.js';
import { Vector3 } from '@babylonjs/core/Maths/math.vector.js';
import { Scene } from '@babylonjs/core/scene.js';
import { deviceFromEngine } from '@pmndrs/bvh/babylon';
import { Overlay } from '../overlay.ts';
import { fail, watchDevice } from '../smoke.ts';

export type BabylonEngineKind = 'webgpu' | 'webgl';

export interface BabylonApp {
    readonly engine: AbstractEngine;
    readonly kind: BabylonEngineKind;
    readonly scene: Scene;
    readonly camera: ArcRotateCamera;
    readonly canvas: HTMLCanvasElement;
    readonly overlay: Overlay;
    /** The engine's `GPUDevice` (`engine._device`, through `deviceFromEngine`); null on the WebGL engine. */
    readonly device: GPUDevice | null;
}

export interface BabylonAppOptions {
    /** The page needs WebGPU: without it the page says so and stops. */
    readonly requireWebGpu?: boolean;
}

export async function createBabylonApp(title: string, folder: string, options: BabylonAppOptions = {}): Promise<BabylonApp> {
    document.title = `${title} · @pmndrs/bvh examples`;
    const overlay = new Overlay(title, folder);
    const canvas = document.createElement('canvas');
    canvas.style.cssText = 'display:block;width:100vw;height:100vh;touch-action:none;outline:none';
    document.body.append(canvas);

    const asked = new URLSearchParams(location.search).get('engine');
    if (asked !== null && asked !== 'webgl' && asked !== 'webgpu') throw new Error(`?engine= is webgl or webgpu, got "${asked}"`);
    const wantWebGpu = asked === 'webgpu' || (asked === null && (await WebGPUEngine.IsSupportedAsync));
    if (options.requireWebGpu && !wantWebGpu) {
        const message = 'This page traces on the engine\'s WebGPU device: it needs a browser with WebGPU (Chrome, Edge or Safari 26) and no ?engine=webgl.';
        overlay.error(message);
        throw new Error(message);
    }

    let engine: AbstractEngine;
    let kind: BabylonEngineKind;
    if (wantWebGpu) {
        // Babylon asks for WebGPU's default limits and no optional features: a scene past 128 MiB of triangles needs the adapter's maximum,
        // and the GPU builders want `subgroups`, so both are requested (three does the same for features).
        engine = await WebGPUEngine.CreateAsync(canvas, { antialias: true, enableAllFeatures: true, setMaximumLimits: true });
        kind = 'webgpu';
    } else {
        engine = new Engine(canvas, true, { preserveDrawingBuffer: true });
        kind = 'webgl';
    }
    const device = kind === 'webgpu' ? deviceFromEngine(engine) : null;
    if (device) watchDevice(device);

    const scene = new Scene(engine);
    scene.clearColor = new Color4(0.08, 0.09, 0.11, 1);
    const camera = new ArcRotateCamera('camera', -Math.PI / 2.4, Math.PI / 2.6, 9, Vector3.Zero(), scene);
    camera.attachControl(canvas, true);
    camera.minZ = 0.05;
    camera.wheelPrecision = 30;
    camera.lowerRadiusLimit = 1.5;
    scene.activeCamera = camera;
    overlay.set('engine', kind === 'webgpu' ? 'WebGPUEngine' : `Engine (WebGL: ${(engine as Engine).getGlInfo().renderer})`);
    addEventListener('resize', () => engine.resize());
    scene.onBeforeRenderObservable.add(() => overlay.frame());
    engine.onContextLostObservable.add(() => fail('the rendering context was lost'));
    return { engine, kind, scene, camera, canvas, overlay, device };
}

/** Starts Babylon's render loop; an error in a frame fails the smoke run (and is shown). */
export function runBabylon(app: BabylonApp, beforeRender: () => void): void {
    app.scene.onBeforeRenderObservable.add(() => {
        try {
            beforeRender();
        } catch (error) {
            app.engine.stopRenderLoop();
            fail(error);
        }
    });
    app.engine.runRenderLoop(() => app.scene.render());
}

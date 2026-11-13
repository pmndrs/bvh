/**
 * The frame every vanilla example starts from, with no engine: a `GPUDevice`
 * asked for `subgroups` (the GPU builders and refits need it; tinybvh builds
 * without it) and `timestamp-query` when the adapter has them, the adapter's
 * storage limits (big scenes), a canvas configured for WebGPU, an orbit
 * camera, the shared overlay, and a frame loop that feeds the overlay's graph.
 *
 * The three pages have the same thing on three's renderer (../app.ts).
 */

import { Overlay } from '../overlay.ts';
import { fail, watchDevice } from '../smoke.ts';
import { OrbitCamera } from './camera.ts';

export interface VanillaApp {
    readonly device: GPUDevice;
    readonly canvas: HTMLCanvasElement;
    readonly context: GPUCanvasContext;
    /** The canvas texture format. */
    readonly format: GPUTextureFormat;
    readonly camera: OrbitCamera;
    readonly overlay: Overlay;
    /** Canvas width over height. */
    readonly aspect: number;
    /** Bumped when the canvas is resized. */
    readonly sizeVersion: number;
    /**
     * Starts the frame loop. Each frame, `update` runs first and may await (a
     * readback; the next frame waits for it), then `render` records into the
     * frame's encoder with the canvas texture as `target`, and the loop
     * submits. `render` doesn't await: the canvas texture is only good for the
     * task it was taken in.
     */
    loop(frame: FrameCallbacks): void;
}

export interface FrameCallbacks {
    update?(now: number, delta: number): void | Promise<void>;
    render(encoder: GPUCommandEncoder, target: GPUTextureView): void;
}

export async function createVanillaApp(title: string, folder: string): Promise<VanillaApp> {
    document.title = `${title} · @pmndrs/bvh examples`;
    const overlay = new Overlay(title, folder);
    const adapter = navigator.gpu ? await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }) : null;
    if (!adapter) {
        const message = 'This browser has no WebGPU adapter. The examples need Chrome, Edge or Safari 26 with WebGPU.';
        overlay.error(message);
        throw new Error(message);
    }
    const wanted: GPUFeatureName[] = ['subgroups', 'timestamp-query'];
    const device = await adapter.requestDevice({
        label: title,
        requiredFeatures: wanted.filter((feature) => adapter.features.has(feature)),
        // Big scenes need storage bindings past WebGPU's default 128 MiB.
        requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
    });
    watchDevice(device);
    if (!device.features.has('subgroups')) overlay.error('No "subgroups" on this device: the GPU builders fall back to tinybvh (see builtBy).');

    const canvas = document.createElement('canvas');
    canvas.style.cssText = 'display:block;width:100vw;height:100vh;touch-action:none';
    document.body.append(canvas);
    const context = canvas.getContext('webgpu')!;
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });
    const camera = new OrbitCamera(canvas);

    let sizeVersion = 0;
    const resize = () => {
        const ratio = Math.min(devicePixelRatio, 2);
        const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
        const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
        if (canvas.width === width && canvas.height === height) return;
        canvas.width = width;
        canvas.height = height;
        sizeVersion++;
    };
    resize();
    addEventListener('resize', resize);

    let running = false;
    return {
        device,
        canvas,
        context,
        format,
        camera,
        overlay,
        get aspect() {
            return canvas.width / canvas.height;
        },
        get sizeVersion() {
            return sizeVersion;
        },
        loop(frame) {
            if (running) throw new Error('loop() already started');
            running = true;
            let busy = false;
            let stopped = false;
            let last = performance.now();
            const tick = async (now: number) => {
                if (stopped) return;
                requestAnimationFrame(tick);
                overlay.frame(now);
                // A frame that awaits (a readback) holds the next one off instead of piling up.
                if (busy) return;
                busy = true;
                try {
                    const delta = Math.min(0.1, (now - last) / 1000);
                    last = now;
                    resize();
                    await frame.update?.(now, delta);
                    const encoder = device.createCommandEncoder({ label: `${title} frame` });
                    frame.render(encoder, context.getCurrentTexture().createView());
                    device.queue.submit([encoder.finish()]);
                } catch (error) {
                    stopped = true;
                    fail(error);
                } finally {
                    busy = false;
                }
            };
            requestAnimationFrame(tick);
        },
    };
}

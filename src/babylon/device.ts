/**
 * The `GPUDevice` of a Babylon `WebGPUEngine`.
 *
 * Babylon has no public accessor for it: the engine keeps it in `_device`, an
 * underscore field. This is the one place the package reads that field, and
 * it checks the engine is a WebGPU one that has finished initializing, so a
 * Babylon release that renames the field fails here with a message, not three
 * calls later with `undefined is not an object`. (`@babylonjs/core` is
 * pinned by the peer range for the same reason; docs/engines/babylon.mdx.)
 */

/** What this module reads of an engine: `isWebGPU` is public API, `_device` is not. */
interface EngineWithDevice {
    readonly isWebGPU?: boolean;
    readonly _device?: GPUDevice | null;
    getClassName?(): string;
}

/**
 * The device a `WebGPUEngine` renders with (`engine._device`). Throws for the
 * WebGL engine, a `NullEngine`, and a WebGPU engine that has not finished
 * `initAsync` (`WebGPUEngine.CreateAsync` resolves after it).
 *
 * Babylon requests the adapter's features it knows, WebGPU's default limits
 * unless `deviceDescriptor.requiredLimits` is given, so a scene past 128 MiB
 * of triangles needs `new WebGPUEngine(canvas, { deviceDescriptor: { requiredLimits } })`.
 */
export function deviceFromEngine(engine: unknown): GPUDevice {
    const candidate = engine as EngineWithDevice | null;
    if (!candidate?.isWebGPU) {
        const name = candidate?.constructor?.name ?? candidate?.getClassName?.() ?? typeof engine;
        throw new Error(`@pmndrs/bvh/babylon: the engine is a ${name}, not a WebGPUEngine; GPU tracing needs WebGPU (CPU picking works on any engine)`);
    }
    const device = candidate._device;
    if (!device || typeof device.createBuffer !== 'function') {
        throw new Error('@pmndrs/bvh/babylon: the WebGPUEngine has no device (`engine._device`); await `engine.initAsync()` (or `WebGPUEngine.CreateAsync`) first, or this Babylon version moved it');
    }
    return device;
}

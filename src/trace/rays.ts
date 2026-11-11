/**
 * Rays on the CPU side, in the same 32-byte layout the WGSL walk's `BvhRay` reads.
 */

/** A ray on the CPU side, as `BvhRay` holds it on the GPU. */
export interface TraceRay {
    readonly origin: readonly [number, number, number];
    /** Normalized. */
    readonly direction: readonly [number, number, number];
    readonly tMin: number;
    readonly tMax: number;
}

/** Bytes per ray in a batch: `BvhRay` is 32 bytes. */
export const RAY_BYTES = 32;

/** Floats per ray in a packed ray array: `origin (3), tMin, direction (3), tMax`. */
export const RAY_FLOATS = RAY_BYTES / 4;

/** Packs rays into the `BvhRay` layout, ready for `writeBuffer`. */
export function packRays(rays: readonly TraceRay[]): Float32Array {
    const packed = new Float32Array(rays.length * RAY_FLOATS);
    rays.forEach((ray, index) => {
        const base = index * RAY_FLOATS;
        packed.set(ray.origin, base);
        packed[base + 3] = ray.tMin;
        packed.set(ray.direction, base + 4);
        packed[base + 7] = ray.tMax;
    });
    return packed;
}

export function unpackRay(packed: Float32Array, index: number): TraceRay {
    const base = index * RAY_FLOATS;
    return {
        origin: [packed[base], packed[base + 1], packed[base + 2]],
        tMin: packed[base + 3],
        direction: [packed[base + 4], packed[base + 5], packed[base + 6]],
        tMax: packed[base + 7],
    };
}

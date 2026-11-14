/**
 * Timestamp queries on the fake GPU (fake-gpu.ts), for code that times its
 * passes: the device gets the `timestamp-query` feature, query sets, and
 * encoders that resolve them; a timing readback (a `MAP_READ` buffer whose
 * label ends in "timestamps read") maps at once and reads every pass as
 * `passMs` long. Test-only.
 */

import type { FakeGpu } from './fake-gpu.ts';

export function addFakeTimestamps(gpu: FakeGpu, passMs = 1.5): { readonly querySets: number; readonly resolves: number } {
    const counts = { querySets: 0, resolves: 0 };
    const device = gpu.device as unknown as Record<string, unknown> & { features: Set<string> };
    device.features.add('timestamp-query');
    device.createQuerySet = () => {
        counts.querySets++;
        return { destroy() {} };
    };
    const createEncoder = (device.createCommandEncoder as () => Record<string, unknown>).bind(device);
    device.createCommandEncoder = () => {
        const encoder = createEncoder();
        encoder.resolveQuerySet = () => {
            counts.resolves++;
        };
        return encoder;
    };
    const createBuffer = (device.createBuffer as (descriptor: GPUBufferDescriptor) => Record<string, unknown>).bind(device);
    device.createBuffer = (descriptor: GPUBufferDescriptor) => {
        const buffer = createBuffer(descriptor);
        if (descriptor.usage & 0x1 && descriptor.label?.endsWith('timestamps read')) {
            buffer.mapAsync = () => Promise.resolve();
            buffer.getMappedRange = (offset = 0, size = descriptor.size - offset) => {
                const stamps = new BigInt64Array(size / 8);
                for (let pass = 0; pass < stamps.length / 2; pass++) {
                    stamps[pass * 2] = BigInt(pass * 10_000_000);
                    stamps[pass * 2 + 1] = BigInt(pass * 10_000_000 + Math.round(passMs * 1e6));
                }
                return stamps.buffer;
            };
        }
        return buffer;
    };
    return counts;
}

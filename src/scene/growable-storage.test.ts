/** The scene's growable node and triangle buffers (ported from ddgi-core), the allocator, and adoption. */

import { describe, expect, it } from 'vitest';
import { createFakeGpu, type FakeBuffer } from '../../tests/support/fake-gpu.ts';
import { GrowableStorage } from './growable-storage.ts';
import { RangeAllocator } from './range-allocator.ts';

describe('growable storage', () => {
    it('doubles when a little more is needed, keeping the ranges in use', () => {
        const gpu = createFakeGpu();
        const storage = new GrowableStorage(gpu.device, 'test', 4, 100);
        expect(storage.allocate(80)).toBe(0);
        storage.write(0, new Uint32Array(80).fill(7).buffer);
        const encoder = gpu.createEncoder();
        storage.grow(encoder, 40, [{ offset: 0, count: 80 }]);
        gpu.submit(encoder);
        expect(storage.byteLength).toBe(200 * 4);
        expect(new Uint32Array((storage.buffer as unknown as FakeBuffer).bytes.buffer, 0, 80).every((word) => word === 7)).toBe(true);
        expect(storage.allocate(40)).toBe(80);
        expect(storage.retiredByteLength).toBe(400);
        storage.releaseRetired();
        expect(storage.retiredByteLength).toBe(0);
    });

    it('sizes a big jump by what must fit, plus the headroom a growth asks for', () => {
        const gpu = createFakeGpu();
        const exact = new GrowableStorage(gpu.device, 'exact', 4, 10);
        const roomy = new GrowableStorage(gpu.device, 'roomy', 4, 10);
        exact.grow(gpu.createEncoder(), 1000, []);
        roomy.grow(gpu.createEncoder(), 1000, [], 1 / 8);
        expect(exact.byteLength).toBe(1000 * 4);
        expect(roomy.byteLength).toBe(1125 * 4);
        expect(() => new GrowableStorage(createFakeGpu({ maxStorageBufferBindingSize: 64 }).device, 'small', 4, 4).grow(gpu.createEncoder(), 100, [])).toThrow(/maxStorageBufferBindingSize/);
    });

    it('adopts a buffer with its ranges in place, only while empty', () => {
        const gpu = createFakeGpu();
        const storage = new GrowableStorage(gpu.device, 'test', 4, 10);
        const outputs = gpu.device.createBuffer({ size: 400, usage: 0x4 | 0x8 | 0x80 });
        storage.adopt(outputs, [{ offset: 0, count: 30 }, { offset: 30, count: 20 }]);
        expect(storage.buffer).toBe(outputs);
        expect(storage.usedBytes).toBe(200);
        // The rest of its capacity is free.
        expect(storage.allocate(50)).toBe(50);
        expect(storage.allocate(1)).toBeNull();
        expect(() => storage.adopt(gpu.device.createBuffer({ size: 400, usage: 0x8c }), [])).toThrow(/holds nothing/);
        expect(() => new GrowableStorage(gpu.device, 'x', 4, 1).adopt(gpu.device.createBuffer({ size: 16, usage: 0x80 }), [])).toThrow(/usage/);
    });
});

describe('range allocator reserve', () => {
    it('marks a range inside a free one as used, splitting it, and refuses overlaps', () => {
        const allocator = new RangeAllocator(100);
        allocator.reserve(10, 20);
        expect(allocator.used).toBe(20);
        expect(allocator.allocate(10)).toBe(0);
        expect(allocator.allocate(5)).toBe(30);
        expect(() => allocator.reserve(25, 10)).toThrow(RangeError);
        allocator.release(10, 20);
        expect(allocator.allocate(20)).toBe(10);
    });
});

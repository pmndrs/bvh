/**
 * A GPU storage buffer holding variable-sized ranges (one per mesh tree) that
 * grows by doubling, for the scene's node and triangle buffers. Ported from
 * ddgi-core's `platform/trace/growable-storage.ts`.
 *
 * Growing makes a new buffer and records copies of the ranges still in use
 * into the caller's encoder. Ranges keep their offsets, so a mesh's place never
 * changes and nothing that points at it needs rewriting. The old buffer is
 * kept until the next {@link GrowableStorage.releaseRetired} (the next
 * update), because this frame's commands still copy out of it.
 *
 * Order on the queue matters: a `queue.writeBuffer` made while encoding lands
 * before the caller submits, and the copies recorded in the encoder run after
 * it. So the growth copies only cover ranges that were already resident,
 * never one being written this frame.
 *
 * {@link GrowableStorage.adopt} takes an existing buffer (a GPU build's
 * output) as the storage itself, with its ranges already in place: the
 * `gpuTreeOutputs: 'adopt'` lever of `SceneTrees`.
 */

import { RangeAllocator } from './range-allocator.ts';

export interface ResidentRange {
    readonly offset: number;
    readonly count: number;
}

const USAGE = { COPY_SRC: 0x4, COPY_DST: 0x8, STORAGE: 0x80 } as const;

export class GrowableStorage {
    private readonly device: GPUDevice;
    private readonly label: string;
    /** Bytes per unit (a node, a triangle). */
    readonly unitBytes: number;
    private current: GPUBuffer;
    private allocator: RangeAllocator;
    private retired: GPUBuffer[] = [];
    private retiredBytes = 0;
    /** Increments every time {@link buffer} is replaced. */
    version = 0;

    constructor(device: GPUDevice, label: string, unitBytes: number, initialUnits: number) {
        this.device = device;
        this.label = label;
        this.unitBytes = unitBytes;
        this.allocator = new RangeAllocator(initialUnits);
        this.current = this.createBuffer(initialUnits);
    }

    get buffer(): GPUBuffer {
        return this.current;
    }

    /** Bytes of the GPU buffer (allocated, not only used). */
    get byteLength(): number {
        return this.current.size;
    }

    /** Bytes of buffers replaced by a growth, alive until {@link releaseRetired}. */
    get retiredByteLength(): number {
        return this.retiredBytes;
    }

    /** Bytes holding live ranges. */
    get usedBytes(): number {
        return this.allocator.used * this.unitBytes;
    }

    allocate(count: number): number | null {
        return this.allocator.allocate(count);
    }

    release(offset: number, count: number): void {
        this.allocator.release(offset, count);
    }

    /**
     * Grows so that `extraUnits` more units fit after the current high-water
     * mark, copying `resident` ranges into the new buffer. Throws a
     * `RangeError` naming the limit when one storage binding can't hold it.
     *
     * The new size is the larger of double the old one and what must fit
     * times `1 + headroom` (default 0). A caller that knows ranges will soon
     * be replaced by slightly larger ones asks for headroom, so they fit
     * without another growth; a growth that doubles has room anyway.
     */
    grow(encoder: GPUCommandEncoder, extraUnits: number, resident: readonly ResidentRange[], headroom = 0): void {
        const limits = this.device.limits;
        const limitBytes = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
        const limitUnits = Math.floor(limitBytes / this.unitBytes);
        const minimumUnits = this.allocator.capacityToFit(extraUnits);
        if (minimumUnits > limitUnits) {
            const limit = limits.maxStorageBufferBindingSize <= limits.maxBufferSize ? 'maxStorageBufferBindingSize' : 'maxBufferSize';
            throw new RangeError(`@pmndrs/bvh: the ${this.label} buffer needs ${limit} >= ${minimumUnits * this.unitBytes} bytes, the device allows ${limitBytes}`);
        }
        // Doubling keeps reallocations logarithmic while a scene loads.
        const units = Math.min(limitUnits, Math.max(Math.ceil(minimumUnits * (1 + headroom)), this.allocator.capacity * 2));
        const buffer = this.createBuffer(units);

        // Neighbouring ranges are copied as one.
        const sorted = [...resident].sort((a, b) => a.offset - b.offset);
        let runStart = -1;
        let runEnd = -1;
        const flush = (): void => {
            if (runStart < 0) return;
            encoder.copyBufferToBuffer(this.current, runStart * this.unitBytes, buffer, runStart * this.unitBytes, (runEnd - runStart) * this.unitBytes);
        };
        for (const range of sorted) {
            if (range.offset === runEnd) {
                runEnd += range.count;
            } else {
                flush();
                runStart = range.offset;
                runEnd = range.offset + range.count;
            }
        }
        flush();

        this.retire(this.current);
        this.current = buffer;
        this.allocator.grow(units);
        this.version++;
    }

    /**
     * Makes `buffer` the storage, with `ranges` already written in it and
     * marked in use; the rest of its capacity is free. Only while nothing is
     * allocated (the old buffer holds nothing anyone reads, and is retired).
     * The buffer needs `STORAGE | COPY_SRC | COPY_DST` usage, and is the
     * storage's from now on: it is destroyed with it or when a growth
     * replaces it.
     */
    adopt(buffer: GPUBuffer, ranges: readonly ResidentRange[]): void {
        if (this.allocator.used > 0) throw new Error(`@pmndrs/bvh: the ${this.label} storage can only adopt a buffer while it holds nothing`);
        const required = USAGE.STORAGE | USAGE.COPY_SRC | USAGE.COPY_DST;
        if ((buffer.usage & required) !== required) throw new Error(`@pmndrs/bvh: an adopted ${this.label} buffer needs STORAGE, COPY_SRC and COPY_DST usage`);
        const units = Math.floor(buffer.size / this.unitBytes);
        const allocator = new RangeAllocator(units);
        for (const range of ranges) if (range.count > 0) allocator.reserve(range.offset, range.count);
        this.retire(this.current);
        this.current = buffer;
        this.allocator = allocator;
        this.version++;
    }

    /** Writes `data` at a range this storage handed out. */
    write(offset: number, data: ArrayBuffer | ArrayBufferView<ArrayBuffer>, byteLength?: number): void {
        if (ArrayBuffer.isView(data)) this.device.queue.writeBuffer(this.current, offset * this.unitBytes, data.buffer, data.byteOffset, byteLength ?? data.byteLength);
        else this.device.queue.writeBuffer(this.current, offset * this.unitBytes, data, 0, byteLength ?? data.byteLength);
    }

    /** Destroys buffers replaced by an earlier `grow` or `adopt`. Call once the frame that copied out of them was submitted. */
    releaseRetired(): void {
        for (const buffer of this.retired) buffer.destroy();
        this.retired = [];
        this.retiredBytes = 0;
    }

    destroy(): void {
        this.releaseRetired();
        this.current.destroy();
    }

    private retire(buffer: GPUBuffer): void {
        this.retired.push(buffer);
        this.retiredBytes += buffer.size;
    }

    private createBuffer(units: number): GPUBuffer {
        return this.device.createBuffer({
            label: this.label,
            // A binding can't be empty, so an empty storage still holds one unit.
            size: Math.max(1, units) * this.unitBytes,
            usage: USAGE.STORAGE | USAGE.COPY_DST | USAGE.COPY_SRC,
        });
    }
}

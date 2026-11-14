/**
 * GPU time of a few compute passes from timestamp queries, for the library's
 * own timed work: `GpuRays` dispatches, `SceneTrees`' GPU build batches
 * (`timestamps: true`) and `createMeshLayoutBuilder`'s batches
 * (`onGpuBatch`). Internal: apps time their own passes with
 * `timestampWrites` where the API takes them (`TraceBatch`,
 * `SceneTrees.update`, `GpuHplocBatch.encode`).
 *
 * Only on a device with the `timestamp-query` feature ({@link hasTimestamps});
 * elsewhere the callers report `null`. Chrome rounds timestamps to 0.1 ms
 * unless WebGPU's developer features are on.
 */

/** WebGPU's flags, spelled out: Node (unit tests) has no `GPUBufferUsage` or `GPUMapMode`. */
const QUERY_RESOLVE = 0x200;
const COPY_SRC = 0x4;
const COPY_DST = 0x8;
const MAP_READ = 0x1;
const MAP_MODE_READ = 0x1;

/** Whether `device` can time passes (the `timestamp-query` feature). */
export function hasTimestamps(device: GPUDevice): boolean {
    return device.features.has('timestamp-query');
}

/**
 * Timestamp pairs for up to `capacity` passes of one measurement: hand
 * {@link writes} to each pass, {@link resolveInto} the encoder that records
 * them, submit, then {@link read}. Reusable once read.
 */
export class PassTimestamps {
    private readonly querySet: GPUQuerySet;
    private readonly resolveBuffer: GPUBuffer;
    private readonly readBuffer: GPUBuffer;
    private used = 0;
    private reading = false;

    private constructor(
        device: GPUDevice,
        readonly capacity: number,
        label: string,
    ) {
        this.querySet = device.createQuerySet({ label: `${label} timestamps`, type: 'timestamp', count: capacity * 2 });
        this.resolveBuffer = device.createBuffer({ label: `${label} timestamps resolve`, size: capacity * 16, usage: QUERY_RESOLVE | COPY_SRC });
        this.readBuffer = device.createBuffer({ label: `${label} timestamps read`, size: capacity * 16, usage: MAP_READ | COPY_DST });
    }

    /** A measurement of up to `capacity` passes, or null without `timestamp-query`. */
    static create(device: GPUDevice, capacity: number, label = 'bvh'): PassTimestamps | null {
        if (!hasTimestamps(device) || capacity < 1) return null;
        return new PassTimestamps(device, capacity, label);
    }

    /** True while a measurement is recorded or being read: don't start another. */
    get busy(): boolean {
        return this.used > 0 || this.reading;
    }

    /** The timestamp writes of the measurement's next pass. */
    writes(): GPUComputePassTimestampWrites {
        if (this.used >= this.capacity) throw new RangeError(`PassTimestamps: more than ${this.capacity} passes`);
        const at = this.used++ * 2;
        return { querySet: this.querySet, beginningOfPassWriteIndex: at, endOfPassWriteIndex: at + 1 };
    }

    /** Resolves the passes recorded so far into `encoder` (after them); {@link read} after the submit. */
    resolveInto(encoder: GPUCommandEncoder): void {
        if (this.used === 0) return;
        encoder.resolveQuerySet(this.querySet, 0, this.used * 2, this.resolveBuffer, 0);
        encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.readBuffer, 0, this.used * 16);
    }

    /** Forgets passes recorded but never submitted (a measurement that failed midway). */
    reset(): void {
        if (!this.reading) this.used = 0;
    }

    /** Milliseconds per pass, in order. Call once the encoder of {@link resolveInto} was submitted. */
    async read(): Promise<number[]> {
        const passes = this.used;
        if (passes === 0) return [];
        this.reading = true;
        try {
            await this.readBuffer.mapAsync(MAP_MODE_READ, 0, passes * 16);
            const stamps = new BigInt64Array(this.readBuffer.getMappedRange(0, passes * 16).slice(0));
            this.readBuffer.unmap();
            const ms: number[] = [];
            for (let pass = 0; pass < passes; pass++) ms.push(Math.max(0, Number(stamps[pass * 2 + 1] - stamps[pass * 2])) / 1e6);
            return ms;
        } finally {
            this.used = 0;
            this.reading = false;
        }
    }

    destroy(): void {
        this.querySet.destroy();
        this.resolveBuffer.destroy();
        this.readBuffer.destroy();
    }
}

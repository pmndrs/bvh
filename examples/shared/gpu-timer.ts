/**
 * GPU time of one or more compute passes from timestamp queries: the
 * overlay's "build GPU" and the ray dispatches' Mrays/s. Null where the
 * device has no `timestamp-query` (Safari, Firefox, some Chrome builds).
 */

export class GpuTimer {
    private readonly querySet: GPUQuerySet;
    private readonly resolve: GPUBuffer;
    private readonly read: GPUBuffer;
    private used = 0;
    private busy = false;

    private constructor(
        private readonly device: GPUDevice,
        private readonly capacity: number,
    ) {
        this.querySet = device.createQuerySet({ type: 'timestamp', count: capacity * 2 });
        this.resolve = device.createBuffer({ size: capacity * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
        this.read = device.createBuffer({ size: capacity * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }

    /** A timer for up to `capacity` passes per measurement, or null without `timestamp-query`. */
    static create(device: GPUDevice, capacity = 16): GpuTimer | null {
        return device.features.has('timestamp-query') ? new GpuTimer(device, capacity) : null;
    }

    /** True while a measurement is being read back (don't start another). */
    get pending(): boolean {
        return this.busy;
    }

    /** Timestamp writes for the next pass of this measurement. */
    pass(): GPUComputePassTimestampWrites {
        if (this.used >= this.capacity) throw new RangeError(`GpuTimer: more than ${this.capacity} passes`);
        const at = this.used++ * 2;
        return { querySet: this.querySet, beginningOfPassWriteIndex: at, endOfPassWriteIndex: at + 1 };
    }

    /** Resolves the passes recorded so far into `encoder`; then {@link result} after the submit. */
    resolveInto(encoder: GPUCommandEncoder): void {
        if (this.used === 0) return;
        encoder.resolveQuerySet(this.querySet, 0, this.used * 2, this.resolve, 0);
        encoder.copyBufferToBuffer(this.resolve, 0, this.read, 0, this.used * 16);
    }

    /** Milliseconds per pass, in order. Call after the encoder from {@link resolveInto} was submitted. */
    async result(): Promise<number[]> {
        const passes = this.used;
        this.used = 0;
        if (passes === 0) return [];
        this.busy = true;
        try {
            await this.read.mapAsync(GPUMapMode.READ, 0, passes * 16);
            const stamps = new BigInt64Array(this.read.getMappedRange(0, passes * 16).slice(0));
            this.read.unmap();
            const ms: number[] = [];
            for (let i = 0; i < passes; i++) ms.push(Math.max(0, Number(stamps[i * 2 + 1] - stamps[i * 2])) / 1e6);
            return ms;
        } finally {
            this.busy = false;
        }
    }

    destroy(): void {
        this.querySet.destroy();
        this.resolve.destroy();
        this.read.destroy();
    }
}

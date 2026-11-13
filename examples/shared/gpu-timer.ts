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

    /** Forgets passes recorded but never resolved and read (a measurement that failed midway). */
    reset(): void {
        if (!this.busy) this.used = 0;
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

/**
 * GPU time of work recorded by someone else's code (a `SceneTrees.update`,
 * which takes no timestamp writes): two tiny compute passes around it, the
 * first writing a timestamp at its start, the second at its end. Includes
 * the two one-thread dispatches; null without `timestamp-query`.
 */
export class GpuBracket {
    private readonly querySet: GPUQuerySet;
    private readonly resolve: GPUBuffer;
    private readonly read: GPUBuffer;
    private readonly pipeline: GPUComputePipeline;
    private state: 'idle' | 'open' | 'recorded' | 'reading' = 'idle';

    private constructor(device: GPUDevice) {
        this.querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
        this.resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
        this.read = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const module = device.createShaderModule({ label: 'bracket', code: '@compute @workgroup_size(1) fn main() {}' });
        this.pipeline = device.createComputePipeline({ label: 'bracket', layout: 'auto', compute: { module, entryPoint: 'main' } });
    }

    static create(device: GPUDevice): GpuBracket | null {
        return device.features.has('timestamp-query') ? new GpuBracket(device) : null;
    }

    /** True while a measurement is recorded or being read (don't start another). */
    get busy(): boolean {
        return this.state !== 'idle';
    }

    private pass(encoder: GPUCommandEncoder, timestampWrites: GPUComputePassTimestampWrites): void {
        const pass = encoder.beginComputePass({ label: 'bracket', timestampWrites });
        pass.setPipeline(this.pipeline);
        pass.dispatchWorkgroups(1);
        pass.end();
    }

    /** Before the work: a timestamp. */
    begin(encoder: GPUCommandEncoder): void {
        if (this.state !== 'idle') throw new Error('GpuBracket: a measurement is already under way');
        this.pass(encoder, { querySet: this.querySet, beginningOfPassWriteIndex: 0 });
        this.state = 'open';
    }

    /** After the work: a timestamp, resolved for {@link result} after the submit. */
    end(encoder: GPUCommandEncoder): void {
        if (this.state !== 'open') throw new Error('GpuBracket: end() without begin()');
        this.pass(encoder, { querySet: this.querySet, endOfPassWriteIndex: 1 });
        encoder.resolveQuerySet(this.querySet, 0, 2, this.resolve, 0);
        encoder.copyBufferToBuffer(this.resolve, 0, this.read, 0, 16);
        this.state = 'recorded';
    }

    /** Milliseconds between the two timestamps, or null if nothing was recorded. Call after the submit. */
    async result(): Promise<number | null> {
        if (this.state !== 'recorded') return null;
        this.state = 'reading';
        try {
            await this.read.mapAsync(GPUMapMode.READ);
            const stamps = new BigInt64Array(this.read.getMappedRange().slice(0));
            this.read.unmap();
            return Math.max(0, Number(stamps[1] - stamps[0])) / 1e6;
        } finally {
            this.state = 'idle';
        }
    }

    destroy(): void {
        this.querySet.destroy();
        this.resolve.destroy();
        this.read.destroy();
    }
}

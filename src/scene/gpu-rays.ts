/**
 * Rays on the GPU without writing a kernel: picking (one ray) and batches
 * (any number) over a scene, with the buffers, the dispatch, the readback and
 * the hit decoding done here. It runs the package's batch kernel
 * (`TraceBatchKernel`, the same WGSL walk an app includes with
 * `traceModuleWgsl`), so its hits are the walk's.
 *
 * **One source type.** A {@link TraceSource} is an uploaded scene
 * (`uploadTraceScene`'s buffers) or anything that keeps its buffers up to
 * date and hands them out as `traceBuffers`, as `SceneTrees` does. Both feed
 * the same kernels; {@link traceBuffersOf} reads the buffers at each trace,
 * so a `SceneTrees` that grew its buffers is traced in its new ones.
 *
 * One trace at a time per `GpuRays` (they share the buffers and the
 * readback): calls queue up, and {@link GpuRays.pending} says one is in flight,
 * so a frame loop can skip a trace rather than queue a backlog.
 */

import { PassTimestamps } from '../gpu-timestamps.ts';
import { TraceBatchKernel, type TraceBatchKernelOptions, type TraceSceneBuffers } from '../trace/trace-gpu.ts';
import { HIT_RECORD_BYTES, HIT_RECORD_WORD, WALK_COUNTS_BYTES, WALK_COUNTS_WORD, type AnyHitOrder } from '../trace/wgsl.ts';
import { RAY_BYTES, RAY_FLOATS, type TraceRay } from '../trace/rays.ts';

const USAGE = { MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, STORAGE: 0x80 } as const;
const MAP_MODE_READ = 0x1;

/**
 * What rays are traced over: an uploaded scene (`uploadTraceScene`'s
 * {@link TraceSceneBuffers}) or a scene that keeps its buffers up to date and
 * hands them out as `traceBuffers` (a `SceneTrees`). The buffers name their
 * formats, so a kernel can be compiled for them.
 */
export type TraceSource = TraceSceneBuffers | { readonly traceBuffers: TraceSceneBuffers };

/** The buffers a {@link TraceSource} holds now (a `SceneTrees`' are replaced when they grow: read them at each trace). */
export function traceBuffersOf(source: TraceSource): TraceSceneBuffers {
    return 'traceBuffers' in source ? source.traceBuffers : source;
}

/** One closest hit, decoded from the batch kernel's 40-byte record (`HIT_RECORD_WORD`). */
export interface GpuRayHit {
    /** Distance along the ray. */
    readonly t: number;
    /** The hit instance's `instanceIndex` (a merged triangle's source instance). */
    readonly instance: number;
    /** The triangle's index in its source mesh. */
    readonly triangle: number;
    /** The authored front side was hit. */
    readonly frontFace: boolean;
    /** Barycentrics of the second and third corners. */
    readonly barycentrics: readonly [number, number];
    /** The triangle's or instance's material, or `NO_MATERIAL`. */
    readonly material: number;
    /** The authored front side's unit normal, world space. */
    readonly normal: readonly [number, number, number];
}

/** What one trace took: CPU wall time always, GPU time where the device times passes. */
export interface GpuRaysTiming {
    readonly count: number;
    /** GPU milliseconds of the dispatch (timestamps), or null without `timestamp-query` (or with `timestamps: false`). */
    readonly gpuMs: number | null;
    /** CPU wall milliseconds from the submit to the results on the CPU (dispatch plus readback). */
    readonly wallMs: number;
}

/** A closest-hit batch's results. */
export interface GpuClosestHits extends GpuRaysTiming {
    /** Rays that hit. */
    readonly hits: number;
    /** The hit of ray `index`, or null on a miss. */
    hit(index: number): GpuRayHit | null;
    /** The raw records, `HIT_RECORD_BYTES` per ray (`HIT_RECORD_WORD`), for code that reads them itself. */
    readonly records: ArrayBuffer;
}

/** An any-hit batch's results. */
export interface GpuAnyHits extends GpuRaysTiming {
    /** Rays blocked. */
    readonly blockedCount: number;
    /** Whether ray `index` is blocked. */
    blocked(index: number): boolean;
    /** One word per ray: 1 blocked, 0 clear. */
    readonly results: Uint32Array;
}

/** One ray's counted walk ({@link GpuRays.countWalk}): its hit and the walk's work, from a `BvhWalkCounts` record. */
export interface GpuWalkCount {
    /** Closest: the ray hit; any: it is blocked. */
    readonly hit: boolean;
    /** Closest: the hit's distance (negative on a miss). Any: 1 blocked, -1 clear. */
    readonly t: number;
    /** Closest hits: the hit's instance and triangle (0 otherwise). */
    readonly instance: number;
    readonly triangle: number;
    /** Iterations of the walk's loop: one per node or leaf processed, both levels. */
    readonly steps: number;
    /** Inner nodes of mesh trees whose two child boxes were tested. */
    readonly meshNodes: number;
    /** Inner nodes of the top level whose two child boxes were tested. */
    readonly topLevelNodes: number;
    /** Instances entered (a merged mesh counts once). */
    readonly instanceEntries: number;
    readonly triangleTests: number;
    /** Instances entered in which the walk accepted a hit (closest) or found the blocker (any). */
    readonly instancesWithHit: number;
}

/** A counted batch's results ({@link GpuRays.countWalk}). */
export interface GpuWalkCounts extends GpuRaysTiming {
    /** The walk's work for ray `index`. */
    counts(index: number): GpuWalkCount;
    /** Steps per ray, averaged over the batch, and the most any ray took. */
    readonly meanSteps: number;
    readonly maxSteps: number;
    /** The raw records, `WALK_COUNTS_BYTES` per ray (`WALK_COUNTS_WORD`), as words. */
    readonly records: Uint32Array;
}

/** Options of {@link GpuRays.countWalk}. */
export interface GpuCountOptions extends GpuTraceOptions {
    /** Which walk to count. Default `'closest'`. */
    readonly mode?: 'closest' | 'any';
}

/** Options of {@link GpuRays.create}. */
export interface GpuRaysOptions {
    /** Label prefix of the GPU objects. */
    readonly label?: string;
    /** Time each dispatch with timestamp queries where the device has `timestamp-query`. Default true. */
    readonly timestamps?: boolean;
    /** The triangle filter (`bvhTrianglePasses`), compiled in; default every triangle blocks. */
    readonly triangleFilter?: string;
    /** The any-hit walk's child order, compiled in. Default `'far'`. */
    readonly anyHitOrder?: AnyHitOrder;
}

/** Options of one trace. */
export interface GpuTraceOptions {
    /** The batch's ray mask (instances whose mask shares no bit are skipped). Default all ones. */
    readonly mask?: number;
}

/** Options of {@link GpuRays.pick}. */
export interface GpuPickOptions extends GpuTraceOptions {
    /** Default 0. */
    readonly tMin?: number;
    /** Default `Infinity` (traced as the largest finite f32). */
    readonly tMax?: number;
}

const MAX_F32 = 3.4028234663852886e38;

/**
 * Picking and batches of rays over a {@link TraceSource}, on the GPU. Rays
 * are packed `RAY_FLOATS` floats each (`packRays`): origin, tMin, direction
 * (normalized), tMax, the walk's `BvhRay`.
 *
 *     const rays = await GpuRays.create(device, trees);          // a SceneTrees, or uploadTraceScene's buffers
 *     const hit = await rays.pick([0, 1, 5], [0, 0, -1]);       // one ray: a GpuRayHit or null
 *     const batch = await rays.traceClosest(packRays(list));      // thousands: batch.hit(i)
 *
 * A `SceneTrees` is traced as of its last `update` that was submitted.
 */
export class GpuRays {
    private rays: GPUBuffer | null = null;
    private results: GPUBuffer | null = null;
    private readback: GPUBuffer | null = null;
    private capacity = 0;
    private queued = 0;
    private destroyed = false;
    private queue: Promise<unknown> = Promise.resolve();
    private readonly timer: PassTimestamps | null;
    /** The counted walk's kernel, compiled at the first {@link countWalk}. */
    private countedKernel: Promise<TraceBatchKernel> | null = null;

    private constructor(
        readonly device: GPUDevice,
        /** What the rays are traced over. */
        readonly source: TraceSource,
        private readonly kernel: TraceBatchKernel,
        private readonly kernelOptions: TraceBatchKernelOptions,
        private readonly label: string,
        timestamps: boolean,
    ) {
        this.timer = timestamps ? PassTimestamps.create(device, 1, label) : null;
    }

    /** Compiles the batch kernel for the source's triangle and node formats (read from its buffers). */
    static async create(device: GPUDevice, source: TraceSource, options: GpuRaysOptions = {}): Promise<GpuRays> {
        const buffers = traceBuffersOf(source);
        const label = options.label ?? 'bvh rays';
        const kernelOptions: TraceBatchKernelOptions = {
            label,
            triangleLeaves: buffers.triangleLeaves,
            positionLayout: buffers.positionLayout,
            nodeFormat: buffers.nodeFormat,
            triangleFilter: options.triangleFilter,
            anyHitOrder: options.anyHitOrder,
        };
        const kernel = await TraceBatchKernel.create(device, kernelOptions);
        return new GpuRays(device, source, kernel, kernelOptions, label, options.timestamps ?? true);
    }

    /** True while a trace is in flight or queued (a frame loop can skip its trace then). */
    get pending(): boolean {
        return this.queued > 0;
    }

    /** Closest hits of `rays` (`RAY_FLOATS` per ray). */
    traceClosest(rays: Float32Array, options: GpuTraceOptions = {}): Promise<GpuClosestHits> {
        return this.enqueue(async () => {
            const { bytes, timing } = await this.run(this.kernel, 'closest', rays, options.mask, HIT_RECORD_BYTES);
            return closestHits(bytes, timing);
        });
    }

    /**
     * The walk's work per ray: the counted walk (`counters: true`, the same
     * walk with step counters, so the same hits) over `rays`, for heat maps
     * and tools. Its kernel is compiled at the first call; its times mean
     * little (it writes more per ray), so measure speed with
     * {@link traceClosest} or {@link traceAny}.
     */
    countWalk(rays: Float32Array, options: GpuCountOptions = {}): Promise<GpuWalkCounts> {
        const mode = options.mode ?? 'closest';
        this.countedKernel ??= TraceBatchKernel.create(this.device, { ...this.kernelOptions, label: `${this.label} counted`, counters: true });
        const kernel = this.countedKernel;
        return this.enqueue(async () => {
            const { bytes, timing } = await this.run(await kernel, mode, rays, options.mask, WALK_COUNTS_BYTES);
            return walkCounts(bytes, timing, mode);
        });
    }

    /** Whether each of `rays` is blocked (stops at the first blocker; shadow and visibility rays). */
    traceAny(rays: Float32Array, options: GpuTraceOptions = {}): Promise<GpuAnyHits> {
        return this.enqueue(async () => {
            const { bytes, timing } = await this.run(this.kernel, 'any', rays, options.mask, 4);
            const results = new Uint32Array(bytes);
            let blockedCount = 0;
            for (const word of results) if (word !== 0) blockedCount++;
            return { ...timing, blockedCount, results, blocked: (index: number) => results[index] !== 0 };
        });
    }

    /** One ray from `origin` along `direction` (normalized here): its closest hit, or null. */
    async pick(origin: ArrayLike<number>, direction: ArrayLike<number>, options: GpuPickOptions = {}): Promise<GpuRayHit | null> {
        const length = Math.hypot(direction[0], direction[1], direction[2]);
        if (!(length > 0)) throw new RangeError('GpuRays.pick: the direction has no length');
        const ray: TraceRay = {
            origin: [origin[0], origin[1], origin[2]],
            direction: [direction[0] / length, direction[1] / length, direction[2] / length],
            tMin: options.tMin ?? 0,
            tMax: Math.min(options.tMax ?? Infinity, MAX_F32),
        };
        const packed = new Float32Array(RAY_FLOATS);
        packed.set(ray.origin, 0);
        packed[3] = ray.tMin;
        packed.set(ray.direction, 4);
        packed[7] = ray.tMax;
        return (await this.traceClosest(packed, options)).hit(0);
    }

    /** Destroys the buffers once the traces already asked for have finished (a page swapping scenes retires the old one). */
    async retire(): Promise<void> {
        await this.queue;
        this.destroy();
    }

    /** Frees the buffers and the kernel's uniforms at once; traces still queued reject. */
    destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
        this.rays?.destroy();
        this.results?.destroy();
        this.readback?.destroy();
        this.timer?.destroy();
        this.kernel.destroy();
        void this.countedKernel?.then((kernel) => kernel.destroy(), () => undefined);
    }

    /** One trace at a time: they share the buffers and the readback. */
    private enqueue<T>(trace: () => Promise<T>): Promise<T> {
        this.queued++;
        const next = this.queue.then(trace).finally(() => this.queued--);
        this.queue = next.catch(() => undefined);
        return next;
    }

    private ensure(count: number, resultBytes: number): void {
        if (count <= this.capacity) return;
        this.rays?.destroy();
        this.results?.destroy();
        this.readback?.destroy();
        this.capacity = Math.max(count, 64);
        // Results sized for closest-hit records; any-hit's words fit in them.
        const bytes = this.capacity * Math.max(resultBytes, HIT_RECORD_BYTES);
        this.rays = this.device.createBuffer({ label: `${this.label} rays`, size: this.capacity * RAY_BYTES, usage: USAGE.STORAGE | USAGE.COPY_DST });
        this.results = this.device.createBuffer({ label: `${this.label} results`, size: bytes, usage: USAGE.STORAGE | USAGE.COPY_SRC });
        this.readback = this.device.createBuffer({ label: `${this.label} readback`, size: bytes, usage: USAGE.MAP_READ | USAGE.COPY_DST });
    }

    /** Uploads the rays, dispatches `kernel`, reads `resultBytes` per ray back. */
    private async run(kernel: TraceBatchKernel, mode: 'closest' | 'any', packed: Float32Array, mask: number | undefined, resultBytes: number): Promise<{ bytes: ArrayBuffer; timing: GpuRaysTiming }> {
        if (this.destroyed) throw new Error('GpuRays: destroyed');
        if (packed.length % RAY_FLOATS !== 0) throw new RangeError(`GpuRays: rays are ${RAY_FLOATS} floats each, got ${packed.length} floats`);
        const count = packed.length / RAY_FLOATS;
        if (count === 0) return { bytes: new ArrayBuffer(0), timing: { count, gpuMs: null, wallMs: 0 } };
        this.ensure(count, resultBytes);
        this.device.queue.writeBuffer(this.rays!, 0, packed.buffer, packed.byteOffset, packed.byteLength);
        const encoder = this.device.createCommandEncoder({ label: `${this.label} ${mode}` });
        const timer = this.timer && !this.timer.busy ? this.timer : null;
        const timestampWrites = timer?.writes();
        try {
            kernel.encode(encoder, traceBuffersOf(this.source), { mode, rays: this.rays!, results: this.results!, count, mask, timestampWrites });
        } catch (error) {
            timer?.reset();
            throw error;
        }
        timer?.resolveInto(encoder);
        const size = count * resultBytes;
        encoder.copyBufferToBuffer(this.results!, 0, this.readback!, 0, size);
        const start = performance.now();
        this.device.queue.submit([encoder.finish()]);
        const [ms] = await Promise.all([timer ? timer.read() : Promise.resolve([]), this.readback!.mapAsync(MAP_MODE_READ, 0, size)]);
        const bytes = this.readback!.getMappedRange(0, size).slice(0);
        this.readback!.unmap();
        return { bytes, timing: { count, gpuMs: ms.length > 0 ? ms[0] : null, wallMs: performance.now() - start } };
    }
}

/** Decodes closest-hit records (`HIT_RECORD_BYTES` each). */
function closestHits(bytes: ArrayBuffer, timing: GpuRaysTiming): GpuClosestHits {
    const floats = new Float32Array(bytes);
    const words = new Uint32Array(bytes);
    const stride = HIT_RECORD_BYTES / 4;
    const w = HIT_RECORD_WORD;
    let hits = 0;
    for (let ray = 0; ray < timing.count; ray++) if (floats[ray * stride + w.t] >= 0) hits++;
    return {
        ...timing,
        hits,
        records: bytes,
        hit(index) {
            if (!(index >= 0 && index < timing.count)) throw new RangeError(`GpuRays: no ray ${index} in a batch of ${timing.count}`);
            const base = index * stride;
            const t = floats[base + w.t];
            if (!(t >= 0)) return null;
            return {
                t,
                instance: words[base + w.instance],
                triangle: words[base + w.triangle],
                frontFace: words[base + w.frontFace] !== 0,
                barycentrics: [floats[base + w.barycentrics], floats[base + w.barycentrics + 1]],
                material: words[base + w.material],
                normal: [floats[base + w.normal], floats[base + w.normal + 1], floats[base + w.normal + 2]],
            };
        },
    };
}

/** Decodes counted-walk records (`WALK_COUNTS_BYTES` each). */
function walkCounts(bytes: ArrayBuffer, timing: GpuRaysTiming, mode: 'closest' | 'any'): GpuWalkCounts {
    const floats = new Float32Array(bytes);
    const words = new Uint32Array(bytes);
    const stride = WALK_COUNTS_BYTES / 4;
    const w = WALK_COUNTS_WORD;
    let totalSteps = 0;
    let maxSteps = 0;
    for (let ray = 0; ray < timing.count; ray++) {
        const steps = words[ray * stride + w.steps];
        totalSteps += steps;
        if (steps > maxSteps) maxSteps = steps;
    }
    return {
        ...timing,
        records: words,
        meanSteps: timing.count > 0 ? totalSteps / timing.count : 0,
        maxSteps,
        counts(index) {
            if (!(index >= 0 && index < timing.count)) throw new RangeError(`GpuRays: no ray ${index} in a batch of ${timing.count}`);
            const base = index * stride;
            const t = floats[base + w.t];
            return {
                hit: mode === 'closest' ? t >= 0 : t > 0,
                t,
                instance: words[base + w.instance],
                triangle: words[base + w.triangle],
                steps: words[base + w.steps],
                meshNodes: words[base + w.meshNodes],
                topLevelNodes: words[base + w.topLevelNodes],
                instanceEntries: words[base + w.instanceEntries],
                triangleTests: words[base + w.triangleTests],
                instancesWithHit: words[base + w.instancesWithHit],
            };
        },
    };
}

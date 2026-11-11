/**
 * Rays traced on the GPU over a `SceneTrees` scene with the package's own
 * batch kernel (`TraceBatchKernel`): one dispatch for any number of rays, the
 * hits read back. The examples use it for picking (one ray) and for batch
 * raycasts (thousands), timed with timestamp queries where the device has them.
 */

import { HIT_RECORD_BYTES, HIT_RECORD_WORD, RAY_BYTES, RAY_FLOATS, TraceBatchKernel, type SceneTrees } from '@pmndrs/bvh';
import type { Camera, Vector2 } from 'three';
import { Raycaster } from 'three';
import { GpuTimer } from './gpu-timer.ts';

export interface GpuHit {
    readonly t: number;
    readonly instance: number;
    readonly triangle: number;
    readonly frontFace: boolean;
    readonly u: number;
    readonly v: number;
    readonly normal: [number, number, number];
}

export interface GpuTraceResult {
    readonly count: number;
    readonly hits: number;
    /** GPU milliseconds of the dispatch (timestamps), or null. */
    readonly gpuMs: number | null;
    /** CPU wall milliseconds from submit to the results being read. */
    readonly wallMs: number;
    /** The hit of ray `index`, or null on a miss. */
    hit(index: number): GpuHit | null;
}

export class GpuRays {
    private rays: GPUBuffer | null = null;
    private results: GPUBuffer | null = null;
    private readback: GPUBuffer | null = null;
    private capacity = 0;
    private busy = false;
    private readonly timer: GpuTimer | null;

    private constructor(
        private readonly device: GPUDevice,
        private readonly trees: SceneTrees,
        private readonly kernel: TraceBatchKernel,
    ) {
        this.timer = GpuTimer.create(device, 1);
    }

    /** A kernel compiled for `trees`' triangle and node formats. */
    static async create(trees: SceneTrees): Promise<GpuRays> {
        const kernel = await TraceBatchKernel.create(trees.device, { label: 'example rays', triangleLeaves: trees.triangleLeaves, positionLayout: trees.positionLayout, nodeFormat: trees.nodeFormat });
        return new GpuRays(trees.device, trees, kernel);
    }

    /** True while a trace is in flight (the examples skip a frame's trace then). */
    get pending(): boolean {
        return this.busy;
    }

    private ensure(count: number): void {
        if (count <= this.capacity) return;
        this.rays?.destroy();
        this.results?.destroy();
        this.readback?.destroy();
        this.capacity = Math.max(count, 64);
        this.rays = this.device.createBuffer({ label: 'example rays', size: this.capacity * RAY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.results = this.device.createBuffer({ label: 'example hits', size: this.capacity * HIT_RECORD_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        this.readback = this.device.createBuffer({ label: 'example hits readback', size: this.capacity * HIT_RECORD_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }

    /**
     * Traces `packed` rays (`RAY_FLOATS` per ray: origin, tMin, direction, tMax)
     * for their closest hits, over the trees as of the last `SceneTrees.update`.
     */
    async traceClosest(packed: Float32Array, mask?: number): Promise<GpuTraceResult> {
        const count = packed.length / RAY_FLOATS;
        this.ensure(count);
        this.busy = true;
        try {
            this.device.queue.writeBuffer(this.rays!, 0, packed as Float32Array<ArrayBuffer>);
            const encoder = this.device.createCommandEncoder({ label: 'example rays' });
            const timestampWrites = this.timer && !this.timer.pending ? this.timer.pass() : undefined;
            this.kernel.encode(encoder, this.trees.traceBuffers, { mode: 'closest', rays: this.rays!, results: this.results!, count, mask, timestampWrites });
            if (timestampWrites) this.timer!.resolveInto(encoder);
            encoder.copyBufferToBuffer(this.results!, 0, this.readback!, 0, count * HIT_RECORD_BYTES);
            const start = performance.now();
            this.device.queue.submit([encoder.finish()]);
            const [ms] = await Promise.all([timestampWrites ? this.timer!.result() : Promise.resolve([]), this.readback!.mapAsync(GPUMapMode.READ, 0, count * HIT_RECORD_BYTES)]);
            const bytes = this.readback!.getMappedRange(0, count * HIT_RECORD_BYTES).slice(0);
            this.readback!.unmap();
            const wallMs = performance.now() - start;
            const floats = new Float32Array(bytes);
            const words = new Uint32Array(bytes);
            const stride = HIT_RECORD_BYTES / 4;
            const w = HIT_RECORD_WORD;
            let hits = 0;
            for (let i = 0; i < count; i++) if (floats[i * stride] >= 0) hits++;
            return {
                count,
                hits,
                gpuMs: ms.length ? ms[0] : null,
                wallMs,
                hit(index) {
                    const base = index * stride;
                    const t = floats[base + w.t];
                    if (!(t >= 0)) return null;
                    return {
                        t,
                        instance: words[base + w.instance],
                        triangle: words[base + w.triangle],
                        frontFace: words[base + w.frontFace] !== 0,
                        u: floats[base + w.barycentrics],
                        v: floats[base + w.barycentrics + 1],
                        normal: [floats[base + w.normal], floats[base + w.normal + 1], floats[base + w.normal + 2]],
                    };
                },
            };
        } finally {
            this.busy = false;
        }
    }

    destroy(): void {
        this.rays?.destroy();
        this.results?.destroy();
        this.readback?.destroy();
        this.timer?.destroy();
        this.kernel.destroy();
    }
}

const raycaster = new Raycaster();

/** The camera ray through `ndc` (-1..1), packed for {@link GpuRays.traceClosest}. */
export function cameraRay(camera: Camera, ndc: Vector2, out = new Float32Array(RAY_FLOATS), at = 0): Float32Array {
    raycaster.setFromCamera(ndc, camera);
    const { origin, direction } = raycaster.ray;
    out.set([origin.x, origin.y, origin.z, 0, direction.x, direction.y, direction.z, 1e30], at * RAY_FLOATS);
    return out;
}

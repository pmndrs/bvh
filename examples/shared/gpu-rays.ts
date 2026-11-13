/**
 * Rays traced on the GPU over a scene with the package's own batch kernel
 * (`TraceBatchKernel`): one dispatch for any number of rays, the hits read
 * back. The examples use it for picking (one ray) and for batch raycasts
 * (thousands), timed with timestamp queries where the device has them.
 *
 * Engine-free: the scene is a `SceneTrees`, or any {@link TraceSource} (an
 * uploaded scene, `traceSourceOf`). The three pages make their camera rays
 * with `cameraRay` (app.ts), the vanilla ones with `OrbitCamera.packRay`.
 */

import {
    assembleTraceScene,
    HIT_RECORD_BYTES,
    HIT_RECORD_WORD,
    RAY_BYTES,
    RAY_FLOATS,
    TraceBatchKernel,
    type GpuLayoutMeshBvh,
    type MeshGeometry,
    type NodeFormat,
    type PositionLayout,
    type TraceSceneBuffers,
    type TraceSceneData,
    type TraceSceneInstance,
    type TriangleLeaves,
} from '@pmndrs/bvh';
import { GpuTimer } from './gpu-timer.ts';

/** What rays are traced over: a `SceneTrees`, or an uploaded scene through {@link traceSourceOf}. */
export interface TraceSource {
    readonly device: GPUDevice;
    /** Read at every trace (a `SceneTrees`' buffers can be replaced as they grow). */
    readonly traceBuffers: TraceSceneBuffers;
    readonly triangleLeaves: TriangleLeaves;
    readonly positionLayout: PositionLayout;
    readonly nodeFormat: NodeFormat;
}

/** An uploaded scene (`uploadTraceScene`) as a {@link TraceSource}, its formats read from the buffers. */
export function traceSourceOf(device: GPUDevice, buffers: TraceSceneBuffers): TraceSource {
    return {
        device,
        traceBuffers: buffers,
        triangleLeaves: buffers.triangleLeaves ?? 'records',
        positionLayout: buffers.positionLayout ?? 'vec4',
        nodeFormat: buffers.nodeFormat ?? 'f32',
    };
}

/**
 * `assembleTraceScene` with the scene's formats taken from the trees
 * themselves: a quantized tree makes a quantized scene, an indexed tree an
 * indexed one (with `geometries`, parallel to `layouts`, for its positions).
 * `assembleTraceScene` alone assumes `f32` records and refuses other trees,
 * so a page that builds with the builders' defaults works whatever they are.
 */
export function assembleLayouts(layouts: readonly GpuLayoutMeshBvh[], instances: readonly TraceSceneInstance[], geometries: readonly MeshGeometry[]): TraceSceneData {
    const quantized = layouts.some((layout) => layout.nodeFormat === 'quantized');
    const indexed = layouts.some((layout) => layout.triangleLeaves === 'indexed');
    return assembleTraceScene(layouts, instances, {}, { ...(quantized ? { nodeFormat: 'quantized' } : {}), ...(indexed ? { triangleLeaves: 'indexed', geometries } : {}) });
}

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
    private queued = 0;
    private destroyed = false;
    private queue: Promise<unknown> = Promise.resolve();
    private readonly timer: GpuTimer | null;

    private constructor(
        private readonly device: GPUDevice,
        private readonly trees: TraceSource,
        private readonly kernel: TraceBatchKernel,
    ) {
        this.timer = GpuTimer.create(device, 1);
    }

    /** A kernel compiled for the source's triangle and node formats. */
    static async create(trees: TraceSource): Promise<GpuRays> {
        const kernel = await TraceBatchKernel.create(trees.device, { label: 'example rays', triangleLeaves: trees.triangleLeaves, positionLayout: trees.positionLayout, nodeFormat: trees.nodeFormat });
        return new GpuRays(trees.device, trees, kernel);
    }

    /** True while a trace is in flight (the examples skip a frame's trace then). */
    get pending(): boolean {
        return this.queued > 0;
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
     * for their closest hits, over the source as it is now (a `SceneTrees`: as of its last `update`).
     */
    traceClosest(packed: Float32Array, mask?: number): Promise<GpuTraceResult> {
        // One trace at a time: they share the buffers and the readback.
        this.queued++;
        const next = this.queue.then(() => this.traceNow(packed, mask)).finally(() => this.queued--);
        this.queue = next.catch(() => undefined);
        return next;
    }

    private async traceNow(packed: Float32Array, mask?: number): Promise<GpuTraceResult> {
        if (this.destroyed) throw new Error('GpuRays: destroyed');
        const count = packed.length / RAY_FLOATS;
        this.ensure(count);
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
    }

    /** Destroys the buffers once the traces already asked for have finished (a page swapping scenes calls this on the old one). */
    async retire(): Promise<void> {
        await this.queue;
        this.destroy();
    }

    destroy(): void {
        this.destroyed = true;
        this.rays?.destroy();
        this.results?.destroy();
        this.readback?.destroy();
        this.timer?.destroy();
        this.kernel.destroy();
    }
}

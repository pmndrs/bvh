/**
 * One mesh's tree from a chosen builder, timed: tinybvh (WebAssembly, on this
 * thread; CPU wall time), GPU H-PLOC or GPU LBVH (on the page's device; CPU
 * wall time from encode to the layout read back, and the batch's passes in
 * one timestamp pair where the device has `timestamp-query`).
 *
 * Without `subgroups` (both GPU builders need it) tinybvh builds instead, and
 * `fallbackReason` says why, as `createMeshLayoutBuilder`'s fallback does. Engine-free: the inspector (three) and the vanilla pages share it.
 */

import {
    createTinyBvhBuilder,
    GpuHplocBuilder,
    GpuLbvhBuilder,
    indexedFromRecords,
    type GpuLayoutMeshBvh,
    type GpuLayoutNodeFormat,
    type GpuLayoutTriangleLeaves,
    type GpuTreeBuffers,
    type MeshInput,
    type MeshLayoutBuilderKind,
    type TinyBvhBuilder,
} from '@pmndrs/bvh';
import { GpuTimer } from './gpu-timer.ts';

export type BuilderKind = MeshLayoutBuilderKind;
export const BUILDER_KINDS: readonly BuilderKind[] = ['tinybvh', 'gpu-hploc', 'gpu-lbvh'];

/** The builders' common options; left out, each builder's own default. */
export interface TimedBuildOptions {
    readonly maxLeafPrimitives?: number;
    readonly triangleLeaves?: GpuLayoutTriangleLeaves;
    readonly nodeFormat?: GpuLayoutNodeFormat;
}

export interface TimedBuild {
    readonly layout: GpuLayoutMeshBvh;
    /** Who built it: the builder asked for, or tinybvh when the device can't run it. */
    readonly builtBy: BuilderKind;
    readonly fallbackReason: string | null;
    /** CPU wall milliseconds, from the call to the tree being usable (tinybvh: on the CPU; GPU builders: on the GPU, before the readback). */
    readonly cpuMs: number;
    /** GPU milliseconds (timestamps), null without `timestamp-query`, undefined for tinybvh. */
    readonly gpuMs: number | null | undefined;
}

export class MeshBuilds {
    private readonly gpuBuilders = new Map<BuilderKind, GpuHplocBuilder | GpuLbvhBuilder>();
    private readonly timer: GpuTimer | null;
    private queue: Promise<unknown> = Promise.resolve();

    private constructor(
        private readonly device: GPUDevice,
        readonly tinybvh: TinyBvhBuilder,
    ) {
        this.timer = GpuTimer.create(device, 1);
    }

    static async create(device: GPUDevice): Promise<MeshBuilds> {
        return new MeshBuilds(device, await createTinyBvhBuilder());
    }

    /**
     * Builds `input` with `kind`. `warmUp` (default true) builds once untimed
     * first, so the timed build doesn't pay for pipelines and first buffers.
     */
    build(kind: BuilderKind, input: MeshInput, options: TimedBuildOptions = {}, warmUp = true): Promise<TimedBuild> {
        // One build at a time: they share the timer, and overlapping builds would time each other.
        const next = this.queue.then(() => this.buildNow(kind, input, options, warmUp));
        this.queue = next.catch(() => undefined);
        return next;
    }

    private async buildNow(kind: BuilderKind, input: MeshInput, options: TimedBuildOptions, warmUp: boolean): Promise<TimedBuild> {
        if (kind !== 'tinybvh') {
            // Both GPU builders need subgroups; without them tinybvh builds, as createMeshLayoutBuilder's fallback does.
            if (GpuLbvhBuilder.isSupported(this.device)) {
                if (warmUp) await this.buildOnGpu(kind, input, options);
                return { ...(await this.buildOnGpu(kind, input, options)), builtBy: kind, fallbackReason: null };
            }
            return { ...this.buildOnCpu(input, options, false), fallbackReason: 'the device has no "subgroups" feature' };
        }
        return { ...this.buildOnCpu(input, options, warmUp), fallbackReason: null };
    }

    private buildOnCpu(input: MeshInput, options: TimedBuildOptions, warmUp: boolean): Omit<TimedBuild, 'fallbackReason'> {
        if (warmUp) this.tinybvh.buildLayout(input, options);
        const start = performance.now();
        const layout = this.tinybvh.buildLayout(input, options).layout;
        return { layout, builtBy: 'tinybvh', cpuMs: performance.now() - start, gpuMs: undefined };
    }

    private async gpuBuilder(kind: 'gpu-hploc' | 'gpu-lbvh'): Promise<GpuHplocBuilder | GpuLbvhBuilder> {
        let builder = this.gpuBuilders.get(kind);
        if (!builder) {
            builder = kind === 'gpu-hploc' ? new GpuHplocBuilder(this.device, { label: 'example' }) : new GpuLbvhBuilder(this.device, { label: 'example' });
            await builder.prepare();
            this.gpuBuilders.set(kind, builder);
        }
        return builder;
    }

    /** One GPU build, recorded and timed: the batch's passes in one timestamp pair. */
    private async buildOnGpu(kind: 'gpu-hploc' | 'gpu-lbvh', input: MeshInput, options: TimedBuildOptions): Promise<Omit<TimedBuild, 'builtBy' | 'fallbackReason'>> {
        const builder = await this.gpuBuilder(kind);
        // The LBVH writes records only; indexed trees are converted after the readback.
        const lbvhIndexed = kind === 'gpu-lbvh' && options.triangleLeaves === 'indexed';
        const start = performance.now();
        const batch = builder.createBatch([input], lbvhIndexed ? { ...options, triangleLeaves: 'records' } : options);
        let trees: GpuTreeBuffers | null = null;
        const timer = this.timer && !this.timer.pending ? this.timer : null;
        let timed = false;
        try {
            const encoder = this.device.createCommandEncoder({ label: 'example build' });
            batch.encode(encoder, timer?.pass());
            timer?.resolveInto(encoder);
            this.device.queue.submit([encoder.finish()]);
            trees = await batch.finish();
            // The tree is usable on the GPU here; the readback below is for the CPU copy the page inspects.
            const cpuMs = performance.now() - start;
            timed = true;
            const gpuMs = timer ? (await timer.result())[0] : null;
            const problem = trees.meshes[0].problem;
            if (problem) throw new Error(`${kind} refused the mesh: ${problem}`);
            const [read] = await builder.readLayouts(trees);
            const layout = lbvhIndexed ? indexedFromRecords(read!, input) : read!;
            return { layout, cpuMs, gpuMs };
        } finally {
            // A build that failed before its timestamps were read leaves the timer as it was before it.
            if (!timed) timer?.reset();
            trees?.destroy();
            batch.destroy();
        }
    }
}

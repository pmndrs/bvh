/**
 * One mesh's tree from a chosen builder, timed: tinybvh (WebAssembly, on this
 * thread; CPU wall time), GPU H-PLOC or GPU LBVH through
 * `createMeshLayoutBuilder({ prefer })` (CPU wall time from the call to the
 * tree on the GPU, and the batch's pass from timestamps where the device has
 * `timestamp-query`, both from `onGpuBatch`).
 *
 * Without `subgroups` (both GPU builders need it) tinybvh builds instead, and
 * `fallbackReason` says why: the chooser's fallback. Engine-free: the inspector (three) and the vanilla pages share it.
 */

import {
    assembleTraceScene,
    createMeshLayoutBuilder,
    createTinyBvhBuilder,
    type GpuLayoutMeshBvh,
    type GpuLayoutNodeFormat,
    type GpuLayoutTriangleLeaves,
    type MeshGeometry,
    type MeshInput,
    type MeshLayoutBuilder,
    type MeshLayoutBuilderKind,
    type TinyBvhBuilder,
    type TraceSceneData,
    type TraceSceneInstance,
} from '@pmndrs/bvh';

export type BuilderKind = MeshLayoutBuilderKind;

/**
 * `assembleTraceScene` for trees of any builder's defaults: an indexed tree
 * makes an indexed scene, which needs each mesh's positions (`geometries`,
 * parallel to `layouts`); records trees take no geometries.
 */
export function assembleLayouts(layouts: readonly GpuLayoutMeshBvh[], instances: readonly TraceSceneInstance[], geometries: readonly MeshGeometry[]): TraceSceneData {
    const indexed = layouts.some((layout) => layout.triangleLeaves === 'indexed');
    return assembleTraceScene(layouts, instances, {}, indexed ? { geometries } : {});
}
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
    /** One chooser per GPU builder, its pipelines compiled once (`createMeshLayoutBuilder`). */
    private readonly choosers = new Map<BuilderKind, Promise<MeshLayoutBuilder>>();
    private queue: Promise<unknown> = Promise.resolve();

    private constructor(
        private readonly device: GPUDevice,
        readonly tinybvh: TinyBvhBuilder,
    ) {}

    static async create(device: GPUDevice): Promise<MeshBuilds> {
        return new MeshBuilds(device, await createTinyBvhBuilder());
    }

    /**
     * Builds `input` with `kind`. `warmUp` (default true) builds once untimed
     * first, so the timed build doesn't pay for first buffers.
     */
    build(kind: BuilderKind, input: MeshInput, options: TimedBuildOptions = {}, warmUp = true): Promise<TimedBuild> {
        // One build at a time: overlapping builds would time each other.
        const next = this.queue.then(() => this.buildNow(kind, input, options, warmUp));
        this.queue = next.catch(() => undefined);
        return next;
    }

    private async buildNow(kind: BuilderKind, input: MeshInput, options: TimedBuildOptions, warmUp: boolean): Promise<TimedBuild> {
        if (kind === 'tinybvh') return { ...this.buildOnCpu(input, options, warmUp), fallbackReason: null };
        let chooser = this.choosers.get(kind);
        if (!chooser) {
            chooser = createMeshLayoutBuilder(this.device, { prefer: kind, label: 'example' });
            this.choosers.set(kind, chooser);
        }
        const builder = await chooser;
        // Both GPU builders need subgroups; without them the chooser is tinybvh, and says why.
        if (builder.kind === 'tinybvh') return { ...this.buildOnCpu(input, options, false), fallbackReason: builder.fallbackReason };
        if (warmUp) await builder.buildLayouts([input], options);
        const start = performance.now();
        let usable: { cpuMs: number; gpuMs: number | null } | null = null;
        // onGpuBatch comes once the tree is usable on the GPU, before the readback of the CPU copy the page inspects.
        const [result] = await builder.buildLayouts([input], { ...options, onGpuBatch: (batch) => (usable = { cpuMs: performance.now() - start, gpuMs: batch.gpuMs }) });
        const timing = usable as { cpuMs: number; gpuMs: number | null } | null;
        // A mesh the GPU builder refused is tinybvh's, and the result says why.
        if (result.builtBy === 'tinybvh' || !timing) return { layout: result.layout, builtBy: 'tinybvh', fallbackReason: result.fallbackReason, cpuMs: performance.now() - start, gpuMs: undefined };
        return { layout: result.layout, builtBy: result.builtBy, fallbackReason: null, cpuMs: timing.cpuMs, gpuMs: timing.gpuMs };
    }

    private buildOnCpu(input: MeshInput, options: TimedBuildOptions, warmUp: boolean): Omit<TimedBuild, 'fallbackReason'> {
        if (warmUp) this.tinybvh.buildLayout(input, options);
        const start = performance.now();
        const layout = this.tinybvh.buildLayout(input, options).layout;
        return { layout, builtBy: 'tinybvh', cpuMs: performance.now() - start, gpuMs: undefined };
    }
}

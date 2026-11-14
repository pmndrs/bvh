/**
 * One entry point for mesh trees in the tracer's layout, whichever builder
 * makes them (the Builder contract, docs/internals/design/bvh-framework.md): a GPU
 * builder when the device has subgroups (the H-PLOC, B2, by default; or the
 * LBVH, B1), tinybvh in WebAssembly (B0) otherwise. That is the BVH's one
 * declared fallback (DECISIONS 2026-11-04: subgroups first, one fallback per
 * feature).
 *
 * The fallback also covers single meshes: a mesh too big for the device's
 * buffers (or for the builder), or a GPU build that was refused (gpu-lbvh.ts
 * and gpu-hploc.ts, "Checks"), is built by tinybvh instead, and the result says so.
 *
 * `maxBatchBytes` (optional, default none) caps each GPU batch's bytes (its
 * inputs, scratch and outputs) for a lower build peak; the batches are built
 * one after another, and H-PLOC's share one scratch pool sized from the
 * largest (docs/internals/design/bvh-builders.md, "Build peak memory"). A mesh bigger
 * than the cap builds alone.
 *
 * **Setup and timing.** The GPU builder's pipelines are compiled once, by
 * the factory (or, with `prepare: false`, by {@link MeshLayoutBuilder.prepare}
 * or the first build), so a build's time is the build's. `onGpuBatch` reports
 * each GPU batch's time: its compute pass from timestamp queries where the
 * device has `timestamp-query`, and the CPU wall time from its submit to the
 * trees on the GPU (before the layouts are read back).
 *
 * tinybvh runs on the calling thread here. ddgi-core runs it in a worker
 * (`mesh-bvh-async.ts`); this package stays free of workers and engines.
 */

import { GpuHplocBuilder, planGpuHplocBatches } from './gpu-hploc.ts';
import type { GpuLayoutMeshBvh } from './gpu-layout.ts';
import { GpuLbvhBuilder, planGpuLbvhBatches } from './gpu-lbvh.ts';
import type { GpuBuildOptions, GpuMeshGeometry, GpuTreeBuffers, MeshInput } from './mesh-input.ts';
import { createTinyBvhBuilder, type TinyBvhBuilder } from './tinybvh.ts';
import { PassTimestamps } from './gpu-timestamps.ts';

export type MeshLayoutBuilderKind = 'gpu-hploc' | 'gpu-lbvh' | 'tinybvh';

export interface MeshLayoutResult {
    readonly layout: GpuLayoutMeshBvh;
    readonly builtBy: MeshLayoutBuilderKind;
    /** Why tinybvh built this mesh although the GPU builder is in use, or null. */
    readonly fallbackReason: string | null;
}

/** One GPU batch of a {@link MeshLayoutBuilder.buildLayouts} call, for {@link MeshLayoutBuildOptions.onGpuBatch}. */
export interface MeshLayoutGpuBatch {
    /** The batch's meshes, as indices into the call's `geometries`. */
    readonly meshes: readonly number[];
    /** GPU milliseconds of the batch's compute pass (timestamps), or null without `timestamp-query`. */
    readonly gpuMs: number | null;
    /** CPU wall milliseconds from the batch's submit to its trees on the GPU (before the layouts' readback). */
    readonly wallMs: number;
}

/** Options of {@link MeshLayoutBuilder.buildLayouts}: the builders' common ones, and timing. */
export interface MeshLayoutBuildOptions extends GpuBuildOptions {
    /**
     * Called once per GPU batch with its times, before the call resolves. A
     * GPU builder only (tinybvh is never called, the fallback included); the
     * batch's pass is timed with timestamp queries only when this is given.
     */
    readonly onGpuBatch?: (batch: MeshLayoutGpuBatch) => void;
}

export interface MeshLayoutBuilder {
    /** The builder meshes go to first. */
    readonly kind: MeshLayoutBuilderKind;
    /**
     * Why the GPU builder is not in use, or null when it is. Its meaning is
     * unchanged: it is set whenever `kind` is `'tinybvh'`, a real fallback
     * (`'no GPU device'`, `'the device has no "subgroups" feature'`) and a
     * choice alike (`'the CPU builder was asked for'`, with
     * `prefer: 'tinybvh'`). {@link fallback} tells the two apart.
     */
    readonly fallbackReason: string | null;
    /**
     * True only for a real fallback: a GPU builder was wanted (no `prefer`, or
     * a GPU one) and the device can't run it, so tinybvh builds. False when a
     * GPU builder is in use, and when tinybvh was asked for
     * (`prefer: 'tinybvh'`), which an overlay should not show as a fallback.
     */
    readonly fallback: boolean;
    /** True once the builder's pipelines are compiled (always for tinybvh). */
    readonly ready: boolean;
    /** Compiles the GPU builder's pipelines, once (the factory does unless `prepare: false`; a build does if nothing did). */
    prepare(): Promise<void>;
    /** Builds every mesh's tree. With the GPU builder, meshes go in as few batches as the device's buffers allow. */
    buildLayouts(geometries: readonly MeshInput[], options?: MeshLayoutBuildOptions): Promise<MeshLayoutResult[]>;
}

/** Options of {@link createMeshLayoutBuilder}. */
export interface MeshLayoutBuilderOptions {
    /** The GPU builder (`'gpu-hploc'`, the default, or `'gpu-lbvh'`), or `'tinybvh'` to force the CPU builder (for comparisons). */
    readonly prefer?: MeshLayoutBuilderKind;
    readonly label?: string;
    /** Cap each GPU batch's bytes (inputs, scratch, outputs) for a lower build peak; default none. */
    readonly maxBatchBytes?: number;
    /** Default true: the factory compiles the GPU builder's pipelines before it resolves. False: {@link MeshLayoutBuilder.prepare} (or the first build) does, so an app can time or schedule it apart. */
    readonly prepare?: boolean;
}

/**
 * A GPU builder if `device` has subgroups, tinybvh otherwise (or with no
 * device). `prefer` picks the GPU builder (`'gpu-hploc'`, the default, or
 * `'gpu-lbvh'`), or forces the CPU builder (`'tinybvh'`), for comparisons.
 * The options of `buildLayouts` are the builders' common ones; H-PLOC's own
 * (its Morton codes and tie rule) keep their defaults here.
 */
export async function createMeshLayoutBuilder(device: GPUDevice | null, options: MeshLayoutBuilderOptions = {}): Promise<MeshLayoutBuilder> {
    let fallbackReason: string | null = null;
    if (options.prefer === 'tinybvh') fallbackReason = 'the CPU builder was asked for';
    else if (!device) fallbackReason = 'no GPU device';
    // Both GPU builders need the same feature.
    else if (!GpuLbvhBuilder.isSupported(device)) fallbackReason = 'the device has no "subgroups" feature';

    let tinyBvh: Promise<TinyBvhBuilder> | null = null;
    const buildOnCpu = async (geometry: MeshInput, gpuOptions: GpuBuildOptions): Promise<GpuLayoutMeshBvh> => {
        tinyBvh ??= createTinyBvhBuilder();
        return (await tinyBvh).buildLayout(geometry, { maxLeafPrimitives: gpuOptions.maxLeafPrimitives, triangleLeaves: gpuOptions.triangleLeaves, nodeFormat: gpuOptions.nodeFormat }).layout;
    };

    if (fallbackReason !== null || !device) {
        return {
            kind: 'tinybvh',
            fallbackReason,
            fallback: options.prefer !== 'tinybvh',
            ready: true,
            async prepare() {},
            async buildLayouts(geometries, buildOptions = {}) {
                const { onGpuBatch: _, ...gpuOptions } = buildOptions;
                const results: MeshLayoutResult[] = [];
                for (const geometry of geometries) results.push({ layout: await buildOnCpu(geometry, gpuOptions), builtBy: 'tinybvh', fallbackReason: null });
                return results;
            },
        };
    }

    const kind = options.prefer === 'gpu-lbvh' ? 'gpu-lbvh' : 'gpu-hploc';
    const gpu = kind === 'gpu-lbvh' ? new GpuLbvhBuilder(device, { label: options.label }) : new GpuHplocBuilder(device, { label: options.label });
    const planOptions = { maxBatchBytes: options.maxBatchBytes };
    const plan = (geometries: readonly GpuMeshGeometry[]) => (kind === 'gpu-lbvh' ? planGpuLbvhBatches : planGpuHplocBatches)(device, geometries, planOptions);
    let preparing: Promise<void> | null = null;
    const prepare = (): Promise<void> => (preparing ??= gpu.prepare());
    if (options.prepare !== false) await prepare();
    // One timer for every timed batch: a builder's batches are built one after another.
    let timer: PassTimestamps | null | undefined;
    return {
        kind,
        fallbackReason: null,
        fallback: false,
        get ready() {
            return gpu.ready;
        },
        prepare,
        async buildLayouts(geometries, buildOptions = {}) {
            await prepare();
            const { onGpuBatch, ...gpuOptions } = buildOptions;
            if (onGpuBatch) timer ??= PassTimestamps.create(device, 1, options.label ?? 'mesh layouts');
            const results: (MeshLayoutResult | null)[] = geometries.map(() => null);
            const batches = plan(geometries);
            for (const index of batches.tooBig) {
                const reason = `the mesh is too big for this builder or this device's GPU buffers (${Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize)} bytes)`;
                results[index] = { layout: await buildOnCpu(geometries[index], gpuOptions), builtBy: 'tinybvh', fallbackReason: reason };
            }
            // Capped H-PLOC batches share one scratch pool, allocated at the largest batch's size and freed after the last.
            const pool =
                gpu instanceof GpuHplocBuilder && options.maxBatchBytes !== undefined && batches.batches.length > 1
                    ? gpu.createScratchPool(batches.batches.map((batch) => batch.map((index) => geometries[index])))
                    : null;
            // One batch: its upload, one submit (timed when asked), its trees.
            const buildBatch = async (indices: readonly number[]): Promise<GpuTreeBuffers> => {
                const batch = indices.map((index) => geometries[index]);
                const built = gpu instanceof GpuHplocBuilder ? gpu.createBatch(batch, gpuOptions, pool ?? undefined) : gpu.createBatch(batch, gpuOptions);
                const timed = onGpuBatch && timer && !timer.busy ? timer : null;
                try {
                    const encoder = device.createCommandEncoder({ label: options.label });
                    built.encode(encoder, timed?.writes());
                    timed?.resolveInto(encoder);
                    const submitted = performance.now();
                    device.queue.submit([encoder.finish()]);
                    const trees = await built.finish();
                    const wallMs = performance.now() - submitted;
                    const ms = timed ? await timed.read().catch((): number[] => []) : [];
                    onGpuBatch?.({ meshes: indices, gpuMs: ms.length > 0 ? ms[0] : null, wallMs });
                    return trees;
                } catch (error) {
                    timed?.reset();
                    throw error;
                } finally {
                    built.destroy();
                }
            };
            try {
                for (const batch of batches.batches) {
                    await buildBatchLayouts(batch, await buildBatch(batch));
                }
            } finally {
                pool?.destroy();
            }
            return results as MeshLayoutResult[];

            async function buildBatchLayouts(batch: readonly number[], trees: GpuTreeBuffers): Promise<void> {
                try {
                    const layouts = await gpu.readLayouts(trees);
                    for (const [at, layout] of layouts.entries()) {
                        const index = batch[at];
                        const problem = trees.meshes[at].problem;
                        if (layout && !problem) results[index] = { layout, builtBy: kind, fallbackReason: null };
                        else results[index] = { layout: await buildOnCpu(geometries[index], gpuOptions), builtBy: 'tinybvh', fallbackReason: problem };
                    }
                } finally {
                    // The layouts are CPU copies; the GPU trees are not needed after the readback.
                    trees.destroy();
                }
            }
        },
    };
}

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
 * tinybvh runs on the calling thread here. ddgi-core runs it in a worker
 * (`mesh-bvh-async.ts`); this package stays free of workers and engines.
 */

import { GpuHplocBuilder, planGpuHplocBatches } from './gpu-hploc.ts';
import type { GpuLayoutMeshBvh } from './gpu-layout.ts';
import { GpuLbvhBuilder, planGpuLbvhBatches } from './gpu-lbvh.ts';
import type { GpuBuildOptions, GpuMeshGeometry, GpuTreeBuffers, MeshInput } from './mesh-input.ts';
import { createTinyBvhBuilder, type TinyBvhBuilder } from './tinybvh.ts';
import { indexedFromRecords } from './indexed-leaves.ts';

export type MeshLayoutBuilderKind = 'gpu-hploc' | 'gpu-lbvh' | 'tinybvh';

export interface MeshLayoutResult {
    readonly layout: GpuLayoutMeshBvh;
    readonly builtBy: MeshLayoutBuilderKind;
    /** Why tinybvh built this mesh although the GPU builder is in use, or null. */
    readonly fallbackReason: string | null;
}

export interface MeshLayoutBuilder {
    /** The builder meshes go to first. */
    readonly kind: MeshLayoutBuilderKind;
    /** Why the GPU builder is not in use, or null when it is. */
    readonly fallbackReason: string | null;
    /** Builds every mesh's tree. With the GPU builder, meshes go in as few batches as the device's buffers allow. */
    buildLayouts(geometries: readonly MeshInput[], options?: GpuBuildOptions): Promise<MeshLayoutResult[]>;
}

/**
 * A GPU builder if `device` has subgroups, tinybvh otherwise (or with no
 * device). `prefer` picks the GPU builder (`'gpu-hploc'`, the default, or
 * `'gpu-lbvh'`), or forces the CPU builder (`'tinybvh'`), for comparisons.
 * The options of `buildLayouts` are the builders' common ones; H-PLOC's own
 * (its Morton codes and tie rule) keep their defaults here.
 */
export async function createMeshLayoutBuilder(
    device: GPUDevice | null,
    options: { prefer?: MeshLayoutBuilderKind; label?: string; maxBatchBytes?: number } = {},
): Promise<MeshLayoutBuilder> {
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
            async buildLayouts(geometries, gpuOptions = {}) {
                const results: MeshLayoutResult[] = [];
                for (const geometry of geometries) results.push({ layout: await buildOnCpu(geometry, gpuOptions), builtBy: 'tinybvh', fallbackReason: null });
                return results;
            },
        };
    }

    const kind = options.prefer === 'gpu-lbvh' ? 'gpu-lbvh' : 'gpu-hploc';
    const gpu = kind === 'gpu-lbvh' ? new GpuLbvhBuilder(device, { label: options.label }) : new GpuHplocBuilder(device, { label: options.label });
    const planOptions = { maxBatchBytes: options.maxBatchBytes };
    // The LBVH writes records only: an indexed build converts them after the readback (indexedFromRecords).
    const gpuBuildOptions = (gpuOptions: GpuBuildOptions): GpuBuildOptions => (kind === 'gpu-lbvh' ? { ...gpuOptions, triangleLeaves: 'records' } : gpuOptions);
    const plan = (geometries: readonly GpuMeshGeometry[]) => (kind === 'gpu-lbvh' ? planGpuLbvhBatches : planGpuHplocBatches)(device, geometries, planOptions);
    await gpu.prepare();
    return {
        kind,
        fallbackReason: null,
        async buildLayouts(geometries, gpuOptions = {}) {
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
            const buildBatch = async (batch: readonly GpuMeshGeometry[]): Promise<GpuTreeBuffers> => {
                if (!pool || !(gpu instanceof GpuHplocBuilder)) return gpu.buildMany(batch, gpuBuildOptions(gpuOptions));
                const built = gpu.createBatch(batch, gpuBuildOptions(gpuOptions), pool);
                try {
                    const encoder = device.createCommandEncoder({ label: options.label });
                    built.encode(encoder);
                    device.queue.submit([encoder.finish()]);
                    return await built.finish();
                } finally {
                    built.destroy();
                }
            };
            try {
                for (const batch of batches.batches) {
                    await buildBatchLayouts(batch, await buildBatch(batch.map((index) => geometries[index])));
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
                        const converted = layout && gpuOptions.triangleLeaves === 'indexed' ? indexedFromRecords(layout, geometries[index]) : layout;
                        if (converted && !problem) results[index] = { layout: converted, builtBy: kind, fallbackReason: null };
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

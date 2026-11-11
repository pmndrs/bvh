/**
 * The loaded section of a timing: one dispatch submitted back to back for
 * about a given time, each pass timed on its own by timestamps, so the GPU
 * runs at its loaded clock instead of ramping down between waited-for
 * dispatches. Ported from the radiance-cache team's trace bench
 * (`timePassesBackToBack`, `timeLoaded`).
 */

import { maximum, median, megaRaysPerSecond } from './statistics.ts';

/** Bounds of the loaded section's dispatch count. */
export const LOADED_DISPATCHES = { min: 10, max: 4_000 } as const;

export interface LoadedTiming {
    medianMs: number;
    minMs: number;
    maxMs: number;
    dispatches: number;
    seconds: number;
    mraysPerSecond: number;
}

/**
 * Submits `count` passes recorded by `encode`, a few per command buffer, and
 * returns each pass's ms from timestamps. Without `timestamp-query` it returns
 * one value, the wall time over the count.
 */
export async function timePassesBackToBack(
    device: GPUDevice,
    encode: (encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites) => void,
    count: number,
): Promise<number[]> {
    if (!device.features.has('timestamp-query')) {
        const start = performance.now();
        for (let run = 0; run < count; run++) {
            const encoder = device.createCommandEncoder({ label: 'loaded pass' });
            encode(encoder);
            device.queue.submit([encoder.finish()]);
        }
        await device.queue.onSubmittedWorkDone();
        return [(performance.now() - start) / count];
    }
    const perSubmit = 8;
    const reads: Promise<number[]>[] = [];
    for (let first = 0; first < count; first += perSubmit) {
        const passes = Math.min(perSubmit, count - first);
        const querySet = device.createQuerySet({ type: 'timestamp', count: passes * 2 });
        const resolve = device.createBuffer({ size: passes * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
        const readback = device.createBuffer({ size: passes * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const encoder = device.createCommandEncoder({ label: 'loaded passes' });
        for (let pass = 0; pass < passes; pass++) encode(encoder, { querySet, beginningOfPassWriteIndex: pass * 2, endOfPassWriteIndex: pass * 2 + 1 });
        encoder.resolveQuerySet(querySet, 0, passes * 2, resolve, 0);
        encoder.copyBufferToBuffer(resolve, 0, readback, 0, passes * 16);
        device.queue.submit([encoder.finish()]);
        reads.push(
            readback.mapAsync(GPUMapMode.READ).then(() => {
                const stamps = new BigUint64Array(readback.getMappedRange().slice(0));
                readback.destroy();
                resolve.destroy();
                querySet.destroy();
                return Array.from({ length: passes }, (_, pass) => Number(stamps[pass * 2 + 1] - stamps[pass * 2]) / 1e6);
            }),
        );
    }
    return (await Promise.all(reads)).flat();
}

/** The loaded section: about `loadedMs` of back-to-back passes, sized from one pass's median time. */
export async function timeLoaded(
    device: GPUDevice,
    encode: (encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites) => void,
    rays: number,
    singleMedianMs: number,
    loadedMs: number,
): Promise<LoadedTiming> {
    const dispatches = Math.min(LOADED_DISPATCHES.max, Math.max(LOADED_DISPATCHES.min, Math.round(loadedMs / Math.max(singleMedianMs, 1e-3))));
    const start = performance.now();
    const samplesMs = await timePassesBackToBack(device, encode, dispatches);
    const medianMs = median(samplesMs);
    return {
        medianMs,
        minMs: Math.min(...samplesMs),
        maxMs: maximum(samplesMs),
        dispatches,
        seconds: (performance.now() - start) / 1000,
        mraysPerSecond: megaRaysPerSecond(rays, medianMs),
    };
}

/**
 * Times one compute dispatch on the GPU.
 *
 * Preferred: timestamp queries around the dispatch's own pass (a failed
 * query throws instead of silently leaving a dispatch untimed). Without the `timestamp-query`
 * feature the fallback is wall time from submit to "queue drained", labelled
 * as such in the results.
 *
 * Every dispatch is timed alone: it is submitted, waited for, and read back
 * before the next starts, so no two timed passes overlap on the GPU.
 */

import { median } from './statistics.ts';
import type { TimingSource, TimingSummary } from './results.ts';

export interface DispatchTimer {
    readonly source: TimingSource;
    /**
     * Records the dispatch with `encode` (which must pass `writes` to the
     * compute pass it begins), runs it, and returns its GPU milliseconds.
     */
    time(encode: (encoder: GPUCommandEncoder, writes: GPUComputePassTimestampWrites | undefined) => void): Promise<number>;
    dispose(): void;
}

const PASS_NAME = 'trace-bench/dispatch';

export async function createDispatchTimer(device: GPUDevice): Promise<DispatchTimer> {
    if (!device.features.has('timestamp-query')) return createWallClockTimer(device);

    const querySet = device.createQuerySet({ type: 'timestamp', count: 2, label: 'trace-bench timestamps' });
    const resolveBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const readBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const timestampWrites: GPUComputePassTimestampWrites = { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };

    return {
        source: 'timestamp-query',
        async time(encode) {
            const encoder = device.createCommandEncoder({ label: 'trace-bench dispatch' });
            encode(encoder, timestampWrites);
            encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
            encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, 16);
            device.queue.submit([encoder.finish()]);
            await readBuffer.mapAsync(GPUMapMode.READ);
            const [begin, end] = new BigUint64Array(readBuffer.getMappedRange().slice(0));
            readBuffer.unmap();
            return Number(end - begin) / 1e6;
        },
        dispose() {
            querySet.destroy();
            resolveBuffer.destroy();
            readBuffer.destroy();
        },
    };
}

function createWallClockTimer(device: GPUDevice): DispatchTimer {
    return {
        source: 'wall-clock',
        async time(encode) {
            const encoder = device.createCommandEncoder({ label: 'trace-bench dispatch' });
            encode(encoder, undefined);
            const start = performance.now();
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
            return performance.now() - start;
        },
        dispose: () => {},
    };
}

/** Runs `timeOnce` for the warm-ups (discarded) and then the timed dispatches. */
export async function measureDispatches(
    timeOnce: () => Promise<number>,
    warmups: number,
    timed: number,
): Promise<TimingSummary> {
    for (let i = 0; i < warmups; i++) await timeOnce();
    const samplesMs: number[] = [];
    for (let i = 0; i < timed; i++) {
        const milliseconds = await timeOnce();
        if (!Number.isFinite(milliseconds) || milliseconds <= 0) throw new Error(`a dispatch reported ${milliseconds} ms`);
        samplesMs.push(milliseconds);
    }
    return {
        medianMs: median(samplesMs),
        minMs: Math.min(...samplesMs),
        maxMs: Math.max(...samplesMs),
        samplesMs,
    };
}

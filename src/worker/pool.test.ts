import { afterEach, describe, expect, it, vi } from 'vitest';
import { buildMeshBvhAsync } from '../trace/mesh-bvh-build.ts';
import type { BuildRequest, BuildResponse } from '../trace/mesh-bvh-worker.ts';
import { meshBvhTransferables, type MeshGeometry } from '../trace/mesh-bvh.ts';
import { tessellatedQuad } from '../trace/oracle-fixtures.ts';
import { createBvhWorkerPool, defaultWorkerPoolSize } from './pool.ts';

const quad = (cells: number): MeshGeometry => tessellatedQuad(cells, 1);

/**
 * A stand-in for a module worker running mesh-bvh-worker.ts: it builds with
 * the same entry point on a later turn and answers through `onmessage`, and
 * detaches transferred inputs the way `postMessage` does.
 */
class FakeWorker {
    static started: FakeWorker[] = [];
    /** Every worker fails to load (its first message errors), as when a bundler didn't ship the file. */
    static failToLoad = false;
    onmessage: ((event: { data: BuildResponse }) => void) | null = null;
    onerror: ((event: { message: string; preventDefault?: () => void }) => void) | null = null;
    terminated = false;
    busy = 0;
    maxBusy = 0;
    received: BuildRequest[] = [];

    constructor() {
        FakeWorker.started.push(this);
    }

    postMessage(request: BuildRequest, transfer: Transferable[] = []): void {
        // A transfer detaches the caller's buffers; structuredClone does the same.
        const moved = structuredClone(request, { transfer });
        this.received.push(moved);
        this.busy++;
        this.maxBusy = Math.max(this.maxBusy, this.busy);
        setTimeout(() => {
            if (this.terminated) return;
            if (FakeWorker.failToLoad) {
                this.onerror?.({ message: 'failed to load' });
                return;
            }
            buildMeshBvhAsync(moved.geometry, moved.options).then((bvh) => {
                this.busy--;
                if (this.terminated) return;
                const answer = structuredClone({ id: moved.id, bvh }, { transfer: meshBvhTransferables(bvh) });
                this.onmessage?.({ data: answer });
            });
        }, 5);
    }

    terminate(): void {
        this.terminated = true;
    }
}

afterEach(() => {
    vi.unstubAllGlobals();
    FakeWorker.started = [];
    FakeWorker.failToLoad = false;
});

describe('createBvhWorkerPool', () => {
    it('builds inline in Node (no Worker), with the same trees as building directly', async () => {
        const pool = createBvhWorkerPool();
        expect(pool.usesWorkers).toBe(false);
        expect(pool.size).toBe(1);
        const [a, b] = await Promise.all([pool.build(quad(4)), pool.build(quad(6), { maxLeafPrimitives: 2 })]);
        const direct = await buildMeshBvhAsync(quad(6), { maxLeafPrimitives: 2 });
        expect(a.triangleCount).toBe(32);
        expect(new Uint8Array(b.nodes)).toEqual(new Uint8Array(direct.nodes));
        expect(pool.stats.finished).toBe(2);
    });

    it("throws for worker: 'always' without a Worker, and rejects a size below 1", () => {
        expect(() => createBvhWorkerPool({ worker: 'always' })).toThrow(/no Worker/);
        expect(() => createBvhWorkerPool({ size: 0 })).toThrow(RangeError);
        expect(defaultWorkerPoolSize()).toBeGreaterThanOrEqual(1);
        expect(defaultWorkerPoolSize()).toBeLessThanOrEqual(4);
    });

    it('spreads builds over at most `size` workers, one build per worker at a time, results moved out', async () => {
        vi.stubGlobal('Worker', FakeWorker);
        const pool = createBvhWorkerPool({ size: 3 });
        expect(pool.usesWorkers).toBe(true);
        const builds = Array.from({ length: 7 }, (_, index) => pool.build(quad(2 + index)));
        expect(pool.stats.running).toBe(3);
        expect(pool.stats.queued).toBe(4);
        const trees = await Promise.all(builds);
        expect(trees.map((tree) => tree.triangleCount)).toEqual(Array.from({ length: 7 }, (_, index) => 2 * (2 + index) ** 2));
        expect(FakeWorker.started).toHaveLength(3);
        expect(FakeWorker.started.every((worker) => worker.maxBusy === 1)).toBe(true);
        expect(pool.stats).toEqual({ queued: 0, running: 0, workers: 3, finished: 7 });
        pool.dispose();
        expect(FakeWorker.started.every((worker) => worker.terminated)).toBe(true);
    });

    it('copies the input by default and moves it with transferInput', async () => {
        vi.stubGlobal('Worker', FakeWorker);
        const pool = createBvhWorkerPool({ size: 1 });
        const kept = quad(3);
        await pool.build(kept);
        expect(kept.positions.length).toBeGreaterThan(0);
        const moved = quad(3);
        await pool.build(moved, {}, { transferInput: true });
        expect(moved.positions.length).toBe(0);
    });

    it('cancels a queued build before it starts, and drops a running one', async () => {
        vi.stubGlobal('Worker', FakeWorker);
        const pool = createBvhWorkerPool({ size: 1 });
        const running = new AbortController();
        const queued = new AbortController();
        const first = pool.build(quad(4), {}, { signal: running.signal });
        const second = pool.build(quad(5), {}, { signal: queued.signal });
        const third = pool.build(quad(6));
        queued.abort();
        running.abort();
        await expect(first).rejects.toMatchObject({ name: 'AbortError' });
        await expect(second).rejects.toMatchObject({ name: 'AbortError' });
        expect((await third).triangleCount).toBe(72);
        // The cancelled queued build never reached the worker.
        expect(FakeWorker.started[0].received.map((request) => request.geometry.positions.length)).toEqual([quad(4).positions.length, quad(6).positions.length]);
        await expect(pool.build(quad(2), {}, { signal: AbortSignal.abort() })).rejects.toMatchObject({ name: 'AbortError' });
    });

    it('falls back to building inline when no worker ever loads, and re-runs the builds it had sent', async () => {
        vi.stubGlobal('Worker', FakeWorker);
        FakeWorker.failToLoad = true;
        const pool = createBvhWorkerPool({ size: 2 });
        const trees = await Promise.all([pool.build(quad(2)), pool.build(quad(3)), pool.build(quad(4))]);
        expect(trees.map((tree) => tree.triangleCount)).toEqual([8, 18, 32]);
        expect(pool.usesWorkers).toBe(false);
        expect(FakeWorker.started.every((worker) => worker.terminated)).toBe(true);
    });

    it('rejects queued and running builds on dispose', async () => {
        vi.stubGlobal('Worker', FakeWorker);
        const pool = createBvhWorkerPool({ size: 1 });
        const builds = [pool.build(quad(2)), pool.build(quad(3))];
        pool.dispose();
        for (const build of builds) await expect(build).rejects.toThrow(/disposed/);
        await expect(pool.build(quad(2))).rejects.toThrow(/disposed/);
    });
});

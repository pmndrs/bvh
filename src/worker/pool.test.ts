import { afterEach, describe, expect, it, vi } from 'vitest';
import { buildMeshBvhAsync } from '../trace/mesh-bvh-build.ts';
import type { BuildRequest, BuildResponse, WorkerReady } from '../trace/mesh-bvh-worker.ts';
import { createMeshBvhBuilder } from '../trace/mesh-bvh-async.ts';
import { meshBvhTransferables, type MeshGeometry } from '../trace/mesh-bvh.ts';
import { tessellatedQuad } from '../trace/oracle-fixtures.ts';
import { createBvhWorkerPool, defaultWorkerPoolSize } from './pool.ts';

const quad = (cells: number): MeshGeometry => tessellatedQuad(cells, 1);

/**
 * A stand-in for a module worker running mesh-bvh-worker.ts: on a later turn
 * it loads and posts `{ ready: true }` (or errs, as a script that can't load
 * does, with an empty message), then builds with the same entry point and
 * answers through `onmessage`, and detaches transferred inputs the way
 * `postMessage` does.
 */
class FakeWorker {
    static started: FakeWorker[] = [];
    /** Every worker fails to load, as when a bundler didn't ship the file or a dev server refused it (Vite's fs.allow). */
    static failToLoad = false;
    /** Only the worker started at this index (0 is the first) fails to load. */
    static failToLoadIndex: number | null = null;
    /** Every worker loads, then dies on its first build. */
    static crashOnBuild = false;
    /** The WASM memory each answer reports (mesh-bvh-worker.ts sends the builder's). */
    static heapBytes = 64 * 2 ** 20;
    onmessage: ((event: { data: BuildResponse | WorkerReady }) => void) | null = null;
    onerror: ((event: { message: string; preventDefault?: () => void }) => void) | null = null;
    terminated = false;
    loaded = false;
    /** Requests posted before the worker said it loaded: the pool must send none. */
    sentBeforeReady = 0;
    busy = 0;
    maxBusy = 0;
    received: BuildRequest[] = [];

    constructor() {
        const index = FakeWorker.started.push(this) - 1;
        setTimeout(() => {
            if (this.terminated) return;
            if (FakeWorker.failToLoad || FakeWorker.failToLoadIndex === index) {
                this.onerror?.({ message: '' });
                return;
            }
            this.loaded = true;
            this.onmessage?.({ data: { ready: true } });
        }, 2);
    }

    postMessage(request: BuildRequest, transfer: Transferable[] = []): void {
        if (!this.loaded) this.sentBeforeReady++;
        // A transfer detaches the caller's buffers; structuredClone does the same.
        const moved = structuredClone(request, { transfer });
        this.received.push(moved);
        this.busy++;
        this.maxBusy = Math.max(this.maxBusy, this.busy);
        setTimeout(() => {
            if (this.terminated) return;
            if (FakeWorker.crashOnBuild) {
                this.onerror?.({ message: 'it crashed' });
                return;
            }
            buildMeshBvhAsync(moved.geometry, moved.options).then((bvh) => {
                this.busy--;
                if (this.terminated) return;
                const answer = structuredClone({ id: moved.id, bvh, heapBytes: FakeWorker.heapBytes }, { transfer: meshBvhTransferables(bvh) });
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
    FakeWorker.crashOnBuild = false;
    FakeWorker.failToLoadIndex = null;
    FakeWorker.heapBytes = 64 * 2 ** 20;
});

describe('createBvhWorkerPool', () => {
    it('builds inline in Node (no Worker), with the same trees as building directly', async () => {
        const pool = createBvhWorkerPool();
        expect(pool.usesWorkers).toBe(false);
        expect(pool.inlineReason).toMatch(/no Worker/);
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
        expect(pool.inlineReason).toBeNull();
        const builds = Array.from({ length: 7 }, (_, index) => pool.build(quad(2 + index)));
        // Three workers start; none has a build until it says it loaded.
        expect(FakeWorker.started).toHaveLength(3);
        expect(pool.stats).toMatchObject({ running: 0, queued: 7, workers: 3 });
        const trees = await Promise.all(builds);
        expect(trees.map((tree) => tree.triangleCount)).toEqual(Array.from({ length: 7 }, (_, index) => 2 * (2 + index) ** 2));
        expect(FakeWorker.started).toHaveLength(3);
        expect(FakeWorker.started.every((worker) => worker.maxBusy === 1 && worker.sentBeforeReady === 0)).toBe(true);
        expect(pool.stats).toEqual({ queued: 0, running: 0, workers: 3, finished: 7, recycled: 0 });
        pool.dispose();
        expect(FakeWorker.started.every((worker) => worker.terminated)).toBe(true);
    });

    it('ends a worker whose build left its WASM memory past recycleAboveBytes, and starts a fresh one for the next build', async () => {
        vi.stubGlobal('Worker', FakeWorker);
        const pool = createBvhWorkerPool({ size: 1 });
        await pool.build(quad(3));
        expect(pool.stats).toMatchObject({ workers: 1, recycled: 0 });
        FakeWorker.heapBytes = 1300 * 2 ** 20;
        const big = await pool.build(quad(4));
        expect(big.triangleCount).toBe(32);
        expect(pool.stats).toMatchObject({ workers: 0, recycled: 1 });
        expect(FakeWorker.started[0].terminated).toBe(true);
        FakeWorker.heapBytes = 64 * 2 ** 20;
        await pool.build(quad(5));
        expect(FakeWorker.started).toHaveLength(2);
        expect(pool.stats).toMatchObject({ workers: 1, recycled: 1 });
        // Infinity keeps the worker whatever its memory.
        FakeWorker.heapBytes = 1300 * 2 ** 20;
        const keeping = createBvhWorkerPool({ size: 1, recycleAboveBytes: Infinity });
        await keeping.build(quad(3));
        await keeping.build(quad(3));
        expect(keeping.stats).toMatchObject({ workers: 1, recycled: 0 });
        expect(() => createBvhWorkerPool({ recycleAboveBytes: -1 })).toThrow(RangeError);
        pool.dispose();
        keeping.dispose();
    });

    it('builds inline, losing nothing, when a fresh worker fails to load after a recycled one answered', async () => {
        vi.stubGlobal('Worker', FakeWorker);
        const pool = createBvhWorkerPool({ size: 1, transferInput: true });
        FakeWorker.heapBytes = 1300 * 2 ** 20;
        await pool.build(quad(3));
        expect(pool.stats).toMatchObject({ workers: 0, recycled: 1 });
        // The fresh worker can't load: it never got the build, so the build runs inline from its own arrays.
        FakeWorker.failToLoad = true;
        const geometry = quad(4);
        expect((await pool.build(geometry)).triangleCount).toBe(32);
        expect(geometry.positions.length).toBe(quad(4).positions.length);
        expect(FakeWorker.started[1].received).toHaveLength(0);
        expect(pool.usesWorkers).toBe(false);
        expect(pool.inlineReason).toMatch(/could not be loaded/);
        pool.dispose();
    });

    it('goes inline when a second worker fails to load, while the loaded one finishes the build it holds', async () => {
        vi.stubGlobal('Worker', FakeWorker);
        const pool = createBvhWorkerPool({ size: 2, transferInput: true });
        // The first worker loads and takes a build; the second can't load.
        await pool.build(quad(1));
        FakeWorker.failToLoadIndex = 1;
        const running = pool.build(quad(6));
        const next = [pool.build(quad(3)), pool.build(quad(4))];
        expect((await Promise.all([running, ...next])).map((tree) => tree.triangleCount)).toEqual([72, 18, 32]);
        expect(FakeWorker.started[0].received).toHaveLength(2);
        expect(FakeWorker.started[1].received).toHaveLength(0);
        expect(FakeWorker.started.every((worker) => worker.terminated)).toBe(true);
        expect(pool.usesWorkers).toBe(false);
        expect(pool.stats).toMatchObject({ queued: 0, running: 0, workers: 0, finished: 4 });
    });

    it('fails only the build of a loaded worker that dies, and keeps using workers', async () => {
        vi.stubGlobal('Worker', FakeWorker);
        const pool = createBvhWorkerPool({ size: 1 });
        expect((await pool.build(quad(2))).triangleCount).toBe(8);
        FakeWorker.crashOnBuild = true;
        await expect(pool.build(quad(3))).rejects.toThrow(/a BVH worker failed: it crashed/);
        FakeWorker.crashOnBuild = false;
        expect((await pool.build(quad(5))).triangleCount).toBe(50);
        expect(pool.usesWorkers).toBe(true);
        expect(FakeWorker.started).toHaveLength(2);
        pool.dispose();
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
        // A first build, so the worker has loaded and the next one starts on it at once.
        await pool.build(quad(1));
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
        expect(FakeWorker.started[0].received.map((request) => request.geometry.positions.length)).toEqual([quad(1), quad(4), quad(6)].map((geometry) => geometry.positions.length));
        await expect(pool.build(quad(2), {}, { signal: AbortSignal.abort() })).rejects.toMatchObject({ name: 'AbortError' });
    });

    it('falls back to building inline when no worker loads, with moved inputs too: no build reached a worker', async () => {
        vi.stubGlobal('Worker', FakeWorker);
        FakeWorker.failToLoad = true;
        // transferInput as SceneTrees' merged groups ask for it: before the handshake these builds were lost.
        const pool = createBvhWorkerPool({ size: 2, transferInput: true });
        const geometries = [quad(2), quad(3), quad(4)];
        const trees = await Promise.all(geometries.map((geometry) => pool.build(geometry)));
        expect(trees.map((tree) => tree.triangleCount)).toEqual([8, 18, 32]);
        expect(geometries.map((geometry) => geometry.positions.length)).toEqual([2, 3, 4].map((cells) => quad(cells).positions.length));
        expect(FakeWorker.started.every((worker) => worker.terminated && worker.received.length === 0)).toBe(true);
        expect(pool.usesWorkers).toBe(false);
        expect(pool.inlineReason).toMatch(/a BVH worker could not be loaded.*server\.fs\.allow/);
        expect(pool.stats).toMatchObject({ queued: 0, running: 0, workers: 0, finished: 3 });
    });

    it('rejects queued and running builds on dispose', async () => {
        vi.stubGlobal('Worker', FakeWorker);
        const pool = createBvhWorkerPool({ size: 1 });
        const builds = [pool.build(quad(2)), pool.build(quad(3))];
        pool.dispose();
        for (const build of builds) await expect(build).rejects.toThrow(/disposed/);
        await expect(pool.build(quad(2))).rejects.toThrow(/disposed/);
    });

    it('rejects inline builds on dispose, the one already running and the one about to start', async () => {
        vi.useFakeTimers({ toFake: ['setTimeout'] });
        const pool = createBvhWorkerPool({ worker: 'never' });
        const running = pool.build(quad(30));
        // Run the inline runner's turn: the build has started and waits on tinybvh (a promise) when dispose comes.
        vi.advanceTimersByTime(0);
        vi.useRealTimers();
        expect(pool.stats.running).toBe(1);
        const queued = pool.build(quad(2));
        pool.dispose();
        await expect(running).rejects.toThrow(/disposed/);
        await expect(queued).rejects.toThrow(/disposed/);
        // And one handed to the inline runner in the same turn as dispose.
        const other = createBvhWorkerPool({ worker: 'never' });
        const scheduled = other.build(quad(2));
        other.dispose();
        await expect(scheduled).rejects.toThrow(/disposed/);
    });
    it('copies views into one shared buffer instead of detaching it, and still moves whole-buffer arrays', async () => {
        vi.stubGlobal('Worker', FakeWorker);
        const pool = createBvhWorkerPool({ size: 1, transferInput: true });
        // Two meshes and their flags viewing one glTF-style BIN buffer.
        const a = quad(3);
        const b = quad(4);
        const triangles = (geometry: MeshGeometry) => geometry.indices!.length / 3;
        const floats = a.positions.length + b.positions.length;
        const words = a.indices!.length + b.indices!.length + triangles(a) + triangles(b);
        const bin = new ArrayBuffer(4 * (floats + words));
        let offset = 0;
        const place = <T extends Float32Array | Uint32Array>(Kind: { new (buffer: ArrayBuffer, offset: number, length: number): T }, source: ArrayLike<number>): T => {
            const view = new Kind(bin, offset, source.length);
            view.set(source);
            offset += 4 * source.length;
            return view;
        };
        const inBin = (geometry: MeshGeometry): MeshGeometry => ({
            positions: place(Float32Array, geometry.positions),
            indices: place(Uint32Array, geometry.indices!),
            triangleFlags: place(Uint32Array, new Uint32Array(triangles(geometry))),
        });
        const sharedA = inBin(a);
        const sharedB = inBin(b);
        const trees = await Promise.all([pool.build(sharedA), pool.build(sharedB)]);
        expect(trees.map((tree) => tree.triangleCount)).toEqual([18, 32]);
        expect(bin.byteLength).toBe(4 * (floats + words));
        expect(sharedB.positions).toEqual(b.positions);
        // Arrays that cover their whole buffer still move.
        const whole = quad(2);
        expect((await pool.build(whole)).triangleCount).toBe(8);
        expect(whole.positions.length).toBe(0);
    });

    it('rejects a build whose input cannot be posted and keeps the worker for the next one', async () => {
        vi.stubGlobal('Worker', FakeWorker);
        const pool = createBvhWorkerPool({ size: 1 });
        const detached = quad(3);
        structuredClone(detached.positions, { transfer: [detached.positions.buffer] });
        // The first is sent when the worker says it loaded; the second from the worker's answer (onmessage).
        const first = pool.build(quad(2));
        const bad = pool.build(detached);
        const after = pool.build(quad(4));
        expect((await first).triangleCount).toBe(8);
        await expect(bad).rejects.toThrow(/could not be sent/);
        expect((await after).triangleCount).toBe(32);
        await expect(pool.build(detached)).rejects.toThrow(/could not be sent/);
        expect(pool.stats).toMatchObject({ queued: 0, running: 0, finished: 4 });
    });

    it("worker: 'always' rejects the builds on workers too when no worker loads, and stops those workers", async () => {
        vi.stubGlobal('Worker', FakeWorker);
        FakeWorker.failToLoad = true;
        const pool = createBvhWorkerPool({ size: 2, worker: 'always' });
        const builds = [pool.build(quad(2)), pool.build(quad(3)), pool.build(quad(4))];
        for (const build of builds) await expect(build).rejects.toThrow(/could not be loaded/);
        expect(FakeWorker.started.length).toBeGreaterThan(0);
        expect(FakeWorker.started.every((worker) => worker.terminated)).toBe(true);
        expect(pool.stats).toMatchObject({ queued: 0, running: 0, workers: 0 });
    });
});

describe('createMeshBvhBuilder (a pool of one)', () => {
    it("uses its worker once it loaded, and with 'auto' builds inline when it can't load, keeping moved inputs", async () => {
        vi.stubGlobal('Worker', FakeWorker);
        const builder = createMeshBvhBuilder({ transferInput: true });
        expect((await builder.build(quad(3))).triangleCount).toBe(18);
        expect(builder.usesWorker).toBe(true);
        expect(FakeWorker.started[0].sentBeforeReady).toBe(0);
        builder.dispose();

        FakeWorker.failToLoad = true;
        const failing = createMeshBvhBuilder({ transferInput: true });
        const geometry = quad(4);
        expect((await failing.build(geometry)).triangleCount).toBe(32);
        expect(geometry.positions.length).toBe(quad(4).positions.length);
        expect(failing.usesWorker).toBe(false);
    });

    it("with 'always' rejects when its worker can't load", async () => {
        vi.stubGlobal('Worker', FakeWorker);
        FakeWorker.failToLoad = true;
        const builder = createMeshBvhBuilder({ worker: 'always' });
        await expect(builder.build(quad(2))).rejects.toThrow(/could not be loaded/);
    });
});

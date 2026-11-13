import { afterEach, describe, expect, it, vi } from 'vitest';
import { createTopLevelBuilder } from './top-level-async.ts';
import { TopLevelShapeKeeper, type TopLevelBuildInput } from './top-level-shape.ts';
import type { TopLevelRequest, TopLevelResponse } from './top-level-worker.ts';
import type { WorkerReady } from './mesh-bvh-worker.ts';

/** `count` unit boxes in a row, identity-scaled and translated along x. */
function rowOfInstances(count: number): TopLevelBuildInput {
    const objectToWorld = new Float64Array(count * 16);
    const meshBounds = new Float32Array(count * 6);
    for (let instance = 0; instance < count; instance++) {
        objectToWorld.set([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, instance * 3, 0, 0, 1], instance * 16);
        meshBounds.set([-1, -1, -1, 1, 1, 1], instance * 6);
    }
    return { slots: Uint32Array.from({ length: count }, (_, index) => index), visible: new Uint8Array(count).fill(1), objectToWorld, meshBounds };
}

/** A stand-in for a module worker running top-level-worker.ts, with the same keeper; it loads (posts ready) or fails to. */
class FakeTopLevelWorker {
    static started: FakeTopLevelWorker[] = [];
    static failToLoad = false;
    onmessage: ((event: { data: TopLevelResponse | WorkerReady }) => void) | null = null;
    onerror: ((event: { message: string; preventDefault?: () => void }) => void) | null = null;
    loaded = false;
    terminated = false;
    sentBeforeReady = 0;
    received = 0;
    private readonly keeper = new TopLevelShapeKeeper();

    constructor() {
        FakeTopLevelWorker.started.push(this);
        setTimeout(() => {
            if (this.terminated) return;
            if (FakeTopLevelWorker.failToLoad) {
                this.onerror?.({ message: '' });
                return;
            }
            this.loaded = true;
            this.onmessage?.({ data: { ready: true } });
        }, 2);
    }

    postMessage(request: TopLevelRequest, transfer: Transferable[] = []): void {
        if (!this.loaded) this.sentBeforeReady++;
        this.received++;
        const { id, mode, input, ratio, generation, currentGeneration } = structuredClone(request, { transfer });
        setTimeout(() => {
            if (this.terminated) return;
            const result = mode === 'build' ? { shape: this.keeper.build(input, generation), relativeCost: 0, builtCost: 0 } : this.keeper.check(input, ratio, currentGeneration, generation);
            this.onmessage?.({ data: { id, ...result, workerMs: 0 } });
        }, 1);
    }

    terminate(): void {
        this.terminated = true;
    }
}

afterEach(() => {
    vi.unstubAllGlobals();
    FakeTopLevelWorker.started = [];
    FakeTopLevelWorker.failToLoad = false;
});

describe('createTopLevelBuilder', () => {
    it('holds requests until the worker says it loaded', async () => {
        vi.stubGlobal('Worker', FakeTopLevelWorker);
        const builder = createTopLevelBuilder();
        const results = await Promise.all([
            builder.request('build', rowOfInstances(8), 0, { current: 0, next: 1 }),
            builder.request('check', rowOfInstances(8), 1.5, { current: 1, next: 2 }),
        ]);
        expect(results[0].shape?.recordSlots.length).toBe(8);
        expect(builder.usesWorker).toBe(true);
        const [worker] = FakeTopLevelWorker.started;
        expect(worker.received).toBe(2);
        expect(worker.sentBeforeReady).toBe(0);
        builder.dispose();
    });

    it("answers inline, with the moved inputs intact, when the worker can't load, and reports it once", async () => {
        vi.stubGlobal('Worker', FakeTopLevelWorker);
        FakeTopLevelWorker.failToLoad = true;
        const warnings: string[] = [];
        const builder = createTopLevelBuilder({ warn: (message) => warnings.push(message) });
        const input = rowOfInstances(8);
        const built = await builder.request('build', input, 0, { current: 0, next: 1 });
        expect(built.shape?.recordSlots.length).toBe(8);
        expect(input.slots.length).toBe(8);
        // Later requests run inline too, against the inline keeper's shape.
        const checked = await builder.request('check', rowOfInstances(8), 1.5, { current: 1, next: 2 });
        expect(checked.shape).toBeNull();
        expect(builder.usesWorker).toBe(false);
        expect(FakeTopLevelWorker.started[0].received).toBe(0);
        expect(FakeTopLevelWorker.started[0].terminated).toBe(true);
        expect(warnings).toHaveLength(1);
        expect(warnings[0]).toMatch(/the top-level worker could not be loaded/);
    });
});

/**
 * `GpuFirstTrees`' capped queue (`maxBatchBytes`, docs/internals/design/bvh-builders.md
 * "Build peak memory"), in Node with a stand-in H-PLOC builder
 * (tests/support/fake-hploc.ts): every capped build of one builder shares one
 * queue and one scratch pool, so at most one batch's scratch and outputs are
 * alive at once however many builds run; a batch that can't start or fails
 * comes back refused (the promises never reject) without leaking its pool or
 * outputs; and a batch's outputs are counted from their allocation until the
 * caller claims them.
 */

import { describe, expect, it } from 'vitest';
import { createFakeGpu } from '../../tests/support/fake-gpu.ts';
import { FakeHploc, settle } from '../../tests/support/fake-hploc.ts';
import { addFakeTimestamps } from '../../tests/support/fake-timestamps.ts';
import { tessellatedQuad } from '../trace/oracle-fixtures.ts';
import type { MeshGeometry } from '../trace/mesh-bvh.ts';
import { GpuFirstTrees, type GpuTreeBatchResult, type GpuTreeBuilds, type GpuTreeRequest } from './gpu-first-trees.ts';

/** Smaller than any one-mesh batch: every mesh is a batch of its own. */
const SMALL_CAP = 4 * 1024;

function setUp() {
    const gpu = createFakeGpu();
    const hploc = new FakeHploc(gpu.device);
    const builder = new GpuFirstTrees(gpu.device, { hploc });
    let next = 0;
    const requests = (sizes: readonly number[]): GpuTreeRequest[] =>
        sizes.map((size) => {
            const geometry: MeshGeometry = tessellatedQuad(size, 1);
            const slot = next++;
            hploc.slotsOf.set(geometry, slot);
            return { slot, generation: 1, geometry };
        });
    return { gpu, hploc, builder, requests };
}

/** What a caller does with a finished batch: counts its outputs, copies its trees out and lets each go. */
function copyOut(result: GpuTreeBatchResult): void {
    for (const built of result.built) {
        built.outputs.claim();
        built.outputs.release();
    }
}

describe('GpuFirstTrees: the capped queue', () => {
    it('runs concurrent capped builds one batch at a time, in order, on one pool grown for the largest batch queued', async () => {
        const { hploc, builder, requests } = setUp();
        const finished: string[] = [];
        const builds: GpuTreeBuilds[] = [];
        const start = (name: string, sizes: number[]) => {
            const builds_ = builder.build(requests(sizes), { maxBatchBytes: SMALL_CAP });
            builds.push(builds_);
            builds_.batches.forEach((batch, index) =>
                batch.then((result) => {
                    finished.push(`${name}${index}`);
                    copyOut(result);
                }),
            );
            return builds_;
        };
        // The load, a deforming mesh's own build and a runtime add, all at once; the last has the biggest mesh.
        const load = start('load', [2, 3, 2]);
        const deforming = start('deforming', [2]);
        const runtime = start('runtime', [3, 12]);
        // Only the first batch started (its arrays read before build returned); the rest wait, arrays untouched.
        expect(hploc.events.filter((event) => event.startsWith('start'))).toEqual(['start 0']);
        expect([...(load.readsGeometryLater ?? [])]).toEqual([1, 2]);
        expect([...(deforming.readsGeometryLater ?? [])]).toEqual([3]);
        expect([...(runtime.readsGeometryLater ?? [])]).toEqual([4, 5]);
        // The pool is counted once, by the build whose batch has it.
        const pool = builds.reduce((sum, b) => sum + b.buildingBytes, 0);
        expect(load.buildingBytes).toBe(pool);
        expect(deforming.buildingBytes + runtime.buildingBytes).toBe(0);

        await Promise.all(builds.flatMap((b) => b.batches));
        await settle();
        expect(hploc.violations).toEqual([]);
        expect(hploc.maxActive).toBe(1);
        expect(hploc.maxPoolsAlive).toBe(1);
        expect(finished).toEqual(['load0', 'load1', 'load2', 'deforming0', 'runtime0', 'runtime1']);
        expect(hploc.batches).toEqual([[0], [1], [2], [3], [4], [5]]);
        // Each batch starts after the one before it let go of its outputs.
        const order = hploc.events.filter((event) => /^(start|let go)/.test(event));
        expect(order).toEqual([0, 1, 2, 3, 4, 5].flatMap((index) => [`start ${index}`, `let go ${index}`]));
        // The first pool held the batches queued when it was made; the 12-quad mesh's batch needed a bigger one, made after the first was freed.
        const pools = hploc.events.filter((event) => event.startsWith('pool'));
        expect(pools.length).toBe(4);
        expect(pools[1]).toBe('pool freed');
        expect(pools[3]).toBe('pool freed');
        expect(Number(pools[2].split(' ')[1])).toBeGreaterThan(Number(pools[0].split(' ')[1]));
        expect(hploc.poolsAlive).toBe(0);
        expect(builds.every((b) => b.buildingBytes === 0)).toBe(true);
        expect(builder.queuedBatches).toBe(0);
    });

    it('never rejects: a batch that throws while it starts, or whose readback fails, comes back refused, frees what it had, and the queue goes on', async () => {
        const { hploc, builder, requests } = setUp();
        // The first batch (started inside build) and a later one throw while recorded; another's readback fails.
        hploc.throwOnEncode.add(0).add(2);
        hploc.failFinish.add(3);
        const builds = builder.build(requests([2, 2, 2, 2, 2]), { maxBatchBytes: SMALL_CAP });
        // The caller copies each batch out as it comes (the next one waits for that).
        const results = await Promise.all(builds.batches.map((batch) => batch.then((result) => (copyOut(result), result))));
        await settle();
        expect(results.map((result) => result.failure === null)).toEqual([false, true, false, false, true]);
        expect(results[0].failure).toMatch(/could not start: batch 0 could not be recorded/);
        expect(results[2].failure).toMatch(/could not start: batch 2 could not be recorded/);
        expect(results[3].failure).toMatch(/failed: batch 3's readback failed/);
        expect(results[0].refused.map((tree) => tree.slot)).toEqual([0]);
        // Each failed batch freed its outputs ("dropped"), the pool was freed after the last, and nothing is counted.
        expect(hploc.events.filter((event) => event.startsWith('dropped'))).toEqual(['dropped 0', 'dropped 2', 'dropped 3']);
        expect(hploc.violations).toEqual([]);
        expect(hploc.active).toBe(0);
        expect(hploc.poolsAlive).toBe(0);
        expect(builds.buildingBytes).toBe(0);
        // The builder still works.
        const again = builder.build(requests([2]), { maxBatchBytes: SMALL_CAP });
        const [result] = await Promise.all(again.batches);
        expect(result.failure).toBeNull();
        copyOut(result);
        await settle();
        expect(hploc.poolsAlive).toBe(0);
    });

    it('counts a batch\'s outputs from their allocation until the caller claims them, through a layout readback', async () => {
        const { hploc, builder, requests } = setUp();
        let openGate: () => void = () => {};
        hploc.layoutGate = new Promise<void>((resolve) => (openGate = resolve));
        const builds = builder.build(requests([4]), { maxBatchBytes: SMALL_CAP, readLayouts: true });
        const outputBytes = 16 * 2 * (64 + 48 + 4);
        const poolBytes = builds.buildingBytes - outputBytes;
        expect(poolBytes).toBeGreaterThan(0);
        // Stats read, layouts still being read back: the outputs exist and are still the build's to count.
        await settle();
        expect(builds.buildingBytes).toBe(poolBytes + outputBytes);
        openGate();
        const [result] = await Promise.all(builds.batches);
        // The pool is freed (nothing else queued), the outputs still counted until claimed.
        expect(builds.buildingBytes).toBe(outputBytes);
        result.built[0].outputs.claim();
        expect(builds.buildingBytes).toBe(0);
        result.built[0].outputs.release();
    });

    it('a cancelled build refuses its batches that waited in the queue', async () => {
        const { hploc, builder, requests } = setUp();
        let cancelled = false;
        const builds = builder.build(requests([2, 2, 2]), { maxBatchBytes: SMALL_CAP, cancelled: () => cancelled });
        const [first] = await Promise.all([builds.batches[0]]);
        cancelled = true;
        copyOut(first);
        const rest = await Promise.all(builds.batches.slice(1));
        expect(rest.map((result) => result.failure)).toEqual(['the GPU mesh tree build was cancelled', 'the GPU mesh tree build was cancelled']);
        await settle();
        expect(hploc.events.filter((event) => event.startsWith('start'))).toEqual(['start 0']);
        expect(hploc.poolsAlive).toBe(0);
    });
});

describe('GpuFirstTrees: timing and batch starts', () => {
    it('times each batch with timestamps when asked (capped and not), and says when each batch is submitted', async () => {
        const { gpu, hploc, builder, requests } = setUp();
        // Without timestamp-query, batches carry no GPU time.
        const plain = builder.build(requests([2]), { timestamps: true });
        const [untimed] = await Promise.all(plain.batches);
        expect(untimed.gpuMs ?? null).toBeNull();
        copyOut(untimed);

        const counts = addFakeTimestamps(gpu, 2.5);
        const started: number[][] = [];
        const onBatchStart = (members: readonly GpuTreeRequest[]) => started.push(members.map((member) => member.slot));
        const uncapped = builder.build(requests([2, 3]), { timestamps: true, onBatchStart });
        expect(started).toEqual([[1, 2]]); // one batch, submitted at once
        const [timed] = await Promise.all(uncapped.batches);
        expect(timed.gpuMs).toBeCloseTo(2.5, 6);
        expect(counts.resolves).toBe(1);
        copyOut(timed);

        hploc.manual = true;
        const capped = builder.build(requests([2, 2]), { maxBatchBytes: SMALL_CAP, timestamps: true, onBatchStart });
        expect(started).toEqual([[1, 2], [3]]); // the second batch waits in the queue
        await hploc.finishNext();
        const first = await capped.batches[0];
        expect(first.gpuMs).toBeCloseTo(2.5, 6);
        copyOut(first);
        await settle();
        expect(started).toEqual([[1, 2], [3], [4]]);
        await hploc.finishNext();
        copyOut(await capped.batches[1]);

        // Not asked: no query set made, no time.
        hploc.manual = false;
        const before = counts.querySets;
        const [result] = await Promise.all(builder.build(requests([2])).batches);
        expect(result.gpuMs ?? null).toBeNull();
        expect(counts.querySets).toBe(before);
        copyOut(result);
    });
});

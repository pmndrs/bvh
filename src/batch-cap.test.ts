/**
 * The build-peak lever (`maxBatchBytes`, docs/internals/design/bvh-builders.md "Build
 * peak memory"): the planners keep each batch's GPU bytes under the cap, a
 * mesh bigger than the cap gets a batch of its own, and the scratch pool is
 * sized from the largest batch. Only the device's limits are read, so this
 * runs in Node.
 */

import { describe, expect, it } from 'vitest';
import { hplocBatchBytes, hplocScratchPoolBytes, planGpuHplocBatches } from './gpu-hploc.ts';
import { geometryTotals } from './gpu-batch.ts';
import { lbvhBatchBytes, planGpuLbvhBatches } from './gpu-lbvh.ts';
import type { GpuMeshGeometry } from './mesh-input.ts';

const device = { limits: { maxStorageBufferBindingSize: 1 << 30, maxBufferSize: 1 << 30 } } as unknown as GPUDevice;

/** A mesh of `triangles` triangles over `triangles + 2` vertices, 32-bit indices. */
const mesh = (triangles: number): GpuMeshGeometry => ({ positions: new Float32Array((triangles + 2) * 3), indices: new Uint32Array(triangles * 3) });

const sizes = [5000, 300, 12000, 800, 800, 40000, 2000, 2000, 2000, 150, 9000, 60];
const meshes = sizes.map(mesh);
const hplocBytes = (indices: readonly number[]) => hplocBatchBytes(geometryTotals(indices.map((index) => meshes[index]))).total;
const lbvhBytes = (indices: readonly number[]) => lbvhBatchBytes(geometryTotals(indices.map((index) => meshes[index]))).total;

describe('maxBatchBytes: the batch planners', () => {
    for (const [name, plan, bytesOf] of [
        ['H-PLOC', planGpuHplocBatches, hplocBytes],
        ['LBVH', planGpuLbvhBatches, lbvhBytes],
    ] as const) {
        it(`${name}: keeps every batch under the cap, in order, each as full as it can be`, () => {
            const cap = bytesOf([2]) + bytesOf([3]); // about two middling meshes
            const { batches, tooBig } = plan(device, meshes, { maxBatchBytes: cap });
            expect(tooBig).toEqual([]);
            expect(batches.flat()).toEqual(meshes.map((_, index) => index));
            expect(batches.length).toBeGreaterThan(2);
            batches.forEach((batch, at) => {
                if (batch.length > 1) expect(bytesOf(batch)).toBeLessThanOrEqual(cap);
                // The next mesh didn't fit: it starts the next batch because it would have taken this one past the cap.
                const next = batches[at + 1]?.[0];
                if (next !== undefined) expect(bytesOf([...batch, next])).toBeGreaterThan(cap);
            });
        });
    }

    it('builds a mesh bigger than the cap alone, in its own batch', () => {
        const cap = hplocBytes([2]); // the 12,000-triangle mesh; the 40,000 one is far over
        const { batches, tooBig } = planGpuHplocBatches(device, meshes, { maxBatchBytes: cap });
        expect(tooBig).toEqual([]);
        expect(batches).toContainEqual([5]);
        expect(hplocBytes([5])).toBeGreaterThan(cap);
        // A cap smaller than any mesh: every mesh alone.
        expect(planGpuHplocBatches(device, meshes, { maxBatchBytes: 1 }).batches).toEqual(meshes.map((_, index) => [index]));
    });

    it("keeps today's single batch without a cap, and with a cap above the whole scene", () => {
        const all = [meshes.map((_, index) => index)];
        expect(planGpuHplocBatches(device, meshes).batches).toEqual(all);
        expect(planGpuHplocBatches(device, meshes, { maxBatchBytes: hplocBytes(all[0]) }).batches).toEqual(all);
    });

    it('still splits at the device limit under a cap', () => {
        const small = { limits: { maxStorageBufferBindingSize: 64 * 20_000, maxBufferSize: 1 << 30 } } as unknown as GPUDevice;
        const { batches } = planGpuHplocBatches(small, [mesh(15_000), mesh(15_000)], { maxBatchBytes: 1 << 30 });
        expect(batches).toEqual([[0], [1]]);
    });

    it('refuses a cap that is not a positive number', () => {
        expect(() => planGpuHplocBatches(device, meshes, { maxBatchBytes: 0 })).toThrow(RangeError);
        expect(() => planGpuLbvhBatches(device, meshes, { maxBatchBytes: Number.NaN })).toThrow(RangeError);
    });

    it('counts the tinybvh codes in the H-PLOC bytes (8 more per triangle, and the uniforms of the second sort)', () => {
        const totals = geometryTotals([mesh(10_000)]);
        expect(hplocBatchBytes(totals, 'tinybvh-63').total - hplocBatchBytes(totals).total).toBe(10_000 * 8 - 16 + 9 * 256); // 7 + 1 and 7 + 1 sort records, against 6 + 1
    });
});

describe('maxBatchBytes: the scratch pool', () => {
    it('is sized from the largest batch: every pooled buffer at the largest size any batch needs', () => {
        const cap = hplocBytes([2]) + hplocBytes([3]);
        const groups = planGpuHplocBatches(device, meshes, { maxBatchBytes: cap }).batches.map((batch) => batch.map((index) => meshes[index]));
        const pool = hplocScratchPoolBytes(groups);
        const perBatch = groups.map((group) => hplocBatchBytes(geometryTotals(group)).buffers);
        for (const [name, bytes] of Object.entries(pool)) {
            expect(bytes).toBe(Math.max(...perBatch.map((buffers) => buffers[name as keyof typeof buffers])));
        }
        // Outputs are not pooled: each batch hands its own to the caller.
        expect(Object.keys(pool)).not.toContain('outNodes');
        // The pool is the largest batch's scratch, give or take the per-mesh tables and stats of a batch with more meshes.
        const largest = groups.reduce((best, group) => (geometryTotals(group).triangles > geometryTotals(best).triangles ? group : best));
        const poolBytes = Object.values(pool).reduce((sum, bytes) => sum + bytes, 0);
        const largestScratch = hplocBatchBytes(geometryTotals(largest)).scratch;
        expect(poolBytes).toBeGreaterThanOrEqual(largestScratch);
        expect(poolBytes - largestScratch).toBeLessThan(1024);
        // Under the cap, except that a mesh over the cap (here the 40,000-triangle one) builds alone, and the pool follows it.
        expect(poolBytes).toBeGreaterThan(cap);
        expect(hplocBatchBytes(geometryTotals(largest)).total).toBeGreaterThan(cap);
        expect(largest).toEqual([meshes[5]]);
    });

    it('holds as many bytes per triangle as the itemization says (Sponza: 136 scratch, 116 outputs, plus the inputs)', () => {
        // One 100,000-triangle mesh: per-triangle shares of each buffer, rounded.
        const triangles = 100_000;
        const { buffers, outputs } = hplocBatchBytes(geometryTotals([mesh(triangles)]));
        const perTriangle = (bytes: number) => Math.round(bytes / triangles);
        expect(perTriangle(outputs)).toBe(64 + 48 + 4);
        expect(perTriangle(buffers.primitiveBoxes + buffers.keysA + buffers.valuesA + buffers.keysB + buffers.valuesB + buffers.deltas + buffers.scratch + buffers.links)).toBe(32 + 16 + 8 + 52 + 28);
        expect(perTriangle(buffers.indices)).toBe(12);
    });
});

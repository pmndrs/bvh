/**
 * `quantizeOnEmit` (H-PLOC writing quantized nodes in its emit pass,
 * docs/internals/design/compressed-nodes.md): the option checks, the batch's
 * byte accounting the planners and `gpuMaxBatchBytes` see, and the ranges
 * `finish` hands out. The words themselves are checked against the `f32` emit
 * plus the quantize pass on the GPU (the bvh-hploc page); here the CPU twin
 * (hploc-cpu.ts, then quantizeNodes) shows where each mesh's words land.
 */

import { describe, expect, it } from 'vitest';
import { checkHplocOptions, emittedQuantizedRanges, hplocBatchBytes, planGpuHplocBatches } from './gpu-hploc.ts';
import { geometryTotals } from './gpu-lbvh.ts';
import { quantizedMeshRanges } from './gpu-quantize.ts';
import { GPU_LAYOUT_QUANTIZED_NODE_WORDS } from './gpu-layout.ts';
import { buildHplocOnCpu } from './hploc-cpu.ts';
import { quantizeNodes } from './quantized-nodes.ts';
import type { GpuMeshGeometry, GpuMeshTreeRange } from './mesh-input.ts';

const device = { limits: { maxStorageBufferBindingSize: 1 << 30, maxBufferSize: 1 << 30 } } as unknown as GPUDevice;

function soup(triangles: number, seed: number): Float32Array {
    let state = seed >>> 0;
    const random = () => ((state = (state * 1664525 + 1013904223) >>> 0) / 2 ** 32) * 20 - 10;
    return Float32Array.from({ length: triangles * 9 }, random);
}

const range = (nodeOffset: number, nodeCount: number, problem: string | null = null): GpuMeshTreeRange => ({
    nodeOffset,
    nodeCount,
    triangleOffset: 0,
    triangleCount: nodeCount + 1,
    leafCount: nodeCount + 1,
    maxDepth: 1,
    bounds: new Float32Array(6),
    problem,
});

describe('quantizeOnEmit: options', () => {
    it('is off by default and needs quantized nodes', () => {
        expect(checkHplocOptions({}).quantizeOnEmit).toBe(false);
        expect(checkHplocOptions({ nodeFormat: 'quantized' }).quantizeOnEmit).toBe(false);
        expect(checkHplocOptions({ nodeFormat: 'quantized', quantizeOnEmit: true }).quantizeOnEmit).toBe(true);
        expect(() => checkHplocOptions({ quantizeOnEmit: true })).toThrow(/nodeFormat: 'quantized'/);
        expect(() => checkHplocOptions({ nodeFormat: 'f32', quantizeOnEmit: true })).toThrow(/nodeFormat: 'quantized'/);
        expect(() => checkHplocOptions({ nodeFormat: 'quantized', quantizeOnEmit: 1 as unknown as boolean })).toThrow(/boolean/);
    });
});

describe('quantizeOnEmit: the batch bytes', () => {
    const meshes: GpuMeshGeometry[] = [3000, 1, 700, 12000].map((triangles, index) => ({ positions: soup(triangles, index + 1) }));
    const totals = geometryTotals(meshes);

    it('output nodes take a 32-byte slot per radix-tree node and per mesh, not 64 bytes per node', () => {
        for (const leaves of ['records', 'indexed'] as const) {
            const f32 = hplocBatchBytes(totals, 'cubic-30', leaves);
            const emitted = hplocBatchBytes(totals, 'cubic-30', leaves, true, true);
            expect(f32.buffers.outNodes).toBe(totals.inner * 64);
            expect(emitted.buffers.outNodes).toBe((totals.inner + totals.meshes) * 32);
            expect(f32.outputs - emitted.outputs).toBe(totals.inner * 32 - totals.meshes * 32);
            // Nothing else changes: the scratch and inputs are the same.
            expect(emitted.scratch).toBe(f32.scratch);
        }
    });

    it('per triangle: 116 to 84 bytes of outputs with records, 84 to 52 indexed', () => {
        const big = geometryTotals([{ positions: soup(100_000, 9) }]);
        const perTriangle = (bytes: number) => Math.round(bytes / big.triangles);
        expect(perTriangle(hplocBatchBytes(big).outputs)).toBe(116);
        expect(perTriangle(hplocBatchBytes(big, 'cubic-30', 'records', true, true).outputs)).toBe(84);
        expect(perTriangle(hplocBatchBytes(big, 'cubic-30', 'indexed').outputs)).toBe(84);
        expect(perTriangle(hplocBatchBytes(big, 'cubic-30', 'indexed', true, true).outputs)).toBe(52);
    });

    it('the planner counts the smaller batches under a cap', () => {
        const cap = hplocBatchBytes(geometryTotals([meshes[0], meshes[1], meshes[2]]), 'cubic-30', 'records', true, true).total;
        const f32 = planGpuHplocBatches(device, meshes, { maxBatchBytes: cap });
        const emitted = planGpuHplocBatches(device, meshes, { maxBatchBytes: cap, quantizeOnEmit: true });
        expect(emitted.batches[0]).toEqual([0, 1, 2]);
        expect(f32.batches[0].length).toBeLessThan(3);
    });
});

describe('quantizeOnEmit: the ranges', () => {
    it("mesh m's frame slot is its f32 node offset + m, its root the slot after; parents stay; problem meshes get no nodes", () => {
        // f32 ranges as meshResult gives them: kept nodes numbered across the batch.
        const f32 = [range(0, 5), range(5, 1), range(6, 3, 'mesh 2: refused'), range(9, 7)];
        const emitted = emittedQuantizedRanges(f32);
        expect(emitted.map((mesh) => [mesh.nodeOffset, mesh.nodeCount, mesh.parentOffset])).toEqual([
            [1, 5, 0],
            [7, 1, 5],
            [0, 0, 6],
            [13, 7, 9],
        ]);
        // Every usable mesh's frame slot and nodes are its own, inside (nodes + meshes) slots.
        const slots = new Set<number>();
        for (const mesh of emitted) {
            if (mesh.nodeCount === 0) continue;
            for (let slot = mesh.nodeOffset - 1; slot < mesh.nodeOffset + mesh.nodeCount; slot++) {
                expect(slots.has(slot)).toBe(false);
                slots.add(slot);
            }
        }
        expect(Math.max(...slots)).toBeLessThan(16 + f32.length);
        // The pass's packing describes the same nodes (other slots, same counts and parents).
        const pass = quantizedMeshRanges(f32).meshes;
        expect(pass.map((mesh) => [mesh.nodeCount, mesh.parentOffset])).toEqual(emitted.map((mesh) => [mesh.nodeCount, mesh.parentOffset]));
    });

    it('the CPU twin: each mesh read from its emitted range is quantizeNodes of its f32 tree', () => {
        // What the emit writes, mesh by mesh, at the ranges finish hands out; read back as readTreeLayouts reads a quantized batch.
        const geometries = [soup(400, 3), soup(1, 4), soup(37, 5), soup(2, 6)].map((positions) => ({ positions }));
        const trees = geometries.map((geometry) => buildHplocOnCpu(geometry));
        let rank = 0;
        const f32 = trees.map((tree) => {
            const mesh = range(rank, tree.nodeCount);
            rank += tree.nodeCount;
            return mesh;
        });
        const emitted = emittedQuantizedRanges(f32);
        const buffer = new Uint32Array((rank + trees.length) * GPU_LAYOUT_QUANTIZED_NODE_WORDS);
        trees.forEach((tree, index) => buffer.set(new Uint32Array(quantizeNodes(tree).nodes), (emitted[index].nodeOffset - 1) * GPU_LAYOUT_QUANTIZED_NODE_WORDS));
        trees.forEach((tree, index) => {
            const mesh = emitted[index];
            const read = buffer.slice((mesh.nodeOffset - 1) * GPU_LAYOUT_QUANTIZED_NODE_WORDS, (mesh.nodeOffset + mesh.nodeCount) * GPU_LAYOUT_QUANTIZED_NODE_WORDS);
            expect(Array.from(read)).toEqual(Array.from(new Uint32Array(quantizeNodes(tree).nodes)));
        });
    });
});

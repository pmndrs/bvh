import { describe, expect, it } from 'vitest';
import { treeReadbackExtents } from './gpu-batch.ts';
import { GPU_LAYOUT_QUANTIZED_NODE_BYTES } from './gpu-layout.ts';
import { quantizedMeshRanges } from './gpu-quantize.ts';
import type { GpuMeshTreeRange } from './mesh-input.ts';

function range(nodeOffset: number, nodeCount: number, triangleOffset: number, problem: string | null = null): GpuMeshTreeRange {
    return { nodeOffset, nodeCount, triangleOffset, triangleCount: 10, leafCount: 5, maxDepth: 3, bounds: new Float32Array(6), problem };
}

describe('quantized batch ranges', () => {
    it('a problem mesh after a usable one keeps the readback inside the quantized node buffer', () => {
        const { meshes, jobs, slots } = quantizedMeshRanges([range(0, 100, 0), range(100, 100, 10, 'bad indices')]);
        expect(jobs).toHaveLength(1);
        expect(slots).toBe(101);
        expect(meshes[1]).toMatchObject({ nodeOffset: 0, nodeCount: 0, parentOffset: 100 });
        const { nodeEnd } = treeReadbackExtents(meshes);
        expect(nodeEnd * GPU_LAYOUT_QUANTIZED_NODE_BYTES).toBeLessThanOrEqual(slots * GPU_LAYOUT_QUANTIZED_NODE_BYTES);
    });

    it('readback extents skip problem meshes even with stale ranges', () => {
        const extents = treeReadbackExtents([range(1, 100, 0), range(100, 100, 10, 'bad indices')]);
        expect(extents).toEqual({ nodeEnd: 101, parentEnd: 101, triangleEnd: 10 });
    });
});

/** `generateCpuMeshBvh` in Node: the pool builds inline, and the tree matches one built on this thread. */

import { describe, expect, it } from 'vitest';
import { CpuMeshBvh } from '../query/cpu-mesh-bvh.ts';
import { createRandom, randomTriangleSoup, tessellatedQuad } from '../trace/oracle-fixtures.ts';
import { generateCpuMeshBvh } from './generate.ts';
import { createBvhWorkerPool } from './pool.ts';

describe('generateCpuMeshBvh', () => {
    it('builds on the pool (inline in Node) the tree CpuMeshBvh.create builds, keeping the input as its source', async () => {
        const pool = createBvhWorkerPool();
        expect(pool.usesWorkers).toBe(false);
        const soup = randomTriangleSoup(createRandom(71), 500, 3, 0.5);
        const progress: number[] = [];
        const tree = await generateCpuMeshBvh(pool, soup, { maxLeafPrimitives: 2, onProgress: (value) => progress.push(value) });
        expect(tree).toBeInstanceOf(CpuMeshBvh);
        expect(progress).toEqual([0, 1]);
        const local = await CpuMeshBvh.create(soup, { maxLeafPrimitives: 2 });
        expect(new Uint8Array(tree.serialize())).toEqual(new Uint8Array(local.serialize()));
        // The source is kept: refit works without arguments.
        soup.positions[0] += 10;
        tree.refit();
        expect(tree.getBoundingBox()[3]).toBeGreaterThanOrEqual(soup.positions[0]);
        // Several at once, and a cancelled one.
        const grids = await Promise.all([4, 8, 16].map((cells) => generateCpuMeshBvh(pool, tessellatedQuad(cells, 1))));
        expect(grids.map((grid) => grid.triangleCount)).toEqual([32, 128, 512]);
        const controller = new AbortController();
        const cancelled = generateCpuMeshBvh(pool, tessellatedQuad(4, 1), {}, { signal: controller.signal });
        controller.abort();
        await expect(cancelled).rejects.toThrow(/cancelled/);
        pool.dispose();
    });
});

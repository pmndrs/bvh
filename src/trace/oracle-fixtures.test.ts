/**
 * The exported shared-edge rays (`sharedEdgeRays`): deterministic, aimed where
 * they say, all inside the quad, and none slips through the CPU walk, plain or
 * through a mirrored, non-uniform transform.
 */

import { describe, expect, it } from 'vitest';
import { createTinyBvhBuilder } from '../index.ts';
import { createCpuTracer } from './cpu-traversal.ts';
import { buildMeshBvhWithTinyBvh } from './mesh-bvh-topology.ts';
import { randomAffine, createRandom, SHARED_EDGE_RAY_CATEGORIES, sharedEdgeRays, tessellatedQuad } from './oracle-fixtures.ts';
import { assembleTraceScene } from './scene-assembly.ts';

const tinyBvh = await createTinyBvhBuilder();
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

describe('sharedEdgeRays', () => {
    it('is deterministic, cycles its categories, and gives f32 unit rays', () => {
        const first = sharedEdgeRays(500);
        const again = sharedEdgeRays(500);
        expect(again).toEqual(first);
        expect(sharedEdgeRays(500, { seed: 1 }).rays).not.toEqual(first.rays);
        expect(first.rays).toHaveLength(500);
        expect([...first.categories.slice(0, 12)]).toEqual([0, 2, 3, 1, 2, 3, 1, 2, 3, 1, 0, 3]);
        for (const ray of first.rays) {
            for (const value of [...ray.origin, ...ray.direction, ray.tMin, ray.tMax]) expect(Math.fround(value)).toBe(value);
            expect(Math.hypot(...ray.direction)).toBeCloseTo(1, 6);
        }
        // With a transform there are no exact vertical rays unless asked for.
        const transformed = sharedEdgeRays(100, { objectToWorld: randomAffine(createRandom(3), { extent: 2 }) });
        expect(transformed.categories.includes(0)).toBe(false);
        expect(SHARED_EDGE_RAY_CATEGORIES).toHaveLength(4);
    });

    it('gives the same rays with no transform as with the identity', () => {
        expect(sharedEdgeRays(2000)).toEqual(sharedEdgeRays(2000, { objectToWorld: IDENTITY, verticalEvery: 10 }));
    });

    it('aims every ray at a seam inside the quad, from above or below', () => {
        const half = 2;
        const step = 0.5;
        const { rays, categories } = sharedEdgeRays(3000, { half, cells: 8 });
        const onGrid = (value: number): boolean => Math.abs(value / step - Math.round(value / step)) < 1e-5;
        rays.forEach((ray, i) => {
            expect(ray.origin[2]).not.toBe(0);
            const t = -ray.origin[2] / ray.direction[2];
            const x = ray.origin[0] + t * ray.direction[0];
            const y = ray.origin[1] + t * ray.direction[1];
            const category = categories[i];
            // Grid-line targets stay 0.2 inside the border; diagonal ones at least 5% of a cell.
            const margin = category === 2 ? 0.05 * step - 1e-5 : 0.2 - 1e-5;
            expect(Math.abs(x)).toBeLessThan(half - margin);
            expect(Math.abs(y)).toBeLessThan(half - margin);
            if (category === 0 || category === 1) expect(onGrid(x) || onGrid(y)).toBe(true);
            if (category === 2) expect(Math.abs((x - y) / step - Math.round((x - y) / step))).toBeLessThan(1e-5);
            if (category === 3) expect(onGrid(x) && onGrid(y)).toBe(true);
        });
    });

    it('lets no ray slip through the CPU walk, plain or through a mirrored, non-uniform instance', () => {
        const quad = tessellatedQuad(8, 2);
        const mesh = buildMeshBvhWithTinyBvh(quad, tinyBvh);
        const random = createRandom(9);
        for (const objectToWorld of [IDENTITY, randomAffine(random, { extent: 3, mirror: true }), randomAffine(random, { extent: 3 })]) {
            const scene = assembleTraceScene([mesh], [{ meshIndex: 0, instanceIndex: 0, objectToWorld }]);
            const tracer = createCpuTracer(scene);
            const { rays } = sharedEdgeRays(20000, { objectToWorld, seed: 4 });
            let slipped = 0;
            for (const ray of rays) if (tracer.closest(ray) === null || !tracer.any(ray)) slipped++;
            expect(slipped).toBe(0);
        }
    });
});

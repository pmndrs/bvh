// The any-hit child order (`anyHitOrder`, docs/internals/design/trace.md, "Any-hit child order"):
// one comparison in the same walk, the same answers in every order.
import { describe, expect, it } from 'vitest';
import { createTinyBvhBuilder } from '../index.ts';
import { createCpuTracer, newTraceStats } from './cpu-traversal.ts';
import { buildMeshBvhWithTinyBvh } from './mesh-bvh-topology.ts';
import { createRandom, randomRays, randomTriangleSoup } from './oracle-fixtures.ts';
import { assembleTraceScene } from './scene-assembly.ts';
import { batchKernelWgsl, traceModuleWgsl, type AnyHitOrder } from './wgsl.ts';

const ORDERS: readonly AnyHitOrder[] = ['far', 'near', 'larger'];
const tinyBvh = await createTinyBvhBuilder();
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function walkText(module: string, name: string): string {
    const start = module.indexOf(`fn ${name}(`);
    return module.slice(start, module.indexOf('\n}\n', start));
}

describe('anyHitOrder in the WGSL', () => {
    it("changes only any-hit's child comparison (and 'larger' adds the area helper); 'far' is the default", () => {
        const far = traceModuleWgsl();
        expect(traceModuleWgsl({ anyHitOrder: 'far' })).toBe(far);
        for (const order of ['near', 'larger'] as const) {
            const module = traceModuleWgsl({ anyHitOrder: order });
            expect(walkText(module, 'bvhWalkClosest')).toBe(walkText(far, 'bvhWalkClosest'));
            const lines = (text: string) => text.split('\n');
            const before = lines(walkText(far, 'bvhWalkAny'));
            const after = lines(walkText(module, 'bvhWalkAny'));
            expect(after).toHaveLength(before.length);
            const changed = after.filter((line, k) => line !== before[k]);
            expect(changed).toHaveLength(1);
            expect(changed[0]).toContain(order === 'near' ? 'leftEntry <= rightEntry' : 'bvhBoxHalfArea(node.leftMin, node.leftMax) >= bvhBoxHalfArea(node.rightMin, node.rightMax)');
            expect(module.includes('fn bvhBoxHalfArea(')).toBe(order === 'larger');
        }
        expect(batchKernelWgsl('any', { anyHitOrder: 'larger' })).toContain('fn bvhBoxHalfArea(');
    });

    it('refuses an unknown order', () => {
        expect(() => traceModuleWgsl({ anyHitOrder: 'nearest' as AnyHitOrder })).toThrow(RangeError);
    });
});

describe('anyHitOrder in the CPU walk', () => {
    it('blocks the same rays in every order, and only the work changes', () => {
        const geometry = randomTriangleSoup(createRandom(11), 800, 5, 1);
        const scene = assembleTraceScene([buildMeshBvhWithTinyBvh(geometry, tinyBvh)], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY }]);
        const rays = randomRays(createRandom(12), 600, [-5, -5, -5, 5, 5, 5], 3, { randomInterval: true });
        const answers = ORDERS.map((anyHitOrder) => {
            const tracer = createCpuTracer(scene, { anyHitOrder });
            const stats = newTraceStats();
            return { blocked: rays.map((ray) => tracer.any(ray, { stats })), nodes: stats.nodesVisited };
        });
        expect(answers[1].blocked).toEqual(answers[0].blocked);
        expect(answers[2].blocked).toEqual(answers[0].blocked);
        expect(answers.some((answer) => answer.nodes !== answers[0].nodes)).toBe(true);
    });

    it("'larger' enters a big occluder before small clutter that a ray from outside meets first", () => {
        // A large wall behind a cloud of small triangles nearer the ray's origin: every ray is blocked by the wall.
        const random = createRandom(13);
        const clutter = randomTriangleSoup(random, 400, 1, 0.05);
        const wall = new Float32Array([-50, -50, 3, 50, -50, 3, 0, 50, 3]);
        const positions = new Float32Array(clutter.positions.length + wall.length);
        positions.set(clutter.positions);
        positions.set(wall, clutter.positions.length);
        const scene = assembleTraceScene([buildMeshBvhWithTinyBvh({ positions }, tinyBvh)], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY }]);
        const rays = Array.from({ length: 200 }, () => ({
            origin: [(random() - 0.5) * 2, (random() - 0.5) * 2, -10] as [number, number, number],
            direction: [0, 0, 1] as [number, number, number],
            tMin: 0,
            tMax: 100,
        }));
        const work = ORDERS.map((anyHitOrder) => {
            const tracer = createCpuTracer(scene, { anyHitOrder });
            const stats = newTraceStats();
            expect(rays.every((ray) => tracer.any(ray, { stats }))).toBe(true);
            return stats.nodesVisited + stats.trianglesTested;
        });
        expect(work[2]).toBeLessThan(work[1]);
    });
});

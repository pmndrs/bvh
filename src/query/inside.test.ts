/**
 * The CPU inside test (`CpuMeshBvh.rayCrossings`, `insideVotes`,
 * `isPointInside`; the twin of the GPU's `bvhRayCrossings` and `bvhIsInside`)
 * on closed, welded meshes whose inside is known: a box and a sphere, nested
 * and overlapping shells, an inverted box, a box with its top missing, and the
 * crossings against a brute-force count of every triangle.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import type { MeshInput } from '../mesh-input.ts';
import { createTinyBvhBuilder, type TinyBvhBuilder } from '../tinybvh.ts';
import { createRandom } from '../trace/oracle-fixtures.ts';
import { INSIDE_TEST_DIRECTIONS } from '../trace/query-wgsl.ts';
import { CpuMeshBvh, insideVotesOf } from './cpu-mesh-bvh.ts';
import { closedBox, closedSphere, inverted, merge } from '../../tests/support/solids.ts';

/** Every triangle the ray crosses (Möller-Trumbore in doubles), and their signed sum: the brute-force reference. */
function bruteCrossings(mesh: MeshInput, origin: readonly number[], direction: readonly number[]): { count: number; winding: number } {
    const p = mesh.positions;
    const index = mesh.indices!;
    let count = 0;
    let winding = 0;
    const sub = (a: number[], b: number[]) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const dot = (a: number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    for (let t = 0; t < index.length; t += 3) {
        const [a, b, c] = [0, 1, 2].map((k) => [p[index[t + k] * 3], p[index[t + k] * 3 + 1], p[index[t + k] * 3 + 2]]);
        const e1 = sub(b, a);
        const e2 = sub(c, a);
        const h = cross(direction as number[], e2);
        const det = dot(e1, h);
        if (det === 0) continue;
        const s = sub(origin as number[], a);
        const u = dot(s, h) / det;
        const q = cross(s, e1);
        const v = dot(q, direction) / det;
        const along = dot(e2, q) / det;
        if (u < 0 || v < 0 || u + v > 1 || along < 0) continue;
        count++;
        winding += dot(cross(e1, e2), direction) > 0 ? 1 : -1;
    }
    return { count, winding };
}

let tiny: TinyBvhBuilder;
beforeAll(async () => {
    tiny = await createTinyBvhBuilder();
});

const tree = (mesh: MeshInput) => CpuMeshBvh.build(mesh, tiny);

describe('CpuMeshBvh inside test', () => {
    it('casts three far-apart rays in different octants, exact f32 numbers', () => {
        expect(INSIDE_TEST_DIRECTIONS).toHaveLength(3);
        for (const d of INSIDE_TEST_DIRECTIONS) {
            expect(Math.hypot(...d)).toBeCloseTo(1, 6);
            for (const v of d) expect(Math.fround(v)).toBe(v);
        }
        const octants = new Set(INSIDE_TEST_DIRECTIONS.map((d) => d.map((v) => (v < 0 ? '-' : '+')).join('')));
        expect(octants.size).toBe(3);
    });

    it('finds a closed box inside and outside, by either rule, and counts crossings as the brute-force reference', () => {
        const mesh = merge(closedBox([-1, -1, -1], [1, 1, 1]));
        const bvh = tree(mesh);
        const random = createRandom(7);
        for (let i = 0; i < 400; i++) {
            const point = [0, 1, 2].map(() => (random() - 0.5) * 4) as [number, number, number];
            // Away from the surface only: a point within rounding of a face may go either way.
            const margin = Math.min(...point.map((v) => Math.abs(Math.abs(v) - 1)));
            if (margin < 1e-3) continue;
            const inside = point.every((v) => Math.abs(v) < 1);
            expect(bvh.isPointInside(point)).toBe(inside);
            expect(bvh.isPointInside(point, 'winding')).toBe(inside);
            expect(bvh.insideVotes(point)).toEqual(inside ? { parity: 3, winding: 3 } : { parity: 0, winding: 0 });
            for (const direction of INSIDE_TEST_DIRECTIONS) expect(bvh.rayCrossings(point, direction)).toEqual(bruteCrossings(mesh, point, direction));
        }
        // From the centre each ray leaves through one front face: one crossing, winding +1.
        expect(bvh.rayCrossings([0, 0, 0], INSIDE_TEST_DIRECTIONS[0])).toEqual({ count: 1, winding: 1 });
        // From outside through the box: in through a front face, out through another.
        expect(bvh.rayCrossings([-3, -1.5, -2], INSIDE_TEST_DIRECTIONS[0])).toEqual({ count: 2, winding: 0 });
    });

    it('a welded sphere: inside by its radius, each ray from its centre crossing once', () => {
        const mesh = merge(closedSphere(1.5, 3, [0.25, -0.5, 0.1]));
        const bvh = tree(mesh);
        const random = createRandom(11);
        // Inscribed radius of a 3-times split octahedron on the unit sphere: above 0.97.
        for (let i = 0; i < 400; i++) {
            const point = [0.25 + (random() - 0.5) * 4, -0.5 + (random() - 0.5) * 4, 0.1 + (random() - 0.5) * 4] as [number, number, number];
            const r = Math.hypot(point[0] - 0.25, point[1] + 0.5, point[2] - 0.1);
            if (r > 1.5 * 0.97 && r < 1.5 * 1.001) continue;
            expect(bvh.isPointInside(point)).toBe(r < 1.5);
        }
        // From the centre every ray leaves once.
        for (const direction of INSIDE_TEST_DIRECTIONS) expect(bvh.rayCrossings([0.25, -0.5, 0.1], direction)).toEqual({ count: 1, winding: 1 });
    });

    it('nested and overlapping shells: parity counts layers, winding counts the solid', () => {
        // Two boxes overlapping in x from 0 to 1.
        const bvh = tree(merge(closedBox([-1, -1, -1], [1, 1, 1]), closedBox([0, -0.5, -0.5], [2, 0.5, 0.5])));
        expect(bvh.isPointInside([0.5, 0, 0])).toBe(false); // two shells: even
        expect(bvh.isPointInside([0.5, 0, 0], 'winding')).toBe(true); // winding 2
        expect(bvh.insideVotes([0.5, 0, 0])).toEqual({ parity: 0, winding: 3 });
        expect(bvh.isPointInside([1.5, 0, 0])).toBe(true);
        expect(bvh.isPointInside([-0.5, 0, 0])).toBe(true);
        // A hollow: a box inside a box, the inner one wound inward (a cavity).
        const hollow = tree(merge(closedBox([-2, -2, -2], [2, 2, 2]), inverted(closedBox([-1, -1, -1], [1, 1, 1]))));
        expect(hollow.isPointInside([0, 0, 0])).toBe(false);
        expect(hollow.isPointInside([0, 0, 0], 'winding')).toBe(false);
        expect(hollow.isPointInside([1.5, 0.2, 0.3])).toBe(true);
        expect(hollow.isPointInside([1.5, 0.2, 0.3], 'winding')).toBe(true);
    });

    it('an inverted mesh is still solid by either rule (winding -1 is not 0)', () => {
        const bvh = tree(merge(inverted(closedBox([-1, -1, -1], [1, 1, 1]))));
        expect(bvh.rayCrossings([0, 0, 0], INSIDE_TEST_DIRECTIONS[1])).toEqual({ count: 1, winding: -1 });
        expect(bvh.isPointInside([0, 0, 0])).toBe(true);
        expect(bvh.isPointInside([0, 0, 0], 'winding')).toBe(true);
    });

    it('a box open at the top: the vote of three rays still finds its middle inside', () => {
        const box = closedBox([-1, -1, -1], [1, 1, 1]);
        const open = { positions: box.positions, indices: box.indices.filter((_, i) => Math.floor(i / 6) !== 1) }; // drop +z
        const bvh = tree(merge(open));
        const votes = bvh.insideVotes([0, 0, 0]);
        expect(votes.parity).toBe(2);
        expect(bvh.isPointInside([0, 0, 0])).toBe(true);
        expect(bvh.isPointInside([0, 0, 3])).toBe(false);
    });

    it('insideVotesOf adds several meshes up as one solid', () => {
        const a = tree(merge(closedBox([-1, -1, -1], [1, 1, 1])));
        const b = tree(merge(closedBox([0, -0.5, -0.5], [2, 0.5, 0.5])));
        const crossings = (origin: readonly [number, number, number], direction: readonly [number, number, number]) => {
            const x = a.rayCrossings(origin, direction);
            const y = b.rayCrossings(origin, direction);
            return { count: x.count + y.count, winding: x.winding + y.winding };
        };
        expect(insideVotesOf([0.5, 0, 0], crossings)).toEqual({ parity: 0, winding: 3 });
        expect(insideVotesOf({ x: 1.5, y: 0, z: 0 }, crossings)).toEqual({ parity: 3, winding: 3 });
    });
});

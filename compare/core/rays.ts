/**
 * The rays and query shapes, from fixed seeds: every library gets the very
 * same numbers, in Node and in the browser.
 *
 * - `outside`: from a sphere around the mesh (radius one diagonal) toward a
 *   random point in its box: the picking case for a model seen from outside.
 * - `inside`: from a random point in the box (shrunk by 10% on each side) in a
 *   random direction: the case for rooms and levels (Sponza, the furnished
 *   room), where every ray starts inside the walls.
 */

import { createRandom } from '@pmndrs/bvh';
import type { CompareGeometry, CompareRay, Vec3 } from '../adapters/types.ts';
import { diagonal } from './geometry.ts';

export type RaySet = 'outside' | 'inside';
export const RAY_SETS: readonly RaySet[] = ['outside', 'inside'];

function unitSphere(random: () => number): Vec3 {
    const z = random() * 2 - 1;
    const a = random() * Math.PI * 2;
    const r = Math.sqrt(1 - z * z);
    return [r * Math.cos(a), r * Math.sin(a), z];
}

function pointIn(bounds: readonly number[], random: () => number, shrink = 0): Vec3 {
    const out: Vec3 = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
        const size = bounds[k + 3] - bounds[k];
        out[k] = bounds[k] + size * (shrink + random() * (1 - 2 * shrink));
    }
    return out;
}

export function makeRays(geometry: CompareGeometry, set: RaySet, count: number, seed = 1): CompareRay[] {
    const random = createRandom(seed * 7919 + (set === 'outside' ? 1 : 2));
    const b = geometry.bounds;
    const center: Vec3 = [(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2];
    const radius = diagonal(geometry);
    const rays: CompareRay[] = [];
    for (let i = 0; i < count; i++) {
        if (set === 'outside') {
            const s = unitSphere(random);
            const origin: Vec3 = [center[0] + s[0] * radius, center[1] + s[1] * radius, center[2] + s[2] * radius];
            const target = pointIn(b, random);
            const d: Vec3 = [target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]];
            const length = Math.hypot(d[0], d[1], d[2]);
            rays.push({ origin, direction: [d[0] / length, d[1] / length, d[2] / length] });
        } else {
            rays.push({ origin: pointIn(b, random, 0.1), direction: unitSphere(random) });
        }
    }
    return rays;
}

/** Axis-aligned boxes around random points of the mesh's box, edges of 1% to 4% of its diagonal. */
export function makeBoxes(geometry: CompareGeometry, count: number, seed = 3): number[][] {
    const random = createRandom(seed);
    const d = diagonal(geometry);
    return Array.from({ length: count }, () => {
        const c = pointIn(geometry.bounds, random);
        const h = [0, 1, 2].map(() => (0.01 + random() * 0.03) * d * 0.5);
        return [c[0] - h[0], c[1] - h[1], c[2] - h[2], c[0] + h[0], c[1] + h[1], c[2] + h[2]];
    });
}

/** Spheres around random points of the mesh's box, radius 0.5% to 2% of its diagonal. */
export function makeSpheres(geometry: CompareGeometry, count: number, seed = 4): { center: Vec3; radius: number }[] {
    const random = createRandom(seed);
    const d = diagonal(geometry);
    return Array.from({ length: count }, () => ({ center: pointIn(geometry.bounds, random), radius: (0.005 + random() * 0.015) * d }));
}

/** Points in the mesh's box grown by 10% on each side, for closest-point queries. */
export function makePoints(geometry: CompareGeometry, count: number, seed = 5): Vec3[] {
    const random = createRandom(seed);
    const b = geometry.bounds;
    const grown = [0, 1, 2].flatMap((k) => [b[k] - (b[k + 3] - b[k]) * 0.1, b[k + 3] + (b[k + 3] - b[k]) * 0.1]);
    const box = [grown[0], grown[2], grown[4], grown[1], grown[3], grown[5]];
    return Array.from({ length: count }, () => pointIn(box, random));
}

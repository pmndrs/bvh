/**
 * The vanilla pages' smoke check: GPU hits against the package's CPU port of
 * the same walk (`createCpuTracer` over the same assembled scene). The two
 * walks agree hit for hit on every GPU the oracle runs on. The one difference
 * allowed is a ray through an edge shared by two triangles: both are hit at
 * the same distance, and either may be reported. A hit against a miss, or a
 * hit at another distance, fails the check.
 *
 * Rays the GPU made itself (a ray-cast kernel's camera rays) can differ from
 * the CPU's copy by an ulp. For those, `perturb` retraces a differing ray on
 * the CPU with its direction nudged by that much on each axis: a difference
 * that a nudge reproduces is a graze along a silhouette, counted and allowed;
 * any other fails.
 */

import { RAY_FLOATS, type CpuHit, type CpuTracer, type TraceRay } from '@pmndrs/bvh';

export interface HitLike {
    readonly t: number;
    readonly instance: number;
    readonly triangle: number;
}

export interface Agreement {
    readonly rays: number;
    readonly hits: number;
    /** Same instance and triangle at the same distance (or both missed). */
    readonly agree: number;
    /** The same distance on another triangle: a ray through a shared edge. */
    readonly sharedEdge: number;
    /** Explained by a nudge of the ray (only with `perturb`). */
    readonly grazes: number;
}

const sameT = (a: number, b: number): boolean => Math.abs(a - b) <= 1e-4 * Math.max(1, b);

/** Compares `gpu(i)` with the CPU walk of packed ray `i`; throws on any difference other than those above. */
export function checkAgainstCpu(packed: Float32Array, gpu: (index: number) => HitLike | null, cpu: CpuTracer, options: { readonly perturb?: number } = {}): Agreement {
    const rays = packed.length / RAY_FLOATS;
    let hits = 0;
    let agree = 0;
    let sharedEdge = 0;
    let grazes = 0;
    const differences: string[] = [];
    const matches = (got: HitLike | null, expected: CpuHit | null): 'same' | 'edge' | 'differ' => {
        if (expected === null || got === null) return expected === got ? 'same' : 'differ';
        if (!sameT(got.t, expected.t)) return 'differ';
        return got.instance === expected.instanceIndex && got.triangle === expected.triangleIndex ? 'same' : 'edge';
    };
    for (let i = 0; i < rays; i++) {
        const o = i * RAY_FLOATS;
        const ray: TraceRay = { origin: [packed[o], packed[o + 1], packed[o + 2]], tMin: packed[o + 3], direction: [packed[o + 4], packed[o + 5], packed[o + 6]], tMax: packed[o + 7] };
        const expected = cpu.closest(ray);
        const got = gpu(i);
        if (expected) hits++;
        const verdict = matches(got, expected);
        if (verdict === 'same') agree++;
        else if (verdict === 'edge') sharedEdge++;
        else if (options.perturb && nudged(ray, options.perturb).some((other) => matches(got, cpu.closest(other)) !== 'differ')) grazes++;
        else if (differences.length < 4) differences.push(`ray ${i}: GPU ${got ? `${got.instance}/${got.triangle} at ${got.t}` : 'miss'}, CPU ${expected ? `${expected.instanceIndex}/${expected.triangleIndex} at ${expected.t}` : 'miss'}`);
        else differences.push('');
    }
    if (differences.length) throw new Error(`${differences.length} of ${rays} GPU hits differ from the CPU walk: ${differences.filter(Boolean).join('; ')}`);
    if (hits === 0) throw new Error(`none of ${rays} check rays hit the scene`);
    return { rays, hits, agree, sharedEdge, grazes };
}

/** The ray with its direction nudged by `amount` along each axis, both ways. */
function nudged(ray: TraceRay, amount: number): TraceRay[] {
    const out: TraceRay[] = [];
    for (let axis = 0; axis < 3; axis++) {
        for (const sign of [-1, 1]) {
            const direction: [number, number, number] = [ray.direction[0], ray.direction[1], ray.direction[2]];
            direction[axis] += sign * amount;
            out.push({ ...ray, direction });
        }
    }
    return out;
}

/**
 * The vanilla pages' smoke check: GPU hits against the package's CPU port of
 * the same walk (`createCpuTracer` over the same assembled scene). The two
 * walks agree hit for hit on every GPU the oracle runs on; a ray that grazes
 * an edge may land on the neighbouring triangle, so a few differences are
 * allowed, never a hit against a miss in open space.
 */

import { RAY_FLOATS, type CpuTracer } from '@pmndrs/bvh';

export interface HitLike {
    readonly t: number;
    readonly instance: number;
    readonly triangle: number;
}

export interface Agreement {
    readonly rays: number;
    readonly hits: number;
    /** Same instance and triangle (or both missed). */
    readonly agree: number;
    readonly differ: number;
}

/** Compares `gpu(i)` with the CPU walk of packed ray `i`; throws when more than `allowed` (a fraction) differ. */
export function checkAgainstCpu(packed: Float32Array, gpu: (index: number) => HitLike | null, cpu: CpuTracer, allowed = 0.01): Agreement {
    const rays = packed.length / RAY_FLOATS;
    let hits = 0;
    let agree = 0;
    const differences: string[] = [];
    for (let i = 0; i < rays; i++) {
        const o = i * RAY_FLOATS;
        const ray = { origin: [packed[o], packed[o + 1], packed[o + 2]] as [number, number, number], tMin: packed[o + 3], direction: [packed[o + 4], packed[o + 5], packed[o + 6]] as [number, number, number], tMax: packed[o + 7] };
        const expected = cpu.closest(ray);
        const got = gpu(i);
        if (expected) hits++;
        const same = expected === null ? got === null : got !== null && got.instance === expected.instanceIndex && got.triangle === expected.triangleIndex && Math.abs(got.t - expected.t) <= 1e-4 * Math.max(1, expected.t);
        if (same) agree++;
        else if (differences.length < 4) differences.push(`ray ${i}: GPU ${got ? `${got.instance}/${got.triangle} at ${got.t}` : 'miss'}, CPU ${expected ? `${expected.instanceIndex}/${expected.triangleIndex} at ${expected.t}` : 'miss'}`);
    }
    const differ = rays - agree;
    if (differ > Math.floor(rays * allowed)) throw new Error(`${differ} of ${rays} GPU hits differ from the CPU walk: ${differences.join('; ')}`);
    if (hits === 0) throw new Error(`none of ${rays} check rays hit the scene`);
    return { rays, hits, agree, differ };
}

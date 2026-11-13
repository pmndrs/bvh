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
import type { CameraBasis } from './camera.ts';
import type { RayCastView } from './raycast-view.ts';

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

/** The rays of a `columns` x `rows` grid of pixels of a `width` x `height` ray-cast view, as its kernel makes them, and where each lands. */
export function viewCheckRays(basis: CameraBasis, width: number, height: number, columns = 32, rows = 32): { readonly packed: Float32Array; readonly at: readonly (readonly [number, number])[] } {
    const packed = new Float32Array(columns * rows * RAY_FLOATS);
    const at: [number, number][] = [];
    for (let j = 0; j < rows; j++) {
        for (let i = 0; i < columns; i++) {
            const x = Math.floor(((i + 0.5) / columns) * width);
            const y = Math.floor(((j + 0.5) / rows) * height);
            const ndc = [((x + 0.5) / width) * 2 - 1, 1 - ((y + 0.5) / height) * 2];
            const d = [0, 1, 2].map((k) => basis.forward[k] + basis.right[k] * ndc[0] + basis.up[k] * ndc[1]);
            const length = Math.hypot(d[0], d[1], d[2]);
            packed.set([...basis.origin, 0, d[0] / length, d[1] / length, d[2] / length, 1e30], at.length * RAY_FLOATS);
            at.push([x, y]);
        }
    }
    return { packed, at };
}

/**
 * The last traced frame of `view` against the CPU walk over `cpu`'s scene:
 * 1,024 pixels (the kernel's own camera rays, so ulp-level grazes along
 * silhouettes are allowed, as in `vanilla/raytrace`). `basis` must be the
 * camera of that frame (hold the camera still while checking).
 */
export async function checkViewAgainstCpu(view: RayCastView, basis: (aspect: number) => CameraBasis, cpu: CpuTracer): Promise<Agreement> {
    const pixels = await view.readPixels();
    const { packed, at } = viewCheckRays(basis(pixels.width / pixels.height), pixels.width, pixels.height);
    return checkAgainstCpu(packed, (index) => pixels.hit(at[index][0], at[index][1]), cpu, { perturb: 1e-6 });
}

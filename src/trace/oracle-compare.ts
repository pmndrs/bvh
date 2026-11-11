/**
 * Compares a tracer's answer with the brute-force reference.
 *
 * The tracers run in f32 and the reference in doubles, so a few rays that land
 * within rounding distance of an edge may legitimately disagree. Those come
 * back as `ambiguous` (counted and reported) instead of failing. Anything
 * else that differs is a `mismatch`.
 */

import { REFERENCE_TOLERANCE, type ReferenceHit, type ReferenceOcclusion, type TraceRay } from './reference-tracer.ts';

/** Largest accepted difference in hit distance, relative to max(1, t). f32 error is around 1e-6. */
export const T_TOLERANCE = 1e-4;

/** Smallest accepted cosine between a tracer's hit normal and the reference's. f32 corners and transforms stay far above it. */
export const NORMAL_COSINE_TOLERANCE = 0.9999;

/** What a tracer returned for one ray. `t < 0` or `hit: false` means a miss. */
export interface TracedHit {
    readonly hit: boolean;
    readonly t: number;
    readonly instanceIndex: number;
    readonly triangleIndex: number;
    readonly u: number;
    readonly v: number;
    readonly frontFace: boolean;
    /** Compared when given: the hit's material index (`NO_MATERIAL` for none). */
    readonly material?: number;
    /** Compared when given: the world-space geometric normal of the authored front side (unit). */
    readonly normal?: readonly [number, number, number];
}

export interface Comparison {
    readonly verdict: 'match' | 'ambiguous' | 'mismatch';
    /** Why, for mismatches and ambiguous cases. */
    readonly reason?: string;
    /** Relative t difference when both found a hit. */
    readonly tError?: number;
}

export function compareClosest(actual: TracedHit | null, expected: ReferenceHit | null, ray: TraceRay): Comparison {
    const actualHit = actual !== null && actual.hit;
    if (!actualHit && expected === null) return { verdict: 'match' };

    if (actualHit && expected === null) {
        const margin = Math.min(actual.u, actual.v, 1 - actual.u - actual.v);
        const slack = REFERENCE_TOLERANCE * Math.max(1, actual.t);
        const nearEdge = margin < REFERENCE_TOLERANCE;
        const nearInterval = Math.abs(actual.t - ray.tMin) <= slack || Math.abs(actual.t - ray.tMax) <= slack;
        return nearEdge || nearInterval
            ? { verdict: 'ambiguous', reason: 'tracer hit at an edge or interval end; reference found nothing' }
            : { verdict: 'mismatch', reason: `tracer hit at t=${actual.t} where the reference found nothing` };
    }

    if (!actualHit && expected !== null) {
        return expected.ambiguous
            ? { verdict: 'ambiguous', reason: 'tracer missed at an edge or interval end' }
            : { verdict: 'mismatch', reason: `tracer missed; reference hit instance ${expected.instanceIndex} triangle ${expected.triangleIndex} at t=${expected.t}` };
    }

    // Both found something.
    const hit = actual as TracedHit;
    const reference = expected as ReferenceHit;
    const tError = Math.abs(hit.t - reference.t) / Math.max(1, reference.t);
    const sameTriangle = hit.instanceIndex === reference.instanceIndex && hit.triangleIndex === reference.triangleIndex;
    const problems: string[] = [];
    if (!sameTriangle) problems.push(`triangle ${hit.instanceIndex}/${hit.triangleIndex} vs ${reference.instanceIndex}/${reference.triangleIndex}`);
    if (tError > T_TOLERANCE) problems.push(`t ${hit.t} vs ${reference.t}`);
    if (sameTriangle && hit.frontFace !== reference.frontFace) problems.push(`frontFace ${hit.frontFace} vs ${reference.frontFace}`);
    if (sameTriangle && (Math.abs(hit.u - reference.u) > REFERENCE_TOLERANCE || Math.abs(hit.v - reference.v) > REFERENCE_TOLERANCE)) {
        problems.push(`barycentrics (${hit.u}, ${hit.v}) vs (${reference.u}, ${reference.v})`);
    }
    if (sameTriangle && hit.material !== undefined && hit.material !== reference.material) problems.push(`material ${hit.material} vs ${reference.material}`);
    if (sameTriangle && hit.normal !== undefined && !(normalCosine(hit.normal, reference.normal) >= NORMAL_COSINE_TOLERANCE)) {
        problems.push(`normal (${hit.normal.join(', ')}) vs (${reference.normal.join(', ')})`);
    }
    if (problems.length === 0) return { verdict: 'match', tError };
    return reference.ambiguous
        ? { verdict: 'ambiguous', reason: problems.join('; '), tError }
        : { verdict: 'mismatch', reason: problems.join('; '), tError };
}

/** Cosine between two unit normals. */
export function normalCosine(a: readonly number[], b: readonly number[]): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function compareAny(actualBlocked: boolean, expected: ReferenceOcclusion): Comparison {
    if (actualBlocked === expected.blocked) return { verdict: 'match' };
    return expected.ambiguous
        ? { verdict: 'ambiguous', reason: 'blocker at an edge or interval end' }
        : { verdict: 'mismatch', reason: `tracer says ${actualBlocked ? 'blocked' : 'clear'}, reference says ${expected.blocked ? 'blocked' : 'clear'}` };
}

/** Running totals for a batch of comparisons. */
export interface ComparisonTally {
    rays: number;
    matches: number;
    ambiguous: number;
    mismatches: number;
    maxTError: number;
    /** The first few mismatches, for the report. */
    examples: string[];
}

export function newTally(): ComparisonTally {
    return { rays: 0, matches: 0, ambiguous: 0, mismatches: 0, maxTError: 0, examples: [] };
}

export function addToTally(tally: ComparisonTally, comparison: Comparison, label: string): void {
    tally.rays++;
    if (comparison.verdict === 'match') tally.matches++;
    else if (comparison.verdict === 'ambiguous') tally.ambiguous++;
    else {
        tally.mismatches++;
        if (tally.examples.length < 5) tally.examples.push(`${label}: ${comparison.reason}`);
    }
    if (comparison.tError !== undefined && comparison.verdict === 'match') {
        tally.maxTError = Math.max(tally.maxTError, comparison.tError);
    }
}

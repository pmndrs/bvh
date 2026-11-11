/**
 * Ray sets for the surface-ray bench, ported from the radiance-cache team's
 * `surface-ray-variants.ts` (webgpu-ddgi, RC5.4), trimmed to what the paired
 * instanced-against-merged measurement needs:
 *
 * - **surface**: RC1.5's screen-probe rays, 64 stratified cosine rays per
 *   probe from visible surfaces, closest hit, no length limit;
 * - **air-grid**: the probe-grid bench's rays, the same number of rays from a
 *   grid of points in the air, one shared sphere of directions;
 * - **uniform-hemisphere** (comes free with the generator, not run by the
 *   page): uniform directions over the hemisphere about the normal.
 *
 * Every function here is pure and CPU-only, so it can be unit-tested; the page
 * (surface-rays-bench.ts) makes the sets and times them.
 */

import { HIT_RECORD_BYTES, HIT_RECORD_WORD, RAY_FLOATS } from '../../../../src/trace/index.ts';
import type { Random } from '../../../../src/trace/oracle-fixtures.ts';
import { RAY_FAR } from './config.ts';
import { COUNTED_WORD, COUNTED_WORDS } from './counted-walk-wgsl.ts';
import type { BenchInstance, BenchMeshGeometry } from './gltf-geometry.ts';
import { cosineDirection, hitSurfaces, offsetRayOrigin, tangentFrame, writeRay } from './rc-rays.ts';
import type { Vec3 } from './rays.ts';

/** Screen probes with what the sets need: the point on the surface, its normal and the triangle it is on. */
export interface SurfaceProbes {
    /** Per probe, `x, y, z` on the triangle (moved slightly inside it), before any push off the surface. */
    positions: Float64Array;
    /** Per probe, the unit geometric normal on the camera's side. */
    normals: Float64Array;
    count: number;
}

/** The probes the placement rays found, one per tile that hit something. */
export function surfaceProbes(placementRays: Float32Array, hitRecords: ArrayBuffer, meshes: readonly BenchMeshGeometry[], instances: readonly BenchInstance[]): SurfaceProbes {
    const surfaces = hitSurfaces(placementRays, hitRecords, meshes, instances);
    return { positions: surfaces.positions, normals: surfaces.normals, count: surfaces.count };
}

/** Where each probe's rays start: the point pushed off the surface by the platform's 256 rounding steps. */
export function probeOrigins(probes: SurfaceProbes): Float64Array {
    const origins = new Float64Array(probes.count * 3);
    for (let probe = 0; probe < probes.count; probe++) {
        origins.set(offsetRayOrigin(probes.positions.subarray(probe * 3, probe * 3 + 3), probes.normals.subarray(probe * 3, probe * 3 + 3)), probe * 3);
    }
    return origins;
}

/**
 * How a probe's rays are spread:
 *
 * - `cosine-stratified`: RC1.5's probe rays: an 8x8 grid over the cosine
 *   hemisphere, jittered, turned by a random angle per probe;
 * - `uniform-hemisphere`: uniform over the hemisphere about the normal;
 * - `shared-sphere`: the probe-grid bench's directions: one fixed set of
 *   sphere directions, the same for every probe.
 */
export type DirectionKind = 'cosine-stratified' | 'uniform-hemisphere' | 'shared-sphere';

/** A uniform direction on the hemisphere about `normal` (cos θ uniform in [0, 1)). */
function uniformHemisphereDirection(normal: ArrayLike<number>, frame: { tangent: Vec3; bitangent: Vec3 }, u1: number, u2: number): Vec3 {
    const z = u1;
    const radius = Math.sqrt(Math.max(0, 1 - z * z));
    const angle = 2 * Math.PI * u2;
    const x = radius * Math.cos(angle);
    const y = radius * Math.sin(angle);
    const { tangent, bitangent } = frame;
    return [0, 1, 2].map((axis) => tangent[axis] * x + bitangent[axis] * y + normal[axis] * z) as Vec3;
}

export interface ProbeRayOptions {
    /** Per probe `x, y, z`. */
    origins: Float64Array;
    /** Per probe, the unit normal the hemisphere kinds are spread about. */
    normals: Float64Array;
    count: number;
    raysPerProbe: number;
    kind: DirectionKind;
    random: Random;
    /** `shared-sphere` only: `raysPerProbe` unit directions, `x, y, z` each. */
    sharedDirections?: Float32Array;
}

/**
 * `raysPerProbe` closest-hit rays per probe, probe-major, from `tMin` 0 to the
 * bench's far end. `cosine-stratified` consumes the generator exactly as the
 * radiance-cache team's `makeProbeRays` does, so with RC1.5's seed and probes
 * it makes RC1.5's probe set.
 */
export function makeVariantProbeRays(options: ProbeRayOptions): Float32Array {
    const { origins, normals, count, raysPerProbe, kind, random } = options;
    const side = Math.round(Math.sqrt(raysPerProbe));
    if (kind === 'cosine-stratified' && side * side !== raysPerProbe) throw new RangeError(`rays per probe must be a square number, got ${raysPerProbe}`);
    if (kind === 'shared-sphere' && (options.sharedDirections?.length ?? 0) !== raysPerProbe * 3) {
        throw new RangeError('shared-sphere needs one shared direction per ray of a probe');
    }
    const rays = new Float32Array(count * raysPerProbe * RAY_FLOATS);
    for (let probe = 0; probe < count; probe++) {
        const origin = origins.subarray(probe * 3, probe * 3 + 3);
        const normal = normals.subarray(probe * 3, probe * 3 + 3);
        const frame = tangentFrame(normal);
        const rotation = kind === 'cosine-stratified' ? random() : 0;
        for (let cell = 0; cell < raysPerProbe; cell++) {
            let direction: Vec3;
            if (kind === 'cosine-stratified') {
                const u1 = (Math.floor(cell / side) + random()) / side;
                const u2 = ((cell % side) + random()) / side + rotation;
                direction = cosineDirection(normal, frame, u1, u2 - Math.floor(u2));
            } else if (kind === 'uniform-hemisphere') {
                direction = uniformHemisphereDirection(normal, frame, random(), random());
            } else {
                const shared = options.sharedDirections as Float32Array;
                direction = [shared[cell * 3], shared[cell * 3 + 1], shared[cell * 3 + 2]];
            }
            writeRay(rays, probe * raysPerProbe + cell, origin, direction, 0, RAY_FAR);
        }
    }
    return rays;
}

/** Means of the counted walk's counters over a set. */
export interface CountSummary {
    rays: number;
    hitFraction: number;
    meanSteps: number;
    p90Steps: number;
    meanMeshNodes: number;
    meanTopLevelNodes: number;
    /** Instances entered per ray (a merged mesh counts once). */
    meanInstanceEntries: number;
    /** Instances entered in which the walk found nothing to keep. */
    meanInstanceEntriesWithoutHit: number;
    meanTriangleTests: number;
}

/** Summarises the counted kernel's output ({@link COUNTED_WORDS} words per ray). */
export function summariseCounts(counts: ArrayBuffer, rayCount: number): CountSummary {
    const words = new Uint32Array(counts);
    const floats = new Float32Array(counts);
    if (words.length < rayCount * COUNTED_WORDS) throw new RangeError('fewer counter records than rays');
    const sums = new Float64Array(COUNTED_WORDS);
    const steps = new Uint32Array(rayCount);
    let hits = 0;
    for (let ray = 0; ray < rayCount; ray++) {
        const base = ray * COUNTED_WORDS;
        for (let word = COUNTED_WORD.steps; word < COUNTED_WORDS; word++) sums[word] += words[base + word];
        steps[ray] = words[base + COUNTED_WORD.steps];
        // Closest: a negative t is a miss. Any: 1.0 is blocked, -1.0 clear.
        if (floats[base + COUNTED_WORD.t] >= 0) hits++;
    }
    const mean = (word: number): number => (rayCount > 0 ? sums[word] / rayCount : 0);
    const sorted = steps.slice().sort();
    return {
        rays: rayCount,
        hitFraction: rayCount > 0 ? hits / rayCount : 0,
        meanSteps: mean(COUNTED_WORD.steps),
        p90Steps: sorted.length > 0 ? sorted[Math.min(sorted.length - 1, Math.floor(0.9 * sorted.length))] : Number.NaN,
        meanMeshNodes: mean(COUNTED_WORD.meshNodes),
        meanTopLevelNodes: mean(COUNTED_WORD.topLevelNodes),
        meanInstanceEntries: mean(COUNTED_WORD.instanceEntries),
        meanInstanceEntriesWithoutHit: mean(COUNTED_WORD.instanceEntries) - mean(COUNTED_WORD.instancesWithHit),
        meanTriangleTests: mean(COUNTED_WORD.triangleTests),
    };
}

/** What two trees answered for the same rays. */
export interface HitAgreement {
    rays: number;
    /** Rays where both missed, or both hit the same instance and triangle. */
    exact: number;
    /** Rays that hit in both but on different triangles (shared edges, coplanar overlaps) at the same distance. */
    sameDistance: number;
    /** Rays where one hit and the other missed. */
    hitMiss: number;
    /** Everything else: different triangle at a different distance. */
    other: number;
}

/** Relative distance within which two different triangles count as a tie. */
const TIE_RELATIVE_T = 1e-4;

/** Compares two arrays of `BvhHitRecord`s ray by ray: the instance and triangle must match exactly. */
export function compareHitRecords(a: ArrayBuffer, b: ArrayBuffer, count: number, stride = 1): HitAgreement {
    const recordWords = HIT_RECORD_BYTES / 4;
    const floatsA = new Float32Array(a);
    const floatsB = new Float32Array(b);
    const wordsA = new Uint32Array(a);
    const wordsB = new Uint32Array(b);
    const result: HitAgreement = { rays: 0, exact: 0, sameDistance: 0, hitMiss: 0, other: 0 };
    for (let ray = 0; ray < count; ray += stride) {
        const base = ray * recordWords;
        const tA = floatsA[base + HIT_RECORD_WORD.t];
        const tB = floatsB[base + HIT_RECORD_WORD.t];
        result.rays++;
        if (tA < 0 && tB < 0) result.exact++;
        else if (tA < 0 || tB < 0) result.hitMiss++;
        else if (wordsA[base + HIT_RECORD_WORD.instance] === wordsB[base + HIT_RECORD_WORD.instance] && wordsA[base + HIT_RECORD_WORD.triangle] === wordsB[base + HIT_RECORD_WORD.triangle]) {
            result.exact++;
        } else if (Math.abs(tA - tB) <= TIE_RELATIVE_T * Math.max(1, tA)) result.sameDistance++;
        else result.other++;
    }
    return result;
}

/** Share of rays (every `stride`-th) with the same blocked flag in two any-hit answers. */
export function compareBlocked(a: ArrayBuffer, b: ArrayBuffer, count: number, stride = 1): { rays: number; agree: number } {
    const blockedA = new Uint32Array(a);
    const blockedB = new Uint32Array(b);
    let rays = 0;
    let agree = 0;
    for (let ray = 0; ray < count; ray += stride) {
        rays++;
        if ((blockedA[ray] !== 0) === (blockedB[ray] !== 0)) agree++;
    }
    return { rays, agree };
}

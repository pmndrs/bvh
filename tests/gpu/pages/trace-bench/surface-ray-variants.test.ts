/**
 * Tests of the surface-ray sets and of the counted walk (no GPU): the surface
 * set is the radiance-cache team's RC1.5 probe set, the air-grid set is the
 * probe-grid bench's, and the counted copy of the walk is the production walk
 * once its counters are taken out.
 */

import { describe, expect, it } from 'vitest';
import { HIT_RECORD_BYTES, HIT_RECORD_WORD, RAY_FLOATS } from '../../../../src/trace/index.ts';
import { createRandom } from '../../../../src/trace/oracle-fixtures.ts';
import { COUNTED_WORD, COUNTED_WORDS, countedKernelWgsl, countedWalkWgsl, productionWalkWgsl } from './counted-walk-wgsl.ts';
import { makePlacementRays, offsetRayOrigin, tileGrid } from './rc-rays.ts';
import { makeProbeGrid, makeProbeRays, makeSphereDirections } from './rays.ts';
import { compareBlocked, compareHitRecords, makeVariantProbeRays, probeOrigins, summariseCounts, type SurfaceProbes } from './surface-ray-variants.ts';

/** Three probes on a floor, a wall and a slope. */
function sampleProbes(): SurfaceProbes {
    const slope = Math.SQRT1_2;
    return {
        positions: Float64Array.from([1, 0, 2, 5, 1.5, -3, -2, 0.5, 0.25]),
        normals: Float64Array.from([0, 1, 0, -1, 0, 0, 0, slope, slope]),
        count: 3,
    };
}

describe('surface-ray sets', () => {
    it('puts a probe origin where the platform offset puts it', () => {
        const origins = probeOrigins(sampleProbes());
        expect(Array.from(origins.subarray(3, 6))).toEqual(offsetRayOrigin([5, 1.5, -3], [-1, 0, 0]));
    });

    it('makes unit directions in the interval [0, far] on the normal side for the stratified cosine kind', () => {
        const probes = sampleProbes();
        const rays = makeVariantProbeRays({ origins: probeOrigins(probes), normals: probes.normals, count: 3, raysPerProbe: 64, kind: 'cosine-stratified', random: createRandom(7) });
        expect(rays.length).toBe(3 * 64 * RAY_FLOATS);
        for (let ray = 0; ray < 192; ray++) {
            const base = ray * RAY_FLOATS;
            const probe = Math.floor(ray / 64);
            expect(Math.hypot(rays[base + 4], rays[base + 5], rays[base + 6])).toBeCloseTo(1, 6);
            expect(rays[base + 3]).toBe(0);
            const cosine = rays[base + 4] * probes.normals[probe * 3] + rays[base + 5] * probes.normals[probe * 3 + 1] + rays[base + 6] * probes.normals[probe * 3 + 2];
            expect(cosine).toBeGreaterThanOrEqual(0);
        }
    });

    it('is deterministic for one seed and differs between seeds', () => {
        const probes = sampleProbes();
        const make = (seed: number) => makeVariantProbeRays({ origins: probeOrigins(probes), normals: probes.normals, count: 3, raysPerProbe: 64, kind: 'cosine-stratified', random: createRandom(seed) });
        expect(Array.from(make(1))).toEqual(Array.from(make(1)));
        expect(Array.from(make(1))).not.toEqual(Array.from(make(2)));
    });

    it('keeps the uniform-hemisphere kind on the normal side', () => {
        const probes = sampleProbes();
        const rays = makeVariantProbeRays({ origins: probeOrigins(probes), normals: probes.normals, count: 3, raysPerProbe: 128, kind: 'uniform-hemisphere', random: createRandom(3) });
        for (let ray = 0; ray < 3 * 128; ray++) {
            const base = ray * RAY_FLOATS;
            const probe = Math.floor(ray / 128);
            const cosine = rays[base + 4] * probes.normals[probe * 3] + rays[base + 5] * probes.normals[probe * 3 + 1] + rays[base + 6] * probes.normals[probe * 3 + 2];
            expect(cosine).toBeGreaterThanOrEqual(-1e-6);
        }
    });

    it("gives every probe of the air-grid set the probe-grid bench's shared directions", () => {
        const grid = makeProbeGrid({ min: [0, 0, 0], max: [4, 2, 4] }, 8);
        const directions = makeSphereDirections(16, 5);
        const reference = makeProbeRays(grid, directions);
        const probes = { origins: new Float64Array(grid.probeCount * 3), normals: new Float64Array(grid.probeCount * 3), count: grid.probeCount };
        const same = makeVariantProbeRays({ ...probes, raysPerProbe: 16, kind: 'shared-sphere', random: createRandom(1), sharedDirections: directions });
        for (let ray = 0; ray < grid.probeCount * 16; ray++) {
            const base = ray * RAY_FLOATS;
            // Directions are the same; the origins are the grid's, set by the caller (here zero).
            for (let component = 4; component < 7; component++) expect(same[base + component]).toBeCloseTo(reference[base + component], 6);
        }
    });

    it('places one probe ray per 16x16 tile of a 1280x720 view', () => {
        expect(tileGrid({ width: 1280, height: 720, tileSize: 16 }).count).toBe(80 * 45);
        const rays = makePlacementRays({ position: [0, 1, 0], target: [1, 1, 0], fovYDeg: 70 }, { width: 1280, height: 720, tileSize: 16 }, 1);
        expect(rays.length).toBe(80 * 45 * RAY_FLOATS);
        expect(rays[4]).toBeGreaterThan(0.2);
    });
});

describe('hit comparison', () => {
    function records(entries: [number, number, number][]): ArrayBuffer {
        const buffer = new ArrayBuffer(entries.length * HIT_RECORD_BYTES);
        const floats = new Float32Array(buffer);
        const words = new Uint32Array(buffer);
        entries.forEach(([t, instance, triangle], ray) => {
            const base = (ray * HIT_RECORD_BYTES) / 4;
            floats[base + HIT_RECORD_WORD.t] = t;
            words[base + HIT_RECORD_WORD.instance] = instance;
            words[base + HIT_RECORD_WORD.triangle] = triangle;
        });
        return buffer;
    }

    it('counts exact matches, ties at one distance, hit/miss splits and other differences', () => {
        const a = records([[-1, 0, 0], [2, 3, 4], [2, 3, 4], [-1, 0, 0], [2, 3, 4]]);
        const b = records([[-1, 9, 9], [2, 3, 4], [2, 3, 5], [1, 0, 0], [3, 3, 5]]);
        expect(compareHitRecords(a, b, 5)).toEqual({ rays: 5, exact: 2, sameDistance: 1, hitMiss: 1, other: 1 });
        expect(compareHitRecords(a, b, 5, 2).rays).toBe(3);
    });

    it('compares blocked flags', () => {
        expect(compareBlocked(Uint32Array.from([1, 0, 1]).buffer, Uint32Array.from([1, 1, 5]).buffer, 3)).toEqual({ rays: 3, agree: 2 });
    });
});

describe('count summary', () => {
    it('gives means, the hit fraction and instance entries without a hit', () => {
        const rays = 64;
        const buffer = new ArrayBuffer(rays * COUNTED_WORDS * 4);
        const words = new Uint32Array(buffer);
        const floats = new Float32Array(buffer);
        for (let ray = 0; ray < rays; ray++) {
            const base = ray * COUNTED_WORDS;
            floats[base + COUNTED_WORD.t] = ray % 2 === 0 ? 2 : -1;
            words[base + COUNTED_WORD.steps] = ray % 32 === 0 ? 32 : 1;
            words[base + COUNTED_WORD.instanceEntries] = 3;
            words[base + COUNTED_WORD.instancesWithHit] = 1;
        }
        const summary = summariseCounts(buffer, rays);
        expect(summary.hitFraction).toBe(0.5);
        expect(summary.meanSteps).toBeCloseTo(63 / 32, 12);
        expect(summary.meanInstanceEntries).toBe(3);
        expect(summary.meanInstanceEntriesWithoutHit).toBe(2);
    });
});

/** Without comments, and with all whitespace runs as one space, so only code is compared. */
function codeOnly(text: string): string {
    return text
        .replace(/\/\/[^\n]*/g, '')
        .replace(/\s+/g, ' ')
        .trim();
}

/** The counted walk with the counters taken back out: their lines dropped, the finished returns unwrapped. */
function withoutCounters(text: string, productionSignature: string): string {
    return text
        .split('\n')
        .filter((line) => !/\bcounts\./.test(line) && !/hitInThisInstance/.test(line) && !/\bvar counts\b/.test(line))
        .join('\n')
        .replace(/benchFinish(?:Closest|Any)\(counts, ([^)]*)\)/g, '$1')
        .replace(/fn \w+\(ray: BvhRay, rayMask: u32\) -> BenchCounts \{/, productionSignature);
}

describe('counted walk', () => {
    for (const [kind, name, returnType] of [['closest', 'bvhWalkClosest', 'BvhHit'], ['any', 'bvhWalkAny', 'bool']] as const) {
        it(`is the production ${name} character for character with counters off`, () => {
            expect(countedWalkWgsl({ kind, name, counters: false })).toBe(productionWalkWgsl(kind));
        });

        it(`is the production ${name} once its counters are taken out`, () => {
            const counted = countedWalkWgsl({ kind, name, counters: true });
            expect(counted).not.toBe(productionWalkWgsl(kind));
            expect(counted).toContain('counts.steps += 1u;');
            expect(counted).toContain('counts.instanceEntries += 1u;');
            expect(counted).toContain('counts.triangleTests += 1u;');
            expect(codeOnly(withoutCounters(counted, `fn ${name}(ray: BvhRay, rayMask: u32) -> ${returnType} {`))).toBe(codeOnly(productionWalkWgsl(kind)));
        });
    }

    it('keeps the production walk the merged-instance select and the masks, which the copy inherits', () => {
        const walk = productionWalkWgsl('closest');
        expect(walk).toContain('bvhHitInstance(instance, triangle, slot)');
        expect(walk).toContain('bvhInstanceMask(instance) & rayMask');
    });

    it('makes whole kernels with one entry point each', () => {
        for (const mode of ['closest', 'any'] as const) {
            const kernel = countedKernelWgsl(mode);
            expect(kernel).toContain('fn benchCounted(');
            expect(kernel).toContain('fn benchCountedWalk(');
        }
    });
});

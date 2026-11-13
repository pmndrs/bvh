/** Tests of the paired method's schedule and ratios, and of the surface-ray CSV rows (no GPU). */

import { describe, expect, it } from 'vitest';
import { hitRecordRoundOrder, pairedRoundOrder, ratioOfRounds, type HitRecordCase, type PairKey } from './paired-ratio.ts';
import { surfaceRaysRows } from './surface-to-csv.ts';
import type { SurfaceRaysData } from './surface-rays-bench.ts';

const pairs: PairKey[] = [
    { set: 'surface', mode: 'closest' },
    { set: 'surface', mode: 'any' },
];

describe('paired schedule', () => {
    it('times the two trees of a pair back to back, instanced first on even rounds and merged first on odd ones', () => {
        const label = (round: number) => pairedRoundOrder(pairs, round).map((step) => `${step.pair.mode}:${step.tree}`);
        expect(label(0)).toEqual(['closest:instanced', 'closest:merged', 'any:instanced', 'any:merged']);
        expect(label(1)).toEqual(['any:merged', 'any:instanced', 'closest:merged', 'closest:instanced']);
        expect(label(2)).toEqual(label(0));
    });

    it('puts every tree before and after the other equally often over an even number of rounds', () => {
        let instancedFirst = 0;
        for (let round = 0; round < 6; round++) if (pairedRoundOrder(pairs, round)[0].tree === 'instanced') instancedFirst++;
        expect(instancedFirst).toBe(3);
    });
});

describe('hit-record schedule', () => {
    const cases: HitRecordCase[] = [
        { tree: 'instanced', set: 'surface' },
        { tree: 'merged', set: 'surface' },
    ];
    const label = (round: number) => hitRecordRoundOrder(cases, round).map((step) => `${step.case.tree}:${step.record}`);

    it('times the two records of a case back to back, the 40-byte one first on even rounds and the 24-byte one first on odd ones', () => {
        expect(label(0)).toEqual(['instanced:record40', 'instanced:record24', 'merged:record40', 'merged:record24']);
        expect(label(1)).toEqual(['merged:record24', 'merged:record40', 'instanced:record24', 'instanced:record40']);
        expect(label(2)).toEqual(label(0));
    });
});

describe('paired ratios', () => {
    it('divides within each round and reports the median and the spread', () => {
        const ratio = ratioOfRounds('r', [20, 18, 30], [10, 10, 10]);
        expect(ratio.perRound).toEqual([2, 1.8, 3]);
        expect(ratio.median).toBe(2);
        expect(ratio.min).toBe(1.8);
        expect(ratio.max).toBe(3);
    });

    it('is not fooled by drift that hits both trees alike', () => {
        // Both trees slow by 30% over the rounds; the merged one is 1.5x faster throughout.
        const instanced = [100, 90, 80, 70];
        const merged = instanced.map((rate) => rate * 1.5);
        const ratio = ratioOfRounds('r', merged, instanced);
        expect(ratio.min).toBeCloseTo(1.5, 12);
        expect(ratio.max).toBeCloseTo(1.5, 12);
    });

    it('refuses rounds of different lengths', () => {
        expect(() => ratioOfRounds('r', [1, 2], [1])).toThrow(RangeError);
    });
});

describe('surface-ray csv rows', () => {
    it('names the ratio, the rates, the counted work and the agreement', () => {
        const counts = { rays: 10, hitFraction: 0.9, meanSteps: 100, p90Steps: 150, meanMeshNodes: 60, meanTopLevelNodes: 20, meanInstanceEntries: 9.8, meanInstanceEntriesWithoutHit: 8.1, meanTriangleTests: 30 };
        const data = {
            benchmark: 'surface-rays',
            scene: 'sponza',
            instanceCount: 400,
            probesPlaced: 3000,
            gridProbes: 3000,
            raysPerSet: { surface: 192000 },
            trees: {
                instanced: { topLevelInstances: 400, topLevelNodes: 799, bytesNodes: 1, bytesTriangles: 2, bytesTopLevel: 3, buildMs: 4 },
                merged: { topLevelInstances: 3, topLevelNodes: 5, bytesNodes: 1, bytesTriangles: 2, bytesTopLevel: 3, buildMs: 4 },
            },
            sets: [{ tree: 'merged', set: 'surface', mode: 'closest', countedAgreement: 1, hitAgreement: { rays: 10, exact: 10, sameDistance: 0, hitMiss: 0, other: 0 }, blockedAgreement: null, ...counts }],
            paired: {
                rounds: 2,
                profile: 'full',
                entries: [{ tree: 'merged', set: 'surface', mode: 'closest', mraysPerRound: [300, 280], singleMraysPerRound: [250, 240], repeatMrays: 270 }],
                ratios: [{ name: 'merged_over_instanced/surface/closest', perRound: [1.9, 1.8], median: 1.9, min: 1.8, max: 1.9 }],
                repeatRatios: [{ name: 'merged_over_instanced/surface/closest', ratio: 1.85 }],
            },
            hitRecord: {
                rounds: 2,
                entries: [{ tree: 'merged', set: 'surface', record: 'record40', recordBytes: 40, mraysPerRound: [300, 280], singleMraysPerRound: [250, 240] }],
                ratios: [{ name: 'hit_record_40_over_24/merged/surface/closest', perRound: [0.97, 0.99], median: 0.98, min: 0.97, max: 0.99 }],
                agreement: [{ tree: 'merged', set: 'surface', rays: 10, same: 10 }],
            },
            anyHitOrder: {
                rounds: 2,
                entries: [{ tree: 'merged', set: 'surface', order: 'larger', mraysPerRound: [500, 520] }],
                ratios: [{ name: 'any_larger_over_far/merged/surface/any', perRound: [0.8, 0.76], median: 0.78, min: 0.76, max: 0.8 }],
                agreement: [{ tree: 'merged', set: 'surface', order: 'larger', rays: 10, same: 10 }],
                counts: [{ tree: 'merged', set: 'surface', order: 'larger', counts: { ...counts, meanSteps: 42 }, countedAgreement: 1 }],
            },
        } as unknown as SurfaceRaysData;
        const { timed, counted } = surfaceRaysRows(data);
        const metric = (name: string) => [...timed, ...counted].find((row) => row.metric === name)?.value;
        expect(metric('surface-rays/paired/ratio/surface/closest/median_of_rounds')).toBe(1.9);
        expect(metric('surface-rays/paired/ratio/surface/closest/round2')).toBe(1.8);
        expect(metric('surface-rays/paired/ratio/surface/closest/repeat_at_end')).toBe(1.85);
        expect(metric('surface-rays/paired/merged/surface/closest/mrays_per_s_max_of_rounds')).toBe(300);
        expect(metric('surface-rays/merged/surface/closest/instances_entered_without_hit')).toBe(8.1);
        expect(metric('surface-rays/merged/surface/closest/agreement_with_instanced')).toBe(1);
        expect(metric('surface-rays/instanced/top_level_instances')).toBe(400);
        expect(metric('surface-rays/paired/hit_record_40_over_24/merged/surface/closest/median_of_rounds')).toBe(0.98);
        expect(metric('surface-rays/paired/hit_record_40_over_24/merged/surface/closest/round1')).toBe(0.97);
        expect(metric('surface-rays/paired/hit_record/merged/surface/closest/record40_mrays_per_s_max_of_rounds')).toBe(300);
        expect(metric('surface-rays/paired/hit_record/merged/surface/closest/compact_agreement')).toBe(1);
        expect(metric('surface-rays/paired/any_larger_over_far/merged/surface/any/median_of_rounds')).toBe(0.78);
        expect(metric('surface-rays/paired/any_larger_over_far/merged/surface/any/round2')).toBe(0.76);
        expect(metric('surface-rays/paired/any_hit_order/merged/surface/any/larger_mrays_per_s_max_of_rounds')).toBe(520);
        expect(metric('surface-rays/paired/any_hit_order/merged/surface/any/larger/agreement_with_far')).toBe(1);
        expect(metric('surface-rays/any_hit_order/merged/surface/any/larger/steps')).toBe(42);
    });
});

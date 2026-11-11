/**
 * The paired method of the surface-ray bench, without the GPU: which timing
 * comes when, and how rates become a ratio.
 *
 * The M5's speed drifts with heat by up to about 30%, so two numbers timed
 * minutes apart say little about each other. Instead the instanced and merged
 * trees are timed one right after the other, for several rounds, with the order
 * reversed every other round (so a drift that goes one way falls on both
 * sides), and the ratio is taken within each round. The ratio is the result.
 */

import { median } from './statistics.ts';

export type Tree = 'instanced' | 'merged';

/** What is timed on both trees: a ray set traced closest-hit or any-hit. */
export interface PairKey {
    set: string;
    mode: 'closest' | 'any';
}

export interface PairedStep {
    pair: PairKey;
    tree: Tree;
}

/**
 * The timings of one round, in order. Even rounds go through `pairs` as given,
 * instanced tree first; odd rounds go through them backwards, merged tree
 * first. The two trees of a pair are always timed back to back.
 */
export function pairedRoundOrder(pairs: readonly PairKey[], round: number): PairedStep[] {
    const reversed = round % 2 === 1;
    const ordered = reversed ? [...pairs].reverse() : pairs;
    const trees: Tree[] = reversed ? ['merged', 'instanced'] : ['instanced', 'merged'];
    return ordered.flatMap((pair) => trees.map((tree) => ({ pair, tree })));
}

/** A ratio of two trees' rates, per round, with the middle and the spread of the rounds. */
export interface PairedRatio {
    name: string;
    perRound: number[];
    median: number;
    min: number;
    max: number;
}

/** `numerator[round] / denominator[round]` for every round, and the median, minimum and maximum of those. */
export function ratioOfRounds(name: string, numerator: readonly number[], denominator: readonly number[]): PairedRatio {
    if (numerator.length !== denominator.length) throw new RangeError(`${name}: ${numerator.length} rounds against ${denominator.length}`);
    const perRound = numerator.map((rate, round) => rate / denominator[round]);
    return { name, perRound, median: median(perRound), min: Math.min(...perRound), max: Math.max(...perRound) };
}

/** The hit-record comparison: one tree and ray set traced closest-hit with the 40-byte record (`hitNormal: true`) and the 24-byte one (`false`). */
export type HitRecord = 'record40' | 'record24';

export interface HitRecordCase {
    tree: Tree;
    set: string;
}

export interface HitRecordStep {
    case: HitRecordCase;
    record: HitRecord;
}

/**
 * The timings of one round of the hit-record comparison, in order. Even rounds
 * go through `cases` as given with the 40-byte record first; odd rounds go
 * backwards with the 24-byte record first. The two records of a case are
 * always timed back to back, as the two trees of a pair are.
 */
export function hitRecordRoundOrder(cases: readonly HitRecordCase[], round: number): HitRecordStep[] {
    const reversed = round % 2 === 1;
    const ordered = reversed ? [...cases].reverse() : cases;
    const records: HitRecord[] = reversed ? ['record24', 'record40'] : ['record40', 'record24'];
    return ordered.flatMap((entry) => records.map((record) => ({ case: entry, record })));
}

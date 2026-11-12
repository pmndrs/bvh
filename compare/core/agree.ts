/**
 * Agreement with our answers. Every library's answer to every query is
 * checked against @pmndrs/bvh's on the same query:
 *
 * - **same**: the same triangle (and the same distance within tolerance);
 * - **tie**: another triangle at the same distance (shared edges, coincident
 *   faces), or, for all hits, a list that differs only by a repeat at a
 *   distance already in it (a hit on a shared edge counted on both sides), or,
 *   for culling, an object whose box touches the query's boundary;
 * - **disagree**: anything else: a hit against a miss, another distance, a
 *   different count of distinct hits, another yes or no.
 *
 * Disagreements are reported, with the first few queries named. A library
 * with any disagreement on a measurement is not ranked against ours on that
 * measurement: a faster wrong answer is not faster.
 */

import type { ClosestPointHit, RayHit } from '../adapters/types.ts';

export interface Agreement {
    checked: number;
    same: number;
    ties: number;
    disagree: number;
    /** The first few disagreements, for the report. */
    examples: string[];
}

export function emptyAgreement(): Agreement {
    return { checked: 0, same: 0, ties: 0, disagree: 0, examples: [] };
}

/** Distances agree within float tolerance: relative 1e-4, with a floor of 1e-6 of the scene's size. */
export function closeEnough(a: number, b: number, scale: number): boolean {
    return Math.abs(a - b) <= 1e-4 * Math.max(Math.abs(a), Math.abs(b)) + 1e-6 * scale;
}

function note(agreement: Agreement, text: string): void {
    agreement.disagree++;
    if (agreement.examples.length < 5) agreement.examples.push(text);
}

export function checkFirst(agreement: Agreement, query: number, ours: RayHit | null, theirs: RayHit | null, scale: number): void {
    agreement.checked++;
    if (!ours && !theirs) agreement.same++;
    else if (!ours || !theirs) note(agreement, `ray ${query}: ours ${ours ? `t=${ours.t.toPrecision(6)}` : 'miss'}, theirs ${theirs ? `t=${theirs.t.toPrecision(6)}` : 'miss'}`);
    else if (!closeEnough(ours.t, theirs.t, scale)) note(agreement, `ray ${query}: ours t=${ours.t.toPrecision(6)} (tri ${ours.triangle}), theirs t=${theirs.t.toPrecision(6)} (tri ${theirs.triangle})`);
    else if (ours.triangle === theirs.triangle) agreement.same++;
    else agreement.ties++;
}

/** Distinct distances of a sorted hit list (repeats within tolerance merged). */
function distinct(hits: readonly RayHit[], scale: number): number[] {
    const out: number[] = [];
    for (const hit of hits) if (out.length === 0 || !closeEnough(out[out.length - 1], hit.t, scale)) out.push(hit.t);
    return out;
}

export function checkAll(agreement: Agreement, query: number, ours: readonly RayHit[], theirs: readonly RayHit[], scale: number): void {
    agreement.checked++;
    if (ours.length === theirs.length && ours.every((hit, i) => closeEnough(hit.t, theirs[i].t, scale))) {
        if (ours.every((hit, i) => hit.triangle === theirs[i].triangle)) agreement.same++;
        else agreement.ties++;
        return;
    }
    const a = distinct(ours, scale);
    const b = distinct(theirs, scale);
    if (a.length === b.length && a.every((t, i) => closeEnough(t, b[i], scale))) {
        agreement.ties++;
        return;
    }
    note(agreement, `ray ${query}: ours ${ours.length} hits (${a.length} distinct), theirs ${theirs.length} (${b.length} distinct)`);
}

export function checkClosest(agreement: Agreement, query: number, ours: ClosestPointHit | null, theirs: ClosestPointHit | null, scale: number): void {
    agreement.checked++;
    if (!ours && !theirs) agreement.same++;
    else if (!ours || !theirs) note(agreement, `point ${query}: ours ${ours ? ours.distance.toPrecision(6) : 'none'}, theirs ${theirs ? theirs.distance.toPrecision(6) : 'none'}`);
    else if (!closeEnough(ours.distance, theirs.distance, scale)) note(agreement, `point ${query}: ours d=${ours.distance.toPrecision(6)} (tri ${ours.triangle}), theirs d=${theirs.distance.toPrecision(6)} (tri ${theirs.triangle})`);
    else if (ours.triangle === theirs.triangle) agreement.same++;
    else agreement.ties++;
}

export function checkBoolean(agreement: Agreement, query: number, ours: boolean, theirs: boolean): void {
    agreement.checked++;
    if (ours === theirs) agreement.same++;
    else note(agreement, `query ${query}: ours ${ours}, theirs ${theirs}`);
}

/**
 * The same ids, in any order. An id only one side has counts as a tie when
 * `onBoundary(id)` says its box touches the query's boundary within float
 * tolerance (the libraries box and test in different precisions).
 */
export function checkSet(agreement: Agreement, query: number, ours: readonly number[], theirs: readonly number[], onBoundary?: (id: number) => boolean): void {
    agreement.checked++;
    const a = new Set(ours);
    const b = new Set(theirs);
    let missing = 0;
    let extra = 0;
    let boundary = 0;
    for (const id of a) {
        if (b.has(id)) continue;
        if (onBoundary?.(id)) boundary++;
        else missing++;
    }
    for (const id of b) {
        if (a.has(id)) continue;
        if (onBoundary?.(id)) boundary++;
        else extra++;
    }
    const repeats = a.size !== ours.length || b.size !== theirs.length;
    if (missing === 0 && extra === 0 && !repeats) {
        if (boundary === 0) agreement.same++;
        else agreement.ties++;
    } else note(agreement, `query ${query}: ours ${a.size} ids, theirs ${b.size} (${missing} of ours missing, ${extra} extra${repeats ? ', repeats' : ''})`);
}

export function agreementText(a: Agreement): string {
    if (a.checked === 0) return '-';
    const ties = a.ties ? `, ${a.ties} ties` : '';
    return a.disagree === 0 ? `${a.same + a.ties}/${a.checked} agree${ties}` : `${a.disagree}/${a.checked} DISAGREE${ties}`;
}

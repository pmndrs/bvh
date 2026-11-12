/**
 * Timing rules (the plan's): warm runs, the median of several runs, and a
 * sentence saying how each number was taken.
 *
 * Query timings are adaptive, so a brute-force baseline can run on Sponza: a
 * first pass (the warm-up) runs the queries in order until all are done or the
 * budget is spent, which fixes how many queries this library runs (`count`),
 * and is repeated until at least 250 ms have been spent warming up.
 * Each timed run then runs those same `count` queries; the rate is `count`
 * over the median run. Agreement is checked on those `count` queries. The
 * libraries compared on one measurement are timed in interleaved rounds
 * ({@link timeQueriesInterleaved}).
 */

export function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = sorted.length >> 1;
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export interface Timing {
    readonly medianMs: number;
    readonly minMs: number;
    readonly maxMs: number;
    readonly samplesMs: readonly number[];
}

export function timingOf(samplesMs: readonly number[]): Timing {
    return { medianMs: median(samplesMs), minMs: Math.min(...samplesMs), maxMs: Math.max(...samplesMs), samplesMs };
}

/** Lets the page breathe between runs (a no-op wait in Node). */
export const yieldNow = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

export interface QueryTiming<T> extends Timing {
    /** Queries per run. */
    readonly count: number;
    /** Queries per second at the median run. */
    readonly perSecond: number;
    /** The answers of the warm-up pass, one per query run. */
    readonly answers: T[];
}

export interface QueryPlan {
    /** Timed runs after the warm-up (default 5). */
    readonly runs?: number;
    /** Milliseconds the warm-up pass may take before it stops early (default 1500). */
    readonly budgetMs?: number;
    /** Fewest queries to run even past the budget (default 16). */
    readonly minQueries?: number;
}

interface Warm<T> {
    readonly count: number;
    readonly answers: T[];
}

/** The warm-up pass: fixes the query count, keeps the answers, and warms the JIT for at least 250 ms. */
function warm<Q, T>(queries: readonly Q[], run: (query: Q) => T, plan: QueryPlan): Warm<T> {
    const budgetMs = plan.budgetMs ?? 1500;
    const minQueries = Math.min(queries.length, plan.minQueries ?? 16);
    const answers: T[] = [];
    const start = performance.now();
    for (const query of queries) {
        answers.push(run(query));
        if (answers.length >= minQueries && performance.now() - start > budgetMs) break;
    }
    const count = answers.length;
    while (performance.now() - start < 250) for (let i = 0; i < count; i++) run(queries[i]);
    return { count, answers };
}

export interface QueryEntry<Q, T> {
    readonly queries: readonly Q[];
    readonly run: (query: Q) => T;
}

/**
 * Several libraries on the same queries, timed in rounds: each round times
 * one run of every library back to back, the order reversed every other round
 * (the surface-ray bench's paired method), so a drift of the machine's speed
 * falls on all of them alike. Each library's rate is its median round.
 */
export async function timeQueriesInterleaved<Q, T>(entries: readonly QueryEntry<Q, T>[], plan: QueryPlan = {}): Promise<QueryTiming<T>[]> {
    const runs = plan.runs ?? 5;
    const warmed = entries.map((entry) => warm(entry.queries, entry.run, plan));
    const samples = entries.map((): number[] => []);
    for (let r = 0; r < runs; r++) {
        const order = entries.map((_, i) => i);
        if (r % 2) order.reverse();
        for (const i of order) {
            await yieldNow();
            const { queries, run } = entries[i];
            const count = warmed[i].count;
            const t0 = performance.now();
            for (let q = 0; q < count; q++) run(queries[q]);
            samples[i].push(performance.now() - t0);
        }
    }
    return entries.map((_, i) => {
        const timing = timingOf(samples[i]);
        return { ...timing, count: warmed[i].count, perSecond: (warmed[i].count / timing.medianMs) * 1000, answers: warmed[i].answers };
    });
}

export async function timeQueries<Q, T>(queries: readonly Q[], run: (query: Q) => T, plan: QueryPlan = {}): Promise<QueryTiming<T>> {
    const [timing] = await timeQueriesInterleaved([{ queries, run }], plan);
    return timing;
}

/** `runs` timed calls of `run` (no warm-up: a build is its own cold start), the median. */
export async function timeBuilds<T>(run: () => T, runs: number, keep: (value: T, last: boolean) => void): Promise<Timing> {
    const samples: number[] = [];
    for (let r = 0; r < runs; r++) {
        await yieldNow();
        const t0 = performance.now();
        const value = run();
        samples.push(performance.now() - t0);
        keep(value, r === runs - 1);
    }
    return timingOf(samples);
}

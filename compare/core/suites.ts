/**
 * The measurements, shared by the Node runner and the compare pages. Each
 * suite takes libraries that are already built (see {@link buildAll}) and
 * returns one {@link Measurement} per number, each with how it was measured
 * and, for query results, its agreement with ours.
 */

import type { BuiltMesh, ClosestPointHit, CompareGeometry, CompareRay, MeshAdapter, RayHit, Vec3 } from '../adapters/types.ts';
import { checkAll, checkBoolean, checkClosest, checkFirst, emptyAgreement, type Agreement } from './agree.ts';
import { diagonal, perturbed } from './geometry.ts';
import { timeBuilds, timeQueriesInterleaved, type QueryPlan } from './measure.ts';
import { makeBoxes, makePoints, makeRays, makeSpheres, RAY_SETS } from './rays.ts';

export interface Measurement {
    readonly page: string;
    readonly scene: string;
    /** Adapter id (`ours`, `tmbvh-sah`, ...). */
    readonly library: string;
    /** What was measured, e.g. `first/outside`. */
    readonly metric: string;
    readonly value: number | null;
    readonly unit: string;
    /** Queries per run, for query rates. */
    readonly count?: number;
    readonly method: string;
    readonly agreement?: Agreement;
    /** Shown against ours only when every answer agreed (or for numbers that aren't answers: build time, bytes). */
    readonly ranked: boolean;
    /** Why there is no value (the library lacks it), or other context. */
    readonly note?: string;
}

export interface Built {
    readonly adapter: MeshAdapter;
    readonly mesh: BuiltMesh;
}

export const OURS = 'ours';

export interface SuiteOptions {
    readonly rays?: number;
    readonly queries?: number;
    readonly plan?: QueryPlan;
    readonly buildRuns?: number;
    /** Progress lines (the runner prints them, a page shows them). */
    readonly log?: (line: string) => void;
}

const missing = (page: string, scene: string, library: string, metric: string, unit: string, note: string): Measurement => ({ page, scene, library, metric, value: null, unit, method: '-', ranked: false, note });

/**
 * Builds every library `buildRuns` times (default 3), keeps the last build,
 * and records build ms (median), bytes (the adapter's count) and tree quality.
 */
export async function buildAll(adapters: readonly MeshAdapter[], geometry: CompareGeometry, options: SuiteOptions = {}): Promise<{ built: Built[]; measurements: Measurement[] }> {
    const built: Built[] = [];
    const measurements: Measurement[] = [];
    const scene = geometry.name;
    for (const adapter of adapters) {
        const reason = adapter.unsupported?.(geometry);
        if (reason) {
            options.log?.(`${adapter.id}: skipped (${reason})`);
            measurements.push(missing('build', scene, adapter.id, 'build', 'ms', reason));
            continue;
        }
        await adapter.prepare?.();
        let kept: BuiltMesh | null = null;
        const timing = await timeBuilds(
            () => adapter.build(geometry),
            options.buildRuns ?? 3,
            (mesh, last) => {
                if (last) kept = mesh;
                else mesh.dispose();
            },
        );
        const mesh = kept as unknown as BuiltMesh;
        mesh.prepareQueries?.();
        built.push({ adapter, mesh });
        options.log?.(`${adapter.id}: built in ${timing.medianMs.toFixed(1)} ms (median of ${timing.samplesMs.length})`);
        measurements.push({ page: 'build', scene, library: adapter.id, metric: 'build', value: timing.medianMs, unit: 'ms', method: `median of ${timing.samplesMs.length} builds on this thread, wall clock: ${adapter.buildMethod}`, ranked: true });
        const memory = mesh.memory();
        measurements.push({ page: 'build', scene, library: adapter.id, metric: 'memory', value: memory.bytes, unit: 'bytes', method: memory.method, ranked: memory.bytes !== null });
        const quality = mesh.quality?.();
        if (quality) {
            const q = (metric: string, value: number | null, unit: string) =>
                measurements.push({ page: 'tree-quality', scene, library: adapter.id, metric, value, unit, method: 'the tree as built above: SAH with traversal 1 and triangle 1.25 over the root box area (getBVHExtremes for ours and three-mesh-bvh, the same sum over bvh-tree\'s nodes)', ranked: value !== null, note: quality.note });
            q('sah', quality.sah, 'relative');
            q('depth', quality.depth, 'levels');
            q('nodes', quality.nodes, 'boxes');
            q('leaves', quality.leaves, 'leaves');
        }
    }
    return { built, measurements };
}

function reference(built: readonly Built[]): BuiltMesh {
    const ours = built.find((entry) => entry.adapter.id === OURS);
    if (!ours) throw new Error('the suites check every library against ours: build ours first');
    return ours.mesh;
}

interface Candidate<Q, T> {
    readonly adapter: MeshAdapter;
    readonly run: (query: Q) => T;
}

/**
 * One measurement over every library that has it: the libraries timed in
 * interleaved rounds on the same queries, each checked against ours.
 */
async function measureAll<Q, T>(
    page: string,
    scene: string,
    metric: string,
    unit: string,
    built: readonly Built[],
    queries: readonly Q[],
    pick: (entry: Built) => ((query: Q) => T) | undefined,
    capability: keyof MeshAdapter['features'],
    check: (agreement: Agreement, index: number, answer: T) => void,
    what: string,
    options: SuiteOptions,
): Promise<Measurement[]> {
    const out: Measurement[] = [];
    const candidates: Candidate<Q, T>[] = [];
    for (const entry of built) {
        const run = pick(entry);
        const feature = entry.adapter.features[capability];
        if (run) candidates.push({ adapter: entry.adapter, run });
        else out.push(missing(page, scene, entry.adapter.id, metric, unit, feature.note ?? 'not available'));
    }
    const timings = await timeQueriesInterleaved(candidates.map((candidate) => ({ queries, run: candidate.run })), options.plan);
    candidates.forEach(({ adapter }, c) => {
        const timing = timings[c];
        const feature = adapter.features[capability];
        const agreement = emptyAgreement();
        timing.answers.forEach((answer, i) => check(agreement, i, answer));
        options.log?.(`${adapter.id} ${metric}: ${(timing.perSecond / 1000).toFixed(1)}k ${unit} over ${timing.count}; ${agreement.disagree} disagree, ${agreement.ties} ties`);
        out.push({
            page,
            scene,
            library: adapter.id,
            metric,
            value: timing.perSecond,
            unit,
            count: timing.count,
            method: `${what}; one warm pass fixes the query count (${timing.count}: all, or what fits in the budget), then the median of ${timing.samplesMs.length} runs, interleaved with the other libraries' runs (order reversed every other round), wall clock${feature.note ? `; ${feature.note}` : ''}`,
            agreement,
            ranked: agreement.disagree === 0,
            note: feature.status === 'partial' ? feature.note : undefined,
        });
    });
    return out;
}

/** CPU first-hit and all-hit raycasts, both ray sets. */
export async function raycastSuite(built: readonly Built[], geometry: CompareGeometry, options: SuiteOptions = {}): Promise<Measurement[]> {
    const out: Measurement[] = [];
    const ours = reference(built);
    const scale = diagonal(geometry);
    const scene = geometry.name;
    for (const set of RAY_SETS) {
        const rays = makeRays(geometry, set, options.rays ?? 4096);
        const first = new Map<number, RayHit | null>();
        const every = new Map<number, RayHit[]>();
        const oursFirst = (i: number) => (first.has(i) ? first.get(i)! : (first.set(i, ours.raycastFirst!(rays[i])), first.get(i)!));
        const oursAll = (i: number) => every.get(i) ?? (every.set(i, ours.raycastAll!(rays[i])), every.get(i)!);
        out.push(
            ...(await measureAll<CompareRay, RayHit | null>('raycast', scene, `first/${set}`, 'rays/s', built, rays, ({ mesh }) => mesh.raycastFirst?.bind(mesh), 'raycastFirst', (a, i, answer) => checkFirst(a, i, oursFirst(i), answer, scale), `${set} rays (seeded), double-sided, closest hit`, options)),
        );
        out.push(
            ...(await measureAll<CompareRay, RayHit[]>('raycast', scene, `all/${set}`, 'rays/s', built, rays, ({ mesh }) => mesh.raycastAll?.bind(mesh), 'raycastAll', (a, i, answer) => checkAll(a, i, oursAll(i), answer, scale), `${set} rays (seeded), double-sided, every hit sorted by distance`, options)),
        );
    }
    return out;
}

/** Box, sphere and closest-point queries. */
export async function queriesSuite(built: readonly Built[], geometry: CompareGeometry, options: SuiteOptions = {}): Promise<Measurement[]> {
    const ours = reference(built);
    const scale = diagonal(geometry);
    const scene = geometry.name;
    const n = options.queries ?? 2048;
    const boxes = makeBoxes(geometry, n);
    const spheres = makeSpheres(geometry, n);
    const points = makePoints(geometry, n);
    const oursBoxes = boxes.map((box) => ours.intersectsBox!(box));
    const oursSpheres = spheres.map((sphere) => ours.intersectsSphere!(sphere.center, sphere.radius));
    const oursPoints = new Map<number, ClosestPointHit | null>();
    const oursPoint = (i: number) => (oursPoints.has(i) ? oursPoints.get(i)! : (oursPoints.set(i, ours.closestPoint!(points[i])), oursPoints.get(i)!));
    return [
        ...(await measureAll<readonly number[], boolean>('queries', scene, 'box', 'queries/s', built, boxes, ({ mesh }) => mesh.intersectsBox?.bind(mesh), 'intersectsBox', (a, i, answer) => checkBoolean(a, i, oursBoxes[i], answer), 'does any triangle meet the box (seeded boxes, edges of 1-4% of the diagonal)', options)),
        ...(await measureAll<{ center: Vec3; radius: number }, boolean>(
            'queries',
            scene,
            'sphere',
            'queries/s',
            built,
            spheres,
            ({ mesh }) => (mesh.intersectsSphere ? (sphere) => mesh.intersectsSphere!(sphere.center, sphere.radius) : undefined),
            'intersectsSphere',
            (a, i, answer) => checkBoolean(a, i, oursSpheres[i], answer),
            'does any triangle meet the sphere (seeded, radius 0.5-2% of the diagonal)',
            options,
        )),
        ...(await measureAll<Vec3, ClosestPointHit | null>('queries', scene, 'closest-point', 'queries/s', built, points, ({ mesh }) => mesh.closestPoint?.bind(mesh), 'closestPoint', (a, i, answer) => checkClosest(a, i, oursPoint(i), answer, scale), 'closest point on the mesh to seeded points in its box grown by 10%', options)),
    ];
}

/** Refit to moved positions: median ms, and 512 first-hit rays checked against our refitted tree. */
export async function refitSuite(built: readonly Built[], geometry: CompareGeometry): Promise<Measurement[]> {
    const out: Measurement[] = [];
    const scene = geometry.name;
    const amount = diagonal(geometry) * 0.002;
    const moved = [perturbed(geometry, amount, 1), perturbed(geometry, amount, 2)];
    const rays = makeRays(geometry, 'outside', 512, 9);
    const ours = reference(built);
    const scale = diagonal(geometry);
    for (const { adapter, mesh } of built) {
        if (!mesh.refit) {
            out.push(missing('build', scene, adapter.id, 'refit', 'ms', adapter.features.refit.note ?? 'not available'));
            continue;
        }
        let k = 0;
        const timing = await timeBuilds(() => mesh.refit!(moved[k++ % 2]), 5, () => {});
        out.push({ page: 'build', scene, library: adapter.id, metric: 'refit', value: timing.medianMs, unit: 'ms', method: `median of 5 refits on this thread, alternating between two moved copies of the positions (a 0.2%-of-diagonal wave)${adapter.features.refit.note ? `; ${adapter.features.refit.note}` : ''}`, ranked: true });
    }
    // Every refit has now run 5 times: the last one applied moved[0] everywhere.
    if (ours.refit) {
        const answers = rays.map((ray) => ours.raycastFirst!(ray));
        for (const { adapter, mesh } of built) {
            if (!mesh.refit || !mesh.raycastFirst || adapter.id === OURS) continue;
            const agreement = emptyAgreement();
            rays.forEach((ray, i) => checkFirst(agreement, i, answers[i], mesh.raycastFirst!(ray), scale));
            out.push({ page: 'build', scene, library: adapter.id, metric: 'refit-agreement', value: agreement.disagree, unit: 'disagreements', method: '512 outside rays, closest hit, after the refits, against our refitted tree', agreement, ranked: agreement.disagree === 0 });
        }
    }
    return out;
}

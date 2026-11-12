/**
 * compare/scene's measurements: a scene-level tree over many objects. Build,
 * frustum and box queries, rays, then frames of motion (every object moves)
 * with each tree's update timed, and the queries checked again after them.
 */

import type { BuiltScene, CompareScene, SceneAdapter } from '../adapters/types.ts';
import { checkFirst, checkSet, emptyAgreement } from './agree.ts';
import { timeBuilds, timeQueries, timingOf, yieldNow, type QueryPlan } from './measure.ts';
import { makeFrusta, makeSceneBoxes, makeSceneRays, moveObjects } from './scene.ts';
import type { Measurement } from './suites.ts';

export interface SceneSuiteOptions {
    readonly queries?: number;
    readonly frames?: number;
    readonly plan?: QueryPlan;
    readonly log?: (line: string) => void;
}

export async function sceneSuite(adapters: readonly SceneAdapter[], scene: CompareScene, options: SceneSuiteOptions = {}): Promise<{ measurements: Measurement[]; built: { adapter: SceneAdapter; scene: BuiltScene }[] }> {
    const out: Measurement[] = [];
    const name = scene.name;
    const built: { adapter: SceneAdapter; scene: BuiltScene }[] = [];
    for (const adapter of adapters) {
        await adapter.prepare?.();
        let kept: BuiltScene | null = null;
        const timing = await timeBuilds(() => adapter.build(scene), 3, (value, last) => (last ? (kept = value) : value.dispose()));
        const result = kept as unknown as BuiltScene;
        built.push({ adapter, scene: result });
        out.push({ page: 'scene', scene: name, library: adapter.id, metric: 'build', value: timing.medianMs, unit: 'ms', method: `median of 3 builds, wall clock: ${adapter.buildMethod}`, ranked: true });
        const memory = result.memory();
        out.push({ page: 'scene', scene: name, library: adapter.id, metric: 'memory', value: memory.bytes, unit: 'bytes', method: memory.method, ranked: memory.bytes !== null });
        options.log?.(`${adapter.id}: scene built in ${timing.medianMs.toFixed(1)} ms`);
    }
    const ours = built[0];
    if (ours.adapter.id !== 'ours') throw new Error('sceneSuite: ours first');
    const n = options.queries ?? 512;
    const frusta = makeFrusta(n);
    const boxes = makeSceneBoxes(n);
    const rays = makeSceneRays(n);
    const scale = 200;

    const queryRound = async (suffix: string, timed: boolean) => {
        const oursFrusta = frusta.map((f) => ours.scene.intersectsFrustum!(f.planes, f.viewProjection));
        const oursBoxes = boxes.map((box) => ours.scene.intersectsBox!(box));
        const oursRays = rays.map((ray) => ours.scene.raycastFirst!(ray));
        for (const { adapter, scene: tree } of built) {
            const kinds: { metric: string; run: ((i: number) => unknown) | undefined; check: (a: ReturnType<typeof emptyAgreement>, i: number, answer: unknown) => void; feature: SceneAdapter['features']['frustum']; what: string }[] = [
                { metric: 'frustum', run: tree.intersectsFrustum ? (i: number) => tree.intersectsFrustum!(frusta[i].planes, frusta[i].viewProjection) : undefined, check: (a: ReturnType<typeof emptyAgreement>, i: number, answer: unknown) => checkSet(a, i, oursFrusta[i], answer as number[]), feature: adapter.features.frustum, what: 'objects whose world box meets a camera frustum (60 degrees, far plane at half the scene)' },
                { metric: 'box', run: tree.intersectsBox ? (i: number) => tree.intersectsBox!(boxes[i]) : undefined, check: (a: ReturnType<typeof emptyAgreement>, i: number, answer: unknown) => checkSet(a, i, oursBoxes[i], answer as number[]), feature: adapter.features.box, what: 'objects whose world box meets a box (edges of 5-15% of the scene)' },
                {
                    metric: 'raycast',
                    run: tree.raycastFirst ? (i: number) => tree.raycastFirst!(rays[i]) : undefined,
                    check: (a: ReturnType<typeof emptyAgreement>, i: number, answer: unknown) => {
                        const theirs = answer as ReturnType<NonNullable<BuiltScene['raycastFirst']>>;
                        const mine = oursRays[i];
                        // The same object and distance (a tie inside one object counts as the same answer).
                        checkFirst(a, i, mine && { triangle: mine.object * 1e7 + mine.triangle, t: mine.t }, theirs && { triangle: theirs.object * 1e7 + theirs.triangle, t: theirs.t }, scale);
                    },
                    feature: adapter.features.raycast,
                    what: 'closest hit over every object\'s triangles, rays from inside the scene',
                },
            ];
            for (const kind of kinds) {
                const metric = `${kind.metric}${suffix}`;
                if (!kind.run) {
                    out.push({ page: 'scene', scene: name, library: adapter.id, metric, value: null, unit: 'queries/s', method: '-', ranked: false, note: kind.feature.note ?? 'not available' });
                    continue;
                }
                const indices = Array.from({ length: n }, (_, i) => i);
                const agreement = emptyAgreement();
                if (timed) {
                    const timing = await timeQueries(indices, kind.run, options.plan);
                    timing.answers.forEach((answer, i) => kind.check(agreement, i, answer));
                    out.push({ page: 'scene', scene: name, library: adapter.id, metric, value: timing.perSecond, unit: 'queries/s', count: timing.count, method: `${kind.what}; one warm pass, then the median of ${timing.samplesMs.length} runs, wall clock${kind.feature.note ? `; ${kind.feature.note}` : ''}`, agreement, ranked: agreement.disagree === 0 });
                    options.log?.(`${adapter.id} ${metric}: ${(timing.perSecond / 1000).toFixed(1)}k/s; ${agreement.disagree} disagree`);
                } else {
                    indices.forEach((i) => kind.check(agreement, i, kind.run!(i)));
                    out.push({ page: 'scene', scene: name, library: adapter.id, metric, value: agreement.disagree, unit: 'disagreements', method: `${kind.what}, checked after the frames of motion`, agreement, ranked: agreement.disagree === 0 });
                }
            }
        }
    };
    await queryRound('', true);

    // Frames of motion: every object moves (not timed), then each tree's update is timed.
    const frames = options.frames ?? 10;
    const samples = new Map<string, number[]>(built.map(({ adapter }) => [adapter.id, []]));
    for (let frame = 0; frame < frames; frame++) {
        moveObjects(scene, frame);
        for (const { adapter, scene: tree } of built) {
            if (!tree.update) continue;
            const t0 = performance.now();
            tree.update(scene.objects);
            samples.get(adapter.id)!.push(performance.now() - t0);
        }
        await yieldNow();
    }
    for (const { adapter, scene: tree } of built) {
        const values = samples.get(adapter.id)!;
        if (!values.length) continue;
        out.push({ page: 'scene', scene: name, library: adapter.id, metric: 'update', value: timingOf(values).medianMs, unit: 'ms', method: `median over ${frames} frames in which every object moves, wall clock (${tree.updateKind}); ${adapter.features.update.note ?? ''}`, ranked: true });
    }
    await queryRound('-after-motion', false);
    return { measurements: out, built };
}

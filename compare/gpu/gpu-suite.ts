/**
 * compare/gpu's measurement, with the surface-ray bench's paired method: the
 * M5's speed drifts with heat by up to about 30%, so ours and three-mesh-bvh's
 * are timed back to back in every round, the order reversed every other
 * round, and the ratio is taken within each round; the median ratio is the
 * result. Each timing is the median of several dispatches after warm-ups.
 * The answers (closest `t`, any-hit blocked) are checked ray by ray.
 */

import { checkBoolean, checkFirst, emptyAgreement } from '../core/agree.ts';
import { median, yieldNow } from '../core/measure.ts';
import type { Measurement } from '../core/suites.ts';
import type { GpuMode, GpuTracer } from './gpu-trace.ts';

export interface GpuPlan {
    readonly rounds?: number;
    readonly warmups?: number;
    readonly dispatches?: number;
}

export interface GpuRayInput {
    readonly closest: Float32Array;
    readonly any: Float32Array;
    /** Rays per mode. */
    readonly count: number;
    /** The any-hit rays' length, for the tie rule at the end of a ray. */
    readonly anyTMax: number;
    /** Scene size, for the distance tolerance. */
    readonly scale: number;
    readonly set: string;
}

export async function gpuSuite(tracers: readonly GpuTracer[], scene: string, input: GpuRayInput, plan: GpuPlan = {}, log?: (line: string) => void): Promise<Measurement[]> {
    const rounds = plan.rounds ?? 5;
    const warmups = plan.warmups ?? 3;
    const dispatches = plan.dispatches ?? 7;
    const modes: GpuMode[] = ['closest', 'any'];
    for (const tracer of tracers) {
        tracer.setRays('closest', input.closest);
        tracer.setRays('any', input.any);
        for (const mode of modes) for (let i = 0; i < warmups; i++) await tracer.dispatch(mode);
    }
    const rates = new Map<string, number[]>();
    for (let round = 0; round < rounds; round++) {
        const reversed = round % 2 === 1;
        const order = reversed ? [...modes].reverse() : modes;
        const who = reversed ? [...tracers].reverse() : tracers;
        for (const mode of order) {
            for (const tracer of who) {
                const samples: number[] = [];
                for (let i = 0; i < dispatches; i++) samples.push(await tracer.dispatch(mode));
                const key = `${tracer.id}/${mode}`;
                rates.set(key, [...(rates.get(key) ?? []), (input.count / median(samples)) * 1000]);
            }
            await yieldNow();
        }
    }
    const out: Measurement[] = [];
    const ours = tracers.find((tracer) => tracer.id === 'ours')!;
    const oursClosest = await ours.read('closest');
    const oursAny = await ours.read('any');
    for (const tracer of tracers) {
        const closest = tracer === ours ? oursClosest : await tracer.read('closest');
        const any = tracer === ours ? oursAny : await tracer.read('any');
        const closestAgreement = emptyAgreement();
        const anyAgreement = emptyAgreement();
        for (let i = 0; i < input.count; i++) {
            const a = oursClosest[i] >= 0 ? { triangle: 0, t: oursClosest[i] } : null;
            const b = closest[i] >= 0 ? { triangle: 0, t: closest[i] } : null;
            checkFirst(closestAgreement, i, a, b, input.scale);
            // A ray whose closest hit sits at its very end may be blocked by one walk and not the other.
            const atEnd = Math.abs(oursClosest[i] - input.anyTMax) <= 1e-4 * input.anyTMax;
            if (atEnd && (oursAny[i] > 0.5) !== (any[i] > 0.5)) {
                anyAgreement.checked++;
                anyAgreement.ties++;
            } else checkBoolean(anyAgreement, i, oursAny[i] > 0.5, any[i] > 0.5);
        }
        for (const mode of modes) {
            const perRound = rates.get(`${tracer.id}/${mode}`)!;
            const agreement = mode === 'closest' ? closestAgreement : anyAgreement;
            out.push({
                page: 'gpu',
                scene,
                library: tracer.id,
                metric: `${mode}/${input.set}`,
                value: median(perRound),
                unit: 'rays/s',
                count: input.count,
                method: `${input.count} ${input.set} rays (seeded), ${mode === 'closest' ? 'closest hit' : `any hit within ${input.anyTMax.toPrecision(3)}`}; per round the median of ${dispatches} dispatches after ${warmups} warm-ups (${tracer.timing}); median of ${rounds} rounds, ours and three-mesh-bvh back to back, the order reversed every other round`,
                agreement,
                ranked: agreement.disagree === 0,
            });
            log?.(`${tracer.id} ${mode}: ${(median(perRound) / 1e6).toFixed(1)} Mrays/s; ${agreement.disagree} disagree`);
        }
        out.push({ page: 'gpu', scene, library: tracer.id, metric: 'gpu-bytes', value: tracer.bytes, unit: 'bytes', method: tracer.bytesMethod, ranked: true });
    }
    // The paired ratios: ours over three-mesh-bvh within each round.
    const other = tracers.find((tracer) => tracer.id === 'tmbvh');
    if (other) {
        for (const mode of modes) {
            const a = rates.get(`ours/${mode}`)!;
            const b = rates.get(`tmbvh/${mode}`)!;
            const perRound = a.map((rate, i) => rate / b[i]);
            const agreement = out.find((m) => m.library === 'tmbvh' && m.metric === `${mode}/${input.set}`)!.agreement!;
            out.push({
                page: 'gpu',
                scene,
                library: 'ours-over-tmbvh',
                metric: `${mode}/${input.set}/ratio`,
                value: median(perRound),
                unit: 'ratio',
                method: `ours' rate over three-mesh-bvh's within each round, median of ${rounds} rounds (min ${Math.min(...perRound).toFixed(2)}, max ${Math.max(...perRound).toFixed(2)})`,
                ranked: agreement.disagree === 0,
            });
        }
    }
    return out;
}

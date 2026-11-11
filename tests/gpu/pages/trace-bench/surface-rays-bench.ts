/**
 * The paired surface-ray bench: rays that start on surfaces, traced through
 * the scene's per-mesh instanced tree and through its merged tree
 * (`buildTraceScene(..., { mergeStaticInstances: true })` with every instance
 * `static: true`), timed in turn so the GPU's drift cancels in the ratio.
 *
 * The question comes from the radiance-cache team (webgpu-ddgi, RC5.4): on
 * their M5, rays leaving surfaces traced 1.86x (Sponza) and 1.24x (Bistro
 * interior) faster closest-hit, and 1.39x and 1.13x any-hit, through one
 * world-space tree than through per-mesh instances. This page makes the same
 * measurement on this package's merged tree.
 *
 * Sets: `surface` (RC1.5's screen-probe rays: 64 stratified cosine rays per
 * probe from visible surfaces) and `air-grid` (the same number of rays from a
 * grid of points in the air), each closest-hit and any-hit.
 *
 * What the page measures:
 *   1. correctness (the verdict, and the only thing that can fail the page):
 *      the merged tree must find the instanced tree's hit, instance and
 *      triangle, on the rays;
 *   2. the walk's work per ray on each tree, from a counted copy of the walk
 *      (counted-walk-wgsl.ts): steps, instances entered, entered without a hit;
 *   3. the paired timing: per round, each set and mode on the instanced tree
 *      and the merged tree back to back, the order reversed every other round
 *      (paired-ratio.ts), then the surface sets once more at the end. The
 *      ratio of the two trees' rates within each round is the result.
 *
 *   4. the cost of the 40-byte hit record: the same trees and sets traced
 *      closest-hit by a kernel with `hitNormal: true` (the default, 40-byte
 *      records with material and normal) and one with `hitNormal: false`
 *      (24-byte records), timed back to back with the same rounds and the
 *      order reversed every other round. The ratio is the 40-byte kernel's
 *      rate over the 24-byte kernel's (below 1: the bigger record costs).
 *      The compact kernel's hit, instance and triangle must equal the full
 *      kernel's on the rays.
 *
 * Timing never fails the page: speeds depend on the GPU and on its heat.
 */

import { buildTraceScene, createMeshBvhBuilder, HIT_RECORD_BYTES, HIT_RECORD_COMPACT_BYTES, HIT_RECORD_COMPACT_WORD, HIT_RECORD_WORD, RAY_FLOATS, type MeshGeometry, type TraceSceneData, type TraceSceneInstance } from '../../../../src/trace/index.ts';
import { createRandom } from '../../../../src/trace/oracle-fixtures.ts';
import type { AdapterSummary } from '../../contract.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { TraceBatchKernel, uploadTraceScene, type TraceSceneBuffers } from '../../../../src/trace/trace-gpu.ts';
import { RAY_DIRECTION_SEED, RC_HALTON_FIRST_INDEX, RC_RAYS_PER_PROBE, RC_SEED, RC_TILE_SIZE, RC_VIEW } from './config.ts';
import { COUNTED_WORD, COUNTED_WORDS, countedKernelWgsl } from './counted-walk-wgsl.ts';
import { createDispatchTimer, measureDispatches } from './dispatch-timer.ts';
import { timeLoaded } from './loaded-timing.ts';
import { readBuffer, uploadRays, yieldToBrowser } from './ours-bench.ts';
import { hitRecordRoundOrder, pairedRoundOrder, ratioOfRounds, type HitRecord, type HitRecordCase, type PairKey, type PairedRatio, type Tree } from './paired-ratio.ts';
import { makePlacementRays, rayProblems, tileGrid } from './rc-rays.ts';
import { makeProbeGrid, makeProbeRays as makeGridRays, makeSphereDirections } from './rays.ts';
import { fullBrowserVersion, requestBenchDevice } from './run.ts';
import { findManifest, loadBenchScene } from './scene-source.ts';
import { compareBlocked, compareHitRecords, makeVariantProbeRays, probeOrigins, summariseCounts, surfaceProbes, type CountSummary, type HitAgreement } from './surface-ray-variants.ts';

/**
 * Timing profiles. `full` is the measurement: 5 paired rounds with a loaded
 * section of 0.5 s per timing. `quick` is a smoke run.
 */
export const SURFACE_PROFILES = {
    full: { rounds: 5, warmups: 2, timed: 5, loadedMs: 500 },
    quick: { rounds: 3, warmups: 1, timed: 3, loadedMs: 150 },
} as const;
export type SurfaceProfileName = keyof typeof SURFACE_PROFILES;

/** The merged tree must find the instanced tree's hit (exact instance and triangle) on at least this share of rays. */
const MIN_HIT_AGREEMENT = 0.999;
/** The counted copy of the walk must find the production kernel's answer on at least this share of rays. */
const MIN_COUNTED_AGREEMENT = 0.9999;

/** The sets, each traced closest-hit and any-hit. */
type SetName = 'surface' | 'air-grid';
const MODES = ['closest', 'any'] as const;

/** A ray set on the GPU. */
interface RaySet {
    name: SetName;
    rays: Float32Array;
    buffer: GPUBuffer;
    count: number;
}

/** What one tree gave for one set and mode: the counted walk's work, and agreement with the instanced tree. */
export interface SurfaceSetResult extends CountSummary {
    tree: Tree;
    set: SetName;
    mode: 'closest' | 'any';
    /** Share of rays where the counted copy of the walk found the production kernel's answer. */
    countedAgreement: number;
    /** Merged tree only, closest: how its hits compare with the instanced tree's. */
    hitAgreement: HitAgreement | null;
    /** Merged tree only, any: share of rays with the same blocked flag. */
    blockedAgreement: number | null;
}

/** One tree's rate on one set and mode in every round, and in the repeat at the end. */
export interface PairedEntry {
    tree: Tree;
    set: SetName;
    mode: 'closest' | 'any';
    /** Loaded Mrays/s per round (back-to-back passes timed by timestamps). */
    mraysPerRound: number[];
    /** Single-dispatch Mrays/s per round (each dispatch waited for). */
    singleMraysPerRound: number[];
    /** The repeat at the end of the batch; NaN for sets not repeated. */
    repeatMrays: number;
}

export interface PairedResult {
    rounds: number;
    profile: SurfaceProfileName;
    entries: PairedEntry[];
    /** Merged rate over instanced rate, per round, for every set and mode (>1: the merged tree is faster). */
    ratios: PairedRatio[];
    /** The same ratio from the repeat at the end of the batch. */
    repeatRatios: { name: string; ratio: number }[];
}

/** One tree's closest-hit rate on one set with one record size, per round. */
export interface HitRecordEntry {
    tree: Tree;
    set: SetName;
    record: HitRecord;
    recordBytes: number;
    mraysPerRound: number[];
    singleMraysPerRound: number[];
}

/** The 40-byte record over the 24-byte one, per tree and set (named `hit_record_40_over_24/<tree>/<set>/closest`). */
export interface HitRecordResult {
    rounds: number;
    entries: HitRecordEntry[];
    ratios: PairedRatio[];
    /** Rays where the compact kernel's t, instance, triangle, front face and barycentrics equal the full kernel's. */
    agreement: { tree: Tree; set: SetName; rays: number; same: number }[];
}

export interface TreeSummary {
    topLevelInstances: number;
    topLevelNodes: number;
    bytesNodes: number;
    bytesTriangles: number;
    bytesTopLevel: number;
    buildMs: number;
}

export interface SurfaceRaysData {
    benchmark: 'surface-rays';
    scene: string;
    meshCount: number;
    instanceCount: number;
    instancedTriangles: number;
    camera: string;
    probesPlaced: number;
    gridProbes: number;
    raysPerProbe: number;
    raysPerSet: Record<string, number>;
    userAgent: string;
    browserVersion: string | null;
    timingSource: string;
    trees: Record<Tree, TreeSummary>;
    sets: SurfaceSetResult[];
    paired: PairedResult;
    hitRecord?: HitRecordResult;
}

function summarizeTree(data: TraceSceneData, buildMs: number): TreeSummary {
    return {
        topLevelInstances: data.topLevel.instanceCount,
        topLevelNodes: data.topLevel.nodeCount,
        bytesNodes: data.nodes.byteLength,
        bytesTriangles: data.triangles.byteLength,
        bytesTopLevel: data.topLevel.buffer.byteLength,
        buildMs,
    };
}

/** The counted copy of the walk as a pipeline per mode. */
class CountedKernel {
    private constructor(
        private readonly device: GPUDevice,
        private readonly pipelines: Record<'closest' | 'any', GPUComputePipeline>,
    ) {}

    static async create(device: GPUDevice): Promise<CountedKernel> {
        const make = (mode: 'closest' | 'any') =>
            device.createComputePipelineAsync({
                label: `counted walk ${mode}`,
                layout: 'auto',
                compute: { module: device.createShaderModule({ label: `counted walk ${mode}`, code: countedKernelWgsl(mode) }), entryPoint: 'benchCounted' },
            });
        const [closest, any] = await Promise.all([make('closest'), make('any')]);
        return new CountedKernel(device, { closest, any });
    }

    /** One dispatch over `count` rays; returns {@link COUNTED_WORDS} words per ray. */
    async count(scene: TraceSceneBuffers, mode: 'closest' | 'any', rays: GPUBuffer, count: number): Promise<ArrayBuffer> {
        const device = this.device;
        const pipeline = this.pipelines[mode];
        const bytes = count * COUNTED_WORDS * 4;
        const output = device.createBuffer({ label: 'counted output', size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        try {
            const encoder = device.createCommandEncoder({ label: 'counted walk' });
            const pass = encoder.beginComputePass({ label: 'counted walk' });
            pass.setPipeline(pipeline);
            pass.setBindGroup(
                0,
                device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: scene.nodes } },
                        { binding: 1, resource: { buffer: scene.triangles } },
                        { binding: 2, resource: { buffer: scene.topLevel } },
                    ],
                }),
            );
            pass.setBindGroup(
                1,
                device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(1),
                    entries: [
                        { binding: 0, resource: { buffer: rays, offset: 0, size: count * RAY_FLOATS * 4 } },
                        { binding: 1, resource: { buffer: output } },
                    ],
                }),
            );
            const workgroups = Math.ceil(count / 64);
            const x = Math.min(workgroups, device.limits.maxComputeWorkgroupsPerDimension);
            pass.dispatchWorkgroups(x, Math.ceil(workgroups / x));
            pass.end();
            device.queue.submit([encoder.finish()]);
            return await readBuffer(device, output, bytes);
        } finally {
            output.destroy();
        }
    }
}

/** Share of rays where the counted walk's answer is the production kernel's. */
function countedAgreement(mode: 'closest' | 'any', production: ArrayBuffer, counted: ArrayBuffer, count: number): number {
    const countedFloats = new Float32Array(counted);
    const countedWords = new Uint32Array(counted);
    let agree = 0;
    if (mode === 'closest') {
        const floats = new Float32Array(production);
        const words = new Uint32Array(production);
        const recordWords = HIT_RECORD_BYTES / 4;
        for (let ray = 0; ray < count; ray++) {
            const t = floats[ray * recordWords + HIT_RECORD_WORD.t];
            const countedT = countedFloats[ray * COUNTED_WORDS + COUNTED_WORD.t];
            const bothMiss = t < 0 && countedT < 0;
            const sameHit =
                t >= 0 &&
                countedT >= 0 &&
                Math.abs(t - countedT) <= 1e-6 * Math.max(1, t) &&
                words[ray * recordWords + HIT_RECORD_WORD.instance] === countedWords[ray * COUNTED_WORDS + COUNTED_WORD.instance] &&
                words[ray * recordWords + HIT_RECORD_WORD.triangle] === countedWords[ray * COUNTED_WORDS + COUNTED_WORD.triangle];
            if (bothMiss || sameHit) agree++;
        }
    } else {
        const blocked = new Uint32Array(production);
        for (let ray = 0; ray < count; ray++) if ((blocked[ray] !== 0) === countedFloats[ray * COUNTED_WORDS + COUNTED_WORD.t] > 0) agree++;
    }
    return count > 0 ? agree / count : 1;
}

function percent(value: number): string {
    return `${(value * 100).toFixed(3)}%`;
}

export async function runSurfaceRaysPage(sceneId: string): Promise<void> {
    const startTime = performance.now();
    const lines: string[] = [];
    const progress = (message: string): void => {
        lines.push(message);
        console.info(`[surface-rays] ${message}`);
    };
    let deviceErrors: string[] = [];
    let adapterSummary: AdapterSummary | null = null;
    let data: SurfaceRaysData | null = null;
    let status: 'pass' | 'fail' = 'pass';
    let message: string | undefined;
    const gpuBuffers: GPUBuffer[] = [];
    try {
        const profileName = (new URLSearchParams(location.search).get('profile') ?? 'full') as SurfaceProfileName;
        const profile = SURFACE_PROFILES[profileName];
        if (!profile) throw new Error(`unknown profile "${profileName}" (known: ${Object.keys(SURFACE_PROFILES).join(', ')})`);
        const { device } = await requestBenchDevice();
        deviceErrors = watchDevice(device);
        adapterSummary = summarizeAdapterInfo(device.adapterInfo, device.features);
        const timer = await createDispatchTimer(device);
        const builder = createMeshBvhBuilder({ worker: 'auto' });
        const sceneBuffers: TraceSceneBuffers[] = [];
        let kernel: TraceBatchKernel | null = null;
        let compactKernel: TraceBatchKernel | null = null;
        try {
            progress(`${sceneId}: profile ${profileName}; loading geometry`);
            const scene = await loadBenchScene(sceneId, 'manifest');
            const camera = (await findManifest(sceneId)).cameras?.[0];
            if (!camera) throw new Error(`the manifest of "${sceneId}" has no camera bookmark to place screen probes from`);
            progress(`${sceneId}: ${scene.meshes.length} meshes, ${scene.instances.length} instances, ${scene.instancedTriangles} triangles; camera "${camera.name}"`);

            // Both trees from the same instances, every one static; only the option differs.
            const geometries = scene.meshes.map((mesh) => ({ positions: mesh.positions, indices: mesh.indices }));
            const instances: TraceSceneInstance[] = scene.instances.map((instance, index) => ({
                meshIndex: instance.meshIndex,
                instanceIndex: index,
                objectToWorld: instance.objectToWorld,
                static: true,
            }));
            const build = (geometry: MeshGeometry) => builder.build(geometry);
            let started = performance.now();
            const instancedData = await buildTraceScene(geometries, instances, { build });
            const instancedMs = performance.now() - started;
            await yieldToBrowser();
            started = performance.now();
            const mergedData = await buildTraceScene(geometries, instances, { build, mergeStaticInstances: true });
            const mergedMs = performance.now() - started;
            await yieldToBrowser();
            const trees: Record<Tree, TreeSummary> = { instanced: summarizeTree(instancedData, instancedMs), merged: summarizeTree(mergedData, mergedMs) };
            progress(
                `${sceneId}: instanced tree: ${trees.instanced.topLevelInstances} top-level instances, built in ${instancedMs.toFixed(0)} ms; ` +
                    `merged tree: ${trees.merged.topLevelInstances} (${mergedData.mergedGroups.length} merged groups holding ${mergedData.mergedGroups.reduce((total, group) => total + group.sourceInstances.length, 0)} instances), built in ${mergedMs.toFixed(0)} ms`,
            );

            const scenes: Record<Tree, TraceSceneBuffers> = {
                instanced: uploadTraceScene(device, instancedData, 'surface instanced'),
                merged: uploadTraceScene(device, mergedData, 'surface merged'),
            };
            sceneBuffers.push(scenes.instanced, scenes.merged);
            kernel = await TraceBatchKernel.create(device, { label: 'surface' });
            compactKernel = await TraceBatchKernel.create(device, { label: 'surface compact', hitNormal: false });
            const counted = await CountedKernel.create(device);

            // Placement, as RC1.5: one primary ray per 16x16 tile from the first bookmark, through the instanced tree.
            const view = { width: RC_VIEW.width, height: RC_VIEW.height, tileSize: RC_TILE_SIZE };
            const placementRays = makePlacementRays({ position: camera.position, target: camera.target, fovYDeg: camera.fovYDeg }, view, RC_HALTON_FIRST_INDEX);
            const placementBuffer = uploadRays(device, 'placement', placementRays);
            gpuBuffers.push(placementBuffer);
            const placementCount = tileGrid(view).count;
            const placementHits = await traceOnce(device, kernel, scenes.instanced, 'closest', placementBuffer, placementCount);
            const probes = surfaceProbes(placementRays, placementHits, scene.meshes, scene.instances);
            if (probes.count === 0) throw new Error(`no placement ray hit the scene from camera "${camera.name}"`);
            const grid = makeProbeGrid(scene.probeBox, probes.count);
            progress(`${sceneId}: ${probes.count} screen probes; air grid ${grid.dimensions.join(' x ')} = ${grid.probeCount} points in the manifest volume`);

            const surface = makeVariantProbeRays({
                origins: probeOrigins(probes),
                normals: probes.normals,
                count: probes.count,
                raysPerProbe: RC_RAYS_PER_PROBE,
                kind: 'cosine-stratified',
                random: createRandom(RC_SEED),
            });
            const airGrid = makeGridRays(grid, makeSphereDirections(RC_RAYS_PER_PROBE, RAY_DIRECTION_SEED));
            const sets: RaySet[] = [
                { name: 'surface', rays: surface, buffer: uploadRays(device, 'surface rays', surface), count: surface.length / RAY_FLOATS },
                { name: 'air-grid', rays: airGrid, buffer: uploadRays(device, 'air-grid rays', airGrid), count: airGrid.length / RAY_FLOATS },
            ];
            for (const set of sets) {
                gpuBuffers.push(set.buffer);
                const problems = rayProblems(set.name, set.rays);
                if (problems.length > 0) throw new Error(`bad rays: ${problems.join('; ')}`);
            }
            progress(`${sceneId}: ${sets.map((set) => `${set.name} ${set.count} rays`).join(', ')}`);

            // 1 and 2: the answers on both trees, the merged tree's agreement, the counted walk's work.
            const results: SurfaceSetResult[] = [];
            for (const set of sets) {
                for (const mode of MODES) {
                    const answers = {
                        instanced: await traceOnce(device, kernel, scenes.instanced, mode, set.buffer, set.count),
                        merged: await traceOnce(device, kernel, scenes.merged, mode, set.buffer, set.count),
                    };
                    for (const tree of ['instanced', 'merged'] as const) {
                        const counts = await counted.count(scenes[tree], mode, set.buffer, set.count);
                        const result: SurfaceSetResult = {
                            tree,
                            set: set.name,
                            mode,
                            countedAgreement: countedAgreement(mode, answers[tree], counts, set.count),
                            hitAgreement: tree === 'merged' && mode === 'closest' ? compareHitRecords(answers.instanced, answers.merged, set.count) : null,
                            blockedAgreement: tree === 'merged' && mode === 'any' ? ((agreeing) => agreeing.agree / Math.max(1, agreeing.rays))(compareBlocked(answers.instanced, answers.merged, set.count)) : null,
                            ...summariseCounts(counts, set.count),
                        };
                        results.push(result);
                        progress(
                            `${sceneId} (${tree}): ${set.name} ${mode}: steps ${result.meanSteps.toFixed(1)} (p90 ${result.p90Steps}), ` +
                                `instances entered ${result.meanInstanceEntries.toFixed(2)} (${result.meanInstanceEntriesWithoutHit.toFixed(2)} without a hit), ` +
                                `top-level nodes ${result.meanTopLevelNodes.toFixed(1)}, mesh nodes ${result.meanMeshNodes.toFixed(1)}, triangle tests ${result.meanTriangleTests.toFixed(1)}, ` +
                                `hit ${percent(result.hitFraction)}, counted agrees ${percent(result.countedAgreement)}` +
                                (result.hitAgreement ? `, merged finds the instanced hit on ${percent(result.hitAgreement.exact / result.hitAgreement.rays)} (${result.hitAgreement.sameDistance} ties, ${result.hitAgreement.hitMiss} hit/miss, ${result.hitAgreement.other} other)` : '') +
                                (result.blockedAgreement === null ? '' : `, merged blocks the same rays on ${percent(result.blockedAgreement)}`),
                        );
                    }
                    await yieldToBrowser();
                }
            }

            // 3: the paired timing.
            const pairs: PairKey[] = sets.flatMap((set) => MODES.map((mode) => ({ set: set.name, mode })));
            const entries: PairedEntry[] = pairs.flatMap((pair) =>
                (['instanced', 'merged'] as const).map((tree): PairedEntry => ({ tree, set: pair.set as SetName, mode: pair.mode, mraysPerRound: [], singleMraysPerRound: [], repeatMrays: Number.NaN })),
            );
            const entryOf = (tree: Tree, pair: PairKey): PairedEntry => {
                const entry = entries.find((candidate) => candidate.tree === tree && candidate.set === pair.set && candidate.mode === pair.mode);
                if (!entry) throw new Error(`no paired entry ${tree} ${pair.set} ${pair.mode}`);
                return entry;
            };
            const resultBuffers = new Map<string, GPUBuffer>();
            for (const pair of pairs) {
                const set = sets.find((candidate) => candidate.name === pair.set) as RaySet;
                const buffer = device.createBuffer({
                    label: `surface ${pair.set} ${pair.mode} results`,
                    size: set.count * (pair.mode === 'closest' ? HIT_RECORD_BYTES : 4),
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
                });
                gpuBuffers.push(buffer);
                resultBuffers.set(`${pair.set}/${pair.mode}`, buffer);
            }
            const timeStep = async (tree: Tree, pair: PairKey): Promise<{ loaded: number; single: number }> => {
                const set = sets.find((candidate) => candidate.name === pair.set) as RaySet;
                const results = resultBuffers.get(`${pair.set}/${pair.mode}`) as GPUBuffer;
                const encode = (encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites): void =>
                    (kernel as TraceBatchKernel).encode(encoder, scenes[tree], { mode: pair.mode, rays: set.buffer, results, count: set.count, timestampWrites });
                const single = await measureDispatches(() => timer.time((encoder, timestampWrites) => encode(encoder, timestampWrites)), profile.warmups, profile.timed);
                const loaded = await timeLoaded(device, encode, set.count, single.medianMs, profile.loadedMs);
                await yieldToBrowser();
                return { loaded: loaded.mraysPerSecond, single: (set.count / (single.medianMs * 1e3)) };
            };
            progress(`${sceneId}: paired timing, ${profile.rounds} rounds of ${pairs.length} pairs, instanced and merged in turn, the order reversed every other round`);
            for (let round = 0; round < profile.rounds; round++) {
                for (const step of pairedRoundOrder(pairs, round)) {
                    const entry = entryOf(step.tree, step.pair);
                    const timed = await timeStep(step.tree, step.pair);
                    entry.mraysPerRound.push(timed.loaded);
                    entry.singleMraysPerRound.push(timed.single);
                }
                progress(
                    `${sceneId}: round ${round + 1} of ${profile.rounds}: ` +
                        pairs
                            .map((pair) => `${pair.set} ${pair.mode} merged/instanced ${(entryOf('merged', pair).mraysPerRound[round] / entryOf('instanced', pair).mraysPerRound[round]).toFixed(3)}`)
                            .join(', '),
                );
            }
            // The repeat at the end of the batch: the surface sets again, in the order the next round would go.
            const repeatPairs = pairs.filter((pair) => pair.set === 'surface');
            for (const step of pairedRoundOrder(repeatPairs, profile.rounds)) {
                entryOf(step.tree, step.pair).repeatMrays = (await timeStep(step.tree, step.pair)).loaded;
            }

            const ratios: PairedRatio[] = pairs.map((pair) =>
                ratioOfRounds(`merged_over_instanced/${pair.set}/${pair.mode}`, entryOf('merged', pair).mraysPerRound, entryOf('instanced', pair).mraysPerRound),
            );
            const repeatRatios = repeatPairs.map((pair) => ({ name: `merged_over_instanced/${pair.set}/${pair.mode}`, ratio: entryOf('merged', pair).repeatMrays / entryOf('instanced', pair).repeatMrays }));
            for (const ratio of ratios) {
                const repeat = repeatRatios.find((candidate) => candidate.name === ratio.name);
                progress(
                    `${sceneId}: ${ratio.name}: median ${ratio.median.toFixed(3)} (${ratio.min.toFixed(3)}-${ratio.max.toFixed(3)} over ${profile.rounds} rounds; per round ${ratio.perRound.map((value) => value.toFixed(3)).join(', ')})` +
                        (repeat ? `; repeat at the end ${repeat.ratio.toFixed(3)}` : ''),
                );
            }
            for (const pair of pairs) {
                const instanced = entryOf('instanced', pair);
                const merged = entryOf('merged', pair);
                const middle = (values: number[]): number => [...values].sort((a, b) => a - b)[values.length >> 1];
                progress(`${sceneId}: ${pair.set} ${pair.mode}: instanced ${middle(instanced.mraysPerRound).toFixed(1)} Mrays/s, merged ${middle(merged.mraysPerRound).toFixed(1)} Mrays/s (median of rounds, loaded)`);
            }

            // 4: the 40-byte hit record against the 24-byte one, closest-hit, on each tree and set.
            const recordCases: HitRecordCase[] = sets.flatMap((set) => (['instanced', 'merged'] as const).map((tree) => ({ tree, set: set.name })));
            const recordBytes: Record<HitRecord, number> = { record40: HIT_RECORD_BYTES, record24: HIT_RECORD_COMPACT_BYTES };
            const recordEntries: HitRecordEntry[] = recordCases.flatMap((entry) =>
                (['record40', 'record24'] as const).map((record): HitRecordEntry => ({ tree: entry.tree, set: entry.set as SetName, record, recordBytes: recordBytes[record], mraysPerRound: [], singleMraysPerRound: [] })),
            );
            const recordEntryOf = (entry: HitRecordCase, record: HitRecord): HitRecordEntry => {
                const found = recordEntries.find((candidate) => candidate.tree === entry.tree && candidate.set === entry.set && candidate.record === record);
                if (!found) throw new Error(`no hit-record entry ${entry.tree} ${entry.set} ${record}`);
                return found;
            };
            const fullKernel = kernel;
            const narrowKernel = compactKernel;
            const recordKernels: Record<HitRecord, TraceBatchKernel> = { record40: fullKernel, record24: narrowKernel };
            const recordResults = new Map<string, GPUBuffer>();
            for (const record of ['record40', 'record24'] as const) {
                for (const set of sets) {
                    const buffer = device.createBuffer({ label: `surface ${set.name} ${record} results`, size: set.count * recordBytes[record], usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
                    gpuBuffers.push(buffer);
                    recordResults.set(`${set.name}/${record}`, buffer);
                }
            }
            const timeRecord = async (entry: HitRecordCase, record: HitRecord): Promise<{ loaded: number; single: number }> => {
                const set = sets.find((candidate) => candidate.name === entry.set) as RaySet;
                const results = recordResults.get(`${set.name}/${record}`) as GPUBuffer;
                const encode = (encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites): void =>
                    recordKernels[record].encode(encoder, scenes[entry.tree], { mode: 'closest', rays: set.buffer, results, count: set.count, timestampWrites });
                const single = await measureDispatches(() => timer.time((encoder, timestampWrites) => encode(encoder, timestampWrites)), profile.warmups, profile.timed);
                const loaded = await timeLoaded(device, encode, set.count, single.medianMs, profile.loadedMs);
                await yieldToBrowser();
                return { loaded: loaded.mraysPerSecond, single: set.count / (single.medianMs * 1e3) };
            };
            const agreement: HitRecordResult['agreement'] = [];
            for (const entry of recordCases) {
                const set = sets.find((candidate) => candidate.name === entry.set) as RaySet;
                const full = new Uint32Array(await traceOnce(device, fullKernel, scenes[entry.tree], 'closest', set.buffer, set.count));
                const compact = new Uint32Array(await traceOnce(device, narrowKernel, scenes[entry.tree], 'closest', set.buffer, set.count, HIT_RECORD_COMPACT_BYTES));
                const fullWords = HIT_RECORD_BYTES / 4;
                const compactWords = HIT_RECORD_COMPACT_BYTES / 4;
                // Words 0-3 and the two barycentric words (the compact record is words 0-5 of the full one).
                let same = 0;
                for (let ray = 0; ray < set.count; ray++) {
                    let equal = true;
                    for (const word of [HIT_RECORD_COMPACT_WORD.t, HIT_RECORD_COMPACT_WORD.instance, HIT_RECORD_COMPACT_WORD.triangle, HIT_RECORD_COMPACT_WORD.frontFace, HIT_RECORD_COMPACT_WORD.barycentrics, HIT_RECORD_COMPACT_WORD.barycentrics + 1]) {
                        equal &&= full[ray * fullWords + word] === compact[ray * compactWords + word];
                    }
                    if (equal) same++;
                }
                agreement.push({ tree: entry.tree, set: entry.set as SetName, rays: set.count, same });
            }
            progress(`${sceneId}: hit record 40 against 24 bytes, ${profile.rounds} rounds of ${recordCases.length} cases, the order reversed every other round; compact kernel agrees on ${agreement.map((a) => `${a.tree} ${a.set} ${percent(a.same / a.rays)}`).join(', ')}`);
            for (let round = 0; round < profile.rounds; round++) {
                for (const step of hitRecordRoundOrder(recordCases, round)) {
                    const entry = recordEntryOf(step.case, step.record);
                    const timed = await timeRecord(step.case, step.record);
                    entry.mraysPerRound.push(timed.loaded);
                    entry.singleMraysPerRound.push(timed.single);
                }
                progress(
                    `${sceneId}: hit record round ${round + 1} of ${profile.rounds}: ` +
                        recordCases.map((entry) => `${entry.tree} ${entry.set} 40/24 ${(recordEntryOf(entry, 'record40').mraysPerRound[round] / recordEntryOf(entry, 'record24').mraysPerRound[round]).toFixed(3)}`).join(', '),
                );
            }
            const recordRatios: PairedRatio[] = recordCases.map((entry) =>
                ratioOfRounds(`hit_record_40_over_24/${entry.tree}/${entry.set}/closest`, recordEntryOf(entry, 'record40').mraysPerRound, recordEntryOf(entry, 'record24').mraysPerRound),
            );
            for (const ratio of recordRatios) {
                progress(`${sceneId}: ${ratio.name}: median ${ratio.median.toFixed(3)} (${ratio.min.toFixed(3)}-${ratio.max.toFixed(3)} over ${profile.rounds} rounds; per round ${ratio.perRound.map((value) => value.toFixed(3)).join(', ')})`);
            }
            const hitRecord: HitRecordResult = { rounds: profile.rounds, entries: recordEntries, ratios: recordRatios, agreement };

            data = {
                benchmark: 'surface-rays',
                scene: sceneId,
                meshCount: scene.meshes.length,
                instanceCount: scene.instances.length,
                instancedTriangles: scene.instancedTriangles,
                camera: camera.name,
                probesPlaced: probes.count,
                gridProbes: grid.probeCount,
                raysPerProbe: RC_RAYS_PER_PROBE,
                raysPerSet: Object.fromEntries(sets.map((set) => [set.name, set.count])),
                userAgent: navigator.userAgent,
                browserVersion: await fullBrowserVersion(),
                timingSource: timer.source,
                trees,
                sets: results,
                paired: { rounds: profile.rounds, profile: profileName, entries, ratios, repeatRatios },
                hitRecord,
            };

            // The verdict is correctness only.
            const failures: string[] = [];
            if (trees.merged.topLevelInstances >= trees.instanced.topLevelInstances) {
                failures.push(`the merged tree has ${trees.merged.topLevelInstances} top-level instances, the instanced one ${trees.instanced.topLevelInstances}: nothing was merged`);
            }
            for (const result of results) {
                if (result.countedAgreement < MIN_COUNTED_AGREEMENT) failures.push(`the counted copy of the walk disagrees with the production kernel on ${result.tree} ${result.set} ${result.mode}: ${percent(result.countedAgreement)}`);
                if (result.hitAgreement && result.hitAgreement.exact / result.hitAgreement.rays < MIN_HIT_AGREEMENT) {
                    failures.push(`the merged tree finds the instanced tree's hit (instance and triangle) on only ${percent(result.hitAgreement.exact / result.hitAgreement.rays)} of ${result.set} ${result.mode} rays`);
                }
                if (result.blockedAgreement !== null && result.blockedAgreement < MIN_HIT_AGREEMENT) {
                    failures.push(`the merged tree blocks the same ${result.set} rays as the instanced one on only ${percent(result.blockedAgreement)}`);
                }
            }
            for (const entry of agreement) {
                if (entry.same / entry.rays < MIN_HIT_AGREEMENT) failures.push(`the compact-record kernel differs from the full one on ${entry.tree} ${entry.set}: ${percent(entry.same / entry.rays)} of rays the same`);
            }
            if (failures.length > 0) {
                status = 'fail';
                message = failures.join(' | ');
            }
        } finally {
            kernel?.destroy();
            compactKernel?.destroy();
            for (const buffers of sceneBuffers) buffers.destroy();
            builder.dispose();
            timer.dispose();
            for (const buffer of gpuBuffers) buffer.destroy();
        }
        await drainDevice(device);
    } catch (error) {
        status = 'fail';
        message = error instanceof Error ? error.message : String(error);
    }
    reportGpuTest({
        status,
        message,
        data: (data ?? { benchmark: 'surface-rays' }) as unknown as Record<string, unknown>,
        adapter: adapterSummary,
        deviceErrors,
        startTime,
        logLines: lines,
    });
}

/** One untimed dispatch; returns the answers (hit records for closest, one word per ray for any). */
async function traceOnce(device: GPUDevice, kernel: TraceBatchKernel, scene: TraceSceneBuffers, mode: 'closest' | 'any', rays: GPUBuffer, count: number, recordBytes = HIT_RECORD_BYTES): Promise<ArrayBuffer> {
    const bytes = count * (mode === 'closest' ? recordBytes : 4);
    const results = device.createBuffer({ label: `surface ${mode} answers`, size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    try {
        const encoder = device.createCommandEncoder({ label: `surface ${mode}` });
        kernel.encode(encoder, scene, { mode, rays, results, count });
        device.queue.submit([encoder.finish()]);
        return await readBuffer(device, results, bytes);
    } finally {
        results.destroy();
    }
}

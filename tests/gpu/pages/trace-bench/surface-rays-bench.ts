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

import {
    buildTraceScene,
    createMeshBvhBuilder,
    HIT_RECORD_BYTES,
    HIT_RECORD_COMPACT_BYTES,
    HIT_RECORD_COMPACT_WORD,
    HIT_RECORD_WORD,
    RAY_FLOATS,
    type MeshBvh,
    type MeshGeometry,
    type TraceSceneData,
    type TraceSceneInstance,
} from '../../../../src/trace/index.ts';
import { createRandom } from '../../../../src/trace/oracle-fixtures.ts';
import type { AdapterSummary } from '../../contract.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { TraceBatchKernel, uploadTraceScene, type TraceSceneBuffers } from '../../../../src/trace/trace-gpu.ts';
import { RAY_DIRECTION_SEED, RC_HALTON_FIRST_INDEX, RC_RAYS_PER_PROBE, RC_SEED, RC_TILE_SIZE, RC_VIEW } from './config.ts';
import { COUNTED_WORD, COUNTED_WORDS, countedKernelWgsl } from './counted-walk-wgsl.ts';
import { createDispatchTimer, measureDispatches } from './dispatch-timer.ts';
import { timeLoaded } from './loaded-timing.ts';
import { readBuffer, uploadRays, yieldToBrowser } from './ours-bench.ts';
import {
    hitRecordRoundOrder,
    leafLayoutRoundOrder,
    pairedRoundOrder,
    type NodeLayout,
    ratioOfRounds,
    type HitRecord,
    type HitRecordCase,
    type LeafLayout,
    type LeafLayoutCase,
    type PairKey,
    type PairedRatio,
    type Tree,
} from './paired-ratio.ts';
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

/** One tree, set and mode in one triangle format, per round. */
export interface LeafLayoutEntry {
    tree: Tree;
    set: SetName;
    mode: 'closest' | 'any';
    layout: LeafLayout;
    mraysPerRound: number[];
    singleMraysPerRound: number[];
}

/**
 * Indexed leaves against records (docs/internals/design/indexed-leaves.md): the same
 * tinybvh trees, the scene assembled in each format, timed in paired rounds;
 * ratios `indexed_over_records/<tree>/<set>/<mode>` and
 * `indexed_packed_over_records/...` (above 1: indexed is faster).
 */
export interface LeafLayoutResult {
    rounds: number;
    entries: LeafLayoutEntry[];
    ratios: PairedRatio[];
    /** Rays whose answer (the whole hit record, or the blocked word) is bit-identical to the records scene's: must be all. */
    agreement: { tree: Tree; set: SetName; mode: 'closest' | 'any'; layout: LeafLayout; rays: number; same: number }[];
    /** The scenes' bytes: nodes, triangles, positions, top level. */
    bytes: Record<string, { nodes: number; triangles: number; positions: number; topLevel: number }>;
}

/**
 * Quantized nodes against f32 nodes (docs/internals/design/compressed-nodes.md): the
 * same tinybvh trees, records, the scene assembled with each node format,
 * timed in paired rounds; ratios `quantized_over_f32/<tree>/<set>/<mode>`
 * (above 1: quantized is faster). Hits: the same `t`, bit for bit, and the
 * same hit record unless another triangle lies at exactly that `t` (a tie,
 * where the walk's order picks and looser boxes can change it).
 */
type AnyHitOrderName = 'far' | 'larger' | 'near';

/** Section 7: the any-hit child orders against the default ('far'), any-hit on records. */
export interface AnyHitOrderResult {
    rounds: number;
    entries: { tree: Tree; set: SetName; order: AnyHitOrderName; mraysPerRound: number[] }[];
    ratios: PairedRatio[];
    /** Rays blocked the same as with 'far' (must be all). */
    agreement: { tree: Tree; set: SetName; order: AnyHitOrderName; rays: number; same: number }[];
    counts: { tree: Tree; set: SetName; order: AnyHitOrderName; counts: CountSummary; countedAgreement: number }[];
}

export interface NodeFormatResult {
    rounds: number;
    entries: (Omit<LeafLayoutEntry, 'layout'> & { layout: NodeLayout })[];
    ratios: PairedRatio[];
    /** Per case: rays answered bit for bit as f32, ties (same t, another triangle), and different (must be 0). */
    agreement: { tree: Tree; set: SetName; mode: 'closest' | 'any'; rays: number; same: number; ties: number; different: number }[];
    /** The counted walk's work per ray in each format, and the quantized counted walk's agreement with its production kernel. */
    counts: { tree: Tree; set: SetName; mode: 'closest' | 'any'; f32: CountSummary; quantized: CountSummary; countedAgreement: number }[];
    /** Node bytes per format and tree. */
    bytes: Record<string, number>;
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
    triangleLeaves?: LeafLayoutResult;
    nodeFormat?: NodeFormatResult;
    combinedLeaves?: CombinedLeavesResult;
    foldedDecode?: FoldedDecodeResult;
    anyHitOrder?: AnyHitOrderResult;
}

/**
 * Section 9: the folded decode of quantized nodes (`quantizedDecode: 'folded'`,
 * docs/internals/design/compressed-nodes.md, question 6) against the exact decode, on the
 * quantized records scenes of section 6 (and section 8's quantized indexed ones when it ran);
 * ratios `folded_over_exact/<tree>/<set>/<mode>` and `folded_indexed_over_exact_indexed/...`
 * (above 1: folded is faster). Answers: the exact walk's, but for ties (same t, another
 * triangle; folded boxes are a few ulps wider). `?skip=folded` leaves it out.
 */
export interface FoldedDecodeResult {
    rounds: number;
    entries: { tree: Tree; set: SetName; mode: 'closest' | 'any'; layout: LeafLayout; decode: 'exact' | 'folded'; mraysPerRound: number[]; singleMraysPerRound: number[] }[];
    ratios: PairedRatio[];
    agreement: { tree: Tree; set: SetName; mode: 'closest' | 'any'; layout: LeafLayout; rays: number; same: number; ties: number; different: number }[];
}

/**
 * Section 8: indexed leaves on quantized nodes against quantized records, the pairing the
 * `triangleLeaves` default waits on; ratios `quantized_indexed_over_quantized_records/<tree>/<set>/<mode>`
 * and `quantized_indexed_packed_over_quantized_records/...` (above 1: indexed is faster).
 * `?skip=combined` leaves it out.
 */
export interface CombinedLeavesResult {
    rounds: number;
    entries: LeafLayoutEntry[];
    ratios: PairedRatio[];
    /** Per case and format: identical answers, ties (same t, another triangle) and different ones (must be 0). */
    agreement: { tree: Tree; set: SetName; mode: 'closest' | 'any'; layout: LeafLayout; rays: number; same: number; ties: number; different: number }[];
    bytes: LeafLayoutResult['bytes'];
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

    static async create(device: GPUDevice, format: { readonly nodeFormat?: 'f32' | 'quantized'; readonly anyHitOrder?: 'far' | 'near' | 'larger' } = {}): Promise<CountedKernel> {
        const make = (mode: 'closest' | 'any') =>
            device.createComputePipelineAsync({
                label: `counted walk ${mode}`,
                layout: 'auto',
                compute: { module: device.createShaderModule({ label: `counted walk ${mode}`, code: countedKernelWgsl(mode, format) }), entryPoint: 'benchCounted' },
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
        // `?skip=hit-record,leaves,combined,folded,any-order`: leave out sections 4, 5, 8, 9 and 7 (for a run that only wants the others).
        const skip = new Set((new URLSearchParams(location.search).get('skip') ?? '').split(',').filter((name) => name.length > 0));
        if (!profile) throw new Error(`unknown profile "${profileName}" (known: ${Object.keys(SURFACE_PROFILES).join(', ')})`);
        const { device } = await requestBenchDevice();
        deviceErrors = watchDevice(device);
        adapterSummary = summarizeAdapterInfo(device.adapterInfo, device.features);
        const timer = await createDispatchTimer(device);
        const builder = createMeshBvhBuilder({ worker: 'auto' });
        const sceneBuffers: TraceSceneBuffers[] = [];
        let kernel: TraceBatchKernel | null = null;
        let compactKernel: TraceBatchKernel | null = null;
        const extraKernels: TraceBatchKernel[] = [];
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
            // Each geometry's tree is kept, so the indexed scenes below have exactly these trees.
            const built = new Map<MeshGeometry, Promise<MeshBvh>>();
            const build = (geometry: MeshGeometry) => {
                let tree = built.get(geometry);
                if (!tree) built.set(geometry, (tree = builder.build(geometry)));
                return tree;
            };
            let started = performance.now();
            // The baseline is the layout before 2026-11-13's defaults (instanced, f32 nodes), named so its rows keep their meaning.
            const instancedData = await buildTraceScene(geometries, instances, { build, mergeStaticInstances: false, nodeFormat: 'f32' });
            const instancedMs = performance.now() - started;
            await yieldToBrowser();
            started = performance.now();
            const mergedData = await buildTraceScene(geometries, instances, { build, mergeStaticInstances: true, nodeFormat: 'f32' });
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
            kernel = await TraceBatchKernel.create(device, { label: 'surface', nodeFormat: 'f32' });
            compactKernel = await TraceBatchKernel.create(device, { label: 'surface compact', hitNormal: false, nodeFormat: 'f32' });
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
            const runRecords = !skip.has('hit-record');
            for (const entry of runRecords ? recordCases : []) {
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
            for (let round = 0; round < (runRecords ? profile.rounds : 0); round++) {
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
            const recordRatios: PairedRatio[] = (runRecords ? recordCases : []).map((entry) =>
                ratioOfRounds(`hit_record_40_over_24/${entry.tree}/${entry.set}/closest`, recordEntryOf(entry, 'record40').mraysPerRound, recordEntryOf(entry, 'record24').mraysPerRound),
            );
            for (const ratio of recordRatios) {
                progress(`${sceneId}: ${ratio.name}: median ${ratio.median.toFixed(3)} (${ratio.min.toFixed(3)}-${ratio.max.toFixed(3)} over ${profile.rounds} rounds; per round ${ratio.perRound.map((value) => value.toFixed(3)).join(', ')})`);
            }
            const hitRecord: HitRecordResult = { rounds: profile.rounds, entries: recordEntries, ratios: recordRatios, agreement };

            // 5: indexed leaves against records (docs/internals/design/indexed-leaves.md): the same trees, both position layouts.
            const leafScenes: Record<LeafLayout, Record<Tree, TraceSceneBuffers>> = { records: scenes, indexed: {} as Record<Tree, TraceSceneBuffers>, 'indexed-packed': {} as Record<Tree, TraceSceneBuffers> };
            const leafBytes: LeafLayoutResult['bytes'] = {};
            const runLeaves = !skip.has('leaves');
            for (const tree of runLeaves ? (['instanced', 'merged'] as const) : []) {
                const recordsData = tree === 'instanced' ? instancedData : mergedData;
                leafBytes[`records/${tree}`] = { nodes: recordsData.nodes.byteLength, triangles: recordsData.triangles.byteLength, positions: 0, topLevel: recordsData.topLevel.buffer.byteLength };
                for (const [layout, positionLayout] of [['indexed', 'vec4'], ['indexed-packed', 'packed']] as const) {
                    const indexed = await buildTraceScene(geometries, instances, { build, mergeStaticInstances: tree === 'merged', triangleLeaves: 'indexed', positionLayout, nodeFormat: 'f32' });
                    leafBytes[`${layout}/${tree}`] = { nodes: indexed.nodes.byteLength, triangles: indexed.triangles.byteLength, positions: indexed.positions.byteLength, topLevel: indexed.topLevel.buffer.byteLength };
                    const buffers = uploadTraceScene(device, indexed, `surface ${layout} ${tree}`);
                    sceneBuffers.push(buffers);
                    leafScenes[layout][tree] = buffers;
                    await yieldToBrowser();
                }
            }
            const leafKernels: Record<LeafLayout, TraceBatchKernel> = {
                records: fullKernel,
                indexed: await TraceBatchKernel.create(device, { label: 'surface indexed', triangleLeaves: 'indexed', nodeFormat: 'f32' }),
                'indexed-packed': await TraceBatchKernel.create(device, { label: 'surface indexed packed', triangleLeaves: 'indexed', positionLayout: 'packed', nodeFormat: 'f32' }),
            };
            extraKernels.push(leafKernels.indexed, leafKernels['indexed-packed']);
            const leafLayouts: LeafLayout[] = ['records', 'indexed', 'indexed-packed'];
            const leafCases: LeafLayoutCase[] = (['instanced', 'merged'] as const).flatMap((tree) => sets.flatMap((set) => MODES.map((mode) => ({ tree, set: set.name, mode }))));
            const leafEntries: LeafLayoutEntry[] = leafCases.flatMap((entry) =>
                leafLayouts.map((layout): LeafLayoutEntry => ({ tree: entry.tree, set: entry.set as SetName, mode: entry.mode, layout, mraysPerRound: [], singleMraysPerRound: [] })),
            );
            const leafEntryOf = (entry: LeafLayoutCase, layout: LeafLayout): LeafLayoutEntry => {
                const found = leafEntries.find((candidate) => candidate.tree === entry.tree && candidate.set === entry.set && candidate.mode === entry.mode && candidate.layout === layout);
                if (!found) throw new Error(`no leaf entry ${entry.tree} ${entry.set} ${entry.mode} ${layout}`);
                return found;
            };
            // Hits must be identical in every format: the same trees, the same corner bits, the same arithmetic.
            const leafAgreement: LeafLayoutResult['agreement'] = [];
            for (const entry of runLeaves ? leafCases : []) {
                const set = sets.find((candidate) => candidate.name === entry.set) as RaySet;
                const expected = new Uint32Array(await traceOnce(device, fullKernel, scenes[entry.tree], entry.mode, set.buffer, set.count));
                for (const layout of ['indexed', 'indexed-packed'] as const) {
                    const actual = new Uint32Array(await traceOnce(device, leafKernels[layout], leafScenes[layout][entry.tree], entry.mode, set.buffer, set.count));
                    const words = expected.length / set.count;
                    let same = 0;
                    for (let ray = 0; ray < set.count; ray++) {
                        let equal = true;
                        for (let word = 0; word < words && equal; word++) equal = expected[ray * words + word] === actual[ray * words + word];
                        if (equal) same++;
                    }
                    leafAgreement.push({ tree: entry.tree, set: entry.set as SetName, mode: entry.mode, layout, rays: set.count, same });
                }
            }
            const leafResults = new Map<string, GPUBuffer>();
            for (const set of sets) {
                for (const mode of MODES) {
                    const buffer = device.createBuffer({ label: `surface ${set.name} ${mode} leaf results`, size: set.count * (mode === 'closest' ? HIT_RECORD_BYTES : 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
                    gpuBuffers.push(buffer);
                    leafResults.set(`${set.name}/${mode}`, buffer);
                }
            }
            const timeLeaf = async (entry: LeafLayoutCase, layout: LeafLayout): Promise<{ loaded: number; single: number }> => {
                const set = sets.find((candidate) => candidate.name === entry.set) as RaySet;
                const results = leafResults.get(`${set.name}/${entry.mode}`) as GPUBuffer;
                const encode = (encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites): void =>
                    leafKernels[layout].encode(encoder, leafScenes[layout][entry.tree], { mode: entry.mode, rays: set.buffer, results, count: set.count, timestampWrites });
                const single = await measureDispatches(() => timer.time((encoder, timestampWrites) => encode(encoder, timestampWrites)), profile.warmups, profile.timed);
                const loaded = await timeLoaded(device, encode, set.count, single.medianMs, profile.loadedMs);
                await yieldToBrowser();
                return { loaded: loaded.mraysPerSecond, single: set.count / (single.medianMs * 1e3) };
            };
            progress(`${sceneId}: indexed leaves against records, ${profile.rounds} rounds of ${leafCases.length} cases in ${leafLayouts.length} formats, the order reversed every other round; identical answers on ${leafAgreement.filter((a) => a.same === a.rays).length} of ${leafAgreement.length}`);
            for (let round = 0; round < (runLeaves ? profile.rounds : 0); round++) {
                for (const step of leafLayoutRoundOrder(leafCases, leafLayouts, round)) {
                    const entry = leafEntryOf(step.case, step.layout);
                    const timed = await timeLeaf(step.case, step.layout);
                    entry.mraysPerRound.push(timed.loaded);
                    entry.singleMraysPerRound.push(timed.single);
                }
                progress(
                    `${sceneId}: leaf round ${round + 1} of ${profile.rounds}: ` +
                        leafCases.map((entry) => `${entry.tree} ${entry.set} ${entry.mode} indexed/records ${(leafEntryOf(entry, 'indexed').mraysPerRound[round] / leafEntryOf(entry, 'records').mraysPerRound[round]).toFixed(3)}`).join(', '),
                );
            }
            const leafRatios: PairedRatio[] = (runLeaves ? leafCases : []).flatMap((entry) => [
                ratioOfRounds(`indexed_over_records/${entry.tree}/${entry.set}/${entry.mode}`, leafEntryOf(entry, 'indexed').mraysPerRound, leafEntryOf(entry, 'records').mraysPerRound),
                ratioOfRounds(`indexed_packed_over_records/${entry.tree}/${entry.set}/${entry.mode}`, leafEntryOf(entry, 'indexed-packed').mraysPerRound, leafEntryOf(entry, 'records').mraysPerRound),
            ]);
            for (const ratio of leafRatios) {
                progress(`${sceneId}: ${ratio.name}: median ${ratio.median.toFixed(3)} (${ratio.min.toFixed(3)}-${ratio.max.toFixed(3)} over ${profile.rounds} rounds)`);
            }
            const triangleLeaves: LeafLayoutResult = { rounds: profile.rounds, entries: leafEntries, ratios: leafRatios, agreement: leafAgreement, bytes: leafBytes };

            // 6: quantized nodes against f32 nodes (docs/internals/design/compressed-nodes.md): the same trees, records.
            const nodeLayouts: NodeLayout[] = ['f32', 'quantized'];
            const nodeScenes: Record<NodeLayout, Record<Tree, TraceSceneBuffers>> = { f32: scenes, quantized: {} as Record<Tree, TraceSceneBuffers> };
            const nodeBytes: Record<string, number> = {};
            for (const tree of ['instanced', 'merged'] as const) {
                nodeBytes[`f32/${tree}`] = (tree === 'instanced' ? instancedData : mergedData).nodes.byteLength;
                const quantized = await buildTraceScene(geometries, instances, { build, mergeStaticInstances: tree === 'merged', nodeFormat: 'quantized' });
                nodeBytes[`quantized/${tree}`] = quantized.nodes.byteLength;
                const buffers = uploadTraceScene(device, quantized, `surface quantized ${tree}`);
                sceneBuffers.push(buffers);
                nodeScenes.quantized[tree] = buffers;
                await yieldToBrowser();
            }
            const nodeKernels: Record<NodeLayout, TraceBatchKernel> = { f32: fullKernel, quantized: await TraceBatchKernel.create(device, { label: 'surface quantized', nodeFormat: 'quantized' }) };
            extraKernels.push(nodeKernels.quantized);
            const quantizedCounted = await CountedKernel.create(device, { nodeFormat: 'quantized' });
            const nodeEntries: NodeFormatResult['entries'] = leafCases.flatMap((entry) =>
                nodeLayouts.map((layout) => ({ tree: entry.tree, set: entry.set as SetName, mode: entry.mode, layout, mraysPerRound: [] as number[], singleMraysPerRound: [] as number[] })),
            );
            const nodeEntryOf = (entry: LeafLayoutCase, layout: NodeLayout) => {
                const found = nodeEntries.find((candidate) => candidate.tree === entry.tree && candidate.set === entry.set && candidate.mode === entry.mode && candidate.layout === layout);
                if (!found) throw new Error(`no node entry ${entry.tree} ${entry.set} ${entry.mode} ${layout}`);
                return found;
            };
            const nodeAgreement: NodeFormatResult['agreement'] = [];
            const nodeCounts: NodeFormatResult['counts'] = [];
            for (const entry of leafCases) {
                const set = sets.find((candidate) => candidate.name === entry.set) as RaySet;
                const expected = new Uint32Array(await traceOnce(device, fullKernel, scenes[entry.tree], entry.mode, set.buffer, set.count));
                const answers = await traceOnce(device, nodeKernels.quantized, nodeScenes.quantized[entry.tree], entry.mode, set.buffer, set.count);
                const actual = new Uint32Array(answers);
                const words = expected.length / set.count;
                let same = 0;
                let ties = 0;
                for (let ray = 0; ray < set.count; ray++) {
                    let equal = true;
                    for (let word = 0; word < words && equal; word++) equal = expected[ray * words + word] === actual[ray * words + word];
                    if (equal) same++;
                    else if (entry.mode === 'closest' && expected[ray * words + HIT_RECORD_WORD.t] === actual[ray * words + HIT_RECORD_WORD.t]) ties++;
                }
                nodeAgreement.push({ tree: entry.tree, set: entry.set as SetName, mode: entry.mode, rays: set.count, same, ties, different: set.count - same - ties });
                const f32Counts = summariseCounts(await counted.count(scenes[entry.tree], entry.mode, set.buffer, set.count), set.count);
                const quantizedCounts = await quantizedCounted.count(nodeScenes.quantized[entry.tree], entry.mode, set.buffer, set.count);
                nodeCounts.push({
                    tree: entry.tree,
                    set: entry.set as SetName,
                    mode: entry.mode,
                    f32: f32Counts,
                    quantized: summariseCounts(quantizedCounts, set.count),
                    countedAgreement: countedAgreement(entry.mode, answers, quantizedCounts, set.count),
                });
                const last = nodeCounts[nodeCounts.length - 1];
                progress(
                    `${sceneId}: quantized ${entry.tree} ${entry.set} ${entry.mode}: steps ${last.f32.meanSteps.toFixed(2)} -> ${last.quantized.meanSteps.toFixed(2)}, ` +
                        `mesh nodes ${last.f32.meanMeshNodes.toFixed(2)} -> ${last.quantized.meanMeshNodes.toFixed(2)}, triangle tests ${last.f32.meanTriangleTests.toFixed(2)} -> ${last.quantized.meanTriangleTests.toFixed(2)}; ` +
                        `same ${same}, ties ${ties}, different ${set.count - same - ties} of ${set.count}`,
                );
                await yieldToBrowser();
            }
            const timeNode = async (entry: LeafLayoutCase, layout: NodeLayout): Promise<{ loaded: number; single: number }> => {
                const set = sets.find((candidate) => candidate.name === entry.set) as RaySet;
                const results = leafResults.get(`${set.name}/${entry.mode}`) as GPUBuffer;
                const encode = (encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites): void =>
                    nodeKernels[layout].encode(encoder, nodeScenes[layout][entry.tree], { mode: entry.mode, rays: set.buffer, results, count: set.count, timestampWrites });
                const single = await measureDispatches(() => timer.time((encoder, timestampWrites) => encode(encoder, timestampWrites)), profile.warmups, profile.timed);
                const loaded = await timeLoaded(device, encode, set.count, single.medianMs, profile.loadedMs);
                await yieldToBrowser();
                return { loaded: loaded.mraysPerSecond, single: set.count / (single.medianMs * 1e3) };
            };
            progress(`${sceneId}: quantized nodes against f32, ${profile.rounds} rounds of ${leafCases.length} cases, the order reversed every other round; node bytes ${Object.entries(nodeBytes).map(([name, bytes]) => `${name} ${(bytes / 2 ** 20).toFixed(1)} MiB`).join(', ')}`);
            for (let round = 0; round < profile.rounds; round++) {
                for (const step of leafLayoutRoundOrder(leafCases, nodeLayouts, round)) {
                    const entry = nodeEntryOf(step.case, step.layout);
                    const timed = await timeNode(step.case, step.layout);
                    entry.mraysPerRound.push(timed.loaded);
                    entry.singleMraysPerRound.push(timed.single);
                }
                progress(
                    `${sceneId}: node round ${round + 1} of ${profile.rounds}: ` +
                        leafCases.map((entry) => `${entry.tree} ${entry.set} ${entry.mode} quantized/f32 ${(nodeEntryOf(entry, 'quantized').mraysPerRound[round] / nodeEntryOf(entry, 'f32').mraysPerRound[round]).toFixed(3)}`).join(', '),
                );
            }
            const nodeRatios: PairedRatio[] = leafCases.map((entry) =>
                ratioOfRounds(`quantized_over_f32/${entry.tree}/${entry.set}/${entry.mode}`, nodeEntryOf(entry, 'quantized').mraysPerRound, nodeEntryOf(entry, 'f32').mraysPerRound),
            );
            for (const ratio of nodeRatios) {
                progress(`${sceneId}: ${ratio.name}: median ${ratio.median.toFixed(3)} (${ratio.min.toFixed(3)}-${ratio.max.toFixed(3)} over ${profile.rounds} rounds)`);
            }
            const nodeFormat: NodeFormatResult = { rounds: profile.rounds, entries: nodeEntries, ratios: nodeRatios, agreement: nodeAgreement, counts: nodeCounts, bytes: nodeBytes };

            // 8: indexed leaves on quantized nodes against quantized records, the pairing the `triangleLeaves` default waits on
            // (HANDOFF, Open): sections 5 and 6 time each lever alone against f32 records.
            const runCombined = !skip.has('combined');
            const combinedScenes: Record<LeafLayout, Record<Tree, TraceSceneBuffers>> = { records: nodeScenes.quantized, indexed: {} as Record<Tree, TraceSceneBuffers>, 'indexed-packed': {} as Record<Tree, TraceSceneBuffers> };
            const combinedBytes: LeafLayoutResult['bytes'] = {};
            const combinedKernels: Record<LeafLayout, TraceBatchKernel> = { records: nodeKernels.quantized, indexed: nodeKernels.quantized, 'indexed-packed': nodeKernels.quantized };
            if (runCombined) {
                for (const tree of ['instanced', 'merged'] as const) {
                    for (const [layout, positionLayout] of [['indexed', 'vec4'], ['indexed-packed', 'packed']] as const) {
                        const combined = await buildTraceScene(geometries, instances, { build, mergeStaticInstances: tree === 'merged', triangleLeaves: 'indexed', positionLayout, nodeFormat: 'quantized' });
                        combinedBytes[`quantized-${layout}/${tree}`] = { nodes: combined.nodes.byteLength, triangles: combined.triangles.byteLength, positions: combined.positions.byteLength, topLevel: combined.topLevel.buffer.byteLength };
                        const buffers = uploadTraceScene(device, combined, `surface quantized ${layout} ${tree}`);
                        sceneBuffers.push(buffers);
                        combinedScenes[layout][tree] = buffers;
                        await yieldToBrowser();
                    }
                }
                combinedKernels.indexed = await TraceBatchKernel.create(device, { label: 'surface quantized indexed', triangleLeaves: 'indexed', nodeFormat: 'quantized' });
                combinedKernels['indexed-packed'] = await TraceBatchKernel.create(device, { label: 'surface quantized indexed packed', triangleLeaves: 'indexed', positionLayout: 'packed', nodeFormat: 'quantized' });
                extraKernels.push(combinedKernels.indexed, combinedKernels['indexed-packed']);
            }
            const combinedEntries: LeafLayoutEntry[] = leafCases.flatMap((entry) =>
                leafLayouts.map((layout): LeafLayoutEntry => ({ tree: entry.tree, set: entry.set as SetName, mode: entry.mode, layout, mraysPerRound: [], singleMraysPerRound: [] })),
            );
            const combinedEntryOf = (entry: LeafLayoutCase, layout: LeafLayout): LeafLayoutEntry => {
                const found = combinedEntries.find((candidate) => candidate.tree === entry.tree && candidate.set === entry.set && candidate.mode === entry.mode && candidate.layout === layout);
                if (!found) throw new Error(`no combined entry ${entry.tree} ${entry.set} ${entry.mode} ${layout}`);
                return found;
            };
            // The same quantized trees in each triangle format: answers should be identical; a tie (same t, another triangle) is reported apart.
            const combinedAgreement: CombinedLeavesResult['agreement'] = [];
            for (const entry of runCombined ? leafCases : []) {
                const set = sets.find((candidate) => candidate.name === entry.set) as RaySet;
                const expected = new Uint32Array(await traceOnce(device, combinedKernels.records, combinedScenes.records[entry.tree], entry.mode, set.buffer, set.count));
                for (const layout of ['indexed', 'indexed-packed'] as const) {
                    const actual = new Uint32Array(await traceOnce(device, combinedKernels[layout], combinedScenes[layout][entry.tree], entry.mode, set.buffer, set.count));
                    const words = expected.length / set.count;
                    let same = 0;
                    let ties = 0;
                    for (let ray = 0; ray < set.count; ray++) {
                        let equal = true;
                        for (let word = 0; word < words && equal; word++) equal = expected[ray * words + word] === actual[ray * words + word];
                        if (equal) same++;
                        else if (entry.mode === 'closest' && expected[ray * words + HIT_RECORD_WORD.t] === actual[ray * words + HIT_RECORD_WORD.t]) ties++;
                    }
                    combinedAgreement.push({ tree: entry.tree, set: entry.set as SetName, mode: entry.mode, layout, rays: set.count, same, ties, different: set.count - same - ties });
                }
            }
            const timeCombined = async (entry: LeafLayoutCase, layout: LeafLayout): Promise<{ loaded: number; single: number }> => {
                const set = sets.find((candidate) => candidate.name === entry.set) as RaySet;
                const results = leafResults.get(`${set.name}/${entry.mode}`) as GPUBuffer;
                const encode = (encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites): void =>
                    combinedKernels[layout].encode(encoder, combinedScenes[layout][entry.tree], { mode: entry.mode, rays: set.buffer, results, count: set.count, timestampWrites });
                const single = await measureDispatches(() => timer.time((encoder, timestampWrites) => encode(encoder, timestampWrites)), profile.warmups, profile.timed);
                const loaded = await timeLoaded(device, encode, set.count, single.medianMs, profile.loadedMs);
                await yieldToBrowser();
                return { loaded: loaded.mraysPerSecond, single: set.count / (single.medianMs * 1e3) };
            };
            if (runCombined) progress(`${sceneId}: indexed leaves on quantized nodes against quantized records, ${profile.rounds} rounds of ${leafCases.length} cases in ${leafLayouts.length} formats, the order reversed every other round`);
            for (let round = 0; round < (runCombined ? profile.rounds : 0); round++) {
                for (const step of leafLayoutRoundOrder(leafCases, leafLayouts, round)) {
                    const entry = combinedEntryOf(step.case, step.layout);
                    const timed = await timeCombined(step.case, step.layout);
                    entry.mraysPerRound.push(timed.loaded);
                    entry.singleMraysPerRound.push(timed.single);
                }
                progress(
                    `${sceneId}: combined round ${round + 1} of ${profile.rounds}: ` +
                        leafCases.map((entry) => `${entry.tree} ${entry.set} ${entry.mode} q-indexed/q-records ${(combinedEntryOf(entry, 'indexed').mraysPerRound[round] / combinedEntryOf(entry, 'records').mraysPerRound[round]).toFixed(3)}`).join(', '),
                );
            }
            const combinedRatios: PairedRatio[] = (runCombined ? leafCases : []).flatMap((entry) => [
                ratioOfRounds(`quantized_indexed_over_quantized_records/${entry.tree}/${entry.set}/${entry.mode}`, combinedEntryOf(entry, 'indexed').mraysPerRound, combinedEntryOf(entry, 'records').mraysPerRound),
                ratioOfRounds(`quantized_indexed_packed_over_quantized_records/${entry.tree}/${entry.set}/${entry.mode}`, combinedEntryOf(entry, 'indexed-packed').mraysPerRound, combinedEntryOf(entry, 'records').mraysPerRound),
            ]);
            for (const ratio of combinedRatios) {
                progress(`${sceneId}: ${ratio.name}: median ${ratio.median.toFixed(3)} (${ratio.min.toFixed(3)}-${ratio.max.toFixed(3)} over ${profile.rounds} rounds)`);
            }
            const combinedLeaves: CombinedLeavesResult = { rounds: profile.rounds, entries: combinedEntries, ratios: combinedRatios, agreement: combinedAgreement, bytes: combinedBytes };

            // 9: the folded decode (quantizedDecode: 'folded') against the exact one, on section 6's quantized records scenes
            // (and section 8's quantized indexed ones when it ran): the same scenes and trees, only the kernel differs.
            const runFolded = !skip.has('folded');
            const foldedLayouts: LeafLayout[] = runCombined ? ['records', 'indexed'] : ['records'];
            const foldedKernels: Partial<Record<LeafLayout, TraceBatchKernel>> = {};
            if (runFolded) {
                for (const layout of foldedLayouts) {
                    const kernel = await TraceBatchKernel.create(device, { label: `surface folded ${layout}`, nodeFormat: 'quantized', quantizedDecode: 'folded', ...(layout === 'indexed' ? { triangleLeaves: 'indexed' as const } : {}) });
                    foldedKernels[layout] = kernel;
                    extraKernels.push(kernel);
                }
            }
            const foldedCases = runFolded ? foldedLayouts.flatMap((layout) => leafCases.map((entry) => ({ ...entry, layout }))) : [];
            const foldedEntries: FoldedDecodeResult['entries'] = foldedCases.flatMap((entry) =>
                (['exact', 'folded'] as const).map((decode) => ({ tree: entry.tree, set: entry.set as SetName, mode: entry.mode, layout: entry.layout, decode, mraysPerRound: [] as number[], singleMraysPerRound: [] as number[] })),
            );
            const foldedEntryOf = (entry: (typeof foldedCases)[number], decode: 'exact' | 'folded') => {
                const found = foldedEntries.find((candidate) => candidate.tree === entry.tree && candidate.set === entry.set && candidate.mode === entry.mode && candidate.layout === entry.layout && candidate.decode === decode);
                if (!found) throw new Error(`no folded entry ${entry.tree} ${entry.set} ${entry.mode} ${entry.layout} ${decode}`);
                return found;
            };
            const foldedKernelOf = (layout: LeafLayout, decode: 'exact' | 'folded'): TraceBatchKernel => (decode === 'folded' ? (foldedKernels[layout] as TraceBatchKernel) : combinedKernels[layout]);
            // Folded boxes are a few ulps wider, so answers must be the exact walk's, but for ties (same t, another triangle).
            const foldedAgreement: FoldedDecodeResult['agreement'] = [];
            for (const entry of foldedCases) {
                const set = sets.find((candidate) => candidate.name === entry.set) as RaySet;
                const expected = new Uint32Array(await traceOnce(device, foldedKernelOf(entry.layout, 'exact'), combinedScenes[entry.layout][entry.tree], entry.mode, set.buffer, set.count));
                const actual = new Uint32Array(await traceOnce(device, foldedKernelOf(entry.layout, 'folded'), combinedScenes[entry.layout][entry.tree], entry.mode, set.buffer, set.count));
                const words = expected.length / set.count;
                let same = 0;
                let ties = 0;
                for (let ray = 0; ray < set.count; ray++) {
                    let equal = true;
                    for (let word = 0; word < words && equal; word++) equal = expected[ray * words + word] === actual[ray * words + word];
                    if (equal) same++;
                    else if (entry.mode === 'closest' && expected[ray * words + HIT_RECORD_WORD.t] === actual[ray * words + HIT_RECORD_WORD.t]) ties++;
                }
                foldedAgreement.push({ tree: entry.tree, set: entry.set as SetName, mode: entry.mode, layout: entry.layout, rays: set.count, same, ties, different: set.count - same - ties });
                await yieldToBrowser();
            }
            const timeFolded = async (entry: (typeof foldedCases)[number], decode: 'exact' | 'folded'): Promise<{ loaded: number; single: number }> => {
                const set = sets.find((candidate) => candidate.name === entry.set) as RaySet;
                const results = leafResults.get(`${set.name}/${entry.mode}`) as GPUBuffer;
                const encode = (encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites): void =>
                    foldedKernelOf(entry.layout, decode).encode(encoder, combinedScenes[entry.layout][entry.tree], { mode: entry.mode, rays: set.buffer, results, count: set.count, timestampWrites });
                const single = await measureDispatches(() => timer.time((encoder, timestampWrites) => encode(encoder, timestampWrites)), profile.warmups, profile.timed);
                const loaded = await timeLoaded(device, encode, set.count, single.medianMs, profile.loadedMs);
                await yieldToBrowser();
                return { loaded: loaded.mraysPerSecond, single: set.count / (single.medianMs * 1e3) };
            };
            if (runFolded) progress(`${sceneId}: folded decode against exact on quantized nodes (${foldedLayouts.join(', ')}), ${profile.rounds} rounds of ${foldedCases.length} cases, the order reversed every other round`);
            for (let round = 0; round < (runFolded ? profile.rounds : 0); round++) {
                for (const step of leafLayoutRoundOrder(foldedCases, ['exact', 'folded'] as const, round)) {
                    const entry = foldedEntryOf(step.case as (typeof foldedCases)[number], step.layout);
                    const timed = await timeFolded(step.case as (typeof foldedCases)[number], step.layout);
                    entry.mraysPerRound.push(timed.loaded);
                    entry.singleMraysPerRound.push(timed.single);
                }
                progress(
                    `${sceneId}: folded round ${round + 1} of ${profile.rounds}: ` +
                        foldedCases.map((entry) => `${entry.layout} ${entry.tree} ${entry.set} ${entry.mode} folded/exact ${(foldedEntryOf(entry, 'folded').mraysPerRound[round] / foldedEntryOf(entry, 'exact').mraysPerRound[round]).toFixed(3)}`).join(', '),
                );
            }
            const foldedRatios: PairedRatio[] = foldedCases.map((entry) =>
                ratioOfRounds(`${entry.layout === 'records' ? 'folded_over_exact' : 'folded_indexed_over_exact_indexed'}/${entry.tree}/${entry.set}/${entry.mode}`, foldedEntryOf(entry, 'folded').mraysPerRound, foldedEntryOf(entry, 'exact').mraysPerRound),
            );
            for (const ratio of foldedRatios) {
                progress(`${sceneId}: ${ratio.name}: median ${ratio.median.toFixed(3)} (${ratio.min.toFixed(3)}-${ratio.max.toFixed(3)} over ${profile.rounds} rounds)`);
            }
            const foldedDecode: FoldedDecodeResult = { rounds: profile.rounds, entries: foldedEntries, ratios: foldedRatios, agreement: foldedAgreement };

            // 7: any-hit child order (TraceModuleOptions.anyHitOrder): 'larger' and 'near' against the default 'far', any-hit, records.
            const runOrders = !skip.has('any-order');
            const orders: AnyHitOrderName[] = ['far', 'larger', 'near'];
            const orderKernels: Record<AnyHitOrderName, TraceBatchKernel> = {
                far: fullKernel,
                larger: await TraceBatchKernel.create(device, { label: 'surface any larger', anyHitOrder: 'larger', nodeFormat: 'f32' }),
                near: await TraceBatchKernel.create(device, { label: 'surface any near', anyHitOrder: 'near', nodeFormat: 'f32' }),
            };
            extraKernels.push(orderKernels.larger, orderKernels.near);
            const orderCases: LeafLayoutCase[] = runOrders ? (['instanced', 'merged'] as const).flatMap((tree) => sets.map((set) => ({ tree, set: set.name, mode: 'any' as const }))) : [];
            const orderEntries: AnyHitOrderResult['entries'] = orderCases.flatMap((entry) => orders.map((order) => ({ tree: entry.tree, set: entry.set as SetName, order, mraysPerRound: [] as number[] })));
            const orderEntryOf = (entry: LeafLayoutCase, order: AnyHitOrderName) => {
                const found = orderEntries.find((candidate) => candidate.tree === entry.tree && candidate.set === entry.set && candidate.order === order);
                if (!found) throw new Error(`no any-hit order entry ${entry.tree} ${entry.set} ${order}`);
                return found;
            };
            const orderAgreement: AnyHitOrderResult['agreement'] = [];
            const orderCounts: AnyHitOrderResult['counts'] = [];
            const orderCounted: Record<AnyHitOrderName, CountedKernel> = { far: counted, larger: await CountedKernel.create(device, { anyHitOrder: 'larger' }), near: await CountedKernel.create(device, { anyHitOrder: 'near' }) };
            for (const entry of orderCases) {
                const set = sets.find((candidate) => candidate.name === entry.set) as RaySet;
                const expected = new Uint32Array(await traceOnce(device, fullKernel, scenes[entry.tree], 'any', set.buffer, set.count));
                for (const order of orders) {
                    const answers = await traceOnce(device, orderKernels[order], scenes[entry.tree], 'any', set.buffer, set.count);
                    const actual = new Uint32Array(answers);
                    let same = 0;
                    for (let ray = 0; ray < set.count; ray++) if (expected[ray] === actual[ray]) same++;
                    orderAgreement.push({ tree: entry.tree, set: entry.set as SetName, order, rays: set.count, same });
                    const counts = await orderCounted[order].count(scenes[entry.tree], 'any', set.buffer, set.count);
                    orderCounts.push({ tree: entry.tree, set: entry.set as SetName, order, counts: summariseCounts(counts, set.count), countedAgreement: countedAgreement('any', answers, counts, set.count) });
                    const last = orderCounts[orderCounts.length - 1];
                    progress(`${sceneId}: any-hit order ${order} ${entry.tree} ${entry.set}: steps ${last.counts.meanSteps.toFixed(2)}, triangle tests ${last.counts.meanTriangleTests.toFixed(2)}; same as far on ${same} of ${set.count}`);
                }
                await yieldToBrowser();
            }
            const timeOrder = async (entry: LeafLayoutCase, order: AnyHitOrderName): Promise<number> => {
                const set = sets.find((candidate) => candidate.name === entry.set) as RaySet;
                const results = leafResults.get(`${set.name}/any`) as GPUBuffer;
                const encode = (encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites): void =>
                    orderKernels[order].encode(encoder, scenes[entry.tree], { mode: 'any', rays: set.buffer, results, count: set.count, timestampWrites });
                const single = await measureDispatches(() => timer.time((encoder, timestampWrites) => encode(encoder, timestampWrites)), profile.warmups, profile.timed);
                const loaded = await timeLoaded(device, encode, set.count, single.medianMs, profile.loadedMs);
                await yieldToBrowser();
                return loaded.mraysPerSecond;
            };
            for (let round = 0; round < (runOrders ? profile.rounds : 0); round++) {
                for (const step of leafLayoutRoundOrder(orderCases, orders, round)) orderEntryOf(step.case, step.layout).mraysPerRound.push(await timeOrder(step.case, step.layout));
                progress(
                    `${sceneId}: any-hit order round ${round + 1} of ${profile.rounds}: ` +
                        orderCases.map((entry) => `${entry.tree} ${entry.set} larger/far ${(orderEntryOf(entry, 'larger').mraysPerRound[round] / orderEntryOf(entry, 'far').mraysPerRound[round]).toFixed(3)}`).join(', '),
                );
            }
            const orderRatios: PairedRatio[] = orderCases.flatMap((entry) =>
                (['larger', 'near'] as const).map((order) => ratioOfRounds(`any_${order}_over_far/${entry.tree}/${entry.set}/any`, orderEntryOf(entry, order).mraysPerRound, orderEntryOf(entry, 'far').mraysPerRound)),
            );
            for (const ratio of orderRatios) {
                progress(`${sceneId}: ${ratio.name}: median ${ratio.median.toFixed(3)} (${ratio.min.toFixed(3)}-${ratio.max.toFixed(3)} over ${profile.rounds} rounds)`);
            }
            const anyHitOrder: AnyHitOrderResult = { rounds: profile.rounds, entries: orderEntries, ratios: orderRatios, agreement: orderAgreement, counts: orderCounts };

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
                ...(runRecords ? { hitRecord } : {}),
                ...(runLeaves ? { triangleLeaves } : {}),
                nodeFormat,
                ...(runCombined ? { combinedLeaves } : {}),
                ...(runFolded ? { foldedDecode } : {}),
                ...(runOrders ? { anyHitOrder } : {}),
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
            for (const entry of leafAgreement) {
                if (entry.same !== entry.rays) failures.push(`${entry.layout} differs from records on ${entry.tree} ${entry.set} ${entry.mode}: ${entry.rays - entry.same} of ${entry.rays} rays`);
            }
            for (const entry of nodeAgreement) {
                if (entry.different > 0) failures.push(`quantized nodes change ${entry.different} of ${entry.rays} answers (not ties) on ${entry.tree} ${entry.set} ${entry.mode}`);
            }
            for (const entry of combinedAgreement) {
                if (entry.different > 0) failures.push(`quantized ${entry.layout} changes ${entry.different} of ${entry.rays} answers (not ties) against quantized records on ${entry.tree} ${entry.set} ${entry.mode}`);
            }
            for (const entry of foldedAgreement) {
                if (entry.different > 0) failures.push(`the folded decode changes ${entry.different} of ${entry.rays} answers (not ties) against the exact one on ${entry.layout} ${entry.tree} ${entry.set} ${entry.mode}`);
            }
            for (const entry of nodeCounts) {
                if (entry.countedAgreement < MIN_COUNTED_AGREEMENT) failures.push(`the quantized counted walk disagrees with its production kernel on ${entry.tree} ${entry.set} ${entry.mode}: ${percent(entry.countedAgreement)}`);
            }
            for (const entry of orderAgreement) {
                if (entry.same !== entry.rays) failures.push(`any-hit order '${entry.order}' blocks differently from 'far' on ${entry.tree} ${entry.set}: ${entry.rays - entry.same} of ${entry.rays} rays`);
            }
            for (const entry of orderCounts) {
                if (entry.countedAgreement < MIN_COUNTED_AGREEMENT) failures.push(`the '${entry.order}' counted walk disagrees with its production kernel on ${entry.tree} ${entry.set}: ${percent(entry.countedAgreement)}`);
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
            for (const extra of extraKernels) extra.destroy();
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

/**
 * Section 8 of the surface-ray pages: wide nodes against today's binary `f32`
 * nodes (docs/internals/design/wide-nodes.md). The same tinybvh trees, records,
 * assembled four ways: `f32` (the Aila-Laine 64-byte binary node), `bvh4`,
 * `cwbvh8` (private stack) and `cwbvh8` with the workgroup short stack (the
 * same scene, another kernel). Three ray sets: `primary` (one jittered camera
 * ray per pixel at 1280x720 from the manifest's first bookmark: coherent),
 * and the page's `surface` and `air-grid` sets (incoherent: cosine bounces
 * from screen probes, and uniform directions from a grid in the air).
 * Closest and any hit, on the instanced and the merged tree.
 *
 * Timed in paired rounds (`leafLayoutRoundOrder`: a case's formats back to
 * back, the order reversed every other round); the ratio is taken within each
 * round, `wide_<format>_over_f32/<tree>/<set>/<mode>` (above 1: the wide
 * format is faster). Hits: every answer must have the `f32` answer's `t`,
 * bit for bit, and the same record unless another triangle lies at exactly
 * that `t` (a tie: the walk order picks). Any-hit must block the same rays.
 * Also reported: node bytes per format and tree, and the time to collapse
 * every tree (CPU, this thread).
 */

import { buildTraceScene, HIT_RECORD_WORD, type MeshBvh, type MeshGeometry, type TraceSceneData, type TraceSceneInstance } from '../../../../src/trace/index.ts';
import { TraceBatchKernel, uploadTraceScene, type TraceSceneBuffers } from '../../../../src/trace/trace-gpu.ts';
import { convertToWide } from '../../../../src/wide-nodes.ts';
import { measureDispatches, type DispatchTimer } from './dispatch-timer.ts';
import { timeLoaded } from './loaded-timing.ts';
import { readBuffer, yieldToBrowser } from './ours-bench.ts';
import { leafLayoutRoundOrder, ratioOfRounds, type LeafLayoutCase, type PairedRatio, type Tree } from './paired-ratio.ts';

/** The formats timed: f32 is the baseline; cwbvh8-workgroup is cwbvh8's scene through the workgroup-stack kernel. */
export type WideLayout = 'f32' | 'bvh4' | 'bvh4-single' | 'cwbvh8' | 'cwbvh8-workgroup';
/** `bvh4-single` is bvh4's scene through the single-loop kernel (`wideWalk: 'single'`, lever L7). */
export const WIDE_LAYOUTS: readonly WideLayout[] = ['f32', 'bvh4', 'bvh4-single', 'cwbvh8', 'cwbvh8-workgroup'];

export interface WideRaySet {
    readonly name: string;
    readonly buffer: GPUBuffer;
    readonly count: number;
}

export interface WideNodesResult {
    rounds: number;
    entries: { tree: Tree; set: string; mode: 'closest' | 'any'; layout: WideLayout; mraysPerRound: number[]; singleMraysPerRound: number[] }[];
    ratios: PairedRatio[];
    /** Per case and format: rays answered bit for bit as f32, ties (same t, another triangle), different (must be 0). */
    agreement: { tree: Tree; set: string; mode: 'closest' | 'any'; layout: WideLayout; rays: number; same: number; ties: number; different: number }[];
    /** Bytes per format and tree: nodes, triangles, top level, and the scene. */
    bytes: Record<string, { nodes: number; triangles: number; topLevel: number; total: number }>;
    /** Milliseconds to collapse every tree of a tree kind into each wide format (this thread), and the tree counts. */
    collapse: Record<string, { ms: number; trees: number; binaryNodes: number; wideNodes: number; childrenPerNode: number; trianglesPerLeaf: number; maxStackNeed: number }>;
    /** Rays per set. */
    rays: Record<string, number>;
}

export interface WideNodesContext {
    readonly device: GPUDevice;
    readonly sceneId: string;
    readonly geometries: readonly MeshGeometry[];
    readonly instances: readonly TraceSceneInstance[];
    /** The page's cached tinybvh build (the same trees for every format). */
    readonly build: (geometry: MeshGeometry) => Promise<MeshBvh>;
    /** The page's f32 scenes and their data. */
    readonly f32Scenes: Record<Tree, TraceSceneBuffers>;
    readonly f32Data: Record<Tree, TraceSceneData>;
    /** The page's f32 kernel (40-byte records, masks on, any-hit far). */
    readonly f32Kernel: TraceBatchKernel;
    readonly sets: readonly WideRaySet[];
    readonly timer: DispatchTimer;
    readonly profile: { readonly rounds: number; readonly warmups: number; readonly timed: number; readonly loadedMs: number };
    readonly progress: (line: string) => void;
    /** Buffers and kernels the page frees at the end. */
    readonly sceneBuffers: TraceSceneBuffers[];
    readonly extraKernels: TraceBatchKernel[];
    readonly gpuBuffers: GPUBuffer[];
}

const MODES = ['closest', 'any'] as const;

async function traceOnce(device: GPUDevice, kernel: TraceBatchKernel, scene: TraceSceneBuffers, mode: 'closest' | 'any', rays: GPUBuffer, count: number): Promise<Uint32Array> {
    const bytes = count * (mode === 'closest' ? 40 : 4);
    const results = device.createBuffer({ label: `wide ${mode} answers`, size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    try {
        const encoder = device.createCommandEncoder({ label: `wide ${mode}` });
        kernel.encode(encoder, scene, { mode, rays, results, count });
        device.queue.submit([encoder.finish()]);
        return new Uint32Array(await readBuffer(device, results, bytes));
    } finally {
        results.destroy();
    }
}

/** Collapse time and shape: every tree a scene of this kind places, into each wide format. */
async function measureCollapse(context: WideNodesContext, tree: Tree): Promise<WideNodesResult['collapse']> {
    // The trees the f32 scene placed: the instanced meshes' (cached), or the merged scene's groups and the meshes left as instances.
    const trees: MeshBvh[] = [];
    if (tree === 'instanced') {
        const used = new Set(context.instances.map((instance) => instance.meshIndex));
        for (const index of used) trees.push(await context.build(context.geometries[index]));
    } else {
        // The merged scene's own trees are not kept by buildTraceScene: rebuild them through a capturing build.
        const captured: MeshBvh[] = [];
        await buildTraceScene(context.geometries, context.instances, {
            build: async (geometry) => {
                const built = await context.build(geometry);
                captured.push(built);
                return built;
            },
            mergeStaticInstances: true,
        });
        trees.push(...captured);
    }
    const out: WideNodesResult['collapse'] = {};
    for (const format of ['bvh4', 'cwbvh8'] as const) {
        const started = performance.now();
        let binaryNodes = 0;
        let wideNodes = 0;
        let slots = 0;
        let leafTriangles = 0;
        let leaves = 0;
        let maxStackNeed = 0;
        for (const mesh of trees) {
            const wide = convertToWide(mesh, format);
            binaryNodes += mesh.nodeCount;
            wideNodes += wide.nodeCount;
            slots += wide.childrenPerNode * wide.nodeCount;
            const leafSlots = wide.trianglesPerLeaf > 0 ? mesh.triangleCount / wide.trianglesPerLeaf : 0;
            leaves += leafSlots;
            leafTriangles += mesh.triangleCount;
            maxStackNeed = Math.max(maxStackNeed, wide.stackNeed);
        }
        out[`${format}/${tree}`] = {
            ms: performance.now() - started,
            trees: trees.length,
            binaryNodes,
            wideNodes,
            childrenPerNode: slots / Math.max(1, wideNodes),
            trianglesPerLeaf: leafTriangles / Math.max(1, leaves),
            maxStackNeed,
        };
        await yieldToBrowser();
    }
    return out;
}

export async function runWideNodesSection(context: WideNodesContext): Promise<WideNodesResult> {
    const { device, sceneId, progress, profile } = context;
    const scenes: Record<WideLayout, Record<Tree, TraceSceneBuffers>> = {
        f32: context.f32Scenes,
        bvh4: {} as Record<Tree, TraceSceneBuffers>,
        'bvh4-single': {} as Record<Tree, TraceSceneBuffers>,
        cwbvh8: {} as Record<Tree, TraceSceneBuffers>,
        'cwbvh8-workgroup': {} as Record<Tree, TraceSceneBuffers>,
    };
    const bytes: WideNodesResult['bytes'] = {};
    const sizeOf = (data: TraceSceneData) => ({ nodes: data.nodes.byteLength, triangles: data.triangles.byteLength, topLevel: data.topLevel.buffer.byteLength, total: data.byteLength });
    const collapse: WideNodesResult['collapse'] = {};
    for (const tree of ['instanced', 'merged'] as const) {
        bytes[`f32/${tree}`] = sizeOf(context.f32Data[tree]);
        Object.assign(collapse, await measureCollapse(context, tree));
        for (const format of ['bvh4', 'cwbvh8'] as const) {
            const data = await buildTraceScene(context.geometries, context.instances, { build: context.build, mergeStaticInstances: tree === 'merged', nodeFormat: format });
            bytes[`${format}/${tree}`] = sizeOf(data);
            const buffers = uploadTraceScene(device, data, `wide ${format} ${tree}`);
            context.sceneBuffers.push(buffers);
            scenes[format][tree] = buffers;
            if (format === 'cwbvh8') scenes['cwbvh8-workgroup'][tree] = buffers;
            if (format === 'bvh4') scenes['bvh4-single'][tree] = buffers;
            await yieldToBrowser();
        }
    }
    for (const [name, entry] of Object.entries(collapse)) {
        progress(
            `${sceneId}: collapse ${name}: ${entry.trees} trees, ${entry.ms.toFixed(0)} ms, ${entry.binaryNodes} binary nodes -> ${entry.wideNodes} wide, ` +
                `${entry.childrenPerNode.toFixed(2)} children per node, ${entry.trianglesPerLeaf.toFixed(2)} triangles per leaf, stack need ${entry.maxStackNeed}`,
        );
    }
    progress(`${sceneId}: wide node bytes ${Object.entries(bytes).map(([name, size]) => `${name} ${(size.nodes / 2 ** 20).toFixed(1)} MiB`).join(', ')}`);
    const kernels: Record<WideLayout, TraceBatchKernel> = {
        f32: context.f32Kernel,
        bvh4: await TraceBatchKernel.create(device, { label: 'wide bvh4', nodeFormat: 'bvh4' }),
        'bvh4-single': await TraceBatchKernel.create(device, { label: 'wide bvh4 single loop', nodeFormat: 'bvh4', wideWalk: 'single' }),
        cwbvh8: await TraceBatchKernel.create(device, { label: 'wide cwbvh8', nodeFormat: 'cwbvh8' }),
        'cwbvh8-workgroup': await TraceBatchKernel.create(device, { label: 'wide cwbvh8 workgroup', nodeFormat: 'cwbvh8', wideStack: 'workgroup' }),
    };
    context.extraKernels.push(kernels.bvh4, kernels['bvh4-single'], kernels.cwbvh8, kernels['cwbvh8-workgroup']);

    const cases: LeafLayoutCase[] = (['instanced', 'merged'] as const).flatMap((tree) => context.sets.flatMap((set) => MODES.map((mode) => ({ tree, set: set.name, mode }))));
    const setOf = (name: string): WideRaySet => context.sets.find((set) => set.name === name)!;

    // Hits first: every format against f32.
    const agreement: WideNodesResult['agreement'] = [];
    for (const entry of cases) {
        const set = setOf(entry.set);
        const expected = await traceOnce(device, kernels.f32, scenes.f32[entry.tree], entry.mode, set.buffer, set.count);
        const words = expected.length / set.count;
        for (const layout of WIDE_LAYOUTS.slice(1)) {
            const actual = await traceOnce(device, kernels[layout], scenes[layout][entry.tree], entry.mode, set.buffer, set.count);
            let same = 0;
            let ties = 0;
            for (let ray = 0; ray < set.count; ray++) {
                let equal = true;
                for (let word = 0; word < words && equal; word++) equal = expected[ray * words + word] === actual[ray * words + word];
                if (equal) same++;
                else if (entry.mode === 'closest' && expected[ray * words + HIT_RECORD_WORD.t] === actual[ray * words + HIT_RECORD_WORD.t]) ties++;
            }
            agreement.push({ tree: entry.tree, set: entry.set, mode: entry.mode, layout, rays: set.count, same, ties, different: set.count - same - ties });
            progress(`${sceneId}: ${layout} ${entry.tree} ${entry.set} ${entry.mode}: same ${same}, ties ${ties}, different ${set.count - same - ties} of ${set.count}`);
        }
        await yieldToBrowser();
    }

    // Paired timing.
    const entries: WideNodesResult['entries'] = cases.flatMap((entry) => WIDE_LAYOUTS.map((layout) => ({ tree: entry.tree, set: entry.set, mode: entry.mode, layout, mraysPerRound: [] as number[], singleMraysPerRound: [] as number[] })));
    const entryOf = (entry: LeafLayoutCase, layout: WideLayout) => entries.find((candidate) => candidate.tree === entry.tree && candidate.set === entry.set && candidate.mode === entry.mode && candidate.layout === layout)!;
    const results = new Map<string, GPUBuffer>();
    for (const set of context.sets) {
        for (const mode of MODES) {
            const buffer = device.createBuffer({ label: `wide ${set.name} ${mode} results`, size: set.count * (mode === 'closest' ? 40 : 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
            context.gpuBuffers.push(buffer);
            results.set(`${set.name}/${mode}`, buffer);
        }
    }
    const timeOne = async (entry: LeafLayoutCase, layout: WideLayout): Promise<{ loaded: number; single: number }> => {
        const set = setOf(entry.set);
        const out = results.get(`${set.name}/${entry.mode}`)!;
        const encode = (encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites): void =>
            kernels[layout].encode(encoder, scenes[layout][entry.tree], { mode: entry.mode, rays: set.buffer, results: out, count: set.count, timestampWrites });
        const single = await measureDispatches(() => context.timer.time((encoder, timestampWrites) => encode(encoder, timestampWrites)), profile.warmups, profile.timed);
        const loaded = await timeLoaded(device, encode, set.count, single.medianMs, profile.loadedMs);
        await yieldToBrowser();
        return { loaded: loaded.mraysPerSecond, single: set.count / (single.medianMs * 1e3) };
    };
    progress(`${sceneId}: wide nodes against f32, ${profile.rounds} rounds of ${cases.length} cases x ${WIDE_LAYOUTS.length} formats, the order reversed every other round`);
    for (let round = 0; round < profile.rounds; round++) {
        for (const step of leafLayoutRoundOrder(cases, WIDE_LAYOUTS, round)) {
            const timed = await timeOne(step.case, step.layout);
            entryOf(step.case, step.layout).mraysPerRound.push(timed.loaded);
            entryOf(step.case, step.layout).singleMraysPerRound.push(timed.single);
        }
        progress(
            `${sceneId}: wide round ${round + 1} of ${profile.rounds}: ` +
                cases
                    .filter((entry) => entry.mode === 'closest')
                    .map((entry) => `${entry.tree} ${entry.set} closest bvh4 ${(entryOf(entry, 'bvh4').mraysPerRound[round] / entryOf(entry, 'f32').mraysPerRound[round]).toFixed(3)} cwbvh8 ${(entryOf(entry, 'cwbvh8').mraysPerRound[round] / entryOf(entry, 'f32').mraysPerRound[round]).toFixed(3)}`)
                    .join(', '),
        );
    }
    const ratios: PairedRatio[] = cases.flatMap((entry) =>
        WIDE_LAYOUTS.slice(1).map((layout) => ratioOfRounds(`wide_${layout}_over_f32/${entry.tree}/${entry.set}/${entry.mode}`, entryOf(entry, layout).mraysPerRound, entryOf(entry, 'f32').mraysPerRound)),
    );
    for (const ratio of ratios) progress(`${sceneId}: ${ratio.name}: median ${ratio.median.toFixed(3)} (${ratio.min.toFixed(3)}-${ratio.max.toFixed(3)} over ${profile.rounds} rounds)`);
    for (const entry of cases) {
        const middle = (values: number[]): number => [...values].sort((a, b) => a - b)[values.length >> 1];
        progress(`${sceneId}: ${entry.tree} ${entry.set} ${entry.mode}: ${WIDE_LAYOUTS.map((layout) => `${layout} ${middle(entryOf(entry, layout).mraysPerRound).toFixed(1)}`).join(', ')} Mrays/s (median of rounds, loaded)`);
    }
    return { rounds: profile.rounds, entries, ratios, agreement, bytes, collapse, rays: Object.fromEntries(context.sets.map((set) => [set.name, set.count])) };
}

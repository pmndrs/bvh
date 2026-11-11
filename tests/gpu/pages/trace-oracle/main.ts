/**
 * Trace oracle: runs every oracle case through the real WGSL tracer on this
 * GPU and compares with the CPU reference (and with answers worked out by hand).
 *
 * Cases: thin walls, mirrored and non-uniform transforms, instancing, a closed
 * thin box no interior ray may escape, random scenes, and about two million
 * rays aimed at the seams of a tessellated quad (they must not slip through).
 * Everything runs once per variant (see VARIANTS), always with the watertight
 * test of Woop et al. (the tracer's only triangle test). The ddgi platform's own
 * variants (`platform*`: scene API, first GPU trees, mid-swap) stayed in webgpu-ddgi:
 * - `watertight-tinybvh`: the same trees assembled and traced directly;
 * - `watertight-tmbuild`: trees built by three-mesh-bvh and converted;
 * - `watertight-lbvh`: trees built on the GPU by the LBVH builder (B1,
 *   `createMeshLayoutBuilder`), read back and assembled the
 *   same way. It runs only when the device has subgroups; without them the
 *   builder would fall back to tinybvh, which the variants above already cover;
 * - `watertight-hploc`: the same with the GPU H-PLOC builder (B2);
 * - `watertight-tinybvh-nomasks`: tinybvh trees through a kernel built with
 *   `rayMasks: false`, the walk with its mask code compiled out;
 * - `merged-tinybvh`, `merged-hploc`: `mergeStaticInstances`. Each instance
 *   gets its own copy of its mesh and two of every three are marked static,
 *   so they merge into world-space meshes (one per flags-and-mask group)
 *   built by tinybvh or the GPU H-PLOC; the rest stay instances. The
 *   reference is the case's own instanced scene, so hits must name the source
 *   instance, its triangle and its authored side. The hook cases also run
 *   merged (`[hooks-merged]`), with the filter flags in the merged geometry.
 * The variants above trace through the masked batch kernel (the default) with
 * a mask of all ones. Then the hook cases (hooks.ts) run on tinybvh trees, with
 * ray masks on and off: a filter on flagged triangles, a filter that depends on
 * the query (a surface that casts no shadow), and instance masks against
 * several ray masks with a filter that reads the ray mask. Each traced mask is
 * checked against the reference given the same filter and mask.
 * The default device limits are used on purpose: the oracle scenes are tiny and
 * the core has to work there.
 *
 * URL option: `?sharedRays=100000` shrinks the shared-edge scenes for a quick look.
 */

import {
    assembleTraceScene,
    mergeStaticGeometry,
    TRIANGLE_FLAG_FILTER,
    type MeshBvh,
    type MeshGeometry,
    type TraceSceneInstance,
    createCpuTracer,
    createMeshBvhBuilder,
    HIT_RECORD_BYTES,
    newTraceStats,
    TRACE_MODULE_ENTRIES,
    type MeshBvhBuilder,
    type TraceSceneData,
} from '../../../../src/trace/index.ts';
import { type TracedHit } from '../../../../src/trace/oracle-compare.ts';
import { createMeshLayoutBuilder, createTinyBvhBuilder, type MeshLayoutBuilder } from '../../../../src/index.ts';
import { buildMeshBvhWithTinyBvh } from '../../../../src/trace/mesh-bvh-topology.ts';
import { RAY_BYTES, TraceBatchKernel, uploadTraceScene } from '../../trace-gpu.ts';
import { assert, runGpuTest } from '../../client.ts';
import {
    buildOracleCases,
    FULL_SIZES,
    sharedEdgeCase,
    unpackRay,
    type OracleCase,
    type SharedEdgeCase,
} from './cases.ts';
import { buildMeshBvhWithTmbvh } from '../trace-bench/tmbvh-convert.ts';
import { buildHookCases, hookMasks, hookPassName, hookReferenceAnswers, writeHookFlags } from './hooks.ts';
import {
    computeReferenceAnswersYielding,
    evaluateCase,
    evaluateSharedEdgeCase,
    sharedEdgeFailures,
    type CaseResult,
    type SharedEdgeReport,
    type TracerAnswers,
} from './check.ts';

/**
 * What each pass of the oracle traces: who built the per-mesh trees. `three-mesh-bvh`
 * trees are built by that library and converted to our layout
 * (../trace-bench/tmbvh-convert.ts), the T0.10 part C experiment; they are traced by
 * the same WGSL walk.
 */
interface OracleVariant {
    name: string;
    treeSource: 'tinybvh' | 'three-mesh-bvh' | 'gpu-lbvh' | 'gpu-hploc';
    /** The batch kernel's `rayMasks`; default true. */
    rayMasks?: boolean;
    /** Merge static instances (see above). */
    merged?: boolean;
}

const VARIANTS: OracleVariant[] = [
    { name: 'watertight-tinybvh', treeSource: 'tinybvh' },
    { name: 'watertight-tmbuild', treeSource: 'three-mesh-bvh' },
    { name: 'watertight-lbvh', treeSource: 'gpu-lbvh' },
    { name: 'watertight-hploc', treeSource: 'gpu-hploc' },
    { name: 'watertight-tinybvh-nomasks', treeSource: 'tinybvh', rayMasks: false },
    { name: 'merged-tinybvh', treeSource: 'tinybvh', merged: true },
    { name: 'merged-hploc', treeSource: 'gpu-hploc', merged: true },
];

/** Rays per hook case and traced mask (hooks.ts); the reference runs once per mask. */
const HOOK_SIZES = { randomRays: 8000, thinBoxRays: 1200 };

/** The GPU builders (B1 LBVH, B2 H-PLOC), made once the device is known. */
const gpuBuilders = new Map<'gpu-lbvh' | 'gpu-hploc', MeshLayoutBuilder>();

/** three-mesh-bvh target leaf size for the converted trees, as in the benchmark's default. */
const TMBUILD_LEAF_SIZE = 4;

/** Rays per shared-edge scene. Two scenes, so 2 million in total for each triangle test. */
const DEFAULT_SHARED_EDGE_RAYS = 1_000_000;

/** Rays the CPU port of the walk traces per case, to report the stack depth it needs. */
const STACK_PROBE_RAYS = 2000;

/** Rays of each shared-edge scene also checked against the brute-force reference. */
const SHARED_EDGE_REFERENCE_RAYS = 20000;

const HIT_RECORD_WORDS = HIT_RECORD_BYTES / 4;

interface GpuAnswers {
    closest: Float32Array;
    closestWords: Uint32Array;
    blocked: Uint32Array;
}

async function readBuffer(device: GPUDevice, source: GPUBuffer, byteLength: number): Promise<ArrayBuffer> {
    const readback = device.createBuffer({
        label: 'oracle readback',
        size: byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = device.createCommandEncoder({ label: 'oracle readback' });
    encoder.copyBufferToBuffer(source, 0, readback, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();
    return copy;
}

/** Records one batch of rays into the encoder. */
type EncodeBatch = (encoder: GPUCommandEncoder, batch: { mode: 'closest' | 'any'; rays: GPUBuffer; results: GPUBuffer; count: number }) => void;

/** Traces the packed rays on the GPU in both modes and reads the results back. */
async function traceOnGpu(device: GPUDevice, encodeBatch: EncodeBatch, packedRays: Float32Array): Promise<GpuAnswers> {
    const count = packedRays.length / 8;
    const rays = device.createBuffer({ label: 'oracle rays', size: count * RAY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(rays, 0, packedRays as Float32Array<ArrayBuffer>);
    const hitBytes = count * HIT_RECORD_BYTES;
    const anyBytes = count * 4;
    const hits = device.createBuffer({ label: 'oracle hits', size: hitBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const occlusion = device.createBuffer({ label: 'oracle occlusion', size: anyBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

    const encoder = device.createCommandEncoder({ label: 'oracle trace' });
    encodeBatch(encoder, { mode: 'closest', rays, results: hits, count });
    encodeBatch(encoder, { mode: 'any', rays, results: occlusion, count });
    device.queue.submit([encoder.finish()]);

    const hitData = await readBuffer(device, hits, hitBytes);
    const anyData = await readBuffer(device, occlusion, anyBytes);
    rays.destroy();
    hits.destroy();
    occlusion.destroy();
    return {
        closest: new Float32Array(hitData),
        closestWords: new Uint32Array(hitData),
        blocked: new Uint32Array(anyData),
    };
}


/** One ray's closest-hit record as the comparison code wants it. */
function hitAt(answers: GpuAnswers, index: number): TracedHit | null {
    const base = index * HIT_RECORD_WORDS;
    const t = answers.closest[base];
    if (!(t >= 0)) return null;
    return {
        hit: true,
        t,
        instanceIndex: answers.closestWords[base + 1],
        triangleIndex: answers.closestWords[base + 2],
        frontFace: answers.closestWords[base + 3] !== 0,
        u: answers.closest[base + 4],
        v: answers.closest[base + 5],
    };
}

/** The read-back results in the form the checks (check.ts) take. */
function tracerAnswers(answers: GpuAnswers): TracerAnswers {
    return {
        closest: (index) => hitAt(answers, index),
        any: (index) => answers.blocked[index] !== 0,
    };
}

interface BuiltScene {
    data: TraceSceneData;
    buildMs: number;
}

async function buildTrees(builder: MeshBvhBuilder, meshes: readonly MeshGeometry[], variant: OracleVariant): Promise<MeshBvh[]> {
    if (variant.treeSource === 'three-mesh-bvh') return meshes.map((mesh) => buildMeshBvhWithTmbvh(mesh, TMBUILD_LEAF_SIZE).bvh);
    if (variant.treeSource === 'gpu-lbvh' || variant.treeSource === 'gpu-hploc') {
        const gpuBuilder = gpuBuilders.get(variant.treeSource);
        assert(gpuBuilder, `the ${variant.treeSource} builder was not made`);
        const results = await gpuBuilder.buildLayouts(meshes);
        for (const result of results) assert(result.builtBy === variant.treeSource, `a mesh fell back to ${result.builtBy}: ${result.fallbackReason}`);
        return results.map((result) => result.layout);
    }
    return Promise.all(meshes.map((mesh) => builder.build(mesh)));
}

/**
 * The case's scene for merging: each instance gets its own copy of its mesh (the same
 * geometry, so the reference of the instanced case still holds), and two of every
 * three are static. `flagged` adds the hooks' filter flags to every copy.
 */
function unsharedForMerge(meshes: readonly MeshGeometry[], instances: readonly TraceSceneInstance[], flagged?: (original: number) => boolean) {
    const copies = instances.map((instance) => {
        const mesh = meshes[instance.meshIndex];
        if (!flagged) return mesh;
        const count = mesh.indices ? mesh.indices.length / 3 : mesh.positions.length / 9;
        return { ...mesh, triangleFlags: Uint32Array.from({ length: count }, (_, t) => (flagged(t) ? TRIANGLE_FLAG_FILTER : 0)) };
    });
    return { meshes: copies, instances: instances.map((instance, position) => ({ ...instance, meshIndex: position, static: position % 3 !== 2 })) };
}

/** Builds a merged scene: the meshes normal instances still use, and one tree per merged group, all by the variant's builder. */
async function buildMergedScene(builder: MeshBvhBuilder, meshes: readonly MeshGeometry[], instances: readonly TraceSceneInstance[], variant: OracleVariant): Promise<TraceSceneData> {
    const plan = mergeStaticGeometry(meshes, instances);
    const needed = meshes.flatMap((mesh, index) => (plan.meshNeeded[index] ? [index] : []));
    const trees = await buildTrees(builder, [...needed.map((index) => meshes[index]), ...plan.groups.map((group) => group.geometry)], variant);
    const sourceTrees: (MeshBvh | null)[] = meshes.map(() => null);
    needed.forEach((index, at) => (sourceTrees[index] = trees[at]));
    return assembleTraceScene(sourceTrees, instances, {}, { merged: { plan, meshes: trees.slice(needed.length) } });
}

async function buildScene(builder: MeshBvhBuilder, oracleCase: OracleCase, variant: OracleVariant): Promise<BuiltScene> {
    const start = performance.now();
    if (variant.merged) {
        const unshared = unsharedForMerge(oracleCase.meshes, oracleCase.instances);
        const data = await buildMergedScene(builder, unshared.meshes, unshared.instances, variant);
        return { data, buildMs: performance.now() - start };
    }
    const data = assembleTraceScene(await buildTrees(builder, oracleCase.meshes, variant), oracleCase.instances);
    return { data, buildMs: performance.now() - start };
}

/** Deepest stack the CPU port of the walk used over the first rays of a case. */
function probeStackDepth(scene: TraceSceneData, oracleCase: OracleCase): number {
    const tracer = createCpuTracer(scene);
    const stats = newTraceStats();
    const rayCount = Math.min(STACK_PROBE_RAYS, oracleCase.rays.length / 8);
    for (let index = 0; index < rayCount; index++) {
        const ray = unpackRay(oracleCase.rays, index);
        tracer.closest(ray, { stats });
        tracer.any(ray, { stats });
    }
    return stats.maxStackSize;
}

/** Traces a case in a variant and returns the answers plus the trees (for the stack and size report). */
async function traceCase(
    device: GPUDevice,
    kernel: TraceBatchKernel,
    builder: MeshBvhBuilder,
    oracleCase: OracleCase,
    variant: OracleVariant,
): Promise<{ answers: GpuAnswers; built: BuiltScene }> {
    const built = await buildScene(builder, oracleCase, variant);
    const buffers = uploadTraceScene(device, built.data, oracleCase.name);
    try {
        return { answers: await traceOnGpu(device, (encoder, batch) => kernel.encode(encoder, buffers, batch), oracleCase.rays), built };
    } finally {
        buffers.destroy();
    }
}

async function runSharedEdgeCase(
    device: GPUDevice,
    kernel: TraceBatchKernel,
    builder: MeshBvhBuilder,
    oracleCase: SharedEdgeCase,
    variant: OracleVariant,
): Promise<SharedEdgeReport> {
    const { answers } = await traceCase(device, kernel, builder, oracleCase, variant);
    return evaluateSharedEdgeCase(oracleCase, tracerAnswers(answers), SHARED_EDGE_REFERENCE_RAYS);
}

function summarize(result: CaseResult, stackUsed: number, scene: TraceSceneData, buildMs: number) {
    return {
        rays: result.rays,
        closest: {
            matches: result.closest.matches,
            ambiguous: result.closest.ambiguous,
            mismatches: result.closest.mismatches,
            maxTError: Number(result.closest.maxTError.toExponential(2)),
        },
        any: { matches: result.any.matches, ambiguous: result.any.ambiguous, mismatches: result.any.mismatches },
        closestEscapes: result.closestEscapes,
        anyEscapes: result.anyEscapes,
        referenceEscapes: result.referenceEscapes,
        expectationFailures: result.expectationFailures.length,
        topLevelDepth: scene.topLevel.maxDepth,
        maxMeshDepth: scene.maxMeshDepth,
        requiredStack: scene.requiredStackSize,
        stackUsedByCpuPort: stackUsed,
        sceneBytes: scene.byteLength,
        buildMs: Math.round(buildMs),
    };
}

const sharedRays = Number(new URLSearchParams(location.search).get('sharedRays')) || DEFAULT_SHARED_EDGE_RAYS;

await runGpuTest(async ({ device, log }) => {
    const builder = createMeshBvhBuilder({ worker: 'auto' });
    assert(builder.usesWorker, 'the BVH builder did not start its worker');

    // The worker must give the same tree as building on this thread.
    {
        const sample = buildOracleCases(FULL_SIZES).find((entry) => entry.name === 'random-soup');
        assert(sample !== undefined, 'random-soup case missing');
        const fromWorker = await builder.build(sample.meshes[0]);
        const inline = buildMeshBvhWithTinyBvh(sample.meshes[0], await createTinyBvhBuilder());
        const inlineNodes = new Uint8Array(inline.nodes);
        assert(
            fromWorker.nodeCount === inline.nodeCount &&
                fromWorker.maxDepth === inline.maxDepth &&
                new Uint8Array(fromWorker.nodes).every((byte, position) => byte === inlineNodes[position]),
            'the worker built a different tree than the main thread',
        );
        log(`worker build matches the inline build (${inline.nodeCount} nodes, depth ${inline.maxDepth})`);
    }

    const failures: string[] = [];
    const byTest: Record<string, { cases: Record<string, unknown>; sharedEdges: SharedEdgeReport[] }> = {};
    let deepestStackUsed = 0;
    let deepestRequired = 0;

    // The brute-force answers are the slow part of the page, so compute them once per case.
    const cases = buildOracleCases(FULL_SIZES);
    const references = [];
    for (const oracleCase of cases) references.push(await computeReferenceAnswersYielding(oracleCase));

    for (const kind of ['gpu-lbvh', 'gpu-hploc'] as const) gpuBuilders.set(kind, await createMeshLayoutBuilder(device, { prefer: kind, label: `oracle ${kind}` }));
    const skipped: string[] = [];
    for (const variant of VARIANTS) {
        const gpuBuilder = variant.treeSource === 'gpu-lbvh' || variant.treeSource === 'gpu-hploc' ? gpuBuilders.get(variant.treeSource) : undefined;
        if (gpuBuilder && gpuBuilder.kind === 'tinybvh') {
            skipped.push(`${variant.name}: ${gpuBuilder.fallbackReason}`);
            log(`[${variant.name}] skipped: ${gpuBuilder.fallbackReason}`);
            continue;
        }
        const kernel = await TraceBatchKernel.create(device, { label: variant.name, rayMasks: variant.rayMasks ?? true });
        const results = { cases: {} as Record<string, unknown>, sharedEdges: [] as SharedEdgeReport[] };
        byTest[variant.name] = results;

        for (const [caseIndex, oracleCase] of cases.entries()) {
            const { answers, built } = await traceCase(device, kernel, builder, oracleCase, variant);

            const result = evaluateCase(oracleCase, tracerAnswers(answers), references[caseIndex]);
            const stackUsed = probeStackDepth(built.data, oracleCase);
            deepestStackUsed = Math.max(deepestStackUsed, stackUsed);
            deepestRequired = Math.max(deepestRequired, built.data.requiredStackSize);
            results.cases[oracleCase.name] = summarize(result, stackUsed, built.data, built.buildMs);
            log(
                `[${variant.name}] ${oracleCase.name}: ${result.rays} rays, closest ${result.closest.mismatches} mismatches / ${result.closest.ambiguous} ambiguous, ` +
                    `any ${result.any.mismatches} / ${result.any.ambiguous}, ` +
                    `escapes closest ${result.closestEscapes.ordinary} + ${result.closestEscapes.edgeAimed} at corners, ` +
                    `any ${result.anyEscapes.ordinary} + ${result.anyEscapes.edgeAimed} at corners`,
            );
            for (const failure of result.failures) failures.push(`[${variant.name}] ${oracleCase.name}: ${failure}`);
        }

        for (const transformed of [false, true]) {
            const report = await runSharedEdgeCase(device, kernel, builder, sharedEdgeCase(sharedRays, transformed), variant);
            results.sharedEdges.push(report);
            log(
                `[${variant.name}] ${report.name}: ${report.slipped} of ${report.rays} rays slipped between triangles ` +
                    `(${report.slippedAny} any-hit)`,
            );
            for (const failure of sharedEdgeFailures(report)) failures.push(`[${variant.name}] ${report.name}: ${failure}`);
        }
        kernel.destroy();
    }

    // The hooks: triangle filters and instance masks, with ray masks on and off.
    const hooks: Record<string, unknown> = {};
    for (const [rayMasks, merged] of [[true, false], [false, false], [true, true]] as const) {
        for (const hook of buildHookCases(HOOK_SIZES)) {
            const kernel = await TraceBatchKernel.create(device, { label: hook.name, triangleFilter: hook.filterWgsl, rayMasks });
            let scene: TraceSceneData;
            let flagged: number;
            if (merged) {
                const unshared = unsharedForMerge(hook.base.meshes, hook.base.instances, hook.flagged);
                scene = await buildMergedScene(builder, unshared.meshes, unshared.instances, VARIANTS[0]);
                flagged = new Uint32Array(scene.triangles).filter((word, at) => at % 12 === 7 && (word & TRIANGLE_FLAG_FILTER) !== 0).length;
            } else {
                scene = assembleTraceScene(await Promise.all(hook.base.meshes.map((mesh) => builder.build(mesh))), hook.base.instances);
                flagged = writeHookFlags(scene, hook);
            }
            const buffers = uploadTraceScene(device, scene, hook.name);
            try {
                for (const mask of hookMasks(hook, rayMasks)) {
                    const answers = await traceOnGpu(device, (encoder, batch) => kernel.encode(encoder, buffers, { ...batch, mask }), hook.base.rays);
                    const result = evaluateCase(hook.base, tracerAnswers(answers), await hookReferenceAnswers(hook, mask, rayMasks));
                    const name = `${merged ? 'merged ' : ''}${hookPassName(hook, mask, rayMasks)}`;
                    hooks[name] = {
                        rays: result.rays,
                        flaggedTriangles: flagged,
                        closest: { matches: result.closest.matches, ambiguous: result.closest.ambiguous, mismatches: result.closest.mismatches },
                        any: { matches: result.any.matches, ambiguous: result.any.ambiguous, mismatches: result.any.mismatches },
                    };
                    log(
                        `[hooks${merged ? '-merged' : ''}] ${name}: ${result.rays} rays, closest ${result.closest.mismatches} mismatches / ${result.closest.ambiguous} ambiguous, ` +
                            `any ${result.any.mismatches} / ${result.any.ambiguous}`,
                    );
                    for (const failure of result.failures) failures.push(`[hooks] ${name}: ${failure}`);
                }
            } finally {
                buffers.destroy();
                kernel.destroy();
            }
        }
    }
    builder.dispose();

    const limits = device.limits;
    assert(failures.length === 0, failures.slice(0, 12).join('\n'));
    const slippedTotals: Record<string, number> = {};
    for (const variant of VARIANTS) {
        if (!byTest[variant.name]) continue;
        slippedTotals[variant.name] = byTest[variant.name].sharedEdges.reduce((sum, report) => sum + report.slipped, 0);
    }
    return {
        skipped,
        sharedEdgeRaysPerTest: byTest[VARIANTS[0].name].sharedEdges.reduce((sum, report) => sum + report.rays, 0),
        slippedTotals,
        byTest,
        hooks,
        traceStorageBuffers: TRACE_MODULE_ENTRIES.length,
        batchKernelStorageBuffers: TRACE_MODULE_ENTRIES.length + 2,
        maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage,
        maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
        deepestStackUsedByCpuPort: deepestStackUsed,
        deepestStackRequired: deepestRequired,
    };
}, {
    // The GPU builders need subgroups; without it their variants are skipped.
    optionalFeatures: ['subgroups'],
});

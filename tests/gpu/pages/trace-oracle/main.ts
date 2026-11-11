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
 * - `watertight-tinybvh-compact`: tinybvh trees through a kernel built with
 *   `hitNormal: false`, whose 24-byte records carry no material or normal;
 * - `merged-tinybvh`, `merged-hploc`: `mergeStaticInstances`. Each instance
 *   gets its own copy of its mesh and two of every three are marked static,
 *   so they merge into world-space meshes (one per flags-and-mask group)
 *   built by tinybvh or the GPU H-PLOC; the rest stay instances. The
 *   reference is the case's own instanced scene, so hits must name the source
 *   instance, its triangle and its authored side. The hook cases also run
 *   merged (`[hooks-merged]`), with the filter flags in the merged geometry.
 * - `indexed-*`, `merged-indexed-*`: indexed leaves (docs/design/indexed-leaves.md),
 *   16-byte triangles read through the scene's positions buffer, with tinybvh,
 *   the GPU LBVH (records converted) and H-PLOC (written on the GPU, checked by
 *   `checkGpuLayout`) trees, `vec4` and `packed` positions, masks compiled out,
 *   and `-weld` (a tolerance weld at 1e-6 before building). The hook cases also
 *   run indexed (`[hooks]` names starting `indexed`).
 * Every variant also runs the seam fixtures (`seamQuad` in cases.ts): split
 * seams, every corner its own vertex with the same bits, must let nothing
 * through; one-ulp seams leak in every format (the double-precision reference
 * misses there too), and the indexed tinybvh variants must leak exactly where
 * `watertight-tinybvh` does, ray for ray; the `-weld` variant must close them.
 * The leaks are reported as `oneUlpSeamLeaks`.
 * The variants above trace through the masked batch kernel (the default) with
 * a mask of all ones. Except for `-compact`, each closest hit's material and
 * world normal (`BvhHit.material`, `BvhHit.normal`) are checked against the
 * reference too (normals to a cosine of 0.9999), merged variants included:
 * several cases give most instances a material, and every variant must report
 * some hits with one. Then the hook cases (hooks.ts) run on tinybvh trees,
 * with ray masks on and off: a filter on flagged triangles, a filter that depends on
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
    INDEXED_FLAG_FILTER,
    type MeshBvh,
    type MeshGeometry,
    type TraceSceneInstance,
    createCpuTracer,
    createMeshBvhBuilder,
    HIT_RECORD_COMPACT_WORD,
    HIT_RECORD_WORD,
    hitRecordBytes,
    NO_MATERIAL,
    newTraceStats,
    TRACE_MODULE_ENTRIES,
    type MeshBvhBuilder,
    type TraceSceneData,
} from '../../../../src/trace/index.ts';
import { type TracedHit } from '../../../../src/trace/oracle-compare.ts';
import { checkGpuLayout, createMeshLayoutBuilder, createTinyBvhBuilder, weldPositions, type MeshLayoutBuilder } from '../../../../src/index.ts';
import { buildMeshBvhWithTinyBvh } from '../../../../src/trace/mesh-bvh-topology.ts';
import { RAY_BYTES } from '../../../../src/trace/rays.ts';
import { TraceBatchKernel, uploadTraceScene } from '../../../../src/trace/trace-gpu.ts';
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
    /** The batch kernel's `hitNormal`; default true. */
    hitNormal?: boolean;
    /** The scene's and kernel's triangle format (docs/design/indexed-leaves.md); default records. */
    triangleLeaves?: 'records' | 'indexed';
    /** Indexed: the positions layout; default vec4. */
    positionLayout?: 'vec4' | 'packed';
    /** Weld each mesh's vertices within this distance before building (`weldPositions`); default 0, off. */
    weldTolerance?: number;
}

/** The tolerance weld's distance in the `-weld` variant: far below the quads' 0.5 grid, far above an ulp at 2 (2.4e-7). */
const SEAM_WELD_TOLERANCE = 1e-6;

const VARIANTS: OracleVariant[] = [
    { name: 'watertight-tinybvh', treeSource: 'tinybvh' },
    { name: 'watertight-tmbuild', treeSource: 'three-mesh-bvh' },
    { name: 'watertight-lbvh', treeSource: 'gpu-lbvh' },
    { name: 'watertight-hploc', treeSource: 'gpu-hploc' },
    { name: 'watertight-tinybvh-nomasks', treeSource: 'tinybvh', rayMasks: false },
    { name: 'watertight-tinybvh-compact', treeSource: 'tinybvh', hitNormal: false },
    { name: 'merged-tinybvh', treeSource: 'tinybvh', merged: true },
    { name: 'merged-hploc', treeSource: 'gpu-hploc', merged: true },
    { name: 'indexed-tinybvh', treeSource: 'tinybvh', triangleLeaves: 'indexed' },
    { name: 'indexed-tinybvh-packed', treeSource: 'tinybvh', triangleLeaves: 'indexed', positionLayout: 'packed' },
    { name: 'indexed-tinybvh-nomasks', treeSource: 'tinybvh', triangleLeaves: 'indexed', rayMasks: false },
    { name: 'indexed-tinybvh-weld', treeSource: 'tinybvh', triangleLeaves: 'indexed', weldTolerance: SEAM_WELD_TOLERANCE },
    { name: 'indexed-lbvh', treeSource: 'gpu-lbvh', triangleLeaves: 'indexed' },
    { name: 'indexed-hploc', treeSource: 'gpu-hploc', triangleLeaves: 'indexed' },
    { name: 'merged-indexed-tinybvh', treeSource: 'tinybvh', merged: true, triangleLeaves: 'indexed' },
    { name: 'merged-indexed-hploc', treeSource: 'gpu-hploc', merged: true, triangleLeaves: 'indexed', positionLayout: 'packed' },
];

/** Variants whose one-ulp seam slips must equal `watertight-tinybvh`'s: the same trees and corner bits, the other format. */
/** The hooks' indexed passes build their trees and scene as this variant does. */
const INDEXED_HOOK_VARIANT: OracleVariant = { name: 'indexed-hooks', treeSource: 'tinybvh', triangleLeaves: 'indexed' };

const ULP_SAME_AS_RECORDS = ['indexed-tinybvh', 'indexed-tinybvh-packed'];

/** Rays per hook case and traced mask (hooks.ts); the reference runs once per mask. */
const HOOK_SIZES = { randomRays: 8000, thinBoxRays: 1200 };

/** The GPU builders (B1 LBVH, B2 H-PLOC), made once the device is known. */
const gpuBuilders = new Map<'gpu-lbvh' | 'gpu-hploc', MeshLayoutBuilder>();

/** three-mesh-bvh target leaf size for the converted trees, as in the benchmark's default. */
const TMBUILD_LEAF_SIZE = 4;

/** Rays per shared-edge scene. Two scenes, so 2 million in total for each triangle test. */
const DEFAULT_SHARED_EDGE_RAYS = 1_000_000;

/** Rays per seam scene (split and one-ulp seams, plain and transformed). */
const SEAM_RAYS = 200_000;

/** Rays the CPU port of the walk traces per case, to report the stack depth it needs. */
const STACK_PROBE_RAYS = 2000;

/** Rays of each shared-edge scene also checked against the brute-force reference. */
const SHARED_EDGE_REFERENCE_RAYS = 20000;

interface GpuAnswers {
    closest: Float32Array;
    closestWords: Uint32Array;
    blocked: Uint32Array;
    /** The records have material and normal (`hitNormal`), 40 bytes; else the compact 24. */
    hitNormal: boolean;
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
async function traceOnGpu(device: GPUDevice, encodeBatch: EncodeBatch, packedRays: Float32Array, hitNormal = true): Promise<GpuAnswers> {
    const count = packedRays.length / 8;
    const rays = device.createBuffer({ label: 'oracle rays', size: count * RAY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(rays, 0, packedRays as Float32Array<ArrayBuffer>);
    const hitBytes = count * hitRecordBytes(hitNormal);
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
        hitNormal,
    };
}


/** One ray's closest-hit record as the comparison code wants it. */
function hitAt(answers: GpuAnswers, index: number): TracedHit | null {
    const word = answers.hitNormal ? HIT_RECORD_WORD : HIT_RECORD_COMPACT_WORD;
    const base = (index * hitRecordBytes(answers.hitNormal)) / 4;
    const t = answers.closest[base + word.t];
    if (!(t >= 0)) return null;
    const hit: TracedHit = {
        hit: true,
        t,
        instanceIndex: answers.closestWords[base + word.instance],
        triangleIndex: answers.closestWords[base + word.triangle],
        frontFace: answers.closestWords[base + word.frontFace] !== 0,
        u: answers.closest[base + word.barycentrics],
        v: answers.closest[base + word.barycentrics + 1],
    };
    if (!answers.hitNormal) return hit;
    // Material and normal are compared with the reference (and hand-worked values) when given.
    const normal = answers.closest.subarray(base + HIT_RECORD_WORD.normal, base + HIT_RECORD_WORD.normal + 3);
    return { ...hit, material: answers.closestWords[base + HIT_RECORD_WORD.material], normal: [normal[0], normal[1], normal[2]] };
}

/** Closest hits whose record names a material (0 with compact records). */
function countMaterialHits(answers: GpuAnswers, rayCount: number): number {
    let count = 0;
    for (let index = 0; index < rayCount; index++) {
        const material = hitAt(answers, index)?.material;
        if (material !== undefined && material !== NO_MATERIAL) count++;
    }
    return count;
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
        const results = await gpuBuilder.buildLayouts(meshes, { triangleLeaves: variant.triangleLeaves });
        for (const result of results) assert(result.builtBy === variant.treeSource, `a mesh fell back to ${result.builtBy}: ${result.fallbackReason}`);
        // GPU-written indexed triangles (H-PLOC natively; the LBVH's records converted) must pass the layout check.
        if (variant.triangleLeaves === 'indexed') results.forEach((result, index) => assert(result.layout.triangleLeaves === 'indexed' && checkGpuLayout(result.layout, meshes[index]).length === 0, `${variant.name}: mesh ${index}'s indexed layout fails the check: ${checkGpuLayout(result.layout, meshes[index]).slice(0, 2).join('; ')}`));
        return results.map((result) => result.layout);
    }
    return Promise.all(meshes.map((mesh) => builder.build(mesh, { triangleLeaves: variant.triangleLeaves })));
}

/** The assembly options of a variant's format: indexed scenes take the geometry for their positions. */
function formatOptions(variant: OracleVariant, geometries: readonly (MeshGeometry | null)[]) {
    return variant.triangleLeaves === 'indexed' ? { triangleLeaves: 'indexed' as const, positionLayout: variant.positionLayout, geometries } : {};
}

/** The variant's input meshes: welded within its tolerance, or as they are. */
function variantMeshes(variant: OracleVariant, meshes: readonly MeshGeometry[]): readonly MeshGeometry[] {
    const tolerance = variant.weldTolerance ?? 0;
    return tolerance > 0 ? meshes.map((mesh) => weldPositions(mesh, { tolerance })) : meshes;
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
    return assembleTraceScene(sourceTrees, instances, {}, { merged: { plan, meshes: trees.slice(needed.length) }, ...formatOptions(variant, meshes) });
}

async function buildScene(builder: MeshBvhBuilder, oracleCase: OracleCase, variant: OracleVariant): Promise<BuiltScene> {
    const start = performance.now();
    const meshes = variantMeshes(variant, oracleCase.meshes);
    if (variant.merged) {
        const unshared = unsharedForMerge(meshes, oracleCase.instances);
        const data = await buildMergedScene(builder, unshared.meshes, unshared.instances, variant);
        return { data, buildMs: performance.now() - start };
    }
    const data = assembleTraceScene(await buildTrees(builder, meshes, variant), oracleCase.instances, {}, formatOptions(variant, meshes));
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
        return { answers: await traceOnGpu(device, (encoder, batch) => kernel.encode(encoder, buffers, batch), oracleCase.rays, variant.hitNormal ?? true), built };
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
    const materialHitsByVariant = new Map<string, number>();
    const seamReports = new Map<string, SharedEdgeReport[]>();
    const seamLeaks: Record<string, { slipped: number; slippedAny: number; referenceMisses: number }> = {};
    for (const variant of VARIANTS) {
        const gpuBuilder = variant.treeSource === 'gpu-lbvh' || variant.treeSource === 'gpu-hploc' ? gpuBuilders.get(variant.treeSource) : undefined;
        if (gpuBuilder && gpuBuilder.kind === 'tinybvh') {
            skipped.push(`${variant.name}: ${gpuBuilder.fallbackReason}`);
            log(`[${variant.name}] skipped: ${gpuBuilder.fallbackReason}`);
            continue;
        }
        const kernel = await TraceBatchKernel.create(device, {
            label: variant.name,
            rayMasks: variant.rayMasks ?? true,
            hitNormal: variant.hitNormal ?? true,
            triangleLeaves: variant.triangleLeaves,
            positionLayout: variant.positionLayout,
        });
        const results = { cases: {} as Record<string, unknown>, sharedEdges: [] as SharedEdgeReport[] };
        byTest[variant.name] = results;

        for (const [caseIndex, oracleCase] of cases.entries()) {
            const { answers, built } = await traceCase(device, kernel, builder, oracleCase, variant);

            const result = evaluateCase(oracleCase, tracerAnswers(answers), references[caseIndex]);
            const stackUsed = probeStackDepth(built.data, oracleCase);
            deepestStackUsed = Math.max(deepestStackUsed, stackUsed);
            deepestRequired = Math.max(deepestRequired, built.data.requiredStackSize);
            const materialHits = countMaterialHits(answers, oracleCase.rays.length / 8);
            if (answers.hitNormal) materialHitsByVariant.set(variant.name, (materialHitsByVariant.get(variant.name) ?? 0) + materialHits);
            results.cases[oracleCase.name] = { ...summarize(result, stackUsed, built.data, built.buildMs), materialHits };
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

        // Seams (indexed-leaves.md, section 2): split seams must hold in every format; one-ulp
        // seams leak alike in both formats, and the tolerance weld closes them.
        const seams: SharedEdgeReport[] = [];
        for (const seam of ['split', 'ulp'] as const) {
            for (const transformed of [false, true]) {
                const report = await runSharedEdgeCase(device, kernel, builder, sharedEdgeCase(SEAM_RAYS, transformed, seam), variant);
                seams.push(report);
                log(`[${variant.name}] ${report.name}: ${report.slipped} of ${report.rays} rays slipped (${report.slippedAny} any-hit)`);
                // Split seams hold everywhere. A one-ulp seam leaks (the double-precision reference
                // misses there too), except through a tolerance weld, which must close it.
                const welded = (variant.weldTolerance ?? 0) > 0;
                if (seam === 'split') for (const failure of sharedEdgeFailures(report)) failures.push(`[${variant.name}] ${report.name}: ${failure}`);
                else if (welded && report.slipped + report.slippedAny > 0) failures.push(`[${variant.name}] ${report.name}: ${report.slipped} (${report.slippedAny} any-hit) rays slipped through the welded seam`);
                if (seam === 'ulp') seamLeaks[`${variant.name} ${report.name}`] = { slipped: report.slipped, slippedAny: report.slippedAny, referenceMisses: report.referenceMisses };
            }
        }
        seamReports.set(variant.name, seams);
        kernel.destroy();
    }
    // The one-ulp seams must behave as with records: the same slips, ray for ray.
    const recordSeams = seamReports.get('watertight-tinybvh');
    for (const name of ULP_SAME_AS_RECORDS) {
        const seams = seamReports.get(name);
        if (!recordSeams || !seams) continue;
        seams.forEach((report, index) => {
            const expected = recordSeams[index];
            if (report.slipped !== expected.slipped || report.slippedAny !== expected.slippedAny || report.firstSlippedRays.join() !== expected.firstSlippedRays.join()) {
                failures.push(`[${name}] ${report.name}: ${report.slipped}/${report.slippedAny} slipped, records ${expected.slipped}/${expected.slippedAny}`);
            }
        });
    }

    // The hooks: triangle filters and instance masks, with ray masks on and off.
    const hooks: Record<string, unknown> = {};
    for (const [rayMasks, merged, indexed] of [[true, false, false], [false, false, false], [true, true, false], [true, false, true], [false, false, true], [true, true, true]] as const) {
        const format = indexed ? INDEXED_HOOK_VARIANT : VARIANTS[0];
        for (const hook of buildHookCases(HOOK_SIZES)) {
            const kernel = await TraceBatchKernel.create(device, { label: hook.name, triangleFilter: hook.filterWgsl, rayMasks, triangleLeaves: format.triangleLeaves });
            let scene: TraceSceneData;
            let flagged: number;
            if (merged) {
                const unshared = unsharedForMerge(hook.base.meshes, hook.base.instances, hook.flagged);
                scene = await buildMergedScene(builder, unshared.meshes, unshared.instances, format);
                flagged = indexed
                    ? new Uint32Array(scene.triangles).filter((word, at) => at % 4 === 3 && (word & INDEXED_FLAG_FILTER) !== 0).length
                    : new Uint32Array(scene.triangles).filter((word, at) => at % 12 === 7 && (word & TRIANGLE_FLAG_FILTER) !== 0).length;
            } else {
                scene = assembleTraceScene(await buildTrees(builder, hook.base.meshes, format), hook.base.instances, {}, formatOptions(format, hook.base.meshes));
                flagged = writeHookFlags(scene, hook);
            }
            const buffers = uploadTraceScene(device, scene, hook.name);
            try {
                for (const mask of hookMasks(hook, rayMasks)) {
                    const answers = await traceOnGpu(device, (encoder, batch) => kernel.encode(encoder, buffers, { ...batch, mask }), hook.base.rays);
                    const result = evaluateCase(hook.base, tracerAnswers(answers), await hookReferenceAnswers(hook, mask, rayMasks));
                    const name = `${indexed ? 'indexed ' : ''}${merged ? 'merged ' : ''}${hookPassName(hook, mask, rayMasks)}`;
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
    // The material comparison only means something if hits carry materials.
    for (const [name, count] of materialHitsByVariant) if (count === 0) failures.push(`[${name}] no closest hit reported a material`);

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
        oneUlpSeamLeaks: seamLeaks,
        materialHits: Object.fromEntries(materialHitsByVariant),
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

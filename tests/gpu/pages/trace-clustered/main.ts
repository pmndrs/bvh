/**
 * Clustered top level (`clusterTraceScene`, PARITY 8.9) on the GPU: the WGSL walk,
 * unchanged, traces scenes whose meshes were cut into clusters, each its own
 * top-level record pointing at a copied subtree.
 *
 * For every scene format (f32 and quantized nodes, records and indexed leaves,
 * merged static groups) and every oracle case, the plain scene and three cuts of it
 * are traced with the batch kernel:
 * - `1`: every leaf of every unshared mesh its own record (the deepest top level);
 * - `4-all`: a limit of 4 on every record, instanced meshes included (copies shared);
 * - `64`: the default (`DEFAULT_CLUSTER_PRIMITIVE_LIMIT`, unshared meshes only).
 * Each cut must pass the oracle's rules against the CPU reference (trace-oracle's
 * `evaluateCase`), and match the plain scene's GPU answers ray for ray: any hits
 * equal, closest hits with the same `t` to the bit and the same instance and
 * triangle unless another triangle lies at that `t` (a tie) or a few ulps from it (a
 * near tie, see `NEAR_TIE_RELATIVE`), both counted apart. The
 * shared-edge quads, cut to leaves, must let no ray through. The closest-point
 * query (`ClosestPointBatchKernel`, the shapecast walk) must give the plain scene's
 * distance to the bit on random points.
 *
 * `?profile=ci` (SwiftShader, `BVH_PAGE_QUERY=profile=ci`): every format, case and cut,
 * with trace-oracle's reduced ray counts.
 */

import {
    buildTraceScene,
    assembleTraceScene,
    clusterTraceScene,
    ClosestPointBatchKernel,
    CLOSEST_POINT_RECORD_BYTES,
    CLOSEST_POINT_RECORD_WORD,
    createMeshBvhBuilder,
    DEFAULT_CLUSTER_PRIMITIVE_LIMIT,
    HIT_RECORD_WORD,
    hitRecordBytes,
    POINT_QUERY_BYTES,
    RAY_BYTES,
    TraceBatchKernel,
    uploadTraceScene,
    type ClusteredTraceSceneData,
    type MeshBvhBuilder,
    type TraceBatch,
    type TraceSceneData,
    type TraceSceneInstance,
    type MeshGeometry,
} from '../../../../src/trace/index.ts';
import type { TracedHit } from '../../../../src/trace/oracle-compare.ts';
import { createRandom } from '../../../../src/trace/oracle-fixtures.ts';
import { assert, runGpuTest } from '../../client.ts';
import { buildOracleCases, FULL_SIZES, sharedEdgeCase, type OracleCase } from '../trace-oracle/cases.ts';
import { computeReferenceAnswersYielding, evaluateCase, evaluateSharedEdgeCase, sharedEdgeFailures, type TracerAnswers } from '../trace-oracle/check.ts';

const reduced = new URLSearchParams(location.search).get('profile') === 'ci';
const CASE_SIZES = reduced ? { randomRays: 2000, thinBoxRays: 2400 } : FULL_SIZES;
const SHARED_EDGE_RAYS = reduced ? 20_000 : 200_000;
const SHARED_EDGE_REFERENCE_RAYS = 5000;
const POINT_QUERIES = reduced ? 1000 : 10_000;

interface Format {
    readonly name: string;
    readonly nodeFormat: 'f32' | 'quantized';
    readonly triangleLeaves: 'records' | 'indexed';
    readonly positionLayout?: 'vec4' | 'packed';
    readonly merged?: boolean;
}
const FORMATS: readonly Format[] = [
    { name: 'f32', nodeFormat: 'f32', triangleLeaves: 'records' },
    { name: 'quantized', nodeFormat: 'quantized', triangleLeaves: 'records' },
    { name: 'f32-indexed-packed', nodeFormat: 'f32', triangleLeaves: 'indexed', positionLayout: 'packed' },
    { name: 'quantized-indexed', nodeFormat: 'quantized', triangleLeaves: 'indexed' },
    { name: 'merged-quantized', nodeFormat: 'quantized', triangleLeaves: 'records', merged: true },
    { name: 'merged-quantized-indexed', nodeFormat: 'quantized', triangleLeaves: 'indexed', merged: true },
];

interface Cut {
    readonly name: string;
    readonly cluster: (scene: TraceSceneData) => ClusteredTraceSceneData;
}
const CUTS: readonly Cut[] = [
    { name: '1', cluster: (scene) => clusterTraceScene(scene, { primitiveLimit: 1 }) },
    { name: '4-all', cluster: (scene) => clusterTraceScene(scene, { primitiveLimit: 4, split: () => true }) },
    { name: String(DEFAULT_CLUSTER_PRIMITIVE_LIMIT), cluster: (scene) => clusterTraceScene(scene) },
];

async function assemble(builder: MeshBvhBuilder, meshes: readonly MeshGeometry[], instances: readonly TraceSceneInstance[], format: Format): Promise<TraceSceneData> {
    const leaves = format.triangleLeaves === 'indexed' ? { triangleLeaves: 'indexed' as const, positionLayout: format.positionLayout } : {};
    if (format.merged) {
        // Every instance static: those whose mesh no other instance uses merge into world-space meshes.
        const statics = instances.map((instance) => ({ ...instance, static: true }));
        return buildTraceScene(meshes, statics, { build: (geometry) => builder.build(geometry), nodeFormat: format.nodeFormat, ...leaves });
    }
    const trees = await Promise.all(meshes.map((mesh) => builder.build(mesh)));
    return assembleTraceScene(trees, instances, {}, { nodeFormat: format.nodeFormat, ...leaves, ...(format.triangleLeaves === 'indexed' ? { geometries: meshes } : {}) });
}

async function readBuffer(device: GPUDevice, source: GPUBuffer, size: number): Promise<ArrayBuffer> {
    const readback = device.createBuffer({ label: 'trace-clustered readback', size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readback, 0, size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();
    return copy;
}

/** The batch kernel's answers for packed rays: 40-byte closest records and any-hit words. */
interface GpuAnswers {
    readonly floats: Float32Array;
    readonly words: Uint32Array;
    readonly blocked: Uint32Array;
}

async function trace(device: GPUDevice, kernel: TraceBatchKernel, scene: TraceSceneData, packedRays: Float32Array): Promise<GpuAnswers> {
    const buffers = uploadTraceScene(device, scene, 'trace-clustered');
    const count = packedRays.length / 8;
    const rays = device.createBuffer({ label: 'trace-clustered rays', size: count * RAY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(rays, 0, packedRays as Float32Array<ArrayBuffer>);
    const hitBytes = count * hitRecordBytes(true);
    const hits = device.createBuffer({ label: 'trace-clustered hits', size: hitBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const occlusion = device.createBuffer({ label: 'trace-clustered any', size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const encoder = device.createCommandEncoder({ label: 'trace-clustered' });
    const batches: TraceBatch[] = [
        { mode: 'closest', rays, results: hits, count },
        { mode: 'any', rays, results: occlusion, count },
    ];
    for (const batch of batches) kernel.encode(encoder, buffers, batch);
    device.queue.submit([encoder.finish()]);
    const hitData = await readBuffer(device, hits, hitBytes);
    const blocked = new Uint32Array(await readBuffer(device, occlusion, count * 4));
    for (const buffer of [rays, hits, occlusion]) buffer.destroy();
    buffers.destroy();
    return { floats: new Float32Array(hitData), words: new Uint32Array(hitData), blocked };
}

function hitAt(answers: GpuAnswers, index: number): TracedHit | null {
    const base = (index * hitRecordBytes(true)) / 4;
    const t = answers.floats[base + HIT_RECORD_WORD.t];
    if (!(t >= 0)) return null;
    const normal = HIT_RECORD_WORD.normal;
    return {
        hit: true,
        t,
        instanceIndex: answers.words[base + HIT_RECORD_WORD.instance],
        triangleIndex: answers.words[base + HIT_RECORD_WORD.triangle],
        frontFace: answers.words[base + HIT_RECORD_WORD.frontFace] !== 0,
        u: answers.floats[base + HIT_RECORD_WORD.barycentrics],
        v: answers.floats[base + HIT_RECORD_WORD.barycentrics + 1],
        material: answers.words[base + HIT_RECORD_WORD.material],
        normal: [answers.floats[base + normal], answers.floats[base + normal + 1], answers.floats[base + normal + 2]],
    };
}

function tracerAnswers(answers: GpuAnswers): TracerAnswers {
    return { closest: (index) => hitAt(answers, index), any: (index) => answers.blocked[index] !== 0 };
}

/**
 * Near ties: two triangles a few ulps apart along the ray (faces meeting at an edge or
 * corner). The f32 walk does not always keep the nearer: once it holds the farther
 * hit, a box whose rounded entry lands past that hit is culled even though its
 * triangle lies an ulp before it. Which box tests come first depends on the tree,
 * so plain and clustered scenes can keep different ones (as two builders' trees
 * can; the oracle counts them as ties within `T_TOLERANCE`).
 */
const NEAR_TIE_RELATIVE = 2 ** -20;

/** Clustered against plain, ray for ray: the record words to the bit, except which triangle at a tied (or near-tied) `t`. */
function comparePlain(plain: GpuAnswers, clustered: GpuAnswers, count: number): { ties: number; nearTies: number; failures: string[] } {
    const failures: string[] = [];
    let ties = 0;
    let nearTies = 0;
    const words = hitRecordBytes(true) / 4;
    for (let index = 0; index < count && failures.length < 4; index++) {
        if (plain.blocked[index] !== clustered.blocked[index]) failures.push(`ray ${index}: any hit ${plain.blocked[index]} plain, ${clustered.blocked[index]} clustered`);
        const base = index * words;
        const sameTriangle =
            plain.words[base + HIT_RECORD_WORD.instance] === clustered.words[base + HIT_RECORD_WORD.instance] && plain.words[base + HIT_RECORD_WORD.triangle] === clustered.words[base + HIT_RECORD_WORD.triangle];
        if (plain.words[base + HIT_RECORD_WORD.t] !== clustered.words[base + HIT_RECORD_WORD.t]) {
            const a = plain.floats[base + HIT_RECORD_WORD.t];
            const b = clustered.floats[base + HIT_RECORD_WORD.t];
            if (!sameTriangle && a >= 0 && b >= 0 && Math.abs(a - b) <= Math.max(a, b) * NEAR_TIE_RELATIVE) nearTies++;
            else failures.push(`ray ${index}: t ${a} plain (instance ${plain.words[base + 1]}, triangle ${plain.words[base + 2]}), ${b} clustered (instance ${clustered.words[base + 1]}, triangle ${clustered.words[base + 2]})`);
            continue;
        }
        let same = true;
        for (let k = 0; k < words; k++) if (plain.words[base + k] !== clustered.words[base + k]) same = false;
        if (same) continue;
        if (sameTriangle) failures.push(`ray ${index}: the same triangle with other fields`);
        else ties++;
    }
    return { ties, nearTies, failures };
}

/** Closest-point records for `points` (vec4f: point, maxDistance Infinity). */
async function closestPoints(device: GPUDevice, kernel: ClosestPointBatchKernel, scene: TraceSceneData, points: Float32Array): Promise<Uint32Array> {
    const buffers = uploadTraceScene(device, scene, 'trace-clustered points');
    const count = points.length / 4;
    const queries = device.createBuffer({ label: 'trace-clustered queries', size: count * POINT_QUERY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(queries, 0, points as Float32Array<ArrayBuffer>);
    const results = device.createBuffer({ label: 'trace-clustered closest points', size: count * CLOSEST_POINT_RECORD_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const encoder = device.createCommandEncoder({ label: 'trace-clustered points' });
    kernel.encode(encoder, buffers, { queries, results, count });
    device.queue.submit([encoder.finish()]);
    const words = new Uint32Array(await readBuffer(device, results, count * CLOSEST_POINT_RECORD_BYTES));
    queries.destroy();
    results.destroy();
    buffers.destroy();
    return words;
}

/** Random points around the case's rays' origins. */
function randomPoints(oracleCase: OracleCase, count: number): Float32Array {
    const random = createRandom(4242);
    const rayCount = oracleCase.rays.length / 8;
    const points = new Float32Array(count * 4);
    for (let index = 0; index < count; index++) {
        const ray = Math.floor(random() * rayCount) * 8;
        for (let axis = 0; axis < 3; axis++) points[index * 4 + axis] = oracleCase.rays[ray + axis] + (random() - 0.5) * 2;
        points[index * 4 + 3] = Infinity;
    }
    return points;
}

await runGpuTest(async ({ device, log }) => {
    const failures: string[] = [];
    const report: Record<string, unknown> = {};
    const builder = createMeshBvhBuilder({ worker: 'auto' });
    const oracleCases = buildOracleCases(CASE_SIZES);
    const references = [];
    for (const oracleCase of oracleCases) references.push(await computeReferenceAnswersYielding(oracleCase));
    try {
        for (const format of FORMATS) {
            const formatOptions = { nodeFormat: format.nodeFormat, triangleLeaves: format.triangleLeaves, positionLayout: format.positionLayout };
            const kernel = await TraceBatchKernel.create(device, { label: `trace-clustered ${format.name}`, ...formatOptions });
            const pointKernel = await ClosestPointBatchKernel.create(device, { label: `trace-clustered ${format.name}`, ...formatOptions });
            for (const [caseIndex, oracleCase] of oracleCases.entries()) {
                const scene = await assemble(builder, oracleCase.meshes, oracleCase.instances, format);
                const rayCount = oracleCase.rays.length / 8;
                const plain = await trace(device, kernel, scene, oracleCase.rays);
                const points = randomPoints(oracleCase, POINT_QUERIES);
                const plainPoints = await closestPoints(device, pointKernel, scene, points);
                for (const cut of CUTS) {
                    const where = `[${format.name} ${cut.name}] ${oracleCase.name}`;
                    const clustered = cut.cluster(scene);
                    const answers = await trace(device, kernel, clustered, oracleCase.rays);
                    const result = evaluateCase(oracleCase, tracerAnswers(answers), references[caseIndex]);
                    const compared = comparePlain(plain, answers, rayCount);
                    const clusteredPoints = await closestPoints(device, pointKernel, clustered, points);
                    let pointMismatches = 0;
                    let pointTies = 0;
                    const stride = CLOSEST_POINT_RECORD_BYTES / 4;
                    for (let index = 0; index < POINT_QUERIES; index++) {
                        const base = index * stride;
                        if (plainPoints[base + CLOSEST_POINT_RECORD_WORD.distance] !== clusteredPoints[base + CLOSEST_POINT_RECORD_WORD.distance]) pointMismatches++;
                        else if (plainPoints[base + CLOSEST_POINT_RECORD_WORD.triangle] !== clusteredPoints[base + CLOSEST_POINT_RECORD_WORD.triangle] || plainPoints[base + CLOSEST_POINT_RECORD_WORD.instance] !== clusteredPoints[base + CLOSEST_POINT_RECORD_WORD.instance]) pointTies++;
                    }
                    report[`${format.name} ${cut.name} ${oracleCase.name}`] = {
                        records: `${clustered.clusters.records} -> ${clustered.clusters.clusterRecords}`,
                        closest: result.closest.mismatches,
                        any: result.any.mismatches,
                        ties: compared.ties,
                        nearTies: compared.nearTies,
                        points: { mismatches: pointMismatches, ties: pointTies },
                    };
                    log(`${where}: ${clustered.clusters.records} -> ${clustered.clusters.clusterRecords} records, closest ${result.closest.mismatches} mismatches, any ${result.any.mismatches}, ${compared.ties} ties and ${compared.nearTies} near ties with plain, points ${pointMismatches} off`);
                    for (const failure of result.failures) failures.push(`${where}: ${failure}`);
                    for (const failure of compared.failures) failures.push(`${where} against plain: ${failure}`);
                    if (pointMismatches > 0) failures.push(`${where}: ${pointMismatches} of ${POINT_QUERIES} closest-point distances differ from the plain scene's`);
                }
            }

            // Shared edges, every leaf its own record: nothing may slip between triangles.
            for (const transformed of [false, true]) {
                const edgeCase = sharedEdgeCase(SHARED_EDGE_RAYS, transformed);
                const clustered = clusterTraceScene(await assemble(builder, edgeCase.meshes, edgeCase.instances, format), { primitiveLimit: 1, split: () => true });
                const edges = evaluateSharedEdgeCase(edgeCase, tracerAnswers(await trace(device, kernel, clustered, edgeCase.rays)), SHARED_EDGE_REFERENCE_RAYS);
                report[`${format.name} ${edges.name}`] = { rays: edges.rays, slipped: edges.slipped, slippedAny: edges.slippedAny, records: clustered.clusters.clusterRecords };
                log(`[${format.name}] ${edges.name}: ${edges.slipped} of ${edges.rays} slipped (${edges.slippedAny} any-hit) over ${clustered.clusters.clusterRecords} records`);
                for (const failure of sharedEdgeFailures(edges)) failures.push(`[${format.name}] ${edges.name}: ${failure}`);
            }
            kernel.destroy();
            pointKernel.destroy();
        }
    } finally {
        builder.dispose();
    }
    assert(failures.length === 0, failures.slice(0, 12).join('\n'));
    return { profile: reduced ? 'ci' : 'full', results: report };
});

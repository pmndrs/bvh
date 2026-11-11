/**
 * Inline tracing from a caller's own kernel: the trace module's WGSL
 * (`traceModuleWgsl`, group 0: nodes, triangles, top level) is placed in front of a
 * kernel that calls `bvhTraceClosest` and `bvhTraceAny` itself, with three storage
 * buffers of its own in group 1 (rays in, hits out, any-hit answers out). That is
 * 3 + 3 = 6 storage buffers, inside the default per-stage limit of 8. The kernel supplies the walk's
 * triangle filter itself (`triangleFilter: null`, then `DEFAULT_TRIANGLE_FILTER_WGSL`: every
 * triangle blocks), and calls the masked closest-hit with all ones next to the plain any-hit.
 *
 * The answers are checked against the CPU reference on the oracle scenes, mirrored
 * boxes and thin walls included, plus rays at shared edges. Default device limits,
 * on purpose.
 *
 * The ddgi-side version of this test (createGI, the scene API, a technique's kernel
 * through the platform's kernel builder, the scene-copy material and normal reads,
 * the 9th-buffer GILimitError) stayed in webgpu-ddgi.
 */

import {
    assembleTraceScene,
    createMeshBvhBuilder,
    TRACE_MODULE_ENTRIES,
    traceModuleWgsl,
    type TraceSceneData,
} from '../../../../src/trace/index.ts';
import type { TracedHit } from '../../../../src/trace/oracle-compare.ts';
import { DEFAULT_TRIANGLE_FILTER_WGSL } from '../../../../src/trace/triangle-filter.ts';
import { assert, runGpuTest } from '../../client.ts';
import { RAY_BYTES, uploadTraceScene } from '../../trace-gpu.ts';
import { buildOracleCases, FULL_SIZES, sharedEdgeCase, type OracleCase } from '../trace-oracle/cases.ts';
import {
    computeReferenceAnswersYielding,
    evaluateCase,
    evaluateSharedEdgeCase,
    sharedEdgeFailures,
    type TracerAnswers,
} from '../trace-oracle/check.ts';

/** Rays per shared-edge scene; the oracle page runs a million, this checks the inline path. */
const SHARED_EDGE_RAYS = 200_000;
const SHARED_EDGE_REFERENCE_RAYS = 5000;

const WORKGROUP_SIZE = 64;
/** 32-bit words the kernel writes per ray. */
const RESULT_WORDS = 8;
/** Written as the instance for a miss. */
const NO_INSTANCE = 0xffffffff;

const KERNEL_WGSL = /* wgsl */ `
struct InlineResult {
    t: f32,
    instance: u32,
    triangle: u32,
    frontFace: u32,
    barycentrics: vec2f,
    padding0: u32,
    padding1: u32,
}

@group(1) @binding(0) var<storage, read> inlineRays: array<BvhRay>;
@group(1) @binding(1) var<storage, read_write> inlineHits: array<InlineResult>;
@group(1) @binding(2) var<storage, read_write> inlineBlocked: array<u32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
    // 2D dispatch, so a large batch stays under the per-dimension workgroup limit.
    let index = id.y * groups.x * ${WORKGROUP_SIZE}u + id.x;
    if (index >= arrayLength(&inlineRays)) {
        return;
    }
    let ray = inlineRays[index];
    // Masked with all ones: the same answers as bvhTraceClosest(ray).
    let hit = bvhTraceClosestMasked(ray, 0xffffffffu);
    var result = InlineResult(hit.t, ${NO_INSTANCE}u, hit.triangle, select(0u, 1u, hit.frontFace), hit.barycentrics, 0u, 0u);
    if (hit.t >= 0.0) {
        result.instance = hit.instance;
    }
    inlineHits[index] = result;
    inlineBlocked[index] = select(0u, 1u, bvhTraceAny(ray));
}
`;

async function readBuffer(device: GPUDevice, source: GPUBuffer, size: number): Promise<ArrayBuffer> {
    const readback = device.createBuffer({ label: 'trace-inline readback', size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readback, 0, size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();
    return copy;
}

/** Builds the case's trees with tinybvh and assembles the scene, as the oracle's tinybvh variant does. */
async function assemble(oracleCase: OracleCase): Promise<TraceSceneData> {
    const builder = createMeshBvhBuilder({ worker: 'auto' });
    try {
        const bvhs = await Promise.all(oracleCase.meshes.map((mesh) => builder.build(mesh)));
        return assembleTraceScene(bvhs, oracleCase.instances);
    } finally {
        builder.dispose();
    }
}

/** Traces one oracle case's rays in the page's own kernel. */
async function traceCase(device: GPUDevice, pipeline: GPUComputePipeline, oracleCase: OracleCase): Promise<TracerAnswers> {
    const scene = uploadTraceScene(device, await assemble(oracleCase), `inline ${oracleCase.name}`);
    const count = oracleCase.rays.length / 8;
    const rayBuffer = device.createBuffer({ label: 'trace-inline rays', size: count * RAY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(rayBuffer, 0, oracleCase.rays as Float32Array<ArrayBuffer>);
    const hitBytes = count * RESULT_WORDS * 4;
    const hits = device.createBuffer({ label: 'trace-inline hits', size: hitBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const blocked = device.createBuffer({ label: 'trace-inline blocked', size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

    const sceneGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: scene.nodes } },
            { binding: 1, resource: { buffer: scene.triangles } },
            { binding: 2, resource: { buffer: scene.topLevel } },
        ],
    });
    const ownGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(1),
        entries: [
            { binding: 0, resource: { buffer: rayBuffer, size: count * RAY_BYTES } },
            { binding: 1, resource: { buffer: hits } },
            { binding: 2, resource: { buffer: blocked } },
        ],
    });
    const workgroups = Math.ceil(count / WORKGROUP_SIZE);
    const x = Math.min(workgroups, device.limits.maxComputeWorkgroupsPerDimension);
    const encoder = device.createCommandEncoder({ label: 'trace-inline' });
    const pass = encoder.beginComputePass({ label: 'trace-inline' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, sceneGroup);
    pass.setBindGroup(1, ownGroup);
    pass.dispatchWorkgroups(x, Math.ceil(workgroups / x));
    pass.end();
    device.queue.submit([encoder.finish()]);

    const hitData = await readBuffer(device, hits, hitBytes);
    const blockedWords = new Uint32Array(await readBuffer(device, blocked, count * 4));
    for (const buffer of [rayBuffer, hits, blocked]) buffer.destroy();
    scene.destroy();

    const floats = new Float32Array(hitData);
    const words = new Uint32Array(hitData);
    const closest = (index: number): TracedHit | null => {
        const base = index * RESULT_WORDS;
        const t = floats[base];
        if (!(t >= 0)) return null;
        return {
            hit: true,
            t,
            instanceIndex: words[base + 1],
            triangleIndex: words[base + 2],
            frontFace: words[base + 3] !== 0,
            u: floats[base + 4],
            v: floats[base + 5],
        };
    };
    return { closest, any: (index) => blockedWords[index] !== 0 };
}

await runGpuTest(async ({ device, log }) => {
    // runGpuTest asks for no limits, so this is a default device.
    assert(device.limits.maxStorageBuffersPerShaderStage === 8, `expected a default-limits device, got ${device.limits.maxStorageBuffersPerShaderStage} storage buffers per stage`);
    const pipeline = await device.createComputePipelineAsync({
        label: 'trace-inline',
        layout: 'auto',
        compute: {
            module: device.createShaderModule({ label: 'trace-inline', code: `${traceModuleWgsl({ traceSlot: { group: 0, firstBinding: 0 }, triangleFilter: null })}\n${DEFAULT_TRIANGLE_FILTER_WGSL}\n${KERNEL_WGSL}` }),
            entryPoint: 'main',
        },
    });
    const failures: string[] = [];
    const cases: Record<string, unknown> = {};

    for (const oracleCase of buildOracleCases(FULL_SIZES)) {
        const reference = await computeReferenceAnswersYielding(oracleCase);
        const answers = await traceCase(device, pipeline, oracleCase);
        const result = evaluateCase(oracleCase, answers, reference);
        cases[oracleCase.name] = {
            rays: result.rays,
            closest: { mismatches: result.closest.mismatches, ambiguous: result.closest.ambiguous },
            any: { mismatches: result.any.mismatches, ambiguous: result.any.ambiguous },
            escapes: result.closestEscapes.ordinary + result.closestEscapes.edgeAimed + result.anyEscapes.ordinary + result.anyEscapes.edgeAimed,
        };
        log(`${oracleCase.name}: ${result.rays} rays, closest ${result.closest.mismatches} mismatches, any ${result.any.mismatches}`);
        for (const failure of result.failures) failures.push(`${oracleCase.name}: ${failure}`);
    }

    const sharedEdges: Record<string, unknown> = {};
    for (const transformed of [false, true]) {
        const oracleCase = sharedEdgeCase(SHARED_EDGE_RAYS, transformed);
        const report = evaluateSharedEdgeCase(oracleCase, await traceCase(device, pipeline, oracleCase), SHARED_EDGE_REFERENCE_RAYS);
        sharedEdges[report.name] = { rays: report.rays, slipped: report.slipped, slippedAny: report.slippedAny };
        log(`${report.name}: ${report.slipped} of ${report.rays} slipped (${report.slippedAny} any-hit)`);
        for (const failure of sharedEdgeFailures(report)) failures.push(`${report.name}: ${failure}`);
    }

    assert(failures.length === 0, failures.slice(0, 12).join('\n'));
    return {
        storageBuffers: { tracer: TRACE_MODULE_ENTRIES.length, own: 3, limit: device.limits.maxStorageBuffersPerShaderStage },
        cases,
        sharedEdges,
    };
});

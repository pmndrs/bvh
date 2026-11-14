/**
 * The CPU walk (`createCpuTracer`, doubles) against this GPU's walk, ray for
 * ray (pmndrs/bvh#1, item 1). The same scene buffers go to the GPU
 * (`TraceBatchKernel`, 40-byte records) and to the CPU walk. For each closest
 * hit the page sorts the pair into: exact (the same instance, triangle, side,
 * material and the same bits of `t`, barycentrics and normal), the same
 * triangle with other bits (and how many f32 steps apart `t` is), a tie
 * (another triangle at the same `t` bits), another triangle, or one hit where
 * the other missed. Any hits must block the same rays. Ray sets:
 *
 * - `primary`: one camera ray per pixel centre from the scene's first camera
 *   bookmark (`?width=384&height=256`, 98,304 rays), and `air`: incoherent rays
 *   from a grid in its manifest volume (`?airRays=`). Through the default scene
 *   (merged static instances, quantized nodes) and the instanced `f32` scene.
 *   Needs the assets (`BVH_ASSETS_DIR`, `BVH_MANIFESTS_DIR`); `?scene=` picks the
 *   manifest (default `sponza`), `?scene=none` leaves it out.
 * - `shared-edges`, `shared-edges-transformed`: the trace oracle's seam rays
 *   (`sharedEdgeRays`, `?sharedRays=`, default 200,000), f32 and quantized nodes.
 * - the trace oracle's random cases (instancing, mirrors, a deep mesh).
 *
 * Why bits are reported, never gated: WGSL lets a GPU round `1.0 / x`, `a / b`
 * and `sqrt` within a few ulp and fuse `dot()`, and the walk's ray setup
 * divides, so no CPU arithmetic reproduces a GPU's bits (the
 * `wgsl-f32-arithmetic` page measures what a GPU does). The verdict: any hits
 * agree on every ray of every set, closest hits never disagree on hit or miss,
 * and on primary rays every closest hit names the GPU's instance and triangle,
 * unless the two hits are a near tie: another triangle whose `t` is within
 * `NEAR_TIE_ULPS` f32 steps of the GPU's, as at overlapping coplanar faces
 * (Sponza has some; the Intel Gen12 walk and the CPU walk order one such pair
 * differently, both hits well inside their triangles, `t` 2 ulp apart). No walk
 * can order surfaces closer than its own rounding, which on one triangle
 * already differs by tens of ulp; near ties are counted, not failed
 * (`?strict=0` reports without failing). Edge-aimed rays may name the other
 * neighbour at a seam; they are counted, not failed.
 */

import {
    buildTraceScene,
    createCpuTracer,
    createMeshBvhBuilder,
    unpackRay,
    type CpuHit,
    type MeshBvh,
    type MeshGeometry,
    type TraceRay,
    type TraceSceneData,
    type TraceSceneInstance,
} from '../../../../src/trace/index.ts';
import { TraceBatchKernel, uploadTraceScene } from '../../../../src/trace/trace-gpu.ts';
import { assert, runGpuTest } from '../../client.ts';
import { traceOnGpu, type GpuHits } from '../../scene-check.ts';
import { buildOracleCases, FULL_SIZES, sharedEdgeCase } from '../trace-oracle/cases.ts';
import { findManifest, loadBenchScene } from '../trace-bench/scene-source.ts';
import { makeProbeGrid, makeProbeRays, makeSphereDirections, normalizeVector } from '../trace-bench/rays.ts';

const query = new URLSearchParams(location.search);
const SCENE = query.get('scene') ?? 'sponza';
const WIDTH = Number(query.get('width')) || 384;
const HEIGHT = Number(query.get('height')) || 256;
const AIR_RAYS = Number(query.get('airRays')) || 65536;
const SHARED_RAYS = Number(query.get('sharedRays')) || 200000;
const STRICT = query.get('strict') !== '0';
const EXAMPLES = 12;
/** Another triangle this close in `t` is a near tie (above the same-triangle spread primary rays show: up to 42 ulp). */
const NEAR_TIE_ULPS = 64;

const bitView = new DataView(new ArrayBuffer(4));
function bits(value: number): number {
    bitView.setFloat32(0, value);
    return bitView.getUint32(0);
}
const hex = (value: number): string => `0x${bits(value).toString(16).padStart(8, '0')}`;
/** Distance in f32 steps between two finite f32 values of the same sign. */
function ulps(a: number, b: number): number {
    return Math.abs(bits(a) - bits(b));
}

interface Tally {
    rays: number;
    /** Both missed. */
    misses: number;
    /** Both hit, same instance, triangle, side, material, and the same t, u, v and normal bits. */
    exact: number;
    /** Same triangle and the same t bits, but the barycentrics or the normal differ in bits. */
    sameTriangleTExact: number;
    /** Same triangle, t differs in bits. */
    sameTriangleTDiffers: number;
    /** Another triangle at exactly the same t (a tie the walks broke differently). */
    tieOtherTriangle: number;
    /** Another triangle within NEAR_TIE_ULPS of the same t (overlapping faces, or a seam). */
    nearTieOtherTriangle: number;
    /** Another triangle at another t. */
    otherTriangle: number;
    /** One hit, the other missed. */
    hitMiss: number;
    /** Same triangle: the largest t difference in f32 steps. */
    maxTUlps: number;
    normalDiffers: number;
    anyAgree: number;
    anyDisagree: number;
    /** The first problem rays of any kind. */
    examples: string[];
    /** The first rays that name another triangle or disagree on hit, kept apart so bit noise cannot crowd them out. */
    gateExamples: string[];
}

function newTally(): Tally {
    return { rays: 0, misses: 0, exact: 0, sameTriangleTExact: 0, sameTriangleTDiffers: 0, tieOtherTriangle: 0, nearTieOtherTriangle: 0, otherTriangle: 0, hitMiss: 0, maxTUlps: 0, normalDiffers: 0, anyAgree: 0, anyDisagree: 0, examples: [], gateExamples: [] };
}

function describeHit(hit: CpuHit | null): string {
    if (!hit) return 'miss';
    return `${hit.instanceIndex}/${hit.triangleIndex} t=${hit.t} (${hex(hit.t)}) uv=(${hit.u}, ${hit.v}) ${hit.frontFace ? 'front' : 'back'}`;
}

function compareRay(tally: Tally, index: number, ray: TraceRay, gpu: CpuHit | null, cpu: CpuHit | null, gpuAny: boolean, cpuAny: boolean): void {
    tally.rays++;
    if (gpuAny === cpuAny) tally.anyAgree++;
    else {
        tally.anyDisagree++;
        if (tally.gateExamples.length < EXAMPLES) tally.gateExamples.push(`ray ${index} any: GPU ${gpuAny}, CPU ${cpuAny}`);
    }
    let problem: string | null = null;
    if (!gpu || !cpu) {
        if (!gpu && !cpu) tally.misses++;
        else {
            tally.hitMiss++;
            problem = 'hit/miss';
        }
    } else if (gpu.instanceIndex === cpu.instanceIndex && gpu.triangleIndex === cpu.triangleIndex) {
        const tSame = bits(gpu.t) === bits(cpu.t);
        const restSame = gpu.frontFace === cpu.frontFace && gpu.material === cpu.material && bits(gpu.u) === bits(cpu.u) && bits(gpu.v) === bits(cpu.v);
        const normalSame = gpu.normal.every((value, axis) => bits(value) === bits(cpu.normal[axis]));
        if (!normalSame) tally.normalDiffers++;
        if (tSame && restSame && normalSame) tally.exact++;
        else if (tSame) {
            tally.sameTriangleTExact++;
            problem = restSame ? 'normal bits' : 'barycentric/side/material bits';
        } else {
            tally.sameTriangleTDiffers++;
            tally.maxTUlps = Math.max(tally.maxTUlps, ulps(gpu.t, cpu.t));
            problem = `t differs by ${ulps(gpu.t, cpu.t)} ulp`;
        }
    } else if (bits(gpu.t) === bits(cpu.t)) {
        tally.tieOtherTriangle++;
        problem = 'tie, other triangle';
    } else if (ulps(gpu.t, cpu.t) <= NEAR_TIE_ULPS) {
        tally.nearTieOtherTriangle++;
        problem = `near tie (${ulps(gpu.t, cpu.t)} ulp), other triangle`;
    } else {
        tally.otherTriangle++;
        problem = 'other triangle';
    }
    if (!problem) return;
    const line = `ray ${index} [${problem}] o=(${ray.origin.map(hex).join(',')}) d=(${ray.direction.map(hex).join(',')}) tMin=${ray.tMin} tMax=${ray.tMax}: GPU ${describeHit(gpu)}; CPU ${describeHit(cpu)}`;
    const gates = problem === 'hit/miss' || problem.endsWith('other triangle');
    if (gates && tally.gateExamples.length < EXAMPLES) tally.gateExamples.push(line);
    else if (!gates && tally.examples.length < EXAMPLES) tally.examples.push(line);
}

interface RaySet {
    readonly name: string;
    readonly rays: readonly TraceRay[];
    /** Every closest hit must name the GPU's instance and triangle. */
    readonly identityGate?: boolean;
}

interface SceneVariant {
    readonly name: string;
    readonly data: TraceSceneData;
    readonly nodeFormat: 'f32' | 'quantized';
}

/** One camera ray through each pixel's centre, as a pinhole camera (row by row, top first). */
function primaryRays(camera: { position: readonly number[]; target: readonly number[]; fovYDeg: number }, width: number, height: number): TraceRay[] {
    const forward = normalizeVector([0, 1, 2].map((axis) => camera.target[axis] - camera.position[axis]));
    const cross = (a: readonly number[], b: readonly number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const right = normalizeVector(cross(forward, [0, 1, 0]));
    const up = cross(right, forward);
    const tanHalfY = Math.tan((camera.fovYDeg * Math.PI) / 360);
    const tanHalfX = tanHalfY * (width / height);
    const origin: [number, number, number] = [Math.fround(camera.position[0]), Math.fround(camera.position[1]), Math.fround(camera.position[2])];
    const rays: TraceRay[] = [];
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const screenX = (2 * (x + 0.5)) / width - 1;
            const screenY = 1 - (2 * (y + 0.5)) / height;
            const direction = normalizeVector([0, 1, 2].map((axis) => forward[axis] + right[axis] * screenX * tanHalfX + up[axis] * screenY * tanHalfY));
            rays.push({ origin, direction: [Math.fround(direction[0]), Math.fround(direction[1]), Math.fround(direction[2])], tMin: 0, tMax: Math.fround(1e30) });
        }
    }
    return rays;
}

function unpackAll(packed: Float32Array): TraceRay[] {
    return Array.from({ length: packed.length / 8 }, (_, index) => unpackRay(packed, index));
}

const yieldToBrowser = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

// Sponza's triangle buffer is over the default 128 MiB binding: ask for what the adapter offers (as the bench pages do).
const limitsAdapter = await navigator.gpu?.requestAdapter();
const requiredLimits = limitsAdapter
    ? { maxStorageBufferBindingSize: limitsAdapter.limits.maxStorageBufferBindingSize, maxBufferSize: limitsAdapter.limits.maxBufferSize }
    : undefined;

await runGpuTest(async ({ device, log }) => {
    const builder = createMeshBvhBuilder({ worker: 'auto' });
    const cache = new Map<MeshGeometry, Promise<MeshBvh>>();
    const build = (geometry: MeshGeometry): Promise<MeshBvh> => {
        let tree = cache.get(geometry);
        if (!tree) cache.set(geometry, (tree = builder.build(geometry)));
        return tree;
    };
    const kernels = new Map<string, TraceBatchKernel>();
    const kernelFor = async (nodeFormat: 'f32' | 'quantized'): Promise<TraceBatchKernel> => {
        let kernel = kernels.get(nodeFormat);
        if (!kernel) kernels.set(nodeFormat, (kernel = await TraceBatchKernel.create(device, { label: `agreement ${nodeFormat}`, nodeFormat })));
        return kernel;
    };

    const groups: { name: string; variants: SceneVariant[]; sets: RaySet[] }[] = [];

    if (SCENE !== 'none') {
        const scene = await loadBenchScene(SCENE, 'manifest');
        const camera = (await findManifest(SCENE)).cameras?.[0];
        assert(camera, `the manifest of "${SCENE}" has no camera bookmark`);
        const geometries = scene.meshes.map((mesh) => ({ positions: mesh.positions, indices: mesh.indices }));
        const instances: TraceSceneInstance[] = scene.instances.map((instance, index) => ({ meshIndex: instance.meshIndex, instanceIndex: index, objectToWorld: instance.objectToWorld, static: true }));
        const merged = await buildTraceScene(geometries, instances, { build });
        const instanced = await buildTraceScene(geometries, instances, { build, mergeStaticInstances: false, nodeFormat: 'f32' });
        const grid = makeProbeGrid(scene.probeBox, Math.max(1, Math.round(AIR_RAYS / 64)));
        const air = unpackAll(makeProbeRays(grid, makeSphereDirections(64, 7)));
        groups.push({
            name: SCENE,
            variants: [
                { name: 'default (merged, quantized)', data: merged, nodeFormat: 'quantized' },
                { name: 'instanced, f32 nodes', data: instanced, nodeFormat: 'f32' },
            ],
            sets: [
                { name: `primary ${WIDTH}x${HEIGHT} from "${camera.name}"`, rays: primaryRays(camera, WIDTH, HEIGHT), identityGate: true },
                { name: `air ${grid.probeCount}x64`, rays: air },
            ],
        });
        log(`${SCENE}: ${scene.meshes.length} meshes, ${scene.instances.length} instances, ${scene.instancedTriangles} triangles`);
    }

    // The trace oracle's shared-edge cases (sharedEdgeRays), plain and through its tilted, mirrored, non-uniform instance.
    for (const transformed of [false, true]) {
        const oracleCase = sharedEdgeCase(SHARED_RAYS, transformed);
        const variants: SceneVariant[] = [];
        for (const nodeFormat of ['f32', 'quantized'] as const) {
            variants.push({ name: `${nodeFormat} nodes`, data: await buildTraceScene(oracleCase.meshes, oracleCase.instances, { build, mergeStaticInstances: false, nodeFormat }), nodeFormat });
        }
        groups.push({ name: oracleCase.name, variants, sets: [{ name: `${SHARED_RAYS} seam rays`, rays: unpackAll(oracleCase.rays) }] });
    }

    // The trace oracle's random cases, default scene options.
    for (const oracleCase of buildOracleCases(FULL_SIZES).filter((entry) => !entry.curated)) {
        const data = await buildTraceScene(oracleCase.meshes, oracleCase.instances, { build, mergeStaticInstances: false });
        groups.push({ name: oracleCase.name, variants: [{ name: 'quantized nodes', data, nodeFormat: 'quantized' }], sets: [{ name: `${oracleCase.rays.length / 8} rays`, rays: unpackAll(oracleCase.rays) }] });
    }

    const report: Record<string, Record<string, Tally & { cpuMs: number }>> = {};
    const failures: string[] = [];
    for (const group of groups) {
        for (const variant of group.variants) {
            const buffers = uploadTraceScene(device, variant.data, `${group.name} ${variant.name}`);
            const kernel = await kernelFor(variant.nodeFormat);
            const tracer = createCpuTracer(variant.data);
            for (const set of group.sets) {
                const gpu: GpuHits = await traceOnGpu(device, kernel, buffers, set.rays);
                const tally = newTally();
                const started = performance.now();
                for (let index = 0; index < set.rays.length; index++) {
                    const ray = set.rays[index];
                    compareRay(tally, index, ray, gpu.closest(index), tracer.closest(ray), gpu.any(index), tracer.any(ray));
                    if (index % 4096 === 4095) await yieldToBrowser();
                }
                const cpuMs = performance.now() - started;
                const key = `${group.name} | ${variant.name} | ${set.name}`;
                (report[group.name] ??= {})[`${variant.name} | ${set.name}`] = { ...tally, cpuMs };
                const t = tally;
                log(
                    `[${key}] ${set.rays.length} rays: exact ${t.exact}, misses ${t.misses}, same triangle ${t.sameTriangleTExact} (t bits equal) + ${t.sameTriangleTDiffers} (t up to ${t.maxTUlps} ulp), ` +
                        `tie ${t.tieOtherTriangle}, near tie ${t.nearTieOtherTriangle}, other triangle ${t.otherTriangle}, hit/miss ${t.hitMiss}, normal bits differ ${t.normalDiffers}, any ${t.anyAgree} agree / ${t.anyDisagree} not`,
                );
                for (const example of t.gateExamples) log(`    gate: ${example}`);
                for (const example of t.examples) log(`    ${example}`);
                if (t.anyDisagree > 0) failures.push(`${key}: any hit disagrees on ${t.anyDisagree} rays`);
                if (t.hitMiss > 0) failures.push(`${key}: closest hit and miss disagree on ${t.hitMiss} rays`);
                if (set.identityGate && t.otherTriangle > 0) failures.push(`${key}: ${t.otherTriangle} primary rays name another triangle more than ${NEAR_TIE_ULPS} ulp away`);
                await yieldToBrowser();
            }
            buffers.destroy();
        }
    }
    builder.dispose();
    for (const kernel of kernels.values()) kernel.destroy();
    if (STRICT) assert(failures.length === 0, failures.join('\n'));
    else if (failures.length > 0) log(`not strict; would fail:\n${failures.join('\n')}`);
    return { scene: SCENE, width: WIDTH, height: HEIGHT, sharedRays: SHARED_RAYS, strict: STRICT, failures, report };
}, { requiredLimits });

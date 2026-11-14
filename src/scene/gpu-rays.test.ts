/**
 * GpuRays in Node: the batch kernel runs as a CPU stand-in (the CPU walk over
 * the bound scene's bytes, writing the WGSL kernel's records), so these check
 * what GpuRays itself does: one source type for an uploaded scene and a
 * SceneTrees, buffers and their growth, the readback, the hit decoding, masks,
 * any hits, picking, and one trace at a time. The kernel's WGSL is checked on
 * GPUs (trace-oracle, trace-inline, the examples' smoke pages).
 */

import { describe, expect, it } from 'vitest';
import { createFakeGpu, type FakeBinding, type FakeBuffer, type FakeKernel } from '../../tests/support/fake-gpu.ts';
import { createCpuTracer, newTraceStats, type CpuHit } from '../trace/cpu-traversal.ts';
import { TOP_HEADER_WORD } from '../trace/layout.ts';
import { buildMeshBvhAsync } from '../trace/mesh-bvh-build.ts';
import { createRandom, randomRays, tessellatedQuad } from '../trace/oracle-fixtures.ts';
import { packRays, RAY_FLOATS, unpackRay } from '../trace/rays.ts';
import { assembleTraceScene, type TraceSceneData } from '../trace/scene-assembly.ts';
import type { TopLevelBvh } from '../trace/top-level.ts';
import { uploadTraceScene } from '../trace/trace-gpu.ts';
import { HIT_RECORD_BYTES, HIT_RECORD_WORD, WALK_COUNTS_BYTES, WALK_COUNTS_WORD } from '../trace/wgsl.ts';
import { GpuRays, traceBuffersOf, type GpuRayHit } from './gpu-rays.ts';
import { SceneTrees } from './scene-trees.ts';

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const translated = (x: number, y: number, z: number) => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];

/** The scene bound in group 0, read as the CPU walk reads it (records, quantized nodes: the defaults). */
function sceneOf(bindings: ReadonlyMap<number, FakeBinding>): TraceSceneData {
    const bytes = (binding: number) => bindings.get(binding)!.buffer.bytes.slice().buffer;
    const top = bytes(2);
    const header = new Uint32Array(top, 0, 4);
    return {
        nodes: bytes(0),
        nodeFormat: 'quantized',
        triangles: bytes(1),
        triangleLeaves: 'records',
        positions: new ArrayBuffer(0),
        positionLayout: 'vec4',
        vertexCount: 0,
        topLevel: { buffer: top, instanceCount: header[TOP_HEADER_WORD.instanceCount], nodeCount: header[TOP_HEADER_WORD.nodeCount] } as TopLevelBvh,
        meshes: [],
        byteLength: 0,
    } as unknown as TraceSceneData;
}

/** `bvhBatch` on the CPU: each ray of group 1 traced over group 0's scene, written as the WGSL kernel writes it. */
function batchStandIn(mode: 'closest' | 'any'): FakeKernel {
    return (bindings, _x, _y, groups) => {
        const tracer = createCpuTracer(sceneOf(bindings));
        const batch = groups.get(1)!;
        const rays = batch.get(0)!;
        const results = batch.get(1)!.buffer;
        const mask = new Uint32Array(batch.get(2)!.buffer.bytes.buffer)[0];
        const packed = new Float32Array(rays.buffer.bytes.buffer, rays.offset, rays.size / 4);
        const floats = new Float32Array(results.bytes.buffer);
        const words = new Uint32Array(results.bytes.buffer);
        const stride = HIT_RECORD_BYTES / 4;
        for (let index = 0; index < packed.length / RAY_FLOATS; index++) {
            const ray = unpackRay(packed, index);
            if (mode === 'any') {
                words[index] = tracer.any(ray, { mask }) ? 1 : 0;
                continue;
            }
            const hit = tracer.closest(ray, { mask });
            const base = index * stride;
            floats[base + HIT_RECORD_WORD.t] = hit ? hit.t : -1;
            if (!hit) continue;
            words[base + HIT_RECORD_WORD.instance] = hit.instanceIndex;
            words[base + HIT_RECORD_WORD.triangle] = hit.triangleIndex;
            words[base + HIT_RECORD_WORD.frontFace] = hit.frontFace ? 1 : 0;
            floats[base + HIT_RECORD_WORD.barycentrics] = hit.u;
            floats[base + HIT_RECORD_WORD.barycentrics + 1] = hit.v;
            words[base + HIT_RECORD_WORD.material] = hit.material;
            floats.set(hit.normal, base + HIT_RECORD_WORD.normal);
        }
    };
}

/** The counted `bvhBatch` on the CPU: the hit, and the CPU walk's own counters where the record has steps and triangle tests. */
function countedStandIn(mode: 'closest' | 'any'): FakeKernel {
    return (bindings, _x, _y, groups) => {
        const tracer = createCpuTracer(sceneOf(bindings));
        const batch = groups.get(1)!;
        const rays = batch.get(0)!;
        const results = batch.get(1)!.buffer;
        const packed = new Float32Array(rays.buffer.bytes.buffer, rays.offset, rays.size / 4);
        const floats = new Float32Array(results.bytes.buffer);
        const words = new Uint32Array(results.bytes.buffer);
        const stride = WALK_COUNTS_BYTES / 4;
        for (let index = 0; index < packed.length / RAY_FLOATS; index++) {
            const stats = newTraceStats();
            const base = index * stride;
            if (mode === 'any') floats[base + WALK_COUNTS_WORD.t] = tracer.any(unpackRay(packed, index), { stats }) ? 1 : -1;
            else {
                const hit = tracer.closest(unpackRay(packed, index), { stats });
                floats[base + WALK_COUNTS_WORD.t] = hit ? hit.t : -1;
                words[base + WALK_COUNTS_WORD.instance] = hit?.instanceIndex ?? 0;
                words[base + WALK_COUNTS_WORD.triangle] = hit?.triangleIndex ?? 0;
            }
            words[base + WALK_COUNTS_WORD.steps] = stats.nodesVisited;
            words[base + WALK_COUNTS_WORD.triangleTests] = stats.trianglesTested;
        }
    };
}

/** A fake GPU whose readbacks complete (the plain fake's never do) and whose batch pipelines run {@link batchStandIn}. */
function gpuForRays() {
    const gpu = createFakeGpu({}, {
        kernels: { 'bvhBatch:closest': batchStandIn('closest'), 'bvhBatch:any': batchStandIn('any'), 'bvhBatch:counted closest': countedStandIn('closest'), 'bvhBatch:counted any': countedStandIn('any') },
    });
    const device = gpu.device as unknown as Record<string, unknown> & GPUDevice;
    const createBuffer = device.createBuffer.bind(device);
    device.createBuffer = (descriptor: GPUBufferDescriptor) => {
        const buffer = createBuffer(descriptor) as unknown as Omit<FakeBuffer, "getMappedRange"> & Record<string, unknown>;
        if (descriptor.usage & 0x1) {
            buffer.mapAsync = () => Promise.resolve();
            buffer.getMappedRange = (offset = 0, size?: number) => (buffer.bytes.buffer as ArrayBuffer).slice(offset, offset + (size ?? buffer.size - offset));
        }
        return buffer as unknown as GPUBuffer;
    };
    // Each batch pipeline's entry point names its mode, so the stand-in knows what to write.
    const createAsync = device.createComputePipelineAsync.bind(device);
    device.createComputePipelineAsync = (descriptor: GPUComputePipelineDescriptor) => {
        const mode = `${descriptor.label?.includes('counted') ? 'counted ' : ''}${descriptor.label?.endsWith(' any') ? 'any' : 'closest'}`;
        return createAsync({ ...descriptor, compute: { ...descriptor.compute, entryPoint: `${descriptor.compute.entryPoint}:${mode}` } });
    };
    return gpu;
}

const key = (hit: GpuRayHit | CpuHit | null) => (hit === null ? 'miss' : 'instanceIndex' in hit ? `${hit.instanceIndex}/${hit.triangleIndex}` : `${hit.instance}/${hit.triangle}`);

describe('GpuRays', () => {
    it('traces an uploaded scene and decodes the records the kernel writes: closest, any, masks, picking', async () => {
        const gpu = gpuForRays();
        const quad = tessellatedQuad(4, 1);
        const bvh = await buildMeshBvhAsync(quad);
        const data = assembleTraceScene(
            [bvh],
            [
                { meshIndex: 0, instanceIndex: 7, objectToWorld: IDENTITY, material: 3, mask: 0x1 },
                { meshIndex: 0, instanceIndex: 9, objectToWorld: translated(0, 0, -2), mask: 0x2 },
            ],
        );
        const buffers = uploadTraceScene(gpu.device, data);
        // An uploaded scene is a TraceSource as it is.
        expect(traceBuffersOf(buffers)).toBe(buffers);
        const rays = await GpuRays.create(gpu.device, buffers);

        const list = randomRays(createRandom(3), 200, [-1, -1, -2, 1, 1, 0], 0.5);
        const packed = packRays(list);
        const closest = await rays.traceClosest(packed);
        const cpu = createCpuTracer(data);
        expect(closest.count).toBe(200);
        expect(closest.gpuMs).toBeNull(); // no timestamp-query on the fake device
        let hits = 0;
        list.forEach((ray, index) => {
            const want = cpu.closest(ray);
            const got = closest.hit(index);
            expect(key(got)).toBe(key(want));
            if (want && got) {
                hits++;
                expect(got.t).toBeCloseTo(want.t, 5);
                expect(got.material).toBe(want.material);
                expect(got.frontFace).toBe(want.frontFace);
                expect(got.barycentrics[0]).toBeCloseTo(want.u, 5);
                expect(got.normal[2]).toBeCloseTo(want.normal[2], 5);
            }
        });
        expect(closest.hits).toBe(hits);
        expect(hits).toBeGreaterThan(20);
        expect(closest.records.byteLength).toBe(200 * HIT_RECORD_BYTES);
        expect(() => closest.hit(200)).toThrow(RangeError);

        const any = await rays.traceAny(packed);
        expect(any.results.length).toBe(200);
        list.forEach((ray, index) => expect(any.blocked(index)).toBe(cpu.any(ray)));

        // Picking: straight down onto instance 7 (material 3); with mask 0x2, through it to instance 9.
        const pick = await rays.pick([0.3, 0.2, 5], [0, 0, -2]);
        expect(pick?.instance).toBe(7);
        expect(pick?.t).toBeCloseTo(5, 5);
        expect(pick?.material).toBe(3);
        expect((await rays.pick([0.3, 0.2, 5], [0, 0, -1], { mask: 0x2 }))?.instance).toBe(9);
        expect(await rays.pick([0.3, 0.2, 5], [0, 0, 1])).toBeNull();
        expect(await rays.pick([0.3, 0.2, 5], [0, 0, -1], { tMax: 4 })).toBeNull();
        await expect(rays.pick([0, 0, 0], [0, 0, 0])).rejects.toThrow(RangeError);

        // The counted walk: its own kernel (compiled at the first call), the same hits, the work per ray.
        const counted = await rays.countWalk(packed);
        list.forEach((ray, index) => {
            const want = cpu.closest(ray);
            const got = counted.counts(index);
            expect(got.hit).toBe(want !== null);
            if (want) expect(`${got.instance}/${got.triangle}`).toBe(key(want));
        });
        expect(counted.maxSteps).toBeGreaterThan(0);
        expect(counted.meanSteps).toBeGreaterThan(0);
        expect(counted.meanSteps).toBeLessThanOrEqual(counted.maxSteps);
        expect(counted.records.length).toBe(200 * (WALK_COUNTS_BYTES / 4));
        const countedAny = await rays.countWalk(packed, { mode: 'any' });
        list.forEach((ray, index) => expect(countedAny.counts(index).hit).toBe(cpu.any(ray)));
        rays.destroy();
        await expect(rays.traceClosest(packed)).rejects.toThrow('destroyed');
    });

    it('takes a SceneTrees as the same source, reading its buffers at each trace (they grow), one trace at a time', async () => {
        const gpu = gpuForRays();
        const trees = await SceneTrees.create(gpu.device, { gpuFirstTrees: false, workers: { worker: 'never' } });
        const frame = () => {
            const encoder = gpu.createEncoder();
            trees.update(encoder);
            gpu.submit(encoder);
        };
        const first = trees.addMesh(tessellatedQuad(2, 1));
        trees.addInstance({ mesh: first, objectToWorld: IDENTITY, instanceIndex: 1 });
        frame();
        await trees.whenRefined();
        frame();
        const rays = await GpuRays.create(gpu.device, trees);
        const down = new Float32Array([0.25, 0.25, 3, 0, 0, 0, -1, 1e30]);
        expect((await rays.traceClosest(down)).hit(0)?.instance).toBe(1);

        // A bigger mesh grows the node and triangle buffers: the next trace binds the new ones.
        const version = trees.version;
        const big = trees.addMesh(tessellatedQuad(64, 1));
        trees.addInstance({ mesh: big, objectToWorld: translated(0, 0, 1), instanceIndex: 2 });
        frame();
        await trees.whenRefined();
        frame();
        expect(trees.version).not.toBe(version);
        // Two traces asked at once: queued, each with its own results.
        expect(rays.pending).toBe(false);
        const both = [rays.traceClosest(down), rays.traceAny(down)];
        expect(rays.pending).toBe(true);
        const [closest, any] = await Promise.all(both);
        expect(rays.pending).toBe(false);
        expect('hit' in closest && closest.hit(0)?.instance).toBe(2);
        expect('blocked' in any && any.blocked(0)).toBe(true);
        await rays.retire();
        trees.dispose();
    });
});

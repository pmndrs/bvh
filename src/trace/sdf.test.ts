/**
 * SDF generation: the CPU reference against analytic fields (a box's signed
 * distance, overlapping boxes by either rule), the voxel sampling, the kernel's
 * WGSL, and `SdfGenerator`'s checks, parameters and slabs on a fake device.
 * The GPU kernel against the CPU reference is tests/gpu/pages/sdf-oracle.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import { createFakeGpu, type FakeBuffer, type FakeEncoder } from '../../tests/support/fake-gpu.ts';
import { closedBox, merge } from '../../tests/support/solids.ts';
import { CpuMeshBvh, insideVotesOf } from '../query/cpu-mesh-bvh.ts';
import { createTinyBvhBuilder, type TinyBvhBuilder } from '../tinybvh.ts';
import { buildMeshBvhWithTinyBvh } from './mesh-bvh-topology.ts';
import { createRandom, randomTriangleSoup } from './oracle-fixtures.ts';
import { INSIDE_TEST_DIRECTIONS } from './query-wgsl.ts';
import { assembleTraceScene } from './scene-assembly.ts';
import { createSdfTexture, generateSdfCpu, SDF_PARAMS_BYTES, SdfGenerator, sdfKernelWgsl, sdfResolution, sdfValueCpu, sdfVoxelCenter, type SdfGrid } from './sdf.ts';
import { IDENTITY_TRANSFORM } from './top-level.ts';
import { uploadTraceScene } from './trace-gpu.ts';
import { traceModuleWgsl } from './wgsl.ts';

let tiny: TinyBvhBuilder;
beforeAll(async () => {
    tiny = await createTinyBvhBuilder();
});

/** The exact signed distance to the box [-h, h] on each axis. */
function boxSdf(p: readonly number[], h: readonly number[]): number {
    const q = p.map((v, k) => Math.abs(v) - h[k]);
    const outside = Math.hypot(...q.map((v) => Math.max(v, 0)));
    return outside + Math.min(Math.max(q[0], q[1], q[2]), 0);
}

describe('the CPU reference (generateSdfCpu)', () => {
    it('gives a closed box its exact signed distance at every voxel centre', () => {
        const bvh = CpuMeshBvh.build(merge(closedBox([-1, -0.5, -0.75], [1, 0.5, 0.75])), tiny);
        const grid: SdfGrid = { min: [-2, -1.5, -1.75], max: [2, 1.5, 1.75], resolution: [10, 7, 9] };
        const field = generateSdfCpu(bvh, grid);
        expect(field).toHaveLength(10 * 7 * 9);
        for (let z = 0; z < 9; z++) {
            for (let y = 0; y < 7; y++) {
                for (let x = 0; x < 10; x++) {
                    const p = sdfVoxelCenter(grid, x, y, z);
                    expect(field[(z * 7 + y) * 10 + x]).toBeCloseTo(boxSdf(p, [1, 0.5, 0.75]), 5);
                }
            }
        }
        // Unsigned: the same magnitudes, never negative.
        const unsigned = generateSdfCpu(bvh, grid, { sign: 'unsigned' });
        expect(Array.from(unsigned)).toEqual(Array.from(field, Math.abs));
        // maxDistance clamps both ways.
        const band = generateSdfCpu(bvh, grid, { maxDistance: 0.25 });
        for (let i = 0; i < field.length; i++) expect(band[i]).toBe(Math.fround(Math.max(-0.25, Math.min(0.25, field[i]))));
    });

    it('signs overlapping shells by the rule asked for, over one tree or several', () => {
        const a = merge(closedBox([-1, -1, -1], [1, 1, 1]));
        const b = merge(closedBox([0, -0.5, -0.5], [2, 0.5, 0.5]));
        const both = CpuMeshBvh.build(merge({ positions: a.positions, indices: a.indices as Uint32Array }, { positions: b.positions, indices: b.indices as Uint32Array }), tiny);
        const separate = [CpuMeshBvh.build(a, tiny), CpuMeshBvh.build(b, tiny)];
        const overlap = [0.5, 0, 0] as const;
        for (const sources of [[both], separate]) {
            expect(sdfValueCpu(sources, overlap)).toBeCloseTo(0.5, 6); // two shells: even crossings, outside
            expect(sdfValueCpu(sources, overlap, { sign: 'winding' })).toBeCloseTo(-0.5, 6);
            expect(sdfValueCpu(sources, [1.5, 0, 0])).toBeCloseTo(-0.5, 6);
        }
    });

    it('signs by the majority of all three rays, though the third is cast only on a split', () => {
        // An open box and a random soup over two trees: the rays often split.
        const box = closedBox([-1, -1, -1], [1, 1, 1]);
        const sources = [CpuMeshBvh.build(merge({ positions: box.positions, indices: box.indices.filter((_, i) => Math.floor(i / 6) !== 1) }), tiny), CpuMeshBvh.build(randomTriangleSoup(createRandom(8), 150, 1.4, 0.6), tiny)];
        const random = createRandom(31);
        for (let i = 0; i < 300; i++) {
            const point = [(random() - 0.5) * 4, (random() - 0.5) * 4, (random() - 0.5) * 4] as [number, number, number];
            const votes = insideVotesOf(point, (origin, direction) => {
                const [x, y] = sources.map((source) => source.rayCrossings(origin, direction));
                return { count: x.count + y.count, winding: x.winding + y.winding };
            });
            const distance = Math.abs(sdfValueCpu(sources, point, { sign: 'unsigned' }));
            expect(sdfValueCpu(sources, point)).toBe(votes.parity >= 2 ? -distance : distance);
            expect(sdfValueCpu(sources, point, { sign: 'winding' })).toBe(votes.winding >= 2 ? -distance : distance);
        }
    });

    it("'first-hit' signs by the side the first hit is on: right for one closed surface, not where shells overlap", () => {
        const bvh = CpuMeshBvh.build(merge(closedBox([-1, -0.5, -0.75], [1, 0.5, 0.75])), tiny);
        const grid: SdfGrid = { min: [-2, -1.5, -1.75], max: [2, 1.5, 1.75], resolution: [10, 7, 9] };
        // Equal everywhere (voxels on the surface may be +0 or -0).
        const plusZero = (field: Float32Array) => Array.from(field, (v) => v + 0);
        expect(plusZero(generateSdfCpu(bvh, grid, { sign: 'first-hit' }))).toEqual(plusZero(generateSdfCpu(bvh, grid)));
        // In box a only, the ray runs on into box b and hits b's front first: outside, though the union is solid there.
        const a = merge(closedBox([-1, -1, -1], [1, 1, 1]));
        const b = merge(closedBox([0, -0.5, -0.5], [2, 0.5, 0.5]));
        const separate = [CpuMeshBvh.build(a, tiny), CpuMeshBvh.build(b, tiny)];
        const inA = [-0.5, -0.2, -0.3] as const;
        expect(sdfValueCpu(separate, inA, { sign: 'winding' })).toBeLessThan(0);
        expect(sdfValueCpu(separate, inA, { sign: 'first-hit' })).toBeGreaterThan(0);
        // Inside both, the first hit is a back face: inside.
        expect(sdfValueCpu(separate, [0.5, 0, 0], { sign: 'first-hit' })).toBeLessThan(0);
        expect(() => generateSdfCpu([{ closestPointToPoint: () => null, rayCrossings: () => ({ count: 0, winding: 0 }) }], grid, { sign: 'first-hit' })).toThrow(/needs sources with intersectRay/);
    });

    it('samples voxel centres, so a texture lookup at (p - min) / (max - min) returns the value at p', () => {
        const grid: SdfGrid = { min: [-1, 0, 2], max: [1, 4, 3], resolution: [4, 2, 1] };
        expect(sdfVoxelCenter(grid, 0, 0, 0)).toEqual([-0.75, 1, 2.5]);
        expect(sdfVoxelCenter(grid, 3, 1, 0)).toEqual([0.75, 3, 2.5]);
        expect(sdfResolution({ ...grid, resolution: 5 })).toEqual([5, 5, 5]);
        expect(() => sdfResolution({ ...grid, resolution: 0 })).toThrow(RangeError);
        expect(() => sdfResolution({ ...grid, resolution: [2, 2.5, 2] })).toThrow(RangeError);
        expect(() => sdfResolution({ min: [0, 0, 0], max: [1, 0, 1], resolution: 2 })).toThrow(/max must be above min/);
        expect(() => generateSdfCpu([], grid, { sign: 'even-odd' as never })).toThrow(/sign must be/);
        expect(() => generateSdfCpu([], grid, { maxDistance: 0 })).toThrow(/maxDistance must be positive/);
    });
});

describe('the kernel (sdfKernelWgsl) and the inside test in the queries module', () => {
    it('declares the inside test with the CPU directions, masked forms only with masks', () => {
        const masked = traceModuleWgsl({ queries: true });
        for (const signature of [
            'fn bvhRayCrossings(origin: vec3f, direction: vec3f) -> BvhRayCrossings',
            'fn bvhRayCrossingsMasked(origin: vec3f, direction: vec3f, mask: u32) -> BvhRayCrossings',
            'fn bvhInsideVotes(point: vec3f) -> vec2u',
            'fn bvhInsideVotesMasked(point: vec3f, mask: u32) -> vec2u',
            'fn bvhIsInside(point: vec3f) -> bool',
            'fn bvhIsInsideMasked(point: vec3f, mask: u32) -> bool',
            'fn bvhIsInsideByRule(point: vec3f, winding: bool) -> bool',
            'fn bvhIsInsideByRuleMasked(point: vec3f, winding: bool, mask: u32) -> bool',
        ]) {
            expect(masked).toContain(signature);
        }
        INSIDE_TEST_DIRECTIONS.forEach((d, i) => {
            // Printed to 9 digits, which reads back as the same f32.
            const literal = masked.match(new RegExp(`BVH_INSIDE_DIRECTION_${i}: vec3f = vec3f\\(([^)]*)\\)`))![1].split(',').map(Number);
            expect(literal.map(Math.fround)).toEqual([...d]);
        });
        const unmasked = traceModuleWgsl({ queries: true, rayMasks: false });
        expect(unmasked).toContain('fn bvhIsInside(point: vec3f) -> bool');
        expect(unmasked).not.toContain('Masked(');
        expect(traceModuleWgsl()).not.toContain('bvhRayCrossings');
    });

    it('writes a storage texture or a buffer, through the masked queries unless masks are compiled out', () => {
        const texture = sdfKernelWgsl();
        expect(texture).toContain('@group(1) @binding(1) var bvhSdfOutput: texture_storage_3d<r32float, write>;');
        expect(texture).toContain('textureStore(bvhSdfOutput, voxel,');
        expect(texture).toContain('bvhClosestPointMasked(point, params.maxDistance, params.mask)');
        // The sign: the early-out majority by the call's rule (the third ray only on a split).
        expect(texture).toContain('bvhIsInsideByRuleMasked(point, params.sign == BVH_SDF_WINDING, params.mask)');
        // 'first-hit': one closest-hit ray along the inside test's first direction.
        expect(texture).toContain('bvhTraceClosestMasked(BvhRay(point, 0.0, BVH_INSIDE_DIRECTION_0, BVH_SDF_RAY_FAR), params.mask)');
        expect(texture).toContain('@compute @workgroup_size(4, 4, 4)\nfn bvhSdfGenerate(');
        const buffer = sdfKernelWgsl({ output: 'buffer', rayMasks: false, nodeFormat: 'f32', triangleLeaves: 'indexed' });
        expect(buffer).toContain('@group(1) @binding(1) var<storage, read_write> bvhSdfOutput: array<f32>;');
        expect(buffer).toContain('bvhClosestPoint(point, params.maxDistance)');
        expect(buffer).not.toContain('Masked(');
        expect(() => sdfKernelWgsl({ output: 'rgba' as never })).toThrow(RangeError);
    });
});

describe('SdfGenerator', () => {
    const scene = () => {
        const tree = buildMeshBvhWithTinyBvh(merge(closedBox([-1, -1, -1], [1, 1, 1])), tiny);
        return assembleTraceScene([tree], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY_TRANSFORM }]);
    };
    const fakeTexture = (width: number, height: number, depth: number, format = 'r32float') => ({ width, height, depthOrArrayLayers: depth, format, dimension: '3d', createView: () => ({}) }) as unknown as GPUTexture;
    const grid: SdfGrid = { min: [-2, -2, -2], max: [2, 2, 2], resolution: [9, 5, 13] };

    it('dispatches 4 x 4 x 4 workgroups over the grid with its parameters in one uniform', async () => {
        const gpu = createFakeGpu();
        const generator = await SdfGenerator.create(gpu.device);
        const buffers = uploadTraceScene(gpu.device, scene());
        const encoder = gpu.createEncoder();
        generator.encode(encoder, buffers, { texture: fakeTexture(9, 5, 13) }, grid, { maxDistance: 1.5, mask: 6, sign: 'winding' });
        const [dispatch] = (encoder as unknown as FakeEncoder).commands.flatMap((c) => ('dispatch' in c ? [c.dispatch] : []));
        expect(dispatch.entryPoint).toBe('bvhSdfGenerate');
        expect([dispatch.workgroupsX, dispatch.workgroupsY]).toEqual([3, 2]);
        const params = gpu.buffers.find((b) => b.label === 'sdf params') as FakeBuffer;
        expect(params.size).toBe(SDF_PARAMS_BYTES);
        const floats = new Float32Array(params.bytes.buffer);
        const words = params.words;
        expect(Array.from(floats.subarray(0, 7))).toEqual([-2, -2, -2, 1.5, Math.fround(4 / 9), 0.8000000119209290, Math.fround(4 / 13)]);
        expect(Array.from(words.subarray(7, 13))).toEqual([2, 9, 5, 13, 6, 0]);
        // The same parameters again reuse the uniform; a slab is another.
        generator.encode(encoder, buffers, { texture: fakeTexture(9, 5, 13) }, grid, { maxDistance: 1.5, mask: 6, sign: 'winding' });
        generator.encode(encoder, buffers, { texture: fakeTexture(9, 5, 13) }, grid, { slices: { start: 4, count: 3 } });
        const all = gpu.buffers.filter((b) => b.label === 'sdf params');
        expect(all).toHaveLength(2);
        const slab = all[1].words;
        expect(slab[12]).toBe(4); // zOffset
        expect(new Float32Array(all[1].bytes.buffer)[3]).toBe(Math.fround(1e18)); // no maxDistance: the closest-point limit
        expect(slab[7]).toBe(1); // parity by default
        generator.encode(encoder, buffers, { texture: fakeTexture(9, 5, 13) }, grid, { sign: 'first-hit' });
        expect(gpu.buffers.filter((b) => b.label === 'sdf params')[2].words[7]).toBe(3);
        generator.destroy();
        expect(gpu.buffers.filter((b) => b.label === 'sdf params').every((b) => b.destroyed)).toBe(true);
    });

    it('refuses a scene, texture or buffer it cannot write, and slices outside the grid', async () => {
        const gpu = createFakeGpu();
        const generator = await SdfGenerator.create(gpu.device, { nodeFormat: 'f32' });
        const quantized = uploadTraceScene(gpu.device, scene());
        const encoder = gpu.createEncoder();
        const texture = { texture: fakeTexture(9, 5, 13) };
        expect(() => generator.encode(encoder, quantized, texture, grid)).toThrow(/reads f32 nodes, the scene holds quantized/);
        const f32 = uploadTraceScene(gpu.device, assembleTraceScene([buildMeshBvhWithTinyBvh(merge(closedBox([-1, -1, -1], [1, 1, 1])), tiny)], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY_TRANSFORM }], {}, { nodeFormat: 'f32' }));
        expect(() => generator.encode(encoder, f32, { texture: fakeTexture(9, 5, 12) }, grid)).toThrow(/the grid needs 9 x 5 x 13/);
        expect(() => generator.encode(encoder, f32, { texture: fakeTexture(9, 5, 13, 'rgba8unorm') }, grid)).toThrow(/must be r32float/);
        expect(() => generator.encode(encoder, f32, { buffer: gpu.device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE }) }, grid)).toThrow(/writes a texture/);
        expect(() => generator.encode(encoder, f32, texture, grid, { slices: { start: 10, count: 4 } })).toThrow(/outside the grid's 13/);
        const buffers = await SdfGenerator.create(gpu.device, { output: 'buffer', nodeFormat: 'f32' });
        const small = gpu.device.createBuffer({ size: 9 * 5 * 13 * 4 - 4, usage: GPUBufferUsage.STORAGE });
        expect(() => buffers.encode(encoder, f32, { buffer: small }, grid)).toThrow(/the grid needs 2340/);
        expect(() => buffers.encode(encoder, f32, texture, grid)).toThrow(/writes a buffer/);
    });

    it('generate cuts the grid into slabs of whole slices, one submit each', async () => {
        const gpu = createFakeGpu();
        const generator = await SdfGenerator.create(gpu.device, { output: 'buffer' });
        const buffers = uploadTraceScene(gpu.device, scene());
        const target = gpu.device.createBuffer({ size: 9 * 5 * 13 * 4, usage: GPUBufferUsage.STORAGE });
        const submitted: number[] = [];
        const submit = gpu.device.queue.submit.bind(gpu.device.queue);
        gpu.device.queue.submit = (commands: Iterable<GPUCommandBuffer>) => {
            for (const command of commands) submitted.push((command as unknown as FakeEncoder).commands.length);
            submit(commands);
        };
        const result = await generator.generate(buffers, { buffer: target }, grid, { voxelsPerSubmit: 9 * 5 * 4 });
        expect(result.submits).toBe(4); // slices 0-3, 4-7, 8-11, 12
        expect(submitted).toEqual([1, 1, 1, 1]);
        const zOffsets = gpu.buffers.filter((b) => b.label === 'sdf params').map((b) => b.words[12]);
        expect(zOffsets).toEqual([0, 4, 8, 12]);
    });

    it('createSdfTexture makes an r32float 3D storage texture that can be sampled', () => {
        const scope = globalThis as unknown as Record<string, unknown>;
        scope.GPUTextureUsage ??= { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 };
        const gpu = createFakeGpu({ maxTextureDimension3D: 64 });
        const texture = createSdfTexture(gpu.device, { resolution: [8, 16, 4] }) as unknown as GPUTextureDescriptor;
        expect(texture.format).toBe('r32float');
        expect(texture.dimension).toBe('3d');
        expect(texture.size).toEqual([8, 16, 4]);
        expect(() => createSdfTexture(gpu.device, { resolution: 65 })).toThrow(/maxTextureDimension3D/);
    });
});

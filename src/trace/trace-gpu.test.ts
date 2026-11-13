import { describe, expect, it } from 'vitest';
import { createFakeGpu, type FakeBuffer, type FakeEncoder } from '../../tests/support/fake-gpu.ts';
import { CLOSEST_POINT_RECORD_BYTES, POINT_QUERY_BYTES } from './query-wgsl.ts';
import { createTinyBvhBuilder } from '../index.ts';
import { buildMeshBvhWithTinyBvh } from './mesh-bvh-topology.ts';
import { tessellatedQuad } from './oracle-fixtures.ts';
import { assembleTraceScene } from './scene-assembly.ts';
import { IDENTITY_TRANSFORM } from './top-level.ts';
import { ClosestPointBatchKernel, traceSourceFormats, uploadTraceScene } from './trace-gpu.ts';

const tinyBvh = await createTinyBvhBuilder();
const scene = assembleTraceScene([buildMeshBvhWithTinyBvh(tessellatedQuad(8, 1), tinyBvh)], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY_TRANSFORM }]);

describe('uploadTraceScene', () => {
    it('copies the three buffers byte for byte and frees them with destroy', () => {
        const gpu = createFakeGpu();
        const buffers = uploadTraceScene(gpu.device, scene);
        const bytes = (buffer: GPUBuffer) => (buffer as unknown as FakeBuffer).bytes;
        expect(bytes(buffers.nodes)).toEqual(new Uint8Array(scene.nodes));
        expect(bytes(buffers.triangles)).toEqual(new Uint8Array(scene.triangles));
        expect(bytes(buffers.topLevel)).toEqual(new Uint8Array(scene.topLevel.buffer));
        expect(buffers.byteLength).toBe(scene.byteLength);
        buffers.destroy();
        expect([buffers.nodes, buffers.triangles, buffers.topLevel].every((buffer) => (buffer as unknown as FakeBuffer).destroyed)).toBe(true);
    });

    it('throws a RangeError naming the limit when a buffer is larger than one binding', () => {
        const gpu = createFakeGpu({ maxStorageBufferBindingSize: scene.triangles.byteLength - 4 });
        expect(() => uploadTraceScene(gpu.device, scene)).toThrow(/triangles.*maxStorageBufferBindingSize/);
        expect(gpu.buffers).toHaveLength(0);
    });
});

describe('ClosestPointBatchKernel', () => {
    it('binds the scene and the batch, and refuses a scene of another format', async () => {
        const gpu = createFakeGpu();
        const kernel = await ClosestPointBatchKernel.create(gpu.device);
        expect([kernel.nodeFormat, kernel.triangleLeaves, kernel.rayMasks]).toEqual(['quantized', 'records', true]);
        const buffers = uploadTraceScene(gpu.device, scene);
        const queries = gpu.device.createBuffer({ size: 4 * POINT_QUERY_BYTES, usage: 0 });
        const results = gpu.device.createBuffer({ size: 4 * CLOSEST_POINT_RECORD_BYTES, usage: 0 });
        const encoder = gpu.createEncoder();
        kernel.encode(encoder, buffers, { queries, results, count: 3, mask: 1 });
        const commands = (encoder as unknown as FakeEncoder).commands;
        expect(commands).toHaveLength(1);
        const dispatch = (commands[0] as unknown as { dispatch: { entryPoint: string; bindings: Map<number, { buffer: unknown }> } }).dispatch;
        expect(dispatch.entryPoint).toBe('bvhClosestPointBatch');
        expect([...dispatch.bindings.values()].map((binding) => binding.buffer)).toEqual([buffers.nodes, buffers.triangles, buffers.topLevel]);
        const f32Scene = assembleTraceScene([buildMeshBvhWithTinyBvh(tessellatedQuad(2, 1), tinyBvh)], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY_TRANSFORM }], {}, { nodeFormat: 'f32' });
        expect(() => kernel.encode(encoder, uploadTraceScene(gpu.device, f32Scene), { queries, results, count: 3 })).toThrow(/ClosestPointBatchKernel: the kernel reads quantized nodes, the scene holds f32/);
        kernel.destroy();
    });
});

describe('formats of buffers handed to a walk (data says what it is)', () => {
    it('uploadTraceScene names every format, and a buffer set that names none is refused', () => {
        const gpu = createFakeGpu();
        const f32Scene = assembleTraceScene([buildMeshBvhWithTinyBvh(tessellatedQuad(2, 1), tinyBvh)], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY_TRANSFORM }], {}, { nodeFormat: 'f32' });
        const uploaded = uploadTraceScene(gpu.device, f32Scene);
        expect([uploaded.nodeFormat, uploaded.triangleLeaves, uploaded.positionLayout]).toEqual(['f32', 'records', 'vec4']);
        expect(uploadTraceScene(gpu.device, scene).nodeFormat).toBe('quantized');
        expect(traceSourceFormats(uploaded, 'test')).toEqual({ nodeFormat: 'f32', triangleLeaves: 'records', positionLayout: 'vec4' });
        expect(() => traceSourceFormats({ triangleLeaves: 'records' }, 'test')).toThrow(/must name their nodeFormat/);
        expect(() => traceSourceFormats({ nodeFormat: 'quantized' }, 'test')).toThrow(/must name their triangleLeaves/);
        expect(() => traceSourceFormats({ nodeFormat: 'quantized', triangleLeaves: 'indexed' }, 'test')).toThrow(/must name their positionLayout/);
        // A records source may leave positionLayout out: it is never read.
        expect(traceSourceFormats({ nodeFormat: 'quantized', triangleLeaves: 'records' }, 'test').positionLayout).toBe('vec4');
    });
});

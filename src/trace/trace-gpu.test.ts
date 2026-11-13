import { describe, expect, it } from 'vitest';
import { createFakeGpu, type FakeBuffer } from '../../tests/support/fake-gpu.ts';
import { createTinyBvhBuilder } from '../index.ts';
import { buildMeshBvhWithTinyBvh } from './mesh-bvh-topology.ts';
import { tessellatedQuad } from './oracle-fixtures.ts';
import { assembleTraceScene } from './scene-assembly.ts';
import { IDENTITY_TRANSFORM } from './top-level.ts';
import { traceSourceFormats, uploadTraceScene } from './trace-gpu.ts';

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

import { describe, expect, it } from 'vitest';
import { createFakeGpu, type FakeBuffer } from '../../tests/support/fake-gpu.ts';
import { createTinyBvhBuilder } from '../index.ts';
import { buildMeshBvhWithTinyBvh } from './mesh-bvh-topology.ts';
import { tessellatedQuad } from './oracle-fixtures.ts';
import { assembleTraceScene } from './scene-assembly.ts';
import { IDENTITY_TRANSFORM } from './top-level.ts';
import { uploadTraceScene } from './trace-gpu.ts';

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

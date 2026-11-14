/**
 * generateSdfTexture in Node, against a stand-in for three's WebGPU backend
 * and a fake device: the generator writes into the GPUTexture three made for
 * the Storage3DTexture (r32float, the grid's size), one generator per format,
 * a texture of the wrong size refused. The field itself is checked on a GPU
 * (tests/gpu/pages/sdf-oracle; the three/sdf example's smoke check).
 */

import { Box3, FloatType, LinearFilter, NearestFilter, RedFormat, Vector3 } from 'three';
import { vec3 } from 'three/tsl';
import { describe, expect, it } from 'vitest';
import { createFakeGpu, type FakeEncoder } from '../../tests/support/fake-gpu.ts';
import { closedBox, merge } from '../../tests/support/solids.ts';
import { createTinyBvhBuilder } from '../tinybvh.ts';
import { buildMeshBvhWithTinyBvh } from '../trace/mesh-bvh-topology.ts';
import { assembleTraceScene } from '../trace/scene-assembly.ts';
import { IDENTITY_TRANSFORM } from '../trace/top-level.ts';
import { uploadTraceScene } from '../trace/trace-gpu.ts';
import { generateSdfTexture } from './sdf.ts';

const tiny = await createTinyBvhBuilder();
const sceneData = assembleTraceScene([buildMeshBvhWithTinyBvh(merge(closedBox([-1, -1, -1], [1, 1, 1])), tiny)], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY_TRANSFORM }]);

function fakeRenderer(features: string[] = ['float32-filterable']) {
    const submitted: FakeEncoder[] = [];
    const gpu = createFakeGpu();
    const submit = gpu.device.queue.submit.bind(gpu.device.queue);
    gpu.device.queue.submit = (commands: Iterable<GPUCommandBuffer>) => {
        for (const command of commands) submitted.push(command as unknown as FakeEncoder);
        submit(commands);
    };
    const data = new WeakMap<object, Record<string, unknown>>();
    const made: { format: string; dimension: string; size: number[] }[] = [];
    const backend = {
        isWebGPUBackend: true,
        device: gpu.device,
        get(object: object) {
            let entry = data.get(object);
            if (!entry) data.set(object, (entry = {}));
            return entry;
        },
    };
    const renderer = {
        backend,
        hasFeature: (name: string) => features.includes(name),
        // three r187 for a Storage3DTexture of RedFormat + FloatType: an r32float 3d texture in backend.get(texture).texture.
        initTexture(texture: { image: { width: number; height: number; depth: number }; format: number; type: number }) {
            const entry = backend.get(texture);
            if (entry.texture) return;
            const format = texture.format === RedFormat && texture.type === FloatType ? 'r32float' : 'rgba8unorm';
            const size = [texture.image.width, texture.image.height, texture.image.depth];
            made.push({ format, dimension: '3d', size });
            entry.texture = { format, dimension: '3d', width: size[0], height: size[1], depthOrArrayLayers: size[2], createView: () => ({ texture }) };
        },
    };
    return { renderer, gpu, submitted, made };
}

describe('generateSdfTexture', () => {
    it('fills the GPUTexture three made for a Storage3DTexture, over the box, in slabs', async () => {
        const { renderer, gpu, submitted, made } = fakeRenderer();
        const buffers = uploadTraceScene(gpu.device, sceneData);
        const box = new Box3(new Vector3(-2, -2, -2), new Vector3(2, 2, 3));
        const sdf = await generateSdfTexture(renderer, { buffers }, { box, resolution: [8, 8, 12], voxelsPerSubmit: 8 * 8 * 4 });
        expect(made).toEqual([{ format: 'r32float', dimension: '3d', size: [8, 8, 12] }]);
        expect(sdf.texture.image).toMatchObject({ width: 8, height: 8, depth: 12 });
        expect([sdf.texture.minFilter, sdf.texture.magFilter]).toEqual([LinearFilter, LinearFilter]);
        expect(sdf.resolution).toEqual([8, 8, 12]);
        expect(sdf.box.equals(box)).toBe(true);
        expect(sdf.box).not.toBe(box);
        expect(submitted).toHaveLength(3);
        for (const encoder of submitted) expect(encoder.commands.flatMap((c) => ('dispatch' in c ? [c.dispatch.entryPoint] : []))).toEqual(['bvhSdfGenerate']);
        // The TSL helpers are nodes.
        expect(sdf.sample(vec3(0, 0, 0)).isNode).toBe(true);

        // Again into the same texture: no new GPU texture, the same generator (no new pipeline).
        const pipelines = gpu.counters.pipelines;
        await generateSdfTexture(renderer, { buffers }, { box, resolution: [8, 8, 12], texture: sdf.texture, sign: 'winding' });
        expect(made).toHaveLength(1);
        expect(gpu.counters.pipelines).toBe(pipelines);
        await expect(generateSdfTexture(renderer, { buffers }, { box, resolution: 9, texture: sdf.texture })).rejects.toThrow(/the texture is 8 x 8 x 12, the field 9 x 9 x 9/);
    });

    it('samples nearest without float32-filterable, and refuses a renderer that is not WebGPU or keeps its textures elsewhere', async () => {
        const { renderer, gpu } = fakeRenderer([]);
        const buffers = uploadTraceScene(gpu.device, sceneData);
        const box = { min: { x: -2, y: -2, z: -2 }, max: { x: 2, y: 2, z: 2 } };
        const sdf = await generateSdfTexture(renderer, { buffers }, { box, resolution: 4 });
        expect(sdf.texture.magFilter).toBe(NearestFilter);
        await expect(generateSdfTexture({ backend: {} }, { buffers }, { box })).rejects.toThrow(/not a WebGPURenderer/);
        const elsewhere = { ...renderer, initTexture() {} };
        await expect(generateSdfTexture(elsewhere, { buffers }, { box, resolution: 4 })).rejects.toThrow(/backend.get\(texture\).texture/);
    });
});

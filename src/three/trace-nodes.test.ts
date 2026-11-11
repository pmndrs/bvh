/**
 * BvhTraceNodes in Node, against a stand-in for three's WebGPU backend: the
 * storage nodes are bound to the source's buffers, rebind only when the
 * version (or a buffer) changes, and retiring never destroys the source's
 * buffers. The shaders themselves are checked on a GPU (tests/gpu/pages/three-trace).
 */

import { describe, expect, it } from 'vitest';
import { BvhTraceNodes } from './trace-nodes.ts';

function fakeRenderer() {
    const data = new WeakMap<object, { buffer?: GPUBuffer }>();
    const created: { destroyed: boolean }[] = [];
    const device = {
        createBuffer: () => {
            const buffer = { destroyed: false, destroy: () => { buffer.destroyed = true; } };
            created.push(buffer);
            return buffer as unknown as GPUBuffer;
        },
    };
    const backend = {
        isWebGPUBackend: true,
        device,
        get(object: object) {
            let entry = data.get(object);
            if (!entry) data.set(object, (entry = {}));
            return entry;
        },
    };
    return { renderer: { backend }, backend, created };
}

const buffer = (label: string) => {
    const made = { label, destroyed: false, destroy: () => { made.destroyed = true; } };
    return made as unknown as GPUBuffer & { destroyed: boolean };
};

describe('BvhTraceNodes', () => {
    it('binds the source buffers, rebinds on a new version, and never destroys them', () => {
        const { renderer, backend } = fakeRenderer();
        const source = { version: 1, buffers: { nodes: buffer('n1'), triangles: buffer('t1'), topLevel: buffer('l1') } };
        const bvh = new BvhTraceNodes(renderer, source);
        const bound = () => [bvh.storage.nodes, bvh.storage.triangles, bvh.storage.topLevel].map((node) => backend.get(node.value as object).buffer);
        expect(bound()).toEqual([source.buffers.nodes, source.buffers.triangles, source.buffers.topLevel]);
        expect(bvh.update()).toBe(false);

        const first = bvh.storage.nodes.value;
        const old = source.buffers;
        source.buffers = { nodes: buffer('n2'), triangles: buffer('t2'), topLevel: old.topLevel };
        source.version = 2;
        expect(bvh.update()).toBe(true);
        expect(bvh.storage.nodes.value).not.toBe(first);
        expect(bound()).toEqual([source.buffers.nodes, source.buffers.triangles, old.topLevel]);
        expect(bvh.update()).toBe(false);
        for (const b of [old.nodes, old.triangles, old.topLevel, source.buffers.nodes]) expect((b as unknown as { destroyed: boolean }).destroyed).toBe(false);

        bvh.dispose();
        for (const b of Object.values(source.buffers)) expect((b as unknown as { destroyed: boolean }).destroyed).toBe(false);
        expect(() => bvh.update()).toThrow(/disposed/);
    });

    it('needs a WebGPU backend, and refuses masked helpers without ray masks', () => {
        expect(() => new BvhTraceNodes({ backend: { isWebGPUBackend: false } }, { buffers: { nodes: buffer('n'), triangles: buffer('t'), topLevel: buffer('l') } })).toThrow(/WebGPURenderer/);
        const { renderer } = fakeRenderer();
        const bvh = new BvhTraceNodes(renderer, { buffers: { nodes: buffer('n'), triangles: buffer('t'), topLevel: buffer('l') } }, { rayMasks: false });
        expect(() => bvh.traceClosestMasked(null, null, 0, 1, null)).toThrow(/rayMasks/);
    });
});

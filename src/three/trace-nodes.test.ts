/**
 * BvhTraceNodes in Node, against a stand-in for three's WebGPU backend: the
 * storage nodes are bound to the source's buffers, rebind only when the
 * version (or a buffer) changes, and retiring never destroys the source's
 * buffers. The shaders themselves are checked on a GPU (tests/gpu/pages/three-trace).
 */

import { describe, expect, it } from 'vitest';
import { installWebGpuGlobals } from '../../tests/support/fake-gpu.ts';
import { BvhTraceNodes } from './trace-nodes.ts';

installWebGpuGlobals();

// three's own attribute bookkeeping and WebGPU attribute utils (0.187), so the
// disposal paths below are three's, not a copy of them.
type ThreeAttributes = { update(attribute: object, type: number): void; has(attribute: object): boolean; dispose(): void };
const attributesModule = 'three/src/renderers/common/Attributes.js';
const utilsModule = 'three/src/renderers/webgpu/utils/WebGPUAttributeUtils.js';
const constantsModule = 'three/src/renderers/common/Constants.js';
const { default: Attributes } = (await import(attributesModule)) as { default: new (backend: object, info: object) => ThreeAttributes };
const { default: WebGPUAttributeUtils } = (await import(utilsModule)) as { default: new (backend: object) => { createAttribute(attribute: object, usage: number): void; destroyAttribute(attribute: object): void } };
const { AttributeType } = (await import(constantsModule)) as { AttributeType: { STORAGE: number } };

function fakeRenderer() {
    let data = new WeakMap<object, Record<string, unknown> & { buffer?: GPUBuffer }>();
    const created: { destroyed: boolean }[] = [];
    const device = {
        createBuffer: () => {
            const buffer = { destroyed: false, destroy: () => { buffer.destroyed = true; }, getMappedRange: () => new ArrayBuffer(16), unmap: () => {} };
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
        delete(object: object) {
            data.delete(object);
        },
        getBufferAttribute: (attribute: object) => attribute,
        createStorageAttribute(attribute: object) {
            utils.createAttribute(attribute, 0x80);
        },
        destroyAttribute(attribute: object) {
            utils.destroyAttribute(attribute);
        },
        forget() {
            data = new WeakMap();
        },
    };
    const utils = new WebGPUAttributeUtils(backend);
    const info = { createStorageAttribute() {}, destroyAttribute() {} };
    const attributes = new Attributes(backend, info);
    const renderer = { backend, _attributes: attributes };
    /** What three does when a shader binds a storage node: make (or take) the attribute's buffer. */
    const use = (attribute: object) => attributes.update(attribute, AttributeType.STORAGE);
    return { renderer, backend, created, attributes, use };
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

    it('binds an indexed source four buffers, the positions too, and rebinds them on a new version', () => {
        const { renderer, backend } = fakeRenderer();
        for (const positionLayout of ['vec4', 'packed'] as const) {
            const source = {
                version: 1,
                triangleLeaves: 'indexed' as const,
                positionLayout,
                buffers: { nodes: buffer('n1'), triangles: buffer('t1'), topLevel: buffer('l1'), positions: buffer('p1') as GPUBuffer | null },
            };
            const bvh = new BvhTraceNodes(renderer, source);
            expect(bvh.triangleLeaves).toBe('indexed');
            const bound = () => [bvh.storage.nodes, bvh.storage.triangles, bvh.storage.topLevel, bvh.storage.positions!].map((node) => backend.get(node.value as object).buffer);
            expect(bound()).toEqual([source.buffers.nodes, source.buffers.triangles, source.buffers.topLevel, source.buffers.positions]);
            source.buffers = { ...source.buffers, positions: buffer('p2') };
            source.version = 2;
            expect(bvh.update()).toBe(true);
            expect(bound()[3]).toBe(source.buffers.positions);
            bvh.dispose();
            expect(() => new BvhTraceNodes(renderer, { ...source, buffers: { ...source.buffers, positions: null } })).toThrow(/positions buffer/);
        }
        const records = new BvhTraceNodes(renderer, { buffers: { nodes: buffer('n'), triangles: buffer('t'), topLevel: buffer('l') } });
        expect(records.storage.positions).toBeNull();
    });

    it('compiles a quantized source in: the nodes read as vec4u, the frame read on instance entry', () => {
        const { renderer, backend } = fakeRenderer();
        for (const triangleLeaves of ['records', 'indexed'] as const) {
            const source = { nodeFormat: 'quantized' as const, triangleLeaves, buffers: { nodes: buffer('n'), triangles: buffer('t'), topLevel: buffer('l'), positions: buffer('p') as GPUBuffer | null } };
            const bvh = new BvhTraceNodes(renderer, source);
            expect(bvh.nodeFormat).toBe('quantized');
            expect((bvh.storage.nodes as unknown as { bufferType?: string; nodeType?: string }).nodeType ?? (bvh.storage.nodes as unknown as { bufferType?: string }).bufferType).toMatch(/uvec4/);
            expect(backend.get(bvh.storage.nodes.value as object).buffer).toBe(source.buffers.nodes);
            bvh.dispose();
        }
        // A source that names no format is the default, quantized (as SceneTrees and assembled scenes are by default).
        expect(new BvhTraceNodes(renderer, { buffers: { nodes: buffer('n'), triangles: buffer('t'), topLevel: buffer('l') } }).nodeFormat).toBe('quantized');
        expect(new BvhTraceNodes(renderer, { nodeFormat: 'f32', buffers: { nodes: buffer('n'), triangles: buffer('t'), topLevel: buffer('l') } }).nodeFormat).toBe('f32');
    });

    it("survives three's disposal: renderer.dispose() (Attributes.dispose) and a stand-in's dispose()", () => {
        const { renderer, backend, created, attributes, use } = fakeRenderer();
        const source = { version: 1, buffers: { nodes: buffer('n1'), triangles: buffer('t1'), topLevel: buffer('l1') } };
        const bvh = new BvhTraceNodes(renderer, source);
        const standIns = () => [bvh.storage.nodes, bvh.storage.triangles, bvh.storage.topLevel].map((node) => node.value as { dispose(): void });
        for (const attribute of standIns()) use(attribute);
        // three took our buffers as made: it created none.
        expect(created).toHaveLength(0);
        expect(standIns().map((attribute) => backend.get(attribute).buffer)).toEqual(Object.values(source.buffers));
        expect(bvh.update()).toBe(false);

        // A stand-in's dispose(): three's onDispose -> Attributes.delete -> backend.destroyAttribute.
        standIns()[0].dispose();
        expect(source.buffers.nodes.destroyed).toBe(false);
        expect(created.length).toBe(1);
        expect(created[0].destroyed).toBe(true);

        // renderer.dispose() runs Attributes.dispose(), which deletes every tracked attribute.
        attributes.dispose();
        for (const b of Object.values(source.buffers)) expect(b.destroyed).toBe(false);
        expect(created.every((throwaway) => throwaway.destroyed)).toBe(true);
    });

    it('throws a clear error when three bound something else to a stand-in it used', () => {
        const { renderer, backend, use } = fakeRenderer();
        const source = { version: 1, buffers: { nodes: buffer('n1'), triangles: buffer('t1'), topLevel: buffer('l1') } };
        const bvh = new BvhTraceNodes(renderer, source);
        // Not used by three yet: nothing to check.
        backend.get(bvh.storage.nodes.value as object).buffer = undefined;
        expect(bvh.update()).toBe(false);
        // A three that looks the buffer up elsewhere: it makes its own (the 16-byte placeholder) and binds that.
        backend.forget();
        use(bvh.storage.nodes.value as object);
        expect(() => bvh.update()).toThrow(/three r\d+ .*unsupported/);
        // Or keeps ours in .buffer but made another one.
        const second = new BvhTraceNodes(renderer, source);
        use(second.storage.triangles.value as object);
        backend.get(second.storage.triangles.value as object).gpuBuffer = renderer.backend.device.createBuffer();
        expect(() => second.update()).toThrow(/unsupported/);
        expect(() => new BvhTraceNodes({ backend: { ...backend, destroyAttribute: undefined }, _attributes: renderer._attributes }, source)).toThrow(/unsupported/);
    });

    it('passes anyHitOrder to the compiled walk', () => {
        const { renderer } = fakeRenderer();
        const source = { buffers: { nodes: buffer('n'), triangles: buffer('t'), topLevel: buffer('l') } };
        const codeOf = (options?: { anyHitOrder?: 'far' | 'near' | 'larger' }) => (new BvhTraceNodes(renderer, source, options).module as unknown as { text: string }).text;
        expect(codeOf()).not.toContain('fn bvhBoxHalfArea(');
        expect(codeOf({ anyHitOrder: 'far' })).toBe(codeOf());
        expect(codeOf({ anyHitOrder: 'larger' })).toContain('fn bvhBoxHalfArea(');
        expect(codeOf({ anyHitOrder: 'near' })).not.toBe(codeOf());
    });

    it('needs a WebGPU backend, and refuses masked helpers without ray masks', () => {
        expect(() => new BvhTraceNodes({ backend: { isWebGPUBackend: false } }, { buffers: { nodes: buffer('n'), triangles: buffer('t'), topLevel: buffer('l') } })).toThrow(/WebGPURenderer/);
        const { renderer } = fakeRenderer();
        const bvh = new BvhTraceNodes(renderer, { buffers: { nodes: buffer('n'), triangles: buffer('t'), topLevel: buffer('l') } }, { rayMasks: false });
        expect(() => bvh.traceClosestMasked(null, null, 0, 1, null)).toThrow(/rayMasks/);
    });
});

import { describe, expect, it } from 'vitest';
import { GPU_LAYOUT_VERSION } from './gpu-layout.ts';
import { checkGpuLayout } from './layout-check.ts';
import { buildLbvhOnCpu } from './lbvh-cpu.ts';
import type { MeshInput } from './mesh-input.ts';
import { deserializeLayout, serializedLayoutBytes, serializeLayout } from './serialize.ts';

/** A small bumpy grid, indexed, so the tree has inner nodes, leaves and real bounds. */
function bumpyGrid(size: number): MeshInput {
    const positions = new Float32Array((size + 1) * (size + 1) * 3);
    for (let z = 0; z <= size; z++) {
        for (let x = 0; x <= size; x++) positions.set([x, Math.sin(x * 0.7) * Math.cos(z * 0.3), z], (z * (size + 1) + x) * 3);
    }
    const indices = new Uint32Array(size * size * 6);
    for (let z = 0; z < size; z++) {
        for (let x = 0; x < size; x++) {
            const corner = z * (size + 1) + x;
            indices.set([corner, corner + size + 1, corner + 1, corner + 1, corner + size + 1, corner + size + 2], (z * size + x) * 6);
        }
    }
    return { positions, indices };
}

const mesh = bumpyGrid(12);
const layout = buildLbvhOnCpu(mesh, { maxLeafPrimitives: 4 });

describe('serializeLayout / deserializeLayout', () => {
    it('round-trips a tree byte for byte', () => {
        const bytes = serializeLayout(layout);
        expect(bytes.byteLength).toBe(serializedLayoutBytes(layout));
        const back = deserializeLayout(bytes);
        expect(back.nodeCount).toBe(layout.nodeCount);
        expect(back.triangleCount).toBe(layout.triangleCount);
        expect(back.maxDepth).toBe(layout.maxDepth);
        expect(back.leafCount).toBe(layout.leafCount);
        expect(Array.from(back.bounds)).toEqual(Array.from(layout.bounds));
        expect(new Uint8Array(back.nodes)).toEqual(new Uint8Array(layout.nodes));
        expect(new Uint8Array(back.triangles)).toEqual(new Uint8Array(layout.triangles));
        expect(Array.from(back.parents)).toEqual(Array.from(layout.parents));
        expect(checkGpuLayout(back, mesh)).toEqual([]);
    });

    it('reads from a view into a larger buffer, and returns copies', () => {
        const bytes = new Uint8Array(serializeLayout(layout));
        const padded = new Uint8Array(bytes.byteLength + 32);
        padded.set(bytes, 16);
        const back = deserializeLayout(padded.subarray(16, 16 + bytes.byteLength));
        expect(new Uint8Array(back.nodes)).toEqual(new Uint8Array(layout.nodes));
        padded.fill(0);
        expect(checkGpuLayout(back, mesh)).toEqual([]);
    });

    it('refuses bytes that are not a tree, are cut short, or come from another layout version', () => {
        const bytes = serializeLayout(layout);
        expect(() => deserializeLayout(new ArrayBuffer(8))).toThrow(/header/);
        expect(() => deserializeLayout(bytes.slice(0, bytes.byteLength - 4))).toThrow(/need/);

        const badMagic = bytes.slice(0);
        new DataView(badMagic).setUint32(0, 0, true);
        expect(() => deserializeLayout(badMagic)).toThrow(/magic/);

        const otherLayout = bytes.slice(0);
        new DataView(otherLayout).setUint32(8, GPU_LAYOUT_VERSION + 1, true);
        expect(() => deserializeLayout(otherLayout)).toThrow(/rebuild/);
    });

    it('refuses trees saved with layout version 1, from before instance masks', () => {
        expect(GPU_LAYOUT_VERSION).toBe(2);
        const oldBytes = serializeLayout(layout);
        new DataView(oldBytes).setUint32(8, 1, true);
        expect(() => deserializeLayout(oldBytes)).toThrow(/GPU layout version 1, this package uses 2; rebuild/);
    });

    it('refuses a layout whose arrays do not match its counts', () => {
        expect(() => serializeLayout({ ...layout, parents: layout.parents.subarray(1) })).toThrow(/parent/);
        expect(() => serializeLayout({ ...layout, nodeCount: layout.nodeCount + 1 })).toThrow(/nodes need/);
    });
});

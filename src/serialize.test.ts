import { describe, expect, it } from 'vitest';
import { GPU_LAYOUT_VERSION } from './gpu-layout.ts';
import { checkGpuLayout } from './layout-check.ts';
import { buildLbvhOnCpu } from './lbvh-cpu.ts';
import type { MeshInput } from './mesh-input.ts';
import { deserializeLayout, isSerializedCurrent, meshGeometryHash, readSerializedHeader, SERIALIZED_FORMAT_VERSION, serializedLayoutBytes, serializeLayout } from './serialize.ts';

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

    it('refuses a scene file, naming the reader for it', () => {
        const bytes = new Uint8Array(serializeLayout(layout));
        new DataView(bytes.buffer).setUint32(0, 0x53564250, true); // 'PBVS'
        expect(() => deserializeLayout(bytes)).toThrow(/scene file.*deserializeTraceScene/);
    });
});

describe('readSerializedHeader / isSerializedCurrent', () => {
    it("reads a tree's header without loading it", () => {
        const bytes = serializeLayout(layout);
        const header = readSerializedHeader(bytes);
        expect(header).toMatchObject({
            kind: 'tree',
            formatVersion: SERIALIZED_FORMAT_VERSION,
            layoutVersion: GPU_LAYOUT_VERSION,
            current: true,
            reason: null,
            triangleLeaves: 'records',
            nodeFormat: 'f32',
            nodeCount: layout.nodeCount,
            triangleCount: layout.triangleCount,
            maxDepth: layout.maxDepth,
            leafCount: layout.leafCount,
            geometryHash: null,
            byteLength: bytes.byteLength,
        });
        expect(header.kind === 'tree' && Array.from(header.bounds)).toEqual(Array.from(layout.bounds));
        expect(isSerializedCurrent(bytes)).toBe(true);
        expect(isSerializedCurrent(new Uint8Array(bytes))).toBe(true);
    });

    it('says why a file is not current, and throws only on bytes that are no file of ours', () => {
        const stale = serializeLayout(layout);
        new DataView(stale).setUint32(8, 1, true);
        expect(readSerializedHeader(stale)).toMatchObject({ kind: 'tree', layoutVersion: 1, current: false, reason: expect.stringMatching(/GPU layout version 1, this package uses 2/) });
        expect(isSerializedCurrent(stale)).toBe(false);

        const newer = serializeLayout(layout);
        new DataView(newer).setUint32(4, SERIALIZED_FORMAT_VERSION + 1, true);
        expect(readSerializedHeader(newer)).toMatchObject({ current: false, reason: expect.stringMatching(/byte format 2, this package reads 1/) });

        const unknownNodes = serializeLayout(layout);
        new DataView(unknownNodes).setUint32(28, 2 << 8, true);
        expect(readSerializedHeader(unknownNodes)).toMatchObject({ current: false, nodeFormat: null, reason: expect.stringMatching(/node format 2/) });

        expect(() => readSerializedHeader(new ArrayBuffer(64))).toThrow(/bad magic/);
        expect(() => readSerializedHeader(new ArrayBuffer(10))).toThrow(/header/);
        expect(isSerializedCurrent(new ArrayBuffer(10))).toBe(false);
    });
});

describe('a tree written with its geometry', () => {
    it("records the geometry's hash in words 14-15 and changes nothing else", () => {
        const plain = new Uint8Array(serializeLayout(layout));
        const withGeometry = new Uint8Array(serializeLayout(layout, { geometry: mesh }));
        const hash = meshGeometryHash(mesh);
        expect(readSerializedHeader(withGeometry)).toMatchObject({ kind: 'tree', geometryHash: hash, current: true });
        expect(Array.from(new Uint32Array(withGeometry.buffer, 56, 2))).toEqual(hash);
        expect(withGeometry.subarray(64)).toEqual(plain.subarray(64));
        expect(withGeometry.subarray(0, 56)).toEqual(plain.subarray(0, 56));
        // Readers that ignore words 14-15 read the same tree.
        expect(new Uint8Array(deserializeLayout(withGeometry).nodes)).toEqual(new Uint8Array(layout.nodes));
    });

    it('loads next to its own geometry and refuses other geometry', () => {
        const bytes = serializeLayout(layout, { geometry: mesh });
        expect(deserializeLayout(bytes, { geometry: mesh }).triangleCount).toBe(layout.triangleCount);
        // 16-bit indices of the same values hash alike.
        expect(() => deserializeLayout(bytes, { geometry: { positions: mesh.positions, indices: Uint16Array.from(mesh.indices!) } })).not.toThrow();

        const moved = { positions: mesh.positions.map((value, index) => (index === 4 ? value + 0.5 : value)), indices: mesh.indices };
        expect(() => deserializeLayout(bytes, { geometry: moved })).toThrow(/baked from other geometry/);
        expect(() => deserializeLayout(bytes, { geometry: bumpyGrid(11) })).toThrow(/the tree has 288 triangles, the geometry 242/);
        // Without a hash in the file, the triangle count is still checked.
        expect(() => deserializeLayout(serializeLayout(layout), { geometry: bumpyGrid(11) })).toThrow(/other geometry/);
        expect(() => serializeLayout(layout, { geometry: bumpyGrid(11) })).toThrow(/other geometry/);
    });

    it("checks an indexed tree's vertex indices against the geometry", () => {
        const indexed = buildLbvhOnCpu(mesh, { maxLeafPrimitives: 4, triangleLeaves: 'indexed' });
        const bytes = serializeLayout(indexed);
        const fewer = { positions: mesh.positions.subarray(0, mesh.positions.length - 30), indices: mesh.indices };
        expect(() => deserializeLayout(bytes, { geometry: fewer })).toThrow(/uses a vertex past the geometry's/);
        expect(deserializeLayout(bytes, { geometry: mesh }).triangleLeaves).toBe('indexed');
    });

    it('hashes differ for different meshes and never read as "none"', () => {
        const a = meshGeometryHash(mesh);
        const b = meshGeometryHash(bumpyGrid(13));
        const c = meshGeometryHash({ positions: mesh.positions });
        expect(a).not.toEqual(b);
        expect(a).not.toEqual(c);
        for (const hash of [a, b, c]) expect(hash[0] !== 0 || hash[1] !== 0).toBe(true);
    });
});

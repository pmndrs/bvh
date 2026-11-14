/**
 * tinybvh's other layout formats written inside the WASM (`tbvh_write_layout_as`):
 * indexed triangles and quantized nodes must be, byte for byte, what the
 * JavaScript conversions (`indexedFromRecords`, `quantizeNodes`) make from the
 * default layout of the same build. And a builder told not to use them
 * (`formatsInWasm: false`, the path an older .wasm takes) gives the same bytes.
 */

import { describe, expect, it } from 'vitest';
import { createTinyBvhBuilder, type TinyBvhBuilder } from './tinybvh.ts';
import type { GpuLayoutMeshBvh } from './gpu-layout.ts';
import type { MeshInput } from './mesh-input.ts';
import { indexedFromRecords } from './indexed-leaves.ts';
import { quantizeNodes } from './quantized-nodes.ts';
import { tessellatedQuad } from './trace/oracle-fixtures.ts';

function random(seed: number): () => number {
    let state = seed >>> 0;
    return () => (state = (state * 1664525 + 1013904223) >>> 0) / 2 ** 32;
}

/** A triangle soup around `center`, `x, y, z` per vertex, three vertices per triangle. */
function soup(count: number, seed: number, center = [0, 0, 0], extent = 10, flat = false): Float32Array {
    const next = random(seed);
    const positions = new Float32Array(count * 9);
    for (let triangle = 0; triangle < count; triangle++) {
        const middle = center.map((value) => value + (next() * 2 - 1) * extent);
        for (let corner = 0; corner < 3; corner++) {
            for (let axis = 0; axis < 3; axis++) {
                positions[triangle * 9 + corner * 3 + axis] = flat && axis === 1 ? center[1] : middle[axis] + next() - 0.5;
            }
        }
    }
    return positions;
}

/** A soup as an indexed mesh: its vertices shuffled, plus unused ones. */
function indexedSoup(count: number, seed: number): MeshInput {
    const positions = soup(count, seed);
    const vertexCount = count * 3;
    const next = random(seed + 1);
    const order = Array.from({ length: vertexCount + 5 }, (_, i) => i).sort(() => next() - 0.5);
    const shuffled = new Float32Array((vertexCount + 5) * 3);
    const indices = new Uint32Array(vertexCount);
    order.forEach((from, to) => {
        if (from < vertexCount) {
            shuffled.set(positions.subarray(from * 3, from * 3 + 3), to * 3);
            indices[from] = to;
        } else {
            shuffled.set([1, 2, 3], to * 3);
        }
    });
    return { positions: shuffled, indices };
}

function flags(count: number, seed: number): Uint32Array {
    const next = random(seed);
    return Uint32Array.from({ length: count }, () => (next() < 0.3 ? 1 : 0));
}

const MESHES: [string, MeshInput][] = [
    ['one triangle', { positions: soup(1, 1) }],
    ['two triangles', { positions: soup(2, 2) }],
    ['a soup', { positions: soup(3000, 3) }],
    ['an indexed soup with unused vertices', indexedSoup(2000, 4)],
    ['flags', { positions: soup(500, 5), triangleFlags: flags(500, 5) }],
    ['materials (bits 16-31 stay with records)', { positions: soup(200, 6), triangleMaterials: Uint32Array.from({ length: 200 }, (_, i) => i % 7) }],
    ['far from the origin', { positions: soup(800, 7, [123456, -98765, 4321], 3) }],
    ['tiny', { positions: soup(300, 8, [0.001, 0.002, -0.003], 1e-4) }],
    ['huge', { positions: soup(300, 9, [0, 0, 0], 1e7) }],
    ['flat', { positions: soup(600, 10, [5, 2, -1], 10, true) }],
    ['a tessellated quad', tessellatedQuad(24, 3)],
];

/** The first byte where two buffers differ, -1 if none (lengths count: a shorter one differs at its end). */
function firstDifference(a: ArrayBuffer, b: ArrayBuffer): number {
    const x = new Uint8Array(a);
    const y = new Uint8Array(b);
    for (let i = 0; i < Math.min(x.length, y.length); i++) if (x[i] !== y[i]) return i;
    return x.length === y.length ? -1 : Math.min(x.length, y.length);
}

function expectSameLayout(actual: GpuLayoutMeshBvh, expected: GpuLayoutMeshBvh): void {
    expect(actual.nodeFormat ?? 'f32').toBe(expected.nodeFormat ?? 'f32');
    expect(actual.triangleLeaves ?? 'records').toBe(expected.triangleLeaves ?? 'records');
    expect(firstDifference(actual.nodes, expected.nodes)).toBe(-1);
    expect(firstDifference(actual.triangles, expected.triangles)).toBe(-1);
    expect(actual.parents).toEqual(expected.parents);
    expect([actual.nodeCount, actual.triangleCount, actual.maxDepth, actual.leafCount]).toEqual([expected.nodeCount, expected.triangleCount, expected.maxDepth, expected.leafCount]);
    expect(actual.bounds).toEqual(expected.bounds);
}

describe('tinybvh writes indexed triangles and quantized nodes inside the WASM', () => {
    let inWasm: TinyBvhBuilder;
    let inJavaScript: TinyBvhBuilder;

    it('has tbvh_write_layout_as, and honours formatsInWasm: false', async () => {
        inWasm = await createTinyBvhBuilder();
        inJavaScript = await createTinyBvhBuilder({ formatsInWasm: false });
        expect(inWasm.formatsInWasm).toBe(true);
        expect(inJavaScript.formatsInWasm).toBe(false);
    });

    for (const [name, mesh] of MESHES) {
        it(`gives the JavaScript conversions' bytes: ${name}`, () => {
            for (const maxLeafPrimitives of [1, 4, 8]) {
                const records = inWasm.buildLayout(mesh, { maxLeafPrimitives }).layout;
                const indexed = indexedFromRecords(records, mesh);
                const expected: Record<string, GpuLayoutMeshBvh> = {
                    indexed,
                    quantized: quantizeNodes(records),
                    both: quantizeNodes(indexed),
                };
                const options = {
                    indexed: { triangleLeaves: 'indexed' },
                    quantized: { nodeFormat: 'quantized' },
                    both: { triangleLeaves: 'indexed', nodeFormat: 'quantized' },
                } as const;
                for (const key of ['indexed', 'quantized', 'both'] as const) {
                    expectSameLayout(inWasm.buildLayout(mesh, { maxLeafPrimitives, ...options[key] }).layout, expected[key]);
                    // The JavaScript path (an older .wasm's) gives the same bytes.
                    expectSameLayout(inJavaScript.buildLayout(mesh, { maxLeafPrimitives, ...options[key] }).layout, expected[key]);
                }
                // The default layout is unchanged by the other path.
                expectSameLayout(inJavaScript.buildLayout(mesh, { maxLeafPrimitives }).layout, records);
            }
        });
    }

    it('does the same with the H-PLOC builder', () => {
        const mesh = { positions: soup(400, 11) };
        const records = inWasm.buildLayout(mesh, { builder: 'hploc', maxLeafPrimitives: 1 }).layout;
        const layout = inWasm.buildLayout(mesh, { builder: 'hploc', maxLeafPrimitives: 1, triangleLeaves: 'indexed', nodeFormat: 'quantized' }).layout;
        expectSameLayout(layout, quantizeNodes(indexedFromRecords(records, mesh)));
    });

    it('still refuses bad input the same way', () => {
        const mesh = { positions: soup(4, 12) };
        expect(() => inWasm.buildLayout({ positions: mesh.positions, indices: Uint32Array.of(0, 1, 2, 0, 1, 99) }, { triangleLeaves: 'indexed' })).toThrow(/index 5 is 99/);
        expect(() => inWasm.buildLayout(mesh, { nodeFormat: 'packed' as never })).toThrow(/nodeFormat/);
        expect(() => inWasm.buildLayout(mesh, { triangleLeaves: 'packed' as never })).toThrow(/triangleLeaves/);
    });
});

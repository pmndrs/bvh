import { GPU_LAYOUT_VERSION } from '../index.ts';
import { describe, expect, it } from 'vitest';
import { TRACER_GPU_LAYOUT_VERSION } from './layout.ts';
import { batchEntryWgsl, batchKernelWgsl, HIT_RECORD_BYTES, HIT_RECORD_COMPACT_BYTES, HIT_RECORD_COMPACT_WORD, HIT_RECORD_WORD, hitRecordBytes } from './wgsl.ts';

/** Size and alignment of the WGSL types a hit record uses (WGSL spec, "Alignment and size"). */
const TYPE_LAYOUT: Record<string, { size: number; align: number }> = {
    f32: { size: 4, align: 4 },
    u32: { size: 4, align: 4 },
    vec2f: { size: 8, align: 8 },
};

/** Byte offset of every member of a WGSL struct, and the struct's size, by the WGSL layout rules. */
function structLayout(wgsl: string, name: string): { offsets: Record<string, number>; size: number } {
    const start = wgsl.indexOf(`struct ${name} {`);
    const body = start < 0 ? undefined : wgsl.slice(wgsl.indexOf('{', start) + 1, wgsl.indexOf('}', start));
    if (!body) throw new Error(`struct ${name} not found`);
    const offsets: Record<string, number> = {};
    let offset = 0;
    let structAlign = 1;
    for (const match of body.matchAll(/(\w+)\s*:\s*(\w+)/g)) {
        const [, member, type] = match;
        const layout = TYPE_LAYOUT[type];
        if (!layout) throw new Error(`no layout for ${type}`);
        offset = Math.ceil(offset / layout.align) * layout.align;
        offsets[member] = offset;
        offset += layout.size;
        structAlign = Math.max(structAlign, layout.align);
    }
    return { offsets, size: Math.ceil(offset / structAlign) * structAlign };
}

describe('BvhHitRecord', () => {
    it('has the layout the contract documents and the readers use, with and without hitNormal', () => {
        const layout = structLayout(batchEntryWgsl('closest', 1), 'BvhHitRecord');
        expect(layout.size).toBe(HIT_RECORD_BYTES);
        expect(HIT_RECORD_BYTES).toBe(40);
        expect(layout.offsets).toEqual({
            t: HIT_RECORD_WORD.t * 4,
            instance: HIT_RECORD_WORD.instance * 4,
            triangle: HIT_RECORD_WORD.triangle * 4,
            frontFace: HIT_RECORD_WORD.frontFace * 4,
            barycentrics: HIT_RECORD_WORD.barycentrics * 4,
            material: HIT_RECORD_WORD.material * 4,
            normalX: HIT_RECORD_WORD.normal * 4,
            normalY: (HIT_RECORD_WORD.normal + 1) * 4,
            normalZ: (HIT_RECORD_WORD.normal + 2) * 4,
        });
        const compactKernel = batchKernelWgsl('closest', { hitNormal: false });
        expect(compactKernel).not.toContain('struct BvhHitRecord {');
        const compact = structLayout(compactKernel, 'BvhHitRecordCompact');
        expect(compact.size).toBe(HIT_RECORD_COMPACT_BYTES);
        expect(HIT_RECORD_COMPACT_BYTES).toBe(24);
        expect(compact.offsets).toEqual(Object.fromEntries(Object.entries(HIT_RECORD_COMPACT_WORD).map(([name, word]) => [name, word * 4])));
        expect([hitRecordBytes(), hitRecordBytes(true), hitRecordBytes(false)]).toEqual([40, 40, 24]);
    });
});

describe('the GPU layout version', () => {
    it("is the one the tracer's WGSL was written against", () => {
        // If this fails, @pmndrs/bvh changed its layout: review layout.ts and the walk in wgsl.ts, then bump TRACER_GPU_LAYOUT_VERSION.
        expect(GPU_LAYOUT_VERSION).toBe(TRACER_GPU_LAYOUT_VERSION);
    });
});

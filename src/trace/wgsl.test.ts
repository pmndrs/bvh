import { GPU_LAYOUT_VERSION } from '../index.ts';
import { describe, expect, it } from 'vitest';
import { TRACER_GPU_LAYOUT_VERSION } from '../gpu-layout.ts';
import { batchEntryWgsl, batchKernelWgsl, closestPointKernelWgsl, HIT_RECORD_BYTES, HIT_RECORD_COMPACT_BYTES, HIT_RECORD_COMPACT_WORD, HIT_RECORD_WORD, hitRecordBytes, traceModuleWgsl, triangleGeometryWgsl } from './wgsl.ts';
import { CLOSEST_POINT_RECORD_BYTES, CLOSEST_POINT_RECORD_WORD, POINT_QUERY_BYTES } from './query-wgsl.ts';

/** Size and alignment of the WGSL types a hit record uses (WGSL spec, "Alignment and size"). */
const TYPE_LAYOUT: Record<string, { size: number; align: number }> = {
    f32: { size: 4, align: 4 },
    u32: { size: 4, align: 4 },
    vec2f: { size: 8, align: 8 },
    vec3f: { size: 12, align: 16 },
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

describe('the spatial queries (queries, shapecast) and the triangle helpers', () => {
    /** The text of one top-level function. */
    const fnText = (module: string, name: string): string => {
        const start = module.indexOf(`\nfn ${name}(`);
        if (start < 0) throw new Error(`fn ${name} not found`);
        return module.slice(start, module.indexOf('\n}\n', start) + 3);
    };

    it('leave the ray walk as it is: no query code unless asked, and the walk text the same either way', () => {
        const plain = traceModuleWgsl();
        expect(plain).not.toContain('bvhClosestPoint(');
        expect(plain).not.toContain('BvhShapecastBounds');
        const withQueries = traceModuleWgsl({ queries: true, shapecast: '' });
        for (const name of ['bvhWalkClosest', 'bvhWalkAny', 'bvhFinishClosest', 'bvhIntersectWatertight', 'bvhLoadMeshNode', 'bvhLoadTriangle']) {
            expect(fnText(withQueries, name)).toBe(fnText(plain, name));
        }
        // The helpers are in every module, and they are the standalone geometry's text.
        expect(plain).toContain(triangleGeometryWgsl());
        expect(triangleGeometryWgsl()).not.toMatch(/@binding/);
        expect(triangleGeometryWgsl()).toContain('fn bvhIntersectRayTriangle(ray: BvhRay, a: vec3f, b: vec3f, c: vec3f) -> BvhTriangleHit');
        expect(triangleGeometryWgsl()).toContain('fn bvhClosestPointToTriangle(point: vec3f, a: vec3f, b: vec3f, c: vec3f) -> BvhTrianglePoint');
        // The helper calls the walk's own test (one copy of the math).
        expect(fnText(plain, 'bvhIntersectRayTriangle')).toContain('bvhIntersectWatertight(');
    });

    it('declare the entry points for the options: masked forms only with ray masks, the shapecast walk only with hooks', () => {
        const masked = traceModuleWgsl({ queries: true });
        for (const signature of [
            'fn bvhClosestPoint(point: vec3f, maxDistance: f32) -> BvhClosestHit',
            'fn bvhClosestPointMasked(point: vec3f, maxDistance: f32, mask: u32) -> BvhClosestHit',
            'fn bvhIntersectsSphere(center: vec3f, radius: f32) -> bool',
            'fn bvhIntersectsSphereMasked(center: vec3f, radius: f32, mask: u32) -> bool',
            'fn bvhIntersectsBox(boxMin: vec3f, boxMax: vec3f) -> bool',
            'fn bvhIntersectsOrientedBox(center: vec3f, halfSize: vec3f, rotation: mat3x3f) -> bool',
        ]) {
            expect(masked).toContain(signature);
        }
        expect(masked).not.toContain('fn bvhShapecast(');
        const unmasked = traceModuleWgsl({ queries: true, rayMasks: false, shapecast: 'fn bvhShapecastBounds(bounds: BvhShapecastBounds) -> f32 { return 0.0; }' });
        expect(unmasked).toContain('fn bvhClosestPoint(point: vec3f, maxDistance: f32) -> BvhClosestHit');
        expect(unmasked).not.toContain('Masked(');
        expect(unmasked).not.toContain('bvhInstanceMask(instance) & mask');
        expect(unmasked).toContain('fn bvhShapecast() -> bool');
        expect(unmasked).toContain('fn bvhShapecastBounds(bounds: BvhShapecastBounds) -> f32 { return 0.0; }');
        // Shapecast alone: its walk and the shared types, not the library's queries.
        const shapecastOnly = traceModuleWgsl({ shapecast: '' });
        expect(shapecastOnly).toContain('fn bvhShapecastMasked(mask: u32) -> bool');
        expect(shapecastOnly).not.toContain('fn bvhClosestPoint(');
        // Every format compiles the same query text around its own loaders.
        expect(traceModuleWgsl({ queries: true, nodeFormat: 'f32', triangleLeaves: 'indexed', positionLayout: 'packed' })).toContain('fn bvhClosestPointWalk(mask: u32) -> bool');
    });

    it('BvhClosestPointRecord has the layout the readers use', () => {
        const layout = structLayout(closestPointKernelWgsl(), 'BvhClosestPointRecord');
        expect(layout.size).toBe(CLOSEST_POINT_RECORD_BYTES);
        expect(layout.offsets).toEqual({
            distance: CLOSEST_POINT_RECORD_WORD.distance * 4,
            instance: CLOSEST_POINT_RECORD_WORD.instance * 4,
            triangle: CLOSEST_POINT_RECORD_WORD.triangle * 4,
            material: CLOSEST_POINT_RECORD_WORD.material * 4,
            barycentrics: CLOSEST_POINT_RECORD_WORD.barycentrics * 4,
            pointX: CLOSEST_POINT_RECORD_WORD.point * 4,
            pointY: (CLOSEST_POINT_RECORD_WORD.point + 1) * 4,
            pointZ: (CLOSEST_POINT_RECORD_WORD.point + 2) * 4,
            normalX: CLOSEST_POINT_RECORD_WORD.normal * 4,
            normalY: (CLOSEST_POINT_RECORD_WORD.normal + 1) * 4,
            normalZ: (CLOSEST_POINT_RECORD_WORD.normal + 2) * 4,
        });
        expect(structLayout(closestPointKernelWgsl(), 'BvhPointQuery').size).toBe(POINT_QUERY_BYTES);
        expect(closestPointKernelWgsl({ rayMasks: false })).not.toContain('@group(1) @binding(2)');
        expect(closestPointKernelWgsl()).toContain('@group(1) @binding(2) var<uniform> bvhClosestPointParams');
    });
});

describe('the walk loops (pmndrs/bvh#3)', () => {
    /** Every walk the module can be built with: binary in each node format and decode, the wide walks, the queries and the counted walk. */
    const variants: Parameters<typeof traceModuleWgsl>[0][] = [
        {},
        { rayMasks: false, anyHitOrder: 'near' },
        { anyHitOrder: 'larger' },
        { nodeFormat: 'quantized' },
        { nodeFormat: 'quantized', quantizedDecode: 'folded' },
        { triangleLeaves: 'indexed' },
        { nodeFormat: 'bvh4' },
        { nodeFormat: 'bvh4', wideWalk: 'single' },
        { nodeFormat: 'bvh4', wideWalk: 'single', rayMasks: false, wideDecode: 'folded' },
        { nodeFormat: 'cwbvh8' },
        { nodeFormat: 'cwbvh8', wideStack: 'workgroup', wideDecode: 'folded' },
        { queries: true },
        { queries: true, nodeFormat: 'f32', rayMasks: false },
        { counters: true },
        { counters: true, rayMasks: false, nodeFormat: 'quantized', quantizedDecode: 'folded' },
    ];

    it('has no loop Tint must count: every loop is a for loop with a literal bound', () => {
        for (const options of variants) {
            const wgsl = traceModuleWgsl(options).replace(/\/\/[^\n]*/g, '');
            expect(wgsl, JSON.stringify(options)).not.toMatch(/\bloop\s*\{/);
            expect(wgsl, JSON.stringify(options)).not.toMatch(/\bwhile\b/);
            // Each for loop declares its index, tests it against the literal first and steps it by ++ (bounded-loop.ts).
            for (const match of wgsl.matchAll(/\bfor\s*\(([^)]*)\)/g)) {
                const head = match[1];
                if (/^var (\w+) = 0u; \1 < 0xffffffffu; \1\+\+$/.test(head)) continue;
                // The leaf loops over a runtime triangle count are left counted (they cost nothing measurable).
                expect(head, `${JSON.stringify(options)}: ${head}`).toMatch(/^var i = 0u; i < (count|\w+); i \+= 1u$/);
            }
        }
    });

    it('ends each walk function that used to end in its loop with a return', () => {
        const wgsl = traceModuleWgsl({});
        expect(wgsl).not.toContain("No return here: the loop above only exits through 'return'.");
        expect(wgsl).toContain('    return bvhFinishClosest(result, hitRecord, hitSlot);\n}');
    });
});

describe('the GPU layout version', () => {
    it("is the one the tracer's WGSL was written against", () => {
        // If this fails, @pmndrs/bvh changed its layout: review gpu-layout.ts and the walk in wgsl.ts, then bump TRACER_GPU_LAYOUT_VERSION.
        expect(GPU_LAYOUT_VERSION).toBe(TRACER_GPU_LAYOUT_VERSION);
    });
});

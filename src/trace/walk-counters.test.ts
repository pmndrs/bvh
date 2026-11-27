/**
 * The counted walk (`counters: true`): off, the module is the same text;
 * on, it only appends (so every declaration of the walk is the same text, in
 * the same place), and each counted function is the production walk with
 * counters added and nothing else. The counted kernel's hits against the
 * production kernel's are checked on GPUs (trace-oracle, the inspector).
 */

import { describe, expect, it } from 'vitest';
import { batchKernelWgsl, traceModuleWgsl, type TraceModuleOptions } from './wgsl.ts';
import { WALK_COUNTS_BYTES, WALK_COUNTS_WORD } from './walk-counters.ts';

const VARIANTS: TraceModuleOptions[] = [
    {},
    { rayMasks: false },
    { nodeFormat: 'f32' },
    { triangleLeaves: 'indexed', positionLayout: 'packed' },
    { anyHitOrder: 'larger' },
    { anyHitOrder: 'near', rayMasks: false },
    { queries: true },
    { triangleFilter: null },
];

/** The text of one top-level function. */
function fnText(module: string, name: string): string {
    const start = module.indexOf(`\nfn ${name}(`);
    if (start < 0) throw new Error(`fn ${name} not found`);
    return module.slice(start + 1, module.indexOf('\n}\n', start) + 2);
}

/** A counted walk with its counters taken out again: renamed back, the counter lines and finishers removed. */
function withoutCounters(counted: string): string {
    return counted
        .replace(/fn (bvhWalk\w+)Counted\(([^)]*)\) -> BvhWalkCounts \{/, (_, name: string, parameters: string) => `fn ${name}(${parameters}) -> ${name === 'bvhWalkClosest' ? 'BvhHit' : 'bool'} {`)
        .replace(/\n {4}var counts = BvhWalkCounts\(\);\n {4}var hitInThisInstance = false;/, '')
        .replace(/\n +counts\.\w+ \+= 1u;/g, '')
        .replace(/\n +hitInThisInstance = false;/g, '')
        .replace(/\n +if \(!hitInThisInstance\) \{ counts\.instancesWithHit \+= 1u; hitInThisInstance = true; \}/g, '')
        .replace(/bvhCountedFinishClosest\(counts, (bvhFinishClosest\(result, hitRecord, hitSlot\))\)/g, '$1')
        .replace('bvhCountedFinishClosest(counts, result)', 'result')
        .replace(/bvhCountedFinishAny\(counts, (true|false)\)/g, '$1');
}

/** Size and alignment of the WGSL types the record uses. */
function structLayout(wgsl: string, name: string): { offsets: Record<string, number>; size: number } {
    const start = wgsl.indexOf(`struct ${name} {`);
    const body = wgsl.slice(wgsl.indexOf('{', start) + 1, wgsl.indexOf('}', start));
    const offsets: Record<string, number> = {};
    let offset = 0;
    for (const match of body.matchAll(/(\w+)\s*:\s*(\w+)/g)) {
        expect(['f32', 'u32']).toContain(match[2]);
        offsets[match[1]] = offset;
        offset += 4;
    }
    return { offsets, size: offset };
}

describe('the counted walk (counters)', () => {
    it('off: the module is the same text; on: it only appends, so every declaration is the same text', () => {
        for (const options of VARIANTS) {
            const plain = traceModuleWgsl(options);
            expect(traceModuleWgsl({ ...options, counters: false })).toBe(plain);
            const counted = traceModuleWgsl({ ...options, counters: true });
            expect(counted.startsWith(plain)).toBe(true);
            expect(plain).not.toContain('BvhWalkCounts');
            const added = counted.slice(plain.length);
            // The appended part declares only the counted walk's names.
            for (const match of added.matchAll(/\n(?:fn|struct|const|var) (\w+)/g)) expect(match[1]).toMatch(/^(bvh\w*Counted\w*|BvhWalkCounts)$/);
        }
        // The default batch kernels are unchanged too.
        for (const mode of ['closest', 'any'] as const) expect(batchKernelWgsl(mode, { counters: false })).toBe(batchKernelWgsl(mode));
    });

    it('each counted function is the production walk with counters added and nothing else', () => {
        for (const options of VARIANTS) {
            const counted = traceModuleWgsl({ ...options, counters: true });
            for (const name of ['bvhWalkClosest', 'bvhWalkAny']) {
                expect(withoutCounters(fnText(counted, `${name}Counted`))).toBe(fnText(counted, name));
            }
        }
    });

    it('declares the entry points for the options, and a record of the documented layout', () => {
        const masked = traceModuleWgsl({ counters: true });
        for (const signature of [
            'fn bvhTraceClosestCounted(ray: BvhRay) -> BvhWalkCounts',
            'fn bvhTraceAnyCounted(ray: BvhRay) -> BvhWalkCounts',
            'fn bvhTraceClosestCountedMasked(ray: BvhRay, mask: u32) -> BvhWalkCounts',
            'fn bvhTraceAnyCountedMasked(ray: BvhRay, mask: u32) -> BvhWalkCounts',
        ]) {
            expect(masked).toContain(signature);
        }
        const unmasked = traceModuleWgsl({ counters: true, rayMasks: false });
        expect(unmasked).toContain('fn bvhTraceClosestCounted(ray: BvhRay) -> BvhWalkCounts');
        expect(unmasked).not.toContain('CountedMasked');
        const layout = structLayout(masked, 'BvhWalkCounts');
        expect(layout.size).toBe(WALK_COUNTS_BYTES);
        expect(layout.offsets).toEqual(Object.fromEntries(Object.entries(WALK_COUNTS_WORD).map(([name, word]) => [name, word * 4])));
    });

    it('the counted batch kernel writes one record per ray in either mode, with the batch mask when masks are on', () => {
        for (const mode of ['closest', 'any'] as const) {
            const kernel = batchKernelWgsl(mode, { counters: true });
            expect(kernel).toContain('var<storage, read_write> bvhBatchResults: array<BvhWalkCounts>;');
            expect(kernel).toContain(`bvhBatchResults[rayIndex] = ${mode === 'closest' ? 'bvhTraceClosestCounted' : 'bvhTraceAnyCounted'}Masked(bvhBatchRays[rayIndex], bvhBatchParams.rayMask);`);
            const unmasked = batchKernelWgsl(mode, { counters: true, rayMasks: false });
            expect(unmasked).not.toContain('@group(1) @binding(2)');
        }
    });
});

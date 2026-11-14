// The folded decode of quantized nodes (`quantizedDecode: 'folded'`, docs/internals/design/compressed-nodes.md,
// question 6): an opt-in walk whose default text is unchanged, and the arithmetic it compiles, done here in
// f32 (Math.fround) so the pad's claim is checked without a GPU: a folded child interval holds the real one.
import { describe, expect, it } from 'vitest';
import { nodeFrameOf } from '../quantized-nodes.ts';
import { createRandom } from './oracle-fixtures.ts';
import { batchKernelWgsl, traceModuleWgsl, type QuantizedDecode } from './wgsl.ts';

const f = Math.fround;
const PAD = 2 ** -21;
const PAD_MAX = f(1e38);
const EXIT_PADDING = f(1.0000004);
const MIN_DIRECTION = f(1e-30);

function walkText(module: string, name: string): string {
    const start = module.indexOf(`fn ${name}(`);
    return module.slice(start, module.indexOf('\n}\n', start));
}

describe('quantizedDecode in the WGSL', () => {
    it("'exact' is the default and leaves every declaration of the module as it was", () => {
        const plain = traceModuleWgsl();
        expect(traceModuleWgsl({ quantizedDecode: 'exact' })).toBe(plain);
        expect(plain).not.toContain('bvhLoadMeshNodeFolded');
        expect(batchKernelWgsl('closest', { quantizedDecode: 'exact' })).toBe(batchKernelWgsl('closest'));
        expect(traceModuleWgsl({ counters: true, quantizedDecode: 'exact' })).toBe(traceModuleWgsl({ counters: true }));
    });

    it("'folded' changes only the inner-node step and the instance entry, and adds its loaders", () => {
        const exact = traceModuleWgsl({ triangleLeaves: 'indexed' });
        const folded = traceModuleWgsl({ triangleLeaves: 'indexed', quantizedDecode: 'folded' });
        for (const name of ['bvhWalkClosest', 'bvhWalkAny']) {
            const text = walkText(folded, name);
            expect(text).toContain('node = bvhLoadMeshNodeFolded(instance.rootNodeOffset + index);');
            expect(text).toContain('bvhEnterMeshNodesFolded(instance.rootNodeOffset, setup);');
            expect(text).not.toContain('bvhLoadMeshNode(');
            // Everything after the node step is the same text, but for the entry call.
            const tail = (module: string) => walkText(module, name).slice(walkText(module, name).indexOf('let leftReference'));
            expect(tail(folded).replace('bvhEnterMeshNodesFolded(instance.rootNodeOffset, setup);', 'bvhEnterMeshNodes(instance.rootNodeOffset);')).toBe(tail(exact));
        }
        // The exact loader stays for the queries, which test boxes, not rays.
        expect(folded).toContain('fn bvhLoadMeshNode(');
        expect(folded).toContain('fn bvhLoadMeshNodeFolded(');
        const queries = traceModuleWgsl({ queries: true, quantizedDecode: 'folded' });
        expect(walkText(queries, 'bvhClosestPointWalk')).toBe(walkText(traceModuleWgsl({ queries: true }), 'bvhClosestPointWalk'));
        // The counted walk follows the folded walk.
        const counted = traceModuleWgsl({ counters: true, quantizedDecode: 'folded' });
        expect(walkText(counted, 'bvhWalkClosestCounted')).toContain('bvhLoadMeshNodeFolded(instance.rootNodeOffset + index);\n                counts.meshNodes += 1u;');
        expect(batchKernelWgsl('any', { quantizedDecode: 'folded' })).toContain('fn bvhFoldedEntry(');
    });

    it("refuses f32 nodes, anyHitOrder 'larger' and unknown values", () => {
        expect(() => traceModuleWgsl({ nodeFormat: 'f32', quantizedDecode: 'folded' })).toThrow(/nodeFormat: 'quantized'/);
        expect(() => traceModuleWgsl({ anyHitOrder: 'larger', quantizedDecode: 'folded' })).toThrow(/larger/);
        expect(() => traceModuleWgsl({ quantizedDecode: 'fast' as QuantizedDecode })).toThrow(RangeError);
        expect(traceModuleWgsl({ nodeFormat: 'f32', quantizedDecode: 'exact' })).toBe(traceModuleWgsl({ nodeFormat: 'f32' }));
    });
});

/** bvhSafeInverse in f32. */
function safeInverse(direction: number): number {
    const magnitude = Math.max(Math.abs(f(direction)), MIN_DIRECTION);
    return f(1 / (direction < 0 ? -magnitude : magnitude));
}

interface Fold {
    scale: number;
    near: number;
    far: number;
    flip: boolean;
}

/** bvhEnterMeshNodesFolded for one axis, in f32 (`pad` 0 shows why the pad is there). */
function enter(origin: number, step: number, rayOrigin: number, inverse: number, pad = PAD): Fold {
    const offset = f(f(origin - rayOrigin) * inverse);
    const padding = Math.min(f(Math.abs(offset) * f(pad)), PAD_MAX);
    return { scale: f(step * inverse), near: f(offset - padding), far: f(offset + padding), flip: inverse < 0 };
}

/** One child axis's folded near and far sides, with a fused or an unfused fma. */
function sides(fold: Fold, qMin: number, qMax: number, fused: boolean): [number, number] {
    const fma = (q: number, c: number) => (fused ? f(q * fold.scale + c) : f(f(q * fold.scale) + c));
    return [fma(fold.flip ? qMax : qMin, fold.near), fma(fold.flip ? qMin : qMax, fold.far)];
}

describe('the folded arithmetic (f32 on the CPU)', () => {
    it('holds the real interval of every plane pair, fused or not, with frames far from the ray', () => {
        const random = createRandom(23);
        let worstWithoutPad = 0;
        for (let i = 0; i < 20000; i++) {
            // A mesh up to 1e4 wide, up to 1e5 from the world origin, and a ray starting near or inside it.
            const size = 10 ** (random() * 6 - 2);
            const lo = f((random() - 0.5) * 2 * 10 ** (random() * 5));
            const frame = nodeFrameOf([lo, 0, 0, f(lo + size), 0, 0]);
            const origin = frame.origin[0];
            const step = frame.step[0];
            const rayOrigin = f(lo + (random() * 1.2 - 0.1) * size);
            const direction = random() < 0.05 ? (random() - 0.5) * 1e-31 : (random() - 0.5) * 2;
            const inverse = safeInverse(direction);
            const qMin = Math.floor(random() * 65000);
            const qMax = qMin + Math.floor(random() * (65532 - qMin));
            // The real distances (in doubles: exact here) to the decoded planes, with the walk's own inverse.
            const planes = [origin + qMin * step, origin + qMax * step].map((plane) => (plane - rayOrigin) * inverse);
            const realNear = Math.min(...planes);
            const realFar = Math.max(...planes);
            for (const fused of [true, false]) {
                const [near, far] = sides(enter(origin, step, rayOrigin, inverse), qMin, qMax, fused);
                // The near side may lie at most the relative rounding of one operation late (the exact
                // path's own slab test has twice that), the far side within the exit padding.
                expect(near).toBeLessThanOrEqual(realNear + Math.abs(realNear) * 2 ** -23);
                expect(Math.max(far * EXIT_PADDING, far / EXIT_PADDING)).toBeGreaterThanOrEqual(realFar);
                const [bareNear, bareFar] = sides(enter(origin, step, rayOrigin, inverse, 0), qMin, qMax, fused);
                const scale = Math.max(Math.abs(realNear), Math.abs(realFar), 1e-30);
                worstWithoutPad = Math.max(worstWithoutPad, (bareNear - realNear) / scale, (realFar - bareFar) / scale);
            }
        }
        // Without the pad, the fold's offset rounding moves planes inward by far more than the exit padding covers.
        expect(worstWithoutPad).toBeGreaterThan(1e-5);
    });

    it('keeps an empty axis (both planes at 65535) and an overflowing offset free of NaN', () => {
        const fold = enter(f(-3), f(2 ** -10), f(1e20), safeInverse(1e-31));
        expect(Number.isFinite(fold.near) || Math.abs(fold.near) === Infinity).toBe(true);
        const [near, far] = sides(fold, 65535, 65535, true);
        expect(Number.isNaN(near) || Number.isNaN(far)).toBe(false);
        const unit = enter(0, 1, 0.5, 1);
        expect(sides(unit, 65535, 65535, true)).toEqual([65534.5, 65534.5]);
    });
});

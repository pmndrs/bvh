import { describe, expect, it } from 'vitest';
import { createCpuTracer, newTraceStats } from './cpu-traversal.ts';
import {
    EMPTY_CHILD_INDEX,
    INSTANCE_FLAG,
    INSTANCE_MASK_ALL,
    INSTANCE_MASK_BITS,
    instanceMaskFromFlags,
    INSTANCE_RECORD_WORDS,
    INSTANCE_WORD,
    MATERIAL_BITS,
    MATERIAL_SHIFT,
    MAX_MATERIAL,
    materialCode,
    materialFromFlags,
    NO_MATERIAL,
    NODE_WORD,
    TOP_HEADER_WORD,
    TOP_LEVEL_HEADER_WORDS,
    TRACE_STACK_SIZE,
} from './layout.ts';
import { createTinyBvhBuilder } from '../index.ts';
import type { MeshBvh, MeshGeometry } from './mesh-bvh.ts';
import { buildMeshBvhWithTinyBvh } from './mesh-bvh-topology.ts';
import { addToTally, compareAny, compareClosest, newTally } from './oracle-compare.ts';
import { createRandom, randomAffine, randomRays, randomTriangleSoup, tessellatedQuad } from './oracle-fixtures.ts';
import { buildReferenceScene, traceAnyReference, traceClosestReference } from './reference-tracer.ts';
import { assembleTraceScene, type TraceSceneInstance } from './scene-assembly.ts';
import { BVH_INSTANCE_BOX_PADDING } from '../gpu-instance-box.ts';
import { buildTopLevel, invertAffine, transformBounds, WORLD_BOX_PADDING } from './top-level.ts';

// Mesh trees come from tinybvh (WASM), the only mesh builder since B0 (DECISIONS 2026-11-04).
const tinyBvh = await createTinyBvhBuilder();
const buildMeshBvh = (geometry: MeshGeometry): MeshBvh => buildMeshBvhWithTinyBvh(geometry, tinyBvh);

function multiplyAffine(a: ArrayLike<number>, b: ArrayLike<number>): number[] {
    // Column-major 4x4 product a * b.
    const out = new Array<number>(16).fill(0);
    for (let column = 0; column < 4; column++) {
        for (let row = 0; row < 4; row++) {
            for (let k = 0; k < 4; k++) out[column * 4 + row] += a[k * 4 + row] * b[column * 4 + k];
        }
    }
    return out;
}

describe('invertAffine', () => {
    it('inverts random rotation/scale/translation transforms', () => {
        const random = createRandom(1);
        for (let i = 0; i < 50; i++) {
            const matrix = randomAffine(random, { extent: 10, mirror: i % 2 === 1 });
            const inverse = invertAffine(matrix);
            expect(inverse).not.toBeNull();
            const rows = inverse!.rows;
            const full = [
                rows[0], rows[4], rows[8], 0,
                rows[1], rows[5], rows[9], 0,
                rows[2], rows[6], rows[10], 0,
                rows[3], rows[7], rows[11], 1,
            ];
            const identity = multiplyAffine(full, matrix);
            for (let k = 0; k < 16; k++) expect(identity[k]).toBeCloseTo(k % 5 === 0 ? 1 : 0, 9);
            expect(inverse!.determinant < 0).toBe(i % 2 === 1);
        }
    });

    it('returns null for a collapsed transform', () => {
        const flat = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        expect(invertAffine(flat)).toBeNull();
    });
});

describe('transformBounds', () => {
    // f32 inputs, so the CPU and the GPU gather see the same numbers.
    const f = Math.fround;
    /** `bvhInstanceWorldBox` (gpu-instance-box.ts) for one axis, every operation rounded to f32 (dot as written, unfused). */
    function gpuAxis(m: readonly number[], b: readonly number[]): [number, number] {
        const dot = (a: number[], v: number[]) => f(f(f(a[0] * v[0]) + f(a[1] * v[1])) + f(a[2] * v[2]));
        const center = [0, 1, 2].map((k) => f(f(b[k] + b[3 + k]) * 0.5));
        const half = [0, 1, 2].map((k) => f(f(b[3 + k] - b[k]) * 0.5));
        const reach = [0, 1, 2].map((k) => Math.max(Math.abs(b[k]), Math.abs(b[3 + k])));
        const abs = m.slice(0, 3).map(Math.abs);
        const worldCenter = f(dot(m.slice(0, 3), center) + m[3]);
        const worldHalf = dot(abs, half);
        const padding = f(f(dot(abs, reach) + Math.abs(m[3])) * BVH_INSTANCE_BOX_PADDING);
        return [f(f(worldCenter - worldHalf) - padding), f(f(worldCenter + worldHalf) + padding)];
    }

    it('pads each axis by WORLD_BOX_PADDING of its summed term magnitude, as the GPU gather does, around the exact corners', () => {
        expect(WORLD_BOX_PADDING).toBe(BVH_INSTANCE_BOX_PADDING);
        const random = createRandom(7);
        const out = new Float32Array(6);
        let checked = 0;
        for (let i = 0; i < 2000; i++) {
            // Near the origin, far from it (a large translation cancels against the box), and boxes off their own origin.
            const extent = [10, 1e4, 1e6][i % 3];
            const matrix = randomAffine(random, { extent, mirror: i % 2 === 1 }).map(f);
            const centre = [0, 1, 2].map(() => (i % 4 === 0 ? (random() * 2 - 1) * 500 : 0));
            const size = [0, 1, 2].map(() => (i % 5 === 0 ? 1e-3 : 0.01 + random() * 4));
            const bounds = [0, 1, 2].map((k) => f(centre[k] - size[k])).concat([0, 1, 2].map((k) => f(centre[k] + size[k])));
            transformBounds(bounds, matrix, out, 0);
            for (let axis = 0; axis < 3; axis++) {
                const m = [matrix[axis], matrix[4 + axis], matrix[8 + axis], matrix[12 + axis]];
                // The exact corners (doubles of f32 inputs: each product and sum is exact to 2^-53).
                let low = Infinity;
                let high = -Infinity;
                for (let corner = 0; corner < 8; corner++) {
                    const value = m[0] * bounds[corner & 1 ? 3 : 0] + m[1] * bounds[corner & 2 ? 4 : 1] + m[2] * bounds[corner & 4 ? 5 : 2] + m[3];
                    low = Math.min(low, value);
                    high = Math.max(high, value);
                }
                const magnitude = [0, 1, 2].reduce((sum, k) => sum + Math.abs(m[k]) * Math.max(Math.abs(bounds[k]), Math.abs(bounds[3 + k])), Math.abs(m[3]));
                const pad = magnitude * WORLD_BOX_PADDING;
                // The real content is inside, with at least 15/16 of the padding left after the store into f32.
                expect(out[axis]).toBeLessThanOrEqual(low - pad * (15 / 16));
                expect(out[3 + axis]).toBeGreaterThanOrEqual(high + pad * (15 / 16));
                expect(out[axis]).toBeGreaterThanOrEqual(low - pad * (17 / 16));
                expect(out[3 + axis]).toBeLessThanOrEqual(high + pad * (17 / 16));
                // The GPU's box, counted the same way, is within half the padding of it on every side.
                const [gpuLow, gpuHigh] = gpuAxis(m, bounds);
                expect(Math.abs(gpuLow - out[axis])).toBeLessThanOrEqual(pad / 2);
                expect(Math.abs(gpuHigh - out[3 + axis])).toBeLessThanOrEqual(pad / 2);
                checked++;
            }
        }
        expect(checked).toBe(6000);
    });
});

describe('buildTopLevel', () => {
    const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const bounds = [-1, -1, -1, 1, 1, 1];

    it('keeps the platform instance index, offsets and mirror flag in each record', () => {
        const random = createRandom(2);
        const instances = Array.from({ length: 40 }, (_, i) => ({
            instanceIndex: 1000 + i * 7,
            nodeOffset: i * 3,
            triangleOffset: i * 11,
            meshBounds: bounds,
            objectToWorld: randomAffine(random, { extent: 30, mirror: i % 3 === 0 }),
        }));
        const top = buildTopLevel(instances);
        const words = new Uint32Array(top.buffer);
        expect(words[TOP_HEADER_WORD.instanceCount]).toBe(40);
        expect(top.instanceCount).toBe(40);
        expect(top.nodeCount).toBe(39);

        const seen = new Set<number>();
        for (let record = 0; record < 40; record++) {
            const base = TOP_LEVEL_HEADER_WORDS + record * INSTANCE_RECORD_WORDS;
            const source = top.recordOrder[record];
            seen.add(source);
            expect(words[base + INSTANCE_WORD.instanceIndex]).toBe(instances[source].instanceIndex);
            expect(words[base + INSTANCE_WORD.rootNodeOffset]).toBe(instances[source].nodeOffset);
            expect(words[base + INSTANCE_WORD.triangleOffset]).toBe(instances[source].triangleOffset);
            const mirrored = (words[base + INSTANCE_WORD.flags] & INSTANCE_FLAG.mirrored) !== 0;
            expect(mirrored).toBe(source % 3 === 0);
        }
        expect(seen.size).toBe(40);
    });

    it('writes the instance mask in flags bits 8-15: 0xff by default, the given one otherwise, other bits unchanged', () => {
        const random = createRandom(4);
        const masks = [undefined, 0, 1, 0x80, 0xff, 0x5a];
        const instances = masks.map((mask, i) => ({
            instanceIndex: i,
            nodeOffset: 0,
            triangleOffset: 0,
            meshBounds: bounds,
            objectToWorld: randomAffine(random, { extent: 10, mirror: i % 2 === 1 }),
            mask,
        }));
        const top = buildTopLevel(instances);
        const words = new Uint32Array(top.buffer);
        for (let record = 0; record < instances.length; record++) {
            const flags = words[TOP_LEVEL_HEADER_WORDS + record * INSTANCE_RECORD_WORDS + INSTANCE_WORD.flags];
            const source = top.recordOrder[record];
            expect(instanceMaskFromFlags(flags)).toBe(masks[source] ?? INSTANCE_MASK_ALL);
            expect(flags & ~INSTANCE_MASK_BITS).toBe(
                INSTANCE_FLAG.contributesToGi | INSTANCE_FLAG.visible | (source % 2 === 1 ? INSTANCE_FLAG.mirrored : 0),
            );
        }
        for (const mask of [256, -1, 1.5]) {
            expect(() => buildTopLevel([{ ...instances[0], mask }])).toThrow(RangeError);
        }
    });

    it('writes material + 1 in flags bits 16-31 (0 for none), and refuses a material out of range or on a merged record', () => {
        const base = { nodeOffset: 0, triangleOffset: 0, meshBounds: [0, 0, 0, 1, 1, 1], objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] };
        const materials = [undefined, 0, 41, MAX_MATERIAL, NO_MATERIAL];
        const top = buildTopLevel(materials.map((material, instanceIndex) => ({ ...base, instanceIndex, material, mask: 0x5a })));
        const words = new Uint32Array(top.buffer);
        for (let record = 0; record < top.instanceCount; record++) {
            const flags = words[TOP_LEVEL_HEADER_WORDS + record * INSTANCE_RECORD_WORDS + INSTANCE_WORD.flags];
            const material = materials[top.recordOrder[record]];
            expect(flags >>> MATERIAL_SHIFT).toBe(materialCode(material, 0));
            expect(materialFromFlags(flags)).toBe(material ?? NO_MATERIAL);
            // The mask and flag bits are untouched.
            expect(instanceMaskFromFlags(flags)).toBe(0x5a);
            expect(flags & ~MATERIAL_BITS & ~INSTANCE_MASK_BITS).toBe(INSTANCE_FLAG.contributesToGi | INSTANCE_FLAG.visible);
        }
        for (const bad of [-1, MAX_MATERIAL + 1, 1.5, Number.NaN]) {
            expect(() => buildTopLevel([{ ...base, instanceIndex: 9, material: bad }]), String(bad)).toThrow(/instance 9: material must be an integer in 0-65534/);
        }
        expect(() => buildTopLevel([{ ...base, instanceIndex: 9, material: 2, merged: true }])).toThrow(/merged record has no material/);
    });

    it('puts every instance in exactly one leaf and every leaf box around its instance', () => {
        const random = createRandom(3);
        const instances = Array.from({ length: 100 }, (_, i) => ({
            instanceIndex: i,
            nodeOffset: 0,
            triangleOffset: 0,
            meshBounds: bounds,
            objectToWorld: randomAffine(random, { extent: 50 }),
        }));
        const top = buildTopLevel(instances);
        const floats = new Float32Array(top.buffer);
        const words = new Uint32Array(top.buffer);
        const firstNodeWord = TOP_LEVEL_HEADER_WORDS + top.instanceCount * INSTANCE_RECORD_WORDS;
        const leafRecords: number[] = [];
        const stack = [0];
        while (stack.length > 0) {
            const node = stack.pop() as number;
            const base = firstNodeWord + node * 16;
            for (const [minWord, maxWord, indexWord, countWord] of [
                [NODE_WORD.leftMin, NODE_WORD.leftMax, NODE_WORD.leftIndex, NODE_WORD.leftCount],
                [NODE_WORD.rightMin, NODE_WORD.rightMax, NODE_WORD.rightIndex, NODE_WORD.rightCount],
            ]) {
                const index = words[base + indexWord];
                const count = words[base + countWord];
                if (count === 0) {
                    stack.push(index);
                    continue;
                }
                expect(count).toBe(1);
                leafRecords.push(index);
                const expected = new Float32Array(6);
                transformBounds(bounds, instances[top.recordOrder[index]].objectToWorld, expected, 0);
                for (let axis = 0; axis < 3; axis++) {
                    expect(floats[base + minWord + axis]).toBe(expected[axis]);
                    expect(floats[base + maxWord + axis]).toBe(expected[3 + axis]);
                }
            }
        }
        expect(leafRecords.sort((a, b) => a - b)).toEqual(Array.from({ length: 100 }, (_, i) => i));
    });

    it('leaves out hidden, non-GI and collapsed instances, and handles an empty scene', () => {
        const make = (extra: object, instanceIndex: number) => ({
            instanceIndex, nodeOffset: 0, triangleOffset: 0, meshBounds: bounds, objectToWorld: identity, ...extra,
        });
        const collapsed = identity.slice();
        collapsed[0] = 0;
        const top = buildTopLevel([
            make({}, 0),
            make({ visible: false }, 1),
            make({ contributesToGi: false }, 2),
            make({ objectToWorld: collapsed }, 3),
        ]);
        expect(top.instanceCount).toBe(1);
        expect(top.skippedSingular).toBe(1);

        const empty = buildTopLevel([]);
        expect(empty.instanceCount).toBe(0);
        expect(empty.nodeCount).toBe(0);
        expect(new Uint32Array(empty.buffer)[TOP_HEADER_WORD.nodeCount]).toBe(0);
    });

    it('stores a lone instance as a root with an empty right child', () => {
        const top = buildTopLevel([{ instanceIndex: 5, nodeOffset: 0, triangleOffset: 0, meshBounds: bounds, objectToWorld: identity }]);
        const words = new Uint32Array(top.buffer);
        const rootWord = TOP_LEVEL_HEADER_WORDS + INSTANCE_RECORD_WORDS;
        expect(words[rootWord + NODE_WORD.rightIndex]).toBe(EMPTY_CHILD_INDEX);
        expect(words[rootWord + NODE_WORD.leftCount]).toBe(1);
    });
});

/** Runs both tracers over the same rays and returns the tallies. */
function compareCpuTracerWithReference(
    meshes: ReturnType<typeof randomTriangleSoup>[],
    instances: TraceSceneInstance[],
    rays: ReturnType<typeof randomRays>,
) {
    const scene = assembleTraceScene(meshes.map((mesh) => buildMeshBvh(mesh)), instances);
    const tracer = createCpuTracer(scene);
    const reference = buildReferenceScene(meshes, instances);
    const closest = newTally();
    const any = newTally();
    const stats = newTraceStats();
    rays.forEach((ray, index) => {
        const actual = tracer.closest(ray, { stats });
        const traced = actual && { hit: true, ...actual };
        addToTally(closest, compareClosest(traced, traceClosestReference(reference, ray), ray), `ray ${index}`);
        addToTally(any, compareAny(tracer.any(ray), traceAnyReference(reference, ray)), `ray ${index}`);
    });
    return { scene, closest, any, stats };
}

describe('CPU traversal against the brute-force reference', () => {
    it('matches on a single mesh with an identity instance', { timeout: 60_000 }, () => {
        const mesh = randomTriangleSoup(createRandom(10), 1500, 10, 1);
        const rays = randomRays(createRandom(11), 3000, [-10, -10, -10, 10, 10, 10], 4);
        const { closest, any } = compareCpuTracerWithReference(
            [mesh],
            [{ meshIndex: 0, instanceIndex: 0, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] }],
            rays,
        );
        expect(closest.mismatches, closest.examples.join('\n')).toBe(0);
        expect(any.mismatches, any.examples.join('\n')).toBe(0);
        expect(closest.matches).toBeGreaterThan(2900);
    });

    it('matches with several meshes, non-uniform scales, rotations and mirrors, and random ray intervals', { timeout: 60_000 }, () => {
        const random = createRandom(20);
        const meshes = [randomTriangleSoup(random, 400, 3, 0.8), randomTriangleSoup(random, 60, 2, 1), tessellatedQuad(6, 2)];
        const instances: TraceSceneInstance[] = Array.from({ length: 24 }, (_, i) => ({
            meshIndex: i % 3,
            // Instance indices deliberately unrelated to the order they are placed in.
            instanceIndex: 500 - i * 3,
            objectToWorld: randomAffine(random, { extent: 12, mirror: i % 4 === 0 }),
            // Some without a material, the rest with one unrelated to the index (compared through the oracle).
            material: i % 5 === 0 ? undefined : (i * 7919) % 1000,
        }));
        const rays = randomRays(createRandom(21), 4000, [-14, -14, -14, 14, 14, 14], 6, { randomInterval: true });
        const { closest, any, scene } = compareCpuTracerWithReference(meshes, instances, rays);
        expect(closest.mismatches, closest.examples.join('\n')).toBe(0);
        expect(any.mismatches, any.examples.join('\n')).toBe(0);
        expect(closest.matches + closest.ambiguous).toBe(4000);
        // Ambiguous results are only the rare edge-grazing rays.
        expect(closest.ambiguous + any.ambiguous).toBeLessThan(40);
        expect(scene.requiredStackSize).toBeLessThanOrEqual(TRACE_STACK_SIZE);
    });

    it('matches with many copies of one small mesh and reports stack use under the limit', { timeout: 60_000 }, () => {
        const random = createRandom(30);
        const meshes = [randomTriangleSoup(random, 40, 1, 0.5)];
        const instances: TraceSceneInstance[] = Array.from({ length: 150 }, (_, i) => ({
            meshIndex: 0,
            instanceIndex: i,
            objectToWorld: randomAffine(random, { extent: 25, mirror: i % 5 === 0 }),
        }));
        const rays = randomRays(createRandom(31), 1500, [-25, -25, -25, 25, 25, 25], 5);
        const { closest, any, scene, stats } = compareCpuTracerWithReference(meshes, instances, rays);
        expect(closest.mismatches, closest.examples.join('\n')).toBe(0);
        expect(any.mismatches, any.examples.join('\n')).toBe(0);
        expect(stats.maxStackSize).toBeGreaterThan(0);
        expect(stats.maxStackSize).toBeLessThanOrEqual(scene.requiredStackSize);
    });

    it('gives front faces on the authored side, also for mirrored instances (DECISIONS 2026-11-04)', () => {
        // One triangle facing +Z, seen from above and below, plain and mirrored in x.
        // Mirroring x keeps the authored +Z side, so both copies show their front from above.
        const mesh = { positions: new Float32Array([-1, -1, 0, 1, -1, 0, 0, 1, 0]) };
        const plain = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        const mirror = [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 0, 0, 1];
        const scene = assembleTraceScene([buildMeshBvh(mesh)], [
            { meshIndex: 0, instanceIndex: 0, objectToWorld: plain },
            { meshIndex: 0, instanceIndex: 1, objectToWorld: mirror },
        ]);
        const tracer = createCpuTracer(scene);
        const down = { direction: [0, 0, -1] as const, tMin: 0, tMax: 100 };
        const up = { direction: [0, 0, 1] as const, tMin: 0, tMax: 100 };
        expect(tracer.closest({ ...down, origin: [0, 0, 5] })?.frontFace).toBe(true);
        expect(tracer.closest({ ...up, origin: [0, 0, -5] })?.frontFace).toBe(false);
        expect(tracer.closest({ ...down, origin: [10, 0, 5] })?.frontFace).toBe(true);
        expect(tracer.closest({ ...up, origin: [10, 0, -5] })?.frontFace).toBe(false);
    });

    it("reports the authored side's world normal and the instance material, also when mirrored, scaled and sheared", () => {
        // One triangle facing (0, -1, 1). Mirrored in x it keeps the authored side, and a
        // non-uniform scale moves the normal by the inverse transpose, not the transform itself.
        const mesh = { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 1]) };
        const scale = [2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        const mirror = [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 0, 0, 1];
        const instances: TraceSceneInstance[] = [
            { meshIndex: 0, instanceIndex: 0, objectToWorld: scale, material: 3 },
            { meshIndex: 0, instanceIndex: 1, objectToWorld: mirror },
        ];
        const scene = assembleTraceScene([buildMeshBvh(mesh)], instances);
        const tracer = createCpuTracer(scene);
        const reference = buildReferenceScene([mesh], instances);
        const rays = [
            { origin: [0.2, 0.2, 5] as const, direction: [0, 0, -1] as const, tMin: 0, tMax: 100 },
            { origin: [9.8, 0.2, 5] as const, direction: [0, 0, -1] as const, tMin: 0, tMax: 100 },
        ];
        const [scaled, mirrored] = rays.map((ray) => tracer.closest(ray));
        expect(scaled?.material).toBe(3);
        expect(mirrored?.material).toBe(NO_MATERIAL);
        // Object normal (0, -1, 1)/sqrt2; the inverse transpose of diag(2, 1, 1) keeps its direction here.
        expect(scaled?.normal[0]).toBeCloseTo(0, 6);
        expect(scaled?.normal[1]).toBeCloseTo(-Math.SQRT1_2, 6);
        expect(mirrored?.normal[2]).toBeCloseTo(Math.SQRT1_2, 6);
        // Front face is the side the normal points to.
        for (const [index, hit] of [scaled, mirrored].entries()) {
            const direction = rays[index].direction;
            expect(hit?.frontFace).toBe(hit!.normal[0] * direction[0] + hit!.normal[1] * direction[1] + hit!.normal[2] * direction[2] < 0);
        }
        rays.forEach((ray) => {
            const actual = tracer.closest(ray);
            expect(compareClosest(actual && { hit: true, ...actual }, traceClosestReference(reference, ray), ray).verdict).toBe('match');
        });
        // A shear of x into z: the normal of the z = 0 plane must lean against it.
        const shear = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        const flat = { positions: new Float32Array([-1, -1, 0, 1, -1, 0, 0, 1, 0]) };
        const sheared = createCpuTracer(assembleTraceScene([buildMeshBvh(flat)], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: shear }]));
        const hit = sheared.closest({ origin: [0, 0, 5], direction: [0, 0, -1], tMin: 0, tMax: 100 });
        // World plane z = x: normal (-1, 0, 1)/sqrt2.
        expect(hit?.normal[0]).toBeCloseTo(-Math.SQRT1_2, 6);
        expect(hit?.normal[1]).toBeCloseTo(0, 6);
        expect(hit?.normal[2]).toBeCloseTo(Math.SQRT1_2, 6);
    });

    it('refuses a scene whose trees could overflow the stack', () => {
        // 70 instances in a line build a top-level tree far shallower than 64, so force it
        // with a degenerate mesh depth: the check compares the sum against the stack size.
        const mesh = buildMeshBvh(randomTriangleSoup(createRandom(1), 50, 5, 1));
        const deepMesh = { ...mesh, maxDepth: TRACE_STACK_SIZE };
        expect(() =>
            assembleTraceScene([deepMesh], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] }]),
        ).toThrow(/stack/);
    });
});

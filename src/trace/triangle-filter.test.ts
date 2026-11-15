/**
 * The walk's triangle filter and ray masks on the CPU: the CPU port of the
 * walk, with the filter flag written by `writeTriangleFilterFlags` and
 * instance masks in the top level, against the brute-force reference given the
 * same filter and masks; and the hooks in the generated WGSL.
 *
 * Every filter here gives the same answer for every hit on one triangle (it
 * reads the instance, the triangle, the side, the query or the ray mask, never
 * t or the barycentrics). The walk runs in doubles over f32 records and the
 * reference in doubles over the original corners, so a filter that changes its
 * mind across a triangle would flip at slightly different points in the two
 * and the comparison would flake at the boundary.
 */

import { describe, expect, it } from 'vitest';
import { createTinyBvhBuilder } from '../index.ts';
import { wgslNameProblems } from '../wgsl-names.ts';
import { createCpuTracer, CPU_TRACE_BACK_SIDE, CPU_TRACE_DOUBLE_SIDE, CPU_TRACE_FRONT_SIDE } from './cpu-traversal.ts';
import {
    BVH_QUERY_ANY,
    BVH_QUERY_CLOSEST,
    RAY_MASK_ALL,
    TRIANGLE_FLAG_FILTER,
    TRIANGLE_WORD,
    TRIANGLE_WORDS,
} from '../gpu-layout.ts';
import type { MeshBvh, MeshGeometry } from './mesh-bvh.ts';
import { buildMeshBvhWithTinyBvh } from './mesh-bvh-topology.ts';
import { addToTally, compareAny, compareClosest, newTally } from './oracle-compare.ts';
import { createRandom, randomAffine, randomRays, tessellatedQuad, toFloat32Ray } from './oracle-fixtures.ts';
import type { TraceRay } from './rays.ts';
import { buildReferenceScene, traceAnyReference, traceClosestReference } from './reference-tracer.ts';
import { assembleTraceScene, type TraceSceneData, type TraceSceneInstance } from './scene-assembly.ts';
import {
    DEFAULT_TRIANGLE_FILTER_WGSL,
    TRIANGLE_FILTER_SIGNATURE_WGSL,
    writeTriangleFilterFlags,
    type TriangleCandidate,
    type TrianglePasses,
} from './triangle-filter.ts';
import { batchKernelWgsl, traceModuleWgsl } from './wgsl.ts';

const tinyBvh = await createTinyBvhBuilder();

const translation = (x: number, y: number, z: number): number[] => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];

/** A 4x4-cell quad (32 triangles, counter-clockwise seen from +z), used for both meshes. */
const QUAD: MeshGeometry = tessellatedQuad(4, 1);
const QUAD_TRIANGLES = (QUAD.indices?.length ?? 0) / 3;

/** Mesh 0 in front at z = 1, mesh 1 behind it at z = -1, both facing +z. */
const STACKED: readonly TraceSceneInstance[] = [
    { meshIndex: 0, instanceIndex: 0, objectToWorld: translation(0, 0, 1) },
    { meshIndex: 1, instanceIndex: 1, objectToWorld: translation(0, 0, -1) },
];

/** Both meshes' trees, mesh `m`'s records flagged where `flagged(m, original)` says (written before assembly). */
function flaggedScene(instances: readonly TraceSceneInstance[], flagged: (meshIndex: number, original: number) => boolean): { scene: TraceSceneData; flaggedCount: number } {
    let flaggedCount = 0;
    const trees: MeshBvh[] = [0, 1].map((meshIndex) => {
        const bvh = buildMeshBvhWithTinyBvh(QUAD, tinyBvh);
        flaggedCount += writeTriangleFilterFlags(bvh.triangles, (original) => flagged(meshIndex, original));
        return bvh;
    });
    return { scene: assembleTraceScene(trees, instances), flaggedCount };
}

/** A grid of rays along -z (or +z) from z = +-5, through cell interiors only (away from every edge). */
function gridRays(direction: 1 | -1, tMax = 1e6): TraceRay[] {
    const rays: TraceRay[] = [];
    for (let row = 0; row < 8; row++) {
        for (let column = 0; column < 8; column++) {
            // Quarter-cell offsets: never on a cell edge or a cell's diagonal.
            const x = -1 + (column + 0.3) * 0.25;
            const y = -1 + (row + 0.6) * 0.25;
            rays.push(toFloat32Ray([x, y, -5 * direction], [0, 0, direction], 0, tMax));
        }
    }
    return rays;
}

interface CompareOptions {
    readonly triangleFilter?: TrianglePasses;
    readonly rayMasks?: boolean;
    /** Ray masks to trace with; undefined means the default (all ones). */
    readonly masks?: readonly (number | undefined)[];
}

function compareWithReference(scene: TraceSceneData, instances: readonly TraceSceneInstance[], rays: readonly TraceRay[], options: CompareOptions = {}) {
    const tracer = createCpuTracer(scene, { triangleFilter: options.triangleFilter, rayMasks: options.rayMasks });
    const reference = buildReferenceScene([QUAD, QUAD], instances, { triangleFilter: options.triangleFilter, rayMasks: options.rayMasks });
    const closest = newTally();
    const any = newTally();
    for (const mask of options.masks ?? [undefined]) {
        rays.forEach((ray, index) => {
            const hit = tracer.closest(ray, { mask });
            const traced = hit ? { hit: true, ...hit } : null;
            const label = `mask ${mask?.toString(16)} ray ${index}`;
            addToTally(closest, compareClosest(traced, traceClosestReference(reference, ray, { mask }), ray), label);
            addToTally(any, compareAny(tracer.any(ray, { mask }), traceAnyReference(reference, ray, { mask })), label);
        });
    }
    return { closest, any };
}

function expectNoMismatches({ closest, any }: ReturnType<typeof compareWithReference>): void {
    expect(closest.mismatches, closest.examples.join('\n')).toBe(0);
    expect(any.mismatches, any.examples.join('\n')).toBe(0);
}

describe('writeTriangleFilterFlags', () => {
    it('sets and clears bit 0 by original index, keeps the other bits, and honours a range', () => {
        const bvh = buildMeshBvhWithTinyBvh(QUAD, tinyBvh);
        const words = new Uint32Array(bvh.triangles);
        for (let triangle = 0; triangle < bvh.triangleCount; triangle++) words[triangle * TRIANGLE_WORDS + TRIANGLE_WORD.flags] = 0b100;

        expect(writeTriangleFilterFlags(bvh.triangles, (original) => original % 2 === 0)).toBe(QUAD_TRIANGLES / 2);
        for (let triangle = 0; triangle < bvh.triangleCount; triangle++) {
            const original = words[triangle * TRIANGLE_WORDS + TRIANGLE_WORD.originalIndex];
            expect(words[triangle * TRIANGLE_WORDS + TRIANGLE_WORD.flags]).toBe(0b100 | (original % 2 === 0 ? TRIANGLE_FLAG_FILTER : 0));
        }

        // Clearing records 0-3 only.
        expect(writeTriangleFilterFlags(words, () => false, { triangleOffset: 0, triangleCount: 4 })).toBe(0);
        for (let triangle = 0; triangle < bvh.triangleCount; triangle++) {
            const original = words[triangle * TRIANGLE_WORDS + TRIANGLE_WORD.originalIndex];
            const expected = triangle >= 4 && original % 2 === 0 ? TRIANGLE_FLAG_FILTER : 0;
            expect(words[triangle * TRIANGLE_WORDS + TRIANGLE_WORD.flags] & TRIANGLE_FLAG_FILTER).toBe(expected);
        }
    });

    it('throws a RangeError for a range outside the buffer, before calling the predicate', () => {
        const bvh = buildMeshBvhWithTinyBvh(QUAD, tinyBvh);
        let calls = 0;
        const predicate = () => (calls++, true);
        for (const range of [
            { triangleOffset: 0, triangleCount: bvh.triangleCount + 1 },
            { triangleOffset: bvh.triangleCount, triangleCount: 1 },
            { triangleOffset: -1, triangleCount: 2 },
            { triangleOffset: 0, triangleCount: -1 },
            { triangleOffset: 0.5, triangleCount: 1 },
        ]) {
            expect(() => writeTriangleFilterFlags(bvh.triangles, predicate, range)).toThrow(RangeError);
        }
        expect(calls).toBe(0);
        // The last record and an empty range at the end are fine.
        expect(writeTriangleFilterFlags(bvh.triangles, predicate, { triangleOffset: bvh.triangleCount - 1, triangleCount: 1 })).toBe(1);
        expect(writeTriangleFilterFlags(bvh.triangles, predicate, { triangleOffset: bvh.triangleCount, triangleCount: 0 })).toBe(0);
    });

    it("writes one mesh of an assembled scene through its TraceSceneData.meshes entry, which the walk then sees", () => {
        const { scene } = flaggedScene(STACKED, () => false);
        expect(scene.meshes.map((mesh) => mesh.triangleCount)).toEqual([QUAD_TRIANGLES, QUAD_TRIANGLES]);
        expect(writeTriangleFilterFlags(scene.triangles, () => true, scene.meshes[0])).toBe(QUAD_TRIANGLES);
        const words = new Uint32Array(scene.triangles);
        for (let triangle = 0; triangle < 2 * QUAD_TRIANGLES; triangle++) {
            expect(words[triangle * TRIANGLE_WORDS + TRIANGLE_WORD.flags] & TRIANGLE_FLAG_FILTER).toBe(triangle < QUAD_TRIANGLES ? 1 : 0);
        }
        const tracer = createCpuTracer(scene, { triangleFilter: () => true });
        for (const ray of gridRays(-1)) expect(tracer.closest(ray)?.instanceIndex).toBe(1);
    });
});

describe('the CPU walk with a triangle filter against the filtered reference', () => {
    it('filterEveryTriangle asks the filter about unflagged triangles too, and the CPU-only side option culls before it', () => {
        const { scene, flaggedCount } = flaggedScene(STACKED, () => false);
        expect(flaggedCount).toBe(0);
        const asked: TriangleCandidate[] = [];
        const passFront: TrianglePasses = (candidate) => {
            asked.push(candidate);
            return candidate.instance === 0;
        };
        const rays = gridRays(-1);
        // Without filterEveryTriangle, unflagged triangles never reach the filter: the front quad blocks.
        const plain = createCpuTracer(scene, { triangleFilter: passFront });
        for (const ray of rays) expect(plain.closest(ray)?.instanceIndex).toBe(0);
        expect(asked).toHaveLength(0);
        // With it, every hit asks: the front quad lets the ray through to the one behind, which blocks.
        const every = createCpuTracer(scene, { triangleFilter: passFront, filterEveryTriangle: true });
        for (const ray of rays) expect(every.closest(ray)?.instanceIndex).toBe(1);
        expect(asked.length).toBe(rays.length * 2);
        expect(asked.every((candidate) => candidate.frontFace)).toBe(true);
        // Side culling: from behind (+z rays) both quads are back faces. Front only: no hit, and the
        // filter isn't asked about culled triangles. Back only: the nearer quad (instance 1).
        asked.length = 0;
        const fromBehind = gridRays(1);
        for (const ray of fromBehind) {
            expect(every.closest(ray, { side: CPU_TRACE_FRONT_SIDE })).toBeNull();
            expect(plain.closest(ray, { side: CPU_TRACE_BACK_SIDE })?.instanceIndex).toBe(1);
            expect(plain.closest(ray, { side: CPU_TRACE_DOUBLE_SIDE })?.instanceIndex).toBe(1);
        }
        expect(asked).toHaveLength(0);
        for (const ray of rays) expect(plain.closest(ray, { side: CPU_TRACE_BACK_SIDE })).toBeNull();
    });

    it('lets rays through a filtered-out front quad to the one behind it', () => {
        const { scene, flaggedCount } = flaggedScene(STACKED, (meshIndex) => meshIndex === 0);
        expect(flaggedCount).toBe(QUAD_TRIANGLES);
        const filter: TrianglePasses = (candidate) => candidate.instance === 0;
        const tracer = createCpuTracer(scene, { triangleFilter: filter });
        for (const ray of gridRays(-1)) {
            const hit = tracer.closest(ray);
            expect(hit?.instanceIndex).toBe(1);
            expect(hit?.t).toBeCloseTo(6, 5);
            expect(tracer.any(ray)).toBe(true);
        }

        const rays = [...gridRays(-1), ...gridRays(1), ...randomRays(createRandom(7), 2000, [-1, -1, -1, 1, 1, 1], 3)];
        expectNoMismatches(compareWithReference(scene, STACKED, rays, { triangleFilter: filter }));
    });

    it('tells the filter everything about the candidate', () => {
        const { scene } = flaggedScene(STACKED, () => true);
        const seen: TriangleCandidate[] = [];
        const tracer = createCpuTracer(scene, { triangleFilter: (candidate) => (seen.push(candidate), false) });
        const ray = gridRays(-1)[9];
        const hit = tracer.closest(ray, { mask: 0x0f })!;
        const asked = seen.find((candidate) => candidate.instance === hit.instanceIndex && candidate.triangle === hit.triangleIndex)!;
        expect(asked).toEqual({
            instance: hit.instanceIndex,
            triangle: hit.triangleIndex,
            t: hit.t,
            barycentrics: [hit.u, hit.v],
            frontFace: hit.frontFace,
            query: BVH_QUERY_CLOSEST,
            rayMask: 0x0f,
        });
        seen.length = 0;
        tracer.any(ray);
        expect(seen.length).toBeGreaterThan(0);
        expect(seen.every((candidate) => candidate.query === BVH_QUERY_ANY && candidate.rayMask === RAY_MASK_ALL)).toBe(true);
    });

    it('filters per triangle: only the flagged half of the front quad lets rays through', () => {
        const flagged = (meshIndex: number, original: number) => meshIndex === 0 && original % 2 === 0;
        const { scene } = flaggedScene(STACKED, flagged);
        // The reference asks about every triangle, so its filter says no where the walk has no flag.
        const filter: TrianglePasses = (candidate) => flagged(candidate.instance, candidate.triangle);
        const tracer = createCpuTracer(scene, { triangleFilter: filter });
        const seen = new Set<number>();
        for (const ray of gridRays(-1)) {
            const hit = tracer.closest(ray)!;
            seen.add(hit.instanceIndex);
            if (hit.instanceIndex === 0) expect(hit.triangleIndex % 2).toBe(1);
        }
        expect([...seen].sort()).toEqual([0, 1]);

        const rays = [...gridRays(-1), ...randomRays(createRandom(8), 2000, [-1, -1, -1, 1, 1, 1], 3)];
        expectNoMismatches(compareWithReference(scene, STACKED, rays, { triangleFilter: filter }));
    });

    it('culls back faces with a filter on frontFace (single-sided), mirrored instances included', () => {
        const singleSided: TrianglePasses = (candidate) => !candidate.frontFace;
        const { scene } = flaggedScene(STACKED, () => true);
        const tracer = createCpuTracer(scene, { triangleFilter: singleSided });
        // The quads face +z: from the front the nearer one is hit, from behind both are culled.
        for (const ray of gridRays(-1)) {
            const hit = tracer.closest(ray);
            expect(hit?.instanceIndex).toBe(0);
            expect(hit?.frontFace).toBe(true);
        }
        for (const ray of gridRays(1)) {
            expect(tracer.closest(ray)).toBeNull();
            expect(tracer.any(ray)).toBe(false);
        }

        // Random and mirrored transforms: the authored side has to agree with the reference's.
        const random = createRandom(9);
        const instances: TraceSceneInstance[] = Array.from({ length: 6 }, (_, index) => ({
            meshIndex: index % 2,
            instanceIndex: index,
            objectToWorld: randomAffine(random, { extent: 2, mirror: index >= 3 }),
        }));
        const shuffled = flaggedScene(instances, () => true).scene;
        const rays = randomRays(random, 4000, [-3, -3, -3, 3, 3, 3], 2);
        expectNoMismatches(compareWithReference(shuffled, instances, rays, { triangleFilter: singleSided }));
        // And it is a real test: with no flags set (every triangle blocks), the answers differ.
        const unfiltered = compareWithReference(flaggedScene(instances, () => false).scene, instances, rays, { triangleFilter: singleSided });
        expect(unfiltered.closest.mismatches).toBeGreaterThan(rays.length * 0.01);
    });

    it('passes the query kind: a filter that passes only for any-hit makes a "no shadow" surface', () => {
        // The front quad casts no shadow: closest-hit still sees it, any-hit goes through it.
        const { scene } = flaggedScene(STACKED, (meshIndex) => meshIndex === 0);
        const noShadow: TrianglePasses = (candidate) => candidate.instance === 0 && candidate.query === BVH_QUERY_ANY;
        const tracer = createCpuTracer(scene, { triangleFilter: noShadow });
        // From z = 5 down to z = 0.5: the front quad (t = 4) is in range, the back one (t = 6) is not.
        for (const ray of gridRays(-1, 4.5)) {
            expect(tracer.closest(ray)?.instanceIndex).toBe(0);
            expect(tracer.any(ray)).toBe(false);
        }
        // With the back quad in range, any-hit is blocked by it.
        for (const ray of gridRays(-1)) {
            expect(tracer.closest(ray)?.instanceIndex).toBe(0);
            expect(tracer.any(ray)).toBe(true);
        }
        const rays = [...gridRays(-1, 4.5), ...gridRays(1, 4.5), ...randomRays(createRandom(11), 2000, [-1, -1, -1, 1, 1, 1], 3)];
        expectNoMismatches(compareWithReference(scene, STACKED, rays, { triangleFilter: noShadow }));
    });
});

describe('flagged triangles without a filter', () => {
    it('block, as in a kernel with DEFAULT_TRIANGLE_FILTER_WGSL', () => {
        const { scene } = flaggedScene(STACKED, () => true);
        const tracer = createCpuTracer(scene);
        for (const ray of gridRays(-1)) {
            expect(tracer.closest(ray)?.instanceIndex).toBe(0);
            expect(tracer.any(ray)).toBe(true);
        }
        const rays = randomRays(createRandom(10), 2000, [-1, -1, -1, 1, 1, 1], 3);
        expectNoMismatches(compareWithReference(scene, STACKED, rays));
    });

    it('and unflagged triangles never reach the filter', () => {
        const { scene } = flaggedScene(STACKED, () => false);
        let calls = 0;
        const tracer = createCpuTracer(scene, { triangleFilter: () => (calls++, true) });
        for (const ray of gridRays(-1)) expect(tracer.closest(ray)?.instanceIndex).toBe(0);
        expect(calls).toBe(0);
    });
});

describe('the reference asks the filter only about real hits', () => {
    it('never for a near miss, a point outside the triangle, or outside the ray interval', () => {
        const instances = [STACKED[0]];
        const reference = buildReferenceScene([QUAD], instances, {
            triangleFilter: (candidate) => {
                const [u, v] = candidate.barycentrics;
                expect(Math.min(u, v, 1 - u - v)).toBeGreaterThanOrEqual(0);
                expect(candidate.t).toBeGreaterThanOrEqual(0);
                expect(candidate.t).toBeLessThanOrEqual(4.0001);
                return true;
            },
        });
        // Rays just outside the quad's border, rays that end just short of it, and rays through it.
        const rays: TraceRay[] = [];
        for (let i = 0; i < 50; i++) {
            const along = -1 + (i + 0.5) / 25;
            rays.push(toFloat32Ray([1.0000001 + 1e-5, along, 5], [0, 0, -1], 0, 1e6));
            rays.push(toFloat32Ray([along, -1.00001, 5], [0, 0, -1], 0, 1e6));
            rays.push(toFloat32Ray([along * 0.9, along * 0.7, 5], [0, 0, -1], 0, 3.9999));
            rays.push(toFloat32Ray([along * 0.9, along * 0.7, 5], [0, 0, -1], 0, 4.0001));
        }
        for (const ray of rays) {
            traceAnyReference(reference, ray);
            traceClosestReference(reference, ray);
        }
    });
});

describe('ray masks on the CPU walk against the reference', () => {
    /** Instance masks covering the corners: none, single bits, several, all. */
    const INSTANCE_MASKS = [0x00, 0x01, 0x02, 0x80, 0x03, 0xff, 0x5a, undefined];
    /** Ray masks covering the corners: 0, single bits, all ones, and bits above the instance mask's 8. */
    const RAY_MASKS = [undefined, RAY_MASK_ALL, 0, 0x01, 0x02, 0x80, 0x0f, 0xff, 0x100, 0xffffff00];

    const random = createRandom(12);
    const instances: TraceSceneInstance[] = INSTANCE_MASKS.map((mask, index) => ({
        meshIndex: index % 2,
        instanceIndex: 100 + index,
        objectToWorld: randomAffine(random, { extent: 2, mirror: index % 3 === 0 }),
        mask,
    }));
    const rays = randomRays(random, 1500, [-3, -3, -3, 3, 3, 3], 2);

    it('skips instances whose mask shares no bit with the ray mask, and agrees with the reference', () => {
        const { scene } = flaggedScene(instances, () => false);
        expectNoMismatches(compareWithReference(scene, instances, rays, { masks: RAY_MASKS }));

        const tracer = createCpuTracer(scene);
        const hits = (mask: number | undefined) => new Set(rays.map((ray) => tracer.closest(ray, { mask })?.instanceIndex).filter((index) => index !== undefined));
        // Mask 0 sees nothing, and neither does a mask with only bits above the instance mask's 8.
        for (const mask of [0, 0x100, 0xffffff00]) {
            expect(hits(mask).size).toBe(0);
            expect(rays.some((ray) => tracer.any(ray, { mask }))).toBe(false);
        }
        // All ones (given or by default) sees every instance but the one with mask 0.
        const visible = instances.filter((instance) => instance.mask !== 0).map((instance) => instance.instanceIndex).sort();
        expect([...hits(undefined)].sort()).toEqual(visible);
        expect([...hits(RAY_MASK_ALL)].sort()).toEqual(visible);
        // A single bit sees exactly the instances that have it.
        const withBit1 = instances.filter((instance) => ((instance.mask ?? 0xff) & 0x02) !== 0).map((instance) => instance.instanceIndex).sort();
        expect([...hits(0x02)].sort()).toEqual(withBit1);
    });

    it('hands the trace mask to the filter (all ones without one)', () => {
        const { scene } = flaggedScene(instances, (_mesh, original) => original % 3 === 0);
        const filter: TrianglePasses = (candidate) => candidate.triangle % 3 === 0 && (candidate.rayMask & 0x02) !== 0;
        expectNoMismatches(compareWithReference(scene, instances, rays, { triangleFilter: filter, masks: RAY_MASKS }));
    });

    it('with rayMasks: false ignores instance masks and hands the filter all ones, as the reference does', () => {
        const { scene } = flaggedScene(instances, (_mesh, original) => original % 3 === 0);
        let maskSeen = 0;
        const filter: TrianglePasses = (candidate) => ((maskSeen |= candidate.rayMask), candidate.triangle % 3 === 0 && candidate.rayMask === RAY_MASK_ALL);
        expectNoMismatches(compareWithReference(scene, instances, rays, { triangleFilter: filter, rayMasks: false, masks: [undefined, 0, 0x01] }));
        expect(maskSeen >>> 0).toBe(RAY_MASK_ALL);

        // Every instance is hit, the mask-0 one included, whatever mask a call passes.
        const tracer = createCpuTracer(scene, { rayMasks: false });
        const hit = new Set(rays.map((ray) => tracer.closest(ray, { mask: 0 })?.instanceIndex).filter((index) => index !== undefined));
        expect([...hit].sort()).toEqual(instances.map((instance) => instance.instanceIndex).sort());
    });
});

describe('the triangle filter and ray masks in the WGSL walk', () => {
    const call = 'bvhTrianglePasses(candidate)';

    it('calls the filter from both walks, for flagged triangles only, with the query and the mask', () => {
        const module = traceModuleWgsl({ triangleFilter: null });
        expect(module.split(call).length - 1).toBe(2);
        expect(module).toContain(`const BVH_TRIANGLE_FLAG_FILTER: u32 = ${TRIANGLE_FLAG_FILTER}u;`);
        expect(module.split('(triangle.flags & BVH_TRIANGLE_FLAG_FILTER) != 0u').length - 1).toBe(2);
        expect(module).toContain('frontFace, BVH_QUERY_CLOSEST, rayMask,');
        expect(module).toContain('frontFace, BVH_QUERY_ANY, rayMask,');
        expect(module).toContain(`const BVH_QUERY_CLOSEST: u32 = ${BVH_QUERY_CLOSEST}u;`);
        expect(module).toContain(`const BVH_QUERY_ANY: u32 = ${BVH_QUERY_ANY}u;`);
        // With triangleFilter: null the kernel supplies the filter.
        expect(module).not.toContain('fn bvhTrianglePasses(');
        expect(module).toContain('struct BvhCandidate {');
    });

    it('includes the default filter unless told otherwise, and takes another', () => {
        expect(DEFAULT_TRIANGLE_FILTER_WGSL).toContain(TRIANGLE_FILTER_SIGNATURE_WGSL);
        const byDefault = traceModuleWgsl();
        expect(byDefault.split('fn bvhTrianglePasses(').length - 1).toBe(1);
        expect(byDefault).toContain(DEFAULT_TRIANGLE_FILTER_WGSL);
        for (const mode of ['closest', 'any'] as const) {
            for (const rayMasks of [true, false]) {
                const kernel = batchKernelWgsl(mode, { rayMasks });
                expect(kernel.split('fn bvhTrianglePasses(').length - 1).toBe(1);
                expect(kernel).toContain(DEFAULT_TRIANGLE_FILTER_WGSL);
                expect(wgslNameProblems(kernel)).toEqual([]);
            }
        }
        const custom = `${TRIANGLE_FILTER_SIGNATURE_WGSL} {\n    return candidate.query == BVH_QUERY_ANY && !candidate.frontFace;\n}\n`;
        for (const kernel of [batchKernelWgsl('closest', { triangleFilter: custom }), traceModuleWgsl({ triangleFilter: custom })]) {
            expect(kernel).toContain(custom);
            expect(kernel).not.toContain(DEFAULT_TRIANGLE_FILTER_WGSL);
            expect(kernel.split('fn bvhTrianglePasses(').length - 1).toBe(1);
        }
        expect(traceModuleWgsl({ traceSlot: { group: 2, firstBinding: 5 } })).toContain('@group(2) @binding(7) var<storage, read> bvhTraceTopLevel');
    });

    it('with ray masks (the default) has the masked functions and one mask test, before entering an instance', () => {
        const module = traceModuleWgsl();
        expect(module).toContain('fn bvhTraceClosest(ray: BvhRay) -> BvhHit {\n    return bvhWalkClosest(ray, BVH_RAY_MASK_ALL);');
        expect(module).toContain('fn bvhTraceAny(ray: BvhRay) -> bool {\n    return bvhWalkAny(ray, BVH_RAY_MASK_ALL);');
        expect(module).toContain('fn bvhTraceClosestMasked(ray: BvhRay, mask: u32) -> BvhHit {');
        expect(module).toContain('fn bvhTraceAnyMasked(ray: BvhRay, mask: u32) -> bool {');
        expect(module.split('if ((bvhInstanceMask(instance) & rayMask) != 0u) {').length - 1).toBe(2);
        // The test sits between loading the instance and moving the ray into it.
        const walk = module.slice(module.indexOf('fn bvhWalkClosest('));
        const load = walk.indexOf('instance = bvhLoadInstance(index);');
        expect(load).toBeGreaterThan(0);
        expect(walk.indexOf('bvhInstanceMask(instance) & rayMask')).toBeGreaterThan(load);
        expect(walk.indexOf('bvhInstanceMask(instance) & rayMask')).toBeLessThan(walk.indexOf('setup = bvhMakeRaySetup('));
    });

    it('with rayMasks: false emits no masked functions and no mask test, and the plain functions still exist', () => {
        const module = traceModuleWgsl({ rayMasks: false });
        expect(module).not.toContain('Masked');
        expect(module).not.toContain('bvhInstanceMask');
        expect(module).not.toContain('BVH_INSTANCE_MASK');
        expect(module).not.toMatch(/rayMask\s*[&)]/);
        expect(module).not.toContain('rayMask: u32)');
        expect(module).toContain('fn bvhTraceClosest(ray: BvhRay) -> BvhHit {\n    return bvhWalkClosest(ray);');
        expect(module).toContain('fn bvhTraceAny(ray: BvhRay) -> bool {\n    return bvhWalkAny(ray);');
        // The filter is still told a mask: all ones.
        expect(module).toContain('frontFace, BVH_QUERY_CLOSEST, BVH_RAY_MASK_ALL,');
        // The batch kernel has no mask uniform either.
        const batch = batchKernelWgsl('any', { rayMasks: false });
        expect(batch).not.toContain('@binding(2) var<uniform>');
        expect(batch).toContain('bvhTraceAny(bvhBatchRays[rayIndex])');
        const masked = batchKernelWgsl('any');
        expect(masked).toContain('@group(1) @binding(2) var<uniform> bvhBatchParams: BvhBatchParams;');
        expect(masked).toContain('bvhTraceAnyMasked(bvhBatchRays[rayIndex], bvhBatchParams.rayMask)');
    });

    it('throws a TypeError, naming traceSlot, on the old positional slot and filter arguments', () => {
        const old = { group: 0, firstBinding: 0 } as never;
        expect(() => traceModuleWgsl(old)).toThrow(TypeError);
        expect(() => traceModuleWgsl(old)).toThrow(/traceSlot/);
        expect(() => traceModuleWgsl({ firstBinding: 3 } as never)).toThrow(/traceSlot/);
        expect(() => batchKernelWgsl('closest', old)).toThrow(TypeError);
        expect(() => batchKernelWgsl('closest', old)).toThrow(/traceSlot/);
        const call = batchKernelWgsl as (...args: unknown[]) => string;
        expect(() => call('any', {}, 'fn bvhTrianglePasses() {}')).toThrow(TypeError);
        expect(() => call('any', { group: 1, firstBinding: 0 }, undefined)).toThrow(/traceSlot/);
    });

    it('keeps every name it declares in the bvh namespace', () => {
        for (const rayMasks of [true, false]) {
            const module = traceModuleWgsl({ rayMasks, triangleFilter: null });
            const names = [...module.matchAll(/^(?:fn|struct|const|@group\(\d+\) @binding\(\d+\) var<[^>]*>)\s+([A-Za-z_]\w*)/gm)].map((match) => match[1]);
            expect(names.length).toBeGreaterThan(20);
            expect(names.filter((name) => !/^(bvh|Bvh|BVH_)/.test(name))).toEqual([]);
        }
    });
});

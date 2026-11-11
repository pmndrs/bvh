/**
 * The walk's triangle filter on the CPU: the CPU port of the walk, with the
 * filter flag written by `writeTriangleFilterFlags`, against the brute-force
 * reference given the same filter; and the hook in the generated WGSL.
 */

import { describe, expect, it } from 'vitest';
import { createTinyBvhBuilder } from '../index.ts';
import { wgslNameProblems } from '../wgsl-names.ts';
import { createCpuTracer } from './cpu-traversal.ts';
import { TRIANGLE_FLAG_FILTER, TRIANGLE_WORD, TRIANGLE_WORDS } from './layout.ts';
import type { MeshBvh, MeshGeometry } from './mesh-bvh.ts';
import { buildMeshBvhWithTinyBvh } from './mesh-bvh-topology.ts';
import { addToTally, compareAny, compareClosest, newTally } from './oracle-compare.ts';
import { createRandom, randomAffine, randomRays, tessellatedQuad, toFloat32Ray } from './oracle-fixtures.ts';
import type { TraceRay } from './rays.ts';
import { buildReferenceScene, traceAnyReference, traceClosestReference } from './reference-tracer.ts';
import { assembleTraceScene, type TraceSceneData, type TraceSceneInstance } from './scene-assembly.ts';
import { NO_TRIANGLE_FILTER_WGSL, writeTriangleFilterFlags, type TriangleFilter } from './triangle-filter.ts';
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

/** Both meshes' trees, mesh `m`'s records flagged where `flagged(m, original)` says. */
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
function gridRays(direction: 1 | -1): TraceRay[] {
    const rays: TraceRay[] = [];
    for (let row = 0; row < 8; row++) {
        for (let column = 0; column < 8; column++) {
            // Quarter-cell offsets: never on a cell edge or a cell's diagonal.
            const x = -1 + (column + 0.3) * 0.25;
            const y = -1 + (row + 0.6) * 0.25;
            rays.push(toFloat32Ray([x, y, -5 * direction], [0, 0, direction], 0, 1e6));
        }
    }
    return rays;
}

function compareWithReference(scene: TraceSceneData, instances: readonly TraceSceneInstance[], rays: readonly TraceRay[], triangleFilter?: TriangleFilter) {
    const tracer = createCpuTracer(scene, { triangleFilter });
    const reference = buildReferenceScene([QUAD, QUAD], instances, { triangleFilter });
    const closest = newTally();
    const any = newTally();
    rays.forEach((ray, index) => {
        const hit = tracer.closest(ray);
        const traced = hit ? { hit: true, ...hit } : null;
        addToTally(closest, compareClosest(traced, traceClosestReference(reference, ray), ray), `ray ${index}`);
        addToTally(any, compareAny(tracer.any(ray), traceAnyReference(reference, ray)), `ray ${index}`);
    });
    return { closest, any };
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
});

describe('the CPU walk with a triangle filter against the filtered reference', () => {
    it('lets rays through a filtered-out front quad to the one behind it', () => {
        const { scene, flaggedCount } = flaggedScene(STACKED, (meshIndex) => meshIndex === 0);
        expect(flaggedCount).toBe(QUAD_TRIANGLES);
        const filter: TriangleFilter = (instanceIndex) => instanceIndex === 0;
        const tracer = createCpuTracer(scene, { triangleFilter: filter });
        for (const ray of gridRays(-1)) {
            const hit = tracer.closest(ray);
            expect(hit?.instanceIndex).toBe(1);
            expect(hit?.t).toBeCloseTo(6, 5);
            expect(tracer.any(ray)).toBe(true);
        }

        const rays = [...gridRays(-1), ...gridRays(1), ...randomRays(createRandom(7), 2000, [-1, -1, -1, 1, 1, 1], 3)];
        const { closest, any } = compareWithReference(scene, STACKED, rays, filter);
        expect(closest.mismatches, closest.examples.join('\n')).toBe(0);
        expect(any.mismatches, any.examples.join('\n')).toBe(0);
    });

    it('filters per triangle: only the flagged half of the front quad lets rays through', () => {
        const flagged = (meshIndex: number, original: number) => meshIndex === 0 && original % 2 === 0;
        const { scene } = flaggedScene(STACKED, flagged);
        // The reference asks about every triangle, so its filter says no where the walk has no flag.
        const filter: TriangleFilter = (instanceIndex, triangleIndex) => flagged(instanceIndex, triangleIndex);
        const tracer = createCpuTracer(scene, { triangleFilter: filter });
        const seen = new Set<number>();
        for (const ray of gridRays(-1)) {
            const hit = tracer.closest(ray)!;
            seen.add(hit.instanceIndex);
            if (hit.instanceIndex === 0) expect(hit.triangleIndex % 2).toBe(1);
        }
        expect([...seen].sort()).toEqual([0, 1]);

        const rays = [...gridRays(-1), ...randomRays(createRandom(8), 2000, [-1, -1, -1, 1, 1, 1], 3)];
        const { closest, any } = compareWithReference(scene, STACKED, rays, filter);
        expect(closest.mismatches, closest.examples.join('\n')).toBe(0);
        expect(any.mismatches, any.examples.join('\n')).toBe(0);
    });

    it('culls back faces with a filter on frontFace (single-sided), mirrored instances included', () => {
        const singleSided: TriangleFilter = (_instance, _triangle, hit) => !hit.frontFace;
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
        const { closest, any } = compareWithReference(shuffled, instances, rays, singleSided);
        expect(closest.mismatches, closest.examples.join('\n')).toBe(0);
        expect(any.mismatches, any.examples.join('\n')).toBe(0);
        // And it is a real test: with no flags set (every triangle blocks), the answers differ.
        const unfiltered = compareWithReference(flaggedScene(instances, () => false).scene, instances, rays, singleSided);
        expect(unfiltered.closest.mismatches).toBeGreaterThan(rays.length * 0.01);
    });
});

describe('flagged triangles without a filter', () => {
    it('block, as in a kernel with NO_TRIANGLE_FILTER_WGSL', () => {
        const { scene } = flaggedScene(STACKED, () => true);
        const tracer = createCpuTracer(scene);
        for (const ray of gridRays(-1)) {
            expect(tracer.closest(ray)?.instanceIndex).toBe(0);
            expect(tracer.any(ray)).toBe(true);
        }
        const rays = randomRays(createRandom(10), 2000, [-1, -1, -1, 1, 1, 1], 3);
        const { closest, any } = compareWithReference(scene, STACKED, rays);
        expect(closest.mismatches, closest.examples.join('\n')).toBe(0);
        expect(any.mismatches, any.examples.join('\n')).toBe(0);
    });

    it('and unflagged triangles never reach the filter', () => {
        const { scene } = flaggedScene(STACKED, () => false);
        let calls = 0;
        const tracer = createCpuTracer(scene, { triangleFilter: () => (calls++, true) });
        for (const ray of gridRays(-1)) expect(tracer.closest(ray)?.instanceIndex).toBe(0);
        expect(calls).toBe(0);
    });
});

describe('the triangle filter in the WGSL walk', () => {
    const call = 'giTriangleFilter(instance.instanceIndex, triangle.originalIndex, candidate.t, candidate.barycentrics, candidate.frontFacing)';
    const declaration = 'fn giTriangleFilter(instance: u32, triangle: u32, t: f32, barycentrics: vec2f, frontFace: bool) -> bool';

    it('is called from both walks, for flagged triangles only', () => {
        const module = traceModuleWgsl({ group: 0, firstBinding: 0 });
        expect(module.split(call).length - 1).toBe(2);
        expect(module).toContain(`const GI_TRIANGLE_FLAG_FILTER: u32 = ${TRIANGLE_FLAG_FILTER}u;`);
        expect(module.split('(triangle.flags & GI_TRIANGLE_FLAG_FILTER) != 0u').length - 1).toBe(2);
        // The module only calls it; the kernel supplies it.
        expect(module).not.toContain('fn giTriangleFilter(');
    });

    it('defaults to the no-filter piece, which declares the signature the walk calls, and takes another', () => {
        expect(NO_TRIANGLE_FILTER_WGSL).toContain(declaration);
        for (const mode of ['closest', 'any'] as const) {
            const kernel = batchKernelWgsl(mode);
            expect(kernel.split('fn giTriangleFilter(').length - 1).toBe(1);
            expect(kernel).toContain(NO_TRIANGLE_FILTER_WGSL);
            expect(wgslNameProblems(kernel)).toEqual([]);
        }
        const custom = `${declaration} {\n    return !frontFace;\n}\n`;
        const kernel = batchKernelWgsl('closest', { group: 0, firstBinding: 0 }, custom);
        expect(kernel).toContain(custom);
        expect(kernel).not.toContain(NO_TRIANGLE_FILTER_WGSL);
        expect(kernel.split('fn giTriangleFilter(').length - 1).toBe(1);
    });
});

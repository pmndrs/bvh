import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { buildLbvhOnCpu, checkGpuLayout, createTinyBvhBuilder } from '../index.ts';
import { createCpuTracer, type CpuHit } from './cpu-traversal.ts';
import {
    INSTANCE_FLAG,
    INSTANCE_RECORD_WORDS,
    INSTANCE_WORD,
    MATERIAL_BITS,
    MATERIAL_SHIFT,
    MAX_MATERIAL,
    MAX_MERGED_SOURCES,
    NO_MATERIAL,
    TOP_LEVEL_HEADER_WORDS,
    TRIANGLE_FLAG_FILTER,
    TRIANGLE_FLAG_FLIP_FRONT_FACE,
    TRIANGLE_WORD,
    TRIANGLE_WORDS,
    materialFromFlags,
} from './layout.ts';
import type { MeshBvh, MeshGeometry } from './mesh-bvh.ts';
import { buildMeshBvhWithTinyBvh } from './mesh-bvh-topology.ts';
import { addToTally, compareAny, compareClosest, newTally, NORMAL_COSINE_TOLERANCE, normalCosine } from './oracle-compare.ts';
import { createRandom, randomAffine, randomRays, randomTriangleSoup, tessellatedQuad, toFloat32Ray } from './oracle-fixtures.ts';
import { buildReferenceScene, traceAnyReference, traceClosestReference } from './reference-tracer.ts';
import { assembleTraceScene, buildTraceScene, meshRefitPlacement, type TraceSceneData, type TraceSceneInstance } from './scene-assembly.ts';
import { MERGED_INSTANCE_INDEX, mergeStaticGeometry, writeMergedTriangleIdentity } from './static-merge.ts';
import { buildTopLevel, IDENTITY_TRANSFORM } from './top-level.ts';
import { buildTopLevelShape, TopLevelShapeKeeper } from './top-level-shape.ts';
import { writeTriangleFilterFlags, type TrianglePasses } from './triangle-filter.ts';

const tinyBvh = await createTinyBvhBuilder();
const build = (geometry: MeshGeometry): MeshBvh => buildMeshBvhWithTinyBvh(geometry, tinyBvh);

const withFlags = (geometry: MeshGeometry, every: number): MeshGeometry => {
    const count = geometry.indices ? geometry.indices.length / 3 : geometry.positions.length / 9;
    return { ...geometry, triangleFlags: Uint32Array.from({ length: count }, (_, t) => (t % every === 0 ? TRIANGLE_FLAG_FILTER : 0)) };
};

/**
 * A mixed scene: unique static meshes (mirrored, non-uniform scales, three masks, filtered
 * triangles, materials on most, the largest one included), a static mesh used twice (must
 * not merge), and non-static instances.
 */
function mixedScene(seed: number): { meshes: MeshGeometry[]; instances: TraceSceneInstance[] } {
    const random = createRandom(seed);
    const meshes: MeshGeometry[] = [];
    const instances: TraceSceneInstance[] = [];
    const masks = [undefined, 0x0f, 0xf0];
    for (let i = 0; i < 18; i++) {
        const mesh = i % 3 === 0 ? tessellatedQuad(4 + (i % 4), 1.5) : randomTriangleSoup(random, 60 + i * 5, 2, 0.6);
        meshes.push(i % 4 === 1 ? withFlags(mesh, 3) : mesh);
        instances.push({
            meshIndex: i,
            instanceIndex: 900 - i * 13,
            objectToWorld: randomAffine(random, { extent: 8, mirror: i % 3 === 1 }),
            static: i % 6 !== 5,
            mask: masks[i % 3],
            material: i % 4 === 3 ? undefined : i === 2 ? MAX_MATERIAL : i * 101,
        });
    }
    // One mesh used twice, both static: stays instanced.
    meshes.push(withFlags(randomTriangleSoup(random, 80, 2, 0.6), 2));
    for (let k = 0; k < 2; k++) {
        instances.push({ meshIndex: 18, instanceIndex: 10 + k, objectToWorld: randomAffine(random, { extent: 8, mirror: k === 1 }), static: true, material: k === 0 ? 7 : undefined });
    }
    return { meshes, instances };
}

/**
 * A filter that answers once per (instance, triangle) and per query, and sees the side. It only
 * lets rays through flagged triangles (meshes 1, 5, 9, 13, 17 and 18), as the reference asks it about all.
 */
const FLAGGED_INSTANCES = new Set([900 - 13, 900 - 5 * 13, 900 - 9 * 13, 900 - 13 * 13, 900 - 17 * 13, 10, 11]);
const filter: TrianglePasses = ({ instance, triangle, frontFace, query }) =>
    FLAGGED_INSTANCES.has(instance) && triangle % (instance < 20 ? 2 : 3) === 0 && (instance * 7 + triangle) % 3 !== 1 ? query === 0 || !frontFace : false;

function sameHit(a: CpuHit | null, b: CpuHit | null): string | null {
    if (a === null || b === null) return a === b ? null : `one missed: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`;
    if (a.instanceIndex !== b.instanceIndex || a.triangleIndex !== b.triangleIndex || a.frontFace !== b.frontFace) {
        return `${a.instanceIndex}/${a.triangleIndex}/${a.frontFace} vs ${b.instanceIndex}/${b.triangleIndex}/${b.frontFace}`;
    }
    if (Math.abs(a.t - b.t) > 1e-4 * Math.max(1, a.t)) return `t ${a.t} vs ${b.t}`;
    if (a.material !== b.material) return `material ${a.material} vs ${b.material}`;
    if (!(normalCosine(a.normal, b.normal) >= NORMAL_COSINE_TOLERANCE)) return `normal ${a.normal.join(', ')} vs ${b.normal.join(', ')}`;
    return null;
}

describe('mergeStaticGeometry', () => {
    it('merges static instances of meshes used once, grouped by flags and mask, and leaves the rest', () => {
        const { meshes, instances } = mixedScene(1);
        const plan = mergeStaticGeometry(meshes, instances);
        const merged = new Set(plan.groups.flatMap((group) => group.sources.map((source) => source.inputIndex)));
        instances.forEach((instance, input) => {
            const expected = instance.static === true && instance.meshIndex !== 18;
            expect(merged.has(input), `instance ${input}`).toBe(expected);
            expect(plan.instanced.includes(input)).toBe(!expected);
        });
        // Three masks, all visible and contributing: three groups.
        expect(plan.groups.map((group) => group.mask).sort()).toEqual([0x0f, 0xf0, 0xff]);
        expect(plan.meshNeeded[18]).toBe(1);
        expect(plan.meshNeeded[0]).toBe(0);
        expect(plan.meshNeeded[5]).toBe(1);
    });

    it('transforms each vertex once, so shared vertices keep identical bits', () => {
        const quad = tessellatedQuad(5, 1);
        const instance: TraceSceneInstance = { meshIndex: 0, instanceIndex: 3, objectToWorld: randomAffine(createRandom(4), { extent: 50, mirror: true }), static: true };
        const [group] = mergeStaticGeometry([quad], [instance]).groups;
        expect(group.geometry.positions.length).toBe(quad.positions.length);
        expect(Array.from(group.geometry.indices!)).toEqual(Array.from(quad.indices!));
        expect(group.flipFrontFace.every((bit) => bit === 1)).toBe(true);
    });

    it('splits a group that would pass the triangle limit', () => {
        const { meshes, instances } = mixedScene(2);
        const plan = mergeStaticGeometry(meshes, instances, { maxTrianglesPerGroup: 200 });
        for (const group of plan.groups) expect(group.sourceInstances.length).toBeLessThanOrEqual(200);
        expect(plan.groups.length).toBeGreaterThan(3);
    });

    it('caps a group at 65,536 sources for an indexed scene only: records group as main did', () => {
        const triangle: MeshGeometry = { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]) };
        const count = MAX_MERGED_SOURCES + 3;
        const meshes = Array.from({ length: count }, () => triangle);
        const instances: TraceSceneInstance[] = meshes.map((_, index) => ({
            meshIndex: index,
            instanceIndex: index,
            objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, index, 0, 0, 1],
            static: true,
        }));
        const sizes = (triangleLeaves?: 'records' | 'indexed') => mergeStaticGeometry(meshes, instances, { triangleLeaves }).groups.map((group) => group.sources.length);
        expect(sizes()).toEqual([count]);
        expect(sizes('records')).toEqual([count]);
        expect(sizes('indexed')).toEqual([MAX_MERGED_SOURCES, 3]);
    });
});

describe('merged static scene against the instanced scene', () => {
    const { meshes, instances } = mixedScene(7);
    const instanced = assembleTraceScene(meshes.map(build), instances);
    const plan = mergeStaticGeometry(meshes, instances);
    const merged = assembleTraceScene(
        meshes.map((mesh, index) => (plan.meshNeeded[index] ? build(mesh) : null)),
        instances,
        {},
        { merged: { plan, meshes: plan.groups.map((group) => build(group.geometry)) } },
    );
    const rays = randomRays(createRandom(8), 4000, [-10, -10, -10, 10, 10, 10], 3, { randomInterval: true });
    const masks = [undefined, 0x0f, 0xf0, 0x01, 0];

    it('writes merged records and the triangles\' source ids, flip bits and filter bits', () => {
        const top = new Uint32Array(merged.topLevel.buffer);
        let mergedRecords = 0;
        for (let record = 0; record < merged.topLevel.instanceCount; record++) {
            const base = TOP_LEVEL_HEADER_WORDS + record * INSTANCE_RECORD_WORDS;
            if (top[base + INSTANCE_WORD.flags] & INSTANCE_FLAG.merged) {
                mergedRecords++;
                expect(top[base + INSTANCE_WORD.instanceIndex]).toBe(MERGED_INSTANCE_INDEX);
                expect(top[base + INSTANCE_WORD.flags] & INSTANCE_FLAG.mirrored).toBe(0);
                expect(merged.topLevel.recordOrder[record]).toBeGreaterThanOrEqual(instances.length);
            }
        }
        expect(mergedRecords).toBe(plan.groups.length);
        // Merged records carry no material; normal records carry their instance's.
        for (let record = 0; record < merged.topLevel.instanceCount; record++) {
            const flags = top[TOP_LEVEL_HEADER_WORDS + record * INSTANCE_RECORD_WORDS + INSTANCE_WORD.flags];
            const input = merged.topLevel.recordOrder[record];
            expect(flags & MATERIAL_BITS).toBe(flags & INSTANCE_FLAG.merged ? 0 : (((instances[input].material ?? -1) + 1) << MATERIAL_SHIFT) >>> 0);
        }
        expect(merged.topLevel.instanceCount).toBe(plan.instanced.length + plan.groups.length);
        const words = new Uint32Array(merged.triangles);
        for (const [g, group] of merged.mergedGroups.entries()) {
            for (let slot = group.triangleOffset; slot < group.triangleOffset + group.triangleCount; slot++) {
                const base = slot * TRIANGLE_WORDS;
                const source = plan.groups[g].sources.find((s) => s.instanceIndex === words[base + TRIANGLE_WORD.padding])!;
                expect(source).toBeDefined();
                const sourceMesh = meshes[source.meshIndex];
                const original = words[base + TRIANGLE_WORD.originalIndex];
                const sourceFlags = sourceMesh.triangleFlags?.[original] ?? 0;
                const material = source.material === NO_MATERIAL ? 0 : source.material + 1;
                expect(words[base + TRIANGLE_WORD.flags]).toBe((sourceFlags | (source.mirrored ? TRIANGLE_FLAG_FLIP_FRONT_FACE : 0) | (material << MATERIAL_SHIFT)) >>> 0);
                expect(materialFromFlags(words[base + TRIANGLE_WORD.flags])).toBe(instances[source.inputIndex].material ?? NO_MATERIAL);
            }
        }
    });

    it('gives the same instance, triangle and side, and t within tolerance, for every unambiguous ray', { timeout: 60_000 }, () => {
        for (const rayMasks of [true, false]) {
            const a = createCpuTracer(instanced, { triangleFilter: filter, rayMasks });
            const b = createCpuTracer(merged, { triangleFilter: filter, rayMasks });
            const reference = buildReferenceScene(meshes, instances, { triangleFilter: filter, rayMasks });
            let compared = 0;
            let hits = 0;
            const problems: string[] = [];
            for (const mask of masks) {
                rays.forEach((ray, index) => {
                    const expected = traceClosestReference(reference, ray, { mask });
                    if (expected?.ambiguous) return;
                    const left = a.closest(ray, { mask });
                    const problem = sameHit(left, b.closest(ray, { mask }));
                    if (problem) problems.push(`ray ${index} mask ${mask}: ${problem}`);
                    const occlusion = traceAnyReference(reference, ray, { mask });
                    if (!occlusion.ambiguous && a.any(ray, { mask }) !== b.any(ray, { mask })) problems.push(`ray ${index} mask ${mask}: any-hit differs`);
                    compared++;
                    if (left) hits++;
                });
            }
            expect(problems.slice(0, 10).join('\n')).toBe('');
            expect(compared).toBeGreaterThan(masks.length * 3900);
            // Materials are on most hits, and some have none.
            const materials = new Set(rays.map((ray) => b.closest(ray)?.material).filter((material) => material !== undefined));
            expect(materials.has(NO_MATERIAL) && materials.has(MAX_MATERIAL) && materials.size > 5).toBe(true);
            expect(hits).toBeGreaterThan(1000);
        }
    });

    it('the CPU walk on the merged scene agrees with the reference, filter and masks included', { timeout: 60_000 }, () => {
        const tracer = createCpuTracer(merged, { triangleFilter: filter });
        const reference = buildReferenceScene(meshes, instances, { triangleFilter: filter });
        const closest = newTally();
        const any = newTally();
        for (const mask of masks) {
            rays.forEach((ray, index) => {
                const hit = tracer.closest(ray, { mask });
                addToTally(closest, compareClosest(hit && { hit: true, ...hit }, traceClosestReference(reference, ray, { mask }), ray), `ray ${index}`);
                addToTally(any, compareAny(tracer.any(ray, { mask }), traceAnyReference(reference, ray, { mask })), `ray ${index}`);
            });
        }
        expect(closest.mismatches, closest.examples.join('\n')).toBe(0);
        expect(any.mismatches, any.examples.join('\n')).toBe(0);
    });

    it('passes the layout check before and after the ids are written, and only with the identity given after', () => {
        plan.groups.forEach((group, g) => {
            const tree = build(group.geometry);
            expect(checkGpuLayout(tree, group.geometry)).toEqual([]);
            const placement = merged.mergedGroups[g];
            const stamped = { ...tree, triangles: merged.triangles.slice(placement.triangleOffset * 48, (placement.triangleOffset + placement.triangleCount) * 48) };
            expect(checkGpuLayout(stamped, group.geometry, { merged: group })).toEqual([]);
            expect(checkGpuLayout(stamped, group.geometry).length).toBeGreaterThan(0);
        });
    });

    it('gives the filter the source instance', () => {
        const seen = new Set<number>();
        const tracer = createCpuTracer(merged, {
            triangleFilter: (candidate) => {
                seen.add(candidate.instance);
                return filter(candidate);
            },
        });
        for (const ray of rays) tracer.closest(ray);
        const mergedSources = new Set(plan.groups.flatMap((group) => group.sources.map((source) => source.instanceIndex)));
        expect(seen.has(MERGED_INSTANCE_INDEX)).toBe(false);
        expect([...seen].some((instance) => mergedSources.has(instance))).toBe(true);
    });

    it('lets no ray aimed at a shared edge of a merged mirrored quad slip through', () => {
        const quad = tessellatedQuad(8, 2);
        const objectToWorld = randomAffine(createRandom(12), { extent: 20, mirror: true });
        const scene = { meshes: [quad], instances: [{ meshIndex: 0, instanceIndex: 4, objectToWorld, static: true }] };
        const mergePlan = mergeStaticGeometry(scene.meshes, scene.instances);
        const data = assembleTraceScene([null], scene.instances, {}, { merged: { plan: mergePlan, meshes: [build(mergePlan.groups[0].geometry)] } });
        const tracer = createCpuTracer(data);
        const world = mergePlan.groups[0].geometry.positions;
        const indices = quad.indices!;
        const random = createRandom(13);
        let slips = 0;
        let rayCount = 0;
        for (let triangle = 0; triangle < indices.length / 3; triangle += 2) {
            // Each cell is (a, b, d) then (a, d, c): the diagonal a-d is shared by both halves.
            const a = indices[triangle * 3] * 3;
            const c = indices[triangle * 3 + 2] * 3;
            for (let k = 0; k < 8; k++) {
                const s = random();
                const target = [0, 1, 2].map((axis) => world[a + axis] + s * (world[c + axis] - world[a + axis]));
                const origin = target.map((value) => value + (random() * 2 - 1) * 5);
                const direction = target.map((value, axis) => value - origin[axis]);
                const length = Math.hypot(...direction);
                const ray = toFloat32Ray(origin, direction.map((d) => d / length), 0, 1e6);
                rayCount++;
                const hit = tracer.closest(ray);
                if (!hit || hit.instanceIndex !== 4) slips++;
            }
        }
        expect(rayCount).toBeGreaterThan(400);
        expect(slips).toBe(0);
    });
});

describe('merged normals and materials', () => {
    it('match the instanced scene for mirrored, scaled and sheared sources, with and without a material', () => {
        // One tilted triangle per instance, each its own mesh, all static: they merge into one group.
        const triangle = { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 1]) };
        const transforms = [
            [2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], // non-uniform scale
            [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 0, 0, 1], // mirror
            [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 20, 0, 0, 1], // shear of x into z
            [-1, 0, 0, 0, 0, 3, 0, 0, 0, 0.5, 1, 0, 30, 0, 0, 1], // mirrored, scaled and sheared
        ];
        const materials = [5, undefined, MAX_MATERIAL, 0];
        const meshes = transforms.map(() => triangle);
        const instances: TraceSceneInstance[] = transforms.map((objectToWorld, i) => ({ meshIndex: i, instanceIndex: 40 + i, objectToWorld, static: true, material: materials[i] }));
        const instanced = createCpuTracer(assembleTraceScene(meshes.map(build), instances));
        const plan = mergeStaticGeometry(meshes, instances);
        expect(plan.groups.length).toBe(1);
        expect(Array.from(plan.groups[0].sourceMaterials)).toEqual(materials.map((material) => (material === undefined ? 0 : material + 1)));
        const merged = createCpuTracer(assembleTraceScene(meshes.map(() => null), instances, {}, { merged: { plan, meshes: plan.groups.map((group) => build(group.geometry)) } }));
        const reference = buildReferenceScene(meshes, instances);
        let hits = 0;
        for (const [i, e] of transforms.entries()) {
            // Aim at the object point 0.2 v1 + 0.15 v2, moved to world space.
            const p = [0.2, 0.15, 0.15];
            const target = [0, 1, 2].map((r) => e[r] * p[0] + e[4 + r] * p[1] + e[8 + r] * p[2] + e[12 + r]);
            for (const direction of [[0, 0, -1], [0, 0, 1], [0.3, -0.2, -1]] as const) {
                const length = Math.hypot(...direction);
                const unit = direction.map((d) => d / length) as [number, number, number];
                const ray = { origin: target.map((value, axis) => value - 5 * unit[axis]) as [number, number, number], direction: unit, tMin: 0, tMax: 100 };
                const a = instanced.closest(ray);
                const b = merged.closest(ray);
                expect(sameHit(a, b), `instance ${i} direction ${direction}`).toBeNull();
                if (!b) continue;
                hits++;
                expect(b.material).toBe(materials[i] ?? NO_MATERIAL);
                expect(b.frontFace).toBe(b.normal[0] * unit[0] + b.normal[1] * unit[1] + b.normal[2] * unit[2] < 0);
                expect(compareClosest({ hit: true, ...b }, traceClosestReference(reference, ray), ray).verdict).toBe('match');
            }
        }
        expect(hits).toBe(12);
    });
});

describe('mergeStaticInstances off', () => {
    const bytes = (scene: TraceSceneData): number[] => [scene.nodes, scene.triangles, scene.topLevel.buffer].flatMap((buffer) => Array.from(new Uint8Array(buffer)));

    it('gives the bytes main gave before merging existed (sha-256 pinned from 8c5d1b8)', async () => {
        const random = createRandom(42);
        const meshes: MeshGeometry[] = [];
        const instances: TraceSceneInstance[] = [];
        for (let i = 0; i < 30; i++) {
            meshes.push(i % 3 === 0 ? tessellatedQuad(5, 1) : randomTriangleSoup(random, 100, 3, 0.5));
            instances.push({ meshIndex: i % 25, instanceIndex: 1000 - i, objectToWorld: randomAffine(random, { extent: 10, mirror: i % 4 === 0 }), mask: i % 2 ? 0x0f : undefined, static: true });
        }
        const scene = await buildTraceScene(meshes, instances, { build });
        const hash = createHash('sha256');
        for (const buffer of [scene.nodes, scene.triangles, scene.topLevel.buffer]) hash.update(new Uint8Array(buffer));
        hash.update(new Uint8Array(scene.topLevel.recordOrder.buffer));
        expect(hash.digest('hex')).toBe('d9637acbc4729725a624009c54848cd80b65a72ad79887a9c1a93f33127abf52');
    });

    it('builds exactly the instanced scene, static or not', async () => {
        const { meshes, instances } = mixedScene(3);
        const today = assembleTraceScene(meshes.map(build), instances.map(({ static: _, ...rest }) => rest));
        const off = await buildTraceScene(meshes, instances, { build });
        expect(bytes(off)).toEqual(bytes(today));
        expect(Array.from(off.topLevel.recordOrder)).toEqual(Array.from(today.topLevel.recordOrder));
        expect(off.mergedGroups).toEqual([]);
        const on = await buildTraceScene(meshes, instances, { build, mergeStaticInstances: true });
        expect(on.mergedGroups.length).toBe(3);
    });
});

describe('a merged instance never moves', () => {
    const { meshes, instances } = mixedScene(5);
    const plan = mergeStaticGeometry(meshes, instances);
    const trees = { plan, meshes: plan.groups.map((group) => build(group.geometry)) };
    const sourceMeshes = meshes.map((mesh, index) => (plan.meshNeeded[index] ? build(mesh) : null));
    const moved = plan.groups[0].sources[0];

    it('assembling with a changed transform throws, naming the instance', () => {
        const changed = instances.map((instance, input) =>
            input === moved.inputIndex ? { ...instance, objectToWorld: Array.from(instance.objectToWorld, (v, k) => (k === 12 ? v + 1 : v)) } : instance,
        );
        expect(() => assembleTraceScene(sourceMeshes, changed, {}, { merged: trees })).toThrow(new RegExp(`instance ${moved.instanceIndex} is a merged static instance`));
        // Unchanged, it assembles.
        expect(() => assembleTraceScene(sourceMeshes, instances, {}, { merged: trees })).not.toThrow();
    });

    it('a top-level rebuild or a top-level shape with a moved merged record throws', () => {
        const record = { instanceIndex: 77, nodeOffset: 0, triangleOffset: 0, meshBounds: [0, 0, 0, 1, 1, 1], merged: true };
        expect(() => buildTopLevel([{ ...record, objectToWorld: IDENTITY_TRANSFORM }])).not.toThrow();
        const shifted = IDENTITY_TRANSFORM.map((v, k) => (k === 13 ? 2 : v));
        expect(() => buildTopLevel([{ ...record, objectToWorld: shifted }])).toThrow(/instance 77 is a merged static instance/);
        const input = () => ({
            slots: Uint32Array.of(77),
            visible: Uint8Array.of(1),
            objectToWorld: Float64Array.from(shifted),
            meshBounds: Float32Array.of(0, 0, 0, 1, 1, 1),
            merged: Uint8Array.of(1),
        });
        expect(() => buildTopLevelShape(input())).toThrow(/instance 77/);
        const keeper = new TopLevelShapeKeeper();
        keeper.build({ ...input(), objectToWorld: Float64Array.from(IDENTITY_TRANSFORM) }, 1);
        expect(() => keeper.check(input(), 1.5, 1, 2)).toThrow(/instance 77/);
    });
});

describe('triangle flags: only bit 0 comes from the input', () => {
    const quad = tessellatedQuad(3, 1);
    const count = quad.indices!.length / 3;
    const flagsWith = (triangle: number, value: number) => Uint32Array.from({ length: count }, (_, t) => (t === triangle ? value : t % 2));

    it('every builder throws, naming the triangle, for bit 1 or a material bit', () => {
        for (const value of [TRIANGLE_FLAG_FLIP_FRONT_FACE, 1 << 16, 0x80000001]) {
            const geometry = { ...quad, triangleFlags: flagsWith(5, value) };
            expect(() => build(geometry)).toThrow(/triangleFlags\[5\] is 0x[0-9a-f]+; only bit 0/);
            expect(() => tinyBvh.buildLayout(geometry)).toThrow(/triangleFlags\[5\]/);
            expect(() => buildLbvhOnCpu(geometry)).toThrow(/triangleFlags\[5\]/);
        }
        // Bit 0 alone is fine.
        expect(() => build({ ...quad, triangleFlags: flagsWith(5, TRIANGLE_FLAG_FILTER) })).not.toThrow();
    });

    it('buildTraceScene and mergeStaticGeometry name the mesh (and the instance)', async () => {
        const meshes = [quad, { ...quad, triangleFlags: flagsWith(2, TRIANGLE_FLAG_FLIP_FRONT_FACE) }];
        const instances: TraceSceneInstance[] = [0, 1].map((meshIndex) => ({ meshIndex, instanceIndex: 40 + meshIndex, objectToWorld: IDENTITY_TRANSFORM, static: true }));
        await expect(buildTraceScene(meshes, instances, { build })).rejects.toThrow(/mesh 1: triangleFlags\[2\]/);
        await expect(buildTraceScene(meshes, instances, { build, mergeStaticInstances: true })).rejects.toThrow(/mesh 1: triangleFlags\[2\]/);
        expect(() => mergeStaticGeometry(meshes, instances)).toThrow(/mesh 1 \(instance 41\): triangleFlags\[2\]/);
    });

    it('writing the merged ids refuses records with reserved bits, so it cannot run twice', () => {
        const instance: TraceSceneInstance = { meshIndex: 0, instanceIndex: 3, objectToWorld: randomAffine(createRandom(21), { extent: 5, mirror: true }), static: true };
        const [group] = mergeStaticGeometry([quad], [instance]).groups;
        const records = build(group.geometry).triangles.slice(0);
        writeMergedTriangleIdentity(records, group);
        expect(() => writeMergedTriangleIdentity(records, group)).toThrow(/has flags 0x2: only bit 0/);
    });
});

describe('a stale merge plan', () => {
    const { meshes, instances } = mixedScene(11);
    const plan = mergeStaticGeometry(meshes, instances);
    const trees = { plan, meshes: plan.groups.map((group) => build(group.geometry)) };
    const sourceMeshes = meshes.map((mesh, index) => (plan.meshNeeded[index] ? build(mesh) : null));
    const source = plan.groups.find((group) => group.mask === 0xff)!.sources[0];
    const changed = (change: Partial<TraceSceneInstance>) => instances.map((instance, input) => (input === source.inputIndex ? { ...instance, ...change } : instance));

    it('throws, naming the instance, when a source changed its material', () => {
        // The source has a material (mixedScene gives input 0 one); dropping it is a change too.
        expect(source.material).not.toBe(NO_MATERIAL);
        for (const change of [{ material: source.material + 1 }, { material: undefined }]) {
            expect(() => assembleTraceScene(sourceMeshes, changed(change), {}, { merged: trees }), JSON.stringify(change)).toThrow(
                new RegExp(`instance ${source.instanceIndex} is a merged static instance, but its material changed`),
            );
        }
        expect(() => assembleTraceScene(sourceMeshes, changed({ material: source.material }), {}, { merged: trees })).not.toThrow();
    });

    it('throws, naming the instance, when a source was hidden, left GI, or changed its mask', () => {
        for (const change of [{ visible: false }, { contributesToGi: false }, { mask: 0x0f }, { mask: 0 }]) {
            expect(() => assembleTraceScene(sourceMeshes, changed(change), {}, { merged: trees }), JSON.stringify(change)).toThrow(
                new RegExp(`instance ${source.instanceIndex} is a merged static instance, but its mask, visible or contributesToGi changed`),
            );
        }
        // Spelling out the defaults is not a change.
        expect(() => assembleTraceScene(sourceMeshes, changed({ visible: true, contributesToGi: true, mask: 0xff }), {}, { merged: trees })).not.toThrow();
    });
});

describe('planning leaves out what the option-off scene would not hit', () => {
    it('keeps hidden, non-GI and empty static instances as instances, and refuses shared indices', () => {
        const random = createRandom(31);
        const meshes = [0, 1, 2, 3].map(() => randomTriangleSoup(random, 20, 2, 0.5));
        meshes.push({ positions: new Float32Array(0) });
        const instances: TraceSceneInstance[] = meshes.map((_, meshIndex) => ({ meshIndex, instanceIndex: meshIndex, objectToWorld: randomAffine(random, { extent: 4 }), static: true }));
        instances[1] = { ...instances[1], visible: false };
        instances[2] = { ...instances[2], contributesToGi: false };
        const plan = mergeStaticGeometry(meshes, instances);
        expect(Array.from(plan.instanced)).toEqual([1, 2, 4]);
        expect(plan.groups.length).toBe(1);
        expect(plan.groups[0].sources.map((s) => s.inputIndex)).toEqual([0, 3]);
        expect(Array.from(plan.meshNeeded)).toEqual([0, 1, 1, 0, 1]);
        // The scene the option gives traces like the instanced one: the hidden instances are skipped either way.
        const merged = assembleTraceScene(meshes.slice(0, 4).map((mesh, index) => (plan.meshNeeded[index] ? build(mesh) : null)), instances.slice(0, 4), {}, {
            merged: { plan: mergeStaticGeometry(meshes.slice(0, 4), instances.slice(0, 4)), meshes: [build(plan.groups[0].geometry)] },
        });
        expect(merged.topLevel.instanceCount).toBe(1);

        const twice = instances.slice(0, 4).map((instance) => ({ ...instance, visible: true, contributesToGi: true, instanceIndex: instance.meshIndex === 3 ? 0 : instance.instanceIndex }));
        expect(() => mergeStaticGeometry(meshes.slice(0, 4), twice)).toThrow(/positions 0 and 3 both have instanceIndex 0/);
    });

    it('the layout check names a merged identity whose sources share an index instead of reporting bad slots', () => {
        const quad = tessellatedQuad(2, 1);
        const instances: TraceSceneInstance[] = [0, 1].map((meshIndex) => ({ meshIndex, instanceIndex: 5 + meshIndex, objectToWorld: IDENTITY_TRANSFORM.map((v, k) => (k === 12 ? v + meshIndex * 3 : v)), static: true }));
        const [group] = mergeStaticGeometry([quad, quad], instances).groups;
        const records = build(group.geometry);
        const stamped = records.triangles.slice(0);
        writeMergedTriangleIdentity(stamped, group);
        expect(checkGpuLayout({ ...records, triangles: stamped }, group.geometry, { merged: group })).toEqual([]);
        const shared = { ...group, sourceInstances: group.sourceInstances.map(() => 5) };
        expect(checkGpuLayout({ ...records, triangles: stamped }, group.geometry, { merged: shared })).toEqual([
            `source instance 5 starts at merged triangles 0 and ${quad.indices!.length / 3}; merged sources need distinct instance indices`,
        ]);
    });
});

describe('filter flags written after assembly on a merged scene', () => {
    // No flags in the input: they are all written into the scenes' buffers afterwards.
    const { meshes: flaggedMeshes, instances } = mixedScene(17);
    const meshes = flaggedMeshes.map(({ positions, indices }) => ({ positions, indices }));
    const flagged = (meshIndex: number, original: number) => (meshIndex * 7 + original) % 3 === 0;
    const meshOfInstance = new Map(instances.map((instance) => [instance.instanceIndex, instance.meshIndex]));
    const plan = mergeStaticGeometry(meshes, instances);
    const instanced = assembleTraceScene(meshes.map(build), instances);
    const merged = assembleTraceScene(
        meshes.map((mesh, index) => (plan.meshNeeded[index] ? build(mesh) : null)),
        instances,
        {},
        { merged: { plan, meshes: plan.groups.map((group) => build(group.geometry)) } },
    );
    // Every flagged triangle passes: a triangle the flags miss blocks, and the scenes differ.
    const passAll: TrianglePasses = () => true;

    it('flags a merged range by source instance, so the filtered walk matches the instanced scene', () => {
        let instancedFlagged = 0;
        instanced.meshes.forEach((range, meshIndex) => {
            instancedFlagged += writeTriangleFilterFlags(instanced.triangles, (original, sourceInstance) => {
                expect(sourceInstance).toBeUndefined();
                return flagged(meshIndex, original);
            }, range);
        });
        let mergedFlagged = 0;
        merged.meshes.forEach((range, meshIndex) => {
            if (range.mergedInto === undefined) mergedFlagged += writeTriangleFilterFlags(merged.triangles, (original) => flagged(meshIndex, original), range);
            else expect(() => writeTriangleFilterFlags(merged.triangles, () => true, range)).toThrow(new RegExp(`merged into static group ${range.mergedInto}.*mergedGroups\\[${range.mergedInto}\\]`));
        });
        // Without the merged ranges flagged, alpha-tested merged geometry goes opaque.
        const unflaggedGroups = { ...merged, triangles: merged.triangles.slice(0) };
        for (const group of merged.mergedGroups) {
            mergedFlagged += writeTriangleFilterFlags(merged.triangles, (original, sourceInstance) => {
                const meshIndex = group.sourceMeshes[group.sourceInstances.indexOf(sourceInstance!)];
                expect(meshIndex).toBe(meshOfInstance.get(sourceInstance!));
                return flagged(meshIndex, original);
            }, group);
        }
        expect(mergedFlagged).toBe(instancedFlagged);
        expect(merged.meshes.some((range) => range.mergedInto !== undefined)).toBe(true);

        const a = createCpuTracer(instanced, { triangleFilter: passAll });
        const b = createCpuTracer(merged, { triangleFilter: passAll });
        const reference = buildReferenceScene(flaggedMeshes, instances, { triangleFilter: ({ instance, triangle }) => flagged(meshOfInstance.get(instance)!, triangle) });
        const opaque = createCpuTracer(unflaggedGroups, { triangleFilter: passAll });
        const unfiltered = createCpuTracer(instanced);
        const problems: string[] = [];
        let passed = 0;
        let opaqueDiffers = 0;
        randomRays(createRandom(18), 3000, [-10, -10, -10, 10, 10, 10], 3, { randomInterval: true }).forEach((ray, index) => {
            const expected = traceClosestReference(reference, ray);
            if (expected?.ambiguous) return;
            const left = a.closest(ray);
            const problem = sameHit(left, b.closest(ray));
            if (problem) problems.push(`ray ${index}: ${problem}`);
            if (a.any(ray) !== b.any(ray)) problems.push(`ray ${index}: any-hit differs`);
            if (sameHit(left, unfiltered.closest(ray)) !== null) passed++;
            if (sameHit(left, opaque.closest(ray)) !== null) opaqueDiffers++;
        });
        expect(problems.slice(0, 10).join('\n')).toBe('');
        // The flags made rays pass somewhere, and leaving the merged ranges unflagged shows, so the comparison bites.
        expect(passed).toBeGreaterThan(50);
        expect(opaqueDiffers).toBeGreaterThan(40);
    });
});

describe('merged meshes are never refit targets', () => {
    it('meshRefitPlacement refuses a merged mesh and gives word offsets for the rest', () => {
        const { meshes, instances } = mixedScene(19);
        const plan = mergeStaticGeometry(meshes, instances);
        const scene = assembleTraceScene(
            meshes.map((mesh, index) => (plan.meshNeeded[index] ? build(mesh) : null)),
            instances,
            {},
            { merged: { plan, meshes: plan.groups.map((group) => build(group.geometry)) } },
        );
        const mergedMesh = scene.meshes.findIndex((range) => range.mergedInto !== undefined);
        expect(() => meshRefitPlacement(scene, mergedMesh)).toThrow(new RegExp(`mesh ${mergedMesh} was merged into static group \\d+.*must not be refit`));
        expect(meshRefitPlacement(scene, 18)).toEqual({ nodeWordOffset: scene.meshes[18].nodeOffset * 16, triangleWordOffset: scene.meshes[18].triangleOffset * TRIANGLE_WORDS });
    });
});

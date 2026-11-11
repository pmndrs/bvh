import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { checkGpuLayout, createTinyBvhBuilder } from '../index.ts';
import { createCpuTracer, type CpuHit } from './cpu-traversal.ts';
import { INSTANCE_FLAG, INSTANCE_RECORD_WORDS, INSTANCE_WORD, TOP_LEVEL_HEADER_WORDS, TRIANGLE_FLAG_FILTER, TRIANGLE_FLAG_FLIP_FRONT_FACE, TRIANGLE_WORD, TRIANGLE_WORDS } from './layout.ts';
import type { MeshBvh, MeshGeometry } from './mesh-bvh.ts';
import { buildMeshBvhWithTinyBvh } from './mesh-bvh-topology.ts';
import { addToTally, compareAny, compareClosest, newTally } from './oracle-compare.ts';
import { createRandom, randomAffine, randomRays, randomTriangleSoup, tessellatedQuad, toFloat32Ray } from './oracle-fixtures.ts';
import { buildReferenceScene, traceAnyReference, traceClosestReference } from './reference-tracer.ts';
import { assembleTraceScene, buildTraceScene, type TraceSceneData, type TraceSceneInstance } from './scene-assembly.ts';
import { MERGED_INSTANCE_INDEX, mergeStaticGeometry } from './static-merge.ts';
import { buildTopLevel, IDENTITY_TRANSFORM } from './top-level.ts';
import { buildTopLevelShape, TopLevelShapeKeeper } from './top-level-shape.ts';
import type { TrianglePasses } from './triangle-filter.ts';

const tinyBvh = await createTinyBvhBuilder();
const build = (geometry: MeshGeometry): MeshBvh => buildMeshBvhWithTinyBvh(geometry, tinyBvh);

const withFlags = (geometry: MeshGeometry, every: number): MeshGeometry => {
    const count = geometry.indices ? geometry.indices.length / 3 : geometry.positions.length / 9;
    return { ...geometry, triangleFlags: Uint32Array.from({ length: count }, (_, t) => (t % every === 0 ? TRIANGLE_FLAG_FILTER : 0)) };
};

/**
 * A mixed scene: unique static meshes (mirrored, non-uniform scales, three masks, filtered
 * triangles), a static mesh used twice (must not merge), and non-static instances.
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
        });
    }
    // One mesh used twice, both static: stays instanced.
    meshes.push(withFlags(randomTriangleSoup(random, 80, 2, 0.6), 2));
    for (let k = 0; k < 2; k++) {
        instances.push({ meshIndex: 18, instanceIndex: 10 + k, objectToWorld: randomAffine(random, { extent: 8, mirror: k === 1 }), static: true });
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
                expect(words[base + TRIANGLE_WORD.flags]).toBe(sourceFlags | (source.mirrored ? TRIANGLE_FLAG_FLIP_FRONT_FACE : 0));
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

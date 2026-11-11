/**
 * The CPU mesh tree against the brute-force reference (reference-tracer.ts,
 * Möller-Trumbore in doubles over every triangle): on random rays through
 * triangle soups, a mirrored copy, a tessellated grid with shared edges and
 * degenerate triangles, `raycast` must return the reference's hits (every
 * one, on the side asked for, within near/far) and `raycastFirst` its closest.
 * A hit only one side finds must sit within rounding distance of an edge or
 * of the interval's end. Then refit, saving, the box walk and the lookups.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import { GPU_LAYOUT_EMPTY_CHILD, GPU_LAYOUT_NODE_WORD, GPU_LAYOUT_NODE_WORDS } from '../gpu-layout.ts';
import type { MeshInput } from '../mesh-input.ts';
import { createTinyBvhBuilder, type TinyBvhBuilder } from '../tinybvh.ts';
import { createRandom, randomAffine, randomTriangleSoup, randomUnitVector, tessellatedQuad, type Random } from '../trace/oracle-fixtures.ts';
import { buildReferenceScene, REFERENCE_TOLERANCE, traceClosestReference } from '../trace/reference-tracer.ts';
import { IDENTITY_TRANSFORM } from '../trace/top-level.ts';
import {
    BVH_BACK_SIDE,
    BVH_CONTAINED,
    BVH_DOUBLE_SIDE,
    BVH_FRONT_SIDE,
    BVH_INTERSECTED,
    BVH_NOT_INTERSECTED,
    CpuMeshBvh,
    type BvhRaycastHit,
} from './cpu-mesh-bvh.ts';

let builder: TinyBvhBuilder;
beforeAll(async () => {
    builder = await createTinyBvhBuilder();
});

/** Applies a column-major affine to every position (a mirror reverses the winding). */
function transformed(input: MeshInput, matrix: readonly number[]): MeshInput {
    const positions = new Float32Array(input.positions.length);
    for (let i = 0; i < positions.length; i += 3) {
        const [x, y, z] = [input.positions[i], input.positions[i + 1], input.positions[i + 2]];
        for (let axis = 0; axis < 3; axis++) positions[i + axis] = matrix[axis] * x + matrix[4 + axis] * y + matrix[8 + axis] * z + matrix[12 + axis];
    }
    return { positions, indices: input.indices };
}

/** A soup with degenerate triangles mixed in: repeated corners, collinear corners, a point. */
function soupWithDegenerates(random: Random, count: number): MeshInput {
    const soup = randomTriangleSoup(random, count, 4, 0.8);
    const positions = soup.positions;
    for (let triangle = 0; triangle < count; triangle += 17) {
        const base = triangle * 9;
        const kind = (triangle / 17) % 3;
        if (kind === 0) positions.copyWithin(base + 3, base, base + 3); // corner 1 = corner 0
        if (kind === 1) for (let axis = 0; axis < 3; axis++) positions[base + 6 + axis] = 2 * positions[base + 3 + axis] - positions[base + axis]; // collinear
        if (kind === 2) for (let corner = 1; corner < 3; corner++) positions.copyWithin(base + corner * 3, base, base + 3); // a point
    }
    return { positions };
}

function boundsOf(positions: Float32Array): [number, number, number, number, number, number] {
    const box: [number, number, number, number, number, number] = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    for (let i = 0; i < positions.length; i += 3) {
        for (let axis = 0; axis < 3; axis++) {
            box[axis] = Math.min(box[axis], positions[i + axis]);
            box[axis + 3] = Math.max(box[axis + 3], positions[i + axis]);
        }
    }
    return box;
}

interface TestRay {
    origin: [number, number, number];
    direction: [number, number, number];
}

/** Random rays from around the box, some axis-aligned (zero direction components), some not unit length. */
function testRays(random: Random, count: number, box: readonly number[]): TestRay[] {
    const rays: TestRay[] = [];
    for (let i = 0; i < count; i++) {
        const origin = [0, 1, 2].map((axis) => Math.fround(box[axis] - 1 + random() * (box[axis + 3] - box[axis] + 2))) as [number, number, number];
        let direction = randomUnitVector(random);
        if (i % 7 === 0) direction = [0, 0, 0].map((_, axis) => (axis === i % 3 ? (random() < 0.5 ? -1 : 1) : 0)) as [number, number, number];
        const scale = i % 5 === 0 ? 0.25 + random() * 3 : 1;
        rays.push({ origin, direction: direction.map((value) => Math.fround(value * scale)) as [number, number, number] });
    }
    return rays;
}

interface ReferenceHitAll {
    t: number;
    triangle: number;
    u: number;
    v: number;
    frontFace: boolean;
}

/** Every reference hit in `[tMin, tMax]` (the reference asks its filter about each real hit; this one collects them all). */
function referenceAll(input: MeshInput, ray: TestRay, tMin: number, tMax: number): ReferenceHitAll[] {
    const hits: ReferenceHitAll[] = [];
    const scene = buildReferenceScene([input], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY_TRANSFORM }], {
        triangleFilter: (candidate) => {
            hits.push({ t: candidate.t, triangle: candidate.triangle, u: candidate.barycentrics[0], v: candidate.barycentrics[1], frontFace: candidate.frontFace });
            return true;
        },
    });
    traceClosestReference(scene, { origin: ray.origin, direction: ray.direction, tMin, tMax });
    return hits;
}

const sideAccepts = (side: number, frontFace: boolean) => side === BVH_DOUBLE_SIDE || (side === BVH_FRONT_SIDE) === frontFace;
const edgeMargin = (u: number, v: number) => Math.min(u, v, 1 - u - v);
const nearEnd = (t: number, tMin: number, tMax: number) => Math.abs(t - tMin) <= REFERENCE_TOLERANCE * Math.max(1, t) || Math.abs(t - tMax) <= REFERENCE_TOLERANCE * Math.max(1, t);

/** Our hits against the reference's for one ray and side; returns problems, counts edge-only differences. */
function compare(tree: CpuMeshBvh, input: MeshInput, ray: TestRay, side: number, near: number, far: number, problems: string[], label: string): { edgeOnly: number; hits: number } {
    const length = Math.hypot(...ray.direction);
    const tMin = near / length;
    const tMax = far / length;
    const expected = referenceAll(input, ray, tMin, tMax).filter((hit) => sideAccepts(side, hit.frontFace));
    const ours = tree.raycast(ray, { side, near, far });
    let edgeOnly = 0;
    const ourByTriangle = new Map(ours.map((hit) => [hit.triangle, hit]));
    const expectedTriangles = new Set(expected.map((hit) => hit.triangle));
    for (const hit of ours) {
        if (expectedTriangles.has(hit.triangle)) continue;
        if (edgeMargin(hit.u, hit.v) < REFERENCE_TOLERANCE || nearEnd(hit.t, tMin, tMax)) edgeOnly++;
        else problems.push(`${label}: only ours hit triangle ${hit.triangle} at t ${hit.t}`);
    }
    for (const want of expected) {
        const got = ourByTriangle.get(want.triangle);
        if (!got) {
            if (edgeMargin(want.u, want.v) < REFERENCE_TOLERANCE || nearEnd(want.t, tMin, tMax)) edgeOnly++;
            else problems.push(`${label}: only the reference hit triangle ${want.triangle} at t ${want.t}`);
            continue;
        }
        if (Math.abs(got.t - want.t) > 1e-4 * Math.max(1, want.t)) problems.push(`${label}: t ${got.t} vs ${want.t}`);
        if (Math.abs(got.u - want.u) > 1e-4 || Math.abs(got.v - want.v) > 1e-4) problems.push(`${label}: barycentrics differ on ${want.triangle}`);
        if (got.frontFace !== want.frontFace) problems.push(`${label}: side differs on ${want.triangle}`);
        if (Math.abs(got.distance - got.t * length) > 1e-6 * Math.max(1, got.distance)) problems.push(`${label}: distance is not t * |d|`);
        const point = ray.origin.map((o, axis) => o + ray.direction[axis] * got.t);
        if (Math.hypot(...point.map((p, axis) => p - got.point[axis])) > 1e-6 * Math.max(1, got.distance)) problems.push(`${label}: point differs`);
    }
    for (let i = 1; i < ours.length; i++) if (ours[i].t < ours[i - 1].t) problems.push(`${label}: hits not sorted`);
    // raycastFirst: the closest the reference finds, unless the closest is ambiguous.
    const first = tree.raycastFirst(ray, side, near, far);
    const closest = expected.reduce<ReferenceHitAll | null>((best, hit) => (!best || hit.t < best.t ? hit : best), null);
    if (closest && first && first.triangle !== closest.triangle) {
        const tie = expected.some((hit) => hit !== closest && hit.t - closest.t <= 1e-5 * Math.max(1, closest.t));
        if (!tie && edgeMargin(closest.u, closest.v) >= REFERENCE_TOLERANCE && edgeMargin(first.u, first.v) >= REFERENCE_TOLERANCE) problems.push(`${label}: first hit ${first.triangle}, reference ${closest.triangle}`);
    } else if (!closest !== !first) {
        const lone = (closest ?? first) as ReferenceHitAll | BvhRaycastHit;
        if (edgeMargin(lone.u, lone.v) >= REFERENCE_TOLERANCE && !nearEnd(lone.t, tMin, tMax)) problems.push(`${label}: first hit on one side only`);
    }
    if (first && ours.length > 0 && first.t > ours[0].t + 1e-9) problems.push(`${label}: raycastFirst is not the first of raycast`);
    return { edgeOnly, hits: expected.length };
}

function checkAgainstReference(tree: CpuMeshBvh, input: MeshInput, seed: number, rayCount: number): { problems: string[]; hits: number; edgeOnly: number } {
    const random = createRandom(seed);
    const problems: string[] = [];
    let hits = 0;
    let edgeOnly = 0;
    const rays = testRays(random, rayCount, boundsOf(input.positions));
    rays.forEach((ray, r) => {
        const side = [BVH_FRONT_SIDE, BVH_BACK_SIDE, BVH_DOUBLE_SIDE][r % 3];
        const near = r % 4 === 0 ? random() * 2 : 0;
        const far = r % 4 === 1 ? near + random() * 6 : Infinity;
        const result = compare(tree, input, ray, side, near, far, problems, `ray ${r} side ${side}`);
        hits += result.hits;
        edgeOnly += result.edgeOnly;
    });
    return { problems, hits, edgeOnly };
}

describe('CpuMeshBvh', () => {
    it('raycast and raycastFirst match the brute-force reference: soup with degenerate triangles, its mirror, a grid with shared edges', () => {
        const random = createRandom(7);
        const soup = soupWithDegenerates(random, 600);
        const mirrored = transformed(soup, randomAffine(random, { extent: 2, mirror: true }));
        const grid = transformed(tessellatedQuad(24, 3), randomAffine(random, { extent: 1 }));
        for (const [name, input, seed] of [['soup', soup, 11], ['mirrored', mirrored, 12], ['grid', grid, 13]] as const) {
            const tree = CpuMeshBvh.build(input, builder);
            const { problems, hits, edgeOnly } = checkAgainstReference(tree, input, seed, 900);
            expect(problems.slice(0, 5), name).toEqual([]);
            expect(hits, name).toBeGreaterThan(60);
            expect(edgeOnly, name).toBeLessThan(hits * 0.01);
        }
    });

    it('takes three-style rays ({x, y, z}), a material-like { side }, and returns nothing for a zero direction', () => {
        const tree = CpuMeshBvh.build(tessellatedQuad(4, 1), builder);
        const ray = { origin: { x: 0.1, y: 0.2, z: 5 }, direction: { x: 0, y: 0, z: -1 } };
        expect(tree.raycastFirst(ray, { side: BVH_FRONT_SIDE })?.distance).toBeCloseTo(5, 6);
        expect(tree.raycastFirst(ray, BVH_BACK_SIDE)).toBeNull();
        expect(tree.raycast({ origin: [0.1, 0.2, -5], direction: [0, 0, 1] }, BVH_BACK_SIDE)).toHaveLength(1);
        expect(tree.raycast({ origin: [0.1, 0.2, 5], direction: [0, 0, 0] }, BVH_DOUBLE_SIDE)).toEqual([]);
        // A ray along the grid's plane, and one through a shared vertex: the watertight test finds one hit there.
        expect(tree.raycast({ origin: [-2, 0.1, 0], direction: [1, 0, 0] }, BVH_DOUBLE_SIDE)).toEqual([]);
        expect(tree.raycast({ origin: [0, 0, 1], direction: [0, 0, -1] }, BVH_DOUBLE_SIDE).length).toBeGreaterThanOrEqual(1);
    });

    it('refit after moving vertices gives the moved mesh\'s hits, in place or from new arrays', () => {
        const random = createRandom(21);
        const soup = randomTriangleSoup(random, 400, 3, 0.6);
        const tree = CpuMeshBvh.build(soup, builder);
        const before = tree.serialize();
        // Move every vertex, some far: the old boxes no longer hold the triangles.
        for (let i = 0; i < soup.positions.length; i++) soup.positions[i] += (random() - 0.5) * (i % 9 === 0 ? 6 : 0.8);
        tree.refit();
        const moved = checkAgainstReference(tree, soup, 22, 500);
        expect(moved.problems.slice(0, 5)).toEqual([]);
        expect(moved.hits).toBeGreaterThan(100);
        const box = tree.getBoundingBox();
        const want = boundsOf(soup.positions);
        for (let i = 0; i < 6; i++) expect(box[i]).toBe(Math.fround(want[i]));
        // The shape is kept: only boxes and corners changed.
        expect(tree.layout.nodeCount).toBe(CpuMeshBvh.deserialize(before).nodeCount);
        // New arrays (an indexed grid, its vertices lifted into a wave).
        const grid = tessellatedQuad(16, 2);
        const gridTree = CpuMeshBvh.build(grid, builder);
        const lifted = new Float32Array(grid.positions);
        for (let i = 0; i < lifted.length; i += 3) lifted[i + 2] = Math.sin(lifted[i] * 2) * 0.5 + Math.cos(lifted[i + 1] * 3) * 0.3;
        const liftedInput = { positions: lifted, indices: grid.indices };
        gridTree.refit(liftedInput);
        const wave = checkAgainstReference(gridTree, liftedInput, 23, 400);
        expect(wave.problems.slice(0, 5)).toEqual([]);
        expect(() => gridTree.refit({ positions: lifted, indices: grid.indices!.subarray(0, 6) })).toThrow(RangeError);
        expect(() => CpuMeshBvh.deserialize(before).refit()).toThrow(/no source/);
    });

    it('serialize and deserialize round-trip the tree, with the same hits', () => {
        const random = createRandom(31);
        const soup = randomTriangleSoup(random, 300, 3, 0.7);
        const tree = CpuMeshBvh.build(soup, builder);
        const bytes = tree.serialize();
        const loaded = CpuMeshBvh.deserialize(bytes, { source: soup });
        expect(new Uint8Array(loaded.serialize())).toEqual(new Uint8Array(bytes));
        expect(loaded.getBoundingBox()).toEqual(tree.getBoundingBox());
        for (const ray of testRays(random, 200, boundsOf(soup.positions))) {
            expect(loaded.raycast(ray, BVH_DOUBLE_SIDE)).toEqual(tree.raycast(ray, BVH_DOUBLE_SIDE));
        }
        expect(() => CpuMeshBvh.deserialize(new Uint8Array(bytes).subarray(0, 100))).toThrow();
    });

    it('traverse visits every leaf once, prunes by box, reports contained ranges, stops when asked, and orders by score', () => {
        const random = createRandom(41);
        const soup = randomTriangleSoup(random, 500, 4, 0.5);
        const tree = CpuMeshBvh.build(soup, builder, { maxLeafPrimitives: 4 });
        // Everything: each tree position exactly once, each leaf at most 4.
        const seen = new Uint8Array(tree.triangleCount);
        let rootAsked = 0;
        tree.traverse({
            intersectsBounds: (_box, isLeaf, _score, depth) => {
                if (depth === 0) rootAsked++;
                if (depth === 0) expect(isLeaf).toBe(false);
                return true;
            },
            intersectsRange: (start, count) => {
                expect(count).toBeLessThanOrEqual(4);
                for (let i = start; i < start + count; i++) seen[i]++;
                return false;
            },
        });
        expect(rootAsked).toBe(1);
        expect(seen.every((count) => count === 1)).toBe(true);
        // A box query: the triangles whose corners' box overlaps the query box, against brute force.
        const query = [-1, -0.5, -1.5, 1.2, 1, 0.5];
        const overlaps = (box: ArrayLike<number>) => box[0] <= query[3] && box[3] >= query[0] && box[1] <= query[4] && box[4] >= query[1] && box[2] <= query[5] && box[5] >= query[2];
        const inside = (box: ArrayLike<number>) => box[0] >= query[0] && box[3] <= query[3] && box[1] >= query[1] && box[4] <= query[4] && box[2] >= query[2] && box[5] <= query[5];
        const triangleBox = (corners: ArrayLike<number>) => [0, 1, 2, 3, 4, 5].map((i) => (i < 3 ? Math.min(corners[i], corners[i + 3], corners[i + 6]) : Math.max(corners[i - 3], corners[i], corners[i + 3])));
        const found = new Set<number>();
        let containedRanges = 0;
        tree.traverse({
            intersectsBounds: (box) => (inside(box) ? BVH_CONTAINED : overlaps(box) ? BVH_INTERSECTED : BVH_NOT_INTERSECTED),
            intersectsRange: (start, count, contained) => {
                if (contained) containedRanges++;
                for (let position = start; position < start + count; position++) {
                    const box = triangleBox(tree.getTriangle(position));
                    if (contained) expect(inside(box)).toBe(true);
                    if (overlaps(box)) found.add(tree.resolveTriangleIndex(position));
                }
                return false;
            },
        });
        const expected = new Set<number>();
        for (let triangle = 0; triangle < 500; triangle++) if (overlaps(triangleBox(soup.positions.subarray(triangle * 9, triangle * 9 + 9)))) expected.add(triangle);
        expect([...found].sort((a, b) => a - b)).toEqual([...expected].sort((a, b) => a - b));
        expect(containedRanges).toBeGreaterThan(0);
        // Stopping: the first range ends the walk.
        let ranges = 0;
        expect(tree.traverse({ intersectsBounds: () => true, intersectsRange: () => ++ranges === 3 })).toBe(true);
        expect(ranges).toBe(3);
        // Ordering: with a score of distance to a point, the first leaf reached is near it.
        const target = [3, 3, 3];
        const distance = (box: ArrayLike<number>) => Math.hypot(...[0, 1, 2].map((axis) => Math.max(box[axis] - target[axis], 0, target[axis] - box[axis + 3])));
        let firstLeafDistance = -1;
        const scores: number[] = [];
        tree.traverse({
            boundsTraverseOrder: distance,
            intersectsBounds: (_box, _isLeaf, score) => {
                scores.push(score!);
                return true;
            },
            intersectsRange: (_start, _count, _contained, _depth, box) => {
                firstLeafDistance = distance(box);
                return true;
            },
        });
        const nearestLeaf = Math.min(
            ...Array.from({ length: tree.nodeCount * 2 }, (_, slot) => {
                const words = new Uint32Array(tree.layout.nodes);
                const floats = new Float32Array(tree.layout.nodes);
                const base = (slot >> 1) * GPU_LAYOUT_NODE_WORDS + (slot & 1) * 8;
                const side = slot & 1;
                if (words[(slot >> 1) * GPU_LAYOUT_NODE_WORDS + (side ? GPU_LAYOUT_NODE_WORD.rightIndex : GPU_LAYOUT_NODE_WORD.leftIndex)] === GPU_LAYOUT_EMPTY_CHILD) return Infinity;
                if (words[(slot >> 1) * GPU_LAYOUT_NODE_WORDS + (side ? GPU_LAYOUT_NODE_WORD.rightCount : GPU_LAYOUT_NODE_WORD.leftCount)] === 0) return Infinity;
                return distance([floats[base], floats[base + 1], floats[base + 2], floats[base + 4], floats[base + 5], floats[base + 6]]);
            }),
        );
        expect(scores.every((score) => typeof score === 'number')).toBe(true);
        expect(firstLeafDistance).toBeLessThan(nearestLeaf + 2);
    });

    it('resolves tree positions, finds hit point info with attributes, reports memory, bounds and progress', () => {
        const grid = tessellatedQuad(8, 2);
        const progress: number[] = [];
        const tree = CpuMeshBvh.build(grid, builder, { maxLeafPrimitives: 2, onProgress: (p) => progress.push(p) });
        expect(progress).toEqual([0, 1]);
        const triangles = grid.indices!.length / 3;
        const resolved = Array.from({ length: triangles }, (_, position) => tree.resolveTriangleIndex(position));
        expect([...resolved].sort((a, b) => a - b)).toEqual(Array.from({ length: triangles }, (_, i) => i));
        for (let triangle = 0; triangle < triangles; triangle++) expect(tree.resolveTriangleIndex(tree.treePositionOf(triangle))).toBe(triangle);
        // UVs from the positions, so the interpolated uv must be the point's x, y.
        const vertexCount = grid.positions.length / 3;
        const uv = new Float32Array(vertexCount * 2);
        for (let vertex = 0; vertex < vertexCount; vertex++) uv.set([grid.positions[vertex * 3], grid.positions[vertex * 3 + 1]], vertex * 2);
        const hit = tree.raycastFirst({ origin: [0.37, -0.81, 3], direction: [0, 0, -1] })!;
        const info = tree.getTriangleHitPointInfo(hit.point, hit.triangle, { attributes: { uv: { array: uv, itemSize: 2 } } });
        expect(info.vertices).toEqual(Array.from(grid.indices!.subarray(hit.triangle * 3, hit.triangle * 3 + 3)));
        expect(info.barycoord![0]).toBeCloseTo(1 - hit.u - hit.v, 5);
        expect(info.barycoord![1]).toBeCloseTo(hit.u, 5);
        expect(info.barycoord![2]).toBeCloseTo(hit.v, 5);
        expect(info.attributes.uv[0]).toBeCloseTo(0.37, 5);
        expect(info.attributes.uv[1]).toBeCloseTo(-0.81, 5);
        expect(info.faceNormal).toEqual([0, 0, 1]);
        // Without a source, attributes need the index.
        const loaded = CpuMeshBvh.deserialize(tree.serialize());
        expect(loaded.getTriangleHitPointInfo(hit.point, hit.triangle).vertices).toBeNull();
        expect(() => loaded.getTriangleHitPointInfo(hit.point, hit.triangle, { attributes: { uv: { array: uv, itemSize: 2 } } })).toThrow(/indices/);
        expect(loaded.getTriangleHitPointInfo(hit.point, hit.triangle, { indices: grid.indices }).vertices).toEqual(info.vertices);
        // Leaves of at most 2.
        const words = new Uint32Array(tree.layout.nodes);
        for (let node = 0; node < tree.nodeCount; node++) {
            expect(words[node * GPU_LAYOUT_NODE_WORDS + GPU_LAYOUT_NODE_WORD.leftCount]).toBeLessThanOrEqual(2);
            expect(words[node * GPU_LAYOUT_NODE_WORDS + GPU_LAYOUT_NODE_WORD.rightCount]).toBeLessThanOrEqual(2);
        }
        expect(tree.estimateMemoryInBytes()).toBeGreaterThanOrEqual(tree.layout.nodes.byteLength + tree.layout.triangles.byteLength);
        const box = { min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 } };
        expect(tree.getBoundingBox(box)).toBe(box);
        expect(box).toEqual({ min: { x: -2, y: -2, z: 0 }, max: { x: 2, y: 2, z: 0 } });
    });

    it('create loads tinybvh itself, and an empty or mismatched layout is refused', async () => {
        const tree = await CpuMeshBvh.create(tessellatedQuad(2, 1));
        expect(tree.triangleCount).toBe(8);
        expect(() => new CpuMeshBvh({ ...tree.layout, nodeCount: tree.nodeCount + 1 })).toThrow(RangeError);
    });
});

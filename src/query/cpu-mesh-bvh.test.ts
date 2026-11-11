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
import { serializeLayout } from '../serialize.ts';
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
    type BvhTraverseCallbacks,
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
        let minDepth = Infinity;
        tree.traverse({
            intersectsBounds: (_box, _isLeaf, _score, depth) => {
                minDepth = Math.min(minDepth, depth);
                return true;
            },
            intersectsRange: (start, count) => {
                expect(count).toBeLessThanOrEqual(4);
                for (let i = start; i < start + count; i++) seen[i]++;
                return false;
            },
        });
        // As three-mesh-bvh: the root's box is never asked about; its children are at depth 1.
        expect(minDepth).toBe(1);
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
            intersectsRange: (_start, _count, _contained, _depth, _nodeIndex, box) => {
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

/** Child slot `slot`'s reference and box, read straight from the layout. */
function slotOf(tree: CpuMeshBvh, slot: number): { index: number; count: number; box: number[]; empty: boolean } {
    const words = new Uint32Array(tree.layout.nodes);
    const floats = new Float32Array(tree.layout.nodes);
    const base = (slot >> 1) * GPU_LAYOUT_NODE_WORDS;
    const left = (slot & 1) === 0;
    const index = words[base + (left ? GPU_LAYOUT_NODE_WORD.leftIndex : GPU_LAYOUT_NODE_WORD.rightIndex)];
    const count = words[base + (left ? GPU_LAYOUT_NODE_WORD.leftCount : GPU_LAYOUT_NODE_WORD.rightCount)];
    const min = base + (left ? GPU_LAYOUT_NODE_WORD.leftMin : GPU_LAYOUT_NODE_WORD.rightMin);
    return { index, count, empty: index === GPU_LAYOUT_EMPTY_CHILD, box: [floats[min], floats[min + 1], floats[min + 2], floats[min + 4], floats[min + 5], floats[min + 6]] };
}

/** The tree positions under a slot, by brute force: [offset, count], and whether they are one run. */
function subtreeRun(tree: CpuMeshBvh, slot: number): { offset: number; count: number; run: boolean } {
    let low = Infinity;
    let high = -Infinity;
    let total = 0;
    const pending = [slot];
    while (pending.length > 0) {
        const at = slotOf(tree, pending.pop()!);
        if (at.empty) continue;
        if (at.count > 0) {
            low = Math.min(low, at.index);
            high = Math.max(high, at.index + at.count);
            total += at.count;
        } else pending.push(at.index * 2, at.index * 2 + 1);
    }
    return { offset: low, count: high - low, run: high - low === total };
}

type TraverseCall = string;

/**
 * three-mesh-bvh 0.9.18's `shapecastTraverse` (src/core/cast/shapecast.js), line for line, over our
 * layout: a node's children are its two slots, a contained child's range runs from its leftmost
 * leaf's offset to its rightmost leaf's end. Logs every callback call.
 */
function threeShapecast(tree: CpuMeshBvh, callbacks: BvhTraverseCallbacks): { result: unknown; log: TraverseCall[] } {
    const log: TraverseCall[] = [];
    const bounds = (box: number[], isLeaf: boolean, score: number | undefined, depth: number, nodeIndex: number) => {
        log.push(`bounds ${box.join()} ${isLeaf} ${score} ${depth} ${nodeIndex}`);
        return callbacks.intersectsBounds(Float32Array.from(box), isLeaf, score, depth, nodeIndex);
    };
    const range = (offset: number, count: number, contained: boolean, depth: number, nodeIndex: number, box: number[]) => {
        log.push(`range ${offset} ${count} ${contained} ${depth} ${nodeIndex} ${box.join()}`);
        return callbacks.intersectsRange(offset, count, contained, depth, nodeIndex, Float32Array.from(box));
    };
    const score = (box: number[]) => {
        log.push(`order ${box.join()}`);
        return callbacks.boundsTraverseOrder!(Float32Array.from(box));
    };
    const visit = (slot: number, depth: number): unknown => {
        const at = slotOf(tree, slot);
        if (at.count > 0) return range(at.index, at.count, false, depth, slot, at.box);
        return inner(at.index, depth);
    };
    const inner = (node: number, depth: number): unknown => {
        let c1 = node * 2;
        let c2 = node * 2 + 1;
        let score1: number | undefined;
        let score2: number | undefined;
        if (callbacks.boundsTraverseOrder) {
            score1 = score(slotOf(tree, c1).box);
            score2 = score(slotOf(tree, c2).box);
            if (score2 < score1) {
                [c1, c2] = [c2, c1];
                [score1, score2] = [score2, score1];
            }
        }
        for (const [child, childScore] of [[c1, score1], [c2, score2]] as const) {
            const at = slotOf(tree, child);
            const answer = bounds(at.box, at.count > 0, childScore, depth + 1, child);
            let stop: unknown;
            if (answer === BVH_CONTAINED) {
                const run = subtreeRun(tree, child);
                stop = range(run.offset, run.count, true, depth + 1, child, at.box);
            } else {
                stop = answer && visit(child, depth + 1);
            }
            if (stop) return true;
        }
        return false;
    };
    return { result: inner(0, 0), log };
}

/** `callbacks` wrapped to log the calls as {@link threeShapecast} does. */
function logged(callbacks: BvhTraverseCallbacks, log: TraverseCall[]): BvhTraverseCallbacks {
    return {
        intersectsBounds: (box, isLeaf, score, depth, nodeIndex) => {
            log.push(`bounds ${Array.from(box).join()} ${isLeaf} ${score} ${depth} ${nodeIndex}`);
            return callbacks.intersectsBounds(box, isLeaf, score, depth, nodeIndex);
        },
        intersectsRange: (offset, count, contained, depth, nodeIndex, box) => {
            log.push(`range ${offset} ${count} ${contained} ${depth} ${nodeIndex} ${Array.from(box).join()}`);
            return callbacks.intersectsRange(offset, count, contained, depth, nodeIndex, box);
        },
        ...(callbacks.boundsTraverseOrder
            ? {
                  boundsTraverseOrder: (box: Float32Array) => {
                      log.push(`order ${Array.from(box).join()}`);
                      return callbacks.boundsTraverseOrder!(box);
                  },
              }
            : {}),
    };
}

describe('CpuMeshBvh.traverse: three-mesh-bvh shapecast contract', () => {
    it('makes the same calls, with the same arguments in the same order, as three-mesh-bvh 0.9.18 shapecast', () => {
        const random = createRandom(51);
        const soup = randomTriangleSoup(random, 700, 4, 0.5);
        const tree = CpuMeshBvh.build(soup, builder, { maxLeafPrimitives: 3 });
        // Answers by box: every kind three takes (falsy ones included), and a stop now and then.
        const answers = [undefined, false, BVH_NOT_INTERSECTED, BVH_INTERSECTED, true, BVH_CONTAINED, BVH_INTERSECTED, true];
        const hash = (box: Float32Array) => Math.abs(Math.floor((box[0] * 7919 + box[4] * 104729 + box[2] * 31) * 1000));
        for (const ordered of [false, true]) {
            for (const stopEvery of [0, 13]) {
                const callbacks: BvhTraverseCallbacks = {
                    intersectsBounds: (box, _isLeaf, _score, depth) => (depth === 1 ? true : answers[hash(box) % answers.length]),
                    intersectsRange: (offset, count) => (stopEvery > 0 && (offset * 31 + count) % stopEvery === 0 ? 'stop' : undefined),
                    ...(ordered ? { boundsTraverseOrder: (box: Float32Array) => box[0] + box[1] * 0.5 - box[5] } : {}),
                };
                const expected = threeShapecast(tree, callbacks);
                const log: TraverseCall[] = [];
                const result = tree.traverse(logged(callbacks, log));
                expect(log, `ordered ${ordered}, stop ${stopEvery}`).toEqual(expected.log);
                expect(result).toBe(Boolean(expected.result));
                expect(log.filter((call) => call.startsWith('range') && call.includes(' true ')).length, 'contained ranges').toBeGreaterThan(0);
            }
        }
    });

    it('skips a box on any falsy answer (undefined, false, 0) and never asks about the root', () => {
        const tree = CpuMeshBvh.build(randomTriangleSoup(createRandom(52), 200, 3, 0.5), builder);
        for (const answer of [undefined, false, 0, null]) {
            const depths: number[] = [];
            let ranges = 0;
            const stopped = tree.traverse({
                intersectsBounds: (_box, _isLeaf, _score, depth) => {
                    depths.push(depth);
                    return answer;
                },
                intersectsRange: () => {
                    ranges++;
                    return false;
                },
            });
            expect(ranges, String(answer)).toBe(0);
            expect(stopped).toBe(false);
            expect(depths, String(answer)).toEqual([1, 1]);
        }
    });

    it('reports a contained inner box as one range over its whole subtree, and passes three\'s argument order', () => {
        const tree = CpuMeshBvh.build(randomTriangleSoup(createRandom(53), 300, 3, 0.5), builder, { maxLeafPrimitives: 2 });
        const left = slotOf(tree, 0);
        expect(left.count).toBe(0); // an inner box
        const calls: unknown[][] = [];
        tree.traverse({
            intersectsBounds: (box, isLeaf, score, depth, nodeIndex) => {
                calls.push(['bounds', Array.from(box), isLeaf, score, depth, nodeIndex]);
                return nodeIndex === 0 ? BVH_CONTAINED : BVH_NOT_INTERSECTED;
            },
            intersectsRange: (offset, count, contained, depth, nodeIndex, box) => {
                calls.push(['range', offset, count, contained, depth, nodeIndex, Array.from(box)]);
                return false;
            },
        });
        const run = subtreeRun(tree, 0);
        expect(run.run).toBe(true);
        expect(calls).toEqual([
            ['bounds', left.box, false, undefined, 1, 0],
            ['range', run.offset, run.count, true, 1, 0, left.box],
            ['bounds', slotOf(tree, 1).box, slotOf(tree, 1).count > 0, undefined, 1, 1],
        ]);
        // Both root children contained: two calls that cover every tree position once.
        const covered: [number, number][] = [];
        tree.traverse({ intersectsBounds: () => BVH_CONTAINED, intersectsRange: (offset, count) => void covered.push([offset, count]) });
        covered.sort((a, b) => a[0] - b[0]);
        expect(covered.length).toBe(2);
        expect(covered[0][0]).toBe(0);
        expect(covered[0][0] + covered[0][1]).toBe(covered[1][0]);
        expect(covered[1][0] + covered[1][1]).toBe(tree.triangleCount);
        // A tree of one leaf: that leaf is the root, reported unasked at depth 0.
        const single = CpuMeshBvh.build({ positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]) }, builder);
        const leafCalls: unknown[][] = [];
        expect(single.traverse({ intersectsBounds: () => leafCalls.push(['bounds']), intersectsRange: (...args) => leafCalls.push(args.slice(0, 5)) })).toBe(true);
        expect(leafCalls).toEqual([[0, 1, false, 0, 0]]);
    });

    it('keeps every subtree one run of tree positions, also over tinybvh trees whose records it reorders', () => {
        const soup = randomTriangleSoup(createRandom(54), 2000, 4, 0.5);
        const layout = builder.buildLayout(soup, {}).layout;
        const before = new Uint32Array(layout.triangles.slice(0));
        const tree = new CpuMeshBvh<BvhRaycastHit>(layout, { source: soup });
        for (let slot = 0; slot < tree.nodeCount * 2; slot++) if (!slotOf(tree, slot).empty) expect(subtreeRun(tree, slot).run).toBe(true);
        // The same records, moved: hits are unchanged against the reference.
        expect(new Uint32Array(layout.triangles).length).toBe(before.length);
        const { problems } = checkAgainstReference(tree, soup, 55, 200);
        expect(problems.slice(0, 5)).toEqual([]);
    });
});

describe('CpuMeshBvh: corrupt layouts and failed refits', () => {
    const soup = randomTriangleSoup(createRandom(61), 120, 3, 0.5);
    /** A copy of a fresh tree's layout with `edit` applied to its node and triangle words. */
    const corrupt = (edit: (nodes: Uint32Array, triangles: Uint32Array, parents: Uint32Array) => void) => {
        const layout = CpuMeshBvh.build(soup, builder).layout;
        const copy = { ...layout, nodes: layout.nodes.slice(0), triangles: layout.triangles.slice(0), parents: layout.parents.slice(), bounds: layout.bounds.slice() };
        edit(new Uint32Array(copy.nodes), new Uint32Array(copy.triangles), copy.parents);
        return copy;
    };
    const innerSlot = (nodes: Uint32Array) => {
        for (let node = 0; ; node++) {
            if (nodes[node * GPU_LAYOUT_NODE_WORDS + GPU_LAYOUT_NODE_WORD.leftCount] === 0) return node * GPU_LAYOUT_NODE_WORDS + GPU_LAYOUT_NODE_WORD.leftIndex;
        }
    };
    const leafSlot = (nodes: Uint32Array) => {
        for (let node = 0; ; node++) {
            if (nodes[node * GPU_LAYOUT_NODE_WORDS + GPU_LAYOUT_NODE_WORD.leftCount] > 0) return node * GPU_LAYOUT_NODE_WORDS + GPU_LAYOUT_NODE_WORD.leftIndex;
        }
    };

    it('throws a clear error for out-of-range children, cycles, shared or unreached nodes, bad leaves and bad original indices', () => {
        const cases: [string, (nodes: Uint32Array, triangles: Uint32Array, parents: Uint32Array) => void, RegExp][] = [
            ['child past the nodes', (nodes) => (nodes[innerSlot(nodes)] = 1_000_000), /past the \d+ nodes/],
            ['cycle to the root', (nodes) => (nodes[innerSlot(nodes)] = 0), /reached twice/],
            [
                'cycle to an ancestor',
                (nodes, _triangles, parents) => {
                    // Point some deep node's left child back at its own parent.
                    for (let node = 1; ; node++) {
                        const base = node * GPU_LAYOUT_NODE_WORDS;
                        if (nodes[base + GPU_LAYOUT_NODE_WORD.leftCount] === 0) {
                            nodes[base + GPU_LAYOUT_NODE_WORD.leftIndex] = parents[node];
                            return;
                        }
                    }
                },
                /reached twice/,
            ],
            ['leaf past the triangles', (nodes) => (nodes[leafSlot(nodes)] = 1_000_000), /past the 120 triangles/],
            ['leaf too big', (nodes) => (nodes[leafSlot(nodes) + 8] = 9), /more than 8/],
            ['original index past the triangles', (_nodes, triangles) => (triangles[3] = 120), /original index 120/],
            ['wrong parent', (nodes, _triangles, parents) => (parents[nodes[innerSlot(nodes)]] = 7777), /parent is 7777/],
        ];
        for (const [name, edit, message] of cases) expect(() => new CpuMeshBvh(corrupt(edit)), name).toThrow(message);
        // Through deserialize as well: the bytes of a corrupt layout.
        const bytes = serializeLayout(corrupt((nodes) => (nodes[innerSlot(nodes)] = 0)));
        expect(() => CpuMeshBvh.deserialize(bytes)).toThrow(/reached twice/);
    });

    it('refit validates everything before writing: a bad vertex or index leaves the tree as it was', () => {
        const tree = CpuMeshBvh.build(soup, builder);
        const before = new Uint8Array(tree.serialize());
        const nan = soup.positions.slice();
        nan[nan.length - 1] = NaN;
        expect(() => tree.refit({ positions: nan })).toThrow(/non-finite/);
        expect(new Uint8Array(tree.serialize())).toEqual(before);
        const indexed = tessellatedQuad(6, 1);
        const grid = CpuMeshBvh.build(indexed, builder);
        const gridBefore = new Uint8Array(grid.serialize());
        const indices = indexed.indices!.slice();
        indices[indices.length - 1] = 9999;
        expect(() => grid.refit({ positions: indexed.positions, indices })).toThrow(/past the/);
        expect(new Uint8Array(grid.serialize())).toEqual(gridBefore);
        // The source is kept too: a plain refit afterwards still uses the original arrays.
        grid.refit();
        expect(new Uint8Array(grid.serialize())).toEqual(gridBefore);
    });

    it('refits a deserialized tree given a source, with the moved hits', () => {
        const tree = CpuMeshBvh.build(soup, builder);
        const loaded = CpuMeshBvh.deserialize(tree.serialize()) as CpuMeshBvh;
        const moved = soup.positions.slice();
        for (let i = 2; i < moved.length; i += 3) moved[i] += 10;
        loaded.refit({ positions: moved });
        const { problems, hits } = checkAgainstReference(loaded, { positions: moved }, 62, 200);
        expect(problems.slice(0, 5)).toEqual([]);
        expect(hits).toBeGreaterThan(3);
        expect(loaded.getBoundingBox()[2]).toBe(Math.fround(boundsOf(moved)[2]));
    });
});

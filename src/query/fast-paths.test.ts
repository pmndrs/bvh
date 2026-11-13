/**
 * The flat CPU queries against the shapecasts they replaced, which stay
 * the reference here: `intersectsBox` against an {@link OrientedBox}
 * shapecast, `closestPointToPoint` against the three-mesh-bvh shapecast
 * with three's `Triangle.closestPointToPoint`, and `refit`'s one pass against
 * `refitRecordsOnCpu`. Answers must be identical, not close: the same
 * booleans, the same bits for points and distances, the same triangle on
 * ties. The edge cases are the ones the shortcuts reason about: boxes that
 * only touch, degenerate triangles, NaN and infinite inputs, inverted boxes,
 * rotated boxes, one-leaf trees, and trees whose children come before their
 * parents.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import { GPU_LAYOUT_EMPTY_CHILD, GPU_LAYOUT_NODE_WORD, GPU_LAYOUT_NODE_WORDS, type GpuLayoutMeshBvh } from '../gpu-layout.ts';
import { buildLbvhOnCpu } from '../lbvh-cpu.ts';
import type { MeshInput } from '../mesh-input.ts';
import { refitRecordsOnCpu } from '../refit-cpu.ts';
import { createTinyBvhBuilder, type TinyBvhBuilder } from '../tinybvh.ts';
import { createRandom, randomTriangleSoup, tessellatedQuad, type Random } from '../trace/oracle-fixtures.ts';
import { CpuMeshBvh, type BvhClosestPoint } from './cpu-mesh-bvh.ts';
import {
    alignedPackedBoxIntersectsBox,
    BvhBox3,
    BvhVector3,
    ExtendedTriangle,
    OrientedBox,
    orientedBoxIntersectsBox,
    orientedBoxIntersectsTriangle,
    packedBoxIntersectsTriangle,
    packOrientedBox,
    PACKED_ORIENTED_BOX_FLOATS,
    triangleClosestPointToPoint,
    type BvhBox3Like,
    type BvhVector3Like,
} from './shapes.ts';

let builder: TinyBvhBuilder;
beforeAll(async () => {
    builder = await createTinyBvhBuilder();
});

/** The intersectsBox this replaced: an OrientedBox shapecast. */
function intersectsBoxReference(tree: CpuMeshBvh, box: BvhBox3Like, matrix?: ArrayLike<number> | null): boolean {
    const obb = new OrientedBox().set(box.min, box.max, matrix);
    return tree.shapecast({ intersectsBounds: (bounds) => obb.intersectsBox(bounds), intersectsTriangle: (triangle) => obb.intersectsTriangle(triangle) });
}

/**
 * The sliver rule, written plainly: on a triangle whose squared sine at `a`
 * is at most 1e-7, the nearest point of its edges (a-b, a-c, b-c) replaces
 * `point` when strictly nearer to `p` (or when `point` is NaN).
 */
function nearerOnSliverEdges(a: BvhVector3Like, b: BvhVector3Like, c: BvhVector3Like, p: BvhVector3Like, point: BvhVector3Like): void {
    const ab = [b.x - a.x, b.y - a.y, b.z - a.z];
    const ac = [c.x - a.x, c.y - a.y, c.z - a.z];
    const n = [ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]];
    const dot = (u: number[], v: number[]) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    if (!(dot(n, n) <= 1e-7 * dot(ab, ab) * dot(ac, ac))) return;
    const distanceSq = (q: number[]) => (q[0] - p.x) ** 2 + (q[1] - p.y) ** 2 + (q[2] - p.z) ** 2;
    let best = distanceSq([point.x, point.y, point.z]);
    if (Number.isNaN(best)) best = Infinity;
    for (const [s, e] of [[a, b], [a, c], [b, c]]) {
        const edge = [e.x - s.x, e.y - s.y, e.z - s.z];
        const lengthSq = dot(edge, edge);
        const t = lengthSq > 0 ? Math.min(1, Math.max(0, ((p.x - s.x) * edge[0] + (p.y - s.y) * edge[1] + (p.z - s.z) * edge[2]) / lengthSq)) : 0;
        const q = [s.x + edge[0] * t, s.y + edge[1] * t, s.z + edge[2] * t];
        if (distanceSq(q) < best) {
            best = distanceSq(q);
            point.x = q[0];
            point.y = q[1];
            point.z = q[2];
        }
    }
}

/** The closestPointToPoint this replaced: three-mesh-bvh's shapecast, step for step, with the sliver rule added. */
function closestReference(tree: CpuMeshBvh, p: readonly number[], minThreshold = 0, maxThreshold = Infinity): BvhClosestPoint | null {
    const [px, py, pz] = p;
    const query = new BvhVector3(px, py, pz);
    const temp = new BvhVector3();
    const best = new BvhVector3();
    let closestSq = Infinity;
    let closestTriangle = -1;
    tree.shapecast({
        boundsTraverseOrder: (box) => {
            const dx = Math.max(box.min.x, Math.min(box.max.x, px)) - px;
            const dy = Math.max(box.min.y, Math.min(box.max.y, py)) - py;
            const dz = Math.max(box.min.z, Math.min(box.max.z, pz)) - pz;
            return dx * dx + dy * dy + dz * dz;
        },
        intersectsBounds: (_box, _isLeaf, score) => score! < closestSq && score! < maxThreshold * maxThreshold,
        intersectsTriangle: (triangle, index) => {
            triangleClosestPointToPoint(triangle.a, triangle.b, triangle.c, query, temp);
            nearerOnSliverEdges(triangle.a, triangle.b, triangle.c, query, temp);
            const dx = px - temp.x;
            const dy = py - temp.y;
            const dz = pz - temp.z;
            const distanceSq = dx * dx + dy * dy + dz * dz;
            if (distanceSq < closestSq) {
                best.copy(temp);
                closestSq = distanceSq;
                closestTriangle = index;
            }
            return distanceSq < minThreshold * minThreshold;
        },
    });
    if (closestSq === Infinity) return null;
    return { point: { x: best.x, y: best.y, z: best.z }, distance: Math.sqrt(closestSq), faceIndex: closestTriangle };
}

/** A soup with degenerate triangles: repeated corners, collinear corners, points. */
function soupWithDegenerates(random: Random, count: number): MeshInput {
    const { positions } = randomTriangleSoup(random, count, 4, 0.8);
    for (let triangle = 0; triangle < count; triangle += 7) {
        const base = triangle * 9;
        const kind = (triangle / 7) % 3;
        if (kind === 0) positions.copyWithin(base + 3, base, base + 3);
        if (kind === 1) for (let axis = 0; axis < 3; axis++) positions[base + 6 + axis] = 2 * positions[base + 3 + axis] - positions[base + axis];
        if (kind === 2) for (let corner = 1; corner < 3; corner++) positions.copyWithin(base + corner * 3, base, base + 3);
    }
    return { positions };
}

function box(minX: number, minY: number, minZ: number, maxX: number, maxY: number, maxZ: number): BvhBox3Like {
    return { min: { x: minX, y: minY, z: minZ }, max: { x: maxX, y: maxY, z: maxZ } };
}

/** Query boxes: random, around a vertex, flush against a vertex or a face of the grid, inverted, flat, NaN, infinite. */
function queryBoxes(random: Random, mesh: MeshInput, count: number): BvhBox3Like[] {
    const p = mesh.positions;
    const vertexCount = p.length / 3;
    const out: BvhBox3Like[] = [];
    for (let i = 0; i < count; i++) {
        const v = Math.floor(random() * vertexCount) * 3;
        const [x, y, z] = [p[v], p[v + 1], p[v + 2]];
        const h = 0.01 + random() * 0.5;
        switch (i % 9) {
            case 0:
                out.push(box(x - h, y - h, z - h, x + h, y + h, z + h));
                break;
            case 1: // the vertex on the box's max corner: touching only
                out.push(box(x - h, y - h, z - h, x, y, z));
                break;
            case 2: // the vertex on the box's min face
                out.push(box(x, y - h, z - h, x + h, y + h, z + h));
                break;
            case 3: // inverted (min above max on one axis)
                out.push(box(x + h, y - h, z - h, x - h, y + h, z + h));
                break;
            case 4: // flat box through the vertex
                out.push(box(x - h, y, z - h, x + h, y, z + h));
                break;
            case 5: // a single point
                out.push(box(x, y, z, x, y, z));
                break;
            case 6:
                out.push(box(NaN, y - h, z - h, x + h, y + h, z + h));
                break;
            case 7:
                out.push(box(-Infinity, y - h, z - h, x + h, y + h, z + h));
                break;
            default: {
                const cx = (random() - 0.5) * 10, cy = (random() - 0.5) * 10, cz = (random() - 0.5) * 10;
                out.push(box(cx - h * 4, cy - h, cz - h, cx + h * 4, cy + h, cz + h));
            }
        }
    }
    return out;
}

/** A rotation (and scale) about a random axis, column-major. */
function rotation(random: Random): number[] {
    const angle = random() * Math.PI * 2;
    const [c, s] = [Math.cos(angle), Math.sin(angle)];
    const scale = 0.5 + random();
    return [c * scale, s * scale, 0, 0, -s * scale, c * scale, 0, 0, 0, 0, scale, 0, random() - 0.5, random() - 0.5, random() - 0.5, 1];
}

/** Whether every inner child's index is above its parent's (the reverse-order refit's condition). */
function childrenFollowParents(layout: GpuLayoutMeshBvh): boolean {
    const words = new Uint32Array(layout.nodes);
    for (let node = 0; node < layout.nodeCount; node++) {
        for (const [indexWord, countWord] of [
            [GPU_LAYOUT_NODE_WORD.leftIndex, GPU_LAYOUT_NODE_WORD.leftCount],
            [GPU_LAYOUT_NODE_WORD.rightIndex, GPU_LAYOUT_NODE_WORD.rightCount],
        ]) {
            const index = words[node * GPU_LAYOUT_NODE_WORDS + indexWord];
            if (index !== GPU_LAYOUT_EMPTY_CHILD && words[node * GPU_LAYOUT_NODE_WORDS + countWord] === 0 && index <= node) return false;
        }
    }
    return true;
}

describe('intersectsBox, flat', () => {
    it('gives the OrientedBox shapecast answer: touching, inverted, flat, NaN and infinite boxes, rotated boxes, degenerate triangles', () => {
        const random = createRandom(11);
        const meshes: [string, MeshInput][] = [
            ['grid', tessellatedQuad(12, 3)],
            ['degenerates', soupWithDegenerates(random, 400)],
            ['soup', randomTriangleSoup(random, 600, 5, 1)],
        ];
        let hits = 0;
        let total = 0;
        for (const [name, mesh] of meshes) {
            const tree = CpuMeshBvh.build(mesh, builder);
            for (const [i, query] of queryBoxes(random, mesh, 900).entries()) {
                const matrix = i % 5 === 4 ? rotation(random) : i % 5 === 3 ? [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] : null;
                const expected = intersectsBoxReference(tree, query, matrix);
                expect(tree.intersectsBox(query, matrix), `${name} #${i}`).toBe(expected);
                hits += expected ? 1 : 0;
                total++;
            }
        }
        // Both answers occur, so the comparison means something.
        expect(hits).toBeGreaterThan(total * 0.2);
        expect(hits).toBeLessThan(total * 0.8);
    });

    it('tests every triangle of a one-leaf tree without a box test, as the shapecast does', () => {
        const mesh = { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]) };
        const tree = CpuMeshBvh.build(mesh, builder);
        for (const query of [box(0.1, 0.1, -1, 0.2, 0.2, 1), box(2, 2, 2, 3, 3, 3), box(1, 0, 0, 2, 1, 1), box(NaN, 0, 0, 1, 1, 1)]) {
            expect(tree.intersectsBox(query)).toBe(intersectsBoxReference(tree, query));
        }
    });

    it('packedBoxIntersectsTriangle is orientedBoxIntersectsTriangle, NaN and degenerate corners included', () => {
        const random = createRandom(5);
        const packed = new Float64Array(PACKED_ORIENTED_BOX_FLOATS);
        const corners = new Float32Array(12);
        const triangle = new ExtendedTriangle();
        let agreeTrue = 0;
        for (let i = 0; i < 20_000; i++) {
            const h = 0.05 + random();
            const obb = new OrientedBox().set({ x: -h, y: -h * random(), z: -h }, { x: h, y: h, z: h * random() }, i % 2 ? rotation(random) : null);
            packOrientedBox(obb, packed);
            for (let k = 0; k < 9; k++) corners[Math.floor(k / 3) * 4 + (k % 3)] = (random() - 0.5) * 3;
            if (i % 11 === 0) corners.copyWithin(4, 0, 3); // a segment
            if (i % 13 === 0) corners.copyWithin(8, 0, 3);
            if (i % 17 === 0) corners[(i % 3) * 4 + (i % 3)] = NaN;
            if (i % 19 === 0) corners[5] = Infinity;
            if (i % 23 === 0) corners[0] = h; // a corner on the box's face
            triangle.set({ x: corners[0], y: corners[1], z: corners[2] }, { x: corners[4], y: corners[5], z: corners[6] }, { x: corners[8], y: corners[9], z: corners[10] });
            const expected = orientedBoxIntersectsTriangle(obb, triangle);
            expect(packedBoxIntersectsTriangle(packed, corners, 0, 4, 8), `#${i}`).toBe(expected);
            agreeTrue += expected ? 1 : 0;
        }
        expect(agreeTrue).toBeGreaterThan(1000);
    });

    it('alignedPackedBoxIntersectsBox is orientedBoxIntersectsBox for an axis-aligned finite box, against touching, inverted, NaN and infinite node boxes', () => {
        const random = createRandom(9);
        const packed = new Float64Array(PACKED_ORIENTED_BOX_FLOATS);
        const node = new BvhBox3();
        const values = [-1, 0, 1, -0, 0.5, 2, NaN, Infinity, -Infinity, 3.4028234663852886e38];
        for (let i = 0; i < 20_000; i++) {
            const pick = () => (random() < 0.4 ? values[Math.floor(random() * values.length)] : (random() - 0.5) * 4);
            const lo = [pick(), pick(), pick()].map((v) => (Number.isFinite(v) ? v : 0));
            const hi = [pick(), pick(), pick()].map((v) => (Number.isFinite(v) ? v : 1));
            const obb = new OrientedBox().set({ x: lo[0], y: lo[1], z: lo[2] }, { x: hi[0], y: hi[1], z: hi[2] }, null);
            packOrientedBox(obb, packed);
            node.min.x = pick();
            node.min.y = pick();
            node.min.z = pick();
            node.max.x = pick();
            node.max.y = pick();
            node.max.z = pick();
            // Touching: share a face with the query box.
            if (i % 7 === 0) node.min.x = hi[0];
            if (i % 7 === 1) node.max.y = lo[1];
            expect(alignedPackedBoxIntersectsBox(packed, node.min.x, node.min.y, node.min.z, node.max.x, node.max.y, node.max.z), `#${i}`).toBe(orientedBoxIntersectsBox(obb, node));
        }
    });
});

describe('closestPointToPoint, flat', () => {
    it('gives the shapecast answer bit for bit (point, distance, triangle on ties): vertices, edges, faces, degenerate triangles, thresholds', () => {
        const random = createRandom(21);
        const grid = tessellatedQuad(10, 2);
        const meshes: [string, MeshInput][] = [
            ['grid', grid],
            ['degenerates', soupWithDegenerates(random, 300)],
            ['soup', randomTriangleSoup(random, 500, 5, 1)],
        ];
        for (const [name, mesh] of meshes) {
            const tree = CpuMeshBvh.build(mesh, builder);
            const p = mesh.positions;
            for (let i = 0; i < 600; i++) {
                // Points on vertices and edge midpoints (shared by several triangles: ties), above them, and anywhere.
                const v = Math.floor(random() * (p.length / 3)) * 3;
                const w = Math.floor(random() * (p.length / 3)) * 3;
                const point =
                    i % 4 === 0
                        ? [p[v], p[v + 1], p[v + 2]]
                        : i % 4 === 1
                          ? [(p[v] + p[w]) / 2, (p[v + 1] + p[w + 1]) / 2, (p[v + 2] + p[w + 2]) / 2 + 0.25]
                          : i % 4 === 2
                            ? [p[v], p[v + 1], p[v + 2] + 1]
                            : [(random() - 0.5) * 12, (random() - 0.5) * 12, (random() - 0.5) * 12];
                const [min, max] = i % 7 === 5 ? [0, 0.6] : i % 7 === 6 ? [0.3, Infinity] : [0, Infinity];
                const expected = closestReference(tree, point, min, max);
                const result = tree.closestPointToPoint(point, {}, min, max);
                if (expected === null) {
                    expect(result, `${name} #${i}`).toBeNull();
                    continue;
                }
                expect(result, `${name} #${i}`).not.toBeNull();
                expect([result!.faceIndex, result!.distance, result!.point.x, result!.point.y, result!.point.z], `${name} #${i}`).toEqual([
                    expected.faceIndex,
                    expected.distance,
                    expected.point.x,
                    expected.point.y,
                    expected.point.z,
                ]);
            }
        }
    });

    it('breaks ties as the shapecast does: a fan of one-triangle leaves around a shared corner, equal child scores', () => {
        // 64 triangles share the corner at the origin; at and right above it every one is equally near, and
        // the boxes around it all score 0, so the answer is whichever triangle the walk order reaches first.
        const count = 64;
        const positions = new Float32Array(count * 9);
        for (let k = 0; k < count; k++) {
            const a0 = (k / count) * Math.PI * 2;
            const a1 = ((k + 1) / count) * Math.PI * 2;
            positions.set([0, 0, 0, Math.cos(a0), Math.sin(a0), (k % 3) * 0.1, Math.cos(a1), Math.sin(a1), (k % 5) * 0.1], k * 9);
        }
        const tree = CpuMeshBvh.build({ positions }, builder, { maxLeafPrimitives: 1 });
        for (const point of [[0, 0, 0], [0, 0, 0.5], [0, 0, -0.5], [0.001, 0, 0], [0, 0, 1e-7]]) {
            const expected = closestReference(tree, point)!;
            const result = tree.closestPointToPoint(point)!;
            expect([result.faceIndex, result.distance, result.point.x, result.point.y, result.point.z], String(point)).toEqual([expected.faceIndex, expected.distance, expected.point.x, expected.point.y, expected.point.z]);
        }
    });

    it("finds a point lying on a sliver on it, where three's regions put it 0.014 away; segments and points too", () => {
        // An f32 sliver (sine at a about 1e-5) and a point on it: three's Triangle.closestPointToPoint picks a
        // wrong region and answers 0.0141; the edges give 1.5e-10.
        const sliver = [-0.0048814634792506695, -0.12525907158851624, 0.1028691977262497, -0.4439702332019806, 0.34214118123054504, -0.9760182499885559, -0.20825335383415222, 0.09122579544782639, -0.3968370258808136];
        const point = [-0.2939024307407032, 0.182397336599706, -0.6072858374554473];
        const corner = (k: number) => new BvhVector3(sliver[k * 3], sliver[k * 3 + 1], sliver[k * 3 + 2]);
        const three = triangleClosestPointToPoint(corner(0), corner(1), corner(2), new BvhVector3(point[0], point[1], point[2]), new BvhVector3());
        expect(Math.hypot(three.x - point[0], three.y - point[1], three.z - point[2])).toBeGreaterThan(0.01);
        // Far triangles around it, so the sliver sits in a tree with boxes to walk.
        const positions = new Float32Array(9 * 9);
        positions.set(sliver);
        for (let k = 1; k < 9; k++) positions.set([k, 5, 0, k + 1, 5, 0, k, 6, 0], k * 9);
        const tree = CpuMeshBvh.build({ positions }, builder, { maxLeafPrimitives: 1 });
        const result = tree.closestPointToPoint(point)!;
        expect(result.faceIndex).toBe(0);
        expect(result.distance).toBeLessThan(1e-6);
        expect(result).toEqual(closestReference(tree, point));
        // A segment (c on the line a-b) and a point: the nearest point of the segment, and the point itself.
        const segment = new Float32Array([0, 0, 0, 2, 0, 0, 1, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5]);
        const segmentTree = CpuMeshBvh.build({ positions: segment }, builder, { maxLeafPrimitives: 1 });
        for (const [p, distance, face] of [[[0.5, 0, 0], 0, 0], [[1.5, 1, 0], 1, 0], [[3, 0, 0], 1, 0], [[5, 5, 6], 1, 1]] as const) {
            const hit = segmentTree.closestPointToPoint([...p])!;
            expect([hit.faceIndex, hit.distance], String(p)).toEqual([face, distance]);
            expect(hit).toEqual(closestReference(segmentTree, [...p]));
        }
    });

    it('takes { x, y, z } points, tests a one-leaf tree without a box test, and returns null for NaN', () => {
        const mesh = { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1]) };
        const tree = CpuMeshBvh.build(mesh, builder, { maxLeafPrimitives: 8 });
        for (const point of [[0.2, 0.2, 5], [3, 3, 3], [-1, -1, 0.5]]) {
            const expected = closestReference(tree, point)!;
            expect(tree.closestPointToPoint({ x: point[0], y: point[1], z: point[2] })).toEqual({ point: new BvhVector3(expected.point.x, expected.point.y, expected.point.z), distance: expected.distance, faceIndex: expected.faceIndex });
        }
        expect(tree.closestPointToPoint([NaN, 0, 0])).toBe(closestReference(tree, [NaN, 0, 0]));
    });
});

describe('refit, one pass', () => {
    it("writes refitRecordsOnCpu's boxes and the moved corners bit for bit, in reverse node order and in the parents' order", () => {
        const random = createRandom(31);
        const mesh = randomTriangleSoup(random, 700, 5, 1);
        const indexed = tessellatedQuad(16, 2);
        const cases: [string, CpuMeshBvh, MeshInput][] = [
            ['tinybvh', CpuMeshBvh.build(mesh, builder), mesh],
            ['tinybvh, indexed', CpuMeshBvh.build(indexed, builder), indexed],
            ['LBVH twin', new CpuMeshBvh(buildLbvhOnCpu(mesh), { source: mesh }), mesh],
        ];
        // The LBVH's children come before their parents (Karras' layout): the order array path.
        expect(childrenFollowParents(cases[0][1].layout)).toBe(true);
        expect(childrenFollowParents(cases[2][1].layout)).toBe(false);
        for (const [name, tree, source] of cases) {
            for (let round = 0; round < 3; round++) {
                const moved = source.positions.map((v, i) => v + Math.sin(i * 0.37 + round) * 0.3);
                const expectedNodes = tree.layout.nodes.slice(0);
                const expectedRecords = new Float32Array(tree.layout.triangles.slice(0));
                const recordWords = new Uint32Array(expectedRecords.buffer);
                for (let position = 0; position < tree.triangleCount; position++) {
                    const triangle = recordWords[position * 12 + 3];
                    for (let corner = 0; corner < 3; corner++) {
                        const vertex = source.indices ? source.indices[triangle * 3 + corner] : triangle * 3 + corner;
                        for (let axis = 0; axis < 3; axis++) expectedRecords[position * 12 + corner * 4 + axis] = moved[vertex * 3 + axis];
                    }
                }
                const { rootBox } = refitRecordsOnCpu(new Float32Array(expectedNodes), new Uint32Array(expectedNodes), 0, tree.nodeCount, tree.layout.parents, expectedRecords);
                tree.refit({ positions: moved, indices: source.indices });
                expect(new Uint8Array(tree.layout.nodes), name).toEqual(new Uint8Array(expectedNodes));
                expect(new Uint8Array(tree.layout.triangles), name).toEqual(new Uint8Array(expectedRecords.buffer));
                expect(Array.from(tree.layout.bounds.subarray(0, 6)), name).toEqual(Array.from(rootBox, (v) => Math.fround(v)));
            }
        }
    });

    it('allows a non-finite vertex no triangle uses, and still refuses a used one or an index past the positions', () => {
        const grid = tessellatedQuad(4, 1);
        const tree = CpuMeshBvh.build(grid, builder);
        // One extra vertex, unused, NaN: the quick scan fails, the per-triangle check passes.
        const extra = new Float32Array(grid.positions.length + 3);
        extra.set(grid.positions);
        extra.fill(NaN, grid.positions.length);
        expect(() => tree.refit({ positions: extra, indices: grid.indices })).not.toThrow();
        const used = extra.slice();
        used[4] = Infinity;
        const before = new Uint8Array(tree.serialize());
        expect(() => tree.refit({ positions: used, indices: grid.indices })).toThrow(/vertex 1 has a non-finite position/);
        expect(new Uint8Array(tree.serialize())).toEqual(before);
    });
});

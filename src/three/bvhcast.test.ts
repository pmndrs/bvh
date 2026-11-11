/**
 * `bvhcast` against three-mesh-bvh 0.9.18 on random mesh pairs and
 * placements (its trees in indirect mode, so `resolveTriangleIndex` gives
 * original indices on both sides): three-bvh-csg's pattern (every pair of
 * triangles from overlapping leaves, kept when `intersectsTriangle` says so,
 * with the intersection segment) must find the same pairs and the same
 * segments. Then the contract: each leaf pair once, stopping, ranges first,
 * `intersectsGeometry` on top.
 */

import { BufferAttribute, BufferGeometry, IcosahedronGeometry, Line3, Matrix4, TorusKnotGeometry, Vector3 } from 'three';
import { MeshBVH as TmbMeshBVH, type ExtendedTriangle as TmbExtendedTriangle } from 'three-mesh-bvh';
import { beforeAll, describe, expect, it } from 'vitest';
import { CpuMeshBvh } from '../query/cpu-mesh-bvh.ts';
import { createTinyBvhBuilder, type TinyBvhBuilder } from '../tinybvh.ts';
import { createRandom, randomAffine, randomTriangleSoup, type Random } from '../trace/oracle-fixtures.ts';
import { BvhBoundsTree, meshInputFromGeometry, prepareBoundsTrees } from './raycast.ts';
import type { ExtendedTriangle } from './shapes.ts';

let builder: TinyBvhBuilder;
beforeAll(async () => {
    await prepareBoundsTrees();
    builder = await createTinyBvhBuilder();
});

interface Mesh {
    readonly name: string;
    readonly geometry: BufferGeometry;
    readonly ours: BvhBoundsTree;
    readonly core: CpuMeshBvh;
    readonly theirs: TmbMeshBVH;
}

function mesh(name: string, geometry: BufferGeometry): Mesh {
    return {
        name,
        geometry,
        ours: new BvhBoundsTree(geometry, { maxLeafSize: 3 }),
        core: CpuMeshBvh.build(meshInputFromGeometry(geometry), builder),
        theirs: new TmbMeshBVH(geometry.clone(), { maxLeafSize: 3, indirect: true }),
    };
}

function soup(random: Random, count: number): BufferGeometry {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(randomTriangleSoup(random, count, 1, 0.2).positions, 3));
    return geometry;
}

function meshes(): Mesh[] {
    const random = createRandom(61);
    return [mesh('soup', soup(random, 1500)), mesh('torus knot', new TorusKnotGeometry(0.7, 0.25, 80, 12)), mesh('sphere', new IcosahedronGeometry(0.9, 6))];
}

function placement(random: Random): Matrix4 {
    const m = new Matrix4().fromArray(randomAffine(random, { extent: 0.6, mirror: random() < 0.3 }));
    return random() < 0.15 ? new Matrix4() : m;
}

type Cast = (
    other: unknown,
    matrix: Matrix4,
    callbacks: { intersectsTriangles: (a: ExtendedTriangle, b: ExtendedTriangle, i1: number, i2: number) => boolean | void },
) => boolean;

/** three-bvh-csg's collectIntersectingTriangles, minus its coplanar special case: pairs (original indices) and their segments. */
function csgPairs(tree: { bvhcast: unknown; resolveTriangleIndex: (i: number) => number }, other: { resolveTriangleIndex: (i: number) => number }, matrix: Matrix4): string[] {
    const found: string[] = [];
    const edge = new Line3();
    (tree.bvhcast as Cast).call(tree, other, matrix, {
        intersectsTriangles(triangleA, triangleB, ia, ib) {
            if ((triangleA as unknown as TmbExtendedTriangle as unknown as ExtendedTriangle).intersectsTriangle(triangleB, edge, true)) {
                found.push(`${tree.resolveTriangleIndex(ia)} ${other.resolveTriangleIndex(ib)} ${edge.start.toArray()} ${edge.end.toArray()}`);
            }
            return false;
        },
    });
    return found.sort();
}

describe('bvhcast against three-mesh-bvh', () => {
    it("finds three-bvh-csg's intersecting pairs and segments on random mesh pairs and placements", () => {
        const random = createRandom(62);
        const all = meshes();
        let pairs = 0;
        let empty = 0;
        const casts = 45;
        for (let cast = 0; cast < casts; cast++) {
            const a = all[cast % 3];
            const b = all[Math.floor(cast / 3) % 3];
            const matrix = placement(random);
            const theirs = csgPairs(a.theirs as never, b.theirs as never, matrix);
            expect(csgPairs(a.ours as never, b.ours as never, matrix), `${a.name} x ${b.name} #${cast}`).toEqual(theirs);
            expect(csgPairs(a.core as never, b.core as never, matrix), `${a.name} x ${b.name} #${cast} (core)`).toEqual(theirs);
            pairs += theirs.length;
            if (theirs.length === 0) empty++;
        }
        expect(pairs).toBeGreaterThan(casts * 20);
        expect(empty).toBeLessThan(casts / 2);
    });

    it('hands each pair of leaves over once, covering every intersecting pair, with leaf ids that name one range each', () => {
        const random = createRandom(63);
        const [soupMesh, knot] = meshes();
        for (let cast = 0; cast < 10; cast++) {
            const matrix = placement(random);
            const seen = new Set<string>();
            const ranges1 = new Map<number, string>();
            let duplicates = 0;
            soupMesh.ours.bvhcast(knot.ours, matrix, {
                intersectsRanges: (offset1, count1, offset2, count2, depth1, nodeIndex1) => {
                    for (let i1 = offset1; i1 < offset1 + count1; i1++) {
                        for (let i2 = offset2; i2 < offset2 + count2; i2++) {
                            const key = `${i1} ${i2}`;
                            if (seen.has(key)) duplicates++;
                            seen.add(key);
                        }
                    }
                    const range = `${offset1} ${count1}`;
                    expect(ranges1.get(nodeIndex1) ?? range).toBe(range);
                    ranges1.set(nodeIndex1, range);
                    expect(depth1).toBeGreaterThan(0);
                    return false;
                },
            });
            expect(duplicates).toBe(0);
            // Every pair three-mesh-bvh finds intersecting is in some range pair we handed over.
            for (const line of csgPairs(soupMesh.theirs as never, knot.theirs as never, matrix)) {
                const [ia, ib] = line.split(' ').map(Number);
                expect(seen.has(`${soupMesh.ours.treePositionOf(ia)} ${knot.ours.treePositionOf(ib)}`)).toBe(true);
            }
        }
    });

    it('handles a tree that is one leaf, on either side', () => {
        const random = createRandom(64);
        const [soupMesh] = meshes();
        const small = mesh('two triangles', soup(random, 2));
        expect(small.ours.nodeCount).toBe(1);
        let pairs = 0;
        for (let cast = 0; cast < 40; cast++) {
            const matrix = placement(random);
            const theirs = csgPairs(small.theirs as never, soupMesh.theirs as never, matrix);
            expect(csgPairs(small.ours as never, soupMesh.ours as never, matrix)).toEqual(theirs);
            expect(csgPairs(soupMesh.ours as never, small.ours as never, matrix)).toEqual(csgPairs(soupMesh.theirs as never, small.theirs as never, matrix));
            pairs += theirs.length;
        }
        expect(pairs).toBeGreaterThan(0);
    });

    it('stops on a truthy return, asks intersectsRanges first, and needs a callback', () => {
        const [soupMesh, knot] = meshes();
        const matrix = new Matrix4();
        let calls = 0;
        expect(soupMesh.ours.bvhcast(knot.ours, matrix, { intersectsTriangles: () => ++calls === 7 })).toBe(true);
        expect(calls).toBe(7);
        let triangles = 0;
        expect(soupMesh.ours.bvhcast(knot.ours, matrix, { intersectsRanges: () => true, intersectsTriangles: () => void triangles++ })).toBe(true);
        expect(triangles).toBe(0);
        expect(soupMesh.ours.bvhcast(knot.ours, matrix, { intersectsRanges: () => false, intersectsTriangles: () => void triangles++ })).toBe(false);
        expect(triangles).toBeGreaterThan(0);
        // Far apart: nothing is handed over.
        expect(soupMesh.ours.bvhcast(knot.ours, new Matrix4().makeTranslation(new Vector3(10, 0, 0)), { intersectsRanges: () => true })).toBe(false);
        expect(() => soupMesh.ours.bvhcast(knot.ours, matrix, {})).toThrow(TypeError);
        // A cast inside a cast's callback works, and leaves the outer triangles alone.
        let inner = 0;
        soupMesh.ours.bvhcast(knot.ours, matrix, {
            intersectsTriangles: (a) => {
                const before = a.a.clone();
                knot.ours.bvhcast(soupMesh.ours, matrix, { intersectsTriangles: () => ++inner > 2 });
                expect(a.a.equals(before)).toBe(true);
                return true;
            },
        });
        expect(inner).toBe(3);
    });
});

/**
 * Shapecasts and the `intersects*` queries against three-mesh-bvh 0.9.18 on
 * the same geometry (its tree in indirect mode, so its triangle indices are
 * the original ones too): for random oriented boxes, spheres, capsules and
 * mesh pairs, the same triangles must be found, the same closest points, the
 * same booleans. The trees differ, so the boxes visited differ; what the
 * shapes touch may not. Both run through `geometry.boundsTree` (three's
 * types) and the core `CpuMeshBvh` (plain types).
 */

import { Box3, BufferAttribute, BufferGeometry, Line3, Matrix4, Sphere, TorusKnotGeometry, Vector3 } from 'three';
import { CONTAINED as TMB_CONTAINED, INTERSECTED as TMB_INTERSECTED, MeshBVH as TmbMeshBVH, OrientedBox as TmbOrientedBox } from 'three-mesh-bvh';
import { beforeAll, describe, expect, it } from 'vitest';
import { BVH_CONTAINED, BVH_INTERSECTED, CpuMeshBvh } from '../query/cpu-mesh-bvh.ts';
import { OrientedBox as CoreOrientedBox, sphereIntersectsBox } from '../query/shapes.ts';
import { createTinyBvhBuilder, type TinyBvhBuilder } from '../tinybvh.ts';
import { createRandom, randomAffine, randomTriangleSoup, type Random } from '../trace/oracle-fixtures.ts';
import { BvhBoundsTree, meshInputFromGeometry, prepareBoundsTrees } from './raycast.ts';
import { ExtendedTriangle, OrientedBox } from './shapes.ts';

let builder: TinyBvhBuilder;
beforeAll(async () => {
    await prepareBoundsTrees();
    builder = await createTinyBvhBuilder();
});

interface Fixture {
    readonly name: string;
    readonly ours: BvhBoundsTree;
    readonly core: CpuMeshBvh;
    readonly theirs: TmbMeshBVH;
    readonly geometry: BufferGeometry;
}

function soupGeometry(random: Random, count: number): BufferGeometry {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(randomTriangleSoup(random, count, 1, 0.15).positions, 3));
    return geometry;
}

function fixture(name: string, geometry: BufferGeometry): Fixture {
    return {
        name,
        geometry,
        ours: new BvhBoundsTree(geometry),
        core: CpuMeshBvh.build(meshInputFromGeometry(geometry), builder),
        theirs: new TmbMeshBVH(geometry.clone(), { indirect: true }),
    };
}

function fixtures(): Fixture[] {
    const random = createRandom(51);
    return [fixture('soup', soupGeometry(random, 3000)), fixture('torus knot', new TorusKnotGeometry(0.7, 0.25, 96, 16))];
}

const sorted = (values: number[]) => [...values].sort((a, b) => a - b);

function randomPoint(random: Random, extent: number): Vector3 {
    return new Vector3((random() * 2 - 1) * extent, (random() * 2 - 1) * extent, (random() * 2 - 1) * extent);
}

function randomBox(random: Random, size: number): Box3 {
    const center = randomPoint(random, 1);
    const half = new Vector3(random() * size, random() * size, random() * size);
    return new Box3(center.clone().sub(half), center.clone().add(half));
}

/** A rotation and non-uniform scale about a point near the mesh (no translation away from it). */
function randomPlacement(random: Random): Matrix4 {
    const m = new Matrix4().fromArray(randomAffine(random, { extent: 0.5, mirror: random() < 0.3 }));
    return m;
}

describe('shapecast against three-mesh-bvh', () => {
    it('finds the same triangles for oriented boxes, and intersectsBox gives the same answer', () => {
        const random = createRandom(52);
        for (const { name, ours, core, theirs } of fixtures()) {
            let found = 0;
            let hits = 0;
            const queries = 150;
            for (let query = 0; query < queries; query++) {
                const box = randomBox(random, 0.4);
                const matrix = random() < 0.2 ? new Matrix4() : randomPlacement(random);
                const theirsObb = new TmbOrientedBox(box.min, box.max);
                theirsObb.set(box.min, box.max, matrix);
                const theirsFound: number[] = [];
                theirs.shapecast({
                    intersectsBounds: (b) => theirsObb.intersectsBox(b),
                    intersectsTriangle: (triangle, index) => {
                        if (theirsObb.intersectsTriangle(triangle)) theirsFound.push(index);
                        return false;
                    },
                });
                const obb = new OrientedBox().set(box.min, box.max, matrix);
                const oursFound: number[] = [];
                ours.shapecast({
                    intersectsBounds: (b) => obb.intersectsBox(b),
                    intersectsTriangle: (triangle, index) => {
                        if (obb.intersectsTriangle(triangle)) oursFound.push(index);
                        return false;
                    },
                });
                const coreObb = new CoreOrientedBox().set(box.min, box.max, matrix.elements);
                const coreFound: number[] = [];
                core.shapecast({
                    intersectsBounds: (b) => coreObb.intersectsBox(b),
                    intersectsTriangle: (triangle, index) => {
                        if (coreObb.intersectsTriangle(triangle)) coreFound.push(index);
                        return false;
                    },
                });
                expect(sorted(oursFound), `${name} #${query}`).toEqual(sorted(theirsFound));
                expect(sorted(coreFound), `${name} #${query}`).toEqual(sorted(theirsFound));
                const theirsHit = theirs.intersectsBox(box, matrix);
                expect(ours.intersectsBox(box, matrix), `${name} #${query}`).toBe(theirsHit);
                expect(core.intersectsBox(box, matrix.elements), `${name} #${query}`).toBe(theirsHit);
                expect(theirsHit).toBe(theirsFound.length > 0);
                found += theirsFound.length;
                if (theirsHit) hits++;
            }
            // Worth having: most boxes hit something, not all.
            expect(found).toBeGreaterThan(queries);
            expect(hits).toBeGreaterThan(queries / 4);
            expect(hits).toBeLessThan(queries);
        }
    });

    it('finds the same triangles for spheres, and intersectsSphere gives the same answer', () => {
        const random = createRandom(53);
        for (const { name, ours, core, theirs } of fixtures()) {
            let hits = 0;
            const queries = 150;
            for (let query = 0; query < queries; query++) {
                const sphere = new Sphere(randomPoint(random, 1.1), random() * 0.3);
                const collect = (tree: { shapecast: (callbacks: never) => boolean }) => {
                    const found: number[] = [];
                    (tree.shapecast as (callbacks: object) => boolean)({
                        intersectsBounds: (b: Box3) => sphereIntersectsBox(sphere, b),
                        intersectsTriangle: (triangle: ExtendedTriangle, index: number) => {
                            if (triangle.intersectsSphere(sphere)) found.push(index);
                            return false;
                        },
                    });
                    return sorted(found);
                };
                const theirsFound = collect(theirs as never);
                expect(collect(ours as never), `${name} #${query}`).toEqual(theirsFound);
                expect(collect(core as never), `${name} #${query}`).toEqual(theirsFound);
                const theirsHit = theirs.intersectsSphere(sphere);
                expect(ours.intersectsSphere(sphere), `${name} #${query}`).toBe(theirsHit);
                expect(core.intersectsSphere(sphere), `${name} #${query}`).toBe(theirsHit);
                if (theirsHit) hits++;
            }
            expect(hits).toBeGreaterThan(queries / 4);
            expect(hits).toBeLessThan(queries);
        }
    });

    it("gives a capsule (homefig's controller: box cull, closestPointToSegment) the same contacts and points", () => {
        const random = createRandom(54);
        for (const { name, ours, theirs } of fixtures()) {
            let contacts = 0;
            for (let query = 0; query < 150; query++) {
                const start = randomPoint(random, 1);
                const segment = new Line3(start, start.clone().add(randomPoint(random, 0.3)));
                const radius = 0.05 + random() * 0.2;
                const bounds = new Box3().makeEmpty().expandByPoint(segment.start).expandByPoint(segment.end);
                bounds.min.addScalar(-radius);
                bounds.max.addScalar(radius);
                const run = (tree: { shapecast: (callbacks: never) => boolean }) => {
                    const found: string[] = [];
                    const onTriangle = new Vector3();
                    const onSegment = new Vector3();
                    (tree.shapecast as (callbacks: object) => boolean)({
                        intersectsBounds: (box: Box3) => box.intersectsBox(bounds),
                        intersectsTriangle: (triangle: ExtendedTriangle, index: number) => {
                            const distance = triangle.closestPointToSegment(segment, onTriangle, onSegment);
                            if (distance < radius) found.push(`${index} ${distance} ${onTriangle.toArray()} ${onSegment.toArray()}`);
                        },
                    });
                    return found.sort();
                };
                const theirsFound = run(theirs as never);
                expect(run(ours as never), `${name} #${query}`).toEqual(theirsFound);
                contacts += theirsFound.length;
            }
            expect(contacts).toBeGreaterThan(100);
        }
    });
});

describe('intersectsGeometry against three-mesh-bvh', () => {
    it('gives the same answer for random placements, tree against tree and against a bare geometry', () => {
        const random = createRandom(55);
        const [soup, knot] = fixtures();
        const other = fixture('small soup', soupGeometry(random, 400));
        let hits = 0;
        const queries = 60;
        for (let query = 0; query < queries; query++) {
            const target = query % 2 === 0 ? soup : knot;
            const matrix = randomPlacement(random).premultiply(new Matrix4().makeTranslation(randomPoint(random, 1.2)));
            // Tree against tree: ours on our trees, theirs on theirs.
            const otherOurs = other.geometry;
            const otherTheirs = other.geometry.clone();
            (otherTheirs as unknown as { boundsTree: TmbMeshBVH }).boundsTree = other.theirs;
            (otherOurs as unknown as { boundsTree: BvhBoundsTree }).boundsTree = other.ours;
            const theirsHit = target.theirs.intersectsGeometry(otherTheirs, matrix);
            expect(target.ours.intersectsGeometry(otherOurs, matrix), `#${query}`).toBe(theirsHit);
            expect(target.core.intersectsGeometry(other.core, matrix.elements), `#${query}`).toBe(theirsHit);
            // A bare geometry (no tree): every triangle of it against our leaves.
            const bare = other.geometry.clone();
            expect(target.ours.intersectsGeometry(bare, matrix), `bare #${query}`).toBe(theirsHit);
            expect(target.core.intersectsGeometry(meshInputFromGeometry(bare), matrix.elements), `bare #${query}`).toBe(theirsHit);
            if (theirsHit) hits++;
        }
        expect(hits).toBeGreaterThan(queries / 5);
        expect(hits).toBeLessThan(queries);
    });
});

describe('shapecast contract', () => {
    it('hands over original indices and the corners of those triangles, with needsUpdate set', () => {
        const [{ ours, geometry }] = fixtures();
        const positions = geometry.getAttribute('position');
        let seen = 0;
        ours.shapecast({
            intersectsBounds: () => BVH_INTERSECTED,
            intersectsTriangle: (triangle, index, contained, depth) => {
                expect(triangle).toBeInstanceOf(ExtendedTriangle);
                expect(triangle.needsUpdate).toBe(true);
                expect(contained).toBe(false);
                expect(depth).toBeGreaterThan(0);
                expect(triangle.a.toArray()).toEqual([positions.getX(index * 3), positions.getY(index * 3), positions.getZ(index * 3)]);
                expect(triangle.c.toArray()).toEqual([positions.getX(index * 3 + 2), positions.getY(index * 3 + 2), positions.getZ(index * 3 + 2)]);
                seen++;
            },
        });
        expect(seen).toBe(ours.triangleCount);
    });

    it('reports a contained box once per triangle, with contained set; stops on a truthy return; lets intersectsRange decide first', () => {
        const [{ ours, theirs }] = fixtures();
        for (const tree of [ours, theirs]) {
            const contained = tree === ours ? BVH_CONTAINED : TMB_CONTAINED;
            const indices: number[] = [];
            const flags = new Set<boolean>();
            (tree as BvhBoundsTree).shapecast({
                intersectsBounds: () => contained,
                intersectsTriangle: (_triangle, index, isContained) => {
                    indices.push(index);
                    flags.add(isContained);
                },
            });
            expect(sorted(indices)).toEqual(Array.from({ length: tree === ours ? ours.triangleCount : indices.length }, (_, i) => i));
            expect([...flags]).toEqual([true]);
            // Neither range nor triangle callback: a contained box stops the cast, an intersected one doesn't.
            expect((tree as BvhBoundsTree).shapecast({ intersectsBounds: () => contained })).toBe(true);
            expect((tree as BvhBoundsTree).shapecast({ intersectsBounds: () => (tree === ours ? BVH_INTERSECTED : TMB_INTERSECTED) })).toBe(false);
            // A truthy intersectsTriangle stops at once.
            let calls = 0;
            expect((tree as BvhBoundsTree).shapecast({ intersectsBounds: () => true, intersectsTriangle: () => ++calls === 5 })).toBe(true);
            expect(calls).toBe(5);
            // intersectsRange first: truthy skips the triangles and stops; falsy lets them through.
            let triangles = 0;
            expect((tree as BvhBoundsTree).shapecast({ intersectsBounds: () => true, intersectsRange: () => true, intersectsTriangle: () => void triangles++ })).toBe(true);
            expect(triangles).toBe(0);
            expect((tree as BvhBoundsTree).shapecast({ intersectsBounds: () => true, intersectsRange: () => false, intersectsTriangle: () => void triangles++ })).toBe(false);
            expect(triangles).toBe(ours.triangleCount);
        }
    });

    it('visits nearer boxes first with boundsTraverseOrder, and nests', () => {
        const [{ ours }] = fixtures();
        const point = new Vector3(0.3, -0.2, 0.1);
        const scores: number[] = [];
        let firstIndex = -1;
        ours.shapecast({
            boundsTraverseOrder: (box) => box.distanceToPoint(point),
            intersectsBounds: (_box, _isLeaf, score) => {
                scores.push(score!);
                return true;
            },
            intersectsTriangle: (_triangle, index) => {
                firstIndex = index;
                return true;
            },
        });
        expect(scores.every((score) => typeof score === 'number')).toBe(true);
        // The first leaf reached holds a triangle near the point.
        const triangle = new ExtendedTriangle();
        ours.getTriangle(ours.treePositionOf(firstIndex), [] as number[]).forEach((value, i) => triangle.points[Math.floor(i / 3)].setComponent(i % 3, value));
        expect(triangle.distanceToPoint(point)).toBeLessThan(0.5);
        // A cast inside a cast's callback gets its own scratch triangle.
        let inner = 0;
        ours.shapecast({
            intersectsBounds: () => true,
            intersectsTriangle: (outer) => {
                const before = outer.a.clone();
                ours.shapecast({ intersectsBounds: () => true, intersectsTriangle: () => ++inner > 3 });
                expect(outer.a.equals(before)).toBe(true);
                return true;
            },
        });
        expect(inner).toBe(4);
    });
});

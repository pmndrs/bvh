/**
 * `closestPointToPoint` and `closestPointToGeometry` against three-mesh-bvh
 * 0.9.18 on the same geometry, for random points and rigid placements: the
 * same distance (to 1e-9; both measure in doubles from the same f32
 * corners), and the same triangle unless another one is equally near (a
 * shared edge or corner: the trees differ, so which of the tied triangles is
 * met first differs). Both run through `geometry.boundsTree` (three's types)
 * and the core `CpuMeshBvh` (plain types).
 *
 * three-mesh-bvh's tree is in indirect mode for `closestPointToPoint`, so its
 * `faceIndex` is the original triangle. For `closestPointToGeometry` it is
 * not (in 0.9.18's indirect mode that `faceIndex` is a tree position), so its
 * geometry's index is reordered and its triangles are matched to ours by
 * their vertex indices.
 */

import { BufferAttribute, BufferGeometry, IcosahedronGeometry, Matrix4, Quaternion, TorusKnotGeometry, Triangle, Vector3 } from 'three';
import { MeshBVH as TmbMeshBVH, type HitPointInfo as TmbHitPointInfo } from 'three-mesh-bvh';
import { beforeAll, describe, expect, it } from 'vitest';
import { CpuMeshBvh, type BvhClosestPoint } from '../query/cpu-mesh-bvh.ts';
import { createRandom, randomTriangleSoup, randomUnitVector, type Random } from '../trace/oracle-fixtures.ts';
import { BvhBoundsTree, meshInputFromGeometry, prepareBoundsTrees, setBoundsTree, type BvhClosestPointInfo } from './raycast.ts';
import { ExtendedTriangle } from './shapes.ts';

beforeAll(async () => {
    await prepareBoundsTrees();
});

const TOLERANCE = 1e-9;

function soupGeometry(random: Random, count: number): BufferGeometry {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(randomTriangleSoup(random, count, 1, 0.15).positions, 3));
    return geometry;
}

function corners(geometry: BufferGeometry, face: number): Triangle {
    const position = geometry.getAttribute('position');
    const vertex = (corner: number) => (geometry.index ? geometry.index.getX(face * 3 + corner) : face * 3 + corner);
    return new Triangle(new Vector3().fromBufferAttribute(position, vertex(0)), new Vector3().fromBufferAttribute(position, vertex(1)), new Vector3().fromBufferAttribute(position, vertex(2)));
}

function randomPoint(random: Random, extent: number): Vector3 {
    return new Vector3((random() * 2 - 1) * extent, (random() * 2 - 1) * extent, (random() * 2 - 1) * extent);
}

function rigid(random: Random, distance: number): Matrix4 {
    const axis = new Vector3(...randomUnitVector(random));
    const rotation = new Quaternion().setFromAxisAngle(axis, random() * Math.PI * 2);
    const offset = new Vector3(...randomUnitVector(random)).multiplyScalar(distance * random());
    return new Matrix4().compose(offset, rotation, new Vector3(1, 1, 1));
}

/** Vertex indices of a triangle, as a key: the same triangle whatever order a tree put it in. */
function vertexKey(geometry: BufferGeometry, face: number): string {
    const index = geometry.index;
    return index ? `${index.getX(face * 3)},${index.getX(face * 3 + 1)},${index.getX(face * 3 + 2)}` : `${face * 3},${face * 3 + 1},${face * 3 + 2}`;
}

function originalFaces(geometry: BufferGeometry): Map<string, number> {
    const count = (geometry.index ? geometry.index.count : geometry.getAttribute('position').count) / 3;
    const map = new Map<string, number>();
    for (let face = 0; face < count; face++) map.set(vertexKey(geometry, face), face);
    return map;
}

describe('closestPointToPoint against three-mesh-bvh', () => {
    const random = createRandom(61);
    const meshes = [
        { name: 'soup', geometry: soupGeometry(random, 3000) },
        { name: 'torus knot', geometry: new TorusKnotGeometry(0.7, 0.25, 96, 16) },
    ];

    it('finds the same distance and triangle (unless tied) for random points', () => {
        let same = 0;
        let tied = 0;
        for (const { name, geometry } of meshes) {
            const ours = new BvhBoundsTree(geometry);
            const theirs = new TmbMeshBVH(geometry.clone(), { indirect: true });
            for (let query = 0; query < 400; query++) {
                const point = randomPoint(random, query % 4 === 0 ? 3 : 1.2);
                const expected = theirs.closestPointToPoint(point)!;
                const result = ours.closestPointToPoint(point)!;
                expect(result.point).toBeInstanceOf(Vector3);
                expect(Math.abs(result.distance - expected.distance), `${name} #${query}`).toBeLessThan(TOLERANCE);
                expect(result.point.distanceTo(point) - result.distance).toBeLessThan(TOLERANCE);
                if (result.faceIndex === expected.faceIndex) {
                    same++;
                    expect(result.point.distanceTo(expected.point)).toBeLessThan(1e-7);
                } else {
                    // Another triangle at the same distance: both are right.
                    const theirFace = corners(geometry, expected.faceIndex).closestPointToPoint(point, new Vector3()).distanceTo(point);
                    const ourFace = corners(geometry, result.faceIndex).closestPointToPoint(point, new Vector3()).distanceTo(point);
                    expect(Math.abs(theirFace - ourFace), `${name} #${query}`).toBeLessThan(TOLERANCE);
                    tied++;
                }
            }
        }
        expect(same + tied).toBe(800);
        console.log(`closestPointToPoint: ${same} same triangle, ${tied} tied, of 800`);
        expect(same).toBeGreaterThan(0);
    });

    it('the core tree gives the same answers with plain types', async () => {
        const { createTinyBvhBuilder } = await import('../tinybvh.ts');
        const builder = await createTinyBvhBuilder();
        for (const { name, geometry } of meshes) {
            const ours = new BvhBoundsTree(geometry);
            const core = CpuMeshBvh.build(meshInputFromGeometry(geometry), builder);
            for (let query = 0; query < 100; query++) {
                const point = randomPoint(random, 1.5);
                const three = ours.closestPointToPoint(point)!;
                const plain = core.closestPointToPoint([point.x, point.y, point.z]) as BvhClosestPoint;
                expect(plain.point).not.toBeInstanceOf(Vector3);
                expect(plain.distance, `${name} #${query}`).toBe(three.distance);
                expect(plain.faceIndex).toBe(three.faceIndex);
            }
        }
    });

    it('follows the thresholds and fills the target as three-mesh-bvh does', () => {
        const { geometry } = meshes[1];
        const ours = new BvhBoundsTree(geometry);
        const theirs = new TmbMeshBVH(geometry.clone(), { indirect: true });
        let early = 0;
        let none = 0;
        for (let query = 0; query < 300; query++) {
            const point = randomPoint(random, 2);
            const exact = theirs.closestPointToPoint(point)!.distance;
            // maxThreshold below the answer: nothing (or something inside a box nearer than it, as theirs).
            const max = exact * (0.5 + random());
            const theirsMax = theirs.closestPointToPoint(point, {} as TmbHitPointInfo, 0, max);
            const oursMax = ours.closestPointToPoint(point, {}, 0, max);
            if (theirsMax === null) {
                none++;
                expect(oursMax === null || oursMax.distance >= max - TOLERANCE, `#${query}`).toBe(true);
            }
            if (oursMax === null) expect(theirsMax === null || theirsMax.distance >= max - TOLERANCE).toBe(true);
            if (oursMax && theirsMax && oursMax.distance < max && theirsMax.distance < max) expect(Math.abs(oursMax.distance - theirsMax.distance)).toBeLessThan(TOLERANCE);
            // minThreshold above the answer: an early answer below it (any triangle that close).
            const min = exact * (1 + random());
            const oursMin = ours.closestPointToPoint(point, {}, min)!;
            expect(oursMin.distance).toBeLessThan(min);
            if (oursMin.distance > exact + TOLERANCE) early++;
            // min = max: "is anything within this distance?"
            const within = exact * (0.5 + random());
            expect(ours.closestPointToPoint(point, {}, within, within) !== null && ours.closestPointToPoint(point, {}, within, within)!.distance < within).toBe(within > exact);
        }
        expect(none).toBeGreaterThan(50);
        expect(early).toBeGreaterThan(0);
        // An existing point is written in place; a missed query leaves the target alone.
        const target = { point: new Vector3(), distance: -1, faceIndex: -1 };
        const kept = target.point;
        expect(ours.closestPointToPoint(new Vector3(5, 0, 0), target)).toBe(target);
        expect(target.point).toBe(kept);
        expect(target.distance).toBeGreaterThan(0);
        const untouched = { point: new Vector3(9, 9, 9), distance: -1, faceIndex: -1 };
        expect(ours.closestPointToPoint(new Vector3(5, 0, 0), untouched, 0, 0.01)).toBeNull();
        expect(untouched).toEqual({ point: new Vector3(9, 9, 9), distance: -1, faceIndex: -1 });
    });
});

describe('closestPointToGeometry against three-mesh-bvh', () => {
    const random = createRandom(62);

    function pair(otherHasTree: boolean) {
        const geometry = new TorusKnotGeometry(0.7, 0.25, 64, 12);
        const other = random() < 0.5 ? new IcosahedronGeometry(0.3, 2) : soupGeometry(random, 300).scale(0.4, 0.4, 0.4);
        const ours = new BvhBoundsTree(geometry);
        if (otherHasTree) setBoundsTree(other, new BvhBoundsTree(other));
        const theirsGeometry = geometry.clone();
        const theirs = new TmbMeshBVH(theirsGeometry);
        const theirsOther = other.clone();
        if (otherHasTree) (theirsOther as unknown as { boundsTree: TmbMeshBVH }).boundsTree = new TmbMeshBVH(theirsOther);
        return { geometry, other, ours, theirs, theirsGeometry, theirsOther };
    }

    for (const otherHasTree of [true, false]) {
        it(`finds the same distances and triangles (unless tied), the other ${otherHasTree ? 'with a tree' : 'a bare mesh'}`, () => {
            let same = 0;
            let tied = 0;
            const cases = otherHasTree ? 60 : 25;
            for (let query = 0; query < cases; query++) {
                const { geometry, other, ours, theirs, theirsGeometry, theirsOther } = pair(otherHasTree);
                const faces = originalFaces(geometry);
                const otherFaces = originalFaces(other);
                const matrix = rigid(random, 2);
                const theirs1 = {} as TmbHitPointInfo;
                const theirs2 = {} as TmbHitPointInfo;
                const expected = theirs.closestPointToGeometry(theirsOther, matrix, theirs1, theirs2)!;
                const ours1: Partial<BvhClosestPointInfo> = {};
                const ours2: Partial<BvhClosestPointInfo> = {};
                const result = ours.closestPointToGeometry(other, matrix, ours1, ours2)!;
                expect(result).toBe(ours1);
                expect(Math.abs(result.distance - expected.distance), `#${query}`).toBeLessThan(TOLERANCE);
                expect(Math.abs(ours2.distance! - theirs2.distance), `#${query}`).toBeLessThan(1e-7);
                // target2's point is in the other's space: moved back, it is result.distance from target1's.
                if (result.distance > 1e-6) expect(Math.abs(ours2.point!.clone().applyMatrix4(matrix).distanceTo(result.point) - result.distance)).toBeLessThan(1e-7);
                const theirFace = faces.get(vertexKey(theirsGeometry, expected.faceIndex))!;
                const theirOtherFace = otherFaces.get(vertexKey(theirsOther, theirs2.faceIndex))!;
                if (theirFace === result.faceIndex && theirOtherFace === ours2.faceIndex) {
                    same++;
                    expect(result.point.distanceTo(expected.point)).toBeLessThan(1e-7);
                } else {
                    const a = new ExtendedTriangle().copy(corners(geometry, theirFace));
                    const b = new ExtendedTriangle().copy(corners(other, theirOtherFace));
                    b.a.applyMatrix4(matrix);
                    b.b.applyMatrix4(matrix);
                    b.c.applyMatrix4(matrix);
                    b.needsUpdate = true;
                    expect(Math.abs(a.distanceToTriangle(b) - result.distance), `#${query}`).toBeLessThan(TOLERANCE);
                    tied++;
                }
            }
            expect(same + tied).toBe(cases);
            console.log(`closestPointToGeometry (${otherHasTree ? "tree" : "mesh"}): ${same} same pair, ${tied} tied, of ${cases}`);
            expect(same).toBeGreaterThan(0);
        });
    }

    it('stops below minThreshold, returns null past maxThreshold, and skips a null target2', () => {
        const { other, ours, theirs, theirsOther } = pair(true);
        const matrix = new Matrix4().makeTranslation(3, 0, 0);
        const exact = theirs.closestPointToGeometry(theirsOther, matrix)!.distance;
        expect(Math.abs(ours.closestPointToGeometry(other, matrix)!.distance - exact)).toBeLessThan(TOLERANCE);
        expect(ours.closestPointToGeometry(other, matrix, {}, {}, 0, exact * 0.9)).toBeNull();
        expect(theirs.closestPointToGeometry(theirsOther, matrix, {} as TmbHitPointInfo, {} as TmbHitPointInfo, 0, exact * 0.9)).toBeNull();
        const early = ours.closestPointToGeometry(other, matrix, {}, {}, exact * 3)!;
        expect(early.distance).toBeLessThan(exact * 3);
        const target1 = {};
        expect(ours.closestPointToGeometry(other, matrix, target1, null)).toBe(target1);
        // The core with plain types and a mesh as the other.
        const core = ours as CpuMeshBvh<unknown>;
        const plain = CpuMeshBvh.prototype.closestPointToGeometry.call(core, meshInputFromGeometry(other), matrix.elements)!;
        expect(Math.abs(plain.distance - exact)).toBeLessThan(TOLERANCE);
    });
});

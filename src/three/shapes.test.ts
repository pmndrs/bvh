/**
 * The shape math against three-mesh-bvh 0.9.18's own classes, bit for bit:
 * random triangles (with coplanar, touching, snapped and degenerate ones),
 * segments, spheres, boxes and transforms, through the core classes
 * (`src/query/shapes.ts`) and the three-flavoured ones (`./shapes.ts`).
 * Every boolean, distance and point must be identical (`Object.is`).
 */

import { Box3, Line3, Matrix4, Sphere, Triangle, Vector3 } from 'three';
import { ExtendedTriangle as TmbExtendedTriangle, OrientedBox as TmbOrientedBox } from 'three-mesh-bvh';
import { describe, expect, it } from 'vitest';
import { ExtendedTriangle as CoreExtendedTriangle, OrientedBox as CoreOrientedBox, BvhSegment, BvhVector3 } from '../query/shapes.ts';
import { createRandom, randomAffine, type Random } from '../trace/oracle-fixtures.ts';
import { ExtendedTriangle, OrientedBox } from './shapes.ts';

type V = { x: number; y: number; z: number };

/** Counts differing numbers (Object.is) between two lists; reports the first. */
class Diffs {
    count = 0;
    checks = 0;
    first = '';
    same(label: string, ours: unknown, theirs: unknown): void {
        this.checks++;
        if (!Object.is(ours, theirs)) {
            this.count++;
            if (!this.first) this.first = `${label}: ours ${String(ours)}, theirs ${String(theirs)}`;
        }
    }
    point(label: string, ours: V, theirs: V): void {
        this.same(`${label}.x`, ours.x, theirs.x);
        this.same(`${label}.y`, ours.y, theirs.y);
        this.same(`${label}.z`, ours.z, theirs.z);
    }
}

/** Snapped coordinates make exact touches and coplanar overlaps likely. */
function coordinate(random: Random, snap: boolean): number {
    const value = random() * 2 - 1;
    return snap ? Math.round(value * 4) / 4 : Math.fround(value);
}

function randomPoint(random: Random, snap: boolean): Vector3 {
    return new Vector3(coordinate(random, snap), coordinate(random, snap), coordinate(random, snap));
}

/** A random triangle: general, snapped, flat at z = 0 (coplanar with others), a segment, or a point. */
function randomTriangle(random: Random): [Vector3, Vector3, Vector3] {
    const kind = Math.floor(random() * 10);
    const snap = kind >= 4;
    const a = randomPoint(random, snap);
    const b = randomPoint(random, snap);
    const c = randomPoint(random, snap);
    if (kind === 6 || kind === 7) a.z = b.z = c.z = 0;
    if (kind === 8) c.copy(random() < 0.5 ? a : b); // a segment
    if (kind === 9) {
        // a point
        b.copy(a);
        c.copy(a);
    }
    return [a, b, c];
}

function randomMatrix(random: Random): Matrix4 {
    return new Matrix4().fromArray(randomAffine(random, { extent: 1, mirror: random() < 0.3 }));
}

function randomBox(random: Random): Box3 {
    const a = randomPoint(random, random() < 0.3);
    const b = randomPoint(random, random() < 0.3);
    return new Box3(a.clone().min(b), a.clone().max(b));
}

describe('ExtendedTriangle against three-mesh-bvh', () => {
    it('gives the same intersections, segments, distances and closest points', () => {
        const random = createRandom(41);
        const diffs = new Diffs();
        let intersecting = 0;
        let coplanar = 0;
        const pairs = 4000;
        const warn = console.warn;
        console.warn = () => {};
        try {
            for (let pair = 0; pair < pairs; pair++) {
                const [a1, b1, c1] = randomTriangle(random);
                const [a2, b2, c2] = random() < 0.2 ? [a1.clone(), randomPoint(random, true), randomPoint(random, true)] : randomTriangle(random);
                const theirs1 = new TmbExtendedTriangle(a1.clone(), b1.clone(), c1.clone());
                const theirs2 = new TmbExtendedTriangle(a2.clone(), b2.clone(), c2.clone());
                const three1 = new ExtendedTriangle(a1.clone(), b1.clone(), c1.clone());
                const three2 = new ExtendedTriangle(a2.clone(), b2.clone(), c2.clone());
                const core1 = new CoreExtendedTriangle(a1, b1, c1);
                const core2 = new CoreExtendedTriangle(a2, b2, c2);
                const plain2 = new Triangle(a2.clone(), b2.clone(), c2.clone());

                const theirsEdge = new Line3();
                const theirsHit = (theirs1 as unknown as ExtendedTriangle).intersectsTriangle(theirs2, theirsEdge, true);
                if (theirsHit) intersecting++;
                if (theirsHit && a1.z === 0 && b1.z === 0 && c1.z === 0 && a2.z === 0 && b2.z === 0 && c2.z === 0) coplanar++;
                for (const [name, ours, other] of [
                    ['three', three1, three2],
                    ['core', core1, core2],
                    ['three/plain', three1, plain2],
                ] as const) {
                    const edge = new Line3();
                    diffs.same(`${name} intersectsTriangle #${pair}`, ours.intersectsTriangle(other as Triangle, edge as Line3, true), theirsHit);
                    if (theirsHit) {
                        diffs.point(`${name} edge.start #${pair}`, edge.start, theirsEdge.start);
                        diffs.point(`${name} edge.end #${pair}`, edge.end, theirsEdge.end);
                    }
                }
                // Without a target too (a different path for the degenerate cases).
                diffs.same(`core intersectsTriangle no target #${pair}`, core2.intersectsTriangle(core1), theirs2.intersectsTriangle(theirs1));

                const theirsT1 = new Vector3();
                const theirsT2 = new Vector3();
                const theirsDistance = (theirs1 as unknown as ExtendedTriangle).distanceToTriangle(theirs2, theirsT1, theirsT2);
                for (const [name, ours, other] of [
                    ['three', three1, three2],
                    ['core', core1, core2],
                ] as const) {
                    const t1 = new Vector3();
                    const t2 = new Vector3();
                    diffs.same(`${name} distanceToTriangle #${pair}`, ours.distanceToTriangle(other as Triangle, t1, t2), theirsDistance);
                    diffs.point(`${name} distanceToTriangle t1 #${pair}`, t1, theirsT1);
                    diffs.point(`${name} distanceToTriangle t2 #${pair}`, t2, theirsT2);
                }

                const segment = new Line3(randomPoint(random, random() < 0.3).multiplyScalar(1.5), randomPoint(random, random() < 0.3).multiplyScalar(1.5));
                if (random() < 0.1) segment.end.copy(segment.start);
                const theirsS1 = new Vector3();
                const theirsS2 = new Vector3();
                const theirsSegment = theirs1.closestPointToSegment(segment, theirsS1, theirsS2);
                for (const [name, ours] of [
                    ['three', three1],
                    ['core', core1],
                ] as const) {
                    const s1 = new Vector3();
                    const s2 = new Vector3();
                    const oursSegment = name === 'core' ? core1.closestPointToSegment(new BvhSegment(segment.start, segment.end), s1, s2) : (ours as ExtendedTriangle).closestPointToSegment(segment, s1, s2);
                    diffs.same(`${name} closestPointToSegment #${pair}`, oursSegment, theirsSegment);
                    diffs.point(`${name} closestPointToSegment t1 #${pair}`, s1, theirsS1);
                    diffs.point(`${name} closestPointToSegment t2 #${pair}`, s2, theirsS2);
                    diffs.same(`${name} distanceToPoint #${pair}`, ours.distanceToPoint(segment.start), theirs1.distanceToPoint(segment.start));
                }

                const sphere = new Sphere(randomPoint(random, false), random() * 0.6);
                const theirsSphere = theirs1.intersectsSphere(sphere);
                diffs.same(`three intersectsSphere #${pair}`, three1.intersectsSphere(sphere), theirsSphere);
                diffs.same(`core intersectsSphere #${pair}`, core1.intersectsSphere(sphere), theirsSphere);
            }
        } finally {
            console.warn = warn;
        }
        expect(diffs.first).toBe('');
        expect(diffs.count).toBe(0);
        // The cases are worth having: some intersect, some of those coplanar.
        expect(intersecting).toBeGreaterThan(pairs / 10);
        expect(coplanar).toBeGreaterThan(20);
    });

    it('marks itself for update on set and copy, and caches what three-mesh-bvh caches', () => {
        const triangle = new ExtendedTriangle();
        triangle.update();
        expect(triangle.needsUpdate).toBe(false);
        triangle.set(new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0));
        expect(triangle.needsUpdate).toBe(true);
        triangle.update();
        expect(triangle.plane.normal.toArray()).toEqual([0, 0, 1]);
        expect(triangle.isDegenerateIntoPoint || triangle.isDegenerateIntoSegment).toBe(false);
        triangle.copy(new Triangle(new Vector3(), new Vector3(1, 1, 1), new Vector3(2, 2, 2)));
        expect(triangle.needsUpdate).toBe(true);
        triangle.update();
        expect(triangle.isDegenerateIntoSegment).toBe(false); // collinear but no zero-length edge: three-mesh-bvh treats it as a triangle
        triangle.copy(new Triangle(new Vector3(), new Vector3(), new Vector3(2, 2, 2)));
        triangle.update();
        expect(triangle.isDegenerateIntoSegment).toBe(true);
        expect(triangle).toBeInstanceOf(Triangle);
    });
});

describe('OrientedBox against three-mesh-bvh', () => {
    it('gives the same box and triangle tests, closest points and distances', () => {
        const random = createRandom(43);
        const diffs = new Diffs();
        let boxHits = 0;
        let triangleHits = 0;
        const count = 3000;
        for (let i = 0; i < count; i++) {
            const local = randomBox(random);
            const matrix = randomMatrix(random);
            const theirs = new TmbOrientedBox(local.min, local.max);
            theirs.set(local.min, local.max, matrix);
            const three = new OrientedBox().set(local.min, local.max, matrix);
            const core = new CoreOrientedBox().set(local.min, local.max, matrix.elements);

            const box = randomBox(random);
            box.min.multiplyScalar(2);
            box.max.multiplyScalar(2);
            const theirsBox = theirs.intersectsBox(box);
            if (theirsBox) boxHits++;
            diffs.same(`three intersectsBox #${i}`, three.intersectsBox(box), theirsBox);
            diffs.same(`core intersectsBox #${i}`, core.intersectsBox(box), theirsBox);

            const [a, b, c] = randomTriangle(random);
            const triangle = new Triangle(a.multiplyScalar(2), b.multiplyScalar(2), c.multiplyScalar(2));
            const theirsTriangle = theirs.intersectsTriangle(triangle);
            if (theirsTriangle) triangleHits++;
            diffs.same(`three intersectsTriangle #${i}`, three.intersectsTriangle(triangle), theirsTriangle);
            diffs.same(`core intersectsTriangle #${i}`, core.intersectsTriangle(triangle), theirsTriangle);
            diffs.same(`three intersectsTriangle (extended) #${i}`, three.intersectsTriangle(new ExtendedTriangle(a, b, c)), theirsTriangle);

            const point = randomPoint(random, false).multiplyScalar(3);
            const theirsClosest = theirs.closestPointToPoint(point, new Vector3()) as unknown as Vector3;
            diffs.point(`three closestPointToPoint #${i}`, three.closestPointToPoint(point, new Vector3()), theirsClosest);
            diffs.point(`core closestPointToPoint #${i}`, core.closestPointToPoint(point, new BvhVector3()), theirsClosest);
            diffs.same(`three distanceToPoint #${i}`, three.distanceToPoint(point), theirs.distanceToPoint(point));

            const threshold = random() < 0.5 ? 0 : random() * 0.5;
            // three-mesh-bvh 0.9.18 calls Box3.closestPointToPoint, gone from three 0.187, for
            // overlapping boxes given targets: ask it for targets only when they don't overlap.
            const theirsT1 = new Vector3();
            const theirsT2 = new Vector3();
            const theirsDistance = theirsBox ? theirs.distanceToBox(box, threshold) : theirs.distanceToBox(box, threshold, theirsT1, theirsT2);
            const t1 = new Vector3();
            const t2 = new Vector3();
            diffs.same(`three distanceToBox #${i}`, three.distanceToBox(box, threshold, t1, t2), theirsDistance);
            if (!theirsBox) {
                diffs.point(`three distanceToBox t1 #${i}`, t1, theirsT1);
                diffs.point(`three distanceToBox t2 #${i}`, t2, theirsT2);
            } else {
                // Ours: the box's point nearest the oriented box's, inside both when they overlap this much.
                expect(box.containsPoint(t2)).toBe(true);
            }
            diffs.same(`core distanceToBox #${i}`, core.distanceToBox(box, threshold), theirsDistance);
        }
        expect(diffs.first).toBe('');
        expect(diffs.count).toBe(0);
        expect(boxHits).toBeGreaterThan(count / 10);
        expect(boxHits).toBeLessThan(count);
        expect(triangleHits).toBeGreaterThan(count / 10);
        expect(triangleHits).toBeLessThan(count);
    });
});

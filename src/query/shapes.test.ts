/**
 * The engine-free query shapes on hand-made cases with known answers (the
 * kinds three-mesh-bvh's own math tests cover): crossing, touching,
 * coplanar, separated and degenerate triangles, segments, spheres, oriented
 * boxes. The bit-for-bit comparison with three-mesh-bvh on random cases is
 * `src/three/shapes.test.ts`.
 */

import { describe, expect, it, vi } from 'vitest';
import { BvhSegment, BvhVector3, closestPointsSegmentToSegment, ExtendedTriangle, OrientedBox, SeparatingAxisBounds, type BvhVector3Like } from './shapes.ts';

const v = (x = 0, y = 0, z = 0) => new BvhVector3(x, y, z);
const xyz = (p: BvhVector3Like) => [p.x, p.y, p.z];
/** The right triangle (0,0,0), (2,0,0), (0,2,0) in the z = 0 plane. */
const base = () => new ExtendedTriangle(v(0, 0, 0), v(2, 0, 0), v(0, 2, 0));

describe('ExtendedTriangle.intersectsTriangle', () => {
    it('finds the segment where two triangles cross', () => {
        const other = new ExtendedTriangle(v(0.5, 0.5, -1), v(0.5, 0.5, 1), v(1.5, 0.5, 0));
        const edge = new BvhSegment();
        expect(base().intersectsTriangle(other, edge)).toBe(true);
        const xs = [edge.start.x, edge.end.x].sort((a, b) => a - b);
        expect(xs[0]).toBeCloseTo(0.5, 12);
        expect(xs[1]).toBeCloseTo(1.5, 12);
        expect([edge.start.y, edge.end.y, edge.start.z, edge.end.z]).toEqual([0.5, 0.5, 0, 0]);
    });

    it('rejects parallel and separated triangles, and accepts ones touching at a corner', () => {
        expect(base().intersectsTriangle(new ExtendedTriangle(v(0, 0, 1), v(2, 0, 1), v(0, 2, 1)))).toBe(false);
        expect(base().intersectsTriangle(new ExtendedTriangle(v(3, 3, -1), v(3, 3, 1), v(4, 3, 0)))).toBe(false);
        expect(base().intersectsTriangle(new ExtendedTriangle(v(0, 0, 0), v(-1, 0, 1), v(-1, 0, -1)))).toBe(true);
    });

    it('tests coplanar triangles by separating axes, with a zero segment and a warning unless suppressed', () => {
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
        const edge = new BvhSegment(v(1, 1, 1), v(1, 1, 1));
        expect(base().intersectsTriangle(new ExtendedTriangle(v(0.5, 0.5, 0), v(3, 0.5, 0), v(0.5, 3, 0)), edge)).toBe(true);
        expect([...xyz(edge.start), ...xyz(edge.end)]).toEqual([0, 0, 0, 0, 0, 0]);
        expect(warn).toHaveBeenCalledTimes(1);
        expect(base().intersectsTriangle(new ExtendedTriangle(v(0.5, 0.5, 0), v(3, 0.5, 0), v(0.5, 3, 0)), edge, true)).toBe(true);
        expect(warn).toHaveBeenCalledTimes(1);
        expect(base().intersectsTriangle(new ExtendedTriangle(v(3, 3, 0), v(4, 3, 0), v(3, 4, 0)))).toBe(false);
        warn.mockRestore();
    });

    it('handles triangles that are a segment or a point', () => {
        const segment = new ExtendedTriangle(v(0.5, 0.5, -1), v(0.5, 0.5, 1), v(0.5, 0.5, 1));
        const edge = new BvhSegment();
        expect(base().intersectsTriangle(segment, edge)).toBe(true);
        expect(xyz(edge.start)).toEqual([0.5, 0.5, 0]);
        segment.update();
        expect(segment.isDegenerateIntoSegment).toBe(true);
        expect(segment.intersectsTriangle(base())).toBe(true);
        expect(base().intersectsTriangle(new ExtendedTriangle(v(3, 3, -1), v(3, 3, 1), v(3, 3, 1)))).toBe(false);
        const point = new ExtendedTriangle(v(0.5, 0.5, 0), v(0.5, 0.5, 0), v(0.5, 0.5, 0));
        expect(base().intersectsTriangle(point)).toBe(true);
        point.update();
        expect(point.isDegenerateIntoPoint).toBe(true);
        expect(base().intersectsTriangle(new ExtendedTriangle(v(0.5, 0.5, 0.5), v(0.5, 0.5, 0.5), v(0.5, 0.5, 0.5)))).toBe(false);
    });

    it('accepts a plain triangle as the other one', () => {
        expect(base().intersectsTriangle({ a: v(0.5, 0.5, -1), b: v(0.5, 0.5, 1), c: v(1.5, 0.5, 0) })).toBe(true);
    });
});

describe('ExtendedTriangle distances', () => {
    it('closestPointToSegment: zero through the face, else the nearest pair', () => {
        const t1 = v(9, 9, 9);
        const t2 = v(9, 9, 9);
        expect(base().closestPointToSegment(new BvhSegment(v(0.5, 0.5, -1), v(0.5, 0.5, 1)), t1, t2)).toBe(0);
        expect(xyz(t1)).toEqual([0.5, 0.5, 0]);
        expect(xyz(t2)).toEqual([0.5, 0.5, 0]);
        expect(base().closestPointToSegment(new BvhSegment(v(0.25, 0.25, 1), v(0.5, 0.25, 1)), t1, t2)).toBe(1);
        expect(xyz(t1)).toEqual([0.25, 0.25, 0]);
        expect(xyz(t2)).toEqual([0.25, 0.25, 1]);
        // Beyond an edge: the edge's point.
        expect(base().closestPointToSegment(new BvhSegment(v(-1, 1, -1), v(-1, 1, 1)), t1, t2)).toBeCloseTo(1, 12);
        expect(xyz(t1)).toEqual([0, 1, 0]);
    });

    it('distanceToPoint, distanceToTriangle and intersectsSphere', () => {
        expect(base().distanceToPoint(v(3, 0, 0))).toBe(1);
        expect(base().distanceToPoint(v(0.5, 0.5, 2))).toBe(2);
        const t1 = v();
        const t2 = v();
        expect(base().distanceToTriangle(new ExtendedTriangle(v(0, 0, 1), v(2, 0, 1), v(0, 2, 1)), t1, t2)).toBe(1);
        expect(t2.z - t1.z).toBe(1);
        expect(base().distanceToTriangle(new ExtendedTriangle(v(0.5, 0.5, -1), v(0.5, 0.5, 1), v(1.5, 0.5, 0)))).toBe(0);
        expect(base().intersectsSphere({ center: v(0.5, 0.5, 0.5), radius: 0.6 })).toBe(true);
        expect(base().intersectsSphere({ center: v(0.5, 0.5, 0.5), radius: 0.4 })).toBe(false);
        expect(base().intersectsSphere({ center: v(-0.5, 1, 0), radius: 0.5 })).toBe(true);
    });

    it('closestPointsSegmentToSegment: crossing, parallel-offset and end-to-end', () => {
        const p1 = v();
        const p2 = v();
        closestPointsSegmentToSegment(new BvhSegment(v(-1, 0, 0), v(1, 0, 0)), new BvhSegment(v(0, -1, 1), v(0, 1, 1)), p1, p2);
        expect([...xyz(p1), ...xyz(p2)]).toEqual([0, 0, 0, 0, 0, 1]);
        closestPointsSegmentToSegment(new BvhSegment(v(0, 0, 0), v(1, 0, 0)), new BvhSegment(v(2, 1, 0), v(3, 1, 0)), p1, p2);
        expect([...xyz(p1), ...xyz(p2)]).toEqual([1, 0, 0, 2, 1, 0]);
    });
});

describe('OrientedBox', () => {
    /** The cube [-1, 1]^3 turned 45 degrees about z: its tips reach x = ±√2. */
    const turned = () => {
        const c = Math.SQRT1_2;
        return new OrientedBox(v(-1, -1, -1), v(1, 1, 1), [c, c, 0, 0, -c, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    };

    it('intersectsBox by separating axes, the box axes and its own', () => {
        expect(turned().intersectsBox({ min: v(1.2, -0.1, -0.1), max: v(2, 0.1, 0.1) })).toBe(true);
        expect(turned().intersectsBox({ min: v(1.5, -0.1, -0.1), max: v(2, 0.1, 0.1) })).toBe(false);
        // Inside the axis-aligned bounds of the turned cube, but off its corner.
        expect(turned().intersectsBox({ min: v(1.1, 1.1, -0.1), max: v(1.3, 1.3, 0.1) })).toBe(false);
    });

    it('intersectsTriangle, closest points and distances', () => {
        const box = turned();
        expect(box.intersectsTriangle({ a: v(0, 0, 0), b: v(5, 0, 0), c: v(0, 5, 0) })).toBe(true);
        expect(box.intersectsTriangle({ a: v(1.2, 1.2, -1), b: v(3, 1.2, -1), c: v(1.2, 3, 1) })).toBe(false);
        const closest = box.closestPointToPoint(v(3, 0, 0), v());
        expect(closest.x).toBeCloseTo(Math.SQRT2, 12);
        expect(closest.y).toBeCloseTo(0, 12);
        expect(box.distanceToPoint(v(3, 0, 0))).toBeCloseTo(3 - Math.SQRT2, 12);
        const straight = new OrientedBox(v(-1, -1, -1), v(1, 1, 1));
        expect(straight.distanceToBox({ min: v(2, -1, -1), max: v(3, 1, 1) })).toBe(1);
        expect(straight.distanceToBox({ min: v(0, 0, 0), max: v(3, 1, 1) })).toBe(0);
    });

    it('updates when set or marked', () => {
        const box = new OrientedBox();
        box.set(v(0, 0, 0), v(1, 1, 1));
        expect(box.needsUpdate).toBe(true);
        expect(box.intersectsBox({ min: v(0.5, 0.5, 0.5), max: v(2, 2, 2) })).toBe(true);
        expect(box.needsUpdate).toBe(false);
        box.min.set(5, 5, 5);
        box.max.set(6, 6, 6);
        box.needsUpdate = true;
        expect(box.intersectsBox({ min: v(0.5, 0.5, 0.5), max: v(2, 2, 2) })).toBe(false);
    });
});

describe('SeparatingAxisBounds', () => {
    it('projects points and tells separated intervals, touching counting as overlap', () => {
        const a = new SeparatingAxisBounds();
        const b = new SeparatingAxisBounds();
        a.setFromPoints(v(1, 0, 0), [v(0, 5, 5), v(1, 0, 0)]);
        b.setFromPoints(v(1, 0, 0), [v(1, 0, 0), v(2, 0, 0)]);
        expect([a.min, a.max, b.min, b.max]).toEqual([0, 1, 1, 2]);
        expect(a.isSeparated(b)).toBe(false);
        b.setFromPointsField([v(1.5, 0, 0), v(2, 0, 0)], 'x');
        expect(a.isSeparated(b)).toBe(true);
        b.setFromBox(v(0, 1, 0), { min: v(0, -3, 0), max: v(1, 4, 1) });
        expect([b.min, b.max]).toEqual([-3, 4]);
    });
});

describe('boxArraysIntersect and NaN', () => {
    it('treats a box with a NaN bound as overlapping nothing, as three does', async () => {
        const { boxArraysIntersect } = await import('./shapes.ts');
        const unit = [0, 0, 0, 1, 1, 1];
        expect(boxArraysIntersect(unit, [0.5, 0.5, 0.5, 2, 2, 2])).toBe(true);
        expect(boxArraysIntersect(unit, [1, 1, 1, 2, 2, 2])).toBe(true); // touching counts
        expect(boxArraysIntersect(unit, [NaN, 0, 0, 1, 1, 1])).toBe(false);
        expect(boxArraysIntersect([0, 0, 0, NaN, 1, 1], unit)).toBe(false);
    });
});

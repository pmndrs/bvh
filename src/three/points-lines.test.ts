/**
 * `acceleratedRaycast` on `Points`, `LineSegments`, `LineLoop` and `Line`
 * (PARITY 3.2, 3.3) against three's own `Points.raycast` / `Line.raycast`,
 * field for field and in three's order (random clouds and lines, indexed or
 * not, rotated, scaled, mirrored and stretched objects, draw ranges from odd
 * and even entries, thresholds including 0, near and far, firstHitOnly), and
 * against three-mesh-bvh 0.9.18's `PointsBVH`, `LineSegmentsBVH`,
 * `LineLoopBVH` and `LineBVH` on the same data where they agree with three
 * (indirect mode, the whole geometry, no stretch).
 */

import { BufferAttribute, BufferGeometry, Euler, Line, Line3, LineLoop, LineSegments, Points, Raycaster, Vector3, type Intersection, type Object3D } from 'three';
import { LineBVH as TmbLineBVH, LineLoopBVH as TmbLineLoopBVH, LineSegmentsBVH as TmbLineSegmentsBVH, PointsBVH as TmbPointsBVH } from 'three-mesh-bvh';
import { describe, expect, it, vi } from 'vitest';
import { createRandom, type Random } from '../trace/oracle-fixtures.ts';
import { validateBounds } from '../query/debug.ts';
import { BVHHelper } from './helper.ts';
import { BvhLineBoundsTree, BvhLineLoopBoundsTree, BvhLineSegmentsBoundsTree, BvhPointsBoundsTree } from './points-lines.ts';
import { acceleratedRaycast, computeBoundsTree, disposeBoundsTree } from './raycast.ts';

const threePoints = Points.prototype.raycast;
const threeLine = Line.prototype.raycast;

type Kind = 'points' | 'segments' | 'loop' | 'strip';
const KINDS: Kind[] = ['points', 'segments', 'loop', 'strip'];
const TREES = { points: BvhPointsBoundsTree, segments: BvhLineSegmentsBoundsTree, loop: BvhLineLoopBoundsTree, strip: BvhLineBoundsTree } as const;
const TMB_TREES = { points: TmbPointsBVH, segments: TmbLineSegmentsBVH, loop: TmbLineLoopBVH, strip: TmbLineBVH } as const;

function makeGeometry(random: Random, kind: Kind, vertices: number, indexed: boolean): BufferGeometry {
    const positions = new Float32Array(vertices * 3);
    for (let i = 0; i < vertices; i++) {
        for (let axis = 0; axis < 3; axis++) {
            // Points spread out; lines as a random walk, so their segments are short.
            positions[i * 3 + axis] = kind === 'points' || i === 0 || random() < 0.05 ? random() * 4 - 2 : positions[(i - 1) * 3 + axis] + (random() - 0.5) * 0.3;
        }
    }
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(positions, 3));
    if (indexed) geometry.setIndex(new BufferAttribute(Uint32Array.from({ length: vertices + 9 }, () => Math.floor(random() * vertices)), 1));
    return geometry;
}

function makeObject(kind: Kind, geometry: BufferGeometry): Points | Line {
    if (kind === 'points') return new Points(geometry);
    if (kind === 'segments') return new LineSegments(geometry);
    if (kind === 'loop') return new LineLoop(geometry);
    return new Line(geometry);
}

function place(random: Random, object: Object3D, stretch: boolean, mirror: boolean): void {
    object.position.set(random() * 2 - 1, random() * 2 - 1, random() * 2 - 1);
    object.quaternion.setFromEuler(new Euler(random() * 6, random() * 6, random() * 6));
    const s = 0.5 + random() * 1.5;
    if (stretch) object.scale.set(s, s * (0.3 + random()), s * (0.5 + random() * 2));
    else object.scale.setScalar(s);
    if (mirror) object.scale.x *= -1;
    object.updateMatrixWorld(true);
}

function aimRay(random: Random, raycaster: Raycaster, object: Object3D): void {
    const origin = new Vector3(random() * 10 - 5, random() * 10 - 5, random() * 10 - 5);
    const target = new Vector3(random() * 3 - 1.5, random() * 3 - 1.5, random() * 3 - 1.5).applyMatrix4(object.matrixWorld);
    raycaster.set(origin, target.sub(origin).normalize());
}

/** As `Raycaster.intersectObject` orders them: a stable sort by distance. */
function sorted(hits: Intersection[]): Intersection[] {
    return [...hits].sort((a, b) => a.distance - b.distance);
}

function threeRaycast(object: Points | Line, raycaster: Raycaster): Intersection[] {
    const hits: Intersection[] = [];
    if ((object as Points).isPoints) threePoints.call(object, raycaster, hits);
    else threeLine.call(object, raycaster, hits);
    return sorted(hits);
}

function ourRaycast(object: Points | Line, raycaster: Raycaster): Intersection[] {
    const hits: Intersection[] = [];
    acceleratedRaycast.call(object as never, raycaster, hits);
    return sorted(hits);
}

/** Compares every field (exactly) with three's. */
function expectSame(ours: Intersection[], theirs: Intersection[], label: string): void {
    expect(ours.length, label).toBe(theirs.length);
    for (let i = 0; i < ours.length; i++) {
        const a = ours[i] as Intersection & { distanceToRay?: number };
        const b = theirs[i] as Intersection & { distanceToRay?: number };
        expect({ ...a, point: a.point.toArray() }, `${label} hit ${i}`).toEqual({ ...b, point: b.point.toArray() });
    }
}

describe('acceleratedRaycast on points and lines, against three', () => {
    for (const kind of KINDS) {
        for (const indexed of [false, true]) {
            it(`${kind}${indexed ? ', indexed' : ''}: every field as three's`, () => {
                const random = createRandom(KINDS.indexOf(kind) * 2 + (indexed ? 1 : 0) + 1);
                const geometry = makeGeometry(random, kind, 1200, indexed);
                const tree = computeBoundsTree.call(geometry, { type: TREES[kind], maxLeafSize: 6 } as never) as unknown as BvhPointsBoundsTree | BvhLineSegmentsBoundsTree;
                expect(tree).toBeInstanceOf(TREES[kind]);
                expect(geometry.boundsTree).toBe(tree);
                expect(validateBounds(tree)).toBe(true);
                const object = makeObject(kind, geometry);
                const raycaster = new Raycaster();
                const length = indexed ? geometry.index!.count : 1200;
                let hits = 0;
                for (let i = 0; i < 240; i++) {
                    place(random, object, i % 3 === 1, i % 4 === 2);
                    aimRay(random, raycaster, object);
                    const threshold = [0.05, 0.2, 0.01, 0][i % 4];
                    raycaster.params.Points.threshold = threshold;
                    raycaster.params.Line.threshold = threshold;
                    raycaster.near = i % 5 === 3 ? 4 : 0;
                    raycaster.far = i % 5 === 4 ? 7 : Infinity;
                    // Draw ranges: whole, from odd and even entries, cut short, empty, past the end.
                    const ranges = [
                        [0, Infinity],
                        [1, 600],
                        [10, 501],
                        [3, length - 40],
                        [7, 1],
                        [0, 0],
                        [length - 2, 50],
                    ];
                    const [start, count] = ranges[i % ranges.length];
                    geometry.setDrawRange(start, count);
                    raycaster.firstHitOnly = false;
                    const theirs = threeRaycast(object, raycaster);
                    if (kind === 'loop' && Math.min(length, start + count) <= start) {
                        // An empty range: three's LineLoop reads entry -1 for its closing segment and reports a NaN distance; we report nothing.
                        expect(theirs.length <= 1 && theirs.every((hit) => Number.isNaN(hit.distance))).toBe(true);
                        expect(ourRaycast(object, raycaster)).toEqual([]);
                        continue;
                    }
                    expectSame(ourRaycast(object, raycaster), theirs, `ray ${i}`);
                    hits += theirs.length;
                    raycaster.firstHitOnly = true;
                    const first = ourRaycast(object, raycaster);
                    expect(first.length).toBe(theirs.length > 0 ? 1 : 0);
                    if (theirs.length > 0) {
                        expectSame(first, theirs.slice(0, 1), `first hit ${i}`);
                    }
                }
                expect(hits).toBeGreaterThan(200);
            });
        }
    }

    it('falls back to three for a changed geometry until refit, and for a tree of another kind', () => {
        const random = createRandom(31);
        const geometry = makeGeometry(random, 'points', 500, false);
        const tree = new BvhPointsBoundsTree(geometry);
        geometry.boundsTree = tree as never;
        const object = new Points(geometry);
        object.updateMatrixWorld(true);
        const raycaster = new Raycaster();
        raycaster.params.Points.threshold = 0.2;
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
        const position = geometry.getAttribute('position') as BufferAttribute;
        for (let i = 0; i < position.count; i++) position.setX(i, position.getX(i) + 0.5);
        position.needsUpdate = true;
        geometry.boundingSphere = null;
        expect(tree.current()).toBe(false);
        for (let i = 0; i < 20; i++) {
            aimRay(random, raycaster, object);
            expectSame(ourRaycast(object, raycaster), threeRaycast(object, raycaster), `stale ray ${i}`);
        }
        expect(warn).toHaveBeenCalledTimes(1);
        tree.refit();
        expect(tree.current()).toBe(true);
        expect(validateBounds(tree)).toBe(true);
        for (let i = 0; i < 20; i++) {
            aimRay(random, raycaster, object);
            expectSame(ourRaycast(object, raycaster), threeRaycast(object, raycaster), `refit ray ${i}`);
        }
        // A points tree on a Line: three's Line.raycast answers.
        const line = new Line(geometry);
        line.updateMatrixWorld(true);
        raycaster.params.Line.threshold = 0.1;
        aimRay(random, raycaster, line);
        expectSame(ourRaycast(line, raycaster), threeRaycast(line, raycaster), 'wrong kind');
        expect(warn).toHaveBeenCalledTimes(2);
        warn.mockRestore();
        // No tree: three's own.
        disposeBoundsTree.call(geometry);
        expectSame(ourRaycast(object, raycaster), threeRaycast(object, raycaster), 'no tree');
    });

    it('shapecasts with three types, finds the closest point, saves, and draws in BVHHelper', () => {
        const random = createRandom(41);
        const geometry = makeGeometry(random, 'strip', 900, false);
        const tree = new BvhLineBoundsTree(geometry);
        let lines = 0;
        tree.shapecast({
            intersectsBounds: () => true,
            intersectsLine: (line, index) => {
                expect(line).toBeInstanceOf(Line3);
                expect(index).toBeLessThan(899);
                lines++;
            },
        });
        expect(lines).toBe(899);
        const q = new Vector3(0.3, -0.2, 0.1);
        const found = tree.closestPointToPoint(q)!;
        expect(found.point).toBeInstanceOf(Vector3);
        let best = Infinity;
        const position = geometry.getAttribute('position') as BufferAttribute;
        for (let i = 0; i < 899; i++) best = Math.min(best, new Line3(new Vector3().fromBufferAttribute(position, i), new Vector3().fromBufferAttribute(position, i + 1)).closestPointToPoint(q, true, new Vector3()).distanceTo(q));
        expect(found.distance).toBeCloseTo(best, 12);
        const copy = new BvhLineBoundsTree(geometry, { layout: BvhLineBoundsTree.deserialize(tree.serialize()).layout });
        expect(copy.serialize()).toEqual(tree.serialize());
        expect(() => new BvhLineLoopBoundsTree(geometry, { layout: tree.layout })).toThrow(/strip/);
        const points = new Points(makeGeometry(random, 'points', 400, false));
        points.geometry.boundsTree = new BvhPointsBoundsTree(points.geometry) as never;
        const helper = new BVHHelper(points, 4);
        expect(helper.children.length).toBe(1);
    });
});

describe('against three-mesh-bvh 0.9.18 on the same data', () => {
    for (const kind of KINDS) {
        it(`${kind}: the same hits as its ${TMB_TREES[kind].name}`, () => {
            const random = createRandom(KINDS.indexOf(kind) + 60);
            const geometry = makeGeometry(random, kind, 1500, false);
            const ours = new TREES[kind](geometry);
            const theirs = new TMB_TREES[kind](geometry, { indirect: true });
            const object = makeObject(kind, geometry);
            const raycaster = new Raycaster();
            let hits = 0;
            for (let i = 0; i < 200; i++) {
                place(random, object, false, i % 3 === 0);
                aimRay(random, raycaster, object);
                const threshold = [0.05, 0.15, 0.01][i % 3];
                raycaster.params.Points.threshold = threshold;
                raycaster.params.Line.threshold = threshold;
                for (const firstHitOnly of [false, true]) {
                    raycaster.firstHitOnly = firstHitOnly;
                    const a = (ours as BvhPointsBoundsTree).raycastObject3D(object as Points, raycaster, []);
                    const b = (theirs as unknown as { raycastObject3D(o: Object3D, r: Raycaster, i: Intersection[]): Intersection[] }).raycastObject3D(object, raycaster, []);
                    // Exact fields, in (distance, index) order: three-mesh-bvh finds hits in tree order.
                    // On an exact tie (two segments meeting at a vertex), three-mesh-bvh's firstHitOnly keeps the one its tree
                    // reaches first, ours the lower entry (three's sorted order, checked above): only the distance is compared there.
                    const key = (hit: Intersection) => (firstHitOnly ? [hit.distance] : [hit.distance, hit.index!, (hit as { distanceToRay?: number }).distanceToRay ?? null, ...hit.point.toArray()]) as readonly (number | null)[];
                    const order = (x: readonly (number | null)[], y: readonly (number | null)[]) => (x[0] as number) - (y[0] as number) || ((x[1] ?? 0) as number) - ((y[1] ?? 0) as number);
                    expect(a.map(key).sort(order), `ray ${i}${firstHitOnly ? ' first' : ''}`).toEqual(b.map(key).sort(order));
                    if (!firstHitOnly) hits += a.length;
                }
            }
            expect(hits).toBeGreaterThan(100);
        });
    }

    it('leaves the index alone, where three-mesh-bvh outside indirect mode reorders it', () => {
        const random = createRandom(70);
        const geometry = makeGeometry(random, 'segments', 600, true);
        const before = Array.from(geometry.index!.array);
        const ours = new BvhLineSegmentsBoundsTree(geometry);
        expect(Array.from(geometry.index!.array)).toEqual(before);
        expect(ours.current()).toBe(true);
        new TmbLineSegmentsBVH(geometry); // sorts geometry.index's pairs in place
        expect(Array.from(geometry.index!.array)).not.toEqual(before);
        expect(ours.current()).toBe(true); // three-mesh-bvh doesn't bump the version: three's line indices changed meaning under both
    });
});

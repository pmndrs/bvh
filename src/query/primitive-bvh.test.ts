/**
 * Points and segments as primitives, engine-free (PARITY 3.2, 3.3): the ray
 * math is three's `Ray` bit for bit, and every query of `CpuPointsBvh` and
 * `CpuSegmentsBvh` answers what a scan of every primitive answers (raycast
 * with a threshold, closest hit, closest point, shapecast), on random,
 * clustered and coincident points, every segment mode and entry range, and
 * after a refit and a save.
 */

import { Line3, Ray, Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import { BVH_CONTAINED, BVH_INTERSECTED, BVH_NOT_INTERSECTED } from './cpu-mesh-bvh.ts';
import { CpuPointsBvh, CpuSegmentsBvh, raySegmentHit, type BvhPrimitiveRaycastHit, type BvhSegmentMode, type BvhSegmentsInput } from './primitive-bvh.ts';
import { rayDistanceSqToPoint, rayDistanceSqToSegment, segmentClosestPointToPoint } from './ray-primitives.ts';
import { createRandom, type Random } from '../trace/oracle-fixtures.ts';
import { getBVHExtremes, validateBounds } from './debug.ts';

type Vec = [number, number, number];

function randomPositions(random: Random, count: number, kind: 'uniform' | 'clustered' | 'coincident'): Float32Array {
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        for (let axis = 0; axis < 3; axis++) {
            if (kind === 'uniform') positions[i * 3 + axis] = random() * 4 - 2;
            else if (kind === 'clustered') positions[i * 3 + axis] = (Math.floor(random() * 4) - 1.5) + (random() - 0.5) * 0.05;
            else positions[i * 3 + axis] = i % 3 === 0 ? 0.25 : random() * 2 - 1;
        }
    }
    return positions;
}

function randomIndices(random: Random, entries: number, vertices: number): Uint32Array {
    return Uint32Array.from({ length: entries }, () => Math.floor(random() * vertices));
}

function randomRay(random: Random): { origin: Vec; direction: Vec } {
    const origin: Vec = [random() * 8 - 4, random() * 8 - 4, random() * 8 - 4];
    const target: Vec = [random() * 3 - 1.5, random() * 3 - 1.5, random() * 3 - 1.5];
    const d = new Vector3(target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]).normalize();
    return { origin, direction: [d.x, d.y, d.z] };
}

function vertexOf(indices: Uint32Array | null, entry: number): number {
    return indices ? indices[entry] : entry;
}

/** Every segment of an input as three's Line.raycast steps: [entry, a, b]. */
function segmentsOf(input: BvhSegmentsInput): [number, number, number][] {
    const indices = (input.indices as Uint32Array | null | undefined) ?? null;
    const length = indices ? indices.length : input.positions.length / 3;
    const start = input.start ?? 0;
    const end = Math.min(length, start + (input.count ?? Infinity));
    const mode = input.mode ?? 'segments';
    const out: [number, number, number][] = [];
    const step = mode === 'segments' ? 2 : 1;
    for (let i = start; i < end - 1; i += step) out.push([i, vertexOf(indices, i), vertexOf(indices, i + 1)]);
    if (mode === 'loop' && end - start >= 1) out.push([end - 1, vertexOf(indices, end - 1), vertexOf(indices, start)]);
    return out;
}

function point(positions: Float32Array, vertex: number): Vec {
    return [positions[vertex * 3], positions[vertex * 3 + 1], positions[vertex * 3 + 2]];
}

/** three's Points.raycast test, every entry, in the tree's space. */
function scanPoints(positions: Float32Array, indices: Uint32Array | null, start: number, end: number, origin: Vec, direction: Vec, threshold: number): BvhPrimitiveRaycastHit[] {
    const ray = new Ray(new Vector3(...origin), new Vector3(...direction));
    const hits: BvhPrimitiveRaycastHit[] = [];
    for (let entry = start; entry < end; entry++) {
        const p = new Vector3(...point(positions, vertexOf(indices, entry)));
        const distanceSq = ray.distanceSqToPoint(p);
        if (!(distanceSq < threshold * threshold)) continue;
        const onRay = ray.closestPointToPoint(p, new Vector3());
        hits.push({ index: entry, distance: ray.origin.distanceTo(onRay), distanceToRay: Math.sqrt(distanceSq), pointOnRay: [onRay.x, onRay.y, onRay.z], pointOnPrimitive: [p.x, p.y, p.z] });
    }
    return hits.sort((a, b) => a.distance - b.distance || a.index - b.index);
}

/** three's Line.raycast test, every segment. */
function scanSegments(input: BvhSegmentsInput, origin: Vec, direction: Vec, threshold: number): BvhPrimitiveRaycastHit[] {
    const ray = new Ray(new Vector3(...origin), new Vector3(...direction));
    const hits: BvhPrimitiveRaycastHit[] = [];
    for (const [entry, a, b] of segmentsOf(input)) {
        const onRay = new Vector3();
        const onSegment = new Vector3();
        const distanceSq = ray.distanceSqToSegment(new Vector3(...point(input.positions, a)), new Vector3(...point(input.positions, b)), onRay, onSegment);
        if (distanceSq > threshold * threshold) continue;
        hits.push({ index: entry, distance: ray.origin.distanceTo(onRay), distanceToRay: Math.sqrt(distanceSq), pointOnRay: [onRay.x, onRay.y, onRay.z], pointOnPrimitive: [onSegment.x, onSegment.y, onSegment.z] });
    }
    return hits.sort((a, b) => a.distance - b.distance || a.index - b.index);
}

describe('ray math, against three', () => {
    it('matches Ray.distanceSqToPoint, closestPointToPoint, distanceSqToSegment and Line3.closestPointToPoint bit for bit', () => {
        const random = createRandom(11);
        const out = new Float64Array(6);
        for (let i = 0; i < 20000; i++) {
            const { origin, direction } = randomRay(random);
            const ray = new Ray(new Vector3(...origin), new Vector3(...direction));
            const p = new Vector3(random() * 4 - 2, random() * 4 - 2, random() * 4 - 2);
            const q = i % 7 === 0 ? p.clone() : i % 11 === 0 ? p.clone().addScaledVector(ray.direction, 0.5) : new Vector3(random() * 4 - 2, random() * 4 - 2, random() * 4 - 2);
            expect(rayDistanceSqToPoint(...origin, ...direction, p.x, p.y, p.z, out, 0)).toBe(ray.distanceSqToPoint(p));
            const onRay = ray.closestPointToPoint(p, new Vector3());
            expect([out[0], out[1], out[2]]).toEqual([onRay.x, onRay.y, onRay.z]);
            const a = new Vector3();
            const b = new Vector3();
            expect(rayDistanceSqToSegment(...origin, ...direction, p.x, p.y, p.z, q.x, q.y, q.z, out, 0)).toBe(ray.distanceSqToSegment(p, q, a, b));
            expect(Array.from(out)).toEqual([a.x, a.y, a.z, b.x, b.y, b.z]);
            if (!p.equals(q)) {
                const line = new Line3(p, q);
                const c = line.closestPointToPoint(new Vector3(...origin), true, new Vector3());
                segmentClosestPointToPoint(p.x, p.y, p.z, q.x, q.y, q.z, ...origin, out, 0);
                expect([out[0], out[1], out[2]]).toEqual([c.x, c.y, c.z]);
            }
        }
    });
});

describe('CpuPointsBvh', () => {
    const cases = [
        { name: 'uniform', count: 3000, kind: 'uniform' as const, indexed: false, leaf: 8 },
        { name: 'clustered, indexed', count: 2500, kind: 'clustered' as const, indexed: true, leaf: 4 },
        { name: 'coincident, leaf 1', count: 1200, kind: 'coincident' as const, indexed: false, leaf: 1 },
        { name: 'one point', count: 1, kind: 'uniform' as const, indexed: false, leaf: 8 },
        { name: 'five points', count: 5, kind: 'uniform' as const, indexed: true, leaf: 8 },
    ];
    for (const fixture of cases) {
        it(`raycasts as three, ${fixture.name}`, () => {
            const random = createRandom(fixture.count + fixture.leaf);
            const positions = randomPositions(random, fixture.count, fixture.kind);
            const indices = fixture.indexed ? randomIndices(random, fixture.count + 7, fixture.count) : null;
            const length = indices ? indices.length : fixture.count;
            const start = fixture.count > 10 ? 13 : 0;
            const end = fixture.count > 10 ? length - 9 : length;
            const tree = CpuPointsBvh.build({ positions, indices, start, count: end - start }, { maxLeafPrimitives: fixture.leaf });
            expect(tree.primitiveCount).toBe(end - start);
            expect(validateBounds(tree)).toBe(true);
            for (let i = 0; i < 300; i++) {
                const { origin, direction } = randomRay(random);
                const threshold = [0.02, 0.1, 0.5, 0][i % 4];
                const expected = scanPoints(positions, indices, start, end, origin, direction, threshold);
                expect(tree.raycast({ origin, direction }, { threshold })).toEqual(expected);
                expect(tree.raycastFirst({ origin, direction }, { threshold })).toEqual(expected[0] ?? null);
                // near / far cut by distance, as three does.
                const near = 1;
                const far = 6;
                expect(tree.raycast({ origin, direction }, { threshold, near, far })).toEqual(expected.filter((hit) => hit.distance >= near && hit.distance <= far));
            }
        });
    }

    it('takes an accept test, and raycastFirst gives the closest that passes', () => {
        const random = createRandom(5);
        const positions = randomPositions(random, 2000, 'uniform');
        const tree = CpuPointsBvh.build({ positions });
        for (let i = 0; i < 100; i++) {
            const { origin, direction } = randomRay(random);
            const all = tree.raycast({ origin, direction }, { threshold: 0.2 });
            const odd = (hit: BvhPrimitiveRaycastHit) => hit.index % 2 === 1;
            expect(tree.raycast({ origin, direction }, { threshold: 0.2, accept: odd })).toEqual(all.filter(odd));
            expect(tree.raycastFirst({ origin, direction }, { threshold: 0.2, accept: odd })).toEqual(all.find(odd) ?? null);
        }
    });

    it('normalizes a direction that is not unit length', () => {
        const random = createRandom(6);
        const tree = CpuPointsBvh.build({ positions: randomPositions(random, 500, 'uniform') });
        const { origin, direction } = randomRay(random);
        const long: Vec = [direction[0] * 3, direction[1] * 3, direction[2] * 3];
        const a = tree.raycast({ origin, direction }, { threshold: 0.3 }).map((hit) => hit.index);
        expect(tree.raycast({ origin, direction: long }, { threshold: 0.3 }).map((hit) => hit.index)).toEqual(a);
    });

    it('finds the closest point, shapecasts, refits and saves', () => {
        const random = createRandom(9);
        const positions = randomPositions(random, 4000, 'clustered');
        const tree = CpuPointsBvh.build({ positions }, { maxLeafPrimitives: 6 });
        for (let i = 0; i < 200; i++) {
            const q: Vec = [random() * 6 - 3, random() * 6 - 3, random() * 6 - 3];
            let bestSq = Infinity;
            let bestIndex = -1;
            for (let entry = 0; entry < 4000; entry++) {
                const p = point(positions, entry);
                const distanceSq = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2;
                if (distanceSq < bestSq) {
                    bestSq = distanceSq;
                    bestIndex = entry;
                }
            }
            const found = tree.closestPointToPoint(q)!;
            expect(found.index).toBe(bestIndex);
            expect(found.distance).toBe(Math.sqrt(bestSq));
            expect(tree.closestPointToPoint(q, {}, 0, Math.sqrt(bestSq) * 0.999)).toBeNull();
        }
        // A box shapecast finds exactly the points in the box, each once, with contained boxes.
        const box = { min: [-1, -1, -0.5], max: [0.6, 1.2, 1] };
        const inside = (p: { x: number; y: number; z: number }) => p.x >= box.min[0] && p.x <= box.max[0] && p.y >= box.min[1] && p.y <= box.max[1] && p.z >= box.min[2] && p.z <= box.max[2];
        const seen: number[] = [];
        let contained = 0;
        tree.shapecast({
            intersectsBounds: (b) => {
                if (b.min.x >= box.min[0] && b.max.x <= box.max[0] && b.min.y >= box.min[1] && b.max.y <= box.max[1] && b.min.z >= box.min[2] && b.max.z <= box.max[2]) return BVH_CONTAINED;
                return b.max.x < box.min[0] || b.min.x > box.max[0] || b.max.y < box.min[1] || b.min.y > box.max[1] || b.max.z < box.min[2] || b.min.z > box.max[2] ? BVH_NOT_INTERSECTED : BVH_INTERSECTED;
            },
            intersectsPoint: (p, index, isContained) => {
                if (isContained) contained++;
                if (isContained || inside(p)) seen.push(index);
            },
        });
        const expected = Array.from({ length: 4000 }, (_, entry) => entry).filter((entry) => {
            const p = point(positions, entry);
            return inside({ x: p[0], y: p[1], z: p[2] });
        });
        expect(seen.sort((a, b) => a - b)).toEqual(expected);
        expect(contained).toBeGreaterThan(0);
        // Move every point, refit: the tree answers for the new positions.
        const moved = positions.map((value, i) => value * 1.5 + (i % 3) * 0.25);
        tree.refit({ positions: moved });
        expect(validateBounds(tree)).toBe(true);
        for (let i = 0; i < 50; i++) {
            const { origin, direction } = randomRay(random);
            expect(tree.raycast({ origin, direction }, { threshold: 0.1 })).toEqual(scanPoints(moved, null, 0, 4000, origin, direction, 0.1));
        }
        // Saved and read back: the same tree.
        const copy = CpuPointsBvh.deserialize(tree.serialize());
        const { origin, direction } = randomRay(random);
        expect(copy.raycast({ origin, direction }, { threshold: 0.3 })).toEqual(tree.raycast({ origin, direction }, { threshold: 0.3 }));
        expect(getBVHExtremes(copy)[0].primitives.max).toBeLessThanOrEqual(6);
        expect(() => CpuSegmentsBvh.deserialize(tree.serialize())).toThrow(/holds points/);
        const broken = new Uint8Array(tree.serialize());
        broken[64 + 3 * 4] = 200; // a child index past the nodes
        expect(() => CpuPointsBvh.deserialize(broken)).toThrow(RangeError);
    });

    it('is empty without entries, and refuses bad input', () => {
        const empty = CpuPointsBvh.build({ positions: new Float32Array(9), start: 3 });
        expect(empty.primitiveCount).toBe(0);
        expect(empty.raycast({ origin: [0, 0, 0], direction: [1, 0, 0] })).toEqual([]);
        expect(empty.closestPointToPoint([0, 0, 0])).toBeNull();
        expect(() => CpuPointsBvh.build({ positions: new Float32Array(9), indices: Uint32Array.of(0, 5) })).toThrow(/vertex 5/);
        expect(() => CpuPointsBvh.build({ positions: Float32Array.of(0, NaN, 0) })).toThrow(/non-finite/);
    });
});

describe('CpuSegmentsBvh', () => {
    const modes: BvhSegmentMode[] = ['segments', 'strip', 'loop'];
    for (const mode of modes) {
        for (const ranged of [false, true]) {
            for (const indexed of [false, true]) {
                it(`raycasts as three, ${mode}${ranged ? ', in a range' : ''}${indexed ? ', indexed' : ''}`, () => {
                    const random = createRandom(modes.indexOf(mode) * 4 + (ranged ? 2 : 0) + (indexed ? 1 : 0) + 100);
                    const vertices = 1500;
                    const positions = randomPositions(random, vertices, 'uniform');
                    // Short segments: each vertex near the previous one, as a line strip draws them.
                    for (let i = 1; i < vertices; i++) for (let axis = 0; axis < 3; axis++) if (random() < 0.9) positions[i * 3 + axis] = positions[(i - 1) * 3 + axis] + (random() - 0.5) * 0.2;
                    const indices = indexed ? randomIndices(random, vertices + 11, vertices) : null;
                    const input: BvhSegmentsInput = ranged ? { positions, indices, mode, start: 7, count: 901 } : { positions, indices, mode };
                    const tree = CpuSegmentsBvh.build(input, { maxLeafPrimitives: 5 });
                    expect(tree.primitiveCount).toBe(segmentsOf(input).length);
                    expect(validateBounds(tree)).toBe(true);
                    for (let i = 0; i < 200; i++) {
                        const { origin, direction } = randomRay(random);
                        const threshold = [0.01, 0.05, 0.3, 0][i % 4];
                        const expected = scanSegments(input, origin, direction, threshold);
                        expect(tree.raycast({ origin, direction }, { threshold })).toEqual(expected);
                        expect(tree.raycastFirst({ origin, direction }, { threshold })).toEqual(expected[0] ?? null);
                    }
                    // Closest point against a scan of three's Line3.closestPointToPoint.
                    for (let i = 0; i < 50; i++) {
                        const q = new Vector3(random() * 6 - 3, random() * 6 - 3, random() * 6 - 3);
                        let best = Infinity;
                        for (const [, a, b] of segmentsOf(input)) {
                            const pa = new Vector3(...point(positions, a));
                            const pb = new Vector3(...point(positions, b));
                            const c = pa.equals(pb) ? pa : new Line3(pa, pb).closestPointToPoint(q, true, new Vector3());
                            best = Math.min(best, c.distanceTo(q));
                        }
                        expect(tree.closestPointToPoint(q)!.distance).toBeCloseTo(best, 12);
                    }
                    // Every segment once through the shapecast, with its entry and endpoints.
                    const visited = new Map<number, string>();
                    tree.shapecast({ intersectsBounds: () => BVH_INTERSECTED, intersectsLine: (line, index) => void visited.set(index, `${line.start.x},${line.end.x}`) });
                    expect(visited.size).toBe(tree.primitiveCount);
                    for (const [entry, a, b] of segmentsOf(input)) expect(visited.get(entry)).toBe(`${point(positions, a)[0]},${point(positions, b)[0]}`);
                });
            }
        }
    }

    it('makes a loop of one entry a closing segment of length 0, and raySegmentHit tests one segment', () => {
        const positions = Float32Array.of(0, 0, 0, 1, 0, 0, 2, 0, 0);
        const loop = CpuSegmentsBvh.build({ positions, mode: 'loop', start: 1, count: 1 });
        expect(loop.primitiveCount).toBe(1);
        const hit = loop.raycastFirst({ origin: [1, 1, 0], direction: [0, -1, 0] }, { threshold: 0.1 })!;
        expect(hit.index).toBe(1);
        expect(hit.distance).toBe(1);
        expect(raySegmentHit([1, 1, 0], [0, -1, 0], [0, 0, 0], [2, 0, 0], 0, 4)).toEqual({ index: 4, distance: 1, distanceToRay: 0, pointOnRay: [1, 0, 0], pointOnPrimitive: [1, 0, 0] });
        expect(raySegmentHit([1, 1, 0], [1, 0, 0], [0, 0, 0], [2, 0, 0], 0.5, 4)).toBeNull();
    });

    it('refits and saves', () => {
        const random = createRandom(77);
        const positions = randomPositions(random, 800, 'uniform');
        const tree = CpuSegmentsBvh.build({ positions, mode: 'strip' });
        const moved = positions.map((value) => value * -2 + 0.5);
        tree.refit({ positions: moved, mode: 'strip' });
        expect(validateBounds(tree)).toBe(true);
        const copy = CpuSegmentsBvh.deserialize(new Uint8Array(tree.serialize()));
        expect(copy.mode).toBe('strip');
        for (let i = 0; i < 30; i++) {
            const { origin, direction } = randomRay(random);
            expect(copy.raycast({ origin, direction }, { threshold: 0.05 })).toEqual(scanSegments({ positions: moved, mode: 'strip' }, origin, direction, 0.05));
        }
    });
});

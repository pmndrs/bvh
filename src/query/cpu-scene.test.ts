/**
 * CpuSceneBvh against brute force: every hit of a ray through a scene of
 * instanced meshes (each triangle of each instance tested on its own), the
 * closest hit, masks, and frustum and box culling (every instance's world
 * box tested on its own), before and after the instances move.
 */

import { describe, expect, it } from 'vitest';
import { createTinyBvhBuilder } from '../tinybvh.ts';
import { buildMeshBvhWithTinyBvh } from '../trace/mesh-bvh-topology.ts';
import type { MeshGeometry } from '../trace/mesh-bvh.ts';
import { createRandom, randomAffine, randomTriangleSoup, tessellatedQuad } from '../trace/oracle-fixtures.ts';
import type { TraceRay } from '../trace/rays.ts';
import type { TraceSceneInstance } from '../trace/scene-assembly.ts';
import { transformBounds } from '../trace/top-level.ts';
import { CpuSceneBvh } from './cpu-scene.ts';

const tinyBvh = await createTinyBvhBuilder();

function corner(geometry: MeshGeometry, triangle: number, k: number): number[] {
    const vertex = geometry.indices ? geometry.indices[triangle * 3 + k] : triangle * 3 + k;
    return [geometry.positions[vertex * 3], geometry.positions[vertex * 3 + 1], geometry.positions[vertex * 3 + 2]];
}

function transformPoint(m: ArrayLike<number>, p: number[]): number[] {
    return [0, 1, 2].map((r) => m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r]);
}

/** Möller-Trumbore in doubles on the world-space corners: every hit's (instance, triangle, t). */
function bruteHits(geometries: MeshGeometry[], instances: TraceSceneInstance[], ray: TraceRay, mask: number): { instance: number; triangle: number; t: number; margin: number }[] {
    const hits: { instance: number; triangle: number; t: number; margin: number }[] = [];
    for (const instance of instances) {
        if (instance.visible === false || ((instance.mask ?? 0xff) & mask) === 0) continue;
        const geometry = geometries[instance.meshIndex];
        const count = (geometry.indices ? geometry.indices.length : geometry.positions.length / 3) / 3;
        for (let triangle = 0; triangle < count; triangle++) {
            const [a, b, c] = [0, 1, 2].map((k) => transformPoint(instance.objectToWorld, corner(geometry, triangle, k)));
            const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            const e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            const d = ray.direction;
            const p = [d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]];
            const det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
            if (Math.abs(det) < 1e-12) continue;
            const s = [ray.origin[0] - a[0], ray.origin[1] - a[1], ray.origin[2] - a[2]];
            const u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) / det;
            const q = [s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0]];
            const v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) / det;
            const t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) / det;
            if (u < 0 || v < 0 || u + v > 1 || t < ray.tMin || t > ray.tMax) continue;
            hits.push({ instance: instance.instanceIndex, triangle, t, margin: Math.min(u, v, 1 - u - v) });
        }
    }
    return hits.sort((x, y) => x.t - y.t);
}

function makeScene(seed: number) {
    const random = createRandom(seed);
    const geometries = [randomTriangleSoup(random, 60, 1.5, 0.8), tessellatedQuad(4, 1), randomTriangleSoup(random, 25, 0.8, 0.5)];
    const instances: TraceSceneInstance[] = Array.from({ length: 40 }, (_, i) => ({
        meshIndex: i % 3,
        instanceIndex: i,
        objectToWorld: randomAffine(random, { extent: 12, mirror: true }),
        mask: i % 5 === 0 ? 0x02 : 0x01,
        visible: i % 11 !== 7,
    }));
    return { random, geometries, instances };
}

function randomRay(random: () => number): TraceRay {
    const origin = [0, 1, 2].map(() => (random() - 0.5) * 30);
    const target = [0, 1, 2].map(() => (random() - 0.5) * 12);
    const direction = target.map((value, k) => value - origin[k]);
    const length = Math.hypot(...direction);
    return { origin: origin as [number, number, number], direction: direction.map((value) => value / length) as [number, number, number], tMin: 0, tMax: 1e6 };
}

function randomPlanes(random: () => number): number[] {
    // A random box-ish region as six planes (inward normals): a frustum's shape.
    const planes: number[] = [];
    const center = [0, 1, 2].map(() => (random() - 0.5) * 16);
    for (let p = 0; p < 6; p++) {
        const normal = [0, 1, 2].map(() => random() - 0.5);
        const length = Math.hypot(...normal);
        const n = normal.map((value) => value / length);
        const distance = 1 + random() * 8;
        // Inside when n . (x - center) + distance >= 0.
        planes.push(n[0], n[1], n[2], distance - (n[0] * center[0] + n[1] * center[1] + n[2] * center[2]));
    }
    return planes;
}

describe('CpuSceneBvh', () => {
    it('returns every hit and the closest one, as brute force over each instance, with masks; and again after setInstances', () => {
        const { random, geometries, instances } = makeScene(5);
        const scene = new CpuSceneBvh(geometries.map((geometry) => buildMeshBvhWithTinyBvh(geometry, tinyBvh)), instances);
        let current = instances;
        let total = 0;
        const problems: string[] = [];
        for (let round = 0; round < 2; round++) {
            for (let r = 0; r < 1500; r++) {
                const ray = randomRay(random);
                const mask = r % 3 === 0 ? 0x02 : 0xff;
                const expected = bruteHits(geometries, current, ray, mask);
                const ours = scene.raycast(ray, { mask });
                total += expected.length;
                const key = (hit: { instance: number; triangle: number }) => `${hit.instance}:${hit.triangle}`;
                const ourKeys = new Set(ours.map(key));
                const expectedKeys = new Set(expected.map(key));
                for (const hit of expected) if (!ourKeys.has(key(hit)) && hit.margin > 1e-6) problems.push(`round ${round} ray ${r}: missed ${key(hit)}`);
                for (const hit of ours) if (!expectedKeys.has(key(hit))) {
                    const near = expected.some((other) => Math.abs(other.t - hit.t) < 1e-5);
                    if (!near) problems.push(`round ${round} ray ${r}: extra ${key(hit)}`);
                }
                for (let i = 1; i < ours.length; i++) if (ours[i].t < ours[i - 1].t) problems.push(`ray ${r}: not sorted`);
                const first = scene.raycast(ray, { mask, firstHitOnly: true });
                if (expected.length > 0 !== first.length > 0 || (first.length > 0 && Math.abs(first[0].t - expected[0].t) > 1e-5 * Math.max(1, expected[0].t))) {
                    problems.push(`round ${round} ray ${r}: closest ${first[0]?.t} vs ${expected[0]?.t}`);
                }
            }
            // Move every instance, show the hidden ones: the top level only.
            current = current.map((instance) => ({ ...instance, objectToWorld: randomAffine(random, { extent: 12, mirror: true }), visible: true }));
            scene.setInstances(current);
        }
        expect(problems.slice(0, 10)).toEqual([]);
        expect(total).toBeGreaterThan(1000);
        // accept: a filter that keeps front faces only.
        const ray = randomRay(random);
        const front = scene.raycast(ray, { accept: (_instance, _triangle, frontFace) => frontFace });
        expect(front.every((hit) => hit.frontFace)).toBe(true);
    });

    it('culls by frustum and box over the top level as brute force over every world box, with masks', () => {
        const { random, geometries, instances } = makeScene(9);
        const trees = geometries.map((geometry) => buildMeshBvhWithTinyBvh(geometry, tinyBvh));
        const scene = new CpuSceneBvh(trees, instances);
        const box = new Float32Array(6);
        const worldBox = (instance: TraceSceneInstance) => {
            transformBounds(trees[instance.meshIndex].bounds, instance.objectToWorld, box, 0);
            return Array.from(box);
        };
        let found = 0;
        for (let round = 0; round < 300; round++) {
            const planes = randomPlanes(random);
            const mask = round % 4 === 0 ? 0x02 : 0xff;
            const expected = instances
                .filter((instance) => instance.visible !== false && ((instance.mask ?? 0xff) & mask) !== 0)
                .filter((instance) => {
                    const b = worldBox(instance);
                    for (let p = 0; p < 6; p++) {
                        const [nx, ny, nz, d] = planes.slice(p * 4, p * 4 + 4);
                        const far = nx * (nx > 0 ? b[3] : b[0]) + ny * (ny > 0 ? b[4] : b[1]) + nz * (nz > 0 ? b[5] : b[2]) + d;
                        if (far < 0) return false;
                    }
                    return true;
                })
                .map((instance) => instance.instanceIndex)
                .sort((a, b) => a - b);
            expect(scene.intersectsFrustum(planes, { mask }).sort((a, b) => a - b)).toEqual(expected);
            found += expected.length;
            const query = [0, 1, 2].map(() => (random() - 0.5) * 20);
            const size = random() * 8;
            const region = [...query, ...query.map((value) => value + size)];
            const expectedBox = instances
                .filter((instance) => instance.visible !== false)
                .filter((instance) => {
                    const b = worldBox(instance);
                    return [0, 1, 2].every((k) => b[k + 3] >= region[k] && b[k] <= region[k + 3]);
                })
                .map((instance) => instance.instanceIndex)
                .sort((a, b) => a - b);
            expect(scene.intersectsBox(region).sort((a, b) => a - b)).toEqual(expectedBox);
        }
        expect(found).toBeGreaterThan(300);
    });
});

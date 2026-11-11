/**
 * `geometry.boundsTree` as three-mesh-bvh's `MeshBVH`: the options its
 * callers pass (homefig, drei, three-gpu-pathtracer, upscaler, three-bvh-csg),
 * `raycast` / `raycastFirst` on a three `Ray` against three's own
 * `Mesh.raycast`, `refit`, saving, `getTriangleHitPointInfo` against
 * three-mesh-bvh's, and `GenerateMeshBvhWorker` in Node (the pool's inline
 * fallback).
 */

import { BackSide, BoxGeometry, type BufferGeometry, DoubleSide, FrontSide, type Intersection, Mesh, MeshBasicMaterial, Ray, Raycaster, SphereGeometry, TorusKnotGeometry, Vector3 } from 'three';
import { getTriangleHitPointInfo as tmbGetTriangleHitPointInfo } from 'three-mesh-bvh';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { GPU_LAYOUT_NODE_WORD, GPU_LAYOUT_NODE_WORDS } from '../gpu-layout.ts';
import { CpuMeshBvh } from '../query/cpu-mesh-bvh.ts';
import { createRandom } from '../trace/oracle-fixtures.ts';
import { GenerateMeshBvhWorker } from './generate.ts';
import {
    acceleratedRaycast,
    AVERAGE,
    BvhBoundsTree,
    CENTER,
    computeBoundsTree,
    deserializeBoundsTree,
    getTriangleHitPointInfo,
    prepareBoundsTrees,
    SAH,
    type BvhBoundsTreeOptions,
} from './raycast.ts';

const threeRaycast = Mesh.prototype.raycast;
beforeAll(async () => {
    await prepareBoundsTrees();
});
afterEach(() => {
    Mesh.prototype.raycast = threeRaycast;
    vi.restoreAllMocks();
});

/** The largest leaf in a tree. */
function largestLeaf(tree: CpuMeshBvh<unknown>): number {
    const words = new Uint32Array(tree.layout.nodes);
    let largest = 0;
    for (let node = 0; node < tree.nodeCount; node++) {
        largest = Math.max(largest, words[node * GPU_LAYOUT_NODE_WORDS + GPU_LAYOUT_NODE_WORD.leftCount], words[node * GPU_LAYOUT_NODE_WORDS + GPU_LAYOUT_NODE_WORD.rightCount]);
    }
    return largest;
}

/** three's own `Mesh.raycast` on an untransformed mesh: the hits a local-space `boundsTree.raycast` must give. */
function threeHits(geometry: BufferGeometry, side: number, ray: Ray, near = 0, far = Infinity): Intersection[] {
    const mesh = new Mesh(geometry, new MeshBasicMaterial({ side: side as 0 | 1 | 2 }));
    mesh.updateMatrixWorld();
    const raycaster = new Raycaster(ray.origin, ray.direction, near, far);
    const hits: Intersection[] = [];
    threeRaycast.call(mesh, raycaster, hits);
    return hits.sort((a, b) => a.distance - b.distance);
}

function randomRay(random: () => number): Ray {
    const origin = new Vector3((random() - 0.5) * 6, (random() - 0.5) * 6, (random() - 0.5) * 6);
    const target = new Vector3((random() - 0.5) * 1.5, (random() - 0.5) * 1.5, (random() - 0.5) * 1.5);
    return new Ray(origin, target.sub(origin).normalize());
}

describe('computeBoundsTree options (three-mesh-bvh names)', () => {
    it("accepts every option the surveyed callers pass, and honours leaf size, progress and setBoundingBox", () => {
        const callers: [string, BvhBoundsTreeOptions, number][] = [
            ['homefig wall/roof', { maxLeafSize: 10 }, 8],
            ['drei useBVH', { strategy: SAH, verbose: false, setBoundingBox: true, maxDepth: 40, maxLeafTris: 10, indirect: false }, 8],
            ['pathtracer', { strategy: SAH, maxLeafTris: 1, indirect: true, onProgress: () => {} }, 1],
            ['upscaler', { strategy: SAH, targetLeafSize: 4 }, 4],
            ['csg Brush', { maxLeafSize: 3, indirect: true, useSharedArrayBuffer: false }, 3],
        ];
        for (const [name, options, leaf] of callers) {
            const geometry = new TorusKnotGeometry(1, 0.3, 64, 8);
            const tree = computeBoundsTree.call(geometry, options);
            expect(largestLeaf(tree), name).toBeLessThanOrEqual(leaf);
            if (leaf === 1) expect(largestLeaf(tree), name).toBe(1);
            expect(geometry.boundingBox !== null, name).toBe(options.setBoundingBox === true);
        }
        // targetLeafSize wins over the older names; below 1 gives 1.
        expect(largestLeaf(new BvhBoundsTree(new TorusKnotGeometry(1, 0.3, 64, 8), { targetLeafSize: 2, maxLeafSize: 8 }))).toBeLessThanOrEqual(2);
        expect(largestLeaf(new BvhBoundsTree(new TorusKnotGeometry(1, 0.3, 64, 8), { maxLeafSize: 0 }))).toBe(1);
        // Every strategy builds the same tree (tinybvh's SAH builder).
        const bytes = [CENTER, AVERAGE, SAH].map((strategy) => new Uint8Array(new BvhBoundsTree(new SphereGeometry(1, 16, 12), { strategy }).serialize()));
        expect(bytes[0]).toEqual(bytes[1]);
        expect(bytes[0]).toEqual(bytes[2]);
        expect(() => new BvhBoundsTree(new SphereGeometry(), { strategy: 7 })).toThrow(RangeError);
        expect(() => new BvhBoundsTree(new SphereGeometry(), { range: { start: 0, count: 30 } })).toThrow(/range/);
        // onProgress: 0, then 1. setBoundingBox: the tree's box, which three would compute too.
        const progress: number[] = [];
        const geometry = new SphereGeometry(1.5, 12, 8);
        new BvhBoundsTree(geometry, { onProgress: (value) => progress.push(value), setBoundingBox: true });
        expect(progress).toEqual([0, 1]);
        const box = geometry.boundingBox!.clone();
        geometry.computeBoundingBox();
        expect(box.min.distanceTo(geometry.boundingBox!.min)).toBeLessThan(1e-6);
        expect(box.max.distanceTo(geometry.boundingBox!.max)).toBeLessThan(1e-6);
    });
});

describe('BvhBoundsTree (geometry.boundsTree)', () => {
    it("is the core CpuMeshBvh, and its raycast / raycastFirst on a three Ray give three's hits", () => {
        const random = createRandom(61);
        const geometry = new TorusKnotGeometry(1, 0.3, 48, 8);
        const tree = computeBoundsTree.call(geometry);
        expect(tree).toBeInstanceOf(CpuMeshBvh);
        let compared = 0;
        for (let r = 0; r < 600; r++) {
            const ray = randomRay(random);
            const side = [FrontSide, BackSide, DoubleSide][r % 3];
            const near = r % 4 === 0 ? 1 : 0;
            const far = r % 4 === 1 ? 4 : Infinity;
            const expected = threeHits(geometry, side, ray, near, far);
            const ours = tree.raycast(ray, side, near, far);
            const material = new MeshBasicMaterial({ side });
            expect(tree.raycast(ray, material, near, far).map((hit) => hit.faceIndex)).toEqual(ours.map((hit) => hit.faceIndex));
            // Edge hits may differ (watertight test against Möller-Trumbore); compare rays with no hit near an edge.
            const nearEdge = [...expected, ...ours].some((hit) => Math.min(hit.barycoord!.x, hit.barycoord!.y, hit.barycoord!.z) < 1e-4);
            if (nearEdge) continue;
            compared++;
            expect(ours.map((hit) => hit.faceIndex)).toEqual(expected.map((hit) => hit.faceIndex));
            ours.forEach((hit, i) => {
                expect(hit.distance).toBeCloseTo(expected[i].distance, 5);
                expect(hit.point.distanceTo(expected[i].point)).toBeLessThan(1e-5);
                expect(hit.face!.normal.distanceTo(expected[i].face!.normal)).toBeLessThan(1e-5);
                expect([hit.face!.a, hit.face!.b, hit.face!.c]).toEqual([expected[i].face!.a, expected[i].face!.b, expected[i].face!.c]);
                expect(hit.uv!.distanceTo(expected[i].uv!)).toBeLessThan(1e-4);
                expect(hit.normal!.distanceTo(expected[i].normal!)).toBeLessThan(1e-4);
            });
            const first = tree.raycastFirst(ray, side, near, far);
            expect(first?.faceIndex ?? null).toBe(expected[0]?.faceIndex ?? null);
        }
        expect(compared).toBeGreaterThan(400);
        // three-bvh-csg's getHitSide: raycastFirst(ray, DoubleSide).face.normal against the ray.
        const inside = new Ray(new Vector3(0, 0, 0), new Vector3(0, 0, 1));
        const box = new BoxGeometry(2, 2, 2);
        const hit = computeBoundsTree.call(box).raycastFirst(inside, DoubleSide)!;
        expect(inside.direction.dot(hit.face!.normal)).toBeGreaterThan(0);
    });

    it('takes an array of materials, each group with its own side', () => {
        const random = createRandom(62);
        const geometry = new BoxGeometry(2, 2, 2, 2, 2, 2);
        const tree = computeBoundsTree.call(geometry);
        const materials = [0, 1, 2, 3, 4, 5].map((i) => new MeshBasicMaterial({ side: [FrontSide, BackSide, DoubleSide][i % 3] }));
        const mesh = new Mesh(geometry, materials);
        mesh.updateMatrixWorld();
        let hits = 0;
        for (let r = 0; r < 300; r++) {
            const ray = randomRay(random);
            const raycaster = new Raycaster(ray.origin, ray.direction);
            const expected: Intersection[] = [];
            threeRaycast.call(mesh, raycaster, expected);
            expected.sort((a, b) => a.distance - b.distance);
            const ours = tree.raycast(ray, materials);
            if ([...expected, ...ours].some((hit) => Math.min(hit.barycoord!.x, hit.barycoord!.y, hit.barycoord!.z) < 1e-4)) continue;
            expect(ours.map((hit) => [hit.faceIndex, hit.face!.materialIndex])).toEqual(expected.map((hit) => [hit.faceIndex, hit.face!.materialIndex]));
            hits += ours.length;
        }
        expect(hits).toBeGreaterThan(100);
    });

    it("refit() follows moved vertices, and acceleratedRaycast uses the tree again (three's hits)", () => {
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
        const random = createRandom(63);
        const geometry = new SphereGeometry(1, 16, 12);
        const tree = computeBoundsTree.call(geometry);
        const mesh = new Mesh(geometry, new MeshBasicMaterial({ side: DoubleSide }));
        mesh.updateMatrixWorld();
        const position = geometry.attributes.position;
        for (let i = 0; i < position.count; i++) position.setXYZ(i, position.getX(i) * 2, position.getY(i) * 0.5, position.getZ(i) + Math.sin(position.getX(i) * 3) * 0.3);
        position.needsUpdate = true;
        geometry.computeBoundingSphere();
        expect(tree.current()).toBe(false);
        tree.refit();
        expect(tree.current()).toBe(true);
        const local = vi.spyOn(tree, 'raycastLocal');
        Mesh.prototype.raycast = acceleratedRaycast;
        let hits = 0;
        for (let r = 0; r < 300; r++) {
            const raycaster = new Raycaster();
            const ray = randomRay(random);
            raycaster.set(ray.origin, ray.direction);
            const ours = raycaster.intersectObject(mesh);
            Mesh.prototype.raycast = threeRaycast;
            const expected = raycaster.intersectObject(mesh);
            Mesh.prototype.raycast = acceleratedRaycast;
            if ([...expected, ...ours].some((hit) => Math.min(hit.barycoord!.x, hit.barycoord!.y, hit.barycoord!.z) < 1e-4)) continue;
            expect(ours.map((hit) => hit.faceIndex)).toEqual(expected.map((hit) => hit.faceIndex));
            ours.forEach((hit, i) => expect(hit.distance).toBeCloseTo(expected[i].distance, 5));
            hits += ours.length;
        }
        expect(hits).toBeGreaterThan(100);
        expect(local).toHaveBeenCalled();
        expect(warn).not.toHaveBeenCalled();
        // A replaced index can't be refitted.
        geometry.setIndex(Array.from(geometry.index!.array));
        expect(() => tree.refit()).toThrow(/computeBoundsTree/);
    });

    it('serializes and deserializes for its geometry, and maps tree positions (indirect)', () => {
        const geometry = new TorusKnotGeometry(1, 0.3, 32, 6);
        const tree = computeBoundsTree.call(geometry);
        const loaded = deserializeBoundsTree(tree.serialize(), geometry);
        expect(loaded).toBeInstanceOf(BvhBoundsTree);
        const ray = new Ray(new Vector3(0, 0, 5), new Vector3(0.05, 0.1, -1).normalize());
        expect(loaded.raycast(ray, DoubleSide).map((hit) => hit.faceIndex)).toEqual(tree.raycast(ray, DoubleSide).map((hit) => hit.faceIndex));
        expect(() => deserializeBoundsTree(tree.serialize(), new SphereGeometry())).toThrow(RangeError);
        // The index is never reordered; positions map back to every triangle once.
        const before = Array.from(geometry.index!.array);
        const triangles = Array.from({ length: tree.triangleCount }, (_, position) => tree.resolveTriangleIndex(position)).sort((a, b) => a - b);
        expect(triangles).toEqual(Array.from({ length: tree.triangleCount }, (_, i) => i));
        expect(Array.from(geometry.index!.array)).toEqual(before);
    });

    it("getTriangleHitPointInfo gives three-mesh-bvh's face, uv and barycoord", () => {
        const random = createRandom(64);
        const geometry = new SphereGeometry(1, 12, 8);
        geometry.addGroup(0, 144, 0);
        geometry.addGroup(144, Infinity, 1);
        const tree = computeBoundsTree.call(geometry);
        let checked = 0;
        for (let r = 0; r < 100; r++) {
            const hit = tree.raycastFirst(randomRay(random), DoubleSide);
            if (!hit) continue;
            const ours = getTriangleHitPointInfo(hit.point, geometry, hit.faceIndex!);
            const theirs = tmbGetTriangleHitPointInfo(hit.point, geometry, hit.faceIndex!);
            expect([ours.face.a, ours.face.b, ours.face.c, ours.face.materialIndex]).toEqual([theirs.face.a, theirs.face.b, theirs.face.c, theirs.face.materialIndex]);
            expect(ours.face.normal.distanceTo(theirs.face.normal)).toBeLessThan(1e-6);
            expect(ours.uv!.distanceTo(theirs.uv!)).toBeLessThan(1e-5);
            expect(ours.barycoord.distanceTo(theirs.barycoord)).toBeLessThan(1e-5);
            expect(ours.barycoord.distanceTo(hit.barycoord!)).toBeLessThan(1e-4);
            checked++;
        }
        expect(checked).toBeGreaterThan(40);
    });
});

describe('GenerateMeshBvhWorker', () => {
    it('generates the same tree as computeBoundsTree in Node (inline), with progress, and several at once', async () => {
        const generator = new GenerateMeshBvhWorker();
        expect(generator.pool.usesWorkers).toBe(false);
        const geometry = new TorusKnotGeometry(1, 0.3, 64, 8);
        const progress: number[] = [];
        const pending = generator.generate(geometry, { strategy: SAH, maxLeafTris: 1, indirect: true, onProgress: (value) => progress.push(value) });
        expect(generator.running).toBe(true);
        const tree = await pending;
        expect(generator.running).toBe(false);
        expect(tree).toBeInstanceOf(BvhBoundsTree);
        expect(progress).toEqual([0, 1]);
        expect(largestLeaf(tree)).toBe(1);
        expect(new Uint8Array(tree.serialize())).toEqual(new Uint8Array(new BvhBoundsTree(geometry, { maxLeafTris: 1 }).serialize()));
        // The geometry's arrays were copied, not moved.
        expect(geometry.attributes.position.array.length).toBeGreaterThan(0);
        const many = await Promise.all([new SphereGeometry(1, 8, 6), new BoxGeometry(1, 1, 1, 3, 3, 3)].map((g) => generator.generate(g)));
        expect(many.map((t) => t.triangleCount)).toEqual([new SphereGeometry(1, 8, 6).index!.count / 3, 108]);
        await expect(generator.generate(geometry, { strategy: 9 })).rejects.toThrow(RangeError);
        generator.dispose();
    });
});

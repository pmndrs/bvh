/**
 * An indexed `BvhBoundsTree` (a layout or file with `triangleLeaves:
 * 'indexed'`) stays indexed: `CpuMeshBvh`'s `corners: 'positions'`, 16 bytes
 * per triangle, corners read from the geometry's positions, never expanded to
 * 48-byte records (docs/internals/design/indexed-leaves.md). Each case builds
 * a records tree and its indexed twin (the same tree, triangles converted,
 * over a second geometry with the same values) and checks every answer is the
 * records tree's exactly: raycast, raycastFirst, `firstHitOnly`, groups, draw
 * ranges (shifted trees), shapecast, closest point, refit, serialization.
 */

import { Box3, BoxGeometry, type BufferGeometry, DoubleSide, FrontSide, type Intersection, Matrix4, Mesh, MeshBasicMaterial, Ray, Raycaster, Sphere, SphereGeometry, TorusKnotGeometry, Vector3 } from 'three';
import { beforeAll, describe, expect, it, vi } from 'vitest';
import { GPU_LAYOUT_INDEXED_TRIANGLE_BYTES, GPU_LAYOUT_TRIANGLE_BYTES } from '../gpu-layout.ts';
import { indexedFromRecords, recordsFromIndexed } from '../indexed-leaves.ts';
import { CpuMeshBvh } from '../query/cpu-mesh-bvh.ts';
import { deserializeLayout } from '../serialize.ts';
import { createRandom } from '../trace/oracle-fixtures.ts';
import { acceleratedRaycast, BvhBoundsTree, computeBoundsTree, deserializeBoundsTree, meshInputFromGeometry, prepareBoundsTrees, setBoundsTree } from './raycast.ts';

// The expansion, watched: CpuMeshBvh calls recordsFromIndexed for an indexed tree taken as records.
vi.mock(import('../indexed-leaves.ts'), async (importOriginal) => {
    const original = await importOriginal();
    return { ...original, recordsFromIndexed: vi.fn(original.recordsFromIndexed) };
});
const expansions = vi.mocked(recordsFromIndexed);

beforeAll(async () => {
    await prepareBoundsTrees();
});

interface Twins {
    readonly recordsGeometry: BufferGeometry;
    readonly indexedGeometry: BufferGeometry;
    readonly records: BvhBoundsTree;
    readonly indexed: BvhBoundsTree;
}

/** A records tree over one geometry and its indexed twin over another with the same values (the same tree, its triangles converted). */
function twins(make: () => BufferGeometry): Twins {
    const recordsGeometry = make();
    const indexedGeometry = make();
    const records = computeBoundsTree.call(recordsGeometry);
    const layout = indexedFromRecords(deserializeLayout(records.serialize()), meshInputFromGeometry(indexedGeometry));
    expansions.mockClear();
    const indexed = new BvhBoundsTree(indexedGeometry, { layout });
    setBoundsTree(indexedGeometry, indexed);
    expect(expansions).not.toHaveBeenCalled();
    return { recordsGeometry, indexedGeometry, records, indexed };
}

const knot = () => new TorusKnotGeometry(1, 0.3, 48, 8);

function randomRay(random: () => number): Ray {
    const origin = new Vector3((random() - 0.5) * 6, (random() - 0.5) * 6, (random() - 0.5) * 6);
    const target = new Vector3((random() - 0.5) * 1.5, (random() - 0.5) * 1.5, (random() - 0.5) * 1.5);
    return new Ray(origin, target.sub(origin).normalize());
}

/** `Mesh.raycast` through `acceleratedRaycast` (all hits, or `firstHitOnly`), without `object` (each twin has its own mesh). */
function meshHits(mesh: Mesh, ray: Ray, firstHitOnly: boolean): Omit<Intersection, 'object'>[] {
    const raycaster = new Raycaster(ray.origin, ray.direction) as Raycaster & { firstHitOnly?: boolean };
    raycaster.firstHitOnly = firstHitOnly;
    return raycaster.intersectObject(mesh).map(({ object: _object, ...hit }) => hit);
}

/** Raycasts both twins every way three-mesh-bvh's callers do; every hit must be the same. Returns the hits seen. */
function compareRaycasts(pair: Twins, seed: number, materials?: MeshBasicMaterial[]): number {
    const random = createRandom(seed);
    const material = materials ?? new MeshBasicMaterial({ side: FrontSide });
    const recordsMesh = new Mesh(pair.recordsGeometry, material);
    const indexedMesh = new Mesh(pair.indexedGeometry, material);
    recordsMesh.raycast = indexedMesh.raycast = acceleratedRaycast;
    recordsMesh.updateMatrixWorld();
    indexedMesh.updateMatrixWorld();
    let hits = 0;
    for (let r = 0; r < 200; r++) {
        const ray = randomRay(random);
        const side = [FrontSide, DoubleSide][r % 2];
        const near = r % 5 === 0 ? 1 : 0;
        const far = r % 5 === 1 ? 4 : Infinity;
        const all = pair.records.raycast(ray, materials ?? side, near, far);
        expect(pair.indexed.raycast(ray, materials ?? side, near, far)).toEqual(all);
        expect(pair.indexed.raycastFirst(ray, materials ?? side, near, far)).toEqual(pair.records.raycastFirst(ray, materials ?? side, near, far));
        for (const firstHitOnly of [false, true]) expect(meshHits(indexedMesh, ray, firstHitOnly)).toEqual(meshHits(recordsMesh, ray, firstHitOnly));
        hits += all.length;
    }
    return hits;
}

/** Everything a shapecast hands its callbacks: boxes, depths, node indices, triangles (index and corners) and `contained`. */
function shapecastLog(tree: BvhBoundsTree, sphere: Sphere): number[] {
    const log: number[] = [];
    tree.shapecast({
        intersectsBounds: (box, isLeaf, score, depth, nodeIndex) => {
            log.push(...box.min.toArray(), ...box.max.toArray(), isLeaf ? 1 : 0, score ?? -1, depth, nodeIndex);
            return sphere.intersectsBox(box);
        },
        intersectsTriangle: (triangle, index, contained, depth) => {
            log.push(index, ...triangle.a.toArray(), ...triangle.b.toArray(), ...triangle.c.toArray(), contained ? 1 : 0, depth, triangle.intersectsSphere(sphere) ? 1 : 0);
            return false;
        },
    });
    return log;
}

describe('BvhBoundsTree over an indexed tree', () => {
    it('stays indexed: 16 bytes per triangle, no records expansion (which a plain CpuMeshBvh does)', () => {
        const pair = twins(knot);
        const count = pair.indexed.triangleCount;
        expect(pair.indexed.layout.triangleLeaves).toBe('indexed');
        expect(pair.indexed.layout.triangles.byteLength).toBe(count * GPU_LAYOUT_INDEXED_TRIANGLE_BYTES);
        expect(pair.records.layout.triangles.byteLength).toBe(count * GPU_LAYOUT_TRIANGLE_BYTES);
        expect(pair.records.estimateMemoryInBytes() - pair.indexed.estimateMemoryInBytes()).toBe(count * (GPU_LAYOUT_TRIANGLE_BYTES - GPU_LAYOUT_INDEXED_TRIANGLE_BYTES));
        // The spy sees the expansion when it happens: the core's default takes an indexed tree as records.
        const input = meshInputFromGeometry(pair.indexedGeometry);
        new CpuMeshBvh(indexedFromRecords(deserializeLayout(pair.records.serialize()), input), { source: input });
        expect(expansions).toHaveBeenCalledTimes(1);
        // A records tree stays records.
        expect(pair.records.layout.triangleLeaves ?? 'records').toBe('records');
    });

    it("gives the records tree's raycast and raycastFirst hits, through Mesh.raycast and firstHitOnly too, indexed and non-indexed geometry", () => {
        expect(compareRaycasts(twins(knot), 81)).toBeGreaterThan(100);
        expect(compareRaycasts(twins(() => knot().toNonIndexed()), 82)).toBeGreaterThan(100);
        // Groups with their own sides.
        const materials = [0, 1, 2, 3, 4, 5].map((i) => new MeshBasicMaterial({ side: [FrontSide, DoubleSide][i % 2] }));
        expect(compareRaycasts(twins(() => new BoxGeometry(2, 2, 2, 3, 3, 3)), 83, materials)).toBeGreaterThan(50);
        expect(expansions).not.toHaveBeenCalled();
    });

    it('builds its shifted trees (a draw range off a multiple of 3) indexed too, with the same hits', () => {
        const pair = twins(knot);
        pair.recordsGeometry.setDrawRange(7, 600);
        pair.indexedGeometry.setDrawRange(7, 600);
        expect(compareRaycasts(pair, 84)).toBeGreaterThan(20);
        const shifted = [...(pair.indexed as unknown as { shifted: Map<string, { tree: CpuMeshBvh }> }).shifted.values()];
        expect(shifted.length).toBeGreaterThan(0);
        for (const { tree } of shifted) expect(tree.layout.triangleLeaves).toBe('indexed');
        expect(expansions).not.toHaveBeenCalled();
    });

    it('gives the same shapecast callbacks, sphere and box tests, bvhcast and geometry tests', () => {
        const pair = twins(knot);
        const random = createRandom(85);
        for (let i = 0; i < 20; i++) {
            const sphere = new Sphere(new Vector3((random() - 0.5) * 3, (random() - 0.5) * 3, (random() - 0.5) * 3), 0.1 + random() * 0.6);
            expect(shapecastLog(pair.indexed, sphere)).toEqual(shapecastLog(pair.records, sphere));
            expect(pair.indexed.intersectsSphere(sphere)).toBe(pair.records.intersectsSphere(sphere));
            const box = sphere.getBoundingBox(new Box3());
            const turn = new Matrix4().makeRotationY(random() * 3);
            expect(pair.indexed.intersectsBox(box, turn)).toBe(pair.records.intersectsBox(box, turn));
        }
        // Against another tree and another geometry, moved.
        const other = new SphereGeometry(0.4, 12, 8);
        const otherTree = computeBoundsTree.call(other);
        const pairs = (tree: BvhBoundsTree, matrix: Matrix4) => {
            const seen: number[] = [];
            tree.bvhcast(otherTree, matrix, { intersectsTriangles: (a, b, i, j) => void seen.push(i, j, ...a.a.toArray(), ...b.a.toArray()) });
            return seen;
        };
        for (let i = 0; i < 10; i++) {
            const matrix = new Matrix4().makeTranslation((random() - 0.5) * 3, (random() - 0.5) * 3, (random() - 0.5) * 3);
            expect(pairs(pair.indexed, matrix)).toEqual(pairs(pair.records, matrix));
            expect(pair.indexed.intersectsGeometry(other, matrix)).toBe(pair.records.intersectsGeometry(other, matrix));
        }
        expect(expansions).not.toHaveBeenCalled();
    });

    it('gives the same closest points (to a point and to a geometry) and getTriangleHitPointInfo', () => {
        const pair = twins(knot);
        const random = createRandom(86);
        for (let i = 0; i < 100; i++) {
            const point = new Vector3((random() - 0.5) * 4, (random() - 0.5) * 4, (random() - 0.5) * 4);
            expect(pair.indexed.closestPointToPoint(point)).toEqual(pair.records.closestPointToPoint(point));
            expect(pair.indexed.closestPointToPoint(point, {}, 0, 0.5)).toEqual(pair.records.closestPointToPoint(point, {}, 0, 0.5));
            const triangle = Math.floor(random() * pair.indexed.triangleCount);
            expect(pair.indexed.getTriangleHitPointInfo(point, triangle)).toEqual(pair.records.getTriangleHitPointInfo(point, triangle));
        }
        const other = new SphereGeometry(0.3, 10, 6);
        for (let i = 0; i < 10; i++) {
            const matrix = new Matrix4().makeTranslation((random() - 0.5) * 4, (random() - 0.5) * 4, (random() - 0.5) * 4);
            const a = [{}, {}] as const;
            const b = [{}, {}] as const;
            expect(pair.indexed.closestPointToGeometry(other, matrix, a[0], a[1])).toEqual(pair.records.closestPointToGeometry(other, matrix, b[0], b[1]));
            expect(a[1]).toEqual(b[1]);
        }
        expect(expansions).not.toHaveBeenCalled();
    });

    it('refits to moved vertices with the same boxes and answers, draw ranges included', () => {
        const pair = twins(knot);
        pair.recordsGeometry.setDrawRange(4, 900);
        pair.indexedGeometry.setDrawRange(4, 900);
        compareRaycasts(pair, 87); // makes the shifted trees, refitted below
        for (const geometry of [pair.recordsGeometry, pair.indexedGeometry]) {
            const position = geometry.attributes.position;
            for (let i = 0; i < position.count; i++) position.setXYZ(i, position.getX(i) * 1.5, position.getY(i) + Math.sin(position.getX(i) * 4) * 0.2, position.getZ(i) * 0.7);
            position.needsUpdate = true;
        }
        pair.records.refit();
        pair.indexed.refit();
        expect(pair.indexed.current()).toBe(true);
        expect(pair.indexed.layout.bounds).toEqual(pair.records.layout.bounds);
        expect(new Float32Array(pair.indexed.layout.nodes)).toEqual(new Float32Array(pair.records.layout.nodes));
        expect(compareRaycasts(pair, 88)).toBeGreaterThan(20);
        expect(pair.indexed.layout.triangleLeaves).toBe('indexed');
        expect(expansions).not.toHaveBeenCalled();
    });

    it('serializes indexed and deserializes either format (ours and three-mesh-bvh\'s)', () => {
        const pair = twins(knot);
        const indexedBytes = pair.indexed.serialize();
        const recordsBytes = pair.records.serialize();
        expect(indexedBytes.byteLength).toBeLessThan(recordsBytes.byteLength);
        const ray = new Ray(new Vector3(0, 0, 5), new Vector3(0.05, 0.1, -1).normalize());
        const want = pair.records.raycast(ray, DoubleSide);
        expect(want.length).toBeGreaterThan(0);
        for (const loaded of [deserializeBoundsTree(indexedBytes, pair.indexedGeometry), BvhBoundsTree.deserialize(indexedBytes, pair.indexedGeometry)]) {
            expect(loaded.layout.triangleLeaves).toBe('indexed');
            expect(loaded.raycast(ray, DoubleSide)).toEqual(want);
            expect(new Uint8Array(loaded.serialize())).toEqual(new Uint8Array(indexedBytes));
        }
        // Records bytes still load as records, for either geometry.
        const fromRecords = deserializeBoundsTree(recordsBytes, pair.indexedGeometry);
        expect(fromRecords.layout.triangleLeaves ?? 'records').toBe('records');
        expect(fromRecords.raycast(ray, DoubleSide)).toEqual(want);
        // three-mesh-bvh's format: the same nodes and indirect buffer from both trees, and it loads back.
        const tmbIndexed = BvhBoundsTree.serialize(pair.indexed);
        const tmbRecords = BvhBoundsTree.serialize(pair.records);
        expect(new Uint8Array(tmbIndexed.roots[0])).toEqual(new Uint8Array(tmbRecords.roots[0]));
        expect(tmbIndexed.indirectBuffer).toEqual(tmbRecords.indirectBuffer);
        expect(BvhBoundsTree.deserialize(tmbIndexed, pair.indexedGeometry).raycast(ray, DoubleSide)).toEqual(want);
        expect(expansions).not.toHaveBeenCalled();
    });
});

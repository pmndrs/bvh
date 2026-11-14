/**
 * `computeBatchedBoundsTree(index)` (PARITY 5.5): a tree per BatchedMesh
 * geometry, as three-mesh-bvh 0.9.18 makes them, and `acceleratedRaycast`
 * walking it for each instance: the same intersections as three-mesh-bvh's
 * accelerated BatchedMesh raycast and as three's own (faceIndex, face,
 * batchId, distance), indexed and not, and three's raycast again once the
 * batch changes until the trees are rebuilt.
 */

import { BatchedMesh, BoxGeometry, type BufferGeometry, DoubleSide, type Intersection, Matrix4, Mesh, MeshBasicMaterial, Object3D, Raycaster, SphereGeometry, TorusKnotGeometry, Vector3 } from 'three';
import { acceleratedRaycast as tmbAcceleratedRaycast, computeBatchedBoundsTree as tmbComputeBatchedBoundsTree } from 'three-mesh-bvh';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { createRandom } from '../trace/oracle-fixtures.ts';
import { acceleratedRaycast, BvhBoundsTree, computeBatchedBoundsTree, disposeBatchedBoundsTree, prepareBoundsTrees } from './raycast.ts';

const threeMeshRaycast = Mesh.prototype.raycast;
const threeBatchedRaycast = BatchedMesh.prototype.raycast;
beforeAll(async () => {
    await prepareBoundsTrees();
});
afterEach(() => {
    Mesh.prototype.raycast = threeMeshRaycast;
    BatchedMesh.prototype.raycast = threeBatchedRaycast;
    vi.restoreAllMocks();
});

function batch(seed: number, indexed: boolean): BatchedMesh {
    const random = createRandom(seed);
    const shapes: BufferGeometry[] = [new TorusKnotGeometry(0.6, 0.2, 40, 6), new SphereGeometry(0.5, 12, 8), new BoxGeometry(0.8, 0.5, 0.7, 2, 2, 2)];
    const geometries = indexed ? shapes : shapes.map((geometry) => geometry.toNonIndexed());
    const mesh = new BatchedMesh(24, 20000, indexed ? 60000 : 0, new MeshBasicMaterial({ side: DoubleSide }));
    const ids = geometries.map((geometry) => mesh.addGeometry(geometry));
    const dummy = new Object3D();
    for (let i = 0; i < 18; i++) {
        const id = mesh.addInstance(ids[i % ids.length]);
        dummy.position.set((random() - 0.5) * 8, (random() - 0.5) * 8, (random() - 0.5) * 8);
        dummy.quaternion.setFromAxisAngle(new Vector3(random(), random(), random()).normalize(), random() * 6);
        dummy.scale.setScalar(0.5 + random());
        dummy.updateMatrix();
        mesh.setMatrixAt(id, new Matrix4().copy(dummy.matrix));
    }
    mesh.setVisibleAt(4, false);
    mesh.updateMatrixWorld(true);
    return mesh;
}

/** A raycast's intersections as comparable keys. */
const keys = (hits: Intersection[]) =>
    hits
        .map((hit) => `${(hit as Intersection & { batchId?: number }).batchId}:${hit.faceIndex}:${hit.face!.a},${hit.face!.b},${hit.face!.c}@${hit.distance.toFixed(5)}`)
        .sort();

/** As {@link keys}, without faceIndex: the corners name the triangle. */
const cornerKeys = (hits: Intersection[]) =>
    hits.map((hit) => `${(hit as Intersection & { batchId?: number }).batchId}:${[hit.face!.a, hit.face!.b, hit.face!.c].sort((a, b) => a - b).join()}@${hit.distance.toFixed(5)}`).sort();

function raycastAll(mesh: BatchedMesh, raycaster: Raycaster): Intersection[] {
    const hits: Intersection[] = [];
    mesh.raycast(raycaster, hits);
    return hits;
}

function randomRay(random: () => number, raycaster: Raycaster): void {
    const origin = new Vector3((random() - 0.5) * 16, (random() - 0.5) * 16, (random() - 0.5) * 16);
    const target = new Vector3((random() - 0.5) * 6, (random() - 0.5) * 6, (random() - 0.5) * 6);
    raycaster.set(origin, target.sub(origin).normalize());
}

describe('computeBatchedBoundsTree', () => {
    it("builds three-mesh-bvh's boundsTrees array and gives its accelerated raycast's and three's intersections", () => {
        for (const indexed of [true, false]) {
            const ours = batch(3, indexed);
            const theirs = batch(3, indexed);
            const plain = batch(3, indexed);
            const all = computeBatchedBoundsTree.call(ours) as (BvhBoundsTree | null)[];
            expect(all).toBe(ours.boundsTrees);
            expect(all).toHaveLength(3);
            for (const tree of all) expect(tree).toBeInstanceOf(BvhBoundsTree);
            // Each tree holds its geometry's triangles only.
            expect(all.map((tree) => tree!.triangleCount)).toEqual([0, 1, 2].map((id) => Math.floor(ours.getGeometryRangeAt(id).count / 3)));
            tmbComputeBatchedBoundsTree.call(theirs as never);

            const random = createRandom(9);
            const raycaster = new Raycaster() as Raycaster & { firstHitOnly?: boolean };
            let hitRays = 0;
            for (let r = 0; r < 400; r++) {
                randomRay(random, raycaster);
                raycaster.firstHitOnly = false;
                Mesh.prototype.raycast = threeMeshRaycast;
                BatchedMesh.prototype.raycast = threeBatchedRaycast;
                const expected = raycastAll(plain, raycaster);
                Mesh.prototype.raycast = tmbAcceleratedRaycast;
                BatchedMesh.prototype.raycast = tmbAcceleratedRaycast;
                const theirHits = raycastAll(theirs, raycaster);
                Mesh.prototype.raycast = acceleratedRaycast;
                BatchedMesh.prototype.raycast = acceleratedRaycast as unknown as BatchedMesh['raycast'];
                const ourHits = raycastAll(ours, raycaster);
                const edge = [...expected, ...ourHits].some((hit) => Math.min(hit.barycoord!.x, hit.barycoord!.y, hit.barycoord!.z) < 1e-5);
                if (edge) continue;
                expect(keys(ourHits), `indexed ${indexed}, ray ${r}`).toEqual(keys(expected));
                // three-mesh-bvh reorders the batch's index in place (no indirect), so its faceIndex numbers differ; the triangles don't.
                expect(cornerKeys(ourHits), `indexed ${indexed}, ray ${r} (three-mesh-bvh)`).toEqual(cornerKeys(theirHits));
                if (expected.length > 0) hitRays++;
                // firstHitOnly: the closest per instance, as three-mesh-bvh gives it.
                raycaster.firstHitOnly = true;
                const firstOurs = raycastAll(ours, raycaster).sort((a, b) => a.distance - b.distance)[0];
                Mesh.prototype.raycast = tmbAcceleratedRaycast;
                BatchedMesh.prototype.raycast = tmbAcceleratedRaycast;
                const firstTheirs = raycastAll(theirs, raycaster).sort((a, b) => a.distance - b.distance)[0];
                expect(firstOurs ? cornerKeys([firstOurs]) : null).toEqual(firstTheirs ? cornerKeys([firstTheirs]) : null);
            }
            expect(hitRays, `indexed ${indexed}`).toBeGreaterThan(80);
        }
    });

    it('builds one geometry by id, returns null past them, disposes, and falls back to three when the batch changes', () => {
        const mesh = batch(5, true);
        expect(computeBatchedBoundsTree.call(mesh, 1)).toBeInstanceOf(BvhBoundsTree);
        expect(mesh.boundsTrees!.map((tree) => tree !== null)).toEqual([false, true, false]);
        expect(computeBatchedBoundsTree.call(mesh, 7)).toBeNull();
        disposeBatchedBoundsTree.call(mesh, 1);
        expect(mesh.boundsTrees!.every((tree) => tree === null)).toBe(true);
        computeBatchedBoundsTree.call(mesh);
        disposeBatchedBoundsTree.call(mesh);
        expect(mesh.boundsTrees!.every((tree) => tree === null)).toBe(true);

        // A changed batch: raycasts fall back to three's (with one warning) until the trees are rebuilt.
        const plain = batch(5, true);
        computeBatchedBoundsTree.call(mesh);
        const replacement = new SphereGeometry(0.5, 12, 8).scale(1.4, 0.6, 1);
        mesh.setGeometryAt(1, replacement);
        plain.setGeometryAt(1, replacement);
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
        Mesh.prototype.raycast = acceleratedRaycast;
        BatchedMesh.prototype.raycast = acceleratedRaycast as unknown as BatchedMesh['raycast'];
        const random = createRandom(12);
        const raycaster = new Raycaster();
        const compare = () => {
            for (let r = 0; r < 150; r++) {
                randomRay(random, raycaster);
                const ourHits = raycastAll(mesh, raycaster);
                BatchedMesh.prototype.raycast = threeBatchedRaycast;
                Mesh.prototype.raycast = threeMeshRaycast;
                const expected = raycastAll(plain, raycaster);
                Mesh.prototype.raycast = acceleratedRaycast;
                BatchedMesh.prototype.raycast = acceleratedRaycast as unknown as BatchedMesh['raycast'];
                if ([...expected, ...ourHits].some((hit) => Math.min(hit.barycoord!.x, hit.barycoord!.y, hit.barycoord!.z) < 1e-5)) continue;
                expect(keys(ourHits)).toEqual(keys(expected));
            }
        };
        compare();
        expect(warn).toHaveBeenCalled();
        computeBatchedBoundsTree.call(mesh);
        warn.mockClear();
        compare();
        expect(warn).not.toHaveBeenCalled();
    });
});

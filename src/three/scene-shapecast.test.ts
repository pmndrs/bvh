/**
 * BvhSceneRaycaster.shapecast against three-mesh-bvh 0.9.18's
 * `ObjectBVH.shapecast` on the same scene (meshes, an InstancedMesh, a
 * BatchedMesh, a hidden object, a multi-material mesh): the same callbacks
 * find the same objects and instances. The boxes and tree shapes differ (ours
 * are the top level's padded mesh boxes, theirs the geometry's box shrunk to
 * its sphere), so the walks are compared by what an exact test in
 * `intersectsObject` keeps, before and after objects move.
 */

import { BatchedMesh, Box3, BoxGeometry, Group, InstancedMesh, Matrix4, Mesh, MeshBasicMaterial, Object3D, Sphere, SphereGeometry, TorusKnotGeometry, Vector3, type BufferGeometry } from 'three';
import { ObjectBVH } from 'three-mesh-bvh';
import { beforeAll, describe, expect, it } from 'vitest';
import { createRandom } from '../trace/oracle-fixtures.ts';
import { CONTAINED, INTERSECTED, NOT_INTERSECTED } from './index.ts';
import { prepareBoundsTrees } from './raycast.ts';
import { BvhSceneRaycaster } from './scene-raycast.ts';

beforeAll(async () => {
    await prepareBoundsTrees();
});

function place(object: Object3D, random: () => number, extent = 10): Object3D {
    object.position.set((random() - 0.5) * extent, (random() - 0.5) * extent, (random() - 0.5) * extent);
    object.quaternion.setFromAxisAngle(new Vector3(random(), random(), random()).normalize(), random() * 6);
    const s = 0.5 + random();
    object.scale.set(s, random() < 0.3 ? -s : s * (0.6 + random() * 0.8), s);
    return object;
}

function testScene(seed: number) {
    const random = createRandom(seed);
    const root = new Group();
    const material = new MeshBasicMaterial();
    const knot = new TorusKnotGeometry(0.8, 0.25, 32, 6);
    for (let i = 0; i < 6; i++) root.add(place(new Mesh(knot, material), random));
    root.add(place(new Mesh(new SphereGeometry(0.8, 10, 8).toNonIndexed(), material), random));
    root.add(place(new Mesh(new BoxGeometry(1.2, 0.8, 1, 2, 3, 2), [material, material, material, material, material, material]), random));
    const instanced = new InstancedMesh(new BoxGeometry(0.5, 0.4, 0.3), material, 14);
    const dummy = new Object3D();
    for (let i = 0; i < 14; i++) {
        place(dummy, random, 11).updateMatrix();
        instanced.setMatrixAt(i, dummy.matrix);
    }
    instanced.count = 12;
    root.add(instanced);
    const batched = new BatchedMesh(8, 4000, 12000, material);
    const ids = [batched.addGeometry(new SphereGeometry(0.6, 10, 8)), batched.addGeometry(new BoxGeometry(0.8, 0.5, 0.7))];
    for (let i = 0; i < 6; i++) {
        const id = batched.addInstance(ids[i % 2]);
        place(dummy, random, 11).updateMatrix();
        batched.setMatrixAt(id, new Matrix4().copy(dummy.matrix));
    }
    root.add(batched);
    const hidden = place(new Mesh(new BoxGeometry(1.5, 1.5, 1.5), material), random);
    hidden.visible = false;
    root.add(hidden);
    root.updateMatrixWorld(true);
    return { root, random, instanced, movers: [root.children[0], root.children[3], hidden] };
}

/** The world matrix of an object or one of its instances. */
function worldMatrix(object: Object3D, instanceId: number): Matrix4 {
    const matrix = new Matrix4();
    const instanced = object as InstancedMesh & BatchedMesh;
    if (instanced.isInstancedMesh || instanced.isBatchedMesh) instanced.getMatrixAt(instanceId, matrix);
    return matrix.premultiply(object.matrixWorld);
}

/** Exact: whether any vertex the object (or instance) draws lies in the sphere, in world space. */
function vertexInSphere(object: Object3D, instanceId: number, sphere: Sphere): boolean {
    const batched = object as BatchedMesh;
    let geometry = (object as Mesh).geometry as BufferGeometry;
    let start = 0;
    let count = Infinity;
    if (batched.isBatchedMesh) {
        const range = batched.getGeometryRangeAt(batched.getGeometryIdAt(instanceId));
        start = range.start;
        count = range.count;
        geometry = batched.geometry;
    }
    const matrix = worldMatrix(object, instanceId);
    const position = geometry.getAttribute('position');
    const index = geometry.index;
    const end = Math.min(start + count, index ? index.count : position.count);
    const point = new Vector3();
    for (let i = start; i < end; i++) {
        point.fromBufferAttribute(position, index ? index.getX(i) : i).applyMatrix4(matrix);
        if (sphere.containsPoint(point)) return true;
    }
    return false;
}

/** The objects and instances a sphere cast keeps, with three-mesh-bvh's callbacks over either walk. */
function sphereCast(cast: (callbacks: { intersectsBounds(box: Box3): number; intersectsObject(object: Object3D, instanceId: number): void }) => unknown, sphere: Sphere): string[] {
    const kept = new Set<string>();
    cast({
        intersectsBounds: (box) => (sphere.intersectsBox(box) ? INTERSECTED : NOT_INTERSECTED),
        intersectsObject: (object, instanceId) => {
            if (vertexInSphere(object, instanceId, sphere)) kept.add(`${object.id}:${instanceId}`);
        },
    });
    return [...kept].sort();
}

describe('BvhSceneRaycaster.shapecast (ObjectBVH.shapecast)', () => {
    it("finds the same objects and instances as three-mesh-bvh's ObjectBVH, before and after objects move", () => {
        const { root, random, movers } = testScene(5);
        const ours = new BvhSceneRaycaster(root);
        let nonEmpty = 0;
        for (let round = 0; round < 2; round++) {
            const theirs = new ObjectBVH(root);
            for (let q = 0; q < 150; q++) {
                const sphere = new Sphere(new Vector3((random() - 0.5) * 12, (random() - 0.5) * 12, (random() - 0.5) * 12), 0.3 + random() * 3);
                const expected = sphereCast((callbacks) => theirs.shapecast(callbacks), sphere);
                const got = sphereCast((callbacks) => ours.shapecast(callbacks), sphere);
                expect(got, `round ${round}, query ${q}`).toEqual(expected);
                if (expected.length > 0) nonEmpty++;
            }
            for (const mover of movers) place(mover, random);
            root.updateMatrixWorld(true);
            ours.update();
        }
        expect(nonEmpty).toBeGreaterThan(100);
    });

    it("hands over three-mesh-bvh's instanceId (0 for a plain mesh), contained boxes, Box3s, and stops on a truthy return", () => {
        const { root, instanced } = testScene(6);
        const ours = new BvhSceneRaycaster(root);
        const seen = new Map<string, number>();
        let boxes = 0;
        ours.shapecast({
            intersectsBounds: (box) => {
                expect(box).toBeInstanceOf(Box3);
                boxes++;
                return CONTAINED;
            },
            intersectsObject: (object, instanceId, contained, _depth, source) => {
                expect(contained).toBe(true);
                expect(source.object).toBe(object);
                const key = `${object.id}:${instanceId}`;
                seen.set(key, (seen.get(key) ?? 0) + 1);
            },
        });
        expect(boxes).toBe(2);
        // Every object and instance once: 6 knots, a sphere, a box, 12 instances, 6 batched, the hidden one.
        expect(seen.size).toBe(27);
        expect([...seen.values()].every((count) => count === 1)).toBe(true);
        expect(seen.has(`${root.children[0].id}:0`)).toBe(true);
        expect(seen.has(`${instanced.id}:11`)).toBe(true);
        let calls = 0;
        expect(ours.shapecast({ intersectsBounds: () => INTERSECTED, intersectsObject: () => ++calls === 4 })).toBe(true);
        expect(calls).toBe(4);
        // Hidden objects can be left out, as in the raycast.
        const visibleOnly = new BvhSceneRaycaster(root, { includeHidden: false });
        let count = 0;
        visibleOnly.shapecast({ intersectsBounds: () => INTERSECTED, intersectsObject: () => void count++ });
        expect(count).toBe(26);
        expect(new BvhSceneRaycaster(new Group()).shapecast({ intersectsBounds: () => INTERSECTED })).toBe(false);
    });
});

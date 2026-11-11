/**
 * The CPU raycast drop-ins against three's own `Mesh.raycast`: on random rays
 * through meshes of every kind (indexed, non-indexed, groups with an array of
 * materials, draw ranges, FrontSide / BackSide / DoubleSide, mirrored and
 * non-uniform transforms, InstancedMesh, near/far), `acceleratedRaycast` must
 * return what three returns: the same intersections, in the same order, with
 * the same fields, within float tolerance. An intersection only one side
 * finds must sit on a triangle edge (three's Möller-Trumbore and our
 * watertight test may decide an exact edge hit differently).
 */

import {
    BackSide,
    BoxGeometry,
    type BufferGeometry,
    DoubleSide,
    FrontSide,
    Group,
    InstancedMesh,
    Mesh,
    MeshBasicMaterial,
    Object3D,
    Raycaster,
    SphereGeometry,
    TorusKnotGeometry,
    Vector3,
    type Intersection,
    type Side,
} from 'three';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { createRandom } from '../trace/oracle-fixtures.ts';
import { acceleratedRaycast, BvhBoundsTree, computeBoundsTree, disposeBoundsTree, prepareBoundsTrees } from './raycast.ts';

const threeRaycast = Mesh.prototype.raycast;

beforeAll(async () => {
    await prepareBoundsTrees();
});
afterEach(() => {
    Mesh.prototype.raycast = threeRaycast;
});

function place(object: Object3D, random: () => number, extent = 6): Object3D {
    object.position.set((random() - 0.5) * extent, (random() - 0.5) * extent, (random() - 0.5) * extent);
    object.quaternion.setFromAxisAngle(new Vector3(random(), random(), random()).normalize(), random() * 6);
    const s = 0.5 + random();
    object.scale.set(s, random() < 0.3 ? -s : s * (0.6 + random() * 0.8), s);
    return object;
}

function testScene(seed: number): Group {
    const random = createRandom(seed);
    const root = new Group();
    const sides: Side[] = [FrontSide, BackSide, DoubleSide];
    const material = (index: number) => new MeshBasicMaterial({ side: sides[index % 3] });
    const knot = new TorusKnotGeometry(0.8, 0.25, 48, 8);
    for (let i = 0; i < 3; i++) root.add(place(new Mesh(knot, material(i)), random));
    root.add(place(new Mesh(new SphereGeometry(0.8, 12, 10).toNonIndexed(), material(0)), random));
    const ranged = new SphereGeometry(0.9, 14, 10);
    ranged.setDrawRange(60, 360);
    root.add(place(new Mesh(ranged, material(2)), random));
    const box = new BoxGeometry(1.2, 0.8, 1, 2, 3, 2);
    root.add(place(new Mesh(box, [0, 1, 2, 3, 4, 5].map(material)), random));
    const groupedRange = new BoxGeometry(1, 1, 1, 2, 2, 2);
    groupedRange.setDrawRange(24, 96);
    root.add(place(new Mesh(groupedRange, [0, 1, 2, 3, 4, 5].map((i) => material(i + 1))), random));
    const instanced = new InstancedMesh(new BoxGeometry(0.5, 0.4, 0.3), material(0), 10);
    const dummy = new Object3D();
    for (let i = 0; i < 10; i++) {
        place(dummy, random, 7).updateMatrix();
        instanced.setMatrixAt(i, dummy.matrix);
    }
    root.add(instanced);
    root.updateMatrixWorld(true);
    root.traverse((object) => {
        const mesh = object as Mesh;
        if (mesh.isMesh) computeBoundsTree.call(mesh.geometry);
    });
    return root;
}

/** How far a hit is from its triangle's edges (the smallest barycentric weight). */
const edgeMargin = (hit: Intersection) => Math.min(hit.barycoord!.x, hit.barycoord!.y, hit.barycoord!.z);

function compareLists(expected: Intersection[], ours: Intersection[], problems: string[], label: string): number {
    let edgeOnly = 0;
    const key = (hit: Intersection) => `${hit.object.id}:${hit.instanceId ?? -1}:${hit.faceIndex}`;
    const ourByKey = new Map(ours.map((hit) => [key(hit), hit]));
    const expectedKeys = new Set(expected.map(key));
    for (const hit of ours) {
        if (!expectedKeys.has(key(hit))) {
            if (edgeMargin(hit) < 1e-5) edgeOnly++;
            else problems.push(`${label}: only ours hit ${key(hit)} at ${hit.distance}`);
        }
    }
    for (const want of expected) {
        const got = ourByKey.get(key(want));
        if (!got) {
            if (edgeMargin(want) < 1e-5) edgeOnly++;
            else problems.push(`${label}: only three hit ${key(want)} at ${want.distance}`);
            continue;
        }
        const close = (a: number, b: number, tolerance: number) => Math.abs(a - b) <= tolerance * Math.max(1, Math.abs(b));
        if (!close(got.distance, want.distance, 1e-5)) problems.push(`${label}: distance ${got.distance} vs ${want.distance}`);
        if (got.point.distanceTo(want.point) > 1e-4) problems.push(`${label}: point ${got.point.toArray()} vs ${want.point.toArray()}`);
        if (got.barycoord!.distanceTo(want.barycoord!) > 1e-4) problems.push(`${label}: barycoord ${got.barycoord!.toArray()} vs ${want.barycoord!.toArray()}`);
        if (got.face!.a !== want.face!.a || got.face!.b !== want.face!.b || got.face!.c !== want.face!.c) problems.push(`${label}: face corners differ`);
        if (got.face!.materialIndex !== want.face!.materialIndex) problems.push(`${label}: materialIndex ${got.face!.materialIndex} vs ${want.face!.materialIndex}`);
        if (got.face!.normal.distanceTo(want.face!.normal) > 1e-5) problems.push(`${label}: face normal differs`);
        if ((got.uv === undefined) !== (want.uv === undefined) || (got.uv && got.uv.distanceTo(want.uv!) > 1e-4)) problems.push(`${label}: uv differs`);
        if ((got.normal === undefined) !== (want.normal === undefined) || (got.normal && got.normal.distanceTo(want.normal!) > 1e-4)) problems.push(`${label}: normal differs`);
        if (got.object !== want.object) problems.push(`${label}: object differs`);
    }
    // Same order (by distance) where both found the same hits.
    if (problems.length === 0 && edgeOnly === 0 && ours.map(key).join() !== expected.map(key).join()) {
        const ties = ours.some((hit, i) => i > 0 && Math.abs(hit.distance - ours[i - 1].distance) < 1e-6);
        if (!ties) problems.push(`${label}: order differs`);
    }
    return edgeOnly;
}

function randomRay(random: () => number, raycaster: Raycaster): void {
    const origin = new Vector3((random() - 0.5) * 14, (random() - 0.5) * 14, (random() - 0.5) * 14);
    const target = new Vector3((random() - 0.5) * 5, (random() - 0.5) * 5, (random() - 0.5) * 5);
    raycaster.set(origin, target.sub(origin).normalize());
}

describe('acceleratedRaycast', () => {
    it("returns three's intersections, all of them, with the same fields", () => {
        const root = testScene(3);
        const random = createRandom(4);
        const raycaster = new Raycaster();
        const problems: string[] = [];
        let total = 0;
        let edgeOnly = 0;
        for (let r = 0; r < 4000; r++) {
            randomRay(random, raycaster);
            if (r % 4 === 1) {
                raycaster.near = random() * 4;
                raycaster.far = raycaster.near + random() * 10;
            } else {
                raycaster.near = 0;
                raycaster.far = Infinity;
            }
            Mesh.prototype.raycast = threeRaycast;
            const expected = raycaster.intersectObject(root, true);
            Mesh.prototype.raycast = acceleratedRaycast;
            const ours = raycaster.intersectObject(root, true);
            total += expected.length;
            edgeOnly += compareLists(expected, ours, problems, `ray ${r}`);
        }
        expect(problems.slice(0, 10)).toEqual([]);
        expect(total).toBeGreaterThan(3000);
        expect(edgeOnly).toBeLessThan(total * 0.001);
    });

    it('returns the closest hit only with firstHitOnly', () => {
        const root = testScene(5);
        const random = createRandom(6);
        const raycaster = new Raycaster() as Raycaster & { firstHitOnly?: boolean };
        const problems: string[] = [];
        let hits = 0;
        for (let r = 0; r < 4000; r++) {
            randomRay(random, raycaster);
            raycaster.firstHitOnly = false;
            Mesh.prototype.raycast = threeRaycast;
            const all = raycaster.intersectObject(root, true);
            Mesh.prototype.raycast = acceleratedRaycast;
            raycaster.firstHitOnly = true;
            const ours = raycaster.intersectObject(root, true);
            // One per mesh (and per instance) hit: the closest of each.
            const closestOf = new Map<string, Intersection>();
            for (const hit of all) {
                const k = `${hit.object.id}:${hit.instanceId ?? -1}`;
                if (!closestOf.has(k)) closestOf.set(k, hit);
            }
            hits += closestOf.size;
            compareLists([...closestOf.values()], ours, problems, `ray ${r}`);
        }
        expect(problems.slice(0, 10)).toEqual([]);
        expect(hits).toBeGreaterThan(1500);
    });

    it("falls back to three's raycast without a tree, and computeBoundsTree/disposeBoundsTree set and clear it", () => {
        const geometry = new BoxGeometry();
        const mesh = new Mesh(geometry, new MeshBasicMaterial());
        mesh.updateMatrixWorld();
        const raycaster = new Raycaster(new Vector3(0.1, 0.2, 5), new Vector3(0, 0, -1));
        const withoutTree: Intersection[] = [];
        acceleratedRaycast.call(mesh, raycaster, withoutTree);
        const tree = computeBoundsTree.call(geometry);
        expect((geometry as BufferGeometry & { boundsTree?: unknown }).boundsTree).toBe(tree);
        expect(tree).toBeInstanceOf(BvhBoundsTree);
        const withTree: Intersection[] = [];
        acceleratedRaycast.call(mesh, raycaster, withTree);
        expect(withTree.map((hit) => hit.faceIndex)).toEqual(withoutTree.map((hit) => hit.faceIndex));
        expect(withTree[0].distance).toBeCloseTo(4.5, 6);
        disposeBoundsTree.call(geometry);
        expect((geometry as BufferGeometry & { boundsTree?: unknown }).boundsTree).toBeUndefined();
    });
});

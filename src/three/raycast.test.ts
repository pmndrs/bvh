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
    BatchedMesh,
    BoxGeometry,
    type BufferGeometry,
    DoubleSide,
    FrontSide,
    Group,
    InstancedMesh,
    Matrix4,
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
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { createRandom } from '../trace/oracle-fixtures.ts';
import { acceleratedRaycast, BvhBoundsTree, computeBoundsTree, disposeBoundsTree, prepareBoundsTrees } from './raycast.ts';

const threeRaycast = Mesh.prototype.raycast;

beforeAll(async () => {
    await prepareBoundsTrees();
});
afterEach(() => {
    Mesh.prototype.raycast = threeRaycast;
    vi.restoreAllMocks();
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
    // three reports a triangle that reads past the index (a range's last one) with every corner NaN, at a NaN
    // distance (its watertight test fails no comparison); there is no such triangle, and we report nothing.
    // Sorting by a NaN distance leaves three's order undefined, so order isn't compared then.
    const unordered = expected.some((hit) => Number.isNaN(hit.distance));
    expected = expected.filter((hit) => !Number.isNaN(hit.distance));
    let edgeOnly = 0;
    // Corners and material too: overlapping groups, and ranges off a multiple of 3, give one faceIndex to two triangles.
    const key = (hit: Intersection) => `${hit.object.id}:${hit.instanceId ?? -1}:${hit.faceIndex}:${hit.face!.a},${hit.face!.b},${hit.face!.c}:${hit.face!.materialIndex}`;
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
    if (!unordered && problems.length === 0 && edgeOnly === 0 && ours.map(key).join() !== expected.map(key).join()) {
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

    /** Our intersections and three's for `rays` random rays through `root`, compared. */
    function compareRays(root: Object3D, seed: number, rays: number): { problems: string[]; total: number } {
        const random = createRandom(seed);
        const raycaster = new Raycaster();
        const problems: string[] = [];
        let total = 0;
        for (let r = 0; r < rays; r++) {
            randomRay(random, raycaster);
            Mesh.prototype.raycast = threeRaycast;
            const expected = raycaster.intersectObject(root, true);
            Mesh.prototype.raycast = acceleratedRaycast;
            const ours = raycaster.intersectObject(root, true);
            total += expected.length;
            compareLists(expected, ours, problems, `ray ${r}`);
        }
        Mesh.prototype.raycast = threeRaycast;
        return { problems, total };
    }

    it("follows a swapped material in an array (its side), after raycasts cached the old one", () => {
        const geometry = new BoxGeometry(2, 2, 2, 3, 3, 3);
        computeBoundsTree.call(geometry);
        const materials = [0, 1, 2, 3, 4, 5].map(() => new MeshBasicMaterial({ side: FrontSide }));
        const mesh = new Mesh(geometry, materials);
        mesh.updateMatrixWorld();
        expect(compareRays(mesh, 31, 300).problems).toEqual([]);
        for (let i = 0; i < 6; i++) materials[i] = new MeshBasicMaterial({ side: i % 2 ? BackSide : DoubleSide });
        const { problems, total } = compareRays(mesh, 32, 300);
        expect(problems.slice(0, 5)).toEqual([]);
        expect(total).toBeGreaterThan(100);
        // A side changed on the same material is read live too.
        materials[0].side = BackSide;
        materials[1].side = FrontSide;
        expect(compareRays(mesh, 33, 300).problems).toEqual([]);
    });

    it("uses the batched geometry's tree for a BatchedMesh, one cached range per geometry", () => {
        const random = createRandom(41);
        const batched = new BatchedMesh(12, 4000, 12000, new MeshBasicMaterial({ side: FrontSide }));
        const ids = [batched.addGeometry(new SphereGeometry(0.6, 10, 8)), batched.addGeometry(new BoxGeometry(0.8, 0.5, 0.7, 2, 2, 2)), batched.addGeometry(new TorusKnotGeometry(0.5, 0.15, 30, 6))];
        const dummy = new Object3D();
        for (let i = 0; i < 12; i++) {
            const id = batched.addInstance(ids[i % 3]);
            place(dummy, random, 7).updateMatrix();
            batched.setMatrixAt(id, new Matrix4().copy(dummy.matrix));
        }
        batched.setVisibleAt(5, false);
        batched.updateMatrixWorld();
        const tree = computeBoundsTree.call(batched.geometry);
        const local = vi.spyOn(tree, 'raycastLocal');
        const { problems, total } = compareRays(batched, 42, 800);
        expect(problems.slice(0, 5)).toEqual([]);
        expect(total).toBeGreaterThan(100);
        expect(local).toHaveBeenCalled();
        // One state per geometry range, each without a per-triangle array.
        const states = (tree as unknown as { rangeCache: WeakMap<object, { passes: { ranges: Uint16Array | null }[] }[]> }).rangeCache.get(batched.material)!;
        expect(states).toHaveLength(3);
        expect(states.every((state) => state.passes.length === 1 && state.passes[0].ranges === null)).toBe(true);
    });

    /** Meshes whose draw ranges and groups start off a multiple of 3, end mid-triangle, overlap, or run past the index. */
    function offGridScene(seed: number): { root: Group; geometries: BufferGeometry[] } {
        const random = createRandom(seed);
        const root = new Group();
        const sides: Side[] = [FrontSide, BackSide, DoubleSide];
        const material = (index: number) => new MeshBasicMaterial({ side: sides[index % 3] });
        const geometries: BufferGeometry[] = [];
        const add = (geometry: BufferGeometry, mat: MeshBasicMaterial | MeshBasicMaterial[]) => {
            geometries.push(geometry);
            root.add(place(new Mesh(geometry, mat), random));
        };
        // Indexed, draw ranges starting at 1 and 2 mod 3, ending mid-triangle; one past the end of the index.
        for (const [start, count, side] of [[4, 301, 2], [2, 200, 0], [301, 1e9, 1]]) {
            const geometry = new SphereGeometry(0.9, 14, 10);
            geometry.setDrawRange(start, count);
            add(geometry, material(side));
        }
        // Non-indexed, starting at 1 mod 3.
        const flat = new TorusKnotGeometry(0.7, 0.25, 30, 6).toNonIndexed();
        flat.setDrawRange(7, 400);
        add(flat, material(2));
        // Groups off a multiple of 3, overlapping, and one cut by the draw range.
        const grouped = new TorusKnotGeometry(0.8, 0.3, 32, 6);
        grouped.clearGroups();
        grouped.addGroup(0, 100, 0);
        grouped.addGroup(50, 101, 1); // overlaps the first, starts at 2 mod 3
        grouped.addGroup(151, 200, 2);
        grouped.addGroup(352, 9e9, 0);
        grouped.setDrawRange(1, 600);
        add(grouped, [0, 1, 2].map((i) => material(i + 2)));
        // Non-indexed groups.
        const flatGroups = new SphereGeometry(0.8, 12, 8).toNonIndexed();
        flatGroups.addGroup(0, 61, 0);
        flatGroups.addGroup(61, 200, 1);
        flatGroups.addGroup(200, 400, 0);
        add(flatGroups, [material(2), material(0)]);
        root.updateMatrixWorld(true);
        for (const geometry of geometries) computeBoundsTree.call(geometry);
        return { root, geometries };
    }

    it("matches three on draw ranges and groups off a multiple of 3, overlapping groups and partial tails", () => {
        const { root } = offGridScene(61);
        const { problems, total } = compareRays(root, 62, 3000);
        expect(problems.slice(0, 10)).toEqual([]);
        expect(total).toBeGreaterThan(600);
        // The shifted trees were made, and without them (tinybvh not loaded) the cast still matches, through three's.
        const random = createRandom(63);
        const raycaster = new Raycaster() as Raycaster & { firstHitOnly?: boolean };
        raycaster.firstHitOnly = true;
        const firstProblems: string[] = [];
        let firsts = 0;
        let ties = 0;
        for (let r = 0; r < 2000; r++) {
            randomRay(random, raycaster);
            raycaster.firstHitOnly = false;
            Mesh.prototype.raycast = threeRaycast;
            // Without three's NaN hits (see compareLists), which leave its sort unordered.
            const all = raycaster
                .intersectObject(root, true)
                .filter((hit) => !Number.isNaN(hit.distance))
                .sort((a, b) => a.distance - b.distance);
            raycaster.firstHitOnly = true;
            Mesh.prototype.raycast = acceleratedRaycast;
            const ours = raycaster.intersectObject(root, true);
            const closestOf = new Map<string, Intersection>();
            for (const hit of all) if (!closestOf.has(`${hit.object.id}`)) closestOf.set(`${hit.object.id}`, hit);
            // Two triangles at the same distance (a shared edge or corner): either is the closest.
            for (const hit of ours) {
                const want = closestOf.get(`${hit.object.id}`);
                if (want && want.faceIndex !== hit.faceIndex && Math.abs(want.distance - hit.distance) <= 1e-9 * Math.max(1, want.distance)) {
                    closestOf.set(`${hit.object.id}`, hit);
                    ties++;
                }
            }
            firsts += closestOf.size;
            compareLists([...closestOf.values()], ours, firstProblems, `ray ${r}`);
        }
        Mesh.prototype.raycast = threeRaycast;
        expect(firstProblems.slice(0, 10)).toEqual([]);
        expect(firsts).toBeGreaterThan(150);
        expect(ties).toBeLessThan(firsts * 0.03);
    });

    it('refits the shifted trees with the tree', () => {
        const { root, geometries } = offGridScene(71);
        expect(compareRays(root, 72, 300).problems).toEqual([]);
        for (const geometry of geometries) {
            const position = geometry.attributes.position;
            for (let i = 0; i < position.count; i++) position.setXYZ(i, position.getX(i) * 1.6, position.getY(i) * 0.7, position.getZ(i) + 0.2 * position.getX(i));
            position.needsUpdate = true;
            geometry.computeBoundingSphere();
            geometry.computeBoundingBox();
            (geometry as BufferGeometry & { boundsTree: BvhBoundsTree }).boundsTree.refit();
        }
        const { problems, total } = compareRays(root, 73, 1500);
        expect(problems.slice(0, 10)).toEqual([]);
        expect(total).toBeGreaterThan(800);
    });

    it("raycasts groups off a multiple of 3 through BvhBoundsTree.raycast with an array of materials", () => {
        const geometry = new TorusKnotGeometry(0.8, 0.3, 32, 6);
        geometry.clearGroups();
        geometry.addGroup(1, 300, 0);
        geometry.addGroup(200, 400, 1);
        const materials = [new MeshBasicMaterial({ side: DoubleSide }), new MeshBasicMaterial({ side: FrontSide })];
        const tree = computeBoundsTree.call(geometry);
        const mesh = new Mesh(geometry, materials);
        mesh.updateMatrixWorld();
        const random = createRandom(81);
        const raycaster = new Raycaster();
        let hits = 0;
        for (let r = 0; r < 500; r++) {
            const origin = new Vector3((random() - 0.5) * 8, (random() - 0.5) * 8, (random() - 0.5) * 8);
            raycaster.set(origin, new Vector3((random() - 0.5) * 1.5, (random() - 0.5) * 1.5, (random() - 0.5) * 1.5).sub(origin).normalize());
            const expected = raycaster.intersectObject(mesh);
            const ours = tree.raycast(raycaster.ray, materials);
            expect(ours.map((hit) => `${hit.faceIndex}:${hit.face!.a}:${hit.face!.materialIndex}`)).toEqual(expected.map((hit) => `${hit.faceIndex}:${hit.face!.a}:${hit.face!.materialIndex}`));
            hits += ours.length;
        }
        expect(hits).toBeGreaterThan(100);
    });

    it("falls back to three's raycast, warning once, after the positions or index changed", () => {
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
        const geometry = new SphereGeometry(1, 12, 10);
        computeBoundsTree.call(geometry);
        const mesh = new Mesh(geometry, new MeshBasicMaterial({ side: DoubleSide }));
        mesh.updateMatrixWorld();
        expect(compareRays(mesh, 51, 200).problems).toEqual([]);
        expect(warn).not.toHaveBeenCalled();
        const position = geometry.attributes.position;
        for (let i = 0; i < position.count; i++) position.setXYZ(i, position.getX(i) * 2, position.getY(i) * 0.5, position.getZ(i));
        position.needsUpdate = true;
        // three culls by both; the tree's setBoundingBox box is stale until refit, as three-mesh-bvh's.
        geometry.computeBoundingSphere();
        geometry.computeBoundingBox();
        const { problems, total } = compareRays(mesh, 52, 200);
        expect(problems).toEqual([]);
        expect(total).toBeGreaterThan(30);
        expect(warn).toHaveBeenCalledTimes(1);
        computeBoundsTree.call(geometry);
        expect(compareRays(mesh, 53, 200).problems).toEqual([]);
        geometry.setIndex(Array.from(geometry.index!.array).reverse());
        expect(compareRays(mesh, 54, 200).problems).toEqual([]);
        expect(warn).toHaveBeenCalledTimes(2);
    });
});

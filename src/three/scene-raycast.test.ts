/**
 * raycastScene against three's own `Raycaster.intersectObject(root, true)`
 * (each object's `Mesh.raycast`, one by one): the same intersections with the
 * same fields over a scene of every kind of mesh, sides, ranges off a
 * multiple of 3, InstancedMesh, BatchedMesh (a hidden instance), hidden
 * objects and layers, before and after objects move; and frustum culling
 * against brute force over each object's world box.
 */

import {
    BackSide,
    BatchedMesh,
    Box3,
    BoxGeometry,
    DoubleSide,
    FrontSide,
    Frustum,
    Group,
    InstancedMesh,
    Matrix4,
    Mesh,
    MeshBasicMaterial,
    Object3D,
    PerspectiveCamera,
    Raycaster,
    SphereGeometry,
    TorusKnotGeometry,
    Vector3,
    type Intersection,
    type Side,
} from 'three';
import { beforeAll, describe, expect, it } from 'vitest';
import { createRandom } from '../trace/oracle-fixtures.ts';
import { prepareBoundsTrees } from './raycast.ts';
import { BvhSceneRaycaster, raycastScene } from './scene-raycast.ts';
import { threeTriangleRanges } from './scene.ts';

beforeAll(async () => {
    await prepareBoundsTrees();
});

function place(object: Object3D, random: () => number, extent = 8): Object3D {
    object.position.set((random() - 0.5) * extent, (random() - 0.5) * extent, (random() - 0.5) * extent);
    object.quaternion.setFromAxisAngle(new Vector3(random(), random(), random()).normalize(), random() * 6);
    const s = 0.5 + random();
    object.scale.set(s, random() < 0.3 ? -s : s * (0.6 + random() * 0.8), s);
    return object;
}

function testScene(seed: number): { root: Group; movers: Object3D[] } {
    const random = createRandom(seed);
    const root = new Group();
    const sides: Side[] = [FrontSide, BackSide, DoubleSide];
    const material = (index: number) => new MeshBasicMaterial({ side: sides[index % 3] });
    const knot = new TorusKnotGeometry(0.8, 0.25, 48, 8);
    const movers: Object3D[] = [];
    for (let i = 0; i < 3; i++) root.add(place(new Mesh(knot, material(i)), random));
    movers.push(root.children[0]);
    root.add(place(new Mesh(new SphereGeometry(0.8, 12, 10).toNonIndexed(), material(0)), random));
    const ranged = new SphereGeometry(0.9, 14, 10);
    ranged.setDrawRange(4, 301); // off a multiple of 3
    root.add(place(new Mesh(ranged, material(2)), random));
    root.add(place(new Mesh(new BoxGeometry(1.2, 0.8, 1, 2, 3, 2), [0, 1, 2, 3, 4, 5].map(material)), random));
    const instanced = new InstancedMesh(new BoxGeometry(0.5, 0.4, 0.3), material(0), 12);
    const dummy = new Object3D();
    for (let i = 0; i < 12; i++) {
        place(dummy, random, 9).updateMatrix();
        instanced.setMatrixAt(i, dummy.matrix);
    }
    instanced.count = 10;
    root.add(instanced);
    movers.push(instanced);
    const batched = new BatchedMesh(8, 4000, 12000, material(2));
    const ids = [batched.addGeometry(new SphereGeometry(0.6, 10, 8)), batched.addGeometry(new BoxGeometry(0.8, 0.5, 0.7))];
    for (let i = 0; i < 6; i++) {
        const id = batched.addInstance(ids[i % 2]);
        place(dummy, random, 9).updateMatrix();
        batched.setMatrixAt(id, new Matrix4().copy(dummy.matrix));
    }
    batched.setVisibleAt(3, false);
    root.add(batched);
    const hidden = place(new Mesh(new BoxGeometry(1.5, 1.5, 1.5), material(2)), random);
    hidden.visible = false; // three's Raycaster still hits it
    const offLayer = place(new Mesh(new BoxGeometry(1.5, 1.5, 1.5), material(2)), random);
    offLayer.layers.set(1);
    root.add(hidden, offLayer);
    movers.push(hidden);
    root.updateMatrixWorld(true);
    return { root, movers };
}

const edgeMargin = (hit: Intersection) => Math.min(hit.barycoord!.x, hit.barycoord!.y, hit.barycoord!.z);
const keyOf = (hit: Intersection) =>
    `${hit.object.id}:${hit.instanceId ?? (hit as Intersection & { batchId?: number }).batchId ?? -1}:${hit.faceIndex}:${hit.face!.a},${hit.face!.b},${hit.face!.c}:${hit.face!.materialIndex}`;

function compare(expected: Intersection[], ours: Intersection[], problems: string[], label: string): number {
    expected = expected.filter((hit) => !Number.isNaN(hit.distance)); // three's NaN hit past an index (see raycast.test.ts)
    let edgeOnly = 0;
    const ourByKey = new Map(ours.map((hit) => [keyOf(hit), hit]));
    const expectedKeys = new Set(expected.map(keyOf));
    for (const hit of ours) {
        if (expectedKeys.has(keyOf(hit))) continue;
        if (edgeMargin(hit) < 1e-5) edgeOnly++;
        else problems.push(`${label}: only ours ${keyOf(hit)} at ${hit.distance}`);
    }
    for (const want of expected) {
        const got = ourByKey.get(keyOf(want));
        if (!got) {
            if (edgeMargin(want) < 1e-5) edgeOnly++;
            else problems.push(`${label}: only three ${keyOf(want)} at ${want.distance}`);
            continue;
        }
        if (Math.abs(got.distance - want.distance) > 1e-5 * Math.max(1, want.distance)) problems.push(`${label}: distance ${got.distance} vs ${want.distance}`);
        if (got.point.distanceTo(want.point) > 1e-4) problems.push(`${label}: point differs`);
        if (got.barycoord!.distanceTo(want.barycoord!) > 1e-4) problems.push(`${label}: barycoord differs`);
        if (got.face!.normal.distanceTo(want.face!.normal) > 1e-5) problems.push(`${label}: face normal differs`);
        if ((got.uv === undefined) !== (want.uv === undefined) || (got.uv && got.uv.distanceTo(want.uv!) > 1e-4)) problems.push(`${label}: uv differs`);
        if ((got.normal === undefined) !== (want.normal === undefined) || (got.normal && got.normal.distanceTo(want.normal!) > 1e-4)) problems.push(`${label}: normal differs`);
    }
    for (let i = 1; i < ours.length; i++) if (ours[i].distance < ours[i - 1].distance) problems.push(`${label}: not sorted`);
    return edgeOnly;
}

function randomRay(random: () => number, raycaster: Raycaster): void {
    const origin = new Vector3((random() - 0.5) * 18, (random() - 0.5) * 18, (random() - 0.5) * 18);
    const target = new Vector3((random() - 0.5) * 7, (random() - 0.5) * 7, (random() - 0.5) * 7);
    raycaster.set(origin, target.sub(origin).normalize());
}

describe('raycastScene', () => {
    it("returns three's intersectObject(root, true), all hits and the closest, before and after objects move", () => {
        const { root, movers } = testScene(3);
        const picker = new BvhSceneRaycaster(root);
        const random = createRandom(4);
        const raycaster = new Raycaster() as Raycaster & { firstHitOnly?: boolean };
        const problems: string[] = [];
        let total = 0;
        let edgeOnly = 0;
        for (let round = 0; round < 2; round++) {
            for (let r = 0; r < 2500; r++) {
                randomRay(random, raycaster);
                if (r % 4 === 1) {
                    raycaster.near = random() * 4;
                    raycaster.far = raycaster.near + random() * 12;
                } else {
                    raycaster.near = 0;
                    raycaster.far = Infinity;
                }
                raycaster.layers.set(r % 7 === 0 ? 1 : 0);
                raycaster.firstHitOnly = false;
                const expected = raycaster.intersectObject(root, true);
                const ours = raycastScene(picker, raycaster);
                total += expected.length;
                edgeOnly += compare(expected, ours, problems, `round ${round} ray ${r}`);
                raycaster.firstHitOnly = true;
                const first = raycastScene(picker, raycaster);
                const closest = expected.filter((hit) => !Number.isNaN(hit.distance)).sort((a, b) => a.distance - b.distance)[0];
                if (!!closest !== first.length > 0 || (closest && Math.abs(first[0].distance - closest.distance) > 1e-5 * Math.max(1, closest.distance))) {
                    problems.push(`round ${round} ray ${r}: closest ${first[0]?.distance} vs ${closest?.distance}`);
                }
            }
            for (const mover of movers) {
                mover.position.x += 1.3;
                mover.rotation.y += 0.7;
            }
            (movers[1] as InstancedMesh).setMatrixAt(2, new Matrix4().makeTranslation(1, 2, 3));
            root.updateMatrixWorld(true);
            expect(picker.update()).toBe(false); // same meshes: the top level only
        }
        expect(problems.slice(0, 10)).toEqual([]);
        expect(total).toBeGreaterThan(3000);
        expect(edgeOnly).toBeLessThan(total * 0.002);
    });

    it('builds trees for new geometry on update, and skips hidden objects with includeHidden: false', () => {
        const { root } = testScene(7);
        const picker = new BvhSceneRaycaster(root, { includeHidden: false });
        const count = picker.instanceCount;
        root.add(new Mesh(new SphereGeometry(2, 10, 8), new MeshBasicMaterial()));
        root.updateMatrixWorld(true);
        expect(picker.update()).toBe(true);
        expect(picker.instanceCount).toBe(count + 1);
        const raycaster = new Raycaster(new Vector3(0, 0, 10), new Vector3(0, 0, -1));
        const hidden = root.children.find((child) => !child.visible)!;
        raycaster.set(new Vector3().setFromMatrixPosition(hidden.matrixWorld).add(new Vector3(0, 0, 20)), new Vector3(0, 0, -1));
        expect(raycastScene(picker, raycaster).some((hit) => hit.object === hidden)).toBe(false);
        expect(raycaster.intersectObject(hidden).length).toBeGreaterThan(0);
    });
});

describe('BvhSceneRaycaster.intersectsFrustum', () => {
    it("returns the objects and instances whose world box meets a camera's frustum, as brute force", () => {
        const { root } = testScene(11);
        const picker = new BvhSceneRaycaster(root);
        const random = createRandom(12);
        const camera = new PerspectiveCamera(50, 1.3, 0.5, 30);
        const frustum = new Frustum();
        // Brute force: each source's range of vertices, boxed in object space, then through its world matrix.
        const sources: { key: string; object: Mesh; box: Box3; mask: number }[] = [];
        root.traverse((node) => {
            const mesh = node as Mesh;
            if (!mesh.isMesh) return;
            const add = (instanceId: number, matrix: Matrix4, start: number, count: number) => {
                const box = new Box3();
                for (const range of threeTriangleRanges(mesh.geometry, mesh.material, { start, count })) {
                    const position = mesh.geometry.attributes.position;
                    for (let i = range.start; i < range.start + range.count; i++) {
                        const vertex = mesh.geometry.index ? mesh.geometry.index.getX(i) : i;
                        box.expandByPoint(new Vector3().fromBufferAttribute(position, vertex));
                    }
                }
                if (!box.isEmpty()) sources.push({ key: `${mesh.id}:${instanceId}`, object: mesh, box: box.applyMatrix4(matrix), mask: mesh.layers.mask & 0xff });
            };
            const instanced = mesh as unknown as InstancedMesh;
            const batched = mesh as unknown as BatchedMesh;
            const m = new Matrix4();
            if (instanced.isInstancedMesh) {
                for (let i = 0; i < instanced.count; i++) add(i, m.fromArray(instanced.instanceMatrix.array, i * 16).premultiply(mesh.matrixWorld).clone(), mesh.geometry.drawRange.start, mesh.geometry.drawRange.count);
            } else if (batched.isBatchedMesh) {
                for (let i = 0; i < 6; i++) {
                    if (!batched.getVisibleAt(i)) continue;
                    const range = batched.getGeometryRangeAt(batched.getGeometryIdAt(i));
                    add(i, batched.getMatrixAt(i, new Matrix4()).premultiply(mesh.matrixWorld), range.start, range.count);
                }
            } else {
                add(-1, mesh.matrixWorld, mesh.geometry.drawRange.start, mesh.geometry.drawRange.count);
            }
        });
        let found = 0;
        for (let view = 0; view < 300; view++) {
            camera.position.set((random() - 0.5) * 24, (random() - 0.5) * 24, (random() - 0.5) * 24);
            camera.lookAt((random() - 0.5) * 6, (random() - 0.5) * 6, (random() - 0.5) * 6);
            camera.updateMatrixWorld();
            frustum.setFromProjectionMatrix(new Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse));
            const layered = view % 5 === 0;
            const expected = sources.filter((source) => (!layered || (source.mask & camera.layers.mask) !== 0) && frustum.intersectsBox(source.box)).map((source) => source.key);
            const ours = picker.intersectsFrustum(frustum, layered ? camera.layers : undefined).map((source) => `${source.object.id}:${source.instanceId}`);
            expect(ours.sort()).toEqual([...new Set(expected)].sort());
            found += ours.length;
        }
        expect(found).toBeGreaterThan(300);
    });
});

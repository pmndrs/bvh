/**
 * SkinnedMeshBVH (PARITY 3.4): a tree over a SkinnedMesh's and a morph mesh's
 * posed triangles, against three-mesh-bvh 0.9.18's `SkinnedMeshBVH` (its
 * accelerated raycast and its shapecast) and three's own raycast of the posed
 * mesh, before and after the pose moves and both trees are refitted.
 */

import { Bone, BufferAttribute, DoubleSide, Float32BufferAttribute, type Intersection, Mesh, MeshBasicMaterial, Raycaster, Skeleton, SkinnedMesh, Sphere, SphereGeometry, TorusKnotGeometry, Uint16BufferAttribute, Vector3 } from 'three';
import { acceleratedRaycast as tmbAcceleratedRaycast, SkinnedMeshBVH as TmbSkinnedMeshBVH } from 'three-mesh-bvh';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { createRandom } from '../trace/oracle-fixtures.ts';
import { acceleratedRaycast, BvhBoundsTree, prepareBoundsTrees } from './raycast.ts';
import { SkinnedMeshBVH } from './skinned-mesh-bvh.ts';

const threeRaycast = Mesh.prototype.raycast;
beforeAll(async () => {
    await prepareBoundsTrees();
});
afterEach(() => {
    Mesh.prototype.raycast = threeRaycast;
});

/** A tube along y skinned to a chain of bones, with a relative morph target. */
function skinnedTube(bones = 4): SkinnedMesh {
    const geometry = new SphereGeometry(0.4, 10, 24);
    geometry.scale(1, 4, 1);
    const position = geometry.attributes.position;
    const count = position.count;
    const skinIndex = new Uint16Array(count * 4);
    const skinWeight = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
        const t = Math.min(bones - 1.001, Math.max(0, ((position.getY(i) + 1.6) / 3.2) * (bones - 1)));
        const bone = Math.floor(t);
        skinIndex.set([bone, bone + 1, 0, 0], i * 4);
        skinWeight.set([1 - (t - bone), t - bone, 0, 0], i * 4);
    }
    geometry.setAttribute('skinIndex', new Uint16BufferAttribute(skinIndex, 4));
    geometry.setAttribute('skinWeight', new Float32BufferAttribute(skinWeight, 4));
    const bulge = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) bulge.set([position.getX(i) * 0.6, 0, position.getZ(i) * 0.6], i * 3);
    geometry.morphAttributes.position = [new BufferAttribute(bulge, 3)];
    geometry.morphTargetsRelative = true;
    const chain: Bone[] = [];
    for (let b = 0; b < bones; b++) {
        const bone = new Bone();
        bone.position.y = b === 0 ? -1.6 : 3.2 / (bones - 1);
        if (b > 0) chain[b - 1].add(bone);
        chain.push(bone);
    }
    const mesh = new SkinnedMesh(geometry, new MeshBasicMaterial({ side: DoubleSide }));
    mesh.add(chain[0]);
    mesh.position.set(0.5, -0.2, 0.3);
    mesh.scale.set(1.2, 1, -0.9); // mirrored
    mesh.updateMatrixWorld(true);
    mesh.bind(new Skeleton(chain));
    return mesh;
}

/** A non-indexed knot with an absolute morph target. */
function morphedKnot(): Mesh {
    const geometry = new TorusKnotGeometry(0.8, 0.25, 40, 8).toNonIndexed();
    const position = geometry.attributes.position;
    const target = new Float32Array(position.count * 3);
    for (let i = 0; i < position.count; i++) target.set([position.getX(i) * 1.3, position.getY(i) + 0.2 * Math.sin(position.getX(i) * 4), position.getZ(i)], i * 3);
    geometry.morphAttributes.position = [new BufferAttribute(target, 3)];
    geometry.morphTargetsRelative = false;
    const mesh = new Mesh(geometry, new MeshBasicMaterial({ side: DoubleSide }));
    mesh.position.set(-0.4, 0.3, 0);
    mesh.updateMatrixWorld(true);
    return mesh;
}

function pose(mesh: Mesh, random: () => number): void {
    const skinned = mesh as SkinnedMesh;
    if (skinned.isSkinnedMesh) for (const bone of skinned.skeleton.bones) bone.rotation.set((random() - 0.5) * 1.4, (random() - 0.5) * 1.4, (random() - 0.5) * 1.4);
    mesh.morphTargetInfluences![0] = random();
    mesh.updateMatrixWorld(true);
    if (skinned.isSkinnedMesh) skinned.skeleton.update();
}

/**
 * The mesh's raycast through `raycast`, with `tree` as the geometry's tree, hits sorted. Called on
 * the object itself: three's `SkinnedMesh` has a raycast of its own, so a patched
 * `Mesh.prototype.raycast` never runs for it (set `SkinnedMesh.prototype.raycast` too).
 */
function raycastWith(mesh: Mesh, raycast: (this: Mesh, raycaster: Raycaster, intersects: Intersection[]) => void, tree: unknown, raycaster: Raycaster): Intersection[] {
    (mesh.geometry as unknown as { boundsTree?: unknown }).boundsTree = tree;
    const hits: Intersection[] = [];
    raycast.call(mesh, raycaster, hits);
    (mesh.geometry as unknown as { boundsTree?: unknown }).boundsTree = undefined;
    return hits.sort((a, b) => a.distance - b.distance);
}

const edge = (hits: Intersection[]) => hits.some((hit) => !hit.barycoord || Math.min(hit.barycoord.x, hit.barycoord.y, hit.barycoord.z) < 1e-4);

describe('SkinnedMeshBVH', () => {
    for (const [label, make] of [
        ['skinned, morphed and mirrored', skinnedTube],
        ['morphed, not indexed', morphedKnot],
    ] as const) {
        it(`${label}: the hits of three-mesh-bvh's SkinnedMeshBVH and three's raycast, before and after the pose moves`, () => {
            const random = createRandom(label.length);
            const mesh = make();
            pose(mesh, random);
            const ours = new SkinnedMeshBVH(mesh);
            expect(ours).toBeInstanceOf(BvhBoundsTree);
            expect(ours.mesh).toBe(mesh);
            // indirect: true keeps three-mesh-bvh off the shared geometry's index.
            const theirs = new TmbSkinnedMeshBVH(mesh as SkinnedMesh, { indirect: true, verbose: false });
            // Our tree is walked for the posed mesh (no fall back to three's raycast).
            const walked = vi.spyOn(ours, 'raycastLocal');
            const raycaster = new Raycaster();
            for (let round = 0; round < 3; round++) {
                if (round > 0) {
                    pose(mesh, random);
                    ours.refit();
                    theirs.refit();
                }
                // three's SkinnedMesh raycast culls with a sphere it computed once: recompute it for the pose.
                (mesh as SkinnedMesh).computeBoundingSphere?.();
                let hitRays = 0;
                for (let r = 0; r < 200; r++) {
                    const origin = new Vector3((random() - 0.5) * 8, (random() - 0.5) * 8, (random() - 0.5) * 8);
                    const aim = new Vector3((random() - 0.5) * 2, (random() - 0.5) * 3, (random() - 0.5) * 2);
                    raycaster.set(origin, aim.sub(origin).normalize());
                    const ourHits = raycastWith(mesh, acceleratedRaycast, ours, raycaster);
                    const theirHits = raycastWith(mesh, tmbAcceleratedRaycast, theirs, raycaster);
                    const threeHits = raycastWith(mesh, (Object.getPrototypeOf(mesh) as Mesh).raycast, undefined, raycaster);
                    if (edge(ourHits) || edge(threeHits)) continue;
                    for (const [name, other] of [['three-mesh-bvh', theirHits], ['three', threeHits]] as const) {
                        expect(ourHits.map((hit) => hit.faceIndex), `${label} round ${round} ray ${r} vs ${name}`).toEqual(other.map((hit) => hit.faceIndex));
                        ourHits.forEach((hit, i) => {
                            expect(Math.abs(hit.distance - other[i].distance)).toBeLessThan(1e-4);
                            expect(hit.point.distanceTo(other[i].point)).toBeLessThan(1e-4);
                        });
                    }
                    // three's fields: the posed face normal, the geometry's interpolated normal and uv.
                    ourHits.forEach((hit, i) => {
                        expect(hit.face!.normal.distanceTo(threeHits[i].face!.normal)).toBeLessThan(1e-4);
                        expect([hit.face!.a, hit.face!.b, hit.face!.c]).toEqual([threeHits[i].face!.a, threeHits[i].face!.b, threeHits[i].face!.c]);
                        if (threeHits[i].uv) expect(hit.uv!.distanceTo(threeHits[i].uv!)).toBeLessThan(1e-4);
                        if (threeHits[i].normal) expect(hit.normal!.distanceTo(threeHits[i].normal!)).toBeLessThan(1e-4);
                    });
                    if (threeHits.length > 0) hitRays++;
                }
                expect(hitRays, `${label} round ${round}`).toBeGreaterThan(20);
                expect(walked.mock.calls.length).toBeGreaterThan(0);

                // shapecast: the posed triangles, as three-mesh-bvh hands them over.
                for (let q = 0; q < 40; q++) {
                    const sphere = new Sphere(new Vector3((random() - 0.5) * 3, (random() - 0.5) * 4, (random() - 0.5) * 3), 0.2 + random() * 0.6);
                    const found = (cast: (callbacks: { intersectsBounds: (box: { intersectsSphere(s: Sphere): boolean }) => boolean; intersectsTriangle: (triangle: { intersectsSphere(s: Sphere): boolean }, index: number) => void }) => unknown, resolve: (i: number) => number) => {
                        const set = new Set<number>();
                        cast({ intersectsBounds: (box) => box.intersectsSphere(sphere), intersectsTriangle: (triangle, index) => void (triangle.intersectsSphere(sphere) && set.add(resolve(index))) });
                        return [...set].sort((a, b) => a - b);
                    };
                    const ourSet = found((callbacks) => ours.shapecast(callbacks as never), (i) => i);
                    const theirSet = found((callbacks) => theirs.shapecast(callbacks as never), (i) => (theirs as unknown as { resolvePrimitiveIndex(i: number): number }).resolvePrimitiveIndex(i));
                    // f32 posed corners (ours) against three-mesh-bvh's doubles: a triangle grazing the sphere may differ.
                    const differ = ourSet.filter((t) => !theirSet.includes(t)).length + theirSet.filter((t) => !ourSet.includes(t)).length;
                    expect(differ, `${label} round ${round} sphere ${q}`).toBeLessThanOrEqual(1);
                }
            }
        });
    }

    it('refuses a non-mesh and a replaced geometry, and refits to arrays given', () => {
        expect(() => new SkinnedMeshBVH({} as Mesh)).toThrow(/must be a Mesh/);
        const mesh = morphedKnot();
        const tree = new SkinnedMeshBVH(mesh);
        expect(tree.followsPose).toBe(true);
        expect(tree.hitGeometry.getAttribute('position').array).toBe(tree.posedPositions);
        tree.refit({ positions: Float32Array.from(tree.posedPositions), indices: null });
        mesh.geometry = mesh.geometry.clone();
        expect(() => tree.refit()).toThrow(/replaced/);
    });
});

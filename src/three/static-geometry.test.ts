/**
 * StaticGeometryGenerator (PARITY 6.3): the baked geometry is three-mesh-bvh
 * 0.9.18's to the bit (positions, normals, index, groups) for skinned,
 * morphed, plain and mirrored meshes; each baked position is three's own CPU
 * pose (`getVertexPosition`, moved by `matrixWorld`); and a CPU tree over it
 * hits what three's raycast of the posed meshes hits, also after the pose
 * moves and the same geometry is regenerated in place and refitted.
 */

import { Bone, BoxGeometry, BufferAttribute, Float32BufferAttribute, Group, type Intersection, Mesh, MeshBasicMaterial, Raycaster, Skeleton, SkinnedMesh, SphereGeometry, TorusKnotGeometry, Uint16BufferAttribute, Vector3 } from 'three';
import { StaticGeometryGenerator as TmbStaticGeometryGenerator } from 'three-mesh-bvh';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { createRandom } from '../trace/oracle-fixtures.ts';
import { acceleratedRaycast, computeBoundsTree, prepareBoundsTrees } from './raycast.ts';
import { StaticGeometryGenerator } from './static-geometry.ts';

const threeRaycast = Mesh.prototype.raycast;
beforeAll(async () => {
    await prepareBoundsTrees();
});
afterEach(() => {
    Mesh.prototype.raycast = threeRaycast;
});

/** A tube along y skinned to a chain of bones, with two relative morph targets. */
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
    const mesh = new SkinnedMesh(geometry, new MeshBasicMaterial());
    mesh.add(chain[0]);
    mesh.updateMatrixWorld(true);
    mesh.bind(new Skeleton(chain));
    return mesh;
}

/** A knot with an absolute morph target. */
function morphedKnot(): Mesh {
    const geometry = new TorusKnotGeometry(0.8, 0.25, 48, 8);
    const position = geometry.attributes.position;
    const target = new Float32Array(position.count * 3);
    for (let i = 0; i < position.count; i++) target.set([position.getX(i) * 1.3, position.getY(i) + 0.2 * Math.sin(position.getX(i) * 4), position.getZ(i)], i * 3);
    geometry.morphAttributes.position = [new BufferAttribute(target, 3)];
    geometry.morphTargetsRelative = false;
    return new Mesh(geometry, new MeshBasicMaterial());
}

function scene() {
    const root = new Group();
    const tube = skinnedTube();
    tube.position.set(-2, 0, 0.5);
    const knot = morphedKnot();
    knot.position.set(2, 0.5, -1);
    knot.morphTargetInfluences![0] = 0.6;
    const mirrored = new Mesh(new BoxGeometry(1, 0.6, 0.8, 2, 2, 2), [0, 1, 2, 3, 4, 5].map(() => new MeshBasicMaterial()));
    mirrored.position.set(0, -2, 1);
    mirrored.scale.set(-1.2, 1, 0.9);
    const hidden = new Mesh(new BoxGeometry(), new MeshBasicMaterial());
    hidden.visible = false;
    root.add(tube, knot, mirrored, hidden);
    root.updateMatrixWorld(true);
    return { root, tube, knot, mirrored };
}

function pose(tube: SkinnedMesh, knot: Mesh, random: () => number): void {
    for (const bone of tube.skeleton.bones) bone.rotation.set((random() - 0.5) * 1.4, (random() - 0.5) * 1.4, (random() - 0.5) * 1.4);
    tube.morphTargetInfluences![0] = random();
    knot.morphTargetInfluences![0] = random();
    tube.parent!.updateMatrixWorld(true);
}

describe('StaticGeometryGenerator', () => {
    it("bakes three-mesh-bvh's geometry bit for bit, and three's own CPU pose", () => {
        const random = createRandom(4);
        const { root, tube, knot } = scene();
        pose(tube, knot, random);
        const ours = new StaticGeometryGenerator(root);
        const theirs = new TmbStaticGeometryGenerator(root);
        expect(ours.meshes).toEqual(theirs.meshes);
        expect(ours.meshes).toHaveLength(3); // the hidden mesh is left out (traverseVisible)
        expect(ours.getMaterials()).toEqual(theirs.getMaterials());
        const baked = ours.generate();
        const reference = theirs.generate();
        expect(Object.keys(baked.attributes).sort()).toEqual(Object.keys(reference.attributes).sort());
        for (const name of Object.keys(reference.attributes)) expect(Array.from(baked.attributes[name].array), name).toEqual(Array.from(reference.attributes[name].array));
        expect(Array.from(baked.index!.array)).toEqual(Array.from(reference.index!.array));
        expect(baked.groups).toEqual(reference.groups);

        // three's CPU pose: getVertexPosition (morph, then skinning), moved by matrixWorld.
        let offset = 0;
        const v = new Vector3();
        for (const mesh of [tube, knot]) {
            const count = mesh.geometry.attributes.position.count;
            for (let i = 0; i < count; i++) {
                mesh.getVertexPosition(i, v).applyMatrix4(mesh.matrixWorld);
                const at = (offset + i) * 3;
                expect([baked.attributes.position.array[at], baked.attributes.position.array[at + 1], baked.attributes.position.array[at + 2]]).toEqual([Math.fround(v.x), Math.fround(v.y), Math.fround(v.z)]);
            }
            offset += count;
        }
    });

    it("gives a CPU tree the live pose: the hits of three's raycast of the posed meshes, after regenerating in place and refit", () => {
        const random = createRandom(9);
        const { root, tube, knot, mirrored } = scene();
        pose(tube, knot, random);
        const generator = new StaticGeometryGenerator(root);
        const geometry = generator.generate();
        const tree = computeBoundsTree.call(geometry);
        const collider = new Mesh(geometry, generator.getMaterials());
        collider.updateMatrixWorld(true);
        const raycaster = new Raycaster();
        const compare = (label: string) => {
            // three's SkinnedMesh raycast culls with a sphere it computed once: recompute it for the current pose.
            tube.computeBoundingSphere();
            let hitRays = 0;
            for (let r = 0; r < 300; r++) {
                const origin = new Vector3((random() - 0.5) * 12, (random() - 0.5) * 12, (random() - 0.5) * 12);
                const aim = new Vector3((random() - 0.5) * 4, (random() - 0.5) * 3, (random() - 0.5) * 3);
                raycaster.set(origin, aim.sub(origin).normalize());
                Mesh.prototype.raycast = threeRaycast;
                const expected: Intersection[] = [];
                for (const mesh of [tube, knot, mirrored]) mesh.raycast(raycaster, expected);
                expected.sort((a, b) => a.distance - b.distance);
                Mesh.prototype.raycast = acceleratedRaycast;
                const ours: Intersection[] = [];
                collider.raycast(raycaster, ours);
                ours.sort((a, b) => a.distance - b.distance);
                const edge = [...expected, ...ours].some((hit) => Math.min(hit.barycoord!.x, hit.barycoord!.y, hit.barycoord!.z) < 1e-4);
                if (edge) continue;
                expect(ours.length, `${label}, ray ${r}`).toBe(expected.length);
                ours.forEach((hit, i) => {
                    expect(Math.abs(hit.distance - expected[i].distance), `${label}, ray ${r}`).toBeLessThan(1e-4);
                    expect(hit.point.distanceTo(expected[i].point)).toBeLessThan(1e-4);
                });
                if (expected.length > 0) hitRays++;
            }
            expect(hitRays, label).toBeGreaterThan(30);
        };
        compare('first pose');

        // The pose moves: the same geometry is regenerated in place (same arrays, index untouched) and refitted.
        const positionArray = geometry.attributes.position.array;
        const index = geometry.index;
        const indexArray = Array.from(index!.array);
        pose(tube, knot, random);
        expect(generator.generate(geometry)).toBe(geometry);
        expect(geometry.attributes.position.array).toBe(positionArray);
        expect(geometry.index).toBe(index);
        expect(Array.from(geometry.index!.array)).toEqual(indexArray);
        tree.refit();
        compare('second pose');
    });

    it('skips meshes that did not change, re-bakes a morph weight change, and throws when a mesh no longer fits the geometry', () => {
        const { root, tube, knot } = scene();
        const generator = new StaticGeometryGenerator(root);
        const geometry = generator.generate();
        const before = Float32Array.from(geometry.attributes.position.array);
        expect(Array.from(generator.generate(geometry).attributes.position.array)).toEqual(Array.from(before));
        // Only a morph weight changes: three-mesh-bvh 0.9.18 keeps the old pose here; this port re-bakes the mesh.
        knot.morphTargetInfluences![0] = 0.1;
        expect(Array.from(generator.generate(geometry).attributes.position.array)).not.toEqual(Array.from(before));
        const replaced = new Float32BufferAttribute(new Float32Array(9), 3);
        replaced.needsUpdate = true; // a new version: the diff sees the mesh changed
        knot.geometry.setAttribute('position', replaced);
        expect(() => generator.generate(geometry)).toThrow();
        expect(new StaticGeometryGenerator([]).generate().attributes.position.count).toBe(0);
        expect(tube.isSkinnedMesh).toBe(true);
    });
});

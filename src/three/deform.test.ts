/**
 * Skinned and morphed meshes in Node: the pose pass's deform box must hold
 * every vertex three's own CPU math (`getVertexPosition`: morphs, then
 * `applyBoneTransform`) puts anywhere, over an animated skeleton and
 * influences; and the sync must add posed objects as deforming meshes of
 * their own, pose them each update, move their box when the pose leaves it,
 * and drop the pass with the object. The GPU pass itself is checked on the
 * `three-skinned` GPU page against the same CPU reference.
 */

import { Bone, BoxGeometry, BufferAttribute, Float32BufferAttribute, Group, Mesh, MeshBasicMaterial, Skeleton, SkinnedMesh, SphereGeometry, Uint16BufferAttribute, Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import { createFakeGpu } from '../../tests/support/fake-gpu.ts';
import type { MeshRefitVertices } from '../gpu-mesh-refit.ts';
import type { MeshGeometry } from '../trace/mesh-bvh.ts';
import type { Mat4 } from '../trace/top-level.ts';
import type { SceneInstanceChange, SceneInstanceDescriptor, SceneMeshOptions } from '../scene/scene-trees.ts';
import { createRandom } from '../trace/oracle-fixtures.ts';
import { BvhPosePass, threeDeforms, type BvhPosePassLike } from './deform.ts';
import { SceneTreesObject3DSync, type SceneTreesTarget } from './scene-trees-sync.ts';

/** A renderer as far as BvhPosePass's constructor and update use one (no shaders run). */
function fakeRenderer() {
    const gpu = createFakeGpu();
    const records = new WeakMap<object, Record<string, unknown>>();
    let computes = 0;
    const renderer = {
        backend: {
            isWebGPUBackend: true,
            device: gpu.device,
            get(object: object) {
                let record = records.get(object);
                if (!record) records.set(object, (record = {}));
                return record;
            },
            destroyAttribute() {},
        },
        _attributes: { has: () => false },
        compute() {
            computes++;
        },
        get computes() {
            return computes;
        },
    };
    return { renderer, gpu };
}

/** A bent tube: a cylinder of `rings` rings along y, skinned to a chain of `bones` bones, with two morph targets. */
function skinnedTube(random: () => number, bones = 4, relative = true): SkinnedMesh {
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
    const twist = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        const x = position.getX(i);
        const y = position.getY(i);
        const z = position.getZ(i);
        if (relative) {
            bulge.set([x * 0.6, 0, z * 0.6], i * 3);
            twist.set([z * 0.3, random() * 0.05, -x * 0.3], i * 3);
        } else {
            bulge.set([x * 1.6, y, z * 1.6], i * 3);
            twist.set([x + z * 0.3, y, z - x * 0.3], i * 3);
        }
    }
    geometry.morphAttributes.position = [new BufferAttribute(bulge, 3), new BufferAttribute(twist, 3)];
    geometry.morphTargetsRelative = relative;
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

/** Moves the skeleton and influences to a random pose (and the root bone away, as root motion does). */
function animate(mesh: SkinnedMesh, random: () => number, frame: number): void {
    for (const bone of mesh.skeleton.bones) bone.rotation.set((random() - 0.5) * 1.6, (random() - 0.5) * 1.6, (random() - 0.5) * 1.6);
    mesh.skeleton.bones[0].position.set(frame * 0.7, -1.6 + Math.sin(frame), frame * -0.3);
    mesh.morphTargetInfluences![0] = random();
    mesh.morphTargetInfluences![1] = random() * 1.5 - 0.25;
    mesh.updateMatrixWorld(true);
}

/** three's CPU pose of every vertex (morphs, then skinning). */
function cpuPose(mesh: Mesh): Vector3[] {
    const count = mesh.geometry.attributes.position.count;
    return Array.from({ length: count }, (_, i) => mesh.getVertexPosition(i, new Vector3()));
}

function contains(box: Float32Array, points: Vector3[]): string[] {
    const outside: string[] = [];
    points.forEach((p, i) => {
        if (p.x < box[0] || p.y < box[1] || p.z < box[2] || p.x > box[3] || p.y > box[4] || p.z > box[5]) outside.push(`vertex ${i} at ${p.toArray()} outside ${Array.from(box)}`);
    });
    return outside;
}

describe('BvhPosePass', () => {
    it("bounds every vertex of three's CPU pose: skinned and morphed, relative and absolute morphs, root motion", () => {
        const { renderer } = fakeRenderer();
        for (const relative of [true, false]) {
            const random = createRandom(relative ? 3 : 4);
            const mesh = skinnedTube(random, 5, relative);
            const pass = new BvhPosePass(renderer, mesh);
            expect(pass.skinned).toBe(true);
            expect(pass.morphTargets).toBe(2);
            let looseness = 0;
            for (let frame = 0; frame < 40; frame++) {
                animate(mesh, random, frame);
                pass.update();
                const box = pass.bounds();
                const points = cpuPose(mesh);
                expect(contains(box, points).slice(0, 3)).toEqual([]);
                const exact = new Float32Array(6).fill(Infinity);
                exact.fill(-Infinity, 3);
                for (const p of points) {
                    exact[0] = Math.min(exact[0], p.x);
                    exact[1] = Math.min(exact[1], p.y);
                    exact[2] = Math.min(exact[2], p.z);
                    exact[3] = Math.max(exact[3], p.x);
                    exact[4] = Math.max(exact[4], p.y);
                    exact[5] = Math.max(exact[5], p.z);
                }
                const size = (b: Float32Array) => (b[3] - b[0]) + (b[4] - b[1]) + (b[5] - b[2]);
                looseness = Math.max(looseness, size(box) / size(exact));
            }
            // Conservative, not wildly so (per-bone boxes carried through each bone).
            expect(looseness).toBeLessThan(4);
            pass.dispose();
        }
    });

    it('bounds a morph-only mesh, and a skinned mesh with negative weights through the CPU pose', () => {
        const { renderer } = fakeRenderer();
        const random = createRandom(9);
        const morphed = new Mesh(new BoxGeometry(1, 2, 1, 2, 2, 2), new MeshBasicMaterial());
        const position = morphed.geometry.attributes.position;
        morphed.geometry.morphAttributes.position = [new Float32BufferAttribute(Array.from({ length: position.count * 3 }, () => random() - 0.5), 3)];
        morphed.geometry.morphTargetsRelative = true;
        morphed.updateMorphTargets();
        expect(threeDeforms(morphed)).toBe(true);
        expect(threeDeforms(new Mesh(new BoxGeometry()))).toBe(false);
        const pass = new BvhPosePass(renderer, morphed);
        for (const influence of [0, 0.5, 1, -0.7, 2]) {
            morphed.morphTargetInfluences![0] = influence;
            pass.update();
            expect(contains(pass.bounds(), cpuPose(morphed))).toEqual([]);
        }
        const tube = skinnedTube(random, 3);
        (tube.geometry.attributes.skinWeight as BufferAttribute).setX(5, -0.2);
        const negative = new BvhPosePass(renderer, tube);
        animate(tube, random, 2);
        negative.update();
        expect(contains(negative.bounds(), cpuPose(tube))).toEqual([]);
    });
});

/** Records what the sync asks of SceneTrees, posing included. */
class PosingTrees implements SceneTreesTarget {
    readonly log: string[] = [];
    readonly meshes = new Map<number, { geometry: MeshGeometry; options?: SceneMeshOptions }>();
    readonly instances = new Map<number, SceneInstanceDescriptor>();
    readonly poses = new Map<number, MeshRefitVertices>();
    readonly bounds = new Map<number, number[]>();
    private next = 0;
    addMesh(geometry: MeshGeometry, options?: SceneMeshOptions): number {
        const handle = this.next++;
        this.meshes.set(handle, { geometry, options });
        this.log.push(`addMesh ${handle}${options?.deforming ? ' deforming' : ''}`);
        return handle;
    }
    removeMesh(mesh: number): void {
        this.meshes.delete(mesh);
        this.log.push(`removeMesh ${mesh}`);
    }
    addInstance(descriptor: SceneInstanceDescriptor): number {
        const handle = this.next++;
        this.instances.set(handle, descriptor);
        return handle;
    }
    setTransform(instance: number, objectToWorld: Mat4): void {
        this.instances.set(instance, { ...this.instances.get(instance)!, objectToWorld: Array.from(objectToWorld) });
    }
    setInstance(_instance: number, _change: SceneInstanceChange): void {}
    removeInstance(instance: number): void {
        this.instances.delete(instance);
    }
    setPose(mesh: number, vertices: MeshRefitVertices): void {
        this.poses.set(mesh, vertices);
        this.log.push(`setPose ${mesh}`);
    }
    setDeformBounds(mesh: number, bounds: ArrayLike<number>): void {
        this.bounds.set(mesh, Array.from(bounds));
        this.log.push(`setDeformBounds ${mesh}`);
    }
}

describe('SceneTreesObject3DSync with skinned and morphed meshes', () => {
    it('adds posed objects as deforming meshes of their own, poses them each update, follows their box, and drops the pass with them', () => {
        const { renderer, gpu } = fakeRenderer();
        const random = createRandom(21);
        const root = new Group();
        const a = skinnedTube(random, 4);
        const b = new SkinnedMesh(a.geometry, [new MeshBasicMaterial(), new MeshBasicMaterial()]); // same geometry, its own pose
        a.geometry.addGroup(0, 300, 0);
        a.geometry.addGroup(300, Infinity, 1);
        b.add(b.children[0] ?? new Bone());
        b.bind(a.skeleton);
        const still = a.geometry.clone();
        still.morphAttributes = {};
        const rigid = new Mesh(still, new MeshBasicMaterial()); // neither skinned nor morphed: a rest-pose mesh
        root.add(a, b, rigid);
        root.updateMatrixWorld(true);
        const passes: BvhPosePassLike[] = [];
        const disposed: BvhPosePassLike[] = [];
        const trees = new PosingTrees();
        const sync = new SceneTreesObject3DSync(trees, root, {
            renderer,
            createPosePass: (mesh) => {
                const pass = new BvhPosePass(renderer, mesh);
                const dispose = pass.dispose.bind(pass);
                pass.dispose = () => {
                    disposed.push(pass);
                    dispose();
                };
                passes.push(pass);
                return pass;
            },
        });
        // a: one mesh; b: two groups; rigid: one rest-pose mesh.
        expect(passes).toHaveLength(2);
        const deforming = [...trees.meshes.values()].filter((mesh) => mesh.options?.deforming);
        expect(deforming).toHaveLength(3);
        expect(trees.meshes.size).toBe(4);

        trees.log.length = 0;
        const before = renderer.computes;
        animate(a, random, 3);
        root.updateMatrixWorld(true);
        sync.update();
        expect(renderer.computes - before).toBe(2);
        expect(trees.log.filter((line) => line.startsWith('setPose'))).toHaveLength(3);
        expect(sync.poseCounts.posed).toBe(2);
        // The root bone walked away: the box moved, and holds the CPU pose.
        expect(sync.poseCounts.boundsMoved).toBeGreaterThan(0);
        const aMesh = [...trees.meshes.entries()].find(([, mesh]) => mesh.options?.deforming && mesh.geometry.indices?.length === a.geometry.index!.count)![0];
        expect(contains(Float32Array.from(trees.bounds.get(aMesh)!), cpuPose(a))).toEqual([]);
        // An indexed range's refit reads our index buffer and the pass's vec4 vertices.
        const vertices = trees.poses.get(aMesh)!;
        expect(vertices).toMatchObject({ positionBuffer: passes[0].buffer, vertexCount: a.geometry.attributes.position.count, positionStrideFloats: 4, indexFormat: 'uint32' });

        // Hidden objects aren't posed.
        b.visible = false;
        sync.update();
        expect(sync.poseCounts.posed).toBe(1);

        // Removing an object drops its meshes and its pass.
        root.remove(b);
        sync.sync();
        expect(disposed).toEqual([passes[1]]);
        expect(trees.meshes.size).toBe(2);
        sync.dispose();
        expect(disposed).toEqual([passes[1], passes[0]]);
        void gpu;
    });

    it('reads skinned meshes in their rest pose without a renderer, or with deforming: false', () => {
        const random = createRandom(31);
        const mesh = skinnedTube(random);
        for (const options of [{}, { deforming: false }]) {
            const trees = new PosingTrees();
            new SceneTreesObject3DSync(trees, mesh, options);
            expect([...trees.meshes.values()].every((entry) => !entry.options?.deforming)).toBe(true);
        }
        expect(() => new SceneTreesObject3DSync(new PosingTrees(), mesh, { deforming: true })).toThrow(/needs a renderer/);
    });
});

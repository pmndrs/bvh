/**
 * Skinned and morphed meshes in Node: the pose pass's deform box must hold
 * every vertex three's own CPU math (`getVertexPosition`: morphs, then
 * `applyBoneTransform`) puts anywhere, over an animated skeleton and
 * influences; and the sync must add posed objects as deforming meshes of
 * their own, pose them each update, move their box when the pose leaves it,
 * and drop the pass with the object. The GPU pass itself is checked on the
 * `three-skinned` GPU page against the same CPU reference.
 */

import { Bone, BoxGeometry, BufferAttribute, Float32BufferAttribute, Group, InstancedMesh, Mesh, MeshBasicMaterial, Skeleton, SkinnedMesh, SphereGeometry, Uint16BufferAttribute, Vector3 } from 'three';
import { describe, expect, it, vi } from 'vitest';
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
    let compiles = 0;
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
        async compileComputeAsync() {
            compiles++;
        },
        get computes() {
            return computes;
        },
        get compiles() {
            return compiles;
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

/**
 * The pose kernel in f32, step by step as the WGSL runs it (\`Math.fround\` after
 * every operation): the morph, \`bindMatrix * vec4(p, 1)\`, each
 * \`(bone * weight) * v\` summed in order, then \`bindMatrixInverse * sum\`.
 */
function kernelF32(mesh: Mesh): Vector3[] {
    const f = Math.fround;
    const geometry = mesh.geometry;
    const position = geometry.attributes.position;
    const targets = geometry.morphAttributes.position ?? [];
    const influences = (mesh.morphTargetInfluences ?? []).map((value) => f(value));
    let sum = 0;
    for (const value of mesh.morphTargetInfluences ?? []) sum += value;
    const base = f(geometry.morphTargetsRelative ? 1 : 1 - sum);
    const skinned = mesh as SkinnedMesh;
    const isSkinned = skinned.isSkinnedMesh === true;
    if (isSkinned) skinned.skeleton.update();
    const bind = Float32Array.from(isSkinned ? skinned.bindMatrix.elements : []);
    const inverse = Float32Array.from(isSkinned ? skinned.bindMatrixInverse.elements : []);
    const bones = isSkinned ? skinned.skeleton.boneMatrices! : new Float32Array();
    const mulMat = (m: ArrayLike<number>, at: number, v: number[]) =>
        [0, 1, 2, 3].map((row) => f(f(f(f(m[at + row] * v[0]) + f(m[at + 4 + row] * v[1])) + f(m[at + 8 + row] * v[2])) + f(m[at + 12 + row] * v[3])));
    return Array.from({ length: position.count }, (_, i) => {
        let p = [f(position.getX(i)), f(position.getY(i)), f(position.getZ(i))];
        if (targets.length > 0) {
            p = p.map((c) => f(c * base));
            targets.forEach((target, t) => {
                const w = influences[t];
                if (w === 0) return;
                const d = [target.getX(i), target.getY(i), target.getZ(i)];
                p = p.map((c, k) => f(c + f(f(d[k]) * w)));
            });
        }
        if (isSkinned) {
            const v = mulMat(bind, 0, [...p, 1]);
            let s = [0, 0, 0, 0];
            for (let k = 0; k < 4; k++) {
                const w = f(geometry.attributes.skinWeight.getComponent(i, k));
                const b = geometry.attributes.skinIndex.getComponent(i, k);
                const scaled = Array.from(bones.subarray(b * 16, b * 16 + 16), (m) => f(m * w));
                const term = mulMat(scaled, 0, v);
                s = k === 0 ? term : s.map((c, j) => f(c + term[j]));
            }
            p = mulMat(inverse, 0, s).slice(0, 3);
        }
        return new Vector3(p[0], p[1], p[2]);
    });
}

/** A tube on a chain of 4 bones, each vertex weighted to 3 or 4 of them (weights summing to 1), with two morph targets and a bind matrix that isn't the identity. */
function fourBoneMesh(random: () => number, relative: boolean, place: (mesh: SkinnedMesh) => void): SkinnedMesh {
    const geometry = new SphereGeometry(0.5, 12, 20);
    geometry.scale(1, 3, 1);
    const position = geometry.attributes.position;
    const count = position.count;
    const skinIndex = new Uint16Array(count * 4);
    const skinWeight = new Float32Array(count * 4);
    const bulge = new Float32Array(count * 3);
    const twist = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        const weights = [random(), random(), random(), i % 2 === 0 ? random() : 0];
        const total = weights.reduce((a, b) => a + b, 0);
        skinIndex.set([0, 1, 2, 3].sort(() => random() - 0.5), i * 4);
        skinWeight.set(weights.map((w) => w / total), i * 4);
        const [x, y, z] = [position.getX(i), position.getY(i), position.getZ(i)];
        bulge.set(relative ? [x * 0.5, random() * 0.1, z * 0.5] : [x * 1.5, y, z * 1.5], i * 3);
        twist.set(relative ? [z * 0.3, 0, -x * 0.3] : [x + z * 0.3, y * 1.2, z - x * 0.3], i * 3);
    }
    geometry.setAttribute('skinIndex', new Uint16BufferAttribute(skinIndex, 4));
    geometry.setAttribute('skinWeight', new Float32BufferAttribute(skinWeight, 4));
    geometry.morphAttributes.position = [new BufferAttribute(bulge, 3), new BufferAttribute(twist, 3)];
    geometry.morphTargetsRelative = relative;
    const chain: Bone[] = [];
    for (let b = 0; b < 4; b++) {
        const bone = new Bone();
        bone.position.y = b === 0 ? -1.5 : 1;
        if (b > 0) chain[b - 1].add(bone);
        chain.push(bone);
    }
    const mesh = new SkinnedMesh(geometry, new MeshBasicMaterial());
    mesh.add(chain[0]);
    place(mesh);
    mesh.updateMatrixWorld(true);
    mesh.bind(new Skeleton(chain));
    return mesh;
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

    it("bounds the kernel's f32 pose far from the origin (the f32 room grows with the coordinates, not the mesh)", () => {
        const { renderer } = fakeRenderer();
        const random = createRandom(41);
        // Rigid single-bone weights and unrotated bones: the box is tight, so f32 rounding at 5,000 units shows.
        for (const far of [
            { root: [5000.3, -3000.7, 2000.1], at: [0, 0, 0] },
            { root: [0.2, -1.5, 0.1], at: [4000.5, -2500.25, 6000.75] },
            { root: [-7000.4, 3000.2, 9000.6], at: [3000.5, 1000.25, -2000.75] },
        ]) {
            const mesh = fourBoneMesh(random, true, (m) => m.position.set(far.at[0], far.at[1], far.at[2]));
            const weights = mesh.geometry.attributes.skinWeight as BufferAttribute;
            for (let i = 0; i < weights.count; i++) weights.setXYZW(i, 1, 0, 0, 0);
            const pass = new BvhPosePass(renderer, mesh);
            for (let frame = 0; frame < 20; frame++) {
                const bones = mesh.skeleton.bones;
                bones[0].position.set(far.root[0] + frame * 0.37, far.root[1] - frame * 0.11, far.root[2] + frame * 0.23);
                for (const bone of bones.slice(1)) bone.rotation.set(frame % 2 ? (random() - 0.5) * 0.4 : 0, 0, 0);
                // Morphs on every third frame (a morphed box is looser: the rest's and the targets' extremes are different vertices).
                mesh.morphTargetInfluences![0] = frame % 3 === 2 ? random() : 0;
                mesh.morphTargetInfluences![1] = frame % 3 === 2 ? random() - 0.5 : 0;
                mesh.updateMatrixWorld(true);
                pass.update();
                const box = pass.bounds();
                expect(contains(box, kernelF32(mesh)).slice(0, 3)).toEqual([]);
                expect(contains(box, cpuPose(mesh)).slice(0, 3)).toEqual([]);
                // Still tight: the room is a few ulps, not a fraction of the distance.
                expect(box[3] - box[0]).toBeLessThan(10);
            }
            pass.dispose();
        }
    });

    it("bounds weights across 3-4 bones, morphs with skinning, relative and absolute targets, a bind matrix that isn't the identity", () => {
        const { renderer } = fakeRenderer();
        for (const relative of [true, false]) {
            const random = createRandom(relative ? 51 : 52);
            const mesh = fourBoneMesh(random, relative, (m) => {
                m.position.set(3, -2, 5);
                m.rotation.set(0.4, -0.9, 0.3);
                m.scale.set(1.5, 0.7, 1.2);
            });
            expect(mesh.bindMatrix.equals(mesh.bindMatrix.clone().identity())).toBe(false);
            const pass = new BvhPosePass(renderer, mesh);
            for (let frame = 0; frame < 30; frame++) {
                animate(mesh, random, frame);
                pass.update();
                const box = pass.bounds();
                const cpu = cpuPose(mesh);
                const kernel = kernelF32(mesh);
                // The emulated kernel is three's math (to f32 rounding), and the box holds both.
                const worst = Math.max(...cpu.map((p, i) => p.distanceTo(kernel[i])));
                expect(worst).toBeLessThan(1e-4);
                expect(contains(box, cpu).slice(0, 3)).toEqual([]);
                expect(contains(box, kernel).slice(0, 3)).toEqual([]);
            }
            pass.dispose();
        }
    });

    it("bounds negative weights that don't sum to 1 on a mesh bound away from the origin with the kernel's formula", () => {
        const { renderer } = fakeRenderer();
        const random = createRandom(61);
        const mesh = fourBoneMesh(random, true, (m) => m.position.set(12, -7, 9));
        const weights = mesh.geometry.attributes.skinWeight as BufferAttribute;
        for (let i = 0; i < weights.count; i++) weights.setXYZW(i, 0.9, 0.35, -0.3, i % 3 === 0 ? -0.15 : 0.1); // sums 0.8 or 1.05
        const pass = new BvhPosePass(renderer, mesh);
        for (let frame = 0; frame < 10; frame++) {
            animate(mesh, random, frame);
            pass.update();
            const box = pass.bounds();
            const kernel = kernelF32(mesh);
            expect(contains(box, kernel).slice(0, 3)).toEqual([]);
            // Tight around the GPU's pose (three's getVertexPosition counts bindMatrixInverse's translation once, the GPU per unit of weight sum).
            const exact = new Float64Array([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
            for (const p of kernel) p.toArray().forEach((c, k) => ((exact[k] = Math.min(exact[k], c)), (exact[k + 3] = Math.max(exact[k + 3], c))));
            for (let k = 0; k < 6; k++) expect(Math.abs(box[k] - exact[k])).toBeLessThan(0.01);
        }
    });

    it('disposes every storage attribute, its output and the compute node', () => {
        const { renderer } = fakeRenderer();
        const mesh = skinnedTube(createRandom(71), 3);
        const disposedAttributes = new Set<unknown>();
        const spy = vi.spyOn(BufferAttribute.prototype, 'dispose').mockImplementation(function (this: BufferAttribute) {
            disposedAttributes.add(this);
        });
        try {
            const pass = new BvhPosePass(renderer, mesh);
            const nodeDispose = vi.spyOn(pass.node, 'dispose');
            pass.dispose();
            pass.dispose();
            // rest, targets, skin index, skin weight, matrices, influences, and the output stand-in.
            expect(disposedAttributes.size).toBe(7);
            expect(nodeDispose).toHaveBeenCalledTimes(1);
        } finally {
            spy.mockRestore();
        }
    });

    it('is keyed on the skeleton: a replaced skeleton or a new bone count makes update and bounds throw', () => {
        const { renderer } = fakeRenderer();
        const random = createRandom(81);
        const mesh = skinnedTube(random, 3);
        const pass = new BvhPosePass(renderer, mesh);
        expect(pass.current()).toBe(true);
        expect(pass.variant).toBe('skin+morph');
        mesh.skeleton = new Skeleton(mesh.skeleton.bones.slice());
        expect(pass.current()).toBe(false);
        expect(() => pass.update()).toThrow(/skeleton .* was replaced/);
        expect(() => pass.bounds()).toThrow(/skeleton .* was replaced/);
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

    it('compiles each pose variant once when its first mesh is added, and dispatches nothing until update()', () => {
        const { renderer } = fakeRenderer();
        const random = createRandom(23);
        const root = new Group();
        const a = skinnedTube(random, 4);
        const b = skinnedTube(random, 3);
        const morphed = new Mesh(new BoxGeometry(1, 1, 1, 2, 2, 2), new MeshBasicMaterial());
        morphed.geometry.morphAttributes.position = [morphed.geometry.attributes.position.clone()];
        morphed.updateMorphTargets();
        root.add(a, b, morphed);
        root.updateMatrixWorld(true);
        const sync = new SceneTreesObject3DSync(new PosingTrees(), root, { renderer });
        expect(renderer.compiles).toBe(2); // skin+morph, morph
        expect(renderer.computes).toBe(0);
        sync.update();
        expect(renderer.computes).toBe(3);
        expect(renderer.compiles).toBe(2);
        sync.dispose();
    });

    it('makes a new pass when the skeleton is replaced, and refuses a pose box that is not finite', () => {
        const { renderer } = fakeRenderer();
        const random = createRandom(25);
        const mesh = skinnedTube(random, 3);
        const passes: BvhPosePass[] = [];
        const trees = new PosingTrees();
        const sync = new SceneTreesObject3DSync(trees, mesh, {
            renderer,
            createPosePass: (object) => {
                const pass = new BvhPosePass(renderer, object);
                passes.push(pass);
                return pass;
            },
        });
        sync.update();
        expect(passes).toHaveLength(1);
        const handle = [...trees.meshes.keys()][0];
        // A new skeleton with one more bone (as a retargeted rig): the old pass would index past its matrices.
        const extra = new Bone();
        mesh.skeleton.bones[2].add(extra);
        mesh.bind(new Skeleton([...mesh.skeleton.bones, extra]));
        animate(mesh, random, 4);
        sync.update();
        expect(passes).toHaveLength(2);
        expect(sync.posePassOf(mesh)).toBe(passes[1]);
        expect(trees.poses.get(handle)!.positionBuffer).toBe(passes[1].buffer);
        expect(contains(Float32Array.from(trees.bounds.get(handle)!), cpuPose(mesh))).toEqual([]);
        expect(() => passes[0].update()).toThrow(/disposed/);
        sync.dispose();

        // A pass whose box turns NaN (a NaN bone matrix, say): an error, not a stale box kept.
        const real = new BvhPosePass(renderer, skinnedTube(random, 3));
        let broken = false;
        const nanSync = new SceneTreesObject3DSync(new PosingTrees(), real.mesh, {
            renderer,
            createPosePass: () => ({
                buffer: real.buffer,
                vertexCount: real.vertexCount,
                update: () => real.update(),
                bounds: (target: Float32Array = new Float32Array(6)): Float32Array => (broken ? target.fill(NaN) : real.bounds(target)),
                dispose: () => real.dispose(),
            }),
        });
        nanSync.update();
        broken = true;
        expect(() => nanSync.update()).toThrow(/pose box .* is not finite/);
    });

    it('refuses an InstancedMesh with per-instance morphs (morphTexture) and more than one instance, warning once', () => {
        const { renderer } = fakeRenderer();
        const geometry = new BoxGeometry(1, 1, 1, 2, 2, 2);
        geometry.morphAttributes.position = [geometry.attributes.position.clone()];
        const many = new InstancedMesh(geometry, new MeshBasicMaterial(), 3);
        const one = new InstancedMesh(geometry, new MeshBasicMaterial(), 1);
        const plain = new InstancedMesh(geometry, new MeshBasicMaterial(), 3);
        for (const mesh of [many, one, plain]) mesh.morphTargetInfluences = [0.5]; // InstancedMesh.updateMorphTargets does nothing
        const morph = new Mesh();
        morph.morphTargetInfluences = [0.5];
        for (let i = 0; i < 3; i++) many.setMorphAt(i, morph);
        one.setMorphAt(0, morph);
        const root = new Group();
        root.add(many, one, plain);
        root.updateMatrixWorld(true);
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
        try {
            const trees = new PosingTrees();
            const sync = new SceneTreesObject3DSync(trees, root, { renderer });
            expect(sync.isDeforming(many)).toBe(false);
            expect(sync.isDeforming(one)).toBe(true);
            expect(sync.isDeforming(plain)).toBe(true);
            sync.sync();
            sync.update();
            expect(warn).toHaveBeenCalledTimes(1);
            expect(String(warn.mock.calls[0][0])).toMatch(/morphTexture.*rest pose/);
            expect(sync.posePassOf(many)).toBeNull();
            // many's 3 instances share one rest-pose mesh; one and plain get a posed mesh each.
            expect([...trees.meshes.values()].filter((mesh) => mesh.options?.deforming)).toHaveLength(2);
            sync.dispose();
        } finally {
            warn.mockRestore();
        }
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

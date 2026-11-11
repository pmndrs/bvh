/**
 * @pmndrs/bvh/three scenes in Node: three objects to buildTraceScene inputs.
 * The oracle is three itself: the CPU walk over the converted scene must find
 * the hits three's own Raycaster finds (same object, instance, face and
 * distance), on random rays through a scene with every kind of mesh the
 * converter reads. Then the SceneTrees sync, against a recording stand-in.
 */

import {
    BatchedMesh,
    BoxGeometry,
    BufferAttribute,
    BufferGeometry,
    DoubleSide,
    Group,
    InstancedMesh,
    InterleavedBuffer,
    InterleavedBufferAttribute,
    Matrix4,
    Mesh,
    MeshBasicMaterial,
    Object3D,
    Quaternion,
    Raycaster,
    SphereGeometry,
    TorusKnotGeometry,
    Vector3,
    type Intersection,
} from 'three';
import { describe, expect, it } from 'vitest';
import { createTinyBvhBuilder } from '../tinybvh.ts';
import { createCpuTracer } from '../trace/cpu-traversal.ts';
import { buildMeshBvhWithTinyBvh } from '../trace/mesh-bvh-topology.ts';
import type { MeshGeometry } from '../trace/mesh-bvh.ts';
import { createRandom } from '../trace/oracle-fixtures.ts';
import { buildTraceScene } from '../trace/scene-assembly.ts';
import type { Mat4 } from '../trace/top-level.ts';
import type { SceneInstanceChange, SceneInstanceDescriptor } from '../scene/scene-trees.ts';
import { bvhSceneFromObject3D, effectivelyVisible, rayMaskFromLayers, threeFaceIndex, threeTriangleRanges, type BvhThreeScene } from './scene.ts';
import { SceneTreesObject3DSync, type SceneTreesTarget } from './scene-trees-sync.ts';

const tinyBvh = await createTinyBvhBuilder();
const build = (geometry: MeshGeometry) => buildMeshBvhWithTinyBvh(geometry, tinyBvh);
const material = () => new MeshBasicMaterial({ side: DoubleSide });

function place(object: Object3D, random: () => number, extent = 6): Object3D {
    object.position.set((random() - 0.5) * extent, (random() - 0.5) * extent, (random() - 0.5) * extent);
    object.quaternion.setFromAxisAngle(new Vector3(random(), random(), random()).normalize(), random() * 6);
    const s = 0.5 + random();
    object.scale.set(s, random() < 0.2 ? -s : s * (0.7 + random() * 0.6), s);
    return object;
}

/** Every kind of mesh the converter reads, in a hierarchy, with hidden and off-layer ones. */
function testScene(seed: number): Group {
    const random = createRandom(seed);
    const root = new Group();
    const parent = place(new Group(), random, 2);
    root.add(parent);
    // Indexed, shared by two meshes.
    const knot = new TorusKnotGeometry(0.8, 0.25, 40, 6);
    parent.add(place(new Mesh(knot, material()), random), place(new Mesh(knot, material()), random));
    // Non-indexed.
    root.add(place(new Mesh(new SphereGeometry(0.7, 10, 8).toNonIndexed(), material()), random));
    // A draw range (indexed, start not a multiple of 3 triangles' worth of groups).
    const ranged = new SphereGeometry(0.9, 12, 10);
    ranged.setDrawRange(30, 300);
    root.add(place(new Mesh(ranged, material()), random));
    // Groups with an array of materials.
    const box = new BoxGeometry(1.2, 0.8, 1, 2, 2, 2);
    const materials = [material(), material(), material(), material(), material(), material()];
    root.add(place(new Mesh(box, materials), random));
    // Interleaved positions.
    const interleaved = new BufferGeometry();
    const source = new SphereGeometry(0.6, 8, 6);
    const data = new Float32Array(source.attributes.position.count * 4);
    for (let i = 0; i < source.attributes.position.count; i++) data.set([source.attributes.position.getX(i), source.attributes.position.getY(i), source.attributes.position.getZ(i), 0], i * 4);
    interleaved.setAttribute('position', new InterleavedBufferAttribute(new InterleavedBuffer(data, 4), 3, 0));
    interleaved.setIndex(new BufferAttribute(Uint16Array.from(source.index!.array), 1));
    root.add(place(new Mesh(interleaved, material()), random));
    // InstancedMesh, count below its capacity.
    const instanced = new InstancedMesh(new BoxGeometry(0.4, 0.4, 0.4), material(), 12);
    const matrix = new Matrix4();
    for (let i = 0; i < 12; i++) {
        const dummy = place(new Object3D(), random, 8);
        dummy.updateMatrix();
        instanced.setMatrixAt(i, dummy.matrix);
    }
    instanced.count = 9;
    root.add(place(instanced, random, 1));
    // BatchedMesh: two geometries, four instances, one hidden.
    const batched = new BatchedMesh(4, 2000, 6000, material());
    const a = batched.addGeometry(new SphereGeometry(0.5, 8, 6));
    const b = batched.addGeometry(new BoxGeometry(0.6, 0.3, 0.9));
    for (const geometryId of [a, b, b, a]) {
        const id = batched.addInstance(geometryId);
        const dummy = place(new Object3D(), random, 7);
        dummy.updateMatrix();
        batched.setMatrixAt(id, matrix.copy(dummy.matrix));
    }
    batched.setVisibleAt(2, false);
    root.add(place(batched, random, 1));
    // Hidden (through its parent) and on layer 1 only.
    const hiddenParent = new Group();
    hiddenParent.visible = false;
    hiddenParent.add(place(new Mesh(new BoxGeometry(2, 2, 2), material()), random));
    root.add(hiddenParent);
    const offLayer = place(new Mesh(new BoxGeometry(1.5, 1.5, 1.5), material()), random);
    offLayer.layers.set(1);
    root.add(offLayer);
    root.updateMatrixWorld(true);
    return root;
}

/** three's first hit on visible objects (its Raycaster ignores `visible`). */
function threeFirstHit(raycaster: Raycaster, root: Object3D): Intersection | null {
    for (const hit of raycaster.intersectObject(root, true)) if (effectivelyVisible(hit.object)) return hit;
    return null;
}

async function compareWithThree(scene: BvhThreeScene, root: Object3D, seed: number, rays = 3000, merge = false) {
    const trace = await buildTraceScene(scene.meshes, scene.instances, { build, mergeStaticInstances: merge });
    const tracer = createCpuTracer(trace);
    const random = createRandom(seed);
    const raycaster = new Raycaster();
    const mask = rayMaskFromLayers(raycaster.layers);
    let hits = 0;
    let ties = 0;
    const problems: string[] = [];
    for (let r = 0; r < rays; r++) {
        const origin = new Vector3((random() - 0.5) * 16, (random() - 0.5) * 16, (random() - 0.5) * 16);
        const target = new Vector3((random() - 0.5) * 6, (random() - 0.5) * 6, (random() - 0.5) * 6);
        const direction = target.sub(origin).normalize();
        raycaster.set(origin, direction);
        const expected = threeFirstHit(raycaster, root);
        const ours = tracer.closest({ origin: origin.toArray(), direction: direction.toArray(), tMin: 0, tMax: 1e30 }, { mask });
        if (!expected || !ours) {
            if (expected || ours) problems.push(`ray ${r}: three ${expected ? expected.distance : 'miss'}, ours ${ours ? ours.t : 'miss'}`);
            continue;
        }
        hits++;
        const source = scene.sources[ours.instanceIndex];
        const faceIndex = threeFaceIndex(scene.meshSources[source.mesh], ours.triangleIndex);
        const expectedInstance = expected.instanceId ?? expected.batchId ?? -1;
        const same = source.object === expected.object && source.instanceId === expectedInstance && faceIndex === expected.faceIndex;
        const closeT = Math.abs(ours.t - expected.distance) <= 1e-4 * Math.max(1, expected.distance);
        if (!closeT) problems.push(`ray ${r}: t ${ours.t} vs three ${expected.distance}`);
        else if (!same) ties++; // the same distance on another triangle: an edge or a coincident face
    }
    return { hits, ties, problems };
}

describe('bvhSceneFromObject3D', () => {
    it("finds three's Raycaster hits: indexed, non-indexed, draw ranges, groups, interleaved, InstancedMesh, BatchedMesh, hidden, layers", async () => {
        const root = testScene(7);
        const scene = bvhSceneFromObject3D(root);
        // Two knots share one mesh; the hidden box and the off-layer box are instances too.
        const { hits, ties, problems } = await compareWithThree(scene, root, 11);
        expect(problems).toEqual([]);
        expect(hits).toBeGreaterThan(500);
        expect(ties).toBeLessThan(hits * 0.01);
    });

    it('gives the same hits with every instance static and merged', async () => {
        const root = testScene(8);
        const scene = bvhSceneFromObject3D(root, { static: true });
        expect(scene.instances.every((instance) => instance.static)).toBe(true);
        const { hits, ties, problems } = await compareWithThree(scene, root, 12, 1500, true);
        expect(problems).toEqual([]);
        expect(hits).toBeGreaterThan(200);
        expect(ties).toBeLessThan(hits * 0.01);
    });

    it('shares one mesh per geometry and range, and keeps instance order', () => {
        const root = testScene(9);
        const scene = bvhSceneFromObject3D(root);
        const knots = scene.sources.filter((source) => (source.object.geometry as TorusKnotGeometry).type === 'TorusKnotGeometry');
        expect(knots).toHaveLength(2);
        expect(knots[0].mesh).toBe(knots[1].mesh);
        scene.instances.forEach((instance, index) => {
            expect(instance.instanceIndex).toBe(index);
            expect(instance.meshIndex).toBe(scene.sources[index].mesh);
        });
        // 9 of the InstancedMesh's 12, 4 batched instances (one hidden), the box's 6 groups.
        expect(scene.sources.filter((source) => (source.object as InstancedMesh).isInstancedMesh)).toHaveLength(9);
        expect(scene.sources.filter((source) => (source.object as BatchedMesh).isBatchedMesh)).toHaveLength(4);
        expect(scene.sources.filter((source) => source.group >= 0).map((source) => source.materialIndex)).toEqual([0, 1, 2, 3, 4, 5]);
    });

    it('maps visibility, layers and materials', () => {
        const root = new Group();
        const geometry = new BoxGeometry();
        const shown = new Mesh(geometry, material());
        const hidden = new Mesh(geometry, material());
        hidden.visible = false;
        const layered = new Mesh(geometry, material());
        layered.layers.enable(3);
        layered.layers.enable(9);
        const high = new Mesh(geometry, material());
        high.layers.set(12);
        root.add(shown, hidden, layered, high);
        root.updateMatrixWorld(true);
        const scene = bvhSceneFromObject3D(root, { material: (mesh) => root.children.indexOf(mesh) * 10 });
        expect(scene.instances.map((instance) => instance.visible)).toEqual([true, false, true, true]);
        expect(scene.instances.map((instance) => instance.mask)).toEqual([0x01, 0x01, 0x09, 0x00]);
        expect(scene.instances.map((instance) => instance.material)).toEqual([0, 10, 20, 30]);
        expect(bvhSceneFromObject3D(root, { layers: false }).instances.map((instance) => instance.mask)).toEqual([0xff, 0xff, 0xff, 0xff]);
        expect(() => bvhSceneFromObject3D(root, { material: () => 70000 })).toThrow(RangeError);
    });

    it("reads the triangle ranges three's raycast reads", () => {
        const geometry = new BoxGeometry(1, 1, 1, 1, 1, 1); // 6 groups of 6 indices
        expect(threeTriangleRanges(geometry, material()).map((r) => [r.start, r.count])).toEqual([[0, 36]]);
        geometry.setDrawRange(4, 20);
        expect(threeTriangleRanges(geometry, material()).map((r) => [r.start, r.count])).toEqual([[4, 18]]);
        const materials = [material(), material(), material()];
        // Groups 0-5 use materials 0-5; only 0-2 exist, clipped to [4, 24).
        expect(threeTriangleRanges(geometry, materials).map((r) => [r.start, r.count, r.group])).toEqual([
            [6, 6, 1],
            [12, 6, 2],
        ]);
        const noGroups = new BufferGeometry().setAttribute('position', new BufferAttribute(new Float32Array(9), 3));
        expect(threeTriangleRanges(noGroups, materials)).toEqual([]);
    });
});

/** Records what the sync asks of SceneTrees. */
class RecordingTrees implements SceneTreesTarget {
    readonly log: string[] = [];
    readonly meshes = new Map<number, MeshGeometry>();
    readonly instances = new Map<number, SceneInstanceDescriptor>();
    private next = 0;
    addMesh(geometry: MeshGeometry): number {
        const handle = this.next++;
        this.meshes.set(handle, geometry);
        this.log.push(`addMesh ${handle}`);
        return handle;
    }
    removeMesh(mesh: number): void {
        this.meshes.delete(mesh);
        this.log.push(`removeMesh ${mesh}`);
    }
    addInstance(descriptor: SceneInstanceDescriptor): number {
        const handle = this.next++;
        this.instances.set(handle, descriptor);
        this.log.push(`addInstance ${handle}`);
        return handle;
    }
    setTransform(instance: number, objectToWorld: Mat4): void {
        this.instances.set(instance, { ...this.instances.get(instance)!, objectToWorld: Array.from(objectToWorld) });
        this.log.push(`setTransform ${instance}`);
    }
    setInstance(instance: number, change: SceneInstanceChange): void {
        this.log.push(`setInstance ${instance} ${JSON.stringify(change)}`);
    }
    removeInstance(instance: number): void {
        this.instances.delete(instance);
        this.log.push(`removeInstance ${instance}`);
    }
}

describe('SceneTreesObject3DSync', () => {
    it('adds the scene, pushes only moved and changed non-static instances, and follows adds and removals', () => {
        const root = new Group();
        const geometry = new BoxGeometry();
        const moving = new Mesh(geometry, material());
        const still = new Mesh(geometry, material());
        const instanced = new InstancedMesh(new SphereGeometry(0.5, 6, 4), material(), 3);
        root.add(moving, still, instanced);
        root.updateMatrixWorld(true);
        const trees = new RecordingTrees();
        const sync = new SceneTreesObject3DSync(trees, root, { static: (object) => object === still });
        expect(trees.meshes.size).toBe(2);
        expect(trees.instances.size).toBe(5);
        expect(sync.sourceOf([...trees.instances.keys()][0])?.object).toBe(moving);

        trees.log.length = 0;
        expect(sync.update()).toMatchObject({ moved: 0, changed: 0 });
        expect(trees.log).toEqual([]);

        moving.position.x = 3;
        still.position.x = 3; // static: never pushed
        instanced.setMatrixAt(1, new Matrix4().compose(new Vector3(0, 2, 0), new Quaternion(), new Vector3(1, 1, 1)));
        instanced.instanceMatrix.needsUpdate = true;
        moving.layers.enable(2);
        root.updateMatrixWorld(true);
        expect(sync.update()).toMatchObject({ moved: 2, changed: 1 });
        expect(trees.log.filter((line) => line.startsWith('setTransform'))).toHaveLength(2);
        expect(trees.log).toContain(`setInstance ${[...trees.instances.keys()][0]} {"mask":5}`);
        const movedDescriptor = trees.instances.get([...trees.instances.keys()][0])!;
        expect(movedDescriptor.objectToWorld[12]).toBe(3);

        trees.log.length = 0;
        moving.visible = false;
        expect(sync.update()).toMatchObject({ moved: 0, changed: 1 });
        expect(trees.log).toEqual([`setInstance ${[...trees.instances.keys()][0]} {"visible":false}`]);

        // A new mesh, a removed one, and the InstancedMesh's count lowered.
        trees.log.length = 0;
        root.remove(still);
        root.add(new Mesh(new SphereGeometry(1, 5, 4), material()));
        instanced.count = 2;
        root.updateMatrixWorld(true);
        expect(sync.sync()).toMatchObject({ added: 1, removed: 2, meshesAdded: 1, meshesRemoved: 0 });
        expect(trees.instances.size).toBe(4);
        root.remove(moving);
        expect(sync.sync()).toMatchObject({ removed: 1, meshesRemoved: 1 });
        expect(trees.meshes.size).toBe(2);

        sync.dispose();
        expect(trees.instances.size).toBe(0);
        expect(trees.meshes.size).toBe(0);
    });
});

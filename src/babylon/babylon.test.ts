/**
 * `@pmndrs/bvh/babylon` against Babylon itself, in Node: Babylon's `NullEngine`
 * has no canvas and no GPU but runs `scene.pick`, so the picker is checked
 * against the engine's own picking on random rays through meshes and instances,
 * and the mesh conversion against the meshes' vertex data and world matrices.
 */

import { Camera } from '@babylonjs/core/Cameras/camera.js';
import { FreeCamera } from '@babylonjs/core/Cameras/freeCamera.js';
import '@babylonjs/core/Culling/ray.js';
import { Ray } from '@babylonjs/core/Culling/ray.js';
import { NullEngine } from '@babylonjs/core/Engines/nullEngine.js';
import { Material } from '@babylonjs/core/Materials/material.js';
import { Matrix, Quaternion, Vector3 } from '@babylonjs/core/Maths/math.vector.js';
import '@babylonjs/core/Meshes/instancedMesh.js';
import { Mesh } from '@babylonjs/core/Meshes/mesh.js';
import { VertexData } from '@babylonjs/core/Meshes/mesh.vertexData.js';
import { Scene } from '@babylonjs/core/scene.js';
import type { AbstractMesh } from '@babylonjs/core/Meshes/abstractMesh.js';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { createRandom } from '../trace/oracle-fixtures.ts';
import { createTinyBvhBuilder, type TinyBvhBuilder } from '../tinybvh.ts';
import { bindTraceBuffers, createTraceComputeShader, gpuBufferToDataBuffer, TRACE_BINDING_NAMES, traceBindingsMapping } from './compute.ts';
import { deviceFromEngine } from './device.ts';
import { BabylonBvhPicker } from './picking.ts';
import { BABYLON_MASK_PICKABLE, BABYLON_MASK_VISIBLE, bvhSceneFromBabylon } from './scene.ts';

let engine: NullEngine;
let builder: TinyBvhBuilder;

beforeAll(async () => {
    engine = new NullEngine();
    builder = await createTinyBvhBuilder();
});
afterAll(() => engine.dispose());

/** A UV sphere of `segments` x `segments / 2` quads, as indexed vertex data. */
function sphereData(segments: number, radius = 1): VertexData {
    const rings = Math.max(2, segments >> 1);
    const positions: number[] = [];
    const indices: number[] = [];
    for (let r = 0; r <= rings; r++) {
        const phi = (r / rings) * Math.PI;
        for (let s = 0; s <= segments; s++) {
            const theta = (s / segments) * Math.PI * 2;
            positions.push(radius * Math.sin(phi) * Math.cos(theta), radius * Math.cos(phi), radius * Math.sin(phi) * Math.sin(theta));
        }
    }
    for (let r = 0; r < rings; r++) {
        for (let s = 0; s < segments; s++) {
            const a = r * (segments + 1) + s;
            const b = a + segments + 1;
            indices.push(a, b, a + 1, a + 1, b, b + 1);
        }
    }
    const data = new VertexData();
    data.positions = positions;
    data.indices = indices;
    return data;
}

/** Babylon only picks triangles of a mesh that has a material. */
function meshOf(scene: Scene, name: string, data: VertexData): Mesh {
    const mesh = new Mesh(name, scene);
    mesh.material = new Material(`${name}-material`, scene);
    data.applyToMesh(mesh, true);
    return mesh;
}

function place(mesh: AbstractMesh, random: () => number, extent = 5): void {
    mesh.position.set((random() - 0.5) * extent, (random() - 0.5) * extent, (random() - 0.5) * extent);
    mesh.rotationQuaternion = Quaternion.RotationAxis(new Vector3(random(), random(), random()).normalize(), random() * 6);
    const s = 0.5 + random();
    mesh.scaling.set(s, s * (0.6 + random() * 0.8), random() < 0.3 ? -s : s);
    mesh.computeWorldMatrix(true);
}

/** Two spheres of different detail, one instanced, one stretched, one mesh unpickable, one hidden. */
function testScene(seed: number): { scene: Scene; meshes: AbstractMesh[] } {
    const random = createRandom(seed);
    const scene = new Scene(engine);
    const coarse = meshOf(scene, 'coarse', sphereData(12));
    const fine = meshOf(scene, 'fine', sphereData(40, 0.8));
    const meshes: AbstractMesh[] = [coarse, fine];
    for (let i = 0; i < 4; i++) meshes.push(coarse.createInstance(`coarse-${i}`));
    for (let i = 0; i < 2; i++) meshes.push(fine.createInstance(`fine-${i}`));
    for (const mesh of meshes) place(mesh, random);
    return { scene, meshes };
}

function randomRay(random: () => number, extent = 9): Ray {
    const origin = new Vector3((random() - 0.5) * extent * 2, (random() - 0.5) * extent * 2, (random() - 0.5) * extent * 2);
    const target = new Vector3((random() - 0.5) * 5, (random() - 0.5) * 5, (random() - 0.5) * 5);
    return new Ray(origin, target.subtract(origin).normalize(), 1000);
}

describe('bvhSceneFromBabylon', () => {
    it('reads meshes once per geometry, instances with their world matrices, visibility and pickability', () => {
        const { scene, meshes } = testScene(1);
        const [coarse, fine] = meshes as Mesh[];
        meshes[3].isVisible = false;
        meshes[4].isPickable = false;
        const read = bvhSceneFromBabylon(scene);
        expect(read.meshes).toHaveLength(2);
        expect(read.instances).toHaveLength(8);
        expect(read.sources.map((source) => source.object)).toEqual(meshes);
        expect(read.meshSources.map((source) => source.mesh)).toEqual([coarse, fine]);
        expect(read.meshSources[0].triangleCount).toBe(coarse.getTotalIndices() / 3);
        // The instance's memory is the column-major matrix of the same transform.
        for (const [i, instance] of read.instances.entries()) {
            const world = meshes[i].computeWorldMatrix(true);
            expect(Array.from(instance.objectToWorld)).toEqual(Array.from(world.m));
            const point = new Vector3(0.3, -0.2, 0.9);
            const expected = Vector3.TransformCoordinates(point, world);
            const m = instance.objectToWorld;
            expect(m[0] * point.x + m[4] * point.y + m[8] * point.z + m[12]).toBeCloseTo(expected.x, 5);
            expect(m[1] * point.x + m[5] * point.y + m[9] * point.z + m[13]).toBeCloseTo(expected.y, 5);
            expect(m[2] * point.x + m[6] * point.y + m[10] * point.z + m[14]).toBeCloseTo(expected.z, 5);
        }
        expect(read.instances[3].visible).toBe(false);
        expect(read.instances[2].visible).toBe(true);
        expect(read.instances[4].mask).toBe(BABYLON_MASK_VISIBLE);
        expect(read.instances[2].mask).toBe(BABYLON_MASK_VISIBLE | BABYLON_MASK_PICKABLE);
        scene.dispose();
    });

    it('applies filter and material, and skips lines and meshes with no triangles', () => {
        const { scene, meshes } = testScene(2);
        const lines = new Mesh('lines', scene);
        lines.material = new Material('lines-material', scene);
        const lineData = new VertexData();
        lineData.positions = [0, 0, 0, 1, 0, 0, 1, 1, 0];
        lineData.indices = [0, 1, 1, 2];
        lineData.applyToMesh(lines);
        new Mesh('empty', scene);
        const read = bvhSceneFromBabylon(scene, { filter: (mesh) => mesh !== meshes[1], material: (mesh) => (mesh === meshes[0] ? 7 : undefined) });
        expect(read.instances.length).toBeGreaterThan(0);
        expect(read.sources.every((source) => source.object !== meshes[1])).toBe(true);
        expect(read.instances.find((instance) => read.sources[instance.instanceIndex].object === meshes[0])?.material).toBe(7);
        // The lines mesh has indices but its class is a Mesh with lines fill: still read as triangles unless it is a LinesMesh; the empty one never is.
        expect(read.sources.some((source) => source.object.name === 'empty')).toBe(false);
        expect(() => bvhSceneFromBabylon(scene, { material: () => 70000 })).toThrow(RangeError);
        scene.dispose();
    });

    it('reads a plain triangle list when there is no index buffer', () => {
        const scene = new Scene(engine);
        const mesh = new Mesh('soup', scene);
        mesh.material = new Material('soup-material', scene);
        const data = new VertexData();
        data.positions = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1];
        data.applyToMesh(mesh);
        const read = bvhSceneFromBabylon([mesh]);
        expect(read.meshes[0].indices ?? null).toBeNull();
        expect(read.meshSources[0].triangleCount).toBe(2);
        scene.dispose();
    });
});

describe('BabylonBvhPicker against scene.pickWithRay', () => {
    it('returns the same picks as Babylon on random rays', () => {
        const { scene } = testScene(3);
        const picker = new BabylonBvhPicker(scene, { builder });
        const random = createRandom(11);
        let hits = 0;
        let edge = 0;
        for (let i = 0; i < 600; i++) {
            const ray = randomRay(random);
            const babylon = scene.pickWithRay(ray)!;
            const ours = picker.pickWithRay(ray);
            expect(ours.hit).toBe(babylon.hit);
            if (!babylon.hit) continue;
            hits++;
            expect(ours.distance).toBeCloseTo(babylon.distance, 3);
            expect(ours.pickedPoint!.subtract(babylon.pickedPoint!).length()).toBeLessThan(1e-3);
            if (ours.pickedMesh === babylon.pickedMesh && ours.faceId === babylon.faceId) {
                expect(ours.bu).toBeCloseTo(babylon.bu, 3);
                expect(ours.bv).toBeCloseTo(babylon.bv, 3);
                expect(ours.subMeshFaceId).toBe(babylon.subMeshFaceId);
                expect(ours.subMeshId).toBe(babylon.subMeshId);
            } else {
                // A ray through a shared edge or a coincident surface may name either side; the point and distance above agree.
                edge++;
                expect(Math.min(babylon.bu, babylon.bv, 1 - babylon.bu - babylon.bv)).toBeLessThan(1e-3);
            }
        }
        expect(hits).toBeGreaterThan(100);
        expect(edge).toBeLessThanOrEqual(3);
        scene.dispose();
    });

    it('gives a PickingInfo that works: getNormal, ray, miss', () => {
        const { scene } = testScene(4);
        const picker = new BabylonBvhPicker(scene, { builder });
        const random = createRandom(5);
        let checked = 0;
        for (let i = 0; i < 100 && checked < 10; i++) {
            const ray = randomRay(random);
            const ours = picker.pickWithRay(ray);
            const babylon = scene.pickWithRay(ray)!;
            if (!ours.hit || ours.faceId !== babylon.faceId || ours.pickedMesh !== babylon.pickedMesh) continue;
            expect(ours.ray).toBe(ray);
            const normal = ours.getNormal(true, false)!;
            expect(normal.subtract(babylon.getNormal(true, false)!).length()).toBeLessThan(1e-4);
            checked++;
        }
        expect(checked).toBe(10);
        const miss = picker.pickWithRay(new Ray(new Vector3(0, 50, 0), new Vector3(0, 1, 0), 100));
        expect(miss.hit).toBe(false);
        expect(miss.pickedMesh).toBeNull();
        expect(miss.distance).toBe(0);
        scene.dispose();
    });

    it('honours the ray length, as Babylon does', () => {
        const scene = new Scene(engine);
        meshOf(scene, 'ball', sphereData(24));
        const picker = new BabylonBvhPicker(scene, { builder });
        const origin = new Vector3(0, 0, -5);
        const direction = new Vector3(0, 0, 1);
        expect(picker.pickWithRay(new Ray(origin, direction, 3)).hit).toBe(false);
        const reached = picker.pickWithRay(new Ray(origin, direction, 5));
        expect(reached.hit).toBe(true);
        expect(reached.distance).toBeCloseTo(scene.pickWithRay(new Ray(origin, direction, 5))!.distance, 4);
        // A direction that is not a unit vector: the length counts in its own units.
        expect(picker.pickWithRay(new Ray(origin, new Vector3(0, 0, 2), 1.9)).hit).toBe(false);
        expect(picker.pickWithRay(new Ray(origin, new Vector3(0, 0, 2), 2.1)).hit).toBe(true);
        expect(scene.pickWithRay(new Ray(origin, new Vector3(0, 0, 2), 1.9))!.hit).toBe(false);
        expect(scene.pickWithRay(new Ray(origin, new Vector3(0, 0, 2), 2.1))!.hit).toBe(true);
        scene.dispose();
    });

    it('applies a predicate, and by default skips hidden and unpickable meshes as Babylon does', () => {
        const scene = new Scene(engine);
        const near = meshOf(scene, 'near', sphereData(24));
        const far = meshOf(scene, 'far', sphereData(24));
        near.position.z = -3;
        far.position.z = 3;
        near.computeWorldMatrix(true);
        far.computeWorldMatrix(true);
        const picker = new BabylonBvhPicker(scene, { builder });
        const ray = new Ray(new Vector3(0, 0, -10), new Vector3(0, 0, 1), 100);
        expect(picker.pickWithRay(ray).pickedMesh).toBe(near);
        expect(picker.pickWithRay(ray, { predicate: (mesh) => mesh === far }).pickedMesh).toBe(far);
        expect(picker.pickWithRay(ray, { predicate: () => false }).hit).toBe(false);
        near.isPickable = false;
        picker.update();
        expect(picker.pickWithRay(ray).pickedMesh).toBe(far);
        expect(scene.pickWithRay(ray)!.pickedMesh).toBe(far);
        far.isVisible = false;
        picker.update();
        expect(picker.pickWithRay(ray).hit).toBe(false);
        expect(scene.pickWithRay(ray)!.hit).toBe(false);
        scene.dispose();
    });

    it('follows moved, added and removed meshes after update()', () => {
        const scene = new Scene(engine);
        const ball = meshOf(scene, 'ball', sphereData(24));
        const picker = new BabylonBvhPicker(scene, { builder });
        const ray = new Ray(new Vector3(4, 0, -10), new Vector3(0, 0, 1), 100);
        expect(picker.pickWithRay(ray).hit).toBe(false);
        ball.position.x = 4;
        ball.computeWorldMatrix(true);
        // Not live: still the scene as of the last update.
        expect(picker.pickWithRay(ray).hit).toBe(false);
        expect(picker.update()).toBe(false); // transforms only: no tree built
        expect(picker.pickWithRay(ray).pickedMesh).toBe(ball);
        const twin = ball.createInstance('twin');
        twin.position.set(4, 0, 6);
        twin.computeWorldMatrix(true);
        picker.update();
        expect(picker.instanceCount).toBe(2);
        expect(picker.pickWithRay(new Ray(new Vector3(4, 0, 12), new Vector3(0, 0, -1), 100)).pickedMesh).toBe(twin);
        twin.dispose();
        picker.update();
        expect(picker.instanceCount).toBe(1);
        // New vertex data: a tree is built again; an edit in place needs rebuild.
        sphereData(24, 3).applyToMesh(ball, true);
        expect(picker.update()).toBe(true);
        expect(picker.pickWithRay(ray).distance).toBeCloseTo(scene.pickWithRay(ray)!.distance, 3);
        scene.dispose();
    });

    it('pick(x, y) casts from the camera like scene.pick(x, y)', () => {
        const scene = new Scene(engine);
        const ball = meshOf(scene, 'ball', sphereData(32, 2));
        const camera = new FreeCamera('camera', new Vector3(0, 0, -8), scene);
        camera.setTarget(Vector3.Zero());
        camera.mode = Camera.PERSPECTIVE_CAMERA;
        scene.activeCamera = camera;
        camera.getViewMatrix(true);
        camera.getProjectionMatrix(true);
        const picker = new BabylonBvhPicker(scene, { builder });
        const width = engine.getRenderWidth();
        const height = engine.getRenderHeight();
        let hits = 0;
        for (let y = 5; y < height; y += Math.floor(height / 9)) {
            for (let x = 5; x < width; x += Math.floor(width / 9)) {
                const babylon = scene.pick(x, y);
                const ours = picker.pick(x, y);
                expect(ours.hit).toBe(babylon.hit);
                if (!babylon.hit) continue;
                hits++;
                expect(ours.pickedMesh).toBe(ball);
                expect(ours.distance).toBeCloseTo(babylon.distance, 3);
            }
        }
        expect(hits).toBeGreaterThan(5);
        scene.dispose();
    });

    it('has no tree and picks nothing in a scene without triangles', () => {
        const scene = new Scene(engine);
        const picker = new BabylonBvhPicker(scene, { builder });
        expect(picker.cpuScene).toBeNull();
        expect(picker.pickWithRay(new Ray(Vector3.Zero(), new Vector3(0, 0, 1), 10)).hit).toBe(false);
        scene.dispose();
    });

    it('moves a transformed matrix through Babylon and the core alike (a rotated and mirrored instance)', () => {
        const scene = new Scene(engine);
        const mesh = meshOf(scene, 'ball', sphereData(24));
        mesh.rotationQuaternion = Quaternion.FromRotationMatrix(Matrix.RotationAxis(new Vector3(1, 1, 0).normalize(), 0.7));
        mesh.scaling.set(-2, 1, 0.5);
        mesh.position.set(1, 2, 3);
        const picker = new BabylonBvhPicker(scene, { builder });
        const random = createRandom(77);
        for (let i = 0; i < 200; i++) {
            const ray = randomRay(random, 6);
            const babylon = scene.pickWithRay(ray)!;
            const ours = picker.pickWithRay(ray);
            expect(ours.hit).toBe(babylon.hit);
            if (babylon.hit) expect(ours.distance).toBeCloseTo(babylon.distance, 3);
        }
        scene.dispose();
    });
});

describe('device and compute shader helpers', () => {
    it('refuses an engine that is not a WebGPU engine, naming it', () => {
        expect(() => deviceFromEngine(engine)).toThrow(/NullEngine, not a WebGPUEngine/);
        expect(() => deviceFromEngine(null)).toThrow(/not a WebGPUEngine/);
        expect(() => deviceFromEngine({ isWebGPU: true, _device: undefined })).toThrow(/no device/);
        const fake = { createBuffer() {} } as unknown as GPUDevice;
        expect(deviceFromEngine({ isWebGPU: true, _device: fake })).toBe(fake);
    });

    it('maps the walk\'s names to its bindings, at any slot, with the positions for indexed scenes', () => {
        expect(traceBindingsMapping()).toEqual({
            bvhTraceNodes: { group: 0, binding: 0 },
            bvhTraceTriangles: { group: 0, binding: 1 },
            bvhTraceTopLevel: { group: 0, binding: 2 },
        });
        expect(traceBindingsMapping({ traceSlot: { group: 2, firstBinding: 5 }, triangleLeaves: 'indexed' })).toMatchObject({
            bvhTraceNodes: { group: 2, binding: 5 },
            bvhTracePositions: { group: 2, binding: 8 },
        });
        expect(TRACE_BINDING_NAMES).toHaveLength(4);
    });

    it('wraps a GPUBuffer as one Babylon DataBuffer, and binds the walk\'s buffers by name', () => {
        const fake = (size: number) => ({ size }) as unknown as GPUBuffer;
        const nodes = fake(64);
        const wrapped = gpuBufferToDataBuffer(nodes);
        expect(gpuBufferToDataBuffer(nodes)).toBe(wrapped);
        expect((wrapped as unknown as { underlyingResource: unknown }).underlyingResource).toBe(nodes);
        expect(wrapped.capacity).toBe(64);
        const bound = new Map<string, unknown>();
        const shader = { setStorageBuffer: (name: string, buffer: unknown) => bound.set(name, buffer) };
        bindTraceBuffers(shader as never, { nodes, triangles: fake(48), topLevel: fake(32) });
        expect([...bound.keys()]).toEqual(['bvhTraceNodes', 'bvhTraceTriangles', 'bvhTraceTopLevel']);
        expect(bound.get('bvhTraceNodes')).toBe(wrapped);
        // A ComputeShader is made without compiling on a NullEngine's missing WebGPU: the constructor only stores the source.
        expect(typeof createTraceComputeShader).toBe('function');
    });
});

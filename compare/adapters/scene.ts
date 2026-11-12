/**
 * Scene-level trees over many objects (compare/scene): ours (`CpuSceneBvh`,
 * the CPU top level over instances of one mesh tree), bvh.js (agargaro, a
 * dynamic box BVH, its objects being the scene's objects) and three-mesh-bvh's
 * `ObjectBVH` (over three `Mesh` objects sharing a geometry with a `MeshBVH`).
 *
 * Every tree boxes an object the same way: the mesh's box with its corners
 * transformed (`worldBox`). bvh.js has no triangle queries, so it does no
 * rays here: it is a box tree over objects, which is what it is for.
 */

import { BVH, HybridBuilder, WebGLCoordinateSystem } from 'bvh.js';
import { CpuSceneBvh, buildMeshBvhWithTinyBvh, type TraceSceneInstance } from '@pmndrs/bvh';
import { BufferAttribute, BufferGeometry, DoubleSide, Mesh, MeshBasicMaterial, Raycaster, Vector3, type Intersection } from 'three';
import { acceleratedRaycast, CONTAINED, INTERSECTED, MeshBVH, NOT_INTERSECTED, ObjectBVH, SAH } from 'three-mesh-bvh';
import { oursTinyBvh } from './ours.ts';
import { reorderedToOriginal, tmbvhBytes } from './three-mesh-bvh.ts';
import type { BuiltScene, CompareScene, SceneAdapter } from './types.ts';
import { worldBox } from '../core/scene.ts';

/** Where a box lies against planes (`ax + by + cz + d >= 0` inside): the test the culling walks share. */
export function classifyBox(planes: ArrayLike<number>, min: { x: number; y: number; z: number }, max: { x: number; y: number; z: number }): 'outside' | 'inside' | 'crossing' {
    let inside = true;
    for (let p = 0; p < planes.length; p += 4) {
        const a = planes[p];
        const b = planes[p + 1];
        const c = planes[p + 2];
        const d = planes[p + 3];
        const far = a * (a > 0 ? max.x : min.x) + b * (b > 0 ? max.y : min.y) + c * (c > 0 ? max.z : min.z) + d;
        if (far < 0) return 'outside';
        const near = a * (a > 0 ? min.x : max.x) + b * (b > 0 ? min.y : max.y) + c * (c > 0 ? min.z : max.z) + d;
        if (near < 0) inside = false;
    }
    return inside ? 'inside' : 'crossing';
}

export const oursSceneAdapter: SceneAdapter = {
    id: 'ours',
    name: '@pmndrs/bvh (CpuSceneBvh)',
    library: '@pmndrs/bvh (this repository)',
    license: 'MIT',
    buildMethod: 'one tinybvh tree for the mesh, CpuSceneBvh over the instances (its CPU top level, SAH)',
    features: {
        build: { status: 'yes' },
        update: { status: 'yes', note: 'CPU: setInstances rebuilds the top level; GPU: SceneTrees refits it on the GPU (the compare/scene page)' },
        frustum: { status: 'yes', note: 'intersectsFrustum' },
        box: { status: 'yes', note: 'intersectsBox' },
        raycast: { status: 'yes', note: 'one walk: top level, then the instance\'s mesh tree' },
        gpu: { status: 'yes', note: 'the same top level in the WGSL walk' },
    },
    async prepare() {
        tinyBvhNow = await oursTinyBvh();
    },
    build(scene) {
        return buildOursScene(scene);
    },
};

let tinyBvhNow: Awaited<ReturnType<typeof oursTinyBvh>> | null = null;

function buildOursScene(scene: CompareScene): BuiltScene {
    const tinyBvh = tinyBvhNow;
    if (!tinyBvh) throw new Error('oursSceneAdapter.prepare() first');
    const tree = buildMeshBvhWithTinyBvh({ positions: scene.mesh.positions, indices: scene.mesh.indices }, tinyBvh);
    const instances = (): TraceSceneInstance[] => scene.objects.map((object) => ({ meshIndex: 0, instanceIndex: object.id, objectToWorld: object.matrix }));
    const bvh = new CpuSceneBvh([tree], instances());
    return {
        updateKind: 'rebuild',
        intersectsFrustum: (planes) => bvh.intersectsFrustum(planes),
        intersectsBox: (box) => bvh.intersectsBox(box),
        raycastFirst(ray) {
            const [hit] = bvh.raycast({ origin: ray.origin, direction: ray.direction, tMin: 0, tMax: Infinity }, { firstHitOnly: true });
            return hit ? { object: hit.instance, triangle: hit.triangle, t: hit.t } : null;
        },
        update: () => bvh.setInstances(instances()),
        memory() {
            const s = bvh.scene;
            return { bytes: s.topLevel.buffer.byteLength + s.nodes.byteLength + s.triangles.byteLength, method: 'byteLength of the top level (instance records and nodes) and the mesh tree (nodes and triangle records)' };
        },
        dispose() {},
    };
}


type BvhJsNode = { box: Float32Array; object?: number };

export const bvhJsSceneAdapter: SceneAdapter = {
    id: 'bvh.js',
    name: 'bvh.js (HybridBuilder)',
    library: 'bvh.js@0.0.13',
    license: 'MIT',
    buildMethod: 'new BVH(new HybridBuilder()), createFromArray(ids, worldBoxes) (its top-down build), margin 0',
    features: {
        build: { status: 'yes' },
        update: { status: 'yes', note: 'move(node) per object: reinserts a leaf whose box left its parent\'s (Catto\'s dynamic BVH with rotations)' },
        frustum: { status: 'yes', note: 'frustumCulling(viewProjection)' },
        box: { status: 'yes', note: 'intersectsBox' },
        raycast: { status: 'no', note: 'rays meet boxes (rayIntersections), not triangles: an app tests the triangles itself' },
        gpu: { status: 'no' },
    },
    build(scene) {
        const bvh = new BVH<object, number>(new HybridBuilder(), WebGLCoordinateSystem);
        const boxes = scene.objects.map((object) => worldBox(scene.mesh, object.matrix) as Float32Array);
        // bvh.js boxes are [minX, maxX, minY, maxY, minZ, maxZ].
        const toBvhJs = (box: ArrayLike<number>, out: Float32Array = new Float32Array(6)) => {
            out[0] = box[0];
            out[1] = box[3];
            out[2] = box[1];
            out[3] = box[4];
            out[4] = box[2];
            out[5] = box[5];
            return out;
        };
        const leaves: BvhJsNode[] = new Array(scene.objects.length);
        bvh.createFromArray(
            scene.objects.map((object) => object.id),
            boxes.map((box) => toBvhJs(box)),
            (node) => (leaves[(node as BvhJsNode).object!] = node as BvhJsNode),
            0,
        );
        const scratch = new Float32Array(6);
        const query = new Float32Array(6);
        return {
            updateKind: 'incremental',
            intersectsFrustum(_planes, viewProjection) {
                const out: number[] = [];
                bvh.frustumCulling(viewProjection, (node) => void out.push((node as BvhJsNode).object!));
                return out;
            },
            intersectsBox(box) {
                const out: number[] = [];
                bvh.intersectsBox(toBvhJs(box, query), (id) => (out.push(id), false));
                return out;
            },
            update(objects) {
                for (const object of objects) {
                    const node = leaves[object.id];
                    toBvhJs(worldBox(scene.mesh, object.matrix, scratch), node.box);
                    bvh.move(node as never, 0);
                }
            },
            memory() {
                return { bytes: null, method: 'JS node objects with a Float32Array box each: the Node runner measures its heap growth' };
            },
            dispose() {},
        };
    },
};

export const tmbvhSceneAdapter: SceneAdapter = {
    id: 'tmbvh',
    name: 'three-mesh-bvh ObjectBVH',
    library: 'three-mesh-bvh@0.9.18',
    license: 'MIT',
    buildMethod: 'a MeshBVH (SAH) on the shared geometry, one three Mesh per object, new ObjectBVH(meshes) (its defaults: leaves of 1, geometry bounding boxes)',
    features: {
        build: { status: 'yes' },
        update: { status: 'yes', note: 'refit() after the objects\' matrixWorld change' },
        frustum: { status: 'partial', note: 'no frustum query: shapecast with a plane test on its boxes' },
        box: { status: 'partial', note: 'no box query: shapecast with a box test on its boxes' },
        raycast: { status: 'yes', note: 'ObjectBVH.raycast with firstHitOnly, each mesh through acceleratedRaycast' },
        gpu: { status: 'yes', note: 'BVHComputeData over the same objects (three-mesh-bvh/webgpu)' },
    },
    build(scene) {
        const geometry = new BufferGeometry();
        geometry.setAttribute('position', new BufferAttribute(scene.mesh.positions, 3));
        geometry.setIndex(new BufferAttribute(scene.mesh.indices.slice(), 1));
        geometry.computeBoundingBox();
        const meshBvh = new MeshBVH(geometry, { strategy: SAH });
        // MeshBVH reordered the index: its face indices map back to the mesh's triangles (not timed).
        const map = reorderedToOriginal(scene.mesh.indices, geometry.index!.array);
        (geometry as unknown as { boundsTree: MeshBVH }).boundsTree = meshBvh;
        const material = new MeshBasicMaterial({ side: DoubleSide });
        const meshes = scene.objects.map((object) => {
            const mesh = new Mesh(geometry, material);
            mesh.raycast = acceleratedRaycast;
            mesh.matrixAutoUpdate = false;
            mesh.matrixWorld.fromArray(object.matrix);
            mesh.userData.id = object.id;
            return mesh;
        });
        const bvh = new ObjectBVH(meshes);
        const raycaster = new Raycaster() as Raycaster & { firstHitOnly?: boolean };
        raycaster.firstHitOnly = true;
        const hits: Intersection[] = [];
        const boxes = scene.objects.map((object) => worldBox(scene.mesh, object.matrix) as Float32Array);
        const min = new Vector3();
        const max = new Vector3();
        const cull = (test: (min: Vector3, max: Vector3) => 'outside' | 'inside' | 'crossing') => {
            const out: number[] = [];
            bvh.shapecast({
                intersectsBounds: (box) => {
                    const where = test(box.min, box.max);
                    return where === 'outside' ? NOT_INTERSECTED : where === 'inside' ? CONTAINED : INTERSECTED;
                },
                intersectsObject: (object, _instance, contained) => {
                    const id = (object as Mesh).userData.id as number;
                    const b = boxes[id];
                    if (contained || test(min.set(b[0], b[1], b[2]), max.set(b[3], b[4], b[5])) !== 'outside') out.push(id);
                },
            });
            return out;
        };
        return {
            updateKind: 'refit',
            intersectsFrustum: (planes) => cull((lo, hi) => classifyBox(planes, lo, hi)),
            intersectsBox(box) {
                return cull((lo, hi) => {
                    if (hi.x < box[0] || lo.x > box[3] || hi.y < box[1] || lo.y > box[4] || hi.z < box[2] || lo.z > box[5]) return 'outside';
                    return lo.x >= box[0] && hi.x <= box[3] && lo.y >= box[1] && hi.y <= box[4] && lo.z >= box[2] && hi.z <= box[5] ? 'inside' : 'crossing';
                });
            },
            raycastFirst(ray) {
                raycaster.ray.origin.set(ray.origin[0], ray.origin[1], ray.origin[2]);
                raycaster.ray.direction.set(ray.direction[0], ray.direction[1], ray.direction[2]);
                hits.length = 0;
                bvh.raycast(raycaster, hits);
                const hit = hits[0];
                return hit ? { object: (hit.object as Mesh).userData.id as number, triangle: map[hit.faceIndex!], t: hit.distance } : null;
            },
            update(objects) {
                for (const object of objects) {
                    meshes[object.id].matrixWorld.fromArray(object.matrix);
                    worldBox(scene.mesh, object.matrix, boxes[object.id]);
                }
                bvh.refit();
            },
            memory() {
                const internal = bvh as unknown as { _roots: ArrayBuffer[]; primitiveBuffer: Uint32Array };
                const bytes = internal._roots.reduce((sum, root) => sum + root.byteLength, 0) + internal.primitiveBuffer.byteLength + tmbvhBytes(meshBvh);
                return { bytes, method: 'byteLength of the ObjectBVH node buffers and primitive buffer, plus the shared MeshBVH\'s nodes; the three Mesh objects are the app\'s' };
            },
            dispose() {
                geometry.dispose();
                material.dispose();
            },
        };
    },
};

export const SCENE_ADAPTERS: readonly SceneAdapter[] = [oursSceneAdapter, bvhJsSceneAdapter, tmbvhSceneAdapter];

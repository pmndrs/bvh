// Code samples for docs/three/*.mdx and docs/engines/*.mdx (see getting-started.ts for how they are checked).

// #region renderer-device
import { WebGPURenderer } from 'three/webgpu';
import { deviceFromRenderer } from '@pmndrs/bvh/three';

// #endregion

export async function rendererDevice(): Promise<GPUDevice> {
    // #region renderer-device
    const adapter = await navigator.gpu.requestAdapter();
    // three asks for every adapter feature (subgroups included), but WebGPU's default limits:
    // a scene past 128 MiB of triangles needs the adapter's limits.
    const renderer = new WebGPURenderer({
        requiredLimits: adapter ? { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } : undefined,
    });
    await renderer.init();
    const device = deviceFromRenderer(renderer); // renderer.backend.device; throws on the WebGL backend
    // #endregion
    return device;
}

// #region sync
import { SceneTrees } from '@pmndrs/bvh';
import { SceneTreesObject3DSync } from '@pmndrs/bvh/three';
import type { Mesh, Object3D } from 'three';

// #endregion

export async function sync(device: GPUDevice, scene: Object3D, materialIndexOf: (mesh: Mesh) => number): Promise<SceneTreesObject3DSync> {
    // #region sync
    const trees = await SceneTrees.create(device, { mergeStaticInstances: true });
    scene.updateMatrixWorld(); // the world matrices are read as they are
    const sync = new SceneTreesObject3DSync(trees, scene, {
        static: (object) => !object.userData.moves, // static instances can merge, and are never pushed again
        material: (mesh) => materialIndexOf(mesh), // becomes hit.material (0-65534)
        layers: true, // layers 0-7 become the instance mask (default)
    });

    // After adding or removing objects, or changing geometry, groups or materials:
    sync.sync();
    // Every frame: moved, shown, hidden or re-layered objects only.
    sync.update();
    // #endregion
    return sync;
}

// #region source-of
import { threeGroupOf, threeMaterialOf } from '@pmndrs/bvh/three';

// #endregion

export function sourceOf(sync: SceneTreesObject3DSync, hitInstance: number, hitTriangle: number): void {
    // #region source-of
    const source = sync.sourceOf(hitInstance); // { object, instanceId, group, materialIndex, mesh, static }
    const meshSource = sync.meshSourceOf(hitInstance);
    if (source && meshSource) {
        console.log(source.object.name, source.instanceId);
        // A folded multi-material mesh has group -1: the hit's triangle names its group and three material.
        const group = threeGroupOf(meshSource, hitTriangle); // { group, materialIndex, material, ... } or null
        const material = threeMaterialOf(source, meshSource, hitTriangle); // object.material[group.materialIndex], or object.material
        console.log(group?.materialIndex, material?.name);
    }
    // #endregion
}

// #region posing
import type { Scene } from 'three/webgpu';

// #endregion

export async function posing(renderer: WebGPURenderer, trees: SceneTrees, scene: Scene): Promise<void> {
    const device = trees.device;
    // #region posing
    // With a renderer, every SkinnedMesh and morph-target mesh is posed on the GPU (BvhPosePass).
    const sync = new SceneTreesObject3DSync(trees, scene, { renderer });
    await sync.ready; // the pose pipelines are compiled: every object is posed from the first update

    // Each frame, after the animation mixer:
    scene.updateMatrixWorld();
    sync.update(); // transforms, then each visible posed object's pass, then SceneTrees.setPose
    const encoder = device.createCommandEncoder();
    trees.update(encoder); // refits the posed trees on the GPU: nothing is read back
    device.queue.submit([encoder.finish()]);
    console.log(sync.poseCounts); // { posed, boundsMoved, waiting }
    // #endregion
}

// #region trace-nodes
import { wgslFn } from 'three/tsl';
import { BvhTraceNodes } from '@pmndrs/bvh/three';

// #endregion

export function traceNodes(renderer: WebGPURenderer, trees: SceneTrees): BvhTraceNodes {
    // #region trace-nodes
    const bvh = new BvhTraceNodes(renderer, trees); // binds SceneTrees' own buffers, no copy

    // WGSL: three's wgslFn with the walk as an include, calling it by name.
    const shadow = wgslFn(
        /* wgsl */ `fn shadow(p: vec3f, l: vec3f, d: f32) -> f32 {
    return select(1.0, 0.0, bvhTraceAny(BvhRay(p, 1e-3, l, d)));
}`,
        [bvh.module],
    );
    // #endregion
    console.log(shadow);
    return bvh;
}

export function traceNodesMasked(renderer: WebGPURenderer, trees: SceneTrees): void {
    // #region trace-nodes-options
    const bvh = new BvhTraceNodes(renderer, trees, {
        rayMasks: true, // default: traceClosestMasked and traceAnyMasked exist
        triangleFilter: /* wgsl */ `fn bvhTrianglePasses(candidate: BvhCandidate) -> bool { return candidate.query == BVH_QUERY_ANY; }`,
    });
    // #endregion
    bvh.dispose();
}

// #region cpu-raycast
import { BufferGeometry, Mesh as ThreeMesh, Raycaster, SphereGeometry, MeshBasicMaterial, DoubleSide, Vector3 } from 'three';
import { acceleratedRaycast, computeBoundsTree, disposeBoundsTree, prepareBoundsTrees, SAH } from '@pmndrs/bvh/three';

// #endregion

export async function cpuRaycast(): Promise<number> {
    // #region cpu-raycast
    await prepareBoundsTrees(); // once: loads tinybvh
    BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
    BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
    ThreeMesh.prototype.raycast = acceleratedRaycast;

    const mesh = new ThreeMesh(new SphereGeometry(1, 256, 128), new MeshBasicMaterial({ side: DoubleSide }));
    mesh.geometry.computeBoundsTree({ strategy: SAH, maxLeafSize: 8 }); // three-mesh-bvh's options are accepted

    const raycaster = new Raycaster(new Vector3(0, 0, 5), new Vector3(0, 0, -1));
    raycaster.firstHitOnly = true;
    const [hit] = raycaster.intersectObject(mesh); // distance, point, face, faceIndex, barycoord, uv, normal, ...
    // #endregion
    return hit ? hit.distance : -1;
}

// #region bounds-tree
import { deserializeBoundsTree, getBVHExtremes, MeshBVHHelper, type BvhBoundsTree } from '@pmndrs/bvh/three';
import { Box3, Sphere } from 'three';

// #endregion

export function boundsTree(mesh: ThreeMesh): void {
    // #region bounds-tree
    const tree: BvhBoundsTree = mesh.geometry.boundsTree ?? mesh.geometry.computeBoundsTree();

    const hits = tree.raycast(new Raycaster().ray, DoubleSide); // three intersections in the geometry's space
    const nearby = tree.intersectsSphere(new Sphere(new Vector3(), 0.5));
    const closest = tree.closestPointToPoint(new Vector3(0, 3, 0)); // { point: Vector3, distance, faceIndex }
    const box = tree.getBoundingBox(new Box3());

    // After editing the position attribute (same index):
    mesh.geometry.attributes.position.needsUpdate = true;
    tree.refit();

    // Save and load: our layout's bytes, not three-mesh-bvh's format.
    const bytes = tree.serialize();
    const loaded = deserializeBoundsTree(bytes, mesh.geometry);

    const helper = new MeshBVHHelper(mesh, 10); // a Group of the boxes down to depth 10
    // #endregion
    console.log(hits.length, nearby, closest, box, loaded.triangleCount, getBVHExtremes(tree), helper.depth);
}

// #region shapecast-three
import { INTERSECTED, NOT_INTERSECTED, type ExtendedTriangle } from '@pmndrs/bvh/three';

// #endregion

export function shapecastThree(tree: BvhBoundsTree): number {
    // #region shapecast-three
    const sphere = new Sphere(new Vector3(0, 0, 1), 0.5);
    let count = 0;
    tree.shapecast({
        intersectsBounds: (box: Box3) => (sphere.intersectsBox(box) ? INTERSECTED : NOT_INTERSECTED),
        intersectsTriangle: (triangle: ExtendedTriangle) => {
            if (triangle.intersectsSphere(sphere)) count++;
            return false;
        },
    });
    // #endregion
    return count;
}

// #region scene-raycast
import { BvhSceneRaycaster, raycastScene } from '@pmndrs/bvh/three';
import { Frustum, Matrix4, type Camera } from 'three';

// #endregion

export async function sceneRaycast(scene: Object3D, camera: Camera, raycaster: Raycaster): Promise<void> {
    // #region scene-raycast
    await prepareBoundsTrees();
    const picker = new BvhSceneRaycaster(scene); // a tree per geometry and range, and the top level
    const hits = raycastScene(picker, raycaster); // what raycaster.intersectObject(scene, true) returns

    // After objects move, show, hide, or come and go:
    scene.updateMatrixWorld();
    picker.update(); // the top level only, unless meshes changed

    const frustum = new Frustum().setFromProjectionMatrix(new Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse));
    const seen = picker.intersectsFrustum(frustum, camera.layers); // [{ object, instanceId, group }]
    // #endregion
    console.log(hits.length, seen.length);
}

// #region generate
import { GenerateMeshBvhWorker } from '@pmndrs/bvh/three';

// #endregion

export async function generate(geometry: BufferGeometry): Promise<void> {
    // #region generate
    const generator = new GenerateMeshBvhWorker(); // on @pmndrs/bvh/worker's pool; inline in Node
    geometry.boundsTree = await generator.generate(geometry, { maxLeafSize: 8, onProgress: (p) => console.log(p) });
    generator.dispose(); // stops the pool and rejects unfinished builds
    // #endregion
}

// #region meshbvh
import { MeshBVH } from '@pmndrs/bvh/three'; // was: import { MeshBVH } from 'three-mesh-bvh';

// #endregion

export function meshBvh(geometry: BufferGeometry): MeshBVH {
    // #region meshbvh
    const bvh = new MeshBVH(geometry, { maxLeafSize: 3, indirect: true }); // after await prepareBoundsTrees()
    // #endregion
    return bvh;
}

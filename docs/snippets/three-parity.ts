// Code samples for the three-mesh-bvh parity features in docs/three/*.mdx, docs/cpu-queries/helpers.mdx and
// docs/serialization (see getting-started.ts for how they are checked).

import { Vector3 } from 'three';

// #region scene-shapecast
import { BvhSceneRaycaster, CONTAINED, INTERSECTED, NOT_INTERSECTED } from '@pmndrs/bvh/three';
import type { Object3D, Sphere } from 'three';

// #endregion

export function sceneShapecast(scene: Object3D, sphere: Sphere): Object3D[] {
    // #region scene-shapecast
    const picker = new BvhSceneRaycaster(scene); // after `await prepareBoundsTrees()`
    const near: Object3D[] = [];
    picker.shapecast({
        // World boxes of the objects (and instances); CONTAINED takes a subtree whole.
        intersectsBounds: (box) => {
            if (!sphere.intersectsBox(box)) return NOT_INTERSECTED;
            return sphere.containsPoint(box.min) && sphere.containsPoint(box.max) ? CONTAINED : INTERSECTED;
        },
        // Each object in an entered box: instanceId is the InstancedMesh or BatchedMesh instance, else 0.
        intersectsObject: (object, instanceId) => {
            console.log(object.name, instanceId);
            near.push(object);
            return false; // true stops the cast
        },
    });
    // #endregion
    return near;
}

// #region batched-bounds-tree
import { BatchedMesh, Mesh } from 'three';
import { acceleratedRaycast, computeBatchedBoundsTree, disposeBatchedBoundsTree } from '@pmndrs/bvh/three';

// #endregion

export function batchedBoundsTree(batched: BatchedMesh, changedGeometry: number): void {
    // #region batched-bounds-tree
    Mesh.prototype.raycast = acceleratedRaycast; // other meshes; the BatchedMesh needs only its own line below
    BatchedMesh.prototype.raycast = acceleratedRaycast; // each instance walks its geometry's tree
    BatchedMesh.prototype.computeBoundsTree = computeBatchedBoundsTree;
    BatchedMesh.prototype.disposeBoundsTree = disposeBatchedBoundsTree;

    batched.computeBoundsTree(); // a tree per geometry, in batched.boundsTrees
    // After batched.setGeometryAt(changedGeometry, geometry): that geometry's tree again, the others refitted.
    batched.computeBoundsTree(changedGeometry);
    for (const tree of batched.boundsTrees ?? []) tree?.refit();
    // #endregion
}

// #region three-mesh-bvh-format
import { MeshBVH } from '@pmndrs/bvh/three';
import type { BufferGeometry } from 'three';

// #endregion

export function threeMeshBvhFormat(geometry: BufferGeometry, theirs: ReturnType<typeof MeshBVH.serialize>): void {
    // #region three-mesh-bvh-format
    // Ours to three-mesh-bvh: its MeshBVH.deserialize(data, geometry) reads this.
    const data = MeshBVH.serialize(new MeshBVH(geometry)); // { version: 1, roots, index, indirectBuffer }

    // three-mesh-bvh's to ours: its MeshBVH.serialize(bvh) output, structured-cloned from a worker or stored.
    const tree = MeshBVH.deserialize(theirs, geometry); // setIndex: true sets the index three-mesh-bvh wrote
    geometry.boundsTree = tree;

    // Our own bytes still go through the same call.
    const again = MeshBVH.deserialize(tree.serialize(), geometry);
    // #endregion
    console.log(data, again);
}

// #region json-structure
import { getJSONStructure, type CpuMeshBvh } from '@pmndrs/bvh';
import { getJSONStructure as getJSONStructureThree } from '@pmndrs/bvh/three';

// #endregion

export function jsonStructure(tree: CpuMeshBvh, geometry: BufferGeometry): void {
    // #region json-structure
    const root = getJSONStructure(tree); // { bounds: BvhBox3, left, right }, or { bounds, offset, count } at a leaf
    console.log(JSON.stringify(root, null, 2));
    // In three, the boxes are Box3s:
    const threeRoot = getJSONStructureThree(geometry.boundsTree!);
    // #endregion
    console.log(threeRoot.bounds.getSize(new Vector3()));
}

// #region update-async
import { BvhTraceNodes, SceneTreesObject3DSync, deviceFromRenderer, updateSceneTreesAsync } from '@pmndrs/bvh/three';
import { SceneTrees } from '@pmndrs/bvh';
import type { WebGPURenderer } from 'three/webgpu';

// #endregion

export async function updateAsync(renderer: WebGPURenderer, scene: Object3D): Promise<BvhTraceNodes> {
    // #region update-async
    const trees = await SceneTrees.create(deviceFromRenderer(renderer));
    const sync = new SceneTreesObject3DSync(trees, scene, { renderer });
    const bvh = new BvhTraceNodes(renderer, trees);

    // three-mesh-bvh's `await bvh.updateAsync()`: every tree built and placed before the first traced frame.
    await bvh.updateAsync({ sync }); // or updateSceneTreesAsync(trees, { sync, nodes: bvh, until: 'refined' })
    // #endregion
    await updateSceneTreesAsync(trees, { sync, nodes: bvh });
    return bvh;
}


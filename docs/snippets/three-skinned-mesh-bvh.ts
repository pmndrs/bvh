// Code sample for docs/three/skinned-and-morphed.mdx, "A tree that follows the pose" (see getting-started.ts for how samples are checked).

// #region skinned-mesh-bvh
import { SkinnedMeshBVH, acceleratedRaycast, prepareBoundsTrees, setBoundsTree } from '@pmndrs/bvh/three';
import { Mesh, SkinnedMesh, type Raycaster } from 'three';

// #endregion

export async function skinnedMeshBvh(character: SkinnedMesh, raycaster: Raycaster): Promise<() => void> {
    // #region skinned-mesh-bvh
    await prepareBoundsTrees();
    Mesh.prototype.raycast = acceleratedRaycast;
    SkinnedMesh.prototype.raycast = acceleratedRaycast; // three's SkinnedMesh has a raycast of its own

    const bvh = new SkinnedMeshBVH(character); // the current pose, in the mesh's own space
    setBoundsTree(character.geometry, bvh); // geometry.boundsTree = bvh

    // Each frame, after the animation mixer and updateMatrixWorld():
    const frame = () => {
        bvh.refit(); // poses every vertex again (three's getVertexPosition) and refits the boxes
        raycaster.firstHitOnly = true;
        const [hit] = raycaster.intersectObject(character); // the live pose
        console.log(hit?.point, hit?.face?.normal);
    };
    // #endregion
    return frame;
}

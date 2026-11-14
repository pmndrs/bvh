// Code sample for docs/three/skinned-and-morphed.mdx, "On the CPU" (see getting-started.ts for how samples are checked).

// #region static-geometry
import { StaticGeometryGenerator, computeBoundsTree } from '@pmndrs/bvh/three';
import { Mesh, type Raycaster, type SkinnedMesh } from 'three';

// #endregion

export function staticGeometry(character: SkinnedMesh, raycaster: Raycaster): () => void {
    // #region static-geometry
    const generator = new StaticGeometryGenerator(character); // every visible mesh under it
    const geometry = generator.generate(); // the current pose, in world space, a group per mesh
    computeBoundsTree.call(geometry);
    const collider = new Mesh(geometry, generator.getMaterials()); // identity transform: the pose is in world space

    // Each frame, after the animation mixer and updateMatrixWorld():
    const frame = () => {
        generator.generate(geometry); // the same arrays; the index is left alone
        geometry.boundsTree!.refit();
        raycaster.firstHitOnly = true;
        const [hit] = raycaster.intersectObject(collider); // the live pose
        console.log(hit?.point);
    };
    // #endregion
    return frame;
}

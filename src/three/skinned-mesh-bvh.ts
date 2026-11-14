/**
 * three-mesh-bvh's `SkinnedMeshBVH` (PARITY 3.4): a CPU tree over a
 * `SkinnedMesh`'s (or a morph-target mesh's) posed triangles, in the mesh's
 * own space, that follows the pose when `refit()` is called.
 *
 * ```ts
 * const bvh = new SkinnedMeshBVH(character);           // after `await prepareBoundsTrees()`
 * character.geometry.boundsTree = bvh;                 // acceleratedRaycast walks it for this mesh
 * // each frame, after the animation mixer and updateMatrixWorld():
 * bvh.refit();                                         // re-poses every vertex, refits the boxes
 * ```
 *
 * **As three-mesh-bvh does it.** Each vertex is posed with three's own
 * `mesh.getVertexPosition` (morph targets, then skinning: three's CPU pose,
 * the one three's raycast tests), in the mesh's space, so `matrixWorld`
 * places the tree as it places the mesh. three-mesh-bvh builds its tree over
 * the posed bounds and `refit()` recomputes them; here the posed positions go
 * into a `BvhBoundsTree` (tinybvh's build on the first pose) and `refit()` is
 * `CpuMeshBvh.refit` over the new pose, the tree's shape kept. Queries
 * (`raycast`, `raycastFirst`, `shapecast`, `closestPointToPoint`, ...) read
 * the posed corners; a hit's `face.normal` comes from them, its `uv` and
 * interpolated `normal` from the geometry's attributes, as three's raycast of
 * a `SkinnedMesh` gives them.
 *
 * Not `StaticGeometryGenerator`: that bakes every visible mesh under an object
 * into world space and flips a mirrored mesh's winding, while this tree is one
 * mesh in its own space, with the geometry's own triangle numbering.
 */

import { BufferAttribute, BufferGeometry, Vector3, type Mesh } from 'three';
import type { MeshInput } from '../mesh-input.ts';
import { BvhBoundsTree, positionVersionOf, type BvhBoundsTreeOptions, type BvhGeometrySnapshot } from './raycast.ts';

const vertex = new Vector3();

/** The mesh's vertices in its current pose, `x, y, z` each (three's `getVertexPosition`), into `target`. */
function posedPositions(mesh: Mesh, target: Float32Array): Float32Array {
    for (let i = 0, count = target.length / 3; i < count; i++) {
        mesh.getVertexPosition(i, vertex);
        target[i * 3] = vertex.x;
        target[i * 3 + 1] = vertex.y;
        target[i * 3 + 2] = vertex.z;
    }
    return target;
}

/** The geometry's index cut to whole triangles, as `meshInputFromGeometry` takes it. */
function indicesOf(geometry: BufferGeometry): Uint16Array | Uint32Array | null {
    const index = geometry.index;
    if (!index) return null;
    const length = Math.floor(index.count / 3) * 3;
    const array = index.array;
    return array instanceof Uint16Array || array instanceof Uint32Array ? array.subarray(0, length) : Uint32Array.from({ length }, (_, i) => index.getX(i));
}

/**
 * three-mesh-bvh's `SkinnedMeshBVH(mesh, options)`: a {@link BvhBoundsTree}
 * over the posed triangles of `mesh` (see the file comment). The options are
 * `computeBoundsTree`'s.
 */
export class SkinnedMeshBVH extends BvhBoundsTree {
    /** The mesh whose pose the tree follows. */
    readonly mesh: Mesh;
    /** The posed positions (`x, y, z` per vertex), rewritten by {@link refit}. */
    readonly posedPositions: Float32Array;
    private readonly posedView: BufferGeometry;

    constructor(mesh: Mesh, options: BvhBoundsTreeOptions = {}) {
        if (!(mesh as { isMesh?: boolean })?.isMesh) throw new Error('SkinnedMeshBVH: First argument must be a Mesh.');
        const geometry = mesh.geometry;
        const position = geometry.getAttribute('position') as BufferAttribute | undefined;
        if (!position) throw new Error('SkinnedMeshBVH: the geometry has no position attribute');
        const positions = posedPositions(mesh, new Float32Array(position.count * 3));
        const indices = indicesOf(geometry);
        const input: MeshInput = { positions: indices ? positions : positions.subarray(0, Math.floor(position.count / 3) * 9), indices };
        const snapshot: BvhGeometrySnapshot = { input, position, positionVersion: positionVersionOf(position), index: geometry.index, indexVersion: geometry.index ? geometry.index.version : 0 };
        super(geometry, { ...options, snapshot, layout: options.layout });
        this.mesh = mesh;
        this.posedPositions = positions;
        // The geometry's attributes with the posed positions: hits read their face normal from the pose.
        this.posedView = new BufferGeometry();
        for (const [name, attribute] of Object.entries(geometry.attributes)) this.posedView.setAttribute(name, attribute);
        this.posedView.setAttribute('position', new BufferAttribute(positions, 3));
        this.posedView.setIndex(geometry.index);
    }

    override get followsPose(): boolean {
        return true;
    }

    override get hitGeometry(): BufferGeometry {
        return this.posedView;
    }

    /**
     * Poses every vertex again (`mesh.getVertexPosition`) and refits the tree
     * to it, keeping its shape: three-mesh-bvh's `refit()` after the skeleton
     * or the morph influences moved. With `input`, refits to those arrays
     * instead (as the core does).
     */
    override refit(input?: MeshInput): void {
        if (input) {
            super.refit(input);
            return;
        }
        const geometry = this.mesh.geometry;
        if (geometry !== this.geometry || geometry.getAttribute('position') !== this.position || geometry.index !== this.index) {
            throw new Error('SkinnedMeshBVH.refit: the mesh\'s geometry, position attribute or index was replaced; make a new SkinnedMeshBVH');
        }
        posedPositions(this.mesh, this.posedPositions);
        const indices = indicesOf(geometry);
        super.refit({ positions: indices ? this.posedPositions : this.posedPositions.subarray(0, Math.floor(this.posedPositions.length / 9) * 9), indices });
    }
}

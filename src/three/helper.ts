/**
 * three-mesh-bvh's `BVHHelper` (and its older name `MeshBVHHelper`): a
 * `Group` that draws a tree's boxes, to a chosen depth, as line segments (or
 * see-through faces). The boxes and the index buffer come from the core
 * (`bvhHelperPositions`, `bvhHelperIndices` in `src/query/debug.ts`); this
 * file only puts them in three objects.
 */

import { BufferAttribute, BufferGeometry, Group, LineBasicMaterial, Matrix4, Mesh, MeshBasicMaterial, Object3D, type Color, type Frustum, type Material, type Vector3 } from 'three';
import { CpuMeshBvh } from '../query/cpu-mesh-bvh.ts';
import { bvhHelperIndices, bvhHelperPositions } from '../query/debug.ts';

const instanceMatrix = new Matrix4();

/** One root's boxes: drawn as `LineSegments` or as a `Mesh`, as `displayEdges` says (three-mesh-bvh's `BVHRootHelper`). */
class BvhRootHelper extends Object3D {
    material: Material;
    geometry = new BufferGeometry();
    depth: number;
    displayParents = false;
    displayEdges = true;
    bvh: CpuMeshBvh<unknown> | null;

    get isMesh(): boolean {
        return !this.displayEdges;
    }

    get isLineSegments(): boolean {
        return this.displayEdges;
    }

    get isLine(): boolean {
        return this.displayEdges;
    }

    constructor(bvh: CpuMeshBvh<unknown> | null, material: Material, depth = 10) {
        super();
        this.name = 'BVHRootHelper';
        this.material = material;
        this.depth = depth;
        this.bvh = bvh;
    }

    /** So `Box3.setFromObject` reads the boxes' corners. */
    getVertexPosition(index: number, target: Vector3): Vector3 {
        return Mesh.prototype.getVertexPosition.call(this as unknown as Mesh, index, target);
    }

    override raycast(): void {}

    /** three r186+ frustum-culls through this method, as for `Mesh` and `Line`. */
    intersectsFrustum(frustum: Frustum): boolean {
        return frustum.intersectsObject(this);
    }

    update(): void {
        this.geometry.dispose();
        this.visible = false;
        if (this.bvh) {
            const positions = bvhHelperPositions(this.bvh, this.depth, this.displayParents);
            const geometry = new BufferGeometry();
            geometry.setIndex(new BufferAttribute(bvhHelperIndices(positions, this.displayEdges), 1, false));
            geometry.setAttribute('position', new BufferAttribute(positions, 3, false));
            this.geometry = geometry;
            this.visible = true;
        }
    }
}

/** An object with a tree on it, as the helper looks for one. */
type WithTrees = Object3D & { boundsTree?: unknown; geometry?: { boundsTree?: unknown }; isInstancedMesh?: boolean; isBatchedMesh?: boolean; getMatrixAt?: (index: number, matrix: Matrix4) => Matrix4 };

/**
 * three-mesh-bvh's `BVHHelper`: draws `bvh` (or `mesh.geometry.boundsTree`)
 * as boxes down to `depth`, placed on `mesh` (its world matrix, and an
 * instance's with `instanceId` for an `InstancedMesh` or `BatchedMesh`).
 * Call {@link update} after changing the tree, `depth`, `displayParents` or
 * `displayEdges`. Our trees have one root, so there is one child.
 *
 * Constructor: `(mesh)`, `(mesh, depth)`, `(bvh, depth)` or `(mesh, bvh, depth)`;
 * `depth` defaults to 10.
 */
export class BVHHelper extends Group {
    depth: number;
    mesh: Object3D | null;
    bvh: CpuMeshBvh<unknown> | null;
    displayParents = false;
    displayEdges = true;
    instanceId = 0;
    readonly edgeMaterial: LineBasicMaterial;
    readonly meshMaterial: MeshBasicMaterial;
    private readonly roots: BvhRootHelper[] = [];

    /** `edgeMaterial.color`, shared with `meshMaterial`. */
    get color(): Color {
        return this.edgeMaterial.color;
    }

    /** Both materials' opacity. */
    get opacity(): number {
        return this.edgeMaterial.opacity;
    }

    set opacity(value: number) {
        this.edgeMaterial.opacity = value;
        this.meshMaterial.opacity = value;
    }

    constructor(meshOrBvh?: Object3D | CpuMeshBvh<unknown> | null, bvhOrDepth?: CpuMeshBvh<unknown> | number | null, depth = 10) {
        let mesh: Object3D | null = null;
        let bvh: CpuMeshBvh<unknown> | null = null;
        if (meshOrBvh instanceof CpuMeshBvh) {
            bvh = meshOrBvh;
            depth = typeof bvhOrDepth === 'number' && bvhOrDepth ? bvhOrDepth : 10;
        } else {
            mesh = meshOrBvh ?? null;
            if (typeof bvhOrDepth === 'number') depth = bvhOrDepth;
            else bvh = bvhOrDepth ?? null;
        }
        super();
        this.name = 'BVHHelper';
        this.depth = depth;
        this.mesh = mesh;
        this.bvh = bvh;
        this.edgeMaterial = new LineBasicMaterial({ color: 0x00ff88, transparent: true, opacity: 0.3, depthWrite: false });
        this.meshMaterial = new MeshBasicMaterial({ color: 0x00ff88, transparent: true, opacity: 0.3, depthWrite: false });
        this.meshMaterial.color = this.edgeMaterial.color;
        this.update();
    }

    /** The tree drawn: `bvh`, else the mesh's `boundsTree` or `geometry.boundsTree` when it is one of ours. */
    private findTree(): CpuMeshBvh<unknown> | null {
        if (this.bvh) return this.bvh;
        const mesh = this.mesh as WithTrees | null;
        const tree = mesh ? (mesh.boundsTree ?? mesh.geometry?.boundsTree) : null;
        return tree instanceof CpuMeshBvh ? tree : null;
    }

    /** Rebuilds the boxes from the tree as it is now, with the current `depth`, `displayParents` and `displayEdges`. */
    update(): void {
        const bvh = this.findTree();
        const rootCount = bvh ? 1 : 0;
        while (this.roots.length > rootCount) {
            const root = this.roots.pop()!;
            root.geometry.dispose();
            this.remove(root);
        }
        for (let i = 0; i < rootCount; i++) {
            if (i >= this.roots.length) {
                const root = new BvhRootHelper(bvh, this.edgeMaterial, this.depth);
                this.add(root);
                this.roots.push(root);
            }
            const root = this.roots[i];
            root.bvh = bvh;
            root.depth = this.depth;
            root.displayParents = this.displayParents;
            root.displayEdges = this.displayEdges;
            root.material = this.displayEdges ? this.edgeMaterial : this.meshMaterial;
            root.update();
        }
    }

    /** Follows the mesh: its world matrix relative to this helper's parent (and the instance's matrix). */
    override updateMatrixWorld(force?: boolean): void {
        const mesh = this.mesh as WithTrees | null;
        if (mesh) {
            mesh.updateWorldMatrix(true, false);
            if (this.parent) this.matrix.copy(this.parent.matrixWorld).invert().multiply(mesh.matrixWorld);
            else this.matrix.copy(mesh.matrixWorld);
            if ((mesh.isInstancedMesh || mesh.isBatchedMesh) && this.instanceId >= 0 && mesh.getMatrixAt) {
                mesh.getMatrixAt(this.instanceId, instanceMatrix);
                this.matrix.multiply(instanceMatrix);
            }
            this.matrix.decompose(this.position, this.quaternion, this.scale);
        }
        super.updateMatrixWorld(force);
    }

    override copy(source: this, recursive?: boolean): this {
        void recursive;
        this.depth = source.depth;
        this.mesh = source.mesh;
        this.bvh = source.bvh;
        this.opacity = source.opacity;
        this.color.copy(source.color);
        return this;
    }

    override clone(): this {
        return new (this.constructor as new () => this)().copy(this);
    }

    /** Disposes of the materials and the box geometry. */
    dispose(): void {
        this.edgeMaterial.dispose();
        this.meshMaterial.dispose();
        for (const root of this.roots) root.geometry.dispose();
    }
}

/** three-mesh-bvh's older name for {@link BVHHelper} (it warns there; not here). drei's story uses it. */
export class MeshBVHHelper extends BVHHelper {}

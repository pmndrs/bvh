/**
 * `BVHHelper` / `MeshBVHHelper` in Node: the line segments it builds for a
 * tree whose boxes are known (two triangles, a root and two leaves), and for a
 * larger tree the counts `getBVHExtremes` and the node walk predict, at each
 * depth, with and without parents; faces with `displayEdges` off; the
 * constructor forms of three-mesh-bvh; following the mesh's transform.
 */

import { BufferAttribute, BufferGeometry, InstancedMesh, LineBasicMaterial, Matrix4, Mesh, MeshBasicMaterial, Scene, TorusKnotGeometry, Vector3 } from 'three';
import { beforeAll, describe, expect, it } from 'vitest';
import { getBVHExtremes } from '../query/debug.ts';
import { BVHHelper, MeshBVHHelper } from './helper.ts';
import { BvhBoundsTree, prepareBoundsTrees, setBoundsTree } from './raycast.ts';

beforeAll(async () => {
    await prepareBoundsTrees();
});

interface Drawn {
    isLineSegments: boolean;
    isMesh: boolean;
    visible: boolean;
    geometry: BufferGeometry;
    material: unknown;
}

const root = (helper: BVHHelper): Drawn => helper.children[0] as unknown as Drawn;
/** Line segments drawn (two indices each). */
const segments = (helper: BVHHelper): number => root(helper).geometry.index!.count / 2;

function twoTriangleMesh(): Mesh {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 10, 0, 0, 11, 0, 0, 10, 1, 0]), 3));
    setBoundsTree(geometry, new BvhBoundsTree(geometry, { maxLeafSize: 1 }));
    return new Mesh(geometry);
}

describe('BVHHelper', () => {
    it('draws the two leaves of a known tree as 24 segments, the root too with displayParents, and the root alone at depth 1', () => {
        const mesh = twoTriangleMesh();
        const helper = new MeshBVHHelper(mesh);
        expect(helper).toBeInstanceOf(BVHHelper);
        expect(helper.depth).toBe(10);
        expect(helper.children.length).toBe(1);
        expect(root(helper).isLineSegments).toBe(true);
        expect(root(helper).isMesh).toBe(false);
        expect(segments(helper)).toBe(24);
        const position = root(helper).geometry.getAttribute('position');
        expect(position.count).toBe(16);
        // The leaves' corners: the two unit boxes, flat in z.
        const xs = new Set<number>();
        for (let i = 0; i < position.count; i++) {
            xs.add(position.getX(i));
            expect(position.getZ(i)).toBe(0);
        }
        expect([...xs].sort((a, b) => a - b)).toEqual([0, 1, 10, 11]);

        helper.displayParents = true;
        helper.update();
        expect(segments(helper)).toBe(36);
        helper.displayParents = false;
        helper.depth = 1;
        helper.update();
        expect(segments(helper)).toBe(12);
        const rootBox = root(helper).geometry.getAttribute('position');
        expect([rootBox.getX(0), rootBox.getY(0), rootBox.getX(7), rootBox.getY(7)]).toEqual([0, 0, 11, 1]);
    });

    it('matches the node counts of a larger tree at every depth', () => {
        const geometry = new TorusKnotGeometry(1, 0.3, 128, 16);
        setBoundsTree(geometry, new BvhBoundsTree(geometry));
        const tree = (geometry as unknown as { boundsTree: BvhBoundsTree }).boundsTree;
        const [extremes] = getBVHExtremes(tree);
        const helper = new BVHHelper(tree, 64);
        expect(segments(helper)).toBe(12 * extremes.leafNodeCount);
        helper.displayParents = true;
        helper.update();
        expect(segments(helper)).toBe(12 * extremes.nodeCount);
        for (const depth of [1, 2, 3, 5, 8]) {
            // Boxes at depth - 1, and leaves above it.
            let expected = 0;
            let withParents = 0;
            tree.traverseNodes((nodeDepth, isLeaf) => {
                const stop = nodeDepth >= depth - 1 || isLeaf;
                if (stop) expected++;
                withParents++;
                return stop;
            });
            helper.depth = depth;
            helper.displayParents = false;
            helper.update();
            expect(segments(helper), `depth ${depth}`).toBe(12 * expected);
            helper.displayParents = true;
            helper.update();
            expect(segments(helper), `depth ${depth}, parents`).toBe(12 * withParents);
        }
        // Faces instead of edges: 12 triangles per box, with the mesh material.
        helper.displayEdges = false;
        helper.displayParents = false;
        helper.depth = 64;
        helper.update();
        expect(root(helper).isMesh).toBe(true);
        expect(root(helper).material).toBeInstanceOf(MeshBasicMaterial);
        expect(root(helper).geometry.index!.count).toBe(36 * extremes.leafNodeCount);
        helper.dispose();
    });

    it('takes three-mesh-bvh constructor forms, its colour and opacity, and follows the mesh', () => {
        const mesh = twoTriangleMesh();
        const tree = (mesh.geometry as unknown as { boundsTree: BvhBoundsTree }).boundsTree;
        expect(new BVHHelper(mesh, 3).depth).toBe(3);
        expect(new BVHHelper(tree, 4).bvh).toBe(tree);
        expect(new BVHHelper(tree, 4).depth).toBe(4);
        expect(new BVHHelper(mesh, tree, 5).depth).toBe(5);
        // No tree yet: nothing drawn, and update() picks it up later.
        const bare = new Mesh(new TorusKnotGeometry(1, 0.3, 16, 4));
        const waiting = new BVHHelper(bare);
        expect(waiting.children.length).toBe(0);
        setBoundsTree(bare.geometry, new BvhBoundsTree(bare.geometry));
        waiting.update();
        expect(waiting.children.length).toBe(1);

        const helper = new BVHHelper(mesh);
        expect(helper.color.getHex()).toBe(0x00ff88);
        expect(helper.opacity).toBe(0.3);
        helper.opacity = 0.7;
        expect(helper.meshMaterial.opacity).toBe(0.7);
        expect(root(helper).material).toBeInstanceOf(LineBasicMaterial);

        const scene = new Scene();
        scene.add(mesh, helper);
        mesh.position.set(1, 2, 3);
        scene.updateMatrixWorld(true);
        expect(helper.position).toEqual(new Vector3(1, 2, 3));

        // An instance of an InstancedMesh: its matrix on top of the mesh's.
        const instanced = new InstancedMesh(mesh.geometry, new MeshBasicMaterial(), 2);
        instanced.setMatrixAt(1, new Matrix4().makeTranslation(0, 5, 0));
        const instanceHelper = new BVHHelper(instanced);
        instanceHelper.instanceId = 1;
        scene.add(instanced, instanceHelper);
        scene.updateMatrixWorld(true);
        expect(instanceHelper.position).toEqual(new Vector3(0, 5, 0));
    });
});

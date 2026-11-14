/** `SceneTreesObject3DSync.meshHandles`: the mesh handles the sync holds, as meshes come and go. */

import { BoxGeometry, Group, InstancedMesh, Mesh, MeshBasicMaterial, SphereGeometry } from 'three';
import { describe, expect, it } from 'vitest';
import type { MeshGeometry } from '../trace/mesh-bvh.ts';
import type { SceneInstanceDescriptor } from '../scene/scene-trees.ts';
import { SceneTreesObject3DSync, type SceneTreesTarget } from './scene-trees-sync.ts';

/** A target that hands out handles and keeps the meshes it has. */
class Handles implements SceneTreesTarget {
    readonly meshes = new Set<number>();
    private next = 100;
    addMesh(_geometry: MeshGeometry): number {
        const handle = this.next++;
        this.meshes.add(handle);
        return handle;
    }
    removeMesh(mesh: number): void {
        this.meshes.delete(mesh);
    }
    addInstance(_descriptor: SceneInstanceDescriptor): number {
        return this.next++;
    }
    setTransform(): void {}
    setInstance(): void {}
    removeInstance(): void {}
}

describe('SceneTreesObject3DSync.meshHandles', () => {
    it('lists the handles of the meshes it added, and follows adds and removals', () => {
        const root = new Group();
        const box = new BoxGeometry();
        const a = new Mesh(box, new MeshBasicMaterial());
        const b = new Mesh(box, new MeshBasicMaterial()); // shares a's mesh
        const spheres = new InstancedMesh(new SphereGeometry(0.5, 6, 4), new MeshBasicMaterial(), 3);
        root.add(a, b, spheres);
        root.updateMatrixWorld(true);
        const trees = new Handles();
        const sync = new SceneTreesObject3DSync(trees, root);
        expect(sync.meshHandles.sort()).toEqual([...trees.meshes].sort());
        expect(sync.meshHandles).toHaveLength(2);
        expect(sync.meshHandles).toHaveLength(sync.meshCount);

        root.remove(spheres);
        sync.sync();
        expect(sync.meshHandles).toEqual([...trees.meshes]);
        expect(sync.meshHandles).toHaveLength(1);
        sync.dispose();
    });
});

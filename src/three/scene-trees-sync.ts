/**
 * Keeps a `SceneTrees` in step with a three scene: the meshes and instances
 * {@link walkObject3D} finds are added, and each frame {@link SceneTreesObject3DSync.update}
 * pushes what changed (`setTransform` for moved instances, `setInstance` for
 * visibility, layers and material), leaving static instances alone.
 *
 * - **`update()`** is the per-frame call: it reads the world matrices (call
 *   `updateMatrixWorld()` first, or let the renderer do it) of the instances it
 *   already knows, and pushes only the ones that changed. It does not walk the
 *   scene: objects added or removed since, a changed `InstancedMesh.count` or
 *   a new geometry need {@link SceneTreesObject3DSync.sync}.
 * - **`sync()`** walks the scene again and adds and removes meshes and
 *   instances to match (an object whose geometry or triangle range changed is
 *   removed and added again). It is cheap for a scene that didn't change, but
 *   it walks every object: call it when the scene's contents change, not every frame.
 * - **Static instances** (`static` option) never get a `setTransform` or
 *   `setInstance` from the sync; with `mergeStaticInstances` SceneTrees would
 *   throw for them. Their state at the first sync is what rays see.
 *
 * `BvhHit.instance` is the SceneTrees instance's `instanceIndex`, which the
 * sync sets to its handle: {@link SceneTreesObject3DSync.sourceOf} maps it to
 * the three object, instance and group.
 */

import type { Object3D } from 'three';
import type { MeshGeometry } from '../trace/mesh-bvh.ts';
import type { Mat4 } from '../trace/top-level.ts';
import type { SceneInstanceChange, SceneInstanceDescriptor, SceneMeshOptions } from '../scene/scene-trees.ts';
import { meshGeometryOf, walkMesh, walkObject3D, type BvhSceneFromObject3DOptions, type BvhThreeMeshSource, type BvhThreeSource, type WalkedInstance } from './scene.ts';

/** The part of `SceneTrees` the sync drives (a `SceneTrees` is one). */
export interface SceneTreesTarget {
    addMesh(geometry: MeshGeometry, options?: SceneMeshOptions): number;
    removeMesh(mesh: number): void;
    addInstance(descriptor: SceneInstanceDescriptor): number;
    setTransform(instance: number, objectToWorld: Mat4): void;
    setInstance(instance: number, change: SceneInstanceChange): void;
    removeInstance(instance: number): void;
}

/** What one {@link SceneTreesObject3DSync.update} or {@link SceneTreesObject3DSync.sync} pushed. */
export interface Object3DSyncCounts {
    readonly moved: number;
    readonly changed: number;
    readonly added: number;
    readonly removed: number;
    readonly meshesAdded: number;
    readonly meshesRemoved: number;
}

interface TrackedMesh {
    readonly handle: number;
    readonly source: BvhThreeMeshSource;
    users: number;
}

interface TrackedInstance {
    readonly key: string;
    readonly handle: number;
    readonly meshKey: string;
    readonly source: BvhThreeSource;
    readonly objectToWorld: Float64Array;
    visible: boolean;
    mask: number;
    material: number | undefined;
}

function sameMatrix(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
    for (let i = 0; i < 16; i++) if (a[i] !== b[i]) return false;
    return true;
}

export class SceneTreesObject3DSync {
    private readonly meshes = new Map<string, TrackedMesh>();
    private readonly instances = new Map<string, TrackedInstance>();
    private readonly byHandle = new Map<number, TrackedInstance>();
    private disposed = false;

    /** Adds every mesh and instance under `root` to `trees` (one {@link sync}). */
    constructor(
        readonly trees: SceneTreesTarget,
        readonly root: Object3D | readonly Object3D[],
        private readonly options: BvhSceneFromObject3DOptions = {},
    ) {
        this.sync();
    }

    /** The three object behind a hit's `instance` (the SceneTrees `instanceIndex`), or null. */
    sourceOf(instanceIndex: number): BvhThreeSource | null {
        return this.byHandle.get(instanceIndex)?.source ?? null;
    }

    /** The geometry and range of a hit instance's mesh (for {@link threeFaceIndex}), or null. */
    meshSourceOf(instanceIndex: number): BvhThreeMeshSource | null {
        const tracked = this.byHandle.get(instanceIndex);
        return tracked ? (this.meshes.get(tracked.meshKey)?.source ?? null) : null;
    }

    get instanceCount(): number {
        return this.instances.size;
    }

    get meshCount(): number {
        return this.meshes.size;
    }

    /** Walks the scene again: adds new meshes and instances, removes gone ones, then pushes changes as {@link update} does. */
    sync(): Object3DSyncCounts {
        this.checkLive();
        const walked = walkObject3D(this.root, this.options);
        const seen = new Set<string>();
        let added = 0;
        let removed = 0;
        let meshesAdded = 0;
        let meshesRemoved = 0;
        let moved = 0;
        let changed = 0;
        for (const instance of walked) {
            seen.add(instance.key);
            const tracked = this.instances.get(instance.key);
            if (tracked && tracked.meshKey === instance.meshKey) {
                const pushed = this.push(tracked, instance);
                moved += pushed.moved;
                changed += pushed.changed;
                continue;
            }
            if (tracked) {
                this.removeTracked(instance.key, tracked);
                removed++;
            }
            let mesh = this.meshes.get(instance.meshKey);
            if (!mesh) {
                const source = { geometry: instance.geometry, start: instance.range.start, count: instance.range.count };
                mesh = { handle: this.trees.addMesh(meshGeometryOf(source)), source, users: 0 };
                this.meshes.set(instance.meshKey, mesh);
                meshesAdded++;
            }
            mesh.users++;
            // No instanceIndex: SceneTrees uses the handle, which sourceOf() maps back.
            const handle = this.trees.addInstance({
                mesh: mesh.handle,
                objectToWorld: instance.objectToWorld,
                visible: instance.visible,
                mask: instance.mask,
                material: instance.material,
                static: instance.static,
            });
            const record: TrackedInstance = {
                key: instance.key,
                handle,
                meshKey: instance.meshKey,
                source: {
                    object: instance.object,
                    instanceId: instance.instanceId,
                    group: instance.range.group,
                    materialIndex: instance.range.materialIndex,
                    mesh: mesh.handle,
                    static: instance.static,
                },
                objectToWorld: Float64Array.from(instance.objectToWorld),
                visible: instance.visible,
                mask: instance.mask,
                material: instance.material,
            };
            this.instances.set(instance.key, record);
            this.byHandle.set(handle, record);
            added++;
        }
        for (const [key, tracked] of [...this.instances]) {
            if (seen.has(key)) continue;
            this.removeTracked(key, tracked);
            removed++;
        }
        for (const [key, mesh] of [...this.meshes]) {
            if (mesh.users > 0) continue;
            this.trees.removeMesh(mesh.handle);
            this.meshes.delete(key);
            meshesRemoved++;
        }
        return { moved, changed, added, removed, meshesAdded, meshesRemoved };
    }

    /**
     * Pushes the moved, shown, hidden, re-layered and re-materialed instances
     * the sync knows (not static ones). Call once per frame, after
     * `updateMatrixWorld()` and before `SceneTrees.update`.
     */
    update(): Object3DSyncCounts {
        this.checkLive();
        let moved = 0;
        let changed = 0;
        const groups = new Map<string, TrackedInstance[]>();
        for (const tracked of this.instances.values()) {
            if (tracked.source.static) continue;
            // Instances of one object are read together (one walk of an InstancedMesh's matrices).
            const key = `${tracked.source.object.id}`;
            let list = groups.get(key);
            if (!list) groups.set(key, (list = []));
            list.push(tracked);
        }
        for (const list of groups.values()) {
            const object = list[0].source.object;
            const fresh: WalkedInstance[] = [];
            walkMesh(object, this.options, fresh);
            const byKey = new Map<string, WalkedInstance>();
            for (const walked of fresh) byKey.set(walked.key, walked);
            for (const tracked of list) {
                const walked = byKey.get(tracked.key);
                if (!walked) continue; // gone (count lowered): sync() removes it
                const pushed = this.push(tracked, walked);
                moved += pushed.moved;
                changed += pushed.changed;
            }
        }
        return { moved, changed, added: 0, removed: 0, meshesAdded: 0, meshesRemoved: 0 };
    }

    /** Removes every instance and mesh the sync added from the SceneTrees. */
    dispose(): void {
        if (this.disposed) return;
        for (const [key, tracked] of [...this.instances]) this.removeTracked(key, tracked);
        for (const mesh of this.meshes.values()) this.trees.removeMesh(mesh.handle);
        this.meshes.clear();
        this.disposed = true;
    }

    private push(tracked: TrackedInstance, walked: WalkedInstance): { moved: number; changed: number } {
        if (tracked.source.static) return { moved: 0, changed: 0 };
        let moved = 0;
        let changed = 0;
        if (!sameMatrix(tracked.objectToWorld, walked.objectToWorld)) {
            tracked.objectToWorld.set(walked.objectToWorld);
            this.trees.setTransform(tracked.handle, walked.objectToWorld);
            moved = 1;
        }
        const change: { visible?: boolean; mask?: number; material?: number | null } = {};
        if (tracked.visible !== walked.visible) change.visible = tracked.visible = walked.visible;
        if (tracked.mask !== walked.mask) change.mask = tracked.mask = walked.mask;
        if (tracked.material !== walked.material) {
            tracked.material = walked.material;
            change.material = walked.material ?? null;
        }
        if (Object.keys(change).length > 0) {
            this.trees.setInstance(tracked.handle, change);
            changed = 1;
        }
        return { moved, changed };
    }

    private removeTracked(key: string, tracked: TrackedInstance): void {
        this.trees.removeInstance(tracked.handle);
        this.instances.delete(key);
        this.byHandle.delete(tracked.handle);
        const mesh = this.meshes.get(tracked.meshKey);
        if (mesh) mesh.users--;
    }

    private checkLive(): void {
        if (this.disposed) throw new Error('@pmndrs/bvh/three: this SceneTreesObject3DSync was disposed');
    }
}

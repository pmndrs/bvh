/**
 * Keeps a `SceneTrees` in step with a three scene: the meshes and instances
 * {@link walkObject3D} finds are added, and each frame {@link SceneTreesObject3DSync.update}
 * pushes what changed (`setTransform` for moved instances, `setInstance` for
 * visibility, layers and material), leaving static instances alone.
 *
 * - **`update()`** is the per-frame call: it reads the world matrices (call
 *   `updateMatrixWorld()` first, or let the renderer do it) of the instances it
 *   already knows, and pushes only the ones that changed. An object whose
 *   `matrixWorld`, effective visibility, layers and instance matrices
 *   (`instanceMatrix.version`, a `BatchedMesh`'s matrices texture version)
 *   are unchanged is skipped without touching its instances (a `BatchedMesh`'s
 *   per-instance visibility is still compared, as three gives it no version).
 *   It does not walk the scene: objects added or removed since, a changed
 *   `InstancedMesh.count` or a new geometry need {@link SceneTreesObject3DSync.sync}.
 *   The `material` callback is read at `sync()` only: a material index that
 *   changes needs a `sync()`.
 * - **`sync()`** walks the scene again and adds and removes meshes and
 *   instances to match (an object whose geometry or triangle range changed is
 *   removed and added again). It is cheap for a scene that didn't change, but
 *   it walks every object: call it when the scene's contents change, not every frame.
 * - **Static instances** (`static` option) never get a `setTransform` or
 *   `setInstance` from the sync; with `mergeStaticInstances` SceneTrees would
 *   throw for them. Their state at the first sync is what rays see.
 * - **Merged static content can't change.** Once `SceneTrees.update` merged
 *   an object's static instances (`mergeStaticInstances`), removing the
 *   object or changing its geometry, draw range, groups or materials is
 *   skipped: the merged triangles stay as they were, and `sync()` warns once
 *   per object. Static means static; mark objects that may go away
 *   `static: false`. A new object on a geometry whose mesh was merged away
 *   gets that mesh added again.
 *
 * `BvhHit.instance` is the SceneTrees instance's `instanceIndex`, which the
 * sync sets to its handle: {@link SceneTreesObject3DSync.sourceOf} maps it to
 * the three object, instance and group.
 */

import type { BatchedMesh, InstancedMesh, Mesh, Object3D } from 'three';
import { Matrix4 } from 'three';
import type { MeshGeometry } from '../trace/mesh-bvh.ts';
import type { Mat4 } from '../trace/top-level.ts';
import type { SceneInstanceChange, SceneInstanceDescriptor, SceneMeshOptions } from '../scene/scene-trees.ts';
import {
    effectivelyVisible,
    instanceMaskFromLayers,
    meshGeometryOf,
    multiplyMatrices,
    walkObject3D,
    type BvhSceneFromObject3DOptions,
    type BvhThreeMeshSource,
    type BvhThreeSource,
    type WalkedInstance,
} from './scene.ts';

/** The part of `SceneTrees` the sync drives (a `SceneTrees` is one). */
export interface SceneTreesTarget {
    addMesh(geometry: MeshGeometry, options?: SceneMeshOptions): number;
    removeMesh(mesh: number): void;
    addInstance(descriptor: SceneInstanceDescriptor): number;
    setTransform(instance: number, objectToWorld: Mat4): void;
    setInstance(instance: number, change: SceneInstanceChange): void;
    removeInstance(instance: number): void;
    /** Whether a mesh was merged away (`SceneTrees.meshStatus`). Without it, no mesh is. */
    meshStatus?(mesh: number): { readonly mergedInto: number | null };
    /** Whether an instance was merged (`SceneTrees.instanceStatus`). Without it, no instance is. */
    instanceStatus?(instance: number): { readonly mergedInto: number | null };
}

/** What one {@link SceneTreesObject3DSync.update} or {@link SceneTreesObject3DSync.sync} pushed. */
export interface Object3DSyncCounts {
    readonly moved: number;
    readonly changed: number;
    readonly added: number;
    readonly removed: number;
    readonly meshesAdded: number;
    readonly meshesRemoved: number;
    /** Changes to merged static objects that were skipped (see the file comment). */
    readonly skipped: number;
}

interface TrackedMesh {
    readonly key: string;
    readonly handle: number;
    readonly source: BvhThreeMeshSource;
    users: number;
}

interface TrackedInstance {
    readonly key: string;
    readonly handle: number;
    readonly mesh: TrackedMesh;
    readonly source: BvhThreeSource;
    readonly objectToWorld: Float64Array;
    visible: boolean;
    mask: number;
    material: number | undefined;
    /** Known merged into a static group (never changes back). */
    merged: boolean;
}

/** One object's non-static instances and what {@link SceneTreesObject3DSync.update} last saw of it. */
interface TrackedObject {
    readonly object: Mesh;
    readonly instances: TrackedInstance[];
    readonly world: Float64Array;
    visible: boolean;
    mask: number;
    matrices: number;
}

function sameMatrix(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
    for (let i = 0; i < 16; i++) if (a[i] !== b[i]) return false;
    return true;
}

/** The version of an object's per-instance matrices: `instanceMatrix.version`, a BatchedMesh's matrices texture, or 0. */
function matricesVersion(object: Mesh): number {
    const instanced = object as unknown as InstancedMesh;
    if (instanced.isInstancedMesh) return instanced.instanceMatrix.version;
    const batched = object as unknown as BatchedMesh & { _matricesTexture?: { version: number } };
    if (batched.isBatchedMesh) return batched._matricesTexture?.version ?? -1;
    return 0;
}

type BatchedInfo = { active: boolean; visible: boolean }[];

const scratchWorld = new Float64Array(16);
const scratchMatrix = new Matrix4();

export class SceneTreesObject3DSync {
    private readonly meshes = new Map<string, TrackedMesh>();
    private readonly instances = new Map<string, TrackedInstance>();
    private readonly byHandle = new Map<number, TrackedInstance>();
    private readonly objects = new Map<Mesh, TrackedObject>();
    private readonly warned = new WeakSet<Object3D>();
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
        return this.byHandle.get(instanceIndex)?.mesh.source ?? null;
    }

    get instanceCount(): number {
        return this.instances.size;
    }

    /** Meshes the sync can still share (a mesh merged away is no longer counted). */
    get meshCount(): number {
        return this.meshes.size;
    }

    /**
     * Walks the scene again: adds new meshes and instances, removes gone ones,
     * then pushes changes as {@link update} does (and material indices). The
     * walk and the plan come first, so a throwing `material` callback changes
     * nothing; each SceneTrees call then updates the sync's records with it.
     */
    sync(): Object3DSyncCounts {
        this.checkLive();
        const walked = walkObject3D(this.root, this.options);

        // Plan.
        const mergedObjects = new Set<Object3D>();
        for (const tracked of this.instances.values()) if (this.isMerged(tracked)) mergedObjects.add(tracked.source.object);
        const seen = new Set<string>();
        const keep: [TrackedInstance, WalkedInstance][] = [];
        const toRemove: TrackedInstance[] = [];
        const toAdd: WalkedInstance[] = [];
        let skipped = 0;
        for (const instance of walked) {
            seen.add(instance.key);
            const tracked = this.instances.get(instance.key);
            if (tracked && tracked.mesh.key === instance.meshKey) {
                keep.push([tracked, instance]);
                continue;
            }
            if (mergedObjects.has(instance.object)) {
                // A merged object's geometry, range or groups changed: static means static.
                this.warnMerged(instance.object);
                skipped++;
                continue;
            }
            if (tracked) toRemove.push(tracked);
            toAdd.push(instance);
        }
        for (const [key, tracked] of this.instances) {
            if (seen.has(key)) continue;
            if (tracked.merged) {
                this.warnMerged(tracked.source.object);
                skipped++;
                continue;
            }
            toRemove.push(tracked);
        }

        // Apply.
        let added = 0;
        let removed = 0;
        let meshesAdded = 0;
        let meshesRemoved = 0;
        let moved = 0;
        let changed = 0;
        try {
            for (const tracked of toRemove) {
                this.removeTracked(tracked);
                removed++;
            }
            for (const instance of toAdd) {
                let mesh = this.meshes.get(instance.meshKey);
                if (mesh && this.trees.meshStatus?.(mesh.handle).mergedInto != null) {
                    // SceneTrees dropped its geometry at the merge: a new instance needs the mesh again.
                    this.meshes.delete(instance.meshKey);
                    mesh = undefined;
                }
                if (!mesh) {
                    const source = { geometry: instance.geometry, start: instance.range.start, count: instance.range.count };
                    mesh = { key: instance.meshKey, handle: this.trees.addMesh(meshGeometryOf(source)), source, users: 0 };
                    this.meshes.set(instance.meshKey, mesh);
                    meshesAdded++;
                }
                // No instanceIndex: SceneTrees uses the handle, which sourceOf() maps back.
                const handle = this.trees.addInstance({
                    mesh: mesh.handle,
                    objectToWorld: instance.objectToWorld,
                    visible: instance.visible,
                    mask: instance.mask,
                    material: instance.material,
                    static: instance.static,
                });
                mesh.users++;
                const record: TrackedInstance = {
                    key: instance.key,
                    handle,
                    mesh,
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
                    merged: false,
                };
                this.instances.set(instance.key, record);
                this.byHandle.set(handle, record);
                added++;
            }
            for (const [tracked, instance] of keep) {
                if (tracked.source.static) continue;
                if (this.pushTransform(tracked, instance.objectToWorld)) moved++;
                if (this.pushState(tracked, instance.visible, instance.mask, instance.material, true)) changed++;
            }
            for (const [key, mesh] of [...this.meshes]) {
                if (mesh.users > 0) continue;
                this.trees.removeMesh(mesh.handle);
                this.meshes.delete(key);
                meshesRemoved++;
            }
        } finally {
            this.rebuildObjects();
        }
        return { moved, changed, added, removed, meshesAdded, meshesRemoved, skipped };
    }

    /**
     * Pushes the moved, shown, hidden and re-layered instances the sync knows
     * (not static ones), skipping objects that didn't change (see the file
     * comment). Call once per frame, after `updateMatrixWorld()` and before
     * `SceneTrees.update`.
     */
    update(): Object3DSyncCounts {
        this.checkLive();
        let moved = 0;
        let changed = 0;
        const useLayers = this.options.layers ?? true;
        for (const tracked of this.objects.values()) {
            const object = tracked.object;
            const world = object.matrixWorld.elements;
            const visible = effectivelyVisible(object);
            const mask = instanceMaskFromLayers(object.layers, useLayers);
            const matrices = matricesVersion(object);
            const batched = (object as unknown as BatchedMesh).isBatchedMesh === true;
            const same = tracked.visible === visible && tracked.mask === mask && tracked.matrices === matrices && sameMatrix(tracked.world, world);
            if (same && !batched) continue;
            tracked.visible = visible;
            tracked.mask = mask;
            tracked.matrices = matrices;
            tracked.world.set(world);
            const instanced = object as unknown as InstancedMesh;
            const info = batched ? (object as unknown as { _instanceInfo: BatchedInfo })._instanceInfo : null;
            for (const instance of tracked.instances) {
                const id = instance.source.instanceId;
                let instanceVisible = visible;
                let objectToWorld: ArrayLike<number> = world;
                if (info) {
                    const entry = info[id];
                    if (!entry?.active) continue; // gone: sync() removes it
                    instanceVisible = visible && entry.visible;
                    if (!same) {
                        (object as unknown as BatchedMesh).getMatrixAt(id, scratchMatrix);
                        objectToWorld = multiplyMatrices(world, scratchMatrix.elements, scratchWorld);
                    }
                } else if (instanced.isInstancedMesh) {
                    if (id >= instanced.count) continue; // gone (count lowered): sync() removes it
                    objectToWorld = multiplyMatrices(world, instanced.instanceMatrix.array, scratchWorld, id * 16);
                }
                if (!same && this.pushTransform(instance, objectToWorld)) moved++;
                if (this.pushState(instance, instanceVisible, mask, instance.material, false)) changed++;
            }
        }
        return { moved, changed, added: 0, removed: 0, meshesAdded: 0, meshesRemoved: 0, skipped: 0 };
    }

    /**
     * Removes every instance and mesh the sync added from the SceneTrees,
     * except merged static instances (SceneTrees can't remove them) and their meshes.
     */
    dispose(): void {
        if (this.disposed) return;
        for (const tracked of [...this.instances.values()]) {
            if (this.isMerged(tracked)) continue;
            this.removeTracked(tracked);
        }
        for (const mesh of this.meshes.values()) if (mesh.users === 0) this.trees.removeMesh(mesh.handle);
        this.meshes.clear();
        this.objects.clear();
        this.disposed = true;
    }

    private isMerged(tracked: TrackedInstance): boolean {
        if (tracked.merged) return true;
        if (!tracked.source.static || !this.trees.instanceStatus) return false;
        tracked.merged = this.trees.instanceStatus(tracked.handle).mergedInto !== null;
        return tracked.merged;
    }

    private warnMerged(object: Object3D): void {
        if (this.warned.has(object)) return;
        this.warned.add(object);
        console.warn(
            `@pmndrs/bvh/three: "${object.name || object.type}" (id ${object.id}) is merged static content (mergeStaticInstances); ` +
                'removing it or changing its geometry, groups or materials is ignored and rays keep seeing it as merged. Mark it static: false to change it later',
        );
    }

    private pushTransform(tracked: TrackedInstance, objectToWorld: ArrayLike<number>): boolean {
        if (sameMatrix(tracked.objectToWorld, objectToWorld)) return false;
        this.trees.setTransform(tracked.handle, objectToWorld as Mat4);
        tracked.objectToWorld.set(objectToWorld);
        return true;
    }

    private pushState(tracked: TrackedInstance, visible: boolean, mask: number, material: number | undefined, withMaterial: boolean): boolean {
        const visibleChanged = tracked.visible !== visible;
        const maskChanged = tracked.mask !== mask;
        const materialChanged = withMaterial && tracked.material !== material;
        if (!visibleChanged && !maskChanged && !materialChanged) return false;
        const change: { visible?: boolean; mask?: number; material?: number | null } = {};
        if (visibleChanged) change.visible = visible;
        if (maskChanged) change.mask = mask;
        if (materialChanged) change.material = material ?? null;
        this.trees.setInstance(tracked.handle, change);
        tracked.visible = visible;
        tracked.mask = mask;
        if (materialChanged) tracked.material = material;
        return true;
    }

    private removeTracked(tracked: TrackedInstance): void {
        this.trees.removeInstance(tracked.handle);
        this.instances.delete(tracked.key);
        this.byHandle.delete(tracked.handle);
        tracked.mesh.users--;
    }

    /** Groups the non-static instances by object, with each object's current state (sync() pushed it). */
    private rebuildObjects(): void {
        this.objects.clear();
        const useLayers = this.options.layers ?? true;
        for (const tracked of this.instances.values()) {
            if (tracked.source.static) continue;
            const object = tracked.source.object;
            let entry = this.objects.get(object);
            if (!entry) {
                entry = {
                    object,
                    instances: [],
                    world: Float64Array.from(object.matrixWorld.elements),
                    visible: effectivelyVisible(object),
                    mask: instanceMaskFromLayers(object.layers, useLayers),
                    matrices: matricesVersion(object),
                };
                this.objects.set(object, entry);
            }
            entry.instances.push(tracked);
        }
    }

    private checkLive(): void {
        if (this.disposed) throw new Error('@pmndrs/bvh/three: this SceneTreesObject3DSync was disposed');
    }
}

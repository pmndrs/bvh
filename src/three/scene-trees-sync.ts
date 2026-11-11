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
 * - **Skinned and morphed meshes** (with a `renderer`): a `SkinnedMesh`, or a
 *   mesh with position morph targets, is added as a deforming mesh of its own
 *   (not shared with other objects on its geometry), and each `update()` runs
 *   its `BvhPosePass` (three's skinning and morph math in a TSL compute
 *   pass) and hands the posed vertices to `SceneTrees.setPose`, which refits
 *   its tree on the GPU. Its top-level box follows the skeleton
 *   (`setDeformBounds`, from `BvhPosePass.bounds`, with `deformSlack`
 *   room so it changes rarely). Hidden objects aren't posed. Without a
 *   renderer (or with `deforming: false`) they are read in their rest pose.
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
import type { MeshRefitVertices } from '../gpu-mesh-refit.ts';
import { BvhPosePass, poseVertices, threeDeforms, type BvhPosePassLike } from './deform.ts';
import {
    deviceFromRenderer,
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
    /** A deforming mesh's pose (`SceneTrees.setPose`); needed for skinned and morphed meshes. */
    setPose?(mesh: number, vertices: MeshRefitVertices): void;
    /** A deforming mesh's top-level box (`SceneTrees.setDeformBounds`); needed for skinned and morphed meshes. */
    setDeformBounds?(mesh: number, bounds: ArrayLike<number>): void;
}

/** {@link SceneTreesObject3DSync}'s options: the scene reader's, plus skinned and morphed meshes. */
export interface SceneTreesObject3DSyncOptions extends BvhSceneFromObject3DOptions {
    /**
     * The `WebGPURenderer` (initialized, on the SceneTrees' device) that runs
     * the pose passes of skinned and morphed meshes. Without it they are read
     * in their rest pose, as `bvhSceneFromObject3D` reads them.
     */
    readonly renderer?: unknown;
    /**
     * Which meshes are posed on the GPU each update. Default: with a renderer,
     * every `SkinnedMesh` and every mesh with position morph targets
     * (`threeDeforms`); without one, none. `false` reads every mesh in its rest pose.
     */
    readonly deforming?: boolean | ((mesh: Mesh) => boolean);
    /**
     * Room added on every side of a posed mesh's top-level box, as a fraction
     * of its largest extent, when the pose leaves the box (or the box grows to
     * more than twice the pose's padded size). More room: fewer top-level updates, looser
     * boxes. Default 0.25.
     */
    readonly deformSlack?: number;
    /** Makes a mesh's pose pass. Default `new BvhPosePass(renderer, mesh)`. */
    readonly createPosePass?: (mesh: Mesh) => BvhPosePassLike;
    /** The device for posed meshes' index buffers. Default: the renderer's. */
    readonly device?: GPUDevice;
}

/** What one {@link SceneTreesObject3DSync.pose} posed. */
export interface Object3DPoseCounts {
    /** Pose passes run. */
    readonly posed: number;
    /** Deform boxes moved (once per object, for all its meshes). */
    readonly boundsMoved: number;
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
    /** A posed mesh's pose (its object's), the vertices its refit reads, and their index buffer (indexed geometry). */
    readonly pose: TrackedPose | null;
    readonly vertices: MeshRefitVertices | null;
    readonly indexBuffer: GPUBuffer | null;
}

/** One deforming object's pose pass and the box its meshes have in the top level. */
interface TrackedPose {
    readonly object: Mesh;
    readonly geometry: Mesh['geometry'];
    readonly pass: BvhPosePassLike;
    readonly box: Float32Array;
    readonly meshes: Set<TrackedMesh>;
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

/** `box` holds `needed`, and isn't more than twice the size `needed` gets with its slack on any axis. */
function fits(box: ArrayLike<number>, needed: ArrayLike<number>, slack: number): boolean {
    const pad = slack * Math.max(needed[3] - needed[0], needed[4] - needed[1], needed[5] - needed[2]);
    for (let k = 0; k < 3; k++) {
        if (needed[k] < box[k] || needed[k + 3] > box[k + 3]) return false;
        if (box[k + 3] - box[k] > 2 * (needed[k + 3] - needed[k] + 2 * pad) + 1e-6) return false;
    }
    return true;
}

/** `needed` with `slack` times its largest extent added on every side, into `out`. */
function padBox(needed: ArrayLike<number>, slack: number, out: Float32Array): Float32Array {
    const extent = Math.max(needed[3] - needed[0], needed[4] - needed[1], needed[5] - needed[2]);
    const pad = extent * slack;
    for (let k = 0; k < 3; k++) {
        out[k] = Math.fround(needed[k] - pad);
        out[k + 3] = Math.fround(needed[k + 3] + pad);
    }
    return out;
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
    private readonly poses = new Map<Mesh, TrackedPose>();
    private readonly scratchBox = new Float32Array(6);
    private disposed = false;
    /** What the last {@link pose} posed. */
    poseCounts: Object3DPoseCounts = { posed: 0, boundsMoved: 0 };

    /** Adds every mesh and instance under `root` to `trees` (one {@link sync}). */
    constructor(
        readonly trees: SceneTreesTarget,
        readonly root: Object3D | readonly Object3D[],
        private readonly options: SceneTreesObject3DSyncOptions = {},
    ) {
        if (options.deforming === true && !options.renderer && !options.createPosePass) {
            throw new Error('@pmndrs/bvh/three: deforming: true needs a renderer (SceneTreesObject3DSync options.renderer) to run the pose passes');
        }
        this.sync();
    }

    /** The pose pass of a posed object (its `buffer` holds the last pose), or null. */
    posePassOf(object: Mesh): BvhPosePassLike | null {
        return this.poses.get(object)?.pass ?? null;
    }

    /** Whether `mesh` is posed on the GPU (see {@link SceneTreesObject3DSyncOptions.deforming}). */
    isDeforming(mesh: Mesh): boolean {
        const option = this.options.deforming;
        if (option === false) return false;
        if (!this.options.renderer && !this.options.createPosePass) return false;
        if ((mesh as unknown as { isInstancedMesh?: boolean; isBatchedMesh?: boolean }).isBatchedMesh) return false;
        return threeDeforms(mesh) && (typeof option !== 'function' || option(mesh));
    }

    /** The key of a walked instance's mesh: a posed object's meshes are its own. */
    private meshKeyOf(instance: WalkedInstance): string {
        return this.isDeforming(instance.object) ? `${instance.meshKey}:pose:${instance.object.id}` : instance.meshKey;
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
        const meshKeys = new Map<WalkedInstance, string>(walked.map((instance) => [instance, this.meshKeyOf(instance)]));
        for (const instance of walked) {
            seen.add(instance.key);
            const tracked = this.instances.get(instance.key);
            if (tracked && tracked.mesh.key === meshKeys.get(instance)) {
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
                const meshKey = meshKeys.get(instance)!;
                let mesh = this.meshes.get(meshKey);
                if (mesh && this.trees.meshStatus?.(mesh.handle).mergedInto != null) {
                    // SceneTrees dropped its geometry at the merge: a new instance needs the mesh again.
                    this.meshes.delete(meshKey);
                    mesh = undefined;
                }
                if (!mesh) {
                    mesh = meshKey === instance.meshKey ? this.addPlainMesh(instance) : this.addPosedMesh(instance, meshKey);
                    this.meshes.set(meshKey, mesh);
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
                this.forgetMesh(mesh);
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
        if (this.poses.size > 0) this.pose();
        return { moved, changed, added: 0, removed: 0, meshesAdded: 0, meshesRemoved: 0, skipped: 0 };
    }

    /**
     * Runs the pose pass of every visible posed object (skinned and morphed
     * meshes) and hands its vertices to `SceneTrees.setPose`, moving its
     * top-level box when the pose leaves it. {@link update} calls it last.
     */
    pose(): Object3DPoseCounts {
        this.checkLive();
        let posed = 0;
        let boundsMoved = 0;
        const slack = this.options.deformSlack ?? 0.25;
        for (const pose of this.poses.values()) {
            if (pose.meshes.size === 0 || !effectivelyVisible(pose.object)) continue;
            pose.pass.update();
            posed++;
            const needed = pose.pass.bounds(this.scratchBox);
            if (!fits(pose.box, needed, slack)) {
                padBox(needed, slack, pose.box);
                for (const mesh of pose.meshes) this.trees.setDeformBounds!(mesh.handle, pose.box);
                boundsMoved++;
            }
            for (const mesh of pose.meshes) this.trees.setPose!(mesh.handle, mesh.vertices!);
        }
        this.poseCounts = { posed, boundsMoved };
        return this.poseCounts;
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
        for (const mesh of this.meshes.values()) {
            if (mesh.users === 0) this.trees.removeMesh(mesh.handle);
            mesh.indexBuffer?.destroy();
        }
        for (const pose of this.poses.values()) pose.pass.dispose();
        this.poses.clear();
        this.meshes.clear();
        this.objects.clear();
        this.disposed = true;
    }

    private addPlainMesh(instance: WalkedInstance): TrackedMesh {
        const source = { geometry: instance.geometry, start: instance.range.start, count: instance.range.count };
        return { key: instance.meshKey, handle: this.trees.addMesh(meshGeometryOf(source)), source, users: 0, pose: null, vertices: null, indexBuffer: null };
    }

    /** A posed object's mesh for one range: the pose pass (one per object and geometry), an index buffer, the deform box. */
    private addPosedMesh(instance: WalkedInstance, key: string): TrackedMesh {
        if (!this.trees.setPose || !this.trees.setDeformBounds) throw new Error('@pmndrs/bvh/three: posed meshes need a SceneTrees target with setPose and setDeformBounds');
        const object = instance.object;
        let pose = this.poses.get(object);
        if (pose && pose.geometry !== object.geometry) pose = undefined; // its old meshes go at the end of this sync, and their pass with them
        if (!pose) {
            const pass = this.options.createPosePass ? this.options.createPosePass(object) : new BvhPosePass(this.options.renderer, object);
            pass.update();
            const box = padBox(pass.bounds(new Float32Array(6)), this.options.deformSlack ?? 0.25, new Float32Array(6));
            pose = { object, geometry: object.geometry, pass, box, meshes: new Set() };
            this.poses.set(object, pose);
        }
        const source = { geometry: instance.geometry, start: instance.range.start, count: instance.range.count };
        const geometry = meshGeometryOf(source);
        let indexBuffer: GPUBuffer | null = null;
        if (geometry.indices) {
            const device = this.options.device ?? deviceFromRenderer(this.options.renderer);
            const indices = Uint32Array.from(geometry.indices);
            indexBuffer = device.createBuffer({
                label: `@pmndrs/bvh/three pose indices ${object.name || object.id}`,
                size: Math.max(16, Math.ceil(indices.byteLength / 16) * 16),
                usage: 0x80 | 0x8 | 0x4, // STORAGE | COPY_DST | COPY_SRC (drift rebuilds copy it)
            });
            device.queue.writeBuffer(indexBuffer, 0, indices);
        }
        const handle = this.trees.addMesh(geometry, { deforming: { bounds: pose.box } });
        const mesh: TrackedMesh = { key, handle, source, users: 0, pose, vertices: poseVertices(pose.pass, instance.geometry, source.start, source.count, indexBuffer), indexBuffer };
        pose.meshes.add(mesh);
        return mesh;
    }

    /** A mesh left the SceneTrees: its index buffer goes, and its pose pass with its pose's last mesh. */
    private forgetMesh(mesh: TrackedMesh): void {
        mesh.indexBuffer?.destroy();
        const pose = mesh.pose;
        if (!pose || !pose.meshes.delete(mesh) || pose.meshes.size > 0) return;
        pose.pass.dispose();
        if (this.poses.get(pose.object) === pose) this.poses.delete(pose.object);
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

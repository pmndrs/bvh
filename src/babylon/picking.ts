/**
 * Picking through the core's CPU raycast, with `scene.pick`'s result.
 *
 * ```ts
 * const picker = await BabylonBvhPicker.create(scene);   // loads tinybvh, builds a tree per geometry
 * const info = picker.pick(scene.pointerX, scene.pointerY);   // a PickingInfo, as scene.pick returns
 * // after meshes moved, showed, hid, were added or removed: picker.update();
 * ```
 *
 * **What matches `scene.pick`.** The closest triangle hit along the ray, as a
 * Babylon `PickingInfo`: `hit`, `distance` (world units), `pickedPoint`,
 * `pickedMesh` (the `Mesh` or `InstancedMesh`), `bu` and `bv` (the weights of
 * vertices 0 and 1, as Babylon reports them), `faceId` (the triangle in the mesh's index buffer),
 * `subMeshId` and `subMeshFaceId`, `ray`, and `getNormal()` through them.
 * Babylon tests triangles from both sides, as this does.
 *
 * **Differs from `scene.pick`.**
 *
 * - Babylon picks a mesh with no material by its bounding box (`subMeshId`
 *   -1); this always reads triangles.
 * - A `predicate` is asked once per instance that has a hit candidate, with
 *   the mesh (Babylon's also gets the thin instance index, always -1 here).
 *   Without one, an instance counts as Babylon's default does: enabled,
 *   visible and `isPickable`, as of the last {@link update}.
 * - The picker sees the scene as of the last {@link update}, not live: call
 *   it (it reads every world matrix again) after meshes move, show, hide or
 *   change `isPickable`, and after meshes are added or removed. It builds
 *   trees only for geometry it has not seen, and rebuilds one whose vertex or
 *   index array was replaced; edits in place (`updateVerticesData`) need
 *   `update({ rebuild: true })`.
 * - `fastCheck` is accepted and ignored (the closest hit is returned);
 *   `trianglePredicate` is not supported. Thin instances, skinned and morphed
 *   meshes are read as `bvhSceneFromBabylon` reads them (not at all, and in
 *   their rest pose).
 */

import { Matrix, Vector3 } from '@babylonjs/core/Maths/math.vector.js';
import { PickingInfo } from '@babylonjs/core/Collisions/pickingInfo.js';
// The import also registers Scene.createPickingRay and scene.pick (a side effect of ray.js).
import '@babylonjs/core/Culling/ray.js';
import type { Ray } from '@babylonjs/core/Culling/ray.js';
import type { Camera } from '@babylonjs/core/Cameras/camera.js';
import type { AbstractMesh } from '@babylonjs/core/Meshes/abstractMesh.js';
import type { Mesh } from '@babylonjs/core/Meshes/mesh.js';
import type { Scene } from '@babylonjs/core/scene.js';
import { CpuSceneBvh, type CpuSceneHit } from '../query/cpu-scene.ts';
import { createTinyBvhBuilder, type TinyBvhBuilder } from '../tinybvh.ts';
import { buildMeshBvhWithTinyBvh } from '../trace/mesh-bvh-topology.ts';
import type { MeshBvh } from '../trace/mesh-bvh.ts';
import { BABYLON_MASK_PICKABLE, babylonMeshesOf, bvhSceneFromBabylon, type BvhBabylonScene, type BvhBabylonSource, type BvhSceneFromBabylonOptions } from './scene.ts';

export interface BabylonBvhPickerOptions extends BvhSceneFromBabylonOptions {
    /** The tinybvh builder. Default: `BabylonBvhPicker.create` loads one; the constructor needs it. */
    readonly builder?: TinyBvhBuilder;
}

/** Options of {@link BabylonBvhPicker.pick} and {@link BabylonBvhPicker.pickWithRay}, as `scene.pick`'s arguments. */
export interface BvhPickOptions {
    /** Which meshes count (Babylon's `predicate`). Default: enabled, visible and pickable at the last update. */
    readonly predicate?: (mesh: AbstractMesh) => boolean;
    /** Accepted for `scene.pick`'s signature; the closest hit is returned either way. */
    readonly fastCheck?: boolean;
    /** The camera `pick(x, y)` casts from. Default: the scene's active camera. */
    readonly camera?: Camera | null;
}

/** What a tree was built from: replacing either array (Babylon does on `setVerticesData`) makes a new tree. */
interface BuiltTree {
    readonly positions: ArrayLike<number>;
    readonly indices: ArrayLike<number> | undefined;
    readonly tree: MeshBvh;
}

const scratchPoint = new Vector3();

/** The CPU picker of a Babylon scene; see the file comment. */
export class BabylonBvhPicker {
    private bvh: CpuSceneBvh | null = null;
    private current: BvhBabylonScene = { meshes: [], meshSources: [], instances: [], sources: [] };
    private readonly trees = new Map<Mesh, BuiltTree>();
    private readonly builder: TinyBvhBuilder;

    /** Loads tinybvh (WebAssembly), then builds the picker. */
    static async create(source: Scene | readonly AbstractMesh[], options: BabylonBvhPickerOptions = {}): Promise<BabylonBvhPicker> {
        return new BabylonBvhPicker(source, { ...options, builder: options.builder ?? (await createTinyBvhBuilder()) });
    }

    /** `options.builder` is required here; {@link create} loads one. */
    constructor(
        readonly source: Scene | readonly AbstractMesh[],
        private readonly options: BabylonBvhPickerOptions & { readonly builder: TinyBvhBuilder },
    ) {
        this.builder = options.builder;
        this.update();
    }

    /** The core scene (null for a scene without triangles). */
    get cpuScene(): CpuSceneBvh | null {
        return this.bvh;
    }

    /** The scene as last read: meshes, instances, and the Babylon objects behind them. */
    get scene(): BvhBabylonScene {
        return this.current;
    }

    /** Instances (meshes and `InstancedMesh`es) the picker sees. */
    get instanceCount(): number {
        return this.current.instances.length;
    }

    /** The Babylon object behind instance `instanceIndex` (a hit's `instance`). */
    sourceOf(instanceIndex: number): BvhBabylonSource | null {
        return this.current.sources[instanceIndex] ?? null;
    }

    /**
     * Reads the scene again (world matrices, visibility, `isPickable`, added
     * and removed meshes) and builds trees for geometry it has not seen.
     * `rebuild: true` rebuilds every tree (after editing vertex data in place).
     * Returns whether any mesh was built.
     */
    update(options: { readonly rebuild?: boolean } = {}): boolean {
        const scene = bvhSceneFromBabylon(this.source, this.options);
        let built = false;
        const trees: MeshBvh[] = scene.meshSources.map((meshSource, i) => {
            const geometry = scene.meshes[i];
            const kept = this.trees.get(meshSource.mesh);
            if (kept && !options.rebuild && kept.positions === geometry.positions && kept.indices === geometry.indices) return kept.tree;
            const tree = buildMeshBvhWithTinyBvh(geometry, this.builder);
            this.trees.set(meshSource.mesh, { positions: geometry.positions, indices: geometry.indices ?? undefined, tree });
            built = true;
            return tree;
        });
        const used = new Set(scene.meshSources.map((meshSource) => meshSource.mesh));
        for (const mesh of [...this.trees.keys()]) if (!used.has(mesh)) this.trees.delete(mesh);
        this.current = scene;
        this.bvh = trees.length > 0 ? new CpuSceneBvh(trees, scene.instances) : null;
        return built;
    }

    /**
     * The closest triangle along `ray`, as `scene.pickWithRay` returns it (a
     * `PickingInfo` with `hit` false on a miss). The ray's `length` limits it.
     */
    pickWithRay(ray: Ray, options: BvhPickOptions = {}): PickingInfo {
        const info = new PickingInfo();
        info.ray = ray;
        const bvh = this.bvh;
        if (!bvh) return info;
        const length = Math.hypot(ray.direction.x, ray.direction.y, ray.direction.z);
        if (!(length > 0)) return info;
        const direction: [number, number, number] = [ray.direction.x / length, ray.direction.y / length, ray.direction.z / length];
        const predicate = options.predicate;
        const asked = predicate ? new Map<number, boolean>() : null;
        const [hit] = bvh.raycast(
            { origin: [ray.origin.x, ray.origin.y, ray.origin.z], direction, tMin: 0, tMax: ray.length * length },
            {
                firstHitOnly: true,
                // Babylon's default: enabled, visible, pickable. A predicate replaces it, as in Babylon (every instance is a candidate).
                mask: predicate ? 0xff : BABYLON_MASK_PICKABLE,
                accept: predicate
                    ? (instance) => {
                          let ok = asked!.get(instance);
                          if (ok === undefined) asked!.set(instance, (ok = predicate(this.current.sources[instance].object)));
                          return ok;
                      }
                    : undefined,
            },
        );
        if (!hit) return info;
        return this.fill(info, hit, ray, direction);
    }

    /**
     * `scene.pick(x, y)`: the closest triangle under the pixel (`x`, `y`) of the
     * scene's canvas, cast from `options.camera` or the active camera.
     */
    pick(x: number, y: number, options: BvhPickOptions = {}): PickingInfo {
        const scene = this.sceneOf();
        if (!scene) return new PickingInfo();
        return this.pickWithRay(scene.createPickingRay(x, y, Matrix.Identity(), options.camera ?? null), options);
    }

    private sceneOf(): Scene | null {
        const source = this.source;
        if (!Array.isArray(source)) return source as Scene;
        return babylonMeshesOf(source)[0]?.getScene() ?? null;
    }

    private fill(info: PickingInfo, hit: CpuSceneHit, ray: Ray, direction: readonly [number, number, number]): PickingInfo {
        const source = this.current.sources[hit.instance];
        const mesh = this.current.meshSources[source.mesh].mesh;
        scratchPoint.set(ray.origin.x + direction[0] * hit.t, ray.origin.y + direction[1] * hit.t, ray.origin.z + direction[2] * hit.t);
        info.hit = true;
        info.distance = hit.t;
        info.pickedPoint = scratchPoint.clone();
        info.pickedMesh = source.object;
        // Babylon's bu and bv are the weights of vertices 0 and 1 (the core's u and v are those of vertices 1 and 2).
        info.bu = 1 - hit.u - hit.v;
        info.bv = hit.u;
        info.faceId = hit.triangle;
        info.subMeshFaceId = hit.triangle;
        info.subMeshId = 0;
        // The submesh holding the triangle (Babylon's subMeshFaceId counts from the submesh's first triangle).
        const subMeshes = mesh.subMeshes;
        if (subMeshes && subMeshes.length > 1) {
            const first = hit.triangle * 3;
            for (let i = 0; i < subMeshes.length; i++) {
                const subMesh = subMeshes[i];
                if (first >= subMesh.indexStart && first < subMesh.indexStart + subMesh.indexCount) {
                    info.subMeshId = i;
                    info.subMeshFaceId = hit.triangle - subMesh.indexStart / 3;
                    break;
                }
            }
        }
        return info;
    }
}

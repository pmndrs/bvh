/**
 * A ray against a whole three scene on the CPU, and frustum culling (PARITY
 * 5.2 and 5.6, three-mesh-bvh's `ObjectBVH`): {@link BvhSceneRaycaster} keeps
 * the scene's meshes and instances (`bvhSceneFromObject3D`'s) as the core's
 * `CpuSceneBvh`, and {@link raycastScene} returns what
 * `raycaster.intersectObject(root, true)` returns, from one walk over the
 * scene's top level and its mesh trees instead of a raycast per object.
 *
 * ```ts
 * await prepareBoundsTrees();                         // tinybvh, once
 * const picker = new BvhSceneRaycaster(scene);        // builds a tree per geometry and range
 * // each frame (or when objects moved): scene.updateMatrixWorld(); picker.update();
 * const hits = raycastScene(picker, raycaster);       // three Intersections, sorted
 * const seen = picker.intersectsFrustum(frustum);     // the objects (and instances) whose world box meets it
 * ```
 *
 * **What matches three.** The intersections `Raycaster.intersectObject(root,
 * true)` gives with three's own `Mesh.raycast`: the triangles three tests (draw
 * range, groups, ranges off a multiple of 3; `threeTriangleRanges`), the
 * material's `side`, `near`/`far`, `InstancedMesh` instances (`instanceId`) and
 * active, visible `BatchedMesh` instances (`batchId`), with three's fields
 * (`distance`, `point`, `object`, `face`, `faceIndex`, `barycoord`, `uv`,
 * `uv1`, `normal`). As three's `Raycaster`, hidden objects are hit too
 * (`includeHidden: false` skips them). `raycaster.firstHitOnly` gives the
 * closest hit only (one, over the whole scene, as three-mesh-bvh's `ObjectBVH`).
 *
 * **Differs from three.** Layers: three tests all 32 layers, the walk's
 * instance mask holds layers 0-7 (`rayMaskFromLayers`), so an object only on
 * layers 8-31 is never hit (`layers: false` in the options ignores layers on
 * both sides). Skinned and morphed meshes are read in their rest pose (three
 * poses them on the CPU; use their own raycast, or the GPU path). Points,
 * lines and sprites are not read.
 *
 * **Moving objects.** A raycast sees the scene as of the last
 * {@link BvhSceneRaycaster.update}: after moving, showing, hiding, adding or
 * removing objects, changing instance matrices or counts, or editing a
 * geometry's positions or index (with `needsUpdate`), call
 * `scene.updateMatrixWorld()` and then `update()`. It reads the scene again:
 * when only transforms, visibility, layers or instance matrices changed it
 * rebuilds the top level only; new geometry or ranges get their trees (each
 * geometry and range is built once and kept while used), and a geometry whose
 * position attribute or index changed (object or version) gets its tree
 * rebuilt. A raycast that finds a geometry edited since `update()` warns once
 * (it still answers from the last `update()`); moved objects aren't detected.
 */

import { BackSide, type BufferGeometry, DoubleSide, FrontSide, Vector3, type BatchedMesh, type Frustum, type Intersection, type Material, type Mesh, type Object3D, type Raycaster } from 'three';
import { CpuSceneBvh, type CpuSceneHit } from '../query/cpu-scene.ts';
import type { TinyBvhBuilder } from '../tinybvh.ts';
import { buildMeshBvhWithTinyBvh } from '../trace/mesh-bvh-topology.ts';
import type { MeshBvh } from '../trace/mesh-bvh.ts';
import type { TraceSceneInstance } from '../trace/scene-assembly.ts';
import { invertAffineInto } from '../trace/top-level.ts';
import { intersectionFields, positionVersionOf, preparedBoundsTreeBuilder } from './raycast.ts';
import { meshGeometryOf, rayMaskFromLayers, threeFaceIndex, walkObject3D, type BvhSceneFromObject3DOptions, type BvhThreeMeshSource, type BvhThreeSource } from './scene.ts';

export interface BvhSceneRaycasterOptions extends BvhSceneFromObject3DOptions {
    /** The tinybvh builder; default the one `prepareBoundsTrees` loaded. */
    readonly builder?: TinyBvhBuilder;
    /** Hit hidden objects too, as three's `Raycaster` does. Default true; false skips objects that aren't `visible` with every ancestor. */
    readonly includeHidden?: boolean;
}

/** One instance of the scene: its three source and mesh range. */
interface SceneEntry {
    readonly source: BvhThreeSource;
    readonly meshSource: BvhThreeMeshSource;
    readonly meshKey: string;
    readonly objectToWorld: Float64Array;
}

const scratchRows = new Float64Array(12);

/** What a mesh's tree was built from: the position attribute and index, and their versions. */
interface MeshVersions {
    readonly geometry: BufferGeometry;
    readonly position: object | undefined;
    readonly positionVersion: number;
    readonly index: object | null;
    readonly indexVersion: number;
}

function versionsOf(geometry: BufferGeometry): MeshVersions {
    const position = geometry.getAttribute('position') as Parameters<typeof positionVersionOf>[0] | undefined;
    const index = geometry.index;
    return { geometry, position, positionVersion: position ? positionVersionOf(position) : 0, index, indexVersion: index ? index.version : 0 };
}

/** Whether the geometry still has the position attribute and index (and versions) `versions` saw. */
function current(versions: MeshVersions): boolean {
    const geometry = versions.geometry;
    const position = geometry.getAttribute('position') as Parameters<typeof positionVersionOf>[0] | undefined;
    const index = geometry.index;
    return position === versions.position && (position ? positionVersionOf(position) : 0) === versions.positionVersion && index === versions.index && (index ? index.version : 0) === versions.indexVersion;
}

/** Small ids for attribute objects, for tree keys. */
const objectIds = new WeakMap<object, number>();
let nextObjectId = 1;
function objectId(object: object | null | undefined): number {
    if (!object) return 0;
    let id = objectIds.get(object);
    if (id === undefined) objectIds.set(object, (id = nextObjectId++));
    return id;
}

/** The CPU scene of a three scene; see the file comment. */
export class BvhSceneRaycaster {
    private bvh: CpuSceneBvh | null = null;
    private entries: SceneEntry[] = [];
    private meshKeys: string[] = [];
    /** Per mesh, what its tree was built from: a raycast compares it to the geometry. */
    private meshVersions: MeshVersions[] = [];
    private staleWarned = false;
    private readonly trees = new Map<string, MeshBvh>();
    private readonly builder: TinyBvhBuilder;

    constructor(
        readonly root: Object3D | readonly Object3D[],
        private readonly options: BvhSceneRaycasterOptions = {},
    ) {
        const builder = options.builder ?? preparedBoundsTreeBuilder();
        if (!builder) throw new Error('@pmndrs/bvh/three: call `await prepareBoundsTrees()` once before new BvhSceneRaycaster (it loads tinybvh)');
        this.builder = builder;
        this.update();
    }

    /** The core scene (null for a scene without meshes). */
    get cpuScene(): CpuSceneBvh | null {
        return this.bvh;
    }

    /** The three object behind instance `instanceIndex` (a hit's `instance`). */
    sourceOf(instanceIndex: number): BvhThreeSource | null {
        return this.entries[instanceIndex]?.source ?? null;
    }

    /** Instances (objects, instances of instanced and batched meshes, and groups) in the scene. */
    get instanceCount(): number {
        return this.entries.length;
    }

    /**
     * Reads the scene again (call `updateMatrixWorld()` first): needed after
     * objects move or change and after geometry edits (see the file comment).
     * Rebuilds only the top level when the meshes are the same; builds trees
     * for new geometry or ranges, or edited positions or index, and drops
     * unused ones. Returns whether meshes changed.
     */
    update(): boolean {
        const includeHidden = this.options.includeHidden ?? true;
        const walked = walkObject3D(this.root, this.options);
        const meshIndex = new Map<string, number>();
        const meshKeys: string[] = [];
        const meshSources: BvhThreeMeshSource[] = [];
        const meshVersions: MeshVersions[] = [];
        const instances: TraceSceneInstance[] = [];
        const entries: SceneEntry[] = [];
        for (const instance of walked) {
            let mesh = meshIndex.get(instance.meshKey);
            if (mesh === undefined) {
                mesh = meshKeys.length;
                meshIndex.set(instance.meshKey, mesh);
                // The tree's key: the geometry and range, and what its positions and index are now.
                const versions = versionsOf(instance.geometry);
                meshKeys.push(`${instance.meshKey}|${objectId(versions.position)}.${versions.positionVersion}|${objectId(versions.index)}.${versions.indexVersion}`);
                meshVersions.push(versions);
                meshSources.push({ geometry: instance.geometry, start: instance.range.start, count: instance.range.count });
            }
            const batched = instance.object as unknown as BatchedMesh;
            // three's Raycaster ignores `visible`; a BatchedMesh skips its hidden instances.
            const visible = includeHidden ? (batched.isBatchedMesh ? batched.getVisibleAt(instance.instanceId) : true) : instance.visible;
            const instanceIndex = instances.length;
            instances.push({ meshIndex: mesh, instanceIndex, objectToWorld: instance.objectToWorld, visible, mask: instance.mask, material: instance.material });
            entries.push({
                source: { object: instance.object, instanceId: instance.instanceId, group: instance.range.group, materialIndex: instance.range.materialIndex, mesh, static: instance.static },
                meshSource: meshSources[mesh],
                meshKey: instance.meshKey,
                objectToWorld: instance.objectToWorld,
            });
        }
        const sameMeshes = this.bvh !== null && meshKeys.length === this.meshKeys.length && meshKeys.every((key, i) => key === this.meshKeys[i]);
        this.entries = entries;
        this.meshVersions = meshVersions;
        this.staleWarned = false;
        if (sameMeshes) {
            this.bvh!.setInstances(instances);
            return false;
        }
        const trees = meshKeys.map((key, i) => {
            let tree = this.trees.get(key);
            if (!tree) this.trees.set(key, (tree = buildMeshBvhWithTinyBvh(meshGeometryOf(meshSources[i]), this.builder)));
            return tree;
        });
        const used = new Set(meshKeys);
        for (const key of [...this.trees.keys()]) if (!used.has(key)) this.trees.delete(key);
        this.meshKeys = meshKeys;
        this.bvh = trees.length > 0 ? new CpuSceneBvh(trees, instances) : null;
        return true;
    }

    /**
     * What `raycaster.intersectObject(root, true)` returns (see the file
     * comment), appended to `intersects` and sorted by distance.
     */
    raycast(raycaster: Raycaster, intersects: Intersection[] = []): Intersection[] {
        const bvh = this.bvh;
        if (!bvh) return intersects;
        if (!this.staleWarned) this.warnIfStale();
        const { origin, direction } = raycaster.ray;
        const length = direction.length();
        if (!(length > 0)) return intersects;
        const firstHitOnly = (raycaster as Raycaster & { firstHitOnly?: boolean }).firstHitOnly === true;
        const unit = direction.clone().divideScalar(length);
        const hits = bvh.raycast(
            { origin: [origin.x, origin.y, origin.z], direction: [unit.x, unit.y, unit.z], tMin: raycaster.near, tMax: raycaster.far },
            {
                firstHitOnly,
                mask: (this.options.layers ?? true) ? rayMaskFromLayers(raycaster.layers) : 0xff,
                accept: (instance, _triangle, frontFace) => {
                    const side = this.materialOf(this.entries[instance].source)?.side ?? FrontSide;
                    if (side === FrontSide) return frontFace;
                    if (side === BackSide) return !frontFace;
                    return side === DoubleSide;
                },
            },
        );
        const start = intersects.length;
        for (const hit of hits) intersects.push(this.intersectionOf(hit, origin, unit));
        if (start > 0) intersects.sort((a, b) => a.distance - b.distance);
        return intersects;
    }

    /**
     * The sources (object, instance, group) whose world box meets `frustum`
     * (three's `Frustum.intersectsBox` test, on the boxes the walk uses), each
     * object or instance once. `layers` (a camera's) limits them to objects on
     * its layers 0-7.
     */
    intersectsFrustum(frustum: Frustum, layers?: { readonly mask: number }): BvhThreeSource[] {
        const bvh = this.bvh;
        if (!bvh) return [];
        const planes = new Float64Array(frustum.planes.length * 4);
        frustum.planes.forEach((plane, p) => planes.set([plane.normal.x, plane.normal.y, plane.normal.z, plane.constant], p * 4));
        const seen = new Set<string>();
        const out: BvhThreeSource[] = [];
        for (const instance of bvh.intersectsFrustum(planes, layers ? { mask: layers.mask & 0xff } : {})) {
            const source = this.entries[instance].source;
            const key = `${source.object.id}:${source.instanceId}`;
            if (seen.has(key)) continue;
            seen.add(key);
            out.push(source);
        }
        return out;
    }

    /** Warns once (until the next update) when a geometry's positions or index changed since update(). */
    private warnIfStale(): void {
        for (const versions of this.meshVersions) {
            if (current(versions)) continue;
            this.staleWarned = true;
            console.warn(
                `@pmndrs/bvh/three: geometry "${versions.geometry.name || versions.geometry.uuid}" changed (positions or index) since BvhSceneRaycaster.update(); raycasts read it as it was until update() runs again`,
            );
            return;
        }
    }

    private materialOf(source: BvhThreeSource): Material | null {
        const material = source.object.material;
        return Array.isArray(material) ? (material[source.materialIndex] ?? null) : (material ?? null);
    }

    private intersectionOf(hit: CpuSceneHit, origin: Vector3, unit: Vector3): Intersection {
        const entry = this.entries[hit.instance];
        const { source, meshSource } = entry;
        const geometry: BufferGeometry = meshSource.geometry;
        // The ray in the instance's space, for the interpolated normal's facing (as three's local ray).
        invertAffineInto(entry.objectToWorld, scratchRows);
        const m = scratchRows;
        const localDirection = new Vector3(m[0] * unit.x + m[1] * unit.y + m[2] * unit.z, m[4] * unit.x + m[5] * unit.y + m[6] * unit.z, m[8] * unit.x + m[9] * unit.y + m[10] * unit.z);
        const faceIndex = threeFaceIndex(meshSource, hit.triangle);
        const firstIndex = meshSource.start + hit.triangle * 3;
        const intersection: Intersection = {
            distance: hit.t,
            point: origin.clone().addScaledVector(unit, hit.t),
            object: source.object as Mesh,
            ...intersectionFields(geometry, faceIndex, firstIndex, hit.u, hit.v, localDirection, source.group >= 0 ? source.materialIndex : 0),
        };
        if (source.instanceId >= 0) {
            if ((source.object as unknown as BatchedMesh).isBatchedMesh) (intersection as Intersection & { batchId?: number }).batchId = source.instanceId;
            else intersection.instanceId = source.instanceId;
        }
        return intersection;
    }
}

/** `raycaster.intersectObject(root, true)` over a {@link BvhSceneRaycaster}: one walk over the whole scene. */
export function raycastScene(scene: BvhSceneRaycaster, raycaster: Raycaster, intersects: Intersection[] = []): Intersection[] {
    return scene.raycast(raycaster, intersects);
}

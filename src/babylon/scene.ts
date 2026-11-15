/**
 * Babylon meshes in, the core's inputs out: the meshes and instances
 * `buildTraceScene`, `SceneTrees` and `CpuSceneBvh` take, plus the Babylon
 * object behind every instance so a hit can be mapped back.
 *
 * What is read:
 *
 * - **Meshes.** Every `Mesh` and `InstancedMesh` (the scene's `meshes`, or the
 *   list given) with triangles: the whole index buffer (`getIndices()`; none
 *   means a plain triangle list) and `position` (`getVerticesData`, so strided
 *   and normalized buffers work). One mesh per geometry, shared by every
 *   `Mesh` and `InstancedMesh` that draws it. Lines, points (`fillMode`
 *   other than triangles), meshes with no positions and thin instances are
 *   skipped; skinning and morph targets are not applied (the rest pose).
 * - **Instances.** One per mesh and per `InstancedMesh`, with its world matrix
 *   (`computeWorldMatrix(true)`). Babylon's matrices are row-vector, and their
 *   memory is the column-major layout of the same transform, so the 16 floats
 *   go in as they are.
 * - **Visibility.** `visible` is `isEnabled() && isVisible`, as Babylon's
 *   picking and rendering use them. Mask bit 0 (`BABYLON_MASK_VISIBLE`) is set
 *   on every instance; bit 1 (`BABYLON_MASK_PICKABLE`) on those with
 *   `isPickable`, which {@link BabylonBvhPicker} ray-masks on, so shadow
 *   rays and `bvhTraceClosest` (all ones) still see unpickable meshes.
 * - **Materials.** Optional: `material(mesh)` returns an index (0-65534) for
 *   `BvhHit.material`.
 *
 * Nothing here needs an engine or a GPU: it runs in Node (with a `NullEngine`).
 */

import type { AbstractMesh } from '@babylonjs/core/Meshes/abstractMesh.js';
import type { InstancedMesh } from '@babylonjs/core/Meshes/instancedMesh.js';
import type { Mesh } from '@babylonjs/core/Meshes/mesh.js';
import type { Scene } from '@babylonjs/core/scene.js';
import { MAX_MATERIAL } from '../gpu-layout.ts';
import type { MeshGeometry } from '../trace/mesh-bvh.ts';
import type { TraceSceneInstance } from '../trace/scene-assembly.ts';

/** Instance mask bit 0: every instance has it. */
export const BABYLON_MASK_VISIBLE = 0x01;
/** Instance mask bit 1: the mesh `isPickable`. Picking rays carry it. */
export const BABYLON_MASK_PICKABLE = 0x02;

export interface BvhSceneFromBabylonOptions {
    /** Which meshes to include. Default: every mesh with triangles. */
    readonly filter?: (mesh: AbstractMesh) => boolean;
    /** The material index (0-65534) of a mesh's triangles, reported as `BvhHit.material`. Default: none. */
    readonly material?: (mesh: AbstractMesh) => number | undefined;
    /**
     * Instances that never move (`TraceSceneInstance.static`, for
     * `mergeStaticInstances`). `true` for all, or a function of the mesh. Default false.
     */
    readonly static?: boolean | ((mesh: AbstractMesh) => boolean);
}

/** The Babylon geometry behind one scene mesh. */
export interface BvhBabylonMeshSource {
    /** The `Mesh` whose vertex data this is (an `InstancedMesh`'s `sourceMesh`). */
    readonly mesh: Mesh;
    /** Triangles in the mesh (the whole index buffer, or every three vertices). */
    readonly triangleCount: number;
}

/** The Babylon object behind one scene instance. */
export interface BvhBabylonSource {
    /** The mesh or `InstancedMesh` itself (what `scene.pick` would return as `pickedMesh`). */
    readonly object: AbstractMesh;
    /** Index into {@link BvhBabylonScene.meshes}. */
    readonly mesh: number;
}

/** `buildTraceScene` / `SceneTrees` / `CpuSceneBvh` inputs, with the Babylon objects behind them. */
export interface BvhBabylonScene {
    /** One per geometry. Positions are the vertex buffer's own array when it is tightly packed floats (not copied). */
    readonly meshes: readonly MeshGeometry[];
    readonly meshSources: readonly BvhBabylonMeshSource[];
    /** `instanceIndex` is the instance's position here and in {@link sources}. */
    readonly instances: readonly TraceSceneInstance[];
    readonly sources: readonly BvhBabylonSource[];
}

function isInstanced(mesh: AbstractMesh): mesh is InstancedMesh {
    return (mesh as InstancedMesh).sourceMesh !== undefined;
}

/** Whether Babylon would draw (and so pick) this as triangles. */
function hasTriangles(mesh: AbstractMesh): boolean {
    const className = mesh.getClassName();
    if (className.includes('Lines') || className === 'GreasedLineMesh') return false;
    const fillMode = mesh.material?.fillMode;
    return fillMode === undefined || fillMode === 0;
}

/**
 * Babylon often keeps vertex data as plain arrays (`VertexData.positions = [...]`), which are
 * converted once per array here, so the same array gives the same typed array every time
 * (the picker tells replaced vertex data from unchanged by that identity).
 */
const converted = new WeakMap<object, Float32Array | Uint32Array>();

/** Tightly packed `x, y, z` floats of a mesh, or null without positions. */
function positionsOfMesh(mesh: Mesh): Float32Array | null {
    const data = mesh.getVerticesData('position');
    if (!data || data.length < 9) return null;
    if (data instanceof Float32Array) return data;
    let floats = converted.get(data) as Float32Array | undefined;
    if (!floats) converted.set(data, (floats = Float32Array.from(data)));
    return floats;
}

/** The mesh's index buffer, or null for a plain triangle list (Babylon gives an empty array). */
function indicesOfMesh(mesh: Mesh): Uint16Array | Uint32Array | null {
    const data = mesh.getIndices();
    if (!data || data.length === 0) return null;
    if (data instanceof Uint16Array || data instanceof Uint32Array) return data;
    let words = converted.get(data) as Uint32Array | undefined;
    if (!words) converted.set(data, (words = Uint32Array.from(data)));
    return words;
}

/** The mesh's geometry as the core takes it (what {@link bvhSceneFromBabylon} reads), or null without triangles. */
export function meshGeometryFromBabylon(mesh: Mesh): MeshGeometry | null {
    const positions = positionsOfMesh(mesh);
    if (!positions) return null;
    const indices = indicesOfMesh(mesh);
    if (indices && indices.length < 3) return null;
    return indices ? { positions, indices } : { positions };
}

/** The meshes of a scene or list, instances after their source meshes are not special: every `AbstractMesh` in order. */
export function babylonMeshesOf(source: Scene | readonly AbstractMesh[]): readonly AbstractMesh[] {
    return Array.isArray(source) ? (source as readonly AbstractMesh[]) : (source as Scene).meshes;
}

/**
 * A Babylon scene (or a list of meshes) as the core's meshes and instances;
 * see the file comment. Call it after the world matrices are right
 * (`computeWorldMatrix(true)` is called on each mesh here).
 */
export function bvhSceneFromBabylon(source: Scene | readonly AbstractMesh[], options: BvhSceneFromBabylonOptions = {}): BvhBabylonScene {
    const meshes: MeshGeometry[] = [];
    const meshSources: BvhBabylonMeshSource[] = [];
    const meshIndexOf = new Map<Mesh, number>();
    const instances: TraceSceneInstance[] = [];
    const sources: BvhBabylonSource[] = [];

    for (const object of babylonMeshesOf(source)) {
        const mesh = (isInstanced(object) ? object.sourceMesh : object) as Mesh;
        if (!mesh || typeof mesh.getVerticesData !== 'function' || !hasTriangles(object)) continue;
        if (options.filter && !options.filter(object)) continue;
        let meshIndex = meshIndexOf.get(mesh);
        if (meshIndex === undefined) {
            const geometry = meshGeometryFromBabylon(mesh);
            // Remembered as skipped, so a mesh with no triangles is read once.
            if (!geometry) {
                meshIndexOf.set(mesh, -1);
                continue;
            }
            meshIndex = meshes.length;
            meshIndexOf.set(mesh, meshIndex);
            meshes.push(geometry);
            meshSources.push({ mesh, triangleCount: geometry.indices ? Math.floor(geometry.indices.length / 3) : Math.floor(geometry.positions.length / 9) });
        }
        if (meshIndex < 0) continue;

        let material: number | undefined;
        if (options.material) {
            material = options.material(object);
            if (material !== undefined && !(Number.isInteger(material) && material >= 0 && material <= MAX_MATERIAL)) {
                throw new RangeError(`bvhSceneFromBabylon: material(${object.name}) returned ${material}; a material is an integer in 0-${MAX_MATERIAL}`);
            }
        }
        const world = object.computeWorldMatrix(true);
        const instanceIndex = instances.length;
        instances.push({
            meshIndex,
            instanceIndex,
            objectToWorld: Array.from(world.m),
            visible: object.isEnabled() && object.isVisible,
            mask: BABYLON_MASK_VISIBLE | (object.isPickable ? BABYLON_MASK_PICKABLE : 0),
            ...(material === undefined ? {} : { material }),
            ...(typeof options.static === 'function' ? { static: options.static(object) } : options.static ? { static: true } : {}),
        });
        sources.push({ object, mesh: meshIndex });
    }
    return { meshes, meshSources, instances, sources };
}

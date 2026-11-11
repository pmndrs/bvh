/**
 * three.js objects in, engine-free scene inputs out: the meshes and instances
 * `buildTraceScene` and `SceneTrees` take, plus the three object behind every
 * instance so a hit can be mapped back.
 *
 * What is read, and how it matches three:
 *
 * - **Objects.** Every `Mesh` under the root (`InstancedMesh` and
 *   `BatchedMesh` included, `SkinnedMesh` in its rest pose; morph targets and
 *   skinning are not applied). Points, lines and sprites are skipped.
 * - **Triangles.** The ones three's own `Mesh.raycast` tests: the draw range,
 *   and with an array of materials the geometry groups within it (a group
 *   whose material is missing is skipped, and an array of materials without
 *   groups has no triangles, as in three). Indexed and non-indexed geometry;
 *   positions are read through the attribute, so interleaved, normalized and
 *   non-float positions work.
 * - **Meshes.** One mesh per geometry and triangle range, shared by every
 *   object (and every instance) that uses it. With an array of materials, each
 *   group is its own mesh, so each can carry its own material index.
 * - **Instances.** One per `Mesh`, per instance of an `InstancedMesh` (below
 *   its `count`: `matrixWorld x instanceMatrix`), and per active instance of a
 *   `BatchedMesh` (`matrixWorld x getMatrixAt(i)`, its geometry's range). The
 *   world matrices are read as they are: call `updateMatrixWorld()` first.
 * - **Visibility.** An object is visible when it and every ancestor is
 *   (`visible`), and a `BatchedMesh` instance also when `getVisibleAt(i)`.
 *   Hidden objects are kept as hidden instances (`visible: false`), so showing
 *   one again needs no new tree.
 * - **Layers to instance masks.** three's 32 layers become the 8-bit instance
 *   mask: layers 0-7 are mask bits 0-7, and layers 8-31 are dropped. An object
 *   on none of layers 0-7 gets mask 0, which no ray hits. A default object is on
 *   layer 0 (mask `0x01`), which the plain `bvhTraceClosest`/`bvhTraceAny`
 *   (all ones) and a ray mask from a default camera or raycaster
 *   ({@link rayMaskFromLayers}) both hit. `layers: false` gives every instance
 *   `0xff` instead.
 * - **Materials.** Optional: `material(object, info)` returns an index
 *   (0-65534) per instance and group, reported as `BvhHit.material`.
 * - **Static.** `static` (true, or a function of the object and instance)
 *   marks instances that never move, for `mergeStaticInstances`, and the
 *   SceneTrees sync never pushes their transforms.
 *
 * Nothing here needs a renderer or a GPU: it runs in Node.
 */

import { Matrix4, type BatchedMesh, type BufferAttribute, type BufferGeometry, type InstancedMesh, type InterleavedBufferAttribute, type Layers, type Material, type Mesh, type Object3D } from 'three';
import { MAX_MATERIAL } from '../trace/layout.ts';
import type { MeshGeometry } from '../trace/mesh-bvh.ts';
import type { TraceSceneInstance } from '../trace/scene-assembly.ts';

/** What {@link BvhSceneFromObject3DOptions.material} is told about the instance it names a material for. */
export interface BvhThreeMaterialInfo {
    /** The instance of an `InstancedMesh` or `BatchedMesh`, or -1 for a plain mesh. */
    readonly instanceId: number;
    /** The geometry group, or -1 when the mesh has one material (draw range only). */
    readonly group: number;
    /** The group's `materialIndex` into the mesh's array of materials, or -1. */
    readonly materialIndex: number;
    /** The three material that draws these triangles (the group's, or the mesh's). */
    readonly threeMaterial: Material | null;
}

export interface BvhSceneFromObject3DOptions {
    /** Which meshes to include. Default: every mesh. */
    readonly filter?: (mesh: Mesh) => boolean;
    /**
     * Instances that never move (`TraceSceneInstance.static`, for
     * `mergeStaticInstances`). `true` for all, or a function of the object and
     * instance (-1 for a plain mesh). Default false.
     */
    readonly static?: boolean | ((object: Mesh, instanceId: number) => boolean);
    /** Map three's layers 0-7 to the instance mask (see the file comment). Default true; false: every instance `0xff`. */
    readonly layers?: boolean;
    /** The material index (0-65534) of an instance's triangles, reported as `BvhHit.material`. Default: none. */
    readonly material?: (mesh: Mesh, info: BvhThreeMaterialInfo) => number | undefined;
}

/** The three object behind one instance. `BvhHit.instance` (and `CpuHit.instanceIndex`) indexes these. */
export interface BvhThreeSource {
    readonly object: Mesh;
    /** The instance of an `InstancedMesh` or `BatchedMesh`, or -1 for a plain mesh. */
    readonly instanceId: number;
    /** The geometry group, or -1. */
    readonly group: number;
    /** The group's `materialIndex`, or -1. */
    readonly materialIndex: number;
    /** Position of this instance's mesh in {@link BvhThreeScene.meshes}. */
    readonly mesh: number;
    readonly static: boolean;
}

/** Where one of {@link BvhThreeScene.meshes} came from. */
export interface BvhThreeMeshSource {
    readonly geometry: BufferGeometry;
    /** First index (vertex, for non-indexed geometry) of the range, as three's `drawRange.start` and `group.start`. */
    readonly start: number;
    /** Indices (vertices) in the range, a multiple of 3. */
    readonly count: number;
}

/** `buildTraceScene` / `SceneTrees` inputs, with the three objects behind them. */
export interface BvhThreeScene {
    /** One per geometry and triangle range. Positions may be the geometry's own array (not copied). */
    readonly meshes: readonly MeshGeometry[];
    readonly meshSources: readonly BvhThreeMeshSource[];
    /** `instanceIndex` is the instance's position here and in {@link sources}. */
    readonly instances: readonly TraceSceneInstance[];
    readonly sources: readonly BvhThreeSource[];
}

/** Converts a hit's triangle (its index within its mesh) to three's `faceIndex` on the source geometry. */
export function threeFaceIndex(meshSource: BvhThreeMeshSource, triangle: number): number {
    return Math.floor((meshSource.start + triangle * 3) / 3);
}

/** A ray mask from three's layers (a camera's or a raycaster's): layers 0-7 as bits 0-7. */
export function rayMaskFromLayers(layers: Layers): number {
    return layers.mask & 0xff;
}

/** The instance mask an object gets: its layers 0-7, or `0xff` with `layers: false`. */
export function instanceMaskFromLayers(layers: Layers, useLayers = true): number {
    return useLayers ? layers.mask & 0xff : 0xff;
}

/** The device of an initialized three `WebGPURenderer` (`renderer.backend.device`); throws for a WebGL backend or before `init()`. */
export function deviceFromRenderer(renderer: unknown): GPUDevice {
    const backend = (renderer as { backend?: { isWebGPUBackend?: boolean; device?: GPUDevice | null } } | null)?.backend;
    if (!backend?.isWebGPUBackend) throw new Error('@pmndrs/bvh/three: the renderer is not a WebGPURenderer on its WebGPU backend');
    if (!backend.device) throw new Error('@pmndrs/bvh/three: the renderer has no device yet; await renderer.init() first');
    return backend.device;
}

/** Like {@link deviceFromRenderer}, after awaiting `renderer.init()`. */
export async function deviceFromRendererAsync(renderer: { init(): Promise<unknown> }): Promise<GPUDevice> {
    await renderer.init();
    return deviceFromRenderer(renderer);
}

// ---------------------------------------------------------------------------

/** `x, y, z` per vertex, tightly packed: the attribute's own array when it already is. */
export function positionsOf(attribute: BufferAttribute | InterleavedBufferAttribute, start = 0, count = attribute.count): Float32Array {
    const interleaved = (attribute as InterleavedBufferAttribute).isInterleavedBufferAttribute === true;
    const plain = attribute as BufferAttribute;
    if (!interleaved && plain.array instanceof Float32Array && plain.itemSize === 3 && !plain.normalized) {
        return start === 0 && count === attribute.count ? plain.array : plain.array.subarray(start * 3, (start + count) * 3);
    }
    const out = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        out[i * 3] = attribute.getX(start + i);
        out[i * 3 + 1] = attribute.getY(start + i);
        out[i * 3 + 2] = attribute.getZ(start + i);
    }
    return out;
}

interface TriangleRange {
    readonly start: number;
    readonly count: number;
    readonly group: number;
    readonly materialIndex: number;
    readonly threeMaterial: Material | null;
}

/**
 * The triangle ranges three's `Mesh.raycast` tests (`_computeIntersections`):
 * the draw range, or with an array of materials each group within it.
 */
export function threeTriangleRanges(
    geometry: BufferGeometry,
    material: Material | Material[] | null | undefined,
    drawRange: { readonly start: number; readonly count: number } = geometry.drawRange,
): TriangleRange[] {
    const position = geometry.getAttribute('position');
    if (!position) return [];
    const total = geometry.index ? geometry.index.count : position.count;
    const drawStart = Math.max(0, drawRange.start);
    const drawEnd = Math.min(total, drawRange.start + drawRange.count);
    const whole = (start: number, end: number) => Math.max(0, Math.floor((end - start) / 3) * 3);
    if (Array.isArray(material)) {
        const ranges: TriangleRange[] = [];
        geometry.groups.forEach((group, index) => {
            const groupMaterial = material[group.materialIndex ?? 0];
            if (!groupMaterial) return;
            const start = Math.max(group.start, drawStart);
            const end = Math.min(total, group.start + group.count, drawEnd);
            const count = whole(start, end);
            if (count > 0) ranges.push({ start, count, group: index, materialIndex: group.materialIndex ?? 0, threeMaterial: groupMaterial });
        });
        return ranges;
    }
    const count = whole(drawStart, drawEnd);
    return count > 0 ? [{ start: drawStart, count, group: -1, materialIndex: -1, threeMaterial: material ?? null }] : [];
}

/** One range of a geometry as a mesh input. */
function meshInputOf(geometry: BufferGeometry, start: number, count: number): MeshGeometry {
    const position = geometry.getAttribute('position') as BufferAttribute | InterleavedBufferAttribute;
    const index = geometry.index;
    if (!index) return { positions: positionsOf(position, start, count), indices: null };
    const array = index.array;
    let indices: Uint16Array | Uint32Array;
    if (array instanceof Uint16Array || array instanceof Uint32Array) {
        indices = start === 0 && count === array.length ? array : array.subarray(start, start + count);
    } else {
        indices = Uint32Array.from({ length: count }, (_, i) => index.getX(start + i));
    }
    return { positions: positionsOf(position), indices };
}

/** Visible, with every ancestor. */
export function effectivelyVisible(object: Object3D): boolean {
    for (let node: Object3D | null = object; node; node = node.parent) if (!node.visible) return false;
    return true;
}

/** One instance as the walk found it, before it gets an index. */
export interface WalkedInstance {
    readonly key: string;
    readonly object: Mesh;
    readonly instanceId: number;
    readonly range: TriangleRange;
    readonly meshKey: string;
    readonly geometry: BufferGeometry;
    readonly objectToWorld: Float64Array;
    readonly visible: boolean;
    readonly mask: number;
    readonly material: number | undefined;
    readonly static: boolean;
}

const scratch = new Float64Array(16);

/** `out = a x b`, column-major. */
export function multiplyMatrices(a: ArrayLike<number>, b: ArrayLike<number>, out: Float64Array): Float64Array {
    for (let column = 0; column < 4; column++) {
        for (let row = 0; row < 4; row++) {
            let sum = 0;
            for (let k = 0; k < 4; k++) sum += a[k * 4 + row] * b[column * 4 + k];
            scratch[column * 4 + row] = sum;
        }
    }
    out.set(scratch);
    return out;
}

function checkMaterial(material: number | undefined, object: Mesh): number | undefined {
    if (material === undefined) return undefined;
    if (!Number.isInteger(material) || material < 0 || material > MAX_MATERIAL) {
        throw new RangeError(`@pmndrs/bvh/three: material for "${object.name || object.type}" must be an integer in 0-${MAX_MATERIAL}, got ${material}`);
    }
    return material;
}

interface InstanceRecordInput {
    readonly objectToWorld: Float64Array;
    readonly visible: boolean;
}

/** The matrices (and BatchedMesh visibilities) of a mesh's instances; one entry for a plain mesh. */
export function instancesOf(object: Mesh, out: (instanceId: number, record: InstanceRecordInput, geometryRange: { start: number; count: number } | null) => void): void {
    const world = object.matrixWorld.elements;
    const instanced = object as unknown as InstancedMesh;
    const batched = object as unknown as BatchedMesh;
    if (instanced.isInstancedMesh) {
        const matrices = instanced.instanceMatrix.array;
        for (let i = 0; i < instanced.count; i++) {
            out(i, { objectToWorld: multiplyMatrices(world, matrices.subarray(i * 16, i * 16 + 16), new Float64Array(16)), visible: true }, null);
        }
    } else if (batched.isBatchedMesh) {
        const info = (batched as unknown as { _instanceInfo: { active: boolean; visible: boolean; geometryIndex: number }[] })._instanceInfo;
        const matrix = new Matrix4();
        const range = { start: 0, count: 0 };
        for (let i = 0; i < info.length; i++) {
            if (!info[i].active) continue;
            batched.getMatrixAt(i, matrix);
            const geometryRange = batched.getGeometryRangeAt(info[i].geometryIndex);
            range.start = geometryRange.start;
            range.count = geometryRange.count;
            out(i, { objectToWorld: multiplyMatrices(world, matrix.elements, new Float64Array(16)), visible: info[i].visible }, range);
        }
    } else {
        out(-1, { objectToWorld: Float64Array.from(world), visible: true }, null);
    }
}

/** One mesh's instances with their keys and state (its children are not visited). */
export function walkMesh(mesh: Mesh, options: BvhSceneFromObject3DOptions, out: WalkedInstance[]): void {
    if (!mesh.isMesh || !mesh.geometry?.getAttribute('position')) return;
    const staticOption = options.static ?? false;
    const visible = effectivelyVisible(mesh);
    const mask = instanceMaskFromLayers(mesh.layers, options.layers ?? true);
    instancesOf(mesh, (instanceId, record, batchRange) => {
        const ranges = threeTriangleRanges(mesh.geometry, mesh.material, batchRange ?? mesh.geometry.drawRange);
        const isStatic = typeof staticOption === 'function' ? staticOption(mesh, instanceId) : staticOption;
        for (const range of ranges) {
            const material = checkMaterial(
                options.material?.(mesh, { instanceId, group: range.group, materialIndex: range.materialIndex, threeMaterial: range.threeMaterial }),
                mesh,
            );
            out.push({
                key: `${mesh.id}:${instanceId}:${range.group}`,
                object: mesh,
                instanceId,
                range,
                meshKey: `${mesh.geometry.uuid}:${range.start}:${range.count}`,
                geometry: mesh.geometry,
                objectToWorld: record.objectToWorld,
                visible: visible && record.visible,
                mask,
                material,
                static: isStatic,
            });
        }
    });
}

/** Walks the root: every instance with its key, mesh key and state (the shared part of the scene builder and the sync). */
export function walkObject3D(root: Object3D | readonly Object3D[], options: BvhSceneFromObject3DOptions = {}): WalkedInstance[] {
    const filter = options.filter;
    const out: WalkedInstance[] = [];
    const visit = (node: Object3D) => {
        const mesh = node as Mesh;
        if (!mesh.isMesh || (filter && !filter(mesh))) return;
        walkMesh(mesh, options, out);
    };
    for (const node of Array.isArray(root) ? root : [root as Object3D]) node.traverse(visit);
    return out;
}

/**
 * The meshes and instances of a three scene (or several roots), ready for
 * `buildTraceScene(scene.meshes, scene.instances, { build, mergeStaticInstances })`.
 * Instance `i` is `sources[i]`; that is `BvhHit.instance`.
 */
export function bvhSceneFromObject3D(root: Object3D | readonly Object3D[], options: BvhSceneFromObject3DOptions = {}): BvhThreeScene {
    const meshes: MeshGeometry[] = [];
    const meshSources: BvhThreeMeshSource[] = [];
    const meshIndex = new Map<string, number>();
    const instances: TraceSceneInstance[] = [];
    const sources: BvhThreeSource[] = [];
    for (const walked of walkObject3D(root, options)) {
        let mesh = meshIndex.get(walked.meshKey);
        if (mesh === undefined) {
            mesh = meshes.length;
            meshes.push(meshInputOf(walked.geometry, walked.range.start, walked.range.count));
            meshSources.push({ geometry: walked.geometry, start: walked.range.start, count: walked.range.count });
            meshIndex.set(walked.meshKey, mesh);
        }
        const instanceIndex = instances.length;
        instances.push({
            meshIndex: mesh,
            instanceIndex,
            objectToWorld: walked.objectToWorld,
            visible: walked.visible,
            mask: walked.mask,
            material: walked.material,
            static: walked.static,
        });
        sources.push({ object: walked.object, instanceId: walked.instanceId, group: walked.range.group, materialIndex: walked.range.materialIndex, mesh, static: walked.static });
    }
    return { meshes, meshSources, instances, sources };
}

/** {@link meshInputOf}, for the sync. */
export function meshGeometryOf(source: BvhThreeMeshSource): MeshGeometry {
    return meshInputOf(source.geometry, source.start, source.count);
}

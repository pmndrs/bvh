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
 *   groups has no triangles, as in three). Ranges that start off a multiple
 *   of 3, end mid-triangle or overlap are read as three steps through them
 *   ({@link threeTriangleRanges}). Indexed and non-indexed geometry;
 *   positions are read through the attribute, so interleaved, normalized and
 *   non-float positions work.
 * - **Meshes.** One mesh per geometry and triangle range, shared by every
 *   object (and every instance) that uses it. With an array of materials
 *   (`multiMaterial: 'triangles'`, the default), one mesh of all the groups
 *   (in group order) with one instance per object (and per instance of an
 *   instanced mesh), and each triangle carries its group's material
 *   (`MeshInput.triangleMaterials`, from `material(object, info)` per group);
 *   `BvhThreeMeshSource.groups`, {@link threeGroupOf} and
 *   {@link threeMaterialOf} map a hit's triangle back to its group and three
 *   material. `bvhSceneFromObject3D`, `SceneTreesObject3DSync` and
 *   `BvhSceneRaycaster` fold the same way, so instance N is the same object
 *   in all three. `multiMaterial: 'groups'` keeps one mesh and instance per
 *   group, with the group's material on the instance.
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
 *   (0-65534) per instance and group, reported as `BvhHit.material`. For a
 *   multi-material mesh it is asked once per group (`info.group`,
 *   `info.materialIndex`: the group's `materialIndex` into the array), and the
 *   answer becomes that group's triangles' material; undefined leaves them
 *   without one. An object whose instances give a group different answers
 *   (an `InstancedMesh` with a material per instance) is not folded: it
 *   keeps one mesh per group, shared by its instances, with each instance's
 *   answer on the instance, as `'groups'` does. A hit's `material` is then
 *   what the callback returned for its group; to get back to three's
 *   `object.material[i]`, return (or map from) `info.materialIndex`, or call
 *   {@link threeMaterialOf} with the hit's triangle.
 * - **Static.** `static` (true, or a function of the object and instance)
 *   marks instances that never move, for `mergeStaticInstances`, and the
 *   SceneTrees sync never pushes their transforms.
 *
 * Nothing here needs a renderer or a GPU: it runs in Node.
 */

import { Matrix4, type BatchedMesh, type BufferAttribute, type BufferGeometry, type InstancedMesh, type InterleavedBufferAttribute, type Layers, type Material, type Mesh, type Object3D } from 'three';
import { MAX_MATERIAL } from '../trace/layout.ts';
import { TRIANGLE_MATERIAL_NONE } from '../mesh-input.ts';
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
    /**
     * How a multi-material mesh (an array of materials, with groups) becomes
     * meshes, the same in `bvhSceneFromObject3D`, `SceneTreesObject3DSync`
     * and `BvhSceneRaycaster`. `'triangles'` (the default): one mesh and
     * instance holding every group, each triangle with its group's material
     * (`triangleMaterials`); its source's `group` and `materialIndex` are -1
     * ({@link threeGroupOf} and {@link threeMaterialOf} give a hit's). An
     * object whose instances give a group different materials falls back to
     * `'groups'`. `'groups'`: one mesh and instance per group, with the
     * group's material on the instance.
     */
    readonly multiMaterial?: 'triangles' | 'groups';
}

/** The three object behind one instance. `BvhHit.instance` (and `CpuHit.instanceIndex`) indexes these. */
export interface BvhThreeSource {
    readonly object: Mesh;
    /** The instance of an `InstancedMesh` or `BatchedMesh`, or -1 for a plain mesh. */
    readonly instanceId: number;
    /**
     * The geometry group, or -1: for a mesh with one material, and for a
     * folded multi-material mesh (`multiMaterial: 'triangles'`), which holds
     * every group ({@link threeGroupOf} gives a hit triangle's).
     */
    readonly group: number;
    /** The group's `materialIndex`, or -1 (see {@link group}; {@link threeMaterialOf} gives a hit triangle's three material). */
    readonly materialIndex: number;
    /** Position of this instance's mesh in {@link BvhThreeScene.meshes}. */
    readonly mesh: number;
    readonly static: boolean;
}

/** One geometry group of a multi-material mesh ({@link BvhThreeMeshSource.groups}). */
export interface BvhThreeGroupRange {
    /** First index (vertex) of the group's triangles, and how many indices, a multiple of 3. */
    readonly start: number;
    readonly count: number;
    /** The group's position in `geometry.groups`, and its `materialIndex`. */
    readonly group: number;
    readonly materialIndex: number;
    /** What `material(object, info)` gave its triangles (`triangleMaterials`), or undefined. */
    readonly material: number | undefined;
}

/** Where one of {@link BvhThreeScene.meshes} came from. */
export interface BvhThreeMeshSource {
    readonly geometry: BufferGeometry;
    /** First index (vertex, for non-indexed geometry) of the range, as three's `drawRange.start` and `group.start`. */
    readonly start: number;
    /** Indices (vertices) in the range, a multiple of 3; `start` need not be (see `threeTriangleRanges`). */
    readonly count: number;
    /**
     * A folded multi-material mesh (`multiMaterial: 'triangles'`): its groups' ranges, one
     * after another in the mesh's triangles; `start` and `count` are then the
     * first group's start and the total. Absent for one range.
     */
    readonly groups?: readonly BvhThreeGroupRange[];
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
    return Math.floor(threeFirstIndex(meshSource, triangle) / 3);
}

/** The first index (vertex, for non-indexed geometry) of a hit's triangle in the source geometry, as three steps through it. */
export function threeFirstIndex(meshSource: BvhThreeMeshSource, triangle: number): number {
    const found = groupAt(meshSource, triangle);
    return found ? found.range.start + found.local * 3 : meshSource.start + triangle * 3;
}

/** The geometry group a hit's triangle came from, in a folded multi-material mesh (`multiMaterial: 'triangles'`); null for a mesh of one range. */
export function threeGroupOf(meshSource: BvhThreeMeshSource, triangle: number): BvhThreeGroupRange | null {
    return groupAt(meshSource, triangle)?.range ?? null;
}

/**
 * The three material that draws a hit's triangle: `object.material[i]` of
 * its group (a folded mesh's from the triangle, a per-group instance's from
 * its source), or the object's one material. `source` and `meshSource` are
 * the hit instance's (`sourceOf`, `meshSourceOf` or `scene.meshSources[source.mesh]`).
 */
export function threeMaterialOf(source: BvhThreeSource, meshSource: BvhThreeMeshSource, triangle: number): Material | null {
    const material = source.object.material;
    if (!Array.isArray(material)) return material ?? null;
    const index = threeGroupOf(meshSource, triangle)?.materialIndex ?? source.materialIndex;
    return material[index] ?? null;
}

function groupAt(meshSource: BvhThreeMeshSource, triangle: number): { range: BvhThreeGroupRange; local: number } | null {
    if (!meshSource.groups) return null;
    let first = 0;
    for (const range of meshSource.groups) {
        const count = range.count / 3;
        if (triangle < first + count) return { range, local: triangle - first };
        first += count;
    }
    return null;
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
 *
 * Matched step for step, including ranges that don't start or end on a
 * multiple of 3: three steps `j = start, start + 3, ...` while `j < end`, and
 * tests the triangle of indices `j, j + 1, j + 2`, so a range that isn't a
 * whole number of triangles ends in a triangle that reads past it (into the
 * next group, or past the draw range), and one that starts off a multiple of 3
 * tests triangles made of neighbouring triangles' corners. A triangle that
 * would read past the index (or, without one, the positions) is dropped:
 * three reads `undefined` there and never hits it. Each range's `count` is
 * therefore a multiple of 3. Overlapping groups give overlapping ranges, as
 * three tests (and reports) such triangles once per group. A group whose
 * material is missing is skipped (three throws on it).
 */
export function threeTriangleRanges(
    geometry: BufferGeometry,
    material: Material | Material[] | null | undefined,
    drawRange: { readonly start: number; readonly count: number } = geometry.drawRange,
): TriangleRange[] {
    const position = geometry.getAttribute('position');
    if (!position) return [];
    const total = geometry.index ? geometry.index.count : position.count;
    /** three's loop from `start` while `j < end`, without the triangles that read outside `[0, total)`. */
    const span = (start: number, end: number): { start: number; count: number } => {
        let first = start;
        if (first < 0) first += Math.ceil(-first / 3) * 3; // indices below 0 read undefined: never hit
        if (end <= first) return { start: first, count: 0 };
        let triangles = Math.ceil((end - first) / 3);
        if (first + triangles * 3 > total) triangles = Math.max(0, Math.floor((total - first) / 3));
        return { start: first, count: triangles * 3 };
    };
    const drawEnd = drawRange.start + drawRange.count;
    if (Array.isArray(material)) {
        const ranges: TriangleRange[] = [];
        geometry.groups.forEach((group, index) => {
            const groupMaterial = material[group.materialIndex ?? 0];
            if (!groupMaterial) return;
            const { start, count } = span(Math.max(group.start, drawRange.start), Math.min(total, group.start + group.count, drawEnd));
            if (count > 0) ranges.push({ start, count, group: index, materialIndex: group.materialIndex ?? 0, threeMaterial: groupMaterial });
        });
        return ranges;
    }
    const { start, count } = span(Math.max(0, drawRange.start), Math.min(total, drawEnd));
    return count > 0 ? [{ start, count, group: -1, materialIndex: -1, threeMaterial: material ?? null }] : [];
}

/** The version of a position attribute's data (an interleaved attribute's is its buffer's). */
function positionVersion(attribute: BufferAttribute | InterleavedBufferAttribute): number {
    const interleaved = attribute as InterleavedBufferAttribute;
    return interleaved.isInterleavedBufferAttribute === true ? interleaved.data.version : (attribute as BufferAttribute).version;
}

/** Compacted positions per attribute (interleaved, normalized or non-float ones), shared by every range of the geometry while the data's version holds. */
const compacted = new WeakMap<object, { readonly version: number; readonly positions: Float32Array }>();

/** Every vertex's `x, y, z`, tightly packed: the attribute's own array, or one shared compacted copy. */
function sharedPositionsOf(attribute: BufferAttribute | InterleavedBufferAttribute): Float32Array {
    const plain = attribute as BufferAttribute;
    if ((attribute as InterleavedBufferAttribute).isInterleavedBufferAttribute !== true && plain.array instanceof Float32Array && plain.itemSize === 3 && !plain.normalized) {
        return plain.array.length === attribute.count * 3 ? plain.array : plain.array.subarray(0, attribute.count * 3);
    }
    const version = positionVersion(attribute);
    const cached = compacted.get(attribute);
    if (cached && cached.version === version && cached.positions.length === attribute.count * 3) return cached.positions;
    const positions = positionsOf(attribute);
    compacted.set(attribute, { version, positions });
    return positions;
}

/** One range of a geometry as a mesh input. */
function meshInputOf(geometry: BufferGeometry, start: number, count: number): MeshGeometry {
    const position = geometry.getAttribute('position') as BufferAttribute | InterleavedBufferAttribute;
    const index = geometry.index;
    if (!index) return { positions: sharedPositionsOf(position).subarray(start * 3, (start + count) * 3), indices: null };
    const array = index.array;
    let indices: Uint16Array | Uint32Array;
    if (array instanceof Uint16Array || array instanceof Uint32Array) {
        indices = start === 0 && count === array.length ? array : array.subarray(start, start + count);
    } else {
        indices = Uint32Array.from({ length: count }, (_, i) => index.getX(start + i));
    }
    return { positions: sharedPositionsOf(position), indices };
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
    /** The triangles: one range, or a folded mesh's whole (group and materialIndex -1). */
    readonly range: TriangleRange;
    readonly meshKey: string;
    /** The instance's mesh: its range, or a folded mesh's groups. */
    readonly meshSource: BvhThreeMeshSource;
    readonly geometry: BufferGeometry;
    readonly objectToWorld: Float64Array;
    readonly visible: boolean;
    readonly mask: number;
    readonly material: number | undefined;
    readonly static: boolean;
}

const scratch = new Float64Array(16);

/** `out = a x b`, column-major; `b` starts at `bOffset` (an `instanceMatrix` array without a subarray). */
export function multiplyMatrices(a: ArrayLike<number>, b: ArrayLike<number>, out: Float64Array, bOffset = 0): Float64Array {
    for (let column = 0; column < 4; column++) {
        for (let row = 0; row < 4; row++) {
            let sum = 0;
            for (let k = 0; k < 4; k++) sum += a[k * 4 + row] * b[bOffset + column * 4 + k];
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
            out(i, { objectToWorld: multiplyMatrices(world, matrices, new Float64Array(16), i * 16), visible: true }, null);
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

/** One mesh's instances with their keys and state (its children are not visited); a multi-material mesh folded per {@link BvhSceneFromObject3DOptions.multiMaterial}. */
export function walkMesh(mesh: Mesh, options: BvhSceneFromObject3DOptions, out: WalkedInstance[]): void {
    if (!mesh.isMesh || !mesh.geometry?.getAttribute('position')) return;
    const staticOption = options.static ?? false;
    const visible = effectivelyVisible(mesh);
    const mask = instanceMaskFromLayers(mesh.layers, options.layers ?? true);
    const geometry = mesh.geometry;
    const walked: { instanceId: number; record: InstanceRecordInput; isStatic: boolean; ranges: { range: TriangleRange; material: number | undefined }[] }[] = [];
    instancesOf(mesh, (instanceId, record, batchRange) => {
        const ranges = threeTriangleRanges(geometry, mesh.material, batchRange ?? geometry.drawRange);
        const isStatic = typeof staticOption === 'function' ? staticOption(mesh, instanceId) : staticOption;
        walked.push({
            instanceId,
            record,
            isStatic,
            ranges: ranges.map((range) => ({
                range,
                material: checkMaterial(options.material?.(mesh, { instanceId, group: range.group, materialIndex: range.materialIndex, threeMaterial: range.threeMaterial }), mesh),
            })),
        });
    });
    const fold = Array.isArray(mesh.material) && options.multiMaterial !== 'groups' && sameGroupMaterials(walked);
    for (const { instanceId, record, isStatic, ranges } of walked) {
        const common = { object: mesh, instanceId, geometry, objectToWorld: record.objectToWorld, visible: visible && record.visible, mask, static: isStatic };
        if (fold && ranges.length > 0) {
            const groups: BvhThreeGroupRange[] = ranges.map(({ range, material }) => ({ start: range.start, count: range.count, group: range.group, materialIndex: range.materialIndex, material }));
            const count = groups.reduce((sum, group) => sum + group.count, 0);
            out.push({
                ...common,
                key: `${mesh.id}:${instanceId}:groups`,
                range: { start: groups[0].start, count, group: -1, materialIndex: -1, threeMaterial: null },
                // The groups and their materials: objects on the geometry whose answers differ get meshes of their own.
                meshKey: `${geometry.uuid}:groups:${groups.map((group) => `${group.start}.${group.count}.${group.material ?? '-'}`).join(',')}`,
                meshSource: { geometry, start: groups[0].start, count, groups },
                // A folded mesh's materials are its triangles'.
                material: undefined,
            });
            continue;
        }
        for (const { range, material } of ranges) {
            out.push({
                ...common,
                key: `${mesh.id}:${instanceId}:${range.group}`,
                range,
                meshKey: `${geometry.uuid}:${range.start}:${range.count}`,
                meshSource: { geometry, start: range.start, count: range.count },
                material,
            });
        }
    }
}

/**
 * Every instance gives each group the same material: then one folded mesh
 * serves them all. Otherwise (an `InstancedMesh` with a material per
 * instance) folding would copy the mesh once per instance.
 */
function sameGroupMaterials(walked: readonly { ranges: readonly { range: TriangleRange; material: number | undefined }[] }[]): boolean {
    const byGroup = new Map<number, number | undefined>();
    for (const { ranges } of walked) {
        for (const { range, material } of ranges) {
            if (!byGroup.has(range.group)) byGroup.set(range.group, material);
            else if (byGroup.get(range.group) !== material) return false;
        }
    }
    return true;
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
            meshes.push(meshGeometryOf(walked.meshSource));
            meshSources.push(walked.meshSource);
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
        sources.push(sourceOfWalked(walked, mesh));
    }
    return { meshes, meshSources, instances, sources };
}

/** A walked instance's source, with its mesh's position (or handle). */
export function sourceOfWalked(walked: WalkedInstance, mesh: number): BvhThreeSource {
    return { object: walked.object, instanceId: walked.instanceId, group: walked.range.group, materialIndex: walked.range.materialIndex, mesh, static: walked.static };
}

/** {@link meshInputOf}, for the sync. */
export function meshGeometryOf(source: BvhThreeMeshSource): MeshGeometry {
    return source.groups ? groupedMeshInputOf(source.geometry, source.groups) : meshInputOf(source.geometry, source.start, source.count);
}

/** A multi-material mesh's groups as one mesh input: their triangles one after another, each with its group's material. */
function groupedMeshInputOf(geometry: BufferGeometry, groups: readonly BvhThreeGroupRange[]): MeshGeometry {
    const position = geometry.getAttribute('position') as BufferAttribute | InterleavedBufferAttribute;
    const index = geometry.index;
    const total = groups.reduce((sum, range) => sum + range.count, 0);
    const indices = new Uint32Array(total);
    const triangleMaterials = new Uint16Array(total / 3);
    let at = 0;
    for (const range of groups) {
        for (let k = 0; k < range.count; k++) indices[at + k] = index ? index.getX(range.start + k) : range.start + k;
        triangleMaterials.fill(range.material ?? TRIANGLE_MATERIAL_NONE, at / 3, (at + range.count) / 3);
        at += range.count;
    }
    return { positions: sharedPositionsOf(position), indices, triangleMaterials };
}

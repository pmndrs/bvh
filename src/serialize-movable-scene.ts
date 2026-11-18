/**
 * A movable scene file: a scene's mesh trees and its instance list, loaded
 * into `SceneTrees` so that the top level is built at load and the instances
 * can then move. The other scene file (serialize-scene.ts, 'PBVS') is a
 * snapshot: its top level is baked, so nothing in it moves.
 *
 * What it holds:
 *
 * - **One tree per mesh**, as `serializeLayout` writes it (a whole 'PBVH'
 *   file, with the mesh's geometry hash), except meshes that only merged
 *   static instances use (their triangles are in a group's tree).
 * - **The instances**: mesh, `instanceIndex`, transform, mask, material,
 *   `visible`, `contributesToGi`, and `static`.
 * - **The merged static groups' trees.** Static instances are merged at bake
 *   time as `SceneTrees` merges them at its first update (`mergeStaticGeometry`
 *   over the same meshes and instances), and each group's tree is stored with
 *   its world-space geometry's hash. `SceneTrees.placeMergedTrees` gives them
 *   to the groups the first update makes, if they are the same groups; if
 *   not, the groups are built at load.
 * - **Optionally the meshes' geometry** (`serializeMovableScene(data, {
 *   geometry: true })`), so the file loads with no model; else the app passes
 *   its own geometry to `deserializeMovableScene`, which checks every mesh's
 *   hash.
 *
 *     header (16 words, little-endian)
 *         0  magic 'PBVM'
 *         1  format version (SERIALIZED_MOVABLE_SCENE_FORMAT_VERSION)
 *         2  GPU layout version (GPU_LAYOUT_VERSION)
 *         3  formats: bits 0-7 triangles (0 records, 1 indexed), bits 8-15
 *            nodes (0 f32, 1 quantized): the scene's, for SceneTrees.create;
 *            bits 16-31 0
 *         4  meshCount
 *         5  mergedGroupCount (group trees in the file)
 *         6  instanceCount
 *         7  treeCount (mesh trees in the file)
 *         8  description bytes (JSON, padded with spaces to a multiple of 4)
 *         9  tree bytes
 *         10 geometry bytes (0: written without the geometry)
 *         11-15 0
 *     description  JSON: per mesh its geometry hash, where its tree is, its
 *                  `deforming` and where its geometry is; per group where its
 *                  tree is; the instances
 *     trees        'PBVH' tree files, one after another (mesh trees, then
 *                  the groups'), each with its geometry's hash
 *     geometry     positions, indices, triangle flags and materials, each
 *                  padded to a word
 *
 * Words 0-2 are the tree file's (serialize.ts), so `readSerializedHeader`
 * reads all three kinds. A file from another format or GPU layout version is
 * refused, not converted, as the others are.
 */

import { DEFAULT_NODE_FORMAT, GPU_LAYOUT_VERSION, type GpuLayoutMeshBvh, type GpuLayoutNodeFormat, type TriangleLeaves } from './gpu-layout.ts';
import { checkTriangleLeaves } from './indexed-leaves.ts';
import { checkNodeFormat, quantizeNodes } from './quantized-nodes.ts';
import type { SceneMergedTree, SceneTrees } from './scene/scene-trees.ts';
import {
    deserializeLayout,
    encodeSerializedFormats,
    meshGeometryHash,
    MOVABLE_SCENE_HEADER_WORD,
    MOVABLE_SCENE_MAGIC,
    readSerializedHeader,
    SERIALIZED_HEADER_BYTES,
    SERIALIZED_MOVABLE_SCENE_FORMAT_VERSION,
    serializeLayoutWithHash,
} from './serialize.ts';
import type { MeshBvh, MeshGeometry } from './trace/mesh-bvh.ts';
import { assembleTraceScene, type TraceSceneData, type TraceSceneInstance } from './trace/scene-assembly.ts';
import type { SahBuildOptions } from './trace/sah-builder.ts';
import { mergeStaticGeometry, type StaticMergePlan } from './trace/static-merge.ts';
import { countTriangles } from './trace/triangle-packing.ts';

/** A mesh of a {@link MovableSceneData}. */
export interface MovableSceneMesh {
    /** The mesh's tree in the scene's triangle format, or null when only merged static instances use the mesh (its triangles are in a group's tree). */
    readonly tree: GpuLayoutMeshBvh | null;
    /** `meshGeometryHash` of the geometry the tree was built from. */
    readonly geometryHash: readonly [number, number];
    /**
     * The mesh's geometry: what `buildMovableScene` was given, the file's own
     * (written with `{ geometry: true }`), or the app's (given to
     * `deserializeMovableScene`). {@link placeMovableScene} needs it.
     */
    readonly geometry?: MeshGeometry;
    /** The mesh deforms (`SceneMeshOptions.deforming`): its tree is `f32`, and its instances never merge. */
    readonly deforming?: true | { readonly bounds: readonly number[] };
}

/** A scene's mesh trees and instances, for `SceneTrees` ({@link buildMovableScene}, {@link deserializeMovableScene}). */
export interface MovableSceneData {
    /** The scene's triangle format: create the `SceneTrees` with it (indexed: with `weld: false`). */
    readonly triangleLeaves: TriangleLeaves;
    /** The scene's node format: create the `SceneTrees` with it (an `f32` scene refuses quantized trees). */
    readonly nodeFormat: GpuLayoutNodeFormat;
    readonly meshes: readonly MovableSceneMesh[];
    /** Every instance, merged ones too (`static: true`): a `SceneTrees` merges them again at its first update. */
    readonly instances: readonly TraceSceneInstance[];
    /** The merged static groups' trees, in `mergeStaticGeometry`'s order (`SceneTrees.placeMergedTrees`). */
    readonly mergedGroups: readonly SceneMergedTree[];
}

/** Options of {@link buildMovableScene}. */
export interface MovableSceneBuildOptions {
    /** Builds one mesh's tree (and each merged group's): tinybvh (`buildMeshBvhWithTinyBvh`, `buildLayout`), a worker builder's `build`, any builder writing the GPU layout. */
    readonly build: (geometry: MeshGeometry) => MeshBvh | Promise<MeshBvh>;
    /** The scene's triangle format; `build` must return trees in it. Default `'records'`. */
    readonly triangleLeaves?: TriangleLeaves;
    /** The scene's node format. Default `'quantized'`: `f32` trees are quantized here (not the deforming meshes'). */
    readonly nodeFormat?: GpuLayoutNodeFormat;
    /**
     * Merge the static instances (`static: true`) as `SceneTrees` will at its
     * first update, and store each group's tree, so no group is built at
     * load. Default true. With false, no group trees are stored: a merging
     * `SceneTrees` builds the groups at load.
     */
    readonly mergeStaticInstances?: boolean;
    /** Per mesh, the mesh deforms (`SceneMeshOptions.deforming`: true, or the box every pose stays in): its tree stays `f32` and its instances never merge. */
    readonly deforming?: readonly (boolean | { readonly bounds: ArrayLike<number> } | undefined)[];
}

const isDeforming = (mesh: MovableSceneMesh | undefined): boolean => mesh?.deforming !== undefined;

/** The instances as `SceneTrees` plans its merge: a deforming mesh's instances never merge. */
function planInstances(meshes: readonly MovableSceneMesh[], instances: readonly TraceSceneInstance[]): TraceSceneInstance[] {
    return instances.map((instance) => ({ ...instance, static: instance.static === true && !isDeforming(meshes[instance.meshIndex]) }));
}

/** A built tree in the scene's formats: the triangle format checked, `f32` nodes quantized for a quantized scene (not a deforming mesh's). */
function sceneTree(tree: MeshBvh, geometry: MeshGeometry, leaves: TriangleLeaves, nodeFormat: GpuLayoutNodeFormat, deforming: boolean, name: string): GpuLayoutMeshBvh {
    const where = 'buildMovableScene';
    if ((tree.triangleLeaves ?? 'records') !== leaves) throw new RangeError(`${where}: build returned ${tree.triangleLeaves ?? 'records'} triangles for ${name}, the scene is ${leaves}; build with triangleLeaves: '${leaves}'`);
    if (tree.triangleCount !== countTriangles(geometry)) throw new RangeError(`${where}: build returned ${tree.triangleCount} triangles for ${name}, which has ${countTriangles(geometry)}`);
    const nodes = tree.nodeFormat ?? 'f32';
    if (deforming) {
        if (nodes !== 'f32') throw new RangeError(`${where}: ${name} deforms, so its tree must be f32 (its refit keeps an f32 copy); build it f32`);
        return tree;
    }
    if (nodes === 'quantized' && nodeFormat === 'f32') throw new RangeError(`${where}: build returned a quantized tree for ${name}, and the scene is f32`);
    return nodeFormat === 'quantized' ? quantizeNodes(tree) : tree;
}

/**
 * Builds a scene's mesh trees for a movable scene file: one tree per mesh
 * (the ones only merged static instances use get none), and, with
 * `mergeStaticInstances` (the default), each merged static group's tree,
 * merged as `SceneTrees` merges at its first update. The result goes to
 * {@link serializeMovableScene}, or straight to {@link placeMovableScene}.
 * Nothing is assembled: the top level is built where the scene loads.
 */
export async function buildMovableScene(meshes: readonly MeshGeometry[], instances: readonly TraceSceneInstance[], options: MovableSceneBuildOptions): Promise<MovableSceneData> {
    const where = 'buildMovableScene';
    const triangleLeaves = checkTriangleLeaves(options.triangleLeaves, where);
    const nodeFormat = checkNodeFormat(options.nodeFormat, where, DEFAULT_NODE_FORMAT);
    instances.forEach((instance, index) => {
        if (!Number.isInteger(instance.meshIndex) || instance.meshIndex < 0 || instance.meshIndex >= meshes.length) throw new RangeError(`${where}: instance ${index} names mesh ${instance.meshIndex}, out of the ${meshes.length} meshes`);
    });
    const deforming = meshes.map((_, index): MovableSceneMesh['deforming'] => {
        const flag = options.deforming?.[index];
        if (flag === undefined || flag === false) return undefined;
        return flag === true ? true : { bounds: Array.from(flag.bounds) };
    });
    const shells = meshes.map((geometry, index): MovableSceneMesh => ({ tree: null, geometryHash: [0, 0], geometry, ...(deforming[index] ? { deforming: deforming[index] } : {}) }));
    const planned = planInstances(shells, instances);
    const plan = options.mergeStaticInstances !== false && planned.some((instance) => instance.static) ? mergeStaticGeometry(meshes, planned, { triangleLeaves }) : null;
    // Meshes only merged sources use need no tree of their own; every other mesh gets one (an unused one too: the app may place instances later).
    const mergedAway = new Set<number>();
    plan?.groups.forEach((group) => group.sources.forEach((source) => !plan.meshNeeded[source.meshIndex] && mergedAway.add(source.meshIndex)));

    const trees = await Promise.all(meshes.map((geometry, index) => (mergedAway.has(index) ? null : options.build(geometry))));
    const groupTrees = await Promise.all((plan?.groups ?? []).map((group) => options.build(group.geometry)));
    return {
        triangleLeaves,
        nodeFormat,
        meshes: meshes.map((geometry, index) => ({
            tree: trees[index] ? sceneTree(trees[index], geometry, triangleLeaves, nodeFormat, deforming[index] !== undefined, `mesh ${index}`) : null,
            geometryHash: meshGeometryHash(geometry),
            geometry,
            ...(deforming[index] ? { deforming: deforming[index] } : {}),
        })),
        instances: instances.map((instance) => ({ ...instance, objectToWorld: Array.from(instance.objectToWorld) })),
        mergedGroups: (plan?.groups ?? []).map((group, index) => ({
            tree: sceneTree(groupTrees[index], group.geometry, triangleLeaves, nodeFormat, false, `merged group ${index}`),
            geometryHash: meshGeometryHash(group.geometry),
        })),
    };
}

// ---- The file -----------------------------------------------------------------

/** A part of a section: its byte offset and byte length. */
type Span = [offset: number, bytes: number];

interface GeometryJson {
    readonly positions: Span;
    readonly indices?: [offset: number, bytes: number, bits: 16 | 32];
    readonly triangleFlags?: Span;
    readonly triangleMaterials?: Span;
}

interface MeshJson {
    readonly geometryHash: [number, number];
    readonly tree?: Span;
    readonly deforming?: true | number[];
    readonly geometry?: GeometryJson;
}

interface InstanceJson {
    readonly meshIndex: number;
    readonly instanceIndex: number;
    readonly objectToWorld: number[];
    readonly mask?: number;
    readonly material?: number;
    readonly visible?: false;
    readonly contributesToGi?: false;
    readonly static?: true;
}

interface MovableDescription {
    readonly meshes: readonly MeshJson[];
    readonly mergedGroups: readonly Span[];
    readonly instances: readonly InstanceJson[];
}

/** An instance's fields in one order, so the same scene always writes the same bytes. */
function instanceJson(instance: TraceSceneInstance): InstanceJson {
    return {
        meshIndex: instance.meshIndex,
        instanceIndex: instance.instanceIndex,
        objectToWorld: Array.from({ length: 16 }, (_, k) => instance.objectToWorld[k]),
        ...(instance.mask !== undefined ? { mask: instance.mask } : {}),
        ...(instance.material !== undefined ? { material: instance.material } : {}),
        ...(instance.visible === false ? { visible: false as const } : {}),
        ...(instance.contributesToGi === false ? { contributesToGi: false as const } : {}),
        ...(instance.static === true ? { static: true as const } : {}),
    };
}

/** Byte parts laid one after another, each padded to a word. */
class Section {
    readonly parts: Uint8Array[] = [];
    bytes = 0;

    add(part: Uint8Array): Span {
        const at: Span = [this.bytes, part.byteLength];
        this.parts.push(part);
        this.bytes += part.byteLength;
        const pad = (4 - (part.byteLength % 4)) % 4;
        if (pad > 0) {
            this.parts.push(new Uint8Array(pad));
            this.bytes += pad;
        }
        return at;
    }
}

const bytesOf = (array: ArrayBufferView): Uint8Array => new Uint8Array(array.buffer, array.byteOffset, array.byteLength);

function writeGeometry(geometry: MeshGeometry, section: Section, name: string): GeometryJson {
    const where = 'serializeMovableScene';
    const triangles = countTriangles(geometry);
    const { indices, triangleFlags, triangleMaterials } = geometry;
    let materials: Uint16Array | null = null;
    if (triangleMaterials) {
        if (triangleMaterials.length < triangles) throw new RangeError(`${where}: ${name}'s triangleMaterials has ${triangleMaterials.length} entries for ${triangles} triangles`);
        materials = Uint16Array.from({ length: triangles }, (_, k) => {
            const value = triangleMaterials[k];
            if (!Number.isInteger(value) || value < 0 || (value > 0xffff && value !== 0xffffffff)) throw new RangeError(`${where}: ${name}'s triangle ${k} has material ${value}`);
            return value === 0xffffffff ? 0xffff : value;
        });
    }
    return {
        positions: section.add(bytesOf(geometry.positions)),
        ...(indices ? { indices: [...section.add(bytesOf(indices)), indices instanceof Uint16Array ? 16 : 32] as [number, number, 16 | 32] } : {}),
        ...(triangleFlags ? { triangleFlags: section.add(bytesOf(triangleFlags.subarray(0, triangles))) } : {}),
        ...(materials ? { triangleMaterials: section.add(bytesOf(materials)) } : {}),
    };
}

/** Options of {@link serializeMovableScene}. */
export interface SerializeMovableSceneOptions {
    /**
     * Store every mesh's geometry (positions, indices, triangle flags and
     * materials), so the file loads with no model. Default false: the app
     * passes its own geometry to `deserializeMovableScene`, which checks it
     * against each mesh's hash.
     */
    readonly geometry?: boolean;
}

/** Writes a movable scene as one `ArrayBuffer` (see the file comment for the format). */
export function serializeMovableScene(scene: MovableSceneData, options: SerializeMovableSceneOptions = {}): ArrayBuffer {
    const where = 'serializeMovableScene';
    const triangleLeaves = checkTriangleLeaves(scene.triangleLeaves, where);
    const nodeFormat = checkNodeFormat(scene.nodeFormat, where);
    const checkTree = (tree: GpuLayoutMeshBvh, name: string) => {
        if ((tree.triangleLeaves ?? 'records') !== triangleLeaves) throw new RangeError(`${where}: ${name}'s tree has ${tree.triangleLeaves ?? 'records'} triangles, the scene ${triangleLeaves}`);
        if ((tree.nodeFormat ?? 'f32') === 'quantized' && nodeFormat === 'f32') throw new RangeError(`${where}: ${name}'s tree is quantized, and the scene is f32`);
    };
    scene.instances.forEach((instance, index) => {
        if (!Number.isInteger(instance.meshIndex) || instance.meshIndex < 0 || instance.meshIndex >= scene.meshes.length) throw new RangeError(`${where}: instance ${index} names mesh ${instance.meshIndex}, out of the ${scene.meshes.length} meshes`);
    });

    const trees = new Section();
    const geometry = new Section();
    const meshes = scene.meshes.map((mesh, index): MeshJson => {
        const name = `mesh ${index}`;
        let tree: Span | undefined;
        if (mesh.tree) {
            checkTree(mesh.tree, name);
            tree = trees.add(new Uint8Array(serializeLayoutWithHash(mesh.tree, mesh.geometryHash)));
        }
        let stored: GeometryJson | undefined;
        if (options.geometry) {
            if (!mesh.geometry) throw new RangeError(`${where}: ${name} has no geometry to store (geometry: true needs every mesh's)`);
            stored = writeGeometry(mesh.geometry, geometry, name);
        }
        return {
            geometryHash: [mesh.geometryHash[0], mesh.geometryHash[1]],
            ...(tree ? { tree } : {}),
            ...(mesh.deforming ? { deforming: mesh.deforming === true ? true : Array.from(mesh.deforming.bounds) } : {}),
            ...(stored ? { geometry: stored } : {}),
        };
    });
    const mergedGroups = scene.mergedGroups.map(({ tree, geometryHash }, index) => {
        checkTree(tree, `merged group ${index}`);
        return trees.add(new Uint8Array(serializeLayoutWithHash(tree, geometryHash)));
    });
    const description: MovableDescription = { meshes, mergedGroups, instances: scene.instances.map(instanceJson) };
    let json = JSON.stringify(description);
    // Pad with spaces (JSON ignores them) so the trees start on a word.
    let text = new TextEncoder().encode(json);
    if (text.byteLength % 4 !== 0) {
        json += ' '.repeat(4 - (text.byteLength % 4));
        text = new TextEncoder().encode(json);
    }

    const out = new ArrayBuffer(SERIALIZED_HEADER_BYTES + text.byteLength + trees.bytes + geometry.bytes);
    const header = new DataView(out, 0, SERIALIZED_HEADER_BYTES);
    const set = (word: number, value: number) => header.setUint32(word * 4, value, true);
    const at = MOVABLE_SCENE_HEADER_WORD;
    set(at.magic, MOVABLE_SCENE_MAGIC);
    set(at.formatVersion, SERIALIZED_MOVABLE_SCENE_FORMAT_VERSION);
    set(at.layoutVersion, GPU_LAYOUT_VERSION);
    set(at.formats, encodeSerializedFormats(triangleLeaves, nodeFormat));
    set(at.meshCount, scene.meshes.length);
    set(at.mergedGroupCount, scene.mergedGroups.length);
    set(at.instanceCount, scene.instances.length);
    set(at.treeCount, scene.meshes.filter((mesh) => mesh.tree).length);
    set(at.descriptionBytes, text.byteLength);
    set(at.treeBytes, trees.bytes);
    set(at.geometryBytes, geometry.bytes);

    const bytes = new Uint8Array(out);
    let offset = SERIALIZED_HEADER_BYTES;
    for (const part of [text, ...trees.parts, ...geometry.parts]) {
        bytes.set(part, offset);
        offset += part.byteLength;
    }
    return out;
}

/** Options of {@link deserializeMovableScene}. */
export interface DeserializeMovableSceneOptions {
    /**
     * The meshes' geometry, one per mesh in the file's order: what the app
     * draws, given to `SceneTrees.addMesh` by {@link placeMovableScene}. Each
     * is checked against the mesh's hash (and tree: triangle count, an indexed
     * tree's vertex indices), and a mismatch throws. Needed when the file was
     * written without its geometry; used instead of the file's when given.
     */
    readonly geometries?: readonly MeshGeometry[];
}

const sameHash = (a: readonly [number, number], b: readonly [number, number]) => a[0] === b[0] && a[1] === b[1];
const hex = (hash: readonly [number, number]) => hash.map((part) => part.toString(16).padStart(8, '0')).join('');

/**
 * Reads bytes written by {@link serializeMovableScene}. The trees and the
 * geometry are copies, so `bytes` may be reused. Throws, saying why, on
 * anything that isn't a whole movable scene of this package's versions
 * (another kind of file names its reader; another version says to bake
 * again), and on geometry the scene wasn't baked from (`options.geometries`).
 */
export function deserializeMovableScene(bytes: ArrayBuffer | ArrayBufferView, options: DeserializeMovableSceneOptions = {}): MovableSceneData {
    const where = 'deserializeMovableScene';
    const view = ArrayBuffer.isView(bytes) ? new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength) : new Uint8Array(bytes);
    if (view.byteLength < SERIALIZED_HEADER_BYTES) throw new RangeError(`${where}: ${view.byteLength} bytes is shorter than the ${SERIALIZED_HEADER_BYTES}-byte header`);
    let header: ReturnType<typeof readSerializedHeader>;
    try {
        header = readSerializedHeader(view);
    } catch {
        throw new Error(`${where}: not a serialized @pmndrs/bvh scene (bad magic)`);
    }
    if (header.kind === 'tree') throw new Error(`${where}: these bytes are one mesh's tree (serializeLayout); read them with deserializeLayout`);
    if (header.kind === 'scene') throw new Error(`${where}: these bytes are a scene snapshot (serializeTraceScene); read them with deserializeTraceScene`);
    if (!header.current) throw new Error(`${where}: ${header.reason}`);
    if (view.byteLength !== header.byteLength) throw new RangeError(`${where}: the header's sizes add up to ${header.byteLength} bytes, got ${view.byteLength}; the file is cut short or corrupt`);
    const words = new DataView(view.buffer, view.byteOffset, SERIALIZED_HEADER_BYTES);
    const word = (index: number) => words.getUint32(index * 4, true);
    const at = MOVABLE_SCENE_HEADER_WORD;
    const descriptionStart = SERIALIZED_HEADER_BYTES;
    const treeStart = descriptionStart + word(at.descriptionBytes);
    const geometryStart = treeStart + word(at.treeBytes);
    const geometryEnd = geometryStart + word(at.geometryBytes);

    let description: MovableDescription;
    try {
        description = JSON.parse(new TextDecoder().decode(view.subarray(descriptionStart, treeStart))) as MovableDescription;
    } catch (error) {
        throw new Error(`${where}: the scene's description is not valid JSON (${String(error)}); the file is corrupt`);
    }
    const { meshes, mergedGroups, instances } = description;
    if (!Array.isArray(meshes) || meshes.length !== header.meshCount || !Array.isArray(mergedGroups) || mergedGroups.length !== header.mergedGroupCount || !Array.isArray(instances) || instances.length !== header.instanceCount) {
        throw new RangeError(`${where}: the description doesn't match the header (${header.meshCount} meshes, ${header.mergedGroupCount} merged groups, ${header.instanceCount} instances); the file is corrupt`);
    }
    const appGeometries = options.geometries;
    if (appGeometries && appGeometries.length !== meshes.length) throw new RangeError(`${where}: the scene has ${meshes.length} meshes, and ${appGeometries.length} geometries were given`);

    /** Bytes `[offset, offset + length)` of the section `[start, end)`, checked to lie within it. */
    const part = (span: unknown, start: number, end: number, name: string): Uint8Array => {
        const [offset, length] = Array.isArray(span) ? (span as number[]) : [];
        if (!Number.isInteger(offset) || !Number.isInteger(length) || offset < 0 || length < 0 || start + offset + length > end) {
            throw new RangeError(`${where}: ${name} lies outside its section; the file is corrupt`);
        }
        return view.subarray(start + offset, start + offset + length);
    };
    const copy = (source: Uint8Array): ArrayBuffer => source.slice().buffer;
    const readGeometry = (json: GeometryJson, name: string): MeshGeometry => {
        const read = (span: unknown, what: string) => part(span, geometryStart, geometryEnd, `${name}'s ${what}`);
        const positions = new Float32Array(copy(read(json.positions, 'positions')));
        const indexBytes = json.indices ? read(json.indices, 'indices') : null;
        const indices = indexBytes ? (json.indices![2] === 16 ? new Uint16Array(copy(indexBytes)) : new Uint32Array(copy(indexBytes))) : undefined;
        const triangleFlags = json.triangleFlags ? new Uint32Array(copy(read(json.triangleFlags, 'triangle flags'))) : undefined;
        const triangleMaterials = json.triangleMaterials ? new Uint16Array(copy(read(json.triangleMaterials, 'triangle materials'))) : undefined;
        return { positions, ...(indices ? { indices } : {}), ...(triangleFlags ? { triangleFlags } : {}), ...(triangleMaterials ? { triangleMaterials } : {}) };
    };
    const readTree = (span: unknown, geometry: MeshGeometry | undefined, name: string): { tree: GpuLayoutMeshBvh; hash: readonly [number, number] | null } => {
        const treeBytes = part(span, treeStart, geometryStart, `${name}'s tree`);
        let tree: GpuLayoutMeshBvh;
        let hash: readonly [number, number] | null;
        try {
            const treeHeader = readSerializedHeader(treeBytes);
            hash = treeHeader.kind === 'tree' ? treeHeader.geometryHash : null;
            // Checks the geometry against the tree: triangle count, an indexed tree's vertex indices, the hash.
            tree = deserializeLayout(treeBytes, geometry ? { geometry } : {});
        } catch (error) {
            throw new RangeError(`${where}: ${name}: ${(error as Error).message}`);
        }
        if ((tree.triangleLeaves ?? 'records') !== header.triangleLeaves) throw new RangeError(`${where}: ${name}'s tree has ${tree.triangleLeaves ?? 'records'} triangles, the scene ${header.triangleLeaves}; the file is corrupt`);
        return { tree, hash };
    };

    const loadedMeshes = meshes.map((json, index): MovableSceneMesh => {
        const name = `mesh ${index}`;
        const hash = json.geometryHash;
        if (!Array.isArray(hash) || hash.length !== 2) throw new RangeError(`${where}: ${name} has no geometry hash; the file is corrupt`);
        const geometryHash: [number, number] = [hash[0] >>> 0, hash[1] >>> 0];
        const geometry = appGeometries?.[index] ?? (json.geometry ? readGeometry(json.geometry, name) : undefined);
        if (appGeometries) {
            const own = meshGeometryHash(appGeometries[index]);
            if (!sameHash(own, geometryHash)) {
                throw new RangeError(`${where}: ${name} was baked from other geometry (its hash ${hex(geometryHash)}, this geometry's ${hex(own)}): pass the positions and indices it was built from, or bake it again`);
            }
        }
        const tree = json.tree ? readTree(json.tree, geometry, name) : null;
        if (tree?.hash && !sameHash(tree.hash, geometryHash)) throw new RangeError(`${where}: ${name}'s tree and description name other geometry; the file is corrupt`);
        const deforming = json.deforming === true ? true : Array.isArray(json.deforming) ? { bounds: json.deforming } : undefined;
        return { tree: tree?.tree ?? null, geometryHash, ...(geometry ? { geometry } : {}), ...(deforming ? { deforming } : {}) };
    });
    const loadedGroups = mergedGroups.map((span, index): SceneMergedTree => {
        const { tree, hash } = readTree(span, undefined, `merged group ${index}`);
        if (!hash) throw new RangeError(`${where}: merged group ${index}'s tree has no geometry hash; the file is corrupt`);
        return { tree, geometryHash: hash };
    });
    const loadedInstances = instances.map((json, index): TraceSceneInstance => {
        const transform = json.objectToWorld;
        if (!Number.isInteger(json.meshIndex) || json.meshIndex < 0 || json.meshIndex >= meshes.length || !Array.isArray(transform) || transform.length !== 16 || !transform.every(Number.isFinite)) {
            throw new RangeError(`${where}: instance ${index} is not a mesh index and 16 finite numbers; the file is corrupt`);
        }
        return {
            meshIndex: json.meshIndex,
            instanceIndex: json.instanceIndex,
            objectToWorld: transform,
            ...(json.mask !== undefined ? { mask: json.mask } : {}),
            ...(json.material !== undefined ? { material: json.material } : {}),
            ...(json.visible === false ? { visible: false } : {}),
            ...(json.contributesToGi === false ? { contributesToGi: false } : {}),
            ...(json.static === true ? { static: true } : {}),
        };
    });
    return { triangleLeaves: header.triangleLeaves!, nodeFormat: header.nodeFormat!, meshes: loadedMeshes, instances: loadedInstances, mergedGroups: loadedGroups };
}

// ---- Loading ------------------------------------------------------------------

/** What {@link placeMovableScene} added. */
export interface PlacedMovableScene {
    /** The `SceneTrees` mesh handle of each mesh, in the scene's order. */
    readonly meshes: readonly number[];
    /** The `SceneTrees` instance handle of each instance, in the scene's order: move them with `setTransform`. */
    readonly instances: readonly number[];
    /**
     * The merged groups' trees were handed over (`SceneTrees.placeMergedTrees`):
     * false when the scene merges nothing more (`mergeStaticInstances: false`,
     * or past its first update), so the static instances stay instances and
     * the meshes only they used are built.
     */
    readonly mergedTreesPlaced: boolean;
}

/**
 * Loads a movable scene into a `SceneTrees`: every mesh (`addMesh`, with its
 * geometry and `deforming`), its tree (`placeTree`), every instance
 * (`addInstance`, static ones too), and the merged groups' trees
 * (`placeMergedTrees`). Given before the scene's first update, no tree is
 * built: that update places them all and builds the top level, and from then
 * on every instance that isn't merged moves like any other
 * (`setTransform(placed.instances[i], matrix)`).
 *
 * The `SceneTrees` must have the scene's formats: `triangleLeaves`, a
 * quantized scene for quantized trees (`scene.nodeFormat`), and `weld: false`
 * when indexed. Every mesh needs its geometry: the file's, or the app's given
 * to `deserializeMovableScene`. Loaded after the first update (streaming a
 * part in), static instances stay ordinary instances, and the meshes only
 * they used are built.
 */
export function placeMovableScene(trees: SceneTrees, scene: MovableSceneData): PlacedMovableScene {
    const where = 'placeMovableScene';
    if (scene.triangleLeaves !== trees.triangleLeaves) throw new RangeError(`${where}: the scene's trees have ${scene.triangleLeaves} triangles; create the SceneTrees with triangleLeaves: '${scene.triangleLeaves}'`);
    if (scene.nodeFormat === 'quantized' && trees.nodeFormat === 'f32') throw new RangeError(`${where}: the scene's trees are quantized; create the SceneTrees with nodeFormat: 'quantized' (the default)`);
    scene.meshes.forEach((mesh, index) => {
        if (!mesh.geometry) throw new RangeError(`${where}: mesh ${index} has no geometry: the file was written without it, so pass the app's to deserializeMovableScene (geometries)`);
    });
    const meshes = scene.meshes.map((mesh) => {
        const handle = trees.addMesh(mesh.geometry!, mesh.deforming ? { deforming: mesh.deforming } : {});
        if (mesh.tree) trees.placeTree(handle, mesh.tree);
        return handle;
    });
    const instances = scene.instances.map((instance) =>
        trees.addInstance({
            mesh: meshes[instance.meshIndex],
            objectToWorld: instance.objectToWorld,
            instanceIndex: instance.instanceIndex,
            ...(instance.mask !== undefined ? { mask: instance.mask } : {}),
            ...(instance.material !== undefined ? { material: instance.material } : {}),
            ...(instance.visible !== undefined ? { visible: instance.visible } : {}),
            ...(instance.contributesToGi !== undefined ? { contributesToGi: instance.contributesToGi } : {}),
            ...(instance.static ? { static: true } : {}),
        }),
    );
    const mergedTreesPlaced = scene.mergedGroups.length > 0 && trees.placeMergedTrees(scene.mergedGroups);
    return { meshes, instances, mergedTreesPlaced };
}

/** Options of {@link assembleMovableScene}. */
export interface AssembleMovableSceneOptions {
    /** The top level's SAH options (`assembleTraceScene`'s third argument). */
    readonly topLevel?: Partial<SahBuildOptions>;
}

/**
 * The scene a movable scene describes, assembled as it stands
 * (`assembleTraceScene`, the merged groups from their stored trees): what
 * `buildTraceScene` gives for the same meshes and instances, for
 * `uploadTraceScene` or the CPU walk (`createCpuTracer`) when nothing needs to
 * move, or to check a `SceneTrees` the scene was loaded into. Static instances
 * of a scene baked without merging (`mergeStaticInstances: false`) are placed
 * as ordinary instances. Needs every mesh's geometry when the scene is
 * indexed or has merged groups.
 */
export function assembleMovableScene(scene: MovableSceneData, options: AssembleMovableSceneOptions = {}): TraceSceneData {
    const where = 'assembleMovableScene';
    const indexed = scene.triangleLeaves === 'indexed';
    const needsGeometry = indexed || scene.mergedGroups.length > 0;
    const geometries = scene.meshes.map((mesh, index) => {
        if (needsGeometry && !mesh.geometry) throw new RangeError(`${where}: mesh ${index} has no geometry (an indexed scene, or merged groups, need every mesh's)`);
        return mesh.geometry ?? null;
    });
    let merged: { plan: StaticMergePlan; meshes: MeshBvh[] } | undefined;
    if (scene.mergedGroups.length > 0) {
        const plan = mergeStaticGeometry(geometries as MeshGeometry[], planInstances(scene.meshes, scene.instances), { triangleLeaves: scene.triangleLeaves });
        if (plan.groups.length !== scene.mergedGroups.length) throw new RangeError(`${where}: the instances merge into ${plan.groups.length} groups, and the scene has ${scene.mergedGroups.length} group trees`);
        plan.groups.forEach((group, index) => {
            if (!sameHash(meshGeometryHash(group.geometry), scene.mergedGroups[index].geometryHash)) throw new RangeError(`${where}: merged group ${index}'s tree was baked from other geometry`);
        });
        merged = { plan, meshes: scene.mergedGroups.map((group) => group.tree) };
    }
    return assembleTraceScene(
        scene.meshes.map((mesh) => mesh.tree),
        scene.instances,
        options.topLevel,
        {
            triangleLeaves: scene.triangleLeaves,
            nodeFormat: scene.nodeFormat,
            ...(merged ? { merged } : {}),
            ...(indexed ? { geometries, weld: false } : {}),
        },
    );
}

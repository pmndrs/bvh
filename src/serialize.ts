/**
 * Saving a built tree and loading it again: build once (in Node, a worker, or
 * at build time), store the bytes, and load them later with no build at all.
 * A whole assembled scene has its own file (serialize-scene.ts); both start
 * with the same three words, which {@link readSerializedHeader} reads.
 *
 * The bytes are a 64-byte header followed by the layout's three arrays as they
 * are uploaded to the GPU:
 *
 *     header (16 words, little-endian)
 *         0  magic 'PBVH' (a scene file: 'PBVS')
 *         1  format version (SERIALIZED_FORMAT_VERSION)
 *         2  GPU layout version (GPU_LAYOUT_VERSION)
 *         3  nodeCount
 *         4  triangleCount
 *         5  maxDepth
 *         6  leafCount
 *         7  formats: bits 0-7 the triangle format, 0 records, 1 indexed
 *            (`triangleLeaves`); bits 8-15 the node format, 0 `f32`, 1
 *            quantized (`nodeFormat`); bits 16-31 0
 *         8-13  bounds: minX, minY, minZ, maxX, maxY, maxZ (f32)
 *         14-15 the geometry's hash (`meshGeometryHash`), when the tree was
 *               written with its geometry; 0 and 0 when it wasn't
 *     nodes      nodeCount × 64 bytes (f32), or (nodeCount + 1) × 32 (quantized: the frame slot, then the nodes)
 *     triangles  triangleCount × 48 bytes (records) or × 16 (indexed)
 *     parents    nodeCount × 4 bytes
 *
 * Words 0-2 keep their places in every version of either file, so a reader
 * can always tell what it was given and whether it reads it.
 *
 * The arrays are copied byte for byte, so they keep the machine's byte order.
 * WebGPU buffers are little-endian, as is every machine that runs WebGPU, so
 * the header is written little-endian to match.
 *
 * Bytes from another layout version are refused, not converted: a tree is
 * cheap to rebuild, and a wrong word position would only show up as rays
 * missing. The triangle format is a variant within GPU layout version 2:
 * files written before it existed have 0 in word 7, which is records, so they
 * still load. The node format is one too, in bits 8-15 of the same word: a
 * reader from before it sees 256 or more and refuses the file. A quantized
 * file carries its frame in its first node slot. A reader refuses a format it
 * does not know. The geometry hash went into words 14-15, which were 0 before
 * and which earlier readers ignore: files from either side load in both.
 *
 * An indexed file holds no positions: load it next to the same geometry it
 * was built from (`CpuMeshBvh` takes it as `source`, a scene as its
 * `geometries`, `SceneTrees` as the mesh's). Written with its geometry, the
 * file carries the geometry's hash, and `deserializeLayout(bytes, { geometry
 * })` refuses other geometry.
 */

import { GPU_LAYOUT_INDEXED_TRIANGLE_BYTES, GPU_LAYOUT_VERSION, gpuLayoutNodeBufferBytes, gpuLayoutTriangleBytes, type GpuLayoutMeshBvh, type GpuLayoutNodeFormat, type GpuLayoutTriangleLeaves } from './gpu-layout.ts';
import type { MeshInput } from './mesh-input.ts';

/** Header word 7, bits 0-7: the triangle format. */
const TRIANGLE_FORMAT = { records: 0, indexed: 1 } as const;
/** Header word 7, bits 8-15: the node format. */
const NODE_FORMAT = { f32: 0, quantized: 1 } as const;
const NODE_FORMAT_SHIFT = 8;

/** 'PBVH' read as a little-endian u32: a tree. */
const MAGIC = 0x48564250;
/** 'PBVS' read as a little-endian u32: a scene (serialize-scene.ts). */
export const SCENE_MAGIC = 0x53564250;

/** Version of the byte format above (the header and the order of the arrays), apart from the GPU layout's own version. */
export const SERIALIZED_FORMAT_VERSION = 1;
/** Version of the scene file's byte format (serialize-scene.ts), apart from the GPU layout's own version. */
export const SERIALIZED_SCENE_FORMAT_VERSION = 1;

/** Both files' header: 16 little-endian words. */
export const SERIALIZED_HEADER_BYTES = 64;
const HEADER_BYTES = SERIALIZED_HEADER_BYTES;

const HEADER_WORD = {
    magic: 0,
    formatVersion: 1,
    layoutVersion: 2,
    nodeCount: 3,
    triangleCount: 4,
    maxDepth: 5,
    leafCount: 6,
    formats: 7,
    bounds: 8,
    geometryHash: 14,
} as const;

/** The scene file's header words (serialize-scene.ts); 0-2 are the tree's. */
export const SCENE_HEADER_WORD = {
    magic: 0,
    formatVersion: 1,
    layoutVersion: 2,
    /** Bits 0-7 triangles, 8-15 nodes (the tree's codes), 16-23 positions (0 vec4, 1 packed). */
    formats: 3,
    meshCount: 4,
    mergedGroupCount: 5,
    /** The top level's instance records. */
    instanceCount: 6,
    vertexCount: 7,
    requiredStackSize: 8,
    /** Bytes of the JSON description, padded with spaces to a multiple of 4. */
    descriptionBytes: 9,
    nodeBytes: 10,
    triangleBytes: 11,
    topLevelBytes: 12,
    /** Entries (u32) of `topLevel.recordOrder`. */
    recordOrderLength: 13,
    positionBytes: 14,
} as const;

/** Header word 3 of a scene, bits 16-23: the positions' layout. */
export const SCENE_POSITION_LAYOUT = { vec4: 0, packed: 1 } as const;
export const SCENE_POSITION_LAYOUT_SHIFT = 16;

/** The triangle and node formats of header word 7 (a tree) or 3 (a scene); null for a code this package doesn't know. */
export function decodeSerializedFormats(formats: number): { triangleLeaves: GpuLayoutTriangleLeaves | null; nodeFormat: GpuLayoutNodeFormat | null } {
    const triangles = formats & 0xff;
    const nodes = (formats >>> NODE_FORMAT_SHIFT) & 0xff;
    return {
        triangleLeaves: triangles === TRIANGLE_FORMAT.records ? 'records' : triangles === TRIANGLE_FORMAT.indexed ? 'indexed' : null,
        nodeFormat: nodes === NODE_FORMAT.f32 ? 'f32' : nodes === NODE_FORMAT.quantized ? 'quantized' : null,
    };
}

/** Header word 7 (a tree) or bits 0-15 of word 3 (a scene). */
export function encodeSerializedFormats(triangleLeaves: GpuLayoutTriangleLeaves | undefined, nodeFormat: GpuLayoutNodeFormat | undefined): number {
    const triangles = triangleLeaves === 'indexed' ? TRIANGLE_FORMAT.indexed : TRIANGLE_FORMAT.records;
    const nodes = nodeFormat === 'quantized' ? NODE_FORMAT.quantized : NODE_FORMAT.f32;
    return triangles | (nodes << NODE_FORMAT_SHIFT);
}

/** What both kinds of file say in their header. */
interface SerializedHeaderCommon {
    /** Word 1: the byte format the file was written with ({@link SERIALIZED_FORMAT_VERSION} for a tree, {@link SERIALIZED_SCENE_FORMAT_VERSION} for a scene). */
    readonly formatVersion: number;
    /** Word 2: the GPU layout the file was written with (`GPU_LAYOUT_VERSION`). */
    readonly layoutVersion: number;
    /**
     * True when this package loads the file: both versions are its own and it
     * knows the formats. When false, {@link reason} says why: bake again.
     */
    readonly current: boolean;
    /** Why the file isn't current (the message loading it would throw), or null. */
    readonly reason: string | null;
    /** `'records'` or `'indexed'`; null for a format code this package doesn't know. */
    readonly triangleLeaves: GpuLayoutTriangleLeaves | null;
    /** `'f32'` or `'quantized'`; null for a format code this package doesn't know. */
    readonly nodeFormat: GpuLayoutNodeFormat | null;
    /** Bytes the whole file should have (header included), from the header's counts. */
    readonly byteLength: number;
}

/** A tree's header ({@link serializeLayout}). */
export interface SerializedTreeHeader extends SerializedHeaderCommon {
    readonly kind: 'tree';
    readonly nodeCount: number;
    readonly triangleCount: number;
    readonly maxDepth: number;
    readonly leafCount: number;
    /** The tree's box: minX, minY, minZ, maxX, maxY, maxZ. */
    readonly bounds: Float32Array;
    /** The hash of the geometry it was written with ({@link meshGeometryHash}), or null when it was written without one. */
    readonly geometryHash: readonly [number, number] | null;
}

/** A scene's header (`serializeTraceScene`). */
export interface SerializedSceneHeader extends SerializedHeaderCommon {
    readonly kind: 'scene';
    /** `'vec4'` or `'packed'` (an indexed scene's positions); null for a code this package doesn't know. */
    readonly positionLayout: 'vec4' | 'packed' | null;
    /** The input meshes (`TraceSceneData.meshes`). */
    readonly meshCount: number;
    /** Merged static groups (`TraceSceneData.mergedGroups`). */
    readonly mergedGroupCount: number;
    /** Instance records in the top level (merged groups' included). */
    readonly instanceCount: number;
    /** Vertices in the positions (0 for records). */
    readonly vertexCount: number;
    /** Trace stack entries the scene needs (`TraceSceneData.requiredStackSize`). */
    readonly requiredStackSize: number;
}

/** What {@link readSerializedHeader} returns: `kind` tells a tree from a scene. */
export type SerializedHeader = SerializedTreeHeader | SerializedSceneHeader;

function bytesView(bytes: ArrayBuffer | ArrayBufferView): Uint8Array {
    return ArrayBuffer.isView(bytes) ? new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength) : new Uint8Array(bytes);
}

/**
 * Reads the 64-byte header of a tree ({@link serializeLayout}) or a scene
 * (`serializeTraceScene`) without loading it: what it is, the versions it
 * was written with, whether this package loads it (`current`, and `reason`
 * when not), its formats and counts. Throws only when the bytes are neither
 * (too short, or another magic). Words 3 onwards are read as this package's
 * format version places them: when `formatVersion` differs, `current` is
 * false and those fields may mean nothing.
 */
export function readSerializedHeader(bytes: ArrayBuffer | ArrayBufferView): SerializedHeader {
    const view = bytesView(bytes);
    if (view.byteLength < HEADER_BYTES) throw new RangeError(`readSerializedHeader: ${view.byteLength} bytes is shorter than the ${HEADER_BYTES}-byte header`);
    const header = new DataView(view.buffer, view.byteOffset, HEADER_BYTES);
    const word = (index: number) => header.getUint32(index * 4, true);
    const magic = word(HEADER_WORD.magic);
    if (magic !== MAGIC && magic !== SCENE_MAGIC) throw new Error('readSerializedHeader: not a serialized @pmndrs/bvh tree or scene (bad magic)');
    const kind = magic === MAGIC ? 'tree' : 'scene';
    const formatVersion = word(HEADER_WORD.formatVersion);
    const layoutVersion = word(HEADER_WORD.layoutVersion);
    const ownFormat = kind === 'tree' ? SERIALIZED_FORMAT_VERSION : SERIALIZED_SCENE_FORMAT_VERSION;

    const formatsWord = word(kind === 'tree' ? HEADER_WORD.formats : SCENE_HEADER_WORD.formats);
    const { triangleLeaves, nodeFormat } = decodeSerializedFormats(formatsWord);
    const unknownBits = kind === 'tree' ? formatsWord >>> 16 : formatsWord >>> 24;
    const positionCode = (formatsWord >>> SCENE_POSITION_LAYOUT_SHIFT) & 0xff;
    const positionLayout = positionCode === SCENE_POSITION_LAYOUT.vec4 ? 'vec4' : positionCode === SCENE_POSITION_LAYOUT.packed ? 'packed' : null;

    const what = kind === 'tree' ? 'tree' : 'scene';
    let reason: string | null = null;
    if (formatVersion !== ownFormat) reason = `byte format ${formatVersion}, this package reads ${ownFormat}; bake the ${what} again`;
    else if (layoutVersion !== GPU_LAYOUT_VERSION) reason = `GPU layout version ${layoutVersion}, this package uses ${GPU_LAYOUT_VERSION}; rebuild the ${what} (bake it again)`;
    else if (triangleLeaves === null) reason = `triangle format ${formatsWord & 0xff} is not one this package reads (0 records, 1 indexed); rebuild the ${what}`;
    else if (nodeFormat === null) reason = `node format ${(formatsWord >>> NODE_FORMAT_SHIFT) & 0xff} is not one this package reads (0 f32, 1 quantized); rebuild the ${what}`;
    else if (kind === 'scene' && positionLayout === null) reason = `position layout ${positionCode} is not one this package reads (0 vec4, 1 packed); rebuild the scene`;
    else if (unknownBits !== 0) reason = `the formats word has bits this package doesn't know (0x${formatsWord.toString(16)}); rebuild the ${what}`;
    const common = { formatVersion, layoutVersion, current: reason === null, reason, triangleLeaves, nodeFormat };

    if (kind === 'scene') {
        const at = SCENE_HEADER_WORD;
        const byteLength =
            HEADER_BYTES + word(at.descriptionBytes) + word(at.nodeBytes) + word(at.triangleBytes) + word(at.topLevelBytes) + word(at.recordOrderLength) * 4 + word(at.positionBytes);
        return {
            kind,
            ...common,
            byteLength,
            positionLayout,
            meshCount: word(at.meshCount),
            mergedGroupCount: word(at.mergedGroupCount),
            instanceCount: word(at.instanceCount),
            vertexCount: word(at.vertexCount),
            requiredStackSize: word(at.requiredStackSize),
        };
    }
    const nodeCount = word(HEADER_WORD.nodeCount);
    const triangleCount = word(HEADER_WORD.triangleCount);
    const bounds = new Float32Array(6);
    for (let i = 0; i < 6; i++) bounds[i] = header.getFloat32((HEADER_WORD.bounds + i) * 4, true);
    const hash: [number, number] = [word(HEADER_WORD.geometryHash), word(HEADER_WORD.geometryHash + 1)];
    return {
        kind,
        ...common,
        byteLength: serializedLayoutBytes({ nodeCount, triangleCount, triangleLeaves: triangleLeaves ?? 'records', nodeFormat: nodeFormat ?? 'f32' }),
        nodeCount,
        triangleCount,
        maxDepth: word(HEADER_WORD.maxDepth),
        leafCount: word(HEADER_WORD.leafCount),
        bounds,
        geometryHash: hash[0] === 0 && hash[1] === 0 ? null : hash,
    };
}

/**
 * True when `bytes` are a tree or a scene this package loads (the versions
 * its own, the formats known): the check before loading a bake, or in CI over
 * the committed ones. False for anything else, never a throw.
 */
export function isSerializedCurrent(bytes: ArrayBuffer | ArrayBufferView): boolean {
    try {
        return readSerializedHeader(bytes).current;
    } catch {
        return false;
    }
}

/**
 * A 64-bit hash (two u32s, never both 0) of a mesh's positions (their bits)
 * and indices (their values, so 16- and 32-bit indices hash alike): what a
 * tree written with its geometry records, so that loading it next to other
 * geometry is refused. A check against mistakes, not a cryptographic one.
 * Triangle flags and materials are not part of it.
 */
export function meshGeometryHash(geometry: Pick<MeshInput, 'positions' | 'indices'>): [number, number] {
    const { positions, indices } = geometry;
    let a = 0x811c9dc5;
    let b = 0x2545f491;
    const mix = (value: number) => {
        a = Math.imul(a ^ value, 0x01000193);
        b = Math.imul(b ^ value, 0x5bd1e995);
        b ^= b >>> 15;
    };
    mix(positions.length);
    mix(indices ? indices.length : 0xffffffff);
    const bits = new Uint32Array(positions.buffer, positions.byteOffset, positions.length);
    for (let i = 0; i < bits.length; i++) mix(bits[i]);
    if (indices) for (let i = 0; i < indices.length; i++) mix(indices[i]);
    // Finish both (murmur3's fmix32), so nearby inputs land far apart.
    const finish = (h: number) => {
        h ^= h >>> 16;
        h = Math.imul(h, 0x85ebca6b);
        h ^= h >>> 13;
        h = Math.imul(h, 0xc2b2ae35);
        return (h ^ (h >>> 16)) >>> 0;
    };
    const hash: [number, number] = [finish(a), finish(b)];
    if (hash[0] === 0 && hash[1] === 0) hash[1] = 1; // 0, 0 means "no geometry" in a header
    return hash;
}

/** Bytes `serializeLayout` writes for this tree. */
export function serializedLayoutBytes(layout: Pick<GpuLayoutMeshBvh, 'nodeCount' | 'triangleCount' | 'triangleLeaves' | 'nodeFormat'>): number {
    return HEADER_BYTES + gpuLayoutNodeBufferBytes(layout.nodeCount, layout.nodeFormat) + layout.nodeCount * 4 + layout.triangleCount * gpuLayoutTriangleBytes(layout.triangleLeaves);
}

/** Options of {@link serializeLayout}. */
export interface SerializeLayoutOptions {
    /**
     * The geometry the tree was built from: its hash ({@link meshGeometryHash})
     * goes into header words 14-15, and `deserializeLayout(bytes, { geometry
     * })` then refuses other geometry. Worth it for an indexed tree, which
     * only works next to its own positions. The file is otherwise the same.
     */
    readonly geometry?: Pick<MeshInput, 'positions' | 'indices'>;
}

/** Writes a tree as one `ArrayBuffer` (see the file comment for the format). */
export function serializeLayout(layout: GpuLayoutMeshBvh, options: SerializeLayoutOptions = {}): ArrayBuffer {
    const nodeBytes = gpuLayoutNodeBufferBytes(layout.nodeCount, layout.nodeFormat);
    const triangleBytes = layout.triangleCount * gpuLayoutTriangleBytes(layout.triangleLeaves);
    if (layout.nodes.byteLength !== nodeBytes) throw new RangeError(`serializeLayout: ${layout.nodeCount} ${layout.nodeFormat ?? 'f32'} nodes need ${nodeBytes} bytes, got ${layout.nodes.byteLength}`);
    if (layout.triangles.byteLength !== triangleBytes) {
        throw new RangeError(`serializeLayout: ${layout.triangleCount} triangles need ${triangleBytes} bytes, got ${layout.triangles.byteLength}`);
    }
    if (layout.parents.length !== layout.nodeCount) throw new RangeError(`serializeLayout: needs one parent per node (${layout.nodeCount}), got ${layout.parents.length}`);
    if (layout.bounds.length !== 6) throw new RangeError('serializeLayout: bounds must hold 6 floats');
    if (options.geometry) checkGeometry(layout, options.geometry, 'serializeLayout');

    const out = new ArrayBuffer(serializedLayoutBytes(layout));
    const header = new DataView(out, 0, HEADER_BYTES);
    header.setUint32(HEADER_WORD.magic * 4, MAGIC, true);
    header.setUint32(HEADER_WORD.formatVersion * 4, SERIALIZED_FORMAT_VERSION, true);
    header.setUint32(HEADER_WORD.layoutVersion * 4, GPU_LAYOUT_VERSION, true);
    header.setUint32(HEADER_WORD.nodeCount * 4, layout.nodeCount, true);
    header.setUint32(HEADER_WORD.triangleCount * 4, layout.triangleCount, true);
    header.setUint32(HEADER_WORD.maxDepth * 4, layout.maxDepth, true);
    header.setUint32(HEADER_WORD.leafCount * 4, layout.leafCount, true);
    header.setUint32(HEADER_WORD.formats * 4, encodeSerializedFormats(layout.triangleLeaves, layout.nodeFormat), true);
    for (let i = 0; i < 6; i++) header.setFloat32((HEADER_WORD.bounds + i) * 4, layout.bounds[i], true);
    if (options.geometry) {
        const [a, b] = meshGeometryHash(options.geometry);
        header.setUint32(HEADER_WORD.geometryHash * 4, a, true);
        header.setUint32((HEADER_WORD.geometryHash + 1) * 4, b, true);
    }

    const bytes = new Uint8Array(out);
    let offset = HEADER_BYTES;
    bytes.set(new Uint8Array(layout.nodes), offset);
    offset += nodeBytes;
    bytes.set(new Uint8Array(layout.triangles), offset);
    offset += triangleBytes;
    bytes.set(new Uint8Array(layout.parents.buffer, layout.parents.byteOffset, layout.parents.byteLength), offset);
    return out;
}

/** Options of {@link deserializeLayout}. */
export interface DeserializeLayoutOptions {
    /**
     * The geometry the tree will be used with (for an indexed tree, the
     * positions it reads). Checked: the triangle count, an indexed tree's
     * vertex indices against its vertex count, and, when the file was written
     * with its geometry, the hash. A mismatch throws instead of tracing wrong
     * triangles.
     */
    readonly geometry?: Pick<MeshInput, 'positions' | 'indices'>;
}

/**
 * Reads bytes written by {@link serializeLayout}. The returned arrays are
 * copies, so `bytes` may be reused. Throws on anything that isn't a whole tree
 * of this package's layout version (a scene file too: `deserializeTraceScene`
 * reads those), and on a `geometry` the tree wasn't built from.
 */
export function deserializeLayout(bytes: ArrayBuffer | ArrayBufferView, options: DeserializeLayoutOptions = {}): GpuLayoutMeshBvh {
    const view = bytesView(bytes);
    if (view.byteLength < HEADER_BYTES) throw new RangeError(`deserializeLayout: ${view.byteLength} bytes is shorter than the ${HEADER_BYTES}-byte header`);
    let header: SerializedHeader;
    try {
        header = readSerializedHeader(view);
    } catch {
        throw new Error('deserializeLayout: not a serialized @pmndrs/bvh tree (bad magic)');
    }
    if (header.kind === 'scene') throw new Error('deserializeLayout: these bytes are a scene file (serializeTraceScene); read them with deserializeTraceScene');
    if (header.formatVersion !== SERIALIZED_FORMAT_VERSION) throw new Error(`deserializeLayout: byte format ${header.formatVersion}, this package reads ${SERIALIZED_FORMAT_VERSION}`);
    if (!header.current) throw new Error(`deserializeLayout: ${header.reason}`);
    const { nodeCount, triangleCount } = header;
    const triangleLeaves = header.triangleLeaves!;
    const nodeFormat = header.nodeFormat!;
    if (view.byteLength !== header.byteLength) throw new RangeError(`deserializeLayout: ${nodeCount} nodes and ${triangleCount} triangles need ${header.byteLength} bytes, got ${view.byteLength}`);

    const nodeBytes = gpuLayoutNodeBufferBytes(nodeCount, nodeFormat);
    const triangleBytes = triangleCount * gpuLayoutTriangleBytes(triangleLeaves);
    let offset = view.byteOffset + HEADER_BYTES;
    const nodes = view.buffer.slice(offset, offset + nodeBytes) as ArrayBuffer;
    offset += nodeBytes;
    const triangles = view.buffer.slice(offset, offset + triangleBytes) as ArrayBuffer;
    offset += triangleBytes;
    const parents = new Uint32Array(view.buffer.slice(offset, offset + nodeCount * 4) as ArrayBuffer);

    const layout: GpuLayoutMeshBvh = {
        nodes,
        triangles,
        ...(triangleLeaves === 'indexed' ? { triangleLeaves } : {}),
        ...(nodeFormat === 'quantized' ? { nodeFormat } : {}),
        parents,
        nodeCount,
        triangleCount,
        maxDepth: header.maxDepth,
        leafCount: header.leafCount,
        bounds: header.bounds,
    };
    if (options.geometry) {
        checkGeometry(layout, options.geometry, 'deserializeLayout');
        const hash = header.geometryHash;
        if (hash) {
            const own = meshGeometryHash(options.geometry);
            if (own[0] !== hash[0] || own[1] !== hash[1]) {
                throw new RangeError(
                    `deserializeLayout: the tree was baked from other geometry (its hash ${hex(hash)}, this geometry's ${hex(own)}): load it next to the positions and indices it was built from, or bake it again`,
                );
            }
        }
    }
    return layout;
}

const hex = (hash: readonly [number, number]) => hash.map((part) => part.toString(16).padStart(8, '0')).join('');

/** The geometry's triangle count, and an indexed tree's vertex indices within its vertices. */
function checkGeometry(layout: GpuLayoutMeshBvh, geometry: Pick<MeshInput, 'positions' | 'indices'>, where: string): void {
    const vertexCount = Math.floor(geometry.positions.length / 3);
    const triangles = (geometry.indices ? geometry.indices.length : vertexCount) / 3;
    if (triangles !== layout.triangleCount) throw new RangeError(`${where}: the tree has ${layout.triangleCount} triangles, the geometry ${triangles}; it was built from other geometry`);
    if (layout.triangleLeaves !== 'indexed') return;
    const words = new Uint32Array(layout.triangles, 0, (layout.triangleCount * GPU_LAYOUT_INDEXED_TRIANGLE_BYTES) / 4);
    for (let at = 0; at < words.length; at += 4) {
        if (words[at] >= vertexCount || words[at + 1] >= vertexCount || words[at + 2] >= vertexCount) {
            throw new RangeError(`${where}: indexed triangle slot ${at / 4} uses a vertex past the geometry's ${vertexCount}; the tree was built from other geometry`);
        }
    }
}

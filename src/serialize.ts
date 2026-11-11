/**
 * Saving a built tree and loading it again: build once (in Node, a worker, or
 * at build time), store the bytes, and load them later with no build at all.
 *
 * The bytes are a 64-byte header followed by the layout's three arrays as they
 * are uploaded to the GPU:
 *
 *     header (16 words, little-endian)
 *         0  magic 'PBVH'
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
 *         14-15 unused (0)
 *     nodes      nodeCount × 64 bytes (f32), or (nodeCount + 1) × 32 (quantized: the frame slot, then the nodes)
 *     triangles  triangleCount × 48 bytes (records) or × 16 (indexed)
 *     parents    nodeCount × 4 bytes
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
 * does not know. An indexed file
 * holds no positions: load it next to the same geometry it was built from
 * (`CpuMeshBvh` takes it as `source`, a scene as its `geometries`).
 */

import { GPU_LAYOUT_VERSION, gpuLayoutNodeBufferBytes, gpuLayoutTriangleBytes, type GpuLayoutMeshBvh } from './gpu-layout.ts';

/** Header word 7, bits 0-7: the triangle format. */
const TRIANGLE_FORMAT = { records: 0, indexed: 1 } as const;
/** Header word 7, bits 8-15: the node format. */
const NODE_FORMAT = { f32: 0, quantized: 1 } as const;
const NODE_FORMAT_SHIFT = 8;

/** 'PBVH' read as a little-endian u32. */
const MAGIC = 0x48564250;

/** Version of the byte format above (the header and the order of the arrays), apart from the GPU layout's own version. */
export const SERIALIZED_FORMAT_VERSION = 1;

const HEADER_WORDS = 16;
const HEADER_BYTES = HEADER_WORDS * 4;

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
} as const;

/** Bytes `serializeLayout` writes for this tree. */
export function serializedLayoutBytes(layout: Pick<GpuLayoutMeshBvh, 'nodeCount' | 'triangleCount' | 'triangleLeaves' | 'nodeFormat'>): number {
    return HEADER_BYTES + gpuLayoutNodeBufferBytes(layout.nodeCount, layout.nodeFormat) + layout.nodeCount * 4 + layout.triangleCount * gpuLayoutTriangleBytes(layout.triangleLeaves);
}

/** Writes a tree as one `ArrayBuffer` (see the file comment for the format). */
export function serializeLayout(layout: GpuLayoutMeshBvh): ArrayBuffer {
    const nodeBytes = gpuLayoutNodeBufferBytes(layout.nodeCount, layout.nodeFormat);
    const triangleBytes = layout.triangleCount * gpuLayoutTriangleBytes(layout.triangleLeaves);
    if (layout.nodes.byteLength !== nodeBytes) throw new RangeError(`serializeLayout: ${layout.nodeCount} ${layout.nodeFormat ?? 'f32'} nodes need ${nodeBytes} bytes, got ${layout.nodes.byteLength}`);
    if (layout.triangles.byteLength !== triangleBytes) {
        throw new RangeError(`serializeLayout: ${layout.triangleCount} triangles need ${triangleBytes} bytes, got ${layout.triangles.byteLength}`);
    }
    if (layout.parents.length !== layout.nodeCount) throw new RangeError(`serializeLayout: needs one parent per node (${layout.nodeCount}), got ${layout.parents.length}`);
    if (layout.bounds.length !== 6) throw new RangeError('serializeLayout: bounds must hold 6 floats');

    const out = new ArrayBuffer(serializedLayoutBytes(layout));
    const header = new DataView(out, 0, HEADER_BYTES);
    header.setUint32(HEADER_WORD.magic * 4, MAGIC, true);
    header.setUint32(HEADER_WORD.formatVersion * 4, SERIALIZED_FORMAT_VERSION, true);
    header.setUint32(HEADER_WORD.layoutVersion * 4, GPU_LAYOUT_VERSION, true);
    header.setUint32(HEADER_WORD.nodeCount * 4, layout.nodeCount, true);
    header.setUint32(HEADER_WORD.triangleCount * 4, layout.triangleCount, true);
    header.setUint32(HEADER_WORD.maxDepth * 4, layout.maxDepth, true);
    header.setUint32(HEADER_WORD.leafCount * 4, layout.leafCount, true);
    const triangleFormat = layout.triangleLeaves === 'indexed' ? TRIANGLE_FORMAT.indexed : TRIANGLE_FORMAT.records;
    const nodeFormat = layout.nodeFormat === 'quantized' ? NODE_FORMAT.quantized : NODE_FORMAT.f32;
    header.setUint32(HEADER_WORD.formats * 4, triangleFormat | (nodeFormat << NODE_FORMAT_SHIFT), true);
    for (let i = 0; i < 6; i++) header.setFloat32((HEADER_WORD.bounds + i) * 4, layout.bounds[i], true);

    const bytes = new Uint8Array(out);
    let offset = HEADER_BYTES;
    bytes.set(new Uint8Array(layout.nodes), offset);
    offset += nodeBytes;
    bytes.set(new Uint8Array(layout.triangles), offset);
    offset += triangleBytes;
    bytes.set(new Uint8Array(layout.parents.buffer, layout.parents.byteOffset, layout.parents.byteLength), offset);
    return out;
}

/**
 * Reads bytes written by {@link serializeLayout}. The returned arrays are
 * copies, so `bytes` may be reused. Throws on anything that isn't a whole tree
 * of this package's layout version.
 */
export function deserializeLayout(bytes: ArrayBuffer | ArrayBufferView): GpuLayoutMeshBvh {
    const view = ArrayBuffer.isView(bytes) ? new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength) : new Uint8Array(bytes);
    if (view.byteLength < HEADER_BYTES) throw new RangeError(`deserializeLayout: ${view.byteLength} bytes is shorter than the ${HEADER_BYTES}-byte header`);
    const header = new DataView(view.buffer, view.byteOffset, HEADER_BYTES);
    const word = (index: number) => header.getUint32(index * 4, true);

    if (word(HEADER_WORD.magic) !== MAGIC) throw new Error('deserializeLayout: not a serialized @pmndrs/bvh tree (bad magic)');
    const formatVersion = word(HEADER_WORD.formatVersion);
    if (formatVersion !== SERIALIZED_FORMAT_VERSION) throw new Error(`deserializeLayout: byte format ${formatVersion}, this package reads ${SERIALIZED_FORMAT_VERSION}`);
    const layoutVersion = word(HEADER_WORD.layoutVersion);
    if (layoutVersion !== GPU_LAYOUT_VERSION) throw new Error(`deserializeLayout: GPU layout version ${layoutVersion}, this package uses ${GPU_LAYOUT_VERSION}; rebuild the tree`);

    const nodeCount = word(HEADER_WORD.nodeCount);
    const triangleCount = word(HEADER_WORD.triangleCount);
    const formats = word(HEADER_WORD.formats);
    const format = formats & 0xff;
    const nodeFormatCode = formats >>> NODE_FORMAT_SHIFT;
    if (format !== TRIANGLE_FORMAT.records && format !== TRIANGLE_FORMAT.indexed) {
        throw new Error(`deserializeLayout: triangle format ${format} is not one this package reads (0 records, 1 indexed); rebuild the tree`);
    }
    if (nodeFormatCode !== NODE_FORMAT.f32 && nodeFormatCode !== NODE_FORMAT.quantized) {
        throw new Error(`deserializeLayout: node format ${nodeFormatCode} is not one this package reads (0 f32, 1 quantized); rebuild the tree`);
    }
    const triangleLeaves = format === TRIANGLE_FORMAT.indexed ? 'indexed' : 'records';
    const nodeFormat = nodeFormatCode === NODE_FORMAT.quantized ? 'quantized' : 'f32';
    const expected = serializedLayoutBytes({ nodeCount, triangleCount, triangleLeaves, nodeFormat });
    if (view.byteLength !== expected) throw new RangeError(`deserializeLayout: ${nodeCount} nodes and ${triangleCount} triangles need ${expected} bytes, got ${view.byteLength}`);

    const bounds = new Float32Array(6);
    for (let i = 0; i < 6; i++) bounds[i] = header.getFloat32((HEADER_WORD.bounds + i) * 4, true);

    const nodeBytes = gpuLayoutNodeBufferBytes(nodeCount, nodeFormat);
    const triangleBytes = triangleCount * gpuLayoutTriangleBytes(triangleLeaves);
    let offset = view.byteOffset + HEADER_BYTES;
    const nodes = view.buffer.slice(offset, offset + nodeBytes) as ArrayBuffer;
    offset += nodeBytes;
    const triangles = view.buffer.slice(offset, offset + triangleBytes) as ArrayBuffer;
    offset += triangleBytes;
    const parents = new Uint32Array(view.buffer.slice(offset, offset + nodeCount * 4) as ArrayBuffer);

    return {
        nodes,
        triangles,
        ...(triangleLeaves === 'indexed' ? { triangleLeaves } : {}),
        ...(nodeFormat === 'quantized' ? { nodeFormat } : {}),
        parents,
        nodeCount,
        triangleCount,
        maxDepth: word(HEADER_WORD.maxDepth),
        leafCount: word(HEADER_WORD.leafCount),
        bounds,
    };
}

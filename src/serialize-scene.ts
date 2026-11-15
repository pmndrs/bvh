/**
 * Saving a whole assembled scene (`TraceSceneData`, from `assembleTraceScene`
 * or `buildTraceScene`) and loading it again: every mesh tree, the merged
 * static groups, the top level with its instances (transforms, masks,
 * materials), the triangles with their materials, the indexed positions when
 * the scene has them, and the formats. A loaded scene goes straight to
 * `uploadTraceScene`: no build and no assembly.
 *
 *     header (16 words, little-endian)
 *         0  magic 'PBVS'
 *         1  format version (SERIALIZED_SCENE_FORMAT_VERSION)
 *         2  GPU layout version (GPU_LAYOUT_VERSION)
 *         3  formats: bits 0-7 triangles (0 records, 1 indexed), bits 8-15
 *            nodes (0 f32, 1 quantized), bits 16-23 positions (0 vec4,
 *            1 packed), bits 24-31 0
 *         4  meshCount (TraceSceneData.meshes)
 *         5  mergedGroupCount (TraceSceneData.mergedGroups)
 *         6  instanceCount (the top level's records)
 *         7  vertexCount
 *         8  requiredStackSize
 *         9  description bytes (JSON, padded with spaces to a multiple of 4)
 *         10 node bytes
 *         11 triangle bytes
 *         12 top-level bytes
 *         13 recordOrder entries
 *         14 position bytes
 *         15 0
 *     description  JSON: the placements (`meshes`, `mergedGroups`), the top
 *                  level's node count, depth and singular instances, and
 *                  `maxMeshDepth`
 *     nodes, triangles, top level   the scene's buffers, byte for byte
 *     recordOrder                   u32 each
 *     positions                     the indexed scene's positions (none for records)
 *
 * Words 0-2 are the tree file's (serialize.ts), so `readSerializedHeader`
 * reads both. A scene from another GPU layout or format version is refused,
 * not converted, as a tree is. A scene file is the snapshot of one assembled
 * scene: its instances can't move (the top level is baked), and
 * `clusterTraceScene`'s statistics are not kept (its placements are).
 */

import { GPU_LAYOUT_VERSION, gpuLayoutNodeBytes, INDEXED_TRIANGLE_BYTES, positionStrideFloats, TRACE_STACK_SIZE, TRIANGLE_BYTES } from './gpu-layout.ts';
import {
    encodeSerializedFormats,
    readSerializedHeader,
    SCENE_HEADER_WORD,
    SCENE_MAGIC,
    SCENE_POSITION_LAYOUT,
    SCENE_POSITION_LAYOUT_SHIFT,
    SERIALIZED_HEADER_BYTES,
    SERIALIZED_SCENE_FORMAT_VERSION,
} from './serialize.ts';
import type { TraceSceneData, TraceSceneMergedGroup, TraceSceneMeshPlacement } from './trace/scene-assembly.ts';
import type { TopLevelBvh } from './trace/top-level.ts';

/** The JSON between the header and the buffers. */
interface SceneDescription {
    readonly meshes: readonly TraceSceneMeshPlacement[];
    readonly mergedGroups: readonly TraceSceneMergedGroup[];
    readonly topLevel: { readonly nodeCount: number; readonly maxDepth: number; readonly skippedSingular: number };
    readonly maxMeshDepth: number;
}

/** A placement's fields in one order, so the same scene always writes the same bytes. */
function placementJson(placement: TraceSceneMeshPlacement): TraceSceneMeshPlacement {
    return {
        nodeOffset: placement.nodeOffset,
        nodeCount: placement.nodeCount,
        triangleOffset: placement.triangleOffset,
        triangleCount: placement.triangleCount,
        ...(placement.mergedInto !== undefined ? { mergedInto: placement.mergedInto } : {}),
        ...(placement.vertexBase !== undefined ? { vertexBase: placement.vertexBase } : {}),
        ...(placement.vertexCount !== undefined ? { vertexCount: placement.vertexCount } : {}),
        ...(placement.welded !== undefined ? { welded: placement.welded } : {}),
        ...(placement.clustered ? { clustered: true as const } : {}),
    };
}

function groupJson(group: TraceSceneMergedGroup): TraceSceneMergedGroup {
    return { ...placementJson(group), merged: true, sourceInstances: [...group.sourceInstances], sourceMeshes: [...group.sourceMeshes] };
}

/** Writes an assembled scene as one `ArrayBuffer` (see the file comment for the format). */
export function serializeTraceScene(scene: TraceSceneData): ArrayBuffer {
    const where = 'serializeTraceScene';
    const leaves = scene.triangleLeaves ?? 'records';
    const indexed = leaves === 'indexed';
    const positionLayout = scene.positionLayout ?? 'vec4';
    if (positionLayout !== 'vec4' && positionLayout !== 'packed') throw new RangeError(`${where}: positionLayout must be 'vec4' or 'packed', got ${String(positionLayout)}`);
    const positions = indexed ? scene.positions : new ArrayBuffer(0);
    const { topLevel } = scene;
    if (topLevel.recordOrder.length !== topLevel.instanceCount) {
        throw new RangeError(`${where}: the top level has ${topLevel.instanceCount} instances but ${topLevel.recordOrder.length} recordOrder entries`);
    }
    for (const [name, buffer] of [['nodes', scene.nodes], ['triangles', scene.triangles], ['top level', topLevel.buffer], ['positions', positions]] as const) {
        if (buffer.byteLength % 4 !== 0) throw new RangeError(`${where}: the scene's ${name} hold ${buffer.byteLength} bytes, not whole words`);
    }

    const description: SceneDescription = {
        meshes: scene.meshes.map(placementJson),
        mergedGroups: scene.mergedGroups.map(groupJson),
        topLevel: { nodeCount: topLevel.nodeCount, maxDepth: topLevel.maxDepth, skippedSingular: topLevel.skippedSingular },
        maxMeshDepth: scene.maxMeshDepth,
    };
    let json = JSON.stringify(description);
    // Pad with spaces (JSON ignores them) so the buffers start on a word.
    let text = new TextEncoder().encode(json);
    if (text.byteLength % 4 !== 0) {
        json += ' '.repeat(4 - (text.byteLength % 4));
        text = new TextEncoder().encode(json);
    }

    const recordOrderBytes = topLevel.recordOrder.length * 4;
    const total = SERIALIZED_HEADER_BYTES + text.byteLength + scene.nodes.byteLength + scene.triangles.byteLength + topLevel.buffer.byteLength + recordOrderBytes + positions.byteLength;
    const out = new ArrayBuffer(total);
    const header = new DataView(out, 0, SERIALIZED_HEADER_BYTES);
    const set = (word: number, value: number) => header.setUint32(word * 4, value, true);
    const at = SCENE_HEADER_WORD;
    set(at.magic, SCENE_MAGIC);
    set(at.formatVersion, SERIALIZED_SCENE_FORMAT_VERSION);
    set(at.layoutVersion, GPU_LAYOUT_VERSION);
    set(at.formats, (encodeSerializedFormats(leaves, scene.nodeFormat) | (SCENE_POSITION_LAYOUT[positionLayout] << SCENE_POSITION_LAYOUT_SHIFT)) >>> 0);
    set(at.meshCount, scene.meshes.length);
    set(at.mergedGroupCount, scene.mergedGroups.length);
    set(at.instanceCount, topLevel.instanceCount);
    set(at.vertexCount, indexed ? scene.vertexCount : 0);
    set(at.requiredStackSize, scene.requiredStackSize);
    set(at.descriptionBytes, text.byteLength);
    set(at.nodeBytes, scene.nodes.byteLength);
    set(at.triangleBytes, scene.triangles.byteLength);
    set(at.topLevelBytes, topLevel.buffer.byteLength);
    set(at.recordOrderLength, topLevel.recordOrder.length);
    set(at.positionBytes, positions.byteLength);

    const bytes = new Uint8Array(out);
    let offset = SERIALIZED_HEADER_BYTES;
    const put = (part: Uint8Array) => {
        bytes.set(part, offset);
        offset += part.byteLength;
    };
    put(text);
    put(new Uint8Array(scene.nodes));
    put(new Uint8Array(scene.triangles));
    put(new Uint8Array(topLevel.buffer));
    put(new Uint8Array(topLevel.recordOrder.buffer, topLevel.recordOrder.byteOffset, recordOrderBytes));
    put(new Uint8Array(positions));
    return out;
}

/**
 * Reads bytes written by {@link serializeTraceScene} into a `TraceSceneData`
 * ready for `uploadTraceScene` (or `createCpuTracer`). The buffers are
 * copies, so `bytes` may be reused. Throws, saying why, on anything that
 * isn't a whole scene of this package's versions: another magic (a tree file
 * says to use `deserializeLayout`), another scene format or GPU layout
 * version (bake again), a format it doesn't know, a size that doesn't add up,
 * or a scene deeper than the walk's stack.
 */
export function deserializeTraceScene(bytes: ArrayBuffer | ArrayBufferView): TraceSceneData {
    const where = 'deserializeTraceScene';
    const view = ArrayBuffer.isView(bytes) ? new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength) : new Uint8Array(bytes);
    if (view.byteLength < SERIALIZED_HEADER_BYTES) throw new RangeError(`${where}: ${view.byteLength} bytes is shorter than the ${SERIALIZED_HEADER_BYTES}-byte header`);
    let header: ReturnType<typeof readSerializedHeader>;
    try {
        header = readSerializedHeader(view);
    } catch {
        throw new Error(`${where}: not a serialized @pmndrs/bvh scene (bad magic)`);
    }
    if (header.kind === 'tree') throw new Error(`${where}: these bytes are one mesh's tree (serializeLayout); read them with deserializeLayout`);
    if (!header.current) throw new Error(`${where}: ${header.reason}`);
    if (view.byteLength !== header.byteLength) throw new RangeError(`${where}: the header's sizes add up to ${header.byteLength} bytes, got ${view.byteLength}; the file is cut short or corrupt`);
    const triangleLeaves = header.triangleLeaves!;
    const nodeFormat = header.nodeFormat!;
    const positionLayout = header.positionLayout!;
    const indexed = triangleLeaves === 'indexed';
    if (header.requiredStackSize > TRACE_STACK_SIZE) {
        throw new RangeError(`${where}: the scene needs ${header.requiredStackSize} trace stack entries, this package's walk holds ${TRACE_STACK_SIZE}; bake it again`);
    }

    const words = new DataView(view.buffer, view.byteOffset, SERIALIZED_HEADER_BYTES);
    const word = (index: number) => words.getUint32(index * 4, true);
    const at = SCENE_HEADER_WORD;
    let offset = view.byteOffset + SERIALIZED_HEADER_BYTES;
    const take = (length: number): ArrayBuffer => {
        const part = view.buffer.slice(offset, offset + length) as ArrayBuffer;
        offset += length;
        return part;
    };
    let description: SceneDescription;
    try {
        description = JSON.parse(new TextDecoder().decode(take(word(at.descriptionBytes)))) as SceneDescription;
    } catch (error) {
        throw new Error(`${where}: the scene's description is not valid JSON (${String(error)}); the file is corrupt`);
    }
    const nodes = take(word(at.nodeBytes));
    const triangles = take(word(at.triangleBytes));
    const topBuffer = take(word(at.topLevelBytes));
    const recordOrder = new Uint32Array(take(word(at.recordOrderLength) * 4));
    const positions = take(word(at.positionBytes));

    const nodeBytes = gpuLayoutNodeBytes(nodeFormat);
    const unitBytes = indexed ? INDEXED_TRIANGLE_BYTES : TRIANGLE_BYTES;
    if (nodes.byteLength % nodeBytes !== 0) throw new RangeError(`${where}: ${nodes.byteLength} node bytes are not whole ${nodeBytes}-byte ${nodeFormat} nodes`);
    if (triangles.byteLength % unitBytes !== 0) throw new RangeError(`${where}: ${triangles.byteLength} triangle bytes are not whole ${unitBytes}-byte ${triangleLeaves} triangles`);
    if (recordOrder.length !== header.instanceCount) throw new RangeError(`${where}: ${header.instanceCount} instances but ${recordOrder.length} recordOrder entries`);
    if (!indexed && (positions.byteLength !== 0 || header.vertexCount !== 0)) throw new RangeError(`${where}: a records scene holds positions; the file is corrupt`);
    if (indexed && positions.byteLength < header.vertexCount * positionStrideFloats(positionLayout) * 4) {
        throw new RangeError(`${where}: ${header.vertexCount} ${positionLayout} vertices don't fit in ${positions.byteLength} position bytes`);
    }

    const { meshes, mergedGroups, topLevel: top, maxMeshDepth } = description;
    if (!Array.isArray(meshes) || meshes.length !== header.meshCount || !Array.isArray(mergedGroups) || mergedGroups.length !== header.mergedGroupCount || !top) {
        throw new RangeError(`${where}: the description doesn't match the header (${header.meshCount} meshes, ${header.mergedGroupCount} merged groups); the file is corrupt`);
    }
    const nodeSlots = nodes.byteLength / nodeBytes;
    const triangleUnits = triangles.byteLength / unitBytes;
    const checkPlacement = (placement: TraceSceneMeshPlacement, name: string) => {
        const fields = [placement.nodeOffset, placement.nodeCount, placement.triangleOffset, placement.triangleCount];
        if (!fields.every((value) => Number.isInteger(value) && value >= 0)) throw new RangeError(`${where}: ${name} has a placement that isn't whole numbers; the file is corrupt`);
        if (placement.nodeOffset + placement.nodeCount > nodeSlots || placement.triangleOffset + placement.triangleCount > triangleUnits) {
            throw new RangeError(`${where}: ${name} lies past the scene's ${nodeSlots} node slots or ${triangleUnits} triangle units; the file is corrupt`);
        }
    };
    meshes.forEach((placement, index) => checkPlacement(placement, `mesh ${index}`));
    mergedGroups.forEach((group, index) => checkPlacement(group, `merged group ${index}`));

    const topLevel: TopLevelBvh = {
        buffer: topBuffer,
        instanceCount: header.instanceCount,
        nodeCount: top.nodeCount,
        maxDepth: top.maxDepth,
        skippedSingular: top.skippedSingular,
        recordOrder,
    };
    return {
        nodes,
        nodeFormat,
        triangles,
        triangleLeaves,
        positions,
        positionLayout,
        vertexCount: header.vertexCount,
        topLevel,
        meshes,
        maxMeshDepth,
        requiredStackSize: header.requiredStackSize,
        byteLength: nodes.byteLength + triangles.byteLength + topBuffer.byteLength + positions.byteLength,
        mergedGroups,
    };
}

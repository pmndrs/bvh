/**
 * The tracer's GPU layout, as tinybvh's WASM writes it (stage B0b): what a
 * builder hands to the tracer when it writes the layout itself instead of a
 * `MeshTreeTopology` (topology.ts). The layout is defined, word by word, in
 * `src/trace/layout.ts` (and the WGSL generated
 * from it); `native/tinybvh-wasm.cpp` repeats it as C structs.
 *
 * The sizes and limits below are the one definition: layout.ts imports them
 * (this package can't import the core). A test in ddgi-core checks that the
 * WASM's bytes equal the JavaScript conversion's.
 */

/**
 * Version of the GPU layout below. The layout is part of the package's API:
 * a WGSL walk written against it reads every word position here. Any change
 * to a size, a word position or a meaning bumps this number, and that is a
 * breaking change. Code with its own walk checks it at startup; serialized
 * trees (serialize.ts) carry it and are refused when it differs.
 *
 * History: 1, the first layout. 2, instance records carry an 8-bit instance
 * mask in bits 8-15 of their `flags` word (written as 0xff, visible to every
 * ray, unless the instance gives one). Also in 2, before its release: merged
 * static meshes. Instance flags bit 3 (`merged`) says the hit's instance is
 * the triangle's `padding` word (the source instance index), and triangle
 * flags bit 1 flips `frontFace`. Builders still write 0 to both; nothing moved.
 * Also in 2, before its release: bits 16-31 of the instance `flags` word hold
 * the instance's material + 1 (0 for none), and bits 16-31 of each
 * triangle's `flags` word the triangle's (`MeshInput.triangleMaterials`,
 * which builders write; in a merged mesh its source's, which the scene
 * assembly writes). A hit takes the triangle's, else the instance's.
 * Also in 2, before its release: indexed leaves, a second triangle format
 * beside the records ({@link GpuLayoutTriangleLeaves}). It is a variant of
 * version 2, not a new version: serialized trees name their format in header
 * word 7 (0 records, 1 indexed), and records are unchanged.
 * Also in 2, before its release: quantized nodes, a second node format beside
 * the 64-byte nodes ({@link GpuLayoutNodeFormat}). A variant of version 2 too:
 * serialized trees name it in bits 8-15 of header word 7 (0 `f32`, 1
 * quantized), and 64-byte nodes are unchanged.
 */
export const GPU_LAYOUT_VERSION = 2;

/** One node: two child boxes (Aila-Laine layout), 64 bytes. */
export const GPU_LAYOUT_NODE_BYTES = 64;
export const GPU_LAYOUT_NODE_WORDS = GPU_LAYOUT_NODE_BYTES / 4;

/**
 * Word positions inside a node, matching the tracer's WGSL `BvhNode`:
 *
 *     leftMin: vec3f,  leftIndex: u32,
 *     leftMax: vec3f,  rightIndex: u32,
 *     rightMin: vec3f, leftCount: u32,
 *     rightMax: vec3f, rightCount: u32,
 *
 * A child with `count > 0` is a leaf holding `count` primitives starting at
 * `index`; `count == 0` is an inner node at `index`. Child `side` (0 left,
 * 1 right) keeps its box at words `8 * side` (min) and `8 * side + 4` (max).
 */
export const GPU_LAYOUT_NODE_WORD = {
    leftMin: 0,
    leftIndex: 3,
    leftMax: 4,
    rightIndex: 7,
    rightMin: 8,
    leftCount: 11,
    rightMax: 12,
    rightCount: 15,
} as const;

/**
 * How a tree stores its nodes (docs/internals/design/compressed-nodes.md). Every tree
 * and scene is one or the other, and the walk is compiled for one
 * (`traceModuleWgsl({ nodeFormat })`):
 *
 * - `'f32'`: 64-byte nodes with `f32` child boxes,
 *   {@link GPU_LAYOUT_NODE_WORD}. The format of a single mesh's tree from
 *   every builder unless asked otherwise (refits run on it), and of data
 *   that names no format.
 * - `'quantized'` (the default for scenes and walks,
 *   {@link DEFAULT_NODE_FORMAT}): 32-byte nodes, {@link GPU_LAYOUT_QUANTIZED_NODE_WORD}: each
 *   child box as 16-bit planes on a grid (the frame) that covers the mesh, and
 *   each child as one stack reference. The buffer starts with the frame slot
 *   (32 bytes, {@link GPU_LAYOUT_NODE_FRAME_WORD}), then the nodes: node `i`
 *   is slot `i + 1`, so `nodes` holds `nodeCount + 1` slots. A scene keeps the
 *   frame slot right before each mesh's root (`rootNodeOffset - 1`). Boxes are
 *   rounded outward, so a decoded box always contains the exact one.
 */
export type GpuLayoutNodeFormat = 'f32' | 'quantized';

/**
 * The node format of every scene and walk option that isn't given one:
 * `SceneTrees`, `assembleTraceScene`, `buildTraceScene`, `traceModuleWgsl`,
 * `batchKernelWgsl`, `TraceBatchKernel` and `BvhTraceNodes` (Dennis,
 * 2026-11-13, after the RTX 3060 bench: docs/internals/design/compressed-nodes.md).
 * Mesh builders still write `'f32'` trees unless asked; a scene quantizes
 * them as it places them.
 */
export const DEFAULT_NODE_FORMAT: GpuLayoutNodeFormat = 'quantized';

/** One quantized node: two child boxes as 16-bit planes and two references, 32 bytes (two `vec4u`). */
export const GPU_LAYOUT_QUANTIZED_NODE_BYTES = 32;
export const GPU_LAYOUT_QUANTIZED_NODE_WORDS = GPU_LAYOUT_QUANTIZED_NODE_BYTES / 4;

/**
 * Word positions inside a quantized node. Child `side` (0 left, 1 right) has
 * its three axis words at `4 * side` and its reference at `4 * side + 3`. An
 * axis word holds the min plane in bits 0-15 and the max plane in bits 16-31,
 * each a step count from the frame's origin: plane = `origin + q * step`. A
 * reference is the walk's stack reference: index in bits 0-27, leaf count in
 * bits 28-31 (0 for an inner node); a missing right child is
 * {@link GPU_LAYOUT_EMPTY_CHILD}.
 */
export const GPU_LAYOUT_QUANTIZED_NODE_WORD = {
    leftX: 0,
    leftY: 1,
    leftZ: 2,
    leftReference: 3,
    rightX: 4,
    rightY: 5,
    rightZ: 6,
    rightReference: 7,
} as const;

/**
 * Word positions inside a frame slot (32 bytes, the size of a quantized
 * node): the grid's origin and its step per axis, as `f32`. Each step is a
 * power of two and each origin a multiple of its step, so every plane
 * `origin + q * step` is an exact `f32` (`nodeFrameOf`, quantized-nodes.ts).
 * Words 3 and 7 are 0.
 */
export const GPU_LAYOUT_NODE_FRAME_WORD = {
    origin: 0,
    step: 4,
} as const;

/** Bytes per node slot of a format. */
export function gpuLayoutNodeBytes(format: GpuLayoutNodeFormat | undefined): number {
    return format === 'quantized' ? GPU_LAYOUT_QUANTIZED_NODE_BYTES : GPU_LAYOUT_NODE_BYTES;
}

/** Slots a tree's `nodes` takes: `nodeCount`, plus the frame slot when quantized. */
export function gpuLayoutNodeSlots(nodeCount: number, format: GpuLayoutNodeFormat | undefined): number {
    return format === 'quantized' ? nodeCount + 1 : nodeCount;
}

/** Bytes a tree's `nodes` takes in a format. */
export function gpuLayoutNodeBufferBytes(nodeCount: number, format: GpuLayoutNodeFormat | undefined): number {
    return gpuLayoutNodeSlots(nodeCount, format) * gpuLayoutNodeBytes(format);
}

/** Words between the left child's box and the right child's box (see {@link GPU_LAYOUT_NODE_WORD}). */
export const GPU_LAYOUT_CHILD_STRIDE_WORDS = 8;

/**
 * `index` of a child that does not exist. Only a tree with a single primitive
 * has one: its root holds that primitive on the left and nothing on the right.
 */
export const GPU_LAYOUT_EMPTY_CHILD = 0xffffffff;

/** Parent of the root node in a parent array. */
export const GPU_LAYOUT_NO_PARENT = 0xffffffff;

/** One triangle in tree order: three corners, the original index and flags, 48 bytes. */
export const GPU_LAYOUT_TRIANGLE_BYTES = 48;

/**
 * Word positions inside a triangle, matching the tracer's WGSL `BvhTriangle`
 * (and ddgi-core's `TRIANGLE_WORD`): each corner is a `vec3f` followed by one
 * word, the original triangle index after the first, the flags after the second.
 * The word after the third (`padding`) is 0 as every builder writes it; in a
 * merged static mesh the scene assembly writes the source instance index there.
 */
export const GPU_LAYOUT_TRIANGLE_WORD = {
    vertex0: 0,
    originalIndex: 3,
    vertex1: 4,
    flags: 7,
    vertex2: 8,
    padding: 11,
} as const;

/**
 * How a tree stores its triangles (docs/internals/design/indexed-leaves.md). Every tree
 * and scene is one or the other, and the walk is compiled for one
 * (`traceModuleWgsl({ triangleLeaves })`):
 *
 * - `'records'` (the default): 48-byte records with the three corners copied,
 *   {@link GPU_LAYOUT_TRIANGLE_WORD}.
 * - `'indexed'`: 16-byte records `vec4u(i0, i1, i2, id + flags)`,
 *   {@link GPU_LAYOUT_INDEXED_TRIANGLE_WORD}, whose corners are read from a
 *   positions buffer the scene owns. In a tree (`GpuLayoutMeshBvh`) the vertex
 *   indices are the mesh's own. A scene keeps them mesh-relative too, and puts
 *   the mesh's vertex base in a header before its triangles
 *   ({@link GPU_LAYOUT_MESH_HEADER_WORD}), so trees are placed and moved
 *   without rewriting their records.
 */
export type GpuLayoutTriangleLeaves = 'records' | 'indexed';

/** One indexed triangle in tree order: three vertex indices, then the original index and flags, 16 bytes. */
export const GPU_LAYOUT_INDEXED_TRIANGLE_BYTES = 16;

/** Word positions inside an indexed triangle (a `vec4u`). */
export const GPU_LAYOUT_INDEXED_TRIANGLE_WORD = {
    vertex0: 0,
    vertex1: 1,
    vertex2: 2,
    /** Bits 0-27 the original triangle index, bit 28 the filter flag, bit 29 the flip flag, bits 30-31 reserved (0). */
    idFlags: 3,
} as const;

/** Bits 0-27 of an indexed triangle's word 3: the original index (it fits: {@link GPU_LAYOUT_MAX_TRIANGLES}). */
export const GPU_LAYOUT_INDEXED_ID_MASK = 0x0fffffff;
/**
 * Where the records' flags bits 0-1 (filter, flip) sit in an indexed
 * triangle's word 3: `flags << 28`. A record's material bits (16-31, merged
 * meshes only) move to the merged side table instead.
 */
export const GPU_LAYOUT_INDEXED_FLAG_SHIFT = 28;

/** Bytes per triangle of a format. */
export function gpuLayoutTriangleBytes(leaves: GpuLayoutTriangleLeaves | undefined): number {
    return leaves === 'indexed' ? GPU_LAYOUT_INDEXED_TRIANGLE_BYTES : GPU_LAYOUT_TRIANGLE_BYTES;
}

/**
 * An indexed scene puts one 16-byte header (a `vec4u`) right before each
 * mesh's triangles, at `triangleOffset - 1` in 16-byte units: the walk reads
 * it once per instance it enters. A merged mesh keeps its side tables before
 * the header.
 */
export const GPU_LAYOUT_MESH_HEADER_WORD = {
    /** The mesh's first vertex in the scene's positions buffer. */
    vertexBase: 0,
    /** Merged meshes: `triangleOffset` minus the side table's first `vec4u` (0 otherwise). */
    sideTableDistance: 1,
    /** Merged meshes: `triangleOffset` minus the source table's first `vec4u` (0 otherwise). */
    sourceTableDistance: 2,
    /**
     * Other meshes with per-triangle materials: `triangleOffset` minus their
     * material table's first `vec4u` (0: none). The table holds each original
     * triangle's material code (material + 1, 0 for none) as 16 bits, two to a
     * word, eight to a `vec4u`, by original index. A merged mesh keeps its
     * materials in its side table instead.
     */
    materialTableDistance: 3,
} as const;

/** The tracer's largest leaf; larger leaves are cut in half until they fit. */
export const GPU_LAYOUT_MAX_LEAF_PRIMITIVES = 8;

/** Most triangles one mesh may have: triangle positions must fit the tracer's 28-bit references. */
export const GPU_LAYOUT_MAX_TRIANGLES = 0x0fffffff;

/** A mesh's tree in the GPU layout. The same shape as the tracer's `MeshBvh`. */
export interface GpuLayoutMeshBvh {
    /**
     * `nodeCount` nodes, 64 bytes each; node 0 is the root. With
     * `nodeFormat: 'quantized'`, the frame slot and then `nodeCount` 32-byte
     * nodes ({@link gpuLayoutNodeBufferBytes}); node `i` is slot `i + 1`.
     */
    readonly nodes: ArrayBuffer;
    /** How `nodes` stores them; omitted means `'f32'`. */
    readonly nodeFormat?: GpuLayoutNodeFormat;
    /** `triangleCount` triangles in tree order: 48 bytes each, or 16 with `triangleLeaves: 'indexed'`. */
    readonly triangles: ArrayBuffer;
    /** How `triangles` stores them; omitted means `'records'`. */
    readonly triangleLeaves?: GpuLayoutTriangleLeaves;
    /** Parent of each node (0xffffffff for the root), for a bottom-up refit. */
    readonly parents: Uint32Array;
    readonly nodeCount: number;
    readonly triangleCount: number;
    /** Levels of nodes on the longest path. */
    readonly maxDepth: number;
    readonly leafCount: number;
    /** Object-space box around the mesh: `[minX, minY, minZ, maxX, maxY, maxZ]`. */
    readonly bounds: Float32Array;
}

/**
 * The GPU layout, in one place: a mesh's tree as every builder writes it, and
 * the scene around the trees as the walk reads it. The CPU and GPU builders,
 * the refits and the WGSL must agree on every number here, so the WGSL source
 * is generated from these constants rather than repeating them, and
 * `native/tinybvh-wasm.cpp` repeats the tree's part as C structs (tinybvh's
 * WASM writes the layout itself, stage B0b). A test checks that the WASM's
 * bytes equal the JavaScript conversion's.
 *
 * The first part is the trees (`GPU_LAYOUT_*`, the names the package's
 * builders use). The second is the tracer's names for the same words and the
 * rest of a scene: instance records, the top level, stack references, masks
 * and materials. (The second part was src/trace/layout.ts, which moved from
 * ddgi-core in BX.2.)
 *
 * Everything is plain 32-bit words in a few buffers so it survives a worker
 * hand-off (transferable `ArrayBuffer`s) and a GPU upload without conversion.
 * See docs/internals/design/trace.md for the reasoning.
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
 * The node format of every option that isn't given one, where the library
 * makes both sides: `SceneTrees`, `assembleTraceScene`, `buildTraceScene`,
 * `traceModuleWgsl`, `batchKernelWgsl` and `TraceBatchKernel.create` (Dennis,
 * 2026-11-13, after the RTX 3060 bench: docs/internals/design/compressed-nodes.md).
 * Mesh builders still write `'f32'` trees unless asked; a scene quantizes
 * them as it places them.
 *
 * Options default to quantized; data must say what it is. Buffers handed to a
 * walk (`TraceSceneBuffers`, a `BvhTraceNodes` source) must name their format
 * and are refused when they don't; CPU data that names none (`MeshBvh`,
 * layouts, `TraceSceneData`) is `f32`, as before the option existed.
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

// ---- The tracer's names, and the scene around the trees ----------------------

/**
 * The version of the GPU layout ({@link GPU_LAYOUT_VERSION}) that the
 * tracer's WGSL structs and the walk in trace/wgsl.ts were written against. A test
 * fails when the package's version moves, so a layout change can't reach the
 * tracer without someone reviewing the walk.
 */
export const TRACER_GPU_LAYOUT_VERSION = 2;

/**
 * How a scene stores its mesh nodes (docs/internals/design/compressed-nodes.md):
 * `'quantized'` (the default, `DEFAULT_NODE_FORMAT`: 32-byte nodes on a
 * per-mesh frame, `GpuLayoutNodeFormat`) or `'f32'` (the 64-byte nodes
 * below, what builders write). The walk is compiled
 * for one (`traceModuleWgsl({ nodeFormat })`); the top level is always `f32`.
 */
export type NodeFormat = GpuLayoutNodeFormat;
/** One quantized node: 32 bytes, two `vec4u` (`GPU_LAYOUT_QUANTIZED_NODE_WORD`). The frame slot before a mesh's root is the same size. */
export const QUANTIZED_NODE_BYTES = GPU_LAYOUT_QUANTIZED_NODE_BYTES;
export const QUANTIZED_NODE_WORDS = QUANTIZED_NODE_BYTES / 4;

/** One BVH node: two child boxes (Aila-Laine layout), 16 words. */
export const BVH_NODE_BYTES = GPU_LAYOUT_NODE_BYTES;
export const BVH_NODE_WORDS = BVH_NODE_BYTES / 4;

/**
 * Word positions inside a node. Matches the WGSL `BvhNode`:
 *
 *     leftMin: vec3f,  leftIndex: u32,
 *     leftMax: vec3f,  rightIndex: u32,
 *     rightMin: vec3f, leftCount: u32,
 *     rightMax: vec3f, rightCount: u32,
 *
 * A child with `count > 0` is a leaf holding `count` primitives starting at
 * `index`. A child with `count == 0` is an inner node at `index`. The same
 * object as {@link GPU_LAYOUT_NODE_WORD}, which the GPU refit writes too.
 */
export const NODE_WORD = GPU_LAYOUT_NODE_WORD;

/**
 * `index` value of a child that does not exist. Only a tree with a single
 * primitive has one: its root node holds that primitive on the left and
 * nothing on the right.
 */
export const EMPTY_CHILD_INDEX = GPU_LAYOUT_EMPTY_CHILD;

/** Parent index of the root node in the parent array. */
export const NO_PARENT = GPU_LAYOUT_NO_PARENT;

/**
 * Triangles are tested with the watertight test of Woop, Benthin and Wald
 * (2013), the only test the tracer has: Moller-Trumbore let 3-12% of rays aimed
 * at shared edges slip through (T0.10 part A; DECISIONS 2026-11-04). The test
 * must see the *same bits* for a vertex shared by two triangles, so a triangle
 * stores its three corners as they are.
 */
/** One triangle in tree order: three corners, the original index and flags, 48 bytes. */
export const TRIANGLE_BYTES = GPU_LAYOUT_TRIANGLE_BYTES;
export const TRIANGLE_WORDS = TRIANGLE_BYTES / 4;

/**
 * Word positions inside a triangle. Matches the WGSL `BvhTriangle`.
 */
export const TRIANGLE_WORD = {
    vertex0: 0,
    originalIndex: 3,
    vertex1: 4,
    flags: 7,
    vertex2: 8,
    padding: 11,
} as const;

/**
 * Triangle `flags` bit 0: consult the triangle filter. The walk asks the
 * kernel's `bvhTrianglePasses` (the CPU walk its `triangleFilter`) whether the
 * ray passes before taking the hit (triangle-filter.ts).
 */
export const TRIANGLE_FLAG_FILTER = 1;

/**
 * Triangle `flags` bit 1: flip the hit's `frontFace`. Set by
 * `assembleTraceScene` on a merged mesh's triangles (`mergeStaticInstances`,
 * static-merge.ts) that come from a mirrored source instance: they are stored
 * in world space, where a mirror reverses the authored winding. Builders never
 * write it, and an input's `triangleFlags` may not set it (only bit 0;
 * bits 1-31 throw). Bits 2-15 are reserved, 0. Bits 16-31 hold the
 * triangle's material ({@link MATERIAL_SHIFT}: material + 1, 0 for none):
 * the mesh input's `triangleMaterials`, which every builder writes, and in a
 * merged mesh its source's (the source triangle's, else the source
 * instance's), which `assembleTraceScene` writes.
 */
export const TRIANGLE_FLAG_FLIP_FRONT_FACE = 2;

/**
 * The triangle record's `padding` word (word 11): the *source instance index*
 * of a triangle in a merged mesh (the record's instance has
 * {@link INSTANCE_FLAG}`.merged`), which the walk returns as `BvhHit.instance`
 * and `BvhCandidate.instance`. 0 in every other mesh; builders always write 0,
 * and `assembleTraceScene` writes the index after the build.
 */
export const TRIANGLE_WORD_SOURCE_INSTANCE = TRIANGLE_WORD.padding;

/**
 * How a scene stores its triangles (docs/internals/design/indexed-leaves.md):
 * `'records'` (the default, the 48-byte records above) or `'indexed'`
 * (16-byte `vec4u(i0, i1, i2, id + flags)` read through the scene's
 * positions buffer). The walk is compiled for one
 * (`traceModuleWgsl({ triangleLeaves })`, `createCpuTracer` reads it from the scene).
 */
export type TriangleLeaves = GpuLayoutTriangleLeaves;

/** One indexed triangle: three vertex indices, then the original index and flags, 16 bytes. */
export const INDEXED_TRIANGLE_BYTES = GPU_LAYOUT_INDEXED_TRIANGLE_BYTES;
export const INDEXED_TRIANGLE_WORDS = INDEXED_TRIANGLE_BYTES / 4;
/** Word positions inside an indexed triangle. */
export const INDEXED_TRIANGLE_WORD = GPU_LAYOUT_INDEXED_TRIANGLE_WORD;
/** Bits 0-27 of an indexed triangle's word 3: the original index. */
export const INDEXED_ID_MASK = GPU_LAYOUT_INDEXED_ID_MASK;
/** Bit 28 of an indexed triangle's word 3: {@link TRIANGLE_FLAG_FILTER}. */
export const INDEXED_FLAG_FILTER = (TRIANGLE_FLAG_FILTER << GPU_LAYOUT_INDEXED_FLAG_SHIFT) >>> 0;
/** Bit 29 of an indexed triangle's word 3: {@link TRIANGLE_FLAG_FLIP_FRONT_FACE}. */
export const INDEXED_FLAG_FLIP_FRONT_FACE = (TRIANGLE_FLAG_FLIP_FRONT_FACE << GPU_LAYOUT_INDEXED_FLAG_SHIFT) >>> 0;
export const INDEXED_FLAG_SHIFT = GPU_LAYOUT_INDEXED_FLAG_SHIFT;

/**
 * The 16-byte header an indexed scene puts before each mesh's triangles (at
 * `triangleOffset - 1`, in 16-byte units): the mesh's vertex base, and for a
 * merged mesh how far back its side table and source table start.
 */
export const MESH_HEADER_WORD = GPU_LAYOUT_MESH_HEADER_WORD;

/**
 * A merged mesh's side table entry (indexed scenes; one `u32` per triangle,
 * by tree slot, four to a `vec4u`): bits 0-15 the source's slot in the
 * group's source table, bits 16-31 its material code (material + 1, 0: none).
 * The source table holds each source's `instanceIndex`, four to a `vec4u`.
 * Records keep both in the triangle instead (word 11, flags bits 16-31).
 */
export const MERGED_SIDE_SOURCE_BITS = 0xffff;
export const MERGED_SIDE_MATERIAL_SHIFT = 16;
/** Most sources one merged group may hold: the side table's 16-bit slot. */
export const MAX_MERGED_SOURCES = 0x10000;

/**
 * How an indexed scene stores its positions buffer: `'vec4'` (the default,
 * `vec4f` per vertex, 16 bytes, one load per corner) or `'packed'` (`f32 x 3`,
 * 12 bytes, three loads per corner). A compile-time choice of the walk, like
 * the triangle format.
 */
export type PositionLayout = 'vec4' | 'packed';

/** Floats per vertex in the positions buffer. */
export function positionStrideFloats(layout: PositionLayout | undefined): number {
    return layout === 'packed' ? 3 : 4;
}

/** Units (16 bytes each) of an unmerged indexed mesh's material table: 16 bits per triangle. */
export function indexedMaterialTableUnits(triangleCount: number): number {
    return Math.ceil(triangleCount / 8);
}

/**
 * Units (16 bytes each) of an indexed scene's region before a mesh's first
 * triangle: the header, and a merged mesh's side and source tables, or an
 * unmerged mesh's material table (`materials`, when it has per-triangle materials).
 */
export function indexedLeadingUnits(triangleCount: number, sourceCount = 0, materials = false): number {
    if (sourceCount > 0) return 1 + Math.ceil(triangleCount / 4) + Math.ceil(sourceCount / 4);
    return materials ? 1 + indexedMaterialTableUnits(triangleCount) : 1;
}

/**
 * Writes an unmerged indexed mesh's header (its vertex base, and with
 * `codes` the material table's distance) and its material table (`codes`:
 * each original triangle's material code, by original index) into `words`
 * (4 per 16-byte unit), the mesh's first triangle at unit `triangleOffset`.
 * The region must be {@link indexedLeadingUnits}`(n, 0, codes !== null)` units.
 */
export function writeIndexedMeshHeader(words: Uint32Array, triangleOffset: number, vertexBase: number, codes: Uint16Array | null): void {
    const header = (triangleOffset - 1) * INDEXED_TRIANGLE_WORDS;
    words.fill(0, header, header + INDEXED_TRIANGLE_WORDS);
    words[header + MESH_HEADER_WORD.vertexBase] = vertexBase;
    if (!codes) return;
    const units = indexedMaterialTableUnits(codes.length);
    const start = (triangleOffset - 1 - units) * INDEXED_TRIANGLE_WORDS;
    if (start < 0) throw new RangeError(`the mesh at ${triangleOffset} has no room for its material table`);
    words.fill(0, start, header);
    for (let triangle = 0; triangle < codes.length; triangle++) words[start + (triangle >> 1)] |= codes[triangle] << ((triangle & 1) * 16);
    words[header + MESH_HEADER_WORD.materialTableDistance] = 1 + units;
}

/** An unmerged indexed mesh's material code (material + 1, 0: none) for `original`, from its material table; 0 without one. */
export function indexedMaterialCode(words: Uint32Array, triangleOffset: number, original: number): number {
    const distance = words[(triangleOffset - 1) * INDEXED_TRIANGLE_WORDS + MESH_HEADER_WORD.materialTableDistance];
    if (distance === 0) return 0;
    const word = words[(triangleOffset - distance) * INDEXED_TRIANGLE_WORDS + (original >> 1)];
    return (word >>> ((original & 1) * 16)) & 0xffff;
}

/** @deprecated The old name of {@link TRIANGLE_FLAG_FILTER}, from when the bit only meant an alpha test. */
export const TRIANGLE_FLAG_ALPHA_TEST = TRIANGLE_FLAG_FILTER;

/**
 * An instance record: the world-to-object transform as three rows of
 * `vec4f` (the 3x4 matrix), then `(rootNodeOffset, triangleOffset,
 * instanceIndex, flags)` as four u32.
 */
export const INSTANCE_RECORD_BYTES = 64;
export const INSTANCE_RECORD_WORDS = INSTANCE_RECORD_BYTES / 4;

/** Word positions inside an instance record. */
export const INSTANCE_WORD = {
    worldToObjectRow0: 0,
    worldToObjectRow1: 4,
    worldToObjectRow2: 8,
    rootNodeOffset: 12,
    triangleOffset: 13,
    instanceIndex: 14,
    flags: 15,
} as const;

/**
 * Instance `flags` bits 0-3. Bits 4-7 are reserved; bits 8-15 hold the
 * instance mask ({@link INSTANCE_MASK_SHIFT}); bits 16-31 the instance's
 * material ({@link MATERIAL_SHIFT}: material + 1, 0 for none). A merged
 * record ({@link INSTANCE_FLAG}`.merged`) leaves them 0, and its triangles
 * carry their source's material in triangle flags bits 16-31 instead.
 */
export const INSTANCE_FLAG = {
    /** Rays that hit this instance count (an instance can be hidden from GI only). */
    contributesToGi: 1,
    /** The instance is drawn at all. */
    visible: 2,
    /**
     * The object-to-world transform has a negative determinant (a mirror).
     * Informational: the walk tests triangles in object space, where the
     * winding is the authored one, so `frontFace` needs no flip (front faces
     * follow the authored side, DECISIONS 2026-11-04).
     */
    mirrored: 4,
    /**
     * A merged mesh of static instances (`mergeStaticInstances`,
     * static-merge.ts): its triangles are in world space under an identity
     * transform, and each names its source instance in the padding word
     * ({@link TRIANGLE_WORD_SOURCE_INSTANCE}). The walk returns that index as
     * the hit's instance instead of the record's `instanceIndex`.
     */
    merged: 8,
} as const;

/**
 * The instance mask: 8 bits (as in DXR) at bits 8-15 of the instance `flags`
 * word. A masked trace skips an instance whose `(mask & rayMask) == 0` at the
 * top level, before entering its tree. The builders write
 * {@link INSTANCE_MASK_ALL} unless the instance gives a mask. A trace module
 * built with `rayMasks: false` ignores it.
 */
export const INSTANCE_MASK_SHIFT = 8;
/** The mask's bits in the `flags` word. */
export const INSTANCE_MASK_BITS = 0xff << INSTANCE_MASK_SHIFT;
/** The default instance mask: visible to every ray. */
export const INSTANCE_MASK_ALL = 0xff;
/** The default ray mask: every instance mask bit (all 32 bits set, so it is also right if the mask grows). */
export const RAY_MASK_ALL = 0xffffffff;

/** An instance's mask, from its `flags` word. */
export function instanceMaskFromFlags(flags: number): number {
    return (flags & INSTANCE_MASK_BITS) >>> INSTANCE_MASK_SHIFT;
}

/** Checks an instance mask from user input: an integer in 0-255. Undefined gives {@link INSTANCE_MASK_ALL}. */
export function checkInstanceMask(mask: number | undefined, instanceIndex: number): number {
    if (mask === undefined) return INSTANCE_MASK_ALL;
    if (!Number.isInteger(mask) || mask < 0 || mask > INSTANCE_MASK_ALL) {
        throw new RangeError(`instance ${instanceIndex}: mask must be an integer in 0-255, got ${mask}`);
    }
    return mask;
}

/**
 * The material: bits 16-31 of the instance `flags` word, and of the triangle
 * `flags` word (a mesh input's `triangleMaterials`; in a merged mesh, its
 * source's), stored as `material + 1` so that 0 means none. A hit's material
 * ({@link hitMaterialCode}) is the triangle's when it has one, else the
 * instance's; the walk reports it as `BvhHit.material`, {@link NO_MATERIAL}
 * for none. Indexed scenes keep the triangle's code in a per-mesh table
 * instead (`MESH_HEADER_WORD.materialTableDistance`, or a merged side table).
 * Written only by this package; a mesh input's `triangleFlags` may still set
 * bit 0 only.
 */
export const MATERIAL_SHIFT = 16;
/** The material's bits in an instance or triangle `flags` word. */
export const MATERIAL_BITS = 0xffff0000;
/** Largest material index (16 bits, minus the "none" code 0). */
export const MAX_MATERIAL = 0xfffe;
/** `BvhHit.material` (WGSL `BVH_NO_MATERIAL`) and `CpuHit.material` of a hit on an instance given no material. */
export const NO_MATERIAL = 0xffffffff;

/**
 * Checks a material index from user input and returns its stored code
 * (`material + 1`, the value of bits 16-31). Undefined or
 * {@link NO_MATERIAL} give 0 (none); anything but an integer in
 * 0-{@link MAX_MATERIAL} is a `RangeError`.
 */
export function materialCode(material: number | undefined, instanceIndex: number): number {
    if (material === undefined || material === NO_MATERIAL) return 0;
    if (!Number.isInteger(material) || material < 0 || material > MAX_MATERIAL) {
        throw new RangeError(`instance ${instanceIndex}: material must be an integer in 0-${MAX_MATERIAL}, got ${material}`);
    }
    return material + 1;
}

/** A hit's material code: the triangle's (`triangleCode`, material + 1) when it has one, else the instance's from its `flags` word. */
export function hitMaterialCode(triangleCode: number, instanceFlags: number): number {
    return triangleCode !== 0 ? triangleCode : instanceFlags >>> MATERIAL_SHIFT;
}

/** The material index of a code (material + 1, 0: none), or {@link NO_MATERIAL}. */
export function materialFromCode(code: number): number {
    return code === 0 ? NO_MATERIAL : code - 1;
}

/** The material index in an instance's or a triangle's `flags` word, or {@link NO_MATERIAL}. */
export function materialFromFlags(flags: number): number {
    const code = flags >>> MATERIAL_SHIFT;
    return code === 0 ? NO_MATERIAL : code - 1;
}

/**
 * Which query a triangle filter is asked from: `query` in `BvhCandidate`
 * (WGSL `BVH_QUERY_CLOSEST`, `BVH_QUERY_ANY`) and in `TriangleCandidate`.
 */
export const BVH_QUERY_CLOSEST = 0;
export const BVH_QUERY_ANY = 1;
export type BvhQuery = typeof BVH_QUERY_CLOSEST | typeof BVH_QUERY_ANY;

/**
 * The top-level buffer is `[header][instance records][top-level nodes]`.
 * The header is 64 bytes so records and nodes stay 64-byte aligned.
 */
export const TOP_LEVEL_HEADER_BYTES = 64;
export const TOP_LEVEL_HEADER_WORDS = TOP_LEVEL_HEADER_BYTES / 4;

/** Word positions inside the top-level header. */
export const TOP_HEADER_WORD = {
    instanceCount: 0,
    nodeCount: 1,
    /** Informational: how many levels the top-level tree has. */
    maxDepth: 2,
} as const;

/**
 * The traversal stack is shared by both levels (see the WGSL). Stack entries
 * are 32-bit "child references": the index in the low bits and the leaf
 * primitive count in the top bits (0 = inner node).
 */
export const TRACE_STACK_SIZE = 64;
export const REF_COUNT_SHIFT = 28;
/** Node and triangle indices must fit under the reference index mask: the 28 bits below the count. */
export const MAX_REFERENCE_INDEX = GPU_LAYOUT_MAX_TRIANGLES;
export const REF_INDEX_MASK = MAX_REFERENCE_INDEX;
/** Pushed when entering an instance; popping it switches back to the world-space ray. */
export const REF_INSTANCE_MARKER = 0xffffffff;
/** Largest leaf the reference encoding can hold without colliding with {@link REF_INSTANCE_MARKER}. */
export const MAX_LEAF_PRIMITIVES = GPU_LAYOUT_MAX_LEAF_PRIMITIVES;

// The reference encoding needs the index mask to be exactly the bits below the count, and a
// leaf count below 15 (count 15 with an all-ones index would be the instance marker).
if (REF_INDEX_MASK !== 2 ** REF_COUNT_SHIFT - 1) throw new Error('layout: the reference index mask must be the bits below REF_COUNT_SHIFT');
if (MAX_LEAF_PRIMITIVES >= 2 ** (32 - REF_COUNT_SHIFT) - 1) throw new Error('layout: leaves this large collide with the instance marker');

export function makeReference(index: number, count: number): number {
    return ((count << REF_COUNT_SHIFT) | index) >>> 0;
}

export function referenceIndex(reference: number): number {
    return reference & REF_INDEX_MASK;
}

export function referenceCount(reference: number): number {
    return reference >>> REF_COUNT_SHIFT;
}

/**
 * Read-only typed views over a node array, so builders, tests and the CPU
 * traversal share one way to touch the words.
 */
export interface NodeView {
    readonly floats: Float32Array;
    readonly words: Uint32Array;
}

export function nodeView(buffer: ArrayBuffer, byteOffset = 0, byteLength = buffer.byteLength - byteOffset): NodeView {
    return {
        floats: new Float32Array(buffer, byteOffset, byteLength / 4),
        words: new Uint32Array(buffer, byteOffset, byteLength / 4),
    };
}

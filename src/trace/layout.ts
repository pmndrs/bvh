/**
 * Memory layout of the ray-tracing acceleration structures, in one place.
 *
 * The CPU builders write these layouts and the WGSL reads them, so every
 * number the two sides must agree on lives here, and the WGSL source is
 * generated from these constants rather than repeating them. The sizes and
 * limits every builder also needs (node and triangle bytes, the largest leaf,
 * the most triangles per mesh) are defined in `../gpu-layout.ts` and taken
 * from it here. (This file moved from ddgi-core in BX.2.)
 *
 * Everything is plain 32-bit words in a few buffers so it survives a worker
 * hand-off (transferable `ArrayBuffer`s) and a GPU upload without conversion.
 * See docs/internals/design/trace.md for the reasoning.
 */

import {
    GPU_LAYOUT_EMPTY_CHILD,
    GPU_LAYOUT_INDEXED_FLAG_SHIFT,
    GPU_LAYOUT_INDEXED_ID_MASK,
    GPU_LAYOUT_INDEXED_TRIANGLE_BYTES,
    GPU_LAYOUT_INDEXED_TRIANGLE_WORD,
    GPU_LAYOUT_MAX_LEAF_PRIMITIVES,
    GPU_LAYOUT_MAX_TRIANGLES,
    GPU_LAYOUT_MESH_HEADER_WORD,
    GPU_LAYOUT_NO_PARENT,
    GPU_LAYOUT_NODE_BYTES,
    GPU_LAYOUT_NODE_WORD,
    GPU_LAYOUT_TRIANGLE_BYTES,
    GPU_LAYOUT_QUANTIZED_NODE_BYTES,
    type GpuLayoutNodeFormat,
    type GpuLayoutTriangleLeaves,
} from '../gpu-layout.ts';

/**
 * The version of `@pmndrs/bvh`'s GPU layout (`GPU_LAYOUT_VERSION`) that this
 * file's WGSL structs and the walk in wgsl.ts were written against. A test
 * fails when the package's version moves, so a layout change can't reach the
 * tracer without someone reviewing the walk.
 */
export const TRACER_GPU_LAYOUT_VERSION = 2;

/**
 * How a scene stores its mesh nodes (docs/internals/design/compressed-nodes.md):
 * `'f32'` (the default, the 64-byte nodes below) or `'quantized'` (32-byte
 * nodes on a per-mesh frame, `GpuLayoutNodeFormat`). The walk is compiled
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
 * `index`. A child with `count == 0` is an inner node at `index`. Defined in
 * `@pmndrs/bvh` (its gpu-layout.ts), whose GPU refit writes the same words.
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
 * bits 1-31 throw). Bits 2-15 are reserved, 0. Bits 16-31 hold the source
 * instance's material in a merged mesh ({@link MATERIAL_SHIFT}: material + 1,
 * 0 for none), written by `assembleTraceScene`, because a merged record has
 * no single material; 0 everywhere else.
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

/** Units (16 bytes each) of an indexed scene's region before a mesh's first triangle: the header, and a merged mesh's side and source tables. */
export function indexedLeadingUnits(triangleCount: number, sourceCount = 0): number {
    return sourceCount > 0 ? 1 + Math.ceil(triangleCount / 4) + Math.ceil(sourceCount / 4) : 1;
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
 * The per-instance material: bits 16-31 of the instance `flags` word, and of
 * the triangle `flags` word in a merged mesh (where each triangle carries its
 * source instance's), stored as `material + 1` so that 0 means none. The walk
 * reports it as `BvhHit.material`, {@link NO_MATERIAL} for none. Written only
 * by this package (`buildTopLevel`, `assembleTraceScene`); a mesh input's
 * `triangleFlags` may still set bit 0 only.
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

/** The material index in an instance's or a merged triangle's `flags` word, or {@link NO_MATERIAL}. */
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

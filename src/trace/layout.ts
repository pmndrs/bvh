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
 * See docs/design/trace.md for the reasoning.
 */

import { GPU_LAYOUT_EMPTY_CHILD, GPU_LAYOUT_MAX_LEAF_PRIMITIVES, GPU_LAYOUT_MAX_TRIANGLES, GPU_LAYOUT_NO_PARENT, GPU_LAYOUT_NODE_BYTES, GPU_LAYOUT_NODE_WORD, GPU_LAYOUT_TRIANGLE_BYTES } from '../gpu-layout.ts';

/**
 * The version of `@pmndrs/bvh`'s GPU layout (`GPU_LAYOUT_VERSION`) that this
 * file's WGSL structs and the walk in wgsl.ts were written against. A test
 * fails when the package's version moves, so a layout change can't reach the
 * tracer without someone reviewing the walk.
 */
export const TRACER_GPU_LAYOUT_VERSION = 2;

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
 * write it. Bits 2-15 are reserved, 0. Bits 16-31 are reserved for a
 * per-instance material (planned: material + 1, 0 = none), which a merged
 * mesh will carry per triangle because its record has no single material;
 * not written yet.
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
 * instance mask ({@link INSTANCE_MASK_SHIFT}); bits 16-31 are reserved for
 * a per-instance material index (planned: material + 1, 0 = none). A merged
 * record ({@link INSTANCE_FLAG}`.merged`) will leave them 0 and its triangles
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

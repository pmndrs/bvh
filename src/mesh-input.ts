/**
 * What every builder takes and what the GPU builders give back. These types
 * are shared by all builders (tinybvh, the GPU LBVH, the GPU H-PLOC), so they
 * live here rather than in one builder's file: a builder can be swapped for
 * another without changing the code that feeds it or reads its trees
 * (docs/internals/design/bvh-framework.md, "Builder contract").
 */

/** Triangle geometry in object space, as CPU arrays. Every builder takes it. */
export interface MeshInput {
    /** `x, y, z` per vertex. */
    readonly positions: Float32Array;
    /** Three vertex indices per triangle; omit for a plain triangle list. */
    readonly indices?: Uint16Array | Uint32Array | null;
    /**
     * Per-triangle flags, indexed by the original triangle number; they default
     * to 0. Only bit 0 (`TRIANGLE_FLAG_FILTER`: ask the triangle filter) may be
     * set. Bits 1-31 are reserved: bit 1 flips `frontFace` on merged static
     * meshes' triangles, bits 2-15 are 0, and bits 16-31 carry a merged
     * triangle's source material, which the scene assembly writes from the
     * instance's `material` (or from {@link triangleMaterials}). Every builder
     * throws, naming the triangle, if a reserved bit is set.
     */
    readonly triangleFlags?: Uint32Array | null;
    /**
     * Per-triangle material index, by original triangle number: 0-65534, or
     * {@link TRIANGLE_MATERIAL_NONE} (0xffff; `NO_MATERIAL`, 0xffffffff, too)
     * for none. A hit on a triangle with a material reports it as
     * `BvhHit.material`; one without falls back to its instance's material.
     * Records keep it in flags bits 16-31 (material + 1), every builder writes
     * it there; indexed scenes keep a per-mesh table before the mesh's header
     * (docs/internals/design/indexed-leaves.md). Omit it (the default) for
     * the instance's material only. Anything else throws, naming the triangle.
     */
    readonly triangleMaterials?: Uint16Array | Uint32Array | ArrayLike<number> | null;
}

/** A {@link MeshInput.triangleMaterials} entry meaning "no material here": the hit takes its instance's. */
export const TRIANGLE_MATERIAL_NONE = 0xffff;
/** Largest per-triangle material (as `MAX_MATERIAL` in gpu-layout.ts). */
const MAX_TRIANGLE_MATERIAL = 0xfffe;
/** Bits 16-31 of a triangle's flags: its material + 1 (`MATERIAL_SHIFT` in gpu-layout.ts). */
const TRIANGLE_MATERIAL_SHIFT = 16;

/**
 * The stored codes of `materials` (material + 1, 0 for none), one per
 * triangle, or null when there are none. Throws a `RangeError` naming
 * `where` and the triangle for a short array or a value outside 0-65534
 * that is not {@link TRIANGLE_MATERIAL_NONE} or 0xffffffff.
 */
export function triangleMaterialCodes(
    materials: MeshInput['triangleMaterials'],
    triangleCount: number,
    where: string,
): Uint16Array | null {
    if (!materials) return null;
    if (materials.length < triangleCount) throw new RangeError(`${where}: triangleMaterials must have one entry per triangle (${materials.length} for ${triangleCount})`);
    const codes = new Uint16Array(triangleCount);
    for (let triangle = 0; triangle < triangleCount; triangle++) {
        const material = materials[triangle];
        if (material === TRIANGLE_MATERIAL_NONE || material === 0xffffffff) continue;
        if (!Number.isInteger(material) || material < 0 || material > MAX_TRIANGLE_MATERIAL) {
            throw new RangeError(`${where}: triangleMaterials[${triangle}] is ${material}; a material is an integer in 0-${MAX_TRIANGLE_MATERIAL}, or 0xffff for none`);
        }
        codes[triangle] = material + 1;
    }
    return codes;
}

/**
 * The flags word every builder writes per original triangle: the input's
 * `triangleFlags` (checked: bit 0 only) with the triangle's material code
 * from `triangleMaterials` in bits 16-31. The input's own array when it has
 * no materials, null when it has neither.
 */
export function packedTriangleFlags(geometry: MeshInput, triangleCount: number, where: string): Uint32Array | null {
    const { triangleFlags } = geometry;
    if (triangleFlags && triangleFlags.length < triangleCount) throw new RangeError(`${where}: triangleFlags must have one entry per triangle`);
    checkTriangleFlags(triangleFlags, triangleCount, where);
    const codes = triangleMaterialCodes(geometry.triangleMaterials, triangleCount, where);
    if (!codes) return triangleFlags ?? null;
    const flags = new Uint32Array(triangleCount);
    for (let triangle = 0; triangle < triangleCount; triangle++) {
        flags[triangle] = ((triangleFlags ? triangleFlags[triangle] : 0) | (codes[triangle] << TRIANGLE_MATERIAL_SHIFT)) >>> 0;
    }
    return flags;
}

/** The triangle flag bits a mesh's input may set: bit 0, `TRIANGLE_FLAG_FILTER` (gpu-layout.ts). */
export const TRIANGLE_FLAGS_INPUT_MASK = 1;

/**
 * Throws a `RangeError` naming `where` and the triangle if any of the first
 * `triangleCount` entries sets a reserved bit (anything but
 * {@link TRIANGLE_FLAGS_INPUT_MASK}). Masking them silently would hide a
 * caller's bug and break merged meshes' flip bit or material.
 */
export function checkTriangleFlags(triangleFlags: Uint32Array | null | undefined, triangleCount: number, where: string): void {
    if (!triangleFlags) return;
    const count = Math.min(triangleCount, triangleFlags.length);
    for (let triangle = 0; triangle < count; triangle++) {
        const flags = triangleFlags[triangle];
        if ((flags & ~TRIANGLE_FLAGS_INPUT_MASK) !== 0) {
            throw new RangeError(
                `${where}: triangleFlags[${triangle}] is 0x${flags.toString(16)}; only bit 0 (TRIANGLE_FLAG_FILTER) may be set, bits 1-31 are reserved`,
            );
        }
    }
}

/**
 * Geometry already on the GPU, for example a deformed vertex buffer an
 * adapter wrote. A GPU build copies the ranges it needs into its own buffers
 * when it is recorded, so the buffers need `COPY_SRC` usage.
 */
export interface GpuMeshBuffers {
    /** f32 vertex positions. */
    readonly positionBuffer: GPUBuffer;
    readonly vertexCount: number;
    /** Floats from one vertex to the next. Default 3 (tightly packed). */
    readonly positionStrideFloats?: number;
    /** Float index of vertex 0's x. Default 0. */
    readonly positionOffsetFloats?: number;
    /** Three indices per triangle; omit for a plain triangle list. */
    readonly indexBuffer?: GPUBuffer | null;
    readonly indexFormat?: 'uint16' | 'uint32';
    /** Index of the mesh's first index in `indexBuffer` (not bytes). Default 0. */
    readonly indexOffset?: number;
    readonly triangleCount: number;
    /** One `u32` per triangle, by original triangle number, from offset 0. Default 0. Only bit 0 may be set, as in `MeshInput.triangleFlags`; a GPU buffer is not checked. */
    readonly triangleFlagsBuffer?: GPUBuffer | null;
}

/** What the GPU builders take: CPU arrays (uploaded for the build) or buffers already on the GPU. */
export type GpuMeshGeometry = MeshInput | GpuMeshBuffers;

/** Options every GPU builder takes. H-PLOC adds its own (`HplocOptions`). */
export interface GpuBuildOptions {
    /** Largest leaf the SAH collapse may make. 1 to 8; default 8, the layout's largest leaf. */
    readonly maxLeafPrimitives?: number;
    /** SAH cost of visiting a node, relative to {@link intersectionCost}. Default 1. */
    readonly traversalCost?: number;
    /** SAH cost of testing one triangle. Default 1. */
    readonly intersectionCost?: number;
    /**
     * The triangle format the tree is written in: `'records'` (48 bytes, the
     * default) or `'indexed'` (16 bytes, mesh-relative vertex indices;
     * docs/internals/design/indexed-leaves.md). The nodes are the same either way.
     */
    readonly triangleLeaves?: 'records' | 'indexed';
    /**
     * The node format the tree is written in: `'f32'` (64 bytes, the default
     * for a mesh tree; a scene quantizes it as it places it, `DEFAULT_NODE_FORMAT`)
     * or `'quantized'` (32-byte nodes on a per-mesh grid, the frame slot
     * first; docs/internals/design/compressed-nodes.md).
     * A coordinate of magnitude 2^100 or more, or a non-finite one, makes a quantized box empty on that axis (hits there are lost).
     */
    readonly nodeFormat?: 'f32' | 'quantized';
    /**
     * Per mesh of a batch: triangles (in the output format's units, 48 or 16
     * bytes) left free before the mesh's triangles in the batch's triangle
     * buffer, so a scene that adopts the buffer can write what goes there (an
     * indexed scene's mesh header and side tables: `SceneTrees` with
     * `gpuTreeOutputs: 'adopt'`). Each range's `triangleOffset` is after its
     * gap. Default: none (every mesh's triangles right after the last's).
     */
    readonly triangleLeading?: readonly number[];
}

/** One mesh's tree inside a batch's shared buffers ({@link GpuTreeBuffers}). */
export interface GpuMeshTreeRange {
    /**
     * The mesh's node 0 (its root) is this node of the shared node buffer
     * (and of the parent buffer, unless {@link parentOffset} says otherwise).
     * Child indices are the mesh's own. With quantized nodes, the root's
     * 32-byte slot; the mesh's frame slot is the one before.
     */
    readonly nodeOffset: number;
    /** Where the mesh's parents start in the shared parent buffer, when not at {@link nodeOffset} (quantized trees). */
    readonly parentOffset?: number;
    readonly nodeCount: number;
    /** The mesh's first triangle in the shared triangle buffer. Leaf indices are the mesh's own. */
    readonly triangleOffset: number;
    readonly triangleCount: number;
    readonly leafCount: number;
    /** Levels of nodes on the longest path. */
    readonly maxDepth: number;
    /** Object-space box around the mesh: `[minX, minY, minZ, maxX, maxY, maxZ]`. */
    readonly bounds: Float32Array;
    /** Why this tree must not be used, or null. */
    readonly problem: string | null;
}

/**
 * The trees of one GPU batch, in the GPU layout: every mesh's nodes,
 * triangles and parents in three shared buffers, each mesh at its offsets.
 * They belong to the caller: `destroy()` frees them.
 */
export interface GpuTreeBuffers {
    /** 64-byte nodes; or with `nodeFormat: 'quantized'`, each mesh's frame slot and 32-byte nodes. */
    readonly nodes: GPUBuffer;
    /** How `nodes` stores them; omitted means `'f32'`. */
    readonly nodeFormat?: 'f32' | 'quantized';
    /** 48-byte triangles (16 with `triangleLeaves: 'indexed'`), each mesh's in tree order. */
    readonly triangles: GPUBuffer;
    /** How `triangles` stores them; omitted means `'records'`. */
    readonly triangleLeaves?: 'records' | 'indexed';
    /** One `u32` per node: its parent within its mesh, or 0xffffffff for a root; a placeholder with {@link withoutParents}. */
    readonly parents: GPUBuffer;
    /**
     * The batch was built without its parents output (`GpuHplocBatchOptions.parents: false`):
     * `parents` holds nothing, and a readback finds each tree's parents from its nodes.
     */
    readonly withoutParents?: true;
    readonly meshes: readonly GpuMeshTreeRange[];
    destroy(): void;
}

export function isGpuMeshBuffers(geometry: GpuMeshGeometry): geometry is GpuMeshBuffers {
    return 'positionBuffer' in geometry;
}

export function meshTriangleCount(geometry: GpuMeshGeometry): number {
    if (isGpuMeshBuffers(geometry)) return geometry.triangleCount;
    const entries = geometry.indices ? geometry.indices.length : geometry.positions.length / 3;
    return entries / 3;
}

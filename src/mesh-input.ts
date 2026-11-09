/**
 * What every builder takes and what the GPU builders give back. These types
 * are shared by all builders (tinybvh, the GPU LBVH, the GPU H-PLOC), so they
 * live here rather than in one builder's file: a builder can be swapped for
 * another without changing the code that feeds it or reads its trees
 * (docs/design/bvh-framework.md, "Builder contract").
 */

/** Triangle geometry in object space, as CPU arrays. Every builder takes it. */
export interface MeshInput {
    /** `x, y, z` per vertex. */
    readonly positions: Float32Array;
    /** Three vertex indices per triangle; omit for a plain triangle list. */
    readonly indices?: Uint16Array | Uint32Array | null;
    /**
     * Per-triangle bit flags, indexed by the original triangle number. Only the
     * GPU layout stores them; they default to 0.
     */
    readonly triangleFlags?: Uint32Array | null;
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
    /** One `u32` per triangle, by original triangle number, from offset 0. Default 0. */
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
}

/** One mesh's tree inside a batch's shared buffers ({@link GpuTreeBuffers}). */
export interface GpuMeshTreeRange {
    /** The mesh's node 0 (its root) is this node of the shared node and parent buffers. Child indices are the mesh's own. */
    readonly nodeOffset: number;
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
    /** 64-byte nodes. */
    readonly nodes: GPUBuffer;
    /** 48-byte triangles, each mesh's in tree order. */
    readonly triangles: GPUBuffer;
    /** One `u32` per node: its parent within its mesh, or 0xffffffff for a root. */
    readonly parents: GPUBuffer;
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

/**
 * The one interface every library in the comparison suite implements (BX.10,
 * `compare/RESULTS.md`). Plain data in, plain data out, so the same adapters
 * run in Node (`pnpm compare:node`) and on the compare pages.
 *
 * Rules every adapter keeps:
 * - It answers on the geometry it is given, in that geometry's space: world
 *   positions, triangle `i` is `indices[3i..3i+2]`. Hits name that original
 *   triangle, whatever the library reorders inside.
 * - Every query is double-sided, from `t = 0` to infinity.
 * - A capability the library doesn't have is left out (the method is
 *   undefined) and its `features` row says so. Nothing is emulated with our
 *   code or with brute force behind the library's back.
 * - `memory()` counts what the library holds beyond the app's own geometry
 *   (positions and indices the app keeps anyway), as docs/internals/design/memory.md
 *   does, and says how it was counted.
 */

export type Vec3 = [number, number, number];

/** One mesh in world space: what every library is given. */
export interface CompareGeometry {
    readonly name: string;
    readonly positions: Float32Array;
    readonly indices: Uint32Array;
    readonly triangleCount: number;
    /** `[minX, minY, minZ, maxX, maxY, maxZ]`. */
    readonly bounds: readonly number[];
}

/** A ray with a unit-length direction, so `t` is a distance. */
export interface CompareRay {
    readonly origin: Vec3;
    readonly direction: Vec3;
}

export interface RayHit {
    /** The original triangle (`indices[3 * triangle]`...). */
    readonly triangle: number;
    /** Distance along the ray. */
    readonly t: number;
}

export interface ClosestPointHit {
    readonly triangle: number;
    readonly distance: number;
    readonly point: Vec3;
}

export interface MemoryReport {
    /** Bytes beyond the app's geometry, or null when the library gives no way to count them here. */
    readonly bytes: number | null;
    /** How `bytes` was counted. */
    readonly method: string;
}

/** Tree shape, for `compare/tree-quality`. Null fields: the library's structure has no such thing or can't be read. */
export interface TreeQuality {
    /** Surface-area score with three-mesh-bvh's costs (traversal 1, triangle 1.25) over the root box's area. */
    readonly sah: number | null;
    readonly depth: number | null;
    readonly nodes: number | null;
    readonly leaves: number | null;
    readonly bytes: number | null;
    readonly note?: string;
}

export type Capability =
    | 'raycastFirst'
    | 'raycastAll'
    | 'gpuTrace'
    | 'intersectsBox'
    | 'intersectsSphere'
    | 'closestPoint'
    | 'refit'
    | 'workerBuild'
    | 'gpuBuild'
    | 'treeStats'
    | 'serialize';

export const CAPABILITIES: readonly Capability[] = ['raycastFirst', 'raycastAll', 'gpuTrace', 'intersectsBox', 'intersectsSphere', 'closestPoint', 'refit', 'workerBuild', 'gpuBuild', 'treeStats', 'serialize'];

export interface Feature {
    readonly status: 'yes' | 'partial' | 'no';
    readonly note?: string;
}

export type FeatureTable = Readonly<Record<Capability, Feature>>;

/** A mesh as one library holds it, with the queries it has. */
export interface BuiltMesh {
    /**
     * Work the comparison needs after the timed build and the library doesn't
     * (mapping its reordered triangles back to the original ids). Not timed.
     */
    prepareQueries?(): void;
    raycastFirst?(ray: CompareRay): RayHit | null;
    /** Every hit, sorted by `t`. */
    raycastAll?(ray: CompareRay): RayHit[];
    /** Does any triangle meet the box (`[minX, minY, minZ, maxX, maxY, maxZ]`)? */
    intersectsBox?(box: readonly number[]): boolean;
    intersectsSphere?(center: Vec3, radius: number): boolean;
    closestPoint?(point: Vec3): ClosestPointHit | null;
    /** The positions moved (same length, same triangles): bring the structure up to date. */
    refit?(positions: Float32Array): void;
    memory(): MemoryReport;
    quality?(): TreeQuality;
    dispose(): void;
}

/** One library (or one configuration of it, such as three-mesh-bvh with SAH). */
export interface MeshAdapter {
    /** Short id for CSV rows: `ours`, `tmbvh-sah`, ... */
    readonly id: string;
    readonly name: string;
    /** Package and exact version. */
    readonly library: string;
    readonly license: string;
    readonly features: FeatureTable;
    /** What `build` does, in a sentence. */
    readonly buildMethod: string;
    /** Loads WASM and the like, once, outside any timing. */
    prepare?(): Promise<void>;
    build(geometry: CompareGeometry): BuiltMesh;
    /**
     * Bytes of the app's geometry the adapter copies because the library
     * changes it (three-mesh-bvh reorders the index): in an app that copy is
     * the app's own geometry, so heap measurements take it off.
     */
    geometryCopyBytes?(geometry: CompareGeometry): number;
    /**
     * Fails fast on scenes this library can't take (memory, a 32-bit limit),
     * with the reason, instead of running out of memory halfway.
     */
    unsupported?(geometry: CompareGeometry): string | null;
}

/** Feature table helper: everything `no` unless listed. */
export function features(rows: Partial<Record<Capability, Feature | Feature['status']>>): FeatureTable {
    const table = {} as Record<Capability, Feature>;
    for (const capability of CAPABILITIES) {
        const row = rows[capability];
        table[capability] = row === undefined ? { status: 'no' } : typeof row === 'string' ? { status: row } : row;
    }
    return table;
}

// Scene level: many objects, each a box in world space (compare/scene).

/** One object of a scene: its world box (`[minX, minY, minZ, maxX, maxY, maxZ]`) and its transform. */
export interface SceneObject {
    readonly id: number;
    /** Column-major 4x4 object-to-world matrix (three's `Matrix4.elements`). */
    readonly matrix: Float32Array;
}

/** A scene of objects sharing one mesh, for the scene-level trees. */
export interface CompareScene {
    readonly name: string;
    /** The shared mesh, in object space. */
    readonly mesh: CompareGeometry;
    readonly objects: readonly SceneObject[];
}

export interface BuiltScene {
    /** Ids of the objects whose world box meets the frustum (`[a, b, c, d]` per plane, `ax + by + cz + d >= 0` inside). */
    intersectsFrustum?(planes: Float32Array, viewProjection: Float32Array): number[];
    /** Ids of the objects whose world box meets `box`. */
    intersectsBox?(box: readonly number[]): number[];
    /** The closest hit over every object: object id, triangle and distance. */
    raycastFirst?(ray: CompareRay): (RayHit & { readonly object: number }) | null;
    /** New matrices for the same objects (`objects[i].matrix` was updated in place): bring the tree up to date. */
    update?(objects: readonly SceneObject[]): void;
    /** How `update` works: `refit` keeps the shape, `rebuild` builds again, `incremental` moves leaves. */
    readonly updateKind?: 'refit' | 'rebuild' | 'incremental';
    memory(): MemoryReport;
    dispose(): void;
}

export interface SceneAdapter {
    readonly id: string;
    readonly name: string;
    readonly library: string;
    readonly license: string;
    readonly buildMethod: string;
    readonly features: Readonly<Record<'build' | 'update' | 'frustum' | 'box' | 'raycast' | 'gpu', Feature>>;
    prepare?(): Promise<void>;
    build(scene: CompareScene): BuiltScene;
}

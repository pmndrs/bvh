/**
 * `SceneTrees`: a scene's mesh trees and top level on one `GPUDevice`, kept up
 * to date as meshes and instances come and go. Ported from ddgi-core's
 * `platform/trace/ray-queries.ts` (its tree half: T2.12's fast first trees,
 * T3.2's runtime meshes), made engine-free: meshes and instances come in
 * through this class's own small API instead of ddgi's scene copy.
 *
 * - **First trees from the GPU.** On a device with subgroups, every mesh an
 *   {@link SceneTrees.update} brings (the whole scene at load, a few at a time
 *   while it runs) is built with H-PLOC in one batch (gpu-first-trees.ts), so
 *   rays see it a few frames later.
 * - **tinybvh swaps in.** Every mesh is also built by tinybvh in the worker
 *   pool (`@pmndrs/bvh/worker`), biggest first. Each tree replaces the mesh's
 *   GPU tree as it arrives: only that mesh's range and its mesh-table entry
 *   change, so nothing recompiles and the top level keeps its shape. A mesh's
 *   first tree is whichever arrives first. Without subgroups (or with
 *   `gpuFirstTrees: false`) every mesh waits for tinybvh: the declared fallback.
 * - **One node buffer, one triangle buffer, one top level**, the three
 *   storage buffers `traceModuleWgsl` binds ({@link SceneTrees.buffers}).
 *   Node and triangle buffers grow by doubling (growable-storage.ts); a
 *   growth bumps {@link SceneTrees.version}, so bind groups are made again
 *   (nothing recompiles).
 * - **The top level** is `GpuTopLevel` (refit on the GPU, shapes from a
 *   worker) on a device with subgroups, and the CPU top level
 *   (`buildTopLevel`, rebuilt on this thread when something changed) without.
 * - **Merged static instances** (`mergeStaticInstances`, on by default): at
 *   the first update, the instances added so far with `static: true` are
 *   merged (static-merge.ts; nothing else is), and each
 *   group's world-space mesh is built the same way as any mesh (GPU tree first,
 *   tinybvh's from a worker later: the merged Sponza takes 1.5 s, so it never
 *   runs on the main thread). Merged instances never move: moving, hiding,
 *   re-masking or removing one throws.
 *
 * **GPU build outputs** (`gpuTreeOutputs`, the P9 peak-memory lever):
 *
 * - `'copy'` (default; ddgi's path): each GPU tree is copied from its batch's
 *   output buffers into the scene's buffers, and the outputs are freed at the
 *   next update. For one update both are alive: on Sponza (ddgi, RTX 3060)
 *   1.16 GB, 666 MB of it outputs.
 * - `'adopt'`: when the scene's buffers are empty (the load), the load batch's
 *   output buffers *become* the scene's node and triangle buffers, with no
 *   copy, and its parents buffer is freed at once. Each GPU tree's range is
 *   freed at its swap; tinybvh's tree goes into the same range when it fits,
 *   or into the batch's spare node room. tinybvh trees that arrive before the
 *   load batch wait for it (so the buffers are still empty when it comes).
 *   Peak is about the outputs alone; the node buffer keeps the batch's size
 *   (one node per triangle) after the swaps. A deforming mesh's own batch
 *   (built apart, it reads its layout back) waits while the main load batch
 *   builds, so the main one is adopted; trees that arrive with the adopted
 *   batch are placed into its buffers at the next update.
 *
 * **A cap on the GPU batches' bytes** (`gpuMaxBatchBytes`, 256 MiB by default,
 * `Infinity` for none: the P9 build-peak lever; docs/internals/design/bvh-builders.md,
 * "Build peak memory"). The
 * load's H-PLOC build is the biggest memory number (Sponza: 1,483 MB at peak,
 * scratch and outputs of one batch of 5.74M triangles, against 470 MB
 * settled). With a cap, an update's meshes are built in batches of at most
 * that many bytes, one after another on one scratch pool (every capped build
 * of the scene, in one queue), and each batch starts once the previous one's
 * outputs were let go. So the peak is about the storage plus one batch.
 *
 * - **Batches go on without updates.** When a capped batch's stats are read,
 *   its trees are copied into the scene's buffers at once, in a submit of
 *   their own, and its outputs freed (or, adopting, the first batch becomes
 *   the buffers), so the next batch starts without waiting for an update.
 *   The trees are only *stored* then: the next update places them (top
 *   level, events), as it does any tree. So `frame(); await whenBuilt();
 *   frame()` works with a cap as without one.
 * - **Room for them is reserved.** The first growth of a capped load also
 *   sets room aside for the meshes whose GPU build is still to come
 *   (triangles exactly, nodes at the stored GPU trees' nodes per triangle):
 *   later batches go into it with no growth. The room of a mesh that is
 *   removed, refused, or gets tinybvh's tree first is given back.
 * - **Growing between updates** happens only when it is safe: when the
 *   storages hold nothing yet (the load), or the last update recorded
 *   nothing into them (no copies, growth, identity pass or refit that its
 *   encoder, maybe not submitted yet, still has to run). Otherwise, and when
 *   the room runs out, the batch waits for the next update as uncapped
 *   batches do, and the batches behind it wait with it; `whenBuilt`, and the
 *   update, then take tinybvh's tree for the meshes behind it rather than
 *   wait for an update that may not come. That update copies the batch out
 *   in a submit of its own and lets its outputs go at once, so a stall costs
 *   one update, not two. A growth between updates bumps
 *   {@link SceneTrees.version} then, not in an update. (The contract this
 *   rests on: the caller submits each update's encoder before the next
 *   update.)
 *
 * A mesh bigger than the cap builds alone in its batch. While later batches
 * wait, their meshes' geometry stays with the GPU build, so it is copied into
 * the worker rather than moved (`transferGeometry`, merged groups). Undo:
 * `gpuMaxBatchBytes: Infinity`.
 *
 * Call {@link SceneTrees.update} once per frame, before anything traces, with
 * the frame's encoder; it records copies and passes, and the caller submits.
 */

import { checkTriangleFlags, packedTriangleFlags, triangleMaterialCodes } from '../mesh-input.ts';
import { refitLayoutToPose } from '../refit-cpu.ts';
import {
    BVH_NODE_BYTES,
    checkInstanceMask,
    materialCode,
    TOP_LEVEL_HEADER_BYTES,
    TOP_LEVEL_HEADER_WORDS,
    TRACE_STACK_SIZE,
    TRIANGLE_BYTES,
    INDEXED_TRIANGLE_BYTES,
    INDEXED_TRIANGLE_WORDS,
    MESH_HEADER_WORD,
    indexedLeadingUnits,
    positionStrideFloats,
    writeIndexedMeshHeader,
    QUANTIZED_NODE_BYTES,
    type NodeFormat,
    type PositionLayout,
    type TriangleLeaves,
} from '../trace/layout.ts';
import type { MeshBvh, MeshBvhBuildOptions, MeshGeometry } from '../trace/mesh-bvh.ts';
import { nodeQuantizerFor, type QuantizeJob } from '../gpu-quantize.ts';
import { quantizeNodes } from '../quantized-nodes.ts';
import { weldPositions } from '../indexed-leaves.ts';
import { meshRefitPlacement, type TraceSceneMeshPlacement } from '../trace/scene-assembly.ts';
import type { MeshRefitVertices } from '../gpu-mesh-refit.ts';
import { DEFAULT_NODE_FORMAT, type GpuLayoutMeshBvh } from '../gpu-layout.ts';
import { DeformingTrees, type DeformingTreeCounts } from './deforming-trees.ts';
import { MERGED_INSTANCE_INDEX, mergeStaticGeometry, writeMergedIndexedIdentity, writeMergedTriangleIdentity, type StaticMergeGroup, type StaticMergePlan } from '../trace/static-merge.ts';
import { buildTopLevel, IDENTITY_TRANSFORM, throwMergedTransformChange, type Mat4, type TraceInstance } from '../trace/top-level.ts';
import { GpuTopLevel, type GpuTopLevelInstance } from '../trace/top-level-gpu.ts';
import type { TraceSceneBuffers } from '../trace/trace-gpu.ts';
import { countTriangles } from '../trace/triangle-packing.ts';
import { createBvhWorkerPool, type BvhWorkerPool, type BvhWorkerPoolOptions } from '../worker/pool.ts';
import { GpuFirstTrees, type FirstTreeBuilder, type GpuBuiltTree, type GpuTreeBatchResult, type GpuTreeBuilds, type GpuTreeOutputs, type GpuTreeRequest } from './gpu-first-trees.ts';
import { GrowableStorage } from './growable-storage.ts';
import { GpuMergedIdentity } from './merged-identity-gpu.ts';

/**
 * Levels kept free on the trace stack for the top level and the instance
 * marker. A GPU tree deeper than {@link GPU_TREE_MAX_DEPTH} waits for
 * tinybvh's rather than risk a scene that can't be traced.
 */
const TOP_LEVEL_STACK_RESERVE = 16;
export const GPU_TREE_MAX_DEPTH = TRACE_STACK_SIZE - 1 - TOP_LEVEL_STACK_RESERVE;

/**
 * Room asked for when placing GPU trees makes the node buffer grow. tinybvh's
 * trees replace them soon after and can have more nodes (Sponza: 0.6% more in
 * all, more for single meshes). Without room the first replacement that didn't
 * fit its old range doubled the buffer: Sponza's nodes went from 193 MB to
 * 386 MB (ddgi T2.12). Only GPU trees ask: tinybvh's are final.
 */
const GPU_TREE_NODE_HEADROOM = 1 / 8;

/**
 * `gpuMaxBatchBytes` when it is left out: 256 MiB. On Sponza (Apple M5) it
 * takes the load's peak from 1,483 to 711 MiB (−771 MiB) and settles 11 MiB
 * lower; the first trees cost about +165 ms at 60 Hz as first measured, when
 * each batch waited for two updates, and nothing now (166 ms against 171)
 * that batches go on between updates. docs/internals/design/bvh-builders.md,
 * "Build peak memory".
 */
export const DEFAULT_GPU_MAX_BATCH_BYTES = 256 * 1024 * 1024;

const INITIAL_NODES = 1024;
const INITIAL_TRIANGLES = 2048;
const USAGE = { COPY_SRC: 0x4, COPY_DST: 0x8, STORAGE: 0x80 } as const;

/** Which builder made a tree. */
export type TreeBuilder = 'gpu' | 'tinybvh';

/** How GPU build outputs reach the scene's buffers; see the file comment. */
export type GpuTreeOutputsMode = 'copy' | 'adopt';

export interface SceneTreesOptions {
    /** Prefix for GPU object labels. */
    readonly label?: string;
    /** First trees from the GPU (H-PLOC) on a device with subgroups. Default true; false: tinybvh only. */
    readonly gpuFirstTrees?: boolean;
    /** How GPU trees reach the scene's buffers: `'copy'` (default) or `'adopt'` (see the file comment). `'adopt'` needs `nodeFormat: 'f32'`. */
    readonly gpuTreeOutputs?: GpuTreeOutputsMode;
    /**
     * Cap each GPU build batch's bytes (inputs, scratch and outputs) and build
     * the batches one after another (see the file comment). Default 256 MiB
     * ({@link DEFAULT_GPU_MAX_BATCH_BYTES}). `Infinity` turns the cap off: one
     * batch for all of an update's meshes, as big as the device allows (the
     * path before the cap).
     * The batches go on between updates (each is copied out in its own
     * submit), but their trees reach rays at the next {@link SceneTrees.update},
     * as any tree does.
     */
    readonly gpuMaxBatchBytes?: number;
    /** The worker pool for tinybvh builds, or options for one this scene makes (and disposes). */
    readonly workers?: BvhWorkerPool | BvhWorkerPoolOptions;
    /**
     * Move each mesh's geometry arrays into the worker instead of copying them
     * (the caller's arrays are empty afterwards). Default false. Merged groups'
     * geometry is the scene's own and always moves.
     */
    readonly transferGeometry?: boolean;
    /**
     * Merge static instances at the first update (see the file comment).
     * Default true. Only instances added with `static: true` before the first
     * update are merged; every other instance is untouched. `false` keeps
     * static instances as ordinary ones.
     */
    readonly mergeStaticInstances?: boolean;
    /**
     * tinybvh's trees wait instead of replacing a mesh's GPU tree, until
     * {@link SceneTrees.releaseRefinements}. A mesh with no tree yet still takes
     * tinybvh's. For tools that check a scene mid-swap. Default false.
     */
    readonly holdRefinements?: boolean;
    /**
     * Trees deeper than this are refused, with a `fallbackReason`: a GPU tree leaves the mesh to tinybvh, a tinybvh
     * tree leaves it on the tree it has (or none). Default {@link GPU_TREE_MAX_DEPTH}, which keeps every placed tree
     * within the trace stack.
     */
    readonly gpuMaxDepth?: number;
    /** `'auto'` (default): the GPU top level with subgroups, the CPU one without. `'cpu'`: always the CPU top level. */
    readonly topLevel?: 'auto' | 'cpu';
    /** Read the GPU top level's refit stats back (counters, warnings). Default false: no readbacks frame after frame. */
    readonly diagnostics?: boolean;
    /** Where problems are reported once (a GPU build that failed, a refit repaired). Default `console.warn`. */
    readonly warn?: (message: string) => void;
    /** Called when a tree is placed (a mesh's first, or a swap). */
    readonly onTree?: (event: SceneTreeEvent) => void;
    /** Builds the first trees instead of the GPU H-PLOC, whatever the device's features: for unit tests in Node. */
    readonly firstTreeBuilder?: FirstTreeBuilder;
    /**
     * The triangle format (docs/internals/design/indexed-leaves.md): `'records'` (the
     * default, 48 bytes per triangle) or `'indexed'` (16 bytes per triangle,
     * plus the scene's own positions, `buffers.positions`). The walk that
     * reads the scene must be built with the same `triangleLeaves` and
     * `positionLayout`. Indexed needs `gpuTreeOutputs: 'copy'` (each mesh
     * gets a header before its triangles, which a batch's buffers don't have).
     */
    readonly triangleLeaves?: TriangleLeaves;
    /** Indexed: `'vec4'` (the default, 16 bytes per vertex) or `'packed'` (12) positions. */
    readonly positionLayout?: PositionLayout;
    /**
     * The node format (docs/internals/design/compressed-nodes.md): `'quantized'`
     * (the default, `DEFAULT_NODE_FORMAT`: 32 bytes per node, plus a 32-byte
     * frame slot per mesh before its root) or `'f32'` (64 bytes per node). The
     * walk that reads the scene must be built with the same `nodeFormat`
     * ({@link SceneTrees.nodeFormat}; the walk's default is quantized too). tinybvh trees come
     * quantized from the workers (deforming meshes' come `f32`: their refit
     * keeps an `f32` copy, and they are packed when placed); GPU trees are
     * built `f32` and packed by a compute pass on their way into the storage.
     * Quantized needs `gpuTreeOutputs: 'copy'` (an adopted batch's nodes are
     * `f32`), so `'adopt'` needs `nodeFormat: 'f32'`. Independent of `triangleLeaves`.
     * A coordinate of magnitude 2^100 or more, or a non-finite one, makes a quantized box empty on that axis (hits there are lost).
     */
    readonly nodeFormat?: NodeFormat;
    /**
     * Indexed: weld each mesh's vertices by exact bits when it is added (not
     * deforming meshes'). It changes no coordinate, so rays see the same
     * triangles, and saves position memory. Default true.
     */
    readonly weld?: boolean;
}

/** A tree was placed: a mesh's (or merged group's) first, or a swap. */
export interface SceneTreeEvent {
    /** The mesh handle, or null for a merged group. */
    readonly mesh: number | null;
    /** The merged group's index, or null. */
    readonly group: number | null;
    readonly builtBy: TreeBuilder;
    /** The tree it replaced: `'gpu'` for a swap, null for a first tree. */
    readonly replaced: TreeBuilder | null;
    /** Why the mesh's first tree didn't come from the GPU (no subgroups, refused, too deep, tinybvh was faster, ...), or null. */
    readonly fallbackReason: string | null;
    readonly triangleCount: number;
    /** Milliseconds from the update that started the mesh's builds to this placement. */
    readonly ms: number;
}

/** One mesh's (or merged group's) tree, for tools and tests. */
export interface SceneMeshStatus {
    /** Which builder's tree rays see now, or null (none yet, or a mesh only merged instances use). */
    readonly builtBy: TreeBuilder | null;
    /** Why no GPU tree was (or will be) used, or null. */
    readonly fallbackReason: string | null;
    /** The mesh's triangles live in merged group `mergedInto` (and it has no tree of its own). */
    readonly mergedInto: number | null;
    /** Where the tree sits in {@link SceneTrees.buffers}, when placed. */
    readonly placement: TraceSceneMeshPlacement | null;
    readonly maxDepth: number;
}

/** One instance's merge state ({@link SceneTrees.instanceStatus}). */
export interface SceneInstanceStatus {
    /** The merged static group holding the instance, or null. */
    readonly mergedInto: number | null;
}

/** Counts, for tools and stats. */
export interface SceneTreeCounts {
    readonly meshes: number;
    readonly instances: number;
    readonly mergedGroups: number;
    readonly onGpuTrees: number;
    readonly onTinybvhTrees: number;
    readonly waitingForFirstTree: number;
    readonly tinybvhBuilding: number;
    readonly gpuTreesBuilt: number;
    readonly gpuTreesRefused: number;
    readonly swaps: number;
    readonly lastGpuBuildMs: number;
    /** GPU build batches whose stats were read. */
    readonly gpuBatches: number;
    /** The sum of every GPU batch's time from its submit to its stats on the CPU. */
    readonly gpuBuildMs: number;
}

/** GPU bytes the scene's trees hold. */
export interface SceneTreesMemory {
    readonly nodes: number;
    readonly nodesInUse: number;
    readonly triangles: number;
    readonly trianglesInUse: number;
    readonly topLevel: number;
    /** Indexed scenes: the positions buffer (0 for records). Part of `resident`. */
    readonly positions: number;
    /** Deforming meshes' refits (their topology, scratch and stats). */
    readonly meshRefits: number;
    /** GPU trees in their batches' buffers until copied out (copy mode), or until every tree of an adopted batch swapped (only its parents, freed at once). */
    readonly gpuBuildOutputs: number;
    /** Batches still building: their inputs, scratch and outputs, until their stats are read and the scene counts the outputs. */
    readonly gpuBuilding: number;
    /** Room in the node and triangle buffers set aside for capped GPU batches still to come (part of `nodes` and `triangles`, not of `nodesInUse` or `trianglesInUse`). */
    readonly reserved: number;
    /** Buffers replaced this update (growths, adoptions), freed at the next. */
    readonly retired: number;
    /** What stays: nodes, triangles, the top level and refits. */
    readonly resident: number;
    /** What goes away soon: outputs, batches building, retired buffers. */
    readonly transient: number;
    readonly total: number;
    /** The largest `total` seen since the scene was made (sampled at each update and each batch's start and end). */
    readonly peak: number;
}

/** When the scene got its trees, in milliseconds from the first update that had meshes. */
export interface SceneTreesTimeline {
    /** Every mesh with instances had a first tree on the GPU (placed by an update), or null. */
    readonly firstTreesMs: number | null;
    /** Every tinybvh tree was placed, or null. */
    readonly tinybvhTreesMs: number | null;
}

export interface SceneMeshOptions {
    /**
     * The mesh deforms ({@link SceneTrees.setPose}): its tree is refit from each
     * pose on the GPU and rebuilt when the refit drifts. `bounds` is the box
     * every pose stays in (`[min, max]`, object space), which the top level
     * uses so a pose never changes it; `true` uses the rest pose's box. Its
     * instances never merge.
     */
    readonly deforming?: boolean | { readonly bounds: ArrayLike<number> };
}

export interface SceneInstanceDescriptor {
    readonly mesh: number;
    readonly objectToWorld: Mat4;
    /** Returned in `BvhHit.instance`. Default: the instance's handle. */
    readonly instanceIndex?: number;
    readonly visible?: boolean;
    readonly contributesToGi?: boolean;
    /** 0-255; default 0xff. */
    readonly mask?: number;
    /** Per-instance material index (0-65534), written as `material + 1` in the record's flags bits 16-31. */
    readonly material?: number;
    /** Never moves: merged with `mergeStaticInstances` if it is added before the first update. */
    readonly static?: boolean;
}

/** What {@link SceneTrees.setInstance} can change. */
export interface SceneInstanceChange {
    readonly visible?: boolean;
    readonly contributesToGi?: boolean;
    readonly mask?: number;
    readonly material?: number | null;
}

/** Where a mesh's tree sits. Numbers only: it must never hold a tree's arrays (they would stay on the heap). */
interface PlacedTree {
    readonly nodeOffset: number;
    readonly nodeCount: number;
    readonly triangleOffset: number;
    readonly triangleCount: number;
    /** Indexed: units before `triangleOffset` that belong to the tree (its header, a merged group's side tables); 0 for records. */
    readonly triangleLeading: number;
    readonly maxDepth: number;
    readonly bounds: readonly number[];
    readonly source: TreeBuilder;
    readonly generation: number;
}

interface MeshSlot {
    readonly slot: number;
    /** Deforms: refit from poses, never merged. */
    readonly deforming: boolean;
    generation: number;
    /** The merged group this slot builds, or -1 for a mesh the caller added. */
    readonly group: number;
    /** Kept until the builds start. Indexed: welded (unless deforming or `weld: false`). */
    geometry: MeshGeometry | null;
    /** Indexed: where the mesh's positions sit in the positions buffer, once the builds started. */
    vertices: { offset: number; count: number } | null;
    readonly triangleCount: number;
    readonly bounds: Float32Array;
    /** Indexed, unmerged: each original triangle's material code (`triangleMaterials`), for the table before its header; else null. */
    readonly materialCodes: Uint16Array | null;
    /** Deforming meshes: what a CPU pose ({@link SceneTrees.setPoseCpu}) needs; null for the others. */
    readonly cpuPose: CpuPose | null;
    gpu: 'none' | 'building' | 'built' | 'refused';
    tinybvh: 'none' | 'building' | 'built' | 'failed';
    fallbackReason: string | null;
    /** Only merged instances use it: it has no tree of its own. */
    mergedInto: number | null;
    users: number;
    startedAt: number;
    abort: AbortController | null;
}

/** A deforming mesh's CPU pose state. */
interface CpuPose {
    /** The mesh's own indices (a copy: the geometry may move into a worker), or null for a triangle list. */
    readonly indices: Uint16Array | Uint32Array | null;
    readonly vertexCount: number;
    /** The placed tree's layout on the CPU (tinybvh's, or a GPU tree's read back), or null before one arrives. */
    layout: GpuLayoutMeshBvh | null;
    /** The latest CPU pose, or null. */
    positions: Float32Array | null;
    /** The pose still has to be written into the placed tree. */
    due: boolean;
    /** A CPU pose moved the mesh's top-level box off its deform box. */
    moved: boolean;
}

interface InstanceSlot {
    readonly slot: number;
    readonly mesh: number;
    objectToWorld: Float64Array;
    readonly instanceIndex: number;
    visible: boolean;
    contributesToGi: boolean;
    mask: number;
    material: number | null;
    readonly static: boolean;
    /** The merged group holding it, or -1. */
    merged: number;
}

interface MergedGroup {
    readonly group: StaticMergeGroup;
    readonly meshSlot: number;
    readonly instanceSlot: number;
}

type ArrivedTree =
    | { readonly source: 'tinybvh'; readonly slot: number; readonly generation: number; readonly bvh: MeshBvh }
    | {
          readonly source: 'gpu';
          readonly slot: number;
          readonly generation: number;
          readonly built: GpuBuiltTree;
          /** A deforming mesh's drift rebuild: it replaces whatever tree the mesh has, tinybvh's too. */
          readonly rebuild: boolean;
          /** Already in the scene's buffers at these offsets (a capped batch copied out, or adopted, between updates); the next update only places it. */
          readonly stored?: StoredAt;
      };
type GpuArrivedTree = Extract<ArrivedTree, { source: 'gpu' }>;

/** Where a tree stored between updates sits: its offsets, and (indexed) its leading units before the triangles. */
interface StoredAt {
    readonly nodeOffset: number;
    readonly triangleOffset: number;
    readonly triangleLeading: number;
}

/**
 * Room set aside in a storage for trees still to come (a capped load's later
 * batches): ranges marked in use that hold nothing. A tree takes its range
 * out of them; what is not needed any more is given back to the storage.
 */
class StorageReserve {
    private ranges: { offset: number; count: number }[] = [];

    get units(): number {
        return this.ranges.reduce((total, range) => total + range.count, 0);
    }

    /** The reserved ranges (for tests and growth plans). */
    get all(): readonly { readonly offset: number; readonly count: number }[] {
        return this.ranges;
    }

    /** Sets `count` more units aside, if the storage has them free. */
    add(storage: GrowableStorage, count: number): void {
        if (count <= 0) return;
        const offset = storage.allocate(count);
        if (offset !== null) this.ranges.push({ offset, count });
    }

    /** A range of `count` units out of the reserve (it stays in use, now the tree's), or null. */
    take(count: number): number | null {
        if (count <= 0) return null;
        const index = this.ranges.findIndex((range) => range.count >= count);
        if (index < 0) return null;
        const range = this.ranges[index];
        const offset = range.offset;
        range.offset += count;
        range.count -= count;
        if (range.count === 0) this.ranges.splice(index, 1);
        return offset;
    }

    /** Puts a range {@link take} gave back (a plan that failed). */
    giveBack(offset: number, count: number): void {
        this.ranges.push({ offset, count });
    }

    /** Frees every unit past the first `keep`, the last ranges first. */
    shrink(storage: GrowableStorage, keep: number): void {
        let excess = this.units - Math.max(0, keep);
        while (excess > 0 && this.ranges.length > 0) {
            const last = this.ranges[this.ranges.length - 1];
            const freed = Math.min(excess, last.count);
            storage.release(last.offset + last.count - freed, freed);
            last.count -= freed;
            if (last.count === 0) this.ranges.pop();
            excess -= freed;
        }
    }
}

/** A pool of free slot numbers, lowest first. */
class SlotPool {
    private next = 0;
    private readonly free: number[] = [];
    take(): number {
        if (this.free.length > 0) {
            this.free.sort((a, b) => b - a);
            return this.free.pop()!;
        }
        return this.next++;
    }
    give(slot: number): void {
        this.free.push(slot);
    }
}

function geometryBounds(positions: Float32Array): Float32Array {
    const bounds = new Float32Array([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
    for (let at = 0; at < positions.length; at += 3) {
        for (let axis = 0; axis < 3; axis++) {
            const value = positions[at + axis];
            if (value < bounds[axis]) bounds[axis] = value;
            if (value > bounds[3 + axis]) bounds[3 + axis] = value;
        }
    }
    return bounds;
}

function describe(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

export class SceneTrees {
    /** Makes the scene and compiles its kernels (the GPU builder first: the first trees wait on it). */
    static async create(device: GPUDevice, options: SceneTreesOptions = {}): Promise<SceneTrees> {
        const scene = new SceneTrees(device, options);
        await scene.prepare();
        return scene;
    }

    readonly device: GPUDevice;
    /** `'gpu'` (the GPU top level, refit) on a device with subgroups; `'cpu'` otherwise or when asked. */
    readonly topLevelPath: 'gpu' | 'cpu';
    readonly gpuTreeOutputs: GpuTreeOutputsMode;

    private readonly label: string;
    private readonly warn: (message: string) => void;
    private readonly onTree: ((event: SceneTreeEvent) => void) | null;
    private readonly pool: BvhWorkerPool;
    private readonly ownsPool: boolean;
    private readonly transferGeometry: boolean;
    private readonly mergeStatic: boolean;
    private readonly gpuMaxDepth: number;
    private refinementHold: ((meshSlot: number) => boolean) | null;
    private firstTrees: FirstTreeBuilder | null;
    private firstTreesReady = false;
    private noGpuReason: string | null;

    private readonly nodes: GrowableStorage;
    private readonly triangles: GrowableStorage;
    /** The scene's triangle format, its triangles' unit, and (indexed) the positions and their layout. */
    readonly triangleLeaves: TriangleLeaves;
    readonly positionLayout: PositionLayout;
    private readonly triangleBytes: number;
    /** The scene's node format, its nodes' unit, and the slots before each root (the frame slot when quantized). */
    readonly nodeFormat: NodeFormat;
    private readonly nodeBytes: number;
    private readonly nodeLeading: number;
    /** Quantize passes recorded into the caller's encoder (GPU trees placed), freed at the next update. */
    private quantizeJobsToRelease: QuantizeJob[] = [];
    private readonly weld: boolean;
    private readonly positions: GrowableStorage | null;
    /** Removed meshes' position ranges, freed at the next update. */
    private releasePositionsAtUpdate: { offset: number; count: number }[] = [];
    private readonly gpuTopLevel: GpuTopLevel | null;
    private cpuTopLevel: GPUBuffer;
    private cpuTopLevelStale = true;
    private cpuTopLevelInstanceCount = 0;
    private readonly retiredTopLevel: GPUBuffer[] = [];
    private readonly identity: GpuMergedIdentity;
    /** Deforming meshes' refits and drift rebuilds; null without subgroups (they keep the pose their tree was built in). */
    private readonly deforming: DeformingTrees | null;
    private posesSkipped = 0;
    private posesSkippedWarned = false;
    private cpuPoseLayoutWarned = false;
    /** CPU poses written ({@link setPoseCpu}). */
    private cpuPoses = 0;
    /** Staging buffers of CPU poses copied in the last update's encoder, freed at the next update. */
    private retiredStaging: GPUBuffer[] = [];

    private readonly meshSlots = new SlotPool();
    private readonly instanceSlots = new SlotPool();
    private readonly meshes = new Map<number, MeshSlot>();
    private readonly instances = new Map<number, InstanceSlot>();
    private readonly placed = new Map<number, PlacedTree>();
    private readonly groups: MergedGroup[] = [];
    private plan: StaticMergePlan | null = null;
    /** Meshes added since the last update: their builds start at the next. */
    private newMeshes: number[] = [];
    /** Instances whose top-level entry must be written (or removed) at the next update. */
    private readonly dirtyInstances = new Set<number>();
    private arrived: ArrivedTree[] = [];
    private pendingTinybvh = 0;
    private readonly liveOutputs = new Set<GpuTreeOutputs>();
    private outputsToRelease: GpuTreeOutputs[] = [];
    /** GPU builds still running; their bytes count in `memory.gpuBuilding`. */
    private readonly buildsRunning = new Set<GpuTreeBuilds>();
    /** Meshes whose geometry a capped GPU build still has to upload: not moved into a worker. */
    private readonly gpuReadsGeometryLater = new Set<number>();
    private readonly maxBatchBytes: number | undefined;
    /** Room set aside for capped batches still to come (`gpuMaxBatchBytes`). */
    private readonly reserve = { nodes: new StorageReserve(), triangles: new StorageReserve() };
    /** The nodes per triangle the node reserve is sized at: the GPU trees' last seen. */
    private reserveNodesPerTriangle = 1;
    /**
     * The last update recorded something into the node or triangle buffer
     * (copies, a growth, the identity pass, a refit) that its encoder still
     * has to run: growing between updates would lose it.
     */
    private encoderWritesStorage = false;
    /** Ranges of trees removed between updates: freed at the next update (frames recorded before may still trace them). */
    private releaseAtUpdate: PlacedTree[] = [];
    /** Capped batches' outputs that only an update can let go (their trees could not be stored between updates): the batches behind them wait. */
    private readonly outputsWaitingForUpdate = new Set<GpuTreeOutputs>();
    private gpuBatches = 0;
    private gpuBuildMs = 0;
    private buildFailure: Error | null = null;
    private builtWaiters: (() => void)[] = [];
    private refinedWaiters: (() => void)[] = [];
    private gpuTreesBuilt = 0;
    private gpuTreesRefused = 0;
    private swaps = 0;
    private lastGpuBuildMs = 0;
    private updates = 0;
    private loadStartedAt: number | null = null;
    private firstTreesAt: number | null = null;
    private tinybvhTreesAt: number | null = null;
    private peakBytes = 0;
    private versionBase = 0;
    private traceBuffersCache: { version: number; buffers: TraceSceneBuffers } | null = null;
    private prepared = false;
    private disposed = false;

    private constructor(device: GPUDevice, options: SceneTreesOptions) {
        this.device = device;
        this.label = options.label ?? 'bvh';
        this.warn = options.warn ?? ((message) => console.warn(message));
        this.onTree = options.onTree ?? null;
        this.gpuTreeOutputs = options.gpuTreeOutputs ?? 'copy';
        const maxBatchBytes = options.gpuMaxBatchBytes ?? DEFAULT_GPU_MAX_BATCH_BYTES;
        if (!(maxBatchBytes > 0)) throw new RangeError(`gpuMaxBatchBytes must be a positive number of bytes (Infinity: no cap), got ${maxBatchBytes}`);
        this.maxBatchBytes = maxBatchBytes === Infinity ? undefined : maxBatchBytes;
        this.transferGeometry = options.transferGeometry ?? false;
        this.mergeStatic = options.mergeStaticInstances ?? true;
        this.gpuMaxDepth = options.gpuMaxDepth ?? GPU_TREE_MAX_DEPTH;
        this.refinementHold = options.holdRefinements ? () => true : null;
        const workers = options.workers;
        if (workers && 'build' in workers) {
            this.pool = workers;
            this.ownsPool = false;
        } else {
            this.pool = createBvhWorkerPool(workers ?? {});
            this.ownsPool = true;
        }
        if (options.gpuFirstTrees === false) {
            this.firstTrees = null;
            this.noGpuReason = 'GPU first trees are off (gpuFirstTrees: false)';
        } else if (options.firstTreeBuilder) {
            this.firstTrees = options.firstTreeBuilder;
            this.noGpuReason = null;
        } else if (GpuFirstTrees.isSupported(device)) {
            this.firstTrees = new GpuFirstTrees(device, { label: this.label, triangleLeaves: options.triangleLeaves });
            this.noGpuReason = null;
        } else {
            this.firstTrees = null;
            this.noGpuReason = 'the device has no subgroups, so there is no GPU builder (tinybvh is the fallback)';
        }
        this.triangleLeaves = options.triangleLeaves ?? 'records';
        if (this.triangleLeaves !== 'records' && this.triangleLeaves !== 'indexed') throw new RangeError(`triangleLeaves must be 'records' or 'indexed', got ${String(this.triangleLeaves)}`);
        this.positionLayout = options.positionLayout ?? 'vec4';
        this.weld = options.weld ?? true;
        const indexed = this.triangleLeaves === 'indexed';
        if (indexed && this.gpuTreeOutputs === 'adopt') throw new RangeError("SceneTrees: triangleLeaves: 'indexed' needs gpuTreeOutputs: 'copy' (an adopted batch's buffers have no mesh headers)");
        this.triangleBytes = indexed ? INDEXED_TRIANGLE_BYTES : TRIANGLE_BYTES;
        this.nodeFormat = options.nodeFormat ?? DEFAULT_NODE_FORMAT;
        if (this.nodeFormat !== 'f32' && this.nodeFormat !== 'quantized') throw new RangeError(`nodeFormat must be 'f32' or 'quantized', got ${String(this.nodeFormat)}`);
        if (this.nodeFormat === 'quantized' && this.gpuTreeOutputs === 'adopt') {
            throw new RangeError(
                options.nodeFormat === undefined
                    ? "SceneTrees: gpuTreeOutputs: 'adopt' needs nodeFormat: 'f32' (an adopted batch's nodes are f32, and the default node format is 'quantized')"
                    : "SceneTrees: nodeFormat: 'quantized' needs gpuTreeOutputs: 'copy' (an adopted batch's nodes are f32)",
            );
        }
        this.nodeBytes = this.nodeFormat === 'quantized' ? QUANTIZED_NODE_BYTES : BVH_NODE_BYTES;
        this.nodeLeading = this.nodeFormat === 'quantized' ? 1 : 0;
        this.nodes = new GrowableStorage(device, `${this.label} trace nodes`, this.nodeBytes, INITIAL_NODES);
        this.triangles = new GrowableStorage(device, `${this.label} trace triangles`, this.triangleBytes, indexed ? INITIAL_TRIANGLES * 3 : INITIAL_TRIANGLES);
        this.positions = indexed ? new GrowableStorage(device, `${this.label} trace positions`, positionStrideFloats(this.positionLayout) * 4, INITIAL_TRIANGLES) : null;
        this.topLevelPath = options.topLevel !== 'cpu' && GpuTopLevel.isSupported(device) ? 'gpu' : 'cpu';
        this.gpuTopLevel =
            this.topLevelPath === 'gpu'
                ? new GpuTopLevel(device, { label: this.label, buildIn: this.pool.usesWorkers ? 'worker' : 'main', diagnostics: options.diagnostics, warn: this.warn })
                : null;
        // An empty top level: the header says 0 nodes, so every ray misses.
        this.cpuTopLevel = this.createTopLevelBuffer(TOP_LEVEL_HEADER_BYTES);
        device.queue.writeBuffer(this.cpuTopLevel, 0, new Uint32Array(TOP_LEVEL_HEADER_WORDS));
        this.identity = new GpuMergedIdentity(device, this.label);
        this.deforming = DeformingTrees.isSupported(device) ? new DeformingTrees(device, { label: this.label }) : null;
    }

    /** Compiles the GPU builder, the top level and the merged-identity pass, all at once. */
    private async prepare(): Promise<void> {
        await Promise.all([this.prepareFirstTrees(), this.gpuTopLevel?.prepare(), this.firstTrees ? this.identity.prepare() : undefined, this.deforming?.prepare()]);
        this.prepared = true;
    }

    /** If the GPU builder fails to compile, meshes wait for tinybvh (the fallback always stays). */
    private async prepareFirstTrees(): Promise<void> {
        if (!this.firstTrees) return;
        try {
            await this.firstTrees.prepare();
            this.firstTreesReady = true;
        } catch (error) {
            this.firstTrees = null;
            this.noGpuReason = `the GPU builder could not be compiled: ${describe(error)}`;
            this.warn(`@pmndrs/bvh: ${this.noGpuReason}; mesh trees come from tinybvh only`);
        }
    }

    // -----------------------------------------------------------------------
    // What an app binds
    // -----------------------------------------------------------------------

    /**
     * The buffers `traceModuleWgsl` binds (`bvhTraceNodes`, `bvhTraceTriangles`,
     * `bvhTraceTopLevel`, and with `triangleLeaves: 'indexed'` the fourth,
     * `bvhTracePositions`; null for records).
     */
    get buffers(): { readonly nodes: GPUBuffer; readonly triangles: GPUBuffer; readonly topLevel: GPUBuffer; readonly positions: GPUBuffer | null } {
        return { nodes: this.nodes.buffer, triangles: this.triangles.buffer, topLevel: this.topLevelBuffer, positions: this.positions?.buffer ?? null };
    }

    /**
     * Increments whenever one of {@link buffers} is replaced: make bind groups
     * again then. Usually in an update; with `gpuMaxBatchBytes` also between
     * updates, when a capped batch grows the buffers on its own (the old ones
     * stay valid until the next update).
     */
    get version(): number {
        return this.versionBase + this.nodes.version + this.triangles.version + (this.positions?.version ?? 0) + (this.gpuTopLevel?.version ?? 0);
    }

    /** The buffers as `TraceBatchKernel` takes them; the same object until {@link version} changes. */
    get traceBuffers(): TraceSceneBuffers {
        const version = this.version;
        if (this.traceBuffersCache?.version !== version) {
            const { nodes, triangles, topLevel, positions } = this.buffers;
            this.traceBuffersCache = {
                version,
                buffers: {
                    nodes,
                    triangles,
                    topLevel,
                    positions,
                    triangleLeaves: this.triangleLeaves,
                    positionLayout: this.positionLayout,
                    nodeFormat: this.nodeFormat,
                    byteLength: nodes.size + triangles.size + topLevel.size + (positions?.size ?? 0),
                    destroy: () => {},
                },
            };
        }
        return this.traceBuffersCache.buffers;
    }

    private get topLevelBuffer(): GPUBuffer {
        return this.gpuTopLevel ? this.gpuTopLevel.buffer : this.cpuTopLevel;
    }

    // -----------------------------------------------------------------------
    // Meshes and instances
    // -----------------------------------------------------------------------

    /**
     * Adds a mesh. Its trees are built from the next {@link update}. The
     * geometry is read then (and copied into the worker, unless
     * `transferGeometry`). Returns the mesh's handle.
     */
    addMesh(geometry: MeshGeometry, options: SceneMeshOptions = {}): number {
        this.checkLive();
        const triangleCount = countTriangles(geometry);
        checkTriangleFlags(geometry.triangleFlags, triangleCount, 'SceneTrees.addMesh');
        const materialCodes = triangleMaterialCodes(geometry.triangleMaterials, triangleCount, 'SceneTrees.addMesh');
        if (triangleCount === 0) throw new RangeError('SceneTrees.addMesh: the mesh has no triangles');
        const slot = this.meshSlots.take();
        const state = this.newMeshSlot(slot, -1, geometry, triangleCount, options.deforming !== undefined && options.deforming !== false, materialCodes);
        this.meshes.set(slot, state);
        if (state.deforming && this.deforming) {
            const box = typeof options.deforming === 'object' ? options.deforming.bounds : state.bounds;
            // A drift rebuild reads the pose from the GPU: it gets the flags and materials from a buffer of its own.
            const flags = geometry.triangleFlags || geometry.triangleMaterials ? packedTriangleFlags(geometry, triangleCount, 'SceneTrees.addMesh') : null;
            this.deforming.meshAdded(slot, triangleCount, box, flags);
        }
        this.newMeshes.push(slot);
        return slot;
    }

    /**
     * A deforming mesh's new pose: its moved vertices on the GPU (the mesh's
     * own indices and triangle order, `STORAGE` usage; `COPY_SRC` too for drift
     * rebuilds). The next update refits its tree from them, before anything
     * traces. Without subgroups the pose is skipped (counted in
     * `deformingCounts.posesSkipped`, with one warning) and rays see the pose
     * the tree was built in: give such devices the pose on the CPU with
     * {@link setPoseCpu} instead.
     */
    setPose(mesh: number, vertices: MeshRefitVertices): void {
        this.checkLive();
        const state = this.meshes.get(mesh);
        if (!state || state.group >= 0) throw new RangeError(`SceneTrees.setPose: no mesh ${mesh}`);
        if (!state.deforming) throw new RangeError(`SceneTrees.setPose: mesh ${mesh} was not added as deforming`);
        if (!this.deforming) {
            this.posesSkipped++;
            if (!this.posesSkippedWarned) {
                this.posesSkippedWarned = true;
                this.warn(
                    '@pmndrs/bvh: SceneTrees.setPose was skipped: the device has no subgroups, so there is no GPU refit and the mesh is traced in the pose its tree was built in. ' +
                        'Give the pose on the CPU with SceneTrees.setPoseCpu(mesh, positions) instead (deformingCounts.posesSkipped counts the skipped poses).',
                );
            }
            return;
        }
        this.deforming.posed(mesh, vertices);
        // The latest pose wins: a CPU pose given before is not written again, and the top level gets the deform box back.
        if (state.cpuPose) {
            state.cpuPose.positions = null;
            state.cpuPose.due = false;
            if (state.cpuPose.moved) {
                state.cpuPose.moved = false;
                const box = this.deforming.deformBox(mesh);
                if (box) this.moveTopLevelBox(mesh, box);
            }
        }
    }

    /**
     * A deforming mesh's new pose on the CPU: `positions` holds `x, y, z` per
     * vertex in the mesh's own numbering (as many vertices as the geometry it
     * was added with; its indices and triangle order are kept). The next
     * {@link update} refits the mesh's tree on the CPU (`refitLayoutToPose`),
     * writes its nodes (re-quantized with `nodeFormat: 'quantized'`) and its
     * corners (indexed: its vertices in the positions buffer) through the
     * update's encoder, and moves the mesh's top-level box to the posed
     * tree's root box. With or without subgroups: the path for devices
     * without them, where {@link setPose} is skipped, and for poses an app
     * computes on the CPU. A tree that arrives later (a swap, a rebuild) gets
     * the latest pose too. It costs a pass over the triangles and an upload
     * of the mesh's nodes and triangles per pose, so it suits small meshes or
     * a few poses a second. `positions` is read at the update; it must not
     * change before then. With subgroups a later {@link setPose} replaces it;
     * without them `setPose` is skipped, so this pose (or a later `setPoseCpu`) stays.
     */
    setPoseCpu(mesh: number, positions: Float32Array): void {
        this.checkLive();
        const state = this.meshes.get(mesh);
        if (!state || state.group >= 0) throw new RangeError(`SceneTrees.setPoseCpu: no mesh ${mesh}`);
        if (!state.deforming || !state.cpuPose) throw new RangeError(`SceneTrees.setPoseCpu: mesh ${mesh} was not added as deforming`);
        if (positions.length !== state.cpuPose.vertexCount * 3) {
            throw new RangeError(`SceneTrees.setPoseCpu: mesh ${mesh} has ${state.cpuPose.vertexCount} vertices, the pose ${positions.length / 3}`);
        }
        state.cpuPose.positions = positions;
        state.cpuPose.due = true;
        // The latest pose wins: a GPU pose given before is not refit again.
        this.deforming?.unposed(mesh);
    }

    /**
     * A deforming mesh's new deform box: the box (`[minX, minY, minZ, maxX,
     * maxY, maxZ]`, object space) every pose from now on stays in, which the
     * top level holds for its instances. For poses that leave the box given at
     * {@link addMesh} (a skinned character walking: `@pmndrs/bvh/three`'s sync
     * sets it from the skeleton each update). Every instance of the mesh is
     * refit into the top level at the next {@link update}. Without subgroups
     * (poses are skipped) it does nothing.
     */
    setDeformBounds(mesh: number, bounds: ArrayLike<number>): void {
        this.checkLive();
        const state = this.meshes.get(mesh);
        if (!state || state.group >= 0) throw new RangeError(`SceneTrees.setDeformBounds: no mesh ${mesh}`);
        if (!state.deforming) throw new RangeError(`SceneTrees.setDeformBounds: mesh ${mesh} was not added as deforming`);
        for (let k = 0; k < 3; k++) {
            if (!(bounds[k] <= bounds[k + 3])) throw new RangeError(`SceneTrees.setDeformBounds: [${Array.from(bounds).join(', ')}] is not a box`);
        }
        if (!this.deforming?.setDeformBox(mesh, bounds)) return;
        const placed = this.placed.get(mesh);
        if (placed) {
            const moved: PlacedTree = { ...placed, bounds: Array.from({ length: 6 }, (_, k) => bounds[k]) };
            this.placed.set(mesh, moved);
            this.gpuTopLevel?.setMeshTree(mesh, moved);
        }
        for (const [slot, instance] of this.instances) if (instance.mesh === mesh) this.dirtyInstances.add(slot);
        this.cpuTopLevelStale = true;
    }

    /** Deforming meshes' refits, drift rebuilds, poses skipped without subgroups, and CPU poses written ({@link setPoseCpu}). */
    get deformingCounts(): DeformingTreeCounts & { readonly posesSkipped: number; readonly cpuPoses: number } {
        return { ...(this.deforming?.counts ?? { refitting: 0, refits: 0, driftRebuilds: 0 }), posesSkipped: this.posesSkipped, cpuPoses: this.cpuPoses };
    }

    /** A deforming mesh's latest refit cost over its built tree's (NaN before a sample, or without subgroups). */
    driftRatio(mesh: number): number {
        return this.deforming?.driftRatio(mesh) ?? NaN;
    }

    private newMeshSlot(slot: number, group: number, geometry: MeshGeometry, triangleCount: number, deforming = false, materialCodes: Uint16Array | null = null): MeshSlot {
        const previous = this.retiredGenerations.get(slot) ?? 0;
        // Indexed: the exact-bit weld (no coordinate moves). A deforming mesh's pose is copied vertex for vertex, so it stays unwelded.
        const welded = this.triangleLeaves === 'indexed' && this.weld && !deforming ? weldPositions(geometry) : geometry;
        return {
            slot,
            deforming,
            generation: previous + 1,
            group,
            geometry: welded,
            vertices: null,
            triangleCount,
            bounds: geometryBounds(geometry.positions),
            materialCodes: this.triangleLeaves === 'indexed' ? materialCodes : null,
            cpuPose: deforming ? { indices: geometry.indices ? geometry.indices.slice() : null, vertexCount: geometry.positions.length / 3, layout: null, positions: null, due: false, moved: false } : null,
            gpu: 'none',
            tinybvh: 'none',
            fallbackReason: null,
            mergedInto: null,
            users: 0,
            startedAt: 0,
            abort: null,
        };
    }

    /** Generations of removed slots, so a reused slot never takes a late tree of its old mesh. */
    private readonly retiredGenerations = new Map<number, number>();

    /** Removes a mesh no instance uses. Its tree's range is freed; a build still running is cancelled or ignored. */
    removeMesh(mesh: number): void {
        this.checkLive();
        const state = this.meshes.get(mesh);
        if (!state || state.group >= 0) throw new RangeError(`SceneTrees.removeMesh: no mesh ${mesh}`);
        if (state.users > 0) throw new RangeError(`SceneTrees.removeMesh: mesh ${mesh} is still used by ${state.users} instance(s)`);
        state.abort?.abort();
        // Its ranges are freed at the next update: a capped batch stored between updates must not reuse them while a frame recorded before may still trace them.
        const tree = this.placed.get(mesh);
        if (tree) {
            this.releaseAtUpdate.push(tree);
            this.forgetTree(mesh);
        }
        if (state.vertices) this.releasePositionsAtUpdate.push(state.vertices);
        this.deforming?.meshRemoved(mesh);
        this.meshes.delete(mesh);
        this.newMeshes = this.newMeshes.filter((slot) => slot !== mesh);
        this.retiredGenerations.set(mesh, state.generation);
        this.meshSlots.give(mesh);
        this.cpuTopLevelStale = true;
        this.reconcileReserve();
        this.notifyWaiters();
    }

    /** Adds an instance of a mesh. It becomes visible to rays once its mesh has a tree and (GPU top level) a shape holds it. */
    addInstance(descriptor: SceneInstanceDescriptor): number {
        this.checkLive();
        const mesh = this.meshes.get(descriptor.mesh);
        if (!mesh || mesh.group >= 0) throw new RangeError(`SceneTrees.addInstance: no mesh ${descriptor.mesh}`);
        if (mesh.mergedInto !== null) {
            throw new RangeError(
                `SceneTrees.addInstance: mesh ${descriptor.mesh} was merged away into static group ${mesh.mergedInto} (only merged instances used it, so its geometry was dropped at the first update); ` +
                    'add the mesh again with addMesh for new instances. To keep a mesh for later instances, add its instances without static: true, or create the scene with mergeStaticInstances: false (merging is on by default)',
            );
        }
        const slot = this.instanceSlots.take();
        const mask = checkInstanceMask(descriptor.mask, descriptor.instanceIndex ?? slot);
        checkMaterial(descriptor.material);
        this.instances.set(slot, {
            slot,
            mesh: descriptor.mesh,
            objectToWorld: Float64Array.from({ length: 16 }, (_, k) => descriptor.objectToWorld[k]),
            instanceIndex: descriptor.instanceIndex ?? slot,
            visible: descriptor.visible !== false,
            contributesToGi: descriptor.contributesToGi !== false,
            mask,
            material: descriptor.material ?? null,
            static: descriptor.static === true,
            merged: -1,
        });
        mesh.users++;
        this.dirtyInstances.add(slot);
        this.cpuTopLevelStale = true;
        return slot;
    }

    /** Moves an instance. Throws for a merged static instance. */
    setTransform(instance: number, objectToWorld: Mat4): void {
        const state = this.instanceState(instance, 'setTransform');
        if (state.merged >= 0) throwMergedTransformChange(state.instanceIndex, 'SceneTrees.setTransform');
        for (let k = 0; k < 16; k++) state.objectToWorld[k] = objectToWorld[k];
        this.dirtyInstances.add(instance);
        this.cpuTopLevelStale = true;
    }

    /** Hides, shows, re-masks or re-materials an instance. A merged instance's visibility and mask can't change. */
    setInstance(instance: number, change: SceneInstanceChange): void {
        const state = this.instanceState(instance, 'setInstance');
        const mask = change.mask === undefined ? state.mask : checkInstanceMask(change.mask, state.instanceIndex);
        const visible = change.visible ?? state.visible;
        const contributesToGi = change.contributesToGi ?? state.contributesToGi;
        if (state.merged >= 0 && (mask !== state.mask || visible !== state.visible || contributesToGi !== state.contributesToGi)) {
            throw new RangeError(
                `SceneTrees.setInstance: instance ${state.instanceIndex} is merged into static group ${state.merged}; its visibility and mask are baked into the group and can't change. Add it with static: false to change them`,
            );
        }
        if (change.material !== undefined) {
            if (change.material !== null) checkMaterial(change.material);
            // A merged instance's material travels in its group's triangles (flags bits 16-31), baked at the merge.
            if (state.merged >= 0 && change.material !== state.material) throw new RangeError(`SceneTrees.setInstance: instance ${state.instanceIndex} is merged; its material can't change`);
            state.material = change.material;
        }
        state.mask = mask;
        state.visible = visible;
        state.contributesToGi = contributesToGi;
        this.dirtyInstances.add(instance);
        this.cpuTopLevelStale = true;
    }

    /** Removes an instance. Throws for a merged static instance. */
    removeInstance(instance: number): void {
        const state = this.instanceState(instance, 'removeInstance');
        if (state.merged >= 0) {
            throw new RangeError(`SceneTrees.removeInstance: instance ${state.instanceIndex} is merged into static group ${state.merged}; merged instances can't be removed. Add it with static: false to remove it later`);
        }
        this.instances.delete(instance);
        const mesh = this.meshes.get(state.mesh);
        if (mesh) mesh.users--;
        this.dirtyInstances.add(instance);
        this.cpuTopLevelStale = true;
        this.notifyWaiters();
    }

    private instanceState(instance: number, where: string): InstanceSlot {
        this.checkLive();
        const state = this.instances.get(instance);
        if (!state) throw new RangeError(`SceneTrees.${where}: no instance ${instance}`);
        return state;
    }

    // -----------------------------------------------------------------------
    // Status
    // -----------------------------------------------------------------------

    /** True while a mesh with instances waits for its first tree, or (GPU top level) a shape is being built. */
    get building(): boolean {
        return this.meshesWaitingForFirstTree() > 0 || this.newMeshes.length > 0 || (this.gpuTopLevel?.building ?? false);
    }

    /** True while tinybvh trees are still being built (each replaces a GPU tree, or is a mesh's first). */
    get refining(): boolean {
        return this.pendingTinybvh > 0 || this.arrived.some((tree) => tree.source === 'tinybvh');
    }

    /** Whether first trees come from the GPU, and if not, why. */
    get gpuFirstTrees(): { readonly enabled: boolean; readonly reason: string | null } {
        return { enabled: this.firstTrees !== null, reason: this.firstTrees ? null : this.noGpuReason };
    }

    /** Whether tinybvh builds run in workers. */
    get usesWorkers(): boolean {
        return this.pool.usesWorkers;
    }

    get counts(): SceneTreeCounts {
        let onGpuTrees = 0;
        let onTinybvhTrees = 0;
        for (const tree of this.placed.values()) {
            if (tree.source === 'gpu') onGpuTrees++;
            else onTinybvhTrees++;
        }
        let meshes = 0;
        for (const mesh of this.meshes.values()) if (mesh.group < 0) meshes++;
        return {
            meshes,
            instances: this.instances.size,
            mergedGroups: this.groups.length,
            onGpuTrees,
            onTinybvhTrees,
            waitingForFirstTree: this.meshesWaitingForFirstTree(),
            tinybvhBuilding: this.pendingTinybvh,
            gpuTreesBuilt: this.gpuTreesBuilt,
            gpuTreesRefused: this.gpuTreesRefused,
            swaps: this.swaps,
            lastGpuBuildMs: this.lastGpuBuildMs,
            gpuBatches: this.gpuBatches,
            gpuBuildMs: this.gpuBuildMs,
        };
    }

    get timeline(): SceneTreesTimeline {
        const since = (at: number | null) => (at === null || this.loadStartedAt === null ? null : at - this.loadStartedAt);
        return { firstTreesMs: since(this.firstTreesAt), tinybvhTreesMs: since(this.tinybvhTreesAt) };
    }

    /** A mesh's tree: which builder made it, why not the GPU, where it is. */
    meshStatus(mesh: number): SceneMeshStatus {
        const state = this.meshes.get(mesh);
        if (!state || state.group >= 0) throw new RangeError(`SceneTrees.meshStatus: no mesh ${mesh}`);
        return this.statusOf(state);
    }

    /**
     * An instance's merge state: `mergedInto` is the static group it was
     * merged into at the first update (then it can't move, change or be
     * removed), or null.
     */
    instanceStatus(instance: number): SceneInstanceStatus {
        const state = this.instances.get(instance);
        if (!state) throw new RangeError(`SceneTrees.instanceStatus: no instance ${instance}`);
        return { mergedInto: state.merged >= 0 ? state.merged : null };
    }

    /** A merged group's tree, by group index (0 to `counts.mergedGroups - 1`). */
    groupStatus(group: number): SceneMeshStatus & { readonly sourceInstances: readonly number[] } {
        const entry = this.groups[group];
        if (!entry) throw new RangeError(`SceneTrees.groupStatus: no merged group ${group}`);
        return { ...this.statusOf(this.meshes.get(entry.meshSlot)!), sourceInstances: entry.group.sources.map((source) => source.instanceIndex) };
    }

    private statusOf(state: MeshSlot): SceneMeshStatus {
        const tree = this.placed.get(state.slot);
        return {
            builtBy: tree?.source ?? null,
            fallbackReason: state.fallbackReason,
            mergedInto: state.mergedInto,
            placement: tree ? { nodeOffset: tree.nodeOffset, nodeCount: tree.nodeCount, triangleOffset: tree.triangleOffset, triangleCount: tree.triangleCount } : null,
            maxDepth: tree?.maxDepth ?? 0,
        };
    }

    get memory(): SceneTreesMemory {
        const nodes = this.nodes.byteLength;
        const triangles = this.triangles.byteLength;
        const topLevel = this.gpuTopLevel ? this.gpuTopLevel.memoryBytes : this.cpuTopLevel.size;
        const meshRefits = this.deforming?.memoryBytes ?? 0;
        const positions = this.positions?.byteLength ?? 0;
        let gpuBuildOutputs = 0;
        for (const outputs of this.liveOutputs) gpuBuildOutputs += outputs.bytes;
        const retired = this.nodes.retiredByteLength + this.triangles.retiredByteLength + (this.positions?.retiredByteLength ?? 0) + this.retiredTopLevel.reduce((sum, buffer) => sum + buffer.size, 0) + this.identity.pendingBytes;
        const resident = nodes + triangles + topLevel + positions + meshRefits;
        let gpuBuilding = 0;
        for (const builds of this.buildsRunning) gpuBuilding += builds.buildingBytes;
        const transient = gpuBuildOutputs + gpuBuilding + retired;
        const total = resident + transient;
        if (total > this.peakBytes) this.peakBytes = total;
        const reservedNodes = this.reserve.nodes.units * this.nodeBytes;
        const reservedTriangles = this.reserve.triangles.units * this.triangleBytes;
        return {
            nodes,
            nodesInUse: this.nodes.usedBytes - reservedNodes,
            triangles,
            trianglesInUse: this.triangles.usedBytes - reservedTriangles,
            topLevel,
            positions,
            meshRefits,
            gpuBuildOutputs,
            gpuBuilding,
            reserved: reservedNodes + reservedTriangles,
            retired,
            resident,
            transient,
            total,
            peak: this.peakBytes,
        };
    }

    private samplePeak(): void {
        void this.memory;
    }

    /**
     * Resolves once every mesh with instances has its first tree (or none is
     * coming) and no top-level shape is building. They reach the GPU at the
     * next {@link update}.
     *
     * **Builds start in {@link update}, not in `addMesh`.** A mesh added since
     * the last update counts as waiting, so awaiting this with no update to
     * come never resolves. Call `update` once after adding meshes (and keep
     * calling it once per frame), then await:
     *
     *     frame();                  // update(encoder) + submit: starts the builds
     *     await trees.whenBuilt();  // every mesh with instances has a tree
     *     frame();                  // places them
     *
     * After that one update it needs no more, with `gpuMaxBatchBytes` too (its
     * batches go on between updates). Only when a capped batch has to wait for
     * an update (see the file comment) do the meshes behind it count once
     * tinybvh's tree is in. A frame loop that is already running needs nothing
     * extra: its next update starts the builds.
     */
    async whenBuilt(): Promise<void> {
        if (this.meshesWaitingForFirstTree() > 0 || this.newMeshes.length > 0) await new Promise<void>((resolve) => this.builtWaiters.push(resolve));
        await this.gpuTopLevel?.whenIdle();
    }

    /** Resolves once every tinybvh build has finished (held trees still wait for {@link releaseRefinements}). Like {@link whenBuilt}, it needs an {@link update} after `addMesh` to start them. */
    async whenRefined(): Promise<void> {
        if (this.pendingTinybvh > 0 || this.newMeshes.length > 0) await new Promise<void>((resolve) => this.refinedWaiters.push(resolve));
        await this.whenBuilt();
    }

    /**
     * Lets tinybvh trees held by `holdRefinements` replace GPU trees from the
     * next update: all of them, or only the meshes `which` accepts (by mesh
     * handle; merged groups by `-1 - group`).
     */
    /** Turns `holdRefinements` on or off from now on (off releases every held tree at the next update). */
    setHoldRefinements(hold: boolean): void {
        this.refinementHold = hold ? () => true : null;
    }

    releaseRefinements(which?: (mesh: number) => boolean): void {
        const held = this.refinementHold;
        if (!held) return;
        if (!which) {
            this.refinementHold = null;
            return;
        }
        const key = (slot: number) => {
            const group = this.meshes.get(slot)?.group ?? -1;
            return group >= 0 ? -1 - group : slot;
        };
        this.refinementHold = (slot) => held(slot) && !which(key(slot));
    }

    // -----------------------------------------------------------------------
    // The frame
    // -----------------------------------------------------------------------

    /**
     * Brings the buffers up to date: starts builds for meshes added since the
     * last update, places trees that arrived (writes, and copies recorded in
     * `encoder`), and brings the top level up to date. Call once per frame
     * before anything traces; the caller submits `encoder`, **before the next
     * update** (the next update frees what this frame's commands still read,
     * and a capped batch stored between updates relies on it: see the file
     * comment).
     *
     * Throws in two ways:
     *
     * - **Before recording anything**: a tree that needs the node or triangle
     *   buffer past a device limit (both growths are checked before either
     *   happens; the trees stay queued, so the next update throws again), or a
     *   tinybvh build that failed (reported once). The encoder is untouched and
     *   the scene is as it was: the caller may drop the frame.
     * - **After recording the frame** (`RangeError`s of the top level: one too
     *   deep for the trace stack, or past a device limit). The frame's tree
     *   copies, refits and growth copies are in `encoder`, and the buffers they
     *   replace are destroyed at the next update: **submit the encoder anyway**,
     *   or the scene's buffers hold empty trees.
     */
    update(encoder: GPUCommandEncoder): void {
        this.checkLive();
        if (!this.prepared) throw new Error('@pmndrs/bvh: SceneTrees is not prepared; make it with SceneTrees.create');
        this.updates++;
        // The frame the last update recorded into was submitted: its writes into the storages are on the queue.
        this.encoderWritesStorage = false;
        for (const job of this.quantizeJobsToRelease.splice(0)) job.destroy();
        for (const tree of this.releaseAtUpdate.splice(0)) {
            this.nodes.release(tree.nodeOffset - this.nodeLeading, tree.nodeCount + this.nodeLeading);
            this.triangles.release(tree.triangleOffset - tree.triangleLeading, tree.triangleCount + tree.triangleLeading);
        }
        for (const range of this.releasePositionsAtUpdate.splice(0)) this.positions?.release(range.offset, range.count);
        // Refit stats copied in earlier updates: the drift rule reads them.
        this.deforming?.pollStats();
        this.nodes.releaseRetired();
        this.triangles.releaseRetired();
        this.positions?.releaseRetired();
        for (const buffer of this.retiredTopLevel.splice(0)) buffer.destroy();
        for (const buffer of this.retiredStaging.splice(0)) buffer.destroy();
        this.identity.releaseRetired();
        // The caller has submitted the frame that copied these GPU trees out (or dropped them).
        for (const outputs of this.outputsToRelease) this.releaseOutputs(outputs);
        this.outputsToRelease = [];

        if (this.updates === 1 && this.mergeStatic) this.planMerges();
        if (this.newMeshes.length > 0) {
            this.loadStartedAt ??= performance.now();
            this.startBuilds(encoder);
        }

        // Errors before anything is recorded: the frame's encoder is untouched, the scene as it was.
        if (this.buildFailure) {
            const failure = this.buildFailure;
            this.buildFailure = null;
            throw failure;
        }
        const arrived = this.takeArrived();
        if (arrived.length > 0) {
            this.placeTrees(encoder, arrived);
            this.cpuTopLevelStale = true;
        }
        // CPU poses, after the trees placed above (their copies and growths are in the encoder before these).
        this.writeCpuPoses(encoder);
        let failure: unknown = null;
        // Poses and trees placed above, refit before anything traces; then a drift rebuild if one is due.
        if (this.deforming?.encode(encoder, { nodes: this.nodes.buffer, triangles: this.triangles.buffer, positions: this.positions?.buffer ?? null }, this.updates)) this.encoderWritesStorage = true;
        this.startDriftRebuild();
        this.syncInstances();
        try {
            this.updateTopLevel(encoder);
        } catch (error) {
            failure ??= error;
        }
        this.noteTimeline();
        this.samplePeak();
        if (failure !== null) throw failure;
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        for (const mesh of this.meshes.values()) mesh.abort?.abort();
        if (this.ownsPool) this.pool.dispose();
        this.gpuTopLevel?.dispose();
        this.identity.dispose();
        this.deforming?.dispose();
        for (const job of this.quantizeJobsToRelease.splice(0)) job.destroy();
        this.nodes.destroy();
        this.triangles.destroy();
        this.positions?.destroy();
        this.cpuTopLevel.destroy();
        for (const buffer of this.retiredTopLevel.splice(0)) buffer.destroy();
        for (const buffer of this.retiredStaging.splice(0)) buffer.destroy();
        for (const outputs of this.liveOutputs) outputs.destroy();
        this.liveOutputs.clear();
        for (const resolve of this.builtWaiters.splice(0)) resolve();
        for (const resolve of this.refinedWaiters.splice(0)) resolve();
    }

    // -----------------------------------------------------------------------
    // Merging
    // -----------------------------------------------------------------------

    /** The first update: merges the static instances added so far into world-space group meshes. */
    private planMerges(): void {
        const meshSlots = [...this.meshes.values()].filter((mesh) => mesh.group < 0 && mesh.geometry !== null);
        const indexOf = new Map(meshSlots.map((mesh, index) => [mesh.slot, index]));
        const instances = [...this.instances.values()].filter((instance) => indexOf.has(instance.mesh));
        if (!instances.some((instance) => instance.static)) return;
        const plan = mergeStaticGeometry(
            meshSlots.map((mesh) => mesh.geometry!),
            instances.map((instance) => ({
                meshIndex: indexOf.get(instance.mesh)!,
                instanceIndex: instance.instanceIndex,
                objectToWorld: instance.objectToWorld,
                visible: instance.visible,
                contributesToGi: instance.contributesToGi,
                mask: instance.mask,
                material: instance.material ?? undefined,
                // A deforming mesh's instances never merge: a merged range is never refit.
                static: instance.static && !this.meshes.get(instance.mesh)!.deforming,
            })),
            { triangleLeaves: this.triangleLeaves },
        );
        if (plan.groups.length === 0) return;
        this.plan = plan;
        plan.groups.forEach((group, groupIndex) => {
            const meshSlot = this.meshSlots.take();
            this.meshes.set(meshSlot, this.newMeshSlot(meshSlot, groupIndex, group.geometry, group.sourceInstances.length));
            this.newMeshes.push(meshSlot);
            const instanceSlot = this.instanceSlots.take();
            this.groups.push({ group, meshSlot, instanceSlot });
            for (const source of group.sources) {
                const instance = instances[source.inputIndex];
                instance.merged = groupIndex;
                this.dirtyInstances.add(instance.slot);
            }
            this.dirtyInstances.add(instanceSlot);
        });
        plan.meshNeeded.forEach((needed, index) => {
            if (needed) return;
            const mesh = meshSlots[index];
            const group = plan.groups.findIndex((candidate) => candidate.sources.some((source) => source.meshIndex === index));
            mesh.mergedInto = group;
            mesh.geometry = null;
            this.newMeshes = this.newMeshes.filter((slot) => slot !== mesh.slot);
        });
    }

    // -----------------------------------------------------------------------
    // Builds
    // -----------------------------------------------------------------------

    /** Starts the GPU batch for the new meshes, then their tinybvh builds, biggest first. */
    private startBuilds(encoder: GPUCommandEncoder): void {
        const now = performance.now();
        const states = this.newMeshes.map((slot) => this.meshes.get(slot)!).filter((state) => state.geometry !== null);
        // Indexed: the positions go in first, while the geometry is still here (tinybvh's build may move it into a worker).
        // A growth past the device limit throws here with nothing changed and the meshes still new, so a later update retries them.
        if (this.positions) this.placePositions(states, encoder);
        this.newMeshes = [];
        const builder = this.firstTreesReady ? this.firstTrees : null;
        for (const state of states) {
            state.startedAt = now;
            state.gpu = builder ? 'building' : 'none';
            state.tinybvh = 'building';
            if (!builder) state.fallbackReason = this.noGpuReason ?? 'the GPU builder is not ready';
        }
        if (builder && states.length > 0) {
            const request = (state: MeshSlot): GpuTreeRequest => ({ slot: state.slot, generation: state.generation, geometry: state.geometry! });
            // Deforming meshes' trees are read back too (the refit needs their shape), so they get a batch of their own.
            // Split by the mesh's own flag, not by the GPU refit: without subgroups a CPU pose (setPoseCpu) needs the layout as much.
            const others = states.filter((state) => !state.deforming).map(request);
            const deforming = states.filter((state) => state.deforming).map(request);
            if (others.length > 0) this.startGpuBuilds(builder, others, { readLayouts: false, rebuild: false });
            if (deforming.length > 0) this.startGpuBuilds(builder, deforming, { readLayouts: true, rebuild: false });
        }
        // tinybvh after the GPU build has started: the GPU's writeBuffer has copied the arrays by now,
        // so moving them into a worker is safe (keep this order).
        const bySize = [...states].sort((a, b) => b.triangleCount - a.triangleCount);
        for (const state of bySize) this.startTinybvhBuild(state);
    }

    /** Starts GPU builds. `rebuild` marks a deforming mesh's drift rebuild: refused, the mesh keeps refitting its tree. Returns whether anything started. */
    private startGpuBuilds(builder: FirstTreeBuilder, requests: readonly GpuTreeRequest[], kind: { readonly readLayouts: boolean; readonly rebuild: boolean }): boolean {
        const refuse = (slot: number, generation: number, reason: string): void => {
            if (kind.rebuild) this.deforming?.rebuildRefused(slot);
            else this.gpuRefused(slot, generation, reason);
        };
        let builds: GpuTreeBuilds;
        try {
            builds = builder.build(requests, {
                readLayouts: kind.readLayouts,
                maxBatchBytes: this.maxBatchBytes,
                cancelled: () => this.disposed,
                onMemoryChange: () => this.samplePeak(),
            });
        } catch (error) {
            this.warn(`@pmndrs/bvh: the GPU mesh tree build failed, so ${kind.rebuild ? 'the mesh keeps its refit tree' : 'the meshes wait for tinybvh'}: ${describe(error)}`);
            for (const { slot, generation } of requests) refuse(slot, generation, `the GPU build failed: ${describe(error)}`);
            return false;
        }
        for (const { slot, generation, reason } of builds.notBuilt) refuse(slot, generation, reason);
        for (const slot of builds.readsGeometryLater ?? []) this.gpuReadsGeometryLater.add(slot);
        let left = builds.batches.length;
        if (left > 0) this.buildsRunning.add(builds);
        this.samplePeak();
        for (const batch of builds.batches) {
            void batch.then((result) => {
                if (--left === 0) {
                    this.samplePeak();
                    this.buildsRunning.delete(builds);
                }
                this.gpuBatchFinished(result, kind.rebuild, this.maxBatchBytes !== undefined);
            });
        }
        return builds.batches.length > 0;
    }

    /** Starts the drift rebuild of the deforming mesh whose refit has drifted longest, from its latest pose. */
    private startDriftRebuild(): void {
        const deforming = this.deforming;
        const builder = this.firstTreesReady ? this.firstTrees : null;
        if (!deforming || !builder) return;
        const slot = deforming.rebuildDue(this.updates);
        if (slot === null) return;
        const state = this.meshes.get(slot);
        if (!state || state.gpu === 'building' || state.tinybvh === 'building') return;
        const geometry = deforming.rebuildGeometry(slot);
        if (!geometry) return;
        const started = this.startGpuBuilds(builder, [{ slot, generation: state.generation, geometry }], { readLayouts: true, rebuild: true });
        deforming.rebuildStarted(slot, started);
    }

    private gpuBatchFinished(result: GpuTreeBatchResult, rebuild: boolean, capped: boolean): void {
        this.lastGpuBuildMs = result.buildMs;
        if (result.failure === null) {
            this.gpuBatches++;
            this.gpuBuildMs += result.buildMs;
        }
        const batchOutputs = new Set(result.built.map((built) => built.outputs));
        for (const outputs of batchOutputs) {
            // Counted here from now on (the builder counted them until this claim).
            outputs.claim();
            if (this.disposed) outputs.destroy();
            else this.liveOutputs.add(outputs);
        }
        if (this.disposed) return;
        const refuse = (slot: number, generation: number, reason: string): void => {
            if (rebuild) this.deforming?.rebuildRefused(slot);
            else this.gpuRefused(slot, generation, reason);
        };
        for (const { slot, generation, reason } of result.refused) refuse(slot, generation, reason);
        if (result.failure) this.warn(`@pmndrs/bvh: ${result.failure}, so ${rebuild ? 'the mesh keeps its refit tree' : 'the meshes wait for tinybvh'}`);
        for (const built of result.built) {
            if (built.outputs.nodeFormat !== 'f32') {
                // The scene quantizes f32 trees as it copies them in; already-quantized ones can't be copied.
                const reason = 'the first-tree builder returned quantized nodes; SceneTrees needs f32 GPU trees (it quantizes them itself)';
                if (this.currentState(built.slot, built.generation)) {
                    this.warn(`@pmndrs/bvh: ${reason}`);
                    refuse(built.slot, built.generation, reason);
                }
                this.releaseOutputs(built.outputs);
                continue;
            }
            const state = this.currentState(built.slot, built.generation);
            if (!state || built.tree.maxDepth > this.gpuMaxDepth) {
                if (state) refuse(built.slot, built.generation, `the GPU tree is ${built.tree.maxDepth} levels deep, more than gpuMaxDepth (${this.gpuMaxDepth})`);
                // Never copied: let go at once (a capped build's next batch waits for it).
                this.releaseOutputs(built.outputs);
                continue;
            }
            // A rebuild's mesh already had its first tree: its state stays as the first builds left it.
            if (!rebuild) state.gpu = 'built';
            this.gpuTreesBuilt++;
            this.arrived.push({ source: 'gpu', slot: built.slot, generation: built.generation, built, rebuild });
        }
        if (capped) this.storeBetweenUpdates(batchOutputs);
        this.reconcileReserve();
        this.samplePeak();
        this.notifyWaiters();
    }

    private gpuRefused(slot: number, generation: number, reason: string): void {
        const state = this.currentState(slot, generation);
        if (!state) return;
        state.gpu = 'refused';
        state.fallbackReason = reason;
        this.gpuTreesRefused++;
        this.reconcileReserve();
    }

    private startTinybvhBuild(state: MeshSlot): void {
        const geometry = state.geometry!;
        const generation = state.generation;
        state.geometry = null;
        state.abort = new AbortController();
        this.pendingTinybvh++;
        // A capped GPU build that hasn't uploaded this mesh yet still reads its arrays: copy them into the worker instead.
        // Deleted first, whatever the rest says: an entry left behind would outlive the build.
        const readsLater = this.gpuReadsGeometryLater.delete(state.slot);
        const transferInput = (state.group >= 0 || this.transferGeometry) && !readsLater;
        const finished = (bvh: MeshBvh | null): void => {
            this.pendingTinybvh--;
            const current = this.disposed ? undefined : this.currentState(state.slot, generation);
            if (current) {
                current.abort = null;
                if (bvh && bvh.maxDepth > this.gpuMaxDepth) {
                    // Refused like a too-deep GPU tree: the mesh keeps the tree it has (or gets none), and the trace stack stays safe.
                    const reason = `tinybvh's tree is ${bvh.maxDepth} levels deep, more than gpuMaxDepth (${this.gpuMaxDepth})`;
                    current.fallbackReason = current.fallbackReason ? `${current.fallbackReason}; ${reason}` : reason;
                    this.warn(`@pmndrs/bvh: ${reason}, so it is not used`);
                    bvh = null;
                }
                current.tinybvh = bvh ? 'built' : 'failed';
                if (bvh) this.arrived.push({ source: 'tinybvh', slot: state.slot, generation, bvh });
            }
            this.notifyWaiters();
        };
        this.pool.build(geometry, this.tinybvhOptions(state.slot), { signal: state.abort.signal, transferInput }).then(finished, (error: unknown) => {
            const cancelled = error instanceof Error && error.name === 'AbortError';
            if (!this.disposed && !cancelled) this.buildFailure = new Error(`@pmndrs/bvh: building a mesh tree with tinybvh failed: ${describe(error)}`);
            finished(null);
        });
    }

    private currentState(slot: number, generation: number): MeshSlot | undefined {
        const state = this.meshes.get(slot);
        return state && state.generation === generation ? state : undefined;
    }

    private holdsRefinement(slot: number): boolean {
        return this.refinementHold !== null && this.refinementHold(slot);
    }

    /** Meshes (with instances, or merged groups) whose current version has no tree yet and still has one coming. */
    private meshesWaitingForFirstTree(): number {
        let waiting = 0;
        const adoptionPending = this.gpuTreeOutputs === 'adopt' && this.nothingStored();
        const loadBatchBuilding = adoptionPending && this.loadBatchBuilding();
        for (const state of this.meshes.values()) {
            if (state.mergedInto !== null || (state.group < 0 && state.users === 0)) continue;
            if (this.placed.has(state.slot)) continue;
            // An arrived tree counts (it is placed at the next update), except a tinybvh tree that waits: the same rule as takeArrived's.
            const waitsForGpu = this.tinybvhTreeWaits(state, state.gpu === 'building', adoptionPending, loadBatchBuilding);
            const gpuWaits = loadBatchBuilding && this.deformingSlot(state.slot);
            const arrived = this.arrived.some((tree) => tree.slot === state.slot && tree.generation === state.generation && (tree.source === 'gpu' ? !gpuWaits : !waitsForGpu));
            if (arrived) continue;
            if (state.gpu === 'building' || state.tinybvh === 'building') waiting++;
        }
        return waiting;
    }

    /**
     * Whether a mesh's arrived tinybvh tree waits rather than being placed at
     * the next update: held while the GPU's tree comes first
     * (`holdRefinements`), or while the buffers wait for an adoption. One rule
     * for `takeArrived` (what an update places) and `whenBuilt` (what counts
     * as arrived), so they never disagree. Except: while a capped batch waits
     * for an update (`outputsWaitingForUpdate`), a mesh with no tree whose GPU
     * build hasn't arrived takes tinybvh's tree rather than wait behind it.
     */
    private tinybvhTreeWaits(state: MeshSlot, gpuTreeFirst: boolean, waitForAdoption: boolean, loadBatchBuilding: boolean): boolean {
        if (this.outputsWaitingForUpdate.size > 0 && state.gpu === 'building' && !this.placed.has(state.slot)) return false;
        return (gpuTreeFirst && this.holdsRefinement(state.slot)) || (waitForAdoption && (state.gpu === 'building' || loadBatchBuilding));
    }

    /** The mesh was added as deforming (with or without the GPU refit: its first trees come in a batch of their own, with layouts). */
    private deformingSlot(slot: number): boolean {
        return this.meshes.get(slot)?.deforming ?? false;
    }

    /** No tree is placed, and none stored for the next update: the storages are empty (what adoption needs). */
    private nothingStored(): boolean {
        return this.placed.size === 0 && !this.arrived.some((tree) => tree.source === 'gpu' && tree.stored !== undefined);
    }

    /** A first-tree GPU build of a mesh that doesn't deform (the main load batch, for adoption) is still running. */
    private loadBatchBuilding(): boolean {
        for (const state of this.meshes.values()) if (state.gpu === 'building' && !state.deforming) return true;
        return false;
    }

    private notifyWaiters(): void {
        if (this.meshesWaitingForFirstTree() === 0 && this.newMeshes.length === 0) for (const resolve of this.builtWaiters.splice(0)) resolve();
        if (this.pendingTinybvh === 0 && this.newMeshes.length === 0) for (const resolve of this.refinedWaiters.splice(0)) resolve();
    }

    /**
     * The finished builds to place this update, for meshes still in the version
     * they were built from. A GPU tree whose mesh already has tinybvh's is
     * dropped: it would only be worse. Held tinybvh trees (`holdRefinements`)
     * wait while the mesh's GPU build runs or would replace its GPU tree. With
     * `gpuTreeOutputs: 'adopt'`, tinybvh trees also wait while nothing is placed
     * and the mesh's GPU build (or the main load batch) is running, and so do a
     * deforming mesh's GPU trees while the main load batch runs, so the load
     * batch finds the buffers empty.
     */
    private takeArrived(): ArrivedTree[] {
        const taken: ArrivedTree[] = [];
        const waiting: ArrivedTree[] = [];
        const tinybvhNow = new Set<number>();
        const gpuQueued = new Set(this.arrived.filter((tree) => tree.source === 'gpu' && !tree.rebuild && this.currentState(tree.slot, tree.generation)).map((tree) => tree.slot));
        const waitForAdoption = this.gpuTreeOutputs === 'adopt' && this.nothingStored();
        const loadBatchBuilding = waitForAdoption && this.loadBatchBuilding();
        for (const tree of this.arrived) {
            const state = this.currentState(tree.slot, tree.generation);
            if (!state || tree.source !== 'tinybvh') continue;
            const placed = this.placed.get(tree.slot);
            const gpuTreeFirst = state.gpu === 'building' || (placed?.source === 'gpu' && placed.generation === tree.generation) || gpuQueued.has(tree.slot);
            if (this.tinybvhTreeWaits(state, gpuTreeFirst, waitForAdoption, loadBatchBuilding)) {
                waiting.push(tree);
            } else {
                taken.push(tree);
                tinybvhNow.add(tree.slot);
            }
        }
        for (const tree of this.arrived) {
            if (tree.source !== 'gpu') continue;
            if (loadBatchBuilding && !tree.rebuild && this.deformingSlot(tree.slot) && this.currentState(tree.slot, tree.generation)) {
                // A deforming mesh's own batch waits for the main load batch, which is the one to adopt.
                waiting.push(tree);
                continue;
            }
            const placed = this.placed.get(tree.slot);
            const hasTinybvhTree = tinybvhNow.has(tree.slot) || (placed?.source === 'tinybvh' && placed.generation === tree.generation);
            const state = this.currentState(tree.slot, tree.generation);
            // A drift rebuild is newer than any tree its mesh has (built from a later pose), tinybvh's too.
            if (state && (tree.rebuild || !hasTinybvhTree)) {
                taken.push(tree);
            } else {
                if (state && hasTinybvhTree && state.fallbackReason === null) state.fallbackReason = "tinybvh's tree arrived before the GPU's";
                if (tree.rebuild) this.deforming?.rebuildRefused(tree.slot);
                if (tree.stored) {
                    // Stored between updates, never placed: its ranges hold nothing anyone traces.
                    this.nodes.release(tree.stored.nodeOffset - this.nodeLeading, tree.built.tree.nodeCount + this.nodeLeading);
                    this.triangles.release(tree.stored.triangleOffset - tree.stored.triangleLeading, tree.built.tree.triangleCount + tree.stored.triangleLeading);
                } else {
                    this.outputsToRelease.push(tree.built.outputs);
                }
            }
        }
        // A mesh whose first tree is tinybvh's, with the GPU tree still building: say why.
        for (const tree of taken) {
            if (tree.source !== 'tinybvh') continue;
            const state = this.meshes.get(tree.slot)!;
            if (!this.placed.has(tree.slot) && state.gpu === 'building' && state.fallbackReason === null) state.fallbackReason = "tinybvh's tree arrived before the GPU's";
        }
        this.arrived = waiting;
        return taken;
    }

    private releaseTree(slot: number): void {
        const tree = this.placed.get(slot);
        if (!tree) return;
        this.nodes.release(tree.nodeOffset - this.nodeLeading, tree.nodeCount + this.nodeLeading);
        this.triangles.release(tree.triangleOffset - tree.triangleLeading, tree.triangleCount + tree.triangleLeading);
        this.forgetTree(slot);
    }

    /** Everything of {@link releaseTree} but freeing the ranges. */
    private forgetTree(slot: number): void {
        this.placed.delete(slot);
        this.gpuTopLevel?.setMeshTree(slot, null);
        this.deforming?.treeReleased(slot);
    }

    private releaseOutputs(outputs: GpuTreeOutputs): void {
        outputs.release();
        if (outputs.isDestroyed) this.liveOutputs.delete(outputs);
    }

    /**
     * Places arrived trees: tinybvh's written from the CPU, GPU ones copied out
     * of their batch's buffers in `encoder` after any growth copies, or (adopt)
     * served in place; trees a capped batch stored between updates only get
     * their placement. Merged groups get their identity written after.
     *
     * All or nothing: the ranges of every tree are planned first, both storages'
     * growths checked against the device's limits, and only then is anything
     * grown, written or recorded. On a failure it throws with the encoder
     * untouched and the scene as it was (the trees stay queued for the next update).
     */
    private placeTrees(encoder: GPUCommandEncoder, arrived: readonly ArrivedTree[]): void {
        const replaced = new Map<number, TreeBuilder>();
        for (const tree of arrived) {
            const old = this.placed.get(tree.slot);
            if (old) replaced.set(tree.slot, old.source);
        }

        if (this.gpuTreeOutputs === 'adopt') {
            const batch = this.adoptableBatch(arrived);
            if (batch) {
                this.adoptLoadBatch(encoder, batch, replaced);
                // The others are placed at the next update, into the adopted buffers: nothing can fail after an adoption.
                const adopted = new Set<ArrivedTree>(batch);
                this.arrived.push(...arrived.filter((tree) => !adopted.has(tree)));
                return;
            }
        }

        // The room of meshes getting a tree now is not needed any more: give it back first, so these trees can use it.
        this.reconcileReserve(new Set(arrived.map((tree) => tree.slot)));
        const sizes = arrived.map(treeSize);
        // Trees stored between updates are in the buffers already; the others need ranges.
        const fresh = arrived.flatMap((tree, index) => (tree.source === 'gpu' && tree.stored ? [] : [index]));
        // Quantized: a tree's node range starts with its frame slot.
        const nodeCounts = fresh.map((index) => sizes[index].nodeCount + this.nodeLeading);
        // Indexed: a tree's range starts with its leading units (header, side tables).
        const leadings = fresh.map((index) => this.leadingOf(arrived[index].slot, sizes[index].triangleCount));
        const triangleCounts = fresh.map((index, at) => sizes[index].triangleCount + leadings[at]);
        const fromReserve = fresh.map((index) => arrived[index].source === 'gpu');
        // The replaced trees' ranges are freed in the allocators first, so a new tree can take its old range; undone on a failure.
        const oldSlots = [...replaced.keys()];
        const oldTrees = oldSlots.map((slot) => this.placed.get(slot)!);
        for (const old of oldTrees) {
            this.nodes.release(old.nodeOffset - this.nodeLeading, old.nodeCount + this.nodeLeading);
            this.triangles.release(old.triangleOffset - old.triangleLeading, old.triangleCount + old.triangleLeading);
        }
        let nodeRanges: TakenRanges | null;
        let triangleRanges: TakenRanges | null;
        try {
            nodeRanges = takeRanges(this.nodes, this.reserve.nodes, nodeCounts, fromReserve);
            triangleRanges = takeRanges(this.triangles, this.reserve.triangles, triangleCounts, fromReserve);
            try {
                // Both checks before either growth: they succeed or fail together.
                if (nodeRanges === null) this.nodes.checkGrowth(sum(nodeCounts));
                if (triangleRanges === null) this.triangles.checkGrowth(sum(triangleCounts));
            } catch (error) {
                nodeRanges?.undo();
                triangleRanges?.undo();
                throw error;
            }
        } catch (error) {
            for (const old of oldTrees) {
                this.nodes.reserve(old.nodeOffset - this.nodeLeading, old.nodeCount + this.nodeLeading);
                this.triangles.reserve(old.triangleOffset - old.triangleLeading, old.triangleCount + old.triangleLeading);
            }
            this.arrived.push(...arrived);
            throw error;
        }

        // From here on nothing throws: forget the replaced trees, grow, write and copy.
        for (const slot of oldSlots) this.forgetTree(slot);
        const gpuSizes = fresh.filter((_, at) => fromReserve[at]).map((index) => sizes[index]);
        const growth = this.grow(encoder, nodeRanges === null ? nodeCounts : null, triangleRanges === null ? triangleCounts : null, gpuSizes, arrived);
        const nodeOffsets = nodeRanges?.offsets ?? growth.nodeOffsets!;
        const triangleOffsets = triangleRanges?.offsets ?? growth.triangleOffsets!;
        if (growth.grew) this.encoderWritesStorage = true;

        const freshAt = new Map(fresh.map((index, at) => [index, at]));
        // Capped: a GPU tree that waited for this update is copied in a submit of its own (to the fresh ranges, which
        // nothing in `encoder` touches), so its outputs go now and the queue behind it starts this frame.
        const ownCopies = this.maxBatchBytes !== undefined ? this.device.createCommandEncoder({ label: `${this.label} waiting batch copy` }) : null;
        const copiedNow: GpuTreeOutputs[] = [];
        arrived.forEach((tree, index) => {
            const size = sizes[index];
            if (tree.source === 'gpu' && tree.stored) {
                this.recordPlacement(tree, size, tree.stored.nodeOffset, tree.stored.triangleOffset, replaced.get(tree.slot) ?? null, tree.stored.triangleLeading);
                return;
            }
            const at = freshAt.get(index)!;
            const nodeOffset = nodeOffsets[at] + this.nodeLeading;
            const triangleOffset = triangleOffsets[at] + leadings[at];
            const state = this.meshes.get(tree.slot)!;
            if (tree.source === 'tinybvh') {
                this.writeTinybvhNodes(tree.bvh, nodeOffset);
                if (this.positions) {
                    this.writeIndexedTree(state, tree.bvh, triangleOffset, leadings[at]);
                } else {
                    if (state.group >= 0) writeMergedTriangleIdentity(tree.bvh.triangles, this.groups[state.group].group, 0, tree.bvh.triangleCount);
                    this.triangles.write(triangleOffset, tree.bvh.triangles, size.triangleCount * TRIANGLE_BYTES);
                }
            } else if (ownCopies) {
                this.copyGpuTree(ownCopies, tree, size, nodeOffset, triangleOffset);
                copiedNow.push(tree.built.outputs);
            } else {
                this.copyGpuTree(encoder, tree, size, nodeOffset, triangleOffset);
                // Let go of at the next update, once the caller has submitted these copies.
                this.outputsToRelease.push(tree.built.outputs);
                this.encoderWritesStorage = true;
            }
            // The tree's own sizes, at the scene's offsets (a GPU tree's range also carries its offsets in the batch's buffers).
            this.recordPlacement(tree, size, nodeOffset, triangleOffset, replaced.get(tree.slot) ?? null, leadings[at]);
        });
        if (ownCopies && copiedNow.length > 0) {
            this.device.queue.submit([ownCopies.finish()]);
            for (const outputs of copiedNow) this.releaseOutputs(outputs);
        }
        this.reconcileReserve();
    }

    /** Records a GPU tree's copy out of its batch's buffers into the scene's, and a merged group's identity pass after it. */
    private copyGpuTree(encoder: GPUCommandEncoder, tree: GpuArrivedTree, size: TreeSize, nodeOffset: number, triangleOffset: number): void {
        const { outputs, tree: built } = tree.built;
        if (this.nodeFormat === 'quantized') {
            // The batch's nodes are f32: packed into the frame slot and the nodes on the way in.
            const job = nodeQuantizerFor(this.device).prepare(outputs.nodes, this.nodes.buffer, [{ sourceNodeOffset: built.nodeOffset, targetRootSlot: nodeOffset, nodeCount: size.nodeCount }], `${this.label} quantize`);
            job.encode(encoder);
            this.quantizeJobsToRelease.push(job);
        } else {
            encoder.copyBufferToBuffer(outputs.nodes, built.nodeOffset * BVH_NODE_BYTES, this.nodes.buffer, nodeOffset * BVH_NODE_BYTES, size.nodeCount * BVH_NODE_BYTES);
        }
        const bytes = this.triangleBytes;
        encoder.copyBufferToBuffer(outputs.triangles, built.triangleOffset * bytes, this.triangles.buffer, triangleOffset * bytes, size.triangleCount * bytes);
        const state = this.meshes.get(tree.slot)!;
        const vertexBase = state.vertices?.offset ?? 0;
        if (state.group >= 0) this.identity.encode(encoder, this.triangles.buffer, this.groups[state.group].group, triangleOffset, size.triangleCount, this.positions ? { vertexBase } : undefined);
        else if (this.positions) {
            const leading = this.leadingOf(tree.slot, size.triangleCount);
            const words = new Uint32Array(leading * INDEXED_TRIANGLE_WORDS);
            writeIndexedMeshHeader(words, leading, vertexBase, state.materialCodes);
            this.triangles.write(triangleOffset - leading, words);
        }
    }

    /**
     * Grows the storages whose counts are given (checked to fit before), with
     * room for every count after the high-water mark, and with a capped load's
     * reserve for the GPU trees still to come; returns the counts' ranges.
     * The growth copies every range that holds a tree: placed, and stored
     * between updates.
     */
    private grow(
        encoder: GPUCommandEncoder,
        nodeCounts: readonly number[] | null,
        triangleCounts: readonly number[] | null,
        gpuSizes: readonly TreeSize[],
        arriving: readonly ArrivedTree[],
    ): { nodeOffsets: number[] | null; triangleOffsets: number[] | null; grew: boolean } {
        if (!nodeCounts && !triangleCounts) return { nodeOffsets: null, triangleOffsets: null, grew: false };
        const gpuNodes = sum(gpuSizes.map((size) => size.nodeCount + this.nodeLeading));
        const gpuTriangles = sum(gpuSizes.map((size) => size.triangleCount));
        if (gpuTriangles > 0) this.reserveNodesPerTriangle = gpuNodes / gpuTriangles;
        // A capped load's later batches are still coming: grow once for them too, not by doubling batch after batch.
        const target = this.reserveTarget(new Set(arriving.map((tree) => tree.slot)));
        const residentNodes = [...this.placed.values()].map((tree) => ({ offset: tree.nodeOffset - this.nodeLeading, count: tree.nodeCount + this.nodeLeading }));
        const residentTriangles = [...this.placed.values()].map((tree) => ({ offset: tree.triangleOffset - tree.triangleLeading, count: tree.triangleCount + tree.triangleLeading }));
        for (const tree of [...arriving, ...this.arrived]) {
            if (tree.source !== 'gpu' || !tree.stored) continue;
            residentNodes.push({ offset: tree.stored.nodeOffset - this.nodeLeading, count: tree.built.tree.nodeCount + this.nodeLeading });
            residentTriangles.push({ offset: tree.stored.triangleOffset - tree.stored.triangleLeading, count: tree.built.tree.triangleCount + tree.stored.triangleLeading });
        }
        let nodeOffsets: number[] | null = null;
        let triangleOffsets: number[] | null = null;
        if (nodeCounts) {
            const headroom = gpuSizes.length > 0 ? GPU_TREE_NODE_HEADROOM : 0;
            const extra = sum(nodeCounts);
            const reserve = fitReserve(this.nodes, extra, Math.max(0, target.nodes - this.reserve.nodes.units));
            this.nodes.grow(encoder, extra + reserve, residentNodes, headroom);
            nodeOffsets = allocateGrown(this.nodes, nodeCounts);
            this.reserve.nodes.add(this.nodes, reserve);
        }
        if (triangleCounts) {
            const extra = sum(triangleCounts);
            const reserve = fitReserve(this.triangles, extra, Math.max(0, target.triangles - this.reserve.triangles.units));
            this.triangles.grow(encoder, extra + reserve, residentTriangles);
            triangleOffsets = allocateGrown(this.triangles, triangleCounts);
            this.reserve.triangles.add(this.triangles, reserve);
        }
        return { nodeOffsets, triangleOffsets, grew: true };
    }

    /**
     * With `gpuMaxBatchBytes`, the room the meshes whose GPU build is still to
     * come need (not `arriving`, which get a tree now): their triangles
     * exactly, their nodes at the GPU trees' last nodes per triangle. A mesh
     * removed, refused, or with a tree already (tinybvh's came first) needs
     * none. Without a cap, none: today's growth.
     */
    private reserveTarget(arriving: ReadonlySet<number> = new Set()): { nodes: number; triangles: number } {
        if (this.maxBatchBytes === undefined) return { nodes: 0, triangles: 0 };
        let triangles = 0;
        let meshes = 0;
        for (const state of this.meshes.values()) {
            if (state.gpu === 'building' && !this.placed.has(state.slot) && !arriving.has(state.slot)) {
                triangles += state.triangleCount + this.leadingOf(state.slot, state.triangleCount);
                meshes++;
            }
        }
        // Quantized: each tree to come takes a frame slot too (nodes per triangle is measured on whole ranges, so this is a little over).
        return { nodes: Math.ceil(triangles * this.reserveNodesPerTriangle) + this.nodeLeading * meshes, triangles };
    }

    /** Gives the reserve's room that no mesh still needs back to the storages (it holds nothing, so at any time). */
    private reconcileReserve(arriving?: ReadonlySet<number>): void {
        if (this.maxBatchBytes === undefined) return;
        const target = this.reserveTarget(arriving);
        this.reserve.nodes.shrink(this.nodes, target.nodes);
        this.reserve.triangles.shrink(this.triangles, target.triangles);
    }

    /**
     * A capped batch's trees, stored between updates (the "no update needed"
     * half of `gpuMaxBatchBytes`, see the file comment): copied into the
     * scene's buffers in a submit of their own, or (adopt, buffers empty) the
     * batch's buffers adopted, so its outputs are let go now and the next
     * batch starts. The next update places them. A tree whose mesh is gone or
     * has its tree already is dropped now; a deforming mesh's tree that waits
     * for the load batch (adopt) stays for the update, and so does every tree
     * when there is no room and growing now isn't safe.
     */
    private storeBetweenUpdates(batchOutputs: ReadonlySet<GpuTreeOutputs>): void {
        if (this.disposed) return;
        const loadBatchBuilding = this.gpuTreeOutputs === 'adopt' && this.nothingStored() && this.loadBatchBuilding();
        const candidates: GpuArrivedTree[] = [];
        const dropped = new Set<ArrivedTree>();
        for (const tree of this.arrived) {
            if (tree.source !== 'gpu' || tree.stored || !batchOutputs.has(tree.built.outputs)) continue;
            const state = this.currentState(tree.slot, tree.generation);
            const placed = this.placed.get(tree.slot);
            if (!state || (!tree.rebuild && placed?.generation === tree.generation)) {
                // What the next update's takeArrived would do: the mesh is gone, or tinybvh's tree came first.
                if (state && state.fallbackReason === null) state.fallbackReason = "tinybvh's tree arrived before the GPU's";
                if (tree.rebuild) this.deforming?.rebuildRefused(tree.slot);
                dropped.add(tree);
                this.releaseOutputs(tree.built.outputs);
                continue;
            }
            if (loadBatchBuilding && !tree.rebuild && this.deformingSlot(tree.slot)) continue;
            candidates.push(tree);
        }
        let stored = new Map<GpuArrivedTree, StoredAt>();
        if (candidates.length > 0) stored = this.adoptBetweenUpdates(candidates) ?? this.copyBetweenUpdates(candidates) ?? stored;
        if (dropped.size > 0 || stored.size > 0) {
            this.arrived = this.arrived.flatMap((tree): ArrivedTree[] => {
                if (dropped.has(tree)) return [];
                const at = tree.source === 'gpu' ? stored.get(tree) : undefined;
                return at ? [{ ...(tree as GpuArrivedTree), stored: at }] : [tree];
            });
        }
        // Outputs still holding a tree for the update: the queue behind them waits for it.
        for (const outputs of batchOutputs) {
            if (outputs.isDestroyed || outputs.isAdopted) continue;
            if (!this.arrived.some((tree) => tree.source === 'gpu' && !tree.stored && tree.built.outputs === outputs)) continue;
            this.outputsWaitingForUpdate.add(outputs);
            void outputs.letGo.then(() => {
                this.outputsWaitingForUpdate.delete(outputs);
                this.notifyWaiters();
            });
        }
    }

    /** Adopts a capped load's first batch between updates: only while the buffers hold nothing, and only all of the batch's usable trees. */
    private adoptBetweenUpdates(trees: readonly GpuArrivedTree[]): Map<GpuArrivedTree, StoredAt> | null {
        if (this.gpuTreeOutputs !== 'adopt' || !this.storagesEmpty() || trees.some((tree) => tree.rebuild)) return null;
        const outputs = trees[0].built.outputs;
        const usable = outputs.meshes.filter((mesh) => mesh.problem === null).length;
        if (outputs.isDestroyed || trees.length !== usable || trees.some((tree) => tree.built.outputs !== outputs)) return null;
        this.nodes.adopt(outputs.nodes, trees.map((tree) => ({ offset: tree.built.tree.nodeOffset, count: tree.built.tree.nodeCount })));
        this.triangles.adopt(outputs.triangles, trees.map((tree) => ({ offset: tree.built.tree.triangleOffset, count: tree.built.tree.triangleCount })));
        outputs.markAdopted();
        this.liveOutputs.delete(outputs);
        this.versionBase++;
        const merged = trees.filter((tree) => this.meshes.get(tree.slot)!.group >= 0);
        if (merged.length > 0) {
            const encoder = this.device.createCommandEncoder({ label: `${this.label} adopted batch identity` });
            for (const tree of merged) {
                const built = tree.built.tree;
                this.identity.encode(encoder, this.triangles.buffer, this.groups[this.meshes.get(tree.slot)!.group].group, built.triangleOffset, built.triangleCount);
            }
            this.device.queue.submit([encoder.finish()]);
        }
        return new Map(trees.map((tree) => [tree, { nodeOffset: tree.built.tree.nodeOffset, triangleOffset: tree.built.tree.triangleOffset, triangleLeading: 0 }]));
    }

    /**
     * Copies trees into the scene's buffers between updates, in a submit of
     * their own: into the reserve or free room, or after a growth when growing
     * now is safe (see the file comment). Null, with nothing changed, when
     * they must wait for an update.
     */
    private copyBetweenUpdates(trees: readonly GpuArrivedTree[]): Map<GpuArrivedTree, StoredAt> | null {
        const sizes = trees.map(treeSize);
        const nodeCounts = sizes.map((size) => size.nodeCount + this.nodeLeading);
        const leadings = trees.map((tree, index) => this.leadingOf(tree.slot, sizes[index].triangleCount));
        const triangleCounts = sizes.map((size, index) => size.triangleCount + leadings[index]);
        const fromReserve = trees.map(() => true);
        const nodeRanges = takeRanges(this.nodes, this.reserve.nodes, nodeCounts, fromReserve);
        const triangleRanges = takeRanges(this.triangles, this.reserve.triangles, triangleCounts, fromReserve);
        if (nodeRanges === null || triangleRanges === null) {
            // Growing now copies the buffers in a submit that runs before the last update's encoder: only safe when that encoder writes
            // nothing into them, or when they hold nothing (no range to copy, and nothing placed for a refit or a copy to write).
            let canGrow = !this.encoderWritesStorage || this.storagesEmpty();
            try {
                if (canGrow && nodeRanges === null) this.nodes.checkGrowth(sum(nodeCounts));
                if (canGrow && triangleRanges === null) this.triangles.checkGrowth(sum(triangleCounts));
            } catch {
                canGrow = false;
            }
            if (!canGrow) {
                nodeRanges?.undo();
                triangleRanges?.undo();
                return null;
            }
        }
        const encoder = this.device.createCommandEncoder({ label: `${this.label} capped batch copy` });
        const growth = this.grow(encoder, nodeRanges === null ? nodeCounts : null, triangleRanges === null ? triangleCounts : null, sizes, trees);
        const nodeOffsets = nodeRanges?.offsets ?? growth.nodeOffsets!;
        const triangleOffsets = triangleRanges?.offsets ?? growth.triangleOffsets!;
        trees.forEach((tree, index) => this.copyGpuTree(encoder, tree, sizes[index], nodeOffsets[index] + this.nodeLeading, triangleOffsets[index] + leadings[index]));
        this.device.queue.submit([encoder.finish()]);
        // The copies are submitted: the outputs can go (a buffer destroyed after a submit stays alive for it).
        for (const tree of trees) this.releaseOutputs(tree.built.outputs);
        return new Map(trees.map((tree, index) => [tree, { nodeOffset: nodeOffsets[index] + this.nodeLeading, triangleOffset: triangleOffsets[index] + leadings[index], triangleLeading: leadings[index] }]));
    }

    /** Both storages hold nothing: no tree placed or stored, no reserve, no range waiting to be freed. */
    private storagesEmpty(): boolean {
        return this.nothingStored() && this.nodes.usedBytes === 0 && this.triangles.usedBytes === 0;
    }

    /**
     * The batch whose arrived GPU trees (all of its usable trees) can become the
     * storage: every tree placed now is one of them (so the storages are empty
     * once they are released). The main load batch comes before a deforming
     * mesh's own batch (`takeArrived` holds those back while it builds), so
     * the peak-memory saving goes to the big one. Null: everything is copied.
     */
    private adoptableBatch(arrived: readonly ArrivedTree[]): GpuArrivedTree[] | null {
        // A stalled capped batch's update also places tinybvh trees (tinybvhTreeWaits): copy, so they all land in this update (an adoption defers them by one).
        if (this.outputsWaitingForUpdate.size > 0 && arrived.some((tree) => tree.source === 'tinybvh')) return null;
        // Trees stored between updates (or waiting to be freed) hold ranges: the buffers can't be swapped then.
        if (arrived.some((tree) => tree.source === 'gpu' && tree.stored) || this.arrived.some((tree) => tree.source === 'gpu' && tree.stored) || this.reserve.nodes.units > 0 || this.reserve.triangles.units > 0) return null;
        const placedUnits = { nodes: 0, triangles: 0 };
        for (const tree of this.placed.values()) {
            placedUnits.nodes += tree.nodeCount;
            placedUnits.triangles += tree.triangleCount;
        }
        if (this.nodes.usedBytes !== placedUnits.nodes * this.nodeBytes || this.triangles.usedBytes !== placedUnits.triangles * this.triangleBytes) return null;
        const byOutputs = new Map<GpuTreeOutputs, GpuArrivedTree[]>();
        for (const tree of arrived) {
            if (tree.source !== 'gpu') continue;
            const list = byOutputs.get(tree.built.outputs) ?? [];
            list.push(tree);
            byOutputs.set(tree.built.outputs, list);
        }
        let best: { trees: GpuArrivedTree[]; deforming: boolean; triangles: number } | null = null;
        for (const [outputs, trees] of byOutputs) {
            const usable = outputs.meshes.filter((mesh) => mesh.problem === null).length;
            if (trees.length !== usable || outputs.isDestroyed) continue;
            const slots = new Set(trees.map((tree) => tree.slot));
            if ([...this.placed.keys()].some((slot) => !slots.has(slot))) continue;
            const deforming = trees.some((tree) => this.deformingSlot(tree.slot));
            const triangles = trees.reduce((total, tree) => total + tree.built.tree.triangleCount, 0);
            if (!best || (best.deforming && !deforming) || (best.deforming === deforming && triangles > best.triangles)) best = { trees, deforming, triangles };
        }
        return best?.trees ?? null;
    }

    /** The adopt lever: the batch's output buffers become the storage, with its trees in place. */
    private adoptLoadBatch(encoder: GPUCommandEncoder, trees: readonly GpuArrivedTree[], replaced: ReadonlyMap<number, TreeBuilder>): void {
        for (const tree of trees) this.releaseTree(tree.slot);
        const outputs = trees[0].built.outputs;
        this.nodes.adopt(outputs.nodes, trees.map((tree) => ({ offset: tree.built.tree.nodeOffset, count: tree.built.tree.nodeCount })));
        this.triangles.adopt(outputs.triangles, trees.map((tree) => ({ offset: tree.built.tree.triangleOffset, count: tree.built.tree.triangleCount })));
        outputs.markAdopted();
        this.liveOutputs.delete(outputs);
        this.versionBase++;
        for (const tree of trees) {
            const built = tree.built.tree;
            const state = this.meshes.get(tree.slot)!;
            if (state.group >= 0) {
                this.identity.encode(encoder, this.triangles.buffer, this.groups[state.group].group, built.triangleOffset, built.triangleCount);
                this.encoderWritesStorage = true;
            }
            this.recordPlacement(tree, treeSize(tree), built.nodeOffset, built.triangleOffset, replaced.get(tree.slot) ?? null);
        }
        this.reconcileReserve();
    }

    /**
     * What a mesh's tinybvh build asks the workers for: the triangle format,
     * and quantized nodes unless the mesh deforms (its refit keeps an `f32`
     * copy, which an `f32` tree gives exactly; it is packed when placed).
     */
    private tinybvhOptions(slot: number): MeshBvhBuildOptions {
        const indexed = this.triangleLeaves === 'indexed' ? { triangleLeaves: 'indexed' as const } : {};
        // Deforming meshes' trees stay f32 (the refits work on f32 nodes; they are quantized on the way in).
        return this.nodeFormat === 'quantized' && !this.meshes.get(slot)?.deforming ? { ...indexed, nodeFormat: 'quantized' } : indexed;
    }

    /** A tinybvh tree's nodes into the storage at its root `nodeOffset` (quantized: its frame slot first; packed here if the worker sent f32). */
    private writeTinybvhNodes(bvh: MeshBvh, nodeOffset: number): void {
        if (this.nodeFormat !== 'quantized') {
            this.nodes.write(nodeOffset, bvh.nodes, bvh.nodeCount * BVH_NODE_BYTES);
            return;
        }
        const quantized = quantizeNodes(bvh);
        this.nodes.write(nodeOffset - 1, quantized.nodes, (bvh.nodeCount + 1) * QUANTIZED_NODE_BYTES);
    }

    /**
     * Indexed: units before a mesh's first triangle (its header, and a merged
     * group's side and source tables); 0 for records.
     */
    private leadingOf(slot: number, triangleCount: number): number {
        if (!this.positions) return 0;
        const state = this.meshes.get(slot);
        const sources = state && state.group >= 0 ? this.groups[state.group].group.sources.length : 0;
        return indexedLeadingUnits(triangleCount, sources, Boolean(state?.materialCodes));
    }

    /**
     * Indexed: each starting mesh's positions (as its trees' vertex indices
     * name them: welded, or the mesh's own) at a range of the positions
     * buffer, growing it in `encoder` if need be. A mesh keeps its range
     * across tree swaps: every tree of a mesh is built from the same geometry.
     */
    private placePositions(states: readonly MeshSlot[], encoder: GPUCommandEncoder): void {
        const positions = this.positions!;
        const stride = positionStrideFloats(this.positionLayout);
        const pending = states.filter((state) => state.vertices === null);
        const counts = pending.map((state) => state.geometry!.positions.length / 3);
        const residents = [...this.meshes.values()].flatMap((state) => (state.vertices ? [state.vertices] : [])).concat(this.releasePositionsAtUpdate);
        let offsets = counts.map((count) => positions.allocate(Math.max(1, count)));
        if (offsets.some((offset) => offset === null)) {
            // Give back what fitted, then check before growing: a throw leaves the storage as it was.
            offsets.forEach((offset, index) => offset !== null && positions.release(offset, Math.max(1, counts[index])));
            positions.checkGrowth(sum(counts.map((count) => Math.max(1, count))));
            positions.grow(encoder, sum(counts.map((count) => Math.max(1, count))), residents);
            offsets = allocateGrown(positions, counts.map((count) => Math.max(1, count)));
        }
        pending.forEach((state, index) => {
            const source = state.geometry!.positions;
            const count = counts[index];
            const data = new Uint32Array(Math.max(1, count) * stride);
            const bits = new Uint32Array(source.buffer, source.byteOffset, count * 3);
            for (let vertex = 0; vertex < count; vertex++) {
                data[vertex * stride] = bits[vertex * 3];
                data[vertex * stride + 1] = bits[vertex * 3 + 1];
                data[vertex * stride + 2] = bits[vertex * 3 + 2];
            }
            positions.write(offsets[index]!, data);
            state.vertices = { offset: offsets[index]!, count: Math.max(1, count) };
        });
    }

    /**
     * Indexed: a tinybvh tree's triangles (converted if the worker wrote
     * records), with its header, or a merged group's side tables and identity,
     * written in one piece from its leading units on.
     */
    private writeIndexedTree(state: MeshSlot, bvh: MeshBvh, triangleOffset: number, leading: number): void {
        if (bvh.triangleLeaves !== 'indexed') throw new Error('@pmndrs/bvh: an indexed scene got a records tree from tinybvh');
        const words = new Uint32Array((leading + bvh.triangleCount) * INDEXED_TRIANGLE_WORDS);
        words.set(new Uint32Array(bvh.triangles, 0, bvh.triangleCount * INDEXED_TRIANGLE_WORDS), leading * INDEXED_TRIANGLE_WORDS);
        if (state.group >= 0) {
            writeMergedIndexedIdentity(words, this.groups[state.group].group, leading, bvh.triangleCount);
            words[(leading - 1) * INDEXED_TRIANGLE_WORDS + MESH_HEADER_WORD.vertexBase] = state.vertices?.offset ?? 0;
        } else {
            writeIndexedMeshHeader(words, leading, state.vertices?.offset ?? 0, state.materialCodes);
        }
        this.triangles.write(triangleOffset - leading, words);
    }

    /**
     * Writes every due CPU pose ({@link setPoseCpu}) into its mesh's placed
     * tree: refit on the CPU, then copied in from staging buffers through
     * `encoder`, after whatever the update recorded before (a tree's copy or
     * a growth would otherwise overwrite a queue write). The mesh's top-level
     * box becomes the posed root box.
     */
    private writeCpuPoses(encoder: GPUCommandEncoder): void {
        for (const state of this.meshes.values()) {
            const pose = state.cpuPose;
            if (!pose?.due || !pose.positions) continue;
            const placed = this.placed.get(state.slot);
            if (!placed) continue;
            if (!pose.layout) {
                // The placed tree came without its layout (its readback failed, or a custom first-tree builder ignored readLayouts):
                // the pose waits for the next tree that has one (tinybvh's refinement, or a drift rebuild).
                if (!this.cpuPoseLayoutWarned) {
                    this.cpuPoseLayoutWarned = true;
                    this.warn(
                        `@pmndrs/bvh: SceneTrees.setPoseCpu: mesh ${state.slot}'s placed tree (${placed.source}) has no layout on the CPU (the readback failed, or the first-tree builder ignored readLayouts), ` +
                            "so its CPU pose waits for the next tree that has one (tinybvh's refinement: don't hold refinements forever)",
                    );
                }
                continue;
            }
            const posed = refitLayoutToPose(pose.layout, pose.positions, pose.indices);
            if (this.nodeFormat === 'quantized') {
                const quantized = quantizeNodes({ ...pose.layout, nodes: posed.nodes, nodeFormat: 'f32' });
                this.stagedCopy(encoder, this.nodes, placed.nodeOffset - 1, quantized.nodes);
            } else {
                this.stagedCopy(encoder, this.nodes, placed.nodeOffset, posed.nodes);
            }
            if (posed.triangles) {
                this.stagedCopy(encoder, this.triangles, placed.triangleOffset, posed.triangles);
            } else if (this.positions && state.vertices) {
                // Indexed: deforming meshes are never welded, so pose vertex v is scene vertex base + v.
                const stride = positionStrideFloats(this.positionLayout);
                const count = pose.vertexCount;
                const data = new Float32Array(Math.max(1, count) * stride);
                for (let vertex = 0; vertex < count; vertex++) {
                    data[vertex * stride] = pose.positions[vertex * 3];
                    data[vertex * stride + 1] = pose.positions[vertex * 3 + 1];
                    data[vertex * stride + 2] = pose.positions[vertex * 3 + 2];
                }
                this.stagedCopy(encoder, this.positions, state.vertices.offset, data.buffer);
            }
            this.moveTopLevelBox(state.slot, posed.bounds);
            this.encoderWritesStorage = true;
            pose.due = false;
            pose.moved = true;
            this.cpuPoses++;
        }
    }

    /** A placed mesh's box in the top level, for every instance of it at the next update. */
    private moveTopLevelBox(mesh: number, box: ArrayLike<number>): void {
        const placed = this.placed.get(mesh);
        if (!placed) return;
        const moved: PlacedTree = { ...placed, bounds: Array.from({ length: 6 }, (_, k) => box[k]) };
        this.placed.set(mesh, moved);
        this.gpuTopLevel?.setMeshTree(mesh, moved);
        for (const [slot, instance] of this.instances) if (instance.mesh === mesh) this.dirtyInstances.add(slot);
        this.cpuTopLevelStale = true;
    }

    /** Copies `data` into `storage` at unit `offset` through a staging buffer, in `encoder`. */
    private stagedCopy(encoder: GPUCommandEncoder, storage: GrowableStorage, offset: number, data: ArrayBuffer): void {
        const size = Math.ceil(data.byteLength / 4) * 4;
        const staging = this.device.createBuffer({ label: `${this.label} cpu pose`, size, usage: 0x4 /* COPY_SRC */, mappedAtCreation: true });
        new Uint8Array(staging.getMappedRange()).set(new Uint8Array(data));
        staging.unmap();
        encoder.copyBufferToBuffer(staging, 0, storage.buffer, offset * storage.unitBytes, size);
        this.retiredStaging.push(staging);
    }

    private recordPlacement(tree: ArrivedTree, size: TreeSize, nodeOffset: number, triangleOffset: number, replaced: TreeBuilder | null, triangleLeading = 0): void {
        const state = this.meshes.get(tree.slot)!;
        // A deforming mesh's box in the top level is the box every pose stays in, so a pose never changes the top level.
        const box = this.deforming?.deformBox(tree.slot);
        const placed: PlacedTree = {
            nodeOffset,
            nodeCount: size.nodeCount,
            triangleOffset,
            triangleCount: size.triangleCount,
            triangleLeading,
            maxDepth: size.maxDepth,
            bounds: box ? Array.from(box) : size.bounds,
            source: tree.source,
            generation: tree.generation,
        };
        this.placed.set(tree.slot, placed);
        this.gpuTopLevel?.setMeshTree(tree.slot, placed);
        if (this.deforming?.has(tree.slot)) {
            const layout: GpuLayoutMeshBvh | null = tree.source === 'tinybvh' ? tree.bvh : tree.built.layout;
            // meshRefitPlacement refuses merged meshes: merged ranges are never refit.
            // Indexed: the refit copies the pose to the mesh's vertex base (deforming meshes are unwelded, so pose vertex v is scene vertex base + v).
            const placement = meshRefitPlacement(
                {
                    meshes: [
                        {
                            nodeOffset,
                            nodeCount: placed.nodeCount,
                            triangleOffset,
                            triangleCount: placed.triangleCount,
                            ...(state.mergedInto !== null ? { mergedInto: state.mergedInto } : {}),
                            ...(state.vertices ? { vertexBase: state.vertices.offset, vertexCount: state.vertices.count, welded: false } : {}),
                        },
                    ],
                    triangleLeaves: this.triangleLeaves,
                    positionLayout: this.positionLayout,
                    nodeFormat: this.nodeFormat,
                },
                0,
            );
            this.deforming.treePlaced(tree.slot, placement, layout);
        }
        if (state.cpuPose) {
            // A CPU pose given before is written into the new tree too, at this update.
            state.cpuPose.layout = tree.source === 'tinybvh' ? tree.bvh : tree.built.layout;
            state.cpuPose.moved = false;
            state.cpuPose.due = state.cpuPose.positions !== null;
        }
        if (replaced === 'gpu' && tree.source === 'tinybvh') this.swaps++;
        this.onTree?.({
            mesh: state.group >= 0 ? null : state.slot,
            group: state.group >= 0 ? state.group : null,
            builtBy: tree.source,
            replaced,
            fallbackReason: tree.source === 'tinybvh' && replaced === null ? state.fallbackReason : null,
            triangleCount: placed.triangleCount,
            ms: performance.now() - state.startedAt,
        });
    }

    // -----------------------------------------------------------------------
    // Top level
    // -----------------------------------------------------------------------

    /** The top-level instance of an instance slot (or a merged group's record), or null when it isn't in the top level. */
    private topLevelInstance(slot: number): GpuTopLevelInstance | null {
        const group = this.groups.find((entry) => entry.instanceSlot === slot);
        if (group) {
            return { meshSlot: group.meshSlot, instanceIndex: MERGED_INSTANCE_INDEX, objectToWorld: IDENTITY_TRANSFORM, mask: group.group.mask, merged: true };
        }
        const instance = this.instances.get(slot);
        if (!instance || instance.merged >= 0) return null;
        return {
            meshSlot: instance.mesh,
            instanceIndex: instance.instanceIndex,
            objectToWorld: instance.objectToWorld,
            visible: instance.visible,
            contributesToGi: instance.contributesToGi,
            mask: instance.mask,
            material: instance.material ?? undefined,
        };
    }

    private syncInstances(): void {
        if (!this.gpuTopLevel) {
            for (const slot of this.dirtyInstances) if (!this.instances.has(slot) && !this.groups.some((group) => group.instanceSlot === slot)) this.instanceSlots.give(slot);
            this.dirtyInstances.clear();
            return;
        }
        for (const slot of this.dirtyInstances) {
            const instance = this.topLevelInstance(slot);
            if (instance && this.gpuTopLevel.isMerged(slot)) continue;
            if (!instance) {
                this.gpuTopLevel.setInstance(slot, null);
                if (!this.instances.has(slot) && !this.groups.some((group) => group.instanceSlot === slot)) this.instanceSlots.give(slot);
                continue;
            }
            this.gpuTopLevel.setInstance(slot, instance);
        }
        this.dirtyInstances.clear();
    }

    /** The deepest placed tree that rays can reach: a mesh with instances, or a merged group. */
    private deepestPlacedMesh(): number {
        let deepest = 0;
        for (const [slot, tree] of this.placed) {
            const state = this.meshes.get(slot);
            if (state && (state.group >= 0 || state.users > 0)) deepest = Math.max(deepest, tree.maxDepth);
        }
        return deepest;
    }

    private updateTopLevel(encoder: GPUCommandEncoder): void {
        if (this.gpuTopLevel) {
            this.gpuTopLevel.update(encoder, { deepestMesh: this.deepestPlacedMesh() });
            return;
        }
        if (!this.cpuTopLevelStale) return;
        this.cpuTopLevelStale = false;
        const traced: TraceInstance[] = [];
        let deepestMesh = 0;
        const add = (slot: number) => {
            const instance = this.topLevelInstance(slot);
            if (!instance) return;
            const tree = this.placed.get(instance.meshSlot);
            if (!tree) return;
            deepestMesh = Math.max(deepestMesh, tree.maxDepth);
            traced.push({
                instanceIndex: instance.instanceIndex,
                nodeOffset: tree.nodeOffset,
                triangleOffset: tree.triangleOffset,
                meshBounds: tree.bounds,
                objectToWorld: instance.objectToWorld,
                contributesToGi: instance.contributesToGi,
                visible: instance.visible,
                mask: instance.mask,
                merged: instance.merged,
                material: instance.material,
            });
        };
        for (const slot of this.instances.keys()) add(slot);
        for (const group of this.groups) add(group.instanceSlot);
        const topLevel = buildTopLevel(traced);
        const requiredStack = topLevel.maxDepth + 1 + deepestMesh;
        if (topLevel.instanceCount > 0 && requiredStack > TRACE_STACK_SIZE) {
            this.cpuTopLevelStale = true;
            throw new RangeError(`@pmndrs/bvh: the trace stack holds ${TRACE_STACK_SIZE} entries, but this scene can need ${requiredStack} (top level ${topLevel.maxDepth} + 1 + deepest mesh tree ${deepestMesh}).`);
        }
        const bytes = topLevel.buffer.byteLength;
        if (bytes > this.cpuTopLevel.size) {
            const limits = this.device.limits;
            const limitBytes = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
            if (bytes > limitBytes) {
                this.cpuTopLevelStale = true;
                throw new RangeError(`@pmndrs/bvh: the top level needs ${bytes} bytes in one binding, the device allows ${limitBytes}`);
            }
            this.retiredTopLevel.push(this.cpuTopLevel);
            const cap = Math.floor(limitBytes / 16) * 16;
            this.cpuTopLevel = this.createTopLevelBuffer(Math.min(cap, Math.max(bytes, this.cpuTopLevel.size * 2)));
            this.versionBase++;
        }
        this.device.queue.writeBuffer(this.cpuTopLevel, 0, topLevel.buffer);
        this.cpuTopLevelInstanceCount = topLevel.instanceCount;
    }

    /** Instances in the current top level (traced, or with the GPU top level also hidden ones in its shape). */
    get topLevelInstanceCount(): number {
        return this.gpuTopLevel ? this.gpuTopLevel.instanceCount : this.cpuTopLevelInstanceCount;
    }

    private createTopLevelBuffer(size: number): GPUBuffer {
        return this.device.createBuffer({ label: `${this.label} trace top level`, size: Math.ceil(size / 16) * 16, usage: USAGE.STORAGE | USAGE.COPY_DST | USAGE.COPY_SRC });
    }

    private noteTimeline(): void {
        if (this.loadStartedAt === null) return;
        const now = performance.now();
        if (this.firstTreesAt === null && this.meshesWaitingForFirstTree() === 0 && this.placed.size > 0) this.firstTreesAt = now;
        if (this.tinybvhTreesAt === null && this.pendingTinybvh === 0 && !this.arrived.some((tree) => tree.source === 'tinybvh')) {
            const allTinybvh = [...this.meshes.values()].every((state) => state.mergedInto !== null || state.tinybvh !== 'built' || this.placed.get(state.slot)?.source === 'tinybvh');
            if (allTinybvh && this.placed.size > 0) this.tinybvhTreesAt = now;
        }
    }

    private checkLive(): void {
        if (this.disposed) throw new Error('@pmndrs/bvh: the SceneTrees was disposed');
    }
}

function checkMaterial(material: number | undefined): void {
    materialCode(material, -1);
}

interface TreeSize {
    readonly nodeCount: number;
    readonly triangleCount: number;
    readonly maxDepth: number;
    readonly bounds: readonly number[];
}

/** A tree's counts, depth and box, copied out field by field (never a spread: a `MeshBvh` carries its node and triangle arrays). */
function sum(values: readonly number[]): number {
    return values.reduce((total, value) => total + value, 0);
}

/** Ranges a placement took (from a reserve or free room), and how to give them back. */
interface TakenRanges {
    readonly offsets: number[];
    undo(): void;
}

/**
 * Ranges for every count: those marked `fromReserve` (GPU trees of a capped
 * load) out of the reserve when it has room, the rest (and the reserve's
 * misses) from the storage's free room. Null, with nothing taken, when one
 * doesn't fit.
 */
function takeRanges(storage: GrowableStorage, reserve: StorageReserve, counts: readonly number[], fromReserve: readonly boolean[]): TakenRanges | null {
    const offsets: number[] = [];
    const reserved: boolean[] = [];
    const undo = () =>
        offsets.forEach((offset, index) => {
            if (reserved[index]) reserve.giveBack(offset, counts[index]);
            else storage.release(offset, counts[index]);
        });
    for (const [index, count] of counts.entries()) {
        const fromReserved = fromReserve[index] ? reserve.take(count) : null;
        const offset = fromReserved ?? storage.allocate(count);
        if (offset === null) {
            undo();
            return null;
        }
        offsets.push(offset);
        reserved.push(fromReserved !== null);
    }
    return { offsets, undo };
}

/** Ranges after a growth, which made room for all of them after the high-water mark. */
function allocateGrown(storage: GrowableStorage, counts: readonly number[]): number[] {
    return counts.map((count) => {
        const offset = storage.allocate(count);
        if (offset === null) throw new Error('@pmndrs/bvh: trace buffer allocation failed after growing');
        return offset;
    });
}

/** `reserve` if the storage can grow by `extra + reserve` within the device's limits, else 0 (the reservation is only a hint). */
function fitReserve(storage: GrowableStorage, extra: number, reserve: number): number {
    if (reserve <= 0) return 0;
    try {
        storage.checkGrowth(extra + reserve);
        return reserve;
    } catch {
        return 0;
    }
}


function treeSize(tree: ArrivedTree): TreeSize {
    const source = tree.source === 'tinybvh' ? tree.bvh : tree.built.tree;
    return { nodeCount: source.nodeCount, triangleCount: source.triangleCount, maxDepth: source.maxDepth, bounds: Array.from(source.bounds) };
}


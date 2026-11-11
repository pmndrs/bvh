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
 * - **Merged static instances** (`mergeStaticInstances`): at the first update,
 *   the static instances added so far are merged (static-merge.ts), and each
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
 *   (one node per triangle) after the swaps.
 *
 * Call {@link SceneTrees.update} once per frame, before anything traces, with
 * the frame's encoder; it records copies and passes, and the caller submits.
 */

import { checkTriangleFlags } from '../mesh-input.ts';
import {
    BVH_NODE_BYTES,
    checkInstanceMask,
    materialCode,
    TOP_LEVEL_HEADER_BYTES,
    TOP_LEVEL_HEADER_WORDS,
    TRACE_STACK_SIZE,
    TRIANGLE_BYTES,
} from '../trace/layout.ts';
import type { MeshBvh, MeshGeometry } from '../trace/mesh-bvh.ts';
import type { TraceSceneMeshPlacement } from '../trace/scene-assembly.ts';
import { MERGED_INSTANCE_INDEX, mergeStaticGeometry, writeMergedTriangleIdentity, type StaticMergeGroup, type StaticMergePlan } from '../trace/static-merge.ts';
import { buildTopLevel, IDENTITY_TRANSFORM, throwMergedTransformChange, type Mat4, type TraceInstance } from '../trace/top-level.ts';
import { GpuTopLevel, type GpuTopLevelInstance } from '../trace/top-level-gpu.ts';
import type { TraceSceneBuffers } from '../trace/trace-gpu.ts';
import { countTriangles } from '../trace/triangle-packing.ts';
import { createBvhWorkerPool, type BvhWorkerPool, type BvhWorkerPoolOptions } from '../worker/pool.ts';
import { GpuFirstTrees, type FirstTreeBuilder, type GpuBuiltTree, type GpuTreeBatchResult, type GpuTreeOutputs, type GpuTreeRequest } from './gpu-first-trees.ts';
import { GrowableStorage, type ResidentRange } from './growable-storage.ts';
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
    /** How GPU trees reach the scene's buffers: `'copy'` (default) or `'adopt'` (see the file comment). */
    readonly gpuTreeOutputs?: GpuTreeOutputsMode;
    /** The worker pool for tinybvh builds, or options for one this scene makes (and disposes). */
    readonly workers?: BvhWorkerPool | BvhWorkerPoolOptions;
    /**
     * Move each mesh's geometry arrays into the worker instead of copying them
     * (the caller's arrays are empty afterwards). Default false. Merged groups'
     * geometry is the scene's own and always moves.
     */
    readonly transferGeometry?: boolean;
    /** Merge static instances at the first update (see the file comment). Default false. */
    readonly mergeStaticInstances?: boolean;
    /**
     * tinybvh's trees wait instead of replacing a mesh's GPU tree, until
     * {@link SceneTrees.releaseRefinements}. A mesh with no tree yet still takes
     * tinybvh's. For tools that check a scene mid-swap. Default false.
     */
    readonly holdRefinements?: boolean;
    /** GPU trees deeper than this are refused, and the mesh waits for tinybvh. Default {@link GPU_TREE_MAX_DEPTH}. */
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
}

/** GPU bytes the scene's trees hold. */
export interface SceneTreesMemory {
    readonly nodes: number;
    readonly nodesInUse: number;
    readonly triangles: number;
    readonly trianglesInUse: number;
    readonly topLevel: number;
    /** Deforming meshes' refits (their topology, scratch and stats). */
    readonly meshRefits: number;
    /** GPU trees in their batches' buffers until copied out (copy mode), or until every tree of an adopted batch swapped (only its parents, freed at once). */
    readonly gpuBuildOutputs: number;
    /** Batches still building: their inputs, scratch and outputs, until their stats are read. */
    readonly gpuBuilding: number;
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

interface PlacedTree {
    readonly nodeOffset: number;
    readonly nodeCount: number;
    readonly triangleOffset: number;
    readonly triangleCount: number;
    readonly maxDepth: number;
    readonly bounds: Float32Array;
    readonly source: TreeBuilder;
    readonly generation: number;
}

interface MeshSlot {
    readonly slot: number;
    generation: number;
    /** The merged group this slot builds, or -1 for a mesh the caller added. */
    readonly group: number;
    /** Kept until the builds start. */
    geometry: MeshGeometry | null;
    readonly triangleCount: number;
    readonly bounds: Float32Array;
    gpu: 'none' | 'building' | 'built' | 'refused';
    tinybvh: 'none' | 'building' | 'built' | 'failed';
    fallbackReason: string | null;
    /** Only merged instances use it: it has no tree of its own. */
    mergedInto: number | null;
    users: number;
    startedAt: number;
    abort: AbortController | null;
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
    | { readonly source: 'gpu'; readonly slot: number; readonly generation: number; readonly built: GpuBuiltTree };

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
    private readonly gpuTopLevel: GpuTopLevel | null;
    private cpuTopLevel: GPUBuffer;
    private cpuTopLevelStale = true;
    private cpuTopLevelInstanceCount = 0;
    private readonly retiredTopLevel: GPUBuffer[] = [];
    private readonly identity: GpuMergedIdentity;

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
    private gpuBuilding = 0;
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
        this.transferGeometry = options.transferGeometry ?? false;
        this.mergeStatic = options.mergeStaticInstances ?? false;
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
            this.firstTrees = new GpuFirstTrees(device, { label: this.label });
            this.noGpuReason = null;
        } else {
            this.firstTrees = null;
            this.noGpuReason = 'the device has no subgroups, so there is no GPU builder (tinybvh is the fallback)';
        }
        this.nodes = new GrowableStorage(device, `${this.label} trace nodes`, BVH_NODE_BYTES, INITIAL_NODES);
        this.triangles = new GrowableStorage(device, `${this.label} trace triangles`, TRIANGLE_BYTES, INITIAL_TRIANGLES);
        this.topLevelPath = options.topLevel !== 'cpu' && GpuTopLevel.isSupported(device) ? 'gpu' : 'cpu';
        this.gpuTopLevel =
            this.topLevelPath === 'gpu'
                ? new GpuTopLevel(device, { label: this.label, buildIn: this.pool.usesWorkers ? 'worker' : 'main', diagnostics: options.diagnostics, warn: this.warn })
                : null;
        // An empty top level: the header says 0 nodes, so every ray misses.
        this.cpuTopLevel = this.createTopLevelBuffer(TOP_LEVEL_HEADER_BYTES);
        device.queue.writeBuffer(this.cpuTopLevel, 0, new Uint32Array(TOP_LEVEL_HEADER_WORDS));
        this.identity = new GpuMergedIdentity(device, this.label);
    }

    /** Compiles the GPU builder, the top level and the merged-identity pass, all at once. */
    private async prepare(): Promise<void> {
        await Promise.all([this.prepareFirstTrees(), this.gpuTopLevel?.prepare(), this.firstTrees ? this.identity.prepare() : undefined]);
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

    /** The three buffers `traceModuleWgsl` binds (`bvhTraceNodes`, `bvhTraceTriangles`, `bvhTraceTopLevel`). */
    get buffers(): { readonly nodes: GPUBuffer; readonly triangles: GPUBuffer; readonly topLevel: GPUBuffer } {
        return { nodes: this.nodes.buffer, triangles: this.triangles.buffer, topLevel: this.topLevelBuffer };
    }

    /** Increments whenever one of {@link buffers} is replaced: make bind groups again then. */
    get version(): number {
        return this.versionBase + this.nodes.version + this.triangles.version + (this.gpuTopLevel?.version ?? 0);
    }

    /** The buffers as `TraceBatchKernel` takes them; the same object until {@link version} changes. */
    get traceBuffers(): TraceSceneBuffers {
        const version = this.version;
        if (this.traceBuffersCache?.version !== version) {
            const { nodes, triangles, topLevel } = this.buffers;
            this.traceBuffersCache = { version, buffers: { nodes, triangles, topLevel, byteLength: nodes.size + triangles.size + topLevel.size, destroy: () => {} } };
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
    addMesh(geometry: MeshGeometry): number {
        this.checkLive();
        const triangleCount = countTriangles(geometry);
        checkTriangleFlags(geometry.triangleFlags, triangleCount, 'SceneTrees.addMesh');
        if (triangleCount === 0) throw new RangeError('SceneTrees.addMesh: the mesh has no triangles');
        const slot = this.meshSlots.take();
        this.meshes.set(slot, this.newMeshSlot(slot, -1, geometry, triangleCount));
        this.newMeshes.push(slot);
        return slot;
    }

    private newMeshSlot(slot: number, group: number, geometry: MeshGeometry, triangleCount: number): MeshSlot {
        const previous = this.retiredGenerations.get(slot) ?? 0;
        return {
            slot,
            generation: previous + 1,
            group,
            geometry,
            triangleCount,
            bounds: geometryBounds(geometry.positions),
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
        this.releaseTree(mesh);
        this.meshes.delete(mesh);
        this.newMeshes = this.newMeshes.filter((slot) => slot !== mesh);
        this.retiredGenerations.set(mesh, state.generation);
        this.meshSlots.give(mesh);
        this.cpuTopLevelStale = true;
        this.notifyWaiters();
    }

    /** Adds an instance of a mesh. It becomes visible to rays once its mesh has a tree and (GPU top level) a shape holds it. */
    addInstance(descriptor: SceneInstanceDescriptor): number {
        this.checkLive();
        const mesh = this.meshes.get(descriptor.mesh);
        if (!mesh || mesh.group >= 0) throw new RangeError(`SceneTrees.addInstance: no mesh ${descriptor.mesh}`);
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
        const meshRefits = 0;
        let gpuBuildOutputs = 0;
        for (const outputs of this.liveOutputs) gpuBuildOutputs += outputs.bytes;
        const retired = this.nodes.retiredByteLength + this.triangles.retiredByteLength + this.retiredTopLevel.reduce((sum, buffer) => sum + buffer.size, 0) + this.identity.pendingBytes;
        const resident = nodes + triangles + topLevel + meshRefits;
        const transient = gpuBuildOutputs + this.gpuBuilding + retired;
        const total = resident + transient;
        if (total > this.peakBytes) this.peakBytes = total;
        return {
            nodes,
            nodesInUse: this.nodes.usedBytes,
            triangles,
            trianglesInUse: this.triangles.usedBytes,
            topLevel,
            meshRefits,
            gpuBuildOutputs,
            gpuBuilding: this.gpuBuilding,
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
     */
    async whenBuilt(): Promise<void> {
        if (this.meshesWaitingForFirstTree() > 0 || this.newMeshes.length > 0) await new Promise<void>((resolve) => this.builtWaiters.push(resolve));
        await this.gpuTopLevel?.whenIdle();
    }

    /** Resolves once every tinybvh build has finished (held trees still wait for {@link releaseRefinements}). */
    async whenRefined(): Promise<void> {
        if (this.pendingTinybvh > 0 || this.newMeshes.length > 0) await new Promise<void>((resolve) => this.refinedWaiters.push(resolve));
        await this.whenBuilt();
    }

    /**
     * Lets tinybvh trees held by `holdRefinements` replace GPU trees from the
     * next update: all of them, or only the meshes `which` accepts (by mesh
     * handle; merged groups by `-1 - group`).
     */
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
     * before anything traces; the caller submits `encoder`. Throws (after
     * leaving everything consistent) if a buffer would pass a device limit or
     * a build failed.
     */
    update(encoder: GPUCommandEncoder): void {
        this.checkLive();
        if (!this.prepared) throw new Error('@pmndrs/bvh: SceneTrees is not prepared; make it with SceneTrees.create');
        this.updates++;
        this.nodes.releaseRetired();
        this.triangles.releaseRetired();
        for (const buffer of this.retiredTopLevel.splice(0)) buffer.destroy();
        this.identity.releaseRetired();
        // The caller has submitted the frame that copied these GPU trees out (or dropped them).
        for (const outputs of this.outputsToRelease) this.releaseOutputs(outputs);
        this.outputsToRelease = [];

        if (this.updates === 1 && this.mergeStatic) this.planMerges();
        if (this.newMeshes.length > 0) {
            this.loadStartedAt ??= performance.now();
            this.startBuilds(this.newMeshes.splice(0));
        }

        let failure: unknown = null;
        const arrived = this.takeArrived();
        if (arrived.length > 0) {
            try {
                this.placeTrees(encoder, arrived);
            } catch (error) {
                failure = error;
            }
            this.cpuTopLevelStale = true;
        }
        this.syncInstances();
        try {
            this.updateTopLevel(encoder);
        } catch (error) {
            failure ??= error;
        }
        this.noteTimeline();
        this.samplePeak();
        if (failure === null && this.buildFailure) {
            failure = this.buildFailure;
            this.buildFailure = null;
        }
        if (failure !== null) throw failure;
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        for (const mesh of this.meshes.values()) mesh.abort?.abort();
        if (this.ownsPool) this.pool.dispose();
        this.gpuTopLevel?.dispose();
        this.identity.dispose();
        this.nodes.destroy();
        this.triangles.destroy();
        this.cpuTopLevel.destroy();
        for (const buffer of this.retiredTopLevel.splice(0)) buffer.destroy();
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
                static: instance.static,
            })),
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

    /** Starts the GPU batch for these meshes, then their tinybvh builds, biggest first. */
    private startBuilds(slots: readonly number[]): void {
        const now = performance.now();
        const states = slots.map((slot) => this.meshes.get(slot)!).filter((state) => state.geometry !== null);
        const builder = this.firstTreesReady ? this.firstTrees : null;
        for (const state of states) {
            state.startedAt = now;
            state.gpu = builder ? 'building' : 'none';
            state.tinybvh = 'building';
            if (!builder) state.fallbackReason = this.noGpuReason ?? 'the GPU builder is not ready';
        }
        if (builder && states.length > 0) {
            const requests: GpuTreeRequest[] = states.map((state) => ({ slot: state.slot, generation: state.generation, geometry: state.geometry! }));
            this.startGpuBuilds(builder, requests);
        }
        // tinybvh after the GPU build has started: the GPU's writeBuffer has copied the arrays by now,
        // so moving them into a worker is safe (keep this order).
        const bySize = [...states].sort((a, b) => b.triangleCount - a.triangleCount);
        for (const state of bySize) this.startTinybvhBuild(state);
    }

    private startGpuBuilds(builder: FirstTreeBuilder, requests: readonly GpuTreeRequest[]): void {
        let builds;
        try {
            builds = builder.build(requests);
        } catch (error) {
            this.warn(`@pmndrs/bvh: the GPU mesh tree build failed, so the meshes wait for tinybvh: ${describe(error)}`);
            for (const { slot, generation } of requests) this.gpuRefused(slot, generation, `the GPU build failed: ${describe(error)}`);
            return;
        }
        for (const { slot, generation, reason } of builds.notBuilt) this.gpuRefused(slot, generation, reason);
        this.gpuBuilding += builds.buildingBytes;
        this.samplePeak();
        let left = builds.batches.length;
        if (left === 0) this.gpuBuilding -= builds.buildingBytes;
        for (const batch of builds.batches) {
            void batch.then((result) => {
                if (--left === 0) {
                    this.samplePeak();
                    this.gpuBuilding -= builds.buildingBytes;
                }
                this.gpuBatchFinished(result);
            });
        }
    }

    private gpuBatchFinished(result: GpuTreeBatchResult): void {
        this.lastGpuBuildMs = result.buildMs;
        for (const outputs of new Set(result.built.map((built) => built.outputs))) {
            if (this.disposed) outputs.destroy();
            else this.liveOutputs.add(outputs);
        }
        if (this.disposed) return;
        for (const { slot, generation, reason } of result.refused) this.gpuRefused(slot, generation, reason);
        if (result.failure) this.warn(`@pmndrs/bvh: ${result.failure}, so the meshes wait for tinybvh`);
        for (const built of result.built) {
            const state = this.currentState(built.slot, built.generation);
            if (!state || built.tree.maxDepth > this.gpuMaxDepth) {
                if (state) this.gpuRefused(built.slot, built.generation, `the GPU tree is ${built.tree.maxDepth} levels deep, more than gpuMaxDepth (${this.gpuMaxDepth})`);
                this.outputsToRelease.push(built.outputs);
                continue;
            }
            state.gpu = 'built';
            this.gpuTreesBuilt++;
            this.arrived.push({ source: 'gpu', slot: built.slot, generation: built.generation, built });
        }
        this.samplePeak();
        this.notifyWaiters();
    }

    private gpuRefused(slot: number, generation: number, reason: string): void {
        const state = this.currentState(slot, generation);
        if (!state) return;
        state.gpu = 'refused';
        state.fallbackReason = reason;
        this.gpuTreesRefused++;
    }

    private startTinybvhBuild(state: MeshSlot): void {
        const geometry = state.geometry!;
        const generation = state.generation;
        state.geometry = null;
        state.abort = new AbortController();
        this.pendingTinybvh++;
        const transferInput = state.group >= 0 || this.transferGeometry;
        const finished = (bvh: MeshBvh | null): void => {
            this.pendingTinybvh--;
            const current = this.disposed ? undefined : this.currentState(state.slot, generation);
            if (current) {
                current.abort = null;
                current.tinybvh = bvh ? 'built' : 'failed';
                if (bvh) this.arrived.push({ source: 'tinybvh', slot: state.slot, generation, bvh });
            }
            this.notifyWaiters();
        };
        this.pool.build(geometry, {}, { signal: state.abort.signal, transferInput }).then(finished, (error: unknown) => {
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
        for (const state of this.meshes.values()) {
            if (state.mergedInto !== null || (state.group < 0 && state.users === 0)) continue;
            if (this.placed.has(state.slot)) continue;
            // An arrived tree counts (it is placed at the next update), except a tinybvh tree that waits for the GPU's.
            const waitsForGpu = state.gpu === 'building' && (this.holdsRefinement(state.slot) || (this.gpuTreeOutputs === 'adopt' && this.placed.size === 0));
            const arrived = this.arrived.some((tree) => tree.slot === state.slot && tree.generation === state.generation && (tree.source === 'gpu' || !waitsForGpu));
            if (arrived) continue;
            if (state.gpu === 'building' || state.tinybvh === 'building') waiting++;
        }
        return waiting;
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
     * and the mesh's GPU build is running, so the load batch finds the
     * buffers empty.
     */
    private takeArrived(): ArrivedTree[] {
        const taken: ArrivedTree[] = [];
        const waiting: ArrivedTree[] = [];
        const tinybvhNow = new Set<number>();
        const gpuQueued = new Set(this.arrived.filter((tree) => tree.source === 'gpu' && this.currentState(tree.slot, tree.generation)).map((tree) => tree.slot));
        const waitForAdoption = this.gpuTreeOutputs === 'adopt' && this.placed.size === 0;
        for (const tree of this.arrived) {
            const state = this.currentState(tree.slot, tree.generation);
            if (!state || tree.source !== 'tinybvh') continue;
            const placed = this.placed.get(tree.slot);
            const gpuTreeFirst = state.gpu === 'building' || (placed?.source === 'gpu' && placed.generation === tree.generation) || gpuQueued.has(tree.slot);
            if ((gpuTreeFirst && this.holdsRefinement(tree.slot)) || (waitForAdoption && state.gpu === 'building')) {
                waiting.push(tree);
            } else {
                taken.push(tree);
                tinybvhNow.add(tree.slot);
            }
        }
        for (const tree of this.arrived) {
            if (tree.source !== 'gpu') continue;
            const placed = this.placed.get(tree.slot);
            const hasTinybvhTree = tinybvhNow.has(tree.slot) || (placed?.source === 'tinybvh' && placed.generation === tree.generation);
            const state = this.currentState(tree.slot, tree.generation);
            if (state && !hasTinybvhTree) {
                taken.push(tree);
            } else {
                if (state && hasTinybvhTree && state.fallbackReason === null) state.fallbackReason = "tinybvh's tree arrived before the GPU's";
                this.outputsToRelease.push(tree.built.outputs);
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
        this.nodes.release(tree.nodeOffset, tree.nodeCount);
        this.triangles.release(tree.triangleOffset, tree.triangleCount);
        this.placed.delete(slot);
        this.gpuTopLevel?.setMeshTree(slot, null);
    }

    private releaseOutputs(outputs: GpuTreeOutputs): void {
        outputs.release();
        if (outputs.isDestroyed) this.liveOutputs.delete(outputs);
    }

    /**
     * Places arrived trees: tinybvh's written from the CPU, GPU ones copied out
     * of their batch's buffers in `encoder` after any growth copies, or (adopt)
     * served in place. Merged groups get their identity written after.
     */
    private placeTrees(encoder: GPUCommandEncoder, arrived: readonly ArrivedTree[]): void {
        const replaced = new Map<number, TreeBuilder>();
        for (const tree of arrived) {
            const old = this.placed.get(tree.slot);
            if (old) replaced.set(tree.slot, old.source);
            this.releaseTree(tree.slot);
        }

        let rest = arrived;
        const adoptedSlots = new Set<number>();
        if (this.gpuTreeOutputs === 'adopt' && this.nodes.usedBytes === 0 && this.triangles.usedBytes === 0) {
            const adopted = this.adoptLoadBatch(encoder, arrived);
            if (adopted) {
                for (const tree of adopted) adoptedSlots.add(tree.slot);
                rest = arrived.filter((tree) => !adoptedSlots.has(tree.slot));
            }
        }

        const sizes = rest.map(treeSize);
        const nodeCounts = sizes.map((size) => size.nodeCount);
        const triangleCounts = sizes.map((size) => size.triangleCount);
        let nodeOffsets: number[] | null = null;
        let triangleOffsets: number[];
        try {
            const nodeHeadroom = rest.some((tree) => tree.source === 'gpu') ? GPU_TREE_NODE_HEADROOM : 0;
            nodeOffsets = this.allocateAll(this.nodes, encoder, nodeCounts, (placed) => ({ offset: placed.nodeOffset, count: placed.nodeCount }), nodeHeadroom);
            triangleOffsets = this.allocateAll(this.triangles, encoder, triangleCounts, (placed) => ({ offset: placed.triangleOffset, count: placed.triangleCount }));
        } catch (error) {
            // Nothing was written: keep the trees queued, so the next update tries again (and reports the same error).
            nodeOffsets?.forEach((offset, index) => this.nodes.release(offset, nodeCounts[index]));
            this.arrived.push(...rest);
            throw error;
        }

        rest.forEach((tree, index) => {
            const size = sizes[index];
            const nodeOffset = nodeOffsets[index];
            const triangleOffset = triangleOffsets[index];
            const state = this.meshes.get(tree.slot)!;
            if (tree.source === 'tinybvh') {
                if (state.group >= 0) writeMergedTriangleIdentity(tree.bvh.triangles, this.groups[state.group].group, 0, tree.bvh.triangleCount);
                this.nodes.write(nodeOffset, tree.bvh.nodes, size.nodeCount * BVH_NODE_BYTES);
                this.triangles.write(triangleOffset, tree.bvh.triangles, size.triangleCount * TRIANGLE_BYTES);
            } else {
                const { outputs, tree: built } = tree.built;
                encoder.copyBufferToBuffer(outputs.nodes, built.nodeOffset * BVH_NODE_BYTES, this.nodes.buffer, nodeOffset * BVH_NODE_BYTES, size.nodeCount * BVH_NODE_BYTES);
                encoder.copyBufferToBuffer(outputs.triangles, built.triangleOffset * TRIANGLE_BYTES, this.triangles.buffer, triangleOffset * TRIANGLE_BYTES, size.triangleCount * TRIANGLE_BYTES);
                // Let go of at the next update, once the caller has submitted these copies.
                this.outputsToRelease.push(outputs);
                if (state.group >= 0) this.identity.encode(encoder, this.triangles.buffer, this.groups[state.group].group, triangleOffset, size.triangleCount);
            }
            this.recordPlacement(tree, { nodeOffset, triangleOffset, ...size }, replaced.get(tree.slot) ?? null);
        });
    }

    /**
     * The adopt lever: the arrived GPU trees of one batch, all of its usable
     * trees, become the storage. Returns the adopted trees, or null when no
     * batch qualifies (then everything is copied).
     */
    private adoptLoadBatch(encoder: GPUCommandEncoder, arrived: readonly ArrivedTree[]): ArrivedTree[] | null {
        const byOutputs = new Map<GpuTreeOutputs, Extract<ArrivedTree, { source: 'gpu' }>[]>();
        for (const tree of arrived) {
            if (tree.source !== 'gpu') continue;
            const list = byOutputs.get(tree.built.outputs) ?? [];
            list.push(tree);
            byOutputs.set(tree.built.outputs, list);
        }
        let best: { outputs: GpuTreeOutputs; trees: Extract<ArrivedTree, { source: 'gpu' }>[]; triangles: number } | null = null;
        for (const [outputs, trees] of byOutputs) {
            const usable = outputs.meshes.filter((mesh) => mesh.problem === null).length;
            if (trees.length !== usable || outputs.isDestroyed) continue;
            const triangles = trees.reduce((sum, tree) => sum + tree.built.tree.triangleCount, 0);
            if (!best || triangles > best.triangles) best = { outputs, trees, triangles };
        }
        if (!best) return null;
        const { outputs, trees } = best;
        this.nodes.adopt(outputs.nodes, trees.map((tree) => ({ offset: tree.built.tree.nodeOffset, count: tree.built.tree.nodeCount })));
        this.triangles.adopt(outputs.triangles, trees.map((tree) => ({ offset: tree.built.tree.triangleOffset, count: tree.built.tree.triangleCount })));
        outputs.markAdopted();
        this.liveOutputs.delete(outputs);
        this.versionBase++;
        for (const tree of trees) {
            const built = tree.built.tree;
            const state = this.meshes.get(tree.slot)!;
            if (state.group >= 0) this.identity.encode(encoder, this.triangles.buffer, this.groups[state.group].group, built.triangleOffset, built.triangleCount);
            this.recordPlacement(tree, { nodeOffset: built.nodeOffset, triangleOffset: built.triangleOffset, ...treeSize(tree) }, null);
        }
        return trees;
    }

    private recordPlacement(tree: ArrivedTree, at: { nodeOffset: number; triangleOffset: number; nodeCount: number; triangleCount: number; maxDepth: number; bounds: Float32Array }, replaced: TreeBuilder | null): void {
        const state = this.meshes.get(tree.slot)!;
        const placed: PlacedTree = { ...at, bounds: Float32Array.from(at.bounds), source: tree.source, generation: tree.generation };
        this.placed.set(tree.slot, placed);
        this.gpuTopLevel?.setMeshTree(tree.slot, placed);
        if (replaced === 'gpu' && tree.source === 'tinybvh') this.swaps++;
        this.onTree?.({
            mesh: state.group >= 0 ? null : state.slot,
            group: state.group >= 0 ? state.group : null,
            builtBy: tree.source,
            replaced,
            fallbackReason: tree.source === 'tinybvh' && replaced === null ? state.fallbackReason : null,
            triangleCount: at.triangleCount,
            ms: performance.now() - state.startedAt,
        });
    }

    private allocateAll(storage: GrowableStorage, encoder: GPUCommandEncoder, counts: readonly number[], rangeOf: (tree: PlacedTree) => ResidentRange, headroom = 0): number[] {
        const offsets: number[] = [];
        for (const count of counts) {
            const offset = storage.allocate(count);
            if (offset === null) {
                offsets.forEach((placedOffset, index) => storage.release(placedOffset, counts[index]));
                const total = counts.reduce((sum, value) => sum + value, 0);
                storage.grow(encoder, total, [...this.placed.values()].map(rangeOf), headroom);
                return counts.map((value) => {
                    const grown = storage.allocate(value);
                    if (grown === null) throw new Error('@pmndrs/bvh: trace buffer allocation failed after growing');
                    return grown;
                });
            }
            offsets.push(offset);
        }
        return offsets;
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

    private deepestPlacedMesh(): number {
        let deepest = 0;
        for (const tree of this.placed.values()) deepest = Math.max(deepest, tree.maxDepth);
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

function treeSize(tree: ArrivedTree): { readonly nodeCount: number; readonly triangleCount: number; readonly maxDepth: number; readonly bounds: Float32Array } {
    return tree.source === 'tinybvh' ? tree.bvh : tree.built.tree;
}


/**
 * The top-level tree on the GPU: refit every frame something moved from the
 * instances' transforms, with a new shape built in a worker now and then.
 * Ported from ddgi-core's `platform/trace/top-level-gpu.ts` (stage B1a there,
 * DECISIONS 2026-11-04), made engine-free: the instances live in this class's
 * own table on the GPU instead of ddgi's scene copy.
 *
 * **Every frame something moved**, one compute pass, no CPU work per instance
 * beyond writing the moved instances' table entries:
 *
 * 1. *Gather* ({@link topLevelGatherWgsl}), one thread per instance record:
 *    reads the instance's table entry (object-to-world rows, world-to-object
 *    rows inverted on the CPU in doubles, mesh slot, instance index, flags)
 *    and its mesh tree's offsets and object box from the mesh table; writes
 *    the record the walk reads and the instance's world box
 *    (`bvhInstanceWorldBox`). An instance that is hidden, removed, doesn't
 *    contribute, has a singular transform, or whose mesh has no tree gets an
 *    empty box and a zero record: rays never enter it. The record's flags are
 *    the entry's, packed on the CPU from the instance input as `buildTopLevel`
 *    packs them: bits 0-3 (`INSTANCE_FLAG`), the mask in bits 8-15 (0xff
 *    unless the instance gives one) and the material in bits 16-31
 *    (`materialCode`: material + 1, 0 for none).
 * 2. *Refit* (`GpuBvhRefit`): recomputes every node box bottom-up from those
 *    world boxes, keeping the tree's shape, with its repair pass.
 *
 * **Now and then, a new shape**, built on the CPU in a worker
 * (top-level-async.ts) and swapped in at the start of a later `update`;
 * refits go on with the old shape meanwhile. A new shape is built when an
 * instance is added or removed, and when the refit tree's quality decayed:
 * while things move, every {@link QUALITY_CHECK_INTERVAL_MS} the worker refits
 * the current shape on the CPU from the current transforms and builds a new
 * one if its relative SAH cost passed {@link TOP_LEVEL_REBUILD_COST_RATIO}
 * times its cost as built. Nothing is read back from the GPU for this.
 *
 * Shapes include hidden instances too (their boxes are empty while hidden), so
 * hiding and showing take effect in the same frame. A new instance becomes
 * visible to rays when its shape arrives, usually a frame or two later. Mesh
 * trees placed or swapped only change the mesh table, so they need no shape.
 *
 * **Merged static instances** (`mergeStaticInstances`): a merged group's
 * record has `INSTANCE_FLAG.merged`, `instanceIndex` 0xffffffff and the
 * identity transform. Its transform can never change: {@link GpuTopLevel.setTransform}
 * and {@link GpuTopLevel.setInstance} throw for one, and shape builds refuse
 * a merged record that isn't the identity.
 *
 * Needs `subgroups` (the refit). Without them the declared fallback is the
 * CPU top level (`buildTopLevel`), rebuilt when something changed; `SceneTrees`
 * chooses.
 *
 * **Diagnostics** (`diagnostics: true`): the refit's stats are read back a
 * frame or two late, and a climb that gave up (and was repaired) is counted
 * and reported once through `warn`. Without diagnostics there are no readbacks.
 */

import { BVH_INSTANCE_BOX_WGSL } from '../gpu-instance-box.ts';
import { GpuBvhRefit, type BvhRefitStats } from '../gpu-refit.ts';
import { BVH_FLOAT_MAX, PRIMITIVE_BOX_FLOATS } from '../refit-cpu.ts';
import {
    BVH_NODE_BYTES,
    checkInstanceMask,
    INSTANCE_FLAG,
    INSTANCE_MASK_SHIFT,
    INSTANCE_RECORD_BYTES,
    MATERIAL_SHIFT,
    materialCode,
    materialFromFlags,
    NO_MATERIAL,
    TOP_HEADER_WORD,
    TOP_LEVEL_HEADER_BYTES,
    TOP_LEVEL_HEADER_WORDS,
    TRACE_STACK_SIZE,
} from './layout.ts';
import { MERGED_INSTANCE_INDEX } from './static-merge.ts';
import { createTopLevelBuilder, type TopLevelBuilder, type TopLevelResult } from './top-level-async.ts';
import { invertAffineInto, isIdentityTransform, throwMergedTransformChange, type Mat4 } from './top-level.ts';
import type { TopLevelBuildInput, TopLevelShape } from './top-level-shape.ts';

/**
 * Rebuild when the refit tree's relative SAH cost exceeds the as-built cost by
 * this factor (ddgi DECISIONS 2026-11-05, B1a). The cost is the expected
 * number of top-level nodes a ray visits, a small part of a ray's work, so 25%
 * more costs a few percent of trace time at most; a lower ratio would rebuild
 * continuously in a scene that never stops moving.
 */
export const TOP_LEVEL_REBUILD_COST_RATIO = 1.25;

/**
 * While instances move, the worker checks the shape's quality at most this
 * often. Each check packs every instance's transform on the main thread
 * (about 1.3 ms for 10,000), so it isn't done every frame.
 */
export const QUALITY_CHECK_INTERVAL_MS = 250;

/** Gather kernel workgroup size. */
const GATHER_WORKGROUP_SIZE = 64;

/** Mesh table entry: two vec4u per mesh slot, `(boundsMin.xyz, rootNodeOffset)` and `(boundsMax.xyz, triangleOffset)`. */
export const TOP_LEVEL_MESH_ENTRY_WORDS = 8;
/** `rootNodeOffset` of a mesh slot with no tree: its instances get empty boxes. */
export const TOP_LEVEL_MESH_ABSENT = 0xffffffff;

/**
 * Instance table entry: seven vec4 per instance slot. Rows 0-2 are
 * object-to-world (`xyz` linear part, `w` translation), rows 3-5
 * world-to-object as the walk's record holds them, and row 6 is
 * `(meshSlot, instanceIndex, flags, live)`.
 */
export const TOP_LEVEL_INSTANCE_ENTRY_WORDS = 28;
export const TOP_LEVEL_INSTANCE_ENTRY_WORD = {
    objectToWorldRow0: 0,
    worldToObjectRow0: 12,
    meshSlot: 24,
    instanceIndex: 25,
    flags: 26,
    /** 1 for a slot that holds an instance whose transform can be inverted; 0 otherwise. */
    live: 27,
} as const;

/** A mesh tree placed in the shared buffers, as the mesh table needs it. */
export interface PlacedMeshTree {
    readonly nodeOffset: number;
    readonly triangleOffset: number;
    /** Object-space box `[min, max]` of the tree's root (or a box every pose stays in, for a deforming mesh). */
    readonly bounds: ArrayLike<number>;
}

/** One instance in the GPU top level. */
export interface GpuTopLevelInstance {
    /** The mesh slot its tree is placed under ({@link GpuTopLevel.setMeshTree}). */
    readonly meshSlot: number;
    /** Returned in `BvhHit.instance`. Ignored (0xffffffff) for a merged record. */
    readonly instanceIndex: number;
    /** Object-to-world, column-major 4x4. */
    readonly objectToWorld: Mat4;
    /** Default true. */
    readonly visible?: boolean;
    /** Default true. */
    readonly contributesToGi?: boolean;
    /** Instance mask, 0-255; default 0xff (every ray). */
    readonly mask?: number;
    /** Material index, 0-65534 (`MAX_MATERIAL`), stored as `materialCode` (material + 1) in flags bits 16-31. Default none. A merged record can't have one. */
    readonly material?: number;
    /** A merged static group's record: identity transform, never moves. */
    readonly merged?: boolean;
}

export interface GpuTopLevelOptions {
    readonly label?: string;
    /** `worker` (default) builds shapes in a module worker; `main` on this thread (on a later turn). */
    readonly buildIn?: 'worker' | 'main';
    /** Read the refit's stats back for counters and warnings. Off by default: then nothing is ever read back. */
    readonly diagnostics?: boolean;
    /** Where a refit that had to be repaired is reported (once). Defaults to `console.warn`. */
    readonly warn?: (message: string) => void;
    /** Default {@link TOP_LEVEL_REBUILD_COST_RATIO}. */
    readonly rebuildCostRatio?: number;
    /** Default {@link QUALITY_CHECK_INTERVAL_MS}. */
    readonly qualityCheckIntervalMs?: number;
}

/** What changed in this frame besides the instances, for {@link GpuTopLevel.update}. */
export interface GpuTopLevelFrame {
    /** Deepest placed mesh tree, in levels, for the traversal-stack check. */
    readonly deepestMesh: number;
}

/** Counters for tests, stats and benchmarks. */
export interface GpuTopLevelCounters {
    shapesBuilt: number;
    qualityChecks: number;
    qualityRebuilds: number;
    lastBuildMs: number;
    lastSnapshotMs: number;
    refits: number;
    refitGaveUp: number;
    refitRepairs: number;
}

interface ArrivedShape {
    readonly shape: TopLevelShape;
    readonly generation: number;
}

interface CurrentShape {
    readonly recordCount: number;
    readonly nodeCount: number;
    readonly maxDepth: number;
    readonly generation: number;
}

const USAGE = { COPY_SRC: 0x4, COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80 } as const;
const hex = (value: number): string => `0x${(value >>> 0).toString(16)}u`;

/** The instance flags bits the gather passes from the table into the record: everything but the traced bits, which it sets. */
const RECORD_FLAGS_KEPT = ~(INSTANCE_FLAG.contributesToGi | INSTANCE_FLAG.visible) >>> 0;

/** The gather kernel: group 0 only, its own bindings. Everything it declares starts with `bvhGather`. */
export function topLevelGatherWgsl(): string {
    return /* wgsl */ `${BVH_INSTANCE_BOX_WGSL}
struct BvhGatherParams {
    recordCount: u32,
    meshCapacity: u32,
    instanceCapacity: u32,
    unused: u32,
}

// The tracer's top level: [header][instance records][nodes], as vec4u so no
// integer passes through a float register.
@group(0) @binding(0) var<storage, read_write> bvhGatherTopLevel: array<vec4u>;
// Two vec4u per mesh slot: (boundsMin bits, rootNodeOffset), (boundsMax bits, triangleOffset).
@group(0) @binding(1) var<storage, read> bvhGatherMeshes: array<vec4u>;
// The instance slot each record holds.
@group(0) @binding(2) var<storage, read> bvhGatherRecordSlots: array<u32>;
// The refit's input: two vec4f per record, (min.xyz, 0), (max.xyz, 0).
@group(0) @binding(3) var<storage, read_write> bvhGatherBoxes: array<vec4f>;
@group(0) @binding(4) var<uniform> bvhGatherParams: BvhGatherParams;
// Seven vec4 per instance slot (top-level-gpu.ts, TOP_LEVEL_INSTANCE_ENTRY_WORD).
@group(0) @binding(5) var<storage, read> bvhGatherInstances: array<vec4u>;

const BVH_GATHER_WORKGROUP_SIZE: u32 = ${GATHER_WORKGROUP_SIZE}u;
const BVH_GATHER_HEADER_VEC4S: u32 = ${TOP_LEVEL_HEADER_BYTES / 16}u;
const BVH_GATHER_RECORD_VEC4S: u32 = ${INSTANCE_RECORD_BYTES / 16}u;
const BVH_GATHER_ENTRY_VEC4S: u32 = ${TOP_LEVEL_INSTANCE_ENTRY_WORDS / 4}u;
const BVH_GATHER_BOX_VEC4S: u32 = ${PRIMITIVE_BOX_FLOATS / 4}u;
const BVH_GATHER_MESH_ABSENT: u32 = ${hex(TOP_LEVEL_MESH_ABSENT)};
const BVH_GATHER_FLOAT_MAX: f32 = 0x1.fffffep+127f;   // ${BVH_FLOAT_MAX}
const BVH_GATHER_FLAG_TRACED: u32 = ${INSTANCE_FLAG.contributesToGi | INSTANCE_FLAG.visible}u;
const BVH_GATHER_FLAGS_KEPT: u32 = ${hex(RECORD_FLAGS_KEPT)};

@compute @workgroup_size(${GATHER_WORKGROUP_SIZE})
fn bvhTopLevelGather(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
    // 2D dispatch, so very many instances stay under the per-dimension workgroup limit.
    let record = id.y * groups.x * BVH_GATHER_WORKGROUP_SIZE + id.x;
    if (record >= bvhGatherParams.recordCount) {
        return;
    }
    let slot = bvhGatherRecordSlots[record];
    let recordBase = BVH_GATHER_HEADER_VEC4S + record * BVH_GATHER_RECORD_VEC4S;
    let boxBase = record * BVH_GATHER_BOX_VEC4S;

    // Rays see the instance only if it is live, traced (visible and contributing) and its mesh tree is placed.
    var traced = slot < bvhGatherParams.instanceCapacity;
    var ids = vec4u(0u);
    let entryBase = slot * BVH_GATHER_ENTRY_VEC4S;
    if (traced) {
        ids = bvhGatherInstances[entryBase + 6u];
        traced = ids.w != 0u && (ids.z & BVH_GATHER_FLAG_TRACED) == BVH_GATHER_FLAG_TRACED && ids.x < bvhGatherParams.meshCapacity;
    }
    var meshLower = vec4u(0u);
    var meshUpper = vec4u(0u);
    if (traced) {
        meshLower = bvhGatherMeshes[ids.x * 2u];
        meshUpper = bvhGatherMeshes[ids.x * 2u + 1u];
        traced = meshLower.w != BVH_GATHER_MESH_ABSENT;
    }
    if (!traced) {
        // An empty box: the refit stores it where no ray reaches, so the record is never entered.
        bvhGatherTopLevel[recordBase] = vec4u(0u);
        bvhGatherTopLevel[recordBase + 1u] = vec4u(0u);
        bvhGatherTopLevel[recordBase + 2u] = vec4u(0u);
        bvhGatherTopLevel[recordBase + 3u] = vec4u(0u, 0u, ids.y, 0u);
        bvhGatherBoxes[boxBase] = vec4f(vec3f(BVH_GATHER_FLOAT_MAX), 0.0);
        bvhGatherBoxes[boxBase + 1u] = vec4f(vec3f(-BVH_GATHER_FLOAT_MAX), 0.0);
        return;
    }

    let row0 = bitcast<vec4f>(bvhGatherInstances[entryBase]);
    let row1 = bitcast<vec4f>(bvhGatherInstances[entryBase + 1u]);
    let row2 = bitcast<vec4f>(bvhGatherInstances[entryBase + 2u]);
    // World-to-object, inverted on the CPU in doubles: the same bits the CPU top level writes.
    bvhGatherTopLevel[recordBase] = bvhGatherInstances[entryBase + 3u];
    bvhGatherTopLevel[recordBase + 1u] = bvhGatherInstances[entryBase + 4u];
    bvhGatherTopLevel[recordBase + 2u] = bvhGatherInstances[entryBase + 5u];
    // The flags as packed on the CPU (mirrored, merged, mask bits 8-15, material bits 16-31), plus the traced bits.
    let flags = (ids.z & BVH_GATHER_FLAGS_KEPT) | BVH_GATHER_FLAG_TRACED;
    bvhGatherTopLevel[recordBase + 3u] = vec4u(meshLower.w, meshUpper.w, ids.y, flags);

    let box = bvhInstanceWorldBox(row0, row1, row2, bitcast<vec3f>(meshLower.xyz), bitcast<vec3f>(meshUpper.xyz));
    bvhGatherBoxes[boxBase] = vec4f(box.lower, 0.0);
    bvhGatherBoxes[boxBase + 1u] = vec4f(box.upper, 0.0);
}
`;
}

/**
 * Packs one instance into its table entry (`TOP_LEVEL_INSTANCE_ENTRY_WORDS`
 * words at `base`), and returns whether its transform can be inverted (a
 * singular one is written with `live` 0: rays never see it, as the CPU top
 * level skips it). The record's flags are written as {@link buildTopLevel}
 * writes them.
 */
export function packTopLevelInstance(instance: GpuTopLevelInstance, floats: Float32Array, words: Uint32Array, base: number, scratch = new Float64Array(12)): boolean {
    const entry = TOP_LEVEL_INSTANCE_ENTRY_WORD;
    const e = instance.objectToWorld;
    const merged = instance.merged === true;
    if (merged && !isIdentityTransform(e)) throwMergedTransformChange(instance.instanceIndex, 'GpuTopLevel');
    const mask = checkInstanceMask(instance.mask, instance.instanceIndex);
    const material = materialCode(instance.material, instance.instanceIndex);
    if (merged && material !== 0) throw new RangeError(`instance ${instance.instanceIndex}: a merged record has no material of its own (its triangles carry their sources')`);
    for (let row = 0; row < 3; row++) {
        const at = base + entry.objectToWorldRow0 + row * 4;
        floats[at] = e[row];
        floats[at + 1] = e[4 + row];
        floats[at + 2] = e[8 + row];
        floats[at + 3] = e[12 + row];
    }
    const determinant = invertAffineInto(e, scratch);
    const live = determinant !== 0;
    for (let k = 0; k < 12; k++) floats[base + entry.worldToObjectRow0 + k] = live ? scratch[k] : 0;
    words[base + entry.meshSlot] = instance.meshSlot;
    words[base + entry.instanceIndex] = merged ? MERGED_INSTANCE_INDEX : instance.instanceIndex >>> 0;
    words[base + entry.flags] =
        ((instance.contributesToGi === false ? 0 : INSTANCE_FLAG.contributesToGi) |
            (instance.visible === false ? 0 : INSTANCE_FLAG.visible) |
            (determinant < 0 ? INSTANCE_FLAG.mirrored : 0) |
            (merged ? INSTANCE_FLAG.merged : 0) |
            (mask << INSTANCE_MASK_SHIFT) |
            (material << MATERIAL_SHIFT)) >>>
        0;
    words[base + entry.live] = live ? 1 : 0;
    return live;
}

export class GpuTopLevel {
    static isSupported(device: GPUDevice): boolean {
        return GpuBvhRefit.isSupported(device);
    }

    readonly counters: GpuTopLevelCounters = {
        shapesBuilt: 0,
        qualityChecks: 0,
        qualityRebuilds: 0,
        lastBuildMs: 0,
        lastSnapshotMs: 0,
        refits: 0,
        refitGaveUp: 0,
        refitRepairs: 0,
    };

    private readonly device: GPUDevice;
    private readonly label: string;
    private readonly buildIn: 'worker' | 'main';
    private readonly warn: (message: string) => void;
    private readonly rebuildCostRatio: number;
    private readonly qualityCheckIntervalMs: number;
    private builder: TopLevelBuilder | null = null;
    private readonly refit: GpuBvhRefit;
    private gatherPipeline: GPUComputePipeline | null = null;
    private compiling: Promise<void> | null = null;

    private topLevelBuffer: GPUBuffer;
    private retired: GPUBuffer[] = [];
    private recordSlots: GPUBuffer;
    private primitiveBoxes: GPUBuffer;
    private meshTable: GPUBuffer;
    private meshMirror: Uint32Array<ArrayBuffer>;
    private meshTableDirty = true;
    private instanceTable: GPUBuffer;
    private instanceWords: Uint32Array<ArrayBuffer>;
    private instanceFloats: Float32Array<ArrayBuffer>;
    /** Object-to-world in doubles per slot, for shape builds (the table holds f32). */
    private transforms: Float64Array<ArrayBuffer>;
    /** Slots holding an instance (live or singular). */
    private readonly occupied = new Set<number>();
    private dirtyLow = Infinity;
    private dirtyHigh = -1;
    private instanceTableGrown = false;
    private readonly gatherParams: GPUBuffer;
    private gatherGroup: { group: GPUBindGroup; buffers: GPUBuffer[] } | null = null;
    /** Object-space mesh boxes by mesh slot, six floats each: the input of shape builds and checks. */
    private meshBounds = new Float32Array(16 * 6);
    private readonly scratch = new Float64Array(12);

    private current: CurrentShape = { recordCount: 0, nodeCount: 0, maxDepth: 0, generation: 0 };
    private arrived: ArrivedShape | null = null;
    private inFlight: 'build' | 'check' | null = null;
    private buildQueued = false;
    private lastRequestTime = -Infinity;
    private nextGeneration = 1;
    private buildFailure: Error | null = null;
    private idleWaiters: (() => void)[] = [];
    private seenStats: BvhRefitStats | null = null;
    private warnedGaveUp = false;
    private qualityRebuildsPaused = false;
    /** Instances were added or removed since the last update. */
    private instanceSetChanged = false;
    /** An instance moved, was hidden, shown or changed since the last update. */
    private instancesChanged = false;
    private bufferVersion = 0;
    private disposed = false;

    /** Relative SAH cost of the refit tree: from the last quality check, or the diagnostics readback when on. */
    latestRelativeCost = 0;

    constructor(device: GPUDevice, options: GpuTopLevelOptions = {}) {
        if (!GpuTopLevel.isSupported(device)) throw new Error('@pmndrs/bvh: the GPU top level needs the "subgroups" feature; use the CPU top level (buildTopLevel) without it');
        this.device = device;
        this.label = options.label ?? 'bvh';
        this.buildIn = options.buildIn ?? 'worker';
        this.warn = options.warn ?? ((message) => console.warn(message));
        this.rebuildCostRatio = options.rebuildCostRatio ?? TOP_LEVEL_REBUILD_COST_RATIO;
        this.qualityCheckIntervalMs = options.qualityCheckIntervalMs ?? QUALITY_CHECK_INTERVAL_MS;
        this.refit = new GpuBvhRefit(device, { label: `${this.label} top level refit`, diagnostics: options.diagnostics ?? false });
        // An empty top level: the header says 0 nodes, so every ray misses.
        this.topLevelBuffer = this.createTopLevelBuffer(TOP_LEVEL_HEADER_BYTES);
        device.queue.writeBuffer(this.topLevelBuffer, 0, new Uint32Array(TOP_LEVEL_HEADER_WORDS));
        this.recordSlots = this.createStorage('top level record slots', 64);
        this.primitiveBoxes = this.createStorage('top level instance boxes', 64);
        this.meshMirror = new Uint32Array(16 * TOP_LEVEL_MESH_ENTRY_WORDS).fill(TOP_LEVEL_MESH_ABSENT);
        this.meshTable = this.createStorage('top level mesh table', this.meshMirror.byteLength);
        const initialInstances = 16;
        this.instanceWords = new Uint32Array(initialInstances * TOP_LEVEL_INSTANCE_ENTRY_WORDS);
        this.instanceFloats = new Float32Array(this.instanceWords.buffer);
        this.transforms = new Float64Array(initialInstances * 16);
        this.instanceTable = this.createStorage('top level instances', this.instanceWords.byteLength);
        this.gatherParams = device.createBuffer({ label: `${this.label} top level gather params`, size: 16, usage: USAGE.UNIFORM | USAGE.COPY_DST });
    }

    /** The tracer's top-level buffer: header, instance records, nodes. Bind it as `bvhTraceTopLevel`. */
    get buffer(): GPUBuffer {
        return this.topLevelBuffer;
    }

    /** Increments when {@link buffer} is replaced (rebuild bind groups then). */
    get version(): number {
        return this.bufferVersion;
    }

    /** Records in the current shape (instances rays can see once traced, hidden ones included). */
    get instanceCount(): number {
        return this.current.recordCount;
    }

    /** Levels of the current shape. */
    get maxDepth(): number {
        return this.current.maxDepth;
    }

    /** True while a shape that adds or removes instances is being built or waits to be swapped in. Quality checks don't count. */
    get building(): boolean {
        return this.buildPending || this.arrived !== null;
    }

    /** GPU bytes this top level holds. */
    get memoryBytes(): number {
        return (
            this.topLevelBuffer.size +
            this.recordSlots.size +
            this.primitiveBoxes.size +
            this.meshTable.size +
            this.instanceTable.size +
            this.gatherParams.size +
            this.refit.memoryBytes
        );
    }

    /** Resolves once no shape build is pending (the last may still wait for the next `update` to be swapped in). */
    whenIdle(): Promise<void> {
        if (!this.buildPending) return Promise.resolve();
        return new Promise((resolve) => this.idleWaiters.push(resolve));
    }

    /** Compiles the gather and the refit, and starts the shape worker (so its module loads while they compile). */
    prepare(): Promise<void> {
        this.builder ??= createTopLevelBuilder({ worker: this.buildIn === 'main' ? 'never' : 'auto' });
        this.compiling ??= (async () => {
            const module = this.device.createShaderModule({ label: `${this.label} top level gather`, code: topLevelGatherWgsl() });
            const [gather] = await Promise.all([
                this.device.createComputePipelineAsync({ label: `${this.label} top level gather`, layout: 'auto', compute: { module, entryPoint: 'bvhTopLevelGather' } }),
                this.refit.prepare(),
            ]);
            this.gatherPipeline = gather;
        })();
        return this.compiling;
    }

    get ready(): boolean {
        return this.gatherPipeline !== null && this.refit.ready;
    }

    /** A mesh's object box, which shape builds use until its tree is placed (and after, unless the tree gives another). */
    setMeshBounds(meshSlot: number, bounds: ArrayLike<number>): void {
        if ((meshSlot + 1) * 6 > this.meshBounds.length) {
            const grown = new Float32Array(Math.max((meshSlot + 1) * 6, this.meshBounds.length * 2));
            grown.set(this.meshBounds);
            this.meshBounds = grown;
        }
        for (let index = 0; index < 6; index++) this.meshBounds[meshSlot * 6 + index] = bounds[index];
    }

    /** A mesh's tree was placed (or replaced), or released (`null`). The mesh table reaches the GPU at the next `update`. */
    setMeshTree(meshSlot: number, tree: PlacedMeshTree | null): void {
        if (tree) this.setMeshBounds(meshSlot, tree.bounds);
        if ((meshSlot + 1) * TOP_LEVEL_MESH_ENTRY_WORDS > this.meshMirror.length) {
            let capacity = this.meshMirror.length / TOP_LEVEL_MESH_ENTRY_WORDS;
            while (capacity <= meshSlot) capacity *= 2;
            const grown = new Uint32Array(capacity * TOP_LEVEL_MESH_ENTRY_WORDS).fill(TOP_LEVEL_MESH_ABSENT);
            grown.set(this.meshMirror);
            this.meshMirror = grown;
            // Destroyed at the next update, after the frames that used it were submitted.
            this.retired.push(this.meshTable);
            this.meshTable = this.createStorage('top level mesh table', grown.byteLength);
        }
        const base = meshSlot * TOP_LEVEL_MESH_ENTRY_WORDS;
        if (tree === null) {
            this.meshMirror.fill(TOP_LEVEL_MESH_ABSENT, base, base + TOP_LEVEL_MESH_ENTRY_WORDS);
        } else {
            const floats = new Float32Array(this.meshMirror.buffer, base * 4, TOP_LEVEL_MESH_ENTRY_WORDS);
            floats.set([tree.bounds[0], tree.bounds[1], tree.bounds[2]], 0);
            floats.set([tree.bounds[3], tree.bounds[4], tree.bounds[5]], 4);
            this.meshMirror[base + 3] = tree.nodeOffset;
            this.meshMirror[base + 7] = tree.triangleOffset;
        }
        this.meshTableDirty = true;
    }

    /**
     * Puts an instance in `slot` (adding it, or replacing what the slot held), or
     * removes it (`null`). Adding or removing needs a new shape (built off the
     * frame, so the instance appears a frame or two later); replacing one with
     * the same slot only rewrites its entry. Throws for a merged instance whose
     * transform isn't the identity, or that would replace a merged one with
     * another transform.
     */
    setInstance(slot: number, instance: GpuTopLevelInstance | null): void {
        this.checkLive();
        if (!Number.isInteger(slot) || slot < 0) throw new RangeError(`instance slot must be a non-negative integer, got ${slot}`);
        if (instance === null) {
            if (!this.occupied.delete(slot)) return;
            const base = slot * TOP_LEVEL_INSTANCE_ENTRY_WORDS;
            this.instanceWords.fill(0, base, base + TOP_LEVEL_INSTANCE_ENTRY_WORDS);
            this.markDirty(slot);
            this.instanceSetChanged = true;
            return;
        }
        if (this.occupied.has(slot) && this.isMerged(slot) && !isIdentityTransform(instance.objectToWorld)) {
            throwMergedTransformChange(this.instanceWords[slot * TOP_LEVEL_INSTANCE_ENTRY_WORDS + TOP_LEVEL_INSTANCE_ENTRY_WORD.instanceIndex], 'GpuTopLevel.setInstance');
        }
        this.ensureInstanceCapacity(slot);
        packTopLevelInstance(instance, this.instanceFloats, this.instanceWords, slot * TOP_LEVEL_INSTANCE_ENTRY_WORDS, this.scratch);
        for (let k = 0; k < 16; k++) this.transforms[slot * 16 + k] = instance.objectToWorld[k];
        if (!this.occupied.has(slot)) {
            this.occupied.add(slot);
            this.instanceSetChanged = true;
        }
        this.markDirty(slot);
    }

    /** Moves an instance. Throws for a merged static group's record: its transform is baked into its mesh. */
    setTransform(slot: number, objectToWorld: Mat4): void {
        this.checkLive();
        if (!this.occupied.has(slot)) throw new RangeError(`GpuTopLevel.setTransform: slot ${slot} holds no instance`);
        const base = slot * TOP_LEVEL_INSTANCE_ENTRY_WORDS;
        const words = this.instanceWords;
        const entry = TOP_LEVEL_INSTANCE_ENTRY_WORD;
        if (this.isMerged(slot)) throwMergedTransformChange(words[base + entry.instanceIndex], 'GpuTopLevel.setTransform');
        const flags = words[base + entry.flags];
        const material = materialFromFlags(flags);
        packTopLevelInstance(
            {
                meshSlot: words[base + entry.meshSlot],
                instanceIndex: words[base + entry.instanceIndex],
                objectToWorld,
                visible: (flags & INSTANCE_FLAG.visible) !== 0,
                contributesToGi: (flags & INSTANCE_FLAG.contributesToGi) !== 0,
                mask: (flags >>> INSTANCE_MASK_SHIFT) & 0xff,
                material: material === NO_MATERIAL ? undefined : material,
            },
            this.instanceFloats,
            words,
            base,
            this.scratch,
        );
        for (let k = 0; k < 16; k++) this.transforms[slot * 16 + k] = objectToWorld[k];
        this.markDirty(slot);
    }

    /** Whether `slot` holds a merged static group's record. */
    isMerged(slot: number): boolean {
        if (!this.occupied.has(slot)) return false;
        return (this.instanceWords[slot * TOP_LEVEL_INSTANCE_ENTRY_WORDS + TOP_LEVEL_INSTANCE_ENTRY_WORD.flags] & INSTANCE_FLAG.merged) !== 0;
    }

    /**
     * Brings the top level up to this frame: writes the changed instance and
     * mesh entries, swaps in an arrived shape, starts a build or quality check
     * if one is due, and records the refit pass (gather, then refit) when
     * anything it depends on changed. Call before anything traces in `encoder`.
     */
    update(encoder: GPUCommandEncoder, frame: GpuTopLevelFrame): void {
        this.checkLive();
        for (const buffer of this.retired) buffer.destroy();
        this.retired = [];

        this.refit.pollStats();
        this.collectDiagnostics();
        const setChanged = this.instanceSetChanged;
        const moved = this.instancesChanged || this.meshTableDirty || setChanged;
        this.instanceSetChanged = false;
        this.instancesChanged = false;
        if (setChanged || this.meshTableDirty) this.qualityRebuildsPaused = false;
        const checkDue = performance.now() - this.lastRequestTime >= this.qualityCheckIntervalMs;
        if (setChanged) this.request('build');
        else if (moved && checkDue && this.current.recordCount > 0 && !this.qualityRebuildsPaused) this.request('check');

        // The tables go first, so even a frame that throws below leaves no record pointing at a released tree.
        let changed = moved;
        if (this.meshTableDirty) {
            this.device.queue.writeBuffer(this.meshTable, 0, this.meshMirror);
            this.meshTableDirty = false;
            changed = true;
        }
        this.writeInstanceTable();
        this.writeGatherParams();
        let failure: unknown = this.buildFailure;
        this.buildFailure = null;
        if (this.arrived) {
            // Tried once. A shape that can't be used (a limit, or too deep for the trace stack) is dropped:
            // swap checks before writing anything, so the old shape stays whole and in use.
            const arrived = this.arrived;
            this.arrived = null;
            try {
                this.swap(arrived, frame.deepestMesh);
                changed = true;
            } catch (error) {
                failure ??= error;
                this.qualityRebuildsPaused = true;
            }
        }
        try {
            this.checkStack(this.current.maxDepth, frame.deepestMesh);
        } catch (error) {
            failure ??= error;
        }
        if (changed && this.current.recordCount > 0) this.encodeRefit(encoder);
        if (failure) throw failure;
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.builder?.dispose();
        this.refit.dispose();
        for (const buffer of [...this.retired, this.topLevelBuffer, this.recordSlots, this.primitiveBoxes, this.meshTable, this.instanceTable, this.gatherParams]) buffer.destroy();
        this.retired = [];
        for (const resolve of this.idleWaiters.splice(0)) resolve();
    }

    // -----------------------------------------------------------------------
    // Instances
    // -----------------------------------------------------------------------

    private markDirty(slot: number): void {
        this.dirtyLow = Math.min(this.dirtyLow, slot);
        this.dirtyHigh = Math.max(this.dirtyHigh, slot);
        this.instancesChanged = true;
    }

    private ensureInstanceCapacity(slot: number): void {
        const capacity = this.instanceWords.length / TOP_LEVEL_INSTANCE_ENTRY_WORDS;
        if (slot < capacity) return;
        let grownCapacity = capacity;
        while (grownCapacity <= slot) grownCapacity *= 2;
        const words = new Uint32Array(grownCapacity * TOP_LEVEL_INSTANCE_ENTRY_WORDS);
        words.set(this.instanceWords);
        this.instanceWords = words;
        this.instanceFloats = new Float32Array(words.buffer);
        const transforms = new Float64Array(grownCapacity * 16);
        transforms.set(this.transforms);
        this.transforms = transforms;
        this.retired.push(this.instanceTable);
        this.instanceTable = this.createStorage('top level instances', words.byteLength);
        this.instanceTableGrown = true;
    }

    /** Writes the dirty slots' entries (or the whole table after it grew) before this frame's commands run. */
    private writeInstanceTable(): void {
        if (this.instanceTableGrown) {
            this.device.queue.writeBuffer(this.instanceTable, 0, this.instanceWords);
            this.instanceTableGrown = false;
        } else if (this.dirtyHigh >= 0) {
            const first = this.dirtyLow * TOP_LEVEL_INSTANCE_ENTRY_WORDS;
            const count = (this.dirtyHigh + 1 - this.dirtyLow) * TOP_LEVEL_INSTANCE_ENTRY_WORDS;
            this.device.queue.writeBuffer(this.instanceTable, first * 4, this.instanceWords, first, count);
        }
        this.dirtyLow = Infinity;
        this.dirtyHigh = -1;
    }

    // -----------------------------------------------------------------------
    // Shapes
    // -----------------------------------------------------------------------

    private get buildPending(): boolean {
        return this.inFlight === 'build' || this.buildQueued;
    }

    private request(mode: 'build' | 'check'): void {
        if (this.inFlight !== null) {
            if (mode === 'build') this.buildQueued = true;
            return;
        }
        this.send(mode);
    }

    /** The build input: every occupied slot (hidden ones too, with `visible` 0 while not traced). */
    snapshot(): TopLevelBuildInput {
        const slots = Uint32Array.from([...this.occupied].sort((a, b) => a - b));
        const count = slots.length;
        const visible = new Uint8Array(count);
        const merged = new Uint8Array(count);
        const objectToWorld = new Float64Array(count * 16);
        const meshBounds = new Float32Array(count * 6);
        const entry = TOP_LEVEL_INSTANCE_ENTRY_WORD;
        const traced = INSTANCE_FLAG.contributesToGi | INSTANCE_FLAG.visible;
        let anyMerged = false;
        slots.forEach((slot, index) => {
            const base = slot * TOP_LEVEL_INSTANCE_ENTRY_WORDS;
            const flags = this.instanceWords[base + entry.flags];
            visible[index] = this.instanceWords[base + entry.live] !== 0 && (flags & traced) === traced ? 1 : 0;
            if (flags & INSTANCE_FLAG.merged) {
                merged[index] = 1;
                anyMerged = true;
            }
            objectToWorld.set(this.transforms.subarray(slot * 16, slot * 16 + 16), index * 16);
            const meshSlot = this.instanceWords[base + entry.meshSlot];
            if ((meshSlot + 1) * 6 <= this.meshBounds.length) meshBounds.set(this.meshBounds.subarray(meshSlot * 6, meshSlot * 6 + 6), index * 6);
        });
        return { slots, visible, objectToWorld, meshBounds, ...(anyMerged ? { merged } : {}) };
    }

    private send(mode: 'build' | 'check'): void {
        const start = performance.now();
        const input = this.snapshot();
        this.counters.lastSnapshotMs = performance.now() - start;
        this.lastRequestTime = performance.now();
        if (mode === 'check') this.counters.qualityChecks++;

        const generation = this.nextGeneration++;
        this.inFlight = mode;
        this.builder ??= createTopLevelBuilder({ worker: this.buildIn === 'main' ? 'never' : 'auto' });
        this.builder.request(mode, input, this.rebuildCostRatio, { current: this.current.generation, next: generation }).then(
            (result) => this.finish(mode, generation, result),
            (error: unknown) => {
                if (this.disposed) return;
                this.buildFailure = new Error(`@pmndrs/bvh: building the top level failed: ${error instanceof Error ? error.message : String(error)}`);
                this.finish(mode, generation, null);
            },
        );
    }

    private finish(mode: 'build' | 'check', generation: number, result: TopLevelResult | null): void {
        if (this.disposed) return;
        this.inFlight = null;
        if (result) {
            if (mode === 'check') this.latestRelativeCost = result.relativeCost;
            if (result.shape) {
                this.arrived = { shape: result.shape, generation };
                this.counters.lastBuildMs = result.workerMs;
                if (mode === 'check') this.counters.qualityRebuilds++;
            }
        }
        if (this.buildQueued) {
            // Instances came or went while the worker was busy: build again, so `whenIdle` only resolves
            // once the shape covers the instances as they are now.
            this.buildQueued = false;
            this.send('build');
            return;
        }
        for (const resolve of this.idleWaiters.splice(0)) resolve();
    }

    private checkStack(topLevelDepth: number, deepestMesh: number): void {
        const requiredStack = topLevelDepth + 1 + deepestMesh;
        if (topLevelDepth > 0 && requiredStack > TRACE_STACK_SIZE) {
            throw new RangeError(
                `@pmndrs/bvh: the trace stack holds ${TRACE_STACK_SIZE} entries, but this scene can need ${requiredStack} ` +
                    `(top level ${topLevelDepth} + 1 + deepest mesh tree ${deepestMesh}).`,
            );
        }
    }

    /** Puts an arrived shape in place. The checks come first, so a shape that can't be used leaves every buffer as it was. */
    private swap(arrived: ArrivedShape, deepestMesh: number): void {
        const { shape, generation } = arrived;
        this.checkStack(shape.maxDepth, deepestMesh);
        const recordCount = shape.recordSlots.length;
        const nodeCount = shape.nodeCount;
        const nodeByteOffset = TOP_LEVEL_HEADER_BYTES + recordCount * INSTANCE_RECORD_BYTES;
        this.ensureTopLevelBytes(nodeByteOffset + nodeCount * BVH_NODE_BYTES);

        const header = new Uint32Array(TOP_LEVEL_HEADER_WORDS);
        header[TOP_HEADER_WORD.instanceCount] = recordCount;
        header[TOP_HEADER_WORD.nodeCount] = nodeCount;
        header[TOP_HEADER_WORD.maxDepth] = shape.maxDepth;
        this.device.queue.writeBuffer(this.topLevelBuffer, 0, header);
        if (nodeCount > 0) this.device.queue.writeBuffer(this.topLevelBuffer, nodeByteOffset, shape.nodes, 0, nodeCount * BVH_NODE_BYTES);

        // Records are in tree order; record r holds the instance in slot recordSlots[r].
        const recordSlots = new Uint32Array(Math.max(1, recordCount));
        recordSlots.set(shape.recordSlots);
        if (this.recordSlots.size < recordSlots.byteLength) {
            this.retired.push(this.recordSlots);
            this.recordSlots = this.createStorage('top level record slots', recordSlots.byteLength * 2);
        }
        this.device.queue.writeBuffer(this.recordSlots, 0, recordSlots);
        const boxBytes = Math.max(1, recordCount) * PRIMITIVE_BOX_FLOATS * 4;
        if (this.primitiveBoxes.size < boxBytes) {
            this.retired.push(this.primitiveBoxes);
            this.primitiveBoxes = this.createStorage('top level instance boxes', boxBytes * 2);
        }

        this.refit.setShape({ nodeWordOffset: nodeByteOffset / 4, nodeCount, parentSlots: shape.parentSlots, leafSlots: shape.leafSlots, generation });
        this.current = { recordCount, nodeCount, maxDepth: shape.maxDepth, generation };
        this.writeGatherParams();
        this.counters.shapesBuilt++;
    }

    private collectDiagnostics(): void {
        const stats = this.refit.latestStats;
        if (!stats || stats === this.seenStats) return;
        this.seenStats = stats;
        if (stats.generation === this.current.generation) this.latestRelativeCost = stats.relativeCost;
        if (stats.incomplete === 0) return;
        this.counters.refitGaveUp += stats.incomplete;
        this.counters.refitRepairs += stats.repairs;
        if (this.warnedGaveUp) return;
        this.warnedGaveUp = true;
        this.warn('@pmndrs/bvh: the top-level refit gave up waiting for a box on this GPU and refit the tree again in a slower pass (the result is correct). Counted in GpuTopLevel.counters.');
    }

    // -----------------------------------------------------------------------
    // The refit pass
    // -----------------------------------------------------------------------

    private encodeRefit(encoder: GPUCommandEncoder): void {
        const gather = this.gatherPipeline;
        if (!gather || !this.refit.ready) throw new Error('@pmndrs/bvh: call GpuTopLevel.prepare() before update()');
        const pass = encoder.beginComputePass({ label: `${this.label} top level refit` });
        pass.setPipeline(gather);
        pass.setBindGroup(0, this.gatherBindGroup(gather));
        const workgroups = Math.ceil(this.current.recordCount / GATHER_WORKGROUP_SIZE);
        const x = Math.min(workgroups, this.device.limits.maxComputeWorkgroupsPerDimension);
        pass.dispatchWorkgroups(x, Math.ceil(workgroups / x));
        this.refit.encode(pass, this.topLevelBuffer, this.primitiveBoxes);
        pass.end();
        this.refit.copyStats(encoder);
        this.counters.refits++;
    }

    private gatherBindGroup(gather: GPUComputePipeline): GPUBindGroup {
        const buffers = [this.topLevelBuffer, this.meshTable, this.recordSlots, this.primitiveBoxes, this.gatherParams, this.instanceTable];
        const cached = this.gatherGroup;
        if (cached && cached.buffers.every((buffer, index) => buffer === buffers[index])) return cached.group;
        const group = this.device.createBindGroup({
            label: `${this.label} top level gather`,
            layout: gather.getBindGroupLayout(0),
            entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        });
        this.gatherGroup = { group, buffers };
        return group;
    }

    private writeGatherParams(): void {
        this.device.queue.writeBuffer(
            this.gatherParams,
            0,
            new Uint32Array([this.current.recordCount, this.meshMirror.length / TOP_LEVEL_MESH_ENTRY_WORDS, this.instanceWords.length / TOP_LEVEL_INSTANCE_ENTRY_WORDS, 0]),
        );
    }

    // -----------------------------------------------------------------------
    // Buffers
    // -----------------------------------------------------------------------

    private ensureTopLevelBytes(bytes: number): void {
        if (bytes <= this.topLevelBuffer.size) return;
        const limits = this.device.limits;
        const limitBytes = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
        if (bytes > limitBytes) {
            const limit = limits.maxStorageBufferBindingSize <= limits.maxBufferSize ? 'maxStorageBufferBindingSize' : 'maxBufferSize';
            throw new RangeError(`@pmndrs/bvh: the top level needs ${limit} >= ${bytes}, the device allows ${limitBytes}`);
        }
        this.retired.push(this.topLevelBuffer);
        const cap = Math.floor(limitBytes / 16) * 16;
        this.topLevelBuffer = this.createTopLevelBuffer(Math.min(cap, Math.max(bytes, this.topLevelBuffer.size * 2)));
        this.bufferVersion++;
    }

    private createTopLevelBuffer(size: number): GPUBuffer {
        return this.device.createBuffer({
            label: `${this.label} trace top level`,
            size: Math.ceil(size / 16) * 16,
            // COPY_SRC lets tests and debug tools read the tree back.
            usage: USAGE.STORAGE | USAGE.COPY_DST | USAGE.COPY_SRC,
        });
    }

    private createStorage(name: string, size: number): GPUBuffer {
        return this.device.createBuffer({ label: `${this.label} ${name}`, size: Math.ceil(size / 16) * 16, usage: USAGE.STORAGE | USAGE.COPY_DST });
    }

    private checkLive(): void {
        if (this.disposed) throw new Error('@pmndrs/bvh: the GPU top level was disposed');
    }
}

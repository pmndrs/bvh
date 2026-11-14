/**
 * First mesh trees from the GPU, for `SceneTrees` (ported from ddgi-core's
 * `platform/trace/gpu-mesh-trees.ts`, T2.12 at load and T3.2 at runtime):
 * every mesh an update brings is built on the GPU with H-PLOC, so rays can see
 * it a few frames later instead of after tinybvh's time in a worker (seconds
 * for a whole scene). tinybvh still builds every mesh, and its tree replaces
 * the GPU's when it arrives (the "swap").
 *
 * **One batch for all of an update's meshes.** H-PLOC builds every mesh of a
 * batch with the same dispatches; building meshes one by one is dispatch-bound
 * (ddgi NOTES, LBVH: 97 ms against 32-42 ms for Sponza). The meshes are split
 * into more batches only when one doesn't fit the device's buffers
 * (`planGpuHplocBatches`).
 *
 * **A cap on a batch's bytes** (`maxBatchBytes`, off by default here, 256 MiB
 * through `SceneTrees`' `gpuMaxBatchBytes`: the P9
 * peak-memory lever, docs/internals/design/bvh-builders.md "Build peak memory"). The
 * meshes are split into batches of at most that many GPU bytes each (inputs,
 * scratch and outputs), built one after another. Every capped batch of this
 * builder, whichever `build` asked for it (the load, a deforming mesh's own
 * batch, a runtime add, a drift rebuild), goes through **one queue**: one
 * batch builds at a time, on one scratch pool (`GpuHplocScratchPool`) sized
 * for the largest batch queued (made again, larger, between batches when a
 * bigger one comes), freed once the queue is empty. Each batch starts only
 * once the previous batch's outputs are let go (copied into the scene's
 * buffers and released, or adopted). So the peak is about the scene's
 * storage plus one batch, not the whole scene's scratch and outputs at once.
 * `SceneTrees` lets a capped batch go on its own, without waiting for an
 * update, whenever it can (see its `gpuMaxBatchBytes`). A mesh bigger than
 * the cap builds alone. Undo: leave `maxBatchBytes` unset (or `Infinity`).
 *
 * **Its own submit, and one small readback** of the stats (node count, depth,
 * refused or not: a few words per mesh), never the trees. The readback must
 * come after the build on the queue, and the caller's encoder may be
 * submitted at any time, so the build gets an encoder of its own and is
 * submitted at once. CPU geometry is uploaded with `queue.writeBuffer`, which
 * lands before that submit.
 *
 * Deforming meshes also need their tree's layout on the CPU (the refit takes
 * its shape from it): their builds ask for `readLayouts`, which reads their
 * batch's trees back once, after the stats; `SceneTrees` puts them in a batch
 * of their own.
 *
 * Without subgroups there is no GPU builder: every mesh waits for tinybvh (the
 * declared fallback).
 */

import { GpuHplocBuilder, hplocScratchPoolBytes, planGpuHplocBatches, type GpuHplocBatch, type GpuHplocScratchPool } from '../gpu-hploc.ts';
import type { GpuLayoutMeshBvh } from '../gpu-layout.ts';
import type { GpuMeshGeometry, GpuMeshTreeRange, GpuTreeBuffers } from '../mesh-input.ts';
import { PassTimestamps } from '../gpu-timestamps.ts';

/** A mesh to build: its slot and version, and its geometry, read once when the build starts. */
export interface GpuTreeRequest {
    readonly slot: number;
    readonly generation: number;
    readonly geometry: GpuMeshGeometry;
}

/** Options of one {@link FirstTreeBuilder.build}. */
export interface GpuTreeBuildOptions {
    /** Read every usable tree's layout back too (deforming meshes' refit needs it), once, after the stats. Default false. */
    readonly readLayouts?: boolean;
    /**
     * Cap each batch's GPU bytes (inputs, scratch and outputs) and build the
     * batches one after another, each once the previous one's outputs are let
     * go, in the builder's one queue with every other capped build (see the
     * file comment). Default (or `Infinity`): no cap, every batch in one
     * submit. `SceneTrees` passes its `gpuMaxBatchBytes` (256 MiB unless set).
     */
    readonly maxBatchBytes?: number;
    /** Asked before each capped batch that waited in the queue: true stops it (its meshes come back refused). */
    readonly cancelled?: () => boolean;
    /** Called when the builds' GPU bytes change (a capped batch allocated, the pool made or freed), so the caller can sample its peak. */
    readonly onMemoryChange?: () => void;
    /** Time each batch's compute pass with timestamp queries, where the device has `timestamp-query` (the result's `gpuMs`). Default false. */
    readonly timestamps?: boolean;
    /** Called when a batch is submitted (a capped batch can wait in the queue first), with its meshes. */
    readonly onBatchStart?: (members: readonly GpuTreeRequest[]) => void;
}

/**
 * The output buffers of one batch, shared by its trees. Each tree is copied
 * out of them into the scene's buffers, served from them in place (when the
 * scene adopts them), or dropped; the buffers are freed once every tree has
 * let go of them.
 */
export class GpuTreeOutputs {
    private readonly trees: GpuTreeBuffers;
    private users: number;
    private destroyed = false;
    /** The node and triangle buffers became a storage's own (`GrowableStorage.adopt`): only the parents are left here. */
    private adopted = false;
    private claimed = false;
    private resolveLetGo: () => void = () => {};
    /** Resolves once the buffers are destroyed or adopted: what a capped build waits for before its next batch. */
    readonly letGo: Promise<void> = new Promise((resolve) => (this.resolveLetGo = resolve));

    constructor(trees: GpuTreeBuffers, users: number) {
        this.trees = trees;
        this.users = users;
        if (users === 0) this.destroy();
    }

    get nodes(): GPUBuffer {
        return this.trees.nodes;
    }

    get triangles(): GPUBuffer {
        return this.trees.triangles;
    }

    /** The batch's node format; `SceneTrees` takes only `f32` batches (it quantizes on the way in). */
    get nodeFormat(): 'f32' | 'quantized' {
        return this.trees.nodeFormat ?? 'f32';
    }

    /** Every tree of the batch, refused ones included (their `problem` is set). */
    get meshes(): readonly GpuMeshTreeRange[] {
        return this.trees.meshes;
    }

    /** GPU bytes held until every tree is released (0 once adopted: the storage counts them then). */
    get bytes(): number {
        if (this.destroyed || this.adopted) return 0;
        return this.trees.nodes.size + this.trees.triangles.size + this.trees.parents.size;
    }

    /**
     * The caller now counts these bytes (`SceneTrees.memory.gpuBuildOutputs`).
     * Until then the build that made them counts them in its `buildingBytes`,
     * so they are counted somewhere from their allocation on.
     */
    claim(): void {
        this.claimed = true;
    }

    /** {@link bytes} while nobody has {@link claim}ed them, else 0. */
    get unclaimedBytes(): number {
        return this.claimed ? 0 : this.bytes;
    }

    get isDestroyed(): boolean {
        return this.destroyed;
    }

    get isAdopted(): boolean {
        return this.adopted;
    }

    /** One tree is done with the buffers (copied out in a frame submitted since, swapped, or dropped). */
    release(): void {
        if (this.destroyed) return;
        this.users--;
        if (this.users <= 0) this.destroy();
    }

    /**
     * The node and triangle buffers now belong to a storage that adopted them:
     * the parents are freed at once (the walk never reads them), and nothing
     * here destroys the other two.
     */
    markAdopted(): void {
        if (this.destroyed || this.adopted) return;
        this.adopted = true;
        this.trees.parents.destroy();
        this.resolveLetGo();
    }

    /** Frees the buffers whatever still uses them (the scene is being disposed). Adopted buffers are left to their storage. */
    destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
        this.resolveLetGo();
        if (this.adopted) return;
        this.trees.destroy();
    }
}

/** One mesh's tree built on the GPU, inside its batch's output buffers. */
export interface GpuBuiltTree {
    readonly slot: number;
    readonly generation: number;
    readonly outputs: GpuTreeOutputs;
    /** Where the tree sits in `outputs`, its counts, depth and box. */
    readonly tree: GpuMeshTreeRange;
    /** The tree's layout on the CPU, when the build asked for `readLayouts` and the readback worked; else null. */
    readonly layout: GpuLayoutMeshBvh | null;
}

/** A mesh the GPU didn't give a usable tree: it waits for tinybvh's. */
export interface GpuRefusedTree {
    readonly slot: number;
    readonly generation: number;
    readonly reason: string;
}

/** What one batch gave, once its stats were read back. */
export interface GpuTreeBatchResult {
    readonly built: readonly GpuBuiltTree[];
    readonly refused: readonly GpuRefusedTree[];
    /** Milliseconds from the submit to the stats on the CPU (the build plus the readback's latency). */
    readonly buildMs: number;
    /** GPU milliseconds of the batch's compute pass, from timestamps (`timestamps: true` on a device with `timestamp-query`), else null or absent. */
    readonly gpuMs?: number | null;
    /** Why the whole batch failed (it couldn't start, or its readback failed), or null. Its meshes are then all in `refused`. */
    readonly failure: string | null;
}

/** What {@link FirstTreeBuilder.build} started. */
export interface GpuTreeBuilds {
    /** Meshes too big for any batch on this device (or for H-PLOC): they wait for tinybvh. */
    readonly notBuilt: readonly GpuRefusedTree[];
    /** One promise per batch, in order. They never reject: a failed batch reports its meshes as refused. */
    readonly batches: readonly Promise<GpuTreeBatchResult>[];
    /**
     * GPU bytes the batches hold while they build (inputs, scratch, outputs),
     * until their outputs are claimed ({@link GpuTreeOutputs.claim}) or freed.
     * Read it again as batches finish: a capped build's changes batch by batch
     * (the shared pool is counted once, by the build whose batch has it).
     */
    readonly buildingBytes: number;
    /**
     * Slots whose geometry is read after `build` returns (a capped build's
     * batches that wait in the queue upload theirs when they start): their
     * arrays must not be moved into a worker before then.
     */
    readonly readsGeometryLater?: ReadonlySet<number>;
}

/** Builds first trees for many meshes at once. `SceneTrees`' only GPU builder is H-PLOC ({@link GpuFirstTrees}); tests give a stand-in. */
export interface FirstTreeBuilder {
    /** Compiles the builder's kernels. */
    prepare(): Promise<void>;
    /** Starts building every request's tree; see {@link GpuTreeBuilds}. Needs `prepare` to have finished. */
    build(requests: readonly GpuTreeRequest[], options?: GpuTreeBuildOptions): GpuTreeBuilds;
}

/** What {@link GpuFirstTrees} uses of {@link GpuHplocBuilder} (tests give a stand-in). */
export type HplocBatchBuilder = Pick<GpuHplocBuilder, 'prepare' | 'createBatch' | 'createScratchPool' | 'readLayouts'>;

/** One capped batch waiting in the queue, or building. */
interface QueuedBatch {
    readonly members: readonly GpuTreeRequest[];
    readonly options: GpuTreeBuildOptions;
    /** The build it belongs to: whose `buildingBytes` count its bytes. */
    readonly owner: object;
    readonly resolve: (result: GpuTreeBatchResult) => void;
}

/** H-PLOC first trees ({@link FirstTreeBuilder}) on a device with subgroups. */
export class GpuFirstTrees implements FirstTreeBuilder {
    static isSupported(device: GPUDevice): boolean {
        return GpuHplocBuilder.isSupported(device);
    }

    private readonly device: GPUDevice;
    private readonly hploc: HplocBatchBuilder;
    private readonly label: string;
    /** The trees' triangle format, for every batch. */
    private readonly buildOptions: { readonly triangleLeaves?: 'indexed' };
    /** Capped batches not started yet, in order (every build's: one batch builds at a time). */
    private readonly queue: QueuedBatch[] = [];
    /** A capped batch is building, or its outputs are not let go yet. */
    private busy = false;
    private pool: GpuHplocScratchPool | null = null;
    /** The build whose `buildingBytes` counts the pool: the one with the batch building or next in the queue. */
    private poolOwner: object | null = null;
    /** The capped batch building (its outputs, until claimed), and the build it belongs to. */
    private current: { readonly owner: object; readonly bytes: () => number } | null = null;

    /** `hploc` is for tests: an H-PLOC builder stand-in. */
    constructor(device: GPUDevice, options: { readonly label?: string; readonly hploc?: HplocBatchBuilder; readonly triangleLeaves?: 'records' | 'indexed' } = {}) {
        this.buildOptions = options.triangleLeaves === 'indexed' ? { triangleLeaves: 'indexed' } : {};
        this.device = device;
        this.label = `${options.label ?? 'bvh'} mesh trees`;
        this.hploc = options.hploc ?? new GpuHplocBuilder(device, { label: this.label });
    }

    prepare(): Promise<void> {
        return this.hploc.prepare();
    }

    /** Capped batches queued and not started (for tests and diagnostics). */
    get queuedBatches(): number {
        return this.queue.length;
    }

    build(requests: readonly GpuTreeRequest[], options: GpuTreeBuildOptions = {}): GpuTreeBuilds {
        const geometries = requests.map((request) => request.geometry);
        const plan = planGpuHplocBatches(this.device, geometries, { maxBatchBytes: options.maxBatchBytes, ...this.buildOptions });
        const limitBytes = Math.min(this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize);
        const notBuilt = plan.tooBig.map((index) => ({
            slot: requests[index].slot,
            generation: requests[index].generation,
            reason: `too big for a GPU build on this device (buffers of at most ${limitBytes} bytes)`,
        }));
        if (options.maxBatchBytes !== undefined && options.maxBatchBytes !== Infinity) {
            return this.enqueue(notBuilt, plan.batches.map((indices) => indices.map((index) => requests[index])), options);
        }

        // Every batch goes into one encoder and one submit; each then reads its own stats back.
        const encoder = this.device.createCommandEncoder({ label: this.label });
        const recorded: { members: GpuTreeRequest[]; finish: () => Promise<GpuTreeBuffers>; pass: number | null }[] = [];
        const failed: { members: GpuTreeRequest[]; reason: string }[] = [];
        const timer = options.timestamps ? PassTimestamps.create(this.device, plan.batches.length, this.label) : null;
        let buildingBytes = 0;
        for (const indices of plan.batches) {
            const members = indices.map((index) => requests[index]);
            try {
                const batch = this.hploc.createBatch(members.map((member) => member.geometry), this.buildOptions);
                const writes = timer?.writes();
                batch.encode(encoder, writes);
                buildingBytes += batch.scratchBytes + batch.outputBytes;
                recorded.push({ members, finish: () => batch.finish(), pass: writes ? writes.beginningOfPassWriteIndex! / 2 : null });
            } catch (error) {
                failed.push({ members, reason: `the GPU mesh tree build could not start: ${describe(error)}` });
            }
        }
        timer?.resolveInto(encoder);
        const submittedAt = performance.now();
        this.device.queue.submit([encoder.finish()]);
        for (const { members } of recorded) options.onBatchStart?.(members);
        const times = readTimes(timer);

        const batches = recorded.map(async ({ members, finish, pass }) => {
            const gpuMs = times.then((ms) => (pass === null ? null : (ms[pass] ?? null)));
            return (await this.collect(members, finish, submittedAt, options, gpuMs)).result;
        });
        for (const { members, reason } of failed) batches.push(Promise.resolve({ built: [], refused: refuseAll(members, reason), buildMs: 0, failure: reason }));
        return { notBuilt, batches, buildingBytes };
    }

    /**
     * The capped path: the build's batches join the queue, and the queue
     * starts the first if nothing is building (synchronously, so its arrays
     * are read before `build` returns).
     */
    private enqueue(notBuilt: readonly GpuRefusedTree[], groups: readonly (readonly GpuTreeRequest[])[], options: GpuTreeBuildOptions): GpuTreeBuilds {
        const owner = {};
        const items: QueuedBatch[] = [];
        const batches = groups.map(
            (members) =>
                new Promise<GpuTreeBatchResult>((resolve) => {
                    items.push({ members, options, owner, resolve });
                }),
        );
        this.queue.push(...items);
        this.pump();
        // A getter (never spread: that would read it once): buildingBytes changes batch by batch.
        const bytes = () => this.bytesOf(owner);
        return {
            notBuilt,
            batches,
            get buildingBytes() {
                return bytes();
            },
            readsGeometryLater: new Set(items.filter((item) => this.queue.includes(item)).flatMap((item) => item.members.map((member) => member.slot))),
        };
    }

    private bytesOf(owner: object): number {
        let bytes = 0;
        if (this.pool && this.poolOwner === owner) bytes += this.pool.bytes;
        if (this.current?.owner === owner) bytes += this.current.bytes();
        return bytes;
    }

    /** Starts the next queued batch if none is building. */
    private pump(): void {
        if (this.busy) return;
        const next = this.queue.shift();
        if (!next) return;
        this.busy = true;
        void this.run(next);
    }

    /**
     * Builds one queued batch, resolves its promise (never rejects), and once
     * its outputs are let go starts the next. Synchronous up to the submit.
     */
    private async run(item: QueuedBatch): Promise<void> {
        let outputs: GpuTreeOutputs | null = null;
        let result: GpuTreeBatchResult;
        try {
            const step = await this.start(item);
            result = step.result;
            outputs = step.outputs;
        } catch (error) {
            const failure = `the GPU mesh tree build failed: ${describe(error)}`;
            result = { built: [], refused: refuseAll(item.members, failure), buildMs: 0, failure };
        }
        // The batch's stats are read, so the GPU is done with the pool: free it when nothing else is queued.
        if (this.queue.length === 0) {
            this.pool?.destroy();
            this.pool = null;
            this.poolOwner = null;
        } else {
            this.poolOwner = this.queue[0].owner;
        }
        if (!outputs) this.current = null;
        item.resolve(result);
        item.options.onMemoryChange?.();
        if (outputs) await outputs.letGo;
        this.current = null;
        this.busy = false;
        this.pump();
    }

    /** Uploads, records and submits one batch at once. Never throws: a batch that can't start comes back refused. */
    private start(item: QueuedBatch): Promise<{ result: GpuTreeBatchResult; outputs: GpuTreeOutputs | null }> {
        const { members, options, owner } = item;
        const refused = (failure: string) => Promise.resolve({ result: { built: [], refused: refuseAll(members, failure), buildMs: 0, failure }, outputs: null });
        if (options.cancelled?.()) return refused('the GPU mesh tree build was cancelled');
        let batch: GpuHplocBatch | null = null;
        try {
            const geometries = members.map((member) => member.geometry);
            const pool = this.poolFor(geometries);
            this.poolOwner = owner;
            batch = this.hploc.createBatch(geometries, this.buildOptions, pool);
            const building = batch;
            this.current = { owner, bytes: () => building.outputBytes };
            options.onMemoryChange?.();
            const encoder = this.device.createCommandEncoder({ label: this.label });
            const timer = options.timestamps ? PassTimestamps.create(this.device, 1, this.label) : null;
            batch.encode(encoder, timer?.writes());
            timer?.resolveInto(encoder);
            const submittedAt = performance.now();
            this.device.queue.submit([encoder.finish()]);
            options.onBatchStart?.(members);
            const gpuMs = readTimes(timer).then((ms) => ms[0] ?? null);
            // From finish on, the outputs are counted until the caller claims them.
            return this.collect(members, () => building.finish(), submittedAt, options, gpuMs, (outputs) => (this.current = { owner, bytes: () => outputs.unclaimedBytes }));
        } catch (error) {
            batch?.destroy();
            this.current = null;
            return refused(`the GPU mesh tree build could not start: ${describe(error)}`);
        }
    }

    /**
     * The pool for a batch: the one there is if it holds the batch, else a new
     * one (the old one freed: the batch before has finished) sized for this
     * batch and every one still queued.
     */
    private poolFor(geometries: readonly GpuMeshGeometry[]): GpuHplocScratchPool {
        if (this.pool) {
            const needs = hplocScratchPoolBytes([geometries], this.pool.mortonCodes);
            const sizes = this.pool.sizes;
            if ((Object.keys(needs) as (keyof typeof needs)[]).every((name) => needs[name] <= sizes[name])) return this.pool;
            this.pool.destroy();
            this.pool = null;
        }
        this.pool = this.hploc.createScratchPool([geometries, ...this.queue.map((queued) => queued.members.map((member) => member.geometry))]);
        return this.pool;
    }

    /** Reads a submitted batch's stats (and layouts when asked) and splits its trees into built and refused. */
    private async collect(
        members: readonly GpuTreeRequest[],
        finish: () => Promise<GpuTreeBuffers>,
        submittedAt: number,
        options: GpuTreeBuildOptions,
        gpuTime: Promise<number | null>,
        onOutputs?: (outputs: GpuTreeOutputs) => void,
    ): Promise<{ result: GpuTreeBatchResult; outputs: GpuTreeOutputs | null }> {
        let trees: GpuTreeBuffers;
        try {
            trees = await finish();
        } catch (error) {
            const failure = `the GPU mesh tree build failed: ${describe(error)}`;
            return { result: { built: [], refused: refuseAll(members, failure), buildMs: performance.now() - submittedAt, gpuMs: await gpuTime, failure }, outputs: null };
        }
        const buildMs = performance.now() - submittedAt;
        const gpuMs = await gpuTime;
        const usable = trees.meshes.filter((tree) => tree.problem === null).length;
        const outputs = new GpuTreeOutputs(trees, usable);
        onOutputs?.(outputs);
        // A failed layout readback leaves the trees usable for tracing; only the refit goes without.
        let layouts: (GpuLayoutMeshBvh | null)[] = [];
        if (options.readLayouts) layouts = await this.hploc.readLayouts(trees).catch(() => []);
        const built: GpuBuiltTree[] = [];
        const refused: GpuRefusedTree[] = [];
        trees.meshes.forEach((tree, index) => {
            const { slot, generation } = members[index];
            if (tree.problem === null) built.push({ slot, generation, outputs, tree, layout: layouts[index] ?? null });
            else refused.push({ slot, generation, reason: tree.problem });
        });
        return { result: { built, refused, buildMs, gpuMs, failure: null }, outputs };
    }
}

/** A measurement's milliseconds per pass, once read (none without a timer, or if the read fails); the timer is freed after. */
function readTimes(timer: PassTimestamps | null): Promise<number[]> {
    if (!timer) return Promise.resolve([]);
    return timer
        .read()
        .catch((): number[] => [])
        .finally(() => timer.destroy());
}

function refuseAll(members: readonly GpuTreeRequest[], reason: string): GpuRefusedTree[] {
    return members.map(({ slot, generation }) => ({ slot, generation, reason }));
}

function describe(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

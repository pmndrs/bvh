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
 * **A cap on a batch's bytes** (`maxBatchBytes`, off by default: the P9
 * peak-memory lever, docs/design/bvh-builders.md "Build peak memory"). The
 * meshes are split into batches of at most that many GPU bytes each (inputs,
 * scratch and outputs), built one after another: one scratch pool
 * (`GpuHplocScratchPool`) is allocated at the largest batch's size and freed
 * after the last, and each batch starts only once the previous batch's
 * outputs are let go (copied into the scene's buffers and released, or
 * adopted). So the peak is about the scene's storage plus one batch, not the
 * whole scene's scratch and outputs at once. The batches after the first wait
 * for the caller's updates (`SceneTrees.update` copies a batch out, and the
 * next update releases it), so the caller must keep calling it. A mesh bigger
 * than the cap builds alone. Undo: leave `maxBatchBytes` unset.
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

import { GpuHplocBuilder, planGpuHplocBatches, type GpuHplocBatch, type GpuHplocScratchPool } from '../gpu-hploc.ts';
import type { GpuLayoutMeshBvh } from '../gpu-layout.ts';
import type { GpuMeshGeometry, GpuMeshTreeRange, GpuTreeBuffers } from '../mesh-input.ts';

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
     * go (see the file comment). Default: no cap, every batch in one submit.
     */
    readonly maxBatchBytes?: number;
    /** Asked before each capped batch after the first: true stops the rest (they come back refused). */
    readonly cancelled?: () => boolean;
    /** Called when the builds' GPU bytes change (a capped batch allocated, the pool freed), so the caller can sample its peak. */
    readonly onMemoryChange?: () => void;
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

    /** Every tree of the batch, refused ones included (their `problem` is set). */
    get meshes(): readonly GpuMeshTreeRange[] {
        return this.trees.meshes;
    }

    /** GPU bytes held until every tree is released (0 once adopted: the storage counts them then). */
    get bytes(): number {
        if (this.destroyed || this.adopted) return 0;
        return this.trees.nodes.size + this.trees.triangles.size + this.trees.parents.size;
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
     * until their stats are read. Read it again as batches finish: a capped
     * build's changes batch by batch (its pool, plus the batch building).
     */
    readonly buildingBytes: number;
    /**
     * Slots whose geometry is read after `build` returns (a capped build's
     * later batches upload theirs when they start): their arrays must not be
     * moved into a worker before then.
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

/** H-PLOC first trees ({@link FirstTreeBuilder}) on a device with subgroups. */
export class GpuFirstTrees implements FirstTreeBuilder {
    static isSupported(device: GPUDevice): boolean {
        return GpuHplocBuilder.isSupported(device);
    }

    private readonly device: GPUDevice;
    private readonly hploc: GpuHplocBuilder;
    private readonly label: string;

    constructor(device: GPUDevice, options: { readonly label?: string } = {}) {
        this.device = device;
        this.label = `${options.label ?? 'bvh'} mesh trees`;
        this.hploc = new GpuHplocBuilder(device, { label: this.label });
    }

    prepare(): Promise<void> {
        return this.hploc.prepare();
    }

    build(requests: readonly GpuTreeRequest[], options: GpuTreeBuildOptions = {}): GpuTreeBuilds {
        const geometries = requests.map((request) => request.geometry);
        const plan = planGpuHplocBatches(this.device, geometries, { maxBatchBytes: options.maxBatchBytes });
        const limitBytes = Math.min(this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize);
        const notBuilt = plan.tooBig.map((index) => ({
            slot: requests[index].slot,
            generation: requests[index].generation,
            reason: `too big for a GPU build on this device (buffers of at most ${limitBytes} bytes)`,
        }));
        if (options.maxBatchBytes !== undefined) {
            const sequence = this.buildInSequence(plan.batches.map((indices) => indices.map((index) => requests[index])), options);
            // Not a spread: that would read buildingBytes once, and it changes batch by batch.
            return {
                notBuilt,
                batches: sequence.batches,
                get buildingBytes() {
                    return sequence.buildingBytes;
                },
                readsGeometryLater: sequence.readsGeometryLater,
            };
        }

        // Every batch goes into one encoder and one submit; each then reads its own stats back.
        const encoder = this.device.createCommandEncoder({ label: this.label });
        const recorded: { members: GpuTreeRequest[]; finish: () => Promise<GpuTreeBuffers> }[] = [];
        const failed: { members: GpuTreeRequest[]; reason: string }[] = [];
        let buildingBytes = 0;
        for (const indices of plan.batches) {
            const members = indices.map((index) => requests[index]);
            try {
                const batch = this.hploc.createBatch(members.map((member) => member.geometry));
                batch.encode(encoder);
                buildingBytes += batch.scratchBytes + batch.outputBytes;
                recorded.push({ members, finish: () => batch.finish() });
            } catch (error) {
                failed.push({ members, reason: `the GPU mesh tree build could not start: ${describe(error)}` });
            }
        }
        const submittedAt = performance.now();
        this.device.queue.submit([encoder.finish()]);

        const batches = recorded.map(async ({ members, finish }) => (await this.collect(members, finish, submittedAt, options)).result);
        for (const { members, reason } of failed) batches.push(Promise.resolve({ built: [], refused: refuseAll(members, reason), buildMs: 0, failure: reason }));
        return { notBuilt, batches, buildingBytes };
    }

    /**
     * The capped path: the batches one after another, each with its own submit
     * and stats readback, on one scratch pool, each once the previous batch's
     * outputs are let go.
     */
    private buildInSequence(groups: readonly (readonly GpuTreeRequest[])[], options: GpuTreeBuildOptions): Omit<GpuTreeBuilds, 'notBuilt'> {
        if (groups.length === 0) return { batches: [], buildingBytes: 0 };
        let pool: GpuHplocScratchPool;
        try {
            pool = this.hploc.createScratchPool(groups.map((members) => members.map((member) => member.geometry)));
        } catch (error) {
            const failure = `the GPU mesh tree build could not start: ${describe(error)}`;
            return { batches: groups.map((members) => Promise.resolve({ built: [], refused: refuseAll(members, failure), buildMs: 0, failure })), buildingBytes: 0 };
        }
        let outputsBuilding = 0;
        type Step = { result: GpuTreeBatchResult; outputs: GpuTreeOutputs | null };
        // Uploads, records and submits one batch at once (synchronously: the first batch's arrays are read before build returns).
        const start = (index: number, members: readonly GpuTreeRequest[]): Promise<Step> => {
            const refused = (failure: string): Promise<Step> => Promise.resolve({ result: { built: [], refused: refuseAll(members, failure), buildMs: 0, failure }, outputs: null });
            if (index > 0 && options.cancelled?.()) return refused('the GPU mesh tree build was cancelled');
            let batch: GpuHplocBatch;
            try {
                batch = this.hploc.createBatch(
                    members.map((member) => member.geometry),
                    {},
                    pool,
                );
            } catch (error) {
                return refused(`the GPU mesh tree build could not start: ${describe(error)}`);
            }
            outputsBuilding = batch.outputBytes;
            options.onMemoryChange?.();
            const encoder = this.device.createCommandEncoder({ label: `${this.label} batch ${index}` });
            batch.encode(encoder);
            const submittedAt = performance.now();
            this.device.queue.submit([encoder.finish()]);
            // Once finish returns, the outputs are the caller's (counted as theirs), so they stop counting here first.
            return this.collect(members, () => batch.finish().finally(() => (outputsBuilding = 0)), submittedAt, options);
        };
        const batches: Promise<GpuTreeBatchResult>[] = [];
        let previous: Promise<GpuTreeOutputs | null> = Promise.resolve(null);
        for (const [index, members] of groups.entries()) {
            // Each batch after the first waits until the one before it has let go of its outputs.
            const step =
                index === 0
                    ? start(0, members)
                    : previous.then(async (outputs) => {
                          if (outputs) await outputs.letGo;
                          return start(index, members);
                      });
            batches.push(step.then(({ result }) => result));
            previous = step.then(({ outputs }) => outputs);
        }
        // The last batch's stats are read, so the GPU is done with the pool.
        void previous.then(() => {
            pool.destroy();
            options.onMemoryChange?.();
        });
        return {
            batches,
            get buildingBytes() {
                return pool.bytes + outputsBuilding;
            },
            readsGeometryLater: new Set(groups.slice(1).flatMap((members) => members.map((member) => member.slot))),
        };
    }

    /** Reads a submitted batch's stats (and layouts when asked) and splits its trees into built and refused. */
    private async collect(
        members: readonly GpuTreeRequest[],
        finish: () => Promise<GpuTreeBuffers>,
        submittedAt: number,
        options: GpuTreeBuildOptions,
    ): Promise<{ result: GpuTreeBatchResult; outputs: GpuTreeOutputs | null }> {
        let trees: GpuTreeBuffers;
        try {
            trees = await finish();
        } catch (error) {
            const failure = `the GPU mesh tree build failed: ${describe(error)}`;
            return { result: { built: [], refused: refuseAll(members, failure), buildMs: performance.now() - submittedAt, failure }, outputs: null };
        }
        const buildMs = performance.now() - submittedAt;
        // A failed layout readback leaves the trees usable for tracing; only the refit goes without.
        let layouts: (GpuLayoutMeshBvh | null)[] = [];
        if (options.readLayouts) layouts = await this.hploc.readLayouts(trees).catch(() => []);
        const usable = trees.meshes.filter((tree) => tree.problem === null).length;
        const outputs = new GpuTreeOutputs(trees, usable);
        const built: GpuBuiltTree[] = [];
        const refused: GpuRefusedTree[] = [];
        trees.meshes.forEach((tree, index) => {
            const { slot, generation } = members[index];
            if (tree.problem === null) built.push({ slot, generation, outputs, tree, layout: layouts[index] ?? null });
            else refused.push({ slot, generation, reason: tree.problem });
        });
        return { result: { built, refused, buildMs, failure: null }, outputs };
    }
}

function refuseAll(members: readonly GpuTreeRequest[], reason: string): GpuRefusedTree[] {
    return members.map(({ slot, generation }) => ({ slot, generation, reason }));
}

function describe(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

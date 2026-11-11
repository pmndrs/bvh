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

import { GpuHplocBuilder, planGpuHplocBatches } from '../gpu-hploc.ts';
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
    }

    /** Frees the buffers whatever still uses them (the scene is being disposed). Adopted buffers are left to their storage. */
    destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
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
    /** One promise per batch. They never reject: a failed batch reports its meshes as refused. */
    readonly batches: readonly Promise<GpuTreeBatchResult>[];
    /** GPU bytes the batches hold while they build (inputs, scratch, outputs), until their stats are read. */
    readonly buildingBytes: number;
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
        const plan = planGpuHplocBatches(this.device, geometries);
        const limitBytes = Math.min(this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize);
        const notBuilt = plan.tooBig.map((index) => ({
            slot: requests[index].slot,
            generation: requests[index].generation,
            reason: `too big for a GPU build on this device (buffers of at most ${limitBytes} bytes)`,
        }));

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

        const batches = recorded.map(async ({ members, finish }): Promise<GpuTreeBatchResult> => {
            let trees: GpuTreeBuffers;
            try {
                trees = await finish();
            } catch (error) {
                const failure = `the GPU mesh tree build failed: ${describe(error)}`;
                return { built: [], refused: refuseAll(members, failure), buildMs: performance.now() - submittedAt, failure };
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
            return { built, refused, buildMs, failure: null };
        });
        for (const { members, reason } of failed) batches.push(Promise.resolve({ built: [], refused: refuseAll(members, reason), buildMs: 0, failure: reason }));
        return { notBuilt, batches, buildingBytes };
    }
}

function refuseAll(members: readonly GpuTreeRequest[], reason: string): GpuRefusedTree[] {
    return members.map(({ slot, generation }) => ({ slot, generation, reason }));
}

function describe(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

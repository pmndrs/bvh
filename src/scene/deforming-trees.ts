/**
 * Deforming meshes' trees for `SceneTrees` (ported from ddgi-core's
 * `platform/trace/deforming-trees.ts`, T3.3): every pose refits the mesh's
 * tree on the GPU, and a tree whose refit has drifted for about a second is
 * rebuilt with H-PLOC from the current pose.
 *
 * **The refit.** A deforming mesh keeps the shape of the tree its builder
 * made (H-PLOC's or tinybvh's). Each update after the app gives it a pose
 * ({@link DeformingTrees.posed}: the moved vertices on the GPU),
 * `GpuMeshRefit` rewrites the corners of its triangle records and refits its
 * boxes bottom-up, in the update's encoder: no submit, no CPU work per vertex.
 * A tree placed (or replaced) is refit at once from the latest pose, so a
 * tree built from an older pose never shows. The refit leaves each record's
 * flags word as it was. Where the tree sits comes from `meshRefitPlacement`,
 * which refuses a merged mesh (merged ranges are world space and must never be
 * refit); deforming meshes never merge.
 *
 * **The top level** keeps the mesh's deform box (the box every pose stays in),
 * not the refit root's, so a pose never needs a readback or a new top-level
 * shape (ddgi DECISIONS 2026-11-09).
 *
 * **Held drift.** A refit tree's quality depends only on how far the pose is
 * from the one it was built in. The refit's stats (its relative SAH cost) are
 * copied out at most once every {@link STATS_EVERY_FRAMES} updates per posed
 * mesh, and read back a few frames later (a few words). When every sample
 * since the first drifted one stayed above {@link DRIFT_THRESHOLD} times the
 * built tree's own cost, and that first sample is {@link DRIFT_HOLD_FRAMES}
 * updates old, the mesh is rebuilt with H-PLOC from the current pose and
 * this refit takes the new tree's shape. A sway rarely holds long enough; a
 * bend that stays does.
 *
 * Needs subgroups (the refit). Without them deforming meshes stay in the pose
 * their tree was built in (`SceneTrees` counts the poses it skipped).
 */

import type { GpuLayoutMeshBvh } from '../gpu-layout.ts';
import { GpuMeshRefit, GpuMeshRefitKernels, type MeshRefitVertices } from '../gpu-mesh-refit.ts';
import type { GpuMeshBuffers } from '../mesh-input.ts';
import { relativeSahCost } from '../refit-cpu.ts';
import { BVH_NODE_WORDS } from '../trace/layout.ts';

/** Refit cost over the built tree's above which a pose counts as drifted (within 10% of a fresh build is the BVH's done-when). */
export const DRIFT_THRESHOLD = 1.1;
/** Updates the drift must hold, about a second at 60 Hz, before a rebuild. */
export const DRIFT_HOLD_FRAMES = 60;
/** A posed mesh's refit stats are copied out for reading back at most once in this many updates. */
export const STATS_EVERY_FRAMES = 15;

/** Where a placed tree's words start in the scene's buffers (from `meshRefitPlacement`). */
export interface DeformingTreePlacement {
    readonly nodeWordOffset: number;
    readonly triangleWordOffset: number;
}

export interface DeformingTreeCounts {
    /** Deforming meshes whose tree is refit from their poses now. */
    readonly refitting: number;
    /** Mesh refits recorded. */
    readonly refits: number;
    /** Trees rebuilt because their refit had drifted. */
    readonly driftRebuilds: number;
}

interface DeformingMesh {
    readonly slot: number;
    readonly refit: GpuMeshRefit;
    readonly triangleCount: number;
    /** The box every pose stays in: the top level's box for the mesh. */
    readonly deformBox: Float32Array;
    /** The latest pose, or null while the mesh is in the pose its geometry gave. */
    vertices: MeshRefitVertices | null;
    placement: DeformingTreePlacement | null;
    builtCost: number;
    shapeGeneration: number;
    refitDue: boolean;
    driftSince: number;
    nextStatsFrame: number;
    sampleFrame: number;
    judgedStats: object | null;
    rebuilding: boolean;
}

export class DeformingTrees {
    static isSupported(device: GPUDevice): boolean {
        return GpuMeshRefit.isSupported(device);
    }

    private readonly device: GPUDevice;
    private readonly label: string;
    private readonly kernels: GpuMeshRefitKernels;
    private readonly meshes = new Map<number, DeformingMesh>();
    private nextShapeGeneration = 1;
    private refits = 0;
    private driftRebuilds = 0;

    constructor(device: GPUDevice, options: { readonly label?: string } = {}) {
        this.device = device;
        this.label = `${options.label ?? 'bvh'} mesh refit`;
        this.kernels = new GpuMeshRefitKernels(device, this.label);
    }

    prepare(): Promise<void> {
        return this.kernels.prepare();
    }

    get counts(): DeformingTreeCounts {
        let refitting = 0;
        for (const mesh of this.meshes.values()) if (mesh.placement) refitting++;
        return { refitting, refits: this.refits, driftRebuilds: this.driftRebuilds };
    }

    /** GPU bytes of the refits' own buffers (the trees are in the scene's). */
    get memoryBytes(): number {
        let bytes = 0;
        for (const mesh of this.meshes.values()) bytes += mesh.refit.memoryBytes;
        return bytes;
    }

    has(slot: number): boolean {
        return this.meshes.has(slot);
    }

    deformBox(slot: number): Float32Array | null {
        return this.meshes.get(slot)?.deformBox ?? null;
    }

    /** A deforming mesh was added. */
    meshAdded(slot: number, triangleCount: number, deformBox: ArrayLike<number>): void {
        this.meshes.set(slot, {
            slot,
            refit: new GpuMeshRefit(this.device, { label: `${this.label} ${slot}`, diagnostics: true, kernels: this.kernels }),
            triangleCount,
            deformBox: Float32Array.from(deformBox),
            vertices: null,
            placement: null,
            builtCost: 0,
            shapeGeneration: 0,
            refitDue: false,
            driftSince: -1,
            nextStatsFrame: 0,
            sampleFrame: 0,
            judgedStats: null,
            rebuilding: false,
        });
    }

    meshRemoved(slot: number): void {
        const mesh = this.meshes.get(slot);
        if (!mesh) return;
        mesh.refit.dispose();
        this.meshes.delete(slot);
    }

    /** A new pose: the tree follows it at the next update. */
    posed(slot: number, vertices: MeshRefitVertices): void {
        const mesh = this.meshes.get(slot);
        if (!mesh) return;
        mesh.vertices = vertices;
        mesh.refitDue = true;
    }

    /**
     * A tree of the mesh was placed. With its layout on the CPU (tinybvh's
     * always; a GPU build's when read back) the refit takes its shape and
     * refits it at once from the latest pose; without one the mesh keeps the
     * pose the tree was built in until a tree with a layout arrives.
     */
    treePlaced(slot: number, placement: DeformingTreePlacement, layout: GpuLayoutMeshBvh | null): void {
        const mesh = this.meshes.get(slot);
        if (!mesh) return;
        mesh.rebuilding = false;
        mesh.driftSince = -1;
        mesh.nextStatsFrame = 0;
        if (!layout) {
            mesh.placement = null;
            return;
        }
        mesh.shapeGeneration = this.nextShapeGeneration++;
        mesh.refit.setTree(layout, { ...placement, generation: mesh.shapeGeneration });
        const words = new Uint32Array(layout.nodes, 0, layout.nodeCount * BVH_NODE_WORDS);
        const floats = new Float32Array(layout.nodes, 0, layout.nodeCount * BVH_NODE_WORDS);
        mesh.builtCost = relativeSahCost(floats, words, 0, layout.nodeCount);
        mesh.placement = placement;
        mesh.refitDue = mesh.vertices !== null;
    }

    treeReleased(slot: number): void {
        const mesh = this.meshes.get(slot);
        if (mesh) mesh.placement = null;
    }

    /** Starts reading back stats copied in earlier updates (never waits). */
    pollStats(): void {
        for (const mesh of this.meshes.values()) mesh.refit.pollStats();
    }

    /** Records every due refit into one compute pass in `encoder`, then the stats copies whose turn it is. */
    encode(encoder: GPUCommandEncoder, buffers: { readonly nodes: GPUBuffer; readonly triangles: GPUBuffer }, frame: number): void {
        if (!this.kernels.ready) return;
        const due: DeformingMesh[] = [];
        for (const mesh of this.meshes.values()) if (mesh.refitDue && mesh.placement && mesh.vertices) due.push(mesh);
        if (due.length === 0) return;
        const pass = encoder.beginComputePass({ label: this.label });
        for (const mesh of due) {
            mesh.refit.encode(pass, buffers, mesh.vertices!);
            mesh.refitDue = false;
            this.refits++;
        }
        pass.end();
        for (const mesh of due) {
            if (frame < mesh.nextStatsFrame) continue;
            mesh.refit.copyStats(encoder);
            mesh.sampleFrame = frame;
            mesh.nextStatsFrame = frame + STATS_EVERY_FRAMES;
        }
    }

    /** The mesh to rebuild this update, if any: the one drifted longest, once held for {@link DRIFT_HOLD_FRAMES}. */
    rebuildDue(frame: number): number | null {
        let chosen: DeformingMesh | null = null;
        for (const mesh of this.meshes.values()) {
            this.readDrift(mesh);
            if (mesh.rebuilding || !mesh.placement || !mesh.vertices || mesh.driftSince < 0) continue;
            if (frame - mesh.driftSince < DRIFT_HOLD_FRAMES) continue;
            if (!chosen || mesh.driftSince < chosen.driftSince) chosen = mesh;
        }
        return chosen?.slot ?? null;
    }

    /** The latest pose as a GPU builder reads it (the drift rebuild). The vertex buffers need `COPY_SRC`. */
    rebuildGeometry(slot: number): GpuMeshBuffers | null {
        const mesh = this.meshes.get(slot);
        if (!mesh?.vertices) return null;
        return { ...mesh.vertices, triangleCount: mesh.triangleCount };
    }

    rebuildStarted(slot: number, started: boolean): void {
        const mesh = this.meshes.get(slot);
        if (!mesh) return;
        mesh.driftSince = -1;
        if (!started) return;
        mesh.rebuilding = true;
        this.driftRebuilds++;
    }

    rebuildRefused(slot: number): void {
        const mesh = this.meshes.get(slot);
        if (mesh) mesh.rebuilding = false;
    }

    /** The latest judged relative cost over the built cost, for tools (NaN before a sample). */
    driftRatio(slot: number): number {
        const mesh = this.meshes.get(slot);
        const stats = mesh?.refit.latestStats;
        if (!mesh || !stats || stats.generation !== mesh.shapeGeneration || !(mesh.builtCost > 0)) return NaN;
        return stats.relativeCost / mesh.builtCost;
    }

    dispose(): void {
        for (const mesh of this.meshes.values()) mesh.refit.dispose();
        this.meshes.clear();
    }

    private readDrift(mesh: DeformingMesh): void {
        const stats = mesh.refit.latestStats;
        if (!stats || stats === mesh.judgedStats) return;
        mesh.judgedStats = stats;
        if (stats.generation !== mesh.shapeGeneration || !(mesh.builtCost > 0)) return;
        const drifted = stats.relativeCost > DRIFT_THRESHOLD * mesh.builtCost;
        if (!drifted) mesh.driftSince = -1;
        else if (mesh.driftSince < 0) mesh.driftSince = mesh.sampleFrame;
    }
}

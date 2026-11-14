/**
 * A stand-in H-PLOC builder (`HplocBatchBuilder`) for `GpuFirstTrees` in
 * Node: each batch's "GPU" trees are tinybvh trees with one triangle per leaf
 * in fake buffers laid out like an H-PLOC batch's (room for one node per
 * triangle, the used nodes packed at the front), as the scene-trees tests'
 * stand-in builds them. It records what the capped queue must guarantee:
 * one batch at a time (from its creation until its outputs are let go), one
 * scratch pool at a time that holds every batch built on it, and the order
 * of it all. Test-only.
 */

import { hplocScratchPoolBytes, type GpuHplocBatch, type GpuHplocScratchPool, type HplocOptions } from '../../src/gpu-hploc.ts';
import type { GpuMeshGeometry, GpuMeshTreeRange, GpuTreeBuffers } from '../../src/mesh-input.ts';
import { quantizeNodes } from '../../src/quantized-nodes.ts';
import type { MeshBvh } from '../../src/trace/mesh-bvh.ts';
import type { HplocBatchBuilder } from '../../src/scene/gpu-first-trees.ts';
import { buildMeshBvhAsync } from '../../src/trace/mesh-bvh-build.ts';
import type { MeshGeometry } from '../../src/trace/mesh-bvh.ts';

export class FakeHploc implements HplocBatchBuilder {
    /** What happened, in order: `pool <bytes>`, `pool freed`, `start <batch>`, `let go <batch>`, `dropped <batch>`. */
    readonly events: string[] = [];
    /** Broken guarantees (two batches at once, a batch on a pool too small or freed). */
    readonly violations: string[] = [];
    /** Batches created and not let go yet, and the most there ever were. */
    active = 0;
    maxActive = 0;
    poolsAlive = 0;
    maxPoolsAlive = 0;
    /** The slots of each batch, in the order they were created (a stand-in slot: the geometry's position in `slotsOf`). */
    readonly batches: number[][] = [];
    /** True: each batch's stats wait for {@link finishNext}. False: they come on their own. */
    manual = false;
    /** Batches (by creation order) whose `encode` throws, as a lost device would. */
    readonly throwOnEncode = new Set<number>();
    /** Batches (by creation order) whose `finish` rejects. */
    readonly failFinish = new Set<number>();
    /** While set, `readLayouts` waits for it. */
    layoutGate: Promise<void> | null = null;
    /** Maps geometry objects to the slot the test gave them, for {@link batches}. */
    readonly slotsOf = new Map<GpuMeshGeometry, number>();
    private created = 0;
    private readonly waiting: (() => void)[] = [];

    constructor(private readonly device: GPUDevice) {}

    prepare(): Promise<void> {
        return Promise.resolve();
    }

    createScratchPool(batches: readonly (readonly GpuMeshGeometry[])[]): GpuHplocScratchPool {
        const sizes = hplocScratchPoolBytes(batches);
        const total = Object.values<number>(sizes).reduce((sum, bytes) => sum + bytes, 0);
        this.poolsAlive++;
        this.maxPoolsAlive = Math.max(this.maxPoolsAlive, this.poolsAlive);
        this.events.push(`pool ${total}`);
        let destroyed = false;
        const pool = {
            mortonCodes: 'cubic-30',
            sizes,
            get bytes() {
                return destroyed ? 0 : total;
            },
            get isDestroyed() {
                return destroyed;
            },
            destroy: () => {
                if (destroyed) return;
                destroyed = true;
                this.poolsAlive--;
                this.events.push('pool freed');
            },
        };
        return pool as unknown as GpuHplocScratchPool;
    }

    createBatch(geometries: readonly GpuMeshGeometry[], options?: HplocOptions, pool?: GpuHplocScratchPool): GpuHplocBatch {
        const triangleLeaves = options?.triangleLeaves === 'indexed' ? 'indexed' : 'records';
        const quantizeOnEmit = options?.quantizeOnEmit === true;
        const leading = geometries.map((_, index) => options?.triangleLeading?.[index] ?? 0);
        const index = this.created++;
        if (!pool) this.violations.push(`batch ${index} without a pool`);
        else if (pool.isDestroyed) this.violations.push(`batch ${index} on a freed pool`);
        else {
            const needs = hplocScratchPoolBytes([geometries]);
            for (const name of Object.keys(needs) as (keyof typeof needs)[]) if (needs[name] > pool.sizes[name]) this.violations.push(`batch ${index}: the pool's ${name} is too small`);
        }
        if (this.active > 0) this.violations.push(`batch ${index} started while ${this.active} other batch(es) held their outputs`);
        this.active++;
        this.maxActive = Math.max(this.maxActive, this.active);
        this.events.push(`start ${index}`);
        this.batches.push(geometries.map((geometry) => this.slotsOf.get(geometry) ?? -1));
        // The arrays are read now, as the real batch uploads them.
        const copies = geometries.map((geometry) => {
            const cpu = geometry as MeshGeometry;
            // Flags and materials as the real batch uploads them (packed into the flags word).
            return { positions: cpu.positions.slice(), indices: cpu.indices ? Uint32Array.from(cpu.indices) : null, triangleFlags: cpu.triangleFlags?.slice() ?? null, triangleMaterials: cpu.triangleMaterials ? Array.from(cpu.triangleMaterials) : null };
        });
        const triangleTotal = copies.reduce((sum, mesh) => sum + (mesh.indices ? mesh.indices.length : mesh.positions.length / 3) / 3, 0);
        let finished = false;
        let destroyed = false;
        const letGo = (what: string) => {
            this.active--;
            this.events.push(`${what} ${index}`);
        };
        const batch = {
            meshCount: geometries.length,
            outputBytes: triangleTotal * (64 + 48 + 4),
            scratchBytes: 0,
            encode: () => {
                if (this.throwOnEncode.has(index)) throw new Error(`batch ${index} could not be recorded (test)`);
            },
            finish: async (): Promise<GpuTreeBuffers> => {
                if (this.manual) await new Promise<void>((resolve) => this.waiting.push(resolve));
                if (this.failFinish.has(index)) {
                    destroyed = true;
                    letGo('dropped');
                    throw new Error(`batch ${index}'s readback failed (test)`);
                }
                finished = true;
                return this.trees(copies, leading, triangleLeaves, quantizeOnEmit, () => letGo('let go'));
            },
            destroy: () => {
                if (destroyed) return;
                destroyed = true;
                if (!finished) letGo('dropped');
            },
        };
        return batch as unknown as GpuHplocBatch;
    }

    readLayouts(): Promise<never[]> {
        return this.layoutGate ? this.layoutGate.then(() => []) : Promise.resolve([]);
    }

    /** Batches whose stats wait ({@link manual}). */
    get waitingBatches(): number {
        return this.waiting.length;
    }

    /** Lets the oldest waiting batch's stats come, then lets the promises run. */
    async finishNext(): Promise<void> {
        const resolve = this.waiting.shift();
        if (!resolve) throw new Error('no batch is waiting');
        resolve();
        await settle();
    }

    private async trees(meshes: readonly MeshGeometry[], leading: readonly number[], triangleLeaves: 'records' | 'indexed', quantizeOnEmit: boolean, onLetGo: () => void): Promise<GpuTreeBuffers> {
        const trees = await Promise.all(meshes.map((mesh) => buildMeshBvhAsync(mesh, { maxLeafPrimitives: 1, triangleLeaves })));
        const { nodes, triangles, parents, meshes: ranges } = standInOutputs(this.device, trees, leading.map((triangleLeading) => ({ triangleLeading })), triangleLeaves, quantizeOnEmit, 'fake hploc');
        // The parents go with the outputs either way (destroyed, or freed at an adoption): that is the batch letting go.
        const destroyParents = parents.destroy.bind(parents);
        let letGo = false;
        parents.destroy = () => {
            if (!letGo) {
                letGo = true;
                onLetGo();
            }
            destroyParents();
        };
        return {
            nodes,
            triangles,
            parents,
            meshes: ranges,
            ...(triangleLeaves === 'indexed' ? { triangleLeaves } : {}),
            ...(quantizeOnEmit ? { nodeFormat: 'quantized' as const } : {}),
            destroy: () => [nodes, triangles, parents].forEach((buffer) => buffer.destroy()),
        };
    }
}

/**
 * Fake buffers laid out like an H-PLOC batch's outputs for these trees: room
 * for one node per triangle (with `quantizeOnEmit`, each mesh's frame slot and
 * 32-byte nodes, `quantizeNodes`' bytes: what the emit writes), the used nodes
 * packed at the front, and each mesh's triangles after its `triangleLeading`
 * gap. The ranges are H-PLOC's (`nodeOffset` the root's slot).
 */
export function standInOutputs(
    device: GPUDevice,
    trees: readonly MeshBvh[],
    members: readonly { readonly triangleLeading?: number }[],
    triangleLeaves: 'records' | 'indexed',
    quantizeOnEmit: boolean,
    label: string,
): { nodes: GPUBuffer; triangles: GPUBuffer; parents: GPUBuffer; meshes: GpuMeshTreeRange[] } {
    const bytes = triangleLeaves === 'indexed' ? 16 : 48;
    const triangleTotal = trees.reduce((sum, tree) => sum + tree.triangleCount, 0);
    const leadingTotal = members.reduce((sum, member) => sum + (member.triangleLeading ?? 0), 0);
    const usage = 0x4 | 0x8 | 0x80;
    const nodes = device.createBuffer({ label: `${label} nodes`, size: quantizeOnEmit ? (triangleTotal + trees.length) * 32 : triangleTotal * 64, usage });
    const triangles = device.createBuffer({ label: `${label} triangles`, size: Math.max(1, triangleTotal + leadingTotal) * bytes, usage });
    const parents = device.createBuffer({ label: `${label} parents`, size: triangleTotal * 4, usage });
    const meshes: GpuMeshTreeRange[] = [];
    let nodeSlot = 0;
    let triangleOffset = 0;
    trees.forEach((tree, index) => {
        let nodeOffset: number;
        if (quantizeOnEmit) {
            device.queue.writeBuffer(nodes, nodeSlot * 32, quantizeNodes(tree).nodes, 0, (tree.nodeCount + 1) * 32);
            nodeOffset = nodeSlot + 1;
            nodeSlot += tree.nodeCount + 1;
        } else {
            device.queue.writeBuffer(nodes, nodeSlot * 64, tree.nodes, 0, tree.nodeCount * 64);
            nodeOffset = nodeSlot;
            nodeSlot += tree.nodeCount;
        }
        triangleOffset += members[index]?.triangleLeading ?? 0;
        device.queue.writeBuffer(triangles, triangleOffset * bytes, tree.triangles, 0, tree.triangleCount * bytes);
        meshes.push({ nodeOffset, nodeCount: tree.nodeCount, triangleOffset, triangleCount: tree.triangleCount, leafCount: tree.leafCount, maxDepth: tree.maxDepth, bounds: tree.bounds, problem: null });
        triangleOffset += tree.triangleCount;
    });
    return { nodes, triangles, parents, meshes };
}

/** Lets pending promise callbacks (and a few awaits deep) run. */
export async function settle(rounds = 20): Promise<void> {
    for (let round = 0; round < rounds; round++) await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

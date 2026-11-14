/**
 * Clustered top level (PARITY 8.9; docs/internals/design/clustered-bvh.md):
 * three-mesh-bvh's `ClusteredBVH` on our layout.
 *
 * The top level normally has one leaf per instance, with the instance's whole
 * mesh box. Where big meshes overlap (Sponza, Bistro: one instance per mesh),
 * a ray enters many instances it never hits. `clusterTraceScene` cuts a
 * mesh's tree into subtrees ("clusters") of fewer than `primitiveLimit`
 * triangles and gives every cluster its own top-level leaf, with the
 * cluster's box moved to world space, so the top level can separate the
 * parts of meshes that overlap. This is three-mesh-bvh's `ClusteredBVH`, the
 * top level its `BVHComputeData` always builds (partial re-braiding in the
 * literature, Benthin et al. 2017).
 *
 * **The cut** is `ClusteredBVH`'s (`_traverseClusters`, three-mesh-bvh
 * 0.9.18, MIT; THIRD_PARTY.md): walk the mesh's tree from the root and stop
 * at the first node with fewer than `primitiveLimit` triangles, or at a leaf;
 * each stop is a cluster. A mesh with fewer than `primitiveLimit` triangles
 * is one cluster, its whole tree.
 *
 * **Nothing in the walk changes.** A cluster is an ordinary instance record:
 * the instance's own words (its world-to-object rows, triangle offset,
 * instance index, flags, mask and material, bit for bit), with
 * `rootNodeOffset` pointing at a copy of the cluster's subtree. Child indices
 * in our nodes are relative to the root the walk enters, so each subtree is
 * copied once (shared by every record that uses it) with its inner children
 * renumbered from 0, after the scene's own nodes; quantized subtrees take a
 * copy of their mesh's frame slot in front, so their planes decode to the
 * same values. Leaf indices stay relative to the mesh's triangles, which are
 * not copied: every triangle format (records, indexed, merged side tables)
 * works as it is. A leaf cluster becomes a one-node tree with the leaf on the
 * left and no right child. Hits name the same instance, triangle, side,
 * material and normal as the unclustered scene, and `t` is the same to the
 * bit (the same rows move the ray, the same test runs); only which of two
 * triangles at exactly the same `t` is reported can differ, because the walk
 * visits them in another order.
 *
 * **What it costs:** one 64-byte record and about two top-level nodes per
 * cluster, and a frame slot per quantized cluster. A mesh tree that only
 * split records trace is dropped, as `ClusteredBVH` uploads only the subtrees
 * its leaves enter: its nodes above the cut go, the ones below move into the
 * copies. Its `TraceSceneData.meshes` entry keeps its triangles (filter flags
 * are written there as before) with `nodeCount` 0 and `clustered: true`, and
 * `meshRefitPlacement` refuses it: a refit would not reach the copies, so
 * leave meshes you refit out of the split (`split`). The other trees move up
 * to close the gaps, and their entries say where.
 */

import { DEFAULT_SAH_OPTIONS, buildSahTree, type SahBuildOptions } from './sah-builder.ts';
import {
    BVH_NODE_WORDS,
    EMPTY_CHILD_INDEX,
    INSTANCE_FLAG,
    INSTANCE_RECORD_WORDS,
    INSTANCE_WORD,
    NODE_WORD,
    QUANTIZED_NODE_WORDS,
    REF_COUNT_SHIFT,
    REF_INDEX_MASK,
    TOP_HEADER_WORD,
    TOP_LEVEL_HEADER_BYTES,
    TOP_LEVEL_HEADER_WORDS,
    TRACE_STACK_SIZE,
    BVH_NODE_BYTES,
    INSTANCE_RECORD_BYTES,
    QUANTIZED_NODE_BYTES,
} from './layout.ts';
import { decodeMaxPlane, decodeMinPlane, readNodeFrame, type NodeFrame } from '../quantized-nodes.ts';
import { GPU_LAYOUT_QUANTIZED_NODE_WORD } from '../gpu-layout.ts';
import { invertAffine, transformBounds, type TopLevelBvh } from './top-level.ts';
import type { TraceSceneData, TraceSceneMeshPlacement } from './scene-assembly.ts';

/** three-mesh-bvh's `ClusteredBVH` default: a subtree of fewer triangles than this is one top-level leaf. */
export const DEFAULT_CLUSTER_PRIMITIVE_LIMIT = 64;

/** One instance record of the scene, as the `split` option sees it. */
export interface TraceClusterRecord {
    /** The record's `instanceIndex` (`BvhHit.instance`); `0xffffffff` for a merged group's record. */
    readonly instanceIndex: number;
    /** A merged group of static instances (`INSTANCE_FLAG.merged`). */
    readonly merged: boolean;
    /** The input mesh this record traces (its position in `scene.meshes`), or undefined for a merged group's record. */
    readonly meshIndex: number | undefined;
    /** The merged group this record traces (its position in `scene.mergedGroups`), or undefined. */
    readonly mergedGroup: number | undefined;
    /** The mesh's root in the scene's nodes, counted in nodes. */
    readonly nodeOffset: number;
    /** How many records of the scene trace this mesh (more than 1: an instanced mesh). */
    readonly sharedBy: number;
}

/** Options of {@link clusterTraceScene}. */
export interface TraceClusterOptions {
    /**
     * three-mesh-bvh's `primitiveLimit`: the walk down a mesh's tree stops at
     * the first node with fewer triangles than this (or at a leaf), and that
     * subtree becomes one top-level leaf. Default
     * {@link DEFAULT_CLUSTER_PRIMITIVE_LIMIT} (64). A positive integer, or
     * `Infinity` for no split (the scene comes back as it was).
     */
    readonly primitiveLimit?: number;
    /**
     * Which records to cut into clusters; the others keep one leaf for their
     * whole mesh. Default: records whose mesh no other record traces
     * (`sharedBy === 1`), as `ClusteredBVH` references instanced meshes
     * whole. Leave out meshes you refit (deforming meshes): their cluster
     * copies don't follow a refit.
     */
    readonly split?: (record: TraceClusterRecord) => boolean;
    /** The top level's SAH options (one cluster per leaf, as one instance per leaf). */
    readonly topLevel?: Partial<SahBuildOptions>;
}

/** What {@link clusterTraceScene} did. */
export interface TraceClusterStats {
    /** The scene's instance records before. */
    readonly records: number;
    /** Records cut into clusters (the rest kept one leaf). */
    readonly splitRecords: number;
    /** Top-level records after: one per cluster, one per record kept whole. */
    readonly clusterRecords: number;
    /** Distinct subtrees copied after the scene's nodes (shared by every record of the mesh). */
    readonly copiedSubtrees: number;
    /** Node slots copied, frame slots of quantized subtrees included. */
    readonly copiedNodeSlots: number;
    /** The scene's own trees left out because only split records traced them (their placements say `clustered`). */
    readonly droppedTrees: number;
    /** Node slots those trees held, frame slots included. */
    readonly droppedNodeSlots: number;
    /** Bytes the scene grew by (negative when it shrank): copied nodes and the larger top level, less the dropped trees. */
    readonly addedBytes: number;
}

/** A scene from {@link clusterTraceScene}: a normal `TraceSceneData` plus what was done. */
export interface ClusteredTraceSceneData extends TraceSceneData {
    readonly clusters: TraceClusterStats;
}

/** One cluster of a mesh's tree (from {@link meshClusters}). */
export interface MeshCluster {
    /**
     * The cluster's root node, relative to the mesh's root (0: the whole
     * tree), or -1 for a leaf cluster, which is child `side` of node `parent`.
     */
    readonly node: number;
    readonly parent: number;
    readonly side: number;
    /** The cluster's lowest triangle slot, relative to the mesh's triangles. */
    readonly firstTriangle: number;
    /** Triangles under the cluster. */
    readonly triangleCount: number;
    /** Object-space box `[min x3, max x3]`: the union of the cluster root's child boxes as the nodes store them (a leaf's own box). */
    readonly bounds: Float64Array;
    /** Levels of nodes in the cluster's tree (a leaf cluster's one-node tree counts 1). */
    readonly depth: number;
}

/** Reads one mesh's nodes in the scene's format, by node index relative to the mesh's root. */
interface MeshNodes {
    readonly childIndex: (node: number, side: number) => number;
    readonly childCount: (node: number, side: number) => number;
    /** Child `side`'s box into `out[0..5]` (min x3, max x3), as decoded values. */
    readonly childBox: (node: number, side: number, out: Float64Array) => void;
}

function sceneNodeReader(scene: Pick<TraceSceneData, 'nodes' | 'nodeFormat'>, nodeOffset: number): MeshNodes {
    const words = new Uint32Array(scene.nodes);
    const floats = new Float32Array(scene.nodes);
    if ((scene.nodeFormat ?? 'f32') === 'f32') {
        const base = (node: number) => (nodeOffset + node) * BVH_NODE_WORDS;
        return {
            childIndex: (node, side) => words[base(node) + (side === 0 ? NODE_WORD.leftIndex : NODE_WORD.rightIndex)],
            childCount: (node, side) => words[base(node) + (side === 0 ? NODE_WORD.leftCount : NODE_WORD.rightCount)],
            childBox: (node, side, out) => {
                const at = base(node) + side * 8;
                for (let axis = 0; axis < 3; axis++) {
                    out[axis] = floats[at + axis];
                    out[axis + 3] = floats[at + 4 + axis];
                }
            },
        };
    }
    const frame: NodeFrame = readNodeFrame(words, (nodeOffset - 1) * QUANTIZED_NODE_WORDS);
    const reference = (node: number, side: number) => words[(nodeOffset + node) * QUANTIZED_NODE_WORDS + side * 4 + GPU_LAYOUT_QUANTIZED_NODE_WORD.leftReference];
    return {
        childIndex: (node, side) => {
            const value = reference(node, side);
            return value === EMPTY_CHILD_INDEX ? EMPTY_CHILD_INDEX : value & REF_INDEX_MASK;
        },
        childCount: (node, side) => {
            const value = reference(node, side);
            return value === EMPTY_CHILD_INDEX ? 0 : value >>> REF_COUNT_SHIFT;
        },
        childBox: (node, side, out) => {
            const at = (nodeOffset + node) * QUANTIZED_NODE_WORDS + side * 4;
            for (let axis = 0; axis < 3; axis++) {
                out[axis] = decodeMinPlane(words[at + axis], frame.origin[axis], frame.step[axis]);
                out[axis + 3] = decodeMaxPlane(words[at + axis], frame.origin[axis], frame.step[axis]);
            }
        },
    };
}

/** Per inner node of one mesh: triangles below, lowest triangle slot, levels of nodes below (itself included). */
interface MeshSummary {
    readonly reader: MeshNodes;
    readonly counts: Map<number, number>;
    readonly firsts: Map<number, number>;
    readonly depths: Map<number, number>;
}

function summarizeMesh(scene: Pick<TraceSceneData, 'nodes' | 'nodeFormat'>, nodeOffset: number): MeshSummary {
    const reader = sceneNodeReader(scene, nodeOffset);
    // Pre-order from the root; children come after their parent, so the reverse order sees children first.
    const order: number[] = [];
    const stack = [0];
    while (stack.length > 0) {
        const node = stack.pop()!;
        order.push(node);
        for (let side = 1; side >= 0; side--) {
            const index = reader.childIndex(node, side);
            if (index !== EMPTY_CHILD_INDEX && reader.childCount(node, side) === 0) stack.push(index);
        }
    }
    const counts = new Map<number, number>();
    const firsts = new Map<number, number>();
    const depths = new Map<number, number>();
    for (let k = order.length - 1; k >= 0; k--) {
        const node = order[k];
        let count = 0;
        let first = Infinity;
        let below = 0;
        for (let side = 0; side < 2; side++) {
            const index = reader.childIndex(node, side);
            if (index === EMPTY_CHILD_INDEX) continue;
            const leafCount = reader.childCount(node, side);
            if (leafCount > 0) {
                count += leafCount;
                first = Math.min(first, index);
            } else {
                count += counts.get(index)!;
                first = Math.min(first, firsts.get(index)!);
                below = Math.max(below, depths.get(index)!);
            }
        }
        counts.set(node, count);
        firsts.set(node, first);
        depths.set(node, below + 1);
    }
    return { reader, counts, firsts, depths };
}

function checkPrimitiveLimit(limit: number): number {
    if (limit === Infinity || (Number.isInteger(limit) && limit >= 1)) return limit;
    throw new RangeError(`clusterTraceScene: primitiveLimit must be a positive integer or Infinity, got ${String(limit)}`);
}

function unionOfChildren(reader: MeshNodes, node: number): Float64Array {
    const box = new Float64Array([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
    const child = new Float64Array(6);
    for (let side = 0; side < 2; side++) {
        if (reader.childIndex(node, side) === EMPTY_CHILD_INDEX) continue;
        reader.childBox(node, side, child);
        for (let axis = 0; axis < 3; axis++) {
            box[axis] = Math.min(box[axis], child[axis]);
            box[axis + 3] = Math.max(box[axis + 3], child[axis + 3]);
        }
    }
    return box;
}

function clustersOf(summary: MeshSummary, primitiveLimit: number): MeshCluster[] {
    const { reader, counts, firsts, depths } = summary;
    const clusters: MeshCluster[] = [];
    const nodeCluster = (node: number): MeshCluster => ({
        node,
        parent: -1,
        side: -1,
        firstTriangle: firsts.get(node)!,
        triangleCount: counts.get(node)!,
        bounds: unionOfChildren(reader, node),
        depth: depths.get(node)!,
    });
    // ClusteredBVH's _traverseClusters: stop at a node with fewer triangles than the limit, or at a leaf.
    if (counts.get(0)! < primitiveLimit) return [nodeCluster(0)];
    const visit = (node: number): void => {
        for (let side = 0; side < 2; side++) {
            const index = reader.childIndex(node, side);
            if (index === EMPTY_CHILD_INDEX) continue;
            const leafCount = reader.childCount(node, side);
            if (leafCount > 0) {
                const bounds = new Float64Array(6);
                reader.childBox(node, side, bounds);
                clusters.push({ node: -1, parent: node, side, firstTriangle: index, triangleCount: leafCount, bounds, depth: 1 });
            } else if (counts.get(index)! < primitiveLimit) {
                clusters.push(nodeCluster(index));
            } else {
                visit(index);
            }
        }
    };
    visit(0);
    return clusters;
}

/**
 * The clusters of the mesh whose root is at `nodeOffset` in the scene's nodes
 * (a `TraceSceneData.meshes` or `mergedGroups` entry's `nodeOffset`), cut as
 * {@link clusterTraceScene} cuts it: in the walk's order, left before right.
 */
export function meshClusters(scene: Pick<TraceSceneData, 'nodes' | 'nodeFormat'>, nodeOffset: number, primitiveLimit = DEFAULT_CLUSTER_PRIMITIVE_LIMIT): MeshCluster[] {
    return clustersOf(summarizeMesh(scene, nodeOffset), checkPrimitiveLimit(primitiveLimit));
}

/** Copies one cluster's tree as a standalone tree (root at 0), in the scene's node format: words, frame slot first when quantized. */
function copyCluster(scene: Pick<TraceSceneData, 'nodes' | 'nodeFormat'>, nodeOffset: number, reader: MeshNodes, cluster: MeshCluster): Uint32Array {
    const words = new Uint32Array(scene.nodes);
    const quantized = (scene.nodeFormat ?? 'f32') === 'quantized';
    const slotWords = quantized ? QUANTIZED_NODE_WORDS : BVH_NODE_WORDS;
    const lead = quantized ? 1 : 0;
    if (cluster.node < 0) {
        // A leaf: one node, the leaf on the left, no right child.
        const out = new Uint32Array((1 + lead) * slotWords);
        if (quantized) {
            out.set(words.subarray((nodeOffset - 1) * slotWords, nodeOffset * slotWords), 0);
            const from = (nodeOffset + cluster.parent) * slotWords + cluster.side * 4;
            out.set(words.subarray(from, from + 4), slotWords);
            out.fill(0xffffffff, slotWords + 4, slotWords + 8);
            return out;
        }
        const from = (nodeOffset + cluster.parent) * slotWords + cluster.side * 8;
        for (let k = 0; k < 3; k++) {
            out[NODE_WORD.leftMin + k] = words[from + k];
            out[NODE_WORD.leftMax + k] = words[from + 4 + k];
        }
        out[NODE_WORD.leftIndex] = reader.childIndex(cluster.parent, cluster.side);
        out[NODE_WORD.leftCount] = reader.childCount(cluster.parent, cluster.side);
        out[NODE_WORD.rightIndex] = EMPTY_CHILD_INDEX;
        out[NODE_WORD.rightCount] = 0;
        return out;
    }
    // Pre-order from the cluster's root: new index = position in the order.
    const order: number[] = [];
    const renumbered = new Map<number, number>();
    const stack = [cluster.node];
    while (stack.length > 0) {
        const node = stack.pop()!;
        renumbered.set(node, order.length);
        order.push(node);
        for (let side = 1; side >= 0; side--) {
            const index = reader.childIndex(node, side);
            if (index !== EMPTY_CHILD_INDEX && reader.childCount(node, side) === 0) stack.push(index);
        }
    }
    const out = new Uint32Array((order.length + lead) * slotWords);
    if (quantized) out.set(words.subarray((nodeOffset - 1) * slotWords, nodeOffset * slotWords), 0);
    order.forEach((node, at) => {
        const to = (at + lead) * slotWords;
        out.set(words.subarray((nodeOffset + node) * slotWords, (nodeOffset + node + 1) * slotWords), to);
        for (let side = 0; side < 2; side++) {
            const index = reader.childIndex(node, side);
            if (index === EMPTY_CHILD_INDEX || reader.childCount(node, side) > 0) continue;
            const inner = renumbered.get(index)!;
            if (quantized) out[to + side * 4 + GPU_LAYOUT_QUANTIZED_NODE_WORD.leftReference] = inner;
            else out[to + (side === 0 ? NODE_WORD.leftIndex : NODE_WORD.rightIndex)] = inner;
        }
    });
    return out;
}

/** Object-to-world (column-major 4x4) of a record: the inverse of its world-to-object rows, in doubles. */
function recordObjectToWorld(floats: Float32Array, record: number): number[] {
    const rows = record + INSTANCE_WORD.worldToObjectRow0;
    // The rows as a column-major affine matrix, inverted back.
    const worldToObject = [
        floats[rows], floats[rows + 4], floats[rows + 8], 0,
        floats[rows + 1], floats[rows + 5], floats[rows + 9], 0,
        floats[rows + 2], floats[rows + 6], floats[rows + 10], 0,
        floats[rows + 3], floats[rows + 7], floats[rows + 11], 1,
    ];
    const inverse = invertAffine(worldToObject);
    if (!inverse) throw new RangeError(`clusterTraceScene: instance record ${(record - TOP_LEVEL_HEADER_WORDS) / INSTANCE_RECORD_WORDS} has a singular transform`);
    const r = inverse.rows;
    return [r[0], r[4], r[8], 0, r[1], r[5], r[9], 0, r[2], r[6], r[10], 0, r[3], r[7], r[11], 1];
}

/**
 * Cuts the meshes of an assembled scene into clusters, each its own top-level
 * leaf (three-mesh-bvh's `ClusteredBVH`; see the file comment). Returns a new
 * scene: the same triangles and positions, the scene's trees that are still
 * entered at their roots followed by the copied subtrees, the placements to
 * match, and a new top level; `scene` is not changed. Every walk
 * traces it as it is (the WGSL, `createCpuTracer`, the batch kernels, the
 * queries and `SdfGenerator`), compiled for the scene's formats.
 *
 * With nothing to split (`primitiveLimit: Infinity`, or `split` refusing
 * every record) the input scene comes back as it is, with `clusters`.
 *
 * Rebuild after anything moves: like `buildTopLevel`, this is a CPU build of
 * the whole top level; the GPU top level (`GpuTopLevel`, `SceneTrees`) has no
 * clustered mode.
 */
export function clusterTraceScene(scene: TraceSceneData, options: TraceClusterOptions = {}): ClusteredTraceSceneData {
    const primitiveLimit = checkPrimitiveLimit(options.primitiveLimit ?? DEFAULT_CLUSTER_PRIMITIVE_LIMIT);
    const quantized = (scene.nodeFormat ?? 'f32') === 'quantized';
    const nodeBytes = quantized ? QUANTIZED_NODE_BYTES : BVH_NODE_BYTES;
    const slotWords = nodeBytes / 4;
    const top = scene.topLevel;
    const topWords = new Uint32Array(top.buffer);
    const topFloats = new Float32Array(top.buffer);
    const recordCount = topWords[TOP_HEADER_WORD.instanceCount];
    const recordWord = (record: number) => TOP_LEVEL_HEADER_WORDS + record * INSTANCE_RECORD_WORDS;

    // Which mesh each record traces, and how many records share it.
    const sharedBy = new Map<number, number>();
    for (let record = 0; record < recordCount; record++) {
        const offset = topWords[recordWord(record) + INSTANCE_WORD.rootNodeOffset];
        sharedBy.set(offset, (sharedBy.get(offset) ?? 0) + 1);
    }
    const meshAt = new Map<number, number>();
    scene.meshes.forEach((placement, index) => {
        if (placement.triangleCount > 0) meshAt.set(placement.nodeOffset, index);
    });
    const groupAt = new Map<number, number>();
    scene.mergedGroups.forEach((group, index) => groupAt.set(group.nodeOffset, index));
    const split = options.split ?? ((record: TraceClusterRecord) => record.sharedBy === 1);
    const splits = new Uint8Array(recordCount);
    let splitRecords = 0;
    if (primitiveLimit !== Infinity) {
        for (let record = 0; record < recordCount; record++) {
            const base = recordWord(record);
            const nodeOffset = topWords[base + INSTANCE_WORD.rootNodeOffset];
            const info: TraceClusterRecord = {
                instanceIndex: topWords[base + INSTANCE_WORD.instanceIndex],
                merged: (topWords[base + INSTANCE_WORD.flags] & INSTANCE_FLAG.merged) !== 0,
                meshIndex: meshAt.get(nodeOffset),
                mergedGroup: groupAt.get(nodeOffset),
                nodeOffset,
                sharedBy: sharedBy.get(nodeOffset)!,
            };
            if (split(info)) {
                splits[record] = 1;
                splitRecords++;
            }
        }
    }
    const unchanged: TraceClusterStats = { records: recordCount, splitRecords: 0, clusterRecords: recordCount, copiedSubtrees: 0, copiedNodeSlots: 0, droppedTrees: 0, droppedNodeSlots: 0, addedBytes: 0 };
    if (splitRecords === 0) return { ...scene, clusters: unchanged };

    // Each mesh is summarized and cut once; each cluster's subtree is copied once.
    const summaries = new Map<number, MeshSummary>();
    const summaryOf = (nodeOffset: number): MeshSummary => {
        let summary = summaries.get(nodeOffset);
        if (!summary) {
            summary = summarizeMesh(scene, nodeOffset);
            summaries.set(nodeOffset, summary);
        }
        return summary;
    };
    // A root is the scene's own tree at `tree` (its old offset), or copy `copy`.
    type Root = { readonly tree: number; readonly copy: -1 } | { readonly tree: -1; readonly copy: number };
    const cuts = new Map<number, { clusters: MeshCluster[]; roots: Root[] }>();
    const copies: Uint32Array[] = [];
    const cutOf = (nodeOffset: number) => {
        let cut = cuts.get(nodeOffset);
        if (!cut) {
            const summary = summaryOf(nodeOffset);
            const clusters = clustersOf(summary, primitiveLimit);
            const roots = clusters.map((cluster): Root => {
                if (cluster.node === 0) return { tree: nodeOffset, copy: -1 }; // the whole tree: no copy
                copies.push(copyCluster(scene, nodeOffset, summary.reader, cluster));
                return { tree: -1, copy: copies.length - 1 };
            });
            cut = { clusters, roots };
            cuts.set(nodeOffset, cut);
        }
        return cut;
    };

    // One entry per new record: the source record, the root it enters, its world box, the tree's depth.
    const sources: number[] = [];
    const roots: Root[] = [];
    const depths: number[] = [];
    const worldBoxes: number[] = [];
    const enteredWhole = new Set<number>();
    const box = new Float32Array(6);
    for (let record = 0; record < recordCount; record++) {
        const base = recordWord(record);
        const nodeOffset = topWords[base + INSTANCE_WORD.rootNodeOffset];
        const objectToWorld = recordObjectToWorld(topFloats, base);
        const summary = summaryOf(nodeOffset);
        let entries: { root: Root; bounds: Float64Array; depth: number }[];
        if (splits[record]) {
            const cut = cutOf(nodeOffset);
            entries = cut.clusters.map((cluster, k) => ({ root: cut.roots[k], bounds: cluster.bounds, depth: cluster.depth }));
        } else {
            entries = [{ root: { tree: nodeOffset, copy: -1 }, bounds: unionOfChildren(summary.reader, 0), depth: summary.depths.get(0)! }];
        }
        for (const entry of entries) {
            transformBounds(entry.bounds, objectToWorld, box, 0);
            if (entry.root.tree >= 0) enteredWhole.add(entry.root.tree);
            sources.push(record);
            roots.push(entry.root);
            depths.push(entry.depth);
            for (let k = 0; k < 6; k++) worldBoxes.push(box[k]);
        }
    }

    // The new node buffer: the scene's trees some record still enters at the root (and those no
    // record traces), then the copies. A tree only split records trace is dropped, as
    // ClusteredBVH uploads only the referenced subtrees. A scene whose records enter a root no
    // placement describes keeps its node buffer whole.
    const lead = quantized ? 1 : 0;
    const placements = [...scene.meshes, ...scene.mergedGroups].filter((placement) => placement.nodeCount > 0);
    const described = new Set(placements.map((placement) => placement.nodeOffset));
    const compact = [...sharedBy.keys()].every((root) => described.has(root));
    const dropped = new Set<number>();
    if (compact) for (const root of cuts.keys()) if (!enteredWhole.has(root)) dropped.add(root);
    const moved = new Map<number, number>();
    const pieces: { from: number; slots: number; to: number }[] = [];
    let slot = 0;
    if (compact) {
        for (const placement of [...placements].sort((a, b) => a.nodeOffset - b.nodeOffset)) {
            if (dropped.has(placement.nodeOffset) || moved.has(placement.nodeOffset)) continue;
            pieces.push({ from: placement.nodeOffset - lead, slots: lead + placement.nodeCount, to: slot });
            moved.set(placement.nodeOffset, slot + lead);
            slot += lead + placement.nodeCount;
        }
    } else {
        pieces.push({ from: 0, slots: scene.nodes.byteLength / nodeBytes, to: 0 });
        slot = scene.nodes.byteLength / nodeBytes;
    }
    const keptSlots = slot;
    const copyRoots = copies.map((copy) => {
        const root = slot + lead;
        slot += copy.length / slotWords;
        return root;
    });
    const resolve = (root: Root): number => (root.copy >= 0 ? copyRoots[root.copy] : compact ? moved.get(root.tree)! : root.tree);

    const tree = buildSahTree(new Float32Array(worldBoxes), { ...DEFAULT_SAH_OPTIONS, ...options.topLevel, maxLeafPrimitives: 1 });
    const clusterRecords = sources.length;
    const recordBytes = clusterRecords * INSTANCE_RECORD_BYTES;
    const treeBytes = tree.nodeCount * BVH_NODE_BYTES;
    const buffer = new ArrayBuffer(TOP_LEVEL_HEADER_BYTES + recordBytes + treeBytes);
    const words = new Uint32Array(buffer);
    words[TOP_HEADER_WORD.instanceCount] = clusterRecords;
    words[TOP_HEADER_WORD.nodeCount] = tree.nodeCount;
    words[TOP_HEADER_WORD.maxDepth] = tree.maxDepth;
    const recordOrder = new Uint32Array(clusterRecords);
    let maxMeshDepth = 0;
    for (let at = 0; at < clusterRecords; at++) {
        const entry = tree.order[at];
        const from = recordWord(sources[entry]);
        const to = recordWord(at);
        words.set(topWords.subarray(from, from + INSTANCE_RECORD_WORDS), to);
        words[to + INSTANCE_WORD.rootNodeOffset] = resolve(roots[entry]);
        recordOrder[at] = top.recordOrder[sources[entry]];
        maxMeshDepth = Math.max(maxMeshDepth, depths[entry]);
    }
    new Uint8Array(buffer, TOP_LEVEL_HEADER_BYTES + recordBytes, treeBytes).set(new Uint8Array(tree.nodes));

    const requiredStackSize = tree.maxDepth + 1 + maxMeshDepth;
    if (requiredStackSize > TRACE_STACK_SIZE) {
        throw new RangeError(
            `clusterTraceScene: the trace stack holds ${TRACE_STACK_SIZE} entries but the clustered scene can need ${requiredStackSize} ` +
                `(top level ${tree.maxDepth} + marker + deepest tree ${maxMeshDepth}); raise primitiveLimit`,
        );
    }

    const nodes = new ArrayBuffer(Math.max(1, slot) * nodeBytes);
    const nodeWords = new Uint32Array(nodes);
    const sceneWords = new Uint32Array(scene.nodes);
    for (const piece of pieces) nodeWords.set(sceneWords.subarray(piece.from * slotWords, (piece.from + piece.slots) * slotWords), piece.to * slotWords);
    copies.forEach((copy, k) => nodeWords.set(copy, (copyRoots[k] - lead) * slotWords));

    // Placements follow their trees; a dropped tree's entry keeps its triangles and says so.
    const place = <T extends TraceSceneMeshPlacement>(placement: T): T => {
        if (placement.nodeCount === 0 || !compact) return placement;
        if (dropped.has(placement.nodeOffset)) return { ...placement, nodeCount: 0, clustered: true };
        return { ...placement, nodeOffset: moved.get(placement.nodeOffset)! };
    };

    const topLevel: TopLevelBvh = { buffer, instanceCount: clusterRecords, nodeCount: tree.nodeCount, maxDepth: tree.maxDepth, skippedSingular: top.skippedSingular, recordOrder };
    const byteLength = nodes.byteLength + scene.triangles.byteLength + buffer.byteLength + scene.positions.byteLength;
    let droppedNodeSlots = 0;
    for (const placement of placements) if (dropped.has(placement.nodeOffset)) droppedNodeSlots += lead + placement.nodeCount;
    return {
        ...scene,
        nodes,
        topLevel,
        meshes: scene.meshes.map(place),
        mergedGroups: scene.mergedGroups.map(place),
        maxMeshDepth,
        requiredStackSize,
        byteLength,
        clusters: {
            records: recordCount,
            splitRecords,
            clusterRecords,
            copiedSubtrees: copies.length,
            copiedNodeSlots: slot - keptSlots,
            droppedTrees: dropped.size,
            droppedNodeSlots,
            addedBytes: byteLength - scene.byteLength,
        },
    };
}

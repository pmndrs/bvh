/**
 * three-mesh-bvh's serialized tree (`MeshBVH.serialize` / `MeshBVH.deserialize`,
 * 0.9.18, PARITY 7.1), read and written next to our own format
 * (`serializeLayout`), so an app can hand trees to and from three-mesh-bvh.
 * Engine-free: `@pmndrs/bvh/three` wraps it as `MeshBVH.serialize` and
 * `MeshBVH.deserialize` with three's geometry.
 *
 * **Their format** (`src/core/MeshBVH.js`, `src/core/build/buildTree.js`,
 * `src/core/utils/nodeBufferUtils.js`; MIT): a plain object
 * `{ version: 1, roots, index, indirectBuffer }`. Each root is an
 * `ArrayBuffer` of 32-byte nodes in depth-first order:
 *
 *     float32[0..5]  the node's own box (min xyz, max xyz)
 *     uint32[6]      leaf: first triangle position; inner: the right child, as a node offset from this node
 *     uint32[7]      inner: split axis (0, 1, 2)
 *     uint16[14]     leaf: triangle count
 *     uint16[15]     0xffff marks a leaf
 *
 * The left child of an inner node is the next node. Version 0 files stored the
 * right child as an absolute `uint32` index; they are read as three-mesh-bvh
 * reads them (its `fixupVersion0`). A tree position `p` is the triangle
 * `indirectBuffer[p]` (indirect mode), or triangle `p` of `index`, which
 * three-mesh-bvh reordered in place (otherwise). A geometry with groups has
 * one root per group in the second mode.
 *
 * **Writing** ({@link serializeThreeMeshBvh}) always gives indirect mode: our
 * trees never reorder the geometry (`CpuMeshBvh`, "Triangle indices"), so
 * `indirectBuffer` is each tree position's original triangle and `index` is a
 * copy of the geometry's own (null without one). One root, our nodes and boxes
 * as they are (the root's box is the tree's bounds), the split axis as
 * `CpuMeshBvh.traverseNodes` gives it. three-mesh-bvh reads it with
 * `MeshBVH.deserialize(data, geometry)` (pass `setIndex: false` for a
 * geometry without an index: its `setIndex` makes an index from `data.index`,
 * which is null there).
 *
 * **Reading** ({@link meshBvhFromThreeMeshBvh}) keeps the tree's shape (which
 * triangles share a leaf, which nodes are whose children) and makes our
 * layout from it with `buildMeshBvhFromTopology`: boxes are recomputed from
 * the triangles' f32 corners, so they are exact, and leaves of more than 8
 * triangles (three-mesh-bvh's default leaf size is 10) are cut in half until
 * they fit. Several roots are joined under new inner nodes, pairwise. The tree
 * must hold every triangle of the geometry once: one built over a draw range
 * or groups that leave triangles out can't be read (our trees hold every triangle).
 */

import { buildMeshBvhFromTopology, type MeshTreeTopology } from '../trace/mesh-bvh-topology.ts';
import type { MeshBvh, MeshGeometry } from '../trace/mesh-bvh.ts';
import { singleLeafTopology, TOPOLOGY_WORDS_PER_NODE } from '../topology.ts';
import { countTriangles } from '../trace/mesh-bvh.ts';
import { BvhBox3 } from './shapes.ts';
import type { CpuMeshBvh } from './cpu-mesh-bvh.ts';

/** Bytes, `uint32` words and `uint16` halves per three-mesh-bvh node. */
export const THREE_MESH_BVH_NODE_BYTES = 32;
const WORDS = 8;
const HALVES = 16;
const LEAF_FLAG = 0xffff;
/** The format version {@link serializeThreeMeshBvh} writes, three-mesh-bvh 0.9's. */
export const THREE_MESH_BVH_SERIALIZED_VERSION = 1;

/** three-mesh-bvh's `SerializedBVH`: what `MeshBVH.serialize` returns and `MeshBVH.deserialize` takes. */
export interface ThreeMeshBvhSerialized {
    /** 1 for three-mesh-bvh 0.7 and later; missing (0) for older files. */
    version?: number;
    /** One buffer of 32-byte nodes per root (one per group without `indirect`). */
    roots: ArrayBuffer[];
    /** The geometry's index (reordered by three-mesh-bvh without `indirect`), or null. */
    index: Uint16Array | Uint32Array | Int32Array | null;
    /** Tree position to triangle (indirect mode), or null. */
    indirectBuffer: Uint16Array | Uint32Array | null;
}

export interface SerializeThreeMeshBvhOptions {
    /**
     * The geometry's index, copied into `index` (three-mesh-bvh's deserialize
     * sets it on the geometry with `setIndex`). Default null, as for a geometry
     * without one.
     */
    readonly index?: Uint16Array | Uint32Array | Int32Array | null;
    /** As three-mesh-bvh's: copy `index` (default true) or hand over the same array. The roots and `indirectBuffer` are always new. */
    readonly cloneBuffers?: boolean;
}

/**
 * Writes `tree` in three-mesh-bvh's serialized format (indirect mode; see the
 * file comment). three-mesh-bvh's `MeshBVH.deserialize(data, geometry)` reads it.
 */
export function serializeThreeMeshBvh(tree: CpuMeshBvh<unknown>, options: SerializeThreeMeshBvhOptions = {}): ThreeMeshBvhSerialized {
    const boxes: Float32Array[] = [];
    const leaf: boolean[] = [];
    const first: number[] = [];
    const counts: number[] = [];
    const right: number[] = [];
    const lastAtDepth: number[] = [];
    const leftSeen: boolean[] = [];
    // Depth first, left before right: three-mesh-bvh's node order. A node's parent is the last node one level up.
    tree.traverseNodes((depth, isLeaf, box, offsetOrSplit, count) => {
        const node = boxes.length;
        boxes.push(box);
        leaf.push(isLeaf);
        first.push(offsetOrSplit);
        counts.push(count);
        right.push(0);
        leftSeen.push(false);
        if (depth > 0) {
            const parent = lastAtDepth[depth - 1];
            if (leftSeen[parent]) right[parent] = node - parent;
            else leftSeen[parent] = true;
        }
        lastAtDepth[depth] = node;
    });
    const root = new ArrayBuffer(boxes.length * THREE_MESH_BVH_NODE_BYTES);
    const floats = new Float32Array(root);
    const words = new Uint32Array(root);
    const halves = new Uint16Array(root);
    for (let node = 0; node < boxes.length; node++) {
        floats.set(boxes[node], node * WORDS);
        if (leaf[node]) {
            words[node * WORDS + 6] = first[node];
            halves[node * HALVES + 14] = counts[node];
            halves[node * HALVES + 15] = LEAF_FLAG;
        } else {
            words[node * WORDS + 6] = right[node];
            words[node * WORDS + 7] = first[node];
        }
    }
    // three-mesh-bvh's choice of width (`generateIndirectBuffer`): 32 bits past 2^16 triangles.
    const triangleCount = tree.triangleCount;
    const indirectBuffer = triangleCount > 2 ** 16 ? new Uint32Array(triangleCount) : new Uint16Array(triangleCount);
    for (let position = 0; position < triangleCount; position++) indirectBuffer[position] = tree.resolveTriangleIndex(position);
    const index = options.index ?? null;
    return {
        version: THREE_MESH_BVH_SERIALIZED_VERSION,
        roots: [root],
        index: index && (options.cloneBuffers ?? true) ? (index.slice() as typeof index) : index,
        indirectBuffer,
    };
}

/** A child as the topology takes it: a leaf's positions, or an inner node. */
interface ChildRef {
    readonly index: number;
    readonly count: number;
}

/**
 * The shape of a three-mesh-bvh tree as our Builder contract's topology:
 * `order` is the indirect buffer (indirect mode) or the identity (then tree
 * positions are triangles of the index three-mesh-bvh wrote). `triangleCount`
 * is the geometry's.
 */
export function topologyFromThreeMeshBvh(data: ThreeMeshBvhSerialized, triangleCount: number): MeshTreeTopology {
    const roots = data.roots;
    if (!Array.isArray(roots) || roots.length === 0) throw new RangeError('three-mesh-bvh data: no roots');
    const version = data.version ?? 0;
    if (version !== 0 && version !== 1) throw new RangeError(`three-mesh-bvh data: unknown format version ${version}`);
    let order: Uint32Array;
    if (data.indirectBuffer) {
        if (roots.length !== 1) throw new RangeError(`three-mesh-bvh data: an indirect tree has one root, got ${roots.length}`);
        if (data.indirectBuffer.length !== triangleCount) {
            throw new RangeError(
                `three-mesh-bvh data: the tree holds ${data.indirectBuffer.length} triangles, the geometry ${triangleCount}; a tree over a draw range or groups that leave triangles out can't be read (ours hold every triangle)`,
            );
        }
        order = Uint32Array.from(data.indirectBuffer);
    } else {
        order = new Uint32Array(triangleCount);
        for (let i = 0; i < triangleCount; i++) order[i] = i;
    }

    // Our inner nodes: the combiners joining several roots come first (node 0 is the root), then each root's.
    const children: number[] = new Array<number>(Math.max(0, roots.length - 1) * TOPOLOGY_WORDS_PER_NODE).fill(0);
    let innerCount = roots.length - 1;
    let covered = 0;
    const refs: ChildRef[] = roots.map((buffer, r) => {
        // An ArrayBuffer, or a SharedArrayBuffer with `useSharedArrayBuffer` (a view would be read by value, not as nodes).
        if (!(buffer instanceof ArrayBuffer) && !(typeof SharedArrayBuffer !== 'undefined' && (buffer as unknown) instanceof SharedArrayBuffer)) {
            throw new RangeError(`three-mesh-bvh data: root ${r} is not an ArrayBuffer`);
        }
        const bytes = buffer as ArrayBufferLike;
        if (bytes.byteLength === 0 || bytes.byteLength % THREE_MESH_BVH_NODE_BYTES !== 0) throw new RangeError(`three-mesh-bvh data: root ${r} is ${bytes.byteLength} bytes, not whole 32-byte nodes`);
        const words = new Uint32Array(bytes);
        const halves = new Uint16Array(bytes);
        const nodeCount = bytes.byteLength / THREE_MESH_BVH_NODE_BYTES;
        const isLeaf = (node: number) => halves[node * HALVES + 15] === LEAF_FLAG;
        const rightOf = (node: number) => (version === 0 ? words[node * WORDS + 6] / WORDS : node + words[node * WORDS + 6]);
        const refOf = (node: number, inner: Int32Array): ChildRef => {
            if (!isLeaf(node)) return { index: inner[node], count: 0 };
            const count = halves[node * HALVES + 14];
            const offset = words[node * WORDS + 6];
            if (count === 0) throw new RangeError(`three-mesh-bvh data: root ${r} has an empty leaf (node ${node})`);
            if (offset + count > triangleCount) throw new RangeError(`three-mesh-bvh data: root ${r}'s leaf at node ${node} reads triangles ${offset}-${offset + count - 1} of ${triangleCount}`);
            covered += count;
            return { index: offset, count };
        };
        const inner = new Int32Array(nodeCount).fill(-1);
        for (let node = 0; node < nodeCount; node++) if (!isLeaf(node)) inner[node] = innerCount++;
        for (let node = 0; node < nodeCount; node++) {
            if (isLeaf(node)) continue;
            const left = node + 1;
            const rightNode = rightOf(node);
            if (!(left < nodeCount) || !Number.isInteger(rightNode) || !(rightNode > left && rightNode < nodeCount)) {
                throw new RangeError(`three-mesh-bvh data: root ${r}'s node ${node} has children out of range (version ${version})`);
            }
            const l = refOf(left, inner);
            const rr = refOf(rightNode, inner);
            children.push(l.index, l.count, rr.index, rr.count);
        }
        return refOf(0, inner);
    });
    if (covered !== triangleCount) {
        throw new RangeError(
            `three-mesh-bvh data: the tree's leaves hold ${covered} triangles, the geometry ${triangleCount}; a tree over a draw range or groups that leave triangles out can't be read (ours hold every triangle)`,
        );
    }

    if (refs.length === 1) {
        const only = refs[0];
        if (only.count > 0) return singleLeafTopology(order, only.index, only.count);
        return { order, children: Uint32Array.from(children) };
    }
    // Join the roots pairwise: combiner nodes are numbered in the order they are made, the first being the root.
    let nextCombiner = 0;
    const join = (lo: number, hi: number): ChildRef => {
        if (hi - lo === 1) return refs[lo];
        const node = nextCombiner++;
        const mid = (lo + hi) >> 1;
        const left = join(lo, mid);
        const rightRef = join(mid, hi);
        children.splice(node * TOPOLOGY_WORDS_PER_NODE, TOPOLOGY_WORDS_PER_NODE, left.index, left.count, rightRef.index, rightRef.count);
        return { index: node, count: 0 };
    };
    join(0, refs.length);
    return { order, children: Uint32Array.from(children) };
}

/**
 * Our layout (a `MeshBvh`: 64-byte nodes, 48-byte triangle records) for a
 * three-mesh-bvh tree over `geometry` (see the file comment). Without
 * `indirectBuffer`, tree positions are triangles of the index three-mesh-bvh
 * reordered, so `geometry.indices` must be that index (`data.index`, or the
 * geometry's own after three-mesh-bvh built on it).
 */
export function meshBvhFromThreeMeshBvh(data: ThreeMeshBvhSerialized, geometry: MeshGeometry): MeshBvh {
    return buildMeshBvhFromTopology(geometry, topologyFromThreeMeshBvh(data, countTriangles(geometry)));
}

/** One node of {@link getJSONStructure}: three-mesh-bvh's plain-object tree. */
export interface BvhJsonNode<Box = BvhBox3> {
    bounds: Box;
    /** Leaves: the triangle count. */
    count?: number;
    /** Leaves: the first tree position. */
    offset?: number;
    /** Inner nodes: the children. */
    left?: BvhJsonNode<Box> | null;
    right?: BvhJsonNode<Box> | null;
}

/**
 * three-mesh-bvh's `getJSONStructure(bvh)` (`debug/Debug.js`, 0.9.18): the
 * tree as plain objects, each with `bounds` and either `count` and `offset`
 * (a leaf; tree positions, as `CpuMeshBvh.traverse` reports them) or `left`
 * and `right`. Built the way three-mesh-bvh builds it, over the same node
 * walk (`CpuMeshBvh.traverseNodes`, the root at depth 0 with the tree's
 * bounds). `makeBox` turns `[minX, minY, minZ, maxX, maxY, maxZ]` into a box:
 * a {@link BvhBox3} here, a three `Box3` in `@pmndrs/bvh/three`.
 */
export function getJSONStructure<Box = BvhBox3>(tree: CpuMeshBvh<unknown>, makeBox?: (box: Float32Array) => Box): BvhJsonNode<Box> {
    const toBox =
        makeBox ??
        ((array: Float32Array) => {
            const box = new BvhBox3();
            box.min.x = array[0];
            box.min.y = array[1];
            box.min.z = array[2];
            box.max.x = array[3];
            box.max.y = array[4];
            box.max.z = array[5];
            return box as unknown as Box;
        });
    const depthStack: BvhJsonNode<Box>[] = [];
    tree.traverseNodes((depth, isLeaf, box, offset, count) => {
        const info: BvhJsonNode<Box> = { bounds: toBox(box) };
        if (isLeaf) {
            info.count = count;
            info.offset = offset;
        } else {
            info.left = null;
            info.right = null;
        }
        depthStack[depth] = info;
        // The walk visits the left child, then the right.
        const parent = depth > 0 ? depthStack[depth - 1] : undefined;
        if (parent) {
            if (parent.left === null) parent.left = info;
            else parent.right = info;
        }
    });
    return depthStack[0];
}

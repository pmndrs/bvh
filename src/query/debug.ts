/**
 * three-mesh-bvh's debug functions (`debug/Debug.js`, 0.9.18) on a
 * {@link CpuMeshBvh}: `getBVHExtremes`, `validateBounds` and
 * `estimateMemoryInBytes`, with its output shapes. They walk the tree with
 * {@link CpuMeshBvh.traverseNodes}, three-mesh-bvh's node walk (the root at
 * depth 0). `@pmndrs/bvh/three` exports the same functions.
 */

import type { CpuMeshBvh } from './cpu-mesh-bvh.ts';

/**
 * What the debug functions read of a tree: a {@link CpuMeshBvh}, or a points
 * or segments tree (`CpuPointsBvh`, `CpuSegmentsBvh`), which walk the same way.
 */
export type BvhDebugTree = Pick<CpuMeshBvh<unknown>, 'traverseNodes' | 'writePrimitiveRangeBounds' | 'estimateMemoryInBytes'>;

/** three-mesh-bvh's cost constants (`core/Constants.js`), as `getBVHExtremes` scores with them. */
export const BVH_TRAVERSAL_COST = 1;
export const BVH_PRIMITIVE_INTERSECT_COST = 1.25;

/** One root's figures, three-mesh-bvh's `ExtremeInfo`. */
export interface BvhExtremes {
    /** Every node, leaves included (the root counts). */
    nodeCount: number;
    leafNodeCount: number;
    /**
     * Sum over the nodes of each box's surface area, times
     * {@link BVH_TRAVERSAL_COST} for an inner node or
     * {@link BVH_PRIMITIVE_INTERSECT_COST} times its triangles for a leaf. Not
     * normalised by the root's area. Lower is better; it grows as refits
     * degrade a tree.
     */
    surfaceAreaScore: number;
    /** Shallowest and deepest leaf (the root is at depth 0). */
    depth: { min: number; max: number };
    /** Fewest and most triangles in a leaf. */
    primitives: { min: number; max: number };
    /**
     * Inner nodes split along x, y, z. Our layout doesn't store the axis:
     * it is taken as the one along which the children's centres lie furthest
     * apart ({@link CpuMeshBvh.traverseNodes}).
     */
    splits: [number, number, number];
}

/**
 * three-mesh-bvh's `getBVHExtremes(bvh)`: node and leaf counts, the leaves'
 * depth and size ranges, the split axes and a surface-area score, one entry
 * per root (our trees have one root).
 */
export function getBVHExtremes(tree: BvhDebugTree): BvhExtremes[] {
    const result: BvhExtremes = {
        nodeCount: 0,
        leafNodeCount: 0,
        depth: { min: Infinity, max: -Infinity },
        primitives: { min: Infinity, max: -Infinity },
        splits: [0, 0, 0],
        surfaceAreaScore: 0,
    };
    tree.traverseNodes((depth, isLeaf, box, offsetOrSplit, count) => {
        const l0 = box[3] - box[0];
        const l1 = box[4] - box[1];
        const l2 = box[5] - box[2];
        const surfaceArea = 2 * (l0 * l1 + l1 * l2 + l2 * l0);
        result.nodeCount++;
        if (isLeaf) {
            result.leafNodeCount++;
            result.depth.min = Math.min(depth, result.depth.min);
            result.depth.max = Math.max(depth, result.depth.max);
            result.primitives.min = Math.min(count, result.primitives.min);
            result.primitives.max = Math.max(count, result.primitives.max);
            result.surfaceAreaScore += surfaceArea * BVH_PRIMITIVE_INTERSECT_COST * count;
        } else {
            result.splits[offsetOrSplit]++;
            result.surfaceAreaScore += surfaceArea * BVH_TRAVERSAL_COST;
        }
    });
    if (result.primitives.min === Infinity) result.primitives = { min: 0, max: 0 };
    if (result.depth.min === Infinity) result.depth = { min: 0, max: 0 };
    return [result];
}

function containsBox(outer: ArrayLike<number>, inner: ArrayLike<number>): boolean {
    return outer[0] <= inner[0] && inner[3] <= outer[3] && outer[1] <= inner[1] && inner[4] <= outer[4] && outer[2] <= inner[2] && inner[5] <= outer[5];
}

/**
 * three-mesh-bvh's `validateBounds(bvh)`: whether every leaf's box holds its
 * triangles and every box holds its children. Each failure is reported with
 * `console.assert`, as three-mesh-bvh does. The triangles are read from the
 * tree's source geometry when it has one ({@link CpuMeshBvh.writePrimitiveRangeBounds}),
 * so positions edited without a {@link CpuMeshBvh.refit} fail, as they do in
 * three-mesh-bvh. (`checkGpuLayout` checks the layout itself, bit for bit.)
 */
export function validateBounds(tree: BvhDebugTree): boolean {
    const boxes: Float32Array[] = [];
    const primitives = new Float64Array(6);
    let passes = true;
    tree.traverseNodes((depth, isLeaf, box, offset, count) => {
        boxes[depth] = box;
        if (isLeaf) {
            tree.writePrimitiveRangeBounds(offset, count, primitives, 0);
            const contained = containsBox(box, primitives);
            console.assert(contained, 'Leaf bounds does not fully contain primitives.');
            passes &&= contained;
        }
        const parent = depth > 0 ? boxes[depth - 1] : undefined;
        if (parent) {
            const contained = containsBox(parent, box);
            console.assert(contained, 'Parent bounds does not fully contain child.');
            passes &&= contained;
        }
    });
    return passes;
}

/**
 * The boxes three-mesh-bvh's `BVHHelper` draws, as corner positions: 8
 * corners per box (x outermost, then y, then z; min before max), 24 numbers.
 * A box is drawn when it is a leaf or at depth `depth - 1` (the walk stops
 * there), and, with `displayParents`, every box above those too. Empty boxes
 * are skipped. The root is at depth 0, so `depth` 1 draws the root's box.
 */
export function bvhHelperPositions(tree: BvhDebugTree, depth: number, displayParents: boolean): Float32Array {
    const targetDepth = depth - 1;
    const boxes: Float32Array[] = [];
    tree.traverseNodes((nodeDepth, isLeaf, box) => {
        if (box[3] < box[0] || box[4] < box[1] || box[5] < box[2]) return true;
        const terminate = nodeDepth >= targetDepth || isLeaf;
        if (terminate || displayParents) boxes.push(box);
        return terminate;
    });
    const positions = new Float32Array(boxes.length * 24);
    let at = 0;
    for (const box of boxes) {
        for (let x = 0; x < 2; x++) {
            for (let y = 0; y < 2; y++) {
                for (let z = 0; z < 2; z++) {
                    positions[at++] = box[x * 3];
                    positions[at++] = box[1 + y * 3];
                    positions[at++] = box[2 + z * 3];
                }
            }
        }
    }
    return positions;
}

/** Corner pairs of a box's 12 edges ({@link bvhHelperPositions}' corner order), three-mesh-bvh's. */
const BOX_EDGES = [0, 4, 1, 5, 2, 6, 3, 7, 0, 2, 1, 3, 4, 6, 5, 7, 0, 1, 2, 3, 4, 5, 6, 7];
/** Corner triples of a box's 12 faces' triangles, three-mesh-bvh's. */
const BOX_FACES = [0, 1, 2, 2, 1, 3, 4, 6, 5, 6, 7, 5, 1, 4, 5, 0, 4, 1, 2, 3, 6, 3, 7, 6, 0, 2, 4, 2, 6, 4, 1, 5, 3, 3, 5, 7];

/**
 * The index buffer three-mesh-bvh's `BVHHelper` draws {@link bvhHelperPositions}
 * with: 12 edges per box as line segments (`edges`), or 12 triangles. 16-bit
 * when `positions` has at most 65,535 numbers, as three-mesh-bvh decides.
 */
export function bvhHelperIndices(positions: Float32Array, edges: boolean): Uint16Array | Uint32Array {
    const boxCount = positions.length / 24;
    const pattern = edges ? BOX_EDGES : BOX_FACES;
    const indices = positions.length > 65535 ? new Uint32Array(pattern.length * boxCount) : new Uint16Array(pattern.length * boxCount);
    for (let box = 0; box < boxCount; box++) {
        for (let i = 0; i < pattern.length; i++) indices[box * pattern.length + i] = box * 8 + pattern[i];
    }
    return indices;
}

/**
 * three-mesh-bvh's `estimateMemoryInBytes(bvh)`: the bytes the tree holds,
 * {@link CpuMeshBvh.estimateMemoryInBytes} (the layout, its one-instance top
 * level and lookups made so far; not the source geometry). three-mesh-bvh
 * walks the object graph instead, which counts its geometry's index.
 */
export function estimateMemoryInBytes(tree: BvhDebugTree): number {
    return tree.estimateMemoryInBytes();
}

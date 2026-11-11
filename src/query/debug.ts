/**
 * three-mesh-bvh's debug functions (`debug/Debug.js`, 0.9.18) on a
 * {@link CpuMeshBvh}: `getBVHExtremes`, `validateBounds` and
 * `estimateMemoryInBytes`, with its output shapes. They walk the tree with
 * {@link CpuMeshBvh.traverseNodes}, three-mesh-bvh's node walk (the root at
 * depth 0). `@pmndrs/bvh/three` exports the same functions.
 */

import type { CpuMeshBvh } from './cpu-mesh-bvh.ts';

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
export function getBVHExtremes(tree: CpuMeshBvh<unknown>): BvhExtremes[] {
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
export function validateBounds(tree: CpuMeshBvh<unknown>): boolean {
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
 * three-mesh-bvh's `estimateMemoryInBytes(bvh)`: the bytes the tree holds,
 * {@link CpuMeshBvh.estimateMemoryInBytes} (the layout, its one-instance top
 * level and lookups made so far; not the source geometry). three-mesh-bvh
 * walks the object graph instead, which counts its geometry's index.
 */
export function estimateMemoryInBytes(tree: CpuMeshBvh<unknown>): number {
    return tree.estimateMemoryInBytes();
}

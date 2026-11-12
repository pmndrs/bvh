// Types for bvh-tree 1.0.1 (benraziel/bvh-tree), which ships none: only what the adapter uses.
declare module 'bvh-tree' {
    export interface BvhTreePoint {
        x: number;
        y: number;
        z: number;
    }
    export interface BvhTreeNode {
        _extentsMin: BvhTreePoint;
        _extentsMax: BvhTreePoint;
        _startIndex: number;
        _endIndex: number;
        _node0: BvhTreeNode | null;
        _node1: BvhTreeNode | null;
    }
    export interface BvhTreeIntersection {
        triangle: BvhTreePoint[];
        triangleIndex: number;
        intersectionPoint: BvhTreePoint;
    }
    export class BVH {
        constructor(triangles: BvhTreePoint[][], maxTrianglesPerNode?: number);
        intersectRay(origin: BvhTreePoint, direction: BvhTreePoint, backfaceCulling: boolean): BvhTreeIntersection[];
        _trianglesArray: number[];
        _bboxArray: Float32Array;
        _bboxHelper: Float32Array;
        _rootNode: BvhTreeNode;
    }
    const bvhtree: { BVH: typeof BVH };
    export default bvhtree;
}

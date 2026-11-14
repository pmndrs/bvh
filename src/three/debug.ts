/**
 * three-mesh-bvh's `getJSONStructure` with three's `Box3` (PARITY 7.5): the
 * core's (`src/query/three-mesh-bvh-format.ts`), whose boxes are made as
 * three-mesh-bvh makes them (`arrayToBox` into a new `Box3`).
 */

import { Box3, Vector3 } from 'three';
import { getJSONStructure as coreJsonStructure, type BvhJsonNode } from '../query/three-mesh-bvh-format.ts';
import type { CpuMeshBvh } from '../query/cpu-mesh-bvh.ts';

/**
 * three-mesh-bvh's `getJSONStructure(bvh)`: the tree as plain objects, each
 * with `bounds` (a `Box3`) and either `count` and `offset` (a leaf; tree
 * positions) or `left` and `right`. Any tree of ours (`geometry.boundsTree`,
 * a `CpuMeshBvh`).
 */
export function getJSONStructure(tree: CpuMeshBvh<unknown>): BvhJsonNode<Box3> {
    return coreJsonStructure(tree, (box) => new Box3(new Vector3(box[0], box[1], box[2]), new Vector3(box[3], box[4], box[5])));
}

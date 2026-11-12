/**
 * bvh-tree 1.0.1 (MIT, Ben Raziel): a plain-JS triangle BVH, split at the
 * centre of the longest axis, up to 10 triangles per leaf. Its one query is
 * `intersectRay`, every hit.
 *
 * Its input is an array of triangles of `{x, y, z}` corners; building that
 * array from the typed arrays is part of its build time, because an app has to
 * do it to use the library.
 */

import bvhtree, { type BvhTreeNode } from 'bvh-tree';
import { features, type MeshAdapter, type TreeQuality } from './types.ts';

const TRAVERSAL_COST = 1;
const TRIANGLE_COST = 1.25;

function area(node: BvhTreeNode): number {
    const x = node._extentsMax.x - node._extentsMin.x;
    const y = node._extentsMax.y - node._extentsMin.y;
    const z = node._extentsMax.z - node._extentsMin.z;
    return 2 * (x * y + y * z + z * x);
}

/** SAH with three-mesh-bvh's costs over the root's area, depth and node counts, by walking its node objects. */
function quality(root: BvhTreeNode, exactBytes: number): TreeQuality {
    let score = 0;
    let nodes = 0;
    let leaves = 0;
    let depth = 0;
    const stack: [BvhTreeNode, number][] = [[root, 0]];
    while (stack.length) {
        const [node, d] = stack.pop()!;
        nodes++;
        if (node._node0 || node._node1) {
            score += area(node) * TRAVERSAL_COST;
            if (node._node0) stack.push([node._node0, d + 1]);
            if (node._node1) stack.push([node._node1, d + 1]);
        } else {
            leaves++;
            depth = Math.max(depth, d);
            score += area(node) * TRIANGLE_COST * (node._endIndex - node._startIndex);
        }
    }
    return { sah: score / area(root), depth, nodes, leaves, bytes: exactBytes, note: 'bytes: its arrays only; the node objects are not counted (see the Node heap figure)' };
}

export const bvhTreeAdapter: MeshAdapter = {
    id: 'bvh-tree',
    name: 'bvh-tree',
    library: 'bvh-tree@1.0.1',
    license: 'MIT',
    buildMethod: 'new BVH(triangles, 10): the triangle-of-points array it takes is built from the typed arrays first (timed), then its centre-split build',
    features: features({
        raycastFirst: { status: 'partial', note: 'no first-hit query: every hit is found and the nearest taken' },
        raycastAll: 'yes',
        treeStats: { status: 'partial', note: 'no API; read from its node objects here' },
    }),
    build(geometry) {
        const { positions, indices } = geometry;
        const corner = (v: number) => ({ x: positions[v * 3], y: positions[v * 3 + 1], z: positions[v * 3 + 2] });
        const triangles = new Array(geometry.triangleCount);
        for (let i = 0; i < geometry.triangleCount; i++) triangles[i] = [corner(indices[i * 3]), corner(indices[i * 3 + 1]), corner(indices[i * 3 + 2])];
        const bvh = new bvhtree.BVH(triangles, 10);
        const origin = { x: 0, y: 0, z: 0 };
        const direction = { x: 0, y: 0, z: 0 };
        const all = (ray: { origin: number[]; direction: number[] }) => {
            origin.x = ray.origin[0];
            origin.y = ray.origin[1];
            origin.z = ray.origin[2];
            direction.x = ray.direction[0];
            direction.y = ray.direction[1];
            direction.z = ray.direction[2];
            return bvh
                .intersectRay(origin, direction, false)
                .map((hit) => ({ triangle: hit.triangleIndex, t: Math.hypot(hit.intersectionPoint.x - origin.x, hit.intersectionPoint.y - origin.y, hit.intersectionPoint.z - origin.z) }))
                .sort((a, b) => a.t - b.t);
        };
        // A JS array of doubles (8 bytes each), two Float32Arrays of 7 floats per triangle.
        const exactBytes = bvh._trianglesArray.length * 8 + bvh._bboxArray.byteLength + bvh._bboxHelper.byteLength;
        return {
            raycastFirst: (ray) => all(ray)[0] ?? null,
            raycastAll: all,
            memory() {
                return { bytes: exactBytes, method: 'its arrays (9 doubles and 2 x 7 floats per triangle); its node objects are not counted, so this is a lower bound' };
            },
            quality: () => quality(bvh._rootNode, exactBytes),
            dispose() {},
        };
    },
};

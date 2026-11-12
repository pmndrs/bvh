// Code samples for docs/cpu-queries/*.mdx (see getting-started.ts for how they are checked).

// #region create
import { BVH_DOUBLE_SIDE, CpuMeshBvh, type MeshInput } from '@pmndrs/bvh';

// #endregion

export async function create(input: MeshInput): Promise<CpuMeshBvh> {
    // #region create
    const tree = await CpuMeshBvh.create(input, { maxLeafPrimitives: 8 }); // tinybvh, on this thread
    // #endregion
    return tree;
}

export function raycast(tree: CpuMeshBvh): number {
    // #region raycast
    const ray = { origin: [0, 0, 5], direction: [0, 0, -1] }; // or a three Ray
    const hits = tree.raycast(ray, { side: BVH_DOUBLE_SIDE, near: 0, far: 100 }); // every hit, nearest first
    const first = tree.raycastFirst(ray, BVH_DOUBLE_SIDE); // or null
    if (first) console.log(first.distance, first.point, first.triangle, first.u, first.v, first.frontFace);
    // #endregion
    return hits.length;
}

// #region shapecast
import { sphereIntersectsBox, sphereIntersectsTriangle } from '@pmndrs/bvh';

// #endregion

export function shapecast(tree: CpuMeshBvh): number[] {
    // #region shapecast
    const sphere = { center: { x: 0, y: 0, z: 1 }, radius: 0.5 };
    const touched: number[] = [];
    tree.shapecast({
        intersectsBounds: (box) => sphereIntersectsBox(sphere, box), // false skips the box
        intersectsTriangle: (triangle, triangleIndex) => {
            if (sphereIntersectsTriangle(sphere, triangle)) touched.push(triangleIndex); // the original index
            return false; // true stops the cast
        },
    });
    const any = tree.intersectsSphere(sphere); // the same test as one call
    // #endregion
    return any ? touched : [];
}

// #region box
import { OrientedBox } from '@pmndrs/bvh';

// #endregion

export function box(tree: CpuMeshBvh): boolean {
    // #region box
    const box = { min: { x: -1, y: -1, z: -1 }, max: { x: 1, y: 1, z: 1 } };
    const rotated = [0.7071, 0.7071, 0, 0, -0.7071, 0.7071, 0, 0, 0, 0, 1, 0, 0, 0.5, 0, 1]; // box to mesh, column-major
    const inside = tree.intersectsBox(box, rotated);

    // The oriented box and the triangle class have three-mesh-bvh's methods, over plain points.
    const obb = new OrientedBox();
    obb.min.x = obb.min.y = obb.min.z = -1;
    obb.max.x = obb.max.y = obb.max.z = 1;
    obb.needsUpdate = true;
    console.log(obb.distanceToPoint({ x: 3, y: 0, z: 0 }));
    // #endregion
    return inside;
}

export function bvhcast(tree: CpuMeshBvh, other: CpuMeshBvh, otherToTree: ArrayLike<number>): number {
    // #region bvhcast
    let pairs = 0;
    tree.bvhcast(other, otherToTree, {
        // Tree positions in each tree (three-mesh-bvh's indirect mode): map them with resolveTriangleIndex.
        intersectsTriangles: (triangle1, triangle2, i1, i2) => {
            if (triangle1.intersectsTriangle(triangle2)) {
                pairs++;
                console.log(tree.resolveTriangleIndex(i1), other.resolveTriangleIndex(i2));
            }
            return false;
        },
    });
    const touching = tree.intersectsGeometry(other, otherToTree); // a bvhcast that stops at the first pair
    // #endregion
    return touching ? pairs : 0;
}

export function closestPoint(tree: CpuMeshBvh, other: CpuMeshBvh, otherToTree: ArrayLike<number>): number {
    // #region closest
    const closest = tree.closestPointToPoint({ x: 0, y: 2, z: 0 }); // { point, distance, faceIndex } or null
    // minThreshold: stop below this distance; maxThreshold: look no farther.
    const near = tree.closestPointToPoint({ x: 0, y: 2, z: 0 }, {}, 0, 1);

    const onTree = {};
    const onOther = {};
    const between = tree.closestPointToGeometry(other, otherToTree, onTree, onOther); // onTree is between too
    // #endregion
    return (closest?.distance ?? 0) + (near?.distance ?? 0) + (between?.distance ?? 0);
}

export function hitPointInfo(tree: CpuMeshBvh, normals: Float32Array, uvs: Float32Array): void {
    // #region hit-info
    const hit = tree.raycastFirst({ origin: [0, 0, 5], direction: [0, 0, -1] });
    if (hit) {
        const info = tree.getTriangleHitPointInfo(hit.point, hit.triangle, {
            attributes: { normal: { array: normals, itemSize: 3 }, uv: { array: uvs, itemSize: 2 } },
        });
        console.log(info.barycoord, info.faceNormal, info.attributes.uv);
    }
    // #endregion
}

export function refit(tree: CpuMeshBvh, input: MeshInput): void {
    // #region refit
    input.positions[0] += 0.1; // the vertices moved, the triangles are the same
    tree.refit(); // rereads the source positions, keeps the shape; or tree.refit(input)
    // #endregion
}

// #region helpers
import { bvhHelperIndices, bvhHelperPositions, estimateMemoryInBytes, getBVHExtremes, validateBounds } from '@pmndrs/bvh';

// #endregion

export function helpers(tree: CpuMeshBvh): void {
    // #region helpers
    console.log(getBVHExtremes(tree)); // [{ nodeCount, leafNodeCount, depth, primitives, splits, surfaceAreaScore }]
    console.log(validateBounds(tree)); // every box holds its triangles (reads the source geometry)
    console.log(estimateMemoryInBytes(tree)); // the tree's own bytes, not the geometry's

    // The boxes down to depth 10 as line segments, for any engine to draw.
    const positions = bvhHelperPositions(tree, 10, false);
    const indices = bvhHelperIndices(positions, true);
    // #endregion
    console.log(indices.length);
}

// #region scene
import { buildMeshBvhWithTinyBvh, CpuSceneBvh, createTinyBvhBuilder, type MeshGeometry, type TraceSceneInstance } from '@pmndrs/bvh';

// #endregion

export async function scene(meshes: readonly MeshGeometry[], instances: readonly TraceSceneInstance[]): Promise<number> {
    // #region scene
    const tiny = await createTinyBvhBuilder();
    const picker = CpuSceneBvh.build(meshes, instances, (mesh) => buildMeshBvhWithTinyBvh(mesh, tiny));
    const hits = picker.raycast({ origin: [0, 1, 5], direction: [0, 0, -1], tMin: 0, tMax: 1e30 }, { firstHitOnly: true, mask: 0x1 });

    // Frustum culling over the top level: planes as [a, b, c, d] with a*x + b*y + c*z + d >= 0 inside.
    const visible = picker.intersectsFrustum([0, 0, -1, 10, 0, 0, 1, 10]);
    picker.setInstances(instances); // moved instances: rebuilds the top level only
    // #endregion
    return hits.length + visible.length;
}

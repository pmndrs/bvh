/**
 * @pmndrs/bvh: `CpuMeshBvh` over a tinybvh tree (the CPU queries), and the
 * same tree's layout for the GPU walk (compare/gpu).
 */

import { BVH_DOUBLE_SIDE, buildHplocOnCpu, buildLbvhOnCpu, CpuMeshBvh, createTinyBvhBuilder, getBVHExtremes, type GpuLayoutMeshBvh, type TinyBvhBuilder } from '@pmndrs/bvh';
import { features, type BuiltMesh, type CompareGeometry, type MeshAdapter, type TreeQuality } from './types.ts';

let tinyBvh: TinyBvhBuilder | null = null;

/** tinybvh's WASM, loaded once. */
export async function oursTinyBvh(): Promise<TinyBvhBuilder> {
    tinyBvh ??= await createTinyBvhBuilder();
    return tinyBvh;
}

/** SAH and shape of one of our trees, with three-mesh-bvh's costs (`getBVHExtremes`). */
export function oursQuality(tree: CpuMeshBvh): TreeQuality {
    const [extremes] = getBVHExtremes(tree);
    const b = tree.layout.bounds;
    const rootArea = 2 * ((b[3] - b[0]) * (b[4] - b[1]) + (b[4] - b[1]) * (b[5] - b[2]) + (b[5] - b[2]) * (b[3] - b[0]));
    return {
        sah: extremes.surfaceAreaScore / rootArea,
        depth: extremes.depth.max,
        // getBVHExtremes counts boxes (three-mesh-bvh's nodes); our 64-byte nodes hold two boxes each.
        nodes: extremes.nodeCount,
        leaves: extremes.leafNodeCount,
        bytes: tree.estimateMemoryInBytes(),
        note: `${tree.nodeCount.toLocaleString('en-US')} 64-byte nodes of two boxes; 48-byte triangle records`,
    };
}

function wrap(tree: CpuMeshBvh, geometry: CompareGeometry): BuiltMesh {
    let current = { positions: geometry.positions, indices: geometry.indices };
    return {
        raycastFirst(ray) {
            const hit = tree.raycastFirst(ray, BVH_DOUBLE_SIDE);
            return hit ? { triangle: hit.triangle, t: hit.distance } : null;
        },
        raycastAll(ray) {
            return tree.raycast(ray, BVH_DOUBLE_SIDE).map((hit) => ({ triangle: hit.triangle, t: hit.distance }));
        },
        intersectsBox(box) {
            return tree.intersectsBox({ min: { x: box[0], y: box[1], z: box[2] }, max: { x: box[3], y: box[4], z: box[5] } });
        },
        intersectsSphere(center, radius) {
            return tree.intersectsSphere({ center: { x: center[0], y: center[1], z: center[2] }, radius });
        },
        closestPoint(point) {
            const hit = tree.closestPointToPoint(point);
            if (!hit) return null;
            return { triangle: hit.faceIndex, distance: hit.distance, point: [hit.point.x, hit.point.y, hit.point.z] };
        },
        refit(positions) {
            current = { positions, indices: geometry.indices };
            tree.refit(current);
        },
        memory() {
            return {
                bytes: tree.estimateMemoryInBytes(),
                method: 'CpuMeshBvh.estimateMemoryInBytes: exact byteLength of the nodes, triangle records (which copy the corners), parents, bounds and the one-instance top level',
            };
        },
        quality: () => oursQuality(tree),
        dispose() {},
    };
}

export const oursAdapter: MeshAdapter = {
    id: 'ours',
    name: '@pmndrs/bvh (tinybvh)',
    library: '@pmndrs/bvh (this repository)',
    license: 'MIT',
    buildMethod: "CpuMeshBvh.build: tinybvh's SAH builder in WebAssembly (leaves of up to 8), written into the GPU layout, on this thread",
    features: features({
        raycastFirst: 'yes',
        raycastAll: 'yes',
        gpuTrace: { status: 'yes', note: 'the WGSL walk over the same layout (TraceBatchKernel, BvhTraceNodes)' },
        intersectsBox: 'yes',
        intersectsSphere: 'yes',
        closestPoint: 'yes',
        refit: { status: 'yes', note: 'CPU refit; GPU refit of BLAS and the top level in SceneTrees' },
        workerBuild: { status: 'yes', note: 'tinybvh worker pool (@pmndrs/bvh/worker), no cross-origin isolation' },
        gpuBuild: { status: 'yes', note: 'GPU H-PLOC and LBVH builders' },
        treeStats: 'yes',
        serialize: 'yes',
    }),
    async prepare() {
        await oursTinyBvh();
    },
    build(geometry) {
        if (!tinyBvh) throw new Error('oursAdapter.prepare() first');
        const tree = CpuMeshBvh.build({ positions: geometry.positions, indices: geometry.indices }, tinyBvh);
        return wrap(tree, geometry);
    },
};

/** Our other builders' trees on the CPU (the GPU builders' twins), for compare/tree-quality and compare/build. */
export const oursBuilders: readonly { readonly id: string; readonly name: string; build(geometry: CompareGeometry): GpuLayoutMeshBvh }[] = [
    { id: 'ours-hploc', name: '@pmndrs/bvh H-PLOC (CPU twin of the GPU builder)', build: (geometry) => buildHplocOnCpu({ positions: geometry.positions, indices: geometry.indices }) },
    { id: 'ours-lbvh', name: '@pmndrs/bvh LBVH (CPU twin of the GPU builder)', build: (geometry) => buildLbvhOnCpu({ positions: geometry.positions, indices: geometry.indices }) },
];

/**
 * The cut of `clusterTraceScene` (src/trace/clustered-scene.ts) against
 * three-mesh-bvh's own `ClusteredBVH` (0.9.18, `src/webgpu/ClusteredBVH.js`,
 * internal to its `BVHComputeData`): the same tree, built by three-mesh-bvh
 * and converted to our layout (../trace-bench/tmbvh-convert.ts, which keeps
 * its node structure and triangle order), must give the same clusters, each
 * the same run of tree positions, for every `primitiveLimit`.
 */

import { BufferAttribute, BufferGeometry, Mesh } from 'three';
import { MeshBVH, SAH } from 'three-mesh-bvh';
import { describe, expect, it, vi } from 'vitest';
import { assembleTraceScene, buildMeshBvhFromTopology, countTriangles, IDENTITY_TRANSFORM, meshClusters } from '../../../../src/trace/index.ts';
import { createRandom, randomTriangleSoup, tessellatedQuad } from '../../../../src/trace/oracle-fixtures.ts';
import { topologyFromMeshBvh } from '../trace-bench/tmbvh-convert.ts';

/** The parts of three-mesh-bvh's internal modules this test reads (they ship no types). */
interface ThreeClusteredBvh {
    readonly primitiveBuffer: Uint32Array;
    readonly primitiveBufferStride: number;
    getBVHRootIndex(compositeNodeId: number): number;
    getBVHNodeIndex(compositeNodeId: number): number;
}
interface ThreeTraversalHelper {
    setBVH(bvh: MeshBVH, root: number): void;
    getRangeStart(node32Index: number): number;
    getRangeEnd(node32Index: number): number;
    reset(): void;
}
// Not exported from the package's entry points; its `src/*` export map reaches them.
const { ClusteredBVH } = (await import('three-mesh-bvh/src/webgpu/ClusteredBVH.js' as string)) as {
    ClusteredBVH: new (root: Mesh[], options: Record<string, unknown>) => ThreeClusteredBvh;
};
const { BVHTraversalHelper } = (await import('three-mesh-bvh/src/core/BVHTraversalHelper.js' as string)) as { BVHTraversalHelper: ThreeTraversalHelper };

/** three-mesh-bvh's clusters of one mesh, as `start:count` runs of tree positions, sorted. */
function threeClusters(bvh: MeshBVH, geometry: BufferGeometry, primitiveLimit: number): string[] {
    // It warns once per leaf at or above the limit (ours cuts the same leaves, silently).
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const clustered = new ClusteredBVH([new Mesh(geometry)], { getBVH: () => bvh, primitiveLimit, strategy: SAH });
    warn.mockRestore();
    const runs: string[] = [];
    for (let at = 0; at < clustered.primitiveBuffer.length; at += clustered.primitiveBufferStride) {
        const composite = clustered.primitiveBuffer[at + 1];
        BVHTraversalHelper.setBVH(bvh, clustered.getBVHRootIndex(composite));
        const node32 = clustered.getBVHNodeIndex(composite);
        const start = BVHTraversalHelper.getRangeStart(node32);
        runs.push(`${start}:${BVHTraversalHelper.getRangeEnd(node32) - start}`);
        BVHTraversalHelper.reset();
    }
    return runs.sort();
}

describe("clusterTraceScene's cut against three-mesh-bvh's ClusteredBVH", () => {
    const meshes = {
        soup: randomTriangleSoup(createRandom(31), 5000, 12, 0.5),
        quad: tessellatedQuad(40, 3),
    };
    for (const [name, mesh] of Object.entries(meshes)) {
        it(`${name}: the same clusters for every primitiveLimit`, () => {
            const geometry = new BufferGeometry();
            geometry.setAttribute('position', new BufferAttribute(mesh.positions, 3));
            if (mesh.indices) geometry.setIndex(new BufferAttribute(mesh.indices, 1));
            // Leaves of at most 8 (our layout's cap), so the conversion keeps three-mesh-bvh's tree as it is.
            const bvh = new MeshBVH(geometry, { strategy: SAH, targetLeafSize: 4, indirect: true });
            const tree = buildMeshBvhFromTopology(mesh, topologyFromMeshBvh(bvh, countTriangles(mesh)));
            const scene = assembleTraceScene([tree], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY_TRANSFORM }], {}, { nodeFormat: 'f32' });
            for (const limit of [1, 3, 8, 16, 64, 300, countTriangles(mesh), countTriangles(mesh) + 1]) {
                const ours = meshClusters(scene, scene.meshes[0].nodeOffset, limit)
                    .map((cluster) => `${cluster.firstTriangle}:${cluster.triangleCount}`)
                    .sort();
                expect(ours, `primitiveLimit ${limit}`).toEqual(threeClusters(bvh, geometry, limit));
            }
        });
    }
});

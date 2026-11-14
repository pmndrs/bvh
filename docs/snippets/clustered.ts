// Code samples for docs/scenes/clustered-top-level.mdx (see getting-started.ts for how they are checked).

// #region clustered-scene
import { buildMeshBvhWithTinyBvh, buildTraceScene, clusterTraceScene, createTinyBvhBuilder, uploadTraceScene, type MeshGeometry, type TraceSceneInstance } from '@pmndrs/bvh';

// #endregion

export async function clusteredScene(device: GPUDevice, meshes: readonly MeshGeometry[], instances: readonly TraceSceneInstance[]) {
    // #region clustered-scene
    const tiny = await createTinyBvhBuilder();
    const scene = await buildTraceScene(meshes, instances, { build: (geometry) => buildMeshBvhWithTinyBvh(geometry, tiny) });

    // Cut every unshared mesh's tree into subtrees of fewer than 64 triangles, each its own top-level leaf.
    const clustered = clusterTraceScene(scene, { primitiveLimit: 64 });
    console.log(clustered.clusters.records, '->', clustered.clusters.clusterRecords, 'top-level records');

    // The same walk, kernels and queries trace it.
    const buffers = uploadTraceScene(device, clustered);
    // #endregion
    return buffers;
}

export function clusteredSplit(scene: Parameters<typeof clusterTraceScene>[0], characterMesh: number) {
    // #region clustered-split
    const clustered = clusterTraceScene(scene, {
        // Instanced meshes too, but not the mesh you refit (its cluster copies would not follow).
        split: (record) => record.meshIndex !== characterMesh,
    });
    // #endregion
    return clustered;
}

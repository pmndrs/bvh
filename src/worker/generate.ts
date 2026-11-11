/**
 * A CPU query tree built off the main thread (three-mesh-bvh's
 * `GenerateMeshBVHWorker.generate`, engine-free): tinybvh builds the layout
 * on the worker pool (`pool.ts`), and the tree object is made on the calling
 * thread around the layout the worker sends back (no copy). Where there is no
 * `Worker` (Node), the pool builds inline on a later turn, so this works the
 * same there.
 */

import type { GpuLayoutMeshBvh } from '../gpu-layout.ts';
import type { MeshInput } from '../mesh-input.ts';
import { CpuMeshBvh, type CpuMeshBvhBuildOptions } from '../query/cpu-mesh-bvh.ts';
import type { BvhBuildRequest, BvhWorkerPool } from './pool.ts';

/** Builds `input`'s layout on `pool`, calling `onProgress` with 0 when asked and 1 when built. */
export async function generateMeshBvhLayout(pool: BvhWorkerPool, input: MeshInput, options: CpuMeshBvhBuildOptions = {}, request: BvhBuildRequest = {}): Promise<GpuLayoutMeshBvh> {
    options.onProgress?.(0);
    const layout = await pool.build(input, options.maxLeafPrimitives !== undefined ? { maxLeafPrimitives: options.maxLeafPrimitives } : {}, request);
    options.onProgress?.(1);
    return layout;
}

/**
 * A {@link CpuMeshBvh} for `input`, built on `pool`. The tree keeps `input`
 * as its source (for `refit()`) unless the build moved its arrays into the
 * worker (`transferInput`); then `refit` needs the geometry passed again.
 */
export async function generateCpuMeshBvh(pool: BvhWorkerPool, input: MeshInput, options: CpuMeshBvhBuildOptions = {}, request: BvhBuildRequest = {}): Promise<CpuMeshBvh> {
    const layout = await generateMeshBvhLayout(pool, input, options, request);
    const detached = input.positions.byteLength === 0 && layout.triangleCount > 0;
    return new CpuMeshBvh(layout, { source: detached ? null : input });
}

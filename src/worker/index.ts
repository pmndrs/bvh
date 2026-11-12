/**
 * tinybvh builds off the main thread: a pool of single-threaded module workers, with no cross-origin isolation needed.
 *
 * @module @pmndrs/bvh/worker
 */

// @pmndrs/bvh/worker: tinybvh builds off the main thread (D3). A pool of
// single-threaded module workers for mesh trees (no cross-origin isolation
// needed), the one-worker mesh builder it grew from, and the top-level shape
// builder. Everything falls back to the calling thread where there is no
// Worker (Node) or one can't start.
export {
    createBvhWorkerPool,
    defaultWorkerPoolSize,
    type BvhBuildRequest,
    type BvhWorkerPool,
    type BvhWorkerPoolOptions,
    type BvhWorkerPoolStats,
} from './pool.ts';
export { generateCpuMeshBvh } from './generate.ts';
export { createMeshBvhBuilder, type MeshBvhBuilder, type MeshBvhBuilderConfig } from '../trace/mesh-bvh-async.ts';
export { createTopLevelBuilder, type TopLevelBuilder, type TopLevelBuilderConfig, type TopLevelResult } from '../trace/top-level-async.ts';
export type { MeshBvh, MeshBvhBuildOptions, MeshGeometry } from '../trace/mesh-bvh.ts';

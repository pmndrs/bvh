/**
 * Builds mesh BVHs in a module worker when the environment has one, and on the
 * calling thread otherwise (Node tests, or a host that blocks workers).
 *
 * It is a pool of one worker (`@pmndrs/bvh/worker`'s `createBvhWorkerPool`),
 * so it shares the pool's rules: no build is sent before the worker said it
 * loaded, `'auto'` builds inline when it can't load, views into a shared
 * buffer are copied rather than detached, and a worker whose WASM memory grew
 * past 256 MiB is replaced.
 */

import type { MeshBvh, MeshBvhBuildOptions, MeshGeometry } from './mesh-bvh.ts';
import { createBvhWorkerPool } from '../worker/pool.ts';

export interface MeshBvhBuilderConfig {
    /**
     * `'auto'` (default) uses a worker when `Worker` exists, and builds on the
     * calling thread if it can't start or load (no build is lost). `'never'`
     * always builds inline; `'always'` throws when there is no `Worker` and
     * rejects the builds when the worker can't start or load.
     */
    readonly worker?: 'auto' | 'never' | 'always';
    /**
     * Move the input buffers into the worker instead of copying them. The
     * caller's arrays become unusable (empty) afterwards. Off by default.
     */
    readonly transferInput?: boolean;
}

export interface MeshBvhBuilder {
    /** Whether builds run in a worker (false once `'auto'` fell back to the calling thread). */
    readonly usesWorker: boolean;
    build(geometry: MeshGeometry, options?: MeshBvhBuildOptions): Promise<MeshBvh>;
    /** Stops the worker. Pending builds reject. */
    dispose(): void;
}

export function createMeshBvhBuilder(config: MeshBvhBuilderConfig = {}): MeshBvhBuilder {
    const pool = createBvhWorkerPool({ size: 1, worker: config.worker, transferInput: config.transferInput ?? false });
    return {
        get usesWorker() {
            return pool.usesWorkers;
        },
        build: (geometry, options = {}) => pool.build(geometry, options),
        dispose: () => pool.dispose(),
    };
}

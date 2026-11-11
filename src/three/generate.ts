/**
 * three-mesh-bvh's `GenerateMeshBVHWorker` over `@pmndrs/bvh/worker`'s pool:
 * `generate(geometry, options)` resolves to a {@link BvhBoundsTree} built by
 * tinybvh in a worker. Unlike three-mesh-bvh's, the geometry's arrays are
 * copied, not moved, so the geometry stays usable during the build, several
 * builds may run at once (one per pool worker), and Node builds inline.
 *
 * ```ts
 * const generator = new GenerateMeshBvhWorker();
 * geometry.boundsTree = await generator.generate(geometry, { strategy: SAH, maxLeafTris: 1, onProgress });
 * ```
 */

import type { BufferGeometry } from 'three';
import { createBvhWorkerPool, type BvhWorkerPool, type BvhWorkerPoolOptions } from '../worker/pool.ts';
import { generateMeshBvhLayout } from '../worker/generate.ts';
import { BvhBoundsTree, buildOptionsFromThreeMeshBvh, snapshotGeometry, type BvhBoundsTreeOptions } from './raycast.ts';

export interface GenerateMeshBvhWorkerOptions extends BvhWorkerPoolOptions {
    /** A pool to build on, shared with other users; it is not disposed with this object. Default: a pool of its own. */
    readonly pool?: BvhWorkerPool;
}

export class GenerateMeshBvhWorker {
    readonly pool: BvhWorkerPool;
    private readonly ownsPool: boolean;
    private pending = 0;
    /** Aborted by {@link dispose}: rejects this object's unfinished builds, also on a shared pool. */
    private readonly controller = new AbortController();

    constructor(options: GenerateMeshBvhWorkerOptions = {}) {
        this.ownsPool = !options.pool;
        this.pool = options.pool ?? createBvhWorkerPool({ ...options, transferInput: false });
    }

    /** True while a `generate` is unfinished. */
    get running(): boolean {
        return this.pending > 0;
    }

    /**
     * Builds `geometry`'s tree in a worker. Takes {@link computeBoundsTree}'s
     * options (three-mesh-bvh's), checked before the build starts; `onProgress`
     * gets 0 when the build is asked for and 1 when it is back. Doesn't set
     * `geometry.boundsTree`.
     *
     * The tree is built from the positions as they are when `generate` is
     * called: edits made while it builds leave it stale (raycasts use three's
     * own until `boundsTree.refit()`).
     */
    async generate(geometry: BufferGeometry, options: BvhBoundsTreeOptions = {}): Promise<BvhBoundsTree> {
        const buildOptions = buildOptionsFromThreeMeshBvh(options);
        // The versions now, before the copy is posted: an edit during the build must count as one.
        const snapshot = snapshotGeometry(geometry);
        this.pending++;
        try {
            const layout = await generateMeshBvhLayout(this.pool, snapshot.input, buildOptions, { transferInput: false, signal: this.controller.signal });
            return new BvhBoundsTree(geometry, { ...options, onProgress: undefined, layout, snapshot });
        } finally {
            this.pending--;
        }
    }

    /**
     * Stops the pool's workers (if the pool is this object's own). This
     * object's unfinished builds reject, on a shared pool too (a build already
     * running on a shared pool finishes there and its result is dropped).
     */
    dispose(): void {
        if (this.ownsPool) this.pool.dispose();
        this.controller.abort();
    }
}

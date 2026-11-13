/**
 * A `/worker` pool wrapped so a page can see each tinybvh build: when it was
 * asked for and when the pool answered (queued, then building, in one span).
 * The pool is the app's own (`createBvhWorkerPool`), handed to `SceneTrees`
 * as `workers`; `SceneTrees` sees an ordinary `BvhWorkerPool`.
 */

import type { MeshGeometry } from '@pmndrs/bvh';
import type { BvhWorkerPool } from '@pmndrs/bvh/worker';

/** `pool`, calling `onAnswered(geometry, askedAt)` when a build succeeds (`askedAt`: `performance.now()` of the call). */
export function watchPool(pool: BvhWorkerPool, onAnswered: (geometry: MeshGeometry, askedAt: number) => void): BvhWorkerPool {
    return {
        get usesWorkers() {
            return pool.usesWorkers;
        },
        get size() {
            return pool.size;
        },
        get stats() {
            return pool.stats;
        },
        build(geometry, options, request) {
            const asked = performance.now();
            const built = pool.build(geometry, options, request);
            built.then(
                () => onAnswered(geometry, asked),
                () => undefined, // the caller sees the rejection
            );
            return built;
        },
        dispose: () => pool.dispose(),
    };
}

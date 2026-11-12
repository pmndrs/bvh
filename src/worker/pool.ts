/**
 * A pool of module workers that build mesh trees with tinybvh (WASM), each
 * mesh on one worker, several meshes at once (`@pmndrs/bvh/worker`, D3).
 *
 * Each worker is single-threaded (tinybvh's WASM build has no threads), so the
 * pool needs no SharedArrayBuffer and no cross-origin isolation, unlike
 * three-mesh-bvh's `ParallelMeshBVHWorker`. Parallelism is across meshes: a
 * scene of many meshes spreads over the pool, while one big mesh (a merged
 * static group, say: 1.5 s on Sponza) occupies one worker and leaves the rest
 * to the others. Results come back as transferable `ArrayBuffer`s, so nothing
 * is copied on the way out; inputs are copied unless `transferInput` is set.
 *
 * Where there is no `Worker` (Node, a host that blocks workers) or one can't
 * start, the pool builds on the calling thread, one mesh at a time on later
 * turns, so callers see the same asynchronous behaviour (`worker: 'auto'`, the
 * default; `'always'` throws instead, `'never'` always builds inline).
 *
 * Builds are started in the order they were asked for (first in, first out).
 * A caller that wants the makespan short asks for its biggest meshes first.
 *
 * A WASM memory can grow but never shrink, and tinybvh's grows to about 230
 * bytes per triangle for the build in flight. So a worker whose build left its
 * memory past `recycleAboveBytes` (256 MiB by default: a mesh of about a
 * million triangles) is ended once it has answered, and the next build starts
 * a fresh one; the operating system gets the memory back at once. Big builds
 * belong in this pool for that reason: on the main thread the memory stays
 * until the garbage collector frees the builder's dropped instance.
 */

import { buildMeshBvhAsync } from '../trace/mesh-bvh-build.ts';
import type { MeshBvh, MeshBvhBuildOptions, MeshGeometry } from '../trace/mesh-bvh.ts';
import type { BuildRequest, BuildResponse } from '../trace/mesh-bvh-worker.ts';

export interface BvhWorkerPoolOptions {
    /**
     * Workers in the pool. Default {@link defaultWorkerPoolSize}: one less than
     * the logical cores, at least 1 and at most 4. Workers start when builds
     * need them, not before, so an idle pool costs nothing.
     */
    readonly size?: number;
    /**
     * `'auto'` (default) uses workers when `Worker` exists, and builds inline if
     * none can be started or the first one fails before answering. `'never'`
     * always builds inline; `'always'` throws instead of falling back.
     */
    readonly worker?: 'auto' | 'never' | 'always';
    /**
     * Move every build's input arrays into the worker instead of copying them;
     * the caller's arrays are empty afterwards. An array that views only part of
     * its buffer (glTF accessors into one BIN buffer) is copied instead, so the
     * buffer's other arrays stay usable. Default false. A build can
     * choose for itself with {@link BvhBuildRequest.transferInput}.
     */
    readonly transferInput?: boolean;
    /**
     * A worker whose last build left its tinybvh WASM memory larger than this
     * many bytes is ended after answering, and a fresh worker takes the next
     * build: the memory goes back to the system at once. Default
     * {@link WORKER_RECYCLE_ABOVE_BYTES} (256 MiB); `Infinity` keeps workers
     * for the pool's life. Inline builds (no `Worker`) release their WASM
     * instance the same way (`createTinyBvhBuilder`'s `releaseAboveBytes`).
     */
    readonly recycleAboveBytes?: number;
}

/** {@link BvhWorkerPoolOptions.recycleAboveBytes}'s default: 256 MiB. */
export const WORKER_RECYCLE_ABOVE_BYTES = 256 * 2 ** 20;

/** Per-build options of {@link BvhWorkerPool.build}, besides the tree's own. */
export interface BvhBuildRequest {
    /**
     * Cancels the build. A build still waiting for a worker never starts; one
     * already running finishes in its worker (a tinybvh build can't be
     * interrupted) and its result is dropped. Either way the promise rejects
     * with an `AbortError` at once.
     */
    readonly signal?: AbortSignal;
    /** Overrides the pool's `transferInput` for this build. */
    readonly transferInput?: boolean;
}

/** Counts for tools and tests. */
export interface BvhWorkerPoolStats {
    /** Builds waiting for a worker. */
    readonly queued: number;
    /** Builds running now. */
    readonly running: number;
    /** Workers started and alive. */
    readonly workers: number;
    /** Builds finished (built, failed or cancelled while running) since the pool was made. */
    readonly finished: number;
    /** Workers ended because a build left their WASM memory past `recycleAboveBytes`. */
    readonly recycled: number;
}

export interface BvhWorkerPool {
    /** Whether builds run in workers (false: inline, on the calling thread). */
    readonly usesWorkers: boolean;
    /** The most builds that run at once: the pool's size, or 1 inline. */
    readonly size: number;
    readonly stats: BvhWorkerPoolStats;
    /** Builds one mesh's tree with tinybvh on the next free worker. */
    build(geometry: MeshGeometry, options?: MeshBvhBuildOptions, request?: BvhBuildRequest): Promise<MeshBvh>;
    /** Stops every worker. Builds still queued or running reject. */
    dispose(): void;
}

/** One less than the logical cores (the main thread keeps one), between 1 and 4. */
export function defaultWorkerPoolSize(): number {
    const cores = typeof navigator !== 'undefined' && navigator.hardwareConcurrency > 0 ? navigator.hardwareConcurrency : 2;
    return Math.max(1, Math.min(4, cores - 1));
}

interface Job {
    readonly geometry: MeshGeometry;
    readonly options: MeshBvhBuildOptions;
    readonly transferInput: boolean;
    readonly signal?: AbortSignal;
    readonly resolve: (bvh: MeshBvh) => void;
    readonly reject: (error: Error) => void;
    /** Set once the promise is settled (a cancelled build settles before its worker is done). */
    settled: boolean;
    /** The input was moved into a worker: it can't be built again elsewhere. */
    transferred: boolean;
    onAbort?: () => void;
}

interface PoolWorker {
    readonly worker: Worker;
    job: Job | null;
    nextId: number;
}

function abortError(): Error {
    return typeof DOMException !== 'undefined' ? new DOMException('the BVH build was cancelled', 'AbortError') : Object.assign(new Error('the BVH build was cancelled'), { name: 'AbortError' });
}

function toError(error: unknown): Error {
    return error instanceof Error ? error : new Error(String(error));
}

/** The worker entry's URL. It stays inline in `new Worker(new URL(...))` so bundlers find the file. */
function startWorker(): Worker {
    return new Worker(new URL('../trace/mesh-bvh-worker.ts', import.meta.url), { type: 'module' });
}

/** Whether a buffer was transferred away. Without ES2024's `detached`, any empty buffer counts (there is nothing to move). */
function isDetached(buffer: ArrayBuffer): boolean {
    return (buffer as { detached?: boolean }).detached ?? buffer.byteLength === 0;
}

/**
 * The geometry to post and the buffers to transfer with it. A view that covers only part of its
 * buffer (glTF meshes viewing one shared BIN buffer) is copied into a compact array first, which is
 * transferred: cloning the view would copy the whole buffer, and transferring it would empty every
 * other view into it. With `transferInput`, a view that covers its whole `ArrayBuffer` moves into the
 * worker. Each buffer is listed once, and a detached or shared one never.
 */
function postable(geometry: MeshGeometry, transferInput: boolean): { geometry: MeshGeometry; transfer: Transferable[]; moved: boolean } {
    const transfer = new Set<ArrayBuffer>();
    let moved = false;
    const prepare = <T extends Float32Array | Uint16Array | Uint32Array>(view: T): T => {
        const buffer = view.buffer;
        if (!(buffer instanceof ArrayBuffer) || isDetached(buffer)) return view;
        if (view.byteOffset !== 0 || view.byteLength !== buffer.byteLength) {
            const copy = view.slice() as T;
            transfer.add(copy.buffer as ArrayBuffer);
            return copy;
        }
        if (transferInput) {
            transfer.add(buffer);
            moved = true;
        }
        return view;
    };
    const posted: MeshGeometry = {
        positions: prepare(geometry.positions),
        indices: geometry.indices ? prepare(geometry.indices) : geometry.indices,
        triangleFlags: geometry.triangleFlags ? prepare(geometry.triangleFlags) : geometry.triangleFlags,
    };
    return { geometry: posted, transfer: [...transfer], moved };
}

export function createBvhWorkerPool(options: BvhWorkerPoolOptions = {}): BvhWorkerPool {
    const mode = options.worker ?? 'auto';
    const size = options.size ?? defaultWorkerPoolSize();
    if (!Number.isInteger(size) || size < 1) throw new RangeError(`a worker pool needs a size of at least 1, got ${size}`);
    const recycleAboveBytes = options.recycleAboveBytes ?? WORKER_RECYCLE_ABOVE_BYTES;
    if (!(recycleAboveBytes >= 0)) throw new RangeError(`a worker pool's recycleAboveBytes must be a number of bytes (or Infinity), got ${recycleAboveBytes}`);
    let recycled = 0;
    let inline = mode === 'never' || typeof Worker === 'undefined';
    if (inline && mode === 'always') throw new Error('@pmndrs/bvh/worker: a worker was required, but this environment has no Worker');

    const queue: Job[] = [];
    const workers: PoolWorker[] = [];
    let inlineRunning = 0;
    /** Builds running inline now (a tinybvh build can't be stopped, so dispose settles them and drops their result). */
    const inlineJobs = new Set<Job>();
    let finished = 0;
    /** Whether any worker of this pool ever answered, even one since recycled: until then a failure may mean workers can't load. */
    let everAnswered = false;
    let disposed = false;

    const settle = (job: Job, outcome: { bvh: MeshBvh } | { error: Error }): void => {
        if (job.settled) return;
        job.settled = true;
        if (job.onAbort) job.signal?.removeEventListener('abort', job.onAbort);
        if ('bvh' in outcome) job.resolve(outcome.bvh);
        else job.reject(outcome.error);
    };

    const runInline = (job: Job): void => {
        inlineRunning++;
        inlineJobs.add(job);
        // A later turn, so an inline pool is as asynchronous as a worker one.
        setTimeout(() => {
            if (job.settled) {
                inlineJobs.delete(job);
                inlineRunning--;
                finished++;
                pump();
                return;
            }
            buildMeshBvhAsync(job.geometry, job.options).then(
                (bvh) => settle(job, { bvh }),
                (error: unknown) => settle(job, { error: toError(error) }),
            ).finally(() => {
                inlineJobs.delete(job);
                inlineRunning--;
                finished++;
                pump();
            });
        }, 0);
    };

    const fallBackToInline = (reason: string): void => {
        if (mode === 'always') {
            // No fallback: every build the pool holds fails, the ones on a worker included, and the
            // workers stop. A later build starts a fresh worker.
            for (const entry of workers.splice(0)) {
                entry.worker.terminate();
                if (entry.job) {
                    finished++;
                    settle(entry.job, { error: new Error(reason) });
                }
                entry.job = null;
            }
            for (const job of queue.splice(0)) settle(job, { error: new Error(reason) });
            return;
        }
        inline = true;
        for (const entry of workers.splice(0)) {
            entry.worker.terminate();
            const job = entry.job;
            if (!job) continue;
            if (job.transferred) settle(job, { error: new Error(`${reason}; this build's input was moved into the worker, so it can't be built again inline`) });
            else queue.unshift(job);
        }
    };

    const attach = (entry: PoolWorker): void => {
        entry.worker.onmessage = (event: MessageEvent<BuildResponse>) => {
            const job = entry.job;
            if (!job || event.data.id !== entry.nextId - 1) return;
            entry.job = null;
            everAnswered = true;
            finished++;
            const response = event.data;
            settle(job, 'error' in response ? { error: new Error(response.error) } : { bvh: response.bvh });
            if ((response.heapBytes ?? 0) > recycleAboveBytes) {
                // Its WASM memory can't shrink: end the worker, and let pump start a fresh one if it is needed.
                const index = workers.indexOf(entry);
                if (index >= 0) workers.splice(index, 1);
                entry.worker.terminate();
                recycled++;
            }
            pump();
        };
        entry.worker.onerror = (event) => {
            event.preventDefault?.();
            const message = `@pmndrs/bvh/worker: a BVH worker failed: ${event.message || 'it could not be loaded'}`;
            if (!everAnswered) {
                // No worker of this pool has ever answered (a recycled one counts): workers probably can't load here (a bundler that didn't copy
                // the worker file, a blocked origin). Build inline from now on.
                fallBackToInline(message);
                pump();
                return;
            }
            // One worker died: its build fails, and the pool starts a new worker for the next one.
            const index = workers.indexOf(entry);
            if (index >= 0) workers.splice(index, 1);
            entry.worker.terminate();
            if (entry.job) {
                finished++;
                settle(entry.job, { error: new Error(message) });
            }
            entry.job = null;
            pump();
        };
    };

    const send = (entry: PoolWorker, job: Job): void => {
        entry.job = job;
        let moved: boolean;
        try {
            const posted = postable(job.geometry, job.transferInput);
            moved = posted.moved;
            const request: BuildRequest = { id: entry.nextId++, geometry: posted.geometry, options: job.options };
            entry.worker.postMessage(request, posted.transfer);
        } catch (error) {
            // A detached input, a buffer that can't be cloned: this build fails, the worker stays free.
            entry.job = null;
            finished++;
            settle(job, { error: new Error(`@pmndrs/bvh/worker: the build's input could not be sent to a worker: ${toError(error).message}`) });
            return;
        }
        job.transferred = moved;
    };

    /** Starts queued builds on free workers (starting workers up to the pool's size). */
    function pump(): void {
        if (disposed) return;
        while (queue.length > 0) {
            if (queue[0].settled) {
                queue.shift();
                continue;
            }
            if (inline) {
                // One inline build at a time: they share the calling thread.
                if (inlineRunning > 0) return;
                runInline(queue.shift()!);
                continue;
            }
            let entry = workers.find((candidate) => candidate.job === null);
            if (!entry && workers.length < size) {
                try {
                    entry = { worker: startWorker(), job: null, nextId: 1 };
                } catch (error) {
                    fallBackToInline(`@pmndrs/bvh/worker: a worker could not be started: ${toError(error).message}`);
                    continue;
                }
                attach(entry);
                workers.push(entry);
            }
            if (!entry) return;
            send(entry, queue.shift()!);
        }
    }

    return {
        get usesWorkers() {
            return !inline;
        },
        get size() {
            return inline ? 1 : size;
        },
        get stats(): BvhWorkerPoolStats {
            return {
                queued: queue.filter((job) => !job.settled).length,
                running: inline ? inlineRunning : workers.filter((entry) => entry.job !== null).length,
                workers: workers.length,
                finished,
                recycled,
            };
        },
        build(geometry, buildOptions = {}, request = {}) {
            if (disposed) return Promise.reject(new Error('@pmndrs/bvh/worker: the pool was disposed'));
            if (request.signal?.aborted) return Promise.reject(abortError());
            return new Promise<MeshBvh>((resolve, reject) => {
                const job: Job = {
                    geometry,
                    options: buildOptions,
                    transferInput: request.transferInput ?? options.transferInput ?? false,
                    signal: request.signal,
                    resolve,
                    reject,
                    settled: false,
                    transferred: false,
                };
                if (request.signal) {
                    job.onAbort = () => settle(job, { error: abortError() });
                    request.signal.addEventListener('abort', job.onAbort, { once: true });
                }
                queue.push(job);
                pump();
            });
        },
        dispose() {
            if (disposed) return;
            disposed = true;
            const error = new Error('@pmndrs/bvh/worker: the pool was disposed');
            for (const entry of workers.splice(0)) {
                entry.worker.terminate();
                if (entry.job) settle(entry.job, { error });
            }
            for (const job of inlineJobs) settle(job, { error });
            for (const job of queue.splice(0)) settle(job, { error });
        },
    };
}

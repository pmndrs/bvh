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
 * start or load, the pool builds on the calling thread, one mesh at a time on
 * later turns, so callers see the same asynchronous behaviour (`worker: 'auto'`,
 * the default; `'always'` rejects instead, `'never'` always builds inline).
 * `inlineReason` says why.
 *
 * A worker gets no build until it has said it loaded (the worker entry posts
 * `{ ready: true }` once its module ran). So a worker that can't load (a
 * bundler that didn't ship the file, a dev server that refuses it) holds no
 * input, moved or copied, and `'auto'` rebuilds everything inline, merged
 * groups' moved geometry included. It costs one message per worker start.
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

import { buildMeshBvhAsync, releaseInlineTinyBvh } from '../trace/mesh-bvh-build.ts';
import { countTriangles } from '../trace/triangle-packing.ts';
import type { MeshBvh, MeshBvhBuildOptions, MeshGeometry } from '../trace/mesh-bvh.ts';
import type { BuildRequest, BuildResponse, WorkerReady } from '../trace/mesh-bvh-worker.ts';

export interface BvhWorkerPoolOptions {
    /**
     * Workers in the pool. Default {@link defaultWorkerPoolSize}: one less than
     * the logical cores, at least 1 and at most 4. Workers start when builds
     * need them, not before, so an idle pool costs nothing.
     */
    readonly size?: number;
    /**
     * `'auto'` (default) uses workers when `Worker` exists, and builds inline
     * from then on if one can't be started or fails to load (it errs before
     * saying it loaded; no build was sent to it, so none is lost). `'never'`
     * always builds inline; `'always'` throws when there is no `Worker` and
     * rejects the builds when a worker can't start or load.
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
    /**
     * A worker left with nothing to build for this many milliseconds is
     * ended, and its tinybvh WASM memory (64 MiB at least, more after a big
     * build) goes back to the system; the next build starts a fresh worker
     * (a module load and a WASM start, a few milliseconds). Inline, the
     * calling thread's tinybvh instance is dropped the same way. Default
     * `Infinity` ({@link WORKER_IDLE_TIMEOUT_MS}): idle workers live as long
     * as the pool. The P9 lever "freeing CPU copies"
     * (docs/internals/design/memory.md): a pool of 4 holds 256 MiB while idle.
     */
    readonly idleTimeoutMs?: number;
    /**
     * Called for every build as it is queued, started on a worker (or
     * inline) and finished: built, failed or cancelled (see
     * {@link BvhPoolBuildEvent}). For tools that time builds per mesh. A
     * listener that throws doesn't break the pool (the error is rethrown
     * on a later turn).
     */
    readonly onBuild?: (event: BvhPoolBuildEvent) => void;
}

/** One build's progress through a pool ({@link BvhWorkerPoolOptions.onBuild}). */
export interface BvhPoolBuildEvent {
    /** `'queued'` when asked for, `'started'` when a worker (or the calling thread) takes it, `'finished'` when its promise settles. */
    readonly phase: 'queued' | 'started' | 'finished';
    /** The build's number in this pool, from 1: the same in its events. */
    readonly id: number;
    /** The geometry passed to {@link BvhWorkerPool.build} (the same object, to tell meshes apart; its arrays may have moved into a worker). */
    readonly geometry: MeshGeometry;
    readonly triangleCount: number;
    /** Started on a worker (true) or inline on the calling thread (false); for `'queued'`, whether the pool uses workers now; for a build cancelled before it started, false. */
    readonly inWorker: boolean;
    /** Milliseconds since the build was queued (0 for `'queued'`). */
    readonly ms: number;
    /** `'finished'` only: a tree came back, the build failed, or it was cancelled (its signal, or the pool's `dispose`). */
    readonly outcome?: 'built' | 'failed' | 'cancelled';
    /** `'finished'` with `'failed'` or `'cancelled'`: why. */
    readonly error?: string;
}

/** {@link BvhWorkerPoolOptions.recycleAboveBytes}'s default: 256 MiB. */
export const WORKER_RECYCLE_ABOVE_BYTES = 256 * 2 ** 20;

/** {@link BvhWorkerPoolOptions.idleTimeoutMs}'s default: `Infinity`, idle workers are kept. */
export const WORKER_IDLE_TIMEOUT_MS = Infinity;

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
    /** Called once when a worker (or the calling thread) takes this build, after it waited in the queue. */
    readonly onStart?: () => void;
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
    /** Workers ended (or, inline, tinybvh instances dropped) because they sat idle for `idleTimeoutMs`. */
    readonly idleEnded: number;
}

export interface BvhWorkerPool {
    /** Whether builds run in workers (false: inline, on the calling thread). */
    readonly usesWorkers: boolean;
    /**
     * Why builds run inline: `worker: 'never'`, no `Worker` here, or the error
     * of the worker that couldn't start or load (`'auto'` then builds inline).
     * Null while builds run in workers.
     */
    readonly inlineReason: string | null;
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
    readonly id: number;
    readonly queuedAt: number;
    readonly triangleCount: number;
    readonly onStart?: () => void;
    /** Where it started; false until it does. */
    inWorker: boolean;
    started: boolean;
    readonly geometry: MeshGeometry;
    readonly options: MeshBvhBuildOptions;
    readonly transferInput: boolean;
    readonly signal?: AbortSignal;
    readonly resolve: (bvh: MeshBvh) => void;
    readonly reject: (error: Error) => void;
    /** Set once the promise is settled (a cancelled build settles before its worker is done). */
    settled: boolean;
    onAbort?: () => void;
}

interface PoolWorker {
    readonly worker: Worker;
    job: Job | null;
    nextId: number;
    /** The worker said it loaded ({@link WorkerReady}); until then it gets no build. */
    ready: boolean;
    /** The timer that ends it after `idleTimeoutMs` with nothing to build, or null. */
    idleTimer: ReturnType<typeof setTimeout> | null;
}

function abortError(): Error {
    return typeof DOMException !== 'undefined' ? new DOMException('the BVH build was cancelled', 'AbortError') : Object.assign(new Error('the BVH build was cancelled'), { name: 'AbortError' });
}

function toError(error: unknown): Error {
    return error instanceof Error ? error : new Error(String(error));
}

/**
 * The message for a worker that erred before it said it loaded. `detail` is
 * the error event's message, which browsers leave empty for a script that
 * failed to load.
 */
export function workerLoadFailure(what: string, detail?: string): string {
    return (
        `${what} could not be loaded${detail ? ` (${detail})` : ''}. ` +
        'A bundler may not have shipped the worker file, or a dev server refused it. Under Vite: keep @pmndrs/bvh out of ' +
        'optimizeDeps (optimizeDeps.exclude), and for a linked checkout outside the project add its folder to server.fs.allow'
    );
}

/** A timer for `idleTimeoutMs` that doesn't keep Node's process alive on its own (browsers have no `unref`). */
function idleTimer(callback: () => void, ms: number): ReturnType<typeof setTimeout> {
    const timer = setTimeout(callback, ms);
    (timer as { unref?: () => void }).unref?.();
    return timer;
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
function postable(geometry: MeshGeometry, transferInput: boolean): { geometry: MeshGeometry; transfer: Transferable[] } {
    const transfer = new Set<ArrayBuffer>();
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
        }
        return view;
    };
    const posted: MeshGeometry = {
        positions: prepare(geometry.positions),
        indices: geometry.indices ? prepare(geometry.indices) : geometry.indices,
        triangleFlags: geometry.triangleFlags ? prepare(geometry.triangleFlags) : geometry.triangleFlags,
        // Copied, never moved: a scene still reads them (an indexed scene's material table).
        ...(geometry.triangleMaterials ? { triangleMaterials: geometry.triangleMaterials } : {}),
    };
    return { geometry: posted, transfer: [...transfer] };
}

export function createBvhWorkerPool(options: BvhWorkerPoolOptions = {}): BvhWorkerPool {
    const mode = options.worker ?? 'auto';
    const size = options.size ?? defaultWorkerPoolSize();
    if (!Number.isInteger(size) || size < 1) throw new RangeError(`a worker pool needs a size of at least 1, got ${size}`);
    const recycleAboveBytes = options.recycleAboveBytes ?? WORKER_RECYCLE_ABOVE_BYTES;
    if (!(recycleAboveBytes >= 0)) throw new RangeError(`a worker pool's recycleAboveBytes must be a number of bytes (or Infinity), got ${recycleAboveBytes}`);
    const idleTimeoutMs = options.idleTimeoutMs ?? WORKER_IDLE_TIMEOUT_MS;
    if (!(idleTimeoutMs >= 0)) throw new RangeError(`a worker pool's idleTimeoutMs must be a number of milliseconds (or Infinity), got ${idleTimeoutMs}`);
    let recycled = 0;
    let idleEnded = 0;
    /** Inline: the timer that drops the calling thread's tinybvh instance once nothing was built for `idleTimeoutMs`. */
    let inlineIdleTimer: ReturnType<typeof setTimeout> | null = null;
    let inline = mode === 'never' || typeof Worker === 'undefined';
    if (inline && mode === 'always') throw new Error('@pmndrs/bvh/worker: a worker was required, but this environment has no Worker');
    let inlineReason: string | null = mode === 'never' ? "worker: 'never'" : inline ? 'this environment has no Worker' : null;

    const queue: Job[] = [];
    const workers: PoolWorker[] = [];
    let inlineRunning = 0;
    /** Inline: a build ran since the calling thread's tinybvh instance was last dropped (`idleTimeoutMs`). */
    let inlineBuilt = false;
    /** Builds running inline now (a tinybvh build can't be stopped, so dispose settles them and drops their result). */
    const inlineJobs = new Set<Job>();
    let finished = 0;
    let disposed = false;
    let nextId = 1;

    /** Calls the pool's listener; one that throws is reported on a later turn, never into the pool's own flow. */
    const notify = (job: Job, phase: BvhPoolBuildEvent['phase'], extra: Partial<BvhPoolBuildEvent> = {}): void => {
        if (!options.onBuild) return;
        const event: BvhPoolBuildEvent = { phase, id: job.id, geometry: job.geometry, triangleCount: job.triangleCount, inWorker: job.inWorker, ms: phase === 'queued' ? 0 : performance.now() - job.queuedAt, ...extra };
        try {
            options.onBuild(event);
        } catch (error) {
            setTimeout(() => {
                throw error;
            }, 0);
        }
    };

    /** A worker or the calling thread takes `job`. */
    const markStarted = (job: Job, inWorker: boolean): void => {
        job.started = true;
        job.inWorker = inWorker;
        notify(job, 'started');
        try {
            job.onStart?.();
        } catch (error) {
            setTimeout(() => {
                throw error;
            }, 0);
        }
    };

    const settle = (job: Job, outcome: { bvh: MeshBvh } | { error: Error }): void => {
        if (job.settled) return;
        job.settled = true;
        if (job.onAbort) job.signal?.removeEventListener('abort', job.onAbort);
        if ('bvh' in outcome) {
            notify(job, 'finished', { outcome: 'built' });
            job.resolve(outcome.bvh);
        } else {
            const cancelled = outcome.error.name === 'AbortError' || disposed;
            notify(job, 'finished', { outcome: cancelled ? 'cancelled' : 'failed', error: outcome.error.message });
            job.reject(outcome.error);
        }
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
            markStarted(job, false);
            inlineBuilt = true;
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

    const removeWorker = (entry: PoolWorker): void => {
        const index = workers.indexOf(entry);
        if (index >= 0) workers.splice(index, 1);
        if (entry.idleTimer !== null) clearTimeout(entry.idleTimer);
        entry.idleTimer = null;
        entry.worker.terminate();
    };

    /**
     * `idleTimeoutMs`: ends idle loaded workers after the timeout, and (inline) drops the calling thread's
     * tinybvh instance once no build ran or waited for that long. Called after every change of who is busy.
     */
    const watchIdle = (): void => {
        if (idleTimeoutMs === Infinity || disposed) return;
        for (const entry of workers) {
            const idle = entry.ready && entry.job === null;
            if (!idle && entry.idleTimer !== null) {
                clearTimeout(entry.idleTimer);
                entry.idleTimer = null;
            } else if (idle && entry.idleTimer === null) {
                entry.idleTimer = idleTimer(() => {
                    entry.idleTimer = null;
                    if (disposed || entry.job !== null || !workers.includes(entry)) return;
                    removeWorker(entry);
                    idleEnded++;
                }, idleTimeoutMs);
            }
        }
        const inlineIdle = inline && inlineRunning === 0 && !queue.some((job) => !job.settled);
        if (!inlineIdle && inlineIdleTimer !== null) {
            clearTimeout(inlineIdleTimer);
            inlineIdleTimer = null;
        } else if (inlineIdle && inlineIdleTimer === null && inlineBuilt) {
            inlineIdleTimer = idleTimer(() => {
                inlineIdleTimer = null;
                if (disposed || inlineRunning > 0 || queue.some((job) => !job.settled)) return;
                // Only when this pool built something inline since the last drop: another pool's instance is the same one.
                if (releaseInlineTinyBvh()) idleEnded++;
                inlineBuilt = false;
            }, idleTimeoutMs);
        }
    };

    /** A worker couldn't start or load. No build was sent to it (builds wait for {@link WorkerReady}), so none is lost. */
    const workerUnavailable = (reason: string): void => {
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
        // 'auto': inline from now on. Idle workers stop now; a loaded worker still building finishes
        // its build (its input may have been moved there) and stops when it answers.
        inline = true;
        inlineReason = reason;
        for (const entry of [...workers]) if (entry.job === null) removeWorker(entry);
    };

    const attach = (entry: PoolWorker): void => {
        entry.worker.onmessage = (event: MessageEvent<BuildResponse | WorkerReady>) => {
            if ('ready' in event.data) {
                entry.ready = true;
                pump();
                return;
            }
            const job = entry.job;
            if (!job || event.data.id !== entry.nextId - 1) return;
            entry.job = null;
            finished++;
            const response = event.data;
            settle(job, 'error' in response ? { error: new Error(response.error) } : { bvh: response.bvh });
            if (inline) {
                // The pool went inline while this worker was building.
                removeWorker(entry);
            } else if ((response.heapBytes ?? 0) > recycleAboveBytes) {
                // Its WASM memory can't shrink: end the worker, and let pump start a fresh one if it is needed.
                removeWorker(entry);
                recycled++;
            }
            pump();
        };
        entry.worker.onerror = (event) => {
            event.preventDefault?.();
            if (!entry.ready) {
                // It never loaded (a bundler that didn't copy the worker file, a dev server that refused it,
                // a blocked origin), and holds no build: workers can't load here.
                workerUnavailable(`@pmndrs/bvh/worker: ${workerLoadFailure('a BVH worker', event.message)}`);
                pump();
                return;
            }
            // A loaded worker died: its build fails, and the pool starts a new worker for the next one.
            removeWorker(entry);
            if (entry.job) {
                finished++;
                settle(entry.job, { error: new Error(`@pmndrs/bvh/worker: a BVH worker failed: ${event.message || 'it stopped while building'}`) });
            }
            entry.job = null;
            pump();
        };
    };

    const send = (entry: PoolWorker, job: Job): void => {
        entry.job = job;
        try {
            const posted = postable(job.geometry, job.transferInput);
            const request: BuildRequest = { id: entry.nextId++, geometry: posted.geometry, options: job.options };
            entry.worker.postMessage(request, posted.transfer);
            markStarted(job, true);
        } catch (error) {
            // A detached input, a buffer that can't be cloned: this build fails, the worker stays free.
            entry.job = null;
            finished++;
            settle(job, { error: new Error(`@pmndrs/bvh/worker: the build's input could not be sent to a worker: ${toError(error).message}`) });
        }
    };

    /** Starts queued builds on free, loaded workers (starting workers up to the pool's size), then watches the idle ones. */
    function pump(): void {
        if (disposed) return;
        startQueued();
        watchIdle();
    }

    function startQueued(): void {
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
            const entry = workers.find((candidate) => candidate.ready && candidate.job === null);
            if (entry) {
                send(entry, queue.shift()!);
                continue;
            }
            // Start workers for the builds waiting, up to the pool's size; each takes one once it has loaded.
            const loading = workers.filter((candidate) => !candidate.ready).length;
            const waiting = queue.filter((job) => !job.settled).length;
            if (loading >= waiting || workers.length >= size) return;
            let started: PoolWorker;
            try {
                started = { worker: startWorker(), job: null, nextId: 1, ready: false, idleTimer: null };
            } catch (error) {
                workerUnavailable(`@pmndrs/bvh/worker: a worker could not be started: ${toError(error).message}`);
                continue;
            }
            attach(started);
            workers.push(started);
        }
    }

    return {
        get usesWorkers() {
            return !inline;
        },
        get inlineReason() {
            return inlineReason;
        },
        get size() {
            return inline ? 1 : size;
        },
        get stats(): BvhWorkerPoolStats {
            return {
                queued: queue.filter((job) => !job.settled).length,
                running: inlineRunning + workers.filter((entry) => entry.job !== null).length,
                workers: workers.length,
                finished,
                recycled,
                idleEnded,
            };
        },
        build(geometry, buildOptions = {}, request = {}) {
            if (disposed) return Promise.reject(new Error('@pmndrs/bvh/worker: the pool was disposed'));
            if (request.signal?.aborted) return Promise.reject(abortError());
            // For the events only: geometry that isn't triangles fails in the build, as it always did.
            let triangleCount = 0;
            try {
                triangleCount = countTriangles(geometry);
            } catch {
                triangleCount = 0;
            }
            return new Promise<MeshBvh>((resolve, reject) => {
                const job: Job = {
                    id: nextId++,
                    queuedAt: performance.now(),
                    triangleCount,
                    onStart: request.onStart,
                    inWorker: false,
                    started: false,
                    geometry,
                    options: buildOptions,
                    transferInput: request.transferInput ?? options.transferInput ?? false,
                    signal: request.signal,
                    resolve,
                    reject,
                    settled: false,
                };
                if (request.signal) {
                    job.onAbort = () => settle(job, { error: abortError() });
                    request.signal.addEventListener('abort', job.onAbort, { once: true });
                }
                queue.push(job);
                notify(job, 'queued', { inWorker: !inline });
                pump();
            });
        },
        dispose() {
            if (disposed) return;
            disposed = true;
            if (inlineIdleTimer !== null) clearTimeout(inlineIdleTimer);
            const error = new Error('@pmndrs/bvh/worker: the pool was disposed');
            for (const entry of workers.splice(0)) {
                if (entry.idleTimer !== null) clearTimeout(entry.idleTimer);
                entry.worker.terminate();
                if (entry.job) settle(entry.job, { error });
            }
            for (const job of inlineJobs) settle(job, { error });
            for (const job of queue.splice(0)) settle(job, { error });
        },
    };
}

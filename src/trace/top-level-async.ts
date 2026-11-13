/**
 * Builds and checks top-level tree shapes in a module worker when the
 * environment has one, and on the calling thread (on a later turn) otherwise:
 * Node tests, or a host that blocks workers.
 *
 * As the mesh-tree pool (`worker/pool.ts`): requests wait until the worker
 * said it loaded, so one that can't load holds no input, and `'auto'` then
 * answers them inline (the worker never built a shape, so the inline keeper
 * starts from the same nothing).
 */

import { topLevelInputTransferables, TopLevelShapeKeeper, type TopLevelBuildInput, type TopLevelShape } from './top-level-shape.ts';
import type { TopLevelRequest, TopLevelResponse } from './top-level-worker.ts';
import type { WorkerReady } from './mesh-bvh-worker.ts';
import { workerLoadFailure } from '../worker/pool.ts';

export interface TopLevelResult {
    /** A new shape, or null when a check found the old one still good enough. */
    readonly shape: TopLevelShape | null;
    /** For checks: the old shape's relative SAH cost at the current transforms, and as built. */
    readonly relativeCost: number;
    readonly builtCost: number;
    /** CPU time of the request (in the worker when there is one). */
    readonly workerMs: number;
}

export interface TopLevelBuilder {
    /** Whether requests run in a worker (false once `'auto'` fell back to the calling thread). */
    readonly usesWorker: boolean;
    /**
     * `build` makes a new shape; `check` makes one only if the cost of shape
     * `generations.current` (the one in use) at these transforms passed
     * `ratio` times its as-built cost. A new shape is remembered as
     * `generations.next`. The input's arrays move to the worker: they are
     * unusable afterwards.
     */
    request(mode: 'build' | 'check', input: TopLevelBuildInput, ratio: number, generations: { current: number; next: number }): Promise<TopLevelResult>;
    /** Stops the worker. Pending requests reject. */
    dispose(): void;
}

export interface TopLevelBuilderConfig {
    /** `'auto'` (default) uses a worker when `Worker` exists, and works inline if it can't start or load; `'never'` always works inline. */
    readonly worker?: 'auto' | 'never';
    /** Where a worker that couldn't load is reported (once; requests then run inline). Default `console.warn`. */
    readonly warn?: (message: string) => void;
}

export function createTopLevelBuilder(config: TopLevelBuilderConfig = {}): TopLevelBuilder {
    const worker = config.worker === 'never' ? null : tryStartWorker();
    return worker === null ? createInlineBuilder() : createWorkerBuilder(worker, config.warn ?? ((message) => console.warn(message)));
}

function tryStartWorker(): Worker | null {
    if (typeof Worker === 'undefined') return null;
    try {
        // The URL must stay inline in `new Worker(new URL(...))` so bundlers can find the worker file.
        return new Worker(new URL('./top-level-worker.ts', import.meta.url), { type: 'module' });
    } catch {
        return null;
    }
}

function createInlineBuilder(): TopLevelBuilder {
    const keeper = new TopLevelShapeKeeper();
    return {
        usesWorker: false,
        request: (mode, input, ratio, generations) =>
            new Promise((resolve, reject) => {
                // A later turn, so callers see the same asynchronous behaviour as with a worker.
                setTimeout(() => {
                    try {
                        const start = performance.now();
                        const result =
                            mode === 'build'
                                ? { shape: keeper.build(input, generations.next), relativeCost: 0, builtCost: 0 }
                                : keeper.check(input, ratio, generations.current, generations.next);
                        resolve({ ...result, workerMs: performance.now() - start });
                    } catch (error) {
                        reject(error instanceof Error ? error : new Error(String(error)));
                    }
                }, 0);
            }),
        dispose: () => {},
    };
}

function createWorkerBuilder(worker: Worker, warn: (message: string) => void): TopLevelBuilder {
    const pending = new Map<number, { resolve(result: TopLevelResult): void; reject(error: Error): void }>();
    /** Requests asked for before the worker said it loaded; nothing of theirs was posted yet. */
    let waiting: { message: TopLevelRequest; resolve(result: TopLevelResult): void; reject(error: Error): void }[] = [];
    let ready = false;
    /** Set when the worker couldn't load: every request from then on runs here. */
    let inline: TopLevelBuilder | null = null;
    let nextId = 1;
    const failAll = (message: string): void => {
        for (const entry of pending.values()) entry.reject(new Error(message));
        pending.clear();
        for (const entry of waiting) entry.reject(new Error(message));
        waiting = [];
    };
    const post = (message: TopLevelRequest): void => worker.postMessage(message, topLevelInputTransferables(message.input));
    worker.onmessage = (event: MessageEvent<TopLevelResponse | WorkerReady>) => {
        if ('ready' in event.data) {
            ready = true;
            for (const entry of waiting) {
                pending.set(entry.message.id, entry);
                post(entry.message);
            }
            waiting = [];
            return;
        }
        const response = event.data;
        const entry = pending.get(response.id);
        if (!entry) return;
        pending.delete(response.id);
        if ('error' in response) entry.reject(new Error(response.error));
        else entry.resolve(response);
    };
    worker.onerror = (event) => {
        event.preventDefault?.();
        if (ready) {
            failAll(`top-level worker failed: ${event.message}`);
            return;
        }
        // It never loaded and holds nothing: answer its requests, and all later ones, inline.
        worker.terminate();
        warn(`@pmndrs/bvh: ${workerLoadFailure('the top-level worker', event.message)}; building top levels on the main thread`);
        inline = createInlineBuilder();
        for (const entry of waiting) {
            const { mode, input, ratio, generation, currentGeneration } = entry.message;
            inline.request(mode, input, ratio, { current: currentGeneration, next: generation }).then(entry.resolve, entry.reject);
        }
        waiting = [];
    };
    return {
        get usesWorker() {
            return inline === null;
        },
        request(mode, input, ratio, generations) {
            if (inline) return inline.request(mode, input, ratio, generations);
            return new Promise<TopLevelResult>((resolve, reject) => {
                const message: TopLevelRequest = { id: nextId++, mode, input, ratio, generation: generations.next, currentGeneration: generations.current };
                if (!ready) {
                    waiting.push({ message, resolve, reject });
                    return;
                }
                pending.set(message.id, { resolve, reject });
                post(message);
            });
        },
        dispose() {
            worker.terminate();
            failAll('the top-level builder was disposed');
        },
    };
}

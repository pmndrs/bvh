/// <reference types="node" />
/**
 * One library's memory on one scene, in a fresh process (the runner spawns
 * it; `--expose-gc` is required):
 *
 *   node --expose-gc --experimental-transform-types --import ./compare/node/register.mjs compare/node/memory.ts <adapter> <scene>
 *
 * Loads the scene and the library (its WASM included), collects garbage,
 * reads `process.memoryUsage()`, builds, collects again and reads it again
 * with the built structure still alive. The difference in `heapUsed +
 * external` (V8's heap plus array buffers and WASM memories) is what the
 * build left behind; the geometry was there before and is not in it. Prints
 * one JSON line.
 */

import { adapterById } from '../adapters/index.ts';
import { loadScene } from './scenes.ts';

const [adapterId, sceneId] = process.argv.slice(2);
const gc = (globalThis as { gc?: () => void }).gc;
if (!gc) throw new Error('run with --expose-gc');

// Every WASM memory the library instantiates (tinybvh's, Rapier's), so the heap growth can be split.
const memories: WebAssembly.Memory[] = [];
const instantiate = WebAssembly.instantiate;
(WebAssembly as { instantiate: unknown }).instantiate = (...args: unknown[]) =>
    (instantiate as (...a: unknown[]) => Promise<unknown>).apply(WebAssembly, args).then((result) => {
        const instance = (result as { instance?: WebAssembly.Instance }).instance ?? (result as WebAssembly.Instance);
        for (const value of Object.values(instance.exports ?? {})) if (value instanceof WebAssembly.Memory) memories.push(value);
        return result;
    });
const wasmBytes = () => memories.reduce((sum, memory) => sum + memory.buffer.byteLength, 0);

const adapter = adapterById(adapterId);
const geometry = loadScene(sceneId);
await adapter.prepare?.();
const used = () => {
    gc();
    gc();
    const usage = process.memoryUsage();
    return { heap: usage.heapUsed, external: usage.external };
};
const before = used();
const wasmBefore = wasmBytes();
const built = adapter.build(geometry);
const after = used();
const wasmAfter = wasmBytes();
const reported = built.memory();
console.log(
    JSON.stringify({
        adapter: adapterId,
        scene: sceneId,
        triangles: geometry.triangleCount,
        heapDelta: after.heap - before.heap,
        externalDelta: after.external - before.external,
        wasmGrowth: wasmAfter - wasmBefore,
        geometryCopy: adapter.geometryCopyBytes?.(geometry) ?? 0,
        total: after.heap + after.external - before.heap - before.external - (adapter.geometryCopyBytes?.(geometry) ?? 0),
        reported: reported.bytes,
        reportedMethod: reported.method,
    }),
);
built.dispose();

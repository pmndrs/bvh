/**
 * One entry point for building a mesh tree, for the worker and the inline
 * builder in `mesh-bvh-async.ts`. tinybvh (WASM) is the only mesh builder
 * (B0, DECISIONS 2026-11-04); its WASM is started once per thread, on first use.
 */

import { createTinyBvhBuilder, type TinyBvhBuilder } from '../tinybvh.ts';
import type { MeshBvh, MeshBvhBuildOptions, MeshGeometry } from './mesh-bvh.ts';
import { buildMeshBvhWithTinyBvh } from './mesh-bvh-topology.ts';
import { convertToWide } from '../wide-nodes.ts';

let tinyBvh: Promise<TinyBvhBuilder> | null = null;
let started: TinyBvhBuilder | null = null;

/**
 * Bytes this thread's tinybvh WASM memory had grown to at the end of its last
 * build (0 before the first). The builder drops a memory that grew past its
 * `releaseAboveBytes`, but the garbage collector frees it only later; a pool
 * worker reports this so the pool can end the worker instead (pool.ts).
 */
export function tinyBvhHeapBytes(): number {
    return started?.lastBuildHeapBytes ?? 0;
}

export async function buildMeshBvhAsync(geometry: MeshGeometry, options: MeshBvhBuildOptions = {}): Promise<MeshBvh> {
    tinyBvh ??= createTinyBvhBuilder().then((builder) => (started = builder));
    const bvh = buildMeshBvhWithTinyBvh(geometry, await tinyBvh, { maxLeafPrimitives: options.maxLeafPrimitives, triangleLeaves: options.triangleLeaves, nodeFormat: options.nodeFormat });
    if (!options.wideNodes) return bvh;
    // The collapse runs here, where the tree was built (a worker, for the pool), not on the main thread.
    if (bvh.nodeFormat === 'quantized') throw new RangeError('buildMeshBvhAsync: wideNodes collapses f32 trees; leave nodeFormat at f32');
    return { ...bvh, wide: convertToWide(bvh, options.wideNodes) };
}

/**
 * CPU stand-ins of the scene kernels for the fake GPU (fake-gpu.ts): the
 * merged-identity passes and the quantize pass, line by line as their WGSL.
 * Test-only.
 */

import type { FakeBinding, FakeKernel } from './fake-gpu.ts';
import { INDEXED_FLAG_SHIFT, INDEXED_ID_MASK, TRIANGLE_WORD, TRIANGLE_WORDS } from '../../src/gpu-layout.ts';
import { MERGED_SOURCE_WORDS } from '../../src/scene/merged-identity-gpu.ts';
import { quantizeNodeWords } from '../../src/quantized-nodes.ts';

/** The merged-identity WGSL, line by line on the CPU. */
export function mergedIdentityStandIn(bindings: ReadonlyMap<number, FakeBinding>): void {
    const triangles = new Uint32Array(bindings.get(0)!.buffer.bytes.buffer);
    const sources = new Uint32Array(bindings.get(1)!.buffer.bytes.buffer);
    const [wordOffset, count, sourceCount] = new Uint32Array(bindings.get(2)!.buffer.bytes.buffer);
    for (let slot = 0; slot < count; slot++) {
        const base = wordOffset + slot * TRIANGLE_WORDS;
        const merged = triangles[base + TRIANGLE_WORD.originalIndex];
        let low = 0;
        let high = sourceCount;
        while (high - low > 1) {
            const middle = (low + high) >> 1;
            if (sources[middle * MERGED_SOURCE_WORDS] <= merged) low = middle;
            else high = middle;
        }
        const at = low * MERGED_SOURCE_WORDS;
        triangles[base + TRIANGLE_WORD.originalIndex] = merged - sources[at];
        triangles[base + TRIANGLE_WORD.padding] = sources[at + 1];
        triangles[base + TRIANGLE_WORD.flags] = (triangles[base + TRIANGLE_WORD.flags] | sources[at + 2] | (materialOf(bindings, merged) << 16)) >>> 0;
    }
}

/** A merged triangle's material code from the identity pass's per-triangle table (binding 3, two codes to a word). */
function materialOf(bindings: ReadonlyMap<number, FakeBinding>, merged: number): number {
    const table = new Uint32Array(bindings.get(3)!.buffer.bytes.buffer);
    return (table[merged >> 1] >>> ((merged & 1) * 16)) & 0xffff;
}

/** The quantize pass (gpu-quantize.ts, `bvhqQuantize`) on the CPU: the CPU encoder it matches bit for bit (bvh-blas-refit checks that on GPUs). */
export function quantizeStandIn(bindings: ReadonlyMap<number, FakeBinding>): void {
    const source = bindings.get(0)!.buffer.bytes;
    const target = new Uint32Array(bindings.get(1)!.buffer.bytes.buffer);
    const table = new Uint32Array(bindings.get(2)!.buffer.bytes.buffer);
    const [meshCount] = new Uint32Array(bindings.get(3)!.buffer.bytes.buffer);
    for (let mesh = 0; mesh < meshCount; mesh++) {
        const [sourceNodeOffset, rootSlot, nodeCount] = table.subarray(mesh * 4, mesh * 4 + 3);
        const words = new Uint32Array(source.buffer.slice(source.byteOffset + sourceNodeOffset * 64, source.byteOffset + (sourceNodeOffset + nodeCount) * 64));
        quantizeNodeWords(new Float32Array(words.buffer), words, 0, nodeCount, target, (rootSlot - 1) * 8);
    }
}

/** The indexed merged-identity WGSL (`bvhMergedIdentityIndexed`), line by line on the CPU. */
export function mergedIdentityIndexedStandIn(bindings: ReadonlyMap<number, FakeBinding>): void {
    const triangles = new Uint32Array(bindings.get(0)!.buffer.bytes.buffer);
    const sources = new Uint32Array(bindings.get(1)!.buffer.bytes.buffer);
    const [wordOffset, count, sourceCount, sideWordOffset] = new Uint32Array(bindings.get(2)!.buffer.bytes.buffer);
    for (let slot = 0; slot < count; slot++) {
        const word = wordOffset + slot * 4 + 3;
        const merged = triangles[word] & INDEXED_ID_MASK;
        let low = 0;
        let high = sourceCount;
        while (high - low > 1) {
            const middle = (low + high) >> 1;
            if (sources[middle * MERGED_SOURCE_WORDS] <= merged) low = middle;
            else high = middle;
        }
        const at = low * MERGED_SOURCE_WORDS;
        const flags = (triangles[word] >>> INDEXED_FLAG_SHIFT) | (sources[at + 2] & 2);
        triangles[word] = ((merged - sources[at]) | (flags << INDEXED_FLAG_SHIFT)) >>> 0;
        triangles[sideWordOffset + slot] = (low | (materialOf(bindings, merged) << 16)) >>> 0;
    }
}

/** The kernels the SceneTrees tests run on the fake GPU, by entry point. */
export const SCENE_KERNELS: Record<string, FakeKernel> = { bvhMergedIdentity: mergedIdentityStandIn, bvhMergedIdentityIndexed: mergedIdentityIndexedStandIn, bvhqQuantize: quantizeStandIn };

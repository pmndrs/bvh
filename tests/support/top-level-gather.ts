/**
 * The GPU top level's gather (`bvhTopLevelGather`) as a CPU stand-in for the
 * fake GPU: a line-by-line port of its WGSL. Test-only.
 */

import type { FakeBinding } from './fake-gpu.ts';
import { INSTANCE_RECORD_WORDS, INSTANCE_WORD, TOP_LEVEL_HEADER_WORDS } from '../../src/trace/layout.ts';
import { TOP_LEVEL_INSTANCE_ENTRY_WORD, TOP_LEVEL_INSTANCE_ENTRY_WORDS, TOP_LEVEL_MESH_ABSENT, TOP_LEVEL_MESH_ENTRY_WORDS } from '../../src/trace/top-level-gpu.ts';

/** The gather's WGSL, line by line on the CPU: what the fake device runs for `bvhTopLevelGather`. */
export function gatherStandIn(bindings: ReadonlyMap<number, FakeBinding>): void {
    const words = (binding: number) => new Uint32Array(bindings.get(binding)!.buffer.bytes.buffer);
    const topLevel = words(0);
    const meshes = words(1);
    const recordSlots = words(2);
    const boxes = new Float32Array(bindings.get(3)!.buffer.bytes.buffer);
    const [recordCount, meshCapacity, instanceCapacity] = words(4);
    const instances = words(5);
    const entry = TOP_LEVEL_INSTANCE_ENTRY_WORD;
    for (let record = 0; record < recordCount; record++) {
        const slot = recordSlots[record];
        const base = TOP_LEVEL_HEADER_WORDS + record * INSTANCE_RECORD_WORDS;
        const at = slot * TOP_LEVEL_INSTANCE_ENTRY_WORDS;
        const ids = slot < instanceCapacity ? instances.subarray(at + entry.meshSlot, at + entry.meshSlot + 4) : new Uint32Array(4);
        let traced = slot < instanceCapacity && ids[3] !== 0 && (ids[2] & 3) === 3 && ids[0] < meshCapacity;
        if (traced) traced = meshes[ids[0] * TOP_LEVEL_MESH_ENTRY_WORDS + 3] !== TOP_LEVEL_MESH_ABSENT;
        if (!traced) {
            topLevel.fill(0, base, base + INSTANCE_RECORD_WORDS);
            topLevel[base + INSTANCE_WORD.instanceIndex] = ids[1];
            boxes.fill(0, record * 8, record * 8 + 8);
            continue;
        }
        topLevel.set(instances.subarray(at + entry.worldToObjectRow0, at + entry.worldToObjectRow0 + 12), base);
        topLevel[base + INSTANCE_WORD.rootNodeOffset] = meshes[ids[0] * TOP_LEVEL_MESH_ENTRY_WORDS + 3];
        topLevel[base + INSTANCE_WORD.triangleOffset] = meshes[ids[0] * TOP_LEVEL_MESH_ENTRY_WORDS + 7];
        topLevel[base + INSTANCE_WORD.instanceIndex] = ids[1];
        topLevel[base + INSTANCE_WORD.flags] = ((ids[2] & ~3) | 3) >>> 0;
    }
}

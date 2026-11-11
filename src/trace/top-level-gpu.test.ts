/**
 * The GPU top level's CPU side: what goes into the instance table, that the
 * gather turns it into the records the CPU top level writes, and when a new
 * shape is asked for. The fake device runs a CPU stand-in for the gather (a
 * line-by-line port of its WGSL) and nothing for the refit, so the refit and
 * the real gather are checked on GPUs (tests/gpu/pages/scene-trees).
 */

import { describe, expect, it } from 'vitest';
import { createFakeGpu, type FakeBuffer } from '../../tests/support/fake-gpu.ts';
import { gatherStandIn } from '../../tests/support/top-level-gather.ts';
import { INSTANCE_FLAG, INSTANCE_MASK_SHIFT, MATERIAL_SHIFT, INSTANCE_RECORD_WORDS, INSTANCE_WORD, TOP_HEADER_WORD, TOP_LEVEL_HEADER_WORDS } from './layout.ts';
import { createRandom, randomAffine } from './oracle-fixtures.ts';
import { MERGED_INSTANCE_INDEX } from './static-merge.ts';
import { buildTopLevel, IDENTITY_TRANSFORM } from './top-level.ts';
import {
    GpuTopLevel,
    packTopLevelInstance,
    TOP_LEVEL_INSTANCE_ENTRY_WORD,
    TOP_LEVEL_INSTANCE_ENTRY_WORDS,
    topLevelGatherWgsl,
} from './top-level-gpu.ts';

const BOUNDS = [-1, -0.5, -0.25, 1, 0.5, 0.25];

function setUp(options: { kernels?: boolean } = {}) {
    const gpu = createFakeGpu({}, { features: ['subgroups'], kernels: options.kernels === false ? {} : { bvhTopLevelGather: gatherStandIn } });
    const top = new GpuTopLevel(gpu.device, { buildIn: 'main' });
    const frame = (deepestMesh = 1) => {
        const encoder = gpu.createEncoder();
        top.update(encoder, { deepestMesh });
        gpu.submit(encoder);
        return encoder as unknown as { commands: { dispatch?: { entryPoint: string } }[] };
    };
    const gathers = (encoder: { commands: { dispatch?: { entryPoint: string } }[] }) => encoder.commands.filter((command) => command.dispatch?.entryPoint === 'bvhTopLevelGather').length;
    return { gpu, top, frame, gathers };
}

const translation = (x: number) => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 0, 0, 1];

describe('the GPU top level', () => {
    it('packs world-to-object rows and flags with the bits the CPU top level writes', () => {
        const random = createRandom(7);
        const floats = new Float32Array(TOP_LEVEL_INSTANCE_ENTRY_WORDS);
        const words = new Uint32Array(floats.buffer);
        for (let index = 0; index < 40; index++) {
            const objectToWorld = randomAffine(random, { extent: 30, mirror: index % 2 === 0 });
            const mask = index % 3 === 0 ? undefined : index;
            const material = index % 4 === 0 ? undefined : index * 3;
            expect(packTopLevelInstance({ meshSlot: 2, instanceIndex: 100 + index, objectToWorld, mask, material }, floats, words, 0)).toBe(true);
            const cpu = buildTopLevel([{ instanceIndex: 100 + index, nodeOffset: 0, triangleOffset: 0, meshBounds: BOUNDS, objectToWorld, mask, material }]);
            const record = new Uint32Array(cpu.buffer, TOP_LEVEL_HEADER_WORDS * 4, INSTANCE_RECORD_WORDS);
            expect(words.subarray(TOP_LEVEL_INSTANCE_ENTRY_WORD.worldToObjectRow0, TOP_LEVEL_INSTANCE_ENTRY_WORD.worldToObjectRow0 + 12)).toEqual(record.subarray(0, 12));
            const flags = words[TOP_LEVEL_INSTANCE_ENTRY_WORD.flags];
            // Every bit as the CPU top level writes it, material + 1 in bits 16-31 included.
            expect(flags).toBe(record[INSTANCE_WORD.flags]);
            expect(flags >>> MATERIAL_SHIFT).toBe(material === undefined ? 0 : material + 1);
            expect((flags >>> INSTANCE_MASK_SHIFT) & 0xff).toBe(mask ?? 0xff);
        }
        // Merged: identity only, instance index 0xffffffff and bit 3.
        packTopLevelInstance({ meshSlot: 0, instanceIndex: 5, objectToWorld: IDENTITY_TRANSFORM, merged: true }, floats, words, 0);
        expect(words[TOP_LEVEL_INSTANCE_ENTRY_WORD.instanceIndex]).toBe(MERGED_INSTANCE_INDEX);
        expect(words[TOP_LEVEL_INSTANCE_ENTRY_WORD.flags] & INSTANCE_FLAG.merged).toBe(INSTANCE_FLAG.merged);
        expect(() => packTopLevelInstance({ meshSlot: 0, instanceIndex: 5, objectToWorld: translation(1), merged: true }, floats, words, 0)).toThrow(/merged static instance/);
        expect(() => packTopLevelInstance({ meshSlot: 0, instanceIndex: 5, objectToWorld: IDENTITY_TRANSFORM, material: 0x10000 }, floats, words, 0)).toThrow(RangeError);
        expect(() => packTopLevelInstance({ meshSlot: 0, instanceIndex: 5, objectToWorld: IDENTITY_TRANSFORM, merged: true, material: 1 }, floats, words, 0)).toThrow(/merged record/);
        expect(() => packTopLevelInstance({ meshSlot: 0, instanceIndex: 5, objectToWorld: IDENTITY_TRANSFORM, mask: 256 }, floats, words, 0)).toThrow(RangeError);
        // A singular transform is not live: rays never see it.
        expect(packTopLevelInstance({ meshSlot: 0, instanceIndex: 5, objectToWorld: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] }, floats, words, 0)).toBe(false);
    });

    it('declares only bvhGather* names besides the instance box', () => {
        const wgsl = topLevelGatherWgsl();
        expect(wgsl).toContain('fn bvhInstanceWorldBox(');
        expect(wgsl).toContain('fn bvhTopLevelGather(');
        const declared = [...wgsl.matchAll(/(?:var<[^>]+>|const|fn|struct)\s+([A-Za-z_]\w*)/g)].map((match) => match[1]);
        expect(declared.every((name) => /^(bvh|Bvh|BVH_)/.test(name))).toBe(true);
    });

    it('gathers records as the CPU top level writes them, and builds a new shape only when instances come or go', async () => {
        const { gpu, top, frame, gathers } = setUp();
        await top.prepare();
        top.setMeshTree(3, { nodeOffset: 10, triangleOffset: 20, bounds: BOUNDS });
        top.setInstance(0, { meshSlot: 3, instanceIndex: 7, objectToWorld: translation(0), material: 4 });
        top.setInstance(1, { meshSlot: 3, instanceIndex: 8, objectToWorld: translation(3), mask: 0x0f });
        top.setInstance(2, { meshSlot: 3, instanceIndex: 0, objectToWorld: IDENTITY_TRANSFORM, merged: true });
        frame();
        expect(top.building).toBe(true);
        await top.whenIdle();
        frame();
        expect(top.building).toBe(false);
        expect(top.counters.shapesBuilt).toBe(1);
        expect(top.instanceCount).toBe(3);
        const bytes = (top.buffer as unknown as FakeBuffer).bytes;
        const header = new Uint32Array(bytes.buffer, 0, TOP_LEVEL_HEADER_WORDS);
        expect(header[TOP_HEADER_WORD.instanceCount]).toBe(3);

        const recordOf = (instanceIndex: number) => {
            for (let record = 0; record < 3; record++) {
                const words = new Uint32Array(bytes.buffer, (TOP_LEVEL_HEADER_WORDS + record * INSTANCE_RECORD_WORDS) * 4, INSTANCE_RECORD_WORDS);
                if (words[INSTANCE_WORD.instanceIndex] === instanceIndex) return words;
            }
            throw new Error(`no record for ${instanceIndex}`);
        };
        const cpu = buildTopLevel([
            { instanceIndex: 7, nodeOffset: 10, triangleOffset: 20, meshBounds: BOUNDS, objectToWorld: translation(0), material: 4 },
            { instanceIndex: 8, nodeOffset: 10, triangleOffset: 20, meshBounds: BOUNDS, objectToWorld: translation(3), mask: 0x0f },
            { instanceIndex: MERGED_INSTANCE_INDEX, nodeOffset: 10, triangleOffset: 20, meshBounds: BOUNDS, objectToWorld: IDENTITY_TRANSFORM, merged: true },
        ]);
        for (const instanceIndex of [7, 8, MERGED_INSTANCE_INDEX]) {
            const record = recordOf(instanceIndex);
            const expected = (() => {
                for (let record = 0; record < 3; record++) {
                    const words = new Uint32Array(cpu.buffer, (TOP_LEVEL_HEADER_WORDS + record * INSTANCE_RECORD_WORDS) * 4, INSTANCE_RECORD_WORDS);
                    if (words[INSTANCE_WORD.instanceIndex] === instanceIndex) return words;
                }
                throw new Error('missing');
            })();
            expect(record.subarray(0, 15)).toEqual(expected.subarray(0, 15));
            expect(record[INSTANCE_WORD.flags]).toBe(expected[INSTANCE_WORD.flags]);
        }
        expect(recordOf(7)[INSTANCE_WORD.flags] >>> 16).toBe(5);

        // Moving and hiding: one refit pass each, no new shape.
        top.setTransform(0, translation(-2));
        expect(gathers(frame())).toBe(1);
        top.setInstance(1, { meshSlot: 3, instanceIndex: 8, objectToWorld: translation(3), visible: false });
        expect(gathers(frame())).toBe(1);
        expect(recordOf(8)[INSTANCE_WORD.flags]).toBe(0);
        expect(top.counters.shapesBuilt).toBe(1);
        expect(top.building).toBe(false);
        // Nothing changed: no pass.
        expect(gathers(frame())).toBe(0);

        // A merged record never moves.
        expect(() => top.setTransform(2, translation(1))).toThrow(/merged static instance/);
        expect(() => top.setInstance(2, { meshSlot: 3, instanceIndex: 0, objectToWorld: translation(1) })).toThrow(/merged static instance/);

        // A mesh tree released: its instances get empty records at once, with no new shape.
        top.setMeshTree(3, null);
        expect(gathers(frame())).toBe(1);
        expect(recordOf(7)[INSTANCE_WORD.flags]).toBe(0);

        // Removing an instance asks for a new shape.
        top.setInstance(0, null);
        frame();
        await top.whenIdle();
        frame();
        expect(top.counters.shapesBuilt).toBe(2);
        expect(top.instanceCount).toBe(2);
        top.dispose();
        expect(gpu.buffers.filter((buffer) => buffer.label.includes('top level')).every((buffer) => buffer.destroyed)).toBe(true);
    });

    it('refuses a device without subgroups and a scene too deep for the trace stack', async () => {
        expect(() => new GpuTopLevel(createFakeGpu().device)).toThrow(/subgroups/);
        const { top, frame } = setUp();
        await top.prepare();
        top.setMeshTree(0, { nodeOffset: 0, triangleOffset: 0, bounds: BOUNDS });
        for (let slot = 0; slot < 4; slot++) top.setInstance(slot, { meshSlot: 0, instanceIndex: slot, objectToWorld: translation(slot * 3) });
        frame(62);
        await top.whenIdle();
        expect(() => frame(62)).toThrow(/trace stack/);
        expect(top.instanceCount).toBe(0);
        // The dropped shape isn't asked for again until the instances change.
        top.setInstance(4, { meshSlot: 0, instanceIndex: 4, objectToWorld: translation(12) });
        frame(10);
        await top.whenIdle();
        frame(10);
        expect(top.instanceCount).toBe(5);
        top.dispose();
    });
});

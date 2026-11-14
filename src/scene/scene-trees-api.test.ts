/**
 * SceneTrees' tooling API in Node (scene-trees.test.ts has the trees
 * themselves): timestamps around an update and on GPU builds. The first trees
 * come from `GpuFirstTrees` over the stand-in H-PLOC (tests/support/fake-hploc.ts),
 * whose trees are tinybvh's with one triangle per leaf; the quantize pass
 * runs as its CPU twin.
 */

import { describe, expect, it } from 'vitest';
import { createFakeGpu, type FakeBinding, type FakeEncoder, type FakeGpu } from '../../tests/support/fake-gpu.ts';
import { FakeHploc } from '../../tests/support/fake-hploc.ts';
import { addFakeTimestamps } from '../../tests/support/fake-timestamps.ts';
import { quantizeNodeWords } from '../quantized-nodes.ts';
import { tessellatedQuad } from '../trace/oracle-fixtures.ts';
import { GpuFirstTrees } from './gpu-first-trees.ts';
import { SceneTrees, type SceneTreesOptions } from './scene-trees.ts';

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** The quantize pass (gpu-quantize.ts, `bvhqQuantize`) on the CPU. */
function quantizeStandIn(bindings: ReadonlyMap<number, FakeBinding>): void {
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

/** Waits (a few ms at a time, at most a second) until `condition` holds. */
async function until(condition: () => boolean, what = 'the condition'): Promise<void> {
    for (let tries = 0; tries < 200 && !condition(); tries++) await new Promise((resolve) => setTimeout(resolve, 5));
    if (!condition()) throw new Error(`${what} never held`);
}

async function setUp(options: Partial<SceneTreesOptions> & { timestampMs?: number } = {}) {
    const gpu: FakeGpu = createFakeGpu({}, { kernels: { bvhqQuantize: quantizeStandIn } });
    if (options.timestampMs !== undefined) addFakeTimestamps(gpu, options.timestampMs);
    const hploc = new FakeHploc(gpu.device);
    const scene = await SceneTrees.create(gpu.device, { workers: { worker: 'never' }, gpuMaxBatchBytes: Infinity, ...options, firstTreeBuilder: new GpuFirstTrees(gpu.device, { hploc }) });
    const frame = () => {
        const encoder = gpu.createEncoder();
        scene.update(encoder);
        gpu.submit(encoder);
    };
    return { gpu, hploc, scene, frame };
}

describe('SceneTrees timing', () => {
    it('brackets an update with timestamp passes (begin first, end last), also when it throws', async () => {
        const { gpu, scene } = await setUp();
        const querySet = {} as GPUQuerySet;
        const mesh = scene.addMesh(tessellatedQuad(4, 1));
        scene.addInstance({ mesh, objectToWorld: IDENTITY });
        const encoder = gpu.createEncoder();
        scene.update(encoder, { timestampWrites: { querySet, beginningOfPassWriteIndex: 4, endOfPassWriteIndex: 5 } });
        const commands = (encoder as unknown as FakeEncoder).commands;
        const dispatches = commands.flatMap((command) => ('dispatch' in command ? [command.dispatch.entryPoint] : []));
        expect(dispatches[0]).toBe('bvhUpdateTimestamp');
        expect(dispatches.at(-1)).toBe('bvhUpdateTimestamp');
        expect('dispatch' in commands[0] && 'dispatch' in commands.at(-1)!).toBe(true);
        gpu.submit(encoder);
        await scene.whenRefined();

        // Only the indices given are written: no end index, one pass.
        const half = gpu.createEncoder();
        scene.update(half, { timestampWrites: { querySet, beginningOfPassWriteIndex: 0 } });
        expect((half as unknown as FakeEncoder).commands.filter((command) => 'dispatch' in command && command.dispatch.entryPoint === 'bvhUpdateTimestamp')).toHaveLength(1);
        gpu.submit(half);

        // An update that throws still closes its bracket.
        const failing = gpu.createEncoder();
        scene.dispose();
        expect(() => scene.update(failing, { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } })).toThrow('disposed');
        const live = await setUp();
        const broken = gpu.createEncoder();
        (live.scene as unknown as { updateNow: () => void }).updateNow = () => {
            throw new Error('test failure');
        };
        expect(() => live.scene.update(broken, { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } })).toThrow('test failure');
        expect((broken as unknown as FakeEncoder).commands.filter((command) => 'dispatch' in command)).toHaveLength(2);
        live.scene.dispose();
    });

    it('sums GPU build times from timestamps with timestamps: true, and gives null without', async () => {
        for (const [timestamps, timestampMs, expected] of [
            [true, 3, 3],
            [false, 3, null],
            [true, undefined, null],
        ] as const) {
            const { scene, frame } = await setUp({ timestamps, timestampMs });
            const mesh = scene.addMesh(tessellatedQuad(4, 1));
            scene.addInstance({ mesh, objectToWorld: IDENTITY });
            frame();
            await until(() => scene.counts.gpuBatches > 0);
            expect(scene.counts.gpuBatches).toBe(1);
            expect(scene.counts.gpuBuildTimestampMs).toBe(expected);
            scene.dispose();
        }
    });
});

/**
 * SceneTrees' tooling API in Node (scene-trees.test.ts has the trees
 * themselves): build events, GPU-only scenes, the app's own trees, rebuilds
 * on request, and timestamps around an update and on GPU builds. The first
 * trees come from `GpuFirstTrees` over the stand-in H-PLOC
 * (tests/support/fake-hploc.ts), whose trees are tinybvh's with one triangle
 * per leaf; the quantize pass runs as its CPU twin, and the CPU walk reads
 * the scene's buffers from the fake GPU.
 */

import { describe, expect, it } from 'vitest';
import { createFakeGpu, type FakeBinding, type FakeBuffer, type FakeEncoder, type FakeGpu } from '../../tests/support/fake-gpu.ts';
import { FakeHploc } from '../../tests/support/fake-hploc.ts';
import { addFakeTimestamps } from '../../tests/support/fake-timestamps.ts';
import { quantizeNodes, quantizeNodeWords } from '../quantized-nodes.ts';
import { createCpuTracer } from '../trace/cpu-traversal.ts';
import { TOP_HEADER_WORD } from '../gpu-layout.ts';
import { buildMeshBvhAsync } from '../trace/mesh-bvh-build.ts';
import { tessellatedQuad } from '../trace/oracle-fixtures.ts';
import type { TraceSceneData } from '../trace/scene-assembly.ts';
import type { TopLevelBvh } from '../trace/top-level.ts';
import { GpuFirstTrees, type FirstTreeBuilder } from './gpu-first-trees.ts';
import { createBvhWorkerPool, type BvhPoolBuildEvent } from '../worker/pool.ts';
import { SceneTrees, type SceneBuildEvent, type SceneTreeEvent, type SceneTreesOptions } from './scene-trees.ts';

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

describe('SceneTrees build events', () => {
    const line = (event: SceneBuildEvent) => `${event.mesh ?? `g${event.group}`} ${event.builder} ${event.phase}${event.outcome ? ` ${event.outcome}` : ''}`;

    it('reports each mesh requested, started and finished, per builder, through the worker pool to the placement', async () => {
        const events: SceneBuildEvent[] = [];
        const poolEvents: BvhPoolBuildEvent[] = [];
        const pool = createBvhWorkerPool({ worker: 'never', onBuild: (event) => poolEvents.push(event) });
        const { scene, frame } = await setUp({ workers: pool, onBuild: (event) => events.push(event), timestamps: true, timestampMs: 2 });
        const a = scene.addMesh(tessellatedQuad(4, 1));
        const b = scene.addMesh(tessellatedQuad(6, 1));
        scene.addInstance({ mesh: a, objectToWorld: IDENTITY });
        scene.addInstance({ mesh: b, objectToWorld: IDENTITY });
        expect(events).toEqual([]); // builds start in update
        frame();
        // The GPU batch (submitted at once) and both tinybvh builds were asked for in the update.
        expect(events.slice(0, 4).map(line)).toEqual([`${a} gpu requested`, `${b} gpu requested`, `${a} gpu started`, `${b} gpu started`]);
        expect(events.slice(4).map(line)).toEqual([`${b} tinybvh requested`, `${a} tinybvh requested`]); // biggest first
        await scene.whenRefined();
        await until(() => events.filter((event) => event.phase === 'finished').length === 4, 'four finished builds');
        for (const mesh of [a, b]) {
            for (const builder of ['gpu', 'tinybvh'] as const) {
                const phases = events.filter((event) => event.mesh === mesh && event.builder === builder);
                expect(phases.map((event) => event.phase)).toEqual(['requested', 'started', 'finished']);
                expect(phases[2].outcome).toBe('built');
                expect(phases[0].ms).toBe(0);
                expect(phases[2].ms).toBeGreaterThanOrEqual(phases[1].ms);
                expect(phases[2].triangleCount).toBe(mesh === a ? 32 : 72);
                expect(phases[2].rebuild).toBe(false);
                if (builder === 'gpu') expect(phases[2].gpuMs).toBeCloseTo(2, 6);
                else expect(phases[2].gpuMs).toBeUndefined();
            }
        }
        // The pool saw the same builds, with the geometry objects it was given.
        expect(poolEvents.map((event) => event.phase).filter((phase) => phase === 'finished')).toHaveLength(2);
        scene.dispose();
        pool.dispose();
    });

    it('finishes builds as refused (too deep) and cancelled (the mesh removed)', async () => {
        const events: SceneBuildEvent[] = [];
        const { scene, frame } = await setUp({ onBuild: (event) => events.push(event), gpuMaxDepth: 1, warn: () => {} });
        const mesh = scene.addMesh(tessellatedQuad(4, 1));
        frame(); // no instance needed: builds start for every new mesh
        scene.removeMesh(mesh);
        await until(() => events.filter((event) => event.phase === 'finished').length === 2, 'both builds finished');
        const finished = events.filter((event) => event.phase === 'finished');
        expect(finished.find((event) => event.builder === 'tinybvh')).toMatchObject({ outcome: 'cancelled', mesh });
        // The GPU batch was already on its way: removed before its stats came, its tree is dropped.
        expect(finished.find((event) => event.builder === 'gpu')?.outcome).toMatch(/cancelled|refused/);

        events.length = 0;
        const deep = scene.addMesh(tessellatedQuad(4, 1));
        scene.addInstance({ mesh: deep, objectToWorld: IDENTITY });
        frame();
        await until(() => events.filter((event) => event.phase === 'finished').length === 2, 'both builds finished');
        for (const event of events.filter((candidate) => candidate.phase === 'finished')) {
            expect(event.outcome).toBe('refused');
            expect(event.reason).toMatch(/gpuMaxDepth/);
        }
        scene.dispose();
    });
});

/** The scene's buffers as the fake GPU holds them, for the CPU walk. */
function readScene(scene: SceneTrees): TraceSceneData {
    const { nodes, triangles, topLevel, positions } = scene.buffers;
    const bytes = (buffer: GPUBuffer) => (buffer as unknown as FakeBuffer).bytes.slice().buffer;
    const top = bytes(topLevel);
    const header = new Uint32Array(top, 0, 4);
    return {
        nodes: bytes(nodes),
        nodeFormat: scene.nodeFormat,
        triangles: bytes(triangles),
        triangleLeaves: scene.triangleLeaves,
        positions: positions ? bytes(positions) : new ArrayBuffer(0),
        positionLayout: scene.positionLayout,
        topLevel: { buffer: top, instanceCount: header[TOP_HEADER_WORD.instanceCount], nodeCount: header[TOP_HEADER_WORD.nodeCount] } as TopLevelBvh,
    } as unknown as TraceSceneData;
}

const down = (x: number, y: number) => ({ origin: [x, y, 3] as const, direction: [0, 0, -1] as const, tMin: 0, tMax: 100 });

describe('SceneTrees GPU-only (refine: false)', () => {
    it('keeps the GPU trees, builds nothing with tinybvh, and falls back to tinybvh for a mesh the GPU refused', async () => {
        const poolEvents: BvhPoolBuildEvent[] = [];
        const pool = createBvhWorkerPool({ worker: 'never', onBuild: (event) => poolEvents.push(event) });
        const { scene, frame, hploc } = await setUp({ refine: false, workers: pool, warn: () => {} });
        const a = scene.addMesh(tessellatedQuad(4, 1));
        scene.addInstance({ mesh: a, objectToWorld: IDENTITY, instanceIndex: 1 });
        frame();
        await scene.whenRefined();
        frame();
        expect(scene.meshStatus(a).builtBy).toBe('gpu');
        expect(scene.counts).toMatchObject({ onGpuTrees: 1, onTinybvhTrees: 0, tinybvhBuilding: 0, swaps: 0 });
        expect(scene.refining).toBe(false);
        expect(poolEvents).toEqual([]);
        expect(createCpuTracer(readScene(scene)).closest(down(0.1, 0.2))?.instanceIndex).toBe(1);

        // The next batch fails: its mesh falls back to tinybvh, the declared fallback.
        hploc.failFinish.add(1);
        const b = scene.addMesh(tessellatedQuad(3, 1));
        scene.addInstance({ mesh: b, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1], instanceIndex: 2 });
        frame();
        await until(() => scene.counts.onTinybvhTrees === 0 && poolEvents.some((event) => event.phase === 'finished'), 'the fallback build');
        await scene.whenRefined();
        frame();
        expect(scene.meshStatus(b)).toMatchObject({ builtBy: 'tinybvh' });
        expect(scene.meshStatus(b).fallbackReason).toMatch(/failed/);
        expect(scene.meshStatus(a).builtBy).toBe('gpu');
        expect(poolEvents.filter((event) => event.phase === 'queued')).toHaveLength(1);
        expect(createCpuTracer(readScene(scene)).closest(down(5.1, 0.2))?.instanceIndex).toBe(2);
        scene.dispose();
        pool.dispose();
    });
});

describe('SceneTrees.placeTree', () => {
    it("places the app's tree: before the first update no build starts, later ones are cancelled or dropped", async () => {
        const trees: SceneTreeEvent[] = [];
        const builds: SceneBuildEvent[] = [];
        const { scene, frame, hploc } = await setUp({ onTree: (event) => trees.push(event), onBuild: (event) => builds.push(event) });
        const quad = tessellatedQuad(4, 1);
        const own = await buildMeshBvhAsync(quad, { maxLeafPrimitives: 3 });
        const a = scene.addMesh(quad);
        scene.addInstance({ mesh: a, objectToWorld: IDENTITY, instanceIndex: 7 });
        scene.placeTree(a, own);
        frame();
        expect(hploc.batches).toEqual([]); // no GPU build
        expect(builds).toEqual([]); // and no tinybvh build
        expect(scene.meshStatus(a)).toMatchObject({ builtBy: 'app', fallbackReason: null });
        expect(scene.counts).toMatchObject({ onAppTrees: 1, onGpuTrees: 0, onTinybvhTrees: 0, waitingForFirstTree: 0 });
        expect(trees.map((event) => event.builtBy)).toEqual(['app']);
        await scene.whenBuilt();
        expect(createCpuTracer(readScene(scene)).closest(down(0.1, 0.2))?.instanceIndex).toBe(7);

        // After the builds started: tinybvh's is cancelled, the GPU's dropped; the app's tree stays.
        const b = scene.addMesh(tessellatedQuad(5, 1));
        scene.addInstance({ mesh: b, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1], instanceIndex: 8 });
        frame();
        scene.placeTree(b, await buildMeshBvhAsync(tessellatedQuad(5, 1)));
        frame();
        await scene.whenRefined();
        await until(() => builds.filter((event) => event.mesh === b && event.phase === 'finished').length === 2, "b's builds finished");
        frame();
        frame();
        expect(scene.meshStatus(b).builtBy).toBe('app');
        expect(builds.find((event) => event.mesh === b && event.builder === 'tinybvh' && event.phase === 'finished')?.outcome).toBe('cancelled');
        expect(scene.counts.swaps).toBe(0);
        expect(scene.meshStatus(b).fallbackReason).toBeNull();
        const tracer = createCpuTracer(readScene(scene));
        expect(tracer.closest(down(5.1, 0.2))?.instanceIndex).toBe(8);
        expect(tracer.closest(down(0.1, 0.2))?.instanceIndex).toBe(7);

        // Refused: another mesh's tree, the other triangle format, a quantized tree in an f32 scene, too deep.
        expect(() => scene.placeTree(a, own.triangleCount === 32 ? { ...own, triangleCount: 31 } : own)).toThrow(/triangles/);
        expect(() => scene.placeTree(a, { ...own, triangleLeaves: 'indexed' })).toThrow(/indexed/);
        expect(() => scene.placeTree(a, { ...own, maxDepth: 99 })).toThrow(/gpuMaxDepth/);
        scene.dispose();
        const f32 = await setUp({ nodeFormat: 'f32' });
        const mesh = f32.scene.addMesh(quad);
        expect(() => f32.scene.placeTree(mesh, quantizeNodes(own))).toThrow(/quantized scene/);
        f32.scene.dispose();
    });
});

describe('SceneTrees.rebuild', () => {
    it('asks for a deforming mesh\'s rebuild from its GPU pose at the next update, and says when it can\'t', async () => {
        const gpu = createFakeGpu({}, { features: ['subgroups'], kernels: { bvhqQuantize: quantizeStandIn } });
        const requests: { geometry: unknown; readLayouts: boolean }[][] = [];
// Refuses first builds (tinybvh's trees come), and records a rebuild's request (from the GPU pose: never answered here).
        const builder: FirstTreeBuilder = {
            prepare: () => Promise.resolve(),
            build: (batch, options) => {
                requests.push(batch.map((request) => ({ geometry: request.geometry, readLayouts: options?.readLayouts ?? false })));
                const fromPose = batch.filter((request) => 'positionBuffer' in request.geometry);
                return {
                    notBuilt: batch.filter((request) => !fromPose.includes(request)).map(({ slot, generation }) => ({ slot, generation, reason: 'refused (test)' })),
                    batches: fromPose.length > 0 ? [new Promise<never>(() => {})] : [],
                    buildingBytes: 0,
                };
            },
        };
        const builds: SceneBuildEvent[] = [];
        const scene = await SceneTrees.create(gpu.device, { firstTreeBuilder: builder, topLevel: 'cpu', workers: { worker: 'never' }, onBuild: (event) => builds.push(event) });
        const frame = () => {
            const encoder = gpu.createEncoder();
            scene.update(encoder);
            gpu.submit(encoder);
        };
        const rock = scene.addMesh(tessellatedQuad(2, 1));
        const cloth = scene.addMesh(tessellatedQuad(4, 1), { deforming: true });
        scene.addInstance({ mesh: cloth, objectToWorld: IDENTITY });
        expect(() => scene.rebuild(rock)).toThrow(/not added as deforming/);
        expect(scene.rebuild(cloth)).toBe(false); // no tree, no pose
        frame();
        await scene.whenRefined();
        frame();
        expect(scene.meshStatus(cloth).builtBy).toBe('tinybvh');
        expect(scene.rebuild(cloth)).toBe(false); // a tree, but no GPU pose yet
        const positionBuffer = gpu.device.createBuffer({ size: 25 * 12, usage: 0x80 | 0x4 });
        scene.setPose(cloth, { positionBuffer, vertexCount: 25 });
        expect(scene.rebuild(cloth)).toBe(true);
        const before = requests.length;
        frame();
        expect(requests.length).toBe(before + 1);
        expect(requests.at(-1)![0].readLayouts).toBe(true);
        expect((requests.at(-1)![0].geometry as { positionBuffer?: GPUBuffer }).positionBuffer).toBe(positionBuffer);
        expect(scene.deformingCounts.driftRebuilds).toBe(1);
        expect(builds.filter((event) => event.rebuild).map((event) => event.phase)).toEqual(['requested']);
        frame(); // asked once: no second rebuild
        expect(requests.length).toBe(before + 1);
        scene.dispose();

        // Without subgroups there is no GPU builder to rebuild with.
        const plain = await setUp();
        const mesh = plain.scene.addMesh(tessellatedQuad(2, 1), { deforming: true });
        expect(plain.scene.rebuild(mesh)).toBe(false);
        plain.scene.dispose();
    });
});

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

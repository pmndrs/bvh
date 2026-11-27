/**
 * `SceneTreesOptions.onTrianglesWritten` (pmndrs/bvh#5): the hook that names,
 * once per update and with the update's encoder, every mesh whose triangle
 * records the update wrote, so the app can write per-triangle flags of its own
 * (alpha-tested or see-through triangles) right after, in the same frame.
 *
 * Node, the fake GPU and the stand-in H-PLOC, as the other scene-trees tests.
 * The app's flags here are the filter bit on every triangle of one mesh; the
 * CPU walk with a filter that lets every flagged triangle pass then sees
 * through that mesh exactly while the flags are in the records.
 */

import { describe, expect, it } from 'vitest';
import { createFakeGpu, type FakeBuffer, type FakeGpu } from '../../tests/support/fake-gpu.ts';
import { FakeHploc } from '../../tests/support/fake-hploc.ts';
import { SCENE_KERNELS } from '../../tests/support/scene-stand-ins.ts';
import { INDEXED_FLAG_FILTER, INDEXED_TRIANGLE_WORD, INDEXED_TRIANGLE_WORDS, TOP_HEADER_WORD, TRIANGLE_FLAG_FILTER, TRIANGLE_WORD, TRIANGLE_WORDS } from '../gpu-layout.ts';
import { createCpuTracer } from '../trace/cpu-traversal.ts';
import type { MeshGeometry } from '../trace/mesh-bvh.ts';
import { tessellatedQuad } from '../trace/oracle-fixtures.ts';
import type { TraceRay } from '../trace/rays.ts';
import type { TraceSceneData } from '../trace/scene-assembly.ts';
import type { TopLevelBvh } from '../trace/top-level.ts';
import { GpuFirstTrees } from './gpu-first-trees.ts';
import { SceneTrees, type SceneTrianglesWrittenEvent, type SceneTreesOptions } from './scene-trees.ts';

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const MOVED_RIGHT = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1];

async function setUp(options: Partial<SceneTreesOptions> = {}) {
    const gpu: FakeGpu = createFakeGpu({}, { features: ['subgroups'], kernels: SCENE_KERNELS });
    const hploc = new FakeHploc(gpu.device);
    const scene = await SceneTrees.create(gpu.device, {
        workers: { worker: 'never' },
        gpuMaxBatchBytes: Infinity,
        topLevel: 'cpu',
        ...options,
        firstTreeBuilder: new GpuFirstTrees(gpu.device, { hploc, triangleLeaves: options.triangleLeaves }),
    });
    const frame = () => {
        const encoder = gpu.createEncoder();
        scene.update(encoder);
        gpu.submit(encoder);
    };
    return { gpu, scene, frame };
}

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
    } as TraceSceneData;
}

/** A ray straight down onto the quads (they lie in z = 0 over [-1, 1]^2). */
function down(x: number, y: number): TraceRay {
    return { origin: [x, y, 5], direction: [0, 0, -1], tMin: 0, tMax: 100 };
}

/** The bump on a cloth's pose. */
function lifted(geometry: MeshGeometry): Float32Array {
    const positions = geometry.positions.slice();
    for (let at = 2; at < positions.length; at += 3) positions[at] += 0.25;
    return positions;
}

/**
 * The app's side: sets the filter bit on every triangle of the written ranges of
 * the meshes in `flagged`, other bits kept. A real app records a compute pass
 * into `event.encoder`, after everything the update recorded; here the writes
 * wait in `pending` and run after the frame's submit, which is where that pass runs.
 */
function flagWriter(flagged: ReadonlySet<number>, pending: (() => void)[]) {
    return (event: SceneTrianglesWrittenEvent) => {
        const records = event.triangleLeaves === 'records';
        const words = new Uint32Array((event.triangles as unknown as FakeBuffer).bytes.buffer);
        const bit = records ? TRIANGLE_FLAG_FILTER : INDEXED_FLAG_FILTER;
        for (const range of event.written) {
            if (range.mesh === null || !flagged.has(range.mesh)) continue;
            for (let triangle = 0; triangle < range.triangleCount; triangle++) {
                const word = records
                    ? (range.triangleOffset + triangle) * TRIANGLE_WORDS + TRIANGLE_WORD.flags
                    : (range.triangleOffset + triangle) * INDEXED_TRIANGLE_WORDS + INDEXED_TRIANGLE_WORD.idFlags;
                pending.push(() => (words[word] = (words[word] | bit) >>> 0));
            }
        }
    };
}

describe('SceneTrees onTrianglesWritten', () => {
    for (const triangleLeaves of ['records', 'indexed'] as const) {
        it(`${triangleLeaves}: names each placement once, with the placed range, and nothing on a settled update`, async () => {
            const events: SceneTrianglesWrittenEvent[] = [];
            const { scene, frame } = await setUp({ triangleLeaves, onTrianglesWritten: (event) => events.push({ ...event, written: [...event.written] }) });
            const a = scene.addMesh(tessellatedQuad(4, 1));
            const b = scene.addMesh(tessellatedQuad(3, 1));
            scene.addInstance({ mesh: a, objectToWorld: IDENTITY, instanceIndex: 1 });
            scene.addInstance({ mesh: b, objectToWorld: MOVED_RIGHT, instanceIndex: 2 });
            scene.setHoldRefinements(true);
            // The first update only starts the builds: nothing written yet.
            frame();
            expect(events).toHaveLength(0);
            await scene.whenBuilt();
            frame();
            expect(events).toHaveLength(1);
            expect(events[0].triangleLeaves).toBe(triangleLeaves);
            expect(events[0].triangles).toBe(scene.buffers.triangles);
            const named = events[0].written.map((range) => range.mesh).sort();
            expect(named).toEqual([a, b].sort());
            for (const range of events[0].written) {
                const placement = scene.meshStatus(range.mesh!).placement!;
                expect(range).toMatchObject({ group: null, reason: 'placed', triangleOffset: placement.triangleOffset, triangleCount: placement.triangleCount });
            }
            // Settled: no event.
            frame();
            expect(events).toHaveLength(1);
            // tinybvh's refinement replaces the GPU trees: placed again, named again.
            await scene.whenRefined();
            scene.setHoldRefinements(false);
            frame();
            frame();
            const swaps = events.slice(1).flatMap((event) => event.written);
            expect(swaps.map((range) => range.mesh).sort()).toEqual([a, b].sort());
            for (const range of swaps) expect(scene.meshStatus(range.mesh!).builtBy).toBe('tinybvh');
            scene.dispose();
        });
    }

    it('lets the app keep flags of its own on a mesh across the swap to its refined tree', async () => {
        const flagged = new Set<number>();
        const pending: (() => void)[] = [];
        const { scene, frame } = await setUp({ onTrianglesWritten: flagWriter(flagged, pending) });
        const frameThenFlags = () => {
            frame();
            for (const write of pending.splice(0)) write();
        };
        const glass = scene.addMesh(tessellatedQuad(4, 1));
        const wall = scene.addMesh(tessellatedQuad(4, 1));
        flagged.add(glass);
        scene.addInstance({ mesh: glass, objectToWorld: IDENTITY, instanceIndex: 1 });
        // The wall sits under the glass: a ray through the glass hits it.
        scene.addInstance({ mesh: wall, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -1, 1], instanceIndex: 2 });
        const seeThrough = () => createCpuTracer(readScene(scene), { triangleFilter: () => true }).closest(down(0.1, 0.2))?.instanceIndex;
        scene.setHoldRefinements(true);
        frameThenFlags();
        await scene.whenBuilt();
        frameThenFlags();
        expect(scene.meshStatus(glass).builtBy).toBe('gpu');
        expect(seeThrough()).toBe(2);
        await scene.whenRefined();
        scene.setHoldRefinements(false);
        frameThenFlags();
        frameThenFlags();
        expect(scene.meshStatus(glass).builtBy).toBe('tinybvh');
        // The refined tree's records came without the app's bit; the hook put it back.
        expect(seeThrough()).toBe(2);
        scene.dispose();
    });

    it('names a records mesh whose CPU pose rewrote its records', async () => {
        const events: SceneTrianglesWrittenEvent[] = [];
        const { scene, frame } = await setUp({ onTrianglesWritten: (event) => events.push({ ...event, written: [...event.written] }) });
        const sheet = tessellatedQuad(4, 1);
        const cloth = scene.addMesh(sheet, { deforming: { bounds: [-5, -5, -5, 5, 5, 5] } });
        scene.addInstance({ mesh: cloth, objectToWorld: IDENTITY, instanceIndex: 3 });
        frame();
        await scene.whenRefined();
        frame();
        frame();
        events.length = 0;
        scene.setPoseCpu(cloth, lifted(sheet));
        frame();
        const placement = scene.meshStatus(cloth).placement!;
        expect(events.flatMap((event) => event.written)).toEqual([
            { mesh: cloth, group: null, reason: 'pose', triangleOffset: placement.triangleOffset, triangleCount: placement.triangleCount },
        ]);
        scene.dispose();
    });

    it('names merged groups by their group, with the mesh null', async () => {
        const events: SceneTrianglesWrittenEvent[] = [];
        const { scene, frame } = await setUp({ onTrianglesWritten: (event) => events.push({ ...event, written: [...event.written] }) });
        const a = scene.addMesh(tessellatedQuad(4, 1));
        const b = scene.addMesh(tessellatedQuad(3, 1));
        scene.addInstance({ mesh: a, objectToWorld: IDENTITY, instanceIndex: 1, static: true });
        scene.addInstance({ mesh: b, objectToWorld: MOVED_RIGHT, instanceIndex: 2, static: true });
        frame();
        await scene.whenBuilt();
        frame();
        expect(scene.counts.mergedGroups).toBe(1);
        const written = events.flatMap((event) => event.written);
        expect(written.length).toBeGreaterThan(0);
        expect(written.every((range) => range.mesh === null && range.group === 0 && range.reason === 'placed')).toBe(true);
        const group = scene.groupStatus(0).placement!;
        expect(written[0]).toMatchObject({ triangleOffset: group.triangleOffset, triangleCount: group.triangleCount });
        scene.dispose();
    });
});

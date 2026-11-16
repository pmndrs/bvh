import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
    assembleMovableScene,
    assembleTraceScene,
    CpuMeshBvh,
    createCpuTracer,
    createTinyBvhBuilder,
    deserializeLayout,
    deserializeMovableScene,
    deserializeTraceScene,
    GPU_LAYOUT_VERSION,
    IDENTITY_TRANSFORM,
    isSerializedCurrent,
    readSerializedHeader,
    SERIALIZED_FORMAT_VERSION,
    SERIALIZED_MOVABLE_SCENE_FORMAT_VERSION,
    SERIALIZED_SCENE_FORMAT_VERSION,
    serializeLayout,
    type TraceRay,
} from '@pmndrs/bvh';
import { bakeMesh, readModelMesh, readModelScene } from '../../examples/node/bake/bake.ts';
import { bakeMovableScene, bakeScene, DEFAULT_LEAF, defaultMesh, defaultScene } from '../../examples/node/bake/model.ts';

const FIXTURES = join(import.meta.dirname, '../../examples/node/bake/fixtures');

/** Rays from a sphere of radius `radius` towards points near the origin. */
function rays(count: number, radius: number): TraceRay[] {
    return Array.from({ length: count }, (_, i) => {
        const a = i * 2.399963;
        const z = ((i * 7) % 41) / 20 - 1;
        const s = Math.sqrt(1 - z * z);
        const origin: [number, number, number] = [Math.cos(a) * s * radius, z * radius, Math.sin(a) * s * radius];
        const aim = [Math.sin(i) * 0.9 * radius * 0.3, Math.cos(i * 1.7) * 0.9, Math.sin(i * 0.37) * 0.5 * radius * 0.3];
        const direction = [aim[0] - origin[0], aim[1] - origin[1], aim[2] - origin[2]];
        const length = Math.hypot(direction[0], direction[1], direction[2]);
        return { origin, direction: [direction[0] / length, direction[1] / length, direction[2] / length], tMin: 0, tMax: 1e30 };
    });
}

// The node/bake example (examples/node/bake): bake in Node, load with no build, and the loaded tree is the fresh one.

describe('node/bake', () => {
    it('a baked, reloaded tree has the fresh build\'s exact layout bytes', async () => {
        const mesh = defaultMesh();
        const tiny = await createTinyBvhBuilder();
        const baked = await bakeMesh(mesh, {}, tiny);
        const fresh = tiny.buildLayout(mesh, { maxLeafPrimitives: DEFAULT_LEAF }).layout;

        const loaded = deserializeLayout(baked.bytes);
        expect(new Uint8Array(loaded.nodes)).toEqual(new Uint8Array(fresh.nodes));
        expect(new Uint8Array(loaded.triangles)).toEqual(new Uint8Array(fresh.triangles));
        expect(Array.from(loaded.parents)).toEqual(Array.from(fresh.parents));
        expect(loaded.nodeCount).toBe(fresh.nodeCount);
        expect(loaded.maxDepth).toBe(fresh.maxDepth);
        // Baking twice gives the same file (a commit of the fixture doesn't churn).
        expect(new Uint8Array(serializeLayout(fresh, { geometry: mesh }))).toEqual(baked.bytes);
        expect((await bakeMesh(mesh, {}, tiny)).bytes).toEqual(baked.bytes);
    });

    it('raycasts through the loaded tree agree with a freshly built one', async () => {
        const mesh = defaultMesh();
        const tiny = await createTinyBvhBuilder();
        const baked = await bakeMesh(mesh, {}, tiny);
        const loaded = new CpuMeshBvh(deserializeLayout(baked.bytes));
        const fresh = CpuMeshBvh.build(mesh, tiny, { maxLeafPrimitives: DEFAULT_LEAF });

        let hits = 0;
        for (let i = 0; i < 400; i++) {
            // Rays from a sphere of radius 4 towards points near the knot.
            const a = i * 2.399963;
            const z = ((i * 7) % 41) / 20 - 1;
            const s = Math.sqrt(1 - z * z);
            const origin: [number, number, number] = [Math.cos(a) * s * 4, z * 4, Math.sin(a) * s * 4];
            const aim: [number, number, number] = [Math.sin(i) * 0.9, Math.cos(i * 1.7) * 0.9, Math.sin(i * 0.37) * 0.5];
            const direction: [number, number, number] = [aim[0] - origin[0], aim[1] - origin[1], aim[2] - origin[2]];
            const length = Math.hypot(...direction);
            const ray = { origin, direction: direction.map((x) => x / length) as [number, number, number] };
            const a1 = loaded.raycastFirst(ray);
            const b1 = fresh.raycastFirst(ray);
            expect(a1 === null).toBe(b1 === null);
            if (a1 && b1) {
                hits++;
                expect(a1.triangle).toBe(b1.triangle);
                expect(a1.distance).toBe(b1.distance);
            }
        }
        expect(hits).toBeGreaterThan(50);
    });

    it('bakes the other formats too, and every one reloads', async () => {
        const mesh = defaultMesh();
        const tiny = await createTinyBvhBuilder();
        for (const options of [{ nodeFormat: 'quantized' as const }, { maxLeafPrimitives: 2 }]) {
            const baked = await bakeMesh(mesh, options, tiny);
            const loaded = deserializeLayout(baked.bytes);
            expect(loaded.triangleCount).toBe(mesh.indices.length / 3);
            expect(loaded.nodeFormat).toBe(options.nodeFormat);
        }
    });

    it('the header says which versions made the file, and a bake from another version is refused', async () => {
        const baked = await bakeMesh(defaultMesh());
        expect(readSerializedHeader(baked.bytes)).toMatchObject({ kind: 'tree', formatVersion: SERIALIZED_FORMAT_VERSION, layoutVersion: GPU_LAYOUT_VERSION, current: true, triangleCount: 6400 });
        expect(isSerializedCurrent(baked.bytes)).toBe(true);

        const stale = baked.bytes.slice();
        new DataView(stale.buffer).setUint32(8, GPU_LAYOUT_VERSION + 1, true);
        expect(readSerializedHeader(stale)).toMatchObject({ current: false, reason: expect.stringMatching(/GPU layout version/) });
        expect(isSerializedCurrent(stale)).toBe(false);
        expect(() => deserializeLayout(stale)).toThrow(/GPU layout version/);
        expect(() => readSerializedHeader(new Uint8Array(64))).toThrow(/not a serialized/);
    });

    it('an indexed bake loads next to the geometry it was baked from, and is refused next to other geometry', async () => {
        const mesh = defaultMesh();
        const baked = await bakeMesh(mesh, { triangleLeaves: 'indexed' });
        expect(readSerializedHeader(baked.bytes)).toMatchObject({ kind: 'tree', triangleLeaves: 'indexed', geometryHash: expect.any(Array) });
        const layout = deserializeLayout(baked.bytes, { geometry: mesh });
        // As the page does: one instance, the geometry beside the tree.
        const indexed = createCpuTracer(assembleTraceScene([layout], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY_TRANSFORM }], {}, { geometries: [mesh] }));
        const records = new CpuMeshBvh(deserializeLayout((await bakeMesh(mesh)).bytes));
        let hits = 0;
        for (const ray of rays(300, 4)) {
            const a = indexed.closest(ray);
            const b = records.raycastFirst(ray);
            expect(a === null).toBe(b === null);
            if (a && b) {
                hits++;
                expect(a.triangleIndex).toBe(b.triangle);
            }
        }
        expect(hits).toBeGreaterThan(50);
        const other = { positions: mesh.positions.map((value) => value * 1.01), indices: mesh.indices };
        expect(() => deserializeLayout(baked.bytes, { geometry: other })).toThrow(/baked from other geometry/);
    });

    it('--scene: the default scene round-trips, traces as the assembled one does, and the committed file is current and identical to a fresh bake', async () => {
        const tiny = await createTinyBvhBuilder();
        for (const options of [{}, { triangleLeaves: 'indexed' as const, nodeFormat: 'f32' as const }, { mergeStaticInstances: false }]) {
            const baked = await bakeScene(defaultScene(), options, tiny);
            expect(readSerializedHeader(baked.bytes)).toMatchObject({ kind: 'scene', formatVersion: SERIALIZED_SCENE_FORMAT_VERSION, current: true, meshCount: 4 });
            const loaded = deserializeTraceScene(baked.bytes);
            expect(loaded.mergedGroups.length).toBe(options.mergeStaticInstances === false ? 0 : 1);
            const a = createCpuTracer(loaded);
            const b = createCpuTracer(baked.data);
            const materials = new Set<number>();
            for (const ray of rays(400, 9)) {
                const hit = a.closest(ray);
                expect(hit).toEqual(b.closest(ray));
                if (hit) materials.add(hit.material);
            }
            expect(materials.size).toBeGreaterThan(4); // instance materials and the box's own
        }
        const committed = await readFile(join(FIXTURES, 'knots.scene.bvh'));
        expect(isSerializedCurrent(committed)).toBe(true);
        expect(new Uint8Array(committed)).toEqual((await bakeScene(defaultScene(), {}, tiny)).bytes);
        const tree = await readFile(join(FIXTURES, 'torus-knot.bvh'));
        expect(isSerializedCurrent(tree)).toBe(true);
        expect(new Uint8Array(tree)).toEqual((await bakeMesh(defaultMesh(), {}, tiny)).bytes);
    });

    it('--scene --movable: the default scene\'s knots can move, the rest merges, and the file loads with its own geometry', async () => {
        const tiny = await createTinyBvhBuilder();
        for (const options of [{}, { triangleLeaves: 'indexed' as const, nodeFormat: 'f32' as const }, { mergeStaticInstances: false }]) {
            const baked = await bakeMovableScene(defaultScene(), options, tiny);
            const merge = options.mergeStaticInstances !== false;
            expect(readSerializedHeader(baked.bytes)).toMatchObject({
                kind: 'movable-scene',
                formatVersion: SERIALIZED_MOVABLE_SCENE_FORMAT_VERSION,
                current: true,
                meshCount: 4,
                instanceCount: 6,
                treeCount: merge ? 1 : 4, // the knot's tree, the others merged away
                mergedGroupCount: merge ? 1 : 0,
                geometry: true,
            });
            const loaded = deserializeMovableScene(baked.bytes);
            expect(loaded.instances.filter((instance) => !instance.static).map((instance) => instance.meshIndex)).toEqual(merge ? [0, 0, 0] : [0, 0, 0, 1, 2, 3]);
            // The scene it describes traces as the snapshot of the same scene does.
            const a = createCpuTracer(assembleMovableScene(loaded));
            const b = createCpuTracer((await bakeScene(defaultScene(), options, tiny)).data);
            for (const ray of rays(400, 9)) expect(a.closest(ray)).toEqual(b.closest(ray));
        }
    });

    it('reads a glTF file into the mesh to bake (an embedded buffer, a node transform applied)', async () => {
        // One triangle, translated by +2 in x by its node.
        const buffer = new Uint8Array(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]).buffer);
        const gltf = {
            asset: { version: '2.0' },
            buffers: [{ byteLength: buffer.byteLength, uri: `data:application/octet-stream;base64,${Buffer.from(buffer).toString('base64')}` }],
            bufferViews: [{ buffer: 0, byteLength: buffer.byteLength }],
            accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
            meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
            nodes: [{ mesh: 0, translation: [2, 0, 0] }],
            scenes: [{ nodes: [0] }],
        };
        const directory = await mkdtemp(join(tmpdir(), 'bvh-bake-'));
        try {
            const path = join(directory, 'triangle.gltf');
            await writeFile(path, JSON.stringify(gltf));
            const mesh = await readModelMesh(path);
            expect(Array.from(mesh.positions.slice(0, 3))).toEqual([2, 0, 0]);
            const baked = await bakeMesh(mesh);
            expect(deserializeLayout(baked.bytes).triangleCount).toBe(1);
            // As a scene: the node is an instance, its translation in the top level.
            const scene = await readModelScene(path);
            expect(scene.instances.length).toBe(1);
            const loaded = deserializeTraceScene((await bakeScene(scene)).bytes);
            expect(createCpuTracer(loaded).closest({ origin: [2.2, 0.2, 1], direction: [0, 0, -1], tMin: 0, tMax: 10 })?.instanceIndex).toBe(0);
        } finally {
            await rm(directory, { recursive: true, force: true });
        }
    });
});

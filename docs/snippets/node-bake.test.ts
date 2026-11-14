import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { CpuMeshBvh, createTinyBvhBuilder, deserializeLayout, GPU_LAYOUT_VERSION, SERIALIZED_FORMAT_VERSION, serializeLayout } from '@pmndrs/bvh';
import { bakeMesh, readBakedHeader, readModelMesh } from '../../examples/node/bake/bake.ts';
import { DEFAULT_LEAF, defaultMesh } from '../../examples/node/bake/model.ts';

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
        expect(new Uint8Array(serializeLayout(fresh))).toEqual(baked.bytes);
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
        expect(readBakedHeader(baked.bytes)).toMatchObject({ formatVersion: SERIALIZED_FORMAT_VERSION, layoutVersion: GPU_LAYOUT_VERSION, current: true, triangleCount: 6400 });

        const stale = baked.bytes.slice();
        new DataView(stale.buffer).setUint32(8, GPU_LAYOUT_VERSION + 1, true);
        expect(readBakedHeader(stale).current).toBe(false);
        expect(() => deserializeLayout(stale)).toThrow(/GPU layout version/);
        expect(() => readBakedHeader(new Uint8Array(64))).toThrow(/not a baked/);
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
        } finally {
            await rm(directory, { recursive: true, force: true });
        }
    });
});

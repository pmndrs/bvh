/// <reference types="node" />
/**
 * tinybvh's indexed triangles and quantized nodes written inside the WASM
 * (`tbvh_write_layout_as`) against the WASM's records and `f32` nodes
 * converted in JavaScript (the path before, `formatsInWasm: false`), on real
 * scenes, in Node with no GPU:
 *
 *   BVH_ASSETS_DIR=... BVH_MANIFESTS_DIR=... node --experimental-transform-types tools/tinybvh-formats.ts sponza
 *   node --experimental-transform-types tools/tinybvh-formats.ts sponza --merged --rounds 3
 *
 * Per scene (every mesh, or with `--merged` every mesh's triangles as one
 * mesh, the merged group's size): `buildLayout` time with
 * `triangleLeaves: 'indexed', nodeFormat: 'quantized'` both ways, alternating
 * which goes first each round (median of rounds), the WASM memory each way
 * leaves grown (one builder per path, never released, so its memory is the
 * peak of every build), and a byte comparison of every tree (exits non-zero
 * if any differs).
 */

import { createTinyBvhBuilder, type TinyBvhBuilder } from '../src/tinybvh.ts';
import type { GpuLayoutMeshBvh } from '../src/gpu-layout.ts';
import type { MeshInput } from '../src/mesh-input.ts';
import { readSceneMeshes } from './scene-meshes.ts';

const args = process.argv.slice(2);
const flag = (name: string): string | undefined => {
    const at = args.indexOf(name);
    return at >= 0 ? args[at + 1] : undefined;
};
const rounds = Number(flag('--rounds') ?? 3);
const merged = args.includes('--merged');
const sceneIds = args.filter((arg, index) => !arg.startsWith('--') && !args[index - 1]?.startsWith('--'));
if (sceneIds.length === 0) sceneIds.push('sponza');
const OPTIONS = { triangleLeaves: 'indexed', nodeFormat: 'quantized' } as const;
const MIB = 2 ** 20;

function sameBytes(a: GpuLayoutMeshBvh, b: GpuLayoutMeshBvh): boolean {
    return Buffer.from(a.nodes).equals(Buffer.from(b.nodes)) && Buffer.from(a.triangles).equals(Buffer.from(b.triangles)) && Buffer.from(a.parents.buffer, a.parents.byteOffset, a.parents.byteLength).equals(Buffer.from(b.parents.buffer, b.parents.byteOffset, b.parents.byteLength));
}

/** Every mesh's triangles as one indexed mesh. */
function concatenate(meshes: readonly MeshInput[]): MeshInput {
    const vertexCount = meshes.reduce((sum, mesh) => sum + mesh.positions.length / 3, 0);
    const indexCount = meshes.reduce((sum, mesh) => sum + (mesh.indices ? mesh.indices.length : mesh.positions.length / 3), 0);
    const positions = new Float32Array(vertexCount * 3);
    const indices = new Uint32Array(indexCount);
    let vertexBase = 0;
    let indexAt = 0;
    for (const mesh of meshes) {
        positions.set(mesh.positions, vertexBase * 3);
        const count = mesh.positions.length / 3;
        if (mesh.indices) for (const index of mesh.indices) indices[indexAt++] = vertexBase + index;
        else for (let vertex = 0; vertex < count; vertex++) indices[indexAt++] = vertexBase + vertex;
        vertexBase += count;
    }
    return { positions, indices };
}

const median = (values: number[]): number => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

for (const sceneId of sceneIds) {
    const sceneMeshes = readSceneMeshes(sceneId).map(({ geometry }) => ({ positions: geometry.positions, indices: geometry.indices }) as MeshInput);
    const meshes = merged ? [concatenate(sceneMeshes)] : sceneMeshes;
    const triangles = meshes.reduce((sum, mesh) => sum + (mesh.indices ? mesh.indices.length : mesh.positions.length / 3) / 3, 0);
    const inWasm = await createTinyBvhBuilder({ releaseAboveBytes: Infinity });
    const inJavaScript = await createTinyBvhBuilder({ releaseAboveBytes: Infinity, formatsInWasm: false });
    if (!inWasm.formatsInWasm) throw new Error('this .wasm has no tbvh_write_layout_as');
    const time = (builder: TinyBvhBuilder): { ms: number; layouts: GpuLayoutMeshBvh[] } => {
        const start = performance.now();
        const layouts = meshes.map((mesh) => builder.buildLayout(mesh, OPTIONS).layout);
        return { ms: performance.now() - start, layouts };
    };
    const wasmMs: number[] = [];
    const javaScriptMs: number[] = [];
    let differences = 0;
    for (let round = 0; round < rounds; round++) {
        const order = round % 2 === 0 ? [inWasm, inJavaScript] : [inJavaScript, inWasm];
        const results = order.map(time);
        const [a, b] = round % 2 === 0 ? results : [results[1], results[0]];
        wasmMs.push(a.ms);
        javaScriptMs.push(b.ms);
        if (round === 0) a.layouts.forEach((layout, index) => (differences += sameBytes(layout, b.layouts[index]) ? 0 : 1));
    }
    const label = merged ? `${sceneId} (one mesh)` : sceneId;
    console.log(`${label}: ${meshes.length} meshes, ${triangles} triangles, indexed + quantized, ${rounds} rounds`);
    console.log(`  in the WASM:       median ${median(wasmMs).toFixed(0)} ms (${wasmMs.map((ms) => ms.toFixed(0)).join(', ')}), WASM memory ${(inWasm.heapBytes / MIB).toFixed(1)} MiB`);
    console.log(`  converted in JS:   median ${median(javaScriptMs).toFixed(0)} ms (${javaScriptMs.map((ms) => ms.toFixed(0)).join(', ')}), WASM memory ${(inJavaScript.heapBytes / MIB).toFixed(1)} MiB`);
    console.log(`  WASM over JS time: ${(median(wasmMs) / median(javaScriptMs)).toFixed(3)}; trees that differ: ${differences}`);
    if (differences > 0) process.exitCode = 1;
}

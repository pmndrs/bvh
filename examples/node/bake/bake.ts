/**
 * Baking a tree in Node: the part of `node/bake` the CLI (cli.ts) and the unit
 * test share. No GPU and no browser: tinybvh's WebAssembly builds the tree,
 * `serializeLayout` writes it.
 *
 * The model is one mesh: a glTF's meshes merged into one world-space mesh (a
 * bake is a tree, and a tree is one mesh), or the default torus knot.
 */

import { readFile } from 'node:fs/promises';
import { basename, dirname, extname, resolve } from 'node:path';
import { createTinyBvhBuilder, GPU_LAYOUT_VERSION, SERIALIZED_FORMAT_VERSION, serializeLayout, type GpuLayoutMeshBvh, type MeshInput, type TinyBvhBuilder, type TinyBvhLayoutOptions } from '@pmndrs/bvh';
import { mergeToWorld, type IndexedMesh } from '../../shared/vanilla/geometry.ts';
import { DEFAULT_LEAF } from './model.ts';

export interface BakeOptions {
    /** Leaves hold at most this many triangles, 1 to 8. Default 8. */
    readonly maxLeafPrimitives?: number;
    /** `'indexed'` bakes 16-byte indexed triangles: the page needs the same geometry next to the file. Default `'records'`. */
    readonly triangleLeaves?: 'records' | 'indexed';
    /** `'quantized'` bakes 32-byte nodes. Default `'f32'`. */
    readonly nodeFormat?: 'f32' | 'quantized';
}

export interface Baked {
    /** The file's bytes: what `serializeLayout` wrote. */
    readonly bytes: Uint8Array;
    readonly layout: GpuLayoutMeshBvh;
    readonly triangles: number;
    /** tinybvh's build, in ms (what the browser doesn't pay). */
    readonly buildMs: number;
}

/** Builds `mesh`'s tree with tinybvh and serializes it. Pass a `builder` to reuse one WASM instance across meshes. */
export async function bakeMesh(mesh: MeshInput, options: BakeOptions = {}, builder?: TinyBvhBuilder): Promise<Baked> {
    const tiny = builder ?? (await createTinyBvhBuilder());
    const layoutOptions: TinyBvhLayoutOptions = { maxLeafPrimitives: options.maxLeafPrimitives ?? DEFAULT_LEAF, triangleLeaves: options.triangleLeaves, nodeFormat: options.nodeFormat };
    const { layout, buildMs } = tiny.buildLayout(mesh, layoutOptions);
    return { bytes: new Uint8Array(serializeLayout(layout)), layout, triangles: layout.triangleCount, buildMs };
}

/** The mesh of a `.glb` or `.gltf` file: every primitive in world space, merged into one. */
export async function readModelMesh(path: string): Promise<IndexedMesh> {
    // The glTF reader is the examples' own (a browser module): it reads `location` when it is imported, for `?model=`. Node has none.
    (globalThis as { location?: unknown }).location ??= new URL('file:///');
    const { parseGltf } = await import('../../shared/vanilla/gltf.ts');
    const file = resolve(path);
    const bytes = await readFile(file);
    const scene = await parseGltf(
        bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
        async (uri) => {
            const external = await readFile(resolve(dirname(file), uri));
            return external.buffer.slice(external.byteOffset, external.byteOffset + external.byteLength) as ArrayBuffer;
        },
        basename(file),
    );
    return mergeToWorld(scene);
}

/** What a baked file's header says, for a CLI line or a version check before loading. */
export interface BakedHeader {
    readonly formatVersion: number;
    readonly layoutVersion: number;
    /** True when both versions are the ones this package reads (a bake from another version is refused by `deserializeLayout`). */
    readonly current: boolean;
    readonly nodeCount: number;
    readonly triangleCount: number;
}

/** Reads the header's words without loading the tree. Throws when the bytes are not a baked tree. */
export function readBakedHeader(bytes: ArrayBuffer | ArrayBufferView): BakedHeader {
    const view = ArrayBuffer.isView(bytes) ? new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength) : new DataView(bytes);
    if (view.byteLength < 64 || view.getUint32(0, true) !== 0x48564250) throw new Error('not a baked @pmndrs/bvh tree (bad magic or shorter than the 64-byte header)');
    const formatVersion = view.getUint32(4, true);
    const layoutVersion = view.getUint32(8, true);
    return { formatVersion, layoutVersion, current: formatVersion === SERIALIZED_FORMAT_VERSION && layoutVersion === GPU_LAYOUT_VERSION, nodeCount: view.getUint32(12, true), triangleCount: view.getUint32(16, true) };
}

/** A default file name: the model's name, or `torus-knot`, with `.bvh`. */
export function defaultOutputName(model: string | null): string {
    return `${model ? basename(model, extname(model)) : 'torus-knot'}.bvh`;
}

/**
 * Baking a tree in Node: the part of `node/bake` the CLI (cli.ts) and the unit
 * test share. No GPU and no browser: tinybvh's WebAssembly builds the tree,
 * `serializeLayout` writes it (with the mesh's geometry hash, so loading it
 * next to other geometry is refused).
 *
 * The model is one mesh: a glTF's meshes merged into one world-space mesh (a
 * bake is a tree, and a tree is one mesh), or the default torus knot. A whole
 * scene (`--scene`, model.ts's `bakeScene`) keeps a glTF's meshes and nodes.
 */

import { readFile } from 'node:fs/promises';
import { basename, dirname, extname, resolve } from 'node:path';
import { createTinyBvhBuilder, serializeLayout, type GpuLayoutMeshBvh, type MeshInput, type TinyBvhBuilder, type TinyBvhLayoutOptions } from '@pmndrs/bvh';
import { mergeToWorld, type IndexedMesh, type VanillaScene } from '../../shared/vanilla/geometry.ts';
import { DEFAULT_LEAF } from './model.ts';

export interface BakeOptions {
    /** Leaves hold at most this many triangles, 1 to 8. Default 8. */
    readonly maxLeafPrimitives?: number;
    /** `'indexed'` bakes 16-byte indexed triangles: load the file next to the same geometry (the page: the default knot, or `?model=`). Default `'records'`. */
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

/** Builds `mesh`'s tree with tinybvh and serializes it with the mesh's geometry hash. Pass a `builder` to reuse one WASM instance across meshes. */
export async function bakeMesh(mesh: MeshInput, options: BakeOptions = {}, builder?: TinyBvhBuilder): Promise<Baked> {
    const tiny = builder ?? (await createTinyBvhBuilder());
    const layoutOptions: TinyBvhLayoutOptions = { maxLeafPrimitives: options.maxLeafPrimitives ?? DEFAULT_LEAF, triangleLeaves: options.triangleLeaves, nodeFormat: options.nodeFormat };
    const { layout, buildMs } = tiny.buildLayout(mesh, layoutOptions);
    return { bytes: new Uint8Array(serializeLayout(layout, { geometry: mesh })), layout, triangles: layout.triangleCount, buildMs };
}

/** The mesh of a `.glb` or `.gltf` file: every primitive in world space, merged into one. */
export async function readModelMesh(path: string): Promise<IndexedMesh> {
    return mergeToWorld(await readModelScene(path));
}

/** A `.glb` or `.gltf` file's meshes (one per primitive) and instances (one per node using it), for a scene bake. */
export async function readModelScene(path: string): Promise<VanillaScene> {
    // The glTF reader is the examples' own (a browser module): it reads `location` when it is imported, for `?model=`. Node has none.
    (globalThis as { location?: unknown }).location ??= new URL('file:///');
    const { parseGltf } = await import('../../shared/vanilla/gltf.ts');
    const file = resolve(path);
    const bytes = await readFile(file);
    return parseGltf(
        bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
        async (uri) => {
            const external = await readFile(resolve(dirname(file), uri));
            return external.buffer.slice(external.byteOffset, external.byteOffset + external.byteLength) as ArrayBuffer;
        },
        basename(file),
    );
}

/** A default file name: the model's name, or `torus-knot` (`knots` for the default scene), with `.bvh` (`.scene.bvh` for a scene, `.movable.bvh` for a movable one). */
export function defaultOutputName(model: string | null, scene = false, movable = false): string {
    return `${model ? basename(model, extname(model)) : scene ? 'knots' : 'torus-knot'}${scene ? (movable ? '.movable' : '.scene') : ''}.bvh`;
}

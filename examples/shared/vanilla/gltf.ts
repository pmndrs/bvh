/**
 * A minimal glTF 2.0 reader for the vanilla pages: positions and indices of
 * every triangle primitive, and the node hierarchy's transforms, which is all
 * a BVH needs. No engine, no textures, no materials beyond their index (the
 * hits' material number).
 *
 * - `.glb`, or `.gltf` with external `.bin` files or `data:` URIs.
 * - Each primitive is one mesh; each node that uses it is one instance.
 * - Not read: sparse accessors, quantized positions (`KHR_mesh_quantization`),
 *   Draco and meshopt compression (a clear error says so), morph targets and
 *   skins (the bind pose is used).
 *
 * `?model=<path>` loads one from the assets base (assets.ts); a `.glb`, or a
 * `.gltf` with its files, can be dropped on the page.
 */

import { assetUrl } from '../assets.ts';
import { makeScene, type IndexedMesh, type VanillaInstance, type VanillaScene } from './geometry.ts';
import { mat4 } from './math.ts';

interface GltfJson {
    readonly asset?: { readonly version?: string };
    readonly extensionsRequired?: readonly string[];
    readonly buffers?: readonly { readonly uri?: string; readonly byteLength: number }[];
    readonly bufferViews?: readonly { readonly buffer: number; readonly byteOffset?: number; readonly byteLength: number; readonly byteStride?: number }[];
    readonly accessors?: readonly {
        readonly bufferView?: number;
        readonly byteOffset?: number;
        readonly componentType: number;
        readonly count: number;
        readonly type: string;
        readonly normalized?: boolean;
        readonly sparse?: unknown;
    }[];
    readonly meshes?: readonly { readonly primitives: readonly { readonly attributes: Record<string, number>; readonly indices?: number; readonly mode?: number; readonly material?: number }[] }[];
    readonly nodes?: readonly {
        readonly children?: readonly number[];
        readonly mesh?: number;
        readonly matrix?: readonly number[];
        readonly translation?: readonly number[];
        readonly rotation?: readonly number[];
        readonly scale?: readonly number[];
    }[];
    readonly scenes?: readonly { readonly nodes?: readonly number[] }[];
    readonly scene?: number;
}

const COMPONENTS: Readonly<Record<string, number>> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
const UNSUPPORTED = ['KHR_draco_mesh_compression', 'EXT_meshopt_compression', 'KHR_meshopt_compression', 'KHR_mesh_quantization'];

/** Loads a glTF from a `?model=` path or URL. */
export async function loadGltfScene(path: string): Promise<VanillaScene> {
    const url = new URL(assetUrl(path), location.href);
    const start = performance.now();
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    const bytes = await response.arrayBuffer();
    const fetchRelative = async (uri: string) => {
        const relative = await fetch(new URL(uri, url));
        if (!relative.ok) throw new Error(`${uri}: HTTP ${relative.status}`);
        return relative.arrayBuffer();
    };
    return parseGltf(bytes, fetchRelative, path.split('/').pop() ?? path, start);
}

/** Parses a `.glb` or `.gltf` (`bytes`); `resolve` fetches its external buffers by URI. */
export async function parseGltf(bytes: ArrayBuffer, resolve: (uri: string) => Promise<ArrayBuffer>, name: string, start = performance.now()): Promise<VanillaScene> {
    let json: GltfJson;
    let binChunk: ArrayBuffer | null = null;
    const header = new DataView(bytes);
    if (bytes.byteLength >= 12 && header.getUint32(0, true) === 0x46546c67) {
        // GLB: a 12-byte header, then chunks (JSON first, BIN optional).
        let at = 12;
        let text = '';
        while (at < bytes.byteLength) {
            const length = header.getUint32(at, true);
            const type = header.getUint32(at + 4, true);
            if (type === 0x4e4f534a) text = new TextDecoder().decode(new Uint8Array(bytes, at + 8, length));
            else if (type === 0x004e4942) binChunk = bytes.slice(at + 8, at + 8 + length);
            at += 8 + length;
        }
        json = JSON.parse(text) as GltfJson;
    } else {
        json = JSON.parse(new TextDecoder().decode(bytes)) as GltfJson;
    }
    const unsupported = (json.extensionsRequired ?? []).filter((extension) => UNSUPPORTED.includes(extension));
    if (unsupported.length) throw new Error(`${name} needs ${unsupported.join(', ')}, which this minimal reader doesn't decode`);

    const buffers = await Promise.all(
        (json.buffers ?? []).map(async (buffer) => {
            if (buffer.uri === undefined) {
                if (!binChunk) throw new Error(`${name}: a buffer without a URI and no GLB binary chunk`);
                return binChunk;
            }
            if (buffer.uri.startsWith('data:')) return (await fetch(buffer.uri)).arrayBuffer();
            return resolve(decodeURIComponent(buffer.uri));
        }),
    );

    /** An accessor's values as numbers, copied out of the buffer (so a worker may take them). */
    const read = (index: number, kind: 'positions' | 'indices'): Float32Array | Uint32Array => {
        const accessor = json.accessors![index];
        if (accessor.sparse) throw new Error(`${name}: sparse accessors aren't read`);
        const components = COMPONENTS[accessor.type];
        if (kind === 'positions' && (accessor.componentType !== 5126 || components !== 3)) throw new Error(`${name}: positions must be float VEC3 (quantized positions aren't read)`);
        const out = kind === 'positions' ? new Float32Array(accessor.count * 3) : new Uint32Array(accessor.count);
        if (accessor.bufferView === undefined) return out; // all zeros, as the spec says
        const view = json.bufferViews![accessor.bufferView];
        const data = new DataView(buffers[view.buffer], (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0));
        const size = accessor.componentType === 5126 || accessor.componentType === 5125 ? 4 : accessor.componentType === 5123 ? 2 : 1;
        const stride = view.byteStride ?? size * components;
        for (let i = 0; i < accessor.count; i++) {
            for (let c = 0; c < components; c++) {
                const at = i * stride + c * size;
                out[i * components + c] =
                    accessor.componentType === 5126 ? data.getFloat32(at, true) : accessor.componentType === 5125 ? data.getUint32(at, true) : accessor.componentType === 5123 ? data.getUint16(at, true) : data.getUint8(at);
            }
        }
        return out;
    };

    // One mesh per triangle primitive.
    const meshes: IndexedMesh[] = [];
    const primitivesOf: { mesh: number; material: number }[][] = [];
    for (const gltfMesh of json.meshes ?? []) {
        const list: { mesh: number; material: number }[] = [];
        for (const primitive of gltfMesh.primitives) {
            if ((primitive.mode ?? 4) !== 4 || primitive.attributes.POSITION === undefined) continue;
            const positions = read(primitive.attributes.POSITION, 'positions') as Float32Array;
            const indices = primitive.indices !== undefined ? (read(primitive.indices, 'indices') as Uint32Array) : Uint32Array.from({ length: positions.length / 3 }, (_, i) => i);
            if (indices.length < 3) continue;
            list.push({ mesh: meshes.length, material: primitive.material ?? 0 });
            meshes.push({ positions, indices: indices.length % 3 ? indices.slice(0, indices.length - (indices.length % 3)) : indices });
        }
        primitivesOf.push(list);
    }

    // One instance per node and primitive, in world space.
    const instances: VanillaInstance[] = [];
    const visit = (nodeIndex: number, parent: Float32Array) => {
        const node = json.nodes![nodeIndex];
        const local = node.matrix ? new Float32Array(node.matrix) : mat4.compose(node.translation ?? [0, 0, 0], node.rotation ?? [0, 0, 0, 1], node.scale ?? [1, 1, 1]);
        const world = mat4.multiply(parent, local);
        if (node.mesh !== undefined) for (const primitive of primitivesOf[node.mesh]) instances.push({ mesh: primitive.mesh, objectToWorld: world, material: primitive.material });
        for (const child of node.children ?? []) visit(child, world);
    };
    const roots = json.scenes?.[json.scene ?? 0]?.nodes ?? (json.nodes ?? []).map((_, index) => index);
    for (const root of roots) visit(root, mat4.identity());
    if (instances.length === 0) throw new Error(`${name} has no triangle meshes`);
    return makeScene(name, meshes, instances, performance.now() - start);
}

/** Calls `onScene` with each glTF dropped on the page (a `.gltf` finds its `.bin` among the files dropped with it). */
export function enableSceneDrop(onScene: (scene: VanillaScene) => void | Promise<void>, onError: (error: unknown) => void): void {
    const hint = document.createElement('div');
    hint.className = 'bvh-drop';
    hint.textContent = 'Drop a .glb (or a .gltf with its .bin)';
    document.body.append(hint);
    addEventListener('dragover', (event) => {
        event.preventDefault();
        hint.style.display = 'grid';
    });
    addEventListener('dragleave', (event) => {
        if (event.relatedTarget === null) hint.style.display = 'none';
    });
    addEventListener('drop', (event) => {
        event.preventDefault();
        hint.style.display = 'none';
        const files = [...(event.dataTransfer?.files ?? [])];
        const main = files.find((file) => /\.(glb|gltf)$/i.test(file.name));
        if (!main) {
            onError(new Error('drop a .glb or .gltf file'));
            return;
        }
        const byName = new Map(files.map((file) => [file.name, file]));
        const resolve = async (uri: string) => {
            const file = byName.get(uri.split('/').pop() ?? uri);
            if (!file) throw new Error(`${uri} wasn't dropped with ${main.name}`);
            return file.arrayBuffer();
        };
        main.arrayBuffer()
            .then((bytes) => parseGltf(bytes, resolve, main.name))
            .then(onScene)
            .catch(onError);
    });
}

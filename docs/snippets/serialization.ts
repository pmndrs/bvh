// Code samples for docs/serialization/*.mdx (see getting-started.ts for how they are checked).

// #region bake
// bake.ts: node bake.ts  (Node 22+; tinybvh's WebAssembly needs no GPU)
import { writeFile } from 'node:fs/promises';
import { createTinyBvhBuilder, serializeLayout, type MeshInput } from '@pmndrs/bvh';

// #endregion

export async function bake(meshes: readonly MeshInput[], path: (index: number) => string): Promise<void> {
    // #region bake
    const tiny = await createTinyBvhBuilder();
    for (const [index, mesh] of meshes.entries()) {
        const { layout } = tiny.buildLayout(mesh, { maxLeafPrimitives: 8 });
        // One ArrayBuffer: a 64-byte header, then nodes, triangles and parents as the GPU reads them.
        await writeFile(path(index), new Uint8Array(serializeLayout(layout)));
    }
    // #endregion
}

// #region load
import { assembleTraceScene, deserializeLayout, uploadTraceScene, type TraceSceneInstance } from '@pmndrs/bvh';

// #endregion

export async function load(device: GPUDevice, urls: readonly string[], instances: readonly TraceSceneInstance[]): Promise<void> {
    // #region load
    // deserializeLayout refuses bytes from another GPU_LAYOUT_VERSION (or format version): rebuild those.
    const trees = await Promise.all(urls.map(async (url) => deserializeLayout(await (await fetch(url)).arrayBuffer())));
    const scene = assembleTraceScene(trees, instances); // instances name their tree by meshIndex
    const buffers = uploadTraceScene(device, scene); // nodes, triangles, topLevel: bind them to the walk
    // #endregion
    buffers.destroy();
}

// #region cpu-tree
import { CpuMeshBvh } from '@pmndrs/bvh';

// #endregion

export async function cpuTree(input: MeshInput): Promise<CpuMeshBvh<unknown>> {
    // #region cpu-tree
    const tree = await CpuMeshBvh.create(input);
    const bytes = tree.serialize();
    // The layout is checked when it is loaded (children in range, every node reached once),
    // so corrupt bytes throw instead of hanging a walk.
    const loaded = CpuMeshBvh.deserialize(bytes, { source: input });
    // #endregion
    return loaded;
}

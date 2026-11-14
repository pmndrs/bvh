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
        // With the geometry, the header keeps its hash, and loading the tree next to other geometry is refused.
        await writeFile(path(index), new Uint8Array(serializeLayout(layout, { geometry: mesh })));
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

// #region bake-scene
import { buildTraceScene, serializeTraceScene } from '@pmndrs/bvh';

// #endregion

export async function bakeScene(meshes: readonly MeshInput[], instances: readonly TraceSceneInstance[], path: string): Promise<void> {
    // #region bake-scene
    const tiny = await createTinyBvhBuilder();
    // Every tree, merged static instances, the top level, instance and triangle materials, the formats.
    const scene = await buildTraceScene(meshes, instances, { build: (mesh) => tiny.buildLayout(mesh, { maxLeafPrimitives: 8 }).layout });
    await writeFile(path, new Uint8Array(serializeTraceScene(scene)));
    // #endregion
}

// #region load-scene
import { deserializeTraceScene } from '@pmndrs/bvh';

// #endregion

export async function loadScene(device: GPUDevice, url: string): Promise<void> {
    // #region load-scene
    // No build and no assembly: the buffers go to the GPU as they were baked.
    const scene = deserializeTraceScene(await (await fetch(url)).arrayBuffer());
    const buffers = uploadTraceScene(device, scene); // compile the walk for scene.nodeFormat and scene.triangleLeaves
    // #endregion
    buffers.destroy();
}

// #region baked-scene-trees
import { SceneTrees, type MeshGeometry } from '@pmndrs/bvh';

// #endregion

export async function bakedSceneTrees(device: GPUDevice, geometries: readonly MeshGeometry[], urls: readonly string[], matrix: Float32Array): Promise<SceneTrees> {
    // #region baked-scene-trees
    const trees = await SceneTrees.create(device);
    for (const [index, geometry] of geometries.entries()) {
        const mesh = trees.addMesh(geometry);
        const bytes = await (await fetch(urls[index])).arrayBuffer();
        // Checked against the geometry (its hash, when the bake has one); placed before the first update, so no build starts.
        trees.placeTree(mesh, deserializeLayout(bytes, { geometry }));
        trees.addInstance({ mesh, objectToWorld: matrix });
    }
    // The next update places every tree: trees.counts.onAppTrees === geometries.length.
    // #endregion
    return trees;
}

export async function loadIndexed(device: GPUDevice, url: string, geometry: MeshGeometry, instances: readonly TraceSceneInstance[]): Promise<CpuMeshBvh<unknown>> {
    // #region indexed
    // An indexed tree holds vertex indices, not positions: it only works next to the geometry it was baked from.
    const tree = deserializeLayout(await (await fetch(url)).arrayBuffer(), { geometry });
    const scene = assembleTraceScene([tree], instances, {}, { geometries: [geometry] }); // the scene copies the positions
    const buffers = uploadTraceScene(device, scene);
    const cpu = new CpuMeshBvh(tree, { source: geometry }); // the CPU walk reads them from the geometry
    // SceneTrees takes it too: SceneTrees.create(device, { triangleLeaves: 'indexed', weld: false }), then placeTree.
    // #endregion
    buffers.destroy();
    return cpu;
}

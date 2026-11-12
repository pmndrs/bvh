// Code samples for docs/builders/*.mdx and docs/concepts/*.mdx (see getting-started.ts for how they are checked).

// #region chooser
import { createMeshLayoutBuilder, type MeshInput } from '@pmndrs/bvh';

// #endregion

export async function chooser(device: GPUDevice | null, meshes: readonly MeshInput[]): Promise<void> {
    // #region chooser
    const builder = await createMeshLayoutBuilder(device); // compiles the GPU kernels once
    console.log(builder.kind, builder.fallbackReason); // 'gpu-hploc' and null, or 'tinybvh' and why

    const results = await builder.buildLayouts(meshes, { maxLeafPrimitives: 8 });
    for (const { layout, builtBy, fallbackReason } of results) {
        // layout.nodes, layout.triangles and layout.parents are ArrayBuffers in the GPU layout.
        console.log(builtBy, fallbackReason, layout.nodeCount, layout.maxDepth);
    }
    // #endregion
}

// #region tinybvh
import { createTinyBvhBuilder } from '@pmndrs/bvh';

// #endregion

export async function tinybvh(positions: Float32Array, indices: Uint32Array): Promise<number> {
    // #region tinybvh
    const tiny = await createTinyBvhBuilder(); // loads the WebAssembly once
    const { layout, buildMs } = tiny.buildLayout({ positions, indices }, { maxLeafPrimitives: 8 });
    // #endregion
    return layout.nodeCount + buildMs * 0;
}

// #region hploc
import { GpuHplocBuilder, planGpuHplocBatches, type GpuMeshGeometry } from '@pmndrs/bvh';

// #endregion

export async function hploc(device: GPUDevice, meshes: readonly GpuMeshGeometry[]): Promise<void> {
    // #region hploc
    const builder = new GpuHplocBuilder(device);
    await builder.prepare(); // compiles the kernels, asynchronously

    // One batch builds many meshes with the same dispatches. planGpuHplocBatches splits
    // a scene that doesn't fit the device's buffers (or maxBatchBytes) into several.
    const { batches, tooBig } = planGpuHplocBatches(device, meshes, { maxBatchBytes: 256 * 1024 * 1024 });
    for (const batch of batches) {
        const trees = await builder.buildMany(batch.map((index) => meshes[index]));
        // trees.nodes, trees.triangles and trees.parents are GPU buffers the batch's meshes share.
        for (const mesh of trees.meshes) {
            if (mesh.problem) continue; // refused (too deep, for example): build this one with tinybvh
            // mesh.nodeOffset and mesh.triangleOffset say where its tree sits in those buffers.
        }
        trees.destroy(); // when nothing reads them any more
    }
    console.log('for tinybvh:', tooBig);
    // #endregion
}

export function gpuMeshFromBuffers(positionBuffer: GPUBuffer, indexBuffer: GPUBuffer, vertexCount: number, triangleCount: number): GpuMeshGeometry {
    // #region gpu-input
    // A mesh already on the GPU (the output of a skinning pass, say). The buffers need COPY_SRC usage.
    const mesh: GpuMeshGeometry = { positionBuffer, vertexCount, indexBuffer, indexFormat: 'uint32', triangleCount };
    // #endregion
    return mesh;
}

// #region lbvh
import { GpuLbvhBuilder } from '@pmndrs/bvh';

// #endregion

export async function lbvh(device: GPUDevice, meshes: readonly GpuMeshGeometry[]): Promise<void> {
    // #region lbvh
    if (GpuLbvhBuilder.isSupported(device)) {
        const builder = new GpuLbvhBuilder(device);
        await builder.prepare();
        const trees = await builder.buildMany(meshes);
        const layouts = await builder.readLayouts(trees); // CPU copies, for a refit's setTree or a file
        trees.destroy();
        console.log(layouts.map((layout) => layout?.nodeCount));
    }
    // #endregion
}

export async function preferLbvh(device: GPUDevice): Promise<void> {
    // #region prefer
    const lbvhFirst = await createMeshLayoutBuilder(device, { prefer: 'gpu-lbvh' });
    const cpuOnly = await createMeshLayoutBuilder(device, { prefer: 'tinybvh' }); // for comparisons
    const capped = await createMeshLayoutBuilder(device, { maxBatchBytes: 256 * 1024 * 1024 }); // lower build peak
    // #endregion
    console.log(lbvhFirst.kind, cpuOnly.kind, capped.kind);
}

// #region check
import { checkGpuLayout, GPU_LAYOUT_VERSION, treeSahCost, type GpuLayoutMeshBvh } from '@pmndrs/bvh';

// #endregion

export function check(layout: GpuLayoutMeshBvh, positions: Float32Array, indices: Uint32Array): number {
    // #region check
    const problems = checkGpuLayout(layout, { positions, indices }); // [] when the tree is right
    if (problems.length) throw new Error(problems.join('\n'));
    const cost = treeSahCost(layout); // lower is better; compare builders on the same mesh
    // #endregion
    return cost;
}

export function layoutVersion(): void {
    // #region layout-version
    // A WGSL walk of your own reads our word positions: refuse a layout it wasn't written for.
    const WALK_WRITTEN_FOR = 2;
    if (GPU_LAYOUT_VERSION !== WALK_WRITTEN_FOR) throw new Error(`@pmndrs/bvh layout ${GPU_LAYOUT_VERSION}, walk written for ${WALK_WRITTEN_FOR}`);
    // #endregion
}

// #region read-node
import { GPU_LAYOUT_NODE_WORD, GPU_LAYOUT_NODE_WORDS } from '@pmndrs/bvh';

// #endregion

export function readNode(layout: GpuLayoutMeshBvh, node: number): void {
    // #region read-node
    const floats = new Float32Array(layout.nodes);
    const words = new Uint32Array(layout.nodes);
    const base = node * GPU_LAYOUT_NODE_WORDS; // 16 words, 64 bytes
    const leftMin = floats.subarray(base + GPU_LAYOUT_NODE_WORD.leftMin, base + GPU_LAYOUT_NODE_WORD.leftMin + 3);
    const leftCount = words[base + GPU_LAYOUT_NODE_WORD.leftCount]; // 0: an inner node at leftIndex
    const leftIndex = words[base + GPU_LAYOUT_NODE_WORD.leftIndex]; // a leaf: its first triangle
    // #endregion
    console.log(leftMin, leftCount, leftIndex);
}

// Code samples for docs/scenes/*.mdx (see getting-started.ts for how they are checked).

// #region scene-trees
import { SceneTrees, type MeshGeometry } from '@pmndrs/bvh';

// #endregion

export async function sceneTrees(device: GPUDevice, chair: MeshGeometry, chairMatrices: readonly Float32Array[]): Promise<SceneTrees> {
    // #region scene-trees
    const trees = await SceneTrees.create(device, {
        onTree: (event) => console.log(`mesh ${event.mesh}: ${event.builtBy} tree after ${event.ms.toFixed(0)} ms`, event.replaced ? '(swap)' : ''),
    });
    const mesh = trees.addMesh(chair); // one tree per mesh, shared by its instances
    for (const objectToWorld of chairMatrices) trees.addInstance({ mesh, objectToWorld }); // column-major 4x4

    // Every frame (or whenever the scene changed):
    const encoder = device.createCommandEncoder();
    trees.update(encoder);
    device.queue.submit([encoder.finish()]);
    // #endregion
    return trees;
}

export function whereThingsAre(trees: SceneTrees): void {
    // #region scene-state
    console.log(trees.gpuFirstTrees); // { enabled, reason }: why GPU first trees are off, if they are
    console.log(trees.counts.onGpuTrees, trees.counts.onTinybvhTrees, trees.counts.swaps);
    console.log(trees.timeline.firstTreesMs, trees.timeline.tinybvhTreesMs);
    console.log((trees.memory.resident / 2 ** 20).toFixed(1), 'MiB resident,', (trees.memory.peak / 2 ** 20).toFixed(1), 'MiB peak');
    // #endregion
}

export async function waitForTrees(trees: SceneTrees, frame: () => void): Promise<void> {
    // #region when-built
    frame(); // an update starts the builds
    await trees.whenBuilt(); // every mesh with instances has a tree (GPU or tinybvh)
    frame(); // the next update places them
    await trees.whenRefined(); // every GPU tree has been replaced by tinybvh's
    frame();
    // #endregion
}

export async function buildEvents(device: GPUDevice): Promise<SceneTrees> {
    // #region build-events
    const trees = await SceneTrees.create(device, {
        // Each mesh's build, per builder: 'requested' (in an update), 'started', 'finished' (with its outcome).
        onBuild: (event) => {
            const who = event.mesh !== null ? `mesh ${event.mesh}` : `merged group ${event.group}`;
            if (event.phase !== 'finished') console.log(who, event.builder, event.phase);
            else console.log(who, event.builder, event.outcome, `${event.ms.toFixed(0)} ms`, event.reason ?? '', event.gpuMs ?? '');
        },
        // Placed: what rays see from the next frame.
        onTree: (event) => console.log(`mesh ${event.mesh}: ${event.builtBy} tree placed`),
    });
    // #endregion
    return trees;
}

export async function timing(device: GPUDevice): Promise<void> {
    // #region timing
    // timestamps: each GPU build batch's pass, summed in counts.gpuBuildTimestampMs (needs timestamp-query).
    const trees = await SceneTrees.create(device, { timestamps: true });
    const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
    const resolved = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    const encoder = device.createCommandEncoder();
    // A timestamp before and after everything the update records (copies, refits, the top level).
    trees.update(encoder, { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
    encoder.resolveQuerySet(querySet, 0, 2, resolved, 0);
    encoder.copyBufferToBuffer(resolved, 0, read, 0, 16);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const [start, end] = new BigInt64Array(read.getMappedRange());
    console.log(`update: ${Number(end - start) / 1e6} ms on the GPU; builds so far: ${trees.counts.gpuBuildTimestampMs ?? '-'} ms`);
    // #endregion
    read.unmap();
}

export function bindings(trees: SceneTrees, rebind: (buffers: SceneTrees['buffers']) => void): void {
    // #region version
    let boundVersion = -1;
    const afterUpdate = () => {
        if (trees.version !== boundVersion) {
            rebind(trees.buffers); // a buffer grew: make the bind groups again (nothing recompiles)
            boundVersion = trees.version;
        }
    };
    // #endregion
    afterUpdate();
}

export function runtimeChanges(trees: SceneTrees, lamp: MeshGeometry, matrix: Float32Array, moved: Float32Array): void {
    // #region runtime
    const lampMesh = trees.addMesh(lamp);
    const lampInstance = trees.addInstance({ mesh: lampMesh, objectToWorld: matrix, mask: 0x02, material: 7 });

    trees.setTransform(lampInstance, moved); // the GPU top level is refit at the next update
    trees.setInstance(lampInstance, { visible: false }); // or mask, material, contributesToGi

    trees.removeInstance(lampInstance);
    trees.removeMesh(lampMesh); // a mesh no instance uses; its build is cancelled or ignored
    // #endregion
}

export async function mergedStatic(device: GPUDevice, room: readonly MeshGeometry[], matrices: readonly Float32Array[]): Promise<SceneTrees> {
    // #region merged-static
    const trees = await SceneTrees.create(device); // mergeStaticInstances is on by default
    room.forEach((geometry, index) => {
        const mesh = trees.addMesh(geometry);
        // static: never moves. Merged at the first update when its mesh has no other instance.
        trees.addInstance({ mesh, objectToWorld: matrices[index], static: true });
    });
    // #endregion
    return trees;
}

export function deforming(trees: SceneTrees, character: MeshGeometry, skinnedPositions: GPUBuffer, indexBuffer: GPUBuffer, vertexCount: number): void {
    // #region deforming
    // bounds: a box every pose stays in (object space), kept by the top level. true: the rest pose's box.
    const mesh = trees.addMesh(character, { deforming: { bounds: [-1, 0, -1, 1, 2, 1] } });
    trees.addInstance({ mesh, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] });

    // Each frame, after your pass wrote the posed vertices (STORAGE and COPY_SRC usage):
    trees.setPose(mesh, { positionBuffer: skinnedPositions, vertexCount, indexBuffer, indexFormat: 'uint32' });
    // the next trees.update(encoder) refits the tree on the GPU, and rebuilds it if the refit drifted.
    console.log(trees.driftRatio(mesh), trees.deformingCounts);
    // #endregion
}

export function deformingOnCpu(trees: SceneTrees, cloth: MeshGeometry, posedPositions: Float32Array): void {
    // #region deforming-cpu
    const mesh = trees.addMesh(cloth, { deforming: true });
    trees.addInstance({ mesh, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] });

    // x, y, z per vertex, the mesh's own vertices; with or without subgroups.
    trees.setPoseCpu(mesh, posedPositions);
    // the next trees.update(encoder) refits the tree on the CPU, uploads it and moves the top-level box.
    console.log(trees.deformingCounts.cpuPoses);
    // #endregion
}

export function rebuildNow(trees: SceneTrees, mesh: number): void {
    // #region rebuild
    // After a big pose change (a ragdoll, a cut): rebuilt from the latest GPU pose at the next update, drift or not.
    if (!trees.rebuild(mesh)) console.log('nothing to rebuild from: no subgroups, no GPU pose or no tree yet');
    // #endregion
}

export async function placeOwnTree(device: GPUDevice, crate: MeshGeometry, crateLayout: GpuLayoutMeshBvh, matrix: Float32Array): Promise<SceneTrees> {
    // #region place-tree
    const trees = await SceneTrees.create(device);
    const mesh = trees.addMesh(crate);
    // A tree you built (another device, a file, any builder of the GPU layout) from the same geometry:
    // given before the first update, the mesh starts no builds; later, its builds are cancelled or dropped.
    trees.placeTree(mesh, crateLayout);
    trees.addInstance({ mesh, objectToWorld: matrix });
    // The next update places it: trees.meshStatus(mesh).builtBy === 'app'.
    // #endregion
    return trees;
}

export async function gpuOnly(device: GPUDevice): Promise<SceneTrees> {
    // #region gpu-only
    // GPU trees only: tinybvh builds just the meshes the GPU refuses (and every mesh without subgroups).
    const trees = await SceneTrees.create(device, { refine: false });
    // #endregion
    return trees;
}

export async function options(device: GPUDevice): Promise<SceneTrees> {
    // #region options
    const trees = await SceneTrees.create(device, {
        gpuFirstTrees: true, // H-PLOC trees first on a device with subgroups (default)
        gpuMaxBatchBytes: 256 * 1024 * 1024, // the build-peak cap (default); Infinity: one batch
        workers: { size: 4 }, // the tinybvh worker pool, or a pool of your own
        topLevel: 'auto', // the GPU top level with subgroups, the CPU one without (default)
        mergeStaticInstances: true, // the default: static instances merge at the first update
        triangleLeaves: 'records', // the default; or 'indexed' (16-byte triangles)
        nodeFormat: 'quantized', // the default; or 'f32' (64-byte nodes)
    });
    // #endregion
    return trees;
}

// #region refit-top
import { GpuBvhRefit, leafSlotsOf, parentSlotsOf, type GpuLayoutMeshBvh } from '@pmndrs/bvh';

// #endregion

export async function refitTop(device: GPUDevice, layout: GpuLayoutMeshBvh, pass: GPUComputePassEncoder, treeBuffer: GPUBuffer, primitiveBoxes: GPUBuffer): Promise<void> {
    // #region refit-top
    const refit = new GpuBvhRefit(device);
    await refit.prepare();

    // Once per tree shape, from a CPU copy of the tree:
    const words = new Uint32Array(layout.nodes);
    refit.setShape({
        nodeWordOffset: 0,
        nodeCount: layout.nodeCount,
        parentSlots: parentSlotsOf(words, 0, layout.nodeCount, layout.parents),
        leafSlots: leafSlotsOf(words, 0, layout.nodeCount),
        generation: 1,
    });

    // Every frame, after a pass wrote one box per primitive (8 floats: min x, y, z, unused,
    // max x, y, z, unused), in the order the tree's leaves index them:
    refit.encode(pass, treeBuffer, primitiveBoxes);
    // #endregion
}

// #region refit-mesh
import { GpuMeshRefit } from '@pmndrs/bvh';

// #endregion

export async function refitMesh(
    device: GPUDevice,
    layout: GpuLayoutMeshBvh,
    nodeBuffer: GPUBuffer,
    triangleBuffer: GPUBuffer,
    skinnedPositions: GPUBuffer,
    indexBuffer: GPUBuffer,
    vertexCount: number,
): Promise<void> {
    // #region refit-mesh
    const meshRefit = new GpuMeshRefit(device, { diagnostics: true });
    await meshRefit.prepare();
    // Once, from a CPU copy of the tree (any builder's), in its own buffers:
    meshRefit.setTree(layout);
    // For a mesh inside a scene's buffers, the placement says where, and in what node format:
    // meshRefit.setTree(layout, meshRefitPlacement(scene, meshIndex));

    // Every frame, after the pass that writes the moved vertices:
    meshRefit.pollStats(); // maps an earlier frame's stats copy, without waiting
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    meshRefit.encode(pass, { nodes: nodeBuffer, triangles: triangleBuffer }, { positionBuffer: skinnedPositions, vertexCount, indexBuffer, indexFormat: 'uint32' });
    pass.end();
    meshRefit.copyStats(encoder); // with diagnostics only
    device.queue.submit([encoder.finish()]);

    // A frame or two late: the refit tree's SAH cost over its root's area. Rebuild when it drifts up.
    const cost = meshRefit.latestStats?.relativeCost;
    // #endregion
    console.log(cost);
}

// Code samples for docs/workers/*.mdx (see getting-started.ts for how they are checked).

// #region pool
import { buildTraceScene, SceneTrees, type MeshGeometry, type TraceSceneInstance } from '@pmndrs/bvh';
import { createBvhWorkerPool, generateCpuMeshBvh } from '@pmndrs/bvh/worker';

// #endregion

export async function pool(device: GPUDevice, meshes: readonly MeshGeometry[], instances: readonly TraceSceneInstance[]): Promise<void> {
    // #region pool
    const pool = createBvhWorkerPool(); // up to 4 workers; inline where there is no Worker (Node)
    console.log(pool.usesWorkers, pool.size);

    // One mesh per worker, several meshes at once. The result is a tree in the GPU layout.
    const tree = await pool.build(meshes[0], { maxLeafPrimitives: 8 });

    // Any API that takes a `build` function can use the pool.
    const scene = await buildTraceScene(meshes, instances, { build: (geometry) => pool.build(geometry) });

    // SceneTrees makes its own pool unless you hand it one to share.
    const trees = await SceneTrees.create(device, { workers: pool });

    // A CPU query tree built off the main thread.
    const cpuTree = await generateCpuMeshBvh(pool, meshes[0]);
    // #endregion
    console.log(tree.nodeCount, scene.byteLength, trees.usesWorkers, cpuTree.triangleCount);
    trees.dispose();
    pool.dispose();
}

export async function cancel(meshes: readonly MeshGeometry[]): Promise<void> {
    // #region cancel
    const pool = createBvhWorkerPool({ size: 2, transferInput: false });
    const controller = new AbortController();
    const pending = pool.build(meshes[0], {}, { signal: controller.signal });
    controller.abort(); // rejects at once with an AbortError; a running build finishes in its worker and is dropped
    await pending.catch((error: Error) => console.log(error.name));
    console.log(pool.stats); // { queued, running, workers, finished }
    pool.dispose(); // stops the workers; unfinished builds reject
    // #endregion
}

export async function poolEvents(meshes: readonly MeshGeometry[]): Promise<void> {
    // #region pool-events
    const names = new Map(meshes.map((mesh, index) => [mesh, `mesh ${index}`]));
    const pool = createBvhWorkerPool({
        // Every build: 'queued', 'started' (on a worker, or inline) and 'finished' (built, failed or cancelled).
        onBuild: (event) => console.log(names.get(event.geometry), event.phase, event.inWorker ? 'worker' : 'inline', event.outcome ?? '', `${event.ms.toFixed(1)} ms`),
    });
    // Or per build: onStart when a worker takes it.
    await pool.build(meshes[0], {}, { onStart: () => console.log('mesh 0 is building') });
    // #endregion
    pool.dispose();
}

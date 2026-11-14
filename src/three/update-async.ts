/**
 * three-mesh-bvh's `BVHComputeData.updateAsync()` (PARITY 8.4) over our
 * three parts: {@link updateSceneTreesAsync} reads the scene, starts every
 * build, waits until every mesh has a tree and places them, then rebinds the
 * TSL nodes, so the scene can be traced in full from the next shader run.
 *
 * three-mesh-bvh's `updateAsync` awaits `getBVH` for every object and
 * instance (an override may build them asynchronously, all requested at
 * once) and then repacks everything (`update`). Here the pieces are separate
 * objects, and the builds already run asynchronously: `SceneTrees` starts
 * them in `update(encoder)` (GPU H-PLOC first trees, tinybvh in workers) and
 * `whenBuilt()` resolves when every mesh has one. So the equivalent is:
 *
 *     sync.sync(); await sync.ready     // the scene's meshes and instances, pose pipelines compiled
 *     update(encoder); submit           // starts the builds (sync.update() first: transforms, poses)
 *     await trees.whenBuilt()           // or whenRefined(): every tinybvh tree in
 *     update(encoder); submit           // places the trees
 *     nodes.update()                    // rebinds three's storage nodes if a buffer grew
 *
 * A frame loop that already runs `sync.update(); trees.update(encoder);
 * submit; nodes.update()` each frame needs none of this: its next frame
 * starts the builds. This is for a first load (or a big change) that should
 * be complete before the first traced frame.
 */

/** What {@link updateSceneTreesAsync} drives of a `SceneTrees`. */
export interface SceneTreesAsyncTarget {
    readonly device: GPUDevice;
    update(encoder: GPUCommandEncoder): void;
    whenBuilt(): Promise<void>;
    whenRefined(): Promise<void>;
}

/** Options of {@link updateSceneTreesAsync}. */
export interface UpdateSceneTreesAsyncOptions {
    /** The scene's `SceneTreesObject3DSync`: walked (`sync()`) first, its pose pipelines awaited (`ready`), and `update()`d before each `SceneTrees.update`. */
    readonly sync?: { sync(): unknown; update(): unknown; readonly ready?: Promise<void> };
    /** The `BvhTraceNodes` over the trees: rebound (`update()`) at the end. */
    readonly nodes?: { update(): unknown };
    /** Wait for every mesh's first tree (`'built'`, default: GPU trees may still be refined later) or for tinybvh's (`'refined'`). */
    readonly until?: 'built' | 'refined';
}

/** One update of the trees in an encoder of its own, submitted. */
function submitUpdate(trees: SceneTreesAsyncTarget, sync: UpdateSceneTreesAsyncOptions['sync']): void {
    sync?.update();
    const encoder = trees.device.createCommandEncoder({ label: '@pmndrs/bvh/three updateSceneTreesAsync' });
    trees.update(encoder);
    trees.device.queue.submit([encoder.finish()]);
}

/**
 * Brings `trees` up to date with the scene and resolves once every mesh has
 * a tree in place (see the file comment): three-mesh-bvh's
 * `await bvh.updateAsync()`. Runs two updates of its own, each in its own
 * submitted encoder.
 */
export async function updateSceneTreesAsync(trees: SceneTreesAsyncTarget, options: UpdateSceneTreesAsyncOptions = {}): Promise<void> {
    const { sync, nodes, until = 'built' } = options;
    if (sync) {
        sync.sync();
        await sync.ready;
    }
    submitUpdate(trees, sync);
    await (until === 'refined' ? trees.whenRefined() : trees.whenBuilt());
    submitUpdate(trees, sync);
    nodes?.update();
}

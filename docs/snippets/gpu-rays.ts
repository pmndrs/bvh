// Code samples for docs/tracing/batches-and-the-cpu-walk.mdx (GpuRays; see getting-started.ts for how they are checked).

import type { SceneTrees } from '@pmndrs/bvh';

// #region gpu-rays
import { GpuRays, packRays, uploadTraceScene, type TraceSceneData } from '@pmndrs/bvh';

// #endregion

export async function gpuRays(trees: SceneTrees, data: TraceSceneData): Promise<number> {
    const device = trees.device;
    // #region gpu-rays
    // A SceneTrees, or an uploaded scene: both are a TraceSource.
    const rays = await GpuRays.create(device, trees);
    const uploaded = await GpuRays.create(device, uploadTraceScene(device, data));

    // Picking: one ray, its closest hit or null.
    const hit = await rays.pick([0, 1, 5], [0, 0, -1]);
    if (hit) console.log(hit.instance, hit.triangle, hit.t, hit.material, hit.normal);

    // A batch: any number of rays in one dispatch, read back and decoded.
    const batch = packRays(Array.from({ length: 4096 }, (_, i) => ({ origin: [i * 0.01, 1, 5], direction: [0, 0, -1], tMin: 0, tMax: 1e30 })));
    const closest = await rays.traceClosest(batch);
    console.log(`${closest.hits} of ${closest.count} hit, ${closest.gpuMs ?? closest.wallMs} ms`, closest.hit(0));
    const shadows = await uploaded.traceAny(batch, { mask: 0x1 });
    console.log(`${shadows.blockedCount} blocked`, shadows.blocked(0));
    rays.destroy();
    // #endregion
    uploaded.destroy();
    return closest.hits;
}

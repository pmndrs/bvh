/**
 * The overlay's numbers from the library's public API: a tree in the GPU
 * layout (from any builder, or a `BvhBoundsTree`'s `layout`), and a
 * `SceneTrees` scene (which builder its trees came from, its bytes).
 */

import { treeSahCost, type GpuLayoutMeshBvh, type SceneTreeEvent, type SceneTrees } from '@pmndrs/bvh';
import type { BuildInfo, TreeInfo } from './overlay.ts';

/** SAH cost, depth, nodes, triangles and bytes of one tree. */
export function layoutInfo(layout: GpuLayoutMeshBvh): TreeInfo {
    return {
        sah: treeSahCost(layout),
        depth: layout.maxDepth,
        nodes: layout.nodeCount,
        triangles: layout.triangleCount,
        bytes: layout.nodes.byteLength + layout.triangles.byteLength + layout.parents.byteLength,
    };
}

/** The sum over several trees: the SAH cost weighted by triangles, the deepest depth. */
export function sumInfo(infos: readonly TreeInfo[]): TreeInfo {
    let triangles = 0;
    let sah = 0;
    let nodes = 0;
    let bytes = 0;
    let depth = 0;
    for (const info of infos) {
        triangles += info.triangles ?? 0;
        sah += (info.sah ?? 0) * (info.triangles ?? 0);
        nodes += info.nodes ?? 0;
        bytes += info.bytes ?? 0;
        depth = Math.max(depth, info.depth ?? 0);
    }
    return { sah: triangles ? sah / triangles : null, depth, nodes, triangles, bytes };
}

/**
 * Collects a SceneTrees scene's tree events (pass {@link SceneTreesLog.onTree}
 * as `SceneTreesOptions.onTree`): which builder made what rays see now, and why
 * a mesh's first tree didn't come from the GPU.
 */
export class SceneTreesLog {
    readonly events: SceneTreeEvent[] = [];
    readonly started = performance.now();
    readonly onTree = (event: SceneTreeEvent): void => {
        this.events.push(event);
    };

    /** What the overlay's builder section shows for `trees`. */
    build(trees: SceneTrees): BuildInfo {
        const counts = trees.counts;
        const parts: string[] = [];
        const gpuFirst = this.events.filter((event) => event.builtBy === 'gpu').length;
        if (counts.onGpuTrees) parts.push(`gpu (H-PLOC) ×${counts.onGpuTrees}`);
        if (counts.onTinybvhTrees) parts.push(`tinybvh ×${counts.onTinybvhTrees}`);
        if (counts.waitingForFirstTree) parts.push(`waiting ×${counts.waitingForFirstTree}`);
        const reasons = new Set<string>();
        if (trees.gpuFirstTrees.reason) reasons.add(trees.gpuFirstTrees.reason);
        for (const event of this.events) if (event.fallbackReason && event.replaced === null) reasons.add(event.fallbackReason);
        const timeline = trees.timeline;
        return {
            builtBy: (gpuFirst && !counts.onGpuTrees ? `gpu first ×${gpuFirst} → ` : '') + (parts.join(', ') || '-'),
            fallbackReason: reasons.size ? [...reasons].join('; ') : null,
            cpuMs: timeline.firstTreesMs ?? timeline.tinybvhTreesMs,
            // SceneTrees times its GPU batches from submit to stats (not timestamps): shown as such.
            gpuMs: counts.gpuBatches ? counts.gpuBuildMs : undefined,
            gpuSource: 'submit to readback',
        };
    }

    /** The mesh handles seen in tree events (the sync doesn't list its handles). */
    meshes(): number[] {
        return [...new Set(this.events.map((event) => event.mesh).filter((mesh): mesh is number => mesh !== null))];
    }

    /** Node and triangle counts and bytes of the scene (SceneTrees keeps no CPU copy, so no SAH cost here). */
    tree(trees: SceneTrees, meshes: Iterable<number> = this.meshes()): TreeInfo {
        let nodes = 0;
        let triangles = 0;
        let depth = 0;
        for (const mesh of meshes) {
            let status;
            try {
                status = trees.meshStatus(mesh);
            } catch {
                continue; // removed since
            }
            if (!status.placement) continue;
            nodes += status.placement.nodeCount;
            triangles += status.placement.triangleCount;
            depth = Math.max(depth, status.maxDepth);
        }
        return { sah: null, depth, nodes, triangles, bytes: trees.memory.resident };
    }
}

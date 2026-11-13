/**
 * Scene-level CPU queries (PARITY 5.2 and 5.6, three-mesh-bvh's `ObjectBVH`):
 * a ray against every instance of a scene, and the instances whose world
 * boxes meet a frustum or a box, over the scene's own top level.
 *
 * **One walk.** {@link CpuSceneBvh.raycast} runs `createCpuTracer`, the
 * step-for-step port of the WGSL walk, over the scene's buffers (the top
 * level, then each instance's mesh tree in its own space). All hits work as
 * in `CpuMeshBvh`: the walk asks the triangle filter about every triangle
 * (`filterEveryTriangle`), the filter records the hit and lets the ray
 * through, so the walk visits every box the ray crosses. The culling queries
 * walk the top level's nodes (the same boxes the ray walk tests).
 *
 * **Moving instances.** {@link CpuSceneBvh.setInstances} takes the instances
 * again (transforms, visibility, masks, materials) and rebuilds the top level
 * only; the mesh trees stay where they are. New meshes need a new scene.
 *
 * Engine-free: `/three` wraps it as `BvhSceneRaycaster` (three's `Raycaster`
 * and `Frustum` in, three intersections out).
 */

import type { MeshBvh, MeshGeometry } from '../trace/mesh-bvh.ts';
import { createCpuTracer, type CpuTracer } from '../trace/cpu-traversal.ts';
import { INSTANCE_RECORD_WORDS, INSTANCE_WORD, INSTANCE_MASK_SHIFT, NODE_WORD, BVH_NODE_WORDS, TOP_HEADER_WORD, TOP_LEVEL_HEADER_WORDS, RAY_MASK_ALL } from '../trace/layout.ts';
import { assembleTraceScene, type TraceSceneData, type TraceSceneInstance } from '../trace/scene-assembly.ts';
import { buildTopLevel, type TraceInstance } from '../trace/top-level.ts';
import type { TraceRay } from '../trace/rays.ts';
import type { SahBuildOptions } from '../trace/sah-builder.ts';

/** One hit of {@link CpuSceneBvh.raycast}. */
export interface CpuSceneHit {
    /** Distance along the ray (world units when the direction is unit length). */
    readonly t: number;
    /** The instance's `instanceIndex`. */
    readonly instance: number;
    /** The original triangle in the instance's mesh. */
    readonly triangle: number;
    /** Barycentric weights of corners 1 and 2. */
    readonly u: number;
    readonly v: number;
    /** The ray hit the authored front side (object-space winding, mirrored instances included). */
    readonly frontFace: boolean;
}

export interface CpuSceneRaycastOptions {
    /** The closest hit only (default false: every hit, sorted by `t`). */
    readonly firstHitOnly?: boolean;
    /** The ray mask: instances with `(mask & rayMask) == 0` are skipped. Default all ones. */
    readonly mask?: number;
    /** Whether a hit counts (side culling, groups): not counted hits let the ray through. Default: every hit counts. */
    readonly accept?: (instance: number, triangle: number, frontFace: boolean) => boolean;
}

/** `[a, b, c, d]` per plane with `a*x + b*y + c*z + d >= 0` inside, as three's `Plane` (normal, constant). Any number of planes. */
export type CpuFrustumPlanes = ArrayLike<number>;

/** The scene's top level as typed arrays, for the culling walks. */
interface TopLevelView {
    readonly floats: Float32Array;
    readonly words: Uint32Array;
    readonly instanceCount: number;
    readonly nodeCount: number;
    readonly firstNodeWord: number;
}

function viewOf(scene: TraceSceneData): TopLevelView {
    const words = new Uint32Array(scene.topLevel.buffer);
    const instanceCount = words[TOP_HEADER_WORD.instanceCount];
    return {
        floats: new Float32Array(scene.topLevel.buffer),
        words,
        instanceCount,
        nodeCount: words[TOP_HEADER_WORD.nodeCount],
        firstNodeWord: TOP_LEVEL_HEADER_WORDS + instanceCount * INSTANCE_RECORD_WORDS,
    };
}

/** The scene's instances, the mesh trees and a CPU walk over them; see the file comment. */
export class CpuSceneBvh {
    private current: TraceSceneData;
    private view: TopLevelView;
    private tracer: CpuTracer;
    private accept: CpuSceneRaycastOptions['accept'] | null = null;
    private collect: CpuSceneHit[] | null = null;

    /**
     * Places `meshes` (any builder's trees, records) and `instances` in one
     * scene. `meshes[i]` is the tree of instances with `meshIndex` i.
     */
    constructor(
        readonly meshes: readonly MeshBvh[],
        instances: readonly TraceSceneInstance[],
        private readonly topLevelOptions: Partial<SahBuildOptions> = {},
    ) {
        // f32 nodes, as before the scene default became quantized (2026-11-13): the CPU walk skips the decode.
        this.current = assembleTraceScene(meshes, instances, topLevelOptions, { nodeFormat: 'f32' });
        this.view = viewOf(this.current);
        this.tracer = this.makeTracer();
    }

    /** Builds each mesh with `build` (tinybvh's `buildMeshBvhWithTinyBvh`, say) and places the instances. */
    static build(geometries: readonly MeshGeometry[], instances: readonly TraceSceneInstance[], build: (geometry: MeshGeometry) => MeshBvh): CpuSceneBvh {
        return new CpuSceneBvh(geometries.map(build), instances);
    }

    /** The scene's buffers (for `createCpuTracer` or an upload). */
    get scene(): TraceSceneData {
        return this.current;
    }

    /** Instances in the top level (hidden ones and ones whose transform can't be inverted aren't). */
    get instanceCount(): number {
        return this.view.instanceCount;
    }

    /** New transforms, visibility, masks or materials for the same meshes: rebuilds the top level only. */
    setInstances(instances: readonly TraceSceneInstance[]): void {
        const placements = this.current.meshes;
        const traced = instances.map((instance): TraceInstance => {
            const mesh = this.meshes[instance.meshIndex];
            if (!mesh) throw new RangeError(`CpuSceneBvh.setInstances: instance ${instance.instanceIndex} uses mesh ${instance.meshIndex}, which does not exist`);
            return {
                instanceIndex: instance.instanceIndex,
                nodeOffset: placements[instance.meshIndex].nodeOffset,
                triangleOffset: placements[instance.meshIndex].triangleOffset,
                meshBounds: mesh.bounds,
                objectToWorld: instance.objectToWorld,
                contributesToGi: instance.contributesToGi,
                visible: instance.visible,
                mask: instance.mask,
                material: instance.material,
            };
        });
        const topLevel = buildTopLevel(traced, this.topLevelOptions);
        this.current = { ...this.current, topLevel, requiredStackSize: topLevel.maxDepth + 1 + this.current.maxMeshDepth };
        this.view = viewOf(this.current);
        this.tracer = this.makeTracer();
    }

    /** Every hit along the ray (sorted by `t`), or the closest with `firstHitOnly`. */
    raycast(ray: TraceRay, options: CpuSceneRaycastOptions = {}): CpuSceneHit[] {
        const previousAccept = this.accept;
        const previousCollect = this.collect;
        const collect: CpuSceneHit[] | null = options.firstHitOnly ? null : [];
        this.accept = options.accept ?? null;
        this.collect = collect;
        try {
            const hit = this.tracer.closest(ray, { mask: options.mask });
            if (collect) return collect.sort((a, b) => a.t - b.t);
            return hit ? [{ t: hit.t, instance: hit.instanceIndex, triangle: hit.triangleIndex, u: hit.u, v: hit.v, frontFace: hit.frontFace }] : [];
        } finally {
            this.accept = previousAccept;
            this.collect = previousCollect;
        }
    }

    /**
     * The `instanceIndex` of every instance whose world box meets the planes
     * (is not wholly outside any of them), as three's `Frustum.intersectsBox`
     * tests a box. Walks the top level, skipping subtrees outside a plane and
     * taking subtrees inside every plane whole. With `mask`, only instances it
     * names (as rays do).
     */
    intersectsFrustum(planes: CpuFrustumPlanes, options: { readonly mask?: number } = {}): number[] {
        const count = Math.floor(planes.length / 4);
        return this.cull(options.mask, (floats, min, max) => {
            let inside = true;
            for (let p = 0; p < count; p++) {
                const a = planes[p * 4];
                const b = planes[p * 4 + 1];
                const c = planes[p * 4 + 2];
                const d = planes[p * 4 + 3];
                // The corner furthest along the normal (three's "positive vertex") and the nearest.
                const far = a * (a > 0 ? floats[max] : floats[min]) + b * (b > 0 ? floats[max + 1] : floats[min + 1]) + c * (c > 0 ? floats[max + 2] : floats[min + 2]) + d;
                if (far < 0) return 'outside';
                const near = a * (a > 0 ? floats[min] : floats[max]) + b * (b > 0 ? floats[min + 1] : floats[max + 1]) + c * (c > 0 ? floats[min + 2] : floats[max + 2]) + d;
                if (near < 0) inside = false;
            }
            return inside ? 'inside' : 'crossing';
        });
    }

    /** The `instanceIndex` of every instance whose world box meets `box` (`[minX, minY, minZ, maxX, maxY, maxZ]`). */
    intersectsBox(box: ArrayLike<number>, options: { readonly mask?: number } = {}): number[] {
        return this.cull(options.mask, (floats, min, max) => {
            let inside = true;
            for (let k = 0; k < 3; k++) {
                if (floats[max + k] < box[k] || floats[min + k] > box[k + 3]) return 'outside';
                if (floats[min + k] < box[k] || floats[max + k] > box[k + 3]) inside = false;
            }
            return inside ? 'inside' : 'crossing';
        });
    }

    /** The top-level walk of the culling queries: `test` says where a child box (min and max at those float words) lies. */
    private cull(mask: number | undefined, test: (floats: Float32Array, min: number, max: number) => 'outside' | 'inside' | 'crossing'): number[] {
        const { floats, words, nodeCount, firstNodeWord } = this.view;
        const out: number[] = [];
        if (nodeCount === 0) return out;
        const rayMask = (mask ?? RAY_MASK_ALL) >>> 0;
        const record = (index: number) => {
            const base = TOP_LEVEL_HEADER_WORDS + index * INSTANCE_RECORD_WORDS;
            const instanceMask = (words[base + INSTANCE_WORD.flags] >>> INSTANCE_MASK_SHIFT) & 0xff;
            if ((instanceMask & rayMask) !== 0) out.push(words[base + INSTANCE_WORD.instanceIndex]);
        };
        /** Every record under node `node` (the subtree is wholly inside). */
        const all = (node: number) => {
            const stack = [node];
            while (stack.length > 0) {
                const base = firstNodeWord + stack.pop()! * BVH_NODE_WORDS;
                for (const side of [0, 1]) {
                    const index = words[base + (side ? NODE_WORD.rightIndex : NODE_WORD.leftIndex)];
                    const count = words[base + (side ? NODE_WORD.rightCount : NODE_WORD.leftCount)];
                    if (index === 0xffffffff) continue;
                    if (count > 0) for (let k = 0; k < count; k++) record(index + k);
                    else stack.push(index);
                }
            }
        };
        const stack = [0];
        while (stack.length > 0) {
            const base = firstNodeWord + stack.pop()! * BVH_NODE_WORDS;
            for (const side of [0, 1]) {
                const index = words[base + (side ? NODE_WORD.rightIndex : NODE_WORD.leftIndex)];
                const count = words[base + (side ? NODE_WORD.rightCount : NODE_WORD.leftCount)];
                if (index === 0xffffffff) continue;
                const min = base + (side ? NODE_WORD.rightMin : NODE_WORD.leftMin);
                const max = base + (side ? NODE_WORD.rightMax : NODE_WORD.leftMax);
                const where = test(floats, min, max);
                if (where === 'outside') continue;
                if (count > 0) for (let k = 0; k < count; k++) record(index + k);
                else if (where === 'inside') all(index);
                else stack.push(index);
            }
        }
        return out;
    }

    private makeTracer(): CpuTracer {
        return createCpuTracer(this.current, {
            filterEveryTriangle: true,
            triangleFilter: (candidate) => {
                if (this.accept && !this.accept(candidate.instance, candidate.triangle, candidate.frontFace)) return true;
                if (!this.collect) return false;
                this.collect.push({ t: candidate.t, instance: candidate.instance, triangle: candidate.triangle, u: candidate.barycentrics[0], v: candidate.barycentrics[1], frontFace: candidate.frontFace });
                return true;
            },
        });
    }
}

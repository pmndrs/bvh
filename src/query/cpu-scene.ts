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
import { INSTANCE_RECORD_WORDS, INSTANCE_WORD, INSTANCE_MASK_SHIFT, NODE_WORD, BVH_NODE_WORDS, EMPTY_CHILD_INDEX, TOP_HEADER_WORD, TOP_LEVEL_HEADER_WORDS, RAY_MASK_ALL } from '../gpu-layout.ts';
import { BVH_CONTAINED, type BvhTraverseCallbacks } from './cpu-mesh-bvh.ts';
import { BvhBox3, type BvhBox3Like } from './shapes.ts';
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

/**
 * The callbacks of {@link CpuSceneBvh.shapecast}: three-mesh-bvh's
 * `ObjectBVH.shapecast` callbacks (0.9.18) with its argument order, the
 * object named by its instance's `instanceIndex`. `Box` is what the boxes
 * come as (a {@link BvhBox3} here, a three `Box3` in `@pmndrs/bvh/three`).
 */
export interface CpuSceneShapecastCallbacks<Box = BvhBox3Like> {
    /** As `CpuMeshBvh.traverse`'s: whether to enter a world box (falsy skips it, {@link BVH_CONTAINED} takes its subtree whole). */
    intersectsBounds(box: Box, isLeaf: boolean, score: number | undefined, depth: number, nodeIndex: number): unknown;
    /** Each instance of an entered leaf or a contained box, by `instanceIndex`. A truthy return stops the cast. */
    intersectsInstance?(instanceIndex: number, contained: boolean, depth: number): unknown;
    /** A range of top-level positions ({@link CpuSceneBvh.instanceAt}), before its instances; a truthy return stops the cast. */
    intersectsRange?(offset: number, count: number, contained: boolean, depth: number, nodeIndex: number, box: Box): unknown;
    /** Scores a box, once per child; the lower score is entered first. */
    boundsTraverseOrder?(box: Box): number;
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

    /** The `instanceIndex` of the instance record at a top-level position (what {@link traverse} ranges count). */
    instanceAt(position: number): number {
        const { words, instanceCount } = this.view;
        if (!(position >= 0 && position < instanceCount)) throw new RangeError(`CpuSceneBvh.instanceAt: position ${position} is not in the top level (${instanceCount} instances)`);
        return words[TOP_LEVEL_HEADER_WORDS + position * INSTANCE_RECORD_WORDS + INSTANCE_WORD.instanceIndex];
    }

    /**
     * The top level's walk with three-mesh-bvh's `shapecast` contract
     * (`CpuMeshBvh.traverse`'s callbacks, step for step: depth first, the
     * root's box never asked about, each child scored once by
     * `boundsTraverseOrder` and the lower score entered first, a falsy
     * `intersectsBounds` skipping the box, {@link BVH_CONTAINED} reporting the
     * box's subtree as one range, a truthy `intersectsRange` stopping the
     * walk). Boxes are the instances' world boxes (the ones rays test), ranges
     * are top-level positions ({@link instanceAt} maps them), and `nodeIndex`
     * is the child slot (`parentNode * 2 + side`). One instance per leaf.
     * Returns true when it was stopped.
     */
    traverse(callbacks: BvhTraverseCallbacks): boolean {
        const { intersectsBounds, intersectsRange, boundsTraverseOrder: order } = callbacks;
        const { floats, words, nodeCount, firstNodeWord } = this.view;
        if (nodeCount === 0) return false;
        const box = new Float32Array(6);
        const base = (node: number) => firstNodeWord + node * BVH_NODE_WORDS;
        const readBox = (slot: number) => {
            const at = base(slot >> 1);
            const min = at + ((slot & 1) === 0 ? NODE_WORD.leftMin : NODE_WORD.rightMin);
            const max = at + ((slot & 1) === 0 ? NODE_WORD.leftMax : NODE_WORD.rightMax);
            for (let k = 0; k < 3; k++) {
                box[k] = floats[min + k];
                box[k + 3] = floats[max + k];
            }
            return box;
        };
        const indexOf = (slot: number) => words[base(slot >> 1) + ((slot & 1) === 0 ? NODE_WORD.leftIndex : NODE_WORD.rightIndex)];
        const countOf = (slot: number) => words[base(slot >> 1) + ((slot & 1) === 0 ? NODE_WORD.leftCount : NODE_WORD.rightCount)];
        const hasRight = (node: number) => words[base(node) + NODE_WORD.rightIndex] !== EMPTY_CHILD_INDEX;
        // A root with only a left child is that child (one instance: three-mesh-bvh's root leaf, reported unasked).
        if (!hasRight(0) && countOf(0) > 0) return Boolean(intersectsRange.call(callbacks, indexOf(0), countOf(0), false, 0, 0, readBox(0)));
        const slots: number[] = [];
        const depths: number[] = [];
        const scores: (number | undefined)[] = [];
        const limit = 2 * (nodeCount + 2);
        const pushChildren = (node: number, depth: number): void => {
            const right = hasRight(node);
            let first = node * 2;
            let second = node * 2 + 1;
            let firstScore: number | undefined;
            let secondScore: number | undefined;
            if (order) {
                firstScore = order.call(callbacks, readBox(first));
                if (right) {
                    secondScore = order.call(callbacks, readBox(second));
                    if (secondScore < firstScore) {
                        [first, second] = [second, first];
                        [firstScore, secondScore] = [secondScore, firstScore];
                    }
                }
            }
            if (right) {
                slots.push(second);
                depths.push(depth);
                scores.push(secondScore);
            }
            slots.push(first);
            depths.push(depth);
            scores.push(firstScore);
            if (slots.length > limit) throw new RangeError(`CpuSceneBvh.traverse: the stack passed ${limit} entries; the top level is not a tree`);
        };
        /** A contained box's subtree: its leaves are one run of positions (the builder partitions in place). */
        const containedRange = (slot: number, depth: number): boolean => {
            let first = slot;
            while (countOf(first) === 0) first = indexOf(first) * 2;
            let last = slot;
            while (countOf(last) === 0) last = hasRight(indexOf(last)) ? indexOf(last) * 2 + 1 : indexOf(last) * 2;
            const offset = indexOf(first);
            return Boolean(intersectsRange.call(callbacks, offset, indexOf(last) + countOf(last) - offset, true, depth, slot, readBox(slot)));
        };
        pushChildren(0, 1);
        while (slots.length > 0) {
            const slot = slots.pop()!;
            const depth = depths.pop()!;
            const score = scores.pop();
            const index = indexOf(slot);
            const count = countOf(slot);
            const isLeaf = count > 0;
            const answer = intersectsBounds.call(callbacks, readBox(slot), isLeaf, score, depth, slot);
            if (answer === BVH_CONTAINED) {
                if (containedRange(slot, depth)) return true;
            } else if (answer) {
                if (isLeaf) {
                    if (intersectsRange.call(callbacks, index, count, false, depth, slot, readBox(slot))) return true;
                } else {
                    pushChildren(index, depth + 1);
                }
            }
        }
        return false;
    }

    /**
     * three-mesh-bvh's `ObjectBVH.shapecast(callbacks)` over the instances
     * (PARITY 5.3): {@link traverse}, with each entered leaf's (or contained
     * box's) instances handed to `intersectsInstance` by `instanceIndex`.
     * Boxes come as a {@link BvhBox3} (a scratch object, valid during the
     * call). `intersectsRange`, when given, is asked first and a truthy answer
     * stops the cast; with neither, a cast stops at the first contained box,
     * as three-mesh-bvh's does. Returns true when a callback stopped it.
     */
    shapecast(callbacks: CpuSceneShapecastCallbacks): boolean {
        return this.shapecastWith(callbacks, new BvhBox3());
    }

    /** {@link shapecast} with the caller's box type (an engine adapter passes its own; it needs `min` and `max` points). */
    shapecastWith<B extends BvhBox3Like>(callbacks: CpuSceneShapecastCallbacks<B>, box: B): boolean {
        const { intersectsBounds, intersectsInstance, intersectsRange, boundsTraverseOrder } = callbacks;
        const fill = (array: Float32Array): B => {
            box.min.x = array[0];
            box.min.y = array[1];
            box.min.z = array[2];
            box.max.x = array[3];
            box.max.y = array[4];
            box.max.z = array[5];
            return box;
        };
        let range: BvhTraverseCallbacks['intersectsRange'];
        if (intersectsInstance) {
            range = (offset, count, contained, depth, nodeIndex, array) => {
                if (intersectsRange && intersectsRange.call(callbacks, offset, count, contained, depth, nodeIndex, fill(array))) return true;
                for (let position = offset, end = offset + count; position < end; position++) {
                    if (intersectsInstance.call(callbacks, this.instanceAt(position), contained, depth)) return true;
                }
                return false;
            };
        } else if (intersectsRange) {
            range = (offset, count, contained, depth, nodeIndex, array) => intersectsRange.call(callbacks, offset, count, contained, depth, nodeIndex, fill(array));
        } else {
            range = (_offset, _count, contained) => contained;
        }
        return this.traverse({
            intersectsBounds: (array, isLeaf, score, depth, nodeIndex) => intersectsBounds.call(callbacks, fill(array), isLeaf, score, depth, nodeIndex),
            intersectsRange: range,
            boundsTraverseOrder: boundsTraverseOrder ? (array) => boundsTraverseOrder.call(callbacks, fill(array)) : undefined,
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

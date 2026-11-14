/**
 * CpuSceneBvh.traverse and shapecast (PARITY 5.3, three-mesh-bvh's
 * `ObjectBVH.shapecast`): the same calls, with the same arguments in the same
 * order, as three-mesh-bvh 0.9.18's `shapecastTraverse` run line for line
 * over the top level, and the instances a sphere cast finds against brute
 * force over every instance's world box.
 */

import { describe, expect, it } from 'vitest';
import { createTinyBvhBuilder } from '../tinybvh.ts';
import { buildMeshBvhWithTinyBvh } from '../trace/mesh-bvh-topology.ts';
import { BVH_NODE_WORDS, EMPTY_CHILD_INDEX, INSTANCE_RECORD_WORDS, NODE_WORD, TOP_HEADER_WORD, TOP_LEVEL_HEADER_WORDS } from '../trace/layout.ts';
import { createRandom, randomAffine, randomTriangleSoup, tessellatedQuad } from '../trace/oracle-fixtures.ts';
import type { TraceSceneInstance } from '../trace/scene-assembly.ts';
import { transformBounds } from '../trace/top-level.ts';
import { BVH_CONTAINED, BVH_INTERSECTED, BVH_NOT_INTERSECTED, type BvhTraverseCallbacks } from './cpu-mesh-bvh.ts';
import { CpuSceneBvh } from './cpu-scene.ts';

const tinyBvh = await createTinyBvhBuilder();

function makeScene(seed: number, count = 60) {
    const random = createRandom(seed);
    const geometries = [randomTriangleSoup(random, 60, 1.5, 0.8), tessellatedQuad(4, 1), randomTriangleSoup(random, 25, 0.8, 0.5)];
    const instances: TraceSceneInstance[] = Array.from({ length: count }, (_, i) => ({
        meshIndex: i % 3,
        instanceIndex: i,
        objectToWorld: randomAffine(random, { extent: 14, mirror: true }),
        visible: i % 11 !== 7,
    }));
    const scene = new CpuSceneBvh(
        geometries.map((geometry) => buildMeshBvhWithTinyBvh(geometry, tinyBvh)),
        instances,
    );
    return { random, geometries, instances, scene };
}

/** One child slot of the top level: its box, and its leaf range or inner node. */
function slotOf(scene: CpuSceneBvh, slot: number): { box: number[]; index: number; count: number } {
    const buffer = scene.scene.topLevel.buffer;
    const words = new Uint32Array(buffer);
    const floats = new Float32Array(buffer);
    const first = TOP_LEVEL_HEADER_WORDS + words[TOP_HEADER_WORD.instanceCount] * INSTANCE_RECORD_WORDS;
    const base = first + (slot >> 1) * BVH_NODE_WORDS;
    const left = (slot & 1) === 0;
    const min = base + (left ? NODE_WORD.leftMin : NODE_WORD.rightMin);
    const max = base + (left ? NODE_WORD.leftMax : NODE_WORD.rightMax);
    return {
        box: [floats[min], floats[min + 1], floats[min + 2], floats[max], floats[max + 1], floats[max + 2]],
        index: words[base + (left ? NODE_WORD.leftIndex : NODE_WORD.rightIndex)],
        count: words[base + (left ? NODE_WORD.leftCount : NODE_WORD.rightCount)],
    };
}

/** A contained slot's run: its leftmost leaf's offset to its rightmost leaf's end. */
function subtreeRun(scene: CpuSceneBvh, slot: number): { offset: number; count: number } {
    let first = slotOf(scene, slot);
    while (first.count === 0) first = slotOf(scene, first.index * 2);
    let last = slotOf(scene, slot);
    while (last.count === 0) last = slotOf(scene, slotOf(scene, last.index * 2 + 1).index === EMPTY_CHILD_INDEX ? last.index * 2 : last.index * 2 + 1);
    return { offset: first.index, count: last.index + last.count - first.index };
}

/** three-mesh-bvh 0.9.18's `shapecastTraverse` (src/core/cast/shapecast.js), line for line, over the top level; logs every call. */
function threeShapecast(scene: CpuSceneBvh, callbacks: BvhTraverseCallbacks): { result: unknown; log: string[] } {
    const log: string[] = [];
    const bounds = (box: number[], isLeaf: boolean, score: number | undefined, depth: number, nodeIndex: number) => {
        log.push(`bounds ${box.join()} ${isLeaf} ${score} ${depth} ${nodeIndex}`);
        return callbacks.intersectsBounds(Float32Array.from(box), isLeaf, score, depth, nodeIndex);
    };
    const range = (offset: number, count: number, contained: boolean, depth: number, nodeIndex: number, box: number[]) => {
        log.push(`range ${offset} ${count} ${contained} ${depth} ${nodeIndex} ${box.join()}`);
        return callbacks.intersectsRange(offset, count, contained, depth, nodeIndex, Float32Array.from(box));
    };
    const score = (box: number[]) => {
        log.push(`order ${box.join()}`);
        return callbacks.boundsTraverseOrder!(Float32Array.from(box));
    };
    const inner = (node: number, depth: number): unknown => {
        let c1 = node * 2;
        let c2 = node * 2 + 1;
        let score1: number | undefined;
        let score2: number | undefined;
        if (callbacks.boundsTraverseOrder) {
            score1 = score(slotOf(scene, c1).box);
            score2 = score(slotOf(scene, c2).box);
            if (score2 < score1) {
                [c1, c2] = [c2, c1];
                [score1, score2] = [score2, score1];
            }
        }
        for (const [child, childScore] of [[c1, score1], [c2, score2]] as const) {
            const at = slotOf(scene, child);
            const answer = bounds(at.box, at.count > 0, childScore, depth + 1, child);
            let stop: unknown;
            if (answer === BVH_CONTAINED) {
                const run = subtreeRun(scene, child);
                stop = range(run.offset, run.count, true, depth + 1, child, at.box);
            } else if (answer) {
                stop = at.count > 0 ? range(at.index, at.count, false, depth + 1, child, at.box) : inner(at.index, depth + 1);
            }
            if (stop) return true;
        }
        return false;
    };
    return { result: inner(0, 0), log };
}

function logged(callbacks: BvhTraverseCallbacks, log: string[]): BvhTraverseCallbacks {
    return {
        intersectsBounds: (box, isLeaf, score, depth, nodeIndex) => {
            log.push(`bounds ${Array.from(box).join()} ${isLeaf} ${score} ${depth} ${nodeIndex}`);
            return callbacks.intersectsBounds(box, isLeaf, score, depth, nodeIndex);
        },
        intersectsRange: (offset, count, contained, depth, nodeIndex, box) => {
            log.push(`range ${offset} ${count} ${contained} ${depth} ${nodeIndex} ${Array.from(box).join()}`);
            return callbacks.intersectsRange(offset, count, contained, depth, nodeIndex, box);
        },
        ...(callbacks.boundsTraverseOrder
            ? {
                  boundsTraverseOrder: (box: Float32Array) => {
                      log.push(`order ${Array.from(box).join()}`);
                      return callbacks.boundsTraverseOrder!(box);
                  },
              }
            : {}),
    };
}

describe('CpuSceneBvh.traverse: three-mesh-bvh shapecast contract over the top level', () => {
    it('makes the same calls, with the same arguments in the same order, as three-mesh-bvh 0.9.18 shapecast', () => {
        const { scene } = makeScene(81);
        const answers = [undefined, false, BVH_NOT_INTERSECTED, BVH_INTERSECTED, true, BVH_CONTAINED, BVH_INTERSECTED, true];
        const hash = (box: Float32Array) => Math.abs(Math.floor((box[0] * 7919 + box[4] * 104729 + box[2] * 31) * 1000));
        for (const ordered of [false, true]) {
            for (const stopEvery of [0, 7]) {
                const callbacks: BvhTraverseCallbacks = {
                    intersectsBounds: (box, _isLeaf, _score, depth) => (depth === 1 ? true : answers[hash(box) % answers.length]),
                    intersectsRange: (offset, count) => (stopEvery > 0 && (offset * 31 + count) % stopEvery === 0 ? 'stop' : undefined),
                    ...(ordered ? { boundsTraverseOrder: (box: Float32Array) => box[0] + box[1] * 0.5 - box[5] } : {}),
                };
                const expected = threeShapecast(scene, callbacks);
                const log: string[] = [];
                const result = scene.traverse(logged(callbacks, log));
                expect(log, `ordered ${ordered}, stop ${stopEvery}`).toEqual(expected.log);
                expect(result).toBe(Boolean(expected.result));
                if (stopEvery === 0) expect(log.filter((call) => call.startsWith('range') && call.includes(' true ')).length, 'contained ranges').toBeGreaterThan(0);
            }
        }
    });

    it('reports a one-instance scene as its root leaf, unasked, and an empty one not at all', () => {
        const { geometries } = makeScene(82);
        const one = new CpuSceneBvh([buildMeshBvhWithTinyBvh(geometries[0], tinyBvh)], [{ meshIndex: 0, instanceIndex: 5, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] }]);
        const calls: string[] = [];
        one.traverse({ intersectsBounds: () => calls.push('bounds'), intersectsRange: (offset, count, contained, depth, nodeIndex) => calls.push(`range ${offset} ${count} ${contained} ${depth} ${nodeIndex}`) });
        expect(calls).toEqual(['range 0 1 false 0 0']);
        expect(one.instanceAt(0)).toBe(5);
        expect(() => one.instanceAt(1)).toThrow(RangeError);
        const hidden = new CpuSceneBvh([buildMeshBvhWithTinyBvh(geometries[0], tinyBvh)], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], visible: false }]);
        expect(hidden.shapecast({ intersectsBounds: () => true, intersectsInstance: () => true })).toBe(false);
    });
});

describe('CpuSceneBvh.shapecast (ObjectBVH.shapecast)', () => {
    it("finds the instances whose world box meets a sphere, as brute force does; stops when asked; contained boxes hand over every instance", () => {
        const { random, geometries, instances, scene } = makeScene(83, 120);
        const meshes = geometries.map((geometry) => buildMeshBvhWithTinyBvh(geometry, tinyBvh));
        const box = new Float32Array(6);
        const sphereMeetsBox = (center: number[], radius: number, min: { x: number; y: number; z: number }, max: { x: number; y: number; z: number }) => {
            const dx = Math.max(min.x - center[0], 0, center[0] - max.x);
            const dy = Math.max(min.y - center[1], 0, center[1] - max.y);
            const dz = Math.max(min.z - center[2], 0, center[2] - max.z);
            return dx * dx + dy * dy + dz * dz <= radius * radius;
        };
        for (let q = 0; q < 200; q++) {
            const center = [0, 1, 2].map(() => (random() - 0.5) * 30);
            const radius = 0.5 + random() * 6;
            const expected = instances
                .filter((instance) => instance.visible !== false)
                .filter((instance) => {
                    transformBounds(meshes[instance.meshIndex].bounds, instance.objectToWorld, box, 0);
                    return sphereMeetsBox(center, radius, { x: box[0], y: box[1], z: box[2] }, { x: box[3], y: box[4], z: box[5] });
                })
                .map((instance) => instance.instanceIndex)
                .sort((a, b) => a - b);
            const found: number[] = [];
            let contained = 0;
            const stopped = scene.shapecast({
                intersectsBounds: (b, isLeaf) => {
                    if (!sphereMeetsBox(center, radius, b.min, b.max)) return BVH_NOT_INTERSECTED;
                    // A box inside the sphere: its subtree whole (corners all within the radius).
                    const far = [0, 1, 2].map((k) => Math.max(Math.abs([b.min.x, b.min.y, b.min.z][k] - center[k]), Math.abs([b.max.x, b.max.y, b.max.z][k] - center[k])));
                    return !isLeaf && Math.hypot(...far) <= radius ? BVH_CONTAINED : BVH_INTERSECTED;
                },
                intersectsInstance: (instance, isContained) => {
                    if (isContained) contained++;
                    found.push(instance);
                },
            });
            expect(stopped).toBe(false);
            // Contained boxes hold every instance under them, even ones whose own box only touches the sphere's box.
            expect(found.sort((a, b) => a - b).filter((instance) => expected.includes(instance)), `query ${q}`).toEqual(expected);
            if (contained === 0) expect(found).toEqual(expected);
        }
        // A truthy intersectsInstance stops the cast; intersectsRange decides first and can stop it alone.
        let seen = 0;
        expect(scene.shapecast({ intersectsBounds: () => true, intersectsInstance: () => ++seen === 3 })).toBe(true);
        expect(seen).toBe(3);
        let instancesSeen = 0;
        expect(scene.shapecast({ intersectsBounds: () => true, intersectsRange: () => true, intersectsInstance: () => void instancesSeen++ })).toBe(true);
        expect(instancesSeen).toBe(0);
        // With neither, a cast stops at the first contained box (three-mesh-bvh's default).
        expect(scene.shapecast({ intersectsBounds: () => BVH_INTERSECTED })).toBe(false);
        expect(scene.shapecast({ intersectsBounds: () => BVH_CONTAINED })).toBe(true);
        // Every visible instance once, with boundsTraverseOrder too.
        const all: number[] = [];
        scene.shapecast({ intersectsBounds: () => true, intersectsInstance: (instance) => void all.push(instance), boundsTraverseOrder: (b) => b.min.x });
        expect(all.sort((a, b) => a - b)).toEqual(instances.filter((instance) => instance.visible !== false).map((instance) => instance.instanceIndex));
    });
});

// Code sample for docs/three/trace-nodes.mdx, "Shapecasts with TSL hooks" (see getting-started.ts for how samples are checked).

// #region shapecast-fn
import { BvhTraceNodes, type BvhTslNode } from '@pmndrs/bvh/three';
import type { SceneTrees } from '@pmndrs/bvh';
import { Fn, If, float, instanceIndex, max, select, storage, uint, vec3, wgslFn } from 'three/tsl';
import { StorageBufferAttribute, type WebGPURenderer } from 'three/webgpu';

// #endregion

export async function shapecastFn(renderer: WebGPURenderer, trees: SceneTrees, spheres: StorageBufferAttribute, count: number): Promise<Uint32Array> {
    // #region shapecast-fn
    const bvh = new BvhTraceNodes(renderer, trees, { queries: true });

    // A box test as a TSL Fn with a layout (a pure function of its arguments): the box's score, negative to skip it.
    const sphereBounds = Fn(([sphere, boxMin, boxMax]: BvhTslNode[]) => {
        const outside = max(max(boxMin.sub(sphere.xyz), sphere.xyz.sub(boxMax)), vec3(0));
        const distanceSq = outside.dot(outside);
        return select(distanceSq.lessThanEqual(sphere.w.mul(sphere.w)), distanceSq, float(-1));
    }).setLayout({ name: 'sphereBounds', type: 'float', inputs: [{ name: 'sphere', type: 'vec4' }, { name: 'boxMin', type: 'vec3' }, { name: 'boxMax', type: 'vec3' }] });

    // A triangle test as a wgslFn, which may call the module's helpers: true stops the cast.
    const sphereTriangle = wgslFn(
        `fn sphereTriangle(sphere: vec4f, a: vec3f, b: vec3f, c: vec3f) -> bool {
            let offset = bvhClosestPointToTriangle(sphere.xyz, a, b, c).point - sphere.xyz;
            return dot(offset, offset) <= sphere.w * sphere.w;
        }`,
        [bvh.module],
    );

    // Does each sphere (centre, radius) touch the scene?
    const touches = bvh.getShapecastFn({ name: 'sphereTouches', shape: 'vec4', intersectsBounds: sphereBounds, intersectsTriangle: sphereTriangle });

    const sphereNode = storage(spheres, 'vec4', count).toReadOnly();
    const out = new StorageBufferAttribute(new Uint32Array(count), 1);
    const outNode = storage(out, 'uint', count);
    const kernel = Fn(() => {
        If(instanceIndex.lessThan(count), () => {
            outNode.element(instanceIndex).assign(select(touches(sphereNode.element(instanceIndex)), uint(1), uint(0)));
        });
    })().compute(count, [64]);
    await renderer.computeAsync(kernel);
    const touching = new Uint32Array(await renderer.getArrayBufferAsync(out));
    // #endregion
    return touching;
}

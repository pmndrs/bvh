/**
 * Work per ray on compare/gpu, ours against three-mesh-bvh/webgpu, on the same
 * rays: why one walk is faster than the other (`?counts` on the page).
 *
 * - **ours**: the bench's counted copy of the production walk
 *   (tests/gpu/pages/trace-bench/counted-walk-wgsl.ts), cut out of the
 *   module's text, so its traversal order is the production walk's.
 * - **three-mesh-bvh**: its own shapecast loop (`BVHComputeData.getShapecastFn`,
 *   the same loop `bvh_RaycastAnyHit` / `bvh_RaycastFirstHit` run) with copies
 *   of its bounds and range callbacks that also count into the result struct.
 *   Any hit has no `boundsOrderFn` (its fixed order); closest hit has its
 *   split-axis order.
 *
 * Counted: box tests (two per inner node visited, both levels), leaves
 * visited, triangle tests. The times of counted kernels mean nothing.
 */

import { RAY_FLOATS, type TraceSceneBuffers } from '@pmndrs/bvh';
import { Fn, If, instanceIndex, storage, wgslFn } from 'three/tsl';
import { StorageBufferAttribute, StructTypeNode, type WebGPURenderer } from 'three/webgpu';
import * as tmbvhWebgpu from 'three-mesh-bvh/webgpu';
import type { AnyHitOrder } from '@pmndrs/bvh';
import { COUNTED_WORD, COUNTED_WORDS, countedKernelWgsl } from '../../tests/gpu/pages/trace-bench/counted-walk-wgsl.ts';
import type { GpuMode } from './gpu-trace.ts';

type BVHComputeData = tmbvhWebgpu.BVHComputeData;
// Exported by three-mesh-bvh/webgpu's index.js, but not in its index.d.ts.
const { bvhNodeBoundsStruct, bvhNodeStruct, intersectRayTriangle, rayStruct, wgslTagFn } = tmbvhWebgpu as unknown as Record<'bvhNodeBoundsStruct' | 'bvhNodeStruct' | 'intersectRayTriangle' | 'rayStruct', unknown> & {
    wgslTagFn(strings: TemplateStringsArray, ...values: unknown[]): unknown;
};

/** Means per ray over a ray set. */
export interface WorkPerRay {
    readonly boxTests: number;
    readonly leaves: number;
    readonly triangleTests: number;
    /** Share of rays that hit (closest) or are blocked (any). */
    readonly hitShare: number;
    /**
     * Box tests of the slowest ray in each run of 32 consecutive rays, averaged
     * over the runs: what a 32-wide SIMD group waits for (rays are in seed
     * order, so a run is 32 unrelated rays).
     */
    readonly simdMaxBoxTests: number;
}

const SIMD_WIDTH = 32;

function simdMax(perRay: Float32Array): number {
    let sum = 0;
    let groups = 0;
    for (let start = 0; start < perRay.length; start += SIMD_WIDTH) {
        let max = 0;
        for (let i = start; i < Math.min(perRay.length, start + SIMD_WIDTH); i++) max = Math.max(max, perRay[i]);
        sum += max;
        groups++;
    }
    return sum / groups;
}

async function readBuffer(device: GPUDevice, buffer: GPUBuffer, bytes: number): Promise<ArrayBuffer> {
    const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, read, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const out = read.getMappedRange().slice(0);
    read.unmap();
    read.destroy();
    return out;
}

/** Our counted walk over `packed` rays: work per ray. */
export async function countOurs(device: GPUDevice, scene: TraceSceneBuffers, mode: GpuMode, packed: Float32Array, anyHitOrder?: AnyHitOrder): Promise<WorkPerRay> {
    const count = packed.length / RAY_FLOATS;
    const pipeline = await device.createComputePipelineAsync({
        label: `compare counted ${mode}`,
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: countedKernelWgsl(mode, { anyHitOrder }) }), entryPoint: 'benchCounted' },
    });
    const rays = device.createBuffer({ size: packed.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(rays, 0, packed as Float32Array<ArrayBuffer>);
    const bytes = count * COUNTED_WORDS * 4;
    const output = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(
        0,
        device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: scene.nodes } },
                { binding: 1, resource: { buffer: scene.triangles } },
                { binding: 2, resource: { buffer: scene.topLevel } },
            ],
        }),
    );
    pass.setBindGroup(1, device.createBindGroup({ layout: pipeline.getBindGroupLayout(1), entries: [{ binding: 0, resource: { buffer: rays } }, { binding: 1, resource: { buffer: output } }] }));
    const workgroups = Math.ceil(count / 64);
    const x = Math.min(workgroups, device.limits.maxComputeWorkgroupsPerDimension);
    pass.dispatchWorkgroups(x, Math.ceil(workgroups / x));
    pass.end();
    device.queue.submit([encoder.finish()]);
    const words = new Uint32Array(await readBuffer(device, output, bytes));
    const floats = new Float32Array(words.buffer);
    rays.destroy();
    output.destroy();
    let boxTests = 0;
    let leaves = 0;
    let triangles = 0;
    let hits = 0;
    const perRay = new Float32Array(count);
    for (let i = 0; i < count; i++) {
        const base = i * COUNTED_WORDS;
        const nodes = words[base + COUNTED_WORD.meshNodes] + words[base + COUNTED_WORD.topLevelNodes];
        boxTests += 2 * nodes;
        perRay[i] = 2 * nodes;
        // A step is a node, a leaf or an instance entry; leaves are what is left after nodes and entries.
        leaves += words[base + COUNTED_WORD.steps] - nodes - words[base + COUNTED_WORD.instanceEntries];
        triangles += words[base + COUNTED_WORD.triangleTests];
        if (floats[base + COUNTED_WORD.t] >= 0) hits++;
    }
    return { boxTests: boxTests / count, leaves: leaves / count, triangleTests: triangles / count, hitShare: hits / count, simdMaxBoxTests: simdMax(perRay) };
}

const countsStruct = new StructTypeNode({ boxTests: 'uint', leaves: 'uint', triangleTests: 'uint', dist: 'float' }, 'CompareCounts');

/** three-mesh-bvh's walk with counters, over `packed` rays: work per ray. */
export async function countTheirs(renderer: WebGPURenderer, bvhData: BVHComputeData, mode: GpuMode, packed: Float32Array): Promise<WorkPerRay> {
    const { index, attributes } = bvhData.storage as unknown as Record<'index' | 'attributes', unknown>;
    const closest = mode === 'closest';
    // The bounds test of its raycastAnyHit / raycastFirstHit (one mesh, identity transform: no ray scalar), counting.
    const intersectsBoundsFn = wgslTagFn/* wgsl */ `
        fn compareCountedBounds( ray: ${rayStruct}, bounds: ${bvhNodeBoundsStruct}, result: ptr<function, ${countsStruct}> ) -> u32 {
            result.boxTests += 1u;
            let boundsMin = vec3( bounds.min[0], bounds.min[1], bounds.min[2] );
            let boundsMax = vec3( bounds.max[0], bounds.max[1], bounds.max[2] );
            let parallel = ray.direction == vec3f( 0.0 );
            if ( any( parallel & ( ( ray.origin < boundsMin ) | ( ray.origin > boundsMax ) ) ) ) { return 0u; }
            let invDir = 1.0 / select( ray.direction, vec3f( 1.0 ), parallel );
            let tMinPlane = ( boundsMin - ray.origin ) * invDir;
            let tMaxPlane = ( boundsMax - ray.origin ) * invDir;
            let tMinHit = select( min( tMinPlane, tMaxPlane ), vec3f( - 3.402823e38 ), parallel );
            let tMaxHit = select( max( tMinPlane, tMaxPlane ), vec3f( 3.402823e38 ), parallel );
            let t0 = max( max( tMinHit.x, tMinHit.y ), tMinHit.z );
            let t1 = min( min( tMaxHit.x, tMaxHit.y ), tMaxHit.z );
            let near = max( t0, 0.0 );
            let dist = select( near, - 1.0, t1 < near );
            ${closest ? 'return u32( dist >= 0.0 && ( ray.maxDist <= 0.0 || dist < ray.maxDist ) && ( result.dist < 0.0 || dist < result.dist ) );' : 'return u32( dist >= 0.0 && ( ray.maxDist <= 0.0 || dist < ray.maxDist ) );'}
        }
    `;
    const intersectRangeFn = wgslTagFn/* wgsl */ `
        fn compareCountedRange( ray: ${rayStruct}, offset: u32, count: u32, result: ptr<function, ${countsStruct}> ) -> u32 {
            result.leaves += 1u;
            var found = 0u;
            for ( var ti = offset; ti < offset + count; ti = ti + 1u ) {
                result.triangleTests += 1u;
                let a = ${attributes}[ ${index}[ ti * 3u ] ].position.xyz;
                let b = ${attributes}[ ${index}[ ti * 3u + 1u ] ].position.xyz;
                let c = ${attributes}[ ${index}[ ti * 3u + 2u ] ].position.xyz;
                let hit = ${intersectRayTriangle}( ray, a, b, c, 0.0 );
                if ( hit.didHit && ( ray.maxDist <= 0.0 || hit.dist < ray.maxDist ) ) {
                    ${closest ? 'if ( result.dist < 0.0 || hit.dist < result.dist ) { result.dist = hit.dist; found = 1u; }' : 'result.dist = hit.dist;\n                    return 3u;'}
                }
            }
            return found;
        }
    `;
    const boundsOrderFn = closest
        ? wgslTagFn/* wgsl */ `
        fn compareCountedOrder( ray: ${rayStruct}, splitAxis: u32, node: ${bvhNodeStruct} ) -> bool {
            return ray.direction[ splitAxis ] >= 0.0;
        }
    `
        : null;
    const walk = (bvhData as unknown as { getShapecastFn(options: object): unknown }).getShapecastFn({
        name: `compare_counted_${mode}`,
        shapeStruct: rayStruct,
        resultStruct: countsStruct,
        boundsOrderFn,
        intersectsBoundsFn,
        intersectRangeFn,
    });
    const trace = wgslFn(
        /* wgsl */ `
        fn compareCountedTrace( origin: vec3f, direction: vec3f, maxDist: f32 ) -> vec4f {
            var ray: Ray;
            ray.origin = origin;
            ray.direction = direction;
            ray.maxDist = maxDist;
            var counts: CompareCounts;
            counts.dist = -1.0;
            compare_counted_${mode}( ray, &counts );
            return vec4f( counts.dist, f32( counts.boxTests ), f32( counts.leaves ), f32( counts.triangleTests ) );
        }
    `,
        [walk as never],
    );
    const count = packed.length / RAY_FLOATS;
    const input = new StorageBufferAttribute(packed, 4);
    const results = new StorageBufferAttribute(count, 4);
    const rays = storage(input, 'vec4', count * 2).toReadOnly();
    const output = storage(results, 'vec4', count);
    const kernel = Fn(() => {
        If(instanceIndex.lessThan(count), () => {
            const originAndMin = rays.element(instanceIndex.mul(2));
            const directionAndMax = rays.element(instanceIndex.mul(2).add(1));
            output.element(instanceIndex).assign(trace({ origin: originAndMin.xyz, direction: directionAndMax.xyz, maxDist: directionAndMax.w }));
        });
    })().compute(count, [64]);
    await renderer.computeAsync(kernel);
    const vec4s = new Float32Array(await renderer.getArrayBufferAsync(results));
    kernel.dispose();
    input.dispose();
    results.dispose();
    let boxTests = 0;
    let leaves = 0;
    let triangles = 0;
    let hits = 0;
    const perRay = new Float32Array(count);
    for (let i = 0; i < count; i++) {
        if (vec4s[i * 4] >= 0) hits++;
        perRay[i] = vec4s[i * 4 + 1];
        boxTests += vec4s[i * 4 + 1];
        leaves += vec4s[i * 4 + 2];
        triangles += vec4s[i * 4 + 3];
    }
    return { boxTests: boxTests / count, leaves: leaves / count, triangleTests: triangles / count, hitShare: hits / count, simdMaxBoxTests: simdMax(perRay) };
}

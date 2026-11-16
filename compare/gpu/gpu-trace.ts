/**
 * GPU traversal, ours against three-mesh-bvh's `/webgpu` (compare/gpu and the
 * GPU batch on compare/raycast), on one device: three's `WebGPURenderer`'s.
 *
 * - **ours**: the mesh's tinybvh tree assembled as a one-instance scene,
 *   uploaded with `uploadTraceScene` and traced by `TraceBatchKernel` (plain
 *   WebGPU, no three) in its leanest form: no ray masks, the 24-byte compact
 *   hit record. Closest hit writes the record, any hit one `u32`.
 * - **three-mesh-bvh**: a `MeshBVH` with SAH (its default CENTER would make it
 *   slower), `BVHComputeData` over the mesh, and a TSL compute kernel that
 *   calls its `bvh_RaycastFirstHit` / `bvh_RaycastAnyHit` and writes one
 *   `vec4f` per ray (as tests/gpu/pages/trace-bench/tmbvh-bench.ts does).
 *
 * Both read the same packed rays (origin, tMin, direction, tMax). Timing:
 * GPU timestamps of the one compute pass where the device has
 * `timestamp-query` (ours: our own query set; three-mesh-bvh: three's
 * `trackTimestamp`, resolved after each dispatch), else wall clock from
 * submit to `onSubmittedWorkDone`.
 */

import {
    assembleTraceScene,
    buildMeshBvhWithTinyBvh,
    HIT_RECORD_COMPACT_BYTES,
    HIT_RECORD_WORD,
    RAY_FLOATS,
    TraceBatchKernel,
    uploadTraceScene,
    type AnyHitOrder,
    type NodeFormat,
    type TraceSceneBuffers,
} from '@pmndrs/bvh';
import { BufferAttribute, BufferGeometry, Mesh, MeshBasicMaterial } from 'three';
import { Fn, If, instanceIndex, storage, wgslFn } from 'three/tsl';
import { StorageBufferAttribute, type WebGPURenderer } from 'three/webgpu';
import { MeshBVH, SAH } from 'three-mesh-bvh';
import { BVHComputeData } from 'three-mesh-bvh/webgpu';
import { oursTinyBvh } from '../adapters/ours.ts';
import type { CompareGeometry, CompareRay } from '../adapters/types.ts';
import { countOurs, countTheirs, type WorkPerRay } from './gpu-counts.ts';

export type GpuMode = 'closest' | 'any';

export interface GpuTracer {
    /** `ours` (the library's default kernel), `ours-larger` (`anyHitOrder: 'larger'`, any-hit only differs) or `tmbvh`. */
    readonly id: 'ours' | 'ours-larger' | 'tmbvh';
    readonly name: string;
    /** GPU bytes of the structure the walk reads. */
    readonly bytes: number;
    readonly bytesMethod: string;
    readonly timing: 'timestamps' | 'wall clock';
    /** Uploads rays for a mode (once per ray set). */
    setRays(mode: GpuMode, packed: Float32Array): void;
    /** One dispatch over the mode's rays: its milliseconds. */
    dispatch(mode: GpuMode): Promise<number>;
    /** The last dispatch's answers: `t` per ray (negative for a miss), or 1/0 blocked. */
    read(mode: GpuMode): Promise<Float32Array>;
    /** The walk's work per ray on the mode's rays, from a counted copy of the walk (gpu-counts.ts). */
    counts(mode: GpuMode): Promise<WorkPerRay>;
    dispose(): void;
}

/** Rays packed as the kernels read them (`RAY_FLOATS` per ray), with a `tMax`. */
export function packRaysFor(rays: readonly CompareRay[], tMax: number): Float32Array {
    const out = new Float32Array(rays.length * RAY_FLOATS);
    rays.forEach((ray, i) => out.set([ray.origin[0], ray.origin[1], ray.origin[2], 0, ray.direction[0], ray.direction[1], ray.direction[2], tMax], i * RAY_FLOATS));
    return out;
}

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

export interface OursGpuOptions {
    /** The kernel's any-hit child order (`TraceBatchKernelOptions.anyHitOrder`). Default: the library's (`'far'`). */
    readonly anyHitOrder?: AnyHitOrder;
    /** The scene's and the kernel's node format. Default: the library's (quantized since 2026-11-13). */
    readonly nodeFormat?: NodeFormat;
}

export async function createOursGpu(device: GPUDevice, geometry: CompareGeometry, options: OursGpuOptions = {}): Promise<GpuTracer> {
    const { anyHitOrder, nodeFormat } = options;
    const tinyBvh = await oursTinyBvh();
    const tree = buildMeshBvhWithTinyBvh({ positions: geometry.positions, indices: geometry.indices }, tinyBvh);
    const scene = assembleTraceScene([tree], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY }], {}, nodeFormat ? { nodeFormat } : {});
    const buffers: TraceSceneBuffers = uploadTraceScene(device, scene, 'compare ours');
    const kernel = await TraceBatchKernel.create(device, { label: 'compare ours', rayMasks: false, hitNormal: false, anyHitOrder, nodeFormat });
    const timestamps = device.features.has('timestamp-query');
    const querySet = timestamps ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
    const resolve = timestamps ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
    const stampRead = timestamps ? device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }) : null;
    const sets = new Map<GpuMode, { rays: GPUBuffer; results: GPUBuffer; read: GPUBuffer; count: number; bytes: number; packed: Float32Array }>();
    return {
        id: anyHitOrder === 'larger' ? 'ours-larger' : 'ours',
        name: `@pmndrs/bvh (TraceBatchKernel${anyHitOrder ? `, anyHitOrder '${anyHitOrder}'` : ''}${nodeFormat ? `, ${nodeFormat} nodes` : ''})`,
        bytes: buffers.byteLength,
        bytesMethod: 'TraceSceneBuffers.byteLength: nodes, triangle records and the one-instance top level',
        timing: timestamps ? 'timestamps' : 'wall clock',
        setRays(mode, packed) {
            const old = sets.get(mode);
            if (old) [old.rays, old.results, old.read].forEach((buffer) => buffer.destroy());
            const count = packed.length / RAY_FLOATS;
            const bytes = count * (mode === 'closest' ? HIT_RECORD_COMPACT_BYTES : 4);
            const rays = device.createBuffer({ label: 'compare rays', size: packed.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
            device.queue.writeBuffer(rays, 0, packed as Float32Array<ArrayBuffer>);
            const results = device.createBuffer({ label: 'compare hits', size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
            const read = device.createBuffer({ label: 'compare hits read', size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
            sets.set(mode, { rays, results, read, count, bytes, packed });
        },
        async dispatch(mode) {
            const set = sets.get(mode)!;
            const encoder = device.createCommandEncoder({ label: 'compare ours' });
            const timestampWrites = querySet ? { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } : undefined;
            kernel.encode(encoder, buffers, { mode, rays: set.rays, results: set.results, count: set.count, timestampWrites });
            if (querySet) {
                encoder.resolveQuerySet(querySet, 0, 2, resolve!, 0);
                encoder.copyBufferToBuffer(resolve!, 0, stampRead!, 0, 16);
            }
            const start = performance.now();
            device.queue.submit([encoder.finish()]);
            if (!querySet) {
                await device.queue.onSubmittedWorkDone();
                return performance.now() - start;
            }
            await stampRead!.mapAsync(GPUMapMode.READ);
            const stamps = new BigInt64Array(stampRead!.getMappedRange().slice(0));
            stampRead!.unmap();
            return Number(stamps[1] - stamps[0]) / 1e6;
        },
        async read(mode) {
            const set = sets.get(mode)!;
            const encoder = device.createCommandEncoder();
            encoder.copyBufferToBuffer(set.results, 0, set.read, 0, set.bytes);
            device.queue.submit([encoder.finish()]);
            await set.read.mapAsync(GPUMapMode.READ);
            const bytes = set.read.getMappedRange().slice(0);
            set.read.unmap();
            const out = new Float32Array(set.count);
            if (mode === 'closest') {
                const floats = new Float32Array(bytes);
                const stride = HIT_RECORD_COMPACT_BYTES / 4;
                for (let i = 0; i < set.count; i++) out[i] = floats[i * stride + HIT_RECORD_WORD.t];
            } else {
                const words = new Uint32Array(bytes);
                for (let i = 0; i < set.count; i++) out[i] = words[i];
            }
            return out;
        },
        counts(mode) {
            return countOurs(device, buffers, mode, sets.get(mode)!.packed, anyHitOrder);
        },
        dispose() {
            for (const set of sets.values()) [set.rays, set.results, set.read].forEach((buffer) => buffer.destroy());
            buffers.destroy();
            kernel.destroy();
            querySet?.destroy();
            resolve?.destroy();
            stampRead?.destroy();
        },
    };
}

function storageBytes(node: unknown): number {
    return (node as { proxyNode: { value: { array: { byteLength: number } } } }).proxyNode.value.array.byteLength;
}

function tmbvhKernel(bvhData: BVHComputeData, mode: GpuMode, input: StorageBufferAttribute, results: StorageBufferAttribute, count: number) {
    const fns = bvhData.fns as unknown as Record<string, unknown>;
    const trace =
        mode === 'closest'
            ? wgslFn(
                  /* wgsl */ `
                  fn traceClosest( origin: vec3f, direction: vec3f, maxDist: f32 ) -> vec4f {
                      var ray: Ray;
                      ray.origin = origin;
                      ray.direction = direction;
                      ray.maxDist = maxDist;
                      var hit: IntersectionResult;
                      bvh_RaycastFirstHit( ray, &hit );
                      return vec4f( select( -1.0, hit.dist, hit.didHit ), 0.0, 0.0, 0.0 );
                  }
              `,
                  [fns.raycastFirstHit as never],
              )
            : wgslFn(
                  /* wgsl */ `
                  fn traceAny( origin: vec3f, direction: vec3f, maxDist: f32 ) -> vec4f {
                      var ray: Ray;
                      ray.origin = origin;
                      ray.direction = direction;
                      ray.maxDist = maxDist;
                      return vec4f( select( 0.0, 1.0, bvh_RaycastAnyHit( ray ) ), 0.0, 0.0, 0.0 );
                  }
              `,
                  [fns.raycastAnyHit as never],
              );
    const rays = storage(input, 'vec4', count * 2).toReadOnly();
    const output = storage(results, 'vec4', count);
    return Fn(() => {
        If(instanceIndex.lessThan(count), () => {
            const originAndMin = rays.element(instanceIndex.mul(2));
            const directionAndMax = rays.element(instanceIndex.mul(2).add(1));
            output.element(instanceIndex).assign(trace({ origin: originAndMin.xyz, direction: directionAndMax.xyz, maxDist: directionAndMax.w }));
        });
    })().compute(count, [64]);
}

/** three-mesh-bvh on the renderer (made with `trackTimestamp: true` for GPU times). */
export async function createTmbvhGpu(renderer: WebGPURenderer, geometry: CompareGeometry): Promise<GpuTracer> {
    const three = new BufferGeometry();
    three.setAttribute('position', new BufferAttribute(geometry.positions, 3));
    three.setIndex(new BufferAttribute(geometry.indices.slice(), 1));
    (three as unknown as { boundsTree: MeshBVH }).boundsTree = new MeshBVH(three, { strategy: SAH, verbose: false });
    const material = new MeshBasicMaterial();
    const mesh = new Mesh(three, material);
    mesh.updateMatrixWorld();
    const bvhData = new BVHComputeData([mesh], { attributes: { position: 'vec4f' }, autogenerateBvh: false });
    bvhData.update();
    const nodes = bvhData.storage as unknown as Record<'nodes' | 'index' | 'attributes' | 'transforms', unknown>;
    const bytes = storageBytes(nodes.nodes) + storageBytes(nodes.index) + storageBytes(nodes.attributes) + storageBytes(nodes.transforms);
    const timestamps = renderer.hasFeature('timestamp-query') && (renderer as unknown as { backend: { trackTimestamp: boolean } }).backend.trackTimestamp;
    const kernels = new Map<GpuMode, { node: ReturnType<typeof tmbvhKernel>; input: StorageBufferAttribute; results: StorageBufferAttribute; count: number; compiled: boolean; packed: Float32Array }>();
    return {
        id: 'tmbvh',
        name: 'three-mesh-bvh/webgpu (BVHComputeData, SAH)',
        bytes,
        bytesMethod: "byteLength of BVHComputeData's storage: nodes, index, vec4f positions and transforms (its GPU copy of the positions included: the walk reads that copy)",
        timing: timestamps ? 'timestamps' : 'wall clock',
        setRays(mode, packed) {
            const old = kernels.get(mode);
            if (old) {
                old.node.dispose();
                old.input.dispose();
                old.results.dispose();
            }
            const count = packed.length / RAY_FLOATS;
            const input = new StorageBufferAttribute(packed, 4);
            const results = new StorageBufferAttribute(count, 4);
            kernels.set(mode, { node: tmbvhKernel(bvhData, mode, input, results, count), input, results, count, compiled: false, packed });
        },
        async dispatch(mode) {
            const kernel = kernels.get(mode)!;
            if (!kernel.compiled) {
                // The first dispatch compiles the pipeline and uploads the buffers (outside any timing).
                await renderer.computeAsync(kernel.node);
                if (timestamps) await renderer.resolveTimestampsAsync('compute');
                kernel.compiled = true;
            }
            if (timestamps) {
                renderer.compute(kernel.node);
                const ms = await renderer.resolveTimestampsAsync('compute');
                if (typeof ms !== 'number') throw new Error('three returned no compute timestamp');
                return ms;
            }
            const start = performance.now();
            await renderer.computeAsync(kernel.node);
            await (renderer as unknown as { backend: { device: GPUDevice } }).backend.device.queue.onSubmittedWorkDone();
            return performance.now() - start;
        },
        async read(mode) {
            const kernel = kernels.get(mode)!;
            const vec4s = new Float32Array(await renderer.getArrayBufferAsync(kernel.results));
            const out = new Float32Array(kernel.count);
            for (let i = 0; i < kernel.count; i++) out[i] = vec4s[i * 4];
            return out;
        },
        counts(mode) {
            return countTheirs(renderer, bvhData, mode, kernels.get(mode)!.packed);
        },
        dispose() {
            for (const kernel of kernels.values()) {
                kernel.node.dispose();
                kernel.input.dispose();
                kernel.results.dispose();
            }
            (bvhData as unknown as { dispose(): void }).dispose();
            three.dispose();
            material.dispose();
        },
    };
}

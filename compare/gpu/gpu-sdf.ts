/**
 * SDF generation on the GPU, ours against three-mesh-bvh's `webgpu_sdfGeneration`
 * (compare/sdf), on one device: three's `WebGPURenderer`'s.
 *
 * - **ours**: the mesh's tinybvh tree as a one-instance scene (`uploadTraceScene`),
 *   `SdfGenerator` (plain WebGPU, no ray masks) writing an `r32float` 3D
 *   storage texture: per voxel `bvhClosestPoint` and, signed, the inside test's
 *   three all-hit rays and a parity vote. `ours-unsigned` is the closest point alone.
 * - **three-mesh-bvh**: its example's kernel as published (v0.9.18,
 *   `example/webgpu_sdfGeneration.js`, MIT): a `wgslFn` `computeSdf` over
 *   `BVHComputeData` (`bvh_ClosestPointToPoint`, then `bvh_RaycastFirstHit` along
 *   +Z, the hit's `side` the sign) in 4 x 4 x 4 workgroups, writing a three
 *   `Storage3DTexture` (`RedFormat`, `FloatType`). Its tree is built with SAH
 *   (its default is CENTER), as on compare/gpu.
 *
 * Both fill the same grid: `dim`^3 voxel centres over the mesh's box grown by
 * a margin. Their kernel maps a voxel with `vec4f(p, 1) * matrix`, a row
 * vector times the box's compose matrix, which is right only for a box centred
 * on the origin (its example centres the model); so the geometry is moved to
 * centre its box on the origin for both. Both get the same welded geometry
 * (three's knot repeats its seam vertices; a crack lets rays through).
 *
 * Timing: GPU timestamps of the one dispatch where the device has
 * `timestamp-query` (ours: a query set on the pass; theirs: three's
 * `trackTimestamp`), else wall clock from submit to `onSubmittedWorkDone`
 * (their example's own measure).
 */

import { assembleTraceScene, buildMeshBvhWithTinyBvh, createSdfTexture, readSdf, SdfGenerator, uploadTraceScene, type SdfGrid, type SdfSign } from '@pmndrs/bvh';
import { BufferAttribute, BufferGeometry, ClampToEdgeWrapping, FloatType, Matrix4, Mesh, MeshBasicMaterial, Quaternion, RedFormat, Vector3 } from 'three';
import { globalId, storageTexture, uniform, wgslFn } from 'three/tsl';
import { Storage3DTexture, type WebGPURenderer } from 'three/webgpu';
import { MeshBVH, SAH } from 'three-mesh-bvh';
import { BVHComputeData } from 'three-mesh-bvh/webgpu';
import { oursTinyBvh } from '../adapters/ours.ts';
import type { CompareGeometry } from '../adapters/types.ts';

export interface GpuSdf {
    readonly id: 'ours' | 'ours-unsigned' | 'tmbvh';
    readonly name: string;
    readonly timing: 'timestamps' | 'wall clock';
    /** One generation of a `dim`^3 field over the box: its milliseconds. */
    generate(dim: number): Promise<number>;
    /** The last field, x fastest, then y, then z. */
    read(): Promise<Float32Array>;
    dispose(): void;
}

/** The geometry with exactly equal positions merged into one vertex, and its box centred on the origin. */
export function weldedAndCentred(geometry: CompareGeometry): { positions: Float32Array; indices: Uint32Array; center: [number, number, number] } {
    const b = geometry.bounds;
    const center: [number, number, number] = [(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2];
    const map = new Map<string, number>();
    const remap = new Uint32Array(geometry.positions.length / 3);
    const out: number[] = [];
    for (let v = 0; v < remap.length; v++) {
        const p = geometry.positions.subarray(v * 3, v * 3 + 3);
        const key = `${p[0]},${p[1]},${p[2]}`;
        let index = map.get(key);
        if (index === undefined) {
            index = out.length / 3;
            map.set(key, index);
            out.push(p[0] - center[0], p[1] - center[1], p[2] - center[2]);
        }
        remap[v] = index;
    }
    return { positions: new Float32Array(out), indices: Uint32Array.from(geometry.indices, (i) => remap[i]), center };
}

/** The field's box: the centred mesh's box grown by `margin` of its diagonal on every side. */
export function sdfBox(positions: Float32Array, margin: number): { min: [number, number, number]; max: [number, number, number] } {
    const min: [number, number, number] = [Infinity, Infinity, Infinity];
    const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < positions.length; i += 3) {
        for (let k = 0; k < 3; k++) {
            min[k] = Math.min(min[k], positions[i + k]);
            max[k] = Math.max(max[k], positions[i + k]);
        }
    }
    const grow = Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) * margin;
    // Symmetric about the origin, as their kernel needs (see the file comment).
    for (let k = 0; k < 3; k++) {
        const half = Math.max(-min[k], max[k]) + grow;
        min[k] = -half;
        max[k] = half;
    }
    return { min, max };
}

export async function createOursSdf(device: GPUDevice, mesh: { positions: Float32Array; indices: Uint32Array }, box: { min: [number, number, number]; max: [number, number, number] }, sign: SdfSign): Promise<GpuSdf> {
    const tree = buildMeshBvhWithTinyBvh({ positions: mesh.positions, indices: mesh.indices }, await oursTinyBvh());
    const scene = uploadTraceScene(device, assembleTraceScene([tree], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] }]), 'compare sdf');
    const generator = await SdfGenerator.create(device, { label: 'compare sdf', rayMasks: false });
    const timestamps = device.features.has('timestamp-query');
    const querySet = timestamps ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
    const resolve = timestamps ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
    const stampRead = timestamps ? device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }) : null;
    let texture: GPUTexture | null = null;
    let grid: SdfGrid | null = null;
    return {
        id: sign === 'unsigned' ? 'ours-unsigned' : 'ours',
        name: `@pmndrs/bvh (SdfGenerator, ${sign === 'unsigned' ? 'unsigned' : `${sign}, 3 rays`})`,
        timing: timestamps ? 'timestamps' : 'wall clock',
        async generate(dim) {
            if (!grid || grid.resolution !== dim) {
                texture?.destroy();
                grid = { ...box, resolution: dim };
                texture = createSdfTexture(device, grid, 'compare sdf');
            }
            const encoder = device.createCommandEncoder({ label: 'compare sdf' });
            const timestampWrites = querySet ? { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } : undefined;
            generator.encode(encoder, scene, { texture: texture! }, grid, { sign, timestampWrites });
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
        read: () => readSdf(device, { texture: texture! }, grid!),
        dispose() {
            texture?.destroy();
            scene.destroy();
            generator.destroy();
            querySet?.destroy();
            resolve?.destroy();
            stampRead?.destroy();
        },
    };
}

const WORKGROUP_SIZE = [4, 4, 4];

/** three-mesh-bvh's example kernel, as published (see the file comment). */
const COMPUTE_SDF_WGSL = /* wgsl */ `

		fn computeSdf(
			matrix: mat4x4f,
			dim: u32,
			globalId: vec3u,

			output: texture_storage_3d<r32float, write>,
		) -> void {

			if ( globalId.x >= dim || globalId.y >= dim || globalId.z >= dim ) {

				return;

			}

			let pxWidth = 1.0 / f32( dim );
			let halfWidth = 0.5 * pxWidth;
			let pointHomo = vec4f(
				halfWidth + f32( globalId.x ) * pxWidth - 0.5,
				halfWidth + f32( globalId.y ) * pxWidth - 0.5,
				halfWidth + f32( globalId.z ) * pxWidth - 0.5,
				1.0
			) * matrix;
			let point = pointHomo.xyz / pointHomo.w;

			var pointResult: PointQueryResult;
			bvh_ClosestPointToPoint( point, &pointResult );

			var rayResult: IntersectionResult;
			let ray = Ray( point, vec3f( 0.0, 0.0, 1.0 ), 0.0 );
			bvh_RaycastFirstHit( ray, &rayResult );

			let side = select( 1.0, rayResult.side, rayResult.didHit );
			let value = side * sqrt( pointResult.distanceSq );

			textureStore( output, globalId, vec4f( value, 0.0, 0.0, 0.0 ) );

		}

	`;

export async function createTmbvhSdf(renderer: WebGPURenderer, mesh: { positions: Float32Array; indices: Uint32Array }, box: { min: [number, number, number]; max: [number, number, number] }): Promise<GpuSdf> {
    const three = new BufferGeometry();
    three.setAttribute('position', new BufferAttribute(mesh.positions, 3));
    three.setIndex(new BufferAttribute(mesh.indices.slice(), 1));
    (three as unknown as { boundsTree: MeshBVH }).boundsTree = new MeshBVH(three, { strategy: SAH, verbose: false });
    const material = new MeshBasicMaterial();
    const object = new Mesh(three, material);
    object.updateMatrixWorld();
    const bvhData = new BVHComputeData([object], { attributes: { position: 'vec4f' }, autogenerateBvh: false });
    bvhData.update();
    const fns = bvhData.fns as unknown as Record<string, never>;

    // The example's updateSDF: the box as a compose matrix (centre, no rotation, size).
    const matrix = new Matrix4().compose(
        new Vector3((box.min[0] + box.max[0]) / 2, (box.min[1] + box.max[1]) / 2, (box.min[2] + box.max[2]) / 2),
        new Quaternion(),
        new Vector3(box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2]),
    );
    const makeTexture = (dim: number) => {
        const texture = new Storage3DTexture(dim, dim, dim);
        texture.format = RedFormat;
        texture.type = FloatType;
        texture.generateMipmaps = false;
        texture.needsUpdate = true;
        texture.wrapR = texture.wrapS = texture.wrapT = ClampToEdgeWrapping;
        return texture;
    };
    let dim = 0;
    let texture = makeTexture(1);
    const parameters = { matrix: uniform(matrix), dim: uniform(0, 'uint'), globalId, output: storageTexture(texture) };
    const computeShader = wgslFn(COMPUTE_SDF_WGSL, [fns.closestPointToPoint, fns.raycastFirstHit]);
    const kernel = computeShader(parameters as never).computeKernel(WORKGROUP_SIZE);
    const node = kernel as unknown as { computeNode: { parameters: { output: { value: unknown }; dim: { value: number } } } };
    const timestamps = renderer.hasFeature('timestamp-query') && (renderer as unknown as { backend: { trackTimestamp: boolean } }).backend.trackTimestamp;
    const backend = (renderer as unknown as { backend: { device: GPUDevice; get(object: object): { texture?: GPUTexture } } }).backend;
    let compiled = false;
    return {
        id: 'tmbvh',
        name: 'three-mesh-bvh/webgpu (its webgpu_sdfGeneration kernel, SAH)',
        timing: timestamps ? 'timestamps' : 'wall clock',
        async generate(next) {
            if (next !== dim) {
                texture.dispose();
                texture = makeTexture(next);
                dim = next;
                node.computeNode.parameters.output.value = texture;
                node.computeNode.parameters.dim.value = dim;
                compiled = false;
            }
            const dispatchSize = WORKGROUP_SIZE.map((size) => Math.ceil(dim / size));
            if (!compiled) {
                // The first dispatch compiles the pipeline and uploads the buffers (outside any timing).
                await renderer.computeAsync(kernel, dispatchSize);
                if (timestamps) await renderer.resolveTimestampsAsync('compute');
                await backend.device.queue.onSubmittedWorkDone();
                compiled = true;
            }
            if (timestamps) {
                renderer.compute(kernel, dispatchSize);
                const ms = await renderer.resolveTimestampsAsync('compute');
                if (typeof ms !== 'number') throw new Error('three returned no compute timestamp');
                return ms;
            }
            const start = performance.now();
            renderer.compute(kernel, dispatchSize);
            await backend.device.queue.onSubmittedWorkDone();
            return performance.now() - start;
        },
        read() {
            const gpuTexture = backend.get(texture).texture;
            if (!gpuTexture) throw new Error('three made no GPU texture for the field');
            return readSdf(backend.device, { texture: gpuTexture }, { resolution: dim });
        },
        dispose() {
            texture.dispose();
            kernel.dispose();
            (bvhData as unknown as { dispose(): void }).dispose();
            three.dispose();
            material.dispose();
        },
    };
}

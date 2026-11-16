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

import {
    assembleTraceScene,
    buildMeshBvhWithTinyBvh,
    CLOSEST_POINT_MAX_DISTANCE,
    createSdfTexture,
    mergeStaticGeometry,
    readSdf,
    SDF_PARAMS_BYTES,
    SDF_WORKGROUP_SIZE,
    SdfGenerator,
    traceModuleWgsl,
    uploadTraceScene,
    type SdfGrid,
    type SdfSign,
    type TraceSceneBuffers,
} from '@pmndrs/bvh';
import { BufferAttribute, BufferGeometry, ClampToEdgeWrapping, FloatType, Matrix4, Mesh, MeshBasicMaterial, Quaternion, RedFormat, Vector3 } from 'three';
import { globalId, storageTexture, uniform, wgslFn } from 'three/tsl';
import { Storage3DTexture, type WebGPURenderer } from 'three/webgpu';
import { MeshBVH, SAH } from 'three-mesh-bvh';
import { BVHComputeData } from 'three-mesh-bvh/webgpu';
import { oursTinyBvh } from '../adapters/ours.ts';
import type { CompareGeometry } from '../adapters/types.ts';

export interface GpuSdf {
    readonly id: string;
    readonly name: string;
    readonly timing: 'timestamps' | 'wall clock';
    /**
     * One generation of a `dim`^3 field over the box: its milliseconds. With `repeat`
     * above 1, that many dispatches back to back (the GPU at its loaded clock), the
     * mean of their times.
     */
    generate(dim: number, repeat?: number): Promise<number>;
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

type Box = { min: [number, number, number]; max: [number, number, number] };
type WeldedMesh = { positions: Float32Array; indices: Uint32Array };

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** The mesh's tinybvh tree as a one-instance scene, or (`merged`) merged into a world-space tree as a static instance. */
async function oursScene(device: GPUDevice, mesh: WeldedMesh, merged = false): Promise<TraceSceneBuffers> {
    const tiny = await oursTinyBvh();
    if (!merged) return uploadTraceScene(device, assembleTraceScene([buildMeshBvhWithTinyBvh(mesh, tiny)], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY }]), 'compare sdf');
    const instances = [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY, static: true }];
    const plan = mergeStaticGeometry([mesh], instances);
    if (plan.groups.length !== 1) throw new Error('compare sdf: the mesh did not merge');
    const trees = plan.groups.map((group) => buildMeshBvhWithTinyBvh(group.geometry, tiny));
    return uploadTraceScene(device, assembleTraceScene([null], instances, {}, { merged: { plan, meshes: trees } }), 'compare sdf merged');
}

/**
 * Times `encode` (one dispatch writing the field): `repeat` passes back to back in
 * one submit, each with its own timestamps, their mean; wall clock from submit to
 * `onSubmittedWorkDone` over them without `timestamp-query`.
 */
function oursTimer(device: GPUDevice) {
    const timestamps = device.features.has('timestamp-query');
    let capacity = 0;
    let querySet: GPUQuerySet | null = null;
    let resolve: GPUBuffer | null = null;
    let stampRead: GPUBuffer | null = null;
    const grow = (passes: number) => {
        if (passes <= capacity) return;
        querySet?.destroy();
        resolve?.destroy();
        stampRead?.destroy();
        capacity = passes;
        querySet = device.createQuerySet({ type: 'timestamp', count: passes * 2 });
        resolve = device.createBuffer({ size: passes * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
        stampRead = device.createBuffer({ size: passes * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    };
    return {
        timing: (timestamps ? 'timestamps' : 'wall clock') as GpuSdf['timing'],
        async time(encode: (encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites) => void, repeat = 1): Promise<number> {
            const passes = Math.max(1, Math.floor(repeat));
            if (timestamps) grow(passes);
            const encoder = device.createCommandEncoder({ label: 'compare sdf' });
            for (let pass = 0; pass < passes; pass++) encode(encoder, querySet ? { querySet, beginningOfPassWriteIndex: pass * 2, endOfPassWriteIndex: pass * 2 + 1 } : undefined);
            if (querySet) {
                encoder.resolveQuerySet(querySet, 0, passes * 2, resolve!, 0);
                encoder.copyBufferToBuffer(resolve!, 0, stampRead!, 0, passes * 16);
            }
            const start = performance.now();
            device.queue.submit([encoder.finish()]);
            if (!querySet) {
                await device.queue.onSubmittedWorkDone();
                return (performance.now() - start) / passes;
            }
            await stampRead!.mapAsync(GPUMapMode.READ, 0, passes * 16);
            const stamps = new BigInt64Array(stampRead!.getMappedRange(0, passes * 16).slice(0));
            stampRead!.unmap();
            let total = 0;
            for (let pass = 0; pass < passes; pass++) total += Number(stamps[pass * 2 + 1] - stamps[pass * 2]) / 1e6;
            return total / passes;
        },
        destroy() {
            querySet?.destroy();
            resolve?.destroy();
            stampRead?.destroy();
        },
    };
}

export async function createOursSdf(device: GPUDevice, mesh: WeldedMesh, box: Box, sign: SdfSign): Promise<GpuSdf> {
    const scene = await oursScene(device, mesh);
    const generator = await SdfGenerator.create(device, { label: 'compare sdf', rayMasks: false });
    const timer = oursTimer(device);
    let texture: GPUTexture | null = null;
    let grid: SdfGrid | null = null;
    return {
        id: sign === 'unsigned' ? 'ours-unsigned' : sign === 'parity' ? 'ours' : `ours-${sign}`,
        name: `@pmndrs/bvh (SdfGenerator, ${sign === 'unsigned' ? 'unsigned' : sign === 'first-hit' ? 'first hit, 1 ray' : `${sign} vote, 2 or 3 rays`})`,
        timing: timer.timing,
        async generate(dim, repeat) {
            if (!grid || grid.resolution !== dim) {
                texture?.destroy();
                grid = { ...box, resolution: dim };
                texture = createSdfTexture(device, grid, 'compare sdf');
            }
            return timer.time((encoder, timestampWrites) => generator.encode(encoder, scene, { texture: texture! }, grid!, { sign, timestampWrites }), repeat);
        },
        read: () => readSdf(device, { texture: texture! }, grid!),
        dispose() {
            texture?.destroy();
            scene.destroy();
            generator.destroy();
            timer.destroy();
        },
    };
}

/**
 * Kernels for compare/sdf's `?breakdown`: each a part of ours, or another way to
 * sign, on the same trace module (`queries: true`, no ray masks) as `SdfGenerator`'s,
 * writing the same texture. For finding where the time goes; not the library's kernel.
 */
export const OURS_BREAKDOWN = {
    /** The library's kernel before 2026-11-16: closest point, then all three rays, the vote by `sign`. */
    'votes-all': 'closest point + 3 all-hit rays (no early out)',
    /** The third ray only when the first two disagree. */
    'votes-early': 'closest point + 2 or 3 all-hit rays (early out)',
    closest: 'closest point only',
    rays3: '3 all-hit rays only',
    'rays-early': '2 or 3 all-hit rays only (early out)',
    ray1: '1 all-hit ray only',
    'no-winding': 'closest point + 3 all-hit rays, no winding sum',
    'first-hit-only': '1 first-hit ray only',
    'closest-face': 'closest point, signed by the closest face (no ray)',
    'first-hit-bounded': '1 first-hit ray, then the closest point within its hit distance',
    'closest-merged': 'closest point only, merged scene (no per-box transform)',
    'votes-early-merged': 'closest point + 2 or 3 rays, merged scene',
} as const;
export type OursBreakdown = keyof typeof OURS_BREAKDOWN;

const CLOSEST_WGSL = 'let hit = bvhClosestPoint(point, params.maxDistance);\n    let distance = select(params.maxDistance, hit.distance, hit.distance >= 0.0);';
const VOTES_EARLY_WGSL = `let winding = params.sign == 2u;
    let v0 = bvhInsideVote(bvhRayCrossings(point, BVH_INSIDE_DIRECTION_0));
    let v1 = bvhInsideVote(bvhRayCrossings(point, BVH_INSIDE_DIRECTION_1));
    var votes = select(v0.x, v0.y, winding) + select(v1.x, v1.y, winding);
    if (votes == 1u) {
        let v2 = bvhInsideVote(bvhRayCrossings(point, BVH_INSIDE_DIRECTION_2));
        votes += select(v2.x, v2.y, winding);
    }`;
const FIRST_HIT_WGSL = 'let firstHit = bvhTraceClosest(BvhRay(point, 0.0, BVH_INSIDE_DIRECTION_0, 3.0e38));';
const VOTES_ALL_WGSL = `${CLOSEST_WGSL}
    let votes = bvhInsideVotes(point);
    return select(distance, -distance, select(votes.x, votes.y, params.sign == 2u) >= BVH_INSIDE_MAJORITY);`;

const BREAKDOWN_BODIES: Record<OursBreakdown, string> = {
    'votes-all': VOTES_ALL_WGSL,
    'votes-early': `${CLOSEST_WGSL}\n    ${VOTES_EARLY_WGSL}\n    return select(distance, -distance, votes >= BVH_INSIDE_MAJORITY);`,
    closest: `${CLOSEST_WGSL}\n    return distance;`,
    rays3: 'let votes = bvhInsideVotes(point);\n    return f32(votes.x) + f32(votes.y) * 4.0;',
    'rays-early': `${VOTES_EARLY_WGSL}\n    return f32(votes);`,
    ray1: 'let crossings = bvhRayCrossings(point, BVH_INSIDE_DIRECTION_0);\n    return f32(crossings.count) + f32(crossings.winding) * 0.25;',
    'no-winding': VOTES_ALL_WGSL,
    'first-hit-only': `${FIRST_HIT_WGSL}\n    return select(firstHit.t, -firstHit.t, !firstHit.frontFace);`,
    'closest-face': `${CLOSEST_WGSL}\n    return select(distance, -distance, hit.distance > 0.0 && dot(hit.normal, point - hit.point) < 0.0);`,
    // The hit point is on a triangle, so the nearest one is no farther: start the closest-point
    // walk there (a margin for rounding), and walk again unbounded if that found nothing.
    'first-hit-bounded': `${FIRST_HIT_WGSL}
    var limit = params.maxDistance;
    if (firstHit.t >= 0.0) {
        limit = min(limit, firstHit.t * 1.0001 + 1e-30);
    }
    var hit = bvhClosestPoint(point, limit);
    if (hit.distance < 0.0 && limit < params.maxDistance) {
        hit = bvhClosestPoint(point, params.maxDistance);
    }
    let distance = select(params.maxDistance, hit.distance, hit.distance >= 0.0);
    return select(distance, -distance, firstHit.t >= 0.0 && !firstHit.frontFace);`,
    'closest-merged': `${CLOSEST_WGSL}\n    return distance;`,
    'votes-early-merged': `${CLOSEST_WGSL}\n    ${VOTES_EARLY_WGSL}\n    return select(distance, -distance, votes >= BVH_INSIDE_MAJORITY);`,
};

const WINDING_SUM_WGSL = 'bvhCrossingWinding += select(1, -1, hit.frontFacing != triangle.mirrored);';

function breakdownWgsl(variant: OursBreakdown): string {
    let module = traceModuleWgsl({ rayMasks: false, queries: true });
    if (variant === 'no-winding') {
        if (!module.includes(WINDING_SUM_WGSL)) throw new Error('compare sdf: the crossing walk has changed; update no-winding');
        module = module.replace(WINDING_SUM_WGSL, '');
    }
    return `${module}
struct CmpSdfParams {
    boxMin: vec3f,
    maxDistance: f32,
    voxelSize: vec3f,
    sign: u32,
    resolution: vec3u,
    mask: u32,
    zOffset: u32,
}
@group(1) @binding(0) var<uniform> params: CmpSdfParams;
@group(1) @binding(1) var cmpSdfOutput: texture_storage_3d<r32float, write>;

fn cmpSdfValue(point: vec3f) -> f32 {
    ${BREAKDOWN_BODIES[variant]}
}

@compute @workgroup_size(${SDF_WORKGROUP_SIZE}, ${SDF_WORKGROUP_SIZE}, ${SDF_WORKGROUP_SIZE})
fn cmpSdfGenerate(@builtin(global_invocation_id) id: vec3u) {
    if (any(id >= params.resolution)) {
        return;
    }
    let point = params.boxMin + (vec3f(id) + 0.5) * params.voxelSize;
    textureStore(cmpSdfOutput, id, vec4f(cmpSdfValue(point), 0.0, 0.0, 0.0));
}
`;
}

/** One of {@link OURS_BREAKDOWN}'s kernels over the mesh, signed by `sign` where it votes. */
export async function createOursBreakdownSdf(device: GPUDevice, mesh: WeldedMesh, box: Box, variant: OursBreakdown, sign: 'parity' | 'winding' = 'parity'): Promise<GpuSdf> {
    const scene = await oursScene(device, mesh, variant.endsWith('-merged'));
    const code = breakdownWgsl(variant);
    const pipeline = await device.createComputePipelineAsync({
        label: `compare sdf ${variant}`,
        layout: 'auto',
        compute: { module: device.createShaderModule({ label: `compare sdf ${variant}`, code }), entryPoint: 'cmpSdfGenerate' },
    });
    const sceneGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: scene.nodes } },
            { binding: 1, resource: { buffer: scene.triangles } },
            { binding: 2, resource: { buffer: scene.topLevel } },
        ],
    });
    const params = device.createBuffer({ label: 'compare sdf params', size: SDF_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const timer = oursTimer(device);
    let texture: GPUTexture | null = null;
    let fieldGroup: GPUBindGroup | null = null;
    let dim = 0;
    return {
        id: `ours-${variant}${sign === 'winding' ? '-winding' : ''}`,
        name: `@pmndrs/bvh breakdown: ${OURS_BREAKDOWN[variant]}${sign === 'winding' ? ', winding' : ''}`,
        timing: timer.timing,
        async generate(next, repeat) {
            if (next !== dim) {
                dim = next;
                texture?.destroy();
                texture = createSdfTexture(device, { resolution: dim }, 'compare sdf');
                // BvhSdfParams' layout, the voxel size rounded as SdfGenerator rounds it.
                const data = new ArrayBuffer(SDF_PARAMS_BYTES);
                const size = [0, 1, 2].map((axis) => Math.fround((box.max[axis] - box.min[axis]) / dim));
                new Float32Array(data).set([box.min[0], box.min[1], box.min[2], CLOSEST_POINT_MAX_DISTANCE, size[0], size[1], size[2]]);
                new Uint32Array(data).set([sign === 'winding' ? 2 : 1, dim, dim, dim, 0xffffffff, 0], 7);
                device.queue.writeBuffer(params, 0, data);
                fieldGroup = device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(1),
                    entries: [
                        { binding: 0, resource: { buffer: params } },
                        { binding: 1, resource: texture.createView({ dimension: '3d' }) },
                    ],
                });
            }
            const groups = Math.ceil(dim / SDF_WORKGROUP_SIZE);
            return timer.time((encoder, timestampWrites) => {
                const pass = encoder.beginComputePass({ label: `compare sdf ${variant}`, timestampWrites });
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, sceneGroup);
                pass.setBindGroup(1, fieldGroup!);
                pass.dispatchWorkgroups(groups, groups, groups);
                pass.end();
            }, repeat);
        },
        read: () => readSdf(device, { texture: texture! }, { resolution: dim }),
        dispose() {
            texture?.destroy();
            scene.destroy();
            params.destroy();
            timer.destroy();
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

/** Which part of their kernel runs: all of it (as published), the closest point alone, or the +Z first-hit ray alone (compare/sdf's `?breakdown`). */
export type TmbvhPart = 'both' | 'closest' | 'ray';

const CLOSEST_LINES = `			var pointResult: PointQueryResult;
			bvh_ClosestPointToPoint( point, &pointResult );
`;
const RAY_LINES = `			var rayResult: IntersectionResult;
			let ray = Ray( point, vec3f( 0.0, 0.0, 1.0 ), 0.0 );
			bvh_RaycastFirstHit( ray, &rayResult );
`;
const VALUE_LINES = `			let side = select( 1.0, rayResult.side, rayResult.didHit );
			let value = side * sqrt( pointResult.distanceSq );
`;

/** Their kernel, or one part of it (the value then the closest distance, or the ray's signed hit distance). */
function tmbvhKernelWgsl(part: TmbvhPart): string {
    if (part === 'both') return COMPUTE_SDF_WGSL;
    for (const lines of [CLOSEST_LINES, RAY_LINES, VALUE_LINES]) if (!COMPUTE_SDF_WGSL.includes(lines)) throw new Error('compare sdf: their kernel text has changed; update the parts');
    const value = part === 'closest' ? 'let value = sqrt( pointResult.distanceSq );\n' : 'let value = select( 1.0, rayResult.side, rayResult.didHit ) * rayResult.dist;\n';
    return COMPUTE_SDF_WGSL.replace(part === 'closest' ? RAY_LINES : CLOSEST_LINES, '').replace(VALUE_LINES, value);
}

export async function createTmbvhSdf(renderer: WebGPURenderer, mesh: { positions: Float32Array; indices: Uint32Array }, box: { min: [number, number, number]; max: [number, number, number] }, part: TmbvhPart = 'both'): Promise<GpuSdf> {
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
    const computeShader = wgslFn(tmbvhKernelWgsl(part), part === 'both' ? [fns.closestPointToPoint, fns.raycastFirstHit] : [part === 'closest' ? fns.closestPointToPoint : fns.raycastFirstHit]);
    const kernel = computeShader(parameters as never).computeKernel(WORKGROUP_SIZE);
    const node = kernel as unknown as { computeNode: { parameters: { output: { value: unknown }; dim: { value: number } } } };
    const timestamps = renderer.hasFeature('timestamp-query') && (renderer as unknown as { backend: { trackTimestamp: boolean } }).backend.trackTimestamp;
    const backend = (renderer as unknown as { backend: { device: GPUDevice; get(object: object): { texture?: GPUTexture } } }).backend;
    let compiled = false;
    return {
        id: part === 'both' ? 'tmbvh' : `tmbvh-${part}`,
        name: part === 'both' ? 'three-mesh-bvh/webgpu (its webgpu_sdfGeneration kernel, SAH)' : `three-mesh-bvh/webgpu breakdown: ${part === 'closest' ? 'closest point only' : '1 first-hit ray along +Z only'}`,
        timing: timestamps ? 'timestamps' : 'wall clock',
        async generate(next, repeat = 1) {
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
            // 'repeat' dispatches back to back; three sums their timestamps.
            const passes = Math.max(1, Math.floor(repeat));
            if (timestamps) {
                for (let pass = 0; pass < passes; pass++) renderer.compute(kernel, dispatchSize);
                const ms = await renderer.resolveTimestampsAsync('compute');
                if (typeof ms !== 'number') throw new Error('three returned no compute timestamp');
                return ms / passes;
            }
            const start = performance.now();
            for (let pass = 0; pass < passes; pass++) renderer.compute(kernel, dispatchSize);
            await backend.device.queue.onSubmittedWorkDone();
            return (performance.now() - start) / passes;
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

/**
 * Signed distance fields from a trace scene (PARITY 8.10; docs/tracing/sdf.mdx).
 *
 * A grid of voxels over a world-space box; each voxel gets the distance from
 * its centre to the nearest triangle of the scene (`bvhClosestPoint`),
 * negative inside (`bvhIsInsideByRule`: a majority of three rays by parity or
 * by winding; or, opt-in, the side the first hit of one ray is on). The kernel is the trace module with `queries: true` and a small
 * entry point, so it reads every node and triangle format, instanced and
 * merged scenes, `SceneTrees`' buffers and uploads alike, with no copy of the
 * scene.
 *
 * - {@link SdfGenerator}: the GPU kernel. It fills an `r32float` 3D storage
 *   texture ({@link createSdfTexture}) or an `f32` buffer (x fastest, then y,
 *   then z), whole or a slab of z slices at a time.
 * - {@link generateSdfCpu}: the CPU reference over `CpuMeshBvh` trees (or
 *   anything with `closestPointToPoint` and `rayCrossings`, and `intersectRay` for
 *   `'first-hit'`), voxel for voxel
 *   the same points.
 *
 * Voxel `(x, y, z)` of a grid of `n` voxels over `[min, max]` samples the
 * centre `min + (i + 0.5) * (max - min) / n` on each axis, so a texture
 * sampled at `uvw = (p - min) / (max - min)` with linear filtering returns the
 * field at `p` (three-mesh-bvh's sampling, too).
 */

import { DEFAULT_NODE_FORMAT, RAY_MASK_ALL, type NodeFormat, type PositionLayout, type TriangleLeaves } from '../gpu-layout.ts';
import { CLOSEST_POINT_MAX_DISTANCE, INSIDE_TEST_DIRECTIONS, INSIDE_TEST_MAJORITY, type InsideRule } from './query-wgsl.ts';
import { traceSourceFormats, type TraceSceneBuffers } from './trace-gpu.ts';
import { traceModuleWgsl } from './wgsl.ts';

/**
 * How a field is signed: by the inside test's rule (`'parity'`, `'winding'`; see `InsideRule`),
 * by the side of the first triangle one ray hits (`'first-hit'`, three-mesh-bvh's way: cheaper,
 * but it needs one closed, consistently wound surface; docs/tracing/sdf.mdx), or not at all.
 */
export type SdfSign = InsideRule | 'first-hit' | 'unsigned';

/** A grid of voxels over a world-space box. */
export interface SdfGrid {
    /** The box's lower corner, world space. */
    readonly min: readonly [number, number, number];
    /** The box's upper corner, world space. */
    readonly max: readonly [number, number, number];
    /** Voxels per axis: one number for the same count on every axis, or `[x, y, z]`. */
    readonly resolution: number | readonly [number, number, number];
}

/** What a field holds. */
export interface SdfOptions {
    /**
     * Default `'parity'`. `'unsigned'` skips the inside test (an open surface's distance, the
     * closest point alone). `'first-hit'` casts one ray (`INSIDE_TEST_DIRECTIONS[0]`) and signs
     * by the side of the first triangle it hits (the back is inside), as three-mesh-bvh's SDF
     * example does: about a fifth faster than the vote on the RTX 3060, but right only for
     * closed, consistently wound surfaces that don't overlap (a ray through a hole, a flipped
     * triangle or into a second shell decides the voxel alone).
     */
    readonly sign?: SdfSign;
    /**
     * Distances are clamped to this (and to -this inside): the closest-point walk
     * skips boxes farther away, so a narrow band is cheaper. Default: none
     * (`CLOSEST_POINT_MAX_DISTANCE`, 1e18).
     */
    readonly maxDistance?: number;
    /** Only instances whose mask shares a bit with this count. Default all. Ignored by a generator made with `rayMasks: false`. */
    readonly mask?: number;
}

/** Workgroup size of the kernel on each axis (4 x 4 x 4 voxels). */
export const SDF_WORKGROUP_SIZE = 4;

/** Bytes of the kernel's uniform (`BvhSdfParams`). */
export const SDF_PARAMS_BYTES = 64;

/** The kernel's `sign` codes. */
const SIGN_CODE: Record<SdfSign, number> = { unsigned: 0, parity: 1, winding: 2, 'first-hit': 3 };

/** The voxel counts of a grid, checked (positive integers). */
export function sdfResolution(grid: SdfGrid): [number, number, number] {
    const r = grid.resolution;
    const counts: [number, number, number] = typeof r === 'number' ? [r, r, r] : [r[0], r[1], r[2]];
    for (const n of counts) if (!Number.isInteger(n) || n < 1) throw new RangeError(`SDF grid: the resolution must be positive integers, got ${String(r)}`);
    for (let axis = 0; axis < 3; axis++) {
        if (!(grid.max[axis] > grid.min[axis])) throw new RangeError(`SDF grid: max must be above min on every axis, got min ${grid.min.join(', ')} and max ${grid.max.join(', ')}`);
    }
    return counts;
}

/** The f32 voxel size on each axis, as the kernel gets it. */
function voxelSize(grid: SdfGrid, counts: readonly number[]): [number, number, number] {
    return [0, 1, 2].map((axis) => Math.fround((grid.max[axis] - grid.min[axis]) / counts[axis])) as [number, number, number];
}

/**
 * The world-space centre of voxel `(x, y, z)`, rounded as the kernel computes it in
 * f32 (`min + (i + 0.5) * size`; a GPU may fuse the multiply-add, a few ulps apart).
 */
export function sdfVoxelCenter(grid: SdfGrid, x: number, y: number, z: number): [number, number, number] {
    const counts = sdfResolution(grid);
    const size = voxelSize(grid, counts);
    const index = [x, y, z];
    return [0, 1, 2].map((axis) => Math.fround(Math.fround(grid.min[axis]) + Math.fround((index[axis] + 0.5) * size[axis]))) as [number, number, number];
}

function clampedMaxDistance(maxDistance: number | undefined): number {
    const value = maxDistance ?? Infinity;
    if (!(value > 0)) throw new RangeError(`SDF: maxDistance must be positive, got ${String(maxDistance)}`);
    return Math.min(value, CLOSEST_POINT_MAX_DISTANCE);
}

function signOf(options: SdfOptions): SdfSign {
    const sign = options.sign ?? 'parity';
    if (!(sign in SIGN_CODE)) throw new RangeError(`SDF: sign must be 'parity', 'winding', 'first-hit' or 'unsigned', got ${String(sign)}`);
    return sign;
}

// ---- The CPU reference ----------------------------------------------------------------

/** What {@link generateSdfCpu} needs of a tree: `CpuMeshBvh` has all three, in world space (`intersectRay` only for `'first-hit'`). */
export interface SdfCpuSource {
    closestPointToPoint(point: readonly [number, number, number], target?: object, minThreshold?: number, maxThreshold?: number): { readonly distance: number } | null;
    rayCrossings(origin: readonly [number, number, number], direction: readonly [number, number, number]): { readonly count: number; readonly winding: number };
    intersectRay?(origin: readonly [number, number, number], direction: readonly [number, number, number], tMin: number, tMax: number, firstHitOnly: boolean): readonly { readonly t: number; readonly frontFace: boolean }[];
}

/** Whether the first triangle the ray along `INSIDE_TEST_DIRECTIONS[0]` hits, over every source, is hit on its back. */
function firstHitInside(sources: readonly SdfCpuSource[], point: readonly [number, number, number]): boolean {
    let nearest = Infinity;
    let inside = false;
    for (const source of sources) {
        if (!source.intersectRay) throw new TypeError("generateSdfCpu: sign 'first-hit' needs sources with intersectRay (CpuMeshBvh has it)");
        const [hit] = source.intersectRay(point, INSIDE_TEST_DIRECTIONS[0], 0, Infinity, true);
        if (hit && hit.t < nearest) {
            nearest = hit.t;
            inside = !hit.frontFace;
        }
    }
    return inside;
}

/** One voxel's value over several trees: the nearest distance, signed by their crossings added up. */
export function sdfValueCpu(sources: readonly SdfCpuSource[], point: readonly [number, number, number], options: SdfOptions = {}): number {
    const sign = signOf(options);
    const limit = clampedMaxDistance(options.maxDistance);
    let distance = limit;
    for (const source of sources) {
        const hit = source.closestPointToPoint(point, {}, 0, distance);
        if (hit && hit.distance < distance) distance = hit.distance;
    }
    if (sign === 'unsigned') return distance;
    if (sign === 'first-hit') return firstHitInside(sources, point) ? -distance : distance;
    let votes = 0;
    for (let ray = 0; ray < INSIDE_TEST_DIRECTIONS.length; ray++) {
        // As the kernel: the third ray only when the first two split (the same majority).
        if (ray === 2 && votes !== 1) break;
        const direction = INSIDE_TEST_DIRECTIONS[ray];
        let count = 0;
        let winding = 0;
        for (const source of sources) {
            const crossings = source.rayCrossings(point, direction);
            count += crossings.count;
            winding += crossings.winding;
        }
        votes += (sign === 'parity' ? (count & 1) === 1 : winding !== 0) ? 1 : 0;
    }
    return votes >= INSIDE_TEST_MAJORITY ? -distance : distance;
}

/**
 * The field on the CPU, voxel for voxel the kernel's points (x fastest, then
 * y, then z): the reference the GPU is checked against, and a field where
 * there is no GPU. `sources` are world-space trees (a `CpuMeshBvh` per mesh,
 * its instances' positions moved to world space); they count as one scene.
 * Slow for big grids: one closest point and two or three all-hit rays per voxel per tree.
 */
export function generateSdfCpu(sources: SdfCpuSource | readonly SdfCpuSource[], grid: SdfGrid, options: SdfOptions = {}): Float32Array {
    const list = Array.isArray(sources) ? (sources as readonly SdfCpuSource[]) : [sources as SdfCpuSource];
    const [nx, ny, nz] = sdfResolution(grid);
    signOf(options);
    const out = new Float32Array(nx * ny * nz);
    for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) out[(z * ny + y) * nx + x] = sdfValueCpu(list, sdfVoxelCenter(grid, x, y, z), options);
        }
    }
    return out;
}

// ---- The GPU kernel -----------------------------------------------------------------------

/** Where a generator writes: an `r32float` 3D storage texture, or an `f32` buffer. */
export type SdfOutput = 'texture' | 'buffer';

/** Options of {@link sdfKernelWgsl}: the output and the scene's formats, compiled in. */
export interface SdfKernelOptions {
    readonly output?: SdfOutput;
    readonly rayMasks?: boolean;
    readonly triangleLeaves?: TriangleLeaves;
    readonly positionLayout?: PositionLayout;
    readonly nodeFormat?: NodeFormat;
}

/**
 * The SDF kernel: the trace module (group 0, `queries: true`) and its own group 1,
 * `BvhSdfParams` at binding 0 and the output at binding 1. Entry point `bvhSdfGenerate`,
 * workgroups of 4 x 4 x 4 voxels. `bvhSdfValue(point)` is the per-voxel function.
 */
export function sdfKernelWgsl(options: SdfKernelOptions = {}): string {
    const output = options.output ?? 'texture';
    if (output !== 'texture' && output !== 'buffer') throw new RangeError(`sdfKernelWgsl: output must be 'texture' or 'buffer', got ${String(output)}`);
    const rayMasks = options.rayMasks ?? true;
    const masked = (name: string, args: string) => (rayMasks ? `${name}Masked(${args}, params.mask)` : `${name}(${args})`);
    const store =
        output === 'texture'
            ? 'textureStore(bvhSdfOutput, voxel, vec4f(value, 0.0, 0.0, 0.0));'
            : 'bvhSdfOutput[(voxel.z * params.resolution.y + voxel.y) * params.resolution.x + voxel.x] = value;';
    return `${traceModuleWgsl({ rayMasks, triangleLeaves: options.triangleLeaves, positionLayout: options.positionLayout, nodeFormat: options.nodeFormat, queries: true })}
// ---- SDF generation (sdf.ts) ------------------------------------------------------

struct BvhSdfParams {
    boxMin: vec3f,
    maxDistance: f32,     // clamped to BVH_CLOSEST_MAX_DISTANCE on the CPU
    voxelSize: vec3f,
    sign: u32,            // 0 unsigned, 1 parity, 2 winding, 3 first hit
    resolution: vec3u,
    mask: u32,
    zOffset: u32,         // the first z slice of this dispatch
}

const BVH_SDF_UNSIGNED: u32 = 0u;
const BVH_SDF_WINDING: u32 = 2u;
const BVH_SDF_FIRST_HIT: u32 = 3u;
// The first-hit ray's reach: as far as an f32 goes.
const BVH_SDF_RAY_FAR: f32 = 3.0e38;

@group(1) @binding(0) var<uniform> bvhSdfParams: BvhSdfParams;
@group(1) @binding(1) var${output === 'texture' ? ' bvhSdfOutput: texture_storage_3d<r32float, write>' : '<storage, read_write> bvhSdfOutput: array<f32>'};

// The signed distance at 'point': to the nearest triangle (maxDistance when none is nearer),
// negative when the inside test's vote says inside (or, by sign 3, the first hit's back).
fn bvhSdfValue(point: vec3f) -> f32 {
    let params = bvhSdfParams;
    let hit = ${masked('bvhClosestPoint', 'point, params.maxDistance')};
    var distance = select(params.maxDistance, hit.distance, hit.distance >= 0.0);
    if (params.sign == BVH_SDF_FIRST_HIT) {
        let first = ${masked('bvhTraceClosest', 'BvhRay(point, 0.0, BVH_INSIDE_DIRECTION_0, BVH_SDF_RAY_FAR)')};
        distance = select(distance, -distance, first.t >= 0.0 && !first.frontFace);
    } else if (params.sign != BVH_SDF_UNSIGNED) {
        // Two of three rays, the third only when the first two split (bvhIsInsideByRule).
        let inside = ${masked('bvhIsInsideByRule', 'point, params.sign == BVH_SDF_WINDING')};
        distance = select(distance, -distance, inside);
    }
    return distance;
}

@compute @workgroup_size(${SDF_WORKGROUP_SIZE}, ${SDF_WORKGROUP_SIZE}, ${SDF_WORKGROUP_SIZE})
fn bvhSdfGenerate(@builtin(global_invocation_id) id: vec3u) {
    let params = bvhSdfParams;
    let voxel = vec3u(id.x, id.y, id.z + params.zOffset);
    if (any(voxel >= params.resolution)) {
        return;
    }
    let point = params.boxMin + (vec3f(voxel) + 0.5) * params.voxelSize;
    let value = bvhSdfValue(point);
    ${store}
}
`;
}

/** Where {@link SdfGenerator.encode} writes. */
export type SdfTarget =
    | {
          /** `r32float`, dimension `'3d'`, with `STORAGE_BINDING`, at least the grid's size ({@link createSdfTexture}). */
          readonly texture: GPUTexture;
      }
    | {
          /** `STORAGE`, at least `4 * x * y * z` bytes past `offset`: voxel `(x, y, z)` at float `(z * ny + y) * nx + x`. */
          readonly buffer: GPUBuffer;
          /** Bytes; a multiple of the device's `minStorageBufferOffsetAlignment`. Default 0. */
          readonly offset?: number;
      };

/** One dispatch's slices: `count` z slices from `start`. */
export interface SdfSlices {
    readonly start: number;
    readonly count: number;
}

/** Options of {@link SdfGenerator.encode}. */
export interface SdfEncodeOptions extends SdfOptions {
    /** Only these z slices (a slab), for spreading a large grid over several submits. Default all. */
    readonly slices?: SdfSlices;
    /** GPU timing of the dispatch's pass. */
    readonly timestampWrites?: GPUComputePassTimestampWrites;
}

/** Options of {@link SdfGenerator.generate}. */
export interface SdfGenerateOptions extends SdfOptions {
    /**
     * Voxels per submit: the grid is cut into slabs of whole z slices, each its own
     * command buffer, so no single submit runs long enough for the OS to reset the
     * GPU (Windows' TDR is about 2 s). Default 2,097,152 (a 128^3 grid in one).
     */
    readonly voxelsPerSubmit?: number;
}

/** Options of {@link SdfGenerator.create}: the output and the scenes' formats, compiled in, as for `ClosestPointBatchKernel`. */
export interface SdfGeneratorOptions {
    readonly label?: string;
    /** Default `'texture'` (an `r32float` 3D storage texture); `'buffer'` for `f32` in a storage buffer. */
    readonly output?: SdfOutput;
    /** Default true: each call may pass a `mask`. False compiles masks out. */
    readonly rayMasks?: boolean;
    /** The scenes' triangle format. Default `'records'`. */
    readonly triangleLeaves?: TriangleLeaves;
    /** Indexed only: `'vec4'` (the default) or `'packed'` positions. */
    readonly positionLayout?: PositionLayout;
    /** The scenes' node format. Default `'quantized'` (`DEFAULT_NODE_FORMAT`); a generator refuses a scene of the other format. */
    readonly nodeFormat?: NodeFormat;
}

/** A 3D `r32float` texture an {@link SdfGenerator} can fill and a shader can sample. */
export function createSdfTexture(device: GPUDevice, grid: Pick<SdfGrid, 'resolution'>, label = 'sdf'): GPUTexture {
    const r = grid.resolution;
    const [x, y, z] = typeof r === 'number' ? [r, r, r] : r;
    const limit = device.limits.maxTextureDimension3D;
    if (limit !== undefined && Math.max(x, y, z) > limit) throw new RangeError(`createSdfTexture: ${x} x ${y} x ${z} is over the device's maxTextureDimension3D (${limit})`);
    return device.createTexture({
        label,
        size: [x, y, z],
        dimension: '3d',
        format: 'r32float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
}

/**
 * Fills a grid with the signed distance to a scene's triangles on the GPU: one
 * closest-point walk and, for a signed field, two or three all-hit ray walks per
 * voxel (one first-hit ray with `sign: 'first-hit'`; docs/tracing/sdf.mdx). One generator serves any number of scenes of its
 * formats. Group 0 is the scene (as `TraceBatchKernel`'s), group 1 the field's
 * parameters and output.
 */
export class SdfGenerator {
    private readonly sceneGroups = new WeakMap<TraceSceneBuffers, GPUBindGroup>();
    /** One uniform per distinct parameter set (grid, options, slab): dispatches recorded together may differ. */
    private readonly params = new Map<string, GPUBuffer>();

    private constructor(
        private readonly device: GPUDevice,
        private readonly pipeline: GPUComputePipeline,
        readonly output: SdfOutput,
        readonly rayMasks: boolean,
        readonly triangleLeaves: TriangleLeaves,
        readonly positionLayout: PositionLayout,
        readonly nodeFormat: NodeFormat,
    ) {}

    static async create(device: GPUDevice, options: SdfGeneratorOptions = {}): Promise<SdfGenerator> {
        const { label = 'sdf', output = 'texture', rayMasks = true, triangleLeaves = 'records', positionLayout = 'vec4', nodeFormat = DEFAULT_NODE_FORMAT } = options;
        const code = sdfKernelWgsl({ output, rayMasks, triangleLeaves, positionLayout, nodeFormat });
        const pipeline = await device.createComputePipelineAsync({
            label: `${label} generate`,
            layout: 'auto',
            compute: { module: device.createShaderModule({ label: `${label} generate`, code }), entryPoint: 'bvhSdfGenerate' },
        });
        return new SdfGenerator(device, pipeline, output, rayMasks, triangleLeaves, positionLayout, nodeFormat);
    }

    /** Frees the parameter uniforms. The pipeline needs no freeing. */
    destroy(): void {
        for (const buffer of this.params.values()) buffer.destroy();
        this.params.clear();
    }

    private sceneGroup(scene: TraceSceneBuffers): GPUBindGroup {
        let group = this.sceneGroups.get(scene);
        if (!group) {
            const formats = traceSourceFormats(scene, 'SdfGenerator');
            if (formats.triangleLeaves !== this.triangleLeaves) throw new RangeError(`SdfGenerator: the generator reads ${this.triangleLeaves} triangles, the scene holds ${formats.triangleLeaves}`);
            if (formats.triangleLeaves === 'indexed' && formats.positionLayout !== this.positionLayout) {
                throw new RangeError(`SdfGenerator: the generator reads ${this.positionLayout} positions, the scene holds ${formats.positionLayout}`);
            }
            if (formats.nodeFormat !== this.nodeFormat) throw new RangeError(`SdfGenerator: the generator reads ${this.nodeFormat} nodes, the scene holds ${formats.nodeFormat}`);
            const indexed = formats.triangleLeaves === 'indexed';
            if (indexed && !scene.positions) throw new RangeError('SdfGenerator: an indexed scene needs its positions buffer');
            group = this.device.createBindGroup({
                label: 'sdf scene',
                layout: this.pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: scene.nodes } },
                    { binding: 1, resource: { buffer: scene.triangles } },
                    { binding: 2, resource: { buffer: scene.topLevel } },
                    ...(indexed ? [{ binding: 3, resource: { buffer: scene.positions! } }] : []),
                ],
            });
            this.sceneGroups.set(scene, group);
        }
        return group;
    }

    private paramsBuffer(grid: SdfGrid, counts: readonly number[], options: SdfOptions, zOffset: number): GPUBuffer {
        const size = voxelSize(grid, counts);
        const maxDistance = clampedMaxDistance(options.maxDistance);
        const sign = SIGN_CODE[signOf(options)];
        const mask = (options.mask ?? RAY_MASK_ALL) >>> 0;
        const data = new ArrayBuffer(SDF_PARAMS_BYTES);
        const floats = new Float32Array(data);
        const words = new Uint32Array(data);
        floats.set([grid.min[0], grid.min[1], grid.min[2], maxDistance, size[0], size[1], size[2]]);
        words[7] = sign;
        words.set([counts[0], counts[1], counts[2], mask, zOffset], 8);
        const key = Array.from(words).join(',');
        let buffer = this.params.get(key);
        if (!buffer) {
            buffer = this.device.createBuffer({ label: 'sdf params', size: SDF_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            this.device.queue.writeBuffer(buffer, 0, data);
            this.params.set(key, buffer);
        }
        return buffer;
    }

    private outputEntry(target: SdfTarget, counts: readonly number[]): GPUBindGroupEntry {
        const [nx, ny, nz] = counts;
        if ('texture' in target) {
            if (this.output !== 'texture') throw new TypeError("SdfGenerator: this generator writes a buffer (output: 'buffer'); pass { buffer }");
            const texture = target.texture;
            if (texture.format !== undefined && texture.format !== 'r32float') throw new RangeError(`SdfGenerator: the texture must be r32float, got ${texture.format}`);
            if (texture.dimension !== undefined && texture.dimension !== '3d') throw new RangeError(`SdfGenerator: the texture must be 3d, got ${texture.dimension}`);
            if (texture.width < nx || texture.height < ny || texture.depthOrArrayLayers < nz) {
                throw new RangeError(`SdfGenerator: the texture is ${texture.width} x ${texture.height} x ${texture.depthOrArrayLayers}, the grid needs ${nx} x ${ny} x ${nz}`);
            }
            return { binding: 1, resource: texture.createView({ dimension: '3d' }) };
        }
        if (this.output !== 'buffer') throw new TypeError("SdfGenerator: this generator writes a texture (output: 'texture'); pass { texture }");
        const offset = target.offset ?? 0;
        const bytes = nx * ny * nz * 4;
        if (target.buffer.size < offset + bytes) throw new RangeError(`SdfGenerator: the buffer holds ${target.buffer.size - offset} bytes past offset ${offset}, the grid needs ${bytes}`);
        return { binding: 1, resource: { buffer: target.buffer, offset, size: bytes } };
    }

    /** Records one dispatch into `encoder`: the whole grid, or `options.slices`. */
    encode(encoder: GPUCommandEncoder, scene: TraceSceneBuffers, target: SdfTarget, grid: SdfGrid, options: SdfEncodeOptions = {}): void {
        const counts = sdfResolution(grid);
        const start = options.slices?.start ?? 0;
        const count = options.slices?.count ?? counts[2] - start;
        if (!Number.isInteger(start) || !Number.isInteger(count) || start < 0 || count < 0 || start + count > counts[2]) {
            throw new RangeError(`SdfGenerator: slices ${start} to ${start + count} are outside the grid's ${counts[2]}`);
        }
        if (count === 0) return;
        const sceneGroup = this.sceneGroup(scene);
        const fieldGroup = this.device.createBindGroup({
            label: 'sdf field',
            layout: this.pipeline.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this.paramsBuffer(grid, counts, options, start) } }, this.outputEntry(target, counts)],
        });
        const workgroups = (n: number) => Math.ceil(n / SDF_WORKGROUP_SIZE);
        const pass = encoder.beginComputePass({ label: 'sdf generate', timestampWrites: options.timestampWrites });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, sceneGroup);
        pass.setBindGroup(1, fieldGroup);
        pass.dispatchWorkgroups(workgroups(counts[0]), workgroups(counts[1]), workgroups(count));
        pass.end();
    }

    /**
     * Fills the whole grid, a slab of z slices per submit (`voxelsPerSubmit`),
     * and resolves when the GPU is done, with the wall time from the first
     * submit to then (CPU clock: the submits and the GPU's work).
     */
    async generate(scene: TraceSceneBuffers, target: SdfTarget, grid: SdfGrid, options: SdfGenerateOptions = {}): Promise<{ readonly ms: number; readonly submits: number }> {
        const counts = sdfResolution(grid);
        const perSubmit = options.voxelsPerSubmit ?? 1 << 21;
        if (!(perSubmit >= 1)) throw new RangeError(`SdfGenerator: voxelsPerSubmit must be at least 1, got ${perSubmit}`);
        const slabDepth = Math.max(1, Math.floor(perSubmit / (counts[0] * counts[1])));
        const start = performance.now();
        let submits = 0;
        for (let z = 0; z < counts[2]; z += slabDepth) {
            const encoder = this.device.createCommandEncoder({ label: 'sdf generate' });
            this.encode(encoder, scene, target, grid, { ...options, slices: { start: z, count: Math.min(slabDepth, counts[2] - z) } });
            this.device.queue.submit([encoder.finish()]);
            submits++;
        }
        await this.device.queue.onSubmittedWorkDone();
        return { ms: performance.now() - start, submits };
    }
}

/**
 * Reads a field back to the CPU (x fastest, then y, then z), from a texture
 * {@link SdfGenerator} filled or from a buffer. For checks and exports; it
 * waits for the GPU.
 */
export async function readSdf(device: GPUDevice, target: SdfTarget, grid: Pick<SdfGrid, 'resolution'>): Promise<Float32Array> {
    const r = grid.resolution;
    const [nx, ny, nz] = typeof r === 'number' ? [r, r, r] : r;
    const rowBytes = nx * 4;
    const pitch = 'texture' in target ? Math.ceil(rowBytes / 256) * 256 : rowBytes;
    const size = pitch * ny * nz;
    const staging = device.createBuffer({ label: 'sdf read', size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'sdf read' });
    if ('texture' in target) {
        encoder.copyTextureToBuffer({ texture: target.texture }, { buffer: staging, bytesPerRow: pitch, rowsPerImage: ny }, [nx, ny, nz]);
    } else {
        encoder.copyBufferToBuffer(target.buffer, target.offset ?? 0, staging, 0, size);
    }
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    if (pitch === rowBytes) return new Float32Array(bytes.buffer);
    const out = new Float32Array(nx * ny * nz);
    for (let row = 0; row < ny * nz; row++) out.set(new Float32Array(bytes.buffer, row * pitch, nx), row * nx);
    return out;
}

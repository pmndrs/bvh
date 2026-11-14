/**
 * Signed distance fields as three textures: the core's `SdfGenerator` writes
 * straight into a three `Storage3DTexture` (`r32float`) on the renderer's
 * device, from the same trace buffers `BvhTraceNodes` reads (a `SceneTrees`,
 * or an upload), and the result is sampled in TSL with `texture3D` (or
 * {@link BvhSdfTexture.sample}). three-mesh-bvh's `webgpu_sdfGeneration`
 * pattern, without a TSL kernel: the field is filled by plain WebGPU.
 *
 * three creates the texture (`renderer.initTexture`), so it owns and disposes
 * it; the generator only writes into the `GPUTexture` three made, which it
 * reads from the WebGPU backend (`backend.get(texture).texture`, not public
 * API, checked at run time; written against three r187). Don't set the
 * texture's `needsUpdate` afterwards: three would make a new, empty one.
 */

import { Box3, ClampToEdgeWrapping, FloatType, LinearFilter, NearestFilter, RedFormat, Vector3 } from 'three';
import { Storage3DTexture } from 'three/webgpu';
import { texture3D, uniform } from 'three/tsl';
import { SdfGenerator, type SdfOptions } from '../trace/sdf.ts';
import { traceSourceFormats, type TraceSceneBuffers } from '../trace/trace-gpu.ts';
import { deviceFromRenderer } from './scene.ts';
import type { BvhTraceBufferSource, BvhTslNode } from './trace-nodes.ts';

/** Options of {@link generateSdfTexture}. */
export interface BvhSdfTextureOptions extends SdfOptions {
    /** The world-space box the field covers (the texture's extent). */
    readonly box: Box3 | { readonly min: { x: number; y: number; z: number }; readonly max: { x: number; y: number; z: number } };
    /** Voxels per axis. Default 64. */
    readonly resolution?: number | readonly [number, number, number];
    /** Instance masks compiled into the kernel. Default true (`mask` works). */
    readonly rayMasks?: boolean;
    /** Voxels per submit (`SdfGenerator.generate`). */
    readonly voxelsPerSubmit?: number;
    /** A texture to fill again (from an earlier call, same resolution); default a new one. */
    readonly texture?: Storage3DTexture;
}

/** A field on the GPU: the texture, the box it covers, and how to sample it in TSL. */
export interface BvhSdfTexture {
    /** `r32float`, the box's extent; linear filtering where the device has `float32-filterable`, else nearest. */
    readonly texture: Storage3DTexture;
    /** The world-space box the texture covers (a copy). */
    readonly box: Box3;
    readonly resolution: readonly [number, number, number];
    /** Wall time of the generation, submit to done. */
    readonly ms: number;
    /** TSL: the texture coordinate of a world-space position node, `(p - min) / (max - min)`. */
    uvw(position: BvhTslNode): BvhTslNode;
    /** TSL: the signed distance at a world-space position node (a `float`; trilinear inside the box, clamped to its edge outside). */
    sample(position: BvhTslNode): BvhTslNode;
    /** Disposes the texture. */
    dispose(): void;
}

interface RendererLike {
    readonly backend: { readonly isWebGPUBackend?: boolean; readonly device: GPUDevice | null; get(object: object): Record<string, unknown> };
    initTexture(texture: object): void;
    hasFeature(name: string): boolean;
}

/** The GPUTexture three made for `texture`, after `initTexture` (three r187: `backend.get(texture).texture`). */
function gpuTextureOf(renderer: RendererLike, texture: Storage3DTexture): GPUTexture {
    renderer.initTexture(texture);
    const made = renderer.backend.get(texture).texture as GPUTexture | undefined;
    if (!made || typeof made.createView !== 'function') {
        throw new Error('@pmndrs/bvh/three: generateSdfTexture reads the GPUTexture three made from backend.get(texture).texture, written against three r187; this three version keeps it elsewhere');
    }
    if (made.format !== 'r32float' || made.dimension !== '3d') throw new Error(`@pmndrs/bvh/three: three made a ${made.dimension} ${made.format} texture, the SDF needs 3d r32float`);
    return made;
}

/** Generators per device and format (compiling one is the slow part). */
const generators = new WeakMap<GPUDevice, Map<string, Promise<SdfGenerator>>>();

function generatorFor(device: GPUDevice, formats: ReturnType<typeof traceSourceFormats>, rayMasks: boolean): Promise<SdfGenerator> {
    let byFormat = generators.get(device);
    if (!byFormat) generators.set(device, (byFormat = new Map()));
    const key = `${formats.nodeFormat} ${formats.triangleLeaves} ${formats.positionLayout} ${rayMasks}`;
    let generator = byFormat.get(key);
    if (!generator) {
        generator = SdfGenerator.create(device, { label: '@pmndrs/bvh/three sdf', ...formats, rayMasks });
        byFormat.set(key, generator);
    }
    return generator;
}

/** The source's buffers and formats, as the generator takes them. */
function sceneOf(source: BvhTraceBufferSource): TraceSceneBuffers {
    const own = source as { nodeFormat?: unknown; triangleLeaves?: unknown; positionLayout?: unknown };
    const named = own.nodeFormat !== undefined || own.triangleLeaves !== undefined ? own : (source.buffers as Partial<TraceSceneBuffers>);
    const formats = traceSourceFormats(named, '@pmndrs/bvh/three: generateSdfTexture');
    const { nodes, triangles, topLevel, positions } = source.buffers;
    return { nodes, triangles, topLevel, positions: positions ?? null, ...formats, byteLength: 0, destroy() {} };
}

/**
 * Fills a three 3D texture with the signed distance to the scene's triangles
 * (docs/tracing/sdf.mdx): `SdfGenerator` on the renderer's device, over the
 * buffers of `source` (a `SceneTrees` after its `update`, or
 * `{ buffers: uploadTraceScene(...) }`). Resolves when the GPU is done.
 *
 *     const sdf = await generateSdfTexture(renderer, trees, { box, resolution: 96 });
 *     material.colorNode = sdf.sample(positionWorld).lessThan(0).select(inside, outside);
 */
export async function generateSdfTexture(renderer: unknown, source: BvhTraceBufferSource, options: BvhSdfTextureOptions): Promise<BvhSdfTexture> {
    const target = renderer as RendererLike;
    const device = deviceFromRenderer(renderer);
    const r = options.resolution ?? 64;
    const resolution: [number, number, number] = typeof r === 'number' ? [r, r, r] : [r[0], r[1], r[2]];
    const box = new Box3(new Vector3(options.box.min.x, options.box.min.y, options.box.min.z), new Vector3(options.box.max.x, options.box.max.y, options.box.max.z));
    const scene = sceneOf(source);
    const generator = await generatorFor(device, traceSourceFormats(scene, '@pmndrs/bvh/three: generateSdfTexture'), options.rayMasks ?? true);

    let texture = options.texture;
    if (texture) {
        const image = texture.image as { width: number; height: number; depth: number };
        if (image.width !== resolution[0] || image.height !== resolution[1] || image.depth !== resolution[2]) {
            throw new RangeError(`@pmndrs/bvh/three: the texture is ${image.width} x ${image.height} x ${image.depth}, the field ${resolution.join(' x ')}; make a new one`);
        }
    } else {
        texture = new Storage3DTexture(resolution[0], resolution[1], resolution[2]);
        texture.name = '@pmndrs/bvh sdf';
        texture.format = RedFormat;
        texture.type = FloatType;
        texture.generateMipmaps = false;
        texture.wrapS = texture.wrapT = texture.wrapR = ClampToEdgeWrapping;
        const filter = target.hasFeature('float32-filterable') ? LinearFilter : NearestFilter;
        texture.minFilter = texture.magFilter = filter;
    }
    const gpuTexture = gpuTextureOf(target, texture);
    const grid = { min: [box.min.x, box.min.y, box.min.z] as const, max: [box.max.x, box.max.y, box.max.z] as const, resolution };
    const { ms } = await generator.generate(scene, { texture: gpuTexture }, grid, options);

    const boxMin = uniform(box.min.clone());
    const boxSize = uniform(box.getSize(new Vector3()));
    const field = texture;
    const uvw = (position: BvhTslNode) => position.sub(boxMin).div(boxSize);
    return {
        texture: field,
        box,
        resolution,
        ms,
        uvw,
        sample: (position: BvhTslNode) => texture3D(field, uvw(position), 0).r,
        dispose: () => field.dispose(),
    };
}

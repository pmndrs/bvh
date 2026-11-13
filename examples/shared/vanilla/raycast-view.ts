/**
 * A ray-cast view in plain WebGPU: one compute thread per pixel calls the
 * package's WGSL walk (`traceModuleWgsl`, included in the kernel text) for the
 * closest hit and writes a colour from the hit alone (its normal, material,
 * instance, triangle or distance), plus, optionally, one any-hit ray towards
 * a fixed direction for a hard shadow. A tiny render pass copies the image to
 * the canvas.
 *
 * This is a demo of the walk, not a renderer: no sampling, no BSDFs, no
 * accumulation (the package ends at the hit, HANDOFF.md). Each pixel's hit
 * record (t, instance, triangle, material) is kept in a storage buffer, so a
 * page picks by reading back the pixel under the pointer, with no extra ray.
 *
 * The kernel is compiled for the scene's formats (`triangleLeaves`,
 * `positionLayout`, `nodeFormat`, read from the scene, never assumed), and
 * bind groups are made again when the scene's buffers change.
 */

import { traceModuleEntries, traceModuleWgsl, type TraceSceneBuffers } from '@pmndrs/bvh';
import { GpuTimer } from '../gpu-timer.ts';
import type { TraceSource } from '../gpu-rays.ts';
import type { CameraBasis } from './camera.ts';

export const VIEW_MODES = ['normal', 'material', 'instance', 'triangle', 'depth'] as const;
export type ViewMode = (typeof VIEW_MODES)[number];

export interface ViewSettings {
    readonly mode: ViewMode;
    /** One any-hit ray per hit towards `sun`: a hard shadow (the walk's any entry point). */
    readonly shadows: boolean;
    /** Unit direction towards the light. */
    readonly sun: readonly [number, number, number];
    /** The instance to tint, or null. */
    readonly highlight: number | null;
    /** Distance mapped to white in `'depth'` mode, and the shadow ray's offset scale. */
    readonly sceneRadius: number;
}

/** One pixel's hit, from the last traced frame. */
export interface PixelHit {
    readonly t: number;
    readonly instance: number;
    readonly triangle: number;
    readonly material: number;
}

export interface ViewMeasurement {
    /** GPU milliseconds of the trace pass (timestamps), or null without `timestamp-query`. */
    readonly gpuMs: number | null;
    readonly primaryRays: number;
    readonly shadowRays: number;
    readonly hits: number;
}

const VIEW_BYTES = 96;
const NO_HIGHLIGHT = 0xffffffff;
const MODE_INDEX: Readonly<Record<ViewMode, number>> = { normal: 0, material: 1, instance: 2, triangle: 3, depth: 4 };

function kernelWgsl(source: TraceSource): string {
    const walk = traceModuleWgsl({ traceSlot: { group: 0, firstBinding: 0 }, triangleLeaves: source.triangleLeaves, positionLayout: source.positionLayout, nodeFormat: source.nodeFormat });
    return /* wgsl */ `
${walk}

struct View {
    origin: vec3f, mode: u32,
    forward: vec3f, width: u32,
    right: vec3f, height: u32,
    up: vec3f, highlight: u32,
    sun: vec3f, shadows: u32,
    radius: f32, pad0: u32, pad1: u32, pad2: u32,
}

@group(1) @binding(0) var<uniform> view: View;
@group(1) @binding(1) var image: texture_storage_2d<rgba8unorm, write>;
// Per pixel: t (bits; negative on a miss), instance, triangle, material.
@group(1) @binding(2) var<storage, read_write> pixels: array<vec4u>;
// Hits, shadow rays.
@group(1) @binding(3) var<storage, read_write> counters: array<atomic<u32>, 2>;

var<workgroup> localHits: atomic<u32>;
var<workgroup> localShadows: atomic<u32>;

fn hashColor(value: u32) -> vec3f {
    var h = value * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    let c = vec3f(f32(h & 0xffu), f32((h >> 8u) & 0xffu), f32((h >> 16u) & 0xffu)) / 255.0;
    return 0.25 + 0.75 * c;
}

fn background(direction: vec3f) -> vec3f {
    return mix(vec3f(0.07, 0.08, 0.10), vec3f(0.16, 0.19, 0.24), clamp(direction.y * 0.5 + 0.5, 0.0, 1.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) local: u32) {
    if (local == 0u) {
        atomicStore(&localHits, 0u);
        atomicStore(&localShadows, 0u);
    }
    workgroupBarrier();
    let inside = id.x < view.width && id.y < view.height;
    if (inside) {
        let ndc = vec2f((f32(id.x) + 0.5) / f32(view.width) * 2.0 - 1.0, 1.0 - (f32(id.y) + 0.5) / f32(view.height) * 2.0);
        let direction = normalize(view.forward + ndc.x * view.right + ndc.y * view.up);
        let hit = bvhTraceClosest(BvhRay(view.origin, 0.0, direction, 1e30));
        let index = id.y * view.width + id.x;
        var color = background(direction);
        if (hit.t >= 0.0) {
            atomicAdd(&localHits, 1u);
            // The geometric normal of the authored front; turned towards the camera for display.
            var n = hit.normal;
            if (dot(n, direction) > 0.0) { n = -n; }
            var base = n * 0.5 + 0.5;
            switch view.mode {
                case 1u: { base = select(hashColor(hit.material), vec3f(0.6), hit.material == BVH_NO_MATERIAL); }
                case 2u: { base = hashColor(hit.instance); }
                case 3u: { base = hashColor(hit.triangle * 2654435761u + hit.instance); }
                case 4u: { base = vec3f(1.0 - clamp(hit.t / (view.radius * 4.0), 0.0, 1.0)); }
                default: {}
            }
            // A headlight term so shapes read; not a lighting model.
            var light = 0.4 + 0.6 * abs(dot(n, direction));
            if (view.shadows != 0u) {
                atomicAdd(&localShadows, 1u);
                let p = view.origin + direction * hit.t + n * (view.radius * 1e-4);
                if (bvhTraceAny(BvhRay(p, 0.0, view.sun, 1e30))) { light = light * 0.45; }
            }
            color = base * light;
            if (hit.instance == view.highlight) { color = mix(color, vec3f(1.0, 0.78, 0.47), 0.6); }
            pixels[index] = vec4u(bitcast<u32>(hit.t), hit.instance, hit.triangle, hit.material);
        } else {
            pixels[index] = vec4u(bitcast<u32>(-1.0), 0xffffffffu, 0xffffffffu, 0xffffffffu);
        }
        textureStore(image, vec2i(id.xy), vec4f(color, 1.0));
    }
    workgroupBarrier();
    if (local == 0u) {
        atomicAdd(&counters[0], atomicLoad(&localHits));
        atomicAdd(&counters[1], atomicLoad(&localShadows));
    }
}
`;
}

const BLIT_WGSL = /* wgsl */ `
@group(0) @binding(0) var image: texture_2d<f32>;
@vertex fn vertexMain(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fragmentMain(@builtin(position) position: vec4f) -> @location(0) vec4f {
    let size = vec2f(textureDimensions(image));
    return textureLoad(image, vec2i(min(position.xy, size - 1.0)), 0);
}`;

export class RayCastView {
    private pipeline: GPUComputePipeline | null = null;
    private pipelineKey = '';
    private sceneLayout: GPUBindGroupLayout | null = null;
    private ioLayout: GPUBindGroupLayout | null = null;
    private sceneGroup: GPUBindGroup | null = null;
    private sceneKey: readonly GPUBuffer[] = [];
    private ioGroup: GPUBindGroup | null = null;
    private blitGroup: GPUBindGroup | null = null;
    private image: GPUTexture | null = null;
    private pixels: GPUBuffer | null = null;
    private readonly uniform: GPUBuffer;
    private readonly counters: GPUBuffer;
    private readonly countersRead: GPUBuffer;
    private readonly pickRead: GPUBuffer;
    private readonly blit: GPURenderPipeline;
    private readonly timer: GpuTimer | null;
    private width = 0;
    private height = 0;
    // What the last trace recorded, collected by the next `collect()`.
    private measuring = false;
    private picking = false;
    private lastRays = 0;

    private constructor(
        private readonly device: GPUDevice,
        format: GPUTextureFormat,
    ) {
        this.uniform = device.createBuffer({ label: 'raycast view', size: VIEW_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.counters = device.createBuffer({ label: 'raycast counters', size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.countersRead = device.createBuffer({ label: 'raycast counters read', size: 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        this.pickRead = device.createBuffer({ label: 'raycast pick read', size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const module = device.createShaderModule({ label: 'raycast blit', code: BLIT_WGSL });
        this.blit = device.createRenderPipeline({
            label: 'raycast blit',
            layout: 'auto',
            vertex: { module, entryPoint: 'vertexMain' },
            fragment: { module, entryPoint: 'fragmentMain', targets: [{ format }] },
        });
        this.timer = GpuTimer.create(device, 1);
    }

    static create(device: GPUDevice, format: GPUTextureFormat): RayCastView {
        return new RayCastView(device, format);
    }

    /** Compiles the kernel for `source`'s formats (again only when they change). Call before {@link trace}. */
    async prepare(source: TraceSource): Promise<void> {
        const key = `${source.triangleLeaves}/${source.positionLayout}/${source.nodeFormat}`;
        if (this.pipeline && key === this.pipelineKey) return;
        const entries = traceModuleEntries({ triangleLeaves: source.triangleLeaves });
        // Made in locals and swapped in together after the await: frames meanwhile trace with the old kernel and its layouts.
        const sceneLayout = this.device.createBindGroupLayout({ label: 'raycast scene', entries: entries.map((entry, binding) => ({ ...entry, binding })) });
        const ioLayout = this.device.createBindGroupLayout({
            label: 'raycast io',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ],
        });
        const pipeline = await this.device.createComputePipelineAsync({
            label: 'raycast view',
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [sceneLayout, ioLayout] }),
            compute: { module: this.device.createShaderModule({ label: 'raycast view', code: kernelWgsl(source) }), entryPoint: 'main' },
        });
        this.sceneLayout = sceneLayout;
        this.ioLayout = ioLayout;
        this.pipeline = pipeline;
        this.pipelineKey = key;
        this.sceneGroup = null;
        this.ioGroup = null;
    }

    private bindScene(buffers: TraceSceneBuffers): GPUBindGroup {
        const list = [buffers.nodes, buffers.triangles, buffers.topLevel, ...(buffers.positions ? [buffers.positions] : [])];
        // Made again when a buffer was replaced (a SceneTrees' storage grew: its `version` moved).
        if (!this.sceneGroup || list.length !== this.sceneKey.length || list.some((buffer, index) => buffer !== this.sceneKey[index])) {
            this.sceneGroup = this.device.createBindGroup({ label: 'raycast scene', layout: this.sceneLayout!, entries: list.map((buffer, binding) => ({ binding, resource: { buffer } })) });
            this.sceneKey = list;
        }
        return this.sceneGroup;
    }

    private resize(width: number, height: number): void {
        if (this.image && width === this.width && height === this.height && this.ioGroup) return;
        if (width !== this.width || height !== this.height || !this.image) {
            this.image?.destroy();
            this.pixels?.destroy();
            this.image = this.device.createTexture({ label: 'raycast image', size: [width, height], format: 'rgba8unorm', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
            this.pixels = this.device.createBuffer({ label: 'raycast pixels', size: width * height * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
            this.blitGroup = this.device.createBindGroup({ layout: this.blit.getBindGroupLayout(0), entries: [{ binding: 0, resource: this.image.createView() }] });
            this.width = width;
            this.height = height;
        }
        this.ioGroup = this.device.createBindGroup({
            label: 'raycast io',
            layout: this.ioLayout!,
            entries: [
                { binding: 0, resource: { buffer: this.uniform } },
                { binding: 1, resource: this.image!.createView() },
                { binding: 2, resource: { buffer: this.pixels! } },
                { binding: 3, resource: { buffer: this.counters } },
            ],
        });
    }

    get size(): { readonly width: number; readonly height: number } {
        return { width: this.width, height: this.height };
    }

    /**
     * Records the trace pass for a `width` x `height` image. `measure` adds
     * timestamps and copies the counters out; `pick` copies one pixel's hit
     * record out. Both are read by the next {@link collect} (after the submit).
     */
    trace(encoder: GPUCommandEncoder, source: TraceSource, camera: CameraBasis, settings: ViewSettings, width: number, height: number, options: { measure?: boolean; pick?: readonly [number, number] | null } = {}): void {
        if (!this.pipeline) throw new Error('RayCastView: prepare() the kernel first');
        this.resize(width, height);
        const floats = new Float32Array(VIEW_BYTES / 4);
        const words = new Uint32Array(floats.buffer);
        floats.set(camera.origin, 0);
        words[3] = MODE_INDEX[settings.mode];
        floats.set(camera.forward, 4);
        words[7] = width;
        floats.set(camera.right, 8);
        words[11] = height;
        floats.set(camera.up, 12);
        words[15] = settings.highlight ?? NO_HIGHLIGHT;
        floats.set(settings.sun, 16);
        words[19] = settings.shadows ? 1 : 0;
        floats[20] = settings.sceneRadius;
        this.device.queue.writeBuffer(this.uniform, 0, floats);
        encoder.clearBuffer(this.counters);
        const measure = !!options.measure && !this.measuring && !(this.timer?.pending ?? false);
        const timestampWrites = measure ? this.timer?.pass() : undefined;
        const pass = encoder.beginComputePass({ label: 'raycast view', timestampWrites });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindScene(source.traceBuffers));
        pass.setBindGroup(1, this.ioGroup!);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
        if (measure) {
            this.timer?.resolveInto(encoder);
            encoder.copyBufferToBuffer(this.counters, 0, this.countersRead, 0, 8);
            this.measuring = true;
            this.lastRays = width * height;
        }
        const pick = options.pick;
        if (pick && !this.picking && pick[0] >= 0 && pick[1] >= 0 && pick[0] < width && pick[1] < height) {
            encoder.copyBufferToBuffer(this.pixels!, (pick[1] * width + pick[0]) * 16, this.pickRead, 0, 16);
            this.picking = true;
        }
    }

    /** Copies the image to `target` (the canvas). */
    draw(encoder: GPUCommandEncoder, target: GPUTextureView): void {
        if (!this.blitGroup) return;
        const pass = encoder.beginRenderPass({ label: 'raycast blit', colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
        pass.setPipeline(this.blit);
        pass.setBindGroup(0, this.blitGroup);
        pass.draw(3);
        pass.end();
    }

    /**
     * What the last submitted {@link trace} was asked for: its measurement
     * (undefined if none was asked) and the picked pixel (undefined if none
     * was asked, null on a miss).
     */
    async collect(): Promise<{ measurement?: ViewMeasurement; pick?: PixelHit | null }> {
        const out: { measurement?: ViewMeasurement; pick?: PixelHit | null } = {};
        if (this.measuring) {
            const [ms] = await Promise.all([this.timer ? this.timer.result() : Promise.resolve([]), this.countersRead.mapAsync(GPUMapMode.READ)]);
            const counts = new Uint32Array(this.countersRead.getMappedRange().slice(0));
            this.countersRead.unmap();
            this.measuring = false;
            out.measurement = { gpuMs: ms.length ? ms[0] : null, primaryRays: this.lastRays, shadowRays: counts[1], hits: counts[0] };
        }
        if (this.picking) {
            await this.pickRead.mapAsync(GPUMapMode.READ);
            out.pick = decodePixel(new Uint32Array(this.pickRead.getMappedRange().slice(0)), 0);
            this.pickRead.unmap();
            this.picking = false;
        }
        return out;
    }

    /** Every pixel's hit record of the last trace (for checks; one big readback). */
    async readPixels(): Promise<{ readonly width: number; readonly height: number; hit(x: number, y: number): PixelHit | null }> {
        const size = this.width * this.height * 16;
        const read = this.device.createBuffer({ label: 'raycast pixels read', size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const encoder = this.device.createCommandEncoder({ label: 'raycast pixels read' });
        encoder.copyBufferToBuffer(this.pixels!, 0, read, 0, size);
        this.device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const words = new Uint32Array(read.getMappedRange().slice(0));
        read.destroy();
        const width = this.width;
        return { width, height: this.height, hit: (x, y) => decodePixel(words, y * width + x) };
    }

    destroy(): void {
        this.image?.destroy();
        this.pixels?.destroy();
        this.uniform.destroy();
        this.counters.destroy();
        this.countersRead.destroy();
        this.pickRead.destroy();
        this.timer?.destroy();
    }
}

function decodePixel(words: Uint32Array, index: number): PixelHit | null {
    const t = new Float32Array(words.buffer, words.byteOffset + index * 16, 1)[0];
    if (!(t >= 0)) return null;
    return { t, instance: words[index * 4 + 1], triangle: words[index * 4 + 2], material: words[index * 4 + 3] };
}

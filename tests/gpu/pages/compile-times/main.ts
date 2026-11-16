/**
 * compile-times: how long the package's pipelines take to compile on this GPU
 * (a user-facing startup cost: on Windows, Chrome compiles WGSL through Tint
 * to HLSL, then DXC or FXC, then the driver; on macOS through Tint to MSL).
 * Manual; reports only.
 *
 * Each item compiles on a fresh device (`requestAdapter`, `requestDevice`),
 * one item at a time, so nothing else competes for the compiler and a
 * device's own pipeline cache never answers for another item. The time is the
 * wall clock from the first `createShaderModule` to the last pipeline promise
 * resolved: the library's own `prepare()` / `create()` where it has one (its
 * pipelines asked for together, as it does), else one
 * `createComputePipelineAsync`. `trace-closest-sync` is the same kernel made
 * with `createComputePipeline` and waited for with `onSubmittedWorkDone`: the
 * stall a synchronous pipeline puts in front of the queue.
 *
 * **Passes.** The runner starts Chrome on a fresh profile, so pass 1 is cold
 * for the browser (no shader cache); pass 2 compiles the same items again on
 * new devices, and Chrome's shader cache (Dawn's blob cache, kept by the GPU
 * process) can answer: warm, as on a second visit. Run with
 * `BVH_CHROME_FLAGS=--disable-gpu-shader-disk-cache` to see pass 2 without
 * that cache. Items share code (H-PLOC and LBVH share the radix sort and the
 * scan; the mesh refit contains the refit), so an item later in pass 1 can be
 * partly warm: `?items=hploc` runs one item alone. The GPU driver keeps its own
 * shader cache (NVIDIA's DXCache, say), which no browser flag clears: a "cold"
 * pass here is cold for the browser, not necessarily for the driver.
 *
 * `?nonce`: the walk's modules (every one with `BVH_BOX_EXIT_PADDING`: the
 * batch kernels, the queries, the SDF, three's) get that constant nudged by a
 * random number of f32 steps, so their compiled code is new to every cache,
 * the driver's included: how much the driver's own cache saves. The builders
 * have no such constant and are not nudged.
 *
 * Options: `?items=a,b` (names below), `?passes=N` (default 2), `?nonce`.
 */

import { DoubleSide, Mesh, PerspectiveCamera, Scene, TorusKnotGeometry } from 'three';
import { Fn, If, instanceIndex, normalWorld, positionWorld, select, storage, uint, vec3, vec4 } from 'three/tsl';
import { MeshBasicNodeMaterial, StorageBufferAttribute, WebGPURenderer } from 'three/webgpu';
import { GpuHplocBuilder } from '../../../../src/gpu-hploc.ts';
import { GpuLbvhBuilder } from '../../../../src/gpu-lbvh.ts';
import { GpuMeshRefitKernels } from '../../../../src/gpu-mesh-refit.ts';
import { GpuNodeQuantizer } from '../../../../src/gpu-quantize.ts';
import { GpuBvhRefitKernels } from '../../../../src/gpu-refit.ts';
import { SceneTrees } from '../../../../src/scene/scene-trees.ts';
import { BvhTraceNodes, SceneTreesObject3DSync, deviceFromRenderer } from '../../../../src/three/index.ts';
import { DEFAULT_TRIANGLE_FILTER_WGSL } from '../../../../src/trace/index.ts';
import { SdfGenerator } from '../../../../src/trace/sdf.ts';
import { GpuTopLevel } from '../../../../src/trace/top-level-gpu.ts';
import { ClosestPointBatchKernel, TraceBatchKernel } from '../../../../src/trace/trace-gpu.ts';
import { batchEntryWgsl, batchKernelWgsl, traceModuleWgsl } from '../../../../src/trace/wgsl.ts';
import { assert, runGpuTest } from '../../client.ts';

/** One thing to compile on a fresh device; resolves when every pipeline is ready. */
interface Item {
    readonly name: string;
    readonly what: string;
    readonly needsSubgroups?: boolean;
    /** Three items make their own device (WebGPURenderer) and time themselves (ms). */
    readonly three?: () => Promise<number>;
    readonly compile?: (device: GPUDevice) => Promise<void>;
}

const EXIT_PADDING_WGSL = 'const BVH_BOX_EXIT_PADDING: f32 = 1.0000004;';
/** `?nonce`: f32 steps added to the exit padding for this page load (the same for both passes). */
const nonceSteps = new URLSearchParams(location.search).has('nonce') ? 1 + Math.floor(Math.random() * 4000) : 0;

/** With `?nonce`, makes the device's walk modules new to every shader cache (see the file comment). */
function nudged(device: GPUDevice): GPUDevice {
    if (nonceSteps === 0) return device;
    const create = device.createShaderModule.bind(device);
    const padding = Math.fround(1.0000004 + nonceSteps * 2 ** -23);
    device.createShaderModule = (descriptor) => create({ ...descriptor, code: descriptor.code.replace(EXIT_PADDING_WGSL, `const BVH_BOX_EXIT_PADDING: f32 = ${padding.toPrecision(9)};`) });
    return device;
}

/** One compute pipeline from WGSL, asynchronously, layout 'auto'. */
async function pipeline(device: GPUDevice, label: string, code: string, entryPoint: string): Promise<void> {
    await device.createComputePipelineAsync({ label, layout: 'auto', compute: { module: device.createShaderModule({ label, code }), entryPoint } });
}

/** A caller's shapecast: count the triangles within a sphere (query-oracle's kind of hook set). */
const SHAPECAST_HOOKS_WGSL = /* wgsl */ `
var<private> ctCenter: vec3f;
var<private> ctRadiusSq: f32;
var<private> ctCount: u32;

fn bvhShapecastBounds(bounds: BvhShapecastBounds) -> f32 {
    let distanceSq = bvhPointBoxDistanceSq(ctCenter, bounds.min, bounds.max);
    return select(-1.0, distanceSq, distanceSq <= ctRadiusSq);
}

fn bvhShapecastKeep(score: f32) -> bool {
    return true;
}

fn bvhShapecastTriangle(triangle: BvhShapecastTriangle) -> bool {
    let offset = bvhClosestPointToTriangle(ctCenter, triangle.a, triangle.b, triangle.c).point - ctCenter;
    if (dot(offset, offset) <= ctRadiusSq) {
        ctCount += 1u;
    }
    return false;
}

@group(1) @binding(0) var<storage, read> ctSpheres: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> ctCounts: array<u32>;

@compute @workgroup_size(64)
fn ctShapecast(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= arrayLength(&ctSpheres)) {
        return;
    }
    ctCenter = ctSpheres[id.x].xyz;
    ctRadiusSq = ctSpheres[id.x].w * ctSpheres[id.x].w;
    ctCount = 0u;
    _ = bvhShapecast();
    ctCounts[id.x] = ctCount;
}
`;

/** A small scene in SceneTrees on the renderer's device, and BvhTraceNodes over it (not timed). */
async function threeSetup(queries: boolean) {
    const renderer = new WebGPURenderer();
    await renderer.init();
    const device = nudged(deviceFromRenderer(renderer));
    const scene = new Scene();
    scene.add(new Mesh(new TorusKnotGeometry(1, 0.3, 64, 8), new MeshBasicNodeMaterial({ side: DoubleSide })));
    scene.updateMatrixWorld(true);
    const trees = await SceneTrees.create(device, { label: 'compile-times', holdRefinements: true, workers: { size: 1 } });
    const sync = new SceneTreesObject3DSync(trees, scene);
    const frame = () => {
        sync.update();
        const encoder = device.createCommandEncoder();
        trees.update(encoder);
        device.queue.submit([encoder.finish()]);
    };
    frame();
    await trees.whenBuilt();
    frame();
    const bvh = new BvhTraceNodes(renderer, trees, { queries });
    bvh.update();
    await device.queue.onSubmittedWorkDone();
    const dispose = () => {
        bvh.dispose();
        sync.dispose();
        trees.dispose();
        renderer.dispose();
    };
    return { renderer, device, scene, bvh, dispose };
}

const ITEMS: readonly Item[] = [
    { name: 'trace-closest-quantized', what: 'ray walk, closest-hit batch kernel, quantized nodes (the default)', compile: (device) => pipeline(device, 'closest q', batchKernelWgsl('closest', { nodeFormat: 'quantized' }), 'bvhBatch') },
    { name: 'trace-closest-f32', what: 'ray walk, closest-hit batch kernel, f32 nodes', compile: (device) => pipeline(device, 'closest f32', batchKernelWgsl('closest', { nodeFormat: 'f32' }), 'bvhBatch') },
    { name: 'trace-any-quantized', what: 'ray walk, any-hit batch kernel, quantized nodes', compile: (device) => pipeline(device, 'any q', batchKernelWgsl('any', { nodeFormat: 'quantized' }), 'bvhBatch') },
    { name: 'trace-closest-indexed', what: 'ray walk, closest hit, quantized nodes, indexed leaves', compile: (device) => pipeline(device, 'closest q indexed', batchKernelWgsl('closest', { nodeFormat: 'quantized', triangleLeaves: 'indexed' }), 'bvhBatch') },
    {
        name: 'trace-closest-with-queries',
        what: 'closest-hit batch kernel in a module with queries: true (the query walks unused)',
        compile: (device) => pipeline(device, 'closest + queries', `${traceModuleWgsl({ triangleFilter: DEFAULT_TRIANGLE_FILTER_WGSL, queries: true })}\n${batchEntryWgsl('closest', 1)}`, 'bvhBatch'),
    },
    {
        name: 'trace-closest-sync',
        what: 'trace-closest-quantized made with createComputePipeline, waited for with onSubmittedWorkDone',
        compile: async (device) => {
            const code = batchKernelWgsl('closest', { nodeFormat: 'quantized' });
            device.createComputePipeline({ label: 'closest sync', layout: 'auto', compute: { module: device.createShaderModule({ label: 'closest sync', code }), entryPoint: 'bvhBatch' } });
            await device.queue.onSubmittedWorkDone();
        },
    },
    {
        name: 'trace-kernel',
        what: 'TraceBatchKernel.create(): closest and any together, quantized (the library path)',
        compile: async (device) => {
            await TraceBatchKernel.create(device);
        },
    },
    {
        name: 'closest-point-quantized',
        what: 'ClosestPointBatchKernel.create(), quantized nodes',
        compile: async (device) => {
            (await ClosestPointBatchKernel.create(device, { nodeFormat: 'quantized' })).destroy();
        },
    },
    {
        name: 'closest-point-f32',
        what: 'ClosestPointBatchKernel.create(), f32 nodes',
        compile: async (device) => {
            (await ClosestPointBatchKernel.create(device, { nodeFormat: 'f32' })).destroy();
        },
    },
    {
        name: 'shapecast',
        what: "a caller's shapecast (shapecast: hooks), quantized nodes",
        compile: (device) => pipeline(device, 'shapecast', `${traceModuleWgsl({ triangleFilter: DEFAULT_TRIANGLE_FILTER_WGSL, shapecast: SHAPECAST_HOOKS_WGSL })}`, 'ctShapecast'),
    },
    {
        name: 'sdf',
        what: 'SdfGenerator.create(): the SDF kernel (closest point and the inside vote), quantized',
        compile: async (device) => {
            (await SdfGenerator.create(device)).destroy();
        },
    },
    {
        name: 'hploc',
        what: 'GpuHplocBuilder.prepare(): H-PLOC with its radix sort and scan',
        needsSubgroups: true,
        compile: async (device) => {
            await new GpuHplocBuilder(device).prepare();
        },
    },
    {
        name: 'lbvh',
        what: 'GpuLbvhBuilder.prepare(): LBVH with its radix sort and scan',
        needsSubgroups: true,
        compile: async (device) => {
            await new GpuLbvhBuilder(device).prepare();
        },
    },
    {
        name: 'refit',
        what: 'GpuBvhRefitKernels.prepare(): the refit (clear, climb, repair)',
        needsSubgroups: true,
        compile: async (device) => {
            await new GpuBvhRefitKernels(device).prepare();
        },
    },
    {
        name: 'mesh-refit',
        what: 'GpuMeshRefitKernels.prepare(): the mesh refit (triangle boxes, the indexed copy) and the refit',
        needsSubgroups: true,
        compile: async (device) => {
            await new GpuMeshRefitKernels(device).prepare();
        },
    },
    {
        name: 'quantize',
        what: 'GpuNodeQuantizer.create(): the quantize pass',
        compile: async (device) => {
            await GpuNodeQuantizer.create(device);
        },
    },
    {
        name: 'top-level',
        what: 'GpuTopLevel.prepare(): the GPU top level (gather and its refit)',
        needsSubgroups: true,
        compile: async (device) => {
            const topLevel = new GpuTopLevel(device, { buildIn: 'main' });
            await topLevel.prepare();
            topLevel.dispose();
        },
    },
    {
        name: 'three-compute',
        what: 'BvhTraceNodes in a TSL compute kernel (traceClosest per thread): the first computeAsync to the queue drained, less the second',
        three: async () => {
            const { renderer, device, bvh, dispose } = await threeSetup(false);
            const out = storage(new StorageBufferAttribute(new Float32Array(64 * 4), 4), 'vec4', 64);
            const node = Fn(() => {
                If(instanceIndex.lessThan(uint(64)), () => {
                    const hit = bvh.traceClosest(vec3(0, 0, 5), vec3(0, 0, -1), 0, 1e6).toVar();
                    out.element(instanceIndex).assign(vec4(hit.get('t'), 0, 0, 0));
                });
            })().compute(64, [64]);
            let start = performance.now();
            await renderer.computeAsync(node);
            await device.queue.onSubmittedWorkDone();
            const first = performance.now() - start;
            start = performance.now();
            await renderer.computeAsync(node);
            await device.queue.onSubmittedWorkDone();
            const second = performance.now() - start;
            dispose();
            return first - second;
        },
    },
    {
        name: 'three-material',
        what: 'BvhTraceNodes in a node material (an any-hit shadow ray per fragment): renderer.compileAsync',
        three: async () => {
            const { renderer, scene, bvh, dispose } = await threeSetup(false);
            const material = new MeshBasicNodeMaterial();
            material.colorNode = Fn(() => select(bvh.traceAny(positionWorld.add(normalWorld.mul(1e-3)), vec3(0, 1, 0), 0, 1e6), vec3(0.2), vec3(1)))();
            const mesh = new Mesh(new TorusKnotGeometry(0.5, 0.1, 32, 4), material);
            scene.add(mesh);
            const camera = new PerspectiveCamera();
            camera.position.set(0, 0, 5);
            const start = performance.now();
            await renderer.compileAsync(scene, camera);
            const ms = performance.now() - start;
            dispose();
            return ms;
        },
    },
];

const query = new URLSearchParams(location.search);
const wanted = query.get('items')?.split(',');
const passes = Math.max(1, Number(query.get('passes')) || 2);
const items = wanted ? ITEMS.filter((item) => wanted.includes(item.name)) : ITEMS;
assert(items.length > 0, `no item named ${wanted?.join(', ')}; the items: ${ITEMS.map((item) => item.name).join(', ')}`);

await runGpuTest(async ({ adapter, log }) => {
    const subgroups = adapter.features.has('subgroups');
    const results: Record<string, { ms: number[]; what: string }> = {};
    for (let pass = 0; pass < passes; pass++) {
        for (const item of items) {
            if (item.needsSubgroups && !subgroups) continue;
            let ms: number;
            if (item.three) ms = await item.three();
            else {
                const fresh = await navigator.gpu.requestAdapter();
                assert(fresh, 'requestAdapter returned null');
                const device = nudged(await fresh.requestDevice({ requiredFeatures: fresh.features.has('subgroups') ? ['subgroups'] : [] }));
                const start = performance.now();
                await item.compile!(device);
                ms = performance.now() - start;
                device.destroy();
            }
            const entry = (results[item.name] ??= { ms: [], what: item.what });
            entry.ms.push(Math.round(ms * 10) / 10);
            log(`pass ${pass + 1} ${item.name}: ${ms.toFixed(1)} ms`);
        }
    }
    for (const [name, entry] of Object.entries(results)) log(`${name}: ${entry.ms.map((ms) => ms.toFixed(1)).join(' / ')} ms (${entry.what})`);
    return { passes, nonceSteps, items: results };
}, { optionalFeatures: ['subgroups'] });

// Code samples for docs/tracing/*.mdx and docs/concepts/masks-filter-materials.mdx
// (see getting-started.ts for how they are checked).

// #region own-kernel
import { traceModuleEntries, traceModuleWgsl, type SceneTrees } from '@pmndrs/bvh';

// #endregion

export async function ownKernel(device: GPUDevice, trees: SceneTrees, points: GPUBuffer, shade: GPUBuffer, count: number): Promise<void> {
    // #region own-kernel
    // The walk's three bindings land at group 0, bindings 0-2: nodes, triangles, top level.
    const code = /* wgsl */ `
${traceModuleWgsl({ traceSlot: { group: 0, firstBinding: 0 } })}

@group(1) @binding(0) var<storage, read> points: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> shade: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= arrayLength(&shade)) { return; }
    let sun = normalize(vec3f(0.3, 1.0, 0.2));
    let blocked = bvhTraceAny(BvhRay(points[id.x].xyz, 1e-3, sun, 1e30));
    shade[id.x] = select(1.0, 0.0, blocked);
}`;
    const sceneLayout = device.createBindGroupLayout({ entries: traceModuleEntries().map((entry, binding) => ({ ...entry, binding })) });
    const ioLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
    });
    const pipeline = await device.createComputePipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [sceneLayout, ioLayout] }),
        compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
    });

    // Make the scene's bind group again whenever trees.version changes (a buffer grew).
    const { nodes, triangles, topLevel } = trees.buffers;
    const sceneGroup = device.createBindGroup({ layout: sceneLayout, entries: [nodes, triangles, topLevel].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const ioGroup = device.createBindGroup({ layout: ioLayout, entries: [{ binding: 0, resource: { buffer: points } }, { binding: 1, resource: { buffer: shade } }] });

    const encoder = device.createCommandEncoder();
    trees.update(encoder); // before anything traces
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, sceneGroup);
    pass.setBindGroup(1, ioGroup);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    device.queue.submit([encoder.finish()]);
    // #endregion
}

export function matchScene(trees: SceneTrees): string {
    // #region match-scene
    // Compile the walk for the scene's formats (all three are off by default).
    const options = { triangleLeaves: trees.triangleLeaves, positionLayout: trees.positionLayout, nodeFormat: trees.nodeFormat };
    const walk = traceModuleWgsl({ traceSlot: { group: 0, firstBinding: 0 }, ...options });
    const entries = traceModuleEntries(options); // a fourth binding, the positions, for indexed leaves
    // #endregion
    return walk + entries.length;
}

export function maskedWgsl(): string {
    // #region masked
    const code = /* wgsl */ `
${traceModuleWgsl()}
fn shadowFrom(p: vec3f, l: vec3f) -> bool {
    // Only instances whose mask has bit 0 cast shadows here.
    return bvhTraceAnyMasked(BvhRay(p, 1e-3, l, 1e30), 0x1u);
}
fn firstHit(o: vec3f, d: vec3f) -> BvhHit {
    let hit = bvhTraceClosest(BvhRay(o, 0.0, d, 1e30)); // all ones: every instance with a non-zero mask
    // hit.t < 0: a miss. Else hit.instance, hit.triangle, hit.barycentrics (corners 1 and 2),
    // hit.frontFace, hit.material (BVH_NO_MATERIAL for none) and hit.normal (world, geometric).
    return hit;
}`;
    // #endregion
    return code;
}

export function filterWgsl(): string {
    // #region filter
    // Asked only for triangles with TRIANGLE_FLAG_FILTER set. true: the ray passes through.
    const triangleFilter = /* wgsl */ `
fn bvhTrianglePasses(candidate: BvhCandidate) -> bool {
    // Shadow rays pass through flagged triangles (foliage cards, glass); closest-hit rays still see them.
    return candidate.query == BVH_QUERY_ANY;
}`;
    const code = traceModuleWgsl({ triangleFilter });
    // #endregion
    return code;
}

// #region filter-flags
import { TRIANGLE_FLAG_FILTER, type MeshGeometry } from '@pmndrs/bvh';

// #endregion

export function filterFlags(trees: SceneTrees, positions: Float32Array, indices: Uint32Array, isGlass: (triangle: number) => boolean): number {
    // #region filter-flags
    const triangleCount = indices.length / 3;
    const triangleFlags = new Uint32Array(triangleCount);
    for (let triangle = 0; triangle < triangleCount; triangle++) if (isGlass(triangle)) triangleFlags[triangle] = TRIANGLE_FLAG_FILTER;

    const window: MeshGeometry = { positions, indices, triangleFlags }; // only bit 0 may be set
    const mesh = trees.addMesh(window);
    // #endregion
    return mesh;
}

export function masksAndMaterials(trees: SceneTrees, mesh: number, matrix: Float32Array): void {
    // #region masks-materials
    // 8-bit instance masks (default 0xff) and a material index per instance (0-65534).
    trees.addInstance({ mesh, objectToWorld: matrix, mask: 0b0000_0011, material: 12 });
    // A masked trace skips an instance when (instanceMask & rayMask) == 0.
    // hit.material is 12 for this instance's hits, BVH_NO_MATERIAL where none was given.
    // #endregion
}

export function noMasks(): string {
    // #region no-masks
    // No *Masked functions, no mask test in the walk, every instance visible.
    const code = traceModuleWgsl({ rayMasks: false });
    // #endregion
    return code;
}

// #region batch
import { HIT_RECORD_BYTES, packRays, RAY_BYTES, TraceBatchKernel } from '@pmndrs/bvh';

// #endregion

export async function batch(trees: SceneTrees): Promise<void> {
    const device = trees.device;
    // #region batch
    const kernel = await TraceBatchKernel.create(device, {
        triangleLeaves: trees.triangleLeaves,
        positionLayout: trees.positionLayout,
        nodeFormat: trees.nodeFormat,
        hitNormal: true, // 40-byte records with material and normal; false: 24 bytes
    });
    const count = 4096;
    const rays = device.createBuffer({ size: count * RAY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const results = device.createBuffer({ size: count * HIT_RECORD_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const shadowResults = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(rays, 0, packRays(Array.from({ length: count }, (_, i) => ({ origin: [i * 0.01, 1, 5], direction: [0, 0, -1], tMin: 0, tMax: 1e30 }))));

    const encoder = device.createCommandEncoder();
    kernel.encode(encoder, trees.traceBuffers, { mode: 'closest', rays, results, count });
    kernel.encode(encoder, trees.traceBuffers, { mode: 'any', rays, results: shadowResults, count, mask: 0x1 }); // one u32 per ray, 1 = blocked
    device.queue.submit([encoder.finish()]);
    // #endregion
}

// #region cpu-walk
import { buildMeshBvhWithTinyBvh, buildTraceScene, createCpuTracer, createTinyBvhBuilder, IDENTITY_TRANSFORM } from '@pmndrs/bvh';

// #endregion

export async function cpuWalk(geometry: MeshGeometry): Promise<number> {
    // #region cpu-walk
    const tiny = await createTinyBvhBuilder();
    const scene = await buildTraceScene([geometry], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY_TRANSFORM }], {
        build: (mesh) => buildMeshBvhWithTinyBvh(mesh, tiny),
    });
    // The WGSL walk, step for step in TypeScript, over the same buffers.
    const tracer = createCpuTracer(scene);
    const hit = tracer.closest({ origin: [0, 0, 5], direction: [0, 0, -1], tMin: 0, tMax: 1e30 });
    const blocked = tracer.any({ origin: [0, 0, 5], direction: [0, 0, -1], tMin: 0, tMax: 1e30 }, { mask: 0x1 });
    // #endregion
    return (hit ? 1 : 0) + (blocked ? 1 : 0);
}

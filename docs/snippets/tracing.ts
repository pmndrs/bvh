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
    // Compile the walk for the scene's formats (the defaults: records, vec4 positions, quantized nodes).
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

// #region triangle-materials
import { TRIANGLE_MATERIAL_NONE } from '@pmndrs/bvh';

// #endregion

export function triangleMaterials(trees: SceneTrees, positions: Float32Array, indices: Uint32Array, matrix: Float32Array): void {
    // #region triangle-materials
    // One material per triangle (0-65534), by original triangle; TRIANGLE_MATERIAL_NONE (0xffff) takes the instance's.
    const triangleMaterials = new Uint16Array(indices.length / 3).fill(TRIANGLE_MATERIAL_NONE);
    triangleMaterials.fill(3, 0, 100); // triangles 0-99 are material 3
    const mesh = trees.addMesh({ positions, indices, triangleMaterials });
    trees.addInstance({ mesh, objectToWorld: matrix, material: 7 });
    // hit.material: 3 on triangles 0-99, 7 (the instance's) on the others.
    // #endregion
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

// #region gpu-closest-point
import { CLOSEST_POINT_RECORD_BYTES, ClosestPointBatchKernel, POINT_QUERY_BYTES } from '@pmndrs/bvh';

// #endregion

export async function gpuClosestPoint(trees: SceneTrees, points: readonly [number, number, number][]): Promise<void> {
    const device = trees.device;
    // #region gpu-closest-point
    const kernel = await ClosestPointBatchKernel.create(device, {
        triangleLeaves: trees.triangleLeaves,
        positionLayout: trees.positionLayout,
        nodeFormat: trees.nodeFormat,
    });
    const count = points.length;
    // One query per point: vec4f(point.xyz, maxDistance). Only triangles nearer than maxDistance count.
    const data = new Float32Array(count * 4);
    points.forEach((point, i) => data.set([...point, 1e30], i * 4));
    const queries = device.createBuffer({ size: count * POINT_QUERY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const results = device.createBuffer({ size: count * CLOSEST_POINT_RECORD_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(queries, 0, data);

    const encoder = device.createCommandEncoder();
    kernel.encode(encoder, trees.traceBuffers, { queries, results, count, mask: 0xff });
    device.queue.submit([encoder.finish()]);
    // 48-byte records, words in CLOSEST_POINT_RECORD_WORD: distance (negative: nothing within maxDistance),
    // instance, triangle, material, barycentrics, point (world) and normal (the triangle's authored front side).
    // #endregion
}

export function gpuQueriesWgsl(): string {
    // #region gpu-queries
    const code = /* wgsl */ `
${traceModuleWgsl({ queries: true })}
fn nearestSurface(p: vec3f) -> vec3f {
    let hit = bvhClosestPoint(p, 10.0);   // world space; hit.distance < 0: nothing within 10
    return select(p, hit.point, hit.distance >= 0.0);
}
fn overlaps(p: vec3f, axes: mat3x3f) -> bool {
    return bvhIntersectsSphere(p, 0.5)
        || bvhIntersectsBox(p - vec3f(1.0), p + vec3f(1.0))
        || bvhIntersectsOrientedBox(p, vec3f(2.0, 0.5, 0.5), axes);   // axes: orthonormal columns
}`;
    // #endregion
    return code;
}

export function gpuShapecastWgsl(): string {
    // #region gpu-shapecast
    // Count the triangles within a radius of a point: the caller's three hooks, compiled into the walk.
    const shapecast = /* wgsl */ `
var<private> queryCenter: vec3f;
var<private> queryRadiusSq: f32;
var<private> touched: u32;

fn bvhShapecastBounds(bounds: BvhShapecastBounds) -> f32 {
    let distanceSq = bvhPointBoxDistanceSq(queryCenter, bounds.min, bounds.max);
    return select(-1.0, distanceSq, distanceSq <= queryRadiusSq);   // < 0 skips the box; else the lower goes first
}

fn bvhShapecastKeep(score: f32) -> bool {
    return true;   // asked when a pushed box is popped, with its score: false skips it
}

fn bvhShapecastTriangle(triangle: BvhShapecastTriangle) -> bool {
    let offset = bvhClosestPointToTriangle(queryCenter, triangle.a, triangle.b, triangle.c).point - queryCenter;
    if (dot(offset, offset) <= queryRadiusSq) {
        touched += 1u;   // triangle.instance and triangle.triangle name it; its corners are in world space
    }
    return false;   // true stops the cast
}`;
    const code = /* wgsl */ `
${traceModuleWgsl({ shapecast })}
fn countNear(p: vec3f, radius: f32) -> u32 {
    queryCenter = p;
    queryRadiusSq = radius * radius;
    touched = 0u;
    _ = bvhShapecast();
    return touched;
}`;
    // #endregion
    return code;
}

// #region triangle-helpers
import { triangleGeometryWgsl } from '@pmndrs/bvh';

// #endregion

export function triangleHelpers(): string {
    // #region triangle-helpers
    // The walk's watertight test and the closest point on one triangle, with no scene and no bindings.
    const code = /* wgsl */ `
${triangleGeometryWgsl()}
fn pick(ray: BvhRay, a: vec3f, b: vec3f, c: vec3f) -> f32 {
    let hit = bvhIntersectRayTriangle(ray, a, b, c);   // hit.hit, hit.t, hit.barycentrics, hit.frontFacing
    return select(-1.0, hit.t, hit.hit);
}
fn snap(p: vec3f, a: vec3f, b: vec3f, c: vec3f) -> vec3f {
    return bvhClosestPointToTriangle(p, a, b, c).point;   // .barycentrics: the weights of b and c
}`;
    // #endregion
    return code;
}

// #region sdf-generate
import { createSdfTexture, SdfGenerator } from '@pmndrs/bvh';

// #endregion

export async function sdfGenerate(trees: SceneTrees): Promise<GPUTexture> {
    const device = trees.device;
    // #region sdf-generate
    // One generator per scene format (it compiles the walk with the queries); it serves any scene of that format.
    const generator = await SdfGenerator.create(device, {
        triangleLeaves: trees.triangleLeaves,
        positionLayout: trees.positionLayout,
        nodeFormat: trees.nodeFormat,
    });
    // 96 x 96 x 96 voxel centres over a world-space box.
    const grid = { min: [-2, -2, -2], max: [2, 2, 2], resolution: 96 } as const;
    const texture = createSdfTexture(device, grid); // r32float, 3d: storage, sampling and copies
    const { ms } = await generator.generate(trees.traceBuffers, { texture }, grid);
    // The texture holds the distance to the nearest triangle at each voxel centre, negative inside.
    // #endregion
    console.log(`SDF in ${ms.toFixed(1)} ms`);
    return texture;
}

export function sdfEncode(trees: SceneTrees, generator: SdfGenerator, encoder: GPUCommandEncoder): GPUBuffer {
    const device = trees.device;
    // #region sdf-encode
    // Into a buffer (x fastest, then y, then z) in your own encoder, from a generator made with output: 'buffer':
    // a narrow band, the winding rule, the instances with mask bit 0, and z slices 0-15 only.
    const grid = { min: [-2, -2, -2], max: [2, 2, 2], resolution: [64, 64, 32] } as const;
    const buffer = device.createBuffer({ size: 64 * 64 * 32 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    generator.encode(encoder, trees.traceBuffers, { buffer }, grid, { maxDistance: 0.25, sign: 'winding', mask: 0x1, slices: { start: 0, count: 16 } });
    // #endregion
    return buffer;
}

export function sdfInsideWgsl(): string {
    // #region sdf-inside-wgsl
    // A signed distance in your own kernel: the closest point, negated where the inside test says inside.
    const code = /* wgsl */ `
${traceModuleWgsl({ queries: true })}
fn signedDistance(p: vec3f) -> f32 {
    let hit = bvhClosestPoint(p, 1e30);
    return select(hit.distance, -hit.distance, bvhIsInside(p));
}
fn insideByWinding(p: vec3f) -> bool {
    return bvhInsideVotes(p).y >= 2u;   // .x: rays voting inside by parity, .y: by winding (of 3)
}`;
    // #endregion
    return code;
}

// #region sdf-cpu
import { CpuMeshBvh, generateSdfCpu } from '@pmndrs/bvh';

// #endregion

export async function sdfCpu(geometry: MeshGeometry): Promise<{ inside: boolean; center: number }> {
    // #region sdf-cpu
    const tree = await CpuMeshBvh.create(geometry); // world-space positions
    const inside = tree.isPointInside([0, 0, 0]); // or isPointInside(point, 'winding'), or insideVotes(point)
    // The same field on the CPU, voxel for voxel the GPU's points: a reference, or a field with no GPU.
    const field = generateSdfCpu(tree, { min: [-2, -2, -2], max: [2, 2, 2], resolution: 9 });
    const center = field[(4 * 9 + 4) * 9 + 4]; // voxel (4, 4, 4) is the box's centre
    // #endregion
    return { inside, center };
}

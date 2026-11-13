/**
 * Query oracle: the GPU spatial queries (`traceModuleWgsl({ queries, shapecast })`,
 * `ClosestPointBatchKernel`; PARITY 8.5, 8.6, 8.8a) against the CPU query code.
 *
 * The scenes are trace-oracle's cases (cases.ts): thin walls, instancing,
 * mirrored and non-uniformly scaled boxes, random soups under random rotations,
 * scales and mirrors, 200 instances of one mesh, a 30,000-triangle mesh. Each
 * instance also gets an instance mask (one of three bits, some a fourth) so
 * masked queries leave instances out.
 *
 * **The reference** is `CpuMeshBvh`, the package's CPU queries: every kept
 * instance's mesh is moved to world space (in doubles, rounded once to f32)
 * and built as its own tree, and the scene's answer is the best over the
 * instances (`closestPointToPoint`, `intersectsSphere`, `intersectsBox`
 * with and without a matrix, `shapecast`). The GPU walks the real scene: the
 * instanced top level and object-space mesh trees (or the merged world-space
 * trees), in every node and triangle format:
 *
 * - `f32-records`, `quantized-records`, `quantized-indexed`, `f32-indexed-packed`;
 * - `merged-*`: two of every three instances marked static and merged, so hits must
 *   name their source instance and triangle;
 * - `quantized-records-nomasks`: the module with ray masks compiled out;
 * - `materials-merged-quantized-indexed`: per-triangle materials (trace-oracle's
 *   materials.ts), so a closest hit's material is its triangle's, else its instance's.
 *
 * **Checks**, per query point (random in the scene's box, and near or on
 * random triangles):
 * - closest point, unlimited (`maxDistance` 1e30), limited (0.7 or 1.3 times the
 *   reference distance: found exactly when the reference is nearer), and masked
 *   (mask 1); the distance must equal the reference's within the tolerance. Ties
 *   (another triangle at the same distance) agree. The reported triangle must
 *   lie at the reported distance, the point must be its barycentric point, and
 *   the normal (authored front side) and material must be that triangle's;
 * - `bvhIntersectsSphere` with radii around the reference distance, `bvhIntersectsBox`
 *   and `bvhIntersectsOrientedBox` with random sizes and rotations, against
 *   `intersectsSphere` and `intersectsBox`; a shape within the tolerance of
 *   touching is ambiguous (the reference answers differently for the shape
 *   grown and shrunk by the tolerance) and not counted;
 * - a caller's shapecast (`shapecast: hooks`) counting the triangles a sphere
 *   touches, against `shapecast` with the same tests: the count must lie
 *   between the reference's for the sphere shrunk and grown by the tolerance;
 * - the standalone helpers (`triangleGeometryWgsl()`): `bvhClosestPointToTriangle`
 *   against `triangleClosestPointToPoint`, `bvhIntersectRayTriangle` on rays aimed
 *   inside and outside random triangles.
 *
 * `?profile=ci` (SwiftShader in CI, `BVH_PAGE_QUERY=profile=ci`): every variant,
 * case and check, with a tenth of the queries. The pass rules are the same.
 */

import {
    assembleTraceScene,
    mergeStaticGeometry,
    createMeshBvhBuilder,
    ClosestPointBatchKernel,
    CLOSEST_POINT_RECORD_BYTES,
    CLOSEST_POINT_RECORD_WORD,
    NO_MATERIAL,
    POINT_QUERY_BYTES,
    traceModuleEntries,
    traceModuleWgsl,
    triangleGeometryWgsl,
    uploadTraceScene,
    type MeshBvh,
    type MeshBvhBuilder,
    type MeshGeometry,
    type TraceSceneData,
    type TraceSceneInstance,
} from '../../../../src/trace/index.ts';
import { createTinyBvhBuilder, CpuMeshBvh, type TinyBvhBuilder } from '../../../../src/index.ts';
import { sphereIntersectsBox, triangleClosestPointToPoint, BvhVector3 } from '../../../../src/query/shapes.ts';
import { TRIANGLE_MATERIAL_NONE } from '../../../../src/mesh-input.ts';
import { createRandom, randomUnitVector, type Random, type Vector3 } from '../../../../src/trace/oracle-fixtures.ts';
import { assert, runGpuTest } from '../../client.ts';
import { buildOracleCases, SMALL_SIZES, type OracleCase } from '../trace-oracle/cases.ts';
import { withTriangleMaterials } from '../trace-oracle/materials.ts';

interface Variant {
    readonly name: string;
    readonly nodeFormat: 'f32' | 'quantized';
    readonly triangleLeaves: 'records' | 'indexed';
    readonly positionLayout?: 'vec4' | 'packed';
    readonly merged?: boolean;
    readonly rayMasks?: boolean;
    readonly triangleMaterials?: boolean;
}

const VARIANTS: readonly Variant[] = [
    { name: 'f32-records', nodeFormat: 'f32', triangleLeaves: 'records' },
    { name: 'quantized-records', nodeFormat: 'quantized', triangleLeaves: 'records' },
    { name: 'quantized-indexed', nodeFormat: 'quantized', triangleLeaves: 'indexed' },
    { name: 'f32-indexed-packed', nodeFormat: 'f32', triangleLeaves: 'indexed', positionLayout: 'packed' },
    { name: 'merged-quantized-records', nodeFormat: 'quantized', triangleLeaves: 'records', merged: true },
    { name: 'merged-f32-indexed', nodeFormat: 'f32', triangleLeaves: 'indexed', merged: true },
    { name: 'merged-quantized-indexed-packed', nodeFormat: 'quantized', triangleLeaves: 'indexed', positionLayout: 'packed', merged: true },
    { name: 'quantized-records-nomasks', nodeFormat: 'quantized', triangleLeaves: 'records', rayMasks: false },
    { name: 'materials-merged-quantized-indexed', nodeFormat: 'quantized', triangleLeaves: 'indexed', merged: true, triangleMaterials: true },
];

const pageQuery = new URLSearchParams(location.search);
const profile = pageQuery.get('profile') ?? 'full';
assert(profile === 'full' || profile === 'ci', `unknown profile "${profile}" (use full or ci)`);
/** Query points per case, and helper triangles. */
const QUERIES = profile === 'ci' ? 200 : 2000;
const HELPER_CASES = profile === 'ci' ? 2000 : 20000;

/** The masked pass's mask: instances with bit 0. */
const QUERY_MASK = 1;
/** maxDistance for an unlimited query (clamped to CLOSEST_POINT_MAX_DISTANCE on the GPU). */
const UNLIMITED = 1e30;
/** Tolerance in world units: this share of the scene's size plus the point's distance from the origin. */
const RELATIVE_TOLERANCE = 4e-6;
/** Normals must agree to this cosine. */
const NORMAL_COSINE = 0.9999;

// ---- The scene's instances, for the reference ----------------------------------------

/** The case's instances with masks: bit k % 3, and bit 3 on every seventh. */
function withMasks(oracleCase: OracleCase): OracleCase {
    return { ...oracleCase, instances: oracleCase.instances.map((instance, k) => ({ ...instance, mask: (1 << k % 3) | (k % 7 === 0 ? 8 : 0) })) };
}

function triangleCount(mesh: MeshGeometry): number {
    return (mesh.indices ? mesh.indices.length : mesh.positions.length / 3) / 3;
}

function determinant3(e: ArrayLike<number>): number {
    return e[0] * (e[5] * e[10] - e[9] * e[6]) - e[4] * (e[1] * e[10] - e[9] * e[2]) + e[8] * (e[1] * e[6] - e[5] * e[2]);
}

/** One kept instance in world space: its own CPU tree and its corners. */
interface WorldInstance {
    readonly instance: TraceSceneInstance;
    readonly mesh: MeshGeometry;
    readonly tree: CpuMeshBvh;
    /** World corners per original triangle, 9 doubles each. */
    readonly corners: Float64Array;
    readonly mirrored: boolean;
    readonly mask: number;
    /** World box. */
    readonly box: Float64Array;
}

/** The reference's instances: what the top level keeps (visible, contributing, mask not 0, invertible), each in world space. */
function worldInstances(oracleCase: OracleCase, tiny: TinyBvhBuilder): WorldInstance[] {
    const out: WorldInstance[] = [];
    for (const instance of oracleCase.instances) {
        if (instance.visible === false || instance.contributesToGi === false) continue;
        const mask = instance.mask ?? 0xff;
        if (mask === 0) continue;
        const e = instance.objectToWorld;
        const det = determinant3(e);
        if (det === 0) continue;
        const mesh = oracleCase.meshes[instance.meshIndex];
        const count = triangleCount(mesh);
        const corners = new Float64Array(count * 9);
        const positions = new Float32Array(count * 9);
        const box = new Float64Array([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
        for (let t = 0; t < count; t++) {
            for (let c = 0; c < 3; c++) {
                const v = mesh.indices ? mesh.indices[t * 3 + c] : t * 3 + c;
                const x = mesh.positions[v * 3], y = mesh.positions[v * 3 + 1], z = mesh.positions[v * 3 + 2];
                for (let k = 0; k < 3; k++) {
                    const w = e[k] * x + e[4 + k] * y + e[8 + k] * z + e[12 + k];
                    corners[t * 9 + c * 3 + k] = Math.fround(w);
                    positions[t * 9 + c * 3 + k] = w;
                    box[k] = Math.min(box[k], Math.fround(w));
                    box[3 + k] = Math.max(box[3 + k], Math.fround(w));
                }
            }
        }
        const tree = CpuMeshBvh.build({ positions }, tiny);
        out.push({ instance, mesh, tree, corners, mirrored: det < 0, mask, box });
    }
    return out;
}

function boxDistanceSq(box: ArrayLike<number>, p: Vector3): number {
    let sum = 0;
    for (let k = 0; k < 3; k++) {
        const d = Math.max(box[k] - p[k], 0, p[k] - box[3 + k]);
        sum += d * d;
    }
    return sum;
}

interface ReferenceClosest {
    readonly distance: number;
    readonly instanceIndex: number;
    readonly triangle: number;
}

/** The scene's closest point: the best of each instance's `closestPointToPoint`, instances nearest box first. */
function referenceClosest(instances: readonly WorldInstance[], p: Vector3, mask = 0xffffffff): ReferenceClosest | null {
    const order = instances.filter((entry) => (entry.mask & mask) !== 0).map((entry) => ({ entry, boxSq: boxDistanceSq(entry.box, p) })).sort((a, b) => a.boxSq - b.boxSq);
    let best: ReferenceClosest | null = null;
    for (const { entry, boxSq } of order) {
        if (best && boxSq > best.distance * best.distance) break;
        const hit = entry.tree.closestPointToPoint({ x: p[0], y: p[1], z: p[2] });
        if (hit && (!best || hit.distance < best.distance)) best = { distance: hit.distance, instanceIndex: entry.instance.instanceIndex, triangle: hit.faceIndex };
    }
    return best;
}

function referenceSphere(instances: readonly WorldInstance[], center: Vector3, radius: number): boolean {
    const sphere = { center: { x: center[0], y: center[1], z: center[2] }, radius };
    return instances.some((entry) => boxDistanceSq(entry.box, center) <= radius * radius && entry.tree.intersectsSphere(sphere));
}

/** Column-major 4x4 of translation * rotation (rotation given by columns). */
function placeMatrix(center: Vector3, axes: readonly Vector3[]): number[] {
    return [...axes[0], 0, ...axes[1], 0, ...axes[2], 0, ...center, 1];
}

function referenceBox(instances: readonly WorldInstance[], center: Vector3, half: Vector3, axes: readonly Vector3[] | null): boolean {
    const box = axes
        ? { min: { x: -half[0], y: -half[1], z: -half[2] }, max: { x: half[0], y: half[1], z: half[2] } }
        : { min: { x: center[0] - half[0], y: center[1] - half[1], z: center[2] - half[2] }, max: { x: center[0] + half[0], y: center[1] + half[1], z: center[2] + half[2] } };
    const matrix = axes ? placeMatrix(center, axes) : null;
    const reach = axes ? Math.hypot(half[0], half[1], half[2]) : 0;
    return instances.some((entry) => {
        // Cull instances whose box is beyond the query's bounding sphere (an oriented box) or box.
        if (axes ? boxDistanceSq(entry.box, center) > reach * reach : [0, 1, 2].some((k) => entry.box[3 + k] < center[k] - half[k] || entry.box[k] > center[k] + half[k])) return false;
        return entry.tree.intersectsBox(box, matrix);
    });
}

/** Triangles a sphere touches, over the scene: each instance's `shapecast` with three-mesh-bvh's sphere tests. */
function referenceCount(instances: readonly WorldInstance[], center: Vector3, radius: number): number {
    const sphere = { center: { x: center[0], y: center[1], z: center[2] }, radius };
    let count = 0;
    for (const entry of instances) {
        if (boxDistanceSq(entry.box, center) > radius * radius) continue;
        entry.tree.shapecast({
            intersectsBounds: (box) => sphereIntersectsBox(sphere, box),
            intersectsTriangle: (triangle) => {
                if (triangle.intersectsSphere(sphere)) count++;
                return false;
            },
        });
    }
    return count;
}

// ---- Query generation ---------------------------------------------------------------------

interface QueryCase {
    readonly point: Vector3;
    readonly closest: ReferenceClosest | null;
    readonly closestMasked: ReferenceClosest | null;
    readonly tolerance: number;
    /** Limited closest point: maxDistance and whether a point is expected (null: ambiguous). */
    readonly limit: number;
    readonly limitExpect: boolean | null;
    readonly radius: number;
    readonly sphere: boolean | null;
    readonly half: Vector3;
    readonly box: boolean | null;
    readonly axes: Vector3[];
    readonly orientedBox: boolean | null;
    readonly countRadius: number;
    readonly countRange: readonly [number, number];
}

function randomRotation(random: Random): Vector3[] {
    const [qx, qy, qz] = randomUnitVector(random);
    const angle = random() * Math.PI * 2;
    const s = Math.sin(angle / 2);
    const x = qx * s, y = qy * s, z = qz * s, w = Math.cos(angle / 2);
    const f = (v: number[]) => v.map(Math.fround) as Vector3;
    return [
        f([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)]),
        f([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)]),
        f([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)]),
    ];
}

/** An answer that holds for the shape shrunk and grown by the tolerance, or null when the two differ. */
function unambiguous(shrunk: boolean, grown: boolean): boolean | null {
    return shrunk === grown ? shrunk : null;
}

function buildQueries(instances: readonly WorldInstance[], seed: number, count: number): { queries: QueryCase[]; scale: number } {
    const random = createRandom(seed);
    const box = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    for (const entry of instances) for (let k = 0; k < 3; k++) {
        box[k] = Math.min(box[k], entry.box[k]);
        box[3 + k] = Math.max(box[3 + k], entry.box[3 + k]);
    }
    const scale = Math.max(box[3] - box[0], box[4] - box[1], box[5] - box[2]);
    const queries: QueryCase[] = [];
    for (let q = 0; q < count; q++) {
        let point: Vector3;
        if (q % 2 === 0) {
            // In the scene's box, grown by a quarter.
            point = [0, 1, 2].map((k) => box[k] + (random() * 1.5 - 0.25) * (box[3 + k] - box[k])) as Vector3;
        } else {
            // On or near a random triangle: a barycentric point, moved along its normal by up to 2% of the scene (every fourth: none).
            const entry = instances[Math.floor(random() * instances.length)];
            const t = Math.floor(random() * (entry.corners.length / 9));
            const c = entry.corners.subarray(t * 9, t * 9 + 9);
            let u = random(), v = random();
            if (u + v > 1) [u, v] = [1 - u, 1 - v];
            const e1 = [c[3] - c[0], c[4] - c[1], c[5] - c[2]], e2 = [c[6] - c[0], c[7] - c[1], c[8] - c[2]];
            const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            const length = Math.hypot(n[0], n[1], n[2]) || 1;
            const offset = q % 8 === 1 ? 0 : (random() * 2 - 1) * 0.02 * scale;
            point = [0, 1, 2].map((k) => c[k] + e1[k] * u + e2[k] * v + (n[k] / length) * offset) as Vector3;
        }
        point = point.map(Math.fround) as Vector3;
        const tolerance = RELATIVE_TOLERANCE * (scale + Math.hypot(...point));
        const closest = referenceClosest(instances, point);
        const closestMasked = referenceClosest(instances, point, QUERY_MASK);
        const distance = closest?.distance ?? scale;
        const limit = Math.fround(distance * (q % 2 === 0 ? 0.7 : 1.3));
        const limitExpect = closest === null ? false : Math.abs(limit - distance) <= tolerance ? null : distance < limit;
        // Sphere radii around the reference distance (some well inside, some just past it).
        const radius = Math.fround(distance * [0.5, 0.9, 1.1, 2, 1.0005, 0.9995][q % 6]);
        // A shape shrunk past nothing touches nothing.
        const sphere = unambiguous(radius > tolerance && referenceSphere(instances, point, radius - tolerance), referenceSphere(instances, point, radius + tolerance));
        const half = [0, 1, 2].map(() => Math.fround((0.02 + random() * 0.3) * scale)) as Vector3;
        const shrunk = half.map((h) => Math.max(0, h - tolerance)) as Vector3;
        const grown = half.map((h) => h + tolerance) as Vector3;
        const boxAnswer = unambiguous(referenceBox(instances, point, shrunk, null), referenceBox(instances, point, grown, null));
        const axes = randomRotation(random);
        const orientedBox = unambiguous(referenceBox(instances, point, shrunk, axes), referenceBox(instances, point, grown, axes));
        const countRadius = Math.fround((0.05 + random() * 0.25) * scale);
        const countRange = [countRadius > tolerance ? referenceCount(instances, point, countRadius - tolerance) : 0, referenceCount(instances, point, countRadius + tolerance)] as const;
        queries.push({ point, closest, closestMasked, tolerance, limit, limitExpect, radius, sphere, half, box: boxAnswer, axes, orientedBox, countRadius, countRange });
    }
    return { queries, scale };
}

// ---- GPU -------------------------------------------------------------------------------------

async function readBuffer(device: GPUDevice, source: GPUBuffer, size: number): Promise<ArrayBuffer> {
    const readback = device.createBuffer({ label: 'query-oracle readback', size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readback, 0, size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();
    return copy;
}

function storageBuffer(device: GPUDevice, label: string, data: ArrayBufferView | number): GPUBuffer {
    const size = typeof data === 'number' ? data : data.byteLength;
    const buffer = device.createBuffer({ label, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    if (typeof data !== 'number') device.queue.writeBuffer(buffer, 0, data as ArrayBufferView<ArrayBuffer>);
    return buffer;
}

const WORKGROUP = 64;

/**
 * The page's shapecast hooks: count the triangles the query's count sphere
 * touches (closest point within the radius), never stopping.
 */
const COUNT_HOOKS_WGSL = /* wgsl */ `
var<private> pageCountCenter: vec3f;
var<private> pageCountRadiusSq: f32;
var<private> pageCount: u32;

fn bvhShapecastBounds(bounds: BvhShapecastBounds) -> f32 {
    let distanceSq = bvhPointBoxDistanceSq(pageCountCenter, bounds.min, bounds.max);
    return select(-1.0, distanceSq, distanceSq <= pageCountRadiusSq);
}

fn bvhShapecastKeep(score: f32) -> bool {
    return true;
}

fn bvhShapecastTriangle(triangle: BvhShapecastTriangle) -> bool {
    let offset = bvhClosestPointToTriangle(pageCountCenter, triangle.a, triangle.b, triangle.c).point - pageCountCenter;
    if (dot(offset, offset) <= pageCountRadiusSq) {
        pageCount += 1u;
    }
    return false;
}
`;

/** The page's own kernel over the trace module: sphere, box, oriented box and the counting shapecast per query. */
const SHAPES_KERNEL_WGSL = /* wgsl */ `
struct PageQuery {
    centerRadius: vec4f,      // center, sphere radius
    halfCountRadius: vec4f,   // box half size, count sphere radius
    axis0: vec4f,
    axis1: vec4f,
    axis2: vec4f,
}

@group(1) @binding(0) var<storage, read> pageQueries: array<PageQuery>;
@group(1) @binding(1) var<storage, read_write> pageResults: array<vec4u>;

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= arrayLength(&pageQueries)) {
        return;
    }
    let query = pageQueries[id.x];
    let center = query.centerRadius.xyz;
    let half = query.halfCountRadius.xyz;
    let sphere = bvhIntersectsSphere(center, query.centerRadius.w);
    let box = bvhIntersectsBox(center - half, center + half);
    let oriented = bvhIntersectsOrientedBox(center, half, mat3x3f(query.axis0.xyz, query.axis1.xyz, query.axis2.xyz));
    pageCountCenter = center;
    pageCountRadiusSq = query.halfCountRadius.w * query.halfCountRadius.w;
    pageCount = 0u;
    _ = bvhShapecast();
    pageResults[id.x] = vec4u(select(0u, 1u, sphere), select(0u, 1u, box), select(0u, 1u, oriented), pageCount);
}
`;

interface ShapeAnswers {
    readonly words: Uint32Array;
}

async function runShapes(device: GPUDevice, pipeline: GPUComputePipeline, buffers: ReturnType<typeof uploadTraceScene>, queries: readonly QueryCase[]): Promise<ShapeAnswers> {
    const data = new Float32Array(queries.length * 20);
    queries.forEach((query, q) => {
        data.set([...query.point, query.radius, ...query.half, query.countRadius, ...query.axes[0], 0, ...query.axes[1], 0, ...query.axes[2], 0], q * 20);
    });
    const input = storageBuffer(device, 'query-oracle shapes in', data);
    const output = storageBuffer(device, 'query-oracle shapes out', queries.length * 16);
    const scene = [buffers.nodes, buffers.triangles, buffers.topLevel, ...(buffers.positions ? [buffers.positions] : [])];
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: scene.map((buffer, binding) => ({ binding, resource: { buffer } })) }));
    pass.setBindGroup(1, device.createBindGroup({ layout: pipeline.getBindGroupLayout(1), entries: [input, output].map((buffer, binding) => ({ binding, resource: { buffer } })) }));
    pass.dispatchWorkgroups(Math.ceil(queries.length / WORKGROUP));
    pass.end();
    device.queue.submit([encoder.finish()]);
    const words = new Uint32Array(await readBuffer(device, output, queries.length * 16));
    input.destroy();
    output.destroy();
    return { words };
}

async function runClosest(device: GPUDevice, kernel: ClosestPointBatchKernel, buffers: ReturnType<typeof uploadTraceScene>, points: readonly Vector3[], limits: readonly number[], mask?: number) {
    const data = new Float32Array(points.length * 4);
    points.forEach((point, q) => data.set([...point, limits[q]], q * 4));
    assert(POINT_QUERY_BYTES === 16, 'POINT_QUERY_BYTES is 16');
    const queries = storageBuffer(device, 'query-oracle points', data);
    const results = storageBuffer(device, 'query-oracle closest', points.length * CLOSEST_POINT_RECORD_BYTES);
    const encoder = device.createCommandEncoder();
    kernel.encode(encoder, buffers, { queries, results, count: points.length, mask });
    device.queue.submit([encoder.finish()]);
    const bytes = await readBuffer(device, results, points.length * CLOSEST_POINT_RECORD_BYTES);
    queries.destroy();
    results.destroy();
    return { floats: new Float32Array(bytes), words: new Uint32Array(bytes) };
}

// ---- Scenes ----------------------------------------------------------------------------------

function formatOptions(variant: Variant, geometries: readonly (MeshGeometry | null)[]) {
    const nodes = { nodeFormat: variant.nodeFormat };
    return variant.triangleLeaves === 'indexed' ? { triangleLeaves: 'indexed' as const, positionLayout: variant.positionLayout, geometries, ...nodes } : nodes;
}

/** Each instance its own copy of its mesh, two of every three static (trace-oracle's merged variants). */
function unsharedForMerge(meshes: readonly MeshGeometry[], instances: readonly TraceSceneInstance[]) {
    return { meshes: instances.map((instance) => meshes[instance.meshIndex]), instances: instances.map((instance, position) => ({ ...instance, meshIndex: position, static: position % 3 !== 2 })) };
}

async function buildScene(builder: MeshBvhBuilder, oracleCase: OracleCase, variant: Variant): Promise<TraceSceneData> {
    const build = (meshes: readonly MeshGeometry[]) => Promise.all(meshes.map((mesh) => builder.build(mesh, { triangleLeaves: variant.triangleLeaves, nodeFormat: variant.nodeFormat })));
    if (!variant.merged) return assembleTraceScene(await build(oracleCase.meshes), oracleCase.instances, {}, formatOptions(variant, oracleCase.meshes));
    const { meshes, instances } = unsharedForMerge(oracleCase.meshes, oracleCase.instances);
    const plan = mergeStaticGeometry(meshes, instances, { triangleLeaves: variant.triangleLeaves });
    const needed = meshes.flatMap((_, index) => (plan.meshNeeded[index] ? [index] : []));
    const trees = await build([...needed.map((index) => meshes[index]), ...plan.groups.map((group) => group.geometry)]);
    const sourceTrees: (MeshBvh | null)[] = meshes.map(() => null);
    needed.forEach((index, at) => (sourceTrees[index] = trees[at]));
    return assembleTraceScene(sourceTrees, instances, {}, { merged: { plan, meshes: trees.slice(needed.length) }, ...formatOptions(variant, meshes) });
}

// ---- Checks ------------------------------------------------------------------------------------

interface Tally {
    agree: number;
    ties: number;
    ambiguous: number;
    mismatches: number;
    maxError: number;
}

const tally = (): Tally => ({ agree: 0, ties: 0, ambiguous: 0, mismatches: 0, maxError: 0 });

/** What instance `instanceIndex`'s triangle `triangle` gives for point `p`: its distance, point, normal (authored front) and material. */
function triangleAnswer(instances: readonly WorldInstance[], instanceIndex: number, triangle: number, p: Vector3) {
    const answers = [];
    for (const entry of instances) {
        if (entry.instance.instanceIndex !== instanceIndex || triangle >= entry.corners.length / 9) continue;
        const c = entry.corners.subarray(triangle * 9, triangle * 9 + 9);
        const a = new BvhVector3(c[0], c[1], c[2]), b = new BvhVector3(c[3], c[4], c[5]), cc = new BvhVector3(c[6], c[7], c[8]);
        const nearest = triangleClosestPointToPoint(a, b, cc, new BvhVector3(...p), new BvhVector3());
        const e1 = [c[3] - c[0], c[4] - c[1], c[5] - c[2]], e2 = [c[6] - c[0], c[7] - c[1], c[8] - c[2]];
        const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        const length = Math.hypot(n[0], n[1], n[2]);
        const sign = entry.mirrored ? -1 : 1;
        const own = entry.mesh.triangleMaterials?.[triangle];
        const material = own !== undefined && own !== TRIANGLE_MATERIAL_NONE ? own : (entry.instance.material ?? NO_MATERIAL);
        answers.push({
            distance: Math.hypot(nearest.x - p[0], nearest.y - p[1], nearest.z - p[2]),
            corners: c,
            normal: length > 0 ? n.map((x) => (sign * x) / length) : null,
            material,
        });
    }
    return answers;
}

function checkClosest(
    failures: string[],
    where: string,
    result: Tally,
    instances: readonly WorldInstance[],
    query: QueryCase,
    reference: ReferenceClosest | null,
    expectFound: boolean | null,
    floats: Float32Array,
    words: Uint32Array,
    q: number,
): void {
    const base = (q * CLOSEST_POINT_RECORD_BYTES) / 4;
    const W = CLOSEST_POINT_RECORD_WORD;
    const distance = floats[base + W.distance];
    const found = distance >= 0;
    const fail = (message: string) => {
        result.mismatches++;
        if (failures.length < 40) failures.push(`${where} query ${q} at [${query.point.map((x) => x.toFixed(4))}]: ${message}`);
    };
    if (expectFound === null) {
        result.ambiguous++;
        return;
    }
    if (found !== expectFound) return fail(`found ${found}, the reference ${expectFound} (reference distance ${reference?.distance})`);
    if (!found) {
        if (words[base + W.material] !== NO_MATERIAL) return fail('a miss reports a material');
        result.agree++;
        return;
    }
    assert(reference, 'a found point has a reference');
    const error = Math.abs(distance - reference.distance);
    result.maxError = Math.max(result.maxError, error);
    if (error > query.tolerance) return fail(`distance ${distance}, the reference ${reference.distance} (instance ${reference.instanceIndex}, triangle ${reference.triangle}); GPU instance ${words[base + W.instance]}, triangle ${words[base + W.triangle]}`);
    // The reported triangle must be at the reported distance, with this point, normal and material.
    const instanceIndex = words[base + W.instance];
    const triangle = words[base + W.triangle];
    const point = [floats[base + W.point], floats[base + W.point + 1], floats[base + W.point + 2]];
    const [u, v] = [floats[base + W.barycentrics], floats[base + W.barycentrics + 1]];
    const normal = [floats[base + W.normal], floats[base + W.normal + 1], floats[base + W.normal + 2]];
    const material = words[base + W.material];
    const candidates = triangleAnswer(instances, instanceIndex, triangle, query.point);
    const match = candidates.find((candidate) => {
        if (Math.abs(candidate.distance - distance) > query.tolerance) return false;
        const c = candidate.corners;
        const fromBarycentrics = [0, 1, 2].map((k) => c[k] + (c[3 + k] - c[k]) * u + (c[6 + k] - c[k]) * v);
        if (Math.hypot(fromBarycentrics[0] - point[0], fromBarycentrics[1] - point[1], fromBarycentrics[2] - point[2]) > query.tolerance) return false;
        if (Math.abs(Math.hypot(point[0] - query.point[0], point[1] - query.point[1], point[2] - query.point[2]) - distance) > query.tolerance) return false;
        if (candidate.normal && candidate.normal[0] * normal[0] + candidate.normal[1] * normal[1] + candidate.normal[2] * normal[2] < NORMAL_COSINE) return false;
        return candidate.material === material;
    });
    if (!match) return fail(`instance ${instanceIndex} triangle ${triangle} at ${distance} (point [${point}], barycentrics ${u}, ${v}, normal [${normal}], material ${material}) is not that triangle's answer: ${JSON.stringify(candidates.map((c) => ({ distance: c.distance, material: c.material, normal: c.normal })))}`);
    if (instanceIndex === reference.instanceIndex && triangle === reference.triangle) result.agree++;
    else result.ties++;
}

function checkBoolean(failures: string[], where: string, result: Tally, expected: boolean | null, got: boolean, q: number, detail: () => string): void {
    if (expected === null) {
        result.ambiguous++;
        return;
    }
    if (expected === got) {
        result.agree++;
        return;
    }
    result.mismatches++;
    if (failures.length < 40) failures.push(`${where} query ${q}: GPU ${got}, the reference ${expected} (${detail()})`);
}

// ---- The standalone helpers -------------------------------------------------------------------

const HELPER_KERNEL_WGSL = /* wgsl */ `
${triangleGeometryWgsl()}

struct HelperCase {
    a: vec4f,
    b: vec4f,
    c: vec4f,
    origin: vec4f,      // w: tMin
    direction: vec4f,   // w: tMax
    point: vec4f,
}

@group(0) @binding(0) var<storage, read> helperCases: array<HelperCase>;
@group(0) @binding(1) var<storage, read_write> helperResults: array<vec4f>;

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= arrayLength(&helperCases)) {
        return;
    }
    let item = helperCases[id.x];
    let hit = bvhIntersectRayTriangle(BvhRay(item.origin.xyz, item.origin.w, item.direction.xyz, item.direction.w), item.a.xyz, item.b.xyz, item.c.xyz);
    let nearest = bvhClosestPointToTriangle(item.point.xyz, item.a.xyz, item.b.xyz, item.c.xyz);
    helperResults[id.x * 3u] = vec4f(select(0.0, 1.0, hit.hit), hit.t, hit.barycentrics);
    helperResults[id.x * 3u + 1u] = vec4f(select(0.0, 1.0, hit.frontFacing), 0.0, 0.0, 0.0);
    helperResults[id.x * 3u + 2u] = vec4f(nearest.point, 0.0);
}
`;

async function runHelpers(device: GPUDevice, count: number) {
    const random = createRandom(9001);
    const data = new Float32Array(count * 24);
    const expected: { hit: boolean | null; t: number; u: number; v: number; front: boolean; nearest: Vector3; size: number }[] = [];
    for (let i = 0; i < count; i++) {
        const corner = () => [0, 1, 2].map(() => Math.fround((random() * 2 - 1) * 5)) as Vector3;
        const a = corner(), b = corner(), c = corner();
        // A ray toward a barycentric point: inside the triangle for 2 of 3 rays, outside for the rest.
        let u = random(), v = random();
        if (i % 3 !== 2 && u + v > 1) [u, v] = [1 - u, 1 - v];
        if (i % 3 === 2) [u, v] = random() < 0.5 ? [-0.2 - random(), v] : [u + 0.6, v + 0.6];
        const target = [0, 1, 2].map((k) => a[k] + (b[k] - a[k]) * u + (c[k] - a[k]) * v);
        const origin = [0, 1, 2].map((k) => target[k] + (random() * 2 - 1) * 8) as Vector3;
        const toTarget = [0, 1, 2].map((k) => target[k] - origin[k]);
        const length = Math.hypot(...toTarget);
        const direction = toTarget.map((x) => Math.fround(x / length)) as Vector3;
        const tMax = i % 7 === 0 ? Math.fround(length * 0.5) : 1e30;
        const point = [0, 1, 2].map(() => Math.fround((random() * 2 - 1) * 8)) as Vector3;
        data.set([...a, 0, ...b, 0, ...c, 0, ...origin.map(Math.fround), 0, ...direction, tMax, ...point, 0], i * 24);
        const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        const nLength = Math.hypot(n[0], n[1], n[2]);
        const facing = Math.abs(n[0] * direction[0] + n[1] * direction[1] + n[2] * direction[2]) / nLength;
        // Near an edge or edge-on the test may go either way: ambiguous.
        const margin = Math.min(u, v, 1 - u - v);
        const inside = margin > 0;
        const clear = Math.abs(margin) > 1e-3 && facing > 1e-2;
        const hit = !clear ? null : inside && length <= tMax;
        const nearest = triangleClosestPointToPoint(new BvhVector3(...a), new BvhVector3(...b), new BvhVector3(...c), new BvhVector3(...point), new BvhVector3());
        expected.push({ hit, t: length, u, v, front: n[0] * direction[0] + n[1] * direction[1] + n[2] * direction[2] < 0, nearest: [nearest.x, nearest.y, nearest.z], size: 10 });
    }
    const input = storageBuffer(device, 'query-oracle helpers in', data);
    const output = storageBuffer(device, 'query-oracle helpers out', count * 48);
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ label: 'query-oracle helpers', code: HELPER_KERNEL_WGSL }), entryPoint: 'main' } });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [input, output].map((buffer, binding) => ({ binding, resource: { buffer } })) }));
    pass.dispatchWorkgroups(Math.ceil(count / WORKGROUP));
    pass.end();
    device.queue.submit([encoder.finish()]);
    const out = new Float32Array(await readBuffer(device, output, count * 48));
    input.destroy();
    output.destroy();
    const ray = tally();
    const nearest = tally();
    const failures: string[] = [];
    expected.forEach((e, i) => {
        const [hit, t, u, v] = out.subarray(i * 12, i * 12 + 4);
        const front = out[i * 12 + 4] !== 0;
        const got = hit !== 0;
        if (e.hit === null) ray.ambiguous++;
        else if (got !== e.hit) {
            ray.mismatches++;
            if (failures.length < 20) failures.push(`helpers ${i}: bvhIntersectRayTriangle hit ${got}, expected ${e.hit}`);
        } else if (got && (Math.abs(t - e.t) > 1e-4 * e.t || Math.abs(u - e.u) > 1e-3 || Math.abs(v - e.v) > 1e-3 || front !== e.front)) {
            ray.mismatches++;
            if (failures.length < 20) failures.push(`helpers ${i}: bvhIntersectRayTriangle t ${t} (${e.t}), barycentrics ${u}, ${v} (${e.u}, ${e.v}), front ${front} (${e.front})`);
        } else ray.agree++;
        const point = out.subarray(i * 12 + 8, i * 12 + 11);
        const error = Math.hypot(point[0] - e.nearest[0], point[1] - e.nearest[1], point[2] - e.nearest[2]);
        nearest.maxError = Math.max(nearest.maxError, error);
        if (error > 1e-5 * e.size) {
            nearest.mismatches++;
            if (failures.length < 20) failures.push(`helpers ${i}: bvhClosestPointToTriangle [${[...point]}], triangleClosestPointToPoint [${e.nearest}]`);
        } else nearest.agree++;
    });
    return { ray, nearest, failures };
}

// ---- The page ------------------------------------------------------------------------------------

await runGpuTest(async ({ device, log }) => {
    const builder = createMeshBvhBuilder({ worker: 'auto' });
    const tiny = await createTinyBvhBuilder();
    const failures: string[] = [];
    const phaseSeconds: Record<string, number> = {};
    let phaseStart = performance.now();
    const endPhase = (name: string) => {
        const now = performance.now();
        phaseSeconds[name] = Math.round((now - phaseStart) / 100) / 10;
        phaseStart = now;
    };

    // The helpers alone, binding-free.
    const helpers = await runHelpers(device, HELPER_CASES);
    failures.push(...helpers.failures);
    log(`helpers: bvhIntersectRayTriangle ${helpers.ray.agree} agree / ${helpers.ray.ambiguous} ambiguous / ${helpers.ray.mismatches} mismatches; bvhClosestPointToTriangle ${helpers.nearest.agree} / ${helpers.nearest.mismatches} (max error ${helpers.nearest.maxError.toExponential(2)})`);
    endPhase('helpers');

    // The reference answers, once per case (they do not depend on the format).
    const plainCases = buildOracleCases(SMALL_SIZES).map(withMasks);
    const references = plainCases.map((oracleCase, index) => {
        const instances = worldInstances(oracleCase, tiny);
        return { instances, ...buildQueries(instances, 1000 + index, QUERIES) };
    });
    const materialCases = plainCases.map(withTriangleMaterials);
    // With triangle materials only the expected material changes: the same world trees, other meshes.
    const materialInstances = references.map((reference, index) => reference.instances.map((entry) => ({ ...entry, mesh: materialCases[index].meshes[entry.instance.meshIndex] })));
    endPhase('references');

    const report: Record<string, unknown> = {};
    for (const variant of VARIANTS) {
        const rayMasks = variant.rayMasks ?? true;
        const formats = { triangleLeaves: variant.triangleLeaves, positionLayout: variant.positionLayout, nodeFormat: variant.nodeFormat };
        const kernel = await ClosestPointBatchKernel.create(device, { label: variant.name, rayMasks, ...formats });
        assert(kernel.nodeFormat === variant.nodeFormat && kernel.triangleLeaves === variant.triangleLeaves, 'the kernel names the formats it was made for');
        const sceneLayout = device.createBindGroupLayout({ entries: traceModuleEntries(formats).map((entry, binding) => ({ ...entry, binding })) });
        const ioLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ],
        });
        const code = `${traceModuleWgsl({ ...formats, rayMasks, queries: true, shapecast: COUNT_HOOKS_WGSL })}\n${SHAPES_KERNEL_WGSL}`;
        const shapes = await device.createComputePipelineAsync({
            label: `${variant.name} shapes`,
            layout: device.createPipelineLayout({ bindGroupLayouts: [sceneLayout, ioLayout] }),
            compute: { module: device.createShaderModule({ label: `${variant.name} shapes`, code }), entryPoint: 'main' },
        });

        const tallies = { closest: tally(), limited: tally(), masked: tally(), sphere: tally(), box: tally(), orientedBox: tally(), shapecast: tally() };
        for (const [caseIndex, plainCase] of plainCases.entries()) {
            const oracleCase = variant.triangleMaterials ? materialCases[caseIndex] : plainCase;
            const { queries } = references[caseIndex];
            const instances = variant.triangleMaterials ? materialInstances[caseIndex] : references[caseIndex].instances;
            const scene = await buildScene(builder, oracleCase, variant);
            const buffers = uploadTraceScene(device, scene, `${variant.name} ${oracleCase.name}`);
            const where = `[${variant.name}] ${oracleCase.name}`;
            try {
                const points = queries.map((query) => query.point);
                const unlimited = await runClosest(device, kernel, buffers, points, points.map(() => UNLIMITED));
                const limited = await runClosest(device, kernel, buffers, points, queries.map((query) => query.limit));
                const masked = rayMasks ? await runClosest(device, kernel, buffers, points, points.map(() => UNLIMITED), QUERY_MASK) : null;
                const answers = await runShapes(device, shapes, buffers, queries);
                queries.forEach((query, q) => {
                    checkClosest(failures, `${where} closest`, tallies.closest, instances, query, query.closest, query.closest !== null, unlimited.floats, unlimited.words, q);
                    checkClosest(failures, `${where} limited`, tallies.limited, instances, query, query.closest, query.limitExpect, limited.floats, limited.words, q);
                    if (masked) checkClosest(failures, `${where} masked`, tallies.masked, instances, query, query.closestMasked, query.closestMasked !== null, masked.floats, masked.words, q);
                    const [sphere, box, orientedBox, count] = answers.words.subarray(q * 4, q * 4 + 4);
                    checkBoolean(failures, `${where} sphere`, tallies.sphere, query.sphere, sphere !== 0, q, () => `radius ${query.radius}, closest ${query.closest?.distance}`);
                    checkBoolean(failures, `${where} box`, tallies.box, query.box, box !== 0, q, () => `half [${query.half}]`);
                    checkBoolean(failures, `${where} oriented box`, tallies.orientedBox, query.orientedBox, orientedBox !== 0, q, () => `half [${query.half}], axes ${JSON.stringify(query.axes)}`);
                    const [low, high] = query.countRange;
                    if (count >= low && count <= high) {
                        if (low === high) tallies.shapecast.agree++;
                        else tallies.shapecast.ambiguous++;
                    } else {
                        tallies.shapecast.mismatches++;
                        if (failures.length < 40) failures.push(`${where} shapecast query ${q}: counted ${count}, the reference ${low}-${high}`);
                    }
                });
            } finally {
                buffers.destroy();
            }
        }
        kernel.destroy();
        report[variant.name] = tallies;
        log(
            `[${variant.name}] ` +
                Object.entries(tallies)
                    .map(([name, t]) => `${name} ${t.agree}+${t.ties} ties/${t.ambiguous} ambiguous/${t.mismatches} mismatches`)
                    .join(', '),
        );
        endPhase(variant.name);
    }
    builder.dispose();

    // The checks must have had something to check.
    const first = report[VARIANTS[0].name] as Record<string, Tally>;
    for (const [name, t] of Object.entries(first)) assert(t.agree > 0, `${name}: no query agreed, so nothing was checked`);
    // Each shape test must have expected both answers, and the shapecast both empty and non-empty counts.
    const all = references.flatMap((reference) => reference.queries);
    const balance = {
        sphere: [all.filter((query) => query.sphere === true).length, all.filter((query) => query.sphere === false).length],
        box: [all.filter((query) => query.box === true).length, all.filter((query) => query.box === false).length],
        orientedBox: [all.filter((query) => query.orientedBox === true).length, all.filter((query) => query.orientedBox === false).length],
        shapecast: [all.filter((query) => query.countRange[0] > 0).length, all.filter((query) => query.countRange[1] === 0).length],
        limited: [all.filter((query) => query.limitExpect === true).length, all.filter((query) => query.limitExpect === false).length],
    };
    for (const [name, [yes, no]] of Object.entries(balance)) assert(yes > 0 && no > 0, `${name}: the reference must expect both answers (${yes} / ${no})`);
    assert(failures.length === 0, failures.slice(0, 12).join('\n'));
    log(`profile ${profile}: ${QUERIES} queries per case, ${HELPER_CASES} helper cases; expected true / false: ${JSON.stringify(balance)}`);
    return { profile, phaseSeconds, queriesPerCase: QUERIES, helperCases: HELPER_CASES, helpers: { ray: helpers.ray, nearest: helpers.nearest }, byVariant: report, balance };
});

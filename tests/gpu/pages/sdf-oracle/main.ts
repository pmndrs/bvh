/**
 * SDF oracle: the GPU field (`SdfGenerator`: `bvhClosestPoint` and the inside
 * test's `bvhInsideVotes` per voxel; PARITY 8.10) against the CPU reference
 * (`generateSdfCpu` over `CpuMeshBvh` trees), voxel for voxel.
 *
 * **Scenes** (instances under rotations, non-uniform scales and a mirror):
 * - `solids`: welded spheres and boxes, apart (closed: every sign is known);
 * - `overlaps`: two boxes overlapping and a hollow box (a box with an inward-wound
 *   box inside it), where parity and winding disagree (closed);
 * - `open`: a box without its top and a random triangle soup (open: the rays'
 *   votes decide, and must decide the same on both sides).
 *
 * **Reference.** Each kept instance's mesh is moved to world space (in doubles,
 * rounded once to f32, vertices shared as in the mesh) and built as its own
 * `CpuMeshBvh`; `generateSdfCpu` takes them as one scene. The GPU walks the
 * real scene in every node and triangle format (`f32`/`quantized`,
 * records/indexed/packed, merged), and with masks compiled out.
 *
 * **Checks**, per voxel:
 * - distance: |GPU| within `DISTANCE_TOLERANCE` x (scene size + |p|) of |CPU|
 *   (the f32 instance transform against the doubles one; the query oracle's
 *   rule), for every voxel;
 * - sign: equal wherever |CPU| is above `SIGN_BAND` x the scene size (a point
 *   nearer the surface than rounding can go either way). Closed scenes allow no
 *   mismatch. The open scene's rays cross a soup where two of three can split,
 *   so one ray decided differently at an edge (doubles against f32) may flip a
 *   vote: at most `OPEN_SIGN_SHARE` of its voxels may differ, reported;
 * - both rules (`parity`, `winding`), `unsigned`, a `maxDistance` band, and a mask
 *   that leaves instances out;
 * - the buffer output and the texture output of the same field: bit for bit.
 *
 * `?profile=ci` (SwiftShader, `BVH_PAGE_QUERY=profile=ci`): every variant and
 * scene on a 16^3 grid; the full profile uses 32^3. `?resolution=N` sets it.
 */

import {
    assembleTraceScene,
    createMeshBvhBuilder,
    createSdfTexture,
    createTinyBvhBuilder,
    CpuMeshBvh,
    generateSdfCpu,
    mergeStaticGeometry,
    readSdf,
    sdfVoxelCenter,
    SdfGenerator,
    uploadTraceScene,
    type MeshBvh,
    type MeshBvhBuilder,
    type MeshGeometry,
    type SdfGrid,
    type SdfOptions,
    type TinyBvhBuilder,
    type TraceSceneData,
    type TraceSceneInstance,
} from '../../../../src/index.ts';
import { createRandom, randomTriangleSoup } from '../../../../src/trace/oracle-fixtures.ts';
import { closedBox, closedSphere, inverted, merge } from '../../../support/solids.ts';
import { assert, runGpuTest } from '../../client.ts';

interface Variant {
    readonly name: string;
    readonly nodeFormat: 'f32' | 'quantized';
    readonly triangleLeaves: 'records' | 'indexed';
    readonly positionLayout?: 'vec4' | 'packed';
    readonly merged?: boolean;
    readonly rayMasks?: boolean;
}

const VARIANTS: readonly Variant[] = [
    { name: 'quantized-records', nodeFormat: 'quantized', triangleLeaves: 'records' },
    { name: 'f32-records', nodeFormat: 'f32', triangleLeaves: 'records' },
    { name: 'quantized-indexed', nodeFormat: 'quantized', triangleLeaves: 'indexed' },
    { name: 'f32-indexed-packed', nodeFormat: 'f32', triangleLeaves: 'indexed', positionLayout: 'packed' },
    { name: 'merged-quantized-records', nodeFormat: 'quantized', triangleLeaves: 'records', merged: true },
    { name: 'quantized-records-nomasks', nodeFormat: 'quantized', triangleLeaves: 'records', rayMasks: false },
];

const pageQuery = new URLSearchParams(location.search);
const profile = pageQuery.get('profile') ?? 'full';
assert(profile === 'full' || profile === 'ci', `unknown profile "${profile}" (use full or ci)`);
const RESOLUTION = Number(pageQuery.get('resolution')) || (profile === 'ci' ? 16 : 32);

/** Distances agree to this share of the scene's size plus the point's distance from the origin (the query oracle's rule). */
const DISTANCE_TOLERANCE = 4e-6;
/** Signs are compared where the reference's distance is above this share of the scene's size. */
const SIGN_BAND = 1e-5;
/** Share of the open scene's voxels whose sign may differ (see the file comment). */
const OPEN_SIGN_SHARE = 0.002;
/** The masked run's mask: instances with bit 0 (even positions). */
const QUERY_MASK = 1;

interface SdfCase {
    readonly name: string;
    readonly meshes: MeshGeometry[];
    readonly instances: TraceSceneInstance[];
    readonly closed: boolean;
    readonly grid: SdfGrid;
}

/** Column-major translation * rotation (axis-angle) * scale. */
function trs(translation: readonly number[], axis: readonly number[], angle: number, scale: readonly number[]): number[] {
    const length = Math.hypot(axis[0], axis[1], axis[2]);
    const [x, y, z] = axis.map((v) => v / length);
    const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
    const r = [
        [t * x * x + c, t * x * y + s * z, t * x * z - s * y],
        [t * x * y - s * z, t * y * y + c, t * y * z + s * x],
        [t * x * z + s * y, t * y * z - s * x, t * z * z + c],
    ];
    return [...r[0].map((v) => v * scale[0]), 0, ...r[1].map((v) => v * scale[1]), 0, ...r[2].map((v) => v * scale[2]), 0, translation[0], translation[1], translation[2], 1];
}

/** Instances get masks 1 and 2 by turn, so the masked run keeps every other one. */
const masked = (instances: TraceSceneInstance[]) => instances.map((instance, k) => ({ ...instance, mask: k % 2 === 0 ? 1 : 2 }));

function cases(): SdfCase[] {
    const sphere = closedSphere(1, 3);
    const box = merge(closedBox([-0.6, -0.4, -0.5], [0.6, 0.4, 0.5]));
    const solids: SdfCase = {
        name: 'solids',
        meshes: [merge(sphere), box],
        instances: masked([
            { meshIndex: 0, instanceIndex: 0, objectToWorld: trs([-2, 0.2, 0], [0, 1, 0], 0, [1, 1, 1]) },
            { meshIndex: 0, instanceIndex: 1, objectToWorld: trs([1.6, -0.3, 0.4], [1, 2, 0.5], 0.7, [0.8, 1.3, 0.6]) },
            { meshIndex: 1, instanceIndex: 2, objectToWorld: trs([0, 2.2, -0.3], [0.3, 0.2, 1], 1.1, [-1.2, 1, 0.9]) },
            { meshIndex: 1, instanceIndex: 3, objectToWorld: trs([0.2, -2.1, 0.5], [1, 0, 1], 2.3, [1.5, 0.7, 1]) },
        ]),
        closed: true,
        grid: { min: [-3.4, -3.2, -1.8], max: [3, 3.3, 1.9], resolution: RESOLUTION },
    };
    const outer = closedBox([-1, -1, -1], [1, 1, 1]);
    const overlaps: SdfCase = {
        name: 'overlaps',
        meshes: [merge(outer), merge(closedBox([-1.2, -1.2, -1.2], [1.2, 1.2, 1.2]), inverted(closedBox([-0.6, -0.6, -0.6], [0.6, 0.6, 0.6])))],
        instances: masked([
            { meshIndex: 0, instanceIndex: 10, objectToWorld: trs([-1.4, 0, 0], [0, 0, 1], 0.3, [1, 0.8, 0.9]) },
            { meshIndex: 0, instanceIndex: 11, objectToWorld: trs([-0.6, 0.4, 0.2], [1, 1, 0], 0.5, [0.9, 1, 0.8]) },
            { meshIndex: 1, instanceIndex: 12, objectToWorld: trs([2, 0, 0], [0, 1, 1], 0.4, [1, 1, -1]) },
        ]),
        closed: true,
        grid: { min: [-3.2, -2.2, -2.1], max: [3.8, 2.1, 2.2], resolution: RESOLUTION },
    };
    const openBox = closedBox([-1, -1, -1], [1, 1, 1]);
    const open: SdfCase = {
        name: 'open',
        meshes: [{ positions: openBox.positions, indices: openBox.indices.filter((_, i) => Math.floor(i / 6) !== 1) }, randomTriangleSoup(createRandom(99), 120, 1.2, 0.5)],
        instances: masked([
            { meshIndex: 0, instanceIndex: 20, objectToWorld: trs([-1.3, 0, 0], [1, 0, 0], 0.4, [1, 1, 1]) },
            { meshIndex: 1, instanceIndex: 21, objectToWorld: trs([1.4, 0.1, 0], [0, 1, 0], 0.9, [1, 0.8, 1.1]) },
        ]),
        closed: false,
        grid: { min: [-2.8, -1.9, -1.9], max: [3, 1.8, 2], resolution: RESOLUTION },
    };
    return [solids, overlaps, open];
}

/** The reference's trees: each kept instance's mesh in world space (doubles, then f32), shared vertices kept shared. */
function worldTrees(sdfCase: SdfCase, tiny: TinyBvhBuilder, mask?: number): CpuMeshBvh[] {
    return sdfCase.instances
        .filter((instance) => mask === undefined || ((instance.mask ?? 0xff) & mask) !== 0)
        .map((instance) => {
            const mesh = sdfCase.meshes[instance.meshIndex];
            const e = instance.objectToWorld;
            const positions = new Float32Array(mesh.positions.length);
            for (let v = 0; v < mesh.positions.length / 3; v++) {
                const x = mesh.positions[v * 3], y = mesh.positions[v * 3 + 1], z = mesh.positions[v * 3 + 2];
                for (let k = 0; k < 3; k++) positions[v * 3 + k] = e[k] * x + e[4 + k] * y + e[8 + k] * z + e[12 + k];
            }
            return CpuMeshBvh.build({ positions, indices: mesh.indices ?? undefined }, tiny);
        });
}

function formatOptions(variant: Variant, geometries: readonly (MeshGeometry | null)[]) {
    const nodes = { nodeFormat: variant.nodeFormat };
    return variant.triangleLeaves === 'indexed' ? { triangleLeaves: 'indexed' as const, positionLayout: variant.positionLayout, geometries, ...nodes } : nodes;
}

/** As the query oracle: each instance its own copy of its mesh, two of every three static and merged. */
async function buildScene(builder: MeshBvhBuilder, sdfCase: SdfCase, variant: Variant): Promise<TraceSceneData> {
    const build = (meshes: readonly MeshGeometry[]) => Promise.all(meshes.map((mesh) => builder.build(mesh, { triangleLeaves: variant.triangleLeaves, nodeFormat: variant.nodeFormat })));
    if (!variant.merged) return assembleTraceScene(await build(sdfCase.meshes), sdfCase.instances, {}, formatOptions(variant, sdfCase.meshes));
    const meshes = sdfCase.instances.map((instance) => sdfCase.meshes[instance.meshIndex]);
    const instances = sdfCase.instances.map((instance, position) => ({ ...instance, meshIndex: position, static: position % 3 !== 2 }));
    const plan = mergeStaticGeometry(meshes, instances, { triangleLeaves: variant.triangleLeaves });
    const needed = meshes.flatMap((_, index) => (plan.meshNeeded[index] ? [index] : []));
    const trees = await build([...needed.map((index) => meshes[index]), ...plan.groups.map((group) => group.geometry)]);
    const sourceTrees: (MeshBvh | null)[] = meshes.map(() => null);
    needed.forEach((index, at) => (sourceTrees[index] = trees[at]));
    return assembleTraceScene(sourceTrees, instances, {}, { merged: { plan, meshes: trees.slice(needed.length) }, ...formatOptions(variant, meshes) });
}

/** The size of a grid's box: its diagonal. */
const sizeOf = (grid: SdfGrid) => Math.hypot(grid.max[0] - grid.min[0], grid.max[1] - grid.min[1], grid.max[2] - grid.min[2]);

interface Tally {
    voxels: number;
    distanceMismatches: number;
    maxDistanceError: number;
    signChecked: number;
    signMismatches: number;
    insideVoxels: number;
    examples: string[];
}

function compare(gpu: Float32Array, cpu: Float32Array, grid: SdfGrid): Tally {
    const size = sizeOf(grid);
    const n = typeof grid.resolution === 'number' ? [grid.resolution, grid.resolution, grid.resolution] : grid.resolution;
    const tally: Tally = { voxels: cpu.length, distanceMismatches: 0, maxDistanceError: 0, signChecked: 0, signMismatches: 0, insideVoxels: 0, examples: [] };
    for (let i = 0; i < cpu.length; i++) {
        const x = i % n[0], y = Math.floor(i / n[0]) % n[1], z = Math.floor(i / (n[0] * n[1]));
        const p = sdfVoxelCenter(grid, x, y, z);
        const tolerance = DISTANCE_TOLERANCE * (size + Math.hypot(p[0], p[1], p[2]));
        const error = Math.abs(Math.abs(gpu[i]) - Math.abs(cpu[i]));
        tally.maxDistanceError = Math.max(tally.maxDistanceError, error / size);
        if (!(error <= tolerance)) {
            tally.distanceMismatches++;
            if (tally.examples.length < 4) tally.examples.push(`voxel (${x}, ${y}, ${z}): GPU ${gpu[i]}, CPU ${cpu[i]}`);
        }
        if (cpu[i] < 0) tally.insideVoxels++;
        if (Math.abs(cpu[i]) <= SIGN_BAND * size) continue;
        tally.signChecked++;
        if (Math.sign(gpu[i]) !== Math.sign(cpu[i])) {
            tally.signMismatches++;
            if (tally.examples.length < 4) tally.examples.push(`voxel (${x}, ${y}, ${z}) sign: GPU ${gpu[i]}, CPU ${cpu[i]}`);
        }
    }
    return tally;
}

interface Run {
    readonly label: string;
    readonly options: SdfOptions;
    /** Reference without the masked-out instances. */
    readonly mask?: number;
}

await runGpuTest(async ({ device, log }) => {
    const builder = createMeshBvhBuilder({ worker: 'auto' });
    const tiny = await createTinyBvhBuilder();
    const failures: string[] = [];
    const results: Record<string, unknown>[] = [];
    const references = new Map<string, Float32Array>();
    const reference = (sdfCase: SdfCase, run: Run) => {
        const key = `${sdfCase.name} ${run.label}`;
        let field = references.get(key);
        if (!field) {
            const start = performance.now();
            field = generateSdfCpu(worldTrees(sdfCase, tiny, run.mask), sdfCase.grid, run.options);
            log(`reference ${key}: ${(performance.now() - start).toFixed(0)} ms`);
            references.set(key, field);
        }
        return field;
    };

    const allCases = cases();
    for (const variant of VARIANTS) {
        const rayMasks = variant.rayMasks ?? true;
        const formats = { nodeFormat: variant.nodeFormat, triangleLeaves: variant.triangleLeaves, positionLayout: variant.positionLayout ?? 'vec4', rayMasks };
        const generator = await SdfGenerator.create(device, { label: variant.name, ...formats });
        // The first variant also writes a buffer: the same field, bit for bit.
        const bufferGenerator = variant === VARIANTS[0] ? await SdfGenerator.create(device, { label: `${variant.name} buffer`, output: 'buffer', ...formats }) : null;
        for (const sdfCase of allCases) {
            const scene = await buildScene(builder, sdfCase, variant);
            const buffers = uploadTraceScene(device, scene, `${variant.name} ${sdfCase.name}`);
            const runs: Run[] = [
                { label: 'parity', options: { sign: 'parity' } },
                { label: 'winding', options: { sign: 'winding' } },
            ];
            if (variant === VARIANTS[0]) {
                runs.push({ label: 'unsigned', options: { sign: 'unsigned' } });
                runs.push({ label: 'band 0.3', options: { maxDistance: 0.3 } });
            }
            // Masks: not on merged groups (a group has one mask), and not without masks compiled in.
            if (rayMasks && !variant.merged) runs.push({ label: 'mask 1', options: { mask: QUERY_MASK }, mask: QUERY_MASK });
            for (const run of runs) {
                const texture = createSdfTexture(device, sdfCase.grid, `${variant.name} ${sdfCase.name} ${run.label}`);
                const timing = await generator.generate(buffers, { texture }, sdfCase.grid, { ...run.options, voxelsPerSubmit: 1 << 14 });
                const gpu = await readSdf(device, { texture }, sdfCase.grid);
                texture.destroy();
                const tally = compare(gpu, reference(sdfCase, run), sdfCase.grid);
                const key = `${variant.name} ${sdfCase.name} ${run.label}`;
                const allowed = sdfCase.closed ? 0 : Math.floor(OPEN_SIGN_SHARE * tally.signChecked);
                const passed = tally.distanceMismatches === 0 && tally.signMismatches <= allowed;
                log(
                    `${passed ? 'ok' : 'FAIL'} ${key}: ${tally.voxels} voxels (${tally.insideVoxels} inside), distance mismatches ${tally.distanceMismatches} (max error ${tally.maxDistanceError.toExponential(2)} of the size), sign mismatches ${tally.signMismatches} of ${tally.signChecked} (allowed ${allowed}); ${timing.ms.toFixed(1)} ms, ${timing.submits} submits${tally.examples.length ? `; ${tally.examples.join('; ')}` : ''}`,
                );
                if (!passed) failures.push(`${key}: ${tally.distanceMismatches} distance and ${tally.signMismatches} sign mismatches (${tally.examples.join('; ')})`);
                results.push({ variant: variant.name, scene: sdfCase.name, run: run.label, ...tally, ms: timing.ms });
                if (bufferGenerator) {
                    const n = typeof sdfCase.grid.resolution === 'number' ? sdfCase.grid.resolution ** 3 : sdfCase.grid.resolution.reduce((a, b) => a * b, 1);
                    const target = device.createBuffer({ label: 'sdf buffer output', size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
                    await bufferGenerator.generate(buffers, { buffer: target }, sdfCase.grid, run.options);
                    const fromBuffer = await readSdf(device, { buffer: target }, sdfCase.grid);
                    target.destroy();
                    let differ = 0;
                    for (let i = 0; i < n; i++) if (!Object.is(fromBuffer[i], gpu[i])) differ++;
                    if (differ) failures.push(`${key}: the buffer output differs from the texture output on ${differ} voxels`);
                }
            }
            buffers.destroy();
        }
        generator.destroy();
        bufferGenerator?.destroy();
    }
    if (failures.length) throw new Error(`${failures.length} SDF checks failed: ${failures.slice(0, 6).join(' | ')}`);
    return { profile, resolution: RESOLUTION, results };
});

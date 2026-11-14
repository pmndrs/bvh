/**
 * B2: the GPU H-PLOC builder's trees, on this GPU.
 *
 * Every mesh of the B1 cases (../bvh-shared/tree-cases.ts) is built in one
 * batch, twice:
 *
 * 1. **Default** (B1's 30-bit cubic codes, SAH leaf collapse): each tree goes
 *    through the layout checker (src/layout-check.ts), and its SAH
 *    cost and node count are compared with the CPU twin's (hploc-cpu.ts), which
 *    builds the same tree but for float rounding.
 * 2. **Against tinybvh** (tinybvh's 63-bit codes and tie rule, one-triangle
 *    leaves: the tree tinybvh's `BuildHPLOC` makes): checked again, and the
 *    SAH cost must be within 1% of tinybvh's `BuildHPLOC` tree of the same
 *    mesh (the B2 done-when). One case is refused instead: on 3000 identical
 *    triangles tinybvh's rule chains the tree to depth 128 (tinybvh's own tree
 *    is that deep), more than H-PLOC's info word, and the tracer's stack, hold.
 * 3. **tinybvh's codes, the default ties**: against the CPU twin, as 1.
 *
 * 4. **Quantized nodes written by the emit** (`quantizeOnEmit`), with records
 *    and with indexed triangles: every mesh's frame slot, nodes, parents and
 *    triangles must be word for word what the `f32` emit plus the quantize
 *    pass give (`nodeFormat: 'quantized'` alone), and pass the layout check; a
 *    refused mesh in the batch gets no nodes and leaves its neighbours alone.
 *
 * A mesh with an index past its last vertex must be refused, while the other
 * meshes of its batch are not: a refused mesh is how the builder hands a mesh
 * to its fallback (mesh-builder.ts). (A NaN corner is not refused: WGSL's
 * `min` and `max` may drop the NaN, so the boxes stay finite and the tree is built.)
 *
 * Built without the parents output (`parents: false`, P9 lever 7), every tree
 * must be the same, bit for bit, and its read-back parents (found from the
 * nodes on the CPU) the ones the GPU writes.
 *
 * The ray oracle for H-PLOC trees is the trace-oracle page (variant `watertight-hploc`).
 */

import { GpuHplocBuilder, type HplocOptions } from '../../../../src/gpu-hploc.ts';
import { buildHplocOnCpu } from '../../../../src/hploc-cpu.ts';
import { checkGpuLayout, treeSahCost } from '../../../../src/layout-check.ts';
import { createTinyBvhBuilder, type TinyBvhBuilder } from '../../../../src/tinybvh.ts';
import type { MeshInput } from '../../../../src/mesh-input.ts';
import type { GpuLayoutMeshBvh } from '../../../../src/gpu-layout.ts';
import { assert, runGpuTest } from '../../client.ts';
import { randomSoup, treeCases, type TreeCase } from '../bvh-shared/tree-cases.ts';

/** The CPU twin computes the same tree but for float rounding (hploc-cpu.ts); 1% covers that. */
const TWIN_TOLERANCE = 0.01;
/** The B2 done-when: the same SAH cost as tinybvh's BuildHPLOC, within 1%. */
const TINYBVH_TOLERANCE = 0.01;

interface Variant {
    name: string;
    options: HplocOptions;
    /** The tree the GPU's must match, and what it is called. */
    reference: { name: string; tolerance: number; build(geometry: MeshInput): ReturnType<typeof buildHplocOnCpu> };
    /** Cases the GPU must refuse with these options. */
    refused?: readonly string[];
}

function variants(tinyBvh: TinyBvhBuilder): Variant[] {
    return [
        {
            name: 'default',
            options: {},
            reference: { name: 'the CPU twin', tolerance: TWIN_TOLERANCE, build: (geometry) => buildHplocOnCpu(geometry) },
        },
        {
            name: 'tinybvh codes and ties, leaves of 1',
            options: { mortonCodes: 'tinybvh-63', tieBreak: 'tinybvh', maxLeafPrimitives: 1 },
            reference: {
                name: "tinybvh's BuildHPLOC",
                tolerance: TINYBVH_TOLERANCE,
                build: (geometry) => tinyBvh.buildLayout(geometry, { builder: 'hploc', maxLeafPrimitives: 1 }).layout,
            },
            refused: ['all triangles equal'],
        },
        {
            name: 'tinybvh codes',
            options: { mortonCodes: 'tinybvh-63' },
            reference: { name: 'the CPU twin', tolerance: TWIN_TOLERANCE, build: (geometry) => buildHplocOnCpu(geometry, { mortonCodes: 'tinybvh-63' }) },
        },
    ];
}

async function checkVariant(
    builder: GpuHplocBuilder,
    variant: Variant,
    cases: readonly TreeCase[],
    log: (message: string) => void,
): Promise<{ failures: string[]; trees: Record<string, unknown> }> {
    const failures: string[] = [];
    const trees: Record<string, unknown> = {};
    const maxLeafPrimitives = variant.options.maxLeafPrimitives;
    const built = await builder.buildMany(
        cases.map((entry) => entry.input ?? entry.geometry),
        variant.options,
    );
    const layouts = await builder.readLayouts(built);
    built.destroy();
    for (const [index, layout] of layouts.entries()) {
        const { name, geometry } = cases[index];
        const where = `${variant.name}, ${name}`;
        if (variant.refused?.includes(name)) {
            if (layout) failures.push(`${where}: built, but must be refused`);
            log(`${where}: refused as expected (${built.meshes[index].problem})`);
            continue;
        }
        if (!layout) {
            failures.push(`${where}: ${built.meshes[index].problem}`);
            continue;
        }
        const problems = checkGpuLayout(layout, geometry, { maxLeafPrimitives });
        for (const problem of problems) failures.push(`${where}: ${problem}`);
        const cost = treeSahCost(layout);
        const reference = variant.reference.build(geometry);
        const referenceCost = treeSahCost(reference);
        if (Math.abs(cost - referenceCost) > variant.reference.tolerance * referenceCost) {
            failures.push(`${where}: SAH cost ${cost.toFixed(4)}, ${variant.reference.name}'s ${referenceCost.toFixed(4)}`);
        }
        trees[name] = {
            triangles: layout.triangleCount,
            nodes: layout.nodeCount,
            referenceNodes: reference.nodeCount,
            depth: layout.maxDepth,
            sahCost: Number(cost.toFixed(4)),
            sahCostVsReference: Number((cost / referenceCost).toFixed(5)),
        };
        log(
            `${where}: ${layout.triangleCount} triangles, ${layout.nodeCount} nodes (${reference.nodeCount}), depth ${layout.maxDepth} (${reference.maxDepth}), ` +
                `SAH ${cost.toFixed(3)} (${variant.reference.name} ${referenceCost.toFixed(3)}), ${problems.length === 0 ? 'valid' : 'INVALID'}`,
        );
    }
    return { failures, trees };
}

/** `parents: false`: the same nodes and triangles, no parents output, and the parents read back from the nodes equal the GPU's. */
async function checkWithoutParents(builder: GpuHplocBuilder, cases: readonly TreeCase[], log: (message: string) => void): Promise<string[]> {
    const failures: string[] = [];
    const inputs = cases.map((entry) => entry.input ?? entry.geometry);
    const withParents = await builder.buildMany(inputs);
    const without = await builder.buildMany(inputs, { parents: false });
    if (!without.withoutParents || without.parents.size > 16) failures.push(`parents: false still has a ${without.parents.size}-byte parents output`);
    const [expected, actual] = await Promise.all([builder.readLayouts(withParents), builder.readLayouts(without)]);
    const same = (a: ArrayBufferLike, b: ArrayBufferLike) => a.byteLength === b.byteLength && new Uint8Array(a).every((byte, index) => byte === new Uint8Array(b)[index]);
    let checked = 0;
    expected.forEach((layout, index) => {
        const other = actual[index];
        const where = `parents: false, ${cases[index].name}`;
        if (!layout || !other) {
            if (Boolean(layout) !== Boolean(other)) failures.push(`${where}: built in one batch only`);
            return;
        }
        checked++;
        if (!same(layout.nodes, other.nodes)) failures.push(`${where}: the nodes differ`);
        if (!same(layout.triangles, other.triangles)) failures.push(`${where}: the triangles differ`);
        if (!same(layout.parents.buffer, other.parents.buffer)) failures.push(`${where}: the parents found from the nodes are not the GPU's`);
    });
    log(`parents: false: ${checked} trees the same as with parents, outputs ${withParents.parents.size} bytes of parents against ${without.parents.size}`);
    withParents.destroy();
    without.destroy();
    return failures;
}

/** Two layouts' bytes, the first difference by buffer and word, or null when they are equal. */
function firstDifference(a: GpuLayoutMeshBvh, b: GpuLayoutMeshBvh): string | null {
    const wordsOf = (value: ArrayBufferLike | ArrayBufferView): Uint32Array =>
        ArrayBuffer.isView(value) ? new Uint32Array(value.buffer, value.byteOffset, value.byteLength / 4) : new Uint32Array(value);
    const parts: [string, ArrayBufferLike | ArrayBufferView, ArrayBufferLike | ArrayBufferView][] = [
        ['nodes', a.nodes, b.nodes],
        ['triangles', a.triangles, b.triangles],
        ['parents', a.parents, b.parents],
        ['bounds', a.bounds, b.bounds],
    ];
    for (const [name, left, right] of parts) {
        const x = wordsOf(left);
        const y = wordsOf(right);
        if (x.length !== y.length) return `${name}: ${x.length} words against ${y.length}`;
        for (let word = 0; word < x.length; word++) if (x[word] !== y[word]) return `${name} word ${word}: 0x${x[word].toString(16)} against 0x${y[word].toString(16)}`;
    }
    for (const key of ['nodeCount', 'triangleCount', 'maxDepth', 'leafCount', 'nodeFormat', 'triangleLeaves'] as const) {
        if (a[key] !== b[key]) return `${key}: ${String(a[key])} against ${String(b[key])}`;
    }
    return null;
}

/**
 * `quantizeOnEmit`: the emit's quantized nodes against the f32 emit plus the
 * quantize pass, word for word, for records and indexed triangles, and with a
 * refused mesh in the batch.
 */
async function checkQuantizeOnEmit(builder: GpuHplocBuilder, cases: readonly TreeCase[], log: (message: string) => void): Promise<{ failures: string[]; data: Record<string, unknown> }> {
    const failures: string[] = [];
    const data: Record<string, unknown> = {};
    const bad: MeshInput = { positions: randomSoup(200, 11), indices: Uint32Array.from({ length: 600 }, (_, index) => (index === 77 ? 600 : index)) };
    const geometries = [...cases.map((entry) => entry.input ?? entry.geometry), bad, cases[0].input ?? cases[0].geometry];
    const names = [...cases.map((entry) => entry.name), 'bad index', `${cases[0].name} again`];
    const sources: (MeshInput | null)[] = [...cases.map((entry) => entry.geometry), null, cases[0].geometry];
    for (const triangleLeaves of ['records', 'indexed'] as const) {
        const options: HplocOptions = { nodeFormat: 'quantized', triangleLeaves };
        const passBuilt = await builder.buildMany(geometries, options);
        const pass = await builder.readLayouts(passBuilt);
        const emittedBuilt = await builder.buildMany(geometries, { ...options, quantizeOnEmit: true });
        const emitted = await builder.readLayouts(emittedBuilt);
        // Combined with parents: false (P9 lever 7): the same trees, the parents found from the quantized nodes on the CPU.
        const bareBuilt = await builder.buildMany(geometries, { ...options, quantizeOnEmit: true, parents: false });
        const bare = await builder.readLayouts(bareBuilt);
        if (!bareBuilt.withoutParents) failures.push(`quantizeOnEmit, ${triangleLeaves}, parents: false: the batch still has a parents output`);
        bareBuilt.destroy();
        const nodeBytes = { pass: passBuilt.nodes.size, emitted: emittedBuilt.nodes.size };
        const badRange = emittedBuilt.meshes[cases.length];
        passBuilt.destroy();
        emittedBuilt.destroy();
        let equal = 0;
        for (const [index, name] of names.entries()) {
            const where = `quantizeOnEmit, ${triangleLeaves}, ${name}`;
            const source = sources[index];
            const a = pass[index];
            const b = emitted[index];
            if (!source) {
                if (b || a) failures.push(`${where}: built, but must be refused`);
                if (badRange.nodeCount !== 0 || badRange.nodeOffset !== 0) failures.push(`${where}: the refused mesh's range is ${badRange.nodeOffset} + ${badRange.nodeCount}, not 0 + 0`);
                continue;
            }
            if (!a || !b) {
                failures.push(`${where}: ${!a ? 'the pass' : 'the emit'} refused it`);
                continue;
            }
            const difference = firstDifference(a, b);
            if (difference) failures.push(`${where}: differs from the f32 emit and the quantize pass at ${difference}`);
            else equal++;
            const withoutParents = bare[index] ? firstDifference(a, bare[index]) : 'refused';
            if (withoutParents) failures.push(`${where}, parents: false: differs from the pass at ${withoutParents}`);
            for (const problem of checkGpuLayout(b, source)) failures.push(`${where}: ${problem}`);
        }
        log(`quantizeOnEmit, ${triangleLeaves}: ${equal} of ${names.length - 1} trees word for word the pass's; output nodes ${nodeBytes.emitted} bytes (the pass's packed buffer: ${nodeBytes.pass})`);
        data[triangleLeaves] = { equal, trees: names.length - 1, nodeBytes };
    }
    return { failures, data };
}

/** An index past the last vertex must get the mesh refused, and only that mesh. */
async function checkRefusal(builder: GpuHplocBuilder, log: (message: string) => void): Promise<string[]> {
    const positions = randomSoup(200, 11);
    const indices = Uint32Array.from({ length: 600 }, (_, index) => index);
    indices[77] = 600;
    const built = await builder.buildMany([{ positions: randomSoup(200, 10) }, { positions, indices }]);
    const [good, bad] = built.meshes;
    built.destroy();
    log(`mesh with a bad index: ${bad.problem ?? 'NOT refused'}; its neighbour: ${good.problem ?? 'fine'}`);
    const failures: string[] = [];
    if (!bad.problem) failures.push('a mesh with an index past its last vertex was not refused');
    if (good.problem) failures.push(`a good mesh next to a bad one was refused: ${good.problem}`);
    return failures;
}

await runGpuTest(
    async ({ device, log }) => {
        assert(GpuHplocBuilder.isSupported(device), 'this device has no subgroups; the fallback is tested by bvh-lbvh-fallback');
        const builder = new GpuHplocBuilder(device);
        await builder.prepare();
        log(`subgroups ${device.adapterInfo.subgroupMinSize}-${device.adapterInfo.subgroupMaxSize} wide: ${builder.slots} slot(s) per lane`);
        const tinyBvh = await createTinyBvhBuilder();
        const { cases, owned } = treeCases(device);
        const failures: string[] = [];
        const data: Record<string, unknown> = { slots: builder.slots, subgroupMinSize: device.adapterInfo.subgroupMinSize, subgroupMaxSize: device.adapterInfo.subgroupMaxSize };
        for (const variant of variants(tinyBvh)) {
            const result = await checkVariant(builder, variant, cases, log);
            failures.push(...result.failures);
            data[variant.name] = result.trees;
        }
        failures.push(...(await checkRefusal(builder, log)));
        failures.push(...(await checkWithoutParents(builder, cases, log)));
        const emitted = await checkQuantizeOnEmit(builder, cases, log);
        failures.push(...emitted.failures);
        data.quantizeOnEmit = emitted.data;
        for (const buffer of owned) buffer.destroy();
        assert(failures.length === 0, failures.slice(0, 15).join('\n'));
        return data;
    },
    { optionalFeatures: ['subgroups'] },
);

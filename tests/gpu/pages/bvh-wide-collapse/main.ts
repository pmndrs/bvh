/**
 * The GPU wide collapse (src/gpu-wide-collapse.ts; docs/internals/design/wide-nodes.md, section 2).
 * Binary trees built on the CPU (H-PLOC's twin, the LBVH twin, tinybvh), uploaded side by side
 * as one H-PLOC batch's buffers would be, then collapsed on the GPU, one job per format. Every
 * wide tree must pass `checkWideTree`, and a one-mesh scene of each must give the hits of the
 * `f32` scene of the same tree (the triangle test is the same; another triangle at exactly the
 * same t is a tie and counted apart). CWBVH's trees are built with leaves of at most 3, as the
 * GPU collapse expects; BVH4's with the default 8.
 */

import { buildHplocOnCpu } from '../../../../src/hploc-cpu.ts';
import { buildLbvhOnCpu } from '../../../../src/lbvh-cpu.ts';
import { GpuWideCollapse, type WideCollapseInput } from '../../../../src/gpu-wide-collapse.ts';
import { checkWideTree, convertToWide, type WideNodeFormat } from '../../../../src/wide-nodes.ts';
import { createTinyBvhBuilder } from '../../../../src/tinybvh.ts';
import { buildMeshBvhWithTinyBvh } from '../../../../src/trace/mesh-bvh-topology.ts';
import { createRandom, randomAffine, randomRays, randomTriangleSoup, tessellatedQuad } from '../../../../src/trace/oracle-fixtures.ts';
import { assembleTraceScene } from '../../../../src/trace/scene-assembly.ts';
import type { MeshBvh, MeshGeometry } from '../../../../src/trace/mesh-bvh.ts';
import { TraceBatchKernel, uploadTraceScene } from '../../../../src/trace/trace-gpu.ts';
import { assert, runGpuTest } from '../../client.ts';
import { readBuffer, traceOnGpu } from '../../scene-check.ts';

const shifted = (geometry: MeshGeometry, offset: readonly number[], scale = 1): MeshGeometry => ({
    ...geometry,
    positions: geometry.positions.map((value, index) => value * scale + offset[index % 3]),
});

function meshes(): MeshGeometry[] {
    const random = createRandom(23);
    return [
        tessellatedQuad(24, 2),
        randomTriangleSoup(random, 6000, 6, 0.4),
        shifted(randomTriangleSoup(random, 800, 2, 0.3), [1.0e5, -3.0e4, 7.5]),
        shifted(randomTriangleSoup(random, 400, 1, 0.2), [0, 0, 0], 1e-5),
        { positions: tessellatedQuad(7, 1).positions.map((value, index) => (index % 3 === 2 ? 3 : value)), indices: tessellatedQuad(7, 1).indices },
        { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]) },
        { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 2, 2, 3, 2, 2, 2, 3, 2]) },
    ];
}

/** Uploads the trees' nodes and triangles one after another, as one batch's buffers. */
function uploadBatch(device: GPUDevice, trees: readonly MeshBvh[]): { nodes: GPUBuffer; triangles: GPUBuffer; inputs: WideCollapseInput[] } {
    const nodeBytes = trees.reduce((sum, tree) => sum + tree.nodeCount * 64, 0);
    const triangleBytes = trees.reduce((sum, tree) => sum + tree.triangleCount * 48, 0);
    const nodes = device.createBuffer({ label: 'wide collapse source nodes', size: nodeBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const triangles = device.createBuffer({ label: 'wide collapse source triangles', size: triangleBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const inputs: WideCollapseInput[] = [];
    let node = 0;
    let triangle = 0;
    for (const tree of trees) {
        device.queue.writeBuffer(nodes, node * 64, tree.nodes, 0, tree.nodeCount * 64);
        device.queue.writeBuffer(triangles, triangle * 48, tree.triangles, 0, tree.triangleCount * 48);
        inputs.push({ srcNode: node, nodeCount: tree.nodeCount, srcTriangle: triangle, triangleCount: tree.triangleCount, maxDepth: tree.maxDepth });
        node += tree.nodeCount;
        triangle += tree.triangleCount;
    }
    return { nodes, triangles, inputs };
}

await runGpuTest(async ({ device, log }) => {
    const tinyBvh = await createTinyBvhBuilder();
    const failures: string[] = [];
    const results: Record<string, unknown> = {};
    const kernels = { f32: await TraceBatchKernel.create(device), bvh4: await TraceBatchKernel.create(device, { nodeFormat: 'bvh4' }), cwbvh8: await TraceBatchKernel.create(device, { nodeFormat: 'cwbvh8' }) };
    for (const format of ['bvh4', 'cwbvh8'] as const satisfies readonly WideNodeFormat[]) {
        const maxLeafPrimitives = format === 'cwbvh8' ? 3 : 8;
        const builders: [string, (geometry: MeshGeometry) => MeshBvh][] = [
            ['hploc', (geometry) => buildHplocOnCpu(geometry, { maxLeafPrimitives }) as MeshBvh],
            ['lbvh', (geometry) => buildLbvhOnCpu(geometry, { maxLeafPrimitives }) as MeshBvh],
            ['tinybvh', (geometry) => buildMeshBvhWithTinyBvh(geometry, tinyBvh, { maxLeafPrimitives })],
        ];
        const geometries = meshes();
        const trees: { name: string; tree: MeshBvh }[] = [];
        for (const [builder, build] of builders) geometries.forEach((geometry, index) => trees.push({ name: `${builder} mesh ${index}`, tree: build(geometry) }));
        const collapse = await GpuWideCollapse.create(device, { format });
        const batch = uploadBatch(device, trees.map((entry) => entry.tree));
        const encoder = device.createCommandEncoder({ label: `wide collapse ${format}` });
        const started = performance.now();
        const run = collapse.encode(encoder, batch.nodes, batch.triangles, batch.inputs);
        device.queue.submit([encoder.finish()]);
        const counts = await run.counts();
        const gpuMs = performance.now() - started;
        const output = await readBuffer(device, run.output);
        const wideBytes = format === 'cwbvh8' ? 80 : 64;
        let gpuNodes = 0;
        let cpuNodes = 0;
        let ties = 0;
        let compared = 0;
        for (const [index, { name, tree }] of trees.entries()) {
            const count = counts[index];
            const nodes = output.slice(count.dstNode * wideBytes, (count.dstNode + count.nodeCount) * wideBytes);
            const records = format === 'cwbvh8' ? output.slice(run.nodeRegionBytes + count.dstTriangle * 48, run.nodeRegionBytes + (count.dstTriangle + count.triangleCount) * 48) : tree.triangles.slice(0, tree.triangleCount * 48);
            const problems = checkWideTree(format, nodes, count.nodeCount, records, tree.triangleCount);
            for (const problem of problems) failures.push(`${format} ${name}: ${problem}`);
            gpuNodes += count.nodeCount;
            cpuNodes += convertToWide(tree, format).nodeCount;
            if (problems.length > 0) continue;
            // A one-mesh scene: its nodes and triangles at offset 0, so the GPU tree replaces the CPU one in place.
            const instance = [{ meshIndex: 0, instanceIndex: 7, objectToWorld: randomAffine(createRandom(index + 1), { extent: 3, mirror: index % 2 === 1 }) }];
            const exact = assembleTraceScene([tree], instance);
            const cpuWide = assembleTraceScene([tree], instance, {}, { nodeFormat: format });
            const gpuWide = { ...cpuWide, nodes, triangles: records };
            const rays = randomRays(createRandom(100 + index), 3000, [-8, -8, -8, 8, 8, 8], 1);
            const exactBuffers = uploadTraceScene(device, exact, 'f32');
            const wideBuffers = uploadTraceScene(device, gpuWide, format);
            const a = await traceOnGpu(device, kernels.f32, exactBuffers, rays);
            const b = await traceOnGpu(device, kernels[format], wideBuffers, rays);
            exactBuffers.destroy();
            wideBuffers.destroy();
            for (let ray = 0; ray < rays.length; ray++) {
                compared++;
                const x = a.closest(ray);
                const y = b.closest(ray);
                if (a.any(ray) !== b.any(ray)) failures.push(`${format} ${name} ray ${ray}: any-hit differs`);
                if ((x === null) !== (y === null) || (x && y && x.t !== y.t)) failures.push(`${format} ${name} ray ${ray}: ${x?.t ?? 'miss'} against ${y?.t ?? 'miss'}`);
                else if (x && y && x.triangleIndex !== y.triangleIndex) ties++;
            }
            // CWBVH needs one entry per level; a BVH4 root with only leaf children pushes nothing.
            const need = count.stackNeed;
            if (need < (format === 'cwbvh8' ? 1 : 0) || need > (format === 'cwbvh8' ? 32 : 64)) failures.push(`${format} ${name}: stack need ${need}`);
        }
        log(`${format}: ${trees.length} trees in one job, ${gpuMs.toFixed(1)} ms with the readback; GPU nodes ${gpuNodes}, CPU collapse ${cpuNodes}; ${compared} rays compared, ${ties} ties`);
        results[format] = { trees: trees.length, gpuNodes, cpuNodes, gpuMs, rays: compared, ties };
        run.destroy();
        batch.nodes.destroy();
        batch.triangles.destroy();
    }
    for (const kernel of Object.values(kernels)) kernel.destroy();
    assert(failures.length === 0, failures.slice(0, 10).join('\n'));
    return results;
});

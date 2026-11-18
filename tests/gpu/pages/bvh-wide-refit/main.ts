/**
 * The GPU wide refit (docs/internals/design/wide-nodes.md, section 2): `GpuMeshRefit` with a
 * wide target refits the binary tree's `f32` copy from posed vertices, then packs the wide
 * nodes through the tree's refit map (gpu-wide-refit.ts).
 *
 * Each mesh, each format, two trees: tinybvh's collapsed on the CPU (its map from
 * `wideRefitMap`) and H-PLOC's CPU twin collapsed on the GPU with `refitMap: true` (its map
 * from the collapse; CWBVH's built with leaves of at most 3). Each is placed after a few
 * padding nodes and records, then posed and refit on the GPU. Then:
 *
 * - the wide nodes equal, word for word, what the CPU writes for the same pose
 *   (`refitWide` for the CPU collapse, `repackWide` over the collapse's own nodes for the GPU
 *   one), and the padding around them is untouched;
 * - the records equal the posed records in the tree's order (CWBVH: the map's);
 * - the nodes pass `checkWideTree` on the posed records, and a scene of the refit tree gives
 *   the hits of the posed `f32` tree (ties at the same t apart);
 * - the CPU pose's pack (`encodePackFrom`: CPU-refit binary nodes, packed on the GPU) gives
 *   the same nodes from the unposed ones.
 */

import { buildHplocOnCpu } from '../../../../src/hploc-cpu.ts';
import { GpuMeshRefit } from '../../../../src/gpu-mesh-refit.ts';
import { GpuWideCollapse } from '../../../../src/gpu-wide-collapse.ts';
import { refitLayoutToPose } from '../../../../src/refit-cpu.ts';
import { checkWideTree, convertToWide, refitWide, reorderTriangles, repackWide, wideNodeBytes, wideRefitMap, type WideNodeFormat, type WideRefitMap } from '../../../../src/wide-nodes.ts';
import { createTinyBvhBuilder } from '../../../../src/tinybvh.ts';
import { buildMeshBvhWithTinyBvh } from '../../../../src/trace/mesh-bvh-topology.ts';
import { createRandom, randomAffine, randomRays, randomTriangleSoup, tessellatedQuad } from '../../../../src/trace/oracle-fixtures.ts';
import { assembleTraceScene } from '../../../../src/trace/scene-assembly.ts';
import type { MeshBvh, MeshGeometry } from '../../../../src/trace/mesh-bvh.ts';
import { TraceBatchKernel, uploadTraceScene } from '../../../../src/trace/trace-gpu.ts';
import { assert, runGpuTest } from '../../client.ts';
import { readBuffer, traceOnGpu } from '../../scene-check.ts';

const PAD_NODES = 2;
const PAD_TRIANGLES = 3;
const PAD_WORD = 0xa5a5a5a5;

const shifted = (geometry: MeshGeometry, offset: readonly number[], scale = 1): MeshGeometry => ({
    ...geometry,
    positions: geometry.positions.map((value, index) => value * scale + offset[index % 3]),
});

function meshes(): MeshGeometry[] {
    const random = createRandom(31);
    return [
        tessellatedQuad(40, 2),
        randomTriangleSoup(random, 6000, 6, 0.4),
        shifted(randomTriangleSoup(random, 800, 2, 0.3), [1.0e5, -3.0e4, 7.5]),
        shifted(randomTriangleSoup(random, 400, 1, 0.2), [0, 0, 0], 1e-5),
        { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]) },
        { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 2, 2, 3, 2, 2, 2, 3, 2]) },
    ];
}

/** A pose that moves every vertex: a wave across the mesh's extent, and a stretch. */
function posed(positions: Float32Array): Float32Array {
    let lo = Infinity;
    let hi = -Infinity;
    for (const value of positions) {
        lo = Math.min(lo, value);
        hi = Math.max(hi, value);
    }
    const extent = Math.max(hi - lo, 1e-30);
    const moved = new Float32Array(positions.length);
    for (let at = 0; at < positions.length; at += 3) {
        const x = (positions[at] - lo) / extent;
        const z = (positions[at + 2] - lo) / extent;
        moved[at] = positions[at] * 1.2;
        moved[at + 1] = positions[at + 1] + Math.sin(x * 9 + z * 4) * extent * 0.15;
        moved[at + 2] = positions[at + 2] + Math.cos(x * 5) * extent * 0.05;
    }
    return moved;
}

const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

function bufferWith(device: GPUDevice, label: string, data: ArrayBuffer, padBytes: number): GPUBuffer {
    const size = Math.ceil((padBytes * 2 + data.byteLength) / 4) * 4;
    const buffer = device.createBuffer({ label, size, usage: storage });
    const filled = new Uint32Array(size / 4).fill(PAD_WORD);
    new Uint8Array(filled.buffer).set(new Uint8Array(data), padBytes);
    device.queue.writeBuffer(buffer, 0, filled);
    return buffer;
}

/** Words that differ, and the first few of them. */
function differences(actual: ArrayBuffer, expected: ArrayBuffer, offsetBytes: number, label: string, out: string[]): number {
    const a = new Uint32Array(actual, offsetBytes, expected.byteLength / 4);
    const b = new Uint32Array(expected);
    let count = 0;
    for (let word = 0; word < b.length; word++) {
        if (a[word] === b[word]) continue;
        if (count++ < 3) out.push(`${label}: word ${word} is 0x${a[word].toString(16)}, the CPU's 0x${b[word].toString(16)}`);
    }
    return count;
}

/** One mesh's binary tree collapsed on the GPU: its wide nodes and the collapse's refit map. */
async function collapseOnGpu(device: GPUDevice, collapse: GpuWideCollapse, layout: MeshBvh, refitMap: boolean): Promise<{ nodes: ArrayBuffer; nodeCount: number; map: WideRefitMap | undefined }> {
    const wideBytes = wideNodeBytes(collapse.format);
    const sourceNodes = device.createBuffer({ size: layout.nodeCount * 64, usage: storage });
    const sourceTriangles = device.createBuffer({ size: layout.triangleCount * 48, usage: storage });
    device.queue.writeBuffer(sourceNodes, 0, layout.nodes, 0, layout.nodeCount * 64);
    device.queue.writeBuffer(sourceTriangles, 0, layout.triangles, 0, layout.triangleCount * 48);
    const encoder = device.createCommandEncoder({ label: 'collapse' });
    const run = collapse.encode(encoder, sourceNodes, sourceTriangles, [{ srcNode: 0, nodeCount: layout.nodeCount, srcTriangle: 0, triangleCount: layout.triangleCount, maxDepth: layout.maxDepth, refitMap }]);
    device.queue.submit([encoder.finish()]);
    const [count] = await run.counts();
    const output = await readBuffer(device, run.output);
    run.destroy();
    sourceNodes.destroy();
    sourceTriangles.destroy();
    return { nodes: output.slice(count.dstNode * wideBytes, (count.dstNode + count.nodeCount) * wideBytes), nodeCount: count.nodeCount, map: count.refitMap };
}

await runGpuTest(
    async ({ device, log }) => {
        assert(GpuMeshRefit.isSupported(device), 'the mesh refit needs subgroups');
        const tinyBvh = await createTinyBvhBuilder();
        const failures: string[] = [];
        const results: Record<string, unknown> = {};
        const kernels = { f32: await TraceBatchKernel.create(device), bvh4: await TraceBatchKernel.create(device, { nodeFormat: 'bvh4' }), cwbvh8: await TraceBatchKernel.create(device, { nodeFormat: 'cwbvh8' }) };
        for (const format of ['bvh4', 'cwbvh8'] as const satisfies readonly WideNodeFormat[]) {
            const collapse = await GpuWideCollapse.create(device, { format });
            const wideBytes = wideNodeBytes(format);
            let trees = 0;
            let wideNodes = 0;
            let differing = 0;
            let compared = 0;
            let ties = 0;
            for (const [meshIndex, geometry] of meshes().entries()) {
                for (const source of ['cpu', 'gpu'] as const) {
                    const name = `${format} ${source} mesh ${meshIndex}`;
                    // The binary tree, its wide tree's nodes and the refit map, and the CPU's nodes for a pose.
                    let layout: MeshBvh;
                    let nodes: ArrayBuffer;
                    let nodeCount: number;
                    let map: WideRefitMap;
                    let expectedNodes: (posedNodes: ArrayBuffer) => ArrayBuffer;
                    if (source === 'cpu') {
                        const cpuLayout = buildMeshBvhWithTinyBvh(geometry, tinyBvh);
                        const tree = convertToWide(cpuLayout, format);
                        layout = cpuLayout;
                        nodes = tree.nodes;
                        nodeCount = tree.nodeCount;
                        map = wideRefitMap(tree, cpuLayout);
                        expectedNodes = (posedNodes) => refitWide(tree, { ...cpuLayout, nodes: posedNodes });
                    } else {
                        layout = buildHplocOnCpu(geometry, { maxLeafPrimitives: format === 'cwbvh8' ? 3 : 8 }) as MeshBvh;
                        const collapsed = await collapseOnGpu(device, collapse, layout, true);
                        if (!collapsed.map) {
                            failures.push(`${name}: the collapse returned no refit map`);
                            continue;
                        }
                        const gpuMap = collapsed.map;
                        nodes = collapsed.nodes;
                        nodeCount = collapsed.nodeCount;
                        map = gpuMap;
                        expectedNodes = (posedNodes) => repackWide(collapsed.nodes, gpuMap, posedNodes);
                    }
                    trees++;
                    wideNodes += nodeCount;
                    const rest = layout.triangles.slice(0, layout.triangleCount * 48);
                    const pose = posed(geometry.positions);
                    const posedLayout = refitLayoutToPose(layout, pose, geometry.indices);
                    const wantNodes = expectedNodes(posedLayout.nodes);
                    const wantRecords = reorderTriangles(posedLayout.triangles!, 48, map.triangleOrder);

                    // Placed after padding, in buffers of their own; the vertices and indices on the GPU.
                    const nodeBuffer = bufferWith(device, `${name} nodes`, nodes, PAD_NODES * wideBytes);
                    const triangleBuffer = bufferWith(device, `${name} triangles`, reorderTriangles(rest, 48, map.triangleOrder), PAD_TRIANGLES * 48);
                    const vertexBuffer = device.createBuffer({ size: pose.byteLength, usage: storage });
                    device.queue.writeBuffer(vertexBuffer, 0, pose.buffer as ArrayBuffer);
                    const indices = geometry.indices ? Uint32Array.from(geometry.indices) : null;
                    const indexBuffer = indices ? device.createBuffer({ size: indices.byteLength, usage: storage }) : null;
                    if (indices && indexBuffer) device.queue.writeBuffer(indexBuffer, 0, indices);
                    const refit = new GpuMeshRefit(device, { label: name, wideFormat: format });
                    await refit.prepare();
                    refit.setTree(layout, { nodeFormat: format, wide: map, nodeWordOffset: (PAD_NODES * wideBytes) / 4, triangleWordOffset: PAD_TRIANGLES * 12 });
                    const encoder = device.createCommandEncoder({ label: `${name} refit` });
                    const pass = encoder.beginComputePass();
                    refit.encode(pass, { nodes: nodeBuffer, triangles: triangleBuffer }, { positionBuffer: vertexBuffer, vertexCount: pose.length / 3, ...(indexBuffer ? { indexBuffer, indexFormat: 'uint32' as const } : {}) });
                    pass.end();
                    device.queue.submit([encoder.finish()]);
                    const gotNodes = await readBuffer(device, nodeBuffer);
                    const gotTriangles = await readBuffer(device, triangleBuffer);
                    differing += differences(gotNodes, wantNodes, PAD_NODES * wideBytes, `${name} nodes`, failures);
                    differing += differences(gotTriangles, wantRecords, PAD_TRIANGLES * 48, `${name} records`, failures);
                    const after = PAD_NODES * wideBytes + nodes.byteLength;
                    const padding = [...new Uint32Array(gotNodes, 0, (PAD_NODES * wideBytes) / 4), ...new Uint32Array(gotNodes, after, (PAD_NODES * wideBytes) / 4)];
                    if (padding.some((word) => word !== PAD_WORD)) failures.push(`${name}: the refit wrote outside the tree's nodes`);
                    const refitNodes = gotNodes.slice(PAD_NODES * wideBytes, after);
                    for (const problem of checkWideTree(format, refitNodes, nodeCount, wantRecords, layout.triangleCount)) failures.push(`${name}: ${problem}`);

                    // The CPU pose's pack: the unposed nodes again, packed from the CPU-refit binary nodes.
                    device.queue.writeBuffer(nodeBuffer, PAD_NODES * wideBytes, nodes);
                    const cpuPose = device.createCommandEncoder({ label: `${name} cpu pose pack` });
                    const packPass = cpuPose.beginComputePass();
                    refit.encodePackFrom(packPass, { nodes: nodeBuffer }, posedLayout.nodes);
                    packPass.end();
                    device.queue.submit([cpuPose.finish()]);
                    differing += differences(await readBuffer(device, nodeBuffer), wantNodes, PAD_NODES * wideBytes, `${name} cpu pose pack`, failures);

                    // Hits: the refit wide tree against the posed f32 tree, in a one-mesh scene.
                    const posedTree: MeshBvh = { ...layout, nodes: posedLayout.nodes, triangles: posedLayout.triangles! };
                    const instance = [{ meshIndex: 0, instanceIndex: 3, objectToWorld: randomAffine(createRandom(meshIndex + 5), { extent: 3, mirror: meshIndex % 2 === 0 }) }];
                    const exact = assembleTraceScene([posedTree], instance);
                    const wideScene = { ...assembleTraceScene([posedTree], instance, {}, { nodeFormat: format }), nodes: refitNodes, triangles: wantRecords };
                    const rays = randomRays(createRandom(200 + meshIndex), 3000, [-8, -8, -8, 8, 8, 8], 1);
                    const exactBuffers = uploadTraceScene(device, exact, 'f32');
                    const wideBuffers = uploadTraceScene(device, wideScene, format);
                    const a = await traceOnGpu(device, kernels.f32, exactBuffers, rays);
                    const b = await traceOnGpu(device, kernels[format], wideBuffers, rays);
                    exactBuffers.destroy();
                    wideBuffers.destroy();
                    for (let ray = 0; ray < rays.length; ray++) {
                        compared++;
                        const x = a.closest(ray);
                        const y = b.closest(ray);
                        if (a.any(ray) !== b.any(ray)) failures.push(`${name} ray ${ray}: any-hit differs`);
                        if ((x === null) !== (y === null) || (x && y && x.t !== y.t)) failures.push(`${name} ray ${ray}: ${x?.t ?? 'miss'} against ${y?.t ?? 'miss'}`);
                        else if (x && y && x.triangleIndex !== y.triangleIndex) ties++;
                    }
                    refit.dispose();
                    for (const buffer of [nodeBuffer, triangleBuffer, vertexBuffer, indexBuffer]) buffer?.destroy();
                }
            }
            log(`${format}: ${trees} trees refit (${wideNodes} wide nodes), ${differing} words differing from the CPU's; ${compared} rays compared, ${ties} ties`);
            results[format] = { trees, wideNodes, differing, rays: compared, ties };
        }
        for (const kernel of Object.values(kernels)) kernel.destroy();
        assert(failures.length === 0, failures.slice(0, 10).join('\n'));
        return results;
    },
    { requiredFeatures: ['subgroups'] },
);

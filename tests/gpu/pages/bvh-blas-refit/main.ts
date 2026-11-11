/**
 * B1: refit of a mesh's tree (BLAS) from moved vertices on the GPU.
 *
 * Two meshes (a wavy grid with 16-bit indices, a triangle-list soup), each
 * with a tree from the GPU LBVH and one from tinybvh. Each tree is placed
 * inside larger shared buffers at an offset, as a scene would hold it, with
 * marker words around it. Over five frames the vertices deform (waves, a
 * twist, a big move and scale); every frame the new vertices go up as an
 * interleaved buffer, the refit runs, and the tree is read back:
 *
 * - its triangles and boxes must be the deformed mesh's exactly
 *   (src/layout-check.ts: corners bit for bit, every box the
 *   exact box of what is below it), with the shape unchanged;
 * - the marker words around it must be untouched;
 * - the refit's stats must show no gave-up reads.
 */

import { GpuLbvhBuilder } from '../../../../src/gpu-lbvh.ts';
import { GPU_LAYOUT_NODE_BYTES, GPU_LAYOUT_TRIANGLE_BYTES, type GpuLayoutMeshBvh } from '../../../../src/gpu-layout.ts';
import { GpuMeshRefit } from '../../../../src/gpu-mesh-refit.ts';
import { checkGpuLayout } from '../../../../src/layout-check.ts';
import { createTinyBvhBuilder } from '../../../../src/tinybvh.ts';
import type { MeshInput } from '../../../../src/mesh-input.ts';
import { assert, runGpuTest } from '../../client.ts';

/** Nodes and triangles in front of the mesh in the shared buffers, and words of marker after. */
const NODES_BEFORE = 7;
const TRIANGLES_BEFORE = 5;
const MARKER = 0xdeadbeef;
const MARKER_TAIL_WORDS = 64;

/** Vertices go up interleaved: 2 unused floats, x, y, z, 1 unused float. */
const VERTEX_STRIDE = 6;
const VERTEX_OFFSET = 2;

function grid(size: number): MeshInput {
    const positions = new Float32Array((size + 1) * (size + 1) * 3);
    for (let z = 0; z <= size; z++) {
        for (let x = 0; x <= size; x++) positions.set([x * 0.1 - 5, 0, z * 0.1 - 5], (z * (size + 1) + x) * 3);
    }
    const indices = new Uint16Array(size * size * 6);
    for (let z = 0; z < size; z++) {
        for (let x = 0; x < size; x++) {
            const corner = z * (size + 1) + x;
            indices.set([corner, corner + 1, corner + size + 1, corner + 1, corner + size + 2, corner + size + 1], (z * size + x) * 6);
        }
    }
    return { positions, indices };
}

function soup(count: number): MeshInput {
    let state = 12345;
    const random = () => ((state = (state * 1664525 + 1013904223) >>> 0) / 2 ** 32);
    const positions = new Float32Array(count * 9);
    for (let triangle = 0; triangle < count; triangle++) {
        const center = [random() * 10 - 5, random() * 10 - 5, random() * 10 - 5];
        for (let corner = 0; corner < 3; corner++) {
            for (let axis = 0; axis < 3; axis++) positions[triangle * 9 + corner * 3 + axis] = center[axis] + (random() - 0.5) * 0.4;
        }
    }
    return { positions };
}

/** The mesh at frame `frame`: waves, then a twist, then a big move and scale. */
function deform(rest: Float32Array, frame: number): Float32Array {
    const moved = new Float32Array(rest.length);
    for (let vertex = 0; vertex < rest.length / 3; vertex++) {
        let x = rest[vertex * 3];
        let y = rest[vertex * 3 + 1];
        let z = rest[vertex * 3 + 2];
        y += Math.sin(x * 1.3 + frame) * Math.cos(z * 0.7 - frame) * 0.8 * frame;
        if (frame >= 3) {
            const angle = y * 0.4;
            [x, z] = [x * Math.cos(angle) - z * Math.sin(angle), x * Math.sin(angle) + z * Math.cos(angle)];
        }
        if (frame >= 4) {
            x = x * 3 + 1000;
            z = z * 0.5 - 250;
        }
        moved.set([x, y, z], vertex * 3);
    }
    return moved;
}

function interleave(positions: Float32Array): Float32Array {
    const vertexCount = positions.length / 3;
    const out = new Float32Array(vertexCount * VERTEX_STRIDE).fill(Number.NaN);
    for (let vertex = 0; vertex < vertexCount; vertex++) out.set(positions.subarray(vertex * 3, vertex * 3 + 3), vertex * VERTEX_STRIDE + VERTEX_OFFSET);
    return out;
}

function storage(device: GPUDevice, label: string, data: ArrayBufferView): GPUBuffer {
    const buffer = device.createBuffer({ label, size: Math.max(16, Math.ceil(data.byteLength / 4) * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, Math.ceil(data.byteLength / 4) * 4);
    return buffer;
}

async function read(device: GPUDevice, buffer: GPUBuffer): Promise<ArrayBuffer> {
    const readback = device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, readback, 0, buffer.size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0);
    readback.destroy();
    return copy;
}

/** A buffer of marker words with `payload` copied in at `offsetBytes`. */
function placed(payload: ArrayBuffer, offsetBytes: number): Uint32Array {
    const words = new Uint32Array((offsetBytes + payload.byteLength) / 4 + MARKER_TAIL_WORDS).fill(MARKER);
    new Uint8Array(words.buffer).set(new Uint8Array(payload), offsetBytes);
    return words;
}

function markersIntact(words: Uint32Array, offsetBytes: number, payloadBytes: number): boolean {
    for (let at = 0; at < offsetBytes / 4; at++) if (words[at] !== MARKER) return false;
    for (let at = (offsetBytes + payloadBytes) / 4; at < words.length; at++) if (words[at] !== MARKER) return false;
    return true;
}

async function refitAcrossFrames(device: GPUDevice, name: string, geometry: MeshInput, layout: GpuLayoutMeshBvh, log: (message: string) => void): Promise<string[]> {
    const failures: string[] = [];
    const nodeOffsetBytes = NODES_BEFORE * GPU_LAYOUT_NODE_BYTES;
    const triangleOffsetBytes = TRIANGLES_BEFORE * GPU_LAYOUT_TRIANGLE_BYTES;
    const nodes = storage(device, `${name} nodes`, placed(layout.nodes, nodeOffsetBytes));
    const triangles = storage(device, `${name} triangles`, placed(layout.triangles, triangleOffsetBytes));
    let indexBuffer: GPUBuffer | null = null;
    if (geometry.indices) {
        const padded = new Uint16Array(geometry.indices.length + (geometry.indices.length % 2));
        padded.set(geometry.indices);
        indexBuffer = storage(device, `${name} indices`, padded);
    }
    const refit = new GpuMeshRefit(device, { label: `${name} refit`, diagnostics: true });
    await refit.prepare();
    refit.setTree(layout, { nodeWordOffset: nodeOffsetBytes / 4, triangleWordOffset: triangleOffsetBytes / 4 });
    let vertexBuffer: GPUBuffer | null = null;
    try {
        for (let frame = 1; frame <= 5; frame++) {
            const moved = deform(geometry.positions, frame);
            vertexBuffer?.destroy();
            vertexBuffer = storage(device, `${name} vertices`, interleave(moved));
            refit.pollStats();
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            refit.encode(pass, { nodes, triangles }, {
                positionBuffer: vertexBuffer,
                vertexCount: moved.length / 3,
                positionStrideFloats: VERTEX_STRIDE,
                positionOffsetFloats: VERTEX_OFFSET,
                indexBuffer,
                indexFormat: 'uint16',
            });
            pass.end();
            refit.copyStats(encoder);
            device.queue.submit([encoder.finish()]);

            const nodeWords = new Uint32Array(await read(device, nodes));
            const triangleWords = new Uint32Array(await read(device, triangles));
            if (!markersIntact(nodeWords, nodeOffsetBytes, layout.nodes.byteLength)) failures.push(`${name} frame ${frame}: the refit wrote outside the mesh's nodes`);
            if (!markersIntact(triangleWords, triangleOffsetBytes, layout.triangles.byteLength)) failures.push(`${name} frame ${frame}: the refit wrote outside the mesh's triangles`);
            const refitNodes = nodeWords.buffer.slice(nodeOffsetBytes, nodeOffsetBytes + layout.nodes.byteLength);
            const refitTriangles = triangleWords.buffer.slice(triangleOffsetBytes, triangleOffsetBytes + layout.triangles.byteLength);
            // The checker compares the mesh box with the root's; take it from the refit root, the boxes below are checked exactly.
            // A one-triangle tree's right child is empty and has no box.
            const floats = new Float32Array(refitNodes, 0, 16);
            const hasRight = new Uint32Array(refitNodes, 0, 16)[7] !== 0xffffffff;
            const right = hasRight ? 8 : 0;
            const bounds = new Float32Array([0, 1, 2].map((axis) => Math.min(floats[axis], floats[right + axis])).concat([0, 1, 2].map((axis) => Math.max(floats[4 + axis], floats[right + 4 + axis]))));
            const problems = checkGpuLayout({ ...layout, nodes: refitNodes, triangles: refitTriangles, bounds }, { positions: moved, indices: geometry.indices });
            for (const problem of problems.slice(0, 5)) failures.push(`${name} frame ${frame}: ${problem}`);
            log(`${name} frame ${frame}: ${problems.length === 0 ? 'refit matches the deformed mesh exactly' : `${problems.length} problems`}`);
        }
        // The stats of the last copied frame arrive a little later.
        for (let tries = 0; tries < 50 && !refit.latestStats; tries++) {
            await new Promise((resolve) => setTimeout(resolve, 20));
            refit.pollStats();
        }
        const stats = refit.latestStats;
        if (!stats) failures.push(`${name}: no refit stats arrived`);
        else if (stats.incomplete > 0 || stats.repairs > 0) failures.push(`${name}: the refit gave up ${stats.incomplete} reads (repairs ${stats.repairs})`);
    } finally {
        vertexBuffer?.destroy();
        nodes.destroy();
        triangles.destroy();
        indexBuffer?.destroy();
        refit.dispose();
    }
    return failures;
}

/** Vertices in front of the mesh in the scene's positions buffer. */
const VERTICES_BEFORE = 9;

/**
 * An indexed tree (docs/design/indexed-leaves.md): the refit copies the posed
 * vertices into the mesh's range of a positions buffer, leaves the triangles
 * alone, and refits the boxes from the copied corners. Checked: the copied
 * positions are the posed ones bit for bit, the triangles and the marker words
 * around both ranges are untouched, and every box is exact.
 */
async function refitIndexedAcrossFrames(
    device: GPUDevice,
    name: string,
    geometry: MeshInput,
    layout: GpuLayoutMeshBvh,
    positionLayout: 'vec4' | 'packed',
    log: (message: string) => void,
): Promise<string[]> {
    const failures: string[] = [];
    const stride = positionLayout === 'packed' ? 3 : 4;
    const vertexCount = geometry.positions.length / 3;
    const nodeOffsetBytes = NODES_BEFORE * GPU_LAYOUT_NODE_BYTES;
    const triangleOffsetBytes = TRIANGLES_BEFORE * 16;
    const positionOffsetBytes = VERTICES_BEFORE * stride * 4;
    const nodes = storage(device, `${name} nodes`, placed(layout.nodes, nodeOffsetBytes));
    const triangles = storage(device, `${name} triangles`, placed(layout.triangles, triangleOffsetBytes));
    const positionBytes = vertexCount * stride * 4;
    const positions = storage(device, `${name} positions`, placed(new ArrayBuffer(positionBytes), positionOffsetBytes));
    const refit = new GpuMeshRefit(device, { label: `${name} refit`, diagnostics: true });
    await refit.prepare();
    refit.setTree(layout, { nodeWordOffset: nodeOffsetBytes / 4, triangleWordOffset: triangleOffsetBytes / 4, vertexBase: VERTICES_BEFORE, positionLayout });
    let vertexBuffer: GPUBuffer | null = null;
    try {
        for (let frame = 1; frame <= 3; frame++) {
            const moved = deform(geometry.positions, frame);
            vertexBuffer?.destroy();
            vertexBuffer = storage(device, `${name} vertices`, interleave(moved));
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            refit.encode(pass, { nodes, triangles, positions }, { positionBuffer: vertexBuffer, vertexCount, positionStrideFloats: VERTEX_STRIDE, positionOffsetFloats: VERTEX_OFFSET });
            pass.end();
            device.queue.submit([encoder.finish()]);

            const nodeWords = new Uint32Array(await read(device, nodes));
            const triangleWords = new Uint32Array(await read(device, triangles));
            const positionWords = new Uint32Array(await read(device, positions));
            if (!markersIntact(nodeWords, nodeOffsetBytes, layout.nodes.byteLength)) failures.push(`${name} frame ${frame}: the refit wrote outside the mesh's nodes`);
            if (!markersIntact(positionWords, positionOffsetBytes, positionBytes)) failures.push(`${name} frame ${frame}: the refit wrote outside the mesh's positions`);
            const triangleBytes = new Uint8Array(triangleWords.buffer, triangleOffsetBytes, layout.triangles.byteLength);
            if (!triangleBytes.every((byte, at) => byte === new Uint8Array(layout.triangles)[at])) failures.push(`${name} frame ${frame}: the refit changed the indexed triangles`);
            const movedBits = new Uint32Array(moved.buffer);
            for (let vertex = 0; vertex < vertexCount; vertex++) {
                const at = positionOffsetBytes / 4 + vertex * stride;
                if (positionWords[at] !== movedBits[vertex * 3] || positionWords[at + 1] !== movedBits[vertex * 3 + 1] || positionWords[at + 2] !== movedBits[vertex * 3 + 2]) {
                    failures.push(`${name} frame ${frame}: vertex ${vertex} was not copied bit for bit`);
                    break;
                }
            }
            const refitNodes = nodeWords.buffer.slice(nodeOffsetBytes, nodeOffsetBytes + layout.nodes.byteLength);
            const floats = new Float32Array(refitNodes, 0, 16);
            const right = new Uint32Array(refitNodes, 0, 16)[7] !== 0xffffffff ? 8 : 0;
            const bounds = new Float32Array([0, 1, 2].map((axis) => Math.min(floats[axis], floats[right + axis])).concat([0, 1, 2].map((axis) => Math.max(floats[4 + axis], floats[right + 4 + axis]))));
            const problems = checkGpuLayout({ ...layout, nodes: refitNodes, bounds }, { positions: moved, indices: geometry.indices });
            for (const problem of problems.slice(0, 5)) failures.push(`${name} frame ${frame}: ${problem}`);
            log(`${name} frame ${frame}: ${problems.length === 0 ? 'refit matches the deformed mesh exactly' : `${problems.length} problems`}`);
        }
    } finally {
        vertexBuffer?.destroy();
        nodes.destroy();
        triangles.destroy();
        positions.destroy();
        refit.dispose();
    }
    return failures;
}

await runGpuTest(
    async ({ device, log }) => {
        assert(GpuMeshRefit.isSupported(device), 'the BLAS refit needs subgroups');
        const lbvh = new GpuLbvhBuilder(device);
        await lbvh.prepare();
        const tinyBvh = await createTinyBvhBuilder();
        const failures: string[] = [];
        for (const [name, geometry] of [['grid', grid(90)], ['soup', soup(20_000)], ['one triangle', soup(1)]] as const) {
            const gpuTrees = await lbvh.build(geometry);
            const [gpuLayout] = await lbvh.readLayouts(gpuTrees);
            gpuTrees.destroy();
            assert(gpuLayout !== null, `${name}: the GPU LBVH refused the mesh`);
            failures.push(...(await refitAcrossFrames(device, `${name} (GPU LBVH)`, geometry, gpuLayout, log)));
            failures.push(...(await refitAcrossFrames(device, `${name} (tinybvh)`, geometry, tinyBvh.buildLayout(geometry).layout, log)));
            for (const positionLayout of ['vec4', 'packed'] as const) {
                const indexed = tinyBvh.buildLayout(geometry, { triangleLeaves: 'indexed' }).layout;
                failures.push(...(await refitIndexedAcrossFrames(device, `${name} (indexed, ${positionLayout})`, geometry, indexed, positionLayout, log)));
            }
        }
        assert(failures.length === 0, failures.slice(0, 12).join('\n'));
        return {};
    },
    { optionalFeatures: ['subgroups'] },
);

/**
 * Checks a tree in the tracer's GPU layout (gpu-layout.ts) against the mesh
 * it was built from, whoever built it. It is what every builder behind the
 * Builder contract must satisfy before the ray oracle is worth running:
 *
 * - the triangles are the mesh's, each exactly once, corners bit for bit,
 *   with the right original index and flags;
 * - the nodes form one tree from node 0: every node reached once, every
 *   triangle position in exactly one leaf, leaves of 1 to `maxLeafPrimitives`;
 * - every child box is exactly the box of the triangles below it (boxes are
 *   min and max of f32 values, so "exactly" is bit equality); with quantized
 *   nodes, every decoded box contains it with the 1/1024-step margin on every
 *   plane, and the frame is the one `nodeFrameOf` gives the exact root box;
 * - the parents, the depth, the leaf count and the mesh box agree with the nodes.
 *
 * Plain arrays and an explicit stack, so it handles meshes of millions of
 * triangles. Returns the problems found (at most `maxProblems`), empty if none.
 */

import {
    GPU_LAYOUT_EMPTY_CHILD,
    GPU_LAYOUT_INDEXED_FLAG_SHIFT,
    GPU_LAYOUT_INDEXED_ID_MASK,
    GPU_LAYOUT_INDEXED_TRIANGLE_BYTES,
    GPU_LAYOUT_INDEXED_TRIANGLE_WORD,
    GPU_LAYOUT_MAX_LEAF_PRIMITIVES,
    GPU_LAYOUT_NO_PARENT,
    GPU_LAYOUT_NODE_BYTES,
    GPU_LAYOUT_NODE_WORD,
    GPU_LAYOUT_NODE_WORDS,
    GPU_LAYOUT_TRIANGLE_BYTES,
    GPU_LAYOUT_TRIANGLE_WORD,
    type GpuLayoutMeshBvh,
} from './gpu-layout.ts';
import type { MeshInput } from './mesh-input.ts';
import { recordsFromIndexed } from './indexed-leaves.ts';
import { dequantizeNodes, nodeFrameOf, QUANTIZE_MARGIN_STEPS, readNodeFrame, type NodeFrame } from './quantized-nodes.ts';
import { GPU_LAYOUT_QUANTIZED_NODE_BYTES } from './gpu-layout.ts';

const TRIANGLE_WORDS = GPU_LAYOUT_TRIANGLE_BYTES / 4;
const CHILD_INDEX = [GPU_LAYOUT_NODE_WORD.leftIndex, GPU_LAYOUT_NODE_WORD.rightIndex] as const;
const CHILD_COUNT = [GPU_LAYOUT_NODE_WORD.leftCount, GPU_LAYOUT_NODE_WORD.rightCount] as const;
const CHILD_MIN = [GPU_LAYOUT_NODE_WORD.leftMin, GPU_LAYOUT_NODE_WORD.rightMin] as const;
const MAX_OFFSET = GPU_LAYOUT_NODE_WORD.leftMax - GPU_LAYOUT_NODE_WORD.leftMin;

/**
 * A merged static mesh's hit identity (`StaticMergeGroup` in trace/static-merge.ts),
 * for checking a tree whose records the scene assembly has written it into.
 */
export interface LayoutCheckMergedIdentity {
    /** Per triangle of the merged geometry: source instance index, index in the source mesh, and 1 to flip `frontFace`. */
    readonly sourceInstances: Uint32Array;
    readonly sourceTriangles: Uint32Array;
    readonly flipFrontFace: Uint8Array;
    /** Per triangle: the source's material as flags bits 16-31 hold it (material + 1, 0 for none). Omitted: all 0. */
    readonly sourceMaterials?: Uint16Array;
}

export interface LayoutCheckOptions {
    readonly maxLeafPrimitives?: number;
    readonly maxProblems?: number;
    /**
     * The records carry a merged mesh's identity: the original index is the
     * source mesh's triangle, the padding word the source instance, flags
     * bit 1 is set for a mirrored source, and flags bits 16-31 hold the
     * source's material + 1. Without it, every record's padding
     * must be 0 (as every builder writes it).
     */
    readonly merged?: LayoutCheckMergedIdentity;
}

/** Flags bit 1 (trace/layout.ts `TRIANGLE_FLAG_FLIP_FRONT_FACE`). */
const FLIP_FRONT_FACE = 2;
/** Where a merged triangle's material sits in its flags (trace/layout.ts `MATERIAL_SHIFT`). */
const MATERIAL_SHIFT = 16;

export function checkGpuLayout(layout: GpuLayoutMeshBvh, geometry: MeshInput, options: LayoutCheckOptions = {}): string[] {
    return checkLayout(layout, geometry, options, null);
}

/** `frame`: the tree's nodes were quantized on this frame (and `layout` holds them decoded). */
function checkLayout(layout: GpuLayoutMeshBvh, geometry: MeshInput, options: LayoutCheckOptions, frame: NodeFrame | null): string[] {
    if (layout.nodeFormat === 'quantized') {
        const bytes = (layout.nodeCount + 1) * GPU_LAYOUT_QUANTIZED_NODE_BYTES;
        if (layout.nodeCount < 1 || layout.nodes.byteLength !== bytes) return [`${layout.nodeCount} quantized nodes need ${bytes} bytes (the frame slot first), got ${layout.nodes.byteLength}`];
        return checkLayout(dequantizeNodes(layout), geometry, options, readNodeFrame(new Uint32Array(layout.nodes), 0));
    }
    if (layout.triangleLeaves === 'indexed') return checkIndexedLayout(layout, geometry, options, frame);
    const maxLeaf = options.maxLeafPrimitives ?? GPU_LAYOUT_MAX_LEAF_PRIMITIVES;
    const maxProblems = options.maxProblems ?? 20;
    const problems: string[] = [];
    const report = (problem: string): boolean => {
        if (problems.length < maxProblems) problems.push(problem);
        return problems.length >= maxProblems;
    };

    const { positions, indices, triangleFlags } = geometry;
    const triangleCount = indices ? indices.length / 3 : positions.length / 3 / 3;
    const { nodeCount } = layout;
    if (layout.triangleCount !== triangleCount) report(`the layout has ${layout.triangleCount} triangles, the mesh ${triangleCount}`);
    if (nodeCount < 1 || layout.nodes.byteLength < nodeCount * GPU_LAYOUT_NODE_BYTES) report(`${nodeCount} nodes in ${layout.nodes.byteLength} bytes`);
    if (layout.triangles.byteLength !== triangleCount * GPU_LAYOUT_TRIANGLE_BYTES) report(`triangles take ${layout.triangles.byteLength} bytes`);
    if (layout.parents.length < nodeCount) report(`${layout.parents.length} parents for ${nodeCount} nodes`);
    if (problems.length > 0) return problems;

    // ---- Triangles: the mesh's, each once, corners as they are.
    const triangleWords = new Uint32Array(layout.triangles);
    const triangleFloats = new Float32Array(layout.triangles);
    const positionBits = new Uint32Array(positions.buffer, positions.byteOffset, positions.length);
    const seen = new Uint8Array(triangleCount);
    const corners = [GPU_LAYOUT_TRIANGLE_WORD.vertex0, GPU_LAYOUT_TRIANGLE_WORD.vertex1, GPU_LAYOUT_TRIANGLE_WORD.vertex2];
    const merged = options.merged;
    // Merged records name (source instance, source triangle); map that back to the merged geometry's triangle.
    const mergedIndexOf = new Map<number, number>();
    if (merged) {
        for (let triangle = 0; triangle < triangleCount; triangle++) {
            if (merged.sourceTriangles[triangle] !== 0) continue;
            const source = merged.sourceInstances[triangle];
            if (mergedIndexOf.has(source)) {
                // Two sources with one instanceIndex: a merged hit can't say which (mergeStaticGeometry refuses this).
                report(`source instance ${source} starts at merged triangles ${mergedIndexOf.get(source)} and ${triangle}; merged sources need distinct instance indices`);
                return problems;
            }
            mergedIndexOf.set(source, triangle);
        }
    }
    for (let slot = 0; slot < triangleCount; slot++) {
        const base = slot * TRIANGLE_WORDS;
        const padding = triangleWords[base + GPU_LAYOUT_TRIANGLE_WORD.padding];
        let original = triangleWords[base + GPU_LAYOUT_TRIANGLE_WORD.originalIndex];
        if (merged) {
            const first = mergedIndexOf.get(padding);
            const at = first === undefined ? -1 : first + original;
            if (at < 0 || at >= triangleCount || merged.sourceInstances[at] !== padding || merged.sourceTriangles[at] !== original) {
                if (report(`triangle slot ${slot} names source instance ${padding} triangle ${original}, which the merged mesh does not hold`)) return problems;
                continue;
            }
            original = at;
        } else if (padding !== 0) {
            report(`triangle slot ${slot} has padding ${padding}; it is 0 outside a merged mesh`);
        }
        if (original >= triangleCount || seen[original]) {
            if (report(`triangle slot ${slot} holds triangle ${original}, which is out of range or already placed`)) return problems;
            continue;
        }
        seen[original] = 1;
        for (let corner = 0; corner < 3; corner++) {
            const vertex = indices ? indices[original * 3 + corner] : original * 3 + corner;
            for (let axis = 0; axis < 3; axis++) {
                if (triangleWords[base + corners[corner] + axis] !== positionBits[vertex * 3 + axis]) {
                    if (report(`triangle slot ${slot} (triangle ${original}) corner ${corner} differs from the mesh`)) return problems;
                }
            }
        }
        const flags =
            ((triangleFlags ? triangleFlags[original] : 0) |
                (merged?.flipFrontFace[original] ? FLIP_FRONT_FACE : 0) |
                ((merged?.sourceMaterials?.[original] ?? 0) << MATERIAL_SHIFT)) >>>
            0;
        if (triangleWords[base + GPU_LAYOUT_TRIANGLE_WORD.flags] !== flags) report(`triangle slot ${slot} has flags ${triangleWords[base + GPU_LAYOUT_TRIANGLE_WORD.flags]}, not ${flags}`);
    }
    if (problems.length > 0) return problems;

    // ---- Topology: a preorder walk from the root, every node once, every position in one leaf.
    const nodeWords = new Uint32Array(layout.nodes, 0, nodeCount * GPU_LAYOUT_NODE_WORDS);
    const nodeFloats = new Float32Array(layout.nodes, 0, nodeCount * GPU_LAYOUT_NODE_WORDS);
    const order = new Uint32Array(nodeCount);
    const depth = new Uint32Array(nodeCount);
    const reached = new Uint8Array(nodeCount);
    const covered = new Uint8Array(triangleCount);
    let ordered = 0;
    let leaves = 0;
    let deepest = 0;
    const stack: number[] = [0];
    reached[0] = 1;
    depth[0] = 1;
    if (layout.parents[0] !== GPU_LAYOUT_NO_PARENT) report(`the root's parent is ${layout.parents[0]}`);
    while (stack.length > 0) {
        const node = stack.pop() as number;
        order[ordered++] = node;
        deepest = Math.max(deepest, depth[node]);
        for (let side = 0; side < 2; side++) {
            const index = nodeWords[node * GPU_LAYOUT_NODE_WORDS + CHILD_INDEX[side]];
            const count = nodeWords[node * GPU_LAYOUT_NODE_WORDS + CHILD_COUNT[side]];
            if (index === GPU_LAYOUT_EMPTY_CHILD) {
                if (side === 0 || node !== 0 || triangleCount !== 1) report(`node ${node} side ${side} is empty`);
                continue;
            }
            if (count > 0) {
                leaves++;
                if (count > maxLeaf) report(`node ${node} side ${side} is a leaf of ${count} triangles (most ${maxLeaf})`);
                if (index + count > triangleCount) {
                    report(`node ${node} side ${side} covers positions past the last triangle`);
                    continue;
                }
                for (let position = index; position < index + count; position++) {
                    if (covered[position]) report(`position ${position} is in more than one leaf`);
                    covered[position] = 1;
                }
            } else {
                if (index >= nodeCount || reached[index]) {
                    if (report(`node ${node} side ${side} points at node ${index}, which is out of range or already reached`)) return problems;
                    continue;
                }
                reached[index] = 1;
                depth[index] = depth[node] + 1;
                if (layout.parents[index] !== node) report(`node ${index}'s parent is ${layout.parents[index]}, not ${node}`);
                stack.push(index);
            }
        }
    }
    if (ordered !== nodeCount) report(`${nodeCount - ordered} nodes are not reached from the root`);
    for (let position = 0; position < triangleCount; position++) {
        if (!covered[position]) {
            report(`position ${position} is in no leaf`);
            break;
        }
    }
    if (leaves !== layout.leafCount) report(`${leaves} leaves, the layout says ${layout.leafCount}`);
    if (deepest !== layout.maxDepth) report(`the deepest node is at level ${deepest}, the layout says ${layout.maxDepth}`);
    if (problems.length > 0) return problems;

    // ---- Boxes: children before parents (reverse preorder), each child box the exact box below it.
    const boxes = new Float32Array(nodeCount * 6);
    const childBox = new Float32Array(6);
    for (let at = nodeCount - 1; at >= 0; at--) {
        const node = order[at];
        const nodeBox = boxes.subarray(node * 6, node * 6 + 6);
        nodeBox.set([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
        for (let side = 0; side < 2; side++) {
            const index = nodeWords[node * GPU_LAYOUT_NODE_WORDS + CHILD_INDEX[side]];
            const count = nodeWords[node * GPU_LAYOUT_NODE_WORDS + CHILD_COUNT[side]];
            if (index === GPU_LAYOUT_EMPTY_CHILD) continue;
            if (count > 0) {
                childBox.set([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
                for (let slot = index; slot < index + count; slot++) {
                    for (const corner of corners) {
                        for (let axis = 0; axis < 3; axis++) {
                            const value = triangleFloats[slot * TRIANGLE_WORDS + corner + axis];
                            childBox[axis] = Math.min(childBox[axis], value);
                            childBox[3 + axis] = Math.max(childBox[3 + axis], value);
                        }
                    }
                }
            } else {
                childBox.set(boxes.subarray(index * 6, index * 6 + 6));
            }
            const stored = node * GPU_LAYOUT_NODE_WORDS + CHILD_MIN[side];
            for (let axis = 0; axis < 3; axis++) {
                const min = nodeFloats[stored + axis];
                const max = nodeFloats[stored + MAX_OFFSET + axis];
                if (frame) {
                    // Rounded outward with the margin: the decoded planes lie at least 1/1024 of a step outside.
                    const margin = frame.step[axis] * QUANTIZE_MARGIN_STEPS;
                    const empty = !(childBox[axis] <= childBox[3 + axis]);
                    if (!empty && !(childBox[axis] - min >= margin && max - childBox[3 + axis] >= margin)) {
                        if (report(`node ${node} side ${side}'s decoded box [${min}, ${max}] does not contain the box below it [${childBox[axis]}, ${childBox[3 + axis]}] with the margin (axis ${axis})`)) return problems;
                        break;
                    }
                } else if (min !== childBox[axis] || max !== childBox[3 + axis]) {
                    if (report(`node ${node} side ${side}'s box is not the box of what is below it (axis ${axis})`)) return problems;
                    break;
                }
            }
            for (let axis = 0; axis < 3; axis++) {
                nodeBox[axis] = Math.min(nodeBox[axis], childBox[axis]);
                nodeBox[3 + axis] = Math.max(nodeBox[3 + axis], childBox[3 + axis]);
            }
        }
    }
    for (let axis = 0; axis < 6; axis++) {
        if (layout.bounds[axis] !== boxes[axis]) {
            report(`the mesh box differs from the root's box at ${axis}: ${layout.bounds[axis]} vs ${boxes[axis]}`);
            break;
        }
    }
    if (frame) {
        const expected = nodeFrameOf(boxes.subarray(0, 6));
        for (let axis = 0; axis < 3; axis++) {
            if (frame.origin[axis] !== expected.origin[axis] || frame.step[axis] !== expected.step[axis]) {
                report(`the frame on axis ${axis} (origin ${frame.origin[axis]}, step ${frame.step[axis]}) is not the root box's (origin ${expected.origin[axis]}, step ${expected.step[axis]})`);
            }
        }
    }
    return problems;
}

/**
 * The SAH cost of a tree in the layout, relative to the root's box: the
 * expected cost of a random ray through the root's box, counting
 * `traversalCost` per inner node visited and `intersectionCost` per triangle
 * tested. What GPU builders are judged by against tinybvh (bvh-framework.md, B2).
 */
export function treeSahCost(tree: GpuLayoutMeshBvh, traversalCost = 1, intersectionCost = 1): number {
    // Quantized trees are measured on their decoded boxes (what the walk tests).
    const layout = dequantizeNodes(tree);
    const words = new Uint32Array(layout.nodes, 0, layout.nodeCount * GPU_LAYOUT_NODE_WORDS);
    const floats = new Float32Array(layout.nodes, 0, layout.nodeCount * GPU_LAYOUT_NODE_WORDS);
    const area = (base: number): number => {
        const x = floats[base + MAX_OFFSET] - floats[base];
        const y = floats[base + MAX_OFFSET + 1] - floats[base + 1];
        const z = floats[base + MAX_OFFSET + 2] - floats[base + 2];
        return x * y + y * z + z * x;
    };
    let total = 0;
    let rootArea = 0;
    for (let node = 0; node < layout.nodeCount; node++) {
        const base = node * GPU_LAYOUT_NODE_WORDS;
        let lower = [Infinity, Infinity, Infinity];
        let upper = [-Infinity, -Infinity, -Infinity];
        for (let side = 0; side < 2; side++) {
            if (words[base + CHILD_INDEX[side]] === GPU_LAYOUT_EMPTY_CHILD) continue;
            const box = base + CHILD_MIN[side];
            lower = lower.map((value, axis) => Math.min(value, floats[box + axis]));
            upper = upper.map((value, axis) => Math.max(value, floats[box + MAX_OFFSET + axis]));
            const count = words[base + CHILD_COUNT[side]];
            if (count > 0) total += intersectionCost * count * area(box);
        }
        const size = [upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2]];
        const nodeArea = size[0] * size[1] + size[1] * size[2] + size[2] * size[0];
        total += traversalCost * nodeArea;
        if (node === 0) rootArea = nodeArea;
    }
    return rootArea > 0 ? total / rootArea : 0;
}

/**
 * An indexed tree (`triangleLeaves: 'indexed'`): each triangle's three vertex
 * indices must be the mesh's own for its original triangle, in corner order,
 * and word 3 its original index and input flag (bit 28) with bits 29-31
 * clear. The rest (each triangle once, the topology, exact boxes over the
 * corners the indices name) is the records check on the same tree with its
 * corners copied out. A merged identity lives in an indexed scene's side
 * tables, not in the tree, so `merged` is refused here.
 */
function checkIndexedLayout(layout: GpuLayoutMeshBvh, geometry: MeshInput, options: LayoutCheckOptions, frame: NodeFrame | null): string[] {
    const maxProblems = options.maxProblems ?? 20;
    const problems: string[] = [];
    const report = (problem: string): boolean => {
        if (problems.length < maxProblems) problems.push(problem);
        return problems.length >= maxProblems;
    };
    if (options.merged) return ['an indexed tree carries no merged identity (the scene keeps it in side tables); check it without `merged`'];
    const { positions, indices, triangleFlags } = geometry;
    const triangleCount = indices ? indices.length / 3 : positions.length / 3 / 3;
    const vertexCount = positions.length / 3;
    if (layout.triangleCount !== triangleCount) return [`the layout has ${layout.triangleCount} triangles, the mesh ${triangleCount}`];
    if (layout.triangles.byteLength !== triangleCount * GPU_LAYOUT_INDEXED_TRIANGLE_BYTES) return [`indexed triangles take ${layout.triangles.byteLength} bytes`];
    const words = new Uint32Array(layout.triangles);
    for (let slot = 0; slot < triangleCount; slot++) {
        const base = slot * 4;
        const idFlags = words[base + GPU_LAYOUT_INDEXED_TRIANGLE_WORD.idFlags];
        const original = idFlags & GPU_LAYOUT_INDEXED_ID_MASK;
        const flags = idFlags >>> GPU_LAYOUT_INDEXED_FLAG_SHIFT;
        if (original >= triangleCount) {
            if (report(`triangle slot ${slot} holds triangle ${original}, which is out of range`)) return problems;
            continue;
        }
        const expected = triangleFlags ? triangleFlags[original] : 0;
        if (flags !== expected) report(`triangle slot ${slot} has flags ${flags} above its index, not ${expected}`);
        for (let corner = 0; corner < 3; corner++) {
            const vertex = indices ? indices[original * 3 + corner] : original * 3 + corner;
            const stored = words[base + corner];
            if (stored !== vertex || stored >= vertexCount) {
                if (report(`triangle slot ${slot} (triangle ${original}) corner ${corner} names vertex ${stored}, the mesh ${vertex}`)) return problems;
            }
        }
    }
    if (problems.length > 0) return problems;
    return checkLayout(recordsFromIndexed(layout, positions), geometry, options, frame);
}

// Wide nodes (docs/internals/design/wide-nodes.md): the tinybvh collapse and the BVH4 and CWBVH
// packers. Every triangle is reached exactly once, every decoded child box contains every corner
// of every triangle below it (which is what makes the wide walks give the binary walk's hits),
// and the CWBVH invariants hold (leaf slots of at most 3 triangles, offsets under 24, children
// and triangles consecutive in slot order).
import { describe, expect, it } from 'vitest';
import { createTinyBvhBuilder } from './index.ts';
import { TRIANGLE_WORD, TRIANGLE_WORDS } from './trace/layout.ts';
import type { MeshBvh, MeshGeometry } from './trace/mesh-bvh.ts';
import { buildMeshBvhWithTinyBvh } from './trace/mesh-bvh-topology.ts';
import { createRandom, randomTriangleSoup, tessellatedQuad } from './trace/oracle-fixtures.ts';
import { buildLbvhOnCpu } from './lbvh-cpu.ts';
import { convertToWide, decodeWideNode, refitWide, reorderTriangles, type WideMeshTree, type WideNodeFormat } from './wide-nodes.ts';
import { refitLayoutToPose } from './refit-cpu.ts';

const tinyBvh = await createTinyBvhBuilder();
const build = (geometry: MeshGeometry): MeshBvh => buildMeshBvhWithTinyBvh(geometry, tinyBvh);

const shifted = (geometry: MeshGeometry, offset: readonly number[], scale = 1): MeshGeometry => ({
    ...geometry,
    positions: geometry.positions.map((value, index) => value * scale + offset[index % 3]),
});

function meshes(): MeshGeometry[] {
    const random = createRandom(7);
    return [
        tessellatedQuad(17, 2),
        randomTriangleSoup(random, 3000, 5, 0.4),
        shifted(randomTriangleSoup(random, 500, 2, 0.3), [1.0e5, -3.0e4, 7.5]),
        shifted(randomTriangleSoup(random, 300, 1, 0.2), [0, 0, 0], 1e-6),
        shifted(randomTriangleSoup(random, 300, 1, 0.5), [0, 0, 0], 4e4),
        { positions: tessellatedQuad(6, 1).positions.map((value, index) => (index % 3 === 2 ? 3 : value)), indices: tessellatedQuad(6, 1).indices },
        { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]) },
        { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 2, 2, 3, 2, 2, 2, 3, 2]) },
    ];
}

/** Walks the whole wide tree: each triangle slot reached, and each decoded box checked against the corners below it. */
function checkTree(format: WideNodeFormat, mesh: MeshBvh, converted?: WideMeshTree, posedTriangles?: ArrayBuffer): { reached: number[]; nodes: number } {
    const wide = converted ?? convertToWide(mesh, format);
    const triangles = reorderTriangles(posedTriangles ?? mesh.triangles, 48, wide.triangleOrder);
    const floats = new Float32Array(triangles);
    const reached = new Array<number>(mesh.triangleCount).fill(0);
    let visited = 0;
    const failures: string[] = [];
    const triangleBox = (slot: number, box: number[]): void => {
        for (const v of [0, 4, 8]) {
            for (let axis = 0; axis < 3; axis++) {
                const value = floats[slot * TRIANGLE_WORDS + v + axis];
                box[axis] = Math.min(box[axis], value);
                box[axis + 3] = Math.max(box[axis + 3], value);
            }
        }
    };
    /** The exact box of every corner below `node`, after checking each decoded child box holds its part. */
    const visit = (node: number): number[] => {
        visited++;
        const children = decodeWideNode(format, wide.nodes, node);
        if (children.length === 0) failures.push(`node ${node} has no children`);
        const all = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
        let triangleCursor = -1;
        for (const child of children) {
            let exact = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
            if (child.node >= 0) {
                if (child.node <= node || child.node >= wide.nodeCount) failures.push(`node ${node} points at ${child.node}`);
                else exact = visit(child.node);
            } else {
                if (child.triangleCount < 1 || (format === 'cwbvh8' && child.triangleCount > 3)) failures.push(`a leaf of ${child.triangleCount}`);
                if (format === 'cwbvh8') {
                    // Leaf slots' triangles follow one another in slot order.
                    if (triangleCursor >= 0 && child.firstTriangle !== triangleCursor) failures.push(`node ${node}: leaf triangles not consecutive`);
                    triangleCursor = child.firstTriangle + child.triangleCount;
                }
                for (let k = 0; k < child.triangleCount; k++) {
                    reached[child.firstTriangle + k]++;
                    triangleBox(child.firstTriangle + k, exact);
                }
            }
            for (let axis = 0; axis < 3; axis++) {
                if (!(child.box[axis] <= exact[axis]) || !(child.box[axis + 3] >= exact[axis + 3])) {
                    failures.push(`node ${node} slot ${child.slot} axis ${axis}: [${child.box[axis]}, ${child.box[axis + 3]}] does not hold [${exact[axis]}, ${exact[axis + 3]}]`);
                }
                all[axis] = Math.min(all[axis], exact[axis]);
                all[axis + 3] = Math.max(all[axis + 3], exact[axis + 3]);
            }
        }
        return all;
    };
    visit(0);
    expect(failures.slice(0, 5)).toEqual([]);
    expect(visited).toBe(wide.nodeCount);
    return { reached, nodes: wide.nodeCount };
}

describe('wide nodes', () => {
    for (const format of ['bvh4', 'cwbvh8'] as const) {
        it(`${format}: every triangle once, every decoded box around the corners below it`, () => {
            for (const geometry of meshes()) {
                for (const mesh of [build(geometry), buildLbvhOnCpu(geometry)]) {
                    const { reached } = checkTree(format, mesh);
                    expect(reached.every((count) => count === 1)).toBe(true);
                }
            }
        });
    }

    it('cwbvh8 reorders the records and keeps each one whole', () => {
        const mesh = build(randomTriangleSoup(createRandom(3), 700, 4, 0.5));
        const wide = convertToWide(mesh, 'cwbvh8');
        expect(wide.triangleOrder).not.toBeNull();
        const order = wide.triangleOrder!;
        expect([...order].sort((a, b) => a - b)).toEqual([...Array(mesh.triangleCount).keys()]);
        const before = new Uint32Array(mesh.triangles);
        const after = new Uint32Array(reorderTriangles(mesh.triangles, 48, order));
        for (let k = 0; k < order.length; k++) {
            expect(after[k * TRIANGLE_WORDS + TRIANGLE_WORD.originalIndex]).toBe(before[order[k] * TRIANGLE_WORDS + TRIANGLE_WORD.originalIndex]);
        }
        // tinybvh's fullness numbers, for the design note: close to 8 slots and to 3 triangles per leaf on a soup.
        expect(wide.childrenPerNode).toBeGreaterThan(4);
        expect(wide.trianglesPerLeaf).toBeGreaterThan(1.5);
        expect(wide.stackNeed).toBe(wide.depth);
    });

    it('bvh4 keeps the binary order and fits the stack', () => {
        const mesh = build(randomTriangleSoup(createRandom(5), 5000, 6, 0.3));
        const wide = convertToWide(mesh, 'bvh4');
        expect(wide.triangleOrder).toBeNull();
        expect(wide.childrenPerNode).toBeGreaterThan(3);
        expect(wide.stackNeed).toBeLessThanOrEqual(3 * wide.depth);
        expect(wide.depth).toBeLessThan(mesh.maxDepth);
    });

    for (const format of ['bvh4', 'cwbvh8'] as const) {
        it(`${format}: a refit keeps the nodes' layout and holds the posed triangles`, () => {
            const random = createRandom(9);
            const geometry = randomTriangleSoup(random, 2000, 5, 0.4);
            const mesh = build(geometry);
            const wide = convertToWide(mesh, format);
            // The same pose repacks to the same bytes.
            const same = refitLayoutToPose(mesh, geometry.positions, geometry.indices);
            expect(new Uint8Array(refitWide(wide, { ...mesh, nodes: same.nodes }))).toEqual(new Uint8Array(wide.nodes));
            // A new pose: every vertex moved; the refit nodes still hold every triangle.
            const positions = geometry.positions.map((value, index) => value * 1.3 + Math.sin(index) * 0.7);
            const posed = refitLayoutToPose(mesh, positions, geometry.indices);
            const refitted = { ...wide, nodes: refitWide(wide, { ...mesh, nodes: posed.nodes }) };
            const { reached } = checkTree(format, mesh, refitted, posed.triangles!);
            expect(reached.every((count) => count === 1)).toBe(true);
        });
    }
});

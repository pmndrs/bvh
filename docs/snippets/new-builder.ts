// Code samples for docs/builders/adding-a-builder.mdx (see getting-started.ts for how they are checked).

// #region new-builder
import { buildMeshBvhFromTopology, singleLeafTopology, type MeshTreeTopology } from '@pmndrs/bvh';

/** A toy builder: cut the triangles' centres at the median of their longest axis. Yours replaces this function. */
export function medianSplitTopology(positions: Float32Array, indices: Uint32Array): MeshTreeTopology {
    const triangleCount = indices.length / 3;
    const order = Uint32Array.from({ length: triangleCount }, (_, triangle) => triangle);
    if (triangleCount <= 2) return singleLeafTopology(order, 0, triangleCount);
    const centre = (triangle: number, axis: number) =>
        (positions[indices[triangle * 3] * 3 + axis] + positions[indices[triangle * 3 + 1] * 3 + axis] + positions[indices[triangle * 3 + 2] * 3 + axis]) / 3;

    // Four words per node: leftIndex, leftCount, rightIndex, rightCount. A count above 0 is a leaf of
    // order[index .. index + count); a count of 0 is the inner node `index`. Node 0 is the root.
    const children: number[] = [];
    const emit = (start: number, count: number): number => {
        const node = children.length / 4;
        children.push(0, 0, 0, 0);
        const run = order.subarray(start, start + count);
        const extent = [0, 1, 2].map((axis) => Math.max(...run.map((t) => centre(t, axis))) - Math.min(...run.map((t) => centre(t, axis))));
        const axis = extent.indexOf(Math.max(...extent));
        run.sort((a, b) => centre(a, axis) - centre(b, axis));
        const left = count >> 1;
        for (const [side, from, size] of [[0, start, left], [2, start + left, count - left]]) {
            if (size <= 4) children.splice(node * 4 + side, 2, from, size);
            else children[node * 4 + side] = emit(from, size);
        }
        return node;
    };
    emit(0, triangleCount);
    return { order, children: Uint32Array.from(children) };
}

export function treeFromTopology(positions: Float32Array, indices: Uint32Array) {
    // The package writes the boxes, the triangle records and the parents from the shape alone.
    return buildMeshBvhFromTopology({ positions, indices }, medianSplitTopology(positions, indices));
}
// #endregion

// #region new-builder-oracle
import { buildReferenceScene, buildTraceScene, checkGpuLayout, compareClosest, createCpuTracer, createRandom, IDENTITY_TRANSFORM, randomRays, randomTriangleSoup, traceClosestReference, treeSahCost } from '@pmndrs/bvh';

// #endregion

export async function checkNewBuilder(): Promise<{ problems: string[]; sah: number; mismatches: number; rays: number }> {
    // #region new-builder-oracle
    const random = createRandom(1);
    const geometry = randomTriangleSoup(random, 2000, 10, 1.5);
    const mesh = { positions: geometry.positions, indices: new Uint32Array(geometry.positions.length / 3).map((_, i) => i) };
    const instances = [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY_TRANSFORM }];

    // 1. The tree is well formed: every triangle bit for bit, every box exactly the box below it.
    const layout = treeFromTopology(mesh.positions, mesh.indices);
    const problems = checkGpuLayout(layout, mesh); // [] when right
    const sah = treeSahCost(layout); // lower is better; compare with tinybvh's on the same mesh

    // 2. The CPU walk over your trees agrees with the tree-less reference on random rays.
    const scene = await buildTraceScene([mesh], instances, { build: (input) => treeFromTopology(input.positions, input.indices as Uint32Array) });
    const tracer = createCpuTracer(scene);
    const reference = buildReferenceScene([mesh], instances);
    const rays = randomRays(random, 5000, [-10, -10, -10, 10, 10, 10], 2);
    let mismatches = 0;
    for (const ray of rays) {
        const hit = tracer.closest(ray);
        if (compareClosest(hit && { hit: true, ...hit }, traceClosestReference(reference, ray), ray).verdict === 'mismatch') mismatches++;
    }
    // #endregion
    return { problems, sah, mismatches, rays: rays.length };
}

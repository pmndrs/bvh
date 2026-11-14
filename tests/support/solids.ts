/**
 * Closed, welded test meshes whose inside is known (the inside test's and the
 * SDF's unit tests): an axis-aligned box, a sphere from a split octahedron,
 * merging and inverting. Test-only: not exported from the package.
 */

import type { MeshInput } from '../../src/mesh-input.ts';

/** An axis-aligned box as 8 shared corners and 12 outward-wound triangles. */
export function closedBox(min: readonly number[], max: readonly number[]): { positions: Float32Array; indices: Uint32Array } {
    const positions = new Float32Array(24);
    for (let i = 0; i < 8; i++) positions.set([i & 1 ? max[0] : min[0], i & 2 ? max[1] : min[1], i & 4 ? max[2] : min[2]], i * 3);
    // Corner i has x from bit 0, y from bit 1, z from bit 2. Faces wound counter-clockwise seen from outside.
    const quads = [
        [0, 2, 3, 1], // -z
        [4, 5, 7, 6], // +z
        [0, 1, 5, 4], // -y
        [2, 6, 7, 3], // +y
        [0, 4, 6, 2], // -x
        [1, 3, 7, 5], // +x
    ];
    const indices = new Uint32Array(quads.flatMap(([a, b, c, d]) => [a, b, c, a, c, d]));
    return { positions, indices };
}

/** A sphere from an octahedron split `levels` times, corners on the sphere, welded, wound outward. */
export function closedSphere(radius: number, levels: number, center: readonly number[] = [0, 0, 0]): { positions: Float32Array; indices: Uint32Array } {
    const corners: number[][] = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
    let faces = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]];
    for (let level = 0; level < levels; level++) {
        const middle = new Map<string, number>();
        const mid = (a: number, b: number) => {
            const key = a < b ? `${a},${b}` : `${b},${a}`;
            let index = middle.get(key);
            if (index === undefined) {
                const p = [0, 1, 2].map((k) => corners[a][k] + corners[b][k]);
                const length = Math.hypot(p[0], p[1], p[2]);
                index = corners.push(p.map((v) => v / length)) - 1;
                middle.set(key, index);
            }
            return index;
        };
        faces = faces.flatMap(([a, b, c]) => {
            const ab = mid(a, b), bc = mid(b, c), ca = mid(c, a);
            return [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]];
        });
    }
    return { positions: new Float32Array(corners.flatMap((p) => p.map((v, k) => center[k] + v * radius))), indices: new Uint32Array(faces.flat()) };
}

export function merge(...meshes: { positions: Float32Array; indices: Uint32Array }[]): MeshInput {
    const positions: number[] = [];
    const indices: number[] = [];
    for (const mesh of meshes) {
        const base = positions.length / 3;
        positions.push(...mesh.positions);
        indices.push(...Array.from(mesh.indices, (i) => i + base));
    }
    return { positions: new Float32Array(positions), indices: new Uint32Array(indices) };
}

/** The same mesh wound the other way. */
export function inverted(mesh: { positions: Float32Array; indices: Uint32Array }) {
    const indices = mesh.indices.slice();
    for (let t = 0; t < indices.length; t += 3) [indices[t + 1], indices[t + 2]] = [indices[t + 2], indices[t + 1]];
    return { positions: mesh.positions, indices };
}

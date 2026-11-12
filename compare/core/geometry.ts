/**
 * The geometry every library is given: one world-space mesh (positions and a
 * 32-bit index). Scenes with many meshes and instances are merged into one,
 * because most of the libraries compared take one mesh at a time; each
 * library then builds one tree over the same triangles.
 */

import { TorusKnotGeometry } from 'three';
import type { CompareGeometry } from '../adapters/types.ts';

export function boundsOf(positions: Float32Array): number[] {
    const b = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    for (let i = 0; i < positions.length; i += 3) {
        for (let k = 0; k < 3; k++) {
            const v = positions[i + k];
            if (v < b[k]) b[k] = v;
            if (v > b[k + 3]) b[k + 3] = v;
        }
    }
    return b;
}

export function makeGeometry(name: string, positions: Float32Array, indices: Uint32Array): CompareGeometry {
    return { name, positions, indices, triangleCount: indices.length / 3, bounds: boundsOf(positions) };
}

/** three's `TorusKnotGeometry(1, 0.3, tubular, radial)`: 2 x tubular x radial triangles (100k by default). */
export function torusKnot(tubular = 1000, radial = 50): CompareGeometry {
    const knot = new TorusKnotGeometry(1, 0.3, tubular, radial);
    const positions = new Float32Array(knot.getAttribute('position').array as ArrayLike<number>);
    const indices = Uint32Array.from(knot.index!.array as ArrayLike<number>);
    knot.dispose();
    return makeGeometry(`torus knot ${(indices.length / 3 / 1000).toFixed(0)}k`, positions, indices);
}

export interface MeshPart {
    readonly positions: Float32Array;
    readonly indices?: ArrayLike<number> | null;
    /** Column-major object-to-world matrix; omit for identity. */
    readonly matrix?: ArrayLike<number> | null;
}

/** Every part transformed to world space and concatenated, in order: triangle ids run part after part. */
export function mergeParts(name: string, parts: readonly MeshPart[]): CompareGeometry {
    let vertices = 0;
    let triangles = 0;
    for (const part of parts) {
        vertices += part.positions.length / 3;
        triangles += part.indices ? part.indices.length / 3 : part.positions.length / 9;
    }
    const positions = new Float32Array(vertices * 3);
    const indices = new Uint32Array(triangles * 3);
    let v = 0;
    let t = 0;
    for (const part of parts) {
        const m = part.matrix;
        const p = part.positions;
        for (let i = 0; i < p.length; i += 3) {
            const x = p[i];
            const y = p[i + 1];
            const z = p[i + 2];
            const at = (v + i / 3) * 3;
            if (m) {
                positions[at] = m[0] * x + m[4] * y + m[8] * z + m[12];
                positions[at + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
                positions[at + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
            } else {
                positions[at] = x;
                positions[at + 1] = y;
                positions[at + 2] = z;
            }
        }
        const count = p.length / 3;
        if (part.indices) {
            for (let i = 0; i < part.indices.length; i++) indices[t * 3 + i] = part.indices[i] + v;
            t += part.indices.length / 3;
        } else {
            for (let i = 0; i < count; i++) indices[t * 3 + i] = v + i;
            t += count / 3;
        }
        v += count;
    }
    return makeGeometry(name, positions, indices);
}

/** The positions moved a little (a deterministic wave), for the refit timings. */
export function perturbed(geometry: CompareGeometry, amount: number, phase = 1): Float32Array {
    const out = new Float32Array(geometry.positions);
    for (let i = 0; i < out.length; i += 3) {
        out[i] += Math.sin(out[i + 1] * 3 + phase) * amount;
        out[i + 1] += Math.cos(out[i + 2] * 3 + phase) * amount;
        out[i + 2] += Math.sin(out[i] * 3 - phase) * amount;
    }
    return out;
}

export function diagonal(geometry: CompareGeometry): number {
    const b = geometry.bounds;
    return Math.hypot(b[3] - b[0], b[4] - b[1], b[5] - b[2]);
}

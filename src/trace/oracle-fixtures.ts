/**
 * Deterministic test scenes and rays for the BVH tests and the GPU oracle
 * page, exported for consumers' own oracles (a walk built on this one, a
 * renderer's correctness page). Not part of the tracer; nothing in production
 * imports this file.
 *
 * Everything is seeded so a failing case can be reproduced exactly.
 */

import type { MeshGeometry } from './mesh-bvh.ts';
import type { TraceRay } from './rays.ts';

export type Random = () => number;

/** mulberry32: a small seeded generator returning numbers in [0, 1). */
export function createRandom(seed: number): Random {
    let state = seed >>> 0;
    return () => {
        state = (state + 0x6d2b79f5) >>> 0;
        let mixed = state;
        mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
        mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
        return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
    };
}

export type Vector3 = [number, number, number];

export function randomUnitVector(random: Random): Vector3 {
    const z = 1 - 2 * random();
    const angle = 2 * Math.PI * random();
    const radius = Math.sqrt(Math.max(0, 1 - z * z));
    return [radius * Math.cos(angle), radius * Math.sin(angle), z];
}

/** Rounds to f32 so the CPU oracle and the GPU see exactly the same ray. */
export function toFloat32Ray(origin: readonly number[], direction: readonly number[], tMin: number, tMax: number): TraceRay {
    return {
        origin: [Math.fround(origin[0]), Math.fround(origin[1]), Math.fround(origin[2])],
        direction: [Math.fround(direction[0]), Math.fround(direction[1]), Math.fround(direction[2])],
        tMin: Math.fround(tMin),
        tMax: Math.fround(tMax),
    };
}

/** Unindexed triangles scattered in a cube of half-size `extent`, each about `size` wide. */
export function randomTriangleSoup(random: Random, triangleCount: number, extent: number, size: number): MeshGeometry {
    const positions = new Float32Array(triangleCount * 9);
    for (let triangle = 0; triangle < triangleCount; triangle++) {
        const center = [0, 1, 2].map(() => (random() * 2 - 1) * extent);
        for (let corner = 0; corner < 3; corner++) {
            for (let axis = 0; axis < 3; axis++) {
                positions[triangle * 9 + corner * 3 + axis] = center[axis] + (random() * 2 - 1) * size;
            }
        }
    }
    return { positions };
}

/**
 * A flat grid of `cells` x `cells` squares in the XY plane (z = 0) spanning
 * `[-half, half]`, each square split into two triangles along the same
 * diagonal. Vertices are shared, so every interior edge is shared by exactly
 * two triangles.
 */
export function tessellatedQuad(cells: number, half: number): MeshGeometry {
    const verticesPerRow = cells + 1;
    const positions = new Float32Array(verticesPerRow * verticesPerRow * 3);
    for (let row = 0; row <= cells; row++) {
        for (let column = 0; column <= cells; column++) {
            const base = (row * verticesPerRow + column) * 3;
            positions[base] = Math.fround(-half + (2 * half * column) / cells);
            positions[base + 1] = Math.fround(-half + (2 * half * row) / cells);
            positions[base + 2] = 0;
        }
    }
    const indices = new Uint32Array(cells * cells * 6);
    let slot = 0;
    for (let row = 0; row < cells; row++) {
        for (let column = 0; column < cells; column++) {
            const a = row * verticesPerRow + column;
            const b = a + 1;
            const c = a + verticesPerRow;
            const d = c + 1;
            // Counter-clockwise seen from +Z, diagonal from a to d.
            indices.set([a, b, d, a, d, c], slot);
            slot += 6;
        }
    }
    return { positions, indices };
}

/** What each ray of {@link sharedEdgeRays} is aimed at, by its `categories` entry. */
export const SHARED_EDGE_RAY_CATEGORIES = [
    'exact vertical ray on a grid line',
    'oblique ray at a grid-line edge',
    'oblique ray at a diagonal edge',
    'oblique ray at an interior vertex',
] as const;

export type SharedEdgeRayCategory = (typeof SHARED_EDGE_RAY_CATEGORIES)[number];

export interface SharedEdgeRayOptions {
    /** Cells per side of the quad, as `tessellatedQuad(cells, half)`. Default 8. */
    readonly cells?: number;
    /** Half the quad's width, as `tessellatedQuad(cells, half)`. Default 2. */
    readonly half?: number;
    /**
     * The quad instance's object-to-world transform, column-major 4x4 (as
     * `TraceSceneInstance.objectToWorld`). Rays are aimed in object space and
     * moved to world space with it. Default identity.
     */
    readonly objectToWorld?: readonly number[];
    /** Seed for {@link createRandom}. Default 555. */
    readonly seed?: number;
    /**
     * One ray in this many (ray 0, `verticalEvery`, ...) is an exact vertical
     * ray down a grid line (category 0); 0 for none. Default 10 without
     * `objectToWorld`, 0 with one: through a transform the ray is no longer
     * exactly along the line in the space the walk tests it in.
     */
    readonly verticalEvery?: number;
}

/**
 * Rays aimed exactly at the seams of `tessellatedQuad(cells, half)`: interior
 * grid-line edges, the diagonals, and interior vertices where six triangles
 * meet (categories cycle through {@link SHARED_EDGE_RAY_CATEGORIES}). Every
 * target lies strictly inside the quad (grid-line targets at least 0.2 from its
 * border, diagonal ones at least 5% of a cell), and origins sit 0.5 to 3.5
 * above or below it, so **every ray must hit**: a miss is a ray that
 * slipped between two triangles. Watertight walks let none through.
 *
 * Seeded and deterministic: the GPU oracle page (`trace-oracle`) traces exactly
 * these rays (seed 555 plain, 777 through its transform). Rays are rounded to
 * f32 with unit directions, `tMin` 0 and `tMax` 1e6, so a CPU check and the GPU
 * see the same ray. `categories[i]` is the index of ray `i`'s category.
 */
export function sharedEdgeRays(count: number, options: SharedEdgeRayOptions = {}): { rays: TraceRay[]; categories: Uint8Array } {
    const cells = options.cells ?? 8;
    const half = options.half ?? 2;
    const objectToWorld = options.objectToWorld;
    const verticalEvery = options.verticalEvery ?? (objectToWorld ? 0 : 10);
    if (!Number.isInteger(cells) || cells < 2) throw new RangeError(`sharedEdgeRays needs at least 2 cells per side, got ${cells}`);
    if (!(half > 0.2)) throw new RangeError(`sharedEdgeRays needs half larger than 0.2 (targets stay 0.2 inside the border), got ${half}`);
    const step = (2 * half) / cells;
    const random = createRandom(options.seed ?? 555);
    const move = (point: Vector3): Vector3 => (objectToWorld ? transformColumnMajor(objectToWorld, point) : point);

    const rays: TraceRay[] = [];
    const categories = new Uint8Array(count);
    const interiorGridLine = (): number => -half + step * (1 + Math.floor(random() * (cells - 1)));
    const insideCoordinate = (): number => (random() * 2 - 1) * (half - 0.2);

    for (let i = 0; i < count; i++) {
        const category = verticalEvery > 0 && i % verticalEvery === 0 ? 0 : 1 + (i % 3);
        categories[i] = category;

        let target: Vector3;
        if (category === 0 || category === 1) {
            // On a grid line: x or y is a multiple of the cell size, the other is anywhere along the line.
            target = random() < 0.5 ? [interiorGridLine(), insideCoordinate(), 0] : [insideCoordinate(), interiorGridLine(), 0];
        } else if (category === 2) {
            // Along the diagonal of a random cell.
            const column = Math.floor(random() * cells);
            const row = Math.floor(random() * cells);
            const along = 0.05 + random() * 0.9;
            target = [-half + step * (column + along), -half + step * (row + along), 0];
        } else {
            target = [interiorGridLine(), interiorGridLine(), 0];
        }

        const height = 0.5 + random() * 3;
        const above = random() < 0.5 ? 1 : -1;
        const origin: Vector3 =
            category === 0
                ? [target[0], target[1], above * height]
                : [target[0] + (random() * 2 - 1) * height * 2, target[1] + (random() * 2 - 1) * height * 2, above * height];

        const worldOrigin = move(origin);
        const worldTarget = move(target);
        const direction: Vector3 = [worldTarget[0] - worldOrigin[0], worldTarget[1] - worldOrigin[1], worldTarget[2] - worldOrigin[2]];
        const length = Math.hypot(direction[0], direction[1], direction[2]);
        rays.push(toFloat32Ray(worldOrigin, [direction[0] / length, direction[1] / length, direction[2] / length], 0, 1e6));
    }
    return { rays, categories };
}

/** A point through a column-major 4x4 affine transform. */
function transformColumnMajor(matrix: readonly number[], point: Vector3): Vector3 {
    return [
        matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12],
        matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13],
        matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14],
    ];
}

/**
 * A random affine object-to-world transform, column-major: rotation, then
 * per-axis scale (non-uniform), optionally mirrored on x, then translation.
 */
export function randomAffine(random: Random, options: { extent: number; mirror?: boolean }): number[] {
    // Random rotation from a unit quaternion.
    const [qx, qy, qz] = randomUnitVector(random);
    const angle = random() * Math.PI * 2;
    const s = Math.sin(angle / 2);
    const x = qx * s, y = qy * s, z = qz * s, w = Math.cos(angle / 2);
    const rotation = [
        1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w),
        2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
        2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y),
    ];
    const scale = [0.5 + random() * 1.5, 0.5 + random() * 1.5, 0.5 + random() * 1.5];
    if (options.mirror) scale[0] = -scale[0];
    const translation = [0, 1, 2].map(() => (random() * 2 - 1) * options.extent);
    // Each column of the rotation is scaled by that axis's scale.
    return [
        rotation[0] * scale[0], rotation[1] * scale[0], rotation[2] * scale[0], 0,
        rotation[3] * scale[1], rotation[4] * scale[1], rotation[5] * scale[1], 0,
        rotation[6] * scale[2], rotation[7] * scale[2], rotation[8] * scale[2], 0,
        translation[0], translation[1], translation[2], 1,
    ];
}

/** Rays from random points inside the box (grown by `margin`) in random directions. */
export function randomRays(
    random: Random,
    count: number,
    bounds: readonly [number, number, number, number, number, number],
    margin: number,
    options: { randomInterval?: boolean } = {},
): TraceRay[] {
    const rays: TraceRay[] = [];
    for (let i = 0; i < count; i++) {
        const origin = [0, 1, 2].map((axis) => bounds[axis] - margin + random() * (bounds[axis + 3] - bounds[axis] + 2 * margin));
        const direction = randomUnitVector(random);
        const tMin = options.randomInterval ? random() * 2 : 0;
        const tMax = options.randomInterval ? tMin + random() * 30 : 1e6;
        rays.push(toFloat32Ray(origin, direction, tMin, tMax));
    }
    return rays;
}

/**
 * The part of the radiance-cache team's `rc-rays.ts` (webgpu-ddgi, RC1.5) that
 * the surface-ray bench needs: Capsaicin-shaped screen probes. One primary
 * closest-hit ray per 16x16-pixel tile of a 1280x720 view, from the scene's
 * first camera bookmark, through a Halton (2, 3) jittered point of the tile;
 * the hits become probes (a point on the triangle, moved slightly inside it,
 * and the geometric normal turned toward the camera).
 *
 * Conventions are the platform's: rays leave a surface from a point pushed off
 * it along the normal by 256 rounding steps of the point's largest coordinate
 * (about 0.2 mm in Sponza), and hit points are moved 64 rounding steps toward
 * the triangle's centroid.
 */

import { HIT_RECORD_BYTES, HIT_RECORD_WORD, RAY_FLOATS } from '../../../../src/trace/index.ts';
import { RAY_FAR } from './config.ts';
import type { BenchInstance, BenchMeshGeometry } from './gltf-geometry.ts';
import { normalizeVector, type Vec3 } from './rays.ts';

/** One rounding step (ulp) of 1.0 in f32, 2^-23. */
const F32_ULP = 2 ** -23;
/** Below this distance from the origin the rounding step stays that of 1/16 m, as ulps vanish at 0. */
const ROUNDING_STEP_MIN_MAGNITUDE = 0.0625;
/** How far a hit point moves into its triangle, in rounding steps. */
const IN_SURFACE_STEPS = 64;
/** How far a ray's origin moves off the surface, in rounding steps. */
const NORMAL_OFFSET_STEPS = 256;

/** The size of one f32 rounding step of a point's largest coordinate. */
export function roundingStep(point: ArrayLike<number>): number {
    const magnitude = Math.max(Math.abs(point[0]), Math.abs(point[1]), Math.abs(point[2]), ROUNDING_STEP_MIN_MAGNITUDE);
    return magnitude * F32_ULP;
}

/** Where a ray leaving a surface starts: pushed off along the normal of the side it leaves from. */
export function offsetRayOrigin(position: ArrayLike<number>, normal: ArrayLike<number>): Vec3 {
    const push = NORMAL_OFFSET_STEPS * roundingStep(position);
    return [position[0] + normal[0] * push, position[1] + normal[1] * push, position[2] + normal[2] * push];
}

/** The radical inverse of `index` in `base`: the Halton sequence's coordinate for that base. */
export function halton(index: number, base: number): number {
    let result = 0;
    let fraction = 1 / base;
    let remaining = index;
    while (remaining > 0) {
        result += (remaining % base) * fraction;
        remaining = Math.floor(remaining / base);
        fraction /= base;
    }
    return result;
}

function cross(a: readonly number[], b: readonly number[]): Vec3 {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** Two unit tangents that make a right-handed frame with the unit normal `n` (Duff et al., JCGT 2017). */
export function tangentFrame(n: ArrayLike<number>): { tangent: Vec3; bitangent: Vec3 } {
    const sign = n[2] >= 0 ? 1 : -1;
    const a = -1 / (sign + n[2]);
    const b = n[0] * n[1] * a;
    return {
        tangent: [1 + sign * n[0] * n[0] * a, sign * b, -sign * n[0]],
        bitangent: [b, sign + n[1] * n[1] * a, -n[1]],
    };
}

/**
 * Writes one `BvhRay` at ray slot `slot`. The direction is rounded to f32 and
 * then normalised again, so what the GPU reads is a unit vector to f32 precision.
 */
export function writeRay(output: Float32Array, slot: number, origin: ArrayLike<number>, direction: ArrayLike<number>, tMin: number, tMax: number): void {
    const dx = Math.fround(direction[0]);
    const dy = Math.fround(direction[1]);
    const dz = Math.fround(direction[2]);
    const length = Math.hypot(dx, dy, dz);
    const base = slot * RAY_FLOATS;
    output[base] = origin[0];
    output[base + 1] = origin[1];
    output[base + 2] = origin[2];
    output[base + 3] = tMin;
    output[base + 4] = dx / length;
    output[base + 5] = dy / length;
    output[base + 6] = dz / length;
    output[base + 7] = tMax;
}

/**
 * A cosine-weighted direction about the unit normal `n` from two numbers in
 * [0, 1): `u1` picks the height (cos θ = sqrt(1 − u1), Malley's method), `u2`
 * the angle around the normal.
 */
export function cosineDirection(n: ArrayLike<number>, frame: { tangent: Vec3; bitangent: Vec3 }, u1: number, u2: number): Vec3 {
    const radius = Math.sqrt(u1);
    const angle = 2 * Math.PI * u2;
    const x = radius * Math.cos(angle);
    const y = radius * Math.sin(angle);
    const z = Math.sqrt(Math.max(0, 1 - u1));
    const { tangent, bitangent } = frame;
    return [
        tangent[0] * x + bitangent[0] * y + n[0] * z,
        tangent[1] * x + bitangent[1] * y + n[1] * z,
        tangent[2] * x + bitangent[2] * y + n[2] * z,
    ];
}

export interface RcCamera {
    position: Vec3;
    target: Vec3;
    /** Vertical field of view, degrees. */
    fovYDeg: number;
}

export interface RcView {
    width: number;
    height: number;
    /** Pixels per tile side; one probe per tile. */
    tileSize: number;
}

export interface TileGrid {
    columns: number;
    rows: number;
    count: number;
}

export function tileGrid(view: RcView): TileGrid {
    const columns = Math.ceil(view.width / view.tileSize);
    const rows = Math.ceil(view.height / view.tileSize);
    return { columns, rows, count: columns * rows };
}

/**
 * One primary ray per tile, row by row from the top-left tile, through the
 * tile's Halton (2, 3) point of index `tile + haltonFirstIndex`. A pinhole
 * camera looking at its target, y up.
 */
export function makePlacementRays(camera: RcCamera, view: RcView, haltonFirstIndex: number): Float32Array {
    const grid = tileGrid(view);
    const forward = normalizeVector([camera.target[0] - camera.position[0], camera.target[1] - camera.position[1], camera.target[2] - camera.position[2]]);
    const right = normalizeVector(cross(forward, [0, 1, 0]));
    const up = cross(right, forward);
    const tanHalfY = Math.tan((camera.fovYDeg * Math.PI) / 360);
    const tanHalfX = tanHalfY * (view.width / view.height);

    const rays = new Float32Array(grid.count * RAY_FLOATS);
    for (let tile = 0; tile < grid.count; tile++) {
        const column = tile % grid.columns;
        const row = Math.floor(tile / grid.columns);
        const jitterIndex = tile + haltonFirstIndex;
        const pixelX = Math.min(view.width, (column + halton(jitterIndex, 2)) * view.tileSize);
        const pixelY = Math.min(view.height, (row + halton(jitterIndex, 3)) * view.tileSize);
        const screenX = (2 * pixelX) / view.width - 1;
        const screenY = 1 - (2 * pixelY) / view.height;
        const direction = [0, 1, 2].map((axis) => forward[axis] + right[axis] * screenX * tanHalfX + up[axis] * screenY * tanHalfY);
        writeRay(rays, tile, camera.position, normalizeVector(direction), 0, RAY_FAR);
    }
    return rays;
}

/** Where a set of closest-hit rays landed. */
export interface HitSurfaces {
    /** Per surface, the ray that found it. */
    rayIndex: Uint32Array;
    /** Per surface, `x, y, z`: the point on the triangle, moved slightly inside it. */
    positions: Float64Array;
    /** Per surface, `x, y, z`: the unit geometric normal, turned to the side the ray arrived from. */
    normals: Float64Array;
    count: number;
    missCount: number;
    /** Hits on a collapsed triangle, which has no normal; dropped. */
    degenerateCount: number;
}

/**
 * Rebuilds each hit's surface from the hit records (`BvhHitRecord`) and the
 * scene's geometry: the point from the barycentrics on the world-space
 * triangle, moved into the triangle by 64 rounding steps (toward its
 * centroid), and the geometric normal turned to the side the ray came from.
 */
export function hitSurfaces(rays: Float32Array, hitRecords: ArrayBuffer, meshes: readonly BenchMeshGeometry[], instances: readonly BenchInstance[]): HitSurfaces {
    const rayCount = rays.length / RAY_FLOATS;
    if (hitRecords.byteLength < rayCount * HIT_RECORD_BYTES) throw new RangeError('fewer hit records than rays');
    const recordWords = HIT_RECORD_BYTES / 4;
    const floats = new Float32Array(hitRecords);
    const words = new Uint32Array(hitRecords);

    const rayIndex = new Uint32Array(rayCount);
    const positions = new Float64Array(rayCount * 3);
    const normals = new Float64Array(rayCount * 3);
    let count = 0;
    let missCount = 0;
    let degenerateCount = 0;
    const corner = new Float64Array(9);
    const onTriangle = new Float64Array(3);

    for (let ray = 0; ray < rayCount; ray++) {
        const record = ray * recordWords;
        const t = floats[record + HIT_RECORD_WORD.t];
        if (!(t >= 0)) {
            missCount++;
            continue;
        }
        const instanceIndex = words[record + HIT_RECORD_WORD.instance];
        const instance = instances[instanceIndex];
        if (!instance) throw new RangeError(`ray ${ray} reports instance ${instanceIndex}, which does not exist`);
        const mesh = meshes[instance.meshIndex];
        const triangle = words[record + HIT_RECORD_WORD.triangle];
        const m = instance.objectToWorld;
        for (let c = 0; c < 3; c++) {
            const vertex = mesh.indices[triangle * 3 + c] * 3;
            const x = mesh.positions[vertex];
            const y = mesh.positions[vertex + 1];
            const z = mesh.positions[vertex + 2];
            for (let axis = 0; axis < 3; axis++) corner[c * 3 + axis] = m[axis] * x + m[4 + axis] * y + m[8 + axis] * z + m[12 + axis];
        }

        const e1 = [corner[3] - corner[0], corner[4] - corner[1], corner[5] - corner[2]];
        const e2 = [corner[6] - corner[0], corner[7] - corner[1], corner[8] - corner[2]];
        const normal = cross(e1, e2);
        const normalLength = Math.hypot(normal[0], normal[1], normal[2]);
        if (!(normalLength > 0)) {
            degenerateCount++;
            continue;
        }
        const base = ray * RAY_FLOATS;
        const facing = normal[0] * rays[base + 4] + normal[1] * rays[base + 5] + normal[2] * rays[base + 6] > 0 ? -1 : 1;

        const u = Math.min(Math.max(floats[record + HIT_RECORD_WORD.barycentrics], 0), 1);
        const v = Math.min(Math.max(floats[record + HIT_RECORD_WORD.barycentrics + 1], 0), 1);
        const w = Math.min(Math.max(1 - u - v, 0), 1);
        for (let axis = 0; axis < 3; axis++) onTriangle[axis] = corner[axis] * w + corner[3 + axis] * u + corner[6 + axis] * v;
        const toCentroid = [0, 1, 2].map((axis) => (corner[axis] + corner[3 + axis] + corner[6 + axis]) / 3 - onTriangle[axis]);
        const centroidDistance = Math.hypot(toCentroid[0], toCentroid[1], toCentroid[2]);
        const nudge = Math.min(IN_SURFACE_STEPS * roundingStep(onTriangle), centroidDistance);
        const nudgeScale = nudge / Math.max(centroidDistance, 1e-30);

        for (let axis = 0; axis < 3; axis++) {
            positions[count * 3 + axis] = onTriangle[axis] + toCentroid[axis] * nudgeScale;
            normals[count * 3 + axis] = (facing * normal[axis]) / normalLength;
        }
        rayIndex[count] = ray;
        count++;
    }
    return {
        rayIndex: rayIndex.slice(0, count),
        positions: positions.slice(0, count * 3),
        normals: normals.slice(0, count * 3),
        count,
        missCount,
        degenerateCount,
    };
}

/** Per-ray problems of a packed ray set, at most `limit`: non-finite values, a direction that isn't unit length, an empty interval. */
export function rayProblems(name: string, rays: Float32Array, limit = 5): string[] {
    const problems: string[] = [];
    const count = rays.length / RAY_FLOATS;
    for (let ray = 0; ray < count && problems.length < limit; ray++) {
        const values = rays.subarray(ray * RAY_FLOATS, (ray + 1) * RAY_FLOATS);
        if (!values.every(Number.isFinite)) {
            problems.push(`${name} ray ${ray} has a value that isn't finite (${Array.from(values).join(', ')})`);
            continue;
        }
        const length = Math.hypot(values[4], values[5], values[6]);
        if (Math.abs(length - 1) > 1e-5) problems.push(`${name} ray ${ray}'s direction has length ${length}`);
        if (!(values[3] >= 0 && values[7] > values[3])) problems.push(`${name} ray ${ray} has the interval [${values[3]}, ${values[7]}]`);
    }
    return problems;
}

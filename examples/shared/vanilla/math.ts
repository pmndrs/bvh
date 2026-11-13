/**
 * The little matrix math the vanilla pages need, with no engine: column-major
 * 4x4 matrices in a `Float32Array(16)` (what WGSL's `mat4x4f` and
 * `SceneTrees`' `objectToWorld` read), and `[x, y, z]` vectors.
 */

export type Vec3 = [number, number, number];

export const vec3 = {
    add: (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
    sub: (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
    scale: (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s],
    dot: (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
    cross: (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
    length: (a: Vec3): number => Math.hypot(a[0], a[1], a[2]),
    normalize(a: Vec3): Vec3 {
        const length = Math.hypot(a[0], a[1], a[2]) || 1;
        return [a[0] / length, a[1] / length, a[2] / length];
    },
};

export const mat4 = {
    identity(): Float32Array {
        const m = new Float32Array(16);
        m[0] = m[5] = m[10] = m[15] = 1;
        return m;
    },

    /** `a * b` (b applied first). */
    multiply(a: ArrayLike<number>, b: ArrayLike<number>, out: Float32Array = new Float32Array(16)): Float32Array {
        const result = new Float32Array(16);
        for (let column = 0; column < 4; column++) {
            for (let row = 0; row < 4; row++) {
                let sum = 0;
                for (let k = 0; k < 4; k++) sum += a[k * 4 + row] * b[column * 4 + k];
                result[column * 4 + row] = sum;
            }
        }
        out.set(result);
        return out;
    },

    /** Translation, rotation (a unit quaternion `[x, y, z, w]`) and scale, as glTF's TRS. */
    compose(translation: ArrayLike<number>, rotation: ArrayLike<number>, scale: ArrayLike<number>, out: Float32Array = new Float32Array(16)): Float32Array {
        const [x, y, z, w] = [rotation[0], rotation[1], rotation[2], rotation[3]];
        const [sx, sy, sz] = [scale[0], scale[1], scale[2]];
        out[0] = (1 - 2 * (y * y + z * z)) * sx;
        out[1] = 2 * (x * y + z * w) * sx;
        out[2] = 2 * (x * z - y * w) * sx;
        out[3] = 0;
        out[4] = 2 * (x * y - z * w) * sy;
        out[5] = (1 - 2 * (x * x + z * z)) * sy;
        out[6] = 2 * (y * z + x * w) * sy;
        out[7] = 0;
        out[8] = 2 * (x * z + y * w) * sz;
        out[9] = 2 * (y * z - x * w) * sz;
        out[10] = (1 - 2 * (x * x + y * y)) * sz;
        out[11] = 0;
        out[12] = translation[0];
        out[13] = translation[1];
        out[14] = translation[2];
        out[15] = 1;
        return out;
    },

    /** A unit quaternion for `angle` radians about the unit `axis`. */
    axisAngle(axis: Vec3, angle: number): [number, number, number, number] {
        const s = Math.sin(angle / 2);
        return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)];
    },

    /** WebGPU's clip space (depth 0 to 1). */
    perspective(fovY: number, aspect: number, near: number, far: number): Float32Array {
        const f = 1 / Math.tan(fovY / 2);
        const m = new Float32Array(16);
        m[0] = f / aspect;
        m[5] = f;
        m[10] = far / (near - far);
        m[11] = -1;
        m[14] = (near * far) / (near - far);
        return m;
    },

    lookAt(eye: Vec3, target: Vec3, up: Vec3): Float32Array {
        const z = vec3.normalize(vec3.sub(eye, target));
        const x = vec3.normalize(vec3.cross(up, z));
        const y = vec3.cross(z, x);
        // prettier-ignore
        return new Float32Array([
            x[0], y[0], z[0], 0,
            x[1], y[1], z[1], 0,
            x[2], y[2], z[2], 0,
            -vec3.dot(x, eye), -vec3.dot(y, eye), -vec3.dot(z, eye), 1,
        ]);
    },

    transformPoint(m: ArrayLike<number>, p: ArrayLike<number>): Vec3 {
        return [m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13], m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]];
    },
};

/** `[minX, minY, minZ, maxX, maxY, maxZ]` around packed `x, y, z` positions (optionally through a matrix), grown into `box`. */
export function growBounds(positions: Float32Array, box: number[] = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity], matrix?: ArrayLike<number>): number[] {
    for (let i = 0; i < positions.length; i += 3) {
        const p = matrix ? mat4.transformPoint(matrix, [positions[i], positions[i + 1], positions[i + 2]]) : [positions[i], positions[i + 1], positions[i + 2]];
        for (let k = 0; k < 3; k++) {
            box[k] = Math.min(box[k], p[k]);
            box[k + 3] = Math.max(box[k + 3], p[k]);
        }
    }
    return box;
}

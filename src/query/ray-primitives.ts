/*!
 * Ray-to-point and ray-to-segment distances: ported from three.js r187
 * (src/math/Ray.js `distanceSqToPoint`, `closestPointToPoint`,
 * `distanceSqToSegment`; src/math/Line3.js `closestPointToPointParameter`),
 * mrdoob/three.js. three.js credits `distanceSqToSegment` to GeometricTools'
 * GteDistRaySegment.h. See THIRD_PARTY.md.
 *
 * The MIT License
 *
 * Copyright © 2010-2026 three.js authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
/**
 * The point and segment tests of three's `Points.raycast` and `Line.raycast`
 * (PARITY 3.2, 3.3), on plain numbers, with three's operations in three's
 * order: every product and sum is the one three's `Vector3` methods compute,
 * so the answers are three's bit for bit (`src/query/primitive-bvh.test.ts`
 * checks that against three's `Ray`). The ray's direction is taken as given:
 * three's rays are unit length, and so must these be.
 *
 * Results go into a caller's `Float64Array` (`out`), so a walk makes no
 * objects per primitive.
 */

/**
 * three's `Ray.distanceSqToPoint`: the squared distance from the ray (a half
 * line from `o` along the unit `d`) to the point `p`. The point on the ray is
 * written to `out[at..at+2]` when `out` is given (three's `closestPointToPoint`,
 * the same `directionDistance`).
 */
export function rayDistanceSqToPoint(
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    px: number,
    py: number,
    pz: number,
    out?: Float64Array,
    at = 0,
): number {
    // _vector.subVectors(point, this.origin).dot(this.direction)
    const directionDistance = (px - ox) * dx + (py - oy) * dy + (pz - oz) * dz;
    if (directionDistance < 0) {
        if (out) {
            out[at] = ox;
            out[at + 1] = oy;
            out[at + 2] = oz;
        }
        // this.origin.distanceToSquared(point)
        const ex = ox - px;
        const ey = oy - py;
        const ez = oz - pz;
        return ex * ex + ey * ey + ez * ez;
    }
    // _vector.copy(this.origin).addScaledVector(this.direction, directionDistance)
    const vx = ox + dx * directionDistance;
    const vy = oy + dy * directionDistance;
    const vz = oz + dz * directionDistance;
    if (out) {
        out[at] = vx;
        out[at + 1] = vy;
        out[at + 2] = vz;
    }
    // _vector.distanceToSquared(point)
    const ex = vx - px;
    const ey = vy - py;
    const ez = vz - pz;
    return ex * ex + ey * ey + ez * ez;
}

/**
 * three's `Ray.distanceSqToSegment(v0, v1, pointOnRay, pointOnSegment)`: the
 * squared distance between the ray and the segment `v0`-`v1`. Writes the
 * nearest point on the ray to `out[at..at+2]` and the nearest point on the
 * segment to `out[at+3..at+5]`.
 */
export function rayDistanceSqToSegment(
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    ax: number,
    ay: number,
    az: number,
    bx: number,
    by: number,
    bz: number,
    out: Float64Array,
    at = 0,
): number {
    // _segCenter.copy(v0).add(v1).multiplyScalar(0.5)
    const cx = (ax + bx) * 0.5;
    const cy = (ay + by) * 0.5;
    const cz = (az + bz) * 0.5;
    // _segDir.copy(v1).sub(v0).normalize(): divideScalar(length() || 1), which multiplies by 1 / scalar
    let sx = bx - ax;
    let sy = by - ay;
    let sz = bz - az;
    const inverseLength = 1 / (Math.sqrt(sx * sx + sy * sy + sz * sz) || 1);
    sx *= inverseLength;
    sy *= inverseLength;
    sz *= inverseLength;
    // _diff.copy(this.origin).sub(_segCenter)
    const fx = ox - cx;
    const fy = oy - cy;
    const fz = oz - cz;
    // v0.distanceTo(v1) * 0.5
    const gx = ax - bx;
    const gy = ay - by;
    const gz = az - bz;
    const segExtent = Math.sqrt(gx * gx + gy * gy + gz * gz) * 0.5;
    const a01 = -(dx * sx + dy * sy + dz * sz);
    const b0 = fx * dx + fy * dy + fz * dz;
    const b1 = -(fx * sx + fy * sy + fz * sz);
    const c = fx * fx + fy * fy + fz * fz;
    const det = Math.abs(1 - a01 * a01);
    let s0: number;
    let s1: number;
    let sqrDist: number;
    if (det > 0) {
        // The ray and segment are not parallel.
        s0 = a01 * b1 - b0;
        s1 = a01 * b0 - b1;
        const extDet = segExtent * det;
        if (s0 >= 0) {
            if (s1 >= -extDet) {
                if (s1 <= extDet) {
                    // region 0: minimum at interior points of ray and segment.
                    const invDet = 1 / det;
                    s0 *= invDet;
                    s1 *= invDet;
                    sqrDist = s0 * (s0 + a01 * s1 + 2 * b0) + s1 * (a01 * s0 + s1 + 2 * b1) + c;
                } else {
                    // region 1
                    s1 = segExtent;
                    s0 = Math.max(0, -(a01 * s1 + b0));
                    sqrDist = -s0 * s0 + s1 * (s1 + 2 * b1) + c;
                }
            } else {
                // region 5
                s1 = -segExtent;
                s0 = Math.max(0, -(a01 * s1 + b0));
                sqrDist = -s0 * s0 + s1 * (s1 + 2 * b1) + c;
            }
        } else if (s1 <= -extDet) {
            // region 4
            s0 = Math.max(0, -(-a01 * segExtent + b0));
            s1 = s0 > 0 ? -segExtent : Math.min(Math.max(-segExtent, -b1), segExtent);
            sqrDist = -s0 * s0 + s1 * (s1 + 2 * b1) + c;
        } else if (s1 <= extDet) {
            // region 3
            s0 = 0;
            s1 = Math.min(Math.max(-segExtent, -b1), segExtent);
            sqrDist = s1 * (s1 + 2 * b1) + c;
        } else {
            // region 2
            s0 = Math.max(0, -(a01 * segExtent + b0));
            s1 = s0 > 0 ? segExtent : Math.min(Math.max(-segExtent, -b1), segExtent);
            sqrDist = -s0 * s0 + s1 * (s1 + 2 * b1) + c;
        }
    } else {
        // The ray and segment are parallel.
        s1 = a01 > 0 ? -segExtent : segExtent;
        s0 = Math.max(0, -(a01 * s1 + b0));
        sqrDist = -s0 * s0 + s1 * (s1 + 2 * b1) + c;
    }
    // optionalPointOnRay.copy(this.origin).addScaledVector(this.direction, s0)
    out[at] = ox + dx * s0;
    out[at + 1] = oy + dy * s0;
    out[at + 2] = oz + dz * s0;
    // optionalPointOnSegment.copy(_segCenter).addScaledVector(_segDir, s1)
    out[at + 3] = cx + sx * s1;
    out[at + 4] = cy + sy * s1;
    out[at + 5] = cz + sz * s1;
    return sqrDist;
}

/**
 * three's `Line3.closestPointToPoint(point, true, target)`: the point of the
 * segment `a`-`b` nearest `p`, into `out[at..at+2]`; returns the squared
 * distance to it. One departure: a segment of length 0 gives its start, where
 * three's `0 / 0` gives `NaN`.
 */
export function segmentClosestPointToPoint(
    ax: number,
    ay: number,
    az: number,
    bx: number,
    by: number,
    bz: number,
    px: number,
    py: number,
    pz: number,
    out: Float64Array,
    at = 0,
): number {
    // _startP.subVectors(point, this.start); _startEnd.subVectors(this.end, this.start)
    const qx = px - ax;
    const qy = py - ay;
    const qz = pz - az;
    const ex = bx - ax;
    const ey = by - ay;
    const ez = bz - az;
    const startEnd2 = ex * ex + ey * ey + ez * ez;
    const startEndStartP = ex * qx + ey * qy + ez * qz;
    let t = startEnd2 > 0 ? startEndStartP / startEnd2 : 0;
    // MathUtils.clamp(t, 0, 1)
    t = Math.max(0, Math.min(1, t));
    // this.delta(target).multiplyScalar(t).add(this.start)
    const x = ex * t + ax;
    const y = ey * t + ay;
    const z = ez * t + az;
    out[at] = x;
    out[at + 1] = y;
    out[at + 2] = z;
    const rx = x - px;
    const ry = y - py;
    const rz = z - pz;
    return rx * rx + ry * ry + rz * rz;
}

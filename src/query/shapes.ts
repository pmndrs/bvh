/*!
 * Query shapes: ported from three-mesh-bvh 0.9.18 (src/math/ExtendedTriangle.js,
 * src/math/OrientedBox.js, src/math/SeparatingAxisBounds.js,
 * src/math/MathUtilities.js, gkjohnson/three-mesh-bvh) and from three.js r187
 * (src/math/Triangle.js, Plane.js, Line3.js, Box3.js, Matrix4.js, Vector3.js,
 * mrdoob/three.js). See THIRD_PARTY.md. Their licences (licenses/three-mesh-bvh.txt,
 * licenses/three.txt):
 *
 * MIT License
 *
 * Copyright (c) 2018 Garrett Johnson
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ---
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
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
/**
 * Query shapes for {@link CpuMeshBvh.shapecast} and its derived queries,
 * engine-free: three-mesh-bvh 0.9.18's `ExtendedTriangle`, `OrientedBox` and
 * `SeparatingAxisBounds` (and the three.js math they lean on: `Triangle`,
 * `Plane`, `Line3`, `Box3`, `Matrix4.invert`), ported operation for operation
 * so the same inputs give the same booleans and the same points, bit for bit.
 *
 * **One implementation.** Every algorithm is a function over plain shapes:
 * points are anything with `x`, `y`, `z`, a segment anything with `start` and
 * `end`, a plane `normal` and `constant`, a matrix sixteen column-major
 * numbers or anything with `elements`. The functions only read and write those
 * fields, never call methods on them, so a three `Vector3`, `Line3`, `Plane`
 * or `Matrix4` passes as is. The classes here ({@link ExtendedTriangle},
 * {@link OrientedBox}) hold the cached state in plain points; the
 * three-flavoured classes in `@pmndrs/bvh/three` hold it in three objects and
 * call the same functions.
 */

/** A point or direction: a three `Vector3` passes. */
export interface BvhVector3Like {
    x: number;
    y: number;
    z: number;
}

/** A line segment: a three `Line3` passes. */
export interface BvhSegmentLike {
    readonly start: BvhVector3Like;
    readonly end: BvhVector3Like;
}

/** A plane, `normal · p + constant = 0`: a three `Plane` passes. */
export interface BvhPlaneLike {
    readonly normal: BvhVector3Like;
    constant: number;
}

/** A sphere: a three `Sphere` passes. */
export interface BvhSphereLike {
    readonly center: BvhVector3Like;
    readonly radius: number;
}

/** An axis-aligned box: a three `Box3` passes. */
export interface BvhBox3Like {
    readonly min: BvhVector3Like;
    readonly max: BvhVector3Like;
}

/** Sixteen column-major numbers, or anything with `elements` (a three `Matrix4`). */
export type BvhMatrix4Like = ArrayLike<number> | { readonly elements: ArrayLike<number> };

/** A matrix whose numbers can be written: a three `Matrix4` passes. */
export interface BvhMatrix4Target {
    readonly elements: number[] | Float64Array | Float32Array;
}

/** The three corners of a triangle: a three `Triangle` passes. */
export interface BvhTriangleLike {
    readonly a: BvhVector3Like;
    readonly b: BvhVector3Like;
    readonly c: BvhVector3Like;
}

/** A plain point. */
export class BvhVector3 implements BvhVector3Like {
    // Numbers from the start: a field declared without one starts as `undefined`, and V8 then
    // stores every double written to it as a fresh heap number (garbage per write).
    x = 0;
    y = 0;
    z = 0;
    constructor(x = 0, y = 0, z = 0) {
        this.x = x;
        this.y = y;
        this.z = z;
    }
    set(x: number, y: number, z: number): this {
        this.x = x;
        this.y = y;
        this.z = z;
        return this;
    }
    copy(v: BvhVector3Like): this {
        this.x = v.x;
        this.y = v.y;
        this.z = v.z;
        return this;
    }
}

/** A plain segment. */
export class BvhSegment implements BvhSegmentLike {
    readonly start: BvhVector3;
    readonly end: BvhVector3;
    constructor(start: BvhVector3Like = new BvhVector3(), end: BvhVector3Like = new BvhVector3()) {
        this.start = new BvhVector3().copy(start);
        this.end = new BvhVector3().copy(end);
    }
}

/** A plain axis-aligned box (empty by default). */
export class BvhBox3 implements BvhBox3Like {
    readonly min = new BvhVector3(Infinity, Infinity, Infinity);
    readonly max = new BvhVector3(-Infinity, -Infinity, -Infinity);
}

/** A plain plane. */
export class BvhPlane implements BvhPlaneLike {
    readonly normal = new BvhVector3(1, 0, 0);
    constant = 0;
}

/** A plain column-major 4x4 matrix (identity by default). */
export class BvhMatrix4 implements BvhMatrix4Target {
    readonly elements: number[] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    copy(m: BvhMatrix4Like): this {
        copyMatrix(m, this);
        return this;
    }
}

const IDENTITY: readonly number[] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** A matrix's sixteen numbers (identity for `null` / `undefined`). */
export function matrixElements(m: BvhMatrix4Like | null | undefined): ArrayLike<number> {
    if (!m) return IDENTITY;
    return 'elements' in m ? m.elements : m;
}

/** Copies `m` into `target`'s elements. */
export function copyMatrix(m: BvhMatrix4Like | null | undefined, target: BvhMatrix4Target): void {
    const source = matrixElements(m);
    for (let i = 0; i < 16; i++) target.elements[i] = source[i];
}

/** `target` = inverse of `m` (three's `Matrix4.invert`, the same arithmetic; all zero when singular). */
export function invertMatrix(m: BvhMatrix4Like, target: BvhMatrix4Target): void {
    const te = matrixElements(m);
    const n11 = te[0], n21 = te[1], n31 = te[2], n41 = te[3];
    const n12 = te[4], n22 = te[5], n32 = te[6], n42 = te[7];
    const n13 = te[8], n23 = te[9], n33 = te[10], n43 = te[11];
    const n14 = te[12], n24 = te[13], n34 = te[14], n44 = te[15];
    const t1 = n11 * n22 - n21 * n12;
    const t2 = n11 * n32 - n31 * n12;
    const t3 = n11 * n42 - n41 * n12;
    const t4 = n21 * n32 - n31 * n22;
    const t5 = n21 * n42 - n41 * n22;
    const t6 = n31 * n42 - n41 * n32;
    const t7 = n13 * n24 - n23 * n14;
    const t8 = n13 * n34 - n33 * n14;
    const t9 = n13 * n44 - n43 * n14;
    const t10 = n23 * n34 - n33 * n24;
    const t11 = n23 * n44 - n43 * n24;
    const t12 = n33 * n44 - n43 * n34;
    const det = t1 * t12 - t2 * t11 + t3 * t10 + t4 * t9 - t5 * t8 + t6 * t7;
    const out = target.elements;
    if (det === 0) {
        for (let i = 0; i < 16; i++) out[i] = 0;
        return;
    }
    const detInv = 1 / det;
    out[0] = (n22 * t12 - n32 * t11 + n42 * t10) * detInv;
    out[1] = (n31 * t11 - n21 * t12 - n41 * t10) * detInv;
    out[2] = (n24 * t6 - n34 * t5 + n44 * t4) * detInv;
    out[3] = (n33 * t5 - n23 * t6 - n43 * t4) * detInv;
    out[4] = (n32 * t9 - n12 * t12 - n42 * t8) * detInv;
    out[5] = (n11 * t12 - n31 * t9 + n41 * t8) * detInv;
    out[6] = (n34 * t3 - n14 * t6 - n44 * t2) * detInv;
    out[7] = (n13 * t6 - n33 * t3 + n43 * t2) * detInv;
    out[8] = (n12 * t11 - n22 * t9 + n42 * t7) * detInv;
    out[9] = (n21 * t9 - n11 * t11 - n41 * t7) * detInv;
    out[10] = (n14 * t5 - n24 * t3 + n44 * t1) * detInv;
    out[11] = (n23 * t3 - n13 * t5 - n43 * t1) * detInv;
    out[12] = (n22 * t8 - n12 * t10 - n32 * t7) * detInv;
    out[13] = (n11 * t10 - n21 * t8 + n31 * t7) * detInv;
    out[14] = (n24 * t2 - n14 * t4 - n34 * t1) * detInv;
    out[15] = (n13 * t4 - n23 * t2 + n33 * t1) * detInv;
}

/** `p` = `e` · `p`, with the projective divide (three's `Vector3.applyMatrix4`). */
export function transformPoint(p: BvhVector3Like, e: ArrayLike<number>): void {
    const x = p.x, y = p.y, z = p.z;
    const w = 1 / (e[3] * x + e[7] * y + e[11] * z + e[15]);
    p.x = (e[0] * x + e[4] * y + e[8] * z + e[12]) * w;
    p.y = (e[1] * x + e[5] * y + e[9] * z + e[13]) * w;
    p.z = (e[2] * x + e[6] * y + e[10] * z + e[14]) * w;
}

const boxCorner = new BvhVector3();

/**
 * The axis-aligned box around box `[minX, minY, minZ, maxX, maxY, maxZ]`
 * transformed by `e`, into `out` (three's `Box3.applyMatrix4`: an empty box
 * stays as it is).
 */
export function transformBoxArray(box: ArrayLike<number>, e: ArrayLike<number>, out: Float64Array | Float32Array | number[]): void {
    if (box[3] < box[0] || box[4] < box[1] || box[5] < box[2]) {
        for (let i = 0; i < 6; i++) out[i] = box[i];
        return;
    }
    let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let corner = 0; corner < 8; corner++) {
        // three's corner order: x from bit 2, y from bit 1, z from bit 0.
        boxCorner.x = corner & 4 ? box[3] : box[0];
        boxCorner.y = corner & 2 ? box[4] : box[1];
        boxCorner.z = corner & 1 ? box[5] : box[2];
        transformPoint(boxCorner, e);
        minX = Math.min(minX, boxCorner.x);
        minY = Math.min(minY, boxCorner.y);
        minZ = Math.min(minZ, boxCorner.z);
        maxX = Math.max(maxX, boxCorner.x);
        maxY = Math.max(maxY, boxCorner.y);
        maxZ = Math.max(maxZ, boxCorner.z);
    }
    out[0] = minX;
    out[1] = minY;
    out[2] = minZ;
    out[3] = maxX;
    out[4] = maxY;
    out[5] = maxZ;
}

/**
 * Whether two boxes `[min..., max...]` overlap, touching included (three's `Box3.intersectsBox`).
 * Written as three writes it, so a NaN anywhere gives false (no overlap), not true.
 */
export function boxArraysIntersect(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
    return b[3] >= a[0] && b[0] <= a[3] && b[4] >= a[1] && b[1] <= a[4] && b[5] >= a[2] && b[2] <= a[5];
}

/** Whether a sphere touches a box (three's `Sphere.intersectsBox`). */
export function sphereIntersectsBox(sphere: BvhSphereLike, box: BvhBox3Like): boolean {
    const center = sphere.center;
    const x = Math.max(box.min.x, Math.min(box.max.x, center.x));
    const y = Math.max(box.min.y, Math.min(box.max.y, center.y));
    const z = Math.max(box.min.z, Math.min(box.max.z, center.z));
    const dx = x - center.x, dy = y - center.y, dz = z - center.z;
    return dx * dx + dy * dy + dz * dz <= sphere.radius * sphere.radius;
}

// ---- Vector helpers (three's arithmetic, in its order). Products and sums that return a
// number are written out in place (as `a.x * b.x + a.y * b.y + a.z * b.z`, three's order):
// a helper returning a double that V8 doesn't inline allocates a heap number per call.

function sub(a: BvhVector3Like, b: BvhVector3Like, out: BvhVector3Like): BvhVector3Like {
    out.x = a.x - b.x;
    out.y = a.y - b.y;
    out.z = a.z - b.z;
    return out;
}

function copy(v: BvhVector3Like, out: BvhVector3Like): BvhVector3Like {
    out.x = v.x;
    out.y = v.y;
    out.z = v.z;
    return out;
}

/** `out` = `a` × `b` (three's `crossVectors`; `out` may be `a`). */
function cross(a: BvhVector3Like, b: BvhVector3Like, out: BvhVector3Like): BvhVector3Like {
    const ax = a.x, ay = a.y, az = a.z;
    const bx = b.x, by = b.y, bz = b.z;
    out.x = ay * bz - az * by;
    out.y = az * bx - ax * bz;
    out.z = ax * by - ay * bx;
    return out;
}

/** `out` += `v` · `s` (three's `addScaledVector`). */
function addScaled(out: BvhVector3Like, v: BvhVector3Like, s: number): BvhVector3Like {
    out.x += v.x * s;
    out.y += v.y * s;
    out.z += v.z * s;
    return out;
}

// ---- Segments, planes, triangles (three's Line3, Plane, Triangle) ----

const lineStartP = new BvhVector3();
const lineStartEnd = new BvhVector3();

/** Parameter of the point on segment `start`-`end` closest to `point` (three's `Line3.closestPointToPointParameter`). */
export function segmentClosestPointParameter(start: BvhVector3Like, end: BvhVector3Like, point: BvhVector3Like, clampToLine: boolean): number {
    sub(point, start, lineStartP);
    sub(end, start, lineStartEnd);
    const startEnd2 = (lineStartEnd.x * lineStartEnd.x + lineStartEnd.y * lineStartEnd.y + lineStartEnd.z * lineStartEnd.z);
    if (startEnd2 === 0) return 0;
    let t = (lineStartEnd.x * lineStartP.x + lineStartEnd.y * lineStartP.y + lineStartEnd.z * lineStartP.z) / startEnd2;
    if (clampToLine) t = Math.max(0, Math.min(1, t));
    return t;
}

/** `target` = `start + (end - start) · t` (three's `Line3.at`). */
export function segmentAt(start: BvhVector3Like, end: BvhVector3Like, t: number, target: BvhVector3Like): BvhVector3Like {
    target.x = (end.x - start.x) * t + start.x;
    target.y = (end.y - start.y) * t + start.y;
    target.z = (end.z - start.z) * t + start.z;
    return target;
}

/** The point on segment `start`-`end` closest to `point` (three's `Line3.closestPointToPoint`). */
export function segmentClosestPointToPoint(start: BvhVector3Like, end: BvhVector3Like, point: BvhVector3Like, clampToLine: boolean, target: BvhVector3Like): BvhVector3Like {
    // segmentClosestPointParameter, written out (see the vector helpers' note).
    const ex = end.x - start.x, ey = end.y - start.y, ez = end.z - start.z;
    const startEnd2 = ex * ex + ey * ey + ez * ez;
    let t = 0;
    if (startEnd2 !== 0) {
        t = (ex * (point.x - start.x) + ey * (point.y - start.y) + ez * (point.z - start.z)) / startEnd2;
        if (clampToLine) t = Math.max(0, Math.min(1, t));
    }
    target.x = ex * t + start.x;
    target.y = ey * t + start.y;
    target.z = ez * t + start.z;
    return target;
}

/** `normal · point + constant` (three's `Plane.distanceToPoint`). */
export function planeDistanceToPoint(plane: BvhPlaneLike, point: BvhVector3Like): number {
    return (plane.normal.x * point.x + plane.normal.y * point.y + plane.normal.z * point.z) + plane.constant;
}

/** three's `Plane.setFromNormalAndCoplanarPoint`. */
export function setPlaneFromNormalAndPoint(plane: BvhPlaneLike, normal: BvhVector3Like, point: BvhVector3Like): void {
    copy(normal, plane.normal);
    plane.constant = -(point.x * plane.normal.x + point.y * plane.normal.y + point.z * plane.normal.z);
}

const planeDirection = new BvhVector3();

/** Where segment `start`-`end` crosses the plane, into `target`; false when it doesn't (three's `Plane.intersectLine`, clamped). */
export function planeIntersectSegment(plane: BvhPlaneLike, start: BvhVector3Like, end: BvhVector3Like, target: BvhVector3Like): boolean {
    const direction = sub(end, start, planeDirection);
    const denominator = (plane.normal.x * direction.x + plane.normal.y * direction.y + plane.normal.z * direction.z);
    if (denominator === 0) {
        if ((plane.normal.x * start.x + plane.normal.y * start.y + plane.normal.z * start.z + plane.constant) === 0) {
            copy(start, target);
            return true;
        }
        return false;
    }
    const t = -((start.x * plane.normal.x + start.y * plane.normal.y + start.z * plane.normal.z) + plane.constant) / denominator;
    if (t < 0 || t > 1) return false;
    addScaled(copy(start, target), direction, t);
    return true;
}

const normalScratch = new BvhVector3();

/** Unit normal of `a, b, c` (counter-clockwise front), zero when degenerate (three's `Triangle.getNormal`). */
export function triangleGetNormal(a: BvhVector3Like, b: BvhVector3Like, c: BvhVector3Like, target: BvhVector3Like): BvhVector3Like {
    sub(c, b, target);
    sub(a, b, normalScratch);
    cross(target, normalScratch, target);
    const lengthSq = target.x * target.x + target.y * target.y + target.z * target.z;
    if (lengthSq > 0) {
        const s = 1 / Math.sqrt(lengthSq);
        target.x *= s;
        target.y *= s;
        target.z *= s;
        return target;
    }
    target.x = target.y = target.z = 0;
    return target;
}

const baryV0 = new BvhVector3();
const baryV1 = new BvhVector3();
const baryV2 = new BvhVector3();
const baryOut = new BvhVector3();

/** Barycentric weights of `point` (projected) into `target`; false for a degenerate triangle (three's `Triangle.getBarycoord`). */
export function triangleGetBarycoord(point: BvhVector3Like, a: BvhVector3Like, b: BvhVector3Like, c: BvhVector3Like, target: BvhVector3Like): boolean {
    sub(c, a, baryV0);
    sub(b, a, baryV1);
    sub(point, a, baryV2);
    const dot00 = (baryV0.x * baryV0.x + baryV0.y * baryV0.y + baryV0.z * baryV0.z);
    const dot01 = (baryV0.x * baryV1.x + baryV0.y * baryV1.y + baryV0.z * baryV1.z);
    const dot02 = (baryV0.x * baryV2.x + baryV0.y * baryV2.y + baryV0.z * baryV2.z);
    const dot11 = (baryV1.x * baryV1.x + baryV1.y * baryV1.y + baryV1.z * baryV1.z);
    const dot12 = (baryV1.x * baryV2.x + baryV1.y * baryV2.y + baryV1.z * baryV2.z);
    const denom = dot00 * dot11 - dot01 * dot01;
    if (denom === 0) {
        target.x = target.y = target.z = 0;
        return false;
    }
    const invDenom = 1 / denom;
    const u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    const v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    target.x = 1 - u - v;
    target.y = v;
    target.z = u;
    return true;
}

/** Whether `point`, projected on the triangle's plane, is inside it (three's `Triangle.containsPoint`). */
export function triangleContainsPoint(point: BvhVector3Like, a: BvhVector3Like, b: BvhVector3Like, c: BvhVector3Like): boolean {
    if (!triangleGetBarycoord(point, a, b, c, baryOut)) return false;
    return baryOut.x >= 0 && baryOut.y >= 0 && baryOut.x + baryOut.y <= 1;
}

const vab = new BvhVector3();
const vac = new BvhVector3();
const vbc = new BvhVector3();
const vap = new BvhVector3();
const vbp = new BvhVector3();
const vcp = new BvhVector3();

/** The point of triangle `a, b, c` closest to `p` (three's `Triangle.closestPointToPoint`). */
export function triangleClosestPointToPoint(a: BvhVector3Like, b: BvhVector3Like, c: BvhVector3Like, p: BvhVector3Like, target: BvhVector3Like): BvhVector3Like {
    let v: number, w: number;
    sub(b, a, vab);
    sub(c, a, vac);
    sub(p, a, vap);
    const d1 = (vab.x * vap.x + vab.y * vap.y + vab.z * vap.z);
    const d2 = (vac.x * vap.x + vac.y * vap.y + vac.z * vap.z);
    if (d1 <= 0 && d2 <= 0) return copy(a, target);
    sub(p, b, vbp);
    const d3 = (vab.x * vbp.x + vab.y * vbp.y + vab.z * vbp.z);
    const d4 = (vac.x * vbp.x + vac.y * vbp.y + vac.z * vbp.z);
    if (d3 >= 0 && d4 <= d3) return copy(b, target);
    const vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0 && d1 - d3 > 0) {
        v = d1 / (d1 - d3);
        return addScaled(copy(a, target), vab, v);
    }
    sub(p, c, vcp);
    const d5 = (vab.x * vcp.x + vab.y * vcp.y + vab.z * vcp.z);
    const d6 = (vac.x * vcp.x + vac.y * vcp.y + vac.z * vcp.z);
    if (d6 >= 0 && d5 <= d6) return copy(c, target);
    const vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        w = d2 / (d2 - d6);
        return addScaled(copy(a, target), vac, w);
    }
    const va = d3 * d6 - d5 * d4;
    if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
        sub(c, b, vbc);
        w = (d4 - d3) / (d4 - d3 + (d5 - d6));
        return addScaled(copy(b, target), vbc, w);
    }
    const denom = 1 / (va + vb + vc);
    v = vb * denom;
    w = vc * denom;
    return addScaled(addScaled(copy(a, target), vab, v), vac, w);
}

// ---- three-mesh-bvh's MathUtilities ----

const lineDir1 = new BvhVector3();
const lineDir2 = new BvhVector3();
const lineV02 = new BvhVector3();

/** Parameters `[d, d2]` of the closest points of two infinite lines (three-mesh-bvh's `closestPointLineToLine`). */
function closestPointLineToLine(l1: BvhSegmentLike, l2: BvhSegmentLike, result: { x: number; y: number }): void {
    const v10 = lineDir1;
    const v32 = lineDir2;
    sub(l1.start, l2.start, lineV02);
    sub(l1.end, l1.start, lineDir1);
    sub(l2.end, l2.start, lineDir2);
    const d0232 = (lineV02.x * v32.x + lineV02.y * v32.y + lineV02.z * v32.z);
    const d3210 = (v32.x * v10.x + v32.y * v10.y + v32.z * v10.z);
    const d3232 = (v32.x * v32.x + v32.y * v32.y + v32.z * v32.z);
    const d0210 = (lineV02.x * v10.x + lineV02.y * v10.y + lineV02.z * v10.z);
    const d1010 = (v10.x * v10.x + v10.y * v10.y + v10.z * v10.z);
    const denom = d1010 * d3232 - d3210 * d3210;
    const d = denom !== 0 ? (d0232 * d3210 - d0210 * d3232) / denom : 0;
    result.x = d;
    result.y = (d0232 + d * d3210) / d3232;
}

const segmentParams = { x: 0, y: 0 };
const segmentTemp1 = new BvhVector3();
const segmentTemp2 = new BvhVector3();

/** The closest points of two segments, into `target1` (on `l1`) and `target2` (three-mesh-bvh's `closestPointsSegmentToSegment`). */
export function closestPointsSegmentToSegment(l1: BvhSegmentLike, l2: BvhSegmentLike, target1: BvhVector3Like, target2: BvhVector3Like): void {
    closestPointLineToLine(l1, l2, segmentParams);
    const d = segmentParams.x;
    const d2 = segmentParams.y;
    if (d >= 0 && d <= 1 && d2 >= 0 && d2 <= 1) {
        segmentAt(l1.start, l1.end, d, target1);
        segmentAt(l2.start, l2.end, d2, target2);
        return;
    } else if (d >= 0 && d <= 1) {
        segmentAt(l2.start, l2.end, d2 < 0 ? 0 : 1, target2);
        segmentClosestPointToPoint(l1.start, l1.end, target2, true, target1);
        return;
    } else if (d2 >= 0 && d2 <= 1) {
        segmentAt(l1.start, l1.end, d < 0 ? 0 : 1, target1);
        segmentClosestPointToPoint(l2.start, l2.end, target1, true, target2);
        return;
    }
    const p = d < 0 ? l1.start : l1.end;
    const p2 = d2 < 0 ? l2.start : l2.end;
    segmentClosestPointToPoint(l1.start, l1.end, p2, true, segmentTemp1);
    segmentClosestPointToPoint(l2.start, l2.end, p, true, segmentTemp2);
    if (((segmentTemp1.x - p2.x) * (segmentTemp1.x - p2.x) + (segmentTemp1.y - p2.y) * (segmentTemp1.y - p2.y) + (segmentTemp1.z - p2.z) * (segmentTemp1.z - p2.z)) <= ((segmentTemp2.x - p.x) * (segmentTemp2.x - p.x) + (segmentTemp2.y - p.y) * (segmentTemp2.y - p.y) + (segmentTemp2.z - p.z) * (segmentTemp2.z - p.z))) {
        copy(segmentTemp1, target1);
        copy(p2, target2);
    } else {
        copy(p, target1);
        copy(segmentTemp2, target2);
    }
}

const sphereClosest = new BvhVector3();
const sphereProjected = new BvhVector3();
const spherePlane = new BvhPlane();

/** Whether a sphere touches a triangle (three-mesh-bvh's `sphereIntersectTriangle`). */
export function sphereIntersectsTriangle(sphere: BvhSphereLike, triangle: BvhTriangleLike): boolean {
    const { radius, center } = sphere;
    const { a, b, c } = triangle;
    segmentClosestPointToPoint(a, b, center, true, sphereClosest);
    if (Math.sqrt((sphereClosest.x - center.x) * (sphereClosest.x - center.x) + (sphereClosest.y - center.y) * (sphereClosest.y - center.y) + (sphereClosest.z - center.z) * (sphereClosest.z - center.z)) <= radius) return true;
    segmentClosestPointToPoint(a, c, center, true, sphereClosest);
    if (Math.sqrt((sphereClosest.x - center.x) * (sphereClosest.x - center.x) + (sphereClosest.y - center.y) * (sphereClosest.y - center.y) + (sphereClosest.z - center.z) * (sphereClosest.z - center.z)) <= radius) return true;
    segmentClosestPointToPoint(b, c, center, true, sphereClosest);
    if (Math.sqrt((sphereClosest.x - center.x) * (sphereClosest.x - center.x) + (sphereClosest.y - center.y) * (sphereClosest.y - center.y) + (sphereClosest.z - center.z) * (sphereClosest.z - center.z)) <= radius) return true;
    // three's Triangle.getPlane: setFromCoplanarPoints, the normal normalized by its length.
    const n = cross(sub(c, b, spherePlane.normal), sub(a, b, normalScratch), spherePlane.normal);
    // Vector3.normalize: divideScalar(length || 1), which multiplies by the reciprocal.
    const s = 1 / (Math.sqrt(n.x * n.x + n.y * n.y + n.z * n.z) || 1);
    n.x *= s;
    n.y *= s;
    n.z *= s;
    spherePlane.constant = -(a.x * n.x + a.y * n.y + a.z * n.z);
    const dp = Math.abs((spherePlane.normal.x * center.x + spherePlane.normal.y * center.y + spherePlane.normal.z * center.z + spherePlane.constant));
    if (dp <= radius) {
        addScaled(copy(center, sphereProjected), spherePlane.normal, -(spherePlane.normal.x * center.x + spherePlane.normal.y * center.y + spherePlane.normal.z * center.z + spherePlane.constant));
        if (triangleContainsPoint(sphereProjected, a, b, c)) return true;
    }
    return false;
}

// ---- SeparatingAxisBounds ----

/** The interval a shape covers along an axis (three-mesh-bvh's `SeparatingAxisBounds`). */
export class SeparatingAxisBounds {
    min = Infinity;
    max = -Infinity;

    setFromPointsField(points: readonly BvhVector3Like[], field: 'x' | 'y' | 'z'): void {
        let min = Infinity;
        let max = -Infinity;
        for (let i = 0, l = points.length; i < l; i++) {
            const val = points[i][field];
            min = val < min ? val : min;
            max = val > max ? val : max;
        }
        this.min = min;
        this.max = max;
    }

    setFromPoints(axis: BvhVector3Like, points: readonly BvhVector3Like[]): void {
        let min = Infinity;
        let max = -Infinity;
        for (let i = 0, l = points.length; i < l; i++) {
            const val = (axis.x * points[i].x + axis.y * points[i].y + axis.z * points[i].z);
            min = val < min ? val : min;
            max = val > max ? val : max;
        }
        this.min = min;
        this.max = max;
    }

    setFromBox(axis: BvhVector3Like, box: BvhBox3Like): void {
        const boxMin = box.min;
        const boxMax = box.max;
        let min = Infinity;
        let max = -Infinity;
        const p = sabPoint;
        for (let x = 0; x <= 1; x++) {
            for (let y = 0; y <= 1; y++) {
                for (let z = 0; z <= 1; z++) {
                    p.x = boxMin.x * x + boxMax.x * (1 - x);
                    p.y = boxMin.y * y + boxMax.y * (1 - y);
                    p.z = boxMin.z * z + boxMax.z * (1 - z);
                    const val = (axis.x * p.x + axis.y * p.y + axis.z * p.z);
                    min = Math.min(val, min);
                    max = Math.max(val, max);
                }
            }
        }
        this.min = min;
        this.max = max;
    }

    isSeparated(other: SeparatingAxisBounds): boolean {
        return this.min > other.max || other.min > this.max;
    }
}

const sabPoint = new BvhVector3();

// ---- ExtendedTriangle ----

/**
 * The state an extended triangle caches (three-mesh-bvh's `ExtendedTriangle`
 * fields): both {@link ExtendedTriangle} and the three-flavoured one in
 * `@pmndrs/bvh/three` have it, and the functions below work on either.
 */
export interface ExtendedTriangleState extends BvhTriangleLike {
    readonly isExtendedTriangle: true;
    /** `[a, b, c]`. */
    readonly points: readonly BvhVector3Like[];
    /** The face normal, then the edges `a - b`, `b - c`, `c - a`. */
    readonly satAxes: readonly BvhVector3Like[];
    /** The corners' interval along each of {@link satAxes}. */
    readonly satBounds: readonly SeparatingAxisBounds[];
    readonly plane: BvhPlaneLike;
    isDegenerateIntoSegment: boolean;
    isDegenerateIntoPoint: boolean;
    readonly degenerateSegment: BvhSegmentLike;
    /** Set after moving `a`, `b` or `c`; the queries then call {@link updateExtendedTriangle} first. */
    needsUpdate: boolean;
}

/**
 * What a shapecast hands to `intersectsTriangle`: an extended triangle with
 * three-mesh-bvh's queries. {@link ExtendedTriangle} is one, and so is the
 * three-flavoured `ExtendedTriangle` of `@pmndrs/bvh/three`.
 */
export interface BvhQueryTriangle extends ExtendedTriangleState {
    update(): void;
    intersectsTriangle(other: BvhTriangleLike, target?: BvhSegmentLike | null, suppressLog?: boolean): boolean;
    intersectsSphere(sphere: BvhSphereLike): boolean;
    closestPointToSegment(segment: BvhSegmentLike, target1?: BvhVector3Like | null, target2?: BvhVector3Like | null): number;
    distanceToPoint(point: BvhVector3Like): number;
    distanceToTriangle(other: BvhTriangleLike, target1?: BvhVector3Like | null, target2?: BvhVector3Like | null): number;
}

const ZERO_EPSILON = 1e-15;
const ZERO_EPSILON_SQR = ZERO_EPSILON * ZERO_EPSILON;

function isNearZero(value: number): boolean {
    return Math.abs(value) < ZERO_EPSILON;
}

/** Recomputes a triangle's cached axes, plane and degeneracy (three-mesh-bvh's `ExtendedTriangle.update`). */
export function updateExtendedTriangle(t: ExtendedTriangleState): void {
    const { a, b, c, points, satAxes, satBounds } = t;
    const axis0 = satAxes[0];
    triangleGetNormal(a, b, c, axis0);
    satBounds[0].setFromPoints(axis0, points);
    const axis1 = sub(a, b, satAxes[1]);
    satBounds[1].setFromPoints(axis1, points);
    const axis2 = sub(b, c, satAxes[2]);
    satBounds[2].setFromPoints(axis2, points);
    const axis3 = sub(c, a, satAxes[3]);
    satBounds[3].setFromPoints(axis3, points);
    const lengthAB = Math.sqrt(axis1.x * axis1.x + axis1.y * axis1.y + axis1.z * axis1.z);
    const lengthBC = Math.sqrt(axis2.x * axis2.x + axis2.y * axis2.y + axis2.z * axis2.z);
    const lengthCA = Math.sqrt(axis3.x * axis3.x + axis3.y * axis3.y + axis3.z * axis3.z);
    t.isDegenerateIntoPoint = false;
    t.isDegenerateIntoSegment = false;
    const segment = t.degenerateSegment;
    if (lengthAB < ZERO_EPSILON) {
        if (lengthBC < ZERO_EPSILON || lengthCA < ZERO_EPSILON) {
            t.isDegenerateIntoPoint = true;
        } else {
            t.isDegenerateIntoSegment = true;
            copy(a, segment.start);
            copy(c, segment.end);
        }
    } else if (lengthBC < ZERO_EPSILON) {
        if (lengthCA < ZERO_EPSILON) {
            t.isDegenerateIntoPoint = true;
        } else {
            t.isDegenerateIntoSegment = true;
            copy(b, segment.start);
            copy(a, segment.end);
        }
    } else if (lengthCA < ZERO_EPSILON) {
        t.isDegenerateIntoSegment = true;
        copy(c, segment.start);
        copy(b, segment.end);
    }
    setPlaneFromNormalAndPoint(t.plane, axis0, a);
    t.needsUpdate = false;
}

function ensureUpdated(t: ExtendedTriangleState): void {
    if (t.needsUpdate) updateExtendedTriangle(t);
}

const segPoint1 = new BvhVector3();
const segPoint2 = new BvhVector3();
const segEdge = new BvhSegment();

/**
 * Distance from the triangle to a segment; `target1` and `target2` get the
 * closest points on the triangle and on the segment (three-mesh-bvh's
 * `ExtendedTriangle.closestPointToSegment`).
 */
export function triangleClosestPointToSegment(t: ExtendedTriangleState, segment: BvhSegmentLike, target1: BvhVector3Like | null = null, target2: BvhVector3Like | null = null): number {
    ensureUpdated(t);
    const { start, end } = segment;
    const points = t.points;
    let distSq: number;
    let closestDistanceSq = Infinity;
    const startDist = (t.plane.normal.x * start.x + t.plane.normal.y * start.y + t.plane.normal.z * start.z + t.plane.constant);
    const endDist = (t.plane.normal.x * end.x + t.plane.normal.y * end.y + t.plane.normal.z * end.z + t.plane.constant);
    if (startDist * endDist < 0 && planeIntersectSegment(t.plane, start, end, segPoint1) && triangleContainsPoint(segPoint1, t.a, t.b, t.c)) {
        if (target1) copy(segPoint1, target1);
        if (target2) copy(segPoint1, target2);
        return 0;
    }
    for (let i = 0; i < 3; i++) {
        const next = (i + 1) % 3;
        copy(points[i], segEdge.start);
        copy(points[next], segEdge.end);
        closestPointsSegmentToSegment(segEdge, segment, segPoint1, segPoint2);
        distSq = ((segPoint1.x - segPoint2.x) * (segPoint1.x - segPoint2.x) + (segPoint1.y - segPoint2.y) * (segPoint1.y - segPoint2.y) + (segPoint1.z - segPoint2.z) * (segPoint1.z - segPoint2.z));
        if (distSq < closestDistanceSq) {
            closestDistanceSq = distSq;
            if (target1) copy(segPoint1, target1);
            if (target2) copy(segPoint2, target2);
        }
    }
    triangleClosestPointToPoint(t.a, t.b, t.c, start, segPoint1);
    distSq = ((start.x - segPoint1.x) * (start.x - segPoint1.x) + (start.y - segPoint1.y) * (start.y - segPoint1.y) + (start.z - segPoint1.z) * (start.z - segPoint1.z));
    if (distSq < closestDistanceSq) {
        closestDistanceSq = distSq;
        if (target1) copy(segPoint1, target1);
        if (target2) copy(start, target2);
    }
    triangleClosestPointToPoint(t.a, t.b, t.c, end, segPoint1);
    distSq = ((end.x - segPoint1.x) * (end.x - segPoint1.x) + (end.y - segPoint1.y) * (end.y - segPoint1.y) + (end.z - segPoint1.z) * (end.z - segPoint1.z));
    if (distSq < closestDistanceSq) {
        closestDistanceSq = distSq;
        if (target1) copy(segPoint1, target1);
        if (target2) copy(end, target2);
    }
    return Math.sqrt(closestDistanceSq);
}

const pointTarget = new BvhVector3();

/** Distance from the triangle to a point (three-mesh-bvh's `ExtendedTriangle.distanceToPoint`). */
export function triangleDistanceToPoint(t: BvhTriangleLike, point: BvhVector3Like): number {
    triangleClosestPointToPoint(t.a, t.b, t.c, point, pointTarget);
    return Math.sqrt(((point.x - pointTarget.x) * (point.x - pointTarget.x) + (point.y - pointTarget.y) * (point.y - pointTarget.y) + (point.z - pointTarget.z) * (point.z - pointTarget.z)));
}

// intersectsTriangle's scratch.
const satScratchTriangle: ExtendedTriangleState = createExtendedTriangleState();
const cachedSatBounds = new SeparatingAxisBounds();
const cachedSatBounds2 = new SeparatingAxisBounds();
const tmpVec = new BvhVector3();
const dir1 = new BvhVector3();
const dir2 = new BvhVector3();
const tempDir = new BvhVector3();
const edge1 = new BvhSegment();
const edge2 = new BvhSegment();
const tempPoint = new BvhVector3();
const bounds1 = { x: 0, y: 0 };
const bounds2 = { x: 0, y: 0 };
const componentKeys = ['x', 'y', 'z'] as const;

function setSegment(target: BvhSegmentLike | null, start: BvhVector3Like, end: BvhVector3Like): void {
    if (!target) return;
    copy(start, target.start);
    copy(end, target.end);
}

function coplanarIntersectsTriangle(self: ExtendedTriangleState, other: ExtendedTriangleState, target: BvhSegmentLike | null, suppressLog: boolean): boolean {
    const planeNormal = tmpVec;
    if (!self.isDegenerateIntoPoint && !self.isDegenerateIntoSegment) copy(self.plane.normal, planeNormal);
    else copy(other.plane.normal, planeNormal);
    for (let i = 1; i < 4; i++) {
        const sb = self.satBounds[i];
        const sa = self.satAxes[i];
        cachedSatBounds.setFromPoints(sa, other.points);
        if (sb.isSeparated(cachedSatBounds)) return false;
        cross(copy(planeNormal, tempDir), sa, tempDir);
        cachedSatBounds.setFromPoints(tempDir, self.points);
        cachedSatBounds2.setFromPoints(tempDir, other.points);
        if (cachedSatBounds.isSeparated(cachedSatBounds2)) return false;
    }
    for (let i = 1; i < 4; i++) {
        const sb = other.satBounds[i];
        const sa = other.satAxes[i];
        cachedSatBounds.setFromPoints(sa, self.points);
        if (sb.isSeparated(cachedSatBounds)) return false;
        cross(planeNormal, sa, tempDir);
        cachedSatBounds.setFromPoints(tempDir, self.points);
        cachedSatBounds2.setFromPoints(tempDir, other.points);
        if (cachedSatBounds.isSeparated(cachedSatBounds2)) return false;
    }
    if (target) {
        if (!suppressLog) console.warn('ExtendedTriangle.intersectsTriangle: Triangles are coplanar which does not support an output edge. Setting edge to 0, 0, 0.');
        target.start.x = target.start.y = target.start.z = 0;
        target.end.x = target.end.y = target.end.z = 0;
    }
    return true;
}

function findSingleBounds(
    a: BvhVector3Like,
    b: BvhVector3Like,
    c: BvhVector3Like,
    aProj: number,
    bProj: number,
    cProj: number,
    aDist: number,
    bDist: number,
    cDist: number,
    bounds: { x: number; y: number },
    edge: BvhSegment,
): void {
    let t = aDist / (aDist - bDist);
    bounds.x = aProj + (bProj - aProj) * t;
    // three: edge.start.subVectors(b, a).multiplyScalar(t).add(a)
    edge.start.x = (b.x - a.x) * t + a.x;
    edge.start.y = (b.y - a.y) * t + a.y;
    edge.start.z = (b.z - a.z) * t + a.z;
    t = aDist / (aDist - cDist);
    bounds.y = aProj + (cProj - aProj) * t;
    edge.end.x = (c.x - a.x) * t + a.x;
    edge.end.y = (c.y - a.y) * t + a.y;
    edge.end.z = (c.z - a.z) * t + a.z;
}

/** True for the coplanar case. */
function findIntersectionLineBounds(
    self: ExtendedTriangleState,
    aProj: number,
    bProj: number,
    cProj: number,
    abDist: number,
    acDist: number,
    aDist: number,
    bDist: number,
    cDist: number,
    bounds: { x: number; y: number },
    edge: BvhSegment,
): boolean {
    if (abDist > 0) findSingleBounds(self.c, self.a, self.b, cProj, aProj, bProj, cDist, aDist, bDist, bounds, edge);
    else if (acDist > 0) findSingleBounds(self.b, self.a, self.c, bProj, aProj, cProj, bDist, aDist, cDist, bounds, edge);
    else if (bDist * cDist > 0 || aDist != 0) findSingleBounds(self.a, self.b, self.c, aProj, bProj, cProj, aDist, bDist, cDist, bounds, edge);
    else if (bDist != 0) findSingleBounds(self.b, self.a, self.c, bProj, aProj, cProj, bDist, aDist, cDist, bounds, edge);
    else if (cDist != 0) findSingleBounds(self.c, self.a, self.b, cProj, aProj, bProj, cDist, aDist, bDist, bounds, edge);
    else return true;
    return false;
}

function intersectTriangleSegment(triangle: ExtendedTriangleState, degenerate: ExtendedTriangleState, target: BvhSegmentLike | null, suppressLog: boolean): boolean {
    const segment = degenerate.degenerateSegment;
    const startDist = (triangle.plane.normal.x * segment.start.x + triangle.plane.normal.y * segment.start.y + triangle.plane.normal.z * segment.start.z + triangle.plane.constant);
    const endDist = (triangle.plane.normal.x * segment.end.x + triangle.plane.normal.y * segment.end.y + triangle.plane.normal.z * segment.end.z + triangle.plane.constant);
    if (isNearZero(startDist)) {
        if (isNearZero(endDist)) return coplanarIntersectsTriangle(triangle, degenerate, target, suppressLog);
        setSegment(target, segment.start, segment.start);
        return triangleContainsPoint(segment.start, triangle.a, triangle.b, triangle.c);
    } else if (isNearZero(endDist)) {
        setSegment(target, segment.end, segment.end);
        return triangleContainsPoint(segment.end, triangle.a, triangle.b, triangle.c);
    } else if (planeIntersectSegment(triangle.plane, segment.start, segment.end, tmpVec)) {
        setSegment(target, tmpVec, tmpVec);
        return triangleContainsPoint(tmpVec, triangle.a, triangle.b, triangle.c);
    }
    return false;
}

function intersectTrianglePoint(triangle: ExtendedTriangleState, degenerate: ExtendedTriangleState, target: BvhSegmentLike | null): boolean {
    const point = degenerate.a;
    if (isNearZero((triangle.plane.normal.x * point.x + triangle.plane.normal.y * point.y + triangle.plane.normal.z * point.z + triangle.plane.constant)) && triangleContainsPoint(point, triangle.a, triangle.b, triangle.c)) {
        setSegment(target, point, point);
        return true;
    }
    return false;
}

function intersectSegmentPoint(segmentTriangle: ExtendedTriangleState, pointTriangle: ExtendedTriangleState, target: BvhSegmentLike | null): boolean {
    const segment = segmentTriangle.degenerateSegment;
    const point = pointTriangle.a;
    segmentClosestPointToPoint(segment.start, segment.end, point, true, tmpVec);
    if (((point.x - tmpVec.x) * (point.x - tmpVec.x) + (point.y - tmpVec.y) * (point.y - tmpVec.y) + (point.z - tmpVec.z) * (point.z - tmpVec.z)) < ZERO_EPSILON_SQR) {
        setSegment(target, point, point);
        return true;
    }
    return false;
}

function handleDegenerateCases(self: ExtendedTriangleState, other: ExtendedTriangleState, target: BvhSegmentLike | null, suppressLog: boolean): boolean | undefined {
    if (self.isDegenerateIntoSegment) {
        if (other.isDegenerateIntoSegment) {
            const segment1 = self.degenerateSegment;
            const segment2 = other.degenerateSegment;
            const delta1 = sub(segment1.end, segment1.start, dir1);
            const delta2 = sub(segment2.end, segment2.start, dir2);
            const startDelta = sub(segment2.start, segment1.start, tmpVec);
            const denom = delta1.x * delta2.y - delta1.y * delta2.x;
            if (isNearZero(denom)) return false;
            const t = (startDelta.x * delta2.y - startDelta.y * delta2.x) / denom;
            const u = -(delta1.x * startDelta.y - delta1.y * startDelta.x) / denom;
            if (t < 0 || t > 1 || u < 0 || u > 1) return false;
            const z1 = segment1.start.z + delta1.z * t;
            const z2 = segment2.start.z + delta2.z * u;
            if (isNearZero(z1 - z2)) {
                if (target) {
                    addScaled(copy(segment1.start, target.start), delta1, t);
                    addScaled(copy(segment1.start, target.end), delta1, t);
                }
                return true;
            }
            return false;
        } else if (other.isDegenerateIntoPoint) {
            return intersectSegmentPoint(self, other, target);
        }
        return intersectTriangleSegment(other, self, target, suppressLog);
    } else if (self.isDegenerateIntoPoint) {
        if (other.isDegenerateIntoPoint) {
            if (((other.a.x - self.a.x) * (other.a.x - self.a.x) + (other.a.y - self.a.y) * (other.a.y - self.a.y) + (other.a.z - self.a.z) * (other.a.z - self.a.z)) < ZERO_EPSILON_SQR) {
                setSegment(target, self.a, self.a);
                return true;
            }
            return false;
        } else if (other.isDegenerateIntoSegment) {
            return intersectSegmentPoint(other, self, target);
        }
        return intersectTrianglePoint(other, self, target);
    } else if (other.isDegenerateIntoPoint) {
        return intersectTrianglePoint(self, other, target);
    } else if (other.isDegenerateIntoSegment) {
        return intersectTriangleSegment(self, other, target, suppressLog);
    }
    return undefined;
}

function isExtended(t: BvhTriangleLike): t is ExtendedTriangleState {
    return (t as Partial<ExtendedTriangleState>).isExtendedTriangle === true;
}

/** `other` as an updated extended triangle: itself, or a copy in `scratch`. */
function asExtended(other: BvhTriangleLike, scratch: ExtendedTriangleState): ExtendedTriangleState {
    if (!isExtended(other)) {
        copy(other.a, scratch.a);
        copy(other.b, scratch.b);
        copy(other.c, scratch.c);
        updateExtendedTriangle(scratch);
        return scratch;
    }
    ensureUpdated(other);
    return other;
}

/**
 * Whether two triangles intersect; `target` gets the intersection segment
 * (zero for coplanar triangles, with a warning unless `suppressLog`).
 * three-mesh-bvh's `ExtendedTriangle.intersectsTriangle`: Möller's test, with
 * degenerate triangles first and a separating-axis test for coplanar ones.
 */
export function triangleIntersectsTriangle(self: ExtendedTriangleState, otherTriangle: BvhTriangleLike, target: BvhSegmentLike | null = null, suppressLog = false): boolean {
    ensureUpdated(self);
    const other = asExtended(otherTriangle, satScratchTriangle);
    const res = handleDegenerateCases(self, other, target, suppressLog);
    if (res !== undefined) return res;
    const plane1 = self.plane;
    const plane2 = other.plane;
    let a1Dist = (plane2.normal.x * self.a.x + plane2.normal.y * self.a.y + plane2.normal.z * self.a.z + plane2.constant);
    let b1Dist = (plane2.normal.x * self.b.x + plane2.normal.y * self.b.y + plane2.normal.z * self.b.z + plane2.constant);
    let c1Dist = (plane2.normal.x * self.c.x + plane2.normal.y * self.c.y + plane2.normal.z * self.c.z + plane2.constant);
    if (isNearZero(a1Dist)) a1Dist = 0;
    if (isNearZero(b1Dist)) b1Dist = 0;
    if (isNearZero(c1Dist)) c1Dist = 0;
    const a1b1Dist = a1Dist * b1Dist;
    const a1c1Dist = a1Dist * c1Dist;
    if (a1b1Dist > 0 && a1c1Dist > 0) return false;
    let a2Dist = (plane1.normal.x * other.a.x + plane1.normal.y * other.a.y + plane1.normal.z * other.a.z + plane1.constant);
    let b2Dist = (plane1.normal.x * other.b.x + plane1.normal.y * other.b.y + plane1.normal.z * other.b.z + plane1.constant);
    let c2Dist = (plane1.normal.x * other.c.x + plane1.normal.y * other.c.y + plane1.normal.z * other.c.z + plane1.constant);
    if (isNearZero(a2Dist)) a2Dist = 0;
    if (isNearZero(b2Dist)) b2Dist = 0;
    if (isNearZero(c2Dist)) c2Dist = 0;
    const a2b2Dist = a2Dist * b2Dist;
    const a2c2Dist = a2Dist * c2Dist;
    if (a2b2Dist > 0 && a2c2Dist > 0) return false;
    copy(plane1.normal, dir1);
    copy(plane2.normal, dir2);
    const intersectionLine = cross(dir1, dir2, dir1);
    let componentIndex = 0;
    let maxComponent = Math.abs(intersectionLine.x);
    const comp1 = Math.abs(intersectionLine.y);
    if (comp1 > maxComponent) {
        maxComponent = comp1;
        componentIndex = 1;
    }
    const comp2 = Math.abs(intersectionLine.z);
    if (comp2 > maxComponent) componentIndex = 2;
    const key = componentKeys[componentIndex];
    const a1Proj = self.a[key], b1Proj = self.b[key], c1Proj = self.c[key];
    const a2Proj = other.a[key], b2Proj = other.b[key], c2Proj = other.c[key];
    if (findIntersectionLineBounds(self, a1Proj, b1Proj, c1Proj, a1b1Dist, a1c1Dist, a1Dist, b1Dist, c1Dist, bounds1, edge1)) {
        return coplanarIntersectsTriangle(self, other, target, suppressLog);
    }
    if (findIntersectionLineBounds(other, a2Proj, b2Proj, c2Proj, a2b2Dist, a2c2Dist, a2Dist, b2Dist, c2Dist, bounds2, edge2)) {
        return coplanarIntersectsTriangle(self, other, target, suppressLog);
    }
    if (bounds1.y < bounds1.x) {
        const tmp = bounds1.y;
        bounds1.y = bounds1.x;
        bounds1.x = tmp;
        copy(edge1.start, tempPoint);
        copy(edge1.end, edge1.start);
        copy(tempPoint, edge1.end);
    }
    if (bounds2.y < bounds2.x) {
        const tmp = bounds2.y;
        bounds2.y = bounds2.x;
        bounds2.x = tmp;
        copy(edge2.start, tempPoint);
        copy(edge2.end, edge2.start);
        copy(tempPoint, edge2.end);
    }
    if (bounds1.y < bounds2.x || bounds2.y < bounds1.x) return false;
    if (target) {
        copy(bounds2.x > bounds1.x ? edge2.start : edge1.start, target.start);
        copy(bounds2.y < bounds1.y ? edge2.end : edge1.end, target.end);
    }
    return true;
}

const distPoint = new BvhVector3();
const distPoint2 = new BvhVector3();
const distLine1 = new BvhSegment();
const distLine2 = new BvhSegment();
const cornerFields = ['a', 'b', 'c'] as const;

/**
 * Distance between two triangles; `target1` and `target2` get the closest
 * points (three-mesh-bvh's `ExtendedTriangle.distanceToTriangle`).
 */
export function triangleDistanceToTriangle(self: ExtendedTriangleState, other: BvhTriangleLike, target1: BvhVector3Like | null = null, target2: BvhVector3Like | null = null): number {
    const lineTarget = target1 || target2 ? distLine1 : null;
    if (triangleIntersectsTriangle(self, other, lineTarget, true)) {
        if (lineTarget) {
            // three's Line3.getCenter: (start + end) * 0.5
            if (target1) {
                target1.x = (lineTarget.start.x + lineTarget.end.x) * 0.5;
                target1.y = (lineTarget.start.y + lineTarget.end.y) * 0.5;
                target1.z = (lineTarget.start.z + lineTarget.end.z) * 0.5;
            }
            if (target2) {
                target2.x = (lineTarget.start.x + lineTarget.end.x) * 0.5;
                target2.y = (lineTarget.start.y + lineTarget.end.y) * 0.5;
                target2.z = (lineTarget.start.z + lineTarget.end.z) * 0.5;
            }
        }
        return 0;
    }
    let closestDistanceSq = Infinity;
    for (let i = 0; i < 3; i++) {
        let dist: number;
        const field = cornerFields[i];
        const otherVec = other[field];
        triangleClosestPointToPoint(self.a, self.b, self.c, otherVec, distPoint);
        dist = ((otherVec.x - distPoint.x) * (otherVec.x - distPoint.x) + (otherVec.y - distPoint.y) * (otherVec.y - distPoint.y) + (otherVec.z - distPoint.z) * (otherVec.z - distPoint.z));
        if (dist < closestDistanceSq) {
            closestDistanceSq = dist;
            if (target1) copy(distPoint, target1);
            if (target2) copy(otherVec, target2);
        }
        const thisVec = self[field];
        triangleClosestPointToPoint(other.a, other.b, other.c, thisVec, distPoint);
        dist = ((thisVec.x - distPoint.x) * (thisVec.x - distPoint.x) + (thisVec.y - distPoint.y) * (thisVec.y - distPoint.y) + (thisVec.z - distPoint.z) * (thisVec.z - distPoint.z));
        if (dist < closestDistanceSq) {
            closestDistanceSq = dist;
            if (target1) copy(thisVec, target1);
            if (target2) copy(distPoint, target2);
        }
    }
    for (let i = 0; i < 3; i++) {
        copy(self[cornerFields[i]], distLine1.start);
        copy(self[cornerFields[(i + 1) % 3]], distLine1.end);
        for (let i2 = 0; i2 < 3; i2++) {
            copy(other[cornerFields[i2]], distLine2.start);
            copy(other[cornerFields[(i2 + 1) % 3]], distLine2.end);
            closestPointsSegmentToSegment(distLine1, distLine2, distPoint, distPoint2);
            const dist = ((distPoint.x - distPoint2.x) * (distPoint.x - distPoint2.x) + (distPoint.y - distPoint2.y) * (distPoint.y - distPoint2.y) + (distPoint.z - distPoint2.z) * (distPoint.z - distPoint2.z));
            if (dist < closestDistanceSq) {
                closestDistanceSq = dist;
                if (target1) copy(distPoint, target1);
                if (target2) copy(distPoint2, target2);
            }
        }
    }
    return Math.sqrt(closestDistanceSq);
}

/** A fresh state of plain points (needs an update). */
function createExtendedTriangleState(): ExtendedTriangleState {
    const a = new BvhVector3();
    const b = new BvhVector3();
    const c = new BvhVector3();
    return {
        isExtendedTriangle: true,
        a,
        b,
        c,
        points: [a, b, c],
        satAxes: [new BvhVector3(), new BvhVector3(), new BvhVector3(), new BvhVector3()],
        satBounds: [new SeparatingAxisBounds(), new SeparatingAxisBounds(), new SeparatingAxisBounds(), new SeparatingAxisBounds()],
        plane: new BvhPlane(),
        isDegenerateIntoSegment: false,
        isDegenerateIntoPoint: false,
        degenerateSegment: new BvhSegment(),
        needsUpdate: true,
    };
}

/**
 * three-mesh-bvh's `ExtendedTriangle`, engine-free: a triangle with cached
 * separating axes, plane and degeneracy. Set {@link needsUpdate} after moving
 * a corner (the queries update it then).
 */
export class ExtendedTriangle implements BvhQueryTriangle {
    readonly isExtendedTriangle = true as const;
    readonly a: BvhVector3;
    readonly b: BvhVector3;
    readonly c: BvhVector3;
    readonly points: readonly BvhVector3[];
    readonly satAxes: readonly BvhVector3[] = [new BvhVector3(), new BvhVector3(), new BvhVector3(), new BvhVector3()];
    readonly satBounds: readonly SeparatingAxisBounds[] = [new SeparatingAxisBounds(), new SeparatingAxisBounds(), new SeparatingAxisBounds(), new SeparatingAxisBounds()];
    readonly plane = new BvhPlane();
    isDegenerateIntoSegment = false;
    isDegenerateIntoPoint = false;
    readonly degenerateSegment = new BvhSegment();
    needsUpdate = true;

    constructor(a?: BvhVector3Like, b?: BvhVector3Like, c?: BvhVector3Like) {
        this.a = new BvhVector3().copy(a ?? { x: 0, y: 0, z: 0 });
        this.b = new BvhVector3().copy(b ?? { x: 0, y: 0, z: 0 });
        this.c = new BvhVector3().copy(c ?? { x: 0, y: 0, z: 0 });
        this.points = [this.a, this.b, this.c];
    }

    /** Sets the corners (and {@link needsUpdate}). */
    set(a: BvhVector3Like, b: BvhVector3Like, c: BvhVector3Like): this {
        this.a.copy(a);
        this.b.copy(b);
        this.c.copy(c);
        this.needsUpdate = true;
        return this;
    }

    /** Copies another triangle's corners (and sets {@link needsUpdate}). */
    copy(other: BvhTriangleLike): this {
        return this.set(other.a, other.b, other.c);
    }

    update(): void {
        updateExtendedTriangle(this);
    }

    getNormal<T extends BvhVector3Like>(target: T): T {
        triangleGetNormal(this.a, this.b, this.c, target);
        return target;
    }

    containsPoint(point: BvhVector3Like): boolean {
        return triangleContainsPoint(point, this.a, this.b, this.c);
    }

    closestPointToPoint<T extends BvhVector3Like>(point: BvhVector3Like, target: T): T {
        triangleClosestPointToPoint(this.a, this.b, this.c, point, target);
        return target;
    }

    intersectsSphere(sphere: BvhSphereLike): boolean {
        return sphereIntersectsTriangle(sphere, this);
    }

    intersectsTriangle(other: BvhTriangleLike, target: BvhSegmentLike | null = null, suppressLog = false): boolean {
        return triangleIntersectsTriangle(this, other, target, suppressLog);
    }

    closestPointToSegment(segment: BvhSegmentLike, target1: BvhVector3Like | null = null, target2: BvhVector3Like | null = null): number {
        return triangleClosestPointToSegment(this, segment, target1, target2);
    }

    distanceToPoint(point: BvhVector3Like): number {
        return triangleDistanceToPoint(this, point);
    }

    distanceToTriangle(other: BvhTriangleLike, target1: BvhVector3Like | null = null, target2: BvhVector3Like | null = null): number {
        return triangleDistanceToTriangle(this, other, target1, target2);
    }
}

// ---- OrientedBox ----

/** The state an oriented box caches (three-mesh-bvh's `OrientedBox` fields). */
export interface OrientedBoxState {
    readonly isOrientedBox: true;
    readonly min: BvhVector3Like;
    readonly max: BvhVector3Like;
    readonly matrix: BvhMatrix4Target;
    readonly invMatrix: BvhMatrix4Target;
    /** The eight corners, transformed; corner `i` takes x from bit 0, y from bit 1, z from bit 2. */
    readonly points: readonly BvhVector3Like[];
    readonly satAxes: readonly BvhVector3Like[];
    readonly satBounds: readonly SeparatingAxisBounds[];
    readonly alignedSatBounds: readonly SeparatingAxisBounds[];
    needsUpdate: boolean;
}

/** Recomputes a box's corners, axes and inverse (three-mesh-bvh's `OrientedBox.update`). */
export function updateOrientedBox(box: OrientedBoxState): void {
    const { min, max, points } = box;
    const e = box.matrix.elements;
    for (let x = 0; x <= 1; x++) {
        for (let y = 0; y <= 1; y++) {
            for (let z = 0; z <= 1; z++) {
                const v = points[x | (y << 1) | (z << 2)];
                v.x = x ? max.x : min.x;
                v.y = y ? max.y : min.y;
                v.z = z ? max.z : min.z;
                transformPoint(v, e);
            }
        }
    }
    const minVec = points[0];
    for (let i = 0; i < 3; i++) {
        const axis = sub(minVec, points[1 << i], box.satAxes[i]);
        box.satBounds[i].setFromPoints(axis, points);
    }
    box.alignedSatBounds[0].setFromPointsField(points, 'x');
    box.alignedSatBounds[1].setFromPointsField(points, 'y');
    box.alignedSatBounds[2].setFromPointsField(points, 'z');
    invertMatrix(box.matrix, box.invMatrix);
    box.needsUpdate = false;
}

const aabbBounds = new SeparatingAxisBounds();

/** Whether the oriented box touches an axis-aligned box (three-mesh-bvh's `OrientedBox.intersectsBox`). */
export function orientedBoxIntersectsBox(obb: OrientedBoxState, box: BvhBox3Like): boolean {
    if (obb.needsUpdate) updateOrientedBox(obb);
    const { min, max } = box;
    const aligned = obb.alignedSatBounds;
    aabbBounds.min = min.x;
    aabbBounds.max = max.x;
    if (aligned[0].isSeparated(aabbBounds)) return false;
    aabbBounds.min = min.y;
    aabbBounds.max = max.y;
    if (aligned[1].isSeparated(aabbBounds)) return false;
    aabbBounds.min = min.z;
    aabbBounds.max = max.z;
    if (aligned[2].isSeparated(aabbBounds)) return false;
    for (let i = 0; i < 3; i++) {
        aabbBounds.setFromBox(obb.satAxes[i], box);
        if (obb.satBounds[i].isSeparated(aabbBounds)) return false;
    }
    return true;
}

const obbScratchTriangle: ExtendedTriangleState = createExtendedTriangleState();
const obbTrianglePoints: BvhVector3Like[] = [];
const obbAxis = new BvhVector3();

/** Whether the oriented box touches a triangle (three-mesh-bvh's `OrientedBox.intersectsTriangle`). */
export function orientedBoxIntersectsTriangle(obb: OrientedBoxState, triangleLike: BvhTriangleLike): boolean {
    if (obb.needsUpdate) updateOrientedBox(obb);
    const triangle = asExtended(triangleLike, obbScratchTriangle);
    const { satBounds, satAxes, points } = obb;
    obbTrianglePoints[0] = triangle.a;
    obbTrianglePoints[1] = triangle.b;
    obbTrianglePoints[2] = triangle.c;
    for (let i = 0; i < 3; i++) {
        cachedSatBounds.setFromPoints(satAxes[i], obbTrianglePoints);
        if (satBounds[i].isSeparated(cachedSatBounds)) return false;
    }
    for (let i = 0; i < 3; i++) {
        cachedSatBounds.setFromPoints(triangle.satAxes[i], points);
        if (triangle.satBounds[i].isSeparated(cachedSatBounds)) return false;
    }
    for (let i = 0; i < 3; i++) {
        for (let i2 = 0; i2 < 4; i2++) {
            cross(satAxes[i], triangle.satAxes[i2], obbAxis);
            cachedSatBounds.setFromPoints(obbAxis, obbTrianglePoints);
            cachedSatBounds2.setFromPoints(obbAxis, points);
            if (cachedSatBounds.isSeparated(cachedSatBounds2)) return false;
        }
    }
    return true;
}

/** Floats of a box packed by {@link packOrientedBox}: 8 corners, 3 axes, their 3 intervals, the 3 aligned intervals. */
export const PACKED_ORIENTED_BOX_FLOATS = 45;
const PACKED_AXES = 24;
const PACKED_BOUNDS = 33;
const PACKED_ALIGNED = 39;

/**
 * An updated oriented box's corners, axes and intervals as plain numbers
 * (`CpuMeshBvh.intersectsBox` reads them per triangle, with no objects):
 * corners `x, y, z` at 0-23, axes at 24-32, `min, max` per axis at 33-38,
 * the aligned `min, max` per axis at 39-44.
 */
export function packOrientedBox(obb: OrientedBoxState, out: Float64Array): Float64Array {
    if (obb.needsUpdate) updateOrientedBox(obb);
    for (let i = 0; i < 8; i++) {
        const p = obb.points[i];
        out[i * 3] = p.x;
        out[i * 3 + 1] = p.y;
        out[i * 3 + 2] = p.z;
    }
    for (let i = 0; i < 3; i++) {
        const axis = obb.satAxes[i];
        out[PACKED_AXES + i * 3] = axis.x;
        out[PACKED_AXES + i * 3 + 1] = axis.y;
        out[PACKED_AXES + i * 3 + 2] = axis.z;
        out[PACKED_BOUNDS + i * 2] = obb.satBounds[i].min;
        out[PACKED_BOUNDS + i * 2 + 1] = obb.satBounds[i].max;
        out[PACKED_ALIGNED + i * 2] = obb.alignedSatBounds[i].min;
        out[PACKED_ALIGNED + i * 2 + 1] = obb.alignedSatBounds[i].max;
    }
    return out;
}

/**
 * Scratch for {@link packedBoxIntersectsTriangle}: the corners (0-8), the axis
 * being tested (9-11) and the triangle's interval on it (12, 13). The helpers
 * read and write it instead of taking numbers, because a call V8 doesn't
 * inline boxes every number argument (an allocation per call).
 */
const sat = new Float64Array(14);

/** The triangle's interval along `sat`'s axis, as `SeparatingAxisBounds.setFromPoints` finds it (a NaN projection is skipped). */
function triangleRange(s: Float64Array): void {
    const x = s[9], y = s[10], z = s[11];
    let min = Infinity;
    let max = -Infinity;
    let v = x * s[0] + y * s[1] + z * s[2];
    min = v < min ? v : min;
    max = v > max ? v : max;
    v = x * s[3] + y * s[4] + z * s[5];
    min = v < min ? v : min;
    max = v > max ? v : max;
    v = x * s[6] + y * s[7] + z * s[8];
    min = v < min ? v : min;
    max = v > max ? v : max;
    s[12] = min;
    s[13] = max;
}

/** Whether the triangle's interval and the packed box's corners are apart along `sat`'s axis (`SeparatingAxisBounds.isSeparated`, triangle first). */
function separatedFromBox(box: Float64Array, s: Float64Array): boolean {
    triangleRange(s);
    const x = s[9], y = s[10], z = s[11];
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < 24; i += 3) {
        const v = x * box[i] + y * box[i + 1] + z * box[i + 2];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }
    return s[12] > max || min > s[13];
}

/**
 * {@link orientedBoxIntersectsTriangle} on a {@link packOrientedBox} box and a
 * triangle whose corners are `x, y, z` at offsets `a`, `b` and `c` of
 * `corners`: the same separating axes with the same arithmetic, so the same
 * answer bit for bit (NaN included), with no triangle object or cached
 * state. The triangle's normal and edges are worked out here as
 * `updateExtendedTriangle` works them out.
 */
export function packedBoxIntersectsTriangle(box: Float64Array, corners: Float32Array | Float64Array, a: number, b: number, c: number): boolean {
    const s = sat;
    const ax = (s[0] = corners[a]), ay = (s[1] = corners[a + 1]), az = (s[2] = corners[a + 2]);
    const bx = (s[3] = corners[b]), by = (s[4] = corners[b + 1]), bz = (s[5] = corners[b + 2]);
    const cx = (s[6] = corners[c]), cy = (s[7] = corners[c + 1]), cz = (s[8] = corners[c + 2]);
    // The box's own axes against the triangle's corners (the cheap test that turns most triangles away).
    for (let i = 0; i < 3; i++) {
        const axis = PACKED_AXES + i * 3;
        s[9] = box[axis];
        s[10] = box[axis + 1];
        s[11] = box[axis + 2];
        triangleRange(s);
        if (box[PACKED_BOUNDS + i * 2] > s[13] || s[12] > box[PACKED_BOUNDS + i * 2 + 1]) return false;
    }
    // triangleGetNormal: (c - b) x (a - b), normalized, zero when degenerate.
    const ux = cx - bx, uy = cy - by, uz = cz - bz;
    const vx = ax - bx, vy = ay - by, vz = az - bz;
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    const lengthSq = nx * nx + ny * ny + nz * nz;
    if (lengthSq > 0) {
        const scale = 1 / Math.sqrt(lengthSq);
        nx *= scale;
        ny *= scale;
        nz *= scale;
    } else {
        nx = ny = nz = 0;
    }
    // The triangle's axes: the normal, a - b, b - c (and c - a for the cross products).
    const e1x = ax - bx, e1y = ay - by, e1z = az - bz;
    const e2x = bx - cx, e2y = by - cy, e2z = bz - cz;
    const e3x = cx - ax, e3y = cy - ay, e3z = cz - az;
    s[9] = nx;
    s[10] = ny;
    s[11] = nz;
    if (separatedFromBox(box, s)) return false;
    s[9] = e1x;
    s[10] = e1y;
    s[11] = e1z;
    if (separatedFromBox(box, s)) return false;
    s[9] = e2x;
    s[10] = e2y;
    s[11] = e2z;
    if (separatedFromBox(box, s)) return false;
    // cross(box axis, triangle axis) for each pair (three's crossVectors).
    for (let i = 0; i < 3; i++) {
        const axis = PACKED_AXES + i * 3;
        const px = box[axis], py = box[axis + 1], pz = box[axis + 2];
        for (let i2 = 0; i2 < 4; i2++) {
            const tx = i2 === 0 ? nx : i2 === 1 ? e1x : i2 === 2 ? e2x : e3x;
            const ty = i2 === 0 ? ny : i2 === 1 ? e1y : i2 === 2 ? e2y : e3y;
            const tz = i2 === 0 ? nz : i2 === 1 ? e1z : i2 === 2 ? e2z : e3z;
            s[9] = py * tz - pz * ty;
            s[10] = pz * tx - px * tz;
            s[11] = px * ty - py * tx;
            if (separatedFromBox(box, s)) return false;
        }
    }
    return true;
}

/**
 * {@link orientedBoxIntersectsBox} for a packed box that is axis-aligned
 * (made with no matrix, or the identity) with finite corners: only the
 * aligned intervals are tested. The box's own three axes are then
 * `(min - max)` along x, y and z, and a product with a fixed number is
 * monotone, so those three tests can't part what the aligned ones didn't
 * (NaN and infinite node corners part nothing on either). Same answer.
 */
export function alignedPackedBoxIntersectsBox(box: Float64Array, minX: number, minY: number, minZ: number, maxX: number, maxY: number, maxZ: number): boolean {
    return !(
        box[PACKED_ALIGNED] > maxX ||
        minX > box[PACKED_ALIGNED + 1] ||
        box[PACKED_ALIGNED + 2] > maxY ||
        minY > box[PACKED_ALIGNED + 3] ||
        box[PACKED_ALIGNED + 4] > maxZ ||
        minZ > box[PACKED_ALIGNED + 5]
    );
}

/** The point of the oriented box (solid) closest to `point`, into `target` (three-mesh-bvh's `OrientedBox.closestPointToPoint`). */
export function orientedBoxClosestPointToPoint<T extends BvhVector3Like>(obb: OrientedBoxState, point: BvhVector3Like, target: T): T {
    if (obb.needsUpdate) updateOrientedBox(obb);
    copy(point, target);
    transformPoint(target, obb.invMatrix.elements);
    target.x = Math.max(obb.min.x, Math.min(obb.max.x, target.x));
    target.y = Math.max(obb.min.y, Math.min(obb.max.y, target.y));
    target.z = Math.max(obb.min.z, Math.min(obb.max.z, target.z));
    transformPoint(target, obb.matrix.elements);
    return target;
}

const obbPointTarget = new BvhVector3();

/** Distance from the oriented box to a point (three-mesh-bvh's `OrientedBox.distanceToPoint`). */
export function orientedBoxDistanceToPoint(obb: OrientedBoxState, point: BvhVector3Like): number {
    orientedBoxClosestPointToPoint(obb, point, obbPointTarget);
    return Math.sqrt(((point.x - obbPointTarget.x) * (point.x - obbPointTarget.x) + (point.y - obbPointTarget.y) * (point.y - obbPointTarget.y) + (point.z - obbPointTarget.z) * (point.z - obbPointTarget.z)));
}

const boxSegments1 = Array.from({ length: 12 }, () => new BvhSegment());
const boxSegments2 = Array.from({ length: 12 }, () => new BvhSegment());
const boxPoint1 = new BvhVector3();
const boxPoint2 = new BvhVector3();
const xyzFields = ['x', 'y', 'z'] as const;

/**
 * Distance from the oriented box to an axis-aligned box, returning early
 * below `threshold`; `target1` and `target2` get the closest points
 * (three-mesh-bvh's `OrientedBox.distanceToBox`).
 */
export function orientedBoxDistanceToBox(obb: OrientedBoxState, box: BvhBox3Like, threshold = 0, target1: BvhVector3Like | null = null, target2: BvhVector3Like | null = null): number {
    if (obb.needsUpdate) updateOrientedBox(obb);
    const { min, max } = box;
    if (orientedBoxIntersectsBox(obb, box)) {
        if (target1 || target2) {
            // three: box.getCenter(point2); obb.closestPointToPoint(point2, point1); box.closestPointToPoint(point1, point2)
            boxPoint2.x = (min.x + max.x) * 0.5;
            boxPoint2.y = (min.y + max.y) * 0.5;
            boxPoint2.z = (min.z + max.z) * 0.5;
            orientedBoxClosestPointToPoint(obb, boxPoint2, boxPoint1);
            boxPoint2.x = Math.max(min.x, Math.min(max.x, boxPoint1.x));
            boxPoint2.y = Math.max(min.y, Math.min(max.y, boxPoint1.y));
            boxPoint2.z = Math.max(min.z, Math.min(max.z, boxPoint1.z));
            if (target1) copy(boxPoint1, target1);
            if (target2) copy(boxPoint2, target2);
        }
        return 0;
    }
    const threshold2 = threshold * threshold;
    const points = obb.points;
    let closestDistanceSq = Infinity;
    for (let i = 0; i < 8; i++) {
        const p = points[i];
        boxPoint2.x = Math.max(min.x, Math.min(max.x, p.x));
        boxPoint2.y = Math.max(min.y, Math.min(max.y, p.y));
        boxPoint2.z = Math.max(min.z, Math.min(max.z, p.z));
        const dist = ((p.x - boxPoint2.x) * (p.x - boxPoint2.x) + (p.y - boxPoint2.y) * (p.y - boxPoint2.y) + (p.z - boxPoint2.z) * (p.z - boxPoint2.z));
        if (dist < closestDistanceSq) {
            closestDistanceSq = dist;
            if (target1) copy(p, target1);
            if (target2) copy(boxPoint2, target2);
            if (dist < threshold2) return Math.sqrt(dist);
        }
    }
    let count = 0;
    for (let i = 0; i < 3; i++) {
        for (let i1 = 0; i1 <= 1; i1++) {
            for (let i2 = 0; i2 <= 1; i2++) {
                const nextIndex = (i + 1) % 3;
                const nextIndex2 = (i + 2) % 3;
                const index = (i1 << nextIndex) | (i2 << nextIndex2);
                const index2 = (1 << i) | (i1 << nextIndex) | (i2 << nextIndex2);
                const line1 = boxSegments1[count];
                copy(points[index], line1.start);
                copy(points[index2], line1.end);
                const f1 = xyzFields[i];
                const f2 = xyzFields[nextIndex];
                const f3 = xyzFields[nextIndex2];
                const { start, end } = boxSegments2[count];
                start[f1] = min[f1];
                start[f2] = i1 ? min[f2] : max[f2];
                start[f3] = i2 ? min[f3] : max[f3];
                end[f1] = max[f1];
                end[f2] = i1 ? min[f2] : max[f2];
                end[f3] = i2 ? min[f3] : max[f3];
                count++;
            }
        }
    }
    for (let x = 0; x <= 1; x++) {
        for (let y = 0; y <= 1; y++) {
            for (let z = 0; z <= 1; z++) {
                boxPoint2.x = x ? max.x : min.x;
                boxPoint2.y = y ? max.y : min.y;
                boxPoint2.z = z ? max.z : min.z;
                orientedBoxClosestPointToPoint(obb, boxPoint2, boxPoint1);
                const dist = ((boxPoint2.x - boxPoint1.x) * (boxPoint2.x - boxPoint1.x) + (boxPoint2.y - boxPoint1.y) * (boxPoint2.y - boxPoint1.y) + (boxPoint2.z - boxPoint1.z) * (boxPoint2.z - boxPoint1.z));
                if (dist < closestDistanceSq) {
                    closestDistanceSq = dist;
                    if (target1) copy(boxPoint1, target1);
                    if (target2) copy(boxPoint2, target2);
                    if (dist < threshold2) return Math.sqrt(dist);
                }
            }
        }
    }
    for (let i = 0; i < 12; i++) {
        for (let i2 = 0; i2 < 12; i2++) {
            closestPointsSegmentToSegment(boxSegments1[i], boxSegments2[i2], boxPoint1, boxPoint2);
            const dist = ((boxPoint1.x - boxPoint2.x) * (boxPoint1.x - boxPoint2.x) + (boxPoint1.y - boxPoint2.y) * (boxPoint1.y - boxPoint2.y) + (boxPoint1.z - boxPoint2.z) * (boxPoint1.z - boxPoint2.z));
            if (dist < closestDistanceSq) {
                closestDistanceSq = dist;
                if (target1) copy(boxPoint1, target1);
                if (target2) copy(boxPoint2, target2);
                if (dist < threshold2) return Math.sqrt(dist);
            }
        }
    }
    return Math.sqrt(closestDistanceSq);
}

/**
 * three-mesh-bvh's `OrientedBox`, engine-free: the box `min`-`max` placed by
 * `matrix`, with cached corners and separating axes. Set {@link needsUpdate}
 * after changing `min`, `max` or `matrix` ({@link set} does).
 */
export class OrientedBox implements OrientedBoxState {
    readonly isOrientedBox = true as const;
    readonly min = new BvhVector3();
    readonly max = new BvhVector3();
    readonly matrix = new BvhMatrix4();
    readonly invMatrix = new BvhMatrix4();
    readonly points: readonly BvhVector3[] = Array.from({ length: 8 }, () => new BvhVector3());
    readonly satAxes: readonly BvhVector3[] = [new BvhVector3(), new BvhVector3(), new BvhVector3()];
    readonly satBounds: readonly SeparatingAxisBounds[] = [new SeparatingAxisBounds(), new SeparatingAxisBounds(), new SeparatingAxisBounds()];
    readonly alignedSatBounds: readonly SeparatingAxisBounds[] = [new SeparatingAxisBounds(), new SeparatingAxisBounds(), new SeparatingAxisBounds()];
    needsUpdate = false;

    constructor(min?: BvhVector3Like, max?: BvhVector3Like, matrix?: BvhMatrix4Like) {
        if (min) this.min.copy(min);
        if (max) this.max.copy(max);
        if (matrix) this.matrix.copy(matrix);
        // three-mesh-bvh leaves needsUpdate false here; a box made with arguments is updated before use instead.
        if (min || max || matrix) this.needsUpdate = true;
    }

    set(min: BvhVector3Like, max: BvhVector3Like, matrix?: BvhMatrix4Like | null): this {
        this.min.copy(min);
        this.max.copy(max);
        copyMatrix(matrix, this.matrix);
        this.needsUpdate = true;
        return this;
    }

    copy(other: OrientedBoxState): this {
        return this.set(other.min, other.max, other.matrix);
    }

    update(): void {
        updateOrientedBox(this);
    }

    intersectsBox(box: BvhBox3Like): boolean {
        return orientedBoxIntersectsBox(this, box);
    }

    intersectsTriangle(triangle: BvhTriangleLike): boolean {
        return orientedBoxIntersectsTriangle(this, triangle);
    }

    closestPointToPoint<T extends BvhVector3Like>(point: BvhVector3Like, target: T): T {
        return orientedBoxClosestPointToPoint(this, point, target);
    }

    distanceToPoint(point: BvhVector3Like): number {
        return orientedBoxDistanceToPoint(this, point);
    }

    distanceToBox(box: BvhBox3Like, threshold = 0, target1: BvhVector3Like | null = null, target2: BvhVector3Like | null = null): number {
        return orientedBoxDistanceToBox(this, box, threshold, target1, target2);
    }
}

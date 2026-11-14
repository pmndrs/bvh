/*!
 * The selection tools and triangle selection of three-mesh-bvh's `selection`
 * example, ported to TypeScript over our tree: example/src/Selection.js,
 * example/src/computeSelectedTriangles.js and example/utils/math/
 * (getConvexHull.js, lineCrossesLine.js, pointRayCrossesSegments.js) of
 * gkjohnson/three-mesh-bvh v0.9.18. See THIRD_PARTY.md.
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
 */

import { DoubleSide, Line3, Matrix4, Ray, Vector2, Vector3, type Box3, type Camera, type Mesh } from 'three';
import { CONTAINED, INTERSECTED, NOT_INTERSECTED, type BvhBoundsTree } from '@pmndrs/bvh/three';

// ---- The pointer tools (Selection.js).

/** A selection drawn with the pointer: its points as x, y, 0 triplets in normalized device coordinates. */
export abstract class SelectionTool {
    dragging = false;
    abstract get points(): number[];
    handlePointerDown(_event: PointerEvent): void {
        this.dragging = true;
    }
    handlePointerUp(): void {
        this.dragging = false;
    }
    abstract handlePointerMove(event: PointerEvent): { changed: boolean };
    /** Screen coordinates to -1..1 over the element. */
    static normalizePoint(element: Element, x: number, y: number): [number, number] {
        const rect = element.getBoundingClientRect();
        return [((x - rect.left) / rect.width) * 2 - 1, 1 - ((y - rect.top) / rect.height) * 2];
    }
}

const v0 = new Vector2();
const v1 = new Vector2();
const v2 = new Vector2();

/** Adds a point every few pixels of drag (replacing the last when it goes on in the same direction). */
export class LassoSelection extends SelectionTool {
    lassoPoints: number[] = [];
    private prevX = -Infinity;
    private prevY = -Infinity;

    override handlePointerDown(event: PointerEvent): void {
        super.handlePointerDown(event);
        this.prevX = event.clientX;
        this.prevY = event.clientY;
        this.lassoPoints = [];
    }

    handlePointerMove(event: PointerEvent): { changed: boolean } {
        const ex = event.clientX;
        const ey = event.clientY;
        const [nx, ny] = SelectionTool.normalizePoint(event.currentTarget as Element, ex, ey);
        if (Math.abs(ex - this.prevX) < 3 && Math.abs(ey - this.prevY) < 3) return { changed: false };
        const i3 = (this.lassoPoints.length / 3 - 1) * 3;
        let replace = false;
        if (this.lassoPoints.length > 3) {
            v0.set(this.lassoPoints[i3 - 3], this.lassoPoints[i3 - 2]);
            v1.set(this.lassoPoints[i3], this.lassoPoints[i3 + 1]).sub(v0).normalize();
            v0.set(this.lassoPoints[i3], this.lassoPoints[i3 + 1]);
            v2.set(nx, ny).sub(v0).normalize();
            replace = v1.dot(v2) > 0.99;
        }
        if (replace) {
            this.lassoPoints[i3] = nx;
            this.lassoPoints[i3 + 1] = ny;
        } else this.lassoPoints.push(nx, ny, 0);
        this.prevX = ex;
        this.prevY = ey;
        return { changed: true };
    }

    get points(): number[] {
        return this.lassoPoints;
    }
}

/** A screen-aligned rectangle from where the drag started to the pointer. */
export class BoxSelection extends SelectionTool {
    startX = 0;
    startY = 0;
    currentX = 0;
    currentY = 0;
    private prevX = 0;
    private prevY = 0;

    override handlePointerDown(event: PointerEvent): void {
        super.handlePointerDown(event);
        this.prevX = event.clientX;
        this.prevY = event.clientY;
        [this.startX, this.startY] = SelectionTool.normalizePoint(event.currentTarget as Element, event.clientX, event.clientY);
        this.currentX = this.startX;
        this.currentY = this.startY;
    }

    handlePointerMove(event: PointerEvent): { changed: boolean } {
        [this.currentX, this.currentY] = SelectionTool.normalizePoint(event.currentTarget as Element, event.clientX, event.clientY);
        if (event.clientX === this.prevX && event.clientY === this.prevY) return { changed: false };
        this.prevX = event.clientX;
        this.prevY = event.clientY;
        return { changed: true };
    }

    get points(): number[] {
        return [this.startX, this.startY, 0, this.currentX, this.startY, 0, this.currentX, this.currentY, 0, this.startX, this.currentY, 0];
    }
}

// ---- 2D geometry (utils/math).

/** The convex hull of points (Graham scan), or null below three distinct directions. Reorders `points`. */
function getConvexHull(points: Vector3[]): Vector3[] | null {
    const orientation = (p: Vector3, q: Vector3, r: Vector3) => {
        const value = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
        return value === 0 ? 0 : value > 0 ? 1 : 2;
    };
    const distSq = (a: Vector3, b: Vector3) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
    let lowest = -1;
    let lowestY = Infinity;
    points.forEach((p, i) => {
        if (p.y < lowestY) {
            lowest = i;
            lowestY = p.y;
        }
    });
    const p0 = points[lowest];
    points[lowest] = points[0];
    points[0] = p0;
    points = points.sort((a, b) => {
        const o = orientation(p0, a, b);
        if (o === 0) return distSq(p0, b) >= distSq(p0, a) ? -1 : 1;
        return o === 2 ? -1 : 1;
    });
    let m = 1;
    const n = points.length;
    for (let i = 1; i < n; i++) {
        while (i < n - 1 && orientation(p0, points[i], points[i + 1]) === 0) i++;
        points[m++] = points[i];
    }
    if (m < 3) return null;
    const hull = [points[0], points[1], points[2]];
    for (let i = 3; i < m; i++) {
        while (orientation(hull[hull.length - 2], hull[hull.length - 1], points[i]) !== 2) hull.pop();
        hull.push(points[i]);
    }
    return hull;
}

/** Whether two segments cross. */
function lineCrossesLine(l1: Line3, l2: Line3): boolean {
    const ccw = (a: Vector3, b: Vector3, c: Vector3) => (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x);
    const { start: a, end: b } = l1;
    const { start: c, end: d } = l2;
    return ccw(a, c, d) !== ccw(b, c, d) && ccw(a, b, c) !== ccw(a, b, d);
}

/** Whether a ray from `point` to the right crosses `line` (the peak rule as the original's). */
function pointRayCrossesLine(point: Vector3, line: Line3, prevSegmentGoesDown: boolean, thisSegmentGoesDown: boolean): boolean {
    const { start, end } = line;
    const px = point.x;
    const py = point.y;
    const sy = start.y;
    const ey = end.y;
    if (sy === ey) return false;
    if (py > sy && py > ey) return false;
    if (py < sy && py < ey) return false;
    const sx = start.x;
    const ex = end.x;
    if (px > sx && px > ex) return false;
    if (px < sx && px < ex) return !(py === sy && prevSegmentGoesDown !== thisSegmentGoesDown);
    const perpx = ey - sy;
    const perpy = -(ex - sx);
    const dot = perpx * (px - sx) + perpy * (py - sy);
    return Math.sign(dot) !== Math.sign(perpx);
}

/** Point in polygon by the crossing count. */
function isPointInsidePolygon(point: Vector3, polygon: readonly Line3[]): boolean {
    let crossings = 0;
    const first = polygon[polygon.length - 1];
    let prevGoesDown = first.start.y > first.end.y;
    for (const line of polygon) {
        const goesDown = line.start.y > line.end.y;
        if (pointRayCrossesLine(point, line, prevGoesDown, goesDown)) crossings++;
        prevGoesDown = goesDown;
    }
    return crossings % 2 === 1;
}

// ---- computeSelectedTriangles.

export interface SelectionParams {
    readonly selectionMode: 'centroid' | 'centroid-visible' | 'intersection';
    readonly selectWholeModel: boolean;
    readonly useBoundsTree: boolean;
}

const invWorldMatrix = new Matrix4();
const camLocalPosition = new Vector3();
const tempRay = new Ray();
const centroid = new Vector3();
const screenCentroid = new Vector3();
const faceNormal = new Vector3();
const toScreenSpaceMatrix = new Matrix4();
const boxPoints = Array.from({ length: 8 }, () => new Vector3());
const boxLines = Array.from({ length: 12 }, () => new Line3());

function extractBoxVertices(box: Box3, target: Vector3[]): Vector3[] {
    const { min, max } = box;
    let index = 0;
    for (let x = 0; x <= 1; x++) for (let y = 0; y <= 1; y++) for (let z = 0; z <= 1; z++) target[index++].set(x === 0 ? min.x : max.x, y === 0 ? min.y : max.y, z === 0 ? min.z : max.z);
    return target;
}

/** Whether the segment is not wholly left of, above or below the box's screen extent. */
function isSegmentToTheRight(segment: Line3, minX: number, minY: number, maxY: number): boolean {
    const { x: sx, y: sy } = segment.start;
    const { x: ex, y: ey } = segment.end;
    if (sx < minX && ex < minX) return false;
    if (sy > maxY && ey > maxY) return false;
    if (sy < minY && ey < minY) return false;
    return true;
}

function connectPointsWithLines(points: readonly Vector3[], target: Line3[] | null = null): Line3[] {
    const lines = target ?? points.map(() => new Line3());
    return points.map((p, i) => lines[i].set(p, points[(i + 1) % points.length]));
}

function convertTripletsToPoints(array: readonly number[]): Vector3[] {
    const points: Vector3[] = [];
    for (let i = 0; i < array.length; i += 3) points.push(new Vector3(array[i], array[i + 1], array[i + 2]));
    return points;
}

/**
 * The triangles under the selection: index-buffer positions (three per
 * triangle) of the selected triangles. The tree's `shapecast` narrows the
 * search: each box is projected to the screen, its convex hull tested against
 * the lasso (`INTERSECTED`, `CONTAINED` when the hull lies inside it, else
 * `NOT_INTERSECTED`), and each depth keeps only the lasso segments that can
 * still matter for the boxes below it.
 */
export function computeSelectedTriangles(mesh: Mesh, tree: BvhBoundsTree, camera: Camera, tool: SelectionTool, params: SelectionParams): number[] {
    toScreenSpaceMatrix.copy(mesh.matrixWorld).premultiply(camera.matrixWorldInverse).premultiply(camera.projectionMatrix);
    invWorldMatrix.copy(mesh.matrixWorld).invert();
    camLocalPosition.set(0, 0, 0).applyMatrix4(camera.matrixWorld).applyMatrix4(invWorldMatrix);
    const lassoSegments = connectPointsWithLines(convertTripletsToPoints(tool.points));
    const perBoundsSegmentCache: Line3[][] = [];
    const indices: number[] = [];
    tree.shapecast({
        intersectsBounds: (box, _isLeaf, _score, depth) => {
            if (!params.useBoundsTree) return INTERSECTED;
            const projected = extractBoxVertices(box, boxPoints).map((v) => v.applyMatrix4(toScreenSpaceMatrix));
            let minY = Infinity;
            let maxY = -Infinity;
            let minX = Infinity;
            for (const point of projected) {
                if (point.y < minY) minY = point.y;
                if (point.y > maxY) maxY = point.y;
                if (point.x < minX) minX = point.x;
            }
            // Segments wholly left of, above or below the box can't matter to it or below (the crossing test casts rays to the right).
            const parentSegments = perBoundsSegmentCache[depth - 1] ?? lassoSegments;
            const segmentsToCheck = parentSegments.filter((segment) => isSegmentToTheRight(segment, minX, minY, maxY));
            perBoundsSegmentCache[depth] = segmentsToCheck;
            if (segmentsToCheck.length === 0) return NOT_INTERSECTED;
            const hull = getConvexHull(projected);
            if (!hull) return INTERSECTED;
            const hullSegments = connectPointsWithLines(hull, boxLines);
            if (isPointInsidePolygon(segmentsToCheck[0].start, hullSegments)) return INTERSECTED;
            for (const hullSegment of hullSegments) for (const segment of segmentsToCheck) if (lineCrossesLine(hullSegment, segment)) return INTERSECTED;
            return isPointInsidePolygon(hull[0], segmentsToCheck) ? CONTAINED : NOT_INTERSECTED;
        },
        intersectsTriangle: (triangle, index, contained, depth) => {
            const i3 = index * 3;
            const segmentsToCheck = (params.useBoundsTree ? perBoundsSegmentCache[depth] : undefined) ?? lassoSegments;
            const select = () => {
                indices.push(i3, i3 + 1, i3 + 2);
                return params.selectWholeModel;
            };
            if (params.selectionMode === 'centroid' || params.selectionMode === 'centroid-visible') {
                centroid.copy(triangle.a).add(triangle.b).add(triangle.c).multiplyScalar(1 / 3);
                screenCentroid.copy(centroid).applyMatrix4(toScreenSpaceMatrix);
                if (contained || isPointInsidePolygon(screenCentroid, segmentsToCheck)) {
                    if (params.selectionMode === 'centroid-visible') {
                        // Visible: nothing between the centroid and the camera.
                        triangle.getNormal(faceNormal);
                        tempRay.origin.copy(centroid).addScaledVector(faceNormal, 1e-6);
                        tempRay.direction.subVectors(camLocalPosition, centroid);
                        if (tree.raycastFirst(tempRay, DoubleSide)) return false;
                    }
                    return select();
                }
                return false;
            }
            if (contained) return select();
            const projected = [triangle.a, triangle.b, triangle.c].map((v) => v.applyMatrix4(toScreenSpaceMatrix));
            for (const point of projected) if (isPointInsidePolygon(point, segmentsToCheck)) return select();
            for (const segment of connectPointsWithLines(projected, boxLines)) for (const selectionSegment of segmentsToCheck) if (lineCrossesLine(segment, selectionSegment)) return select();
            return false;
        },
    });
    return indices;
}

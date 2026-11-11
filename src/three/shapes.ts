/**
 * three-mesh-bvh's `ExtendedTriangle` and `OrientedBox` with three's types,
 * so code written against three-mesh-bvh 0.9.18 (three-bvh-csg's splitters,
 * a character controller's `closestPointToSegment`) compiles and runs with an
 * import change. `ExtendedTriangle` extends three's `Triangle`; `OrientedBox`
 * is its own class with `Vector3` / `Matrix4` fields, as three-mesh-bvh's is.
 *
 * No math lives here: the cached state is held in three objects (which have
 * the `x`, `y`, `z` / `start`, `end` / `normal`, `constant` / `elements`
 * fields the core reads), and every method calls the core's function
 * (`src/query/shapes.ts`).
 */

import { Line3, Matrix4, Plane, Triangle, Vector3, type Box3, type Sphere } from 'three';
import {
    copyMatrix,
    orientedBoxClosestPointToPoint,
    orientedBoxDistanceToBox,
    orientedBoxDistanceToPoint,
    orientedBoxIntersectsBox,
    orientedBoxIntersectsTriangle,
    SeparatingAxisBounds,
    sphereIntersectsTriangle,
    triangleClosestPointToSegment,
    triangleDistanceToPoint,
    triangleDistanceToTriangle,
    triangleIntersectsTriangle,
    updateExtendedTriangle,
    updateOrientedBox,
    type ExtendedTriangleState,
    type OrientedBoxState,
} from '../query/shapes.ts';

export { SeparatingAxisBounds };

/**
 * three-mesh-bvh's `ExtendedTriangle`: a three `Triangle` with cached
 * separating axes, plane and degeneracy. Set `needsUpdate` after moving `a`,
 * `b` or `c` (`set` and `copy` do; three's inherited `setFromPointsAndIndices`
 * and `setFromAttributeAndIndices` don't, as in three-mesh-bvh).
 */
export class ExtendedTriangle extends Triangle implements ExtendedTriangleState {
    readonly isExtendedTriangle = true as const;
    readonly points: Vector3[];
    readonly satAxes: Vector3[] = [new Vector3(), new Vector3(), new Vector3(), new Vector3()];
    readonly satBounds: SeparatingAxisBounds[] = [new SeparatingAxisBounds(), new SeparatingAxisBounds(), new SeparatingAxisBounds(), new SeparatingAxisBounds()];
    readonly plane = new Plane();
    isDegenerateIntoSegment = false;
    isDegenerateIntoPoint = false;
    readonly degenerateSegment = new Line3();
    needsUpdate = true;

    constructor(a?: Vector3, b?: Vector3, c?: Vector3) {
        super(a, b, c);
        this.points = [this.a, this.b, this.c];
    }

    override set(a: Vector3, b: Vector3, c: Vector3): this {
        super.set(a, b, c);
        this.needsUpdate = true;
        return this;
    }

    override copy(triangle: Triangle): this {
        super.copy(triangle);
        this.needsUpdate = true;
        return this;
    }

    update(): void {
        updateExtendedTriangle(this);
    }

    intersectsSphere(sphere: Sphere): boolean {
        return sphereIntersectsTriangle(sphere, this);
    }

    /** Whether the triangles intersect; `target` gets the intersection segment (zero for coplanar ones, with a warning unless `suppressLog`). */
    intersectsTriangle(other: Triangle, target: Line3 | null = null, suppressLog = false): boolean {
        return triangleIntersectsTriangle(this, other, target, suppressLog);
    }

    /** Distance to the segment; `target1` / `target2` get the closest points on the triangle / the segment. */
    closestPointToSegment(segment: Line3, target1: Vector3 | null = null, target2: Vector3 | null = null): number {
        return triangleClosestPointToSegment(this, segment, target1, target2);
    }

    distanceToPoint(point: Vector3): number {
        return triangleDistanceToPoint(this, point);
    }

    /** Distance to the triangle; `target1` / `target2` get the closest points on this one / the other. */
    distanceToTriangle(other: Triangle, target1: Vector3 | null = null, target2: Vector3 | null = null): number {
        return triangleDistanceToTriangle(this, other, target1, target2);
    }
}

/**
 * three-mesh-bvh's `OrientedBox`: the box `min`-`max` placed by `matrix`.
 * Set `needsUpdate` after changing `min`, `max` or `matrix` (`set` does).
 */
export class OrientedBox implements OrientedBoxState {
    readonly isOrientedBox = true as const;
    readonly min = new Vector3();
    readonly max = new Vector3();
    readonly matrix = new Matrix4();
    readonly invMatrix = new Matrix4();
    readonly points: Vector3[] = Array.from({ length: 8 }, () => new Vector3());
    readonly satAxes: Vector3[] = [new Vector3(), new Vector3(), new Vector3()];
    readonly satBounds: SeparatingAxisBounds[] = [new SeparatingAxisBounds(), new SeparatingAxisBounds(), new SeparatingAxisBounds()];
    readonly alignedSatBounds: SeparatingAxisBounds[] = [new SeparatingAxisBounds(), new SeparatingAxisBounds(), new SeparatingAxisBounds()];
    needsUpdate = false;

    constructor(min?: Vector3, max?: Vector3, matrix?: Matrix4) {
        if (min) this.min.copy(min);
        if (max) this.max.copy(max);
        if (matrix) this.matrix.copy(matrix);
        if (min || max || matrix) this.needsUpdate = true;
    }

    set(min: Vector3, max: Vector3, matrix: Matrix4): this {
        this.min.copy(min);
        this.max.copy(max);
        copyMatrix(matrix, this.matrix);
        this.needsUpdate = true;
        return this;
    }

    copy(other: OrientedBox): this {
        return this.set(other.min, other.max, other.matrix);
    }

    update(): void {
        updateOrientedBox(this);
    }

    intersectsBox(box: Box3): boolean {
        return orientedBoxIntersectsBox(this, box);
    }

    intersectsTriangle(triangle: Triangle): boolean {
        return orientedBoxIntersectsTriangle(this, triangle);
    }

    /** The point of the (solid) box closest to `point`, into `target`. */
    closestPointToPoint(point: Vector3, target: Vector3 = new Vector3()): Vector3 {
        return orientedBoxClosestPointToPoint(this, point, target);
    }

    distanceToPoint(point: Vector3): number {
        return orientedBoxDistanceToPoint(this, point);
    }

    distanceToBox(box: Box3, threshold = 0, target1: Vector3 | null = null, target2: Vector3 | null = null): number {
        return orientedBoxDistanceToBox(this, box, threshold, target1, target2);
    }
}

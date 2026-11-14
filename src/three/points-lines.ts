/*!
 * Points and lines raycast over a tree: `raycastObject3D` follows three.js
 * r187's `Points.raycast` and `Line.raycast` (src/objects/Points.js,
 * src/objects/Line.js, mrdoob/three.js) step for step. See THIRD_PARTY.md.
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
 * Trees for `Points`, `LineSegments`, `LineLoop` and `Line` (PARITY 3.2,
 * 3.3): three-mesh-bvh's `PointsBVH`, `LineSegmentsBVH`, `LineLoopBVH` and
 * `LineBVH`, over the core's `CpuPointsBvh` and `CpuSegmentsBvh`.
 *
 * ```ts
 * import { acceleratedRaycast, computeBoundsTree, PointsBVH } from '@pmndrs/bvh/three';
 * BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
 * Points.prototype.raycast = acceleratedRaycast;
 * points.geometry.computeBoundsTree({ type: PointsBVH });
 * raycaster.params.Points.threshold = 0.01;
 * raycaster.intersectObject(points);         // three's answer, from the tree
 * ```
 *
 * **What matches three's raycast.** {@link BvhPointsBoundsTree.raycastObject3D}
 * and the line trees' do what three's `Points.raycast` and `Line.raycast` do,
 * in the same order: the bounding-sphere test (the sphere grown by the world
 * threshold), the ray in the object's space, the threshold over the object's
 * mean `scale`, the draw range read at each raycast (a `LineSegments` range
 * from an odd entry walks a second tree, made on first use; a `LineLoop`'s
 * closing segment for the range is tested on its own), `near` and `far` on
 * the world distance, and the same fields: `distance`, `distanceToRay`
 * (points), `point`, `index` (the vertex for points, the segment's first entry
 * for lines), `face`, `faceIndex` and `barycoord` null, `object`. Hits are
 * pushed in three's order, so after `Raycaster`'s stable sort the list is
 * three's. `raycaster.firstHitOnly` keeps only the closest, as three-mesh-bvh.
 *
 * **Where three-mesh-bvh differs** (and we follow three): it skips the
 * bounding-sphere test, builds over the draw range and groups at build time
 * (three's raycast ignores groups and reads the draw range when it runs; a
 * `LineSegments` range from an odd entry and a `LineLoop` range not starting
 * at 0 come out wrong there), and outside indirect mode reorders (or adds) the
 * geometry's index, so its line `index` changes meaning. These trees never
 * touch the geometry: `indirect` is accepted and ignored, as for meshes.
 *
 * **Staleness.** As `BvhBoundsTree`: when the position attribute's version
 * or the index changed since the build, raycasts use three's own (with a
 * one-time warning) until `refit()` or `computeBoundsTree` runs again. A tree
 * of the wrong kind for the object (a `PointsBVH` on a `Line`) does too.
 */

import {
    Box3,
    type BufferAttribute,
    Line,
    Line3,
    Matrix4,
    Points,
    Ray,
    Sphere,
    Vector3,
    type BufferGeometry,
    type Intersection,
    type InterleavedBufferAttribute,
    type Object3D,
    type Raycaster,
} from 'three';
import { clampLeafPrimitives } from '../query/cpu-mesh-bvh.ts';
import {
    buildPrimitiveLayout,
    CpuPointsBvh,
    type CpuPrimitiveBvh,
    CpuSegmentsBvh,
    raySegmentHit,
    type BvhPointsInput,
    type BvhPointsShapecastCallbacks,
    type BvhPrimitiveBuildOptions,
    type BvhPrimitiveClosestPoint,
    type BvhPrimitiveLayout,
    type BvhPrimitiveRaycastHit,
    type BvhSegmentMode,
    type BvhSegmentsInput,
    type BvhSegmentsShapecastCallbacks,
} from '../query/primitive-bvh.ts';
import { positionsOf } from './scene.ts';

/**
 * Options of `computeBoundsTree({ type: PointsBVH, ... })` and the
 * constructors: three-mesh-bvh's, mapped where they map.
 */
export interface BvhPrimitiveBoundsTreeOptions {
    /**
     * Accepted for three-mesh-bvh code (0, 1, 2); every value builds the same
     * tree, a centre split (three-mesh-bvh's default `CENTER`). Other values throw.
     */
    readonly strategy?: number;
    /** Largest leaf, in primitives: 1 to 8 (10, three-mesh-bvh's default, gives 8). Default 8. */
    readonly maxLeafSize?: number;
    /** three-mesh-bvh 0.9's name for {@link maxLeafSize}; wins over the other two. */
    readonly targetLeafSize?: number;
    /** three-mesh-bvh's deprecated name for {@link maxLeafSize}. */
    readonly maxLeafTris?: number;
    /** Accepted, not honoured: past depth 40 the builder splits at the median, which keeps trees shallow. */
    readonly maxDepth?: number;
    /** Sets `geometry.boundingBox` from the tree when the geometry has none. Default true. */
    readonly setBoundingBox?: boolean;
    /** Accepted and ignored: the geometry's index is never touched (each record carries its entry). */
    readonly indirect?: boolean;
    /** Called with 0 before the build and 1 after it. */
    readonly onProgress?: (progress: number) => void;
    /** Accepted and ignored. */
    readonly verbose?: boolean;
    /** Accepted and ignored. */
    readonly useSharedArrayBuffer?: boolean;
    /** Not supported: the tree holds every entry, and raycasts follow the draw range. Throws when set. */
    readonly range?: unknown;
    /** A tree already built for this geometry (`CpuPointsBvh.deserialize(bytes).layout`): the build is skipped. */
    readonly layout?: BvhPrimitiveLayout;
}

function buildOptionsOf(options: BvhPrimitiveBoundsTreeOptions): BvhPrimitiveBuildOptions {
    if (options.strategy !== undefined && options.strategy !== 0 && options.strategy !== 1 && options.strategy !== 2) {
        throw new RangeError(`@pmndrs/bvh/three: unknown strategy ${String(options.strategy)} (CENTER 0, AVERAGE 1, SAH 2)`);
    }
    if (options.range !== undefined && options.range !== null) {
        throw new Error('@pmndrs/bvh/three: the `range` option is not supported; build over the whole geometry (raycasts follow its draw range)');
    }
    const leaf = options.targetLeafSize ?? options.maxLeafSize ?? options.maxLeafTris;
    return {
        ...(leaf !== undefined ? { maxLeafPrimitives: clampLeafPrimitives(leaf) } : {}),
        ...(options.onProgress ? { onProgress: options.onProgress } : {}),
    };
}

/** A position attribute's version (an interleaved one's buffer's). */
function versionOf(attribute: BufferAttribute | InterleavedBufferAttribute): number {
    const interleaved = attribute as InterleavedBufferAttribute;
    return interleaved.isInterleavedBufferAttribute === true ? interleaved.data.version : (attribute as BufferAttribute).version;
}

/** Every entry of a geometry, as the core takes them: the positions as f32 and the whole index. */
export function pointsInputFromGeometry(geometry: BufferGeometry): BvhPointsInput {
    const position = geometry.getAttribute('position') as BufferAttribute | InterleavedBufferAttribute | undefined;
    if (!position) throw new Error('@pmndrs/bvh/three: a bounds tree needs a position attribute');
    const index = geometry.index;
    let indices: Uint16Array | Uint32Array | null = null;
    if (index) {
        const array = index.array;
        indices = array instanceof Uint16Array || array instanceof Uint32Array ? array.subarray(0, index.count) : Uint32Array.from({ length: index.count }, (_, i) => index.getX(i));
    }
    return { positions: positionsOf(position), indices };
}

/** The geometry, its versions and the box this tree set: the part every /three primitive tree keeps. */
class GeometryState {
    readonly position: BufferAttribute | InterleavedBufferAttribute;
    positionVersion: number;
    readonly index: BufferAttribute | null;
    readonly indexVersion: number;
    staleWarned = false;
    ownBoundingBox: Box3 | null = null;

    constructor(readonly geometry: BufferGeometry) {
        this.position = geometry.getAttribute('position') as BufferAttribute | InterleavedBufferAttribute;
        this.positionVersion = versionOf(this.position);
        this.index = geometry.index;
        this.indexVersion = geometry.index ? geometry.index.version : 0;
    }

    current(): boolean {
        const { geometry } = this;
        return geometry.getAttribute('position') === this.position && versionOf(this.position) === this.positionVersion && geometry.index === this.index && (this.index ? this.index.version : 0) === this.indexVersion;
    }

    /** True (after one warning) when the tree can't answer for `object`: the geometry changed, or the tree is of another kind. */
    refuses(object: Object3D, fits: boolean): boolean {
        if (fits && this.current()) return false;
        if (!this.staleWarned) {
            this.staleWarned = true;
            console.warn(
                fits
                    ? "@pmndrs/bvh/three: a geometry's positions or index changed since computeBoundsTree; raycasts against it use three's own raycast until boundsTree.refit() or computeBoundsTree runs again"
                    : `@pmndrs/bvh/three: the geometry's bounds tree is not for a ${object.type} (build it with the matching type: PointsBVH, LineSegmentsBVH, LineLoopBVH or LineBVH); three's own raycast is used`,
            );
        }
        return true;
    }

    /** Checks a refit's geometry and takes its versions; returns its entries. */
    refitInput(where: string): BvhPointsInput {
        const { geometry } = this;
        if (geometry.getAttribute('position') !== this.position || geometry.index !== this.index || (this.index ? this.index.version : 0) !== this.indexVersion) {
            throw new Error(`${where}: the position attribute or the index was replaced or changed; call computeBoundsTree again`);
        }
        this.positionVersion = versionOf(this.position);
        this.staleWarned = false;
        return pointsInputFromGeometry(geometry);
    }

    setBoundingBox(tree: CpuPrimitiveBvh, options: BvhPrimitiveBoundsTreeOptions): void {
        if ((options.setBoundingBox ?? true) && !this.geometry.boundingBox) this.geometry.boundingBox = this.ownBoundingBox = tree.getBoundingBox(new Box3());
    }

    updateBoundingBox(tree: CpuPrimitiveBvh): void {
        if (this.ownBoundingBox && this.geometry.boundingBox === this.ownBoundingBox) tree.getBoundingBox(this.ownBoundingBox);
    }
}

// three's own raycasts, captured before anyone replaces them.
const threePointsRaycast = Points.prototype.raycast;
const threeLineRaycast = Line.prototype.raycast;

// Scratch, as three's module-level objects.
const inverseMatrix = new Matrix4();
const localRay = new Ray();
const sphere = new Sphere();
const worldPoint = new Vector3();
const vStart = new Vector3();
const vEnd = new Vector3();

/** three's own raycast for the object: `Points.raycast`, or `Line.raycast` (which `LineSegments` and `LineLoop` inherit). */
function threeRaycast(object: Object3D, raycaster: Raycaster, intersects: Intersection[]): void {
    if ((object as Object3D as { isPoints?: boolean }).isPoints === true) threePointsRaycast.call(object as Points, raycaster, intersects);
    else threeLineRaycast.call(object as Line, raycaster, intersects);
}

/**
 * The closest hit by world distance, as three's sorted list has it first (ties: the earlier entry).
 * The tree's closest is by distance in the object's space, whose order can differ from the world's
 * in the last bit; so every hit within a hair of that one is ranked by world distance.
 */
function closestByWorld(tree: CpuPrimitiveBvh, ray: { origin: number[]; direction: number[] }, options: { threshold: number; accept: (hit: BvhPrimitiveRaycastHit) => boolean }, raycaster: Raycaster, matrixWorld: Matrix4): BvhPrimitiveRaycastHit[] {
    const first = tree.raycastFirst(ray, options);
    if (!first) return [];
    const ties = tree.raycast(ray, { ...options, far: first.distance * (1 + 1e-9) + 1e-12 });
    let best = first;
    let bestDistance = worldDistance(raycaster, matrixWorld, first.pointOnRay);
    for (const hit of ties) {
        const distance = worldDistance(raycaster, matrixWorld, hit.pointOnRay);
        if (distance < bestDistance || (distance === bestDistance && hit.index < best.index)) {
            best = hit;
            bestDistance = distance;
        }
    }
    return [best];
}

/**
 * The prelude of three's `Points.raycast` and `Line.raycast`: the
 * bounding-sphere test, then the ray in the object's space and the local
 * threshold. Null when the sphere is missed.
 */
function localQuery(object: Object3D & { geometry: BufferGeometry }, raycaster: Raycaster, threshold: number): { origin: [number, number, number]; direction: [number, number, number]; threshold: number } | null {
    const geometry = object.geometry;
    const matrixWorld = object.matrixWorld;
    if (geometry.boundingSphere === null) geometry.computeBoundingSphere();
    sphere.copy(geometry.boundingSphere!);
    sphere.applyMatrix4(matrixWorld);
    sphere.radius += threshold;
    if (raycaster.ray.intersectsSphere(sphere) === false) return null;
    inverseMatrix.copy(matrixWorld).invert();
    localRay.copy(raycaster.ray).applyMatrix4(inverseMatrix);
    const localThreshold = threshold / ((object.scale.x + object.scale.y + object.scale.z) / 3);
    return {
        origin: [localRay.origin.x, localRay.origin.y, localRay.origin.z],
        direction: [localRay.direction.x, localRay.direction.y, localRay.direction.z],
        threshold: localThreshold,
    };
}

/** three's `[start, end)` over a geometry's entries for its draw range. */
function drawEntries(geometry: BufferGeometry): [number, number] {
    const drawRange = geometry.drawRange;
    const length = geometry.index ? geometry.index.count : geometry.getAttribute('position').count;
    return [Math.max(0, drawRange.start), Math.min(length, drawRange.start + drawRange.count)];
}

/** `raycaster.ray.origin.distanceTo(point.applyMatrix4(matrixWorld))`, into {@link worldPoint}. */
function worldDistance(raycaster: Raycaster, matrixWorld: Matrix4, local: readonly number[]): number {
    worldPoint.set(local[0], local[1], local[2]).applyMatrix4(matrixWorld);
    return raycaster.ray.origin.distanceTo(worldPoint);
}

/**
 * three-mesh-bvh's `PointsBVH`: a tree over a `Points` geometry's entries
 * (the core's `CpuPointsBvh` with three's types). Build it with
 * `geometry.computeBoundsTree({ type: PointsBVH })` or `new PointsBVH(geometry)`.
 */
export class BvhPointsBoundsTree extends CpuPointsBvh {
    readonly geometry: BufferGeometry;
    private readonly state: GeometryState;

    constructor(geometry: BufferGeometry, options: BvhPrimitiveBoundsTreeOptions = {}) {
        const buildOptions = buildOptionsOf(options);
        const input = pointsInputFromGeometry(geometry);
        super(options.layout ?? buildPrimitiveLayout('points', input, buildOptions), { source: input });
        this.geometry = geometry;
        this.state = new GeometryState(geometry);
        this.state.setBoundingBox(this, options);
    }

    /** Whether the geometry is as the tree was built from it (or last refitted to). */
    current(): boolean {
        return this.state.current();
    }

    /** Refits to the geometry's current positions (three-mesh-bvh's `refit()`); with `input`, to those arrays. */
    override refit(input?: BvhPointsInput): void {
        super.refit(input ?? this.state.refitInput('PointsBVH.refit'));
        this.state.updateBoundingBox(this);
    }

    /** three-mesh-bvh's `PointsBVH.shapecast`, with `Box3` boxes and `Vector3` points (scratch objects). */
    override shapecast(callbacks: BvhPointsShapecastCallbacks<Vector3, Box3>): boolean {
        return this.shapecastPoints(callbacks, new Vector3(), new Box3());
    }

    /** The point nearest `point` (geometry space): `{ point, distance, index }`, `point` a `Vector3`. */
    override closestPointToPoint(point: Vector3, target: Partial<BvhPrimitiveClosestPoint<Vector3>> = {}, minThreshold = 0, maxThreshold = Infinity): BvhPrimitiveClosestPoint<Vector3> | null {
        target.point ??= new Vector3();
        return super.closestPointToPoint(point, target, minThreshold, maxThreshold) as BvhPrimitiveClosestPoint<Vector3> | null;
    }

    /** three's `Points.raycast` for `object` (whose geometry this tree is), over the tree: see the file comment. */
    raycastObject3D(object: Points, raycaster: Raycaster, intersects: Intersection[] = []): Intersection[] {
        if (object.geometry !== this.geometry || this.state.refuses(object, (object as Object3D as { isPoints?: boolean }).isPoints === true)) {
            threeRaycast(object, raycaster, intersects);
            return intersects;
        }
        const query = localQuery(object, raycaster, raycaster.params.Points.threshold);
        if (!query) return intersects;
        const [start, end] = drawEntries(this.geometry);
        const matrixWorld = object.matrixWorld;
        const accept = (hit: BvhPrimitiveRaycastHit): boolean => {
            if (hit.index < start || hit.index >= end) return false;
            const distance = worldDistance(raycaster, matrixWorld, hit.pointOnRay);
            return !(distance < raycaster.near || distance > raycaster.far);
        };
        const ray = { origin: query.origin, direction: query.direction };
        const options = { threshold: query.threshold, accept };
        const hits = raycaster.firstHitOnly === true ? closestByWorld(this, ray, options, raycaster, matrixWorld) : this.raycast(ray, options).sort((a, b) => a.index - b.index);
        const index = this.geometry.index;
        for (const hit of hits) {
            const distance = worldDistance(raycaster, matrixWorld, hit.pointOnRay);
            intersects.push({
                distance,
                distanceToRay: hit.distanceToRay,
                point: worldPoint.clone(),
                index: index ? index.getX(hit.index) : hit.index,
                face: null,
                faceIndex: null,
                barycoord: null,
                object,
            } as unknown as Intersection);
        }
        return intersects;
    }
}

/** What each line object is: three's `isLineSegments`, `isLineLoop`, else a `Line`. */
function modeOfObject(object: Object3D): BvhSegmentMode | null {
    const line = object as Object3D & { isLine?: boolean; isLineSegments?: boolean; isLineLoop?: boolean };
    if (line.isLine !== true) return null;
    return line.isLineSegments ? 'segments' : line.isLineLoop ? 'loop' : 'strip';
}

/**
 * three-mesh-bvh's `LineSegmentsBVH` (and the base of {@link BvhLineLoopBoundsTree}
 * and {@link BvhLineBoundsTree}): a tree over a line geometry's segments,
 * the core's `CpuSegmentsBvh` with three's types. Build it with
 * `geometry.computeBoundsTree({ type: LineSegmentsBVH })` for a `LineSegments`.
 */
export class BvhLineSegmentsBoundsTree extends CpuSegmentsBvh {
    /** How this class pairs entries: `'segments'`; `LineLoopBVH` `'loop'`, `LineBVH` `'strip'`. */
    static readonly segmentMode: BvhSegmentMode = 'segments';
    readonly geometry: BufferGeometry;
    private readonly state: GeometryState;
    private readonly buildOptions: BvhPrimitiveBuildOptions;
    /** `LineSegments` drawn from an odd entry: the segments that start at odd entries, made on first use. */
    private oddTree: CpuSegmentsBvh | null = null;

    constructor(geometry: BufferGeometry, options: BvhPrimitiveBoundsTreeOptions = {}) {
        const mode = (new.target as typeof BvhLineSegmentsBoundsTree).segmentMode;
        const buildOptions = buildOptionsOf(options);
        const input: BvhSegmentsInput = { ...pointsInputFromGeometry(geometry), mode };
        const layout = options.layout ?? buildPrimitiveLayout('segments', input, buildOptions);
        if (layout.mode !== mode) throw new RangeError(`@pmndrs/bvh/three: the layout pairs entries as '${layout.mode}', this tree as '${mode}'`);
        super(layout, { source: input });
        this.geometry = geometry;
        this.buildOptions = buildOptions;
        this.state = new GeometryState(geometry);
        this.state.setBoundingBox(this, options);
    }

    current(): boolean {
        return this.state.current();
    }

    /** Refits to the geometry's current positions (three-mesh-bvh's `refit()`); with `input`, to those arrays. */
    override refit(input?: BvhPointsInput): void {
        super.refit({ ...(input ?? this.state.refitInput('LineSegmentsBVH.refit')), mode: this.mode });
        this.oddTree = null;
        this.state.updateBoundingBox(this);
    }

    /** three-mesh-bvh's `LineSegmentsBVH.shapecast`, with `Box3` boxes and `Line3` segments (scratch objects). */
    override shapecast(callbacks: BvhSegmentsShapecastCallbacks<Line3, Box3>): boolean {
        return this.shapecastSegments(callbacks, new Line3(), new Box3());
    }

    /** The nearest point on any segment to `point` (geometry space): `{ point, distance, index }`. */
    override closestPointToPoint(point: Vector3, target: Partial<BvhPrimitiveClosestPoint<Vector3>> = {}, minThreshold = 0, maxThreshold = Infinity): BvhPrimitiveClosestPoint<Vector3> | null {
        target.point ??= new Vector3();
        return super.closestPointToPoint(point, target, minThreshold, maxThreshold) as BvhPrimitiveClosestPoint<Vector3> | null;
    }

    /** three's `Line.raycast` for `object` (whose geometry this tree is), over the tree: see the file comment. */
    raycastObject3D(object: Line, raycaster: Raycaster, intersects: Intersection[] = []): Intersection[] {
        if (object.geometry !== this.geometry || this.state.refuses(object, modeOfObject(object) === this.mode)) {
            threeRaycast(object, raycaster, intersects);
            return intersects;
        }
        const query = localQuery(object, raycaster, raycaster.params.Line.threshold);
        if (!query) return intersects;
        const [start, end] = drawEntries(this.geometry);
        const matrixWorld = object.matrixWorld;
        const accept = (hit: BvhPrimitiveRaycastHit): boolean => {
            // three: for (i = start; i < end - 1; i += step). A loop's closing segment for the range is tested below.
            if (hit.index < start || hit.index >= end - 1) return false;
            const distance = worldDistance(raycaster, matrixWorld, hit.pointOnRay);
            return !(distance < raycaster.near || distance > raycaster.far);
        };
        const tree = this.mode === 'segments' && start % 2 === 1 ? this.odd() : this;
        const ray = { origin: query.origin, direction: query.direction };
        const options = { threshold: query.threshold, accept };
        const firstHitOnly = raycaster.firstHitOnly === true;
        let hits = firstHitOnly ? closestByWorld(tree, ray, options, raycaster, matrixWorld) : tree.raycast(ray, options).sort((a, b) => a.index - b.index);
        if (this.mode === 'loop' && end - start >= 1) {
            // three: the segment from the range's last entry back to its first, pushed last.
            const index = this.geometry.index;
            const position = this.geometry.getAttribute('position');
            vStart.fromBufferAttribute(position, index ? index.getX(end - 1) : end - 1);
            vEnd.fromBufferAttribute(position, index ? index.getX(start) : start);
            const closing = raySegmentHit(query.origin, query.direction, vStart, vEnd, query.threshold, end - 1);
            const distance = closing ? worldDistance(raycaster, matrixWorld, closing.pointOnRay) : NaN;
            if (closing && !(distance < raycaster.near || distance > raycaster.far)) {
                if (!firstHitOnly) hits.push(closing);
                else if (hits.length === 0 || distance < worldDistance(raycaster, matrixWorld, hits[0].pointOnRay)) hits = [closing];
            }
        }
        for (const hit of hits) {
            const distance = worldDistance(raycaster, matrixWorld, hit.pointOnRay);
            intersects.push({
                distance,
                point: new Vector3(hit.pointOnPrimitive[0], hit.pointOnPrimitive[1], hit.pointOnPrimitive[2]).applyMatrix4(matrixWorld),
                index: hit.index,
                face: null,
                faceIndex: null,
                barycoord: null,
                object,
            } as unknown as Intersection);
        }
        return intersects;
    }

    /** The tree of segments starting at odd entries (a `LineSegments` draw range from an odd entry). */
    private odd(): CpuSegmentsBvh {
        this.oddTree ??= CpuSegmentsBvh.build({ ...pointsInputFromGeometry(this.geometry), mode: 'segments', start: 1 }, this.buildOptions);
        return this.oddTree;
    }
}

/** three-mesh-bvh's `LineLoopBVH`: for a `LineLoop`, its closing segment included. */
export class BvhLineLoopBoundsTree extends BvhLineSegmentsBoundsTree {
    static override readonly segmentMode: BvhSegmentMode = 'loop';
}

/** three-mesh-bvh's `LineBVH`: for a `Line`, each entry to the next. */
export class BvhLineBoundsTree extends BvhLineLoopBoundsTree {
    static override readonly segmentMode: BvhSegmentMode = 'strip';
}

/**
 * `acceleratedRaycast` for `Points`, `LineSegments`, `LineLoop` and `Line`:
 * the geometry's tree when it is one of these, else three's own raycast.
 */
export function raycastPointsOrLines(object: Object3D & { geometry: BufferGeometry }, raycaster: Raycaster, intersects: Intersection[]): void {
    const tree = (object.geometry as unknown as { boundsTree?: unknown }).boundsTree;
    if (tree instanceof BvhPointsBoundsTree) tree.raycastObject3D(object as unknown as Points, raycaster, intersects);
    else if (tree instanceof BvhLineSegmentsBoundsTree) tree.raycastObject3D(object as unknown as Line, raycaster, intersects);
    else threeRaycast(object, raycaster, intersects);
}

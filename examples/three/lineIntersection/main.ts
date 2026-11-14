/**
 * three/lineIntersection: a port of three-mesh-bvh's `lineIntersection` example.
 *
 * What it demonstrates: raycasting a polyline of hundreds of thousands of
 * segments with a pick threshold. three-mesh-bvh does it with `LineBVH`;
 * `@pmndrs/bvh` has no line primitive (leaves hold triangles only, see
 * "API gaps" in docs/three/coming-from-three-mesh-bvh.mdx), so each segment
 * goes into a hidden geometry as a degenerate triangle (a, b, b), the tree
 * is the ordinary triangle tree, and `tree.shapecast` does the ray test:
 *
 * - `intersectsBounds` widens each box by the line threshold and tests the ray;
 * - `intersectsTriangle` computes three's `Line.raycast` arithmetic
 *   (`ray.distanceSqToSegment`) on the degenerate triangle's one edge.
 *
 * "use BVH" off is three's own `Line.raycast` over every segment. The smoke
 * run casts rays both ways and requires the same segment and distance.
 */

import { Box3, BufferAttribute, BufferGeometry, Color, Float32BufferAttribute, Line, LineBasicMaterial, Matrix4, Mesh, MeshBasicMaterial, Ray, Raycaster, SphereGeometry, Vector2, Vector3, type Intersection } from 'three';
import { BVHHelper, INTERSECTED, NOT_INTERSECTED, computeBoundsTree, prepareBoundsTrees, type BvhBoundsTree } from '@pmndrs/bvh/three';
import { createApp } from '../../shared/app.ts';
import { layoutInfo } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

const app = await createApp('Line intersection', 'three/lineIntersection');
const { device, scene, camera, overlay } = app;
await prepareBoundsTrees();
camera.position.set(2, 1, 2);

const SIZES = { '20k': 20_000, '250k': 250_000, '1M': 1_000_000 } as const;
type Size = keyof typeof SIZES;
const THRESHOLD = 0.01;
const params = { useBVH: true, displayBVH: false, displayDepth: 10, rotate: !isSmoke };
let size: Size = isSmoke ? '20k' : '250k';

const threeLineRaycast = Line.prototype.raycast;
const raycaster = new Raycaster();
raycaster.params.Line.threshold = THRESHOLD;
const mouse = new Vector2();

const collision = new Mesh(new SphereGeometry(0.01, 16, 16), new MeshBasicMaterial({ color: 0xff0000, transparent: true, opacity: 0.75 }));
collision.visible = false;
scene.add(collision);

let line: Line;
let proxy: Mesh; // the degenerate-triangle geometry the tree is built over
let tree: BvhBoundsTree;
let helper: BVHHelper;

/** The torus-knot-wrapped curve of the original, `segments` + 1 points. */
function generateCurve(segments: number): Float32Array {
    const points = new Float32Array((segments + 1) * 3);
    const norm = new Vector3();
    const tangent = new Vector3();
    const v0 = new Vector3();
    const v1 = new Vector3();
    const surface = (t: number, target: Vector3): void => {
        const p = 3;
        const q = 10;
        const R = 1.0;
        const r = 0.4;
        const theta = t * Math.PI * 2;
        const phi = p * theta;
        const psi = q * theta;
        target.set((R + r * Math.cos(psi)) * Math.cos(phi), (R + r * Math.cos(psi)) * Math.sin(phi), r * Math.sin(psi));
    };
    for (let i = 0; i <= segments; i++) {
        const t0 = i / segments;
        surface(t0, v0);
        surface(t0 + 1e-4, v1);
        norm.copy(v0).normalize();
        tangent.subVectors(v1, v0).normalize();
        norm.applyAxisAngle(tangent, 1000 * t0 * 2 * Math.PI);
        v0.addScaledVector(norm, 0.05 * (Math.sin(50 * t0 * Math.PI) + 2));
        v0.toArray(points, i * 3);
    }
    return points;
}

function makeLine(): void {
    if (line) {
        scene.remove(line, helper);
        line.geometry.dispose();
        proxy.geometry.dispose();
        helper.dispose();
    }
    const segments = SIZES[size];
    const points = generateCurve(segments);
    const colors = new Float32Array((segments + 1) * 3);
    const color = new Color();
    for (let i = 0; i <= segments; i++) color.setHSL((i / segments) * 3, 1.0, 0.6).toArray(colors, i * 3);
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new Float32BufferAttribute(points, 3));
    geometry.setAttribute('color', new Float32BufferAttribute(colors, 3));
    line = new Line(geometry, new LineBasicMaterial({ vertexColors: true }));

    // Segment i, (p[i], p[i+1]), is triangle i: (a, b, b).
    const triangles = new Float32Array(segments * 9);
    for (let i = 0; i < segments; i++) {
        triangles.set(points.subarray(i * 3, i * 3 + 6), i * 9);
        triangles.set(points.subarray(i * 3 + 3, i * 3 + 6), i * 9 + 6);
    }
    const proxyGeometry = new BufferGeometry();
    proxyGeometry.setAttribute('position', new BufferAttribute(triangles, 3));
    proxy = new Mesh(proxyGeometry, new MeshBasicMaterial());
    proxy.visible = false;
    line.add(proxy);
    const start = performance.now();
    tree = computeBoundsTree.call(proxyGeometry);
    overlay.set('tree build (tinybvh)', `${(performance.now() - start).toFixed(1)} ms`);
    overlay.tree(layoutInfo(tree.layout));
    overlay.set('segments', segments);
    helper = new BVHHelper(proxy, params.displayDepth);
    scene.add(line, helper);
}

const inverse = new Matrix4();
const localRay = new Ray();
const widened = new Box3();
const interRay = new Vector3();
const interSegment = new Vector3();
const hitPoint = new Vector3();
const boxHit = new Vector3();

interface LineHit {
    readonly index: number;
    readonly distance: number;
    readonly point: Vector3;
    /** The ray's own nearest point (local), which three measures the distance to. */
    readonly onRay?: Vector3;
}

/** The nearest segment within the threshold of the ray, found through the tree: LineBVH's raycastFirst, made of a shapecast. */
function raycastFirstBvh(ray: Ray, threshold: number): LineHit | null {
    // As three's Line.raycast: the threshold is local, the ray goes into the line's space.
    inverse.copy(line.matrixWorld).invert();
    localRay.copy(ray).applyMatrix4(inverse);
    const localThreshold = threshold / ((line.scale.x + line.scale.y + line.scale.z) / 3);
    const thresholdSq = localThreshold * localThreshold;
    let best = null as LineHit | null;
    let bestDistance = Infinity;
    tree.shapecast({
        intersectsBounds: (box) => {
            widened.min.copy(box.min).subScalar(localThreshold);
            widened.max.copy(box.max).addScalar(localThreshold);
            if (!localRay.intersectsBox(widened)) return NOT_INTERSECTED;
            // Nothing in a box beyond the best hit can beat it: the ray's nearest point to a segment lies inside the widened box, no nearer than where the ray enters it.
            if (best && !widened.containsPoint(localRay.origin) && localRay.intersectBox(widened, boxHit) && localRay.origin.distanceTo(boxHit) > bestDistance) return NOT_INTERSECTED;
            return INTERSECTED;
        },
        intersectsTriangle: (triangle, index) => {
            const distSq = localRay.distanceSqToSegment(triangle.a, triangle.b, interRay, interSegment);
            if (distSq > thresholdSq) return false;
            const distance = localRay.origin.distanceTo(interRay);
            if (distance < bestDistance) {
                bestDistance = distance;
                best = { index, distance, point: interSegment.clone(), onRay: interRay.clone() };
            }
            return false;
        },
    });
    if (!best) return null;
    const found = best as LineHit;
    // Back to world space; the distance in world units like three's.
    hitPoint.copy(found.point).applyMatrix4(line.matrixWorld);
    return { index: found.index, distance: ray.origin.distanceTo(found.onRay!.applyMatrix4(line.matrixWorld)), point: hitPoint.clone() };
}

/** three's own Line.raycast over every segment, the nearest hit. */
function raycastFirstThree(): LineHit | null {
    const hits: Intersection[] = [];
    threeLineRaycast.call(line, raycaster, hits);
    hits.sort((a, b) => a.distance - b.distance);
    const hit = hits[0];
    return hit ? { index: hit.faceIndex ?? hit.index ?? -1, distance: hit.distance, point: hit.point } : null;
}

function pick(): { hit: LineHit | null; ms: number } {
    raycaster.setFromCamera(mouse, camera);
    const start = performance.now();
    const hit = params.useBVH ? raycastFirstBvh(raycaster.ray, THRESHOLD) : raycastFirstThree();
    return { hit, ms: performance.now() - start };
}

overlay.checkbox('use BVH', params.useBVH, (v) => (params.useBVH = v));
overlay.checkbox('rotate', params.rotate, (v) => (params.rotate = v));
overlay.checkbox('display BVH', params.displayBVH, (v) => (params.displayBVH = v));
overlay.range('BVH depth', 1, 25, 1, params.displayDepth, (v) => {
    params.displayDepth = v;
    helper.depth = v;
    helper.update();
});
overlay.select('segments', Object.keys(SIZES) as Size[], size, (v) => {
    size = v;
    makeLine();
});
overlay.note('Demonstrates tree.shapecast as a line raycast: @pmndrs/bvh has no LineBVH, so every segment is a degenerate triangle in a hidden mesh and the shapecast does the ray-segment test with the pick threshold. "use BVH" off is three\'s own Line.raycast.');
addEventListener('pointermove', (event) => mouse.set((event.clientX / innerWidth) * 2 - 1, -(event.clientY / innerHeight) * 2 + 1));
makeLine();

let frames = 0;
app.loop(async () => {
    helper.visible = params.displayBVH;
    if (params.rotate) line.rotation.y = performance.now() * 1e-4;
    line.updateMatrixWorld(true);
    const { hit, ms } = pick();
    collision.visible = hit !== null;
    if (hit) collision.position.copy(hit.point);
    overlay.set('pick', hit ? `segment ${hit.index}, ${ms.toFixed(2)} ms ${params.useBVH ? '(BVH)' : '(three Line.raycast)'}` : `miss, ${ms.toFixed(2)} ms`);
    frames++;
    if (isSmoke && frames === 3) {
        // Rays through the middle of the view, both ways; the line is rotated.
        line.rotation.y = 0.9;
        line.updateMatrixWorld(true);
        let hits = 0;
        let bvhMs = 0;
        let threeMs = 0;
        const spread = 40;
        for (let k = 0; k < spread; k++) {
            mouse.set(-0.4 + (0.8 * (k % 8)) / 7, -0.3 + (0.6 * Math.floor(k / 8)) / 4);
            raycaster.setFromCamera(mouse, camera);
            const t0 = performance.now();
            const ours = raycastFirstBvh(raycaster.ray, THRESHOLD);
            const t1 = performance.now();
            const theirs = raycastFirstThree();
            const t2 = performance.now();
            bvhMs += t1 - t0;
            threeMs += t2 - t1;
            if (!ours !== !theirs) throw new Error(`ray ${k}: BVH ${ours?.index}, three ${theirs?.index}`);
            if (ours && theirs) {
                hits++;
                if (Math.abs(ours.distance - theirs.distance) > 1e-4 * Math.max(1, theirs.distance)) throw new Error(`ray ${k}: distance ${ours.distance} vs three's ${theirs.distance}`);
            }
        }
        if (hits === 0) throw new Error('no smoke ray hit the line');
        overlay.set('smoke', `${hits}/${spread} rays hit, same answers; BVH ${(bvhMs / spread).toFixed(3)} ms, three ${(threeMs / spread).toFixed(2)} ms per ray`);
        await ready(device, { ...overlay.snapshot(), hits, rays: spread, bvhMsPerRay: bvhMs / spread, threeMsPerRay: threeMs / spread });
    }
});

/**
 * three/lineIntersection: a port of three-mesh-bvh's `lineIntersection` example.
 * Source: three-mesh-bvh's example/lineIntersection.js (v0.9.18, MIT, Garrett Johnson); see THIRD_PARTY.md.
 *
 * What it demonstrates: raycasting a polyline of hundreds of thousands of
 * segments with a pick threshold, as three-mesh-bvh does with `LineBVH`:
 * `computeBoundsTree({ type: LineBVH })` builds a tree over the line's
 * segments (docs/three/points-and-lines.mdx), and `acceleratedRaycast` on the
 * `Line` answers what three's `Line.raycast` answers, from the tree, with
 * `raycaster.params.Line.threshold` and `firstHitOnly`.
 *
 * "use BVH" off is three's own `Line.raycast` over every segment. The smoke
 * run casts rays both ways and requires the same segment and distance.
 */

import { BufferGeometry, Color, Float32BufferAttribute, Line, LineBasicMaterial, Mesh, MeshBasicMaterial, Raycaster, SphereGeometry, Vector2, Vector3, type Intersection } from 'three';
import { BVHHelper, LineBVH, acceleratedRaycast, computeBoundsTree } from '@pmndrs/bvh/three';
import { createApp } from '../../shared/app.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

const app = await createApp('Line intersection', 'three/lineIntersection');
const { device, scene, camera, overlay } = app;
camera.position.set(2, 1, 2);

const SIZES = { '20k': 20_000, '250k': 250_000, '1M': 1_000_000 } as const;
type Size = keyof typeof SIZES;
const THRESHOLD = 0.01;
const params = { useBVH: true, displayBVH: false, displayDepth: 10, rotate: !isSmoke };
let size: Size = isSmoke ? '20k' : '250k';

const threeLineRaycast = Line.prototype.raycast;
// This site's TypeScript program also reads three-mesh-bvh's types for BufferGeometry (the compare pages import it),
// so computeBoundsTree is called directly; in an app, `geometry.computeBoundsTree({ type: LineBVH })` typechecks as is.
const computeLineTree = computeBoundsTree as (this: BufferGeometry, options: { type: typeof LineBVH }) => LineBVH;
const raycaster = new Raycaster();
raycaster.params.Line.threshold = THRESHOLD;
raycaster.firstHitOnly = true; // acceleratedRaycast keeps the closest hit only
const mouse = new Vector2();

const collision = new Mesh(new SphereGeometry(0.01, 16, 16), new MeshBasicMaterial({ color: 0xff0000, transparent: true, opacity: 0.75 }));
collision.visible = false;
scene.add(collision);

let line: Line;
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

    const start = performance.now();
    const tree = computeLineTree.call(geometry, { type: LineBVH });
    overlay.set('tree build (LineBVH)', `${(performance.now() - start).toFixed(1)} ms`);
    overlay.tree({ depth: tree.depth, nodes: tree.nodeCount, bytes: tree.estimateMemoryInBytes() });
    overlay.set('segments', segments);
    helper = new BVHHelper(line, params.displayDepth);
    scene.add(line, helper);
}

interface LineHit {
    readonly index: number;
    readonly distance: number;
    readonly point: Intersection['point'];
}

/** The nearest hit through `raycast` (acceleratedRaycast with the tree, or three's own Line.raycast). */
function raycastFirst(raycast: typeof threeLineRaycast): LineHit | null {
    const hits: Intersection[] = [];
    raycast.call(line, raycaster, hits);
    hits.sort((a, b) => a.distance - b.distance);
    const hit = hits[0];
    return hit ? { index: hit.index ?? -1, distance: hit.distance, point: hit.point } : null;
}

function pick(): { hit: LineHit | null; ms: number } {
    raycaster.setFromCamera(mouse, camera);
    const start = performance.now();
    const hit = raycastFirst(params.useBVH ? acceleratedRaycast : threeLineRaycast);
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
overlay.note('Demonstrates LineBVH: computeBoundsTree({ type: LineBVH }) builds a tree over the segments, and acceleratedRaycast on the Line answers as three\'s Line.raycast with the pick threshold (firstHitOnly). "use BVH" off is three\'s own Line.raycast.');
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
            const ours = raycastFirst(acceleratedRaycast);
            const t1 = performance.now();
            const theirs = raycastFirst(threeLineRaycast);
            const t2 = performance.now();
            bvhMs += t1 - t0;
            threeMs += t2 - t1;
            if (!ours !== !theirs) throw new Error(`ray ${k}: BVH ${ours?.index}, three ${theirs?.index}`);
            if (ours && theirs) {
                hits++;
                if (ours.distance !== theirs.distance || ours.index !== theirs.index) throw new Error(`ray ${k}: segment ${ours.index} at ${ours.distance} vs three's ${theirs.index} at ${theirs.distance}`);
            }
        }
        if (hits === 0) throw new Error('no smoke ray hit the line');
        overlay.set('smoke', `${hits}/${spread} rays hit, same answers; BVH ${(bvhMs / spread).toFixed(3)} ms, three ${(threeMs / spread).toFixed(2)} ms per ray`);
        await ready(device, { ...overlay.snapshot(), hits, rays: spread, bvhMsPerRay: bvhMs / spread, threeMsPerRay: threeMs / spread });
    }
});

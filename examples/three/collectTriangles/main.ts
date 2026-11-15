/**
 * three/collectTriangles: a port of three-mesh-bvh's `collectTriangles` example.
 * Source: three-mesh-bvh's example/collectTriangles.js (v0.9.18, MIT, Garrett Johnson); see THIRD_PARTY.md.
 *
 * What it demonstrates: `tree.shapecast` with a sphere as the shape, collecting
 * every triangle the sphere touches, so a brush can paint a mesh of hundreds
 * of thousands of triangles every frame. `intersectsBounds` returns
 * `CONTAINED` when a box lies wholly inside the sphere (its triangles are
 * taken with no test at all), `INTERSECTED` when the sphere crosses it, and
 * `NOT_INTERSECTED` otherwise. The pointer is found with `acceleratedRaycast`
 * (`firstHitOnly`).
 *
 * Left button paints, right button erases, the wheel sets the brush size.
 * The smoke run puts the brush on the surface and checks the collected
 * triangles against a pass over every triangle with `ExtendedTriangle.intersectsSphere`.
 */

import { BufferAttribute, DynamicDrawUsage, Matrix4, Mesh, MeshStandardMaterial, Raycaster, SphereGeometry, Sphere, TorusKnotGeometry, Vector2, Vector3 } from 'three';
import { CONTAINED, ExtendedTriangle, INTERSECTED, NOT_INTERSECTED, acceleratedRaycast, computeBoundsTree, disposeBoundsTree, prepareBoundsTrees, type BvhBoundsTree } from '@pmndrs/bvh/three';
import { createApp } from '../../shared/app.ts';
import { layoutInfo } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

const app = await createApp('Collect triangles', 'three/collectTriangles');
const { device, scene, camera, controls, overlay } = app;
await prepareBoundsTrees();

// The setup three-mesh-bvh's examples use, here with our functions.
Mesh.prototype.raycast = acceleratedRaycast;
const SIZES = { '40k': [200, 100], '160k': [400, 200], '640k': [800, 400] } as const;
type Size = keyof typeof SIZES;
const params = { size: 0.1, rotate: !isSmoke };
let size: Size = isSmoke ? '40k' : '160k';
camera.position.set(3, 3, 3);
controls.update();

let target: Mesh;
let tree: BvhBoundsTree;
const brush = new Mesh(
    new SphereGeometry(1, 40, 40),
    new MeshStandardMaterial({ color: 0xec407a, roughness: 0.75, transparent: true, opacity: 0.5, emissive: 0xec407a, emissiveIntensity: 0.5 }),
);
brush.visible = false;
scene.add(brush);

function makeTarget(): void {
    if (target) {
        scene.remove(target);
        disposeBoundsTree.call(target.geometry);
        target.geometry.dispose();
    }
    const [tubular, radial] = SIZES[size];
    const geometry = new TorusKnotGeometry(1, 0.4, tubular, radial).toNonIndexed();
    const colors = new Uint8Array(geometry.attributes.position.count * 3).fill(255);
    const attribute = new BufferAttribute(colors, 3, true);
    attribute.setUsage(DynamicDrawUsage);
    geometry.setAttribute('color', attribute);
    target = new Mesh(geometry, new MeshStandardMaterial({ color: 0xffffff, roughness: 0.3, metalness: 0, vertexColors: true }));
    const start = performance.now();
    tree = computeBoundsTree.call(geometry);
    overlay.set('tree build (tinybvh)', `${(performance.now() - start).toFixed(1)} ms`);
    overlay.tree(layoutInfo(tree.layout));
    overlay.set('model', `torus knot, ${geometry.attributes.position.count / 3} triangles (non-indexed)`);
    scene.add(target);
}

const tempVec = new Vector3();
const triangle = new ExtendedTriangle();

/** Triangle indices the sphere (in the target's local space) touches. */
function collect(sphere: Sphere): number[] {
    const found: number[] = [];
    tree.shapecast({
        intersectsBounds: (box) => {
            if (!sphere.intersectsBox(box)) return NOT_INTERSECTED;
            // Every corner inside the sphere: its triangles all are, no per-triangle test needed.
            const { min, max } = box;
            for (let x = 0; x <= 1; x++) {
                for (let y = 0; y <= 1; y++) {
                    for (let z = 0; z <= 1; z++) {
                        tempVec.set(x === 0 ? min.x : max.x, y === 0 ? min.y : max.y, z === 0 ? min.z : max.z);
                        if (!sphere.containsPoint(tempVec)) return INTERSECTED;
                    }
                }
            }
            return CONTAINED;
        },
        intersectsTriangle: (tri, index, contained) => {
            if (contained || tri.intersectsSphere(sphere)) found.push(index);
            return false;
        },
    });
    return found;
}

/** The same answer with no tree: every triangle against the sphere. */
function collectBrute(sphere: Sphere): number[] {
    const found: number[] = [];
    const position = target.geometry.attributes.position;
    for (let i = 0; i < position.count / 3; i++) {
        triangle.a.fromBufferAttribute(position, i * 3);
        triangle.b.fromBufferAttribute(position, i * 3 + 1);
        triangle.c.fromBufferAttribute(position, i * 3 + 2);
        triangle.needsUpdate = true;
        if (triangle.intersectsSphere(sphere)) found.push(i);
    }
    return found;
}

function localSphere(point: Vector3, radius: number): Sphere {
    const inverse = new Matrix4().copy(target.matrixWorld).invert();
    return new Sphere(point.clone().applyMatrix4(inverse), radius);
}

function paint(triangles: readonly number[], erase: boolean): void {
    const colorAttr = target.geometry.getAttribute('color') as BufferAttribute;
    const index = target.geometry.index;
    const [r, g, b] = erase ? [1, 1, 1] : [15 / 255, 78 / 255, 85 / 255];
    for (const triangleIndex of triangles) {
        for (let k = 0; k < 3; k++) {
            const vertex = index ? index.getX(triangleIndex * 3 + k) : triangleIndex * 3 + k;
            colorAttr.setXYZ(vertex, r, g, b);
        }
    }
    colorAttr.needsUpdate = true;
}

// Pointer handling as in the original.
const mouse = new Vector2();
let mouseButton = -1;
let brushActive = false;
let orbiting = false;
const raycaster = new Raycaster();
raycaster.firstHitOnly = true;

addEventListener('pointermove', (event) => {
    mouse.set((event.clientX / innerWidth) * 2 - 1, -(event.clientY / innerHeight) * 2 + 1);
    brushActive = true;
});
addEventListener(
    'pointerdown',
    (event) => {
        mouse.set((event.clientX / innerWidth) * 2 - 1, -(event.clientY / innerHeight) * 2 + 1);
        mouseButton = event.button;
        raycaster.setFromCamera(mouse, camera);
        brushActive = true;
        // Over the model, a drag paints; elsewhere it orbits.
        controls.enabled = raycaster.intersectObject(target, true).length === 0;
    },
    true,
);
addEventListener(
    'pointerup',
    (event) => {
        mouseButton = -1;
        if (event.pointerType === 'touch') brushActive = false;
    },
    true,
);
addEventListener('contextmenu', (event) => event.preventDefault());
addEventListener('wheel', (event) => {
    params.size = Math.max(Math.min(params.size + event.deltaY * (event.deltaMode ? 40 : 1) * 0.0005, 1), 0.1);
    sizeInput.value = String(params.size);
    sizeInput.dispatchEvent(new Event('input'));
});
controls.addEventListener('start', () => (orbiting = true));
controls.addEventListener('end', () => (orbiting = false));

const sizeInput = overlay.range('brush size', 0.1, 1, 0.1, params.size, (v) => (params.size = v));
overlay.checkbox('rotate', params.rotate, (v) => (params.rotate = v));
overlay.select('mesh', Object.keys(SIZES) as Size[], size, (v) => {
    size = v;
    makeTarget();
});
overlay.note('Demonstrates tree.shapecast with a sphere: CONTAINED boxes hand over all their triangles untested. Left drag paints, right drag erases, the wheel sizes the brush.');
makeTarget();

let lastTime = performance.now();
let lastCollect = { count: 0, ms: 0 };
let smokeFrames = 0;

app.loop(async () => {
    const now = performance.now();
    if (params.rotate) target.rotation.y += (now - lastTime) * 0.001;
    lastTime = now;
    target.updateMatrixWorld(true);

    if (orbiting || !brushActive) {
        brush.visible = false;
    } else {
        brush.scale.setScalar(params.size);
        raycaster.setFromCamera(mouse, camera);
        const hits = raycaster.intersectObject(target, true);
        if (hits.length) {
            brush.position.copy(hits[0].point);
            brush.visible = true;
            const start = performance.now();
            const triangles = collect(localSphere(brush.position, params.size));
            lastCollect = { count: triangles.length, ms: performance.now() - start };
            overlay.set('collected', `${triangles.length.toLocaleString('en-US')} triangles, ${lastCollect.ms.toFixed(3)} ms`);
            if (mouseButton === 0 || mouseButton === 2) paint(triangles, mouseButton === 2);
        } else {
            controls.enabled = true;
            brush.visible = false;
        }
    }

    if (isSmoke && ++smokeFrames === 3) {
        // The brush on the surface in the middle of the view, rotated, at two sizes.
        target.rotation.y = 0.7;
        target.updateMatrixWorld(true);
        raycaster.setFromCamera(new Vector2(0, 0), camera);
        const hit = raycaster.intersectObject(target, true)[0];
        if (!hit) throw new Error('the pointer ray missed the knot');
        const checks: { radius: number; collected: number; treeMs: number; bruteMs: number }[] = [];
        for (const radius of [0.1, 0.4]) {
            const sphere = localSphere(hit.point, radius);
            const t0 = performance.now();
            const viaTree = collect(sphere);
            const t1 = performance.now();
            const viaAll = collectBrute(sphere);
            const t2 = performance.now();
            viaTree.sort((a, b) => a - b);
            if (viaTree.length === 0) throw new Error(`radius ${radius}: collected nothing`);
            if (viaTree.length !== viaAll.length || viaTree.some((v, i) => v !== viaAll[i])) throw new Error(`radius ${radius}: shapecast collected ${viaTree.length} triangles, the brute-force pass ${viaAll.length}`);
            checks.push({ radius, collected: viaTree.length, treeMs: t1 - t0, bruteMs: t2 - t1 });
        }
        overlay.set('smoke', checks.map((c) => `r=${c.radius}: ${c.collected} tris, tree ${c.treeMs.toFixed(2)} ms, all ${c.bruteMs.toFixed(1)} ms`).join('; '));
        await ready(device, { ...overlay.snapshot(), checks, lastCollect });
    }
});

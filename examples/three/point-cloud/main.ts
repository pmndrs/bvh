/**
 * three/point-cloud: three-mesh-bvh's `pointCloudIntersection` example
 * (0.9.18, `example/pointCloudIntersection.js`, MIT, Garrett Johnson), ported
 * to `@pmndrs/bvh/three` and the examples shell.
 *
 * As there: a point cloud (the Porsche scan from gkjohnson/3d-demo-data, or
 * `?ply=<url>`) gets a `PointsBVH` through `computeBoundsTree({ type: PointsBVH })`,
 * `Points.prototype.raycast` is `acceleratedRaycast`, the pointer is picked
 * every frame with `firstHitOnly` and `raycaster.params.Points.threshold`, a
 * sphere marks the hit, and `BVHHelper` draws the tree's boxes. Turning the
 * tree off disposes it, and the same raycast then runs three's own
 * `Points.raycast` over every point.
 *
 * Changed: the shell's WebGPU renderer draws points one pixel wide (WebGPU has
 * no point size), so there is no point-size slider; `strategy` is accepted
 * and builds the same centre-split tree for every value; in smoke mode (and
 * when the scan can't be fetched) the cloud is a generated one, so the page
 * needs no network. Smoke checks the picks against three's `Points.raycast`.
 */

import { BufferAttribute, BufferGeometry, Mesh, MeshBasicMaterial, Points, PointsMaterial, Raycaster, SphereGeometry, Vector2, Vector3, type Intersection } from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { acceleratedRaycast, AVERAGE, BVHHelper, CENTER, computeBoundsTree, disposeBoundsTree, PointsBVH, SAH } from '@pmndrs/bvh/three';
import { createApp } from '../../shared/app.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

const threePointsRaycast = Points.prototype.raycast;
Points.prototype.raycast = acceleratedRaycast;
// The original sets BufferGeometry.prototype.computeBoundsTree / disposeBoundsTree too. This site's TypeScript program
// also reads three-mesh-bvh's types (the compare pages import it), whose BufferGeometry members win there, so the
// functions are called directly. In an app with only @pmndrs/bvh/three, the prototype way typechecks as is.
const computePointsTree = computeBoundsTree as (this: BufferGeometry, options: { strategy: number; type: typeof PointsBVH }) => PointsBVH;

const PLY_URL = 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/point-cloud-porsche/scene.ply';

const app = await createApp('Point cloud', 'three/point-cloud');
const { device, scene, camera, controls, overlay } = app;
camera.position.set(0, 2, 4);
controls.target.set(0, 0, 0);
controls.update();

const params = {
    displayBVH: false,
    displayDepth: 15,
    displayParents: false,
    strategy: 'CENTER' as 'CENTER' | 'AVERAGE' | 'SAH',
    raycastThreshold: 0.005,
    useBVH: true,
};
const STRATEGIES = { CENTER, AVERAGE, SAH } as const;

const mouse = new Vector2(2, 2);
const raycaster = new Raycaster();
raycaster.firstHitOnly = true;

const sphereCollision = new Mesh(new SphereGeometry(0.025, 32, 32), new MeshBasicMaterial({ color: 0xffff00, opacity: 0.9, transparent: true }));
sphereCollision.visible = false;
scene.add(sphereCollision);

/** A seeded random number in [0, 1) (mulberry32), so the generated cloud and the smoke rays repeat. */
function createRandom(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state + 0x6d2b79f5) >>> 0;
        let mixed = Math.imul(state ^ (state >>> 15), state | 1);
        mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
        return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
    };
}

/** A generated scan-like cloud: points on a torus knot's surface with a little noise, coloured by position. */
function generatedCloud(count: number): BufferGeometry {
    const random = createRandom(7);
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        const t = random() * Math.PI * 2;
        const a = random() * Math.PI * 2;
        // The knot's centre line (p = 2, q = 3) and a frame around it.
        const r = 1 + 0.4 * Math.cos(3 * t);
        const cx = r * Math.cos(2 * t);
        const cy = r * Math.sin(2 * t);
        const cz = 0.4 * Math.sin(3 * t);
        const tube = 0.18 + (random() - 0.5) * 0.004;
        const nx = Math.cos(a) * Math.cos(2 * t);
        const ny = Math.cos(a) * Math.sin(2 * t);
        const nz = Math.sin(a);
        positions.set([cx + nx * tube, cz + nz * tube, cy + ny * tube], i * 3);
        colors.set([0.45 + 0.5 * Math.cos(t), 0.55 + 0.4 * Math.sin(a), 0.85], i * 3);
    }
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(positions, 3));
    geometry.setAttribute('color', new BufferAttribute(colors, 3));
    return geometry;
}

async function loadCloud(): Promise<{ geometry: BufferGeometry; name: string }> {
    const url = new URLSearchParams(location.search).get('ply') ?? PLY_URL;
    if (isSmoke) return { geometry: generatedCloud(400_000), name: 'generated (smoke)' };
    try {
        const geometry = await new PLYLoader().loadAsync(url);
        return { geometry, name: url.split('/').slice(-2).join('/') };
    } catch (error) {
        overlay.error(`Could not load ${url} (${String(error)}); showing a generated cloud.`);
        return { geometry: generatedCloud(1_000_000), name: 'generated' };
    }
}

const { geometry, name } = await loadCloud();
const pointCloud = new Points(geometry, new PointsMaterial({ size: 0.005, vertexColors: geometry.hasAttribute('color') }));
// Centre the point cloud, as the original does.
geometry.computeBoundingBox();
geometry.boundingBox!.getCenter(pointCloud.position).multiplyScalar(-1);
pointCloud.position.y += 1;
pointCloud.updateMatrixWorld(true);

const bvhHelper = new BVHHelper(pointCloud, params.displayDepth);
scene.add(pointCloud, bvhHelper);
overlay.set('cloud', name);
overlay.set('points', geometry.getAttribute('position').count);

function updateBVH(): void {
    if (params.useBVH) {
        const start = performance.now();
        const tree = computePointsTree.call(pointCloud.geometry, { strategy: STRATEGIES[params.strategy], type: PointsBVH });
        const ms = performance.now() - start;
        overlay.build({ builtBy: 'PointsBVH (centre split, CPU)', cpuMs: ms });
        overlay.tree({ depth: tree.depth, nodes: tree.nodeCount, bytes: tree.estimateMemoryInBytes() });
    } else {
        disposeBoundsTree.call(pointCloud.geometry);
        overlay.build({ builtBy: "none: three's Points.raycast" });
    }
    bvhHelper.update();
}
updateBVH();

overlay.checkbox('displayBVH', params.displayBVH, (value) => (params.displayBVH = value));
overlay.checkbox('displayParents', params.displayParents, (value) => {
    params.displayParents = value;
    bvhHelper.displayParents = value;
    bvhHelper.update();
});
overlay.range('depth', 1, 25, 1, params.displayDepth, (value) => {
    params.displayDepth = value;
    bvhHelper.depth = value;
    bvhHelper.update();
});
overlay.checkbox('useBVH', params.useBVH, (value) => {
    params.useBVH = value;
    updateBVH();
});
overlay.select('strategy', ['CENTER', 'AVERAGE', 'SAH'] as const, params.strategy, (value) => {
    params.strategy = value;
    updateBVH();
});
overlay.range('raycastThreshold', 0.001, 0.01, 0.001, params.raycastThreshold, (value) => (params.raycastThreshold = value));
overlay.note('Every strategy builds the same centre-split tree here. WebGPU draws points one pixel wide, so there is no point size.');

addEventListener('pointermove', (event) => mouse.set((event.clientX / innerWidth) * 2 - 1, -(event.clientY / innerHeight) * 2 + 1));

/** Smoke: picks through the tree against three's own Points.raycast, on rays across the view. */
function checkAgainstThree(): number {
    const random = createRandom(3);
    const ndc = new Vector2();
    let hits = 0;
    for (let i = 0; i < 64; i++) {
        raycaster.setFromCamera(ndc.set(random() * 1.2 - 0.6, random() * 1.2 - 0.6), camera);
        const ours = raycaster.intersectObject(pointCloud);
        const all: Intersection[] = [];
        threePointsRaycast.call(pointCloud, raycaster, all);
        all.sort((a, b) => a.distance - b.distance);
        const a = ours[0];
        const b = all[0];
        if (!a !== !b || (a && b && (a.distance !== b.distance || a.index !== b.index))) throw new Error(`ray ${i}: PointsBVH picks ${a?.index} at ${a?.distance}, three's Points.raycast ${b?.index} at ${b?.distance}`);
        if (a) hits++;
    }
    return hits;
}

let frames = 0;
app.loop(async () => {
    pointCloud.updateMatrixWorld(true);
    raycaster.params.Points.threshold = params.raycastThreshold;
    bvhHelper.visible = params.displayBVH;

    raycaster.setFromCamera(mouse, camera);
    const startTime = performance.now();
    const intersects = raycaster.intersectObject(pointCloud);
    const hit = intersects[0];
    if (hit) {
        sphereCollision.position.copy(hit.point);
        sphereCollision.visible = true;
    } else {
        sphereCollision.visible = false;
    }
    const delta = performance.now() - startTime;
    overlay.set('raycast', `${delta.toFixed(2)} ms`);
    overlay.set('hit', hit ? `point ${hit.index}, ${(hit as Intersection & { distanceToRay?: number }).distanceToRay?.toFixed(4)} from the ray` : 'none');

    if (isSmoke && ++frames === 10) {
        const checked = checkAgainstThree();
        if (checked === 0) throw new Error('no smoke ray hit the cloud');
        const bounds = new Vector3();
        geometry.boundingBox!.getSize(bounds);
        await ready(device, { ...overlay.snapshot(), checkedAgainstThree: 64, hits: checked, size: bounds.toArray() });
    }
});

/**
 * three/physics: three-mesh-bvh's `physics` example on our tree.
 *
 * An environment baked into one static collider (`StaticGeometryGenerator`,
 * `computeBoundsTree`), and spheres: click to throw one from the camera. Each
 * physics step moves every sphere and asks the tree for the triangles near it
 * (`shapecast`, boxes visited nearest first with `boundsTraverseOrder`); each
 * triangle closer than the radius (`ExtendedTriangle.closestPointToPoint`)
 * pushes the sphere out, and the velocity is reflected off the push with some
 * damping. Spheres also collide with each other (a brute-force pass, as in the
 * original), and every hard contact leaves a ring.
 *
 * The original loads a jungle glTF from the network; this page builds a
 * bowl of noise terrain with rocks, ramps and a knot from three's primitives,
 * so it needs no asset. `?model=` bakes a glTF of yours instead.
 *
 * Ported from three-mesh-bvh's example/physics.js (v0.9.18, MIT, Garrett
 * Johnson); see THIRD_PARTY.md. Changed: the environment, our shell and
 * overlay, no shadows or point lights, the smoke check.
 */

import { Box3, BoxGeometry, Color, DodecahedronGeometry, Group, Mesh, MeshBasicMaterial, MeshStandardMaterial, PlaneGeometry, Raycaster, RingGeometry, Sphere, SphereGeometry, TorusKnotGeometry, Vector2, Vector3, DoubleSide, type BufferGeometry, type Object3D } from 'three';
import { SimplexNoise } from 'three/addons/math/SimplexNoise.js';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { computeBoundsTree, MeshBVHHelper, prepareBoundsTrees, StaticGeometryGenerator, type BvhBoundsTree } from '@pmndrs/bvh/three';
import { createApp } from '../../shared/app.ts';
import { loadGltf, requestedModel } from '../../shared/loader.ts';
import { layoutInfo } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

await prepareBoundsTrees();
const app = await createApp('Physics', 'three/physics');
const { renderer, scene, camera, overlay } = app;
camera.position.set(14, 12, -14);
app.controls.target.set(0, 0, 0);

const params = { gravity: -9.8, physicsSteps: 5, simulationSpeed: 1, sphereSize: 1, pause: false };

// ---- The environment: a bowl of seeded noise, rocks, a ramp and a knot. All indexed (the generator merges indexed
// geometry with indexed only; three's polyhedra are not, so the rocks are welded).
function environmentScene(): Object3D {
    let seed = 5;
    const random = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
    const noise = new SimplexNoise({ random });
    const root = new Group();
    const add = (geometry: BufferGeometry, color: number, x: number, y: number, z: number) => {
        const mesh = new Mesh(geometry, new MeshStandardMaterial({ color, flatShading: true }));
        mesh.position.set(x, y, z);
        mesh.rotation.set(random() * 6, random() * 6, random() * 6);
        root.add(mesh);
        return mesh;
    };
    const ground = new PlaneGeometry(40, 40, 120, 120);
    ground.rotateX(-Math.PI / 2);
    const position = ground.attributes.position;
    for (let i = 0; i < position.count; i++) {
        const x = position.getX(i);
        const z = position.getZ(i);
        const r = Math.hypot(x, z) / 20;
        // Flat in the middle, rising to a rim, with noise.
        position.setY(i, Math.max(0, r - 0.55) ** 2 * 30 + noise.noise(x / 6, z / 6) * 0.6);
    }
    ground.computeVertexNormals();
    root.add(new Mesh(ground, new MeshStandardMaterial({ color: 0x6f8f5f, flatShading: true, side: DoubleSide })));
    for (let i = 0; i < 9; i++) add(mergeVertices(new DodecahedronGeometry(0.8 + random() * 1.4, 1)), 0x8a8580, (random() - 0.5) * 18, 0.4, (random() - 0.5) * 18);
    const ramp = add(new BoxGeometry(5, 0.4, 10), 0xb08a60, -6, 1.6, 4);
    ramp.rotation.set(0.35, 0.4, 0);
    const knot = add(new TorusKnotGeometry(1.6, 0.5, 160, 20), 0x8fb3ff, 4, 3, -3);
    knot.rotation.set(0.5, 0, 0.3);
    return root;
}

const model = requestedModel();
let environment: Object3D;
if (model) {
    environment = (await loadGltf(model)).root;
    const box = new Box3().setFromObject(environment);
    environment.scale.setScalar(40 / box.getSize(new Vector3()).length());
    environment.updateMatrixWorld(true);
    box.setFromObject(environment);
    environment.position.sub(box.getCenter(new Vector3()));
} else environment = environmentScene();
scene.add(environment);
environment.updateMatrixWorld(true);

const bakeStart = performance.now();
const generator = new StaticGeometryGenerator(environment);
generator.attributes = ['position'];
const mergedGeometry = generator.generate();
const bakeMs = performance.now() - bakeStart;
const buildStart = performance.now();
const tree: BvhBoundsTree = computeBoundsTree.call(mergedGeometry);
overlay.build({ builtBy: 'tinybvh (computeBoundsTree)', cpuMs: performance.now() - buildStart });
overlay.tree(layoutInfo(tree.layout));
overlay.set('StaticGeometryGenerator', `${bakeMs.toFixed(1)} ms, ${generator.meshes.length} meshes`);
const collider = new Mesh(mergedGeometry, new MeshBasicMaterial({ wireframe: true, opacity: 0.5, transparent: true }));
collider.visible = false;
scene.add(collider);
const helper = new MeshBVHHelper(collider, 10);
helper.visible = false;
scene.add(helper);

// ---- Spheres and the rings they leave.
interface Ball {
    readonly mesh: Mesh<SphereGeometry, MeshStandardMaterial>;
    readonly collider: Sphere;
    readonly velocity: Vector3;
    readonly mass: number;
}
interface Ring {
    readonly mesh: Mesh<RingGeometry, MeshBasicMaterial>;
    lifetime: number;
    readonly maxLifetime: number;
    readonly maxScale: number;
}
const spheres: Ball[] = [];
const hits: Ring[] = [];
const sphereGeometry = new SphereGeometry(1, 20, 20);
const ringGeometry = new RingGeometry(0, 1, 30);
const forward = new Vector3(0, 0, 1);
let environmentContacts = 0;

function createSphere(random = Math.random): Ball {
    const color = new Color(0x131619).lerp(new Color(0xffffff), random() * 0.5 + 0.5);
    const mesh = new Mesh(sphereGeometry, new MeshStandardMaterial({ color }));
    scene.add(mesh);
    const radius = 0.5 * params.sphereSize * (random() * 0.2 + 0.6);
    mesh.scale.setScalar(radius);
    const ball = { mesh, collider: new Sphere(mesh.position, radius), velocity: new Vector3(), mass: (radius ** 3 * Math.PI * 4) / 3 };
    spheres.push(ball);
    return ball;
}

function removeSphere(index: number): void {
    const [ball] = spheres.splice(index, 1);
    ball.mesh.material.dispose();
    scene.remove(ball.mesh);
}

function onCollide(radius: number, point: Vector3, normal: Vector3, speed: number, offset = 0): void {
    if (speed < Math.max(Math.abs(0.04 * params.gravity), 5)) return;
    const mesh = new Mesh(ringGeometry, new MeshBasicMaterial({ side: DoubleSide, transparent: true, depthWrite: false }));
    mesh.position.copy(point).addScaledVector(normal, offset);
    mesh.quaternion.setFromUnitVectors(forward, normal);
    scene.add(mesh);
    hits.push({ mesh, lifetime: 0, maxLifetime: 0.4, maxScale: Math.max(radius, 0.4) * 2 * Math.max(Math.sin((Math.min(speed / 200, 1) * Math.PI) / 2), 0.35) });
}

const tempSphere = new Sphere();
const deltaVec = new Vector3();
const tempVec = new Vector3();
function updateSphereCollisions(deltaTime: number): void {
    for (let i = 0; i < spheres.length; i++) {
        const ball = spheres[i];
        const sphere = ball.collider;
        ball.velocity.y += params.gravity * deltaTime;
        sphere.center.addScaledVector(ball.velocity, deltaTime);
        if (sphere.center.y < -80) {
            removeSphere(i--);
            continue;
        }
        tempSphere.copy(sphere);
        let collided = false;
        tree.shapecast({
            intersectsBounds: (box) => box.intersectsSphere(tempSphere),
            intersectsTriangle: (triangle) => {
                triangle.closestPointToPoint(tempSphere.center, deltaVec);
                deltaVec.sub(tempSphere.center);
                const distance = deltaVec.length();
                if (distance < tempSphere.radius) {
                    const depth = distance - tempSphere.radius;
                    deltaVec.multiplyScalar(1 / distance);
                    tempSphere.center.addScaledVector(deltaVec, depth);
                    collided = true;
                }
            },
            boundsTraverseOrder: (box) => box.distanceToPoint(tempSphere.center) - tempSphere.radius,
        });
        if (collided) {
            environmentContacts++;
            deltaVec.subVectors(tempSphere.center, sphere.center).normalize();
            ball.velocity.reflect(deltaVec);
            const dot = ball.velocity.dot(deltaVec);
            ball.velocity.addScaledVector(deltaVec, -dot * 0.5);
            ball.velocity.multiplyScalar(Math.max(1 - deltaTime, 0));
            sphere.center.copy(tempSphere.center);
            tempVec.copy(tempSphere.center).addScaledVector(deltaVec, -tempSphere.radius);
            onCollide(sphere.radius, tempVec, deltaVec, dot, 0.05);
        }
    }
    // Sphere against sphere: every pair (as the original).
    for (let i = 0; i < spheres.length; i++) {
        const s1 = spheres[i];
        const c1 = s1.collider;
        for (let j = i + 1; j < spheres.length; j++) {
            const s2 = spheres[j];
            const c2 = s2.collider;
            deltaVec.subVectors(c1.center, c2.center);
            const depth = deltaVec.length() - (c1.radius + c2.radius);
            if (depth >= 0) continue;
            deltaVec.normalize();
            const v1dot = s1.velocity.dot(deltaVec);
            const v2dot = s2.velocity.dot(deltaVec);
            const offsetRatio1 = Math.max(v1dot, 0.2);
            const offsetRatio2 = Math.max(v2dot, 0.2);
            const total = offsetRatio1 + offsetRatio2;
            c1.center.addScaledVector(deltaVec, (-offsetRatio1 / total) * depth);
            c2.center.addScaledVector(deltaVec, (offsetRatio2 / total) * depth);
            const velocityDifference = new Vector3().addScaledVector(deltaVec, -v1dot).addScaledVector(deltaVec, v2dot);
            const velDiff = velocityDifference.length();
            const m1 = s1.mass;
            const m2 = s2.mass;
            const damping = 0.5;
            let newVel1: number;
            let newVel2: number;
            if (velocityDifference.dot(s1.velocity) > velocityDifference.dot(s2.velocity)) {
                newVel1 = (damping * velDiff * (m1 - m2)) / (m1 + m2) - velDiff;
                newVel2 = (damping * velDiff * 2 * m1) / (m1 + m2);
            } else {
                newVel1 = (damping * velDiff * 2 * m2) / (m1 + m2);
                newVel2 = (damping * velDiff * (m2 - m1)) / (m1 + m2) - velDiff;
            }
            velocityDifference.normalize();
            s1.velocity.addScaledVector(velocityDifference, newVel1);
            s2.velocity.addScaledVector(velocityDifference, newVel2);
            tempVec.copy(c1.center).addScaledVector(deltaVec, -c1.radius);
            onCollide(Math.max(c1.radius, c2.radius), tempVec, deltaVec, velDiff, 0);
        }
        s1.mesh.position.copy(c1.center);
    }
}

let lastStepMs = 0;
function update(delta: number): void {
    const start = performance.now();
    const steps = params.physicsSteps;
    for (let i = 0; i < steps; i++) updateSphereCollisions(delta / steps);
    lastStepMs = performance.now() - start;
    for (let i = 0; i < hits.length; i++) {
        const hit = hits[i];
        hit.lifetime += delta;
        const ratio = hit.lifetime / hit.maxLifetime;
        const scale = 1 - (1 - Math.sin((ratio * 4.5 * Math.PI) / 4)) ** 2;
        hit.mesh.scale.setScalar(scale * hit.maxScale);
        hit.mesh.material.opacity = 1 - Math.sin((ratio * 2 * Math.PI) / 4);
        if (ratio >= 1) {
            scene.remove(hit.mesh);
            hit.mesh.material.dispose();
            hits.splice(i--, 1);
        }
    }
}

// Click (without dragging) to throw a sphere from the camera.
const raycaster = new Raycaster();
const down = new Vector2();
renderer.domElement.addEventListener('pointerdown', (event) => down.set(event.clientX, event.clientY));
renderer.domElement.addEventListener('pointerup', (event) => {
    if (Math.abs(event.clientX - down.x) + Math.abs(event.clientY - down.y) > 2) return;
    raycaster.setFromCamera(new Vector2((event.clientX / innerWidth) * 2 - 1, -(event.clientY / innerHeight) * 2 + 1), camera);
    const ball = createSphere();
    ball.collider.center.copy(camera.position).addScaledVector(raycaster.ray.direction, 3);
    ball.velocity
        .set(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5)
        .addScaledVector(raycaster.ray.direction, 10 * Math.random() + 15)
        .multiplyScalar(0.5);
});

overlay.checkbox('display collider', false, (value) => (collider.visible = value));
overlay.checkbox('display BVH', false, (value) => (helper.visible = value));
overlay.checkbox('display parents', false, (value) => {
    helper.displayParents = value;
    helper.update();
});
overlay.range('BVH depth', 1, 20, 1, 10, (value) => {
    helper.depth = value;
    helper.update();
});
overlay.range('physics steps', 0, 30, 1, params.physicsSteps, (value) => (params.physicsSteps = value));
overlay.range('gravity', -100, 100, 0.01, params.gravity, (value) => (params.gravity = value));
overlay.range('simulation speed', 0, 5, 0.01, params.simulationSpeed, (value) => (params.simulationSpeed = value));
overlay.range('sphere size', 0.2, 5, 0.1, params.sphereSize, (value) => (params.sphereSize = value));
overlay.checkbox('pause', params.pause, (value) => (params.pause = value));
overlay.button('step', () => update(0.016));
overlay.button('explode', () => {
    for (const ball of spheres) ball.velocity.addScaledVector(tempVec.copy(ball.collider.center).setY(ball.collider.center.y + 10).normalize(), 120);
});
overlay.button('reset', () => {
    while (spheres.length) removeSphere(0);
    for (const hit of hits) {
        scene.remove(hit.mesh);
        hit.mesh.material.dispose();
    }
    hits.length = 0;
});
overlay.note('Click to throw a sphere (drag to orbit). Every step each sphere asks the tree for the triangles near it (shapecast) and is pushed out of them.');

/** Smoke: 40 spheres dropped on the bowl, 4 simulated seconds; none may fall through the floor or go NaN. */
function dropCheck(): Record<string, number> {
    let seed = 9;
    const random = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
    for (let i = 0; i < 40; i++) {
        const ball = createSphere(random);
        ball.collider.center.set((random() - 0.5) * 16, 6 + random() * 8, (random() - 0.5) * 16);
    }
    for (let i = 0; i < 4 * 60; i++) update(1 / 60);
    let lowest = Infinity;
    for (const ball of spheres) {
        const { center, radius } = ball.collider;
        if (!Number.isFinite(center.x + center.y + center.z)) throw new Error('a sphere went NaN');
        lowest = Math.min(lowest, center.y - radius);
    }
    // The bowl's floor is at 0 give or take the noise's 0.6: a sphere below -1 went through it.
    if (spheres.length !== 40 || lowest < -1) throw new Error(`spheres fell through the environment: ${spheres.length} of 40 left, lowest bottom at ${lowest.toFixed(3)}`);
    if (environmentContacts < 100) throw new Error(`only ${environmentContacts} sphere-environment contacts`);
    return { spheres: spheres.length, lowestBottom: lowest, environmentContacts };
}

let frame = 0;
app.loop((_now, delta) => {
    frame++;
    if (!params.pause) update(params.simulationSpeed * delta);
    if (frame % 10 === 0) {
        overlay.set('spheres', spheres.length);
        overlay.set('physics', `${lastStepMs.toFixed(2)} ms per frame`);
    }
    if (isSmoke && frame === 10 && !model) {
        const result = dropCheck();
        void ready(app.device, { ...overlay.snapshot(), ...result });
    } else if (isSmoke && frame === 10) void ready(app.device, overlay.snapshot());
});

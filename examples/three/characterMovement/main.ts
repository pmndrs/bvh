/**
 * three/characterMovement: three-mesh-bvh's `characterMovement` example on our
 * tree.
 *
 * A level is baked into one static collider with `StaticGeometryGenerator`
 * (ours, a port of three-mesh-bvh's) and given a tree (`computeBoundsTree`,
 * tinybvh). The player is a capsule (a segment and a radius): each physics
 * step it falls and walks, then `boundsTree.shapecast` gathers the triangles
 * whose boxes touch the capsule's box, and each triangle near enough pushes
 * the capsule out along `ExtendedTriangle.closestPointToSegment`. A step that
 * pushed it mostly up means it stands on ground. WASD to walk, space to jump,
 * orbit to look; first person in the controls.
 *
 * The original loads a dungeon glTF from the network; this page builds a
 * level from three's primitives (floor, walls, ramps, stairs, platforms, a
 * knot), so it needs no asset. `?model=` bakes a glTF of yours instead.
 *
 * Ported from three-mesh-bvh's example/characterMovement.js (v0.9.18, MIT,
 * Garrett Johnson); see THIRD_PARTY.md. Changed: the level, our shell and
 * overlay, no shadows, the smoke check.
 */

import { Box3, BoxGeometry, CylinderGeometry, Group, Line3, Matrix4, Mesh, MeshBasicMaterial, MeshStandardMaterial, TorusKnotGeometry, Vector3, type BufferGeometry, type Object3D } from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { computeBoundsTree, MeshBVHHelper, prepareBoundsTrees, StaticGeometryGenerator, type BvhBoundsTree } from '@pmndrs/bvh/three';
import { createApp } from '../../shared/app.ts';
import { loadGltf, requestedModel } from '../../shared/loader.ts';
import { layoutInfo } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

await prepareBoundsTrees();
const app = await createApp('Character movement', 'three/characterMovement');
const { scene, camera, controls, overlay } = app;

const params = { firstPerson: false, gravity: -30, playerSpeed: 10, physicsSteps: 5 };

// ---- The level: three's primitives, one material per colour, all indexed (the generator merges indexed geometry).
function level(): Object3D {
    const root = new Group();
    const add = (geometry: BufferGeometry, color: number, x: number, y: number, z: number, rx = 0, ry = 0, rz = 0) => {
        const mesh = new Mesh(geometry, new MeshStandardMaterial({ color }));
        mesh.position.set(x, y, z);
        mesh.rotation.set(rx, ry, rz);
        root.add(mesh);
        return mesh;
    };
    add(new BoxGeometry(60, 1, 60), 0x5c6b7a, 0, -0.5, 0); // floor, top at 0
    add(new BoxGeometry(30, 6, 1), 0x8a7f72, 0, 3, -12); // wall north, face at z = -11.5
    add(new BoxGeometry(1, 6, 24), 0x8a7f72, 15, 3, 0); // wall east
    add(new BoxGeometry(1, 4, 24), 0x8a7f72, -15, 2, 0); // wall west
    // A ramp whose top runs from the floor at z 10 up to the platform's edge at z -2 (3.4 up).
    const slope = Math.atan2(3.4, 12);
    add(new BoxGeometry(4, 1, Math.hypot(12, 3.4) + 0.4), 0xc8695a, -8, 1.7 - 0.5 * Math.cos(slope), 4 - 0.5 * Math.sin(slope), slope);
    add(new BoxGeometry(8, 1, 8), 0xb0a080, -8, 2.9, -6); // platform, top at 3.4
    for (let i = 0; i < 12; i++) add(new BoxGeometry(4, 0.25 * (i + 1), 0.6), 0x9fb7cc, 8, 0.125 * (i + 1), 6 - i * 0.6); // stairs
    add(new BoxGeometry(4, 1, 4), 0x9fb7cc, 8, 2.5, -2.9); // landing, top at 3
    for (const [x, z] of [[-3, 0], [3, -3], [0, 6]]) add(new CylinderGeometry(0.6, 0.6, 5, 24), 0x7a8f6f, x, 2.5, z); // pillars
    add(new TorusKnotGeometry(1.4, 0.45, 160, 20), 0x8fb3ff, 0, 1.6, -7); // something round to slide along
    return root;
}

const model = requestedModel();
let environment: Object3D;
if (model) {
    environment = (await loadGltf(model)).root;
    // Centre it on the origin at a walkable size, as the original does with its dungeon.
    const box = new Box3().setFromObject(environment);
    const size = box.getSize(new Vector3()).length();
    environment.scale.setScalar(60 / size);
    environment.updateMatrixWorld(true);
    box.setFromObject(environment);
    environment.position.sub(box.getCenter(new Vector3()));
} else environment = level();
scene.add(environment);
environment.updateMatrixWorld(true);

// One static geometry (positions only, world space), one tree.
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

// ---- The player: a rounded box drawn, a capsule simulated.
const player = new Mesh(new RoundedBoxGeometry(1, 2, 1, 10, 0.5), new MeshStandardMaterial({ color: 0xffc878 }));
player.geometry.translate(0, -0.5, 0);
const capsule = { radius: 0.5, segment: new Line3(new Vector3(), new Vector3(0, -1, 0)) };
scene.add(player);

let playerOnGround = false;
const velocity = new Vector3();
const up = new Vector3(0, 1, 0);
const keys = { forward: false, back: false, left: false, right: false };
const tempVector = new Vector3();
const tempVector2 = new Vector3();
const tempBox = new Box3();
const tempMatrix = new Matrix4();
const tempSegment = new Line3();
const spawn = new Vector3(model ? 0 : 0, model ? 10 : 3, model ? 0 : 8);
let contacts = 0;

function reset(): void {
    velocity.set(0, 0, 0);
    player.position.copy(spawn);
    camera.position.sub(controls.target);
    controls.target.copy(player.position);
    camera.position.add(player.position);
    controls.update();
}
camera.position.set(10, 10, 18);
reset();

/** One physics step, three-mesh-bvh's updatePlayer: `angle` is the view's azimuth (forward is -z turned by it). */
function updatePlayer(delta: number, angle: number): void {
    if (playerOnGround) velocity.y = delta * params.gravity;
    else velocity.y += delta * params.gravity;
    player.position.addScaledVector(velocity, delta);
    const walk = (x: number, z: number) => player.position.addScaledVector(tempVector.set(x, 0, z).applyAxisAngle(up, angle), params.playerSpeed * delta);
    if (keys.forward) walk(0, -1);
    if (keys.back) walk(0, 1);
    if (keys.left) walk(-1, 0);
    if (keys.right) walk(1, 0);
    player.updateMatrixWorld();

    // The capsule in the collider's space, and its box.
    tempBox.makeEmpty();
    tempMatrix.copy(collider.matrixWorld).invert();
    tempSegment.copy(capsule.segment);
    tempSegment.start.applyMatrix4(player.matrixWorld).applyMatrix4(tempMatrix);
    tempSegment.end.applyMatrix4(player.matrixWorld).applyMatrix4(tempMatrix);
    tempBox.expandByPoint(tempSegment.start);
    tempBox.expandByPoint(tempSegment.end);
    tempBox.min.addScalar(-capsule.radius);
    tempBox.max.addScalar(capsule.radius);

    tree.shapecast({
        intersectsBounds: (box) => box.intersectsBox(tempBox),
        intersectsTriangle: (triangle) => {
            // Push the capsule out of each triangle it overlaps.
            const distance = triangle.closestPointToSegment(tempSegment, tempVector, tempVector2);
            if (distance < capsule.radius) {
                const depth = capsule.radius - distance;
                const direction = tempVector2.sub(tempVector).normalize();
                tempSegment.start.addScaledVector(direction, depth);
                tempSegment.end.addScaledVector(direction, depth);
                contacts++;
            }
        },
    });

    // The capsule's segment start is the player's origin.
    const newPosition = tempVector.copy(tempSegment.start).applyMatrix4(collider.matrixWorld);
    const deltaVector = tempVector2.subVectors(newPosition, player.position);
    // Pushed mostly up: standing on something.
    playerOnGround = deltaVector.y > Math.abs(delta * velocity.y * 0.25);
    const offset = Math.max(0, deltaVector.length() - 1e-5);
    deltaVector.normalize().multiplyScalar(offset);
    player.position.add(deltaVector);
    if (!playerOnGround) {
        deltaVector.normalize();
        velocity.addScaledVector(deltaVector, -deltaVector.dot(velocity));
    } else velocity.set(0, 0, 0);

    // The camera follows.
    camera.position.sub(controls.target);
    controls.target.copy(player.position);
    camera.position.add(player.position);
    if (player.position.y < -25) reset();
}

addEventListener('keydown', (event) => {
    if (event.code === 'KeyW') keys.forward = true;
    if (event.code === 'KeyS') keys.back = true;
    if (event.code === 'KeyA') keys.left = true;
    if (event.code === 'KeyD') keys.right = true;
    if (event.code === 'Space' && playerOnGround) {
        velocity.y = 10;
        playerOnGround = false;
    }
});
addEventListener('keyup', (event) => {
    if (event.code === 'KeyW') keys.forward = false;
    if (event.code === 'KeyS') keys.back = false;
    if (event.code === 'KeyA') keys.left = false;
    if (event.code === 'KeyD') keys.right = false;
});

overlay.checkbox('first person', params.firstPerson, (value) => {
    params.firstPerson = value;
    if (!value) camera.position.sub(controls.target).normalize().multiplyScalar(10).add(controls.target);
});
overlay.checkbox('display collider', false, (value) => (collider.visible = value));
overlay.checkbox('display BVH', false, (value) => (helper.visible = value));
overlay.range('BVH depth', 1, 20, 1, 10, (value) => {
    helper.depth = value;
    helper.update();
});
overlay.range('physics steps', 0, 30, 1, params.physicsSteps, (value) => (params.physicsSteps = value));
overlay.range('gravity', -100, 100, 0.01, params.gravity, (value) => (params.gravity = value));
overlay.range('player speed', 1, 20, 0.1, params.playerSpeed, (value) => (params.playerSpeed = value));
overlay.button('reset', reset);
overlay.note('WASD to walk, space to jump, drag to look. The capsule is pushed out of every triangle the tree hands back for its box (shapecast, ExtendedTriangle.closestPointToSegment).');

/** Smoke check: the player lands on the floor, walks into the north wall and stops at it, and walks up the ramp onto the platform. */
function walkCheck(): Record<string, number> {
    const run = (seconds: number, angle: number) => {
        for (let i = 0; i < seconds * 60; i++) for (let s = 0; s < 5; s++) updatePlayer(1 / 60 / 5, angle);
    };
    // A clear lane along the west wall, from the south to the north wall.
    reset();
    player.position.set(-13, 3, 8);
    run(1.5, 0);
    const landedY = player.position.y;
    if (!playerOnGround || Math.abs(landedY - 1.5) > 0.05) throw new Error(`the player should stand on the floor at y 1.5 (capsule bottom on y 0), not ${landedY.toFixed(3)} (on ground: ${playerOnGround})`);
    keys.forward = true;
    run(3, 0); // forward is -z: into the north wall's face at z -11.5
    keys.forward = false;
    const wallZ = player.position.z;
    if (wallZ < -11.5 + capsule.radius - 0.05 || wallZ > -10) throw new Error(`the player should stop at the north wall (z about -11), not at z ${wallZ.toFixed(3)}`);
    // The ramp: from its foot, walk -z up onto the platform (top at y 3.4, so y about 4.9).
    player.position.set(-8, 2, 12);
    velocity.set(0, 0, 0);
    run(0.5, 0);
    keys.forward = true;
    run(1.6, 0);
    keys.forward = false;
    run(0.5, 0);
    const climbedY = player.position.y;
    if (Math.abs(climbedY - 4.9) > 0.1 || player.position.z > -2) throw new Error(`the player should have walked up the ramp onto the platform (y about 4.9, z below -2), not to y ${climbedY.toFixed(3)}, z ${player.position.z.toFixed(2)}`);
    return { landedY, wallZ, climbedY, contacts };
}

let frame = 0;
app.loop((_now, delta) => {
    frame++;
    if (params.firstPerson) {
        controls.maxPolarAngle = Math.PI;
        controls.minDistance = controls.maxDistance = 1e-4;
    } else {
        controls.maxPolarAngle = Math.PI / 2;
        controls.minDistance = 1;
        controls.maxDistance = 20;
    }
    const start = performance.now();
    const steps = params.physicsSteps;
    for (let i = 0; i < steps; i++) updatePlayer(delta / steps, controls.getAzimuthalAngle());
    const ms = performance.now() - start;
    if (frame % 10 === 0) {
        overlay.set('physics', `${ms.toFixed(3)} ms per frame (${steps} shapecasts)`);
        overlay.set('on ground', playerOnGround ? 'yes' : 'no');
        overlay.set('position', `${player.position.x.toFixed(1)}, ${player.position.y.toFixed(1)}, ${player.position.z.toFixed(1)}`);
    }
    if (isSmoke && frame === 20 && !model) {
        const result = walkCheck();
        reset();
        void ready(app.device, { ...overlay.snapshot(), ...result });
    } else if (isSmoke && frame === 20) void ready(app.device, overlay.snapshot());
});

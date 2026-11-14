/**
 * three/skinned: skinned meshes whose trees follow their pose on the GPU.
 *
 * three skins in the vertex shader, so nothing of the pose is in a buffer a
 * BVH could read. With a `renderer`, `SceneTreesObject3DSync` gives every
 * `SkinnedMesh` a `BvhPosePass` (a TSL compute pass with three's skinning
 * math) that writes the posed vertices into a storage buffer, and
 * `SceneTrees.setPose` refits the mesh's tree from that buffer
 * (`GpuMeshRefit`), rebuilding it when the refit has drifted too far. Nothing
 * returns to the CPU. Picking (one GPU ray through `TraceBatchKernel`) keeps
 * hitting the moving mesh.
 *
 * The characters are tubes skinned to a chain of bones, made from three's
 * `SphereGeometry`, so the page needs no asset.
 */

import { Bone, Float32BufferAttribute, Group, Mesh, MeshBasicMaterial, MeshStandardMaterial, Raycaster, Skeleton, SkinnedMesh, SphereGeometry, Uint16BufferAttribute, Vector2, Vector3, type BufferGeometry, type Intersection } from 'three';
import { GpuRays, SceneTrees } from '@pmndrs/bvh';
import { SceneTreesObject3DSync } from '@pmndrs/bvh/three';
import { cameraRay, createApp } from '../../shared/app.ts';
import { SceneTreesLog } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

const app = await createApp('Skinned', 'three/skinned');
const { device, renderer, scene, camera, controls, overlay } = app;
camera.position.set(0, 3, 14);
controls.target.set(0, 1.5, 0);

const params = new URLSearchParams(location.search);
const COUNT = Number(params.get('characters') ?? 9);
const BONES = 6;

/** A tube along y (detail: rings of the sphere), skinned to `bones` bones. */
function tube(rings: number, bones: number): BufferGeometry {
    const geometry = new SphereGeometry(0.4, 32, rings);
    geometry.scale(1, 5, 1);
    geometry.translate(0, 2, 0);
    const position = geometry.attributes.position;
    const skinIndex = new Uint16Array(position.count * 4);
    const skinWeight = new Float32Array(position.count * 4);
    for (let i = 0; i < position.count; i++) {
        const t = Math.min(bones - 1.001, Math.max(0, (position.getY(i) / 4) * (bones - 1)));
        const bone = Math.floor(t);
        const f = t - bone;
        skinIndex.set([bone, Math.min(bones - 1, bone + 1), 0, 0], i * 4);
        skinWeight.set([1 - f, f, 0, 0], i * 4);
    }
    geometry.setAttribute('skinIndex', new Uint16BufferAttribute(skinIndex, 4));
    geometry.setAttribute('skinWeight', new Float32BufferAttribute(skinWeight, 4));
    return geometry;
}

function character(geometry: BufferGeometry, color: number): SkinnedMesh {
    const chain: Bone[] = [];
    for (let b = 0; b < BONES; b++) {
        const bone = new Bone();
        bone.position.y = b === 0 ? 0 : 4 / (BONES - 1);
        if (b > 0) chain[b - 1].add(bone);
        chain.push(bone);
    }
    const mesh = new SkinnedMesh(geometry, new MeshStandardMaterial({ color, roughness: 0.55 }));
    mesh.add(chain[0]);
    mesh.updateMatrixWorld(true);
    mesh.bind(new Skeleton(chain));
    return mesh;
}

const geometry = tube(Number(params.get('rings') ?? 400), BONES);
// What the trees hold: the characters and the ground (not the pick marker).
const world = new Group();
scene.add(world);
const characters: SkinnedMesh[] = [];
for (let i = 0; i < COUNT; i++) {
    const mesh = character(geometry, [0x8fb3ff, 0xb0e0a8, 0xf0b3d0][i % 3]);
    mesh.position.set(((i % 3) - 1) * 3.2, 0, (Math.floor(i / 3) - 1) * -3.2);
    characters.push(mesh);
    world.add(mesh);
}
const ground = new Mesh(new SphereGeometry(40, 32, 16, 0, Math.PI * 2, 0, 0.12), new MeshStandardMaterial({ color: 0x2a2f38 }));
ground.position.y = -39.7;
world.add(ground);
const marker = new Mesh(new SphereGeometry(0.09, 12, 8), new MeshBasicMaterial({ color: 0xffc878 }));
marker.visible = false;
scene.add(marker);

let speed = 1;
let time = 0;
function animate(delta: number): void {
    time += delta * speed;
    characters.forEach((mesh, m) => {
        mesh.skeleton.bones.forEach((bone, b) => {
            if (b === 0) return;
            const phase = time * 1.8 + m * 0.7 + b * 0.45;
            bone.rotation.set(Math.sin(phase) * 0.35, 0, Math.cos(phase * 0.8) * 0.3);
        });
    });
}
animate(0);
scene.updateMatrixWorld(true);

// SceneTrees with the renderer: every SkinnedMesh is posed on the GPU each update.
const log = new SceneTreesLog();
const trees = await SceneTrees.create(device, { label: 'skinned', onTree: log.onTree });
// The pose passes compile asynchronously: until they have, update() leaves each character's tree in its
// last pose (here, the rest pose) and counts it in poseCounts.waiting. `await sync.ready` would have them
// posed from the first update; the page doesn't wait, and the first frames' trees are in the rest pose.
const sync = new SceneTreesObject3DSync(trees, world, { renderer });
const rays = await GpuRays.create(device, trees);
function updateTrees(): number {
    const start = performance.now();
    scene.updateMatrixWorld();
    sync.update(); // transforms, then each posed object's pass, then SceneTrees.setPose
    const encoder = device.createCommandEncoder({ label: 'skinned update' });
    trees.update(encoder); // the refits (and drift rebuilds), the top level
    device.queue.submit([encoder.finish()]);
    return performance.now() - start;
}
updateTrees();
await trees.whenBuilt();
updateTrees();
overlay.build(log.build(trees));
overlay.tree(log.tree(trees, sync.meshHandles));
overlay.set('characters', `${COUNT} × ${(geometry.index!.count / 3).toLocaleString('en-US')} triangles`);
overlay.range('speed', 0, 3, 0.1, speed, (value) => (speed = value));
// SceneTrees.rebuild: each character's tree rebuilt now from its current pose, one per update (the drift rule does it on its own).
overlay.button('rebuild every character now', () => {
    let asked = 0;
    for (const handle of sync.meshHandles) if (trees.meshStatus(handle).deforming && trees.rebuild(handle)) asked++;
    overlay.set('rebuilds asked', asked > 0 ? asked : 'none (no GPU builder, or no pose yet)');
});
overlay.note('Hover a character: the GPU ray walks the refit tree, so the marker sits on the skinned surface. three\'s CPU raycast (which skins on the CPU) is the check.');

const pointer = new Vector2(2, 2);
addEventListener('pointermove', (event) => pointer.set((event.clientX / innerWidth) * 2 - 1, -(event.clientY / innerHeight) * 2 + 1));
if (isSmoke) pointer.set(0, 0.1);

/** Smoke check: GPU picks against three's own raycast, which skins each vertex on the CPU. */
async function checkAgainstThree(): Promise<number> {
    const count = 48;
    const packed = new Float32Array(count * 8);
    const ndc: Vector2[] = [];
    for (let i = 0; i < count; i++) {
        const mesh = characters[i % characters.length];
        const point = new Vector3(0, 0.5 + Math.random() * 3, 0).applyMatrix4(mesh.matrixWorld).project(camera);
        ndc.push(new Vector2(point.x + (Math.random() - 0.5) * 0.01, point.y));
        cameraRay(camera, ndc[i], packed, i);
    }
    const result = await rays.traceClosest(packed);
    for (const mesh of characters) {
        mesh.skeleton.update();
        // three culls a SkinnedMesh's raycast with its boundingSphere, which it computes once (the renderer's
        // frustum culling does, at the first frame) and never again: refresh it for the current pose. A stale
        // one made three miss rays that hit the posed surface well inside a triangle.
        mesh.computeBoundingSphere();
    }
    const raycaster = new Raycaster();
    let agreed = 0;
    for (let i = 0; i < count; i++) {
        raycaster.setFromCamera(ndc[i], camera);
        const theirs: Intersection[] = raycaster.intersectObjects(world.children, false);
        const gpu = result.hit(i);
        const cpu = theirs[0];
        if (!gpu !== !cpu) throw new Error(`ray ${i}: GPU ${gpu ? `hit at ${gpu.t}` : 'missed'}, three's raycast ${cpu ? `hit at ${cpu.distance}` : 'missed'}`);
        if (gpu && cpu && Math.abs(gpu.t - cpu.distance) > 1e-3 * Math.max(1, cpu.distance)) throw new Error(`ray ${i}: GPU t ${gpu.t}, three's raycast ${cpu.distance}`);
        if (gpu) agreed++;
    }
    if (agreed < count / 2) throw new Error(`only ${agreed} of ${count} rays hit a character`);
    return agreed;
}

let frame = 0;
let updateMs = 0;
app.loop(async (_now, delta) => {
    frame++;
    animate(delta);
    updateMs = updateMs * 0.9 + updateTrees() * 0.1;
    if (frame % 10 === 0) {
        const deforming = trees.deformingCounts;
        overlay.set('pose + refit (CPU)', `${updateMs.toFixed(2)} ms per frame`);
        overlay.set('refits', deforming.refits);
        overlay.set('drift rebuilds', deforming.driftRebuilds);
        overlay.set('boxes moved', sync.poseCounts.boundsMoved);
        overlay.build(log.build(trees));
    }
    if (!rays.pending && pointer.x <= 1) {
        const result = await rays.traceClosest(cameraRay(camera, pointer));
        const hit = result.hit(0);
        marker.visible = hit !== null;
        if (hit) {
            const packed = cameraRay(camera, pointer);
            marker.position.set(packed[0] + packed[4] * hit.t, packed[1] + packed[5] * hit.t, packed[2] + packed[6] * hit.t);
            const source = sync.sourceOf(hit.instance);
            overlay.set('picked', source ? `character ${characters.indexOf(source.object as SkinnedMesh)}, triangle ${hit.triangle}` : '-');
        } else overlay.set('picked', '-');
    }
    if (isSmoke && frame === 90) {
        // Check on a frozen pose, one the GPU trees were refit to.
        speed = 0;
        animate(0);
        updateTrees();
        const agreed = await checkAgainstThree();
        await ready(device, { ...overlay.snapshot(), agreedWithThree: agreed });
    }
});

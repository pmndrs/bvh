/**
 * Skinned and morphed meshes through `@pmndrs/bvh/three` on a WebGPURenderer.
 *
 * The scene: an indexed SkinnedMesh with two groups (an array of materials)
 * and two morph targets, a non-indexed SkinnedMesh, a second SkinnedMesh on
 * the first's geometry with its own skeleton, a morph-only mesh (absolute
 * targets) and a static knot. `SceneTreesObject3DSync` (with the renderer)
 * adds the posed objects as deforming meshes and, each update, runs their
 * `BvhPosePass` (a TSL compute pass, three's skinning and morph math) into
 * `SceneTrees.setPose`, which refits their trees on the GPU.
 *
 * The animation bends the bones, swings the influences and walks the first
 * character's root bone away (its deform box must follow), then holds a bend
 * long enough for drift rebuilds.
 *
 * The oracle: three's own CPU pose (`getVertexPosition`: morphs, then
 * `SkinnedMesh.applyBoneTransform`).
 *
 * - The pass's posed vertices, read back, must match it (within f32 rounding).
 * - Rays traced on the GPU through the SceneTrees buffers must find what a CPU
 *   walk finds over meshes posed on the CPU with that math (tinybvh trees).
 *
 * Checked at several frames, records and indexed leaves.
 */

import { Bone, BufferAttribute, BufferGeometry, DoubleSide, Group, Mesh, MeshBasicMaterial, type Object3D, Skeleton, SkinnedMesh, SphereGeometry, TorusKnotGeometry, Uint16BufferAttribute, Float32BufferAttribute, Vector3 } from 'three';
import { WebGPURenderer } from 'three/webgpu';
import { SceneTrees } from '../../../../src/scene/scene-trees.ts';
import { createTinyBvhBuilder } from '../../../../src/tinybvh.ts';
import { buildMeshBvhWithTinyBvh } from '../../../../src/trace/mesh-bvh-topology.ts';
import { createRandom } from '../../../../src/trace/oracle-fixtures.ts';
import type { TraceRay } from '../../../../src/trace/rays.ts';
import { buildTraceScene } from '../../../../src/trace/scene-assembly.ts';
import { TraceBatchKernel } from '../../../../src/trace/trace-gpu.ts';
import { SceneTreesObject3DSync, bvhSceneFromObject3D, deviceFromRenderer } from '../../../../src/three/index.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { compareWithReference, readBuffer, readScene, traceOnGpu } from '../../scene-check.ts';

const startTime = performance.now();
const logLines: string[] = [];
const log = (line: string) => logLines.push(line);
let deviceErrors: string[] = [];
let adapter = null as ReturnType<typeof summarizeAdapterInfo> | null;

const material = () => new MeshBasicMaterial({ side: DoubleSide });

/** A tube along y skinned to a chain of bones, with two relative morph targets. */
function tubeGeometry(random: () => number, bones: number): BufferGeometry {
    const geometry = new SphereGeometry(0.45, 16, 40);
    geometry.scale(1, 4, 1);
    const position = geometry.attributes.position;
    const count = position.count;
    const skinIndex = new Uint16Array(count * 4);
    const skinWeight = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
        const t = Math.min(bones - 1.001, Math.max(0, ((position.getY(i) + 1.8) / 3.6) * (bones - 1)));
        const bone = Math.floor(t);
        skinIndex.set([bone, bone + 1, Math.min(bones - 1, bone + 2), 0], i * 4);
        const f = t - bone;
        skinWeight.set([(1 - f) * 0.9, f * 0.9, 0.1, 0], i * 4);
    }
    geometry.setAttribute('skinIndex', new Uint16BufferAttribute(skinIndex, 4));
    geometry.setAttribute('skinWeight', new Float32BufferAttribute(skinWeight, 4));
    const bulge = new Float32Array(count * 3);
    const ripple = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        const x = position.getX(i);
        const y = position.getY(i);
        const z = position.getZ(i);
        bulge.set([x * 0.7, 0, z * 0.7], i * 3);
        ripple.set([Math.sin(y * 3) * 0.15, random() * 0.02, Math.cos(y * 3) * 0.15], i * 3);
    }
    geometry.morphAttributes.position = [new BufferAttribute(bulge, 3), new BufferAttribute(ripple, 3)];
    geometry.morphTargetsRelative = true;
    geometry.clearGroups();
    const half = Math.floor(geometry.index!.count / 6) * 3;
    geometry.addGroup(0, half, 0);
    geometry.addGroup(half, Infinity, 1);
    return geometry;
}

function skinned(geometry: BufferGeometry, bones: number, materials: MeshBasicMaterial | MeshBasicMaterial[]): SkinnedMesh {
    const chain: Bone[] = [];
    for (let b = 0; b < bones; b++) {
        const bone = new Bone();
        bone.position.y = b === 0 ? -1.8 : 3.6 / (bones - 1);
        if (b > 0) chain[b - 1].add(bone);
        chain.push(bone);
    }
    const mesh = new SkinnedMesh(geometry, materials);
    mesh.add(chain[0]);
    mesh.updateMatrixWorld(true);
    mesh.bind(new Skeleton(chain));
    return mesh;
}

interface Characters {
    readonly root: Group;
    readonly skinnedMeshes: SkinnedMesh[];
    readonly morphed: Mesh;
    readonly posed: Mesh[];
}

function buildScene(random: () => number): Characters {
    const root = new Group();
    const tube = tubeGeometry(random, 5);
    const a = skinned(tube, 5, [material(), material()]);
    a.position.set(-2, 0, 0);
    const b = skinned(tube, 5, material()); // the same geometry, its own skeleton and pose
    b.position.set(2, 0.5, -1);
    b.rotation.z = 0.6;
    const flatGeometry = tubeGeometry(random, 4).toNonIndexed();
    flatGeometry.clearGroups();
    const c = skinned(flatGeometry, 4, material());
    c.position.set(0, 0, 2.5);
    c.scale.set(0.8, 1.2, 0.8);
    const morphGeometry = new SphereGeometry(1, 20, 14);
    const p = morphGeometry.attributes.position;
    const squash = new Float32Array(p.count * 3);
    for (let i = 0; i < p.count; i++) squash.set([p.getX(i) * 1.5, p.getY(i) * 0.4, p.getZ(i) * 1.2], i * 3);
    morphGeometry.morphAttributes.position = [new BufferAttribute(squash, 3)];
    morphGeometry.morphTargetsRelative = false;
    const morphed = new Mesh(morphGeometry, material());
    morphed.position.set(0, 3, -2);
    const knot = new Mesh(new TorusKnotGeometry(1, 0.3, 80, 10), material());
    knot.position.set(0, -3, 0);
    root.add(a, b, c, morphed, knot);
    root.updateMatrixWorld(true);
    return { root, skinnedMeshes: [a, b, c], morphed, posed: [a, b, c, morphed] };
}

/** Frame `frame`'s pose: swaying bones, swinging influences, a's root bone walking; from frame 120 a held bend. */
function animate(scene: Characters, frame: number): void {
    scene.skinnedMeshes.forEach((mesh, m) => {
        const held = frame >= 120;
        mesh.skeleton.bones.forEach((bone, b) => {
            if (b === 0) return;
            const phase = frame * 0.07 + m + b * 0.5;
            bone.rotation.set(held ? 0.5 : Math.sin(phase) * 0.6, Math.cos(phase * 0.7) * 0.3, held ? 0.35 : Math.sin(phase * 1.3) * 0.5);
        });
        mesh.morphTargetInfluences![0] = 0.5 + 0.5 * Math.sin(frame * 0.05 + m);
        mesh.morphTargetInfluences![1] = Math.cos(frame * 0.09 + m);
    });
    // Root motion on a: its pose leaves the box it started in.
    scene.skinnedMeshes[0].skeleton.bones[0].position.set(Math.min(frame, 100) * 0.04, -1.8, Math.min(frame, 100) * -0.02);
    scene.morphed.morphTargetInfluences![0] = 0.5 + 0.5 * Math.sin(frame * 0.11);
    scene.root.updateMatrixWorld(true);
}

/** three's CPU pose of `mesh` as a plain mesh with the same index, groups, materials and world matrix. */
function cpuPosed(mesh: Mesh): Mesh {
    const source = mesh.geometry;
    const count = source.attributes.position.count;
    const positions = new Float32Array(count * 3);
    const v = new Vector3();
    for (let i = 0; i < count; i++) mesh.getVertexPosition(i, v).toArray(positions, i * 3);
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(positions, 3));
    if (source.index) geometry.setIndex(source.index);
    for (const group of source.groups) geometry.addGroup(group.start, group.count, group.materialIndex);
    const posed = new Mesh(geometry, mesh.material);
    posed.matrixAutoUpdate = false;
    posed.matrixWorld.copy(mesh.matrixWorld);
    return posed;
}

/** Rays from around the scene toward points near the posed surfaces (most hit something). */
function aimedRays(random: () => number, targets: Vector3[], count: number): TraceRay[] {
    const rays: TraceRay[] = [];
    for (let i = 0; i < count; i++) {
        const origin = new Vector3((random() - 0.5) * 16, (random() - 0.5) * 16, (random() - 0.5) * 16);
        const target = targets[Math.floor(random() * targets.length)].clone().add(new Vector3(random() - 0.5, random() - 0.5, random() - 0.5).multiplyScalar(0.3));
        const direction = target.sub(origin).normalize();
        rays.push({ origin: origin.toArray(), direction: direction.toArray(), tMin: 0, tMax: 1e6 });
    }
    return rays;
}

try {
    const renderer = new WebGPURenderer();
    await renderer.init();
    const device = deviceFromRenderer(renderer);
    deviceErrors = watchDevice(device);
    adapter = summarizeAdapterInfo(device.adapterInfo, device.features);
    if (!device.features.has('subgroups')) log('no subgroups: SceneTrees skips poses; this page needs the refit');

    const failures: string[] = [];
    const results: Record<string, unknown> = {};
    const tinyBvh = await createTinyBvhBuilder();
    const kernel = { records: await TraceBatchKernel.create(device, {}), indexed: await TraceBatchKernel.create(device, { triangleLeaves: 'indexed' }) };

    for (const triangleLeaves of ['records', 'indexed'] as const) {
        const random = createRandom(91);
        const scene = buildScene(random);
        animate(scene, 0);
        const trees = await SceneTrees.create(device, { label: `three-skinned ${triangleLeaves}`, workers: { size: 2 }, triangleLeaves });
        const sync = new SceneTreesObject3DSync(trees, scene.root, { renderer });
        let boundsMoved = 0;
        const frame = () => {
            sync.update();
            boundsMoved += sync.poseCounts.boundsMoved;
            const encoder = device.createCommandEncoder();
            trees.update(encoder);
            device.queue.submit([encoder.finish()]);
        };
        frame();
        await trees.whenBuilt();
        frame();
        await trees.whenRefined();
        frame();

        // Handles by source, for the reference's instance indices.
        const handleOf = new Map<string, number>();
        for (let handle = 0; handle < 1024; handle++) {
            const source = sync.sourceOf(handle);
            if (source) handleOf.set(`${source.object.id}:${source.instanceId}:${source.group}`, handle);
        }

        const check = async (at: number) => {
            const phase = `${triangleLeaves} frame ${at}`;
            await device.queue.onSubmittedWorkDone();
            // 1. The pass's vertices against three's CPU pose.
            let worstVertex = 0;
            for (const mesh of scene.posed) {
                const pass = sync.posePassOf(mesh);
                if (!pass) {
                    failures.push(`${phase}: ${mesh.name || mesh.type} has no pose pass`);
                    continue;
                }
                const gpu = new Float32Array(await readBuffer(device, pass.buffer, pass.vertexCount * 16));
                const v = new Vector3();
                for (let i = 0; i < pass.vertexCount; i++) {
                    mesh.getVertexPosition(i, v);
                    const error = Math.max(Math.abs(gpu[i * 4] - v.x), Math.abs(gpu[i * 4 + 1] - v.y), Math.abs(gpu[i * 4 + 2] - v.z));
                    worstVertex = Math.max(worstVertex, error);
                }
            }
            if (!(worstVertex <= 1e-4)) failures.push(`${phase}: posed vertices differ from three's CPU pose by up to ${worstVertex}`);
            // 2. GPU hits against a CPU walk over CPU-posed meshes.
            const reference: Mesh[] = [];
            const originals = new Map<Object3D, Mesh>();
            for (const child of scene.root.children as Mesh[]) {
                const posed = scene.posed.includes(child) ? cpuPosed(child) : child;
                reference.push(posed);
                originals.set(posed, child);
            }
            const referenceScene = bvhSceneFromObject3D(reference);
            const instances = referenceScene.instances.map((instance, i) => {
                const source = referenceScene.sources[i];
                const handle = handleOf.get(`${originals.get(source.object)!.id}:${source.instanceId}:${source.group}`);
                if (handle === undefined) throw new Error(`${phase}: no sync instance for reference instance ${i}`);
                return { ...instance, instanceIndex: handle };
            });
            const referenceTrace = await buildTraceScene(referenceScene.meshes, instances, { build: (geometry) => buildMeshBvhWithTinyBvh(geometry, tinyBvh) });
            const targets: Vector3[] = [];
            for (const mesh of reference) {
                const position = mesh.geometry.attributes.position;
                for (let i = 0; i < position.count; i += 7) targets.push(new Vector3().fromBufferAttribute(position, i).applyMatrix4(mesh.matrixWorld));
            }
            const rays = aimedRays(createRandom(1000 + at), targets, 16384);
            const gpuHits = await traceOnGpu(device, kernel[triangleLeaves], trees.traceBuffers, rays);
            const gpuScene = await readScene(device, trees.buffers, { triangleLeaves: trees.triangleLeaves, positionLayout: trees.positionLayout });
            const { disagreements, count, hits } = compareWithReference(rays, gpuHits, gpuScene, referenceTrace);
            const counts = trees.deformingCounts;
            results[phase] = { hits, disagreements: count, worstVertex, refits: counts.refits, driftRebuilds: counts.driftRebuilds, boundsMoved, posesSkipped: counts.posesSkipped };
            log(`${phase}: ${hits} hits, ${count} disagreements, worst vertex ${worstVertex.toExponential(2)}, ${counts.refits} refits, ${counts.driftRebuilds} drift rebuilds, ${boundsMoved} box moves`);
            for (const d of disagreements.slice(0, 3)) failures.push(`${phase}: ray ${d.ray}: ${d.what}`);
            if (hits < rays.length / 3) failures.push(`${phase}: only ${hits} hits`);
        };

        let at = 0;
        for (const stop of [1, 40, 100, 200]) {
            while (at < stop) {
                at++;
                animate(scene, at);
                frame();
            }
            // The refit trees, and the GPU top level's boxes, are written by the update before; trace after it.
            await check(at);
        }
        const counts = trees.deformingCounts;
        if (counts.posesSkipped > 0) failures.push(`${triangleLeaves}: ${counts.posesSkipped} poses skipped (no subgroups?)`);
        if (boundsMoved === 0) failures.push(`${triangleLeaves}: the walking root never moved a deform box`);
        results[`${triangleLeaves} counts`] = { ...counts, boundsMoved };
        sync.dispose();
        trees.dispose();
    }
    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data: results,
        adapter,
        deviceErrors,
        startTime,
        logLines,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? `${error.message}\n${error.stack}` : String(error), adapter, deviceErrors, startTime, logLines });
}

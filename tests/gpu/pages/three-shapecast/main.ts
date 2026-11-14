/**
 * BvhTraceNodes.getShapecastFn (PARITY 8.6, three-mesh-bvh's
 * `getShapecastFn`) on a WebGPURenderer: TSL compute kernels run shapecasts
 * whose hooks are three function nodes, over a three scene (meshes with
 * mirrored and scaled transforms, an InstancedMesh) in SceneTrees, loaded with
 * `updateSceneTreesAsync` (PARITY 8.4).
 *
 * - **Any:** a sphere cast with a TSL `Fn` (with a layout) as the box test and
 *   a `wgslFn` calling the module's `bvhClosestPointToTriangle` as the
 *   triangle test, stopping at the first triangle. It must give exactly what
 *   the library's own `bvhIntersectsSphere` gives (the same hooks, written in
 *   WGSL), except where a triangle lies within 1e-4 (relative) of the radius.
 * - **Count:** the same cast with `wgslFn` hooks that count, in a
 *   `var<private>`, the triangles touching the sphere and never stop, with a
 *   `keep` hook. It must match a CPU count over the scene's world triangles
 *   (three's `Triangle.closestPointToPoint`), up to the triangles within 1e-4
 *   of the radius.
 * - **Masked:** the any cast with mask 0 finds nothing.
 *
 * `?profile=ci` runs fewer spheres.
 */

import { Group, InstancedMesh, Matrix4, Mesh, MeshBasicMaterial, Object3D, SphereGeometry, BoxGeometry, TorusKnotGeometry, Triangle, Vector3, type BufferGeometry } from 'three';
import { Fn, If, code, float, instanceIndex, max, select, storage, uint, uvec4, vec3, wgslFn } from 'three/tsl';
import { StorageBufferAttribute, WebGPURenderer } from 'three/webgpu';
import { SceneTrees } from '../../../../src/scene/scene-trees.ts';
import { createRandom } from '../../../../src/trace/oracle-fixtures.ts';
import { BvhTraceNodes, SceneTreesObject3DSync, deviceFromRenderer, updateSceneTreesAsync, type BvhTslNode } from '../../../../src/three/index.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';

const startTime = performance.now();
const logLines: string[] = [];
const log = (line: string) => logLines.push(line);
let deviceErrors: string[] = [];
let adapter = null as ReturnType<typeof summarizeAdapterInfo> | null;
const profile = new URLSearchParams(location.search).get('profile');

function place(object: Object3D, random: () => number, extent = 8): Object3D {
    object.position.set((random() - 0.5) * extent, (random() - 0.5) * extent, (random() - 0.5) * extent);
    object.quaternion.setFromAxisAngle(new Vector3(random(), random(), random()).normalize(), random() * 6);
    const s = 0.6 + random();
    object.scale.set(s, random() < 0.25 ? -s : s * (0.7 + random() * 0.6), s);
    return object;
}

function buildScene(random: () => number): Group {
    const material = new MeshBasicMaterial();
    const root = new Group();
    const knot = new TorusKnotGeometry(0.8, 0.25, 48, 8);
    for (let i = 0; i < 5; i++) root.add(place(new Mesh(knot, material), random));
    root.add(place(new Mesh(new SphereGeometry(0.9, 16, 12).toNonIndexed(), material), random));
    const instanced = new InstancedMesh(new BoxGeometry(0.7, 0.5, 0.4, 2, 2, 2), material, 24);
    const dummy = new Object3D();
    for (let i = 0; i < 24; i++) {
        place(dummy, random, 9).updateMatrix();
        instanced.setMatrixAt(i, dummy.matrix);
    }
    root.add(instanced);
    root.updateMatrixWorld(true);
    return root;
}

/** Every triangle of the scene in world space, as three would draw it. */
function worldTriangles(root: Group): Triangle[] {
    const triangles: Triangle[] = [];
    const add = (geometry: BufferGeometry, matrix: Matrix4) => {
        const position = geometry.getAttribute('position');
        const index = geometry.index;
        const count = (index ? index.count : position.count) / 3;
        const corner = (t: number, k: number) => new Vector3().fromBufferAttribute(position, index ? index.getX(t * 3 + k) : t * 3 + k).applyMatrix4(matrix);
        for (let t = 0; t < count; t++) triangles.push(new Triangle(corner(t, 0), corner(t, 1), corner(t, 2)));
    };
    root.traverse((object) => {
        const mesh = object as Mesh;
        if (!mesh.isMesh) return;
        const instanced = object as InstancedMesh;
        if (instanced.isInstancedMesh) {
            for (let i = 0; i < instanced.count; i++) add(mesh.geometry, instanced.getMatrixAt(i, new Matrix4()).premultiply(mesh.matrixWorld));
        } else {
            add(mesh.geometry, mesh.matrixWorld);
        }
    });
    return triangles;
}

try {
    const renderer = new WebGPURenderer();
    await renderer.init();
    const device = deviceFromRenderer(renderer);
    deviceErrors = watchDevice(device);
    adapter = summarizeAdapterInfo(device.adapterInfo, device.features);
    const failures: string[] = [];
    const random = createRandom(91);
    const root = buildScene(random);
    const trees = await SceneTrees.create(device, { label: 'three-shapecast' });
    const sync = new SceneTreesObject3DSync(trees, root);
    const bvh = new BvhTraceNodes(renderer, trees, { queries: true });
    // PARITY 8.4: every tree built and placed before the first kernel runs.
    await updateSceneTreesAsync(trees, { sync, nodes: bvh, until: 'refined' });
    if (trees.counts.waitingForFirstTree !== 0) failures.push(`${trees.counts.waitingForFirstTree} meshes still had no tree after updateSceneTreesAsync`);

    // The spheres: centre and radius per vec4, from small to large.
    const count = profile === 'ci' ? 1024 : 8192;
    const spheres = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) spheres.set([(random() - 0.5) * 12, (random() - 0.5) * 12, (random() - 0.5) * 12, 0.05 + random() * random() * 2.5], i * 4);
    const sphereNode = storage(new StorageBufferAttribute(spheres, 4), 'vec4', count).toReadOnly();

    // Hooks. The box test is a TSL Fn (a pure function of its arguments, as three's layout cache needs).
    const sphereBounds = Fn(([shape, boxMin, boxMax]: BvhTslNode[]) => {
        const center = shape.xyz;
        const outside = max(max(boxMin.sub(center), center.sub(boxMax)), vec3(0));
        const distanceSq = outside.dot(outside);
        return select(distanceSq.lessThanEqual(shape.w.mul(shape.w)), distanceSq, float(-1));
    }).setLayout({ name: 'shapecastPageSphereBounds', type: 'float', inputs: [{ name: 'shape', type: 'vec4' }, { name: 'boxMin', type: 'vec3' }, { name: 'boxMax', type: 'vec3' }] });
    // The triangle test calls the module's helper, so it is a wgslFn that includes the module.
    const sphereTriangle = wgslFn(
        /* wgsl */ `
fn shapecastPageSphereTriangle(shape: vec4f, a: vec3f, b: vec3f, c: vec3f) -> bool {
    let offset = bvhClosestPointToTriangle(shape.xyz, a, b, c).point - shape.xyz;
    return dot(offset, offset) <= shape.w * shape.w;
}`,
        [bvh.module as never],
    );
    const anyCast = bvh.getShapecastFn({ name: 'shapecastPageAny', shape: 'vec4', intersectsBounds: sphereBounds, intersectsTriangle: sphereTriangle });

    // Counting: private state in a code node the hooks and the readers share.
    const counter = code('var<private> shapecastPageCounter: u32;');
    const countTriangle = wgslFn(
        /* wgsl */ `
fn shapecastPageCountTriangle(shape: vec4f, a: vec3f, b: vec3f, c: vec3f, instance: u32, triangle: u32) -> bool {
    let offset = bvhClosestPointToTriangle(shape.xyz, a, b, c).point - shape.xyz;
    if (dot(offset, offset) <= shape.w * shape.w) {
        shapecastPageCounter += 1u;
    }
    return false;
}`,
        [counter as never, bvh.module as never],
    );
    const countKeep = wgslFn(`fn shapecastPageKeep(shape: vec4f, score: f32) -> bool { return score <= shape.w * shape.w; }`);
    const countCast = bvh.getShapecastFn({ name: 'shapecastPageCount', shape: 'vec4', intersectsBounds: sphereBounds, intersectsTriangle: countTriangle, keep: countKeep });
    const resetCount = wgslFn(`fn shapecastPageResetCount() -> u32 { shapecastPageCounter = 0u; return 0u; }`, [counter as never]);
    const readCount = wgslFn(`fn shapecastPageReadCount() -> u32 { return shapecastPageCounter; }`, [counter as never]);
    const librarySphere = wgslFn(`fn shapecastPageLibrary(shape: vec4f) -> vec2u { return vec2u(select(0u, 1u, bvhIntersectsSphere(shape.xyz, shape.w)), bitcast<u32>(bvhClosestPoint(shape.xyz, 1e30).distance)); }`, [bvh.module as never]);

    const out = new StorageBufferAttribute(new Uint32Array(count * 4), 4);
    const outNode = storage(out, 'uvec4', count);
    const kernel = Fn(() => {
        If(instanceIndex.lessThan(count), () => {
            const shape = sphereNode.element(instanceIndex);
            const any = select(anyCast(shape), uint(1), uint(0)).toVar();
            const masked = select(anyCast(shape, uint(0)), uint(1), uint(0)).toVar();
            const reset = (resetCount() as unknown as BvhTslNode).toVar();
            const stopped = select(countCast(shape), uint(1), uint(0)).toVar();
            const counted = (readCount() as unknown as BvhTslNode).toVar();
            const library = (librarySphere(shape) as unknown as BvhTslNode).toVar();
            outNode.element(instanceIndex).assign(uvec4(any.add(masked.mul(2)).add(stopped.mul(4)).add(reset), counted, library.x, library.y));
        });
    })().compute(count, [64]);
    await renderer.computeAsync(kernel);
    const words = new Uint32Array(await renderer.getArrayBufferAsync(out));
    const floats = new Float32Array(words.buffer);

    // CPU: every world triangle's distance to each sphere.
    const triangles = worldTriangles(root);
    const nearest = new Vector3();
    let anyDisagreements = 0;
    let anyBoundary = 0;
    let countDisagreements = 0;
    let hits = 0;
    let total = 0;
    for (let i = 0; i < count; i++) {
        const center = new Vector3(spheres[i * 4], spheres[i * 4 + 1], spheres[i * 4 + 2]);
        const radius = spheres[i * 4 + 3];
        let inside = 0;
        let boundary = 0;
        for (const triangle of triangles) {
            const distance = triangle.closestPointToPoint(center, nearest).distanceTo(center);
            if (Math.abs(distance - radius) <= 1e-4 * Math.max(1, radius, center.length())) boundary++;
            else if (distance <= radius) inside++;
        }
        const flags = words[i * 4];
        const any = flags & 1;
        const masked = (flags >> 1) & 1;
        const stopped = (flags >> 2) & 1;
        const counted = words[i * 4 + 1];
        const library = words[i * 4 + 2];
        const closest = floats[i * 4 + 3];
        hits += any;
        total += counted;
        if (masked !== 0) failures.push(`sphere ${i}: the cast with mask 0 found a triangle`);
        if (stopped !== 0) failures.push(`sphere ${i}: the counting cast stopped`);
        if (any !== library) {
            if (Math.abs(closest - radius) <= 1e-4 * Math.max(1, radius, center.length())) anyBoundary++;
            else if (anyDisagreements++ < 4) failures.push(`sphere ${i}: TSL shapecast ${any}, bvhIntersectsSphere ${library} (nearest triangle at ${closest}, radius ${radius})`);
        }
        if (counted < inside || counted > inside + boundary) {
            if (countDisagreements++ < 4) failures.push(`sphere ${i}: counted ${counted} triangles, the CPU ${inside} (+${boundary} within 1e-4 of the radius)`);
        }
    }
    log(`${count} spheres over ${triangles.length} world triangles: ${hits} touch the scene, ${total} triangle contacts counted; ${anyBoundary} boundary differences`);
    await drainDevice(device);
    bvh.dispose();
    sync.dispose();
    trees.dispose();
    reportGpuTest({
        status: failures.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data: { spheres: count, triangles: triangles.length, hits, contacts: total, anyDisagreements, anyBoundary, countDisagreements },
        adapter,
        deviceErrors,
        startTime,
        logLines,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? `${error.message}\n${error.stack}` : String(error), adapter, deviceErrors, startTime, logLines });
}

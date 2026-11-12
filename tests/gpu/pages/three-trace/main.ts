/**
 * @pmndrs/bvh/three on a WebGPURenderer: a three scene (indexed, non-indexed,
 * groups, InstancedMesh, BatchedMesh, mirrored, hidden and off-layer objects)
 * goes into SceneTrees on the renderer's own device through
 * SceneTreesObject3DSync, and three TSL compute kernels trace it through
 * BvhTraceNodes:
 *
 * - TSL: `traceClosestMasked` (the camera's layers as the ray mask) and `traceAny`;
 * - WGSL: a `wgslFn` that includes `bvh.module` and calls `bvhTraceClosest` by name.
 *
 * The oracle rule: every hit must match the CPU walk over a scene the CPU
 * assembled from the same three objects (bvhSceneFromObject3D, tinybvh trees),
 * and the TSL and WGSL kernels must give exactly what the package's own batch
 * kernel gives on the same SceneTrees buffers. Checked on the GPU's first
 * trees, after tinybvh's swapped in, and after objects moved and a large mesh
 * made the buffers grow (the storage nodes must rebind to the new buffers;
 * the kernels are not rebuilt). All of it twice: records, then indexed leaves
 * (`triangleLeaves: 'indexed'`; BvhTraceNodes binds the positions as a fourth storage node).
 */

import { BatchedMesh, BoxGeometry, DoubleSide, Group, InstancedMesh, Matrix4, Mesh, MeshBasicMaterial, Object3D, SphereGeometry, TorusKnotGeometry, Vector3 } from 'three';
import { Fn, If, floatBitsToUint, instanceIndex, select as tslSelect, storage, uint, uvec4 as tslUvec4, wgslFn } from 'three/tsl';
import { StorageBufferAttribute, WebGPURenderer, type Node } from 'three/webgpu';
import { SceneTrees } from '../../../../src/scene/scene-trees.ts';
import { createTinyBvhBuilder } from '../../../../src/tinybvh.ts';
import { createCpuTracer, type CpuHit } from '../../../../src/trace/cpu-traversal.ts';
import { buildMeshBvhWithTinyBvh } from '../../../../src/trace/mesh-bvh-topology.ts';
import { createRandom, randomRays } from '../../../../src/trace/oracle-fixtures.ts';
import { packRays, type TraceRay } from '../../../../src/trace/rays.ts';
import { buildTraceScene } from '../../../../src/trace/scene-assembly.ts';
import { TraceBatchKernel } from '../../../../src/trace/trace-gpu.ts';
import { BvhTraceNodes, SceneTreesObject3DSync, bvhSceneFromObject3D, deviceFromRenderer, rayMaskFromLayers, type BvhTslNode } from '../../../../src/three/index.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { sameHit, traceOnGpu } from '../../scene-check.ts';

const startTime = performance.now();
const logLines: string[] = [];
const log = (line: string) => logLines.push(line);
let deviceErrors: string[] = [];
let adapter = null as ReturnType<typeof summarizeAdapterInfo> | null;

function place(object: Object3D, random: () => number, extent = 8): Object3D {
    object.position.set((random() - 0.5) * extent, (random() - 0.5) * extent, (random() - 0.5) * extent);
    object.quaternion.setFromAxisAngle(new Vector3(random(), random(), random()).normalize(), random() * 6);
    const s = 0.6 + random();
    object.scale.set(s, random() < 0.25 ? -s : s * (0.7 + random() * 0.6), s);
    return object;
}

function buildScene(random: () => number): { root: Group; movers: Object3D[] } {
    const material = () => new MeshBasicMaterial({ side: DoubleSide });
    const root = new Group();
    const knot = new TorusKnotGeometry(0.8, 0.25, 64, 8);
    const movers: Object3D[] = [];
    for (let i = 0; i < 6; i++) root.add(place(new Mesh(knot, material()), random));
    root.add(place(new Mesh(new SphereGeometry(0.9, 16, 12).toNonIndexed(), material()), random));
    const box = new BoxGeometry(1.4, 0.9, 1.1, 3, 2, 2);
    root.add(place(new Mesh(box, [0, 1, 2, 3, 4, 5].map(material)), random));
    const instanced = new InstancedMesh(new SphereGeometry(0.4, 10, 8), material(), 40);
    const dummy = new Object3D();
    for (let i = 0; i < 40; i++) {
        place(dummy, random, 12).updateMatrix();
        instanced.setMatrixAt(i, dummy.matrix);
    }
    instanced.count = 32;
    root.add(instanced);
    const batched = new BatchedMesh(8, 4000, 12000, material());
    const a = batched.addGeometry(new SphereGeometry(0.5, 10, 8));
    const b = batched.addGeometry(new BoxGeometry(0.7, 0.4, 1));
    for (const geometryId of [a, b, b, a, b]) {
        const id = batched.addInstance(geometryId);
        place(dummy, random, 10).updateMatrix();
        batched.setMatrixAt(id, new Matrix4().copy(dummy.matrix));
    }
    batched.setVisibleAt(1, false);
    root.add(batched);
    const hidden = place(new Mesh(new BoxGeometry(2, 2, 2), material()), random);
    hidden.visible = false;
    const offLayer = place(new Mesh(new BoxGeometry(1.8, 1.8, 1.8), material()), random);
    offLayer.layers.set(1);
    root.add(hidden, offLayer);
    movers.push(root.children[0], root.children[1], instanced, offLayer);
    root.updateMatrixWorld(true);
    return { root, movers };
}

// three's TSL types cannot follow the hit struct through a wgslFn; these take any node.
const uvec4 = tslUvec4 as unknown as (...parts: BvhTslNode[]) => BvhTslNode;
const select = tslSelect as unknown as (...parts: BvhTslNode[]) => BvhTslNode;

/** Three uvec4 per ray: (t, instance, triangle, frontFace), (u, v, material, blocked), normal. */
const WORDS_PER_RAY = 12;

try {
    const renderer = new WebGPURenderer();
    await renderer.init();
    const device = deviceFromRenderer(renderer);
    deviceErrors = watchDevice(device);
    adapter = summarizeAdapterInfo(device.adapterInfo, device.features);
    if (!device.features.has('subgroups')) log('no subgroups: tinybvh trees and the CPU top level only');

    const failures: string[] = [];
    const results: Record<string, unknown> = {};
    // Records and indexed leaves (docs/internals/design/indexed-leaves.md): BvhTraceNodes takes the format from SceneTrees.
    // Then quantized nodes (docs/internals/design/compressed-nodes.md): BvhTraceNodes takes the node format from SceneTrees too.
    for (const [triangleLeaves, nodeFormat] of [['records', 'f32'], ['indexed', 'f32'], ['records', 'quantized'], ['indexed', 'quantized']] as const) {
        const label = nodeFormat === 'quantized' ? `quantized ${triangleLeaves}` : triangleLeaves;
        const random = createRandom(57);
        const { root, movers } = buildScene(random);
        const materialOf = (mesh: Mesh, info: { instanceId: number; group: number }) => ((mesh.id * 7 + Math.max(0, info.instanceId) * 3 + info.group + 1) % 300);
        const syncOptions = { material: materialOf };
        const trees = await SceneTrees.create(device, { label: `three-trace ${label}`, holdRefinements: true, workers: { size: 2 }, triangleLeaves, nodeFormat });
        const sync = new SceneTreesObject3DSync(trees, root, syncOptions);
        const bvh = new BvhTraceNodes(renderer, trees);
        const kernel = await TraceBatchKernel.create(device, { triangleLeaves, nodeFormat });
        const tinyBvh = await createTinyBvhBuilder();
        let rebinds = 0;

        const frame = () => {
            root.updateMatrixWorld(true);
            sync.update();
            const encoder = device.createCommandEncoder();
            trees.update(encoder);
            device.queue.submit([encoder.finish()]);
            if (bvh.update()) rebinds++;
        };
        const settleFirstTrees = async () => {
            frame();
            await trees.whenBuilt();
            frame();
            frame();
        };

        // The rays: random origins and directions.
        // Half with random [tMin, tMax], half to 1e6; in one storage attribute.
        const rays: TraceRay[] = [...randomRays(random, 8192, [-9, -9, -9, 9, 9, 9], 2, { randomInterval: true }), ...randomRays(random, 8192, [-6, -6, -6, 6, 6, 6], 0)];
        const count = rays.length;
        const rayAttribute = new StorageBufferAttribute(packRays(rays), 4);
        const rayNode = storage(rayAttribute, 'vec4', count * 2).toReadOnly();
        const mask = rayMaskFromLayers(new Object3D().layers); // a default camera: layer 0
        const makeKernel = (wgsl: boolean) => {
            const out = new StorageBufferAttribute(new Uint32Array(count * WORDS_PER_RAY), 4);
            const outNode = storage(out, 'uvec4', count * 3);
            const wgslClosest = wgslFn(
                /* wgsl */ `
                fn threeTraceWgslClosest(a: vec4f, b: vec4f) -> vec4u {
                    let hit = bvhTraceClosest(BvhRay(a.xyz, a.w, b.xyz, b.w));
                    return vec4u(bitcast<u32>(hit.t), hit.instance, hit.triangle, select(0u, 1u, hit.frontFace));
                }`,
                [bvh.module as never],
            ) as unknown as (...args: unknown[]) => Node;
            const node = Fn(() => {
                If(instanceIndex.lessThan(count), () => {
                    const a = rayNode.element(instanceIndex.mul(2));
                    const b = rayNode.element(instanceIndex.mul(2).add(1));
                    const base = instanceIndex.mul(3);
                    if (wgsl) {
                        outNode.element(base).assign(wgslClosest(a, b));
                        return;
                    }
                    const hit = bvh.traceClosestMasked(a.xyz, b.xyz, a.w, b.w, uint(mask)).toVar();
                    const blocked = bvh.traceAny(a.xyz, b.xyz, a.w, b.w);
                    const bary = hit.get('barycentrics');
                    const normal = hit.get('normal');
                    outNode.element(base).assign(uvec4(floatBitsToUint(hit.get('t')), hit.get('instance'), hit.get('triangle'), select(hit.get('frontFace'), uint(1), uint(0))));
                    outNode.element(base.add(1)).assign(uvec4(floatBitsToUint(bary.x), floatBitsToUint(bary.y), hit.get('material'), select(blocked, uint(1), uint(0))));
                    outNode.element(base.add(2)).assign(uvec4(floatBitsToUint(normal.x), floatBitsToUint(normal.y), floatBitsToUint(normal.z), uint(0)));
                });
            })().compute(count, [64]);
            return { node, out };
        };
        const tslKernel = makeKernel(false);
        const wgslKernel = makeKernel(true);

        const check = async (state: string) => {
            const phase = `${label} ${state}`;
            await device.queue.onSubmittedWorkDone();
            const t0 = performance.now();
            await renderer.computeAsync(tslKernel.node);
            await renderer.computeAsync(wgslKernel.node);
            const computeMs = performance.now() - t0;
            const [tslWords, wgslWords] = (await Promise.all([renderer.getArrayBufferAsync(tslKernel.out), renderer.getArrayBufferAsync(wgslKernel.out)])).map((bytes) => new Uint32Array(bytes));
            const tslFloats = new Float32Array(tslWords.buffer);
            const wgslFloats = new Float32Array(wgslWords.buffer);
            // The package's own batch kernel on the same buffers, masked and unmasked.
            const own = await traceOnGpu(device, kernel, trees.traceBuffers, rays, mask);
            const ownUnmasked = await traceOnGpu(device, kernel, trees.traceBuffers, rays);
            // The CPU reference from the same three objects (instance i = sources[i]).
            // One mesh and instance per group, as the sync makes them.
            const scene = bvhSceneFromObject3D(root, { ...syncOptions, multiMaterial: 'groups' });
            const reference = createCpuTracer(await buildTraceScene(scene.meshes, scene.instances, { build: (geometry) => buildMeshBvhWithTinyBvh(geometry, tinyBvh) }));
            const positionOf = new Map(scene.sources.map((source, index) => [`${source.object.id}:${source.instanceId}:${source.group}`, index]));
            const toReference = (hit: CpuHit | null): CpuHit | null => {
                if (!hit) return null;
                const source = sync.sourceOf(hit.instanceIndex);
                const index = source ? positionOf.get(`${source.object.id}:${source.instanceId}:${source.group}`) : undefined;
                return { ...hit, instanceIndex: index ?? -1 };
            };
            const tslHit = (ray: number): CpuHit | null => {
                const base = ray * WORDS_PER_RAY;
                if (!(tslFloats[base] >= 0)) return null;
                return {
                    t: tslFloats[base],
                    instanceIndex: tslWords[base + 1],
                    triangleIndex: tslWords[base + 2],
                    frontFace: tslWords[base + 3] !== 0,
                    u: tslFloats[base + 4],
                    v: tslFloats[base + 5],
                    material: tslWords[base + 6],
                    normal: [tslFloats[base + 8], tslFloats[base + 9], tslFloats[base + 10]],
                };
            };
            let hits = 0;
            let blockedCount = 0;
            let disagreements = 0;
            const note = (what: string) => {
                if (disagreements++ < 4) failures.push(`${phase}: ${what}`);
            };
            for (let ray = 0; ray < count; ray++) {
                const ours = tslHit(ray);
                const batch = own.closest(ray);
                if (ours) hits++;
                // Same walk, same buffers, same device: the same bits.
                if (JSON.stringify(ours) !== JSON.stringify(batch)) note(`ray ${ray}: TSL ${JSON.stringify(ours)} vs batch kernel ${JSON.stringify(batch)}`);
                const blocked = tslWords[ray * WORDS_PER_RAY + 7] !== 0;
                if (blocked) blockedCount++;
                if (blocked !== ownUnmasked.any(ray)) note(`ray ${ray}: TSL any ${blocked} vs batch kernel ${ownUnmasked.any(ray)}`);
                const base = ray * WORDS_PER_RAY;
                const unmasked = ownUnmasked.closest(ray);
                const wgslMiss = !(wgslFloats[base] >= 0);
                const wgslSame = unmasked === null ? wgslMiss : !wgslMiss && wgslFloats[base] === unmasked.t && wgslWords[base + 1] === unmasked.instanceIndex && wgslWords[base + 2] === unmasked.triangleIndex;
                if (!wgslSame) note(`ray ${ray}: WGSL closest (${wgslFloats[base]}, ${wgslWords[base + 1]}, ${wgslWords[base + 2]}) vs batch kernel ${JSON.stringify(unmasked)}`);
                const expected = reference.closest(rays[ray], { mask });
                if (!sameHit(toReference(ours), expected)) note(`ray ${ray}: TSL ${JSON.stringify(toReference(ours))} vs CPU reference ${JSON.stringify(expected)}`);
                if (blocked !== reference.any(rays[ray])) note(`ray ${ray}: TSL any ${blocked} vs CPU reference`);
            }
            results[phase] = { hits, blocked: blockedCount, disagreements, computeMs: Math.round(computeMs), gpuTrees: trees.counts.onGpuTrees, tinybvhTrees: trees.counts.onTinybvhTrees, version: trees.version, instances: sync.instanceCount };
            log(`${phase}: ${hits} hits, ${blockedCount} blocked, ${disagreements} disagreements (${trees.counts.onGpuTrees} GPU / ${trees.counts.onTinybvhTrees} tinybvh trees, version ${trees.version})`);
            if (hits < count / 20) failures.push(`${phase}: only ${hits} hits`);
        };

        await settleFirstTrees();
        if (trees.gpuFirstTrees.enabled && trees.counts.onGpuTrees === 0) failures.push('no GPU first trees in use');
        await check('first-trees');

        await trees.whenRefined();
        trees.setHoldRefinements(false);
        frame();
        frame();
        await check('tinybvh');

        // Moves (the sync pushes them), and a large mesh, so the node and triangle buffers grow.
        const versionBefore = trees.version;
        const rebindsBefore = rebinds;
        for (const mover of movers) {
            mover.position.x += 0.7;
            mover.rotation.y += 0.4;
        }
        (movers[2] as InstancedMesh).setMatrixAt(3, new Matrix4().makeTranslation(1, 2, 3));
        root.add(place(new Mesh(new TorusKnotGeometry(1.5, 0.3, 600, 40), new MeshBasicMaterial({ side: DoubleSide })), random));
        root.updateMatrixWorld(true);
        sync.sync();
        trees.setHoldRefinements(false);
        await settleFirstTrees();
        await trees.whenRefined();
        frame();
        frame();
        if (trees.version === versionBefore) failures.push('the buffers did not grow; the rebind was not exercised');
        if (rebinds === rebindsBefore) failures.push('BvhTraceNodes never rebound after the buffers grew');
        await check('moved-and-grown');
        results[`${label} rebinds`] = rebinds;

        // three's disposal of the stand-ins (as renderer.dispose() does for every attribute) must not destroy SceneTrees' buffers.
        const beforeDispose = await traceOnGpu(device, kernel, trees.traceBuffers, rays);
        for (const node of [bvh.storage.nodes, bvh.storage.triangles, bvh.storage.topLevel, ...(bvh.storage.positions ? [bvh.storage.positions] : [])]) (node.value as StorageBufferAttribute).dispose();
        const afterDispose = await traceOnGpu(device, kernel, trees.traceBuffers, rays);
        let changedByDispose = 0;
        for (let ray = 0; ray < count; ray++) if (beforeDispose.closest(ray)?.t !== afterDispose.closest(ray)?.t) changedByDispose++;
        if (changedByDispose > 0) failures.push(`${changedByDispose} hits changed after three disposed the stand-in attributes`);
        results[`${label} afterStandInDispose`] = { changed: changedByDispose };

        bvh.dispose();
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

/**
 * three-mesh-bvh 0.9.18 (MIT, Garrett Johnson): `MeshBVH` on a three
 * `BufferGeometry`, with its default CENTER split and with SAH. Its GPU side
 * (`three-mesh-bvh/webgpu`) is in compare/gpu.
 *
 * `MeshBVH` reorders the geometry's index in place, so it gets its own copy of
 * the index (the app would hand it its own geometry; the positions are shared
 * and the index copy replaces the app's, so neither is counted as its memory).
 * Its hits name triangles in the reordered index: a lookup made once after the
 * build (not timed, not counted) maps them back to the original triangles.
 */

import { Box3, BufferAttribute, BufferGeometry, DoubleSide, Ray, Sphere, Vector3, Matrix4 } from 'three';
import { CENTER, getBVHExtremes, MeshBVH, SAH } from 'three-mesh-bvh';
import { features, type BuiltMesh, type CompareGeometry, type MeshAdapter, type TreeQuality } from './types.ts';

/** `reordered[i]` -> original triangle, by matching index triples (identical triples are interchangeable). */
export function reorderedToOriginal(original: Uint32Array, reordered: ArrayLike<number>): Uint32Array {
    const count = original.length / 3;
    const order = (array: ArrayLike<number>) => {
        const ids = new Uint32Array(count);
        for (let i = 0; i < count; i++) ids[i] = i;
        return ids.sort((a, b) => array[a * 3] - array[b * 3] || array[a * 3 + 1] - array[b * 3 + 1] || array[a * 3 + 2] - array[b * 3 + 2]);
    };
    const a = order(original);
    const b = order(reordered);
    const map = new Uint32Array(count);
    for (let i = 0; i < count; i++) map[b[i]] = a[i];
    return map;
}

/** Bytes three-mesh-bvh adds beyond the geometry: its node buffers (`_roots`) and, in indirect mode, `_indirectBuffer`. */
export function tmbvhBytes(bvh: MeshBVH): number {
    const internal = bvh as unknown as { _roots: ArrayBuffer[]; _indirectBuffer: ArrayBufferView | null };
    return internal._roots.reduce((sum, root) => sum + root.byteLength, 0) + (internal._indirectBuffer?.byteLength ?? 0);
}

export function tmbvhQuality(bvh: MeshBVH): TreeQuality {
    const [extremes] = getBVHExtremes(bvh) as unknown as { nodeCount: number; leafNodeCount: number; surfaceAreaScore: number; depth: { max: number } }[];
    const box = bvh.getBoundingBox(new Box3());
    const s = box.getSize(new Vector3());
    const rootArea = 2 * (s.x * s.y + s.y * s.z + s.z * s.x);
    return { sah: extremes.surfaceAreaScore / rootArea, depth: extremes.depth.max, nodes: extremes.nodeCount, leaves: extremes.leafNodeCount, bytes: tmbvhBytes(bvh), note: '32-byte nodes of one box' };
}

/** A three geometry over the compare geometry: positions shared, its own index (MeshBVH reorders it). */
export function threeGeometry(geometry: CompareGeometry, copyIndex: boolean): BufferGeometry {
    const out = new BufferGeometry();
    out.setAttribute('position', new BufferAttribute(geometry.positions, 3));
    out.setIndex(new BufferAttribute(copyIndex ? geometry.indices.slice() : geometry.indices, 1));
    return out;
}

function makeAdapter(id: string, name: string, strategy: number, strategyName: string): MeshAdapter {
    return {
        id,
        name,
        library: 'three-mesh-bvh@0.9.18',
        license: 'MIT',
        buildMethod: `new MeshBVH(geometry, { strategy: ${strategyName} }) with its default targetLeafSize (10), on this thread`,
        features: features({
            raycastFirst: 'yes',
            raycastAll: 'yes',
            gpuTrace: { status: 'yes', note: 'three-mesh-bvh/webgpu: WGSL/TSL inside three\'s node system (BVHComputeData)' },
            intersectsBox: 'yes',
            intersectsSphere: 'yes',
            closestPoint: 'yes',
            refit: { status: 'yes', note: 'CPU refit' },
            workerBuild: { status: 'yes', note: 'GenerateMeshBVHWorker; ParallelMeshBVHWorker needs SharedArrayBuffer' },
            gpuBuild: { status: 'no', note: 'builds on the CPU only' },
            treeStats: 'yes',
            serialize: 'yes',
        }),
        geometryCopyBytes: (geometry) => geometry.indices.byteLength,
        build(geometry) {
            const three = threeGeometry(geometry, true);
            const bvh = new MeshBVH(three, { strategy: strategy as never, verbose: false });
            let map: Uint32Array = new Uint32Array(0);
            const ray = new Ray();
            const box = new Box3();
            const sphere = new Sphere();
            const identity = new Matrix4();
            const point = new Vector3();
            const setRay = (r: { origin: number[]; direction: number[] }) => {
                ray.origin.set(r.origin[0], r.origin[1], r.origin[2]);
                ray.direction.set(r.direction[0], r.direction[1], r.direction[2]);
            };
            const built: BuiltMesh = {
                prepareQueries() {
                    map = reorderedToOriginal(geometry.indices, three.index!.array);
                },
                raycastFirst(r) {
                    setRay(r);
                    const hit = bvh.raycastFirst(ray, DoubleSide);
                    return hit ? { triangle: map[hit.faceIndex!], t: hit.distance } : null;
                },
                raycastAll(r) {
                    setRay(r);
                    return bvh
                        .raycast(ray, DoubleSide)
                        .map((hit) => ({ triangle: map[hit.faceIndex!], t: hit.distance }))
                        .sort((a, b) => a.t - b.t);
                },
                intersectsBox(b) {
                    box.min.set(b[0], b[1], b[2]);
                    box.max.set(b[3], b[4], b[5]);
                    return bvh.intersectsBox(box, identity);
                },
                intersectsSphere(center, radius) {
                    sphere.center.set(center[0], center[1], center[2]);
                    sphere.radius = radius;
                    return bvh.intersectsSphere(sphere);
                },
                closestPoint(p) {
                    point.set(p[0], p[1], p[2]);
                    const hit = bvh.closestPointToPoint(point);
                    return hit ? { triangle: map[hit.faceIndex], distance: hit.distance, point: [hit.point.x, hit.point.y, hit.point.z] } : null;
                },
                refit(positions) {
                    three.setAttribute('position', new BufferAttribute(positions, 3));
                    bvh.refit();
                },
                memory() {
                    return { bytes: tmbvhBytes(bvh), method: 'byteLength of every `_roots` node buffer (32 B per node) plus `_indirectBuffer` (none here); the index is the geometry\'s own, reordered in place, as docs/design/memory.md counts it' };
                },
                quality: () => tmbvhQuality(bvh),
                dispose() {
                    three.dispose();
                },
            };
            return built;
        },
    };
}

export const tmbvhCenterAdapter = makeAdapter('tmbvh-center', 'three-mesh-bvh (CENTER, default)', CENTER, 'CENTER');
export const tmbvhSahAdapter = makeAdapter('tmbvh-sah', 'three-mesh-bvh (SAH)', SAH, 'SAH');

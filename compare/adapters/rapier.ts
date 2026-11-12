/**
 * Rapier (`@dimforge/rapier3d-compat` 0.21.0, Apache-2.0, Dimforge): a
 * trimesh collider in a physics world, queried with the world's ray cast,
 * point projection and shape intersection (WASM, parry's BVH inside). A
 * physics engine's raycast, as a baseline.
 *
 * Its WASM heap is the only place its memory lives. `prepare` catches the
 * module's `WebAssembly.Memory` while Rapier loads, so `memory()` is the
 * heap's growth across the build: whole 64 KiB pages, and an upper bound only
 * when it is the first build in the process (a heap never shrinks, so a later
 * build may reuse room an earlier one freed). The Node runner measures each
 * library in a fresh process for that reason.
 */

import RAPIER from '@dimforge/rapier3d-compat';
import { features, type MeshAdapter } from './types.ts';

let ready: Promise<void> | null = null;
let wasmMemory: WebAssembly.Memory | null = null;
/** Builds made in this process: only the first one's heap growth is its size. */
let builds = 0;

async function init(): Promise<void> {
    const original = WebAssembly.instantiate;
    const capture = (result: unknown) => {
        const instance = (result as { instance?: WebAssembly.Instance }).instance ?? (result as WebAssembly.Instance);
        const memory = (instance as WebAssembly.Instance | undefined)?.exports?.memory;
        if (memory instanceof WebAssembly.Memory) wasmMemory = memory;
        return result;
    };
    (WebAssembly as { instantiate: unknown }).instantiate = (...args: unknown[]) => (original as (...a: unknown[]) => Promise<unknown>).apply(WebAssembly, args).then(capture);
    try {
        await RAPIER.init();
    } finally {
        (WebAssembly as { instantiate: unknown }).instantiate = original;
    }
}

/** Rapier's WASM heap in bytes, or null when it couldn't be caught. */
export function rapierHeapBytes(): number | null {
    return wasmMemory ? wasmMemory.buffer.byteLength : null;
}

const MAX_TOI = 1e30;

export const rapierAdapter: MeshAdapter = {
    id: 'rapier',
    name: 'Rapier (trimesh collider)',
    library: '@dimforge/rapier3d-compat@0.21.0',
    license: 'Apache-2.0',
    buildMethod: 'World with no gravity, createCollider(ColliderDesc.trimesh(positions, indices)), one world.step() so the scene queries see it',
    features: features({
        raycastFirst: { status: 'yes', note: 'world.castRayAndGetNormal; no triangle id (its feature id is 0 for every trimesh hit), so it agrees by distance' },
        raycastAll: { status: 'no', note: 'intersectionsWithRay gives one hit per collider, not per triangle' },
        intersectsBox: { status: 'yes', note: 'world.intersectionsWithShape with a Cuboid' },
        intersectsSphere: { status: 'yes', note: 'world.intersectionsWithShape with a Ball' },
        closestPoint: { status: 'yes', note: 'world.projectPointAndGetFeature; its feature is a vertex, edge or face of the triangle, not the triangle, so it agrees by distance' },
        refit: { status: 'no', note: 'a trimesh is rebuilt (setShape), not refitted' },
        serialize: { status: 'yes', note: 'World.takeSnapshot' },
    }),
    async prepare() {
        ready ??= init();
        await ready;
    },
    build(geometry) {
        if (!ready) throw new Error('rapierAdapter.prepare() first');
        const first = builds++ === 0;
        const heapBefore = rapierHeapBytes();
        const world = new RAPIER.World({ x: 0, y: 0, z: 0 });
        const collider = world.createCollider(RAPIER.ColliderDesc.trimesh(geometry.positions, geometry.indices));
        world.step();
        const heapAfter = rapierHeapBytes();
        const ray = new RAPIER.Ray({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 1 });
        const identity = { x: 0, y: 0, z: 0, w: 1 };
        const ball = new RAPIER.Ball(1);
        return {
            raycastFirst(r) {
                ray.origin = { x: r.origin[0], y: r.origin[1], z: r.origin[2] };
                ray.dir = { x: r.direction[0], y: r.direction[1], z: r.direction[2] };
                const hit = world.castRayAndGetNormal(ray, MAX_TOI, false);
                // Rapier 0.21 reports feature 0 for every trimesh ray hit, so the triangle is unknown: agreement is by distance.
                return hit ? { triangle: -1, t: hit.timeOfImpact } : null;
            },
            intersectsBox(b) {
                const shape = new RAPIER.Cuboid((b[3] - b[0]) / 2, (b[4] - b[1]) / 2, (b[5] - b[2]) / 2);
                let found = false;
                world.intersectionsWithShape({ x: (b[0] + b[3]) / 2, y: (b[1] + b[4]) / 2, z: (b[2] + b[5]) / 2 }, identity, shape, () => ((found = true), false));
                return found;
            },
            intersectsSphere(center, radius) {
                ball.radius = radius;
                let found = false;
                world.intersectionsWithShape({ x: center[0], y: center[1], z: center[2] }, identity, ball, () => ((found = true), false));
                return found;
            },
            closestPoint(p) {
                const hit = world.projectPointAndGetFeature({ x: p[0], y: p[1], z: p[2] });
                if (!hit) return null;
                // The feature id is the vertex, edge or face within the triangle, not the triangle: unknown.
                return { triangle: -1, distance: Math.hypot(hit.point.x - p[0], hit.point.y - p[1], hit.point.z - p[2]), point: [hit.point.x, hit.point.y, hit.point.z] };
            },
            memory() {
                const grown = first && heapBefore !== null && heapAfter !== null ? heapAfter - heapBefore : null;
                return { bytes: grown, method: first ? 'growth of Rapier\'s WASM heap across the first build in this process (64 KiB pages)' : 'not countable here: a later build reuses heap an earlier one freed (the Node runner counts it in a fresh process)' };
            },
            dispose() {
                world.removeCollider(collider, false);
                world.free();
            },
        };
    },
};

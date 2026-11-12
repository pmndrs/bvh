/**
 * Babylon.js picking (`@babylonjs/core` 9.35.0, Apache-2.0): an engine's own
 * ray picking on a mesh, on a `NullEngine` (no canvas, no GPU), so it runs in
 * Node and on the pages alike.
 *
 * - **octree**: the mesh is cut into submeshes (`Mesh.subdivide`, one per
 *   about 1,000 triangles), then `AbstractMesh.createOrUpdateSubmeshesOctree`
 *   with its defaults (64 per block, depth 2), as Babylon's docs advise for
 *   picking on big meshes; `mesh.intersects(ray)` then tests the octree's
 *   candidate submeshes' boxes and their triangles.
 * - **brute force**: one submesh, every triangle tested (after the mesh's
 *   bounding sphere and box).
 *
 * `mesh.intersects` returns the closest hit only: Babylon has no all-hits pick
 * on one mesh (`multiPickWithRay` gives one hit per mesh). The first pick
 * builds Babylon's `Vector3` copy of the positions, so the build does one pick
 * (timed with it).
 */

import { NullEngine } from '@babylonjs/core/Engines/nullEngine.js';
import { Scene } from '@babylonjs/core/scene.js';
import { Mesh } from '@babylonjs/core/Meshes/mesh.js';
import { VertexData } from '@babylonjs/core/Meshes/mesh.vertexData.js';
import { Ray } from '@babylonjs/core/Culling/ray.js';
import { Vector3 } from '@babylonjs/core/Maths/math.vector.js';
import { Material } from '@babylonjs/core/Materials/material.js';
import '@babylonjs/core/Culling/Octrees/octreeSceneComponent.js';
import { features, type MeshAdapter } from './types.ts';

let engine: NullEngine | null = null;

const TRIANGLES_PER_SUBMESH = 1000;

function makeAdapter(id: string, name: string, octree: boolean): MeshAdapter {
    return {
        id,
        name,
        library: '@babylonjs/core@9.35.0',
        license: 'Apache-2.0',
        buildMethod: octree
            ? `VertexData.applyToMesh, Mesh.subdivide(triangles / ${TRIANGLES_PER_SUBMESH}), createOrUpdateSubmeshesOctree() (64 per block, depth 2), one pick to make its position cache`
            : 'VertexData.applyToMesh (one submesh), one pick to make its position cache',
        features: features({
            raycastFirst: { status: 'yes', note: octree ? 'mesh.intersects through the submesh octree' : 'mesh.intersects, every triangle' },
            raycastAll: { status: 'no', note: 'one hit per mesh (multiPickWithRay); no all-hits pick on a mesh' },
            refit: { status: 'partial', note: octree ? 'refreshBoundingInfo + createOrUpdateSubmeshesOctree rebuilds the octree' : 'refreshBoundingInfo' },
            gpuTrace: { status: 'no', note: 'GPU picking renders ids to a texture (GPUPicker), no ray queries' },
        }),
        build(geometry) {
            engine ??= new NullEngine();
            const scene = new Scene(engine);
            const mesh = new Mesh(`compare-${id}`, scene);
            // Submeshes without a material are not picked (Babylon falls back to the bounding box).
            mesh.material = new Material('compare', scene);
            const data = new VertexData();
            data.positions = geometry.positions;
            data.indices = geometry.indices;
            data.applyToMesh(mesh, true);
            if (octree) {
                mesh.subdivide(Math.max(1, Math.ceil(geometry.triangleCount / TRIANGLES_PER_SUBMESH)));
                mesh.createOrUpdateSubmeshesOctree();
            }
            const ray = new Ray(new Vector3(), new Vector3(0, 0, 1));
            const pick = (r: { origin: number[]; direction: number[] }) => {
                ray.origin.set(r.origin[0], r.origin[1], r.origin[2]);
                ray.direction.set(r.direction[0], r.direction[1], r.direction[2]);
                ray.length = Number.MAX_VALUE;
                const info = mesh.intersects(ray, false);
                return info.hit && info.distance >= 0 ? { triangle: info.faceId, t: info.distance } : null;
            };
            pick({ origin: [0, 0, 0], direction: [0, 0, 1] });
            return {
                raycastFirst: pick,
                refit(positions) {
                    mesh.updateVerticesData('position', positions, false, false);
                    mesh.refreshBoundingInfo();
                    if (octree) mesh.createOrUpdateSubmeshesOctree();
                },
                memory() {
                    return { bytes: null, method: 'Babylon gives no count; the Node runner measures its heap growth (vertex buffers, submeshes, octree and the Vector3 position cache picking makes)' };
                },
                dispose() {
                    scene.dispose();
                },
            };
        },
    };
}

export const babylonOctreeAdapter = makeAdapter('babylon-octree', 'Babylon.js picking (submesh octree)', true);
export const babylonBruteAdapter = makeAdapter('babylon-brute', 'Babylon.js picking (brute force)', false);

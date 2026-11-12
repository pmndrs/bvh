/**
 * three's own `Mesh.raycast` (MIT, three.js authors): no tree, every triangle
 * tested after the bounding-sphere check. The no-BVH baseline.
 */

import { BufferAttribute, BufferGeometry, DoubleSide, Mesh, MeshBasicMaterial, Raycaster, type Intersection } from 'three';
import { features, type MeshAdapter } from './types.ts';

const threeRaycast = Mesh.prototype.raycast;

export const threeRaycastAdapter: MeshAdapter = {
    id: 'three',
    name: "three Mesh.raycast (no BVH)",
    library: 'three@0.187.0',
    license: 'MIT',
    buildMethod: 'new Mesh(geometry) and computeBoundingSphere(): nothing else is built',
    features: features({
        raycastFirst: { status: 'partial', note: 'no first-hit mode: every hit is found and sorted, the first taken' },
        raycastAll: 'yes',
    }),
    build(geometry) {
        const three = new BufferGeometry();
        three.setAttribute('position', new BufferAttribute(geometry.positions, 3));
        three.setIndex(new BufferAttribute(geometry.indices, 1));
        three.computeBoundingSphere();
        const mesh = new Mesh(three, new MeshBasicMaterial({ side: DoubleSide }));
        mesh.raycast = threeRaycast;
        mesh.updateMatrixWorld();
        const raycaster = new Raycaster();
        const hits: Intersection[] = [];
        const cast = (ray: { origin: number[]; direction: number[] }) => {
            raycaster.ray.origin.set(ray.origin[0], ray.origin[1], ray.origin[2]);
            raycaster.ray.direction.set(ray.direction[0], ray.direction[1], ray.direction[2]);
            hits.length = 0;
            mesh.raycast(raycaster, hits);
            hits.sort((a, b) => a.distance - b.distance);
            return hits;
        };
        return {
            raycastFirst(ray) {
                const hit = cast(ray)[0];
                return hit ? { triangle: hit.faceIndex!, t: hit.distance } : null;
            },
            raycastAll(ray) {
                return cast(ray).map((hit) => ({ triangle: hit.faceIndex!, t: hit.distance }));
            },
            memory() {
                return { bytes: 0, method: 'nothing beyond the geometry (a bounding sphere, 16 bytes, not counted)' };
            },
            dispose() {
                three.dispose();
            },
        };
    },
};

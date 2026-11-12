/// <reference types="node" />
/**
 * The runner's scenes, each merged into one world-space mesh:
 * - `torus-knot`: three's `TorusKnotGeometry(1, 0.3, 1000, 50)`, 100k triangles;
 * - `furnished-room`: the Breakfast Room (BVH_ASSETS_DIR / BVH_MANIFESTS_DIR, tests/gpu/README.md);
 * - `sponza`: Intel New Sponza with curtains, as the trace benches load it.
 * Instances are flattened to world space, so a mesh used twice is there twice.
 */

import { readSceneInstances } from '../../tools/scene-meshes.ts';
import type { CompareGeometry } from '../adapters/types.ts';
import { mergeParts, torusKnot } from '../core/geometry.ts';

export const SCENES = ['torus-knot', 'furnished-room', 'sponza'] as const;

export function loadScene(id: string): CompareGeometry {
    if (id === 'torus-knot') return { ...torusKnot(), name: 'torus-knot' };
    const { meshes, instances } = readSceneInstances(id);
    return mergeParts(
        id,
        instances.map((instance) => ({ positions: meshes[instance.meshIndex].positions, indices: meshes[instance.meshIndex].indices, matrix: instance.objectToWorld })),
    );
}

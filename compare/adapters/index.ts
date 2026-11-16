/** Every mesh-level library of the comparison, ours first (the reference every answer is checked against). */

import { babylonBruteAdapter, babylonOctreeAdapter } from './babylon.ts';
import { bvhTreeAdapter } from './bvh-tree.ts';
import { oursAdapter, oursPositionsAdapter } from './ours.ts';
import { rapierAdapter } from './rapier.ts';
import { tmbvhCenterAdapter, tmbvhSahAdapter } from './three-mesh-bvh.ts';
import { threeRaycastAdapter } from './three-raycast.ts';
import type { MeshAdapter } from './types.ts';

export const MESH_ADAPTERS: readonly MeshAdapter[] = [oursAdapter, oursPositionsAdapter, tmbvhCenterAdapter, tmbvhSahAdapter, bvhTreeAdapter, babylonOctreeAdapter, babylonBruteAdapter, rapierAdapter, threeRaycastAdapter];

export function adapterById(id: string): MeshAdapter {
    const adapter = MESH_ADAPTERS.find((entry) => entry.id === id);
    if (!adapter) throw new Error(`no adapter ${id}; there are ${MESH_ADAPTERS.map((entry) => entry.id).join(', ')}`);
    return adapter;
}

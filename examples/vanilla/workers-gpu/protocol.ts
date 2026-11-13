/**
 * The messages between vanilla/workers-gpu's page and its worker
 * (gpu-worker.ts). Plain data: geometry arrays are copied in, built trees are
 * transferred out.
 */

import type { GpuLayoutMeshBvh, SceneTreeCounts } from '@pmndrs/bvh';
import type { CameraBasis } from '../../shared/vanilla/camera.ts';
import type { IndexedMesh, VanillaInstance } from '../../shared/vanilla/geometry.ts';

export type GpuBuilderKind = 'gpu-hploc' | 'gpu-lbvh';

export type ToWorker =
    /** Way 2: build these meshes on the worker's own device and send the trees back. */
    | { readonly type: 'build'; readonly id: number; readonly builder: GpuBuilderKind; readonly meshes: readonly IndexedMesh[] }
    /** Way 3: render the scene into this canvas, with the worker's own device and SceneTrees. */
    | { readonly type: 'render'; readonly id: number; readonly canvas: OffscreenCanvas; readonly meshes: readonly IndexedMesh[]; readonly instances: readonly VanillaInstance[]; readonly radius: number }
    /** The camera of the page's next frame (way 3), and the canvas size. */
    | { readonly type: 'camera'; readonly basis: CameraBasis; readonly width: number; readonly height: number }
    /** Way 3: the hit records of a 32 x 32 grid of pixels of the last frame. */
    | { readonly type: 'check'; readonly id: number }
    /** Way 3: stop rendering and free the scene. */
    | { readonly type: 'stop'; readonly id: number };

export interface PixelRecord {
    readonly t: number;
    readonly instance: number;
    readonly triangle: number;
}

export type FromWorker =
    | {
          readonly type: 'built';
          readonly id: number;
          readonly layouts: readonly GpuLayoutMeshBvh[];
          readonly builtBy: GpuBuilderKind | 'tinybvh';
          readonly fallbackReason: string | null;
          /** Requesting the worker's device: the first build only, else 0. */
          readonly deviceMs: number;
          /** Compiling the builder's pipelines: its first build only, else 0. */
          readonly prepareMs: number;
          /** Encode to the trees being usable on the worker's device. */
          readonly buildMs: number;
          /** Reading the trees back into CPU arrays. */
          readonly readMs: number;
          readonly bytes: number;
      }
    /** Way 3: every mesh has a tree (`'first'`), or tinybvh's (`'best'`). */
    | { readonly type: 'trees'; readonly id: number; readonly stage: 'first' | 'best'; readonly counts: SceneTreeCounts }
    | { readonly type: 'stats'; readonly id: number; readonly counts: SceneTreeCounts; readonly resident: number; readonly frames: number; readonly mrays: number | null }
    | { readonly type: 'pixels'; readonly id: number; readonly basis: CameraBasis; readonly width: number; readonly height: number; readonly hits: readonly (PixelRecord | null)[] }
    | { readonly type: 'stopped'; readonly id: number }
    | { readonly type: 'error'; readonly message: string };

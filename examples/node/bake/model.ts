/**
 * The default models of `node/bake` and the scene bake, shared by the CLI
 * (which bakes them) and the page (which, in a smoke run only, bakes them
 * fresh to compare with the committed files). Browser-safe: no Node imports.
 */

import {
    buildMovableScene,
    buildTraceScene,
    createTinyBvhBuilder,
    serializeMovableScene,
    serializeTraceScene,
    type MovableSceneData,
    type TinyBvhBuilder,
    type TraceSceneData,
    type TraceSceneInstance,
} from '@pmndrs/bvh';
import { box, ground, makeScene, sphere, torusKnot, type IndexedMesh, type VanillaScene } from '../../shared/vanilla/geometry.ts';
import { mat4 } from '../../shared/vanilla/math.ts';

/** A 6,400-triangle torus knot, small enough to commit its bake (about 0.5 MB). */
export function defaultMesh(): IndexedMesh {
    return torusKnot(1, 0.3, 200, 16);
}

/** The bake's leaf size: what the CLI uses with no `--leaf`. */
export const DEFAULT_LEAF = 8;

/**
 * The default scene of `--scene`: the knot three times (one tree, three
 * instances), and a ground, a sphere and a box used once each (merged into
 * one world-space mesh, being static). The box's faces carry their own
 * materials (`triangleMaterials`); every instance has one too.
 */
export function defaultScene(): VanillaScene {
    const knot = defaultMesh();
    const cube = box(0.8, 0.8, 0.8);
    const faced: IndexedMesh = { ...cube, triangleMaterials: Uint16Array.from({ length: cube.indices.length / 3 }, (_, k) => 6 + ((k >> 1) % 3)) };
    const meshes = [knot, ground(12, 8), sphere(0.6, 16, 8), faced];
    const turn = (angle: number) => mat4.axisAngle([0, 1, 0], angle);
    return makeScene('knots', meshes, [
        { mesh: 0, objectToWorld: mat4.compose([-3.5, 1.8, 0], turn(0.6), [1, 1, 1]), material: 1 },
        { mesh: 0, objectToWorld: mat4.compose([0, 1.8, -0.5], turn(0), [1, 1, 1]), material: 2 },
        { mesh: 0, objectToWorld: mat4.compose([3.5, 1.8, 0], turn(-0.6), [1, 1, 1]), material: 3 },
        { mesh: 1, objectToWorld: mat4.identity(), material: 0 },
        { mesh: 2, objectToWorld: mat4.compose([0.4, 0.6, 2.5], turn(0), [1, 1, 1]), material: 4 },
        { mesh: 3, objectToWorld: mat4.compose([-2.5, 0.4, 2.5], turn(0.4), [1, 1, 1]), material: 5 },
    ]);
}

export interface SceneBakeOptions {
    /** Leaves hold at most this many triangles, 1 to 8. Default 8. */
    readonly maxLeafPrimitives?: number;
    /** `'indexed'`: 16-byte triangles, and the scene file carries the positions. Default `'records'`. */
    readonly triangleLeaves?: 'records' | 'indexed';
    /** The scene's nodes. Default `'quantized'` (the library's). */
    readonly nodeFormat?: 'f32' | 'quantized';
    /** Merge the static instances whose mesh is used once (every instance of a baked scene is static). Default true. */
    readonly mergeStaticInstances?: boolean;
}

export interface BakedScene {
    /** The file's bytes: what `serializeTraceScene` wrote. */
    readonly bytes: Uint8Array;
    readonly data: TraceSceneData;
    /** tinybvh's builds and the assembly, in ms. */
    readonly buildMs: number;
}

/**
 * Builds a whole scene with tinybvh (`buildTraceScene`) and serializes it.
 * Every instance is static (a baked top level can't move anything), so
 * instances of meshes used once are merged unless `mergeStaticInstances` is
 * false. Instance `i` reports `instanceIndex` `i`.
 */
export async function bakeScene(scene: VanillaScene, options: SceneBakeOptions = {}, builder?: TinyBvhBuilder): Promise<BakedScene> {
    const tiny = builder ?? (await createTinyBvhBuilder());
    const instances: TraceSceneInstance[] = scene.instances.map((instance, index) => ({ meshIndex: instance.mesh, instanceIndex: index, objectToWorld: instance.objectToWorld, material: instance.material, static: true }));
    const start = performance.now();
    const data = await buildTraceScene(scene.meshes, instances, {
        build: (geometry) => tiny.buildLayout(geometry, { maxLeafPrimitives: options.maxLeafPrimitives ?? DEFAULT_LEAF, triangleLeaves: options.triangleLeaves }).layout,
        triangleLeaves: options.triangleLeaves,
        nodeFormat: options.nodeFormat,
        mergeStaticInstances: options.mergeStaticInstances,
    });
    const buildMs = performance.now() - start;
    return { bytes: new Uint8Array(serializeTraceScene(data)), data, buildMs };
}

export interface BakedMovableScene {
    /** The file's bytes: what `serializeMovableScene` wrote, with the meshes' geometry. */
    readonly bytes: Uint8Array;
    readonly data: MovableSceneData;
    /** tinybvh's builds, in ms. */
    readonly buildMs: number;
}

/**
 * Builds a movable scene with tinybvh (`buildMovableScene`) and serializes it
 * with the meshes' geometry, so the page loads it with no model. Instances of
 * a mesh used once are static and merge (the level: the default scene's
 * ground, sphere and box); instances of a mesh used more than once can move
 * (the props: the three knots). With `mergeStaticInstances: false`, every
 * instance can move. Instance `i` reports `instanceIndex` `i`.
 */
export async function bakeMovableScene(scene: VanillaScene, options: SceneBakeOptions = {}, builder?: TinyBvhBuilder): Promise<BakedMovableScene> {
    const tiny = builder ?? (await createTinyBvhBuilder());
    const uses = new Map<number, number>();
    for (const instance of scene.instances) uses.set(instance.mesh, (uses.get(instance.mesh) ?? 0) + 1);
    const merge = options.mergeStaticInstances !== false;
    const instances: TraceSceneInstance[] = scene.instances.map((instance, index) => ({
        meshIndex: instance.mesh,
        instanceIndex: index,
        objectToWorld: instance.objectToWorld,
        material: instance.material,
        ...(merge && uses.get(instance.mesh) === 1 ? { static: true } : {}),
    }));
    const start = performance.now();
    const data = await buildMovableScene(scene.meshes, instances, {
        build: (geometry) => tiny.buildLayout(geometry, { maxLeafPrimitives: options.maxLeafPrimitives ?? DEFAULT_LEAF, triangleLeaves: options.triangleLeaves }).layout,
        triangleLeaves: options.triangleLeaves,
        nodeFormat: options.nodeFormat,
        mergeStaticInstances: options.mergeStaticInstances,
    });
    const buildMs = performance.now() - start;
    return { bytes: new Uint8Array(serializeMovableScene(data, { geometry: true })), data, buildMs };
}

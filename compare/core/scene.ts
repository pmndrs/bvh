/**
 * The scene for compare/scene: many objects sharing one mesh, placed from a
 * fixed seed, and the queries over them (camera frusta, boxes, rays), also
 * seeded. `moveObjects` gives every object a new transform, as a frame of an
 * animated scene would.
 */

import { createRandom } from '@pmndrs/bvh';
import { Euler, Frustum, Matrix4, PerspectiveCamera, Quaternion, Vector3 } from 'three';
import type { CompareGeometry, CompareRay, CompareScene, SceneObject, Vec3 } from '../adapters/types.ts';
import { torusKnot } from './geometry.ts';

/** Objects spread through a cube of `extent` units. */
export function makeScene(count: number, extent = 200, seed = 11): CompareScene {
    const mesh = torusKnot(64, 8);
    const random = createRandom(seed);
    const objects: SceneObject[] = [];
    const matrix = new Matrix4();
    for (let id = 0; id < count; id++) {
        place(matrix, random, extent);
        objects.push({ id, matrix: new Float32Array(matrix.elements) });
    }
    return { name: `${count.toLocaleString('en-US')} objects`, mesh, objects };
}

function place(matrix: Matrix4, random: () => number, extent: number): void {
    const position = new Vector3((random() - 0.5) * extent, (random() - 0.5) * extent, (random() - 0.5) * extent);
    const rotation = new Quaternion().setFromEuler(new Euler(random() * 6.28, random() * 6.28, random() * 6.28));
    const s = 0.5 + random() * 1.5;
    matrix.compose(position, rotation, new Vector3(s, s, s));
}

/** Every object moved a little and turned (frame `frame` of a seeded motion), matrices updated in place. */
export function moveObjects(scene: CompareScene, frame: number, extent = 200): void {
    const random = createRandom(1000 + frame);
    const matrix = new Matrix4();
    const step = new Matrix4();
    for (const object of scene.objects) {
        matrix.fromArray(object.matrix);
        step.makeRotationY((random() - 0.5) * 0.2).setPosition((random() - 0.5) * extent * 0.01, (random() - 0.5) * extent * 0.01, (random() - 0.5) * extent * 0.01);
        matrix.premultiply(step);
        object.matrix.set(matrix.elements);
    }
}

/** World box of an object: the mesh's box, its eight corners transformed (as every scene tree here boxes an object). */
export function worldBox(mesh: CompareGeometry, matrix: ArrayLike<number>, out: Float32Array | number[] = new Float32Array(6)): Float32Array | number[] {
    const b = mesh.bounds;
    out[0] = out[1] = out[2] = Infinity;
    out[3] = out[4] = out[5] = -Infinity;
    for (let c = 0; c < 8; c++) {
        const x = c & 1 ? b[3] : b[0];
        const y = c & 2 ? b[4] : b[1];
        const z = c & 4 ? b[5] : b[2];
        for (let k = 0; k < 3; k++) {
            const v = matrix[k] * x + matrix[4 + k] * y + matrix[8 + k] * z + matrix[12 + k];
            if (v < out[k]) out[k] = v;
            if (v > out[k + 3]) out[k + 3] = v;
        }
    }
    return out;
}

export interface FrustumQuery {
    readonly planes: Float32Array;
    readonly viewProjection: Float32Array;
}

/** Cameras inside the cube looking in random directions (60 degree field of view, far plane at half the cube). */
export function makeFrusta(count: number, extent = 200, seed = 12): FrustumQuery[] {
    const random = createRandom(seed);
    const camera = new PerspectiveCamera(60, 16 / 9, 0.1, extent / 2);
    const frustum = new Frustum();
    const viewProjection = new Matrix4();
    return Array.from({ length: count }, () => {
        camera.position.set((random() - 0.5) * extent, (random() - 0.5) * extent, (random() - 0.5) * extent);
        camera.lookAt(camera.position.x + random() - 0.5, camera.position.y + random() - 0.5, camera.position.z + random() - 0.5);
        camera.updateMatrixWorld();
        viewProjection.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
        frustum.setFromProjectionMatrix(viewProjection);
        const planes = new Float32Array(24);
        frustum.planes.forEach((plane, i) => planes.set([plane.normal.x, plane.normal.y, plane.normal.z, plane.constant], i * 4));
        return { planes, viewProjection: new Float32Array(viewProjection.elements) };
    });
}

/** Boxes with edges of 5% to 15% of the cube. */
export function makeSceneBoxes(count: number, extent = 200, seed = 13): number[][] {
    const random = createRandom(seed);
    return Array.from({ length: count }, () => {
        const c = [0, 1, 2].map(() => (random() - 0.5) * extent);
        const h = [0, 1, 2].map(() => (0.05 + random() * 0.1) * extent * 0.5);
        return [c[0] - h[0], c[1] - h[1], c[2] - h[2], c[0] + h[0], c[1] + h[1], c[2] + h[2]];
    });
}

/** Rays from random points of the cube in random directions. */
export function makeSceneRays(count: number, extent = 200, seed = 14): CompareRay[] {
    const random = createRandom(seed);
    return Array.from({ length: count }, () => {
        const origin: Vec3 = [(random() - 0.5) * extent, (random() - 0.5) * extent, (random() - 0.5) * extent];
        const z = random() * 2 - 1;
        const a = random() * Math.PI * 2;
        const r = Math.sqrt(1 - z * z);
        return { origin, direction: [r * Math.cos(a), r * Math.sin(a), z] };
    });
}

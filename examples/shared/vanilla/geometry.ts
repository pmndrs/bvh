/**
 * Scenes for the vanilla pages, with no engine: meshes as plain `positions`
 * and `indices` (`MeshGeometry`, what every builder and `SceneTrees` take) and
 * instances as column-major matrices with a material number.
 *
 * - Procedural shapes (torus knot, sphere, box, torus, ground), so every page
 *   works with no assets at all.
 * - {@link VanillaScene}: what `gltf.ts` reads from a glTF too.
 * - {@link mergeToWorld}: a whole scene as one world-space mesh, for the
 *   pages about one mesh's tree.
 */

import type { MeshGeometry } from '@pmndrs/bvh';
import { growBounds, mat4 } from './math.ts';

/** A mesh with an index (every shape here has one; the glTF reader makes one for unindexed primitives). */
export interface IndexedMesh extends MeshGeometry {
    readonly positions: Float32Array;
    readonly indices: Uint32Array;
}

export interface VanillaInstance {
    readonly mesh: number;
    /** Column-major object-to-world. */
    readonly objectToWorld: Float32Array;
    /** The material number hits report (`BvhHit.material`): the glTF material index, or a palette slot. */
    readonly material: number;
}

export interface VanillaScene {
    readonly name: string;
    readonly meshes: readonly IndexedMesh[];
    readonly instances: readonly VanillaInstance[];
    /** World box: `[minX, minY, minZ, maxX, maxY, maxZ]`. */
    readonly bounds: number[];
    /** Triangles over every instance. */
    readonly triangles: number;
    /** Fetch and parse milliseconds, for a loaded scene. */
    readonly loadMs?: number;
}

/** The torus knot three's `TorusKnotGeometry` makes (p = 2, q = 3), positions and indices only. */
export function torusKnot(radius = 1, tube = 0.3, tubularSegments = 600, radialSegments = 40, p = 2, q = 3): IndexedMesh {
    const curve = (u: number): [number, number, number] => {
        const cu = Math.cos(u);
        const su = Math.sin(u);
        const quOverP = (q / p) * u;
        const cs = Math.cos(quOverP);
        return [radius * (2 + cs) * 0.5 * cu, radius * (2 + cs) * su * 0.5, radius * Math.sin(quOverP) * 0.5];
    };
    const positions = new Float32Array((tubularSegments + 1) * (radialSegments + 1) * 3);
    let at = 0;
    for (let i = 0; i <= tubularSegments; i++) {
        const u = (i / tubularSegments) * p * Math.PI * 2;
        const p1 = curve(u);
        const p2 = curve(u + 0.01);
        const t = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        const n = [p2[0] + p1[0], p2[1] + p1[1], p2[2] + p1[2]];
        let b = [t[1] * n[2] - t[2] * n[1], t[2] * n[0] - t[0] * n[2], t[0] * n[1] - t[1] * n[0]];
        let nn = [b[1] * t[2] - b[2] * t[1], b[2] * t[0] - b[0] * t[2], b[0] * t[1] - b[1] * t[0]];
        const bl = Math.hypot(b[0], b[1], b[2]);
        const nl = Math.hypot(nn[0], nn[1], nn[2]);
        b = b.map((v) => v / bl);
        nn = nn.map((v) => v / nl);
        for (let j = 0; j <= radialSegments; j++) {
            const v = (j / radialSegments) * Math.PI * 2;
            const cx = -tube * Math.cos(v);
            const cy = tube * Math.sin(v);
            for (let k = 0; k < 3; k++) positions[at++] = p1[k] + (cx * nn[k] + cy * b[k]);
        }
    }
    return { positions, indices: gridIndices(tubularSegments, radialSegments) };
}

/** A UV sphere. */
export function sphere(radius = 1, widthSegments = 48, heightSegments = 24): IndexedMesh {
    const positions = new Float32Array((widthSegments + 1) * (heightSegments + 1) * 3);
    let at = 0;
    for (let y = 0; y <= heightSegments; y++) {
        const theta = (y / heightSegments) * Math.PI;
        for (let x = 0; x <= widthSegments; x++) {
            const phi = (x / widthSegments) * Math.PI * 2;
            positions[at++] = -radius * Math.cos(phi) * Math.sin(theta);
            positions[at++] = radius * Math.cos(theta);
            positions[at++] = radius * Math.sin(phi) * Math.sin(theta);
        }
    }
    return { positions, indices: withoutDegenerate(positions, gridIndices(heightSegments, widthSegments)) };
}

/** Drops zero-area triangles (a sphere's poles). */
function withoutDegenerate(positions: Float32Array, indices: Uint32Array): Uint32Array {
    const kept: number[] = [];
    for (let t = 0; t < indices.length; t += 3) {
        const [a, b, c] = [indices[t] * 3, indices[t + 1] * 3, indices[t + 2] * 3];
        const e1 = [positions[b] - positions[a], positions[b + 1] - positions[a + 1], positions[b + 2] - positions[a + 2]];
        const e2 = [positions[c] - positions[a], positions[c + 1] - positions[a + 1], positions[c + 2] - positions[a + 2]];
        const area = Math.hypot(e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]);
        if (area > 1e-12) kept.push(indices[t], indices[t + 1], indices[t + 2]);
    }
    return new Uint32Array(kept);
}

/** A torus around the y axis. */
export function torus(radius = 1, tube = 0.35, radialSegments = 20, tubularSegments = 48): IndexedMesh {
    const positions = new Float32Array((tubularSegments + 1) * (radialSegments + 1) * 3);
    let at = 0;
    for (let i = 0; i <= tubularSegments; i++) {
        const u = (i / tubularSegments) * Math.PI * 2;
        for (let j = 0; j <= radialSegments; j++) {
            const v = (j / radialSegments) * Math.PI * 2;
            positions[at++] = (radius + tube * Math.cos(v)) * Math.cos(u);
            positions[at++] = tube * Math.sin(v);
            positions[at++] = (radius + tube * Math.cos(v)) * Math.sin(u);
        }
    }
    return { positions, indices: gridIndices(tubularSegments, radialSegments) };
}

/** An axis-aligned box centred on the origin (24 vertices, 12 triangles, outward faces). */
export function box(width = 1, height = 1, depth = 1): IndexedMesh {
    const [x, y, z] = [width / 2, height / 2, depth / 2];
    // prettier-ignore
    const faces = [
        [[x, -y, z], [x, -y, -z], [x, y, -z], [x, y, z]], [[-x, -y, -z], [-x, -y, z], [-x, y, z], [-x, y, -z]],
        [[-x, y, z], [x, y, z], [x, y, -z], [-x, y, -z]], [[-x, -y, -z], [x, -y, -z], [x, -y, z], [-x, -y, z]],
        [[-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]], [[x, -y, -z], [-x, -y, -z], [-x, y, -z], [x, y, -z]],
    ];
    const positions = new Float32Array(faces.flat(2));
    const indices = new Uint32Array(36);
    for (let face = 0; face < 6; face++) indices.set([0, 1, 2, 0, 2, 3].map((corner) => face * 4 + corner), face * 6);
    return { positions, indices };
}

/** A square in the xz plane facing +y, split into `segments`² quads. */
export function ground(size = 10, segments = 8): IndexedMesh {
    const positions = new Float32Array((segments + 1) * (segments + 1) * 3);
    let at = 0;
    for (let i = 0; i <= segments; i++) {
        for (let j = 0; j <= segments; j++) {
            positions[at++] = (j / segments - 0.5) * size;
            positions[at++] = 0;
            positions[at++] = (i / segments - 0.5) * size;
        }
    }
    const indices = new Uint32Array(segments * segments * 6);
    at = 0;
    for (let i = 0; i < segments; i++) {
        for (let j = 0; j < segments; j++) {
            const a = i * (segments + 1) + j;
            const b = a + segments + 1;
            indices.set([a, b, a + 1, a + 1, b, b + 1], at);
            at += 6;
        }
    }
    return { positions, indices };
}

/** Two triangles per cell of a `(rows + 1) x (columns + 1)` vertex grid. */
function gridIndices(rows: number, columns: number): Uint32Array {
    const indices = new Uint32Array(rows * columns * 6);
    let at = 0;
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < columns; j++) {
            const a = i * (columns + 1) + j;
            const b = (i + 1) * (columns + 1) + j;
            indices.set([a, b, a + 1, b, b + 1, a + 1], at);
            at += 6;
        }
    }
    return indices;
}

/** A scene from meshes and instances: its world box and triangle count. */
export function makeScene(name: string, meshes: readonly IndexedMesh[], instances: readonly VanillaInstance[], loadMs?: number): VanillaScene {
    const local = meshes.map((mesh) => growBounds(mesh.positions));
    const bounds = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    let triangles = 0;
    for (const instance of instances) {
        const box = local[instance.mesh];
        // The 8 corners of the mesh's box through the matrix: a box around the instance.
        for (let corner = 0; corner < 8; corner++) {
            const p = mat4.transformPoint(instance.objectToWorld, [box[corner & 1 ? 3 : 0], box[corner & 2 ? 4 : 1], box[corner & 4 ? 5 : 2]]);
            for (let k = 0; k < 3; k++) {
                bounds[k] = Math.min(bounds[k], p[k]);
                bounds[k + 3] = Math.max(bounds[k + 3], p[k]);
            }
        }
        triangles += meshes[instance.mesh].indices.length / 3;
    }
    return { name, meshes, instances, bounds, triangles, loadMs };
}

/** One mesh, one instance. */
export function singleMeshScene(name: string, mesh: IndexedMesh): VanillaScene {
    return makeScene(name, [mesh], [{ mesh: 0, objectToWorld: mat4.identity(), material: 0 }]);
}

/** A small showcase: a ground, a torus knot, spheres, boxes and tori, each with its own material number. */
export function showcaseScene(): VanillaScene {
    const meshes = [ground(14, 16), torusKnot(1, 0.3, 400, 32), sphere(0.7), box(1, 1, 1), torus(0.6, 0.2)];
    const instances: VanillaInstance[] = [{ mesh: 0, objectToWorld: mat4.identity(), material: 0 }];
    instances.push({ mesh: 1, objectToWorld: mat4.compose([0, 1.9, 0], [0, 0, 0, 1], [1.2, 1.2, 1.2]), material: 1 });
    const ring = 8;
    for (let i = 0; i < ring; i++) {
        const angle = (i / ring) * Math.PI * 2;
        const at: [number, number, number] = [Math.cos(angle) * 4.2, 0, Math.sin(angle) * 4.2];
        const kind = 2 + (i % 3);
        const spin = mat4.axisAngle([0, 1, 0], angle);
        const lift = kind === 2 ? 0.7 : kind === 3 ? 0.5 : 0.2;
        instances.push({ mesh: kind, objectToWorld: mat4.compose([at[0], lift, at[2]], spin, [1, 1, 1]), material: 2 + i });
    }
    return makeScene('showcase', meshes, instances);
}

/** Every instance of `scene` baked into one world-space mesh (for the pages about one mesh's tree). */
export function mergeToWorld(scene: VanillaScene): IndexedMesh {
    if (scene.instances.length === 1 && isIdentity(scene.instances[0].objectToWorld)) return scene.meshes[scene.instances[0].mesh];
    let vertexCount = 0;
    let indexCount = 0;
    for (const instance of scene.instances) {
        vertexCount += scene.meshes[instance.mesh].positions.length / 3;
        indexCount += scene.meshes[instance.mesh].indices.length;
    }
    const positions = new Float32Array(vertexCount * 3);
    const indices = new Uint32Array(indexCount);
    let vertexBase = 0;
    let indexAt = 0;
    for (const instance of scene.instances) {
        const mesh = scene.meshes[instance.mesh];
        const m = instance.objectToWorld;
        const source = mesh.positions;
        for (let i = 0; i < source.length; i += 3) {
            const [x, y, z] = [source[i], source[i + 1], source[i + 2]];
            const at = vertexBase * 3 + i;
            positions[at] = m[0] * x + m[4] * y + m[8] * z + m[12];
            positions[at + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
            positions[at + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
        }
        for (let i = 0; i < mesh.indices.length; i++) indices[indexAt + i] = mesh.indices[i] + vertexBase;
        vertexBase += source.length / 3;
        indexAt += mesh.indices.length;
    }
    return { positions, indices };
}

function isIdentity(m: ArrayLike<number>): boolean {
    for (let i = 0; i < 16; i++) if (m[i] !== (i % 5 === 0 ? 1 : 0)) return false;
    return true;
}

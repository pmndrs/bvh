/**
 * three/sdf: a signed distance field generated on the GPU from our trees, in
 * the manner of three-mesh-bvh's `webgpu_sdfGeneration`.
 *
 * - The scene goes into `SceneTrees` on the renderer's device (GPU H-PLOC
 *   first trees, tinybvh's from the worker pool swapped in), as any example.
 * - `generateSdfTexture` (`@pmndrs/bvh/three`) fills a three `Storage3DTexture`
 *   (`r32float`) from those trees with the core's `SdfGenerator`: per voxel the
 *   distance to the nearest triangle (`bvhClosestPoint`), negative where the
 *   inside test's three rays vote inside (`bvhInsideVotes`, parity or winding).
 * - Shown three ways: a TSL raymarch of the field through the box (sphere
 *   tracing, normals from the field's gradient), a slice with distance
 *   contours, or the geometry itself.
 *
 * The meshes are welded on load (`mergeVertices` on positions): three's
 * built-in geometries repeat the vertices along their UV seams, and a seam
 * whose two copies round apart leaves a crack the inside test's rays can slip
 * through. `?model=` loads a glTF; drop one on the page.
 */

import { BackSide, Box3, Box3Helper, BoxGeometry, Color, DoubleSide, Group, Mesh, MeshStandardMaterial, PlaneGeometry, RenderTarget, TorusKnotGeometry, Vector3, type BufferGeometry, type Object3D } from 'three';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import { abs, Break, cameraPosition, Discard, float, Fn, fract, If, Loop, max, min, mix, normalize, positionWorld, smoothstep, uniform, vec3, vec4 } from 'three/tsl';
import { CpuMeshBvh, readSdf, SceneTrees, sdfValueCpu, sdfVoxelCenter, type SdfSign } from '@pmndrs/bvh';
import { generateSdfTexture, SceneTreesObject3DSync, type BvhSdfTexture, type BvhTslNode } from '@pmndrs/bvh/three';
import { createApp } from '../../shared/app.ts';
import { enableDrop, frameObject, loadGltf, meshesOf, requestedModel, type LoadedModel } from '../../shared/loader.ts';
import { SceneTreesLog } from '../../shared/stats.ts';
import { isSmoke, ready } from '../../shared/smoke.ts';

const app = await createApp('SDF generation', 'three/sdf');
const { device, renderer, scene, camera, controls, overlay } = app;

type Mode = 'raymarch' | 'slice' | 'geometry';
const params = {
    resolution: isSmoke ? 32 : 64,
    margin: 0.1,
    sign: 'parity' as SdfSign,
    mode: 'raymarch' as Mode,
    surface: 0,
    slice: 0.5,
};

let root: Object3D = new Group();
let trees: SceneTrees | null = null;
let sync: SceneTreesObject3DSync | null = null;
let log = new SceneTreesLog();
let sdf: BvhSdfTexture | null = null;
let generating: Promise<void> | null = null;

const boxHelper = new Box3Helper(new Box3(), new Color(0xffc878));
scene.add(boxHelper);

// The raymarch: a box mesh the size of the field, drawn from inside (BackSide) so the camera may enter it.
const surface = uniform(0);
const boxMin = uniform(new Vector3());
const boxMax = uniform(new Vector3());
const voxel = uniform(new Vector3());
const raymarchMaterial = new MeshBasicNodeMaterial({ side: BackSide, transparent: false });
const raymarchMesh = new Mesh(new BoxGeometry(1, 1, 1), raymarchMaterial);
// The slice: a plane across the box at a height, coloured by the field with contour lines.
const sliceMaterial = new MeshBasicNodeMaterial({ side: DoubleSide });
const sliceMesh = new Mesh(new PlaneGeometry(1, 1), sliceMaterial);
sliceMesh.rotation.x = -Math.PI / 2;
scene.add(raymarchMesh, sliceMesh);

/** Sphere tracing through the box: steps of the field's value (at least a tenth of a voxel), 192 at most. */
function raymarchNode(field: BvhSdfTexture): BvhTslNode {
    return Fn(() => {
        const origin = cameraPosition;
        const direction = normalize(positionWorld.sub(cameraPosition));
        const inverse = vec3(1).div(direction);
        const toMin = boxMin.sub(origin).mul(inverse);
        const toMax = boxMax.sub(origin).mul(inverse);
        const near = min(toMin, toMax);
        const far = max(toMin, toMax);
        const tEnter = max(max(near.x, near.y), max(near.z, 0));
        const tExit = min(min(far.x, far.y), far.z);
        const t = float(tEnter).toVar();
        const hit = float(0).toVar();
        const minStep = min(voxel.x, min(voxel.y, voxel.z)).mul(0.1);
        Loop(192, () => {
            const p = origin.add(direction.mul(t));
            const d = field.sample(p).sub(surface);
            If(d.lessThan(minStep), () => {
                hit.assign(1);
                Break();
            });
            t.addAssign(max(d, minStep));
            If(t.greaterThan(tExit), () => {
                Break();
            });
        });
        Discard(hit.lessThan(0.5));
        const p = origin.add(direction.mul(t));
        // The field's gradient by central differences one voxel apart.
        const ex = vec3(voxel.x, 0, 0);
        const ey = vec3(0, voxel.y, 0);
        const ez = vec3(0, 0, voxel.z);
        const normal = normalize(vec3(field.sample(p.add(ex)).sub(field.sample(p.sub(ex))), field.sample(p.add(ey)).sub(field.sample(p.sub(ey))), field.sample(p.add(ez)).sub(field.sample(p.sub(ez)))));
        const light = normal.dot(normalize(vec3(0.5, 0.8, 0.3))).mul(0.5).add(0.5);
        return vec4(mix(vec3(0.12, 0.16, 0.28), vec3(0.56, 0.7, 1), light), 1);
    })();
}

/** Inside blue, outside orange, a line every tenth of the box's size, the surface in white. */
function sliceNode(field: BvhSdfTexture, size: number): BvhTslNode {
    return Fn(() => {
        const d = field.sample(positionWorld);
        const base = mix(vec3(0.95, 0.55, 0.25), vec3(0.3, 0.5, 0.95), d.lessThan(0).select(1, 0));
        const bands = abs(fract(d.mul(10 / size)).sub(0.5)).mul(2);
        const contour = smoothstep(0.85, 1, bands).mul(0.35);
        const zero = smoothstep(size * 0.006, 0, abs(d));
        return vec4(mix(base.mul(float(1).sub(contour)), vec3(1), zero), 1);
    })();
}

/** Positions only, welded, so seams are shared vertices (the inside test needs a closed surface). */
function welded(geometry: BufferGeometry): BufferGeometry {
    const positionsOnly = geometry.clone();
    for (const name of Object.keys(positionsOnly.attributes)) if (name !== 'position') positionsOnly.deleteAttribute(name);
    const merged = mergeVertices(positionsOnly);
    merged.computeVertexNormals();
    return merged;
}

function defaultScene(): Object3D {
    const group = new Group();
    group.add(new Mesh(welded(new TorusKnotGeometry(1, 0.3, 400, 48)), new MeshStandardMaterial({ color: 0x8fb3ff, roughness: 0.5 })));
    return group;
}

async function useScene(next: Object3D, label: string): Promise<void> {
    await generating;
    scene.remove(root);
    sync?.dispose();
    trees?.dispose();
    root = next;
    for (const mesh of meshesOf(root)) mesh.geometry = welded(mesh.geometry);
    scene.add(root);
    frameObject(root, camera, controls);
    overlay.set('scene', label);
    log = new SceneTreesLog();
    trees = await SceneTrees.create(device, { label: 'sdf', onTree: log.onTree });
    sync = new SceneTreesObject3DSync(trees, root);
    updateTrees();
    await trees.whenBuilt();
    updateTrees();
    overlay.build(log.build(trees));
    overlay.set('triangles', meshesOf(root).reduce((sum, mesh) => sum + (mesh.geometry.index?.count ?? 0) / 3, 0));
    await generate();
}

function updateTrees(): void {
    if (!trees || !sync) return;
    root.updateMatrixWorld(true);
    sync.update();
    const encoder = device.createCommandEncoder();
    trees.update(encoder);
    device.queue.submit([encoder.finish()]);
}

async function generate(): Promise<void> {
    if (!trees) return;
    const run = async () => {
        root.updateMatrixWorld(true);
        const bounds = new Box3().setFromObject(root);
        const margin = bounds.getSize(new Vector3()).length() * params.margin;
        const box = bounds.clone().expandByScalar(margin);
        const previous = sdf;
        sdf = await generateSdfTexture(renderer, trees!, { box, resolution: params.resolution, sign: params.sign });
        previous?.dispose();
        const voxels = params.resolution ** 3;
        overlay.set('SDF', `${params.resolution}³ voxels, ${params.sign}`);
        overlay.set('SDF build', `${sdf.ms.toFixed(1)} ms (wall, submit to done)`);
        overlay.set('voxels/s', `${(voxels / sdf.ms / 1e3).toFixed(2)} M`);
        boxHelper.box.copy(box);
        boxMin.value.copy(box.min);
        boxMax.value.copy(box.max);
        const size = box.getSize(new Vector3());
        voxel.value.copy(size).divideScalar(params.resolution);
        raymarchMesh.position.copy(box.getCenter(new Vector3()));
        raymarchMesh.scale.copy(size);
        raymarchMaterial.colorNode = raymarchNode(sdf);
        raymarchMaterial.needsUpdate = true;
        sliceMesh.scale.set(size.x, size.z, 1);
        sliceMaterial.colorNode = sliceNode(sdf, size.length());
        sliceMaterial.needsUpdate = true;
        placeSlice();
        showMode();
    };
    generating = run();
    await generating;
    generating = null;
}

function placeSlice(): void {
    if (!sdf) return;
    sliceMesh.position.copy(sdf.box.getCenter(new Vector3()));
    sliceMesh.position.y = sdf.box.min.y + (sdf.box.max.y - sdf.box.min.y) * params.slice;
}

function showMode(): void {
    raymarchMesh.visible = params.mode === 'raymarch';
    sliceMesh.visible = params.mode === 'slice';
    root.visible = params.mode !== 'raymarch';
}

overlay.select('show', ['raymarch', 'slice', 'geometry'] as const, params.mode, (mode) => {
    params.mode = mode;
    showMode();
});
overlay.range('surface', -0.2, 0.3, 0.01, params.surface, (value) => (surface.value = params.surface = value));
overlay.range('slice height', 0, 1, 0.01, params.slice, (value) => {
    params.slice = value;
    placeSlice();
});
overlay.range('resolution', 16, 160, 8, params.resolution, (value) => (params.resolution = value));
overlay.range('margin', 0, 0.3, 0.01, params.margin, (value) => (params.margin = value));
overlay.select('sign', ['parity', 'winding', 'first-hit', 'unsigned'] as const, params.sign, (value) => (params.sign = value));
overlay.button('regenerate', () => void generate());
overlay.note('The field is generated on the GPU from the same trees the walk traces. Parity: an odd number of crossings is inside; winding: a non-zero signed sum (overlapping parts stay solid); first hit: the side of the first triangle one ray hits (cheaper, closed meshes only).');

enableDrop(
    (model: LoadedModel) => useScene(model.root, model.name),
    (error) => overlay.error(String(error)),
);
const model = requestedModel();
if (model) {
    const loaded = await loadGltf(model);
    await useScene(loaded.root, loaded.name);
} else {
    await useScene(defaultScene(), 'torus knot (welded), 38k triangles');
}

/**
 * The smoke check: 256 voxels against the CPU reference (`sdfValueCpu` over a
 * `CpuMeshBvh` of the world-space geometry): distances within 1e-5 of the
 * box's size, signs equal away from the surface.
 */
async function checkAgainstCpu(): Promise<number> {
    const field = sdf!;
    const values = await readSdf(device, { texture: field.gpuTexture }, field.grid);
    const positions: number[] = [];
    const indices: number[] = [];
    for (const mesh of meshesOf(root)) {
        const position = mesh.geometry.getAttribute('position');
        const base = positions.length / 3;
        for (let i = 0; i < position.count; i++) {
            const p = new Vector3(position.getX(i), position.getY(i), position.getZ(i)).applyMatrix4(mesh.matrixWorld);
            positions.push(p.x, p.y, p.z);
        }
        const index = mesh.geometry.index!;
        for (let i = 0; i < index.count; i++) indices.push(base + index.getX(i));
    }
    const tree = await CpuMeshBvh.create({ positions: new Float32Array(positions), indices: new Uint32Array(indices) });
    const size = field.box.getSize(new Vector3()).length();
    const n = params.resolution;
    let checked = 0;
    for (let k = 0; k < 256; k++) {
        const x = (k * 37) % n, y = (k * 61 + 3) % n, z = (k * 17 + 7) % n;
        const expected = sdfValueCpu([tree], sdfVoxelCenter(field.grid, x, y, z), { sign: params.sign });
        const got = values[(z * n + y) * n + x];
        if (Math.abs(Math.abs(got) - Math.abs(expected)) > 1e-5 * size) throw new Error(`voxel (${x}, ${y}, ${z}): GPU ${got}, CPU ${expected}`);
        if (Math.abs(expected) > 1e-4 * size && Math.sign(got) !== Math.sign(expected)) throw new Error(`voxel (${x}, ${y}, ${z}) sign: GPU ${got}, CPU ${expected}`);
        checked++;
    }
    return checked;
}

/** Share of a 256 x 256 view of the raymarch alone (the box helper hidden) that the field's surface covers. */
async function raymarchCoverage(): Promise<number> {
    const target = new RenderTarget(256, 256);
    boxHelper.visible = false;
    renderer.setRenderTarget(target);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    boxHelper.visible = true;
    const pixels = (await renderer.readRenderTargetPixelsAsync(target, 0, 0, 256, 256)) as Uint8Array;
    target.dispose();
    let covered = 0;
    // The background's blue is 0x1d; the shading's is at least 0.28 (about 0x47).
    for (let i = 0; i < 256 * 256; i++) if (pixels[i * 4 + 2] > 0x38) covered++;
    return covered / (256 * 256);
}

let frames = 0;
app.loop(async () => {
    if (isSmoke && ++frames === 10) {
        const checked = await checkAgainstCpu();
        // The other two views compile and draw too.
        for (const mode of ['slice', 'geometry', 'raymarch'] as const) {
            params.mode = mode;
            showMode();
            renderer.render(scene, camera);
        }
        const coverage = await raymarchCoverage();
        if (!(coverage > 0.02)) throw new Error(`the raymarch drew the surface on ${(coverage * 100).toFixed(2)}% of the view`);
        await ready(device, { ...overlay.snapshot(), checkedAgainstCpu: checked, raymarchCoverage: coverage });
    }
});

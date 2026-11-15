/**
 * The scene loader every example shares:
 *
 * - **defaults** made from three's built-in geometries, so every page works
 *   with no assets at all;
 * - **bundled assets** by URL: `?model=downloads/wind-tree/wind-tree.glb`
 *   loads a glTF from the assets base, which is `/assets/` (the dev server maps
 *   it to `BVH_ASSETS_DIR`, see vite.config.ts) or `?assets=<url>`;
 * - **drag and drop** of a `.glb`, or a `.gltf` with its `.bin` and textures
 *   dropped together.
 */

import { Box3, LoadingManager, type Mesh, type Object3D, Vector3, type PerspectiveCamera } from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import type { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { assetLoadError, assetUrl } from './assets.ts';

// The asset base and `?model=` live in assets.ts, shared with the three-free glTF reader (vanilla/gltf.ts).
export { assetsBase, requestedModel } from './assets.ts';

export interface LoadedModel {
    readonly root: Object3D;
    readonly name: string;
    /** Fetch and parse milliseconds. */
    readonly loadMs: number;
}

/** Loads a glTF from a URL (`?model=` paths resolve against the assets base). */
export async function loadGltf(path: string, manager?: LoadingManager): Promise<LoadedModel> {
    const url = assetUrl(path);
    const start = performance.now();
    const gltf = await new GLTFLoader(manager).loadAsync(url).catch((error: unknown) => {
        throw assetLoadError(path, error);
    });
    return { root: gltf.scene, name: path.split('/').pop() ?? path, loadMs: performance.now() - start };
}

/**
 * Calls `onModel` with each glTF dropped on the page. A `.gltf` finds its
 * `.bin` and textures among the files dropped with it.
 */
export function enableDrop(onModel: (model: LoadedModel) => void | Promise<void>, onError: (error: unknown) => void): void {
    const hint = document.createElement('div');
    hint.className = 'bvh-drop';
    hint.textContent = 'Drop a .glb (or a .gltf with its files)';
    document.body.append(hint);
    addEventListener('dragover', (event) => {
        event.preventDefault();
        hint.style.display = 'grid';
    });
    addEventListener('dragleave', (event) => {
        if (event.relatedTarget === null) hint.style.display = 'none';
    });
    addEventListener('drop', (event) => {
        event.preventDefault();
        hint.style.display = 'none';
        const files = [...(event.dataTransfer?.files ?? [])];
        const main = files.find((file) => /\.(glb|gltf)$/i.test(file.name));
        if (!main) {
            onError(new Error('drop a .glb or .gltf file'));
            return;
        }
        const urls = new Map(files.map((file) => [file.name, URL.createObjectURL(file)]));
        const manager = new LoadingManager();
        manager.setURLModifier((url) => urls.get(decodeURIComponent(url.split('/').pop() ?? url)) ?? url);
        loadGltf(urls.get(main.name)!, manager)
            .then((model) => onModel({ ...model, name: main.name }))
            .catch(onError)
            .finally(() => urls.forEach((url) => URL.revokeObjectURL(url)));
    });
}

/** Every `Mesh` under `root` (InstancedMesh, BatchedMesh and SkinnedMesh included). */
export function meshesOf(root: Object3D): Mesh[] {
    const meshes: Mesh[] = [];
    root.traverse((object) => {
        if ((object as Mesh).isMesh) meshes.push(object as Mesh);
    });
    return meshes;
}

/** Points `camera` and `controls` at `object`'s world box. */
export function frameObject(object: Object3D, camera: PerspectiveCamera, controls?: OrbitControls): Box3 {
    object.updateMatrixWorld(true);
    const box = new Box3().setFromObject(object);
    if (box.isEmpty()) return box;
    const center = box.getCenter(new Vector3());
    const radius = box.getSize(new Vector3()).length() / 2;
    const distance = radius / Math.sin((camera.fov * Math.PI) / 360);
    camera.position.copy(center).add(new Vector3(0.6, 0.45, 0.75).normalize().multiplyScalar(distance * 0.9));
    camera.near = Math.max(radius / 1000, 1e-3);
    camera.far = distance * 10;
    camera.updateProjectionMatrix();
    controls?.target.copy(center);
    controls?.update();
    return box;
}

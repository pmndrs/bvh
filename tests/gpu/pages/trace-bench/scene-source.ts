/**
 * Finds a benchmark scene's geometry: reads the scene manifest and downloads the
 * model file, both from outside this repository (the test server serves them from
 * BVH_MANIFESTS_DIR and BVH_ASSETS_DIR; see tests/gpu/README.md), and returns
 * meshes, instances, the probe volume and the sun direction.
 *
 * Browser only: it uses `fetch` against the test site's server.
 */

import { ASSETS_BASE, ASSETS_LIST, MANIFESTS_BASE } from '../../contract.ts';
import type { ModelPart, SceneManifest } from '../../scene-manifest.ts';
import { BENCH_SCENE_SUBSETS, FALLBACK_TOWARD_SUN } from './config.ts';
import { fetchGltfGeometry, modelTransformMatrix, type BenchInstance, type BenchMeshGeometry } from './gltf-geometry.ts';
import { normalizeVector, worldBounds, type Box, type Vec3 } from './rays.ts';

/** Folder under the assets directory that holds the downloaded models. */
const DOWNLOAD_FOLDER = 'downloads';

interface AssetListing {
    assets: string[];
    manifests: string[];
}

/**
 * Which files the server has, known without probing. Fetching a missing file would log a 404,
 * and the test runner fails a page on any logged error.
 */
let listing: Promise<AssetListing> | null = null;
function listAssets(): Promise<AssetListing> {
    listing ??= fetch(ASSETS_LIST).then((response) => response.json() as Promise<AssetListing>);
    return listing;
}

export async function findManifest(id: string): Promise<SceneManifest> {
    const { manifests } = await listAssets();
    const file = `${id}.json`;
    if (!manifests.includes(file)) {
        throw new Error(
            `no scene manifest "${file}" (known: ${manifests.map((name) => name.replace(/\.json$/, '')).sort().join(', ') || 'none'}). ` +
                'Set BVH_MANIFESTS_DIR, see tests/gpu/README.md.',
        );
    }
    const response = await fetch(`${MANIFESTS_BASE}${file}`);
    if (!response.ok) throw new Error(`could not read manifest ${file}: HTTP ${response.status}`);
    return (await response.json()) as SceneManifest;
}

async function findModelUrl(asset: string, path: string): Promise<string> {
    const { assets } = await listAssets();
    const file = `${DOWNLOAD_FOLDER}/${asset}/${path}`;
    if (!assets.includes(file)) {
        throw new Error(`model file not found (looked for ${file}). Set BVH_ASSETS_DIR, see tests/gpu/README.md.`);
    }
    return `${ASSETS_BASE}${file}`;
}

export interface BenchScene {
    id: string;
    meshes: BenchMeshGeometry[];
    /** The tracer's instance index is the position in this array. */
    instances: BenchInstance[];
    /** Triangles stored once per mesh. */
    uniqueTriangles: number;
    /** Triangles counting every instance. */
    instancedTriangles: number;
    /** Box around all geometry. */
    geometryBounds: Box;
    /** The box the probe grid fills. */
    probeBox: Box;
    probeBoxSource: 'manifest' | 'scene';
    /** Unit vector from a surface toward the sun. */
    towardSun: Vec3;
    sunSource: 'manifest' | 'fallback';
    skippedPrimitives: number;
    /** Where the model came from (for the record). */
    modelFiles: string[];
    loadMs: number;
}

function toSun(manifest: SceneManifest): { towardSun: Vec3; source: BenchScene['sunSource'] } {
    const sun = manifest.lights.find((light) => light.type === 'directional');
    if (sun?.type === 'directional') {
        const travel = normalizeVector(sun.direction);
        return { towardSun: [-travel[0], -travel[1], -travel[2]], source: 'manifest' };
    }
    return { towardSun: normalizeVector(FALLBACK_TOWARD_SUN), source: 'fallback' };
}

export function volumeBox(volume: SceneManifest['volume']): Box {
    return {
        min: [0, 1, 2].map((axis) => volume.center[axis] - volume.size[axis] / 2) as Vec3,
        max: [0, 1, 2].map((axis) => volume.center[axis] + volume.size[axis] / 2) as Vec3,
    };
}

/**
 * Downloads and flattens a manifest scene, or a subset of one
 * (`BENCH_SCENE_SUBSETS`: only the listed assets, with the manifest's volume
 * and sun). `volume` picks which box the probes fill.
 */
export async function loadBenchScene(id: string, volume: 'manifest' | 'scene'): Promise<BenchScene> {
    const start = performance.now();
    const subset = BENCH_SCENE_SUBSETS[id];
    const manifest = await findManifest(subset?.manifest ?? id);
    const meshes: BenchMeshGeometry[] = [];
    const instances: BenchInstance[] = [];
    const modelFiles: string[] = [];
    let skippedPrimitives = 0;

    const parts = subset ? manifest.parts.filter((part) => part.kind === 'model' && subset.assets.includes(part.asset)) : manifest.parts;
    if (subset && parts.length !== subset.assets.length) {
        throw new Error(`scene "${id}" needs the parts ${subset.assets.join(', ')} of "${subset.manifest}", and the manifest doesn't have them all`);
    }
    for (const part of parts) {
        if (part.kind !== 'model') throw new Error(`scene "${id}" has a procedural part; the benchmark only loads model files`);
        const modelPart: ModelPart = part;
        const url = await findModelUrl(modelPart.asset, modelPart.path);
        modelFiles.push(url.slice(ASSETS_BASE.length));
        const geometry = await fetchGltfGeometry(url, modelTransformMatrix(modelPart.transform));
        const meshOffset = meshes.length;
        meshes.push(...geometry.meshes);
        for (const instance of geometry.instances) {
            instances.push({ meshIndex: instance.meshIndex + meshOffset, objectToWorld: instance.objectToWorld });
        }
        skippedPrimitives += geometry.skippedPrimitives;
    }
    if (instances.length === 0) throw new Error(`scene "${id}" has no triangles`);

    const geometryBounds = worldBounds(meshes, instances);
    const uniqueTriangles = meshes.reduce((total, mesh) => total + mesh.indices.length / 3, 0);
    const instancedTriangles = instances.reduce((total, instance) => total + meshes[instance.meshIndex].indices.length / 3, 0);
    const sun = toSun(manifest);
    return {
        id,
        meshes,
        instances,
        uniqueTriangles,
        instancedTriangles,
        geometryBounds,
        probeBox: volume === 'manifest' ? volumeBox(manifest.volume) : geometryBounds,
        probeBoxSource: volume,
        towardSun: sun.towardSun,
        sunSource: sun.source,
        skippedPrimitives,
        modelFiles,
        loadMs: performance.now() - start,
    };
}

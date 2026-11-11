/**
 * The part of a webgpu-ddgi scene manifest (`scenes/manifests/<id>.json`) that the benches and
 * tools read: which model files make up the scene, its directional light and its probe volume.
 * The manifests and the models they name live outside this repository (BVH_MANIFESTS_DIR,
 * BVH_ASSETS_DIR; see tests/gpu/README.md).
 */

export type Vec3 = [number, number, number];

export interface ModelTransform {
    /** Metres, applied last. */
    position: Vec3;
    /** Euler XYZ in degrees. */
    rotationDeg: Vec3;
    /** Uniform scale that brings the model to metres. */
    scale: number;
}

export interface ModelPart {
    kind: 'model';
    /** An asset folder name under `<assets>/downloads/`. */
    asset: string;
    /** The model file, relative to `<assets>/downloads/<asset>/`. */
    path: string;
    transform: ModelTransform;
}

/** A part the benches don't load (procedural geometry, light sets). */
export interface OtherPart {
    kind: 'procedural' | 'light-set';
}

export type ScenePart = ModelPart | OtherPart;

export type ManifestLight = { type: 'directional'; direction: Vec3 } | { type: 'point' | 'spot' | 'area' };

export interface SceneManifest {
    id: string;
    parts: ScenePart[];
    lights: ManifestLight[];
    /** The box the probe grid fills. */
    volume: { center: Vec3; size: Vec3 };
    /** Camera bookmarks; the surface-ray benches place their screen probes from the first. */
    cameras?: { name: string; position: Vec3; target: Vec3; fovYDeg: number }[];
}

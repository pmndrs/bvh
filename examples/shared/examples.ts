/**
 * Every example page: the index page links them, the Vite build takes them as
 * inputs, and the GPU runner's `--examples` mode smoke-runs each. Plain data
 * (no DOM), so Node can import it too.
 */

export interface ExampleEntry {
    /** Runner and build name, for example `three-raycast`. */
    readonly name: string;
    /** The page's folder under examples/, for example `three/raycast`. */
    readonly path: string;
    readonly title: string;
    readonly description: string;
    /** Smoke timeout in milliseconds. */
    readonly timeoutMs: number;
}

export const EXAMPLES: readonly ExampleEntry[] = [
    {
        name: 'three-raycast',
        path: 'three/raycast',
        title: 'Raycast',
        description: 'acceleratedRaycast against three\'s own Mesh.raycast, rays per second live, plus a GPU batch of thousands of rays in one dispatch through SceneTrees and TraceBatchKernel.',
        timeoutMs: 60_000,
    },
    {
        name: 'three-pathtracer',
        path: 'three/pathtracer',
        title: 'Path tracer',
        description: 'A small TSL compute path tracer that reads SceneTrees through BvhTraceNodes: closest hits, any-hit shadows, progressive accumulation.',
        timeoutMs: 90_000,
    },
    {
        name: 'three-skinned',
        path: 'three/skinned',
        title: 'Skinned',
        description: 'BvhPosePass skins the vertices on the GPU, SceneTrees refits the tree from that buffer, and GPU picking keeps hitting the moving mesh.',
        timeoutMs: 60_000,
    },
    {
        name: 'three-instances',
        path: 'three/instances',
        title: 'Instances',
        description: 'InstancedMesh and BatchedMesh with thousands of moving instances: a top-level refit every frame, GPU picking and per-instance highlight.',
        timeoutMs: 60_000,
    },
    {
        name: 'three-load-gltf',
        path: 'three/load-gltf',
        title: 'Load glTF (fast first)',
        description: 'GPU H-PLOC trees first, tinybvh trees from the worker pool swapped in mesh by mesh, with a per-mesh timeline. Drop a glTF.',
        timeoutMs: 180_000,
    },
    {
        name: 'three-inspector',
        path: 'three/inspector',
        title: 'Inspector',
        description: 'MeshBVHHelper boxes by depth, stats per builder (tinybvh, H-PLOC, LBVH), leaf size and layout options, and a heat map of nodes visited per ray.',
        timeoutMs: 90_000,
    },
];

/** Where an example's source is browsed. */
export const SOURCE_TREE = 'https://github.com/pmndrs/bvh/tree/main/examples/';

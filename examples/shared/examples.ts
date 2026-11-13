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
        name: 'vanilla-hello',
        path: 'vanilla/hello',
        title: 'Hello',
        description: 'The smallest use, raw WebGPU and no engine: createMeshLayoutBuilder picks the builder and says why, the tree\'s boxes in a line pass, a picking ray and a batch of thousands whose hits are drawn straight from the GPU buffer.',
        timeoutMs: 60_000,
    },
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
    {
        name: 'compare-raycast',
        path: 'compare/raycast',
        title: 'compare/raycast',
        description: 'CPU first-hit and all-hit raycasts per second in every library on the same seeded rays, answers checked against ours, and a GPU batch against three-mesh-bvh/webgpu. Pick with any library.',
        timeoutMs: 240_000,
    },
    {
        name: 'compare-build',
        path: 'compare/build',
        title: 'compare/build',
        description: 'Build and refit times, memory beyond the geometry, and main-thread stalls of worker builds (ours and three-mesh-bvh\'s), plus a GPU H-PLOC build only ours has.',
        timeoutMs: 240_000,
    },
    {
        name: 'compare-tree-quality',
        path: 'compare/tree-quality',
        title: 'compare/tree-quality',
        description: 'SAH cost, depth, nodes and bytes of every tree: tinybvh, H-PLOC and LBVH against three-mesh-bvh\'s CENTER and SAH and bvh-tree.',
        timeoutMs: 180_000,
    },
    {
        name: 'compare-queries',
        path: 'compare/queries',
        title: 'compare/queries',
        description: 'Box, sphere and closest-point queries against three-mesh-bvh and Rapier, with a live probe; the CPU tree uploaded as is for the GPU walk.',
        timeoutMs: 180_000,
    },
    {
        name: 'compare-scene',
        path: 'compare/scene',
        title: 'compare/scene',
        description: 'A scene-level tree over thousands of objects: ours against bvh.js and three-mesh-bvh\'s ObjectBVH for build, updates as everything moves, frustum and box culling, rays; the GPU top-level refit.',
        timeoutMs: 180_000,
    },
    {
        name: 'compare-gpu',
        path: 'compare/gpu',
        title: 'compare/gpu',
        description: 'GPU traversal, ours against three-mesh-bvh/webgpu, closest and any hit, in paired rounds; a GPU-built tree traced by the same kernel.',
        timeoutMs: 240_000,
    },
];

/** Where an example's source is browsed. */
export const SOURCE_TREE = 'https://github.com/pmndrs/bvh/tree/main/examples/';

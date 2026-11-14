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
        name: 'vanilla-raytrace',
        path: 'vanilla/raytrace',
        title: 'Ray-cast view',
        description: 'One ray per pixel through the WGSL walk in the page\'s own compute kernel, shown by normal, material, instance or distance, any-hit shadows; switch tinybvh, H-PLOC and LBVH live and watch build ms, SAH and Mrays/s.',
        timeoutMs: 120_000,
    },
    {
        name: 'vanilla-instances',
        path: 'vanilla/instances',
        title: 'Instances',
        description: 'SceneTrees with thousands of moving instances, a top-level refit in every frame\'s encoder, runtime adds and removals of instances and meshes, GPU H-PLOC first trees swapped for tinybvh\'s, and picking from the ray-cast view.',
        timeoutMs: 120_000,
    },
    {
        name: 'vanilla-builders',
        path: 'vanilla/builders',
        title: 'Builders',
        description: 'tinybvh, GPU H-PLOC and GPU LBVH on the same mesh, side by side: build ms (CPU and GPU timestamps), SAH against tinybvh, depth, nodes, bytes, Mrays/s on the same rays, and agreement of the hits. The builder guide in a page.',
        timeoutMs: 120_000,
    },
    {
        name: 'vanilla-refit',
        path: 'vanilla/refit',
        title: 'Refit',
        description: 'A deforming mesh posed by a compute pass every frame and refit on the GPU with GpuMeshRefit in the scene\'s own buffers (meshRefitPlacement): refit ms, SAH drift against the built tree and against a fresh build of the same pose, "rebuild now" and an automatic rebuild past a drift limit.',
        timeoutMs: 120_000,
    },
    {
        name: 'vanilla-fast-first',
        path: 'vanilla/fast-first',
        title: 'Fast first',
        description: 'ddgi\'s load pattern as a recipe: GPU H-PLOC trees placed for every mesh at once, tinybvh trees from the worker pool swapped in mesh by mesh, a timeline row per mesh, and the walk tracing every frame throughout. Hold and release the swaps, add and remove meshes mid-load, drop a glTF.',
        timeoutMs: 180_000,
    },
    {
        name: 'vanilla-workers-cpu',
        path: 'vanilla/workers-cpu',
        title: 'Workers (CPU)',
        description: 'tinybvh builds for many meshes on the main thread, in one worker and in a pool (createBvhWorkerPool, no cross-origin isolation), each tree placed as it arrives while every mesh spins: time to every tree, the worst frame and the long tasks of each, and a timeline row per mesh.',
        timeoutMs: 180_000,
    },
    {
        name: 'vanilla-workers-gpu',
        path: 'vanilla/workers-gpu',
        title: 'Workers (GPU)',
        description: 'Three ways to run a GPU build on the same meshes: on the render device with no copy, on a worker\'s own device and transferred through the CPU, or everything in a worker with an OffscreenCanvas; against tinybvh\'s worker pool alone, the time to a first usable scene against the time to the best trees.',
        timeoutMs: 180_000,
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
        description: 'MeshBVHHelper boxes by depth, stats per builder (tinybvh, H-PLOC, LBVH), leaf size and layout options, and a heat map of steps per ray from the GPU walk (its counted walk).',
        timeoutMs: 90_000,
    },
    {
        name: 'babylon-picking',
        path: 'babylon/picking',
        title: 'Babylon picking',
        description: 'scene.pick over a big mesh against the CPU raycast of @pmndrs/bvh/babylon, with picks per second for each. CPU picking needs no WebGPU: it runs on Babylon\'s WebGL engine too (?engine=webgl).',
        timeoutMs: 120_000,
    },
    {
        name: 'babylon-gpu-trace',
        path: 'babylon/gpu-trace',
        title: 'Babylon GPU trace',
        description: 'Hard shadows from the WGSL walk in a Babylon ComputeShader on the WebGPUEngine\'s own device, bound to SceneTrees\' buffers with no copy and blended over Babylon\'s render. Babylon is "it works" only: it reaches into engine._device.',
        timeoutMs: 120_000,
    },
    {
        name: 'three-sdf',
        path: 'three/sdf',
        title: 'SDF generation',
        description: 'A signed distance field generated on the GPU from the scene\'s trees (closest point plus a three-ray inside vote per voxel) into a three 3D texture, shown by a TSL raymarch or a slice with contours. Resolution, margin and the inside rule live.',
        timeoutMs: 120_000,
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
    {
        name: 'compare-sdf',
        path: 'compare/sdf',
        title: 'compare/sdf',
        description: 'SDF generation time on the GPU against three-mesh-bvh\'s webgpu_sdfGeneration kernel at several resolutions, in paired rounds, distances and signs checked; the scene\'s GPU-built tree straight into a three texture, shown as a slice.',
        timeoutMs: 240_000,
    },
    // ---- three-mesh-bvh compares and example ports, set 1: skinned, pathtracer; shapecast, distancecast,
    // characterMovement, physics, sculpt, selection.
    {
        name: 'compare-skinned',
        path: 'compare/skinned',
        title: 'compare/skinned',
        description: "Skinned characters kept on their pose every frame: three-mesh-bvh's SkinnedMeshBVH.refit() on the CPU against our GPU pose pass and refit (SceneTrees.setPose), CPU and GPU ms per frame; a plain WebGPU pass three never sees traces the refit trees.",
        timeoutMs: 240_000,
    },
    {
        name: 'compare-pathtracer',
        path: 'compare/pathtracer',
        title: 'compare/pathtracer',
        description: "The same small path tracer on three-mesh-bvh/webgpu's WGSL BVH functions and on our walk (BvhTraceNodes): Mrays/s for the whole tracer, its closest-hit rays and its any-hit shadow rays, in paired rounds, first hits checked; the scene's GPU-built trees traced too.",
        timeoutMs: 300_000,
    },
    {
        name: 'three-shapecast',
        path: 'three/shapecast',
        title: 'Shapecast',
        description: "three-mesh-bvh's shapecast example on our tree: a sphere, a box or another mesh moved over a spinning 80k-triangle knot with intersectsSphere, intersectsBox and intersectsGeometry (tree against tree), red where they touch; the query ms live.",
        timeoutMs: 120_000,
    },
    {
        name: 'three-distancecast',
        path: 'three/distancecast',
        title: 'Distancecast',
        description: "three-mesh-bvh's distancecast example on our tree: closestPointToGeometry joins a cylinder you move to a noise terrain (tree against tree), and a marching-cubes volume of closestPointToPoint shows where the terrain or the cylinder is within a distance.",
        timeoutMs: 120_000,
    },
    {
        name: 'three-characterMovement',
        path: 'three/characterMovement',
        title: 'Character movement',
        description: "three-mesh-bvh's characterMovement example on our tree: a level baked by StaticGeometryGenerator into one collider, a capsule walked, jumped and pushed out of the triangles shapecast hands back (closestPointToSegment). WASD and space.",
        timeoutMs: 120_000,
    },
    {
        name: 'three-physics',
        path: 'three/physics',
        title: 'Physics',
        description: "three-mesh-bvh's physics example on our tree: spheres thrown into an environment baked by StaticGeometryGenerator, each pushed out of the triangles shapecast hands back (nearest boxes first), bouncing off each other, rings on hard contacts. Click to throw.",
        timeoutMs: 120_000,
    },
    {
        name: 'three-sculpt',
        path: 'three/sculpt',
        title: 'Sculpt',
        description: "three-mesh-bvh's sculpt example on our tree: clay, normal and flatten brushes on a welded icosphere, the brush placed by acceleratedRaycast, its vertices found by shapecast (contained boxes taken whole), normals mended and the tree refit after each stroke. Symmetry across x.",
        timeoutMs: 120_000,
    },
    {
        name: 'three-selection',
        path: 'three/selection',
        title: 'Selection',
        description: "three-mesh-bvh's selection example on our tree: lasso or box selection of a 60k-triangle knot by centroid, visible centroid or any overlap, the tree's shapecast narrowing it by each box's screen hull; with or without the tree, timed.",
        timeoutMs: 120_000,
    },
    // ---- end of set 1.
    {
        name: 'node-bake',
        path: 'node/bake',
        title: 'Node bake',
        description: 'A tree or a whole scene baked in Node (pnpm examples:bake, --scene: tinybvh, serializeLayout or serializeTraceScene, a file) loaded here with no build at all: builtBy is the baked file, and the walk traces it. Drop another .bvh, or open ?bake=<url>.',
        timeoutMs: 90_000,
    },
    // three-mesh-bvh example ports (P8): the `@pmndrs/bvh/three` CPU queries on the originals' scenes.
    {
        name: 'three-clippedEdges',
        path: 'three/clippedEdges',
        title: 'Clipped edges',
        description: 'A plane sweeps through a model and tree.shapecast turns every triangle it crosses into an outline segment (intersectsBounds keeps the crossed boxes), capped with the stencil trick; three-mesh-bvh\'s clippedEdges. Switch the BVH off to visit every triangle: the same segments, checked in the smoke run.',
        timeoutMs: 180_000,
    },
    {
        name: 'three-collectTriangles',
        path: 'three/collectTriangles',
        title: 'Collect triangles',
        description: 'A paint brush that is a sphere: tree.shapecast collects every triangle it touches (CONTAINED boxes hand theirs over untested) on hundreds of thousands of triangles; three-mesh-bvh\'s collectTriangles. The smoke run checks the triangles against a pass over every triangle.',
        timeoutMs: 180_000,
    },
    {
        name: 'three-edgeIntersect',
        path: 'three/edgeIntersect',
        title: 'Edge intersect',
        description: 'tree.bvhcast walks two trees against each other and ExtendedTriangle.intersectsTriangle gives the segment where each pair of triangles crosses: two spinning torus knots and the line where they meet; three-mesh-bvh\'s edgeIntersect. The smoke run checks the crossings against every pair of triangles.',
        timeoutMs: 180_000,
    },
    {
        name: 'three-lineIntersection',
        path: 'three/lineIntersection',
        title: 'Line intersection',
        description: "Raycasting a polyline of hundreds of thousands of segments with a pick threshold, three-mesh-bvh's lineIntersection: LineBVH through computeBoundsTree({ type: LineBVH }) and acceleratedRaycast on the Line. Checked against three's Line.raycast in the smoke run.",
        timeoutMs: 240_000,
    },
    {
        name: 'three-triangleIntersect',
        path: 'three/triangleIntersect',
        title: 'Triangle intersect',
        description: 'ExtendedTriangle.intersectsTriangle on its own: two triangles with six draggable corners and the segment where they cross; three-mesh-bvh\'s triangleIntersect. The smoke run checks thousands of random pairs against a second method.',
        timeoutMs: 90_000,
    },
    {
        name: 'three-voxelize',
        path: 'three/voxelize',
        title: 'Voxelize',
        description: 'A grid of voxels over a model: tree.intersectsBox(box, boxToMesh) finds the voxels on the surface, tree.raycastFirst with DoubleSide the ones inside it, filled a few ms per frame; three-mesh-bvh\'s voxelize. The smoke run checks every voxel against three\'s Triangle.intersectsBox and Mesh.raycast.',
        timeoutMs: 240_000,
    },
    {
        name: 'three-batchedMesh',
        path: 'three/batchedMesh',
        title: 'BatchedMesh',
        description: 'computeBatchedBoundsTree and acceleratedRaycast on a BatchedMesh of three geometries, with hundreds of raycasters firing every frame and a switch to three\'s own raycast; three-mesh-bvh\'s batchedMesh. The smoke run checks the nearest hits against three\'s.',
        timeoutMs: 240_000,
    },
    {
        name: 'three-point-cloud',
        path: 'three/point-cloud',
        title: 'Point cloud',
        description: "three-mesh-bvh's pointCloudIntersection, ported: a PointsBVH over a scanned point cloud, the pointer picked every frame through acceleratedRaycast with Raycaster.params.Points.threshold, BVHHelper boxes, and three's own Points.raycast when the tree is off.",
        timeoutMs: 90_000,
    },
];

/** Where an example's source is browsed. */
export const SOURCE_TREE = 'https://github.com/pmndrs/bvh/tree/main/examples/';

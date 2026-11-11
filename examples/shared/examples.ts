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
];

/** Where an example's source is browsed. */
export const SOURCE_TREE = 'https://github.com/pmndrs/bvh/tree/main/examples/';

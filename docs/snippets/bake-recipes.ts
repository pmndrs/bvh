// Code sample for docs/serialization/serialization-and-baking.mdx: baking at build time
// (see getting-started.ts for how they are checked).

// #region bake-plugin
// vite.config.ts: bake at build time; the trees are emitted next to the app's other assets.
import { createTinyBvhBuilder, serializeLayout, type MeshInput } from '@pmndrs/bvh';
import type { Plugin } from 'vite';

/** `meshes` reads the app's models and returns each one's positions and indices, keyed by file name. */
export function bakeTrees(meshes: () => Promise<Record<string, MeshInput>>): Plugin {
    return {
        name: 'bake-trees',
        async generateBundle() {
            const tiny = await createTinyBvhBuilder();
            for (const [name, mesh] of Object.entries(await meshes())) {
                const { layout } = tiny.buildLayout(mesh, { maxLeafPrimitives: 8 });
                this.emitFile({ type: 'asset', fileName: `trees/${name}.bvh`, source: new Uint8Array(serializeLayout(layout)) });
            }
        },
    };
}
// #endregion

// Code sample for docs/serialization/serialization-and-baking.mdx: the version check
// (see getting-started.ts for how they are checked).

// #region version-check
import { createTinyBvhBuilder, deserializeLayout, GPU_LAYOUT_VERSION, SERIALIZED_FORMAT_VERSION, type GpuLayoutMeshBvh, type MeshInput } from '@pmndrs/bvh';

/** The baked tree at `url`, or a fresh build of `mesh` when the file is from another version of the package. */
export async function loadOrRebuild(url: string, mesh: MeshInput): Promise<GpuLayoutMeshBvh> {
    const bytes = await (await fetch(url)).arrayBuffer();
    // The header's words 1 and 2 (little-endian u32): the byte format and the GPU layout the file was written with.
    const header = new DataView(bytes);
    if (header.getUint32(4, true) === SERIALIZED_FORMAT_VERSION && header.getUint32(8, true) === GPU_LAYOUT_VERSION) {
        return deserializeLayout(bytes);
    }
    // A stale bake (the package moved on since the file was written): build it again, or fail the deploy.
    const tiny = await createTinyBvhBuilder();
    return tiny.buildLayout(mesh, { maxLeafPrimitives: 8 }).layout;
}
// #endregion

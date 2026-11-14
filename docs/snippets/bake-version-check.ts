// Code sample for docs/serialization/serialization-and-baking.mdx: the version check
// (see getting-started.ts for how they are checked).

// #region version-check
import { createTinyBvhBuilder, deserializeLayout, readSerializedHeader, type GpuLayoutMeshBvh, type MeshInput } from '@pmndrs/bvh';

/** The baked tree at `url`, or a fresh build of `mesh` when the file is from another version of the package. */
export async function loadOrRebuild(url: string, mesh: MeshInput): Promise<GpuLayoutMeshBvh> {
    const bytes = await (await fetch(url)).arrayBuffer();
    // The 64-byte header alone: the kind of file, the versions it was written with, and whether this package reads it.
    const header = readSerializedHeader(bytes);
    if (header.kind === 'tree' && header.current) return deserializeLayout(bytes, { geometry: mesh });
    // A stale bake (header.reason says why): build it again, or fail the deploy.
    console.warn(`${url}: ${header.reason ?? 'a scene file, not a tree'}; building instead`);
    const tiny = await createTinyBvhBuilder();
    return tiny.buildLayout(mesh, { maxLeafPrimitives: 8 }).layout;
}
// #endregion

// #region ci-check
import { readdir, readFile } from 'node:fs/promises';
import { isSerializedCurrent } from '@pmndrs/bvh';

// #endregion

export async function checkBakes(folder: string): Promise<void> {
    // #region ci-check
    // In CI: every committed bake must be one this package loads, or the deploy ships files it refuses.
    const stale: string[] = [];
    for (const name of (await readdir(folder)).filter((file) => file.endsWith('.bvh'))) {
        if (!isSerializedCurrent(await readFile(`${folder}/${name}`))) stale.push(name);
    }
    if (stale.length) throw new Error(`bake these again: ${stale.join(', ')}`);
    // #endregion
}

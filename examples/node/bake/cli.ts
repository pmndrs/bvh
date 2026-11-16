/**
 * node/bake: bake a model's BVH in Node and write it to a file.
 *
 *     pnpm examples:bake [model.glb|model.gltf] [--out file.bvh] [--leaf 8] [--indexed] [--quantized | --f32] [--scene [--movable] [--no-merge]]
 *
 * With no model, the default torus knot is baked to examples/node/bake/fixtures/torus-knot.bvh,
 * the file the example page loads. `--scene` bakes a whole scene instead (every mesh's tree, the
 * top level, instances and materials: `serializeTraceScene`); with no model, the default scene goes
 * to examples/node/bake/fixtures/knots.scene.bvh. `--scene --movable` writes a movable scene instead
 * (mesh trees, merged groups' trees and instances: `serializeMovableScene`), which the page loads into
 * `SceneTrees` so its instances can move. The page (index.html) loads every kind with `?bake=<url>`
 * or a dropped file, and builds nothing.
 */

import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { readSerializedHeader } from '@pmndrs/bvh';
import { bakeMesh, defaultOutputName, readModelMesh, readModelScene } from './bake.ts';
import { bakeMovableScene, bakeScene, defaultMesh, defaultScene } from './model.ts';

const USAGE = `Usage: pnpm examples:bake [model.glb|model.gltf] [options]

  --out <file>    Where to write the bake (default: <model>.bvh, or <model>.scene.bvh with --scene, in the current
                  folder; the default model goes to examples/node/bake/fixtures/torus-knot.bvh, the default scene to
                  examples/node/bake/fixtures/knots.scene.bvh).
  --leaf <n>      Largest leaf, 1 to 8 triangles (default 8).
  --indexed       Bake 16-byte indexed triangles. A tree file is then loaded next to the same geometry; a scene file
                  carries its positions.
  --quantized     Bake 32-byte quantized nodes (the default for a scene).
  --f32           Bake 64-byte f32 nodes (the default for a tree).
  --scene         Bake the whole scene: one tree per glTF mesh, the nodes as instances, the top level, materials.
                  Every instance is static, so meshes used once are merged into one world-space mesh.
  --movable       With --scene: a movable scene file, loaded into SceneTrees so the instances can move (the top level
                  is built at load). Instances of a mesh used once are static and merge; instances of a mesh used
                  more than once can move. Carries the meshes' geometry. Default name <model>.movable.bvh.
  --no-merge      With --scene: keep every instance an instance (no merged static mesh; with --movable, every
                  instance can move).
  --help          This text.`;

const { values, positionals } = parseArgs({
    options: {
        out: { type: 'string' },
        leaf: { type: 'string' },
        indexed: { type: 'boolean' },
        quantized: { type: 'boolean' },
        f32: { type: 'boolean' },
        scene: { type: 'boolean' },
        'no-merge': { type: 'boolean' },
        movable: { type: 'boolean' },
        help: { type: 'boolean' },
    },
    allowPositionals: true,
});
if (values.help) {
    console.log(USAGE);
    process.exit(0);
}
if (positionals.length > 1) {
    console.error(`one model at a time (got ${positionals.length})\n\n${USAGE}`);
    process.exit(2);
}
if (values.quantized && values.f32) {
    console.error(`--quantized or --f32, not both\n\n${USAGE}`);
    process.exit(2);
}
if (values['no-merge'] && !values.scene) {
    console.error(`--no-merge is for --scene\n\n${USAGE}`);
    process.exit(2);
}
if (values.movable && !values.scene) {
    console.error(`--movable is for --scene\n\n${USAGE}`);
    process.exit(2);
}

const model = positionals[0] ?? null;
const scene = values.scene ?? false;
const movable = values.movable ?? false;
const out = resolve(values.out ?? (model ? defaultOutputName(model, scene, movable) : resolve(import.meta.dirname, 'fixtures', defaultOutputName(null, scene, movable))));
const maxLeafPrimitives = values.leaf === undefined ? undefined : Number(values.leaf);
const triangleLeaves = values.indexed ? 'indexed' : 'records';
const name = model ?? (scene ? 'default scene' : 'default torus knot');

let bytes: Uint8Array;
if (scene && movable) {
    const source = model ? await readModelScene(model) : defaultScene();
    const baked = await bakeMovableScene(source, { maxLeafPrimitives, triangleLeaves, nodeFormat: values.f32 ? 'f32' : 'quantized', mergeStaticInstances: !values['no-merge'] });
    bytes = baked.bytes;
    const merged = baked.data.instances.filter((instance) => instance.static).length;
    const trees = baked.data.meshes.filter((mesh) => mesh.tree).length;
    const groups = baked.data.mergedGroups.length;
    console.log(
        `${name}: ${source.meshes.length} meshes (${trees} tree${trees === 1 ? '' : 's'}), ${source.instances.length} instances (${source.instances.length - merged} can move, ${merged} merged into ${groups} static group${groups === 1 ? '' : 's'}), ` +
            `built by tinybvh in ${baked.buildMs.toFixed(1)} ms; the top level is built where it loads`,
    );
} else if (scene) {
    const source = model ? await readModelScene(model) : defaultScene();
    const baked = await bakeScene(source, { maxLeafPrimitives, triangleLeaves, nodeFormat: values.f32 ? 'f32' : 'quantized', mergeStaticInstances: !values['no-merge'] });
    bytes = baked.bytes;
    const merged = baked.data.mergedGroups.reduce((sum, group) => sum + group.sourceInstances.length, 0);
    console.log(
        `${name}: ${source.meshes.length} meshes, ${source.instances.length} instances (${merged} merged into ${baked.data.mergedGroups.length} static group${baked.data.mergedGroups.length === 1 ? '' : 's'}), ` +
            `${source.triangles.toLocaleString('en-US')} triangles placed, built by tinybvh and assembled in ${baked.buildMs.toFixed(1)} ms`,
    );
} else {
    const mesh = model ? await readModelMesh(model) : defaultMesh();
    const baked = await bakeMesh(mesh, { maxLeafPrimitives, triangleLeaves, nodeFormat: values.quantized ? 'quantized' : 'f32' });
    bytes = baked.bytes;
    console.log(`${name}: ${baked.triangles.toLocaleString('en-US')} triangles, ${baked.layout.nodeCount.toLocaleString('en-US')} nodes, built by tinybvh in ${baked.buildMs.toFixed(1)} ms`);
}
await mkdir(dirname(out), { recursive: true });
await writeFile(out, bytes);

const header = readSerializedHeader(bytes);
const formats = `${header.triangleLeaves} triangles, ${header.nodeFormat} nodes`;
console.log(`wrote ${out} (${bytes.byteLength.toLocaleString('en-US')} bytes, a ${header.kind}; ${formats}; byte format ${header.formatVersion}, GPU layout version ${header.layoutVersion})`);

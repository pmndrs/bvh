/**
 * node/bake: bake a model's BVH in Node and write it to a file.
 *
 *     pnpm examples:bake [model.glb|model.gltf] [--out file.bvh] [--leaf 8] [--indexed] [--quantized]
 *
 * With no model, the default torus knot is baked to examples/node/bake/fixtures/torus-knot.bvh,
 * the file the example page loads. The page (index.html) loads any bake with
 * `?bake=<url>` or a dropped file, and builds nothing.
 */

import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { bakeMesh, defaultOutputName, readBakedHeader, readModelMesh } from './bake.ts';
import { defaultMesh } from './model.ts';

const USAGE = `Usage: pnpm examples:bake [model.glb|model.gltf] [options]

  --out <file>    Where to write the baked tree (default: <model>.bvh in the current folder; the default model goes to
                  examples/node/bake/fixtures/torus-knot.bvh).
  --leaf <n>      Largest leaf, 1 to 8 triangles (default 8).
  --indexed       Bake 16-byte indexed triangles (load the file next to the same geometry).
  --quantized     Bake 32-byte quantized nodes.
  --help          This text.`;

const { values, positionals } = parseArgs({
    options: { out: { type: 'string' }, leaf: { type: 'string' }, indexed: { type: 'boolean' }, quantized: { type: 'boolean' }, help: { type: 'boolean' } },
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

const model = positionals[0] ?? null;
const mesh = model ? await readModelMesh(model) : defaultMesh();
const out = resolve(values.out ?? (model ? defaultOutputName(model) : resolve(import.meta.dirname, 'fixtures', defaultOutputName(null))));
const baked = await bakeMesh(mesh, {
    maxLeafPrimitives: values.leaf === undefined ? undefined : Number(values.leaf),
    triangleLeaves: values.indexed ? 'indexed' : 'records',
    nodeFormat: values.quantized ? 'quantized' : 'f32',
});
await mkdir(dirname(out), { recursive: true });
await writeFile(out, baked.bytes);

const header = readBakedHeader(baked.bytes);
console.log(`${model ?? 'default torus knot'}: ${baked.triangles.toLocaleString('en-US')} triangles, ${header.nodeCount.toLocaleString('en-US')} nodes, built by tinybvh in ${baked.buildMs.toFixed(1)} ms`);
console.log(`wrote ${out} (${baked.bytes.byteLength.toLocaleString('en-US')} bytes; byte format ${header.formatVersion}, GPU layout version ${header.layoutVersion})`);

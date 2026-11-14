// Fails the build when dist/ is not what the package promises: every `exports`
// target exists, no relative `.ts` import is left, and the files that embed
// third-party code still carry their licence text (a build step can drop
// comments; the /*! */ headers must survive).
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const root = resolve(import.meta.dirname, '..');
const dist = resolve(root, 'dist');
const problems = [];

const manifest = JSON.parse(readFileSync(resolve(root, 'package.json'), 'utf8'));
for (const [subpath, conditions] of Object.entries(manifest.exports)) {
    for (const target of Object.values(conditions)) {
        if (!existsSync(resolve(root, target))) problems.push(`exports["${subpath}"] points at missing ${target}`);
    }
}
for (const file of ['README.md', 'THIRD_PARTY.md', 'licenses/tinybvh.txt', 'licenses/emscripten-runtime.txt', 'licenses/hiprt.txt']) {
    if (!existsSync(resolve(root, file))) problems.push(`missing ${file}`);
}

/** Built file -> text its licence comment must still contain. */
const requiredLicenseText = {
    'tinybvh-wasm.js': ['Copyright (c) 2024 Jacco Bikker', 'Copyright (c) 2010-2014 Emscripten authors', 'Rich Felker'],
    'gpu-lbvh-wgsl.js': ['hiprt', 'Permission is hereby granted, free of charge'],
    'gpu-hploc-wgsl.js': ['tinybvh', 'Permission is hereby granted, free of charge'],
    'wide-nodes.js': ['tinybvh', 'Permission is hereby granted, free of charge'],
    'trace/wgsl-wide.js': ['tinybvh', 'Permission is hereby granted, free of charge'],
    'gpu-wide-collapse-wgsl.js': ['tinybvh', 'Permission is hereby granted, free of charge'],
    'gpu-wide-refit.js': ['tinybvh', 'Permission is hereby granted, free of charge'],
};
for (const [name, lines] of Object.entries(requiredLicenseText)) {
    const file = resolve(dist, name);
    if (!existsSync(file)) {
        problems.push(`missing dist/${name}`);
        continue;
    }
    const text = readFileSync(file, 'utf8');
    if (!text.startsWith('/*!')) problems.push(`dist/${name} lost its leading /*! licence comment`);
    for (const line of lines) if (!text.includes(line)) problems.push(`dist/${name} lacks the licence text "${line}"`);
}

// Any built file embedding WebAssembly (base64 of "\0asm", version 1) must carry the wasm licences.
const embeddedWasmMarker = 'AGFzbQEAAAA';
const workerEntries = new Set();
for (const entry of existsSync(dist) ? readdirSync(dist, { recursive: true, withFileTypes: true }) : []) {
    if (!entry.isFile()) continue;
    const file = resolve(entry.parentPath, entry.name);
    const text = readFileSync(file, 'utf8');
    if (entry.name.endsWith('.js') && text.includes(embeddedWasmMarker)) {
        for (const line of requiredLicenseText['tinybvh-wasm.js']) {
            if (!text.includes(line)) problems.push(`${file} embeds WebAssembly but lacks the licence text "${line}"`);
        }
    }
    if (/\.(js|d\.ts)$/.test(entry.name) && /(?:from\s*|import\s*\(\s*)["']\.{1,2}\/[^"']*\.ts["']/.test(text)) {
        problems.push(`${file} still imports a relative .ts file`);
    }
    if (/\.test\.(js|d\.ts)$/.test(entry.name)) problems.push(`${file} is a test file in dist`);
    // Worker entries: `new URL('./x.js', import.meta.url)` must name a built file, never a .ts source.
    if (entry.name.endsWith('.js')) {
        for (const match of text.matchAll(/new URL\(\s*["'](\.{1,2}\/[^"']*)["']\s*,\s*import\.meta\.url\s*\)/g)) {
            const target = resolve(entry.parentPath, match[1]);
            if (match[1].endsWith('.ts')) problems.push(`${file} starts a worker from the source file ${match[1]}`);
            else if (!existsSync(target)) problems.push(`${file} starts a worker from ${match[1]}, which is not in dist`);
            else workerEntries.add(target);
        }
    }
}

// The /worker subpath must start its workers from built files (the mesh tree worker at least).
if (manifest.exports['./worker'] && ![...workerEntries].some((file) => file.endsWith('mesh-bvh-worker.js'))) {
    problems.push('no built file starts the mesh tree worker (dist/trace/mesh-bvh-worker.js)');
}

// The /three subpath must import in Node with three installed (its peer), and say what it exports.
if (manifest.exports['./three'] && problems.length === 0) {
    try {
        const three = await import(pathToFileURL(resolve(root, manifest.exports['./three'].import)).href);
        for (const name of ['bvhSceneFromObject3D', 'SceneTreesObject3DSync']) {
            if (typeof three[name] !== 'function') problems.push(`dist/three does not export ${name}`);
        }
    } catch (error) {
        problems.push(`dist/three does not import in Node: ${error instanceof Error ? error.message : String(error)}`);
    }
}

if (problems.length) {
    console.error(problems.join('\n'));
    process.exit(1);
}
console.log('Verified dist: exports exist, licence text intact in tinybvh-wasm, gpu-lbvh-wgsl, gpu-hploc-wgsl, wide-nodes, trace/wgsl-wide, gpu-wide-collapse-wgsl and gpu-wide-refit, no .ts imports, no tests, worker entries built (' + workerEntries.size + '), /three imports in Node.');

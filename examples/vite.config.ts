import { existsSync, readFileSync, statSync } from 'node:fs';
import { extname, resolve, sep } from 'node:path';
import { defineConfig, type Plugin } from 'vite';
import { EXAMPLES } from './shared/examples.ts';

// The examples site: an index page and one folder per example
// (`three/raycast/index.html` + `main.ts`, ...). `pnpm examples:dev` serves it,
// `pnpm examples:build` builds it into examples/dist. The library is read from
// source through the package's public entry points only (the aliases below), so
// an example imports exactly what an app would.
const root = import.meta.dirname;
const repoRoot = resolve(root, '..');

/**
 * The bundled scene assets (Sponza, Bistro, ...) are large and live outside
 * this repository. The dev server serves `BVH_ASSETS_DIR` (the folder that
 * holds `downloads/<asset>/...`, the same variable the GPU runner reads) at
 * `/assets/`. A built site has no such folder: pass `?assets=<url>` to point the
 * loader elsewhere.
 */
const assetsDir = resolve(repoRoot, process.env.BVH_ASSETS_DIR ?? '../webgpu-ddgi/assets');

const CONTENT_TYPES: Readonly<Record<string, string>> = {
    '.glb': 'model/gltf-binary',
    '.gltf': 'model/gltf+json',
    '.bin': 'application/octet-stream',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.ktx2': 'image/ktx2',
    '.webp': 'image/webp',
};

function sceneAssets(): Plugin {
    return {
        name: 'bvh-scene-assets',
        configureServer(server) {
            server.middlewares.use('/assets/', (request, response) => {
                const urlPath = decodeURIComponent((request.url ?? '/').split('?')[0]);
                const path = resolve(assetsDir, `.${urlPath}`);
                if (!path.startsWith(assetsDir + sep) || !existsSync(path) || !statSync(path).isFile()) {
                    response.statusCode = 404;
                    response.end(`Not found in BVH_ASSETS_DIR (${assetsDir})`);
                    return;
                }
                response.setHeader('Content-Type', CONTENT_TYPES[extname(path).toLowerCase()] ?? 'application/octet-stream');
                response.end(readFileSync(path));
            });
            // Chrome asks for /favicon.ico; a 404 is a console error, which the smoke runner counts.
            server.middlewares.use('/favicon.ico', (_request, response) => {
                response.statusCode = 204;
                response.end();
            });
        },
    };
}

export default defineConfig({
    root,
    // A project site under /<repo>/ sets EXAMPLES_BASE; links between pages are relative.
    base: process.env.EXAMPLES_BASE ?? '/',
    plugins: [sceneAssets()],
    resolve: {
        // Exact-match aliases per public subpath, so nothing else of src/ is reachable by package name.
        alias: [
            { find: /^@pmndrs\/bvh$/, replacement: resolve(repoRoot, 'src/index.ts') },
            { find: /^@pmndrs\/bvh\/three$/, replacement: resolve(repoRoot, 'src/three/index.ts') },
            { find: /^@pmndrs\/bvh\/babylon$/, replacement: resolve(repoRoot, 'src/babylon/index.ts') },
            { find: /^@pmndrs\/bvh\/worker$/, replacement: resolve(repoRoot, 'src/worker/index.ts') },
        ],
    },
    cacheDir: resolve(repoRoot, 'node_modules/.vite-examples'),
    // bvh-tree (compare/) is CommonJS: pre-bundled so the dev server can import it (the smoke run skips discovery).
    optimizeDeps: { include: ['bvh-tree'] },
    // The worker pool's entry is an ES module (`new Worker(url, { type: 'module' })`).
    worker: { format: 'es' },
    build: {
        target: 'esnext',
        outDir: resolve(root, 'dist'),
        emptyOutDir: true,
        // Rapier (compare/ only) inlines its WASM: a 4.4 MB chunk of its own.
        chunkSizeWarningLimit: 5000,
        rollupOptions: {
            input: {
                index: resolve(root, 'index.html'),
                ...Object.fromEntries(EXAMPLES.map((example) => [example.name, resolve(root, example.path, 'index.html')])),
            },
            output: {
                manualChunks(id) {
                    if (id.includes('/node_modules/three/')) return 'three';
                    if (id.includes('/node_modules/@dimforge/')) return 'rapier';
                    if (id.includes('/node_modules/@babylonjs/')) return 'babylon';
                    if (id.includes('/src/three/')) return 'bvh-three';
                    if (id.includes('/src/babylon/')) return 'bvh-babylon';
                    if (id.startsWith(resolve(repoRoot, 'src') + sep)) return 'bvh';
                    return undefined;
                },
            },
        },
    },
    server: {
        port: 5310,
        strictPort: false,
        open: false,
        fs: { allow: [repoRoot] },
    },
});

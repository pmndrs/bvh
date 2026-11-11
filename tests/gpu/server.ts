/**
 * Serves the test pages with vite's dev server, so a page can import
 * TypeScript (including the shared client helper) with no build step.
 */

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { extname, join, relative, resolve, sep } from 'node:path';
import type { ServerResponse } from 'node:http';
import { createServer, type Connect, type ViteDevServer } from 'vite';
import { ASSETS_BASE, ASSETS_LIST, MANIFESTS_BASE } from './contract.ts';

/** The folder served as the site root: it holds client.ts, preflight.html and pages/. */
export const SITE_ROOT = import.meta.dirname;

export interface TestSiteServer {
    /** For example http://127.0.0.1:5290 */
    origin: string;
    close(): Promise<void>;
}

/**
 * Where the scene models and manifests are (the benches' Sponza, Bistro and furnished
 * room). They are large and live outside this repository: set BVH_ASSETS_DIR to the folder
 * that holds `downloads/<asset>/...` and BVH_MANIFESTS_DIR to the folder of scene manifest
 * JSON files. Defaults are the sibling webgpu-ddgi checkout. See README.md.
 */
export const REPOSITORY_ROOT = resolve(SITE_ROOT, '..', '..');
export const ASSETS_DIR = resolve(REPOSITORY_ROOT, process.env.BVH_ASSETS_DIR ?? '../webgpu-ddgi/assets');
export const MANIFESTS_DIR = resolve(REPOSITORY_ROOT, process.env.BVH_MANIFESTS_DIR ?? '../webgpu-ddgi/scenes/manifests');

const CONTENT_TYPES: Readonly<Record<string, string>> = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json',
    '.wasm': 'application/wasm',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.svg': 'image/svg+xml',
    '.glb': 'model/gltf-binary',
    '.gltf': 'model/gltf+json',
    '.bin': 'application/octet-stream',
};

/** Every file under `directory` (paths relative to it, with `/`); empty when the directory doesn't exist. */
function listFiles(directory: string): string[] {
    if (!existsSync(directory)) return [];
    const files: string[] = [];
    const walk = (folder: string): void => {
        for (const entry of readdirSync(folder, { withFileTypes: true })) {
            const path = join(folder, entry.name);
            if (entry.isDirectory()) walk(path);
            else files.push(relative(directory, path).split(sep).join('/'));
        }
    };
    walk(directory);
    return files;
}

/** Serves the files of `directory` as they are; 404 for anything missing or outside it. */
function serveStatic(directory: string): Connect.NextHandleFunction {
    return (request, response: ServerResponse) => {
        const urlPath = decodeURIComponent((request.url ?? '/').split('?')[0]);
        const path = resolve(directory, `.${urlPath}`);
        if (!path.startsWith(directory + sep) || !existsSync(path) || !statSync(path).isFile()) {
            response.statusCode = 404;
            response.end('Not found');
            return;
        }
        response.setHeader('Content-Type', CONTENT_TYPES[extname(path)] ?? 'application/octet-stream');
        response.end(readFileSync(path));
    };
}

export async function startTestSite(): Promise<TestSiteServer> {
    const server: ViteDevServer = await createServer({
        root: SITE_ROOT,
        // Do not pick up a vite config from the repo: the test site is self-contained.
        configFile: false,
        logLevel: 'warn',
        appType: 'mpa',
        // The pages have no npm dependencies; skip the dependency pre-bundling scan.
        optimizeDeps: { noDiscovery: true, include: [] },
        server: {
            host: '127.0.0.1',
            port: 5290,
            // Take the next free port if a dev server is already running.
            strictPort: false,
            // Pages are reloaded by the runner; hot reload would only add noise.
            hmr: false,
            watch: null,
        },
        plugins: [
            {
                // The scene models and manifests, from outside the repository (ASSETS_DIR, MANIFESTS_DIR).
                name: 'scene-assets',
                configureServer(devServer) {
                    devServer.middlewares.use(ASSETS_LIST, (_request, response) => {
                        response.setHeader('Content-Type', 'application/json');
                        response.end(JSON.stringify({ assets: listFiles(ASSETS_DIR), manifests: listFiles(MANIFESTS_DIR) }));
                    });
                    devServer.middlewares.use(ASSETS_BASE, serveStatic(ASSETS_DIR));
                    devServer.middlewares.use(MANIFESTS_BASE, serveStatic(MANIFESTS_DIR));
                },
            },
            {
                // Chrome asks for /favicon.ico on every page and logs a 404 as a
                // console error, which the runner would count as a failure.
                name: 'no-favicon-404',
                configureServer(devServer) {
                    devServer.middlewares.use('/favicon.ico', (_request, response) => {
                        response.statusCode = 204;
                        response.end();
                    });
                },
            },
        ],
    });
    await server.listen();
    const origin = server.resolvedUrls?.local[0]?.replace(/\/$/, '');
    if (!origin) {
        await server.close();
        throw new Error('The vite test server started but reported no local address.');
    }
    return { origin, close: () => server.close() };
}

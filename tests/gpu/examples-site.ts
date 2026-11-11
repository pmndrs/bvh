/**
 * `--examples`: the runner smoke-runs the examples site (examples/) instead of
 * the test pages. The site is served by its own vite config
 * (examples/vite.config.ts, with its aliases and the BVH_ASSETS_DIR mapping),
 * plus the runner's preflight page. Each example reports through the same
 * contract as a test page (examples/shared/smoke.ts), so the runner judges it
 * the same way: a failed or missing report, any browser error or WebGPU
 * validation warning fails it.
 */

import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { createServer } from 'vite';
import { EXAMPLES } from '../../examples/shared/examples.ts';
import type { TestPage } from './pages.ts';
import { REPOSITORY_ROOT, SITE_ROOT, type TestSiteServer } from './server.ts';

/** The example pages as runner pages (`--filter` matches their names). */
export function examplePages(): TestPage[] {
    return EXAMPLES.map((example) => ({
        name: example.name,
        urlPath: `/${example.path}/index.html`,
        description: example.title,
        expectFailure: false,
        expectMessage: null,
        timeoutMs: example.timeoutMs,
        manual: false,
    }));
}

export async function startExamplesSite(): Promise<TestSiteServer> {
    const preflight = readFileSync(join(SITE_ROOT, 'preflight.html'));
    const server = await createServer({
        configFile: resolve(REPOSITORY_ROOT, 'examples/vite.config.ts'),
        logLevel: 'warn',
        // three is plain ES modules; skip the dependency scan, which a short run would cancel midway.
        optimizeDeps: { noDiscovery: true, include: [] },
        server: { host: '127.0.0.1', port: 5291, strictPort: false, hmr: false, watch: null },
        plugins: [
            {
                name: 'runner-preflight',
                configureServer(devServer) {
                    devServer.middlewares.use('/preflight.html', (_request, response) => {
                        response.setHeader('Content-Type', 'text/html; charset=utf-8');
                        response.end(preflight);
                    });
                },
            },
        ],
    });
    await server.listen();
    const origin = server.resolvedUrls?.local[0]?.replace(/\/$/, '');
    if (!origin) {
        await server.close();
        throw new Error('The examples site started but reported no local address.');
    }
    return { origin, close: () => server.close() };
}

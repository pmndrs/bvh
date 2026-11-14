import { resolve } from 'node:path';
import { configDefaults, defineConfig } from 'vitest/config';

const root = import.meta.dirname;

export default defineConfig({
    resolve: {
        // The docs' code samples (docs/snippets) import the package by its public names, as an app would.
        alias: [
            { find: /^@pmndrs\/bvh$/, replacement: resolve(root, 'src/index.ts') },
            { find: /^@pmndrs\/bvh\/three$/, replacement: resolve(root, 'src/three/index.ts') },
            { find: /^@pmndrs\/bvh\/babylon$/, replacement: resolve(root, 'src/babylon/index.ts') },
            { find: /^@pmndrs\/bvh\/worker$/, replacement: resolve(root, 'src/worker/index.ts') },
        ],
    },
    test: {
        environment: 'node',
        // The CPU-walk and scene tests take 1-3 s alone. Under heavy machine load (GPU pages,
        // other builds) the 5 s default timed them out, a flake rather than a failure.
        testTimeout: 30_000,
        include: ['src/**/*.test.ts', 'tests/**/*.test.ts', 'docs/snippets/**/*.test.ts'],
        exclude: [...configDefaults.exclude, '**/dist/**', 'tools/emsdk/**'],
    },
});

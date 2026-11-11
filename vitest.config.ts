import { configDefaults, defineConfig } from 'vitest/config';

export default defineConfig({
    test: {
        environment: 'node',
        // The CPU-walk and scene tests take 1-3 s alone. Under heavy machine load (GPU pages,
        // other builds) the 5 s default timed them out, a flake rather than a failure.
        testTimeout: 30_000,
        include: ['src/**/*.test.ts', 'tests/**/*.test.ts'],
        exclude: [...configDefaults.exclude, '**/dist/**', 'tools/emsdk/**'],
    },
});

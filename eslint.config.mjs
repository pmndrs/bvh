import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import globals from 'globals';

const ENGINE_MESSAGE =
    '@pmndrs/bvh must stay engine-free: no three, babylon, pixi or ddgi code in src/. Engines belong in tests/, tools/ or an adapter package.';

export default tseslint.config(
    { ignores: ['**/dist/', 'node_modules/', '.claude/', 'native/build/', 'tools/emsdk/', 'bench/results/'] },
    js.configs.recommended,
    ...tseslint.configs.recommended,
    {
        languageOptions: { globals: { ...globals.node, ...globals.browser } },
        rules: {
            '@typescript-eslint/no-unused-vars': ['warn', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
            '@typescript-eslint/no-explicit-any': 'error',
            '@typescript-eslint/consistent-type-imports': ['warn', { prefer: 'type-imports', fixStyle: 'inline-type-imports' }],
        },
    },
    {
        // The library: engine-free, and independent of the GI code it was split from.
        files: ['src/**/*.ts'],
        ignores: ['src/**/*.test.ts'],
        rules: {
            'no-restricted-imports': ['error', {
                patterns: [{
                    group: [
                        'three',
                        'three/*',
                        'three-mesh-bvh',
                        'three-mesh-bvh/*',
                        '@babylonjs/*',
                        'pixi.js',
                        'pixi.js/*',
                        '@pmndrs/gi',
                        '@pmndrs/gi/*',
                        '@pmndrs/ddgi-core',
                        '@pmndrs/ddgi-core/*',
                        '@pmndrs/ddgi-three',
                        '@pmndrs/ddgi-three/*',
                        '**/ddgi-core/**',
                        '**/ddgi-three/**',
                        '**/ddgi/src/**',
                    ],
                    message: ENGINE_MESSAGE,
                }],
            }],
            // no-restricted-imports only sees static imports, so catch import('three') too.
            'no-restricted-syntax': ['error', {
                selector: "ImportExpression[source.value=/^(three|three-mesh-bvh|@babylonjs\\/|pixi\\.js|@pmndrs\\/gi(\\/|$))|ddgi-(core|three)/]",
                message: ENGINE_MESSAGE,
            }],
        },
    },
    {
        // Tests, GPU pages and tools may use three (as the reference builder and loader), the DOM and WebGPU.
        files: ['tests/**/*.ts', 'tools/**/*.ts', 'bench/**/*.ts', 'native/**/*.ts'],
        rules: { 'no-restricted-imports': 'off', 'no-restricted-syntax': 'off' },
    },
);

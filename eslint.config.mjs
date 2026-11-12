import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import globals from 'globals';

const ENGINE_MESSAGE =
    '@pmndrs/bvh must stay engine-free: no three, babylon, pixi or ddgi code in src/. three belongs in src/three/ (the @pmndrs/bvh/three subpath) only; other engines in tests/, tools/ or an adapter.';

/** Imports no file in src/ may make; src/three/ may import three (below). */
const forbidden = (allowThree) => [
    // The core may not reach into src/three/ either (relative '../three/...').
    ...(allowThree ? [] : ['three', 'three/*', '**/three/**']),
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
];
const engineRules = (allowThree) => ({
    'no-restricted-imports': ['error', { patterns: [{ group: forbidden(allowThree), message: ENGINE_MESSAGE }] }],
    // no-restricted-imports only sees static imports, so catch import('three') too.
    'no-restricted-syntax': ['error', {
        selector: allowThree
            ? "ImportExpression[source.value=/^(three-mesh-bvh|@babylonjs\\/|pixi\\.js|@pmndrs\\/gi(\\/|$))|ddgi-(core|three)/]"
            : "ImportExpression[source.value=/^(three|three-mesh-bvh|@babylonjs\\/|pixi\\.js|@pmndrs\\/gi(\\/|$))|ddgi-(core|three)/]",
        message: ENGINE_MESSAGE,
    }],
});

export default tseslint.config(
    { ignores: ['**/dist/', 'examples/dist/', 'api-docs/', 'docs-site/', 'node_modules/', '.claude/', 'native/build/', 'tools/emsdk/', 'bench/results/'] },
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
        ignores: ['src/**/*.test.ts', 'src/three/**'],
        rules: engineRules(false),
    },
    {
        // @pmndrs/bvh/three: the one place three (a peer dependency) may be imported.
        files: ['src/three/**/*.ts'],
        ignores: ['src/three/**/*.test.ts'],
        rules: engineRules(true),
    },
    {
        // Tests, GPU pages and tools may use three (as the reference builder and loader), the DOM and WebGPU.
        files: ['tests/**/*.ts', 'tools/**/*.ts', 'bench/**/*.ts', 'native/**/*.ts'],
        rules: { 'no-restricted-imports': 'off', 'no-restricted-syntax': 'off' },
    },
    {
        // The examples are apps: three is allowed, and the library only through its public entry points
        // (@pmndrs/bvh, /three, /worker, aliased to src/ by examples/vite.config.ts), never by a path into src/.
        files: ['examples/**/*.ts', 'docs/snippets/**/*.ts'],
        rules: {
            'no-restricted-imports': ['error', { patterns: [{ group: ['**/src/**', '../../src/*', '../../../src/*'], message: 'Examples use the package by its public names: @pmndrs/bvh, @pmndrs/bvh/three, @pmndrs/bvh/worker.' }] }],
            'no-restricted-syntax': 'off',
        },
    },
);

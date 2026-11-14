import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import globals from 'globals';

const ENGINE_MESSAGE =
    '@pmndrs/bvh must stay engine-free: no three, babylon, pixi or ddgi code in src/. three belongs in src/three/ (the @pmndrs/bvh/three subpath) and Babylon in src/babylon/ (@pmndrs/bvh/babylon) only; other engines in tests/, tools/ or an adapter.';

/**
 * Imports no file in src/ may make. `engine` is the one engine a folder may import: 'three' (src/three/), 'babylon' (src/babylon/) or null (the core).
 * The core may not reach into either folder (relative '../three/...'), and an engine's folder takes no other engine.
 */
const forbidden = (engine) => [
    ...(engine === 'three' ? [] : ['three', 'three/*', '**/three/**']),
    ...(engine === 'babylon' ? [] : ['@babylonjs/*', '**/babylon/**']),
    'three-mesh-bvh',
    'three-mesh-bvh/*',
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
const dynamicImportSelector = (engine) =>
    `ImportExpression[source.value=/^(${engine === 'three' ? '' : 'three|'}three-mesh-bvh|${engine === 'babylon' ? '' : '@babylonjs\\/|'}pixi\\.js|@pmndrs\\/gi(\\/|$))|ddgi-(core|three)/]`;
const engineRules = (engine) => ({
    'no-restricted-imports': ['error', { patterns: [{ group: forbidden(engine), message: ENGINE_MESSAGE }] }],
    // no-restricted-imports only sees static imports, so catch import('three') too.
    'no-restricted-syntax': ['error', { selector: dynamicImportSelector(engine), message: ENGINE_MESSAGE }],
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
        ignores: ['src/**/*.test.ts', 'src/three/**', 'src/babylon/**'],
        rules: engineRules(null),
    },
    {
        // @pmndrs/bvh/three: the one place three (a peer dependency) may be imported.
        files: ['src/three/**/*.ts'],
        ignores: ['src/three/**/*.test.ts'],
        rules: engineRules('three'),
    },
    {
        // @pmndrs/bvh/babylon: the one place @babylonjs/* (a peer dependency) may be imported.
        files: ['src/babylon/**/*.ts'],
        ignores: ['src/babylon/**/*.test.ts'],
        rules: engineRules('babylon'),
    },
    {
        // Tests, GPU pages and tools may use three (as the reference builder and loader), the DOM and WebGPU.
        files: ['tests/**/*.ts', 'tools/**/*.ts', 'bench/**/*.ts', 'native/**/*.ts'],
        rules: { 'no-restricted-imports': 'off', 'no-restricted-syntax': 'off' },
    },
    {
        // The examples and the compare suite are apps: three (and the libraries compared) are allowed, and the library only through its public entry points
        // (@pmndrs/bvh, /three, /worker, aliased to src/ by examples/vite.config.ts), never by a path into src/.
        files: ['examples/**/*.ts', 'compare/**/*.ts', 'docs/snippets/**/*.ts'],
        rules: {
            'no-restricted-imports': ['error', { patterns: [{ group: ['**/src/**', '../../src/*', '../../../src/*'], message: 'Examples use the package by its public names: @pmndrs/bvh, @pmndrs/bvh/three, @pmndrs/bvh/worker.' }] }],
            'no-restricted-syntax': 'off',
        },
    },
    {
        // The vanilla examples prove the "not a three library" claim: raw WebGPU and @pmndrs/bvh only, and only the shared modules that are three-free.
        files: ['examples/vanilla/**/*.ts', 'examples/shared/vanilla/**/*.ts'],
        rules: {
            'no-restricted-imports': [
                'error',
                {
                    paths: [{ name: 'three', message: 'The vanilla examples use no engine.' }],
                    patterns: [
                        { group: ['**/src/**', '../../src/*', '../../../src/*'], message: 'Examples use the package by its public names: @pmndrs/bvh, @pmndrs/bvh/three, @pmndrs/bvh/worker.' },
                        { group: ['three/*', '@pmndrs/bvh/three'], message: 'The vanilla examples use no engine.' },
                        { group: ['**/shared/app.ts', '**/shared/loader.ts', '**/shared/compare.ts'], message: 'Three shell modules; the vanilla pages use shared/vanilla/*.' },
                    ],
                },
            ],
        },
    },
);

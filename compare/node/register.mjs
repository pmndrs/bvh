// Node resolve hook for the compare runner: the package's public names map to
// its sources, as examples/vite.config.ts aliases them, so the adapters import
// exactly what an app would (and need no `pnpm build` first).
//
//   node --experimental-transform-types --import ./compare/node/register.mjs compare/node/run.ts

import { register } from 'node:module';

register(
    'data:text/javascript,' +
        encodeURIComponent(`
const root = ${JSON.stringify(new URL('../../', import.meta.url).href)};
const aliases = {
    '@pmndrs/bvh': root + 'src/index.ts',
    '@pmndrs/bvh/three': root + 'src/three/index.ts',
    '@pmndrs/bvh/worker': root + 'src/worker/index.ts',
};
export async function resolve(specifier, context, next) {
    if (Object.hasOwn(aliases, specifier)) return { url: aliases[specifier], shortCircuit: true };
    return next(specifier, context);
}
`),
);

/**
 * Builds the docs website with pmndrs/docs, as the docs workflow does.
 *
 *     pnpm docs:api               # TypeDoc into api-docs/ (optional; copied to <out>/api when present)
 *     pnpm docs:site [out]        # the website into out/ (default docs-site/)
 *     npx -y serve docs-site      # look at it
 *
 * pmndrs/docs renders every .md and .mdx file under the folder it is given, so
 * the site is built from a copy of docs/ without the contributor notes
 * (docs/internals, plain Markdown that isn't MDX) and the code samples
 * (docs/snippets). The pages' relative paths stay the same, so "edit this
 * page" links still point at docs/<page>.mdx.
 *
 * The site's settings come from the environment where set (NEXT_PUBLIC_URL,
 * MDX_BASEURL, SOURCECODE_BASEURL, EDIT_BASEURL, BASE_PATH, ...: see
 * `npx @pmndrs/docs build --help`), else from the defaults below.
 */

import { spawnSync } from 'node:child_process';
import { cpSync, existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const DOCS = join(ROOT, 'docs');
const LEFT_OUT = ['internals', 'snippets'].map((name) => join(DOCS, name));

const out = resolve(ROOT, process.argv[2] ?? 'docs-site');
const stage = mkdtempSync(join(tmpdir(), 'bvh-docs-'));
try {
    const mdx = join(stage, 'docs');
    cpSync(DOCS, mdx, { recursive: true, filter: (source) => !LEFT_OUT.some((dir) => source === dir || source.startsWith(dir + sep)) });

    const env = {
        NEXT_PUBLIC_LIBNAME: '@pmndrs/bvh',
        NEXT_PUBLIC_LIBNAME_SHORT: 'bvh',
        NEXT_PUBLIC_LIBNAME_DOTSUFFIX_LABEL: 'docs',
        NEXT_PUBLIC_LIBNAME_DOTSUFFIX_HREF: 'https://docs.pmnd.rs',
        HOME_REDIRECT: '/getting-started/introduction',
        ICON: '🌳',
        GITHUB: 'https://github.com/pmndrs/bvh',
        SOURCECODE_BASEURL: 'https://github.com/pmndrs/bvh/tree/main',
        EDIT_BASEURL: 'https://github.com/pmndrs/bvh/edit/main/docs',
        MDX_BASEURL: 'https://github.com/pmndrs/bvh/raw/main/docs',
        ...process.env,
    };
    rmSync(out, { recursive: true, force: true });
    const build = spawnSync('npx', ['-y', '@pmndrs/docs@4', 'build', mdx, out, '--format', 'website'], { stdio: 'inherit', env });
    if (build.status !== 0) process.exit(build.status ?? 1);

    const api = join(ROOT, 'api-docs');
    if (existsSync(api)) {
        cpSync(api, join(out, 'api'), { recursive: true });
        console.log(`docs: API reference copied to ${relative(ROOT, join(out, 'api'))}`);
    }
    console.log(`docs: website in ${relative(ROOT, out) || '.'}`);
} finally {
    rmSync(stage, { recursive: true, force: true });
}

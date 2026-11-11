// Builds dist/: tsc emits ESM .js and .d.ts per source file (comments kept, so the
// /*! */ licence headers survive), then the .d.ts files' relative `.ts` imports,
// which tsc leaves as written, are rewritten to `.js` so consumers resolve them.
import { execFileSync } from 'node:child_process';
import { readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const dist = resolve(root, 'dist');

rmSync(dist, { recursive: true, force: true });
execFileSync(process.execPath, [resolve(root, 'node_modules/typescript/bin/tsc'), '-p', 'tsconfig.build.json'], { cwd: root, stdio: 'inherit' });

const relativeTs = /((?:from\s*|import\s*\(\s*|import\s+)["']\.{1,2}\/[^"']*)\.ts(["'])/g;
// Worker entries are started with `new Worker(new URL('./x.ts', import.meta.url))`, which tsc leaves as
// written: point them at the built `.js` (bundlers follow the same expression).
const workerUrlTs = /(new URL\(\s*["']\.{1,2}\/[^"']*)\.ts(["']\s*,\s*import\.meta\.url\s*\))/g;
for (const entry of readdirSync(dist, { recursive: true, withFileTypes: true })) {
    if (!entry.isFile()) continue;
    const isDeclaration = entry.name.endsWith('.d.ts');
    if (!isDeclaration && !entry.name.endsWith('.js')) continue;
    const file = resolve(entry.parentPath, entry.name);
    const text = readFileSync(file, 'utf8');
    const rewritten = isDeclaration ? text.replace(relativeTs, '$1.js$2') : text.replace(workerUrlTs, '$1.js$2');
    if (rewritten !== text) writeFileSync(file, rewritten);
}
console.log('Built dist/');

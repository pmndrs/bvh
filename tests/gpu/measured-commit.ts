/// <reference types="node" />
/**
 * Which commit a GPU measurement belongs to. The runner records it when a run
 * starts (`RunSummary.commit`), so CSV rows written later still name the code
 * that was measured (T1.3: `to-csv.ts` used to read it when it wrote the CSV).
 */

import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';

const REPOSITORY_ROOT = resolve(import.meta.dirname, '..', '..');

/**
 * The files whose changes can move a trace-bench number, as git pathspecs:
 * - the library (`src`: tracer, builders, tinybvh);
 * - the GPU runner and page client (the files directly in tests/gpu);
 * - the trace pages themselves (trace-bench*, trace-oracle);
 * - the lockfile, which pins three and three-mesh-bvh.
 * Work elsewhere (native/, other pages) must not mark every row dirty, so a
 * whole-tree check is not used. bench/results is output, and the scene assets
 * live outside this repository, so neither counts.
 */
export const MEASURED_PATHS = [
    'src',
    ':(glob)tests/gpu/*',
    ':(glob)tests/gpu/pages/trace-*/**',
    'pnpm-lock.yaml',
] as const;

/** HEAD's short hash, with "-dirty" when a measured file differs from HEAD (modified, staged or new and not ignored). */
export function gitCommit(): string {
    const git = (...args: string[]) => execFileSync('git', args, { cwd: REPOSITORY_ROOT, encoding: 'utf8' }).trim();
    const hash = git('rev-parse', '--short', 'HEAD');
    const changes = git('status', '--porcelain', '--untracked-files=all', '--', ...MEASURED_PATHS);
    return changes === '' ? hash : `${hash}-dirty`;
}


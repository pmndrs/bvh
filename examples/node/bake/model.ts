/**
 * The default model of `node/bake`, shared by the CLI (which bakes it) and the
 * page (which, in a smoke run only, builds it fresh to compare with the bake).
 * Browser-safe: no Node imports.
 */

import { torusKnot, type IndexedMesh } from '../../shared/vanilla/geometry.ts';

/** A 6,400-triangle torus knot, small enough to commit its bake (about 0.5 MB). */
export function defaultMesh(): IndexedMesh {
    return torusKnot(1, 0.3, 200, 16);
}

/** The bake's leaf size: what the CLI uses with no `--leaf`. */
export const DEFAULT_LEAF = 8;

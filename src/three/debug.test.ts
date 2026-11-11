/**
 * The debug functions from `/three` on a `geometry.boundsTree`: the same
 * output shape as three-mesh-bvh's, and `validateBounds` seeing edits to the
 * geometry until `refit()`, as three-mesh-bvh's does.
 */

import { TorusKnotGeometry } from 'three';
import { getBVHExtremes as tmbGetBVHExtremes, MeshBVH as TmbMeshBVH, validateBounds as tmbValidateBounds } from 'three-mesh-bvh';
import { beforeAll, describe, expect, it, vi } from 'vitest';
import { estimateMemoryInBytes, getBVHExtremes, validateBounds } from './index.ts';
import { BvhBoundsTree, prepareBoundsTrees } from './raycast.ts';

beforeAll(async () => {
    await prepareBoundsTrees();
});

const shape = (value: unknown): unknown =>
    value && typeof value === 'object' ? Object.fromEntries(Object.entries(value).map(([key, inner]) => [key, Array.isArray(inner) ? inner.length : shape(inner)])) : typeof value;

describe('debug functions in /three', () => {
    it('getBVHExtremes has three-mesh-bvh output shape, and validateBounds follows geometry edits', () => {
        const geometry = new TorusKnotGeometry(1, 0.3, 64, 8);
        const ours = new BvhBoundsTree(geometry);
        const theirs = new TmbMeshBVH(geometry.clone());
        const [extremes] = getBVHExtremes(ours);
        const [theirExtremes] = tmbGetBVHExtremes(theirs);
        expect(shape(extremes)).toEqual(shape(theirExtremes));
        expect(Object.keys(extremes).sort()).toEqual(Object.keys(theirExtremes).sort());
        // Both trees hold every triangle; ours has at most 8 per leaf (three-mesh-bvh's default target is 10).
        expect(extremes.primitives.max).toBeLessThanOrEqual(8);
        expect(estimateMemoryInBytes(ours)).toBe(ours.estimateMemoryInBytes());

        const assert = vi.spyOn(console, 'assert').mockImplementation(() => {});
        try {
            expect(validateBounds(ours)).toBe(true);
            expect(tmbValidateBounds(theirs)).toBe(true);
            const position = geometry.getAttribute('position');
            position.setX(0, position.getX(0) + 10);
            position.needsUpdate = true;
            expect(validateBounds(ours)).toBe(false);
            ours.refit();
            expect(validateBounds(ours)).toBe(true);
        } finally {
            assert.mockRestore();
        }
    });
});

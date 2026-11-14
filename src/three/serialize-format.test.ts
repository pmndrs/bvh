/**
 * three-mesh-bvh's serialized format (PARITY 7.1) and `getJSONStructure`
 * (7.5), checked against three-mesh-bvh 0.9.18 itself: our trees written in
 * its format are read by its `MeshBVH.deserialize` and give its own walks the
 * same answers as ours (raycasts, box casts, `validateBounds`, the same
 * `getJSONStructure` node for node); its trees, indirect or not, with groups
 * (several roots) and in the version 0 layout, are read by ours with their
 * shape kept.
 */

import { BufferAttribute, BufferGeometry, DoubleSide, Ray, SphereGeometry, TorusKnotGeometry, Vector3, Box3, Matrix4 } from 'three';
import { getJSONStructure as tmbGetJSONStructure, MeshBVH as TmbMeshBVH, SAH as TMB_SAH, validateBounds as tmbValidateBounds } from 'three-mesh-bvh';
import { beforeAll, describe, expect, it, vi } from 'vitest';
import { createRandom } from '../trace/oracle-fixtures.ts';
import { getJSONStructure as coreGetJSONStructure, topologyFromThreeMeshBvh, type BvhJsonNode, type ThreeMeshBvhSerialized } from '../query/three-mesh-bvh-format.ts';
import { getJSONStructure, MeshBVH, validateBounds } from './index.ts';
import { BvhBoundsTree, prepareBoundsTrees } from './raycast.ts';

beforeAll(async () => {
    await prepareBoundsTrees();
});

function randomRay(random: () => number): Ray {
    const origin = new Vector3((random() - 0.5) * 6, (random() - 0.5) * 6, (random() - 0.5) * 6);
    const target = new Vector3((random() - 0.5) * 1.5, (random() - 0.5) * 1.5, (random() - 0.5) * 1.5);
    return new Ray(origin, target.sub(origin).normalize());
}

/** Hits as `faceIndex@distance`, sorted, rounded past f32 noise. */
const hitKeys = (hits: readonly { faceIndex?: number | null; distance: number }[]) => hits.map((hit) => `${hit.faceIndex}@${hit.distance.toFixed(5)}`).sort();

/** A JSON tree's shape: leaves' offsets and counts, inner nodes' children, no boxes. */
type Shape = { offset: number; count: number } | { left: Shape; right: Shape };
function shapeOf(node: BvhJsonNode<Box3>): Shape {
    if (node.count !== undefined) return { offset: node.offset!, count: node.count };
    return { left: shapeOf(node.left!), right: shapeOf(node.right!) };
}

/** Every box of `inner` lies in the matching box of `outer` (same shape). */
function boxesWithin(inner: BvhJsonNode<Box3>, outer: BvhJsonNode<Box3>): boolean {
    if (!outer.bounds.containsBox(inner.bounds)) return false;
    if (inner.count !== undefined) return true;
    return boxesWithin(inner.left!, outer.left!) && boxesWithin(inner.right!, outer.right!);
}

/** Three random triangles' soup, no index. */
function soup(count: number, seed: number): BufferGeometry {
    const random = createRandom(seed);
    const positions = new Float32Array(count * 9);
    for (let t = 0; t < count; t++) {
        const centre = [(random() - 0.5) * 4, (random() - 0.5) * 4, (random() - 0.5) * 4];
        for (let k = 0; k < 9; k++) positions[t * 9 + k] = centre[k % 3] + (random() - 0.5) * 0.6;
    }
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(positions, 3));
    return geometry;
}

describe("three-mesh-bvh's serialized format: ours read by three-mesh-bvh", () => {
    it('MeshBVH.serialize writes a tree three-mesh-bvh deserializes and walks to the same answers', () => {
        const random = createRandom(7);
        for (const [name, geometry] of [
            ['indexed knot', new TorusKnotGeometry(1, 0.3, 64, 10)],
            ['index-free soup', soup(700, 3)],
        ] as const) {
            const ours = new BvhBoundsTree(geometry);
            const data = MeshBVH.serialize(ours);
            expect(data.version).toBe(1);
            expect(data.roots).toHaveLength(1);
            expect(data.indirectBuffer!.length).toBe(ours.triangleCount);
            // three-mesh-bvh's own reader; a geometry without an index needs setIndex: false (its data.index is null).
            const indexed = geometry.index !== null;
            const theirs = TmbMeshBVH.deserialize(data, geometry.clone(), { setIndex: indexed });
            expect(theirs.indirect, name).toBe(true);
            const assert = vi.spyOn(console, 'assert').mockImplementation(() => {});
            try {
                expect(tmbValidateBounds(theirs), name).toBe(true);
            } finally {
                assert.mockRestore();
            }
            // getJSONStructure node for node: the same shape, offsets, counts and boxes.
            expect(JSON.parse(JSON.stringify(getJSONStructure(ours))), name).toEqual(JSON.parse(JSON.stringify(tmbGetJSONStructure(theirs))));
            for (let r = 0; r < 300; r++) {
                const ray = randomRay(random);
                expect(hitKeys(theirs.raycast(ray, DoubleSide)), name).toEqual(hitKeys(ours.raycast(ray, DoubleSide)));
                expect(theirs.raycastFirst(ray, DoubleSide)?.faceIndex ?? null, name).toBe(ours.raycastFirst(ray, DoubleSide)?.faceIndex ?? null);
            }
            for (let b = 0; b < 100; b++) {
                const centre = new Vector3((random() - 0.5) * 4, (random() - 0.5) * 4, (random() - 0.5) * 4);
                const box = new Box3().setFromCenterAndSize(centre, new Vector3(random(), random(), random()).multiplyScalar(0.8));
                const matrix = new Matrix4().makeRotationY(random() * 3);
                expect(theirs.intersectsBox(box, matrix), name).toBe(ours.intersectsBox(box, matrix));
            }
        }
    });

    it('copies the index unless cloneBuffers is false, and reads back into ours unchanged', () => {
        const geometry = new TorusKnotGeometry(1, 0.3, 32, 6);
        const ours = new BvhBoundsTree(geometry);
        expect(MeshBVH.serialize(ours).index).not.toBe(geometry.index!.array);
        expect(MeshBVH.serialize(ours).index).toEqual(geometry.index!.array);
        expect(MeshBVH.serialize(ours, { cloneBuffers: false }).index).toBe(geometry.index!.array);
        // Our own written tree comes back with the same shape (boxes made exact again, so equal here too).
        const back = MeshBVH.deserialize(MeshBVH.serialize(ours), geometry.clone());
        expect(shapeOf(getJSONStructure(back))).toEqual(shapeOf(getJSONStructure(ours)));
        expect(new Uint8Array(back.layout.triangles)).toEqual(new Uint8Array(ours.layout.triangles));
        // Our own bytes still load through the same call.
        const fromBytes = MeshBVH.deserialize(ours.serialize(), geometry);
        expect(new Uint8Array(fromBytes.serialize())).toEqual(new Uint8Array(ours.serialize()));
    });
});

describe("three-mesh-bvh's serialized format: three-mesh-bvh's read by ours", () => {
    it('keeps the shape of an indirect tree, with exact boxes inside its padded ones, and gives its hits', () => {
        const random = createRandom(11);
        const geometry = new TorusKnotGeometry(1, 0.3, 64, 10);
        // Leaves of at most 8 (our limit), so no leaf is cut and the shapes match exactly.
        const theirs = new TmbMeshBVH(geometry.clone(), { strategy: TMB_SAH, targetLeafSize: 8, indirect: true, verbose: false });
        const data = TmbMeshBVH.serialize(theirs) as ThreeMeshBvhSerialized;
        const ours = MeshBVH.deserialize(data, geometry);
        expect(validateBounds(ours)).toBe(true);
        const theirJson = tmbGetJSONStructure(theirs) as BvhJsonNode<Box3>;
        expect(shapeOf(getJSONStructure(ours))).toEqual(shapeOf(theirJson));
        expect(boxesWithin(getJSONStructure(ours), theirJson)).toBe(true);
        for (let r = 0; r < 300; r++) {
            const ray = randomRay(random);
            expect(hitKeys(ours.raycast(ray, DoubleSide))).toEqual(hitKeys(theirs.raycast(ray, DoubleSide)));
        }
    });

    it('reads a tree that reordered the index, with groups (several roots) and leaves past 8', () => {
        const random = createRandom(23);
        const source = new TorusKnotGeometry(1, 0.3, 48, 8);
        const indexCount = source.index!.count;
        source.addGroup(0, Math.floor(indexCount / 6) * 3, 0);
        source.addGroup(Math.floor(indexCount / 6) * 3, indexCount - Math.floor(indexCount / 6) * 3, 1);
        const built = source.clone();
        // Not indirect: three-mesh-bvh reorders `built.index` in place, one root per group; default leaves of 10.
        const theirs = new TmbMeshBVH(built, { verbose: false });
        const data = TmbMeshBVH.serialize(theirs) as ThreeMeshBvhSerialized;
        expect(data.roots).toHaveLength(2);
        expect(data.indirectBuffer).toBeNull();
        // setIndex (the default) gives our geometry the reordered index, as three-mesh-bvh's deserialize does.
        const target = source.clone();
        const ours = MeshBVH.deserialize(data, target);
        expect(Array.from(target.index!.array)).toEqual(Array.from(built.index!.array));
        expect(validateBounds(ours)).toBe(true);
        for (let r = 0; r < 300; r++) {
            const ray = randomRay(random);
            expect(hitKeys(ours.raycast(ray, DoubleSide))).toEqual(hitKeys(theirs.raycast(ray, DoubleSide)));
        }
        // The topology: one combiner over the two roots.
        const topology = topologyFromThreeMeshBvh(data, indexCount / 3);
        expect(topology.children.length % 4).toBe(0);
    });

    it('reads version 0 data (absolute right-child words), and refuses trees that leave triangles out', () => {
        const geometry = new SphereGeometry(1, 24, 16);
        const theirs = new TmbMeshBVH(geometry.clone(), { indirect: true, targetLeafSize: 6, verbose: false });
        const data = TmbMeshBVH.serialize(theirs) as ThreeMeshBvhSerialized;
        const ours = MeshBVH.deserialize(data, geometry.clone());
        // Version 0: the right child is an absolute uint32 index (three-mesh-bvh's fixupVersion0 undoes this).
        const old = data.roots[0].slice(0);
        const words = new Uint32Array(old);
        const halves = new Uint16Array(old);
        for (let node = 0; node < old.byteLength / 32; node++) if (halves[node * 16 + 15] !== 0xffff) words[node * 8 + 6] = (node + words[node * 8 + 6]) * 8;
        const fromOld = MeshBVH.deserialize({ roots: [old], index: data.index, indirectBuffer: data.indirectBuffer }, geometry.clone());
        expect(new Uint8Array(fromOld.serialize())).toEqual(new Uint8Array(ours.serialize()));
        // Engine-free: the core reads the same data into the same JSON shape.
        expect(shapeOf(coreGetJSONStructure(ours, (box) => new Box3(new Vector3(box[0], box[1], box[2]), new Vector3(box[3], box[4], box[5]))))).toEqual(shapeOf(getJSONStructure(ours)));

        const partial = geometry.clone();
        partial.setDrawRange(0, 300);
        const cut = TmbMeshBVH.serialize(new TmbMeshBVH(partial, { indirect: true, verbose: false })) as ThreeMeshBvhSerialized;
        expect(() => MeshBVH.deserialize(cut, geometry.clone())).toThrow(/leave triangles out/);
        expect(() => MeshBVH.deserialize({ ...data, version: 7 }, geometry.clone())).toThrow(/version 7/);
    });
});

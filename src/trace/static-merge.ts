/**
 * Merging static instances into world-space meshes (`mergeStaticInstances`,
 * off by default; docs/design/trace.md, "Merged static instances").
 *
 * The two-level walk tests every instance box a ray enters. In scenes with
 * one instance per mesh (Sponza, Bistro) those boxes overlap a lot, and a ray
 * that starts on a surface enters many instances that it never hits. Merging
 * puts such instances' triangles into one world-space tree instead, under an
 * identity transform, so the walk descends one tree.
 *
 * **What merges.** An instance marked `static: true` whose mesh no other
 * instance uses, with an invertible transform. Instances merge in groups of
 * equal `contributesToGi`, `visible` and mask, so each group still has one
 * instance record with the right flags and mask. Everything else stays a
 * normal instance.
 *
 * **Hit identity.** A merged mesh is a normal mesh, so any builder builds it
 * from {@link StaticMergeGroup.geometry}. The ids stay out of the builder:
 * `assembleTraceScene` writes them into its copy of the built records
 * ({@link writeMergedTriangleIdentity}): the original index becomes the
 * triangle's index in its source mesh, the padding word the source instance
 * index, and flags bit 1 marks triangles of a mirrored source (world-space
 * winding is the reverse of the authored one). The walk takes the hit's
 * instance from the triangle when the record has `INSTANCE_FLAG.merged`, and
 * flips `frontFace` on bit 1. Nothing grows: the words were already there.
 *
 * **Watertightness.** Each source mesh's vertices are transformed once each
 * (the indices are kept), so a vertex two triangles share has identical
 * world-space bits in both, and the watertight test still holds.
 */

import { MAX_REFERENCE_INDEX, TRIANGLE_FLAG_FLIP_FRONT_FACE, TRIANGLE_WORD, TRIANGLE_WORDS, checkInstanceMask } from './layout.ts';
import type { MeshGeometry } from './mesh-bvh.ts';
import { countTriangles } from './triangle-packing.ts';
import { invertAffine, type Mat4 } from './top-level.ts';
import type { TraceSceneInstance } from './scene-assembly.ts';

/** The instance index written to a merged group's own record. The walk never returns it: hits take the triangle's source instance. */
export const MERGED_INSTANCE_INDEX = 0xffffffff;

/** One instance merged into a group. */
export interface StaticMergeSource {
    /** Position in the instance array given to {@link mergeStaticGeometry}. */
    readonly inputIndex: number;
    readonly instanceIndex: number;
    readonly meshIndex: number;
    /** The source's first triangle in the group's geometry, and how many it has. */
    readonly firstTriangle: number;
    readonly triangleCount: number;
    /** The transform's determinant is negative: the source's triangles carry the flip bit. */
    readonly mirrored: boolean;
    /** The transform the triangles were moved with (a copy). Assembly refuses the instance if its transform differs. */
    readonly objectToWorld: Float64Array;
}

/** One merged mesh: what to build, and the ids assembly writes into the built records. */
export interface StaticMergeGroup {
    /** World-space positions (each source vertex transformed once), `Uint32Array` indices, and the sources' triangle flags (filter bit). */
    readonly geometry: MeshGeometry;
    /** Per triangle of `geometry`: the source instance index (`TraceSceneInstance.instanceIndex`). */
    readonly sourceInstances: Uint32Array;
    /** Per triangle of `geometry`: the triangle's index in its source mesh. */
    readonly sourceTriangles: Uint32Array;
    /** Per triangle of `geometry`: 1 when it comes from a mirrored source (`TRIANGLE_FLAG_FLIP_FRONT_FACE`). */
    readonly flipFrontFace: Uint8Array;
    readonly sources: readonly StaticMergeSource[];
    readonly contributesToGi: boolean;
    readonly visible: boolean;
    readonly mask: number;
}

/** What {@link mergeStaticGeometry} decided. */
export interface StaticMergePlan {
    readonly groups: readonly StaticMergeGroup[];
    /** Positions (in the input array) of the instances that stay normal instances, in input order. */
    readonly instanced: Uint32Array;
    /** Per input mesh: 1 when a normal instance uses it, so it needs its own tree; 0 when only merged sources use it. */
    readonly meshNeeded: Uint8Array;
    /** Length of the instance array the plan was made for. */
    readonly instanceCount: number;
}

/** Options of {@link mergeStaticGeometry}. */
export interface StaticMergeOptions {
    /** Most triangles in one merged mesh; a group that would pass it is split. Default (and at most) the layout's limit, 2^28 - 1. */
    readonly maxTrianglesPerGroup?: number;
}

/**
 * Picks the instances that merge and produces each group's world-space
 * geometry, as a normal mesh input for any builder, with the per-triangle
 * source ids and flip bits kept beside it. Pure: nothing is built.
 */
export function mergeStaticGeometry(
    meshes: readonly MeshGeometry[],
    instances: readonly TraceSceneInstance[],
    options: StaticMergeOptions = {},
): StaticMergePlan {
    const maxTriangles = Math.min(options.maxTrianglesPerGroup ?? MAX_REFERENCE_INDEX, MAX_REFERENCE_INDEX);
    const users = new Uint32Array(meshes.length);
    for (const instance of instances) {
        if (!meshes[instance.meshIndex]) throw new RangeError(`instance ${instance.instanceIndex} uses mesh ${instance.meshIndex}, which does not exist`);
        users[instance.meshIndex]++;
    }

    interface Pending {
        readonly key: string;
        readonly contributesToGi: boolean;
        readonly visible: boolean;
        readonly mask: number;
        readonly members: { inputIndex: number; mirrored: boolean }[];
        triangles: number;
        vertices: number;
    }
    const open = new Map<string, Pending>();
    const closed: Pending[] = [];
    const meshNeeded = new Uint8Array(meshes.length);
    const instanced: number[] = [];

    instances.forEach((instance, inputIndex) => {
        const mask = checkInstanceMask(instance.mask, instance.instanceIndex);
        const inverse = instance.static === true && users[instance.meshIndex] === 1 ? invertAffine(instance.objectToWorld) : null;
        if (inverse === null) {
            // Not static, a shared mesh, or a singular transform (which the top level skips as an instance).
            instanced.push(inputIndex);
            meshNeeded[instance.meshIndex] = 1;
            return;
        }
        const geometry = meshes[instance.meshIndex];
        const triangles = countTriangles(geometry);
        const vertices = geometry.positions.length / 3;
        const contributesToGi = instance.contributesToGi !== false;
        const visible = instance.visible !== false;
        const key = `${contributesToGi ? 1 : 0}:${visible ? 1 : 0}:${mask}`;
        let group = open.get(key);
        if (group && group.triangles + triangles > maxTriangles) {
            closed.push(group);
            group = undefined;
        }
        if (!group) {
            if (triangles > maxTriangles) throw new RangeError(`instance ${instance.instanceIndex}: ${triangles} triangles do not fit one merged mesh (${maxTriangles})`);
            group = { key, contributesToGi, visible, mask, members: [], triangles: 0, vertices: 0 };
            open.set(key, group);
        }
        group.members.push({ inputIndex, mirrored: inverse.determinant < 0 });
        group.triangles += triangles;
        group.vertices += vertices;
    });
    closed.push(...open.values());
    // Groups in the order their first member came, so the plan doesn't depend on Map details.
    closed.sort((a, b) => a.members[0].inputIndex - b.members[0].inputIndex);

    const groups = closed.map((pending): StaticMergeGroup => {
        const positions = new Float32Array(pending.vertices * 3);
        const indices = new Uint32Array(pending.triangles * 3);
        const sourceInstances = new Uint32Array(pending.triangles);
        const sourceTriangles = new Uint32Array(pending.triangles);
        const flipFrontFace = new Uint8Array(pending.triangles);
        const anyFlags = pending.members.some((member) => meshes[instances[member.inputIndex].meshIndex].triangleFlags);
        const triangleFlags = anyFlags ? new Uint32Array(pending.triangles) : null;
        const sources: StaticMergeSource[] = [];
        let vertexBase = 0;
        let triangleBase = 0;
        for (const member of pending.members) {
            const instance = instances[member.inputIndex];
            const mesh = meshes[instance.meshIndex];
            const triangles = countTriangles(mesh);
            const vertexCount = mesh.positions.length / 3;
            transformPositions(mesh.positions, instance.objectToWorld, positions, vertexBase * 3);
            for (let triangle = 0; triangle < triangles; triangle++) {
                for (let corner = 0; corner < 3; corner++) {
                    const vertex = mesh.indices ? mesh.indices[triangle * 3 + corner] : triangle * 3 + corner;
                    if (vertex >= vertexCount) throw new RangeError(`instance ${instance.instanceIndex}: triangle ${triangle} uses vertex ${vertex}, but there are ${vertexCount}`);
                    indices[(triangleBase + triangle) * 3 + corner] = vertexBase + vertex;
                }
            }
            sourceInstances.fill(instance.instanceIndex, triangleBase, triangleBase + triangles);
            for (let triangle = 0; triangle < triangles; triangle++) sourceTriangles[triangleBase + triangle] = triangle;
            if (member.mirrored) flipFrontFace.fill(1, triangleBase, triangleBase + triangles);
            if (triangleFlags && mesh.triangleFlags) {
                if (mesh.triangleFlags.length < triangles) throw new RangeError(`mesh ${instance.meshIndex}: triangleFlags must have one entry per triangle`);
                triangleFlags.set(mesh.triangleFlags.subarray(0, triangles), triangleBase);
            }
            sources.push({
                inputIndex: member.inputIndex,
                instanceIndex: instance.instanceIndex,
                meshIndex: instance.meshIndex,
                firstTriangle: triangleBase,
                triangleCount: triangles,
                mirrored: member.mirrored,
                objectToWorld: Float64Array.from({ length: 16 }, (_, k) => instance.objectToWorld[k]),
            });
            vertexBase += vertexCount;
            triangleBase += triangles;
        }
        return {
            geometry: { positions, indices, triangleFlags },
            sourceInstances,
            sourceTriangles,
            flipFrontFace,
            sources,
            contributesToGi: pending.contributesToGi,
            visible: pending.visible,
            mask: pending.mask,
        };
    });

    return { groups, instanced: Uint32Array.from(instanced), meshNeeded, instanceCount: instances.length };
}

/**
 * Object-space positions to world space, one vertex at a time, in doubles and
 * rounded once to f32: the same input bits always give the same output bits.
 */
function transformPositions(source: Float32Array, e: Mat4, out: Float32Array, offset: number): void {
    const m00 = e[0], m01 = e[4], m02 = e[8], m03 = e[12];
    const m10 = e[1], m11 = e[5], m12 = e[9], m13 = e[13];
    const m20 = e[2], m21 = e[6], m22 = e[10], m23 = e[14];
    for (let at = 0; at < source.length; at += 3) {
        const x = source[at];
        const y = source[at + 1];
        const z = source[at + 2];
        out[offset + at] = m00 * x + m01 * y + m02 * z + m03;
        out[offset + at + 1] = m10 * x + m11 * y + m12 * z + m13;
        out[offset + at + 2] = m20 * x + m21 * y + m22 * z + m23;
    }
}

/**
 * Writes a merged group's hit identity into a tree's triangle records, built
 * from `group.geometry` by any builder: each record's original index (the
 * builder's, an index into the group's geometry) becomes the source mesh's
 * triangle index, the padding word the source instance index, and flags bit 1
 * is set for a mirrored source. Every other flag bit is kept.
 *
 * Call it once on fresh records (it reads the builder's original index):
 * `assembleTraceScene` does, on its own copy, so a `MeshBvh` can be
 * assembled again or swapped for another builder's tree the same way.
 */
export function writeMergedTriangleIdentity(records: ArrayBuffer, group: StaticMergeGroup, byteOffset = 0, triangleCount = group.sourceInstances.length): void {
    if (triangleCount !== group.sourceInstances.length) {
        throw new RangeError(`the merged tree has ${triangleCount} triangles, its group ${group.sourceInstances.length}`);
    }
    const words = new Uint32Array(records, byteOffset, triangleCount * TRIANGLE_WORDS);
    for (let slot = 0; slot < triangleCount; slot++) {
        const base = slot * TRIANGLE_WORDS;
        const merged = words[base + TRIANGLE_WORD.originalIndex];
        if (merged >= triangleCount) throw new RangeError(`merged triangle record ${slot} holds triangle ${merged}, out of range`);
        words[base + TRIANGLE_WORD.originalIndex] = group.sourceTriangles[merged];
        words[base + TRIANGLE_WORD.padding] = group.sourceInstances[merged];
        if (group.flipFrontFace[merged]) words[base + TRIANGLE_WORD.flags] |= TRIANGLE_FLAG_FLIP_FRONT_FACE;
    }
}

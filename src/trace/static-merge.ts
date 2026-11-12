/**
 * Merging static instances into world-space meshes (`mergeStaticInstances`,
 * off by default; docs/internals/design/trace.md, "Merged static instances").
 *
 * The two-level walk tests every instance box a ray enters. In scenes with
 * one instance per mesh (Sponza, Bistro) those boxes overlap a lot, and a ray
 * that starts on a surface enters many instances that it never hits. Merging
 * puts such instances' triangles into one world-space tree instead, under an
 * identity transform, so the walk descends one tree.
 *
 * **What merges.** An instance marked `static: true` whose mesh no other
 * instance uses and has triangles, with an invertible transform, that is
 * visible and contributes to GI (the top level skips the others anyway, so
 * they stay normal instances and the scene is what it is with the option
 * off). Instances merge in groups of equal mask, so each group still has one
 * instance record with the right mask. Merged sources must have distinct
 * `instanceIndex` values, since a hit names its source by that index.
 * Everything else stays a normal instance.
 *
 * **Hit identity.** A merged mesh is a normal mesh, so any builder builds it
 * from {@link StaticMergeGroup.geometry}. The ids stay out of the builder:
 * `assembleTraceScene` writes them into its copy of the built records
 * ({@link writeMergedTriangleIdentity}): the original index becomes the
 * triangle's index in its source mesh, the padding word the source instance
 * index, and flags bit 1 marks triangles of a mirrored source (world-space
 * winding is the reverse of the authored one), and flags bits 16-31 the
 * triangle's material + 1 (0 for none): the source mesh's per-triangle
 * material (`triangleMaterials`) if it has one, else the source instance's. The walk takes the hit's instance and
 * material from the triangle when the record has `INSTANCE_FLAG.merged`, and
 * flips `frontFace` (and the hit normal) on bit 1. Nothing grows: the words
 * were already there. Material is not a grouping key.
 *
 * **Watertightness.** Each source mesh's vertices are transformed once each
 * (the indices are kept), so a vertex two triangles share has identical
 * world-space bits in both, and the watertight test still holds.
 */

import {
    INDEXED_FLAG_SHIFT,
    INDEXED_ID_MASK,
    INDEXED_TRIANGLE_WORDS,
    MATERIAL_SHIFT,
    MAX_MERGED_SOURCES,
    MERGED_SIDE_MATERIAL_SHIFT,
    MESH_HEADER_WORD,
    MAX_REFERENCE_INDEX,
    NO_MATERIAL,
    TRIANGLE_FLAG_FILTER,
    TRIANGLE_FLAG_FLIP_FRONT_FACE,
    TRIANGLE_WORD,
    TRIANGLE_WORDS,
    checkInstanceMask,
    materialCode,
    type TriangleLeaves,
} from './layout.ts';
import { checkTriangleLeaves } from '../indexed-leaves.ts';
import type { MeshGeometry } from './mesh-bvh.ts';
import { checkTriangleFlags, triangleMaterialCodes } from '../mesh-input.ts';
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
    /** The instance's material index, `NO_MATERIAL` for none. Assembly refuses the instance if its material differs. */
    readonly material: number;
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
    /**
     * Per triangle of `geometry`: its material as stored in flags bits 16-31
     * (material + 1, 0 for none): the source mesh's `triangleMaterials` entry,
     * else the source instance's material. The group's `geometry` carries no
     * materials; assembly writes these.
     */
    readonly sourceMaterials: Uint16Array;
    readonly sources: readonly StaticMergeSource[];
    /** Always true: instances that don't contribute to GI, or aren't visible, never merge. */
    readonly contributesToGi: boolean;
    /** Always true (see {@link contributesToGi}). */
    readonly visible: boolean;
    /** The sources' mask (0-255, 0xff when they give none). */
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
    /**
     * The format of the scene the plan is for. `'indexed'` also closes a group
     * at {@link MAX_MERGED_SOURCES} sources (an indexed scene names a merged
     * triangle's source by a 16-bit slot). Default `'records'`: no source cap,
     * the grouping it always had.
     */
    readonly triangleLeaves?: TriangleLeaves;
}

/**
 * Picks the instances that merge and produces each group's world-space
 * geometry, as a normal mesh input for any builder, with the per-triangle
 * source ids and flip bits kept beside it. Pure: nothing is built.
 *
 * Throws if a merged source's `triangleFlags` set a reserved bit (only bit 0
 * may be set), if two merged sources share an `instanceIndex`, or for a
 * `material` outside 0-65534. Material doesn't split groups: each triangle
 * carries its own source's.
 *
 * **Static means static.** The source transforms are baked into the
 * geometry. Only assembling the scene again (`assembleTraceScene` with this
 * plan) notices a source that moved, or whose mask, `visible` or
 * `contributesToGi` changed: nothing checks between assemblies. A merged
 * group's records must never be refit (`GpuMeshRefit`): they hold world-space
 * corners and source-mesh triangle indices, so a refit from a source's
 * object-space vertices would write wrong triangles.
 */
export function mergeStaticGeometry(
    meshes: readonly MeshGeometry[],
    instances: readonly TraceSceneInstance[],
    options: StaticMergeOptions = {},
): StaticMergePlan {
    const maxTriangles = Math.min(options.maxTrianglesPerGroup ?? MAX_REFERENCE_INDEX, MAX_REFERENCE_INDEX);
    const sourceCap = checkTriangleLeaves(options.triangleLeaves, 'mergeStaticGeometry') === 'indexed';
    const users = new Uint32Array(meshes.length);
    for (const instance of instances) {
        if (!meshes[instance.meshIndex]) throw new RangeError(`instance ${instance.instanceIndex} uses mesh ${instance.meshIndex}, which does not exist`);
        users[instance.meshIndex]++;
    }

    interface Pending {
        readonly key: string;
        readonly mask: number;
        readonly members: { inputIndex: number; mirrored: boolean }[];
        triangles: number;
        vertices: number;
    }
    const open = new Map<string, Pending>();
    const closed: Pending[] = [];
    const meshNeeded = new Uint8Array(meshes.length);
    const instanced: number[] = [];

    const mergedInstanceIndices = new Map<number, number>();
    instances.forEach((instance, inputIndex) => {
        const mask = checkInstanceMask(instance.mask, instance.instanceIndex);
        materialCode(instance.material, instance.instanceIndex);
        const geometry = meshes[instance.meshIndex];
        const triangles = countTriangles(geometry);
        // The top level skips hidden and non-GI instances, so they stay instances: the scene is then what it is with the option off.
        const candidate = instance.static === true && users[instance.meshIndex] === 1 && triangles > 0 && instance.visible !== false && instance.contributesToGi !== false;
        const inverse = candidate ? invertAffine(instance.objectToWorld) : null;
        if (inverse === null) {
            // Not static, a shared or empty mesh, hidden, or a singular transform (which the top level skips as an instance).
            instanced.push(inputIndex);
            meshNeeded[instance.meshIndex] = 1;
            return;
        }
        const sameIndex = mergedInstanceIndices.get(instance.instanceIndex);
        if (sameIndex !== undefined) {
            throw new RangeError(
                `instances at positions ${sameIndex} and ${inputIndex} both have instanceIndex ${instance.instanceIndex}; merged static instances need distinct indices, because a merged hit names its source by it`,
            );
        }
        mergedInstanceIndices.set(instance.instanceIndex, inputIndex);
        checkTriangleFlags(geometry.triangleFlags, triangles, `mesh ${instance.meshIndex} (instance ${instance.instanceIndex})`);
        const vertices = geometry.positions.length / 3;
        const key = `${mask}`;
        let group = open.get(key);
        // An indexed scene names a merged triangle's source by a 16-bit slot (MAX_MERGED_SOURCES); records have no such cap.
        if (group && (group.triangles + triangles > maxTriangles || (sourceCap && group.members.length >= MAX_MERGED_SOURCES))) {
            closed.push(group);
            group = undefined;
        }
        if (!group) {
            if (triangles > maxTriangles) throw new RangeError(`instance ${instance.instanceIndex}: ${triangles} triangles do not fit one merged mesh (${maxTriangles})`);
            group = { key, mask, members: [], triangles: 0, vertices: 0 };
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
        const sourceMaterials = new Uint16Array(pending.triangles);
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
            const material = materialCode(instance.material, instance.instanceIndex);
            // The source triangle's own material wins over its instance's.
            const own = triangleMaterialCodes(mesh.triangleMaterials, triangles, `mesh ${instance.meshIndex} (instance ${instance.instanceIndex})`);
            if (own) {
                for (let triangle = 0; triangle < triangles; triangle++) sourceMaterials[triangleBase + triangle] = own[triangle] !== 0 ? own[triangle] : material;
            } else if (material !== 0) {
                sourceMaterials.fill(material, triangleBase, triangleBase + triangles);
            }
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
                material: material === 0 ? NO_MATERIAL : material - 1,
            });
            vertexBase += vertexCount;
            triangleBase += triangles;
        }
        return {
            geometry: { positions, indices, triangleFlags },
            sourceInstances,
            sourceTriangles,
            flipFrontFace,
            sourceMaterials,
            sources,
            contributesToGi: true,
            visible: true,
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
 * triangle index, the padding word the source instance index, flags bit 1
 * is set for a mirrored source, and flags bits 16-31 get the triangle's
 * material + 1 (`group.sourceMaterials`: its source triangle's, else its
 * source instance's; 0 for none). Bit 0 (the filter flag) is kept; a record with
 * any other flag bit set throws, since builders only write bit 0.
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
        const flags = words[base + TRIANGLE_WORD.flags];
        if ((flags & ~TRIANGLE_FLAG_FILTER) !== 0) {
            throw new RangeError(
                `merged triangle record ${slot} (source instance ${group.sourceInstances[merged]}, triangle ${group.sourceTriangles[merged]}) has flags 0x${flags.toString(16)}: ` +
                    'only bit 0 may be set in fresh records (bits 1-31 are reserved; were the ids already written?)',
            );
        }
        words[base + TRIANGLE_WORD.originalIndex] = group.sourceTriangles[merged];
        words[base + TRIANGLE_WORD.padding] = group.sourceInstances[merged];
        words[base + TRIANGLE_WORD.flags] =
            (flags | (group.flipFrontFace[merged] ? TRIANGLE_FLAG_FLIP_FRONT_FACE : 0) | (group.sourceMaterials[merged] << MATERIAL_SHIFT)) >>> 0;
    }
}

/**
 * The indexed twin of {@link writeMergedTriangleIdentity}, for a merged
 * group's region in an indexed scene's triangle buffer (`words`, 4 words per
 * 16-byte unit; the group's first triangle at unit `triangleOffset`). Each
 * triangle's word 3 gets the source mesh's triangle index, its filter flag
 * kept and the flip flag for a mirrored source. The region before the header
 * (`indexedLeadingUnits(triangleCount, sources)` units before
 * `triangleOffset`) gets the side table (per tree slot: source slot | material
 * code << 16) and the source table (per slot: `instanceIndex`), and the
 * header their distances. The header's vertex base is the caller's.
 *
 * Call it once on fresh indexed triangles (it reads the builder's original
 * index); a word 3 with any bit but the filter flag above the index throws.
 */
export function writeMergedIndexedIdentity(words: Uint32Array, group: StaticMergeGroup, triangleOffset: number, triangleCount = group.sourceInstances.length): void {
    if (triangleCount !== group.sourceInstances.length) {
        throw new RangeError(`the merged tree has ${triangleCount} triangles, its group ${group.sourceInstances.length}`);
    }
    const sourceCount = group.sources.length;
    if (sourceCount > MAX_MERGED_SOURCES) throw new RangeError(`a merged group holds ${sourceCount} sources; an indexed scene allows ${MAX_MERGED_SOURCES} (make the plan with triangleLeaves: 'indexed')`);
    const sideUnits = Math.ceil(triangleCount / 4);
    const sourceUnits = Math.ceil(sourceCount / 4);
    const sideStart = (triangleOffset - 1 - sideUnits) * 4;
    const sourceStart = (triangleOffset - 1 - sideUnits - sourceUnits) * 4;
    if (sourceStart < 0) throw new RangeError(`the merged group at ${triangleOffset} has no room for its side tables`);
    // Source slot of each merged triangle.
    const slotOf = new Uint16Array(triangleCount);
    group.sources.forEach((source, slot) => slotOf.fill(slot, source.firstTriangle, source.firstTriangle + source.triangleCount));
    words.fill(0, sourceStart, (triangleOffset - 1) * 4);
    group.sources.forEach((source, slot) => (words[sourceStart + slot] = source.instanceIndex));
    const header = (triangleOffset - 1) * 4;
    words[header + MESH_HEADER_WORD.sideTableDistance] = triangleOffset - (triangleOffset - 1 - sideUnits);
    words[header + MESH_HEADER_WORD.sourceTableDistance] = triangleOffset - (triangleOffset - 1 - sideUnits - sourceUnits);
    for (let slot = 0; slot < triangleCount; slot++) {
        const word = (triangleOffset + slot) * INDEXED_TRIANGLE_WORDS + 3;
        const idFlags = words[word];
        const merged = idFlags & INDEXED_ID_MASK;
        const flags = idFlags >>> INDEXED_FLAG_SHIFT;
        if (merged >= triangleCount) throw new RangeError(`merged triangle ${slot} holds triangle ${merged}, out of range`);
        if ((flags & ~TRIANGLE_FLAG_FILTER) !== 0) {
            throw new RangeError(`merged indexed triangle ${slot} has flags 0x${flags.toString(16)} above its index: only the filter flag may be set in fresh triangles (were the ids already written?)`);
        }
        const flip = group.flipFrontFace[merged] ? TRIANGLE_FLAG_FLIP_FRONT_FACE : 0;
        words[word] = (group.sourceTriangles[merged] | ((flags | flip) << INDEXED_FLAG_SHIFT)) >>> 0;
        words[sideStart + slot] = (slotOf[merged] | (group.sourceMaterials[merged] << MERGED_SIDE_MATERIAL_SHIFT)) >>> 0;
    }
}

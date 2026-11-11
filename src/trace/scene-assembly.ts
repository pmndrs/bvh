/**
 * Puts finished mesh BVHs and the instance list together into the three
 * buffers the tracer reads: all nodes, all triangles, and the top level.
 *
 * Mesh nodes keep mesh-relative indices, so a mesh can be built once, in a
 * worker, and placed anywhere in the shared buffers. The instance record
 * carries the two offsets the traversal adds.
 *
 * The meshes' node and triangle records are *copied* into the scene's
 * buffers. Changing a `MeshBvh` afterwards (its triangle flags, say) does not
 * reach the scene; write the scene's buffers instead (`meshes[i]` gives each
 * mesh's range).
 */

import type { Mat4 } from './top-level.ts';
import { BVH_NODE_BYTES, TRACE_STACK_SIZE, TRIANGLE_BYTES, TRIANGLE_WORDS, checkInstanceMask, materialCode } from './layout.ts';
import { checkTriangleFlags } from '../mesh-input.ts';
import { countTriangles } from './triangle-packing.ts';
import type { MeshBvh, MeshGeometry } from './mesh-bvh.ts';
import { buildTopLevel, IDENTITY_TRANSFORM, throwMergedTransformChange, type TopLevelBvh, type TraceInstance } from './top-level.ts';
import type { SahBuildOptions } from './sah-builder.ts';
import { MERGED_INSTANCE_INDEX, mergeStaticGeometry, writeMergedTriangleIdentity, type StaticMergeOptions, type StaticMergePlan } from './static-merge.ts';

/** One placed copy of one of the meshes passed to {@link assembleTraceScene}. */
export interface TraceSceneInstance {
    /** Position in the `meshes` array. */
    readonly meshIndex: number;
    /** The platform instance index, returned in `BvhHit.instance`. */
    readonly instanceIndex: number;
    readonly objectToWorld: Mat4;
    readonly contributesToGi?: boolean;
    readonly visible?: boolean;
    /** The instance mask, 0-255; defaults to 0xff, visible to every ray (see `TraceInstance.mask`). */
    readonly mask?: number;
    /**
     * The material index (0-65534), reported as `BvhHit.material`; omit for
     * none (`NO_MATERIAL`). Not a merge-grouping key: a merged instance's
     * material travels in its triangles' flags (see `TraceInstance.material`).
     */
    readonly material?: number;
    /**
     * The instance never moves. With `mergeStaticInstances` on, a static
     * instance whose mesh no other instance uses is merged into a world-space
     * mesh with others of the same flags and mask (static-merge.ts). Ignored
     * otherwise. Default false.
     */
    readonly static?: boolean;
}

/** Where one input mesh landed in the scene's shared buffers. A `TriangleRecordRange` for `writeTriangleFilterFlags`. */
export interface TraceSceneMeshPlacement {
    /** First node, counted in nodes. */
    readonly nodeOffset: number;
    readonly nodeCount: number;
    /** First triangle record, counted in triangles. */
    readonly triangleOffset: number;
    readonly triangleCount: number;
    /**
     * Set on an input mesh that only merged sources use: its triangles live in
     * `TraceSceneData.mergedGroups[mergedInto]`, and this entry is empty
     * (`triangleCount` 0). `writeTriangleFilterFlags` refuses it.
     */
    readonly mergedInto?: number;
}

export interface TraceSceneData {
    /** Every mesh's nodes, one after another. */
    readonly nodes: ArrayBuffer;
    /** Every mesh's triangles, one after another. */
    readonly triangles: ArrayBuffer;
    readonly topLevel: TopLevelBvh;
    /**
     * Where each input mesh landed. The scene holds copies of the meshes'
     * records: to change a mesh's triangle flags after assembly, write
     * `triangles` with the mesh's entry as the range. A mesh only merged
     * sources use has an empty entry with `mergedInto`: flag its triangles on
     * that merged group's range instead, by source instance.
     */
    readonly meshes: readonly TraceSceneMeshPlacement[];
    /** Deepest mesh tree among the meshes the instances use. */
    readonly maxMeshDepth: number;
    /** Stack entries the worst case needs: top-level depth + one marker + mesh depth. */
    readonly requiredStackSize: number;
    /** Bytes the three buffers hold. */
    readonly byteLength: number;
    /**
     * The merged meshes, one per {@link StaticMergePlan} group, after the
     * input meshes in the buffers; empty without `mergeStaticInstances`. With
     * merged groups, `topLevel.recordOrder` gives the input position for a
     * normal instance, and `instances.length + group` for a group's record.
     */
    readonly mergedGroups: readonly TraceSceneMergedGroup[];
}

/**
 * Where a merged group landed, and which instances it holds. A
 * `TriangleRecordRange` for `writeTriangleFilterFlags`, whose callback then
 * gets each record's source instance (`merged: true`). Never a refit target.
 */
export interface TraceSceneMergedGroup extends TraceSceneMeshPlacement {
    readonly merged: true;
    /** The source instances' `instanceIndex`, in the group's order. */
    readonly sourceInstances: readonly number[];
    /** Each source's mesh (its position in the `meshes` array), parallel to {@link sourceInstances}. */
    readonly sourceMeshes: readonly number[];
}

/** The merged meshes for {@link assembleTraceScene}: the plan and one tree per group, from any builder. */
export interface MergedStaticTrees {
    /** From {@link mergeStaticGeometry}, over the same instances (with the same transforms). */
    readonly plan: StaticMergePlan;
    /** `plan.groups[g].geometry` built by any builder (tinybvh, GPU H-PLOC, LBVH): records as the builder wrote them. */
    readonly meshes: readonly MeshBvh[];
}

/** Options of {@link assembleTraceScene}. */
export interface TraceSceneAssemblyOptions {
    /**
     * Merged static meshes. The instances in the plan's groups are not placed
     * as instances; each group gets one identity-transform instance record
     * with `INSTANCE_FLAG.merged`. Omitted (the default): every instance is a
     * normal instance, exactly as without this option.
     */
    readonly merged?: MergedStaticTrees;
}

export function assembleTraceScene(
    meshes: readonly (MeshBvh | null)[],
    instances: readonly TraceSceneInstance[],
    topLevelOptions: Partial<SahBuildOptions> = {},
    options: TraceSceneAssemblyOptions = {},
): TraceSceneData {
    const merged = options.merged;
    if (merged) checkMergedTrees(merged, instances);
    const placed: readonly (MeshBvh | null)[] = merged ? [...meshes, ...merged.meshes] : meshes;
    const mergedInto = new Map<number, number>();
    merged?.plan.groups.forEach((group, groupIndex) => {
        for (const source of group.sources) if (!merged.plan.meshNeeded[source.meshIndex]) mergedInto.set(source.meshIndex, groupIndex);
    });

    let totalNodes = 0;
    let totalTriangles = 0;
    const allPlacements = placed.map((mesh, index): TraceSceneMeshPlacement => {
        const placement: TraceSceneMeshPlacement = {
            nodeOffset: totalNodes,
            nodeCount: mesh?.nodeCount ?? 0,
            triangleOffset: totalTriangles,
            triangleCount: mesh?.triangleCount ?? 0,
            ...(mergedInto.has(index) ? { mergedInto: mergedInto.get(index) } : {}),
        };
        totalNodes += placement.nodeCount;
        totalTriangles += placement.triangleCount;
        return placement;
    });

    const nodes = new ArrayBuffer(Math.max(1, totalNodes) * BVH_NODE_BYTES);
    const triangles = new ArrayBuffer(Math.max(1, totalTriangles) * TRIANGLE_BYTES);
    placed.forEach((mesh, index) => {
        if (!mesh) return;
        new Uint8Array(nodes, allPlacements[index].nodeOffset * BVH_NODE_BYTES, mesh.nodeCount * BVH_NODE_BYTES).set(
            new Uint8Array(mesh.nodes, 0, mesh.nodeCount * BVH_NODE_BYTES),
        );
        new Uint8Array(triangles, allPlacements[index].triangleOffset * TRIANGLE_BYTES, mesh.triangleCount * TRIANGLE_BYTES).set(
            new Uint8Array(mesh.triangles, 0, mesh.triangleCount * TRIANGLE_BYTES),
        );
    });

    let maxMeshDepth = 0;
    const placeInstance = (instance: TraceSceneInstance): TraceInstance => {
        const mesh = meshes[instance.meshIndex];
        if (!mesh) throw new RangeError(`instance ${instance.instanceIndex} uses mesh ${instance.meshIndex}, which does not exist`);
        maxMeshDepth = Math.max(maxMeshDepth, mesh.maxDepth);
        return {
            instanceIndex: instance.instanceIndex,
            nodeOffset: allPlacements[instance.meshIndex].nodeOffset,
            triangleOffset: allPlacements[instance.meshIndex].triangleOffset,
            meshBounds: mesh.bounds,
            objectToWorld: instance.objectToWorld,
            contributesToGi: instance.contributesToGi,
            visible: instance.visible,
            mask: instance.mask,
            material: instance.material,
        };
    };

    let traceInstances: TraceInstance[];
    let inputOf: Uint32Array | null = null;
    const mergedGroups: TraceSceneMergedGroup[] = [];
    if (!merged) {
        traceInstances = instances.map(placeInstance);
    } else {
        const { plan } = merged;
        traceInstances = Array.from(plan.instanced, (input) => placeInstance(instances[input]));
        inputOf = new Uint32Array(plan.instanced.length + plan.groups.length);
        inputOf.set(plan.instanced);
        plan.groups.forEach((group, groupIndex) => {
            const tree = merged.meshes[groupIndex];
            const placement = allPlacements[meshes.length + groupIndex];
            // The ids go into the scene's copy, so the tree itself stays as its builder wrote it.
            writeMergedTriangleIdentity(triangles, group, placement.triangleOffset * TRIANGLE_BYTES, tree.triangleCount);
            maxMeshDepth = Math.max(maxMeshDepth, tree.maxDepth);
            traceInstances.push({
                instanceIndex: MERGED_INSTANCE_INDEX,
                nodeOffset: placement.nodeOffset,
                triangleOffset: placement.triangleOffset,
                meshBounds: tree.bounds,
                objectToWorld: IDENTITY_TRANSFORM,
                contributesToGi: group.contributesToGi,
                visible: group.visible,
                mask: group.mask,
                merged: true,
            });
            inputOf![plan.instanced.length + groupIndex] = instances.length + groupIndex;
            mergedGroups.push({
                ...placement,
                merged: true,
                sourceInstances: group.sources.map((source) => source.instanceIndex),
                sourceMeshes: group.sources.map((source) => source.meshIndex),
            });
        });
    }
    const topLevel = buildTopLevel(traceInstances, topLevelOptions);
    if (inputOf) {
        // Record order in terms of the caller's instances (see TraceSceneData.mergedGroups).
        const order = topLevel.recordOrder;
        for (let record = 0; record < order.length; record++) order[record] = inputOf[order[record]];
    }

    const requiredStackSize = topLevel.maxDepth + 1 + maxMeshDepth;
    if (requiredStackSize > TRACE_STACK_SIZE) {
        throw new RangeError(
            `the trace stack holds ${TRACE_STACK_SIZE} entries but this scene can need ${requiredStackSize} ` +
                `(top level ${topLevel.maxDepth} + marker + deepest mesh ${maxMeshDepth})`,
        );
    }

    return {
        nodes,
        triangles,
        topLevel,
        meshes: allPlacements.slice(0, meshes.length),
        maxMeshDepth,
        requiredStackSize,
        byteLength: nodes.byteLength + triangles.byteLength + topLevel.buffer.byteLength,
        mergedGroups,
    };
}

/** The plan must be for these instances, unmoved, with the same mask, flags and material, and each tree for its group. */
function checkMergedTrees(merged: MergedStaticTrees, instances: readonly TraceSceneInstance[]): void {
    const { plan } = merged;
    if (plan.instanceCount !== instances.length) {
        throw new RangeError(`the merge plan was made for ${plan.instanceCount} instances, the scene has ${instances.length}`);
    }
    if (merged.meshes.length !== plan.groups.length) {
        throw new RangeError(`the merge plan has ${plan.groups.length} groups but ${merged.meshes.length} merged trees were given`);
    }
    plan.groups.forEach((group, groupIndex) => {
        const tree = merged.meshes[groupIndex];
        if (tree.triangleCount !== group.sourceInstances.length) {
            throw new RangeError(`merged tree ${groupIndex} has ${tree.triangleCount} triangles, its group ${group.sourceInstances.length}`);
        }
        for (const source of group.sources) {
            const instance = instances[source.inputIndex];
            if (instance.instanceIndex !== source.instanceIndex || instance.meshIndex !== source.meshIndex || instance.static !== true) {
                throw new RangeError(`instance ${instance.instanceIndex} at position ${source.inputIndex} is not the one the merge plan merged (instance ${source.instanceIndex})`);
            }
            for (let k = 0; k < 16; k++) {
                if (instance.objectToWorld[k] !== source.objectToWorld[k]) throwMergedTransformChange(source.instanceIndex, 'assembleTraceScene');
            }
            const mask = checkInstanceMask(instance.mask, instance.instanceIndex);
            const contributesToGi = instance.contributesToGi !== false;
            const visible = instance.visible !== false;
            if (mask !== group.mask || contributesToGi !== group.contributesToGi || visible !== group.visible) {
                throw new RangeError(
                    `instance ${instance.instanceIndex} is a merged static instance, but its mask, visible or contributesToGi changed since the merge plan ` +
                        `(mask ${mask}, visible ${visible}, contributesToGi ${contributesToGi}; the plan has ${group.mask}, ${group.visible}, ${group.contributesToGi}): ` +
                        'make a new plan with mergeStaticGeometry, or leave it out of the merge (static: false)',
                );
            }
            // The plan holds the material its triangles get; a changed one needs a new plan.
            if (materialCode(instance.material, instance.instanceIndex) !== materialCode(source.material, source.instanceIndex)) {
                throw new RangeError(
                    `instance ${instance.instanceIndex} is a merged static instance, but its material changed since the merge plan (${describeMaterial(instance.material)}; the plan has ${describeMaterial(source.material)}): ` +
                        'make a new plan with mergeStaticGeometry',
                );
            }
        }
    });
}

/** Options of {@link buildTraceScene}. */
export interface TraceSceneBuildOptions extends StaticMergeOptions {
    /** Builds one mesh's tree: tinybvh (`buildMeshBvhWithTinyBvh`), a worker builder's `build`, or a GPU builder's layout. */
    readonly build: (geometry: MeshGeometry) => MeshBvh | Promise<MeshBvh>;
    /**
     * Merge static instances whose mesh is used once into world-space meshes
     * (static-merge.ts; docs/design/trace.md). Default false: every instance
     * is a normal instance, and the scene is exactly what building each mesh
     * and calling `assembleTraceScene` gives.
     */
    readonly mergeStaticInstances?: boolean;
    readonly topLevel?: Partial<SahBuildOptions>;
}

/**
 * Builds every mesh tree with `options.build` and assembles the scene. With
 * `mergeStaticInstances`, meshes only merged instances use get no tree of
 * their own, and each merged group's geometry is built instead.
 *
 * Every mesh's `triangleFlags` may set bit 0 only; a reserved bit throws,
 * naming the mesh and triangle.
 *
 * **Static means static.** A merged instance's transform is baked into the
 * merged mesh. Only assembling again notices that one moved (or that its
 * mask, `visible` or `contributesToGi` changed); nothing checks between
 * assemblies, so a caller that moves a merged instance and only updates the
 * top level gets the old position. A merged group's range
 * (`TraceSceneData.mergedGroups`) must never be refit: use
 * {@link meshRefitPlacement}, which refuses merged meshes, to get refit offsets.
 */
export async function buildTraceScene(
    geometries: readonly MeshGeometry[],
    instances: readonly TraceSceneInstance[],
    options: TraceSceneBuildOptions,
): Promise<TraceSceneData> {
    geometries.forEach((geometry, index) => checkTriangleFlags(geometry.triangleFlags, countTriangles(geometry), `mesh ${index}`));
    if (!options.mergeStaticInstances) {
        const meshes = await Promise.all(geometries.map((geometry) => options.build(geometry)));
        return assembleTraceScene(meshes, instances, options.topLevel);
    }
    const plan = mergeStaticGeometry(geometries, instances, options);
    const meshes = await Promise.all(geometries.map((geometry, index) => (plan.meshNeeded[index] ? options.build(geometry) : null)));
    const mergedMeshes = await Promise.all(plan.groups.map((group) => options.build(group.geometry)));
    return assembleTraceScene(meshes, instances, options.topLevel, { merged: { plan, meshes: mergedMeshes } });
}

/**
 * The word offsets a `GpuMeshRefit.setTree` needs for input mesh `meshIndex`
 * of an assembled scene (a deforming mesh refit in the scene's shared
 * buffers). Throws for a mesh merged into a static group: merged records are
 * world space and name source-mesh triangles, so a refit from the source's
 * object-space vertices would corrupt them. Merged ranges are never refit.
 */
export function meshRefitPlacement(scene: Pick<TraceSceneData, 'meshes'>, meshIndex: number): { nodeWordOffset: number; triangleWordOffset: number } {
    const placement = scene.meshes[meshIndex];
    if (!placement) throw new RangeError(`meshRefitPlacement: the scene has no mesh ${meshIndex}`);
    if (placement.mergedInto !== undefined) {
        throw new RangeError(
            `meshRefitPlacement: mesh ${meshIndex} was merged into static group ${placement.mergedInto} (TraceSceneData.mergedGroups); merged meshes never move and must not be refit. ` +
                'To deform it, leave its instance out of the merge (static: false)',
        );
    }
    if (placement.triangleCount === 0) throw new RangeError(`meshRefitPlacement: mesh ${meshIndex} has no tree in this scene`);
    return { nodeWordOffset: placement.nodeOffset * (BVH_NODE_BYTES / 4), triangleWordOffset: placement.triangleOffset * TRIANGLE_WORDS };
}

/** A material for an error message: `none` for undefined or NO_MATERIAL (0xffffffff). */
function describeMaterial(material: number | undefined): string {
    return material === undefined || material >>> 0 === 0xffffffff ? 'none' : String(material);
}

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
import {
    INDEXED_TRIANGLE_BYTES,
    INDEXED_TRIANGLE_WORDS,
    MESH_HEADER_WORD,
    TRACE_STACK_SIZE,
    TRIANGLE_BYTES,
    TRIANGLE_WORDS,
    checkInstanceMask,
    indexedLeadingUnits,
    materialCode,
    positionStrideFloats,
    type PositionLayout,
    type TriangleLeaves,
} from './layout.ts';
import { checkNodeFormat, quantizeNodes } from '../quantized-nodes.ts';
import { gpuLayoutNodeBytes, gpuLayoutNodeSlots, type GpuLayoutNodeFormat } from '../gpu-layout.ts';
import { checkTriangleLeaves, indexedFromRecords, weldExactBits, weldPositions } from '../indexed-leaves.ts';
import { checkTriangleFlags } from '../mesh-input.ts';
import { countTriangles } from './triangle-packing.ts';
import { markIndexedSceneTriangles } from './indexed-scene-buffers.ts';
import type { MeshBvh, MeshGeometry } from './mesh-bvh.ts';
import { buildTopLevel, IDENTITY_TRANSFORM, throwMergedTransformChange, type TopLevelBvh, type TraceInstance } from './top-level.ts';
import type { SahBuildOptions } from './sah-builder.ts';
import { MERGED_INSTANCE_INDEX, mergeStaticGeometry, writeMergedIndexedIdentity, writeMergedTriangleIdentity,type StaticMergeOptions, type StaticMergePlan } from './static-merge.ts';

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
    /**
     * The mesh's root, counted in nodes (in the scene's node size). A
     * quantized scene keeps the mesh's frame slot right before it, at `nodeOffset - 1`.
     */
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
    /** Indexed scenes: the mesh's first vertex in `TraceSceneData.positions` (its header's vertex base). */
    readonly vertexBase?: number;
    /** Indexed scenes: the mesh's vertices in `TraceSceneData.positions`. */
    readonly vertexCount?: number;
    /**
     * Indexed scenes: the mesh's vertices were welded (the exact-bit weld, on
     * by default, or `weldTolerance`), so the scene's vertex numbering is not
     * the app's and `meshRefitPlacement` refuses the mesh.
     */
    readonly welded?: boolean;
}

export interface TraceSceneData {
    /** Every mesh's nodes, one after another (quantized: each mesh's frame slot, then its nodes). */
    readonly nodes: ArrayBuffer;
    /** How `nodes` stores them. Default (absent) `'f32'`; assembly sets it, `'quantized'` when assembled with `nodeFormat: 'quantized'`. */
    readonly nodeFormat?: GpuLayoutNodeFormat;
    /**
     * Every mesh's triangles, one after another. Indexed scenes: 16-byte
     * units, and each mesh's region starts with its header (and a merged
     * mesh's side tables) before `triangleOffset`.
     */
    readonly triangles: ArrayBuffer;
    /** How `triangles` stores them. `'records'` unless assembled with `triangleLeaves: 'indexed'`. */
    readonly triangleLeaves: TriangleLeaves;
    /**
     * Indexed scenes: every mesh's positions (welded unless `weld: false`),
     * one after another at each mesh's vertex base, in {@link positionLayout}.
     * Empty for records.
     */
    readonly positions: ArrayBuffer;
    /** How {@link positions} stores a vertex: `'vec4'` (16 bytes) or `'packed'` (12). */
    readonly positionLayout: PositionLayout;
    /** Vertices in {@link positions}. 0 for records. */
    readonly vertexCount: number;
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
    /** Bytes the three buffers hold (four with an indexed scene's positions). */
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
    /**
     * The scene's triangle format (docs/design/indexed-leaves.md). Default
     * `'records'`: the scene is byte for byte what it was before the option
     * existed. `'indexed'` needs {@link geometries}; trees built as records
     * are converted (`indexedFromRecords`), indexed trees are placed as they are.
     */
    readonly triangleLeaves?: TriangleLeaves;
    /** Indexed scenes: `'vec4'` (the default) or `'packed'` positions. */
    readonly positionLayout?: PositionLayout;
    /**
     * Indexed scenes: each mesh's geometry, parallel to `meshes` (null where
     * the mesh is null), whose positions go into the scene's positions buffer.
     * Merged groups take theirs from the plan.
     */
    readonly geometries?: readonly (MeshGeometry | null)[];
    /**
     * Indexed scenes: weld each mesh's vertices by exact bits as they are
     * copied (`weldExactBits`). It changes no coordinate, so hits are the
     * same, and saves memory. Default true.
     */
    readonly weld?: boolean;
    /**
     * The scene's node format (docs/design/compressed-nodes.md). Default
     * `'f32'`: the scene is byte for byte what it was before the option
     * existed. `'quantized'`: 32-byte nodes, each mesh's frame slot before its
     * root; `f32` trees are quantized (`quantizeNodes`), quantized trees are
     * placed as they are. Independent of {@link triangleLeaves}.
     */
    readonly nodeFormat?: GpuLayoutNodeFormat;
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

    const leaves = checkTriangleLeaves(options.triangleLeaves, 'assembleTraceScene');
    const indexed = leaves === 'indexed';
    const positionLayout: PositionLayout = options.positionLayout ?? 'vec4';
    if (positionLayout !== 'vec4' && positionLayout !== 'packed') throw new RangeError(`assembleTraceScene: positionLayout must be 'vec4' or 'packed', got ${String(positionLayout)}`);
    const geometryOf = (index: number): MeshGeometry => {
        if (index >= meshes.length) return merged!.plan.groups[index - meshes.length].geometry;
        const geometry = options.geometries?.[index];
        if (!geometry) throw new RangeError(`assembleTraceScene: an indexed scene needs geometries[${index}], the positions of mesh ${index}`);
        return geometry;
    };
    if (!indexed && (options.geometries || options.weld !== undefined || options.positionLayout !== undefined)) {
        throw new RangeError("assembleTraceScene: geometries, weld and positionLayout are for triangleLeaves: 'indexed'");
    }

    const nodeFormat = checkNodeFormat(options.nodeFormat, 'assembleTraceScene');
    const quantized = nodeFormat === 'quantized';
    const nodeBytes = gpuLayoutNodeBytes(nodeFormat);
    let totalNodes = 0;
    let totalTriangles = 0;
    const allPlacements = placed.map((mesh, index): TraceSceneMeshPlacement => {
        const sources = index >= meshes.length ? merged!.plan.groups[index - meshes.length].sources.length : 0;
        const leading = indexed && mesh ? indexedLeadingUnits(mesh.triangleCount, sources) : 0;
        // A quantized mesh's frame slot comes right before its root.
        if (quantized && mesh) totalNodes++;
        const placement: TraceSceneMeshPlacement = {
            nodeOffset: totalNodes,
            nodeCount: mesh?.nodeCount ?? 0,
            triangleOffset: totalTriangles + leading,
            triangleCount: mesh?.triangleCount ?? 0,
            ...(mergedInto.has(index) ? { mergedInto: mergedInto.get(index) } : {}),
        };
        totalNodes += placement.nodeCount;
        totalTriangles += leading + placement.triangleCount;
        return placement;
    });

    const nodes = new ArrayBuffer(Math.max(1, totalNodes) * nodeBytes);
    const triangles = new ArrayBuffer(Math.max(1, totalTriangles) * (indexed ? INDEXED_TRIANGLE_BYTES : TRIANGLE_BYTES));
    placed.forEach((mesh, index) => {
        if (!mesh) return;
        if (!quantized && mesh.nodeFormat === 'quantized') throw new RangeError(`assembleTraceScene: mesh ${index} has quantized nodes; assemble with nodeFormat: 'quantized'`);
        // Quantized: the frame slot and the nodes, from one before the root.
        const tree = quantized ? quantizeNodes(mesh) : mesh;
        const slots = gpuLayoutNodeSlots(tree.nodeCount, nodeFormat);
        new Uint8Array(nodes, (allPlacements[index].nodeOffset - (quantized ? 1 : 0)) * nodeBytes, slots * nodeBytes).set(new Uint8Array(tree.nodes, 0, slots * nodeBytes));
        if (indexed) return;
        if (mesh.triangleLeaves === 'indexed') throw new RangeError(`assembleTraceScene: mesh ${index} is indexed; assemble with triangleLeaves: 'indexed'`);
        new Uint8Array(triangles, allPlacements[index].triangleOffset * TRIANGLE_BYTES, mesh.triangleCount * TRIANGLE_BYTES).set(
            new Uint8Array(mesh.triangles, 0, mesh.triangleCount * TRIANGLE_BYTES),
        );
    });
    const placedPositions = indexed ? placeIndexedTriangles(placed, allPlacements, geometryOf, triangles, positionLayout, options.weld ?? true) : null;
    if (placedPositions) {
        const welded = options.weld ?? true;
        allPlacements.forEach((placement, index) => {
            allPlacements[index] = { ...placement, vertexBase: placedPositions.bases[index], vertexCount: placedPositions.counts[index], welded };
        });
    }

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
            if (indexed) writeMergedIndexedIdentity(new Uint32Array(triangles), group, placement.triangleOffset, tree.triangleCount);
            else writeMergedTriangleIdentity(triangles, group, placement.triangleOffset * TRIANGLE_BYTES, tree.triangleCount);
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

    const positions = placedPositions?.positions ?? new ArrayBuffer(0);
    if (indexed) markIndexedSceneTriangles(triangles);
    return {
        nodes,
        nodeFormat,
        triangles,
        triangleLeaves: leaves,
        positions,
        positionLayout,
        vertexCount: placedPositions?.vertexCount ?? 0,
        topLevel,
        meshes: allPlacements.slice(0, meshes.length),
        maxMeshDepth,
        requiredStackSize,
        byteLength: nodes.byteLength + triangles.byteLength + topLevel.buffer.byteLength + positions.byteLength,
        mergedGroups,
    };
}

/**
 * An indexed scene's triangles and positions: each placed tree's indexed
 * triangles (converted from records if need be) with its vertex indices
 * through the mesh's weld, its header's vertex base, and its positions at
 * that base. Merged identity is written afterwards.
 */
function placeIndexedTriangles(
    placed: readonly (MeshBvh | null)[],
    placements: readonly TraceSceneMeshPlacement[],
    geometryOf: (index: number) => MeshGeometry,
    triangles: ArrayBuffer,
    positionLayout: PositionLayout,
    weld: boolean,
): { positions: ArrayBuffer; vertexCount: number; bases: number[]; counts: number[] } {
    const words = new Uint32Array(triangles);
    const welds = placed.map((mesh, index) => {
        if (!mesh) return null;
        const geometry = geometryOf(index);
        if (weld) return weldExactBits(geometry.positions);
        const count = geometry.positions.length / 3;
        return { positions: geometry.positions, remap: Uint32Array.from({ length: count }, (_, k) => k), vertexCount: count };
    });
    let vertexCount = 0;
    const bases = welds.map((result) => {
        const base = vertexCount;
        vertexCount += result?.vertexCount ?? 0;
        return base;
    });
    const stride = positionStrideFloats(positionLayout);
    const positions = new ArrayBuffer(Math.max(1, vertexCount) * stride * 4);
    const bits = new Uint32Array(positions);
    placed.forEach((mesh, index) => {
        const result = welds[index];
        if (!mesh || !result) return;
        const tree = indexedFromRecords(mesh, geometryOf(index));
        const source = new Uint32Array(tree.triangles, 0, tree.triangleCount * INDEXED_TRIANGLE_WORDS);
        const offset = placements[index].triangleOffset;
        const { remap } = result;
        for (let slot = 0; slot < tree.triangleCount; slot++) {
            const from = slot * INDEXED_TRIANGLE_WORDS;
            const to = (offset + slot) * INDEXED_TRIANGLE_WORDS;
            for (let corner = 0; corner < 3; corner++) {
                const vertex = source[from + corner];
                if (vertex >= remap.length) throw new RangeError(`assembleTraceScene: mesh ${index} triangle slot ${slot} uses vertex ${vertex}, its geometry has ${remap.length}`);
                words[to + corner] = remap[vertex];
            }
            words[to + 3] = source[from + 3];
        }
        words[(offset - 1) * 4 + MESH_HEADER_WORD.vertexBase] = bases[index];
        // Bits, not values: the walk must see exactly the bits the builders' boxes came from.
        const kept = new Uint32Array(result.positions.buffer, result.positions.byteOffset, result.vertexCount * 3);
        for (let vertex = 0; vertex < result.vertexCount; vertex++) {
            const at = (bases[index] + vertex) * stride;
            bits[at] = kept[vertex * 3];
            bits[at + 1] = kept[vertex * 3 + 1];
            bits[at + 2] = kept[vertex * 3 + 2];
        }
    });
    return { positions, vertexCount, bases, counts: welds.map((result) => result?.vertexCount ?? 0) };
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
    /** The scene's triangle format (see `TraceSceneAssemblyOptions.triangleLeaves`). Default `'records'`. */
    readonly triangleLeaves?: TriangleLeaves;
    /** Indexed scenes: `'vec4'` (the default) or `'packed'` positions. */
    readonly positionLayout?: PositionLayout;
    /** Indexed scenes: the exact-bit weld at assembly. Default true. */
    readonly weld?: boolean;
    /** The scene's node format (see `TraceSceneAssemblyOptions.nodeFormat`). Default `'f32'`; trees are built `f32` and quantized at assembly. */
    readonly nodeFormat?: GpuLayoutNodeFormat;
    /**
     * Opt-in: weld each mesh's vertices closer than this on every axis before
     * building (`weldPositions`), in either format. It moves geometry away
     * from what the app draws by up to this much, and closes seams whose
     * copies differ by an ulp or so (which leak otherwise, in both formats).
     * Within one mesh only. Default 0: off.
     */
    readonly weldTolerance?: number;
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
    inputs: readonly MeshGeometry[],
    instances: readonly TraceSceneInstance[],
    options: TraceSceneBuildOptions,
): Promise<TraceSceneData> {
    inputs.forEach((geometry, index) => checkTriangleFlags(geometry.triangleFlags, countTriangles(geometry), `mesh ${index}`));
    const tolerance = options.weldTolerance ?? 0;
    const geometries = tolerance > 0 ? inputs.map((geometry) => weldPositions(geometry, { tolerance })) : inputs;
    const indexed = checkTriangleLeaves(options.triangleLeaves, 'buildTraceScene') === 'indexed';
    const nodeOptions: TraceSceneAssemblyOptions = checkNodeFormat(options.nodeFormat, 'buildTraceScene') === 'quantized' ? { nodeFormat: 'quantized' } : {};
    const layoutOptions: TraceSceneAssemblyOptions = indexed
        ? { triangleLeaves: 'indexed', geometries, positionLayout: options.positionLayout, weld: options.weld, ...nodeOptions }
        : nodeOptions;
    // A tolerance weld renumbers vertices too: the scene's are not the app's, so no refit (meshRefitPlacement).
    const markWelded = (scene: TraceSceneData): TraceSceneData =>
        indexed && tolerance > 0 ? { ...scene, meshes: scene.meshes.map((placement) => ({ ...placement, welded: true })) } : scene;
    if (!options.mergeStaticInstances) {
        const meshes = await Promise.all(geometries.map((geometry) => options.build(geometry)));
        return markWelded(assembleTraceScene(meshes, instances, options.topLevel, layoutOptions));
    }
    const plan = mergeStaticGeometry(geometries, instances, options);
    const meshes = await Promise.all(geometries.map((geometry, index) => (plan.meshNeeded[index] ? options.build(geometry) : null)));
    const mergedMeshes = await Promise.all(plan.groups.map((group) => options.build(group.geometry)));
    return markWelded(assembleTraceScene(meshes, instances, options.topLevel, { ...layoutOptions, merged: { plan, meshes: mergedMeshes } }));
}

/** What `GpuMeshRefit.setTree` needs for a scene mesh (from {@link meshRefitPlacement}). */
export interface MeshRefitPlacement {
    /** The mesh's root, in words (quantized scenes: its frame slot is the 8 words before). */
    readonly nodeWordOffset: number;
    /** The scene's node format: `GpuMeshRefit` refits an `f32` copy and packs it into a quantized scene. */
    readonly nodeFormat?: GpuLayoutNodeFormat;
    readonly triangleWordOffset: number;
    /** Indexed scenes: the mesh's first vertex in the scene's positions. */
    readonly vertexBase?: number;
    /** Indexed scenes: the mesh's vertices there; the posed buffer must have exactly this many. */
    readonly vertexCount?: number;
    /** Indexed scenes: the positions buffer's layout. */
    readonly positionLayout?: PositionLayout;
}

/**
 * What a `GpuMeshRefit.setTree` needs for input mesh `meshIndex` of an
 * assembled scene (a deforming mesh refit in the scene's shared buffers): the
 * word offsets and, in an indexed scene, the mesh's vertex base, vertex count
 * and the positions layout. Throws for a mesh merged into a static group:
 * merged records are world space and name source-mesh triangles, so a refit
 * from the source's object-space vertices would corrupt them. Merged ranges
 * are never refit. Throws for a welded mesh of an indexed scene too: the
 * refit copies the app's vertex v to scene vertex `vertexBase + v`, which only
 * holds when the scene kept the app's numbering. Assemble meshes you will
 * refit with `weld: false` (and no `weldTolerance`).
 */
export function meshRefitPlacement(
    scene: Pick<TraceSceneData, 'meshes'> & Partial<Pick<TraceSceneData, 'triangleLeaves' | 'positionLayout' | 'nodeFormat'>>,
    meshIndex: number,
): MeshRefitPlacement {
    const placement = scene.meshes[meshIndex];
    if (!placement) throw new RangeError(`meshRefitPlacement: the scene has no mesh ${meshIndex}`);
    if (placement.mergedInto !== undefined) {
        throw new RangeError(
            `meshRefitPlacement: mesh ${meshIndex} was merged into static group ${placement.mergedInto} (TraceSceneData.mergedGroups); merged meshes never move and must not be refit. ` +
                'To deform it, leave its instance out of the merge (static: false)',
        );
    }
    if (placement.triangleCount === 0) throw new RangeError(`meshRefitPlacement: mesh ${meshIndex} has no tree in this scene`);
    const nodeWordOffset = placement.nodeOffset * (gpuLayoutNodeBytes(scene.nodeFormat) / 4);
    const nodes = scene.nodeFormat === 'quantized' ? { nodeFormat: 'quantized' as const } : {};
    if (scene.triangleLeaves !== 'indexed') return { nodeWordOffset, ...nodes, triangleWordOffset: placement.triangleOffset * TRIANGLE_WORDS };
    if (placement.welded) {
        throw new RangeError(
            `meshRefitPlacement: mesh ${meshIndex} of this indexed scene was welded, so its vertex numbering is not the app's and a refit from the app's vertices would write the wrong vertices. ` +
                'Assemble meshes you will refit with weld: false (and no weldTolerance)',
        );
    }
    if (placement.vertexBase === undefined || placement.vertexCount === undefined) {
        throw new RangeError(`meshRefitPlacement: mesh ${meshIndex} of this indexed scene has no vertexBase or vertexCount`);
    }
    return {
        nodeWordOffset,
        ...nodes,
        triangleWordOffset: placement.triangleOffset * INDEXED_TRIANGLE_WORDS,
        vertexBase: placement.vertexBase,
        vertexCount: placement.vertexCount,
        positionLayout: scene.positionLayout ?? 'vec4',
    };
}

/** A material for an error message: `none` for undefined or NO_MATERIAL (0xffffffff). */
function describeMaterial(material: number | undefined): string {
    return material === undefined || material >>> 0 === 0xffffffff ? 'none' : String(material);
}

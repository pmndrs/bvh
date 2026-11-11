/**
 * Skinned and morphed meshes on the GPU (the plan's `three/skinned` path):
 * {@link BvhPosePass} is a TSL compute pass that morphs and skins a mesh's
 * vertices into a storage buffer of our own, which `SceneTrees.setPose` then
 * refits the mesh's tree from (`GpuMeshRefit`). Nothing comes back to the CPU.
 * `SceneTreesObject3DSync` makes one per `SkinnedMesh` and per mesh with
 * position morph targets, and runs it each `update()`.
 *
 * **three's math, in three's order.** The pass is three r187's own vertex
 * stage, as a compute shader:
 *
 * - morph (`Morph.js`, `morphReference`): `p *= base` (1 with
 *   `morphTargetsRelative`, else `1 - sum(influences)`), then for each target
 *   whose influence isn't 0, `p += target * influence`;
 * - skinning (`Skinning.js`, `getSkinnedPosition`): `v = bindMatrix * vec4(p, 1)`,
 *   `bindMatrixInverse * (boneX * wX * v + boneY * wY * v + boneZ * wZ * v + boneW * wW * v)`,
 *   with `skeleton.boneMatrices` (`skeleton.update()` is called first, as
 *   three's skinning node calls it each frame).
 *
 * Morph before skinning, as three's `setupPosition`. A mesh's per-instance
 * morphs (`InstancedMesh.morphTexture`) are not read: every instance gets
 * `morphTargetInfluences`.
 *
 * **Inputs** are read through the attributes once, at construction
 * (interleaved, normalized and non-float attributes work) and packed into
 * storage buffers: positions and morph targets as `vec4`, skin indices as
 * `uvec4`, weights as `vec4`. Bone matrices (with the bind matrices) and
 * influences are uploaded each {@link BvhPosePass.update}. A geometry whose
 * rest positions, skin attributes or morph targets change needs a new pass
 * (the sync makes one when the geometry object changes).
 *
 * **Compile cost.** One shader per variant (skinned, morphed, both): vertex
 * and target counts are uniforms, the morph loop is a `Loop` over a uniform
 * bound, and the storage arrays are runtime-sized, so every mesh of a
 * variant shares one pipeline and the WGSL text never changes between loads.
 * {@link BvhPosePass.prepare} compiles it with `compileComputeAsync`.
 *
 * **The deform box.** `SceneTrees` keeps one box per deforming mesh in its
 * top level, the box every pose stays in. {@link BvhPosePass.bounds} gives a
 * box that holds the current pose without reading it back: per bone, the box
 * of the (morphed) bind-space vertices it moves, carried through its bone
 * matrix; a skinned vertex is a weighted mean of those, so it lies in their
 * union (`skinWeight`s must not be negative: with a negative weight the box
 * is computed from every vertex on the CPU instead, through three's
 * `getVertexPosition`). Weights that don't sum to 1 scale the result, and the
 * box is scaled to match. Cost per update: 16 corner transforms per bone,
 * plus 6 floats per bone and morph target.
 */

import { Box3, Vector3, type BufferAttribute, type BufferGeometry, type InterleavedBufferAttribute, type Mesh, type SkinnedMesh } from 'three';
import { Fn, If, Loop, add, float, instanceIndex, mat4, storage, uint, uniform, vec4 } from 'three/tsl';
import { StorageBufferAttribute, type ComputeNode, type Node } from 'three/webgpu';
import type { MeshRefitVertices } from '../gpu-mesh-refit.ts';
import { checkStandIn, storageStandIn, type BvhTslNode } from './trace-nodes.ts';

/** Whether three deforms the mesh when it draws it: a `SkinnedMesh` with a skeleton, or position morph targets with influences. */
export function threeDeforms(mesh: Mesh): boolean {
    const skinned = mesh as unknown as SkinnedMesh;
    if (skinned.isSkinnedMesh && skinned.skeleton && mesh.geometry.getAttribute('skinIndex') && mesh.geometry.getAttribute('skinWeight')) return true;
    return morphTargetCount(mesh) > 0;
}

function morphTargetCount(mesh: Mesh): number {
    const targets = mesh.geometry.morphAttributes.position;
    return targets && targets.length > 0 && mesh.morphTargetInfluences ? targets.length : 0;
}

/** What the sync needs of a pose pass (a {@link BvhPosePass}; tests use a stand-in). */
export interface BvhPosePassLike {
    /** Posed vertices, one `vec4` (x, y, z, 1) per vertex of the geometry, `STORAGE | COPY_SRC`. */
    readonly buffer: GPUBuffer;
    readonly vertexCount: number;
    /** Reads the skeleton and influences, uploads them, and dispatches the pass (submitted at once, by three). */
    update(): void;
    /** A box (object space) that holds the pose of the last {@link update}. */
    bounds(target?: Float32Array): Float32Array;
    dispose(): void;
}

export interface BvhPosePassOptions {
    /** Compute workgroup size. Default 64. */
    readonly workgroupSize?: number;
}

type AnyAttribute = BufferAttribute | InterleavedBufferAttribute;

const scratchVector = new Vector3();

/** `out = box carried through the affine column-major matrix m`: the box of its 8 corners' images. */
function transformBox(m: ArrayLike<number>, offset: number, box: ArrayLike<number>, out: Float64Array): void {
    for (let row = 0; row < 3; row++) {
        let lo = m[offset + 12 + row];
        let hi = lo;
        for (let column = 0; column < 3; column++) {
            const a = m[offset + column * 4 + row] * box[column];
            const b = m[offset + column * 4 + row] * box[column + 3];
            lo += Math.min(a, b);
            hi += Math.max(a, b);
        }
        out[row] = lo;
        out[row + 3] = hi;
    }
}

function emptyBox(out: Float64Array): void {
    out.fill(Infinity, 0, 3);
    out.fill(-Infinity, 3, 6);
}

function growBox(out: Float64Array, box: ArrayLike<number>): void {
    for (let k = 0; k < 3; k++) {
        if (box[k] < out[k]) out[k] = box[k];
        if (box[k + 3] > out[k + 3]) out[k + 3] = box[k + 3];
    }
}

/** A morph or skin pass for one mesh; see the file comment. */
export class BvhPosePass implements BvhPosePassLike {
    readonly buffer: GPUBuffer;
    readonly vertexCount: number;
    readonly skinned: boolean;
    readonly morphTargets: number;
    /** The compute node (one dispatch over the vertices). */
    readonly node: ComputeNode;

    private readonly renderer: { compute(node: ComputeNode): unknown; compileComputeAsync?(node: ComputeNode): Promise<unknown> };
    private readonly output: StorageBufferAttribute;
    private readonly matrices: StorageBufferAttribute | null;
    private readonly influences: StorageBufferAttribute | null;
    /** Per group (bone, or the one group of a morph-only mesh): the box of its vertices' rest positions, then of each target's values. */
    private readonly groupBoxes: Float64Array;
    private readonly groupUsed: Uint8Array;
    /** The range of the vertices' weight sums. */
    private readonly weightSum: readonly [number, number];
    /** A negative weight: {@link bounds} reads every vertex instead. */
    private readonly negativeWeights: boolean;
    private disposed = false;

    constructor(
        renderer: unknown,
        readonly mesh: Mesh,
        options: BvhPosePassOptions = {},
    ) {
        const device = (renderer as { backend?: { device?: GPUDevice | null } }).backend?.device;
        if (!device) throw new Error('@pmndrs/bvh/three: BvhPosePass needs an initialized WebGPURenderer (await renderer.init())');
        this.renderer = renderer as BvhPosePass['renderer'];
        const geometry = mesh.geometry;
        const position = geometry.getAttribute('position') as AnyAttribute;
        const count = position.count;
        this.vertexCount = count;
        const skinnedMesh = mesh as unknown as SkinnedMesh;
        this.skinned = skinnedMesh.isSkinnedMesh === true && !!skinnedMesh.skeleton && !!geometry.getAttribute('skinIndex') && !!geometry.getAttribute('skinWeight');
        this.morphTargets = morphTargetCount(mesh);
        if (!this.skinned && this.morphTargets === 0) throw new Error('@pmndrs/bvh/three: BvhPosePass needs a SkinnedMesh or position morph targets');

        // Inputs, read through the attributes.
        const rest = new Float32Array(count * 4);
        for (let i = 0; i < count; i++) rest.set([position.getX(i), position.getY(i), position.getZ(i), 1], i * 4);
        const targets = this.morphTargets > 0 ? new Float32Array(count * 4 * this.morphTargets) : null;
        if (targets) {
            geometry.morphAttributes.position!.forEach((target, t) => {
                for (let i = 0; i < count; i++) targets.set([target.getX(i), target.getY(i), target.getZ(i), 0], (t * count + i) * 4);
            });
        }
        const bones = this.skinned ? skinnedMesh.skeleton.bones.length : 0;
        let skinIndex: Uint32Array | null = null;
        let skinWeight: Float32Array | null = null;
        let low = Infinity;
        let high = -Infinity;
        let negative = false;
        if (this.skinned) {
            const indexAttribute = geometry.getAttribute('skinIndex') as AnyAttribute;
            const weightAttribute = geometry.getAttribute('skinWeight') as AnyAttribute;
            skinIndex = new Uint32Array(count * 4);
            skinWeight = new Float32Array(count * 4);
            for (let i = 0; i < count; i++) {
                let sum = 0;
                for (let k = 0; k < 4; k++) {
                    skinIndex[i * 4 + k] = indexAttribute.getComponent(i, k);
                    const weight = (skinWeight[i * 4 + k] = weightAttribute.getComponent(i, k));
                    if (weight < 0) negative = true;
                    sum += Math.fround(weight);
                }
                low = Math.min(low, sum);
                high = Math.max(high, sum);
            }
        }
        this.weightSum = this.skinned ? [low, high] : [1, 1];
        this.negativeWeights = negative;

        // Per-group boxes for bounds(): a group per bone, or one for a morph-only mesh.
        const groups = this.skinned ? bones : 1;
        const stride = 6 * (1 + this.morphTargets);
        this.groupBoxes = new Float64Array(groups * stride);
        this.groupUsed = new Uint8Array(groups);
        for (let g = 0; g < groups; g++) for (let b = 0; b <= this.morphTargets; b++) emptyBox(this.groupBoxes.subarray(g * stride + b * 6, g * stride + b * 6 + 6));
        const include = (g: number, i: number) => {
            if (g >= groups) return;
            this.groupUsed[g] = 1;
            const base = g * stride;
            growBox(this.groupBoxes.subarray(base, base + 6), [rest[i * 4], rest[i * 4 + 1], rest[i * 4 + 2], rest[i * 4], rest[i * 4 + 1], rest[i * 4 + 2]]);
            for (let t = 0; t < this.morphTargets; t++) {
                const at = (t * count + i) * 4;
                growBox(this.groupBoxes.subarray(base + (t + 1) * 6, base + (t + 2) * 6), [targets![at], targets![at + 1], targets![at + 2], targets![at], targets![at + 1], targets![at + 2]]);
            }
        };
        for (let i = 0; i < count; i++) {
            if (!this.skinned) include(0, i);
            else for (let k = 0; k < 4; k++) if (skinWeight![i * 4 + k] !== 0) include(skinIndex![i * 4 + k], i);
        }

        // Buffers: the output is ours (SceneTrees reads it); three binds a stand-in for it.
        this.buffer = device.createBuffer({ label: `@pmndrs/bvh/three pose ${mesh.name || mesh.id}`, size: Math.max(16, count * 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.output = storageStandIn(renderer, this.buffer, 4);
        const out = storage(this.output, 'vec4', 0);
        const restNode = storage(new StorageBufferAttribute(rest, 4), 'vec4', 0).toReadOnly();
        const countNode = uniform(count, 'uint');
        // Matrices: bindMatrix, bindMatrixInverse, then each bone, as 4 vec4 columns.
        this.matrices = this.skinned ? new StorageBufferAttribute(new Float32Array((2 + bones) * 16), 4) : null;
        this.influences = this.morphTargets > 0 ? new StorageBufferAttribute(new Float32Array(1 + this.morphTargets), 1) : null;
        const matricesNode = this.matrices ? storage(this.matrices, 'vec4', 0).toReadOnly() : null;
        const influencesNode = this.influences ? storage(this.influences, 'float', 0).toReadOnly() : null;
        const targetsNode = targets ? storage(new StorageBufferAttribute(targets, 4), 'vec4', 0).toReadOnly() : null;
        const targetCountNode = uniform(this.morphTargets, 'uint');
        const skinIndexNode = skinIndex ? storage(new StorageBufferAttribute(skinIndex, 4), 'uvec4', 0).toReadOnly() : null;
        const skinWeightNode = skinWeight ? storage(new StorageBufferAttribute(skinWeight, 4), 'vec4', 0).toReadOnly() : null;
        const matrixAt = (index: BvhTslNode): BvhTslNode => {
            const base = uint(index).mul(4);
            return mat4(matricesNode!.element(base), matricesNode!.element(base.add(1)), matricesNode!.element(base.add(2)), matricesNode!.element(base.add(3)));
        };

        const kernel = Fn(() => {
            const i = instanceIndex;
            If(i.lessThan(countNode), () => {
                const p = (restNode.element(i) as BvhTslNode).xyz.toVar();
                if (influencesNode && targetsNode) {
                    // three's morphReference: p *= base; p += target * influence for each influence != 0.
                    p.mulAssign(influencesNode.element(0));
                    Loop({ start: uint(0), end: targetCountNode, type: 'uint', condition: '<' }, ({ i: t }: { i: BvhTslNode }) => {
                        const influence = float(influencesNode.element(t.add(1))).toVar();
                        If(influence.notEqual(0), () => {
                            p.addAssign((targetsNode.element(t.mul(countNode).add(i)) as BvhTslNode).mul(influence).xyz);
                        });
                    });
                }
                if (matricesNode && skinIndexNode && skinWeightNode) {
                    // three's getSkinnedPosition, term for term.
                    const skinIndexValue = (skinIndexNode.element(i) as BvhTslNode).toVar();
                    const skinWeightValue = (skinWeightNode.element(i) as BvhTslNode).toVar();
                    const bindMatrix = matrixAt(uint(0));
                    const bindMatrixInverse = matrixAt(uint(1));
                    const skinVertex = bindMatrix.mul(vec4(p, 1)).toVar();
                    const skinned = add(
                        matrixAt(skinIndexValue.x.add(2)).mul(skinWeightValue.x).mul(skinVertex),
                        matrixAt(skinIndexValue.y.add(2)).mul(skinWeightValue.y).mul(skinVertex),
                        matrixAt(skinIndexValue.z.add(2)).mul(skinWeightValue.z).mul(skinVertex),
                        matrixAt(skinIndexValue.w.add(2)).mul(skinWeightValue.w).mul(skinVertex),
                    );
                    p.assign(bindMatrixInverse.mul(skinned).xyz);
                }
                out.element(i).assign(vec4(p, 1));
            });
        });
        this.node = (kernel() as unknown as Node & { compute(count: number, workgroupSize: number[]): ComputeNode }).compute(count, [options.workgroupSize ?? 64]);
    }

    /** Compiles the pass's pipeline ahead of the first {@link update} (`compileComputeAsync`), so the first frame doesn't compile synchronously. */
    async prepare(): Promise<void> {
        await this.renderer.compileComputeAsync?.(this.node);
    }

    update(): void {
        this.checkLive();
        if (this.matrices) {
            const skinned = this.mesh as unknown as SkinnedMesh;
            skinned.skeleton.update();
            const array = this.matrices.array as Float32Array;
            array.set(skinned.bindMatrix.elements, 0);
            array.set(skinned.bindMatrixInverse.elements, 16);
            array.set(skinned.skeleton.boneMatrices!.subarray(0, (array.length / 16 - 2) * 16), 32);
            this.matrices.needsUpdate = true;
        }
        if (this.influences) {
            const array = this.influences.array as Float32Array;
            const influences = this.mesh.morphTargetInfluences!;
            let sum = 0;
            for (let t = 0; t < this.morphTargets; t++) {
                array[t + 1] = influences[t] ?? 0;
                sum += influences[t] ?? 0;
            }
            array[0] = this.mesh.geometry.morphTargetsRelative ? 1 : 1 - sum;
            this.influences.needsUpdate = true;
        }
        this.renderer.compute(this.node);
        checkStandIn(this.renderer, this.output, this.buffer, 'pose');
    }

    bounds(target = new Float32Array(6)): Float32Array {
        if (this.negativeWeights) return this.boundsFromVertices(target);
        const morphs = this.morphTargets;
        const influences = this.mesh.morphTargetInfluences;
        let sum = 0;
        for (let t = 0; t < morphs; t++) sum += influences![t] ?? 0;
        const base = this.mesh.geometry.morphTargetsRelative ? 1 : 1 - sum;
        const stride = 6 * (1 + morphs);
        const morphed = new Float64Array(6);
        const carried = new Float64Array(6);
        const union = new Float64Array(6);
        emptyBox(union);
        const skinned = this.skinned ? (this.mesh as unknown as SkinnedMesh) : null;
        const boneMatrices = skinned?.skeleton.boneMatrices ?? null;
        for (let g = 0; g < this.groupUsed.length; g++) {
            if (!this.groupUsed[g]) continue;
            // The morphed positions: base * rest + sum(influence * target), as boxes (a negative factor swaps the ends).
            const at = g * stride;
            for (let k = 0; k < 3; k++) {
                const a = base * this.groupBoxes[at + k];
                const b = base * this.groupBoxes[at + k + 3];
                morphed[k] = Math.min(a, b);
                morphed[k + 3] = Math.max(a, b);
            }
            for (let t = 0; t < morphs; t++) {
                const influence = influences![t] ?? 0;
                if (influence === 0) continue;
                for (let k = 0; k < 3; k++) {
                    const a = influence * this.groupBoxes[at + (t + 1) * 6 + k];
                    const b = influence * this.groupBoxes[at + (t + 1) * 6 + k + 3];
                    morphed[k] += Math.min(a, b);
                    morphed[k + 3] += Math.max(a, b);
                }
            }
            if (!skinned) {
                growBox(union, morphed);
                continue;
            }
            // Into bind space, then through the bone.
            transformBox(skinned.bindMatrix.elements, 0, morphed, carried);
            transformBox(boneMatrices!, g * 16, carried, morphed);
            growBox(union, morphed);
        }
        if (!(union[0] <= union[3])) {
            target.fill(0);
            return target;
        }
        let result = union;
        if (skinned) {
            // A skinned vertex is S * bindMatrixInverse(y), y in the union and S its weight sum.
            const back = new Float64Array(6);
            transformBox(skinned.bindMatrixInverse.elements, 0, union, back);
            result = new Float64Array(6);
            emptyBox(result);
            for (const scale of this.weightSum) {
                for (let k = 0; k < 3; k++) {
                    const a = scale * back[k];
                    const b = scale * back[k + 3];
                    result[k] = Math.min(result[k], a, b);
                    result[k + 3] = Math.max(result[k + 3], a, b);
                }
            }
        }
        // f32 rounding on the GPU: a little slack.
        const extent = Math.max(result[3] - result[0], result[4] - result[1], result[5] - result[2]);
        const pad = extent * 1e-4 + 1e-6;
        for (let k = 0; k < 3; k++) {
            target[k] = Math.fround(result[k] - pad);
            target[k + 3] = Math.fround(result[k + 3] + pad);
        }
        return target;
    }

    /** Every vertex through three's `getVertexPosition` (morphs and skinning): the exact box, at a cost per vertex. */
    private boundsFromVertices(target: Float32Array): Float32Array {
        const box = new Box3();
        for (let i = 0; i < this.vertexCount; i++) box.expandByPoint(this.mesh.getVertexPosition(i, scratchVector));
        const pad = Math.max(...box.getSize(scratchVector).toArray()) * 1e-4 + 1e-6;
        target.set([box.min.x - pad, box.min.y - pad, box.min.z - pad, box.max.x + pad, box.max.y + pad, box.max.z + pad]);
        return target;
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.output.dispose();
        this.buffer.destroy();
    }

    private checkLive(): void {
        if (this.disposed) throw new Error('@pmndrs/bvh/three: this BvhPosePass was disposed');
    }
}

/** The vertices `SceneTrees.setPose` takes for one of a pose pass's meshes (a geometry range). */
export function poseVertices(pass: BvhPosePassLike, geometry: BufferGeometry, start: number, count: number, indexBuffer: GPUBuffer | null): MeshRefitVertices {
    if (geometry.index) {
        if (!indexBuffer) throw new Error('@pmndrs/bvh/three: an indexed range needs its index buffer');
        return { positionBuffer: pass.buffer, vertexCount: pass.vertexCount, positionStrideFloats: 4, indexBuffer, indexFormat: 'uint32' };
    }
    return { positionBuffer: pass.buffer, vertexCount: count, positionStrideFloats: 4, positionOffsetFloats: start * 4 };
}


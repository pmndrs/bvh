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
 * morphs (`InstancedMesh.morphTexture`) are not read: every instance would get
 * `morphTargetInfluences`, so `SceneTreesObject3DSync` refuses an
 * `InstancedMesh` with a `morphTexture` and more than one instance (it warns
 * once and reads it in its rest pose).
 *
 * **Inputs** are read through the attributes once, at construction
 * (interleaved, normalized and non-float attributes work) and packed into
 * storage buffers: positions and morph targets as `vec4`, skin indices as
 * `uvec4`, weights as `vec4`. Bone matrices (with the bind matrices) and
 * influences are uploaded each {@link BvhPosePass.update}. A geometry whose
 * rest positions, skin attributes or morph targets change needs a new pass
 * (the sync makes one when the geometry object changes). So does a replaced
 * skeleton, or one with a different bone count: the pass is keyed on the
 * skeleton object and its bone count ({@link BvhPosePass.current}), throws
 * from {@link BvhPosePass.update} and {@link BvhPosePass.bounds} once they
 * change, and the sync makes a new one.
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
 * is computed from every vertex on the CPU instead, with the kernel's own
 * formula, so `bindMatrixInverse`'s translation counts once per unit of weight
 * sum as on the GPU, not once as in three's `getVertexPosition`). Weights that
 * don't sum to 1 scale the result, and the box is scaled to match. Cost per
 * update: 16 corner transforms per bone, plus 6 floats per bone and morph target.
 *
 * **f32 room.** The box is computed exactly (f64, from the f32 values the GPU
 * reads), but the kernel rounds. Its chain per coordinate: the morph (1
 * rounding for `p *= base`, 2 per target), `bindMatrix * p` (a 4-term dot
 * product: 4 roundings), `bone * weight` (1), `* v` (4), the sum over 4 bones
 * (3) and `bindMatrixInverse * …` (4): 16 roundings plus 2 per morph target.
 * Each is at most 2⁻²⁴ (half an ulp) of the value it rounds, and every value
 * in the chain is at most the largest coordinate the chain sees (the morph's
 * terms, the bind-space and bone-space boxes and the result: `M`) for
 * near-rigid matrices, so the box gets `(8 + targets) · 2⁻²³ · M` on every side,
 * on top of `1e-4` of its extent (which covers scaled bones and products that
 * cancel near the origin). That room grows with the distance from the origin,
 * not with the mesh's size: 5,000 units out, f32 steps are 4.9e-4 apart
 * whatever the mesh's extent. The box's ends are then rounded outward to f32.
 */

import { type BufferAttribute, type BufferGeometry, type InterleavedBufferAttribute, type Mesh, type Skeleton, type SkinnedMesh } from 'three';
import { Fn, If, Loop, add, float, instanceIndex, mat4, storage, uint, uniform, vec4 } from 'three/tsl';
import { StorageBufferAttribute, type ComputeNode, type Node } from 'three/webgpu';
import type { MeshRefitVertices } from '../gpu-mesh-refit.ts';
import { checkStandIn, storageStandIn, type BvhTslNode } from './trace-nodes.ts';

/** Whether three deforms the mesh when it draws it: a `SkinnedMesh` with a skeleton, or position morph targets with influences. */
export function threeDeforms(mesh: Mesh): boolean {
    return skinningSkeletonOf(mesh) !== null || morphTargetCount(mesh) > 0;
}

/** The skeleton three skins `mesh` with (a `SkinnedMesh` with a skeleton and skin attributes), or null. */
export function skinningSkeletonOf(mesh: Mesh): Skeleton | null {
    const skinned = mesh as unknown as SkinnedMesh;
    return skinned.isSkinnedMesh && skinned.skeleton && mesh.geometry.getAttribute('skinIndex') && mesh.geometry.getAttribute('skinWeight') ? skinned.skeleton : null;
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
    /** Compiles the pass's pipeline ahead of its first {@link update}. */
    prepare?(): Promise<void>;
    /** Passes with the same variant share a pipeline: the sync prepares one of each. */
    readonly variant?: string;
    dispose(): void;
}

export interface BvhPosePassOptions {
    /** Compute workgroup size. Default 64. */
    readonly workgroupSize?: number;
}

type AnyAttribute = BufferAttribute | InterleavedBufferAttribute;

/** Half an f32 ulp, relative: the most one f32 rounding moves a value, as a fraction of it. */
const F32_ROUNDING = 2 ** -24;
/** The kernel's roundings per coordinate, before the morph targets' (see the file comment). */
const KERNEL_ROUNDINGS = 16;

const f32Scratch = new Float32Array(1);
const i32Scratch = new Int32Array(f32Scratch.buffer);

/** The f32 next to `f` (an f32) toward +Infinity (`up`) or -Infinity. */
function f32Step(f: number, up: boolean): number {
    if (f === 0) return up ? 2 ** -149 : -(2 ** -149);
    if (!Number.isFinite(f)) return f;
    f32Scratch[0] = f;
    i32Scratch[0] += f > 0 === up ? 1 : -1;
    return f32Scratch[0];
}

/** The largest f32 at most `x`. */
export function f32Below(x: number): number {
    const f = Math.fround(x);
    return f > x ? f32Step(f, false) : f;
}

/** The smallest f32 at least `x`. */
export function f32Above(x: number): number {
    const f = Math.fround(x);
    return f < x ? f32Step(f, true) : f;
}

function maxAbs(box: ArrayLike<number>): number {
    let m = 0;
    for (let k = 0; k < 6; k++) m = Math.max(m, Math.abs(box[k]));
    return m;
}

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

/** The inputs the negative-weight box reads per vertex (kept only for such a mesh). */
interface VertexInputs {
    readonly rest: Float32Array;
    readonly targets: Float32Array | null;
    readonly skinIndex: Uint32Array | null;
    readonly skinWeight: Float32Array | null;
}

/** A morph or skin pass for one mesh; see the file comment. */
export class BvhPosePass implements BvhPosePassLike {
    readonly buffer: GPUBuffer;
    readonly vertexCount: number;
    readonly skinned: boolean;
    readonly morphTargets: number;
    /** The compute node (one dispatch over the vertices). */
    readonly node: ComputeNode;
    /** The pipeline this pass shares with others: `skin`, `morph` or `skin+morph`. */
    readonly variant: string;

    private readonly renderer: { compute(node: ComputeNode): unknown; compileComputeAsync?(node: ComputeNode): Promise<unknown> };
    private readonly output: StorageBufferAttribute;
    private readonly matrices: StorageBufferAttribute | null;
    private readonly influences: StorageBufferAttribute | null;
    /** Every storage attribute the pass made (inputs, matrices, influences): {@link dispose} releases them. */
    private readonly attributes: StorageBufferAttribute[] = [];
    /** The skeleton and bone count the pass was made for (null and 0 unskinned). */
    private readonly skeleton: Skeleton | null;
    private readonly boneCount: number;
    /** Per group (bone, or the one group of a morph-only mesh): the box of its vertices' rest positions, then of each target's values. */
    private readonly groupBoxes: Float64Array;
    private readonly groupUsed: Uint8Array;
    /** The range of the vertices' weight sums. */
    private readonly weightSum: readonly [number, number];
    /** With a negative weight, the inputs {@link bounds} reads every vertex of; else null. */
    private readonly vertexInputs: VertexInputs | null;
    /** `bindMatrix` and `bindMatrixInverse` as the GPU reads them (f32). */
    private readonly bind32 = new Float32Array(16);
    private readonly bindInverse32 = new Float32Array(16);
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
        this.skeleton = skinningSkeletonOf(mesh);
        this.skinned = this.skeleton !== null;
        this.morphTargets = morphTargetCount(mesh);
        if (!this.skinned && this.morphTargets === 0) throw new Error('@pmndrs/bvh/three: BvhPosePass needs a SkinnedMesh or position morph targets');
        this.variant = [this.skinned ? 'skin' : '', this.morphTargets > 0 ? 'morph' : ''].filter(Boolean).join('+');

        // Inputs, read through the attributes.
        const rest = new Float32Array(count * 4);
        for (let i = 0; i < count; i++) rest.set([position.getX(i), position.getY(i), position.getZ(i), 1], i * 4);
        const targets = this.morphTargets > 0 ? new Float32Array(count * 4 * this.morphTargets) : null;
        if (targets) {
            geometry.morphAttributes.position!.forEach((target, t) => {
                for (let i = 0; i < count; i++) targets.set([target.getX(i), target.getY(i), target.getZ(i), 0], (t * count + i) * 4);
            });
        }
        const bones = this.skeleton ? this.skeleton.bones.length : 0;
        this.boneCount = bones;
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
        this.vertexInputs = negative ? { rest, targets, skinIndex, skinWeight } : null;

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
        const attribute = (array: Float32Array | Uint32Array, itemSize: number) => {
            const made = new StorageBufferAttribute(array, itemSize);
            this.attributes.push(made);
            return made;
        };
        this.buffer = device.createBuffer({ label: `@pmndrs/bvh/three pose ${mesh.name || mesh.id}`, size: Math.max(16, count * 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.output = storageStandIn(renderer, this.buffer, 4);
        const out = storage(this.output, 'vec4', 0);
        const restNode = storage(attribute(rest, 4), 'vec4', 0).toReadOnly();
        const countNode = uniform(count, 'uint');
        // Matrices: bindMatrix, bindMatrixInverse, then each bone, as 4 vec4 columns.
        this.matrices = this.skinned ? attribute(new Float32Array((2 + bones) * 16), 4) : null;
        this.influences = this.morphTargets > 0 ? attribute(new Float32Array(1 + this.morphTargets), 1) : null;
        const matricesNode = this.matrices ? storage(this.matrices, 'vec4', 0).toReadOnly() : null;
        const influencesNode = this.influences ? storage(this.influences, 'float', 0).toReadOnly() : null;
        const targetsNode = targets ? storage(attribute(targets, 4), 'vec4', 0).toReadOnly() : null;
        const targetCountNode = uniform(this.morphTargets, 'uint');
        const skinIndexNode = skinIndex ? storage(attribute(skinIndex, 4), 'uvec4', 0).toReadOnly() : null;
        const skinWeightNode = skinWeight ? storage(attribute(skinWeight, 4), 'vec4', 0).toReadOnly() : null;
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

    /**
     * Whether the mesh still has the skeleton (the same object, with the same
     * bone count) the pass was made for. A pass that isn't current throws from
     * {@link update} and {@link bounds}: make a new one.
     */
    current(): boolean {
        const skeleton = skinningSkeletonOf(this.mesh);
        return skeleton === this.skeleton && (skeleton === null || skeleton.bones.length === this.boneCount);
    }

    update(): void {
        this.checkLive();
        this.checkSkeleton();
        if (this.matrices) {
            const skinned = this.mesh as unknown as SkinnedMesh;
            skinned.skeleton.update();
            const array = this.matrices.array as Float32Array;
            array.set(skinned.bindMatrix.elements, 0);
            array.set(skinned.bindMatrixInverse.elements, 16);
            array.set(skinned.skeleton.boneMatrices!.subarray(0, this.boneCount * 16), 32);
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

    bounds(target: Float32Array = new Float32Array(6)): Float32Array {
        this.checkSkeleton();
        const skinned = this.skinned ? (this.mesh as unknown as SkinnedMesh) : null;
        if (skinned) {
            this.bind32.set(skinned.bindMatrix.elements);
            this.bindInverse32.set(skinned.bindMatrixInverse.elements);
        }
        if (this.vertexInputs) return this.boundsFromVertices(this.vertexInputs, target);
        const morphs = this.morphTargets;
        const influences = this.mesh.morphTargetInfluences;
        const base = this.morphBase();
        const stride = 6 * (1 + morphs);
        const morphed = new Float64Array(6);
        const carried = new Float64Array(6);
        const union = new Float64Array(6);
        emptyBox(union);
        const boneMatrices = skinned?.skeleton.boneMatrices ?? null;
        // The largest coordinate the kernel's f32 steps see (see the file comment).
        let magnitude = 0;
        for (let g = 0; g < this.groupUsed.length; g++) {
            if (!this.groupUsed[g]) continue;
            // The morphed positions: base * rest + sum(influence * target), as boxes (a negative factor swaps the ends).
            const at = g * stride;
            let terms = Math.abs(base) * maxAbs(this.groupBoxes.subarray(at, at + 6));
            for (let k = 0; k < 3; k++) {
                const a = base * this.groupBoxes[at + k];
                const b = base * this.groupBoxes[at + k + 3];
                morphed[k] = Math.min(a, b);
                morphed[k + 3] = Math.max(a, b);
            }
            for (let t = 0; t < morphs; t++) {
                const influence = influences![t] ?? 0;
                if (influence === 0) continue;
                const box = at + (t + 1) * 6;
                terms += Math.abs(influence) * maxAbs(this.groupBoxes.subarray(box, box + 6));
                for (let k = 0; k < 3; k++) {
                    const a = influence * this.groupBoxes[box + k];
                    const b = influence * this.groupBoxes[box + k + 3];
                    morphed[k] += Math.min(a, b);
                    morphed[k + 3] += Math.max(a, b);
                }
            }
            magnitude = Math.max(magnitude, terms);
            if (!skinned) {
                growBox(union, morphed);
                continue;
            }
            // Into bind space, then through the bone.
            transformBox(this.bind32, 0, morphed, carried);
            transformBox(boneMatrices!, g * 16, carried, morphed);
            magnitude = Math.max(magnitude, maxAbs(carried), maxAbs(morphed));
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
            transformBox(this.bindInverse32, 0, union, back);
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
        return this.padded(result, magnitude, target);
    }

    /** `p *= base` of three's morphReference: 1 for relative targets, else 1 - sum(influences). */
    private morphBase(): number {
        if (this.morphTargets === 0 || this.mesh.geometry.morphTargetsRelative) return 1;
        const influences = this.mesh.morphTargetInfluences!;
        let sum = 0;
        for (let t = 0; t < this.morphTargets; t++) sum += influences[t] ?? 0;
        return 1 - sum;
    }

    /** `box` with room for the kernel's f32 rounding (see the file comment), its ends rounded outward to f32, into `target`. */
    private padded(box: Float64Array, magnitude: number, target: Float32Array): Float32Array {
        const extent = Math.max(box[3] - box[0], box[4] - box[1], box[5] - box[2]);
        magnitude = Math.max(magnitude, maxAbs(box));
        const pad = extent * 1e-4 + 1e-6 + (KERNEL_ROUNDINGS + 2 * this.morphTargets) * F32_ROUNDING * magnitude;
        for (let k = 0; k < 3; k++) {
            target[k] = f32Below(box[k] - pad);
            target[k + 3] = f32Above(box[k + 3] + pad);
        }
        return target;
    }

    /**
     * Every vertex through the kernel's own formula (in f64): the morph, then
     * `bindMatrixInverse * sum(w * bone * (bindMatrix * p))` with the sum's `w`
     * component the weight sum, so the inverse's translation counts that many
     * times, as on the GPU. For negative weights; costs a pass over the vertices.
     */
    private boundsFromVertices(inputs: VertexInputs, target: Float32Array): Float32Array {
        const { rest, targets, skinIndex, skinWeight } = inputs;
        const count = this.vertexCount;
        const morphs = this.morphTargets;
        const influences = this.mesh.morphTargetInfluences;
        const base = this.morphBase();
        const bind = this.bind32;
        const inverse = this.bindInverse32;
        const bones = this.skinned ? (this.mesh as unknown as SkinnedMesh).skeleton.boneMatrices! : null;
        const box = new Float64Array(6);
        emptyBox(box);
        let magnitude = 0;
        for (let i = 0; i < count; i++) {
            let px = rest[i * 4] * base;
            let py = rest[i * 4 + 1] * base;
            let pz = rest[i * 4 + 2] * base;
            let terms = Math.abs(base) * Math.max(Math.abs(rest[i * 4]), Math.abs(rest[i * 4 + 1]), Math.abs(rest[i * 4 + 2]));
            for (let t = 0; t < morphs; t++) {
                const influence = influences![t] ?? 0;
                if (influence === 0) continue;
                const at = (t * count + i) * 4;
                px += targets![at] * influence;
                py += targets![at + 1] * influence;
                pz += targets![at + 2] * influence;
                terms += Math.abs(influence) * Math.max(Math.abs(targets![at]), Math.abs(targets![at + 1]), Math.abs(targets![at + 2]));
            }
            magnitude = Math.max(magnitude, terms);
            if (bones) {
                const vx = bind[0] * px + bind[4] * py + bind[8] * pz + bind[12];
                const vy = bind[1] * px + bind[5] * py + bind[9] * pz + bind[13];
                const vz = bind[2] * px + bind[6] * py + bind[10] * pz + bind[14];
                let sx = 0;
                let sy = 0;
                let sz = 0;
                let sw = 0;
                let boneTerms = 0;
                for (let k = 0; k < 4; k++) {
                    const w = skinWeight![i * 4 + k];
                    const m = skinIndex![i * 4 + k] * 16;
                    if (m >= bones.length) continue;
                    const bx = bones[m] * vx + bones[m + 4] * vy + bones[m + 8] * vz + bones[m + 12];
                    const by = bones[m + 1] * vx + bones[m + 5] * vy + bones[m + 9] * vz + bones[m + 13];
                    const bz = bones[m + 2] * vx + bones[m + 6] * vy + bones[m + 10] * vz + bones[m + 14];
                    sx += w * bx;
                    sy += w * by;
                    sz += w * bz;
                    sw += w;
                    boneTerms += Math.abs(w) * Math.max(Math.abs(bx), Math.abs(by), Math.abs(bz));
                }
                magnitude = Math.max(magnitude, Math.abs(vx), Math.abs(vy), Math.abs(vz), boneTerms);
                px = inverse[0] * sx + inverse[4] * sy + inverse[8] * sz + inverse[12] * sw;
                py = inverse[1] * sx + inverse[5] * sy + inverse[9] * sz + inverse[13] * sw;
                pz = inverse[2] * sx + inverse[6] * sy + inverse[10] * sz + inverse[14] * sw;
            }
            growBox(box, [px, py, pz, px, py, pz]);
        }
        if (!(box[0] <= box[3])) {
            target.fill(0);
            return target;
        }
        return this.padded(box, magnitude, target);
    }

    /** Releases the pass: its output buffer, every storage attribute and the compute node. */
    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.output.dispose();
        for (const attribute of this.attributes) attribute.dispose();
        this.attributes.length = 0;
        this.node.dispose();
        this.buffer.destroy();
    }

    private checkLive(): void {
        if (this.disposed) throw new Error('@pmndrs/bvh/three: this BvhPosePass was disposed');
    }

    private checkSkeleton(): void {
        if (this.current()) return;
        throw new Error(
            `@pmndrs/bvh/three: the skeleton of "${this.mesh.name || this.mesh.type}" (id ${this.mesh.id}) was replaced, or its bone count changed, since its BvhPosePass was made; make a new pass (SceneTreesObject3DSync does)`,
        );
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


/*!
 * StaticGeometryGenerator: ported from three-mesh-bvh 0.9.18
 * (src/utils/StaticGeometryGenerator.js, gkjohnson/three-mesh-bvh). See THIRD_PARTY.md.
 *
 * MIT License
 *
 * Copyright (c) 2018 Garrett Johnson
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
/**
 * Skinned and morph-target meshes baked to one static `BufferGeometry` on the
 * CPU (PARITY 6.3), so a CPU tree (`computeBoundsTree`, `acceleratedRaycast`,
 * `shapecast`) sees the live pose: three-mesh-bvh's `StaticGeometryGenerator`,
 * ported as it is (class shape, fields, the attribute checks, the update in
 * place, the diff that skips unchanged meshes).
 *
 * ```ts
 * const generator = new StaticGeometryGenerator(character);   // every visible mesh under it
 * const geometry = generator.generate();                      // world-space pose, groups per material
 * geometry.computeBoundsTree();
 * const collider = new Mesh(geometry, generator.getMaterials()); // identity transform
 * // each frame, after the animation mixer:
 * generator.generate(geometry);                               // same arrays, index untouched
 * geometry.boundsTree.refit();
 * ```
 *
 * **three's math.** The morph is three's `Mesh.getVertexPosition`'s (each
 * target with an influence, relative or absolute), the skinning three's
 * `SkinnedMesh.applyBoneTransform` (morph first, as three's `getVertexPosition`
 * and its shaders), so each baked position is what three's CPU raycast tests.
 * Normals and tangents are skinned with the blended bone matrix (three-mesh-bvh's
 * `boneNormalTransform`). With `applyWorldTransforms` (default) every vertex
 * is moved by its mesh's `matrixWorld`; a mirrored mesh's triangles are
 * flipped to keep their winding.
 *
 * The GPU path (`SceneTreesObject3DSync` with a renderer, `BvhPosePass`)
 * poses on the GPU and reads nothing back; this one is for CPU queries.
 */

import { BufferAttribute, BufferGeometry, Matrix3, Matrix4, Vector3, Vector4, type InterleavedBufferAttribute, type Material, type Mesh, type Object3D, type SkinnedMesh } from 'three';

type Attribute = BufferAttribute | InterleavedBufferAttribute;
type TypedArray = Float32Array | Uint32Array | Uint16Array | Uint8Array | Int32Array | Int16Array | Int8Array | Float64Array;

const _positionVector = new Vector3();
const _normalVector = new Vector3();
const _tangentVector = new Vector3();
const _tangentVector4 = new Vector4();

const _morphVector = new Vector3();
const _temp = new Vector3();

const _skinIndex = new Vector4();
const _skinWeight = new Vector4();
const _matrix = new Matrix4();
const _boneMatrix = new Matrix4();

/** Confirms that the two attributes are compatible (both missing, or the same count, normalization, array type and item size). */
function validateAttributes(attr1: Attribute | null | undefined, attr2: Attribute | null | undefined): void {
    if (!attr1 && !attr2) return;
    const sameCount = attr1!.count === attr2!.count;
    const sameNormalized = attr1!.normalized === attr2!.normalized;
    const sameType = attr1!.array.constructor === attr2!.array.constructor;
    const sameItemSize = attr1!.itemSize === attr2!.itemSize;
    if (!sameCount || !sameNormalized || !sameType || !sameItemSize) throw new Error();
}

/** A new attribute like `attr` (array type, item size, normalization) with no data. */
function createAttributeClone(attr: Attribute, countOverride: number | null = null): BufferAttribute {
    const cons = attr.array.constructor as new (length: number) => TypedArray;
    const normalized = attr.normalized;
    const itemSize = attr.itemSize;
    const count = countOverride === null ? attr.count : countOverride;
    return new BufferAttribute(new cons(itemSize * count), itemSize, normalized);
}

/** Copies `attr` into `target`, starting `targetOffset` items in. */
function copyAttributeContents(attr: Attribute, target: BufferAttribute, targetOffset = 0): void {
    if ((attr as InterleavedBufferAttribute).isInterleavedBufferAttribute) {
        const itemSize = attr.itemSize;
        for (let i = 0, l = attr.count; i < l; i++) {
            const io = i + targetOffset;
            target.setX(io, attr.getX(i));
            if (itemSize >= 2) target.setY(io, attr.getY(i));
            if (itemSize >= 3) target.setZ(io, attr.getZ(i));
            if (itemSize >= 4) target.setW(io, attr.getW(i));
        }
    } else {
        const array = target.array as TypedArray;
        const cons = array.constructor as new (buffer: ArrayBufferLike, byteOffset: number, length: number) => TypedArray;
        const byteOffset = array.BYTES_PER_ELEMENT * attr.itemSize * targetOffset;
        const temp = new cons(array.buffer, byteOffset, attr.array.length);
        temp.set(attr.array as ArrayLike<number>);
    }
}

/** Adds `matrix` times `scale` to `target`. */
function addScaledMatrix(target: Matrix4, matrix: Matrix4, scale: number): void {
    const targetArray = target.elements;
    const matrixArray = matrix.elements;
    for (let i = 0, l = matrixArray.length; i < l; i++) targetArray[i] += matrixArray[i] * scale;
}

/** `SkinnedMesh.applyBoneTransform` for a direction (normals, tangents). */
function boneNormalTransform(mesh: SkinnedMesh, index: number, target: Vector3): Vector3 {
    const skeleton = mesh.skeleton;
    const geometry = mesh.geometry;
    const bones = skeleton.bones;
    const boneInverses = skeleton.boneInverses;
    _skinIndex.fromBufferAttribute(geometry.attributes.skinIndex as BufferAttribute, index);
    _skinWeight.fromBufferAttribute(geometry.attributes.skinWeight as BufferAttribute, index);
    _matrix.elements.fill(0);
    for (let i = 0; i < 4; i++) {
        const weight = _skinWeight.getComponent(i);
        if (weight !== 0) {
            const boneIndex = _skinIndex.getComponent(i);
            _boneMatrix.multiplyMatrices(bones[boneIndex].matrixWorld, boneInverses[boneIndex]);
            addScaledMatrix(_matrix, _boneMatrix, weight);
        }
    }
    _matrix.multiply(mesh.bindMatrix).premultiply(mesh.bindMatrixInverse);
    target.transformDirection(_matrix);
    return target;
}

/** Applies the morph targets with an influence to `target` (three's `Mesh.getVertexPosition` morph). */
function applyMorphTarget(morphData: Attribute[], morphInfluences: number[], morphTargetsRelative: boolean, i: number, target: Vector3): void {
    _morphVector.set(0, 0, 0);
    for (let j = 0, jl = morphData.length; j < jl; j++) {
        const influence = morphInfluences[j];
        const morphAttribute = morphData[j];
        if (influence === 0) continue;
        _temp.fromBufferAttribute(morphAttribute, i);
        if (morphTargetsRelative) _morphVector.addScaledVector(_temp, influence);
        else _morphVector.addScaledVector(_temp.sub(target), influence);
    }
    target.add(_morphVector);
}

interface MergeOptions {
    readonly useGroups?: boolean;
    readonly updateIndex?: boolean;
    readonly skipAttributes?: boolean[];
}

/** three's `BufferGeometryUtils.mergeGeometries`, ignoring morph targets and updating the attributes of `targetGeometry` in place. */
function mergeBufferGeometries(geometries: BufferGeometry[], options: MergeOptions = { useGroups: false, updateIndex: false, skipAttributes: [] }, targetGeometry = new BufferGeometry()): BufferGeometry {
    const isIndexed = geometries[0].index !== null;
    const { useGroups = false, updateIndex = false, skipAttributes = [] } = options;
    const attributesUsed = new Set(Object.keys(geometries[0].attributes));
    const attributes: Record<string, Attribute[]> = {};
    let offset = 0;
    targetGeometry.clearGroups();
    for (let i = 0; i < geometries.length; ++i) {
        const geometry = geometries[i];
        let attributesCount = 0;
        // ensure that all geometries are indexed, or none
        if (isIndexed !== (geometry.index !== null)) {
            throw new Error('StaticGeometryGenerator: All geometries must have compatible attributes; make sure index attribute exists among all geometries, or in none of them.');
        }
        // gather attributes, exit early if they're different
        for (const name in geometry.attributes) {
            if (!attributesUsed.has(name)) {
                throw new Error('StaticGeometryGenerator: All geometries must have compatible attributes; make sure "' + name + '" attribute exists among all geometries, or in none of them.');
            }
            if (attributes[name] === undefined) attributes[name] = [];
            attributes[name].push(geometry.attributes[name] as Attribute);
            attributesCount++;
        }
        // ensure geometries have the same number of attributes
        if (attributesCount !== attributesUsed.size) throw new Error('StaticGeometryGenerator: Make sure all geometries have the same number of attributes.');
        if (useGroups) {
            let count: number;
            if (isIndexed) count = geometry.index!.count;
            else if (geometry.attributes.position !== undefined) count = geometry.attributes.position.count;
            else throw new Error('StaticGeometryGenerator: The geometry must have either an index or a position attribute');
            targetGeometry.addGroup(offset, count, i);
            offset += count;
        }
    }
    // merge indices
    if (isIndexed) {
        let forceUpdateIndex = false;
        if (!targetGeometry.index) {
            let indexCount = 0;
            for (let i = 0; i < geometries.length; ++i) indexCount += geometries[i].index!.count;
            targetGeometry.setIndex(new BufferAttribute(new Uint32Array(indexCount), 1, false));
            forceUpdateIndex = true;
        }
        if (updateIndex || forceUpdateIndex) {
            const targetIndex = targetGeometry.index!;
            let targetOffset = 0;
            let indexOffset = 0;
            for (let i = 0; i < geometries.length; ++i) {
                const geometry = geometries[i];
                const index = geometry.index!;
                if (skipAttributes[i] !== true) {
                    for (let j = 0; j < index.count; ++j) {
                        targetIndex.setX(targetOffset, index.getX(j) + indexOffset);
                        targetOffset++;
                    }
                }
                indexOffset += geometry.attributes.position.count;
            }
        }
    }
    // merge attributes
    for (const name in attributes) {
        const attrList = attributes[name];
        if (!(name in targetGeometry.attributes)) {
            let count = 0;
            for (const key in attrList) count += attrList[key].count;
            targetGeometry.setAttribute(name, createAttributeClone(attributes[name][0], count));
        }
        const targetAttribute = targetGeometry.attributes[name] as BufferAttribute;
        let attributeOffset = 0;
        for (let i = 0, l = attrList.length; i < l; i++) {
            const attr = attrList[i];
            if (skipAttributes[i] !== true) copyAttributeContents(attr, targetAttribute, attributeOffset);
            attributeOffset += attr.count;
        }
    }
    return targetGeometry;
}

function checkTypedArrayEquality(a: ArrayLike<number> | null, b: ArrayLike<number> | null): boolean {
    if (a === null || b === null) return a === b;
    if (a.length !== b.length) return false;
    for (let i = 0, l = a.length; i < l; i++) if (a[i] !== b[i]) return false;
    return true;
}

/** Flips every triangle's winding (corners 0 and 2). */
function invertGeometry(geometry: BufferGeometry): BufferGeometry {
    const { index, attributes } = geometry;
    if (index) {
        for (let i = 0, l = index.count; i < l; i += 3) {
            const v0 = index.getX(i);
            const v2 = index.getX(i + 2);
            index.setX(i, v2);
            index.setX(i + 2, v0);
        }
    } else {
        for (const key in attributes) {
            const attr = attributes[key] as BufferAttribute;
            const itemSize = attr.itemSize;
            for (let i = 0, l = attr.count; i < l; i += 3) {
                for (let j = 0; j < itemSize; j++) {
                    const v0 = attr.getComponent(i, j);
                    const v2 = attr.getComponent(i + 2, j);
                    attr.setComponent(i, j, v2);
                    attr.setComponent(i + 2, j, v0);
                }
            }
        }
    }
    return geometry;
}

/**
 * Whether a mesh changed (world matrix, positions version, bone matrices,
 * triangle count, and here morph influences) since the last bake.
 * three-mesh-bvh 0.9.18 doesn't compare the influences, so a mesh whose
 * only change is a morph weight keeps its old pose there; this port
 * compares them too (the one departure from its source).
 */
class GeometryDiff {
    readonly matrixWorld = new Matrix4();
    geometryHash: number | null = null;
    boneMatrices: Float32Array | null = null;
    morphInfluences: number[] | null = null;
    primitiveCount = -1;

    constructor(readonly mesh: Mesh) {
        this.update();
    }

    update(): void {
        const mesh = this.mesh;
        const geometry = mesh.geometry;
        const skeleton = (mesh as SkinnedMesh).skeleton;
        const primitiveCount = (geometry.index ? geometry.index.count : geometry.attributes.position.count) / 3;
        this.matrixWorld.copy(mesh.matrixWorld);
        this.geometryHash = (geometry.attributes.position as BufferAttribute).version;
        this.primitiveCount = primitiveCount;
        this.morphInfluences = mesh.morphTargetInfluences ? mesh.morphTargetInfluences.slice() : null;
        if (skeleton) {
            // ensure the bone matrix array is updated to the appropriate length
            if (!skeleton.boneTexture) skeleton.computeBoneTexture();
            skeleton.update();
            // copy data if possible otherwise clone it
            const boneMatrices = skeleton.boneMatrices!;
            if (!this.boneMatrices || this.boneMatrices.length !== boneMatrices.length) this.boneMatrices = boneMatrices.slice();
            else this.boneMatrices.set(boneMatrices);
        } else {
            this.boneMatrices = null;
        }
    }

    didChange(): boolean {
        const mesh = this.mesh;
        const geometry = mesh.geometry;
        const primitiveCount = (geometry.index ? geometry.index.count : geometry.attributes.position.count) / 3;
        const skeleton = (mesh as SkinnedMesh).skeleton;
        const identical =
            this.matrixWorld.equals(mesh.matrixWorld) &&
            this.geometryHash === (geometry.attributes.position as BufferAttribute).version &&
            checkTypedArrayEquality((skeleton && skeleton.boneMatrices) || null, this.boneMatrices) &&
            checkTypedArrayEquality(mesh.morphTargetInfluences ?? null, this.morphInfluences) &&
            this.primitiveCount === primitiveCount;
        return !identical;
    }
}

/**
 * three-mesh-bvh's `StaticGeometryGenerator`: a set of skinned, morphed (or
 * plain) meshes baked into one static geometry that a tree can be built for
 * (see the file comment).
 */
export class StaticGeometryGenerator {
    /** The meshes baked: every visible `Mesh` under the objects given (`traverseVisible`), in order. */
    readonly meshes: Mesh[];
    /** With true (default), one group per mesh, so `getMaterials()` maps them; an array of materials on a mesh is not split further. */
    useGroups = true;
    /** Move each vertex (and normal, tangent) by its mesh's world transform. Default true. */
    applyWorldTransforms = true;
    /** The attributes copied onto the static geometry. */
    attributes = ['position', 'normal', 'color', 'tangent', 'uv', 'uv2'];
    private readonly _intermediateGeometry: BufferGeometry[];
    private readonly _diffMap = new WeakMap<Mesh, GeometryDiff>();

    /** Takes an object hierarchy (or an array of them) whose visible meshes to bake. */
    constructor(meshes: Object3D | Object3D[]) {
        const roots = Array.isArray(meshes) ? meshes : [meshes];
        const finalMeshes: Mesh[] = [];
        roots.forEach((object) => {
            object.traverseVisible((c) => {
                if ((c as Mesh).isMesh) finalMeshes.push(c as Mesh);
            });
        });
        this.meshes = finalMeshes;
        this._intermediateGeometry = finalMeshes.map(() => new BufferGeometry());
    }

    /** The meshes' materials in group order: `new Mesh(geometry, generator.getMaterials())`. */
    getMaterials(): Material[] {
        const materials: Material[] = [];
        this.meshes.forEach((mesh) => {
            if (Array.isArray(mesh.material)) materials.push(...mesh.material);
            else materials.push(mesh.material);
        });
        return materials;
    }

    /**
     * Bakes the meshes into `targetGeometry` (a new geometry by default).
     * Passing the same geometry again updates its attributes in place and
     * leaves its index alone, so a tree built for it stays valid: call
     * `boundsTree.refit()` after. Throws when a mesh's attributes changed in
     * a way that doesn't fit (count, type, item size). Meshes that didn't
     * change since the last call are skipped.
     */
    generate(targetGeometry = new BufferGeometry()): BufferGeometry {
        // track which attributes have been updated and which to skip to avoid unnecessary attribute copies
        const skipAttributes: boolean[] = [];
        const { meshes, useGroups, _intermediateGeometry, _diffMap } = this;
        for (let i = 0, l = meshes.length; i < l; i++) {
            const mesh = meshes[i];
            const geom = _intermediateGeometry[i];
            const diff = _diffMap.get(mesh);
            if (!diff || diff.didChange()) {
                this._convertToStaticGeometry(mesh, geom);
                skipAttributes.push(false);
                if (!diff) _diffMap.set(mesh, new GeometryDiff(mesh));
                else diff.update();
            } else {
                skipAttributes.push(true);
            }
        }
        if (_intermediateGeometry.length === 0) {
            // if there are no geometries then just create a fake empty geometry to provide
            targetGeometry.setIndex(null);
            // remove all geometry
            const attrs = targetGeometry.attributes;
            for (const key in attrs) targetGeometry.deleteAttribute(key);
            // create dummy attributes
            for (const key in this.attributes) targetGeometry.setAttribute(this.attributes[key], new BufferAttribute(new Float32Array(0), 4, false));
        } else {
            mergeBufferGeometries(_intermediateGeometry, { useGroups, skipAttributes }, targetGeometry);
        }
        for (const key in targetGeometry.attributes) (targetGeometry.attributes[key] as BufferAttribute).needsUpdate = true;
        return targetGeometry;
    }

    private _convertToStaticGeometry(mesh: Mesh, targetGeometry = new BufferGeometry()): BufferGeometry {
        const geometry = mesh.geometry;
        const applyWorldTransforms = this.applyWorldTransforms;
        const includeNormal = this.attributes.includes('normal');
        const includeTangent = this.attributes.includes('tangent');
        const attributes = geometry.attributes as Record<string, Attribute>;
        const targetAttributes = targetGeometry.attributes as Record<string, BufferAttribute>;
        // initialize the attributes if they don't exist
        if (!targetGeometry.index && geometry.index) targetGeometry.index = geometry.index.clone();
        if (!targetAttributes.position) targetGeometry.setAttribute('position', createAttributeClone(attributes.position));
        if (includeNormal && !targetAttributes.normal && attributes.normal) targetGeometry.setAttribute('normal', createAttributeClone(attributes.normal));
        if (includeTangent && !targetAttributes.tangent && attributes.tangent) targetGeometry.setAttribute('tangent', createAttributeClone(attributes.tangent));
        // ensure the attributes are consistent
        validateAttributes(geometry.index, targetGeometry.index);
        validateAttributes(attributes.position, targetAttributes.position);
        if (includeNormal) validateAttributes(attributes.normal, targetAttributes.normal);
        if (includeTangent) validateAttributes(attributes.tangent, targetAttributes.tangent);
        // generate transformed vertex attribute data
        const position = attributes.position;
        const normal = includeNormal ? attributes.normal : null;
        const tangent = includeTangent ? attributes.tangent : null;
        const morphPosition = geometry.morphAttributes.position as Attribute[] | undefined;
        const morphNormal = geometry.morphAttributes.normal as Attribute[] | undefined;
        const morphTangent = (geometry.morphAttributes as Record<string, Attribute[] | undefined>).tangent;
        const morphTargetsRelative = geometry.morphTargetsRelative;
        const morphInfluences = mesh.morphTargetInfluences;
        const normalMatrix = new Matrix3();
        normalMatrix.getNormalMatrix(mesh.matrixWorld);
        // copy the index
        if (geometry.index) (targetGeometry.index!.array as TypedArray).set(geometry.index.array as ArrayLike<number>);
        // copy and apply other attributes
        for (let i = 0, l = attributes.position.count; i < l; i++) {
            _positionVector.fromBufferAttribute(position, i);
            if (normal) _normalVector.fromBufferAttribute(normal as BufferAttribute, i);
            if (tangent) {
                _tangentVector4.fromBufferAttribute(tangent as BufferAttribute, i);
                _tangentVector.fromBufferAttribute(tangent as BufferAttribute, i);
            }
            // apply morph target transform
            if (morphInfluences) {
                if (morphPosition) applyMorphTarget(morphPosition, morphInfluences, morphTargetsRelative, i, _positionVector);
                if (morphNormal) applyMorphTarget(morphNormal, morphInfluences, morphTargetsRelative, i, _normalVector);
                if (morphTangent) applyMorphTarget(morphTangent, morphInfluences, morphTargetsRelative, i, _tangentVector);
            }
            // apply bone transform
            if ((mesh as SkinnedMesh).isSkinnedMesh) {
                (mesh as SkinnedMesh).applyBoneTransform(i, _positionVector);
                if (normal) boneNormalTransform(mesh as SkinnedMesh, i, _normalVector);
                if (tangent) boneNormalTransform(mesh as SkinnedMesh, i, _tangentVector);
            }
            // update the vectors of the attributes
            if (applyWorldTransforms) _positionVector.applyMatrix4(mesh.matrixWorld);
            targetAttributes.position.setXYZ(i, _positionVector.x, _positionVector.y, _positionVector.z);
            if (normal) {
                if (applyWorldTransforms) _normalVector.applyNormalMatrix(normalMatrix);
                targetAttributes.normal.setXYZ(i, _normalVector.x, _normalVector.y, _normalVector.z);
            }
            if (tangent) {
                if (applyWorldTransforms) _tangentVector.transformDirection(mesh.matrixWorld);
                targetAttributes.tangent.setXYZW(i, _tangentVector.x, _tangentVector.y, _tangentVector.z, _tangentVector4.w);
            }
        }
        // copy other attributes over
        for (const i in this.attributes) {
            const key = this.attributes[i];
            if (key === 'position' || key === 'tangent' || key === 'normal' || !(key in attributes)) continue;
            if (!targetAttributes[key]) targetGeometry.setAttribute(key, createAttributeClone(attributes[key]));
            validateAttributes(attributes[key], targetAttributes[key]);
            copyAttributeContents(attributes[key], targetAttributes[key]);
        }
        if (mesh.matrixWorld.determinant() < 0) invertGeometry(targetGeometry);
        return targetGeometry;
    }
}

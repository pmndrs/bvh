/**
 * The oracle's per-triangle material variants (`MeshInput.triangleMaterials`):
 * the same case with materials on every other mesh's triangles (every fourth
 * triangle without one, so it takes its instance's), and the reference's
 * answers moved to the material rule (the triangle's when it has one, else the
 * instance's) without tracing the reference again: the hits are the same,
 * only the material differs.
 */

import type { ReferenceHit } from '../../../../src/trace/reference-tracer.ts';
import { TRIANGLE_MATERIAL_NONE } from '../../../../src/mesh-input.ts';
import type { OracleCase } from './cases.ts';
import type { ReferenceAnswers } from './check.ts';

/** Triangle `t` of mesh `mesh`'s material: 0-65534, every fourth none; on even meshes only. */
export function oracleTriangleMaterial(mesh: number, triangle: number): number {
    if (mesh % 2 === 1 || triangle % 4 === 3) return TRIANGLE_MATERIAL_NONE;
    return (triangle * 7919 + mesh * 31) % 65535;
}

/** The case with per-triangle materials on its even meshes; hand-worked materials are dropped (the reference checks them). */
export function withTriangleMaterials(oracleCase: OracleCase): OracleCase {
    const meshes = oracleCase.meshes.map((mesh, index) => {
        if (index % 2 === 1) return mesh;
        const count = (mesh.indices ? mesh.indices.length : mesh.positions.length / 3) / 3;
        return { ...mesh, triangleMaterials: Uint16Array.from({ length: count }, (_, triangle) => oracleTriangleMaterial(index, triangle)) };
    });
    const curated = oracleCase.curated?.map((entry) => {
        const { material: _material, ...expected } = entry.expected;
        return { ...entry, expected };
    });
    return { ...oracleCase, meshes, ...(curated ? { curated } : {}) };
}

/**
 * The reference's answers for {@link withTriangleMaterials}'s case, from the
 * plain case's: each closest hit's material by the rule. Null when two
 * instances share an `instanceIndex` (a hit could not name its mesh): trace
 * the reference again then.
 */
export function triangleMaterialReference(materialCase: OracleCase, reference: ReferenceAnswers): ReferenceAnswers | null {
    const meshOf = new Map<number, number>();
    for (const instance of materialCase.instances) {
        if (meshOf.has(instance.instanceIndex)) return null;
        meshOf.set(instance.instanceIndex, instance.meshIndex);
    }
    const closest = reference.closest.map((hit): ReferenceHit | null => {
        if (!hit) return hit;
        const own = materialCase.meshes[meshOf.get(hit.instanceIndex)!].triangleMaterials?.[hit.triangleIndex];
        return own === undefined || own === TRIANGLE_MATERIAL_NONE ? hit : { ...hit, material: own };
    });
    return { closest, any: reference.any };
}

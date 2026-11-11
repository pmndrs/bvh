/**
 * The oracle's cases for the walk's hooks: the triangle filter
 * (`bvhTrianglePasses`, with flagged triangles) and instance masks against ray
 * masks. Each case is one scene with flags, a filter written twice (WGSL for
 * the GPU, TypeScript for the CPU walk and the reference) and the ray masks to
 * trace with. Plain data, so the CPU tests (`cases.test.ts`) and the GPU page
 * (`main.ts`) run exactly the same rays.
 *
 * Every filter here gives the same answer for every hit on one triangle (it
 * reads the instance, the triangle, the query or the ray mask, never t or the
 * barycentrics). The GPU walks in f32 and the reference in doubles, so a filter
 * that changed its mind across a triangle would flip at slightly different
 * points and the comparison would flake there.
 *
 * The reference has no flags and asks its filter about every triangle, so each
 * TypeScript filter returns true only for triangles the case flags.
 */

import {
    BVH_QUERY_ANY,
    RAY_MASK_ALL,
    TRIANGLE_FILTER_SIGNATURE_WGSL,
    writeTriangleFilterFlags,
    type TraceSceneData,
    type TraceSceneInstance,
    type TrianglePasses,
} from '../../../../src/trace/index.ts';
import { buildReferenceScene, traceAnyReference, traceClosestReference, type ReferenceHit, type ReferenceOcclusion } from '../../../../src/trace/reference-tracer.ts';
import { buildOracleCases, unpackRay, type CaseSizes, type OracleCase } from './cases.ts';
import type { ReferenceAnswers } from './check.ts';

export interface HookCase {
    readonly name: string;
    readonly description: string;
    /** The scene (instances with their masks), the rays and the ambiguity allowance. */
    readonly base: OracleCase;
    /** Which original triangles get `TRIANGLE_FLAG_FILTER`, in every mesh. */
    readonly flagged: (originalIndex: number) => boolean;
    /** The filter for the CPU walk and the reference; must match `filterWgsl`. */
    readonly passes: TrianglePasses;
    /** `bvhTrianglePasses` for the GPU. */
    readonly filterWgsl: string;
    /** Ray masks traced in a module with ray masks. A module without them traces once, with all ones. */
    readonly masks: readonly number[];
}

/** Instance masks of the masks case, by instance position: none, single bits, several, all. */
const INSTANCE_MASKS = [0x00, 0x01, 0x02, 0x04, 0x03, 0xff, 0x80, 0x5a];

/** Ray masks of the masks case: all ones, 0, single bits, several, and a bit above the instance mask's 8. */
const MASK_CASE_RAY_MASKS = [RAY_MASK_ALL, 0x00, 0x01, 0x06, 0x80, 0x41, 0x100];

const filterWgsl = (body: string): string => `${TRIANGLE_FILTER_SIGNATURE_WGSL} {\n    return ${body};\n}\n`;

/** The hook cases, on the random-instanced scene (four meshes, 30 instances, mirrors) with `sizes.randomRays` rays. */
export function buildHookCases(sizes: CaseSizes): HookCase[] {
    const scene = buildOracleCases(sizes).find((entry) => entry.name === 'random-instanced');
    if (!scene) throw new Error('the random-instanced case is missing');
    const masked: TraceSceneInstance[] = scene.instances.map((instance, index) => ({ ...instance, mask: INSTANCE_MASKS[index % INSTANCE_MASKS.length] }));
    return [
        {
            name: 'filter-flagged',
            description: 'Every third triangle flagged; the filter lets rays through those on even instances, the rest of the flagged ones block.',
            base: scene,
            flagged: (original) => original % 3 === 0,
            passes: (candidate) => candidate.triangle % 3 === 0 && candidate.instance % 2 === 0,
            filterWgsl: filterWgsl('candidate.triangle % 3u == 0u && candidate.instance % 2u == 0u'),
            masks: [RAY_MASK_ALL],
        },
        {
            name: 'filter-query',
            description: 'Every other triangle flagged and casts no shadow: closest-hit sees it, any-hit passes through.',
            base: scene,
            flagged: (original) => original % 2 === 0,
            passes: (candidate) => candidate.triangle % 2 === 0 && candidate.query === BVH_QUERY_ANY,
            filterWgsl: filterWgsl('candidate.triangle % 2u == 0u && candidate.query == BVH_QUERY_ANY'),
            masks: [RAY_MASK_ALL],
        },
        {
            name: 'instance-masks',
            description:
                'Instance masks 0, 1, 2, 4, 3, 0xff, 0x80, 0x5a against ray masks all ones, 0, 1, 6, 0x80, 0x41, 0x100; every fourth triangle flagged and passed when the ray mask has bit 6.',
            base: { ...scene, name: 'instance-masks', instances: masked },
            flagged: (original) => original % 4 === 0,
            passes: (candidate) => candidate.triangle % 4 === 0 && (candidate.rayMask & 0x40) !== 0,
            filterWgsl: filterWgsl('candidate.triangle % 4u == 0u && (candidate.rayMask & 0x40u) != 0u'),
            masks: MASK_CASE_RAY_MASKS,
        },
    ];
}

/** Flags every mesh of an assembled scene as the case says, through each mesh's range. */
export function writeHookFlags(scene: TraceSceneData, hook: HookCase): number {
    let flagged = 0;
    for (const mesh of scene.meshes) flagged += writeTriangleFilterFlags(scene.triangles, hook.flagged, mesh);
    return flagged;
}

/** The ray masks a case is traced with in a module with or without ray masks. */
export function hookMasks(hook: HookCase, rayMasks: boolean): readonly number[] {
    return rayMasks ? hook.masks : [RAY_MASK_ALL];
}

/** The reference's answers for one case, mask and mode, handing control back to the browser now and then. */
export async function hookReferenceAnswers(hook: HookCase, mask: number, rayMasks: boolean, yieldEvery = 500): Promise<ReferenceAnswers> {
    const scene = buildReferenceScene(hook.base.meshes, hook.base.instances, { triangleFilter: hook.passes, rayMasks });
    const rayCount = hook.base.rays.length / 8;
    const closest: (ReferenceHit | null)[] = [];
    const any: ReferenceOcclusion[] = [];
    for (let index = 0; index < rayCount; index++) {
        const ray = unpackRay(hook.base.rays, index);
        closest.push(traceClosestReference(scene, ray, { mask }));
        any.push(traceAnyReference(scene, ray, { mask }));
        if (index % yieldEvery === yieldEvery - 1) await new Promise((resolve) => setTimeout(resolve, 0));
    }
    return { closest, any };
}

/** A label for one traced pass of a case. */
export function hookPassName(hook: HookCase, mask: number, rayMasks: boolean): string {
    return rayMasks ? `${hook.name} mask 0x${mask.toString(16)}` : `${hook.name} masks off`;
}

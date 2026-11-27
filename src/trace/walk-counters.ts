/**
 * The counted walk: the production walk with step counters, compiled in only
 * when asked (`traceModuleWgsl({ counters: true })`,
 * `TraceBatchKernel.create(device, { counters: true })`). For heat maps and
 * tools that ask why one tree is faster than another; the benches' counted
 * copy (tests/gpu/pages/trace-bench/counted-walk-wgsl.ts) counts the same
 * things the same way.
 *
 * **Not retyped.** The counted functions are cut from the production walk's
 * own text (wgsl.ts `walkWgsl`, the same options) and changed by a fixed list
 * of insertions, each at an anchor that must occur exactly once (or the
 * stated number of times). A change to the walk either flows into the counted
 * copy or throws here (and the unit tests fail). Traversal order is the
 * production walk's, so the counted walk finds the same hits.
 *
 * **The production walk is untouched.** Without the option nothing is added:
 * the module is the same text; with it, the module gains the declarations
 * below and every other declaration is the same text (wgsl.test.ts checks
 * both). The counted walk's speed means nothing (it carries more state and
 * writes more per ray); time the production walk.
 */

import { boundedLoopWgsl } from './bounded-loop.ts';

/** Bytes of one `BvhWalkCounts` record (nine 4-byte words). */
export const WALK_COUNTS_BYTES = 36;

/** Where each word of a `BvhWalkCounts` record sits ({@link WALK_COUNTS_BYTES} per ray). */
export const WALK_COUNTS_WORD = {
    /** Closest: the hit's `t` (f32), negative for a miss. Any: 1.0 blocked, -1.0 clear. */
    t: 0,
    /** Closest: the hit's instance (a merged mesh's triangles report their source instance). */
    instance: 1,
    /** Closest: the hit's triangle within its source mesh. */
    triangle: 2,
    /** Iterations of the walk's loop: one per node or leaf processed, both levels. */
    steps: 3,
    /** Inner nodes of mesh trees whose two child boxes were tested. */
    meshNodes: 4,
    /** Inner nodes of the top level whose two child boxes were tested. */
    topLevelNodes: 5,
    /** Instances entered (top-level leaves whose mask passed; a merged mesh counts once). */
    instanceEntries: 6,
    /** Ray-triangle tests. */
    triangleTests: 7,
    /** Instances entered in which the walk accepted a hit (closest) or found the blocker (any). */
    instancesWithHit: 8,
} as const;

/** The record, its finishers, and the public counted entry points (with `*Masked` forms when `rayMasks`). */
export function walkCountsTypesWgsl(rayMasks: boolean): string {
    const masked = rayMasks
        ? `
// As bvhTraceClosestCounted, but instances whose mask shares no bit with 'mask' are skipped.
fn bvhTraceClosestCountedMasked(ray: BvhRay, mask: u32) -> BvhWalkCounts {
    return bvhWalkClosestCounted(ray, mask);
}

// As bvhTraceAnyCounted, but instances whose mask shares no bit with 'mask' are skipped.
fn bvhTraceAnyCountedMasked(ray: BvhRay, mask: u32) -> BvhWalkCounts {
    return bvhWalkAnyCounted(ray, mask);
}
`
        : '';
    const all = rayMasks ? ', BVH_RAY_MASK_ALL' : '';
    return /* wgsl */ `
// ---- The counted walk (counters: true): the walk's work per ray, for heat maps and tools ----

// t, instance, triangle as the hit (any: t is 1.0 blocked, -1.0 clear), then the counts (WALK_COUNTS_WORD).
struct BvhWalkCounts {
    t: f32,
    instance: u32,
    triangle: u32,
    steps: u32,
    meshNodes: u32,
    topLevelNodes: u32,
    instanceEntries: u32,
    triangleTests: u32,
    instancesWithHit: u32,
}

fn bvhCountedFinishClosest(countsIn: BvhWalkCounts, hit: BvhHit) -> BvhWalkCounts {
    var counts = countsIn;
    counts.t = hit.t;
    counts.instance = hit.instance;
    counts.triangle = hit.triangle;
    return counts;
}

fn bvhCountedFinishAny(countsIn: BvhWalkCounts, blocked: bool) -> BvhWalkCounts {
    var counts = countsIn;
    counts.t = select(-1.0, 1.0, blocked);
    return counts;
}

// The closest-hit walk with its work counted (the same walk, the same hits).
fn bvhTraceClosestCounted(ray: BvhRay) -> BvhWalkCounts {
    return bvhWalkClosestCounted(ray${all});
}

// The any-hit walk with its work counted.
fn bvhTraceAnyCounted(ray: BvhRay) -> BvhWalkCounts {
    return bvhWalkAnyCounted(ray${all});
}
${masked}`;
}

/** Replaces `anchor`, which must occur exactly `times` times in `text`. */
function replaceAnchor(text: string, anchor: string, replacement: string, times = 1): string {
    const found = text.split(anchor).length - 1;
    if (found !== times) throw new Error(`@pmndrs/bvh: the counted walk is out of step with the walk: expected ${times} of "${anchor}", found ${found}`);
    return text.split(anchor).join(replacement);
}

/**
 * The counted copy of one production walk function (`walk`: its text as
 * wgsl.ts writes it, `bvhWalkClosest` or `bvhWalkAny`), renamed
 * `bvhWalkClosestCounted` / `bvhWalkAnyCounted` and returning `BvhWalkCounts`.
 */
export function countedWalkWgsl(walk: string, kind: 'closest' | 'any', rayMasks: boolean): string {
    const closest = kind === 'closest';
    const name = closest ? 'bvhWalkClosest' : 'bvhWalkAny';
    const returnType = closest ? 'BvhHit' : 'bool';
    const parameters = rayMasks ? 'ray: BvhRay, rayMask: u32' : 'ray: BvhRay';
    let text = replaceAnchor(
        walk,
        `fn ${name}(${parameters}) -> ${returnType} {`,
        `fn ${name}Counted(${parameters}) -> BvhWalkCounts {\n    var counts = BvhWalkCounts();\n    var hitInThisInstance = false;`,
    );
    // The descent loop's head is bounded-loop.ts's, counter bvhStep.
    const descent = `    ${boundedLoopWgsl('bvhStep')}\n        let count = current >> BVH_REF_COUNT_SHIFT;`;
    text = replaceAnchor(text, descent, descent.replace('\n', '\n        counts.steps += 1u;\n'));
    // The folded decode (quantizedDecode: 'folded') loads mesh nodes with bvhLoadMeshNodeFolded.
    const meshLoad = walk.includes('node = bvhLoadMeshNodeFolded(') ? 'node = bvhLoadMeshNodeFolded(instance.rootNodeOffset + index);' : 'node = bvhLoadMeshNode(instance.rootNodeOffset + index);';
    text = replaceAnchor(text, meshLoad, `${meshLoad}\n                counts.meshNodes += 1u;`);
    text = replaceAnchor(text, 'node = bvhLoadTopLevelNode(firstTopLevelNode, index);', 'node = bvhLoadTopLevelNode(firstTopLevelNode, index);\n                counts.topLevelNodes += 1u;');
    text = replaceAnchor(text, 'stack[stackSize] = BVH_REF_INSTANCE_MARKER;', 'stack[stackSize] = BVH_REF_INSTANCE_MARKER;\n                counts.instanceEntries += 1u;\n                hitInThisInstance = false;');
    text = replaceAnchor(text, 'let triangle = bvhLoadTriangle(slot, meshVertexBase);', 'let triangle = bvhLoadTriangle(slot, meshVertexBase);\n                counts.triangleTests += 1u;');
    if (closest) {
        text = replaceAnchor(text, 'result = BvhHit(triangleHit.t', 'if (!hitInThisInstance) { counts.instancesWithHit += 1u; hitInThisInstance = true; }\n                    result = BvhHit(triangleHit.t');
        text = replaceAnchor(text, 'return result;', 'return bvhCountedFinishClosest(counts, result);');
        // Twice: at the empty stack, and the final return after the bounded loop (never taken).
        text = replaceAnchor(text, 'return bvhFinishClosest(result, hitRecord, hitSlot);', 'return bvhCountedFinishClosest(counts, bvhFinishClosest(result, hitRecord, hitSlot));', 2);
    } else {
        text = replaceAnchor(text, 'return true;', 'counts.instancesWithHit += 1u;\n                return bvhCountedFinishAny(counts, true);');
        // An empty top level, the empty stack, and the final return after the bounded loop (never taken).
        text = replaceAnchor(text, 'return false;', 'return bvhCountedFinishAny(counts, false);', 3);
    }
    return text;
}

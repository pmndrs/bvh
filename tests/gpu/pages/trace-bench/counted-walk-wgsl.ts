/**
 * A bench-only counted copy of the production walk (`bvhWalkClosest` and
 * `bvhWalkAny` of `src/trace/wgsl.ts`, the masked build with the merged-instance
 * select): the same loop with counters added, to see why one tree is faster
 * than the other. Counters are for the bench only; production shaders carry no
 * test hooks.
 *
 * The copy is not retyped. It is cut out of the production module's text and
 * changed by a fixed list of insertions, each at an anchor that must occur
 * exactly once (or the stated number of times), so a change to the production
 * walk either flows into the copy or fails loudly here. With `counters: false`
 * nothing is inserted and the text is the production function's, character for
 * character (`counted-walk.test.ts` checks that), so traversal order is the
 * production walk's. The page also checks that the counted kernel finds the
 * production kernel's hits on the real rays.
 *
 * The counted walk's times mean nothing (it writes more words per ray); the
 * bench times the production kernel.
 */

import { traceModuleWgsl } from '../../../../src/trace/wgsl.ts';

/** Words per ray the counted kernel writes, in the order of {@link COUNTED_WORD}. */
export const COUNTED_WORDS = 9;

/** Where each counter sits in a ray's {@link COUNTED_WORDS} words. */
export const COUNTED_WORD = {
    /** The closest hit's `t` (f32 bits), negative for a miss; any-hit: 1.0 blocked, -1.0 clear. */
    t: 0,
    /** The hit's instance (a merged mesh's triangles report their source instance). */
    instance: 1,
    /** The hit's triangle within its source mesh. */
    triangle: 2,
    /** Iterations of the walk's loop: one per node or leaf processed. */
    steps: 3,
    /** Inner nodes of a mesh tree whose two child boxes were tested. */
    meshNodes: 4,
    /** Inner nodes of the top level whose two child boxes were tested. */
    topLevelNodes: 5,
    /** Instances entered (top-level leaves reached whose mask passed). A merged mesh counts once. */
    instanceEntries: 6,
    /** Ray-triangle tests. */
    triangleTests: 7,
    /** Instances entered in which the walk accepted at least one hit (closest) or found the blocker (any). */
    instancesWithHit: 8,
} as const;

/** The text of one of the production walk functions, as `traceModuleWgsl` writes it (masks on). */
export function productionWalkWgsl(kind: 'closest' | 'any'): string {
    const name = kind === 'closest' ? 'bvhWalkClosest' : 'bvhWalkAny';
    const module = traceModuleWgsl({ rayMasks: true });
    const start = module.indexOf(`fn ${name}(`);
    if (start < 0) throw new Error(`the trace module has no ${name}`);
    const end = module.indexOf('\n}\n', start);
    if (end < 0) throw new Error(`could not find the end of ${name}`);
    return module.slice(start, end + 2);
}

/** Replaces `anchor` (which must occur exactly `times` times) with `replacement`. */
function replaceAnchor(text: string, anchor: string, replacement: string, times = 1): string {
    const found = text.split(anchor).length - 1;
    if (found !== times) throw new Error(`the production walk changed: expected ${times} of "${anchor}", found ${found}`);
    return text.split(anchor).join(replacement);
}

export interface CountedWalkOptions {
    kind: 'closest' | 'any';
    /** The function's name. */
    name: string;
    /** Add the counters (the function then returns `BenchCounts`). False: the production function, only renamed. */
    counters: boolean;
}

/** The production walk with counters added (or, with `counters: false`, as it is). */
export function countedWalkWgsl(options: CountedWalkOptions): string {
    const { kind, name, counters } = options;
    const closest = kind === 'closest';
    const productionName = closest ? 'bvhWalkClosest' : 'bvhWalkAny';
    let text = productionWalkWgsl(kind);
    if (!counters) return replaceAnchor(text, `fn ${productionName}(`, `fn ${name}(`);

    const returnType = closest ? 'BvhHit' : 'bool';
    text = replaceAnchor(
        text,
        `fn ${productionName}(ray: BvhRay, rayMask: u32) -> ${returnType} {`,
        `fn ${name}(ray: BvhRay, rayMask: u32) -> BenchCounts {\n    var counts = BenchCounts();\n    var hitInThisInstance = false;`,
    );
    text = replaceAnchor(text, '    loop {\n        let count = current >> BVH_REF_COUNT_SHIFT;', '    loop {\n        counts.steps += 1u;\n        let count = current >> BVH_REF_COUNT_SHIFT;');
    text = replaceAnchor(text, 'node = bvhLoadMeshNode(instance.rootNodeOffset + index);', 'node = bvhLoadMeshNode(instance.rootNodeOffset + index);\n                counts.meshNodes += 1u;');
    text = replaceAnchor(text, 'node = bvhLoadTopLevelNode(firstTopLevelNode, index);', 'node = bvhLoadTopLevelNode(firstTopLevelNode, index);\n                counts.topLevelNodes += 1u;');
    text = replaceAnchor(text, 'stack[stackSize] = BVH_REF_INSTANCE_MARKER;', 'stack[stackSize] = BVH_REF_INSTANCE_MARKER;\n                counts.instanceEntries += 1u;\n                hitInThisInstance = false;');
    text = replaceAnchor(
        text,
        'let triangle = bvhLoadTriangle(instance.triangleOffset + index + i);',
        'let triangle = bvhLoadTriangle(instance.triangleOffset + index + i);\n                counts.triangleTests += 1u;',
    );
    if (closest) {
        text = replaceAnchor(text, 'result = BvhHit(triangleHit.t', 'if (!hitInThisInstance) { counts.instancesWithHit += 1u; hitInThisInstance = true; }\n                    result = BvhHit(triangleHit.t');
        text = replaceAnchor(text, 'return result;', 'return benchFinishClosest(counts, result);', 2);
    } else {
        text = replaceAnchor(text, 'return true;', 'counts.instancesWithHit += 1u;\n                return benchFinishAny(counts, true);');
        text = replaceAnchor(text, 'return false;', 'return benchFinishAny(counts, false);', 2);
    }
    return text;
}

/** The counters' record and the functions that finish a walk. */
const BENCH_HELPERS_WGSL = /* wgsl */ `
struct BenchCounts {
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

fn benchFinishClosest(countsIn: BenchCounts, hit: BvhHit) -> BenchCounts {
    var counts = countsIn;
    counts.t = hit.t;
    counts.instance = hit.instance;
    counts.triangle = hit.triangle;
    return counts;
}

fn benchFinishAny(countsIn: BenchCounts, blocked: bool) -> BenchCounts {
    var counts = countsIn;
    counts.t = select(-1.0, 1.0, blocked);
    return counts;
}
`;

/**
 * The counted batch kernel: the production trace module in group 0 (masked
 * build, every ray with all mask bits), then in group 1 the rays and the
 * counters out, one {@link COUNTED_WORDS}-word record per ray. Dispatches are
 * 2D, as the production batch kernel's.
 */
export function countedKernelWgsl(mode: 'closest' | 'any'): string {
    return /* wgsl */ `${traceModuleWgsl({ rayMasks: true })}
${BENCH_HELPERS_WGSL}
${countedWalkWgsl({ kind: mode, name: 'benchCountedWalk', counters: true })}

@group(1) @binding(0) var<storage, read> benchRays: array<BvhRay>;
@group(1) @binding(1) var<storage, read_write> benchCounts: array<BenchCounts>;

@compute @workgroup_size(64)
fn benchCounted(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroupCount: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let rayIndex = (workgroup.y * workgroupCount.x + workgroup.x) * 64u + localIndex;
    if (rayIndex >= arrayLength(&benchRays)) {
        return;
    }
    benchCounts[rayIndex] = benchCountedWalk(benchRays[rayIndex], BVH_RAY_MASK_ALL);
}
`;
}

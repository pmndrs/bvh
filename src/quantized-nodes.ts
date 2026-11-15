/**
 * Quantized nodes (docs/internals/design/compressed-nodes.md, option A): 32-byte
 * two-child nodes whose child boxes are 16-bit planes on a per-mesh grid,
 * beside the 64-byte `f32` nodes. Every builder writes `f32` nodes; these
 * helpers pack them (and unpack them for the CPU).
 *
 * **The frame.** Each mesh gets a grid per axis: a power-of-two `step` and an
 * `origin` that is a multiple of it ({@link nodeFrameOf}, from the root's box).
 * The step is the smallest power of two for which (a) every coordinate of the
 * box is under 2^23 steps from 0 and (b) the box spans at most
 * {@link FRAME_MAX_SPAN} steps. (a) makes every grid plane `origin + q * step`
 * an exact `f32`, so a fused or unfused decode, `f32` or `f64`, CPU or GPU,
 * gives the same bits. (b) leaves room for the margin below. The rule is
 * monotone in the step, so the smallest step is one number however it is
 * searched for; the GPU passes (gpu-quantize.ts) find the same one.
 *
 * **Rounding outward.** A min plane is rounded down to a grid point, a max
 * plane up, and a plane that lands closer than 1/1024 of a step to the exact
 * value moves one more step out (B3's rule: tight planes let 1.7% of
 * edge-aimed rays through on Intel). Every test is on values `x / step`,
 * which are exact, so the CPU and GPU encoders agree bit for bit too. A
 * decoded box always contains the exact box; a looser box costs node visits,
 * never hits.
 *
 * **An empty box** (min above max, a point at the largest float as the refit
 * stores one, or anything not finite) encodes as both planes at the grid's
 * top, {@link QUANTIZED_EMPTY_AXIS}: its leaf has nothing a ray can hit, so
 * any box is right. A missing right child keeps the reference
 * `GPU_LAYOUT_EMPTY_CHILD`, which the walk tests before it tests the box.
 */

import {
    GPU_LAYOUT_CHILD_STRIDE_WORDS,
    GPU_LAYOUT_EMPTY_CHILD,
    GPU_LAYOUT_NODE_BYTES,
    GPU_LAYOUT_NODE_FRAME_WORD,
    GPU_LAYOUT_NODE_WORD,
    GPU_LAYOUT_NODE_WORDS,
    GPU_LAYOUT_QUANTIZED_NODE_BYTES,
    GPU_LAYOUT_QUANTIZED_NODE_WORDS,
    type GpuLayoutMeshBvh,
    type GpuLayoutNodeFormat,
} from './gpu-layout.ts';

/** Bits of the reference word below the leaf count (`REF_COUNT_SHIFT` in gpu-layout.ts). */
const REFERENCE_COUNT_SHIFT = 28;
const REFERENCE_INDEX_MASK = 0x0fffffff;

/** Most steps a frame's box may span on an axis: the margin needs one step below and two above, in 16 bits. */
export const FRAME_MAX_SPAN = 65532;
/** Coordinates must stay under this many steps from 0, so every grid plane is an exact `f32`. */
export const FRAME_EXACT_STEPS = 2 ** 23;
/** The smallest step a frame uses (2^-100): far above `f32`'s smallest normal, so nothing on the grid is subnormal. */
export const FRAME_MIN_EXPONENT = -100;
/** Boxes with a coordinate this large or larger (or not finite) count as empty; a frame over one is the unit grid at 0. */
export const FRAME_LIMIT = 2 ** 100;
/** A plane closer than this many steps to the exact value moves one more step out. */
export const QUANTIZE_MARGIN_STEPS = 1 / 1024;
/** The axis word of an empty box: min and max both at the grid's top. */
export const QUANTIZED_EMPTY_AXIS = 0xffffffff;

/** One axis of a frame: the grid's origin and step, and the step's exponent. */
export interface NodeFrame {
    /** `origin[axis]`, an exact `f32`, a multiple of `step[axis]`. */
    readonly origin: readonly [number, number, number];
    /** `step[axis]`, a power of two. */
    readonly step: readonly [number, number, number];
    /** `log2(step[axis])`. */
    readonly exponent: readonly [number, number, number];
}

/** Checks a `nodeFormat` option: undefined is `fallback` (`'f32'` for a mesh builder, `DEFAULT_NODE_FORMAT` for a scene). */
export function checkNodeFormat(format: GpuLayoutNodeFormat | undefined, where: string, fallback: GpuLayoutNodeFormat = 'f32'): GpuLayoutNodeFormat {
    if (format === undefined) return fallback;
    if (format === 'f32') return 'f32';
    if (format === 'quantized') return 'quantized';
    throw new RangeError(`${where}: nodeFormat must be 'f32' or 'quantized', got ${String(format)}`);
}

/** A box `[minX, minY, minZ, maxX, maxY, maxZ]` (or a node's stored child box) that holds nothing a ray can reach. */
function isEmptyAxis(min: number, max: number): boolean {
    return !(min <= max) || !(Math.abs(min) < FRAME_LIMIT) || !(Math.abs(max) < FRAME_LIMIT);
}

/** `floor(log2(x))` for a positive finite x, or one below it (only a lower bound is needed). */
function exponentBelow(x: number): number {
    return Math.floor(Math.log2(x)) - 1;
}

function axisFrame(lo: number, hi: number): { origin: number; step: number; exponent: number } {
    if (isEmptyAxis(lo, hi)) return { origin: 0, step: 1, exponent: 0 };
    // Any start at or below the answer gives the answer: the rule is monotone in the exponent.
    let exponent = FRAME_MIN_EXPONENT;
    const largest = Math.max(Math.abs(lo), Math.abs(hi));
    if (largest > 0) exponent = Math.max(exponent, exponentBelow(largest) - 24);
    if (hi > lo) exponent = Math.max(exponent, exponentBelow(hi - lo) - 17);
    for (;;) {
        const step = 2 ** exponent;
        const exact = Math.abs(lo) < FRAME_EXACT_STEPS * step && Math.abs(hi) < FRAME_EXACT_STEPS * step;
        if (exact && Math.ceil(hi / step) - Math.floor(lo / step) <= FRAME_MAX_SPAN) {
            return { origin: (Math.floor(lo / step) - 1) * step, step, exponent };
        }
        exponent++;
    }
}

/**
 * The frame of a box `[minX, minY, minZ, maxX, maxY, maxZ]` (the union of
 * the root's children): see the file comment. An axis that is empty or out of
 * range gets the unit grid at 0.
 */
export function nodeFrameOf(box: ArrayLike<number>): NodeFrame {
    const axes = [0, 1, 2].map((axis) => axisFrame(box[axis], box[axis + 3]));
    return {
        origin: [axes[0].origin, axes[1].origin, axes[2].origin],
        step: [axes[0].step, axes[1].step, axes[2].step],
        exponent: [axes[0].exponent, axes[1].exponent, axes[2].exponent],
    };
}

/**
 * One axis of a box as an axis word (min plane in bits 0-15, max in 16-31),
 * rounded outward with the margin. Throws if the box does not fit the frame
 * (a box outside the root's).
 */
export function quantizeAxis(min: number, max: number, origin: number, step: number): number {
    if (isEmptyAxis(min, max)) return QUANTIZED_EMPTY_AXIS;
    const base = origin / step;
    const low = min / step;
    const high = max / step;
    let qMin = Math.floor(low) - base;
    if (low - Math.floor(low) < QUANTIZE_MARGIN_STEPS) qMin--;
    let qMax = Math.ceil(high) - base;
    if (Math.ceil(high) - high < QUANTIZE_MARGIN_STEPS) qMax++;
    if (qMin < 0 || qMax > 0xffff) {
        throw new RangeError(`quantizeNodes: a child box [${min}, ${max}] does not fit the frame at ${origin} with step ${step}; is it outside the root's box?`);
    }
    return (qMin | (qMax << 16)) >>> 0;
}

/** The decoded min plane of an axis word. Exact: `origin + q * step` is an `f32` by the frame rule. */
export function decodeMinPlane(word: number, origin: number, step: number): number {
    return origin + (word & 0xffff) * step;
}

/** The decoded max plane of an axis word. */
export function decodeMaxPlane(word: number, origin: number, step: number): number {
    return origin + (word >>> 16) * step;
}

/** The union of a 64-byte node's non-empty child boxes, `[min x3, max x3]` (the frame's box). */
export function nodeChildUnion(floats: Float32Array, words: Uint32Array, nodeWord: number): Float64Array {
    const box = new Float64Array([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
    for (let side = 0; side < 2; side++) {
        const indexWord = side === 0 ? GPU_LAYOUT_NODE_WORD.leftIndex : GPU_LAYOUT_NODE_WORD.rightIndex;
        if (words[nodeWord + indexWord] === GPU_LAYOUT_EMPTY_CHILD) continue;
        const at = nodeWord + side * GPU_LAYOUT_CHILD_STRIDE_WORDS;
        for (let axis = 0; axis < 3; axis++) {
            const min = floats[at + axis];
            const max = floats[at + 4 + axis];
            if (isEmptyAxis(min, max)) continue;
            box[axis] = Math.min(box[axis], min);
            box[axis + 3] = Math.max(box[axis + 3], max);
        }
    }
    return box;
}

/** Writes a frame slot (8 words) at `out[wordOffset..]`. */
export function writeNodeFrame(out: Uint32Array, wordOffset: number, frame: NodeFrame): void {
    const floats = new Float32Array(out.buffer, out.byteOffset, out.length);
    for (let axis = 0; axis < 3; axis++) {
        floats[wordOffset + GPU_LAYOUT_NODE_FRAME_WORD.origin + axis] = frame.origin[axis];
        floats[wordOffset + GPU_LAYOUT_NODE_FRAME_WORD.step + axis] = frame.step[axis];
    }
    out[wordOffset + 3] = 0;
    out[wordOffset + 7] = 0;
}

/** Reads a frame slot at `words[wordOffset..]`. */
export function readNodeFrame(words: Uint32Array, wordOffset: number): NodeFrame {
    const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
    const origin = [0, 1, 2].map((axis) => floats[wordOffset + GPU_LAYOUT_NODE_FRAME_WORD.origin + axis]);
    const step = [0, 1, 2].map((axis) => floats[wordOffset + GPU_LAYOUT_NODE_FRAME_WORD.step + axis]);
    return {
        origin: [origin[0], origin[1], origin[2]],
        step: [step[0], step[1], step[2]],
        exponent: [Math.log2(step[0]), Math.log2(step[1]), Math.log2(step[2])],
    };
}

/**
 * Packs `nodeCount` 64-byte nodes (`floats`/`words` from `nodeWordOffset`,
 * node 0 the root) into a frame slot and `nodeCount` 32-byte nodes at
 * `out[outWordOffset..]`. The frame comes from the root's children. This is
 * both the builders' packing and the CPU half of a refit's re-quantize: refit
 * the 64-byte copy (`refitOnCpu`), then pack it again. Returns the frame.
 */
export function quantizeNodeWords(
    floats: Float32Array,
    words: Uint32Array,
    nodeWordOffset: number,
    nodeCount: number,
    out: Uint32Array,
    outWordOffset: number,
): NodeFrame {
    const frame = nodeFrameOf(nodeCount > 0 ? nodeChildUnion(floats, words, nodeWordOffset) : [0, 0, 0, -1, -1, -1]);
    writeNodeFrame(out, outWordOffset, frame);
    for (let node = 0; node < nodeCount; node++) {
        const from = nodeWordOffset + node * GPU_LAYOUT_NODE_WORDS;
        const to = outWordOffset + (node + 1) * GPU_LAYOUT_QUANTIZED_NODE_WORDS;
        for (let side = 0; side < 2; side++) {
            const index = words[from + (side === 0 ? GPU_LAYOUT_NODE_WORD.leftIndex : GPU_LAYOUT_NODE_WORD.rightIndex)];
            const count = words[from + (side === 0 ? GPU_LAYOUT_NODE_WORD.leftCount : GPU_LAYOUT_NODE_WORD.rightCount)];
            const box = from + side * GPU_LAYOUT_CHILD_STRIDE_WORDS;
            const at = to + side * 4;
            if (index === GPU_LAYOUT_EMPTY_CHILD) {
                out[at] = out[at + 1] = out[at + 2] = QUANTIZED_EMPTY_AXIS;
                out[at + 3] = GPU_LAYOUT_EMPTY_CHILD;
                continue;
            }
            for (let axis = 0; axis < 3; axis++) out[at + axis] = quantizeAxis(floats[box + axis], floats[box + 4 + axis], frame.origin[axis], frame.step[axis]);
            if (index > REFERENCE_INDEX_MASK || count >= 15) throw new RangeError(`quantizeNodes: node ${node} side ${side} (index ${index}, count ${count}) does not fit a reference`);
            out[at + 3] = ((count << REFERENCE_COUNT_SHIFT) | index) >>> 0;
        }
    }
    return frame;
}

/**
 * Unpacks a frame slot and `nodeCount` quantized nodes at
 * `words[wordOffset..]` into 64-byte nodes at `outFloats`/`outWords` from
 * `outWordOffset`: each child box decoded (looser than the exact box, always
 * around it), each reference split into index and count.
 */
export function dequantizeNodeWords(words: Uint32Array, wordOffset: number, nodeCount: number, outFloats: Float32Array, outWords: Uint32Array, outWordOffset: number): void {
    const frame = readNodeFrame(words, wordOffset);
    for (let node = 0; node < nodeCount; node++) {
        decodeQuantizedNode(words, wordOffset + (node + 1) * GPU_LAYOUT_QUANTIZED_NODE_WORDS, frame, outFloats, outWords, outWordOffset + node * GPU_LAYOUT_NODE_WORDS);
    }
}

/** One quantized node at `words[nodeWord..]` as a 64-byte node at `outFloats`/`outWords[outWord..]`. */
export function decodeQuantizedNode(words: Uint32Array, nodeWord: number, frame: NodeFrame, outFloats: Float32Array, outWords: Uint32Array, outWord: number): void {
    for (let side = 0; side < 2; side++) {
        const at = nodeWord + side * 4;
        const box = outWord + side * GPU_LAYOUT_CHILD_STRIDE_WORDS;
        for (let axis = 0; axis < 3; axis++) {
            const word = words[at + axis];
            outFloats[box + axis] = decodeMinPlane(word, frame.origin[axis], frame.step[axis]);
            outFloats[box + 4 + axis] = decodeMaxPlane(word, frame.origin[axis], frame.step[axis]);
        }
        const reference = words[at + 3];
        const empty = reference === GPU_LAYOUT_EMPTY_CHILD;
        outWords[outWord + (side === 0 ? GPU_LAYOUT_NODE_WORD.leftIndex : GPU_LAYOUT_NODE_WORD.rightIndex)] = empty ? GPU_LAYOUT_EMPTY_CHILD : reference & REFERENCE_INDEX_MASK;
        outWords[outWord + (side === 0 ? GPU_LAYOUT_NODE_WORD.leftCount : GPU_LAYOUT_NODE_WORD.rightCount)] = empty ? 0 : reference >>> REFERENCE_COUNT_SHIFT;
    }
}

/**
 * The same tree with quantized nodes: the frame slot, then 32 bytes per node
 * (`nodes` is a new buffer; the rest is shared). A quantized tree comes back
 * as it is.
 */
export function quantizeNodes<T extends Pick<GpuLayoutMeshBvh, 'nodes' | 'nodeFormat' | 'nodeCount'>>(layout: T): T {
    if (layout.nodeFormat === 'quantized') return layout;
    const nodes = new ArrayBuffer((layout.nodeCount + 1) * GPU_LAYOUT_QUANTIZED_NODE_BYTES);
    const source = layout.nodes;
    const count = layout.nodeCount * GPU_LAYOUT_NODE_WORDS;
    quantizeNodeWords(new Float32Array(source, 0, count), new Uint32Array(source, 0, count), 0, layout.nodeCount, new Uint32Array(nodes), 0);
    return { ...layout, nodes, nodeFormat: 'quantized' } as T;
}

/**
 * The same tree with 64-byte nodes whose boxes are the quantized ones decoded
 * (they contain the exact boxes; a refit makes them exact again). An `f32`
 * tree comes back as it is. CPU trees (`CpuMeshBvh`) use this.
 */
export function dequantizeNodes<T extends Pick<GpuLayoutMeshBvh, 'nodes' | 'nodeFormat' | 'nodeCount'>>(layout: T): T {
    if (layout.nodeFormat !== 'quantized') return layout;
    const nodes = new ArrayBuffer(layout.nodeCount * GPU_LAYOUT_NODE_BYTES);
    dequantizeNodeWords(new Uint32Array(layout.nodes), 0, layout.nodeCount, new Float32Array(nodes), new Uint32Array(nodes), 0);
    const { nodeFormat: _format, ...rest } = layout;
    return { ...rest, nodes } as unknown as T;
}

/**
 * What a builder hands back for its `nodeFormat` option: its `f32` tree,
 * quantized when asked. (A quantized tree is never turned back into `f32`
 * here: decoding loses the exact boxes.)
 */
export function builderNodeFormat<T extends GpuLayoutMeshBvh>(layout: T, format: GpuLayoutNodeFormat | undefined, where: string): T {
    return checkNodeFormat(format, where) === 'quantized' ? quantizeNodes(layout) : layout;
}

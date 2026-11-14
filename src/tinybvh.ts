/**
 * tinybvh's default triangle builder (SIMD Bonsai), compiled to WebAssembly (stage B0 of
 * docs/internals/design/bvh-framework.md). Its scalar H-PLOC builder can be asked for
 * instead (`builder: 'hploc'`): the oracle the GPU H-PLOC (B2) is compared with.
 *
 * The .wasm is built from native/tinybvh-wasm.cpp by native/build-wasm.ts and
 * stored base64-encoded in tinybvh-wasm.ts, so it loads the same way in a
 * worker, on a page, in Node and in any bundler. It is single-threaded: run it
 * in a worker for big meshes.
 *
 * Two outputs from the same tree:
 * - `build`: the tree's shape (`MeshTreeTopology`, the Builder contract's plain description);
 * - `buildLayout` (stage B0b): the tracer's GPU layout, written inside the WASM.
 *   It replaces a JavaScript pass over every node and triangle, which was
 *   memory-bound and slow on big meshes (timings: PROGRESS.md, B0b). Indexed
 *   triangles and quantized nodes are written there too (`tbvh_write_layout_as`),
 *   when the .wasm has it ({@link TinyBvhBuilder.formatsInWasm}).
 *
 * tinybvh (https://github.com/jbikker/tinybvh, MIT, Jacco Bikker). Commit and
 * hashes: native/build-wasm.ts and THIRD_PARTY.md. Licence texts (tinybvh's and
 * the Emscripten runtime's): licenses/, also embedded at the top of tinybvh-wasm.ts.
 */

import {
    GPU_LAYOUT_INDEXED_TRIANGLE_BYTES,
    GPU_LAYOUT_MAX_LEAF_PRIMITIVES,
    GPU_LAYOUT_MAX_TRIANGLES,
    GPU_LAYOUT_NODE_BYTES,
    GPU_LAYOUT_QUANTIZED_NODE_BYTES,
    GPU_LAYOUT_TRIANGLE_BYTES,
    type GpuLayoutMeshBvh,
} from './gpu-layout.ts';
import { singleLeafTopology, TOPOLOGY_WORDS_PER_NODE, type MeshTreeTopology } from './topology.ts';
import { TINYBVH_WASM_BASE64 } from './tinybvh-wasm.ts';
import { packedTriangleFlags, type MeshInput } from './mesh-input.ts';
import { builderNodeFormat, checkNodeFormat } from './quantized-nodes.ts';
import { checkTriangleLeaves, indexedTrianglesFromRecordWords } from './indexed-leaves.ts';

/** tinybvh's node: aabbMin (3 floats), leftFirst, aabbMax (3 floats), triCount. 32 bytes. */
const TINYBVH_NODE_WORDS = 8;
const NODE_LEFT_FIRST = 3;
const NODE_TRIANGLE_COUNT = 7;
/** Floats in the mesh box the WASM writes: min x, y, z, then max x, y, z. */
const BOUNDS_FLOATS = 6;

/** The `Status` values of native/tinybvh-wasm.cpp. */
const BUILD_STATUS = {
    ok: 0,
    noTriangles: 1,
    outOfMemory: 2,
    indexOutOfRange: 3,
    nothingBuilt: 4,
    badLeafLimit: 5,
    badOrder: 6,
    badTree: 7,
    tooLarge: 8,
    badBuilder: 9,
    badFormat: 10,
    quantizeFailed: 11,
} as const;

/** The `Builder` values of native/tinybvh-wasm.cpp. */
const BUILDER = { default: 0, hploc: 1 } as const;

/** The `LayoutFormat` bits of native/tinybvh-wasm.cpp (`tbvh_write_layout_as`). */
const LAYOUT_FORMAT = { indexedTriangles: 1, quantizedNodes: 2 } as const;

/**
 * The largest byte size `tbvh_alloc` takes: its argument is a 32-bit WASM
 * integer, so a bigger size would wrap to a small one.
 */
const MAX_ALLOCATION_BYTES = 0xffffffff;

/** What `badTree` found (`TreeProblem` in native/tinybvh-wasm.cpp), read from `tbvh_error_detail()`. */
const TREE_PROBLEMS: Record<number, string> = {
    1: 'a child points past the node pool',
    2: 'a node is reached twice',
    3: 'there are more nodes than the triangles allow',
    4: 'a leaf covers positions past the last triangle',
    5: 'a position is in more than one leaf',
    6: 'a position is in no leaf',
};

interface TinyBvhExports {
    memory: WebAssembly.Memory;
    _initialize(): void;
    tbvh_alloc(bytes: number): number;
    tbvh_free(pointer: number): void;
    tbvh_build(positions: number, vertexCount: number, indices: number, triangleCount: number, builder: number): number;
    tbvh_error_detail(): number;
    tbvh_node_slots(): number;
    tbvh_nodes(): number;
    tbvh_triangle_order(): number;
    tbvh_triangle_order_length(): number;
    tbvh_write_layout(triangleFlags: number, maxLeafPrimitives: number): number;
    /** In .wasm builds since 2026-11-14 only: the layout with indexed triangles and/or quantized nodes ({@link LAYOUT_FORMAT}). */
    tbvh_write_layout_as?(triangleFlags: number, maxLeafPrimitives: number, format: number): number;
    tbvh_layout_nodes(): number;
    tbvh_layout_node_count(): number;
    tbvh_layout_triangles(): number;
    tbvh_layout_parents(): number;
    tbvh_layout_bounds(): number;
    tbvh_layout_max_depth(): number;
    tbvh_layout_leaf_count(): number;
    tbvh_release(): void;
}

export interface TinyBvhBuild {
    readonly topology: MeshTreeTopology;
    /** Everything: the copies in and out of WASM memory, tinybvh's `Build`, and reading its tree, in ms. */
    readonly buildMs: number;
    /** tinybvh's `Build` alone, in ms. */
    readonly wasmBuildMs: number;
}

export interface TinyBvhBuildOptions {
    /**
     * `'default'`: tinybvh's default builder (SIMD Bonsai), the tracer's trees.
     * `'hploc'`: its scalar `BuildHPLOC` (leaves of one triangle), slow, for
     * comparing the GPU H-PLOC's trees with (B2).
     */
    readonly builder?: keyof typeof BUILDER;
}

export interface TinyBvhLayoutOptions extends TinyBvhBuildOptions {
    /** Leaves with more triangles than this are cut in half until they fit. 1 to 8; default 8. */
    readonly maxLeafPrimitives?: number;
    /**
     * `'indexed'`: 16-byte indexed triangles (docs/internals/design/indexed-leaves.md),
     * written inside the WASM (no 48-byte records are made). With a .wasm
     * from before `tbvh_write_layout_as`, or `formatsInWasm: false`, the WASM
     * writes records and they are converted on the way out of its memory.
     * Default `'records'`.
     */
    readonly triangleLeaves?: 'records' | 'indexed';
    /**
     * `'quantized'`: 32-byte nodes (docs/internals/design/compressed-nodes.md), packed
     * inside the WASM after its 64-byte nodes are written (or, as for
     * `triangleLeaves`, in JavaScript on the way out). Default `'f32'`.
     */
    readonly nodeFormat?: 'f32' | 'quantized';
    /**
     * Copy each node's parent out (4 bytes per node). Default true. False (P9
     * lever 7: a tree no refit will use) gives an empty `parents` array; such a
     * tree can be placed and traced, but not refit or made a `CpuMeshBvh`.
     */
    readonly parents?: boolean;
}

export interface TinyBvhLayoutBuild {
    readonly layout: GpuLayoutMeshBvh;
    /** Everything: the copies in and out of WASM memory, tinybvh's `Build`, and writing the layout, in ms. */
    readonly buildMs: number;
    /** tinybvh's `Build` alone, in ms. */
    readonly wasmBuildMs: number;
    /** Writing the GPU layout inside the WASM, in ms. */
    readonly layoutMs: number;
}

export interface TinyBvhBuilder {
    /**
     * Bytes of the WASM memory this builder holds now. A WASM memory grows to
     * a build's peak (tinybvh's working arrays, the input copies and the
     * output, about 230 bytes per triangle) and can never shrink, so after a
     * build that grew it past {@link TinyBvhBuilderOptions.releaseAboveBytes}
     * the builder starts a fresh WASM instance and drops the old one, which
     * the garbage collector then frees.
     */
    readonly heapBytes: number;
    /** WASM instances this builder has dropped after big builds (see {@link TinyBvhBuilderOptions.releaseAboveBytes}). */
    readonly releases: number;
    /** Bytes of the WASM memory at the end of the last build, before any release (0 before the first build). */
    readonly lastBuildHeapBytes: number;
    /**
     * Whether `buildLayout` writes indexed triangles and quantized nodes inside
     * the WASM: the .wasm has `tbvh_write_layout_as` and
     * {@link TinyBvhBuilderOptions.formatsInWasm} is not false. Otherwise they
     * are converted in JavaScript. The bytes are the same either way.
     */
    readonly formatsInWasm: boolean;
    /** Builds one mesh's tree and returns its shape. Synchronous; one at a time. */
    build(geometry: MeshInput, options?: TinyBvhBuildOptions): TinyBvhBuild;
    /**
     * Builds one mesh's tree and returns it in the tracer's GPU layout, written
     * inside the WASM. Byte for byte what ddgi-core's `buildMeshBvhFromTopology`
     * makes from `build`'s topology of the same mesh: the boxes are recomputed
     * from the triangles, not taken from tinybvh. Synchronous; one at a time.
     */
    buildLayout(geometry: MeshInput, options?: TinyBvhLayoutOptions): TinyBvhLayoutBuild;
}

export interface TinyBvhBuilderOptions {
    /**
     * After a build that leaves the WASM memory larger than this many bytes,
     * the builder replaces its WASM instance with a fresh one (64 MiB), so the
     * grown memory is garbage once nothing else holds it (every result is
     * copied out of it). Default {@link TINYBVH_RELEASE_ABOVE_BYTES}, 256 MiB:
     * one build of about a million triangles or more. The next build pays a
     * synchronous instantiation of the compiled module (well under a
     * millisecond for its 130 KB). `Infinity` keeps one instance for the
     * builder's life, as before.
     */
    readonly releaseAboveBytes?: number;
    /**
     * Default true: `buildLayout`'s `triangleLeaves: 'indexed'` and
     * `nodeFormat: 'quantized'` are written inside the WASM when it can
     * (`tbvh_write_layout_as`). False: the WASM writes records and 64-byte
     * nodes and JavaScript converts them, the path before; the same bytes,
     * kept for comparisons and as the path for an older .wasm.
     */
    readonly formatsInWasm?: boolean;
}

/** {@link TinyBvhBuilderOptions.releaseAboveBytes}'s default: 256 MiB. */
export const TINYBVH_RELEASE_ABOVE_BYTES = 256 * 2 ** 20;

let compiled: Promise<WebAssembly.Module> | null = null;

function decodeBase64(text: string): Uint8Array<ArrayBuffer> {
    const binary = atob(text);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
}

/**
 * The imports a standalone Emscripten module asks for. tinybvh only writes to
 * stdout/stderr on a fatal error and then exits, so writes are collected and
 * turned into the error message.
 */
function makeImports(getMemory: () => WebAssembly.Memory, output: string[]): WebAssembly.Imports {
    const decoder = new TextDecoder();
    return {
        env: {
            emscripten_notify_memory_growth: () => {},
        },
        wasi_snapshot_preview1: {
            fd_write: (_fd: number, iovs: number, iovsLength: number, writtenPointer: number): number => {
                const view = new DataView(getMemory().buffer);
                let written = 0;
                for (let i = 0; i < iovsLength; i++) {
                    const pointer = view.getUint32(iovs + i * 8, true);
                    const length = view.getUint32(iovs + i * 8 + 4, true);
                    output.push(decoder.decode(new Uint8Array(getMemory().buffer, pointer, length)));
                    written += length;
                }
                view.setUint32(writtenPointer, written, true);
                return 0;
            },
            fd_close: () => 0,
            fd_seek: () => 70, // ESPIPE: stdout cannot seek
            proc_exit: (code: number): never => {
                throw new Error(`tinybvh stopped (exit ${code}): ${output.join('').trim() || 'no message'}`);
            },
        },
    };
}

/** One started tinybvh: its exports, with the memory the imports write messages from. */
function startInstance(module: WebAssembly.Module): TinyBvhExports {
    const output: string[] = [];
    let memory: WebAssembly.Memory | null = null;
    // Synchronous: the module is compiled already, and small, so this is allowed on a page's main thread too.
    const instance = new WebAssembly.Instance(module, makeImports(() => memory as WebAssembly.Memory, output));
    const wasm = instance.exports as unknown as TinyBvhExports;
    memory = wasm.memory;
    wasm._initialize();
    return wasm;
}

/**
 * Compiles and starts tinybvh. Each builder has its own WASM memory, which a
 * big build grows; past `options.releaseAboveBytes` the builder starts again
 * on a fresh one (see {@link TinyBvhBuilderOptions}).
 */
export async function createTinyBvhBuilder(options: TinyBvhBuilderOptions = {}): Promise<TinyBvhBuilder> {
    const releaseAboveBytes = options.releaseAboveBytes ?? TINYBVH_RELEASE_ABOVE_BYTES;
    if (!(releaseAboveBytes >= 0)) throw new RangeError(`createTinyBvhBuilder: releaseAboveBytes must be a number of bytes (or Infinity), got ${releaseAboveBytes}`);
    compiled ??= WebAssembly.compile(decodeBase64(TINYBVH_WASM_BASE64));
    const module = await compiled;
    let wasm: TinyBvhExports | null = startInstance(module);
    // Capability check: an older .wasm has no tbvh_write_layout_as, and converts in JavaScript.
    const formatsInWasm = options.formatsInWasm !== false && typeof wasm.tbvh_write_layout_as === 'function';
    let releases = 0;
    let lastBuildHeapBytes = 0;
    // The WASM memory is dropped after the build that grew it (the result is a copy), and the
    // next build starts a fresh instance; an idle builder then holds no big memory.
    const run = <T>(build: (exports: TinyBvhExports) => T): T => {
        const exports = (wasm ??= startInstance(module));
        try {
            return build(exports);
        } finally {
            lastBuildHeapBytes = exports.memory.buffer.byteLength;
            if (lastBuildHeapBytes > releaseAboveBytes) {
                wasm = null;
                releases++;
            }
        }
    };
    return {
        get heapBytes() {
            return wasm ? wasm.memory.buffer.byteLength : 0;
        },
        get releases() {
            return releases;
        },
        get lastBuildHeapBytes() {
            return lastBuildHeapBytes;
        },
        formatsInWasm,
        build: (geometry, buildOptions = {}) => run((exports) => buildTopology(exports, geometry, buildOptions)),
        buildLayout: (geometry, buildOptions = {}) => run((exports) => buildGpuLayout(exports, geometry, buildOptions, formatsInWasm)),
    };
}

/** Counts, after checking that the arrays describe whole vertices and triangles. */
function countInput(geometry: MeshInput): { vertexCount: number; triangleCount: number } {
    const { positions, indices } = geometry;
    const vertexCount = positions.length / 3;
    const triangleCount = indices ? indices.length / 3 : vertexCount / 3;
    if (!Number.isInteger(vertexCount) || !Number.isInteger(triangleCount) || triangleCount < 1) {
        throw new RangeError('tinybvh needs at least one triangle, with three floats per vertex and three indices per triangle');
    }
    if (triangleCount > GPU_LAYOUT_MAX_TRIANGLES) throw new RangeError(`too many triangles for one tree: ${triangleCount}`);
    return { vertexCount, triangleCount };
}

/** Turns a failed WASM status into an exception. */
function throwOnStatus(wasm: TinyBvhExports, status: number, geometry: MeshInput, vertexCount: number): void {
    if (status === BUILD_STATUS.ok) return;
    const detail = wasm.tbvh_error_detail();
    switch (status) {
        case BUILD_STATUS.indexOutOfRange:
            throw new RangeError(`index ${detail} is ${geometry.indices?.[detail] ?? detail}, but there are ${vertexCount} vertices`);
        case BUILD_STATUS.outOfMemory:
            throw new Error('tinybvh: out of WASM memory');
        case BUILD_STATUS.tooLarge:
            throw new RangeError('tinybvh: the mesh is too big for 32-bit WebAssembly (one of its arrays would be 4 GB or more)');
        case BUILD_STATUS.badOrder:
            throw new Error('tinybvh ordered its triangles wrongly: not every triangle exactly once');
        case BUILD_STATUS.badTree:
            throw new Error(`tinybvh's nodes do not form a tree over every triangle: ${TREE_PROBLEMS[detail] ?? `problem ${detail}`}`);
        case BUILD_STATUS.quantizeFailed:
            throw new RangeError(`tinybvh: node ${detail} does not fit the quantized format (a child box outside the root's, or a reference past 28 bits)`);
        default:
            throw new Error(`tinybvh failed with status ${status}`);
    }
}

/** Allocates WASM memory for an input array, refusing sizes the 32-bit allocator would wrap. */
function allocateInput(wasm: TinyBvhExports, bytes: number, what: string): number {
    if (bytes > MAX_ALLOCATION_BYTES) throw new RangeError(`tinybvh: the ${what} are too big for 32-bit WebAssembly (${bytes} bytes)`);
    const pointer = wasm.tbvh_alloc(bytes);
    if (pointer === 0) throw new Error(`tinybvh: out of WASM memory for the ${what}`);
    return pointer;
}

/**
 * Copies the input into WASM memory, has tinybvh build, runs `read` while the
 * tree is still in WASM memory, and frees everything. The copies are plain
 * typed-array `set`s; the WASM checks the indices and widens the vertices to
 * the 16 bytes tinybvh reads.
 */
function withTinyBvhTree<T>(wasm: TinyBvhExports, geometry: MeshInput, options: TinyBvhBuildOptions, read: (wasmBuildMs: number) => T): T {
    const builder = BUILDER[options.builder ?? 'default'];
    if (builder === undefined) throw new RangeError(`unknown tinybvh builder: ${String(options.builder)}`);
    const { positions, indices } = geometry;
    const { vertexCount, triangleCount } = countInput(geometry);
    let positionPointer = allocateInput(wasm, positions.length * 4, 'positions');
    let indexPointer = 0;
    try {
        if (indices) indexPointer = allocateInput(wasm, indices.length * 4, 'indices');
        // Views are made after every allocation: growing the memory replaces its buffer.
        new Float32Array(wasm.memory.buffer, positionPointer, positions.length).set(positions);
        if (indices) new Uint32Array(wasm.memory.buffer, indexPointer, indices.length).set(indices);

        const wasmStart = performance.now();
        const status = wasm.tbvh_build(positionPointer, vertexCount, indexPointer, triangleCount, builder);
        const wasmBuildMs = performance.now() - wasmStart;
        // tinybvh keeps its own 16-byte copy of the vertices (the indices it reads in place), so ours
        // goes now: the layout's arrays can reuse the space (Sponza as one mesh: 1,349 -> 1,316 MiB).
        wasm.tbvh_free(positionPointer);
        positionPointer = 0;
        throwOnStatus(wasm, status, geometry, vertexCount);
        return read(wasmBuildMs);
    } finally {
        wasm.tbvh_release();
        if (positionPointer) wasm.tbvh_free(positionPointer);
        if (indexPointer) wasm.tbvh_free(indexPointer);
    }
}

function buildTopology(wasm: TinyBvhExports, geometry: MeshInput, options: TinyBvhBuildOptions): TinyBvhBuild {
    const start = performance.now();
    const { triangleCount } = countInput(geometry);
    return withTinyBvhTree(wasm, geometry, options, (wasmBuildMs) => {
        const nodeSlots = wasm.tbvh_node_slots();
        if (nodeSlots === 0) throw new Error('tinybvh built no tree');
        const orderLength = wasm.tbvh_triangle_order_length();
        if (orderLength !== triangleCount) throw new Error(`tinybvh ordered ${orderLength} triangles, expected ${triangleCount}`);
        const nodes = new Uint32Array(wasm.memory.buffer, wasm.tbvh_nodes(), nodeSlots * TINYBVH_NODE_WORDS);
        const order = new Uint32Array(wasm.memory.buffer, wasm.tbvh_triangle_order(), triangleCount).slice();
        const topology = topologyFromTinyBvhNodes(nodes, order);
        return { topology, buildMs: performance.now() - start, wasmBuildMs };
    });
}

function buildGpuLayout(wasm: TinyBvhExports, geometry: MeshInput, options: TinyBvhLayoutOptions, formatsInWasm: boolean): TinyBvhLayoutBuild {
    const start = performance.now();
    const maxLeaf = options.maxLeafPrimitives ?? GPU_LAYOUT_MAX_LEAF_PRIMITIVES;
    if (!Number.isInteger(maxLeaf) || maxLeaf < 1 || maxLeaf > GPU_LAYOUT_MAX_LEAF_PRIMITIVES) {
        throw new RangeError(`maxLeafPrimitives must be an integer from 1 to ${GPU_LAYOUT_MAX_LEAF_PRIMITIVES}, got ${maxLeaf}`);
    }
    const indexed = checkTriangleLeaves(options.triangleLeaves, 'tinybvh') === 'indexed';
    const quantized = checkNodeFormat(options.nodeFormat, 'tinybvh') === 'quantized';
    // With tbvh_write_layout_as the WASM writes the formats; without, it writes records and f32 nodes and they are converted below.
    const format = formatsInWasm ? (indexed ? LAYOUT_FORMAT.indexedTriangles : 0) + (quantized ? LAYOUT_FORMAT.quantizedNodes : 0) : 0;
    const { vertexCount, triangleCount } = countInput(geometry);
    // The flags the records get: the input's filter bit and its triangles' materials in bits 16-31.
    const triangleFlags = packedTriangleFlags(geometry, triangleCount, 'mesh');

    return withTinyBvhTree(wasm, geometry, options, (wasmBuildMs) => {
        const flagsPointer = triangleFlags ? allocateInput(wasm, triangleCount * 4, 'triangle flags') : 0;
        try {
            if (triangleFlags) new Uint32Array(wasm.memory.buffer, flagsPointer, triangleCount).set(triangleFlags.subarray(0, triangleCount));

            const layoutStart = performance.now();
            const status = format === 0 ? wasm.tbvh_write_layout(flagsPointer, maxLeaf) : wasm.tbvh_write_layout_as!(flagsPointer, maxLeaf, format);
            throwOnStatus(wasm, status, geometry, vertexCount);
            const layoutMs = performance.now() - layoutStart;

            // Copied out (`slice` copies): the WASM memory is reused by the next build.
            const memory = wasm.memory.buffer;
            const nodeCount = wasm.tbvh_layout_node_count();
            const nodesAt = wasm.tbvh_layout_nodes();
            const trianglesAt = wasm.tbvh_layout_triangles();
            const wasmIndexed = (format & LAYOUT_FORMAT.indexedTriangles) !== 0;
            const wasmQuantized = (format & LAYOUT_FORMAT.quantizedNodes) !== 0;
            const layout: GpuLayoutMeshBvh = {
                // Quantized in the WASM: the frame slot, then the nodes.
                ...(wasmQuantized
                    ? { nodes: memory.slice(nodesAt, nodesAt + (nodeCount + 1) * GPU_LAYOUT_QUANTIZED_NODE_BYTES), nodeFormat: 'quantized' as const }
                    : { nodes: memory.slice(nodesAt, nodesAt + nodeCount * GPU_LAYOUT_NODE_BYTES) }),
                ...(wasmIndexed
                    ? { triangles: memory.slice(trianglesAt, trianglesAt + triangleCount * GPU_LAYOUT_INDEXED_TRIANGLE_BYTES), triangleLeaves: 'indexed' as const }
                    : indexed
                      ? {
                            triangles: indexedTrianglesFromRecordWords(new Uint32Array(memory, trianglesAt, triangleCount * (GPU_LAYOUT_TRIANGLE_BYTES / 4)), triangleCount, geometry),
                            triangleLeaves: 'indexed' as const,
                        }
                      : { triangles: memory.slice(trianglesAt, trianglesAt + triangleCount * GPU_LAYOUT_TRIANGLE_BYTES) }),
                parents: options.parents === false ? new Uint32Array(0) : new Uint32Array(memory, wasm.tbvh_layout_parents(), nodeCount).slice(),
                nodeCount,
                triangleCount,
                maxDepth: wasm.tbvh_layout_max_depth(),
                leafCount: wasm.tbvh_layout_leaf_count(),
                bounds: new Float32Array(memory, wasm.tbvh_layout_bounds(), BOUNDS_FLOATS).slice(),
            };
            return { layout: builderNodeFormat(layout, options.nodeFormat, 'tinybvh'), buildMs: performance.now() - start, wasmBuildMs, layoutMs };
        } finally {
            if (flagsPointer) wasm.tbvh_free(flagsPointer);
        }
    });
}

/**
 * tinybvh's node pool to a {@link MeshTreeTopology}. Its root is node 0 and an
 * inner node's children are `leftFirst` and `leftFirst + 1` (slot 1 is unused).
 * Inner nodes are numbered in the order a depth-first walk reaches them.
 */
export function topologyFromTinyBvhNodes(nodes: Uint32Array, order: Uint32Array): MeshTreeTopology {
    const isLeaf = (node: number) => nodes[node * TINYBVH_NODE_WORDS + NODE_TRIANGLE_COUNT] > 0;
    if (isLeaf(0)) {
        return singleLeafTopology(order, nodes[NODE_LEFT_FIRST], nodes[NODE_TRIANGLE_COUNT]);
    }
    const slots = nodes.length / TINYBVH_NODE_WORDS;
    // Inner nodes are at most half the slots; grown to the real count at the end.
    const children = new Uint32Array(Math.ceil(slots / 2) * TOPOLOGY_WORDS_PER_NODE);
    let innerCount = 1;
    // Pairs of (tinybvh node, our index), flat, with an explicit end: a big mesh has
    // millions of nodes, and a tuple per node (or shrinking the array on every pop)
    // costs more than the walk itself.
    const stack: number[] = [0, 0];
    let stackEnd = 2;
    while (stackEnd > 0) {
        stackEnd -= 2;
        const node = stack[stackEnd];
        const ours = stack[stackEnd + 1];
        const left = nodes[node * TINYBVH_NODE_WORDS + NODE_LEFT_FIRST];
        for (let side = 0; side < 2; side++) {
            const child = left + side;
            if (child >= slots) throw new RangeError(`tinybvh node ${node} points past the node pool`);
            const target = ours * TOPOLOGY_WORDS_PER_NODE + side * 2;
            if (isLeaf(child)) {
                children[target] = nodes[child * TINYBVH_NODE_WORDS + NODE_LEFT_FIRST];
                children[target + 1] = nodes[child * TINYBVH_NODE_WORDS + NODE_TRIANGLE_COUNT];
            } else {
                if (innerCount * TOPOLOGY_WORDS_PER_NODE >= children.length) throw new RangeError('tinybvh nodes do not form a tree');
                const index = innerCount++;
                children[target] = index;
                children[target + 1] = 0;
                stack[stackEnd] = child;
                stack[stackEnd + 1] = index;
                stackEnd += 2;
            }
        }
    }
    return { order, children: children.slice(0, innerCount * TOPOLOGY_WORDS_PER_NODE) };
}

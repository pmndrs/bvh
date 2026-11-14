// A small C interface over tinybvh's triangle builders, compiled to
// WebAssembly (see build-wasm.ts). With AVX available (Emscripten emulates it on
// WASM SIMD), `BVH::Build` runs tinybvh's SIMD Bonsai builder (Ganestam et al.,
// 2015: small binned-SAH trees over a grid of cells, joined by a top tree). On
// Sponza it is about 3x faster than tinybvh's scalar reference builder and its
// trees have a lower SAH cost. tinybvh's scalar H-PLOC builder (BuildHPLOC)
// can be asked for instead: it is the oracle the GPU H-PLOC (stage B2) is
// compared with, not a builder the tracer uses.
//
// It builds one mesh at a time and leaves the result in WASM memory for
// JavaScript to read (src/tinybvh.ts), in one of two forms:
// - tinybvh's own 32-byte nodes and triangle order, which JavaScript turns into
//   the plain tree description of the Builder contract (MeshTreeTopology);
// - the tracer's GPU layout (stage B0b): 64-byte two-child nodes, 48-byte
//   triangles in tree order, parent indices and the mesh's box, written here.
//   This replaces a JavaScript pass over every node and triangle, which was
//   bound by memory traffic and object handling; the timings are in PROGRESS.md
//   (B0b).
//   tbvh_write_layout_as can write the layout's other formats instead: 16-byte
//   indexed triangles (docs/internals/design/indexed-leaves.md; the 48-byte
//   records are then never allocated) and quantized nodes, a frame slot and
//   32-byte nodes packed from the 64-byte ones in place
//   (docs/internals/design/compressed-nodes.md). Both are byte for byte what
//   src/indexed-leaves.ts and src/quantized-nodes.ts make from the default
//   layout; tests in src/tinybvh-formats.test.ts check it.
//
// The GPU layout is defined in src/trace/layout.ts
// (and the WGSL generated from it). The structs below repeat it; a test there
// checks that this file's output is byte for byte what the JavaScript
// conversion (buildMeshBvhFromTopology) makes from the same tinybvh tree.
//
// tinybvh: https://github.com/jbikker/tinybvh (MIT), by Jacco Bikker. The
// pinned commit and header hashes are in build-wasm.ts and THIRD_PARTY.md.

#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include <wasm_simd128.h>

#define EXPORT extern "C" __attribute__((used, visibility("default")))

// ---------------------------------------------------------------------------
// Results. src/tinybvh.ts mirrors these numbers (BUILD_STATUS) and turns them
// into error messages.
// ---------------------------------------------------------------------------

enum Status : uint32_t {
    STATUS_OK = 0,
    // The mesh has no triangles.
    STATUS_NO_TRIANGLES = 1,
    // malloc failed (the 4 GB WASM memory is full).
    STATUS_OUT_OF_MEMORY = 2,
    // An index points past the vertices; tbvh_error_detail() is its position in the index buffer.
    STATUS_INDEX_OUT_OF_RANGE = 3,
    // tbvh_write_layout was called without a successful tbvh_build first.
    STATUS_NOTHING_BUILT = 4,
    // maxLeafPrimitives is outside 1 .. MAX_LEAF_PRIMITIVES.
    STATUS_BAD_LEAF_LIMIT = 5,
    // tinybvh's triangle order does not list every triangle exactly once.
    STATUS_BAD_ORDER = 6,
    // tinybvh's nodes do not form a tree whose leaves cover every triangle once.
    // tbvh_error_detail() says which check failed (TreeProblem).
    STATUS_BAD_TREE = 7,
    // An array for this mesh would be 4 GB or more, which 32-bit WASM cannot address.
    STATUS_TOO_LARGE = 8,
    // tbvh_build was asked for a builder it does not have (see Builder).
    STATUS_BAD_BUILDER = 9,
    // tbvh_write_layout_as was asked for a format bit it does not have (see LayoutFormat).
    STATUS_BAD_FORMAT = 10,
    // A node did not fit the quantized format (a child box outside its root's, or a
    // reference past 28 bits); tbvh_error_detail() is the node.
    STATUS_QUANTIZE_FAILED = 11,
};

// The layout's formats, bits of tbvh_write_layout_as's `format`. Mirrored in src/tinybvh.ts.
enum LayoutFormat : uint32_t {
    // 16-byte indexed triangles instead of 48-byte records.
    LAYOUT_INDEXED_TRIANGLES = 1,
    // A frame slot and 32-byte quantized nodes instead of 64-byte nodes.
    LAYOUT_QUANTIZED_NODES = 2,
};

// Which of tinybvh's builders tbvh_build runs. Mirrored in src/tinybvh.ts.
enum Builder : uint32_t {
    BUILDER_DEFAULT = 0,  // BVH::Build: the SIMD Bonsai builder
    BUILDER_HPLOC = 1,    // BVH::BuildHPLOC (settings.useHPLOC): leaves of one triangle
};

// What STATUS_BAD_TREE found, in tbvh_error_detail(). Mirrored in src/tinybvh.ts.
enum TreeProblem : uint32_t {
    TREE_CHILD_PAST_POOL = 1,     // a child index points past the used nodes
    TREE_NODE_REACHED_TWICE = 2,  // a node is the child of two nodes
    TREE_TOO_MANY_NODES = 3,      // more inner nodes than the triangles allow
    TREE_LEAF_PAST_END = 4,       // a leaf covers positions past the last triangle
    TREE_POSITION_TWICE = 5,      // a position is in two leaves
    TREE_POSITION_MISSING = 6,    // a position is in no leaf
};

static uint32_t errorDetail = 0;

// The byte size of an array of `count` elements, or false if it is 4 GB or more.
// size_t is 32 bits on wasm32, so `size_t(count) * elementBytes` would wrap for
// a big enough mesh (above about 67 M triangles for the 64-byte nodes) and
// malloc would return a buffer far too small for the writes that follow. The
// product is taken in 64 bits instead (count < 2^32 and elementBytes <= 64, so
// it cannot overflow there). Such a mesh could not fit in the 4 GB WASM memory
// anyway; this makes it fail clearly instead of writing past a buffer.
static bool arrayBytes(uint64_t count, uint64_t elementBytes, size_t& bytes)
{
    const uint64_t total = count * elementBytes;
    if (total > SIZE_MAX) return false;
    bytes = size_t(total);
    return true;
}

// ---------------------------------------------------------------------------
// The tracer's GPU layout (src/trace/layout.ts).
// ---------------------------------------------------------------------------

// One node: the boxes of its two children (Aila-Laine layout). A child with
// count > 0 is a leaf of `count` triangles from position `index` of the
// triangle array; count == 0 means the inner node `index`. Matches the WGSL `BvhNode`.
struct GpuNode {
    float leftMin[3];
    uint32_t leftIndex;
    float leftMax[3];
    uint32_t rightIndex;
    float rightMin[3];
    uint32_t leftCount;
    float rightMax[3];
    uint32_t rightCount;
};
static_assert(sizeof(GpuNode) == 64, "BVH_NODE_BYTES");
static_assert(offsetof(GpuNode, leftIndex) == 3 * 4 && offsetof(GpuNode, leftMax) == 4 * 4, "NODE_WORD.leftIndex, leftMax");
static_assert(offsetof(GpuNode, rightIndex) == 7 * 4 && offsetof(GpuNode, rightMin) == 8 * 4, "NODE_WORD.rightIndex, rightMin");
static_assert(offsetof(GpuNode, leftCount) == 11 * 4 && offsetof(GpuNode, rightMax) == 12 * 4, "NODE_WORD.leftCount, rightMax");
static_assert(offsetof(GpuNode, rightCount) == 15 * 4, "NODE_WORD.rightCount");
static_assert(offsetof(GpuNode, leftMin) == 0, "NODE_WORD.leftMin");
// The box code loads each box corner as 16 bytes (x, y, z and the word after
// it; see keepSmaller). These keep those loads inside the node: each corner is
// three floats followed by a 4-byte word, and the last ends at the node's end.
static_assert(sizeof(GpuNode::leftMin) == 12 && sizeof(GpuNode::leftMax) == 12, "box corners are three floats");
static_assert(sizeof(GpuNode::rightMin) == 12 && sizeof(GpuNode::rightMax) == 12, "box corners are three floats");
static_assert(offsetof(GpuNode, rightMax) + 16 == sizeof(GpuNode), "a 16-byte load at the last corner stays inside the node");

// One triangle in tree order: its three corners exactly as the input has them
// (the watertight test needs a shared vertex to have the same bits in every
// triangle), its original index and its flags. Matches the WGSL `BvhTriangle`.
struct GpuTriangle {
    float vertex0[3];
    uint32_t originalIndex;
    float vertex1[3];
    uint32_t flags;
    float vertex2[3];
    uint32_t padding;
};
static_assert(sizeof(GpuTriangle) == 48, "TRIANGLE_BYTES");
static_assert(offsetof(GpuTriangle, originalIndex) == 3 * 4 && offsetof(GpuTriangle, vertex1) == 4 * 4, "TRIANGLE_WORD");
static_assert(offsetof(GpuTriangle, flags) == 7 * 4 && offsetof(GpuTriangle, vertex2) == 8 * 4, "TRIANGLE_WORD");
static_assert(offsetof(GpuTriangle, vertex0) == 0 && offsetof(GpuTriangle, padding) == 11 * 4, "TRIANGLE_WORD");
// writeLeaf stores each corner as 16 bytes (x, y, z and the vertex's unused w)
// and then overwrites the fourth word with the field that belongs there. These
// guarantee that the fourth word of every corner is exactly originalIndex,
// flags or padding, and that the last store ends at the triangle's end.
static_assert(sizeof(GpuTriangle::vertex0) == 12 && sizeof(GpuTriangle::vertex1) == 12 && sizeof(GpuTriangle::vertex2) == 12, "corners are three floats");
static_assert(offsetof(GpuTriangle, originalIndex) == offsetof(GpuTriangle, vertex0) + 12, "word after corner 0");
static_assert(offsetof(GpuTriangle, flags) == offsetof(GpuTriangle, vertex1) + 12, "word after corner 1");
static_assert(offsetof(GpuTriangle, padding) == offsetof(GpuTriangle, vertex2) + 12, "word after corner 2");
static_assert(offsetof(GpuTriangle, vertex2) + 16 == sizeof(GpuTriangle), "the last 16-byte store stays inside the triangle");

// One indexed triangle in tree order (GPU_LAYOUT_INDEXED_TRIANGLE_WORD): the
// mesh's own vertex indices of its three corners, then the original index in
// bits 0-27 with the flags' bits 0-1 (filter, flip) in bits 28-29.
struct GpuIndexedTriangle {
    uint32_t vertex[3];
    uint32_t idFlags;
};
static_assert(sizeof(GpuIndexedTriangle) == 16, "INDEXED_TRIANGLE_BYTES");
static_assert(offsetof(GpuIndexedTriangle, vertex) == 0 && offsetof(GpuIndexedTriangle, idFlags) == 3 * 4, "INDEXED_TRIANGLE_WORD");
// GPU_LAYOUT_INDEXED_ID_MASK and GPU_LAYOUT_INDEXED_FLAG_SHIFT.
static const uint32_t INDEXED_ID_MASK = 0x0fffffffu;
static const uint32_t INDEXED_FLAG_SHIFT = 28;
static const uint32_t INDEXED_MOVED_FLAGS = 3;

// One quantized node (GPU_LAYOUT_QUANTIZED_NODE_WORD): per child, an axis word
// per axis (min plane in bits 0-15, max in 16-31) and the stack reference
// (index in bits 0-27, count in 28-31).
struct GpuQuantizedNode {
    uint32_t left[3];
    uint32_t leftReference;
    uint32_t right[3];
    uint32_t rightReference;
};
static_assert(sizeof(GpuQuantizedNode) == 32, "QUANTIZED_NODE_BYTES");
static_assert(offsetof(GpuQuantizedNode, leftReference) == 3 * 4 && offsetof(GpuQuantizedNode, right) == 4 * 4, "QUANTIZED_NODE_WORD");
static_assert(offsetof(GpuQuantizedNode, rightReference) == 7 * 4, "QUANTIZED_NODE_WORD.rightReference");

// The frame slot before a quantized tree's root (GPU_LAYOUT_NODE_FRAME_WORD).
struct GpuNodeFrame {
    float origin[3];
    uint32_t zero0;
    float step[3];
    uint32_t zero1;
};
static_assert(sizeof(GpuNodeFrame) == sizeof(GpuQuantizedNode), "the frame takes one node slot");
static_assert(offsetof(GpuNodeFrame, origin) == 0 && offsetof(GpuNodeFrame, step) == 4 * 4, "NODE_FRAME_WORD");
// Packing in place (packQuantizedNodes) needs two quantized slots per 64-byte node.
static_assert(2 * sizeof(GpuQuantizedNode) == sizeof(GpuNode), "a 64-byte node holds two quantized slots");

// `index` of a right child that does not exist (only a one-triangle mesh has one).
static const uint32_t EMPTY_CHILD_INDEX = 0xffffffffu;
// Parent of the root.
static const uint32_t NO_PARENT = 0xffffffffu;
// The tracer's largest leaf (its stack entries keep a leaf's count in 4 bits).
static const uint32_t MAX_LEAF_PRIMITIVES = 8;

// ---------------------------------------------------------------------------
// The current build. One mesh at a time: the worker builds meshes in sequence.
// ---------------------------------------------------------------------------

struct CurrentBuild {
    tinybvh::BVH* bvh = nullptr;
    // The positions as tinybvh reads them: x, y, z and an unused w, 16 bytes per vertex.
    tinybvh::bvhvec4* vertices = nullptr;
    // JavaScript's index buffer (or null for a triangle list). It stays allocated until tbvh_release.
    const uint32_t* indices = nullptr;
    uint32_t vertexCount = 0;
    uint32_t triangleCount = 0;
};

struct CurrentLayout {
    // 64-byte nodes, or (LAYOUT_QUANTIZED_NODES) the frame slot then 32-byte nodes, packed in the same memory.
    GpuNode* nodes = nullptr;
    // One of the two is written: records, or (LAYOUT_INDEXED_TRIANGLES) indexed triangles.
    GpuTriangle* triangles = nullptr;
    GpuIndexedTriangle* indexedTriangles = nullptr;
    uint32_t format = 0;
    uint32_t* parents = nullptr;
    uint32_t nodeCount = 0;
    uint32_t maxDepth = 0;
    uint32_t leafCount = 0;
    // The mesh's box: minX, minY, minZ, maxX, maxY, maxZ.
    float bounds[6] = {};
};

// writeLeaf loads each vertex as one 16-byte vector: x, y, z, then the unused w.
static_assert(sizeof(tinybvh::bvhvec4) == 16 && offsetof(tinybvh::bvhvec4, x) == 0 && offsetof(tinybvh::bvhvec4, z) == 8, "a vertex is x, y, z, w");

static CurrentBuild current;
static CurrentLayout layout;

static void releaseLayout()
{
    free(layout.nodes);
    free(layout.triangles);
    free(layout.indexedTriangles);
    free(layout.parents);
    layout = CurrentLayout();
}

static void releaseAll()
{
    releaseLayout();
    delete current.bvh;
    free(current.vertices);
    current = CurrentBuild();
}

// ---------------------------------------------------------------------------
// Building
// ---------------------------------------------------------------------------

// Builds a tree over `triangleCount` triangles. `positions` holds x, y, z per
// vertex. `indices` holds three vertex indices per triangle, or is null for a
// plain triangle list (three vertices per triangle); it must stay allocated
// until tbvh_release, because tbvh_write_layout reads it again. `builder` is a
// Builder. Returns a Status.
EXPORT uint32_t tbvh_build(const float* positions, uint32_t vertexCount, const uint32_t* indices, uint32_t triangleCount, uint32_t builder)
{
    releaseAll();
    errorDetail = 0;
    if (builder != BUILDER_DEFAULT && builder != BUILDER_HPLOC) return STATUS_BAD_BUILDER;
    if (triangleCount == 0) return STATUS_NO_TRIANGLES;
    // Three indices per triangle must be countable in 32 bits, and the vertex
    // copy must fit (which also keeps `vertex * 3` below from wrapping).
    size_t vertexBytes = 0;
    if (uint64_t(triangleCount) * 3 > UINT32_MAX || !arrayBytes(vertexCount, sizeof(tinybvh::bvhvec4), vertexBytes)) return STATUS_TOO_LARGE;
    if (indices) {
        const uint32_t indexCount = triangleCount * 3;
        for (uint32_t i = 0; i < indexCount; i++) {
            if (indices[i] >= vertexCount) {
                errorDetail = i;
                return STATUS_INDEX_OUT_OF_RANGE;
            }
        }
    } else if (vertexCount < triangleCount * 3) {
        errorDetail = vertexCount;
        return STATUS_INDEX_OUT_OF_RANGE;
    }

    // tinybvh reads vertices as 16-byte vectors; the copy keeps the input's bits.
    // tinybvh declares bvhvec4 16-byte aligned; vertexBytes is a multiple of 16, as aligned_alloc requires.
    current.vertices = static_cast<tinybvh::bvhvec4*>(aligned_alloc(alignof(tinybvh::bvhvec4), vertexBytes));
    if (!current.vertices && vertexCount > 0) return STATUS_OUT_OF_MEMORY;
    for (uint32_t vertex = 0; vertex < vertexCount; vertex++) {
        current.vertices[vertex] = tinybvh::bvhvec4(positions[vertex * 3], positions[vertex * 3 + 1], positions[vertex * 3 + 2], 0.0f);
    }
    current.indices = indices;
    current.vertexCount = vertexCount;
    current.triangleCount = triangleCount;

    current.bvh = new tinybvh::BVH();
    current.bvh->settings.useHPLOC = builder == BUILDER_HPLOC;
    const tinybvh::bvhvec4slice slice(current.vertices, vertexCount, sizeof(tinybvh::bvhvec4));
    if (indices) {
        current.bvh->Build(slice, indices, triangleCount);
    } else {
        current.bvh->Build(slice);
    }
    return STATUS_OK;
}

// More detail on the last error (see Status).
EXPORT uint32_t tbvh_error_detail()
{
    return errorDetail;
}

// Node slots tinybvh used (it leaves slot 1 unused so siblings share a cache line).
EXPORT uint32_t tbvh_node_slots()
{
    return current.bvh ? current.bvh->usedNodes : 0;
}

// tinybvh's nodes: per node aabbMin (3 floats), leftFirst, aabbMax (3 floats), triCount.
// triCount > 0: a leaf of triCount triangles from position leftFirst of the triangle order.
// triCount == 0: an inner node whose children are leftFirst and leftFirst + 1.
EXPORT const void* tbvh_nodes()
{
    return current.bvh ? static_cast<const void*>(current.bvh->bvhNode) : nullptr;
}

// The triangle order: position i of the tree holds original triangle tbvh_triangle_order()[i].
EXPORT const uint32_t* tbvh_triangle_order()
{
    return current.bvh ? current.bvh->primIdx : nullptr;
}

// Entries in the triangle order (equal to the triangle count: without spatial splits or presplitting, which
// are off by default, tinybvh never splits a triangle).
EXPORT uint32_t tbvh_triangle_order_length()
{
    return current.bvh ? current.bvh->idxCount : 0;
}

// ---------------------------------------------------------------------------
// Writing the GPU layout
//
// This makes exactly what buildMeshBvhFromTopology (packages/ddgi-core/src/
// src/trace/mesh-bvh-topology.ts) makes from tinybvh's tree, as
// topologyFromTinyBvhNodes (src/tinybvh.ts) describes it, so the bytes come out
// the same:
// 1. Walk tinybvh's tree depth first, left child first, numbering our nodes as
//    they are reached. A node's two inner children get consecutive numbers when
//    the node is written, so every child has a larger number than its parent.
//    Leaves with more triangles than the tracer accepts are cut in half until
//    they fit. When a leaf is placed, its triangles are written (in tinybvh's
//    order) and its box is computed from their corners, while they are at hand.
// 2. Last node first, each inner child's box becomes the box around that
//    child's two boxes (already written, since children come later).
// The boxes tinybvh computed are not used: recomputing them from the triangles
// guarantees that each box contains its triangles exactly (DECISIONS 2026-11-04).
// ---------------------------------------------------------------------------

// Boxes are grown with WebAssembly SIMD, one corner (x, y, z) per 128-bit
// vector. Each lane keeps the rule of the JavaScript conversion,
// `if (value < low) low = value`: the new value only if it is strictly smaller
// (larger), so a tie between -0 and +0, or a NaN, keeps the running value and
// the bytes come out the same. (fminf/fmaxf and f32x4.min/max treat those
// cases differently.) A compare and a bit-select per lane have no branches.
// Written as scalar selects, V8 turned them into branches that mispredict, and
// computing the boxes was several times slower (PROGRESS.md, B0b).
//
// A corner is held as four floats: x, y, z and one more word (a vertex's
// unused w; or, in a node, the index or count stored after a box corner). The
// fourth lane is never stored into a box.

static inline v128_t keepSmaller(v128_t running, v128_t value)
{
    return wasm_v128_bitselect(value, running, wasm_f32x4_lt(value, running));
}

static inline v128_t keepLarger(v128_t running, v128_t value)
{
    return wasm_v128_bitselect(value, running, wasm_f32x4_gt(value, running));
}

static inline void storeXyz(float* target, v128_t vector)
{
    target[0] = wasm_f32x4_extract_lane(vector, 0);
    target[1] = wasm_f32x4_extract_lane(vector, 1);
    target[2] = wasm_f32x4_extract_lane(vector, 2);
}

static const v128_t EMPTY_LOW = wasm_f32x4_const_splat(INFINITY);
static const v128_t EMPTY_HIGH = wasm_f32x4_const_splat(-INFINITY);

// A child still to be written: an inner node of tinybvh's tree, or a run of
// triangle positions [start, start + count) that becomes a leaf or gets cut.
struct ChildSource {
    uint32_t tinyNode;  // NOT_A_TINY_NODE for a run
    uint32_t start;
    uint32_t count;
};
static const uint32_t NOT_A_TINY_NODE = 0xffffffffu;

static ChildSource runOf(uint32_t start, uint32_t count)
{
    return { NOT_A_TINY_NODE, start, count };
}

// The fields of one side (left or right) of a GpuNode.
struct ChildSlot {
    uint32_t* index;
    uint32_t* count;
    float* boxMin;
    float* boxMax;
};

static ChildSlot leftSlot(GpuNode& node)
{
    return { &node.leftIndex, &node.leftCount, node.leftMin, node.leftMax };
}

static ChildSlot rightSlot(GpuNode& node)
{
    return { &node.rightIndex, &node.rightCount, node.rightMin, node.rightMax };
}

// A node of ours still to be written, with what it stands for.
struct PendingNode {
    uint32_t outputNode;
    uint32_t depth;
    ChildSource source;
};

// No node was allocated for a child (it became a leaf).
static const uint32_t NO_NEW_NODE = 0xffffffffu;

// Step 1: numbers and links every node, and writes the triangles and leaf boxes.
class TreeWriter {
public:
    TreeWriter(const tinybvh::BVH& bvh, const uint32_t* triangleFlags, uint32_t maxLeaf, uint32_t maxNodes, uint8_t* seenTinyNodes, uint8_t* coveredPositions)
        : bvh(bvh),
          order(bvh.primIdx),
          triangleFlags(triangleFlags),
          triangleCount(current.triangleCount),
          maxLeaf(maxLeaf),
          maxNodes(maxNodes),
          seenTinyNodes(seenTinyNodes),
          coveredPositions(coveredPositions)
    {
    }

    // Returns a Status.
    uint32_t writeTree()
    {
        layout.parents[0] = NO_PARENT;
        layout.nodeCount = 1;
        layout.maxDepth = 1;
        layout.leafCount = 0;

        const auto& root = bvh.bvhNode[0];
        if (root.triCount > 0) {
            // The whole mesh is one leaf, but the root still needs two children, so the
            // run is cut in two; one triangle leaves the right child empty.
            // (singleLeafTopology rounds the left half up.)
            const uint32_t count = root.triCount;
            const uint32_t leftCount = (count + 1) / 2;
            const bool hasRight = count > 1;
            if (!writeNode(0, 1, runOf(root.leftFirst, leftCount), hasRight, runOf(root.leftFirst + leftCount, count - leftCount))) return failure;
        } else {
            pending.push_back({ 0, 1, { 0, 0, 0 } });
        }

        while (!pending.empty()) {
            const PendingNode next = pending.back();
            pending.pop_back();
            if (next.depth > layout.maxDepth) layout.maxDepth = next.depth;
            ChildSource left;
            ChildSource right;
            if (next.source.tinyNode != NOT_A_TINY_NODE) {
                const uint32_t tinyNode = next.source.tinyNode;
                if (seenTinyNodes[tinyNode]) return fail(TREE_NODE_REACHED_TWICE);
                seenTinyNodes[tinyNode] = 1;
                const uint32_t firstChild = bvh.bvhNode[tinyNode].leftFirst;
                if (!sourceOfTinyNode(firstChild, left) || !sourceOfTinyNode(firstChild + 1, right)) return fail(TREE_CHILD_PAST_POOL);
            } else {
                // A leaf too big for the tracer: cut the run in half (rounding the left half down).
                const uint32_t half = next.source.count / 2;
                left = runOf(next.source.start, half);
                right = runOf(next.source.start + half, next.source.count - half);
            }
            if (!writeNode(next.outputNode, next.depth, left, true, right)) return failure;
        }

        // Every position must be in a leaf: otherwise its triangle was never written.
        for (uint32_t position = 0; position < triangleCount; position++) {
            if (!coveredPositions[position]) return fail(TREE_POSITION_MISSING);
        }
        return STATUS_OK;
    }

private:
    const tinybvh::BVH& bvh;
    const uint32_t* const order;
    const uint32_t* const triangleFlags;
    const uint32_t triangleCount;
    const uint32_t maxLeaf;
    const uint32_t maxNodes;
    uint8_t* const seenTinyNodes;
    uint8_t* const coveredPositions;
    std::vector<PendingNode> pending;
    uint32_t failure = STATUS_OK;

    uint32_t fail(TreeProblem problem)
    {
        errorDetail = problem;
        failure = STATUS_BAD_TREE;
        return failure;
    }

    // tinybvh node `slot` as a child: a leaf becomes its run of positions, an inner node stays a node.
    bool sourceOfTinyNode(uint32_t slot, ChildSource& source) const
    {
        if (slot >= bvh.usedNodes) return false;
        const auto& node = bvh.bvhNode[slot];
        source = node.triCount > 0 ? runOf(node.leftFirst, node.triCount) : ChildSource { slot, 0, 0 };
        return true;
    }

    // Writes both children of `outputNode` and queues the new inner ones, right
    // first so that the left subtree is walked (and numbered) first.
    bool writeNode(uint32_t outputNode, uint32_t depth, const ChildSource& left, bool hasRight, const ChildSource& right)
    {
        GpuNode& node = layout.nodes[outputNode];
        uint32_t leftChild;
        if (!placeChild(outputNode, leftSlot(node), left, leftChild)) return false;
        uint32_t rightChild = NO_NEW_NODE;
        if (hasRight) {
            if (!placeChild(outputNode, rightSlot(node), right, rightChild)) return false;
        } else {
            // Only the root of a one-triangle mesh gets here. Its right box stays zero.
            node.rightIndex = EMPTY_CHILD_INDEX;
            node.rightCount = 0;
            memset(node.rightMin, 0, sizeof(node.rightMin));
            memset(node.rightMax, 0, sizeof(node.rightMax));
        }
        if (rightChild != NO_NEW_NODE) pending.push_back({ rightChild, depth + 1, right });
        if (leftChild != NO_NEW_NODE) pending.push_back({ leftChild, depth + 1, left });
        return true;
    }

    // Writes one child into `slot`. A run that fits the tracer's leaves becomes a
    // leaf (newNode = NO_NEW_NODE); anything else gets the next node number.
    bool placeChild(uint32_t parent, const ChildSlot& slot, const ChildSource& source, uint32_t& newNode)
    {
        const bool isRun = source.tinyNode == NOT_A_TINY_NODE;
        if (isRun && source.count <= maxLeaf) {
            // Written as a sum in 64 bits: start + count could wrap in 32.
            if (uint64_t(source.start) + source.count > triangleCount) {
                fail(TREE_LEAF_PAST_END);
                return false;
            }
            for (uint32_t position = source.start; position < source.start + source.count; position++) {
                if (coveredPositions[position]) {
                    fail(TREE_POSITION_TWICE);
                    return false;
                }
                coveredPositions[position] = 1;
            }
            *slot.index = source.start;
            *slot.count = source.count;
            writeLeaf(slot, source.start, source.count);
            layout.leafCount++;
            newNode = NO_NEW_NODE;
            return true;
        }
        if (layout.nodeCount >= maxNodes) {
            fail(TREE_TOO_MANY_NODES);
            return false;
        }
        newNode = layout.nodeCount++;
        layout.parents[newNode] = parent;
        *slot.index = newNode;
        *slot.count = 0;
        return true;
    }

    // Writes the triangles at positions [start, start + count) in tinybvh's order,
    // corners copied bit for bit (the watertight test needs a vertex shared by two
    // triangles to have the same bits in both), and the leaf's box around them.
    // Corners are visited triangle by triangle, corner 0 to 2, as in the
    // JavaScript, which decides ties between -0 and +0.
    void writeLeaf(const ChildSlot& slot, uint32_t start, uint32_t count)
    {
        v128_t low = EMPTY_LOW;
        v128_t high = EMPTY_HIGH;
        if (layout.indexedTriangles) {
            // Indexed: the vertex indices and the id word; the box from the same corners, in the same order.
            for (uint32_t position = start; position < start + count; position++) {
                const uint32_t original = order[position];
                GpuIndexedTriangle& triangle = layout.indexedTriangles[position];
                for (uint32_t corner = 0; corner < 3; corner++) {
                    const uint32_t vertex = current.indices ? current.indices[original * 3 + corner] : original * 3 + corner;
                    const v128_t xyzw = wasm_v128_load(&current.vertices[vertex]);
                    low = keepSmaller(low, xyzw);
                    high = keepLarger(high, xyzw);
                    triangle.vertex[corner] = vertex;
                }
                const uint32_t flags = triangleFlags ? triangleFlags[original] : 0;
                triangle.idFlags = (original & INDEXED_ID_MASK) | ((flags & INDEXED_MOVED_FLAGS) << INDEXED_FLAG_SHIFT);
            }
            storeXyz(slot.boxMin, low);
            storeXyz(slot.boxMax, high);
            return;
        }
        for (uint32_t position = start; position < start + count; position++) {
            const uint32_t original = order[position];
            GpuTriangle& triangle = layout.triangles[position];
            float* const corners[3] = { triangle.vertex0, triangle.vertex1, triangle.vertex2 };
            for (uint32_t corner = 0; corner < 3; corner++) {
                const uint32_t vertex = current.indices ? current.indices[original * 3 + corner] : original * 3 + corner;
                const v128_t xyzw = wasm_v128_load(&current.vertices[vertex]);
                // Stores x, y, z and the vertex's unused w, which the words after the corner overwrite below.
                wasm_v128_store(corners[corner], xyzw);
                low = keepSmaller(low, xyzw);
                high = keepLarger(high, xyzw);
            }
            triangle.originalIndex = original;
            triangle.flags = triangleFlags ? triangleFlags[original] : 0;
            triangle.padding = 0;
        }
        storeXyz(slot.boxMin, low);
        storeXyz(slot.boxMax, high);
    }
};

// The box around `node`'s child boxes, starting from an empty box: left, then
// right if it exists, mins with mins and maxes with maxes, as in the JavaScript.
static void unionOfChildBoxes(const GpuNode& node, float* boxMin, float* boxMax)
{
    v128_t low = keepSmaller(EMPTY_LOW, wasm_v128_load(node.leftMin));
    v128_t high = keepLarger(EMPTY_HIGH, wasm_v128_load(node.leftMax));
    if (node.rightIndex != EMPTY_CHILD_INDEX) {
        low = keepSmaller(low, wasm_v128_load(node.rightMin));
        high = keepLarger(high, wasm_v128_load(node.rightMax));
    }
    storeXyz(boxMin, low);
    storeXyz(boxMax, high);
}

// Step 2: every inner child's box, last node first, then the mesh's box. (Leaf boxes are written in step 1.)
static void writeInnerBoxes()
{
    for (uint32_t index = layout.nodeCount; index-- > 0;) {
        GpuNode& node = layout.nodes[index];
        if (node.leftCount == 0) unionOfChildBoxes(layout.nodes[node.leftIndex], node.leftMin, node.leftMax);
        if (node.rightIndex != EMPTY_CHILD_INDEX && node.rightCount == 0) {
            unionOfChildBoxes(layout.nodes[node.rightIndex], node.rightMin, node.rightMax);
        }
    }
    unionOfChildBoxes(layout.nodes[0], layout.bounds, layout.bounds + 3);
}

// Checks that tinybvh's order lists every triangle once. `seen` has triangleCount zeroed bytes.
static bool isPermutation(const uint32_t* order, uint32_t triangleCount, uint8_t* seen)
{
    for (uint32_t position = 0; position < triangleCount; position++) {
        const uint32_t triangle = order[position];
        if (triangle >= triangleCount || seen[triangle]) return false;
        seen[triangle] = 1;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Quantized nodes: src/quantized-nodes.ts (nodeFrameOf, quantizeAxis,
// quantizeNodeWords) in C, step for step and in doubles as there, so the
// words come out the same. Every value tested is a float divided by a power
// of two, which is exact.
// ---------------------------------------------------------------------------

// FRAME_MAX_SPAN, FRAME_EXACT_STEPS, FRAME_MIN_EXPONENT, FRAME_LIMIT, QUANTIZE_MARGIN_STEPS, QUANTIZED_EMPTY_AXIS.
static const double FRAME_MAX_SPAN = 65532.0;
static const double FRAME_EXACT_STEPS = 8388608.0;  // 2^23
static const int FRAME_MIN_EXPONENT = -100;
static const double FRAME_LIMIT = 1267650600228229401496703205376.0;  // 2^100
static const double QUANTIZE_MARGIN_STEPS = 1.0 / 1024.0;
static const uint32_t QUANTIZED_EMPTY_AXIS = 0xffffffffu;
// The reference's count shift and index mask (trace/layout.ts).
static const uint32_t REFERENCE_COUNT_SHIFT = 28;
static const uint32_t REFERENCE_INDEX_MASK = 0x0fffffffu;

static bool isEmptyAxis(double low, double high)
{
    return !(low <= high) || !(fabs(low) < FRAME_LIMIT) || !(fabs(high) < FRAME_LIMIT);
}

// floor(log2(x)) - 1 for a positive finite x: a lower bound, which is all the search needs.
static int exponentBelow(double x)
{
    return ilogb(x) - 1;
}

struct AxisFrame {
    double origin;
    double step;
};

static AxisFrame axisFrame(double low, double high)
{
    if (isEmptyAxis(low, high)) return { 0.0, 1.0 };
    // Any start at or below the answer gives the answer: the rule is monotone in the exponent.
    int exponent = FRAME_MIN_EXPONENT;
    const double largest = fabs(low) > fabs(high) ? fabs(low) : fabs(high);
    if (largest > 0) exponent = exponentBelow(largest) - 24 > exponent ? exponentBelow(largest) - 24 : exponent;
    if (high > low) exponent = exponentBelow(high - low) - 17 > exponent ? exponentBelow(high - low) - 17 : exponent;
    for (;;) {
        const double step = ldexp(1.0, exponent);
        const bool exact = fabs(low) < FRAME_EXACT_STEPS * step && fabs(high) < FRAME_EXACT_STEPS * step;
        if (exact && ceil(high / step) - floor(low / step) <= FRAME_MAX_SPAN) return { (floor(low / step) - 1) * step, step };
        exponent++;
    }
}

// One axis of a child box as an axis word, rounded outward with the margin. False if it doesn't fit the frame.
static bool quantizeAxis(double low, double high, const AxisFrame& frame, uint32_t& word)
{
    if (isEmptyAxis(low, high)) {
        word = QUANTIZED_EMPTY_AXIS;
        return true;
    }
    const double base = frame.origin / frame.step;
    const double lowSteps = low / frame.step;
    const double highSteps = high / frame.step;
    double qMin = floor(lowSteps) - base;
    if (lowSteps - floor(lowSteps) < QUANTIZE_MARGIN_STEPS) qMin--;
    double qMax = ceil(highSteps) - base;
    if (ceil(highSteps) - highSteps < QUANTIZE_MARGIN_STEPS) qMax++;
    if (qMin < 0 || qMax > 65535) return false;
    word = uint32_t(qMin) | (uint32_t(qMax) << 16);
    return true;
}

// Math.min and Math.max as JavaScript runs them on the root's boxes (-0 below +0).
static double jsMin(double running, double value)
{
    return value < running || (value == running && signbit(value)) ? value : running;
}

static double jsMax(double running, double value)
{
    return value > running || (value == running && !signbit(value)) ? value : running;
}

// One quantized node from a 64-byte node, in `frame`. False if a child doesn't fit.
static bool quantizeNode(const GpuNode& node, const AxisFrame* frame, GpuQuantizedNode& out)
{
    const float* const mins[2] = { node.leftMin, node.rightMin };
    const float* const maxes[2] = { node.leftMax, node.rightMax };
    const uint32_t indices[2] = { node.leftIndex, node.rightIndex };
    const uint32_t counts[2] = { node.leftCount, node.rightCount };
    uint32_t* const words[2] = { out.left, out.right };
    uint32_t* const references[2] = { &out.leftReference, &out.rightReference };
    for (uint32_t side = 0; side < 2; side++) {
        if (indices[side] == EMPTY_CHILD_INDEX) {
            words[side][0] = words[side][1] = words[side][2] = QUANTIZED_EMPTY_AXIS;
            *references[side] = EMPTY_CHILD_INDEX;
            continue;
        }
        for (uint32_t axis = 0; axis < 3; axis++) {
            if (!quantizeAxis(mins[side][axis], maxes[side][axis], frame[axis], words[side][axis])) return false;
        }
        if (indices[side] > REFERENCE_INDEX_MASK || counts[side] >= 15) return false;
        *references[side] = (counts[side] << REFERENCE_COUNT_SHIFT) | indices[side];
    }
    return true;
}

// Packs layout.nodes in place into the frame slot and 32-byte nodes: slot i + 1
// (node i) lies inside 64-byte node (i + 1) / 2, which is node i or one before
// it, so each node is read before anything overwrites it. Node 0 is read whole
// before the frame and its slot are written over it. Returns a Status.
static uint32_t packQuantizedNodes()
{
    GpuNode root;
    memcpy(&root, layout.nodes, sizeof(root));
    double box[6] = { INFINITY, INFINITY, INFINITY, -INFINITY, -INFINITY, -INFINITY };
    const float* const mins[2] = { root.leftMin, root.rightMin };
    const float* const maxes[2] = { root.leftMax, root.rightMax };
    const uint32_t indices[2] = { root.leftIndex, root.rightIndex };
    for (uint32_t side = 0; side < 2; side++) {
        if (indices[side] == EMPTY_CHILD_INDEX) continue;
        for (uint32_t axis = 0; axis < 3; axis++) {
            const double low = mins[side][axis];
            const double high = maxes[side][axis];
            if (isEmptyAxis(low, high)) continue;
            box[axis] = jsMin(box[axis], low);
            box[axis + 3] = jsMax(box[axis + 3], high);
        }
    }
    AxisFrame frame[3];
    for (uint32_t axis = 0; axis < 3; axis++) frame[axis] = axisFrame(box[axis], box[axis + 3]);

    GpuQuantizedNode first;
    if (!quantizeNode(root, frame, first)) {
        errorDetail = 0;
        return STATUS_QUANTIZE_FAILED;
    }
    // Through memcpy only: the same bytes are read as GpuNode and written as slots.
    uint8_t* const bytes = reinterpret_cast<uint8_t*>(layout.nodes);
    GpuNodeFrame frameSlot;
    for (uint32_t axis = 0; axis < 3; axis++) {
        frameSlot.origin[axis] = float(frame[axis].origin);
        frameSlot.step[axis] = float(frame[axis].step);
    }
    frameSlot.zero0 = 0;
    frameSlot.zero1 = 0;
    memcpy(bytes, &frameSlot, sizeof(frameSlot));
    memcpy(bytes + sizeof(GpuQuantizedNode), &first, sizeof(first));
    for (uint32_t index = 1; index < layout.nodeCount; index++) {
        GpuNode node;
        memcpy(&node, bytes + size_t(index) * sizeof(GpuNode), sizeof(node));
        GpuQuantizedNode packed;
        if (!quantizeNode(node, frame, packed)) {
            errorDetail = index;
            return STATUS_QUANTIZE_FAILED;
        }
        memcpy(bytes + size_t(index + 1) * sizeof(GpuQuantizedNode), &packed, sizeof(packed));
    }
    return STATUS_OK;
}

// tbvh_write_layout and tbvh_write_layout_as: the layout in `format` (LayoutFormat bits).
static uint32_t writeLayout(const uint32_t* triangleFlags, uint32_t maxLeafPrimitives, uint32_t format)
{
    releaseLayout();
    errorDetail = 0;
    if ((format & ~uint32_t(LAYOUT_INDEXED_TRIANGLES | LAYOUT_QUANTIZED_NODES)) != 0) return STATUS_BAD_FORMAT;
    if (!current.bvh) return STATUS_NOTHING_BUILT;
    if (maxLeafPrimitives < 1 || maxLeafPrimitives > MAX_LEAF_PRIMITIVES) return STATUS_BAD_LEAF_LIMIT;
    const tinybvh::BVH& bvh = *current.bvh;
    const uint32_t triangleCount = current.triangleCount;
    if (bvh.idxCount != triangleCount || bvh.usedNodes == 0) return STATUS_BAD_ORDER;

    // A binary tree whose leaves hold at least one triangle has at most (triangles - 1) inner nodes.
    const uint32_t maxNodes = triangleCount > 1 ? triangleCount - 1 : 1;
    size_t nodeBytes = 0;
    size_t parentBytes = 0;
    size_t triangleBytes = 0;
    const bool indexed = (format & LAYOUT_INDEXED_TRIANGLES) != 0;
    if (!arrayBytes(maxNodes, sizeof(GpuNode), nodeBytes) || !arrayBytes(maxNodes, sizeof(uint32_t), parentBytes) ||
        !arrayBytes(triangleCount, indexed ? sizeof(GpuIndexedTriangle) : sizeof(GpuTriangle), triangleBytes)) {
        return STATUS_TOO_LARGE;
    }
    layout.format = format;
    layout.nodes = static_cast<GpuNode*>(malloc(nodeBytes));
    layout.parents = static_cast<uint32_t*>(malloc(parentBytes));
    // Only the triangles of the format asked for: indexed, the 48-byte records are never allocated.
    if (indexed) {
        layout.indexedTriangles = static_cast<GpuIndexedTriangle*>(malloc(triangleBytes));
    } else {
        layout.triangles = static_cast<GpuTriangle*>(malloc(triangleBytes));
    }
    // Scratch: which triangles the order has listed, which tinybvh nodes the walk has
    // reached, and which positions a leaf covers.
    uint8_t* const seenTriangles = static_cast<uint8_t*>(calloc(triangleCount, 1));
    uint8_t* const seenTinyNodes = static_cast<uint8_t*>(calloc(bvh.usedNodes, 1));
    uint8_t* const coveredPositions = static_cast<uint8_t*>(calloc(triangleCount, 1));

    uint32_t status = STATUS_OK;
    if (!layout.nodes || !layout.parents || (!layout.triangles && !layout.indexedTriangles) || !seenTriangles || !seenTinyNodes || !coveredPositions) {
        status = STATUS_OUT_OF_MEMORY;
    } else if (!isPermutation(bvh.primIdx, triangleCount, seenTriangles)) {
        // Checked first: writing a leaf looks up each original triangle's vertices.
        status = STATUS_BAD_ORDER;
    } else {
        TreeWriter writer(bvh, triangleFlags, maxLeafPrimitives, maxNodes, seenTinyNodes, coveredPositions);
        status = writer.writeTree();
    }
    free(seenTriangles);
    free(seenTinyNodes);
    free(coveredPositions);
    if (status != STATUS_OK) {
        releaseLayout();
        return status;
    }
    writeInnerBoxes();
    if (format & LAYOUT_QUANTIZED_NODES) {
        // After the boxes and the mesh's box: the frame comes from the root's exact children.
        status = packQuantizedNodes();
        if (status != STATUS_OK) {
            releaseLayout();
            return status;
        }
    }
    return STATUS_OK;
}

// Writes the last build's tree in the GPU layout. `triangleFlags` holds one
// word per original triangle, or is null for all zeros. Leaves larger than
// `maxLeafPrimitives` are cut in half until they fit. Returns a Status; on
// success the tbvh_layout_* functions describe the result until tbvh_release
// or the next build.
EXPORT uint32_t tbvh_write_layout(const uint32_t* triangleFlags, uint32_t maxLeafPrimitives)
{
    return writeLayout(triangleFlags, maxLeafPrimitives, 0);
}

// tbvh_write_layout in other formats: `format` is a sum of LayoutFormat bits
// (0 is tbvh_write_layout). The tree, the leaf boxes and every f32 box are the
// same; LAYOUT_INDEXED_TRIANGLES writes 16-byte indexed triangles instead of the
// records (the flags' bits 0-1 only: materials are the scene's), and
// LAYOUT_QUANTIZED_NODES packs the nodes afterwards, in place, into the frame
// slot and nodeCount 32-byte nodes.
EXPORT uint32_t tbvh_write_layout_as(const uint32_t* triangleFlags, uint32_t maxLeafPrimitives, uint32_t format)
{
    return writeLayout(triangleFlags, maxLeafPrimitives, format);
}

// The format the last layout was written in (0 after tbvh_write_layout).
EXPORT uint32_t tbvh_layout_format()
{
    return layout.format;
}

// The layout's parts: nodeCount nodes of 64 bytes (LAYOUT_QUANTIZED_NODES: the
// frame slot then nodeCount nodes of 32 bytes), triangleCount triangles of 48
// bytes (LAYOUT_INDEXED_TRIANGLES: 16), nodeCount parent indices, and six
// floats of mesh box.
EXPORT const GpuNode* tbvh_layout_nodes()
{
    return layout.nodes;
}

EXPORT uint32_t tbvh_layout_node_count()
{
    return layout.nodeCount;
}

EXPORT const void* tbvh_layout_triangles()
{
    return layout.indexedTriangles ? static_cast<const void*>(layout.indexedTriangles) : static_cast<const void*>(layout.triangles);
}

EXPORT const uint32_t* tbvh_layout_parents()
{
    return layout.parents;
}

EXPORT const float* tbvh_layout_bounds()
{
    return layout.bounds;
}

// Levels of nodes on the longest path (the traversal stack needs at most this many entries).
EXPORT uint32_t tbvh_layout_max_depth()
{
    return layout.maxDepth;
}

EXPORT uint32_t tbvh_layout_leaf_count()
{
    return layout.leafCount;
}

// Frees the tree, the layout and tinybvh's vertex copy. JavaScript frees its own inputs after this.
EXPORT void tbvh_release()
{
    releaseAll();
}

// Memory for the inputs, owned by JavaScript between a build's start and end.
EXPORT void* tbvh_alloc(uint32_t bytes)
{
    return malloc(bytes);
}

EXPORT void tbvh_free(void* pointer)
{
    free(pointer);
}

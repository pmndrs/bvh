/**
 * WGSL for the spatial queries over a trace scene (PARITY 8.5, 8.6): closest
 * point to a point, sphere and box tests, and a generic shapecast whose box
 * and triangle tests the caller writes. They are compiled into the trace
 * module (`traceModuleWgsl({ queries: true })`, `{ shapecast }`) and read the
 * same buffers through the same loaders as the ray walk, so every node format
 * and triangle format the walk reads works here too. The ray walk is not
 * touched: with neither option the module is the one it always was.
 *
 * **One walk, generated per hook set.** WGSL has no function pointers, so the
 * shapecast walk is a TypeScript template ({@link shapecastWalkWgsl}) stamped
 * out once per query with that query's hooks:
 *
 * - `bounds(box: BvhShapecastBounds) -> f32`: a box's score, or a negative
 *   number to skip it. Of two boxes the lower score is entered first, and the
 *   other is pushed with its score;
 * - `keep(score: f32) -> bool`: asked when a pushed box comes off the stack,
 *   with the score it was pushed with; false skips it (distance culling: the
 *   best answer may have improved since);
 * - `triangle(triangle: BvhShapecastTriangle) -> bool`: true stops the walk.
 *
 * The library's own queries are such hook sets over private state
 * (`bvhClosestPoint`, `bvhIntersectsSphere`, `bvhIntersectsBox`,
 * `bvhIntersectsOrientedBox`, and the inside test's `bvhRayCrossings`, which
 * `bvhInsideVotes` and `bvhIsInside` cast three times), and `shapecast: wgsl` stamps the walk for the
 * caller's `bvhShapecastBounds`, `bvhShapecastKeep` and `bvhShapecastTriangle`.
 *
 * **World space throughout.** Rays can be moved into an instance's space
 * because a ray's `t` survives any affine map; distances and shapes don't
 * (a sphere under a non-uniform scale is an ellipsoid). So the query stays in
 * world space and the scene comes to it: on entering an instance the walk
 * inverts its world-to-object rows once (by the adjugate), each mesh box
 * becomes the world box around the moved box (centre moved, half size by the
 * absolute matrix, grown by {@link QUERY_BOX_PADDING} of the magnitudes the
 * f32 transform adds, so the corners it moves stay inside), and each
 * triangle's corners are moved to world space before the hook sees them. A
 * merged mesh is in world space already and is read as it is. The boxes a
 * hook sees are a little larger than the exact world boxes, by that padding;
 * the triangles are exact up to the f32 transform.
 *
 * **Nesting.** A caller's `bvhShapecastTriangle` may call the module's other
 * walks (`bvhTraceAny`, `bvhClosestPoint`, `bvhIntersects*`): those enter
 * instances of their own and overwrite the private node frame of quantized
 * nodes (`bvhEnterMeshNodes`), so the caller's walk re-reads its instance's
 * frame after each leaf's hook calls. `bvhShapecastBounds` and
 * `bvhShapecastKeep` must not call a walk (a node is decoded right after
 * them), nor may the ray walk's `bvhTrianglePasses`. The library's own hook
 * sets call no walk.
 *
 * The walk itself is the ray walk's shape (one loop, one stack of child
 * references shared by both levels, an instance marker on entry), with a
 * second stack of the pushed boxes' scores beside it.
 */

/**
 * How much a moved mesh box is grown, as a share of the magnitudes the f32
 * transform adds up (the moved box centre's terms, the translation and the
 * half size): about 8 f32 steps of them, so the corners moved by the f32
 * matrix stay inside even where large terms cancel (an instance far from the
 * origin whose geometry is offset back near it).
 */
export const QUERY_BOX_PADDING = 1e-6;

/**
 * The largest `maxDistance` a closest-point query uses; larger ones (and
 * infinity) are clamped to it so its square stays a finite f32.
 */
export const CLOSEST_POINT_MAX_DISTANCE = 1e18;

/**
 * The inside test's ray directions (`bvhIsInside`, `CpuMeshBvh.isPointInside`):
 * three unit vectors, far apart, along no axis or diagonal (an authored
 * mesh's edges and faces mostly lie along those), into different octants (no
 * one hemisphere holds all three, so a mesh open on one side still has rays
 * that reach its walls), each component an exact f32, so the CPU and the GPU
 * cast the same rays. A point is inside when
 * at least {@link INSIDE_TEST_MAJORITY} of the rays vote it inside.
 */
export const INSIDE_TEST_DIRECTIONS: readonly (readonly [number, number, number])[] = [
    [Math.fround(0.816543519), Math.fround(0.272114486), Math.fround(0.50912714)],
    [Math.fround(-0.340902179), Math.fround(0.931205928), Math.fround(-0.129000828)],
    [Math.fround(-0.464902997), Math.fround(-0.242553741), Math.fround(0.85148865)],
];

/** Votes of {@link INSIDE_TEST_DIRECTIONS} that make a point inside: a majority of three. */
export const INSIDE_TEST_MAJORITY = 2;

/**
 * How the inside test reads one ray's crossings:
 *
 * - `'parity'`: an odd number of crossings is inside (even-odd rule). Needs no
 *   consistent winding; overlapping closed shells cancel where they overlap.
 * - `'winding'`: a non-zero sum of signed crossings is inside (non-zero rule;
 *   +1 leaving through a triangle's authored front side, -1 entering through
 *   it): the generalized winding number along the ray, an integer for closed
 *   meshes. Overlapping closed shells stay inside; it needs consistently wound
 *   triangles (either way round).
 */
export type InsideRule = 'parity' | 'winding';

const wgslVec3 = (v: readonly number[]) => `vec3f(${v.map((x) => x.toPrecision(9)).join(', ')})`;

/** The hook names a stamped shapecast walk calls. */
export interface ShapecastHookNames {
    readonly bounds: string;
    readonly keep: string;
    readonly triangle: string;
}

/**
 * One shapecast walk, named `name`, calling `hooks` (see the file comment).
 * With `rayMasks` it takes the query's instance mask (`mask: u32`) and skips
 * an instance whose `(instanceMask & mask) == 0`, as the masked ray walk does;
 * without, it takes no arguments. Returns true when the triangle hook stopped it.
 */
export function shapecastWalkWgsl(name: string, hooks: ShapecastHookNames, rayMasks: boolean, nestable = false): string {
    // A caller's hooks may run another walk, which leaves another mesh's quantized frame behind.
    const restoreFrame = nestable
        ? `
            // The hooks may have run another walk (bvhTraceAny, bvhClosestPoint, ...), which reads
            // the frames of the meshes it enters: read this instance's again before its next node.
            bvhEnterMeshNodes(instance.rootNodeOffset);`
        : '';
    return /* wgsl */ `
fn ${name}(${rayMasks ? 'mask: u32' : ''}) -> bool {
    let header = bvhTraceTopLevel[0];
    let instanceCount = header.x;
    if (header.y == 0u) {
        return false;
    }
    let firstTopLevelNode = BVH_TOP_HEADER_VEC4S + instanceCount * BVH_TOP_VEC4_PER_RECORD;

    var insideInstance = false;
    var instance: BvhInstance;
    var record = 0u;                        // the instance record being walked
    var placement = bvhQueryIdentity();     // its object-to-world map
    var meshVertexBase = 0u;                // indexed leaves: its mesh's first vertex

    // Child references as the ray walk's, and beside each the score it was pushed with.
    var stack: array<u32, BVH_STACK_SIZE>;
    var scores: array<f32, BVH_STACK_SIZE>;
    var stackSize = 0u;
    var current = bvhMakeReference(0u, 0u);   // the root of the top level

    loop {
        let count = current >> BVH_REF_COUNT_SHIFT;
        let index = current & BVH_REF_INDEX_MASK;

        if (count == 0u) {
            // An inner node: score both children in world space, enter the lower, push the other.
            var node: BvhNode;
            var left: BvhShapecastBounds;
            var right: BvhShapecastBounds;
            if (insideInstance) {
                node = bvhLoadMeshNode(instance.rootNodeOffset + index);
                left = bvhQueryWorldBounds(placement, node.leftMin, node.leftMax);
                right = bvhQueryWorldBounds(placement, node.rightMin, node.rightMax);
            } else {
                node = bvhLoadTopLevelNode(firstTopLevelNode, index);
                left = BvhShapecastBounds(node.leftMin, node.leftMax);
                right = BvhShapecastBounds(node.rightMin, node.rightMax);
            }
            let leftScore = ${hooks.bounds}(left);
            var rightScore = -1.0;
            if (node.rightIndex != BVH_EMPTY_CHILD) {
                rightScore = ${hooks.bounds}(right);
            }
            let leftReference = bvhMakeReference(node.leftIndex, node.leftCount);
            let rightReference = bvhMakeReference(node.rightIndex, node.rightCount);
            let hitLeft = leftScore >= 0.0;
            let hitRight = rightScore >= 0.0;

            if (hitLeft && hitRight) {
                if (leftScore <= rightScore) {
                    stack[stackSize] = rightReference;
                    scores[stackSize] = rightScore;
                    current = leftReference;
                } else {
                    stack[stackSize] = leftReference;
                    scores[stackSize] = leftScore;
                    current = rightReference;
                }
                stackSize += 1u;
                continue;
            }
            if (hitLeft) {
                current = leftReference;
                continue;
            }
            if (hitRight) {
                current = rightReference;
                continue;
            }
            // Neither child: fall through to pop.
        } else if (!insideInstance) {
            // A top-level leaf is one instance (its world box was scored as the child above).
            instance = bvhLoadInstance(index);
            ${rayMasks ? 'if ((bvhInstanceMask(instance) & mask) != 0u) {' : '{'}
                record = index;
                placement = bvhQueryPlacement(instance);
                meshVertexBase = bvhMeshVertexBase(instance.triangleOffset);
                bvhEnterMeshNodes(instance.rootNodeOffset);
                stack[stackSize] = BVH_REF_INSTANCE_MARKER;
                stackSize += 1u;
                insideInstance = true;
                current = bvhMakeReference(0u, 0u);
                continue;
            }
        } else {
            // A mesh leaf: 'count' triangles starting at 'index', handed over in world space.
            for (var i = 0u; i < count; i += 1u) {
                let slot = instance.triangleOffset + index + i;
                let triangle = bvhLoadTriangle(slot, meshVertexBase);
                let candidate = BvhShapecastTriangle(
                    bvhQueryToWorld(placement, triangle.vertex0),
                    bvhQueryToWorld(placement, triangle.vertex1),
                    bvhQueryToWorld(placement, triangle.vertex2),
                    bvhHitInstance(instance, triangle, slot),
                    triangle.originalIndex,
                    placement.mirrored != ((triangle.flags & BVH_TRIANGLE_FLAG_FLIP_FRONT_FACE) != 0u),
                    record,
                    slot,
                );
                if (${hooks.triangle}(candidate)) {
                    return true;
                }
            }${restoreFrame}
        }

        // Take the next entry off the stack that the query still wants.
        loop {
            if (stackSize == 0u) {
                return false;
            }
            stackSize -= 1u;
            let entry = stack[stackSize];
            if (entry == BVH_REF_INSTANCE_MARKER) {
                insideInstance = false;
                continue;
            }
            if (!${hooks.keep}(scores[stackSize])) {
                continue;
            }
            current = entry;
            break;
        }
    }
    // No return here: the loop above only exits through 'return'.
}
`;
}

/**
 * The identity fast path's test (a lever, 2026-11-16). An instance whose
 * world-to-object rows equal the identity (`-0` counts as 0) has the identity
 * as its object-to-world map too: the adjugate of the identity is the
 * identity and `1 / 1` is 1. The shapecast walk then builds its padded world
 * boxes (`bvhQueryWorldBounds`) and moves triangle corners
 * (`bvhQueryToWorld`) without the matrix, and gets the general path's bits:
 * every term it leaves out is a product with an exact 0 or an exact 1, which
 * changes no normal finite value whatever the order or fusing of the sum.
 * Values where it could change a bit (a zero, whose sign the zeros' signs
 * decide; a subnormal, which a multiply may flush; inf or NaN) still go
 * through the matrix (`bvhQueryIdentityExact`). So boxes, corners and every
 * answer are the same bits as before; query-oracle checks that on each GPU.
 * It does not drop the box padding: an identity instance's boxes stay padded
 * (a merged instance's are exact), because exact boxes could change the walk
 * order and so which of two triangles at the same distance a closest point
 * names.
 *
 * Measured on the RTX 3060 (docs/internals/WINDOWS-RUNBOOK.md, section 12;
 * `docs/internals/design/trace.md`, "Spatial queries"). To undo it, make
 * this function return `false` ({@link withoutQueryIdentityFastPath}:
 * query-oracle's identity check and compare/sdf's `-general` kernels do that
 * to check and time it).
 */
export const QUERY_IDENTITY_CHECK_WGSL = /* wgsl */ `fn bvhQueryIsIdentity(instance: BvhInstance) -> bool {
    return all(instance.worldToObjectRow0 == vec4f(1.0, 0.0, 0.0, 0.0))
        && all(instance.worldToObjectRow1 == vec4f(0.0, 1.0, 0.0, 0.0))
        && all(instance.worldToObjectRow2 == vec4f(0.0, 0.0, 1.0, 0.0));
}`;

/**
 * A trace module's text with the identity fast path off ({@link QUERY_IDENTITY_CHECK_WGSL}
 * answering false): every instance takes the general path, as before 2026-11-16.
 * For the pages that check and time the lever; throws if the module has no queries.
 */
export function withoutQueryIdentityFastPath(module: string): string {
    if (!module.includes(QUERY_IDENTITY_CHECK_WGSL)) throw new Error('withoutQueryIdentityFastPath: the module has no query identity check');
    return module.replace(QUERY_IDENTITY_CHECK_WGSL, 'fn bvhQueryIsIdentity(instance: BvhInstance) -> bool {\n    return false;\n}');
}

/** The shapecast's structs and the world-space placement of an instance, shared by every stamped walk. */
const SHAPECAST_COMMON_WGSL = /* wgsl */ `
// ---- Spatial queries: the shapecast walk's types (query-wgsl.ts) --------------

// A box the walk asks about, in world space: a top-level box as stored, a mesh
// box as the world box around it moved by its instance, padded for the f32 transform's rounding.
struct BvhShapecastBounds {
    min: vec3f,
    max: vec3f,
}

// A triangle the walk hands over, its corners in world space.
struct BvhShapecastTriangle {
    a: vec3f,
    b: vec3f,
    c: vec3f,
    instance: u32,   // its instance's instanceIndex (a merged triangle: its source's), as BvhHit.instance
    triangle: u32,   // the original triangle index in that instance's mesh, as BvhHit.triangle
    mirrored: bool,  // the world-space winding is the reverse of the authored one (a mirrored instance or merged source)
    record: u32,     // where it lies in the scene's buffers: the instance record...
    slot: u32,       // ...and the triangle's slot (for bvhShapecastMaterial)
}

// An instance's object-to-world map, made once per instance entered.
struct BvhQueryPlacement {
    linear: mat3x3f,      // object-to-world, by columns
    absolute: mat3x3f,    // its entries' absolute values, for box half sizes
    translation: vec3f,
    transformed: bool,    // false for a merged mesh, already in world space
    identity: bool,       // the map is exactly the identity: boxes and corners may skip the matrix
    mirrored: bool,       // the map's determinant is negative
}

const BVH_QUERY_BOX_PADDING: f32 = ${QUERY_BOX_PADDING};
const BVH_QUERY_F32_MAX: f32 = 3.4e38;            // below the largest f32 (a bound, not the exact value)
const BVH_QUERY_F32_MIN_NORMAL: f32 = 1.2e-38;    // above the smallest normal f32

fn bvhQueryIdentity() -> BvhQueryPlacement {
    let identity = mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));
    return BvhQueryPlacement(identity, identity, vec3f(0.0), false, false, false);
}

// Whether an instance's world-to-object rows are exactly the identity (then so is its object-to-world map).
${QUERY_IDENTITY_CHECK_WGSL}

// The inverse of the instance's world-to-object rows, by the adjugate: column i of the
// inverse of a matrix with rows r0, r1, r2 is the cross of the other two rows over the determinant.
fn bvhQueryPlacement(instance: BvhInstance) -> BvhQueryPlacement {
    if ((instance.flags & BVH_INSTANCE_FLAG_MERGED) != 0u) {
        return bvhQueryIdentity();
    }
    let r0 = instance.worldToObjectRow0;
    let r1 = instance.worldToObjectRow1;
    let r2 = instance.worldToObjectRow2;
    let c0 = cross(r1.xyz, r2.xyz);
    let c1 = cross(r2.xyz, r0.xyz);
    let c2 = cross(r0.xyz, r1.xyz);
    let determinant = dot(r0.xyz, c0);
    let inverseDeterminant = 1.0 / determinant;
    let linear = mat3x3f(c0 * inverseDeterminant, c1 * inverseDeterminant, c2 * inverseDeterminant);
    let absolute = mat3x3f(abs(linear[0]), abs(linear[1]), abs(linear[2]));
    let translation = -(linear * vec3f(r0.w, r1.w, r2.w));
    return BvhQueryPlacement(linear, absolute, translation, true, bvhQueryIsIdentity(instance), determinant < 0.0);
}

// The identity fast path's guard: true when leaving the matrix out gives its bits. Under the
// exact identity a moved value is 1 * x + 0 * y + 0 * z + 0, which is x for any normal finite
// x, whatever the zeros' signs and however the sum is ordered or fused. A zero (the zeros' signs
// decide its sign), a subnormal (1 * x may flush it to 0) or a non-finite value (0 * inf is NaN)
// goes through the matrix; 'size' (a half size, multiplied by the absolute map) may also be 0.
fn bvhQueryIdentityExact(value: vec3f, size: vec3f) -> bool {
    let normal = (abs(value) >= vec3f(BVH_QUERY_F32_MIN_NORMAL)) & (abs(value) <= vec3f(BVH_QUERY_F32_MAX));
    let sized = ((abs(size) >= vec3f(BVH_QUERY_F32_MIN_NORMAL)) & (abs(size) <= vec3f(BVH_QUERY_F32_MAX))) | (size == vec3f(0.0));
    return all(normal & sized);
}

fn bvhQueryToWorld(placement: BvhQueryPlacement, point: vec3f) -> vec3f {
    if (!placement.transformed || (placement.identity && bvhQueryIdentityExact(point, vec3f(0.0)))) {
        return point;
    }
    return placement.linear * point + placement.translation;
}

fn bvhQueryWorldBounds(placement: BvhQueryPlacement, boxMin: vec3f, boxMax: vec3f) -> BvhShapecastBounds {
    if (!placement.transformed) {
        return BvhShapecastBounds(boxMin, boxMax);
    }
    let local = (boxMin + boxMax) * 0.5;
    let localHalf = (boxMax - boxMin) * 0.5;
    if (placement.identity && bvhQueryIdentityExact(local, localHalf)) {
        // The lines below with the identity's exact 1s and 0s taken out: the centre is 'local',
        // the absolute map leaves sizes as they are, the translation is 0. The same padded box.
        let grown = localHalf + (abs(local) + localHalf) * BVH_QUERY_BOX_PADDING;
        return BvhShapecastBounds(local - grown, local + grown);
    }
    let center = placement.linear * local + placement.translation;
    let half = placement.absolute * ((boxMax - boxMin) * 0.5);
    // Padded by the terms' sizes, not the result's: far-off terms that cancel round as large numbers.
    let grown = half + (placement.absolute * abs(local) + abs(placement.translation) + half) * BVH_QUERY_BOX_PADDING;
    return BvhShapecastBounds(center - grown, center + grown);
}

// Squared distance from a point to a box (0 inside it).
fn bvhPointBoxDistanceSq(point: vec3f, boxMin: vec3f, boxMax: vec3f) -> f32 {
    let outside = max(max(boxMin - point, point - boxMax), vec3f(0.0));
    return dot(outside, outside);
}

// A triangle's material, as BvhHit.material: its own (triangleMaterials, or a merged
// triangle's source's) when it has one, else its instance's, else BVH_NO_MATERIAL.
fn bvhShapecastMaterial(triangle: BvhShapecastTriangle) -> u32 {
    let instance = bvhLoadInstance(triangle.record);
    let stored = bvhLoadTriangle(triangle.slot, bvhMeshVertexBase(instance.triangleOffset));
    let triangleCode = bvhTriangleMaterialCode(instance, stored, triangle.slot);
    return bvhMaterialFromCode(select(instance.flags >> BVH_MATERIAL_SHIFT, triangleCode, triangleCode != 0u));
}

// The world-space normal of a shapecast triangle's authored front side, unit length (0 if degenerate).
fn bvhShapecastNormal(triangle: BvhShapecastTriangle) -> vec3f {
    let normal = cross(triangle.b - triangle.a, triangle.c - triangle.a);
    let normalLength = length(normal);
    let unit = select(vec3f(0.0), normal / normalLength, normalLength > 0.0);
    return select(unit, -unit, triangle.mirrored);
}
`;

/** `fn name(args) -> bool` forwarding to a stamped walk, with and without the mask. */
function entryPoints(rayMasks: boolean, name: string, parameters: string, setup: string, walk: string, result: string, returnType: string): string {
    const body = (mask: string) => `${setup === '' ? '' : `${setup}\n    `}${result === '' ? `return ${walk}(${mask});` : `_ = ${walk}(${mask});\n    return ${result};`}`;
    const plain = `fn ${name}(${parameters}) -> ${returnType} {
    ${body(rayMasks ? 'BVH_RAY_MASK_ALL' : '')}
}`;
    if (!rayMasks) return plain;
    return `${plain}

// As ${name}, but instances whose mask shares no bit with 'mask' are left out.
fn ${name}Masked(${parameters === '' ? '' : `${parameters}, `}mask: u32) -> ${returnType} {
    ${body('mask')}
}`;
}

/** The library's queries: closest point, sphere, axis-aligned and oriented box. */
function libraryQueriesWgsl(rayMasks: boolean): string {
    return /* wgsl */ `
// ---- Spatial queries: closest point (bvhClosestPoint) ---------------------------

// The triangle point nearest to a query point, over the whole scene.
struct BvhClosestHit {
    distance: f32,        // world-space distance; negative when nothing is nearer than maxDistance
    instance: u32,        // as BvhHit.instance
    triangle: u32,        // as BvhHit.triangle
    material: u32,        // as BvhHit.material (BVH_NO_MATERIAL for none, and on a miss)
    barycentrics: vec2f,  // weights of corners 1 and 2 of the triangle
    point: vec3f,         // the nearest point, in world space
    normal: vec3f,        // world-space normal of the triangle's authored front side, unit length; 0 on a miss
}

var<private> bvhClosestQuery: vec3f;
var<private> bvhClosestBestSq: f32;
var<private> bvhClosestFound: bool;
var<private> bvhClosestTriangleFound: BvhShapecastTriangle;
var<private> bvhClosestPointFound: BvhTrianglePoint;

const BVH_CLOSEST_MAX_DISTANCE: f32 = ${CLOSEST_POINT_MAX_DISTANCE.toExponential()};

fn bvhClosestBounds(bounds: BvhShapecastBounds) -> f32 {
    let distanceSq = bvhPointBoxDistanceSq(bvhClosestQuery, bounds.min, bounds.max);
    return select(-1.0, distanceSq, distanceSq < bvhClosestBestSq);
}

fn bvhClosestKeep(score: f32) -> bool {
    return score < bvhClosestBestSq;
}

fn bvhClosestTriangle(triangle: BvhShapecastTriangle) -> bool {
    let nearest = bvhClosestPointToTriangle(bvhClosestQuery, triangle.a, triangle.b, triangle.c);
    let offset = nearest.point - bvhClosestQuery;
    let distanceSq = dot(offset, offset);
    if (distanceSq < bvhClosestBestSq) {
        bvhClosestBestSq = distanceSq;
        bvhClosestFound = true;
        bvhClosestTriangleFound = triangle;
        bvhClosestPointFound = nearest;
        // On the surface: nothing can be nearer.
        return distanceSq == 0.0;
    }
    return false;
}

${shapecastWalkWgsl('bvhClosestPointWalk', { bounds: 'bvhClosestBounds', keep: 'bvhClosestKeep', triangle: 'bvhClosestTriangle' }, rayMasks)}

fn bvhClosestPointStart(point: vec3f, maxDistance: f32) {
    let limit = clamp(maxDistance, 0.0, BVH_CLOSEST_MAX_DISTANCE);
    bvhClosestQuery = point;
    bvhClosestBestSq = limit * limit;
    bvhClosestFound = false;
}

fn bvhClosestPointResult() -> BvhClosestHit {
    if (!bvhClosestFound) {
        return BvhClosestHit(-1.0, 0u, 0u, BVH_NO_MATERIAL, vec2f(0.0), vec3f(0.0), vec3f(0.0));
    }
    let triangle = bvhClosestTriangleFound;
    return BvhClosestHit(
        sqrt(bvhClosestBestSq), triangle.instance, triangle.triangle, bvhShapecastMaterial(triangle),
        bvhClosestPointFound.barycentrics, bvhClosestPointFound.point, bvhShapecastNormal(triangle),
    );
}

// The point of the scene's triangles nearest to 'point' (world space), among those
// nearer than maxDistance (strictly; clamped to BVH_CLOSEST_MAX_DISTANCE). Every triangle
// counts: the triangle filter is not asked. Of triangles at the same distance, one is returned.
${entryPoints(rayMasks, 'bvhClosestPoint', 'point: vec3f, maxDistance: f32', 'bvhClosestPointStart(point, maxDistance);', 'bvhClosestPointWalk', 'bvhClosestPointResult()', 'BvhClosestHit')}

// ---- Spatial queries: sphere (bvhIntersectsSphere) ------------------------------

var<private> bvhSphereCenter: vec3f;
var<private> bvhSphereRadiusSq: f32;

fn bvhSphereBounds(bounds: BvhShapecastBounds) -> f32 {
    let distanceSq = bvhPointBoxDistanceSq(bvhSphereCenter, bounds.min, bounds.max);
    return select(-1.0, distanceSq, distanceSq <= bvhSphereRadiusSq);
}

fn bvhSphereKeep(score: f32) -> bool {
    return true;
}

fn bvhSphereTriangle(triangle: BvhShapecastTriangle) -> bool {
    let offset = bvhClosestPointToTriangle(bvhSphereCenter, triangle.a, triangle.b, triangle.c).point - bvhSphereCenter;
    return dot(offset, offset) <= bvhSphereRadiusSq;
}

${shapecastWalkWgsl('bvhSphereWalk', { bounds: 'bvhSphereBounds', keep: 'bvhSphereKeep', triangle: 'bvhSphereTriangle' }, rayMasks)}

// Whether any triangle touches the solid sphere (world space; touching counts).
${entryPoints(rayMasks, 'bvhIntersectsSphere', 'center: vec3f, radius: f32', 'bvhSphereCenter = center;\n    bvhSphereRadiusSq = radius * radius;', 'bvhSphereWalk', '', 'bool')}

// ---- Spatial queries: boxes (bvhIntersectsBox, bvhIntersectsOrientedBox) ---------

var<private> bvhBoxCenter: vec3f;
var<private> bvhBoxHalf: vec3f;          // half size along the box's own axes
var<private> bvhBoxAxes: mat3x3f;        // the box's axes in world space, by columns (orthonormal)
var<private> bvhBoxWorldHalf: vec3f;     // half size of the world box around it

// Whether the triangle's corners, projected on 'axis', miss the box's interval [-radius, radius].
fn bvhBoxSeparatedOn(axis: vec3f, half: vec3f, a: vec3f, b: vec3f, c: vec3f) -> bool {
    let radius = dot(half, abs(axis));
    let pa = dot(axis, a);
    let pb = dot(axis, b);
    let pc = dot(axis, c);
    return min(pa, min(pb, pc)) > radius || max(pa, max(pb, pc)) < -radius;
}

// The separating-axis test of a box centred at the origin with half size 'half' and
// a triangle in the box's frame (Akenine-Moller): the box's 3 face normals, the
// triangle's normal, and the 9 crosses of the box's axes with the triangle's edges.
// Touching counts as overlapping.
fn bvhBoxTriangleOverlap(half: vec3f, a: vec3f, b: vec3f, c: vec3f) -> bool {
    if (any(min(a, min(b, c)) > half) || any(max(a, max(b, c)) < -half)) {
        return false;
    }
    let ab = b - a;
    let bc = c - b;
    let ca = a - c;
    if (bvhBoxSeparatedOn(cross(ab, bc), half, a, b, c)) {
        return false;
    }
    // cross(x, e) = (0, -e.z, e.y); cross(y, e) = (e.z, 0, -e.x); cross(z, e) = (-e.y, e.x, 0).
    if (bvhBoxSeparatedOn(vec3f(0.0, -ab.z, ab.y), half, a, b, c) || bvhBoxSeparatedOn(vec3f(ab.z, 0.0, -ab.x), half, a, b, c) || bvhBoxSeparatedOn(vec3f(-ab.y, ab.x, 0.0), half, a, b, c)) {
        return false;
    }
    if (bvhBoxSeparatedOn(vec3f(0.0, -bc.z, bc.y), half, a, b, c) || bvhBoxSeparatedOn(vec3f(bc.z, 0.0, -bc.x), half, a, b, c) || bvhBoxSeparatedOn(vec3f(-bc.y, bc.x, 0.0), half, a, b, c)) {
        return false;
    }
    if (bvhBoxSeparatedOn(vec3f(0.0, -ca.z, ca.y), half, a, b, c) || bvhBoxSeparatedOn(vec3f(ca.z, 0.0, -ca.x), half, a, b, c) || bvhBoxSeparatedOn(vec3f(-ca.y, ca.x, 0.0), half, a, b, c)) {
        return false;
    }
    return true;
}

// The query box against a world box: the world axes (the box around the query box),
// then the query box's own three axes, as three-mesh-bvh's OrientedBox.intersectsBox.
fn bvhOrientedBoxBounds(bounds: BvhShapecastBounds) -> f32 {
    if (any(bvhBoxCenter - bvhBoxWorldHalf > bounds.max) || any(bvhBoxCenter + bvhBoxWorldHalf < bounds.min)) {
        return -1.0;
    }
    let offset = (bounds.min + bounds.max) * 0.5 - bvhBoxCenter;
    let half = (bounds.max - bounds.min) * 0.5;
    let distance = abs(transpose(bvhBoxAxes) * offset);
    let reach = bvhBoxHalf + transpose(mat3x3f(abs(bvhBoxAxes[0]), abs(bvhBoxAxes[1]), abs(bvhBoxAxes[2]))) * half;
    return select(0.0, -1.0, any(distance > reach));
}

fn bvhOrientedBoxKeep(score: f32) -> bool {
    return true;
}

fn bvhOrientedBoxTriangle(triangle: BvhShapecastTriangle) -> bool {
    let toBox = transpose(bvhBoxAxes);
    return bvhBoxTriangleOverlap(bvhBoxHalf, toBox * (triangle.a - bvhBoxCenter), toBox * (triangle.b - bvhBoxCenter), toBox * (triangle.c - bvhBoxCenter));
}

${shapecastWalkWgsl('bvhOrientedBoxWalk', { bounds: 'bvhOrientedBoxBounds', keep: 'bvhOrientedBoxKeep', triangle: 'bvhOrientedBoxTriangle' }, rayMasks)}

fn bvhOrientedBoxStart(center: vec3f, halfSize: vec3f, rotation: mat3x3f) {
    bvhBoxCenter = center;
    bvhBoxHalf = halfSize;
    bvhBoxAxes = rotation;
    bvhBoxWorldHalf = mat3x3f(abs(rotation[0]), abs(rotation[1]), abs(rotation[2])) * halfSize;
}

// Whether any triangle touches the solid box centred at 'center' with half size 'halfSize'
// along its axes, the columns of 'rotation' (orthonormal; fold a scale into halfSize).
// Touching counts.
${entryPoints(rayMasks, 'bvhIntersectsOrientedBox', 'center: vec3f, halfSize: vec3f, rotation: mat3x3f', 'bvhOrientedBoxStart(center, halfSize, rotation);', 'bvhOrientedBoxWalk', '', 'bool')}

// Whether any triangle touches the solid axis-aligned box [boxMin, boxMax] (world space).
${entryPoints(rayMasks, 'bvhIntersectsBox', 'boxMin: vec3f, boxMax: vec3f', 'bvhOrientedBoxStart((boxMin + boxMax) * 0.5, (boxMax - boxMin) * 0.5, mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0)));', 'bvhOrientedBoxWalk', '', 'bool')}
${insideTestWgsl(rayMasks)}`;
}

/**
 * The inside test (PARITY 8.10): every crossing of a ray with the scene's
 * triangles, counted and signed, through the shapecast walk with a ray-box
 * score (nearer boxes first, though order does not matter: nothing stops it)
 * and the walk's own watertight triangle test on the world-space corners. A
 * corner shared by two triangles of one instance moves to the same world bits
 * in both, so a ray cannot slip through a shared edge; a ray exactly on an
 * edge counts both triangles (the walk's rule), which the vote of three rays
 * outweighs.
 *
 * `bvhIsInside` and `bvhIsInsideByRule` (the SDF kernel's sign) cast the third
 * ray only when the first two disagree: the same answer as the majority of
 * all three, a third fewer walks where the first two agree (most points; on
 * the RTX 3060 it took the signed SDF 8% faster, docs/tracing/sdf.mdx, "Speed").
 * To undo it, return `bvhInsideVotes(point)` by the rule `>= BVH_INSIDE_MAJORITY`
 * from `bvhIsInsideByRule`, as before 2026-11-16.
 */
function insideTestWgsl(rayMasks: boolean): string {
    const [d0, d1, d2] = INSIDE_TEST_DIRECTIONS.map(wgslVec3);
    const votes = (masked: boolean) => {
        const call = (direction: string) => `bvhInsideVote(bvhRayCrossings${masked ? 'Masked' : ''}(point, ${direction}${masked ? ', mask' : ''}))`;
        return `${call('BVH_INSIDE_DIRECTION_0')}\n        + ${call('BVH_INSIDE_DIRECTION_1')}\n        + ${call('BVH_INSIDE_DIRECTION_2')}`;
    };
    // The majority of three by one rule, the third ray only on a split.
    const majority = (masked: boolean) => {
        const vote = (direction: string) => `bvhInsideRuleVote(bvhRayCrossings${masked ? 'Masked' : ''}(point, ${direction}${masked ? ', mask' : ''}), winding)`;
        return `let first = ${vote('BVH_INSIDE_DIRECTION_0')};
    if (first == ${vote('BVH_INSIDE_DIRECTION_1')}) {
        return first;
    }
    return ${vote('BVH_INSIDE_DIRECTION_2')};`;
    };
    const maskedForms = rayMasks
        ? `

// As bvhInsideVotes, over the instances whose mask shares a bit with 'mask'.
fn bvhInsideVotesMasked(point: vec3f, mask: u32) -> vec2u {
    return ${votes(true)};
}

// As bvhIsInsideByRule, over the instances whose mask shares a bit with 'mask'.
fn bvhIsInsideByRuleMasked(point: vec3f, winding: bool, mask: u32) -> bool {
    ${majority(true)}
}

// As bvhIsInside, over the instances whose mask shares a bit with 'mask'.
fn bvhIsInsideMasked(point: vec3f, mask: u32) -> bool {
    return bvhIsInsideByRuleMasked(point, false, mask);
}`
        : '';
    return /* wgsl */ `
// ---- Spatial queries: inside test (bvhRayCrossings, bvhInsideVotes, bvhIsInside) ----

// A ray's crossings with the scene's triangles, at t >= 0 (world space).
struct BvhRayCrossings {
    count: u32,     // every triangle the ray crosses
    winding: i32,   // +1 per triangle left through its authored front side, -1 per one entered through it
}

// The inside test's rays (INSIDE_TEST_DIRECTIONS): along no axis or diagonal, the same f32 values as the CPU's.
const BVH_INSIDE_DIRECTION_0: vec3f = ${d0};
const BVH_INSIDE_DIRECTION_1: vec3f = ${d1};
const BVH_INSIDE_DIRECTION_2: vec3f = ${d2};
// Votes (of three) that make a point inside.
const BVH_INSIDE_MAJORITY: u32 = ${INSIDE_TEST_MAJORITY}u;

var<private> bvhCrossingSetup: BvhRaySetup;
var<private> bvhCrossingCount: u32;
var<private> bvhCrossingWinding: i32;

fn bvhCrossingBounds(bounds: BvhShapecastBounds) -> f32 {
    let entry = bvhBoxEntry(bounds.min, bounds.max, bvhCrossingSetup, 0.0, BVH_BOX_MISS);
    return select(-1.0, entry, entry < BVH_BOX_MISS);
}

fn bvhCrossingKeep(score: f32) -> bool {
    return true;
}

fn bvhCrossingTriangle(triangle: BvhShapecastTriangle) -> bool {
    let hit = bvhIntersectWatertight(BvhTriangle(triangle.a, 0u, triangle.b, 0u, triangle.c, 0u), bvhCrossingSetup, 0.0, BVH_BOX_MISS);
    if (hit.hit) {
        bvhCrossingCount += 1u;
        // frontFacing is against the world winding; a mirrored triangle's authored side is the other one.
        bvhCrossingWinding += select(1, -1, hit.frontFacing != triangle.mirrored);
    }
    return false;
}

${shapecastWalkWgsl('bvhCrossingWalk', { bounds: 'bvhCrossingBounds', keep: 'bvhCrossingKeep', triangle: 'bvhCrossingTriangle' }, rayMasks)}

fn bvhRayCrossingsStart(origin: vec3f, direction: vec3f) {
    bvhCrossingSetup = bvhMakeRaySetup(origin, direction);
    bvhCrossingCount = 0u;
    bvhCrossingWinding = 0;
}

// Every triangle the ray from 'origin' along 'direction' crosses at t >= 0 (world space; the
// triangle filter is not asked): how many, and their signed sum. The walk cannot stop early.
${entryPoints(rayMasks, 'bvhRayCrossings', 'origin: vec3f, direction: vec3f', 'bvhRayCrossingsStart(origin, direction);', 'bvhCrossingWalk', 'BvhRayCrossings(bvhCrossingCount, bvhCrossingWinding)', 'BvhRayCrossings')}

// One ray's vote: (odd crossings, non-zero winding), each 1 for inside.
fn bvhInsideVote(crossings: BvhRayCrossings) -> vec2u {
    return vec2u(crossings.count & 1u, select(0u, 1u, crossings.winding != 0));
}

// One ray's vote by one rule: odd crossings (parity), or a non-zero winding.
fn bvhInsideRuleVote(crossings: BvhRayCrossings, winding: bool) -> bool {
    return select((crossings.count & 1u) == 1u, crossings.winding != 0, winding);
}

// The inside test's votes for 'point' over its three rays: x by parity, y by winding
// (each 0 to 3; 2 or more is inside). Three full ray walks.
fn bvhInsideVotes(point: vec3f) -> vec2u {
    return ${votes(false)};
}

// Whether at least two of the three rays vote 'point' inside, by the winding rule when
// 'winding', else by parity: bvhInsideVotes(point) by that rule >= 2u, but the third ray
// is cast only when the first two disagree (two walks for most points, three at most).
fn bvhIsInsideByRule(point: vec3f, winding: bool) -> bool {
    ${majority(false)}
}

// Whether 'point' is inside the scene's closed surfaces: odd crossings on at least two of
// three rays (parity; bvhIsInsideByRule(point, true) for the winding rule).
fn bvhIsInside(point: vec3f) -> bool {
    return bvhIsInsideByRule(point, false);
}${maskedForms}
`;
}

/** The caller's shapecast: the walk over its hooks, and its entry points. */
function userShapecastWgsl(rayMasks: boolean, hooks: string): string {
    return /* wgsl */ `
// ---- Spatial queries: the caller's shapecast (bvhShapecast) ----------------------

${shapecastWalkWgsl('bvhShapecastWalk', { bounds: 'bvhShapecastBounds', keep: 'bvhShapecastKeep', triangle: 'bvhShapecastTriangle' }, rayMasks, true)}

// Walks the scene with the caller's bvhShapecastBounds, bvhShapecastKeep and
// bvhShapecastTriangle; true when bvhShapecastTriangle stopped it. bvhShapecastTriangle
// may call the module's other walks (bvhTraceAny, bvhClosestPoint, bvhIntersects*);
// bvhShapecastBounds and bvhShapecastKeep must not.
${entryPoints(rayMasks, 'bvhShapecast', '', '', 'bvhShapecastWalk', '', 'bool')}

${hooks === '' ? '// bvhShapecastBounds, bvhShapecastKeep and bvhShapecastTriangle are supplied by the kernel.' : hooks}
`;
}

/** Options of {@link queriesWgsl}, as `traceModuleWgsl` passes them. */
export interface QueriesWgslOptions {
    readonly rayMasks: boolean;
    /** The library's queries (closest point, sphere, boxes). */
    readonly queries: boolean;
    /** The caller's shapecast hooks, `''` for hooks the kernel declares, or undefined for no shapecast. */
    readonly shapecast?: string;
}

/**
 * The query part of a trace module: empty unless asked for, so a module
 * without queries is the ray walk's alone. It uses the module's loaders,
 * `bvhInstanceMask` (with ray masks) and `bvhClosestPointToTriangle`.
 */
export function queriesWgsl(options: QueriesWgslOptions): string {
    const shapecast = options.shapecast !== undefined;
    if (!options.queries && !shapecast) return '';
    return [SHAPECAST_COMMON_WGSL, options.queries ? libraryQueriesWgsl(options.rayMasks) : '', shapecast ? userShapecastWgsl(options.rayMasks, options.shapecast!) : ''].join('\n');
}

/**
 * The closest-point batch kernel's own part: its bindings in `group`
 * (queries in, results out, and with `rayMasks` the batch's mask as a
 * uniform at binding 2) and the entry point `bvhClosestPointBatch`.
 *
 *     struct BvhPointQuery { point: vec3f, maxDistance: f32 }   // 16 bytes
 *
 * Results are `BvhClosestPointRecord` ({@link CLOSEST_POINT_RECORD_BYTES}).
 */
export function closestPointBatchEntryWgsl(group: number, rayMasks: boolean, workgroupSize: number): string {
    return /* wgsl */ `
struct BvhPointQuery {
    point: vec3f,
    maxDistance: f32,
}

struct BvhClosestPointRecord {
    distance: f32,
    instance: u32,
    triangle: u32,
    material: u32,
    barycentrics: vec2f,
    pointX: f32,    // scalars, not vec3f, so the record has no padding
    pointY: f32,
    pointZ: f32,
    normalX: f32,
    normalY: f32,
    normalZ: f32,
}

@group(${group}) @binding(0) var<storage, read> bvhClosestPointQueries: array<BvhPointQuery>;
@group(${group}) @binding(1) var<storage, read_write> bvhClosestPointResults: array<BvhClosestPointRecord>;
${
    rayMasks
        ? `
struct BvhClosestPointParams {
    mask: u32,
}

@group(${group}) @binding(2) var<uniform> bvhClosestPointParams: BvhClosestPointParams;`
        : ''
}

@compute @workgroup_size(${workgroupSize})
fn bvhClosestPointBatch(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroupCount: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    // Dispatches are 2D so a very large batch does not exceed 65535 workgroups in x.
    let queryIndex = (workgroup.y * workgroupCount.x + workgroup.x) * ${workgroupSize}u + localIndex;
    if (queryIndex >= arrayLength(&bvhClosestPointQueries)) {
        return;
    }
    let query = bvhClosestPointQueries[queryIndex];
    let hit = ${rayMasks ? 'bvhClosestPointMasked(query.point, query.maxDistance, bvhClosestPointParams.mask)' : 'bvhClosestPoint(query.point, query.maxDistance)'};
    bvhClosestPointResults[queryIndex] = BvhClosestPointRecord(
        hit.distance, hit.instance, hit.triangle, hit.material, hit.barycentrics,
        hit.point.x, hit.point.y, hit.point.z, hit.normal.x, hit.normal.y, hit.normal.z,
    );
}
`;
}

/** Bytes per query of the closest-point batch (`BvhPointQuery`: point, then `maxDistance`). */
export const POINT_QUERY_BYTES = 16;

/**
 * Bytes per result of the closest-point batch (`BvhClosestPointRecord`), words in
 * {@link CLOSEST_POINT_RECORD_WORD}. A miss has `distance` -1, `material`
 * `NO_MATERIAL` and zeros elsewhere.
 */
export const CLOSEST_POINT_RECORD_BYTES = 48;

/** Word positions inside a `BvhClosestPointRecord`. */
export const CLOSEST_POINT_RECORD_WORD = {
    distance: 0,
    instance: 1,
    triangle: 2,
    material: 3,
    barycentrics: 4,
    point: 6,
    normal: 9,
} as const;

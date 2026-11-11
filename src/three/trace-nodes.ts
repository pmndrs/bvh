/**
 * The walk inside three's node system: our trace buffers as three storage
 * nodes, the WGSL walk (`traceModuleWgsl`) as one code node, and TSL helpers
 * that call it. It stands in for three-mesh-bvh's `BVHComputeData` (whose
 * `fns.raycastFirstHit` / `raycastAnyHit` are WGSL functions compiled into
 * three's shaders), over this package's layout and buffers.
 *
 * Two ways to use it (docs/three.md):
 *
 * - **WGSL**, as three-mesh-bvh's users write it: include {@link BvhTraceNodes.module}
 *   in a three `wgslFn`, then call the walk by name:
 *
 *   ```ts
 *   const shade = wgslFn(`fn shade(o: vec3f, d: vec3f) -> f32 {
 *       let hit = bvhTraceClosest(BvhRay(o, 0.0, d, 1e30));
 *       return select(0.0, hit.t, hit.t >= 0.0);
 *   }`, [bvh.module]);
 *   ```
 *
 * - **TSL**: `bvh.traceClosest(origin, direction, tMin, tMax)` returns a
 *   `BvhTraceHit` struct node (`.get('t')`, `'instance'`, `'triangle'`,
 *   `'barycentrics'`, `'frontFace'`, `'material'`, `'normal'`; `t < 0` on a
 *   miss), `bvh.traceAny(...)` a bool; the `*Masked` forms take a ray mask.
 *
 * **Buffers.** Three storage nodes (`bvhTraceNodes`, `bvhTraceTriangles`,
 * `bvhTraceTopLevel`, read-only) bound to the source's `GPUBuffer`s, which must
 * live on the renderer's device (`deviceFromRenderer`). three doesn't create
 * them: each node's `StorageBufferAttribute` is a stand-in whose backend
 * buffer is set to ours before three first sees it, so no copy is made and
 * three never writes it.
 *
 * **Rebinding.** Call {@link BvhTraceNodes.update} once per frame, after
 * `SceneTrees.update` and before three renders or computes. When the
 * source's `version` changed (SceneTrees grew a buffer), each storage node
 * gets a new stand-in attribute bound to the new buffer, and three makes new
 * bind groups at its next use. Nothing recompiles: the WGSL doesn't change.
 * Skipping it after a growth leaves three binding the old, destroyed buffer.
 *
 * **Disposal.** three tracks the stand-ins like its own attributes, so
 * `renderer.dispose()` or a stand-in's `dispose()` would destroy the
 * source's buffers. The renderer's `backend.destroyAttribute` is wrapped
 * (once per backend) so that, for a stand-in, a throwaway buffer takes our
 * buffer's place before three destroys it.
 *
 * **Compile cost.** The walk is emitted once per shader as plain WGSL
 * functions (one loop, no unrolling), and every TSL helper is one small
 * `wgslFn`; the names are fixed, so the shader text is the same on every load
 * (three's and Chrome's shader caches hit).
 *
 * This file reaches into three's WebGPU backend, which is not public API:
 * it is written against three 0.187 (the peer range), and every internal it
 * uses is in {@link threeInternals}.
 */

import { CodeNode, REVISION, StorageBufferAttribute, StructTypeNode, type Node, type NodeBuilder } from 'three/webgpu';
import { float, storage, wgslFn } from 'three/tsl';
import { traceModuleWgsl } from '../trace/wgsl.ts';

/** Where the trace buffers come from: a `SceneTrees` (buffers and version), or fixed buffers (`uploadTraceScene`). */
export interface BvhTraceBufferSource {
    /** `positions`: an indexed scene's fourth buffer (docs/design/indexed-leaves.md); null or absent for records. */
    readonly buffers: { readonly nodes: GPUBuffer; readonly triangles: GPUBuffer; readonly topLevel: GPUBuffer; readonly positions?: GPUBuffer | null };
    /** Changes whenever a buffer is replaced. Fixed buffers may omit it. */
    readonly version?: number;
    /** The source's triangle format (`SceneTrees.triangleLeaves`, `TraceSceneBuffers.triangleLeaves`); absent means records. */
    readonly triangleLeaves?: 'records' | 'indexed';
    /** An indexed source's position layout; absent means `'vec4'`. */
    readonly positionLayout?: 'vec4' | 'packed';
}

export interface BvhTraceNodesOptions {
    /** The WGSL `fn bvhTrianglePasses(candidate: BvhCandidate) -> bool`; default: every triangle blocks (see `traceModuleWgsl`). */
    readonly triangleFilter?: string;
    /** As `traceModuleWgsl`'s `rayMasks`. Default true: the `*Masked` helpers exist. */
    readonly rayMasks?: boolean;
}

/** The parts of three's `WebGPURenderer` this file uses (see {@link threeInternals}). */
interface RendererLike {
    readonly backend: BackendLike;
    readonly _attributes?: { has(attribute: object): boolean };
}

interface BackendLike {
    readonly isWebGPUBackend?: boolean;
    readonly device: GPUDevice | null;
    get(object: object): Record<string, unknown> & { buffer?: GPUBuffer };
    destroyAttribute?(attribute: object): void;
}

const unsupported = (what: string) =>
    new Error(
        `@pmndrs/bvh/three: three r${REVISION} ${what}. BvhTraceNodes binds its buffers through three's WebGPU backend internals, written against three r187 (peer range >=0.187.0 <0.188.0); this three version is unsupported`,
    );

/** Stand-ins per backend whose buffer is ours: `destroyAttribute` swaps in a throwaway for them. */
const guarded = new WeakMap<BackendLike, WeakSet<object>>();

/**
 * Every three internal BvhTraceNodes relies on, checked against three 0.187's source:
 *
 * 1. `backend.get(attribute).buffer` is a storage attribute's `GPUBuffer`.
 *    `WebGPUAttributeUtils.createAttribute` makes one only when it is
 *    undefined, and `WebGPUBindingUtils.createBindGroup` binds it for a
 *    storage binding (`backend.get(binding.attribute).buffer`).
 * 2. `backend.destroyAttribute(attribute)` destroys that buffer. Three's
 *    `Attributes.delete` calls it on the attribute's `dispose` event and for
 *    every tracked attribute in `renderer.dispose()` (`Attributes.dispose`).
 * 3. `renderer._attributes.has(attribute)` is true once three used the attribute.
 *
 * The constructor checks 2 and 3 exist; `update()` checks 1 for every
 * stand-in three used (its buffer is still ours, and three made no other).
 */
const threeInternals = {
    check(renderer: RendererLike): void {
        if (typeof renderer.backend.destroyAttribute !== 'function') throw unsupported('has no backend.destroyAttribute');
        if (typeof renderer._attributes?.has !== 'function') throw unsupported('has no renderer._attributes');
    },
    bufferOf(backend: BackendLike, attribute: object): GPUBuffer | undefined {
        return backend.get(attribute).buffer;
    },
    bind(backend: BackendLike, attribute: object, buffer: GPUBuffer): void {
        backend.get(attribute).buffer = buffer;
    },
    usedByThree(renderer: RendererLike, attribute: object): boolean {
        return renderer._attributes!.has(attribute);
    },
    /** A GPUBuffer in the backend's record other than `ours` (three made its own). */
    otherBuffer(backend: BackendLike, attribute: object, ours: GPUBuffer): boolean {
        return Object.values(backend.get(attribute)).some((value) => value !== ours && typeof (value as GPUBuffer | null)?.getMappedRange === 'function');
    },
    /** Makes `backend.destroyAttribute` spare `attribute`'s buffer: a throwaway takes its place first. */
    guardDestroy(backend: BackendLike, attribute: object): void {
        let set = guarded.get(backend);
        if (!set) {
            const ours = (set = new WeakSet());
            guarded.set(backend, ours);
            const destroy = backend.destroyAttribute!;
            backend.destroyAttribute = function (this: BackendLike, target: object) {
                if (ours.has(target)) {
                    ours.delete(target);
                    backend.get(target).buffer = backend.device!.createBuffer({ label: '@pmndrs/bvh/three retired stand-in', size: 16, usage: 0x80 /* STORAGE */ });
                }
                return destroy.call(this, target);
            };
        }
        set.add(attribute);
    },
};

/**
 * A `StorageBufferAttribute` three binds to `buffer` (ours, on the renderer's
 * device), as {@link BvhTraceNodes} binds its buffers: three never creates,
 * writes from the CPU or destroys it. For the pose pass (`deform.ts`), which
 * writes posed vertices into a buffer SceneTrees reads.
 */
export function storageStandIn(renderer: unknown, buffer: GPUBuffer, itemSize = 4): StorageBufferAttribute {
    const target = renderer as RendererLike;
    if (!target?.backend?.isWebGPUBackend || !target.backend.device) throw new Error('@pmndrs/bvh/three: needs an initialized WebGPURenderer on its WebGPU backend (await renderer.init())');
    threeInternals.check(target);
    const attribute = new StorageBufferAttribute(new Float32Array(itemSize), itemSize);
    attribute.name = buffer.label;
    threeInternals.bind(target.backend, attribute, buffer);
    threeInternals.guardDestroy(target.backend, attribute);
    return attribute;
}

/** Throws unless three, having used `attribute` ({@link storageStandIn}), still binds `buffer` and made none of its own. */
export function checkStandIn(renderer: unknown, attribute: object, buffer: GPUBuffer, what: string): void {
    const target = renderer as RendererLike;
    if (!threeInternals.usedByThree(target, attribute)) return;
    if (threeInternals.bufferOf(target.backend, attribute) !== buffer || threeInternals.otherBuffer(target.backend, attribute, buffer)) {
        throw unsupported(`did not bind the ${what} buffer @pmndrs/bvh set (backend.get(attribute).buffer)`);
    }
}

/** A three storage buffer node, as far as this file uses one. */
type StorageNode = Node & { value: unknown; setName(name: string): unknown };
type NodeInput = BvhTslNode | number;

/**
 * A TSL node, loosely typed: three's TSL types cannot follow a struct returned
 * through a `wgslFn`, so the helpers take and return this (as three-mesh-bvh's do).
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type BvhTslNode = any;

/** The struct the TSL helpers return. Its members are `BvhHit`'s. */
export const bvhTraceHitStruct = new StructTypeNode(
    {
        t: 'float',
        instance: 'uint',
        triangle: 'uint',
        barycentrics: 'vec2',
        frontFace: 'bool',
        material: 'uint',
        normal: 'vec3',
    },
    'BvhTraceHit',
);

type BufferKey = 'nodes' | 'triangles' | 'topLevel' | 'positions';
type BufferName = 'bvhTraceNodes' | 'bvhTraceTriangles' | 'bvhTraceTopLevel' | 'bvhTracePositions';
interface BufferSpec {
    readonly name: BufferName;
    readonly key: BufferKey;
    readonly type: 'vec4' | 'uvec4' | 'float';
    readonly itemSize: number;
    readonly array: () => Float32Array | Uint32Array;
}

/** The storage nodes of a format: three, or with indexed leaves four (the triangles read as `vec4u`, then the positions). */
function buffersOf(triangleLeaves: 'records' | 'indexed', positionLayout: 'vec4' | 'packed'): readonly BufferSpec[] {
    const indexed = triangleLeaves === 'indexed';
    return [
        { name: 'bvhTraceNodes', key: 'nodes', type: 'vec4', itemSize: 4, array: () => new Float32Array(4) },
        indexed
            ? { name: 'bvhTraceTriangles', key: 'triangles', type: 'uvec4', itemSize: 4, array: () => new Uint32Array(4) }
            : { name: 'bvhTraceTriangles', key: 'triangles', type: 'vec4', itemSize: 4, array: () => new Float32Array(4) },
        { name: 'bvhTraceTopLevel', key: 'topLevel', type: 'uvec4', itemSize: 4, array: () => new Uint32Array(4) },
        ...(indexed
            ? [
                  positionLayout === 'packed'
                      ? { name: 'bvhTracePositions' as const, key: 'positions' as const, type: 'float' as const, itemSize: 1, array: () => new Float32Array(4) }
                      : { name: 'bvhTracePositions' as const, key: 'positions' as const, type: 'vec4' as const, itemSize: 4, array: () => new Float32Array(4) },
              ]
            : []),
    ];
}

/** The walk's text without its binding declarations: three declares the storage nodes instead. */
function moduleWithoutBindings(options: BvhTraceNodesOptions, triangleLeaves: 'records' | 'indexed', positionLayout: 'vec4' | 'packed'): string {
    const text = traceModuleWgsl({ triangleFilter: options.triangleFilter, rayMasks: options.rayMasks, triangleLeaves, positionLayout });
    let removed = 0;
    const stripped = text.replace(/^@group\(\d+\) @binding\(\d+\) var<storage, read> bvhTrace(?:Nodes|Triangles|TopLevel|Positions): array<(?:vec4[fu]|f32)>;$/gm, () => {
        removed++;
        return '';
    });
    const expected = triangleLeaves === 'indexed' ? 4 : 3;
    if (removed !== expected) throw new Error(`@pmndrs/bvh/three: expected the walk's ${expected} bindings, found ${removed}`);
    return stripped;
}

/** The walk as a three code node: its buffer names become the storage nodes' references in each shader. */
class BvhTraceModuleNode extends CodeNode {
    // Node-valued properties: three sets them up and analyzes them as this node's children.
    bvhTraceNodes: StorageNode;
    bvhTraceTriangles: StorageNode;
    bvhTraceTopLevel: StorageNode;
    /** Indexed leaves only. */
    bvhTracePositions: StorageNode | null;

    constructor(
        private readonly text: string,
        storages: Readonly<Partial<Record<BufferName, StorageNode>>>,
    ) {
        super('', [], 'wgsl');
        this.bvhTraceNodes = storages.bvhTraceNodes!;
        this.bvhTraceTriangles = storages.bvhTraceTriangles!;
        this.bvhTraceTopLevel = storages.bvhTraceTopLevel!;
        this.bvhTracePositions = storages.bvhTracePositions ?? null;
    }

    generate(builder: NodeBuilder): string {
        const names: Record<string, string> = {
            bvhTraceNodes: this.bvhTraceNodes.build(builder, 'property') as string,
            bvhTraceTriangles: this.bvhTraceTriangles.build(builder, 'property') as string,
            bvhTraceTopLevel: this.bvhTraceTopLevel.build(builder, 'property') as string,
            ...(this.bvhTracePositions ? { bvhTracePositions: this.bvhTracePositions.build(builder, 'property') as string } : {}),
        };
        const code = this.text.replace(/\bbvhTrace(?:Nodes|Triangles|TopLevel|Positions)\b/g, (name) => names[name]);
        const nodeCode = (builder as unknown as { getCodeFromNode(node: Node, type: string): { code: string } }).getCodeFromNode(this, this.getNodeType(builder));
        nodeCode.code = code;
        return code;
    }
}

const closestWgsl = (name: string, masked: boolean) => /* wgsl */ `
fn ${name}(origin: vec3f, direction: vec3f, tMin: f32, tMax: f32${masked ? ', mask: u32' : ''}) -> BvhTraceHit {
    let hit = ${masked ? 'bvhTraceClosestMasked' : 'bvhTraceClosest'}(BvhRay(origin, tMin, direction, tMax)${masked ? ', mask' : ''});
    return BvhTraceHit(hit.t, hit.instance, hit.triangle, hit.barycentrics, hit.frontFace, hit.material, hit.normal);
}`;

const anyWgsl = (name: string, masked: boolean) => /* wgsl */ `
fn ${name}(origin: vec3f, direction: vec3f, tMin: f32, tMax: f32${masked ? ', mask: u32' : ''}) -> bool {
    return ${masked ? 'bvhTraceAnyMasked' : 'bvhTraceAny'}(BvhRay(origin, tMin, direction, tMax)${masked ? ', mask' : ''});
}`;

/** One `wgslFn` per helper, with a TSL-friendly call (default `tMin` 0, `tMax` 1e30). */
function helper(code: string, includes: Node[]) {
    const fn = wgslFn(code, includes as never) as unknown as (...args: unknown[]) => BvhTslNode;
    return (origin: BvhTslNode, direction: BvhTslNode, tMin: NodeInput = 0, tMax: NodeInput = 1e30, ...mask: BvhTslNode[]): BvhTslNode =>
        fn(origin, direction, typeof tMin === 'number' ? float(tMin) : tMin, typeof tMax === 'number' ? float(tMax) : tMax, ...mask);
}

/**
 * Our trace buffers in three's node system; see the file comment.
 *
 *   const bvh = new BvhTraceNodes(renderer, sceneTrees);
 *   // each frame: sync.update(); sceneTrees.update(encoder); submit; bvh.update(); renderer.compute(...)
 */
export class BvhTraceNodes {
    /** The storage nodes, for reading the layout directly (`positions` only for indexed leaves). */
    readonly storage: { readonly nodes: StorageNode; readonly triangles: StorageNode; readonly topLevel: StorageNode; readonly positions: StorageNode | null };
    /** The source's triangle format, compiled into the walk. */
    readonly triangleLeaves: 'records' | 'indexed';
    readonly positionLayout: 'vec4' | 'packed';
    private readonly specs: readonly BufferSpec[];
    /** The walk (`bvhTraceClosest`, `bvhTraceAny`, `BvhRay`, `BvhHit`, ...): include it in a `wgslFn` to call it from WGSL. */
    readonly module: Node;
    readonly rayMasks: boolean;
    /** `(origin, direction, tMin = 0, tMax = 1e30) => BvhTraceHit` struct node. */
    readonly traceClosest: (origin: BvhTslNode, direction: BvhTslNode, tMin?: NodeInput, tMax?: NodeInput) => BvhTslNode;
    /** `(origin, direction, tMin = 0, tMax = 1e30) => bool` node: anything in `[tMin, tMax]`. */
    readonly traceAny: (origin: BvhTslNode, direction: BvhTslNode, tMin?: NodeInput, tMax?: NodeInput) => BvhTslNode;
    /** As {@link traceClosest} with a `u32` ray mask; throws with `rayMasks: false`. */
    readonly traceClosestMasked: (origin: BvhTslNode, direction: BvhTslNode, tMin: NodeInput, tMax: NodeInput, mask: BvhTslNode) => BvhTslNode;
    /** As {@link traceAny} with a `u32` ray mask; throws with `rayMasks: false`. */
    readonly traceAnyMasked: (origin: BvhTslNode, direction: BvhTslNode, tMin: NodeInput, tMax: NodeInput, mask: BvhTslNode) => BvhTslNode;

    private readonly renderer: RendererLike;
    private boundVersion: number | undefined;
    private bound: BvhTraceBufferSource['buffers'];
    private disposed = false;
    private readonly verified = new WeakSet<object>();

    constructor(
        renderer: unknown,
        readonly source: BvhTraceBufferSource,
        options: BvhTraceNodesOptions = {},
    ) {
        this.renderer = renderer as RendererLike;
        if (!this.renderer.backend?.isWebGPUBackend || !this.renderer.backend.device) {
            throw new Error('@pmndrs/bvh/three: BvhTraceNodes needs an initialized WebGPURenderer on its WebGPU backend (await renderer.init())');
        }
        threeInternals.check(this.renderer);
        this.rayMasks = options.rayMasks ?? true;
        this.triangleLeaves = source.triangleLeaves ?? 'records';
        this.positionLayout = source.positionLayout ?? 'vec4';
        this.specs = buffersOf(this.triangleLeaves, this.positionLayout);
        this.bound = source.buffers;
        this.boundVersion = source.version;
        const storages: Partial<Record<BufferName, StorageNode>> = {};
        for (const { name, key, type, itemSize, array } of this.specs) {
            const node = storage(this.standIn(this.bufferOf(this.bound, key), array(), itemSize), type, 0).toReadOnly() as unknown as StorageNode;
            node.setName(name);
            storages[name] = node;
        }
        this.storage = { nodes: storages.bvhTraceNodes!, triangles: storages.bvhTraceTriangles!, topLevel: storages.bvhTraceTopLevel!, positions: storages.bvhTracePositions ?? null };
        this.module = new BvhTraceModuleNode(moduleWithoutBindings(options, this.triangleLeaves, this.positionLayout), storages);
        const includes = [bvhTraceHitStruct, this.module];
        this.traceClosest = helper(closestWgsl('bvhThreeTraceClosest', false), includes);
        this.traceAny = helper(anyWgsl('bvhThreeTraceAny', false), includes);
        const unmasked = () => {
            throw new Error('@pmndrs/bvh/three: this BvhTraceNodes was made with rayMasks: false');
        };
        this.traceClosestMasked = this.rayMasks ? helper(closestWgsl('bvhThreeTraceClosestMasked', true), includes) : unmasked;
        this.traceAnyMasked = this.rayMasks ? helper(anyWgsl('bvhThreeTraceAnyMasked', true), includes) : unmasked;
    }

    /** The version bound now (the source's at the last {@link update}). */
    get version(): number | undefined {
        return this.boundVersion;
    }

    /**
     * Rebinds the storage nodes when the source's buffers were replaced
     * (`version` changed, or a buffer object differs). Returns true when it rebound.
     * Throws when three bound something other than our buffers to a stand-in
     * it used (a three version whose internals differ; see {@link threeInternals}).
     */
    update(): boolean {
        if (this.disposed) throw new Error('@pmndrs/bvh/three: this BvhTraceNodes was disposed');
        this.checkBindings();
        const buffers = this.source.buffers;
        const changed =
            this.source.version !== this.boundVersion ||
            buffers.nodes !== this.bound.nodes ||
            buffers.triangles !== this.bound.triangles ||
            buffers.topLevel !== this.bound.topLevel ||
            (buffers.positions ?? null) !== (this.bound.positions ?? null);
        if (!changed) return false;
        for (const { key, itemSize, array } of this.specs) {
            const node = this.storage[key]!;
            const old = node.value as StorageBufferAttribute;
            node.value = this.standIn(this.bufferOf(buffers, key), array(), itemSize);
            this.retire(old);
        }
        this.bound = buffers;
        this.boundVersion = this.source.version;
        return true;
    }

    /** Each stand-in three used still has our buffer, and three made none of its own. */
    private checkBindings(): void {
        const backend = this.renderer.backend;
        for (const { key } of this.specs) {
            const attribute = this.storage[key]!.value as StorageBufferAttribute;
            if (this.verified.has(attribute) || !threeInternals.usedByThree(this.renderer, attribute)) continue;
            const ours = this.bufferOf(this.bound, key);
            if (threeInternals.bufferOf(backend, attribute) !== ours || threeInternals.otherBuffer(backend, attribute, ours)) {
                throw unsupported(`did not bind the ${key} buffer BvhTraceNodes set (backend.get(attribute).buffer)`);
            }
            this.verified.add(attribute);
        }
    }

    /** Releases the stand-in attributes. The source's buffers are not destroyed (they are the source's). */
    dispose(): void {
        if (this.disposed) return;
        for (const { key } of this.specs) this.retire(this.storage[key]!.value as StorageBufferAttribute);
        this.disposed = true;
    }

    /** A `StorageBufferAttribute` whose GPU buffer is `buffer`: three takes the backend's buffer as made, never creates one, and never destroys it. */
    private standIn(buffer: GPUBuffer, array: Float32Array | Uint32Array, itemSize = 4): StorageBufferAttribute {
        const attribute = new StorageBufferAttribute(array, itemSize);
        attribute.name = buffer.label;
        threeInternals.bind(this.renderer.backend, attribute, buffer);
        threeInternals.guardDestroy(this.renderer.backend, attribute);
        return attribute;
    }

    /** One of the source's buffers; an indexed source must have its positions. */
    private bufferOf(buffers: BvhTraceBufferSource['buffers'], key: BufferKey): GPUBuffer {
        const buffer = buffers[key];
        if (!buffer) throw new Error(`@pmndrs/bvh/three: an indexed trace source needs its ${key} buffer`);
        return buffer;
    }

    /** Lets three forget an attribute: the destroy guard swaps a throwaway in for our buffer, which three then destroys. */
    private retire(attribute: StorageBufferAttribute): void {
        attribute.dispose();
    }
}

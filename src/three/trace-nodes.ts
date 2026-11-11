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
 *
 * **Compile cost.** The walk is emitted once per shader as plain WGSL
 * functions (one loop, no unrolling), and every TSL helper is one small
 * `wgslFn`; the names are fixed, so the shader text is the same on every load
 * (three's and Chrome's shader caches hit).
 *
 * This file reaches into three's WebGPU backend (`backend.get(attribute).buffer`),
 * which is not public API: it is written against three 0.187.
 */

import { CodeNode, StorageBufferAttribute, StructTypeNode, type Node, type NodeBuilder } from 'three/webgpu';
import { float, storage, wgslFn } from 'three/tsl';
import { traceModuleWgsl } from '../trace/wgsl.ts';

/** Where the trace buffers come from: a `SceneTrees` (buffers and version), or fixed buffers (`uploadTraceScene`). */
export interface BvhTraceBufferSource {
    readonly buffers: { readonly nodes: GPUBuffer; readonly triangles: GPUBuffer; readonly topLevel: GPUBuffer };
    /** Changes whenever a buffer is replaced. Fixed buffers may omit it. */
    readonly version?: number;
}

export interface BvhTraceNodesOptions {
    /** The WGSL `fn bvhTrianglePasses(candidate: BvhCandidate) -> bool`; default: every triangle blocks (see `traceModuleWgsl`). */
    readonly triangleFilter?: string;
    /** As `traceModuleWgsl`'s `rayMasks`. Default true: the `*Masked` helpers exist. */
    readonly rayMasks?: boolean;
}

/** The parts of three's `WebGPURenderer` this file uses. */
interface RendererLike {
    readonly backend: { readonly isWebGPUBackend?: boolean; readonly device: GPUDevice | null; get(object: object): { buffer?: GPUBuffer } };
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

const BUFFERS = [
    { name: 'bvhTraceNodes', key: 'nodes', type: 'vec4', array: () => new Float32Array(4) },
    { name: 'bvhTraceTriangles', key: 'triangles', type: 'vec4', array: () => new Float32Array(4) },
    { name: 'bvhTraceTopLevel', key: 'topLevel', type: 'uvec4', array: () => new Uint32Array(4) },
] as const;

/** The walk's text without its three binding declarations: three declares the storage nodes instead. */
function moduleWithoutBindings(options: BvhTraceNodesOptions): string {
    const text = traceModuleWgsl({ triangleFilter: options.triangleFilter, rayMasks: options.rayMasks });
    let removed = 0;
    const stripped = text.replace(/^@group\(\d+\) @binding\(\d+\) var<storage, read> bvhTrace(?:Nodes|Triangles|TopLevel): array<vec4[fu]>;$/gm, () => {
        removed++;
        return '';
    });
    if (removed !== 3) throw new Error(`@pmndrs/bvh/three: expected the walk's three bindings, found ${removed}`);
    return stripped;
}

/** The walk as a three code node: its buffer names become the storage nodes' references in each shader. */
class BvhTraceModuleNode extends CodeNode {
    // Node-valued properties: three sets them up and analyzes them as this node's children.
    bvhTraceNodes: StorageNode;
    bvhTraceTriangles: StorageNode;
    bvhTraceTopLevel: StorageNode;

    constructor(
        private readonly text: string,
        storages: Readonly<Record<'bvhTraceNodes' | 'bvhTraceTriangles' | 'bvhTraceTopLevel', StorageNode>>,
    ) {
        super('', [], 'wgsl');
        this.bvhTraceNodes = storages.bvhTraceNodes;
        this.bvhTraceTriangles = storages.bvhTraceTriangles;
        this.bvhTraceTopLevel = storages.bvhTraceTopLevel;
    }

    generate(builder: NodeBuilder): string {
        const names: Record<string, string> = {
            bvhTraceNodes: this.bvhTraceNodes.build(builder, 'property') as string,
            bvhTraceTriangles: this.bvhTraceTriangles.build(builder, 'property') as string,
            bvhTraceTopLevel: this.bvhTraceTopLevel.build(builder, 'property') as string,
        };
        const code = this.text.replace(/\bbvhTrace(?:Nodes|Triangles|TopLevel)\b/g, (name) => names[name]);
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
    /** The storage nodes, for reading the layout directly. */
    readonly storage: { readonly nodes: StorageNode; readonly triangles: StorageNode; readonly topLevel: StorageNode };
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

    constructor(
        renderer: unknown,
        readonly source: BvhTraceBufferSource,
        options: BvhTraceNodesOptions = {},
    ) {
        this.renderer = renderer as RendererLike;
        if (!this.renderer.backend?.isWebGPUBackend || !this.renderer.backend.device) {
            throw new Error('@pmndrs/bvh/three: BvhTraceNodes needs an initialized WebGPURenderer on its WebGPU backend (await renderer.init())');
        }
        this.rayMasks = options.rayMasks ?? true;
        this.bound = source.buffers;
        this.boundVersion = source.version;
        const storages = {} as Record<(typeof BUFFERS)[number]['name'], StorageNode>;
        for (const { name, key, type, array } of BUFFERS) {
            const node = storage(this.standIn(this.bound[key], array()), type, 0).toReadOnly() as unknown as StorageNode;
            node.setName(name);
            storages[name] = node;
        }
        this.storage = { nodes: storages.bvhTraceNodes, triangles: storages.bvhTraceTriangles, topLevel: storages.bvhTraceTopLevel };
        this.module = new BvhTraceModuleNode(moduleWithoutBindings(options), storages);
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
     */
    update(): boolean {
        if (this.disposed) throw new Error('@pmndrs/bvh/three: this BvhTraceNodes was disposed');
        const buffers = this.source.buffers;
        const changed =
            this.source.version !== this.boundVersion || buffers.nodes !== this.bound.nodes || buffers.triangles !== this.bound.triangles || buffers.topLevel !== this.bound.topLevel;
        if (!changed) return false;
        for (const { key, array } of BUFFERS) {
            const node = this.storage[key];
            const old = node.value as StorageBufferAttribute;
            node.value = this.standIn(buffers[key], array());
            this.retire(old);
        }
        this.bound = buffers;
        this.boundVersion = this.source.version;
        return true;
    }

    /** Releases the stand-in attributes. The source's buffers are not destroyed (they are the source's). */
    dispose(): void {
        if (this.disposed) return;
        for (const { key } of BUFFERS) this.retire(this.storage[key].value as StorageBufferAttribute);
        this.disposed = true;
    }

    /** A `StorageBufferAttribute` whose GPU buffer is `buffer`: three takes the backend's buffer as made and never creates one. */
    private standIn(buffer: GPUBuffer, array: Float32Array | Uint32Array): StorageBufferAttribute {
        const attribute = new StorageBufferAttribute(array, 4);
        attribute.name = buffer.label;
        this.renderer.backend.get(attribute).buffer = buffer;
        return attribute;
    }

    /**
     * Lets three forget an attribute without destroying our buffer: its
     * backend buffer becomes a small throwaway, which three destroys on dispose.
     */
    private retire(attribute: StorageBufferAttribute): void {
        const device = this.renderer.backend.device!;
        this.renderer.backend.get(attribute).buffer = device.createBuffer({ label: '@pmndrs/bvh/three retired stand-in', size: 16, usage: 0x80 /* STORAGE */ });
        attribute.dispose();
    }
}

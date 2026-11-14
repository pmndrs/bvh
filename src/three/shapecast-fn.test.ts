/**
 * BvhTraceNodes.getShapecastFn (PARITY 8.6) in Node: three's own WGSL node
 * builder (r187, `WebGPUBackend.createNodeBuilder`) compiles a TSL compute
 * kernel that calls the cast, and the shader text has the caller's hooks as
 * functions of their own (a `wgslFn`, a TSL `Fn` with a layout), the
 * adapters calling them by the names three gave them with the shape first
 * and only the triangle arguments they declare, the walk stamped for them,
 * and the entry point. Wrong hooks are refused with clear errors. The cast
 * itself runs on a GPU: tests/gpu/pages/three-shapecast.
 */

import { Fn, If, code, context, float, instanceIndex, select, storage, uint, uvec4, vec4, wgslFn } from 'three/tsl';
import { StorageBufferAttribute, StructTypeNode, WebGPUBackend } from 'three/webgpu';
import { describe, expect, it } from 'vitest';
import { installWebGpuGlobals } from '../../tests/support/fake-gpu.ts';
import { BvhTraceNodes, type BvhTslNode } from './trace-nodes.ts';

installWebGpuGlobals();

const buffer = (label: string) => ({ label, destroy() {} }) as unknown as GPUBuffer;
const source = { nodeFormat: 'f32', triangleLeaves: 'records', positionLayout: 'vec4', buffers: { nodes: buffer('n'), triangles: buffer('t'), topLevel: buffer('l') } } as const;

/** Enough of a WebGPURenderer for BvhTraceNodes and three's WGSL node builder (no device work happens). */
function stubRenderer() {
    const backend = { isWebGPUBackend: true, device: { features: new Set(), limits: {} }, get: () => ({}), destroyAttribute() {}, utils: {}, compatibilityMode: false };
    return { backend, _attributes: { has: () => false }, debug: { diagnostics: { keywords: false }, onNodeBuilderCreated: null }, hasFeature: () => false, hasCompatibility: () => false, library: { fromMaterial: () => null }, contextNode: context(), getRenderTarget: () => null, getMRT: () => null, _bindings: null };
}

/** The compute shader three generates for a kernel calling `cast(args)` per invocation. */
function computeShader(renderer: unknown, call: () => BvhTslNode): string {
    const out = storage(new StorageBufferAttribute(new Uint32Array(64), 1), 'uint', 64);
    const kernel = Fn(() => {
        If(instanceIndex.lessThan(64), () => {
            out.element(instanceIndex).assign(select(call(), uint(1), uint(0)));
        });
    })().compute(64, [64]);
    const builder = (WebGPUBackend.prototype as unknown as { createNodeBuilder(object: unknown, renderer: unknown): { build(): void; computeShader: string; compute: unknown } }).createNodeBuilder(null, renderer);
    builder.compute = kernel;
    builder.build();
    return builder.computeShader;
}

/** The compute shader three generates for a kernel whose body `body` writes into `out` (64 `uvec4`s). */
function kernelShader(renderer: unknown, body: (out: BvhTslNode) => void): string {
    const out = storage(new StorageBufferAttribute(new Uint32Array(64 * 4), 4), 'uvec4', 64);
    const kernel = Fn(() => {
        If(instanceIndex.lessThan(64), () => body(out));
    })().compute(64, [64]);
    const builder = (WebGPUBackend.prototype as unknown as { createNodeBuilder(object: unknown, renderer: unknown): { build(): void; computeShader: string; compute: unknown } }).createNodeBuilder(null, renderer);
    builder.compute = kernel;
    builder.build();
    return builder.computeShader;
}

/**
 * Every module-scope name the shader declares more than once: functions,
 * structs and module variables (`var<private>`, `var<storage, ...>`,
 * `var<uniform>`, `var<workgroup>`; a function's own `var` has no address
 * space). WGSL refuses such a shader ("redeclaration of ...").
 */
function redeclared(shader: string): string[] {
    const names = [...shader.matchAll(/^\s*(?:fn|struct|var<[^>]+>)\s+([A-Za-z_][A-Za-z0-9_]*)/gm)].map((match) => match[1]);
    return [...new Set(names.filter((name, i) => names.indexOf(name) !== i))];
}

const occurrences = (text: string, pattern: RegExp) => (text.match(new RegExp(pattern.source, 'g')) ?? []).length;

describe('BvhTraceNodes.getShapecastFn: one declaration per name', () => {
    it('declares every hook, adapter, walk and variable once: two casts sharing a hook, private state, the closest, any and closest-point nodes', () => {
        const renderer = stubRenderer();
        const bvh = new BvhTraceNodes(renderer, source, { queries: true });
        // The three-shapecast page's kernel: a TSL box test shared by two casts, wgslFn triangle tests, a counter in a code node.
        const sphereBounds = Fn(([shape, boxMin, boxMax]: BvhTslNode[]) => {
            const outside = boxMin.sub(shape.xyz).max(shape.xyz.sub(boxMax)).max(0);
            const distanceSq = outside.dot(outside);
            return select(distanceSq.lessThanEqual(shape.w.mul(shape.w)), distanceSq, float(-1));
        }).setLayout({ name: 'pageSphereBounds', type: 'float', inputs: [{ name: 'shape', type: 'vec4' }, { name: 'boxMin', type: 'vec3' }, { name: 'boxMax', type: 'vec3' }] });
        const sphereTriangle = wgslFn(
            `fn pageSphereTriangle(shape: vec4f, a: vec3f, b: vec3f, c: vec3f) -> bool { let o = bvhClosestPointToTriangle(shape.xyz, a, b, c).point - shape.xyz; return dot(o, o) <= shape.w * shape.w; }`,
            [bvh.module as never],
        );
        const counter = code('var<private> pageCounter: u32;');
        // Hooks named the way a cast's own adapters were named (`<cast>Triangle`, `<cast>Keep`): the cast must not reuse those names.
        const countTriangle = wgslFn(`fn pageCountTriangle(shape: vec4f, a: vec3f) -> bool { if (distance(shape.xyz, a) <= shape.w) { pageCounter += 1u; } return false; }`, [counter as never]);
        const countKeep = wgslFn(`fn pageCountKeep(shape: vec4f, score: f32) -> bool { return score <= shape.w * shape.w; }`);
        const anyCast = bvh.getShapecastFn({ name: 'pageAny', shape: 'vec4', intersectsBounds: sphereBounds, intersectsTriangle: sphereTriangle });
        const countCast = bvh.getShapecastFn({ name: 'pageCount', shape: 'vec4', intersectsBounds: sphereBounds, intersectsTriangle: countTriangle, keep: countKeep });
        const readCount = wgslFn(`fn pageReadCount() -> u32 { return pageCounter; }`, [counter as never]);
        const shader = kernelShader(renderer, (out) => {
            const shape = vec4(1, 2, 3, 0.5);
            const any = select(anyCast(shape), uint(1), uint(0)).toVar();
            const masked = select(anyCast(shape, uint(0)), uint(1), uint(0)).toVar();
            const stopped = select(countCast(shape), uint(1), uint(0)).toVar();
            const counted = (readCount() as unknown as BvhTslNode).toVar();
            const hit = bvh.traceClosest(shape.xyz, shape.xyz).toVar();
            const blocked = select(bvh.traceAny(shape.xyz, shape.xyz), uint(1), uint(0)).toVar();
            const point = bvh.closestPoint(shape.xyz).toVar();
            out.element(instanceIndex).assign(uvec4(any.add(masked).add(stopped).add(blocked), counted, hit.get('instance'), point.get('triangle')));
        });
        expect(redeclared(shader)).toEqual([]);
        for (const hook of ['pageSphereBounds', 'pageSphereTriangle', 'pageCountTriangle', 'pageCountKeep', 'pageReadCount']) expect(occurrences(shader, new RegExp(`fn ${hook}\\s*\\(`)), hook).toBe(1);
        expect(occurrences(shader, /var<private> pageCounter\b/)).toBe(1);
        for (const cast of ['pageAny', 'pageCount']) expect(occurrences(shader, new RegExp(`fn ${cast}\\s*\\(`)), cast).toBe(1);
        // The module (with its own walks) is in once, though every helper and both casts include it.
        expect(occurrences(shader, /fn bvhTraceClosest\s*\(/)).toBe(1);
        expect(occurrences(shader, /fn bvhClosestPointWalk\s*\(/)).toBe(1);
    });

    it('refuses a hook that has the name of its own cast', () => {
        const renderer = stubRenderer();
        const bvh = new BvhTraceNodes(renderer, source, { queries: true });
        const cast = bvh.getShapecastFn({
            name: 'sameName',
            intersectsBounds: wgslFn(`fn otherBounds(boxMin: vec3f, boxMax: vec3f) -> f32 { return 0.0; }`),
            intersectsTriangle: wgslFn(`fn sameName(a: vec3f) -> bool { return true; }`),
        });
        expect(() => kernelShader(renderer, (out) => out.element(instanceIndex).assign(uvec4(select(cast(), uint(1), uint(0)))))).toThrow(/named sameName, as the cast/);
    });
});

describe('BvhTraceNodes.getShapecastFn', () => {
    it("compiles a wgslFn and a TSL Fn with a layout into the walk, with the shape first and only the declared triangle arguments", () => {
        const renderer = stubRenderer();
        const bvh = new BvhTraceNodes(renderer, source, { queries: true });
        const bounds = Fn(([shape, boxMin, boxMax]: BvhTslNode[]) => {
            const center = shape.xyz;
            const outside = boxMin.sub(center).max(center.sub(boxMax)).max(0);
            const distanceSq = outside.dot(outside);
            return select(distanceSq.lessThanEqual(shape.w.mul(shape.w)), distanceSq, float(-1));
        }).setLayout({ name: 'testSphereBounds', type: 'float', inputs: [{ name: 'shape', type: 'vec4' }, { name: 'boxMin', type: 'vec3' }, { name: 'boxMax', type: 'vec3' }] });
        const triangle = wgslFn(`fn testSphereTriangle(shape: vec4f, a: vec3f, b: vec3f, c: vec3f, instance: u32) -> bool { return distance(shape.xyz, a) < shape.w && instance != 7u; }`);
        const keep = wgslFn(`fn testSphereKeep(shape: vec4f, score: f32) -> bool { return score <= shape.w * shape.w; }`);
        const cast = bvh.getShapecastFn({ name: 'testSphereCast', shape: 'vec4', intersectsBounds: bounds, intersectsTriangle: triangle, keep });
        const shader = computeShader(renderer, () => cast(vec4(1, 2, 3, 0.5)));
        // The hooks, each once, as functions of their own.
        expect(occurrences(shader, /fn testSphereBounds\s*\(/)).toBe(1);
        expect(occurrences(shader, /fn testSphereTriangle\s*\(/)).toBe(1);
        expect(occurrences(shader, /fn testSphereKeep\s*\(/)).toBe(1);
        // The adapters: the shape first, then the walk's own values (only the five triangle arguments declared).
        expect(shader).toContain('var<private> testSphereCast_bvhShape: vec4f;');
        expect(shader).toMatch(/return f32\(testSphereBounds\(\s*testSphereCast_bvhShape, bounds\.min, bounds\.max\s*\)\);/);
        expect(shader).toMatch(/return testSphereKeep\(\s*testSphereCast_bvhShape, score\s*\);/);
        expect(shader).toMatch(/return testSphereTriangle\(\s*testSphereCast_bvhShape, triangle\.a, triangle\.b, triangle\.c, triangle\.instance\s*\);/);
        expect(shader).not.toMatch(/testSphereTriangle\([^)]*bvhShapecastMaterial/);
        // The walk stamped for the adapters, nestable (it re-reads its instance's frame after a leaf), and the entry with the mask.
        expect(shader).toContain('fn testSphereCast_bvhWalk(mask: u32) -> bool');
        expect(shader).toContain('let leftScore = testSphereCast_bvhBounds(left);');
        expect(shader).toContain('if (testSphereCast_bvhTriangle(candidate)) {');
        expect(shader).toContain('if (!testSphereCast_bvhKeep(scores[stackSize])) {');
        expect(shader).toMatch(/fn testSphereCast\s*\(\s*shape: vec4f, mask: u32\s*\) -> bool/);
        expect(shader).toMatch(/testSphereCast_bvhShape = shape;\s*return testSphereCast_bvhWalk\(mask\);/);
        // The walk reads the bound storage nodes, as the module does.
        expect(shader).toMatch(/var<storage, read> bvhTraceTopLevel/);
        // The call: the default mask is every instance.
        expect(shader).toMatch(/testSphereCast\(\s*vec4<f32>\(\s*1\.0, 2\.0, 3\.0, 0\.5\s*\), 4294967295u\s*\)/);
    });

    it('takes no shape and no mask, a struct shape, and every triangle argument', () => {
        const renderer = stubRenderer();
        const noMasks = new BvhTraceNodes(renderer, source, { queries: true, rayMasks: false });
        const any = noMasks.getShapecastFn({
            name: 'testEverything',
            intersectsBounds: wgslFn(`fn testAllBounds(boxMin: vec3f, boxMax: vec3f) -> f32 { return 0.0; }`),
            intersectsTriangle: wgslFn(`fn testAllTriangle(a: vec3f, b: vec3f, c: vec3f, instance: u32, triangle: u32, material: u32, normal: vec3f) -> bool { return material == 3u && normal.y > 0.0; }`),
        });
        const shader = computeShader(renderer, () => any());
        expect(shader).toMatch(/fn testEverything\s*\(\s*\) -> bool/);
        expect(shader).toMatch(/testAllTriangle\(\s*triangle\.a, triangle\.b, triangle\.c, triangle\.instance, triangle\.triangle, bvhShapecastMaterial\(triangle\), bvhShapecastNormal\(triangle\)\s*\)/);
        expect(shader).toContain('fn testEverything_bvhKeep(score: f32) -> bool {\n    return true;');
        expect(() => any(vec4(0))).toThrow(/takes nothing/);

        const capsule = new StructTypeNode({ start: 'vec3', end: 'vec3', radius: 'float' }, 'TestCapsule');
        const masked = new BvhTraceNodes(renderer, source, { queries: true });
        const capsuleCast = masked.getShapecastFn({
            name: 'testCapsuleCast',
            shape: capsule,
            intersectsBounds: wgslFn(`fn testCapsuleBounds(shape: TestCapsule, boxMin: vec3f, boxMax: vec3f) -> f32 { return shape.radius; }`),
            intersectsTriangle: wgslFn(`fn testCapsuleTriangle(shape: TestCapsule, a: vec3f) -> bool { return distance(a, shape.start) < shape.radius; }`),
        });
        const capsuleShader = computeShader(renderer, () => capsuleCast(wgslFn(`fn testMakeCapsule() -> TestCapsule { return TestCapsule(vec3f(0.0), vec3f(1.0), 0.5); }`, [capsule as never])(), uint(3)));
        expect(capsuleShader).toContain('var<private> testCapsuleCast_bvhShape: TestCapsule;');
        expect(capsuleShader).toMatch(/struct TestCapsule\s*\{/);
        expect(capsuleShader).toMatch(/testCapsuleCast\(\s*testMakeCapsule\(\s*\), 3u\s*\)/);
    });

    it('refuses wrong hooks and missing queries with clear errors', () => {
        const renderer = stubRenderer();
        const bvh = new BvhTraceNodes(renderer, source, { queries: true });
        const triangle = wgslFn(`fn testOkTriangle(a: vec3f) -> bool { return a.x > 0.0; }`);
        const noLayout = bvh.getShapecastFn({ name: 'testNoLayout', intersectsBounds: Fn(() => float(0)), intersectsTriangle: triangle });
        expect(() => computeShader(renderer, () => noLayout())).toThrow(/without setLayout/);
        const tooMany = bvh.getShapecastFn({
            name: 'testTooMany',
            intersectsBounds: wgslFn(`fn testThreeBounds(boxMin: vec3f, boxMax: vec3f, extra: f32) -> f32 { return extra; }`),
            intersectsTriangle: triangle,
        });
        expect(() => computeShader(renderer, () => tooMany())).toThrow(/intersectsBounds takes 3 arguments; expected 2/);
        expect(() => bvh.getShapecastFn({ name: 'not a name', intersectsBounds: triangle, intersectsTriangle: triangle })).toThrow(/WGSL identifier/);
        expect(() => bvh.getShapecastFn({ shape: 'vec5', intersectsBounds: triangle, intersectsTriangle: triangle })).toThrow(/shape type "vec5"/);
        expect(() => new BvhTraceNodes(renderer, source).getShapecastFn({ intersectsBounds: triangle, intersectsTriangle: triangle })).toThrow(/queries: true/);
        const shaped = bvh.getShapecastFn({ name: 'testShaped', shape: 'vec4', intersectsBounds: triangle, intersectsTriangle: triangle });
        expect(() => shaped()).toThrow(/shape first/);
    });
});

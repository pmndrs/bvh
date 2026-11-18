/*!
 * The pack below writes BVH4 and CWBVH node boxes as tinybvh's BVH4_GPU::ConvertFrom and
 * BVH8_CWBVH::ConvertFrom (tiny_bvh_base.h) do, in WGSL;
 * its licence (licenses/tinybvh.txt):
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2024-2026, Jacco Bikker / Breda University of Applied Sciences.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */


/**
 * The GPU wide refit's pack (docs/internals/design/wide-nodes.md, section 2): after a deforming
 * mesh's binary tree is refit (gpu-mesh-refit.ts runs the B1a refit on an `f32` copy of it),
 * every wide node is packed again from the boxes its {@link WideRefitMap} slots point at: the
 * node's corner (the union of its children), its exponents and its planes, rounded outward and
 * checked in `f32` as the walk decodes them. One thread per wide node; nothing depends on
 * another node, since a corner is its own children's union, so it is one dispatch.
 *
 * Child words, CWBVH's imask, meta and bases, and the triangles' order stay as the collapse
 * left them: the tree keeps its shape, as a binary refit keeps its builder's splits. The bytes
 * are what the CPU writes for the same boxes (`refitWide`, `repackWide` in wide-nodes.ts),
 * since both round the same way.
 */

import { WIDE_PLANES_WGSL } from './gpu-wide-collapse-wgsl.ts';
import { WORKGROUP_INDEX_WGSL } from './gpu-workgroup-scan.ts';
import { BVH4_NODE_WORD, BVH4_NODE_WORDS, CWBVH_NODE_WORD, CWBVH_NODE_WORDS, WIDE_REFIT_NO_SLOT, WIDE_REFIT_ROOT_SLOT, wideRefitSlots, type WideNodeFormat } from './wide-nodes.ts';

/** Workgroup size of the pack. */
export const WIDE_REFIT_WORKGROUP = 64;
/** Bytes of the pack's uniform: wide node count, the target's first node word, two spare. */
export const WIDE_REFIT_PARAMS_BYTES = 16;

const hex = (value: number): string => `0x${(value >>> 0).toString(16)}u`;

/** WGSL of the pack for one format: entry point `bvhwRefitPack`, bindings 0-3 (binary nodes, slots, wide nodes, params). */
export function wideRefitWgsl(format: WideNodeFormat): string {
    const M = wideRefitSlots(format);
    const cwbvh = format === 'cwbvh8';
    const B = BVH4_NODE_WORD;
    const C = CWBVH_NODE_WORD;
    return /* wgsl */ `
// ---- GPU wide refit pack (${format}; gpu-wide-refit.ts) ----
struct BvhwRefitParams {
    nodeCount: u32,    // wide nodes to pack
    wideWord: u32,     // the first one's word in the target
    pad0: u32,
    pad1: u32,
}

struct BvhwRefitBox {
    lo: vec3f,
    hi: vec3f,
}

@group(0) @binding(0) var<storage, read> bvhwrBinary: array<u32>;
@group(0) @binding(1) var<storage, read> bvhwrSlots: array<u32>;
@group(0) @binding(2) var<storage, read_write> bvhwrWide: array<u32>;
@group(0) @binding(3) var<uniform> bvhwrParams: BvhwRefitParams;

const M: u32 = ${M}u;
const WIDE_WORDS: u32 = ${cwbvh ? CWBVH_NODE_WORDS : BVH4_NODE_WORDS}u;
const NO_SLOT: u32 = ${hex(WIDE_REFIT_NO_SLOT)};
const ROOT_SLOT: u32 = ${hex(WIDE_REFIT_ROOT_SLOT)};
${WIDE_PLANES_WGSL}
${WORKGROUP_INDEX_WGSL}

fn bvhwrVec(word: u32) -> vec3f {
    return vec3f(bitcast<f32>(bvhwrBinary[word]), bitcast<f32>(bvhwrBinary[word + 1u]), bitcast<f32>(bvhwrBinary[word + 2u]));
}

// A slot's box: a child box of the binary nodes (min at 'word', max 4 words on), or the root's (its children's union).
fn bvhwrBoxOf(word: u32) -> BvhwRefitBox {
    if (word == ROOT_SLOT) {
        return BvhwRefitBox(min(bvhwrVec(0u), bvhwrVec(8u)), max(bvhwrVec(4u), bvhwrVec(12u)));
    }
    return BvhwRefitBox(bvhwrVec(word), bvhwrVec(word + 4u));
}

@compute @workgroup_size(${WIDE_REFIT_WORKGROUP})
fn bvhwRefitPack(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let node = flatWorkgroupIndex(workgroup, groups) * ${WIDE_REFIT_WORKGROUP}u + localIndex;
    if (node >= bvhwrParams.nodeCount) {
        return;
    }
    // The corner: the union of the children's boxes (an axis no child covers is 0), as the collapse writes it.
    var boxLo: array<vec3f, ${M}>;
    var boxHi: array<vec3f, ${M}>;
    var used = 0u;
    var lo = vec3f(BVHW_BIG);
    var hi = vec3f(-BVHW_BIG);
    for (var s = 0u; s < M; s += 1u) {
        let word = bvhwrSlots[node * M + s];
        if (word == NO_SLOT) {
            continue;
        }
        let box = bvhwrBoxOf(word);
        boxLo[s] = box.lo;
        boxHi[s] = box.hi;
        used |= 1u << s;
        if (!bvhwEmpty(box.lo, box.hi)) {
            lo = min(lo, box.lo);
            hi = max(hi, box.hi);
        }
    }
    let emptyAxis = lo > hi;
    lo = select(lo, vec3f(0.0), emptyAxis);
    hi = select(hi, vec3f(0.0), emptyAxis);
    let exponent = vec3i(bvhwExponent(lo.x, hi.x), bvhwExponent(lo.y, hi.y), bvhwExponent(lo.z, hi.z));
    let scale = vec3f(bvhwPow2(exponent.x), bvhwPow2(exponent.y), bvhwPow2(exponent.z));
    // Planes: lo x, y, z, hi x, y, z (${cwbvh ? 'two words each, a byte per slot' : 'one word each, a byte per child'}); an unused slot's are 0.
    var planes: array<u32, 12>;
    for (var w = 0u; w < 12u; w += 1u) {
        planes[w] = 0u;
    }
    for (var s = 0u; s < M; s += 1u) {
        if ((used & (1u << s)) == 0u) {
            continue;
        }
        var q = array<u32, 6>(255u, 255u, 255u, 0u, 0u, 0u);
        if (!bvhwEmpty(boxLo[s], boxHi[s])) {
            q[0] = bvhwQuantizeMin(boxLo[s].x, lo.x, scale.x);
            q[1] = bvhwQuantizeMin(boxLo[s].y, lo.y, scale.y);
            q[2] = bvhwQuantizeMin(boxLo[s].z, lo.z, scale.z);
            q[3] = bvhwQuantizeMax(boxHi[s].x, lo.x, scale.x);
            q[4] = bvhwQuantizeMax(boxHi[s].y, lo.y, scale.y);
            q[5] = bvhwQuantizeMax(boxHi[s].z, lo.z, scale.z);
        }
        for (var k = 0u; k < 6u; k += 1u) {
            ${cwbvh ? 'planes[k * 2u + (s >> 2u)] |= q[k] << (8u * (s & 3u));' : 'planes[k] |= q[k] << (8u * s);'}
        }
    }
    let out = bvhwrParams.wideWord + node * WIDE_WORDS;
    bvhwrWide[out + ${cwbvh ? C.p : B.min}u] = bitcast<u32>(lo.x);
    bvhwrWide[out + ${cwbvh ? C.p : B.min}u + 1u] = bitcast<u32>(lo.y);
    bvhwrWide[out + ${cwbvh ? C.p : B.min}u + 2u] = bitcast<u32>(lo.z);
    ${
        cwbvh
            ? `// The exponents, keeping the imask byte; the bases and meta stay.
    let imask = bvhwrWide[out + ${C.exponentsAndImask}u] & 0xff000000u;
    bvhwrWide[out + ${C.exponentsAndImask}u] = u32(exponent.x + 127) | (u32(exponent.y + 127) << 8u) | (u32(exponent.z + 127) << 16u) | imask;
    let planeWords = array<u32, 6>(${C.xMin}u, ${C.yMin}u, ${C.zMin}u, ${C.xMax}u, ${C.yMax}u, ${C.zMax}u);
    for (var k = 0u; k < 6u; k += 1u) {
        bvhwrWide[out + planeWords[k]] = planes[k * 2u];
        bvhwrWide[out + planeWords[k] + 1u] = planes[k * 2u + 1u];
    }`
            : `bvhwrWide[out + ${B.scale}u] = bitcast<u32>(scale.x);
    bvhwrWide[out + ${B.scale}u + 1u] = bitcast<u32>(scale.y);
    bvhwrWide[out + ${B.scale}u + 2u] = bitcast<u32>(scale.z);
    bvhwrWide[out + ${B.xMin}u] = planes[0];
    bvhwrWide[out + ${B.yMin}u] = planes[1];
    bvhwrWide[out + ${B.zMin}u] = planes[2];
    bvhwrWide[out + ${B.xMax}u] = planes[3];
    bvhwrWide[out + ${B.yMax}u] = planes[4];
    bvhwrWide[out + ${B.zMax}u] = planes[5];
    // The child words stay.`
    }
}
`;
}

/** The pack's pipeline for one format, compiled once per device and format and shared by every mesh refit. */
export class GpuWideRefitKernels {
    readonly format: WideNodeFormat;
    private readonly device: GPUDevice;
    private readonly label: string;
    private compiled: GPUComputePipeline | null = null;
    private compiling: Promise<void> | null = null;

    constructor(device: GPUDevice, format: WideNodeFormat, label = 'wide refit') {
        this.device = device;
        this.format = format;
        this.label = label;
    }

    prepare(): Promise<void> {
        this.compiling ??= (async () => {
            const module = this.device.createShaderModule({ label: `${this.label} ${this.format}`, code: wideRefitWgsl(this.format) });
            this.compiled = await this.device.createComputePipelineAsync({ label: `${this.label} ${this.format} pack`, layout: 'auto', compute: { module, entryPoint: 'bvhwRefitPack' } });
        })();
        return this.compiling;
    }

    /** The pipeline, or null before `prepare` has finished. */
    get pipeline(): GPUComputePipeline | null {
        return this.compiled;
    }
}

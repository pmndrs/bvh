/**
 * How this GPU rounds the WGSL f32 arithmetic the walk is built from, op by op
 * (pmndrs/bvh#1, item 1). 2^18 random inputs (magnitudes 2^-8 to 2^7, both
 * signs) go through one compute pass; each result is compared bit for bit with
 * CPU models: correctly rounded, fused multiply-add or not (and which product
 * a fused form takes), division as a multiply by the reciprocal. A model at
 * 100% is what this GPU does. Reports only, never fails on a model.
 *
 * Why it matters: a CPU port of the walk can match a GPU bit for bit only if
 * the GPU rounds every step as IEEE f32 does or as a model the port can copy.
 * WGSL allows otherwise: division and `1.0 / x` within 2.5 ulp, `sqrt` and
 * `inverseSqrt` within a few ulp, and an implementation may fuse. The walk's
 * ray setup divides (the shear factors, `bvhSafeInverse`), so where division
 * is not correctly rounded no CPU port can be bit-exact. The `bvhEdgeFunction`
 * corner order exists because Metal fuses `a * b - c * d` (DIVERGENCES T1);
 * this page shows whether a GPU does.
 *
 * RTX 3060 Laptop, D3D12 (2026-11-13): written products and sums are rounded
 * one by one (not fused); `dot()` is `fma(z, fma(y, x * x))`; `1.0 / x` is
 * correctly rounded 86.9% of the time, otherwise 1 ulp off; `a / b` is about
 * `a * (1 / b)` with that reciprocal; `sqrt` and `inverseSqrt` are 1 ulp off at times.
 */
import { createRandom } from '../../../../src/trace/oracle-fixtures.ts';
import { runGpuTest } from '../../client.ts';

/** Inputs per op. */
const N = 1 << 18;
/** Results per input. */
const OUT = 12;
const wgsl = /* wgsl */ `
@group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> outputs: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= ${N}u) { return; }
    let p = inputs[i * 2u];
    let q = inputs[i * 2u + 1u];
    let o = i * ${OUT}u;
    outputs[o + 0u] = 1.0 / p.x;
    outputs[o + 1u] = p.y / p.x;
    outputs[o + 2u] = p.x * p.y + p.z;
    outputs[o + 3u] = dot(p.xyz, q.xyz);
    outputs[o + 4u] = p.x * p.y - p.z * p.w;
    outputs[o + 5u] = sqrt(abs(p.x));
    outputs[o + 6u] = inverseSqrt(abs(p.x));
    outputs[o + 7u] = length(p.xyz);
    let n = normalize(q.xyz);
    outputs[o + 8u] = n.x;
    let v = q.xyz / length(q.xyz);
    outputs[o + 9u] = v.x;
    outputs[o + 10u] = (p.x * p.y + p.z * p.w) + q.x * q.y;
    outputs[o + 11u] = p.y * (1.0 / p.x);
}`;

const view = new DataView(new ArrayBuffer(4));
const bits = (x: number) => (view.setFloat32(0, x), view.getUint32(0));
const f = Math.fround;
const fma = (a: number, b: number, c: number) => f(a * b + c);

await runGpuTest(async ({ device, log }) => {
    const random = createRandom(99);
    const input = new Float32Array(N * 8);
    for (let i = 0; i < input.length; i++) input[i] = (random() * 2 - 1) * Math.pow(2, Math.floor(random() * 16) - 8);
    const inBuffer = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(inBuffer, 0, input);
    const outBytes = N * OUT * 4;
    const outBuffer = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code: wgsl }), entryPoint: 'main' } });
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: inBuffer } }, { binding: 1, resource: { buffer: outBuffer } }] });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(N / 64);
    pass.end();
    encoder.copyBufferToBuffer(outBuffer, 0, read, 0, outBytes);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();

    const models: Record<string, { slot: number; candidates: Record<string, (p: number[], q: number[]) => number> }> = {
        'rcp 1/x': { slot: 0, candidates: { exact: (p) => f(1 / p[0]) } },
        'div y/x': { slot: 1, candidates: { exact: (p) => f(p[1] / p[0]), 'y*rcp(x) exact rcp': (p) => f(p[1] * f(1 / p[0])) } },
        'x*y+z': { slot: 2, candidates: { separate: (p) => f(f(p[0] * p[1]) + p[2]), fused: (p) => fma(p[0], p[1], p[2]) } },
        dot3: { slot: 3, candidates: {
            separate: (p, q) => f(f(f(p[0] * q[0]) + f(p[1] * q[1])) + f(p[2] * q[2])),
            'fma(z, fma(y, x*x))': (p, q) => fma(p[2], q[2], fma(p[1], q[1], f(p[0] * q[0]))),
            'fma(x, fma(y, z*z))': (p, q) => fma(p[0], q[0], fma(p[1], q[1], f(p[2] * q[2]))),
        } },
        'x*y - z*w': { slot: 4, candidates: {
            separate: (p) => f(f(p[0] * p[1]) - f(p[2] * p[3])),
            'fma(x,y,-(z*w))': (p) => fma(p[0], p[1], -f(p[2] * p[3])),
            'fma(-z,w,x*y)': (p) => fma(-p[2], p[3], f(p[0] * p[1])),
        } },
        sqrt: { slot: 5, candidates: { exact: (p) => f(Math.sqrt(Math.abs(p[0]))), 'x*rsqrt exact': (p) => f(Math.abs(p[0]) * f(1 / Math.sqrt(Math.abs(p[0])))) } },
        inverseSqrt: { slot: 6, candidates: { exact: (p) => f(1 / Math.sqrt(Math.abs(p[0]))) } },
        'length(xyz)': { slot: 7, candidates: {
            'sqrt(separate dot)': (p) => f(Math.sqrt(f(f(f(p[0] * p[0]) + f(p[1] * p[1])) + f(p[2] * p[2])))),
            'sqrt(fma chain)': (p) => f(Math.sqrt(fma(p[2], p[2], fma(p[1], p[1], f(p[0] * p[0]))))),
        } },
        'normalize(q).x': { slot: 8, candidates: {
            'x*rsqrt(chain)': (_, q) => f(q[0] * f(1 / Math.sqrt(fma(q[2], q[2], fma(q[1], q[1], f(q[0] * q[0])))))),
            'x/sqrt(chain)': (_, q) => f(q[0] / f(Math.sqrt(fma(q[2], q[2], fma(q[1], q[1], f(q[0] * q[0])))))),
        } },
        'q/length(q) .x': { slot: 9, candidates: {
            'x/len(chain)': (_, q) => f(q[0] / f(Math.sqrt(fma(q[2], q[2], fma(q[1], q[1], f(q[0] * q[0])))))),
            'x*rcp(len(chain))': (_, q) => f(q[0] * f(1 / f(Math.sqrt(fma(q[2], q[2], fma(q[1], q[1], f(q[0] * q[0]))))))),
        } },
        '(xy+zw)+qx*qy': { slot: 10, candidates: {
            separate: (p, q) => f(f(f(p[0] * p[1]) + f(p[2] * p[3])) + f(q[0] * q[1])),
            'fma(qx,qy,fma(x,y,z*w))': (p, q) => fma(q[0], q[1], fma(p[0], p[1], f(p[2] * p[3]))),
            'fma(qx,qy,fma(z,w,x*y))': (p, q) => fma(q[0], q[1], fma(p[2], p[3], f(p[0] * p[1]))),
        } },
        'y*(1/x)': { slot: 11, candidates: { 'exact rcp': (p) => f(p[1] * f(1 / p[0])) } },
    };
    const report: Record<string, Record<string, string>> = {};
    for (const [name, { slot, candidates }] of Object.entries(models)) {
        report[name] = {};
        for (const [label, fn] of Object.entries(candidates)) {
            let same = 0, maxUlp = 0, counted = 0;
            const ulpHist = [0, 0, 0, 0];
            for (let i = 0; i < N; i++) {
                const p = Array.from(input.subarray(i * 8, i * 8 + 4)), q = Array.from(input.subarray(i * 8 + 4, i * 8 + 8));
                const gpu = out[i * OUT + slot], cpu = fn(p, q);
                if (!Number.isFinite(gpu) || !Number.isFinite(cpu)) continue;
                counted++;
                if (bits(gpu) === bits(cpu)) { same++; continue; }
                const d = Math.abs(bits(gpu) - bits(cpu));
                maxUlp = Math.max(maxUlp, d);
                ulpHist[Math.min(d, 3)]++;
            }
            report[name][label] = `${(100 * same / counted).toFixed(3)}% exact, max ${maxUlp} ulp, 1ulp ${ulpHist[1]} 2ulp ${ulpHist[2]} 3+ ${ulpHist[3]} (of ${counted})`;
            log(`${name} | ${label}: ${report[name][label]}`);
        }
    }
    return { report };
});

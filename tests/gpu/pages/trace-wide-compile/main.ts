/**
 * Wide nodes (docs/internals/design/wide-nodes.md): every wide batch kernel compiles, and each
 * stays under the WGSL size budget (Windows' DXC compile time grows with it). The f32 and
 * quantized kernels are compiled too, as the baseline for the sizes and compile times
 * (the Windows runbook compares them). Pipelines are created one at a time so each time is its own.
 */

import { batchKernelWgsl } from '../../../../src/trace/wgsl.ts';
import { assert, runGpuTest } from '../../client.ts';

/** Generated kernels larger than this fail the page (the experiment's budget for DXC). */
const MAX_KERNEL_BYTES = 50 * 1024;

type Format = 'f32' | 'quantized' | 'bvh4' | 'cwbvh8';

interface Kernel {
    name: string;
    mode: 'closest' | 'any';
    nodeFormat: Format;
    wideStack?: 'private' | 'workgroup';
    wideWalk?: 'nested' | 'single';
    wideDecode?: 'exact' | 'folded';
    rayMasks: boolean;
    triangleLeaves: 'records' | 'indexed';
    hitNormal: boolean;
    anyHitOrder?: 'far' | 'near';
}

function kernels(): Kernel[] {
    const list: Kernel[] = [];
    const formats: { nodeFormat: Format; wideStack?: 'private' | 'workgroup'; wideWalk?: 'nested' | 'single'; wideDecode?: 'exact' | 'folded' }[] = [
        { nodeFormat: 'f32' },
        { nodeFormat: 'quantized' },
        { nodeFormat: 'bvh4' },
        { nodeFormat: 'bvh4', wideWalk: 'single' },
        { nodeFormat: 'bvh4', wideDecode: 'folded' },
        { nodeFormat: 'cwbvh8', wideStack: 'private', wideDecode: 'folded' },
        { nodeFormat: 'cwbvh8', wideStack: 'private' },
        { nodeFormat: 'cwbvh8', wideStack: 'workgroup' },
    ];
    for (const format of formats) {
        for (const mode of ['closest', 'any'] as const) {
            for (const rayMasks of [true, false]) {
                for (const triangleLeaves of ['records', 'indexed'] as const) {
                    for (const hitNormal of mode === 'closest' ? [true, false] : [true]) {
                        const name = `${format.nodeFormat}${format.wideStack ? `-${format.wideStack}` : ''}${format.wideWalk ? `-${format.wideWalk}` : ''}${format.wideDecode ? `-${format.wideDecode}` : ''} ${mode}${rayMasks ? '' : ' nomasks'} ${triangleLeaves}${hitNormal ? '' : ' compact'}`;
                        list.push({ name, mode, ...format, rayMasks, triangleLeaves, hitNormal });
                    }
                }
            }
            if (mode === 'any' && (format.nodeFormat === 'bvh4' || format.nodeFormat === 'cwbvh8')) {
                list.push({ name: `${format.nodeFormat}${format.wideStack ? `-${format.wideStack}` : ''}${format.wideWalk ? `-${format.wideWalk}` : ''}${format.wideDecode ? `-${format.wideDecode}` : ''} any near`, mode, ...format, rayMasks: true, triangleLeaves: 'records', hitNormal: true, anyHitOrder: 'near' });
            }
        }
    }
    return list;
}

await runGpuTest(async ({ device, log }) => {
    const failures: string[] = [];
    const results: { name: string; bytes: number; compileMs: number }[] = [];
    for (const kernel of kernels()) {
        const code = batchKernelWgsl(kernel.mode, {
            rayMasks: kernel.rayMasks,
            hitNormal: kernel.hitNormal,
            triangleLeaves: kernel.triangleLeaves,
            nodeFormat: kernel.nodeFormat,
            wideStack: kernel.wideStack,
            wideWalk: kernel.wideWalk,
            wideDecode: kernel.wideDecode,
            anyHitOrder: kernel.anyHitOrder,
        });
        const bytes = new TextEncoder().encode(code).length;
        const module = device.createShaderModule({ label: kernel.name, code });
        const info = await module.getCompilationInfo();
        const errors = info.messages.filter((message) => message.type === 'error');
        if (errors.length > 0) {
            failures.push(`${kernel.name}: ${errors.map((message) => `${message.lineNum}:${message.linePos} ${message.message}`).join('; ')}`);
            continue;
        }
        const started = performance.now();
        try {
            await device.createComputePipelineAsync({ label: kernel.name, layout: 'auto', compute: { module, entryPoint: 'bvhBatch' } });
        } catch (error) {
            failures.push(`${kernel.name}: pipeline: ${error instanceof Error ? error.message : String(error)}`);
            continue;
        }
        const compileMs = performance.now() - started;
        results.push({ name: kernel.name, bytes, compileMs });
        if (bytes > MAX_KERNEL_BYTES) failures.push(`${kernel.name}: ${bytes} bytes of WGSL, over the ${MAX_KERNEL_BYTES} budget`);
        log(`${kernel.name}: ${(bytes / 1024).toFixed(1)} KiB, pipeline ${compileMs.toFixed(0)} ms`);
    }
    assert(failures.length === 0, failures.slice(0, 8).join('\n'));
    return { kernels: results, maxBytes: Math.max(...results.map((result) => result.bytes)) };
});

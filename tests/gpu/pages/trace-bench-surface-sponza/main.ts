/**
 * Paired surface-ray bench on Sponza: rays that start on surfaces, through the instanced tree and
 * the merged tree (mergeStaticInstances), timed in turn. Add ?profile=quick for a smoke run.
 * The verdict is correctness only; the ratio is the result (see ../trace-bench/surface-rays-bench.ts).
 */
import { runSurfaceRaysPage } from '../trace-bench/surface-rays-bench.ts';

await runSurfaceRaysPage('sponza');

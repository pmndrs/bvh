/**
 * The head of the walks' loops (pmndrs/bvh#3): a `for` loop with a literal
 * bound in place of `loop {`.
 *
 * Tint's `PreventInfiniteLoops` (run on every backend for robustness) gives
 * every loop it can't prove finite a 64-bit iteration counter. It leaves a loop
 * alone only in this one shape: the index declared in the `for` init, tested
 * against a literal constant first, and stepped by `++`. Inlined into a large
 * kernel, the counters of the walk's descent and pop loops made DXC's Shader
 * Model 6.6+ code 3.7x slower on an RTX 3060 (Chrome D3D12; Vulkan and Metal
 * never paid it). The bound, 2^32 - 1 steps, is never reached by a walk that
 * ends, so the paths are the same; a function that ended in its loop then needs
 * a final `return` that is never taken.
 *
 * `counter` names the loop's index; nested loops need different names.
 */
export function boundedLoopWgsl(counter: string): string {
    return `for (var ${counter} = 0u; ${counter} < 0xffffffffu; ${counter}++) {`;
}

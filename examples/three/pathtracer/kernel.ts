/**
 * The small path tracer of three/pathtracer as a kernel builder, so the same
 * sampling runs over any walk: our `BvhTraceNodes` (three/pathtracer,
 * compare/pathtracer) or three-mesh-bvh's `BVHComputeData` functions
 * (compare/pathtracer). Demo code, not the package's: sampling, random numbers
 * and shading live above the hit (HANDOFF.md, "The package ends at the hit").
 *
 * One invocation per pixel, one sample per dispatch:
 * - a camera ray jittered in the pixel, then up to `bounces` diffuse bounces
 *   (a `Loop` with a uniform bound, so the walk is in the shader once);
 * - at each hit, one any-hit ray toward the sun for a hard shadow (unless
 *   `shadows: false`, which compiles it out: the closest-hit rays stay the
 *   same, since the shadow ray draws no random number);
 * - the sample is added to an accumulation buffer, its mean tone-mapped into a
 *   storage texture, and the rays traced counted with one atomic add per pixel.
 */

import { Break, Fn, If, Loop, abs, atomicAdd, cos, cross, dot, exp, float, instanceIndex, max, mix, normalize, pow, select, sin, sqrt, storage, textureStore, uint, uniform, uvec2, vec3, vec4 } from 'three/tsl';
import { Matrix4, Vector3 } from 'three';
import { StorageBufferAttribute, type ComputeNode, type StorageTexture } from 'three/webgpu';

// TSL nodes (three's node types are loose; the package types them as any too).
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TslNode = any;

/** What the path tracer needs from a walk, as TSL nodes (called inside the kernel's `Fn`). */
export interface PathTracerWalk {
    /** The closest hit in `[tMin, tMax]`: `t` (negative on a miss), the geometric normal (world space, unit), the surface's albedo. */
    closest(origin: TslNode, direction: TslNode, tMin: number, tMax: number): { t: TslNode; normal: TslNode; albedo: TslNode };
    /** Whether anything blocks the ray in `[tMin, tMax]` (a bool node). */
    any(origin: TslNode, direction: TslNode, tMin: number, tMax: number): TslNode;
}

/** The uniforms one path tracer reads; share them between kernels to trace the same frame. */
export function pathTracerUniforms() {
    return {
        cameraWorld: uniform(new Matrix4()),
        projectionInverse: uniform(new Matrix4()),
        cameraPosition: uniform(new Vector3()),
        frameIndex: uniform(0, 'uint'),
        bounces: uniform(2, 'uint'),
        sunDirection: uniform(new Vector3(0.45, 0.8, 0.35).normalize()),
        width: uniform(1, 'uint'),
        height: uniform(1, 'uint'),
    };
}
export type PathTracerUniforms = ReturnType<typeof pathTracerUniforms>;

/** The buffers one kernel writes: accumulation (vec4 per pixel), the image, the ray counter, and optionally each pixel's primary-hit `t`. */
export interface PathTracerTargets {
    readonly accum: StorageBufferAttribute;
    readonly target: StorageTexture;
    readonly rayCounter: StorageBufferAttribute;
    readonly primary?: StorageBufferAttribute;
}

export function pathTracerTargets(width: number, height: number, target: StorageTexture, primary = false): PathTracerTargets {
    return {
        accum: new StorageBufferAttribute(new Float32Array(width * height * 4), 4),
        target,
        rayCounter: new StorageBufferAttribute(new Uint32Array(4), 1),
        primary: primary ? new StorageBufferAttribute(new Float32Array(width * height), 1) : undefined,
    };
}

/** The compute node: `width * height` invocations. */
export function pathTracerKernel(walk: PathTracerWalk, u: PathTracerUniforms, targets: PathTracerTargets, width: number, height: number, options: { readonly shadows?: boolean } = {}): ComputeNode {
    const shadows = options.shadows ?? true;
    const accumNode = storage(targets.accum, 'vec4', width * height);
    const rayCounterNode = storage(targets.rayCounter, 'uint', 4).toAtomic();
    const primaryNode = targets.primary ? storage(targets.primary, 'float', width * height) : null;
    return Fn(() => {
        const pixel = instanceIndex;
        const x = pixel.mod(u.width);
        const y = pixel.div(u.width);
        // PCG random numbers (the state is one u32 per invocation).
        const state = pixel.mul(uint(9781)).add(u.frameIndex.mul(uint(6271))).add(uint(1)).toVar();
        const random = () => {
            state.assign(state.mul(uint(747796405)).add(uint(2891336453)));
            const word = state.shiftRight(state.shiftRight(uint(28)).add(uint(4))).bitXor(state).mul(uint(277803737)).toVar();
            return float(word.shiftRight(uint(22)).bitXor(word)).div(4294967295.0);
        };
        // The camera ray, jittered within the pixel.
        const ndcX = float(x).add(random()).div(float(u.width)).mul(2).sub(1);
        const ndcY = float(1).sub(float(y).add(random()).div(float(u.height)).mul(2));
        const view = u.projectionInverse.mul(vec4(ndcX, ndcY, 1, 1));
        const far = u.cameraWorld.mul(vec4(view.xyz.div(view.w), 1)).xyz;
        const origin = vec3(u.cameraPosition).toVar();
        const direction = normalize(far.sub(u.cameraPosition)).toVar();
        const throughput = vec3(1).toVar();
        const radiance = vec3(0).toVar();
        const traced = uint(0).toVar();
        const bounce = primaryNode ? uint(0).toVar() : null;

        Loop(u.bounces.add(uint(1)), () => {
            const hit = walk.closest(origin, direction, 1e-4, 1e30);
            traced.addAssign(uint(1));
            const t = hit.t;
            if (primaryNode && bounce) {
                // Each pixel's first hit, for checking two walks against each other.
                If(bounce.equal(uint(0)), () => {
                    primaryNode.element(pixel).assign(t);
                });
                bounce.addAssign(uint(1));
            }
            If(t.lessThan(0), () => {
                // Sky.
                const up = max(direction.y, 0);
                radiance.addAssign(throughput.mul(mix(vec3(0.75, 0.8, 0.9), vec3(0.35, 0.5, 0.85), pow(up, 0.6))));
                Break();
            });
            const geometric = vec3(hit.normal);
            const normal = select(dot(geometric, direction).greaterThan(0), geometric.negate(), geometric).toVar();
            const albedo = vec3(hit.albedo).toVar();
            const point = origin.add(direction.mul(t)).add(normal.mul(1e-3)).toVar();
            // Sun: one shadow ray.
            const facing = max(dot(normal, u.sunDirection), 0);
            If(facing.greaterThan(0), () => {
                if (shadows) {
                    const blocked = walk.any(point, u.sunDirection, 1e-4, 1e30);
                    traced.addAssign(uint(1));
                    radiance.addAssign(select(blocked, vec3(0), throughput.mul(albedo).mul(facing).mul(vec3(2.6, 2.4, 2.1))));
                } else radiance.addAssign(throughput.mul(albedo).mul(facing).mul(vec3(2.6, 2.4, 2.1)));
            });
            // A cosine-weighted diffuse bounce.
            const u1 = random();
            const u2 = random().mul(6.2831853);
            const r = sqrt(u1);
            const helper = select(abs(normal.x).greaterThan(0.9), vec3(0, 1, 0), vec3(1, 0, 0));
            const tangent = normalize(cross(helper, normal));
            const bitangent = cross(normal, tangent);
            direction.assign(normalize(tangent.mul(r.mul(cos(u2))).add(bitangent.mul(r.mul(sin(u2)))).add(normal.mul(sqrt(float(1).sub(u1))))));
            origin.assign(point);
            throughput.mulAssign(albedo);
        });

        const sum = accumNode.element(pixel);
        sum.addAssign(vec4(radiance, 1));
        const mean = sum.xyz.div(sum.w);
        // Exposure and gamma.
        const mapped = pow(vec3(1).sub(exp(mean.negate().mul(0.9))), vec3(1 / 2.2));
        textureStore(targets.target, uvec2(x, y), vec4(mapped, 1));
        atomicAdd(rayCounterNode.element(0), traced);
    })().compute(width * height, [64]) as unknown as ComputeNode;
}

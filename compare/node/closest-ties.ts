/// <reference types="node" />
/**
 * Why quantized nodes cost 46% on compare/gpu's furnished room from outside
 * (2026-11-17, docs/internals/WINDOWS-RUNBOOK.md section 14): closest-hit work
 * per ray under child-order tie rules, on `f32` and quantized boxes, in Node
 * with no GPU.
 *
 *   BVH_ASSETS_DIR=... BVH_MANIFESTS_DIR=... node --experimental-transform-types --no-warnings --import ./compare/node/register.mjs compare/node/closest-ties.ts furnished-room torus-knot sponza bistro-interior [--rays 8192] [--rules left,face-right]
 *
 * Per scene (one merged world-space mesh, as compare/ gives it; Bistro is
 * merged the same way), the tinybvh tree: its quantization frame (step, steps
 * across), the SAH cost of the decoded boxes over the exact ones, then the
 * nodes visited, triangles tested and the slowest of each 32 consecutive rays
 * (what a SIMD group waits for), on compare's `outside` and `inside` rays and on
 * secondary rays off the inside rays' hits (a surface-ray stand-in).
 *
 * The walk here is a model of the production closest-hit walk's order (nearer
 * child first; the slab test in `f32` as `bvhBoxEntry`, exit padded; the
 * triangle test in doubles, so `t` may differ in the last bits): it counts
 * work, it is not the oracle. The rules decide only exact ties of the two entry
 * distances:
 * - `left`: the walk's (and tinybvh's) rule, the left child;
 * - `face-right`: a tie at `tMin` (the ray starts inside both boxes) goes left,
 *   a tie further on (both boxes entered through one shared plane) goes right;
 * - `right`, `larger` (larger half-area first), `nearer-exit`, `hybrid` (inside
 *   both: the nearer exit; through a shared plane: the larger): measured, not
 *   recommended (see the runbook).
 */

import {
    buildMeshBvhWithTinyBvh,
    createRandom,
    createTinyBvhBuilder,
    dequantizeNodes,
    GPU_LAYOUT_EMPTY_CHILD,
    GPU_LAYOUT_NODE_WORD as W,
    quantizeNodes,
    randomUnitVector,
    readNodeFrame,
    treeSahCost,
} from '@pmndrs/bvh';
import { readSceneInstances } from '../../tools/scene-meshes.ts';
import { mergeParts } from '../core/geometry.ts';
import { makeRays } from '../core/rays.ts';
import { loadScene } from './scenes.ts';

type Rule = 'left' | 'face-right' | 'right' | 'larger' | 'nearer-exit' | 'hybrid';
const RULES: readonly Rule[] = ['left', 'face-right', 'right', 'larger', 'nearer-exit', 'hybrid'];

const args = process.argv.slice(2);
const flag = (name: string): string | undefined => {
    const at = args.indexOf(name);
    return at >= 0 ? args[at + 1] : undefined;
};
const rayCount = Number(flag('--rays') ?? 8192);
const rules = (flag('--rules') ?? 'left,face-right').split(',') as Rule[];
for (const rule of rules) if (!RULES.includes(rule)) throw new RangeError(`unknown rule ${rule}; one of ${RULES.join(', ')}`);
const sceneIds = args.filter((arg, index) => !arg.startsWith('--') && !args[index - 1]?.startsWith('--'));
if (sceneIds.length === 0) sceneIds.push('furnished-room');

const load = (id: string) => {
    if (id.startsWith('bistro')) {
        const { meshes, instances } = readSceneInstances(id);
        return mergeParts(id, instances.map((i) => ({ positions: meshes[i.meshIndex].positions, indices: meshes[i.meshIndex].indices, matrix: i.objectToWorld })));
    }
    return loadScene(id);
};

const tinyBvh = await createTinyBvhBuilder();
for (const id of sceneIds) {
    const geometry = load(id);
    const tree = buildMeshBvhWithTinyBvh({ positions: geometry.positions, indices: geometry.indices }, tinyBvh);
    const quantized = quantizeNodes(tree);
    const frame = readNodeFrame(new Uint32Array(quantized.nodes), 0);
    const b = tree.bounds;
    const across = [0, 1, 2].map((k) => Math.round((b[k + 3] - b[k]) / frame.step[k]));
    console.log(`\n${id}: ${geometry.triangleCount} triangles, ${tree.nodeCount} nodes; frame step ${frame.step.join(' / ')} (${across.join(' / ')} steps across); SAH quantized / f32 ${(treeSahCost(quantized) / treeSahCost(tree)).toFixed(4)}`);

    const words = new Uint32Array(tree.nodes);
    const boxes = { f32: new Float32Array(tree.nodes), quantized: new Float32Array(dequantizeNodes(quantized).nodes) };
    const corners = new Float32Array(tree.triangles);
    const entryExit = (F: Float32Array, at: number, o: number[], inv: number[], tMax: number): [number, number] => {
        let enter = 0;
        let exit = Infinity;
        for (let k = 0; k < 3; k++) {
            const a = Math.fround((F[at + k] - o[k]) * inv[k]);
            const c = Math.fround((F[at + 4 + k] - o[k]) * inv[k]);
            enter = Math.max(enter, Math.min(a, c));
            exit = Math.min(exit, Math.max(a, c));
        }
        return enter <= Math.min(exit * (1 + 2 ** -22), tMax) ? [enter, exit] : [Infinity, Infinity];
    };
    const halfArea = (F: Float32Array, at: number): number => {
        const x = F[at + 4] - F[at];
        const y = F[at + 5] - F[at + 1];
        const z = F[at + 6] - F[at + 2];
        return x * y + y * z + z * x;
    };
    const edges = (i: number) => {
        const at = i * 12;
        return {
            at,
            e1: [corners[at + 4] - corners[at], corners[at + 5] - corners[at + 1], corners[at + 6] - corners[at + 2]],
            e2: [corners[at + 8] - corners[at], corners[at + 9] - corners[at + 1], corners[at + 10] - corners[at + 2]],
        };
    };
    const hitTriangle = (i: number, o: number[], d: number[]): number => {
        const { at, e1, e2 } = edges(i);
        const p = [d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]];
        const det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
        if (det === 0) return Infinity;
        const s = [o[0] - corners[at], o[1] - corners[at + 1], o[2] - corners[at + 2]];
        const u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) / det;
        if (u < 0 || u > 1) return Infinity;
        const q = [s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0]];
        const v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) / det;
        if (v < 0 || u + v > 1) return Infinity;
        const t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) / det;
        return t > 0 ? t : Infinity;
    };
    const walk = (F: Float32Array, rule: Rule, o: number[], d: number[]) => {
        const inv = d.map((v) => 1 / v);
        let best = Infinity;
        let bestTriangle = -1;
        let nodes = 0;
        let triangles = 0;
        const stack: number[] = [];
        let index = 0;
        let count = 0;
        for (;;) {
            if (count === 0) {
                nodes++;
                const n = index * 16;
                const rightIndex = words[n + W.rightIndex];
                const [le, lx] = entryExit(F, n + W.leftMin, o, inv, best);
                const [re, rx] = rightIndex === GPU_LAYOUT_EMPTY_CHILD ? [Infinity, Infinity] : entryExit(F, n + W.rightMin, o, inv, best);
                let next: number[] | null = null;
                const left = [words[n + W.leftIndex], words[n + W.leftCount]];
                const right = [rightIndex, words[n + W.rightCount]];
                if (le < Infinity && re < Infinity) {
                    let leftFirst = le < re;
                    if (le === re) {
                        if (rule === 'left') leftFirst = true;
                        else if (rule === 'face-right') leftFirst = le === 0;
                        else if (rule === 'right') leftFirst = false;
                        else if (rule === 'larger') leftFirst = halfArea(F, n + W.leftMin) >= halfArea(F, n + W.rightMin);
                        else if (rule === 'nearer-exit') leftFirst = lx <= rx;
                        else leftFirst = le === 0 ? lx <= rx : halfArea(F, n + W.leftMin) >= halfArea(F, n + W.rightMin);
                    }
                    const later = leftFirst ? right : left;
                    stack.push(later[0], later[1]);
                    next = leftFirst ? left : right;
                } else if (le < Infinity) next = left;
                else if (re < Infinity) next = right;
                if (next) {
                    [index, count] = next;
                    continue;
                }
            } else {
                for (let i = 0; i < count; i++) {
                    triangles++;
                    const t = hitTriangle(index + i, o, d);
                    if (t < best) {
                        best = t;
                        bestTriangle = index + i;
                    }
                }
            }
            if (stack.length === 0) break;
            count = stack.pop() as number;
            index = stack.pop() as number;
        }
        return { nodes, triangles, t: best, triangle: bestTriangle };
    };

    const sets: Record<string, { o: number[]; d: number[] }[]> = {};
    for (const set of ['outside', 'inside'] as const) sets[set] = makeRays(geometry, set, rayCount, 3).map((ray) => ({ o: ray.origin.map(Math.fround), d: ray.direction.map(Math.fround) }));
    const random = createRandom(99);
    sets.secondary = [];
    for (const ray of sets.inside) {
        const hit = walk(boxes.f32, 'left', ray.o, ray.d);
        if (hit.triangle < 0) continue;
        const { e1, e2 } = edges(hit.triangle);
        let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        const length = Math.hypot(n[0], n[1], n[2]) || 1;
        n = n.map((v) => v / length);
        if (n[0] * ray.d[0] + n[1] * ray.d[1] + n[2] * ray.d[2] > 0) n = n.map((v) => -v);
        const point = ray.o.map((v, k) => v + ray.d[k] * hit.t);
        const nudge = Math.max(1e-4, ...point.map(Math.abs)) * 1e-4;
        const sample = randomUnitVector(random);
        const sign = sample[0] * n[0] + sample[1] * n[1] + sample[2] * n[2] < 0 ? -1 : 1;
        sets.secondary.push({ o: point.map((v, k) => Math.fround(v + n[k] * nudge)), d: sample.map((v) => Math.fround(v * sign)) });
    }
    for (const [set, rays] of Object.entries(sets)) {
        console.log(`  ${set} (${rays.length} rays): nodes / triangles / slowest of 32 per ray`);
        for (const [format, F] of Object.entries(boxes)) {
            const cells: string[] = [];
            for (const rule of rules) {
                let nodes = 0;
                let triangles = 0;
                let slowest = 0;
                let groupMax = 0;
                rays.forEach((ray, i) => {
                    const work = walk(F, rule, ray.o, ray.d);
                    nodes += work.nodes;
                    triangles += work.triangles;
                    groupMax = Math.max(groupMax, work.nodes);
                    if (i % 32 === 31 || i === rays.length - 1) {
                        slowest += groupMax;
                        groupMax = 0;
                    }
                });
                cells.push(`${rule} ${(nodes / rays.length).toFixed(2)} / ${(triangles / rays.length).toFixed(2)} / ${(slowest / Math.ceil(rays.length / 32)).toFixed(1)}`);
            }
            console.log(`    ${format.padEnd(9)} ${cells.join('   ')}`);
        }
    }
}

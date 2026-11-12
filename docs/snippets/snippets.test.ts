// The docs site's own checks: every code block is its type-checked sample, every
// internal link goes to a page (and heading) that exists, every WGSL name the pages
// show is one the walk declares, and the samples that need no GPU run and give answers.

import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, relative } from 'node:path';
import { BufferGeometry, DoubleSide, Float32BufferAttribute, Mesh, MeshBasicMaterial, PerspectiveCamera, Raycaster, Scene, SphereGeometry, Vector2, Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import { buildMeshBvhWithTinyBvh, createTinyBvhBuilder, deserializeLayout, IDENTITY_TRANSFORM, traceModuleWgsl, type MeshGeometry } from '@pmndrs/bvh';
import { DOCS_DIR, docPages, syncDocs } from '../../tools/docs-snippets.ts';
import * as builders from './builders.ts';
import * as cpu from './cpu-queries.ts';
import * as start from './getting-started.ts';
import * as serialization from './serialization.ts';
import * as three from './three.ts';
import * as tracing from './tracing.ts';
import * as workers from './workers.ts';

/** A unit sphere at the origin: the samples' rays come from (0, 0, 5) along -z. */
function sphere(): MeshGeometry {
    const geometry = new SphereGeometry(1, 32, 16);
    return { positions: geometry.attributes.position.array as Float32Array, indices: new Uint32Array(geometry.index!.array) };
}

/** github-slugger's rule, as rehype-slug applies it to headings. */
function slug(heading: string): string {
    return heading
        .toLowerCase()
        .replace(/`/g, '')
        .replace(/[^\p{L}\p{N}\s_-]/gu, '')
        .replace(/\s/g, '-');
}

describe('docs pages', () => {
    it('show every sample exactly as docs/snippets has it, and every sample is used', () => {
        const { problems, unused } = syncDocs(true);
        expect(problems.map((problem) => `${problem.page}:${problem.line}: ${problem.message}`)).toEqual([]);
        expect(unused).toEqual([]);
    });

    it('link only to pages and headings that exist', () => {
        const pages = new Map<string, Set<string>>();
        for (const page of docPages()) {
            const route = '/' + relative(DOCS_DIR, page).replace(/\.mdx$/, '');
            const headings = [...readFileSync(page, 'utf8').matchAll(/^#{2,4} (.+)$/gm)].map((match) => slug(match[1]));
            pages.set(route, new Set(headings));
        }
        const broken: string[] = [];
        for (const page of docPages()) {
            const text = readFileSync(page, 'utf8').replace(/```[\s\S]*?```/g, '');
            for (const [, target] of text.matchAll(/\]\((\/[^)\s]*)\)/g)) {
                const [route, anchor] = target.split('#');
                const headings = pages.get(route);
                if (!headings) broken.push(`${relative(DOCS_DIR, page)}: ${target} (no such page)`);
                else if (anchor && !headings.has(anchor)) broken.push(`${relative(DOCS_DIR, page)}: ${target} (no such heading)`);
            }
        }
        expect(broken).toEqual([]);
    });

    it('name only WGSL the walk declares', () => {
        const module = traceModuleWgsl();
        const missing: string[] = [];
        for (const page of docPages()) {
            for (const [, block] of readFileSync(page, 'utf8').matchAll(/```wgsl\n([\s\S]*?)```/g)) {
                for (const [, name] of block.matchAll(/\b(?:fn|struct)\s+(\w+)/g)) {
                    if (!new RegExp(`\\b(fn|struct)\\s+${name}\\b`).test(module)) missing.push(`${relative(DOCS_DIR, page)}: ${name}`);
                }
            }
        }
        expect(missing).toEqual([]);
    });

    it('have a title, a description and a nav position, each nav position once', () => {
        const navs = new Map<number, string>();
        for (const page of docPages()) {
            const front = /^---\n([\s\S]*?)\n---\n/.exec(readFileSync(page, 'utf8'))?.[1] ?? '';
            expect(front, page).toMatch(/^title: .+$/m);
            expect(front, page).toMatch(/^description: .+$/m);
            const nav = Number(/^nav: (\d+)$/m.exec(front)?.[1]);
            expect(Number.isInteger(nav), page).toBe(true);
            expect(navs.get(nav), `nav ${nav} of ${page}`).toBeUndefined();
            navs.set(nav, page);
            const source = /^sourcecode: (.+)$/m.exec(front)?.[1];
            if (source) expect(() => readFileSync(join(DOCS_DIR, '..', source)), `${page}: sourcecode ${source}`).not.toThrow();
        }
    });
});

describe('docs samples that need no GPU', () => {
    it('build with tinybvh and the chooser (no device), and check the tree', async () => {
        const mesh = sphere();
        expect(await builders.tinybvh(mesh.positions, mesh.indices as Uint32Array)).toBeGreaterThan(0);
        await builders.chooser(null, [mesh]);
        const tiny = await createTinyBvhBuilder();
        const { layout } = tiny.buildLayout(mesh);
        expect(builders.check(layout, mesh.positions, mesh.indices as Uint32Array)).toBeGreaterThan(0);
        builders.layoutVersion();
        builders.readNode(layout, 0);
    });

    it('run the CPU walk and the WGSL samples', async () => {
        expect(await tracing.cpuWalk(sphere())).toBe(2);
        expect(tracing.maskedWgsl()).toContain('fn bvhTraceAnyMasked');
        expect(tracing.filterWgsl()).toContain('candidate.query == BVH_QUERY_ANY');
        expect(tracing.noMasks()).not.toContain('bvhTraceAnyMasked');
    });

    it('raycast, shapecast, bvhcast and closest point on a CpuMeshBvh', async () => {
        const input = sphere();
        const tree = await cpu.create(input);
        expect(cpu.raycast(tree)).toBeGreaterThan(0);
        expect(cpu.shapecast(tree).length).toBeGreaterThan(0);
        cpu.box(tree);
        const other = await cpu.create(sphere());
        const shift = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.2, 0, 0, 1];
        expect(cpu.bvhcast(tree, other, shift)).toBeGreaterThan(0);
        expect(cpu.closestPoint(tree, other, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1])).toBeGreaterThan(0);
        const vertexCount = input.positions.length / 3;
        cpu.hitPointInfo(tree, new Float32Array(vertexCount * 3), new Float32Array(vertexCount * 2));
        cpu.helpers(tree);
        cpu.refit(tree, input);
        const scene = await cpu.scene([input], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY_TRANSFORM }]);
        expect(scene).toBeGreaterThan(0);
    });

    it('build on the worker pool (inline in Node)', async () => {
        const tiny = await createTinyBvhBuilder();
        const mesh = sphere();
        expect(buildMeshBvhWithTinyBvh(mesh, tiny).nodeCount).toBeGreaterThan(0);
        await workers.cancel([mesh]);
    });

    it('bake, load and reload trees', async () => {
        const dir = await mkdtemp(join(tmpdir(), 'bvh-docs-'));
        try {
            await serialization.bake([sphere()], (index) => join(dir, `mesh-${index}.bvh`));
            const bytes = await readFile(join(dir, 'mesh-0.bvh'));
            expect(deserializeLayout(bytes).triangleCount).toBe(sphere().indices!.length / 3);
        } finally {
            await rm(dir, { recursive: true, force: true });
        }
        expect((await serialization.cpuTree(sphere())).triangleCount).toBeGreaterThan(0);
    });

    it('raycast in three with the drop-ins', async () => {
        expect(await three.cpuRaycast()).toBeGreaterThan(0);
        const mesh = new Mesh(new SphereGeometry(1, 32, 16), new MeshBasicMaterial({ side: DoubleSide }));
        const camera = new PerspectiveCamera();
        camera.position.set(0, 0, 5);
        camera.updateMatrixWorld();
        expect(start.quickStartThreeCpu(mesh, camera, new Vector2(0, 0))).toBeGreaterThan(0);
        three.boundsTree(mesh);
        expect(three.shapecastThree(mesh.geometry.boundsTree!)).toBeGreaterThan(0);
        expect(three.meshBvh(mesh.geometry).triangleCount).toBeGreaterThan(0);

        const scene = new Scene();
        scene.add(mesh);
        await three.sceneRaycast(scene, camera, new Raycaster(new Vector3(0, 0, 5), new Vector3(0, 0, -1)));

        const geometry = new BufferGeometry();
        geometry.setAttribute('position', new Float32BufferAttribute(sphere().positions, 3));
        await three.generate(geometry);
        expect(geometry.boundsTree?.triangleCount).toBeGreaterThan(0);
    });
});

/**
 * The compare pages' shell (BX.10, compare/RESULTS.md): the examples' app and
 * overlay, the geometry every library is given, a results panel with the
 * measurement tables (the same suites the Node runner runs, compare/core/),
 * a "what else you can do" note, and CSV rows to copy.
 *
 * Scenes: `?scene=torus-knot` (default, 100k triangles), `?scene=furnished-room`,
 * `?scene=sponza` (from the assets base, see examples/README.md), or `?model=`
 * a glTF; the model's meshes are merged into one world-space mesh, as the Node
 * runner merges a scene. Smoke runs and `?quick` use a 20k-triangle knot and
 * short timings; `?full` gives a smoke run the full ones (with
 * `BVH_PAGE_QUERY='scene=sponza&full'`, the runner's JSON then carries the
 * page's CSV rows as `data.csv`).
 */

import { BufferAttribute, BufferGeometry, DoubleSide, Mesh, MeshStandardMaterial } from 'three';
import { MESH_ADAPTERS } from '../../compare/adapters/index.ts';
import type { CompareGeometry } from '../../compare/adapters/types.ts';
import { mergeParts, torusKnot } from '../../compare/core/geometry.ts';
import { cell, csvRows, CSV_HEADER } from '../../compare/core/report.ts';
import type { Measurement } from '../../compare/core/suites.ts';
import { createApp, type ExampleApp } from './app.ts';
import { frameObject, loadGltf, meshesOf } from './loader.ts';
import { isSmoke } from './smoke.ts';

const params = new URLSearchParams(location.search);

/** Short runs: smoke mode (unless `?full`), or `?quick`. */
export const quick = (isSmoke && !params.has('full')) || params.has('quick');

const SCENE_MODELS: Record<string, string[]> = {
    'furnished-room': ['downloads/furnished-room/BreakfastRoom.glb'],
    sponza: ['downloads/sponza-intel/main_sponza_1k/NewSponza_Main_1k.gltf', 'downloads/sponza-intel-curtains/pkg_a_curtains_1k/NewSponza_Curtains_1k.gltf'],
};

export const LIBRARY_NAMES: Record<string, string> = {
    ...Object.fromEntries(MESH_ADAPTERS.map((adapter) => [adapter.id, adapter.name])),
    'ours-hploc': '@pmndrs/bvh H-PLOC (CPU twin)',
    'ours-lbvh': '@pmndrs/bvh LBVH (CPU twin)',
    'bvh.js': 'bvh.js (HybridBuilder)',
    tmbvh: 'three-mesh-bvh',
};

const STYLE = `
.compare-panel { position: fixed; top: 10px; right: 10px; bottom: 10px; z-index: 10; width: min(760px, calc(100vw - 340px)); overflow: auto;
  background: rgba(10, 12, 16, 0.9); border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; padding: 10px 14px;
  font: 12px/1.45 system-ui, sans-serif; color: #cfd8e3; }
.compare-panel h2 { font: 600 14px/1.3 system-ui, sans-serif; margin: 4px 0 6px; }
.compare-panel h3 { font: 600 11px/1 system-ui, sans-serif; text-transform: uppercase; letter-spacing: .08em; color: #8592a3; margin: 14px 0 6px; }
.compare-panel table { border-collapse: collapse; width: 100%; font: 11px/1.35 ui-monospace, Menlo, monospace; }
.compare-panel th, .compare-panel td { border-bottom: 1px solid rgba(255,255,255,0.08); padding: 3px 6px; text-align: left; vertical-align: top; }
.compare-panel th { color: #8592a3; font-weight: 600; }
.compare-panel td.bad { color: #ff9b7b; }
.compare-panel .status { color: #8592a3; white-space: pre-wrap; font: 11px/1.35 ui-monospace, Menlo, monospace; max-height: 120px; overflow: auto; }
.compare-panel .else { border-left: 3px solid #ffc878; padding: 6px 10px; background: rgba(255,200,120,0.07); margin: 10px 0; }
.compare-panel .method { color: #8592a3; font-size: 11px; }
.compare-panel textarea { width: 100%; height: 90px; font: 10px/1.3 ui-monospace, Menlo, monospace; color: #cfd8e3; background: #11151b; border: 1px solid rgba(255,255,255,0.12); }
`;

export class ResultsPanel {
    readonly root = document.createElement('div');
    private readonly statusLine = document.createElement('div');
    private readonly body = document.createElement('div');
    private readonly all: Measurement[] = [];
    /** The CSV text {@link csv} wrote, for the smoke report. */
    csvText = '';
    /** The status lines, for the smoke report. */
    statusText = '';

    constructor(title: string, intro: string) {
        const style = document.createElement('style');
        style.textContent = STYLE;
        document.head.append(style);
        this.root.className = 'compare-panel';
        const heading = document.createElement('h2');
        heading.textContent = title;
        const lead = document.createElement('div');
        lead.className = 'method';
        lead.textContent = `${intro} Every answer is checked against ours; a library that disagrees is not ranked. Numbers from this machine are indicative.`;
        this.statusLine.className = 'status';
        this.root.append(heading, lead, this.statusLine, this.body);
        document.body.append(this.root);
    }

    status(line: string): void {
        this.statusLine.textContent = `${this.statusLine.textContent ? `${this.statusLine.textContent}\n` : ''}${line}`;
        this.statusLine.scrollTop = this.statusLine.scrollHeight;
    }

    /** One row per library, one column per metric; each cell its value, ratio to ours, agreement. */
    table(title: string, measurements: readonly Measurement[], metrics: readonly string[], names: Record<string, string> = LIBRARY_NAMES): void {
        for (const m of measurements) if (!this.all.includes(m)) this.all.push(m);
        const heading = document.createElement('h3');
        heading.textContent = title;
        const table = document.createElement('table');
        const head = table.createTHead().insertRow();
        for (const text of ['Library', ...metrics]) head.append(Object.assign(document.createElement('th'), { textContent: text }));
        const libraries = [...new Set(measurements.map((m) => m.library))];
        const find = (library: string, metric: string) => measurements.find((m) => m.library === library && m.metric === metric);
        for (const library of libraries) {
            if (!metrics.some((metric) => find(library, metric))) continue;
            const row = table.insertRow();
            row.insertCell().textContent = names[library] ?? library;
            for (const metric of metrics) {
                const m = find(library, metric);
                const td = row.insertCell();
                td.textContent = cell(m, find('ours', metric)).replace(/\*\*/g, '');
                if (m?.agreement?.disagree) td.className = 'bad';
                if (m) td.title = m.method;
            }
        }
        this.body.append(heading, table);
    }

    /** The plan's "what else you can do": one thing only our pipeline allows, shown live. */
    whatElse(text: string): HTMLElement {
        const box = document.createElement('div');
        box.className = 'else';
        box.innerHTML = `<b>What else you can do</b><br>`;
        box.append(text);
        this.body.append(box);
        return box;
    }

    paragraph(text: string): void {
        const p = document.createElement('div');
        p.className = 'method';
        p.textContent = text;
        this.body.append(p);
    }

    /** The CSV rows of everything shown (hover a cell for how it was measured). */
    csv(device: GPUDevice): string {
        const info = device.adapterInfo;
        const rows = csvRows({ date: new Date().toISOString().slice(0, 10), commit: 'page', device: info.description || `${info.vendor} ${info.architecture}`, browser: navigator.userAgent.match(/Chrome\/[\d.]+/)?.[0] ?? 'browser' }, this.all);
        const text = `${CSV_HEADER}\n${rows.join('\n')}\n`;
        const heading = document.createElement('h3');
        heading.textContent = 'CSV rows';
        const area = document.createElement('textarea');
        area.readOnly = true;
        area.value = text;
        this.body.append(heading, area);
        this.csvText = text;
        this.statusText = this.statusLine.textContent ?? '';
        return text;
    }

    measurements(): readonly Measurement[] {
        return this.all;
    }
}

export interface ComparePage {
    readonly app: ExampleApp;
    readonly geometry: CompareGeometry;
    /** The geometry, drawn. */
    readonly mesh: Mesh;
    readonly panel: ResultsPanel;
    readonly sceneName: string;
}

async function loadGeometry(): Promise<{ geometry: CompareGeometry; name: string }> {
    const scene = params.get('scene');
    const model = params.get('model');
    const paths = model ? [model] : scene && SCENE_MODELS[scene] ? SCENE_MODELS[scene] : null;
    if (!paths) {
        const knot = quick ? torusKnot(200, 50) : torusKnot();
        return { geometry: { ...knot, name: 'torus-knot' }, name: knot.name };
    }
    const parts = [];
    for (const path of paths) {
        const loaded = await loadGltf(path);
        loaded.root.updateMatrixWorld(true);
        for (const mesh of meshesOf(loaded.root)) {
            const position = mesh.geometry.getAttribute('position');
            const positions = new Float32Array(position.count * 3);
            for (let i = 0; i < position.count; i++) positions.set([position.getX(i), position.getY(i), position.getZ(i)], i * 3);
            parts.push({ positions, indices: mesh.geometry.index ? Array.from(mesh.geometry.index.array as ArrayLike<number>) : null, matrix: mesh.matrixWorld.elements });
        }
    }
    const name = scene ?? (model!.split('/').pop() ?? model!);
    return { geometry: mergeParts(name, parts), name };
}

export async function createComparePage(title: string, folder: string, intro: string, options: { readonly trackTimestamp?: boolean } = {}): Promise<ComparePage> {
    const app = await createApp(title, folder, options);
    const panel = new ResultsPanel(title, intro);
    const { geometry, name } = await loadGeometry();
    const display = new BufferGeometry();
    display.setAttribute('position', new BufferAttribute(geometry.positions, 3));
    display.setIndex(new BufferAttribute(geometry.indices, 1));
    display.computeVertexNormals();
    const mesh = new Mesh(display, new MeshStandardMaterial({ color: 0x8fb3ff, roughness: 0.6, side: DoubleSide }));
    app.scene.add(mesh);
    frameObject(mesh, app.camera, app.controls);
    app.overlay.set('scene', `${name}, ${geometry.triangleCount.toLocaleString('en-US')} triangles`);
    panel.status(`${name}: ${geometry.triangleCount.toLocaleString('en-US')} triangles${quick ? ' (quick run)' : ''}`);
    app.overlay.checkbox('results panel', true, (value) => (panel.root.hidden = !value));
    return { app, geometry, mesh, panel, sceneName: name };
}

/**
 * The smoke check: no disagreement between ours and the libraries that should
 * agree exactly, on the quick run's knot. On a full run a disagreement is a
 * result (it is in the tables), not a failure: it is only logged.
 */
export function expectAgreement(measurements: readonly Measurement[], libraries: readonly string[], log?: (line: string) => void): number {
    let checked = 0;
    for (const m of measurements) {
        if (!m.agreement || !libraries.includes(m.library)) continue;
        if (m.agreement.disagree > 0) {
            const message = `${m.library} ${m.metric}: ${m.agreement.disagree} answers disagree with ours (${m.agreement.examples.join('; ')})`;
            if (quick) throw new Error(message);
            log?.(message);
        }
        checked += m.agreement.checked;
    }
    if (checked === 0) throw new Error('nothing was checked against ours');
    return checked;
}

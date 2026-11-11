/**
 * The overlay every example shares (docs/design/bvh-package-split.md, "P4"):
 *
 * - which builder made the trees and why (`builtBy`, `fallbackReason`);
 * - build milliseconds: CPU wall time, and GPU timestamps where the device has them;
 * - the tree: SAH cost, depth, node and triangle counts, bytes;
 * - Mrays/s where rays are traced;
 * - a main-thread frame-time graph.
 *
 * Plus a few controls and free rows per example, a back link and a "view
 * source" link. Everything it shows is also in {@link Overlay.snapshot}, which
 * the smoke run reports.
 */

import { SOURCE_TREE } from './examples.ts';
import { onExampleError } from './smoke.ts';

export interface BuildInfo {
    /** Which builder made the trees rays see now: `'gpu'`, `'tinybvh'`, `'gpu-hploc'`, a mix ... */
    readonly builtBy: string;
    readonly fallbackReason?: string | null;
    /** CPU wall milliseconds, from the call to the tree being usable. */
    readonly cpuMs?: number | null;
    /** GPU milliseconds, or null without `timestamp-query`. */
    readonly gpuMs?: number | null;
    /** How `gpuMs` was measured: timestamp queries (default), or wall time from submit to the results on the CPU. */
    readonly gpuSource?: 'timestamps' | 'submit to readback';
}

export interface TreeInfo {
    readonly sah?: number | null;
    readonly depth?: number | null;
    readonly nodes?: number | null;
    readonly triangles?: number | null;
    readonly bytes?: number | null;
}

const STYLE = `
.bvh-overlay { position: fixed; top: 10px; left: 10px; z-index: 10; width: 300px; max-height: calc(100vh - 20px); overflow: auto;
  background: rgba(10, 12, 16, 0.86); border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; padding: 10px 12px;
  font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; color: #cfd8e3; }
.bvh-overlay a { color: #ffc878; text-decoration: none; }
.bvh-overlay a:hover { text-decoration: underline; }
.bvh-overlay header { display: flex; justify-content: space-between; gap: 8px; font: 600 13px/1.4 system-ui, sans-serif; margin-bottom: 4px; }
.bvh-overlay h3 { font: 600 10px/1 system-ui, sans-serif; text-transform: uppercase; letter-spacing: .08em; color: #8592a3; margin: 10px 0 4px; }
.bvh-overlay .row { display: flex; justify-content: space-between; gap: 10px; }
.bvh-overlay .row span:first-child { color: #8592a3; white-space: nowrap; }
.bvh-overlay .row span:last-child { text-align: right; overflow-wrap: anywhere; }
.bvh-overlay .reason { color: #f0b36a; font-size: 11px; }
.bvh-overlay .error { color: #ff7b7b; white-space: pre-wrap; font-size: 11px; max-height: 160px; overflow: auto; }
.bvh-overlay label { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin: 3px 0; color: #8592a3; }
.bvh-overlay select, .bvh-overlay input[type=range], .bvh-overlay button { font: inherit; color: #cfd8e3; background: #1b2028; border: 1px solid rgba(255,255,255,0.15); border-radius: 5px; }
.bvh-overlay button { padding: 3px 8px; cursor: pointer; margin: 3px 4px 0 0; }
.bvh-overlay canvas { display: block; width: 100%; height: 44px; margin-top: 4px; }
.bvh-overlay .note { color: #8592a3; font: 11px/1.4 system-ui, sans-serif; margin-top: 6px; }
.bvh-drop { position: fixed; inset: 0; z-index: 20; display: none; place-items: center; background: rgba(255,200,120,0.12); border: 3px dashed #ffc878;
  font: 600 20px system-ui, sans-serif; color: #ffc878; pointer-events: none; }
`;

const fmt = (value: number | null | undefined, digits = 2): string => (value === null || value === undefined || !Number.isFinite(value) ? '-' : value.toFixed(digits));
const count = (value: number | null | undefined): string => (value === null || value === undefined ? '-' : value.toLocaleString('en-US'));
export function bytes(value: number | null | undefined): string {
    if (value === null || value === undefined) return '-';
    if (value < 1024) return `${value} B`;
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
    return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

class Section {
    readonly element: HTMLElement;
    private readonly rows = new Map<string, HTMLSpanElement>();
    constructor(parent: HTMLElement, title: string) {
        this.element = document.createElement('section');
        const heading = document.createElement('h3');
        heading.textContent = title;
        this.element.append(heading);
        parent.append(this.element);
        // Hidden until it has something to show.
        this.element.hidden = true;
    }
    show(): void {
        this.element.hidden = false;
    }
    set(key: string, value: string): void {
        let span = this.rows.get(key);
        if (!span) {
            this.element.hidden = false;
            const row = document.createElement('div');
            row.className = 'row';
            const name = document.createElement('span');
            name.textContent = key;
            span = document.createElement('span');
            row.append(name, span);
            this.element.append(row);
            this.rows.set(key, span);
        }
        if (span.textContent !== value) span.textContent = value;
    }
}

/** The main-thread frame-time graph: the last 120 frames, 16.7 ms and 33.3 ms marked. */
class FrameGraph {
    private readonly canvas = document.createElement('canvas');
    private readonly times = new Float32Array(120);
    private at = 0;
    private last = 0;
    readonly label = document.createElement('div');
    constructor(parent: HTMLElement) {
        this.canvas.width = 276 * devicePixelRatio;
        this.canvas.height = 44 * devicePixelRatio;
        this.label.className = 'row';
        parent.append(this.canvas, this.label);
    }
    tick(now: number): void {
        if (this.last > 0) {
            this.times[this.at % this.times.length] = now - this.last;
            this.at++;
        }
        this.last = now;
        if (this.at % 4 !== 0) return;
        const context = this.canvas.getContext('2d');
        if (!context) return;
        const { width, height } = this.canvas;
        const scale = height / 50; // 50 ms full height
        context.clearRect(0, 0, width, height);
        const n = Math.min(this.at, this.times.length);
        const bar = width / this.times.length;
        let sum = 0;
        let worst = 0;
        for (let i = 0; i < n; i++) {
            const ms = this.times[(this.at - n + i) % this.times.length];
            sum += ms;
            worst = Math.max(worst, ms);
            context.fillStyle = ms > 33.4 ? '#ff7b7b' : ms > 17.5 ? '#f0b36a' : '#6fcf97';
            const h = Math.min(height, ms * scale);
            context.fillRect((this.times.length - n + i) * bar, height - h, Math.max(1, bar - 1), h);
        }
        context.fillStyle = 'rgba(255,255,255,0.25)';
        for (const line of [16.7, 33.3]) context.fillRect(0, height - line * scale, width, 1);
        this.label.innerHTML = `<span>frame</span><span>${n ? (sum / n).toFixed(1) : '-'} ms avg, ${worst.toFixed(1)} ms worst</span>`;
    }
    stats(): { avgMs: number; worstMs: number; frames: number } {
        const n = Math.min(this.at, this.times.length);
        let sum = 0;
        let worst = 0;
        for (let i = 0; i < n; i++) {
            sum += this.times[i];
            worst = Math.max(worst, this.times[i]);
        }
        return { avgMs: n ? sum / n : 0, worstMs: worst, frames: this.at };
    }
}

export class Overlay {
    readonly root = document.createElement('div');
    private readonly builder: Section;
    private readonly treeSection: Section;
    private readonly raysSection: Section;
    private readonly extra: Section;
    private readonly controls: HTMLElement;
    private readonly reason = document.createElement('div');
    private readonly errors = document.createElement('div');
    private readonly graph: FrameGraph;
    private readonly data: Record<string, unknown> = {};

    constructor(title: string, folder: string) {
        const style = document.createElement('style');
        style.textContent = STYLE;
        document.head.append(style);
        this.root.className = 'bvh-overlay';
        const header = document.createElement('header');
        const base = import.meta.env.BASE_URL;
        header.innerHTML = `<a href="${base}">← Examples</a><span>${title}</span><a href="${SOURCE_TREE}${folder}" target="_blank" rel="noopener">Source ↗</a>`;
        this.root.append(header);
        this.builder = new Section(this.root, 'Builder');
        this.builder.show();
        this.reason.className = 'reason';
        this.builder.element.append(this.reason);
        this.treeSection = new Section(this.root, 'Tree');
        this.raysSection = new Section(this.root, 'Rays');
        this.extra = new Section(this.root, 'Scene');
        const controls = new Section(this.root, 'Controls');
        controls.show();
        this.controls = controls.element;
        const frames = new Section(this.root, 'Main thread');
        frames.show();
        this.graph = new FrameGraph(frames.element);
        this.errors.className = 'error';
        this.root.append(this.errors);
        document.body.append(this.root);
        onExampleError((message) => this.error(message));
    }

    /** Which builder, why, and how long. */
    build(info: BuildInfo): void {
        this.builder.set('builtBy', info.builtBy);
        this.builder.set('build CPU', `${fmt(info.cpuMs)} ms`);
        const source = info.gpuSource && info.gpuSource !== 'timestamps' ? ` (${info.gpuSource})` : '';
        this.builder.set('build GPU', info.gpuMs === undefined ? '-' : info.gpuMs === null ? 'no timestamp-query' : `${fmt(info.gpuMs, 3)} ms${source}`);
        this.reason.textContent = info.fallbackReason ? `fallback: ${info.fallbackReason}` : '';
        this.data.build = info;
    }

    tree(info: TreeInfo): void {
        this.treeSection.set('SAH cost', fmt(info.sah));
        this.treeSection.set('depth', count(info.depth));
        this.treeSection.set('nodes', count(info.nodes));
        this.treeSection.set('triangles', count(info.triangles));
        this.treeSection.set('bytes', bytes(info.bytes));
        this.data.tree = info;
    }

    /** Rays traced per second (`label` names which). */
    rays(label: string, raysPerSecond: number): void {
        const mrays = raysPerSecond / 1e6;
        this.raysSection.set(label, mrays >= 0.1 ? `${mrays.toFixed(2)} Mrays/s` : `${Math.round(raysPerSecond).toLocaleString('en-US')} rays/s`);
        (this.data.rays ??= {} as Record<string, number>);
        (this.data.rays as Record<string, number>)[label] = raysPerSecond;
    }

    /** A free row in the "Scene" section. */
    set(key: string, value: string | number): void {
        this.extra.set(key, typeof value === 'number' ? value.toLocaleString('en-US') : value);
        this.data[key] = value;
    }

    select<T extends string>(label: string, options: readonly T[], value: T, onChange: (value: T) => void): HTMLSelectElement {
        const select = document.createElement('select');
        for (const option of options) select.append(new Option(option, option, option === value, option === value));
        select.addEventListener('change', () => onChange(select.value as T));
        this.labelled(label, select);
        return select;
    }

    range(label: string, min: number, max: number, step: number, value: number, onChange: (value: number) => void): HTMLInputElement {
        const input = document.createElement('input');
        Object.assign(input, { type: 'range', min: String(min), max: String(max), step: String(step), value: String(value) });
        const shown = document.createElement('span');
        shown.textContent = String(value);
        input.addEventListener('input', () => {
            shown.textContent = input.value;
            onChange(Number(input.value));
        });
        const wrap = document.createElement('span');
        wrap.append(input, ' ', shown);
        this.labelled(label, wrap);
        return input;
    }

    checkbox(label: string, value: boolean, onChange: (value: boolean) => void): HTMLInputElement {
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = value;
        input.addEventListener('change', () => onChange(input.checked));
        this.labelled(label, input);
        return input;
    }

    button(text: string, onClick: () => void): HTMLButtonElement {
        const button = document.createElement('button');
        button.textContent = text;
        button.addEventListener('click', onClick);
        this.controls.append(button);
        return button;
    }

    note(text: string): void {
        const note = document.createElement('div');
        note.className = 'note';
        note.textContent = text;
        this.controls.append(note);
    }

    private labelled(text: string, control: HTMLElement): void {
        const label = document.createElement('label');
        label.append(text, control);
        this.controls.append(label);
    }

    /** Call once per frame (the graph's clock). */
    frame(now = performance.now()): void {
        this.graph.tick(now);
    }

    error(message: string): void {
        this.errors.textContent += `${message}\n`;
    }

    /** Everything shown, for the smoke report. */
    snapshot(): Record<string, unknown> {
        return { ...this.data, frames: this.graph.stats() };
    }
}

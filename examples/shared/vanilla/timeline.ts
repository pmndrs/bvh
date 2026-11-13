/**
 * A per-mesh timeline strip for the vanilla pages about when trees arrive
 * (fast-first, the worker pages): one row per mesh, time across, a tick for
 * each event (a GPU tree placed, a tinybvh tree swapped in, ...) and bars for
 * spans (a build running). A canvas along the page's bottom, redrawn at most
 * once a frame, with a legend.
 *
 * {@link hashColor} is the ray-cast view's `hashColor` (raycast-view.ts) in
 * TypeScript, so a page that tints instances by material number can show the
 * same colour in a legend.
 */

export interface TimelineKind {
    readonly label: string;
    /** CSS colour. */
    readonly color: string;
    /** `'mark'` (a tick) or `'span'` (a bar). */
    readonly shape: 'mark' | 'span';
}

interface TimelineEvent {
    readonly lane: number;
    readonly kind: string;
    readonly from: number;
    readonly to: number;
}

const STYLE = `
.bvh-timeline { position: fixed; left: 336px; right: 16px; bottom: 16px; background: rgba(12, 14, 18, 0.88); border: 1px solid rgba(255,255,255,0.1);
  border-radius: 8px; padding: 8px 10px; font: 11px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; color: #cfd8e3; z-index: 5; }
.bvh-timeline header { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 4px; flex-wrap: wrap; }
.bvh-timeline .legend span { margin-left: 10px; white-space: nowrap; }
.bvh-timeline .legend i { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
.bvh-timeline canvas { display: block; width: 100%; }
@media (max-width: 720px) { .bvh-timeline { left: 16px; } }
`;

export class Timeline {
    readonly root = document.createElement('div');
    private readonly canvas = document.createElement('canvas');
    private readonly heading = document.createElement('span');
    private readonly events: TimelineEvent[] = [];
    private lanes: number;
    private start = performance.now();
    private dirty = true;

    constructor(
        title: string,
        private readonly kinds: Readonly<Record<string, TimelineKind>>,
        lanes: number,
    ) {
        this.lanes = lanes;
        if (!document.getElementById('bvh-timeline-style')) {
            const style = document.createElement('style');
            style.id = 'bvh-timeline-style';
            style.textContent = STYLE;
            document.head.append(style);
        }
        this.root.className = 'bvh-timeline';
        const header = document.createElement('header');
        this.heading.textContent = title;
        const legend = document.createElement('span');
        legend.className = 'legend';
        for (const kind of Object.values(kinds)) {
            const entry = document.createElement('span');
            const swatch = document.createElement('i');
            swatch.style.background = kind.color;
            entry.append(swatch, kind.label);
            legend.append(entry);
        }
        header.append(this.heading, legend);
        this.root.append(header, this.canvas);
        document.body.append(this.root);
    }

    /** Clears the events and starts the clock again (a new run), with `lanes` rows. */
    reset(lanes: number, title?: string, start = performance.now()): void {
        this.events.length = 0;
        this.lanes = lanes;
        this.start = start;
        if (title) this.heading.textContent = title;
        this.dirty = true;
    }

    /** Makes room for `lanes` rows (a mesh added during a run), keeping the events. */
    grow(lanes: number): void {
        if (lanes <= this.lanes) return;
        this.lanes = lanes;
        this.dirty = true;
    }

    /** The clock's zero (`performance.now()` time). */
    get origin(): number {
        return this.start;
    }

    /** An event at `at` (`performance.now()` time). */
    mark(lane: number, kind: string, at = performance.now()): void {
        this.events.push({ lane, kind, from: at, to: at });
        this.dirty = true;
    }

    /** A span from `from` to `to`. */
    span(lane: number, kind: string, from: number, to = performance.now()): void {
        this.events.push({ lane, kind, from, to });
        this.dirty = true;
    }

    /** Events of `kind` per lane (for checks). */
    count(kind: string): number[] {
        const counts = new Array<number>(this.lanes).fill(0);
        for (const event of this.events) if (event.kind === kind && event.lane < this.lanes) counts[event.lane]++;
        return counts;
    }

    /** Milliseconds from the clock's zero to the last event of `kind` (or of any kind), or null. */
    lastMs(kind?: string): number | null {
        let last = -Infinity;
        for (const event of this.events) if (!kind || event.kind === kind) last = Math.max(last, event.to);
        return Number.isFinite(last) ? last - this.start : null;
    }

    /** Redraws if something changed (call once a frame). */
    draw(): void {
        if (!this.dirty) return;
        this.dirty = false;
        const ratio = Math.min(devicePixelRatio, 2);
        const cssWidth = Math.max(100, this.canvas.clientWidth);
        const rowHeight = this.lanes > 60 ? 1 : this.lanes > 20 ? 3 : 5;
        const axis = 14;
        const cssHeight = Math.min(160, Math.max(40, this.lanes * rowHeight + axis + 4));
        if (this.canvas.width !== Math.round(cssWidth * ratio) || this.canvas.height !== Math.round(cssHeight * ratio)) {
            this.canvas.width = Math.round(cssWidth * ratio);
            this.canvas.height = Math.round(cssHeight * ratio);
            this.canvas.style.height = `${cssHeight}px`;
        }
        const context = this.canvas.getContext('2d');
        if (!context) return;
        context.setTransform(ratio, 0, 0, ratio, 0, 0);
        context.clearRect(0, 0, cssWidth, cssHeight);
        const end = Math.max(50, (this.lastMs() ?? 0) * 1.05);
        const height = cssHeight - axis;
        const row = height / Math.max(1, this.lanes);
        const x = (ms: number) => (ms / end) * (cssWidth - 2) + 1;
        context.fillStyle = 'rgba(255,255,255,0.04)';
        for (let lane = 0; lane < this.lanes; lane += 2) context.fillRect(0, lane * row, cssWidth, row);
        for (const event of this.events) {
            const kind = this.kinds[event.kind];
            if (!kind || event.lane >= this.lanes) continue;
            context.fillStyle = kind.color;
            const y = event.lane * row;
            const h = Math.max(1, row - (row > 3 ? 1 : 0));
            const from = x(event.from - this.start);
            if (kind.shape === 'span') context.fillRect(from, y + h * 0.25, Math.max(1, x(event.to - this.start) - from), Math.max(1, h * 0.5));
            else context.fillRect(x(event.to - this.start) - 1, y, 3, h);
        }
        // The axis: a tick every round number of milliseconds.
        const step = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000].find((candidate) => end / candidate <= 8) ?? 20000;
        context.fillStyle = 'rgba(255,255,255,0.45)';
        context.font = '10px ui-monospace, monospace';
        for (let ms = 0; ms <= end; ms += step) {
            context.fillRect(x(ms), height, 1, 4);
            context.fillText(ms >= 1000 ? `${ms / 1000} s` : `${ms} ms`, x(ms) + 2, cssHeight - 2);
        }
    }

    destroy(): void {
        this.root.remove();
    }
}

/** The ray-cast view's `hashColor(value)` as a CSS colour (the same bits as the WGSL). */
export function hashColor(value: number): string {
    let h = (Math.imul(value >>> 0, 747796405) + 2891336453) >>> 0;
    h = Math.imul((h >>> ((h >>> 28) + 4)) ^ h, 277803737) >>> 0;
    h = ((h >>> 22) ^ h) >>> 0;
    const channel = (shift: number) => Math.round((0.25 + 0.75 * (((h >>> shift) & 0xff) / 255)) * 255);
    return `rgb(${channel(0)}, ${channel(8)}, ${channel(16)})`;
}

/**
 * Measurements to CSV rows (bench/results/compare.csv, trace.csv's columns
 * plus `method`) and to markdown tables (the runner prints them;
 * compare/RESULTS.md is written from them).
 */

import { agreementText } from './agree.ts';
import type { Measurement } from './suites.ts';

export interface RunContext {
    readonly date: string;
    readonly commit: string;
    readonly device: string;
    /** trace.csv's `browser` column: the browser, or `Node v22...` for the Node runner. */
    readonly browser: string;
}

export const CSV_HEADER = 'date,commit,device,browser,scene,metric,value,method';

const quote = (text: string) => (/[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text);

/** One row per value, plus `.../checked` and `.../disagree` rows for answers that were checked against ours. */
export function csvRows(context: RunContext, measurements: readonly Measurement[]): string[] {
    const rows: string[] = [];
    const prefix = [context.date, context.commit, context.device, context.browser].map(quote).join(',');
    for (const m of measurements) {
        const metric = `compare/${m.page}/${m.metric}/${m.library}`;
        const row = (name: string, value: number | string, method: string) => rows.push(`${prefix},${quote(m.scene)},${quote(name)},${value},${quote(method)}`);
        if (m.value === null) row(`${metric}/${m.unit}`, '', `not measured: ${m.note ?? m.method}`);
        else row(`${metric}/${m.unit}`, Number.isInteger(m.value) ? m.value : Number(m.value.toPrecision(6)), m.method);
        if (m.count !== undefined) row(`${metric}/queries`, m.count, 'queries per timed run');
        if (m.agreement) {
            row(`${metric}/checked`, m.agreement.checked, 'answers checked against ours');
            row(`${metric}/disagree`, m.agreement.disagree, 'answers that disagree with ours (see compare/core/agree.ts)');
            row(`${metric}/ties`, m.agreement.ties, 'answers equal to ours up to a tie');
        }
    }
    return rows;
}

export function formatValue(value: number | null, unit: string): string {
    if (value === null) return 'n/a';
    if (unit === 'bytes') return value >= 1024 * 1024 ? `${(value / 1024 / 1024).toFixed(1)} MiB` : `${(value / 1024).toFixed(1)} KiB`;
    if (unit === 'ms') return value >= 100 ? `${value.toFixed(0)} ms` : `${value.toFixed(value >= 10 ? 1 : 2)} ms`;
    if (unit.endsWith('/s')) return value >= 1e6 ? `${(value / 1e6).toFixed(2)} M/s` : value >= 1e3 ? `${(value / 1e3).toFixed(1)} k/s` : `${value.toFixed(0)}/s`;
    if (unit === 'relative') return value.toFixed(1);
    return value.toLocaleString('en-US');
}

/**
 * A cell: the value, its ratio to ours (for rates `x` faster when above 1;
 * for times and bytes `x` ours), and the agreement. Unranked values (answers
 * that disagree) get no ratio.
 */
export function cell(m: Measurement | undefined, ours: Measurement | undefined): string {
    if (!m) return '';
    if (m.value === null) return `n/a${m.note ? ` (${m.note})` : ''}`;
    let text = formatValue(m.value, m.unit);
    if (ours && ours !== m && ours.value && m.ranked && m.unit !== 'levels' && m.unit !== 'boxes' && m.unit !== 'leaves') {
        const ratio = m.value / ours.value;
        text += ` (${ratio.toFixed(2)}x ours)`;
    }
    if (m.agreement) text += m.agreement.disagree ? `, **${m.agreement.disagree}/${m.agreement.checked} disagree**` : `, ${agreementText(m.agreement)}`;
    return text;
}

/** A markdown table: one row per library, one column per metric. */
export function table(measurements: readonly Measurement[], metrics: readonly string[], names: Readonly<Record<string, string>>): string {
    const libraries = [...new Set(measurements.map((m) => m.library))];
    const find = (library: string, metric: string) => measurements.find((m) => m.library === library && m.metric === metric);
    const lines = [`| Library | ${metrics.join(' | ')} |`, `| --- | ${metrics.map(() => '---').join(' | ')} |`];
    for (const library of libraries) {
        if (!metrics.some((metric) => find(library, metric))) continue;
        lines.push(`| ${names[library] ?? library} | ${metrics.map((metric) => cell(find(library, metric), find('ours', metric))).join(' | ')} |`);
    }
    return lines.join('\n');
}

/**
 * How busy the Mac (or PC) is, recorded with every timed page: timings taken
 * on a loaded machine do not count as a gate result (tests/gpu/README.md,
 * "Timed pages"). The load average is the OS's run-queue average; the GPU
 * process count is how many Chrome GPU processes other jobs have running.
 */

import { execFileSync } from 'node:child_process';
import { loadavg } from 'node:os';

export interface MachineLoad {
    /** 1, 5 and 15 minute load averages (0 on Windows, where Node does not report them). */
    readonly loadAverage: readonly [number, number, number];
    /** Chrome GPU processes running (ours included), or -1 when `ps` is unavailable. */
    readonly chromeGpuProcesses: number;
    readonly cpus: number;
    readonly at: string;
}

/** A machine counts as quiet when its 1-minute load is below this many runnable tasks per core. */
export const QUIET_LOAD_PER_CORE = 0.5;

export function machineLoad(cpus: number): MachineLoad {
    let chromeGpuProcesses = -1;
    if (process.platform !== 'win32') {
        try {
            const lines = execFileSync('ps', ['-Ao', 'command'], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 }).split('\n');
            chromeGpuProcesses = lines.filter((line) => /Chrome/.test(line) && /--type=gpu-process/.test(line)).length;
        } catch {
            // no ps
        }
    }
    const [one, five, fifteen] = loadavg();
    return { loadAverage: [one, five, fifteen], chromeGpuProcesses, cpus, at: new Date().toISOString() };
}

/**
 * Quiet: the 1-minute load below {@link QUIET_LOAD_PER_CORE} per core. Windows reports no load
 * average (0), so there a run is quiet only when the operator says so with `BVH_MACHINE_QUIET=1`.
 */
export function isQuiet(load: MachineLoad): boolean {
    if (process.platform === 'win32') return process.env.BVH_MACHINE_QUIET === '1';
    return load.loadAverage[0] < QUIET_LOAD_PER_CORE * load.cpus;
}

export function describeLoad(load: MachineLoad): string {
    return `load ${load.loadAverage.map((value) => value.toFixed(1)).join(' / ')} on ${load.cpus} cores, ${load.chromeGpuProcesses} Chrome GPU processes${isQuiet(load) ? ' (quiet)' : ' (LOADED: timings are provisional)'}`;
}

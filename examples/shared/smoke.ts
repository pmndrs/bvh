/**
 * Smoke mode: the GPU runner (`pnpm examples:smoke`, which is `tests/gpu/run.ts
 * --examples`) opens each example with `?token=<id>`. The example runs as it
 * would for a person, and once it has shown what it is about (trees built,
 * rays traced, a few frames drawn) it calls {@link ready}, which leaves the
 * runner's report on `window.__gpuTestReport` (the contract in
 * tests/gpu/RUNNER.md). Any uncaught error, unhandled rejection or WebGPU
 * `uncapturederror` fails it; the runner also fails on console errors.
 *
 * Without a token nothing here changes the page, except that errors are shown
 * in the overlay.
 */

const params = new URLSearchParams(location.search);

/** The runner's token, or null when a person opened the page. */
export const smokeToken = params.get('token');
export const isSmoke = smokeToken !== null;

const startTime = performance.now();
const deviceErrors: string[] = [];
const errorListeners: ((message: string) => void)[] = [];
let adapter: Record<string, unknown> | null = null;
let reported = false;

function report(status: 'pass' | 'fail', data: Record<string, unknown>, message?: string): void {
    if (reported) return;
    reported = true;
    if (status === 'pass' && deviceErrors.length > 0) {
        status = 'fail';
        message = deviceErrors.join(' | ');
    }
    const plain = JSON.parse(JSON.stringify(data, (_key, value: unknown) => (ArrayBuffer.isView(value) ? Array.from(value as unknown as ArrayLike<number>) : value))) as Record<string, unknown>;
    (window as unknown as Record<string, unknown>).__gpuTestReport = {
        token: smokeToken,
        status,
        message,
        data: plain,
        adapter,
        deviceErrors: [...deviceErrors],
        durationMs: performance.now() - startTime,
    };
    (window as unknown as Record<string, unknown>).__exampleReady = status === 'pass';
}

/** Called with every error the page sees (the overlay shows them). */
export function onExampleError(listener: (message: string) => void): void {
    errorListeners.push(listener);
}

/** Fails the smoke run (and shows the error). */
export function fail(error: unknown): void {
    const message = error instanceof Error ? `${error.message}${error.stack ? `\n${error.stack}` : ''}` : String(error);
    for (const listener of errorListeners) listener(message);
    console.error(message);
    report('fail', {}, message);
}

addEventListener('error', (event) => fail(event.error ?? event.message));
addEventListener('unhandledrejection', (event) => fail(event.reason));

/** Records the device's `uncapturederror`s and loss, and the adapter for the runner's GPU check. */
export function watchDevice(device: GPUDevice): void {
    const info = device.adapterInfo;
    adapter = {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
        isFallbackAdapter: (info as { isFallbackAdapter?: boolean }).isFallbackAdapter ?? false,
        features: [...device.features].sort(),
    };
    device.addEventListener('uncapturederror', (event) => {
        const message = `WebGPU: ${(event as GPUUncapturedErrorEvent).error.message}`;
        deviceErrors.push(message);
        for (const listener of errorListeners) listener(message);
    });
    void device.lost.then((lost) => {
        if (lost.reason !== 'destroyed') fail(`device lost: ${lost.message}`);
    });
}

/**
 * The example has shown what it is about. In smoke mode, waits for the GPU to
 * go idle (late validation errors) and reports `data`; otherwise does nothing.
 */
export async function ready(device: GPUDevice | null, data: Record<string, unknown>): Promise<void> {
    if (!isSmoke || reported) return;
    if (device) {
        await device.queue.onSubmittedWorkDone();
        await new Promise((resolve) => setTimeout(resolve, 50));
    }
    report('pass', data);
}

/**
 * Where scene assets come from, shared by the three loader (loader.ts) and the
 * three-free glTF reader (vanilla/gltf.ts): `?model=<path>` names a glTF under
 * the assets base, which is `/assets/` (the dev server maps it to
 * `BVH_ASSETS_DIR`, see vite.config.ts) or `?assets=<url>`.
 */

const params = new URLSearchParams(location.search);

/** Where `?model=` paths are read from. */
export function assetsBase(): string {
    const base = params.get('assets') ?? `${import.meta.env.BASE_URL}assets/`;
    return base.endsWith('/') ? base : `${base}/`;
}

/** The `?model=` path, or null. */
export function requestedModel(): string | null {
    return params.get('model');
}

/** A `?model=` path as a URL (absolute, `blob:` and `http(s):` URLs pass through). */
export function assetUrl(path: string): string {
    return /^(https?:|blob:|\/)/.test(path) ? path : assetsBase() + path;
}

/**
 * A failed `?model=` load as an error that says what to do: a built site (the
 * hosted examples) has no scene assets, so a bundled path needs `?assets=` there.
 */
export function assetLoadError(path: string, cause: unknown): Error {
    const message = cause instanceof Error ? cause.message : String(cause);
    const hosted = import.meta.env.PROD && !params.has('assets') && !/^(https?:|blob:|\/)/.test(path);
    const hint = hosted
        ? ' The scene assets (Sponza, Bistro, ...) are not part of this site: add ?assets=<url> of a folder that holds the path, or drop a glTF on the page.'
        : '';
    return new Error(`Could not load ${path} (${message}).${hint}`, { cause });
}

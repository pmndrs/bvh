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

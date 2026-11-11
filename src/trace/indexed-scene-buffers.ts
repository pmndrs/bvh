// Which triangle buffers are indexed scenes' (internal: not re-exported).
// `writeTriangleFilterFlags` uses it to refuse an indexed scene's buffer passed
// bare as records, or without a range.

const indexedSceneTriangles = new WeakSet<ArrayBufferLike>();

/** Marks an indexed scene's triangle buffer (`assembleTraceScene`): it holds headers and side tables between meshes. */
export function markIndexedSceneTriangles(triangles: ArrayBuffer): void {
    indexedSceneTriangles.add(triangles);
}

export function isIndexedSceneTriangles(triangles: ArrayBufferLike): boolean {
    return indexedSceneTriangles.has(triangles);
}

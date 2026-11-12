/**
 * compare/queries: box, sphere and closest-point queries per second against
 * three-mesh-bvh and Rapier (the others have none), on the same seeded
 * shapes, each answer checked against ours. Live: a probe sphere drifts
 * through the mesh; the chosen library says whether it touches the mesh and
 * where the closest point is.
 */

import { BufferGeometry, Line, LineBasicMaterial, Mesh, MeshBasicMaterial, SphereGeometry, Vector3 } from 'three';
import { BVH_DOUBLE_SIDE, buildHplocOnCpu, CpuMeshBvh } from '@pmndrs/bvh';
import { MESH_ADAPTERS } from '../../../compare/adapters/index.ts';
import type { Vec3 } from '../../../compare/adapters/types.ts';
import { checkClosest, checkFirst, emptyAgreement, agreementText } from '../../../compare/core/agree.ts';
import { diagonal } from '../../../compare/core/geometry.ts';
import { makePoints, makeRays } from '../../../compare/core/rays.ts';
import { buildAll, queriesSuite } from '../../../compare/core/suites.ts';
import { createComparePage, expectAgreement, quick } from '../../shared/compare.ts';
import { ready } from '../../shared/smoke.ts';

const page = await createComparePage('compare/queries', 'compare/queries', 'Box, sphere and closest-point queries per second on the same seeded shapes. bvh-tree, Babylon\'s picking and three\'s raycast have no such queries.');
const { app, geometry, panel } = page;
const log = (line: string) => panel.status(line);

const adapters = MESH_ADAPTERS.filter((adapter) => adapter.features.closestPoint.status !== 'no' || adapter.features.intersectsBox.status !== 'no');
const { built } = await buildAll(adapters, geometry, { log, buildRuns: 1 });
const results = await queriesSuite(built, geometry, { log, queries: quick ? 512 : 2048, plan: quick ? { runs: 3, budgetMs: 250 } : {} });
panel.table('Queries per second', results, ['box', 'sphere', 'closest-point']);
panel.paragraph('Rapier reports no triangle for a closest point (its feature is a vertex, edge or face of the triangle), so it agrees by distance: its answers count as ties. three-mesh-bvh ties are another triangle at the same distance (a shared vertex or edge).');

// What else: the tree our GPU H-PLOC builder writes (its CPU twin builds the same one) answers the same queries.
const ours = built.find((entry) => entry.adapter.id === 'ours')!.mesh;
const hploc = new CpuMeshBvh(buildHplocOnCpu({ positions: geometry.positions, indices: geometry.indices }));
const points = makePoints(geometry, 256, 21);
const rays = makeRays(geometry, 'inside', 256, 21);
const closestAgreement = emptyAgreement();
const rayAgreement = emptyAgreement();
const scale = diagonal(geometry);
points.forEach((point, i) => {
    const hit = hploc.closestPointToPoint(point);
    checkClosest(closestAgreement, i, ours.closestPoint!(point), hit && { triangle: hit.faceIndex, distance: hit.distance, point: [hit.point.x, hit.point.y, hit.point.z] }, scale);
});
rays.forEach((ray, i) => {
    const hit = hploc.raycastFirst(ray, BVH_DOUBLE_SIDE);
    checkFirst(rayAgreement, i, ours.raycastFirst!(ray), hit && { triangle: hit.triangle, t: hit.distance }, scale);
});
panel.whatElse(
    `Every builder writes one layout, so a tree built on the GPU answers the CPU queries too: the tree of our GPU H-PLOC builder (built here by its CPU twin, the same tree) against the tinybvh tree: closest points ${agreementText(closestAgreement)}, closest hits ${agreementText(rayAgreement)}. Swapping builders never changes the query code.`,
);
panel.csv(app.device);

// Live probe.
const radius = diagonal(geometry) * 0.03;
const probe = new Mesh(new SphereGeometry(radius, 20, 14), new MeshBasicMaterial({ color: 0x6fcf97, wireframe: true }));
const line = new Line(new BufferGeometry().setFromPoints([new Vector3(), new Vector3()]), new LineBasicMaterial({ color: 0xffc878 }));
app.scene.add(probe, line);
const libraries = new Map(built.map((entry) => [entry.adapter.name, entry.mesh]));
let chosen = [...libraries.keys()][0];
app.overlay.select('query with', [...libraries.keys()], chosen, (value) => (chosen = value));
const b = geometry.bounds;
const checked = expectAgreement(results, ['tmbvh-center', 'tmbvh-sah', 'rapier'], log);
let frames = 0;
app.loop(async (now) => {
    const s = now / 4000;
    const center: Vec3 = [b[0] + (b[3] - b[0]) * (0.5 + 0.45 * Math.sin(s)), b[1] + (b[4] - b[1]) * (0.5 + 0.45 * Math.sin(s * 1.3)), b[2] + (b[5] - b[2]) * (0.5 + 0.45 * Math.cos(s * 0.7))];
    probe.position.set(...center);
    const mesh = libraries.get(chosen)!;
    const start = performance.now();
    const touches = mesh.intersectsSphere?.(center, radius) ?? null;
    const closest = mesh.closestPoint?.(center) ?? null;
    const ms = performance.now() - start;
    (probe.material as MeshBasicMaterial).color.set(touches ? 0xff7b7b : 0x6fcf97);
    if (closest) line.geometry.setFromPoints([probe.position, new Vector3(...closest.point)]);
    line.visible = closest !== null;
    app.overlay.set('probe', `${touches === null ? 'no sphere query' : touches ? 'touches' : 'clear'}, closest ${closest ? closest.distance.toPrecision(4) : 'n/a'}, ${ms.toFixed(3)} ms`);
    if (++frames === 10) await ready(app.device, { ...app.overlay.snapshot(), csv: panel.csvText, status: panel.statusText, checkedAgainstOurs: checked, hplocAgrees: closestAgreement.disagree + rayAgreement.disagree === 0 });
});

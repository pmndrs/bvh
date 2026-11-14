// Code samples for docs/three/points-and-lines.mdx (see getting-started.ts for how they are checked).

// #region three-points
import { BufferGeometry, Line, Points, Raycaster } from 'three';
import { acceleratedRaycast, computeBoundsTree, disposeBoundsTree, LineBVH, PointsBVH } from '@pmndrs/bvh/three';

// #endregion

export function threePoints(cloud: Points, line: Line): number {
    // #region three-points
    BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
    BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
    Points.prototype.raycast = acceleratedRaycast;
    Line.prototype.raycast = acceleratedRaycast; // LineSegments and LineLoop inherit it

    const tree = cloud.geometry.computeBoundsTree({ type: PointsBVH }); // no prepareBoundsTrees() needed
    line.geometry.computeBoundsTree({ type: LineBVH }); // LineSegmentsBVH, LineLoopBVH for the others

    const raycaster = new Raycaster();
    raycaster.params.Points.threshold = 0.01;
    raycaster.params.Line.threshold = 0.02;
    raycaster.firstHitOnly = true; // optional: only the closest
    const [hit] = raycaster.intersectObjects([cloud, line]); // distance, distanceToRay, point, index, object
    // #endregion
    return (hit ? 1 : 0) + tree.primitiveCount;
}

// #region three-points-queries
import { Box3, Vector3 } from 'three';

// #endregion

export function threePointsQueries(cloud: Points): number[] {
    // #region three-points-queries
    const tree = cloud.geometry.computeBoundsTree({ type: PointsBVH });
    const box = new Box3(new Vector3(-1, 0, -1), new Vector3(1, 1, 1));
    const inside: number[] = [];
    tree.shapecast({
        intersectsBounds: (bounds) => box.intersectsBox(bounds),
        intersectsPoint: (point, index) => {
            if (box.containsPoint(point)) inside.push(index); // the entry; its vertex is geometry.index?.getX(index) ?? index
            return false;
        },
    });
    const nearest = tree.closestPointToPoint(new Vector3(0, 2, 0)); // { point: Vector3, distance, index } or null
    tree.refit(); // after editing the positions in place (bump the attribute's version)
    const bytes = tree.serialize(); // new PointsBVH(geometry, { layout: PointsBVH.deserialize(bytes).layout })
    // #endregion
    return nearest ? [...inside, bytes.byteLength] : inside;
}

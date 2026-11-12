# compare:node, 2026-11-12, 4824f5b, Apple M5 Pro, Node v22.22.2 (indicative)

## torus-knot: 100,000 triangles, 51,051 vertices

### Build, refit and memory

| Library | build | refit | memory | heap | refit-agreement |
| --- | --- | --- | --- | --- | --- |
| @pmndrs/bvh (tinybvh) | 33.2 ms | 32.3 ms | 8.0 MiB | 8.1 MiB |  |
| three-mesh-bvh (CENTER, default) | 24.9 ms (0.75x ours) | 4.91 ms (0.15x ours) | 873.6 KiB (0.11x ours) | 962.7 KiB (0.12x ours) | 0, 512/512 agree |
| three-mesh-bvh (SAH) | 110 ms (3.32x ours) | 4.53 ms (0.14x ours) | 873.3 KiB (0.11x ours) | 993.6 KiB (0.12x ours) | 0, 512/512 agree |
| bvh-tree | 91.7 ms (2.77x ours) | n/a (not available) | 12.2 MiB (1.53x ours) | 19.6 MiB (2.40x ours) |  |
| Babylon.js picking (submesh octree) | 9.87 ms (0.30x ours) | 2.21 ms (0.07x ours) | n/a | 6.6 MiB (0.81x ours) | 0, 512/512 agree |
| Babylon.js picking (brute force) | 5.48 ms (0.17x ours) | 0.32 ms (0.01x ours) | n/a | 6.1 MiB (0.75x ours) | 0, 512/512 agree |
| Rapier (trimesh collider) | 21.6 ms (0.65x ours) | n/a (a trimesh is rebuilt (setShape), not refitted) | 0.0 KiB (0.00x ours) | 20.3 MiB (2.49x ours) |  |
| three Mesh.raycast (no BVH) | 0.66 ms (0.02x ours) | n/a (not available) | 0.0 KiB (0.00x ours) | 48.5 KiB (0.01x ours) |  |
| @pmndrs/bvh H-PLOC (CPU twin) | 282 ms (8.50x ours) |  | 7.9 MiB (0.99x ours) |  |  |
| @pmndrs/bvh LBVH (CPU twin) | 113 ms (3.39x ours) |  | 8.2 MiB (1.03x ours) |  |  |

### Tree quality

| Library | sah | depth | nodes | leaves |
| --- | --- | --- | --- | --- |
| @pmndrs/bvh (tinybvh) | 58.5 | 20 | 104,763 | 52,382 |
| three-mesh-bvh (CENTER, default) | 83.0 (1.42x ours) | 18 | 27,955 | 13,978 |
| three-mesh-bvh (SAH) | 76.0 (1.30x ours) | 16 | 27,947 | 13,974 |
| bvh-tree | 83.6 (1.43x ours) | 18 | 27,959 | 13,980 |
| @pmndrs/bvh H-PLOC (CPU twin) | 59.3 (1.01x ours) | 22 | 103,187 | 51,594 |
| @pmndrs/bvh LBVH (CPU twin) | 64.4 (1.10x ours) | 25 | 111,995 | 55,998 |

### CPU raycasts (rays/s)

| Library | first/outside | first/inside | all/outside | all/inside |
| --- | --- | --- | --- | --- |
| @pmndrs/bvh (tinybvh) | 413.0 k/s, 4096/4096 agree | 392.0 k/s, 4096/4096 agree | 181.0 k/s, 4096/4096 agree | 222.7 k/s, 4096/4096 agree |
| three-mesh-bvh (CENTER, default) | 347.7 k/s (0.84x ours), 4096/4096 agree | 331.7 k/s (0.85x ours), 4096/4096 agree | 135.0 k/s (0.75x ours), 4096/4096 agree | 183.9 k/s (0.83x ours), 4096/4096 agree |
| three-mesh-bvh (SAH) | 374.2 k/s (0.91x ours), 4096/4096 agree | 364.6 k/s (0.93x ours), 4096/4096 agree | 150.3 k/s (0.83x ours), 4096/4096 agree | 183.0 k/s (0.82x ours), 4096/4096 agree |
| bvh-tree | 142.1 k/s (0.34x ours), 4096/4096 agree | 207.9 k/s (0.53x ours), 4096/4096 agree | 118.0 k/s (0.65x ours), 4096/4096 agree | 150.5 k/s (0.68x ours), 4096/4096 agree |
| Babylon.js picking (submesh octree) | 8.6 k/s (0.02x ours), 4096/4096 agree, 11 ties | 10.3 k/s (0.03x ours), 4096/4096 agree, 9 ties | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) |
| Babylon.js picking (brute force) | 530/s (0.00x ours), 771/771 agree, 1 ties | 514/s (0.00x ours), 661/661 agree, 1 ties | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) |
| Rapier (trimesh collider) | 381.2 k/s (0.92x ours), 4096/4096 agree, 2929 ties | 501.2 k/s (1.28x ours), 4096/4096 agree, 2739 ties | n/a (intersectionsWithRay gives one hit per collider, not per triangle) | n/a (intersectionsWithRay gives one hit per collider, not per triangle) |
| three Mesh.raycast (no BVH) | 276/s (0.00x ours), 480/480 agree | 301/s (0.00x ours), 329/329 agree | 255/s (0.00x ours), 481/481 agree | 244/s (0.00x ours), 353/353 agree |

### Queries (queries/s)

| Library | box | sphere | closest-point |
| --- | --- | --- | --- |
| bvh-tree | n/a (not available) | n/a (not available) | n/a (not available) |
| Babylon.js picking (submesh octree) | n/a (not available) | n/a (not available) | n/a (not available) |
| Babylon.js picking (brute force) | n/a (not available) | n/a (not available) | n/a (not available) |
| three Mesh.raycast (no BVH) | n/a (not available) | n/a (not available) | n/a (not available) |
| @pmndrs/bvh (tinybvh) | 274.7 k/s, 2048/2048 agree | 560.7 k/s, 2048/2048 agree | 33.5 k/s, 2048/2048 agree |
| three-mesh-bvh (CENTER, default) | 333.1 k/s (1.21x ours), 2048/2048 agree | 523.8 k/s (0.93x ours), 2048/2048 agree | 47.4 k/s (1.41x ours), 2048/2048 agree, 468 ties |
| three-mesh-bvh (SAH) | 351.4 k/s (1.28x ours), 2048/2048 agree | 560.9 k/s (1.00x ours), 2048/2048 agree | 49.5 k/s (1.48x ours), 2048/2048 agree, 503 ties |
| Rapier (trimesh collider) | 631.6 k/s (2.30x ours), 2048/2048 agree | 98.4 k/s (0.18x ours), 2048/2048 agree | 81.4 k/s (2.43x ours), 2048/2048 agree, 2048 ties |

## furnished-room: 269,764 triangles, 160,380 vertices

### Build, refit and memory

| Library | build | refit | memory | heap | refit-agreement |
| --- | --- | --- | --- | --- | --- |
| @pmndrs/bvh (tinybvh) | 84.4 ms | 74.8 ms | 20.1 MiB | 20.2 MiB |  |
| three-mesh-bvh (CENTER, default) | 82.1 ms (0.97x ours) | 13.1 ms (0.18x ours) | 2.4 MiB (0.12x ours) | 2.5 MiB (0.12x ours) | 0, 512/512 agree |
| three-mesh-bvh (SAH) | 352 ms (4.17x ours) | 12.8 ms (0.17x ours) | 2.3 MiB (0.11x ours) | 2.4 MiB (0.12x ours) | 0, 512/512 agree |
| bvh-tree | 270 ms (3.19x ours) | n/a (not available) | 32.9 MiB (1.64x ours) | 54.9 MiB (2.71x ours) |  |
| Babylon.js picking (submesh octree) | 17.1 ms (0.20x ours) | 4.64 ms (0.06x ours) | n/a | 19.0 MiB (0.94x ours) | 0, 512/512 agree, 1 ties |
| Babylon.js picking (brute force) | 12.8 ms (0.15x ours) | 0.57 ms (0.01x ours) | n/a | 17.7 MiB (0.87x ours) | 0, 512/512 agree, 1 ties |
| Rapier (trimesh collider) | 69.9 ms (0.83x ours) | n/a (a trimesh is rebuilt (setShape), not refitted) | 0.0 KiB (0.00x ours) | 54.5 MiB (2.69x ours) |  |
| three Mesh.raycast (no BVH) | 2.30 ms (0.03x ours) | n/a (not available) | 0.0 KiB (0.00x ours) | 48.2 KiB (0.00x ours) |  |
| @pmndrs/bvh H-PLOC (CPU twin) | 872 ms (10.33x ours) |  | 20.2 MiB (1.00x ours) |  |  |
| @pmndrs/bvh LBVH (CPU twin) | 335 ms (3.97x ours) |  | 20.9 MiB (1.04x ours) |  |  |

### Tree quality

| Library | sah | depth | nodes | leaves |
| --- | --- | --- | --- | --- |
| @pmndrs/bvh (tinybvh) | 40.2 | 29 | 238,383 | 119,192 |
| three-mesh-bvh (CENTER, default) | 62.2 (1.55x ours) | 30 | 79,077 | 39,539 |
| three-mesh-bvh (SAH) | 52.3 (1.30x ours) | 25 | 74,759 | 37,380 |
| bvh-tree | 64.3 (1.60x ours) | 42 | 84,509 | 42,255 |
| @pmndrs/bvh H-PLOC (CPU twin) | 39.3 (0.98x ours) | 37 | 241,251 | 120,626 |
| @pmndrs/bvh LBVH (CPU twin) | 48.5 (1.20x ours) | 37 | 264,817 | 132,409 |

### CPU raycasts (rays/s)

| Library | first/outside | first/inside | all/outside | all/inside |
| --- | --- | --- | --- | --- |
| @pmndrs/bvh (tinybvh) | 508.9 k/s, 4096/4096 agree | 406.5 k/s, 4096/4096 agree | 264.1 k/s, 4096/4096 agree | 342.8 k/s, 4096/4096 agree |
| three-mesh-bvh (CENTER, default) | 300.9 k/s (0.59x ours), 4096/4096 agree | 279.4 k/s (0.69x ours), 4096/4096 agree, 1 ties | 183.1 k/s (0.69x ours), 4096/4096 agree, 64 ties | 274.2 k/s (0.80x ours), 4096/4096 agree, 38 ties |
| three-mesh-bvh (SAH) | 406.7 k/s (0.80x ours), 4096/4096 agree | 373.7 k/s (0.92x ours), 4096/4096 agree, 2 ties | 229.8 k/s (0.87x ours), 4096/4096 agree, 65 ties | 324.7 k/s (0.95x ours), 4096/4096 agree, 37 ties |
| bvh-tree | 206.3 k/s (0.41x ours), 4096/4096 agree | 265.7 k/s (0.65x ours), 4096/4096 agree, 1 ties | 196.0 k/s (0.74x ours), 4096/4096 agree, 110 ties | 281.9 k/s (0.82x ours), 4096/4096 agree, 53 ties |
| Babylon.js picking (submesh octree) | 5.6 k/s, **13/4096 disagree** | 7.2 k/s, **2/4096 disagree** | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) |
| Babylon.js picking (brute force) | 121/s, **1/198 disagree** | 125/s (0.00x ours), 149/149 agree | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) |
| Rapier (trimesh collider) | 520.4 k/s (1.02x ours), 4096/4096 agree, 4096 ties | 470.6 k/s (1.16x ours), 4096/4096 agree, 4010 ties | n/a (intersectionsWithRay gives one hit per collider, not per triangle) | n/a (intersectionsWithRay gives one hit per collider, not per triangle) |
| three Mesh.raycast (no BVH) | 69/s (0.00x ours), 109/109 agree | 73/s (0.00x ours), 74/74 agree | 67/s (0.00x ours), 105/105 agree, 2 ties | 71/s (0.00x ours), 101/101 agree, 1 ties |

### Queries (queries/s)

| Library | box | sphere | closest-point |
| --- | --- | --- | --- |
| bvh-tree | n/a (not available) | n/a (not available) | n/a (not available) |
| Babylon.js picking (submesh octree) | n/a (not available) | n/a (not available) | n/a (not available) |
| Babylon.js picking (brute force) | n/a (not available) | n/a (not available) | n/a (not available) |
| three Mesh.raycast (no BVH) | n/a (not available) | n/a (not available) | n/a (not available) |
| @pmndrs/bvh (tinybvh) | 204.3 k/s, 2048/2048 agree | 949.8 k/s, 2048/2048 agree | 132.1 k/s, 2048/2048 agree |
| three-mesh-bvh (CENTER, default) | 239.0 k/s (1.17x ours), 2048/2048 agree | 412.2 k/s (0.43x ours), 2048/2048 agree | 102.1 k/s (0.77x ours), 2048/2048 agree, 211 ties |
| three-mesh-bvh (SAH) | 246.7 k/s (1.21x ours), 2048/2048 agree | 506.0 k/s (0.53x ours), 2048/2048 agree | 186.8 k/s (1.41x ours), 2048/2048 agree, 213 ties |
| Rapier (trimesh collider) | 542.3 k/s (2.65x ours), 2048/2048 agree | 242.6 k/s (0.26x ours), 2048/2048 agree | 321.6 k/s (2.43x ours), 2048/2048 agree, 2048 ties |

Disagreements (first examples):
- Babylon.js picking (submesh octree) first/outside: ray 70: ours t=4.23747 (tri 256), theirs t=4.23186 (tri 253); ray 370: ours t=4.51405 (tri 288), theirs t=4.50751 (tri 290); ray 1469: ours t=5.17079 (tri 235), theirs t=5.16983 (tri 279)
- Babylon.js picking (brute force) first/outside: ray 70: ours t=4.23747 (tri 256), theirs t=4.23186 (tri 253)
- Babylon.js picking (submesh octree) first/inside: ray 2049: ours t=0.491506 (tri 201434), theirs t=0.490868 (tri 201399); ray 3297: ours t=0.294537 (tri 201177), theirs t=0.294499 (tri 201116)

## Scene-level trees: 10,000 objects (one 1k-triangle mesh each)

| Library | build | memory | update | frustum | box | raycast | frustum-after-motion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| @pmndrs/bvh (CpuSceneBvh) | 36.8 ms | 1.3 MiB | 32.7 ms | 8.4 k/s, 1024/1024 agree | 177.1 k/s, 1024/1024 agree | 199.5 k/s, 1024/1024 agree | 0, 1024/1024 agree |
| bvh.js (HybridBuilder) | 13.1 ms (0.36x ours) | n/a | 51.1 ms (1.56x ours) | 12.0 k/s, **1/1024 disagree** | 214.5 k/s (1.21x ours), 1024/1024 agree | n/a (rays meet boxes (rayIntersections), not triangles: an app tests the triangles itself) | 2, **2/1024 disagree** |
| three-mesh-bvh ObjectBVH | 62.7 ms (1.70x ours) | 672.9 KiB (0.50x ours) | 15.9 ms (0.49x ours) | 3.3 k/s (0.40x ours), 1024/1024 agree | 61.9 k/s (0.35x ours), 1024/1024 agree | 68.7 k/s (0.34x ours), 1024/1024 agree | 0, 1024/1024 agree, 1 ties |

## Scene-level trees: 100,000 objects (one 1k-triangle mesh each)

| Library | build | memory | update | frustum | box | raycast | frustum-after-motion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| @pmndrs/bvh (CpuSceneBvh) | 391 ms | 12.3 MiB | 396 ms | 960/s, 1024/1024 agree | 22.7 k/s, 1024/1024 agree | 85.5 k/s, 1024/1024 agree | 0, 1024/1024 agree |
| bvh.js (HybridBuilder) | 124 ms (0.32x ours) | n/a | 1134 ms (2.87x ours) | 854/s, **14/1024 disagree** | 16.7 k/s (0.73x ours), 1024/1024 agree | n/a (rays meet boxes (rayIntersections), not triangles: an app tests the triangles itself) | 8, **8/1024 disagree** |
| three-mesh-bvh ObjectBVH | 491 ms (1.26x ours) | 6.5 MiB (0.53x ours) | 176 ms (0.45x ours) | 262/s (0.27x ours), 350/350 agree, 3 ties | 5.7 k/s (0.25x ours), 1024/1024 agree | 33.3 k/s (0.39x ours), 1024/1024 agree | 0, 1024/1024 agree, 11 ties |

## sponza: 5,744,384 triangles, 3,108,999 vertices

### Build, refit and memory

| Library | build | refit | memory | heap | refit-agreement |
| --- | --- | --- | --- | --- | --- |
| @pmndrs/bvh (tinybvh) | 1499 ms | 1178 ms | 459.7 MiB | 1740.3 MiB |  |
| three-mesh-bvh (CENTER, default) | 1538 ms (1.03x ours) | 170 ms (0.14x ours) | 51.5 MiB (0.11x ours) | 47.1 MiB (0.03x ours) | 0, 512/512 agree |
| three-mesh-bvh (SAH) | 7011 ms (4.68x ours) | 177 ms (0.15x ours) | 50.6 MiB (0.11x ours) | 46.2 MiB (0.03x ours) | 0, 512/512 agree |
| bvh-tree | 11301 ms (7.54x ours) | n/a (not available) | 701.2 MiB (1.53x ours) | 1151.6 MiB (0.66x ours) |  |
| Babylon.js picking (submesh octree) | 421 ms (0.28x ours) | 56.6 ms (0.05x ours) | n/a | 363.0 MiB (0.21x ours) | 1, **1/512 disagree** |
| Babylon.js picking (brute force) | 420 ms (0.28x ours) | 6.68 ms (0.01x ours) | n/a | 338.2 MiB (0.19x ours) | 1, **1/512 disagree** |
| Rapier (trimesh collider) | 1329 ms (0.89x ours) | n/a (a trimesh is rebuilt (setShape), not refitted) | 0.0 KiB (0.00x ours) | 1140.8 MiB (0.66x ours) |  |
| three Mesh.raycast (no BVH) | 33.1 ms (0.02x ours) | n/a (not available) | 0.0 KiB (0.00x ours) | 39.3 KiB (0.00x ours) |  |
| @pmndrs/bvh H-PLOC (CPU twin) | 10699 ms (7.14x ours) |  | 458.8 MiB (1.00x ours) |  |  |
| @pmndrs/bvh LBVH (CPU twin) | 9170 ms (6.12x ours) |  | 494.3 MiB (1.08x ours) |  |  |

### Tree quality

| Library | sah | depth | nodes | leaves |
| --- | --- | --- | --- | --- |
| @pmndrs/bvh (tinybvh) | 90.2 | 35 | 6,067,561 | 3,033,781 |
| three-mesh-bvh (CENTER, default) | 140.6 (1.56x ours) | 35 | 1,687,521 | 843,761 |
| three-mesh-bvh (SAH) | 121.6 (1.35x ours) | 30 | 1,656,493 | 828,247 |
| bvh-tree | 204.6 (2.27x ours) | 48 | 1,752,923 | 876,462 |
| @pmndrs/bvh H-PLOC (CPU twin) | 88.4 (0.98x ours) | 46 | 6,040,519 | 3,020,260 |
| @pmndrs/bvh LBVH (CPU twin) | 112.6 (1.25x ours) | 38 | 7,133,901 | 3,566,951 |

### CPU raycasts (rays/s)

| Library | first/outside | first/inside | all/outside | all/inside |
| --- | --- | --- | --- | --- |
| @pmndrs/bvh (tinybvh) | 579.5 k/s, 4096/4096 agree | 478.5 k/s, 4096/4096 agree | 157.5 k/s, 4096/4096 agree | 175.4 k/s, 4096/4096 agree |
| three-mesh-bvh (CENTER, default) | 332.4 k/s (0.57x ours), 4096/4096 agree | 239.3 k/s (0.50x ours), 4096/4096 agree | 102.2 k/s (0.65x ours), 4096/4096 agree, 3 ties | 141.7 k/s (0.81x ours), 4096/4096 agree |
| three-mesh-bvh (SAH) | 436.5 k/s (0.75x ours), 4096/4096 agree | 300.6 k/s (0.63x ours), 4096/4096 agree | 115.2 k/s (0.73x ours), 4096/4096 agree, 3 ties | 156.2 k/s (0.89x ours), 4096/4096 agree |
| bvh-tree | 69.2 k/s (0.12x ours), 4096/4096 agree | 106.9 k/s (0.22x ours), 4096/4096 agree | 69.1 k/s (0.44x ours), 4096/4096 agree, 3 ties | 92.5 k/s (0.53x ours), 4096/4096 agree |
| Babylon.js picking (submesh octree) | 1.0 k/s, **1/1700 disagree** | 1.6 k/s, **3/2461 disagree** | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) |
| Babylon.js picking (brute force) | 9/s (0.00x ours), 16/16 agree | 9/s (0.00x ours), 16/16 agree | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) |
| Rapier (trimesh collider) | 550.4 k/s (0.95x ours), 4096/4096 agree, 3844 ties | 470.4 k/s (0.98x ours), 4096/4096 agree, 3820 ties | n/a (intersectionsWithRay gives one hit per collider, not per triangle) | n/a (intersectionsWithRay gives one hit per collider, not per triangle) |
| three Mesh.raycast (no BVH) | 4/s (0.00x ours), 16/16 agree | 5/s (0.00x ours), 16/16 agree | 5/s (0.00x ours), 16/16 agree | 4/s (0.00x ours), 16/16 agree |

### Queries (queries/s)

| Library | box | sphere | closest-point |
| --- | --- | --- | --- |
| bvh-tree | n/a (not available) | n/a (not available) | n/a (not available) |
| Babylon.js picking (submesh octree) | n/a (not available) | n/a (not available) | n/a (not available) |
| Babylon.js picking (brute force) | n/a (not available) | n/a (not available) | n/a (not available) |
| three Mesh.raycast (no BVH) | n/a (not available) | n/a (not available) | n/a (not available) |
| @pmndrs/bvh (tinybvh) | 318.3 k/s, 2048/2048 agree | 922.9 k/s, 2048/2048 agree | 165.3 k/s, 2048/2048 agree |
| three-mesh-bvh (CENTER, default) | 374.9 k/s (1.18x ours), 2048/2048 agree | 796.9 k/s (0.86x ours), 2048/2048 agree | 135.9 k/s (0.82x ours), 2048/2048 agree, 237 ties |
| three-mesh-bvh (SAH) | 458.6 k/s (1.44x ours), 2048/2048 agree | 925.9 k/s (1.00x ours), 2048/2048 agree | 175.0 k/s (1.06x ours), 2048/2048 agree, 245 ties |
| Rapier (trimesh collider) | 853.4 k/s (2.68x ours), 2048/2048 agree | 262.5 k/s (0.28x ours), 2048/2048 agree | 276.6 k/s (1.67x ours), 2048/2048 agree, 2048 ties |

Disagreements (first examples):
- Babylon.js picking (submesh octree) first/outside: ray 1545: ours t=35.1606 (tri 3312996), theirs t=32.2458 (tri 905379)
- Babylon.js picking (submesh octree) first/inside: ray 984: ours t=5.94529 (tri 1564627), theirs t=5.92652 (tri 2979321); ray 2315: ours t=0.269316 (tri 1161464), theirs t=0.269041 (tri 1160538); ray 2321: ours t=5.48092 (tri 905375), theirs t=0.792092 (tri 1564451)
- Babylon.js picking (submesh octree) refit-agreement: ray 114: ours t=44.2180 (tri 3662609), theirs t=44.1347 (tri 3746695)
- Babylon.js picking (brute force) refit-agreement: ray 114: ours t=44.2180 (tri 3662609), theirs t=44.1347 (tri 3746695)

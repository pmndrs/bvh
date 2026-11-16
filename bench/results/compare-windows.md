# compare:node, 2026-11-16, 69ae6cc-dirty, 11th Gen Intel(R) Core(TM) i9-11900H @ 2.50GHz, Node v24.13.1 (indicative)

## torus-knot: 100,000 triangles, 51,051 vertices

### Build, refit and memory

| Library | build | refit | memory | heap | heap-wasm | refit-agreement |
| --- | --- | --- | --- | --- | --- | --- |
| @pmndrs/bvh (tinybvh) | 45.6 ms | 8.32 ms | 8.0 MiB | 8.1 MiB | 0.0 KiB |  |
| @pmndrs/bvh (tinybvh, corners: 'positions') | 47.2 ms (1.04x ours) | 6.13 ms (0.74x ours) | 4.9 MiB (0.62x ours) | 5.1 MiB (0.63x ours) | 0.0 KiB | 0, 512/512 agree |
| three-mesh-bvh (CENTER, default) | 24.5 ms (0.54x ours) | 3.84 ms (0.46x ours) | 873.6 KiB (0.11x ours) | 935.5 KiB (0.11x ours) | 0.0 KiB | 0, 512/512 agree |
| three-mesh-bvh (SAH) | 118 ms (2.59x ours) | 4.05 ms (0.49x ours) | 873.3 KiB (0.11x ours) | 936.6 KiB (0.11x ours) | 0.0 KiB | 0, 512/512 agree |
| bvh-tree | 102 ms (2.24x ours) | n/a (not available) | 12.2 MiB (1.53x ours) | 19.6 MiB (2.40x ours) | 0.0 KiB |  |
| Babylon.js picking (submesh octree) | 6.32 ms (0.14x ours) | 1.97 ms (0.24x ours) | n/a | 6.6 MiB (0.82x ours) | 0.0 KiB | 0, 512/512 agree |
| Babylon.js picking (brute force) | 4.91 ms (0.11x ours) | 0.35 ms (0.04x ours) | n/a | 6.1 MiB (0.75x ours) | 0.0 KiB | 0, 512/512 agree |
| Rapier (trimesh collider) | 48.0 ms (1.05x ours) | n/a (a trimesh is rebuilt (setShape), not refitted) | n/a | 20.3 MiB (2.50x ours) | 20.3 MiB |  |
| three Mesh.raycast (no BVH) | 1.40 ms (0.03x ours) | n/a (not available) | 0.0 KiB (0.00x ours) | 43.3 KiB (0.01x ours) | 0.0 KiB |  |
| @pmndrs/bvh H-PLOC (CPU twin) | 257 ms (5.63x ours) |  | 7.9 MiB (0.99x ours) |  |  |  |
| @pmndrs/bvh LBVH (CPU twin) | 168 ms (3.68x ours) |  | 8.2 MiB (1.03x ours) |  |  |  |

### Tree quality

| Library | sah | depth | nodes | leaves |
| --- | --- | --- | --- | --- |
| @pmndrs/bvh (tinybvh) | 58.5 | 20 | 104,763 | 52,382 |
| @pmndrs/bvh (tinybvh, corners: 'positions') | 58.5 (1.00x ours) | 20 | 104,763 | 52,382 |
| three-mesh-bvh (CENTER, default) | 83.0 (1.42x ours) | 18 | 27,955 | 13,978 |
| three-mesh-bvh (SAH) | 76.0 (1.30x ours) | 16 | 27,947 | 13,974 |
| bvh-tree | 83.6 (1.43x ours) | 18 | 27,959 | 13,980 |
| @pmndrs/bvh H-PLOC (CPU twin) | 59.3 (1.01x ours) | 22 | 103,187 | 51,594 |
| @pmndrs/bvh LBVH (CPU twin) | 64.4 (1.10x ours) | 25 | 111,995 | 55,998 |

### CPU raycasts (rays/s)

| Library | first/outside | first/inside | all/outside | all/inside |
| --- | --- | --- | --- | --- |
| @pmndrs/bvh (tinybvh) | 236.7 k/s, 4096/4096 agree | 224.7 k/s, 4096/4096 agree | 120.6 k/s, 4096/4096 agree | 159.6 k/s, 4096/4096 agree |
| @pmndrs/bvh (tinybvh, corners: 'positions') | 232.0 k/s (0.98x ours), 4096/4096 agree | 223.3 k/s (0.99x ours), 4096/4096 agree | 127.6 k/s (1.06x ours), 4096/4096 agree | 163.4 k/s (1.02x ours), 4096/4096 agree |
| three-mesh-bvh (CENTER, default) | 318.0 k/s (1.34x ours), 4096/4096 agree | 291.7 k/s (1.30x ours), 4096/4096 agree | 155.3 k/s (1.29x ours), 4096/4096 agree | 224.7 k/s (1.41x ours), 4096/4096 agree |
| three-mesh-bvh (SAH) | 328.8 k/s (1.39x ours), 4096/4096 agree | 308.9 k/s (1.37x ours), 4096/4096 agree | 164.5 k/s (1.36x ours), 4096/4096 agree | 235.0 k/s (1.47x ours), 4096/4096 agree |
| bvh-tree | 132.2 k/s (0.56x ours), 4096/4096 agree | 189.1 k/s (0.84x ours), 4096/4096 agree | 131.5 k/s (1.09x ours), 4096/4096 agree | 188.8 k/s (1.18x ours), 4096/4096 agree |
| Babylon.js picking (submesh octree) | 6.3 k/s (0.03x ours), 4096/4096 agree, 11 ties | 8.3 k/s (0.04x ours), 4096/4096 agree, 9 ties | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) |
| Babylon.js picking (brute force) | 398/s (0.00x ours), 590/590 agree, 1 ties | 397/s (0.00x ours), 624/624 agree, 1 ties | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) |
| Rapier (trimesh collider) | 278.2 k/s (1.18x ours), 4096/4096 agree, 2929 ties | 262.7 k/s (1.17x ours), 4096/4096 agree, 2739 ties | n/a (intersectionsWithRay gives one hit per collider, not per triangle) | n/a (intersectionsWithRay gives one hit per collider, not per triangle) |
| three Mesh.raycast (no BVH) | 298/s (0.00x ours), 451/451 agree | 276/s (0.00x ours), 414/414 agree | 300/s (0.00x ours), 446/446 agree | 282/s (0.00x ours), 426/426 agree |

### Queries (queries/s)

| Library | box | sphere | closest-point |
| --- | --- | --- | --- |
| bvh-tree | n/a (not available) | n/a (not available) | n/a (not available) |
| Babylon.js picking (submesh octree) | n/a (not available) | n/a (not available) | n/a (not available) |
| Babylon.js picking (brute force) | n/a (not available) | n/a (not available) | n/a (not available) |
| three Mesh.raycast (no BVH) | n/a (not available) | n/a (not available) | n/a (not available) |
| @pmndrs/bvh (tinybvh) | 666.4 k/s, 2048/2048 agree | 519.9 k/s, 2048/2048 agree | 111.1 k/s, 2048/2048 agree |
| @pmndrs/bvh (tinybvh, corners: 'positions') | 690.8 k/s (1.04x ours), 2048/2048 agree | 508.1 k/s (0.98x ours), 2048/2048 agree | 110.6 k/s (1.00x ours), 2048/2048 agree |
| three-mesh-bvh (CENTER, default) | 272.7 k/s (0.41x ours), 2048/2048 agree | 529.7 k/s (1.02x ours), 2048/2048 agree | 64.2 k/s (0.58x ours), 2048/2048 agree, 468 ties |
| three-mesh-bvh (SAH) | 297.4 k/s (0.45x ours), 2048/2048 agree | 568.8 k/s (1.09x ours), 2048/2048 agree | 65.6 k/s (0.59x ours), 2048/2048 agree, 503 ties |
| Rapier (trimesh collider) | 490.2 k/s (0.74x ours), 2048/2048 agree | 108.9 k/s (0.21x ours), 2048/2048 agree | 111.9 k/s (1.01x ours), 2048/2048 agree, 2048 ties |

## furnished-room: 269,764 triangles, 160,380 vertices

### Build, refit and memory

| Library | build | refit | memory | heap | heap-wasm | refit-agreement |
| --- | --- | --- | --- | --- | --- | --- |
| @pmndrs/bvh (tinybvh) | 133 ms | 23.2 ms | 20.1 MiB | 20.2 MiB | 0.0 KiB |  |
| @pmndrs/bvh (tinybvh, corners: 'positions') | 122 ms (0.92x ours) | 18.4 ms (0.79x ours) | 11.8 MiB (0.59x ours) | 12.0 MiB (0.59x ours) | 0.0 KiB | 0, 512/512 agree |
| three-mesh-bvh (CENTER, default) | 79.4 ms (0.60x ours) | 10.7 ms (0.46x ours) | 2.4 MiB (0.12x ours) | 2.5 MiB (0.12x ours) | 0.0 KiB | 0, 512/512 agree |
| three-mesh-bvh (SAH) | 353 ms (2.65x ours) | 10.8 ms (0.46x ours) | 2.3 MiB (0.11x ours) | 2.3 MiB (0.11x ours) | 0.0 KiB | 0, 512/512 agree |
| bvh-tree | 424 ms (3.18x ours) | n/a (not available) | 32.9 MiB (1.64x ours) | 54.9 MiB (2.72x ours) | 0.0 KiB |  |
| Babylon.js picking (submesh octree) | 14.5 ms (0.11x ours) | 4.13 ms (0.18x ours) | n/a | 19.0 MiB (0.94x ours) | 0.0 KiB | 0, 512/512 agree, 1 ties |
| Babylon.js picking (brute force) | 14.3 ms (0.11x ours) | 0.60 ms (0.03x ours) | n/a | 17.6 MiB (0.87x ours) | 0.0 KiB | 0, 512/512 agree, 1 ties |
| Rapier (trimesh collider) | 156 ms (1.17x ours) | n/a (a trimesh is rebuilt (setShape), not refitted) | n/a | 54.5 MiB (2.70x ours) | 54.4 MiB |  |
| three Mesh.raycast (no BVH) | 2.74 ms (0.02x ours) | n/a (not available) | 0.0 KiB (0.00x ours) | 32.2 KiB (0.00x ours) | 0.0 KiB |  |
| @pmndrs/bvh H-PLOC (CPU twin) | 758 ms (5.69x ours) |  | 20.2 MiB (1.00x ours) |  |  |  |
| @pmndrs/bvh LBVH (CPU twin) | 432 ms (3.24x ours) |  | 20.9 MiB (1.04x ours) |  |  |  |

### Tree quality

| Library | sah | depth | nodes | leaves |
| --- | --- | --- | --- | --- |
| @pmndrs/bvh (tinybvh) | 40.2 | 29 | 238,383 | 119,192 |
| @pmndrs/bvh (tinybvh, corners: 'positions') | 40.2 (1.00x ours) | 29 | 238,383 | 119,192 |
| three-mesh-bvh (CENTER, default) | 62.2 (1.55x ours) | 30 | 79,077 | 39,539 |
| three-mesh-bvh (SAH) | 52.3 (1.30x ours) | 25 | 74,759 | 37,380 |
| bvh-tree | 64.3 (1.60x ours) | 42 | 84,509 | 42,255 |
| @pmndrs/bvh H-PLOC (CPU twin) | 39.3 (0.98x ours) | 37 | 241,251 | 120,626 |
| @pmndrs/bvh LBVH (CPU twin) | 48.5 (1.20x ours) | 37 | 264,817 | 132,409 |

### CPU raycasts (rays/s)

| Library | first/outside | first/inside | all/outside | all/inside |
| --- | --- | --- | --- | --- |
| @pmndrs/bvh (tinybvh) | 354.4 k/s, 4096/4096 agree | 370.0 k/s, 4096/4096 agree | 215.6 k/s, 4096/4096 agree | 312.9 k/s, 4096/4096 agree |
| @pmndrs/bvh (tinybvh, corners: 'positions') | 341.9 k/s (0.96x ours), 4096/4096 agree | 355.8 k/s (0.96x ours), 4096/4096 agree | 199.3 k/s (0.92x ours), 4096/4096 agree | 307.1 k/s (0.98x ours), 4096/4096 agree |
| three-mesh-bvh (CENTER, default) | 272.6 k/s (0.77x ours), 4096/4096 agree | 318.8 k/s (0.86x ours), 4096/4096 agree, 1 ties | 221.7 k/s (1.03x ours), 4096/4096 agree, 62 ties | 322.8 k/s (1.03x ours), 4096/4096 agree, 37 ties |
| three-mesh-bvh (SAH) | 403.5 k/s (1.14x ours), 4096/4096 agree | 437.4 k/s (1.18x ours), 4096/4096 agree, 2 ties | 265.1 k/s (1.23x ours), 4096/4096 agree, 63 ties | 394.2 k/s (1.26x ours), 4096/4096 agree, 36 ties |
| bvh-tree | 233.8 k/s (0.66x ours), 4096/4096 agree | 356.8 k/s (0.96x ours), 4096/4096 agree, 1 ties | 234.7 k/s (1.09x ours), 4096/4096 agree, 110 ties | 362.8 k/s (1.16x ours), 4096/4096 agree, 53 ties |
| Babylon.js picking (submesh octree) | 5.8 k/s, **13/4096 disagree** | 7.6 k/s, **2/4096 disagree** | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) |
| Babylon.js picking (brute force) | 128/s, **1/194 disagree** | 128/s (0.00x ours), 194/194 agree, 1 ties | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) |
| Rapier (trimesh collider) | 441.8 k/s (1.25x ours), 4096/4096 agree, 4096 ties | 405.7 k/s (1.10x ours), 4096/4096 agree, 4010 ties | n/a (intersectionsWithRay gives one hit per collider, not per triangle) | n/a (intersectionsWithRay gives one hit per collider, not per triangle) |
| three Mesh.raycast (no BVH) | 95/s (0.00x ours), 149/149 agree | 95/s (0.00x ours), 145/145 agree | 95/s (0.00x ours), 148/148 agree, 3 ties | 95/s (0.00x ours), 145/145 agree, 2 ties |

### Queries (queries/s)

| Library | box | sphere | closest-point |
| --- | --- | --- | --- |
| bvh-tree | n/a (not available) | n/a (not available) | n/a (not available) |
| Babylon.js picking (submesh octree) | n/a (not available) | n/a (not available) | n/a (not available) |
| Babylon.js picking (brute force) | n/a (not available) | n/a (not available) | n/a (not available) |
| three Mesh.raycast (no BVH) | n/a (not available) | n/a (not available) | n/a (not available) |
| @pmndrs/bvh (tinybvh) | 960.0 k/s, 2048/2048 agree | 986.5 k/s, 2048/2048 agree | 431.2 k/s, 2048/2048 agree |
| @pmndrs/bvh (tinybvh, corners: 'positions') | 989.3 k/s (1.03x ours), 2048/2048 agree | 948.1 k/s (0.96x ours), 2048/2048 agree | 393.2 k/s (0.91x ours), 2048/2048 agree |
| three-mesh-bvh (CENTER, default) | 262.1 k/s (0.27x ours), 2048/2048 agree | 536.2 k/s (0.54x ours), 2048/2048 agree | 131.3 k/s (0.30x ours), 2048/2048 agree, 211 ties |
| three-mesh-bvh (SAH) | 337.2 k/s (0.35x ours), 2048/2048 agree | 688.6 k/s (0.70x ours), 2048/2048 agree | 240.4 k/s (0.56x ours), 2048/2048 agree, 213 ties |
| Rapier (trimesh collider) | 592.0 k/s (0.62x ours), 2048/2048 agree | 236.9 k/s (0.24x ours), 2048/2048 agree | 314.5 k/s (0.73x ours), 2048/2048 agree, 2048 ties |

Disagreements (first examples):
- Babylon.js picking (submesh octree) first/outside: ray 70: ours t=4.23747 (tri 256), theirs t=4.23186 (tri 253); ray 370: ours t=4.51405 (tri 288), theirs t=4.50751 (tri 290); ray 1469: ours t=5.17079 (tri 235), theirs t=5.16983 (tri 279)
- Babylon.js picking (brute force) first/outside: ray 70: ours t=4.23747 (tri 256), theirs t=4.23186 (tri 253)
- Babylon.js picking (submesh octree) first/inside: ray 2049: ours t=0.491506 (tri 201434), theirs t=0.490868 (tri 201399); ray 3297: ours t=0.294537 (tri 201177), theirs t=0.294499 (tri 201116)

## sponza: 5,744,384 triangles, 3,108,999 vertices

### Build, refit and memory

| Library | build | refit | memory | heap | heap-wasm | refit-agreement |
| --- | --- | --- | --- | --- | --- | --- |
| @pmndrs/bvh (tinybvh) | 3161 ms | 595 ms | 459.7 MiB | 395.8 MiB | -65536.0 KiB |  |
| @pmndrs/bvh (tinybvh, corners: 'positions') | 3036 ms (0.96x ours) | 441 ms (0.74x ours) | 284.4 MiB (0.62x ours) | 220.1 MiB (0.56x ours) | -65536.0 KiB | 0, 512/512 agree |
| three-mesh-bvh (CENTER, default) | 2045 ms (0.65x ours) | 239 ms (0.40x ours) | 51.5 MiB (0.11x ours) | 51.2 MiB (0.13x ours) | 0.0 KiB | 0, 512/512 agree |
| three-mesh-bvh (SAH) | 8599 ms (2.72x ours) | 231 ms (0.39x ours) | 50.6 MiB (0.11x ours) | 50.6 MiB (0.13x ours) | 0.0 KiB | 0, 512/512 agree |
| bvh-tree | 15461 ms (4.89x ours) | n/a (not available) | 701.2 MiB (1.53x ours) | 1155.7 MiB (2.92x ours) | 0.0 KiB |  |
| Babylon.js picking (submesh octree) | 639 ms (0.20x ours) | 86.8 ms (0.15x ours) | n/a | 362.9 MiB (0.92x ours) | 0.0 KiB | 1, **1/512 disagree** |
| Babylon.js picking (brute force) | 626 ms (0.20x ours) | 10.3 ms (0.02x ours) | n/a | 338.2 MiB (0.85x ours) | 0.0 KiB | 1, **1/512 disagree** |
| Rapier (trimesh collider) | 3695 ms (1.17x ours) | n/a (a trimesh is rebuilt (setShape), not refitted) | n/a | 1140.8 MiB (2.88x ours) | 1145.2 MiB |  |
| three Mesh.raycast (no BVH) | 46.5 ms (0.01x ours) | n/a (not available) | 0.0 KiB (0.00x ours) | -7.2 KiB (-0.00x ours) | 0.0 KiB |  |
| @pmndrs/bvh H-PLOC (CPU twin) | 17586 ms (5.56x ours) |  | 458.8 MiB (1.00x ours) |  |  |  |
| @pmndrs/bvh LBVH (CPU twin) | 15009 ms (4.75x ours) |  | 494.3 MiB (1.08x ours) |  |  |  |

### Tree quality

| Library | sah | depth | nodes | leaves |
| --- | --- | --- | --- | --- |
| @pmndrs/bvh (tinybvh) | 90.2 | 35 | 6,067,561 | 3,033,781 |
| @pmndrs/bvh (tinybvh, corners: 'positions') | 90.2 (1.00x ours) | 35 | 6,067,561 | 3,033,781 |
| three-mesh-bvh (CENTER, default) | 140.6 (1.56x ours) | 35 | 1,687,521 | 843,761 |
| three-mesh-bvh (SAH) | 121.6 (1.35x ours) | 30 | 1,656,493 | 828,247 |
| bvh-tree | 204.6 (2.27x ours) | 48 | 1,752,923 | 876,462 |
| @pmndrs/bvh H-PLOC (CPU twin) | 88.4 (0.98x ours) | 46 | 6,040,519 | 3,020,260 |
| @pmndrs/bvh LBVH (CPU twin) | 112.6 (1.25x ours) | 38 | 7,133,901 | 3,566,951 |

### CPU raycasts (rays/s)

| Library | first/outside | first/inside | all/outside | all/inside |
| --- | --- | --- | --- | --- |
| @pmndrs/bvh (tinybvh) | 211.6 k/s, 4096/4096 agree | 174.9 k/s, 4096/4096 agree | 68.0 k/s, 4096/4096 agree | 94.8 k/s, 4096/4096 agree |
| @pmndrs/bvh (tinybvh, corners: 'positions') | 205.3 k/s (0.97x ours), 4096/4096 agree | 170.6 k/s (0.98x ours), 4096/4096 agree | 67.7 k/s (1.00x ours), 4096/4096 agree | 93.9 k/s (0.99x ours), 4096/4096 agree |
| three-mesh-bvh (CENTER, default) | 196.7 k/s (0.93x ours), 4096/4096 agree | 140.6 k/s (0.80x ours), 4096/4096 agree | 66.9 k/s (0.98x ours), 4096/4096 agree, 3 ties | 98.5 k/s (1.04x ours), 4096/4096 agree |
| three-mesh-bvh (SAH) | 249.2 k/s (1.18x ours), 4096/4096 agree | 172.4 k/s (0.99x ours), 4096/4096 agree | 76.4 k/s (1.12x ours), 4096/4096 agree, 3 ties | 112.5 k/s (1.19x ours), 4096/4096 agree |
| bvh-tree | 38.0 k/s (0.18x ours), 4096/4096 agree | 56.8 k/s (0.32x ours), 4096/4096 agree | 40.4 k/s (0.59x ours), 4096/4096 agree, 3 ties | 60.8 k/s (0.64x ours), 4096/4096 agree |
| Babylon.js picking (submesh octree) | 586/s (0.00x ours), 921/921 agree, 4 ties | 786/s, **1/1186 disagree** | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) |
| Babylon.js picking (brute force) | 5/s (0.00x ours), 16/16 agree | 5/s (0.00x ours), 16/16 agree | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) |
| Rapier (trimesh collider) | 241.8 k/s (1.14x ours), 4096/4096 agree, 3844 ties | 197.7 k/s (1.13x ours), 4096/4096 agree, 3820 ties | n/a (intersectionsWithRay gives one hit per collider, not per triangle) | n/a (intersectionsWithRay gives one hit per collider, not per triangle) |
| three Mesh.raycast (no BVH) | 4/s (0.00x ours), 16/16 agree | 4/s (0.00x ours), 16/16 agree | 4/s (0.00x ours), 16/16 agree | 4/s (0.00x ours), 16/16 agree |

### Queries (queries/s)

| Library | box | sphere | closest-point |
| --- | --- | --- | --- |
| bvh-tree | n/a (not available) | n/a (not available) | n/a (not available) |
| Babylon.js picking (submesh octree) | n/a (not available) | n/a (not available) | n/a (not available) |
| Babylon.js picking (brute force) | n/a (not available) | n/a (not available) | n/a (not available) |
| three Mesh.raycast (no BVH) | n/a (not available) | n/a (not available) | n/a (not available) |
| @pmndrs/bvh (tinybvh) | 578.3 k/s, 2048/2048 agree | 517.9 k/s, 2048/2048 agree | 232.2 k/s, 2048/2048 agree |
| @pmndrs/bvh (tinybvh, corners: 'positions') | 571.2 k/s (0.99x ours), 2048/2048 agree | 477.6 k/s (0.92x ours), 2048/2048 agree | 232.1 k/s (1.00x ours), 2048/2048 agree |
| three-mesh-bvh (CENTER, default) | 206.9 k/s (0.36x ours), 2048/2048 agree | 371.4 k/s (0.72x ours), 2048/2048 agree | 90.8 k/s (0.39x ours), 2048/2048 agree, 237 ties |
| three-mesh-bvh (SAH) | 237.5 k/s (0.41x ours), 2048/2048 agree | 471.7 k/s (0.91x ours), 2048/2048 agree | 123.6 k/s (0.53x ours), 2048/2048 agree, 245 ties |
| Rapier (trimesh collider) | 452.5 k/s (0.78x ours), 2048/2048 agree | 130.4 k/s (0.25x ours), 2048/2048 agree | 156.2 k/s (0.67x ours), 2048/2048 agree, 2048 ties |

Disagreements (first examples):
- Babylon.js picking (submesh octree) first/inside: ray 984: ours t=5.94529 (tri 1564627), theirs t=5.92652 (tri 2979321)
- Babylon.js picking (submesh octree) refit-agreement: ray 114: ours t=44.2180 (tri 3662609), theirs t=44.1347 (tri 3746695)
- Babylon.js picking (brute force) refit-agreement: ray 114: ours t=44.2180 (tri 3662609), theirs t=44.1347 (tri 3746695)

## Scene-level trees: 10,000 objects (one 1k-triangle mesh each)

| Library | build | memory | update | frustum | box | raycast | frustum-after-motion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| @pmndrs/bvh (CpuSceneBvh) | 36.7 ms | 1.3 MiB | 32.6 ms | 8.2 k/s, 1024/1024 agree | 243.5 k/s, 1024/1024 agree | 138.1 k/s, 1024/1024 agree | 0, 1024/1024 agree |
| bvh.js (HybridBuilder) | 9.55 ms (0.26x ours) | n/a | 34.2 ms (1.05x ours) | 18.5 k/s, **1/1024 disagree** | 363.5 k/s (1.49x ours), 1024/1024 agree | n/a (rays meet boxes (rayIntersections), not triangles: an app tests the triangles itself) | 2, **2/1024 disagree** |
| three-mesh-bvh ObjectBVH | 39.4 ms (1.07x ours) | 672.9 KiB (0.50x ours) | 14.1 ms (0.43x ours) | 6.3 k/s (0.77x ours), 1024/1024 agree, 1 ties | 131.2 k/s (0.54x ours), 1024/1024 agree | 82.6 k/s (0.60x ours), 1024/1024 agree | 0, 1024/1024 agree, 2 ties |

## Scene-level trees: 100,000 objects (one 1k-triangle mesh each)

| Library | build | memory | update | frustum | box | raycast | frustum-after-motion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| @pmndrs/bvh (CpuSceneBvh) | 414 ms | 12.3 MiB | 412 ms | 1.1 k/s, 1024/1024 agree | 38.4 k/s, 1024/1024 agree | 74.8 k/s, 1024/1024 agree | 0, 1024/1024 agree |
| bvh.js (HybridBuilder) | 150 ms (0.36x ours) | n/a | 992 ms (2.40x ours) | 976/s, **14/1024 disagree** | 19.3 k/s (0.50x ours), 1024/1024 agree | n/a (rays meet boxes (rayIntersections), not triangles: an app tests the triangles itself) | 8, **8/1024 disagree** |
| three-mesh-bvh ObjectBVH | 479 ms (1.16x ours) | 6.5 MiB (0.53x ours) | 173 ms (0.42x ours) | 313/s (0.30x ours), 457/457 agree, 14 ties | 7.1 k/s (0.19x ours), 1024/1024 agree | 42.1 k/s (0.56x ours), 1024/1024 agree | 0, 1024/1024 agree, 23 ties |

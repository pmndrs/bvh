# compare:node, 2026-11-13, ad229e5, Apple M5 Pro, Node v22.22.2 (indicative)

## torus-knot: 100,000 triangles, 51,051 vertices

### Build, refit and memory

| Library | build | refit | memory | heap | heap-wasm | refit-agreement |
| --- | --- | --- | --- | --- | --- | --- |
| @pmndrs/bvh (tinybvh) | 31.6 ms | 3.18 ms | 8.0 MiB | 8.2 MiB | 0.0 KiB |  |
| three-mesh-bvh (CENTER, default) | 22.3 ms (0.71x ours) | 3.11 ms (0.98x ours) | 873.6 KiB (0.11x ours) | 961.2 KiB (0.12x ours) | 0.0 KiB | 0, 512/512 agree |
| three-mesh-bvh (SAH) | 109 ms (3.47x ours) | 3.06 ms (0.96x ours) | 873.3 KiB (0.11x ours) | 992.1 KiB (0.12x ours) | 0.0 KiB | 0, 512/512 agree |
| bvh-tree | 89.6 ms (2.84x ours) | n/a (not available) | 12.2 MiB (1.53x ours) | 19.6 MiB (2.40x ours) | 0.0 KiB |  |
| Babylon.js picking (submesh octree) | 8.83 ms (0.28x ours) | 1.44 ms (0.45x ours) | n/a | 6.6 MiB (0.81x ours) | 0.0 KiB | 0, 512/512 agree |
| Babylon.js picking (brute force) | 5.81 ms (0.18x ours) | 0.20 ms (0.06x ours) | n/a | 6.1 MiB (0.75x ours) | 0.0 KiB | 0, 512/512 agree |
| Rapier (trimesh collider) | 25.1 ms (0.79x ours) | n/a (a trimesh is rebuilt (setShape), not refitted) | n/a | 20.3 MiB (2.49x ours) | 20.3 MiB |  |
| three Mesh.raycast (no BVH) | 0.82 ms (0.03x ours) | n/a (not available) | 0.0 KiB (0.00x ours) | 47.2 KiB (0.01x ours) | 0.0 KiB |  |
| @pmndrs/bvh H-PLOC (CPU twin) | 193 ms (6.13x ours) |  | 7.9 MiB (0.99x ours) |  |  |  |
| @pmndrs/bvh LBVH (CPU twin) | 95.2 ms (3.02x ours) |  | 8.2 MiB (1.03x ours) |  |  |  |

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
| @pmndrs/bvh (tinybvh) | 394.8 k/s, 4096/4096 agree | 364.8 k/s, 4096/4096 agree | 199.8 k/s, 4096/4096 agree | 303.9 k/s, 4096/4096 agree |
| three-mesh-bvh (CENTER, default) | 314.1 k/s (0.80x ours), 4096/4096 agree | 287.7 k/s (0.79x ours), 4096/4096 agree | 154.1 k/s (0.77x ours), 4096/4096 agree | 258.7 k/s (0.85x ours), 4096/4096 agree |
| three-mesh-bvh (SAH) | 353.5 k/s (0.90x ours), 4096/4096 agree | 290.3 k/s (0.80x ours), 4096/4096 agree | 156.5 k/s (0.78x ours), 4096/4096 agree | 276.5 k/s (0.91x ours), 4096/4096 agree |
| bvh-tree | 118.1 k/s (0.30x ours), 4096/4096 agree | 179.5 k/s (0.49x ours), 4096/4096 agree | 127.5 k/s (0.64x ours), 4096/4096 agree | 223.1 k/s (0.73x ours), 4096/4096 agree |
| Babylon.js picking (submesh octree) | 7.4 k/s (0.02x ours), 4096/4096 agree, 11 ties | 9.1 k/s (0.03x ours), 4096/4096 agree, 9 ties | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) |
| Babylon.js picking (brute force) | 456/s (0.00x ours), 662/662 agree, 1 ties | 527/s (0.00x ours), 797/797 agree, 1 ties | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) |
| Rapier (trimesh collider) | 397.6 k/s (1.01x ours), 4096/4096 agree, 2929 ties | 449.3 k/s (1.23x ours), 4096/4096 agree, 2739 ties | n/a (intersectionsWithRay gives one hit per collider, not per triangle) | n/a (intersectionsWithRay gives one hit per collider, not per triangle) |
| three Mesh.raycast (no BVH) | 262/s (0.00x ours), 432/432 agree | 285/s (0.00x ours), 456/456 agree | 311/s (0.00x ours), 383/383 agree | 329/s (0.00x ours), 475/475 agree |

### Queries (queries/s)

| Library | box | sphere | closest-point |
| --- | --- | --- | --- |
| bvh-tree | n/a (not available) | n/a (not available) | n/a (not available) |
| Babylon.js picking (submesh octree) | n/a (not available) | n/a (not available) | n/a (not available) |
| Babylon.js picking (brute force) | n/a (not available) | n/a (not available) | n/a (not available) |
| three Mesh.raycast (no BVH) | n/a (not available) | n/a (not available) | n/a (not available) |
| @pmndrs/bvh (tinybvh) | 1.09 M/s, 2048/2048 agree | 827.9 k/s, 2048/2048 agree | 172.8 k/s, 2048/2048 agree |
| three-mesh-bvh (CENTER, default) | 443.5 k/s (0.41x ours), 2048/2048 agree | 779.3 k/s (0.94x ours), 2048/2048 agree | 88.3 k/s (0.51x ours), 2048/2048 agree, 468 ties |
| three-mesh-bvh (SAH) | 457.3 k/s (0.42x ours), 2048/2048 agree | 873.1 k/s (1.05x ours), 2048/2048 agree | 92.5 k/s (0.54x ours), 2048/2048 agree, 503 ties |
| Rapier (trimesh collider) | 904.3 k/s (0.83x ours), 2048/2048 agree | 148.9 k/s (0.18x ours), 2048/2048 agree | 171.2 k/s (0.99x ours), 2048/2048 agree, 2048 ties |

## furnished-room: 269,764 triangles, 160,380 vertices

### Build, refit and memory

| Library | build | refit | memory | heap | heap-wasm | refit-agreement |
| --- | --- | --- | --- | --- | --- | --- |
| @pmndrs/bvh (tinybvh) | 68.9 ms | 11.5 ms | 20.1 MiB | 20.3 MiB | 0.0 KiB |  |
| three-mesh-bvh (CENTER, default) | 60.3 ms (0.87x ours) | 11.5 ms (0.99x ours) | 2.4 MiB (0.12x ours) | 2.5 MiB (0.12x ours) | 0.0 KiB | 0, 512/512 agree |
| three-mesh-bvh (SAH) | 261 ms (3.78x ours) | 10.3 ms (0.89x ours) | 2.3 MiB (0.11x ours) | 2.4 MiB (0.12x ours) | 0.0 KiB | 0, 512/512 agree |
| bvh-tree | 257 ms (3.72x ours) | n/a (not available) | 32.9 MiB (1.64x ours) | 54.9 MiB (2.71x ours) | 0.0 KiB |  |
| Babylon.js picking (submesh octree) | 16.9 ms (0.24x ours) | 3.88 ms (0.34x ours) | n/a | 19.0 MiB (0.94x ours) | 0.0 KiB | 0, 512/512 agree, 1 ties |
| Babylon.js picking (brute force) | 13.2 ms (0.19x ours) | 0.43 ms (0.04x ours) | n/a | 17.7 MiB (0.87x ours) | 0.0 KiB | 0, 512/512 agree, 1 ties |
| Rapier (trimesh collider) | 64.5 ms (0.94x ours) | n/a (a trimesh is rebuilt (setShape), not refitted) | n/a | 54.5 MiB (2.69x ours) | 54.4 MiB |  |
| three Mesh.raycast (no BVH) | 1.71 ms (0.02x ours) | n/a (not available) | 0.0 KiB (0.00x ours) | 46.6 KiB (0.00x ours) | 0.0 KiB |  |
| @pmndrs/bvh H-PLOC (CPU twin) | 685 ms (9.94x ours) |  | 20.2 MiB (1.00x ours) |  |  |  |
| @pmndrs/bvh LBVH (CPU twin) | 276 ms (4.00x ours) |  | 20.9 MiB (1.04x ours) |  |  |  |

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
| @pmndrs/bvh (tinybvh) | 710.5 k/s, 4096/4096 agree | 510.5 k/s, 4096/4096 agree | 389.8 k/s, 4096/4096 agree | 401.0 k/s, 4096/4096 agree |
| three-mesh-bvh (CENTER, default) | 397.9 k/s (0.56x ours), 4096/4096 agree | 327.4 k/s (0.64x ours), 4096/4096 agree, 1 ties | 297.4 k/s (0.76x ours), 4096/4096 agree, 64 ties | 335.1 k/s (0.84x ours), 4096/4096 agree, 38 ties |
| three-mesh-bvh (SAH) | 554.9 k/s (0.78x ours), 4096/4096 agree | 437.5 k/s (0.86x ours), 4096/4096 agree, 2 ties | 344.6 k/s (0.88x ours), 4096/4096 agree, 65 ties | 441.0 k/s (1.10x ours), 4096/4096 agree, 37 ties |
| bvh-tree | 332.4 k/s (0.47x ours), 4096/4096 agree | 366.2 k/s (0.72x ours), 4096/4096 agree, 1 ties | 321.8 k/s (0.83x ours), 4096/4096 agree, 110 ties | 348.5 k/s (0.87x ours), 4096/4096 agree, 53 ties |
| Babylon.js picking (submesh octree) | 9.0 k/s, **13/4096 disagree** | 8.5 k/s, **2/4096 disagree** | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) |
| Babylon.js picking (brute force) | 189/s, **1/295 disagree** | 137/s (0.00x ours), 88/88 agree | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) |
| Rapier (trimesh collider) | 878.8 k/s (1.24x ours), 4096/4096 agree, 4096 ties | 577.3 k/s (1.13x ours), 4096/4096 agree, 4010 ties | n/a (intersectionsWithRay gives one hit per collider, not per triangle) | n/a (intersectionsWithRay gives one hit per collider, not per triangle) |
| three Mesh.raycast (no BVH) | 98/s (0.00x ours), 161/161 agree | 78/s (0.00x ours), 42/42 agree | 107/s (0.00x ours), 143/143 agree, 3 ties | 78/s (0.00x ours), 132/132 agree, 2 ties |

### Queries (queries/s)

| Library | box | sphere | closest-point |
| --- | --- | --- | --- |
| bvh-tree | n/a (not available) | n/a (not available) | n/a (not available) |
| Babylon.js picking (submesh octree) | n/a (not available) | n/a (not available) | n/a (not available) |
| Babylon.js picking (brute force) | n/a (not available) | n/a (not available) | n/a (not available) |
| three Mesh.raycast (no BVH) | n/a (not available) | n/a (not available) | n/a (not available) |
| @pmndrs/bvh (tinybvh) | 1.08 M/s, 2048/2048 agree | 1.09 M/s, 2048/2048 agree | 596.8 k/s, 2048/2048 agree |
| three-mesh-bvh (CENTER, default) | 284.9 k/s (0.26x ours), 2048/2048 agree | 519.5 k/s (0.48x ours), 2048/2048 agree | 138.0 k/s (0.23x ours), 2048/2048 agree, 211 ties |
| three-mesh-bvh (SAH) | 386.4 k/s (0.36x ours), 2048/2048 agree | 708.0 k/s (0.65x ours), 2048/2048 agree | 250.8 k/s (0.42x ours), 2048/2048 agree, 213 ties |
| Rapier (trimesh collider) | 720.8 k/s (0.67x ours), 2048/2048 agree | 354.6 k/s (0.33x ours), 2048/2048 agree | 416.8 k/s (0.70x ours), 2048/2048 agree, 2048 ties |

Disagreements (first examples):
- Babylon.js picking (submesh octree) first/outside: ray 70: ours t=4.23747 (tri 256), theirs t=4.23186 (tri 253); ray 370: ours t=4.51405 (tri 288), theirs t=4.50751 (tri 290); ray 1469: ours t=5.17079 (tri 235), theirs t=5.16983 (tri 279)
- Babylon.js picking (brute force) first/outside: ray 70: ours t=4.23747 (tri 256), theirs t=4.23186 (tri 253)
- Babylon.js picking (submesh octree) first/inside: ray 2049: ours t=0.491506 (tri 201434), theirs t=0.490868 (tri 201399); ray 3297: ours t=0.294537 (tri 201177), theirs t=0.294499 (tri 201116)

## sponza: 5,744,384 triangles, 3,108,999 vertices

### Build, refit and memory

| Library | build | refit | memory | heap | heap-wasm | refit-agreement |
| --- | --- | --- | --- | --- | --- | --- |
| @pmndrs/bvh (tinybvh) | 1922 ms | 364 ms | 459.7 MiB | 395.5 MiB | -65536.0 KiB |  |
| three-mesh-bvh (CENTER, default) | 1859 ms (0.97x ours) | 248 ms (0.68x ours) | 51.5 MiB (0.11x ours) | 47.2 MiB (0.12x ours) | 0.0 KiB | 0, 512/512 agree |
| three-mesh-bvh (SAH) | 7433 ms (3.87x ours) | 241 ms (0.66x ours) | 50.6 MiB (0.11x ours) | 46.2 MiB (0.12x ours) | 0.0 KiB | 0, 512/512 agree |
| bvh-tree | 15610 ms (8.12x ours) | n/a (not available) | 701.2 MiB (1.53x ours) | 1151.6 MiB (2.91x ours) | 0.0 KiB |  |
| Babylon.js picking (submesh octree) | 606 ms (0.32x ours) | 86.2 ms (0.24x ours) | n/a | 363.0 MiB (0.92x ours) | 0.0 KiB | 1, **1/512 disagree** |
| Babylon.js picking (brute force) | 651 ms (0.34x ours) | 8.24 ms (0.02x ours) | n/a | 338.2 MiB (0.86x ours) | 0.0 KiB | 1, **1/512 disagree** |
| Rapier (trimesh collider) | 1789 ms (0.93x ours) | n/a (a trimesh is rebuilt (setShape), not refitted) | n/a | 1140.6 MiB (2.88x ours) | 1145.2 MiB |  |
| three Mesh.raycast (no BVH) | 32.3 ms (0.02x ours) | n/a (not available) | 0.0 KiB (0.00x ours) | 37.6 KiB (0.00x ours) | 0.0 KiB |  |
| @pmndrs/bvh H-PLOC (CPU twin) | 15522 ms (8.07x ours) |  | 458.8 MiB (1.00x ours) |  |  |  |
| @pmndrs/bvh LBVH (CPU twin) | 13029 ms (6.78x ours) |  | 494.3 MiB (1.08x ours) |  |  |  |

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
| @pmndrs/bvh (tinybvh) | 367.7 k/s, 4096/4096 agree | 129.7 k/s, 4096/4096 agree | 87.3 k/s, 4096/4096 agree | 52.4 k/s, 4096/4096 agree |
| three-mesh-bvh (CENTER, default) | 231.1 k/s (0.63x ours), 4096/4096 agree | 115.9 k/s (0.89x ours), 4096/4096 agree | 63.4 k/s (0.73x ours), 4096/4096 agree, 3 ties | 64.9 k/s (1.24x ours), 4096/4096 agree |
| three-mesh-bvh (SAH) | 316.3 k/s (0.86x ours), 4096/4096 agree | 139.5 k/s (1.08x ours), 4096/4096 agree | 76.2 k/s (0.87x ours), 4096/4096 agree, 3 ties | 72.0 k/s (1.37x ours), 4096/4096 agree |
| bvh-tree | 44.0 k/s (0.12x ours), 4096/4096 agree | 41.1 k/s (0.32x ours), 4096/4096 agree | 43.5 k/s (0.50x ours), 4096/4096 agree, 3 ties | 39.7 k/s (0.76x ours), 4096/4096 agree |
| Babylon.js picking (submesh octree) | 676/s (0.00x ours), 997/997 agree, 5 ties | 876/s, **1/1168 disagree** | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) |
| Babylon.js picking (brute force) | 6/s (0.00x ours), 16/16 agree | 6/s (0.00x ours), 16/16 agree | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) | n/a (one hit per mesh (multiPickWithRay); no all-hits pick on a mesh) |
| Rapier (trimesh collider) | 447.0 k/s (1.22x ours), 4096/4096 agree, 3844 ties | 203.6 k/s (1.57x ours), 4096/4096 agree, 3820 ties | n/a (intersectionsWithRay gives one hit per collider, not per triangle) | n/a (intersectionsWithRay gives one hit per collider, not per triangle) |
| three Mesh.raycast (no BVH) | 4/s (0.00x ours), 16/16 agree | 4/s (0.00x ours), 16/16 agree | 4/s (0.00x ours), 16/16 agree | 4/s (0.00x ours), 16/16 agree |

### Queries (queries/s)

| Library | box | sphere | closest-point |
| --- | --- | --- | --- |
| bvh-tree | n/a (not available) | n/a (not available) | n/a (not available) |
| Babylon.js picking (submesh octree) | n/a (not available) | n/a (not available) | n/a (not available) |
| Babylon.js picking (brute force) | n/a (not available) | n/a (not available) | n/a (not available) |
| three Mesh.raycast (no BVH) | n/a (not available) | n/a (not available) | n/a (not available) |
| @pmndrs/bvh (tinybvh) | 730.5 k/s, 2048/2048 agree | 455.8 k/s, 2048/2048 agree | 311.7 k/s, 2048/2048 agree |
| three-mesh-bvh (CENTER, default) | 243.2 k/s (0.33x ours), 2048/2048 agree | 461.5 k/s (1.01x ours), 2048/2048 agree | 86.6 k/s (0.28x ours), 2048/2048 agree, 237 ties |
| three-mesh-bvh (SAH) | 301.2 k/s (0.41x ours), 2048/2048 agree | 565.8 k/s (1.24x ours), 2048/2048 agree | 114.4 k/s (0.37x ours), 2048/2048 agree, 245 ties |
| Rapier (trimesh collider) | 411.1 k/s (0.56x ours), 2048/2048 agree | 170.4 k/s (0.37x ours), 2048/2048 agree | 198.5 k/s (0.64x ours), 2048/2048 agree, 2048 ties |

Disagreements (first examples):
- Babylon.js picking (submesh octree) first/inside: ray 984: ours t=5.94529 (tri 1564627), theirs t=5.92652 (tri 2979321)
- Babylon.js picking (submesh octree) refit-agreement: ray 114: ours t=44.2180 (tri 3662609), theirs t=44.1347 (tri 3746695)
- Babylon.js picking (brute force) refit-agreement: ray 114: ours t=44.2180 (tri 3662609), theirs t=44.1347 (tri 3746695)

## Scene-level trees: 10,000 objects (one 1k-triangle mesh each)

| Library | build | memory | update | frustum | box | raycast | frustum-after-motion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| @pmndrs/bvh (CpuSceneBvh) | 31.9 ms | 1.3 MiB | 26.1 ms | 10.7 k/s, 1024/1024 agree | 302.8 k/s, 1024/1024 agree | 286.6 k/s, 1024/1024 agree | 0, 1024/1024 agree |
| bvh.js (HybridBuilder) | 18.0 ms (0.56x ours) | n/a | 33.7 ms (1.29x ours) | 18.5 k/s, **1/1024 disagree** | 356.6 k/s (1.18x ours), 1024/1024 agree | n/a (rays meet boxes (rayIntersections), not triangles: an app tests the triangles itself) | 2, **2/1024 disagree** |
| three-mesh-bvh ObjectBVH | 34.3 ms (1.07x ours) | 672.9 KiB (0.50x ours) | 10.4 ms (0.40x ours) | 5.8 k/s (0.54x ours), 1024/1024 agree | 97.2 k/s (0.32x ours), 1024/1024 agree | 100.7 k/s (0.35x ours), 1024/1024 agree | 0, 1024/1024 agree, 1 ties |

## Scene-level trees: 100,000 objects (one 1k-triangle mesh each)

| Library | build | memory | update | frustum | box | raycast | frustum-after-motion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| @pmndrs/bvh (CpuSceneBvh) | 328 ms | 12.3 MiB | 302 ms | 1.4 k/s, 1024/1024 agree | 42.5 k/s, 1024/1024 agree | 126.0 k/s, 1024/1024 agree | 0, 1024/1024 agree |
| bvh.js (HybridBuilder) | 97.7 ms (0.30x ours) | n/a | 779 ms (2.58x ours) | 1.3 k/s, **14/1024 disagree** | 29.1 k/s (0.68x ours), 1024/1024 agree | n/a (rays meet boxes (rayIntersections), not triangles: an app tests the triangles itself) | 8, **8/1024 disagree** |
| three-mesh-bvh ObjectBVH | 309 ms (0.94x ours) | 6.5 MiB (0.53x ours) | 116 ms (0.38x ours) | 296/s (0.22x ours), 535/535 agree, 7 ties | 6.7 k/s (0.16x ours), 1024/1024 agree | 44.2 k/s (0.35x ours), 1024/1024 agree | 0, 1024/1024 agree, 11 ties |

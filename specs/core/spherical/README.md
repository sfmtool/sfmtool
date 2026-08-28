# Spherical Specifications

Spherical tiling and panorama compositing. Implemented in
`crates/sfmtool-core/src/spherical/`.

| Document | Description |
|----------|-------------|
| [spherical-tiles-rig.md](spherical-tiles-rig.md) | Discretizing the sphere as a rig of pinhole tiles, and resampling an atlas back out through it. |
| [per-spherical-tile-source-stack.md](per-spherical-tile-source-stack.md) | The per-tile stack of source patches that compositing consumes. |
| [photometric-subsets-ransac.md](photometric-subsets-ransac.md) | Per-tile RANSAC subset partition for photometric refinement. |
| [tile-batched-consensus-atlas.md](tile-batched-consensus-atlas.md) | Bounded-memory panorama compositing, orchestrating the three above one tile batch at a time. |

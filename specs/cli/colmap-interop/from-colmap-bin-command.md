# `sfm from-colmap-bin` Command

## Overview

Imports a COLMAP binary reconstruction into `.sfmr` format. Reads COLMAP's `cameras.bin`,
`images.bin`, and `points3D.bin` files — plus `rigs.bin` and `frames.bin` when the solve
wrote them — and converts them to a single `.sfmr` file. `--image-dir` must sit inside an
initialized workspace (`sfm ws init`); the command fails up front otherwise, because the
workspace is what resolves image names and their `.sift` files.

## Coordinate Convention

This command is a convention boundary: COLMAP binary files use COLMAP's
+Z-forward, Y-down convention, while `.sfmr` data is stored in the
canonical Z-up / −Z-forward convention. On import the COLMAP→canonical
conversion is applied — the camera-frame flip `S` on every pose and the
world canonicalization `W` on world-space world points — so poses and
points are never copied verbatim. Rig sensor poses, being camera-frame
relative poses, are conjugated by `S` instead. COLMAP stores every point
as a finite `(x, y, z)`, so there are no infinity directions to rotate
at this boundary; `--detect-infinity` runs after the conversion and
reclassifies from the converted geometry. `sfm
to-colmap-bin` applies the inverse, so an import/export round trip is
stable. See the "Coordinate System Conventions" section of
[`sfmr-file-format.md`](../../formats/sfmr-file-format.md) for the transform
definitions.

## Command Syntax

```bash
sfm from-colmap-bin <COLMAP_DIR> --image-dir <DIR> --output <OUTPUT.sfmr> [OPTIONS...]
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--image-dir` | path | required | Directory containing images (for workspace and `.sift` resolution) |
| `--output / -o` | path | required | Output `.sfmr` file path. A path whose extension is not `.sfmr` (compared case-insensitively) is a usage error, reported before anything is read. |
| `--tool-name` | string | `unknown` | Tool provenance tag (e.g., `colmap`, `glomap`) |
| `--detect-infinity / --no-detect-infinity` | bool | `true` | Reclassify ill-conditioned points (depth the solve could not pin down) as points at infinity (`w = 0`). See [sfmr file format §7](../../formats/sfmr-file-format.md). |

## Input Directory

The `COLMAP_DIR` should contain COLMAP's standard binary output (typically a numbered
subdirectory like `colmap_output/0/`):

```
colmap_output/0/
  cameras.bin
  images.bin
  points3D.bin
  rigs.bin        # optional
  frames.bin      # optional
```

`rigs.bin` and `frames.bin` are read when present, so a rig solve keeps its rig
structure through the import: non-camera sensors are dropped, camera and image IDs are
remapped to the `.sfmr`'s 0-based indexes, and the rig/frame tables are carried into the
output file.

## Point errors

Each point stores the RMS of its observations' reprojection errors, measured
from the `.sift` keypoints, the imported cameras and the converted poses
([sfmr-file-format.md](../../formats/sfmr-file-format.md#points3dreprojection_errorsnfloat32zst)).
The error column of `points3D.bin` is not read into it: COLMAP's per-point error
is the mean of the observation errors, and GLOMAP's is in normalised image
coordinates. `colmap_binary_to_rust_sfmr` passes zeros and
`SfmrReconstruction.from_data` scores every point. `sfm solve` writes its
models through the same path, and through `pycolmap_to_rust_sfmr` for a rig
solve, so a solve's output stores the RMS too.

## Repeat observations

A GLOMAP model can hold a point whose track has several observations in one
image. The import keeps, in each such image, the observation with the lowest
reprojection error against the point and drops the others, then scores the
point's RMS again over what it keeps; this runs before `--detect-infinity`.
When it drops anything it prints `Dropped N repeat observation(s) from M
point(s)`. The rule, its tie-break and its handling of points at infinity are
in [drop-repeat-observations.md](../../core/reconstruction/drop-repeat-observations.md).
`sfm solve` writes its models through the same import, so a solve's output has
no repeat observations either.

## Usage Examples

```bash
# Import a COLMAP reconstruction
sfm from-colmap-bin colmap_output/0/ --image-dir images/ -o imported.sfmr --tool-name colmap

# Import a GLOMAP result
sfm from-colmap-bin glomap_output/0/ --image-dir images/ -o imported.sfmr --tool-name glomap
```

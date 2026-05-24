# `sfm inspect` Command

## Overview

Inspects a single sfmtool file or image and prints a summary. The file type is
determined by extension; one summary printer handles each type. Without
`--verbose`, prints a compact label/value block; with `--verbose`, prints the
full detail available for that type.

For deep-analysis reports on a `.sfmr` reconstruction (covisibility graph,
frustum intersection, depth ranges, per-image metrics), use `sfm analyze`
(see `specs/cli/analyze-command.md`).

## Command Syntax

```bash
sfm inspect <PATH> [--verbose / -v]
```

`PATH` is a single file. Directories and multiple paths are not accepted.

## Supported File Types

| Extension | Type | Inspected with |
|-----------|------|----------------|
| `.sfmr` | Reconstruction | `read_sfmr_metadata` / `verify_sfmr` (default), full load (verbose) |
| `.sift` | Feature file | `read_sift_metadata` / `verify_sift` |
| `.matches` | Feature matches | `read_matches_metadata` / `verify_matches` (default), `read_matches` (verbose) |
| `.camrig` | Camera rig | `read_camrig_metadata` / `verify_camrig` (default), `read_camrig` (verbose) |
| `.png` `.jpg` `.jpeg` | Image | Image dimensions; pycolmap EXIF inference (verbose) |

An unsupported extension is rejected with a message listing the supported
types.

## Integrity

For the four sfmtool formats, the matching `verify_*` function runs and the
result appears as an `Integrity:` line (`OK` or `FAILED`). When verification
fails, the error messages are printed and the command exits non-zero. Image
files have no integrity check.

## Default Summary

The default output is a compact label/value block. The fields per type:

- **`.sfmr`** — format version, operation (tool + version), image / camera /
  3D point / observation counts, rig counts (if present), integrity. When the
  reconstruction holds any points at infinity (`w == 0`), the 3D point count is
  annotated with how many are at infinity (e.g. `206,413  (105,773 at
  infinity)`), read from the format's stored `infinity_point_count`.
- **`.sift`** — image name and dimensions, feature count, feature tool,
  integrity.
- **`.matches`** — format version, matching method (tool + version), image /
  image-pair / match counts, whether two-view geometries are present,
  integrity.
- **`.camrig`** — format version, name, rig type, sensor / camera counts,
  integrity.
- **image** — format, dimensions, file size.

## Verbose Output (`--verbose` / `-v`)

- **`.sfmr`** — the full reconstruction report: metadata, workspace, per-camera
  parameter tables, rig configuration, 3D point statistics with histograms,
  reprojection error, observation statistics, nearest-neighbor distances. The
  3D point count carries the same `(N at infinity)` annotation as the default
  summary when any points at infinity are present.
- **`.sift`** — adds image file size and hashes, feature tool and content
  hashes, feature tool options, and the top 5 features by size.
- **`.matches`** — adds timestamp, workspace, matching options, matches-per-pair
  and descriptor-distance histograms, and two-view-geometry inlier statistics.
- **`.camrig`** — adds `rig_attributes`, content hash, and per-camera details.
- **image** — adds the pycolmap EXIF-inferred camera (model, dimensions,
  focal length).

## Usage Examples

```bash
# Compact summary of a reconstruction
sfm inspect sfmr/solve_001.sfmr

# Full reconstruction report
sfm inspect sfmr/solve_001.sfmr --verbose

# Inspect a feature file
sfm inspect image_001.sift -v

# Inspect a matches file
sfm inspect matches/exhaustive.matches

# Inspect a camera rig
sfm inspect tiles.camrig

# Inspect an image
sfm inspect photo.jpg -v
```

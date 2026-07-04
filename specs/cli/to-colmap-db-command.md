# `sfm to-colmap-db` Command

## Overview

Creates a COLMAP database from a `.sfmr` reconstruction or `.matches` file. Useful for
re-running COLMAP operations on existing data or for interoperability with COLMAP tools.

The `sfm` CLI uses this general approach to implement many of its operations, converting the
workspace data into a COLMAP .db, running the COLMAP operation, then converting back to `.sfmr`.

## Coordinate Convention

This command is a convention boundary: `.sfmr` and `.matches` data are
stored in the canonical Z-up / −Z-forward convention, while a COLMAP
database holds COLMAP-convention (+Z-forward, Y-down) data. On export the
canonical→COLMAP conversion is applied: relative and rig poses are
S-conjugated into COLMAP camera frames, and world-space data such as pose
priors gets the inverse world canonicalization `W⁻¹` — so a solver run
against the database, whose output is imported back through the forward
conversion (see `solve-command.md`), round-trips stably. Pixel-space
two-view geometry (fundamental/essential matrices, keypoints, matches) is
invariant under the camera-frame flip and is written unchanged. See the
"Coordinate System Conventions" section of
[`sfmr-file-format.md`](../formats/sfmr-file-format.md) for the transform
definitions.

## Command Syntax

```bash
sfm to-colmap-db <INPUT_PATH> <DATABASE.db> [OPTIONS...]
```

`INPUT_PATH` can be a `.sfmr` or `.matches` file. The output database path is a
required positional argument (matching the positional `OUTPUT` convention of
`to-colmap-bin` and `to-nerfstudio`).

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--max-features` | int | all | Maximum features per image (`.sfmr` only) |
| `--no-guided-matching` | flag | | Disable two-view geometry pre-population (`.sfmr` only) |
| `--camera-model` | choice | auto | Camera model override (`.matches` only); one of the 11 shared COLMAP model names (same vocabulary as `solve` / `match` / `camrig create`) |

## Input Modes

### From `.sfmr`

Populates the database with camera intrinsics, pose priors, keypoints, descriptors, and
(by default) two-view geometries with fundamental matrices for guided matching.

### From `.matches`

Populates the database with cameras, keypoints, descriptors, and pre-computed matches and
two-view geometries (if included) from the matches file.

## Camera Intrinsics (`.matches` mode only)

If any image referenced by the `.matches` file resolves a `camera_config.json` (closest-ancestor
walk from its parent directory up to the workspace root), the file's intrinsics are used and
`--camera-model` is rejected with an error before any database work begins. See
[`../workspace/camera-config.md`](../workspace/camera-config.md).

## Usage Examples

```bash
# Create database from reconstruction
sfm to-colmap-db sfmr/solve_001.sfmr colmap/database.db

# From matches, with camera model override
sfm to-colmap-db matches/match_001.matches colmap/database.db --camera-model OPENCV

# Without guided matching
sfm to-colmap-db solve.sfmr database.db --no-guided-matching
```

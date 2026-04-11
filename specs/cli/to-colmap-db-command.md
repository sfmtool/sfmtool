# `sfm to-colmap-db` Command

## Overview

Creates a COLMAP database from a `.sfmr` reconstruction or `.matches` file. Useful for
re-running COLMAP operations on existing data or for interoperability with COLMAP tools.

The `sfm` CLI uses this general approach to implement many of its operations, converting the
workspace data into a COLMAP .db, running the COLMAP operation, then converting back to `.sfmr`.

## Command Syntax

```bash
sfm to-colmap-db <INPUT_PATH> --out-db <DATABASE.db> [OPTIONS...]
```

`INPUT_PATH` can be a `.sfmr` or `.matches` file.

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--out-db` | path | required | Output database file path |
| `--max-features` | int | all | Maximum features per image (`.sfmr` only) |
| `--no-guided-matching` | flag | | Disable two-view geometry pre-population (`.sfmr` only) |
| `--camera-model` | string | auto | Camera model override (`.matches` only) |

## Input Modes

### From `.sfmr`

Populates the database with camera intrinsics, pose priors, keypoints, descriptors, and
(by default) two-view geometries with fundamental matrices for guided matching.

### From `.matches`

Populates the database with cameras, keypoints, descriptors, and pre-computed matches and
two-view geometries (if included) from the matches file.

## Usage Examples

```bash
# Create database from reconstruction
sfm to-colmap-db sfmr/solve_001.sfmr --out-db colmap/database.db

# From matches, with camera model override
sfm to-colmap-db matches/match_001.matches --out-db colmap/database.db --camera-model OPENCV

# Without guided matching
sfm to-colmap-db solve.sfmr --out-db database.db --no-guided-matching
```

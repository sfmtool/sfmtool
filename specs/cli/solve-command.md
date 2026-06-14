# `sfm solve` Command

## Overview

Runs Structure from Motion to produce a 3D reconstruction (`.sfmr` file) from images or
pre-computed matches. Uses pycolmap to run either COLMAP's incremental mapper
or GLOMAP's global mapper.

## Command Syntax

```bash
sfm solve [PATHS...] --incremental | --global [OPTIONS...]
```

`PATHS` can be image files/directories (matching is run internally) or a single `.matches`
file (pre-computed matches are used directly).

## Solver Selection

Exactly one is required:

| Flag | Description |
|------|-------------|
| `--incremental / -i` | COLMAP incremental SfM (sequential image registration) |
| `--global / -g` | GLOMAP global SfM (simultaneous pose estimation) |

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--seed / -s` | int | | Random seed for reproducibility |
| `--output / -o` | path | auto | Output `.sfmr` file path (default: timestamped in `sfmr/`) |
| `--colmap-dir` | path | auto | Directory for COLMAP database and intermediates |
| `--sfmr-dir` | path | `sfmr/` | Directory for `.sfmr` output files |
| `--seq-overlap` | string | | Sequential overlap mode: `WINDOW,OVERLAP` (e.g., `100,20`) |
| `--refine-rig / --no-refine-rig` | bool | `true` | Refine sensor-from-rig poses in bundle adjustment |
| `--flow-match` | flag | | Use optical flow matching instead of exhaustive |
| `--flow-preset` | `fast` \| `default` \| `high_quality` | `default` | Flow quality preset |
| `--flow-skip` | int | 5 | Sliding window size for flow matching |
| `--max-features` | int | | Maximum features per image |
| `--camera-model` | choice | auto | Camera model override; one of the 10 supported COLMAP model names (`SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`, `OPENCV_FISHEYE`, `SIMPLE_RADIAL_FISHEYE`, `RADIAL_FISHEYE`, `THIN_PRISM_FISHEYE`, `RAD_TAN_THIN_PRISM_FISHEYE`) |
| `--range / -r` | string | | Range expression for file numbers |
| `--detect-infinity / --no-detect-infinity` | bool | `true` | Reclassify points whose depth the solve could not pin down as points at infinity (`w = 0`). See [sfmr file format §7](../formats/sfmr-file-format.md). |

## Input Modes

### From images (matching + solving)

When `PATHS` are image files or directories, the command extracts features (if needed), runs
matching, and then solves. The matching strategy is exhaustive by default, or flow-based with
`--flow-match`.

### From `.matches` file

When a single `.matches` file is provided, pre-computed matches are loaded directly into the
COLMAP database, skipping feature extraction and matching.

### Sequential overlap mode

`--seq-overlap WINDOW,OVERLAP` reconstructs the sequence in overlapping windows of size
`WINDOW` with `OVERLAP` shared images, then aligns and merges the sub-reconstructions. Useful
for long sequences that fail with a single solve.

`--seq-overlap` cannot be combined with `--output` (each window writes its own
automatically-named output) or with a `.matches` input file (the windows drive
their own feature matching).

## Outputs and Multiple Models

The mapper can return more than one reconstruction (it splits a model whenever
it cannot register every image into a single one). `sfm solve` writes one
`.sfmr` per surviving model:

- **Degenerate models are skipped.** A model the mapper abandoned with zero
  3-D points is dropped with a warning rather than aborting the run — a single
  junk fragment must never discard the good models that share the output loop.
  Only when *no* model survives does the command raise
  `No 3D points found in reconstruction.`. A split is also a hint the run may be
  worth re-seeding, so it is surfaced loudly rather than swallowed silently.
- **The largest surviving model is primary.** Surviving models are ordered by
  registered image count (ties broken by observation count), and that ordering
  — not the mapper's internal model index — decides output naming: with an
  explicit `--output`, the largest model takes that exact path and the rest get
  `{stem}-{N}{suffix}` siblings; without it, each model is auto-named from the
  images it actually contains. This keeps a small fragment that happened to land
  at a lower model index from claiming the requested output name.

## Rig Support

If the workspace contains a rig configuration, the command automatically sets up rig
constraints. Use `--refine-rig / --no-refine-rig` to control whether sensor-from-rig poses
are refined during bundle adjustment.

## Camera Intrinsics

If any image being processed resolves a `camera_config.json` (closest-ancestor walk from
its parent directory up to the workspace root), the file's intrinsics are used and
`--camera-model` is rejected with an error before any solve work begins. See
[`../workspace/camera-config.md`](../workspace/camera-config.md).

## Camera Rig Files (`.camrig`)

`sfm solve` auto-discovers `.camrig` files in the workspace and uses the one
covering every image being solved. A `.camrig` takes precedence over
`camera_config.json` and `rig_config.json` for the images it covers — a note is
printed when it overrides one.

- A **single-sensor** `.camrig` (such as one written by `sfm camrig create`)
  supplies the camera intrinsics prior for the images its stored image pattern
  matches.
- A **multi-sensor** `.camrig` (such as one written by `sfm insv2rig`) drives
  rig-aware SfM. Each sensor's image pattern carries a frame field, and images
  from different sensors that share a frame index form one rig frame. The
  command builds one COLMAP camera per sensor (intrinsics from the rig's camera
  pool, scaled to the actual image resolution) and one rig whose reference is
  the lowest-indexed sensor present — its `cam_from_rig` is the identity and
  the others are rebased relative to it. Same-frame image pairs are excluded
  from matching, and `--refine-rig / --no-refine-rig` controls whether the
  `sensor_from_rig` poses are refined during bundle adjustment.

The command fails up front, with a message naming the offending files, when the
discovered `.camrig` files cannot be used:

- the images being solved span more than one `.camrig`;
- a `.camrig` covers only some of the images being solved;
- `--camera-model` is given alongside a matching `.camrig`.

See [`../formats/camrig-file-format.md`](../formats/camrig-file-format.md).

## Usage Examples

```bash
# Incremental SfM from images
sfm solve --incremental

# Global SfM from pre-computed matches
sfm solve matches/2026-01-15_match_001.matches --global

# Flow-based matching for video, with seed
sfm solve --incremental --flow-match --flow-preset high_quality --seed 42

# Sequential overlap for long sequences
sfm solve --global --seq-overlap 100,20

# Solve a subset of images
sfm solve --incremental --range 1:200 --max-features 8192
```

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
| `--camera-model` | string | auto | Camera model override |
| `--range / -r` | string | | Range expression for file numbers |

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

## Rig Support

If the workspace contains a rig configuration, the command automatically sets up rig
constraints. Use `--refine-rig / --no-refine-rig` to control whether sensor-from-rig poses
are refined during bundle adjustment.

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

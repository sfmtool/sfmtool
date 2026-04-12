# `sfm match` Command

## Overview

Matches SIFT features between image pairs and writes a `.matches` file. Requires a workspace
with previously extracted SIFT features. Uses COLMAP to perform the matching, except for the
experimental "flow" mode, which has some promise for videos.

## Command Syntax

```bash
sfm match [PATHS...] --exhaustive | --sequential | --flow [OPTIONS...]
sfm match --merge FILE1.matches FILE2.matches ... -o OUTPUT.matches
```

`PATHS` are image directories or files. Exactly one matching method must be specified,
or `--merge` to combine existing `.matches` files.

## Matching Methods

| Method | Description |
|--------|-------------|
| `--exhaustive / -e` | Match every pair of images against every other |
| `--sequential / -s` | Match each image against its nearby neighbors in sequence order |
| `--flow` | Use dense optical flow to guide feature matching |
| `--merge` | Merge multiple `.matches` files into one |

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--sequential-overlap` | int | 10 | Number of overlapping neighbors for sequential matching |
| `--flow-preset` | `fast` \| `default` \| `high_quality` | `default` | Optical flow quality preset |
| `--flow-skip` | int | 5 | Sliding window size for flow matching |
| `--max-features` | int | | Maximum features per image |
| `--output / -o` | path | auto | Output `.matches` file path (default: timestamped, required for `--merge`) |
| `--range / -r` | string | | Range expression for file numbers |
| `--camera-model` | string | auto | Camera model override (e.g., `SIMPLE_RADIAL`, `OPENCV`) |

## Process

1. Loads workspace config and SIFT features
2. Populates a COLMAP database with images, cameras, keypoints, and descriptors
3. Runs the selected matching strategy
4. Computes descriptor distances for matched pairs
5. Writes a timestamped `.matches` file

## Merge

`--merge` combines multiple `.matches` files into a single file. This is useful
for combining results from different matching strategies (e.g., sequential + exhaustive)
before running a solve.

The merge process:
- Builds a unified image list from all input files
- Validates that images with the same name have identical SIFT content hashes
- Concatenates matches for each pair across all input files
- Deduplicates matches with the same feature index pair (keeps lowest descriptor distance)
- Preserves two-view geometry (TVG) data when present in any input file
  - For pairs with TVG in multiple inputs, keeps the TVG with the most inliers
  - Handles index remapping by transforming F/E/H matrices and inverting poses as needed
- Records source file names and methods in the output metadata

## Usage Examples

```bash
# Exhaustive matching (small datasets)
sfm match --exhaustive

# Sequential matching for ordered image sequences
sfm match --sequential --sequential-overlap 20

# Flow-based matching for video frames
sfm match --flow --flow-preset high_quality --flow-skip 10

# Match a subset of images
sfm match --exhaustive --range 1:100 --max-features 4096

# Merge matches from different strategies
sfm match --merge seq.matches exhaustive.matches -o combined.matches
```

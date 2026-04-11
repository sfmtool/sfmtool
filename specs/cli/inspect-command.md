# `sfm inspect` Command

## Overview

Inspects a `.sfmr` reconstruction, providing summary statistics, per-image analysis,
covisibility graphs, depth estimation, and quality metrics. Multiple inspection modes are
available; the default prints a summary.

## Command Syntax

```bash
sfm inspect <RECONSTRUCTION.sfmr> [OPTIONS...]
```

## Inspection Modes

Without any mode flags, prints a default summary (camera count, image count, point count,
observation statistics, track length distribution).

| Mode | Description |
|------|-------------|
| `--images` | Per-image table: observation count, feature usage, track statistics |
| `--coviz` | Covisibility graph: pairs of images sharing 3D points |
| `--frustum` | Frustum intersection pairs: images whose viewing volumes overlap |
| `--z-range` | Estimate near/far depth planes from point cloud histogram |
| `--metrics` | Per-image quality metrics: reprojection error, track length (see below) |

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--range / -r` | string | | Range expression for file numbers |
| `--near-percentile` | float | 1.0 | Near depth percentile (with `--z-range`) |
| `--far-percentile` | float | 99.0 | Far depth percentile (with `--z-range`) |
| `--samples` | int | 10 | Number of sample images for depth estimation (with `--z-range`) |

## Per-Image Quality Metrics (`--metrics`)

Computes per-observation reprojection errors by projecting each observed 3D point through
the camera model (including distortion) and comparing against the feature's pixel position
from the `.sift` file.

### Metrics per image

| Metric | Column | Description |
|--------|--------|-------------|
| Observation count | Obs | Number of track observations |
| Mean reprojection error | MeanErr | Mean per-observation error |
| Median reprojection error | MedErr | Median error (robust to outliers) |
| Max reprojection error | MaxErr | Worst single observation error |
| Mean track length | MeanTL | Mean observation count of observed 3D points |

### Output

Images are sorted by mean reprojection error (descending). Outlier flags:

- `!!` — mean error > 2× reconstruction median
- `!` — mean error > 1.5× reconstruction median
- `--` — zero observations (registered but contributing no points)

### Edge Cases

- **Zero-observation images**: Show 0 observations, N/A for error metrics.
- **Zero-point reconstructions**: Print a message and return.

## Usage Examples

```bash
# Default summary
sfm inspect sfmr/solve_001.sfmr

# Per-image details
sfm inspect solve.sfmr --images

# Quality metrics to find problematic images
sfm inspect solve.sfmr --metrics

# Covisibility analysis
sfm inspect solve.sfmr --coviz

# Depth estimation for rendering
sfm inspect solve.sfmr --z-range --near-percentile 2 --far-percentile 98

# Inspect a subset
sfm inspect solve.sfmr --images --range 1:50
```

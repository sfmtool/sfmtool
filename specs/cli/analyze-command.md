# `sfm analyze` Command

## Overview

Runs a deep-analysis report on a `.sfmr` reconstruction: covisibility graphs,
frustum intersection, depth-range estimation, and per-image quality metrics.
Exactly one analysis mode must be selected per invocation.

For a quick summary of any sfmtool file, use `sfm inspect`
(see `specs/cli/inspect-command.md`).

## Command Syntax

```bash
sfm analyze <RECONSTRUCTION.sfmr> (--coviz | --z-range | --frustum | --images | --metrics) [OPTIONS...]
```

`RECONSTRUCTION` must be a `.sfmr` file.

## Analysis Modes

Exactly one is required:

| Mode | Description |
|------|-------------|
| `--coviz` | Covisibility graph: pairs of images sharing 3D points |
| `--z-range` | Per-image Z depth ranges and histograms from stored depth statistics |
| `--frustum` | Frustum intersection graph: images whose viewing volumes overlap |
| `--images` | Per-image connectivity table: observations, distances, closest images, graph metrics |
| `--metrics` | Per-image quality metrics: reprojection error, track length (see below) |

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--range / -r` | string | | Range expression for file numbers (only with `--metrics`) |
| `--near-percentile` | float | 5.0 | Near depth percentile (only with `--frustum`) |
| `--far-percentile` | float | 95.0 | Far depth percentile (only with `--frustum`) |
| `--samples` | int | 100 | Monte Carlo samples per frustum pair (only with `--frustum`) |

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
# Covisibility analysis
sfm analyze solve.sfmr --coviz

# Per-image connectivity details
sfm analyze solve.sfmr --images

# Quality metrics to find problematic images
sfm analyze solve.sfmr --metrics

# Quality metrics for a subset
sfm analyze solve.sfmr --metrics --range 1-50

# Depth estimation for rendering
sfm analyze solve.sfmr --frustum --near-percentile 2 --far-percentile 98
```

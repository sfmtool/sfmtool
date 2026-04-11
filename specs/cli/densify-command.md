# `sfm densify` Command

## Overview

Densifies a 3D point cloud by finding new feature correspondences via sweep matching between
covisible image pairs. Optionally discovers additional pairs through frustum intersection
analysis.

NOTE: This command is experimental, you may find it creates many false matches, and the default
settings are not well-tuned. I think it makes sense to somehow merge into the 'sfm xform' as
a subcommand, where you can pipeline it with point filtering and bundle adjustment in a more generic way.

## Command Syntax

```bash
sfm densify <INPUT.sfmr> <OUTPUT.sfmr> [OPTIONS...]
```

## Options

### Matching

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--max-features` | int | all | Maximum features per image (uses largest features) |
| `--sweep-window-size` | int | 30 | Window size for sort-and-sweep matching |
| `--distance-threshold` | float | | Maximum descriptor distance for matches |

### Pair Selection

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--close-pair-threshold` | int | 4 | Max image distance for "close" pairs |
| `--max-close-pairs` | int | | Maximum close pairs to keep |
| `--max-distant-pairs` | int | 5000 | Maximum distant pairs to keep |
| `--distant-pair-search-multiplier` | int | 3 | Search multiplier for distant pair candidates |
| `--frustum` | flag | | Find frustum intersection pairs (slower, finds pairs without shared points) |

### Bundle Adjustment

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--ba-refine-focal-length` | flag | | Refine focal length |
| `--ba-refine-principal-point` | flag | | Refine principal point |
| `--ba-refine-extra-params` | flag | | Refine extra camera parameters |

### Point Filtering

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--filter-max-reproj-error` | float | 4.0 | Max reprojection error (pixels) |
| `--filter-min-track-length` | int | 3 | Minimum observations per point |
| `--filter-min-tri-angle` | float | 1.5 | Minimum triangulation angle (degrees) |
| `--filter-isolated-median-ratio` | float | 2.0 | Isolated point filter ratio (0 = disable) |

### Geometric Filtering

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--enable-geometric-filtering` | flag | | Enable motion-invariant geometric filtering |
| `--geometric-size-ratio-max` | float | 1.25 | Max feature size ratio |
| `--geometric-angle-diff-max` | float | 15.0 | Max feature angle difference (degrees) |

## Process

1. **Prune image pairs** — Select covisibility pairs (images already sharing 3D points).
   With `--frustum`, also find pairs whose camera frustums intersect but share no points.
2. **Match pairs** — Run sort-and-sweep matching on each pair to find new correspondences.
3. **Triangulate** — Triangulate new tracks from matched features.
4. **Bundle adjust** — Refine the combined reconstruction.
5. **Filter** — Remove points failing reprojection error, track length, triangulation angle,
   or isolation criteria.
6. **Align** — Align the result back to the original coordinate frame.

## Usage Examples

```bash
# Basic densification
sfm densify solve_001.sfmr densified.sfmr

# With frustum pairs and geometric filtering
sfm densify solve_001.sfmr densified.sfmr --frustum --enable-geometric-filtering

# Conservative settings
sfm densify solve_001.sfmr densified.sfmr \
  --filter-max-reproj-error 2.0 \
  --filter-min-track-length 4 \
  --filter-min-tri-angle 3.0

# Limit features for speed
sfm densify solve_001.sfmr densified.sfmr --max-features 4096 --sweep-window-size 50
```

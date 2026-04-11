# `sfm heatmap` Command

## Overview

Visualizes reconstruction quality metrics as colored overlays on images. Each observed feature
is drawn as a colored circle, with color encoding the metric value.

## Command Syntax

```bash
sfm heatmap <RECONSTRUCTION.sfmr> --output <DIR> [OPTIONS...]
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output / -o` | path | required | Output directory for heatmap images |
| `--metric` | `reproj` \| `tracks` \| `angle` \| `all` | `all` | Metric to visualize |
| `--colormap` | `viridis` \| `plasma` \| `jet` \| `coolwarm` \| `error` \| `tracks` | auto | Colormap (default depends on metric) |
| `--radius` | int | 5 | Radius of feature circles (pixels) |
| `--alpha` | float | 0.7 | Opacity of overlay (0.0–1.0) |

## Metrics

| Metric | Description | Default Colormap |
|--------|-------------|-----------------|
| `reproj` | Reprojection error in pixels | `error` |
| `tracks` | Track length (number of observations) | `tracks` |
| `angle` | Triangulation angle in degrees | `viridis` |
| `all` | Generate all three metrics | per-metric defaults |

Each output image includes a colorbar legend showing the value range.

## Output

Images are saved as `{image_stem}_{metric}.png` (e.g., `image_001_reproj.png`). When
`--metric all` is used, three images are generated per input image.

## Usage Examples

```bash
# Generate all heatmaps
sfm heatmap solve.sfmr -o heatmaps/

# Reprojection error only with custom colormap
sfm heatmap solve.sfmr -o heatmaps/ --metric reproj --colormap plasma

# Larger markers, more transparent
sfm heatmap solve.sfmr -o heatmaps/ --radius 8 --alpha 0.5
```

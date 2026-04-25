# `sfm pano2rig` Command

## Overview

Converts equirectangular (360°) panorama images into a 6-face cubemap rig suitable for SfM
processing. Each panorama produces 6 perspective images (front, right, back, left, top,
bottom) with 90° field of view, plus a rig configuration file.

NOTE: This command hasn't been validated across a variety of datasets. For real 360
captures from a dual-fisheye rig, it's likely better to reconstruct from the actual
fisheye images with the dual rig and a known distance between them.

## Command Syntax

```bash
sfm pano2rig <INPUT_DIR> --output <OUTPUT_DIR> [OPTIONS...]
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output / -o` | path | required | Output directory (must be inside a workspace) |
| `--face-size` | int | pano_width / 4 | Face image size in pixels |
| `--jpeg-quality` | int | 95 | JPEG quality for output (1–100) |

## Output Structure

For each panorama, 6 face images are written:

```
output_dir/
  pano_001_front.jpg
  pano_001_right.jpg
  pano_001_back.jpg
  pano_001_left.jpg
  pano_001_top.jpg
  pano_001_bottom.jpg
  ...
  rig_config.json
```

The rig configuration file defines the 6-camera rig geometry (identity position, cubemap
rotations) for use by `sfm solve`.

## Constraints

The output directory must be inside an initialized workspace (`sfm ws init`).

## Usage Examples

```bash
# Convert panoramas with default face size
sfm pano2rig ./panoramas --output ./faces

# Custom face size and quality
sfm pano2rig ./panoramas --output ./faces --face-size 1024 --jpeg-quality 90
```

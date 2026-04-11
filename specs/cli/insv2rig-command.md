# `sfm insv2rig` Command

## Overview

Extracts dual-fisheye frames from Insta360 `.insv` video files. Each video frame produces
two fisheye images (one per lens) plus a rig configuration describing the Insta360 X5
camera geometry.

The distance between the two sensors in the rig was determined by capturing a ruler with
an Insta360 X5 from nearby, and tweaking so the reconstruction scale was approximately
correct in meters. This means if part of your capture is close enough to subjects that
some parallax is comparable to the inter-camera distance, there's a chance to
get a reconstruction scaled for meters.

## Command Syntax

```bash
sfm insv2rig <INPUT_FILE> --output <OUTPUT_DIR>
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output / -o` | path | required | Output directory (must be inside a workspace) |

## Rig Geometry

The Insta360 X5 rig is modeled as dual back-to-back fisheye cameras:

- **Left lens**: Forward-facing (identity rotation)
- **Right lens**: Rotated 180° around Y axis
- **Baseline**: 29 mm between optical centers

## Process

1. Probes the `.insv` file with `ffprobe` to detect stream layout (dual-stream or
   side-by-side)
2. Extracts frames with `ffmpeg`
3. Splits each frame into left and right fisheye images
4. Writes rig configuration file

## Constraints

- Requires `ffmpeg` and `ffprobe` in PATH
- Output directory must be inside an initialized workspace

## Usage Examples

```bash
# Extract frames from Insta360 video
sfm insv2rig recording.insv --output ./fisheye_frames
```

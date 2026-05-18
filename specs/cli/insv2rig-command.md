# `sfm insv2rig` Command

## Overview

Extracts dual-fisheye frames from Insta360 `.insv` video files. Each video frame produces
two fisheye images (one per lens) plus a `.camrig` camera rig file describing the
Insta360 X5 camera geometry.

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
2. Extracts frames with `ffmpeg`, named `frame_%06d.jpg`
3. Splits each frame into left and right fisheye images, under `fisheye_left/`
   and `fisheye_right/`
4. Writes a `.camrig` camera rig file

## Output

The fisheye images are written to `<OUTPUT_DIR>/fisheye_left/` and
`<OUTPUT_DIR>/fisheye_right/`. A `.camrig` file named after the input video
(`<INPUT_STEM>.camrig`) is written to `<OUTPUT_DIR>` — the rig root — describing
a two-sensor `fisheye_360` rig: both sensors share the calibrated Insta360 X5
`OPENCV_FISHEYE` camera, sensor 0 (`fisheye_left`) sits at the identity
`sensor_from_rig` pose and sensor 1 (`fisheye_right`) is rotated 180° about Y
with the calibrated baseline. The sensor image patterns are
`fisheye_left/frame_%06d.jpg` and `fisheye_right/frame_%06d.jpg`.

`sfm solve` auto-discovers the `.camrig` and runs rig-aware SfM; see
[`solve-command.md`](solve-command.md) and
[`../formats/camrig-file-format.md`](../formats/camrig-file-format.md).

## Constraints

- Requires `ffmpeg` and `ffprobe` in PATH
- Output directory must be inside an initialized workspace

## Usage Examples

```bash
# Extract frames from Insta360 video
sfm insv2rig recording.insv --output ./fisheye_frames
```

# `sfm pano2rig` Command

## Overview

Converts equirectangular (360°) panorama images into a 6-face cubemap rig suitable for SfM
processing. Each panorama produces 6 perspective images (front, right, back, left, top,
bottom) with 90° field of view, plus a `.camrig` camera rig file.

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

Each face gets its own subdirectory, and the faces of one panorama share a frame
index — the panorama's position in sorted order — so they pair up into rig frames:

```
output_dir/
  front/frame_000000.jpg
  right/frame_000000.jpg
  back/frame_000000.jpg
  left/frame_000000.jpg
  top/frame_000000.jpg
  bottom/frame_000000.jpg
  front/frame_000001.jpg
  ...
  cubemap.camrig
```

`cubemap.camrig` is a six-sensor `cubemap` rig written into `output_dir` — the rig
root. All six faces share one square 90°-FOV `PINHOLE` camera and one optical centre,
so every `sensor_from_rig` translation is zero and only the rotation varies per face;
sensor 0 (`front`) sits at the identity pose. Each sensor's image pattern is
`<face>/frame_%06d.jpg`. `sfm solve` auto-discovers the `.camrig` and runs rig-aware
SfM; see [`solve-command.md`](solve-command.md) and
[`../formats/camrig-file-format.md`](../formats/camrig-file-format.md).

## Constraints

The output directory must be inside an initialized workspace (`sfm ws init`).

## Usage Examples

```bash
# Convert panoramas with default face size
sfm pano2rig ./panoramas --output ./faces

# Custom face size and quality
sfm pano2rig ./panoramas --output ./faces --face-size 1024 --jpeg-quality 90
```

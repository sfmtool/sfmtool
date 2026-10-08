# `sfm flow` Command

The flow command is a diagnostic for
[flow-based matching](../../core/features/flow-based-matching.md): it computes
dense optical flow from one image to another, moves the first image's SIFT
keypoints along it, and reports how many land near a keypoint of the second
image, optionally drawing the result or comparing it with the correspondences in
a reconstruction. It works on one image pair at a time.

## Overview

The flow is DIS optical flow ([optical-flow.md](../../core/features/optical-flow.md))
from IMAGE1 to IMAGE2, computed on the grayscale images. Each IMAGE1 keypoint is
moved along the flow to its *advected position*. An advected position that lies
inside IMAGE2 and within `--tolerance` pixels of an IMAGE2 keypoint is a *hit*,
paired with the nearest IMAGE2 keypoint. The command prints:

- the flow magnitude (mean, max, median) and histograms of its horizontal and
  vertical components;
- the keypoint count of each image and the number of hits, with a histogram of
  hit distances;
- the L2 descriptor distance of each hit pair, and the hits whose distance is at
  most `--descriptor-threshold`, with their own histograms;
- with `--reconstruction`, the comparison counts described under
  "Comparison mode" below.

A hit is the nearest keypoint and nothing more: unlike the flow-based matcher,
the command does not choose among nearby keypoints by descriptor, and the
descriptor threshold filters only the printed statistics. Its default of 100 is
lower than the matcher's 250
([flow-based-matching.md](../../core/features/flow-based-matching.md) explains
the choice of 250).

### Prerequisites

- **A `.sift` file for each image**, as written by
  [`sfm sift --extract`](../image-feature/sift-command.md). For an image inside a
  workspace it is looked up in the workspace's `feature_prefix_dir` under the
  image's own directory
  ([workspace.md](../../workspace/workspace.md#feature-storage-convention));
  outside a workspace, at the path that extraction with the COLMAP tool and its
  default options writes. A missing file is an error that names the path looked for.
- **Images of identical dimensions.** The flow is computed only between images
  of the same width and height; otherwise the command fails with
  `img_a and img_b must have the same shape`.

## Command Syntax

```bash
sfm flow <IMAGE1> <IMAGE2> [OPTIONS...]
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--draw / -d` | path | | Save visualization (omit for statistics only). Writes derived files next to the path: `<stem>_flow<ext>` for the flow field, plus `<stem>_A<ext>` and `<stem>_B<ext>` for the annotated images, or the path itself with `--side-by-side` |
| `--preset` | `fast` \| `default` \| `high_quality` | `default` | Flow quality preset |
| `--reconstruction / -r` | path | | `.sfmr` file to compare against; any other extension is a usage error |
| `--max-features` | int ≥ 1 | all | Maximum number of hits or correspondences drawn. Affects the drawing only; the printed statistics always cover every keypoint. See the drawing modes for how it is applied |
| `--tolerance` | float ≥ 0.1 | 3.0 | Pixel distance within which an advected position counts as landing on an IMAGE2 keypoint, and within which a reconstruction correspondence counts as agreeing with the flow |
| `--descriptor-threshold` | float ≥ 0 | 100.0 | L2 descriptor distance at or below which a hit is counted as descriptor-filtered. Changes the printed statistics only, never which hits are drawn |
| `--feature-size` | int ≥ 1 | 4 | Feature marker radius (pixels) |
| `--line-thickness` | int ≥ 1 | 1 | Line thickness (pixels) |
| `--side-by-side / --separate` | bool | `false` | Single combined image or two separate files |

Out-of-range values are rejected by the option parser before any work is done.

## Visualization

With `--draw`, every mode writes `<stem>_flow<ext>`: the flow field as a colour
image of IMAGE1's size. Hue encodes the flow direction (an HSV hue wheel over
the flow angle) and saturation the magnitude, normalized to the field's 99th
percentile magnitude; brightness is always full, so zero flow is white. A
legend in the top-left corner shows the colours of right, left, down and up.

The annotated pair is drawn on IMAGE1 and IMAGE2 as two files, `<stem>_A<ext>`
and `<stem>_B<ext>`, or with `--side-by-side` as one image at the `--draw` path,
IMAGE1 on the left and IMAGE2 on the right. Since both images have the same
dimensions, the side-by-side image is twice the width of one image.

### Flow only (no `--reconstruction`)

Draws only the hits, each in its own colour from a categorical palette (the
colour tells one hit from another; it does not encode the flow). On IMAGE1 each
hit's keypoint is a filled dot. On IMAGE2 each hit has a filled dot at the
advected position, a ring at the IMAGE2 keypoint it landed near, and a line
between them. IMAGE1 keypoints that are not hits are not drawn. `--max-features
N` keeps the first N hits, in IMAGE1 keypoint order.

### Comparison mode (`--reconstruction`)

The reconstruction's correspondences for the pair are the feature pairs, one per
3D point, that observe the same point in both images. The command compares them
with the flow:

- **Green** — Agreement: the reconstruction correspondence's IMAGE1 feature
  advects to within `--tolerance` of its IMAGE2 feature
- **Red** — Reconstruction only: the correspondence's IMAGE1 feature advects
  farther than `--tolerance` from its IMAGE2 feature
- **Yellow** — Flow only: a hit whose feature pair is not a reconstruction
  correspondence

`--max-features N` shrinks the three categories in proportion to their sizes,
rounding each down but keeping at least one of any non-empty category, so up to
N + 2 markers can be drawn.

Each image is found in the reconstruction by its path relative to the
reconstruction's workspace, so on a rig `fisheye_left/frame_01.jpg` and
`fisheye_right/frame_01.jpg` are told apart. An image inside the workspace that
the reconstruction does not contain is an error. An image outside the workspace
is matched by file name, and the command fails when no reconstruction image or
more than one has that name.

## Usage Examples

```bash
# Compute flow statistics between two images
sfm flow image_001.jpg image_002.jpg

# Visualize flow
sfm flow image_001.jpg image_002.jpg --draw flow.png --preset high_quality

# Compare flow against reconstruction
sfm flow image_001.jpg image_002.jpg --draw compare.png -r solve.sfmr
```

> Adjacent-pairs batch mode is not supported by `sfm flow`. Use `sfm epipolar
> --pairs-dir` for batched per-image visualizations, or wrap `sfm flow` in a
> shell loop for batch flow.

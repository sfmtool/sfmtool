# Discontinuity Analysis

## Motivation

Datasets used for Structure from Motion can come from many sources: frames extracted
from video, photographs taken along a motion path, fully unordered photo collections, or
a mixture of all of these. When images are sequential — as in video or a walk-along
capture — adjacent frames share dense visual correspondence beyond what sparse SIFT
features capture. This frame-to-frame correspondence can be used to cross-check SfM
solution quality.

Optical flow is a classical computer vision algorithm that estimates per-pixel motion
between two images. We have a Dense Inverse Search (DIS) implementation in the codebase
that can compute this between adjacent frames. By comparing what optical flow says about
frame-to-frame relationships against what the SfM solution says, we can find places where
the solution may have gone wrong — or where the input data has discontinuities of its
own, such as cuts in edited video or frames where an object passes right in front of the
camera.

This spec describes two analysis modes, for looking at a sequence of images without and
with a corresponding SfM solution:

1. **Image sequence analysis** — Uses optical flow between adjacent frames to find
   discontinuities in the raw data before any SfM processing.

2. **Reconstruction analysis** — Starts from camera motion in a `.sfmr` file to find
   pose discontinuities, then cross-checks each one against optical flow to classify it
   as either a real data discontinuity or a likely SfM error.

## Definitions

**Discontinuity**: A pair of adjacent frames (i, i+1) where the transition significantly
deviates from the local norm.

**Flow discontinuity**: Optical flow between adjacent frames is absent, unreliable, or
shows abnormally large displacement — indicating the images are not smoothly connected.

**Pose discontinuity**: Camera translation distance or rotation angle between adjacent
frames in a reconstruction significantly exceeds the local trend.


## Image Sequence Analysis

### Input

An unorganized collection of image file paths. The tool uses
`summarize_paths_by_sequence` from the `deadline` library to detect numbered sequences
among the filenames (e.g., `frame_001.jpg` through `frame_100.jpg`). Each detected
sequence is analyzed independently. Files that don't belong to any numbered sequence are
skipped.

### Method

All optical flow is computed using the DIS high quality preset.

Rather than computing flow for every consecutive pair in a long sequence, the analysis
samples pairs at two scales:

At each sample point i, compute two flow fields from the same base image:

1. **Local flow**: Frame i to frame i+1. This gives the frame-to-frame motion at that
   point in the sequence.

2. **Stride flow**: Frame i to frame i+N, where N is the current stride. The local flow
   scaled by N can be used as an initial estimate (via `compute_optical_flow_with_init`),
   which should help DIS converge when the motion is smooth. If the motion is smooth
   throughout the interval, the stride flow magnitude should be roughly N times the local
   flow magnitude. A large deviation suggests something changed in between — a
   discontinuity, a change in camera speed, or a change in scene content.

The stride N is adaptive. Start with an initial stride (e.g., N=8). After each sample
point, adjust N based on how consistent the ratio was:

- If the stride-to-local ratio is close to N (smooth motion), increase N — there's no
  need to sample densely in a region where nothing is changing. The next sample point
  advances by N.
- If the ratio deviates significantly from N, decrease N to sample more densely and
  narrow down where the discontinuity is.

This way, smooth sections of a sequence are skipped over quickly, while regions with
discontinuities or changing motion are examined more closely. The total number of flow
computations adapts to the content rather than being fixed.

For each computed pair, build two complementary representations of the flow:

1. **Global (u, v) histogram**: Bin all per-pixel flow vectors into a 2D histogram over
   (u, v) components. This captures the overall distribution of motion — a rigid camera
   motion produces a tight cluster, while a scene cut or mixed motion produces a
   scattered or multi-modal distribution.

2. **Spatial tile means**: Divide the image into a grid of tiles (e.g., 4x4) and compute
   the mean flow vector per tile. This captures spatially varying motion like parallax
   and rotation that a global histogram averages away.

To compare local flow against stride flow, scale the local flow representations by N
(scale the histogram bin coordinates, scale the tile mean vectors) and compare against
the stride representations. The two representations complement each other: the global
histogram catches changes in the overall motion distribution, while the tile means catch
changes in spatial structure.

Flag a discontinuity when:
- The scaled local histogram diverges from the stride histogram (e.g., by histogram
  intersection or chi-squared distance).
- The scaled local tile means diverge from the stride tile means (e.g., by mean L2
  distance across tiles).

### Output

- Per-sequence summary: sequence pattern, frame count, number of sampled pairs.
- Per-pair flow representations: global (u, v) histogram and spatial tile means.
- Stride comparison analysis: scaled local vs stride histogram and tile mean distances.
- List of flagged discontinuities with their metrics.
- Suggested segmentation: contiguous runs of consistent motion between discontinuities.

## Reconstruction Analysis

### Input

A `.sfmr` reconstruction file and the corresponding image files. As with image sequence
analysis, `summarize_paths_by_sequence` is used to detect numbered sequences among the
reconstruction's image names. Each detected sequence is analyzed independently.

### Step 1: Pose Discontinuity Detection

Within each detected sequence, evaluate each frame i by extrapolating to it from both
sides:

1. **Left extrapolation**: Use frames i-3, i-2, i-1 to extrapolate a predicted pose at
   frame i. Fit a quadratic (or cubic) through the camera centers and use corresponding
   higher-order interpolation for the rotations.

2. **Right extrapolation**: Use frames i+1, i+2, i+3 to extrapolate backward to frame i.

3. **Extrapolation error**: Measure the translation distance and rotation angle between
   each extrapolated pose and the actual pose at frame i.

At a smooth part of the sequence, both extrapolations should predict the actual pose
well. At a discontinuity, the extrapolation from one or both sides will miss badly. When
the left extrapolation is good but the right extrapolation is poor (or vice versa), the
discontinuity is between frame i and the side that extrapolates poorly — the sequence is
smooth up to that point and then breaks.

Flag frames where either extrapolation error exceeds a threshold. Frames near the start
or end of a sequence that don't have three neighbors on one side use a shorter
extrapolation window or are skipped.

### Step 2: Optical Flow Cross-Check

For each pose discontinuity found in Step 1, use the same adaptive stride flow analysis
as image sequence mode: compute local and stride flow pairs around the flagged frame,
build the global (u, v) histogram and spatial tile mean representations, and compare
scaled local flow against stride flow.

If the flow representations are consistent across the flagged region (the scaled local
histograms and tile means match the stride representations), the images are visually
continuous and the pose discontinuity is suspect — it may be an SfM error. If the flow
representations also show a discontinuity, it corroborates the pose analysis and suggests
a real break in the data.

### Step 3: Reprojection Error Context

For each flagged discontinuity, also report:

- Mean reprojection error for the images involved (from existing
  `compute_observation_reprojection_errors`).
- Number of shared 3D points between adjacent frames (from covisibility).
- Whether the images are connected in the covisibility graph (graph distance).

High reprojection error and low covisibility strengthen the case that a pose
discontinuity is an SfM error.

### Output

- Per-sequence summary: sequence pattern, frame count.
- Per-frame extrapolation errors (left and right).
- Per-pair flow analysis: global histogram and tile mean comparisons.
- Flagged discontinuities with pose and flow agreement/disagreement.
- Reprojection error and covisibility context for flagged frames.


## Existing Infrastructure

Code and patterns to build on:

| Component | Location | Relevance |
|---|---|---|
| Camera center computation | `_inspect_images.py:_compute_camera_centers()` | World-space positions |
| Rotation angle computation | `_inspect_images.py:_compute_rotation_angle()` | Rotation between poses |
| DIS optical flow | `_sfmtool.compute_optical_flow()` | Flow computation |
| DIS with initial flow | `_sfmtool.compute_optical_flow_with_init()` | Stride flow from scaled local |
| Sequence detection | `deadline.job_attachments.api.summarize_paths_by_sequence` | Input grouping |
| Reprojection errors | `_inspect_metrics.py` | Error context for flagged frames |
| Covisibility graph | `_image_pair_graph.py` | Shared point counts |

## Open Questions

- **Extrapolation order**: Quadratic vs cubic fit for pose extrapolation. Three points
  allow a quadratic fit; using more neighbors could improve accuracy but may smooth over
  real changes in motion.

- **Adaptive stride tuning**: How aggressively should the stride grow and shrink? What
  are good thresholds for the histogram and tile mean comparisons to trigger stride
  changes?

- **Histogram comparison metric**: Histogram intersection, chi-squared distance, and
  earth mover's distance are all options. Which works best in practice for flow
  distributions?

- **Non-sequential data**: This design assumes images are ordered sequentially. For
  unordered image sets, the "adjacent" concept doesn't apply. Should there be a mode that
  analyzes all covisibility-connected pairs instead?

- **Visualization**: Should discontinuities be visualizable (e.g., side-by-side images
  with flow overlay at each flagged pair)? The existing `sfm flow --draw` infrastructure
  could be reused.

- **Segmentation output**: When discontinuities split a sequence into segments, should
  the tool output segment boundaries in a machine-readable format (e.g., JSON) for
  downstream use (e.g., solving each segment separately)?

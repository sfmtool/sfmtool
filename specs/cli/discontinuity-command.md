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
   pose discontinuities, with reprojection error and covisibility context. Optical flow
   cross-check (Step 2) is designed but not yet wired into the output.

## Definitions

**Discontinuity**: A pair of adjacent frames (i, i+1) where the transition significantly
deviates from the local norm.

**Flow discontinuity**: Optical flow between adjacent frames is absent, unreliable, or
shows abnormally large displacement — indicating the images are not smoothly connected.

**Pose discontinuity**: Camera translation distance or rotation angle between adjacent
frames in a reconstruction significantly exceeds the local trend.


## CLI

```
sfm discontinuity [OPTIONS] PATHS...
```

### Arguments

| Argument | Description |
|----------|-------------|
| `PATHS` | Image paths, directories, or a single `.sfmr` file. |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--range`, `-r` | | Range expression of file numbers to use from input directories. |
| `--initial-stride N` | 1 | Starting stride for adaptive sampling. |
| `--min-stride N` | 1 | Minimum stride (floor for adaptive shrinking). |
| `--max-stride N` | 32 | Maximum stride (ceiling for adaptive growing). |
| `--no-adaptive` | false | Disable adaptive stride adjustment; keep stride fixed. |
| `--save-flow-dir PATH` | | Directory to save optical flow color images (Middlebury convention). |

### Examples

```bash
# Analyze all images in a directory (adaptive stride starting at 1)
sfm discontinuity images/

# Start with a larger stride to skip smooth regions faster
sfm discontinuity images/ --initial-stride 16

# Analyze a subset of frames
sfm discontinuity images/ -r 1-100

# Fixed stride, no adaptation
sfm discontinuity images/ --initial-stride 4 --no-adaptive

# Save flow visualizations
sfm discontinuity images/ --save-flow-dir /tmp/flow_output
```


## Image Sequence Analysis

### Input

An unorganized collection of image file paths. The tool uses
`summarize_paths_by_sequence` from the `deadline` library to detect numbered sequences
among the filenames (e.g., `frame_001.jpg` through `frame_100.jpg`). Each detected
sequence is analyzed independently. Files that don't belong to any numbered sequence are
skipped.

### Method

All optical flow is computed using the DIS high quality preset.

At each sample point i, compute two flow fields from the same base image:

1. **Local flow**: Frame i to frame i+1. This gives the frame-to-frame motion at that
   point in the sequence.

2. **Stride flow**: Frame i to frame i+N, where N is the current stride (minimum 2). The
   local flow scaled by N is used as an initial estimate (via
   `compute_optical_flow_with_init`), which helps DIS converge when the motion is smooth.
   If the motion is smooth throughout the interval, the stride flow magnitude should be
   roughly N times the local flow magnitude. A large deviation suggests something changed
   in between — a discontinuity, a change in camera speed, or a change in scene content.

When stride is 1, every frame is sampled: local flow is computed from i to i+1, and
the comparison stride flow is computed from i to i+2 (the comparison stride is bumped to
`max(stride, 2)` since comparing a flow to itself is not useful). The effective stride is
also clamped to `n_images - 1 - i` to stay within the sequence.

After each sample point, the sequence advances by the current stride value (`i += stride`).

#### Adaptive Stride

The stride is adaptive by default (disable with `--no-adaptive`). After each sample
point, the stride is adjusted based on two signals: the magnitude ratio and the in-bounds
pixel coverage.

**Magnitude ratio**: The ratio of stride flow median magnitude to local flow median
magnitude is divided by the stride to get a normalized ratio. For smooth motion, this
normalized ratio should be close to 1.0. The ratio is evaluated against three
log-symmetric bands:

| Band | Condition | Action |
|------|-----------|--------|
| Shrink | `ratio/stride < 0.75` or `ratio/stride > 1.33` | Halve stride |
| Keep | Between shrink and grow thresholds | Hold stride |
| Grow | `0.85 < ratio/stride < 1.18` | Double stride |

The thresholds are multiplicatively invariant — `0.75` and `1/0.75 ≈ 1.33` for the
outer band, `0.85` and `1/0.85 ≈ 1.18` for the inner band.

**In-bounds coverage overrides**:

| Coverage | Effect |
|----------|--------|
| < 25% | Force shrink (data too sparse to trust) |
| 25–50% | Suppress grow (keep or shrink only) |
| > 50% | Use ratio bands as normal |

**Retry on shrink**: When the stride shrinks, the same frame is re-analyzed at the
smaller stride before advancing. The local flow (i→i+1) is reused from the first
attempt — only the stride flow is recomputed. The result from the larger stride is
marked as `superseded` and excluded from the summary. A retry only happens when the
new effective stride is strictly smaller than the previous effective stride (avoiding
infinite loops at the end of a sequence where the effective stride is clamped).

When growing, the stride doubles (up to `--max-stride`). When shrinking, it halves
(down to `--min-stride`).

#### Flow Representations

For each computed pair, build two complementary representations of the flow:

1. **6x6 direction histogram**: Bin all per-pixel flow vectors into a 6x6 2D histogram
   over (u, v) components. The bin range is fixed at 1/3 of the larger image dimension
   on each side of zero, so the full histogram covers the range of plausible single-frame
   motion. Rows correspond to vertical flow (v), columns to horizontal flow (u). The
   histogram captures the overall distribution of motion directions and magnitudes — a
   rigid camera motion produces a tight cluster, while a scene cut or mixed motion
   produces a scattered or multi-modal distribution.

2. **3x3 spatial tile grid**: Divide the image into a 3x3 grid of tiles and compute the
   median flow magnitude and mean flow vector per tile. This captures spatially varying
   motion like parallax and rotation that a global histogram averages away.

Both representations are computed for the scaled local flow (local × N) and the stride
flow. A third difference grid (stride − local×N) is also computed to highlight where
the two disagree.

#### In-bounds Masking

When comparing local flow scaled by N against stride flow, pixels whose scaled local
flow would take them outside the image bounds cannot contribute meaningful data to either
representation. The stride flow at those pixels has no corresponding content to match
against in the target frame. An in-bounds mask is computed from the scaled local flow — a
pixel at (x, y) is in-bounds if (x + N×flow_u, y + N×flow_v) stays within the image
dimensions — and applied to all representations (histograms, tile magnitudes, tile means).
The percentage of in-bounds pixels is reported alongside the output.

### Output

For each sample point, the tool prints:

- Frame number, filenames, and median flow magnitude for the local pair.
- Stride target filename, median magnitude, magnitude ratio vs expected ratio.
- Side-by-side 3x3 tile grids showing: scaled local magnitudes with direction arrows,
  stride magnitudes with direction arrows, and signed difference magnitudes with direction
  arrows.
- In-bounds pixel percentage.
- Side-by-side 6x6 direction histograms rendered as block-character bar charts (shared
  scale), for scaled local and stride flow.

When the stride changes, a message is printed with the old and new stride values and the
reason (e.g., `↓ stride 8→4 (ratio/stride=0.62, outside [0.75, 1.33])`).

After all sample points, a **summary** lists detected discontinuities — frames where
the normalized ratio falls outside the [0.75, 1.33] band. Each entry shows the frame
number, filename, normalized ratio, stride, local magnitude, and a classification:

| Normalized ratio | Classification |
|-----------------|----------------|
| < 0.5 | strong deceleration |
| 0.5–0.75 | deceleration |
| 1.33–2.0 | acceleration |
| > 2.0 | strong acceleration |

Superseded results (from retried frames) are excluded from the summary.

### Flow Image Saving

When `--save-flow-dir` is provided, optical flow color images are saved using the
Middlebury color convention (same as `sfm flow --draw`). Images are saved incrementally
as each sample point is computed, not held in memory.

For each sample point, up to two images are saved:

- **Local flow**: `{sequence_name}_from_{i}_to_{i+1}.jpg`
- **Stride flow**: `{sequence_name}_from_{i}_to_{i+N}.jpg`

The sequence name is derived from the detected sequence pattern by stripping the
extension and printf format specifier (e.g., `seoul_bull_sculpture_%02d.jpg` becomes
`seoul_bull_sculpture`). Each image includes a Middlebury color wheel legend.

### Image Caching

Grayscale images are cached in memory to avoid redundant loads. The cache is evicted as
the analysis advances — images more than one position behind the current sample point are
removed to limit memory usage for long sequences.


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

### Step 2: Optical Flow Cross-Check (Not Yet Implemented)

For each pose discontinuity found in Step 1, use the same adaptive stride flow analysis
as image sequence mode: compute local and stride flow pairs around the flagged frame,
build the global (u, v) histogram and spatial tile mean representations, and compare
scaled local flow against stride flow.

If the flow representations are consistent across the flagged region (the scaled local
histograms and tile means match the stride representations), the images are visually
continuous and the pose discontinuity is suspect — it may be an SfM error. If the flow
representations also show a discontinuity, it corroborates the pose analysis and suggests
a real break in the data.

> **Status:** The infrastructure for this step exists (the adaptive stride flow code is
> reusable from image sequence mode), but detailed per-edge flow histogram output is not
> yet wired into the reconstruction analysis output.

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
| Flow visualization | `visualization/_flow_display.py:_flow_to_color()`, `_draw_flow_legend()` | Middlebury color images |
| Reprojection errors | `_inspect_metrics.py` | Error context for flagged frames |
| Covisibility graph | `_image_pair_graph.py` | Shared point counts |

## Open Questions

- **Extrapolation order**: Quadratic vs cubic fit for pose extrapolation. Three points
  allow a quadratic fit; using more neighbors could improve accuracy but may smooth over
  real changes in motion.

- **Histogram comparison metric**: Histogram intersection, chi-squared distance, and
  earth mover's distance are all options. Which works best in practice for flow
  distributions?

- **Non-sequential data**: This design assumes images are ordered sequentially. For
  unordered image sets, the "adjacent" concept doesn't apply. Should there be a mode that
  analyzes all covisibility-connected pairs instead?

- **Segmentation output**: When discontinuities split a sequence into segments, should
  the tool output segment boundaries in a machine-readable format (e.g., JSON) for
  downstream use (e.g., solving each segment separately)?

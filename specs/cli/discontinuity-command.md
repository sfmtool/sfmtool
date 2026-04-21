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

2. **Reconstruction analysis** — Starts from a `.sfmr` file and combines four signals
   to catch discontinuities: pose extrapolation errors, local step-size ratio,
   covisibility drop across the edge, and per-image observation-count outliers.
   Reprojection error and shared 3D point context are reported for each flagged edge.

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

### Step 2: Secondary Signals

Pose extrapolation on its own misses subtle discontinuities because polynomial
extrapolators silently absorb smooth scale or slope changes — a sequence that goes from
"walking" to "slow pan" across one edge stays smooth at the pose level but has nothing
covisible across the break. Three complementary signals catch these cases without
depending on pose smoothness:

**Step-size ratio (per edge).** For edge i → (i+1), take a window of `STEP_RATIO_WINDOW`
edges on each side (default 8, excluding the edge itself). Let `m_pre` and `m_post` be
the median step length `‖C_{k+1} − C_k‖` in the two windows. Flag the edge when
`max(m_pre / m_post, m_post / m_pre) > STEP_RATIO_THRESHOLD` (default 1.5). Catches
scale shifts where the camera's translational velocity changes abruptly. The test is
two-tailed — it fires whether motion speeds up or slows down — unlike the pose-residual
test which fires mostly on spikes.

**Covisibility drop (per edge).** For edge i → (i+1), collect `P_pre` = union of 3D
points observed across images [i − w + 1, i] and `P_post` = union across [i + 1, i + w],
with `w = OVERLAP_WINDOW` (default 16). Compute the per-edge overlap ratio
`cross = |P_pre ∩ P_post| / min(|P_pre|, |P_post|)`, then define the drop factor as
`median(neighbor cross values) / cross`. Flag when the drop factor exceeds
`OVERLAP_DROP_THRESHOLD` (default 1.8). Baseline neighbors are collected from a
`OVERLAP_BASELINE_WINDOW` of ±24 edges around the candidate. A `cross` of 0 (no tracks
survive the edge) yields an infinite drop factor and always flags. Robust to loop
closure — revisits that inflate cross only suppress detection, never cause false
positives.

**Observation-count outlier (per frame).** Image i may be a "bridge frame" — motion
blur, occlusion, brief tracking slip — if its track count drops well below the local
norm. Using a symmetric rolling window of `OBS_WINDOW` frames (default 24) centered on
i (excluding i itself), compute the MAD-based z-score
`z = (nobs[i] − median) / (1.4826 · MAD)`. Flag frame i when `z < −OBS_Z_THRESHOLD`
(default 2.5). Unlike the edge signals above, a frame-level obs outlier does **not**
flag an edge on its own — a single dim image is often benign. Instead, obs flags attach
to adjacent edges as endpoint context when those edges are already flagged by one of
the other signals.

**Combined rule.** An edge is reported as a discontinuity when any of the following
fires: pose extrapolation (`L.t`, `L.r`, `R.t`, `R.r`), step-size ratio (`Step`), or
covisibility drop (`Cov`). Clustering deduplicates adjacent edges in the same break.
The output partitions discontinuities into:

- **Single-signal (low confidence):** only one of {P, S, C} fired. Usually pose-only
  hits on edges where a polynomial extrapolator mildly overshot.
- **Multi-signal (high confidence):** two or more of {P, S, C} fired, or at least one
  plus Obs context at an endpoint.

Scale-shift discontinuities that the pose test misses (as in the KerryPark 831→832
case) reliably trigger both Step and Cov simultaneously and show up as high-confidence
hits.

### Step 3: Reprojection Error Context

For each flagged discontinuity, also report:

- Mean reprojection error for the images involved (from existing
  `compute_observation_reprojection_errors`).
- Number of shared 3D points between adjacent frames (from covisibility).
- Whether the images are connected in the covisibility graph (graph distance).

High reprojection error and low covisibility strengthen the case that a pose
discontinuity is an SfM error.

### Output

- Header: threshold values for all four signals.
- Per-sequence line: sequence pattern, frame count.
- Per-frame table with columns: frame number, image name, left/right pose
  extrapolation residuals (`L.trans`, `L.rot`, `R.trans`, `R.rot`), step-size ratio
  (`StepR`) and coviz drop (`CovR`) for the landing edge, obs-count z-score (`ObsZ`),
  and a `Flag` column listing fired codes (`L.t`, `L.r`, `R.t`, `R.r`, `Step`, `Cov`,
  `Obs`). StepR and CovR attach to the landing frame of each edge so row i shows the
  metrics for edge (i−1, i).
- Summary table (one row per detected discontinuity): edge, raw edge translation and
  rotation, StepR, CovR, endpoint obs z-scores, shared 3D point count, per-endpoint
  mean reprojection errors, and a `Signals` column listing fired codes
  (e.g. `P S C O`). Values that exceeded their signal's threshold are wrapped in
  `< >` brackets.
- Footer: total count, plus a partition into single-signal (low-confidence) vs
  multi-signal (high-confidence) hits, and a legend for the signal codes.


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

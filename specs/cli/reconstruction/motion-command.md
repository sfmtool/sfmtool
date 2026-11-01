# Camera Motion Analysis

`sfm motion` looks for discontinuities in a numbered sequence of images: places
where two neighbouring frames are not smoothly connected. Given images, it
computes optical flow from a frame to the next one and to one several frames
ahead, and flags the frames where the longer flow is out of proportion to the
shorter one, a sign that the motion speeds up, slows down or breaks. Given a
single `.sfmr` reconstruction, it orders the cameras by frame number and flags
the transitions where a camera's pose departs from what its neighbours predict,
where the typical step between cameras changes across the transition, or where
the cameras on either side share far fewer 3D points than they do at the
transitions around it. For a reconstruction it also gives a verdict: which of
the flagged transitions are breaks, places where the solve put two parts of the
camera path in inconsistent places, and it ends its report with one line naming
them. It is for finding cuts and occlusions in the input before solving, and
places where a solve went wrong after. It prints a report of the flagged
transitions and can also write it as JSON.

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
   Reprojection error and shared 3D point context are reported for each flagged edge,
   and each flagged edge is classified as a break in the solve or not.

## Definitions

**Discontinuity**: A pair of adjacent frames (i, i+1) where the transition significantly
deviates from the local norm.

**Flow discontinuity**: Optical flow between adjacent frames is absent, unreliable, or
shows abnormally large displacement — indicating the images are not smoothly connected.

**Pose discontinuity**: Camera translation distance or rotation angle between adjacent
frames in a reconstruction significantly exceeds the local trend.


## CLI

```
sfm motion [OPTIONS] PATHS...
```

### Arguments

| Argument | Description |
|----------|-------------|
| `PATHS` | Image paths, directories, or a single `.sfmr` file. |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--range`, `-r` | | Range expression of file numbers to use from input directories. |
| `--initial-stride N` | 1 | Starting stride for adaptive sampling. Must be at least 1. |
| `--min-stride N` | 1 | Minimum stride (floor for adaptive shrinking). Must be at least 1. |
| `--max-stride N` | 32 | Maximum stride (ceiling for adaptive growing). Must be at least 2; Click rejects a smaller value. |
| `--no-adaptive` | false | Disable adaptive stride adjustment; keep stride fixed. |
| `--save-flow-dir PATH` | | Directory to save optical flow color images (hue encodes direction, saturation magnitude). |
| `--json PATH` | | Write a machine-readable JSON report of the analysis to this path. Human-readable output is unchanged. |

### Examples

```bash
# Analyze all images in a directory (adaptive stride starting at 1)
sfm motion images/

# Start with a larger stride to skip smooth regions faster
sfm motion images/ --initial-stride 16

# Analyze a subset of frames
sfm motion images/ -r 1-100

# Fixed stride, no adaptation
sfm motion images/ --initial-stride 4 --no-adaptive

# Save flow visualizations
sfm motion images/ --save-flow-dir /tmp/flow_output

# Emit a machine-readable report alongside the printed output
sfm motion reconstruction.sfmr --json discontinuity.json
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
| Shrink | `ratio/stride < 0.75` or `ratio/stride > 1/0.75` | Halve stride |
| Keep | Between shrink and grow thresholds | Hold stride |
| Grow | `0.85 < ratio/stride < 1/0.85` | Double stride |

The thresholds are multiplicatively invariant — `0.75` and `1/0.75 ≈ 1.33` for the
outer band, `0.85` and `1/0.85 ≈ 1.18` for the inner band. The outer band's edges,
and the classifier below that reads them, live in one place:
[`motion/ratio_band.py`](../../../src/sfmtool/motion/ratio_band.py). The
displayed `1.33` is `1/0.75` formatted to two decimals, not an independent
threshold.

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
| `1/0.75`–2.0 | acceleration |
| > 2.0 | strong acceleration |

Superseded results (from retried frames) are excluded from the summary.

### Flow Image Saving

When `--save-flow-dir` is provided, optical flow color images are saved in the
same colouring as `sfm flow --draw`: hue encodes flow direction and saturation its
magnitude. Images are saved incrementally
as each sample point is computed, not held in memory.

For each sample point, up to two images are saved:

- **Local flow**: `{sequence_name}_from_{i}_to_{i+1}.jpg`
- **Stride flow**: `{sequence_name}_from_{i}_to_{i+N}.jpg`

The sequence name is derived from the detected sequence pattern by stripping the
extension and printf format specifier (e.g., `seoul_bull_sculpture_%02d.jpg` becomes
`seoul_bull_sculpture`). Each image includes a legend of the right, left, down and up colours.

### Image Caching

Grayscale images are cached in memory to avoid redundant loads. The cache is evicted as
the analysis advances — images more than one position behind the current sample point are
removed to limit memory usage for long sequences.


## Reconstruction Analysis

### Input

A single `.sfmr` reconstruction file. The analysis reads the reconstruction's camera
poses, tracks and reprojection errors and never opens an image file, so the images
need not be present. Loading the file does need the workspace it belongs to:
`SfmrReconstruction.load` looks for it at the workspace path recorded in the file,
relative and then absolute, and then searches upward from the file for a
`.sfm-workspace.json` marker, and fails with "Could not resolve workspace directory"
when none of these finds one. As with image sequence analysis,
`summarize_paths_by_sequence` is used to detect numbered sequences among the
reconstruction's image names. Each detected sequence is analyzed independently.

### Step 1: Pose Discontinuity Detection

Within each detected sequence, evaluate each frame i by extrapolating to it from both
sides:

1. **Left extrapolation**: Two fits are made and the **smaller** error of the two is
   reported — a linear fit through frames i-2, i-1 and a quadratic through i-3, i-2,
   i-1, each through the camera centers with corresponding-order interpolation for the
   rotations. Taking the minimum keeps a discontinuity at the far end of the 3-frame
   window from inflating the error for the near side: the 2-frame fit still sees a
   smooth run. Where fewer than three frames precede i, only the linear fit is made.

2. **Right extrapolation**: The same pair of fits mirrored — linear from i+2, i+1 and
   quadratic from i+3, i+2, i+1 — extrapolating backward to frame i.

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

**Step-size ratio (per edge).** For edge i → (i+1), take the `STEP_RATIO_WINDOW − 1`
edges on each side (default window 8, so 7 edges each way, the edge itself excluded);
an edge with fewer than 2 edges of context on either side reports no ratio. Let `m_pre`
and `m_post` be the median step length `‖C_{k+1} − C_k‖` in the two windows. Flag the
edge when
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
`OVERLAP_BASELINE_WINDOW` of ±24 edges around the candidate, and an edge with fewer
than 3 such neighbours reports no drop. A `cross` of 0 (no tracks survive the edge)
yields an infinite drop factor and always flags. Robust to loop closure — revisits that
inflate cross only suppress detection, never cause false positives.

The whole signal is off on short sequences: a sequence of fewer than `3 · w` frames (48
at the default window) reports no drop for any edge, because typical track lifetimes
there exceed the window and the test cannot tell a real break from natural track aging.
All four checked-in datasets are shorter than that, so `Cov` never fires on them.

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

A scale-shift discontinuity that the pose test misses changes both the step length
and the set of tracks seen across the edge, so it usually fires Step and Cov together
and is reported as a high-confidence hit.

The confidence partition says how many signals agree, not whether the solve is wrong:
a sound solve of a video lists many discontinuities, several of them on two or more
signals. Whether a listed discontinuity is an error in the solve is decided by the
break verdict of Step 4.

### Step 3: Reprojection Error Context

For each flagged discontinuity, also report:

- Mean reprojection error for the images involved (from existing
  `compute_observation_reprojection_errors`), one value per endpoint of each core edge.
- Number of shared 3D points between the core edge's two frames (from covisibility).
  That count is the whole connectivity answer the report carries: `0` says the two
  images share no surviving track or that their viewing directions differ by more
  than 90°, and no graph distance is computed.

High reprojection error and low covisibility strengthen the case that a pose
discontinuity is an SfM error.

### Step 4: Break Verdict

Each listed discontinuity, an edge (a, a+1) between consecutive registered images, is
classified as a **break** or not. A break is a place where the solve put the parts of
the camera path on either side in inconsistent places, as a global solve of a video
can while its registered count, points per image and reprojection error all look
normal. The rule compares the edge with four references taken over every consecutive
pair of registered images in the sequence, each a median so that the few breaks it is
used to find do not raise it:

- `rotation_per_frame_deg`: the median of `rotation / frame-number gap`, the camera's
  typical rotation per unit of frame number.
- `median_frame_gap`: the median frame-number gap between consecutive images.
- `median_shared_points`: the median number of 3D points two consecutive images share.
- `median_nearby_shared_points`: the median of the nearby shared count defined below.

For an edge with frame-number gap `g`, relative rotation `θ`, `s` shared 3D points and
`c` nearby shared points:

```
expected_rotation = rotation_per_frame_deg × max(g, median_frame_gap)
rotation_ratio    = θ / expected_rotation
break             = rotation_ratio > BREAK_ROTATION_FACTOR
                    and s < BREAK_SHARED_FRACTION × median_shared_points
                    and c < BREAK_SHARED_FRACTION × median_nearby_shared_points
```

with `BREAK_ROTATION_FACTOR = 4`, `BREAK_SHARED_FRACTION = 0.2` and
`BREAK_WINDOW = 8`, in [`motion/constants.py`](../../../src/sfmtool/motion/constants.py).

The **nearby shared count** of edge (a, a+1) is the number of 3D points observed both
in an image on the before side and in an image on the after side, where the before
side is every registered image whose frame number lies from that of image
a − `BREAK_WINDOW` + 1 to that of image a, and the after side every one from image
a + 1 to image a + `BREAK_WINDOW` (the sides stop at the ends of the sequence). The
sides are the sequence's own `BREAK_WINDOW` images on each side of the edge, plus the
images of any other sequence of the reconstruction that carry those frame numbers,
which in a rig are the other sensors' images of the same frames.

**Why the expected rotation comes from the frame-number gap.** The frame list a solve
used can skip frames (a run that failed to register, or frames left out) and can have
blurred frames swapped for a neighbour, so the gap between consecutive registered
images varies around the stride. The command does not know that frame list. Taking the
rate per unit of frame number, and multiplying it by each edge's own gap, allows an
edge across dropped frames the rotation of every frame it skips, with no need to know
which frames were dropped or what the stride was. The `max` with the median gap allows
an edge that is closer in frame number than the typical pair a typical pair's
rotation, so in an irregularly numbered photo set one closely numbered pair is not
judged against a fraction of a step. When the expected rotation is 0, in a sequence
whose cameras do not turn, any rotation counts as exceeding it.

**Why each condition is needed.** The rule was measured on 77 reconstructions: sound
and broken global and incremental solves of three videos at strides 2 to 13, with 24
to 383 registered images each, and sound solves and ground truths of the checked-in
photo datasets (`seoul_bull_sculpture`, `dino_dog_toy`, `seattle_backyard`,
`kerry_park`). Its 30 breaks were in eight broken video solves; no condition can be
dropped without misclassifying pairs from the others.

- Rotation is needed because a video's pairs that share few points mostly turn no
  more than usual. A sound DinoLedge solve that dropped frames 1081-1099 has a pair
  1072->1108 that fires three signals and shares no points, but turns 20.7° over 36
  frame numbers, 1.2 times its expected rotation, and other sound pairs that share
  under 0.02 of the median turn 1.5 to 2.7 times theirs.
- The pair's shared points are needed because sound video pairs can turn sharply. A
  sound Daegu stride-7 solve has a listed pair that turns 4.5 times its expected
  rotation (5.0 times in a solve of 30 of its images), but its images share 136 to 183
  points, 0.32 to 0.69 of the median. Without this condition the closest pair the
  rotation factor would have to clear is one that turns 3.7 times its expected
  rotation, shares 0.27 of the median pair count and 0.16 of the median nearby count,
  rather than one at 3.4 times.
- The nearby shared count is needed because in a photo set a large move between two
  consecutive photos is often real. `dino_dog_toy` photographs a toy in rings, and the
  first photo of a ring turns 5.4 to 6.5 times the usual rotation from the last photo
  of the ring before and shares under 0.2 of the median pair count with it; the `kerry_park` rig turns
  179° between frames 17 and 18, 6.8 times the usual rotation, and those two images
  share no points. The photos a few steps either side of such a move still see the
  same points, so its nearby shared count is 0.77 to 1.44 of the median. Across a
  break in a video solve the parts on either side are not tied together near it at
  all, and the breaks' nearby counts were at most 0.10 of the median. The window of 8
  is wide enough to reach the photos that see the same points again: with 2 images a
  side some `dino_dog_toy` ring starts fall to 0.01 of the median. The other
  sensors of a rig are counted because a solve can leave one sensor's images on either
  side of a turn with almost nothing in common while the other sensor's images, at the
  same frames, tie the rig together: in three `kerry_park` solves the right sensor's
  own images give frames 17 and 18 a nearby count under 0.2 of the median (0.04 in
  one, whose relative rotation there is within 0.5° of the ground truth's), and with
  the left sensor's images counted it is 1.0 to 1.1 of the median.

The breaks turned 5.5 to 32 times their expected rotation and shared 0 to 66 points
(at most 0.13 of the median pair count, and at most 0.10 of the median nearby count).
Among the pairs that are not breaks, those that pass both shared-points conditions
turned at most 3.4 times their expected rotation, and those that turned over 4 times
shared at least 0.31 of the median pair count or at least 0.77 of the median nearby
count. The factor 4 sits between 3.4 and 5.5, nearer the 3.4, so that a break turning
less than the ones measured is still called one; the fraction 0.2 sits between the
breaks' 0.13 and 0.10 and the other pairs' 0.31 and 0.77. The thresholds on points are
fractions of the sequence's own medians rather than fixed counts because point density
differs between captures: the 17 Seoul bull photos share a median of 58 points per
consecutive pair and at most 93, so a fixed count of 100 would call every one of their
pairs poorly connected.

**Photo sets and points at infinity.** Numbered photos that are not a sequence give
large consecutive rotations, so their median rotation is large, and once it exceeds 45°
no rotation (at most 180°) reaches 4 times it. The verdict reads camera rotations and
track membership only, never point positions, so a point at infinity (`w = 0`) counts
as a shared point like any other track: its direction ties the two cameras' rotations
together. The pair's shared count is the covisibility count of Step 3, which leaves out
pairs whose viewing directions differ by more than 90°, so such a pair shares 0 points;
the nearby count reads the tracks directly and has no such filter.

Only listed discontinuities are classified. Clustering keeps one edge of each run of
adjacent flagged edges, so a break inside a run whose kept edge is another break is
not named separately.

### Output

- Header: threshold values for all four signals.
- Per-sequence line: sequence pattern, frame count.
- Per-frame table with columns: frame number, image name, left/right pose
  extrapolation residuals (`L.trans`, `L.rot`, `R.trans`, `R.rot`), step-size ratio
  (`StepR`) and coviz drop (`CovR`) for the landing edge, obs-count z-score (`ObsZ`),
  and a `Flag` column listing fired codes (`L.t`, `L.r`, `R.t`, `R.r`, `Step`, `Cov`,
  `Obs`). StepR and CovR attach to the landing frame of each edge so row i shows the
  metrics for edge (i−1, i).
- Summary, per sequence: a line with the frame, discontinuity and break counts, a
  `Break rule:` line giving the sequence's references (rotation per frame number,
  median gap, and the two shared-points thresholds), and a table with one row per
  detected discontinuity: edge, raw edge translation and rotation, StepR, CovR,
  endpoint obs z-scores, shared 3D point count, per-endpoint mean reprojection errors,
  a `Signals` column listing fired codes (e.g. `P S C O`), `NearPts` (the nearby shared
  count of Step 4), `RotR` (its rotation ratio) and `Break` (`BREAK` or `-`). Values
  that exceeded their signal's threshold are wrapped in `< >` brackets.
- Footer: total count, plus a partition into single-signal (low-confidence) vs
  multi-signal (high-confidence) hits, and a legend for the signal codes and the
  break columns.
- Verdict, the last line before the timing: `Verdict: no breaks`, or
  `Verdict: N breaks: A->B, ...` naming each break by its frame numbers (`1 break`
  for one), with the sequence name before each edge when the reconstruction has more
  than one sequence. For example `Verdict: 1 break: 2225->2229`.

The command exits 0 whatever the verdict, as it does for every finding; a script reads
the verdict from `break_count` in the `--json` report.


## JSON Output

When `--json PATH` is provided, the tool writes a JSON file in addition to its
normal printed output. The file captures the per-frame metrics, the detected
discontinuities, and (for reconstruction mode) the segment partition implied
by those discontinuities, so downstream tools can act on the results without
parsing the human-readable tables.

### Schema (version 1)

Top-level object:

```jsonc
{
  "schema_version": 1,
  "mode": "reconstruction" | "image_sequence",
  "thresholds": { ... },           // global, mode-specific (see below)
  "sequences": [ { ... }, ... ]    // one per detected numbered sequence
}
```

`schema_version` increments on backwards-incompatible changes — renames,
type changes, removals, or semantic changes to existing fields. Adding a new
optional field, a new enum value, or widening a value's range is additive and
does not bump the version.

All numbers are JSON numbers. `NaN`, `+inf`, and `-inf` are emitted as `null`
to keep the output strictly RFC-8259 compliant. Tuple-keyed dicts (edges,
image-pair counts) are emitted as objects with `[a, b]` array fields rather
than encoded keys. Per-frame numpy arrays used for human display (flow
histograms, tile grids) are **not** emitted.

### Reconstruction mode

`thresholds`:

| Field | Description |
|-------|-------------|
| `pose_trans_factor` | Multiplier applied to each sequence's median step length. Fixed at `3.0` in v1 — not a CLI flag. The resolved per-sequence threshold is in `sequences[].pose_trans_threshold`. |
| `pose_rot_deg` | Fixed rotation residual threshold (`15.0`). Echoed per sequence as `pose_rot_threshold` for symmetry with the translation threshold. |
| `step_ratio` | `STEP_RATIO_THRESHOLD` (default `1.5`). |
| `overlap_drop` | `OVERLAP_DROP_THRESHOLD` (default `1.8`). |
| `obs_z` | `OBS_Z_THRESHOLD` (default `2.5`). |
| `break_rotation_factor` | `BREAK_ROTATION_FACTOR` (`4.0`), the break verdict's rotation-ratio threshold. |
| `break_shared_fraction` | `BREAK_SHARED_FRACTION` (`0.2`), the break verdict's fraction of the median shared and nearby shared counts. |
| `break_window` | `BREAK_WINDOW` (`8`), the images on each side of an edge that its nearby shared count reads. |

Top-level fields in reconstruction mode, beside `thresholds` and `sequences`:

| Field | Description |
|-------|-------------|
| `break_count` | Number of discontinuities classified as breaks, over all sequences. `0` is the verdict "no breaks". |
| `breaks` | One `{"sequence": name, "frame_a": A, "frame_b": B}` object per break, in sequence and frame order. |

Each entry in `sequences`:

| Field | Description |
|-------|-------------|
| `pattern` | Sequence pattern from `summarize_paths_by_sequence` (e.g. `frame_%04d.jpg`). |
| `name` | Base name derived from the pattern (e.g. `frame`). |
| `frame_count` | Number of frames analyzed for this sequence. |
| `median_trans` | Median translation between successive frames in this sequence. |
| `median_rot_deg` | Median rotation (degrees) between successive frames. |
| `pose_trans_threshold` | Resolved translation threshold = `pose_trans_factor × median_trans`. |
| `pose_rot_threshold` | Resolved rotation threshold (degrees) — currently equal to `thresholds.pose_rot_deg`. |
| `rotation_per_frame_deg` | Median of rotation / frame-number gap over consecutive images (Step 4). |
| `median_frame_gap` | Median frame-number gap between consecutive images. |
| `median_shared_points` | Median shared 3D point count between consecutive images. |
| `break_shared_points_threshold` | `break_shared_fraction × median_shared_points`; an edge needs fewer shared points than this to be a break. |
| `median_nearby_shared_points` | Median nearby shared count over the sequence's edges (Step 4). |
| `break_nearby_shared_points_threshold` | `break_shared_fraction × median_nearby_shared_points`; an edge needs a lower nearby shared count than this to be a break. |
| `break_count` | Number of this sequence's discontinuities classified as breaks. |
| `frames` | Per-frame array (length = `frame_count`). |
| `discontinuities` | Per-clustered-edge array (one entry per `core_edge`). |
| `segments` | Segment partition derived from `discontinuities`. |

Each entry in `frames` (corresponds to one row of the per-frame table):

| Field | Description |
|-------|-------------|
| `seq_index` | 0-based index into this sequence's frame list. |
| `frame_number` | File number from the sequence pattern (e.g. `123` for `frame_0123.jpg`). |
| `image_name` | Image filename as stored in the reconstruction. |
| `left_trans_err` | Translation residual from the left-side extrapolation, or `null` near the start of the sequence. |
| `left_rot_err` | Rotation residual (degrees) from the left-side extrapolation, or `null`. |
| `right_trans_err` | Translation residual from the right-side extrapolation, or `null`. |
| `right_rot_err` | Rotation residual (degrees) from the right-side extrapolation, or `null`. |
| `edge_translation` | Euclidean distance between camera centers at `(i-1, i)` — the landing edge. `null` at `seq_index == 0`. |
| `edge_rotation_deg` | Rotation angle (degrees) at the landing edge `(i-1, i)`. `null` at `seq_index == 0`. |
| `step_ratio` | Step-size ratio for the landing edge `(i-1, i)`, or `null` at `seq_index == 0`. |
| `overlap_drop` | Covisibility drop factor for the landing edge, or `null` at `seq_index == 0`. |
| `obs_z` | Per-frame observation-count z-score, or `null` when the window is too short. |
| `flags` | Array of fired codes (subset of `["L.t", "L.r", "R.t", "R.r", "Step", "Cov", "Obs"]`). |

Each entry in `discontinuities` (one per `core_edge` after clustering):

| Field | Description |
|-------|-------------|
| `edge` | `[a, b]` — 0-based sequence indexes of the two endpoints. |
| `frame_a`, `frame_b` | File numbers of the endpoint images. |
| `image_a`, `image_b` | Filenames of the endpoint images. |
| `translation` | Euclidean distance between the two camera centers. |
| `rotation_deg` | Rotation angle (degrees) between the two camera orientations. |
| `step_ratio` | Step-size ratio at this edge, or `null` near sequence ends. |
| `overlap_drop` | Covisibility drop factor at this edge, or `null` near ends. |
| `obs_z_a`, `obs_z_b` | Per-frame obs z-scores for the two endpoints, or `null`. |
| `shared_points` | Number of 3D points observed in both endpoint images. The counts come from `build_covisibility_pairs` in [`_image_pair_graph.py`](../../../src/sfmtool/_image_pair_graph.py), which drops pairs whose viewing directions differ by more than 90°, so `0` means either no shared points or a pair removed by that angle filter. |
| `reproj_error_a`, `reproj_error_b` | Mean reprojection error (pixels) for each endpoint image, or `null` if not available. |
| `signals` | Array of fired primary codes (subset of `["P", "S", "C", "O"]`, sorted). `P` aggregates the four pose-residual flags. |
| `confidence` | `"high"` if `len(signals) >= 2`, else `"low"`. Matches the footer partition rule. |
| `frame_gap` | `frame_b - frame_a`. |
| `expected_rotation_deg` | `rotation_per_frame_deg × max(frame_gap, median_frame_gap)`. |
| `rotation_ratio` | `rotation_deg / expected_rotation_deg`, or `null` when the expected rotation is 0 and the edge turns (an infinite ratio). |
| `nearby_shared_points` | The edge's nearby shared count: 3D points seen both in the `break_window` images up to the edge and in the `break_window` images after it, counting other sequences' images at those frame numbers. |
| `is_break` | Whether the break verdict calls this edge a break: `rotation_ratio > break_rotation_factor`, `shared_points < break_shared_points_threshold` and `nearby_shared_points < break_nearby_shared_points_threshold`. |

`segments` — partition of `[0..frame_count-1]` at each discontinuity edge.
Frame `a` of edge `(a, b)` ends the preceding segment; frame `b` starts the
next. If `discontinuities` is empty, `segments` contains a single full-length
entry. Single-frame segments are emitted as-is — a discontinuity on the
first edge `(0, 1)` produces a singleton first segment `[0, 0]`, two
adjacent discontinuities at `(3, 4)` and `(4, 5)` produce a singleton middle
segment `[4, 4]`. Consumers that require a minimum segment length should
filter.

| Field | Description |
|-------|-------------|
| `start_index`, `end_index` | Inclusive 0-based sequence indexes. |
| `start_frame`, `end_frame` | Inclusive file numbers of the endpoint frames. |
| `frame_count` | Number of frames in the segment. |

### Image-sequence mode

`thresholds`:

| Field | Description |
|-------|-------------|
| `ratio_lower` | Lower edge of the in-band ratio interval (`0.75`). |
| `ratio_upper` | Upper edge of the in-band ratio interval (`1 / ratio_lower`). |

Each entry in `sequences`:

| Field | Description |
|-------|-------------|
| `pattern` | Sequence pattern. |
| `name` | Base name. |
| `sample_count` | Number of non-superseded sample points emitted. |
| `samples` | Per-sample-point array. |

Each entry in `samples` (superseded points are dropped):

| Field | Description |
|-------|-------------|
| `frame_number` | File number of the local-flow source frame. |
| `frame_name` | Filename of the local-flow source. |
| `next_frame_name` | Filename of `i + 1`. |
| `stride_frame_name` | Filename of `i + N`, or `null` if no stride flow was computed. |
| `stride` | Effective stride used at this sample point. |
| `local_median_magnitude` | Median in-bounds magnitude of the local flow. |
| `stride_median_magnitude` | Median in-bounds magnitude of the stride flow, or `null`. |
| `expected_magnitude_ratio` | The stride used as the expected ratio (= `stride` when `stride >= 2`), or `null`. |
| `actual_magnitude_ratio` | Observed ratio `stride_median / local_median`, or `null`. |
| `in_bounds_pct` | Percentage of pixels whose scaled local flow stays in-bounds, or `null`. |
| `classification` | `null` (in-band) or one of `"strong deceleration"`, `"deceleration"`, `"acceleration"`, `"strong acceleration"` — same band table as the human-readable summary. |

Image-sequence mode emits per-sample metrics but no `segments` array. The
adaptive-stride shrink threshold drives the band classification, but it is
deliberately permissive (it must trigger on benign motion changes to keep the
stride small) and is not a reliable segment boundary. Choosing a robust
segmentation rule for raw image sequences is left as future work — see
specs/core/ for the discontinuity-detection design notes.

### Examples

Reconstruction mode, one discontinuity, classified as a break:

```jsonc
{
  "schema_version": 1,
  "mode": "reconstruction",
  "thresholds": {
    "pose_trans_factor": 3.0,
    "pose_rot_deg": 15.0,
    "step_ratio": 1.5,
    "overlap_drop": 1.8,
    "obs_z": 2.5,
    "break_rotation_factor": 4.0,
    "break_shared_fraction": 0.2,
    "break_window": 8
  },
  "break_count": 1,
  "breaks": [{"sequence": "frame", "frame_a": 101, "frame_b": 102}],
  "sequences": [
    {
      "pattern": "frame_%04d.jpg",
      "name": "frame",
      "frame_count": 5,
      "median_trans": 0.052,
      "median_rot_deg": 0.31,
      "pose_trans_threshold": 0.156,
      "pose_rot_threshold": 15.0,
      "rotation_per_frame_deg": 0.3,
      "median_frame_gap": 1.0,
      "median_shared_points": 410.0,
      "break_shared_points_threshold": 82.0,
      "median_nearby_shared_points": 900.0,
      "break_nearby_shared_points_threshold": 180.0,
      "break_count": 1,
      "frames": [
        {"seq_index": 0, "frame_number": 100, "image_name": "frame_0100.jpg",
         "left_trans_err": null, "left_rot_err": null,
         "right_trans_err": 0.013, "right_rot_err": 0.4,
         "edge_translation": null, "edge_rotation_deg": null,
         "step_ratio": null, "overlap_drop": null, "obs_z": null,
         "flags": []},
        // ... three more frames ...
        {"seq_index": 4, "frame_number": 104, "image_name": "frame_0104.jpg",
         "left_trans_err": 0.011, "left_rot_err": 0.3,
         "right_trans_err": null, "right_rot_err": null,
         "edge_translation": 0.054, "edge_rotation_deg": 0.29,
         "step_ratio": 1.1, "overlap_drop": 1.0, "obs_z": -0.4,
         "flags": []}
      ],
      "discontinuities": [
        {"edge": [1, 2], "frame_a": 101, "frame_b": 102,
         "image_a": "frame_0101.jpg", "image_b": "frame_0102.jpg",
         "translation": 0.45, "rotation_deg": 9.0,
         "step_ratio": 3.2, "overlap_drop": 8.1,
         "obs_z_a": -0.2, "obs_z_b": -3.1,
         "shared_points": 12,
         "reproj_error_a": 0.78, "reproj_error_b": 0.93,
         "signals": ["C", "O", "S"], "confidence": "high",
         "frame_gap": 1, "expected_rotation_deg": 0.3, "rotation_ratio": 30.0,
         "nearby_shared_points": 20, "is_break": true}
      ],
      "segments": [
        {"start_index": 0, "end_index": 1, "start_frame": 100, "end_frame": 101, "frame_count": 2},
        {"start_index": 2, "end_index": 4, "start_frame": 102, "end_frame": 104, "frame_count": 3}
      ]
    }
  ]
}
```

### Error behavior

- `--json` writes the file after analysis completes. If the file cannot be
  created (permission denied, parent does not exist), `click.ClickException`
  is raised. The printed analysis is not suppressed.
- When no numbered sequences are detected, no JSON file is written
  (image-sequence mode raises `ClickException` before reaching the writer;
  reconstruction mode echoes the no-sequences message and returns).
- When a sequence is skipped (e.g. fewer than four frames in reconstruction
  mode), it is omitted from `sequences`. The remaining sequences are still
  emitted.

### Tests

[`tests/test_motion_report.py`](../../../tests/test_motion_report.py) covers the
`--json` report. Most cases use the `seoul_bull_workspace` fixture, a `.sfmr`
of the 17 `seoul_bull_sculpture` images built at the ground-truth poses:

- **Planted discontinuity.** Every image with file number 11 or higher is moved
  50 units along x. The report has `schema_version == 1` and
  `mode == "reconstruction"`, one sequence with `frame_count == 17`, a
  discontinuity between frames 10 and 11 with `P` in its `signals`, and
  segments that partition the sequence (one more segment than discontinuities,
  frame counts summing to `frame_count`).
- **No discontinuity.** The unmodified reconstruction gives
  `discontinuities == []` and one segment covering `[0, frame_count - 1]`.
- **Thresholds.** The `thresholds` block echoes the module constants, and
  `pose_trans_threshold == 3.0 * median_trans`.
- **Strict JSON.** `json.dumps(report, allow_nan=False)` succeeds on the
  planted-discontinuity report.
- **Infinite overlap drop.** A synthetic three-frame sequence result whose
  landing edge has `overlap_drop = inf` serializes that field as `null` and
  still lists `"Cov"` in the frame's `flags`. A synthetic input is used because
  overlap drop needs `3 * OVERLAP_WINDOW` (48) frames and the checked-in
  datasets are shorter.
- **Segment edge cases.** `_segments_from_core_edges` is called directly for no
  edges, an edge at `(0, 1)` (singleton first segment), an edge at the last
  position (singleton last segment), and two adjacent edges (singleton middle
  segment).
- **Image-sequence mode.** `sfm motion` on `seoul_bull_sculpture` frames 1-4
  with `--no-adaptive --json` gives `mode == "image_sequence"`,
  `ratio_lower` and `ratio_upper` in `thresholds`, no `segments` field,
  `sample_count == len(samples)`, the documented fields on every sample, and a
  report that passes `json.dumps(allow_nan=False)`.
- **CLI.** `sfm motion <file>.sfmr --json PATH` writes a reconstruction-mode
  report, and a `--json` path whose parent directory does not exist exits
  non-zero with "Failed to write JSON report".

The console output and option handling of both modes are tested in
[`tests/test_motion.py`](../../../tests/test_motion.py).

[`tests/test_motion_breaks.py`](../../../tests/test_motion_breaks.py) covers the break
verdict:

- **Constructed sequences.** `_break_reference` and `_classify_break` are run on
  per-edge rotations, frame numbers and shared counts: a clean loop has no breaks; one
  pair that turns 12 times the usual rotation and shares 3 points is the only break; a
  large turn that keeps its points and a pair that shares nothing but turns as usual
  are not breaks; a photo set's large move whose nearby shared count is normal is not
  a break, and is one once that count is low; a gap of dropped frames crossed with a
  smooth turn is not a break;
  swapped frames with uneven gaps give no breaks and still find a real break among
  them; a closely numbered pair is allowed a typical pair's rotation; a photo set in no
  order has no breaks; and cameras that never turn give an infinite ratio to the one
  pair that does.
- **Nearby shared count.** `_nearby_shared_point_counts` matches a count made with
  sets from its definition on random tracks over an irregularly numbered sequence at
  three window sizes, counts a rig's other sensor at the same frame numbers, and
  skips images without a frame number.
- **Reconstructions.** The Seoul bull ground truth, which has points at infinity, has
  no breaks. A planted break at 10->11 (frames 11-17 turned 90° and the points the two
  images share removed) gives the same references and breaks after every point is
  stored as a direction (`w = 0`).
- **CLI.** On the ground truth the output has `Verdict: no breaks` and the report has
  `break_count == 0` and the break fields on the sequence; on the planted break it
  prints `Verdict: 1 break: 10->11` and a `BREAK` row, and the report lists that edge
  in `breaks`, with `is_break`, `frame_gap`, `expected_rotation_deg`,
  `rotation_ratio` and `nearby_shared_points` on its row.

## Implementation

| Part | Location |
|---|---|
| CLI command and option parsing | [`_commands/motion.py`](../../../src/sfmtool/_commands/motion.py) |
| Image-sequence analysis (adaptive stride, DIS flow via `compute_optical_flow` and `compute_optical_flow_with_init`) | [`motion/image_sequence.py`](../../../src/sfmtool/motion/image_sequence.py), [`motion/flow_stats.py`](../../../src/sfmtool/motion/flow_stats.py), [`motion/ratio_band.py`](../../../src/sfmtool/motion/ratio_band.py) |
| Image-sequence console output and flow images | [`visualization/_discontinuity_display.py`](../../../src/sfmtool/visualization/_discontinuity_display.py) |
| Reconstruction analysis (pose extrapolation, step ratio, overlap drop, obs z-score, edge clustering, break verdict) | [`motion/recon_discontinuity.py`](../../../src/sfmtool/motion/recon_discontinuity.py), [`motion/constants.py`](../../../src/sfmtool/motion/constants.py) |
| Reconstruction console output | [`motion/_recon_console.py`](../../../src/sfmtool/motion/_recon_console.py) |
| `--json` report | [`motion/report.py`](../../../src/sfmtool/motion/report.py) |
| Sequence detection | [`_path_summary.py`](../../../src/sfmtool/_path_summary.py) (`summarize_paths_by_sequence`) |
| Shared point counts | [`_image_pair_graph.py`](../../../src/sfmtool/_image_pair_graph.py) (`build_covisibility_pairs`) |

## Open Questions

- **Histogram comparison metric**: Histogram intersection, chi-squared distance, and
  earth mover's distance are all options. Which works best in practice for flow
  distributions?

- **Non-sequential data**: This design assumes images are ordered sequentially. For
  unordered image sets, the "adjacent" concept doesn't apply. Should there be a mode that
  analyzes all covisibility-connected pairs instead?

- **Segmentation output (image-sequence mode)**: Reconstruction mode emits
  segment boundaries in the `--json` report (see the JSON Output section).
  Image-sequence mode does not — the adaptive-stride shrink threshold is
  permissive enough that using it as a segment classifier would yield many
  false positives. A dedicated image-sequence segmentation rule (e.g. a
  tighter ratio band, requiring `min_stride` saturation, or a covisibility
  proxy that doesn't need an SfM solution) is still open.

# `sfm analyze` Command

## Overview

Runs a deep-analysis report on a `.sfmr` reconstruction: covisibility graphs,
frustum intersection, depth-range estimation, and per-image quality metrics.
Exactly one analysis mode must be selected per invocation.

For a quick summary of any sfmtool file, use `sfm inspect`
(see `specs/cli/reconstruction/inspect-command.md`).

## Command Syntax

```bash
sfm analyze <RECONSTRUCTION.sfmr> (--coviz | --z-range | --frustum | --images | --metrics | --depth-reliability) [OPTIONS...]
```

`RECONSTRUCTION` must be a `.sfmr` file.

## Analysis Modes

Exactly one is required:

| Mode | Description |
|------|-------------|
| `--coviz` | Covisibility graph: pairs of images sharing 3D points |
| `--z-range` | Per-image Z depth ranges and histograms from stored depth statistics |
| `--frustum` | Frustum intersection graph: images whose viewing volumes overlap |
| `--images` | Per-image connectivity table: observations, distances, closest images, graph metrics |
| `--metrics` | Per-image quality metrics: reprojection error, track length (see below) |
| `--depth-reliability` | Per-point triangulation conditioning (inverse-depth z-score, condition number) beside the point-or-bearing likelihood-ratio test (see below) |

`--coviz` and `--frustum` are backed by the pair-graph builders in
`crates/sfmtool-core/src/analysis/image_pair_graph.rs` — covisibility from
shared track points, frustum overlap by Monte Carlo intersection of
depth-histogram-sized view frustums. See
[`specs/core/analysis/image-pair-graph.md`](../../core/analysis/image-pair-graph.md) for the
algorithms and parameter semantics.

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--range / -r` | string | | Range expression for file numbers (only with `--metrics`) |
| `--near-percentile` | float (0–100) | 5.0 | Near depth percentile (only with `--frustum`; must be less than `--far-percentile`) |
| `--far-percentile` | float (0–100) | 95.0 | Far depth percentile (only with `--frustum`) |
| `--samples` | int (min 100) | 100 | Monte Carlo samples per frustum pair (only with `--frustum`) |
| `--sigma-px` | float (finite, > 0) | measured | Per-axis pixel noise the point-or-bearing test weights its rays by (only with `--depth-reliability`) |
| `--depth-likelihood-ratio-threshold` | float (finite, ≥ 0) | 25 | The threshold the test's verdict (`is_finite`) applies: a point is finite when its bearing cost and either its depth score or its midpoint bound reach it (only with `--depth-reliability`) |

## Depth Reliability (`--depth-reliability`)

The report has two parts. The first is the inverse-depth z rule, which the
bench still decides finite points against bearings with; the second is the
likelihood-ratio test specified in
[core/reconstruction/batch-triangulation-api.md](../../core/reconstruction/batch-triangulation-api.md)
§ "Point or bearing", which reclassification (`sfm xform
--classify-points-at-infinity`) and discovery (`--find-points-at-infinity`)
decide with and the others are to move to
(the [amendment draft](../../drafts/point-or-bearing-likelihood-ratio.md) says
what moves when). Neither part changes the file. The second part's
disagreements with the stored representation are what reclassification would
change, and it says so. The printer is
[`analyze/point_or_bearing.py`](../../../src/sfmtool/analyze/point_or_bearing.py),
over the bindings `SfmrReconstruction.point_or_bearing_scores` and
`classify_points_at_infinity`.

### The inverse-depth z rule

`triangulation_diagnostics(noise_px=1.0)` over the finite points with two or
more observations: the median, mean and range of the inverse-depth z, the count
below the cutoff of 4, its histogram (clipped at the 99th percentile), and the
condition number of the normal matrix with a log-scale histogram. The per-ray
noise is `max(point error, 1 px) / f`. Points at infinity have no stored depth
and are left out of this part. A camera with a non-positive or non-finite focal
length prints a warning, since its points' z collapses to about 0 while the
condition number, being purely geometric, looks normal.

### The point-or-bearing test

- **Noise level.** The σ the rays are weighted by, and where it came from: the
  measured reprojection noise, the number of observations of finite points it
  was measured over and the number left out of it as outliers (see
  batch-triangulation-api.md § "The measured noise level"), and, with more than
  one camera, each camera's value and count. With `--sigma-px` the given value is used and the measured one is
  printed beside it. A reconstruction with no observation of a finite point has
  no measured value, and one whose `.sift` files cannot be read cannot be
  measured; without `--sigma-px` the part then prints `Unavailable` and why.
  A NaN or infinite `--sigma-px` or threshold is a usage error.
- **Threshold**, and the count of points left unscored because fewer than two
  of their observations give a usable ray (omitted when zero).
- **One block per stored representation**, finite points and points at
  infinity: the median, minimum and maximum depth score, the count with each
  verdict, and a histogram of `log10` of the depth score (scores below 1 clipped
  to 1, since a track whose rays spread over a wide angle scores near 0 and the
  scores span several orders of magnitude). The verdict that agrees with the
  stored representation is listed first. The finite count says how many were
  decided on the midpoint bound alone (depth score below the threshold), and
  the bearing count how many have a bearing cost under the threshold (which
  bounds `Λ`, so the bearing verdict needs nothing else for them).
- **Disagreements with the stored representation**: the count of finite points
  the test calls bearings and of points at infinity it calls finite. Each list
  shows at most 20 rows, the finite points with the lowest depth score first and
  the points at infinity with the highest first, then `... and N more`. A row
  is the point ID (`pt3d_<hash>_<index>`, which `sfm inspect` takes), its views,
  depth score, midpoint bound, `Λ`, the current rule's `z`, and the fitted
  distance. `Λ` and the distance come from the plain least-squares fit of both
  models (`fit=True`), and only the listed points are fitted, since the fit
  iterates and the decision does not need it. The distance is from the
  observing cameras' centroid, the fit's anchor (`1 / inverse_depth`). For a
  finite point `z` is the first part's value: at the stored point, with
  per-point noise `max(stored error, 1 px)`. A point at infinity has no stored
  depth, and its stored error is the bearing's residual, so its `z` is read
  from a copy of the reconstruction with the fitted points put in and their
  errors recomputed there: the z rule at the fitted point, with noise
  `max(error at the fitted point, 1 px)`. That is the `z` the z rule would
  give the point the test places, and it says whether that rule would call it
  a bearing. `z` prints `n/a` and the distance `inf` when the fit leaves
  the point at infinity (inverse depth 0), which happens to points the test
  calls finite only through a low threshold (at
  `--depth-likelihood-ratio-threshold 0` the seoul bull ground truth lists 11
  such rows). `z` also prints `n/a` if the errors at the fitted points cannot
  be recomputed for want of pixels, which cannot happen once the points have
  been scored, since scoring reads the same pixels. Numbers too wide for their
  column print in exponent form, and noise levels to four significant
  figures.
- **What reclassification would do**: `classify_points_at_infinity` run at the
  same noise level. The changes it would make come first: its promotions and
  demotions, and, when there are any, the finite points it would move off a
  stored position behind or on top of a camera (`refitted`). Then the points
  it declines to change, each count shown only when not zero: a finite point
  whose bearing is behind a camera (left finite), a point at infinity whose
  fit gives no usable point (left a bearing), and a finite point stored
  behind or on top of a camera with no usable point and its bearing behind a
  camera too (left where it is, `left_unusable`). Reclassification decides at the default threshold, so with
  `--depth-likelihood-ratio-threshold` the line says it decides at 25, and
  its counts can differ from the disagreements listed. It prints
  `unavailable` and the reason when the pixels cannot be read.

Both parts are cheap: the score is closed-form per track, and the only fits are
the listed rows.

On the Kerry Park ground-truth candidate `tk117` (12 of 387 points at
infinity), the measured noise is 0.2156 px over 3,510 observations with none
left out (0.2155 and 0.2157 px for the two lenses), no finite point is called a
bearing, 12 finite points are finite on the midpoint bound alone, and three
points at infinity are called finite:

```
    Points at infinity the test calls finite: 3
      Point              Views        Score        Bound       Lambda       z    Distance
      pt3d_a9665942_298     10        132.5        132.1        132.5    2.18       511.1
      pt3d_a9665942_294     21         83.6         77.9         83.6    1.82     1,076.5
      pt3d_a9665942_295     18         31.1         13.2         31.0    1.10     1,160.3
    Reclassification would promote 3 and demote 0
```

The z rule's `z` at each fitted point is under 4, so it would call all three
bearings. After `sfm xform --classify-points-at-infinity` the report on the
result lists no disagreement (the noise level is then 0.2149 px over 3,559
observations, the promoted points' among them). On the in-repo seoul bull
ground truth (0.4677 px over 1,229 observations, four mismatched keypoints of
7 to 16 px left out as outliers) the test agrees with every stored point; with
`--sigma-px 0.216` it calls bearing 188 finite (score 32.5), and
reclassification at that level would promote it.

## Per-Image Quality Metrics (`--metrics`)

Computes per-observation reprojection errors by projecting each observed 3D point through
the camera model (including distortion) and comparing against the observation's pixel
position. That position is the reconstruction's inline `keypoints_xy` row when the file
carries the column (every `embedded_patches` file does, and a `sift_files` file may), and
otherwise the feature's position in the image's `.sift` file. The computation is
`SfmrReconstruction::compute_observation_reprojection_errors` in
[`recompute.rs`](../../../crates/sfmtool-core/src/reconstruction/data/recompute.rs).

### Metrics per image

| Metric | Column | Description |
|--------|--------|-------------|
| Observation count | Obs | Number of track observations |
| Mean reprojection error | MeanErr | Mean per-observation error |
| Median reprojection error | MedErr | Median error (robust to outliers) |
| Max reprojection error | MaxErr | Worst single observation error |
| Mean track length | MeanTL | Mean observation count of observed 3D points |

### Output

Images are sorted by mean reprojection error (descending). Outlier flags:

- `!!` — mean error > 2× reconstruction median
- `!` — mean error > 1.5× reconstruction median
- `--` — zero observations (registered but contributing no points)

### Edge Cases

- **Zero-observation images**: Show 0 observations, N/A for error metrics.
- **Zero-point reconstructions**: Print a message and return.

## Usage Examples

```bash
# Covisibility analysis
sfm analyze solve.sfmr --coviz

# Per-image connectivity details
sfm analyze solve.sfmr --images

# Quality metrics to find problematic images
sfm analyze solve.sfmr --metrics

# Quality metrics for a subset
sfm analyze solve.sfmr --metrics --range 1-50

# Depth estimation for rendering
sfm analyze solve.sfmr --frustum --near-percentile 2 --far-percentile 98

# Finite points against bearings, at a given noise level
sfm analyze solve.sfmr --depth-reliability --sigma-px 0.25
```

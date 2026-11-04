# Frame Pair Matching

Before a video is solved, it helps to know how far the camera's view moved
between two of its frames, and whether that motion is one a solve can use.
Frame pair matching answers this from the SIFT features of the two frames
alone: it matches the features, reports the median distance the matched
features moved across the image, and reports three readings of the matches'
geometry. The first says how many of the matches are consistent with one
camera motion through a static scene. The second says how many of those are
also consistent with a camera that only turned, or with a scene that is one
plane. The third says how close that turning motion is to a pure rotation. A
caller choosing frames from a video reads the median distance to space its
choices and the three readings to tell a usable step from a moving scene or a
camera that turned on the spot.

The matching is the mutual best match of descriptors through one KD-forest per
frame, with Lowe's ratio test. It measures a step of any length, including the
long steps between frames chosen for a solve, at which dense optical flow loses
track of the motion
([pick-images-command.md § Why SIFT matching rather than optical flow](../../drafts/pick-images-command.md#why-sift-matching-rather-than-optical-flow)).

## Rust API

The types and the function live in
[pick_images/frame_pair.rs](../../../crates/sfmtool-core/src/analysis/pick_images/frame_pair.rs),
re-exported from `sfmtool_core::analysis::pick_images`, and are bound as
`sfmtool.analysis.FrameFeatures` and `sfmtool.analysis.match_frame_pair`.

```rust
/// One frame's features, with the forest over its descriptors built once.
pub struct FrameFeatures { /* positions, image size, forest */ }

impl FrameFeatures {
    /// `positions_xy` in pixels, `descriptors` `positions_xy.len() × 128`
    /// bytes, row-major, as a `.sift` file holds them, and the image's size.
    pub fn new(
        positions_xy: Vec<[f32; 2]>,
        descriptors: Vec<u8>,
        width: u32,
        height: u32,
        forest: KdForestParams,
    ) -> Result<Self, FramePairError>;
    /// The features `read_sift_features` read, with the image size from the
    /// same file's `read_sift_metadata`.
    pub fn from_sift(
        features: SiftFeatures,
        metadata: &SiftMetadata,
        forest: KdForestParams,
    ) -> Result<Self, FramePairError>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn width(&self) -> u32;
    pub fn height(&self) -> u32;
}

/// How `match_frame_pair` matches and what it fits.
pub struct FramePairOptions {
    pub ratio: f32,                        // 0.8, Lowe's ratio on distances
    pub min_matches: usize,                // 30; fewer gives no shares
    pub max_leaf_checks: Option<usize>,    // None: the queried forest's own
    pub fundamental_error_per_960px: f64,  // 1.0, scaled by the long side
    pub homography_error_per_960px: f64,   // 2.0
    pub seed: u64,                         // 0, for every RANSAC run
}

pub enum FramePairError {
    /// An option out of range: a ratio outside `(0, 1]`, a `min_matches`
    /// below 7, an error bound that is not positive and finite, or a forest
    /// parameter of 0, which the forest's build asserts against.
    Options { name: &'static str, reason: String },
    /// The descriptors are not `128 × positions_xy.len()` bytes, a position
    /// is not finite, there are more features than a forest indexes
    /// (`u32::MAX`), or a `.sift` file's features and metadata disagree on
    /// the count.
    FeatureArrays { reason: String },
    /// A width or height of 0.
    ImageSize { width: u32, height: u32 },
    /// The two frames' images differ in size.
    SizeMismatch { a: (u32, u32), b: (u32, u32) },
}

pub fn match_frame_pair(
    a: &FrameFeatures,
    b: &FrameFeatures,
    options: &FramePairOptions,
) -> Result<FramePairStats, FramePairError>;

pub struct FramePairStats {
    pub matches: usize,
    pub step_percent: f64,            // NaN with no matches
    pub f_inlier_share: Option<f64>,  // None under min_matches
    pub h_over_f: Option<f64>,        // None under min_matches F-inliers too
    pub rotation_error: Option<f64>,
}
```

**One forest per frame, built once.** A frame chosen from a video is matched
with the frame before it and with each candidate after it, so the forest over
its descriptors is built once in `FrameFeatures` and kept while the frame is
still to be matched. `match_frame_pair` borrows both frames and builds nothing.
The descriptors are not stored a second time: the forest keeps the copy it was
built from, and the queries read it back through `KdForest::points`.

**No `Progress` parameter.** Neither call runs long enough to need one
([STANDARDS.md § Long-running functions take a Progress](../../STANDARDS.md#long-running-functions-take-a-progress)).
Building one frame's forest takes about 3 ms for 8192 features and 50 ms for
100,000 on 32 cores, since a frame holds the features of one image, not of a
capture. A pair at 960 pixels with 8192 features per frame matches in 5 to
35 ms, and the slowest input, 8000 matches of which none fit one motion, runs
every RANSAC to its iteration cap in about half a second. A caller walking
many frames reports and checks for cancellation between pairs.

**Its own error type.** `match_frame_pair` and `FrameFeatures::new` return
`FramePairError` rather than an error type of a larger operation, because a
caller that matches two frames on its own has no capture, no file and no walk,
and an error type naming them would describe failures that call cannot have. A
larger operation wraps `FramePairError` with the frames it was raised for.
Every way the caller's input could make a kernel panic is a variant: the forest
build asserts on its lengths and its parameters, so `FrameFeatures::new`
checks them first
([STANDARDS.md § Caller input never panics](../../STANDARDS.md#caller-input-never-panics)).
Inputs that are valid but degenerate, such as every match at one place or on
one line, give statistics, not an error.

**The frames must be the same size.** The step is measured in percent of the
long side, and the rotation error places the principal point at the image
centre, so two frames of different sizes have no single image to measure
against. A caller comparing two cameras is outside what the statistics mean.

**Example.**

```rust
use sfmtool_core::analysis::pick_images::{match_frame_pair, FrameFeatures, FramePairOptions};
use sfmtool_core::features::kdforest::KdForestParams;
use sfmtool_sift_format::{read_sift_features, read_sift_metadata};

let frame = |path: &std::path::Path| -> Result<FrameFeatures, Box<dyn std::error::Error>> {
    let (_, metadata, _) = read_sift_metadata(path)?;
    let features = read_sift_features(path)?;
    Ok(FrameFeatures::from_sift(features, &metadata, KdForestParams::balanced())?)
};
let a = frame("frames/features/walk_0100.jpg.sift".as_ref())?;
let b = frame("frames/features/walk_0110.jpg.sift".as_ref())?;
let stats = match_frame_pair(&a, &b, &FramePairOptions::default())?;
println!("{} matches, step {:.1}%", stats.matches, stats.step_percent);
```

## Theory

### The step

The **step** between two frames is the median length of the displacements of
their mutual matches, `‖x_b − x_a‖` in pixels, divided by the long side of the
image and given in percent. Dividing by the long side makes it independent of
the resolution the frames were extracted at, so a step of 10% means the same
motion at 960 pixels and at full size. The median, not the mean, is taken so
that a minority of wrong matches, which can be anywhere in the image, does not
move it. It is computed over every mutual match, before any geometry is
fitted, since the step is wanted also for pairs whose geometry fails.

### The ratio test and mutual matches

Each feature of frame `a` is queried in `b`'s forest for its two nearest
descriptors, and each feature of `b` in `a`'s. A query's nearest neighbour is
kept when it is closer than `ratio` times the second nearest (Lowe's ratio
test), which drops a feature whose descriptor is about as close to two
features of the other frame, as on a repeated texture. The forest reports
squared distances, so the test compares them against `ratio²`. A query that
found only one neighbour has nothing to be compared with and keeps it.

A match is kept when it is mutual: feature `i` of `a` keeps `j` of `b` and
feature `j` keeps `i`. Each feature is then in at most one match, so a
feature of `b` that several features of `a` resemble counts once.

### The F-inlier share

The **F-inlier share** is the share of the mutual matches that are inliers of
one fundamental matrix, estimated by LO-RANSAC with the seven-point solver
([epipolar-estimation.md](../geometry/epipolar-estimation.md)) and an inlier
bound on the Sampson distance of `fundamental_error_per_960px` pixels per 960
pixels of the long side. A static scene seen by a moving camera puts nearly
every correct match on one fundamental matrix, so a low share means that much
of what was matched moved on its own, or that the matches are not to be
trusted.

**A homography's inliers are a fundamental matrix's.** When the camera only
turned, or the scene is one plane, the matches fit one homography `H`, and
then they fit every fundamental matrix `[e']ₓ H` for any epipole `e'`. A
seven-point sample of such matches has fewer than seven independent
constraints, and the solver returns nothing for it, so the estimator can fail
on exactly the captures with the cleanest matches. A homography is therefore
also fitted to all the matches, at the fundamental matrix's own bound, and its
inliers stand in for the fundamental matrix's when they are more. A match
within that bound of `H` is, to first order, within it of every `[e']ₓ H`:
the epipolar line of `x_a` under each of them passes through `H x_a`, so
`x_b` is no further from the line than from `H x_a`. On a tie the fundamental
matrix's inliers are used. When neither fits, the share is 0.

The seven matches of a minimal sample fit their own fundamental matrix exactly,
so the estimator accepts any consensus of at least seven, and the share
reports the best consensus however small. Under `min_matches` mutual matches
there is no share: a share of a handful of matches says nothing about the
motion.

### H/F

**H/F** is the share of the F-inliers that also fit one homography, fitted by
LO-RANSAC to the F-inliers alone, with an inlier bound on the symmetric
transfer error of `homography_error_per_960px` pixels per 960 of the long side.
A camera that moved through a scene with depth gives matches that fit a
fundamental matrix but no single homography, since nearer points move further
across the image than farther ones. A share near 1 means the matches carry no
parallax: the camera turned without moving, or everything matched lies on one
plane. It is computed only when there are at least `min_matches` F-inliers,
since a homography fits a few matches whatever they are; it is 0 when no
homography fits them.

### The rotation error

A camera that turned about its centre without moving maps the image through
`H = K R K⁻¹`, with `K` its camera matrix and `R` the rotation. So for the
right `K`, `M = K⁻¹ H K` is a rotation, and `M Mᵀ = I`. The **rotation error**
is the smallest Frobenius norm `‖M Mᵀ − I‖` over focal lengths `f` from 0.3 to
3 times the long side, with `K` the camera matrix of focal `f`, square pixels
and the principal point at the image centre, and `M` scaled to determinant 1
(a homography is defined up to scale). It is 0 for a camera that turned
without moving, and grows with the parallax the homography absorbed: a plane
seen from two places gives a homography that is not conjugate to any rotation.

The minimum is found in two stages. A grid of 80 focal lengths spaced evenly
in `log f` finds the basin, and a golden-section search in `log f` between the
best grid value's two neighbours, 40 steps long, finds its floor. The grid
alone would leave a pure rotation whose focal length falls between two grid
values an error that grows with the angle turned, up to about 0.015 for a turn
of 20°; with the refinement a pure rotation scores below `10⁻⁴` for any focal
length in the range and any turn up to 45°.

The error separates a camera that only turned, which has an H/F near 1 and an
error near 0, from a flat scene seen from two places, which also has an H/F
near 1 but a large error. The range of focal lengths covers fields of view
across the long side from about 118° down to about 19°. It is `None` where
H/F is `None`, where no homography fits, or where no focal length on the grid
gives a finite `M` with a non-zero determinant.

## Implementation notes

- **Inlier bounds scale with the image.** Both bounds are stated per 960
  pixels of the long side and multiplied by `long_side / 960`, so the same
  options mean the same angular tolerance at any resolution the frames were
  extracted at, as the step does.
- **One seed for every run.** The fundamental matrix, the homography standing
  in for it and the homography on the F-inliers all draw their samples with
  `options.seed`, and the forest is built from `KdForestParams::seed`, so a
  pair's statistics repeat exactly.
- **The homography on the F-inliers is fitted to them alone,** not to all the
  matches, so that a plane among the matches that do not fit the camera
  motion, such as the side of a passing vehicle, cannot become the homography
  H/F is measured against.

## Determinism and precision

Positions stay `f32`, as a `.sift` file stores them, and descriptors `u8`.
Each match's two positions are widened to `f64` once, because the two RANSAC
kernels and `sfmtool_core::numeric::median` take `f64`; widening an `f32` is
exact, and the arithmetic on a few thousand matches is negligible beside the
queries. The forest's squared descriptor distances are `f32` and exact, since
the largest, `128 · 255²`, is below `2²⁴`. The rotation error is computed in
`f64` because it subtracts the identity from `M Mᵀ`, whose entries are near 1
for the rotations it is meant to recognize.

The statistics are bit-identical for every thread count: the forest's trees
are each seeded, each query is independent of the others and of the order they
run in, and the RANSAC runs are sequential and seeded.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `ratio` | `0.8` | Lowe's ratio on descriptor distances, in `(0, 1]` |
| `min_matches` | `30` | the fewest mutual matches the shares are computed from, at least 7 |
| `max_leaf_checks` | `None` | the per-query budget; `None` uses the queried forest's `max_leaf_checks` |
| `fundamental_error_per_960px` | `1.0` | the Sampson inlier bound, pixels per 960 of the long side |
| `homography_error_per_960px` | `2.0` | the symmetric transfer inlier bound, pixels per 960 of the long side |
| `seed` | `0` | the seed of every RANSAC run |

The defaults are `FramePairOptions::default()`. The RANSAC confidence and
iteration cap are those of `FundamentalOptions::default()` and
`HomographyOptions::default()`.

## Python bindings

The bindings are in
[analysis/pick_images.rs](../../../crates/sfmtool-py/src/analysis/pick_images.rs).

```python
from sfmtool.analysis import FrameFeatures, match_frame_pair

a = FrameFeatures.from_sift("frames/features/walk_0100.jpg.sift")
b = FrameFeatures(positions_xy, descriptors, width, height)  # (N, 2) float32, (N, 128) uint8
stats = match_frame_pair(a, b, min_matches=30)
# {"matches": 412, "step_percent": 9.7, "f_inlier_share": 0.94,
#  "h_over_f": 0.53, "rotation_error": 0.34}
```

`FrameFeatures` takes the forest as `sfmtool.spatial.KdForest` does, a
`preset` with `num_trees`, `leaf_size`, `max_leaf_checks` and `seed`
overrides, all keyword-only. `FrameFeatures.from_sift` takes a `.sift` path or
a `SiftFile`, whose metadata it reuses, and reads the positions and
descriptors in Rust, so no feature array crosses into Python. A `float64`
`positions_xy` is accepted and narrowed to the `float32` the file stores.
`match_frame_pair` takes the fields of `FramePairOptions` as keyword-only
arguments; one left out takes the Rust default. It returns the statistics as a
dict, with `None` for a share it did not compute.

A `FramePairError` is a `ValueError` with the Rust message, a wrong dtype a
`TypeError`, and a `.sift` file that cannot be read the `OSError` of
[python-bindings.md § Format errors](../../python-bindings.md#format-errors).
The GIL is released while a forest is built and while a pair is matched.

## Testing

The Rust tests in
[frame_pair/tests.rs](../../../crates/sfmtool-core/src/analysis/pick_images/frame_pair/tests.rs)
build frames from random descriptors at synthetic positions:

- a known translation gives its step, and the step is the median, not the
  mean, of the displacements;
- a pure rotation's matches give an F-inlier share and an H/F near 1 and a
  rotation error below 0.01, and the rotation error is 0 for a rotation at a
  focal length on the grid, below `10⁻⁴` for focal lengths midway between grid
  values and turns up to 45°, and large for a zoom;
- a scene with depth seen from two places gives an F-inlier share near 1, an
  H/F below 0.5 and a rotation error above 0.05;
- fewer than `min_matches` matches give no shares, and no matches a NaN step;
- the ratio test drops a feature with two equally close neighbours, and a
  match that is not mutual is dropped;
- every error variant is returned for its input, and the bounds of each
  option are accepted;
- every match at one place, or on one line, gives statistics;
- the same seed gives the same statistics on a noisy pair with outliers.

[test_pick_images_rust_bindings.py](../../../tests/rust_bindings/analysis/test_pick_images_rust_bindings.py)
checks the bindings, their errors and their defaults, and matches neighbouring
images of the seoul_bull dataset from the `.sift` files a fixture extracts
into its own copy of the images, through a path and through a `SiftFile`.

## Non-goals

- Choosing frames from a video, which [pick-images.md](pick-images.md) does
  with these statistics.
- Matching two cameras of different image sizes, such as two sensors of a
  rig. A rig's motion is measured from one sensor's frames.
- Geometric verification for a solve. The matches are not stored, and a solve
  matches its images with `sfm match`.

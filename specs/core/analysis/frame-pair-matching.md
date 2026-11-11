# Frame Pair Matching

Before a video is solved, it helps to know how far the camera's view moved
between two of its frames. Frame pair matching answers this from the SIFT
features of the two frames alone: it matches the features and reports how
many matches it found and the median distance the matched features moved
across the image, the **step**. A caller choosing frames from a video reads
the step to space its choices and the number of matches to tell a step it can
follow from one across which the frames no longer show the same things.

The matching is the mutual best match of descriptors through one KD-forest per
frame, with Lowe's ratio test. It measures a step of any length, including the
long steps between frames chosen for a solve. The step and the count are
all it reports unless the caller gives a camera: then it also measures the
pair's **essential split**, the rotation between the two frames and the
**parallax** of the matches once that rotation is taken out
([The essential split](#the-essential-split)). The step mixes the two: a
camera that only turns moves every feature across the image and gives a long
step with no parallax, where a solve needs parallax to place its points.

**Why matching rather than optical flow.** Dense optical flow follows motion
from coarse to fine levels of an image pyramid, and over a gap of many frames,
where the image has moved by a tenth of its width or more, it loses track and
reports a fraction of the true displacement. Descriptor matching does not
depend on how far a feature moved, so it measures the step a walk takes
directly, rather than measuring short gaps and extrapolating. Where the
matches themselves fail, across a blurred run say, their count falls, so the
step says when it is not to be trusted, where the flow gives no such sign. The
cost is small for a caller that has the frames' `.sift` files anyway, as
[`sfm pick-images`](../../cli/image-feature/pick-images-command.md) does for
the readings and for the solve that follows: building two forests and
querying them takes a few milliseconds per pair.

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

/// How `match_frame_pair` matches.
pub struct FramePairOptions {
    pub ratio: f32,                        // 0.8, Lowe's ratio on distances
    pub max_leaf_checks: Option<usize>,    // None: the queried forest's own
    pub essential_split: Option<EssentialSplitOptions>, // None: no split
}

/// How the essential split is measured, and the camera it is measured with.
pub struct EssentialSplitOptions {
    pub camera: CameraIntrinsics, // at the frames' image size
    pub tolerance_px: f64,        // 3.0, the consensus bound at the focal
    pub min_inliers: usize,       // 12, at least 8
    pub samples: usize,           // 512, from 1 to MAX_RAY_SAMPLES (65,536)
    pub seed: u64,                // 0
}
impl EssentialSplitOptions {
    /// `camera` with the defaults above.
    pub fn new(camera: CameraIntrinsics) -> Self;
}

pub enum FramePairError {
    /// An option out of range: a ratio outside `(0, 1]`, a forest parameter
    /// of 0, which the forest's build asserts against, or an essential split
    /// option out of range (`essential_split.camera` for a focal length that
    /// is not positive and finite, a principal point that is not finite or
    /// rolling-shutter rates the camera cannot apply).
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
    /// The essential split's camera is of another size than the frames.
    CameraSize { camera: (u32, u32), frames: (u32, u32) },
}

pub fn match_frame_pair(
    a: &FrameFeatures,
    b: &FrameFeatures,
    options: &FramePairOptions,
) -> Result<FramePairStats, FramePairError>;

pub struct FramePairStats {
    pub matches: usize,
    pub step_percent: f64,            // NaN with no matches
    pub essential_split: Option<EssentialSplitOutcome>, // None: no camera given
}

/// The essential split of matched positions, `points_a[i]` matching
/// `points_b[i]`, in pixels of `options.camera`.
pub fn measure_essential_split(
    points_a: &[[f64; 2]],
    points_b: &[[f64; 2]],
    options: &EssentialSplitOptions,
) -> Result<EssentialSplitOutcome, FramePairError>;

pub enum EssentialSplitOutcome {
    Measured(EssentialSplit),
    NotMeasured { matches: usize, reason: SplitNotMeasured },
}

pub struct EssentialSplit {
    pub matches: usize,          // the matches considered
    pub inliers: usize,          // the essential matrix's inliers among them
    pub rotation_deg: f64,
    pub parallax_p50_deg: f64,   // over the inliers
    pub parallax_p90_deg: f64,
}

pub enum SplitNotMeasured { TooFewMatches, NoConsensus, NoPoseInFront }
```

The split lives in
[pick_images/essential_split.rs](../../../crates/sfmtool-core/src/analysis/pick_images/essential_split.rs),
re-exported beside the rest.

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
capture, and a pair at 960 pixels with 8192 features per frame matches in
under 35 ms. A caller walking many frames reports and checks for cancellation
between pairs.

**Its own error type.** `match_frame_pair` and `FrameFeatures::new` return
`FramePairError` rather than an error type of a larger operation, because a
caller that matches two frames on its own has no capture, no file and no walk,
and an error type naming them would describe failures that call cannot have. A
larger operation wraps `FramePairError` with the frames it was raised for.
Every way the caller's input could make a kernel panic is a variant: the forest
build asserts on its lengths and its parameters, so `FrameFeatures::new`
checks them first
([STANDARDS.md § Caller input never panics](../../STANDARDS.md#caller-input-never-panics)).

**The frames must be the same size.** The step is measured in percent of the
long side, so two frames of different sizes have no single long side to
measure against. A caller comparing two cameras is outside what the
statistics mean.

**The essential split is optional, and measured in the same call.** A pair's
split needs its matches, which `match_frame_pair` finds and does not keep, so a
caller that wants the split asks for it in the options, and it is measured
from the matches before they are dropped. Without a camera nothing more is
computed and `essential_split` is `None`. A caller with matched positions of
its own calls `measure_essential_split` directly. The camera must be of the
frames' image size, since its pixels are taken to be theirs; a caller with a
camera at another resolution scales it first.

**Not measured is an outcome, not an error.** Too few matches, no consensus
and no pose in front of the cameras happen on real pairs, across a blurred
run or a featureless wall, so a caller walking many pairs gets an
`EssentialSplitOutcome::NotMeasured` with its reason for that pair and goes
on. Only options out of range, and a camera of another size, are errors.

**What counts as enough matches is the caller's.** `match_frame_pair` reports
the count and leaves the threshold to the caller, since how many matches make
a step worth following depends on what the step is for;
[pick-images.md](pick-images.md) takes 600 by default for a step that passes
over frames, and none for the step to the next frame.

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
move it. It is NaN when there are no matches.

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
feature of `b` that several features of `a` resemble counts once. The
**matches** of the statistics are these mutual matches.

### The essential split

Given a camera, each match's two positions are turned into unit rays by the
camera's own `pixel_to_ray`, so a pinhole, a distorted and a fisheye camera
are each mapped by their model, and rays past 90° off the axis take part. A
match either of whose positions gives no finite ray is dropped; the rest are
the split's **matches**.

The essential matrix is fitted to the rays by
`geometry::relative_pose::estimate_essential_rays`
([relative-pose.md](../geometry/relative-pose.md)): seeded eight-ray samples,
a consensus whose bound on the angle between a ray and its epipolar plane,
on both sides, is `tolerance_px / f`, `f` the mean of the camera's two focal
lengths, and local optimization on the consensus. With the default of 3 pixels
that is the angle a 3-pixel match error covers at the image centre, where an
angle covers the fewest pixels for a pinhole or an equidistant fisheye. Its
**inliers** are the consensus.

`geometry::relative_pose::decompose_essential_rays` decomposes the matrix
into its two rotations and two translation signs and keeps, of the four
poses, the one that puts the most inliers in front of both cameras, depth
measured along each ray. The **rotation** is that pose's `R`, and its angle in
degrees is `rotation_deg`.

A match's **parallax** is the angle between its first ray turned into the
second frame, `R·ray_a`, and its second ray `ray_b`: the angle at which the
two rays meet, which is the triangulation angle of its point. A point at
infinity, or any point of a pure rotation, has none. `parallax_p50_deg` and
`parallax_p90_deg` are the 0.5 and 0.9 quantiles of the parallax over the
inliers, from `sfmtool_core::numeric`. A pair whose matches are mostly far
away, with a few near ones, reads a small P50 and a much larger P90. The
angle is computed as `atan2(|a × b|, a · b)`, which keeps its precision at the
hundredths of a degree a distant scene shows.

The split is **not measured** (`SplitNotMeasured`) when:

- `TooFewMatches`: fewer matches give rays than `min_inliers`;
- `NoConsensus`: no essential matrix has `min_inliers` inliers, as for
  matches that are mostly wrong;
- `NoPoseInFront`: none of the four poses puts an inlier in front of both
  cameras.

A pure rotation is measured, not refused: the epipolar constraint holds for
every translation direction there, the fitted matrix is `[t]×·R` for some
`t`, and its decomposition still gives `R`, so the parallax comes out near
zero, at the level of the match noise.

**Cost.** Most of it is the essential fit's 512 samples, each scored
against every match, so it grows with the matches: in a release build on a
32-core machine, about 3.4 ms for a pair of 750 to 820 matches at 360 × 640,
whose matching takes 1.2 ms, and 5 to 6 ms for 1,400 to 2,300 matches at
2040 × 1536, whose matching takes 4 ms. The rays take 0.2 to 0.3 ms and the
decomposition under 0.1 ms. It is spent only when a camera is given.

## Determinism and precision

Positions stay `f32`, as a `.sift` file stores them, and descriptors `u8`.
Each match's two positions are widened to `f64` once, for the displacement,
because `sfmtool_core::numeric::median` takes `f64`; widening an `f32` is
exact, and the arithmetic on a few thousand matches is negligible beside the
queries. The forest's squared descriptor distances are `f32` and exact, since
the largest, `128 · 255²`, is below `2²⁴`.

The statistics are bit-identical for every thread count: the forest's trees
are each built from `KdForestParams::seed`, and each query is independent of
the others and of the order they run in. The essential split is too: its
samples are drawn from `EssentialSplitOptions::seed`, its rays are computed
per match, and its parallax quantiles are taken over the inliers in match
order. It is computed in `f64`, the precision of the camera's ray map, and is
the same from run to run on one machine; agreement between platforms is not
promised, as for `estimate_essential_rays`.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `ratio` | `0.8` | Lowe's ratio on descriptor distances, in `(0, 1]` |
| `max_leaf_checks` | `None` | the per-query budget; `None` uses the queried forest's `max_leaf_checks` |
| `essential_split` | `None` | the split's options and camera; `None` measures no split |
| `essential_split.tolerance_px` | `3.0` | the consensus bound, pixels at the camera's focal; positive and finite |
| `essential_split.min_inliers` | `12` | the fewest inliers, and so matches, a split is measured from; at least 8 |
| `essential_split.samples` | `512` | the minimal samples the consensus draws; from 1 to 65,536, the estimator's `MAX_RAY_SAMPLES` |
| `essential_split.seed` | `0` | the sampler's seed |

The defaults are `FramePairOptions::default()` and
`EssentialSplitOptions::new(camera)`.

## Python bindings

The bindings are in
[analysis/pick_images.rs](../../../crates/sfmtool-py/src/analysis/pick_images.rs).

```python
from sfmtool.analysis import FrameFeatures, match_frame_pair

a = FrameFeatures.from_sift("frames/features/walk_0100.jpg.sift")
b = FrameFeatures(positions_xy, descriptors, width, height)  # (N, 2) float32, (N, 128) uint8
stats = match_frame_pair(a, b, ratio=0.8)
# {"matches": 412, "step_percent": 9.7, "essential_split": None}
stats = match_frame_pair(a, b, camera=camera)  # sfmtool.geometry.CameraIntrinsics
# {..., "essential_split": {"status": "measured", "matches": 412, "inliers": 377,
#   "rotation_deg": 4.1, "parallax_p50_deg": 0.6, "parallax_p90_deg": 2.3}}
# or {"status": "not_measured", "matches": 9, "reason": "too_few_matches"}
```

`FrameFeatures` takes the forest as `sfmtool.spatial.KdForest` does, a
`preset` with `num_trees`, `leaf_size`, `max_leaf_checks` and `seed`
overrides, all keyword-only. `FrameFeatures.from_sift` takes a `.sift` path or
a `SiftFile`, whose metadata it reuses, and reads the positions and
descriptors in Rust, so no feature array crosses into Python. A `float64`
`positions_xy` is accepted and narrowed to the `float32` the file stores.
`match_frame_pair` takes the fields of `FramePairOptions` as keyword-only
arguments, with the essential split's `camera` (a
`sfmtool.geometry.CameraIntrinsics`), `tolerance_px`, `min_inliers`, `samples`
and `seed` flattened beside `ratio` and `max_leaf_checks`; one left out takes
the Rust default, and a split option given without a camera is a
`ValueError`. It returns the statistics as a dict, `essential_split` `None`
without a camera and otherwise the outcome as the report serializes it, its
`status` `"measured"` or `"not_measured"`.

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
- no matches give a NaN step;
- the ratio test drops a feature with two equally close neighbours, and a
  match that is not mutual is dropped;
- every error variant is returned for its input, and the bounds of each
  option are accepted;
- a camera adds a measured split, with the rotation of a frame turned 5°,
  and leaves the matches and the step as they were; a camera of another size
  is refused.

The Rust tests in
[essential_split/tests.rs](../../../crates/sfmtool-core/src/analysis/pick_images/essential_split/tests.rs)
measure synthetic pairs of points 2 to 20 units away, seen before and after a
known motion, with up to 0.3 pixels of noise:

- a pinhole pair and an equidistant fisheye pair with rays past 90° give
  their rotation within 0.05° and the parallax quantiles of the true
  triangulation angles within 2%;
- a pure rotation gives its rotation and a parallax at the noise level;
- one match fewer than `min_inliers`, a position with no ray and no matches
  are `TooFewMatches`; unrelated positions are `NoConsensus`;
- the split repeats exactly for a seed, on a pair with wrong matches and
  few samples, where another seed changes it;
- every option out of range is refused by name, and the bounds are accepted.

[test_pick_images_rust_bindings.py](../../../tests/rust_bindings/analysis/test_pick_images_rust_bindings.py)
checks the bindings, their errors and their defaults, and matches neighbouring
images of the seoul_bull dataset from the `.sift` files a fixture extracts
into its own copy of the images, through a path and through a `SiftFile`.

## Non-goals

- Choosing frames from a video, which [pick-images.md](pick-images.md) does
  with these statistics, measuring the essential split of each step it
  chooses for its local density term.
- Matching two cameras of different image sizes, such as two sensors of a
  rig. A rig's motion is measured from one sensor's frames.
- Geometric verification of the matches for a solve. The essential split
  measures the motion; it removes no match, the step and the count are of
  all mutual matches, and a solve matches its images with `sfm match`.
- Finding the camera. The split is measured with the camera the caller
  gives; a wrong focal or a wrong lens model gives a wrong split.

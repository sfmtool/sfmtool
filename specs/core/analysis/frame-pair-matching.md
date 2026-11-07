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
long steps between frames chosen for a solve. Nothing is fitted to the
matches: the step and the count are all it reports.

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
}

pub enum FramePairError {
    /// An option out of range: a ratio outside `(0, 1]`, or a forest
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

## Determinism and precision

Positions stay `f32`, as a `.sift` file stores them, and descriptors `u8`.
Each match's two positions are widened to `f64` once, for the displacement,
because `sfmtool_core::numeric::median` takes `f64`; widening an `f32` is
exact, and the arithmetic on a few thousand matches is negligible beside the
queries. The forest's squared descriptor distances are `f32` and exact, since
the largest, `128 · 255²`, is below `2²⁴`.

The statistics are bit-identical for every thread count: the forest's trees
are each built from `KdForestParams::seed`, and each query is independent of
the others and of the order they run in.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `ratio` | `0.8` | Lowe's ratio on descriptor distances, in `(0, 1]` |
| `max_leaf_checks` | `None` | the per-query budget; `None` uses the queried forest's `max_leaf_checks` |

The defaults are `FramePairOptions::default()`.

## Python bindings

The bindings are in
[analysis/pick_images.rs](../../../crates/sfmtool-py/src/analysis/pick_images.rs).

```python
from sfmtool.analysis import FrameFeatures, match_frame_pair

a = FrameFeatures.from_sift("frames/features/walk_0100.jpg.sift")
b = FrameFeatures(positions_xy, descriptors, width, height)  # (N, 2) float32, (N, 128) uint8
stats = match_frame_pair(a, b, ratio=0.8)
# {"matches": 412, "step_percent": 9.7}
```

`FrameFeatures` takes the forest as `sfmtool.spatial.KdForest` does, a
`preset` with `num_trees`, `leaf_size`, `max_leaf_checks` and `seed`
overrides, all keyword-only. `FrameFeatures.from_sift` takes a `.sift` path or
a `SiftFile`, whose metadata it reuses, and reads the positions and
descriptors in Rust, so no feature array crosses into Python. A `float64`
`positions_xy` is accepted and narrowed to the `float32` the file stores.
`match_frame_pair` takes the fields of `FramePairOptions` as keyword-only
arguments; one left out takes the Rust default. It returns the statistics as a
dict.

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
  option are accepted.

[test_pick_images_rust_bindings.py](../../../tests/rust_bindings/analysis/test_pick_images_rust_bindings.py)
checks the bindings, their errors and their defaults, and matches neighbouring
images of the seoul_bull dataset from the `.sift` files a fixture extracts
into its own copy of the images, through a path and through a `SiftFile`.

## Non-goals

- Choosing frames from a video, which [pick-images.md](pick-images.md) does
  with these statistics.
- Matching two cameras of different image sizes, such as two sensors of a
  rig. A rig's motion is measured from one sensor's frames.
- Fitting a camera motion to the matches, or any other geometric
  verification. The matches are not stored, and a solve matches its images
  with `sfm match`.

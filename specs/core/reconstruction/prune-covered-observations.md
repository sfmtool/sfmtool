# Prune Covered Observations

A reconstruction built from features at many scales ends up holding, in the same
patch of the same photograph, a wide feature and a narrow one that describes the
same surface better. The wide one's support spans many pixels, so it averages
over whatever detail lies under it, and where that detail sits at more than one
depth its own triangulated depth is a blend of them. This operation retires
those coarse sightings, hands the structure over to the finer features that
supersede it, and drops the points left standing on too little.

It is a **subtraction, not a solve**. No point moves, no camera moves, no lens
moves, nothing is re-triangulated and nothing is adjusted. A surviving point
comes back with the position, patch frame, bitmap, colour and constraint it went
in with, and only its observation list is shorter. What a caller does afterwards
-- re-read the structure, adjust it, or neither -- is the caller's.

The rule it applies is [covered-by-finer.md](../analysis/covered-by-finer.md),
which knows nothing about a reconstruction; this is that rule asked of a value.

## Rust API

The operation lives in
[prune_covered.rs](../../../crates/sfmtool-core/src/reconstruction/prune_covered.rs),
bound as `EditedReconstruction.prune_covered_observations` on
`sfmtool._sfmtool.reconstruction`.

```rust
/// The rules one prune judges each observation by.
pub struct PruneCoveredOptions {
    pub footprint_fraction: f64,
    pub ratio: f64,
    pub min_fine_radius_px: f64,
    pub min_observations: usize,
}

/// What one prune did.
pub struct PruneCoveredReport {
    pub census: CoveredCensus,
    pub degenerate_rows: usize,
    pub protected_rows: usize,
    pub protected_rows_spared: usize,
    pub points_before: usize,
    pub points_after: usize,
    pub observations_before: usize,
    pub observations_after: usize,
    pub changed: bool,
    pub bands: Vec<PruneCoveredBand>,
}

pub fn prune_covered_observations(
    edited: &EditedReconstruction,
    options: &PruneCoveredOptions,
    progress: &Progress<'_>,
) -> Result<(EditedReconstruction, PointMap, PruneCoveredReport), PruneCoveredError>;
```

**The shape is a bulk edit's**, the same triple
[triangulation-rules.md](triangulation-rules.md)'s whole-value retriangulation
returns: the next value, the map from this value's point indexes to that one's,
and a report. `edited` is left exactly as it was. The map is not decoration --
this edit **drops** points, so a caller holding a selection, a copied id or a
constraint row has to follow it through.

**A fraction rather than a radius.** The one number a caller is likely to want
to move is how far an observation reaches, and it is stated as a fraction of the
observation's own projected extent rather than in pixels, because the right
answer scales with the feature: a wide feature claims a wide disk and a narrow
one a narrow disk, which is the whole content of the rule.

**Named errors, one sentence each**, because the caller is a menu entry that has
to say why the entry is greyed:

```rust
pub enum PruneCoveredError {
    NoPatchFrames,
    NoKeypoints,
    NoPosedImages,
    BadFootprintFraction(f64),
    Rule(CoveredByFinerError),
    Cancelled,
}
```

```rust
use sfmtool_core::progress::Progress;
use sfmtool_core::reconstruction::prune_covered::{
    prune_covered_observations, PruneCoveredOptions,
};

let (next, map, report) = prune_covered_observations(
    edited,
    &PruneCoveredOptions::default(),
    &Progress::none(),
)?;
println!(
    "{} of {} observations retired, {} points dropped",
    report.census.rows_removed,
    report.observations_before,
    report.points_before - report.points_after,
);
```

## Theory

### The two lengths, off one projection

Per observation the operation projects the point's patch frame into the
observing camera -- the algebra
[`patch_affine_shape`](../../formats/sfmr-file-format.md) states and that the
Track View reports -- and reads both of the rule's lengths off the result:

- the **radius** is the mean of the projected frame's two column norms, which is
  the same size measure a `.sift` affine shape yields for a `sift_files`
  reconstruction;
- the **footprint** containment is asked within is `footprint_fraction` times
  that radius.

One projection, two readings. A second measurement -- a stored scale, a `.sift`
file -- would be a second thing that could disagree with the geometry the value
actually holds.

### Why the fraction defaults to a half

A patch frame carries the extent it was **embedded** at, and that is several
feature sizes across. A reconstruction embedded at patch size 11 gives each
point a frame spanning 11 feature sizes edge to edge, so the projected radius is
5.5 of them. The half-extent a keypoint's *support* is stated at is 2.5 feature
sizes -- `PatchExtent`'s default sizing policy, and the number the surfel
occupies. On such a file the faithful footprint is therefore
`2.5 / 5.5 = 0.4545` of the projected radius, and the default rounds that up a
little.

The units hazard this dodges is real and was measured: reading containment at
`2.5 x` the projected radius, which is what a caller carrying the multiplier
over from a scale-domain rule would write, asks it over a disk more than five
times too wide and pairs features hundreds of pixels apart.

The embedding size is deliberately **not** read back out of the value's
metadata. A file carries no trustworthy statement of what it was embedded at,
and a guess at one would retire evidence on a number nobody measured. So the
fraction is a stated option with a default that is right to within about ten
percent on the common case, and a caller that knows its file states the exact
value.

### Why the fine-radius floor defaults to a pixel

The radius here is *measured*, not detected, and a measurement can collapse: a
patch frame seen almost edge-on projects to a sliver, and a frame of a point far
outside the capture's own scale projects to a fraction of a pixel. Such a row
satisfies every ratio test against everything, so without a floor it retires
whatever it lands on. On a measured 13-image reconstruction, 70 rows projecting
between 0.003 and 0.11 px drove 3.8 % of all the retirements. A pixel is the
natural bar: below it there is no image structure to have resolved.

### Points at infinity are in

The rule is image-space throughout. A point at infinity has a patch tangent to
the direction sphere, it projects to a footprint in every image that sees it,
and the pixels it claims are claimed just as thoroughly as a finite point's. So
it is read like any other, and it is retired or spared on the same evidence.

### Constrained points are protected

A point the value **ranges** or **holds** is a statement somebody made by hand:
a ground-truth pin, a measured distance. None of its observations is retired,
however well covered, because the operation must not quietly cost a reviewer the
structure they placed. It still **covers** other points' observations, because
the reason it was pinned says nothing about the evidence it offers its
neighbours. The report says how many rows carried the protection and how many of
those a passing pair would otherwise have retired, which is what the protection
actually bought.

### Nothing to do is an answer

A prune that retires nothing hands the input value straight back, with an empty
map and `changed` false, rather than producing an identical copy. A caller
pushing versions then has nothing to push, which is the honest outcome: a
version that changed nothing is a row in a history nobody can tell from one that
did something.

## Implementation notes

**The overlay is folded in first, and only when there is one.** An empty overlay
materialises to its own base, so the measurement reads the base directly and the
copy is not paid for. When there is an overlay, the materialisation's row map
becomes the first step of the chain the call returns.

**The map is stated, not scanned.** A whole-value edit that cannot say what it
did has its map read back off its two values by `RowMap::by_scan`, which
identifies a point by the images that see it. That is not good enough here: this
edit's whole effect is to drop points, and on a value whose points are all seen
by the same images the scan cannot tell a dropped point from the one after it,
so a selection would follow to the wrong point. The prune knows exactly which
indexes it dropped, so it builds `RowMap::by_removal` instead, which is exact in
both directions and costs a sort of the dropped list.

**The degenerate row states `NaN` for both lengths.** That is the enumeration's
documented "asks nothing" value on the reach, and on the radius it fails every
comparison, so such a row is neither retired nor able to retire anything. One
representation covers both halves of what a failed measurement should do, and
the report counts the rows it happened to.

**The radius is read off `f32` columns.** `observation_affine_shape` returns the
projected frame as `f32`, which is what every other consumer of that projection
reads, and the norms are taken from those values. A `f64` reading of the same
projection would disagree with the Track View about what an observation's size
is, and would move a handful of borderline pairs across the ratio bar for no
stated reason.

**The bands are anchored at the largest radius measured**, not at a fixed pixel
grid, so the table says where the prune bit relative to this value's own scale
spread. A point's band is its widest observation's, which is the band a reader
asking "what did this cost the coarse structure" means.

## Parameters

Defined by `PruneCoveredOptions::default()` in
[prune_covered.rs](../../../crates/sfmtool-core/src/reconstruction/prune_covered.rs).

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `footprint_fraction` | `0.5` | What fraction of an observation's projected patch radius its footprint is. |
| `ratio` | `2.0` | How many times finer the covering observation has to be. One octave. |
| `min_fine_radius_px` | `1.0` | A covering observation projecting below this says nothing. |
| `min_observations` | `2` | Surviving observations a point needs to be kept. |

## Python bindings

`EditedReconstruction.prune_covered_observations(*, footprint_fraction=0.5,
ratio=2.0, min_fine_radius_px=1.0, min_observations=2)` returns
`(EditedReconstruction, report)`.

The map travels inside the report, under `"map"`, as a `PointMap` -- the same
shape `commit` hands its map back in -- so the tuple stays the two the other
bulk edits on this class return. The report also carries `changed`,
`points_before`, `points_after`, `observations_before`, `observations_after`,
`degenerate_rows`, `protected_rows`, `protected_rows_spared`, `census` and
`bands`.

```python
after, report = edited.prune_covered_observations(footprint_fraction=0.4545)
if report["changed"]:
    value = after.materialize()[0]
    survivor = report["map"].forward(old_index)
```

Every refusal is a `ValueError` carrying the Rust error's sentence.

## Testing

`reconstruction/prune_covered/tests.rs` runs over a synthetic scene of three
cameras on a short arc and six points in three pairs -- a coarse feature with a
ten-times finer one two pixels away, the same pair pulled apart until it is out
of reach, and a same-scale pair sitting on top of one another -- so each
observation's projected radius is a known multiple of its neighbour's.

- **The covered coarse point is retired whole**, on every image that sees its
  cover, and the point goes with it; the map says the dropped index resolves to
  nothing and every survivor moved down by one.
- **A surviving point keeps everything but the rows**: position, frame, bitmap,
  colour, constraint.
- **The pairs the rule does not fire on are left alone**, and the same-scale
  pair is counted contained, so it is the scale test that spares it.
- **The input is left as it was**, and two prunes of one value agree.
- **A constrained point is never retired**, ranged or held, and the report says
  how many rows that spared; a protected row still covers.
- **Each option decides what it says it decides**: the fraction the reach, the
  ratio what counts as finer, the floor a collapsed projection, the bar which
  points survive.
- **The bands account for every row that projected**, every retirement and every
  dropped point.
- **An overlay is folded in first**, and the map carries an index through both
  steps.
- **Every refusal**: no patch frames, no inline keypoints, no posed images, an
  unusable fraction, and a cancelled prune, which writes nothing.
- **A point with no frame of its own** is counted degenerate and read past,
  covering nothing.

`tests/rust_bindings/test_edited_reconstruction_rust_bindings.py` holds the
binding over the 17-image solve converted to `embedded_patches`: the report
accounts for every row, a prune that retires nothing hands the value back, a
prune that bites shortens tracks while moving no point and keeping the survivors
in order, and pinning the longest tracks spares their rows.

## Non-goals

- **Re-solving anything.** No point is re-triangulated and no adjustment runs.
  Those are [triangulation-rules.md](triangulation-rules.md) and
  [bundle-adjust.md](bundle-adjust.md), asked for separately so a reader can say
  which of the three produced what they are looking at.
- **Adding an observation.** The operation only subtracts. Deciding which
  sightings a track should have is the bench's work.
- **Reading a second file.** Everything is read off the value: its frames, its
  pixels, its poses and its constraints. No `.sift` file and no workspace enters.
- **One shared lens.** Unlike the retriangulation, each observation projects
  through its own image's camera, so a value whose images are taken through
  several lenses is read rather than refused.

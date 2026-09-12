# Adding an observation to a track from a pixel

A reconstruction's track is the record of which photographs saw one 3D point and
where. A photograph that plainly shows the point but is missing from the track is
a gap a human eye can see and a matcher did not close, and closing it by hand
means naming the photograph and pointing at the pixel. That pixel is a seed
rather than an answer: a hand cannot place a keypoint to a fraction of a pixel,
and the reconstruction's accuracy depends on it being placed that well. This spec
describes the function that turns "this photograph sees this point, roughly
here" into an observation the rest of the toolkit cannot distinguish from one the
embed pass produced: the clicked pixel is registered photometrically against the
point's own stored patch, by the same kernel and at the same settings
`sfm embed-patches` runs, and the track is then re-triangulated with the new
sighting in it.

The function is pure. It takes the reconstruction value, the point, the image,
the pixel and the decoded views, and returns a new value plus a report; the base
behind the input's shared pointer is never written, and the caller's value is
left exactly as it was whether the edit succeeds or is refused.

It is defined on `embedded_patches` reconstructions only. An observation placed
at a clicked pixel has a keypoint and a patch behind it and no `.sift` feature
index, which is what an observation *is* in that mode and is not what one is in
`sift_files`
([`../../formats/sfmr-file-format.md`](../../formats/sfmr-file-format.md)); on a
`sift_files` value the call refuses rather than inventing a feature.

Related specs:
[`edited-reconstruction.md`](edited-reconstruction.md) (the value, `PointRecord`,
`replace_point`, and why a modification is delete-and-re-add),
[`../patch/patch-keypoint-localization.md`](../patch/patch-keypoint-localization.md)
and
[`../patch/keypoint-subpixel-refinement.md`](../patch/keypoint-subpixel-refinement.md)
(the two kernel stages this chains),
[`../../cli/reconstruction/embed-patches-command.md`](../../cli/reconstruction/embed-patches-command.md)
(the pass whose parameters are the defaults here), and
[`batch-triangulation-api.md`](batch-triangulation-api.md) (the solve the
re-triangulation runs).

## Rust API

The function lives in
[add_observation.rs](../../../crates/sfmtool-core/src/reconstruction/add_observation.rs),
re-exported as
`sfmtool_core::{add_observation, AddObservationOptions, AddObservationReport,
AddObservationError}`.

```rust
pub fn add_observation(
    edited: &EditedReconstruction,
    point: u32,
    image: u32,
    pixel: [f32; 2],
    views: &[ProjectedImage<'_>],
    options: &AddObservationOptions,
    progress: &Progress<'_>,
) -> Result<(EditedReconstruction, AddObservationReport), AddObservationError>;

pub struct AddObservationOptions {
    pub localize: KeypointLocalizeParams,
    pub refine: KeypointSubpixelParams,
    /// The leave-one-out ZNCC the new view has to reach.
    pub min_zncc: f64,
}

pub struct AddObservationReport {
    pub point: u32,
    pub replaced: u32,
    pub image: u32,
    pub clicked_pixel: [f32; 2],
    pub keypoint: [f32; 2],
    pub shift_px: f64,
    pub zncc: f64,
    pub observation_count: usize,
    pub position_shift: f64,
    pub from_infinity: bool,
    pub condition_number: f64,
}

pub enum AddObservationError {
    NotEmbeddedPatches,
    NoSuchPoint(u32),
    ImageOutOfRange { image: u32, image_count: usize },
    ImageAlreadyInTrack(u32),
    PixelOutsideImage { pixel: [f32; 2], size: (u32, u32) },
    NoPatchFrame(u32),
    ViewsMissing { got: usize, expected: usize },
    LocalizationRefused(u32),
    BelowAcceptanceBar { zncc: f64, bar: f64 },
    Triangulation,
    Edit(EditError),
}
```

`progress` is where this call names its two kernel stages, `localize` and
`refine`, so a caller can see which of them a slow registration spent its time
in. The kernels themselves take no `Progress`: each call registers one patch
over a handful of views with no stage inside it worth a row, and every other
caller runs them once per point inside a rayon loop, where a phase per call
would be per-item timing. `&Progress::none()` reports nothing, and the
registration it performs is identical either way
([../../gui/operation-progress.md](../../gui/operation-progress.md)).


### Why it is shaped this way

**A value in, a value out.** The edit is a function from one reconstruction value
to the next, which is what lets a caller keep both and undo by pointing at the
one it had
([`edited-reconstruction.md`](edited-reconstruction.md)). There is no `&mut` on
the input, so a refusal cannot leave a half-applied edit behind, and the returned
value shares the input's base `Arc`: the cost of the edit is the size of the one
track it touched.

**The decoded views are a named input rather than something the value carries.**
A reconstruction holds poses, lenses and thumbnails, and not the photographs.
The photometric fit needs the photographs, so the caller supplies one
[`ProjectedImage`](../../../crates/sfmtool-core/src/patch/normal_refine/params.rs)
per image of the base, indexed by image index, exactly as every other patch
kernel takes them. That keeps the function pure and keeps decoding and caching a
concern of whoever already has the pixels: the viewer has them in its image
panel, and an offline caller reads them the way `embed-patches` does.

**The report names the fit, not just the outcome.** `zncc`, `shift_px` and
`condition_number` are what say whether an accepted observation should be
trusted: a fit that scored 0.52 after walking 2.9 px is a different claim from
one that scored 0.98 after walking 0.3 px, and only the caller knows which it
wanted. `replaced` and `point` are both reported because a modification takes a
new index while remaining the same point, and a caller holding the old index
needs the new one.

**Every refusal names its subject.** The caller is a menu entry that has to say
in one sentence why nothing happened, so each variant carries the index, the
pixel or the score that did not hold, and `Display` writes that sentence.

### Example

```rust
use sfmtool_core::{add_observation, AddObservationOptions, EditedReconstruction};

let edited = EditedReconstruction::new(base);
let (next, report) = add_observation(
    &edited,
    42,                  // the point, by edited index
    7,                   // the image it is not yet seen in
    [120.5, 88.25],      // where the user pointed
    &views,              // one ProjectedImage per image
    &AddObservationOptions::default(),
)?;

assert_eq!(report.replaced, 42);
assert!(std::sync::Arc::ptr_eq(&edited.base, &next.base));
println!(
    "placed at {:?}, {:.2} px off the click, ZNCC {:.3}",
    report.keypoint, report.shift_px, report.zncc,
);
```

## What the call does

### 1. The refusals

The call refuses, producing nothing and touching nothing, when the base's
observations are `.sift` feature indexes, when the edited index names no live
point, when the image index is past the image table, when that image already
observes the point, when the pixel is outside the image's own sensor rectangle,
when fewer views were supplied than the base has images, or when the point
carries no patch frame. The order is the cheap checks first, so a caller greying a
menu entry gets the same answers without decoding anything.

### 1a. The point at infinity, which this edit makes finite

A point at infinity is a bearing: it has a direction and no distance, and a
second bearing is exactly what fixes one. So an observation added to such a
point does not merely join its track -- it carries the point across the
finite/infinity boundary, and the report says so in `from_infinity`.

Two things are different for it, and only two.

**The fit takes two passes.** The photometric fit anchors every view at the
point's own projection, and a bearing has none worth anchoring on: a `w = 0`
point projects into a second camera as the ray *parallel* to it rather than as
the place the surface is, so under any parallax those are different pixels and a
search anchored there would pull the sighting back onto the bearing's projection
and undo the very depth the click supplies. What the fit needs is a patch that
stands at a depth, and the click is exactly what supplies one:

1. **Provisional triangulation.** The track plus the clicked pixel is
   triangulated as it stands, giving a finite provisional position. A degenerate
   solve refuses here, before anything is rendered -- a click along the bearing
   itself states the same direction twice and fixes nothing.
2. **The fit**, over the finite patch that provisional position gives: the
   stored angular half-vectors scaled by the provisional placement distance,
   standing at the provisional position, with the same axes. That patch projects
   near the surface in both views, so the ordinary finite-point path runs
   unchanged -- the existing view seeded at its stored keypoint, the new one at
   the click, the same two kernel stages, the same acceptance bar -- and the
   report carries the ZNCC and the shift it really measured.
3. **The final triangulation**, from the track with the *fitted* keypoint in it,
   and the frame resized at **that** depth rather than at the provisional one:
   the scale is recomputed from the final position, so a fit that moved the
   sighting moves the patch's size with it.

The registration a created point gets this way is limited by its frame's
orientation rather than by the fit: a point created from one view carries a
fronto-parallel frame, which is a guess about the surface. That is a reason to
expect a pixel of residual on a slanted surface, not a reason to skip the fit --
the sighting still lands on the appearance the track is known by rather than
where the hand happened to fall.

**The frame is resized at the triangulated depth.** A `w = 0` point's stored
half-vectors are angular extents tangent to the direction sphere; the same
numbers on a finite point metres away would describe a patch the size of a
radian. They are multiplied by the placement distance -- the distance from the
camera-cloud centroid to the triangulated position, the same reference and the
same rescale
`SfmrReconstruction::materialize_points_at_infinity` applies -- so the patch
keeps the apparent size it had. The bitmap is kept: resizing the frame does not
change what the tile shows, and the tile is the appearance the point is known
by. The normal, zero on a `w = 0` row, becomes the resized frame's own, which is
the fronto-parallel surfel the tangent frame turns into.

`position_shift` is zero for such a point: it had no position to move from. The
confidence written to `observation_confidence` is the fit's own score, as for any
finite point. Everything else -- the refusals, the record, the
materialisation -- is the same call.

### 2. The patch the fit registers against

The point's stored frame is the patch. The two half-vector columns give the
in-plane axes and the world-space half-sizes, and the point's position is the
centre, so the
[`OrientedPatch`](../../../crates/sfmtool-core/src/patch/cloud.rs) the kernel
takes is read straight off the value with no refit. That is what makes the
observation an observation *of this point*: it is placed where the appearance the
track already agreed on is found again, not where a fresh detector fires.

### 3. The photometric fit

Two kernel stages, chained as `embed-patches` chains them.

The **discrete** stage is
[`localize_patch_keypoints`](../patch/patch-keypoint-localization.md) over a view
set that is the track's images followed by the new one. Every view the track
already holds is seeded at its stored keypoint and the new view at the clicked
pixel, which is what gives the new view a consensus to be scored against: a
single-view localization congeals against nothing and reports no leave-one-out
score at all.

The **sub-pixel** stage is
[`refine_patch_keypoints`](../patch/keypoint-subpixel-refinement.md), seeded at
the discrete answer, and only the new view's keypoint is read out of it. The
track's existing observations keep their stored pixels: this edit places one
sighting and moves none.

The new view has to survive the localizer's own gates, which are the ones that
decide membership in the embed pass, and then reach `min_zncc`, which defaults to
the localizer's absolute floor. A view the kernel dropped is
`LocalizationRefused`; one it kept below the bar is `BelowAcceptanceBar`, with
the score it reached.

### 4. Re-triangulation

The point is re-solved from **all** of its observations, the new one included, by
[`triangulate_batch`](batch-triangulation-api.md) over the world rays their
keypoints unproject to at the stored poses. This is the solve the patch spawn
runs, and it is refused on the same three signals: a non-finite position, an
infinite condition number (the depth is not observable, and the "position" is
then the minimum-norm point of the observable subspace rather than a
triangulation), or a solution behind one of the cameras that observe it. No
bundle adjustment runs.

### 5. What is carried over

The point's colour, normal, patch frame, patch bitmap, stored error and
constraint are the ones it had. The embed pass recomputes the frame, the normal
and the bitmap because it moves every keypoint of every track and so moves what
they were fused from; this edit moves no existing keypoint, so nothing they were
fused from has moved. Recomputing them here would also move the target after the
fit, which is to say it would change the appearance the observation was accepted
for agreeing with.

The record is produced through `replace_point`, so the point is deleted from the
base and re-added whole with its new track, `replaces` records the index it came
from, and a materialisation puts it back in that slot.

The added observation goes into the track at its position in image order, which
is the order the format stores a track in, so the value needs no re-sort and a
materialisation of it is a merge like any other.

## Implementation notes

**The observation's confidence, when the column exists.** A base carrying
`observation_confidence` needs a value for the new row, because a record must
carry exactly the base's columns. The value is the fit's leave-one-out ZNCC in
the column's byte scale, clamped to `[0, 1]` and scaled by 255. It is the only
per-observation quality this edit measures, and it is the quantity the column is
for.

**`NaN` clears no bar.** The localizer reports `NaN` for a view no round ever
scored. A bare `zncc < bar` test passes such a view, so the score is checked for
`NaN` first and refused.

**The re-triangulation reads the record, not the value.** The rays are built by
walking the record's observations after the new one has been inserted, so the
solve sees exactly the track the edit is about to store, and there is no second
place where the new sighting has to be remembered.

## Testing

`crates/sfmtool-core/src/reconstruction/add_observation/tests.rs` builds the
scene the localization kernel's own tests use, pinhole cameras looking down world
`+z` at a textured plane, and wraps it in an `embedded_patches` reconstruction
whose stored keypoints are the exact projections, so the truth for the added
observation is known to the pixel. What it pins:

- Each refusal, by its own variant: a `sift_files` base, an image already in the
  track, an image past the table, a point that was deleted, a pixel off the
  sensor on each side, too few views, and an acceptance bar no score can clear.
- A refusal leaves the input value's point count and overlay exactly as they
  were.
- The fit walks from a click two pixels off the truth to within a pixel of it,
  scoring above 0.9 against the same surface, and the track comes out in image
  order.
- Against a point displaced off the plane, whose stored observations are still
  the truth's projections, the re-triangulation moves the point closer to the
  truth than it was.
- The returned value shares the input's base by pointer, and the input value is
  unchanged.
- The colour, normal and patch frame come through untouched; the modification
  materialises back into the base index it replaced, with the added observation
  as the track's third row.
- On a point at infinity: the promotion to `w = 1`, the frame scaled at the
  placement distance, the bitmap kept and the normal taken from the resized
  frame, and a second sighting along the same bearing refused by the
  triangulation. Those live beside the create-point tests
  ([`create-point.md`](create-point.md)), since the point they act on is one a
  click created.

## Non-goals

- Moving or removing an observation the track already holds. Both are edits of
  their own.
- Adding an observation on a `sift_files` reconstruction, which would need the
  format to carry an observation with no feature behind it.
- Refitting the patch frame or the bitmap. Those are what the added observation
  is measured against. The frame of a point crossing from infinity is rescaled,
  which is a change of units rather than a refit, and the normal it then states
  is that frame read back.
- Bundle adjustment after the re-triangulation.
- Deciding whether the fit is good enough for a particular purpose. The report
  carries the score and the caller sets the bar.

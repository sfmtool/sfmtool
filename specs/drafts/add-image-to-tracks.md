# Add Image to Tracks

**Status:** Draft. Decided: the operation is a new core operation, its interface,
the measurements it takes and the judging rules it offers. Not decided: which
judging rule is the default, which is settled by the evaluation harness in
[`scripts/add_image_to_tracks/`](../../scripts/add_image_to_tracks/) and recorded
under "Evaluation" below. The viewer wiring (an image menu entry next to *Resect
Image*) is a separate step and is not part of this draft.

## Purpose

When an image's pose has been re-estimated, for example by *Resect Image*, the
image still observes only the tracks it observed before, and after a resection
of an image that had lost its observations that is often none. Add Image to
Tracks gives such an image its observations back. For every point of the
reconstruction that the image does not already observe, it checks whether the
point can be seen from the image, finds where the point's patch appears in the
image, checks that the appearance there agrees with the point's other
observations, and adds the observation when it does. Nothing else moves: the
point's position, patch frame and bitmap and every existing observation stay as
they are, and no triangulation or bundle adjustment runs. What a caller does
afterwards (retriangulate, adjust, or nothing) is the caller's.

It answers a narrower question than the bench's evaluation. The bench reads a
track by congealing all of its observations together, which moves every
keypoint and puts the new view into the consensus it is judged against. Here the
existing observations are the reference and are not touched; only the new view's
keypoint is searched for, against a consensus it did not contribute to.

## Why a core operation and not a composition

The pieces that exist do not compose into this cheaply or correctly:

- The bench's per-track path (`create_track`, then adding an observation, then
  `evaluate`) builds an editable track per point, runs one congealing round over
  every observation including the new one, and writes the track back with
  `commit`, which replaces the point and renumbers it. For every point of a
  reconstruction that is thousands of bench items. `evaluate` also scores every
  view against a leave-one-out consensus that includes the new view in the
  others' templates, which is not the question here.
- `EditedReconstruction::replace_point` per point writes through the overlay, so
  every touched point moves to a new index when the value is materialised.
- The localizer's consensus-basis *tail registration*
  ([keypoint-localization-consensus-basis.md](../core/patch/keypoint-localization-consensus-basis.md),
  Phase B) is exactly the right kernel: one search of one view against a
  finished consensus that view did not help build. But it is reachable only
  from inside a congealing run, where the basis views are first moved.

So the operation is new, and it reuses the kernels underneath rather than the
steps above them: the localizer's render-once tile, member localizability gate
and windowed-ZNCC shift search, the robust (IRLS) consensus, and the sub-pixel
ECC refinement against a fixed template.

## Rust API

The operation lives in
[add_image_to_tracks.rs](../../crates/sfmtool-core/src/reconstruction/add_image_to_tracks.rs)
and is bound as `EditedReconstruction.add_image_to_tracks` on
`sfmtool._sfmtool.reconstruction`. The per-point kernels live in
[reference.rs](../../crates/sfmtool-core/src/patch/keypoint_localize/reference.rs)
(the reference consensus, the search and the score) and in
[keypoint_subpixel.rs](../../crates/sfmtool-core/src/patch/keypoint_subpixel.rs)
(`refine_view_against_references`).

```rust
pub fn add_image_to_tracks(
    recon: &SfmrReconstruction,
    image: usize,
    views: &[Option<ProjectedImage<'_>>],
    options: &AddImageToTracksOptions,
    progress: &Progress<'_>,
) -> Result<(SfmrReconstruction, AddImageToTracksReport), AddImageToTracksError>;

pub struct AddImageToTracksOptions {
    pub rule: AcceptRule,
    pub min_zncc: f64,
    pub position_gate: PositionGate,
    pub template: TemplateSource,
    pub require_facing: bool,
    pub subpixel: bool,
    pub min_keypoint_separation_px: f64,
    pub localize: KeypointLocalizeParams,
    pub refine: KeypointSubpixelParams,
}

pub enum AcceptRule {
    FixedZncc,
    TrackBasis { statistic: BasisStatistic, pair: PairRule },
    PooledBasis { statistic: BasisStatistic },
}
pub enum BasisStatistic { Min, MedianMinusMad { k: f64 }, FractionOfMedian { fraction: f64 } }
pub struct PairRule { pub statistic: PairStatistic, pub factor: f64 }
pub enum PairStatistic { Min, Mean, Max }
pub enum PositionGate { Off, MaxPx(f64), ImageMad { k: f64, floor_px: f64 } }
pub enum TemplateSource { Rendered, StoredBitmap }
```

**Why it takes a plain reconstruction and returns one.** Adding observations
renumbers no point, so the operation is a bulk edit in the sense
`bundle_adjust` and `move_camera` are: a whole new value comes back, with every
point at the index it had. The binding materialises an `EditedReconstruction`
first, as `EditedReconstruction.bundle_adjust` does.

**Why `views` is one optional entry per image.** The target image must be to
hand, and so must the images of the observations used as references, because
the reference consensus and the leave-one-out scores are measured, not read.
An image whose photograph is not decoded is left out of every reference set
rather than failing the call, the rule `fuse_patch_cloud_bitmaps` follows.

**Why the rule is an enum.** The rule that decides which views to accept is the
open question the harness answers; the enum lets it run every candidate rule
through the same code. Once the evaluation has picked one, it is the default and
the others stay available for comparison.

**Why every candidate is reported.** Each point the image does not observe gets
one [`CandidateReport`] with its outcome (accepted, or refused with a named
reason) and the numbers the verdict was made on, so a caller can see why a point
was not added and a harness can re-judge without re-running.

Example:

```rust
use sfmtool_core::progress::Progress;
use sfmtool_core::reconstruction::add_image_to_tracks::{
    add_image_to_tracks, AddImageToTracksOptions,
};
let (next, report) = add_image_to_tracks(
    &recon, image, &views, &AddImageToTracksOptions::default(), &Progress::none(),
)?;
println!("{} of {} candidates added", report.accepted, report.candidates.len());
```

## What happens to one point

For a point `p` with existing observations in images `J`, and the target image
`t` not in `J`:

1. **Visibility.** The point's patch must project into `t`: in front of the
   camera (the localizer's projection, which leaves ray-path fisheye models to
   their own domain rather than testing `z`), inside the frame, not grazing
   (`|d̂·n̂| ≥ min_grazing_cos`), and, with `require_facing`, the target camera on
   the same side of the patch plane as the majority of the cameras in `J`. The
   plane's side is read from the existing observers rather than from the stored
   normal's sign, which no writer promises. A point at infinity has no plane
   side and skips that check.
2. **Reference consensus.** Each existing observation is rendered on the patch
   grid anchored at its own keypoint (the in-plane offset its keypoint states,
   exactly as the bench and the bitmap fuse anchor it). The renders are
   z-normalised and combined into the robust IRLS consensus. From the same
   renders come each reference's leave-one-out ZNCC (against the consensus of
   the others) and the pairwise ZNCCs between references. With
   `TemplateSource::StoredBitmap` the template is the point's stored bitmap
   instead, and the references still supply the leave-one-out and pairwise
   numbers.
3. **Search.** The target view's context tile is rendered once around the
   point's projection, the member localizability gate is applied to its own
   core, and one windowed-ZNCC shift search over `±search` patch-grid pixels
   finds the peak (the tail registration's search, run exhaustively because it
   is one view per point). A peak on the edge of the window is refused: the
   true peak lies outside what was searched.
4. **Sub-pixel.** With `subpixel`, the keypoint is refined by the ECC
   Gauss-Newton solve against the fixed reference consensus, moving only the
   target.
5. **Score.** At the final keypoint the target's core is rendered and its ZNCC
   against the consensus and against each reference is taken, by the same
   scorer that took the references' leave-one-out ZNCCs. The target never
   contributed to the consensus, so its ZNCC is a leave-one-out number and is
   comparable with theirs.
6. **Judge.** The rule decides (below). `min_zncc` is a floor every rule also
   applies (`0` disables it for the data-driven rules).
7. **Place.** The positional gate, then one observation per place: two accepted
   keypoints (or an accepted keypoint and an existing observation of the image)
   closer than `min_keypoint_separation_px` cannot both be the same pixel of two
   surfaces; the one with the higher ZNCC is kept and the other refused.

Accepted observations are written with their keypoint and, where the value has
the column, their ZNCC in `observation_confidence` on the byte scale the bench
commit uses (`round(255·clamp(z, 0, 1))`, raised to `1` because `0` means
unmeasured). The stored per-point error is left as it was.

## The judging rules

- `FixedZncc`: accept when the ZNCC is at least `min_zncc`. The baseline.
- `TrackBasis`: the point's own observations set the bar. With three or more
  references, the bar is a statistic of their leave-one-out ZNCCs: the minimum,
  the median minus `k` times the scaled median absolute deviation, or a
  fraction of the median. With exactly two references each one's leave-one-out
  ZNCC is just their pairwise ZNCC, so a statistic of two copies of one number
  says little; instead the target's ZNCC against each reference is compared
  with the ZNCC between the two references: the minimum, mean or maximum of the
  target's two pairwise ZNCCs must reach `factor` times the references' own.
- `PooledBasis`: the same statistics over the leave-one-out ZNCCs of every
  reference of every candidate point of this call, one bar for the whole image.
  Useful where tracks are short and a per-track basis is too thin.

## Positional gate

`MaxPx` refuses a keypoint more than that many source pixels from the point's
projection. `ImageMad` derives the bound from the call's own accepted
candidates: the pose error of a resected image moves every projection by a
similar amount, so the offsets of the photometrically accepted candidates are
pooled and a candidate is refused when its offset exceeds the median plus `k`
times the scaled MAD, and never below `floor_px`.

## Parameters

Defaults are those of `AddImageToTracksOptions::default()`, set after the
evaluation. See "Evaluation".

## Python binding

`EditedReconstruction.add_image_to_tracks(image, images, *, rule=..., ...)`
returns `(EditedReconstruction, report)`. `images` is one decoded image per
image of the reconstruction, or an `ImagePyramidSet`. The rule is spelled as a
string plus its parameters (`rule="track_basis"`, `basis="min"`, …), so a
script can sweep rules without constructing Rust enums.

## Testing

Unit tests in
[add_image_to_tracks/tests.rs](../../crates/sfmtool-core/src/reconstruction/add_image_to_tracks/tests.rs)
on a synthetic textured scene: an observation removed from a track is found
again within a small fraction of a pixel of where it was; a point behind the
camera, out of frame or back-facing is refused with that reason; a target whose
content has been replaced by another texture is refused by the photometric
rule; the existing observations, positions and bitmaps come back unchanged; the
confidence column is extended in lockstep. Binding tests in
`tests/rust_bindings/test_add_image_to_tracks_rust_bindings.py`.

## Evaluation

To be filled in from the harness.

## Non-goals

No retriangulation, no bundle adjustment, no new points. An occlusion test
against other geometry is not made: an occluded point is expected to fail the
photometric rule, because the target sees a different surface there.

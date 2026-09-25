# Add Image to Tracks

When an image's pose has been re-estimated, for example by resecting it, the
image still observes only the tracks it observed before, and an image that had
lost its observations observes none. Adding an image to tracks gives it its
observations back. For every point of the reconstruction that the image does not
already observe, it checks whether the point can be seen from the image, finds
where the point's patch appears in the photograph, checks that the appearance
there agrees with the point's other observations, and adds the observation when
it does. Nothing else moves: every point keeps its index, position, patch frame
and bitmap, every existing observation stays as it is, and nothing is
re-triangulated or adjusted. What a caller does afterwards (retriangulate,
adjust, or nothing) is the caller's.

It answers a narrower question than the bench's evaluation
([editable-track.md](../bench/editable-track.md)). The bench reads a track by
congealing all of its observations together, which moves every keypoint and puts
the view being judged into the consensus it is judged against. Here the existing
observations are the reference and are not touched; only the new view's
keypoint is searched for, against a consensus it did not contribute to. That is
the question the localizer's consensus-basis tail registration asks
([keypoint-localization-consensus-basis.md](../patch/keypoint-localization-consensus-basis.md),
Phase B), with the basis supplied rather than congealed.

## Rust API

The operation lives in
[add_image_to_tracks.rs](../../../crates/sfmtool-core/src/reconstruction/add_image_to_tracks.rs),
bound as `EditedReconstruction.add_image_to_tracks` on
`sfmtool._sfmtool.reconstruction`. Its per-point kernels are
`ReferenceConsensus` in
[reference.rs](../../../crates/sfmtool-core/src/patch/keypoint_localize/reference.rs)
(build the references' consensus, search one view against it, score one view at
a keypoint) and `refine_view_against_references` in
[keypoint_subpixel.rs](../../../crates/sfmtool-core/src/patch/keypoint_subpixel.rs).
The viewer's image menu entry is
[../../gui/edits/add-image-to-tracks.md](../../gui/edits/add-image-to-tracks.md).

```rust
pub fn add_image_to_tracks(
    recon: &SfmrReconstruction,
    image: usize,
    pyramids: &[Option<&ImageU8Pyramid>],
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
    pub ascend_on_edge: bool,
    pub min_keypoint_separation_px: f64,
    pub localize: KeypointLocalizeParams,
    pub refine: KeypointSubpixelParams,
}

pub enum AcceptRule {
    FixedZncc,
    TrackBasis { statistic: BasisStatistic, pair: PairRule },
    PooledBasis { statistic: BasisStatistic },
    PooledOrTrack { pooled: BasisStatistic, track: BasisStatistic, pair: PairRule },
}
pub enum BasisStatistic { Min, MedianMinusMad { k: f64 }, FractionOfMedian { fraction: f64 } }
pub struct PairRule { pub statistic: PairStatistic, pub factor: f64 }
pub enum PairStatistic { Min, Mean, Max }
pub enum PositionGate { Off, MaxPx(f64), ImageMad { k: f64, floor_px: f64 } }
pub enum TemplateSource { Rendered, StoredBitmap }

pub struct AddImageToTracksReport {
    pub image: usize,
    pub candidates: Vec<CandidateReport>,
    pub accepted: usize,
    pub observations_before: usize,
    pub observations_after: usize,
    pub pooled_bar: Option<f64>,
    pub position_bound_px: Option<f64>,
}
```

**Why it takes a plain reconstruction and returns one.** Adding observations
renumbers no point, so the operation is a bulk edit in the sense
[bundle-adjust.md](bundle-adjust.md) and [move-camera.md](move-camera.md) are:
a whole new value comes back, with every point at the index it had. The binding
materialises an `EditedReconstruction` first, as
`EditedReconstruction.bundle_adjust` does.

**Why it takes decoded photographs rather than posed views.** The target image
must be decoded, and so must the images of the observations used as references,
because the reference consensus and the leave-one-out scores are measured, not
read. Poses and cameras are read from `recon`, so a view can never disagree
with the value it is being added to. An image whose photograph is `None` is left
out of every reference set rather than failing the call, the rule
`fuse_patch_cloud_bitmaps` follows.

**Why the rule is an enum.** Several rules are useful, and they are compared
with each other by the leave-one-image-out harness in
[`scripts/add_image_to_tracks/`](../../../scripts/add_image_to_tracks/README.md);
the default is one of them.

**Why every candidate is reported.** Each point the image does not observe gets
one `CandidateReport` with its outcome (`refusal` is `None` for an added
observation, or a named `Refusal`) and the numbers the verdict was made on: the
projection, the searched and final keypoints, the offset from the projection,
the member localizability score, the ZNCC against the consensus and against
each reference, the references' leave-one-out and pairwise ZNCCs, and the
number the rule compared with its bar. A caller can see why a point was not
added, and a harness can re-judge without re-running. The refusals are
`no_patch`, `not_in_frame`, `grazing`, `back_facing`, `too_few_references`,
`unlocalizable`, `no_peak`, `peak_at_edge`, `unscorable`, `below_floor`,
`below_bar`, `too_far` and `shared_keypoint`.

The call itself is refused (`AddImageToTracksError`) for an image index out of
range, an unposed image, fewer pyramids than images, a target whose own
photograph is missing, a `sift_files` value (an added observation has no feature
index to name) and a value without patch frames.

```rust
use sfmtool_core::progress::Progress;
use sfmtool_core::reconstruction::add_image_to_tracks::{
    add_image_to_tracks, AddImageToTracksOptions,
};
let (next, report) = add_image_to_tracks(
    &recon, image, &pyramids, &AddImageToTracksOptions::default(), &Progress::none(),
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
   side and skips the grazing and facing checks.
2. **Reference consensus.** Each existing observation in a decoded image is
   rendered on the patch grid anchored at its own keypoint (the in-plane offset
   its keypoint states, as the bench and the bitmap fuse anchor it). The renders
   are z-normalised and combined into the robust IRLS consensus. From the same
   renders come each reference's leave-one-out ZNCC (against the robust
   consensus of the others) and the pairwise ZNCCs between references. Fewer
   than two references in frame is `too_few_references`. With
   `TemplateSource::StoredBitmap` the search template is the point's stored
   bitmap instead, on the bitmap's own grid, and the references still supply the
   leave-one-out and pairwise numbers.
3. **Search.** The target's context tile is rendered once around the point's
   projection and its own core is scored for localizability (`unlocalizable`
   above `max_member_keypoint_uncertainty`). One windowed-ZNCC shift search over
   `±search` patch-grid pixels then finds the peak: the tail registration's
   search, run exhaustively because it is one view per point. A peak on the edge
   of the window is `peak_at_edge`, because the true maximum may lie outside
   what was searched. With `ascend_on_edge`, such a window is searched again by
   the "+"-descent from the projection, and the local maximum it climbs to is
   taken when it is inside the window: a repeated texture can put a stronger
   correlation a period away, and the projection is the evidence for which
   period is meant.
4. **Sub-pixel.** With `subpixel`, the keypoint is refined by the ECC
   Gauss-Newton solve against the references' frozen consensus, moving only the
   target, with the solve's never-worse guard.
5. **Score.** At the final keypoint the target's core is rendered, and its ZNCC
   against the template and against each reference is taken by the same scorer
   that took the references' leave-one-out ZNCCs. The target never contributed
   to the consensus, so its ZNCC is a leave-one-out number and comparable with
   theirs.
6. **Judge.** The rule decides (below). `min_zncc` is a floor the basis rules
   also apply (`0` disables it); under `FixedZncc` it is the whole rule.
7. **Place.** The positional gate, then one observation per place: an accepted
   keypoint closer than `min_keypoint_separation_px` to an observation the image
   already has is refused, and of two accepted keypoints that close the one with
   the higher ZNCC is kept (`shared_keypoint`). Two points at one pixel of one
   image are two tracks of one surface, and adding the image to both would
   repeat that.

Accepted observations are written with their keypoint and, where the value has
the column, their ZNCC in `observation_confidence` on the byte scale the bench
commit uses (`round(255·clamp(z, 0, 1))`), raised to `1` because `0` means
unmeasured. Each track stays in image order. The stored per-point error is left
as it was, since nothing moved the point.

## The judging rules

- `FixedZncc`: accept when the ZNCC is at least `min_zncc`.
- `TrackBasis`: the point's own references set the bar. With three or more, the
  bar is a statistic of their leave-one-out ZNCCs: the minimum, the median minus
  `k` times the scaled median absolute deviation (MAD × 1.4826), or a fraction
  of the median. With exactly two, each reference's leave-one-out ZNCC is their
  pairwise ZNCC, so a statistic of two copies of one number says little;
  instead the minimum, mean or maximum of the target's ZNCC against each
  reference must reach `factor` times the ZNCC between the two references.
- `PooledBasis`: the statistic over the leave-one-out ZNCCs of every reference
  of every candidate that reached the verdict, which is one bar for the whole
  image.
- `PooledOrTrack`: accept a candidate that reaches either the pooled bar or its
  own track's bar (`TrackBasis` with `track` and `pair`).

**Why the default is what it is.** The default is `PooledOrTrack` with the
pooled bar at the median minus three scaled deviations, the track's at 0.9 of
its references' median (the pair rule at 0.9 of the mean for two), a floor of
0.5, and the `ImageMad` positional gate. Every bar is read off the call's own
data, so it follows the capture's texture and the pose's error rather than a
constant: a fixed ZNCC bar that suits one capture refuses good sightings on
another. The pooled bar is what keeps a track whose references barely agree with
each other from lowering the bar for itself; the track's own bar is what keeps a
good sighting on a surface harder than the rest of the image from being refused
for it. A new view's ZNCC runs lower than its references' leave-one-out numbers
for the same quality of match, because their keypoints were fitted together and
its keypoint was not, which is why no bar is the minimum of the references'.
The positional gate is what removes most wrong sightings that correlate well.

## Positional gate

`MaxPx` refuses a keypoint more than that many source pixels from the point's
projection. `ImageMad` derives the bound from the call's own photometrically
accepted candidates: a re-posed image's pose error moves every projection by a
similar amount, so their offsets are pooled and a candidate is refused when its
offset exceeds the median plus `k` scaled MADs, never below `floor_px`. With
fewer than three such candidates no bound is derived.

## Parameters

Defined in `AddImageToTracksOptions::default()` and mirrored as the binding's
keyword defaults.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `rule` | `PooledOrTrack { pooled: MedianMinusMad { k: 3.0 }, track: FractionOfMedian { fraction: 0.9 }, pair: { Mean, 0.9 } }` | The photometric rule |
| `min_zncc` | `0.5` | ZNCC floor under every rule |
| `position_gate` | `ImageMad { k: 3.0, floor_px: 1.0 }` | Positional bound |
| `template` | `Rendered` | Search template |
| `require_facing` | `true` | Back-facing check |
| `subpixel` | `true` | ECC sub-pixel step |
| `ascend_on_edge` | `false` | Ascent from the projection when the peak is on the window's edge |
| `min_keypoint_separation_px` | `1.0` | One observation per place |
| `localize.search` | `6.0` | Search radius, patch-grid px |
| `localize.resolution` | `24` | Patch grid (a stored bitmap's own grid under `StoredBitmap`) |
| `localize.max_member_keypoint_uncertainty` | `0.35` | Localizability gate `τ` |
| `localize.min_grazing_cos` | `0.1` | Grazing cutoff |

## Python bindings

`EditedReconstruction.add_image_to_tracks(image, images, *, rule, basis,
basis_k, basis_fraction, track_basis, track_basis_k, track_basis_fraction,
pair_statistic, pair_factor, min_zncc, position_gate, position_max_px,
position_k, position_floor_px, template, require_facing, subpixel,
ascend_on_edge, min_keypoint_separation_px, search, max_keypoint_uncertainty,
min_grazing_cos, resolution)` in
[add_image_to_tracks.rs](../../../crates/sfmtool-py/src/reconstruction/add_image_to_tracks.rs)
returns `(EditedReconstruction, report)`. `images` is one decoded image per
image of the reconstruction, or an `ImagePyramidSet`. The rule and gates are
strings plus their numbers (`rule="pooled_or_track"`, `basis="median_minus_mad"`,
`position_gate="image_mad"`, …) so a script can sweep them without building
Rust enums; `basis` is the pooled statistic and `track_basis` the track's. The
report's per-candidate numbers come back as columns under `candidates`: numpy
arrays for the scalars and `(N, 2)` arrays for the keypoints, lists for the
ragged fields. A refused call raises `ValueError`.

```python
from sfmtool._sfmtool.reconstruction import EditedReconstruction
nxt, report = EditedReconstruction(recon).add_image_to_tracks(3, pyramids)
print(report["accepted"], report["refusal_counts"])
```

## Testing

Unit tests in
[add_image_to_tracks/tests.rs](../../../crates/sfmtool-core/src/reconstruction/add_image_to_tracks/tests.rs)
on a synthetic capture (pinhole cameras over a textured plane): a removed
observation is found again within 0.1 px of its projection; the same with a
stored bitmap as the template, fused by `fuse_patch_bitmap`, which pins the
bitmap's grid orientation; existing observations, positions and frames come back
unchanged and tracks stay in image order; a two-reference track is judged by the
pair rule; `PooledOrTrack` accepts what either bar accepts; a photograph of a
different texture is refused by every rule; a point out of frame is
`not_in_frame`; a camera behind the plane is `back_facing`; two points at one
place keep one observation; the confidence column grows in lockstep; a missing
reference image leaves that reference out; the preconditions refuse by name.
Binding tests in
[test_add_image_to_tracks_rust_bindings.py](../../../tests/rust_bindings/test_add_image_to_tracks_rust_bindings.py)
remove one image's observations from the seoul_bull ground truth and find them
again at its ground-truth pose. The leave-one-image-out harness in
[`scripts/add_image_to_tracks/`](../../../scripts/add_image_to_tracks/README.md)
measures recall, keypoint error and the retriangulation residuals of the tracks
an image joins on ground-truth captures.

## Non-goals

No retriangulation, no bundle adjustment, no new points. No occlusion test
against other geometry is made: an occluded point fails the photometric rule,
because the target sees a different surface there. The stored
`observation_confidence` column is not read as the basis: the leave-one-out
ZNCC the operation measures is the same measurement as the new view's, which a
column written by another step need not be.

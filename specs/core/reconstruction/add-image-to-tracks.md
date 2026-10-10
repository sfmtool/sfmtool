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
re-triangulated or adjusted. Each point's `reference_observations` entry moves
with its observation where the new one lands before it in the track, so it names
the same image as before; the bitmap is not re-rendered. What a caller does
afterwards (retriangulate, adjust, or nothing) is the caller's.

It answers a narrower question than the bench's evaluation
([editable-track.md](../bench/editable-track.md)), which aligns every row of a
track to its reference render and can move every keypoint but the reference's.
Here the existing observations are not touched; only the new view's keypoint is
searched for. Like every kernel that places a view, it aligns the new view to the
point's reference render: the point's stored bitmap, or the bitmap the point
would store, rendered from its existing observations
([patch-keypoint-localization.md](../patch/patch-keypoint-localization.md)).

## Rust API

The operation lives in
[add_image_to_tracks.rs](../../../crates/sfmtool-core/src/reconstruction/add_image_to_tracks.rs),
bound as `EditedReconstruction.add_image_to_tracks` on
`sfmtool.reconstruction`. Its per-point kernels are
`TrackReferences` in
[reference.rs](../../../crates/sfmtool-core/src/patch/keypoint_localize/reference.rs)
(render the existing observations and settle the template, search one view
against it, score one view at a keypoint) and `refine_view_against_reference` in
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
pub enum TemplateKind { StoredBitmap, ReferenceObservation, FusedMean }

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
because the template and the references' scores against it are measured, not
read. Poses and cameras are read from `recon`, so a view can never disagree
with the value it is being added to. An image whose photograph is `None` is left
out of every reference set rather than failing the call, the rule
`render_patch_cloud_bitmaps` follows.

**Why the rule is an enum.** Several rules are useful, and they are compared
with each other by the leave-one-image-out harness in
[`scripts/add_image_to_tracks/`](../../../scripts/add_image_to_tracks/README.md);
the default is one of them.

**Why every candidate is reported.** Each point the image does not observe gets
one `CandidateReport` with its outcome (`refusal` is `None` for an added
observation, or a named `Refusal`) and the numbers the verdict was made on: the
projection, the searched and final keypoints, the offset from the projection,
the ZNCC self-similarity radius of the new view's core
(`zncc_self_similarity_radius`), what the template was (`template`, a
`TemplateKind`, and `reference_observation`, the position among the references
of the observation the template is rendered from), the new view's blur-matched
score against the template (`zncc`) and its plain ZNCC against each reference
(`pair_zncc`), the references' blur-matched scores against the template
(`reference_zncc`) and their plain pairwise ZNCCs, and the number the rule
compared with its bar. A caller can see why a point was not
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
2. **Template.** Each existing observation in a decoded image (a reference) is
   rendered on the patch grid anchored at its own keypoint (the in-plane offset
   its keypoint states, as the bench and the bitmap render anchor it). Fewer
   than two references in frame is `too_few_references`. The template
   (`TemplateKind`) is the first of these that applies:
   - with `TemplateSource::StoredBitmap`, the point's stored bitmap where it has
     a textured one, on the bitmap's own grid (`StoredBitmap`);
   - the render of the point's reference observation at its own keypoint, which
     is the bitmap the point would store: its `reference_observations` entry
     where that observation rendered, else the [reference-view
     rule](../patch/reference-view.md)'s pick from the references' renders
     (`ReferenceObservation`);
   - the fused mean of the references' renders, where the point has no
     reference observation and the rule picks none it would store
     (`FusedMean`).

   The template the search and the sub-pixel step read is never blurred. The
   pairwise ZNCCs between references, which only the pair rule reads, are read
   plain from the same renders.
3. **Search.** The target's context tile is rendered once around the point's
   projection and its own core's [ZNCC self-similarity
   radius](../patch/zncc-self-similarity-radius.md) is read (`unlocalizable`
   over `max_member_zncc_self_similarity_radius`, when that gate is on). One windowed-ZNCC shift search over
   `±search` patch-grid pixels then finds the peak, run exhaustively because it is
   one view per point. A peak on the edge
   of the window is `peak_at_edge`, because the true maximum may lie outside
   what was searched. With `ascend_on_edge`, such a window is searched again by
   the "+"-descent from the projection, and the local maximum it climbs to is
   taken when it is inside the window: a repeated texture can put a stronger
   correlation a period away, and the projection is the evidence for which
   period is meant.
4. **Sub-pixel.** With `subpixel`, the keypoint is refined by the ECC
   Gauss-Newton solve against the same template
   (`TrackReferences::refine_template`), moving only the target, with the
   solve's never-worse guard.
5. **Score.** The bars judge each view's **blur-matched score** against the
   template, read as the bench reads a row against the stored bitmap
   ([../patch/blur-matched-zncc.md](../patch/blur-matched-zncc.md) § "Scores
   against the stored bitmap"): the target's tile rendered at its final
   keypoint and each reference's at its own keypoint, each scored by
   `BitmapScorer` against the template as an RGBA bitmap (the stored bitmap,
   the reference observation's render or the fused mean), the bitmap alone
   blurred to the tile's sharpness where it is sharper along every direction.
   The target's is `zncc`, each reference's `reference_zncc` (`1.0` for the
   reference observation where the template is its render), so the target's is
   comparable with theirs. The target's plain ZNCC against each reference
   (`pair_zncc`), which the pair rule reads, is the search's, read plain. A
   point with no stored bitmap is scored against the render it would store, so
   no point needs another path. A candidate refused before this step carries
   no `reference_zncc`, so `reference_zncc` holds blur-matched scores only
   (the plain `reference_pair_zncc` remains), and a
   target whose tile cannot be read against the template (a `NaN` score) is
   `unscorable`.
6. **Judge.** The rule decides (below). `min_zncc` is a floor the basis rules
   also apply (`0` disables it); under `FixedZncc` it is the whole rule.
7. **Place.** The positional gate, then one observation per place: an accepted
   keypoint closer than `min_keypoint_separation_px` to an observation the image
   already has is refused, and of two accepted keypoints that close the one with
   the higher ZNCC is kept (`shared_keypoint`). Two points at one pixel of one
   image are two tracks of one surface, and adding the image to both would
   repeat that.

Accepted observations are written with their keypoint and, where the value has
the column, their blur-matched score (`zncc`) in `observation_confidence` on
the byte scale the bench commit uses (`round(255·clamp(z, 0, 1))`), raised to `1` because `0` means
unmeasured. Each also carries its
[observation readings](../../formats/sfmr-file-format.md#observation-readings-optional-version-12)
(`CandidateReport::reading`), read on the tile its score was read on: the
tile's self-similarity ellipse, viewing angle, tilt and zoom, and its plain
and blur-matched scores against the template where the template is the point's
stored bitmap, `NaN` where it is the render the point would store. A value
without readings gains the columns under the options the new views were read
under, every existing row not measured; one whose
readings stand under other options than those the new views were read under
(another resolution, sampler or window) gets new rows with
nothing measured. Every existing row is carried with its observation. Each
track stays in image order. The stored per-point error is left as it was,
since nothing moved the point.

## The judging rules

Every rule but the pair rule reads blur-matched scores against the template
(`zncc`, `reference_zncc`); the pair rule reads the search's plain pairwise
ZNCCs (`pair_zncc`, `reference_pair_zncc`), which compare one view with
another rather than with the template.

- `FixedZncc`: accept when the ZNCC is at least `min_zncc`.
- `TrackBasis`: the point's own references set the bar. With three or more, the
  bar is a statistic of their ZNCCs against the template, with the reference
  observation's own score left out (`CandidateReport::bar_zncc`), since it is
  the template or what the template was rendered from: the minimum, the median
  minus `k` times the scaled median absolute deviation (MAD × 1.4826), or a
  fraction of the median. With exactly two, one of them is usually the
  template's source, so a statistic of what is left says little; instead the
  minimum, mean or maximum of the target's ZNCC against each reference must
  reach `factor` times the ZNCC between the two references.
- `PooledBasis`: the statistic over the ZNCCs against the template of every
  reference but the reference observation, of every candidate that reached the
  verdict, which is one bar for the whole image.
- `PooledOrTrack`: accept a candidate that reaches either the pooled bar or its
  own track's bar (`TrackBasis` with `track` and `pair`).

**Why the default is what it is.** The default is `PooledOrTrack` with the
pooled bar at the median minus two scaled deviations, the track's at 0.9 of
its references' median (the pair rule at 0.9 of the mean for two), a floor of
0.5, and the `ImageMad` positional gate. Every bar is read off the call's own
data, so it follows the capture's texture and the pose's error rather than a
constant: a fixed ZNCC bar that suits one capture refuses good sightings on
another. The pooled bar is what keeps a track whose references barely agree with
each other from lowering the bar for itself; the track's own bar is what keeps a
good sighting on a surface harder than the rest of the image from being refused
for it. The positional gate is what removes most wrong sightings that correlate
well.

**How the bars were measured on the reference render.** The bars were chosen
when the template was the consensus of the references and their scores were
leave-one-out against it, and were measured again once both read the reference
render
([`scripts/add_image_to_tracks/README.md`](../../../scripts/add_image_to_tracks/README.md)
§ "The bars once the new view is aligned to the reference render"). One image's
observations are removed and the image is added back at its resected pose.
"Err" is the rejoined keypoints' distance from the ground truth's, at the
median and the 90th percentile, and ">2 px" counts rejoined keypoints more
than 2 px from it. "Extra" counts the tracks the image joins that it was not
in, "bad" those of them whose new observation's residual exceeds 2 px once the
point is retriangulated with it, and "worse" those whose largest residual over
all observations grew by more than 1 px. Measured 2026-10-09, after the
localizer's sub-pixel step became a 3×3 quadratic fit:

| Rule | seoul_bull recall | err med / p90 px | >2 px | extra | bad | worse | kerry_park recall | err med / p90 px | >2 px | extra | bad | worse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Leave-one-out bars, `k = 3` (before 2026-10-09) | 82.8% | 0.061 / 0.215 | 16 | 115 | 0 | 1 | 79.2% | 0.060 / 0.346 | 13 | 1107 | 0 | 8 |
| **Default, pooled bar `k = 2`** | **79.8%** | **0.099 / 0.341** | **13** | **106** | **0** | **0** | **77.8%** | **0.089 / 0.464** | **25** | **1143** | **1** | **13** |
| Pooled bar `k = 3` | 81.5% | 0.101 / 0.364 | 13 | 139 | 0 | 1 | 79.7% | 0.092 / 0.472 | 28 | 1327 | 6 | 28 |
| Pooled bar `k = 4` | 81.9% | 0.101 / 0.371 | 15 | 152 | 0 | 2 | 80.9% | 0.094 / 0.474 | 28 | 1519 | 9 | 42 |

The references score lower and spread wider against one sharp render than they
did against the consensus of the others, so the pooled bar at `k = 3` falls
from about 0.68 to 0.55 on seoul_bull and the `0.5` floor applies more often.
At `k = 3` the image joins about 20% more extra tracks on kerry_park than with
the leave-one-out bars, 6 of them bad and 28 worse against 0 and 8, and its
rejoined keypoints are further from the ground truth's: 0.09 to 0.10 px at the
median against 0.06, and 0.36 to 0.47 px at the 90th percentile against 0.22
to 0.35, with 28 against 13 over 2 px on kerry_park. A view aligned to one
reference shares whatever offset that reference's keypoint has, and the ground
truth's keypoints were placed by congealing. The pooled bar is therefore at
`k = 2`, a maintainer's decision: against `k = 3` it gives up 1.7 points of
recall on seoul_bull and 1.9 on kerry_park, and on kerry_park joins 1143 extra
tracks rather than 1327, 1 of them bad and 13 worse rather than 6 and 28. The
seoul_bull runs on this date prepared a fresh cache, so their cluster tracks
can differ slightly from the run before 2026-10-09. The track bar and the pair
rule change recall by under 1.2 points between 0.85 and 0.95, and 0.8 and 1.0;
the full table is in the README.

**The bars on the blur-matched score.** Measured again on 2026-10-10 with the
same sweep, the bars reading the search's plain score and then the
blur-matched score
([`scripts/add_image_to_tracks/README.md`](../../../scripts/add_image_to_tracks/README.md)
§ "The bars on the blur-matched score"; resected pose, recall / extra / bad /
worse):

| Rule | seoul_bull, plain | seoul_bull, blur-matched | kerry_park, plain | kerry_park, blur-matched |
|---|---|---|---|---|
| **Default, pooled bar `k = 2`** | 79.8% / 106 / 0 / 0 | **79.9% / 108 / 0 / 0** | 77.8% / 1143 / 1 / 13 | **77.8% / 1148 / 1 / 16** |
| Pooled bar `k = 3` | 81.5% / 139 / 0 / 1 | 81.5% / 140 / 0 / 1 | 79.7% / 1327 / 6 / 28 | 79.4% / 1321 / 5 / 28 |
| Pooled bar `k = 4` | 81.9% / 152 / 0 / 2 | 81.9% / 152 / 0 / 2 | 80.9% / 1519 / 9 / 42 | 80.6% / 1505 / 8 / 40 |
| Floor `0.6` | 78.2% / 105 / 0 / 0 | 78.3% / 107 / 0 / 0 | 77.4% / 1128 / 1 / 15 | 77.6% / 1135 / 1 / 16 |

Every bar is set from the references' own scores, and blur matching raises
those together with the target's, so the switch moves recall by under half a
point and the extra tracks by a few. The track bar at 0.85 and 0.95, the pair
rule at 0.8 and 1.0 and the floor at 0.4 move as little on the blur-matched
score as on the plain one. The defaults are kept and read the blur-matched
score.

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
| `rule` | `PooledOrTrack { pooled: MedianMinusMad { k: 2.0 }, track: FractionOfMedian { fraction: 0.9 }, pair: { Mean, 0.9 } }` | The photometric rule |
| `min_zncc` | `0.5` | ZNCC floor under every rule |
| `position_gate` | `ImageMad { k: 3.0, floor_px: 1.0 }` | Positional bound |
| `template` | `Rendered` | Search template |
| `require_facing` | `true` | Back-facing check |
| `subpixel` | `true` | ECC sub-pixel step |
| `ascend_on_edge` | `false` | Ascent from the projection when the peak is on the window's edge |
| `min_keypoint_separation_px` | `1.0` | One observation per place |
| `localize.search` | `6.0` | Search radius, patch-grid px |
| `localize.resolution` | `24` | Patch grid (a stored bitmap's own grid under `StoredBitmap`) |
| `localize.max_member_zncc_self_similarity_radius` | `2.5` | Self-similarity gate: the largest ZNCC self-similarity radius of the new view's core, patch-grid px ([why 2.5](../patch/patch-keypoint-localization.md#the-member-gates-default)) |
| `localize.min_grazing_cos` | `0.1` | Grazing cutoff |

## Python bindings

`EditedReconstruction.add_image_to_tracks(image, images, *, rule, basis,
basis_k, basis_fraction, track_basis, track_basis_k, track_basis_fraction,
pair_statistic, pair_factor, min_zncc, position_gate, position_max_px,
position_k, position_floor_px, template, require_facing, subpixel,
ascend_on_edge, min_keypoint_separation_px, search,
max_zncc_self_similarity_radius, min_grazing_cos, resolution)` in
[add_image_to_tracks.rs](../../../crates/sfmtool-py/src/reconstruction/add_image_to_tracks.rs)
returns `(EditedReconstruction, report)`. `images` is one decoded image per
image of the reconstruction, or an `ImagePyramidSet`. The rule and gates are
strings plus their numbers (`rule="pooled_or_track"`, `basis="median_minus_mad"`,
`position_gate="image_mad"`, …) so a script can sweep them without building
Rust enums; `basis` is the pooled statistic and `track_basis` the track's. The
report's per-candidate numbers come back as columns under `candidates`: numpy
arrays for the scalars and `(N, 2)` arrays for the keypoints, lists for the
ragged fields (`reference_zncc`, `reference_pair_zncc`, `pair_zncc`).
`template` is a list of `"stored_bitmap"`, `"reference_observation"`,
`"fused_mean"` or `None`, and `reference_observation` an `int64` array with `-1`
for none. A refused call raises `ValueError`.

```python
from sfmtool.reconstruction import EditedReconstruction
nxt, report = EditedReconstruction(recon).add_image_to_tracks(3, pyramids)
print(report["accepted"], report["refusal_counts"])
```

## Testing

Unit tests in
[add_image_to_tracks/tests.rs](../../../crates/sfmtool-core/src/reconstruction/add_image_to_tracks/tests.rs)
on a synthetic capture (pinhole cameras over a textured plane): a removed
observation is found again within 0.1 px of its projection; the same with a
stored bitmap as the template, fused by `fuse_patch_bitmap`, which pins the
bitmap's grid orientation; the template is the stored bitmap where there is
one, else the stored reference observation's render, else the rule's pick, else
the fused mean, and the bars read the references' scores against it; existing
observations, positions and frames come back
unchanged and tracks stay in image order; a reference observation moves with
its observation where the new one lands before it in the track and stays where
it lands after it; a two-reference track is judged by the
pair rule; `PooledOrTrack` accepts what either bar accepts; a photograph of a
different texture is refused by every rule; a point out of frame is
`not_in_frame`; a camera behind the plane is `back_facing`; two points at one
place keep one observation; the confidence column grows in lockstep; each
added observation carries its candidate's readings, every existing row not
measured where the value had none; the new
view's score, judged and stored in the confidence column, is its blur-matched
score against the template, as `BitmapScorer` reads it directly; a missing
reference image leaves that reference out; the preconditions refuse by name.
Binding tests in
[test_add_image_to_tracks_rust_bindings.py](../../../tests/rust_bindings/reconstruction/test_add_image_to_tracks_rust_bindings.py)
remove one image's observations from the seoul_bull ground truth and find them
again at its ground-truth pose, and check that a self-similarity bar refuses as
`unlocalizable` exactly the candidates whose radius is over it. The leave-one-image-out harness in
[`scripts/add_image_to_tracks/`](../../../scripts/add_image_to_tracks/README.md)
measures recall, keypoint error and the retriangulation residuals of the tracks
an image joins on ground-truth captures.

## Non-goals

No retriangulation, no bundle adjustment, no new points. No occlusion test
against other geometry is made: an occluded point fails the photometric rule,
because the target sees a different surface there. The stored
`observation_confidence` column is not read as the basis: the ZNCC against the
template the operation measures is the same measurement as the new view's, which
a column written by another step need not be.

# Add Image to Tracks

**Status:** Draft. Built and evaluated: the core operation, its kernels, the
Python binding and the leave-one-image-out harness. Decided by the evaluation
below: the default rule (one ZNCC bar per image read off every candidate's
references, plus a positional bound read off the image's own accepted
keypoints). Not built: the viewer's image menu entry beside *Resect Image*,
which is the next step. The draft is filed once that entry exists.

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
  `evaluate`) builds an editable track per point, runs a congealing round over
  every observation including the new one, and writes the track back with
  `commit`, which replaces the point and renumbers it. For every point of a
  reconstruction that is hundreds or thousands of bench items. `evaluate` also
  scores every view against a leave-one-out consensus whose templates include
  the new view, which is not the question here.
- `EditedReconstruction::replace_point` per point writes through the overlay, so
  every touched point moves to a new index when the value is materialised.
- The localizer's consensus-basis *tail registration*
  ([keypoint-localization-consensus-basis.md](../core/patch/keypoint-localization-consensus-basis.md),
  Phase B) is the right kernel: one search of one view against a finished
  consensus that view did not help build. But it is reachable only from inside
  a congealing run, where the basis views are first moved.

So the operation is new, and it reuses the kernels underneath rather than the
steps above them: the localizer's render-once context tile, its member
localizability score and its windowed-ZNCC shift search, the robust (IRLS)
consensus, and the sub-pixel ECC Gauss-Newton solve, here against a fixed
template.

## Rust API

The operation lives in
[add_image_to_tracks.rs](../../crates/sfmtool-core/src/reconstruction/add_image_to_tracks.rs)
and is bound as `EditedReconstruction.add_image_to_tracks` on
`sfmtool._sfmtool.reconstruction`. The per-point kernels are
`ReferenceConsensus` in
[reference.rs](../../crates/sfmtool-core/src/patch/keypoint_localize/reference.rs)
(build the references' consensus, search one view against it, score one view
at a keypoint) and `refine_view_against_references` in
[keypoint_subpixel.rs](../../crates/sfmtool-core/src/patch/keypoint_subpixel.rs).

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
`bundle_adjust` and `move_camera` are: a whole new value comes back, with every
point at the index it had. The binding materialises an `EditedReconstruction`
first, as `EditedReconstruction.bundle_adjust` does.

**Why it takes decoded photographs rather than posed views.** The target image
must be decoded, and so must the images of the observations used as
references, because the reference consensus and the leave-one-out scores are
measured, not read. Poses and cameras are read from `recon`, so a view can
never disagree with the value it is being added to. An image whose photograph
is `None` is left out of every reference set rather than failing the call, the
rule `fuse_patch_cloud_bitmaps` follows.

**Why the rule is an enum.** Which views to accept was the open question the
harness answered; the enum lets every candidate rule run through the same code,
and the others stay available for comparison.

**Why every candidate is reported.** Each point the image does not observe gets
one `CandidateReport` with its outcome (`refusal` is `None` for an accepted
point, or a named `Refusal`) and the numbers the verdict was made on: the
projection, the searched and final keypoints, the offset from the projection,
the member localizability score, the ZNCC against the consensus and against
each reference, the references' leave-one-out and pairwise ZNCCs, and the
number the rule compared with its bar. A caller can see why a point was not
added, and a harness can re-judge without re-running.

The refusals are `no_patch`, `not_in_frame`, `grazing`, `back_facing`,
`too_few_references`, `unlocalizable`, `no_peak`, `peak_at_edge`,
`unscorable`, `below_floor`, `below_bar`, `too_far` and `shared_keypoint`.

Example:

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
   its keypoint states, exactly as the bench and the bitmap fuse anchor it).
   The renders are z-normalised and combined into the robust IRLS consensus.
   From the same renders come each reference's leave-one-out ZNCC (against the
   robust consensus of the others) and the pairwise ZNCCs between references.
   Fewer than two references in frame is `too_few_references`. With
   `TemplateSource::StoredBitmap` the search template is the point's stored
   bitmap instead, on the bitmap's own grid, and the references still supply
   the leave-one-out and pairwise numbers.
3. **Search.** The target view's context tile is rendered once around the
   point's projection and its own core is scored for localizability
   (`unlocalizable` above `max_member_keypoint_uncertainty`). One windowed-ZNCC
   shift search over `±search` patch-grid pixels then finds the peak: the tail
   registration's search, run exhaustively because it is one view per point.
   A peak on the edge of the window is `peak_at_edge`, because the true maximum
   may lie outside what was searched.
4. **Sub-pixel.** With `subpixel`, the keypoint is refined by the ECC
   Gauss-Newton solve against the references' frozen consensus, moving only the
   target, with the solve's never-worse guard.
5. **Score.** At the final keypoint the target's core is rendered, and its ZNCC
   against the template and against each reference is taken by the same scorer
   that took the references' leave-one-out ZNCCs. The target never contributed
   to the consensus, so its ZNCC is a leave-one-out number and is comparable
   with theirs.
6. **Judge.** The rule decides. `min_zncc` is a floor the basis rules also
   apply (`0` disables it); under `FixedZncc` it is the whole rule.
7. **Place.** The positional gate, then one observation per place: an accepted
   keypoint closer than `min_keypoint_separation_px` to an observation the image
   already has is refused, and of two accepted keypoints that close the one with
   the higher ZNCC is kept (`shared_keypoint`).

Accepted observations are written with their keypoint and, where the value has
the column, their ZNCC in `observation_confidence` on the byte scale the bench
commit uses (`round(255·clamp(z, 0, 1))`), raised to `1` because `0` means
unmeasured. The stored per-point error is left as it was, and the tracks stay
in image order within each point. A `sift_files` value is refused, because an
added observation has no feature index to name.

## The judging rules

- `FixedZncc`: accept when the ZNCC is at least `min_zncc`.
- `TrackBasis`: the point's own references set the bar. With three or more,
  the bar is a statistic of their leave-one-out ZNCCs: the minimum, the median
  minus `k` times the scaled median absolute deviation (MAD × 1.4826), or a
  fraction of the median. With exactly two, each reference's leave-one-out
  ZNCC is their pairwise ZNCC, so a statistic of two copies of one number says
  little; instead the minimum, mean or maximum of the target's ZNCC against each
  reference must reach `factor` times the ZNCC between the two references.
- `PooledBasis`: the statistic over the leave-one-out ZNCCs of every reference
  of every candidate that reached the verdict: one bar for the whole image.

## Positional gate

`MaxPx` refuses a keypoint more than that many source pixels from the point's
projection. `ImageMad` derives the bound from the call's own photometrically
accepted candidates: a resected image's pose error moves every projection by a
similar amount, so their offsets are pooled and a candidate is refused when its
offset exceeds the median plus `k` scaled MADs, never below `floor_px`. With
fewer than three such candidates no bound is derived.

## Parameters

Defined in `AddImageToTracksOptions::default()` and mirrored as the binding's
keyword defaults.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `rule` | `PooledBasis { MedianMinusMad { k: 3.0 } }` | The photometric rule |
| `min_zncc` | `0.5` | ZNCC floor under every rule |
| `position_gate` | `ImageMad { k: 3.0, floor_px: 1.0 }` | Positional bound |
| `template` | `Rendered` | Search template |
| `require_facing` | `true` | Back-facing check |
| `subpixel` | `true` | ECC sub-pixel step |
| `min_keypoint_separation_px` | `1.0` | One observation per place |
| `localize.search` | `6.0` | Search radius, patch-grid px |
| `localize.resolution` | `24` | Patch grid |
| `localize.max_member_keypoint_uncertainty` | `0.35` | Localizability gate `τ` |
| `localize.min_grazing_cos` | `0.1` | Grazing cutoff |

## Python binding

`EditedReconstruction.add_image_to_tracks(image, images, *, rule, basis,
basis_k, basis_fraction, pair_statistic, pair_factor, min_zncc, position_gate,
position_max_px, position_k, position_floor_px, template, require_facing,
subpixel, min_keypoint_separation_px, search, max_keypoint_uncertainty,
min_grazing_cos, resolution)` in
[add_image_to_tracks.rs](../../crates/sfmtool-py/src/reconstruction/add_image_to_tracks.rs)
returns `(EditedReconstruction, report)`. `images` is one decoded image per
image of the reconstruction, or an `ImagePyramidSet`. The rule and gates are
strings plus their numbers (`rule="pooled_basis"`, `basis="median_minus_mad"`,
`position_gate="image_mad"`, …) so a script can sweep them without building
Rust enums. The report's per-candidate numbers come back as columns under
`candidates`: numpy arrays for the scalars and `(N, 2)` arrays for the
keypoints, lists for the ragged fields.

```python
from sfmtool._sfmtool.reconstruction import EditedReconstruction
nxt, report = EditedReconstruction(recon).add_image_to_tracks(3, pyramids)
print(report["accepted"], report["refusal_counts"])
```

## Testing

Unit tests in
[add_image_to_tracks/tests.rs](../../crates/sfmtool-core/src/reconstruction/add_image_to_tracks/tests.rs)
on a synthetic capture (pinhole cameras over a textured plane): a removed
observation is found again within 0.1 px of its projection; the same with a
stored bitmap as the template, fused by `fuse_patch_bitmap`, which pins the
bitmap's grid orientation; existing observations, positions and frames come back
unchanged and tracks stay in image order; a two-reference track is judged by
the pair rule; a photograph of a different texture is refused by every rule; a
point out of frame is `not_in_frame`; a camera behind the plane is
`back_facing`; two points at one place keep one observation; the confidence
column grows in lockstep; a missing reference image leaves that reference out;
the preconditions refuse by name. Binding tests in
[test_add_image_to_tracks_rust_bindings.py](../../tests/rust_bindings/test_add_image_to_tracks_rust_bindings.py)
remove one image's observations from the seoul_bull ground truth and find them
again at its ground-truth pose.

## Evaluation

The harness in [`scripts/add_image_to_tracks/`](../../scripts/add_image_to_tracks/)
(`harness.py`, `strategies.py`, `summarize.py`, `datasets.py`) runs, for every
image of a ground truth:

1. Remove the image from the ground truth: delete all of its observations, and
   delete the points left with fewer than two (counted as lost; they cannot
   come back). The image keeps its ground-truth pose in the value, which the
   resection replaces.
2. Resect it back with `geometry.resect_images` over the tracks and a
   cluster-patches `.matches` file, as *Resect Image* does. With all of its
   observations gone the image has no tracks, so every resection here is posed
   from clusters alone.
3. Run the operation under every strategy, at the resected pose and at the
   ground-truth pose, so the error the pose brings and the error the matching
   brings can be told apart.
4. Measure recall on the tracks the image was in that survived the removal;
   each rejoined keypoint's distance from the original; the rejoined keypoints
   more than 1 and 2 px from the original (precision problems on known tracks);
   the tracks it joins that it was not in ("extra"), with their ZNCC, their
   distance from the projection, and the new observation's residual after the
   point is retriangulated with it (a least-squares ray intersection); and the
   same residual for rejoined known tracks as the baseline.

Datasets:

- **seoul_bull** (checked-in ground truth, 17 images, 280 points, one
  SIMPLE_RADIAL camera): copied with its images into a cache directory, where
  `.sift` files and the index files are built by the track-at-pixel harness's
  own preparation. 1235 known observations to recover; 42 points lost on
  removal. Every resection accepted, rotation error median 0.107°, max 0.219°;
  centre error median 0.0019, max 0.0039 of the scene scale.
- **kerry_park** candidate `tk106` (48 images, 384 points, two OPENCV_FISHEYE
  cameras): read in place, with the cluster-patches file built for `tk105`,
  which names the same images (the resection matches images by name). 2225
  known observations; 64 points lost on removal. Every resection accepted. Over
  the 45 images that have observations in the ground truth: rotation error
  median 0.114°, p90 0.300°, max 1.83° (image 37, `fisheye_right/frame_14`);
  centre error median 0.0033, max 0.041 of the scene scale. Images 21 to 23
  (`fisheye_left/frame_22` to `frame_24`) have **no observations in the ground
  truth**, so their stored poses are not registrations; the resection moves
  them by 8 to 10° and about 19 scene scales, and the operation then adds 12 to
  19 observations to each at a median retriangulation residual of 0.19 to
  0.32 px, where at the stored pose it adds none. The stored poses of those
  three images look wrong, and resection followed by this operation registers
  them.

All figures are summed over every image of the dataset. Measurement: rendered
template, sub-pixel step, localizability gate on.

### seoul_bull, resected pose

| strategy | recall | err med / p90 px | >1 px | >2 px | extra | extra ZNCC | extra new res med / p90 px | extra res > 2 px |
|---|---|---|---|---|---|---|---|---|
| fixed 0.6 | 97.4% | 0.062 / 0.277 | 57 | 48 | 233 | 0.743 | 0.66 / 1.81 | 20 |
| fixed 0.7 | 94.6% | 0.060 / 0.269 | 56 | 47 | 162 | 0.781 | 0.59 / 1.63 | 11 |
| fixed 0.8 | 86.2% | 0.058 / 0.213 | 46 | 38 | 68 | 0.847 | 0.49 / 1.22 | 2 |
| fixed 0.9 | 55.1% | 0.052 / 0.182 | 23 | 18 | 15 | 0.944 | 0.50 / 0.66 | 0 |
| track min, pair mean ×1.0 | 75.9% | 0.062 / 0.323 | 52 | 43 | 73 | 0.734 | 0.90 / 2.74 | 11 |
| track min, pair min ×1.0 | 74.1% | 0.062 / 0.307 | 49 | 41 | 72 | 0.733 | 0.90 / 2.76 | 11 |
| track min, pair max ×1.0 | 80.2% | 0.062 / 0.316 | 52 | 43 | 73 | 0.734 | 0.90 / 2.74 | 11 |
| track min, pair mean ×0.9 | 85.1% | 0.062 / 0.303 | 54 | 45 | 76 | 0.758 | 0.80 / 2.68 | 11 |
| track med−3·MAD | 89.1% | 0.062 / 0.316 | 58 | 47 | 80 | 0.792 | 0.61 / 2.59 | 10 |
| track 0.9·median | 92.4% | 0.062 / 0.303 | 58 | 47 | 75 | 0.833 | 0.53 / 1.91 | 7 |
| pooled med−2·MAD | 90.8% | 0.059 / 0.248 | 51 | 42 | 106 | 0.821 | 0.55 / 1.52 | 5 |
| pooled med−3·MAD | 95.3% | 0.061 / 0.269 | 56 | 47 | 165 | 0.780 | 0.60 / 1.62 | 11 |
| track min, pair mean ×0.9 + image MAD | 79.8% | 0.058 / 0.200 | 23 | 18 | 44 | 0.801 | 0.54 / 1.10 | 0 |
| track 0.9·median + image MAD | 86.2% | 0.058 / 0.198 | 24 | 18 | 53 | 0.836 | 0.38 / 0.74 | 0 |
| fixed 0.7 + image MAD | 89.1% | 0.057 / 0.193 | 25 | 19 | 117 | 0.792 | 0.42 / 0.92 | 0 |
| **pooled med−3·MAD + image MAD (default)** | **89.7%** | **0.058 / 0.193** | **25** | **19** | **119** | **0.792** | **0.44 / 0.96** | **0** |
| pooled med−3·MAD + image MAD, k = 5 | 91.4% | 0.058 / 0.200 | 32 | 25 | 129 | 0.789 | 0.48 / 1.10 | 0 |
| pooled med−3·MAD + image MAD, floor 2 px | 92.7% | 0.059 / 0.211 | 34 | 26 | 138 | 0.790 | 0.50 / 1.22 | 0 |
| pooled med−3·MAD + max 2 px | 92.7% | 0.059 / 0.211 | 34 | 26 | 137 | 0.792 | 0.50 / 1.21 | 0 |
| pooled med−3·MAD + max 3 px | 94.0% | 0.060 / 0.222 | 42 | 33 | 152 | 0.781 | 0.55 / 1.49 | 4 |

Rejoined known tracks retriangulate with a new-observation residual median of
0.26 to 0.28 px under every strategy. At the ground-truth pose the default
recovers 89.8% at 0.056 px median, with 13 rejoined keypoints over 2 px and 102
extra tracks: the resected pose costs almost nothing on this capture.

### kerry_park, resected pose

| strategy | recall | err med / p90 px | >1 px | >2 px | extra | extra ZNCC | extra new res med / p90 px | extra res > 2 px |
|---|---|---|---|---|---|---|---|---|
| fixed 0.6 | 96.8% | 0.068 / 0.444 | 94 | 65 | 2417 | 0.857 | 0.36 / 1.45 | 144 |
| fixed 0.7 | 95.6% | 0.067 / 0.436 | 87 | 59 | 2013 | 0.882 | 0.34 / 1.21 | 89 |
| fixed 0.8 | 92.3% | 0.065 / 0.434 | 79 | 53 | 1558 | 0.909 | 0.33 / 1.15 | 58 |
| fixed 0.9 | 76.4% | 0.060 / 0.411 | 60 | 44 | 859 | 0.941 | 0.30 / 0.87 | 19 |
| track min, pair mean ×1.0 | 84.0% | 0.061 / 0.411 | 74 | 53 | 1169 | 0.912 | 0.37 / 1.42 | 63 |
| track min, pair min ×1.0 | 80.5% | 0.060 / 0.397 | 71 | 51 | 1140 | 0.912 | 0.37 / 1.45 | 63 |
| track min, pair max ×1.0 | 85.7% | 0.062 / 0.414 | 78 | 55 | 1195 | 0.912 | 0.37 / 1.40 | 63 |
| track min, pair mean ×0.9 | 90.5% | 0.066 / 0.438 | 83 | 59 | 1258 | 0.912 | 0.36 / 1.35 | 63 |
| track med−3·MAD | 88.2% | 0.065 / 0.438 | 79 | 56 | 1179 | 0.914 | 0.34 / 1.18 | 44 |
| track 0.9·median | 93.0% | 0.065 / 0.433 | 81 | 57 | 1430 | 0.915 | 0.33 / 1.05 | 46 |
| pooled med−2·MAD | 88.4% | 0.063 / 0.417 | 71 | 51 | 1292 | 0.922 | 0.33 / 1.03 | 41 |
| pooled med−3·MAD | 92.4% | 0.065 / 0.434 | 79 | 53 | 1581 | 0.908 | 0.33 / 1.15 | 60 |
| track min, pair mean ×0.9 + image MAD | 86.0% | 0.060 / 0.363 | 29 | 15 | 1040 | 0.919 | 0.29 / 0.78 | 3 |
| track 0.9·median + image MAD | 88.6% | 0.060 / 0.359 | 28 | 14 | 1220 | 0.920 | 0.28 / 0.72 | 2 |
| fixed 0.7 + image MAD | 91.3% | 0.062 / 0.364 | 33 | 15 | 1705 | 0.892 | 0.30 / 0.79 | 6 |
| **pooled med−3·MAD + image MAD (default)** | **88.0%** | **0.061 / 0.364** | **29** | **13** | **1356** | **0.913** | **0.29 / 0.74** | **2** |
| pooled med−3·MAD + image MAD, k = 5 | 89.8% | 0.062 / 0.378 | 41 | 18 | 1444 | 0.911 | 0.30 / 0.83 | 7 |
| pooled med−3·MAD + image MAD, floor 2 px | 90.1% | 0.063 / 0.385 | 47 | 22 | 1451 | 0.910 | 0.30 / 0.80 | 2 |
| pooled med−3·MAD + max 2 px | 89.8% | 0.063 / 0.385 | 47 | 22 | 1440 | 0.910 | 0.30 / 0.79 | 2 |
| pooled med−3·MAD + max 3 px | 90.8% | 0.063 / 0.390 | 51 | 25 | 1510 | 0.909 | 0.32 / 0.90 | 13 |

Rejoined known tracks retriangulate with a new-observation residual median of
0.21 to 0.24 px. At the ground-truth pose the default recovers 89.1% at
0.060 px median, with 9 rejoined keypoints over 2 px and 1280 extra tracks.

### Measurement settings

Against the default rule at the resected pose:

- **Sub-pixel step.** Without it the rejoined keypoints' median error rises from
  0.058 to 0.088 px on seoul_bull and from 0.061 to 0.074 px on kerry_park, and
  the counts move by a few. It stays on.
- **Localizability gate.** It refuses 0 to 29 candidates per kerry_park image
  and almost none on seoul_bull. Turning it off adds 3 rejoined and 10 extra
  observations on kerry_park and 1 extra on seoul_bull, with no change in the
  far or bad counts: what it refuses mostly fails the photometric rule anyway.
  It stays on at the localizer's `τ = 0.35`.
- **Stored bitmap as the template** (kerry_park only; the seoul_bull ground
  truth is a minimal file with no bitmaps): recall 86.9% against 88.0% rendered,
  10 rejoined keypoints over 2 px against 13, 1247 extra tracks against 1356.
  Slightly lower ZNCC throughout, as a bitmap quantised to bytes and fused with
  its own weights is a different reference. The rendered consensus stays the
  default; the references have to be rendered for their leave-one-out ZNCCs in
  any case, so the bitmap saves no rendering.

### Readings

- **The positional gate is what separates the strategies.** Every photometric
  rule alone leaves 3 to 4% of rejoined keypoints more than 2 px from the
  original and several percent of extra tracks whose new observation does not
  agree with the retriangulated point. The image-MAD bound halves or better the
  first (seoul 47 → 19, kerry 53 → 13) and all but removes the second (seoul
  11 → 0, kerry 60 → 2), at a cost of 4 to 6 points of recall. Its bound came
  out at a median of 1.2 px on both captures (1.0 to 2.5 px per image).
- **The per-track leave-one-out basis, as first proposed, costs recall and buys
  no precision.** With the minimum of the references' leave-one-out ZNCCs as the
  bar, and the pair rule at factor 1, recall is 76% on seoul_bull and 84% on
  kerry_park, and the extra tracks it accepts are no better than a fixed bar's.
  The references' keypoints were fitted together and the new view's was not,
  so the new view's ZNCC runs lower than theirs for the same quality of match,
  and the minimum of several numbers is a demanding bar. A softer per-track
  statistic (0.9 × median) recovers most of the recall. Within the pair rule,
  `max` beats `mean` beats `min` on recall with the same precision, and a factor
  of 0.9 gains 6 to 9 points of recall over 1.0 at no cost in precision.
- **The pooled basis follows the capture.** The median−3·MAD bar over every
  candidate's references came out at 0.64 to 0.72 on seoul_bull and 0.75 to
  0.85 on kerry_park; a fixed bar right for one is wrong for the other (fixed
  0.8 recovers 86% on seoul_bull against the pooled bar's 95%; fixed 0.7 admits
  89 bad extra observations on kerry_park against its 60). Under the positional gate the pooled bar gives
  the best recall of the data-driven rules on seoul_bull and matches the
  per-track 0.9 × median on kerry_park.
- **What remains.** Of the 19 (seoul) and 13 (kerry) rejoined keypoints over
  2 px under the default, 13 and 10 sit where the ground truth's own keypoint
  is more than 2 px from the point's projection: the ground truth disagrees with
  its own point there, and the operation found the patch where the point says
  it is. The known observations the default refuses are mostly `too_far` (69
  seoul, 96 kerry) and `below_bar` (40, 104), then `peak_at_edge` and, on
  kerry_park, `shared_keypoint` (24).
- **One observation per place bites on kerry_park.** Across its 96 default
  calls, 290 candidates were refused as `shared_keypoint`: the candidate
  ground truth holds pairs of points at one place in one image. On seoul_bull
  it never fires.
- **Cost.** One call takes 8 ms on seoul_bull (280 candidates) and 16 ms on
  kerry_park (384 candidates), measuring every candidate in parallel.

### Chosen default and why

`PooledBasis { MedianMinusMad { k: 3 } }` with `ImageMad { k: 3, floor_px: 1 }`
and a 0.5 ZNCC floor. Both bars are read off the call's own data, so they
follow the capture's texture and the pose's error rather than a constant. It
has the fewest bad extra observations of any rule that keeps recall near 90%
on both captures (0 of 119 and 2 of 1356), and it halves the far rejoined
keypoints against every rule without a positional gate. A caller that wants
more recall at some cost in precision widens the positional bound
(`position_floor_px = 2`: +3 points of recall on seoul_bull and +2 on
kerry_park, for 7 and 9 more rejoined keypoints over 2 px).

## Non-goals

No retriangulation, no bundle adjustment, no new points. An occlusion test
against other geometry is not made: an occluded point is expected to fail the
photometric rule, because the target sees a different surface there, and the
evaluation's extra tracks show that it does. The stored `observation_confidence`
column is not read as the basis: neither ground truth carries it, and the
leave-one-out ZNCC the operation measures is the same measurement as the new
view's, which a column written by another step need not be.

## Open questions

- Whether the viewer offers the operation as its own image menu entry or runs
  it at the end of *Resect Image*.
- Whether a second pass after a bundle adjustment should re-offer the
  `too_far` refusals, whose offsets come mostly from the pose.

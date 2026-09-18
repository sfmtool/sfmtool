# Refitting the constellation query's affine from its consensus

**Status:** Draft. Amends
[core/features/kdf-constellation-query.md](../core/features/kdf-constellation-query.md):
it replaces the first non-goal ("no refinement of the affine from its inliers"),
adds one parameter and one field of `Constellation`, and changes what
`ConstellationMatch::affine` holds. Decided: the refit, its weighting and its
default, all from measurement. Decided against, and recorded here so it is not
measured a third time: a staged or progressive query. Open: the two questions
under "Open questions".

The [patch constellation query](../core/features/kdf-constellation-query.md)
finds the images that contain a patch and gives each a 2x3 affine placing the
patch in it. That affine is today the single best **three-point** model RANSAC
drew: it passes exactly through three correspondences and is scored by the
rest, but it is never fitted to them. Three keypoints carry three keypoints'
worth of localisation noise, and a model through them carries all of it. This
amendment fits the reported affine to the whole consensus by least squares,
weighted towards the patch centre, because the caller applies the warp at the
centre. It costs a 3x3 solve per reported image and changes nothing about which
images are found.

## What was measured, and what it decided

Two rounds, reports
[2026-09-17](../../reports/exp/2026-09-17-constellation-progressive-eval.md) and
[2026-09-18](../../reports/exp/2026-09-18-constellation-two-stage-eval.md),
harness
[`scripts/kdf_constellation_progressive_eval.py`](../../scripts/kdf_constellation_progressive_eval.py).

The question going in was whether to query progressively: fit the nearest 10
features, return if an image matched, otherwise add 10 and fit again, up to the
50 the standing spec recommends. The first round found the premise half right.
A feature's forest hits do not depend on which other features are in the
constellation, so stage `n` of any schedule is exactly a fresh query on the
nearest `n`, and every schedule could be replayed offline from one table of
prefix queries. Over 1,200 patch-stage comparisons on five captures a smaller
prefix found **no** image the nearest-50 missed (two single candidates), so any
early return is a pure loss of recall and a schedule has nothing to govern. But
where a small stage and the cap both found an image, the small stage's warp was
more accurate *near the centre*, which suggested a two-stage query (25, then 50)
that freezes each image's warp at the first stage accepting it.

The second round tested that directly and overturned it. Its metric is the
caller's own quantity: patch centres are keypoints with a ground-truth track,
the centre feature is **held out** of the constellation so no three-point model
can pass through it, and the error is the distance from the warped centre to
the track's observation in the found image. 400+ patches per corpus, 13,416
(patch, image) cases over eight corpora -- the five from before with the four
small ones re-solved for denser tracks, a second wide-baseline stills capture,
and DinoLedge decimated to every 10th and 20th frame as a controlled widening of
one trusted solve -- all arms paired over identical cases, bootstrap over
patches.

Change in the share of found images whose affine places the centre within 3 px,
against the shipped three-point model, 95% CI:

| corpus | cases | best staged lock | least squares | **weighted, sigma 0.5 R** | weighted, 0.25 R |
|---|--:|---|---|---|---|
| DinoLedge | 1,977 | +0.019 [+0.003, +0.034] | +0.067 | **+0.086 [+0.071, +0.102]** | +0.108 |
| DinoLedge, every 10th | 649 | +0.043 [+0.016, +0.071] | +0.089 | **+0.122 [+0.093, +0.150]** | +0.145 |
| DinoLedge, every 20th | 290 | +0.076 [+0.041, +0.114] | +0.138 | **+0.183 [+0.134, +0.233]** | +0.224 |
| Daegu | 1,036 | +0.014 [−0.002, +0.031] | +0.089 | **+0.115 [+0.089, +0.141]** | +0.118 |
| dino_dog_toy | 3,286 | +0.043 [+0.027, +0.059] | +0.089 | **+0.121 [+0.107, +0.135]** | +0.145 |
| kerry_park | 1,963 | +0.017 [+0.001, +0.036] | +0.082 | **+0.104 [+0.088, +0.122]** | +0.114 |
| seattle_backyard | 2,729 | +0.015 [+0.006, +0.024] | +0.058 | **+0.075 [+0.064, +0.087]** | +0.084 |
| seoul_bull | 1,486 | +0.013 [−0.003, +0.031] | +0.059 | **+0.076 [+0.061, +0.091]** | +0.079 |

What the first round saw was mostly the three-point model's **variance**, with
locality a smaller effect on top. A staged lock swaps one three-point model for
another drawn from fewer, nearer points; a refit removes the three-point noise
outright and can still lean towards the centre. Head to head the best lock,
chosen post hoc per corpus, is distinguishably worse than the refit on all eight
(−0.065 to −0.148), and the gap *widens* with the baseline in the decimation
series, so there is no capture shape where staging wins. The lock also degrades
the warp over the rest of the disc (DinoLedge: share of images with the ground
truth's correspondences within 3 px over the whole disc 0.78 -> 0.69, where the
weighted refit takes it to 0.93) and cannot skip work: 38,966 of 38,970
candidates' correspondence lists grew between stage 25 and the cap, so every
candidate is fitted twice, for +9 to +27% wall time against the refit's +0.7 to
+3.6% (numpy, an upper bound).

So: **no stages, no start size, no increment, no per-stage acceptance bar.** The
constellation stays at fifty features, `min_inliers` stays 8, and the one change
is what is done with the consensus once RANSAC has found it.

## Interface

```rust
pub enum AffineRefit {
    /// Report the best three-point model as drawn.
    None,
    /// Least squares over the consensus, every inlier weighted alike.
    LeastSquares,
    /// Least squares with Gaussian weight in the distance of an inlier's
    /// constellation position from `Constellation::center`, the standard
    /// deviation being `sigma` times the constellation's radius about it.
    CenterWeighted { sigma: f64 },
}

pub struct ConstellationParams {
    // ... as today, plus:
    pub refit: AffineRefit,        // DEFAULT: CenterWeighted { sigma: 0.5 }
}

pub struct Constellation<'a, S> {
    pub positions: &'a [[f32; 2]],
    pub descriptors: ConstellationDescriptors<'a, S>,
    pub image_index: Option<u32>,
    pub center: Option<[f32; 2]>,  // new
}
```

`ConstellationMatch` keeps its fields. `affine` holds the refitted model;
`inliers` and `inlier_correspondences` remain the consensus of the three-point
model that won, which is the set the refit was computed from.

**Why the centre is part of the constellation and not of the params.** It is a
fact about the patch, like the positions are, and the two `*_at_pixel` /
`*_from_keypoints` entry points already hold it: they select the constellation
around it and then throw it away. They now pass it through, so the
[bench's descriptor search](../core/bench/editable-track.md), which calls
`constellation_from_keypoints`, gets the centre-weighted warp with no change of
its own. A caller of `constellation_query` that assembled its positions some
other way may have no centre to give; `None` under `CenterWeighted` fits as
`LeastSquares`, which is the honest reading of "no point matters more than
another" and is already most of the gain (the third column above).

**Why the radius is measured and not passed.** The weight's scale is `sigma`
times the largest distance from the centre to any constellation position. The
radius a caller *asked* for can be much larger than the disc its features fill
-- a radius rule is a prediction -- and a sigma proportional to an empty rim
would flatten the weights towards `LeastSquares` without anyone having chosen
that.

**Why an enum and not a bare `sigma`.** Off, unweighted and weighted are three
behaviours, and encoding two of them as `0.0` and infinity would put a
correctness condition on a float comparison. `None` is kept because the
resident/file-backed parity tests and anyone diagnosing RANSAC itself want the
model as drawn.

**Why 0.5.** `0.25` is the best centre warp on all eight corpora, and the worst
refit over the disc (0.80 to 0.93 of images within 3 px, against 0.91 to 0.98
unweighted), because it all but discards the rim. `0.5` gives back 0.01 to 0.04
of the centre share and keeps essentially all of the disc share; at the rim its
weight is `exp(-2)`, about an eighth, so the rim still constrains the linear
part. The bench uses both halves of the warp -- the translation seeds the
observation's pixel, the 2x2 seeds its shape -- and the linear part's error
against a ground-truth local affine roughly halves at 0.5 (rotation 2.13° ->
1.25° on seoul_bull). A caller that only ever maps the centre sets 0.25.

Python: both query methods take `refit="center_weighted" | "least_squares" |
"none"` and `refit_sigma=0.5`, and `constellation_query` takes `center=None`.

## The fit

For inliers `i` with constellation positions `p_i`, matched positions `q_i` and
weights `w_i`, the affine minimises `sum w_i |A p_i + t - q_i|^2`. The two rows
of `[A | t]` share one 3x3 normal matrix `sum w_i [p_i; 1][p_i; 1]^T` and differ
only in the right-hand side, so it is one factorisation and two back-solves.
Positions are taken relative to the centre (or the inliers' centroid without
one) before the sums are formed and the translation is carried back afterwards:
pixel coordinates reach the thousands, and squaring them uncentred spends
digits the solve then needs.

The refitted model goes through the same determinant guards as a three-point
one -- non-positive determinant, or `sqrt(det)` outside `[1/max_scale,
max_scale]` -- and so does a singular normal matrix, which is what inliers
collinear in the query image produce. Any refusal reports the three-point model
instead; the candidate is never dropped on the refit's account, because the
consensus that admitted it still stands.

**One pass, no re-selection.** Re-selecting inliers under the refitted model and
fitting again, up to three rounds, moved the centre share by at most 0.007 on
any corpus. Not iterating also keeps `inliers` meaning one thing: within
`threshold_px` of the model RANSAC chose. An inlier may therefore sit slightly
outside `threshold_px` of the *reported* affine, and the spec will say so.

**Determinism is untouched.** The refit is a pure function of the consensus, and
the consensus is already identical across the two forests, so
`the_two_forests_answer_identically` keeps asserting bit-equal warps with the
refit on.

## What folds into the standing spec

- The non-goal "no refinement of the affine from its inliers" goes; "no
  homography" stays.
- "Affine, and three points" gains the distinction this draft turns on: three
  points are how a model is *found*, because the sample size is what RANSAC's
  cost is exponential in, and are a poor way to *report* one.
- "Choosing the constellation size" keeps fifty, and gains the staged-query
  result in a paragraph: a smaller prefix finds nothing the cap misses, so there
  is no schedule to choose.
- The parameters table gains `refit`.

## Testing

In `constellation/tests.rs`, over the existing planted corpus:

- plant the warped image with per-keypoint jitter of a pixel or two, so a
  three-point model is measurably off, and assert the refitted affine is nearer
  the planted one at the centre than the `AffineRefit::None` model from the same
  query, and that both report the same inlier set;
- drive the fit directly: exact correspondences return the exact affine under
  every mode; a perspective-like warp (affine plus a quadratic term) is fitted
  closer at the centre by `CenterWeighted` than by `LeastSquares`; collinear
  inliers and a consensus whose least-squares model is a reflection both fall
  back to the three-point model;
- `center: None` under `CenterWeighted` equals `LeastSquares` bit for bit;
- the parity test runs with the default refit.

The Python binding test checks the three `refit` spellings and that a bad one is
a `ValueError`.

## Open questions

- **Should the scale guard on the refitted model be the same `max_scale`?** It
  is here, and the fallback fired too rarely to show in any column, so nothing
  measured argues for a different bound. Revisit only if a capture shows it
  firing.
- **Does the bench want 0.25?** It maps the centre and warps a shape; 0.5 was
  chosen for serving both. Settling it needs the bench's own downstream score
  (does the refinement converge from the seed) rather than a residual, which
  neither round measured.

## Not part of this amendment

The second round's DinoLedge timings needed a 1 GiB cache: at
`LazyKdForestOptions`' default `cache_bytes` of 256 MiB a repeated 50-feature query
against that 1,494 MB index costs 490 ms, and at 512 MiB it costs 14 ms. That is
worth more to a caller than anything above, but it is a property of the
[lazy query path](../core/features/lazy-kdforest-query.md) and wants its own
amendment there -- a budget derived from the file's size, or a documented rule.

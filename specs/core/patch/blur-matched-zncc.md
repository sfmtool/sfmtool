# Blur-Matched ZNCC

Two photographs of the same piece of surface rarely show it equally sharply:
one is farther away, out of focus, moved during the exposure, or seen through
a part of the lens that smears it along one direction. Normalized correlation
(ZNCC) between the two views' tiles charges the sharper one for the fine detail
the other lacks, so a sharp, well-aligned view reads as if it disagreed with
its track, and a blurry view looks worse beside a sharp one than it does beside
other blurry views. **Blur-matched ZNCC** removes that charge: before two tiles
are correlated, the sharper one is blurred to the other's sharpness, direction
by direction, so that what remains of the disagreement is content (an
occluder, another surface, a shadow, parallax) rather than sharpness. It is a
score for judging views. It is never used to place a view: blurring a template
before aligning to it moves the correlation peak no closer to the truth (see
[Non-goals](#non-goals)).

Each tile's sharpness is read from its ZNCC self-similarity ellipse
([zncc-self-similarity-radius.md](zncc-self-similarity-radius.md)): the
ellipse of the shifts at which the tile still matches itself, short along the
directions in which it holds fine detail. The difference between two tiles'
ellipses says which tile is sharper along which direction and by how much, and
an empirically fitted mapping turns that into the width of a Gaussian blur.

Two consumers read it. The bench's reference-view rule reads blur-matched
agreement by default ([reference-view.md](reference-view.md) § "Blur-matched
agreement"); member-coherence validation can read it, and does not by default
([member-coherence-validation.md](member-coherence-validation.md) § "Blur
matching"). The measurements behind both choices are below.

## Rust API

The kernel lives in
[blur_matched.rs](../../../crates/sfmtool-core/src/patch/blur_matched.rs), with
the blur in [blur.rs](../../../crates/sfmtool-core/src/patch/blur_matched/blur.rs)
and the pair readings in
[tiles.rs](../../../crates/sfmtool-core/src/patch/blur_matched/tiles.rs). The
track-level reading the bench uses is `blur_matched_agreement` in
[agreement.rs](../../../crates/sfmtool-core/src/patch/reference_view/agreement.rs).

```rust
// patch::blur_matched
pub enum PairMatching { Plain, BlurMatched, BlurMatchedAboveRatio(f64) }
impl PairMatching {
    pub fn min_ratio(self) -> Option<f64>;   // None for Plain, 1 for BlurMatched
    pub fn is_blur_matched(self) -> bool;
    pub fn name(self) -> &'static str;       // "plain", "blur_matched", "blur_matched_above_ratio"
    pub fn from_name(name: &str, ratio: f64) -> Option<Self>;
}
pub struct BlurCovariance { pub xx: f64, pub xy: f64, pub yy: f64 } // grid px², x column-right, y row-down
pub struct PairBlur { pub a: BlurCovariance, pub b: BlurCovariance }

pub fn blur_sigma(blurrier: f64, sharper: f64) -> f64;             // the fitted mapping
pub fn isotropic_blur_sigma(blurrier: f64, sharper: f64) -> f64;   // its isotropic fit
pub fn pair_blur(a: &[[f64; 2]; 2], b: &[[f64; 2]; 2], min_ratio: f64) -> PairBlur;
pub fn pair_blur_for(a: &[[f64; 2]; 2], b: &[[f64; 2]; 2], matching: PairMatching) -> PairBlur;
pub fn ladder_level(sigma: f64) -> Option<usize>;

pub fn blur_tile(values: &[f32], channels: usize, side: usize, data: &[bool],
    cov: BlurCovariance, out: &mut [f32], scratch: &mut BlurScratch);

pub struct TilePlanes { pub values: Vec<f32>, pub data: Vec<bool>, pub side: usize, pub channels: usize }
pub struct PairReadings { pub whole: f64, pub grid: [[f64; 3]; 3] }
pub fn pair_zncc_readings(a: &TilePlanes, b: &TilePlanes, window: &[f64]) -> PairReadings;

pub enum BlurMatchKernel { Anisotropic, IsotropicLadder }
pub struct BlurMatchedPairs {
    pub k: usize, pub whole: Vec<f64>, pub grid: Vec<[[f64; 3]; 3]>,
    pub blurred: Vec<bool>, pub pairs: usize, pub pairs_blurred: usize,
}
pub fn blur_matched_pairs(tiles: &[&TilePlanes], ellipses: &[Option<[[f64; 2]; 2]>],
    matching: PairMatching, kernel: BlurMatchKernel, window: PatchWindow,
    rows: Option<&[bool]>, progress: &Progress<'_>) -> BlurMatchedPairs;

// patch::reference_view
pub struct BlurMatchedAgreement { pub pair_zncc: Vec<f64>, pub cells: CellAgreement, pub pairs: BlurMatchedPairs }
pub fn blur_matched_agreement(tiles: &[&ViewTile], ellipses: &[Option<[[f64; 2]; 2]>],
    matching: PairMatching, kernel: BlurMatchKernel, window: PatchWindow,
    progress: &Progress<'_>) -> BlurMatchedAgreement;
```

**Why it is shaped this way.** The kernel takes the ellipse matrices rather
than tiles to read them from, because every consumer already holds them: the
bench reads each view's self-similarity once for its own columns, and member
coherence reads them on its own renders. `pair_blur` is a pure function of two
ellipses, so the choice of blur can be tested on constructed numbers, and the
blur is a function of one tile and a covariance, so it can be checked against a
direct 2-D convolution and an exact answer. `PairMatching` is one enum for
every consumer, so a consumer's option reads the same on the wire, in Python
and in Rust. `blur_matched_pairs` reads both the whole tile and the ZNCC grid's
cells from each pair it blurs, because the reference rule's agreement test and
its cell check both read the same pairs, and the blur is most of the cost.
`rows` limits the pairs read to those with a view in a set, for a caller that
needs only some views' medians.

```rust
use sfmtool_core::patch::blur_matched::{
    blur_matched_pairs, BlurMatchKernel, PairMatching, TilePlanes,
};
use sfmtool_core::patch::normal_refine::PatchWindow;
use sfmtool_core::progress::Progress;

fn report(tiles: &[TilePlanes], ellipses: &[Option<[[f64; 2]; 2]>]) {
let refs: Vec<&TilePlanes> = tiles.iter().collect();
let pairs = blur_matched_pairs(
    &refs,
    ellipses,
    PairMatching::BlurMatchedAboveRatio(1.25),
    BlurMatchKernel::Anisotropic,
    PatchWindow::GaussianDisk { sigma: 0.6 },
    None,
    &Progress::none(),
);
println!("{} of {} pairs blurred; view 0's median {:.3}",
    pairs.pairs_blurred, pairs.pairs, pairs.row_middle(0));
}
```

## Theory

### Which tile is blurred, along which direction

Let `E_a` and `E_b` be the two tiles' self-similarity ellipse matrices, in grid
px² (each ellipse is `dᵀ E⁻¹ d = 1`, its eigenvalues the squared semi-axes).
The eigenvectors `u₁`, `u₂` of `E_b − E_a` are the directions along which the
two ellipses differ most and least. Along each `u`, the two half-widths are
`l_a = √(uᵀ E_a u)` and `l_b = √(uᵀ E_b u)`. The tile with the shorter one is
the sharper along `u`, and it is blurred along `u` by a 1-D Gaussian of width
`σ = blur_sigma(longer, shorter)`. Its covariance `σ² u uᵀ` is added to that
tile's blur. Either tile, or both, may be blurred: a view sharp along one axis
and blurry along the other, which is what an oblique or fisheye view usually
is, is blurred only along the axis where the other view is blurrier. A view
blurred along every axis by an isotropic Gaussian instead loses the detail
across the blur direction that the blurrier view still holds; on a striped
tile the isotropic blur reached a ZNCC of 0.83 where the directional one
reached 0.90.

Blurring the blurrier tile as well, to a common coarser width, is the simple
alternative. It throws away detail both views share, and in the prototype it
was the worst design on every separation test: the chance that a member scores
above a planted wrong view was 0.953, against 0.966 plain and 0.967 for the
directional blur.

### The ellipse-to-blur mapping

`σ = k · s^q · (r² − 1)^p`, with `r = longer / s` and `s` the shorter length,
read as at least `BLUR_MAP_MIN_LENGTH` = 0.05 grid px, and capped at
`MAX_BLUR_SIGMA` = 3 grid px:

| Constant | Value |
|---|---|
| `BLUR_MAP_SCALE` (`k`) | `0.9975` |
| `BLUR_MAP_RATIO_POWER` (`p`) | `0.1744` |
| `BLUR_MAP_LENGTH_POWER` (`q`) | `0.2613` |
| `ISOTROPIC_BLUR_MAP` (`k`, `p`, `q`, on semi-major axes) | `0.8361`, `0.3635`, `0.5588` |

The mapping was fitted, not derived. On every track of the two ground truths,
the reference view's tile and the next two views' tiles were blurred by known
anisotropic Gaussians (`σ` up to 2.5 by up to 1.0, at a per-track angle), the
ellipse read again after each blur, and the model fitted by least squares in
logs over the samples whose length along the blur changed measurably
(`r > 1.02`) and stayed under the reading's 3 px cap: 11,949 of 19,539
samples, a spread of a factor of 1.66 about the fit. The isotropic form was
fitted the same way on isotropic blurs from 0.25 to 2.5 (6,406 of 10,521
samples, a factor of 1.36).

The response is far from the `σ² = longer² − shorter²` a Gaussian model of the
ellipse would give, for three reasons:

- **A small blur does not register.** A blur of 0.25 grid px never changes
  the ellipse: the reading's whole-pixel shifts and its tolerance hide it. So
  any measurable change in length maps to a blur of at least about 0.5 grid
  px, and two ellipses that differ at all are blurred by at least that much.
  This is why the skip ratio below exists.
- **A large blur saturates.** At `σ` 2 to 2.5, half or more of the tiles reach
  the 3 px cap, and the fit, made on uncapped samples, reads `σ = 2.5` back as
  about 1.6. Over the ellipse's capped range the mapping stays under about 1.9
  grid px, so the cap at 3 only guards a reading from a wider search.
- **A blur along a direction with no detail barely changes the ellipse.** On a
  directional texture, 24–45% of the samples showed no change; such a blur
  also does not change the match.

### Skipping pairs that barely differ

`PairMatching::BlurMatchedAboveRatio` leaves a direction alone where the two
lengths along it differ by less than a factor, and a pair left alone along both
directions is correlated plain at no extra cost.
`DEFAULT_MIN_ELLIPSE_RATIO` is `1.25`. On real tracks it skips about half the
pairs, and changes what a consumer reads by little:

| Ratio | Bench pairs blurred (36,563 pairs, 661 tracks) | Member-coherence pairs blurred (425,283 pairs, 8,484 points) | Change in a view's blur-matched median, median / p90 | Change in its cell deficit, median / p90 |
|---|---|---|---|---|
| 1 (every difference) | 100% | 100% | - | - |
| 1.1 | 82% | - | 0.0007 / 0.005 | 0.003 / 0.018 |
| 1.25 | 55% | 48% | 0.004 / 0.013 | 0.007 / 0.033 |
| 1.5 | 29% | 24% | 0.009 / 0.023 | 0.011 / 0.048 |

The share varies with the data: on BadlandPanorama, whose distant views are
all about equally sharp, a ratio of 1.25 blurs 6% of the pairs; on
MossyRailing, video with varying motion blur, 91%.

### The blur

The blur is an anisotropic Gaussian applied by normalized convolution over the
samples that carry data: the colour planes are blurred premultiplied by the
data mask, the mask is blurred with them, and the blurred colour is divided by
the blurred mask at each sample that carries data. A sample without data (off
the photograph, or outside member coherence's common support) neither
contributes a value nor receives one, and neither does anything past the
tile's edge, so a tile next to the border is not darkened by it.

The Gaussian with covariance `Σ = [[a, b], [b, d]]` is applied as two 1-D
passes (Geusebroek, Smeulders and van de Weijer, "Fast anisotropic Gauss
filtering", 2003): one along the slanted line `(b/d, 1)`, with variance `d`
measured in rows, and one along `x` with the remaining variance `a − b²/d`.
The two covariances add up to `Σ`. Where `a > d` the axes swap roles, so the
slope is at most 1 in magnitude. A tap of the slanted pass falls between two
samples of its row and is read by linear interpolation, which adds a variance
of `f(1 − f)` along the axis for a tap at fraction `f`; the axis pass subtracts
the tap-weighted mean of it from its own variance. A 1-D blur along `u` has no
axis pass left, so the interpolation's own variance is the one error not taken
back.

## Cost

One thread, release build, a 24 × 24 three-channel tile (the mask is a fourth
plane):

| Step | µs |
|---|---|
| Plain pair readings (whole tile and the nine cells) | 3.5 |
| The two passes, isotropic `σ` 1 / 1-D `σ` 1 at 30° / `σ` 1.5 × 0.8 at 30° / isotropic `σ` 2 | 4.9 / 4.3 / 9.3 / 8.9 |
| Direct 2-D convolution, the same isotropic `σ` 1 / 1.5 × 0.8 at 30° / `σ` 2 | 48 / 112 / 157 |
| `pair_blur` from two ellipses | 0.1 |
| A blur-matched pair, anisotropic (one tile blurred, readings) | 14.2 |
| A blur-matched pair, isotropic ladder | 11.9 |

The two passes run 10 to 18 times faster than the direct convolution. The
self-similarity reading that supplies an ellipse costs about 7 µs per reading
where every sample carries data, and about 50 µs on member coherence's renders,
whose common support leaves the corners without data and so takes the slower
route that visits every sample.

**No AVX2 form.** Every pass is a sum of whole shifted rows, `dst[x] += w ·
src[x + s]`, which the compiler vectorizes. The same code compiled for AVX2
and FMA, chosen at run time, ran from 13% faster to 13% slower than the
default build depending on the kernel's shape (rows of 24 to 32 samples leave
little for wider registers), so the kernel has none.

**The isotropic ladder.** `BlurMatchKernel::IsotropicLadder` blurs the
sharper tile, by semi-major axis, isotropically, with the width snapped to the
nearest of six levels `0.5 · √2ⁿ` (`LADDER_SIGMAS`), and keeps each view's
blurred levels, so each view is blurred at most once per level however many
pairs it is in. On the bench its reference-view phase cost 12 to 25% less than
the anisotropic kernel's on tracks of up to 20 views and 36% less on tracks of
21 views or more (4.3 against 6.7 ms); with a ratio of 1.25 on both, 4.1
against 5.0 ms. Its readings agree with the anisotropic kernel's to within one
or two of the review cases and to within 0.7 points on the
planted blur below. It is isotropic, which the anisotropic kernel was chosen
over for oblique and directional views, so it is an option rather than the
default.

## Accuracy

**The two passes against the exact blur.** On a tile of four sinusoids, whose
blur by a Gaussian is known exactly (each amplitude scales by `exp(−½ kᵀ Σ
k)`), over 72 covariances (semi-axes from 0.6 to 2, with 1-D blurs, at nine
angles), the passes are at most 3.6 grey levels off the exact answer on
sinusoids spanning ±115, against 0.34 for the direct 2-D convolution, which
has no 1-D form. The error is the linear interpolation's, largest for the
narrowest blurs at angles between the grid's axes and its diagonals. On a
textured tile with samples missing, the two blurs agree to a ZNCC above 0.998.

**The kernel against the prototype.** The Rust kernel reproduces the Python
prototype the design was measured with (in Python, on the same tiles)
on 5,922 pairs of pool tiles to a median difference of 0.0004 in ZNCC (p90
0.02, correlation 0.992). The prototype used clamped borders where the kernel
renormalizes, and a sampled 2-D kernel where the kernel uses the two passes.

## What it buys

**A member made blurrier.** From 987 pool tracks of ten datasets (at least five
views each), one member's tile was blurred by `σ` 1 or 2 grid px (its
semi-major axis went from 0.98 to 1.72 and 2.86 px, median), its ellipse read
again, and its median ZNCC with the other members compared with the threshold
that catches 90% of planted wrong views (lookalike tiles of other points,
neighbours, and members shifted by 2 to 4 px):

| Reading | Change in its median, `σ` 1 / 2 | Below the threshold, `σ` 1 / 2 |
|---|---|---|
| Plain | +0.003 / −0.033 | 8.1% / 19.5% |
| Blur-matched, anisotropic | +0.016 / +0.012 | 4.0% / 3.0% |
| Blur-matched, ratio 1.25 | +0.019 / +0.015 | 4.1% / 3.0% |
| Blur-matched, isotropic ladder | +0.019 / +0.025 | 4.7% / 2.7% |

So a member blurred well past its track, which plain ZNCC rejects one time in
five, is kept, while wrong views stay separated: measured with the prototype
on the same planted wrong views, the chance that a member scores above a wrong
view was 0.967 blur-matched against 0.966 plain. The real tracks contain few members much blurrier than all their partners, which
is why a consumer's verdicts move little on real data (below).

**On the consumers**, measured on the review cases and samples of the ten
datasets of the reference-view work:

| Consumer | Option | Added cost | Effect | Default |
|---|---|---|---|---|
| Reference view: agreement test and cell check ([reference-view.md](reference-view.md)) | blur-matched, ratio 1.25 | +0.20 ms per track (median; p90 1.4 ms), 3.2% of an evaluation | 30 of 77 hand picks exactly, against 28 plain (tune half 19, held-out 11 against 18 and 10) | on |
| Member coherence's decision ([member-coherence-validation.md](member-coherence-validation.md)) | blur-matched, ratio 1.25 | 2.3 × the plain run on one thread (2.1 × on all) | a planted member blurred by `σ` 2 is evicted 2.0% of the time against 4.9% plain; verdicts change on 2.1% of real points, in both directions | off |

## Implementation notes

**The ellipse is the measurement of the tile being blurred.** A consumer passes
the ellipse of the very render it correlates: the bench the `R×R` tile's whole
reading, member coherence a reading of its own render over the common support.
An ellipse read on another render of the view (another resolution, sampler or
support) describes another tile.

**The first pass covers the band the second reads.** The tile sits in a buffer
padded with zeros by both passes' reach, and the first pass writes the tile and
the second pass's reach round it, so the two passes compose as one 2-D
convolution of the tile extended by zeros rather than of a tile cut off after
the first pass. A band is computed in whole 8-sample groups; the columns past
the tile it adds are written and never read back. The buffer's padding stays
zero between calls with the same geometry, since the second pass writes only
the tile's rows, and is cleared when the geometry changes.

**A pair left plain reads exactly the plain value.** `pair_zncc_readings`
gathers the cells' sums in the same order and by the same formula as
`pair_zncc_grid`(reference-view.md), so a pair the ratio leaves alone reads
the plain cell grid bit for bit, and member coherence copies its plain value
for such a pair.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `DEFAULT_MIN_ELLIPSE_RATIO` | `1.25` | The factor by which two lengths along a direction must differ for `BlurMatchedAboveRatio` to blur along it; it skips about half the pairs on real tracks |
| `BLUR_MAP_MIN_LENGTH` | `0.05` grid px | The shortest sharper length the mapping reads; the shortest length among the samples it was fitted on |
| `MAX_BLUR_SIGMA` | `3` grid px | The widest blur the mapping returns |
| `LADDER_SIGMAS` | `0.5 · √2ⁿ`, `n = 0 .. 5` | The isotropic ladder's widths; a width under 0.35 is not applied |
| `MIN_WINDOWED_SAMPLES` | `8` | The fewest samples with data in both tiles and weight in the window a whole-tile reading is taken over |

Each consumer's option and its default are in that consumer's spec. The
constants are defined in
[blur_matched.rs](../../../crates/sfmtool-core/src/patch/blur_matched.rs) and
[tiles.rs](../../../crates/sfmtool-core/src/patch/blur_matched/tiles.rs).

## Python bindings

`sfmtool._sfmtool.patches.blur_matched_zncc_matrix(tiles, *, ellipses=None,
matching="blur_matched", min_ellipse_ratio=1.25, kernel="anisotropic",
window="gaussian_disk", window_sigma=0.6)` reads every pair of a `(k, R, R, C)`
uint8 stack of tiles (a fourth channel is alpha, 0 marking a sample without
data). `ellipses` is `(k, 2, 2)` float64, NaN for a tile without one, or `None`
to read each tile's whole self-similarity here. It returns a dict: `zncc`
`(k, k)`, `zncc_grid` `(k, k, 3, 3)`, `blurred` `(k, k)` bool, `pairs`,
`pairs_blurred` and `ellipse_matrix` `(k, 2, 2)`. It raises `ValueError` for a
stack that is not square tiles of 3 or more on a side with 1 to 4 channels, an
`ellipses` of the wrong shape, an unknown name, or a ratio under 1.

```python
from sfmtool._sfmtool.patches import blur_matched_zncc_matrix

out = blur_matched_zncc_matrix(tiles, matching="blur_matched_above_ratio")
print(out["pairs_blurred"], "of", out["pairs"], "pairs blurred")
```

`PatchCloud.validate_member_coherence(..., matching=, min_ellipse_ratio=)` and
the bench's readings carry the consumers' forms
([member-coherence-validation.md](member-coherence-validation.md),
[editable-track.md](../bench/editable-track.md) § "Python bindings").

## Testing

[blur_matched/tests.rs](../../../crates/sfmtool-core/src/patch/blur_matched/tests.rs)
checks the mapping (zero without a difference, growing with it, its fitted
value, the floor); that equal ellipses blur nothing; that each tile is blurred
only along the directions it is sharper in, on axis-aligned and rotated pairs;
the skip ratio, per direction; the two passes against the exact blur of a tile
of sinusoids and against the direct 2-D convolution, and the two against each
other round missing samples; that a sample without data neither gives nor takes
a value and a flat tile stays flat at the edge and round holes; that a 1-D blur
leaves a texture along it alone; that a tile and a blurred copy of it read a
ZNCC near 1 blur-matched and well above plain; that equally sharp views read
the plain value; the ladder against the anisotropic kernel; and `rows`. An
ignored test, `timing`, prints the cost table above.
[reference_view/tests.rs](../../../crates/sfmtool-core/src/patch/reference_view/tests.rs),
[member_coherence/tests.rs](../../../crates/sfmtool-core/src/patch/member_coherence/tests.rs)
and [bench/tests/reference_view.rs](../../../crates/sfmtool-core/src/bench/tests/reference_view.rs)
test the consumers. The Python tests are in
[test_blur_matched_rust_bindings.py](../../../tests/rust_bindings/patches/test_blur_matched_rust_bindings.py).

## Non-goals

- **Alignment.** No kernel blurs a template to place a view. On the two ground
  truths, blurring the reference view to each aligned view, by the footprint,
  by the closest self-similarity ellipse or by any of the blur-matched designs,
  changed the localization error by −0.003 to +0.012 px on seoul_bull and made
  it worse by 0.01 to 0.05 px on kerry_park, under both the localizer's search
  and a coarse-to-fine one; the blur that raised the ZNCC most was worst by
  0.03 to 0.05 px. A symmetric blur of the template is a symmetric blur of the
  correlation surface: it leaves the peak where it is on average and lowers its
  curvature, so the peak is placed less precisely. Alignment runs against the
  unblurred tiles, and a blur-matched ZNCC may be computed afterwards for the
  score.
- **The bench's ZNCC bars.** A row's `zncc` is the localizer's leave-one-out
  ZNCC against the IRLS-fused consensus of the other rows, scored inside the
  localizer's search at the correlation peak. Blur-matching it would need each
  row's leave-one-out template and its self-similarity reading, which the
  localizer builds per round and does not return, and the bars would need
  measuring again; the localizer is also the alignment, which stays unblurred.
  Proposed with Part 6 of
  [../../drafts/sharper-patch-bitmap.md](../../drafts/sharper-patch-bitmap.md).
- **The fuse's IRLS residuals.** Each view's residual against the weighted mean
  is recomputed every iteration, and the mean, the blurrier of every pair,
  changes with the weights, so blur-matching the residuals needs the mean's
  ellipse every iteration and changes the stored bitmap, not only a score. How
  the bitmap is computed is Part 5 of the same draft.
- **Deconvolution.** A blurry tile is not sharpened; the sharper one is
  blurred.

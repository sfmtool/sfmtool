# Blur-Matched ZNCC

Two photographs of the same piece of surface rarely show it equally sharply:
one is farther away, out of focus, moved during the exposure, or seen through
a part of the lens that smears it along one direction. Normalized correlation
(ZNCC) between the two views' tiles scores the sharper one lower for detail the
other lacks, so a sharp, well-aligned view reads as if it disagreed with its
track, and a blurry view reads worse beside a sharp one than it does beside
other blurry views. **Blur-matched ZNCC** removes that penalty: before two tiles
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
ellipses says which tile is sharper along which direction and by how much. The
width of the blur is found by measurement: the sharper tile is blurred, its
ellipse read again with the same reading, and the width corrected until the
blurred tile's ellipse is as long as the blurrier tile's along each direction,
so the blurred tile is as sharp as its partner by the same measure that said it
was sharper.

Two consumers read it. The bench's reference-view rule reads blur-matched
agreement by default ([reference-view.md](reference-view.md) § "Blur-matched
agreement"); member-coherence validation can read it, and does not by default
([member-coherence-validation.md](member-coherence-validation.md) § "Blur
matching"). The measurements behind both choices are below.

## Rust API

The kernel lives in
[blur_matched.rs](../../../crates/sfmtool-core/src/patch/blur_matched.rs), with
the blur in [blur.rs](../../../crates/sfmtool-core/src/patch/blur_matched/blur.rs),
the search for its width in
[search.rs](../../../crates/sfmtool-core/src/patch/blur_matched/search.rs)
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

// Which tile is blurred along which direction, and how long it should become.
pub struct BlurDirection { pub u: [f64; 2], pub sharper: f64, pub blurrier: f64 }
pub struct TileDirections { /* at most two */ }
impl TileDirections {
    pub fn as_slice(&self) -> &[BlurDirection];
    pub fn is_empty(&self) -> bool;
    pub fn estimated_blur(&self) -> BlurCovariance; // the first width tried, per direction
}
pub struct PairDirections { pub a: TileDirections, pub b: TileDirections }
pub fn pair_directions(a: &[[f64; 2]; 2], b: &[[f64; 2]; 2], min_ratio: f64) -> PairDirections;
pub fn pair_directions_for(a: &[[f64; 2]; 2], b: &[[f64; 2]; 2], matching: PairMatching) -> PairDirections;
pub fn estimated_blur_sigma(blurrier: f64, sharper: f64) -> f64;
pub fn length_along(e: &[[f64; 2]; 2], u: [f64; 2]) -> f64;

// The width, by measurement: blur, read the ellipse again, correct.
pub struct MatchedBlur { pub cov: BlurCovariance, pub probes: u32, pub error: f64 }
pub fn match_blur(tile: &TilePlanes, dirs: &[BlurDirection],
    read: impl FnMut(&[f32]) -> Option<[[f64; 2]; 2]>,
    out: &mut Vec<f32>, scratch: &mut MatchScratch) -> MatchedBlur;

pub fn blur_tile(values: &[f32], channels: usize, side: usize, data: &[bool],
    cov: BlurCovariance, out: &mut [f32], scratch: &mut BlurScratch);

pub struct TilePlanes { pub values: Vec<f32>, pub data: Vec<bool>, pub side: usize, pub channels: usize }
impl TilePlanes {
    // An interleaved u8 tile: 1 or 2 channels grey (and alpha), 3 or 4 RGB (and alpha).
    pub fn from_interleaved(samples: &[u8], side: usize, stride: usize, data: &[bool]) -> Self;
    pub fn blurred(&self, cov: BlurCovariance, scratch: &mut BlurScratch) -> Self;
}
// The whole-tile self-similarity reading over the samples with data.
pub fn read_tile_ellipse(values: &[f32], channels: usize, side: usize, data: &[bool])
    -> Option<[[f64; 2]; 2]>;
pub struct PairReadings { pub whole: f64, pub grid: [[f64; 3]; 3] }
pub fn pair_zncc_readings(a: &TilePlanes, b: &TilePlanes, window: &[f64]) -> PairReadings;

pub enum BlurMatchKernel { Anisotropic, IsotropicLadder }
pub struct BlurMatchedPairs {
    pub k: usize, pub whole: Vec<f64>, pub grid: Vec<[[f64; 3]; 3]>,
    pub blurred: Vec<bool>, pub pairs: usize, pub pairs_blurred: usize,
    pub ellipse_reads: usize,
}
impl BlurMatchedPairs {
    pub fn row_middle(&self, v: usize) -> f64; // median of view v's whole-tile readings
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
coherence reads them on its own renders. `pair_directions` is a pure function
of two ellipses, so which tile is blurred along which direction can be tested
on constructed numbers. `match_blur` takes the reading as a closure, because
the width is set by comparing a blurred tile's reading with its partner's, and
the two must be the same reading: the bench and the bindings read the whole
tile over its samples with data (`read_tile_ellipse`), member coherence reads
the largest square inside its common support. The blur is a function of one
tile and a covariance, so it can be checked against a direct 2-D convolution
and an exact answer. `PairMatching` is one enum for every consumer, so a
consumer's option reads the same on the wire, in Python and in Rust.
`blur_matched_pairs` reads both the whole tile and the ZNCC grid's cells from
each pair it blurs, because the reference rule's agreement test and its cell
check both read the same pairs, and the blur is most of the cost. `rows`
limits the pairs read to those with a view in a set, for a caller that needs
only some views' medians.

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
    println!(
        "{} of {} pairs blurred; view 0's median {:.3}",
        pairs.pairs_blurred,
        pairs.pairs,
        pairs.row_middle(0)
    );
}
```

## Theory

### Which tile is blurred, along which direction

Let `E_a` and `E_b` be the two tiles' self-similarity ellipse matrices, in grid
px² (each ellipse is `dᵀ E⁻¹ d = 1`, its eigenvalues the squared semi-axes).
The eigenvectors `u₁`, `u₂` of `E_b − E_a` are the directions along which the
two ellipses differ most and least. Along each `u`, the two half-widths are
`l_a = √(uᵀ E_a u)` and `l_b = √(uᵀ E_b u)`. The tile with the shorter one is
the sharper along `u`, and it is blurred along `u` by a 1-D Gaussian whose
width is found by measurement (below). Its covariance `σ² u uᵀ` is added to
that tile's blur. Either tile, or both, may be blurred: a view sharp along one axis
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

### The width, found by measurement

Along each direction `u` a tile is to be blurred along, `match_blur` looks for
the width `σ` at which the blurred tile's ellipse is as long along `u` as the
blurrier tile's: it blurs the tile, reads the blurred tile's ellipse with the
reading the two lengths were read with, and corrects `σ` until the length is
within `BLUR_MATCH_TOLERANCE` (5%) of the one aimed for. Both directions of a
tile are searched together, each blurred tile read once for both. The blur is
set by the same measurement on both sides, so whatever the reading makes of a
tile, its tolerance, its whole-pixel shifts and the texture's own spectrum,
cancels instead of having to be modelled.

- **The first width tried** is `estimated_blur_sigma`, a difference of squares:
  `σ² = (l_b² − s²) / (k · s^p)`, with `s` the sharper length (at least
  `MIN_SHARPER_LENGTH`, 0.05 grid px), `l_b` the length aimed for,
  `k = BLUR_GROWTH_SCALE = 1.243` and `p = BLUR_GROWTH_POWER = 1.247`. A
  Gaussian blur adds its variance to a texture's correlation length, so the
  square of a tile's ellipse length grows about linearly with `σ²`; the rate,
  `d(l²)/d(σ²)`, rises with the tile's own length, because a longer ellipse
  comes from a fainter or smoother texture, whose tolerance is wider. The two
  constants were fitted in logs on 3,697 real tile directions of ten datasets,
  each tile blurred along the direction a pair asked for at eleven widths from
  0 to 4 and read again. A single tile's rate is spread about the fit by a
  factor of about 2 either way (the 25th and 75th percentiles of the rate at
  `σ` 1 are 0.48 and 1.32 against a median of 0.86), which is why the estimate
  is only the first width tried.
- **The correction** is a secant step on `l²` against `σ²`, which grows about
  linearly: through the narrowest width that overshot and the widest that fell
  short, or, while none has overshot, through the last two that fell short. It
  is kept strictly inside that bracket, and a length that did not grow doubles
  the width. At most `BLUR_MATCH_MAX_PROBES` (2) blurred tiles are read. Where
  the second is still more than the tolerance off, the tile is blurred once
  more by the next step, unread, if that step lies between a width that fell
  short and one that overshot along every direction still off; otherwise by
  the width read closest to its target.
- **The length aimed for is at most `MAX_MATCHED_LENGTH` (2 grid px).** A
  blurrier tile longer than that is matched as 2 long. Past about 2 grid px the
  blurred tile holds so little detail across a 24-sample tile that it is hard
  to tell from another surface's: matched to the full length, members were
  blurred towards smooth tiles of other points by `σ` 2 to 3, and those
  lookalikes' scores rose more than the members' (below). The reading itself
  stays informative up to about 2.5: its axes start to be flagged as lower
  bounds (5% of readings) from 2.5 and most are from 2.75.
- **The widest blur is `MAX_BLUR_SIGMA` (3 grid px).** A direction blurred by
  3 and still short is left there.

**Why it is measured rather than mapped.** The kernel first took the width from
a mapping fitted on synthetic blurs, `σ = 0.9975 · s^0.26 · (r² − 1)^0.17` with
`r` the ratio of the two lengths, fitted on the ground truths' tiles blurred by
known Gaussians (`σ` 0.5 to 2.5) with `σ` regressed on the ratio. Its spread
about the fit was a factor of 1.66, and regressing the width on so noisy a
predictor flattened the fit towards the middle of the widths it was fitted on:
it read a blur of 0.5 back as 0.96. Real pairs mostly differ by little, so it
blurred them too much. Measured on 16,204 blurred directions of pool pairs of
ten datasets (40 tracks each, ratio 1.25), with the blurred tile's ellipse read
again and compared with its partner's along the direction blurred:

| Width | `σ`, median | Blurred length / partner's, p10 / median / p90 | Within 10% | Over by more than 10% | Gradient energy along `u`, blurred / partner, median | Power at 0.12–0.32 cycles/px along `u`, median |
|---|---|---|---|---|---|---|
| Fitted mapping (before) | 0.90 | 0.88 / 1.20 / 1.71 | 25% | 63% | 0.89 | 0.77 |
| Measured (this design) | 0.62 | 0.91 / 0.99 / 1.04 | 89% | 2% | 1.06 | 1.00 |

The two image-domain columns do not read the ellipse: the mean square of the
derivative along `u`, and the share of each tile's power in a band of spatial
frequencies along `u`, each over the samples with data, the blurred tile's over
its partner's. Under 1, the blurred tile is the blurrier. With the mapping the
blurred tile was the blurrier by both; measured, it is level with its partner
in the band and slightly sharper by the derivative, which the partner's noise,
blurred away in the blurred tile, raises. By how much the two lengths differed,
the mapping gave `σ` 0.57 where the measured width is 0.22 (ratio under 1.1),
0.84 against 0.58 (1.25 to 1.5) and 1.25 against 1.70 (3 or more): too much
for the small differences most pairs have and too little for the largest.

Three other designs were measured the same way and set aside: the mapping
scaled down by 0.6 (median 0.89, and under-blurring large differences:
a member blurred by `σ` 3 fell below the wrong-view threshold 25% of the time);
the difference of squares alone, with no reading (median 1.11, p90 1.61, since
single tiles stray from the fitted rate); and matching each pair to its full
length with no cap (as accurate, but lookalike tiles of other points then rose
in score and the separation of wrong views fell, below).

### Skipping pairs that barely differ

`PairMatching::BlurMatchedAboveRatio` leaves a direction alone where the two
lengths along it differ by less than a factor, and a pair left alone along both
directions is correlated plain at no extra cost.
`DEFAULT_MIN_ELLIPSE_RATIO` is `1.25`. On real tracks it skips about half the
pairs. With the measured width the shares blurred are:

| Ratio | Bench pairs blurred (661 tracks) | Member-coherence pairs blurred (425,283 pairs, 8,484 points) |
|---|---|---|
| 1 (every difference) | 99% | 99% |
| 1.25 | 52% | 50% |

Under 100% at a ratio of 1, since two lengths both past `MAX_MATCHED_LENGTH`
are matched as equal. How much the skip changes what a consumer reads was
measured with the fitted mapping the kernel used before, which blurred small
differences by more than the measured width does, so it bounds the change now:
at 1.1, 1.25 and 1.5 the bench blurred 82%, 55% and 29% of 36,563 pairs, and a
view's blur-matched median moved by 0.0007, 0.004 and 0.009 (median; p90
0.005, 0.013, 0.023), its cell deficit by 0.003, 0.007 and 0.011 (p90 0.018,
0.033, 0.048).

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
| Plain pair readings (whole tile and the nine cells) | 3.6 |
| The two passes, isotropic `σ` 1 / 1-D `σ` 1 at 30° / `σ` 1.5 × 0.8 at 30° / isotropic `σ` 2 | 5.1 / 5.4 / 10.3 / 9.3 |
| Direct 2-D convolution, the same isotropic `σ` 1 / 1.5 × 0.8 at 30° / `σ` 2 | 46 / 109 / 150 |
| `pair_directions` from two ellipses | 0.1 |
| The whole self-similarity reading of a tile | 8.8 |
| `match_blur` of one tile, two readings | 33 |
| A blur-matched pair, anisotropic (one tile blurred, readings) | 42 |

The two passes run 9 to 16 times faster than the direct convolution. The
width's search is most of a blurred pair's cost: each blurred tile it reads
costs a blur and a whole self-similarity reading, about 14 µs, where every
sample carries data. On the bench's pairs it reads 1.9 blurred tiles per pair
blurred, and the pair costs about three times what it did with the fitted
mapping, which read none. A tile with samples off the photograph takes the
reading's slower route, which visits every sample, at about 50 µs a reading.
Member coherence reads its ellipses on the largest square inside its common
support, where every sample carries data, and its blurred renders the same way.

**No AVX2 form.** Every pass is a sum of whole shifted rows, `dst[x] += w ·
src[x + s]`, which the compiler vectorizes. The same code compiled for AVX2
and FMA, chosen at run time, ran from 13% faster to 13% slower than the
default build depending on the kernel's shape (rows of 24 to 32 samples leave
little for wider registers), so the kernel has none.

**The isotropic ladder.** `BlurMatchKernel::IsotropicLadder` blurs the
sharper tile, by semi-major axis, isotropically, by one of eight widths
`0.25 · √2ⁿ` (`LADDER_SIGMAS`): the one whose blurred tile's semi-major axis
comes closest, in the logarithm, to the other tile's (at most
`MAX_MATCHED_LENGTH`), or none where the tile as it is comes closer. It fills a
view's levels from the narrowest, blurring and reading each, until one reaches
the length aimed for, and keeps them, so each view is blurred and read at most
once per level however many pairs it is in: on the bench it read 1.0 blurred
tiles per pair blurred. Its reference-view phase costs about 13% less than the
anisotropic kernel's on tracks of up to 6 views and 43% less on tracks of 21
views or more (6.1 against 10.7 ms, ratio 1.25 on both). It picks the same
reference view as the anisotropic kernel on all but two of the 79 review cases. It
is isotropic, which the anisotropic kernel was chosen over for oblique and
directional views, so it is an option rather than the default.

## Accuracy

**The width against a known blur.** Where the blurry tile is the sharp tile
blurred by a known Gaussian along the grid's axes or isotropically (where the
two passes are an exact Gaussian), at widths 0.4 to 1.5 and elongations up to
1.2 × 0.5, on a texture whose ellipse is about 0.45 grid px, the width found
along each axis is within 7% of the one planted, or within 0.15 grid px for a
blur of 0.4: a narrow blur lengthens the ellipse little, so a length within the
tolerance pins its width less closely. A 1-D blur at an angle between the
grid's axes also blurs across itself, by the slanted pass's interpolation, and
the search finds that blur too.

**The two passes against the exact blur.** On a tile of four sinusoids, whose
blur by a Gaussian is known exactly (each amplitude scales by `exp(−½ kᵀ Σ
k)`), the passes are at most 3.6 grey levels off the exact answer on sinusoids
spanning ±115, over 108 covariances (semi-axes from 0.5 to 2, with 1-D blurs,
at nine angles). At the widths most blurred pairs get, semi-axes of 0.5 to 0.7
with 1-D blurs, at every 5°, they are at most 2.7 off and 0.6 on average. The
direct 2-D convolution is 0.34 off where its minor axis is 0.6 or more; it
samples the kernel as it is, which falls short of its width below that, and it
has no 1-D form. The passes' error is the linear interpolation's, largest for
1-D blurs at angles between the grid's axes and its diagonals. On a textured
tile with samples missing, the two blurs agree to a ZNCC above 0.998.

**The Python port.** The illustration's port of the choice of blur (directions,
first width and search, reading each blurred tile through the binding) draws
blurred tiles whose ZNCC is within 0.011 of the kernel's on every pair drawn.
It blurs by successive 1-D passes rather than the kernel's two-pass split, so
its widths can differ from the kernel's by a step of the search.

## What it changes

**A member made blurrier.** From 987 pool tracks of ten datasets (at least five
views each), one member's tile was blurred by `σ` 1 or 2 grid px (its
semi-major axis went from 0.98 to 1.72 and 2.86 px, median), its ellipse read
again, and its median ZNCC with the other members compared with the threshold
that catches 90% of planted wrong views (lookalike tiles of other points,
neighbours, and members shifted by 2 to 4 px), the threshold read with the
prototype's readings:

| Reading | Change in its median, `σ` 1 / 2 | Below the threshold, `σ` 1 / 2 | The same with the fitted mapping |
|---|---|---|---|
| Plain | +0.003 / −0.033 | 8.1% / 19.5% | |
| Blur-matched, anisotropic | +0.019 / +0.024 | 4.2% / 2.5% | 4.0% / 3.0% |
| Blur-matched, ratio 1.25 | +0.020 / +0.026 | 4.4% / 2.5% | 4.1% / 3.0% |
| Blur-matched, isotropic ladder | +0.020 / +0.020 | 5.0% / 4.4% | 4.7% / 2.7% |

At `σ` 3, where the member's semi-major axis reaches the reading's 3 px cap,
5.4% fall below it (9.2% with the fitted mapping).

**Wrong views.** The same kind of test with each reading's own threshold, on
990 pool tracks (100 a dataset, seeded), with the kernel's readings throughout
and a bootstrap over tracks for the difference:

| Width | Members over wrong views (AUC) | Members below the threshold that catches 90% of wrong views | Blurred member below it, `σ` 1 / 2 / 3 |
|---|---|---|---|
| Plain | 0.9628 | 9.4% | 9.7% / 21.9% / 46.5% |
| Fitted mapping, ratio 1.25 | 0.9658 | 8.4% | 5.8% / 4.9% / 11.2% |
| Measured, no cap on the length matched | 0.9642 | 9.6% | 5.5% / 3.6% / 4.1% |
| Measured, matched up to 2 grid px (this design), ratio 1.25 | 0.9661 | 8.8% | 5.3% / 3.5% / 6.1% |

Against the fitted mapping the design's AUC differs by +0.0003 (95% interval
−0.0003 to +0.0008) and its share of members below the threshold by +0.35
points (−0.20 to +0.86): no measurable change. Matched to the full length,
the AUC fell by 0.0015 (−0.0021 to −0.0009), almost all of it on lookalike
tiles of other points, which are smooth and long in ellipse: the member was
blurred towards them by `σ` 2 to 3, where the fitted mapping had stopped at
about 1.5, and their score rose by 0.01 to 0.04 on such pairs. Capping the length
matched at 2.5 took back half of that, at 2 all of it, and costs the member
blurred by `σ` 3 two points.

**On the consumers**, measured on the review cases and samples of the ten
datasets of the reference-view work:

| Consumer | Option | Added cost | Effect | Default |
|---|---|---|---|---|
| Reference view: agreement test and cell check ([reference-view.md](reference-view.md)) | blur-matched, ratio 1.25 | +0.59 ms per track (median; p90 4.4 ms), 8.3% of an evaluation (+0.29 ms, 3.6%, with the fitted mapping) | 30 of 77 hand picks exactly, against 28 plain (tune half 19, held-out 11 against 18 and 10), as with the fitted mapping; the pick differs from the fitted mapping's on 14 of 661 tracks | on |
| Member coherence's decision ([member-coherence-validation.md](member-coherence-validation.md)) | blur-matched, ratio 1.25 | 2.7 × the plain run on one thread (1.6 × with the fitted mapping) | a planted member blurred by `σ` 2 is evicted 1.2% of the time against 4.9% plain (2.0% with the fitted mapping); verdicts change on 2.1% of real points, in both directions | off |

## Implementation notes

**The ellipse is the measurement of the tile being blurred.** A consumer passes
the ellipse of the very render it correlates: the bench the `R×R` tile's whole
reading over its samples on the photograph, member coherence a reading of its
own render over the largest square inside the common support. An ellipse read
on another render of the view (another resolution, sampler or support)
describes another tile. The blurred tile must then be read the same way, since
the search compares the two readings: `blur_matched_pairs` reads it with
`read_tile_ellipse`, member coherence with its own square, and an ellipse read
any other way would set the blur against a different scale.

**The search is deterministic.** Each step is a fixed sequence of `f64`
operations on readings that are themselves the same on every run, so a pair
reads the same in a track as alone and on every platform; a test checks the
former bit for bit.

**The first pass covers the band the second reads.** The tile sits in a buffer
padded with zeros by both passes' reach, and the first pass writes the tile and
the second pass's reach round it, so the two passes compose as one 2-D
convolution of the tile extended by zeros rather than of a tile cut off after
the first pass. A band is computed in whole 8-sample groups; the columns past
the tile it adds are written and never read back. The padded buffer is all
zeros between calls: each call writes it only on the tile's rows, from the
tile's first column, and clears what it wrote before it returns. Two
covariances can share a buffer size and differ in their padding, so a buffer
cleared only when its size changed would hand the previous tile to the next
call; a test reuses one scratch over many shapes and compares each blur with a
fresh scratch's, bit for bit.

**Each pass's taps have the variance asked for.** A Gaussian sampled at whole
pixels has less variance than its width says once the width is under about 1:
0.215 for 0.25 at `σ` 0.5, and under a tenth of 0.09 at `σ` 0.3. The slanted
pass is often that narrow along its rows, so its taps would blur less than
asked, by most at 1-D blurs between the axes and the diagonals. Under `σ` 1 the
taps are those of a sampled Gaussian whose width is found by bisection so their
variance is `σ²`, which brought the worst error at widths 0.5 to 0.7 from 7.8
grey levels to 2.7. Integrating the Gaussian over each pixel instead overshoots
the variance by about 1/12 and was worse at every width.

**A pair left plain reads exactly the plain value.** `pair_zncc_readings`
gathers the cells' sums in the same order and by the same formula as
`pair_zncc_grid` ([reference-view.md](reference-view.md)), so a pair the ratio leaves alone reads
the plain cell grid bit for bit, and member coherence copies its plain value
for such a pair.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `DEFAULT_MIN_ELLIPSE_RATIO` | `1.25` | The factor by which two lengths along a direction must differ for `BlurMatchedAboveRatio` to blur along it; it skips about half the pairs on real tracks |
| `MAX_MATCHED_LENGTH` | `2` grid px | The longest ellipse length blur matching aims for; a longer blurrier length is matched as this one |
| `BLUR_MATCH_TOLERANCE` | `0.05` | How close, as a fraction of the length aimed for, the blurred tile's length must come for the search to stop |
| `BLUR_MATCH_MAX_PROBES` | `2` | The most blurred tiles the search reads for one tile of a pair |
| `BLUR_GROWTH_SCALE`, `BLUR_GROWTH_POWER` | `1.243`, `1.247` | `k` and `p` of the first width tried, `σ² = (l² − s²) / (k · s^p)`, fitted on real tiles |
| `MIN_SHARPER_LENGTH` | `0.05` grid px | The shortest sharper length the skip ratio and the first width read |
| `MAX_BLUR_SIGMA` | `3` grid px | The widest blur along a direction |
| `LADDER_SIGMAS` | `0.25 · √2ⁿ`, `n = 0 .. 7` | The isotropic ladder's widths |
| `MIN_WINDOWED_SAMPLES` | `8` | The fewest samples with data in both tiles and weight in the window a whole-tile reading is taken over |

Each consumer's option and its default are in that consumer's spec. The
constants are defined in
[blur_matched.rs](../../../crates/sfmtool-core/src/patch/blur_matched.rs) and
[tiles.rs](../../../crates/sfmtool-core/src/patch/blur_matched/tiles.rs).

## Python bindings

`sfmtool._sfmtool.patches.blur_matched_zncc_matrix(tiles, *, valid=None,
ellipses=None, matching="blur_matched", min_ellipse_ratio=1.25,
kernel="anisotropic", window="gaussian_disk", window_sigma=0.6)` reads every
pair of a `(k, R, R, C)` uint8 stack of tiles. One channel is grey, two grey
and alpha, three RGB and four RGB and alpha; alpha is not correlated, and 0
there marks a sample without data. `valid` is an optional `(k, R, R)` bool
stack, as `OrientedPatch.render_view_tile` returns it, `False` marking a sample
without data. `ellipses` is `(k, 2, 2)` float64, NaN for a tile without one, or
`None` to read each tile's whole self-similarity here, over the samples with
data, which is how the blurred tiles are read; ellipses passed in should be
read that way too, or the blur is set against a different reading. It returns a dict: `zncc` `(k, k)`, `zncc_grid` `(k, k, 3, 3)`, `blurred`
`(k, k)` bool, `pairs`, `pairs_blurred` and `ellipse_matrix` `(k, 2, 2)`. It
raises `ValueError` for a stack that is not square tiles of 3 or more on a side
with 1 to 4 channels, a `valid` or `ellipses` of the wrong shape, an unknown
name, or a ratio under 1 or not finite.

The bench's readings are those of the tiles `render_view_tile` renders at each
view's keypoint, at the evaluation's resolution and with its sampler, read with
their `valid` flags, the ellipses left to the function, a ratio of 1.25 and the
default kernel and window:

```python
from sfmtool._sfmtool.patches import blur_matched_zncc_matrix

rendered = [
    patch.render_view_tile(camera, pose, image, keypoint=kp)
    for camera, pose, image, kp in views
]
out = blur_matched_zncc_matrix(
    np.stack([t["samples"] for t in rendered]),
    valid=np.stack([t["valid"] for t in rendered]),
    matching="blur_matched_above_ratio",
)
print(out["pairs_blurred"], "of", out["pairs"], "pairs blurred")
```

`PatchCloud.validate_member_coherence(..., matching=, min_ellipse_ratio=)` and
the bench's readings carry the consumers' forms
([member-coherence-validation.md](member-coherence-validation.md),
[editable-track.md](../bench/editable-track.md) § "Python bindings").

## Testing

[blur_matched/tests.rs](../../../crates/sfmtool-core/src/patch/blur_matched/tests.rs)
checks the first width (zero without a difference, growing with it, its
formula, the floor and the cap); that equal ellipses blur nothing; that each tile is blurred
only along the directions it is sharper in, on axis-aligned and rotated pairs;
the skip ratio, per direction; that matching a tile against a copy of it
blurred by a known Gaussian recovers that Gaussian along each axis and matches
the copy's ellipse, and that a search with no reading falls back to the first
width; the two passes against the exact blur of a tile
of sinusoids and against the direct 2-D convolution, and the two against each
other round missing samples; that one scratch reused over many blurs of
different shapes gives what a fresh one gives, bit for bit, and that a pair
reads the same in a track as alone; that a sample without data neither gives nor takes
a value and a flat tile stays flat at the edge and round holes; that a 1-D blur
leaves a texture along it alone; that a tile and a blurred copy of it read a
ZNCC near 1 blur-matched and well above plain; that equally sharp views read
the plain value; the ladder against the anisotropic kernel; and `rows`. An
ignored test, `timing`, prints the cost table above.
[member_coherence/tests.rs](../../../crates/sfmtool-core/src/patch/member_coherence/tests.rs)
checks that a member blurred in its photograph is lifted blur-matched.
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
  Whether the bars should switch at all is an open question of Part 6 of
  [../../drafts/sharper-patch-bitmap.md](../../drafts/sharper-patch-bitmap.md):
  blur-matched, they would stop reacting to views that are out of focus.
- **The fuse's IRLS residuals.** Each view's residual against the weighted mean
  is recomputed every iteration, and the mean, the blurrier of every pair,
  changes with the weights, so blur-matching the residuals needs the mean's
  ellipse every iteration and changes the stored bitmap, not only a score. How
  the bitmap is computed is Part 5 of the same draft.
- **Deconvolution.** A blurry tile is not sharpened; the sharper one is
  blurred.

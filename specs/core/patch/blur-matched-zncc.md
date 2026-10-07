# Blur-Matched ZNCC

Two photographs of the same piece of surface rarely show it equally sharply:
one is farther away, out of focus or moved during the exposure. Normalized
correlation (ZNCC) between the two views' tiles scores the sharper one lower
for detail the other lacks, so a sharp, well-aligned view reads as if it
disagreed with its track, and a blurry view reads worse beside a sharp one than
it does beside other blurry views. **Blur-matched ZNCC** takes part of that
penalty away: where one tile is sharper than the other along every direction,
that tile is blurred by a round Gaussian until it is no sharper than the other
is along its sharpest direction, and the two are then correlated. It is a
score for judging views. It is never used to place a view: blurring a template
before aligning to it moves the correlation peak no closer to the truth (see
[Non-goals](#non-goals)).

Each tile's sharpness is read from its ZNCC self-similarity ellipse
([zncc-self-similarity-radius.md](zncc-self-similarity-radius.md)): the
ellipse of the shifts at which the tile still matches itself, short along the
directions in which it holds fine detail. A tile is the **sharper** of a pair
when its semi-major axis is shorter than the other tile's semi-minor axis. Only
that tile is blurred, isotropically, and the blur's width is set so that its
semi-major axis reaches the other tile's semi-minor axis (at most 2 grid px).
No direction of the sharper tile is then blurred past what the blurrier tile
shows along its sharpest direction. A pair in which neither tile is shorter in
that sense is correlated plain. That covers a view blurry along one direction
only (an oblique or fisheye view, motion blur along a line) and two tiles of a
texture with a grain, whose ellipses are long along the grain whatever the
sharpness: a long axis is not read as blur. The width comes from the sharper
tile's **blur assessment**, how its own semi-major axis grows when it is
blurred: each tile is blurred twice, by 0.4 and by 1 grid px, and read again,
once, and every pair the tile is in reads its width off those readings without
reading again. The blur is done to one tile; pairing the tiles, and choosing
which to blur to what length, is the consumers' part.

Two consumers read it. The bench's reference-view rule reads blur-matched
agreement by default ([reference-view.md](reference-view.md) § "Blur-matched
agreement"); member-coherence validation can read it, and does not by default
([member-coherence-validation.md](member-coherence-validation.md) § "Blur
matching"). The measurements behind both choices are below.

## Rust API

Blur matching is done to one tile; which tile of a pair is blurred, and to what
length, is the caller's choice. The per-tile operations live in
[blur_matched.rs](../../../crates/sfmtool-core/src/patch/blur_matched.rs), with
the assessment and the blur to a length in
[assess.rs](../../../crates/sfmtool-core/src/patch/blur_matched/assess.rs), the
blur in [blur.rs](../../../crates/sfmtool-core/src/patch/blur_matched/blur.rs)
and the tile and its reading in
[tiles.rs](../../../crates/sfmtool-core/src/patch/blur_matched/tiles.rs). The
pairing rule the two consumers share is in
[pair_sharpness.rs](../../../crates/sfmtool-core/src/patch/pair_sharpness.rs),
and the reference-view rule's pairwise readings in
[pair_readings.rs](../../../crates/sfmtool-core/src/patch/reference_view/pair_readings.rs);
member coherence reads its pairs in
[matrix.rs](../../../crates/sfmtool-core/src/patch/member_coherence/matrix.rs).

```rust
// patch::blur_matched: one tile.
pub struct TilePlanes { pub values: Vec<f32>, pub data: Vec<bool>, pub side: usize, pub channels: usize }
impl TilePlanes {
    // An interleaved u8 tile: 1 or 2 channels grey (and alpha), 3 or 4 RGB (and alpha).
    pub fn from_interleaved(samples: &[u8], side: usize, stride: usize, data: &[bool]) -> Self;
    pub fn blurred(&self, sigma: f64, scratch: &mut BlurScratch) -> Self;
}
// The whole-tile self-similarity reading over the samples with data.
pub fn read_tile_ellipse(values: &[f32], channels: usize, side: usize, data: &[bool])
    -> Option<[[f64; 2]; 2]>;
pub fn semi_axes(e: &[[f64; 2]; 2]) -> [f64; 2]; // [major, minor], grid px

// How sharp one tile is, and how that changes under round blur. Read once per tile.
pub const GROWTH_PROBE_SIGMAS: [f64; 2]; // [0.4, 1.0]
pub struct BlurAssessment {
    pub semi_axes: [f64; 2],       // [major, minor] of the tile as read, grid px
    pub growth: [[f64; 2]; 2],     // [major, minor] after each of GROWTH_PROBE_SIGMAS
}
impl BlurAssessment {
    // The round blur width that brings the semi-major axis to `length`; 0 when already there.
    pub fn sigma_to_reach(&self, length: f64) -> Option<f64>;
}
pub fn assess_blur(tile: &TilePlanes, ellipse: &[[f64; 2]; 2],
    read: impl FnMut(&[f32]) -> Option<[[f64; 2]; 2]>,
    scratch: &mut BlurScratch) -> Option<BlurAssessment>;
// Blur until the semi-major axis reaches `length` (no blur where it already has): the tile and the width.
pub fn blur_to_length(tile: &TilePlanes, assessment: &BlurAssessment, length: f64,
    scratch: &mut BlurScratch) -> Option<(TilePlanes, f64)>;
pub fn blur_to_length_into(tile: &TilePlanes, assessment: &BlurAssessment, length: f64,
    out: &mut TilePlanes, scratch: &mut BlurScratch) -> Option<f64>;

// The isotropic blur, by normalized convolution over the samples with data.
pub fn blur_tile(values: &[f32], channels: usize, side: usize, data: &[bool],
    sigma: f64, out: &mut [f32], scratch: &mut BlurScratch);

// patch::pair_sharpness: which tile of a pair is blurred, and to what length.
pub enum PairMatching { Plain, BlurMatched, BlurMatchedAboveRatio(f64) }
impl PairMatching {
    pub fn min_ratio(self) -> Option<f64>;   // None for Plain, 1 for BlurMatched
    pub fn is_blur_matched(self) -> bool;
    pub fn name(self) -> &'static str;       // "plain", "blur_matched", "blur_matched_above_ratio"
    pub fn from_name(name: &str, ratio: f64) -> Option<Self>;
}
pub struct PairBlur { pub sharper: usize, pub major: f64, pub target: f64 }
pub fn pair_blur(a: &[[f64; 2]; 2], b: &[[f64; 2]; 2], min_ratio: f64) -> Option<PairBlur>;
// A track's views: the assessments of the views some pair blurs, each read once.
pub struct TrackBlurs { /* ... */ }
pub struct PairTarget<'a> { pub view: usize, pub assessment: &'a BlurAssessment, pub target: f64 }
impl TrackBlurs {
    pub fn assess(ellipses: &[Option<[[f64; 2]; 2]>], matching: PairMatching,
        assess: impl FnMut(usize, &[[f64; 2]; 2]) -> Option<BlurAssessment>) -> Self;
    pub fn pair(&self, a: usize, b: usize) -> Option<PairTarget<'_>>;
    pub fn assessment(&self, v: usize) -> Option<&BlurAssessment>;
    pub fn assessed(&self) -> usize;
}

// patch::reference_view: the reference-view rule's pairwise readings.
pub struct PairReadings { pub whole: f64, pub grid: [[f64; 3]; 3] }
pub fn pair_zncc_readings(a: &TilePlanes, b: &TilePlanes, window: &[f64]) -> PairReadings;
pub struct BlurMatchedPairs {
    pub k: usize, pub whole: Vec<f64>, pub grid: Vec<[[f64; 3]; 3]>,
    pub blurred: Vec<bool>, pub sigma: Vec<f64>, // sigma[a*k + b]: the width a was blurred by against b
    pub pairs: usize, pub pairs_blurred: usize, pub ellipse_reads: usize,
}
impl BlurMatchedPairs {
    pub fn row_middle(&self, v: usize) -> f64; // median of view v's whole-tile readings
}
pub fn blur_matched_pairs(tiles: &[&TilePlanes], ellipses: &[Option<[[f64; 2]; 2]>],
    matching: PairMatching, window: PatchWindow, progress: &Progress<'_>) -> BlurMatchedPairs;
pub struct BlurMatchedAgreement { pub pair_zncc: Vec<f64>, pub cells: CellAgreement, pub pairs: BlurMatchedPairs }
pub fn blur_matched_agreement(tiles: &[&ViewTile], ellipses: &[Option<[[f64; 2]; 2]>],
    matching: PairMatching, window: PatchWindow, progress: &Progress<'_>) -> BlurMatchedAgreement;
```

**Why it is shaped this way.** A tile's **blur assessment** depends on that
tile alone: its own self-similarity semi-axes, and the semi-axes of the tile
blurred by each probe width. So it is read once per tile, whatever the tile is
later compared with, and every comparison reads its width off the assessment
without blurring the tile again (`sigma_to_reach`, a function of a few numbers
that can be tested on constructed readings). The probes already read the whole
ellipse, so the assessment keeps both axes of each reading; the width uses the
semi-major ones. Being a few numbers per tile that depend on the tile alone,
the assessment is what a reconstruction could store per observation beside its
self-similarity ellipse; the `.sfmr` format does not store it.

`assess_blur` takes the ellipse already read and a reading as a closure,
because the width is set by comparing a blurred tile's reading with another
tile's, and they must be the same reading: the bench and the bindings read the
whole tile over its samples with data (`read_tile_ellipse`), member coherence
reads the largest square inside its common support. `blur_to_length` is the one
operation the consumers apply to a tile; `blur_to_length_into` is the same into
a reused buffer, for a caller that blurs pair after pair.

Pairing is the caller's business. `pair_blur` is a pure function of two
ellipses, so which tile is blurred and to what length can be tested on
constructed numbers. `TrackBlurs` holds a track's pairing for both consumers:
it assesses each view that `pair_blur` names the sharper of some pair, once and
in view order, and names, for a pair, the view to blur, its assessment and the
length; the consumer blurs that view's tile (`blur_to_length_into`) and
correlates the pair with its own ZNCC code. The views to assess depend on the
ellipses alone, so a view that no pair can blur is never blurred or read again.
`PairMatching` is one enum for every consumer, so a consumer's option reads the
same on the wire, in Python and in Rust. `blur_matched_pairs` reads both the
whole tile and the ZNCC grid's cells from each pair it blurs, because the
reference rule's agreement test and its cell check both read the same pairs.
`sigma` reports each width, so a caller that draws the blurred tiles (the
illustration, a test) blurs by the kernel's own width rather than working it
out again. The blur is a function of one tile and a width, so it can be checked
against a direct 2-D convolution and an exact answer.

```rust
use sfmtool_core::patch::blur_matched::{
    assess_blur, blur_to_length, read_tile_ellipse, semi_axes, BlurScratch, TilePlanes,
};

/// Blur `sharp` until it is no sharper than `blurry` along its sharpest direction.
fn match_to(sharp: &TilePlanes, blurry: &TilePlanes) -> Option<(TilePlanes, f64)> {
    let mut scratch = BlurScratch::default();
    let read = |values: &[f32]| read_tile_ellipse(values, sharp.channels, sharp.side, &sharp.data);
    let ellipse = read(&sharp.values)?;
    let assessment = assess_blur(sharp, &ellipse, read, &mut scratch)?;
    let other = read_tile_ellipse(&blurry.values, blurry.channels, blurry.side, &blurry.data)?;
    blur_to_length(sharp, &assessment, semi_axes(&other)[1], &mut scratch)
}
```

Over a track, the consumers pair the views and blur through `TrackBlurs`:

```rust
use sfmtool_core::patch::blur_matched::{
    assess_blur, blur_to_length_into, read_tile_ellipse, BlurScratch, TilePlanes,
};
use sfmtool_core::patch::pair_sharpness::{PairMatching, TrackBlurs};

fn blurred_pairs(tiles: &[TilePlanes], ellipses: &[Option<[[f64; 2]; 2]>]) {
    let mut scratch = BlurScratch::default();
    let blurs = TrackBlurs::assess(ellipses, PairMatching::BlurMatchedAboveRatio(1.25), |v, e| {
        let t = &tiles[v];
        let read = |values: &[f32]| read_tile_ellipse(values, t.channels, t.side, &t.data);
        assess_blur(t, e, read, &mut scratch)
    });
    let mut blurred = TilePlanes::default();
    for a in 0..tiles.len() {
        for b in (a + 1)..tiles.len() {
            let Some(p) = blurs.pair(a, b) else { continue };
            if let Some(sigma) =
                blur_to_length_into(&tiles[p.view], p.assessment, p.target, &mut blurred, &mut scratch)
            {
                println!("pair {a}-{b}: view {} blurred by {sigma:.2} grid px", p.view);
            }
        }
    }
}
```

## Theory

### Which tile is blurred, and to what length

Let `E_a` and `E_b` be the two tiles' self-similarity ellipse matrices, in grid
px² (each ellipse is `dᵀ E⁻¹ d = 1`, its eigenvalues the squared semi-axes),
and `M_a ≥ m_a`, `M_b ≥ m_b` their semi-major and semi-minor axes. Tile `a` is
the sharper when `M_a < m_b`: along every direction its ellipse is shorter than
`b`'s is along any direction. Tile `b` is the sharper when `M_b < m_a`. The two
cannot both hold, since `M_a < m_b ≤ M_b < m_a ≤ M_a` is a contradiction, so at
most one tile of a pair is blurred. The target is the blurrier tile's
semi-minor axis, at most `MAX_MATCHED_LENGTH` (2 grid px). The sharper tile is
blurred isotropically by the width at which its semi-major axis reaches the
target (below); its semi-minor axis, shorter, ends no longer than that. The
blurred tile is then no blurrier than its partner along any direction, and
along the partner's blurry directions it is still the sharper.

The rule leaves most pairs plain on real tracks: on 22,035 pool pairs it blurs
12% of them with every difference blurred and 5% at the default ratio, where a
blur along each direction in which the two ellipses differ blurred 90% and 50%.
What it leaves plain, and why that is the intent:

- **A view blurry along one direction only.** An oblique view is blurry along
  the direction it is foreshortened and sharp across it, and its semi-minor
  axis is often no longer than a frontal view's semi-major axis. Blurring the
  frontal view along the oblique view's long axis took 21% of the pairs blurred
  per direction (2,312 pool pairs whose blurrier view is foreshortened by 1.5 or
  more, or seen at 55° or more), and raised their ZNCC by 0.020 on average.
  Those pairs are read plain.
- **Grain and stripes.** A texture with a grain (bark, a railing, a striped
  awning) reads a long ellipse along the grain in a sharp view. Two such views,
  or a sharp grained view and another view, differ in ellipse along the grain
  without either being blurrier, and a blur along that difference smooths away
  detail the partner does show. On MossyRailing point 817, a striped track of
  six views, a blur per direction blurred 11 of its 15 pairs; this rule blurs
  the 3 pairs of its one round, sharp view (semi-axes 0.32 × 0.19) against the
  views whose semi-minor axis is past it, and leaves the 8 pairs among the
  striped views plain.

On pool pairs that keeps 15% of the ZNCC gain a blur per direction gave over
plain ZNCC, and 61% on the pairs both blur. The planted wrong views below show
what that trade does to telling members from other surfaces.

### The width, from each tile's blur assessment

A Gaussian blur adds its variance to a texture's correlation length, so the
square of a tile's semi-major axis grows with `σ²`, about linearly, at a rate
that differs from tile to tile. Each view's tile is blurred by each of
`GROWTH_PROBE_SIGMAS` (0.4 and 1 grid px) and each blurred tile's ellipse read
with the reading the view's own ellipse came from (`assess_blur`). The square
of the semi-major axis is then known at `σ²` = 0, 0.16 and 1, and taken to be
piecewise linear between them and along its last piece past 1. The width is
the `σ` at which that line reaches the target
(`BlurAssessment::sigma_to_reach`), at most `MAX_BLUR_SIGMA`. Each view some
pair blurs is assessed once and kept for every pair (`TrackBlurs`), so a track
pays two blurred readings for each view it blurs, not for each pair. The probes are isotropic, as the
blur is, so the line is read on the same kind of blur the pair gets.

- **Two probes, because the growth bends.** A tile whose ellipse is well under
  a grid px long hardly lengthens until the blur reaches the scale of a pixel,
  and then lengthens faster. A straight line from one probe at 0.6 blurred 6.1%
  of the tiles more than 10% past the target; the wider probe follows the bend.
- **Past the widest probe** the line is extrapolated along its last piece. On
  the pool pairs 6% of the widths are past 1 grid px and they land within 3% of
  the target (p90); holding them to the width of a rate fitted on real tiles,
  as an extra limit, changed none of them.
- **Where a view's assessment cannot be read**, or its readings do not grow, the
  pair is read plain. That happened on none of the pool pairs.
- **The target is at most `MAX_MATCHED_LENGTH` (2 grid px).** Past about 2 grid
  px the blurred tile holds so little detail across a 24-sample tile that it is
  hard to tell from another surface's: matched to the full length, members were
  blurred towards smooth tiles of other points, and those lookalikes' scores
  rose more than the members'. The reading itself stays informative up to about
  2.5: its axes start to be flagged as lower bounds (5% of readings) from 2.5
  and most are from 2.75.
- **The widest blur is `MAX_BLUR_SIGMA` (3 grid px).**

### The widths measured, and the alternatives

Each rule was run on the 1,088 pool pairs of ten datasets (40 tracks each) the
default ratio blurs, each blurred tile's ellipse read again and compared with
the target, the blurrier tile's semi-minor axis:

| Width rule | Blurred tiles read per blurred pair | Semi-major after / target, p10 / median / p90 | Over by more than 10% | Gradient energy, blurred / partner, along the partner's minor / major axis | Median width |
|---|---|---|---|---|---|
| Two probes per view, 0.4 and 1 (this design) | 0.80 | 0.92 / 0.99 / 1.02 | 1.0% | 1.13 / 1.45 | 0.47 |
| Two probes, 0.3 and 0.8 | 0.80 | 0.88 / 0.97 / 1.01 | 1.1% | 1.14 / 1.46 | 0.46 |
| Two probes, 0.5 and 1.25 | 0.80 | 0.90 / 0.98 / 1.03 | 0.8% | 1.13 / 1.46 | 0.49 |
| One probe, 0.6 | 0.40 | 0.86 / 0.96 / 1.05 | 6.1% | 1.15 / 1.47 | 0.46 |
| Ladder of blurred copies per view, each read: the widest of `0.25 · √2ⁿ` whose semi-major axis is at most the target | 1.61 | 0.80 / 0.90 / 0.98 | 0% | 1.18 / 1.53 | 0.35 |

The gradient energy is the mean square of the derivative along a direction
over the tile's variance, over the samples with data, the blurred tile's over
its partner's. Over 1 the blurred tile keeps more fine detail than its partner
along that direction. Along the partner's minor axis it keeps a little more
(1.13), the lean of a blur that stops at the target; along the partner's major
axis it keeps much more (1.45), the part of the partner's blur that a round
blur aimed at the minor axis leaves alone. A blur per direction ended at 1.02
and 1.09 on the same measures, with the blurred tile's semi-major axis past the
partner's semi-minor axis on 93% of tiles (by 1.73 at the median) and past the
partner's semi-major axis by more than 10% on 12.7%; this rule's passes the
partner's semi-major axis by more than 10% on 0.1%.

The alternatives were set aside for these reasons:

- **The ladder** reads every level it climbs through, so it never ends past
  the target, but it reads twice as many blurred tiles and ends 10% short at
  the median, since the level below the target is often well below it. The two
  probes end within 1% of the target at the median and past it by more than
  10% on 1% of tiles, at half the reads.
- **One probe** costs half the readings and over-blurs six times as often.
- **Other probe pairs** read the same within a percent or two; 0.4 and 1 bracket
  the widths most pairs need (median 0.47).
- **A blur along each direction in which the two ellipses differ** (each tile
  blurred along the directions in which it is the shorter, to the other's
  length there) rescues oblique views and lifts every member's agreement more,
  but reads grain and stripes as blur, blurs a view past the partner's sharpest
  direction, and tells members from lookalike tiles of other points no better
  than plain ZNCC (below).

### Skipping pairs that barely differ

`PairMatching::BlurMatchedAboveRatio` leaves a pair plain where the target is
less than a factor past the sharper tile's semi-major axis, which costs nothing
beyond the plain ZNCC. `DEFAULT_MIN_ELLIPSE_RATIO` is `1.25`. Even
`BlurMatched` leaves a pair plain where the target is within
`MATCHED_LENGTH_TOLERANCE` (5%) of the semi-major axis, a difference within the
spread of the width the assessment gives, and a sharper tile whose semi-major axis
is near 2 against a blurrier one past it is left alone.

| Ratio | Pool pairs blurred (22,035) | Bench pairs blurred (661 tracks) | Member-coherence pairs blurred (8,484 points) |
|---|---|---|---|
| 1 (`BlurMatched`) | 12.4% | 12.2% | 10% |
| 1.25 | 4.9% | 5.3% | 4.4% |

### The blur

The blur is an isotropic Gaussian applied by normalized convolution over the
samples that carry data: the colour planes are blurred premultiplied by the
data mask, the mask is blurred with them, and the blurred colour is divided by
the blurred mask at each sample that carries data. A sample without data (off
the photograph, or outside member coherence's common support) neither
contributes a value nor receives one, and neither does anything past the
tile's edge, so a tile next to the border is not darkened by it. A round
Gaussian is the product of a Gaussian along each grid axis, so it is applied as
two 1-D passes, down the columns and then along the rows, each a sum of whole
shifted rows.

## Cost

One thread, release build, a 24 × 24 three-channel tile (the mask is a fourth
plane):

| Step | µs |
|---|---|
| Plain pair readings (whole tile and the nine cells) | 3.6 |
| The two passes, `σ` 0.5 / 1 / 2 | 4.8 / 5.7 / 9.6 |
| Direct 2-D convolution, `σ` 0.5 / 1 / 2 | 26 / 49 / 160 |
| `pair_blur` from two ellipses | under 0.05 |
| The whole self-similarity reading of a tile | 8.7 |
| One tile's blur assessment, `assess_blur` (two blurs, two readings) | 28 |
| A blurred pair once its tile is assessed (blur, readings) | 8.6 |

The two passes run 5 to 17 times faster than the direct convolution. A tile's
assessment costs two blurs and two whole self-similarity readings, paid once
for each view a track blurs; a blurred pair then costs a blur and the pair's
readings. On the pool tracks the default adds 0.047 ms to a track's pairs
(17 µs per blurred pair with the assessments spread over them, 0.80 reads a
pair), where a blur per direction added 0.35 ms (12.7 µs per blurred pair, on
ten times as many pairs). A tile with samples off the photograph takes the
reading's slower route, which visits every sample, at about 50 µs a reading.
Member coherence reads its ellipses on the largest square inside its common
support, where every sample carries data, and its blurred renders the same
way.

**No AVX2 form.** Every pass is a sum of whole shifted rows, `dst[x] += w ·
src[x + s]`, which the compiler vectorizes. The same code compiled for AVX2
and FMA, chosen at run time, ran from 13% faster to 13% slower than the
default build depending on the kernel's shape (rows of 24 to 32 samples leave
little for wider registers), so the kernel has none.

## Accuracy

**The width against a known blur.** Where the blurry tile is the sharp tile
blurred by a known round Gaussian of 0.8 to 2 grid px, on a texture whose
ellipse is about 0.45 grid px, the rule picks the sharp tile and the blurred
tile's semi-major axis ends at 0.98 to 1.05 of the blurry tile's semi-minor
axis; the width is no wider than the planted one, and narrower where the
planted blur left the blurry tile's ellipse longer along one axis than the
other. A planted blur of 0.5 lengthens such a tile's ellipse by less than the
5% tolerance and is left plain.

**The two passes against the exact blur.** On a tile of four sinusoids, whose
blur by a Gaussian is known exactly (each amplitude scales by
`exp(−½ σ² |k|²)`), the passes are at most 0.41 grey levels off the exact
answer on sinusoids spanning ±115, at widths from 0.3 to 2. The direct 2-D
convolution is 0.61 off from a width of 0.6 up; it samples the kernel at `σ`
itself, which falls short of its width below that, where the passes' taps are
set so their variance is `σ²`. On a textured tile with samples missing, the two
agree to within 0.05 grey levels from `σ` 1 up.

**The Python port.** The illustration draws the blurred tiles from the width
the binding reports (`blur_sigma`) with two 1-D passes whose taps are the
kernel's; their ZNCC is the kernel's to three decimals on every pair drawn.

## What it changes

**Members, wrong views and blurred members.** On 990 pool tracks (100 a
dataset, seeded), each track's members were scored by their median ZNCC with
the other members against planted wrong views (lookalike tiles of other
points, neighbours, random tiles, and members shifted by 2 to 4 px), each
reading with the threshold that catches 90% of its own wrong views; and one
member's tile was blurred by a round Gaussian of `σ` 1, 2 or 3 grid px:

| Reading | Members over wrong views (AUC) | Members below the threshold | Blurred member below it, `σ` 1 / 2 / 3 | AUC on lookalike tiles / on a member shifted by 2 px |
|---|---|---|---|---|
| Plain | 0.9628 | 9.4% | 9.7% / 21.9% / 46.5% | 0.9562 / 0.9089 |
| Blur-matched, ratio 1.25 (this design) | 0.9635 | 9.2% | 6.9% / 6.0% / 12.4% | 0.9558 / 0.9110 |
| Blur-matched, every difference | 0.9637 | 9.1% | 6.8% / 5.7% / 11.8% | |
| A blur per direction, ratio 1.25 | 0.9657 | 8.6% | 4.8% / 3.5% / 5.8% | 0.9547 / 0.9179 |

Against plain ZNCC the design gains 0.0007 in AUC (bootstrap over tracks, 95%
interval 0.0003 to 0.0010); against a blur per direction it gives up 0.0022
(0.0016 to 0.0029), all of it on shifted members, where a blur per direction
lifts the member's agreement more. On lookalike tiles of other points, the
wrong views a blur can make look like members, a blur per direction tells
them apart worse than plain ZNCC does (0.9547 against 0.9562), and this design
about as well as plain (0.9558). A member blurred by a round Gaussian is the
case the rule is made for, and it brings the member below the threshold back
from 21.9% to 6.0% at `σ` 2; at `σ` 3 its semi-major axis reaches the
reading's 3 px cap and the target's 2 px cap, and 12.4% fall below.

**On the consumers**, measured on the review cases and samples of the ten
datasets of the reference-view work, all in one session on one machine:

| Consumer | Option | Added cost | Effect | Default |
|---|---|---|---|---|
| Reference view: agreement test and cell check ([reference-view.md](reference-view.md)) | blur-matched, ratio 1.25 | +0.13 ms per track (median; p90 0.81 ms), 2.1% of an evaluation (a blur per direction: +0.37 ms, 5.8%) | 28 of 77 hand picks exactly, as plain (tune half 18, held-out 10), against 30 for a blur per direction; the pick differs from plain on 25 of 661 tracks | on |
| Member coherence's decision ([member-coherence-validation.md](member-coherence-validation.md)) | blur-matched, ratio 1.25 | 1.18 × the plain run on one thread (a blur per direction 1.77 ×) | a planted member blurred by `σ` 2 is evicted 2.9% of the time against 4.9% plain (1.8% for a blur per direction); verdicts change on 0.4% of real points, in both directions | off |

## Implementation notes

**The ellipse is the measurement of the tile being blurred.** A consumer passes
the ellipse of the very render it correlates: the bench the `R×R` tile's whole
reading over its samples on the photograph, member coherence a reading of its
own render over the largest square inside the common support. An ellipse read
on another render of the view (another resolution, sampler or support)
describes another tile. The probes' blurred tiles must then be read the same
way, since the width compares their readings with the partner's:
`blur_matched_pairs` reads them with `read_tile_ellipse`, member coherence with
its own square, and an ellipse read any other way would set the blur against a
different scale.

**A tile's assessment depends on its own tile only.** It is read once for each
view some pair blurs, from that view's tile and ellipse alone, and every pair
reuses it; which views are assessed depends on the ellipses alone. So a pair
reads the same in a track as alone, bit for bit, and
the same whatever order the views come in (to rounding, since reversing a
pair's two tiles reorders the sums); tests check both. Which tile is blurred
does not depend on the order either: `pair_blur` swaps its answer with its
arguments.

**The first pass covers the band the second reads.** The tile sits in a buffer
padded with zeros by both passes' reach, and the first pass writes the tile and
the second pass's reach round it, so the two passes compose as one 2-D
convolution of the tile extended by zeros rather than of a tile cut off after
the first pass. A band is computed in whole 8-sample groups; the columns past
the tile it adds are written and never read back. The padded buffer is all
zeros between calls: each call writes it only on the tile's rows, from the
tile's first column, and clears what it wrote before it returns. Two widths can
share a buffer size and differ in their padding, so a buffer cleared only when
its size changed would hand the previous tile to the next call; a test reuses
one scratch over many widths and sides and compares each blur with a fresh
scratch's, bit for bit.

**Each pass's taps have the variance asked for.** A Gaussian sampled at whole
pixels has less variance than its width says once the width is under about 1:
0.215 for 0.25 at `σ` 0.5, and under a tenth of 0.09 at `σ` 0.3. Most pairs are
blurred by less than 1, so under `σ` 1 the taps are those of a sampled Gaussian
whose width is found by bisection so their variance is `σ²`.

**A pair left plain reads exactly the plain value.** `pair_zncc_readings`
gathers the cells' sums in the same order and by the same formula as
`pair_zncc_grid` ([reference-view.md](reference-view.md)), so a pair the rule
leaves alone reads the plain cell grid bit for bit, and member coherence copies
its plain value for such a pair.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `DEFAULT_MIN_ELLIPSE_RATIO` | `1.25` | The factor by which the target must exceed the sharper tile's semi-major axis for `BlurMatchedAboveRatio` to blur the pair; it leaves 95% of the pairs on real tracks plain |
| `MAX_MATCHED_LENGTH` | `2` grid px | The longest semi-major axis blur matching aims for; a longer semi-minor axis of the blurrier tile is read as this one |
| `GROWTH_PROBE_SIGMAS` | `[0.4, 1.0]` grid px | The isotropic blurs a tile is blurred by, once each, for its blur assessment |
| `MATCHED_LENGTH_TOLERANCE` | `0.05` | A target within this fraction of the sharper tile's semi-major axis is not blurred to, whatever the ratio |
| `MIN_SHARPER_LENGTH` | `0.05` grid px | The shortest semi-major axis the skip ratio reads |
| `MAX_BLUR_SIGMA` | `3` grid px | The widest blur |
| `MIN_WINDOWED_SAMPLES` | `8` | The fewest samples with data in both tiles and weight in the window a whole-tile reading is taken over |

Each consumer's option and its default are in that consumer's spec. The
constants are defined in
[blur_matched.rs](../../../crates/sfmtool-core/src/patch/blur_matched.rs)
(`GROWTH_PROBE_SIGMAS`, `MAX_BLUR_SIGMA`),
[pair_sharpness.rs](../../../crates/sfmtool-core/src/patch/pair_sharpness.rs)
(the pairing rule's) and
[pair_readings.rs](../../../crates/sfmtool-core/src/patch/reference_view/pair_readings.rs)
(`MIN_WINDOWED_SAMPLES`).

## Python bindings

The per-tile operations take one tile as `OrientedPatch.render_view_tile`
renders it, an `(R, R, C)` uint8 array (one channel grey, two grey and alpha,
three RGB, four RGB and alpha; alpha 0 marks a sample without data), with its
optional `(R, R)` bool `valid` flags:

- `sfmtool._sfmtool.patches.assess_blur(samples, *, valid=None, ellipse=None)`
  returns the tile's blur assessment as a dict, `semi_axes` `(2,)`, `growth`
  `(2, 2)` (one row of [major, minor] per probe), `probe_sigmas` `(2,)` and
  `ellipse_matrix` `(2, 2)`, or `None` where the ellipse or a probe's cannot be
  read. The ellipse is read here, over the samples with data, unless one is
  passed in; one passed in should be read that way too.
- `blur_sigma_to_reach(assessment, length)` returns the width that brings the
  tile's semi-major axis to `length`, 0 where it is already that long, and
  `None` where the readings do not grow.
- `blur_to_length(samples, assessment, length, *, valid=None)` returns a dict
  of `samples`, the colour channels blurred (`(R, R, c)` float32, alpha left
  out), and `sigma`, the width; `None` where the assessment gives no width.

They raise `ValueError` for a tile that is not square, 3 or more on a side,
with 1 to 4 channels, for a `valid` or `ellipse` of the wrong shape, and for an
assessment without `semi_axes` of 2 and `growth` of `(2, 2)`.

```python
from sfmtool._sfmtool.patches import assess_blur, blur_to_length

sharp = patch.render_view_tile(camera_a, pose_a, image_a, keypoint=kp_a)
blurry = patch.render_view_tile(camera_b, pose_b, image_b, keypoint=kp_b)
a = assess_blur(sharp["samples"], valid=sharp["valid"])
b = assess_blur(blurry["samples"], valid=blurry["valid"])
if a["semi_axes"][0] < b["semi_axes"][1]:  # sharper along every direction
    out = blur_to_length(sharp["samples"], a, min(b["semi_axes"][1], 2.0), valid=sharp["valid"])
```

The pair-matrix convenience below reads a track's tiles the way the bench
does, built from the same per-tile operations and the pairing rule of
`pair_sharpness`.

`sfmtool._sfmtool.patches.blur_matched_zncc_matrix(tiles, *, valid=None,
ellipses=None, matching="blur_matched", min_ellipse_ratio=1.25,
window="gaussian_disk", window_sigma=0.6)` reads every pair of a
`(k, R, R, C)` uint8 stack of tiles. One channel is grey, two grey and alpha,
three RGB and four RGB and alpha; alpha is not correlated, and 0 there marks a
sample without data. `valid` is an optional `(k, R, R)` bool stack, as
`OrientedPatch.render_view_tile` returns it, `False` marking a sample without
data. `ellipses` is `(k, 2, 2)` float64, NaN for a tile without one, or `None`
to read each tile's whole self-similarity here, over the samples with data,
which is how the blurred tiles are read; ellipses passed in should be read that
way too, or the blur is set against a different reading. It returns a dict:
`zncc` `(k, k)`, `zncc_grid` `(k, k, 3, 3)`, `blurred` `(k, k)` bool,
`blur_sigma` `(k, k)` (the width the row's tile was blurred by against the
column's, 0 where it was not), `pairs`, `pairs_blurred` and `ellipse_matrix`
`(k, 2, 2)`. It raises `ValueError` for a stack that is not square tiles of 3
or more on a side with 1 to 4 channels, a `valid` or `ellipses` of the wrong
shape, an unknown name, or a ratio under 1 or not finite. The blur has one
shape, so there is no option to choose one; an unknown keyword raises
`TypeError`.

The bench's readings are those of the tiles `render_view_tile` renders at each
view's keypoint, at the evaluation's resolution and with its sampler, read with
their `valid` flags, the ellipses left to the function, a ratio of 1.25 and the
default window:

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
checks the semi-axes read off a matrix; that a tile's assessment keeps both
axes of its own reading and of each probe's, each what a separate reading of
the tile blurred by that probe gives, bit for bit, with a reused scratch as
with a fresh one; that a tile whose probes cannot be read has none; the width
read off constructed assessments (exact on a linear growth, a flat piece passed
over, no growth giving none, the cap); that a tile blurred to a copy of it
blurred by a known round Gaussian ends with its semi-major axis within 5% of
the copy's semi-minor axis, by no more than the planted width; that a tile
already long enough comes back unblurred with a width of 0; the two passes
against the exact blur of a tile of sinusoids and against the direct 2-D
convolution, and the two against each other round missing samples; that one
scratch reused over many blurs of different widths and sides gives what a
fresh one gives, bit for bit; and that a sample without data neither gives nor
takes a value and a flat tile stays flat at the edge and round holes. An
ignored test, `timing`, prints the cost table above.
[pair_sharpness/tests.rs](../../../crates/sfmtool-core/src/patch/pair_sharpness/tests.rs)
checks that equal or unreadable ellipses blur nothing; that the sharper tile is
the one whose semi-major axis is shorter than the other's semi-minor axis, its
target that semi-minor axis capped at 2, and that swapping the pair swaps only
which tile is named; that a tile blurry along one direction only, and grain at
two angles, blur nothing; the skip ratio and the 5% tolerance, and that a
planted blur within it is left plain; that only the views some pair names the
sharper are assessed, each once and in view order, and none under `Plain`;
that a view without an assessment leaves its pairs plain; and that which views
are assessed, and which view of each pair is blurred to what length, do not
depend on the order of the views.
[reference_view/pair_readings/tests.rs](../../../crates/sfmtool-core/src/patch/reference_view/pair_readings/tests.rs)
checks that real grained tiles at three angles read every pair plain, bit for
bit; that only one tile of a pair is blurred, and the same tile by the same
width with the views reversed; that each view some pair blurs is assessed once
however many pairs blur it; that the readings do not depend on the order of
the views; that a pair reads the same in a track as alone, bit for bit; that a
tile and a blurred copy of it read a ZNCC near 1 blur-matched and well above
plain; and that equally sharp views read the plain value.
[member_coherence/tests.rs](../../../crates/sfmtool-core/src/patch/member_coherence/tests.rs)
checks that a member blurred in its photograph is lifted blur-matched, more
than any pair of the sharp members is.
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
- **Directional blur.** No tile is blurred along one direction more than
  another. A view blurry along one direction only is read plain against a view
  sharp along it (§ "Which tile is blurred, and to what length").
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

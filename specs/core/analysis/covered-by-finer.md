# Covered by Finer

A feature detector answers at a scale, and a wide answer averages over whatever
detail lies under it. So when a set of tracks holds both a wide feature and a
narrow one at the same place in the same photograph, the wide one is the worse
evidence: its support spans many pixels, and where the detail under it sits at
more than one depth its own triangulated depth is a blend of them. This rule
finds those places and says which observations to retire.

The domain is the image plane, and the unit is one observation. Each carries the
image it was seen in, the thing it belongs to, its pixel position, the extent of
the footprint it draws, and its own feature size. A row is retired where another
row **in the same image, on another owner**, sits inside the first row's
footprint and is at least a fixed ratio finer. The coarse side is the one
retired, never the fine one.

The rule knows nothing about a reconstruction, a solve or a file: it reads flat
arrays and hands back flat masks, so the same statement serves a caller holding
`.matches` cluster arrays and one holding a reconstruction's tracks. The
reconstruction-level operation built on it is
[prune-covered-observations.md](../reconstruction/prune-covered-observations.md);
the neighbourhood enumeration underneath it is
[keypoint-reach.md](keypoint-reach.md).

## Rust API

The rule lives in
[covered_by_finer.rs](../../../crates/sfmtool-core/src/analysis/covered_by_finer.rs),
bound as `sfmtool._sfmtool.analysis.covered_by_finer`.

```rust
/// One row per observation, over whatever set of them the caller tracks.
pub struct CoveredRows<'a> {
    pub image_of_row: &'a [i64],
    pub owner_of_row: &'a [i64],
    pub xy_px: &'a [f64],
    pub reach_px: &'a [f64],
    pub radius_px: &'a [f64],
    pub protected: Option<&'a [bool]>,
}

/// The rule's thresholds.
pub struct CoveredOptions {
    pub ratio: f64,
    pub min_fine_radius_px: f64,
    pub min_observations: usize,
}

/// The rule's verdict, per row and per owner.
pub struct CoveredByFiner {
    pub flagged: Vec<bool>,
    pub keep_row: Vec<bool>,
    pub keep_owner: Vec<bool>,
    pub census: CoveredCensus,
}

pub fn covered_by_finer(
    rows: CoveredRows<'_>,
    owner_count: usize,
    options: &CoveredOptions,
    progress: &Progress<'_>,
) -> Result<CoveredByFiner, CoveredByFinerError>;
```

**Two radii, not one.** `reach_px` and `radius_px` are different lengths and the
rule needs both: containment is asked at the reach, and "finer" is asked of the
radius. A caller whose footprint is a multiple of its feature size states that
multiple once, on the way in, rather than handing over one length and a factor
the rule would have to know what to do with. Keeping them apart is also what
lets a caller draw the footprint from something else entirely -- a stored
extent, a fixed pixel budget -- without the scale test changing meaning.

**An owner rather than a point.** The rows are grouped by an opaque `i64`, so
the rule serves a caller whose rows belong to reconstruction points and one
whose rows belong to match clusters without either having to translate. What the
owner buys is two things: nothing covers itself, and the sweep at the end has
something to count.

**One verdict struct rather than three returns.** `flagged`, `keep_row` and
`keep_owner` are three readings of one decision, and a caller that took only the
first would be one sweep away from a set of rows whose owners it would not keep.
Handing all three back together is what makes that mistake unavailable.

**A census, always.** The counts are what a caller reports and what a caller
tunes against: a rule that fired nowhere and a rule that found nothing contained
are different situations, and the census is what tells them apart without
re-running anything.

```rust
use sfmtool_core::analysis::covered_by_finer::{
    covered_by_finer, CoveredOptions, CoveredRows,
};
use sfmtool_core::progress::Progress;

// Two rows of one image: a wide one at the origin and a fine one 1 px away,
// on different owners and exactly one octave apart.
let out = covered_by_finer(
    CoveredRows {
        image_of_row: &[0, 0],
        owner_of_row: &[0, 1],
        xy_px: &[0.0, 0.0, 1.0, 0.0],
        reach_px: &[10.0, 2.5],
        radius_px: &[4.0, 2.0],
        protected: None,
    },
    2,
    &CoveredOptions { min_observations: 1, ..CoveredOptions::default() },
    &Progress::none(),
)?;
assert_eq!(out.flagged, [true, false]);
```

## Theory

### The three tests

A pair `(big, small)` retires `big` when all three hold:

1. **Containment.** `small`'s centre lies within `big`'s own reach:
   `d <= reach[big]`. The relation is directed and the disk belongs to the row
   being judged, so a wide feature is judged over the area it actually claims
   rather than over its neighbour's.
2. **Another owner, strictly smaller.** `owner[small] != owner[big]` and
   `radius[small] < radius[big]`. Nothing covers itself, and nothing covers
   something its own size or larger.
3. **A band apart.** `radius[big] >= ratio * radius[small]`, and
   `radius[small] >= min_fine_radius_px`.

The scale test is what makes this a statement about evidence rather than about
density. In a well-filled set of tracks almost every feature has *something*
smaller somewhere inside it, so a rule without it would retire nearly
everything; what the ratio asks is whether the neighbour is finer by enough that
it resolves detail the coarse feature cannot. One octave is the natural bar,
because an octave is the spacing a scale-space detector's own bands are cut on.

The comparison is **non-strict**: a pair exactly one octave apart is finer. A
strict bar would make the verdict turn on the last bit of a float for the most
common pair in the population, which is the one place a rule must not be
delicate.

The fine-radius floor exists because a *measured* radius can collapse. A
projected footprint of a hundredth of a pixel is not a fine feature; it is a
degenerate projection that happens to satisfy every ratio. Off by default, since
a caller whose radii come from a detector has no such failure mode.

### Order independence

A row is retired by the **existence** of a cover, and every cover is measured
against the rows as they stand, so no pass ever reads a decision an earlier pass
made. That makes the verdict a pure function of the input set: the same rows in
any order give the same flags, and no scheduling of the enumeration underneath
can change them. Determinism here is not a convenience -- it is what lets the
rule's answer be compared byte for byte against another implementation of the
same statement.

### Protection

A protected row is never retired and still covers. The asymmetry is the point: a
row a caller marks is one somebody decided by hand, and the rule must not undo
that; but the reason it was marked says nothing about the evidence it offers its
neighbours, so it goes on covering them. The census counts protected rows a
passing pair would otherwise have retired, which is what protection actually
bought as opposed to how many rows carried the mark.

### The sweep

A rule that retires observations can leave an owner with too few to be worth
keeping. Rather than hand that back for every caller to notice, the rule applies
the bar itself: an owner whose surviving rows number fewer than
`min_observations` is dropped, and its survivors go with it. Two is the natural
floor for a caller whose owners are triangulated points, since one sighting
fixes a bearing and no position.

## Implementation notes

The enumeration is [`spatial::keypoint_reach`](keypoint-reach.md), which emits
`(row, candidate, distance)` for every candidate inside `row`'s own reach. The
direction matters and is easy to get backwards: the emitted `row` is the one
whose disk was asked, so it is the **coarse** side of the pair here, and the
`candidate` is the potential cover.

The containment test is restated over the pair stream although the enumeration
has already applied it. That is deliberate and costs one comparison per pair: it
keeps the rule readable as the three tests it is, and it means a future
enumeration that widened its answer would not silently widen the rule.

The distance is compared against the reach with `<=`, and the enumeration
computes it as `sqrt(dx * dx + dy * dy)` rather than through a fused or scaled
formula, so that a NumPy transcription of the same statement rounds the same way.
That is what the binding's parity test rests on.

`NaN` is load-bearing in two different places and means two different things. A
`NaN` **reach** asks nothing: the row is judged by no disk of its own, and still
appears as a candidate of other rows. A `NaN` **radius** fails every comparison,
so the row neither covers nor is covered. A caller whose measurement failed
writes `NaN` to both and gets exactly the behaviour a failed measurement should
have.

The three owner counts in the census are over owners that hold at least one row.
An owner the caller declared but that no row names is in none of them, because
nothing was read about it and nothing should be claimed about it.

## Parameters

Defined by `CoveredOptions::default()` in
[covered_by_finer.rs](../../../crates/sfmtool-core/src/analysis/covered_by_finer.rs).

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `ratio` | `2.0` | How many times finer the covering row has to be. One octave; the comparison is non-strict. |
| `min_fine_radius_px` | `0.0` | A covering row below this says nothing. `0.0` is off. |
| `min_observations` | `2` | Surviving rows an owner needs to be kept. |

## Python bindings

`sfmtool._sfmtool.analysis.covered_by_finer(image_of_row, owner_of_row, xy_px,
reach_px, radius_px, n_owners, *, ratio=2.0, min_fine_radius_px=0.0,
min_observations=2, protected=None)`.

`image_of_row` and `owner_of_row` are `(n,)` `int64`, `xy_px` is `(n, 2)`
`float64`, `reach_px` and `radius_px` are `(n,)` `float64`, and `protected`, when
given, is `(n,)` `bool`. Arrays are accepted in either memory order.

Returns a dict with `flagged`, `keep_row` and `keep_owner` as `bool` arrays and
`census` as a dict of the counts. The argument names are the Rust ones, except
`n_owners` for the Rust positional `owner_count`, which is the name the
surrounding NumPy code already uses for that quantity.

```python
from sfmtool._sfmtool.analysis import covered_by_finer

out = covered_by_finer(image_of_row, owner_of_row, xy_px, reach_px, radius_px,
                       n_owners, min_fine_radius_px=1.0)
kept_rows = rows[out["keep_row"]]
```

Every refusal is a `ValueError` carrying the Rust error's sentence: disagreeing
row counts, an `xy_px` that is not `(n, 2)`, a row naming an owner outside
`[0, n_owners)`, an unusable ratio or floor, and a negative reach, which is named
by row.

## Testing

`analysis/covered_by_finer/tests.rs` holds the rule's own cases, ported one for
one from the NumPy stage this was lifted out of:

- **The disk the rule reads is the reach**, not the radius: a pair separated by
  more than the reach and less than the radius is left alone.
- **The fine row is never the one retired**, on any pair.
- **A same-scale neighbour retires nothing**, and it is the scale test that
  spares it: the pair is still counted contained.
- **A row never covers itself**, two rows of one owner never pair, and rows of
  different images never pair even at identical pixels.
- **The ratio bar is read where the band edge is**: exactly one octave apart is
  finer, a hair under it is not.
- **The floor refuses a collapsed measurement**, and is off at its default.
- **A row of unstated radius pairs with nothing**, in either direction.
- **A protected row is spared and still covers**, and the census says how many
  rows that bought.
- **An owner under the bar goes whole**, survivors included; an owner no row
  names is in neither drop count.
- **The whole rule** over an end-to-end set, verdict for verdict against the
  counts the NumPy stage reported on it, and the same set scaled down until no
  footprint reaches anything.
- **Determinism**: two readings agree, and reversing the row order gives the
  reversed verdict and the identical census.
- **Refusals**: disagreeing lengths, a short protection mask, an owner out of
  range, an unusable threshold, a negative reach, and a cancelled reading.

`tests/rust_bindings/test_covered_by_finer_rust_bindings.py` holds the binding
to a brute-force NumPy transcription of the same statement -- an `O(n^2)` double
loop per image -- over seeded random rows of a few thousand. The generated
population is built to sit on the rule's edges: radii are exact powers of two, so
octave ties are common and land on the non-strict `>=`, and a tenth of the rows
share pixel positions exactly, so containment is decided at distance zero. Every
bit of the verdict has to agree, the census included, across the ratio, the
floor, the observation bar and the protection mask.

## Non-goals

- **Deciding what a footprint is.** The reach arrives measured. Turning a
  detector scale, a stored patch extent or a projected frame into one is the
  caller's, because each caller has a different right answer.
- **Moving or re-solving anything.** The rule produces masks. What a caller does
  with them, and whether the geometry is re-read afterwards, is the caller's.
- **Choosing which of two equal-sized neighbours to keep.** The rule is about
  scale, and two rows of the same size are evidence of the same quality; a rule
  for that question would be a statement about density, which this deliberately
  is not.

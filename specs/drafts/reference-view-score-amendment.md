# One score per view for the reference-view rule (amendment)

**Status:** Draft

Amends [`../core/patch/reference-view.md`](../core/patch/reference-view.md)
§ "The rule", § "Each threshold and its reason" and § "The stored bitmap",
which points back here, and the reference-view rule of
[`sharper-patch-bitmap.md`](sharper-patch-bitmap.md) Part 5, which also points
back here. Where it is built it also changes
[`../core/patch/keypoint-subpixel-refinement.md`](../core/patch/keypoint-subpixel-refinement.md#the-fused-mean),
[`../core/patch/patch-keypoint-localization.md`](../core/patch/patch-keypoint-localization.md),
[`../core/reconstruction/add-image-to-tracks.md`](../core/reconstruction/add-image-to-tracks.md),
[`../core/patch/patch-normal-refinement.md`](../core/patch/patch-normal-refinement.md),
[`../core/bench/editable-track.md`](../core/bench/editable-track.md),
[`../gui/track-view.md`](../gui/track-view.md),
[`../gui/mcp-server.md`](../gui/mcp-server.md) and the
[`../GLOSSARY.md`](../GLOSSARY.md) entry for the reference view (§ "What
changes elsewhere").

The reference-view rule picks the one view of a point whose `R×R` tile is
stored as the point's patch bitmap. Today it passes the views through a series
of all-or-nothing tests: hard gates on coverage, clipping, viewing angle and
cell deficit, dropped in turn when no view passes them; then a gate on
agreement with the best candidate; then the smallest self-similarity radius
among the views left. This draft replaces the tests with one cost per view, an
effective radius in grid px that the view's sharpness, its viewing angle and
soft penalties for disagreement, a differing ninth, partial coverage and
clipping all enter, and picks the view with the lowest cost. It keeps two hard
limits. It also removes the fused mean of the views as a stored bitmap: every
point that has a view meeting the hard limits stores that view's tile and
names it as its reference.

The draft specifies the form of the cost, each term and each parameter, with
starting values. The values are not tuned; § "How the values will be set"
describes how the implementation sets them, and this draft states no results.

## What the rule does today, and the two problems

[`choose_reference_view`](../../crates/sfmtool-core/src/patch/reference_view.rs)
takes a `ReferenceReadings` per view and:

1. keeps as **candidates** the views with coverage `≥ 0.99`, clipped share
   `≤ 0.05`, viewing angle `≤ 65°` and cell deficit `≤ 0.3`;
2. where no view passes, drops the 65° limit (`without_angle`), then the cell
   check (`without_angle_or_cells`), then coverage and clipping
   (`without_any`); under every fallback a view at `90°` or more, edge on or
   from behind, is turned away;
3. keeps the candidates whose pair ZNCC is within `0.15` of the best
   candidate's;
4. picks the one with the smallest self-similarity semi-major axis, a tie
   going to the smaller semi-minor axis and then to the earlier view. Where
   none of them has a self-similarity reading it picks nothing.

The stored bitmap
([`stored_bitmap.rs`](../../crates/sfmtool-core/src/patch/stored_bitmap.rs))
is the pick's tile, except where the rule picks nothing or reaches its pick
only through `without_any` (`ReferenceRender::stored_reference`). There it is
the fused mean of the views, and the point records the reference `-1` (a
`without_any` pick is stored after all where no fused mean renders). On a
solve of `dino_dog_toy`, 559 of 18,991 points take the fused mean through
`without_any`, as
[reference-view.md](../core/patch/reference-view.md#the-stored-bitmap)
reports.

**Problem 1: the angle has no say below 65°, and each criterion acts alone.**
The angle is only a gate, so among the views under 65° it does not count at
all. A view at 20° with a semi-major axis of 0.62 grid px loses to a view at
55° with 0.60, though the first depends far less on the patch's normal being
right and is the better reference. The other criteria behave the same way:
a view just past one threshold is out however good it is on the others, and a
view just inside it counts the same as one far inside it. The rule needs a
balance between the criteria rather than a sequence of gates.

**Problem 2: some points still store a fused mean.** A point stores the mean
of its views, and names no reference, where no view left after the agreement
test has a self-similarity reading, where every view is at 90° or more, and
where the pick was reached only by `without_any`. The mean has the problems
the reference view was introduced to avoid (it mixes exposures and blurs the
detail), and a point with a `-1` reference has no anchor for the localizer.
Every point that has a usable view should get a reference.

## The cost

Each view `v` gets a cost in grid px, and the view with the lowest cost is the
reference:

```
cost_v = r_eff,v · A_v · K_v · C_v · P_v
r_eff,v = √(r_v² + (c · tan θ_v)²)
c = (R / √12) · σ_n
```

- `r_v` is the self-similarity semi-major axis of the view's `R×R` tile, in
  grid px: the reading the rule ranks by today.
- `θ_v` is the viewing angle, measured on the side of the patch the pick is
  made on (§ "The facing side").
- `A_v`, `K_v`, `C_v` and `P_v` are the soft penalties for agreement, cell
  deficit, coverage and clipping (§ "The soft penalties"), each `1` inside its
  tolerance band.

A tie on the cost goes to the smaller semi-minor axis, a missing one counting
as infinite, and then to the earlier view, as today.

**Why a product on an effective radius.** The cost stays in the unit the rule
already ranks by, so a cost reads as "this view counts as if its tile were
this blurry". The self-similarity radius depends on the texture as well as on
the view and is only compared within one track; a multiplicative penalty makes
the same relative trade-off on a track of fine texture as on one of coarse
texture, where an additive penalty in grid px would weigh more on the first.
A factor of `2` reads as "this view counts as twice as blurry", which is how
the person reading Track View will see it.

### The angle term

An error `σ_n` in the patch's normal, in radians, tilts the patch's plane
about its centre. A sample at distance `d` from the centre, along the
direction the plane tilts, moves out of the plane by about `d · σ_n`, and a
view at angle `θ` sees that out-of-plane step as a shift of about
`d · σ_n · tan θ` along its tilt direction
([reference-view.md](../core/patch/reference-view.md#the-measurements)
§ "Tilt direction"). The tile is re-anchored on the keypoint, so its centre
does not move, and the shift grows with the distance from the centre. Over an
`R×R` tile the root mean square of `d` along one axis is `R/√12`, the standard
deviation of a uniform spread over `R`, so the root mean square shift is
`c · tan θ` grid px with `c = (R/√12) · σ_n`. It is added to the radius in
quadrature, as two independent spreads in the same unit: a view facing the
patch costs its radius alone, and the angle term overtakes the radius only
where `c · tan θ` approaches it.

**The parameter `σ_n`** is the normal error the rule expects, the one free
parameter of the term. Starting value `3°` (`0.052` rad), untuned. At
`R = 24`, `c ≈ 0.36` grid px, and `c · tan θ` is about `0.21` at 30°, `0.36`
at 45°, `0.78` at 65° and `2.06` at 80°. At `R = 12`, `c ≈ 0.18`. With
`σ_n = 0` the term is gone and the rule ranks by the radius alone, as today
inside the 65° gate.

**On the case of Problem 1**, at `c ≈ 0.36`: the 20° view's cost is
`√(0.62² + (0.36 · 0.364)²) ≈ 0.63` and the 55° view's
`√(0.60² + (0.36 · 1.428)²) ≈ 0.79`, so the 20° view is picked. At `c = 0.3`
the two read about `0.63` and `0.74`.

The term replaces the 65° gate. On the two ground truths, removing the gate
made the 15 picks it changed localize the other views 0.12 px worse on average
([reference-view.md](../core/patch/reference-view.md#each-threshold-and-its-reason));
the tuning checks that the angle term does not bring those picks back
(§ "How the values will be set").

**A view at exactly `90°`** has `tan θ` infinite and so an infinite cost. It is
ranked after every view with a finite cost, so it is picked only where every
view the pick is made from has an infinite cost.

### The soft penalties

Each penalty is a **hinge**: `1` up to a tolerance `x₀` and growing linearly
past it with a scale `s`,

```
F(x) = 1 + max(0, x − x₀) / s
```

so `s` is the excess over the band at which the view counts as twice as blurry.
A hinge is chosen over a smooth curve (a quadratic past the band, a softplus,
an exponential) because:

- inside the band the factor is exactly `1`, so the views that pass today's
  thresholds are ranked by `r_eff` alone, and the band edges start at today's
  thresholds, of which the agreement margin and the cell deficit bar were
  tuned against the hand picks;
- past the band it keeps growing without a cliff, so a view slightly past a
  threshold is slightly worse rather than out, which is the balance Problem 1
  asks for;
- it is continuous, so a small change in a reading moves the cost by a small
  amount, and it has two parameters per term, each with a plain meaning, which
  is the fewest that set both where the penalty starts and how fast it grows.

| Factor | Reading `x` | Band `x₀` (start) | Scale `s` (start) | Today's test it replaces |
|---|---|---|---|---|
| `A_v`, agreement | shortfall of the view's pair ZNCC below the best pair ZNCC among the views the pick is made from | `0.15` | `0.15` | within `REFERENCE_AGREEMENT_MARGIN` (`0.15`) of the best candidate |
| `K_v`, cells | cell deficit | `0.3` | `0.3` | `REFERENCE_MAX_CELL_DEFICIT` (`0.3`) |
| `C_v`, coverage | shortfall of the coverage below `0.99`, `max(0, 0.99 − coverage)` | `0` | `0.03` | `REFERENCE_MIN_COVERAGE` (`0.99`) |
| `P_v`, clipping | clipped share | `0.05` | `0.05` | `REFERENCE_MAX_CLIPPED_SHARE` (`0.05`) |

Every value in the table is a starting value and untuned. With them, a view
`0.30` below the best pair ZNCC, a view with a cell deficit of `0.6`, a view
covering `0.96` of its tile and a view with a tenth of its pixels clipped each
count as twice as blurry; a view at the coverage floor of `0.9`
(§ "Hard limits") counts as four times as blurry.

**A sharp view is not penalised for detail the others lack.** That is why the
agreement margin exists today: a sharp view correlates worse with blurrier
views, since its detail is missing from them, and where the earlier margin of
`0.05` missed, the hand-picked view sat `0.06` to `0.21` below the best
([reference-view.md](../core/patch/reference-view.md#each-threshold-and-its-reason)).
The agreement term keeps that protection in two ways: the reading is the
shortfall below the best view's pair ZNCC, not an absolute level, and the
band `x₀` starts at the margin, so a sharp view within `0.15` of the best pays
nothing for its lower agreement. Today the rule reads the agreement plain
because blur-matched agreement changed no exact pick as a gate
([reference-view.md](../core/patch/reference-view.md#why-the-agreement-is-read-plain));
a term that acts on every view, rather than a gate, may be more sensitive to
it, so the tuning measures the term on blur-matched pair ZNCC beside the plain
one.

The cell deficit already measures a view's ninths against the track's typical
agreement rather than against an absolute level, and its band keeps a view
that differs from the others in a ninth by less than `0.3` unpenalised, so a
sharp view is not penalised by it either.

### Missing readings

A reading the view does not have is handled so that a missing reading never
makes a view look better than one that has it:

- **No self-similarity radius** (`semi_major` is `None`): `r_v` is the largest
  `r` among the views the pick is made from that have one, so the view ranks
  as no sharper than the bluntest view that was read, and the angle and the
  penalties still order it. Where no view has one, `r_v = 1` grid px for every
  view (a starting value; the per-dataset medians of the stored bitmap's
  radius in
  [reference-view.md](../core/patch/reference-view.md#cost-of-the-stored-bitmap)
  run from 0.40 to 1.06 grid px), so the pick is made by the angle and the
  penalties alone. Today the rule picks nothing in that case.
- **No pair ZNCC**: the shortfall is the largest shortfall among the views
  that have one. Where no view has one, `A_v = 1` for every view, as today the
  agreement test then passes every candidate.
- **No viewing angle**: the angle term is the largest angle term among the
  views that have an angle. Where none has one, the term is `0` for every
  view.
- **No clipped share or cell deficit**: the factor is `1`, as today a missing
  clipped share or cell deficit passes, since nothing is there to count
  against the view.

## Hard limits

Two limits stay all-or-nothing. A view that fails either is never the
reference, whatever its cost:

- **No tile.** A view that renders no tile, so has no coverage reading.
- **Coverage below `0.9`.** A view with less than 90% of its tile's samples on
  the photograph is never the reference, since the bitmap taken from it would
  have holes over a tenth of the patch or more. This floor is the maintainer's
  decision and is not tuned. Between `0.9` and `0.99` the coverage penalty
  applies.

A view meeting both is **eligible**. The 65° angle gate, the cell and clipping
gates and every fallback are removed.

### The facing side

Today a view at `90°` or more is turned away under every fallback, since it
sees the patch edge on or from behind. The rule now picks a side of the patch
first and measures every angle on it:

- **Where any eligible view sees the patch at less than `90°`**, the pick is
  made from the eligible views at `90°` or less (and those with no angle
  reading), with `θ` as read. The views at more than `90°` are excluded, as
  they are today.
- **Where every eligible view sees the patch at `90°` or more**, the pick is
  made from all of them, with the angle measured against the flipped normal
  `−n`: `θ' = 180° − θ`.

Why measuring against `−n` is sound there: a tile samples the same points of
the patch's grid in every view, so a tile rendered from behind the patch is
laid out like one from in front, sample for sample, and all the views of such
a point see the patch from the same side. The likely cause is a normal that
points into the surface, and under that reading `θ'` is the angle each view
sees the surface at. The fused mean the stored bitmap holds today for these
points is a mean of the same tiles (it leaves out a view that does not
project the point, not a view behind the patch), so a single view's
tile is as consistent with the others as the mean was, and sharper. Normal
refinement's obliquity prior already weights by `|v̂ · n|`, which is the same
for both sides.

**Views on both sides.** Where some views face the patch and some see it from
behind, the facing side is kept, as above, and the views behind are excluded.
That keeps today's pick on every track that has a facing view (up to the
cost's own changes), and changes only the points that store a fused mean
today because every view is at `90°` or more. The alternative, picking the
side that more views, or the lowest-cost view, see, would turn a point's
reference to the back of its patch on a track where most views happen to be
behind a normal that is nearly edge on to them, which is a normal that normal
refinement should correct rather than one the reference should follow. The
mixed tracks are counted during the tuning.

The best pair ZNCC that the agreement shortfall is read against is the best
among the views the pick is made from: eligible and on the chosen side.

## The stored bitmap

**Every point with an eligible view stores the pick's tile** and names it as
its reference. The fused mean is no longer written as a stored bitmap:
`ReferenceRender::stored_reference` and `stored_view` go, since the pick is
always the stored view, and `render_patch_bitmap` returns the pick's tile or
nothing.

**A point with no eligible view gets no bitmap** and the reference `-1`: no
view has 90% of the patch on its photograph, or none renders a tile. The
maintainer set the floor so that a tile under it is not a reference, so the
rule does not fall back to the most covered view under it. The kernels report
no bitmap the way they report a point with fewer than two views today
([keypoint-subpixel-refinement.md](../core/patch/keypoint-subpixel-refinement.md#outputs)
§ "Outputs"), a whole-cloud render
(`render_patch_cloud_bitmaps`) gives such a point a zero row and `-1`, and
`sfm embed-patches` drops it as it drops any point with no bitmap. How many
points that is, against the 3% that store a fused mean today on
`dino_dog_toy` through `without_any` alone, is counted during the tuning. A point with fewer than two
views keeps getting no bitmap, as today.

**A fused mean already in a file** stays readable. A file older than `.sfmr`
version 12, or one written before this change, can hold a fused mean beside
`-1`; readers treat it as they treat any bitmap beside `-1`, and the next
render replaces it with the rule's pick (on the bench, at the first evaluation
that picks a row, which is now every evaluation where some row is eligible).

### What still reads the fused mean

- **Normal refinement's weighted consensus** Φ stays: it chooses a normal by
  scoring the views against their own weighted mean, not against the stored
  bitmap. Its `representative` output
  is the pick's tile; where no view is eligible, `representative` and
  `reference` are `None` in place of the fused mean of the retained render.
- **The localizer and the sub-pixel refiner** align to the fused mean today
  where the rule picks no view it would store
  ([patch-keypoint-localization.md](../core/patch/patch-keypoint-localization.md)
  § "The template: the reference render", step 3 of *Which reference*). That
  step goes: where no view is eligible
  there is no reference, and every view keeps its starting keypoint, unscored,
  which is what both already do where nothing renders to align to.
  `robust_iters`, which these kernels' parameter tables describe as the fused
  mean's IRLS passes, is checked for any other use and removed from their
  parameters, and from the `localize-keypoints` and `refine-keypoints` command
  specs, where it has none.
- **Add Image to Tracks** uses the fused mean of the references' renders as
  its template (`TemplateKind::FusedMean`, `"fused_mean"` in Python) where the
  point has no reference and the rule picks none it would store. That template
  goes: where no reference is eligible, the point is not extended, with a new
  skip reason beside `too_few_references` (proposed: `no_reference_view`).
- **The bench's bitmap for judging** is the rule's pick among every row with a
  keypoint, or their fused mean today; it becomes the pick, or no bitmap where
  no row is eligible.
- `fuse_patch_bitmap`, `fuse_patch_bitmap_reporting`,
  `BitmapKind::FusedMean` and `PatchViewStack::fuse` (whose only callers are
  normal refinement's and the sub-pixel refiner's fused-mean bitmaps, not Φ)
  lose every caller outside their tests and are removed, unless the implementation keeps `fuse_patch_bitmap` for the
  comparisons in [sharper-patch-bitmap.md](sharper-patch-bitmap.md#evaluation).

## What changes elsewhere

These are proposals; the names follow [GLOSSARY.md](../GLOSSARY.md) (its
`*Options` entry under § "Rust type names" for the settings struct) and are settled in its reference-view
entry when the change is built.

**Core API**
([`reference_view.rs`](../../crates/sfmtool-core/src/patch/reference_view.rs)):

```rust
pub struct ReferenceViewOptions {   // impl Default, the starting values above
    pub normal_error_deg: f64,      // σ_n, 3
    pub agreement_band: f64,        // 0.15
    pub agreement_scale: f64,       // 0.15
    pub cell_deficit_band: f64,     // 0.3
    pub cell_deficit_scale: f64,    // 0.3
    pub coverage_band: f64,         // 0.99, the coverage C_v is 1 at or above
    pub coverage_scale: f64,        // 0.03
    pub clipped_band: f64,          // 0.05
    pub clipped_scale: f64,         // 0.05
    pub min_coverage: f64,          // 0.9, the hard limit
    pub unread_radius: f64,         // 1, r where no view has a radius
}
pub enum ReferenceTerm { Sharpness, Angle, Agreement, Cells, Coverage, Clipped }
pub enum ReferenceExclusion { NoTile, Coverage, OtherSide }
pub enum PatchSide { Front, Back }   // Back: angles measured against −n
pub struct ReferenceCost {
    pub radius: f64, pub radius_read: bool, // r, and whether it was the view's own
    pub angle_term: f64,                    // c·tanθ, grid px
    pub agreement: f64, pub cells: f64,     // the four factors
    pub coverage: f64, pub clipped: f64,
    pub cost: f64,
}
pub struct ReferenceChoice {
    pub reference: Option<usize>,
    pub side: Option<PatchSide>,
    pub costs: Vec<Result<ReferenceCost, ReferenceExclusion>>,
}
pub fn choose_reference_view(views: &[ReferenceReadings], resolution: usize,
    options: &ReferenceViewOptions) -> ReferenceChoice;
```

- `ReferenceTest`, `ReferenceFallback`, `ReferenceChoice::fallback` and
  `ReferenceChoice::rejected_by` go. `REFERENCE_MAX_VIEWING_ANGLE_DEG` goes,
  replaced by the angle term, and `REFERENCE_FACING_LIMIT_DEG` by the choice
  of side. The constants `REFERENCE_MIN_COVERAGE`,
  `REFERENCE_MAX_CLIPPED_SHARE`, `REFERENCE_MAX_CELL_DEFICIT` and
  `REFERENCE_AGREEMENT_MARGIN` become the bands of `ReferenceViewOptions`,
  so the tuning can sweep them without rebuilding. `REFERENCE_MIN_JUDGED_CELL_ZNCC` and
  `REFERENCE_MIN_CELL_SAMPLES` are part of how the cell deficit is read, not of
  the rule, and stay.
- The rule needs `R` for `c`, so it takes the resolution; `read_track` passes
  the tiles' own.
- `ReferenceStanding` carries, per view, its `ReferenceCost` or its
  `ReferenceExclusion`, and the **costliest term**: the term that adds most to
  its cost over the pick's. `ln cost` splits exactly into
  `ln r + ln(r_eff / r) + ln A + ln K + ln C + ln P`, so the difference from
  the pick splits into one part per term (`Sharpness`, `Angle`, `Agreement`,
  `Cells`, `Coverage`, `Clipped`), and the costliest term is the one with the
  largest part. It takes the place of `rejected_by`, the first test that turned
  the view away.
- `ReferenceRender::stored_reference` and `stored_view` go
  (§ "The stored bitmap"), and `PatchBitmap::reference` is `Some` wherever
  there is a bitmap.

**The wire** ([mcp-server.md](../gui/mcp-server.md#the-bench-family)): each
`in` row's `reference_view` becomes `{"is_reference", "cost", "cost_ratio",
"terms", "costliest_term", "excluded_by", "side"}`, with `cost_ratio` the
row's cost over the pick's, `terms` an object of the radius, the angle term
and the four factors, `costliest_term` one of `"sharpness"`, `"angle"`,
`"agreement"`, `"cells"`, `"coverage"`, `"clipped"` or null for the pick,
`excluded_by` one of `"no_tile"`, `"coverage"`, `"other_side"` or null, and
`side` `"front"` or `"back"`. `rejected_by` and `fallback` go.
`reference_view_observation` stays and is null only where no row is eligible.

**The Python bindings** carry the same dict on each observation's `"track"`
dict ([editable-track.md](../core/bench/editable-track.md#python-bindings)).

**Track View's *Reference* column**
([track-view.md](../gui/track-view.md#the-observation-table)): the marks for
the reference and the pick stay (green, red, grey). Every other row's first
line is its cost over the pick's and the word for its costliest term, with
today's words (`less sharp`, `oblique`, `agrees less`, `ninth differs`,
`partial`, `clipped`), for example `×1.25 oblique`; an excluded row reads
`partial` for the coverage floor, `no tile` or `other side`. The second line
stays the viewing angle and the pair ZNCC. The hover lists the radius (and
whether it was the row's own), the angle term, each factor beside its band
and scale, and the side. The *No reference* case for a fused mean and
`ReferenceRows::pick_by_last_fallback` go; `ReferenceMark::ReferenceWithoutPick`
stays for a track where no row is eligible.

**The bench** ([editable-track.md](../core/bench/editable-track.md#the-stored-bitmaps-reference)):
a bitmap that names no row follows the rule's pick at the first evaluation that
picks a row, with no exception for the last fallback.

## How the values will be set

The tuned parameters are `σ_n`, the band and scale of each of the four
penalties and the unread radius; the coverage floor of `0.9` is fixed. They
are set the way today's thresholds were
([reference-view.md](../core/patch/reference-view.md#why-the-agreement-is-read-plain)):

- **Against the maintainer's hand picks** of the reference view on 77 tracks
  from ten datasets, split in half by dataset with a seeded split, the values
  chosen on one half and reported on the held-out half. A setting is scored by
  the **lenient** count (the pick within 0.1 px of semi-major axis and 0.05 of
  pair ZNCC of the hand pick) and then the **exact** count (the hand pick
  itself), taking the middle of a run of tied settings. The lenient bounds
  read no angle, so a pick at 55° can count as matching a hand pick at 20°;
  a lenient count that also bounds the difference in viewing angle is
  reported beside it.
- **Against today's rule** on the 661 pool tracks: how many picks change, and
  which term the changed picks turn on.
- **Localization on the seoul_bull and kerry_park ground truths**: for the
  tracks whose pick changes, the error of the other views' localized keypoints
  against the ground truth, as the 65° limit was judged.
- **Counts** of the points that change from a fused mean to a pick, of the
  points left with no eligible view, and of the tracks with views on both
  sides of the patch, on a solve of `dino_dog_toy` and on the ten datasets.

`σ_n` is swept from `0` (no angle term) upward, the bands around today's
thresholds, and each scale over a range from a tenth of its band to several
times it. The agreement term is measured on plain and blur-matched pair ZNCC.
The values chosen, and what they were measured on, go into the standing spec's
table of parameters when the change is filed.

## Cost

The rule reads the same readings it reads today; the cost is a few operations
per view, against the renders, self-similarity readings and member coherence
matrix the readings already cost. Nothing is rendered or correlated in
addition.

## Testing

On constructed readings: the case of Problem 1 picking the 20° view; each
penalty `1` inside its band and growing at its scale past it; a view at the
coverage floor eligible and one just under it excluded; a track whose every
view is behind the patch picking by `θ'`, and a track with views on both sides
picking among the facing ones; each kind of missing reading, including a track
where no view has a radius; a view at exactly 90°; the tie breaks; and the
costliest term of each view summing with the others to its cost over the
pick's. On the stored bitmap: that a point whose every view is behind the
patch, and one whose best view covers between 0.9 and 0.99 of its tile, store
a view's tile and name it, and that a point with no view at 0.9 coverage gets
no bitmap and `-1`.

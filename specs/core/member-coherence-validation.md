# Member-coherence validation: pairwise track agreement and the max-support block

**Status:** Implemented (2026-08-04) — kernel in
`crates/sfmtool-core/src/patch/member_coherence.rs`, exposed to Python as
`PatchCloud.validate_member_coherence`.

## Overview

A track's members are the observations of one 3D point. Some tracks are
**chimeras**: their members image two different surfaces. Scoring each member
against the fused cross-view consensus does not find them — the consensus is
built *from* the members, so a balanced split makes it a compromise blend that
every member scores well against, and the more even the split the better the
blend flatters both sides.

The disagreement is only visible **between** members. This module builds the
`k×k` matrix of pairwise agreement between a point's members and reads a verdict
off it: keep the track whole, keep a majority block and reject the rest, or
retire the point because its evidence supports two incompatible surfaces with
neither prevailing.

The matrix is a diagnostic in its own right — it says *which* members go
together, not just *that* something is wrong — so building it and deciding on it
are separate entry points.

## The matrix

For a point with `k` members (image indices, deduplicated first-seen-wins), entry
`(i, j)` is the windowed ZNCC between member `i`'s and member `j`'s patch, each
sampled from its own source image through the point's own patch frame, **anchored
at that member's stored keypoint** (below).

Sampling is **identical to [view selection](patch-view-selection.md)'s** — the
same `build_level_context` / `normalized_stack` / z-normalize path that builds the
reference appearance a candidate view is admitted against:

- The patch is placed at the point's own normal on an `R×R` grid.
- Each member is gated on its window-weighted valid-pixel fraction
  (`min_valid_fraction`); survivors' validity masks are **intersected** into one
  frozen common support, and every pair is correlated over that same support. Its
  size is reported as `n_support` — one number per point, because the support is
  frozen per *point*, not per pair.
- A member with no texture on that support at all is dropped from the stack
  (below).
- Each member's render is z-normalized per colour channel over the windowed
  support, with `√w` folded in, so a plain dot product is the windowed ZNCC. A
  channel that is flat in *any* remaining member is dropped for *every* member,
  keeping all the inner products in one space.
- A pair's score is the mean of its per-channel correlations.

Sharing that path keeps a member's pairwise agreement in the same photometric
*space* as the member-vs-consensus score the rest of the patch pipeline gates on:
same window, same sampler, same channel treatment, same frozen-support discipline,
and they cannot drift apart as the render conventions evolve. It is **not the same
estimator** — selection scores one view against the fused consensus, and does it
through the [affine fast path](patch-view-selection.md#affine-candidate-scoring-2026-07)
wherever that path's residual and border gates allow. The two are the same metric
family and agree to that path's documented tolerance; they are not the identical
number, and a rule calibrated on one should not be assumed transferable to the
other without a check.

The diagonal is `1.0`. A member the validity gate drops is left **unscored** — its
row and column are `NaN`, reported as a per-member `scored` flag (which is exactly
"has at least one finite off-diagonal entry", the same definition the decision rule
uses). A track whose members cannot form a common support at all yields a matrix
with only its diagonal, which decides as `KeepAll` (no evidence is not contrary
evidence).

### The same agreement, measured again on coarser grids

Alongside the `R×R` table, the same `k×k` agreement is measured on **box-downsampled
copies of the very same renders**, one table per factor in `{2, 4}` that divides `R`
and leaves a grid of at least `4×4` — at the default `R = 24`, the `12×12` and `6×6`
grids. They are reported as `zncc_coarse`, coarsest last, with their factors.

Nothing is re-sampled. Each coarse cell is the plain mean of the frozen common
support's pixels inside it, so every scale reads the same pixels through a wider
aperture; the window is recomputed on the coarse grid and the pair scores are then
formed by the identical estimator (per-channel z-normalization, unit-norm dot
product). A cell exists when at least one support pixel lands in it and its
recomputed window weight is positive. Because the support is common to every
member, the surviving cells are the same for all of them, so a member unscored at
full scale is unscored at every scale and the tables are index-compatible with
`zncc`.

These tables exist because a **single** scale cannot say *why* two members
disagree. Two members of the same surface, one of them soft, and two members of
different surfaces produce the same number at full resolution. They stop looking
alike the moment the fine detail is removed, and that is what the decision rule and
the per-member sharpness below both read.

### Members are sampled at their keypoints, not at the reprojection

Each member's render is **recentred in-plane so it is anchored at that member's
stored keypoint** — the sub-pixel location the feature actually occupies in that
image — rather than at the pixel the point's current geometry reprojects to. The
same anchors build the per-member validity mask, so the frozen common support is
the intersection of where the members are *sampled*.

The two anchors differ by the member's reprojection residual, and that is a
**geometric** quantity: it says the position, the pose or the intrinsics do not yet
explain this observation. Sampling at the reprojection carries that error into the
render — the member's window slides off its own content by the residual — and every
pairwise ZNCC the member takes part in is deflated by the misalignment rather than
by disagreement about what is being imaged. The matrix exists to answer "do these
members image the same surface", and it must not answer "does the current solve
already fit them", which the reprojection cull and the bundle adjustment already
own. A residual that is an appreciable fraction of the patch half-width costs
several tenths of ZNCC: a member with a 0.92 px residual against a 7.9 px
half-width (≈ 12%) scored 0.28–0.73 against its siblings at its projection while
imaging exactly the content they did.

The anchor is per member and optional. A member with no stored keypoint — a
`sift_files` reconstruction (feature indexes, no inline keypoints), a
`CameraViews` scene (no tracks at all), a hand-supplied member list naming an image
the point does not observe — falls back to projection anchoring individually, and a
caller can turn the whole thing off.

**`bar` is calibrated per anchoring.** Keypoint anchoring raises scores for exactly
the members whose residual was deflating them, so the score distribution shifts up
and a threshold picked against projection-anchored numbers is effectively looser
against these. The two are not the same measurement, and a bar carried across
without a re-check is a bar whose operating point has moved.

### A textureless member does not sink the track

The shared z-normalization drops a colour channel that is flat in *any* member, for
*every* member — right for a consensus over one surface, wrong here: a single blown
highlight or patch of sky flattens every channel and silently leaves the whole
track unscored. Such a member is therefore excluded from the stack **before** the
shared gate runs, by the same `FLAT_NORM_SQ_EPS` criterion, and comes out
`scored = false` — the same outcome as failing the coverage gate, for the same
reason (nothing about it can be correlated). The rest of the members score
normally, and the channels the flat member would have killed survive for them.

This is done locally in this module rather than in the shared helpers, which keep
their behaviour for view selection and normal refinement.

### The support floor, and what the matrix is not

`min_support_pixels` is a floor on `n_support`: below it the track is left entirely
unscored (fail-open — `KeepAll`), with the count still reported. It defaults to `8`,
the floor the shared support builder already enforces, so it changes nothing on its
own. **Consumers vetting wide-baseline tracks want it higher**: the intersection of
`k` validity masks can shrink to a sliver of the `R×R` grid, and a ZNCC over a
handful of pixels is noise being thresholded at three decimal places.

That intersection is also why **matrices built over different member subsets are
not comparable**. Entry `(i, j)` is not a property of members `i` and `j`: it is
computed over the support they share with *every other member of the list*. Add a
member with a narrow view and every entry changes; drop one and they change back.
Two consequences worth stating plainly: a verdict cannot be checked by re-running
the rule on a subset of the members, and the bar's effective strictness is
track-size dependent — a larger track correlates over a smaller, more central
support, where members agree more.

## The decision rule

**The rule runs over the scored members only.** An unscored member carries no
pairwise evidence at all, so it is not a hypothesis, not a tie-break candidate, not
a term in either margin side, and not a unit in the majority denominator; it is
also never in the reported `block`. It simply passes through `kept`. Anything else
double-counts an absence: letting it dilute the denominator retires tracks whose
scored members cleanly split, and letting a `Split` evict it removes an observation
on zero evidence. Fewer than two scored members means no evidence at all —
`KeepAll`, empty block, `support = 0`, undefined margin. With every member scored
this is the plain rule below, unchanged.

Every scored member is a hypothesis. Its **support** is the set of scored members
whose pairwise ZNCC to it reaches the admission bar, itself always included; a
member with no partner supports a block of one. The **max-support block** wins.
Note this is an inlier set, not a clique: two members of the winning block need not
agree with each other, only with the hypothesis.

The admission bar the sweep runs at is `effective_bar`, and the margin floor is
`effective_margin_gate` — the absolute `bar` / `margin_gate` pair, tightened by
the track's own coherence (see [below](#the-self-normalized-admission-bar)) and
reported per point alongside the `core_center` / `core_scatter` they came from.
With `self_bar_k = 0` they *are* the absolute pair, and the rule below is the
absolute rule.

Given the winning block `B` of size `s` out of the `m` **scored** members (of `k`
total):

1. **`s == m`** — every scored member is in the block. Verdict `KeepAll`.
2. **Separation margin.** Let `min_intra` be the weakest finite link inside `B`
   and `max_cross` the strongest finite link from `B` to a scored member outside
   it; the margin is `min_intra − max_cross`. When the margin is undefined
   (`s < 2`, `s == m`, or a side with no finite link) or does not exceed
   `effective_margin_gate`, verdict `KeepAll`. `NaN` therefore means *no cut was
   on the table* — a different thing from a cut the gate refused, which reports a
   finite margin at or below the gate.
3. **`2s > m`** — the block is a strict majority of the scored members. Verdict
   `Split`: the block is kept (plus the unscored members), the scored members
   outside it are rejected.
4. **Otherwise** — verdict `Retire`. The point ships nothing — including its
   unscored members, since it is the point that is refused, not its observations.

### The self-normalized admission bar

`bar` and `margin_gate` are absolute, and an absolute threshold can only be
calibrated against one kind of disagreement. A member imaging a **different**
surface scores 0.2–0.5 against the rest of the track, and `0.65` separates it. A
member imaging an **occluder in front of the same repeating texture** does not:
it shares the core's dominant structure and scores 0.85–0.95 — against a core
whose members agree with each other at 0.98–1.00. The block structure is real,
and it sits entirely above the bar.

So both thresholds are re-derived per track, from **that track's own coherence**,
in two passes over the same matrix:

1. Sweep the max-support block at the absolute `bar`.
2. Measure that block's **core coherence**: the centre `c` and scatter `σ` of its
   intra-block links (below).
3. `effective_bar = max(bar, min(c − self_bar_k · σ, 0.99))` and
   `effective_margin_gate = min(margin_gate, σ)`.
4. Re-sweep the block at `effective_bar` — only when it actually rose — and run
   the margin and majority tests on **that** block, against
   `effective_margin_gate`.

A tightly-coherent track therefore demands tight agreement of a newcomer; a noisy
or drifting one has a large `σ`, the relative term falls below the absolute floor,
and the rule is the absolute rule.

The margin floor moves with the bar because it is the same problem one level
down. A margin is a *difference* of two ZNCCs, and the noise on that difference is
the core's own pair-to-pair scatter. A tight core separates from an occluder by
0.02–0.04 — a real gap in its own units, and one an absolute `0.05` refuses
outright, which is why tightening the bar alone changes almost nothing on this
family. Both terms relax back to the absolute pair together, exactly when `σ` is
large, which is what a drift chain and a genuinely noisy track have in common.

#### Centre and scatter

The sample is every finite pairwise link **inside** the pass-1 block, each
counted once.

- **Centre** `c` is its median.
- **Scatter** `σ` is the **upper** semi-interquartile distance, made
  normal-consistent the way a MAD is: `1.4826 · (Q₇₅ − median)`, floored at
  `0.005`.

The one-sidedness is the point. The pass-1 block is the very thing under
suspicion — on a track with an occluding member it still contains it, and that
member's links sit in the **lower** tail. A two-sided MAD reads that tail as
spread and inflates `σ`, handing the members the bar is meant to exclude the
power to loosen it. The half above the median is the part of the sample the
contamination cannot reach (it is a minority, or the block would not be the
core). For a symmetric sample the two coincide.

It is read as an order statistic rather than as the median of the links above the
centre, because those differ exactly when the sample has a **mass at the
median** — a two-population matrix whose median lands on the lower mode. Counting
the ties would report such a matrix as tightly coherent; the quartile distance
reports the spread that is really there, and the relative term collapses, which is
the right answer for a track with no single core.

The floor on `σ` is what a perfectly uniform block needs: its measured spread is
zero, which would put the bar on the centre itself and the margin floor at zero.
The `0.99` ceiling is the other end of the same guard — a perfect core cannot
demand perfection of a newcomer.

#### One pass, and the circularity

The circularity is real: admission defines the block whose statistics set the
admission bar. It is **cut, not solved**. Pass 1 runs at the loose absolute bar so
the block is the widest defensible one; the scatter estimator is one-sided so the
suspects cannot inflate the scale; and the tightening runs **once**. It is
deliberately not iterated to a fixed point — each further pass would shrink the
block, re-measure a tighter core off the survivors and shrink it again,
converging on the tightest sub-clique of every track whether or not anything is
wrong with it.

#### Small blocks, and what `self_bar_k` costs

Below **six** intra-block links — a block of three or fewer members — the centre
and its quartile distance are being read off two or three numbers, which measures
nothing; the relative term stays inactive and the absolute thresholds decide.
`self_bar_k = 0` disables it everywhere, reproducing the absolute rule exactly.

`self_bar_k` **trades occlusion recall against collateral**, and the trade is not
avoidable inside a single-scale measurement. Every member that trails a tight core
for an innocent reason — motion blur, an exposure step, a grazing view — is a member
the tightened bar is also more willing to evict, and nothing in the pairwise matrix
*at one scale* distinguishes it from an occluder. Lower `self_bar_k` catches more of
both. [Multi-scale exoneration](#multi-scale-exoneration) is what refunds the
innocents, by measuring the same agreement again with the detail taken away.

### Multi-scale exoneration

The self-normalized bar cannot avoid taking innocents with the occluders, because
at one scale it cannot see the difference. Removing the fine detail is what makes
the difference visible:

- **Structural** disagreement — an occluder, a different surface — is present at
  every scale. The member's low frequencies are *already* the wrong content, so
  coarsening both sides changes nothing about how badly they agree.
- **Spectral** disagreement — a defocused or motion-blurred frame of the *same*
  surface — lives entirely in the detail. Coarsen both sides and it evaporates: the
  member's low frequencies are the core's low frequencies.

So for each member the relative term alone would evict, the **agreement deficit** is
measured twice, over the tables the matrix already carries:

```
deficit(scale) = mean(core↔core at scale) − mean(member↔core at scale)
```

where the core is the winning block **minus the member itself** (so the quantity
means the same thing for a member inside the block and one outside it), and both
means run over finite links only. Their quotient is the member's **retained
deficit**. A member whose retained deficit is at or below `exoneration_ratio` is
**exonerated**: it is kept, it stays out of `block`, and it is reported in
`exonerated` alongside its ratio.

The comparison scale is the **first** coarse table — one halving — not the coarsest.
The test is whether the disagreement *survives* removing detail, and a grid coarse
enough washes out structure along with blur. Measured against the two labelled
populations, one halving separates them; two halvings collapses the occluders into
the same range as the soft frames and the separation is gone. That is also why the
threshold sits high: at `0.90` it is asking "did *anything* decay", not "did most of
it".

#### Only the relative term's evictions are exonerable

A member the **absolute** `bar` rejects is not a soft frame of the track's surface.
It correlates 0.2–0.5 — the cross-surface chimera the absolute rule was calibrated
on — and how its disagreement is distributed across scales says nothing about
whether it belongs. Blur is not a defence against imaging a different thing.
Exoneration therefore never loosens the absolute rule; it only refunds what
tightening the bar per track took. A member outside the pass-1 block is not flagged,
gets no ratio computed, and cannot be spared.

#### Where it sits in the rule

Exoneration runs **after** the margin gate and **before** the majority test, and it
re-admits individual members rather than re-running the sweep:

- `block`, `support`, `margin`, `min_intra` and `max_cross` all keep describing the
  cut the sweep proposed. An exonerated member is a *spared* member, not a different
  block, and the reported quantities say so.
- When every rejected member is spared there is no cut left and the verdict falls
  back to `KeepAll`.
- The majority test counts the spared members on the kept side — they ship, so they
  are part of the side that would prevail. A `Retire` whose block regains a majority
  this way becomes a `Split`.

`exoneration_ratio = 0` disables it. It is also inert whenever the relative term is
(nothing is ever flagged), and whenever the matrix carries no coarse scale.

### Per-member sharpness

The same two deficits give a quantity that is **not** about any verdict:

```
sharpness_deficit = deficit(full) − deficit(coarsest)
```

the part of a member's disagreement that exists only at fine scale. It is `0` for a
member whose disagreement (if any) is scale-free — including an occluder, which is
sharp, just wrong — and grows for a member that agrees with its core coarsely and
not finely, which is what defocus and motion blur do.

It is computed for **every** scored member, flagged or not, because it describes the
observations the point *ships* rather than the ones it was thinking about evicting.
It reads the **coarsest** table, not the first: this is a magnitude rather than a
survival test, and the widest span available is the most sensitive one. Measured
across one capture's frames, the separation is an order of magnitude — a visibly
soft frame's members sit above the crisp frames' 99th percentile.

`NaN` where the two deficits are not both defined: no coarse scale, or a block with
fewer than two other members.

### The margin gate refuses to cut a continuum

A single surfel seen across a wide baseline sweep produces a **banded** matrix:
every consecutive pair agrees, only the far ends do not. Thresholding such a
matrix always yields a block that is a strict subset, and cutting it picks an
arbitrary place along a chain — there is no second surface there to evict, only
one surface seen too obliquely at one end.

The margin distinguishes the two shapes. A chimera puts a **gap** between the two
sides: its weakest internal link still beats its strongest external one. A drift
chain does the opposite — the block's weakest internal link (the two far ends of
the band it spans) is *below* its strongest external one (the neighbour just past
the cut) — so the margin comes out at or below zero and the track is kept whole.
Step 2 runs before the majority test, so it protects `Split` and `Retire` alike.

### Verdict semantics

- `KeepAll` — every member ships. The track is one surface, one continuum, or has
  too little evidence to cut.
- `Split` — the block's members ship, as do any unscored ones; the rest are
  rejected observations of some other surface. The point survives on the block.
- `Retire` — the point should not ship at all. A block with no majority means the
  track's members split into two comparably-supported and mutually incompatible
  groups, and nothing in the matrix says which one is the point. The winning block
  is still reported, but on a balanced split the two sides are **interchangeable**
  and only the tie-break decides which is named: the block is
  **informational** for a retirement, not a recommendation.

### Determinism

The block choice is fully determined by the matrix, with no dependence on
iteration or thread order:

1. Highest support count.
2. Ties broken on the block's own **mean coherence** — the mean of its finite
   intra-block links (`-1` when it has none).
3. Remaining ties broken on the **lowest member index**.

The batch entry is per point, so results are returned in cloud order regardless of
how rayon schedules them.

## Parameters

| parameter | default | meaning |
|---|---|---|
| `bar` | `0.65` | pairwise ZNCC at or above which two members agree — a **floor**, raised per track by `self_bar_k` |
| `margin_gate` | `0.05` | separation margin a cut must exceed; below it the track is kept whole. A **ceiling**, lowered per track by `self_bar_k` |
| `self_bar_k` | `1.5` | units of the track's own core scatter the effective bar sits below its core centre; `0` disables the relative term |
| `exoneration_ratio` | `0.90` | retained deficit at or below which a relative-flagged member is spared; `0` disables exoneration |
| `resolution` (R) | `24` | patch grid members are rendered and correlated on |
| `window` | `gaussian_disk` (σ 0.6) | per-pixel scoring weight |
| `sampler` | `bilinear_mip` | source-pyramid sampling |
| `min_valid_fraction` | `0.6` | per-member floor on the window-weighted valid-pixel fraction |
| `min_support_pixels` | `8` | floor on the common support `n_support`; below it the track is unscored |

`bar` and `margin_gate` are **calibration defaults, not constants** — callers
override them. `bar` in particular is calibrated *for the render conventions in
the same table*: it is a threshold on a ZNCC whose value depends on the window,
the sampler, the channel treatment, the support and the **anchoring**, so a caller
that changes `resolution`, `window`, `sampler` or `keypoint_anchor` must re-pick
it.

The rule was calibrated on a grayscale, hard-disk, per-pair-joint-support
prototype at `bar = 0.60`. This implementation instead scores per-channel RGB over
a Gaussian-disk-weighted **common** support — the price of sharing view
selection's sampler — and the two metrics are not the same number. Measured over
247k member pairs on three reconstructions, they agree closely in rank
(Spearman 0.96–0.97) but the native metric **compresses** the range: a
least-squares fit gives `native ≈ 0.90·offline + 0.09` (per-dataset slopes
0.88–1.02), so pairs in the bulk near 0.95 come out ~0.005 lower while pairs down
near the threshold come out *higher* — the fit puts offline `0.60` at native
`0.63`, and the prototype's clearest chimera pairs move up by 0.04–0.17.
Down-weighting and clipping the patch periphery is what does it: that is where a
wrong-surface member disagrees most. Carrying `0.60` over therefore cuts **half**
as many tracks as the calibration did (per-dataset `7 / 0 / 5` against the
prototype's `15 / 0 / 14`). `0.65` restores the operating point
(`11 / 0 / 14`), reproduces the validated exemplar verdicts, and leaves the
human-approved tracks clear of the threshold by 0.07–0.28 on their weakest pair —
where `0.70` would squeeze the tightest of them to 0.02 and start cutting the
flat-tabletop set that must not be cut. `margin_gate` carries over unchanged: a
margin is a *difference* of ZNCCs, which the ~0.9 slope rescales by less than the
gate's own precision.

`self_bar_k` is calibrated against the occluding-member family the absolute pair
structurally cannot reach — members on a repeating-texture surface that correlate
0.85–0.95 with a core agreeing at 0.98–1.00. Measured against nine hand-verified
exemplar tracks, `1.5` is the **largest** value that still cuts all of the ones
this rule can reach; past ≈1.9 the bar loosens enough to re-admit an intermediate
member on the tightest of them, which collapses the margin and the cut is lost.
Below `1.5` nothing further is caught and the collateral stops falling (the split
count is flat across `1.0–1.5`), so the operating point is an elbow rather than a
slope. Two exemplars are **outside** the rule's reach at any `self_bar_k` and are
the documented limit:

- a track whose *core* is itself bimodal (two sub-groups agreeing at 0.96, so
  σ ≈ 0.048) with the occluder 0.04 below it — under one scatter unit, which is
  exactly the "no single core, keep the absolute bar" case; and
- a track with an occluder at 0.93–0.98 against a core at ~1.00, where an
  intermediate member bridges the two and the margin is **negative** at every
  admission bar. Reaching it needs a discriminator this matrix does not
  contain — a geometric one — not a lower `self_bar_k`. Trying anyway is
  expensive: across two reconstructions, `1.5` already evicts 405 members whose
  strongest surviving link is in `0.94–0.97`, and any bar high enough to reach
  this track adds the 261 above `0.97` on top.

## API

```rust
pub struct MemberCoherenceParams {
    pub bar: f64,                 // 0.65
    pub margin_gate: f64,         // 0.05
    pub resolution: u32,          // 24
    pub window: PatchWindow,      // GaussianDisk { sigma: 0.6 }
    pub sampler: Sampler,         // BilinearMip
    pub min_valid_fraction: f64,  // 0.6
    pub min_support_pixels: u32,  // 8
    pub self_bar_k: f64,          // 1.5; 0 disables the relative term
    pub exoneration_ratio: f64,   // 0.90; 0 disables exoneration
}

pub const SELF_BAR_CEILING: f64 = 0.99;      // cap on the effective bar
pub const SELF_BAR_MIN_SCATTER: f64 = 0.005; // floor on the core scatter
pub const SELF_BAR_MIN_PAIRS: usize = 6;     // fewest intra-block links to estimate from
pub const COARSE_FACTORS: [u32; 2] = [2, 4]; // box-downsampling factors, coarsest last
pub const MIN_COARSE_RESOLUTION: u32 = 4;    // smallest coarse grid built
pub const EXONERATION_MIN_DEFICIT: f64 = 0.01; // smallest deficit a ratio is taken from

pub enum MemberVerdict { KeepAll, Split, Retire }

pub struct MemberMatrix {         // members, k*k row-major zncc, per-member scored
    pub members: Vec<u32>, pub zncc: Vec<f64>, pub scored: Vec<bool>,
    pub n_support: u32,           // common-support pixels (0 for `from_zncc`)
    // The same agreement on box-downsampled copies of the same renders, one
    // k*k table per factor, coarsest last. Empty when none could be built.
    pub zncc_coarse: Vec<Vec<f64>>, pub coarse_factors: Vec<u32>,
}
pub struct MemberDecision {       // verdict + what it was decided on
    pub verdict: MemberVerdict, pub kept: Vec<bool>, pub block: Vec<bool>,
    pub support: u32, pub margin: f64, pub min_intra: f64, pub max_cross: f64,
    // The thresholds the sweep and the margin test really ran at, and the
    // statistics they came from. NaN core_* means the relative term was inactive;
    // NaN thresholds mean no sweep ran at all.
    pub effective_bar: f64, pub effective_margin_gate: f64,
    pub core_center: f64, pub core_scatter: f64,
    // Multi-scale exoneration: who the relative term alone rejected, who was
    // spared, and the ratio that decided. NaN ratio = not flagged, or undefined.
    pub relative_flagged: Vec<bool>, pub exonerated: Vec<bool>,
    pub retained_deficit: Vec<f64>,
    // Photometric sharpness relative to the track consensus, for EVERY scored
    // member: the part of its deficit that exists only at fine scale.
    pub sharpness_deficit: Vec<f64>,
}
pub struct MemberCoherence { pub matrix: MemberMatrix, pub decision: MemberDecision }

// "Scored" — at least one finite off-diagonal entry — in one place, shared by the
// matrix and the decision rule.
pub fn scored_mask(zncc: &[f64], k: usize) -> Vec<bool>;

// One block's own (centre, scatter): the statistics the self-normalized bar is
// measured in. `None` below SELF_BAR_MIN_PAIRS intra-block links.
pub fn core_coherence(zncc: &[f64], k: usize, block: &[bool]) -> Option<(f64, f64)>;

// One member's agreement deficit against a block on one scale's table, with the
// member excluded from the core on both sides. NaN when either side is undefined.
pub fn core_deficit(zncc: &[f64], s: usize, block: &[bool], member: usize) -> f64;

// The COARSE_FACTORS a given grid admits.
pub fn coarse_factors_for(resolution: u32) -> Vec<u32>;

// Matrix and decision are separate so a caller can inspect or supply either.
// `member_keypoints` is parallel to the INPUT `members` slice (deduplicated
// alongside it); `None` — for the slice or for one member — anchors at the
// projection.
pub fn member_zncc_matrix(
    patch, views, members: &[u32],
    member_keypoints: Option<&[Option<[f64; 2]>]>, params,
) -> MemberMatrix;
pub fn decide_member_coherence(matrix: &MemberMatrix, params) -> MemberDecision;
pub fn validate_member_coherence(
    patch, views, members, member_keypoints, params,
) -> MemberCoherence;

// Batch over a cloud, rayon-parallel across points, results in cloud order.
pub fn validate_patch_cloud_member_coherence(
    cloud: &PatchCloud, views: &[ProjectedImage<'_>], member_views: &[Vec<u32>],
    member_keypoints: Option<&[Vec<Option<[f64; 2]>>]>,
    params: &MemberCoherenceParams, progress: Option<&AtomicUsize>,
) -> Vec<MemberCoherence>;

pub fn member_views_from_reconstruction(recon, cloud) -> Vec<Vec<u32>>;
// The stored keypoint of each of those members, in the same order; all `None`
// for a `sift_files` reconstruction.
pub fn member_keypoints_from_reconstruction(recon, cloud) -> Vec<Vec<Option<[f64; 2]>>>;
```

`MemberMatrix::from_zncc` builds a matrix from an already-computed table, so a
caller holding its own pairwise scores can use the decision rule on its own; it
carries no coarse scale, which leaves exoneration and sharpness inactive.
`from_zncc_scales` takes the coarse tables too.

The Python binding mirrors `PatchCloud.select_views`:

```python
PatchCloud.validate_member_coherence(
    recon, images, *, bar=0.65, margin_gate=0.05, self_bar_k=1.5,
    exoneration_ratio=0.90, resolution=24,
    window="gaussian_disk", window_sigma=0.6, sampler="bilinear_mip",
    min_valid_fraction=0.6, min_support_pixels=8, point_indexes=None,
    member_views=None, keypoint_anchor=True, return_matrix=False, progress=None,
) -> list[dict]
```

`recon` is a reconstruction (member lists come from the tracks) or a
`CameraViews` (then `member_views` — a `point_index -> [image_index, ...]` map —
is required); `images` is a per-image list or a prebuilt `ImagePyramidSet`.
`keypoint_anchor` (default `True`) sources each member's stored keypoint from the
reconstruction; an overridden `member_views` entry is re-keyed against the point's
own track, so an image it does not observe is anchored at its projection.
`keypoint_anchor=False` — and a `CameraViews` scene, which has no keypoints to
source — anchors every member at its projection.
Each returned dict carries `point_index`, `members` (**uint32**, the deduplicated
member order every other per-member array follows), `verdict`
(`"keep_all"` / `"split"` / `"retire"`), `kept` / `block` / `scored` (bool),
`support`, `n_support`, `margin`, `min_intra`, `max_cross`, `effective_bar` /
`effective_margin_gate` (the thresholds the block sweep and the margin test really
ran at — `effective_bar > bar` is exactly "the relative term engaged"),
`core_center` / `core_scatter` (the statistics they were derived from, `NaN` when
the term was inactive), `relative_flagged` / `exonerated` (bool — the members the
relative term alone put outside the block, and the subset exoneration spared),
`retained_deficit` (float64, `NaN` off the flagged members and where undefined),
`sharpness_deficit` (float64, for every scored member), and — under
`return_matrix=True` — the `k×k` float64 `zncc`, the list of coarse tables
`zncc_coarse` and their `coarse_factors`.

## Testing

Sibling `tests.rs` under `patch/member_coherence/` covers the decision rule on
synthetic matrices with known block structure — all-agree; a clean 3+2 split; a
balanced 2+2 retirement; a 2+1+1 at the `2s == m` majority boundary; the block
tie-break on mean coherence and then on member index; a monotone drift chain kept
whole by the margin gate; the same matrix flipping between `KeepAll` and `Split` on
`margin_gate` alone; the `k = 3` cases (majority block splits, all-isolated keeps);
`bar` and `margin_gate` locked at their inclusive boundaries on exactly
representable values; `k ≤ 2` and empty tracks; and `bar` reshaping the block.

Unscored members have their own set: an all-`NaN` member left outside the rule and
passed through; a split that evicts the scored outlier and keeps the unscored
member; two unscored members that do *not* dilute a clean 2-of-3 majority into a
retirement; a track with no pairwise evidence at all; and a single unscoreable
*pair* whose two members are both still in play (a missing entry is skipped by the
margin, not read as disagreement).

The self-normalized bar has its own set: parity at `self_bar_k = 0` across every
branch of the rule (the absolute verdicts and the absolute thresholds, with no
statistics reported); the relative term engaging on a tight core plus a 0.90
outsider that the absolute rule keeps whole, with the bar checked to be exactly
`centre − k·scatter` and the margin floor checked to have moved with it; a
two-population matrix whose scatter is wide, where verdict, membership *and* both
thresholds fall back to the absolute rule; a monotone drift chain untouched under
both settings; the `SELF_BAR_MIN_PAIRS` boundary either side (a 3-member block
inactive, a 4-member one estimating); the ceiling and the scatter floor on a
perfectly uniform core; the one-sidedness (the statistics do not move as the
contaminated members are pushed arbitrarily far below the centre, where a plain
standard deviation quadruples) and `core_coherence` returning `None` for a block
of one; and determinism plus the one-pass property — the block landed on is the
one a *single* re-sweep at the tightened bar gives.

End-to-end builds over the rendered synthetic plane scene cover matrix symmetry and
unit diagonal, one odd member out splitting, a single surface kept whole, a
balanced two-surface track retiring, member deduplication, a textureless member
left unscored while the rest score against each other, and `n_support` being
reported and gated on by `min_support_pixels`.

Anchoring has three of its own: keypoints handed in *at* the reprojections
reproduce the unanchored matrix entry for entry (anchoring is a strict
generalization, not a second render path); a member whose pose carries a ~3 px
lateral error against an image that does not — the reprojection-residual case —
is depressed at its projection and recovers by more than 0.1 ZNCC against every
sibling when anchored at its keypoint, with the track's weakest link (what `bar`
reads) moving up with it; and a duplicated member keeping the keypoint of its
first occurrence through the dedup.

Multi-scale exoneration has its own set, driven by a second hand-built table
standing in for the half-scale measurement so the retained deficit can be dialled
directly: a structural outsider (deficit unchanged across the halving) still
evicted; a spectral one (deficit evaporated) spared, with the verdict falling back
to `KeepAll` and `block` still reporting the sweep's own cut; the verdict flipping
on `exoneration_ratio` alone, inclusive at the threshold; the three inert
configurations agreeing (the knob at zero — which still *reports* the ratio, only
declining to act on it — a matrix with no coarse scale, and the relative term
disabled); a member the **absolute** bar rejects never flagged and never spared
however far its deficit decays; a `Retire` becoming a `Split` when sparing restores
the majority, with `support` and `block` still describing the sweep's block; and a
deficit under `EXONERATION_MIN_DEFICIT` yielding no ratio and no sparing. Sharpness
is covered separately, because it is reported on every scored member whatever the
verdict, and reports `NaN` rather than zero with no coarse scale.

The coarse scales themselves are covered end-to-end on the rendered scene — built
from the same stack, symmetric, unit diagonal, over exactly the factors the
resolution admits, with scoredness identical at every scale and a genuine
cross-surface pair staying weak through the halving — plus `coarse_factors_for`
against divisibility and the minimum-grid floor.

`tests/patch/test_member_coherence.py` covers the binding surface against a real
reconstruction: the dict keys and dtypes, the `k×k` `zncc` under `return_matrix`
(symmetry and unit diagonal), `point_indexes` subsetting, `member_views` override
and its first-seen-wins dedup, the `CameraViews`-without-`member_views` error, and
the unscored-member-kept contract end to end.

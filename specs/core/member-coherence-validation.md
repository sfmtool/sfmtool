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
whose pairwise ZNCC to it reaches `bar`, itself always included; a member with no
partner supports a block of one. The **max-support block** wins. Note this is an
inlier set, not a clique: two members of the winning block need not agree with each
other, only with the hypothesis.

Given the winning block `B` of size `s` out of the `m` **scored** members (of `k`
total):

1. **`s == m`** — every scored member is in the block. Verdict `KeepAll`.
2. **Separation margin.** Let `min_intra` be the weakest finite link inside `B`
   and `max_cross` the strongest finite link from `B` to a scored member outside
   it; the margin is `min_intra − max_cross`. When the margin is undefined
   (`s < 2`, `s == m`, or a side with no finite link) or does not exceed
   `margin_gate`, verdict `KeepAll`. `NaN` therefore means *no cut was on the
   table* — a different thing from a cut the gate refused, which reports a finite
   margin at or below `margin_gate`.
3. **`2s > m`** — the block is a strict majority of the scored members. Verdict
   `Split`: the block is kept (plus the unscored members), the scored members
   outside it are rejected.
4. **Otherwise** — verdict `Retire`. The point ships nothing — including its
   unscored members, since it is the point that is refused, not its observations.

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
| `bar` | `0.65` | pairwise ZNCC at or above which two members agree |
| `margin_gate` | `0.05` | separation margin a cut must exceed; below it the track is kept whole |
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
}

pub enum MemberVerdict { KeepAll, Split, Retire }

pub struct MemberMatrix {         // members, k*k row-major zncc, per-member scored
    pub members: Vec<u32>, pub zncc: Vec<f64>, pub scored: Vec<bool>,
    pub n_support: u32,           // common-support pixels (0 for `from_zncc`)
}
pub struct MemberDecision {       // verdict + what it was decided on
    pub verdict: MemberVerdict, pub kept: Vec<bool>, pub block: Vec<bool>,
    pub support: u32, pub margin: f64, pub min_intra: f64, pub max_cross: f64,
}
pub struct MemberCoherence { pub matrix: MemberMatrix, pub decision: MemberDecision }

// "Scored" — at least one finite off-diagonal entry — in one place, shared by the
// matrix and the decision rule.
pub fn scored_mask(zncc: &[f64], k: usize) -> Vec<bool>;

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
caller holding its own pairwise scores can use the decision rule on its own.

The Python binding mirrors `PatchCloud.select_views`:

```python
PatchCloud.validate_member_coherence(
    recon, images, *, bar=0.65, margin_gate=0.05, resolution=24,
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
`support`, `n_support`, `margin`, `min_intra`, `max_cross`, and — under
`return_matrix=True` — the `k×k` float64 `zncc`.

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

`tests/patch/test_member_coherence.py` covers the binding surface against a real
reconstruction: the dict keys and dtypes, the `k×k` `zncc` under `return_matrix`
(symmetry and unit diagonal), `point_indexes` subsetting, `member_views` override
and its first-seen-wins dedup, the `CameraViews`-without-`member_views` error, and
the unscored-member-kept contract end to end.

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
sampled from its own source image through the point's own patch frame.

Sampling is **identical to [view selection](patch-view-selection.md)'s** — the
same `build_level_context` / `normalized_stack` / z-normalize path that builds the
reference appearance a candidate view is admitted against:

- The patch is placed at the point's own normal on an `R×R` grid.
- Each member is gated on its window-weighted valid-pixel fraction
  (`min_valid_fraction`); survivors' validity masks are **intersected** into one
  frozen common support, and every pair is correlated over that same support.
- Each member's render is z-normalized per colour channel over the windowed
  support, with `√w` folded in, so a plain dot product is the windowed ZNCC. A
  channel that is flat in *any* member is dropped for *every* member, keeping all
  the inner products in one space.
- A pair's score is the mean of its per-channel correlations.

Sharing that path is the point: a member's pairwise agreement is then expressed in
the same photometric metric as the member-vs-consensus score the rest of the patch
pipeline gates on, so the two numbers are directly comparable and cannot drift
apart as the render conventions evolve.

The diagonal is `1.0`. A member the validity gate drops is left **unscored** — its
row and column are `NaN`, reported separately as a per-member `scored` flag. A
track whose members cannot form a common support at all yields a matrix with only
its diagonal, which decides as `KeepAll` (no evidence is not contrary evidence).

## The decision rule

Every member is a hypothesis. Its **support** is the set of members whose pairwise
ZNCC to it reaches `bar`, itself always included; a member with no partner
supports a block of one. The **max-support block** wins. Note this is an inlier
set, not a clique: two members of the winning block need not agree with each
other, only with the hypothesis.

Given the winning block `B` of size `s` out of `k` members:

1. **`s == k`** — every member is in the block. Verdict `KeepAll`.
2. **Separation margin.** Let `min_intra` be the weakest finite link inside `B`
   and `max_cross` the strongest finite link from `B` to a member outside it; the
   margin is `min_intra − max_cross`. When the margin is undefined (`s < 2`, or a
   side with no finite link) or does not exceed `margin_gate`, verdict `KeepAll`.
3. **`2s > k`** — the block is a strict majority. Verdict `Split`: the block is
   kept, the members outside it are rejected.
4. **Otherwise** — verdict `Retire`. The point ships nothing.

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
- `Split` — the block's members ship; the rest are rejected observations of some
  other surface. The point survives on the block.
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

`bar` and `margin_gate` are **calibration defaults, not constants** — callers
override them. `bar` in particular is calibrated *for the render conventions in
the same table*: it is a threshold on a ZNCC whose value depends on the window,
the sampler, the channel treatment and the support, so a caller that changes
`resolution`, `window` or `sampler` must re-pick it.

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
}

pub enum MemberVerdict { KeepAll, Split, Retire }

pub struct MemberMatrix {         // members, k*k row-major zncc, per-member scored
    pub members: Vec<u32>, pub zncc: Vec<f64>, pub scored: Vec<bool>,
}
pub struct MemberDecision {       // verdict + what it was decided on
    pub verdict: MemberVerdict, pub kept: Vec<bool>, pub block: Vec<bool>,
    pub support: u32, pub margin: f64, pub min_intra: f64, pub max_cross: f64,
}
pub struct MemberCoherence { pub matrix: MemberMatrix, pub decision: MemberDecision }

// Matrix and decision are separate so a caller can inspect or supply either.
pub fn member_zncc_matrix(patch, views, members: &[u32], params) -> MemberMatrix;
pub fn decide_member_coherence(matrix: &MemberMatrix, params) -> MemberDecision;
pub fn validate_member_coherence(patch, views, members, params) -> MemberCoherence;

// Batch over a cloud, rayon-parallel across points, results in cloud order.
pub fn validate_patch_cloud_member_coherence(
    cloud: &PatchCloud, views: &[ProjectedImage<'_>], member_views: &[Vec<u32>],
    params: &MemberCoherenceParams, progress: Option<&AtomicUsize>,
) -> Vec<MemberCoherence>;

pub fn member_views_from_reconstruction(recon, cloud) -> Vec<Vec<u32>>;
```

`MemberMatrix::from_zncc` builds a matrix from an already-computed table, so a
caller holding its own pairwise scores can use the decision rule on its own.

The Python binding mirrors `PatchCloud.select_views`:

```python
PatchCloud.validate_member_coherence(
    recon, images, *, bar=0.65, margin_gate=0.05, resolution=24,
    window="gaussian_disk", window_sigma=0.6, sampler="bilinear_mip",
    min_valid_fraction=0.6, point_indexes=None, member_views=None,
    return_matrix=False, progress=None,
) -> list[dict]
```

`recon` is a reconstruction (member lists come from the tracks) or a
`CameraViews` (then `member_views` — a `point_index -> [image_index, ...]` map —
is required); `images` is a per-image list or a prebuilt `ImagePyramidSet`.
Each returned dict carries `point_index`, `members` (int32, the deduplicated
member order every other per-member array follows), `verdict`
(`"keep_all"` / `"split"` / `"retire"`), `kept` / `block` / `scored` (bool),
`support`, `margin`, `min_intra`, `max_cross`, and — under `return_matrix=True` —
the `k×k` float64 `zncc`.

## Testing

Sibling `tests.rs` under `patch/member_coherence/` covers the decision rule on
synthetic matrices with known block structure — all-agree; a clean 3+2 split; a
balanced 2+2 retirement; the block tie-break on mean coherence and then on member
index; a monotone drift chain kept whole by the margin gate; the same matrix
flipping between `KeepAll` and `Split` on `margin_gate` alone; the `k = 3` cases
(majority block splits, all-isolated keeps); unscored (`NaN`) members; `k ≤ 2` and
empty tracks; and `bar` reshaping the block — plus end-to-end builds over the
rendered synthetic plane scene: matrix symmetry and unit diagonal, one odd member
out splitting, a single surface kept whole, a balanced two-surface track retiring,
and member deduplication.

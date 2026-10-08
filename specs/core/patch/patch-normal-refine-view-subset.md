# Patch-normal refinement — view-subset selection (D-optimal refinement basis)

A patch's surface normal is refined photometrically: the patch is rendered into
every view that observes it and the renders are compared, so the cost grows
linearly with the number of views, while the quantity being estimated is only a
direction with two degrees of freedom. View-subset selection picks a small set of
views to refine against — the views that carry the most information about the
normal, chosen by a greedy D-optimal rule — so refinement runs on a handful of
views rather than dozens. It changes only which views the refinement compares;
every observation stays in the reconstruction, and the stored patch bitmap is
the render of the reference view chosen from all of them, not from the subset
(`refine_patch_normal` keeps the full view list for that pick).

`sfm embed-patches` uses it in its second and later refinement rounds, where the
view set has been expanded by view selection
([patch-view-selection.md](patch-view-selection.md)). On the 250-image
reconstruction used for profiling, that expanded set averages about 36 views per
point (from the localizer counters: 24,701,485 searches over 681,396 rounds),
and the round-2 normal refinement is the most expensive pass in the pipeline
(334 s, 38 % of wall time), with its per-view prerender at 49 % of the pass.
Five well-chosen views already over-determine a two-degree-of-freedom normal;
the rest add render cost and pull the consensus toward oblique, smeared views.

## Interface

The selection is `select_refine_subset` in
[view_subset.rs](../../../crates/sfmtool-core/src/patch/normal_refine/view_subset.rs).
It is crate-internal; callers turn it on through the `max_refine_views` field of
`NormalRefineParams` in
[params.rs](../../../crates/sfmtool-core/src/patch/normal_refine/params.rs),
which `refine_patch_normal_impl` in
[normal_refine.rs](../../../crates/sfmtool-core/src/patch/normal_refine.rs) reads. It
is bound as `PatchCloud.refine_normals(max_refine_views=…)` in
[refine_normals.rs](../../../crates/sfmtool-py/src/patches/refine_normals.rs)
(default `0`), used by `embed_patches(max_refine_views=8)` in
[_embed_patches.py](../../../src/sfmtool/_embed_patches.py), and exposed as
`sfm embed-patches --refine-max-views` (integer ≥ 0, default `8`; see
[embed-patches-command.md](../../cli/reconstruction/embed-patches-command.md)).

```rust
pub struct NormalRefineParams {
    // …
    /// Cap on the per-patch refinement basis. `0` (default) disables the cap.
    pub max_refine_views: u32,
}

/// The (at most) `k` most normal-informative views of `patch`, as ascending
/// indices into `view_dirs` (unit surface→camera directions).
pub(super) fn select_refine_subset(
    patch: &OrientedPatch,
    view_dirs: &[Vector3<f64>],
    k: u32,
) -> Vec<usize>;
```

The cap is a field on the existing refinement parameters rather than a separate
pass because the selection needs the per-view directions that refinement already
computes, and because the result has to replace the view list for the rest of
that one patch's refinement only. Returning indices rather than a filtered view
list lets the caller gather every per-view array it holds (views, directions,
camera centres, stored keypoints) with the same selection.

The crate default is `0`, so every caller that does not ask for the cap — view
selection, `inspect`/`compare --strips`, the tests — refines over all views
exactly as before. `embed_patches` sets `8` and applies it only to the
refinement calls of rounds 2 and later; the round-1 refinement over the raw
tracks is not capped. When the cap is active it logs one line saying so.

```python
cloud.refine_normals(emb, pyramids, resolution=12, max_refine_views=8)
```

## What constrains the normal

The appearance of a planar patch in a second view is related to the first by the
plane-induced homography `H = R − t·nᵀ/d`. The only term carrying the normal `n`
is the rank-1 `t·nᵀ/d`, so a view's sensitivity to the normal scales with its
baseline from the reference viewpoint over depth — that is, with how obliquely
it sees the patch. A near-frontal view (`v̂·n ≈ 1`) is nearly stationary in `n`
and contributes little constraint; the oblique views carry the information. The
normal has two degrees of freedom, so the obliquity has to be spread across
azimuth around the normal, or one tilt direction stays loose.

Choosing the views is therefore a D-optimal experimental-design problem: pick the
views whose combined information matrix about the two-degree-of-freedom normal
has the largest determinant.

## Algorithm

Given the patch's current unit normal `n` (the previous round's result), the
unit surface→camera direction `dᵢ` of each of the `m` views, and the cap `K`:

1. **No-op cases.** If `K == 0`, or `m ≤ K`, or the point is at infinity
   (`patch.w == 0`; its normal is fixed and refinement skips it), return all
   views.
2. **Per-view information vector.** For each view:
   - `cosθᵢ = clamp(dᵢ·n, −1, 1)`. A view with `cosθᵢ ≤ 0` faces the back of
     the patch; it is never selected and carries no information. Views in a set
     that passed view selection should not face backwards, but the guard is
     there.
   - The tangent projection `gᵢ = dᵢ − cosθᵢ·n` has length `sinθᵢ`. Expressed in
     the tangent basis `(t₁, t₂)` of `n` (`parameterization::tangent_basis`),
     the information vector is `wᵢ = (gᵢ·t₁, gᵢ·t₂)` — the obliquity `sinθᵢ`
     times the unit azimuth direction. A view with `‖gᵢ‖ ≤ 1e-6` is treated
     as frontal and gets `wᵢ = 0`.
   - The view's contribution to the 2×2 information matrix is `wᵢ wᵢᵀ`.
3. **Anchor.** The selected set starts with the least-oblique front-facing view
   (largest `cosθᵢ`). It is the view with the least foreshortening, which keeps
   the consensus appearance the subset fuses sharp. If no view faces the patch,
   all views are returned.
4. **Greedy D-optimal fill.** With `M = Σ_{i∈S} wᵢ wᵢᵀ` over the selected set
   `S`, repeatedly add the unselected front-facing view that maximises
   `det(M + wᵢ wᵢᵀ)`, until `|S| == K` or no front-facing view is left. This
   favours oblique views in azimuths that the views already chosen do not cover.
   Ties go to the lowest index, so the result is deterministic.

The selection does no rendering. It is `O(m·K)` per patch and runs inside the
existing per-patch parallel map, so its cost is small next to the renders it
removes.

### No fall-back to all views

The greedy returns the best-conditioned `K` views available, and the selection
returns them whenever an anchor exists. If those `K` views still leave one tilt
direction loose — a point whose views all lie along one azimuth arc — the full
view set is no better conditioned, because conditioning depends on the
directions of the views, not on their number. Using all views there would add
render cost without constraining the loose direction. That direction is settled
by the fronto-parallel prior during refinement, the same way as for any point
with little parallax.

The obvious fall-back — widen to all views when `λ_min(M_S) < γ·λ_min(M_full)` —
does not work. Information adds across views (`M = Σ wᵢwᵢᵀ`), so that ratio is
roughly `K/m` and the test fires for almost every point with `m ≳ 2K`, whatever
its conditioning. On the Spain Soapmaker sweep it fired on 57 % of eligible
points, left only 22 % capped, and made `K = 5` about 4 % slower than no cap.
What a many-view consensus does buy over a five-view one is photometric
robustness, which is a separate question from conditioning (see Non-goals).

### Choice of `K`

The Spain Soapmaker sweep measured the round-2 refinement 3.4–16.8× faster for
`K` from 10 down to 3, and end-to-end wall time 29–37 % lower. The angle between
the capped and the all-views normals grows as `K` shrinks: at `K = 5` the median
is 6.4° and the 95th percentile 36°; at `K = 10`, 3.0° and 31°. That difference
is not known to be error — reprojection error does not depend on the normal, and
the all-views normals are not ground truth either — so the pipeline default
`K = 8` is a balance between speed and agreement with the all-views result, not
a calibrated optimum.

## Implementation notes

- **Where the subset applies.** `refine_patch_normal_impl` computes each view's
  camera centre and surface→camera direction, then, when
  `max_refine_views > 0` and the patch has more views than the cap, calls
  `select_refine_subset` and rebinds its local views, directions, centres and
  stored keypoints to gathered copies of the selected entries. Everything after
  that — the seed search, coarse-to-fine search and final scoring — runs on the
  subset without knowing it is one. The returned patch is still a re-posing of
  the full input patch (centre and extent kept), so the refined normal applies
  to the whole surfel. The reported `valid_view_count` and related counts
  describe the subset.
- **Floor at `min_views`.** The cap passed to the selection is
  `max(max_refine_views, min_views)`, so a cap below the refinement's minimum
  view count cannot leave a patch with too few views to refine.
- **Why the output loses nothing.** Refinement changes only the patch normal; it
  does not touch the reconstruction's tracks. In `embed_patches` the stored
  patch bitmaps are rendered by the sub-pixel keypoint pass of the final round,
  with the reference-view rule choosing among the full view set, not by the
  normal refinement. And every view in the
  round-2 set already passed view selection's ZNCC threshold, so the photometric
  quality floor is enforced by membership; the subset only has to choose among
  vetted views by geometry.
- **Profiling.** The selection has its own `view_subset` phase in
  [prof.rs](../../../crates/sfmtool-core/src/patch/normal_refine/prof.rs), which
  also counts the patches whose basis was capped and those where a cap was
  requested but the selection returned every view because no view faced the
  patch.

## Validation harness

[validate_refine_subset.py](../../../scripts/validate_refine_subset.py) runs
`sfm embed-patches` on a given `.sfmr` once per `--refine-max-views` value
(`0` is the all-views baseline), each with `SFMTOOL_PROFILE=1`, and reports for
each run:

- the wall time of each normal-refinement pass (from the profile output) and end
  to end, with the capped and no-anchor patch counters;
- the angle between each surviving point's normal and the baseline's — mean,
  median and 95th percentile;
- point and observation counts, which should match the baseline;
- the reprojection-error distribution (mean and 95th percentile).

Its acceptance criterion is that at `K = 5` the round-2 refinement is
substantially faster (at least 2× is the aim), the median normal angle against
the baseline stays around a degree, and the 95th-percentile reprojection error
does not get worse.

## Testing

- [view_subset/tests.rs](../../../crates/sfmtool-core/src/patch/normal_refine/view_subset/tests.rs):
  the no-op cases (`K == 0`, `m ≤ K`, a point at infinity) return every view;
  the greedy pick prefers oblique, azimuthally spread views over a near-frontal
  cluster; the anchor is the least-oblique view; a large view set is capped to
  `K`; a view set along a single azimuth arc returns the best `K` rather than
  every view; back-facing views are never selected; the selection is
  deterministic.
- [normal_refine/tests.rs](../../../crates/sfmtool-core/src/patch/normal_refine/tests.rs)
  (`max_refine_views_caps_basis_close_to_full_result`): on a synthetic cloud, a
  cap of 4 out of 6 views refines over at most 4 views and lands within 2° of
  the all-views normal, and a cap at or above the view count reproduces the
  uncapped result exactly.
- [test_embed_patches_command.py](../../../tests/patch/test_embed_patches_command.py):
  `test_embed_patches_refine_max_views_is_lossless` checks that a run with
  `max_refine_views=5` keeps the point and observation counts of an all-views
  run to within 1 %, and `test_embed_patches_cli_refine_max_views_forwards`
  checks that `--refine-max-views` reaches `embed_patches` and that a negative
  value is rejected.

## Non-goals

The pick is geometric only: a view's information contribution `wᵢ wᵢᵀ` is not
weighted by how well that view matches the consensus. This matters more than it
might seem, because a geometry-only D-optimal pick deliberately chooses the most
oblique views, which are also the photometrically noisiest. Weighting the pick
by per-view ZNCC is proposed in
[patch-normal-refine-zncc-weighted-selection-amendment.md](../../drafts/patch-normal-refine-zncc-weighted-selection-amendment.md).

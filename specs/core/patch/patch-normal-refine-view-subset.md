# Patch-normal refinement — view-subset selection (D-optimal refinement basis)

## Motivation

Refining a patch's surface normal photometrically means rendering that patch into
every view that observes it and comparing the renders — cost that grows linearly
with the number of views, while the quantity being estimated is only a
two-degree-of-freedom direction. View-subset selection picks a small basis of
views to refine against: a greedy D-optimal choice of the views that carry the
most information about the normal, so refinement runs on a handful of views
rather than dozens without loosening what constrains the answer. Every
observation stays in the output — only the refinement basis shrinks.

In `sfm embed-patches`, the round-2+ (fine-tuning) normal-refinement pass runs
over the *expanded* view set produced by `select_views` — on the profiled
250-image reconstruction that averages **~36 views/point** (from the localize
counters: `searches / rounds` = 24,701,485 / 681,396). But
`refine_patch_normal` only estimates a **2-DOF surface normal**. Five
well-chosen views over-determine that; the remaining ~31 mostly inflate the
`cache_prerender` cost (49 % of the pass, and linear in view count) and drag the
consensus toward oblique smears.

Round-2 `refine_normals` is the single most expensive pass in the pipeline
(334 s, 38 % of wall on that dataset). Capping the *refinement basis* at a small,
carefully chosen subset is the biggest available lever.

Key facts that make this safe:

- `refine_patch_normal` only mutates the patch **normal** (`center`,
  `half_extent`, in-plane convention preserved). It does not touch the
  reconstruction's tracks.
- In `embed_patches`, the stored patch **bitmaps** are fused by the *subpixel*
  pass over the full view set (`_embed_patches.py`, `_refine_subpixel(...,
  render_bitmaps=r == rounds)`), **not** by `refine_normals`. So subsetting the
  refinement basis is lossless for the output: all observations remain, and the
  consensus texture is still fused from all of them.
- Every view in the round-2 set already cleared the `select_views` ZNCC bar, so
  the photometric quality floor is **already enforced by membership**. The
  subset job is therefore purely geometric: pick the most *observability*-rich
  views among already-vetted ones.

## What constrains the normal (the geometry)

The cross-view appearance link is the plane-induced homography
`H = R − t·nᵀ/d`. The only term carrying the normal `n` is the rank-1
`t·nᵀ/d`, so a view's sensitivity to the normal scales with its **baseline from
the reference viewpoint over depth** — i.e. with how *oblique* it sees the
surfel. A near-frontal view (`v̂·n ≈ 1`) is nearly stationary in `n` and
contributes little constraint; the oblique views carry the information. Two DOF
means we need obliquity spread across **azimuth** around the normal, or one tilt
direction stays loose.

This is a D-optimal experimental-design problem: maximise the information the
selected views carry about the 2-DOF normal.

## Algorithm — `select_refine_subset`

`select_refine_subset` lives in
[view_subset.rs](../../../crates/sfmtool-core/src/patch/normal_refine/view_subset.rs)
and is applied inside `refine_patch_normal_impl` through
`NormalRefineParams::max_refine_views`, exposed as
`PatchCloud.refine_normals(max_refine_views=…)` and
`sfm embed-patches --refine-max-views`; the validation harness is
[validate_refine_subset.py](../../../scripts/validate_refine_subset.py).

Per patch, given the incoming unit normal `n` (the patch's current normal, i.e.
the previous round's result), the point position `X`, the observing camera
centers `cᵢ` for the `m` views in the current set, and a cap `K`
(`max_refine_views`):

1. **No-op cases.** If `K == 0`, or `m ≤ K`, or the point is at infinity
   (`patch.w == 0`, normal fixed — refinement skips it), return **all** views.
2. **Per-view tangent geometry.** For each view `i`:
   - `dᵢ = normalize(cᵢ − X)` (unit surface→camera direction).
   - `cosθᵢ = clamp(dᵢ·n, −1, 1)`; skip (exclude) any view with `cosθᵢ ≤ 0`
     (back-facing; should not occur in a vetted set, guard anyway).
   - Tangent projection `gᵢ = dᵢ − cosθᵢ·n`; `sᵢ = ‖gᵢ‖ = sinθᵢ` (obliquity
     sensitivity); azimuth unit `ûᵢ = gᵢ / sᵢ` when `sᵢ > 1e-6`, else `ûᵢ = 0`
     (a perfectly frontal view carries no tangent information).
   - Express `ûᵢ` in the 2-D tangent basis `(t₁, t₂)` of `n` (use
     `parameterization::tangent_basis`). The per-view **information vector** is
     `wᵢ = sᵢ · (ûᵢ·t₁, ûᵢ·t₂)` (a 2-vector). Its outer product `wᵢ wᵢᵀ` is the
     view's contribution to the 2×2 information matrix.
3. **Anchor.** Seed the selected set `S` with the **least-oblique** view
   (max `cosθᵢ`) — a clean, low-foreshortening appearance anchor so the
   consensus reference the subset fuses stays sharp.
4. **Greedy D-optimal fill.** Maintain `M = Σ_{i∈S} wᵢ wᵢᵀ`. Repeatedly add the
   not-yet-selected view maximising `det(M + wᵢ wᵢᵀ)` until `|S| == K`. (Adding
   the view that most enlarges the information volume — naturally favours
   oblique views that are azimuthally complementary to those already chosen.)
5. **No conditioning fallback — always keep the best `K`.** The greedy already
   returns the best-conditioned `K` views available. If that subset still leaves
   one tilt DOF loose (a degenerate single-azimuth-arc point), the **full** view
   set is no better conditioned — conditioning is a property of the
   view-direction *geometry*, not the count — so falling back to all views would
   only add render cost without constraining the loose DOF. That DOF is resolved
   by the fronto-parallel prior at refine time, exactly as for any low-parallax
   point. So the selection returns the best `K` unconditionally (the only
   all-views returns are the no-op cases in step 1, and the degenerate case where
   no view is front-facing).

   The obvious fallback — inflate back to all views when
   `λ_min(M_S) < γ·λ_min(M_full)` — has no correct form, and it is worth saying
   why, because it is the first thing a reader will reach for. Information is
   *additive across views* (`M = Σ wᵢwᵢᵀ`), so that ratio is ≈ `K/m` and fires for
   essentially every point with `m ≳ 2K` regardless of conditioning: on the Spain
   sweep it tripped on 57 % of eligible points, left only 22 % actually capped,
   and made `K = 5` a net ~4 % *slower* than no cap at all. More fundamentally,
   even when it fires on a genuinely degenerate point, all views are no
   better conditioned than the best `K`. Photometric **robustness** — the real
   reason a many-view consensus beats a five-view one — is an orthogonal axis,
   and is what the ZNCC-weighted pick would address.

The function returns the selected view **indices** (into the caller's `views`
slice), or all indices for the step-1 no-op cases (and the no-front-facing-view
degenerate case). It performs **no rendering** — pure geometry, O(`m·K`) per
patch, run inside the existing per-patch rayon map, so its cost is negligible
against the renders it saves.

### Parameters / constants

- `max_refine_views: u32` — new field on `NormalRefineParams`. `0` (default) =
  **disabled** (use all views; byte-for-byte the current behavior). `K ≥ 1` caps
  the refinement basis at `K`. Guard `K` up to at least `min_views` internally
  so a cap below the refine floor can't strand a patch.

The pipeline sets `K = 8` (`embed_patches(max_refine_views=8)`,
`sfm embed-patches --refine-max-views 8`), while the crate-level default stays
`0` so non-`embed-patches` callers are unaffected. The Spain Soapmaker sweep
measured round-2 refine dropping 3.4–16.8× from `K = 10` down to `K = 3`, and
end-to-end wall ~29–37 %, with the normal difference against the all-views
baseline growing as `K` shrinks (`K = 5`: median 6.4° / p95 36°; `K = 10`:
median 3.0° / p95 31°). That divergence is not established as *error* —
reprojection is blind to normals, and the all-views normals are themselves not
ground truth — so `K = 8` is a speed/quality balance rather than a calibrated
optimum; settling it needs a normal-quality signal (a `Φ_full` comparison, or
visual inspection) and the ZNCC-weighted selection below.

## Where the subset restriction happens

Inside `refine_patch_normal_impl` (`normal_refine/mod.rs`), **after** the
`centers` / `view_dirs` are computed (currently lines ~147–166) and **before**
the seed search:

- When `params.max_refine_views > 0` and `patch.w != 0` and
  `views.len() > max_refine_views`, call `select_refine_subset` to get the kept
  indices, then rebind local `views`, `view_dirs`, `centers`, and
  `view_keypoints` to subset copies (all `Copy`/cheap to gather). Everything
  downstream (`coarse_to_fine`, `build_final_context`, `score`/`eval_phi`) then
  operates on the subset unchanged. The returned `patch` is still a
  `repose_patch` of the input patch (center/extent preserved), so the refined
  normal applies to the full surfel.
- `valid_view_count` etc. reflect the subset (the refinement basis) — that is
  correct and intended.

Since the default is `0`, **all existing callers** (`select_views`,
inspect/compare strips, tests) are unaffected.

## Where it lives

The selection is
[`view_subset.rs`](../../../crates/sfmtool-core/src/patch/normal_refine/view_subset.rs)
(`select_refine_subset`, on the tangent basis from `parameterization::tangent_basis`),
driven by `NormalRefineParams::max_refine_views` in
[`params.rs`](../../../crates/sfmtool-core/src/patch/normal_refine/params.rs) and
applied inside `refine_patch_normal_impl` in
[`mod.rs`](../../../crates/sfmtool-core/src/patch/normal_refine/mod.rs).
`prof.rs` counts the patches whose basis was capped and those where a cap was
requested but no anchor was available, and gives the selection its own
`view_subset` profiling phase.

The binding is `PatchCloud.refine_normals(max_refine_views=…)`
([`refine_normals.rs`](../../../crates/sfmtool-py/src/patches/refine_normals.rs),
default `0`). The pipeline reaches it through
`embed_patches(max_refine_views=8)`
([`_embed_patches.py`](../../../src/sfmtool/_embed_patches.py)), which applies the
cap to the round-2-and-later `refine_normals` calls only, leaving round 1 — the
raw-track pass — uncapped, and logs one line when the cap is active. The CLI flag
is `sfm embed-patches --refine-max-views` (int, default `8`, `IntRange(min=0)`);
see [embed-patches-command.md](../../cli/reconstruction/embed-patches-command.md).

## Non-goals

The pick is **geometric only**: a view's information contribution `wᵢ wᵢᵀ` is not
weighted by how well that view actually matches the consensus. Weighting it by
per-view ZNCC is proposed in
[`../../drafts/patch-normal-refine-zncc-weighted-selection-amendment.md`](../../drafts/patch-normal-refine-zncc-weighted-selection-amendment.md),
and it matters more than a refinement usually would: geometry-only D-optimal
deliberately picks the *most oblique* views, which are also the photometrically
noisiest, so an unweighted subset trades robustness for observability.

## Validation harness

[`scripts/validate_refine_subset.py`](../../../scripts/validate_refine_subset.py)
runs `sfm embed-patches` on a given `.sfmr` once per `--refine-max-views` value
(`0` = the all-views baseline), each under `SFMTOOL_PROFILE=1`, and reports:

- **Wall time** per pass (parsed from the profile blocks) and end-to-end.
- **Normal agreement vs. the `0` baseline**: per-surviving-point angular Δ
  between the subset run's normal and the baseline normal — mean / median / p95.
  A good subset keeps this small (baseline round-2 normal Δ vs. seed was 4.8°).
- **Output shape**: point and observation counts (should be ~unchanged —
  lossless claim).
- **Quality**: per-point reprojection-error distribution (mean / p95) subset vs.
  baseline.

Acceptance target: at `K = 5`, round-2 `refine_normals` wall time drops
substantially (aim ≥ 2×) while median normal Δ vs. baseline stays small (order a
degree) and reproj-error p95 does not regress. The harness output is the
evidence for choosing `K`.

## Tests

- **Rust unit tests** (`view_subset.rs`):
  - `m ≤ K`, `K == 0`, infinity point → returns all indices.
  - A synthetic point with views clustered near-frontal in one azimuth plus a
    few oblique views spread in azimuth → the greedy pick includes the oblique,
    azimuthally-complementary views (not just the highest-`cosθ` cluster).
  - Anchor is the least-oblique view.
  - A degenerate single-azimuth-arc view set returns the best `K` rather than
    falling back to all views.
  - Back-facing views are never selected.
  - Determinism: same inputs → same selection.
- **Rust** (`normal_refine/tests.rs`): `refine_patch_cloud_normals` with
  `max_refine_views = K` on a small synthetic cloud produces normals close to
  the full-set result, and `max_refine_views = 0` is byte-for-byte the current
  path.
- **Python** (`tests/test_cli_embed_patches*.py` or a new module): the CLI
  accepts `--refine-max-views`, a small end-to-end run with `--refine-max-views
  5` on the seoul_bull fixture succeeds and produces an `embedded_patches` recon
  with the same point/observation counts as the default run (lossless), within
  tolerance.

# Patch-normal refinement — view-subset selection (D-optimal refinement basis)

> Status: **implemented (2026-07-05)** —
> `crates/sfmtool-core/src/patch/normal_refine/view_subset.rs`
> (`select_refine_subset`, `SUBSET_CONDITIONING_FLOOR`), wired into
> `refine_patch_normal_impl` via `NormalRefineParams::max_refine_views` (default
> `0` = off), exposed as `PatchCloud.refine_normals(max_refine_views=…)` and
> `sfm embed-patches --refine-max-views`; validation harness in
> `scripts/validate_refine_subset.py`. Design owner: profiling of `embed-patches`
> on the 250-image Spain Soapmaker reconstruction (2026-07-04).

## Motivation

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
5. **Well-conditioning fallback (data-derived).** Let `λ_min(M_S)` be the
   smaller eigenvalue of the selected set's `M`, and `λ_min(M_full)` the same
   over **all** `m` views. Keep the subset only if
   `λ_min(M_S) ≥ γ · λ_min(M_full)` (default `γ = 0.5`). Otherwise the subset
   lost too much observability of one tilt DOF (e.g. a nearly-frontal-only
   track) — return **all** views for that patch rather than a degenerate `K`.

The function returns the selected view **indices** (into the caller's `views`
slice), or all indices when any no-op / fallback condition holds. It performs
**no rendering** — pure geometry, O(`m·K`) per patch, run inside the existing
per-patch rayon map, so its cost is negligible against the renders it saves.

### Parameters / constants

- `max_refine_views: u32` — new field on `NormalRefineParams`. `0` (default) =
  **disabled** (use all views; byte-for-byte the current behavior). `K ≥ 1` caps
  the refinement basis at `K`. Guard `K` up to at least `min_views` internally
  so a cap below the refine floor can't strand a patch.
- `SUBSET_CONDITIONING_FLOOR: f64 = 0.5` (the `γ` above) — a module constant.

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

## Plumbing

1. **`crates/sfmtool-core`**
   - New module `patch/normal_refine/view_subset.rs`: `select_refine_subset(...)`
     + the constant, with unit tests. Reuse `parameterization::tangent_basis`.
   - `params.rs`: add `max_refine_views: u32` to `NormalRefineParams`
     (`Default = 0`), documented.
   - `mod.rs`: wire the subset restriction into `refine_patch_normal_impl` as
     above; add `mod view_subset;`.
   - `prof.rs` (optional, nice-to-have): a counter for how many patches used the
     subset vs. fell back, reported in the profile summary.
2. **`crates/sfmtool-py`** (`py_patch_cloud.rs`)
   - `refine_normals`: add kwarg `max_refine_views=0`; set
     `params.max_refine_views`. Document it in the method docstring.
3. **Python** (`src/sfmtool/_embed_patches.py`)
   - `embed_patches(...)`: add `max_refine_views: int = 0`; pass it to the
     round-2..N `cloud_r.refine_normals(...)` call **only** (leave round 1 — the
     raw-track pass — untouched). Add a one-line progress note when active.
4. **CLI** (`src/sfmtool/_commands/embed_patches.py`)
   - Add `--refine-max-views` (int, default `0`, `IntRange(min=0)`), forwarded
     to `embed_patches(max_refine_views=...)`. Document: `0` = use all views;
     `N` caps the round-2+ normal-refinement basis at the `N` most
     normal-informative views (D-optimal), leaving all observations in the
     output.
5. **Specs**
   - Update `specs/cli/embed-patches-command.md` with the new flag.
   - Cross-link `specs/core/patch-normal-refinement.md` to this file.

## Deferred (follow-up, not this task)

- **ZNCC-weighted selection.** Weight each view's information contribution
  `wᵢ wᵢᵀ` by its per-view ZNCC-to-consensus (SNR), so the D-optimal pick
  discriminates among already-vetted views by photometric quality too. Blocked
  on persisting `select_views`'s per-view `scores` (currently discarded in
  `_embed_patches.py`) through the per-round compaction (point indices are
  renumbered each round). Ties into the separate "persist select_views scores"
  analysis work.

## Validation harness (required — the risk is under-constraining)

Add `scripts/validate_refine_subset.sh` (or a small Python driver) that runs
`embed-patches` on a given `.sfmr` with `--refine-max-views ∈ {0, 3, 5, 8}`
(0 = baseline), each under `SFMTOOL_PROFILE=1`, and reports:

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
evidence for choosing a default `K` (and eventually flipping the default from
`0`).

## Tests

- **Rust unit tests** (`view_subset.rs`):
  - `m ≤ K`, `K == 0`, infinity point → returns all indices.
  - A synthetic point with views clustered near-frontal in one azimuth plus a
    few oblique views spread in azimuth → the greedy pick includes the oblique,
    azimuthally-complementary views (not just the highest-`cosθ` cluster).
  - Anchor is the least-oblique view.
  - Fallback: an all-near-frontal view set (no parallax) → conditioning floor
    trips → returns all views.
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

## Task-completion checks (from AGENTS.md)

- Rust: `pixi run cargo fmt && pixi run cargo clippy --workspace`.
- Rebuild bindings (this touches code re-exported through `sfmtool-py`):
  `pixi run -e test maturin develop --release`.
- Python: `pixi run fmt && pixi run check`, then `pixi run test -- <modules>`.
- Rust tests: `pixi run cargo test --workspace` (llvm-cov excludes `sfmtool-py`).
</content>

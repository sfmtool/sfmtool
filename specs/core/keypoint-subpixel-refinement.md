# Photometric Subpixel Keypoint Refinement

_Status: **MVP implemented** (Phase 2). The forward-additive ECC Gauss–Newton
refiner lives in `crates/sfmtool-core/src/patch/keypoint_subpixel.rs` (exposed to
Python as `PatchCloud.refine_keypoints`). Both the single-pass-frozen variant
(`max_outer_sweeps = 1`, the default) and the per-sweep-refresh variant
(`max_outer_sweeps > 1`, with mean-per-view-move early-exit) are wired so the
trade-off is measurable, not chosen at design time. The deferred work — per-move
(Gauss–Seidel) incremental consensus, leave-one-out consensus, the sampler
value+gradient interface and anisotropic value+gradient Jacobian,
inverse-compositional ECC, and the joint bundle — remains as described in
"Open questions" / "Design details" below. A
**standalone** algorithm: given a keypoint that
is **already close** to correct, refine it to sub-pixel by **local** continuous
optimization of image photoconsistency (gradients, fractional sampling). It does
no global search and no view selection. Its typical caller is keypoint
localization, but it is specified independently and usable on any
approximately-correct keypoint set.
Intended module: `sfmtool-core/src/patch/keypoint_subpixel.rs`._

This is the **high-accuracy** refinement: a continuous solve reaches the true
optimum of the photometric objective, so it best approximates ground truth. A
(supersampled) grid search is always *discrete* — quantized to its grid — so this
is the quality reference, the option to reach for when accuracy matters most,
regardless of how good a faster grid is. It is needed as that option; the question
is never whether to keep it, only when a cheaper approximation suffices instead.

## Contract and scope

This is a **local refiner**, and its guarantees hold only inside that scope:

- **Precondition: the seed is close.** Each input keypoint must already lie within
  the local convergence basin of the true optimum (in practice ≲ 1 px). The
  algorithm linearizes around the seed; a seed outside the basin can converge to a
  wrong local optimum or be rejected, not rescued. Putting the keypoint *in* the
  basin is the **caller's** job (e.g. a discrete search).
- **Scope: local only.** It searches no grid and visits no distant candidates; it
  takes a few Gauss–Newton steps from the seed. It will not recover a grossly
  mislocalized keypoint.
- **Does not change membership (one exception).** It moves keypoints; it never
  *adds* a view, and the only drop is the projection gate: a view in which the
  patch centre fails to project (behind the camera or outside the frame) is
  dropped, since the per-view offset is reported relative to that projection.
  Selection beyond that belongs to whoever produced the set.
- **Never worse than the seed.** A step is accepted only if it raises the ECC
  score against the **current** consensus `T` (and stays in frame); if none does,
  the seed is kept — so refining is safe even when a particular keypoint turns
  out not to be refinable. With a refreshed consensus (`max_outer_sweeps > 1`)
  the guarantee is **within a sweep**: each accepted step is non-decreasing
  against that sweep's `T`. Across sweeps `T` changes, so the final score against
  the final `T` is not bit-bounded below by the seed score against the seed `T`
  (the single-pass-frozen default `max_outer_sweeps = 1` is the case where the
  two coincide).

## Objective

The objective is to maximize **total cross-view photoconsistency** over all
views' offsets `{δ_v}` (2 DOF each, in-plane translation):

```
E({δ}) = Σ_v Σ_k w_k · ẑ(I_v(x_k(δ_v)))_k · T({δ})_k
```

where `k` runs over the `R×R` window support, `w_k` the window weights,
`x_k(δ)` the source sample point for grid pixel `k` at offset `δ`, `ẑ` the
weighted z-normalization, and `T({δ})` the (robust) cross-view **consensus** of
the views *at their current offsets*. The per-view term is the **Enhanced
Correlation Coefficient** (Psarakis & Evangelidis, "An Enhanced Correlation-Based
Method for Stereo Correspondence with Sub-Pixel Accuracy," ICCV 2005; generalized
to parametric alignment in Evangelidis & Psarakis 2008) — a zero-mean normalized
correlation, illumination-invariant. Maximizing it over all views is the
continuous form of congealing — the same coupled criterion the discrete search
optimizes by grid argmax, here by gradient ascent.

**The reference is not fixed.** `T` depends on `{δ}`, so refinement **alternates**
— refresh `T`, move the views, repeat — with `T` held fixed only within a view's
inner Gauss–Newton solve. Refreshing earns its keep: in prototyping the consensus
visibly **sharpened** as the views came into alignment (well-registered patches
average without blurring detail away), which sharpens the objective in turn.

## Inputs

- One 3D point with its **patch frame** (centre, in-plane axes, normal) — fixed.
  The centre is homogeneous, so the patch may be at **infinity** (`w = 0`): it
  projects as a direction (a ray) and `δ` shifts it in-plane on its tangent-sphere
  frame. Infinity patches **must be refined like any finite one** — they carry
  keypoints too — with the same objective, sampling, and warp Jacobian
  (`WarpMap::from_patch` and the projection already handle `w = 0`, and the
  Jacobian is just central differences on the resulting warp coords). They are
  *not* skipped the way normal refinement skips them.
- A set of **views**, each with a **seed keypoint**.
- An optional **reference patch** to start from — its `R×R` grid sets the scoring
  resolution (the views' sampled cores match it). If omitted, it is built (at a
  chosen resolution) from the seed-aligned views.
- The **`Sampler`** (interpolation: `Bilinear` / `Anisotropic`) to sample the
  source pyramid with — see "Sampling."

## Algorithm: ECC (forward-additive Gauss–Newton)

Maximize the ECC criterion (Objective) by forward-additive Gauss–Newton — a few
steps per view from the seed. ECC's formulation handles the `ẑ` normalization
analytically inside the step, so we don't naively differentiate a per-iteration
mean/norm. Because the reference (consensus) depends on where the views sit, it is
**alternating** (outer) over a few sweeps until the views stop moving:

1. **Refresh the consensus.** (Re)build the robust consensus `T` from the views at
   their current offsets. (Shared `T` for all views, or leave-one-out per view —
   Open questions.)
2. **Move each view (inner, 2–3 GN steps from its current `δ`), holding `T` fixed:**
   - Sample the view's patch core at the current `δ` (fractional, via the `Sampler`).
   - z-normalize it; form the ECC residual against `T`.
   - Take the ECC Gauss–Newton step on the 2-DOF in-plane offset, using the image
     Jacobian `∂I/∂δ` of the sampled core. The **target** is the sampler's analytic
     image-gradient Jacobian (see Sampling) — composed with the warp's
     `∂(image coords)/∂δ` — so the gradient is LOD-consistent with the value. The
     **MVP** computes `∂I/∂δ` by central finite differences on the warp/sample
     coords instead (one extra render per axis); the analytic path is deferred to
     a later phase (see "Design details"). Update `δ`. Stop at `‖Δδ‖ < ε` (e.g.
     0.01 px) or the cap.
3. **Repeat** from 1 until the mean per-view move is below `ε` or the sweep cap;
   one or two sweeps usually suffice (the seed is close, so the consensus barely
   moves — see the single-pass note).
4. **Guard, per view.** Accept a step only if it raises the score (otherwise
   backtrack; if no improving step is found, keep the current `δ`) — so the result
   is never worse than the seed. A step that drives a sample out of frame is
   invalid. A near-singular system (low-texture / aperture problem) → keep the
   seed. (Keeping the seed *close* is the caller's precondition, not re-checked
   here.)

Output per view: refined `δ_v` → `keypoint_v = project_i(X_p) + δ_v` (source px,
via the grid-to-image scale).

### Consensus refresh granularity

Because the seed is close, offsets move sub-pixel and the consensus is nearly
stationary, so how often `T` is rebuilt is a speed/convergence knob. From coarsest
to finest:

- **Single pass, frozen** (cheapest). Refresh once at the seed, move every view,
  done. Ignores the second-order feedback of moved views on `T`; a good
  approximation at sub-pixel scale.
- **Per-sweep refresh.** Rebuild `T` between sweeps (the alternating loop above).
- **Per-move (Gauss–Seidel) incremental.** Update `T` after *each* view moves, so
  the next view aligns to a consensus that already reflects the last (fastest
  convergence). Cheap if `T` is kept as a **running sum** `S = Σ_v w_v · ẑ_v` of
  the z-normalized view patches: moving view `v` from `δ` to `δ'` is
  `S += w_v · (ẑ_v' − ẑ_v)` (O(`n·channels`)), then renormalize `S → T`. The moved
  view's `ẑ_v'` is already computed by its GN step, so the only added work per move
  is the delta plus one renormalization — negligible next to sampling/gradients.
  **Bonus:** leave-one-out is then free — view `v`'s reference is
  `normalize(S − w_v·ẑ_v)`.

**Robustness with the incremental sum: two update frequencies.** A robust (IRLS)
consensus couples every view's weight to all residuals, so one move perturbs them
all and a pure delta no longer holds. Decouple the frequencies: delta-update the
weighted **sum** every move (cheap, exact for fixed weights), and recompute the
IRLS **weights** at a **lower frequency** (e.g. once per sweep). The weights go
slightly stale between refreshes — fine at sub-pixel scale, where the views barely
move — and the per-move path stays cheap. This is the intended default for the
incremental variant.

A fully **joint** bundle (all `δ_v` and `T` solved simultaneously, not
alternating) is the limiting case — more coupling for little expected gain at this
scale; an extension, not the default.

## Sampling

The refiner samples each view's patch core from the **source pyramid** at the
current fractional `δ`, using the **existing** machinery — not a new sampler
abstraction. (The grid-search cache is *not* used here: it's a fixed render, so an
LOD-consistent gradient would have to be decoupled from the pyramid the Jacobian
naturally comes from. The cache is the grid search's, not the fine tune's.)

- **Interpolation** is the existing `Sampler` enum (`Bilinear` / `Anisotropic`) —
  the same knob `refine-normals` and `localize` already take. Anisotropic suits
  grazing / foreshortened views where bilinear under-samples; bilinear is the cheap
  default. This enum is "the sampling parameter."
- **Rendering** reuses `WarpMap::from_patch` + `remap_bilinear` /
  `remap_aniso_with_pyramid`. Gradients come from value+gradient variants of those
  two functions (Design details), giving the analytic image gradient at the *same
  interpolation/LOD* as the value — no finite differences (which across a mip
  boundary would be LOD-inconsistent).

## Outputs

Per view: the refined sub-pixel `keypoint_v`, its offset from the projection, and
the final score. The view set is returned unchanged (a guard-failed view stays at
its seed, so set and ordering are preserved).

## Validation

It deliberately produces a **better** sub-pixel result than a parabolic / discrete
estimate, so it is *not* validated by equivalence to one:

- **Synthetic recovery.** On rendered views with a known planted sub-pixel offset
  (seed within the basin), recover `δ` to < 0.02 px.
- **Quality, not equivalence.** Refined keypoints are **as good or better** than
  the seed by cross-view photoconsistency (mean windowed ZNCC) and/or
  reprojection.
- **Patch sharpness** on the test datasets. The consensus patch should *sharpen*
  after refinement (the effect prototyping observed — well-registered views average
  without blurring detail away). Measure a sharpness metric of the consensus (e.g.
  gradient energy / variance of the Laplacian) before vs after, expecting it to
  rise — a complementary signal to ZNCC/reprojection.
- **Points at infinity.** A `w = 0` patch is refined (not skipped) — synthetic
  recovery and the never-worse guard hold for it as for a finite patch.
- **Guard correctness.** A flat / out-of-frame / non-improving case keeps the seed
  (never worse by the score). A seed *outside* the basin is a contract violation
  the refiner won't detect — it may converge to a wrong nearby optimum; keeping the
  seed in the basin is the caller's job.

## Composition

- **A producer (example):** the discrete congealing search
  ([patch-keypoint-localization.md](patch-keypoint-localization.md), accelerated
  by [keypoint-localization-search-cache.md](keypoint-localization-search-cache.md))
  lands each keypoint in the basin and yields a kept view set + consensus — an
  ideal seed/template source for this refiner. But the dependency is one-way and
  optional: this algorithm only needs a patch, views, seeds, and a template.
- **A consumer:** `sfm embed-patches` writes the refined keypoints as the
  per-observation `keypoints_xy` (the geometric anchor is defined in the v4
  [sfmr-file-format.md](../formats/sfmr-file-format.md)).

## Open questions

- When does a cheaper approximation suffice instead? A **supersampled grid search**
  (`m > 1` in
  [keypoint-localization-search-cache.md](keypoint-localization-search-cache.md))
  resolves sub-pixel directly in one all-SIMD kernel — faster, but *discrete*. This
  continuous refinement stays the high-accuracy / ground-truth option; the question
  is where the grid is good enough and where this is worth its cost (measure the
  accuracy gap), not whether to keep this.
- Inverse-compositional ECC as an accelerator (precompute the template Hessian
  once instead of per step — the Lucas–Kanade variants, Baker & Matthews, IJCV
  2004). Its payoff is weak here twice over: the consensus refreshes, so the
  precompute is per refresh (per view, for leave-one-out) not once-ever; and the
  forward-additive per-step gradient is already cheap via the sampler's Jacobian.
  Start forward-additive; adopt IC only if the precompute clearly amortizes.
- Consensus refresh granularity: is the single-pass frozen consensus enough, or do
  per-sweep / per-move (incremental) refreshes measurably help? For the incremental
  variant, how stale can the lower-frequency IRLS weights get before it hurts?
  And **shared** vs **leave-one-out** consensus — LOO (free with the running sum)
  avoids a view aligning to itself; is the self-pollution at sub-pixel scale even
  worth caring about?
- Which `Sampler` — does bilinear suffice, or do grazing / foreshortened views
  need anisotropic? Pick the default.
- Per-view vs. joint — does the joint bundle beat per-view enough to justify the
  coupling?

## Design details: gradient-capable sampling interface

Implementation note for the sampler's analytic Jacobian (see "Sampling"). Per
support pixel the GN step needs the value **and** `∂I/∂δ` (a 1×2 per channel) to
accumulate the ECC normal equations. By the chain rule
`∂I/∂δ = ∇_src I · J`, where `∇_src I` is the image gradient in source-pixel
coords (from the interpolation) and `J = ∂(source coords)/∂(patch grid)` is the
warp Jacobian — `δ` is an in-plane patch-grid translation, so `∂(source)/∂δ = J`.

What already exists (`crates/sfmtool-core/src/camera/`):

- **Value sampling.** `remap::sample_bilinear_u8` (bilinear) and
  `remap::remap_aniso_with_pyramid` (LOD from `sigma_minor`, elliptical footprint
  along the major axis).
- **Warp Jacobian.** `WarpMap::compute_svd` already forms
  `J = [[dx/dcol, dx/drow], [dy/dcol, dy/drow]]` per pixel (central differences on
  the warp coords) and stores its SVD; `WarpMap::get_svd` exposes
  `(sigma_major, sigma_minor, major_dx, major_dy)` — what the aniso sampler reads.

New interface functions to add:

1. **Bilinear value+gradient** — `sample_bilinear_with_grad_u8(img, x, y, ch) ->
   (val, dI_dx, dI_dy)`. From the same four taps `v00,v10,v01,v11` (no extra
   fetch):

   ```
   dI_dx = (1-fy)·(v10 − v00) + fy·(v11 − v01)
   dI_dy = (1-fx)·(v01 − v00) + fx·(v11 − v10)
   ```

2. **Aniso/pyramid value+gradient** — `remap_aniso_with_grad(...)` (and a per-pixel
   `sample_aniso_with_grad`) returning `(val, dI_dx, dI_dy)` computed at the **same
   level(s)/footprint as the value**: take the per-level bilinear gradient (via #1),
   scale by the level's `2^level` to express it in level-0 source px, and blend the
   two levels with the same `frac` the value uses. (Finite differences across a mip
   boundary would not be LOD-consistent — this is the whole reason to compute it
   inside the sampler.)

3. **Warp-Jacobian accessor** — expose the raw per-pixel `J`
   (`WarpMap::get_jacobian(col, row) -> [[f32; 2]; 2]`); `compute_svd` already
   computes it internally, so this is just surfacing it (or reconstruct
   `J = U·S·Vᵀ` from `get_svd`). The refiner composes `∇_src I · J` → `∂I/∂δ`.

Per support pixel/channel the refiner gets `(value, ∂I/∂δ)` by calling #1 or #2
(per the `Sampler`) and composing with #3; value-only callers keep the existing
`remap_*` functions unchanged.

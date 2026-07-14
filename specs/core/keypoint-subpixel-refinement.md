# Photometric Subpixel Keypoint Refinement

_Status: **MVP + analytic Jacobian + per-move consensus implemented + wired
into `embed_patches`, on by default (1 sweep)** (Phases 2 and 3B). The
forward-additive ECC Gauss–Newton refiner lives in
`crates/sfmtool-core/src/patch/keypoint_subpixel.rs` (exposed to Python as
`PatchCloud.refine_keypoints`). All three "Consensus refresh granularity"
variants are wired so the trade-off is measurable: the single-pass-frozen
variant (`max_outer_sweeps = 1`, the default), the per-sweep-refresh variant
(`max_outer_sweeps > 1`, `consensus_refresh = "per_sweep"`, with
mean-per-view-move early-exit), and the per-move (Gauss–Seidel) incremental
variant (`consensus_refresh = "per_move"`, IRLS weights refreshed at the
per-sweep boundary — the spec's two-frequency design). The sampler value+gradient
interface (`remap_bilinear_with_grad`, `remap_aniso_with_grad`,
`WarpMap::get_jacobian`) is implemented per the "Design details" section below,
collapsing the per-GN-step gradient build from 5 renders (value + 4 FD) to 1
(value + analytic gradient composed with the warp Jacobian). The
`embed_patches` wiring exposes LK as an **integer `subpixel` kwarg** (and
`--subpixel` CLI option): a per-round outer-sweep count where `0` disables the
sub-pixel pass (the localizer's keypoints are used as is) and `N >= 1` runs the
refiner with `max_outer_sweeps = N`. The pipeline always uses the per-sweep
consensus variant (`consensus_refresh = "per_sweep"`); the per-move
(Gauss–Seidel) variant is reachable only via the direct
`PatchCloud.refine_keypoints(consensus_refresh="per_move")` binding, not through
`embed_patches`. The supersampled grid is exposed in parallel via a new
`search_resolution_multiplier` kwarg on `PatchCloud.localize_keypoints`
(and pass-through on `embed_patches`). The production default is
`subpixel=1` (one LK sweep per round — **on by default**),
`search_resolution_multiplier=1.0`. `PatchCloud.refine_keypoints` and `PatchCloud.refine_normals`
both default to seeding each view at that observation's inline
stored keypoint when the recon carries them (an embedded_patches
recon), with per-view fall-through to the reprojected center for views
that have no inline observation (e.g. ones admitted by `select_views`
beyond the SIFT track). The two functions diverge on what to do when
the recon has no inline keypoints at all (a sift-files recon):
`refine_keypoints` is a **local refiner — it strictly requires
starting keypoints**, so a sift-files call without explicit
`starting_keypoints` raises `ValueError` (the projection alone isn't
a "real" keypoint for the purposes of a local refiner; convert the
recon to embedded_patches first, or supply explicit per-point seeds).
`refine_normals` is a normal optimizer that can legitimately anchor
each view at the reprojected center, so its `use_stored_keypoints`
flag stays a plain bool (default `True`): when set, anchor at the
stored keypoint per view with per-view fall-through to the reprojected
center; when explicitly `False`, anchor every view at the reprojected
center regardless of recon kind (used by callers like
`sfm compare --strips` and the cross-validation script that want a
defined comparison reference independent of whether the recon
carries inline keypoints). The refiner can additionally fuse each
point's **representative bitmap** at the final keypoints
(`render_bitmaps` — see "Outputs" below); `embed-patches` sources its
stored `patch_bitmaps` and its culled-point drop signal from that,
no longer from normal refinement. The remaining deferred work —
leave-one-out consensus
(measured-and-rejected; see below), inverse-compositional ECC, the joint
bundle, and SIMD of the new sampler functions — is described in "Open
questions" below._

_**Per-move shared T (not LOO).** The "free with running sum" leave-one-out
bonus the spec lists as the incremental variant's natural default was
measured against the shared running consensus on `dino_dog_toy` (300 patches,
1369 views): LOO yielded mean ECC **0.82** at 5 sweeps vs shared T's **0.87**
— a clear regression at the small view counts of real tracks (3–5 views),
where the chain of LOO updates amplifies drift more than the self-pollution it
removes. The per-move path therefore uses shared `T`. This is recorded as the
measured negative result; LOO remains a one-line cost change if a future
measurement (large-N tracks?) shifts the verdict. See
`crates/sfmtool-core/src/patch/keypoint_subpixel.rs::RunningConsensus`. On real
data per-move (shared) matches per-sweep within noise (mean ECC 0.8725 vs
0.8716 at 5 sweeps), with a small one-sweep convergence advantage (0.8610 vs
0.8584). It lands as **opt-in** behind `consensus_refresh = "per_move"`; the
default stays `per_sweep`._

A **standalone** algorithm: given a keypoint that
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
     Jacobian `∂I/∂δ` of the sampled core. `∂I/∂δ = ∇_src I · J` where `∇_src I`
     comes from the sampler's analytic image-gradient interface (see Sampling and
     "Design details") and `J = ∂(image coords)/∂(patch grid)` is the warp
     Jacobian (`WarpMap::get_jacobian`) — so the gradient is LOD-consistent with
     the value. One value+gradient render per GN step replaces the MVP's previous
     5 renders (value + 4 axis-FD); the FD path is gone. Update `δ`. Stop at
     `‖Δδ‖ < ε` (e.g. 0.01 px) or the cap.
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
  the next view aligns to a consensus that already reflects the last (the
  intuition: tighter coupling within a sweep). Cheap if `T` is kept as a **running
  sum** `S = Σ_v w_v · ẑ_v` of
  the z-normalized view patches: moving view `v` from `δ` to `δ'` is
  `S += w_v · (ẑ_v' − ẑ_v)` (O(`n·channels`)), then renormalize `S → T`. The moved
  view's `ẑ_v'` is already computed by its GN step, so the only added work per move
  is the delta plus one renormalization — negligible next to sampling/gradients.
  **Bonus:** leave-one-out is then free — view `v`'s reference is
  `normalize(S − w_v·ẑ_v)`.
  - _Measured outcome (see implementation status above): on dino_dog_toy
    per-move matched per-sweep within noise at converged sweep counts; per-move
    at a single sweep narrowly beat single-pass-frozen at the same cost
    (ECC 0.8610 vs 0.8584). LOO regressed (0.82 vs 0.87 shared at 5 sweeps) at
    real-track view counts (3–5), so the implementation uses shared `T`. The
    "fastest convergence" intuition above does not strictly hold on this data —
    treat the variant as "tighter within-sweep coupling" rather than a
    convergence-speed claim, and `N = 2` is not recommended (shared-`T`
    self-pollution dominates at the minimal view count)._

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
abstraction.

- **Interpolation** is the existing `Sampler` enum (`Bilinear` / `BilinearMip` /
  `Anisotropic`) — the same knob `refine-normals` and `localize` already take.
  Anisotropic suits grazing / foreshortened views where bilinear under-samples;
  `BilinearMip` (one bilinear tap from the mip level nearest the warp's
  compression) bounds cross-scale aliasing at ≈ bilinear cost; bilinear is the
  cheap default. This enum is "the sampling parameter."
- **Rendering** reuses `WarpMap::from_patch` + `remap_bilinear` /
  `remap_bilinear_mip` / `remap_aniso_with_pyramid`. Gradients come from
  value+gradient variants of those functions (Design details), giving the
  analytic image gradient at the *same interpolation/LOD* as the value — no
  finite differences (which across a mip boundary would be LOD-inconsistent).

### Render-once context tile (implemented 2026-07)

An earlier revision of this section argued the grid-search cache could not
serve the fine tune (fixed render, LOD-decoupled gradient). That objection is
answered by storing the **gradient planes in the tile**: since the patch frame
is fixed and only the 2-DOF `δ` moves, every render of a (point, view) pair is
the same patch→image map at a sub-pixel shift, and the solver evaluates ~10–14
of them per pair (GN steps + line-search probes). The implementation
(`RefineTile` in `keypoint_subpixel.rs`) therefore prerenders **one** expanded
patch-grid-aligned tile per pair — the localizer's `ContextTile` idea (see
[keypoint-localization-search-cache.md](keypoint-localization-search-cache.md)),
adapted to fractional reads:

- The tile is centred at the view's **seed** offset and sized
  `R + 2·(⌈max_offset_px⌉ + 2)` so every offset the line search can accept
  reads in-bounds (out-of-coverage reads fall back to a direct render; ~0 in
  practice). It stores the sampler's unquantized values **plus** the
  pre-composed patch-grid image Jacobian `∇_src I · J` per texel — the same
  analytic, LOD-consistent gradient the direct path computes — so a GN step is
  a tile read, not a render.
- Planes are stored as **cubic B-spline coefficients** (Unser's IIR prefilter)
  and reads evaluate the cardinal cubic spline (4×4 kernel; integer shifts
  reproduce texels exactly). The interpolator choice is load-bearing: bilinear
  reads' phase-dependent smoothing displaced the ECC optimum by up to
  ~0.065 px (pixel locking) and Catmull-Rom still left ~0.02–0.045 px on
  near-Nyquist content; the prefiltered spline keeps the planted-offset
  recovery inside the < 0.02 px target above. (A 2× supersampled bilinear
  tile was measured-and-rejected: it quadruples the prerender, the new
  dominant cost.)
- **Coarse-grid gate:** a tile is built only when the patch grid samples at
  least as densely as the source (≤ ~1.2 source px per grid px, estimated
  from the projected core corners). A coarser grid would freeze the sampling
  phase of above-grid-Nyquist source content that direct rendering samples
  continuously (measured as spurious displacement on a coarse-grid synthetic
  fixture), and a source-density supersampled tile costs more than the ~14
  direct renders it replaces — so coarse views (large-scale keypoints, ~28% of
  dino's pairs) simply keep the exact direct path.
- **Measured (dino, 85 imgs / 46k pts, 2 rounds):** subpixel CPU 596 → 401 s
  and wall 18.6 → 12.6 s (round 1); command wall 101 → 89 s. Membership and
  point survival vs the direct-render baseline are identical (churn 0.047%
  over two rounds, 0 in a single round); keypoint deltas are median 0.03 px
  with a ~2% tail > 0.5 px on weakly-determined patches — re-scoring both
  keypoint sets with the same scorer shows the tile run's final ECC is equal
  or better (mean +0.00015 overall; on the tail 88.9% of moved observations
  score *higher*, mean +0.0028), i.e. the smoother spline value/gradient
  fields let GN converge further where the u8-quantized direct evaluations
  stalled.

## Outputs

Per view: the refined sub-pixel `keypoint_v`, its offset from the projection, and
the final score. The view set is returned unchanged (a guard-failed view stays at
its seed, so set and ordering are preserved).

Per point (opt-in, `KeypointSubpixelParams::render_bitmaps` / the binding's
`refine_keypoints(render_bitmaps=True)`): the fused **representative RGBA
texture** (`R·R·4`), rendered at the **final** per-view keypoints and fused with
the final IRLS view weights — the same weighted-mean-RGB +
agreement·coverage-alpha fusion normal refinement uses (`PatchViewStack::fuse`,
shared across the two modules). Because the refiner settles the final keypoints,
this is where the pipeline's stored reference bitmaps come from (they previously
came from normal refinement and lagged the final sub-pixel refinement by one
round). Two properties matter to consumers:

- **The representative render uses the refine `sampler`** — the same knob the
  refine loop samples with (bilinear by default, anisotropic when requested). The
  stored texture and the cores the IRLS weights are scored against are then
  sampled the same way, so the fused reference bitmap matches the pixels that
  drove the refinement.
- **`None` is the uniform culled-point signal.** A point whose final-offset
  renders leave fewer than two usable views has no cross-view consensus and gets
  no representative — finite and infinity alike (a `w = 0` point renders through
  the same path and gets a *real* consensus bitmap, not a zero row).
  `sfm embed-patches` **drops** such points instead of keeping them with an
  all-black bitmap.

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
- Consensus refresh granularity: now measurable — all three variants are
  implemented. On real data (dino_dog_toy, 300 patches, 1369 views) per-sweep
  and per-move (shared T) are within noise at 5 sweeps (mean ECC 0.8716 vs
  0.8725); per-move at 1 sweep has a small edge (0.8610 vs 0.8584). The
  open sub-question is when per-move's slightly faster convergence is worth
  its small per-call overhead in production — a wiring/decision-gate question
  to revisit when the refiner is wired into `_embed_patches.py`.
  **Shared vs LOO**: measured-and-rejected. LOO consistently underperformed
  shared T on real-data tracks (mean ECC 0.82 vs 0.87 at 5 sweeps) because at
  N ≈ 3–5 views LOO drops effective averaging enough that the per-view
  template is noisier than shared, and the within-sweep LOO chain amplifies
  drift more than the self-pollution it removes. The self-pollution at
  sub-pixel scale turns out to be a damping term that helps. LOO remains a
  one-line cost change if a future large-N measurement shifts the verdict.
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
   **divide** by the level's `2^level` (i.e. multiply by `1/2^level`) to express it
   in level-0 source px (`x_level = x_0 / 2^level` ⇒
   `∂I/∂x_0 = (∂I/∂x_level) / 2^level`), and blend the two levels with the same
   `frac` the value uses. (Finite differences across a mip boundary would not be
   LOD-consistent — this is the whole reason to compute it inside the sampler.)

3. **Warp-Jacobian accessor** — expose the raw per-pixel `J`
   (`WarpMap::get_jacobian(col, row) -> [[f32; 2]; 2]`); `compute_svd` already
   computes it internally, so this is just surfacing it (or reconstruct
   `J = U·S·Vᵀ` from `get_svd`). The refiner composes `∇_src I · J` → `∂I/∂δ`.

Per support pixel/channel the refiner gets `(value, ∂I/∂δ)` by calling #1 or #2
(per the `Sampler`) and composing with #3; value-only callers keep the existing
`remap_*` functions unchanged.

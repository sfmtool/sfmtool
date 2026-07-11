# Patch Localizability (Keypoint Self-Similarity Score)

_Status: **implemented** (v1). A per-point score of how well a patch pins its own
keypoint — the curvature of the patch's ZNCC self-similarity surface, i.e. the
classic structure tensor measured in the pipeline's own correlation metric. It
grades the **conditioning** of a keypoint's localization (corner vs edge vs
flat), independently of whether the views agree. Prototyped end-to-end on five
datasets (see [Evidence](#evidence)); this spec is the design for the production
crate function, Python binding, `xform` filter, and `embed-patches` cull._

## Problem

Keypoint [localization](patch-keypoint-localization.md) and
[sub-pixel refinement](keypoint-subpixel-refinement.md) place each observation's
keypoint on image content and report a **leave-one-out ZNCC** — "do the views
agree?". That signal is blind to a distinct failure mode: a patch straddling a
straight **edge** can have LOO ZNCC ≈ 1 (every view sees the same edge and
registers perfectly) while the keypoint is free to **slide along the edge** — the
classic aperture problem. A **flat / low-contrast** patch is worse still: nothing
constrains the keypoint in any direction. Neither is caught by the existing
agreement gate (`min_relative_zncc`), because agreement and localizability are
different questions. Poorly-localized points inject positional noise into the
reconstruction; we want a signal to score and cull them.

## What it measures

Compare a patch against itself shifted by `Δ` and watch the ZNCC. Near `Δ = 0`,

```
1 − ZNCC(Δ)  ≈  ½ Δᵀ H Δ ,        H = M / (W · σ²)
M = Σ_k w_k · ∇I_k ∇I_kᵀ         (the 2×2 structure tensor over the R×R window)
```

where `∇I_k` is the image gradient at window pixel `k`, `w_k` the window weight,
`W = Σ_k w_k`, and `σ²` the window-weighted patch variance. `M` is the classical
second-moment / structure tensor [[Harris & Stephens 1988](#references)]. Its two eigenvalues
`λ₁ ≥ λ₂ ≥ 0` and the eigenvector of `λ₂` classify the keypoint:

| patch | ZNCC falloff | `λ₁, λ₂` | localizes? |
|---|---|---|---|
| corner / blob | steep every direction | both large | **yes** — pinned in 2D |
| edge / ridge | steep across, flat along | `λ₁` large, `λ₂ ≈ 0` | **no** — slides along edge |
| flat / low-contrast | flat everywhere | both `≈ 0` | **no** — unconstrained |

`λ₂` (the **Shi–Tomasi** score [[Shi & Tomasi 1994](#references)], the weaker axis) is the single "does this pin a
2D position" number; the **eigenvector of `λ₂` is the slide direction** — the axis
of greatest positional ambiguity, useful for diagnostics and (future) per-view
uncertainty propagation.

The analytic structure tensor is the small-shift limit of the literal
shift-and-score surface, so one gradient pass reproduces the whole
autocorrelation picture — **no shifting or search** (measured: eigenvalue
correlation +0.85 to +0.95 against the discrete surface, [Evidence](#evidence)).

### Where it is measured

On each point's rendered **cross-view consensus** patch — the fused
representative texture `embed-patches` already produces
(`refine_keypoints(render_bitmaps=True)`, stored as `patch_bitmaps`). This is the
right frame and the right input: it is exposure-normalized (per-channel
z-normalized like the rest of the pipeline), denoised by robust averaging, and
already paid for. The structure tensor is accumulated over the consensus channels
(matching the pipeline's per-channel ZNCC); luminance is an acceptable
simplification. Because the score derives from
`patch_bitmaps` alone, **it needs no source images** — a whole solve scores in
seconds (530k points in 30 s, [Evidence](#evidence)).

**Coverage / the alpha channel — not used.** Uncovered consensus pixels are hard
zero-filled in RGB *and* alpha (`PatchViewStack::fuse`), so in principle a gradient
over raw RGB could pick up a spurious step-edge at a coverage boundary. In practice
that boundary is essentially never inside the scored `gaussian_disk` support:
measured across the four repo datasets, **99.8–100 % of scored patches are fully
covered within the disk** (partial coverage 0.0–0.2 %), because the surfel fills the
patch and under-covered points are already dropped by the `<2`-view / validity cull.
The scorer therefore computes the structure tensor on RGB and **ignores the alpha
channel** — no per-pixel masking. (If a future change scores a wider window or
admits partially-covered patches, revisit: the fix would be to zero the window where
`alpha == 0`.)

## Two normalizations, and why noise-normalized wins

The tensor can be normalized two ways, and the choice is load-bearing:

- **Variance-normalized** `λ₂ / σ²` — contrast-*invariant*; scores the *shape* of
  the texture. Great at edge/flat-shape discrimination, but by construction blind
  to contrast: a faint corner and a bold one score identically. It therefore
  **cannot see the "low-contrast, ZNCC stays flat" case** — ZNCC divides contrast
  out.
- **Noise-normalized** `λ₂ / σ²_noise` — the actual **localization covariance**
  `Cov(δ) = σ²_noise · M_sum⁻¹` (`M_sum = W · M`; the classical Förstner precision
  ellipse [[Förstner & Gülch 1987](#references)]). Its eigenvalues come out as
  **positional uncertainty in pixels**:

  ```
  σ_pos = σ_noise / √λ₂_sum          (px, along the weak axis; the worse of the two)
  ```

  A faint corner correctly scores *worse* than a bold one, because its signal
  sits closer to the noise floor even when its shape is identical.

The two differ by exactly the patch contrast (`noise = variance × σ²/σ²_noise`),
so they agree on the dominant bad population (edges/flats: Spearman ≈ 0.75–0.81)
and diverge only on the faint patches — the ~2–4% the variance form waves through
([Evidence](#evidence)). **We use the noise-normalized form.** It is the
theoretically correct precision metric, it captures all three failure modes, and
— decisively — it expresses the score as **σ_pos in pixels** (patch-grid px), a
physical unit that transfers across datasets (see [Threshold](#threshold)).

### σ_noise (v1: global constant)

`σ_noise` is the consensus photometric noise in intensity units. **v1 uses a
single calibrated global constant** (order a few gray levels for u8 consensus).
Consequences and rationale:

- The precision is then **purely intrinsic** — the per-reconstruction *ranking*
  is by `λ₂_sum` (the contrast-carrying raw Shi–Tomasi eigenvalue); `σ_noise` only
  sets the absolute pixel scale for the threshold.
- It is **orthogonal to cross-view agreement**, so it composes cleanly with the
  existing `min_relative_zncc` LOO gate — the two cull for different reasons.
- One number, no per-point plumbing.

_Extension (not v1):_ estimate `σ_noise` **per patch** from the congealing
leave-one-out residuals the localizer already computes. That yields
precision = structure / (noise + disagreement) — more correct, but it re-couples
the score to agreement and needs the residuals plumbed from the localizer into
the scorer. Deferred; see [Open questions](#open-questions).

### Grid pixels are the cull unit

`M` is computed on the `R×R` patch grid, so `σ_pos = σ_noise / √λ₂_sum` is
natively in **grid** px — a fraction of the patch, independent of how large that
patch projects into any image. **This grid-px `σ_pos` is the cull quantity.**

We want the score to be a property of the **patch itself**, not of how it happens
to project into any view — a measure of the consensus texture's own
localizability. Grid px is exactly that: it depends only on the consensus
appearance (contrast/structure vs. noise), so a single `τ` behaves comparably
across datasets of different resolution ([Threshold](#threshold)). Mapping `σ_pos`
into source-image px via the projected scale (`half_extent / (R/2) · f / depth`,
median over the kept views) reintroduces a per-view focal/depth factor that makes a
fixed `τ` resolution-dependent; that source-px value is computed and exposed as a
**diagnostic** but is not the cull quantity.

## Threshold

The cut is an **absolute `σ_pos` threshold `τ` in grid px**, supplied by the
caller — modelled on `--filter-by-reprojection-error <px>`: drop points with
`σ_pos > τ`. This is the simple, robust choice:

- **Idempotent.** `σ_pos` is a per-point quantity, so after one pass every
  survivor is `≤ τ`; re-running drops nothing. (A pure filter is exactly
  idempotent; a BA between passes nudges `σ_pos` slightly.)
- **No stored state.** `τ` lives in the command/pipeline, like the reprojection
  filter's threshold — nothing is persisted in the `.sfmr`. `σ_pos` is recomputed
  on demand from the current `patch_bitmaps`, so it never goes stale, and — being a
  pure grid-px quantity — it is invariant under both similarity transforms and
  resolution.

**`τ` transfers reasonably in grid px, but is not perfectly universal.** Calibration
([Evidence](#evidence)) measured the cull fraction at a *matched* `τ` across the
repo datasets: **~7–12×** spread in grid px versus **~50–100×** in source px (or
angular). Grid px is decisively the better unit — a single `τ` culls a comparable
order-of-magnitude fraction everywhere, where a source-px `τ` is inert on
low-resolution data and a wrecking ball on high-resolution data. A residual ~7–12×
spread remains because a dataset with a genuinely heavier flat/edge tail (e.g. lots
of smooth texture) really does have more poorly-localized points — which is the
*intended* behaviour of a **conservative tail threshold**: set `τ` out in the tail
and it removes egregious points *where a dataset has them* and little where it
doesn't. Across the repo datasets `τ ≈ 0.3–0.35` grid px culls ~1–3 % on
well-textured low-resolution sets and ~7–12 % on a high-resolution set with a
smooth-texture tail. The tool reports the `σ_pos` distribution to help fine-tune
`τ`; it never auto-derives one.

`embed-patches` uses a **conservative default `τ = 0.35` grid px** (overridable) —
a safe tail cut in the spirit of the reprojection filter's default. It self-limits,
so it will not hurt datasets that have no poorly-localized tail. (Grid px is
relative to the patch grid `R`, so `τ` is calibrated for the default `R = 24`; a
run that changes `resolution` should re-pick `τ` — it does not rescale
automatically.)

## Surfaces

One scorer, three entry points (plus an internal consumer: the
[cluster-patch refinement](cluster-patch-refinement.md) kernel gates each
cluster member on the localizability of its own template-grid patch —
`max_keypoint_uncertainty`, same default `τ`, status
`rejected_unlocalizable`; added 2026-07-10):

1. **Crate function** (`crates/sfmtool-core/src/patch/localizability/`, a new
   submodule sibling of `keypoint_localize` / `normal_refine`):

   ```
   fn patch_localizability(patch: &[f32] /* R×R×C consensus */, resolution, channels,
                           window: &Support, sigma_noise: f64) -> Localizability
   struct Localizability { lam1: f64, lam2: f64, theta: f64, sigma_pos_grid: f64 }
   // public batch entry (builds Support internally, rayon-parallel):
   fn score_localizability_stack(patches, num_patches, resolution, channels,
                                 window: PatchWindow, sigma_noise) -> Vec<Localizability>
   ```

   Pure and standalone (structure tensor + 2×2 eig). The batch entry scores a
   `(P, R, R, C)` stack (rayon-parallel). Reuses the existing `Support` window from
   `normal_refine`. (`patch_localizability` itself is `pub(in crate::patch)` since
   `Support` is crate-private; the batch entry takes a `PatchWindow` and is the
   public surface.)

2. **Python binding** — `PatchCloud.score_localizability(recon, patch_bitmaps, …)`
   scores the batch over `patch_bitmaps`, returning per-point
   `{lam1, lam2, theta, sigma_pos_grid, sigma_pos_px}`. `sigma_pos_grid` is the cull
   quantity; `sigma_pos_px` (source px, via the recon-geometry grid→px map, median
   over views) is a diagnostic. No images required.

3. **`xform` filter** — exposed as `--filter-by-keypoint-uncertainty <grid-px>`
   (the same "keypoint uncertainty" wording as the embed flag), implemented by
   `FilterByLocalizabilityTransform` in
   `src/sfmtool/xform/_filter_by_localizability.py` (the internal class keeps the
   `localizability` concept name), wired like
   [`--filter-by-reprojection-error`](../cli/xform-command.md): compute per-point
   `σ_pos` (grid px) from the recon's `patch_bitmaps`, keep-mask `σ_pos ≤ τ`,
   delegate to `recon.filter_points_by_mask(...)`. Runs offline on any
   `embedded_patches` recon (no images), composes in the pipeline, and is the
   **re-runnable home of the cull policy** — retune `τ` without re-embedding.
   Errors if the recon has no `patch_bitmaps` (needs a consensus to score).

## `embed-patches` integration

`embed-patches` culls poorly-localized points **by default, early** — after the
first round's localization + sub-pixel refine, before the multi-round refinement
that dominates cost (see the [pipeline](../../src/sfmtool/_embed_patches.py)).

**Why early culling is safe here.** Localizability is **intrinsic and per-point
independent**: a point's score depends only on its own consensus appearance, and
culling one point does not change any other point's consensus (unlike
view-dropping, which reshapes the consensus the survivors register against). So an
early cull has **no feedback effects** — it simply removes doomed points from the
working set, and the truly-unlocalizable cases (flat / edge) are intrinsic and
will not be rescued by further normal/keypoint refinement.

**Placement (v1).** In `embed_patches`, right after the round-1 sub-pixel pass:

1. Render the round-1 fused consensus bitmaps (enable `render_bitmaps` at round 1;
   today they are only rendered on the final round for a multi-round run).
2. Score `σ_pos` per point on those bitmaps (the same scorer the `xform` filter
   uses).
3. **Drop `σ_pos > τ` points from the `localizations` list** before rounds 2..N
   (and before the final compaction for a single-round run). The compaction
   renumbers survivors, so removing a point from `localizations` propagates
   cleanly — no separate mask plumbing. Culled points are then absent from every
   subsequent refinement pass **and** the output.

A `--max-keypoint-uncertainty` CLI option carries `τ` (the largest predicted
keypoint position uncertainty, in **grid px**, to keep — default `0.35`,
overridable; `0`/disabled opts out). The final `compact_to_embedded_patches`
keeps its existing `min_views` / validity cull unchanged; **no second
localizability cull is needed** — the early one is authoritative.

**Cost — measured net-neutral to favourable.** Enabling round-1 bitmaps adds one
render pass over the full point set (~+1.6 s / round-1 render + a sub-second score
on a 55 k-point set), offset by shrinking rounds 2..N and the final render to the
survivors. On dino (55.7 k pts, 8 % cull) end-to-end wall time went 99 s → 91 s;
on the small low-resolution sets the difference is sub-second noise. Enabling the
cull never costs runtime.

**Mis-cull caveat.** At round 1 the normal is refined only once, so a genuinely
good point whose normal has not converged renders a slightly-blurred consensus and
scores lower than it eventually would. A corner still reads as a corner, so a
**not-overly-tight `τ`** avoids dropping refinable points; the calibration
experiment should compare early-vs-final-round scores to bound how many points
change side of `τ` between the two (see [Open questions](#open-questions)).

**Further optimization (deferred).** Culling *before* the round-1 sub-pixel pass
(scoring a render-only consensus fuse at the localizer's keypoints) would also skip
that pass on doomed points, at the cost of an extra render-only fuse. Weigh once
the round-1-vs-final mis-cull delta is measured.

## Parameters (defaults)

| parameter | default | meaning |
|---|---|---|
| `resolution` (R) | 24 | patch grid the tensor is computed on (matches the consensus) |
| `window` | `gaussian_disk` | scoring window (shared with the rest of the pipeline) |
| `sigma_noise` | ~3 gray levels (global constant) | sets the absolute px scale of `σ_pos`; only the *ranking* is scale-free |
| `max_keypoint_uncertainty` (`τ`) | `0.35` grid px | drop points with `σ_pos > τ` (**grid px** — transfers across resolution, see [Threshold](#threshold)); conservative self-limiting tail cut |

## Evidence (prototype)

Measured (throwaway prototype scripts, not committed) on the five embedded
datasets (dino_dog_toy, seattle_backyard, kerry_park, seoul_bull, and
DinoLedge — 530,674 points / 1196 images, scored in 30 s):

- **A genuinely new signal.** `corr(localizability, cross-view agreement)` ranges
  −0.28 … +0.02 — uncorrelated everywhere. Localizability and agreement measure
  different things.
- **The blind spot is real, ~1 in 7.** 11–15% of points are *high-agreement AND
  low-localizability* — the aperture case the `min_relative_zncc` gate passes but
  that still slides. The montages make it literal: on seattle's architectural
  edges the computed slide-axis lies exactly along the edge.
- **The cheap analytic form is the real thing.** Structure-tensor eigenvalues
  correlate +0.85 … +0.95 with the discrete shift-and-score ZNCC surface.
- **Noise-norm adds the faint case.** It agrees with the variance form on the
  bulk (Spearman 0.75–0.81) and uniquely rejects an extra 2–4% — unambiguously the
  low-contrast patches (median contrast 0.1–0.3× the rest), with ~2× worse
  `σ_pos` — while reframing the score as physical px uncertainty.

### Calibration (held-out cross-validation)

Method (throwaway harness, not committed): hold out observations of trusted
(best-half-`σ_pos`) points, cull the worst-half pool by {localizability /
reprojection-error / random} at a matched count, bundle-adjust, and measure
reprojection error on the held-out observations (never in the BA), 4 seeds. On
the four repo datasets:

- **The cull helps, and it's a distinct signal.** Ordering **loc ≥ reproj ≥
  random** held on every dataset, so `σ_pos` removes genuinely harmful points, not
  just "any" points — and it beats the existing reprojection filter on 3 of 4
  (a tie on the fourth). The effect is small on three datasets (1–2% over ~4
  seeds), so treat that ordering as directional support, not a strong result; the
  load-bearing evidence is the one dataset with a real bad-point population:
- **Effect is dataset-dependent.** Large on the fisheye rig (kerry_park:
  held-out error −9% at ~15–20% cull, ~−7 pts vs random), small on
  dino (~−2%), marginal on seattle/seoul. The benefit concentrates where
  poorly-localized points cluster (lens distortion, low texture). _Caveat:_ on the
  fisheye set the poorly-localized points are also the image-periphery points, and
  the calibration did not isolate localizability from a plain radius/distortion
  effect — some of the kerry_park win may be "drop the periphery," not purely
  "drop the poorly-localized." A radius-controlled arm is the follow-up.
- **Grid px transfers across resolution.** The `σ_pos` *median* spans ~2.6–3.6×
  across datasets in every unit (grid px 2.6×, source px 3.4×, angular mrad 3.6×),
  but what matters for a threshold is the **cull fraction at a matched `τ`**: that
  spread is **~7–12× in grid px** versus **~50–100× in source px**, where the
  low-resolution sets flatline at 0 % culled while a high-resolution set still culls
  double digits. The residual ~7–12× spread in grid px is genuine — a heavier
  flat/edge tail really does mean more poorly-localized points — so `τ` stays a
  conservative caller-set tail cut, not a universal constant ([Threshold](#threshold)).

## Validation

- **Synthetic.** Corner / edge / blob / flat / pure-noise patches: `λ₂` ranks
  them as tabled; the weak eigenvector aligns with the edge slide direction.
  _Done: `crates/sfmtool-core/src/patch/localizability/tests.rs` asserts the λ₂
  ranking, the edge slide-axis alignment, the analytic linear-ramp tensor, and a
  numeric match against the Python prototype._
- **Analytic ≈ empirical.** Structure-tensor eigenvalues match the discrete
  ZNCC-shift surface (as measured above) — the equivalence that justifies the
  one-pass form.
- **Cull tightens the reconstruction — done, positive but modest.** Held-out
  cross-validation ([Calibration](#calibration-held-out-cross-validation))
  confirmed the premise: culling by `σ_pos` improves held-out reprojection and
  beats random / reprojection-error controls, most on distortion-heavy data. It
  did **not** yield a universal `τ` (none exists — the distributions differ per
  dataset), which is why `τ` is a conservative caller-set cut. Still open: the
  **round-1 vs final-round** score delta (Open question 2), to confirm the early
  cull does not drop points a converged consensus would have kept.
- **Diagnostic montage.** Worst/best-localized patches with the slide-axis drawn
  (the prototype montages) — a human check that the ranking is sane per dataset.

## Open questions

1. **Per-patch `σ_noise` from LOO residuals** — folds disagreement into precision
   (more correct) at the cost of re-coupling to agreement and plumbing residuals
   from the localizer. Worth it? (v1 is the global constant.)
2. **Round-1 vs final-round mis-cull delta** — the v1 cull scores on the round-1
   consensus (normal refined once). How many points change side of `τ` between the
   round-1 score and the fully-converged final-round score? If non-trivial, either
   loosen the early `τ` (drop only clearly-hopeless early) and add a tight final
   pass, or move the cull before round-1 sub-pixel (the deferred optimization
   above). Measure in the calibration experiment.
3. **Persist the score in the `.sfmr`?** v1 computes on demand from
   `patch_bitmaps` (no format change). A persisted per-point `keypoint_uncertainty_px`
   would enable GUI coloring and repeat-filtering without recompute, at the cost
   of a format version bump. Add when a consumer needs it.
4. **Per-view image-space uncertainty.** Propagating `M` through each view's warp
   Jacobian yields a per-observation uncertainty ellipse (and could weight the
   bundle), vs the single intrinsic per-point score here. Extension.
5. **Channel treatment.** _Settled (v1): luminance_ (`0.299R + 0.587G + 0.114B`),
   RGB only, alpha ignored. Cheap, prototype-validated, and the coverage
   measurement made per-pixel alpha masking a no-op. Per-channel z-normalized
   summation stays a deferred extension. _Amendment (2026-07-10):_
   sub-3-channel input is grayscale — channel 0 **is** the luminance
   (weight 1.0; applying only Rec.601's red weight would shrink gradients
   ~3.3× and inflate `σ_pos` by the same factor for identical content).
   Consensus stacks are always ≥ 3 channels, so this only affects
   single-channel callers (the cluster-refine gate on grayscale images).
6. **`σ_pos` reduction over views.** Median chosen; mean / worst-case are
   alternatives if a specific view's precision should dominate.

## References

The three classical results this score composes (quotes below were verified
against the PDFs; the Förstner precision result via a standard secondary source):

- **`λ₂` as the feature-quality score ("Shi–Tomasi").** J. Shi and C. Tomasi,
  "Good Features to Track," IEEE CVPR 1994, pp. 593–600. Selects a window as a good
  feature by the **smaller** eigenvalue of the second-moment matrix `Z`: §4.1 states
  it verbatim — "if the two eigenvalues of `Z` are `λ₁` and `λ₂`, we accept a window
  if `min(λ₁, λ₂) > λ`." Freely available as Cornell TR 93-1399 (Dec 1993, identical
  content): <https://users.cs.duke.edu/~tomasi/papers/shi/TR_93-1399_Cornell.pdf>.
  Origins in C. Tomasi and T. Kanade, "Detection and Tracking of Point Features,"
  CMU-CS-91-132, 1991.
- **The structure tensor / cornerness.** C. Harris and M. Stephens, "A Combined
  Corner and Edge Detector," 4th Alvey Vision Conference, 1988, pp. 147–151. Defines
  the second-moment matrix `M = Σ w ∇I∇Iᵀ` (its eigenvalues "proportional to the
  principal curvatures of the local auto-correlation function") and the response
  `R = Det(M) − k·Tr(M)²`. PDF (BMVA archive):
  <https://bmva-archive.org.uk/bmvc/1988/avc-88-023.pdf>.
- **Localization covariance `σ²·M⁻¹` — the error ellipse behind noise-normalized
  `σ_pos`.** W. Förstner and E. Gülch, "A Fast Operator for Detection and Precise
  Location of Distinct Points, Corners and Centres of Circular Features," ISPRS
  Intercommission Conference on Fast Processing of Photogrammetric Data, Interlaken,
  1987, pp. 281–305. Estimates the point by least squares from the structure tensor
  and gives its **precision as the inverse of that matrix** (position covariance
  `∝ M⁻¹`) — exactly this spec's `Cov(δ) = σ²_noise · M_sum⁻¹`. PDF:
  <https://cseweb.ucsd.edu/classes/sp02/cse252/foerstner/foerstner.pdf>.

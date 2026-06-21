# Photometric Patch-Normal Refinement

**Status:** Builds on `specs/core/patch-cloud.md`
(`OrientedPatch`, `WarpMap::from_patch`, `remap_*`).

> _Status (2026-06-12): v1 implemented — `sfmtool-core/src/patch_normal_refine.rs`
> (`ProjectedImage`, `Objective`, `PatchWindow`, `NormalRefineParams`,
> `NormalRefineResult`, `refine_patch_normal`, `refine_patch_cloud`,
> `patch_view_indices_from_reconstruction`) and the PyO3 binding
> `PatchCloud.refine_normals`. Deferred from v1: the analytic Gauss-Newton polish
> and its centered-Hessian confidence (v1 uses a finite-difference grid-curvature
> confidence and does not yet gate on a confidence threshold to keep the init
> normal); the geometric/PCA seed (v1 seeds from the patch's current normal and
> the mean-viewing direction); stochastic view subsets; and the
> atlas-backed textured patch cloud. The `scripts/patch_crossval.py` strip
> renderer drives this routine via `--refine-normal`._
>
> _Status (2026-06-20): the `representative` texture output is now produced — a
> per-patch fused RGBA bitmap rendered at the found normal, gated on the new
> `NormalRefineParams::render_bitmap` (Python `refine_normals(render_bitmaps=…)`),
> scattered to per-3D-point rows and persisted as `patch_bitmaps_y_x_rgba` by
> `sfm xform --refine-normals bitmaps=true`. RGB is the cross-view (robust)
> fusion; alpha is per-pixel cross-view agreement. The final scoring pass renders
> each candidate's per-view stack (`PatchViewStack`) and keeps the winner's, so
> the bitmap and the consensus view-weights it fuses with come from that one
> render — no extra render or IRLS pass. This is the simple fused-render form; the
> **joint normal + per-pixel robust template** of item 7 (a free latent `m`,
> super-resolvable, atlas-backed) remains beyond v1._
>
> _Status (2026-06-13): performance characterized — see
> `reports/2026-06-13-perf-patch-normal-refinement.md` (phase breakdown,
> per-knob perf-vs-benefit, prioritized optimization list). The search defaults
> hold up; the **sampler default is now `Bilinear`** (the analysis found
> anisotropic barely changes the found normal at 1.6–3× the cost — it stays an
> opt-in for unbiased `Φ`/confidence). Landed behavior-preserving fixes: small
> warp/remap grids run sequentially instead of nested-rayon, and `remap_aniso`
> skips zero-weight hi-level taps. `SFMTOOL_PROFILE=1` enables hot-path phase
> timers; `scripts/bench_normal_refine.py` and the `patch_render` criterion bench
> reproduce the measurements._

## Problem

A reconstructed 3D point `X` is seen by cameras `{(Kᵢ, Tᵢ)}`. Its surface around
`X` is locally a small plane — the surfel. `PatchCloud::from_reconstruction`
gives that surfel an *initial* normal (e.g. the mean viewing direction), which is
typically the camera-facing plane, not the true surface plane. We want the normal
`n` that maximizes **photometric consistency** across the views: the plane through
`X` whose rendered patches agree the most.

This is the planar-patch case of multi-view stereo: for a pinhole camera the
patch→image map is a homography. We use the general per-pixel projection
(`WarpMap::from_patch`), so distortion / fisheye work unchanged.

## Degrees of freedom

The normal has **2 degrees of freedom**, and that is all we optimize — it is the
only thing that changes which 3D plane is sampled, hence the only thing that
affects cross-view consistency.

Everything else is fixed: the patch center, its size, and the rotation about the
normal. The rotation can't affect the score (it rotates the `(s, t)` grid the same
way in every view), so it needs no `up` hint — the input patch already carries
`u`/`v`, and the routine preserves that frame as closely as possible.

So refinement is a **2-DOF search on the sphere** around the initial normal.

## Objective

Photoconsistency `Φ(n)` over the patches `{pᵢ(n)}` rendered into each view. We
want a single, well-defined scalar — not an ad-hoc aggregation — so we build it
from a **consensus**.

**Per-view normalized patch.** Render view `i` under `n`, restrict to the
commonly-valid pixels, and z-normalize each colour channel independently over the
windowed support: subtract the windowed mean, divide by the windowed norm, giving
a unit-norm, zero-mean vector `xᵢ` (per channel). Independent per-channel
normalization makes the score invariant to a *per-channel* affine (gain/offset) —
robust to per-camera white-balance and exposure while still using chrominance
(stacking the channels into one vector instead assumes a single shared affine and
is less white-balance robust). Census or mutual information handle non-affine
changes.

**Consensus = all-pairs mean ZNCC, in one sweep.** With `x̄ = (1/V) Σᵢ xᵢ`, the
mean pairwise ZNCC over all `C(V, 2)` view pairs has a closed form:

```
ρ̄(n) = (V·‖x̄‖² − 1) / (V − 1)       (averaged over channels)
```

because `Σ_{i<j} xᵢ·xⱼ = ½(‖Σxᵢ‖² − V)`. Equivalently, since `‖xᵢ‖ = 1`, the
across-view variance is `Var = 1 − ‖x̄‖²` and `ρ̄ = 1 − V·Var/(V−1)` — so
**maximizing the all-pairs mean ZNCC is exactly minimizing the photometric
variance of the z-normalized stack.** (`‖x̄‖² ∈ [0, 1]`, so `ρ̄ ∈ [−1/(V−1), 1]`:
the floor is `V`-dependent and uncorrelated views score ≈ 0, not −1, so any
absolute `Φ` threshold or keep-vs-init margin must scale with `V`.)

This makes the *full all-pairs* objective as cheap as a reference-view one.
Evaluating `Φ` is dominated by **rendering the V patches** — every objective
renders all V views, a reference no fewer — and the consensus aggregates them in
one sweep (sum the normalized patches, take `‖x̄‖²`). So we get a symmetric,
reference-free `Φ` over all pairs at no extra cost, and no reason to pick a
reference view.

**Robustness (occlusion) via weighted consensus.** A few occluded / wrong-surface
views shouldn't drag the optimum. Use a weighted consensus `x̄_w = Σ wᵢ xᵢ`
(`Σwᵢ = 1`); weighting pair `(i, j)` by `wᵢwⱼ`, the weighted mean-pairwise
correlation keeps the same single-sweep form:

```
ρ̄_w = (‖x̄_w‖² − Σ wᵢ²) / (1 − Σ wᵢ²)
```

Set `wᵢ` by IRLS from each view's residual `‖xᵢ − x̄_w‖` (a robust M-estimator —
e.g. Tukey, with a scale from the residual MAD), re-forming `x̄_w` and
re-weighting a few times. This stays a smooth consensus while down-weighting
outliers, instead of a non-smooth median over pairs. The view *count* is gated by
`min_views` (and `min_valid_fraction`) when forming the support; separately, gate
the robust *effective* view count `1/Σwᵢ² ≥ 2` — a pure degeneracy floor, since
as weight concentrates on one view `Σwᵢ² → 1` and `ρ̄_w → 0/0`. (Don't reuse
`min_views` for this: `1/Σwᵢ² ≤ V` with equality only for exactly uniform weights,
so a clean `V == min_views` track would be falsely rejected.)

**Validity.** A candidate normal can project the patch partly out of frame (NaN)
or behind a camera in some views. Score only over commonly-valid pixels; require a
per-view minimum valid fraction and a minimum number of valid views, else mark the
view (or the whole candidate) invalid. Two subtleties the identity above assumes:

- **Common support per channel and per view.** All views' windowed mean/norm must
  use the *same* pixel set (window × validity); otherwise the inner products don't
  live in one space and the closed form breaks. Add an epsilon (or a per-channel
  validity rule) for flat channels whose windowed norm is ~0.
- **Freeze the mask per grid level.** The common-valid set depends on the
  candidate `n`, so scoring each candidate over its own support makes `Φ`
  discontinuous and biases the argmax toward tilts that shrink the support onto an
  easy region. Compute the mask once at the level's center normal and hold it
  fixed across that level's candidates.

## Current prototype algorithm

```
refine_normal(center, init_n, half_extent, views, images):
    u, v = tangent_basis(init_n)        # search basis; any tangent basis of init_n
    grid = linspace(-tan(range), tan(range), steps)        # default range=25°, steps=7
    best = (init_n, Φ(init_n))
    for a in grid, b in grid:                              # steps² candidates
        n = normalize(init_n + a·u + b·v)
        if Φ(n) > best.score: best = (n, Φ(n))
    return best
Φ(n):  render patch under n into each view (from_patch + remap_bilinear at the
       validated resolution); return mean pairwise windowed-NCC.
```

`tangent_basis(init_n)` is only the **search** basis — the two directions the grid
tilts `n` — not the patch's in-plane orientation. The prototype still passes an
`up` to `OrientedPatch.from_center_normal` when rendering, but since in-plane
rotation can't affect `Φ` it's immaterial; the core API drops it.

Two requirements on the search basis:

- **Deterministic in `n`.** `tangent_basis` must be a pure function of `init_n`
  (e.g. the prototype's least-aligned world axis + Gram-Schmidt), so a refinement
  is reproducible. The continuous optimum is basis-independent anyway; on a finite
  grid the basis only rotates the sampling lattice, which coarse-to-fine and the
  gradient polish wash out.
- **Square-grid caveat.** `n = normalize(init_n + a·u + b·v)` is a flat (gnomonic)
  projection: equal `(a, b)` steps are *not* equal angles, and the square
  `[-tan r, tan r]²` reaches `atan(√2·tan r) ≈ √2·r` into the corners. Fine for a
  modest cone; for
  wide cones prefer clamping to the disk `‖(a, b)‖ ≤ tan r` (it also matches the
  radial `GaussianDisk` support and keeps the per-level cone circular).

Single-level dense grid; `steps²·V` renders per point. On seoul it lifts mean
pairwise NCC by ~0.03 on average and up to +0.17 on the least-consistent tracks.

## Optimization

The proposed core replaces the single dense grid with a parameterization and a
two-stage search; the objective is the consensus `Φ` above.

**Parameterization (`δ ∈ ℝ²`, exp-map).** Perturb the normal by a tangent vector
`δ` via the exponential map `n(δ) = cos‖δ‖·n₀ + sin‖δ‖·δ̂` (with `δ` expressed in
the deterministic tangent basis) — i.e. tilt `n₀` by angle `‖δ‖` toward `δ`
(equivalently a rotation about axis `n₀ × δ̂`). This is angle-uniform — equal steps
are equal angles, fixing the corner stretch of
`normalize(n₀ + a·u + b·v)` — and is the natural coordinate for both the grid and
the gradient step. The search domain is the disk `‖δ‖ ≤ angular_range`.

**Stage 1 — global (derivative-free).** Coarse-to-fine grid over the `δ`-disk:
a coarse grid, recenter on the best, shrink the cone, repeat `refine_levels`
times. Handles the multi-modality of `Φ` under repetitive texture; seed from
**multiple inits** (mean-viewing and geometric/PCA normals) and keep the best
basin. Renders per point ≈ `seeds · refine_levels · init_steps² · V`, far below
one dense grid of the same precision.

**Stage 2 — local polish (Gauss-Newton / LK).** `Φ` is smooth, and minimizing the
photometric variance `Σ wᵢ‖xᵢ(δ) − x̄_w‖²` (weights frozen within the step) is a
nonlinear least-squares problem, so a Gauss-Newton step in `δ` converges in 1–3
iterations from the stage-1 basin. Because `Σ wᵢ(xᵢ − x̄_w) = 0`, treating `x̄_w`
as fixed gives the *exact* gradient `∇E = 2 Σ wᵢ Jᵢᵀ(xᵢ − x̄_w)`. The `P×2`
per-view Jacobian chains

```
Jᵢ = ∂xᵢ/∂δ  =  (z-normalize)′ · ∂image/∂pixel · ∂pixel/∂world · ∂world/∂δ
```

— image gradient, the **full `2×3`** projection Jacobian `∂pixel/∂world`, and
`∂world/∂δ = ∂(patch point)/∂n · ∂n/∂δ`. Note `∂pixel/∂world` here is *not* the
`remap_aniso` SVD (that is the in-plane `2×2` map); under a tilt a patch point
moves *out of plane*, so the dominant column is the one the in-plane map omits —
use the camera model's analytic projection Jacobian, or finite-difference `δ`
(2 extra renders/view per step). The z-normalization derivative projects out the
mean/scale directions. (Inverse-compositional LK could precompute steepest-descent
images, but the template `x̄_w` and the weights change each iteration, so the
symmetric multi-view form is a research note, not a v1 plan.) GN is optional for
v1 — the coarse-to-fine grid alone is usable.

**Confidence.** Use the **centered** Gauss-Newton Hessian
`H̃ = Σ wᵢ JᵢᵀJᵢ − J̄ᵀJ̄` (with `J̄ = Σ wᵢ Jᵢ`), i.e. the *between-view* curvature
— not `Σ wᵢ JᵢᵀJᵢ`. This matters: in the narrow-baseline degeneracy we want to
flag, all views nearly coincide, so every `Jᵢ ≈ J̄`, tilting the plane shifts all
patches identically, and `Φ` is genuinely flat — `H̃ ≈ 0`, while `Σ wᵢ JᵢᵀJᵢ`
stays *large* on any textured patch and would falsely report high confidence. Its
smaller eigenvalue measures how tightly the normal is constrained; report
`confidence` from it (normalized for scale — eigenvalues grow with texture
contrast, `R`, and window mass — relative to the trace or an image-noise estimate)
and, below a threshold, keep the init normal and flag it. Degenerate (≤ 2 valid,
single-sided) points skip the search outright.

**Not idempotent — by design.** `refine_patch_normal` is *not* a fixed-point
operation: feeding a refined normal back in can improve it further, and that is
desirable, not a bug. Each pass re-seeds (including the mean-viewing seed), reopens
the cone, and re-freezes the support around the new normal, so a second pass can
reach a sub-grid-better point or a better basin the first grid missed (the
observed drift is small on converged points, larger where a new basin is found).
Each pass still honors never-worse-than-its-own-init, so repeated refinement
drives `Φ` toward the continuous optimum — running to convergence is the
*thorough* setting. Forcing idempotence (e.g. an acceptance threshold or no cone
reopening) would only cap the achievable accuracy; the Gauss-Newton polish above
is the right way to converge in one pass instead.

## Proposed core API

```rust
/// A fully-calibrated source camera: its intrinsics, its world-to-camera pose,
/// and a prebuilt source-image pyramid — projects world points to pixels and
/// samples colour there, everything a patch needs to be rendered from one view.
/// The pyramid is built once and borrowed for every candidate render (the `'_`
/// lifetime), so a refinement allocates no per-candidate image data.
pub struct ProjectedImage<'a> {
    pub camera: &'a CameraIntrinsics,
    pub cam_from_world: &'a RigidTransform,
    pub pyramid: &'a ImageU8Pyramid,
}

/// Photoconsistency `Φ`: the consensus all-pairs mean ZNCC (see "Objective" for
/// the form and why there is no reference-view variant).
pub enum Objective {
    /// Unweighted consensus `ρ̄ = (V‖x̄‖² − 1)/(V − 1)`.
    MeanPairwise,
    /// IRLS-weighted consensus that down-weights outlier (occluded / wrong-
    /// surface) views by a Tukey weight on each view's residual `‖xᵢ − x̄‖`,
    /// re-weighting `iters` times. Recommended default.
    RobustWeighted { iters: u32 },
}

pub struct NormalRefineParams {
    pub angular_range_deg: f64,   // half-extent of the search cone
    pub init_steps: u32,          // coarse grid resolution per axis
    pub refine_levels: u32,       // coarse-to-fine passes (each shrinks the cone)
    pub objective: Objective,     // MeanPairwise | RobustWeighted
    pub window: PatchWindow,      // per-pixel scoring weight / support (below)
    pub min_valid_fraction: f64,  // per-view valid-pixel floor
    pub min_views: u32,
    pub sampler: Sampler,         // how to sample the source pyramids
    pub render_bitmap: bool,      // also render the `representative` RGBA texture
                                  // at the found normal (off by default; one extra
                                  // full-grid source render per kept view per patch)
}

/// How to sample a `ProjectedImage`'s pyramid when rendering a patch.
pub enum Sampler {
    /// Plain bilinear from the full-resolution level. The default — the perf
    /// analysis found the found normal barely differs from anisotropic (≤ ~1° on
    /// pinhole cameras) at a fraction of the cost.
    Bilinear,
    /// Anisotropic over the pyramid (the warp's Jacobian SVD picks the level),
    /// de-aliasing oblique / grazing views. Costs ~1.6–3× more; keeps the reported
    /// `Φ`/confidence unbiased and helps slightly on distorted/fisheye rigs.
    Anisotropic,
}

/// Per-pixel weight applied to the `R×R` patch when scoring (the NCC window).
/// Also sets whether in-plane rotation is *exactly* free of the score: a radial
/// weight (`GaussianDisk`) is rotation-invariant; a square-grid weight only up to corner
/// effects.
pub enum PatchWindow {
    /// Uniform weight over the whole square grid (rotation-leaky; mainly a
    /// baseline).
    Uniform,
    /// Gaussian center weight over the square grid (the prototype default).
    Gaussian { sigma: f64 },
    /// Gaussian weight confined to the inscribed disk — radial, so in-plane
    /// rotation is exactly free and grazing corners don't leak in. Recommended
    /// default.
    GaussianDisk { sigma: f64 },
    // Future: `Alpha` — an explicit per-(s, t) mask carried by the patch (e.g. to
    // exclude occluders or non-planar pixels). See "Improvements".
}

pub struct NormalRefineResult {
    /// The input patch with its normal replaced by the optimum; `center`,
    /// `half_extent`, and the in-plane convention are preserved (`u_axis`
    /// reprojected onto the new plane). The refined normal is `patch.normal()`.
    pub patch: OrientedPatch,
    pub photoconsistency: f64,
    pub init_photoconsistency: f64,
    pub valid_view_count: u32,
    pub confidence: f64,          // peakedness of Φ at the optimum (see below)
    /// The canonical appearance in the patch `(s, t)` frame at the found normal:
    /// a fused `R×R` RGBA texture, flat row-major `(row, col, channel)`. RGB is
    /// the cross-view fused colour (the robust IRLS view weights under
    /// `RobustWeighted`, an unweighted mean under `MeanPairwise`); `A` is a
    /// per-pixel cross-view *agreement* confidence (0 where no kept view covers
    /// the pixel). Populated when `NormalRefineParams::render_bitmap` is set;
    /// `None` otherwise. This is the simple fused-render form — the per-pixel
    /// robust *template* `m` of item 7 (a free latent, super-resolvable) is still
    /// beyond v1.
    pub representative: Option<Vec<u8>>,
}

/// Refine one patch's normal. Takes the patch and returns an updated copy.
///
/// In-plane rotation can't affect photoconsistency, so the routine searches only
/// the 2-DOF normal; it reprojects the input `u_axis` onto each plane (`v = n × u`)
/// and keeps the input's `center`/`half_extent`, so the frame moves as little as
/// the new plane forces and no `up` hint is needed.
pub fn refine_patch_normal(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    resolution: u32,
    params: &NormalRefineParams,
) -> NormalRefineResult;

/// Batch over a PatchCloud (parallel across patches). Replaces each patch with
/// the refined one (same `center`/`half_extent`/in-plane convention, new normal).
pub fn refine_patch_cloud(cloud: &mut PatchCloud, views: ..., params: ...) -> Vec<NormalRefineResult>;
```

`refine_patch_normal` composes `WarpMap::from_patch` + `remap` over the
`ProjectedImage` pyramids.

## Improvements to discuss

The objective and the two-stage search above already absorb what were the
highest-value items — coarse-to-fine grid, multiple inits, the robust weighted
consensus, the Gauss-Newton polish, and the Hessian confidence — and make the
reference-view objective unnecessary. What remains open:

1. **Anti-aliased sampling (`remap_aniso`).** Oblique views foreshorten the patch;
   bilinear sampling then aliases and *biases `Φ` downward*, pulling the optimum
   off the true normal. `remap_aniso` (the patch warp's Jacobian SVD picks the
   pyramid level) de-aliases grazing views — this is `Sampler::Anisotropic`, and
   the same Jacobian feeds the GN step. Cost: one pyramid per source image
   (already in `ProjectedImage`).

2. **Back-face / grazing culling + good-view iteration.** Cull **back-facing**
   views (`is_front_facing`), past-grazing views, and views where the patch
   *center* projects out of frame — all before building any warp map (a dot
   product / one projection, not a render), once per grid level rather than per
   candidate. Beyond the soft IRLS weights, an explicit **good-view set** —
   refine, drop views whose residual to `x̄` stays high (occluded / wrong
   surface), re-refine — is the discrete complement to `RobustWeighted` and the
   most impactful add for scenes with real occlusion. The per-patch view set is
   already a first-class input: `PatchCloud.refine_normals(view_indices=…)`
   overrides the track-based lists with an explicit per-patch view set, so a
   good-view (or MVS-expanded) set can be vetted in the caller and fed straight
   back in without touching the core search.

3. **Stochastic view subsets (for large `V`).** The per-step cost is rendering the
   `V` patches, so scoring a candidate on a random `S < V` subset cuts it to `S/V`
   and buys more grid candidates / GN steps for the same budget. The consensus is a
   mean over view pairs, so the within-subset pairwise mean is an *unbiased*
   estimate of `ρ̄` (variance ~`1/S`, and `C(S, 2)` pairs from `S` renders). Make
   it pay off, not mislead:
   - **Common random numbers per level** — score all candidates of a grid level on
     the *same* subset so the noise cancels in their ranking.
   - **Grow `S` over the schedule** — small in coarse levels (just locate the
     basin), toward full `V` for the fine levels / GN polish.
   - **Exact final pass** — evaluate the chosen optimum *and* the init on all `V`
     for the reported `Φ`, the keep-vs-init decision, and the confidence Hessian.
   - Keep `S ≥ min_views`; small subsets fight the robust weighting (they can miss
     or be dominated by an occluded view). A win for orbits / dense rigs, neutral
     for small `V` — an optional lever, not a v1 default.

4. **Render-path constant factors (large clouds).** The hot loop is `V` renders
   per candidate, so at millions of points the per-render constants dominate:
   - **Fused f32 sampling.** Compute source coords and sample in one pass into a
     per-thread scratch buffer — no per-candidate `WarpMap` or `ImageU8`
     allocation — and keep the patch in f32: the `remap_*` u8 output otherwise
     quantizes before z-normalization and injects noise into `Φ` and the GN
     gradients. (So `from_patch + remap` as composed is the prototype path, not
     the fast one.)
   - **Fidelity schedule.** Mirror the view-count schedule on resolution: coarse
     levels only need to rank basins, so run them at reduced `R`, luminance only,
     and a coarser pyramid level (also better anti-aliased); reserve full `R`, all
     channels, and `remap_aniso` for the last level and the exact final pass. Keep
     the top-k coarse candidates, not top-1, against low-fidelity mis-ranking.
   - **Locality.** Order patches by primary observing image so the V pyramids stay
     hot in cache across neighbouring points; pyramids are read-only, so per-point
     parallelism shares them freely.
   - A GPU stage 1 (this is textured-quad sampling; pyramids ≈ mipmaps,
     `remap_aniso` ≈ anisotropic filtering) is the order-of-magnitude follow-up —
     see Batch placement. Keep the API batch-shaped so it can slot under the CPU
     GN polish.

5. **Cloud-level smoothness (later).** Refining points independently can give noisy
   normals on weak points. A light prior (blend toward the mean normal of k-NN
   points) or a post-pass smoothing trades a little photoconsistency for spatial
   coherence. Out of scope for v1.

6. **Patch-carried alpha mask (`PatchWindow::Alpha`).** Generalize the window from
   an analytic shape to an explicit per-`(s, t)` weight attached to the patch — a
   non-rectangular footprint or a downweight of off-surface pixels (occluders,
   depth discontinuities, a foreground matte) so the score sees only the planar
   region. Subsumes `GaussianDisk`/`Gaussian` (those are fixed masks) and pairs with the
   good-view iteration in (2); per-view alpha could even carry per-view occlusion.
   Needs the mask to ride on `OrientedPatch` (or a thin `MaskedPatch` wrapper);
   deferred until a producer exists. Out of scope for v1.

7. **Joint normal + robust representative patch (beyond v1).** The consensus
   already carries an *implicit* representative — the mean `x̄` — and minimizing
   across-view variance is identical to fitting a free template `m`:
   `minₘ Σwᵢ‖xᵢ − m‖²` gives `m = x̄`. So making the representative an explicit
   free variable buys *nothing* under plain L2; it pays off only when the metric
   changes so the best `m` is no longer the mean:
   - **Per-pixel robust loss.** Use `Σᵢ wᵢ Σ_p ρ(xᵢ[p] − m[p])`. Then `m` is a
     per-pixel robust average and the implied weights `wᵢ[p]` are a *learned*
     occlusion mask — rejecting occluded **pixels**, not whole views (which
     `RobustWeighted` and the supplied alpha mask cannot: half-occlusions,
     part-of-patch specularities, thin foreground edges). Solve by alternation:
     fix `n`, update `m` and the weights; fix `m`, GN-step `n` against it. This
     synthesizes a robust reference — unifying the reference-view and consensus
     framings — and subsumes items (2) and (6) (learned vs supplied per-pixel
     weights).
   - **A carryable template.** `m` is a latent patch tied to no single projection
     — super-resolvable, regularizable across neighbouring patches (ties to
     cloud smoothness), or kept as the surfel's canonical appearance. It is then a
     first-class **output** (`NormalRefineResult::representative`): an RGBA
     `PatchTexture` whose `A` is the learned per-pixel coverage — so this and the
     supplied alpha of item 6 are one channel (alpha in, alpha out). Keep it *off*
     the geometric `OrientedPatch` (which `WarpMap::from_patch` consumes and the
     `PatchCloud` stores struct-of-arrays — an inline `R×R` bitmap per point is
     heavy and usually unused); carry it via a thin `TexturedPatch { patch,
     texture }` wrapper or an optional parallel array on `PatchCloud`, with the
     texture holding its own resolution. At cloud scale the textures want a
     **tile atlas** (cf. `tile-batched-consensus-atlas.md`) so the cloud renders
     as instanced textured surfels on the GPU — a separate textured-patch-cloud
     spec, not this one.
   Caveats: gauge-fix `m` (zero-mean, unit-norm — the same z-normalization) or it
   trades scale with the per-view terms; and keep the warp the dominant explainer
   so `m` can't absorb genuine geometric disagreement and mask a wrong `n`. Note
   ZNCC already absorbs a *per-view* affine, so `m`'s added value is the per-pixel
   robustness, not the gain/offset.

## Recommended v1

Exp-map coarse-to-fine grid (3 levels) seeded from mean-viewing + geometric inits,
**bilinear sampling** (the perf analysis found anisotropic barely changes the
found normal at 1.6–3× the cost — keep it as an opt-in for unbiased `Φ`), a
`GaussianDisk` window, back-face/grazing culling, the `RobustWeighted` consensus
objective, and a Hessian-based confidence. Gauss-Newton polish and the iterative
good-view set are fast-follows.

## Open questions

- **Cone schedule** for the coarse-to-fine grid: `angular_range`, `init_steps`,
  `refine_levels`, and the per-level shrink factor (defaults TBD).
- **Confidence threshold** below which we keep the init normal — and how to
  normalize the smaller eigenvalue of the centered Hessian `H̃` for scale (it
  grows with texture contrast, `R`, and window mass).
- **Batch placement:** Rust `refine_patch_cloud` (parallel, pyramids per view) vs
  leaving orchestration to callers.

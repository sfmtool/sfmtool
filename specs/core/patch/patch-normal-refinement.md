# Photometric Patch-Normal Refinement

## Problem

A reconstructed 3D point `X` is seen by cameras `{(Kᵢ, Tᵢ)}`. Its surface around
`X` is locally a small plane — the surfel. `PatchCloud::from_reconstruction`
gives that surfel an *initial* normal (e.g. the mean viewing direction), which is
typically the camera-facing plane, not the true surface plane. We want the normal
`n` that maximizes **photometric consistency** across the views: the plane through
`X` whose rendered patches agree the most.

This is the planar-patch case of multi-view stereo: for a pinhole camera the
patch→image map is a homography. The **source-rendering** path uses the general
per-pixel projection (`WarpMap::from_patch`), so distortion / fisheye work
unchanged there — every grid pixel goes through the model's own `ray_to_pixel`.

Two things around it are model-*conditional*, and this claim does not extend to
them:

- The **fronto-parallel cache** (`cache=fronto`, the `quality=coarse` preset)
  replaces the per-pixel projection with a composition of two three-corner
  affine fits, which is exact only for an affine image map and leaves the
  projection's curvature over the patch as fit error — growing with image radius
  under a wide-angle or equidistant map. See
  [fronto-parallel-patch-cache.md](fronto-parallel-patch-cache.md)
  ("Limitations / when it does not apply").
- The **patch sizing** the refinement inherits from `PatchCloud` reads the
  camera model: a finite point's distance is `|z|` for the perspective family
  and the ray range for a `needs_ray_path` camera, because `|z|` collapses to
  zero at 90° off axis. See [patch-cloud.md](patch-cloud.md) (`PatchExtent`).

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

**View obliquity priors (opt-in).** Two independent uses of the per-view cosine
`cos θᵢ = v̂ᵢ·n` between a view's surface→camera direction `v̂ᵢ` and the candidate
normal `n`, both off by default (`obliquity_weight_power = 0`,
`fronto_prior_weight = 0`) so the objective above is unchanged:

- **(A) obliquity view-weight** `|cos θ|^p` (`obliquity_weight_power = p`): a
  multiplicative prior folded into the IRLS weights (`wᵢ ← wᵢ·|cos θᵢ|^p`, seeded
  and re-multiplied every reweight), so an oblique view contributes less to the
  consensus template and score — a soft, continuous version of a hard grazing-view
  cut. `p = 2` is the `cos²θ` foreshortening weight. On a point whose views span a
  range of obliquities it down-weights the grazing ones; on a low-parallax point
  (all views near-collinear, hence near-equal `cos θ`) the weights renormalize
  away, so it does nothing there — that case is (B). Robust objective only
  (`MeanPairwise` is unweighted by definition).
- **(B) fronto-parallel prior** `λ·mean_v cos²θ` (`fronto_prior_weight = λ`): an
  additive reward on the candidate normal, `Φ' = Φ + λ·mean_v (v̂ᵢ·n)²`, added
  wherever candidates are ranked (the coarse-to-fine search and the final winner
  pass, **not** the confidence stencil — confidence must report the data curvature
  alone). Its maximizer is the normal facing the observing cameras, so it supplies
  the constraint the data can't in the **narrow-baseline degeneracy**: when `Φ` is
  flat (tilting the plane shifts every view's patch almost identically) the tilt is
  unconstrained and a low-parallax surfel drifts to a photometrically-equivalent
  edge-on orientation that renders distorted (a stop sign's octagon shears into a
  cross-view-consistent smear). The prior lands it fronto-parallel instead. It only
  tips near-ties — wherever real parallax curves `Φ` the small prior is overruled,
  so well-constrained normals are unaffected — and self-scales with `mean cos²θ`.
  With the prior active the ranking is `Φ + λ·mean cos²θ`, so the reported (pure)
  `Φ` can dip below `init_photoconsistency` by up to the prior gap (a more-frontal
  normal winning a near-tie). Measured on south-building: with `λ = 0.05`, `p = 2`,
  the surfel of a near-collinear (2° triangulation angle) stop-sign point went from
  ~69° off every view (a smear) to ~2° (a regular octagon) at no NCC cost.

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

## The search

The search is a coarse-to-fine grid on the sphere around the initial normal;
the objective it ranks by is the consensus `Φ` above.

**Parameterization (`δ ∈ ℝ²`, exp-map).** A candidate perturbs the level's
center normal by a tangent vector `δ` via the exponential map
`n(δ) = cos‖δ‖·n₀ + sin‖δ‖·δ̂` (`exp_map_normal` / `exp_map_in_basis`, with `δ`
expressed in the deterministic tangent basis) — i.e. tilt `n₀` by angle `‖δ‖`
toward `δ`, equivalently a rotation about axis `n₀ × δ̂`. This is angle-uniform:
equal steps are equal angles, unlike the flat (gnomonic) `normalize(n₀ + a·u +
b·v)`, whose square `[-tan r, tan r]²` stretches to `atan(√2·tan r) ≈ √2·r` into
the corners. The search domain is the **disk** `‖δ‖ ≤ range`: the square lattice
is generated but candidates outside the disk are skipped, which keeps every
level's cone circular and matches the radial `GaussianDisk` support.

`tangent_basis(n)` is only the **search** basis — the two directions the grid
tilts `n` — not the patch's in-plane orientation. It is a pure function of `n`
(least-aligned world axis + Gram-Schmidt), so a refinement is reproducible; the
continuous optimum is basis-independent anyway, and on a finite grid the basis
only rotates the sampling lattice. In-plane rotation cannot affect `Φ`, so
nothing in the search needs an `up` hint — `repose_patch` reprojects the input
frame onto each candidate plane.

**Seeds.** Two, at most: the patch's current normal, plus the mean-viewing
normal of the supplied views when it differs from it by more than 0.5°. Each
seed runs its own coarse-to-fine walk and contributes its winner; a geometric /
PCA seed is not among them (`PatchCloud::from_reconstruction` can *seed the
cloud* that way, but the refinement does not add one).

**Coarse-to-fine grid (`coarse_to_fine`).** Per seed and per level: freeze the
common-valid support at the level's center normal, evaluate the center and every
in-disk candidate of an `init_steps × init_steps` lattice (at least 3 per axis —
with 2 the only non-center samples are disk corners that clamp away, leaving a
no-op search), recenter on the best, and shrink the cone to **one previous grid
spacing**, `2·range/(steps − 1)`. Repeat `refine_levels` times. Renders per point
are ≈ `seeds · refine_levels · init_steps² · V`, far below one dense grid of the
same precision — and the fronto-parallel cache
([fronto-parallel-patch-cache.md](fronto-parallel-patch-cache.md)) removes most
of that render cost by scoring candidates against one base render per view. The
search may rank with a cheaper objective than the reported one
(`search_robust_iters`); the final pass never does.

**Capping the refinement basis.** `max_refine_views` (`0` = uncapped, the
default) restricts a patch with more views than the cap to the `K` most
normal-informative ones, chosen by a D-optimal geometric selection — a
least-oblique anchor plus a greedy `det(M + wᵢwᵢᵀ)` fill over each view's
tangent-plane information vector. The normal is 2-DOF, so a few azimuthally
spread oblique views already determine it, and the cap cuts the per-candidate
render cost roughly linearly in the views dropped. It is floored at `min_views`
so it cannot strand a patch, and it shrinks only the *refinement basis* — the
returned patch is a repose of the full input patch, and no observation leaves the
reconstruction. See
[patch-normal-refine-view-subset.md](patch-normal-refine-view-subset.md) for the
design, algorithm and validation harness.

**Final pass.** `build_final_context` freezes one support at the *init* normal,
intersects it with each seed winner's validity, and drops a winner that is
back-facing in a kept view or that would leave fewer than 8 commonly-valid
pixels. The init and every survivor are then scored over that single frozen
support, so with the fronto-parallel prior off the returned `Φ` is never below
`init_photoconsistency`. Points failing the validity gates outright — fewer than
`max(min_views, 2)` views, or no scoreable support — are returned unrefined with
NaN scores, and a `w = 0` patch (a point at infinity, whose outward normal is
fixed by its direction) is returned untouched without searching.

With `render_bitmap` set (Python `refine_normals(render_bitmaps=True)`) this pass
scores through a `PatchViewStack` — a retained per-view render — and keeps the
*winner's* stack together with the consensus view-weights that scored it, so the
`representative` texture is fused from that one render with no extra render and
no second IRLS pass. Without it the pass stays on the lean masked-only scorer and
pays nothing for a feature it does not use.

**Local polish (Gauss-Newton / LK) — not implemented.** `Φ` is smooth, and
minimizing the photometric variance `Σ wᵢ‖xᵢ(δ) − x̄_w‖²` (weights frozen within
the step) is a nonlinear least-squares problem, so a Gauss-Newton step in `δ`
would converge in 1–3 iterations from the grid's basin. Because
`Σ wᵢ(xᵢ − x̄_w) = 0`, treating `x̄_w` as fixed gives the *exact* gradient
`∇E = 2 Σ wᵢ Jᵢᵀ(xᵢ − x̄_w)`. The `P×2` per-view Jacobian chains

```
Jᵢ = ∂xᵢ/∂δ  =  (z-normalize)′ · ∂image/∂pixel · ∂pixel/∂world · ∂world/∂δ
```

— image gradient, the **full `2×3`** projection Jacobian `∂pixel/∂world`, and
`∂world/∂δ = ∂(patch point)/∂n · ∂n/∂δ`. Note `∂pixel/∂world` here is *not* the
`remap_aniso` SVD (that is the in-plane `2×2` map); under a tilt a patch point
moves *out of plane*, so the dominant column is the one the in-plane map omits —
it would need the camera model's analytic projection Jacobian, or a
finite-difference in `δ` (2 extra renders/view per step). The z-normalization
derivative projects out the mean/scale directions. (Inverse-compositional LK
could precompute steepest-descent images, but the template `x̄_w` and the weights
change each iteration, so the symmetric multi-view form is a research note.) The
grid alone is what runs, and the sub-grid accuracy the polish would buy in one
pass is instead reachable by refining again — see "Not idempotent" below.

**Confidence.** Computed only when `compute_confidence` is set (it is off by
default, an extra un-cached source-render pass per patch for a purely
informational number; otherwise `confidence` is NaN). `grid_confidence` samples
`Φ` on a 3×3 stencil around the optimum — spacing the schedule's final grid
spacing, clamped to `[0.2°, 5°]` — forms the negated 2×2 finite-difference
Hessian, and reports `λ_min / (λ_max + 1)`: `≈ 0` on a flat `Φ` and `≈ 1` on an
isotropically peaked one, dimensionless in texture contrast because `Φ` is
already a correlation, with the `+1` floor (in `Φ` per radian²) keeping a weakly
curved optimum off full confidence. The stencil excludes the fronto-parallel
prior, so confidence reports the data's curvature alone.

The grid stencil stands in for the analytic **centered** Gauss-Newton Hessian
`H̃ = Σ wᵢ JᵢᵀJᵢ − J̄ᵀJ̄` (with `J̄ = Σ wᵢ Jᵢ`), which is the shape the analytic
form must take if it is ever built: the *between-view* curvature, not
`Σ wᵢ JᵢᵀJᵢ`. In the narrow-baseline degeneracy this exists to flag, all views
nearly coincide, so every `Jᵢ ≈ J̄`, tilting the plane shifts all patches
identically, and `Φ` is genuinely flat — `H̃ ≈ 0`, while `Σ wᵢ JᵢᵀJᵢ` stays
*large* on any textured patch and would falsely report high confidence. The grid
estimate needs no Jacobians and captures the same degeneracy for exactly that
reason: `Φ` itself is flat there. Confidence is **report-only** — nothing gates
on it, and a low-confidence patch keeps the normal the search found rather than
falling back to its init.

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
is the way to converge in one pass instead.

## Rust API

Everything below lives in
[normal_refine/](../../../crates/sfmtool-core/src/patch/normal_refine/), split across
`params` (the config and result types), `parameterization` (the sphere exp-map),
`support` / `level` (window and per-level frozen support), `znorm` (render +
z-normalize), `consensus` (`Φ`), `search` (the coarse-to-fine walk),
`view_stack` (the multi-view render substrate the representative fuses),
`view_subset` (the D-optimal basis cap), `obliquity` (the two priors) and
`fronto_cache` (the candidate cache). The PyO3 binding is
`PatchCloud.refine_normals`. The patch primitives it renders through —
`OrientedPatch`, `WarpMap::from_patch` and `remap_*` — are specified in
[patch-cloud.md](patch-cloud.md).

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

/// How candidate normals are scored: re-rendered from the source images, or
/// resampled from a cached fronto-parallel base patch per view. See
/// `fronto-parallel-patch-cache.md`.
pub enum CacheMode { Off, FrontoParallel }

pub struct NormalRefineParams {
    pub angular_range_deg: f64,   // half-extent of the level-0 search cone
    pub init_steps: u32,          // grid resolution per tangent axis, each level
    pub refine_levels: u32,       // coarse-to-fine passes (each shrinks the cone)
    pub objective: Objective,     // MeanPairwise | RobustWeighted
    pub window: PatchWindow,      // per-pixel scoring weight / support (below)
    pub min_valid_fraction: f64,  // per-view valid-pixel floor
    pub min_views: u32,
    pub sampler: Sampler,         // how to sample the source pyramids
    pub cache: CacheMode,         // source re-render vs. fronto-parallel cache
    pub cache_supersample: f64,   // base density for the cache (ignored when Off)
    pub compute_confidence: bool, // else `confidence` is NaN (an extra pass)
    pub search_robust_iters: Option<u32>,
                                  // cheaper objective for the *search* ranking
                                  // only; the final pass always uses `objective`
    pub obliquity_weight_power: f64,  // `p` of the |cos θ|^p view-weight (A)
    pub fronto_prior_weight: f64,     // `λ` of the fronto-parallel prior (B)
    pub render_bitmap: bool,      // also render the `representative` RGBA texture
                                  // at the found normal (off by default; one extra
                                  // full-grid source render per kept view per patch)
    pub max_refine_views: u32,    // cap the refinement basis at the K most
                                  // normal-informative views (`0` = uncapped)
}

/// How to sample a `ProjectedImage`'s pyramid when rendering a patch.
pub enum Sampler {
    /// Plain bilinear from the full-resolution level — the cheapest tap, within
    /// ~1° of anisotropic on fronto-parallel pinhole views, but it aliases
    /// wherever the patch grid minifies the source, which corrupts the score
    /// surface the search descends.
    Bilinear,
    /// Single bilinear sample from the pyramid level nearest the warp's local
    /// compression (`round(log2(sigma_major))` per pixel, from the Jacobian SVD).
    /// The default, and the middle point: bounds the aliasing `Bilinear` suffers
    /// on compressive warps (e.g. cross-scale views with one camera much closer)
    /// at ≈ bilinear cost, but blurs oblique views whose anisotropic footprint
    /// only `Anisotropic` resolves, and locates a sub-pixel optimum on the
    /// selected level's coarser sample grid.
    BilinearMip,
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
    /// Gaussian center weight over the square grid.
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
    /// `half_extent`, and the in-plane convention are preserved (`v_axis`
    /// reprojected onto the new plane). The refined normal is `patch.normal()`.
    pub patch: OrientedPatch,
    pub photoconsistency: f64,
    pub init_photoconsistency: f64,
    pub valid_view_count: u32,
    pub confidence: f64,          // peakedness of Φ at the optimum, NaN when
                                  // `compute_confidence` is false (see above)
    /// The canonical appearance in the patch `(s, t)` frame at the found normal:
    /// a fused `R×R` RGBA texture, flat row-major `(row, col, channel)`. RGB is
    /// the cross-view fused colour (the robust IRLS view weights under
    /// `RobustWeighted`, an unweighted mean under `MeanPairwise`); `A` is a
    /// per-pixel cross-view *agreement* confidence (0 where no kept view covers
    /// the pixel). Populated when `NormalRefineParams::render_bitmap` is set;
    /// `None` otherwise, or when the patch was not refined. This is the simple
    /// fused-render form — the per-pixel robust *template* `m` of item 7 (a free
    /// latent, super-resolvable) is not built.
    pub representative: Option<Vec<u8>>,
}

/// Refine one patch's normal. Takes the patch and returns an updated copy.
///
/// In-plane rotation can't affect photoconsistency, so the routine searches only
/// the 2-DOF normal; it reprojects the input `v_axis` onto each plane (`u = v × n`)
/// and keeps the input's `center`/`half_extent`, so the frame moves as little as
/// the new plane forces and no `up` hint is needed.
///
/// `view_keypoints`, when given, is parallel to `views`: `Some([x, y])` anchors
/// that view's patch at the given source-image keypoint instead of the
/// reprojected point center, `None` leaves it centered. An all-`None` slice (or
/// `None` outright) is byte-for-byte the un-anchored behavior.
pub fn refine_patch_normal(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    resolution: u32,
    params: &NormalRefineParams,
    view_keypoints: Option<&[Option<[f64; 2]>]>,
) -> NormalRefineResult;

/// Batch over a PatchCloud (parallel across patches, rayon). Replaces each patch
/// with the refined one (same `center`/`half_extent`/in-plane convention, new
/// normal) and returns the per-patch results in order. `patch_views[i]` indexes
/// `views` for patch `i` (see `view_indices_from_reconstruction`), and
/// `progress`, when given, is bumped as each patch finishes so a Python poller
/// can report intra-pass progress with the GIL released.
pub fn refine_patch_cloud_normals(
    cloud: &mut PatchCloud,
    views: &[ProjectedImage<'_>],
    patch_views: &[Vec<u32>],
    resolution: u32,
    params: &NormalRefineParams,
    patch_view_keypoints: Option<&[Vec<Option<[f64; 2]>>]>,
    progress: Option<&std::sync::atomic::AtomicUsize>,
) -> Vec<NormalRefineResult>;

/// For each patch of `cloud` (linked to `recon` by `point_indexes`), the image
/// indices observing its source 3D point — the default `patch_views` above.
pub fn view_indices_from_reconstruction(
    recon: &SfmrReconstruction,
    cloud: &PatchCloud,
) -> Vec<Vec<u32>>;
```

`refine_patch_normal` composes `WarpMap::from_patch` + `remap` over the
`ProjectedImage` pyramids.

**Who consumes the representative.** `sfm xform --refine-normals bitmaps=true`
scatters the per-patch textures to per-3D-point rows and persists them as the
`.sfmr` `patch_bitmaps_y_x_rgba` array, and `sfm inspect --strips` renders
through the same flag. The `sfm embed-patches` pipeline does **not**: it takes
its stored bitmaps from the sub-pixel keypoint refiner instead
(`refine_keypoints(render_bitmaps=True)`, which reuses this module's
`PatchViewStack::render` / `fuse`), because that fuses each point's
representative at the *final* per-view keypoints rather than one round stale,
and covers the points at infinity this refinement skips.

## Improvements to discuss

The objective and the coarse-to-fine search above already absorb what were the
highest-value items — the exp-map grid, multiple seeds, the robust weighted
consensus, and a curvature confidence — and make the reference-view objective
unnecessary. What remains open:

1. **Anti-aliased sampling (`remap_aniso`).** Oblique views foreshorten the patch;
   bilinear sampling then aliases and *biases `Φ` downward*, pulling the optimum
   off the true normal. `remap_aniso` (the patch warp's Jacobian SVD picks the
   pyramid level) de-aliases grazing views — this is `Sampler::Anisotropic`, at
   no extra storage (the pyramid per source image is already in
   `ProjectedImage`). It is not the default because it costs 1.6–3× for a normal
   that differs by ≲ 1° on pinhole views; what stays open is whether the
   unbiased `Φ` it reports is worth that on distorted / fisheye rigs, where the
   measured benefit is small but real.

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
   and buys more grid candidates for the same budget. The consensus is a
   mean over view pairs, so the within-subset pairwise mean is an *unbiased*
   estimate of `ρ̄` (variance ~`1/S`, and `C(S, 2)` pairs from `S` renders). The
   *deterministic* form of this trade is what `max_refine_views` already does;
   the stochastic one is not implemented, and would need to:
   - **Common random numbers per level** — score all candidates of a grid level on
     the *same* subset so the noise cancels in their ranking.
   - **Grow `S` over the schedule** — small in coarse levels (just locate the
     basin), toward full `V` for the fine levels.
   - **Exact final pass** — evaluate the chosen optimum *and* the init on all `V`
     for the reported `Φ`, the keep-vs-init decision, and the confidence.
   - Keep `S ≥ min_views`; small subsets fight the robust weighting (they can miss
     or be dominated by an occluded view). A win for orbits / dense rigs, neutral
     for small `V` — an optional lever, not a default.

4. **Render-path constant factors (large clouds).** The hot loop is `V` renders
   per candidate, so at millions of points the per-render constants dominate:
   - **Fused f32 sampling.** Compute source coords and sample in one pass into a
     per-thread scratch buffer — no per-candidate `WarpMap` or `ImageU8`
     allocation — and keep the patch in f32: the `remap_*` u8 output otherwise
     quantizes before z-normalization and injects noise into `Φ`. (So
     `from_patch + remap` as composed is the source path, not the fast one; the
     fronto cache's resample already writes f32 straight into the scorer's
     layout, and is where that shape is proven.)
   - **Fidelity schedule.** Mirror the view-count schedule on resolution: coarse
     levels only need to rank basins, so run them at reduced `R`, luminance only,
     and a coarser pyramid level (also better anti-aliased); reserve full `R`, all
     channels, and `remap_aniso` for the last level and the exact final pass. Keep
     the top-k coarse candidates, not top-1, against low-fidelity mis-ranking.
   - **Locality.** Order patches by primary observing image so the V pyramids stay
     hot in cache across neighbouring points; pyramids are read-only, so per-point
     parallelism shares them freely.
   - A GPU grid search (this is textured-quad sampling; pyramids ≈ mipmaps,
     `remap_aniso` ≈ anisotropic filtering) is the order-of-magnitude follow-up.
     Keep the API batch-shaped so it can slot under a CPU polish.

5. **Cloud-level smoothness (later).** Refining points independently can give noisy
   normals on weak points. A light prior (blend toward the mean normal of k-NN
   points) or a post-pass smoothing trades a little photoconsistency for spatial
   coherence. Not implemented.

6. **Patch-carried alpha mask (`PatchWindow::Alpha`).** Generalize the window from
   an analytic shape to an explicit per-`(s, t)` weight attached to the patch — a
   non-rectangular footprint or a downweight of off-surface pixels (occluders,
   depth discontinuities, a foreground matte) so the score sees only the planar
   region. Subsumes `GaussianDisk`/`Gaussian` (those are fixed masks) and pairs with the
   good-view iteration in (2); per-view alpha could even carry per-view occlusion.
   Needs the mask to ride on `OrientedPatch` (or a thin `MaskedPatch` wrapper);
   deferred until a producer exists. Not implemented.

7. **Joint normal + robust representative patch.** The consensus
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
     fix `n`, update `m` and the weights; fix `m`, step `n` against it. This
     synthesizes a robust reference — unifying the reference-view and consensus
     framings — and subsumes items (2) and (6) (learned vs supplied per-pixel
     weights).
   - **A carryable template.** `m` is a latent patch tied to no single projection
     — super-resolvable, regularizable across neighbouring patches (ties to
     cloud smoothness), or kept as the surfel's canonical appearance. The
     **output** it would fill is already there: `NormalRefineResult::representative`
     carries the *fused-render* form of exactly this — an RGBA texture whose `A`
     is a per-pixel cross-view agreement rather than a learned coverage — so this
     and the supplied alpha of item 6 are one channel (alpha in, alpha out).
     What is missing is the latent: the texture is fused from the winner's view
     stack, not solved for. It stays *off*
     the geometric `OrientedPatch` (which `WarpMap::from_patch` consumes and the
     `PatchCloud` stores struct-of-arrays — an inline `R×R` bitmap per point is
     heavy and usually unused); today the cloud-level carrier is the
     reconstruction's `patch_bitmaps_y_x_rgba` array. At cloud scale the
     textures want a
     **tile atlas** (cf. `tile-batched-consensus-atlas.md`) so the cloud renders
     as instanced textured surfels on the GPU — a separate textured-patch-cloud
     spec, not this one.
   Caveats: gauge-fix `m` (zero-mean, unit-norm — the same z-normalization) or it
   trades scale with the per-view terms; and keep the warp the dominant explainer
   so `m` can't absorb genuine geometric disagreement and mask a wrong `n`. Note
   ZNCC already absorbs a *per-view* affine, so `m`'s added value is the per-pixel
   robustness, not the gain/offset.

## Parameters

`NormalRefineParams::default()` (in `normal_refine/params.rs`), which the
`PatchCloud.refine_normals` binding mirrors keyword-for-keyword so the two layers
cannot drift:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `angular_range_deg` | `25.0` | half-extent of the level-0 search cone |
| `init_steps` | `7` | grid samples per tangent axis, per level (floored at 3) |
| `refine_levels` | `3` | coarse-to-fine passes; each shrinks to one grid spacing |
| `objective` | `RobustWeighted { iters: 3 }` | consensus objective |
| `window` | `GaussianDisk { sigma: 0.6 }` | scoring window, `sigma` in `(s, t)` units |
| `min_valid_fraction` | `0.6` | per-view floor on the window-weighted valid fraction |
| `min_views` | `3` | minimum kept views (floored at 2 for the outright skip) |
| `sampler` | `BilinearMip` | mip-nearest bilinear tap |
| `cache` | `FrontoParallel` | score candidates off one base render per view |
| `cache_supersample` | `2.0` | base density for that cache |
| `compute_confidence` | `false` | else `confidence` is NaN |
| `search_robust_iters` | `None` | search ranks with `objective` |
| `obliquity_weight_power` | `0.0` | obliquity view-weight (A) off |
| `fronto_prior_weight` | `0.0` | fronto-parallel prior (B) off |
| `render_bitmap` | `false` | no `representative` texture |
| `max_refine_views` | `0` | refinement basis uncapped |

`BilinearMip` is the sampler default rather than `Anisotropic` because the found
normal barely moves (≲ 1° on pinhole views) at 1.6–3× the cost; `Anisotropic`
stays an opt-in for an unbiased `Φ` and confidence. The
`reports/2026-06-13-perf-patch-normal-refinement.md` measurements behind that —
phase breakdown and per-knob perf-vs-benefit — are reproducible with
`scripts/bench_normal_refine.py` and the `patch_render` criterion bench;
`SFMTOOL_PROFILE=1` turns on the hot-path phase timers (`normal_refine/prof.rs`),
which `refine_patch_cloud_normals` reports per batch.

## Open questions

- **Confidence threshold** below which a caller should distrust the refined
  normal — and how to normalize the curvature for scale (it grows with texture
  contrast, `R`, and window mass). The routine itself never gates: it reports
  `confidence` and keeps the normal it found.
- **The analytic centered Hessian** `H̃`, as the replacement for the
  finite-difference stencil — cheaper only alongside a Gauss-Newton polish that
  already forms the `Jᵢ`.

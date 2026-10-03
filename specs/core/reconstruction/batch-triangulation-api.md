# Batch Triangulation API with Observability Diagnostics

Triangulating a track means finding the 3D point that best fits the rays its
observations cast, and the linear solve that does so also reveals how
well-determined that point is — a track seen from one direction only pins down
two of its three coordinates. This is one batch API over whole sets of tracks
that returns each solved point **plus the observability diagnostics the solve
already computes** (the normal matrix's spectrum, and an optional
noise-calibrated depth uncertainty), so callers deciding whether a point is at a
finite depth at all can read the answer off the conditioning instead of
re-deriving it. The alternative they used before — deciding finite-vs-infinity
from the *maximum pairwise viewing angle* — is an extreme order statistic that
keypoint noise inflates and that **grows with view count**, so genuine points at
infinity with many observations were misclassified as finite.

Related: [the decision layer over this API](triangulation-rules.md), which judges a
cluster against an angular floor, cheirality and a reprojection bar and returns
a position or a bearing;
[finding points at infinity in an existing solve](../../cli/reconstruction/xform/find-points-at-infinity.md),
[the v2 points-at-infinity format model](../../formats/sfmr-file-format.md) (§7), and
[rendering points at infinity in the GUI](../../gui/point-cloud-rendering.md).

## The Problem

A point at infinity is a track whose observation rays are parallel to within
measurement noise: its depth is unobservable. The current finite-vs-infinity
test is `parallax_px = alpha_max · f_max` against a 1 px floor, where
`alpha_max = max_viewing_angle(rays)` is the single widest angle among all
`K(K−1)/2` ray pairs (`geometry/viewing_angle.rs:28`). That statistic is wrong for this
job:

- Under pure 1 px localization noise (no real parallax) the **expected** max
  pairwise angle already exceeds the floor once a track has a handful of views,
  and it keeps rising with `K` — so more observations make a true infinity point
  *more* likely to be called finite. (Measured: a 27-view distant track scored
  2.27 px; a direction-only model reprojected it at 0.61 px RMS, and adding a
  finite depth improved RMS by only 0.025 px — the depth explains nothing.)
- The midpoint solve in `classify_track` (then inline in
  `analysis/infinity/discover.rs`) already
  built the normal matrix `A = Σ(I − dᵢdᵢᵀ)` and computed `det(A)`, but only
  used `det` against an ultra-loose gate (`det < 1e-9·‖A‖³`)
  that fires only for *exactly* singular `A`. The
  eigenvalues of `A` — a 1000× separation between genuine and degenerate tracks
  (population medians: condition number 82 vs 89,599; relative depth uncertainty
  1.6% vs 33%) — are discarded.

So the fix and the refactor are the same change: make the triangulation a
reusable batch operation that returns its conditioning, and have the
classifiers decide on that.

## Relationship to the max-track-angle statistic

The older signal for the same question is `max_viewing_angle`
([`geometry/viewing_angle.rs`](../../../crates/sfmtool-core/src/geometry/viewing_angle.rs)):
the widest pairwise angle between a track's rays, computed against the *stored*
point with no solve at all. The GUI presents it as "High = well-triangulated, low
= unreliable", and for genuinely finite points with real parallax that is
accurate — a wide max angle does mean a well-conditioned depth — so it is the
right overlay for the bulk of the cloud.

It misleads in one regime, the distant and near-infinity one, where the parallax
is dominated by localization noise and a *maximum* statistic inflates with view
count: a far point seen by many cameras reads as "well-triangulated" while its
depth is in fact unconstrained. The condition-number and inverse-depth
diagnostics agree with the angle on finite points and additionally get that
regime right, so they sit **alongside** the angle rather than replacing it.
`max_viewing_angle` also still backs `xform --remove-narrow-tracks` through
`compute_narrow_track_mask`, as a fast pre-filter; what it stopped being is the
*classification* signal.

## Rust API

The solver lives in
[triangulation.rs](../../../crates/sfmtool-core/src/reconstruction/triangulation.rs),
bound as `sfmtool._sfmtool.analysis.triangulate_batch` and, over a loaded
reconstruction, as `SfmrReconstruction.triangulation_diagnostics`. Tracks are flattened
CSR-style (the same shape as the reconstruction's `observation_offsets`): track
`t` owns `dirs[offsets[t]..offsets[t+1]]` and the matching `centers`.

```rust
// crates/sfmtool-core/src/reconstruction/triangulation.rs

/// One track's triangulation and the observability diagnostics the linear
/// solve computes alongside the point. Geometric fields are always populated.
pub struct Triangulation {
    /// Least-squares closest point to the rays (the midpoint estimate).
    pub point: Point3<f64>,
    /// Eigenvalues of A = Σ(I − dᵢdᵢᵀ), ascending. `eigenvalues[0] → 0` marks
    /// parallel rays (depth unobservable). Σ eigenvalues = 2·K.
    pub eigenvalues: [f64; 3],
    /// Condition number λ_max / λ_min of A (∞ when exactly degenerate). A
    /// cheap geometric indicator — but note it scales with track length K, so
    /// it is a proxy, not the decision variable (see Diagnostics).
    pub condition_number: f64,
    /// `point` has positive depth in every observing camera (lies in front of
    /// each, not behind). False means the least-squares point landed behind a
    /// camera, so the finite position is non-physical.
    pub in_front_of_all_cameras: bool,
}

/// Triangulate a batch of tracks. `dirs` are unit world-space rays; `centers`
/// the matching camera centers; `offsets` (len M+1) delimits the M tracks.
/// Pure and IO-free: ray construction (un-projection, distortion, pose) is the
/// caller's concern, which keeps this agnostic to where the rays came from.
pub fn triangulate_batch(
    dirs: &[Vector3<f64>],
    centers: &[Point3<f64>],
    offsets: &[usize],
) -> Vec<Triangulation>;
```

The cost is `O(K)` to assemble `A` per track plus one fixed 3×3 symmetric
eigensolve — the eigensolve is a constant per call, independent of how many
points exist. The whole-reconstruction cost is just that per-track cost summed
over tracks (≈100k finite points on the larger external KerryPark360 capture).

### Diagnostics: the decision variable vs the geometric flag

The condition number is free but **not track-length invariant** (λ_max ≈ K for
near-parallel rays), so its threshold drifts with view count. The principled,
scale-free decision variable is the **depth uncertainty**, which needs a
per-ray angular noise σ (e.g. `noise_px / fᵢ`) — a *policy* input that does not
belong inside the geometric solver. Keep it a separate, opt-in batch step:

```rust
/// Depth uncertainty along the mean viewing direction, from the inverse-
/// variance-weighted normal matrix. `sigma_rad` is per-ray angular noise.
pub struct DepthUncertainty {
    pub depth: f64,
    pub sigma: f64,
    /// inverse-depth z-score = depth / sigma. Small (≲ 3-4) ⇒ statistically
    /// indistinguishable from infinity. (kerry_park medians: genuine 62,
    /// discovered "finite" 3.) The finite-vs-∞ test, but reliable only when the
    /// solve is non-degenerate — it divides by the solved depth, which is noise
    /// when the rays are near-parallel. See "Scene-relative resolvability".
    pub inverse_depth_z: f64,
    /// Farthest depth this track's geometry can tell from infinity:
    /// `B⊥ / σ` (perpendicular camera baseline over angular noise) — equivalently
    /// the depth at which `inverse_depth_z` would fall to 1. Independent of the
    /// (possibly garbage) solved depth, so it stays meaningful when the rays are
    /// near-parallel. Gated against `finite_horizon` (see below).
    pub resolvable_distance: f64,
}

pub fn depth_uncertainty_batch(
    tris: &[Triangulation],
    dirs: &[Vector3<f64>],
    centers: &[Point3<f64>],
    offsets: &[usize],
    sigma_rad: &[f64],
) -> Vec<DepthUncertainty>;
```

The two functions layer: `triangulate_batch` computes the point and the
geometric diagnostics (eigenvalues, condition number) that come out of the solve
itself, and `depth_uncertainty_batch` computes the more detailed, noise-dependent
diagnostics (depth σ, inverse-depth z) on top of that result. The split keeps the
noise model out of the geometric solver, so the conditioning is always available
and the noise-calibrated statistics are computed only when a caller asks for them.

### Scene-relative resolvability: the `indeterminate` state

The bare `inverse_depth_z < cutoff` test is reliable only when the solve is
non-degenerate. On the KerryPark360 capture — a walk with frequent "stop and
look around" pauses — it breaks in the no-baseline regime, and it breaks
two-sided:

- A genuinely distant point seen only from one stop is called `∞` (right by
  accident).
- A *near* point seen only from one stop can be called **finite** (wrong).
  Example `pt3d_…_102031`: 23 observations, all from one stop (observing-camera
  spread 0.35 of a 165-unit capture). The near-parallel rays have no real
  intersection, so the least-squares point falls to range 0.24 — *inside* the
  camera cluster — and `inverse_depth_z` came out 4.97, just over the cutoff.
  Leave-one-out swings it between −5 and +5: it is noise that happened to clear
  the bar. Its mirror image `pt3d_…_97221` is the same situation falling the
  other way (→ `∞`).

The cause is structural: `inverse_depth_z = depth / σ_depth ≈ (B⊥/σ) / depth`,
so it divides by the *solved depth*, which is a noise-driven garbage value when
the rays are near-parallel. Both points are the same physical case —
under-observed single-stop clusters where depth is genuinely unknowable — and
the binary test just fell off opposite sides.

The fix anchors the decision to a *stable* reference instead of the solved
depth. Define the **resolvable distance** `D_max = B⊥ / σ` — the perpendicular
camera baseline over the angular noise, equivalently the depth at which
`inverse_depth_z` would fall to 1, i.e. the farthest a point can be and still be
told from infinity by this track's geometry. `depth_uncertainty_batch` returns
it as `resolvable_distance`; it does not depend on the solved depth, so it stays
meaningful exactly where `inverse_depth_z` goes unstable.

A new policy input, **`finite_horizon`**, is the farthest distance at which we
*require* the geometry to distinguish finite from infinity. The classifier then
yields three states instead of two:

- `resolvable_distance < finite_horizon` → **indeterminate**: the baseline could
  not place a point even at the required distance, so neither "finite" nor "at
  infinity" is earned, and the track is **dropped** rather than emitted. (Both
  97221 and 102031 land here.)
- otherwise, decide **finite** vs **at infinity** by `inverse_depth_z` as before.

Dropping keeps the `.sfmr` model binary (`w=1` / `w=0`) — no third state to
store. It is clean in discovery, which is additive: an indeterminate candidate
is simply never appended, so the base solve is untouched and the discovered
cloud holds only tracks whose depth the geometry could actually adjudicate.
`classify_points_at_infinity` stays relabel-only and non-destructive: it never
removes a solve point. It demotes a point to `w=0` only on a *confident*
infinity call (sufficient baseline and `inverse_depth_z` below the cutoff); an
indeterminate solve point — one we lack the baseline to adjudicate — is left as
the finite point the solve produced. So "drop" applies to discovery candidates;
reclassify simply declines to demote.

This makes "at infinity" *scene-relative* and honest: not "infinitely far"
(unprovable), but "farther than this capture's geometry can place within the
extent it explored." The same point in a wider capture would correctly become
finite.

**Finite-vs-∞ is resolvability, not distance.** Among tracks that *clear* the
gate, the split is the `inverse_depth_z` cutoff — and it is emphatically not a
distance threshold. A KerryPark360 pair makes this concrete:

| | `pt3d_…_108877` (**finite**) | `pt3d_…_96414` (**at ∞**) |
|---|---|---|
| range | **261** (beyond the 165 extent) | 122 |
| `inverse_depth_z` | **4.06** (just over cutoff) | 2.48 |
| observing-camera baseline span | **10.6** | 3.7 |
| views | 50 | 17 |

The finite point is the *farther* one. What separates them is the baseline span
of their observing cameras: 108877's span 10.6 (against the 165 camera extent),
so even at range 261 its parallax is significant (`z = 4.06`); 96414's span only
3.7, so at range 122 it is not (`z = 2.48`). They bracket the cutoff almost
exactly — `z ≈ 4` draws the line at ~25% depth uncertainty (`σ/depth ≈ 1/z`).
These near-cutoff, real-baseline points (not the degenerate near-zero-baseline
ones) are precisely what a cutoff sweep should tune against.

**`finite_horizon` defaults to the camera extents** — the spatial spread of the
camera *centers*, not the point-cloud extent. The reference must be independent
of the triangulation being judged: camera centers come straight from the solved
poses and do not move when a point is mis-triangulated, whereas the point-cloud
extent is polluted by the very near-field and spurious-`∞` artifacts we are
trying to catch. And the baseline we gate on is itself a camera-spread, so
normalizing against the camera extent compares like with like.

**Perpendicular caveat.** Parallax comes only from the camera spread
*perpendicular to a point's bearing*, so `D_max` uses `B⊥`, not the scalar
baseline. A scalar camera-extent default is therefore a coarse *upper bound* (a
long thin path has large extent along it and ~none across): failing the scalar
gate means definitely indeterminate, but passing it does not guarantee
resolvability in every direction. The precise per-track quantity is the
perpendicular spread of the observing cameras about the mean viewing direction.

**Placement.** `resolvable_distance` is geometry + noise, so it is a field on
`DepthUncertainty` and adds no input to `depth_uncertainty_batch`.
`finite_horizon` is policy, so it enters the classifier —
`analysis/infinity/convert.rs::classify_rays_at_infinity` and the public
`classify_points_at_infinity` / `find_points_at_infinity` (and the GUI
diagnostics) — defaulting to the reconstruction's camera extents.

## Point or bearing

The z test above asks whether the solved depth is far from zero relative to
its linearised uncertainty. A second family in the same module asks the
question a stored track's representation depends on more directly: does giving
the track a depth explain its rays better than a direction alone does, by more
than the noise could explain? It fits both models to the track's rays and
compares their costs, which is a likelihood-ratio test between two nested
models. A stored reconstruction supplies the noise level from its own residuals
("The measured noise level") and runs the test over its points with one method
("Over a reconstruction"), and both are bound to Python. No classifier or
report reads it yet; moving the four finite-or-bearing rules onto
it is the amendment draft
[point-or-bearing-likelihood-ratio.md](../../drafts/point-or-bearing-likelihood-ratio.md),
which also carries the measurements that motivate it.

### Interface

The functions live in
[point_or_bearing.rs](../../../crates/sfmtool-core/src/reconstruction/triangulation/point_or_bearing.rs),
the ray and weight constructions in
[point_or_bearing/ray_weight.rs](../../../crates/sfmtool-core/src/reconstruction/triangulation/point_or_bearing/ray_weight.rs),
and all are re-exported from `sfmtool_core::reconstruction::triangulation`. They
take the CSR ray layout of `triangulate_batch`, plus one 2×3 noise weight per
ray.

```rust
/// The bearing that best explains one track's rays, and how strongly the rays
/// ask for a depth.
pub struct BearingScore {
    /// Eigenvector of the smallest eigenvalue of M = Σ AᵢᵀAᵢ,
    /// Aᵢ = Wᵢ (I − dᵢdᵢᵀ), signed to point along the weighted mean ray.
    pub bearing: Vector3<f64>,
    /// The bearing's cost in noise units (that eigenvalue). Λ ≤ bearing_cost.
    pub bearing_cost: f64,
    /// The score statistic gᵀH⁻¹g of the point model at (bearing, ρ = 0);
    /// 0 when the Gauss-Newton step in ρ is not positive, or ρ has no leverage.
    pub depth_score: f64,
    /// bearing_cost less the cost at the weighted linear midpoint, when that
    /// lies in front of every camera, else 0. A lower bound on Λ.
    pub midpoint_bound: f64,
    /// dᵢ · bearing > 0 for every ray.
    pub bearing_in_front_of_all_cameras: bool,
    pub num_views: usize,
}

/// Both fits of one track and the exact statistic that compares them.
pub struct PointBearingFit {
    pub bearing: Vector3<f64>,
    pub bearing_cost: f64,
    /// The point model: anchor + direction / inverse_depth, inverse_depth ≥ 0.
    pub anchor: Point3<f64>,
    pub direction: Vector3<f64>,
    pub inverse_depth: f64,
    /// That point, or None when inverse_depth = 0 (the point is the bearing).
    pub point: Option<Point3<f64>>,
    pub point_cost: f64,
    /// bearing_cost − point_cost ≥ 0: Λ under the fit's loss, at the optimum
    /// the fit reaches (it can fall slightly short of the best in-front point).
    /// Meaningless when in_front_of_all_cameras is false.
    pub depth_likelihood_ratio: f64,
    pub in_front_of_all_cameras: bool,
    pub num_views: usize,
}

pub struct PointBearingFitOptions {
    /// None: plain least squares. Some(c): soft-L1 per residual component at
    /// scale c noise units, the loss bundle adjustment uses.
    pub soft_l1_scale: Option<f64>,
    /// Levenberg-Marquardt iterations per fit.
    pub max_iterations: usize,
}
// Default: soft_l1_scale Some(DEFAULT_SOFT_L1_SCALE),
// max_iterations DEFAULT_POINT_FIT_MAX_ITERATIONS.

pub const DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD: f64 = 25.0;
pub const DEFAULT_SOFT_L1_SCALE: f64 = 3.0;
pub const DEFAULT_POINT_FIT_MAX_ITERATIONS: usize = 20;

pub fn bearing_score(dirs: &[Vector3<f64>], centers: &[Point3<f64>],
    weights: &[Matrix2x3<f64>]) -> Option<BearingScore>;
pub fn bearing_score_batch(dirs: &[Vector3<f64>], centers: &[Point3<f64>],
    offsets: &[usize], weights: &[Matrix2x3<f64>]) -> Vec<Option<BearingScore>>;

pub fn fit_point_and_bearing(dirs: &[Vector3<f64>], centers: &[Point3<f64>],
    weights: &[Matrix2x3<f64>], start: Option<Point3<f64>>,
    anchor: Option<Point3<f64>>, options: &PointBearingFitOptions)
    -> Option<PointBearingFit>;
pub fn fit_point_and_bearing_batch(dirs: &[Vector3<f64>], centers: &[Point3<f64>],
    offsets: &[usize], weights: &[Matrix2x3<f64>], starts: Option<&[Point3<f64>]>,
    anchors: Option<&[Point3<f64>]>, options: &PointBearingFitOptions)
    -> Vec<Option<PointBearingFit>>;

/// Finite when bearing_cost ≥ threshold and depth_score or midpoint_bound
/// reaches the threshold; otherwise a bearing.
pub fn is_finite(score: &BearingScore, threshold: f64) -> bool;

/// One observation as a world-frame ray and its weight (1/σ_px)·J·R, J the
/// projection's derivative at the camera-frame ray, R the world-to-camera
/// rotation.
pub struct ObservedRay { pub dir: Vector3<f64>, pub weight: Matrix2x3<f64> }
pub fn observed_ray(camera: &CameraIntrinsics, cam_from_world: &UnitQuaternion<f64>,
    pixel: [f64; 2], sigma_px: f64) -> Option<ObservedRay>;

/// The weight of isotropic angular noise σ (radians): (1/σ)·Bᵀ, B any
/// orthonormal basis perpendicular to the ray.
pub fn isotropic_ray_weight(dir: &Vector3<f64>, sigma_rad: f64) -> Matrix2x3<f64>;
pub fn isotropic_ray_weights(dirs: &[Vector3<f64>], sigma_rad: &[f64]) -> Vec<Matrix2x3<f64>>;
```

```rust
use sfmtool_core::reconstruction::triangulation::{
    bearing_score_batch, is_finite, observed_ray, DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD,
};

// One observation per entry of the CSR layout; sigma_px measured from the
// reconstruction's finite points.
for (cam, image, pixel) in observations {
    let ray = observed_ray(cam, &image.quaternion_wxyz, pixel, sigma_px)?;
    dirs.push(ray.dir);
    weights.push(ray.weight);
    centers.push(image.camera_center());
}
let finite: Vec<bool> = bearing_score_batch(&dirs, &centers, &offsets, &weights)
    .iter()
    .map(|s| s.as_ref().is_some_and(|s| is_finite(s, DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD)))
    .collect();
```

**Why it is shaped this way.**

- **Deciding is separate from placing.** `bearing_score` is one pass over the
  rays, a 3×3 eigensolve and a 3×3 solve, with no iteration, and it is all a
  verdict needs. `fit_point_and_bearing` runs the iterative point fit, for the
  tracks that come out finite and need a position, and for reports that want
  the exact `Λ`. Its `start` takes the position a caller already holds, so
  placing is a few warm-started iterations.
- **Rays in, angles out, weighted per ray.** The residual of ray `i` is
  `Wᵢ (I − dᵢdᵢᵀ) m`: the component of the model's unit direction `m` from that
  camera perpendicular to the ray, which is the sine of the angle between them,
  through the ray's weight. That keeps the functions independent of the camera
  model, as `triangulate_batch` is, and the sine (rather than the tangent) is
  what makes the bearing fit an eigenproblem.
- **The weight is a 2×3 world-frame matrix, not a scalar.** A fisheye's pixels
  per radian differ radially and tangentially, by up to 2.1 to 1 on the Kerry
  Park ground truth. One scalar angular noise per ray moved the statistic by
  up to 28% there when it was σ_px times the geometric mean `√(s₁s₂)` of the
  two stretches, and by up to −52% when it was σ_px over the focal length
  (point 154: 0.64 against 1.33). The weight `(1/σ_px)·J·R` maps a world-frame change of
  direction straight to pixels over the noise, so the residual is, to first
  order, the pixel residual over σ_px. A weight is a 2×3 matrix in world
  coordinates, so there is no basis convention for a caller to share, and any
  2×2 rotation of it describes the same noise. `observed_ray` is the one
  construction from an observation, so every caller (the bindings, the
  reports, the classifiers, the bench) weights alike; it uses the analytic
  projection derivative where the model has one
  ([../camera/projection-jacobian.md](../camera/projection-jacobian.md)) and a
  central difference otherwise, through the camera's own
  `CameraIntrinsics::pixel_jacobian`. It returns `None` for a `sigma_px` that
  is not finite and positive, for a pixel whose un-projected ray does not
  project back to within 1e-3 px of it (a pixel past a fold of the
  distortion), and where a difference probe leaves the model's domain.
  `isotropic_ray_weight` covers a caller with only a scalar angular noise.
- **σ is an input.** The noise level is the one policy choice in the test, and
  the threshold is a separate argument of `is_finite`, for the reason
  `depth_uncertainty_batch` is kept out of `triangulate_batch`.
- **`Option` per track.** A ray is usable when its direction is finite and
  non-zero, its centre finite and its weight finite, non-zero and with a
  squared norm that does not overflow. With fewer
  than two usable rays neither model means anything, and the track is `None`.
- **`point` is an `Option`.** At `inverse_depth = 0` the best point is the
  bearing itself and has no finite position, so the fit reports the anchor,
  direction and inverse depth, and the point only when it is finite.
- **The anchor is an option.** The fitted point and `Λ` do not depend on the
  anchor, but the parametrisation does, and bundle adjustment's inverse-depth
  points keep theirs fixed across rounds.
- **Batches are parallel and exact.** The batch forms run the single-track
  function per track on rayon, so each entry is bit-identical to the
  single-track call.

### Theory

The point model is an anchor `a` (by default the centroid of the observing
camera centres), a unit direction `u` and an inverse depth `ρ ≥ 0`, with the
point at `a + u / ρ`. The direction from camera `i` to the point is
proportional to `u + ρ (a − cᵢ)`, which is smooth through `ρ = 0`, where it is
`u` from every camera: the bearing model. The bearing is the point model with
`ρ` held at its boundary, so `point_cost ≤ bearing_cost` and
`Λ = bearing_cost − point_cost ≥ 0`. For a track truly at infinity with
Gaussian noise of the stated weight, `P(Λ > t) = ½ P(χ²₁ > t)`: half the time
the unconstrained optimum is behind the bearing, and the constrained one sits
at `ρ = 0` with `Λ = 0`. `t = 25` is roughly a 1 in 3.5 million chance of
calling a true bearing finite.

- **The bearing is closed form.** The bearing's cost is `uᵀ M u` with
  `M = Σ AᵢᵀAᵢ`, `Aᵢ = Wᵢ (I − dᵢdᵢᵀ)`, so the best bearing is the eigenvector
  of `M`'s smallest eigenvalue and the cost is that eigenvalue. `bearing_cost`
  is summed from the rays' residuals `‖Aᵢ u‖²` at that eigenvector, which is
  the same number without the eigensolve's round-off relative to `λ_max`.
- **The score is one more pass.** At `(bearing, ρ = 0)` each ray contributes a
  residual and a 2×3 Jacobian with respect to two tangent directions of `u` and
  `ρ`. With `v = u + ρ (a − cᵢ)` and `m = v / ‖v‖`,
  `∂m/∂v = (I − m mᵀ) / ‖v‖`. `depth_score = gᵀH⁻¹g` is the cost reduction a
  Gauss-Newton step predicts, evaluated through the Schur complement of `ρ` so
  that a step towards `ρ < 0` (the rays converge behind the cameras) or a `ρ`
  with no leverage (all cameras at one centre) gives 0. It is the score form of
  the test, linearised at `ρ = 0`, where the inverse-depth model is close to
  linear, so it tracks `Λ` closely wherever the rays are close to parallel,
  which is where the verdict is in question. It does not depend on the anchor.
- **The midpoint bound decides wide-angle tracks.** When the rays spread over a
  wide angle (cameras on an arc around an object, from roughly 100° up to a
  full ring), the best bearing is no description of them at all, its sign
  comes from a weighted mean ray near zero, and the score at `ρ = 0` can be 0
  while `Λ` is in the hundreds of thousands. Any point `X` gives
  `Λ ≥ bearing_cost − cost(X)`, so the cost at the weighted linear midpoint
  `(Σ AᵢᵀAᵢ)⁻¹ Σ AᵢᵀAᵢ cᵢ` (the point minimising `Σ ‖Aᵢ (X − cᵢ)‖²`), when it
  lies in front of every camera, gives a rigorous lower bound on `Λ`. On the
  Kerry Park ground truth it is what calls 12 of the 375 finite points finite.
  For near-parallel rays the midpoint is a poor point and the bound is small,
  and the score decides.
- **Exact exits both ways.** `Λ ≤ bearing_cost`, so `bearing_cost < threshold`
  is a bearing whatever the point fit would find, and
  `midpoint_bound ≥ threshold` is a finite point. `is_finite` is finite when
  `bearing_cost` reaches the threshold and either the score or the bound does.

### Fitting

`fit_point_and_bearing` runs Levenberg-Marquardt over `(u, ρ)`, `u` moving in
its tangent plane and `ρ` clamped at 0. At the bound, a step that would take
`ρ` negative is replaced by a step in `u` alone. It accepts only steps that
lower the cost, so a fit costs no more than its start, and from a start in
front of every camera only steps that keep it in front.

The sine residual cannot tell an angle from its opposite, so `(u, ρ ≥ 0)` also
represents points behind every camera, and a fit can slide to a cheap point
behind one: with a camera 0.02 from the point and three at 5, a fit from the
midpoint did, leaving the bearing at `ρ = 0` as the only result in front. The
in-front rule on steps prevents that. The fit runs from `start` when one is
given, and when there is none, or that fit ends above the bearing's cost or
behind a camera, it also runs from the bearing at `ρ = 0` and from the
weighted linear midpoint (when that lies in front of every camera). When the
warm fit is kept, the midpoint is still refined if it costs less than that
fit. When one camera is very close to the point and the others spread wide,
neither the bearing nor the midpoint need be in front of every camera, and
then no start so far leads in front; the fit runs once more from the cheapest
point in front of every camera found along the rays themselves (on each ray,
the distance that best fits the other rays, and a ladder of distances from
10⁻⁶ to 10³ camera spreads). Without that start, 485 of 187,737 finite
verdicts in a sweep of such geometries came back with a point behind a camera;
with it, none does. Of the results that cost no more than the bearing, which
is always one of them, the fit keeps the cheapest in front of every camera,
or the cheapest when none is.

The midpoint's refinement is an in-front result costing no more than the
midpoint, so with plain least squares `Λ ≥ midpoint_bound` at any
`max_iterations` (none of 7 million adversarial fits fell below it), up to
round-off: the fit evaluates the midpoint's cost through `u + ρ (a − cᵢ)`, and
with an anchor 10⁶ from the cameras and no iterations that differs from the
direct cost by up to 4 parts per million. The one exception is a midpoint
exactly at the anchor, the one point the parametrisation cannot represent,
which the fit skips. `in_front_of_all_cameras` checks
`(u + ρ (a − cᵢ)) · dᵢ > 0` per ray, and `Λ` means nothing when it is false.

A finite verdict from `is_finite` and a fit that comes back without a usable
point can disagree when the score alone decided: the score is a linear
prediction at `ρ = 0`, and the fit is what places the point. A point is not
usable when there is none (`inverse_depth = 0`), when it is not in front of
every camera (`in_front_of_all_cameras` false), or when it lies closer to a
camera centre than a minimum depth the consumer sets. The last case comes
from the start along the rays: in near-camera geometry the in-front point the
fit finds can sit almost on the near camera's centre (532 of 729 fallback
starts in an adversarial sweep ended closer to a camera centre than a
thousandth of the near camera's distance from the point). That is the sine
cost's genuine minimum in front of
every camera, but a point at almost zero depth from a camera is not one to
store. None of the sweeps above gives a finite verdict without a point or with
one behind a camera, but a consumer that needs a position has a rule for all
three cases: it re-fits from the position it already holds when that lies in
front of every camera, and keeps the result if it is usable; otherwise it
treats the track as a bearing.

`depth_likelihood_ratio` is `Λ` at the optimum the fit reaches, which can fall
slightly short of the best point in front of every camera (232 of 30,000
adversarial fits, at worst by 0.08%, none changing the verdict). Decisions
come from the score and the bound, which do not depend on the fit.

The bearing has the same blind spot: rays along `u` and `−u` fit the bearing
`u` at no cost. `bearing_in_front_of_all_cameras` on `BearingScore` says
whether `dᵢ · bearing > 0` for every ray. A classifier does not store a bearing
that is behind a camera on the strength of a bearing verdict; it treats the
track as it treats a finite point behind a camera, pruning the sightings the
bearing is behind or dropping the track. The flag is often false on finite
tracks whose rays spread over a wide angle (63 of the Kerry Park ground
truth's 387 tracks), where the bearing describes nothing and the verdict is
finite.

Weights so large that the bearing's cost overflows make the track `None`
rather than give it an infinite cost.

With `soft_l1_scale` set, the bearing is first refined under the robust loss
from the closed-form one, held in front of every camera when the closed-form
bearing is (so it can stop short of a robust minimum on the far side of a
camera; no case of it has been measured), both costs are robust, and the
normal equations use
the second-order (Triggs) scaling bundle adjustment uses. The robust `Λ` is
smaller than the plain one wherever a residual is past the loss's scale (on the
Kerry Park ground truth, finite point 10 has `Λ` 2,745.9 plain and 661.8
robust, from bearing costs 2,753.0 and 668.8 and point costs 7.18 and 7.00;
point 50 has 85,424 and 5,892), so a report that sets `Λ` beside `depth_score` fits with
`soft_l1_scale: None`.

### The measured noise level

The weights are over a per-axis pixel noise `σ_px`, and a stored
reconstruction measures its own: the RMS per-axis reprojection residual over
the observations of its finite points. It lives in
[analysis/reprojection_noise.rs](../../../crates/sfmtool-core/src/analysis/reprojection_noise.rs).

```rust
pub struct ReprojectionNoise {
    /// √(Σ (du² + dv²) / 2n) over every counted observation; None when none.
    pub sigma_px: Option<f64>,
    pub observation_count: usize,
    /// The same over each camera's observations, indexed as image_table.cameras.
    pub per_camera_sigma_px: Vec<Option<f64>>,
    pub per_camera_observation_count: Vec<usize>,
}

impl SfmrReconstruction {
    pub fn reprojection_noise(&self) -> Result<ReprojectionNoise, ReconstructionError>;
    /// reprojection_noise()?.sigma_px.
    pub fn reprojection_noise_px(&self) -> Result<Option<f64>, ReconstructionError>;
}
```

```rust
let sigma_px = recon.reprojection_noise_px()?.expect("a finite point with observations");
```

- **Finite points only.** The test this measure feeds is about whether a
  track is a bearing, and a bearing's residuals are the residuals of the model
  under question. A reconstruction with no observation of a finite point has no
  measure, and the result is `None` rather than a guessed default.
- **RMS, not a robust spread.** `σ_px` has to cover pose, lens-model and
  keypoint error together, and those are heavier-tailed than a Gaussian. On
  the Kerry Park ground truth the robust spread is 0.137 px against an RMS of
  0.216 px, and at 0.137 px even its 13,000-unit bearings score up to 35.
  The plain RMS is sensitive to a few gross residuals and takes no account
  of the parameters bundle adjustment fitted; whether to trim outliers or
  correct for degrees of freedom is open in the
  [amendment draft](../../drafts/point-or-bearing-likelihood-ratio.md).
- **One value, and one per camera beside it.** The default everywhere is the
  overall value. The per-camera values cost one more accumulator per camera
  and are what a caller reads to decide whether one value is enough: the two
  Kerry Park lenses measure 0.2155 and 0.2157 px. Whether a capture whose
  cameras differ should weight each camera's rays by its own value is the
  draft's open question.
- **Where the pixel comes from.** An observation's pixel is the inline
  keypoint when the reconstruction carries the column (every
  `embedded_patches` file, and a `sift_files` one with the optional copy,
  which a load fills in from the `.sift` files), and otherwise the `.sift`
  position its feature index names, read from the workspace. Both go through
  the same lookup, `SfmrReconstruction::observation_pixels` in
  [recompute.rs](../../../crates/sfmtool-core/src/reconstruction/data/recompute.rs),
  whose `.sift` read (`tracked_sift_positions`) is also the one
  `compute_observation_reprojection_errors` makes. An
  observation is left out when its pixel is not finite or the camera model
  cannot image its point.

On the Kerry Park ground truth `tk117` it is 0.2156 px over 3,510
observations; on the in-repo seoul bull ground truth, 0.646 px over 1,233.

### Over a reconstruction

Every consumer of the test that holds a reconstruction (the reports, the
reclassification and discovery passes, the bench) builds a track's rays the
same way: each `(image, pixel)` observation through `observed_ray`, at that
image's camera and pose and the measured noise. Two pieces in
[analysis/point_or_bearing.rs](../../../crates/sfmtool-core/src/analysis/point_or_bearing.rs)
carry that. `observation_ray` and `track_rays` build rays from any
observations, stored or not: discovery's tracks are assembled from `.sift`
keypoints that belong to no point yet, and the bench's from sightings it has not
committed. Discovery builds its tracks with `track_rays`; the bench, which
holds each sighting's projected image, builds them with `track_rays` or with
`observation_ray` or `observed_ray` per sighting, which are equally direct.
Both call the batch functions on the result. `point_or_bearing_scores` is the convenience
over the stored points, built on `track_rays`, for the reports and the
reclassification pass.

```rust
/// One observation as the test reads it.
pub struct ObservationRay {
    pub dir: Vector3<f64>,
    pub center: Point3<f64>,
    /// (1/σ_px)·J·R, as observed_ray builds it.
    pub weight: Matrix2x3<f64>,
}

/// A batch of tracks of rays in the CSR layout of the batch functions (named
/// apart from the bench's one-track `TrackRays`).
pub struct RayBatch {
    pub dirs: Vec<Vector3<f64>>,
    pub centers: Vec<Point3<f64>>,
    pub weights: Vec<Matrix2x3<f64>>,
    pub offsets: Vec<usize>,
}

/// None for an index past the table, a non-finite pixel, or what observed_ray declines.
pub fn observation_ray(image_table: &ImageTable, image_index: usize, pixel: [f64; 2],
    sigma_px: f64) -> Option<ObservationRay>;
/// One output track per input track (CSR over observations), holding the
/// observations that give a ray.
pub fn track_rays(image_table: &ImageTable, observations: &[(usize, [f64; 2])],
    offsets: &[usize], sigma_px: f64) -> RayBatch;

pub struct PointOrBearingScores {
    /// The noise the rays were weighted by: the caller's, or reprojection_noise_px.
    pub sigma_px: f64,
    /// The points scored, in order: the caller's indexes, or every point.
    pub point_indexes: Vec<usize>,
    /// One per point index; None with fewer than two usable rays.
    pub scores: Vec<Option<BearingScore>>,
    /// With fit options given, the fits, aligned the same way.
    pub fits: Option<Vec<Option<PointBearingFit>>>,
}

pub enum PointOrBearingError {
    NoNoiseLevel,
    InvalidNoiseLevel(f64),
    PointIndexOutOfRange { index: usize, point_count: usize },
    Reconstruction(ReconstructionError),
}

impl SfmrReconstruction {
    pub fn point_or_bearing_scores(
        &self,
        point_indexes: Option<&[usize]>,
        sigma_px: Option<f64>,
        fit: Option<&PointBearingFitOptions>,
    ) -> Result<PointOrBearingScores, PointOrBearingError>;
}
```

```rust
let plain = PointBearingFitOptions { soft_l1_scale: None, ..Default::default() };
let result = recon.point_or_bearing_scores(None, None, Some(&plain))?;
for (k, &point) in result.point_indexes.iter().enumerate() {
    let finite = result.scores[k]
        .as_ref()
        .is_some_and(|s| is_finite(s, DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD));
    // result.fits.as_ref().unwrap()[k] holds the point and Λ.
}
```

- **The ray construction is separate from the stored points.** A track a
  caller assembles itself (discovery's, from `.sift` keypoints; the bench's,
  from its sightings) has no point index, so `track_rays` takes
  `(image, pixel)` observations in CSR form, keeps one output track per input
  track, and leaves the batch calls to the caller. `observation_ray` is the
  single-observation form for a caller adding one sighting at a time.
- **Aligned to the indexes asked for.** A report reads every point; a
  reclassification pass scores every point and then fits only the finite
  verdicts. Each gets results in the order it asked, repeats included, so no
  caller re-maps a compacted batch.
- **Points at infinity are scored.** They are the ones a finite verdict would
  promote, so the method does not filter on the stored `w`. A bearing's
  observations are rays like any other.
- **`None` rather than a dropped row.** An observation gives no ray when its
  pixel is not finite or `observed_ray` declines it (outside the model's
  domain); a point left with fewer than two rays is `None`, as in the batch
  functions.
- **Scoring and fitting share the rays.** The fit is optional, because
  deciding needs only the score. When it runs, each finite point starts from
  its stored position, which is the warm start the primitives take, and a
  point at infinity starts with none. The method passes the options through,
  so the caller chooses the loss; a report that sets `Λ` beside `depth_score`
  passes `soft_l1_scale: None` (see the robust loss under "Fitting").
- **σ defaults to the measured value**, and the result carries the value used,
  so a report can print it. A caller that wants another (an override on the
  command line, or bundle adjustment's per-round value) passes it. When it is
  measured, the pixels are read once for both the measure and the rays, so a
  `sift_files` reconstruction without the inline column reads each `.sift`
  file once.

On `tk117` the method reproduces the 12-bearing table of the amendment draft:
at the measured 0.2156 px, points 298, 294 and 295 score 132.48, 83.62 and 31.10
and are finite, 269 and 270 score 14.40 and 11.67, and none of the 375 finite
points gets a bearing verdict (12 of them are finite on the midpoint bound
alone). At the 0.216 px the draft used, the scores are the draft's 131.97,
83.30, 30.98, 14.34 and 11.63; the scores scale as `1/σ²`. On the seoul bull
ground truth no finite point gets a bearing verdict and none of its 14 bearings
gets a finite one.

## Python bindings

Batch-first and numpy-friendly, matching the existing `read_*` dict-of-arrays
convention rather than returning M Python objects:

```python
# triangulate a batch given rays you already have
out = triangulate_batch(dirs, centers, offsets)  # dict of arrays:
#   points (M,3) f64, eigenvalues (M,3) f64, condition_number (M,) f64, in_front_of_all_cameras (M,) bool

# convenience over an existing reconstruction's stored points (camera→point
# rays, no .sift reads) — for inspect / analyze / notebooks
diag = recon.triangulation_diagnostics(noise_px=1.0)  # dict of arrays incl.
#   condition_number (M,), depth_sigma (M,), inverse_depth_z (M,)
```

The point-or-bearing primitives are in `sfmtool._sfmtool.analysis`, bound in
[analysis/point_or_bearing.rs](../../../crates/sfmtool-py/src/analysis/point_or_bearing.rs),
with the CSR layout above and one `(2, 3)` weight per ray:

```python
from sfmtool._sfmtool.analysis import (
    bearing_score_batch, fit_point_and_bearing_batch, observed_rays,
    DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD, DEFAULT_SOFT_L1_SCALE,
    DEFAULT_POINT_FIT_MAX_ITERATIONS,
)

# one camera's observations: rotations (N,4) wxyz, pixels (N,2)
rays = observed_rays(camera, cam_from_world_wxyz, pixels, sigma_px)
#   valid (N,) bool, dirs (N,3), weights (N,2,3); NaN rows where not valid

score = bearing_score_batch(dirs, centers, offsets, weights, threshold=25.0)
#   scored (M,) bool, bearing (M,3), bearing_cost, depth_score, midpoint_bound (M,),
#   bearing_in_front_of_all_cameras (M,) bool, num_views (M,) int64, is_finite (M,) bool

fit = fit_point_and_bearing_batch(dirs, centers, offsets, weights,
                                  starts=None, anchors=None,
                                  soft_l1_scale=3.0, max_iterations=20)
#   fitted (M,) bool, bearing (M,3), bearing_cost, anchor (M,3), direction (M,3),
#   inverse_depth, point (M,3; NaN where inverse_depth is 0), point_cost,
#   depth_likelihood_ratio (M,), in_front_of_all_cameras (M,) bool, num_views (M,) int64

recon.reprojection_noise_px()  # float or None
recon.reprojection_noise()     # sigma_px, observation_count,
                               # per_camera_sigma_px (C,), per_camera_observation_count (C,)
out = recon.point_or_bearing_scores(point_indexes=None, sigma_px=None, fit=False,
                                    threshold=25.0, soft_l1_scale=None,
                                    max_iterations=20)
#   bearing_score_batch's dict, one row per point index, plus sigma_px and
#   point_indexes (M,) int64; with fit=True, out["fit"] is
#   fit_point_and_bearing_batch's dict, aligned the same way
```

- **Keys are the Rust field names**, `bearing_in_front_of_all_cameras` and
  `in_front_of_all_cameras` included, as `triangulate_batch`'s are, so a
  reader moving between the two layers finds the same word. A row the core
  returns `None` for is `scored` (or `fitted`) false, NaN in its float
  columns, false in its flags and 0 in `num_views`.
- **`is_finite` is a column, not a function.** The verdict is the core's
  `is_finite` at the `threshold` argument, so the rule has one implementation.
- **The fit is a nested dict** in `point_or_bearing_scores`, under `fit`. Its
  `bearing`, `bearing_cost` and `num_views` would otherwise collide with the
  score's, and they differ under a robust loss.
- **Defaults follow the caller.** `fit_point_and_bearing_batch` defaults to
  the core's soft-L1 scale, as the primitive does; `point_or_bearing_scores`
  defaults to plain least squares, because what it is for is setting `Λ`
  beside `depth_score`.
- **`observed_rays` takes one camera** and a rotation per pixel, so a caller
  with several cameras calls it once per camera. A pixel outside the camera
  model's domain is `valid` false rather than an error.

The GUI (`sfm-explorer`, Rust) consumes the core functions directly; the binding
is for the CLI/inspect/analyze/notebook paths.

## Consumers

**Points at infinity.** `analysis/infinity/discover.rs::classify_track`,
`analysis/infinity/convert.rs::classify_points_at_infinity` and the track
bench's `bench/classify.rs::classify_track_rays` share
`classify_rays_at_infinity`, which decides finite-versus-infinity on
`inverse_depth_z` with `condition_number` as a cheap geometric pre-filter. It is
`pub` for that third caller: every bench step that triangulates -- the
track-stage fit and the cluster-to-track upgrade
([`../bench/editable-track.md`](../bench/editable-track.md) § "Finite points and
bearings") -- ends at it with these same defaults, so a hand-edited track and a
whole-reconstruction pass cannot come to disagree about which representation one
set of rays has earned. The bench differs in two places, both about what to do
with an answer rather than about how to reach one: a relabel-only pass leaves an
`Indeterminate` track as the solve produced it, while a fit has to write
something and writes the bearing the numbers describe; and the bench then
reprojects both candidates into the sightings and keeps the criterion's answer
only where the pixels agree, because this criterion says whether a depth is
*observable* and an ill-conditioned midpoint can clear its own bar on a depth
that is not there. `RayClassification` carries the triangulated `point` beside
its `class` for that second caller: a caller that refuses the depth may still
want to know what it refused. The
thresholds — `DEFAULT_INVERSE_DEPTH_Z_CUTOFF = 4.0` and
`CONDITION_NUMBER_PREFILTER = 1e4` — live in
[`analysis/infinity/convert.rs`](../../../crates/sfmtool-core/src/analysis/infinity/convert.rs)
and are provisional; calibrating them against larger datasets is an open question
below. The noise floor reaches the CLI as the fourth component of
`--find-points-at-infinity`
(`eps_deg[,desc_thresh[,min_views[,noise_floor_px]]]`).

**Reports.** Per-point depth reliability appears in `sfm inspect --verbose` and in
`sfm analyze --depth-reliability`, both through the PyO3 surface below.

**The GUI** consumes the core functions directly. Two overlay modes back onto
these diagnostics — "Depth Reliability", driven by `inverse_depth_z` (low =
near-infinity, unconstrained depth), and "Condition Number" on a log scale —
computed per point by `metrics::compute_point_diagnostics` via
`triangulate_batch` / `depth_uncertainty_batch`. The same numbers appear in the
point-track header and in the Image Detail tooltip, next to the max track angle.

## Decisions

- **Always return geometric diagnostics; opt-in statistical.** Eigenvalues +
  condition number are byproducts of the solve, so returning them by default
  adds no work and keeps a depth's reliability attached to the point. Depth σ /
  inverse-depth z need a noise model, so they are a second batch call.
- **Decision variable is `inverse_depth_z` (scale-free), not condition number**
  (grows with K). Condition number is the cheap geometric flag.
- **Gate confident finite/∞ calls on `resolvable_distance ≥ finite_horizon`;
  otherwise `indeterminate`.** `inverse_depth_z` divides by the solved depth and
  goes unstable in the no-baseline regime, so it cannot stand alone there.
  `finite_horizon` (default = camera extents) anchors the call to a stable,
  triangulation-independent scale. See "Scene-relative resolvability".
- **Indeterminate tracks are dropped, not represented.** The `.sfmr` model stays
  binary (`w=1` finite / `w=0` at infinity); a track that fails the
  `resolvable_distance` gate is not given a third state. In discovery it is
  dropped (never appended). `classify_points_at_infinity` stays non-destructive:
  it only demotes confident-infinity points to `w=0` and leaves an indeterminate
  solve point as the finite point the solve produced — it never removes a point.
- **Ray source is the caller's concern.** Discovery (`find`) supplies
  keypoint-un-projected rays; reclassify/GUI supply camera→stored-point rays
  (cheap, no `.sift`). The core function is agnostic.
- **Do not persist diagnostics to `.sfmr`.** Derivable from geometry; storing
  per-point would bloat the format and go stale.

## Open questions

- Threshold calibration (deferred until after the diagnostics land): the
  `inverse_depth_z` cutoff (≈3-4?) and any `condition_number` pre-filter (≈1e4?)
  are provisional, taken from the KerryPark360 population split (genuine z≈62 vs
  discovered z≈3). The plan is to implement the diagnostics first, then sweep the
  cutoff on several larger captures and pick a value. The in-repo fixtures are
  too small and lack enough genuinely-distant content to populate the infinity
  regime, so they cannot calibrate this; they only confirm the cache/plumbing.
  KerryPark360 evaluation since showed the bare cutoff is *unstable* in the
  no-baseline regime (see "Scene-relative resolvability"), motivating the
  `resolvable_distance ≥ finite_horizon` gate and the `indeterminate` state. Open
  sub-questions: the `finite_horizon` multiple of the camera extents (1×? a
  fraction?), and whether to compute the precise per-track perpendicular baseline
  `B⊥` or accept the scalar camera-extent upper bound.
- Noise model: per-camera `noise_px` default, and whether to fold per-point
  reprojection error into σ as `classify_points_at_infinity` does today
  (`noise = max(reproj_error, floor)`). Discovered points carry their mean
  reprojection error against the appended track, so the same fold is
  available to them. Deciding on the likelihood ratio of "Point or bearing"
  instead, with σ measured from the reconstruction's finite points and no
  floor, is proposed in
  [point-or-bearing-likelihood-ratio.md](../../drafts/point-or-bearing-likelihood-ratio.md).
- Weighted vs unweighted midpoint as the default (unweighted matches current
  behavior; inverse-depth² is closer to reprojection error).

## Reuse map

| Need | Existing piece |
|---|---|
| pixel → world ray (all models, fisheye) | `CameraIntrinsics::pixel_to_ray[_batch]` |
| camera center | `SfmrImage::camera_center` (`= −Rᵀt`) |
| max pairwise angle (pre-filter only) | `geometry/viewing_angle.rs::max_viewing_angle` |
| existing 2-view algebraic triangulation | `features/feature_match/geometric_filter.rs::triangulate_point_dlt` |
| bearing-mean fallback for `w = 0` | `analysis/infinity/convert.rs` (`normalise(Σ rᵢ)`) |
| per-track observation slices (CSR) | `observation_offsets`, indexed directly |

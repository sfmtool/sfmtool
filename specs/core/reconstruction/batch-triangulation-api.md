# Batch Triangulation API with Observability Diagnostics

Triangulating a track means finding the 3D point that best fits the rays its
observations cast, and the linear solve that does so also reveals how
well-determined that point is — a track seen from one direction only pins down
two of its three coordinates. This is one batch API over whole sets of tracks
that returns each solved point **plus the observability diagnostics the solve
already computes** (the normal matrix's spectrum, and an optional
noise-calibrated depth uncertainty), so a caller reads a depth's reliability off
the conditioning instead of re-deriving it. Whether a track is stored as a
finite point or as a bearing is decided by a likelihood-ratio test in the same
module (§ "Point or bearing"), which reclassification, discovery and the track
bench all read. The older signal for that decision, the *maximum pairwise
viewing angle*, is an extreme order statistic that keypoint noise inflates and
that **grows with view count**, so genuine points at infinity with many
observations were misclassified as finite.

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
reusable batch operation that returns its conditioning, and give the
classifiers a statistic that does not grow with the view count. They decide on
the likelihood-ratio test of "Point or bearing", which fits a bearing and a
point to the same rays; the conditioning and the depth uncertainty are the
diagnostics beside it.

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

### Diagnostics: the depth uncertainty and the geometric flag

The condition number is free but **not track-length invariant** (λ_max ≈ K for
near-parallel rays), so a threshold on it drifts with view count. The
scale-free reading is the **depth uncertainty**, which needs a per-ray angular
noise σ (e.g. `noise_px / fᵢ`) -- a *policy* input that does not belong inside
the geometric solver. It is a separate, opt-in batch step:

```rust
/// Depth uncertainty along the mean viewing direction, from the inverse-
/// variance-weighted normal matrix. `sigma_rad` is per-ray angular noise.
pub struct DepthUncertainty {
    pub depth: f64,
    pub sigma: f64,
    /// inverse-depth z-score = depth / sigma. Small (≲ 3-4) ⇒ statistically
    /// indistinguishable from infinity. (kerry_park medians: genuine 62,
    /// discovered "finite" 3.) Reliable only when the solve is non-degenerate
    /// — it divides by the solved depth, which is noise when the rays are
    /// near-parallel. See "Scene-relative resolvability".
    pub inverse_depth_z: f64,
    /// Farthest depth this track's geometry can tell from infinity:
    /// `B⊥ / σ` (perpendicular camera baseline over angular noise) — equivalently
    /// the depth at which `inverse_depth_z` would fall to 1. Independent of the
    /// (possibly garbage) solved depth, so it stays meaningful when the rays are
    /// near-parallel.
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

These are **diagnostics**. What decides whether a track is stored as a finite
point or a bearing is the likelihood-ratio test of "Point or bearing" below, of
which `inverse_depth_z` is the Wald form; the reports print `z` beside it, and
the GUI's Depth Reliability overlay colours by it.

### Scene-relative resolvability

`inverse_depth_z` is reliable only when the solve is non-degenerate. On the
KerryPark360 capture — a walk with frequent "stop and look around" pauses — it
breaks in the no-baseline regime, and it breaks two-sided:

- A genuinely distant point seen only from one stop reads as near infinity
  (right by accident).
- A *near* point seen only from one stop can read as resolved. Example
  `pt3d_…_102031`: 23 observations, all from one stop (observing-camera spread
  0.35 of a 165-unit capture). The near-parallel rays have no real
  intersection, so the least-squares point falls to range 0.24 — *inside* the
  camera cluster — and `inverse_depth_z` came out 4.97. Leave-one-out swings it
  between −5 and +5: it is noise. Its mirror image `pt3d_…_97221` is the same
  situation falling the other way.

The cause is structural: `inverse_depth_z = depth / σ_depth ≈ (B⊥/σ) / depth`,
so it divides by the *solved depth*, which is a noise-driven value when the
rays are near-parallel. The **resolvable distance** `D_max = B⊥ / σ` — the
perpendicular camera baseline over the angular noise, the depth at which
`inverse_depth_z` would fall to 1 — does not depend on the solved depth, so it
stays meaningful exactly where `inverse_depth_z` goes unstable.
`depth_uncertainty_batch` returns it as `resolvable_distance`.

Read against the **camera extents** (`camera_extents`, the bounding-box
diagonal of the camera centres, called `finite_horizon` where it is read this
way), it says whether the capture can resolve a point at its own scale: a track
whose `resolvable_distance` is under the extents has a **short baseline**. The
extents are the reference because camera centres come straight from the solved
poses and do not move when a point is mis-triangulated, whereas the point-cloud
extent is polluted by the artifacts being diagnosed. Discovery counts the
bearings it appends that have a short baseline
(`InfinityDiscovery::short_baseline`), and `sfm inspect pt3d_*` prints the
comparison. Parallax comes only from the camera spread *perpendicular to a
point's bearing*, so the scalar extents are a coarse upper bound: falling short
of them means unresolvable, but reaching them does not guarantee
resolvability in every direction.

The likelihood-ratio test needs no such gate: a track the cameras cannot
resolve gets `Λ ≈ 0` and a bearing verdict, which is what its rays say (see
"Consumers"). And among tracks with real baseline, finite against infinity is a
question of resolvability, not distance. A KerryPark360 pair:

| | `pt3d_…_108877` | `pt3d_…_96414` |
|---|---|---|
| range | **261** (beyond the 165 extent) | 122 |
| `inverse_depth_z` | **4.06** | 2.48 |
| observing-camera baseline span | **10.6** | 3.7 |
| views | 50 | 17 |

The farther point has the larger `z`: its observing cameras span 10.6 against
96414's 3.7, so even at range 261 its parallax is significant.

## Point or bearing

The z test above asks whether the solved depth is far from zero relative to
its linearised uncertainty. A second family in the same module asks the
question a stored track's representation depends on more directly: does giving
the track a depth explain its rays better than a direction alone does, by more
than the noise could explain? It fits both models to the track's rays and
compares their costs, which is a likelihood-ratio test between two nested
models. A stored reconstruction supplies the noise level from its own residuals
("The measured noise level") and runs the test over its points with one method
("Over a reconstruction"), and both are bound to Python. Reclassifying a
reconstruction's points and discovering new points at infinity decide on it,
so does the track bench and bundle adjustment's storage decision for the free
points it solves in inverse depth, and the `analyze` and `inspect` reports print
it (see "Consumers").

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
  project back to within 1e-3 px of it (a pixel past the start of a
  fisheye's wide-angle blend, where un-projection deliberately moves toward
  the equidistant ray, or past a fold of the distortion), and where a
  difference probe leaves the model's domain.
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
- **Score against `Λ`.** A test of a nested parameter has three classical
  forms: the Wald form at the full model's fit (`inverse_depth_z`, the fitted
  depth over its linearised deviation), the likelihood ratio from both fits
  (`Λ`), and the score form at the restricted model's fit alone
  (`depth_score`). With one `σ` they agree asymptotically and differ in where
  their linearisation is accurate. Wald linearises at the fitted depth, which
  for a far point is mostly noise and has a lopsided uncertainty; the score
  linearises at `ρ = 0`, where the inverse-depth model is close to linear and
  where the verdict is in question. On the Kerry Park ground truth `tk117` at
  `σ = 0.216` px, the score from `bearing_score` (sine residuals, weights from
  `observed_ray`) against `Λ` from fits of both models to the pixel residuals
  through the camera model:

  | Point | Views | `depth_score` | `Λ` (pixel residuals) |
  |---|---|---|---|
  | 154 | 26 | 1.33 | 1.32 |
  | 153, 155 | 25 | 4.19, 4.21 | 4.19, 4.21 |
  | 157 | 25 | 6.76 | 6.76 |
  | 268 | 26 | 7.94 | 7.95 |
  | 176 | 21 | 8.27 | 8.27 |
  | 156 | 24 | 9.47 | 9.46 |
  | 270 | 17 | 11.63 | 11.62 |
  | 269 | 7 | 14.34 | 14.34 |
  | 295 | 18 | 30.98 | 30.93 |
  | 294 | 21 | 83.30 | 83.23 |
  | 298 | 10 | 131.97 | 132.04 |
  | 10 (finite, 81 units) | 9 | 2,732 | 2,762 |
  | 50 (finite) | 13 | 79,711 | 85,331 |

  The twelve bearings stored in that file are the first eleven rows. The two
  forms agree to the first decimal wherever the verdict is in question and part
  only where both are thousands of times the threshold, or where the rays
  spread so wide that the bound decides (point 91, 4 views over 105°: score 0,
  `Λ` 635,749). Point 298 is the case that motivated the test: its finite point
  explains all ten views to a third of a pixel (0.16 px mean, against 0.62 px
  for the bearing), while the z rule, at its 1 px per-ray noise floor, gave it
  `z = 2.23`, under its cutoff of 4.
- **Why a 2×3 weight per ray.** The Kerry Park fisheyes stretch pixels per
  radian by up to 2.1 to 1 between the radial and tangential directions, and
  one scalar angular noise per ray misplaces the statistic: for point 298,
  160.9 with the geometric mean of the two stretches and 100.8 with `σ_px`
  over the focal length, against 132.0 from pixel residuals through the camera
  model. With the 2×3 weight `observed_ray` builds from the camera model's
  projection derivative, the score and the fitted `Λ` reproduce the
  pixel-residual values (298: 132.0 and 132.0; 294: 83.3 and 83.3; 10: 2,732
  and 2,746 against 2,762).

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

Camera centres that coincide to round-off are one centre. A track's centres
whose offsets from their centroid are all within `1e-12` of their distance
from the origin are made exactly equal, at the centroid, before the test runs,
so `ρ` has no direction, the depth score is 0, there is no midpoint, and the
fit's `Λ` is 0. The comparison is with the centroid rather than the anchor,
which a caller may place anywhere. Such offsets
are what a solver leaves when it collapses a run of frames onto one centre
(the poses keep their rotations and the centres end a few ulps apart), and
the score does not depend on the scale of the offsets: without the rule its
verdict on those tracks was decided by the round-off. On eight collapsed
seoul bull points, four scored 0 and four 854 to 913, while the exact fit's
`Λ` was 0 for all of them. The tolerance is relative, so it also takes a real
baseline under `1e-12` of the centres' distance from the origin for
round-off: under 1 µm for centres a million units out, about 6 µm in
metre-scale ECEF coordinates. Double precision holds such a centre to about
`2e-16` of that distance, so four orders of magnitude separate the round-off
the rule removes from the smallest baseline it keeps; a capture far from its
origin with sub-tolerance baselines should be re-centred. Consistent rays from one centre, as from a camera
that pans in place, give `Λ ≈ 0` and a bearing with or without the rule.

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
the observations of its finite points, with gross outliers left out. It lives
in
[analysis/reprojection_noise.rs](../../../crates/sfmtool-core/src/analysis/reprojection_noise.rs).

```rust
/// How many robust spreads a residual may be before it is an outlier.
pub const OUTLIER_GATE: f64 = 30.0;

pub struct ReprojectionNoise {
    /// √(Σ (du² + dv²) / 2n) over every counted observation; None when none.
    pub sigma_px: Option<f64>,
    pub observation_count: usize,
    /// Observations left out because their residual passed the gate.
    pub outlier_count: usize,
    /// The same over each camera's observations, indexed as image_table.cameras.
    pub per_camera_sigma_px: Vec<Option<f64>>,
    pub per_camera_observation_count: Vec<usize>,
}

impl SfmrReconstruction {
    pub fn reprojection_noise(&self) -> Result<ReprojectionNoise, ReconstructionError>;
    /// reprojection_noise()?.sigma_px.
    pub fn reprojection_noise_px(&self) -> Result<Option<f64>, ReconstructionError>;
}

/// The finest pixel coordinate an f32 keypoint states in any camera; every
/// measure above is raised to it.
pub fn keypoint_resolution_px(recon: &SfmrReconstruction) -> f64;
```

```rust
let noise = recon.reprojection_noise()?;
let sigma_px = noise.sigma_px.expect("a finite point with observations");
println!("{sigma_px:.4} px, {} outliers left out", noise.outlier_count);
```

- **Finite points only.** The test this measure feeds is about whether a
  track is a bearing, and a bearing's residuals are the residuals of the model
  under question. A reconstruction with no observation of a finite point has no
  measure, and the result is `None` rather than a guessed default.
- **Only observations the test reads.** An observation is left out when its
  pixel is not finite, when the camera model cannot image its point, and when
  `observed_ray` declines its pixel (outside the camera model's domain), since
  the test then gives it no ray either. Before this rule, an observation the
  test never read could set the noise level the test read the others at: two
  keypoints moved to 1e9 px made `σ` 7e7 px.
- **An RMS, gated by a robust spread.** `σ_px` has to cover pose, lens-model
  and keypoint error together, and those are heavier-tailed than a Gaussian,
  so it is an RMS rather than a robust spread. On the Kerry Park ground truth
  the robust spread is 0.137 px against an RMS of 0.216 px, and at 0.137 px
  even its 13,000-unit bearings score up to 35. A few gross residuals,
  mismatched keypoints rather than noise, can still dominate an RMS, so each
  camera's robust spread `s = 1.4826 · median |r|` over the per-axis
  components of its residuals sets a gate, and an observation whose residual
  is longer than `OUTLIER_GATE · s` is counted in `outlier_count` and left out.
  The gate is per camera so that a noisier camera keeps its own tail.
- **Why the gate is 30 spreads.** The residuals of real solves are not
  Gaussian, and a gate set for a Gaussian (4 to 6 spreads) cuts into the tail
  `σ` is meant to cover. Measured on the Kerry Park ground truth `tk117`, the
  in-repo seoul bull ground truth, and `inf2`, a 17-image seoul bull
  `sift_files` solve with 365 discovered points at infinity, each estimator's
  `σ` (with the observations a gate leaves out) and the stored points whose
  verdict at the default threshold disagrees with their storage (points at
  infinity called finite / finite points called bearings):

  | Estimator | `tk117` | seoul bull ground truth | `inf2` |
  |---|---|---|---|
  | Plain RMS | 0.2156 (3 / 0) | 0.6461 (0 / 0) | 0.3593 (34 / 0) |
  | Gate at 5 spreads | 0.1652, 129 out (3 / 0) | 0.2461, 80 out (1 / 0) | 0.2858, 101 out (48 / 0) |
  | Gate at 10 | 0.2014, 14 out (3 / 0) | 0.3133, 35 out (0 / 0) | 0.3445, 9 out (36 / 0) |
  | Gate at 20 | 0.2156, 0 out (3 / 0) | 0.4239, 8 out (0 / 0) | 0.3593, 0 out (34 / 0) |
  | **Gate at 30** | **0.2156, 0 out (3 / 0)** | **0.4677, 4 out (0 / 0)** | **0.3593, 0 out (34 / 0)** |
  | RMS less the top 1% | 0.1905 (3 / 0) | 0.4002 (0 / 0) | 0.3237 (42 / 0) |
  | RMS less the top 2% | 0.1783 (3 / 0) | 0.3446 (0 / 0) | 0.3038 (45 / 0) |
  | Robust spread (MAD) | 0.1370 (5 / 0) | 0.2046 (1 / 0) | 0.2363 (61 / 0) |

  The two bundle-adjusted solves have their largest residuals at 18.8 and
  13.1 spreads; a gate at 5 leaves out 3.7% and 3.3% of their observations
  and lowers `σ` by 23% and 20%, and trimming a fixed fraction lowers it
  whether or not anything is gross. On the ground truth the largest four
  residuals, 7 to 16 px at 34 to 79 spreads, stand apart from the rest of its
  tail (27 spreads and under) and carry about half of `Σ e²`. A gate at 30 is
  the one in the table that leaves both solves' tails whole and removes
  exactly those four. Its `σ` gives the same verdicts as the plain RMS on all
  three files; on the ground truth it moves bearing 188's score from 3.6 to
  6.9, and that point turns finite only below 0.246 px.
- **No degrees-of-freedom correction.** The RMS divides by the `2n` residual
  components and takes no account of the parameters fitted to them, so for a
  least-squares fit with Gaussian noise it understates `σ`; the textbook
  correction `2n − 3P − 6I + 7` (`P` finite points, `I` images, less the 7 of
  the similarity gauge) would raise it by 1.118 on `tk117` and 1.25 on the
  seoul bull. It is not applied. The parameter count is not something a file
  records: rigs share poses, intrinsics are released or held per camera, and a
  ground truth or a hand-edited file was not fitted as one least-squares
  problem at all. And the correction is one factor on `σ` for the whole
  reconstruction, which scales every score by `1/factor²`: a change of the
  threshold, which is calibrated against `σ` as measured here (see "Open
  questions").
- **One value, and one per camera beside it.** The default everywhere is the
  overall value. The per-camera values cost one more accumulator per camera
  and are what a caller reads to decide whether one value is enough: the two
  Kerry Park lenses measure 0.2155 and 0.2157 px. Whether a capture whose
  cameras differ should weight each camera's rays by its own value is open
  (see "Open questions").
- **Where the pixel comes from.** An observation's pixel is the inline
  keypoint when the reconstruction carries the column (every
  `embedded_patches` file, and a `sift_files` one with the optional copy,
  which a load fills in from the `.sift` files), and otherwise the `.sift`
  position its feature index names, read from the workspace. Both go through
  the same lookup, `SfmrReconstruction::observation_pixels` in
  [recompute.rs](../../../crates/sfmtool-core/src/reconstruction/data/recompute.rs),
  whose `.sift` read (`tracked_sift_positions`) is also the one
  `compute_observation_reprojection_errors` makes.
- **Never finer than a keypoint can be stored.** Each measure is raised to
  the keypoint resolution of its cameras (`keypoint_resolution_px`): the
  `f32` machine epsilon times the camera's larger dimension, about 10⁻⁴ px on
  a 1,000 px image, the finest pixel coordinate an `f32` keypoint states. A
  measure under it is round-off, and a reconstruction whose keypoints are its
  points' exact projections measures 0, which would weight every ray
  infinitely. Real captures measure tenths of a pixel, so the bound binds only
  on exact data, where every consumer of the test then calls any track with
  parallax a point; because it is in the measure, reclassification, discovery
  and the bench read one level there too. It is a bound of representation and
  not a noise floor (see the glossary).
- **Rough inputs: a known limitation.** Measured by perturbing converged
  solves (the seoul bull `sift_files` solve and `tk117`) without
  re-adjusting them: added Gaussian keypoint noise is followed closely, the
  measured `σ` matching `√(σ₀² + σ²)` (0.437 against 0.438, 1.061 against
  1.063, 3.969 against 4.016 px), and with 1% of observations moved 10 to
  50 px at converged noise the gate leaves out every moved one. Once pose
  error dominates the residuals, though, the robust spread grows with it and
  the gate stops catching such outliers (0 of 31 on the seoul bull solve, 0 of
  35 on `tk117`), which raises `σ` by about 13 to 15% on the seoul bull solve
  and 5% on `tk117` (over different random draws). The verdicts degrade the way
  a larger `σ` makes them: more tracks get bearing verdicts. The measure is
  for a reconstruction that has been bundle-adjusted; on a rough one it
  overstates the noise. Bundle adjustment reads the same estimator on its final
  round's kept observations, which the round's trim has already capped at its
  `trim_px`, so a gross outlier the gate passes there adds a bounded amount
  ([bundle-adjustment.md](../geometry/bundle-adjustment.md) § "Free points:
  inverse depth and the storage decision").

On the Kerry Park ground truth `tk117` it is 0.2156 px over 3,510
observations with none left out; on the in-repo seoul bull ground truth,
0.4677 px over 1,229, four left out as outliers.

### Over a reconstruction

Every consumer of the test that holds a reconstruction (the reports, the
reclassification and discovery passes, the bench) builds a track's rays the
same way: each `(image, pixel)` observation through `observed_ray`, at that
image's camera and pose and the measured noise. Two pieces in
[analysis/point_or_bearing.rs](../../../crates/sfmtool-core/src/analysis/point_or_bearing.rs)
carry that. `observation_ray` and `track_rays` build rays from any
observations, stored or not: discovery's tracks are assembled from `.sift`
keypoints that belong to no point yet, and the bench's from sightings it has not
committed. Discovery builds its tracks with `observation_ray`, one member
keypoint at a time, so that it keeps the members that give a ray, and calls the
batch functions on the result; the bench, which holds each sighting's projected
image rather than the image table, builds one track with `observed_ray` per
sighting (`TrackRays::of_sightings`) and calls the single-track functions. `point_or_bearing_scores` is the convenience
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
  command line) passes it. When it is
  measured, the pixels are read once for both the measure and the rays, so a
  `sift_files` reconstruction without the inline column reads each `.sift`
  file once.

On `tk117` the method reproduces the table of "Score against `Λ`" under
"Theory": at the measured 0.2156 px, points 298, 294 and 295 score 132.48,
83.62 and 31.10 and are finite, 269 and 270 score 14.40 and 11.67, and none of
the 375 finite points gets a bearing verdict (12 of them are finite on the
midpoint bound alone). At the table's 0.216 px the scores are its 131.97,
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
recon.reprojection_noise()     # sigma_px, observation_count, outlier_count,
                               # per_camera_sigma_px (C,), per_camera_observation_count (C,)
out = recon.point_or_bearing_scores(point_indexes=None, sigma_px=None, fit=False,
                                    threshold=25.0, soft_l1_scale=None,
                                    max_iterations=20)
#   bearing_score_batch's dict, one row per point index, plus sigma_px and
#   point_indexes (M,) int64; with fit=True, out["fit"] is
#   fit_point_and_bearing_batch's dict, aligned the same way
reclassified, summary = recon.classify_points_at_infinity(sigma_px=None)
#   summary: sigma_px (float or None), noise (reprojection_noise()'s dict when
#   measured, else None), promoted, demoted, kept, refitted,
#   bearing_behind_camera, no_usable_point, unscored
found, summary = recon.find_points_at_infinity(eps_deg, desc_thresh=200.0,
    ratio=0.8, min_views=2, max_features=None, sigma_px=None)
#   summary: sigma_px, noise as above, candidates, bearings, short_baseline,
#   finite, bearing_behind_camera, unscored
```

`OUTLIER_GATE` and `DEFAULT_MIN_DEPTH_FRACTION` are in `sfmtool._sfmtool.analysis`
as `REPROJECTION_NOISE_OUTLIER_GATE` and `DEFAULT_MIN_DEPTH_FRACTION`.
`classify_points_at_infinity` and `find_points_at_infinity` raise `ValueError`
for a `sigma_px` that is not finite and positive and `OSError` when the pixels
cannot be read (and `find_points_at_infinity` for an `embedded_patches`
reconstruction, which has no `.sift` files to search).

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

**Reclassification.** `SfmrReconstruction::classify_points_at_infinity` in
[analysis/infinity/convert.rs](../../../crates/sfmtool-core/src/analysis/infinity/convert.rs)
decides every point of a reconstruction, finite or at infinity, with the test
of "Point or bearing", and stores each one the way its verdict says.

```rust
/// The minimum depth of a promoted point, as a fraction of the median distance
/// from a finite point to an observing camera's centre.
pub const DEFAULT_MIN_DEPTH_FRACTION: f64 = 0.01;

pub struct InfinityReclassification {
    /// The caller's noise level, or the measured one; None when neither exists.
    pub sigma_px: Option<f64>,
    /// The measurement behind sigma_px, when the pass measured it.
    pub noise: Option<ReprojectionNoise>,
    pub promoted: usize,
    pub demoted: usize,
    pub kept: usize,
    /// Finite points moved off a stored position no point can have.
    pub refitted: usize,
    /// Bearing verdicts left finite: the bearing is behind a camera.
    pub bearing_behind_camera: usize,
    /// Finite verdicts left at infinity: the fit gave no usable point.
    pub no_usable_point: usize,
    /// Finite points stored where no point can be, with no usable fit and a
    /// bearing behind a camera: left where they are.
    pub left_unusable: usize,
    /// Fewer than two usable rays.
    pub unscored: usize,
}

impl SfmrReconstruction {
    pub fn classify_points_at_infinity(
        &self,
        sigma_px: Option<f64>,
    ) -> Result<(Self, InfinityReclassification), PointOrBearingError>;
}
```

```rust
let (reclassified, summary) = recon.classify_points_at_infinity(None)?;
println!("{} promoted, {} demoted", summary.promoted, summary.demoted);
```

Each point's observations become rays through `track_rays` at `sigma_px` (the
measured noise level when `None`), each track is scored with
`bearing_score_batch`, and `is_finite` at
`DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD` gives the verdict. A point is
*usable* where it lies in front of every observing camera, along the ray its
pixel gives, and no closer to any of their centres than the minimum depth.

- **Agreement keeps the point.** A finite point with a finite verdict at a
  usable stored position, and a point at infinity with a bearing verdict, are
  left exactly as stored: the pass does not re-place points it agrees with.
- **Demotion.** A finite point with a bearing verdict becomes `w = 0` at the
  score's closed-form bearing. When that bearing is behind an observing camera
  it does not describe that sighting, and the point is left finite and counted
  in `bearing_behind_camera`: the pass removes no point and prunes no
  sighting, so the "prune or drop" of "Fitting" is left to a caller that may.
- **Promotion.** A point at infinity with a finite verdict is placed by
  `fit_point_and_bearing` with plain least squares (the fit the score
  approximates) and no start, since a bearing holds no position. A usable
  result becomes `w = 1` there; otherwise the point stays a bearing and is
  counted in `no_usable_point`. This is the consumer rule of "Fitting" for a
  caller holding no position. The fit's own `Λ` must also reach the threshold:
  the score is a linear prediction at `ρ = 0` and can exceed `Λ` where the
  geometry is degenerate, and the point stored is the fit's. On `tk117`, the
  seoul bull ground truth and `inf2` no promotion fails it.
- **A finite point stored where no point can be.** A finite verdict on a
  finite point whose stored position is not usable (behind a camera, or on
  top of one) gets the same rule with the position it holds: the fit runs from
  it, the point moves to a usable result (`refitted`), and otherwise the track
  is treated as a bearing and demoted as above. When that bearing is behind a
  camera too, no representation describes every sighting and the point is
  left at its stored position, counted in `left_unusable`, for a caller that
  can prune sightings or drop points.
- **A track whose cameras collapsed onto one centre.** Its centres coincide to
  round-off, so the test gives it no depth leverage (see "Fitting") and a
  bearing verdict, and the point is demoted to its bearing; the stored
  position, on the cameras, leaves no patch extent to convert. That holds
  whether its rays agree (a real pan, `Λ ≈ 0`) or diverge, as a solver's
  collapse that kept the rotations leaves them. Diverging rays fit no bearing,
  but the point fit can only run onto the shared centre, which the minimum
  depth rejects, so a finite verdict would end in the same demotion through
  the refit.
- **What a change carries with it.** A changed point's normal is zeroed and its
  normal confidence set to 0 (the writer fills a finite point's zero normal
  from its viewing directions, and a `w = 0` row is zero by the format), its
  error is recomputed against its observed pixels in the new representation,
  and its constraint is released. Its patch frame crosses the boundary at the
  distance from the camera-cloud centroid: demotion divides the half-vectors by
  the stored point's distance and projects them onto the bearing's tangent
  plane, promotion multiplies them by the placed point's distance, and a frame
  solved at a position that was no depth (a refit, or a demotion from a
  position on a camera) is cleared.
- **The minimum depth** is `DEFAULT_MIN_DEPTH_FRACTION` of the median distance
  from a finite point to the centre of a camera observing it (the camera
  extents when no finite point is observed). On `tk117`, the seoul bull ground
  truth and `inf2` the stored finite point nearest a camera sits at 13%, 42%
  and 39% of that median, so 1% is an order of magnitude below any stored
  point and well above the near-zero depths of the fit's degenerate results.
- **No noise level, no decision.** Without `sigma_px` and with no observation
  of a finite point, nothing is measured and the reconstruction comes back
  unchanged, with `sigma_px: None` in the summary. A `sigma_px` that is not
  finite and positive is `InvalidNoiseLevel`, and pixels that cannot be read
  (a `sift_files` file without its `.sift` files) are `Reconstruction`.

On `tk117` it promotes points 298,
294 and 295 to 511, 1,077 and 1,160 units from their observing cameras
(errors 0.16, 0.19 and 0.22 px, from 0.81, 0.48 and 0.38 as bearings) and
demotes nothing; on the seoul bull ground truth it changes nothing; on `inf2`
it promotes 34 discovered points, at 94 to 400 units (the capture's camera
extent is 11), and demotes nothing. `sfm analyze --depth-reliability` on each
result lists no disagreement.

Callers: `sfm xform --classify-points-at-infinity [<sigma_px>]`
([xform-command.md](../../cli/reconstruction/xform/xform-command.md)), the
COLMAP import (`colmap/io.py`, after a solve or `from-colmap`), and the
pycolmap bundle adjustment (`xform/_bundle_adjust.py`), which materialises
the points at infinity for the solve and runs it afterwards at the adjusted
reconstruction's own noise level.

**Discovery.** `SfmrReconstruction::find_points_at_infinity` in
[analysis/infinity/discover.rs](../../../crates/sfmtool-core/src/analysis/infinity/discover.rs)
assembles candidate tracks from untracked `.sift` keypoints by clustering their
directions, builds each candidate's rays with `observation_ray` at `sigma_px`
(the measured noise level when `None`), and decides them with
`decide_candidate_tracks`: `bearing_score_batch`, then `is_finite` at
`DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD`. A bearing verdict whose bearing is
in front of every observing camera is appended as `w = 0` at the score's
closed-form bearing, with its error measured against its keypoints; a finite
verdict, a bearing behind a camera and a candidate with too few usable rays
are dropped and counted in the summary, `InfinityDiscovery`. The interface,
the counts and the measurements behind each choice are in
[find-points-at-infinity.md](../../cli/reconstruction/xform/find-points-at-infinity.md)
(§ "CLI surface" and § "Decisions"). Three choices bear on this spec:

- **No third state.** A track whose cameras have a baseline too short to
  resolve a point at the camera extents gets `Λ ≈ 0` and is appended as a
  bearing, as reclassification stores such a point; the summary counts them
  as `short_baseline`. On a 48-image kerry_park `sift_files` solve the z rule
  dropped 174 such candidates at `0.1,200,2`, and the test appends 172 of them.
- **Bearings only.** A finite verdict is not appended. The appended bearings
  do not enter the noise measure, so a reclassification pass after discovery
  measures the same `σ`, scores the same rays and changes none of them;
  appending finite verdicts at their fitted points raised the measured `σ` on
  that solve from 0.231 to 0.318 px and set reclassification against 109
  points.
- **The minimum depth and the consumer rule of "Fitting"** do not arise,
  since discovery places no finite point.

On the seoul bull and kerry_park solves `sfm analyze
--depth-reliability` lists no disagreement on any discovery output.

**The bench.** Every bench step that triangulates -- the track-stage fit and
the cluster-to-track upgrade -- ends at `bench/classify.rs::classify_track_rays`
([`../bench/editable-track.md`](../bench/editable-track.md) § "Finite points and
bearings" has the interface). It builds one track's rays from the sightings it
holds, with `observed_ray` at the base reconstruction's measured noise level
(measured once per base and cached on the `EditedReconstruction`), decides with
`bearing_score` and `is_finite` at `DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD`,
and places a finite verdict with `fit_usable_point`, the plain least-squares
point fit started from the track's held position, under the reconstruction's
`min_point_depth`: the consumer rule reclassification places its points by,
shared code rather than a copy
([analysis/point_or_bearing.rs](../../../crates/sfmtool-core/src/analysis/point_or_bearing.rs)).
Two things differ, because a fit has to write something and has just moved the
sightings:

- **A finite verdict is placed by the fit**, warm-started from the held place,
  where reclassification keeps a usable stored position as it is; the bench
  keeps the held place only when the fit gives no usable point and the held
  place is usable (`held_point_kept`), which is where the two would otherwise
  part.
- **A bearing behind a camera** is placed by the point fit with no bar on its
  `Λ` (`bearing_behind_camera`), where reclassification keeps the stored
  point, and a track that neither describes keeps the coordinate it held, or
  takes the bearing when it held none (`left_unusable`).

A reconstruction with no observation of a finite point has no level, and a
fit there needs `sigma_px` given (the viewer's MCP `fit_bench_track` and
`set_bench_track_stage` take it). On the stored sightings of the
seoul bull ground truth (280 scored points), the Kerry Park solve `new05_ba`
(3,713, 2,852 of them discovered bearings) and its pre-discovery `base_ba`
(886), the bench's verdict agrees with `classify_points_at_infinity` on every
point, clean and with keypoint noise of 0.5 and 2 px or pose noise of 0.05° and
0.2° injected (`σ` measured 0.47 to 3.4 px); the finite count falls as `σ`
grows (`base_ba`: 886 at 0.20 px, 820 at 0.54, 397 at 2.0) and no step refuses
or fails.

**Bundle adjustment.** With `FreePointPolicy::cross`, the staged bundle
adjustment solves every free point in inverse depth about the centroid of its
observing cameras, `(u, ρ)` with `ρ ≥ 0` as in the point fit above, so a point
moves between near and infinity within a round, and decides once, at the end of
the solve, how it is stored: `observed_ray` through each observing image's
camera as the solve ended, `bearing_score` and `is_finite` at
`DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD`, a bearing verdict stored as the
score's bearing, a finite one at the position the solve placed (or at
`fit_point_and_bearing`'s point where the solve left it at `ρ = 0`). Its noise
level is not a stored reconstruction's: it is this section's estimator read
over the observations of finite points the final round solved on, at the state
the solve ended at. The interface, the choices, and the measurements against the
rejected alternative of re-deciding a point between rounds while it is carried
as a position or a direction, are in
[bundle-adjustment.md](../geometry/bundle-adjustment.md) § "Free points:
inverse depth and the storage decision". On the four inputs measured there, a
converged solve's stored representation agrees with the test at the level the
decision read on every point, and with the test at the result's own measured
noise, which is what `sfm analyze --depth-reliability` lists, on all but 0 to
5 points.

**Reports.** Per-point depth reliability appears in `sfm inspect --verbose` and in
`sfm analyze --depth-reliability`, both through the PyO3 surface below. Each
prints `inverse_depth_z` and, beside it, the verdicts of "Point or bearing"
from `point_or_bearing_scores` at the measured noise, and counts the points
whose verdict disagrees with how they are stored; `analyze` fits and lists those
points and takes `--sigma-px` and `--depth-likelihood-ratio-threshold`
([analyze-command.md](../../cli/reconstruction/analyze-command.md) § "Depth
Reliability", [inspect-command.md](../../cli/reconstruction/inspect-command.md)).
Neither changes the file. The disagreements they count are the points
reclassification would change, less those it declines to change (a bearing
behind a camera, a fit with no usable point), and `analyze` prints what
reclassification would do at the same noise level. The `inverse_depth_z`
column is a diagnostic: nothing decides on it.

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
- **Every consumer decides on the likelihood-ratio test**, not on
  `inverse_depth_z` or the condition number. Reclassification, discovery and
  the bench share the test, the measured noise level and the consumer rule
  that places a finite verdict, so one set of sightings earns one
  representation whichever step reads it; bundle adjustment's storage
  decision reads the same test and the same estimator of the noise, over its
  final round. `inverse_depth_z` is the Wald form
  of the same question, linearised at the fitted depth, which is the worst
  place for a far point; it stays as a diagnostic, beside the condition number
  (which grows with K).
- **No third state.** The `.sfmr` model is binary (`w=1` finite / `w=0` at
  infinity), and so is the test: a track the cameras cannot resolve gets
  `Λ ≈ 0` and a bearing verdict. `resolvable_distance` against the camera
  extents is a diagnostic of that geometry (`short_baseline`), not a gate. See
  "Scene-relative resolvability".
- **Reclassification is non-destructive and moves points both ways.**
  `classify_points_at_infinity` never adds or removes a point or a sighting.
  It demotes a finite point with a bearing verdict and promotes a point at
  infinity with a finite verdict, and declines either change when the result
  would describe a sighting wrongly (a bearing behind a camera) or place a
  point where none can be (behind or on top of a camera). See "Consumers".
- **Ray source is the caller's concern.** The GUI's diagnostics and
  `triangulation_diagnostics` supply camera→stored-point rays (cheap, no
  `.sift`). Reclassification, discovery and the point-or-bearing reports
  un-project each observation's pixel (`track_rays`, `observation_ray`), so
  they need the inline keypoints or the `.sift` files. The core functions are
  agnostic.
- **Do not persist diagnostics to `.sfmr`.** Derivable from geometry; storing
  per-point would bloat the format and go stale.

## Open questions

- Whether to compute the precise per-track perpendicular baseline `B⊥` for
  the short-baseline diagnostic, or accept the scalar camera-extent upper bound.
- **The threshold.** `25` puts `tk117`'s points 298, 294 and 295 finite and
  keeps 269 (`Λ` 14) and 270 (`Λ` 12) as bearings; `10.8` makes those two finite
  as well, at 2,800 and 4,800 units. Both are defensible on `tk117`; a larger
  capture with a known far field (KerryPark360) should decide it.
- **Per-camera σ.** One level per reconstruction, or one per camera when the
  cameras differ (the two Kerry Park lenses are close, 0.2155 and 0.2157 px;
  other rigs may not be).
- **Finite candidates in discovery.** Discovery appends only bearings (see
  "Consumers"). On the seoul bull `sift_files` solve the finite candidates it
  drops are sound, and on kerry_park they are marginal and raise `σ`; whether a
  quality gate, a bound on the placed point's reprojection error against `σ`
  say, could keep the consistent ones without moving the noise level is open
  ([find-points-at-infinity.md](../../cli/reconstruction/xform/find-points-at-infinity.md)
  § "Decisions" has the measurements).
- **A robust decision.** The decision is plain least squares. Whether a track
  whose score clears the threshold should also have to clear it under the
  soft-L1 point fit, so one bad sighting cannot make it finite, or whether
  outlier trimming upstream is enough.
- **Naming.** `bearing_score`, `BearingScore`, `depth_score`,
  `fit_point_and_bearing`, `PointBearingFit`, `depth_likelihood_ratio` and
  `is_finite` are the names the primitives carry; whether they stand, and a
  glossary row for them, is open.
- Weighted vs unweighted midpoint as the default (unweighted matches current
  behavior; inverse-depth² is closer to reprojection error).

## Reuse map

| Need | Existing piece |
|---|---|
| pixel → world ray (all models, fisheye) | `CameraIntrinsics::pixel_to_ray[_batch]` |
| camera center | `SfmrImage::camera_center` (`= −Rᵀt`) |
| max pairwise angle (pre-filter only) | `geometry/viewing_angle.rs::max_viewing_angle` |
| existing 2-view algebraic triangulation | `features/feature_match/geometric_filter.rs::triangulate_point_dlt` |
| per-track observation slices (CSR) | `observation_offsets`, indexed directly |

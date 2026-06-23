# Batch Triangulation API with Observability Diagnostics

**Status:** Implemented (all four migration phases below landed). The
classifier thresholds (`inverse_depth_z` cutoff, condition-number pre-filter)
remain provisional pending larger-dataset calibration — see Open questions.

Consolidate the scattered triangulation code into one batch API in
`sfmtool-core` (`reconstruction/triangulation.rs`) that returns each track's solved point
**plus the observability diagnostics the solve already computes** (the normal
matrix's spectrum, and an optional noise-calibrated depth uncertainty). The
immediate driver was a classification bug: `find_points_at_infinity` and
`classify_points_at_infinity` decided finite-vs-infinity from the *maximum
pairwise viewing angle*, an extreme order statistic that is inflated by
keypoint noise and **grows with view count**, so genuine points at infinity
with many observations were misclassified as finite. The conditioning of the
triangulation answers the finite-vs-infinity question directly; previously it
was computed and thrown away.

Related: [finding points at infinity in an existing solve](../cli/xform-find-points-at-infinity.md),
[the v2 points-at-infinity format model](../formats/sfmr-file-format.md) (§7), and
[rendering points at infinity in the GUI](../gui/gui-point-cloud-rendering.md).

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
- The midpoint solve in `classify_track` (`analysis/infinity/discover.rs:313-352`) already
  builds the normal matrix `A = Σ(I − dᵢdᵢᵀ)` and computes `det(A)`, but only
  uses `det` against an ultra-loose gate (`det < 1e-9·‖A‖³`,
  `analysis/infinity/discover.rs:330`) that fires only for *exactly* singular `A`. The
  eigenvalues of `A` — a 1000× separation between genuine and degenerate tracks
  (population medians: condition number 82 vs 89,599; relative depth uncertainty
  1.6% vs 33%) — are discarded.

So the fix and the refactor are the same change: make the triangulation a
reusable batch operation that returns its conditioning, and have the
classifiers decide on that.

## Prior state (before this change)

Before this refactor there was no `reconstruction/triangulation.rs` and no batch API.
Triangulation lived in three disconnected places, none sharing a result type,
none batched, none exposed to Python:

| Site | Method | Returns | Diagnostics | Used by |
|---|---|---|---|---|
| `geometric_filter.rs:252` `triangulate_point_dlt` | 2-view DLT (SVD) | `Option<[f64;3]>` (drops `w≈0`) | none | two-view in-front-of-camera check during matching (`:375`) |
| `analysis/infinity/discover.rs:275` `classify_track` (inline) | N-view midpoint | `(Point3, w)` | builds `A`, `det` (gated loosely, then discarded) | `find_points_at_infinity` |
| GUI: `point_track_detail.rs:698`, `image_detail.rs:1022` | **no solve** — max pairwise angle to the *stored* point | `f32` degrees | — | "Max Track Angle" overlay (`state.rs:31`), point/feature detail |

The GUI never triangulates; it reuses the same `max_viewing_angle` statistic (a
third copy of it) and presents it as "High = well-triangulated, low =
unreliable" (`state.rs:29-31`). For genuinely finite points with real parallax
this is an accurate quality signal — a wide max angle does mean a
well-conditioned depth — so the overlay is fine for the bulk of the cloud. It
misleads only in the distant / near-infinity regime, where the parallax is
dominated by localization noise and the *max* statistic inflates with view
count: a far point seen by many cameras reads as "well-triangulated" while its
depth is in fact unconstrained. The proposed condition-number / inverse-depth
diagnostic agrees with the angle on finite points and additionally gets that
regime right, so it belongs alongside the angle overlay as a complementary view
rather than a replacement for it.
`classify_points_at_infinity` (`analysis/infinity/convert.rs:50`) is a fourth consumer of
`max_viewing_angle` (`analysis/infinity/convert.rs:78`).

## Target Rust API

New module `crates/sfmtool-core/src/reconstruction/triangulation.rs`. Tracks are flattened
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

The GUI (`sfm-explorer`, Rust) consumes the core functions directly; the binding
is for the CLI/inspect/analyze/notebook paths.

## Consumers & migration

Phased so the API lands before any behavior changes:

1. **Phase 1 — core. (done)** Added `reconstruction/triangulation.rs` (struct +
   `triangulate_batch` + `depth_uncertainty_batch`) with unit tests. The
   midpoint solve was extracted out of `classify_track` into it. No behavior
   change.
2. **Phase 2 — the fix. (done)** `analysis/infinity/discover.rs::classify_track` and
   `analysis/infinity/convert.rs::classify_points_at_infinity` now share `classify_rays_at_infinity`,
   which decides finite-vs-∞ on `inverse_depth_z` (with `condition_number` as a
   cheap pre-filter) instead of `alpha_max·f_max` and the loose `det` gate. The
   provisional thresholds (`DEFAULT_INVERSE_DEPTH_Z_CUTOFF = 4.0`,
   `CONDITION_NUMBER_PREFILTER = 1e4`) live in `analysis/infinity/convert.rs`; final calibration
   is deferred to larger-dataset evaluation (see Open questions). The noise
   floor is exposed on the `--find-points-at-infinity` CLI as a 4th component
   (`eps_deg[,desc_thresh[,min_views[,noise_floor_px]]]`).
3. **Phase 3 — bindings + reports. (done)** PyO3 `triangulate_batch` (free fn,
   dict of arrays) and `recon.triangulation_diagnostics(noise_px)`; per-point
   depth reliability surfaced in `sfm inspect --verbose` and a new
   `sfm analyze --depth-reliability` mode.
4. **Phase 4 — GUI. (done)** The "Max Track Angle" overlay
   (`compute_max_pairwise_angle` / `compute_max_track_angle_deg`,
   `colormap.rs`) is left in place — it is an accurate quality signal for finite
   points. Two new overlay modes back onto the new diagnostics: "Depth
   Reliability" driven by `inverse_depth_z` (low = near-infinity / unconstrained
   depth), and "Condition Number" (log scale). They compute the per-point
   diagnostics via `triangulate_batch` / `depth_uncertainty_batch`
   (`point_track_detail::compute_point_diagnostics`), and the same numbers
   appear in the point-track header and the image-detail tooltip next to the max
   angle. Additive — no GUI code removed.

**Left in place:** `geometry/viewing_angle.rs::max_viewing_angle` stays — it still backs
`xform --remove-narrow-tracks` (`compute_narrow_track_mask`) as a fast
pre-filter. It simply stops being the *classification* signal.

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
  (`noise = max(reproj_error, floor)`); discovered points currently carry
  `error = 0`.
- Weighted vs unweighted midpoint as the default (unweighted matches current
  behavior; inverse-depth² is closer to reprojection error).

## Reuse map

| Need | Existing piece |
|---|---|
| pixel → world ray (all models, fisheye) | `CameraIntrinsics::pixel_to_ray[_batch]` |
| camera center | `SfmrImage::camera_center` (`= −Rᵀt`) |
| max pairwise angle (pre-filter only) | `geometry/viewing_angle.rs::max_viewing_angle` |
| existing 2-view algebraic triangulation | `geometric_filter.rs::triangulate_point_dlt` |
| bearing-mean fallback for `w = 0` | `analysis/infinity/convert.rs` (`normalise(Σ rᵢ)`) |
| per-track observation slices (CSR) | `observation_offsets` / `observations_for_point` |

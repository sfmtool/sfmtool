# Batch Triangulation API with Observability Diagnostics

**Status:** Draft / refactoring plan. Consolidate the scattered triangulation
code into one batch API in `sfmtool-core` that returns each track's solved
point **plus the observability diagnostics the solve already computes** (the
normal matrix's spectrum, and an optional noise-calibrated depth uncertainty).
The immediate driver is a classification bug: `find_points_at_infinity` and
`classify_points_at_infinity` decide finite-vs-infinity from the *maximum
pairwise viewing angle*, an extreme order statistic that is inflated by
keypoint noise and **grows with view count**, so genuine points at infinity
with many observations are misclassified as finite. The conditioning of the
triangulation answers the finite-vs-infinity question directly, but today it is
computed and thrown away.

Related: [finding points at infinity in an existing solve](xform-find-points-at-infinity.md),
[the v2 points-at-infinity format model](sfmr-v2-points-at-infinity.md), and
[rendering points at infinity in the GUI](gui-points-at-infinity.md).

## The Problem

A point at infinity is a track whose observation rays are parallel to within
measurement noise: its depth is unobservable. The current finite-vs-infinity
test is `parallax_px = alpha_max · f_max` against a 1 px floor, where
`alpha_max = max_viewing_angle(rays)` is the single widest angle among all
`K(K−1)/2` ray pairs (`viewing_angle.rs:28`). That statistic is wrong for this
job:

- Under pure 1 px localization noise (no real parallax) the **expected** max
  pairwise angle already exceeds the floor once a track has a handful of views,
  and it keeps rising with `K` — so more observations make a true infinity point
  *more* likely to be called finite. (Measured: a 27-view distant track scored
  2.27 px; a direction-only model reprojected it at 0.61 px RMS, and adding a
  finite depth improved RMS by only 0.025 px — the depth explains nothing.)
- The midpoint solve in `classify_track` (`find_infinity.rs:313-352`) already
  builds the normal matrix `A = Σ(I − dᵢdᵢᵀ)` and computes `det(A)`, but only
  uses `det` against an ultra-loose gate (`det < 1e-9·‖A‖³`,
  `find_infinity.rs:330`) that fires only for *exactly* singular `A`. The
  eigenvalues of `A` — a 1000× separation between genuine and degenerate tracks
  (population medians: condition number 82 vs 89,599; relative depth uncertainty
  1.6% vs 33%) — are discarded.

So the fix and the refactor are the same change: make the triangulation a
reusable batch operation that returns its conditioning, and have the
classifiers decide on that.

## Current state (what exists today)

There is no `triangulation.rs` and no batch API. Triangulation lives in three
disconnected places, none sharing a result type, none batched, none exposed to
Python:

| Site | Method | Returns | Diagnostics | Used by |
|---|---|---|---|---|
| `geometric_filter.rs:252` `triangulate_point_dlt` | 2-view DLT (SVD) | `Option<[f64;3]>` (drops `w≈0`) | none | two-view in-front-of-camera check during matching (`:375`) |
| `find_infinity.rs:275` `classify_track` (inline) | N-view midpoint | `(Point3, w)` | builds `A`, `det` (gated loosely, then discarded) | `find_points_at_infinity` |
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
`classify_points_at_infinity` (`infinity.rs:50`) is a fourth consumer of
`max_viewing_angle` (`infinity.rs:78`).

## Target Rust API

New module `crates/sfmtool-core/src/triangulation.rs`. Tracks are flattened
CSR-style (the same shape as the reconstruction's `observation_offsets`): track
`t` owns `dirs[offsets[t]..offsets[t+1]]` and the matching `centers`.

```rust
// crates/sfmtool-core/src/triangulation.rs

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
    /// discovered "finite" 3.) This is the recommended finite-vs-∞ test.
    pub inverse_depth_z: f64,
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

1. **Phase 1 — core.** Add `triangulation.rs` (struct + `triangulate_batch` +
   `depth_uncertainty_batch`) with unit tests. Extract the midpoint solve out of
   `classify_track` into it. No behavior change.
2. **Phase 2 — the fix.** Re-point `find_infinity.rs::classify_track` and
   `infinity.rs::classify_points_at_infinity` at the new API; decide
   finite-vs-∞ on `inverse_depth_z` (with `condition_number` as a cheap
   pre-filter) instead of `alpha_max·f_max` and the loose `det` gate.
   Ship with a provisional cutoff (~4) between the KerryPark360 populations
   (genuine z≈62 vs discovered
   z≈3); re-run and report how the counts shift; final threshold calibration is
   deferred to larger-dataset evaluation (see Open questions). Expose the noise
   floor on the `--find-points-at-infinity` CLI (currently hardcoded to 1.0).
3. **Phase 3 — bindings + reports.** PyO3 `triangulate_batch` and
   `recon.triangulation_diagnostics`; surface per-point depth reliability in
   `sfm inspect --verbose` / `sfm analyze`.
4. **Phase 4 — GUI.** Leave the "Max Track Angle" overlay
   (`compute_max_pairwise_angle` / `compute_max_track_angle_deg`, `state.rs:31`,
   `colormap.rs:82`) in place — it is an accurate quality signal for finite
   points. Add one or two new color modes backed by the new diagnostics: a
   "Depth Reliability" mode driven by `inverse_depth_z` (low = near-infinity /
   unconstrained depth), and optionally a "Condition Number" mode. These compute
   the per-point diagnostics directly from `triangulate_batch` /
   `depth_uncertainty_batch`, and surface the same numbers in the point-track and
   image detail panels next to the existing max angle, so the views can be
   compared. Additive — no GUI code is removed.

**Left in place:** `viewing_angle.rs::max_viewing_angle` stays — it still backs
`xform --remove-narrow-tracks` (`compute_narrow_track_mask`) as a fast
pre-filter. It simply stops being the *classification* signal.

## Decisions

- **Always return geometric diagnostics; opt-in statistical.** Eigenvalues +
  condition number are byproducts of the solve, so returning them by default
  adds no work and keeps a depth's reliability attached to the point. Depth σ /
  inverse-depth z need a noise model, so they are a second batch call.
- **Decision variable is `inverse_depth_z` (scale-free), not condition number**
  (grows with K). Condition number is the cheap geometric flag.
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
| max pairwise angle (pre-filter only) | `viewing_angle.rs::max_viewing_angle` |
| existing 2-view algebraic triangulation | `geometric_filter.rs::triangulate_point_dlt` |
| bearing-mean fallback for `w = 0` | `infinity.rs` (`normalise(Σ rᵢ)`) |
| per-track observation slices (CSR) | `observation_offsets` / `observations_for_point` |

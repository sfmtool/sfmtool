# Epipolar Curves for Non-Perspective Cameras

**Status:** Implemented — `crates/sfmtool-core/src/camera/epipolar.rs`
(`plot_epipolar_curve`, `plot_epipolar_curves_batch`, `EpipolarCurveOptions`),
exposed as `sfmtool._sfmtool.epipolar_curves` (`py_epipolar.rs`) and consumed
by `sfm epipolar` via `visualization/_epipolar_display.py`. As shipped, the
anchor depth is a **per-feature argument** (scalar / array), not an options
field — the API blocks below reflect that.

## The Problem

The epipolar constraint `p2ᵀ F p1 = 0` only holds when `p1`, `p2` are pixel
coordinates of a **pinhole** camera (equivalently, normalized image coordinates
of any camera). For a feature in image 1, the set of possible matches in image 2
is then a straight line — the epipolar line.

`sfm epipolar` builds `F` from the pinhole intrinsic matrix `K`
(`CameraIntrinsics::intrinsic_matrix`) and draws `F p1` as a straight segment.
That is correct for `SIMPLE_PINHOLE` / `PINHOLE`, roughly correct for mild
radial distortion (`SIMPLE_RADIAL`, `RADIAL`, `OPENCV`), and **wrong** for
fisheye / wide-FOV models (`OPENCV_FISHEYE`, `RAD_TAN_THIN_PRISM_FISHEYE`,
`FOV`, …), where the locus of possible matches curves in pixel space. There is
no fundamental matrix in fisheye pixel coordinates; the bilinear epipolar
relation lives in normalized bearing space, and the bearing→pixel map is
nonlinear.

So the epipolar geometry must be produced by sampling the constraint in a space
where it is linear (the back-projected ray / normalized bearings) and
reprojecting through the full destination camera model. The output is a polyline
that approximates the true curve.

## Where this lives

This is a `sfmtool-core` concern: it needs `CameraIntrinsics`'s full forward and
inverse projection (`pixel_to_ray` / `ray_to_pixel`, which already invert the
distortion model including fisheye), the image rectangle (`width` / `height` on
`CameraIntrinsics`), `RigidTransform` for the poses, and `is_fisheye()` for the
rectification guard. New API goes in `crates/sfmtool-core/src/camera/epipolar.rs`; a
thin PyO3 wrapper exposes it to the `sfm epipolar` visualization
(`src/sfmtool/visualization/_epipolar_display.py`), which just hands the
returned vertices to `cv2.polylines`.

The image-rectangle clip lives in Rust, not Python: endpoint finding needs to
test "is the projected pixel inside `[0, width) × [0, height)`?" as part of its
in-image predicate, so the rectangle has to be known where the curve is
generated. The Python display layer therefore does no clipping of its own.

`compute_fundamental_matrix` / `compute_epipole*` stay as they are — they are
still used by stereo rectification and sweep matching. The curve API does **not**
route through `F`; it goes ray → world → reproject, which is what makes it
model-agnostic. The in-frame-epipole half-line special case in the current
display code disappears (a polyline through the epipole needs no special
handling).

## Rust API

```rust
// crates/sfmtool-core/src/camera/epipolar.rs

/// Controls how an epipolar curve is sampled into a polyline.
pub struct EpipolarCurveOptions {
    /// Maximum allowed perpendicular distance (pixels) from a segment's
    /// midpoint to its chord before the segment is further subdivided.
    /// Default: 0.5.
    pub curvature_tolerance: f64,
    /// Hard cap on the number of vertices in a single polyline. Stops
    /// subdivision once reached, even if the tolerance is not met. Default: 256.
    pub max_vertices: usize,
}

impl Default for EpipolarCurveOptions {
    fn default() -> Self {
        Self { curvature_tolerance: 0.5, max_vertices: 256 }
    }
}

/// Plot the epipolar curve in camera 2's image for pixel `p1` in camera 1.
///
/// Back-projects `p1` through `cam1`'s full model, brackets the depth interval
/// over which the world ray's reprojection stays inside camera 2's image, and
/// adaptively subdivides that interval until every chord lies within
/// `curvature_tolerance` pixels of the true curve. Returns an empty polyline
/// when the baseline is degenerate or no in-image interval is found.
///
/// The returned polyline is fully inside `[0, cam2.width) × [0, cam2.height)`;
/// the caller can draw it directly with `cv2.polylines` (no further clipping).
///
/// `anchor_depth` is the seed depth in camera 1 used to bracket the in-image
/// interval — typically the reconstructed depth of the observed track when
/// triangulated, otherwise the baseline length `‖C2 − C1‖`. The algorithm
/// walks outward in log-depth from this seed, so an order-of-magnitude
/// estimate is fine.
pub fn plot_epipolar_curve(
    p1: [f64; 2],
    cam1: &CameraIntrinsics,
    pose1: &RigidTransform,          // cam1_from_world
    cam2: &CameraIntrinsics,
    pose2: &RigidTransform,          // cam2_from_world
    anchor_depth: f64,
    opts: &EpipolarCurveOptions,
) -> Vec<[f64; 2]>;

/// Batch form: one polyline per input pixel (with a per-feature anchor
/// depth), parallelized over points. `anchor_depths.len()` must equal
/// `points1.len()`.
pub fn plot_epipolar_curves_batch(
    points1: &[[f64; 2]],
    anchor_depths: &[f64],
    cam1: &CameraIntrinsics,
    pose1: &RigidTransform,
    cam2: &CameraIntrinsics,
    pose2: &RigidTransform,
    opts: &EpipolarCurveOptions,
) -> Vec<Vec<[f64; 2]>>;
```

### Algorithm

The world ray of `p1` is monotonically parametrized by depth `λ > 0`:
`X(λ) = C1 + λ r1`, where `C1 = pose1.inverse_translation_origin()` and
`r1 = pose1.rotation.inverse().rotate_vector(cam1.pixel_to_ray(p1))`. Define an
in-image predicate

```
in_image(λ) := let Xc = pose2.transform_point(X(λ));
               Xc.z < 0        // in front of the camera (canonical −Z-forward
                               // frame; see sfmr-file-format.md conventions)
               && let Some((u, v)) = cam2.ray_to_pixel(Xc)
               && 0.0 <= u && u < cam2.width as f64
               && 0.0 <= v && v < cam2.height as f64
```

The algorithm has two phases — bracket the in-image interval, then adaptively
subdivide it.

#### Phase 1: endpoint bracketing

The goal is to find `(λ_in, λ_out)` with `λ_in < λ_out`, both in-image, such
that `in_image` is false just outside the interval. Operate in log-depth (so
"halving / doubling" reads as ±1 step) and treat `log_anchor = ln(anchor_depth)`
as the seed.

1. **Seed in-image search.** Probe `in_image` at `log_anchor`, then at
   `log_anchor ± k·LOG_STEP` for `k = 1 .. BRACKET_MAX_STEPS` in alternation
   until *some* probe lands in-image, or all are exhausted (→ return empty).
   `LOG_STEP = ln(2)` (one octave per step); `BRACKET_MAX_STEPS = 24` (≈16
   million-fold range, enough for any plausible reconstruction).
2. **Bisect each side.** From the in-image seed, walk down in steps of
   `LOG_STEP` until a probe falls out of image, giving a bracket
   `[log_λ_lo, log_λ_hi]` where `in_image(log_λ_hi)` is true and
   `in_image(log_λ_lo)` is false. Bisect until
   `|log_λ_hi − log_λ_lo| < BRACKET_LOG_TOL` (default `1e-3`, i.e. ~0.1% in λ).
   Take the in-image endpoint as `log_λ_in`.
3. **Walk up.** Same procedure expanding upward to find `log_λ_out`. If
   `BRACKET_MAX_STEPS` are exhausted without finding an out-of-image probe,
   accept `log_λ_out = log_anchor + BRACKET_MAX_STEPS · LOG_STEP` (the
   vanishing-point endpoint is effectively at infinity; the cap is fine in
   practice because curve geometry flattens rapidly there).

The bisection's tolerance is in log-depth, not pixel position. That's
intentional: it's cheap, well-conditioned even when the projection grows
infinitely sensitive near the camera-2 image-plane horizon, and the final
endpoint is then refined by the adaptive subdivision below to whatever pixel
budget the caller wants.

#### Phase 2: adaptive subdivision

Build the polyline by splitting the in-image interval at midpoints until every
chord lies within `curvature_tolerance` pixels of the curve. Two important
design choices:

- **Parameter `t = 1/λ`, not `λ` or `log(λ)`.** For perspective projection of
  a 3D line, normalized image coordinates `(x/z, y/z)` are exactly affine in
  `1/λ` — i.e. equal steps in `t` correspond to equal pixel-space steps along
  the projective ray. Arithmetic midpoints in `t` thus give balanced
  pixel-coverage subdivision. Log-depth midpoints would collapse toward the
  larger-λ endpoint when the bracket is asymmetric (common: the vanishing-
  point side is often at the `BRACKET_MAX_STEPS` cap, i.e. effectively
  infinite λ), making the chord-deviation test trivially accept a curve that
  is in fact curvy.
- **Worst-first order, not depth-first.** Maintain the polyline as a sorted
  list of vertices with a parallel list of gap candidates (each gap caches
  its midpoint projection and chord-deviation). Each iteration picks the gap
  with the maximum cached deviation that still exceeds tolerance, splits it,
  and recomputes the two new gap candidates. This gives the vertex budget to
  the worst regions first, so a tight `max_vertices` cap produces a balanced
  polyline rather than a high-resolution left half and a coarse right half.

Algorithm:

1. Convert the Phase-1 bracket to `t`: `t_in = 1/λ_in`, `t_out = 1/λ_out`
   (note `t_in > t_out > 0`). Initialize vertices `[(t_in, p_in), (t_out, p_out)]`.
2. Evaluate the single initial gap: midpoint `t_m = (t_in + t_out) / 2`,
   reproject to get `p_m`, compute chord deviation.
3. Loop while total vertices `< max_vertices`:
   - Scan gaps for the one with maximum `dev > curvature_tolerance` and a
     projectable midpoint. If none qualifies, done.
   - Insert that midpoint into the vertex list and recompute the two new
     gap candidates flanking it.
4. Emit the final pixel sequence (drop the `t` parameter values).

This collapses to two vertices for pinhole / mild radial cameras (the chord
deviation is zero along a straight line) and only spends samples in
high-curvature regions of fisheye projections.

Midpoints where `π(t_m)` returns `None` (the predicate flipped to false
between two in-image endpoints — see the disconnected-interval note below)
mark the gap as final rather than splitting further; the polyline closes
across the gap with a straight chord through the unsampled region.

The reverse direction (curve in image 1 for a feature in image 2) is the same
call with `(cam1, pose1)` and `(cam2, pose2)` swapped.

### Why this is better everywhere

It drops the pinhole assumption entirely, so the same path is exact for pinhole,
radial, and fisheye models — the "standard vs. undistort vs. rectify" split in
the display code is no longer needed for *correctness*. `--undistort` /
`--rectify` remain purely as *display* options (warp to a rectified frame, draw
straight scanlines there); they are not a workaround.

### Degeneracies

- **Near-zero baseline** (`‖C2 − C1‖ ≈ 0`): the epipolar plane is ill-defined;
  return an empty polyline (caller skips drawing), matching today's near-zero `F`
  behavior.
- **Anchor depth at which the curve isn't visible**: the Phase-1 seed search
  probes ±`BRACKET_MAX_STEPS` octaves around `anchor_depth` looking for any
  in-image point. If none is found, return an empty polyline. With a
  track-depth or baseline-length anchor (see the caller-side seeding strategy
  below) the seed is almost always in-image for a real feature; this case
  mostly happens when a feature is mistakenly fed in for a pair with no
  geometric overlap.
- **Epipole inside the frame** (forward/backward motion, common in fisheye
  walk-throughs): the polyline simply passes through it — no special case.
- **Disconnected in-image intervals**: the in-image set is one connected
  interval in λ for the vast majority of cases. Two cases produce disconnected
  intervals: (a) the curve exits and re-enters the image rectangle along
  different edges, and (b) the world ray crosses behind camera 2 between two
  visible segments. The algorithm returns only the component containing the
  bracketing seed; the omitted component is documented as a known limitation.
  Phase-2 protects against the second component leaking in across an
  in-image segment by truncating subdivision when a midpoint projection
  fails the predicate (see Phase 2 step 2).

## Rectification and Fisheye

A single rectifying homography exists only for pinhole/radial cameras. For
fisheye inputs `--rectify` fails fast (the display layer checks the camera
model name) with a clear message rather than silently undistorting to a cropped
pinhole image. `--undistort` may still be offered for fisheye, with the caveat
that it discards the wide-FOV periphery; the default — curves on the original
images — is the recommended path for those cameras.

## PyO3 Binding

Exposed as `sfmtool._sfmtool.epipolar_curves`
(`crates/sfmtool-py/src/py_epipolar.rs`):

```python
epipolar_curves(
    points1: NDArray[N, 2],
    anchor_depths: NDArray[N],   # per-feature seed depths (required, positional)
    cam1: CameraIntrinsics, q1_wxyz: NDArray[4], t1: NDArray[3],
    cam2: CameraIntrinsics, q2_wxyz: NDArray[4], t2: NDArray[3],
    *, curvature_tolerance: float = 0.5,
    max_vertices: int = 256,
) -> list[NDArray[M, 2]]   # one polyline per input point; lengths vary
```

`_epipolar_display.py` calls this in place of `F @ p1` + `cv2.line`, then draws
each polyline directly with `cv2.polylines`. No image-rectangle clipping
happens on the Python side — Rust already constrains every returned vertex to
be inside the image.

### Caller-side seeding strategy

`_epipolar_display.py` picks `anchor_depth` per feature with a two-tier
fallback:

1. **Triangulated track**: if the feature observation in image 1 is linked to a
   3D point in the reconstruction, use that point's depth as seen from camera 1
   (`(R1 · X + t1).z`). This is guaranteed in-image for the feature's true
   match, so Phase-1's first probe lands in-image with zero seed search.
2. **Otherwise**: use the baseline length `‖C2 − C1‖`. For typical photogrammetry
   pairs, features sit at 10–100× the baseline, so this is within ~3–7 octaves
   of the truth — well inside Phase-1's ±24-octave seed-search range. One
   subtraction and one `norm`, computed once per image pair.

This replaces the previous scene-median computation
(`_median_scene_depth(recon, R1, t1)`), which iterated all 3D points per
image pair. The new path does no per-pair O(N_points) work, only per-feature
O(1) lookups against the track-index.

## Out of Scope

Visualization only (`sfm epipolar`). The polar-sweep and rectified-sweep
matchers (`feature_match/_polar_sweep.py`, `feature_match/_rectified_sweep.py`,
`sfmtool-core/src/camera/rectification.rs`) carry the same pinhole assumption; making
*matching* fisheye-aware (sweeping along the bearing-space epipolar line) is
separate, larger work.

## Parameters

| Name | Default | Notes |
|------|---------|-------|
| `anchor_depth` | observed track depth (if triangulated), else baseline length `‖C2 − C1‖` | Seed depth for Phase-1 bracketing |
| `curvature_tolerance` | `0.5` (pixels) | Max chord-to-midpoint deviation before a segment is further split |
| `max_vertices` | `256` | Hard cap per polyline — stops runaway subdivision |

Tuning knobs that are *not* exposed because their good values are independent
of the camera and scene:

| Constant | Value | Role |
|----------|-------|------|
| `LOG_STEP` | `ln(2)` | One octave per bracketing step |
| `BRACKET_MAX_STEPS` | `24` | ±24 octaves of seed search before giving up |
| `BRACKET_LOG_TOL` | `1e-3` | Bisection tolerance in log-depth |

`curvature_tolerance` could later be surfaced as a CLI option
(`--curve-tolerance`); start with the fixed default. `max_vertices` is a safety
cap, not a quality knob.

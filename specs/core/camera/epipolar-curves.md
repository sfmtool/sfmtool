# Epipolar Curves for Non-Perspective Cameras

Epipolar curve sampling produces, for a pixel in one image, the polyline in a
second image along which its match must lie. It traces that locus through both
cameras' full projection models, so match inspection remains geometrically
correct when fisheye and wide-FOV distortion bends the locus away from a line.

## Geometry

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
rectification guard. The API lives in
[`crates/sfmtool-core/src/camera/epipolar.rs`](../../../crates/sfmtool-core/src/camera/epipolar.rs); a
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
model-agnostic. There is no in-frame-epipole special case on this path: a polyline
that passes through the epipole needs no special handling, and the half-line form
survives only on the `--undistort` branch.

## Rust API

The curve sampling lives in
[epipolar.rs](../../../crates/sfmtool-core/src/camera/epipolar.rs)
(`plot_epipolar_curve`, `plot_epipolar_curves_batch`, `EpipolarCurveOptions`).
The poses passed to both functions are `cam_from_world` transforms.

```rust
pub struct EpipolarCurveOptions {
    pub curvature_tolerance: f64,
    pub max_vertices: usize,
}

pub fn plot_epipolar_curve(
    p1: [f64; 2],
    cam1: &CameraIntrinsics,
    pose1: &RigidTransform,
    cam2: &CameraIntrinsics,
    pose2: &RigidTransform,
    anchor_depth: f64,
    opts: &EpipolarCurveOptions,
) -> Vec<[f64; 2]>;

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

`plot_epipolar_curve` returns vertices inside `[0, cam2.width) × [0,
cam2.height)`. It returns no vertices for a degenerate baseline or when Phase 1
finds no in-image sample. It can return one vertex when the bracket collapses
within `BRACKET_LOG_TOL` or its second endpoint cannot be projected; callers
must therefore treat fewer than two vertices as non-drawable. Otherwise the
vertices form the sampled curve in order. The batch form runs the same operation
in parallel and preserves input order; `anchor_depths.len()` must equal
`points1.len()`.

`anchor_depth` is only a search seed, so an order-of-magnitude estimate is
enough. Rust normalizes it as `max(abs(anchor_depth), MIN_ANCHOR)` before taking
its logarithm: negative finite inputs use their magnitude, and zero or a tiny
magnitude uses `MIN_ANCHOR = 1e-12`.

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

The goal is to find `(λ_in, λ_out)` with `λ_in < λ_out` and both endpoints
in-image. When an out-of-image probe exists, it brackets the corresponding
endpoint; otherwise the endpoint is the farthest in-image probe allowed by the
search cap. Operate in log-depth (so "halving / doubling" reads as ±1 step) and
start at normalized `log_anchor`.

1. **Seed in-image search.** Probe `in_image` at `log_anchor`, then at
   `log_anchor ± k·LOG_STEP` for `k = 1 .. BRACKET_MAX_STEPS` in alternation
   until *some* probe lands in-image, or all are exhausted (→ return empty).
   `LOG_STEP = ln(2)` (one octave per step); `BRACKET_MAX_STEPS = 24` (≈16
   million-fold range, enough for any plausible reconstruction).
2. **Walk down.** From the in-image seed `log_seed`, walk down in steps of
   `LOG_STEP` until a probe falls out of image, giving a bracket
   `[log_λ_lo, log_λ_hi]` where `in_image(log_λ_hi)` is true and
   `in_image(log_λ_lo)` is false. Bisect until
   `|log_λ_hi − log_λ_lo| < BRACKET_LOG_TOL` (`1e-3`, i.e. ~0.1% in λ),
   then take the in-image side as `log_λ_in`. If no out-of-image probe is found,
   use the farthest in-image probe, `log_seed - BRACKET_MAX_STEPS · LOG_STEP`.
3. **Walk up.** Same procedure expanding upward to find `log_λ_out`. If
   `BRACKET_MAX_STEPS` are exhausted without finding an out-of-image probe,
   accept `log_λ_out = log_seed + BRACKET_MAX_STEPS · LOG_STEP` (the
   vanishing-point endpoint is effectively at infinity; the cap is fine in
   practice because curve geometry flattens rapidly there).

The bisection's tolerance is in log-depth, not pixel position. That's
intentional: it's cheap, well-conditioned even when the projection grows
infinitely sensitive near the camera-2 image-plane horizon. The separate
adaptive-subdivision tolerance controls the curve's pixel-space approximation.

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

This normally collapses to two vertices when the projected locus is straight
and only spends additional samples in high-curvature regions. The Phase-1
single-vertex exits happen before subdivision.

Midpoints where `π(t_m)` returns `None` (the predicate flipped to false
between two in-image endpoints — see the disconnected-interval note below)
mark the gap as final rather than splitting further; the polyline closes
across the gap with a straight chord through the unsampled region.

The reverse direction (curve in image 1 for a feature in image 2) is the same
call with `(cam1, pose1)` and `(cam2, pose2)` swapped.

### Why one path serves every model

The sampling drops the pinhole assumption entirely, so it is exact for pinhole,
radial and fisheye models alike, and the "standard vs. undistort vs. rectify" split in
the display code carries no correctness weight. `--undistort` / `--rectify` are purely
*display* options (warp to a rectified frame, draw straight scanlines there), not
workarounds for a model the curve cannot handle.

### Degeneracies

- **Near-zero baseline** (`‖C2 − C1‖ ≈ 0`): the epipolar plane is ill-defined;
  return an empty polyline, which the caller skips drawing.
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
  Phase 2 stops subdividing a gap when its midpoint projection fails the
  predicate (step 3); the retained endpoints are still joined by the straight
  chord described above.

## Rectification and Fisheye

A single rectifying homography exists only for pinhole/radial cameras. For
fisheye inputs `--rectify` fails fast (the display layer checks the camera
model name) with a clear message rather than silently undistorting to a cropped
pinhole image. `--undistort` may still be offered for fisheye, with the caveat
that it discards the wide-FOV periphery; the default — curves on the original
images — is the recommended path for those cameras.

## PyO3 Binding

Exposed as `sfmtool._sfmtool.epipolar_curves` by the
[PyO3 binding](../../../crates/sfmtool-py/src/analysis/epipolar.rs):

```python
epipolar_curves(
    points1: NDArray[N, 2],
    anchor_depths: NDArray[N],   # per-feature seed depths (required, positional)
    cam1: CameraIntrinsics, q1_wxyz: NDArray[4], t1: NDArray[3],
    cam2: CameraIntrinsics, q2_wxyz: NDArray[4], t2: NDArray[3],
    *, curvature_tolerance: float = 0.5,
    max_vertices: int = 256,
) -> list[NDArray[M, 2]]   # one result per input; M may be 0, 1, or more
```

[_epipolar_display.py](../../../src/sfmtool/visualization/_epipolar_display.py)
calls this in place of `F @ p1` + `cv2.line`. It ignores results shorter than
two vertices and passes all others to `cv2.polylines`; it does no
image-rectangle clipping because Rust constrains every returned vertex to the
image.

### Caller-side seeding strategy

`_epipolar_display.py` picks `anchor_depth` per feature as follows:

1. **Triangulated track**: if the feature observation in image 1 is linked to a
   3D point in the reconstruction, use its positive depth in canonical camera-1
   coordinates, `-(R1 · X + t1).z`. This is used only when the point is in
   front of camera 1.
2. **Otherwise**: use the baseline length `‖C2 − C1‖`. For typical photogrammetry
   pairs, features sit at 10–100× the baseline, so this is within ~3–7 octaves
   of the truth — well inside Phase-1's ±24-octave seed-search range. One
   subtraction and one `norm`, computed once per image pair.
3. **Degenerate baseline fallback**: if that length is below `1e-9`, use `1.0`
   as the seed. Rust independently detects the same degenerate camera pair with
   `MIN_BASELINE` and returns an empty result, so the substitute exists only to
   keep the anchor array finite and positive.

This strategy costs nothing per image pair beyond one subtraction and one
`norm`: the anchor is a per-feature O(1) lookup against the track index, never a
scan over the reconstruction's points. That is what keeps anchoring cheap enough
to do per feature rather than once per pair from a scene-wide statistic.

## Out of Scope

Visualization only (`sfm epipolar`). The polar-sweep and rectified-sweep
matchers (`sfmtool-core/src/features/feature_match/{polar,sweep}.rs`,
`sfmtool-core/src/camera/rectification.rs`) carry the same pinhole assumption; making
*matching* fisheye-aware (sweeping along the bearing-space epipolar line) is
separate, larger work.

## Parameters

| Name | Default | Notes |
|------|---------|-------|
| `anchor_depth` | observed positive track depth when available; otherwise baseline length, with `1.0` substituted for a baseline below `1e-9` | Seed depth for Phase-1 bracketing; Rust uses `max(abs(value), MIN_ANCHOR)` |
| `curvature_tolerance` | `0.5` (pixels) | Max chord-to-midpoint deviation before a segment is further split |
| `max_vertices` | `256` | Hard cap per polyline — stops runaway subdivision |

Tuning knobs that are *not* exposed because their good values are independent
of the camera and scene:

| Constant | Value | Role |
|----------|-------|------|
| `MIN_BASELINE` | `1e-9` | Baselines below this length return an empty curve |
| `MIN_ANCHOR` | `1e-12` | Floor applied after taking `abs(anchor_depth)`, before `ln` |
| `LOG_STEP` | `ln(2)` | One octave per bracketing step |
| `BRACKET_MAX_STEPS` | `24` | ±24 octaves of seed search before giving up |
| `BRACKET_LOG_TOL` | `1e-3` | Bisection tolerance in log-depth |

`curvature_tolerance` could later be surfaced as a CLI option
(`--curve-tolerance`); start with the fixed default. `max_vertices` is a safety
cap, not a quality knob.
